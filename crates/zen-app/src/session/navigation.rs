// REQ-APP-08 (pi-334 session selection + continuation of T-AG-04-2 branch-summary orchestration) — tree navigation orchestration.
// Corresponds to the original TS: reference/pi-original/packages/coding-agent/src/core/agent-session.ts
// `navigateTree` / `getUserMessagesForForking` (session-side orchestration for /tree//fork in interactive mode).
// Implementation notes: see tasks/T-AP-08-5-implemented.md.
//
// Structural alignment: follows the compaction.rs precedent (the `impl AgentSession` orchestration block under `session/`,
// Accesses `self.agent` / `self.session_store` / `self.settings` via pub(crate) fields).

use crate::config::settings::SettingsManager;
use crate::session::agent_session::{AgentSession, AgentSessionError};
use tokio_util::sync::CancellationToken;
use zen_ai::api::openai_stream::StreamConfig;
use zen_ai::types::Message;
use zen_ai::utils::retry::RetryPolicy;
use zen_ai::utils::text::content_text_from_list;
use zen_core::compaction::branch::{
    collect_entries_for_branch_summary, generate_branch_summary, BranchSummaryResult,
};
use zen_core::compaction::CompactionError;
use zen_core::session::context::build_session_context;
use zen_core::session::tree::{SessionTree, SessionTreeEntry};

/// All intent inputs for one tree navigation (corresponding to the options of pi `navigateTree(targetId, options)`),
/// Consumed once, with no serialization constraints. Constructed by the interactive main task after a tree node is selected.
pub struct NavigateTreeOptions {
    /// Whether to generate a branch summary for the abandoned segment (pi `options.summarize`).
    pub summarize: bool,
    /// Additional summary instructions (the custom_instructions parameter of `generate_branch_summary`; None = default instructions).
    pub custom_instructions: Option<String>,
    /// When true and `custom_instructions` is present, replaces the default instructions entirely (pi `options.replaceInstructions`).
    pub replace_instructions: bool,
    /// Labels attached to the summary entry/target entry (assigned after tree navigation, matching pi `options.label`).
    pub label: Option<String>,
}

impl NavigateTreeOptions {
    /// Default options (no summary, no extra instructions, no labels) — the default intent for navigating without modifying the discarded segment.
    pub fn none() -> Self {
        NavigateTreeOptions {
            summarize: false,
            custom_instructions: None,
            replace_instructions: false,
            label: None,
        }
    }
}

/// Complete output of tree navigation orchestration (matching the pi `navigateTree` return value + the summary payload required for zen append-style UI deviation),
/// Used by the interactive layer to determine UI actions (backfill the editor / append a summary row / show a prompt / redisplay the tree).
#[derive(Debug)]
pub struct NavigateTreeResult {
    /// Body text of the selected user/custom entry (extracted by `content_text_from_list`, populated in the editor);
    /// None for non-message entries.
    pub editor_text: Option<String>,
    /// User cancellation (including no-op when target==leaf).
    pub cancelled: bool,
    /// Summary generation was aborted (Esc); the caller redisplays the tree picker (matching pi aborted).
    pub aborted: bool,
    /// Full text of the generated summary (`BranchSummaryResult.summary`, including PREAMBLE+file-operation addendum);
    /// None = not summarized (user chose not to summarize / no summarizable messages, short-circuit / aborted).
    /// Required payload for the append-style UI deviation (same pattern as T-AP-04-3) — pi fully rebuilds the chat after navigation,
    /// summaryEntry has no consumer; zen uses append rendering, and the orchestrator supplies the summary row text.
    pub summary: Option<String>,
}

impl AgentSession {
    /// Tree navigation orchestration (shared by the /tree command and double-press Esc): move the leaf from its current position to target—
    /// "When viewing the conversation tree, the user clicks an earlier fork point so the conversation continues from there".
    ///
    /// # Behavior
    /// 1. Early return for `target == old_leaf` (cancelled=false, no operation);
    /// 2. Validate that the target exists (if missing → `AgentSessionError::EntryNotFound`, pi "Entry not found");
    /// 3. Determine the new leaf position and editor text based on target type (user/custom message → parent position + body text,
    ///    others → the target itself) — matching pi's newLeafId derivation;
    /// 4. `collect_entries_for_branch_summary` (discarded segment from old leaf → common ancestor);
    /// 5. Dispatch `generate_branch_summary` when `options.summarize` is set and the discarded segment is non-empty
    ///    (`build_retry_policy` assembly + cancel pass-through + `effective().branch_summary.reserve_tokens`
    ///    budget; if there are no messages to summarize, short-circuit without calling the LLM);
    /// 6. Summary succeeds → `append_branch_summary` (direct the leaf to the new leaf position + attach the summary entry);
    ///    Otherwise, use `branch`/`reset_leaf` for the new leaf position; append `options.label` to the summary entry or target entry;
    /// 7. Rebuild the agent context (`branch_entries` → `build_session_context` → `set_messages`).
    ///
    /// # Returns
    /// `NavigateTreeResult`: `summary` carries the full generated summary (source text for append-style UI rendering);
    /// Summary aborted → `Ok(result{aborted:true})` (not treated as an error; the caller redisplays the tree);
    /// Summary failed → `Err(AgentSessionError::Compaction(SummarizationFailed))`.
    ///
    /// # Decision
    /// - The pi `summarize && !model` error branch is structurally unreachable in zen (model is always present and resolved during assembly), so it is not ported;
    /// - Tree snapshot aspect (decision on open question 2): `collect_entries_for_branch_summary` requires `&SessionTree`,
    ///   The orchestration side rebuilds the SessionTree snapshot using `session_store.entries()` + `get_leaf_id()` (avoid adding the trait
    ///   `tree()` method; reconstruction semantics are clear: entries in full + the current leaf).
    ///
    /// # Consumers
    /// interactive main task (`handle_tree_command`).
    pub async fn navigate_tree(
        &self,
        target_id: &str,
        options: &NavigateTreeOptions,
        cancel: &CancellationToken,
    ) -> Result<NavigateTreeResult, AgentSessionError> {
        let old_leaf_id = self.session_store.get_leaf_id();

        // 1. Target == current leaf: no-op early return (same as pi)
        if Some(target_id.to_string()) == old_leaf_id {
            return Ok(NavigateTreeResult {
                editor_text: None,
                cancelled: false,
                aborted: false,
                summary: None,
            });
        }

        // 2. Target existence validation
        let entries = self.session_store.entries();
        let Some(target) = entries.iter().find(|e| e.id() == target_id).cloned() else {
            return Err(AgentSessionError::EntryNotFound(target_id.to_string()));
        };

        // 3. New leaf position + editor text (aligned with pi: user/custom messages → parent position; others → the target itself)
        let (new_leaf_id, editor_text) = match &target {
            SessionTreeEntry::Message {
                message, parent_id, ..
            } if matches!(message, Message::User(_)) => {
                let text = match message {
                    Message::User(u) => content_text_from_list(&u.content, ""),
                    _ => String::new(),
                };
                (parent_id.clone(), Some(text))
            }
            SessionTreeEntry::CustomMessage {
                content, parent_id, ..
            } => (parent_id.clone(), Some(content_text_from_list(content, ""))),
            _ => (Some(target_id.to_string()), None),
        };

        // 4. Collect the discarded segment (rebuild the tree snapshot; see the doc decision)
        let tree = SessionTree::new(entries, old_leaf_id.clone());
        let collected =
            collect_entries_for_branch_summary(&tree, old_leaf_id.as_deref(), target_id);

        // 5. Summary generation (optional)
        let mut summary_result: Option<BranchSummaryResult> = None;
        if options.summarize && !collected.entries.is_empty() {
            let model = self.agent.model();
            let reserve_tokens = self.settings.effective().branch_summary.reserve_tokens;
            let result = generate_branch_summary(
                &collected.entries,
                &self.agent.stream_fn(),
                &model,
                StreamConfig::default(),
                options.custom_instructions.as_deref(),
                options.replace_instructions,
                reserve_tokens,
                Some(cancel.clone()),
                build_retry_policy(&self.settings),
            )
            .await;
            match result {
                Ok(r) => summary_result = Some(r),
                // Abort → return aborted (not treated as an error; the caller redisplays the tree selector)
                Err(CompactionError::Aborted) => {
                    return Ok(NavigateTreeResult {
                        editor_text: None,
                        cancelled: true,
                        aborted: true,
                        summary: None,
                    });
                }
                Err(CompactionError::SummarizationFailed(msg)) => {
                    return Err(AgentSessionError::Compaction(
                        CompactionError::SummarizationFailed(msg),
                    ));
                }
            }
        }

        // 6. Write tree state: summary → append_branch_summary (point leaf to the new leaf position); otherwise branch/reset_leaf
        let label = options.label.clone();
        if let Some(result) = &summary_result {
            let summary_id =
                self.session_store
                    .append_branch_summary(result, new_leaf_id.as_deref(), false)?;
            if let Some(l) = &label {
                let _ = self.session_store.append_label_change(&summary_id, Some(l));
            }
        } else if let Some(id) = &new_leaf_id {
            self.session_store.branch(id)?;
            if let Some(l) = &label {
                let _ = self.session_store.append_label_change(target_id, Some(l));
            }
        } else {
            self.session_store.reset_leaf();
            if let Some(l) = &label {
                let _ = self.session_store.append_label_change(target_id, Some(l));
            }
        }

        // 7. Rebuild agent context (align with pi `agent.state.messages = buildSessionContext().messages`)
        let branch = self.session_store.branch_entries();
        let ctx = build_session_context(&branch);
        self.agent.set_messages(ctx.messages);

        Ok(NavigateTreeResult {
            editor_text,
            cancelled: false,
            aborted: false,
            summary: summary_result.map(|r| r.summary),
        })
    }

    /// /fork selector data source: lists all user messages that can serve as fork starting points—
    /// "The user wants to open a new branch based on some historical user message; first look at what options exist."
    /// Returns `(entry_id, text)` pairs (in the original entries order; entries with empty text are skipped),
    /// Mapped by the interactive layer to `UserMessageListItem` (the session layer does not depend on the components layer).
    /// Aligns with pi `getUserMessagesForForking`: only message + role==user + non-empty text.
    /// # Consumers
    /// Main interactive task (`handle_fork_command`).
    pub fn user_messages_for_forking(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for entry in self.session_store.entries() {
            if let SessionTreeEntry::Message {
                id,
                message: Message::User(u),
                ..
            } = &entry
            {
                let text = content_text_from_list(&u.content, "");
                if !text.is_empty() {
                    out.push((id.clone(), text));
                }
            }
        }
        out
    }
}

/// Shared retry policy setup (extracted from the private compaction.rs version): maps the settings.retry section to
/// zen-ai RetryPolicy—both compaction and branch summary LLM calls must retry with backoff according to the user configuration.
/// Decision (closes the T-AP-02-4/T-AP-04-3 gaps): `max_retry_delay_ms`/`timeout_ms` have no corresponding fields,
/// v1 discarded (the three-field zen-ai RetryPolicy interface has no maximum-delay field; the backoff cap is set by `retry_assistant_call`
/// internal policy). Returns None when `retry.enabled` is false.
/// # Consumers
/// This module (generate_branch_summary wiring in navigate_tree) + compaction.rs (delegation).
pub(crate) fn build_retry_policy(settings: &SettingsManager) -> Option<RetryPolicy> {
    let retry = settings.effective().retry;
    if !retry.enabled {
        return None;
    }
    Some(RetryPolicy {
        enabled: true,
        max_retries: retry.max_retries,
        base_delay_ms: retry.base_delay_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::agent_session::AgentSession;
    use crate::session::jsonl::CreateSessionOptions;
    use crate::session::session_manager::SessionManager;
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex};
    use zen_ai::api::openai_stream::StreamError;
    use zen_ai::tool::ToolExecutionMode;
    use zen_ai::types::{
        AssistantMessage, AssistantMessageEvent, ContentBlock, ContentBlockList, Model,
        NonErrorStopReason, StopReason, TextContent, ThinkingLevel, Usage, UserMessage,
    };
    use zen_core::harness::agent_hooks::AgentLoopHooks;
    use zen_core::harness::engine::Agent;
    use zen_core::harness::types::AgentEvent;
    use zen_core::session::tree::SessionTreeEntry;

    // ── Test helpers ────────────────────────────────────────────────────

    fn test_dir(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("zen_navigation_{}_{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn user_msg(text: &str, ts: i64) -> Message {
        Message::User(UserMessage {
            content: ContentBlockList::Text(text.to_string()),
            timestamp: ts,
        })
    }

    struct MockHooks;

    #[async_trait::async_trait]
    impl AgentLoopHooks for MockHooks {
        async fn on_event(&self, _event: &AgentEvent) {}
    }

    fn test_model() -> Model {
        Model {
            id: "deepseek-v4-flash".into(),
            name: None,
            provider: "opencode-go".into(),
            api: zen_ai::provider::Api::OpenAiCompletions,
            base_url: "https://opencode.ai/zen/go/v1".into(),
            context_window: 65536,
            max_tokens: 8192,
            reasoning: true,
            headers: None,
            cost_rates: None,
        }
    }

    /// Fake StreamFn that records call arguments (Start + TextDelta + Done replay, following the branch.rs precedent).
    fn fake_stream_fn(text: &'static str, calls: Arc<Mutex<usize>>) -> zen_ai::types::StreamFn {
        Arc::new(
            move |_model: &Model,
                  _ctx: &zen_ai::types::Context,
                  _cfg: Option<StreamConfig>,
                  _cancel: Option<CancellationToken>| {
                *calls.lock().unwrap() += 1;
                let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<AssistantMessageEvent>();
                let partial = AssistantMessage {
                    content: vec![ContentBlock::Text(TextContent {
                        text: text.into(),
                        text_signature: None,
                    })],
                    api: "openai-completions".into(),
                    provider: "deepseek".into(),
                    model: "m".into(),
                    response_model: None,
                    response_id: None,
                    usage: Usage::default(),
                    stop_reason: StopReason::Stop,
                    error_message: None,
                    timestamp: 0,
                };
                let _ = tx.send(AssistantMessageEvent::Start {
                    partial: partial.clone(),
                });
                let _ = tx.send(AssistantMessageEvent::TextDelta {
                    content_index: 0,
                    delta: text.into(),
                    partial: partial.clone(),
                });
                let _ = tx.send(AssistantMessageEvent::Done {
                    reason: NonErrorStopReason::Stop,
                    message: partial,
                });
                drop(tx);
                Box::pin(async move { Ok::<_, StreamError>(rx) })
            },
        )
    }

    fn agent_state(model: Model) -> zen_core::harness::types::AgentState {
        zen_core::harness::types::AgentState {
            system_prompt: "test".into(),
            model,
            thinking_level: ThinkingLevel::Off,
            tools: vec![],
            messages: vec![],
            is_run: false,
            streaming_message: None,
            pending_tool_calls: HashSet::new(),
            error_message: None,
        }
    }

    /// Constructs an AgentSession backed by a real SessionManager (in_memory avoids disk persistence; focuses on tree-navigation logic).
    fn session_with_manager(root: &std::path::Path) -> (AgentSession, Arc<SessionManager>) {
        let sm = Arc::new(
            SessionManager::create(
                "/proj",
                root,
                CreateSessionOptions {
                    session_id: None,
                    parent_session_path: None,
                    metadata: None,
                },
            )
            .unwrap(),
        );
        // Seed conversation: u1 ← u2 ← u3 (main chain), with u4 attached under u2 (target branch)
        let m1 = sm.append_message(&user_msg("u1", 1_000));
        let m2 = sm.append_message(&user_msg("u2", 2_000));
        let m3 = sm.append_message(&user_msg("u3", 3_000));
        let _ = (m1, m2, m3);
        let session = AgentSession::new(
            Agent::new(
                agent_state(test_model()),
                ToolExecutionMode::Sequential,
                Arc::new(MockHooks),
                fake_stream_fn("summary text", Arc::new(Mutex::new(0))),
                None, // T-AI-02-6: test does not carry session affinity key
            ),
            sm.clone(),
        );
        (session, sm)
    }

    #[tokio::test]
    async fn navigate_tree_target_is_leaf_returns_cancelled_noop() {
        let (session, sm) = session_with_manager(&test_dir("noop"));
        let leaf = sm.get_leaf_id().unwrap();
        let result = session
            .navigate_tree(
                &leaf,
                &NavigateTreeOptions::none(),
                &CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(!result.cancelled, "target==leaf is not cancel, but no-op");
        assert!(result.summary.is_none());
        assert_eq!(
            sm.get_leaf_id().as_deref(),
            Some(leaf.as_str()),
            "leaf unchanged"
        );
        let _ = std::fs::remove_dir_all(test_dir("noop"));
    }

    #[tokio::test]
    async fn navigate_tree_missing_target_errors() {
        let (session, _sm) = session_with_manager(&test_dir("missing"));
        let err = session
            .navigate_tree(
                "nope",
                &NavigateTreeOptions::none(),
                &CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, AgentSessionError::EntryNotFound(_)));
        let _ = std::fs::remove_dir_all(test_dir("missing"));
    }

    #[tokio::test]
    async fn navigate_tree_user_target_moves_leaf_to_parent_and_returns_editor_text() {
        // Select u1 (root user message): new leaf position = u1.parent = None (reset_leaf), editor_text = "u1"
        let (session, sm) = session_with_manager(&test_dir("nav"));
        let u1_id = sm.get_entries()[0].id().to_string();
        let result = session
            .navigate_tree(
                &u1_id,
                &NavigateTreeOptions::none(),
                &CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(!result.cancelled);
        assert_eq!(result.editor_text.as_deref(), Some("u1"));
        assert!(
            sm.get_leaf_id().is_none(),
            "user root target → new leaf position = parent(None)"
        );
        // The next append becomes the root
        let m5 = sm.append_message(&user_msg("after", 9_000));
        assert_eq!(sm.get_entry(&m5).unwrap().parent_id(), None);
        let _ = std::fs::remove_dir_all(test_dir("nav"));
    }

    #[tokio::test]
    async fn navigate_tree_no_summary_branches_to_target() {
        // Select u2 (not a leaf, not current): new leaf position = u2 itself (the rule for non-user targets does not apply—u2 is a user!
        // New leaf position = u2.parent = u1). Assert that leaf moves to u1 and the LLM is not called.
        let (session, sm) = session_with_manager(&test_dir("branch"));
        let u2_id = sm.get_entries()[1].id().to_string();
        let result = session
            .navigate_tree(
                &u2_id,
                &NavigateTreeOptions::none(),
                &CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(result.summary.is_none());
        let expect = sm
            .get_entry(&u2_id)
            .unwrap()
            .parent_id()
            .unwrap()
            .to_string();
        assert_eq!(sm.get_leaf_id().as_deref(), Some(expect.as_str()));
        let _ = std::fs::remove_dir_all(test_dir("branch"));
    }

    #[tokio::test]
    async fn navigate_tree_with_summary_appends_and_returns_summary() {
        // Select u2 and summarize: u2 is user → new leaf position = u1; discarded segment = [u3] (after u2); summary is attached under u1
        let (session, sm) = session_with_manager(&test_dir("summary"));
        let u2_id = sm.get_entries()[1].id().to_string();
        let result = session
            .navigate_tree(
                &u2_id,
                &NavigateTreeOptions {
                    summarize: true,
                    custom_instructions: None,
                    replace_instructions: false,
                    label: None,
                },
                &CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(!result.aborted);
        let summary = result.summary.expect("Should have summary payload");
        assert!(summary.contains("summary text"), "Actual: {summary}");
        // Tree view: the BranchSummary entry is attached under u1 (new leaf position), and leaf points to the summary entry
        let entries = sm.get_entries();
        let last = entries.last().unwrap();
        assert!(matches!(last, SessionTreeEntry::BranchSummary { .. }));
        assert_eq!(
            last.parent_id().map(|s| s.to_string()),
            sm.get_entry(&u2_id)
                .unwrap()
                .parent_id()
                .map(|s| s.to_string()),
            "Summary entry parent = new leaf position u1"
        );
        let leaf = sm.get_leaf_id().unwrap();
        assert_eq!(leaf, last.id());
        // Context rebuild: the summary message enters agent messages (u1 + summary + ...)
        assert!(!session.agent.messages().is_empty());
        let _ = std::fs::remove_dir_all(test_dir("summary"));
    }

    #[tokio::test]
    async fn user_messages_for_forking_returns_only_nonempty_user_texts() {
        let (session, sm) = session_with_manager(&test_dir("forklist"));
        // Append an empty-text user message (skipped)
        sm.append_message(&user_msg("", 9_000));
        let out = session.user_messages_for_forking();
        assert_eq!(out.len(), 3, "u1/u2/u3; empty text skipped");
        assert_eq!(out[0].1, "u1");
        let _ = std::fs::remove_dir_all(test_dir("forklist"));
    }
}
