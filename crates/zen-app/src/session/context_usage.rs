// REQ-APP-06 / REQ-APP-11 (pi-301 `getContextUsage`): context occupancy derived from the session tree.
// Corresponds to original TS: reference/pi-original/packages/coding-agent/src/core/agent-session.ts:3154.
// Consumers: `modes::interactive::footer_data` — the footer's `xx.x%/window` reading.
// See tasks/context-usage-resume-fix-implemented.md.

use zen_ai::types::{Message, StopReason};
use zen_ai::utils::estimate::{calculate_context_tokens, estimate_messages};
use zen_core::session::context::{
    build_context_entries, latest_compaction_index, session_entry_to_context_messages,
};
use zen_core::session::tree::SessionTreeEntry;

use crate::session::agent_session::AgentSession;

/// Context occupancy of the current session against the current model's window — the number the footer shows as
/// `xx.x%/200k`. `None` = unknown (right after a compaction, until the next LLM response reports fresh usage).
///
/// Snapshot only: derived on demand, never serialized, never cached across turns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContextUsage {
    /// Estimated context tokens, or `None` when the latest assistant usage predates the newest compaction
    /// (pre-compaction usage no longer describes the current context).
    pub tokens: Option<usize>,
    /// Context window of the model currently in use (always > 0 — a zero window yields `None` from the caller).
    pub context_window: u64,
    /// `tokens / context_window * 100`; `None` whenever `tokens` is `None`.
    pub percent: Option<f64>,
}

/// Derives the footer's context ratio for `session` (pi `AgentSession.getContextUsage`).
///
/// Behavior: `None` when `context_window == 0` (no meaningful ratio); `Some { tokens: None, percent: None }` when
/// the branch's newest `Compaction` is not followed by an assistant with a non-aborted/non-error usage that yields
/// `calculate_context_tokens > 0`; otherwise `Some` with
/// `estimate_messages(flatten(build_context_entries(session.session_store.branch_entries())))`.
///
/// **Data source**: the store, not `session.agent.messages()` — zen's `state.messages` is only written back after
/// the whole run finishes (`engine.rs:305`), so a state-based estimate would freeze for the duration of a turn while
/// the usage totals (also store-based) keep climbing. The persistence listener appends before the event bridge, so
/// the store is at least as fresh as the last emitted `MessageEnd`. The flattening pipeline is the one
/// `restore_history_payload` uses, and it deliberately includes `!!` bash / steer messages, matching pi's
/// `this.messages` (which also carries `BashExecutionMessage`).
///
/// Consumers: `modes::interactive::footer_data` (every footer invalidation: `MessageEnd`, `push_footer`,
/// `CompactionEnd`).
pub fn get_context_usage(session: &AgentSession) -> Option<ContextUsage> {
    let context_window = session.model().context_window;
    if context_window == 0 {
        return None;
    }

    let branch = session.session_store.branch_entries();

    // pi: after the newest compaction the previous usage no longer describes the current context, so the token count
    // stays unknown until an assistant responds again.
    if let Some(compaction_index) = latest_compaction_index(&branch) {
        if !has_post_compaction_usage(&branch, compaction_index) {
            return Some(ContextUsage {
                tokens: None,
                context_window,
                percent: None,
            });
        }
    }

    // Same flattening as `restore_history_payload`: the context the next request would carry.
    let visible = build_context_entries(&branch);
    let mut messages = Vec::new();
    for (index, entry) in visible.iter().enumerate() {
        messages.extend(session_entry_to_context_messages(entry, index, &visible));
    }
    let tokens = estimate_messages(&messages).tokens;

    Some(ContextUsage {
        tokens: Some(tokens),
        context_window,
        percent: Some(tokens as f64 / context_window as f64 * 100.0),
    })
}

/// True when the branch carries an assistant after `compaction_index` whose usage still describes the current
/// context (stop reason neither `Aborted` nor `Error`, and `calculate_context_tokens(usage) > 0`).
///
/// Private helper of [`get_context_usage`]; the compaction boundary itself comes from
/// `zen_core::session::context::latest_compaction_index`.
fn has_post_compaction_usage(branch: &[SessionTreeEntry], compaction_index: usize) -> bool {
    branch[compaction_index + 1..].iter().any(|entry| {
        matches!(
            entry,
            SessionTreeEntry::Message {
                message: Message::Assistant(assistant),
                ..
            } if assistant.stop_reason != StopReason::Aborted
                && assistant.stop_reason != StopReason::Error
                && calculate_context_tokens(&assistant.usage) > 0
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::agent_session::SessionStore;
    use crate::session::session_manager::SessionManagerError;
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use zen_ai::tool::ToolExecutionMode;
    use zen_ai::types::{
        AssistantMessage, ContentBlock, ContentBlockList, Cost, StreamFn, TextContent, Usage,
        UserMessage,
    };
    use zen_ai::types::{Model, ThinkingLevel};
    use zen_core::harness::agent_hooks::AgentLoopHooks;
    use zen_core::harness::engine::Agent;
    use zen_core::harness::types::{AgentEvent, AgentState};

    // ── Fixtures ─────────────────────────────────────────────────

    /// Session store serving a fixed entry list for both views (the footer only needs `branch_entries`).
    #[derive(Default)]
    struct FixedStore {
        entries: Mutex<Vec<SessionTreeEntry>>,
    }

    impl FixedStore {
        fn new(entries: Vec<SessionTreeEntry>) -> Self {
            Self {
                entries: Mutex::new(entries),
            }
        }
    }

    impl SessionStore for FixedStore {
        fn append_message(&self, _message: &Message) {}
        fn append_model_change(&self, _provider: &str, _model_id: &str) {}
        fn append_thinking_level_change(&self, _level: &str) {}
        fn session_id(&self) -> String {
            "test-session".to_string()
        }
        fn cwd(&self) -> &str {
            "/work"
        }
        fn branch_entries(&self) -> Vec<SessionTreeEntry> {
            self.entries.lock().unwrap().clone()
        }
        fn append_compaction(
            &self,
            _result: &zen_core::compaction::CompactionResult,
            _from_hook: bool,
        ) -> String {
            "c-mock".to_string()
        }
        fn entries(&self) -> Vec<SessionTreeEntry> {
            self.branch_entries()
        }
        fn get_leaf_id(&self) -> Option<String> {
            self.entries
                .lock()
                .unwrap()
                .last()
                .map(|entry| entry.id().to_string())
        }
        fn append_branch_summary(
            &self,
            _result: &zen_core::compaction::branch::BranchSummaryResult,
            _new_leaf_id: Option<&str>,
            _from_hook: bool,
        ) -> Result<String, SessionManagerError> {
            Ok("b-mock".to_string())
        }
        fn branch(&self, _branch_from_id: &str) -> Result<(), SessionManagerError> {
            Ok(())
        }
        fn reset_leaf(&self) {}
        fn append_label_change(
            &self,
            _target_id: &str,
            _label: Option<&str>,
        ) -> Result<String, SessionManagerError> {
            Ok("l-mock".to_string())
        }
        fn append_session_info(&self, _name: Option<&str>) -> String {
            "s-mock".to_string()
        }
        fn create_branched_session(
            &self,
            _leaf_id: &str,
        ) -> Result<Option<PathBuf>, SessionManagerError> {
            Ok(None)
        }
    }

    struct NoopHooks;

    #[async_trait::async_trait]
    impl AgentLoopHooks for NoopHooks {
        async fn on_event(&self, _event: &AgentEvent) {}
    }

    fn noop_stream_fn() -> StreamFn {
        Arc::new(
            |_model: &Model,
             _ctx: &zen_ai::types::Context,
             _cfg: Option<zen_ai::api::openai_stream::StreamConfig>,
             _cancel: Option<tokio_util::sync::CancellationToken>| {
                Box::pin(async move {
                    let (_tx, rx) = tokio::sync::mpsc::unbounded_channel::<
                        zen_ai::types::AssistantMessageEvent,
                    >();
                    Ok::<_, zen_ai::api::openai_stream::StreamError>(rx)
                })
            },
        )
    }

    fn session_with(context_window: u64, entries: Vec<SessionTreeEntry>) -> AgentSession {
        let state = AgentState {
            system_prompt: "test".into(),
            model: Model {
                id: "test-model".into(),
                name: None,
                provider: "test".into(),
                api: zen_ai::provider::Api::OpenAiCompletions,
                base_url: "http://localhost".into(),
                context_window,
                max_tokens: 1024,
                reasoning: false,
                headers: None,
                cost_rates: None,
                compat: None,
            },
            thinking_level: ThinkingLevel::Off,
            tools: vec![],
            messages: vec![],
            is_run: false,
            streaming_message: None,
            pending_tool_calls: HashSet::new(),
            error_message: None,
        };
        let agent = Agent::new(
            state,
            ToolExecutionMode::Sequential,
            Arc::new(NoopHooks),
            noop_stream_fn(),
            None,
        );
        AgentSession::new(agent, Arc::new(FixedStore::new(entries)))
    }

    fn usage(input: u64, cache_read: u64, total: u64) -> Usage {
        Usage {
            input,
            output: 0,
            cache_read,
            cache_write: 0,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: total,
            cost: Cost::default(),
        }
    }

    fn assistant_entry(
        id: &str,
        parent: Option<&str>,
        stop_reason: StopReason,
        usage: Usage,
    ) -> SessionTreeEntry {
        SessionTreeEntry::Message {
            id: id.into(),
            parent_id: parent.map(str::to_string),
            timestamp: "2024-01-01T00:00:00Z".into(),
            message: Message::Assistant(AssistantMessage {
                content: vec![ContentBlock::Text(TextContent {
                    text: "hello".into(),
                    text_signature: None,
                })],
                api: "openai".into(),
                provider: "test".into(),
                model: "test-model".into(),
                response_model: None,
                response_id: None,
                usage,
                stop_reason,
                error_message: None,
                // After the compaction entry below (2024-01-02): `estimate_messages` gates usage by timestamp.
                timestamp: 1_705_000_000_000,
            }),
        }
    }

    fn user_entry(id: &str, parent: Option<&str>) -> SessionTreeEntry {
        SessionTreeEntry::Message {
            id: id.into(),
            parent_id: parent.map(str::to_string),
            timestamp: "2024-01-01T00:00:01Z".into(),
            message: Message::User(UserMessage {
                content: ContentBlockList::Text("hi".into()),
                timestamp: 1_705_000_000_000,
            }),
        }
    }

    fn compaction_entry(id: &str, parent: Option<&str>) -> SessionTreeEntry {
        SessionTreeEntry::Compaction {
            id: id.into(),
            parent_id: parent.map(str::to_string),
            timestamp: "2024-01-02T00:00:00Z".into(),
            summary: "compacted".into(),
            first_kept_entry_id: None,
            tokens_before: 500,
            retained_tail: None,
            details: None,
            usage: None,
            from_hook: None,
        }
    }

    // ── get_context_usage ────────────────────────────────────────

    #[test]
    fn context_usage_none_without_a_context_window() {
        let session = session_with(0, vec![]);
        assert_eq!(get_context_usage(&session), None);
    }

    #[test]
    fn context_usage_uses_last_assistant_usage_plus_trailing() {
        // usage total 1000 of a 2000 window → 50%; the trailing user message only adds an estimate.
        let session = session_with(
            2000,
            vec![
                user_entry("u1", None),
                assistant_entry("a1", Some("u1"), StopReason::Stop, usage(100, 900, 1000)),
                user_entry("u2", Some("a1")),
            ],
        );
        let context = get_context_usage(&session).expect("window > 0");
        assert_eq!(context.context_window, 2000);
        assert!(context.tokens.unwrap() >= 1000);
        assert!(context.percent.unwrap() >= 50.0);
    }

    #[test]
    fn context_usage_empty_session_is_zero() {
        let session = session_with(2000, vec![]);
        let context = get_context_usage(&session).expect("window > 0");
        assert_eq!(context.tokens, Some(0));
        assert_eq!(context.percent, Some(0.0));
    }

    #[test]
    fn context_usage_unknown_after_compaction_without_new_assistant() {
        let session = session_with(
            2000,
            vec![
                user_entry("u1", None),
                assistant_entry("a1", Some("u1"), StopReason::Stop, usage(100, 900, 1000)),
                compaction_entry("c1", Some("a1")),
            ],
        );
        let context = get_context_usage(&session).expect("window > 0");
        assert_eq!(context.tokens, None);
        assert_eq!(context.percent, None);
    }

    #[test]
    fn context_usage_known_when_an_assistant_responds_after_compaction() {
        let session = session_with(
            2000,
            vec![
                user_entry("u1", None),
                compaction_entry("c1", Some("u1")),
                user_entry("u2", Some("c1")),
                assistant_entry("a2", Some("u2"), StopReason::Stop, usage(500, 0, 500)),
            ],
        );
        let context = get_context_usage(&session).expect("window > 0");
        assert_eq!(context.tokens, Some(500));
        assert_eq!(context.percent, Some(25.0));
    }

    #[test]
    fn context_usage_unknown_when_post_compaction_usage_is_unusable() {
        for stop_reason in [StopReason::Aborted, StopReason::Error] {
            let session = session_with(
                2000,
                vec![
                    user_entry("u1", None),
                    compaction_entry("c1", Some("u1")),
                    assistant_entry("a2", Some("u1"), stop_reason, usage(500, 0, 500)),
                ],
            );
            assert_eq!(
                get_context_usage(&session).and_then(|c| c.tokens),
                None,
                "an aborted/error assistant does not report usable context usage"
            );
        }
        // Usage of zero is equally unusable.
        let session = session_with(
            2000,
            vec![
                user_entry("u1", None),
                compaction_entry("c1", Some("u1")),
                assistant_entry("a2", Some("u1"), StopReason::Stop, usage(0, 0, 0)),
            ],
        );
        assert_eq!(get_context_usage(&session).and_then(|c| c.tokens), None);
    }
}
