// REQ-APP-04 (pi-311): compaction scheduling — automatic (threshold/overflow) + manual trigger + threshold config wiring.
// Corresponding original TS: reference/pi-original/packages/coding-agent/src/core/agent-session.ts's
// `_checkCompaction` (:1951) / `_runAutoCompaction` (:2045) / `compact` (:1781) orchestration layer —
// The algorithmic side has already landed in zen-core compaction via T-AG-04-1/T-AG-04-2 (dual-copy merge adjudication); this file is
// Orchestration hookup in zen: trigger timing (pre-submit pre-check + post-run check), event broadcast (via
// Agent::emit single bus), persistence (SessionStore::append_compaction), settings wiring
// (SettingsManager::effective() live reads, closing the T-AP-02-3/T-AP-02-4 gaps).

use std::sync::atomic::Ordering;

use tokio_util::sync::CancellationToken;
use zen_ai::api::openai_stream::StreamConfig;
use zen_ai::types::{AssistantMessage, Message, StopReason};
use zen_ai::utils::estimate::{calculate_context_tokens, estimate_messages};
use zen_ai::utils::overflow::is_context_overflow;
use zen_ai::utils::retry::RetryPolicy;
use zen_core::compaction::{
    compact, prepare_compaction, should_compact, CompactionError, CompactionPreparation,
    CompactionReason, CompactionResult, CompactionSettings,
};
use zen_core::harness::types::AgentEvent;
use zen_core::session::context::build_session_context;
use zen_core::session::tree::SessionTreeEntry;

use super::agent_session::{AgentSession, AgentSessionError};

/// pi's "Context overflow recovery failed after one compact-and-retry attempt..." original text
/// (agent-session.ts:1990, user-visible message when the overflow recovery budget is exhausted).
const OVERFLOW_RECOVERY_FAILED: &str = "Context overflow recovery failed after one compact-and-retry attempt. Try reducing context or switching to a larger-context model.";

/// Take the last assistant message (shared input source for pre-submit pre-check and post-run check; aligning pi
/// `_findLastAssistantMessage`）。
/// consumers: this module (check_compaction/execute flow) + agent_session.rs (prompt wiring).
pub(crate) fn last_assistant_message(messages: &[Message]) -> Option<&AssistantMessage> {
    messages.iter().rev().find_map(|m| match m {
        Message::Assistant(a) => Some(a),
        _ => None,
    })
}

impl AgentSession {
    /// Manual compaction executed when the user types /compact (or an extension calls the SDK equivalent) — "stop the current run first,
    /// then make room": abort current run → emit compaction_start{manual} → prepare → LLM summarize →
    /// rebuild on disk → emit compaction_end (aligning pi `compact`, agent-session.ts:1781).
    ///
    /// # Parameters
    /// - `custom_instructions`: user-supplied instructions for the summarizer (free text following /compact).
    ///
    /// # Returns
    /// `Ok(CompactionResult)` is for callers (tests / future UI feedback); interactive paths should ignore the return value —
    /// errors are already delivered via the `CompactionEnd{error_message}` event (pi semantics: interactive catch then
    /// ignore, "will be emitted as an event").
    ///
    /// # Errors
    /// - [`AgentSessionError::AlreadyCompacted`]: the branch's last entry is already a Compaction entry;
    /// - [`AgentSessionError::NothingToCompact`]: session too small to compact (prepare returns None and
    ///   the last entry is not a Compaction);
    /// - [`AgentSessionError::Compaction`]: summarizer LLM failed/cancelled.
    ///   Error paths also emit `CompactionEnd` first (aborted/error semantics distinction, aligning pi :1914 catch) before returning Err.
    ///
    /// # Bounds
    /// - `&self`: the cancel handle is registered via `compaction_cancel`, so `abort()` can cancel this compaction across threads.
    /// - Aborts the current run right away (`await this.abort()`); zen has no `_disconnectFromAgent`
    ///   concept (no RPC), so it's dropped;
    /// - v1 drops extension events `session_before_compact`/`session_compact` (no extension system, REQ-V2-01
    ///   will add before/after hook slots at implementation time).
    pub async fn compact(
        &self,
        custom_instructions: Option<&str>,
    ) -> Result<CompactionResult, AgentSessionError> {
        // pi compact() first two steps: abort the current run + swap in the new compaction cancellation token
        self.abort();
        let cancel = CancellationToken::new();
        *self
            .compaction_cancel
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(cancel.clone());

        // pi: start is always emitted (before the try; emitted even if prepare fails)
        self.agent.emit(AgentEvent::CompactionStart {
            reason: CompactionReason::Manual,
        });

        let result = self.execute_compaction(custom_instructions, &cancel).await;

        // pi finally: clear the cancellation token
        *self
            .compaction_cancel
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;

        match result {
            Ok(r) => {
                self.agent.emit(AgentEvent::CompactionEnd {
                    reason: CompactionReason::Manual,
                    result: Some(Box::new(r.clone())),
                    aborted: false,
                    will_retry: false,
                    error_message: None,
                });
                Ok(r)
            }
            Err(e) => {
                // pi catch: aborted (cancelled/aborted) emits no error_message; otherwise wrap with "Compaction failed: "
                let aborted = matches!(&e, AgentSessionError::Compaction(CompactionError::Aborted));
                let error_message = if aborted {
                    None
                } else {
                    Some(format!("Compaction failed: {e}"))
                };
                self.agent.emit(AgentEvent::CompactionEnd {
                    reason: CompactionReason::Manual,
                    result: None,
                    aborted,
                    will_retry: false,
                    error_message,
                });
                Err(e)
            }
        }
    }

    /// The compaction sentinel of the session loop — checked once before the prompt is submitted and once
    /// after each agent turn ends; decides "should we compact" and actually compacts when needed.
    /// The five gates follow pi `_checkCompaction` (agent-session.ts:1951) order:
    /// ① `settings.enabled` master switch; ② aborted messages are skipped (the pre-submission check with
    /// `skip_aborted_check=false` can waive this, capturing a compaction need from the previous turn's aborted/error state); ③ overflow check only within the same model
    /// (prevents switching from a small-window model to a large-window model from re-triggering compaction due to the old model's overflow errors);
    /// ④ assistant messages older than the most recent compaction boundary are skipped (prevents old usage from re-triggering right after compaction completes);
    /// ⑤ overflow branch (errors or usage exceeds the window; `stop` means "compact only, no retry", otherwise delete trailing error messages,
    /// compact and retry — `overflow_recovery_attempted` guarantees only one recovery attempt) → threshold branch
    /// (usage computed directly; on errors/all-zeros falls back to message estimation + staleness check on the estimation source).
    ///
    /// # Returns
    /// `Ok(true)` = "the loop should continue for one more round" (overflow recovered / queued messages after compaction); the caller
    /// (`prompt`) uses this to call `continue_turn`; `Ok(false)` = nothing happened or compaction finished with no need to continue.
    ///
    /// # Errors
    /// Passed through from [`Self::run_auto_compaction`]; the caller (prompt) tolerates failure with `unwrap_or(false)` —
    /// The error has already been delivered as a `CompactionEnd{error_message}` event and does not block the normal prompt flow (pi semantics).
    pub(crate) async fn check_compaction(
        &self,
        last_assistant: &AssistantMessage,
        skip_aborted_check: bool,
    ) -> Result<bool, AgentSessionError> {
        // Gate ①: master switch (liveness read on each check; user settings changes take effect without rebuilding the session)
        let settings = self.effective_compaction_settings();
        if !settings.enabled {
            return Ok(false);
        }
        // Gate ②: skip aborted (exemptable via skipAbortedCheck — pre-commit precheck needs the previous round's aborted)
        if skip_aborted_check && last_assistant.stop_reason == StopReason::Aborted {
            return Ok(false);
        }

        let model = self.agent.model();
        let context_window = model.context_window;
        // Gate ③: overflow check only for the same model (double compare on provider and model id, aligned with pi sameModel)
        let same_model =
            last_assistant.provider == model.provider && last_assistant.model == model.id;
        // Gate ④: messages earlier than the latest compaction boundary are old-world relics and no longer trigger
        if self.assistant_precedes_latest_compaction(last_assistant) {
            return Ok(false);
        }

        // Gate ⑤a: overflow branch (pi :1985-2009)
        if same_model && is_context_overflow(last_assistant, Some(context_window as usize)) {
            // stopReason=stop: successful response exceeded the window, the answer is complete and continue cannot extend the assistant — compact only, no retry
            let will_retry = last_assistant.stop_reason != StopReason::Stop;
            if !will_retry {
                return self
                    .run_auto_compaction(CompactionReason::Overflow, false)
                    .await;
            }
            // Recovery budget exhausted: emit a "recovery failed" event, then give up (pi :1988-1999)
            if self.overflow_recovery_attempted.load(Ordering::Relaxed) {
                self.agent.emit(AgentEvent::CompactionEnd {
                    reason: CompactionReason::Overflow,
                    result: None,
                    aborted: false,
                    will_retry: false,
                    error_message: Some(OVERFLOW_RECOVERY_FAILED.to_string()),
                });
                return Ok(false);
            }
            self.overflow_recovery_attempted
                .store(true, Ordering::Relaxed);
            // Drop the trailing erroneous assistant message (pi :2002-2006: already persisted to session history, but not kept in the retry context;
            // after compaction the context is rebuilt wholesale, so popping it here clears the way for estimation/retry)
            let mut messages = self.agent.messages();
            if matches!(messages.last(), Some(Message::Assistant(_))) {
                messages.pop();
                self.agent.set_messages(messages);
            }
            return self
                .run_auto_compaction(CompactionReason::Overflow, will_retry)
                .await;
        }

        // Gate ⑤b: threshold branch (pi :2010-2038)
        // Compute from usage directly; on error or all-zero usage (persistent API failure / malformed zero-value response) fall back to message estimation,
        // ensuring such sessions can still compact without resetting context accounting
        let direct_context_tokens = calculate_context_tokens(&last_assistant.usage);
        let context_tokens =
            if last_assistant.stop_reason == StopReason::Error || direct_context_tokens == 0 {
                let messages = self.agent.messages();
                let estimate = estimate_messages(&messages);
                let Some(usage_index) = estimate.last_usage_index else {
                    return Ok(false); // No usage data at all, nothing to judge from
                };
                // Stale estimation-source check (pi :2022-2031): if the usage source is a pre-compaction old message, its values reflect
                // the old large context and would falsely trigger a just-completed compaction — compare the estimation-source message, not the last one
                if let Some(Message::Assistant(usage_msg)) = messages.get(usage_index) {
                    if self.assistant_precedes_latest_compaction(usage_msg) {
                        return Ok(false);
                    }
                }
                estimate.tokens
            } else {
                direct_context_tokens
            };

        if should_compact(context_tokens, context_window, &settings) {
            return self
                .run_auto_compaction(CompactionReason::Threshold, false)
                .await;
        }
        Ok(false)
    }

    /// Auto-compaction executor (pi `_runAutoCompaction`, agent-session.ts:2045): prepare → emit
    /// `CompactionStart` → LLM summarization (with RetryPolicy assembly and a cancellation handle) → persist and rebuild → emit
    /// `CompactionEnd` → on will_retry, drop the trailing erroneous message; otherwise "if queued messages exist, continue with one delivery round".
    ///
    /// Event semantics align with pi's `started` flag: prepare failure (nothing to compact) **silently** returns false (sends no
    /// event — nothing to compact is not an auto-compaction error); only failures after prepare succeeds emit `CompactionEnd`.
    ///
    /// # Returns
    /// `Ok(true)` = keep looping (will_retry retry / queued messages); `Ok(false)` = no need to continue.
    async fn run_auto_compaction(
        &self,
        reason: CompactionReason,
        will_retry: bool,
    ) -> Result<bool, AgentSessionError> {
        // prepare (before started: failure is silent, aligned with pi :2064-2067 and catch's started gate)
        let settings = self.effective_compaction_settings();
        let path_entries = self.session_store.branch_entries();
        let preparation = match prepare_compaction(&path_entries, &settings) {
            Ok(Some(p)) => p,
            // Nothing to compact / errors are both abandoned silently (pi prepare falsy → return false; prepare has no error path)
            Ok(None) => return Ok(false),
            Err(_) => return Ok(false),
        };

        // started: failures after this point must emit a compaction_end error event (pi :2173-2186 catch)
        let cancel = CancellationToken::new();
        *self
            .compaction_cancel
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(cancel.clone());
        self.agent.emit(AgentEvent::CompactionStart { reason });

        let outcome = self
            .execute_prepared_compaction(preparation, None, &cancel)
            .await;

        *self
            .compaction_cancel
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;

        let result = match outcome {
            Ok(r) => r,
            // Cancellation: emit aborted end (pi :2149-2157)
            Err(AgentSessionError::Compaction(CompactionError::Aborted)) => {
                self.agent.emit(AgentEvent::CompactionEnd {
                    reason,
                    result: None,
                    aborted: true,
                    will_retry: false,
                    error_message: None,
                });
                return Ok(false);
            }
            // Other failures: emit error end (message varies by reason, aligned with pi :2179-2184), continue value false
            Err(e) => {
                let prefix = match reason {
                    CompactionReason::Overflow => "Context overflow recovery failed",
                    _ => "Auto-compaction failed",
                };
                self.agent.emit(AgentEvent::CompactionEnd {
                    reason,
                    result: None,
                    aborted: false,
                    will_retry: false,
                    error_message: Some(format!("{prefix}: {e}")),
                });
                return Ok(false);
            }
        };

        self.agent.emit(AgentEvent::CompactionEnd {
            reason,
            result: Some(Box::new(result.clone())),
            aborted: false,
            will_retry,
            error_message: None,
        });

        if will_retry {
            // pi :2188-2192: delete the trailing error assistant (if any) so the retry starts from a clean context
            let mut messages = self.agent.messages();
            if matches!(
                messages.last(),
                Some(Message::Assistant(a)) if a.stop_reason == StopReason::Error
            ) {
                messages.pop();
                self.agent.set_messages(messages);
            }
            return Ok(true);
        }

        // pi :2194-2195: when auto-compaction finishes, if the steer/follow-up queue has pending messages, run another round to deliver them
        Ok(self.has_queued_messages())
    }

    /// Get effective compaction settings — equivalent of pi's `getCompactionSettings()`, a live read on every check
    /// `settings.effective().compaction`; users changing settings need no session rebuild (closes T-AP-02-3/T-AP-02-4 gap).
    fn effective_compaction_settings(&self) -> CompactionSettings {
        self.settings.effective().compaction
    }

    /// Assemble the settings.retry section into a zen-ai `RetryPolicy` for backoff retries on summary calls — pi passes
    /// `getRetrySettings()` as-is to compact(); zen's two sides have different field surfaces, so explicit mapping is needed.
    /// Ruling (closes T-AP-02-4 "decide at the time"): `max_retry_delay_ms` has no corresponding field, dropped in v1
    /// (zen-ai RetryPolicy's three-field surface has no max-delay field; the backoff cap is decided by `retry_assistant_call`'s
    /// internal policy); pi's timeout_ms passthrough likewise has no corresponding field, dropped as well.
    /// Returns: None when `retry.enabled` is false (semantics of zen-core compact's retry parameter).
    ///
    /// # Consumers
    /// Internal to this task (compact/run_auto_compaction); **T-AP-08-5 must consume the same assembly pattern**
    /// (generate_branch_summary assembly; the plan seam note says "reserved").
    fn build_retry_policy(&self) -> Option<RetryPolicy> {
        let retry = self.settings.effective().retry;
        if !retry.enabled {
            return None;
        }
        Some(RetryPolicy {
            enabled: true,
            max_retries: retry.max_retries,
            base_delay_ms: retry.base_delay_ms,
        })
    }

    /// Stale message check — whether the given assistant message predates the most recent Compaction entry on the branch.
    /// If so, the message's usage/errors are relics of the pre-compaction old world and must not trigger compaction again (guards against "compaction just finished
    /// is re-triggered by old usage").
    /// Two call sites: ① overflow/threshold main gate compares against the last assistant message being checked; ② threshold
    /// fallback branch compares against the estimate source (the message `last_usage_index` points to, possibly an earlier assistant).
    ///
    /// # Bounds
    /// - Always false when there is no Compaction entry;
    /// - Timestamp types differ on the two sides: assistant side is `AssistantMessage.timestamp: i64` (Unix milliseconds),
    ///   entry side is `SessionTreeEntry::Compaction.timestamp: String` (RFC3339, produced by
    ///   `create_timestamp()`); comparison is done after parsing to milliseconds (not lexicographic — offsets vary);
    /// - An entry timestamp that fails to parse is treated as false (prefer not skipping over stalling the compaction path on a bad timestamp).
    fn assistant_precedes_latest_compaction(&self, assistant: &AssistantMessage) -> bool {
        let entries = self.session_store.branch_entries();
        let Some(SessionTreeEntry::Compaction { timestamp, .. }) = entries
            .iter()
            .rev()
            .find(|e| matches!(e, SessionTreeEntry::Compaction { .. }))
        else {
            return false;
        };
        match chrono::DateTime::parse_from_rfc3339(timestamp) {
            Ok(dt) => assistant.timestamp <= dt.timestamp_millis(),
            Err(_) => false,
        }
    }

    /// Full flow of one manual compaction (main body of compact()): prepare (None → already compacted/too-small error) →
    /// LLM summary → cancel recheck → persist and rebuild. Events are the caller's (compact) responsibility.
    async fn execute_compaction(
        &self,
        custom_instructions: Option<&str>,
        cancel: &CancellationToken,
    ) -> Result<CompactionResult, AgentSessionError> {
        let settings = self.effective_compaction_settings();
        let path_entries = self.session_store.branch_entries();
        let preparation = prepare_compaction(&path_entries, &settings)
            .map_err(AgentSessionError::Compaction)?
            .ok_or_else(|| match path_entries.last() {
                // pi :1795-1801: distinguish error message by last message type when prepare is falsy
                Some(SessionTreeEntry::Compaction { .. }) => AgentSessionError::AlreadyCompacted,
                _ => AgentSessionError::NothingToCompact,
            })?;
        self.execute_prepared_compaction(preparation, custom_instructions, cancel)
            .await
    }

    /// Execution of a prepared compaction: LLM summary → cancel recheck → `append_compaction` to persist → rebuild context.
    /// Events are not emitted here (the caller wraps with manual/auto semantics).
    ///
    /// Cancel recheck (race where cancel arrives after the LLM completes): pi checks only on the auto path (:2145); zen rechecks on
    /// the manual path too — "user intent takes priority"; a cancelled compaction is not persisted and is registered as an accepted deviation
    /// (pi's manual path would still append at this point).
    async fn execute_prepared_compaction(
        &self,
        preparation: CompactionPreparation,
        custom_instructions: Option<&str>,
        cancel: &CancellationToken,
    ) -> Result<CompactionResult, AgentSessionError> {
        let model = self.agent.model();
        // pi's retry callbacks `_summarizationRetryCallbacks` (auto_retry UI events) dropped in v1 —
        // zen-ai retry_assistant_call has no callback surface
        let result = compact(
            preparation,
            &self.agent.stream_fn(),
            &model,
            StreamConfig::default(),
            custom_instructions,
            Some(self.agent.thinking_level()),
            Some(cancel.clone()),
            self.build_retry_policy(),
        )
        .await
        .map_err(AgentSessionError::Compaction)?;

        // Cancel race after LLM completes, before persist: don't persist the cancelled compaction
        if cancel.is_cancelled() {
            return Err(AgentSessionError::Compaction(CompactionError::Aborted));
        }

        // Persist + rebuild context (pi :2158-2162: appendCompaction → buildSessionContext →
        // agent.state.messages). from_hook=false: app-layer compaction.
        self.session_store.append_compaction(&result, false);
        let entries = self.session_store.branch_entries();
        let ctx = build_session_context(&entries);
        self.agent.set_messages(ctx.messages);

        Ok(result)
    }

    /// Whether there are queued messages (steer/follow-up pending) — equivalent to pi's `agent.hasQueuedMessages()`.
    fn has_queued_messages(&self) -> bool {
        !self.agent.steering_messages().is_empty() || !self.agent.follow_up_messages().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::agent_session::SessionStore;
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex};
    use zen_ai::api::openai_stream::StreamError;
    use zen_ai::types::{
        AssistantMessageEvent, ContentBlock, ContentBlockList, Model, NonErrorStopReason,
        TextContent, Usage,
    };
    use zen_core::harness::agent_hooks::AgentLoopHooks;
    use zen_core::harness::engine::Agent;
    use zen_core::harness::types::AgentState;

    // ── Test helpers ─────────────────────────────────────────────

    /// Unix milliseconds of 2024-01-01T00:00:00Z
    const T1_MS: i64 = 1_704_067_200_000;
    /// Unix milliseconds of 2024-01-02T00:00:00Z
    const T2_MS: i64 = 1_704_153_600_000;

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

    /// StreamFn that produces a fixed text summary (Start + TextDelta + Done)
    fn summary_stream_fn(text: &'static str) -> zen_ai::types::StreamFn {
        use zen_ai::api::openai_stream::StreamConfig;
        Arc::new(
            move |_model: &Model,
                  _ctx: &zen_ai::types::Context,
                  _cfg: Option<StreamConfig>,
                  _cancel: Option<CancellationToken>| {
                Box::pin(async move {
                    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<AssistantMessageEvent>();
                    let partial = AssistantMessage {
                        content: vec![ContentBlock::Text(TextContent {
                            text: text.to_string(),
                            text_signature: None,
                        })],
                        api: "openai-completions".into(),
                        provider: "test".into(),
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
                    let _ = tx.send(AssistantMessageEvent::Done {
                        reason: NonErrorStopReason::Stop,
                        message: partial,
                    });
                    drop(tx);
                    Ok::<_, StreamError>(rx)
                })
            },
        )
    }

    struct MockHooks;

    #[async_trait::async_trait]
    impl AgentLoopHooks for MockHooks {
        async fn on_event(&self, _event: &AgentEvent) {}
    }

    /// SessionStore test stub with fixed entries + recorded compaction writes
    struct FixedStore {
        entries: Mutex<Vec<SessionTreeEntry>>,
        /// append_compaction records (count + entries written)
        compactions: Mutex<Vec<CompactionResult>>,
        /// Timestamp of the entry produced by append_compaction (RFC3339)
        compaction_ts: &'static str,
    }

    impl FixedStore {
        fn new(entries: Vec<SessionTreeEntry>) -> Self {
            Self {
                entries: Mutex::new(entries),
                compactions: Mutex::new(Vec::new()),
                compaction_ts: "2024-01-02T00:00:00Z",
            }
        }

        fn compaction_count(&self) -> usize {
            self.compactions.lock().unwrap().len()
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
            result: &zen_core::compaction::CompactionResult,
            _from_hook: bool,
        ) -> String {
            self.compactions.lock().unwrap().push(result.clone());
            self.entries
                .lock()
                .unwrap()
                .push(SessionTreeEntry::Compaction {
                    id: format!("c{}", self.compaction_count()),
                    parent_id: None,
                    timestamp: self.compaction_ts.to_string(),
                    summary: result.summary.clone(),
                    first_kept_entry_id: Some(result.first_kept_entry_id.clone()),
                    tokens_before: result.tokens_before,
                    retained_tail: (!result.retained_tail.is_empty())
                        .then(|| result.retained_tail.clone()),
                    details: Some(serde_json::json!({
                        "readFiles": result.details.read_files,
                        "modifiedFiles": result.details.modified_files,
                    })),
                    usage: result.usage.clone(),
                    from_hook: Some(_from_hook),
                });
            format!("c{}", self.compactions.lock().unwrap().len())
        }
    }

    fn user_entry(id: &str, text: &str) -> SessionTreeEntry {
        SessionTreeEntry::Message {
            id: id.to_string(),
            parent_id: None,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            message: Message::User(zen_ai::types::UserMessage {
                content: ContentBlockList::Text(text.to_string()),
                timestamp: T1_MS,
            }),
        }
    }

    fn compaction_entry(id: &str) -> SessionTreeEntry {
        SessionTreeEntry::Compaction {
            id: id.to_string(),
            parent_id: None,
            timestamp: "2024-01-02T00:00:00Z".to_string(),
            summary: "old summary".to_string(),
            first_kept_entry_id: None,
            tokens_before: 100,
            retained_tail: None,
            details: None,
            usage: None,
            from_hook: None,
        }
    }

    /// Overflow assistant message (provider/model aligned with test_model to pass the same-model gate)
    fn overflow_error_assistant(timestamp: i64) -> AssistantMessage {
        AssistantMessage {
            content: vec![],
            api: "openai-completions".into(),
            provider: "opencode-go".into(),
            model: "deepseek-v4-flash".into(),
            response_model: None,
            response_id: None,
            usage: Usage::default(),
            stop_reason: StopReason::Error,
            error_message: Some("prompt is too long: 70000 tokens > 65536 maximum".into()),
            timestamp,
        }
    }

    fn assistant_with_usage(total_tokens: u64, timestamp: i64) -> AssistantMessage {
        AssistantMessage {
            content: vec![],
            api: "openai-completions".into(),
            provider: "opencode-go".into(),
            model: "deepseek-v4-flash".into(),
            response_model: None,
            response_id: None,
            usage: Usage {
                total_tokens,
                ..Usage::default()
            },
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp,
        }
    }

    /// Assemble a test session with event capture (settings can inject patches)
    fn test_session(
        store: Arc<FixedStore>,
        settings_patch: Option<crate::config::settings::SettingsData>,
    ) -> (AgentSession, Arc<Mutex<Vec<AgentEvent>>>) {
        let state = AgentState {
            system_prompt: "test".into(),
            model: test_model(),
            thinking_level: zen_ai::types::ThinkingLevel::Off,
            tools: vec![],
            messages: vec![],
            is_run: false,
            streaming_message: None,
            pending_tool_calls: HashSet::new(),
            error_message: None,
        };
        let settings = Arc::new(crate::config::settings::SettingsManager::empty());
        if let Some(patch) = settings_patch {
            settings.apply_overrides(patch);
        }
        let agent = Agent::new(
            state,
            zen_ai::tool::ToolExecutionMode::Parallel,
            Arc::new(MockHooks),
            summary_stream_fn("compacted summary"),
        );
        let events: Arc<Mutex<Vec<AgentEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&events);
        let _ = agent.subscribe(Box::new(move |e: &AgentEvent| {
            sink.lock().unwrap().push(e.clone());
        }));
        (
            AgentSession::new_with_exclusions(
                agent,
                store,
                Arc::new(Mutex::new(HashSet::new())),
                zen_core::env::ShellConfig::default(),
                settings,
                vec![],
                vec![],
            ),
            events,
        )
    }

    fn settings_with(
        compaction: Option<crate::config::settings::CompactionSettingsPatch>,
        retry: Option<crate::config::settings::RetrySettingsPatch>,
    ) -> crate::config::settings::SettingsData {
        crate::config::settings::SettingsData {
            compaction,
            retry,
            ..Default::default()
        }
    }

    fn compaction_events(events: &[AgentEvent]) -> Vec<&AgentEvent> {
        events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    AgentEvent::CompactionStart { .. } | AgentEvent::CompactionEnd { .. }
                )
            })
            .collect()
    }

    // ── Settings wiring (closing T-AP-02-3/T-AP-02-4 seams) ────────

    #[test]
    fn effective_compaction_settings_defaults_from_empty_settings() {
        let (session, _) = test_session(Arc::new(FixedStore::new(vec![])), None);
        let s = session.effective_compaction_settings();
        assert!(s.enabled);
        assert_eq!(s.reserve_tokens, 16_384);
        assert_eq!(s.keep_recent_tokens, 20_000);
    }

    #[test]
    fn effective_compaction_settings_live_read_applies_patch() {
        let (session, _) = test_session(
            Arc::new(FixedStore::new(vec![])),
            Some(settings_with(
                Some(crate::config::settings::CompactionSettingsPatch {
                    enabled: Some(false),
                    reserve_tokens: Some(1_000),
                    keep_recent_tokens: None,
                }),
                None,
            )),
        );
        let s = session.effective_compaction_settings();
        assert!(!s.enabled);
        assert_eq!(s.reserve_tokens, 1_000);
    }

    #[test]
    fn build_retry_policy_enabled_maps_three_fields() {
        let (session, _) = test_session(
            Arc::new(FixedStore::new(vec![])),
            Some(settings_with(
                None,
                Some(crate::config::settings::RetrySettingsPatch {
                    enabled: Some(true),
                    max_retries: Some(5),
                    base_delay_ms: Some(500),
                    ..Default::default()
                }),
            )),
        );
        let policy = session.build_retry_policy().expect("enabled → Some");
        assert!(policy.enabled);
        assert_eq!(policy.max_retries, 5);
        assert_eq!(policy.base_delay_ms, 500);
    }

    #[test]
    fn build_retry_policy_disabled_returns_none() {
        let (session, _) = test_session(
            Arc::new(FixedStore::new(vec![])),
            Some(settings_with(
                None,
                Some(crate::config::settings::RetrySettingsPatch {
                    enabled: Some(false),
                    ..Default::default()
                }),
            )),
        );
        assert!(session.build_retry_policy().is_none());
    }

    // ── Stale message check ────────────────────────────────────────

    #[test]
    fn stale_check_false_without_compaction_entry() {
        let (session, _) = test_session(
            Arc::new(FixedStore::new(vec![user_entry("u1", "hi")])),
            None,
        );
        let a = overflow_error_assistant(T1_MS + 1_000);
        assert!(!session.assistant_precedes_latest_compaction(&a));
    }

    #[test]
    fn stale_check_true_for_pre_compaction_message() {
        let (session, _) = test_session(
            Arc::new(FixedStore::new(vec![
                user_entry("u1", "hi"),
                compaction_entry("c1"),
            ])),
            None,
        );
        // T1 < entry T2 → stale
        let old = overflow_error_assistant(T1_MS);
        assert!(session.assistant_precedes_latest_compaction(&old));
        // after T2 → fresh
        let fresh = overflow_error_assistant(T2_MS + 1_000);
        assert!(!session.assistant_precedes_latest_compaction(&fresh));
    }

    // ── check_compaction gate logic ───────────────────────────────

    #[tokio::test]
    async fn check_compaction_disabled_by_settings_is_noop() {
        let store = Arc::new(FixedStore::new(vec![user_entry("u1", "hi")]));
        let (session, _) = test_session(
            Arc::clone(&store),
            Some(settings_with(
                Some(crate::config::settings::CompactionSettingsPatch {
                    enabled: Some(false),
                    ..Default::default()
                }),
                None,
            )),
        );
        let a = overflow_error_assistant(T1_MS + 1_000);
        let cont = session.check_compaction(&a, true).await.unwrap();
        assert!(!cont);
        assert_eq!(store.compaction_count(), 0);
    }

    #[tokio::test]
    async fn check_compaction_overflow_compacts_and_signals_retry() {
        let store = Arc::new(FixedStore::new(vec![user_entry("u1", "hi")]));
        let (session, events) = test_session(Arc::clone(&store), None);
        let a = overflow_error_assistant(T1_MS + 1_000);
        let cont = session.check_compaction(&a, true).await.unwrap();
        // overflow + non-stop → compact then retry
        assert!(cont);
        assert_eq!(store.compaction_count(), 1);
        // Event sequence: Start{overflow} → End{overflow, will_retry=true, result=Some}
        let guard = events.lock().unwrap();
        let evts = compaction_events(&guard);
        assert!(matches!(
            evts[0],
            AgentEvent::CompactionStart {
                reason: CompactionReason::Overflow
            }
        ));
        match evts[1] {
            AgentEvent::CompactionEnd {
                reason,
                result,
                aborted,
                will_retry,
                error_message,
            } => {
                assert!(matches!(reason, CompactionReason::Overflow));
                assert!(result.is_some());
                assert!(!aborted);
                assert!(will_retry);
                assert!(error_message.is_none());
            }
            _ => panic!("expected CompactionEnd"),
        }
    }

    #[tokio::test]
    async fn check_compaction_overflow_recovery_budget_is_one() {
        let store = Arc::new(FixedStore::new(vec![user_entry("u1", "hi")]));
        let (session, events) = test_session(Arc::clone(&store), None);
        // First overflow: compact + retry (budget set)
        let first = overflow_error_assistant(T1_MS + 1_000);
        assert!(session.check_compaction(&first, true).await.unwrap());
        // Second overflow (timestamp after the compaction entry → not stale): budget exhausted → restore failure event
        let second = overflow_error_assistant(T2_MS + 1_000);
        let cont = session.check_compaction(&second, true).await.unwrap();
        assert!(!cont);
        let guard = events.lock().unwrap();
        let evts = compaction_events(&guard);
        match evts.last().unwrap() {
            AgentEvent::CompactionEnd {
                result,
                error_message,
                will_retry,
                ..
            } => {
                assert!(result.is_none());
                assert!(!*will_retry);
                assert!(error_message
                    .as_deref()
                    .unwrap()
                    .contains("recovery failed"));
            }
            _ => panic!("expected CompactionEnd"),
        }
    }

    #[tokio::test]
    async fn check_compaction_threshold_usage_triggers_compaction() {
        // window 65536 − reserved 16384 = 49152; usage 50000 > threshold → threshold compaction
        let store = Arc::new(FixedStore::new(vec![user_entry("u1", "hi")]));
        let (session, events) = test_session(Arc::clone(&store), None);
        let a = assistant_with_usage(50_000, T1_MS + 1_000);
        let cont = session.check_compaction(&a, true).await.unwrap();
        // not will_retry, no queued messages → false
        assert!(!cont);
        assert_eq!(store.compaction_count(), 1);
        let guard = events.lock().unwrap();
        let evts = compaction_events(&guard);
        assert!(matches!(
            evts[0],
            AgentEvent::CompactionStart {
                reason: CompactionReason::Threshold
            }
        ));
    }

    #[tokio::test]
    async fn check_compaction_stale_error_message_never_retriggers() {
        // Compacted entry is at T2; overflow error timestamp is at T1 (before compaction) → gate ④ skipped, no compaction
        let store = Arc::new(FixedStore::new(vec![
            user_entry("u1", "hi"),
            compaction_entry("c1"),
        ]));
        let (session, _) = test_session(store, None);
        let stale = overflow_error_assistant(T1_MS);
        let cont = session.check_compaction(&stale, true).await.unwrap();
        assert!(!cont);
    }

    // ── Manual compaction ─────────────────────────────────────────────────

    #[tokio::test]
    async fn manual_compact_empty_session_errors_nothing_to_compact() {
        let (session, events) = test_session(Arc::new(FixedStore::new(vec![])), None);
        let err = session.compact(None).await.unwrap_err();
        assert!(matches!(err, AgentSessionError::NothingToCompact));
        // pi semantics: start always emitted; error path emits error end
        let guard = events.lock().unwrap();
        let evts = compaction_events(&guard);
        assert_eq!(evts.len(), 2);
        assert!(matches!(
            evts[0],
            AgentEvent::CompactionStart {
                reason: CompactionReason::Manual
            }
        ));
        match evts[1] {
            AgentEvent::CompactionEnd {
                result,
                error_message,
                ..
            } => {
                assert!(result.is_none());
                assert!(error_message
                    .as_deref()
                    .unwrap()
                    .contains("Nothing to compact"));
            }
            _ => panic!("expected CompactionEnd"),
        }
    }

    #[tokio::test]
    async fn manual_compact_already_compacted_errors() {
        let (session, _) = test_session(
            Arc::new(FixedStore::new(vec![
                user_entry("u1", "hi"),
                compaction_entry("c1"),
            ])),
            None,
        );
        let err = session.compact(None).await.unwrap_err();
        assert!(matches!(err, AgentSessionError::AlreadyCompacted));
    }

    #[tokio::test]
    async fn manual_compact_success_persists_rebuilds_and_emits_events() {
        let store = Arc::new(FixedStore::new(vec![user_entry("u1", "hello world")]));
        let (session, events) = test_session(Arc::clone(&store), None);
        let result = session.compact(Some("focus on errors")).await.unwrap();
        assert_eq!(result.summary, "compacted summary");
        assert_eq!(store.compaction_count(), 1);
        // Context reconstruction: compaction summary message first, retained_tail (original user message) after
        // (transform rule 3: retained_tail Some → [compaction, ...compacted entries])
        let msgs = session.agent.messages();
        assert_eq!(msgs.len(), 2);
        let text = match &msgs[0] {
            Message::User(u) => match &u.content {
                ContentBlockList::Text(s) => s.clone(),
                _ => String::new(),
            },
            _ => String::new(),
        };
        assert!(text.contains("compacted summary"));
        // Event sequence: Start{manual} → End{manual, result=Some}
        let guard = events.lock().unwrap();
        let evts = compaction_events(&guard);
        assert_eq!(evts.len(), 2);
        assert!(matches!(
            evts[0],
            AgentEvent::CompactionStart {
                reason: CompactionReason::Manual
            }
        ));
        match evts[1] {
            AgentEvent::CompactionEnd {
                result: r,
                aborted,
                will_retry,
                error_message,
                ..
            } => {
                assert!(r.is_some());
                assert!(!aborted);
                assert!(!will_retry);
                assert!(error_message.is_none());
            }
            _ => panic!("expected CompactionEnd"),
        }
    }

    #[test]
    fn overflow_event_names_serialize_to_pi_literals() {
        // serde contract: CompactionReason lowercase + AgentEvent snake_case variant names
        // Exactly aligned with pi event name literals ("compaction_start"/"compaction_end"/"manual"…)
        let json = serde_json::to_string(&AgentEvent::CompactionStart {
            reason: CompactionReason::Manual,
        })
        .unwrap();
        assert_eq!(json, r#"{"type":"compaction_start","reason":"manual"}"#);
        let json = serde_json::to_string(&AgentEvent::CompactionStart {
            reason: CompactionReason::Overflow,
        })
        .unwrap();
        assert_eq!(json, r#"{"type":"compaction_start","reason":"overflow"}"#);
        let json = serde_json::to_string(&AgentEvent::CompactionStart {
            reason: CompactionReason::Threshold,
        })
        .unwrap();
        assert_eq!(json, r#"{"type":"compaction_start","reason":"threshold"}"#);
    }
}
