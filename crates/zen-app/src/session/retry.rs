// REQ-APP-01 (pi-301): agent turn-level auto retry orchestration (pi `_handlePostAgentRun` retry branch + `_prepareRetry`)

//! Turn-level auto retry (turn-auto-retry-fix).
//!
//! A transient gateway failure (connection cut, 429 overload, SSE decode failure) must not end the turn and force the
//! user to type "continue" again: pi backs off and re-asks by itself (`agent-session.ts:1080 _handlePostAgentRun` →
//! `_isRetryableError` → `_prepareRetry`). This module is that orchestration on the zen side, reusing everything
//! already in the tree (zen-ai `RetryPolicy` / wording table / backoff formula, `settings.retry`, the status-line
//! component) instead of inventing a second backoff implementation.
//!
//! Layering: the retry lives here (app layer) because it mutates session state (drop the failed assistant message from
//! the retry context, continue the turn) and reports progress to the UI through injected [`AgentEvent`]s — the same
//! shape T-AP-04-3 uses for compaction.

use crate::session::agent_session::AgentSession;
use crate::session::compaction::last_assistant_message;
use crate::session::navigation::build_retry_policy;
use std::sync::atomic::Ordering;
use tokio_util::sync::CancellationToken;
use zen_ai::types::{AssistantMessage, Message, StopReason};
use zen_ai::utils::overflow::is_context_overflow;
use zen_ai::utils::retry::{is_retryable_assistant_error, retry_delay_ms};
use zen_core::harness::types::AgentEvent;

/// Decides whether the just-finished assistant turn is worth backing off and retrying (pi `_isRetryableError`):
/// context overflow is compaction's business and must never be retried in place (retrying only hits the same wall).
///
/// # Behavior
/// 1. Not `Error` → `false` (`Aborted`/`Stop`/`ToolUse`/`Length` are never retried).
/// 2. `is_context_overflow(message, None)` → `false` (pi `_isRetryableError` excludes overflow first).
/// 3. Otherwise defer to `is_retryable_assistant_error` (wording table, non-retryable patterns win).
///
/// # Why there is no `context_window` parameter
/// Non-`Error` messages are already excluded above, and `is_context_overflow`'s "silent overflow" path only applies
/// to `Stop` while its length-truncation path only applies to `Length` (`utils/overflow.rs:94-107`) — so only the
/// error-text regex path is reachable and `context_window` never participates. Passing it would only add a
/// `u64 as usize` conversion (truncating on 32-bit targets) with no effect.
///
/// # Consumers
/// [`AgentSession::check_retry_after_run`]; unit tests.
pub(crate) fn is_retryable_turn(message: &AssistantMessage) -> bool {
    if message.stop_reason != StopReason::Error {
        return false;
    }
    if is_context_overflow(message, None) {
        return false;
    }
    is_retryable_assistant_error(message)
}

/// Before retrying, drop the failed assistant message from the retry context (pi `_prepareRetry`'s
/// `messages.slice(0, -1)`): it was already persisted to the session history on `MessageEnd` (the user still sees the
/// error), but leaving it in `state.messages` would leave the next turn with nothing to answer
/// (`continue_turn` rejects a trailing assistant message with `InvalidLastRole`).
///
/// # Behavior
/// Trailing `Message::Assistant` → popped; otherwise the vector is returned unchanged (user/tool-result messages are
/// never touched).
///
/// # Consumers
/// [`AgentSession::check_retry_after_run`]; `session/compaction.rs:190-194` (overflow branch) delegates here.
/// **Not** used by `compaction.rs:313-319` (will-retry branch): that guard only allows popping a trailing assistant
/// with `stop_reason == Error`, because a normal assistant produced by a successful compaction must be kept.
pub(crate) fn with_trailing_assistant_removed(messages: Vec<Message>) -> Vec<Message> {
    let mut messages = messages;
    if matches!(messages.last(), Some(Message::Assistant(_))) {
        messages.pop();
    }
    messages
}

impl AgentSession {
    /// The post-run retry gate (pi `_handlePostAgentRun`'s retry branch + `_prepareRetry`): "the gateway hiccuped"
    /// must not cost the user another round of typing — this waits it out and asks again.
    ///
    /// # Return
    /// - `true` = `AutoRetryStart` was emitted, the failed message was removed from the context and the abortable backoff
    ///   sleep completed → **the caller should run one more `continue_turn()`**.
    /// - `false` = no retry (success / non-retryable / budget exhausted / backoff cancelled); if a retry sequence was still
    ///   open, `AutoRetryEnd` is emitted and the counter is cleared.
    ///
    /// # Behavior (the order is the contract)
    /// 1. Last assistant message: missing → `false`; `stop_reason != Error` → emit `AutoRetryEnd{success:true}` (only when
    ///    the counter is > 0), clear it, `false`.
    /// 2. `!is_retryable_turn(last)` → when the counter is > 0 emit `AutoRetryEnd{success:false, final_error:
    ///    last.error_message}`, clear it, `false`.
    /// 3. `build_retry_policy(&settings)` is `None` (`settings.retry.enabled == false`) → same as 2.
    /// 4. `attempt = counter + 1`; `attempt > policy.max_retries` → same as 2, and **no sleep**.
    /// 5. `delay = retry_delay_ms(...)`; emit `AutoRetryStart{attempt, max_attempts, delay, error_message}`.
    /// 6. `set_messages(with_trailing_assistant_removed(...))` (pi semantics: kept in session history, dropped from the
    ///    retry context).
    /// 7. Store a fresh `CancellationToken` in `retry_cancel`, then `tokio::select!{ sleep(delay), token.cancelled() }`:
    ///    cancelled → clear the handle, emit `AutoRetryEnd{success:false, final_error:"Retry cancelled"}`, clear the
    ///    counter, `false`; slept through → clear the handle, `true`.
    ///
    /// # Edge Cases
    /// - `&self` (`AtomicU32` + `Mutex<Option<CancellationToken>>`), safe to run concurrently with the main task's
    ///   `abort()` / `abort_retry()`.
    /// - Idempotent: with no last assistant message (empty context) no state is touched.
    /// - No lock is held across the await (`retry_cancel` is written and released immediately).
    ///
    /// # Consumers
    /// `session/agent_session.rs::prompt` (retry loop); unit tests (a fake `StreamFn` injecting retryable errors).
    pub(crate) async fn check_retry_after_run(&self) -> bool {
        let messages = self.agent.messages();
        let Some(last) = last_assistant_message(&messages) else {
            return false;
        };

        // Success / abort / any non-error terminal reason closes the sequence (pi resets the counter on a
        // non-error assistant message).
        if last.stop_reason != StopReason::Error {
            self.finish_retry_sequence(true, last, None);
            return false;
        }
        if !is_retryable_turn(last) {
            self.finish_retry_sequence(false, last, None);
            return false;
        }
        let Some(policy) = build_retry_policy(&self.settings) else {
            self.finish_retry_sequence(false, last, None);
            return false;
        };

        let attempt = self.retry_attempt.load(Ordering::Relaxed) + 1;
        if attempt > policy.max_retries {
            self.finish_retry_sequence(false, last, None);
            return false;
        }
        self.retry_attempt.store(attempt, Ordering::Relaxed);

        let delay_ms = retry_delay_ms(policy.base_delay_ms, attempt);
        self.agent.emit(AgentEvent::AutoRetryStart {
            attempt,
            max_attempts: policy.max_retries,
            delay_ms,
            error_message: last
                .error_message
                .clone()
                .unwrap_or_else(|| "Unknown error".to_string()),
        });

        // pi: the failed assistant message stays in the session history but is removed from the retry context.
        self.agent
            .set_messages(with_trailing_assistant_removed(self.agent.messages()));

        let token = CancellationToken::new();
        {
            let mut slot = self.retry_cancel.lock().unwrap_or_else(|e| e.into_inner());
            *slot = Some(token.clone());
        }

        let cancelled = tokio::select! {
            _ = tokio::time::sleep(std::time::Duration::from_millis(delay_ms)) => false,
            _ = token.cancelled() => true,
        };

        match self.retry_cancel.lock() {
            Ok(mut slot) => *slot = None,
            Err(e) => *e.into_inner() = None,
        }

        if cancelled {
            // pi emits auto_retry_end{success:false, finalError:"Retry cancelled"} from the sleep's catch arm.
            self.retry_attempt.store(0, Ordering::Relaxed);
            self.agent.emit(AgentEvent::AutoRetryEnd {
                success: false,
                attempt,
                final_error: Some("Retry cancelled".to_string()),
            });
            return false;
        }
        true
    }

    /// Emits the closing `AutoRetryEnd` for an open retry sequence and clears the counter (no-op when the counter is 0
    /// so a plain first-turn error does not produce a spurious end event).
    fn finish_retry_sequence(
        &self,
        success: bool,
        last: &AssistantMessage,
        final_error: Option<String>,
    ) {
        let attempt = self.retry_attempt.swap(0, Ordering::Relaxed);
        if attempt == 0 {
            return;
        }
        self.agent.emit(AgentEvent::AutoRetryEnd {
            success,
            attempt,
            final_error: if success {
                None
            } else {
                final_error.or_else(|| last.error_message.clone())
            },
        });
    }

    /// Closes the retry sequence after a failed continuation (used by `prompt` when `continue_turn` returns `Err`),
    /// so the status line never gets stuck on "Retrying..." and the counter does not leak into the next prompt.
    ///
    /// # Consumers
    /// `session/agent_session.rs::prompt` (retry loop's `Err` arm).
    pub(crate) fn finish_retry_after_error(&self, error: &str) {
        let attempt = self.retry_attempt.swap(0, Ordering::Relaxed);
        if attempt == 0 {
            return;
        }
        self.agent.emit(AgentEvent::AutoRetryEnd {
            success: false,
            attempt,
            final_error: Some(error.to_string()),
        });
    }

    /// Cancels the **backoff wait** currently in progress (pi `abortRetry`): the user pressed Esc, meaning "stop
    /// waiting, just leave it here".
    ///
    /// # Behavior
    /// Cancels the token in `retry_cancel`; `None` (not backing off) → no-op. The already-finished turn is untouched
    /// (that is `Agent::abort`'s job).
    ///
    /// # Consumers
    /// `modes/interactive/mod.rs::handle_escape` (Esc while retrying); `session/agent_session.rs::abort`.
    pub fn abort_retry(&self) {
        if let Some(token) = self
            .retry_cancel
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            token.cancel();
        }
    }

    /// Whether a backoff wait is currently in progress (pi `isRetrying`) — the interactive layer routes Esc to
    /// [`Self::abort_retry`] instead of `abort` / double-Esc exit based on this.
    ///
    /// # Consumers
    /// `modes/interactive/mod.rs::handle_escape`.
    pub fn is_retrying(&self) -> bool {
        self.retry_cancel
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::settings::{RetrySettingsPatch, SettingsData, SettingsManager};
    use crate::session::session_manager::SessionManagerError;
    use std::collections::{HashSet, VecDeque};
    use std::sync::atomic::AtomicU32;
    use std::sync::{Arc, Mutex};
    use tokio_util::sync::CancellationToken;
    use zen_ai::api::openai_stream::{StreamConfig, StreamError};
    use zen_ai::provider::Api;
    use zen_ai::tool::ToolExecutionMode;
    use zen_ai::types::{
        AssistantMessageEvent, ContentBlock, Context, Model, NonErrorStopReason, StreamFn,
        TextContent, Usage,
    };
    use zen_core::harness::agent_hooks::AgentLoopHooks;
    use zen_core::harness::engine::Agent;
    use zen_core::harness::types::AgentState;
    use zen_core::session::tree::SessionTreeEntry;

    // ── SessionStore stub (records persisted messages) ──

    #[derive(Default)]
    struct MockStore {
        persisted: Mutex<Vec<Message>>,
    }

    impl MockStore {
        fn persisted(&self) -> Vec<Message> {
            self.persisted.lock().unwrap().clone()
        }
    }

    impl crate::session::agent_session::SessionStore for MockStore {
        fn append_message(&self, message: &Message) {
            self.persisted.lock().unwrap().push(message.clone());
        }
        fn append_model_change(&self, _provider: &str, _model_id: &str) {}
        fn append_thinking_level_change(&self, _level: &str) {}
        fn session_id(&self) -> String {
            "session-1".to_string()
        }
        fn cwd(&self) -> &str {
            "/work"
        }
        fn branch_entries(&self) -> Vec<SessionTreeEntry> {
            Vec::new()
        }
        fn append_compaction(
            &self,
            _result: &zen_core::compaction::CompactionResult,
            _from_hook: bool,
        ) -> String {
            "c1".to_string()
        }
        fn entries(&self) -> Vec<SessionTreeEntry> {
            Vec::new()
        }
        fn get_leaf_id(&self) -> Option<String> {
            None
        }
        fn append_branch_summary(
            &self,
            _result: &zen_core::compaction::branch::BranchSummaryResult,
            _new_leaf_id: Option<&str>,
            _from_hook: bool,
        ) -> Result<String, SessionManagerError> {
            Ok("bs1".to_string())
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
            Ok("l1".to_string())
        }
        fn append_session_info(&self, _name: Option<&str>) -> String {
            "si1".to_string()
        }
        fn create_branched_session(
            &self,
            _leaf_id: &str,
        ) -> Result<Option<std::path::PathBuf>, SessionManagerError> {
            Ok(None)
        }
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
            api: Api::OpenAiCompletions,
            base_url: "https://opencode.ai/zen/go/v1".into(),
            context_window: 65536,
            max_tokens: 8192,
            reasoning: true,
            headers: None,
            cost_rates: None,
            compat: None,
        }
    }

    // ── Scripted stream: each call pops the next scripted outcome ──

    enum Script {
        Ok(&'static str),
        Err(&'static str),
    }

    fn scripted_stream_fn(
        scripts: Arc<Mutex<VecDeque<Script>>>,
        calls: Arc<AtomicU32>,
    ) -> StreamFn {
        Arc::new(
            move |_model: &Model,
                  _ctx: &Context,
                  _cfg: Option<StreamConfig>,
                  _cancel: Option<CancellationToken>| {
                let scripts = Arc::clone(&scripts);
                let calls = Arc::clone(&calls);
                Box::pin(async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    let script = scripts
                        .lock()
                        .unwrap()
                        .pop_front()
                        .unwrap_or(Script::Ok("done"));
                    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<AssistantMessageEvent>();
                    let message =
                        |stop: StopReason, err: Option<&str>, text: &str| AssistantMessage {
                            content: if text.is_empty() {
                                vec![]
                            } else {
                                vec![ContentBlock::Text(TextContent {
                                    text: text.to_string(),
                                    text_signature: None,
                                })]
                            },
                            api: "openai-completions".into(),
                            provider: "opencode-go".into(),
                            model: "deepseek-v4-flash".into(),
                            response_model: None,
                            response_id: None,
                            usage: Usage::default(),
                            stop_reason: stop,
                            error_message: err.map(str::to_string),
                            timestamp: 0,
                        };
                    let _ = tx.send(AssistantMessageEvent::Start {
                        partial: message(StopReason::Stop, None, ""),
                    });
                    match script {
                        Script::Ok(text) => {
                            let _ = tx.send(AssistantMessageEvent::Done {
                                reason: NonErrorStopReason::Stop,
                                message: message(StopReason::Stop, None, text),
                            });
                        }
                        Script::Err(err) => {
                            let _ = tx.send(AssistantMessageEvent::Error {
                                reason: zen_ai::types::ErrorStopReason::Error,
                                error: message(StopReason::Error, Some(err), ""),
                            });
                        }
                    }
                    drop(tx);
                    Ok::<_, StreamError>(rx)
                })
            },
        )
    }

    fn retry_settings(max_retries: u32, base_delay_ms: u64) -> SettingsData {
        SettingsData {
            retry: Some(RetrySettingsPatch {
                enabled: Some(true),
                max_retries: Some(max_retries),
                base_delay_ms: Some(base_delay_ms),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// Assemble a session whose stream is scripted; returns the session, the call counter and the event sink.
    fn test_session(
        scripts: Vec<Script>,
        settings: SettingsData,
    ) -> (AgentSession, Arc<AtomicU32>, Arc<Mutex<Vec<AgentEvent>>>) {
        let (session, calls, events, _store) = test_session_with_store(scripts, settings);
        (session, calls, events)
    }

    /// Retry-orchestration test fixture: session + model-call counter + captured events + persistence store.
    type RetryFixture = (
        AgentSession,
        Arc<AtomicU32>,
        Arc<Mutex<Vec<AgentEvent>>>,
        Arc<MockStore>,
    );

    /// Same as [`test_session`] but also exposes the store so persistence-side semantics can be asserted
    /// (pi keeps the failed assistant message in the session history).
    fn test_session_with_store(scripts: Vec<Script>, settings: SettingsData) -> RetryFixture {
        let queue = Arc::new(Mutex::new(VecDeque::from(scripts)));
        let calls = Arc::new(AtomicU32::new(0));
        let settings_manager = Arc::new(SettingsManager::empty());
        settings_manager.apply_overrides(settings);

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
        let agent = Agent::new(
            state,
            ToolExecutionMode::Parallel,
            Arc::new(MockHooks),
            scripted_stream_fn(queue, Arc::clone(&calls)),
            None,
        );
        let events: Arc<Mutex<Vec<AgentEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&events);
        let _ = agent.subscribe(Box::new(move |e: &AgentEvent| {
            sink.lock().unwrap().push(e.clone());
        }));

        let store = Arc::new(MockStore::default());
        let session = AgentSession::new_with_exclusions(
            agent,
            Arc::clone(&store) as Arc<dyn crate::session::agent_session::SessionStore>,
            Arc::new(Mutex::new(HashSet::new())),
            zen_core::env::ShellConfig::default(),
            settings_manager,
            vec![],
            vec![],
        );
        (session, calls, events, store)
    }

    fn retry_starts(events: &Arc<Mutex<Vec<AgentEvent>>>) -> Vec<(u32, u32, u64)> {
        events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                AgentEvent::AutoRetryStart {
                    attempt,
                    max_attempts,
                    delay_ms,
                    ..
                } => Some((*attempt, *max_attempts, *delay_ms)),
                _ => None,
            })
            .collect()
    }

    fn retry_ends(events: &Arc<Mutex<Vec<AgentEvent>>>) -> Vec<(bool, u32, Option<String>)> {
        events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                AgentEvent::AutoRetryEnd {
                    success,
                    attempt,
                    final_error,
                } => Some((*success, *attempt, final_error.clone())),
                _ => None,
            })
            .collect()
    }

    fn last_assistant(messages: &[Message]) -> Option<&AssistantMessage> {
        messages.iter().rev().find_map(|m| match m {
            Message::Assistant(a) => Some(a),
            _ => None,
        })
    }

    // ── pure helpers ──

    fn assistant(stop: StopReason, err: Option<&str>) -> AssistantMessage {
        AssistantMessage {
            content: vec![],
            api: "openai-completions".into(),
            provider: "opencode-go".into(),
            model: "deepseek-v4-flash".into(),
            response_model: None,
            response_id: None,
            usage: Usage::default(),
            stop_reason: stop,
            error_message: err.map(str::to_string),
            timestamp: 0,
        }
    }

    #[test]
    fn is_retryable_turn_accepts_transient_errors() {
        assert!(is_retryable_turn(&assistant(
            StopReason::Error,
            Some("timeout")
        )));
        assert!(is_retryable_turn(&assistant(
            StopReason::Error,
            Some("SSE stream error: Transport error: error decoding response body")
        )));
    }

    #[test]
    fn is_retryable_turn_rejects_context_overflow() {
        assert!(!is_retryable_turn(&assistant(
            StopReason::Error,
            Some("prompt is too long: 70000 tokens > 65536 maximum")
        )));
    }

    #[test]
    fn is_retryable_turn_rejects_non_error_reasons() {
        assert!(!is_retryable_turn(&assistant(StopReason::Aborted, None)));
        assert!(!is_retryable_turn(&assistant(StopReason::Stop, None)));
        assert!(!is_retryable_turn(&assistant(StopReason::Length, None)));
    }

    #[test]
    fn with_trailing_assistant_removed_pops_only_a_trailing_assistant() {
        let user = Message::User(zen_ai::types::UserMessage {
            content: zen_ai::types::ContentBlockList::Text("hi".into()),
            timestamp: 1,
        });

        let popped = with_trailing_assistant_removed(vec![
            user.clone(),
            Message::Assistant(assistant(StopReason::Error, Some("timeout"))),
        ]);
        assert_eq!(popped.len(), 1);
        assert!(matches!(popped[0], Message::User(_)));

        let untouched = with_trailing_assistant_removed(vec![user.clone()]);
        assert_eq!(untouched.len(), 1);

        // A mid-conversation assistant is not "trailing" and must survive.
        let mid = with_trailing_assistant_removed(vec![
            Message::Assistant(assistant(StopReason::Stop, None)),
            user,
        ]);
        assert_eq!(mid.len(), 2);
    }

    // ── end-to-end orchestration ──

    #[tokio::test]
    async fn retryable_error_then_success_retries_once() {
        let (session, calls, events, store) = test_session_with_store(
            vec![
                Script::Err("error decoding response body"),
                Script::Ok("recovered"),
            ],
            retry_settings(2, 1),
        );

        let out = session
            .prompt("hi", &[])
            .await
            .expect("prompt should complete");

        assert!(!out.is_empty());
        assert_eq!(calls.load(Ordering::SeqCst), 2, "one retry => two calls");
        assert_eq!(
            retry_starts(&events),
            vec![(1, 2, 1)],
            "a single AutoRetryStart with the configured budget/delay"
        );
        assert_eq!(
            retry_ends(&events),
            vec![(true, 1, None)],
            "AutoRetryEnd{{success:true}} must be emitted exactly once"
        );
        assert_eq!(session.retry_attempt.load(Ordering::Relaxed), 0);

        // The failed assistant message left the retry context (still persisted in the session history).
        let messages = session.agent.messages();
        assert_eq!(messages.len(), 2, "user + the successful assistant");
        assert!(matches!(messages[0], Message::User(_)));
        let last = last_assistant(&messages).expect("successful assistant");
        assert_eq!(last.stop_reason, StopReason::Stop);
        // ... while the failed assistant message stays in the persisted session history (pi "keep in session").
        let persisted = store.persisted();
        assert!(
            persisted.iter().any(|m| matches!(
                m,
                Message::Assistant(a) if a.stop_reason == StopReason::Error
            )),
            "failed assistant must still be persisted: {persisted:?}"
        );
    }

    #[tokio::test]
    async fn non_retryable_error_is_not_retried() {
        let (session, calls, events) = test_session(
            vec![Script::Err("API error: status 401: invalid api key")],
            retry_settings(2, 1),
        );

        let _ = session.prompt("hi", &[]).await.expect("prompt completes");

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(retry_starts(&events).is_empty());
        assert!(retry_ends(&events).is_empty());
        let messages = session.agent.messages();
        let last = last_assistant(&messages).expect("error assistant kept");
        assert_eq!(last.stop_reason, StopReason::Error);
    }

    #[tokio::test]
    async fn budget_exhaustion_emits_one_failure_end() {
        let (session, calls, events) = test_session(
            vec![
                Script::Err("timeout"),
                Script::Err("timeout"),
                Script::Err("timeout"),
            ],
            retry_settings(2, 1),
        );

        let _ = session.prompt("hi", &[]).await.expect("prompt completes");

        assert_eq!(calls.load(Ordering::SeqCst), 3, "initial + 2 retries");
        assert_eq!(retry_starts(&events).len(), 2);
        let ends = retry_ends(&events);
        assert_eq!(ends.len(), 1, "one close per sequence");
        assert!(!ends[0].0, "failure");
        assert_eq!(ends[0].1, 2, "attempt at exhaustion");
        assert_eq!(ends[0].2.as_deref(), Some("timeout"));
        assert_eq!(session.retry_attempt.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn escape_during_backoff_cancels_the_retry() {
        let (session, calls, events) = test_session(
            vec![Script::Err("timeout"), Script::Ok("never reached")],
            // 10s backoff: the test never actually waits, `abort_retry` interrupts it.
            retry_settings(3, 10_000),
        );

        // Wake the abort future as soon as the retry is scheduled (notify_one stores a permit, so no race).
        let notify = Arc::new(tokio::sync::Notify::new());
        let notifier = Arc::clone(&notify);
        let _ = session.subscribe(Box::new(move |e: &AgentEvent| {
            if matches!(e, AgentEvent::AutoRetryStart { .. }) {
                notifier.notify_one();
            }
        }));

        let (result, ()) = tokio::join!(session.prompt("hi", &[]), async {
            notify.notified().await;
            assert!(session.is_retrying(), "backoff should be in progress");
            session.abort_retry();
        });

        result.expect("prompt completes after cancellation");
        assert_eq!(calls.load(Ordering::SeqCst), 1, "no second model call");
        let ends = retry_ends(&events);
        assert_eq!(ends.len(), 1);
        assert_eq!(ends[0].2.as_deref(), Some("Retry cancelled"));
        assert!(!session.is_retrying());
        assert_eq!(session.retry_attempt.load(Ordering::Relaxed), 0);
        // The failed message was already dropped from the retry context before the sleep.
        assert_eq!(session.agent.messages().len(), 1);
    }

    #[tokio::test]
    async fn abort_closes_the_backoff_too() {
        let (session, calls, events) = test_session(
            vec![Script::Err("timeout"), Script::Ok("never reached")],
            retry_settings(3, 10_000),
        );

        let notify = Arc::new(tokio::sync::Notify::new());
        let notifier = Arc::clone(&notify);
        let _ = session.subscribe(Box::new(move |e: &AgentEvent| {
            if matches!(e, AgentEvent::AutoRetryStart { .. }) {
                notifier.notify_one();
            }
        }));

        // `AgentSession::abort` (Ctrl+D / shutdown paths) must not leave the run stuck in the backoff sleep.
        let (result, ()) = tokio::join!(session.prompt("hi", &[]), async {
            notify.notified().await;
            session.abort();
        });

        result.expect("prompt completes after abort");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(retry_ends(&events).len(), 1);
    }

    #[tokio::test]
    async fn finish_retry_after_error_closes_the_sequence() {
        let (session, _calls, events) = test_session(vec![], retry_settings(2, 1));
        session.retry_attempt.store(1, Ordering::Relaxed);

        session.finish_retry_after_error("InvalidLastRole");

        assert_eq!(session.retry_attempt.load(Ordering::Relaxed), 0);
        let ends = retry_ends(&events);
        assert_eq!(ends.len(), 1);
        assert!(!ends[0].0);
        assert_eq!(ends[0].2.as_deref(), Some("InvalidLastRole"));
    }

    #[test]
    fn is_retrying_is_false_without_a_backoff() {
        let (session, _calls, _events) = test_session(vec![], retry_settings(2, 1));
        assert!(!session.is_retrying());
        // `abort_retry` while idle is a no-op.
        session.abort_retry();
        assert!(!session.is_retrying());
    }
}
