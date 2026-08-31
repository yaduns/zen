use crate::harness::agent_hooks::AgentLoopHooks;
use std::collections::HashSet;
use tokio_util::sync::CancellationToken;
use zen_ai::types::Message;

/// Before each `stream_fn` call, the agent loop performs the "last transformation before sending to the LLM":
/// First pass the message history to `hooks.filter_messages()` to remove records that should not enter the context (e.g., bash
/// `!!` exclusions, verbose output), then pass the filtered result to `hooks.transform_context()` for
/// final transformation (context window trimming, external context injection). Analogy: "before entering, first check who cannot enter
/// (filter), then change the outfits of the remaining guests (transform)"—the order must not be reversed.
///
/// # Parameters
/// - `messages`: Full set of messages to send to the LLM this time (snapshot of current `context.messages`).
/// - `hooks`: Fixed behavior policy; actual filter/transform semantics are determined by implementer overrides,
///   with the default implementation (trait default methods) being pass-through.
/// - `cancel`: Cancellation token, passed through to `transform_context` so it can, during long operations such as trimming,
///   respond to cancellation.
///
/// # Return Value
/// Transformed message list, used directly to build the `Context.messages` sent to the LLM.
///
/// # Edge Cases
/// - The output of `filter_messages` is the input of `transform_context` (ordering contract).
/// - After filtering the list may be empty—behavior is decided by the LLM provider (usually an error); this function does not intercept.
/// - Extracted from `stream_assistant_response`: the original two inline steps (agent_loop.rs L194-195)
///   are behaviorally equivalent; this only adds testability and an explicit contract.
pub(crate) async fn filter_and_transform_messages(
    messages: Vec<Message>,
    hooks: &dyn AgentLoopHooks,
    cancel: &CancellationToken,
) -> Vec<Message> {
    // Step 1: Filter (remove records that should not enter context, such as bash `!!` exclusions)
    let filtered = hooks.filter_messages(messages).await;

    // Step 2: Transform (context window trimming, external context injection)
    // Note: order must not be reversed—filter output is transform input
    hooks.transform_context(filtered, cancel).await
}

/// When `filter_excluded` determines whether a history message is excluded from the LLM context, it takes the message's
/// Stable identity key:
/// - `Message::ToolResult` → `Some(tool_call_id)` (unique across the transcript, one-to-one with the
///   tool_call in assistant messages);
/// - `Message::User` → `Some(timestamp)` (stringified millisecond timestamp) — user bash record
///   (T-AP-04-2-fix: `!`/`!!` user commands rendered as user messages) uses the construction time as the exclusion identity;
/// - `Message::Assistant` → `None` (currently no producer; never hit by the exclusion set).
///
/// # Return Value
/// `Some(identity key)` / `None` (messages of this type do not participate in exclusion).
/// The key semantics are the contract surface when T-AP-04-2-fix records the `!!` bash exclusion identity (see the plan seam note).
fn exclusion_key(message: &Message) -> Option<String> {
    match message {
        Message::ToolResult(result) => Some(result.tool_call_id.clone()),
        Message::User(user) => Some(user.timestamp.to_string()),
        Message::Assistant(_) => None,
    }
}

// File: crates/zen-core/src/harness/message_filter.rs

/// The application layer's `AgentLoopHooks::filter_messages` uses this function to implement `excludeFromContext`
/// semantics (hooks-layer policy, no markers added at the data layer): remove records matching the exclusion set from message history.
/// Aligns with pi's `!!command` semantics—bash results the user explicitly requests not enter the LLM context (e.g.,
/// long outputs containing secrets) are not sent to the model with session history.
///
/// # Parameters
/// - `messages`: Message list to filter (usually from `build_context` / engine state).
/// - `excluded`: exclusion set; keys are the values of `exclusion_key` (the `!!` bash ToolResult
///   `tool_call_id` / the UserMessage timestamp of user bash). The application layer
///   (`AgentSession::execute_bash` + `SessionHooks`, T-AP-04-2-fix) is responsible for injecting and consuming them.
///
/// # Return Value
/// List of retained messages (all messages not matching the exclusion set are retained; empty exclusion set = always pass-through).
///
/// # Consumers
/// `SessionHooks::filter_messages` (zen-app `session/services.rs`, wired in T-AP-04-2-fix) is reused cross-
/// crate to implement the hooks-layer excludeFromContext semantics.
pub fn filter_excluded(messages: Vec<Message>, excluded: &HashSet<String>) -> Vec<Message> {
    // Empty exclusion set always passes through, avoiding unnecessary allocation (doc comment already promises this behavior)
    if excluded.is_empty() {
        return messages;
    }

    messages
        .into_iter()
        .filter(|msg| {
            // When exclusion_key returns Some, check whether it is in the exclusion set; when None, always retain
            match exclusion_key(msg) {
                Some(key) => !excluded.contains(&key),
                None => true,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};
    use zen_ai::types::{
        AssistantMessage, ContentBlock, ContentBlockList, StopReason, TextContent,
        ToolResultMessage, Usage, UserMessage,
    };

    // ── Construction helpers ─────────────────────────────────────────────

    fn user_msg(text: &str) -> Message {
        user_msg_at(text, 0)
    }

    fn user_msg_at(text: &str, timestamp: i64) -> Message {
        Message::User(UserMessage {
            content: ContentBlockList::Text(text.to_string()),
            timestamp,
        })
    }

    fn assistant_msg() -> Message {
        Message::Assistant(AssistantMessage {
            content: vec![],
            api: "test".into(),
            provider: "test".into(),
            model: "test".into(),
            response_model: None,
            response_id: None,
            usage: Usage::default(),
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: 0,
        })
    }

    fn tool_result_msg(id: &str) -> Message {
        Message::ToolResult(ToolResultMessage {
            tool_call_id: id.to_string(),
            tool_name: "bash".into(),
            content: vec![ContentBlock::Text(TextContent {
                text: "out".into(),
                text_signature: None,
            })],
            details: None,
            usage: Some(Usage::default()),
            is_error: false,
            added_tool_names: None,
            timestamp: 0,
        })
    }

    // ── MockHooks: records inputs to the filter / transform hooks for asserting the ordering contract ──

    struct MockHooks {
        filter_inputs: Arc<Mutex<Vec<Vec<Message>>>>,
        transform_inputs: Arc<Mutex<Vec<Vec<Message>>>>,
        /// Whether filter drops the first item, used to create a scenario where filtering has an actual effect (default retains all)
        drop_first: bool,
    }

    impl MockHooks {
        fn new(drop_first: bool) -> Self {
            Self {
                filter_inputs: Arc::new(Mutex::new(Vec::new())),
                transform_inputs: Arc::new(Mutex::new(Vec::new())),
                drop_first,
            }
        }
    }

    #[async_trait]
    impl AgentLoopHooks for MockHooks {
        async fn filter_messages(&self, messages: Vec<Message>) -> Vec<Message> {
            self.filter_inputs.lock().unwrap().push(messages.clone());
            if self.drop_first && !messages.is_empty() {
                messages.into_iter().skip(1).collect()
            } else {
                messages
            }
        }

        async fn transform_context(
            &self,
            messages: Vec<Message>,
            cancel: &CancellationToken,
        ) -> Vec<Message> {
            let _ = cancel;
            self.transform_inputs.lock().unwrap().push(messages.clone());
            messages
        }
    }

    // ── filter_and_transform_messages ──────────────────────────────────────

    #[tokio::test]
    async fn test_filter_and_transform_default_passthrough() {
        let hooks = MockHooks::new(false);
        let cancel = CancellationToken::new();
        let msgs = vec![user_msg("a"), tool_result_msg("tc1")];

        let result = filter_and_transform_messages(msgs.clone(), &hooks, &cancel).await;

        // Both filter and transform use default pass-through (drop_first=false); result matches the original list
        assert_eq!(result, msgs);
        assert_eq!(hooks.filter_inputs.lock().unwrap()[0], msgs);
        assert_eq!(hooks.transform_inputs.lock().unwrap()[0], msgs);
    }

    #[tokio::test]
    async fn test_filter_and_transform_ordering_contract() {
        let hooks = MockHooks::new(true); // filter drops the first item
        let cancel = CancellationToken::new();
        let msgs = vec![user_msg("a"), tool_result_msg("tc1")];

        let result = filter_and_transform_messages(msgs.clone(), &hooks, &cancel).await;

        // Ordering contract: filter receives the full list, transform receives only the filtered list (missing the first item)
        let expected_filtered: Vec<Message> = msgs.clone().into_iter().skip(1).collect();
        assert_eq!(hooks.filter_inputs.lock().unwrap()[0], msgs);
        assert_eq!(hooks.transform_inputs.lock().unwrap()[0], expected_filtered);
        // Final result = output of transform (i.e., the filtered list)
        assert_eq!(result, expected_filtered);
    }

    #[tokio::test]
    async fn test_filter_and_transform_filter_to_empty() {
        let hooks = MockHooks::new(true); // A single message is entirely filtered out by drop_first
        let cancel = CancellationToken::new();
        let msgs = vec![user_msg("only")];

        let result = filter_and_transform_messages(msgs.clone(), &hooks, &cancel).await;

        // Filtered result is empty → transform receives an empty list → returns empty (this function does not intercept; provider decides)
        assert!(result.is_empty());
        assert!(hooks.transform_inputs.lock().unwrap()[0].is_empty());
    }

    #[tokio::test]
    async fn test_filter_and_transform_cancel_passthrough() {
        // Passing an already-cancelled token into the pipeline does not panic and performs no cancellation check; it still passes through normally
        let hooks = MockHooks::new(false);
        let cancel = CancellationToken::new();
        cancel.cancel();
        let msgs = vec![user_msg("a")];

        let result = filter_and_transform_messages(msgs.clone(), &hooks, &cancel).await;

        assert_eq!(result, msgs);
    }

    // ── exclusion_key ─────────────────────────────────────────────────────

    #[test]
    fn test_exclusion_key_tool_result_returns_tool_call_id() {
        let msg = tool_result_msg("call-42");
        assert_eq!(exclusion_key(&msg), Some("call-42".to_string()));
    }

    #[test]
    fn test_exclusion_key_user_returns_timestamp() {
        // User bash records (T-AP-04-2-fix) use construction-time milliseconds as the exclusion identity
        let msg = user_msg_at("hi", 12345);
        assert_eq!(exclusion_key(&msg), Some("12345".to_string()));
    }

    #[test]
    fn test_exclusion_key_assistant_is_none() {
        assert_eq!(exclusion_key(&assistant_msg()), None);
    }

    // ── filter_excluded ────────────────────────────────────────────────────

    #[test]
    fn test_filter_excluded_empty_set_passthrough() {
        let msgs = vec![user_msg("a"), tool_result_msg("tc1")];
        let excluded: HashSet<String> = HashSet::new();

        // Empty exclusion set always passes through (promised by doc comment); all retained
        assert_eq!(filter_excluded(msgs.clone(), &excluded), msgs);
    }

    #[test]
    fn test_filter_excluded_drops_matching_tool_result() {
        let msgs = vec![
            user_msg("a"),
            tool_result_msg("tc1"),
            tool_result_msg("tc2"),
        ];
        let excluded: HashSet<String> = ["tc1".to_string()].into_iter().collect();

        let result = filter_excluded(msgs.clone(), &excluded);

        // tc1 matches the exclusion set and is dropped; user messages without identity keys are always retained; tc2 does not match and is retained
        assert_eq!(result, vec![user_msg("a"), tool_result_msg("tc2")]);
    }

    #[test]
    fn test_filter_excluded_keeps_all_when_no_match() {
        let msgs = vec![user_msg("a"), assistant_msg(), tool_result_msg("tc1")];
        let excluded: HashSet<String> = ["other".to_string()].into_iter().collect();

        // The exclusion set matches no message identity keys; all are retained
        assert_eq!(filter_excluded(msgs.clone(), &excluded), msgs);
    }

    #[test]
    fn test_filter_excluded_drops_user_by_timestamp() {
        // T-AP-04-2-fix: `!!` user bash uses the UserMessage timestamp as the exclusion identity
        let extra = user_msg("other");
        let msgs = vec![
            user_msg_at("secret", 777),
            tool_result_msg("tc1"),
            extra.clone(),
        ];
        let excluded: HashSet<String> = ["777".to_string()].into_iter().collect();

        let result = filter_excluded(msgs, &excluded);

        assert_eq!(result, vec![tool_result_msg("tc1"), extra]);
    }
}
