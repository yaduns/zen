use crate::compaction::{CompactionReason, CompactionResult};
use serde::Serialize;
use std::collections::HashSet;
use std::sync::Arc;
use zen_ai::tool::{Tool, ToolExecutionMode, ToolExecutionResult};
use zen_ai::types::{
    AssistantMessage, AssistantMessageEvent, Context, Message, Model, ThinkingLevel, ToolCall,
    ToolResultMessage, Usage,
};

// REQ-AGENT-01 (pi-232, pi-205): Agent core types

/// Public runtime snapshot of the agent, containing current configuration, transcript messages, streaming state, and error info.
/// Pure runtime struct; does not derive serde.
pub struct AgentState {
    pub system_prompt: String,
    pub model: Model,
    pub thinking_level: ThinkingLevel,
    /// Available tools
    pub tools: Vec<Arc<dyn Tool>>,
    /// Completed conversation history (all messages are finalized)
    pub messages: Vec<Message>,
    pub is_run: bool,
    /// Current assistant message under construction, not yet finished
    pub streaming_message: Option<Message>,
    /// Set of tool call IDs currently executing
    pub pending_tool_calls: HashSet<String>,
    /// Error message from the most recently failed or interrupted assistant turn
    pub error_message: Option<String>,
}

/// Error returned when attempting to start a new run from a non-idle state (`AgentState::start_run`).
///
/// Aligns with pi's busy semantics (`agent.ts` `runWithLifecycle` "Agent is already processing");
/// under zen, `&mut self` already prevents reentrancy at compile time; this error mainly serves as an explicit guard for callers to check.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("Agent is already running")]
pub struct AlreadyRunningError;

impl AgentState {
    /// Start a new run from the idle state (aligned with pi `runWithLifecycle`, `agent.ts:483`).
    ///
    /// Sets `is_run = true`, clears `streaming_message`.
    ///
    /// # Return Value
    /// - `Ok(())`: started, `is_run` is true.
    /// - `Err(AlreadyRunningError)`: already running.
    ///
    /// # Edge Cases
    /// - Ensure `is_idle()` is true before calling.
    pub fn start_run(&mut self) -> Result<(), AlreadyRunningError> {
        if self.is_run {
            return Err(AlreadyRunningError);
        }
        self.is_run = true;
        self.streaming_message = None;
        Ok(())
    }

    /// Finish the current run and reset transient fields (aligned with pi `finishRun`, `agent.ts:515`).
    ///
    /// Resets `is_run`, clears `streaming_message`, clears `pending_tool_calls`.
    /// `error_message` is **not** cleared (aligned with pi: handled by `reset`). No-op when idle.
    pub fn finish_run(&mut self) {
        self.is_run = false;
        self.streaming_message = None;
        self.pending_tool_calls.clear();
    }

    /// Force-reset transient fields (including `error_message`), callable in any state (aligned with pi `reset`, `agent.ts:327`).
    ///
    /// Note: clearing `messages` is handled by the engine (`Agent::reset`); this method only resets transient fields.
    pub fn reset(&mut self) {
        self.is_run = false;
        self.streaming_message = None;
        self.pending_tool_calls.clear();
        self.error_message = None;
    }

    /// Whether it is currently running (`is_run`).
    pub fn is_run(&self) -> bool {
        self.is_run
    }

    /// Whether it is currently idle (`!is_run`).
    pub fn is_idle(&self) -> bool {
        !self.is_run
    }

    /// Whether there is a pending tool call to execute.
    pub fn has_pending_tool_calls(&self) -> bool {
        !self.pending_tool_calls.is_empty()
    }

    /// Reference to the message currently being built by streaming (if any).
    pub fn streaming_message(&self) -> Option<&Message> {
        self.streaming_message.as_ref()
    }

    /// Update transient fields from an [`AgentEvent`] (aligned with pi `processEvents`, `agent.ts:529-556`).
    ///
    /// Event-driven field reductions:
    /// - `MessageStart`/`MessageUpdate`：`streaming_message = Some(message)`。
    /// - `MessageEnd`: `streaming_message = None` (message is finalized; caller pushes it into `messages`).
    /// - `ToolExecutionStart`：`pending_tool_calls.insert(id)`。
    /// - `ToolExecutionEnd`：`pending_tool_calls.remove(id)`。
    /// - `TurnEnd`: if the message is assistant and carries `error_message`, write it to `self.error_message`.
    /// - `TurnStart`/`AgentStart`/`AgentEnd`/`ToolExecutionUpdate`: no-op
    ///   (run start/end is handled by [`AgentState::start_run`]/[`AgentState::finish_run`]).
    ///
    /// # Edge Cases
    /// - This method does not modify `messages` (the message is finalized at `MessageEnd`; the caller pushes it).
    /// - The engine calls this during event handling (`process_events`); transient fields keep single-point reduction.
    pub fn apply_event(&mut self, event: &AgentEvent) {
        match event {
            AgentEvent::MessageStart { message } | AgentEvent::MessageUpdate { message, .. } => {
                self.streaming_message = Some(message.clone());
            }
            AgentEvent::MessageEnd { .. } => {
                self.streaming_message = None;
            }
            AgentEvent::ToolExecutionStart { tool_call_id, .. } => {
                self.pending_tool_calls.insert(tool_call_id.clone());
            }
            AgentEvent::ToolExecutionEnd { tool_call_id, .. } => {
                self.pending_tool_calls.remove(tool_call_id);
            }
            AgentEvent::TurnEnd { message, .. } => {
                if let Message::Assistant(msg) = message {
                    if let Some(err) = &msg.error_message {
                        self.error_message = Some(err.clone());
                    }
                }
            }
            AgentEvent::TurnStart
            | AgentEvent::AgentStart
            | AgentEvent::AgentEnd { .. }
            | AgentEvent::ToolExecutionUpdate { .. }
            // Compaction lifecycle events are injected and consumed by the app-layer scheduler (T-AP-04-3) via Agent::emit,
            // They are outside the engine state-machine reduction surface (streaming_message/pending_tool_calls are unrelated to compaction)
            | AgentEvent::CompactionStart { .. }
            | AgentEvent::CompactionEnd { .. } => {}
        }
    }
}

/// Events emitted by the agent while running, for UI updates and logging.
/// Events covering four phases: agent lifecycle, turn lifecycle, message lifecycle, and tool execution lifecycle.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// Agent starts processing prompt
    AgentStart,
    /// Agent processing complete
    AgentEnd { messages: Vec<Message> },
    /// New turn starts
    TurnStart,
    /// Turn ends (including tool results)
    TurnEnd {
        message: Message,
        tool_results: Vec<ToolResultMessage>,
    },
    /// New message starts
    MessageStart { message: Message },
    /// Streaming message update
    MessageUpdate {
        message: Message,
        assistant_message_event: AssistantMessageEvent,
    },
    /// Message completed
    MessageEnd { message: Message },
    /// Tool execution starts
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
    },
    /// Tool execution progress update
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
        partial_result: serde_json::Value,
    },
    /// Tool execution completed
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: ToolExecutionResult,
        is_error: bool,
    },
    /// Compaction begins (T-AP-04-3: injected by the app-layer scheduler via `Agent::emit`, not an engine-generated event;
    /// Broadcast after prepare succeeds and before the summary LLM call starts; the UI displays the 「Compacting context...」 status line accordingly)
    CompactionStart { reason: CompactionReason },
    /// Compaction ends (success: result=Some/aborted=false; canceled: aborted=true; failure: error_message=Some).
    /// The result is boxed: CompactionResult is large (contains the summary and retained_tail), substantially larger than the other lightweight variants
    /// (clippy::large_enum_variant); this also reduces the cost of cloning events.
    CompactionEnd {
        reason: CompactionReason,
        result: Option<Box<CompactionResult>>,
        aborted: bool,
        will_retry: bool,
        error_message: Option<String>,
    },
}

// --------------Hook data model-----------

/// Return value of the beforeToolCall hook; block = true prevents tool execution, and the loop will inject an error tool result.
#[derive(Debug, Clone)]
pub struct BeforeToolCallResult {
    pub block: bool,
    pub reason: Option<String>,
}

/// Return value of the afterToolCall hook, used to override certain fields of the executed tool result. Field-level merge semantics (not a deep merge)
#[derive(Debug, Clone)]
pub struct AfterToolCallResult {
    /// Replace the entire content array
    pub content: Option<Vec<zen_ai::types::ContentBlock>>,
    /// Replace the entire details value
    pub details: Option<serde_json::Value>,
    /// Replace error flag
    pub is_error: Option<bool>,
    /// Replace tool execution usage
    pub usage: Option<Usage>,
    /// Replace early-termination notice
    pub terminate: Option<bool>,
}

/// Context parameters for the beforeToolCall hook
pub struct BeforeToolCallContext {
    /// Assistant message that initiates a tool call
    pub assistant_message: AssistantMessage,
    /// Raw tool call block
    pub tool_call: ToolCall,
    /// Parameters validated against the target tool's schema
    pub args: serde_json::Value,
    /// Snapshot of the current agent context
    pub context: Context,
}

/// Context parameters for the afterToolCall hook
pub struct AfterToolCallContext {
    /// Assistant message that initiates a tool call
    pub assistant_message: AssistantMessage,
    /// Raw tool call block
    pub tool_call: ToolCall,
    /// Validated arguments
    pub args: serde_json::Value,
    /// Tool result after execution (hook overrides not applied)
    pub result: ToolExecutionResult,
    /// Whether the current result is considered an error
    pub is_error: bool,
    /// Snapshot of the current agent context
    pub context: Context,
}

/// Context parameters for the shouldStopAfterTurn hook; determines whether the agent loop should stop after the current turn.
pub struct ShouldStopAfterTurnContext {
    /// Assistant message completing the current turn
    pub message: AssistantMessage,
    /// Tool result messages passed to the turn_end event
    pub tool_results: Vec<ToolResultMessage>,
    /// Agent context after appending the assistant message and tool results
    pub context: Context,
    /// Message to return if this loop exits
    pub new_messages: Vec<Message>,
}

/// Type alias for the `prepareNextTurn` hook context, identical to `ShouldStopAfterTurnContext`
pub type PrepareNextTurnContext = ShouldStopAfterTurnContext;

/// Return value of the prepareNextTurn hook; replaces the context/model/thinking state of the next turn.
/// All fields are Option; omitted fields keep their current values.
pub struct AgentLoopTurnUpdate {
    /// Replace the context for the next turn
    pub context: Option<Context>,
    /// Replace the model for the next turn
    pub model: Option<Model>,
    /// Thinking level for the next turn
    pub thinking_level: Option<ThinkingLevel>,
}

/// Pure data configuration for the agent loop.
///
/// Contains only static configuration parameters (model, API key, execution mode, etc.); all dynamic behavior is expressed via
/// the `AgentLoopHooks` trait. Constructed purely at runtime, not serializable.
#[derive(Debug, Clone)]
pub struct AgentLoopConfig {
    /// LLM model to use
    pub model: Model,
    /// Static API key
    pub api_key: Option<String>,
    /// Tool execution mode: parallel (default) or sequential
    pub tool_execution: ToolExecutionMode,
    pub reasoning: Option<ThinkingLevel>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use zen_ai::types::{
        AssistantMessage, AssistantMessageEvent, ContentBlock, ContentBlockList, Cost, StopReason,
        TextContent, UserMessage,
    };

    // ---- helpers ----

    fn default_model() -> Model {
        Model {
            id: "test-model".into(),
            name: None,
            provider: "test".into(),
            api: zen_ai::provider::Api::OpenAiCompletions,
            base_url: "https://api.test.com".into(),
            context_window: 128_000,
            max_tokens: 4096,
            reasoning: false,
            headers: None,
            cost_rates: None,
        }
    }

    fn test_state() -> AgentState {
        AgentState {
            system_prompt: String::new(),
            model: default_model(),
            thinking_level: ThinkingLevel::Off,
            tools: vec![],
            messages: vec![],
            is_run: false,
            streaming_message: None,
            pending_tool_calls: HashSet::new(),
            error_message: None,
        }
    }

    fn running_state() -> AgentState {
        let mut s = test_state();
        s.start_run().unwrap();
        s
    }

    fn toolcalling_state(tool_ids: &[&str]) -> AgentState {
        let mut s = running_state();
        s.pending_tool_calls = tool_ids.iter().map(|s| s.to_string()).collect();
        s
    }

    fn test_user_message() -> Message {
        Message::User(UserMessage {
            content: ContentBlockList::Text("hello".into()),
            timestamp: 1000,
        })
    }

    fn test_updated_user_message() -> Message {
        Message::User(UserMessage {
            content: ContentBlockList::Text("updated".into()),
            timestamp: 2000,
        })
    }

    fn test_assistant_message() -> AssistantMessage {
        AssistantMessage {
            content: vec![ContentBlock::Text(TextContent {
                text: "thinking...".into(),
                text_signature: None,
            })],
            api: "test".into(),
            provider: "test".into(),
            model: "test".into(),
            response_model: None,
            response_id: None,
            usage: test_usage(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 1000,
        }
    }

    fn failing_assistant_message() -> AssistantMessage {
        let mut m = test_assistant_message();
        m.error_message = Some("boom".into());
        m
    }

    fn test_usage() -> Usage {
        Usage {
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: 0,
            cost: Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
        }
    }

    fn tool_result_message(id: &str) -> ToolResultMessage {
        ToolResultMessage {
            tool_call_id: id.into(),
            tool_name: "bash".into(),
            content: vec![ContentBlock::Text(TextContent {
                text: "ok".into(),
                text_signature: None,
            })],
            details: None,
            usage: None,
            is_error: false,
            added_tool_names: None,
            timestamp: 0,
        }
    }

    /// Assert that streaming_message is Some and its text content equals the expected string
    fn assert_streaming_msg_text(state: &AgentState, expected_text: &str) {
        let msg = state
            .streaming_message()
            .expect("expected streaming message");
        match msg {
            Message::User(u) => match &u.content {
                ContentBlockList::Text(t) => assert_eq!(t, expected_text),
                _ => panic!("expected Text content"),
            },
            _ => panic!("expected User message"),
        }
    }

    // ---- start_run ----

    #[test]
    fn test_start_run_from_idle() {
        let mut state = test_state();
        assert!(state.start_run().is_ok());
        assert!(state.is_run());
        assert!(state.streaming_message().is_none());
    }

    #[test]
    fn test_start_run_from_running_returns_err() {
        let mut state = running_state();
        let result = state.start_run();
        assert_eq!(result, Err(AlreadyRunningError));
        assert!(state.is_run());
    }

    #[test]
    fn test_start_run_from_toolcalling_returns_err() {
        let mut state = toolcalling_state(&["call-1"]);
        let result = state.start_run();
        assert_eq!(result, Err(AlreadyRunningError));
        assert!(state.is_run());
    }

    // ---- finish_run ----

    #[test]
    fn test_finish_run_from_running() {
        let mut state = running_state();
        state.finish_run();
        assert!(state.is_idle());
    }

    #[test]
    fn test_finish_run_from_toolcalling() {
        let mut state = toolcalling_state(&["call-1", "call-2"]);
        state.finish_run();
        assert!(state.is_idle());
        assert!(state.streaming_message().is_none());
    }

    #[test]
    fn test_finish_run_from_idle_is_noop() {
        let mut state = test_state();
        state.finish_run();
        assert!(state.is_idle());
    }

    #[test]
    fn test_finish_run_keeps_error_message() {
        let mut state = running_state();
        state.error_message = Some("boom".into());
        state.finish_run();
        assert!(state.is_idle());
        // error_message is cleared by reset (aligned with pi finishRun)
        assert_eq!(state.error_message.as_deref(), Some("boom"));
    }

    // ---- is_run / is_idle ----

    #[test]
    fn test_is_run() {
        assert!(!test_state().is_run());
        assert!(running_state().is_run());
        assert!(toolcalling_state(&["x"]).is_run());
    }

    #[test]
    fn test_is_idle() {
        assert!(test_state().is_idle());
        assert!(!running_state().is_idle());
        assert!(!toolcalling_state(&["x"]).is_idle());
    }

    // ---- has_pending_tool_calls ----

    #[test]
    fn test_has_pending_tool_calls_idle() {
        assert!(!test_state().has_pending_tool_calls());
    }

    #[test]
    fn test_has_pending_tool_calls_running() {
        assert!(!running_state().has_pending_tool_calls());
    }

    #[test]
    fn test_has_pending_tool_calls_toolcalling_empty() {
        let state = toolcalling_state(&[]);
        assert!(!state.has_pending_tool_calls());
    }

    #[test]
    fn test_has_pending_tool_calls_toolcalling_nonempty() {
        let state = toolcalling_state(&["call-1"]);
        assert!(state.has_pending_tool_calls());
        let state = toolcalling_state(&["call-1", "call-2"]);
        assert!(state.has_pending_tool_calls());
    }

    // ---- streaming_message ----

    #[test]
    fn test_streaming_message_idle() {
        assert!(test_state().streaming_message().is_none());
    }

    #[test]
    fn test_streaming_message_running_none() {
        assert!(running_state().streaming_message().is_none());
    }

    #[test]
    fn test_streaming_message_running_some() {
        let mut state = running_state();
        state.streaming_message = Some(test_user_message());
        assert_streaming_msg_text(&state, "hello");
    }

    #[test]
    fn test_streaming_message_toolcalling_none() {
        assert!(toolcalling_state(&["x"]).streaming_message().is_none());
    }

    #[test]
    fn test_streaming_message_toolcalling_some() {
        let mut state = toolcalling_state(&["x"]);
        state.streaming_message = Some(test_user_message());
        assert_streaming_msg_text(&state, "hello");
    }

    // ---- reset ----

    #[test]
    fn test_reset_from_running() {
        let mut state = running_state();
        state.reset();
        assert!(state.is_idle());
    }

    #[test]
    fn test_reset_from_toolcalling() {
        let mut state = toolcalling_state(&["call-1", "call-2"]);
        state.reset();
        assert!(state.is_idle());
    }

    #[test]
    fn test_reset_from_idle_is_noop() {
        let mut state = test_state();
        state.reset();
        assert!(state.is_idle());
    }

    #[test]
    fn test_reset_clears_error_message() {
        let mut state = running_state();
        state.error_message = Some("boom".into());
        state.reset();
        assert!(state.is_idle());
        assert!(state.error_message.is_none());
    }

    // ---- apply_event: MessageStart ----

    #[test]
    fn test_apply_message_start_from_running() {
        let mut state = running_state();
        state.apply_event(&AgentEvent::MessageStart {
            message: test_user_message(),
        });
        assert!(state.is_run());
        assert_streaming_msg_text(&state, "hello");
    }

    #[test]
    fn test_apply_message_start_from_toolcalling_preserves_pending() {
        let mut state = toolcalling_state(&["call-1"]);
        state.apply_event(&AgentEvent::MessageStart {
            message: test_user_message(),
        });
        assert!(state.has_pending_tool_calls());
        assert_streaming_msg_text(&state, "hello");
    }

    #[test]
    fn test_apply_message_start_from_idle_sets_streaming_but_not_running() {
        let mut state = test_state();
        state.apply_event(&AgentEvent::MessageStart {
            message: test_user_message(),
        });
        // Aligned with pi processEvents: set streaming_message unconditionally; is_run is controlled by start_run
        assert!(state.streaming_message().is_some());
        assert!(!state.is_run());
    }

    // ---- apply_event: MessageUpdate ----

    #[test]
    fn test_apply_message_update_overwrites_streaming_message() {
        let mut state = running_state();
        state.apply_event(&AgentEvent::MessageStart {
            message: test_user_message(),
        });
        state.apply_event(&AgentEvent::MessageUpdate {
            message: test_updated_user_message(),
            assistant_message_event: AssistantMessageEvent::Start {
                partial: test_assistant_message(),
            },
        });
        assert_streaming_msg_text(&state, "updated");
    }

    #[test]
    fn test_apply_message_update_from_toolcalling_preserves_pending() {
        let mut state = toolcalling_state(&["call-1"]);
        state.apply_event(&AgentEvent::MessageUpdate {
            message: test_user_message(),
            assistant_message_event: AssistantMessageEvent::Start {
                partial: test_assistant_message(),
            },
        });
        assert!(state.has_pending_tool_calls());
        assert_streaming_msg_text(&state, "hello");
    }

    // ---- apply_event: MessageEnd ----

    #[test]
    fn test_apply_message_end_clears_streaming_message() {
        let mut state = running_state();
        state.apply_event(&AgentEvent::MessageStart {
            message: test_user_message(),
        });
        assert!(state.streaming_message().is_some());

        state.apply_event(&AgentEvent::MessageEnd {
            message: test_user_message(),
        });
        assert!(state.streaming_message().is_none());
        assert!(state.is_run());
    }

    #[test]
    fn test_apply_message_end_from_toolcalling_preserves_pending() {
        let mut state = toolcalling_state(&["call-1", "call-2"]);
        state.apply_event(&AgentEvent::MessageEnd {
            message: test_user_message(),
        });
        assert!(state.has_pending_tool_calls());
        assert!(state.streaming_message().is_none());
    }

    // ---- apply_event: ToolExecutionStart ----

    #[test]
    fn test_apply_tool_execution_start_adds_pending() {
        let mut state = running_state();
        state.apply_event(&AgentEvent::ToolExecutionStart {
            tool_call_id: "call-1".into(),
            tool_name: "bash".into(),
            args: json!({}),
        });
        assert!(state.has_pending_tool_calls());
    }

    #[test]
    fn test_apply_tool_execution_start_appends() {
        let mut state = toolcalling_state(&["call-1"]);
        state.apply_event(&AgentEvent::ToolExecutionStart {
            tool_call_id: "call-2".into(),
            tool_name: "read".into(),
            args: json!({}),
        });
        assert_eq!(state.pending_tool_calls.len(), 2);
        assert!(state.pending_tool_calls.contains("call-1"));
        assert!(state.pending_tool_calls.contains("call-2"));
    }

    #[test]
    fn test_apply_tool_execution_start_from_idle_inserts_pending() {
        let mut state = test_state();
        state.apply_event(&AgentEvent::ToolExecutionStart {
            tool_call_id: "call-1".into(),
            tool_name: "bash".into(),
            args: json!({}),
        });
        // Aligned with pi processEvents: unconditionally insert pending_tool_calls
        assert!(state.pending_tool_calls.contains("call-1"));
    }

    // ---- apply_event: ToolExecutionUpdate (no-op) ----

    #[test]
    fn test_apply_tool_execution_update_does_not_change_state() {
        let mut state = running_state();
        state.apply_event(&AgentEvent::ToolExecutionUpdate {
            tool_call_id: "call-1".into(),
            tool_name: "bash".into(),
            args: json!({}),
            partial_result: json!({}),
        });
        assert!(state.is_run());
        assert!(!state.has_pending_tool_calls());
    }

    // ---- apply_event: TurnStart / TurnEnd / AgentStart / AgentEnd ----

    #[test]
    fn test_apply_noop_events_do_not_change_state() {
        let mut state = running_state();
        state.apply_event(&AgentEvent::TurnStart);
        assert!(state.is_run());
        state.apply_event(&AgentEvent::TurnEnd {
            message: test_user_message(), // User messages have no error_message, so transients are unaffected
            tool_results: vec![],
        });
        assert!(state.is_run());
        state.apply_event(&AgentEvent::AgentStart);
        assert!(state.is_run());
        state.apply_event(&AgentEvent::AgentEnd {
            messages: vec![test_user_message()],
        });
        assert!(state.is_run());
    }

    #[test]
    fn test_apply_turn_end_sets_error_message_from_assistant() {
        let mut state = running_state();
        state.apply_event(&AgentEvent::TurnEnd {
            message: Message::Assistant(failing_assistant_message()),
            tool_results: vec![],
        });
        assert_eq!(state.error_message.as_deref(), Some("boom"));
    }

    // ---- apply_event: ToolExecutionEnd ----

    #[test]
    fn test_apply_tool_execution_end_removes_pending() {
        let mut state = toolcalling_state(&["call-1", "call-2"]);
        state.apply_event(&AgentEvent::ToolExecutionEnd {
            tool_call_id: "call-1".into(),
            tool_name: "bash".into(),
            result: ToolExecutionResult {
                message: tool_result_message("call-1"),
                terminate: false,
            },
            is_error: false,
        });
        assert!(state.has_pending_tool_calls());
        assert!(state.pending_tool_calls.contains("call-2"));
        assert!(!state.pending_tool_calls.contains("call-1"));
    }

    #[test]
    fn test_apply_tool_execution_end_last_clears_pending() {
        let mut state = toolcalling_state(&["call-1"]);
        state.apply_event(&AgentEvent::ToolExecutionEnd {
            tool_call_id: "call-1".into(),
            tool_name: "bash".into(),
            result: ToolExecutionResult {
                message: tool_result_message("call-1"),
                terminate: false,
            },
            is_error: false,
        });
        assert!(!state.has_pending_tool_calls());
    }

    // ---- Multi-tool call sequence ----

    #[test]
    fn test_multiple_tool_execution_starts_accumulate() {
        let mut state = running_state();
        state.apply_event(&AgentEvent::ToolExecutionStart {
            tool_call_id: "call-1".into(),
            tool_name: "bash".into(),
            args: json!({}),
        });
        assert!(state.has_pending_tool_calls());
        state.apply_event(&AgentEvent::ToolExecutionStart {
            tool_call_id: "call-2".into(),
            tool_name: "read".into(),
            args: json!({}),
        });
        assert_eq!(state.pending_tool_calls.len(), 2);
    }

    // ---- Lifecycle: Idle → Running → Idle ----

    #[test]
    fn test_full_lifecycle_idle_running_idle() {
        let mut state = test_state();
        state.start_run().unwrap();
        assert!(state.is_run());
        state.apply_event(&AgentEvent::MessageStart {
            message: test_user_message(),
        });
        assert!(state.streaming_message().is_some());
        state.apply_event(&AgentEvent::MessageEnd {
            message: test_user_message(),
        });
        assert!(state.streaming_message().is_none());
        state.finish_run();
        assert!(state.is_idle());
    }

    // ---- Lifecycle: insert tool call into message stream ----

    #[test]
    fn test_lifecycle_with_tool_call_start_during_streaming() {
        let mut state = test_state();
        state.start_run().unwrap();
        state.apply_event(&AgentEvent::MessageStart {
            message: test_user_message(),
        });
        assert_streaming_msg_text(&state, "hello");

        state.apply_event(&AgentEvent::ToolExecutionStart {
            tool_call_id: "call-1".into(),
            tool_name: "bash".into(),
            args: json!({}),
        });
        assert!(state.has_pending_tool_calls());
        assert_streaming_msg_text(&state, "hello");

        state.apply_event(&AgentEvent::MessageEnd {
            message: test_user_message(),
        });
        assert!(state.has_pending_tool_calls());
        assert!(state.streaming_message().is_none());
    }

    // ---- After reset returns to idle and can start again ----

    #[test]
    fn test_reset_clears_state_and_allows_restart() {
        let mut state = running_state();
        state.apply_event(&AgentEvent::ToolExecutionStart {
            tool_call_id: "stuck-call".into(),
            tool_name: "bash".into(),
            args: json!({}),
        });
        assert!(state.has_pending_tool_calls());
        state.reset();
        assert!(state.is_idle());
        state.start_run().unwrap();
        assert!(state.is_run());
    }
}
