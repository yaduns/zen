use crate::harness::types::AgentEvent;
use std::collections::HashSet;
use zen_ai::types::Message;

/// State machine of the agent runtime, precisely representing the agent's current processing stage.
/// Idle means a new prompt can be accepted; Running means streaming LLM output with no pending tool calls; ToolCalling means executing tool calls (at least one pending).
enum AgentRuntimeState {
    /// Initial state, or after finish_run
    Idle,
    /// After start_run
    Running { streaming_message: Option<Message> },
    /// After a tool_execution_start event
    ToolCalling {
        streaming_message: Option<Message>,
        pending_tool_calls: HashSet<String>,
    },
}

/// Error returned when attempting to start a new run in a non-Idle state.
/// If extension is needed in the future, this can be changed to an enum.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("Agent is already running")]
pub struct AlreadyRunningError;

impl AgentRuntimeState {
    /// Start a new run from the Idle state.
    ///
    /// # Return Value
    /// - `Ok(())`: Successfully transitioned to Running; `streaming_message` initialized to `None`.
    /// - `Err(AlreadyRunningError)`: Already in a non-Idle state.
    ///
    /// # Edge Cases
    /// - Must ensure `is_idle()` is true before calling.
    pub fn start_run(&mut self) -> Result<(), AlreadyRunningError> {
        if !matches!(self, AgentRuntimeState::Idle) {
            return Err(AlreadyRunningError);
        }
        *self = AgentRuntimeState::Running {
            streaming_message: None,
        };
        Ok(())
    }

    /// Complete the current run and reset to Idle.
    ///
    /// Clear all runtime fields (`streaming_message`, `pending_tool_calls`).
    /// No-op when called in the Idle state.
    ///
    /// # Edge Cases
    /// - The caller should handle the `agent_end` event and finalize the message before calling this method.
    /// - This method does not care about `error_message` — `error_message` is a field of `AgentState`, managed by the caller.
    pub fn finish_run(&mut self) {
        *self = AgentRuntimeState::Idle;
    }

    /// Update the state machine state according to the AgentEvent.
    ///
    /// Event-driven state transitions:
    /// - `MessageStart`: set `streaming_message` to the message in the event.
    /// - `MessageUpdate`: update `streaming_message` to the message in the event.
    /// - `MessageEnd`: clear `streaming_message` (the message is finalized; the caller appends it to `messages`).
    /// - `ToolExecutionStart`: if in Running state, transition to ToolCalling and add `tool_call_id`;
    ///   if in ToolCalling state, stay in ToolCalling and add `tool_call_id`;
    ///   if in Idle state, ignore (should not happen; guaranteed by the event sequence).
    /// - `ToolExecutionEnd`: if in ToolCalling state, remove `tool_call_id`;
    ///   if `pending_tool_calls` is empty after removal, transition to Running state;
    ///   if in Running or Idle state, ignore.
    /// - `TurnStart`: no-op.
    /// - `TurnEnd`: no-op (the state machine does not care about the error_message content).
    /// - `AgentStart`: no-op (the state is already set by `start_run`).
    /// - `AgentEnd`: no-op (reset is handled by `finish_run`).
    ///
    /// # Parameters
    /// - `event`: agent event reference
    ///
    /// # Edge Cases
    /// - This method does not modify the `messages` list — on `MessageEnd` the message is finalized; the caller should `push` it itself.
    /// - This method does not handle `error_message` — the caller should check and set it themselves at `TurnEnd`.
    pub fn apply_event(&mut self, event: &AgentEvent) {
        match event {
            // New message started/updated: set streaming_message
            AgentEvent::MessageStart { message } | AgentEvent::MessageUpdate { message, .. } => {
                let prev = std::mem::replace(self, AgentRuntimeState::Idle);
                match prev {
                    AgentRuntimeState::Running { .. } => {
                        *self = AgentRuntimeState::Running {
                            streaming_message: Some(message.clone()),
                        };
                    }
                    AgentRuntimeState::ToolCalling {
                        pending_tool_calls, ..
                    } => {
                        *self = AgentRuntimeState::ToolCalling {
                            streaming_message: Some(message.clone()),
                            pending_tool_calls,
                        };
                    }
                    other => *self = other,
                }
            }
            // Message completed: clear streaming_message
            AgentEvent::MessageEnd { .. } => {
                // Take the contents out of dest for you, and stuff src in as a placeholder.
                let prev = std::mem::replace(self, AgentRuntimeState::Idle);
                match prev {
                    AgentRuntimeState::Running { .. } => {
                        *self = AgentRuntimeState::Running {
                            streaming_message: None,
                        };
                    }
                    AgentRuntimeState::ToolCalling {
                        pending_tool_calls, ..
                    } => {
                        *self = AgentRuntimeState::ToolCalling {
                            streaming_message: None,
                            pending_tool_calls,
                        };
                    }
                    other => *self = other,
                }
            }
            // Tool call started: Running → ToolCalling, ToolCalling → append
            AgentEvent::ToolExecutionStart { tool_call_id, .. } => {
                let prev = std::mem::replace(self, AgentRuntimeState::Idle);
                match prev {
                    AgentRuntimeState::Running { streaming_message } => {
                        let mut pending = HashSet::new();
                        pending.insert(tool_call_id.clone());
                        *self = AgentRuntimeState::ToolCalling {
                            streaming_message,
                            pending_tool_calls: pending,
                        };
                    }
                    AgentRuntimeState::ToolCalling {
                        streaming_message,
                        mut pending_tool_calls,
                    } => {
                        pending_tool_calls.insert(tool_call_id.clone());
                        *self = AgentRuntimeState::ToolCalling {
                            streaming_message,
                            pending_tool_calls,
                        };
                    }
                    other => *self = other,
                }
            }
            // Tool call ended: ToolCalling → remove ID, switch back to Running if empty
            AgentEvent::ToolExecutionEnd { tool_call_id, .. } => {
                let prev = std::mem::replace(self, AgentRuntimeState::Idle);
                match prev {
                    AgentRuntimeState::ToolCalling {
                        streaming_message,
                        mut pending_tool_calls,
                    } => {
                        pending_tool_calls.remove(tool_call_id);
                        if pending_tool_calls.is_empty() {
                            *self = AgentRuntimeState::Running { streaming_message };
                        } else {
                            *self = AgentRuntimeState::ToolCalling {
                                streaming_message,
                                pending_tool_calls,
                            };
                        }
                    }
                    other => *self = other,
                }
            }
            // No-op event
            AgentEvent::TurnStart
            | AgentEvent::TurnEnd { .. }
            | AgentEvent::AgentStart
            | AgentEvent::AgentEnd { .. }
            | AgentEvent::ToolExecutionUpdate { .. } => {}
        }
    }

    /// Determines whether currently running (not Idle).
    pub fn is_streaming(&self) -> bool {
        !matches!(self, AgentRuntimeState::Idle)
    }

    /// Determines whether currently idle.
    pub fn is_idle(&self) -> bool {
        matches!(self, AgentRuntimeState::Idle)
    }

    /// Determines whether there are pending tool calls.
    pub fn has_pending_tool_calls(&self) -> bool {
        // Guards can appear after any match arm or matches!/if let pattern
        // Variables destructured from the pattern (such as pending_tool_calls here) can be used in the guard
        matches!(
            self,
            AgentRuntimeState::ToolCalling {
                pending_tool_calls,
                ..
            } if !pending_tool_calls.is_empty()
        )
    }

    /// Returns a reference to the message currently being built by streaming (if any).
    pub fn streaming_message(&self) -> Option<&Message> {
        match self {
            AgentRuntimeState::Running { streaming_message }
            | AgentRuntimeState::ToolCalling {
                streaming_message, ..
            } => streaming_message.as_ref(),
            // as_ref() converts Option<T> / Result<T, E> into Option<&T> / Result<&T, &E>, letting you borrow the inner value read-only without taking ownership
            // When you need combinators like .map(), .and_then() but don't want to consume the original value, add a .as_ref() at the start.
            AgentRuntimeState::Idle => None,
        }
    }

    /// Forcibly resets to the Idle state, clearing all runtime fields.
    /// Can be called in any state.
    pub fn reset(&mut self) {
        *self = AgentRuntimeState::Idle;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::types::AgentEvent;
    use serde_json::json;
    use zen_ai::types::{
        AssistantMessage, AssistantMessageEvent, ContentBlock, ContentBlockList, Cost, Message,
        TextContent, Usage, UserMessage,
    };

    // ---- helpers ----

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
            stop_reason: zen_ai::types::StopReason::Stop,
            error_message: None,
            timestamp: 1000,
        }
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

    fn running_state() -> AgentRuntimeState {
        AgentRuntimeState::Running {
            streaming_message: None,
        }
    }

    fn toolcalling_state(tool_ids: &[&str]) -> AgentRuntimeState {
        AgentRuntimeState::ToolCalling {
            streaming_message: None,
            pending_tool_calls: tool_ids.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Assert that streaming_message is Some and its text content equals the expected string
    fn assert_streaming_msg_text(state: &AgentRuntimeState, expected_text: &str) {
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
        let mut state = AgentRuntimeState::Idle;
        assert!(state.start_run().is_ok());
        assert!(state.is_streaming());
        assert!(state.streaming_message().is_none());
    }

    #[test]
    fn test_start_run_from_running_returns_err() {
        let mut state = running_state();
        let result = state.start_run();
        assert_eq!(result, Err(AlreadyRunningError));
        assert!(state.is_streaming());
    }

    #[test]
    fn test_start_run_from_toolcalling_returns_err() {
        let mut state = toolcalling_state(&["call-1"]);
        let result = state.start_run();
        assert_eq!(result, Err(AlreadyRunningError));
        assert!(state.is_streaming());
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
        let mut state = AgentRuntimeState::Idle;
        state.finish_run();
        assert!(state.is_idle());
    }

    // ---- is_streaming / is_idle ----

    #[test]
    fn test_is_streaming() {
        assert!(!AgentRuntimeState::Idle.is_streaming());
        assert!(running_state().is_streaming());
        assert!(toolcalling_state(&["x"]).is_streaming());
    }

    #[test]
    fn test_is_idle() {
        assert!(AgentRuntimeState::Idle.is_idle());
        assert!(!running_state().is_idle());
        assert!(!toolcalling_state(&["x"]).is_idle());
    }

    // ---- has_pending_tool_calls ----

    #[test]
    fn test_has_pending_tool_calls_idle() {
        assert!(!AgentRuntimeState::Idle.has_pending_tool_calls());
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
        assert!(AgentRuntimeState::Idle.streaming_message().is_none());
    }

    #[test]
    fn test_streaming_message_running_none() {
        assert!(running_state().streaming_message().is_none());
    }

    #[test]
    fn test_streaming_message_running_some() {
        let state = AgentRuntimeState::Running {
            streaming_message: Some(test_user_message()),
        };
        assert_streaming_msg_text(&state, "hello");
    }

    #[test]
    fn test_streaming_message_toolcalling_none() {
        let state = toolcalling_state(&["x"]);
        assert!(state.streaming_message().is_none());
    }

    #[test]
    fn test_streaming_message_toolcalling_some() {
        let state = AgentRuntimeState::ToolCalling {
            streaming_message: Some(test_user_message()),
            pending_tool_calls: HashSet::from(["x".into()]),
        };
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
        let mut state = AgentRuntimeState::Idle;
        state.reset();
        assert!(state.is_idle());
    }

    // ---- apply_event: MessageStart ----

    #[test]
    fn test_apply_message_start_from_running() {
        let mut state = running_state();
        state.apply_event(&AgentEvent::MessageStart {
            message: test_user_message(),
        });
        assert!(state.is_streaming());
        assert_streaming_msg_text(&state, "hello");
    }

    #[test]
    fn test_apply_message_start_from_toolcalling() {
        let mut state = toolcalling_state(&["call-1"]);
        state.apply_event(&AgentEvent::MessageStart {
            message: test_user_message(),
        });
        assert!(state.has_pending_tool_calls());
        assert_streaming_msg_text(&state, "hello");
    }

    #[test]
    fn test_apply_message_start_from_idle_is_ignored() {
        let mut state = AgentRuntimeState::Idle;
        state.apply_event(&AgentEvent::MessageStart {
            message: test_user_message(),
        });
        assert!(state.is_idle());
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
        assert!(state.is_streaming());
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
    fn test_apply_tool_execution_start_from_running_transitions_to_toolcalling() {
        let mut state = running_state();
        state.apply_event(&AgentEvent::ToolExecutionStart {
            tool_call_id: "call-1".into(),
            tool_name: "bash".into(),
            args: json!({}),
        });
        assert!(state.has_pending_tool_calls());
    }

    #[test]
    fn test_apply_tool_execution_start_from_toolcalling_appends() {
        let mut state = toolcalling_state(&["call-1"]);
        state.apply_event(&AgentEvent::ToolExecutionStart {
            tool_call_id: "call-2".into(),
            tool_name: "read".into(),
            args: json!({}),
        });
        assert!(state.has_pending_tool_calls());
        match &state {
            AgentRuntimeState::ToolCalling {
                pending_tool_calls, ..
            } => {
                assert_eq!(pending_tool_calls.len(), 2);
                assert!(pending_tool_calls.contains("call-1"));
                assert!(pending_tool_calls.contains("call-2"));
            }
            _ => panic!("expected ToolCalling state"),
        }
    }

    #[test]
    fn test_apply_tool_execution_start_from_idle_is_ignored() {
        let mut state = AgentRuntimeState::Idle;
        state.apply_event(&AgentEvent::ToolExecutionStart {
            tool_call_id: "call-1".into(),
            tool_name: "bash".into(),
            args: json!({}),
        });
        assert!(state.is_idle());
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
        assert!(state.is_streaming());
        assert!(!state.has_pending_tool_calls());
    }

    // ---- apply_event: TurnStart / TurnEnd / AgentStart / AgentEnd ----

    #[test]
    fn test_apply_noop_events_do_not_change_state() {
        let mut state = running_state();
        state.apply_event(&AgentEvent::TurnStart);
        assert!(state.is_streaming());
        state.apply_event(&AgentEvent::TurnEnd {
            message: test_user_message(),
            tool_results: vec![],
        });
        assert!(state.is_streaming());
        state.apply_event(&AgentEvent::AgentStart);
        assert!(state.is_streaming());
        state.apply_event(&AgentEvent::AgentEnd {
            messages: vec![test_user_message()],
        });
        assert!(state.is_streaming());
    }

    // ---- apply_event: ToolExecutionEnd ----
    // Note: ToolExecutionResult's fields are private and cannot be constructed from this module.
    // Therefore the full transition test for ToolExecutionEnd (remove ID / switch back to Running when empty)
    // Add this after ToolExecutionResult provides a pub constructor or pub(crate) fields.

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
        match &state {
            AgentRuntimeState::ToolCalling {
                pending_tool_calls, ..
            } => {
                assert_eq!(pending_tool_calls.len(), 2);
            }
            _ => panic!("expected ToolCalling"),
        }
    }

    // ---- Lifecycle: Idle → Running → Idle ----

    #[test]
    fn test_full_lifecycle_idle_running_idle() {
        let mut state = AgentRuntimeState::Idle;
        state.start_run().unwrap();
        assert!(state.is_streaming());
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

    // ---- Lifecycle: Idle → Running → ToolCalling (tool call inserted into message stream) ----

    #[test]
    fn test_lifecycle_with_tool_call_start_during_streaming() {
        let mut state = AgentRuntimeState::Idle;
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

    // ---- After reset, the state machine returns to Idle and can start again ----

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
        assert!(state.is_streaming());
    }
}
