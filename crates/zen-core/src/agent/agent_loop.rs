use crate::agent::types::{
    AfterToolCallContext, AfterToolCallResult, AgentEvent, AgentLoopConfig,
    AgentLoopTurnUpdate, BeforeToolCallContext, BeforeToolCallResult, PrepareNextTurnContext,
    ShouldStopAfterTurnContext, StreamFn, ToolExecutionMode,
};
use async_trait::async_trait;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use zen_ai::tool::{Tool, ToolExecutionResult};
use zen_ai::types::{
    AssistantMessage, ContentBlock, Context, Message, TextContent, ToolCall, ToolResultMessage,
    Usage,
};

/// Fixed behavioral policies for the agent loop.
///
/// Unlike `AgentLoopConfig` (pure data configuration), this trait encapsulates all the "how to do it" policies:
/// message conversion, context transformation, API key retrieval, stop criteria, turn switching, steering/follow-up messages,
/// hooks before and after tool execution, event dispatch.
///
/// Every method has a default implementation covering the most common "do nothing" scenario. Callers only need to override
/// the methods that truly need customization.
///
/// # Design principles
/// These policies are determined at session startup and do not change dynamically while the agent loop runs.
/// Therefore a trait is preferable to function pointer fields — it avoids the non-serializable, non-Clone issues caused by
/// stuffing `Arc<dyn Fn>` into the config struct.
#[async_trait]
pub trait AgentLoopHooks: Send + Sync {
    /// Filter out messages that should not be sent to the LLM context.
    ///
    /// Default implementation: keep everything.
    /// The application layer may override this to skip verbose tool output, etc.
    async fn filter_messages(&self, messages: Vec<Message>) -> Vec<Message> {
        messages
    }

    /// Transform messages after `filter_messages` and before sending to the LLM.
    ///
    /// Used for context window management (truncating old messages) or injecting external context.
    /// The default implementation returns the original messages unchanged.
    async fn transform_context(
        &self,
        messages: Vec<Message>,
        cancel: &CancellationToken,
    ) -> Vec<Message> {
        drop(cancel); // Default implementation ignores cancellation signals
        messages
    }

    /// Dynamically retrieve the API key.
    ///
    /// Only called when `AgentLoopConfig::api_key` is `None`.
    /// Returns `None` by default (uses the static key from the config; if absent, the LLM call fails).
    async fn get_api_key(&self, _provider: &str) -> Option<String> {
        None
    }

    /// Determine whether the agent loop should stop after the current turn.
    ///
    /// Return `true` to stop immediately (before poll steering/follow-up).
    /// Defaults to `false` (never stop on its own).
    async fn should_stop_after_turn(&self, _context: &ShouldStopAfterTurnContext) -> bool {
        false
    }

    /// Prepare the runtime state for the next turn.
    ///
    /// Return `Some(AgentLoopTurnUpdate)` to override the context / model / thinking level for the next turn.
    /// Defaults to `None` (no changes).
    async fn prepare_next_turn(
        &self,
        _context: &PrepareNextTurnContext,
    ) -> Option<AgentLoopTurnUpdate> {
        None
    }

    /// Get steering messages (queued messages injected while the agent is running).
    ///
    /// Called after each tool batch completes; if a non-empty message list is returned, it is injected before the next LLM call.
    /// Returns an empty vec by default.
    async fn get_steering_messages(&self) -> Vec<Message> {
        vec![]
    }

    /// Get follow-up messages (messages injected after the agent stops).
    ///
    /// Called when the agent has no more tool calls and no steering messages.
    /// If a non-empty message list is returned, the agent continues running.
    /// Returns an empty vec by default.
    async fn get_follow_up_messages(&self) -> Vec<Message> {
        vec![]
    }

    /// Called before tool execution; return `Some(BeforeToolCallResult { block: true, .. })` to prevent execution.
    ///
    /// Defaults to `Ok(None)` (do not block).
    /// Returning `Err(reason)` is equivalent to blocking execution, with `reason` as the error message.
    async fn before_tool_call(
        &self,
        _context: &BeforeToolCallContext,
        _cancel: &CancellationToken,
    ) -> Result<Option<BeforeToolCallResult>, String> {
        Ok(None)
    }

    /// Called after tool execution; can override certain fields of the tool result.
    ///
    /// Defaults to `Ok(None)` (no overrides).
    /// Returning `Err(reason)` is equivalent to marking the tool result as an error, with `reason` as the error message.
    async fn after_tool_call(
        &self,
        _context: &AfterToolCallContext,
        _cancel: &CancellationToken,
    ) -> Result<Option<AfterToolCallResult>, String> {
        Ok(None)
    }

    /// Receive every event emitted by the agent loop.
    ///
    /// Default implementation is a no-op (events are not handled).
    /// Print mode should serialize events and write them to stdout in this method;
    /// TUI mode should forward them to the UI rendering thread via a channel in this method;
    /// Tests can collect events into a `Vec` in this method.
    async fn on_event(&self, _event: &AgentEvent) {
        // no-op
    }
}

/// Recoverable error type in the agent loop.
#[derive(Debug, Clone, thiserror::Error)]
pub enum AgentLoopError {
    /// Attempt to resume the loop from an empty context
    #[error("Cannot continue: no messages in context")]
    EmptyContext,

    /// The last message role is "assistant" (cannot continue from an assistant message)
    #[error("Cannot continue from message role: assistant")]
    InvalidLastRole,
}

/// Preprocessed tool call; tool definition found and arguments validated.
pub(crate) struct PreparedToolCall {
    /// Raw tool call
    pub tool_call: ToolCall,
    /// Found tool definition
    pub tool: Arc<dyn Tool>,
    /// Validated arguments
    pub args: serde_json::Value,
}

/// Executed tool call result (not processed by the `after_tool_call` hook).
pub(crate) struct ExecutedToolCallOutcome {
    /// Tool execution result
    pub result: ToolExecutionResult,
    /// Whether it is an error
    pub is_error: bool,
}

/// Finalized tool call result (`after_tool_call` hook overrides applied).
pub(crate) struct FinalizedToolCallOutcome {
    /// Raw tool call
    pub tool_call: ToolCall,
    /// Final tool execution result
    pub result: ToolExecutionResult,
    /// Whether it is an error
    pub is_error: bool,
}

/// Start a new agent loop beginning with the prompt messages.
///
/// Appends `prompts` to the end of `context`, then repeatedly runs the "LLM call → tool execution → LLM call" loop,
/// until the LLM no longer requests tool calls or a stop condition is met. All events are sent via `hooks.on_event()`.
///
/// # Parameters
/// - `prompts`: list of new prompt messages to process
/// - `context`: starting agent context (system prompt, existing messages, available tools). It is mutated (new messages are appended).
/// - `config`: plain-data configuration (model, API key, execution mode, etc.)
/// - `hooks`: fixed behavior policies (message transformation, hooks, stop determination, event handling, etc.)
/// - `cancel`: cancellation token; when a cancellation signal is received the agent loop should stop as soon as possible
/// - `stream_fn`: LLM streaming call function
///
/// # Return Value
/// Returns all messages newly produced during this loop (excluding those already in `context.messages`).
///
/// # Edge Cases
/// - If `prompts` is empty, returns an empty vec immediately (no LLM call is made).
/// - If `stream_fn` returns an error, the error is encoded via the `AgentEvent::AgentEnd` event and the loop terminates.
/// - On cancellation, the loop emits `turn_end` + `agent_end` events and exits without panicking.
/// - If hook methods (`before_tool_call`, `after_tool_call`, etc.) return `Err`, the error is captured and encoded as a tool error result rather than propagated.
/// - Errors occurring inside `hooks.on_event()` are handled by the hooks implementer and are not propagated into the loop.
pub async fn run_agent_loop(
    prompts: Vec<Message>,
    context: &mut Context,
    config: &AgentLoopConfig,
    hooks: &dyn AgentLoopHooks,
    cancel: &CancellationToken,
    stream_fn: StreamFn,
) -> Vec<Message> {
    todo!()
}

/// Continue the agent loop from the current context without adding a new prompt.
///
/// Used for retry scenarios — the context already contains a user message or tool result, proceed directly to the LLM call.
///
/// # Parameters
/// - `context`: current agent context (the last message must convert to a user or toolResult message,
///   otherwise the LLM provider will reject the request)
/// - `config`: pure data configuration
/// - `hooks`: fixed behavior policy
/// - `cancel`: cancellation token
/// - `stream_fn`: LLM streaming call function
///
/// # Return Value
/// Returns the messages added during this loop.
///
/// # Errors
/// If `context.messages` is empty, returns `Err(AgentLoopError::EmptyContext)`.
/// If the last message's role is "assistant", returns `Err(AgentLoopError::InvalidLastRole)`.
///
/// # Edge Cases
/// Same as `run_agent_loop`, but without adding a new prompt.
pub async fn run_agent_loop_continue(
    context: &mut Context,
    config: &AgentLoopConfig,
    hooks: &dyn AgentLoopHooks,
    cancel: &CancellationToken,
    stream_fn: StreamFn,
) -> Result<Vec<Message>, AgentLoopError> {
    todo!()
}

/// Stream the LLM's assistant response.
///
/// Calls `stream_fn` to obtain an `AssistantMessageEventStream`, processes all events (start/delta/end),
/// and assembles the final `AssistantMessage`. Events are forwarded via `hooks.on_event()`.
///
/// In this function, `Message` is converted to `Message` (LLM-compatible format) and sent to the LLM.
///
/// # Parameters
/// - `context`: current agent context (it is mutated: partial messages are appended, eventually replaced by the finalized message)
/// - `config`: pure data configuration
/// - `hooks`: used for `transform_context`, `filter_messages`, and `on_event`
/// - `cancel`: cancellation token
/// - `stream_fn`: LLM streaming call function
///
/// # Return Value
/// Final `AssistantMessage` (with complete content, usage, stop_reason).
///
/// # Edge Cases
/// - If `stream_fn` returns an error, stop_reason is "error" or "aborted".
/// - Calls `hooks.transform_context()` and `hooks.filter_messages()` for message transformation.
/// - If `filter_messages` returns an empty list, behavior is up to the LLM provider (usually an error).
pub(crate) async fn stream_assistant_response(
    context: &mut Context,
    config: &AgentLoopConfig,
    hooks: &dyn AgentLoopHooks,
    cancel: &CancellationToken,
    stream_fn: StreamFn,
) -> AssistantMessage {
    todo!()
}

/// Execute the tool calls in the assistant message.
///
/// Determine the execution strategy based on `config.tool_execution` and each tool's `execution_mode`:
/// - `Sequential`: execute tool calls one by one, starting the next only after the previous completes
/// - `Parallel`: preprocess all tool calls in order, then execute the permitted tools concurrently
///
/// If the assistant message's `stop_reason` is `Length`, all tool calls are treated as truncated and not executed;
/// directly returns error tool results (`fail_tool_calls_from_truncated_message`).
///
/// # Return Value
/// The tool result message list and a flag indicating whether the loop should terminate.
pub(crate) async fn execute_tool_calls(
    context: &mut Context,
    assistant_message: &AssistantMessage,
    config: &AgentLoopConfig,
    hooks: &dyn AgentLoopHooks,
    cancel: &CancellationToken,
) -> (Vec<ToolResultMessage>, bool) {
    todo!()
}

/// Execute tool calls sequentially.
pub(crate) async fn execute_tool_calls_sequential(
    context: &mut Context,
    assistant_message: &AssistantMessage,
    tool_calls: &[ToolCall],
    config: &AgentLoopConfig,
    hooks: &dyn AgentLoopHooks,
    cancel: &CancellationToken,
) -> (Vec<ToolResultMessage>, bool) {
    todo!()
}

/// Execute tool calls concurrently.
///
/// Tool calls are preprocessed in order (`prepare_tool_call`), then the permitted tool calls are executed concurrently;
/// Tool results are ordered by the tool call order in the original assistant message.
pub(crate) async fn execute_tool_calls_parallel(
    context: &mut Context,
    assistant_message: &AssistantMessage,
    tool_calls: &[ToolCall],
    config: &AgentLoopConfig,
    hooks: &dyn AgentLoopHooks,
    cancel: &CancellationToken,
) -> (Vec<ToolResultMessage>, bool) {
    todo!()
}

/// Preprocesses a tool call: find the tool definition, validate arguments, and call `hooks.before_tool_call()`.
///
/// # Return Value
/// `Ok(PreparedToolCall)` means the tool can execute and contains the found tool definition and validated arguments.
/// `Err(ToolExecutionResult)` means tool execution was blocked or the tool was not found; the error result is returned directly.
pub(crate) async fn prepare_tool_call(
    context: &Context,
    assistant_message: &AssistantMessage,
    tool_call: &ToolCall,
    hooks: &dyn AgentLoopHooks,
    cancel: &CancellationToken,
) -> Result<PreparedToolCall, ToolExecutionResult> {
    todo!()
}

/// Execute a preprocessed tool call.
///
/// Invoke `tool.execute()`, handling streaming update callbacks.
///
/// # Parameters
/// - `prepared`: preprocessing result (tool definition and validated arguments)
/// - `cancel`: cancellation token
/// - `hooks`: used to emit `tool_execution_update` events via `on_event`
///
/// # Return Value
/// Tool execution result (success or failure).
pub(crate) async fn execute_prepared_tool_call(
    prepared: &PreparedToolCall,
    cancel: &CancellationToken,
    hooks: &dyn AgentLoopHooks,
) -> ExecutedToolCallOutcome {
    todo!()
}

/// Apply `hooks.after_tool_call()` to an executed tool call.
///
/// Overwrite fields of the tool result with the return value. If the hook returns `Err`, it is caught and encoded as an error result.
pub(crate) async fn finalize_executed_tool_call(
    context: &Context,
    assistant_message: &AssistantMessage,
    prepared: &PreparedToolCall,
    executed: &ExecutedToolCallOutcome,
    hooks: &dyn AgentLoopHooks,
    cancel: &CancellationToken,
) -> FinalizedToolCallOutcome {
    todo!()
}

/// Handle tool calls truncated due to `stop_reason == Length`.
///
/// Streaming tool call arguments may be truncated (even if the final JSON parses successfully), leaving arguments silently incomplete.
/// All truncated tool calls are not executed; an error tool result is returned directly, prompting the model to retry.
///
/// # Return Value
/// List of error tool result messages, with `terminate` marked as false.
pub(crate) async fn fail_tool_calls_from_truncated_message(
    tool_calls: &[ToolCall],
    hooks: &dyn AgentLoopHooks,
) -> (Vec<ToolResultMessage>, bool) {
    let mut messages = Vec::with_capacity(tool_calls.len());

    for tool_call in tool_calls {
        // Send tool_execution_start event
        // Notify the UI (or log) "a tool call has started executing"
        hooks
            .on_event(&AgentEvent::ToolExecutionStart {
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                args: tool_call.arguments.clone(),
            })
            .await;

        // Construct the error message
        let error_text = format!(
            "Tool call \"{}\" was not executed: the response hit the output token limit, \
                so its arguments may be truncated. Re-issue the tool call with complete arguments.",
            tool_call.name
        );

        // - message: a single text whose content is the error message above; details carries no structured metadata
        // - usage: the tool never executed, zero cost
        // - terminate: false — don't end the loop; let the LLM retry
        let result = ToolExecutionResult {
            message: ToolResultMessage {
                tool_call_id: "".into(),
                tool_name: "".into(),
                content: vec![ContentBlock::Text(TextContent {
                    text: error_text,
                    text_signature: None,
                })],
                details: Some(serde_json::Value::Null),
                usage: Usage::default(),
                is_error: true,
                added_tool_names: None,
                timestamp: 0,
            },
            terminate: false,
        };

        // Messages to inject into the LLM context; tool execution results are appended to this collection
        let tool_result_msg = ToolResultMessage {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            content: result.message.content.clone(),
            details: result.message.details.clone(),
            usage: result.message.usage.clone(),
            is_error: result.message.is_error,
            added_tool_names: result.message.added_tool_names.clone(),
            timestamp: result.message.timestamp,
        };

        // Send tool_execution_end event
        hooks
            .on_event(&AgentEvent::ToolExecutionEnd {
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                result,
                is_error: true,
            })
            .await;

        messages.push(tool_result_msg);
    }

    (messages, false)
}
