//! AgentLoopHooks trait — fixed behavior policy for the agent loop.
//!
//! REQ-AGENT-01 (pi-202): hook definitions split out from agent_loop.rs.

use crate::agent::types::{
    AfterToolCallContext, AfterToolCallResult, AgentEvent, AgentLoopTurnUpdate,
    BeforeToolCallContext, BeforeToolCallResult, PrepareNextTurnContext,
    ShouldStopAfterTurnContext,
};
use async_trait::async_trait;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;
use zen_ai::tool::{Tool, ToolExecutionResult, ToolUpdateCallback};
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
        drop(cancel);
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
#[derive(Debug, Clone)]
pub(crate) struct PreparedToolCall {
    /// Raw tool call
    pub tool_call: ToolCall,
    /// Found tool definition
    pub tool: Arc<dyn Tool>,
    /// Validated arguments
    pub args: serde_json::Value,
}

/// Executed tool call result (not processed by the `after_tool_call` hook).
#[derive(Debug, Clone)]
pub(crate) struct ExecutedToolCallOutcome {
    /// Tool execution result
    pub result: ToolExecutionResult,
    /// Whether it is an error
    pub is_error: bool,
}

/// Finalized tool call result (`after_tool_call` hook overrides applied).
#[derive(Debug, Clone)]
pub(crate) struct FinalizedToolCallOutcome {
    /// Raw tool call
    pub tool_call: ToolCall,
    /// Final tool execution result
    pub result: ToolExecutionResult,
    /// Whether it is an error
    pub is_error: bool,
}

pub(crate) async fn prepare_tool_call(
    context: &Context,
    assistant_message: &AssistantMessage,
    tool_call: &ToolCall,
    tools: &[Arc<dyn Tool>],
    hooks: &dyn AgentLoopHooks,
    cancel: &CancellationToken,
) -> Result<PreparedToolCall, ToolExecutionResult> {
    // 1. Find the tool by name
    let tool = tools
        .iter()
        .find(|t| t.name() == tool_call.name)
        .ok_or_else(|| {
            error_tool_result(tool_call, format!("Tool {} not found", tool_call.name))
        })?;

    // 2. Preprocess arguments (type coercion, etc.)
    let args = tool
        .prepare_arguments(&tool_call.arguments)
        .map_err(|err| error_tool_result(tool_call, format!("Invalid arguments: {}", err)))?;

    // 3. Check cancellation signal
    if cancel.is_cancelled() {
        return Err(error_tool_result(tool_call, "Operation aborted".into()));
    }

    // 4. Invoke before_tool_call hook
    let before_ctx = BeforeToolCallContext {
        assistant_message: assistant_message.clone(),
        tool_call: tool_call.clone(),
        args: args.clone(),
        context: context.clone(),
    };

    let before_result = hooks
        .before_tool_call(&before_ctx, cancel)
        .await
        .map_err(|reason| error_tool_result(tool_call, reason))?;

    // 5. Hook blocked execution
    if let Some(result) = before_result {
        if result.block {
            return Err(error_tool_result(
                tool_call,
                result
                    .reason
                    .unwrap_or_else(|| "Tool execution was blocked".into()),
            ));
        }
    }

    // 6. Check cancellation signal again (may be cancelled during hook execution)
    if cancel.is_cancelled() {
        return Err(error_tool_result(tool_call, "Operation aborted".into()));
    }

    // 7. Return the preprocessed tool call
    Ok(PreparedToolCall {
        tool_call: tool_call.clone(),
        tool: Arc::clone(tool),
        args,
    })
}

/// Construct a failed tool execution result.
fn error_tool_result(tool_call: &ToolCall, text: String) -> ToolExecutionResult {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;
    ToolExecutionResult {
        message: ToolResultMessage {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            content: vec![ContentBlock::Text(TextContent {
                text,
                text_signature: None,
            })],
            details: None,
            usage: Usage::default(),
            is_error: true,
            added_tool_names: None,
            timestamp: now,
        },
        terminate: false,
    }
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
    let tool_call_id = prepared.tool_call.id.clone();
    let tool_name = prepared.tool.name().to_string();
    let args = prepared.tool_call.arguments.clone();

    // Queue for collecting partial results (relay from sync callback → async events)
    let update_events: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
    let accepting_updates = Arc::new(AtomicBool::new(true));

    let on_update: ToolUpdateCallback = {
        let update_events = update_events.clone();
        let accepting_updates = accepting_updates.clone();
        Box::new(move |partial| {
            if !accepting_updates.load(Ordering::SeqCst) {
                return;
            }
            if let Ok(partial_value) = serde_json::to_value(&partial) {
                update_events.lock().unwrap().push(partial_value);
            }
        })
    };

    // Execute the tool
    let result = match prepared
        .tool
        .execute(&tool_call_id, &prepared.args, cancel, on_update)
        .await
    {
        Ok(r) => {
            accepting_updates.store(false, Ordering::SeqCst);
            // Flush update events in the queue
            let events = std::mem::take(&mut *update_events.lock().unwrap());
            for partial in events {
                hooks
                    .on_event(&AgentEvent::ToolExecutionUpdate {
                        tool_call_id: tool_call_id.clone(),
                        tool_name: tool_name.clone(),
                        args: args.clone(),
                        partial_result: partial,
                    })
                    .await;
            }
            ExecutedToolCallOutcome {
                result: r,
                is_error: false,
            }
        }
        Err(err) => {
            accepting_updates.store(false, Ordering::SeqCst);
            // Flush remaining update events
            let events = std::mem::take(&mut *update_events.lock().unwrap());
            for partial in events {
                hooks
                    .on_event(&AgentEvent::ToolExecutionUpdate {
                        tool_call_id: tool_call_id.clone(),
                        tool_name: tool_name.clone(),
                        args: args.clone(),
                        partial_result: partial,
                    })
                    .await;
            }
            ExecutedToolCallOutcome {
                result: ToolExecutionResult {
                    message: ToolResultMessage {
                        tool_call_id: tool_call_id.clone(),
                        tool_name: tool_name.clone(),
                        content: vec![ContentBlock::Text(TextContent {
                            text: err.to_string(),
                            text_signature: None,
                        })],
                        details: None,
                        usage: Usage::default(),
                        is_error: true,
                        added_tool_names: None,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as i64,
                    },
                    terminate: false,
                },
                is_error: true,
            }
        }
    };

    result
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
    // Extract result (message content + terminate control signal) and is_error flag from the execution result
    let mut result = executed.result.clone();
    let mut is_error = executed.is_error;

    // Build hook context
    let after_ctx = AfterToolCallContext {
        assistant_message: assistant_message.clone(),
        tool_call: prepared.tool_call.clone(),
        args: prepared.args.clone(),
        result: result.clone(),
        is_error,
        context: context.clone(),
    };

    // Call hook
    match hooks.after_tool_call(&after_ctx, cancel).await {
        // Hook returned overrides
        Ok(Some(overrides)) => {
            if let Some(content) = overrides.content {
                result.message.content = content;
            }
            if let Some(details) = overrides.details {
                result.message.details = Some(details);
            }
            if let Some(usage) = overrides.usage {
                result.message.usage = usage;
            }
            if let Some(terminate) = overrides.terminate {
                result.terminate = terminate;
            }
            if let Some(err) = overrides.is_error {
                is_error = err;
            }
        }
        // Hook did nothing
        Ok(None) => {}
        // Hook execution failed
        Err(error_msg) => {
            result.message.content = vec![ContentBlock::Text(TextContent {
                text: error_msg,
                text_signature: None,
            })];
            result.message.details = None;
            is_error = true;
        }
    }

    FinalizedToolCallOutcome {
        tool_call: prepared.tool_call.clone(),
        result,
        is_error,
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use zen_ai::tool::ToolError;
    use zen_ai::types::StopReason;

    // ─── MockTool ────────────────────────────────────────────────────────

    struct MockTool {
        name: String,
        should_fail: bool,
        fail_prepare: bool,
        cancel_on_execute: bool,
        params: serde_json::Value,
    }

    impl std::fmt::Debug for MockTool {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockTool").field("name", &self.name).finish()
        }
    }

    impl MockTool {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                should_fail: false,
                fail_prepare: false,
                cancel_on_execute: false,
                params: serde_json::Value::Object(serde_json::Map::new()),
            }
        }

        fn with_fail(mut self, fail: bool) -> Self {
            self.should_fail = fail;
            self
        }

        fn with_fail_prepare(mut self, fail: bool) -> Self {
            self.fail_prepare = fail;
            self
        }
    }

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "mock tool"
        }
        fn parameters(&self) -> &serde_json::Value {
            &self.params
        }
        fn label(&self) -> &str {
            "Mock"
        }
        fn prepare_arguments(&self, args: &serde_json::Value) -> Result<serde_json::Value, ToolError> {
            if self.fail_prepare {
                Err(ToolError::InvalidArguments("bad args".into()))
            } else {
                Ok(args.clone())
            }
        }
        async fn execute(
            &self,
            tool_call_id: &str,
            _params: &serde_json::Value,
            _cancel: &CancellationToken,
            _on_update: ToolUpdateCallback,
        ) -> Result<ToolExecutionResult, ToolError> {
            if self.should_fail {
                return Err(ToolError::Execution("exec failed".into()));
            }
            Ok(ToolExecutionResult {
                message: ToolResultMessage {
                    tool_call_id: tool_call_id.to_string(),
                    tool_name: self.name.clone(),
                    content: vec![ContentBlock::Text(TextContent {
                        text: "ok".into(),
                        text_signature: None,
                    })],
                    details: None,
                    usage: Usage::default(),
                    is_error: false,
                    added_tool_names: None,
                    timestamp: 0,
                },
                terminate: false,
            })
        }
    }

    // ─── MockHooks ───────────────────────────────────────────────────────

    struct MockHooks {
        block_before: bool,
        block_reason: Option<String>,
        after_override: Option<AfterToolCallResult>,
        after_error: Option<String>,
        events: Arc<Mutex<Vec<AgentEvent>>>,
    }

    impl MockHooks {
        fn new() -> Self {
            Self {
                block_before: false,
                block_reason: None,
                after_override: None,
                after_error: None,
                events: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn with_block(mut self, reason: &str) -> Self {
            self.block_before = true;
            self.block_reason = Some(reason.to_string());
            self
        }

        fn collected_events(&self) -> Vec<AgentEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl AgentLoopHooks for MockHooks {
        async fn before_tool_call(
            &self,
            _context: &BeforeToolCallContext,
            _cancel: &CancellationToken,
        ) -> Result<Option<BeforeToolCallResult>, String> {
            if self.block_before {
                Ok(Some(BeforeToolCallResult {
                    block: true,
                    reason: self.block_reason.clone(),
                }))
            } else {
                Ok(None)
            }
        }

        async fn after_tool_call(
            &self,
            _context: &AfterToolCallContext,
            _cancel: &CancellationToken,
        ) -> Result<Option<AfterToolCallResult>, String> {
            if let Some(ref err) = self.after_error {
                return Err(err.clone());
            }
            Ok(self.after_override.clone())
        }

        async fn on_event(&self, event: &AgentEvent) {
            self.events.lock().unwrap().push(event.clone());
        }
    }

    // ─── test helpers ────────────────────────────────────────────────────

    fn block_text(block: &ContentBlock) -> &str {
        match block {
            ContentBlock::Text(t) => &t.text,
            _ => panic!("expected Text block"),
        }
    }

    fn dummy_tool_call(name: &str) -> ToolCall {
        ToolCall {
            id: "call-1".into(),
            name: name.into(),
            arguments: json!({"key": "value"}),
            thought_signature: None,
        }
    }

    fn dummy_assistant_message() -> AssistantMessage {
        AssistantMessage {
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
        }
    }

    fn dummy_context() -> Context {
        Context {
            system_prompt: None,
            messages: vec![],
            tools: None,
        }
    }

    // ─── AgentLoopError tests ────────────────────────────────────────────

    #[test]
    fn test_agent_loop_error_display() {
        assert_eq!(
            AgentLoopError::EmptyContext.to_string(),
            "Cannot continue: no messages in context"
        );
        assert_eq!(
            AgentLoopError::InvalidLastRole.to_string(),
            "Cannot continue from message role: assistant"
        );
    }

    #[test]
    fn test_agent_loop_error_debug() {
        let err = AgentLoopError::EmptyContext;
        assert!(format!("{:?}", err).contains("EmptyContext"));
    }

    #[test]
    fn test_agent_loop_error_clone() {
        let err = AgentLoopError::InvalidLastRole;
        let cloned = err.clone();
        assert_eq!(err.to_string(), cloned.to_string());
    }

    // ─── error_tool_result tests ─────────────────────────────────────────

    #[test]
    fn test_error_tool_result_contains_error_text() {
        let tc = dummy_tool_call("bash");
        let result = error_tool_result(&tc, "something went wrong".into());

        assert!(!result.terminate);
        assert!(result.message.is_error);
        assert_eq!(result.message.tool_call_id, "call-1");
        assert_eq!(result.message.tool_name, "bash");
        assert_eq!(result.message.content.len(), 1);
        assert_eq!(block_text(&result.message.content[0]), "something went wrong");
    }

    #[test]
    fn test_error_tool_result_terminate_is_false() {
        let tc = dummy_tool_call("read");
        let result = error_tool_result(&tc, "err".into());
        assert!(!result.terminate);
    }

    // ─── prepare_tool_call tests ─────────────────────────────────────────

    #[tokio::test]
    async fn test_prepare_tool_call_success() {
        let tool: Arc<dyn Tool> = Arc::new(MockTool::new("bash"));
        let tools = [tool];
        let hooks = MockHooks::new();
        let cancel = CancellationToken::new();

        let result = prepare_tool_call(
            &dummy_context(),
            &dummy_assistant_message(),
            &dummy_tool_call("bash"),
            &tools,
            &hooks,
            &cancel,
        )
        .await;

        assert!(result.is_ok());
        let prepared = result.unwrap();
        assert_eq!(prepared.tool_call.name, "bash");
        assert_eq!(prepared.tool.name(), "bash");
        assert_eq!(prepared.args, json!({"key": "value"}));
    }

    #[tokio::test]
    async fn test_prepare_tool_call_tool_not_found() {
        let tools: [Arc<dyn Tool>; 0] = [];
        let hooks = MockHooks::new();
        let cancel = CancellationToken::new();

        let result = prepare_tool_call(
            &dummy_context(),
            &dummy_assistant_message(),
            &dummy_tool_call("nonexistent"),
            &tools,
            &hooks,
            &cancel,
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.is_error);
        assert!(block_text(&err.message.content[0]).contains("not found"));
    }

    #[tokio::test]
    async fn test_prepare_tool_call_invalid_arguments() {
        let tool: Arc<dyn Tool> = Arc::new(MockTool::new("bash").with_fail_prepare(true));
        let tools = [tool];
        let hooks = MockHooks::new();
        let cancel = CancellationToken::new();

        let result = prepare_tool_call(
            &dummy_context(),
            &dummy_assistant_message(),
            &dummy_tool_call("bash"),
            &tools,
            &hooks,
            &cancel,
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(block_text(&err.message.content[0]).contains("Invalid arguments"));
    }

    #[tokio::test]
    async fn test_prepare_tool_call_hook_blocks() {
        let tool: Arc<dyn Tool> = Arc::new(MockTool::new("bash"));
        let tools = [tool];
        let hooks = MockHooks::new().with_block("user said no");
        let cancel = CancellationToken::new();

        let result = prepare_tool_call(
            &dummy_context(),
            &dummy_assistant_message(),
            &dummy_tool_call("bash"),
            &tools,
            &hooks,
            &cancel,
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(block_text(&err.message.content[0]).contains("user said no"));
    }

    #[tokio::test]
    async fn test_prepare_tool_call_hook_blocks_without_reason() {
        struct BlockNoReason;
        #[async_trait]
        impl AgentLoopHooks for BlockNoReason {
            async fn before_tool_call(
                &self,
                _: &BeforeToolCallContext,
                _: &CancellationToken,
            ) -> Result<Option<BeforeToolCallResult>, String> {
                Ok(Some(BeforeToolCallResult {
                    block: true,
                    reason: None,
                }))
            }
        }

        let tool: Arc<dyn Tool> = Arc::new(MockTool::new("bash"));
        let tools = [tool];
        let cancel = CancellationToken::new();

        let result = prepare_tool_call(
            &dummy_context(),
            &dummy_assistant_message(),
            &dummy_tool_call("bash"),
            &tools,
            &BlockNoReason,
            &cancel,
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(block_text(&err.message.content[0]).contains("was blocked"));
    }

    #[tokio::test]
    async fn test_prepare_tool_call_cancelled_before_hook() {
        let tool: Arc<dyn Tool> = Arc::new(MockTool::new("bash"));
        let tools = [tool];
        let hooks = MockHooks::new();
        let cancel = CancellationToken::new();
        cancel.cancel();

        let result = prepare_tool_call(
            &dummy_context(),
            &dummy_assistant_message(),
            &dummy_tool_call("bash"),
            &tools,
            &hooks,
            &cancel,
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(block_text(&err.message.content[0]).contains("aborted"));
    }

    // ─── fail_tool_calls_from_truncated_message tests ────────────────────

    #[tokio::test]
    async fn test_fail_truncated_empty() {
        let hooks = MockHooks::new();
        let (msgs, terminate) = fail_tool_calls_from_truncated_message(&[], &hooks).await;

        assert!(msgs.is_empty());
        assert!(!terminate);
    }

    #[tokio::test]
    async fn test_fail_truncated_single_tool() {
        let hooks = MockHooks::new();
        let calls = [dummy_tool_call("bash")];
        let (msgs, terminate) = fail_tool_calls_from_truncated_message(&calls, &hooks).await;

        assert_eq!(msgs.len(), 1);
        assert!(!terminate);
        assert!(msgs[0].is_error);
        assert!(msgs[0].tool_name == "bash");
        assert!(block_text(&msgs[0].content[0]).contains("output token limit"));

        // Verify events were emitted
        let events = hooks.collected_events();
        assert_eq!(events.len(), 2); // ToolExecutionStart + ToolExecutionEnd
        assert!(matches!(events[0], AgentEvent::ToolExecutionStart { .. }));
        assert!(matches!(events[1], AgentEvent::ToolExecutionEnd { .. }));
    }

    #[tokio::test]
    async fn test_fail_truncated_multiple_tools() {
        let hooks = MockHooks::new();
        let calls = [dummy_tool_call("bash"), dummy_tool_call("read"), dummy_tool_call("write")];
        let (msgs, terminate) = fail_tool_calls_from_truncated_message(&calls, &hooks).await;

        assert_eq!(msgs.len(), 3);
        assert!(!terminate);
        for msg in &msgs {
            assert!(msg.is_error);
        }

        let events = hooks.collected_events();
        assert_eq!(events.len(), 6); // 3x (start + end)
    }
}
