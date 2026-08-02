use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio_util::sync::CancellationToken;
use zen_ai::api::openai_stream::{StreamConfig, StreamError};
use zen_ai::types::{
    AssistantMessage, AssistantMessageEvent, ContentBlock, ContentBlockList, Context, Message,
    Model, ThinkingLevel, ToolCall, ToolResultMessage, Usage,
};
/// StreamFn is the signature of the LLM call function used by the agent loop.
/// - Must return an event stream (`UnboundedReceiver<AssistantMessageEvent>`).
/// - Failures must be encoded via protocol events in the stream (`AssistantMessageEvent::Error`) and the final
///   `AssistantMessage` (stop_reason = "error" / "aborted").
/// - The caller (agent loop) controls cancellation via the `cancel` parameter; on cancellation the stream should close as soon as possible.
///
/// `Arc<dyn Fn>` is used because `setDefaultStreamFn` needs to store the function as a module-level variable.
pub type StreamFn = Arc<
    dyn Fn(
            &Model,
            &Context,
            Option<StreamConfig>,
            Option<CancellationToken>,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<UnboundedReceiver<AssistantMessageEvent>, StreamError>>
                    + Send,
            >,
        > + Send
        + Sync,
>;

/// Controls the execution order of multiple tool calls within a single assistant message.
#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolExecutionMode {
    Sequential,
    Parallel,
}

/// Controls how many queued user messages the agent loop injects at queue emission points
#[derive(Serialize)]
pub enum QueueMode {
    /// Inject all queued messages
    #[serde(rename = "all")]
    All,
    /// Inject only the oldest message each time
    #[serde(rename = "one-at-a-time")]
    OneAtATime,
}

/// Union type for agent transcript messages. Standard LLM messages directly wrap zen-ai's Message; custom messages
/// #[serde(untagged)] tells serde not to add an outer tag and to try each variant's inner type in order, using the first that parses successfully
#[derive(Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum AgentMessage {
    /// Standard LLM message
    Llm(Message),
    /// Application custom message
    Custom(CustomAgentMessage),
    /// Matches JSON objects with a command field
    BashExecution(BashExecutionMessage),
    /// Matches JSON objects with a from_id field
    BranchSummary(BranchSummaryMessage),
    /// Matches JSON objects with a tokens_before field
    CompactionSummary(CompactionSummaryMessage),
}

/// Payload for a custom agent message, embedded in the transcript via the AgentMessage::Custom variant.
/// display controls whether to render in the UI; custom_type lets the UI component select a rendering method by type.
#[derive(Serialize, Deserialize, Clone)]
pub struct CustomAgentMessage {
    /// Custom message type identifier
    pub custom_type: String,
    /// Custom message content (text or content blocks)
    pub content: ContentBlockList,
    /// Structured metadata for UI rendering
    pub details: Option<serde_json::Value>,
    /// Whether to display in the UI
    pub display: bool,
    /// Unix timestamp (milliseconds)
    pub timestamp: i64,
}

/// Tool execution result, including content returned to the LLM, structured metadata, resource usage, termination notices, etc.
#[derive(Serialize)]
pub struct AgentToolResult {
    /// Content returned to the LLM (only Text and Image blocks; Thinking and ToolCall must not appear; callers must ensure this constraint)
    content: Vec<ContentBlock>,
    /// Structured metadata for logging or UI rendering
    details: serde_json::Value,
    /// Resource usage of the tool execution itself, independent of the LLM call's Usage
    usage: Usage,
    /// List of new tool names introduced by this result
    added_tool_names: Option<Vec<String>>,
    /// Indicates whether the agent should terminate early after the current tool batch
    terminate: bool,
}

/// Callback type used to stream partial results during tool execution.
/// The callback is valid only during the current `execute()` call—calls after the promise resolves are ignored.
pub type AgentToolUpdateCallback = Box<dyn Fn(AgentToolResult) + Send + Sync>;

/// Context snapshot passed to the agent loop, including the system prompt, transcript messages, and available tools.
/// Context type received by hook functions (beforeToolCall, afterToolCall, etc.). A runtime-only structure that does not derive serde.
pub struct AgentContext {
    system_prompt: String,
    /// Transcript messages
    messages: Vec<AgentMessage>,
    /// List of tools available in the current turn
    tools: Vec<Arc<dyn AgentTool>>,
}

/// Public runtime snapshot of the agent, including the current configuration, transcript messages, streaming state, and error information. A runtime-only structure that does not derive serde
pub struct AgentState {
    system_prompt: String,
    model: Model,
    thinking_level: ThinkingLevel,
    /// Available tools
    tools: Vec<Arc<dyn AgentTool>>,
    /// Completed conversation history (all messages are finalized)
    messages: Vec<AgentMessage>,
    is_streaming: bool,
    /// Current assistant message under construction, not yet finished
    streaming_message: Option<AgentMessage>,
    /// Set of tool call IDs currently executing
    pending_tool_calls: HashSet<String>,
    /// Error message from the most recently failed or interrupted assistant turn
    error_message: Option<String>,
}

/// Events emitted by the agent during execution, for UI updates and logging. Includes events for the agent, turn, message, and tool execution lifecycle stages.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// Agent starts processing prompt
    AgentStart,
    /// Agent processing complete
    AgentEnd { messages: Vec<AgentMessage> },
    /// New turn starts
    TurnStart,
    /// Turn ends (including tool results)
    TurnEnd {
        message: AgentMessage,
        tool_results: Vec<ToolResultMessage>,
    },
    /// New message starts
    MessageStart { message: AgentMessage },
    /// Streaming message update
    MessageUpdate {
        message: AgentMessage,
        assistant_message_event: AssistantMessageEvent,
    },
    /// Message completed
    MessageEnd { message: AgentMessage },
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
        result: AgentToolResult,
        is_error: bool,
    },
}

// --------------Hook data model-----------

/// Return value of the beforeToolCall hook; block = true prevents tool execution, and the loop will inject an error tool result.
pub struct BeforeToolCallResult {
    block: bool,
    reason: Option<String>,
}

/// Return value of the afterToolCall hook, used to override certain fields of the executed tool result. Field-level merge semantics (not a deep merge)
pub struct AfterToolCallResult {
    /// Replace the entire content array
    content: Option<Vec<ContentBlock>>,
    /// Replace the entire details value
    details: Option<serde_json::Value>,
    /// Replace error flag
    is_error: Option<bool>,
    /// Replace tool execution usage
    usage: Option<Usage>,
    /// Replace early-termination notice
    terminate: Option<bool>,
}

/// Context parameters for the beforeToolCall hook
pub struct BeforeToolCallContext {
    /// Assistant message that initiates a tool call
    assistant_message: AssistantMessage,
    /// Raw tool call block
    tool_call: ToolCall,
    /// Parameters validated against the target tool's schema
    args: serde_json::Value,
    /// Agent context snapshot when executing a tool
    context: AgentContext,
}

/// Context parameters for the afterToolCall hook
pub struct AfterToolCallContext {
    /// Assistant message that initiates a tool call
    assistant_message: AssistantMessage,
    /// Raw tool call block
    tool_call: ToolCall,
    /// Validated arguments
    args: serde_json::Value,
    /// Tool result after execution (hook overrides not applied)
    result: AgentToolResult,
    /// Whether the current result is considered an error
    is_error: bool,
    /// Agent context snapshot when executing a tool
    context: AgentContext,
}

/// Context parameters for the shouldStopAfterTurn hook; determines whether the agent loop should stop after the current turn.
pub struct ShouldStopAfterTurnContext {
    /// Assistant message completing the current turn
    message: AssistantMessage,
    /// Tool result messages passed to the turn_end event
    tool_results: Vec<ToolResultMessage>,
    /// Agent context after appending the assistant message and tool results
    context: AgentContext,
    /// Message to return if this loop exits
    new_messages: Vec<AgentMessage>,
}

/// Type alias for the `prepareNextTurn` hook context, identical to `ShouldStopAfterTurnContext`
pub type PrepareNextTurnContext = ShouldStopAfterTurnContext;

/// Return value of the prepareNextTurn hook, replacing the context/model/thinking state for the next turn. All fields are Option; omitted fields retain their current values.
pub struct AgentLoopTurnUpdate {
    /// Replace the context for the next turn
    context: Option<AgentContext>,
    /// Replace the model for the next turn
    model: Option<Model>,
    /// Replaces the thinking level for the next turn (referencing the zen-ai type).
    thinking_level: Option<ThinkingLevel>,
}

/// A tool definition registered and executed by the agent runtime.
/// - The parameter Schema is returned by `parameters()` as `&serde_json::Value` (JSON Schema format).
/// - Argument validation is performed by `prepare_arguments()`, taking the raw JSON returned by the LLM as input and producing
///   valid JSON after type coercion (`coercePrimitiveByType` logic).
/// - The execution result is returned by `execute()` as `AgentToolResult`, with the `details` field typed as `serde_json::Value`.
/// Contract (inheriting the original TS constraints):
/// - If `execute()` fails, it should throw (returning `Err` in Rust), rather than encoding the error in `content`.
/// - Continuing to call the `on_update` callback after `execute()` returns has no effect.
/// - `prepare_arguments()` is optional; the default implementation returns the input directly.
#[async_trait]
pub trait AgentTool: Send + Sync {
    /// Tool name, used for LLM tool selection and log identification.
    fn name(&self) -> &str;

    /// Tool description, visible to the LLM, used to decide whether to call this tool.
    fn description(&self) -> &str;

    /// Tool parameter JSON Schema, used by the LLM to generate valid arguments.
    fn parameters(&self) -> &serde_json::Value;

    /// Human-readable tool label for UI display.
    fn label(&self) -> &str;

    /// Optional argument preprocessing hook that type-coerces raw arguments before schema validation.
    /// Returns the transformed argument object (must match the Schema defined by `parameters`).
    /// Default implementation returns the input as-is.
    fn prepare_arguments(
        &self,
        args: &serde_json::Value,
    ) -> Result<serde_json::Value, AgentToolError> {
        Ok(args.clone())
    }

    /// Executes the tool call.
    ///
    /// # Parameters
    /// - `tool_call_id`: tool call ID, used to match the tool result message.
    /// - `params`: Arguments after processing by `prepare_arguments`.
    /// - `cancel`: cancellation token; the tool should stop as soon as possible when a cancellation signal is received.
    /// - `on_update`: Optional callback for streaming partial execution results.
    ///
    /// # Errors
    /// Returning `Err(AgentToolError)` indicates tool execution failure. The error message is encoded via `AgentToolError`,
    /// The caller converts it to the error content in `AgentToolResult`.
    async fn execute(
        &self,
        tool_call_id: &str,
        params: &serde_json::Value,
        cancel: &CancellationToken,
        on_update: AgentToolUpdateCallback,
    ) -> Result<AgentToolResult, AgentToolError>;
}

/// Tool execution error type.
#[derive(Debug, thiserror::Error)]
pub enum AgentToolError {
    /// Tool execution failed
    #[error("Tool execution failed: {0}")]
    Execution(String),
    /// Argument validation failed
    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),
    /// User cancelled
    #[error("Tool execution cancelled")]
    Cancelled,
}

/// Record of a bash command execution result, stored in the agent transcript.
/// When exclude_from_context is true, convertToLlm skips this message (it is not sent to the LLM context).
/// full_output_path provides the full output file path when output is truncated.
#[derive(Serialize, Deserialize, Clone)]
pub struct BashExecutionMessage {
    /// The bash command executed
    pub command: String,
    /// Command output text (may be an empty string)
    pub output: String,
    /// Process exit code; None means not captured (e.g., when the command is canceled)
    pub exit_code: Option<i32>,
    /// Whether the command was canceled (user interrupt)
    pub cancelled: bool,
    /// Whether output was truncated (exceeded length limit)
    pub truncated: bool,
    /// Writes the full output to a temporary file and stores the path in fullOutputPath
    pub full_output_path: Option<String>,
    /// Unix timestamp (milliseconds)
    pub timestamp: i64,
    /// Whether to exclude from the LLM context (default false)
    pub exclude_from_context: bool,
}

/// Branch summary injected when rolling back from a fork branch, describing the previous branch contents. from_id is the session entry ID of the summarized branch.
#[derive(Serialize, Deserialize, Clone)]
pub struct BranchSummaryMessage {
    /// Branch content summary text
    pub summary: String,
    /// Session entry ID of the summarized branch (UUIDv7)
    pub from_id: String,
    /// Unix timestamp (milliseconds)
    pub timestamp: i64,
}

/// Record of a context compaction result. When context exceeds the token threshold, old messages are compacted into a summary, and the original token count is recorded in tokens_before for debugging and statistics.
#[derive(Serialize, Deserialize, Clone)]
pub struct CompactionSummaryMessage {
    /// Summary text generated by compaction
    pub summary: String,
    /// Token count of the original messages replaced before compaction
    pub tokens_before: u64,
    /// Unix timestamp (milliseconds)
    pub timestamp: i64,
}
