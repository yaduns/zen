use serde::Serialize;
use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio_util::sync::CancellationToken;
use zen_ai::api::openai_stream::{StreamConfig, StreamError};
use zen_ai::tool::{Tool, ToolExecutionMode, ToolExecutionResult};
use zen_ai::types::{
    AssistantMessage, AssistantMessageEvent, Context, Message, Model, ThinkingLevel, ToolCall,
    ToolResultMessage, Usage,
};

// REQ-AGENT-01 (pi-232, pi-205): Agent core types

/// StreamFn is the signature of the LLM call function used by the agent loop.
/// - Must return an event stream (`UnboundedReceiver<AssistantMessageEvent>`).
/// - Failures must be encoded via protocol events in the stream (`AssistantMessageEvent::Error`) and the final
///   `AssistantMessage` (stop_reason = "error" / "aborted").
/// - The caller (agent loop) controls cancellation via the `cancel` parameter; on cancellation the stream should close as soon as possible.
///
/// StreamFn is passed explicitly via the `run_agent_loop` parameter; no global static is kept.
/// Can be migrated to a `Provider` trait when multiple providers are supported.
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
    pub is_streaming: bool,
    /// Current assistant message under construction, not yet finished
    pub streaming_message: Option<Message>,
    /// Set of tool call IDs currently executing
    pub pending_tool_calls: HashSet<String>,
    /// Error message from the most recently failed or interrupted assistant turn
    pub error_message: Option<String>,
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
    /// max_tokens parameter passed to the LLM
    pub max_tokens: Option<u32>,
    /// temperature parameter passed to the LLM
    pub temperature: Option<f32>,
}
