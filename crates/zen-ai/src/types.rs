use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::tool::Tool;

/// Reason the LLM response terminated, used to decide the agent loop's next action
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StopReason {
    /// LLM finished normally, no tool call requested
    Stop,
    /// LLM truncated by token limit (max_tokens)
    Length,
    /// LLM requested a tool call (agent loop must continue)
    ToolUse,
    /// Request failed, provider returned an error, retries exhausted
    Error,
    /// User cancellation (CancellationToken triggered) or external interruption
    Aborted,
}

/// Token cost breakdown of the LLM request, priced per input/output/cacheRead/cacheWrite dimension.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Cost {
    /// Input token cost ($/million tokens)
    pub input: f64,
    /// Output token cost
    pub output: f64,
    /// Cache hit read cost
    pub cache_read: f64,
    /// Cache write cost
    pub cache_write: f64,
    /// Total = input + output + cache_read + cache_write
    pub total: f64,
}

/// Token usage stats for a single LLM request, including token counts per dimension and cost
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    /// Number of input tokens
    pub input: u64,
    /// Number of output tokens
    pub output: u64,
    /// Number of tokens read from cache hits
    pub cache_read: u64,
    /// Number of tokens written to cache
    pub cache_write: u64,
    /// cache_write subset with 1h retention (only Anthropic reports this split; zen keeps the field but it may be None)
    pub cache_write_1h: Option<u64>,
    /// Number of reasoning/thinking tokens; a subset of output (output already includes this value)
    pub reasoning: Option<u64>,
    /// Total token count = input + outpu
    pub total_tokens: u64,
    /// Cost breakdown
    pub cost: Cost,
}

/// Plain text content block, used for text fragments in the message content array
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextContent {
    /// Text content
    pub text: String,
    /// Text signature ID (message metadata from the OpenAI responses API; field retained by zen)
    pub text_signature: Option<String>,
}

/// LLM reasoning/thinking content block containing the thinking text and its signature, used to pass thinking context across multi-turn conversations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingContent {
    /// Thinking process text
    pub thinking: String,
    /// Thinking signature (passed back to the API in multi-turn conversations)
    pub thinking_signature: Option<String>,
    /// Whether the content was masked by a safety filter (when true, the thinking field is placeholder text and the actual content is in signature)
    pub redacted: Option<bool>,
}

/// Image content block, supporting multimodal input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    /// Base64-encoded image data
    pub data: String,
    /// MIME type, e.g. "image/jpeg", "image/png"
    pub mime_type: String,
}

/// Union type of the message content array, comprising text, thinking, image, and tool-call content blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ContentBlock {
    /// All message types
    Text(TextContent),
    /// AssistantMessage only
    Thinking(ThinkingContent),
    /// UserMessage、ToolResultMessage
    Image(ImageContent),
    /// AssistantMessage only (tool call request)
    ToolCall(ToolCall),
}

/// Description of the tool to be invoked by the LLM request, including tool name, arguments, and current chunk ID.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    /// Tool call ID, used to match the tool result
    pub id: String,
    /// Tool name
    pub name: String,
    /// Tool arguments, a JSON object; parsed from the stringified JSON returned by the LLM
    pub arguments: serde_json::Value,
    /// Google-specific thinking signature, preserved field in zen
    pub thought_signature: Option<String>,
}

/// User message containing text or image content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserMessage {
    /// User content, either a plain text string or an array of content blocks
    pub content: ContentBlockList,
    /// Unix timestamp (milliseconds)
    pub timestamp: i64,
}

/// The content field of a user message can be a simple string or an array of content blocks.
/// serde uses the externally tagged format by default:
/// Text("Hello".to_string()) → {"Text": "Hello"}
/// Blocks(vec![text_block, image_block]) → {"Blocks": [...]}
/// The protocol expects:
/// Plain text → the string itself "Hello"            ✅
/// Rich text → the array itself [{"type": "text", "text": "Hello"}, {"type": "image",...}]  ✅
/// With #[serde(untagged)] serde no longer adds an outer tag, instead it tries matching in variant order
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
#[serde(rename_all = "camelCase")]
pub enum ContentBlockList {
    /// Simple text message
    Text(String),
    /// Rich text message (includes images, etc.)
    Blocks(Vec<ContentBlock>),
}

/// Response message generated by the LLM, containing text/thinking/tool-call content blocks, metadata, usage stats, and stop reason.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessage {
    /// Response content blocks (text, thinking, tool call)
    pub content: Vec<ContentBlock>,
    /// API type used (simplified under single provider, e.g. "openai-completions")
    pub api: String,
    /// Provider ID (e.g. "deepseek")
    pub provider: String,
    /// Model name used for the request
    pub model: String,
    /// Actual response model (when different from the requested model, e.g. OpenRouter auto-routing)
    pub response_model: Option<String>,
    /// Response/message identifier returned by the provider
    pub response_id: Option<String>,
    /// Token usage statistics
    pub usage: Usage,
    /// Termination reason
    pub stop_reason: StopReason,
    /// Description when an error occurs
    pub error_message: Option<String>,
    /// Unix timestamp (milliseconds)
    pub timestamp: i64,
}

/// Tool execution result message returned to the LLM, containing the tool call's output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultMessage {
    /// Corresponding tool call ID
    pub tool_call_id: String,
    /// Tool name
    pub tool_name: String,
    /// Tool execution result content (text and images)
    pub content: Vec<ContentBlock>,
    /// Tool execution metadata; concrete structure determined by the implementation (e.g. bash exit code + duration, read byte count, edit diff summary)
    pub details: Option<serde_json::Value>,
    /// Resource consumption of the tool itself, independent of the LLM call's Usage
    pub usage: Usage,
    /// Whether the tool execution failed
    pub is_error: bool,
    /// List of new tool names introduced by this tool result (for the tools parameter of subsequent requests)
    pub added_tool_names: Option<Vec<String>>,
    /// Unix timestamp (milliseconds)
    pub timestamp: i64,
}

/// Union type of session messages, comprising user message, assistant message, and tool result message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum Message {
    /// User input message
    User(UserMessage),
    /// LLM response message
    Assistant(AssistantMessage),
    /// Tool execution result message
    ToolResult(ToolResultMessage),
}

/// Event stream protocol defining the event types of AssistantMessageEventStream. A `start` event is emitted at the beginning of the stream, incremental updates are sent in between (text_start/text_delta/text_end etc.), and it terminates with `done` (success) or `error` (failure).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantMessageEvent {
    /// Stream started, first partial message sent
    Start { partial: AssistantMessage },
    /// New text block started
    #[serde(rename_all = "camelCase")]
    TextStart {
        content_index: u32,
        partial: AssistantMessage,
    },
    /// Incremental text block content
    #[serde(rename_all = "camelCase")]
    TextDelta {
        content_index: u32,
        delta: String,
        partial: AssistantMessage,
    },
    /// Text block output complete
    #[serde(rename_all = "camelCase")]
    TextEnd {
        content_index: u32,
        content: String,
        partial: AssistantMessage,
    },
    /// New thinking block starts output
    #[serde(rename_all = "camelCase")]
    ThinkingStart {
        content_index: u32,
        partial: AssistantMessage,
    },
    /// Thinking block incremental content
    #[serde(rename_all = "camelCase")]
    ThinkingDelta {
        content_index: u32,
        delta: String,
        partial: AssistantMessage,
    },
    /// Thinking block output complete
    #[serde(rename_all = "camelCase")]
    ThinkingEnd {
        content_index: u32,
        content: String,
        partial: AssistantMessage,
    },
    /// New tool call block starts output
    #[serde(rename_all = "camelCase")]
    ToolcallStart {
        content_index: u32,
        partial: AssistantMessage,
    },
    /// Tool call incremental content
    #[serde(rename_all = "camelCase")]
    ToolcallDelta {
        content_index: u32,
        delta: String,
        partial: AssistantMessage,
    },
    /// Tool call block output complete, carrying the full ToolCall
    #[serde(rename_all = "camelCase")]
    ToolcallEnd {
        content_index: u32,
        tool_call: ToolCall,
        partial: AssistantMessage,
    },
    /// Stream ends normally; reason is Stop/Length/ToolUse
    Done {
        reason: NonErrorStopReason,
        message: AssistantMessage,
    },
    /// Stream terminates abnormally; reason is Aborted/Error
    Error {
        reason: ErrorStopReason,
        error: AssistantMessage,
    },
}

/// Type of the reason field carried in AssistantMessageEvent::Done, restricted to non-error termination reasons.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum NonErrorStopReason {
    Stop,
    Length,
    ToolUse,
}

/// Type of the reason field carried in AssistantMessageEvent::Error, restricted to error termination reasons.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ErrorStopReason {
    Aborted,
    Error,
}

/// Full context for an LLM call: system prompt, message history, and executable tools.
///
/// `tools` holds the **executable tools** (single source of truth, aligned with Pi's `AgentContext.tools`):
/// - Execution side (agent loop) looks up and runs tools by name from `context.tools`;
/// - Tool schemas sent to the LLM (`ToolDefinition`) are derived at the request boundary (`openai_stream`)
///   via `to_opai_definition()`; not pre-serialized here.
///
/// Not `Serialize`/`Deserialize`: Context is an in-memory structure (`Arc<dyn Tool>` is not serializable);
/// serialization happens at request construction — consistent with Pi's `AgentContext`.
#[derive(Debug, Clone)]
pub struct Context {
    /// System prompt
    pub system_prompt: Option<String>,
    /// Message history
    pub messages: Vec<Message>,
    /// List of executable tools (single source of truth)
    pub tools: Option<Vec<Arc<dyn Tool>>>,
}

/// Tool definition, conforming to the tool object format of the OpenAI Chat Completions API.
///
/// Serialized form:
/// ```json
/// {
///   "type": "function",
///   "function": {
///     "name": "get_weather",
///     "description": "...",
///     "parameters": { ... }
///   }
/// }
/// ```
///
/// Runtime behavior is provided by the `crate::tool::Tool` trait.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool type, currently only `"function"` is supported
    #[serde(rename = "type")]
    pub type_: String,
    /// Tool function definition (name, description, parameter schema)
    pub function: ToolFunction,
}

/// Tool function definition, describing the tool's name, description, and parameter JSON Schema.
///
/// Mirrors the `tool.function` object in the OpenAI Chat Completions API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolFunction {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// JSON Schema for tool parameters.
    pub parameters: serde_json::Value,
}

/// Describes an AI model's capabilities and configuration; the source of configuration for all LLM calls in zen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    /// Model identifier, sent to the API as the model field (e.g. "deepseek-v4-flash")
    pub id: String,
    /// Human-readable name (e.g. "DeepSeek V4 Flash")
    pub name: Option<String>,
    /// API endpoint base URL (e.g. "https://api.deepseek.com"), without the /chat/completions path
    pub base_url: String,
    /// Maximum context window (token count), used for overflow detection
    pub context_window: u64,
    /// Maximum output token count
    pub max_tokens: u64,
    /// Whether thinking/reasoning (thinking mode) is supported
    pub reasoning: bool,
    /// Extra HTTP request headers, merged into the API request
    pub headers: Option<HashMap<String, String>>,
    /// Token rates, used for cost calculation. When None, no cost is calculated
    pub cost_rates: Option<CostRates>,
}

/// Describes per-million-token rates for a model, used to compute actual cost from token usage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CostRates {
    /// Input token rate ($/million tokens)
    pub input: f64,
    /// Output token rate
    pub output: f64,
    /// Cache-hit read rate
    pub cache_read: f64,
    /// Cache-write rate
    pub cache_write: f64,
}

/// LLM inference/thinking intensity level, mapping to the OpenAI API's `reasoning_effort` parameter.
/// `Off` means the `reasoning_effort` field is not sent (the model uses its default behavior).
/// The remaining variants map directly to the API's string values.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    /// No thinking enabled (reasoning_effort field not sent)
    Off,
    /// Minimal thinking time, suitable for simple tasks
    Minimal,
    /// Lower thinking intensity
    Low,
    /// Default medium thinking effort
    Medium,
    /// Higher thinking effort
    High,
    /// Even higher thinking effort
    XHigh,
    /// Maximum thinking effort
    Max,
}

/// Controls whether the LLM calls a tool and which tool, corresponding to the OpenAI API's tool_choice parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ToolChoice {
    /// "auto"	default behavior; let the model decide whether to call a tool
    Auto,
    /// "none"	forbid the model from calling tools
    None,
    /// "required"	force the model to call some tool (the model may choose itself)
    Required,
    /// {"type": "function", "function": {"name": "..."}}	force the model to call the tool with the specified name
    Function { name: String },
}
