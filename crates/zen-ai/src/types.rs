use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Cost {
    /// Input token cost ($/million tokens)
    input: f64,
    /// Output token cost
    output: f64,
    /// Cache hit read cost
    cache_read: f64,
    /// Cache write cost
    cache_write: f64,
    /// Total = input + output + cache_read + cache_write
    total: f64,
}

/// Token usage stats for a single LLM request, including token counts per dimension and cost
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    /// Number of input tokens
    input: u64,
    /// Number of output tokens
    output: u64,
    /// Number of tokens read from cache hits
    cache_read: u64,
    /// Number of tokens written to cache
    cache_write: u64,
    /// cache_write subset with 1h retention (only Anthropic reports this split; zen keeps the field but it may be None)
    cache_write_1h: Option<u64>,
    /// Number of reasoning/thinking tokens; a subset of output (output already includes this value)
    reasoning: Option<u64>,
    /// Total token count = input + outpu
    total_tokens: u64,
    /// Cost breakdown
    cost: Cost,
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
    content: ContentBlockList,
    /// Unix timestamp (milliseconds)
    timestamp: i64,
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
    content: Vec<ContentBlock>,
    /// API type used (simplified under single provider, e.g. "openai-completions")
    api: String,
    /// Provider ID (e.g. "deepseek")
    provider: String,
    /// Model name used for the request
    model: String,
    /// Actual response model (when different from the requested model, e.g. OpenRouter auto-routing)
    response_model: Option<String>,
    /// Response/message identifier returned by the provider
    response_id: Option<String>,
    /// Token usage statistics
    usage: Usage,
    /// Termination reason
    stop_reason: StopReason,
    /// Description when an error occurs
    error_message: Option<String>,
    /// Unix timestamp (milliseconds)
    timestamp: i64,
}

/// Tool execution result message returned to the LLM, containing the tool call's output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultMessage {
    /// Corresponding tool call ID
    tool_call_id: String,
    /// Tool name
    tool_name: String,
    /// Tool execution result content (text and images)
    content: Vec<ContentBlock>,
    /// Tool execution metadata; concrete structure determined by the implementation (e.g. bash exit code + duration, read byte count, edit diff summary)
    details: Option<serde_json::Value>,
    /// Resource consumption of the tool itself, independent of the LLM call's Usage
    usage: Usage,
    /// Whether the tool execution failed
    is_error: bool,
    /// Unix timestamp (milliseconds)
    timestamp: i64,
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

/// Full context for an LLM call, including system prompt, message history, and tool definitions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Context {
    /// System prompt
    system_prompt: Option<String>,
    /// Message history
    messages: Vec<Message>,
    /// List of tool definitions
    tools: Option<Vec<Tool>>,
}

/// Tool definition describing the tool's name, description, and parameter JSON Schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    /// Tool name
    name: String,
    /// Tool description
    description: String,
    /// Tool parameter JSON Schema (concrete type designed at implementation time; can be schemars::schema::Schema or serde_json::Value)
    parameters: serde_json::Value,
}
