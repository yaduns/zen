use crate::types::{
    AssistantMessage, AssistantMessageEvent, ContentBlock, ContentBlockList, Context,
    ErrorStopReason, Message, Model, NonErrorStopReason, StopReason, StreamFn, TextContent,
    ThinkingContent, ThinkingLevel, ToolCall, ToolChoice, ToolDefinition, ToolFunction, Usage,
};
use crate::utils::error_body::extract_error_body;
use crate::utils::provider_env::get_provider_env_value;
use eventsource_stream::Eventsource;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio_util::sync::CancellationToken;

/// Optional configuration parameters for the stream() function, controlling non-functional behavior of the API call (authentication, timeouts, tool-selection policy, thinking mode, etc.).
#[derive(Default)]
pub struct StreamConfig {
    pub api_key: Option<String>,
    // model is a field of AgentLoopConfig (the engine derives it from state.model each turn, see engine.rs);
    pub headers: Option<HashMap<String, String>>,
    pub tool_choice: Option<ToolChoice>,
    pub reasoning_effort: Option<ThinkingLevel>,
    pub max_tokens: Option<u64>, // Added: request-edge override bit; None = fall back to model.max_tokens
    pub temperature: Option<f64>, // Added: None = do not send (aligned with the pi default path)
    pub timeout_ms: Option<u64>,
    pub session_id: Option<String>,
}

/// Error types that the stream() function may return, covering scenarios such as missing API key, HTTP request failure, API error response, SSE parsing error, incomplete stream, and user cancellation.
#[derive(thiserror::Error, Debug)]
pub enum StreamError {
    #[error("API key not configured")]
    NoApiKey,

    #[error("HTTP request failed: {0}")]
    HttpRequest(#[source] reqwest::Error),

    #[error("API error: status {status}, body: {body}")]
    ApiError { status: u16, body: String },

    #[error("SSE parse error: {0}")]
    SSEError(String),

    #[error("Incomplete stream: {0}")]
    IncompleteStream(String),

    #[error("Request cancelled")]
    Cancelled,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Request {
    /// Model ID (e.g. `"deepseek-v4-flash"`).
    model: String,
    /// Message array.
    messages: Vec<RequestMessage>,
    /// Tool definition list, set only when `context.tools` is non-empty and the message history contains tool calls/results.
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
    /// Tool selection strategy
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    /// Maximum output token count.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
    /// Whether to use streaming responses. Always `true` for zen.
    stream: bool,
    /// Streaming options, includes `include_usage`.
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptionsParam>,
    /// Temperature parameter, controlling the randomness of the output.
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    /// Top-p sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    /// Stop word list; generation terminates when the model outputs any stop word.
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    /// DeepSeek thinking mode strength (API string value, converted from ThinkingLevel)
    #[serde(skip_serializing_if = "Option::is_none", rename = "reasoning_effort")]
    reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct StreamOptionsParam {
    /// Whether to return usage information in the final chunk.
    /// zen is always set to `true`.
    include_usage: bool,
}

/// Messages in the request body, tagged by the `role` field.
///
/// Corresponds to the OpenAI API `ChatCompletionMessageParam` union type.
/// Four variants: `system` / `user` / `assistant` / `tool`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "role", rename_all = "snake_case")]
enum RequestMessage {
    /// System message, sets the assistant's behavior and role.
    System {
        /// System instruction content.
        content: String,
    },
    /// User message, containing user input (text or text + image).
    User {
        /// User content; can be a plain text string or an array of content blocks.
        content: MessageContent,
    },
    /// Assistant message, containing model-generated text and/or tool call requests.
    Assistant {
        /// Text content. May be `None` (pure tool call response).
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        /// List of tool calls.
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<RequestToolCall>>,
    },
    /// Tool result message, containing the execution result of a tool call.
    Tool {
        /// Corresponding tool call ID.
        tool_call_id: String,
        /// Tool execution result content.
        content: MessageContent,
    },
}

#[derive(Debug, Clone, Serialize)]
struct RequestToolCall {
    id: String,
    r#type: String,
    function: RequestToolCallFunction,
}

#[derive(Debug, Clone, Serialize)]
struct RequestToolCallFunction {
    name: String,
    arguments: String,
}

/// Message content; can be a plain text string or an array of content blocks.
///
/// Corresponds to the OpenAI API `content` field, supports two formats:
/// - Plain text: `"Hello, world!"`
/// - Multipart: `[{"type": "text", "text": "Hello"}, {"type": "image_url", ...}]`
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
enum MessageContent {
    /// Plain text content.
    Text(String),
    /// Multi-part content (text + images, etc.).
    Parts(Vec<ContentPart>),
}

/// A content block in a multi-part message, tagged by the `type` field.
///
/// Mirrors the OpenAI API `ChatCompletionContentPart` union type.
/// Currently supports two types: `text` and `image_url`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentPart {
    /// Text content block.
    Text {
        /// Text content.
        text: String,
    },
    /// Image URL content block.
    ImageUrl {
        /// Image URL object, containing the URL and an optional detail parameter.
        image_url: ContentPartImageUrl,
    },
}

/// Internal structure of an image URL content block.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ContentPartImageUrl {
    /// Image URL (base64 data URI or HTTP URL).
    url: String,
    /// Image detail level (`"auto"` / `"low"` / `"high"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
}

// ═══════════════════════════════════════════════════════════════════
//  Internal response types (Deserialize) — parse SSE streaming response chunks
//  Not exported; used only inside process_chunk / sse_event_loop
// ═══════════════════════════════════════════════════════════════════

/// A single data line in an SSE streaming response, corresponding to the OpenAI API `ChatCompletionChunk`.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Chunk {
    /// Request-level unique identifier.
    id: String,
    /// Object type, always `"chat.completion.chunk"`.
    object: String,
    /// Creation time (Unix timestamp).
    created: u64,
    /// Actual response model name.
    model: String,
    /// Choice list (usually a single item, `n=1`).
    choices: Vec<ChunkChoice>,
    /// Token usage stats (non-`null` only in the final chunk).
    usage: Option<ChunkUsage>,
}

/// A single choice, containing streaming deltas and the finish reason.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChunkChoice {
    /// Choice index (starting from 0).
    index: u32,
    /// Delta content.
    delta: ChunkDelta,
    /// Finish reason: `"stop"` / `"length"` / `"tool_calls"` / `null`.
    /// `null` indicates the stream has not ended yet.
    finish_reason: Option<String>,
    /// Logprobs information (usually `null`).
    logprobs: Option<serde_json::Value>,
}

/// Incremental content in the streaming response.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChunkDelta {
    /// Text delta content. The first chunk may be `null` (only sets `role`).
    #[serde(default)]
    content: Option<String>,
    /// DeepSeek reasoning delta content (the `reasoning_content` field).
    #[serde(default)]
    reasoning_content: Option<String>,
    /// List of tool call deltas.
    #[serde(default)]
    tool_calls: Option<Vec<ChunkToolCallDelta>>,
    /// Role (only `"assistant"` in the first chunk).
    #[serde(default)]
    role: Option<String>,
}

/// Tool call delta content, associated with the same tool call by `index`.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChunkToolCallDelta {
    /// Tool call index, used to associate multiple chunks of the same tool call.
    index: u32,
    /// Tool call ID (only present in the first chunk).
    #[serde(default)]
    id: Option<String>,
    /// Tool type (only present in the first chunk, always `"function"`).
    #[serde(default, rename = "type")]
    type_: Option<String>,
    /// Function call delta.
    #[serde(default)]
    function: Option<ChunkToolCallDeltaFunction>,
}

/// Function details within the tool call delta.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChunkToolCallDeltaFunction {
    /// Tool name (only present in the first chunk).
    #[serde(default)]
    name: Option<String>,
    /// Tool arguments string fragment (incremental JSON string).
    #[serde(default)]
    arguments: Option<String>,
}

/// Token usage statistics in the final chunk.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChunkUsage {
    /// Number of input tokens.
    prompt_tokens: u64,
    /// Number of output tokens.
    completion_tokens: u64,
    /// Total number of tokens.
    total_tokens: u64,
    /// Input token details (including cache hit info).
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
}

/// Input token details, including cache hit info.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PromptTokensDetails {
    /// Number of cache-hit tokens.
    cached_tokens: u64,
}

/// Initiate an OpenAI Chat Completions SSE streaming call.
///
/// This is the full-signature version of `stream()`, accepting all optional parameters.
/// It pushes parsed `AssistantMessageEvent`s asynchronously into the returned
/// `UnboundedReceiver` via `mpsc::unbounded_channel`; callers consume events with `recv()`.
///
/// # Event stream lifecycle
///
/// 1. The first event is always `AssistantMessageEvent::Start`, carrying the initial (empty) `partial` message.
/// 2. Intermediate events are incremental updates: `TextStart/Delta/End`, `ThinkingStart/Delta/End`,
///    `ToolcallStart/Delta/End`, interleaved in LLM output order.
/// 3. The last event is `AssistantMessageEvent::Done` (success) or
///    `AssistantMessageEvent::Error` (failure).
/// 4. After the stream ends, the receiver returns `None` (channel closed).
///
/// # Parameters
///
/// - `model`: Model config, containing the API endpoint, model ID, max_tokens, extra headers, etc.
///   Builds the request URL from `model.base_url` (appending `/chat/completions`).
/// - `context`: Request context, containing the system prompt, message history, and tool definitions.
///   `Message::ToolResult` in the message history may carry the `excludeFromContext` flag
///   (filtering is handled by upstream callers; this function does not filter).
/// - `options`: Optional configuration parameters. When `None`, all parameters use their defaults.
/// - `cancel`: Optional cancellation token. On cancellation, SSE reading terminates immediately, pushing `Error::Cancelled`
///   events and closing the channel.
///
/// # Return Value
///
/// `Ok(UnboundedReceiver<AssistantMessageEvent>)` — the receiving end of the async event stream.
/// Callers should keep calling `recv()` until the channel closes (returns `None`).
/// `Err(StreamError)` — an error that occurred before any events were sent
/// (e.g. missing API key, URL construction failure). Once the event stream has started, errors are delivered
/// via `AssistantMessageEvent::Error` and are not propagated through the returned `Err`.
pub async fn stream(
    model: &Model,
    context: &Context,
    options: Option<StreamConfig>,
    cancel: Option<CancellationToken>,
) -> Result<UnboundedReceiver<AssistantMessageEvent>, StreamError> {
    let cfg = options.unwrap_or_default();
    let cancel = cancel.unwrap_or_default();

    // 1. API Key
    let api_key = cfg
        .api_key
        .clone()
        .or_else(|| match model.provider.as_str() {
            // Per-provider env variable (aligned with pi `envApiKeyAuth`)
            "opencode-go" => get_provider_env_value("OPENCODE_API_KEY", None),
            _ => get_provider_env_value("ARK_API_KEY", None),
        })
        .ok_or(StreamError::NoApiKey)?;

    // 2. Build request body
    let body = build_request_body(model, context, &cfg);

    // 3. Build HTTP request
    let client = reqwest::Client::new();
    let url = model.clone().api_url();

    let mut req = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&body);

    // Merge extra request headers
    if let Some(headers) = &cfg.headers {
        for (k, v) in headers {
            req = req.header(k.as_str(), v.as_str());
        }
    }

    // Timeout
    let req = if let Some(timeout_ms) = cfg.timeout_ms {
        req.timeout(std::time::Duration::from_millis(timeout_ms))
    } else {
        req
    };

    // 4. Send request
    let response = req.send().await.map_err(StreamError::HttpRequest)?;

    // 5. Check status code
    if !response.status().is_success() {
        let status = response.status().as_u16();
        let body = extract_error_body(response, 4000).await.unwrap_or_default();
        return Err(StreamError::ApiError { status, body });
    }

    // 6. Create channel and start background task
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    let model_clone = model.clone();
    tokio::spawn(async move {
        sse_event_loop(tx, &model_clone, response, Some(cancel)).await;
    });

    Ok(rx)
}

/// Perform a streaming call, collect all events, and return the final `AssistantMessage`.
///
/// Calls `stream()` internally, consumes all events, discards intermediate incremental events,
/// and extracts the final `AssistantMessage` from the `Done` event.
/// If an `Error` event is received, returns the corresponding error.
///
/// This is the non-streaming equivalent — callers don't need to care about the event stream,
/// they only need the final result. The `produce` closure of the retry logic (`retry_assistant_call` in T-AI-02-2)
/// calls this function.
///
/// # Parameters
/// Same as `stream()`.
///
/// # Return Value
/// - `Ok(AssistantMessage)`: the complete assistant message, including usage and stop_reason.
/// - `Err(StreamError)`: an error before the stream starts, or a fatal error during the stream.
pub async fn stream_collect(
    model: &Model,
    context: &Context,
    options: Option<StreamConfig>,
    cancel: Option<CancellationToken>,
) -> Result<crate::types::AssistantMessage, StreamError> {
    let mut receiver = stream(model, context, options, cancel).await?;

    while let Some(event) = receiver.recv().await {
        match event {
            AssistantMessageEvent::Done { message, .. } => return Ok(message),
            AssistantMessageEvent::Error { .. } => {
                return Err(StreamError::IncompleteStream("Stream error".into()));
            }
            _ => {} // Ignore intermediate delta events
        }
    }

    Err(StreamError::IncompleteStream(
        "Stream ended without Done or Error event".into(),
    ))
}

/// Builds the OpenAI Chat Completions request body from `Context` + `Model`.
///
/// Converts zen internal types (`Message` / `ContentBlock` / `Tool`) to
/// the OpenAI API JSON request format. Handles:
/// - `UserMessage` → `role: "user"`
/// - `AssistantMessage` → `role: "assistant"` (with tool_calls)
/// - `ToolResultMessage` → `role: "tool"` (with tool_call_id)
/// - SystemPrompt (from `Context.system_prompt`) → `role: "system"`
/// - `ContentBlock::Image` → OpenAI image_url format
/// - `ContentBlock::Thinking` → skipped (OpenAI does not send thinking to the LLM)
/// Map `ThinkingLevel` to the OpenAI API `reasoning_effort` string value.
/// `ThinkingLevel::Off` returns `None` (the field is not sent).
fn thinking_level_to_api_value(level: ThinkingLevel) -> Option<String> {
    match level {
        ThinkingLevel::Off => None,
        ThinkingLevel::Minimal => Some("minimal".to_string()),
        ThinkingLevel::Low => Some("low".to_string()),
        ThinkingLevel::Medium => Some("medium".to_string()),
        ThinkingLevel::High => Some("high".to_string()),
        ThinkingLevel::XHigh => Some("xhigh".to_string()),
        ThinkingLevel::Max => Some("max".to_string()),
    }
}

fn build_request_body(model: &Model, context: &Context, cfg: &StreamConfig) -> Request {
    let mut messages = Vec::new();

    // System prompt as the first message
    if let Some(ref prompt) = context.system_prompt {
        messages.push(RequestMessage::System {
            content: prompt.clone(),
        });
    }

    // Convert all messages
    for msg in &context.messages {
        messages.push(convert_message(msg));
    }

    // Tool definitions: only set when non-empty (schema derived from executable tools at the request boundary, aligned with Pi's convertTools)
    let tools = context.tools.as_ref().and_then(|ts| {
        if ts.is_empty() {
            None
        } else {
            Some(ts.iter().map(|t| t.to_api_definition()).collect())
        }
    });

    let req = Request {
        model: model.id.clone(),
        messages,
        tools,
        tool_choice: cfg.tool_choice.clone(),
        max_tokens: cfg.max_tokens.or(Some(model.max_tokens)),
        stream: true,
        stream_options: Some(StreamOptionsParam {
            include_usage: true,
        }),
        temperature: cfg.temperature,
        top_p: None,
        stop: None,
        reasoning_effort: cfg.reasoning_effort.and_then(thinking_level_to_api_value),
    };

    req
}

/// Converts a `Message` into an OpenAI API message object.
///
/// Corresponds to Pi's internal `convertMessage` function.
fn convert_message(message: &Message) -> RequestMessage {
    match message {
        Message::User(user_message) => match &user_message.content {
            ContentBlockList::Text(text) => RequestMessage::User {
                content: MessageContent::Text(text.to_string()),
            },
            ContentBlockList::Blocks(blocks) => RequestMessage::User {
                content: MessageContent::Parts(
                    blocks
                        .iter()
                        .map(|block| convert_content_block(block))
                        .flatten()
                        .collect(),
                ),
            },
        },

        Message::Assistant(assistant_message) => {
            let mut _text_content = String::new();
            let mut _tool_calls = Vec::new();

            for content in &assistant_message.content {
                match content {
                    ContentBlock::Text(text_content) => _text_content = text_content.text.clone(),
                    ContentBlock::ToolCall(tool_call) => {
                        let request_tool_call = RequestToolCall {
                            id: tool_call.id.clone(),
                            r#type: String::from("function"),
                            function: RequestToolCallFunction {
                                name: tool_call.name.clone(),
                                arguments: tool_call.arguments.to_string(),
                            },
                        };
                        _tool_calls.push(request_tool_call);
                    }
                    _ => (),
                }
            }
            RequestMessage::Assistant {
                content: Some(_text_content),
                tool_calls: Some(_tool_calls),
            }
        }
        Message::ToolResult(tool_result_message) => {
            // Single plain text → use MessageContent::Text for simpler serialization
            if tool_result_message.content.len() == 1
                && matches!(tool_result_message.content[0], ContentBlock::Text(_))
            {
                if let ContentBlock::Text(text) = &tool_result_message.content[0] {
                    // Ownership of tool_result_message.content was already borrowed by the match, so clone is required
                    RequestMessage::Tool {
                        tool_call_id: tool_result_message.tool_call_id.clone(),
                        content: MessageContent::Text(text.text.clone()),
                    }
                } else {
                    unreachable!() // The matches above guarantee this is Text here
                }
            } else {
                // Multiple blocks or contains images → convert block by block, assemble with Parts
                RequestMessage::Tool {
                    tool_call_id: tool_result_message.tool_call_id.clone(),
                    content: MessageContent::Parts(
                        tool_result_message
                            .content
                            .iter()
                            .filter_map(convert_content_block)
                            .collect(),
                    ),
                }
            }
        }
    }
}

/// Convert a `ContentBlock` to an OpenAI content block object.
///
/// Counterpart of Pi's internal `convertContentBlock` function.
fn convert_content_block(block: &crate::types::ContentBlock) -> Option<ContentPart> {
    match block {
        ContentBlock::Text(text_content) => Some(ContentPart::Text {
            text: text_content.text.clone(),
        }),
        ContentBlock::Image(image_content) => Some(ContentPart::ImageUrl {
            image_url: ContentPartImageUrl {
                url: format!(
                    "data:{};base64,{}",
                    image_content.mime_type, image_content.data
                ),
                detail: Some("high".to_string()),
            },
        }),
        _ => None,
    }
}

// Parse a single SSE chunk's JSON into a `StreamEvent` domain event.
///
/// Convert OpenAI's chunk format (`choices[0].delta.{content,reasoning_content,tool_calls}`)
/// Mapped into the `StreamEvent` enum for `StreamAccumulator` to consume.
///
/// Handles:
/// - First chunk (`choices[0].delta.role === "assistant"`) → `ContentBlockStart { block_type: "text" }`
/// - Regular chunk (`delta.content` non-empty) → `TextDelta`
/// - DeepSeek thinking chunk (`delta.reasoning_content` non-empty) → `ThinkingDelta`
/// - First tool call chunk (`delta.tool_calls[0].id` non-empty) → `ContentBlockStart { block_type: "tool_use" }` + `InputJsonDelta`
/// - Tool call incremental chunk (`delta.tool_calls[0].function.arguments` non-empty) → `InputJsonDelta`
/// - Final chunk (`choices[0].finish_reason` non-empty) → `ContentBlockStop` + `MessageDelta`
/// - usage chunk (`usage` non-empty, may exist independently of choices) → `MessageDelta`
///
/// Note: `usage` is processed before `choices` because some APIs return usage as a separate chunk with `choices: []`.
///
/// Combined with Pi's internal functions such as `ensureTextBlock` / `ensureThinkingBlock` / `finishBlock`
/// chunk parsing logic.
fn parse_chunk(chunk: &serde_json::Value) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    // usage → message metadata (final chunk only, may exist independently of choices)
    // Must be handled before the early return, because some APIs only return usage in the final chunk
    // with choices as an empty array, which would cause the early return to skip usage parsing.
    if let Some(u) = chunk.get("usage").filter(|u| u.is_object()) {
        let (input, output, cache_read, total, reasoning) = parse_usage(u);
        events.push(StreamEvent::MessageDelta {
            stop_reason: None,
            usage: Some(Usage {
                input,
                output,
                cache_read,
                total_tokens: total,
                reasoning,
                ..Usage::default()
            }),
        });
    }

    let choice = match chunk.get("choices").and_then(|c| c.as_array()?.first()) {
        Some(c) => c,
        None => return events,
    };

    let delta = &choice["delta"];
    let idx = 0; // n=1, only choice index 0

    // Text delta (Pi mode: text chunks are created lazily by TextDelta, not triggered by role)
    if let Some(t) = delta["content"].as_str().filter(|s| !s.is_empty()) {
        events.push(StreamEvent::TextDelta {
            index: idx,
            text: t.into(),
        });
    }

    // Thinking delta (DeepSeek)
    if let Some(t) = delta["reasoning_content"]
        .as_str()
        .filter(|s| !s.is_empty())
    {
        events.push(StreamEvent::ThinkingDelta {
            index: idx,
            thinking: t.into(),
        });
    }

    // tool call
    if let Some(tcs) = delta["tool_calls"].as_array() {
        for tc in tcs {
            let i = tc["index"].as_u64().unwrap_or(0) as usize;
            if tc["id"].as_str().is_some() {
                events.push(StreamEvent::ContentBlockStart {
                    index: i,
                    block_type: "tool_use".into(),
                    id: tc["id"].as_str().map(String::from),
                    name: tc["function"]["name"].as_str().map(String::from),
                });
            }
            if let Some(a) = tc["function"]["arguments"]
                .as_str()
                .filter(|s| !s.is_empty())
            {
                events.push(StreamEvent::InputJsonDelta {
                    index: i,
                    partial_json: a.into(),
                });
            }
        }
    }

    // finish_reason → block stop + message metadata
    if let Some(r) = choice["finish_reason"].as_str().filter(|s| !s.is_empty()) {
        events.push(StreamEvent::ContentBlockStop { index: idx });
        events.push(StreamEvent::MessageDelta {
            stop_reason: Some(map_finish_reason(r)),
            usage: None,
        });
    }

    events
}

/// Extract raw token counts from the `usage` JSON of the API's final chunk; does not compute cost.
fn parse_usage(usage: &serde_json::Value) -> (u64, u64, u64, u64, Option<u64>) {
    let input = usage["prompt_tokens"].as_u64().unwrap_or(0);
    let output = usage["completion_tokens"].as_u64().unwrap_or(0);
    let total = usage["total_tokens"].as_u64().unwrap_or(0);
    let cache_read = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);
    let reasoning = usage["completion_tokens_details"]["reasoning_tokens"].as_u64();
    (input, output, cache_read, total, reasoning)
}

/// Map OpenAI's `finish_reason` string to `StopReason`.
fn map_finish_reason(reason: &str) -> StopReason {
    match reason {
        "stop" | "Stop" => StopReason::Stop,
        "length" | "Length" => StopReason::Length,
        "tool_calls" | "ToolUse" => StopReason::ToolUse,
        _ => StopReason::Error,
    }
}

/// Domain events parsed from SSE chunks, for `StreamAccumulator` to consume.
///
/// Difference from `AssistantMessageEvent`:
/// - `StreamEvent` is the accumulator's input, with fine granularity (content/json/thinking per block increment)
/// - `AssistantMessageEvent` is the output for external callers, containing the `partial` snapshot
enum StreamEvent {
    /// A new content block begins. `block_type` is `"text"` / `"tool_use"` / `"thinking"`.
    ContentBlockStart {
        index: usize,
        block_type: String,
        id: Option<String>,
        name: Option<String>,
    },
    /// Text delta.
    TextDelta { index: usize, text: String },
    /// Thinking/reasoning content delta.
    ThinkingDelta { index: usize, thinking: String },
    /// Tool call argument JSON delta.
    InputJsonDelta { index: usize, partial_json: String },
    /// Content block ended.
    ContentBlockStop { index: usize },
    /// Overall message delta (stop_reason and usage in the final chunk).
    MessageDelta {
        stop_reason: Option<StopReason>,
        usage: Option<Usage>,
    },
}

// ── Internal accumulator ────────────────────────────────────────

/// Streaming accumulator: incrementally reduces `StreamEvent`s into a complete `AssistantMessage`.
///
/// Modeled after cersei's `StreamAccumulator` design (`crates/cersei-provider/src/stream.rs`).
///
/// Core principles:
/// - `content_blocks` stores only finalized blocks (written at `ContentBlockStop`)
/// - The `partial_*` maps store only incomplete data, indexed by content index
/// - `process_event()` is a pure accumulation operation and returns no event
/// - `into_response()` consumes self and returns the final `AssistantMessage`
#[derive(Default)]
struct StreamAccumulator {
    /// Final output content blocks (only blocks after finalize are here)
    content_blocks: Vec<ContentBlock>,
    /// Incomplete text indexed by content index
    partial_text: HashMap<usize, String>,
    /// Incomplete thinking content indexed by content index
    partial_thinking: HashMap<usize, String>,
    /// Incomplete tool call argument JSON fragments indexed by content index
    partial_json: HashMap<usize, String>,
    /// Block type (text/thinking/tool_use) indexed by content index
    block_types: HashMap<usize, String>,
    /// Tool call IDs indexed by content index
    tool_use_ids: HashMap<usize, String>,
    /// Tool names indexed by content index
    tool_use_names: HashMap<usize, String>,
    /// Stop reason (set by `MessageDelta` events)
    stop_reason: Option<StopReason>,
    /// Usage statistics (set by `MessageDelta` events)
    usage: Usage,
    /// Whether a finish_reason has been received
    has_finish_reason: bool,
}

impl StreamAccumulator {
    /// Consumes a `StreamEvent` and updates the accumulator's internal state.
    /// Returns no events — event pushing is handled by `send()` in `sse_event_loop`.
    fn process_event(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::ContentBlockStart {
                index,
                block_type,
                id,
                name,
            } => {
                self.block_types.insert(index, block_type);
                // Only update id/name if non-empty. Some providers (e.g. DeepSeek)
                // send subsequent chunks with id="" and name="" which would
                // overwrite the correct values from the first chunk.
                if let Some(id) = id {
                    if !id.is_empty() {
                        self.tool_use_ids.insert(index, id);
                    }
                }
                if let Some(name) = name {
                    if !name.is_empty() {
                        self.tool_use_names.insert(index, name);
                    }
                }
            }

            StreamEvent::TextDelta { index, text } => {
                // Lazily create a text block if not already started at this index.
                // This matches Pi's ensureTextBlock() pattern: text blocks are created
                // when text content actually arrives, not when role: "assistant" is seen.
                // This avoids creating empty text blocks for tool-call-only responses.
                self.block_types
                    .entry(index)
                    .or_insert_with(|| "text".to_string());
                self.partial_text.entry(index).or_default().push_str(&text);
            }

            StreamEvent::ThinkingDelta { index, thinking } => {
                self.partial_thinking
                    .entry(index)
                    .or_default()
                    .push_str(&thinking);
            }

            StreamEvent::InputJsonDelta {
                index,
                partial_json,
            } => {
                self.partial_json
                    .entry(index)
                    .or_default()
                    .push_str(&partial_json);
            }

            StreamEvent::ContentBlockStop { index } => {
                let block = match self.block_types.remove(&index).as_deref() {
                    Some("text") => {
                        let text = self.partial_text.remove(&index).unwrap_or_default();
                        ContentBlock::Text(TextContent {
                            text,
                            text_signature: None,
                        })
                    }
                    Some("thinking") => {
                        let thinking = self.partial_thinking.remove(&index).unwrap_or_default();
                        ContentBlock::Thinking(ThinkingContent {
                            thinking,
                            thinking_signature: None,
                            redacted: None,
                        })
                    }
                    Some("tool_use") => {
                        let id = self.tool_use_ids.remove(&index).unwrap_or_default();
                        let name = self.tool_use_names.remove(&index).unwrap_or_default();
                        let json = self.partial_json.remove(&index).unwrap_or_default();
                        let arguments = serde_json::from_str(&json)
                            .unwrap_or(serde_json::Value::Object(Default::default()));
                        ContentBlock::ToolCall(ToolCall {
                            id,
                            name,
                            arguments,
                            thought_signature: None,
                        })
                    }
                    _ => return,
                };
                self.content_blocks.push(block);
            }

            StreamEvent::MessageDelta { stop_reason, usage } => {
                if let Some(reason) = stop_reason {
                    self.stop_reason = Some(reason);
                    self.has_finish_reason = true;
                }
                if let Some(u) = usage {
                    self.usage = u;
                }
            }
        }
    }
}

// ── Event stream background task ────────────────────────────────

/// Background SSE event loop.
/// Runs in a dedicated tokio task, reading the SSE event stream from the HTTP response body.
/// Internal flow:
/// 1. Use `eventsource_stream::Eventsource` to parse SSE framing
/// 2. For each `data: {...}` line, call `parse_chunk()` to convert it into `Vec<StreamEvent>`
/// 3. For each `StreamEvent`, call `accumulator.process_event()` to accumulate
/// 4. At `ContentBlockStop` / `MessageDelta` boundaries, construct `AssistantMessageEvent`
///    and push it to the caller via `sender`
/// 5. On receiving `[DONE]` or `finish_reason`, call `accumulator.into_response()`
///    push the `Done` event and close the channel
/// Parameters:
/// - `sender`: the sender end for pushing events
/// - `model`: model configuration (used to compute cost)
/// - `response`: the `reqwest` streaming response
/// - `cancel`: optional cancellation token
async fn sse_event_loop(
    sender: tokio::sync::mpsc::UnboundedSender<AssistantMessageEvent>,
    model: &Model,
    response: reqwest::Response,
    cancel: Option<CancellationToken>,
) {
    use futures_util::StreamExt;

    // ── Phase 1: Initialization ──────────────────────────────────
    // Create the accumulator, SSE stream, cancellation token, response ID, and error message placeholder.
    let mut accumulator = StreamAccumulator::default();
    let mut stream = response.bytes_stream().eventsource();
    let cancel = cancel.unwrap_or_default();
    let mut response_id: Option<String> = None;
    let mut error_message: Option<String> = None;

    // ── Phase 2: Push Start event ─────────────────────────────────
    // Send an empty partial message to notify the caller that the stream has started.
    let partial = AssistantMessage {
        content: Vec::new(),
        api: "openai-completions".to_string(),
        provider: "deepseek".to_string(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: chrono::Utc::now().timestamp_millis(),
    };
    let _ = sender.send(AssistantMessageEvent::Start { partial });

    // ── Phase 3: SSE event loop ────────────────────────────────────
    // Race two branches: cancellation signal or next SSE event.
    loop {
        tokio::select! {
            biased;
            // 3a. User cancelled → set error message and exit the loop
            _ = cancel.cancelled() => {
                error_message = Some("Request cancelled by user".to_string());
                break;
            }
            // 3b. Read the next SSE event
            event = stream.next() => {
                match event {
                    Some(Ok(eventsource_event)) => {
                        let data = eventsource_event.data.trim().to_string();
                        // 3c. [DONE] signal → normal termination
                        if data == "[DONE]" { break; }
                        // Skip non-JSON lines (e.g., blank lines, comments)
                        if !data.starts_with('{') { continue; }

                        // 3d. Parse the JSON chunk
                        let json: serde_json::Value = match serde_json::from_str(&data) {
                            Ok(v) => v,
                            Err(_) => {
                                error_message = Some(format!("Failed to parse SSE data: {}", data));
                                break;
                            }
                        };

                        // 3e. Record response_id from the first chunk
                        if response_id.is_none() {
                            response_id = json["id"].as_str().map(String::from);
                        }

                                        // 3f. Parse chunk into StreamEvent, accumulate each one
                        let events = parse_chunk(&json);
                        for ev in events {
                            // stop_reason and usage from MessageDelta are written directly into the accumulator
                            if let StreamEvent::MessageDelta { ref stop_reason, ref usage } = ev {
                                if let Some(sr) = stop_reason {
                                    accumulator.stop_reason = Some(sr.clone());
                                }
                                if let Some(u) = usage {
                                    accumulator.usage = u.clone();
                                }
                            }
                            accumulator.process_event(ev);
                        }
                    }
                    // 3g. SSE transport-layer error → record and exit
                    Some(Err(e)) => {
                        error_message = Some(format!("SSE stream error: {}", e));
                        break;
                    }
                    // 3h. Stream ended normally (EOF)
                    None => break,
                }
            }
        }
    }

    // ── Phase 4: Build the final AssistantMessage ───────────────────────
    // Take all finalized content blocks from the accumulator.
    let mut content = Vec::new();
    std::mem::swap(&mut content, &mut accumulator.content_blocks);

    // 4a. Reconcile the relationship between stop_reason and error_message:
    //     - has finish_reason but also an error → keep finish_reason, write the error into error_message
    //     - Has error and no finish_reason → termination reason is Error
    //     - normal termination → use the accumulator's stop_reason
    let (stop_reason, error_msg) = match error_message {
        Some(ref msg) if accumulator.has_finish_reason => (
            accumulator.stop_reason.unwrap_or(StopReason::Stop),
            Some(msg.clone()),
        ),
        Some(ref msg) => (StopReason::Error, Some(msg.clone())),
        None => (accumulator.stop_reason.unwrap_or(StopReason::Stop), None),
    };

    // 4b. Assemble the final message
    let mut msg = AssistantMessage {
        content,
        api: "openai-completions".to_string(),
        provider: model.id.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id,
        usage: accumulator.usage,
        stop_reason,
        error_message: error_msg,
        timestamp: chrono::Utc::now().timestamp_millis(),
    };

    // 4c. If the API did not return usage (e.g. Volcano Engine DeepSeek streaming mode), estimate output from content
    if msg.usage.input == 0 && msg.usage.output == 0 && !msg.content.is_empty() {
        let estimated = crate::utils::estimate::estimate_content_blocks_tokens(&msg.content) as u64;
        msg.usage.output = estimated;
        msg.usage.total_tokens = estimated;
    }

    // ── Phase 5: Push Done / Error events ──────────────────────────
    // Error if there is an error and no finish_reason was received
    // Otherwise → Done (covers normal termination, tool calls, length truncation, etc.)
    if error_message.is_some() && !accumulator.has_finish_reason {
        let reason = if cancel.is_cancelled() {
            ErrorStopReason::Aborted
        } else {
            ErrorStopReason::Error
        };
        let _ = sender.send(AssistantMessageEvent::Error { reason, error: msg });
    } else {
        let reason = match msg.stop_reason {
            StopReason::Stop => NonErrorStopReason::Stop,
            StopReason::Length => NonErrorStopReason::Length,
            StopReason::ToolUse => NonErrorStopReason::ToolUse,
            _ => NonErrorStopReason::Stop, // fallback
        };
        let _ = sender.send(AssistantMessageEvent::Done {
            reason,
            message: msg,
        });
    }
}

/// `StreamFn` wrapper around the existing OpenAI Chat Completions streaming implementation.
///
/// Wrap `pub async fn stream(model, context, options, cancel)` as a `StreamFn`
/// (clone model/context inside the closure then move into the async block, resolving the `'static` future constraint,
/// consistent with the current T-AP-08-3 style).
///
/// # Consumers
/// Called only internally by `stream_for` (same crate).
pub fn openai_completions_stream() -> StreamFn {
    Arc::new(|model, context, options, cancel| {
        let model = model.clone();
        let context = context.clone();
        Box::pin(async move { stream(&model, &context, options, cancel).await })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::Api;
    use crate::types::{
        AssistantMessage, ContentBlock, ContentBlockList, ImageContent, Message, TextContent,
        ThinkingContent, ToolCall, ToolResultMessage, Usage, UserMessage,
    };

    // ═════════════════════════════════════════════════════════════
    //  map_finish_reason
    // ═════════════════════════════════════════════════════════════

    #[test]
    fn test_map_finish_reason_stop() {
        assert_eq!(map_finish_reason("stop"), StopReason::Stop);
    }

    #[test]
    fn test_map_finish_reason_length() {
        assert_eq!(map_finish_reason("length"), StopReason::Length);
    }

    #[test]
    fn test_map_finish_reason_tool_calls() {
        assert_eq!(map_finish_reason("tool_calls"), StopReason::ToolUse);
    }

    #[test]
    fn test_map_finish_reason_unknown() {
        assert_eq!(map_finish_reason(""), StopReason::Error);
        assert_eq!(map_finish_reason("content_filter"), StopReason::Error);
    }

    // ═════════════════════════════════════════════════════════════
    //  parse_usage
    // ═════════════════════════════════════════════════════════════

    #[test]
    fn test_parse_usage_full() {
        let json = serde_json::json!({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": { "cached_tokens": 20 }
        });
        let (input, output, cache_read, total, reasoning) = parse_usage(&json);
        assert_eq!(input, 100);
        assert_eq!(output, 50);
        assert_eq!(cache_read, 20);
        assert_eq!(total, 150);
        assert_eq!(reasoning, None);
    }

    #[test]
    fn test_parse_usage_no_details() {
        let json = serde_json::json!({
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        });
        let (input, output, cache_read, total, reasoning) = parse_usage(&json);
        assert_eq!(input, 10);
        assert_eq!(output, 20);
        assert_eq!(cache_read, 0);
        assert_eq!(total, 30);
        assert_eq!(reasoning, None);
    }

    #[test]
    fn test_parse_usage_empty() {
        let json = serde_json::json!({});
        let (input, output, cache_read, total, reasoning) = parse_usage(&json);
        assert_eq!(input, 0);
        assert_eq!(output, 0);
        assert_eq!(cache_read, 0);
        assert_eq!(total, 0);
        assert_eq!(reasoning, None);
    }

    #[test]
    fn test_parse_usage_with_reasoning() {
        let json = serde_json::json!({
            "prompt_tokens": 100,
            "completion_tokens": 80,
            "total_tokens": 180,
            "completion_tokens_details": { "reasoning_tokens": 30 }
        });
        let (_, _, _, _, reasoning) = parse_usage(&json);
        assert_eq!(reasoning, Some(30));
    }

    // ═════════════════════════════════════════════════════════════
    //  convert_content_block
    // ═════════════════════════════════════════════════════════════

    #[test]
    fn test_convert_content_block_text() {
        let block = ContentBlock::Text(TextContent {
            text: "hello".into(),
            text_signature: None,
        });
        let result = convert_content_block(&block);
        assert!(matches!(result, Some(ContentPart::Text { .. })));
        if let Some(ContentPart::Text { text }) = result {
            assert_eq!(text, "hello");
        }
    }

    #[test]
    fn test_convert_content_block_image() {
        let block = ContentBlock::Image(ImageContent {
            data: "base64data".into(),
            mime_type: "image/png".into(),
        });
        let result = convert_content_block(&block);
        assert!(matches!(result, Some(ContentPart::ImageUrl { .. })));
        if let Some(ContentPart::ImageUrl { image_url }) = result {
            assert_eq!(image_url.url, "data:image/png;base64,base64data");
            assert_eq!(image_url.detail, Some("high".into()));
        }
    }

    #[test]
    fn test_convert_content_block_thinking_is_none() {
        let block = ContentBlock::Thinking(ThinkingContent {
            thinking: "...".into(),
            thinking_signature: None,
            redacted: None,
        });
        assert!(convert_content_block(&block).is_none());
    }

    #[test]
    fn test_convert_content_block_tool_call_is_none() {
        let block = ContentBlock::ToolCall(ToolCall {
            id: "call_1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({}),
            thought_signature: None,
        });
        assert!(convert_content_block(&block).is_none());
    }

    // ═════════════════════════════════════════════════════════════
    //  convert_message
    // ═════════════════════════════════════════════════════════════

    #[test]
    fn test_convert_message_user_text() {
        let msg = Message::User(UserMessage {
            content: ContentBlockList::Text("hello".into()),
            timestamp: 0,
        });
        let result = convert_message(&msg);
        assert!(matches!(
            result,
            RequestMessage::User {
                content: MessageContent::Text(ref t),
                ..
            } if t == "hello"
        ));
    }

    #[test]
    fn test_convert_message_user_blocks() {
        let msg = Message::User(UserMessage {
            content: ContentBlockList::Blocks(vec![
                ContentBlock::Text(TextContent {
                    text: "hello".into(),
                    text_signature: None,
                }),
                ContentBlock::Image(ImageContent {
                    data: "img".into(),
                    mime_type: "image/png".into(),
                }),
            ]),
            timestamp: 0,
        });
        let result = convert_message(&msg);
        assert!(matches!(
            result,
            RequestMessage::User {
                content: MessageContent::Parts(..),
                ..
            }
        ));
    }

    #[test]
    fn test_convert_message_assistant_text_only() {
        let msg = Message::Assistant(AssistantMessage {
            content: vec![ContentBlock::Text(TextContent {
                text: "response".into(),
                text_signature: None,
            })],
            api: "openai-completions".into(),
            provider: "deepseek".into(),
            model: "deepseek-v4-flash".into(),
            response_model: None,
            response_id: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 0,
        });
        let result = convert_message(&msg);
        assert!(matches!(
            result,
            RequestMessage::Assistant {
                content: Some(ref t),
                tool_calls: Some(ref tc),
                ..
            } if t == "response" && tc.is_empty()
        ));
    }

    #[test]
    fn test_convert_message_assistant_with_tool_calls() {
        let msg = Message::Assistant(AssistantMessage {
            content: vec![
                ContentBlock::Text(TextContent {
                    text: "I'll run ls".into(),
                    text_signature: None,
                }),
                ContentBlock::ToolCall(ToolCall {
                    id: "call_1".into(),
                    name: "bash".into(),
                    arguments: serde_json::json!({ "command": "ls" }),
                    thought_signature: None,
                }),
            ],
            api: "openai-completions".into(),
            provider: "deepseek".into(),
            model: "deepseek-v4-flash".into(),
            response_model: None,
            response_id: None,
            usage: Usage::default(),
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: 0,
        });
        let result = convert_message(&msg);
        assert!(matches!(
            result,
            RequestMessage::Assistant {
                content: Some(ref t),
                tool_calls: Some(ref tc),
                ..
            } if t == "I'll run ls" && tc.len() == 1 && tc[0].id == "call_1"
        ));
    }

    #[test]
    fn test_convert_message_assistant_skips_thinking() {
        let msg = Message::Assistant(AssistantMessage {
            content: vec![
                ContentBlock::Thinking(ThinkingContent {
                    thinking: "hidden".into(),
                    thinking_signature: None,
                    redacted: None,
                }),
                ContentBlock::Text(TextContent {
                    text: "visible".into(),
                    text_signature: None,
                }),
            ],
            api: "openai-completions".into(),
            provider: "deepseek".into(),
            model: "deepseek-v4-flash".into(),
            response_model: None,
            response_id: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 0,
        });
        // Only text blocks are extracted; thinking blocks are skipped
        let result = convert_message(&msg);
        assert!(matches!(
            result,
            RequestMessage::Assistant {
                content: Some(ref t),
                ..
            } if t == "visible"
        ));
    }

    #[test]
    fn test_convert_message_tool_result_single_text() {
        let msg = Message::ToolResult(ToolResultMessage {
            tool_call_id: "call_1".into(),
            tool_name: "bash".into(),
            content: vec![ContentBlock::Text(TextContent {
                text: "output".into(),
                text_signature: None,
            })],
            details: None,
            usage: Some(Usage::default()),
            is_error: false,
            added_tool_names: None,
            timestamp: 0,
        });
        let result = convert_message(&msg);
        assert!(matches!(
            result,
            RequestMessage::Tool {
                tool_call_id: ref id,
                content: MessageContent::Text(ref t),
                ..
            } if id == "call_1" && t == "output"
        ));
    }

    #[test]
    fn test_convert_message_tool_result_multi_block() {
        let msg = Message::ToolResult(ToolResultMessage {
            tool_call_id: "call_2".into(),
            tool_name: "bash".into(),
            content: vec![
                ContentBlock::Text(TextContent {
                    text: "text".into(),
                    text_signature: None,
                }),
                ContentBlock::Image(ImageContent {
                    data: "img".into(),
                    mime_type: "image/png".into(),
                }),
            ],
            details: None,
            usage: Some(Usage::default()),
            is_error: false,
            added_tool_names: None,
            timestamp: 0,
        });
        let result = convert_message(&msg);
        assert!(matches!(
            result,
            RequestMessage::Tool {
                tool_call_id: ref id,
                content: MessageContent::Parts(..),
                ..
            } if id == "call_2"
        ));
    }

    // ═════════════════════════════════════════════════════════════
    //  build_request_body
    // ═════════════════════════════════════════════════════════════

    fn sample_model() -> Model {
        Model {
            id: "deepseek-v4-flash".into(),
            name: None,
            provider: "deepseek".into(),
            api: Api::OpenAiCompletions,
            base_url: "https://api.example.com".into(),
            context_window: 65536,
            max_tokens: 8192,
            reasoning: true,
            headers: None,
            cost_rates: None,
        }
    }

    /// Minimal test Tool: `to_api_definition()` uses the default implementation (assembles name/description/parameters).
    #[derive(Debug)]
    struct MockTool {
        name: &'static str,
        description: &'static str,
        parameters: serde_json::Value,
    }

    #[async_trait::async_trait]
    impl crate::tool::Tool for MockTool {
        fn name(&self) -> &str {
            self.name
        }
        fn description(&self) -> &str {
            self.description
        }
        fn parameters(&self) -> &serde_json::Value {
            &self.parameters
        }
        fn label(&self) -> &str {
            self.name
        }
        async fn execute(
            &self,
            _tool_call_id: &str,
            _args: &serde_json::Value,
            _cancel: &tokio_util::sync::CancellationToken,
            _on_update: crate::tool::ToolUpdateCallback,
        ) -> Result<crate::tool::ToolExecutionResult, crate::tool::ToolError> {
            unimplemented!("test mock")
        }
    }

    fn sample_context() -> Context {
        Context {
            system_prompt: Some("You are a helpful assistant.".into()),
            messages: vec![Message::User(UserMessage {
                content: ContentBlockList::Text("hello".into()),
                timestamp: 0,
            })],
            tools: Some(vec![]),
        }
    }

    #[test]
    fn test_build_request_body_basic() {
        let model = sample_model();
        let ctx = sample_context();
        let req = build_request_body(&model, &ctx, &StreamConfig::default());
        assert_eq!(req.model, "deepseek-v4-flash");
        assert_eq!(req.messages.len(), 2); // system + user
        assert!(req.stream);
        assert!(req.stream_options.is_some());
        assert_eq!(req.max_tokens, Some(8192));
    }

    #[test]
    fn test_build_request_body_with_tools() {
        let model = sample_model();
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message::User(UserMessage {
                content: ContentBlockList::Text("list files".into()),
                timestamp: 0,
            })],
            tools: Some(vec![std::sync::Arc::new(MockTool {
                name: "bash",
                description: "run a shell command",
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": { "type": "string" }
                    }
                }),
            })]),
        };
        let cfg = StreamConfig {
            tool_choice: Some(ToolChoice::Auto),
            ..Default::default()
        };
        let req = build_request_body(&model, &ctx, &cfg);
        assert_eq!(req.messages.len(), 1);
        assert!(req.tools.is_some());
        assert_eq!(req.tools.as_ref().unwrap().len(), 1);
        // Request-boundary tool definitions are derived from executable tools' to_api_definition()
        assert_eq!(req.tools.as_ref().unwrap()[0].function.name, "bash");
        assert_eq!(req.tool_choice, Some(ToolChoice::Auto));
    }

    #[test]
    fn test_build_request_body_no_system_prompt() {
        let model = sample_model();
        let ctx = Context {
            system_prompt: None,
            messages: vec![],
            tools: None,
        };
        let req = build_request_body(&model, &ctx, &StreamConfig::default());
        assert_eq!(req.messages.len(), 0);
    }

    // ═════════════════════════════════════════════════════════════
    //  parse_chunk
    // ═════════════════════════════════════════════════════════════

    #[test]
    fn test_parse_chunk_empty() {
        let events = parse_chunk(&serde_json::json!({}));
        assert!(events.is_empty());
    }

    #[test]
    fn test_parse_chunk_first_chunk() {
        // role only, no content, no tool_calls → no events (text block is lazy)
        let chunk = serde_json::json!({
            "id": "chatcmpl-1",
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant" },
                "finish_reason": null
            }]
        });
        let events = parse_chunk(&chunk);
        assert_eq!(events.len(), 0, "role alone should not emit events");
    }

    #[test]
    fn test_parse_chunk_text_delta() {
        let chunk = serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": { "content": "Hello" },
                "finish_reason": null
            }]
        });
        let events = parse_chunk(&chunk);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            StreamEvent::TextDelta {
                text: ref t, ..
            } if t == "Hello"
        ));
    }

    #[test]
    fn test_parse_chunk_thinking_delta() {
        let chunk = serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": { "reasoning_content": "thinking..." },
                "finish_reason": null
            }]
        });
        let events = parse_chunk(&chunk);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            StreamEvent::ThinkingDelta {
                thinking: ref t, ..
            } if t == "thinking..."
        ));
    }

    #[test]
    fn test_parse_chunk_tool_call_start() {
        let chunk = serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": { "name": "bash", "arguments": "" }
                    }]
                },
                "finish_reason": null
            }]
        });
        let events = parse_chunk(&chunk);
        assert_eq!(events.len(), 1); // ContentBlockStart only; arguments="" is filtered out
        assert!(matches!(
            events[0],
            StreamEvent::ContentBlockStart {
                block_type: ref t,
                id: Some(ref id),
                name: Some(ref n),
                ..
            } if t == "tool_use" && id == "call_1" && n == "bash"
        ));
    }

    #[test]
    fn test_parse_chunk_finish_reason() {
        let chunk = serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        });
        let events = parse_chunk(&chunk);
        assert_eq!(events.len(), 2); // ContentBlockStop + MessageDelta
        assert!(matches!(events[0], StreamEvent::ContentBlockStop { .. }));
        assert!(matches!(
            events[1],
            StreamEvent::MessageDelta {
                stop_reason: Some(StopReason::Stop),
                usage: None,
            }
        ));
    }

    #[test]
    fn test_parse_chunk_with_usage() {
        let chunk = serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        });
        let events = parse_chunk(&chunk);
        // MessageDelta(usage) + ContentBlockStop + MessageDelta(stop_reason)
        // usage is now processed before choices, so the index is 0
        assert_eq!(events.len(), 3);
        assert!(matches!(
            events[0],
            StreamEvent::MessageDelta {
                stop_reason: None,
                usage: Some(ref u),
            } if u.input == 10 && u.output == 20
        ));
    }

    #[test]
    fn test_parse_chunk_usage_only() {
        // Some APIs only return usage in the final chunk, with choices as an empty array.
        // See https://platform.openai.com/docs/api-reference/streaming#chat-streaming-usage
        let chunk = serde_json::json!({
            "choices": [],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 25,
                "total_tokens": 40
            }
        });
        let events = parse_chunk(&chunk);
        assert_eq!(events.len(), 1, "should only emit usage MessageDelta");
        assert!(matches!(
            events[0],
            StreamEvent::MessageDelta {
                stop_reason: None,
                usage: Some(ref u),
            } if u.input == 15 && u.output == 25
        ));
    }

    #[test]
    fn test_parse_chunk_no_usage() {
        // Regular chunk, no usage field
        let chunk = serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": {"content": "hello"}
            }]
        });
        let events = parse_chunk(&chunk);
        // Only TextDelta, no MessageDelta
        assert!(events
            .iter()
            .all(|e| !matches!(e, StreamEvent::MessageDelta { .. })));
    }

    // ═════════════════════════════════════════════════════════════
    //  StreamAccumulator (process_event)
    // ═════════════════════════════════════════════════════════════

    #[test]
    fn test_accumulator_text_block() {
        let mut acc = StreamAccumulator::default();
        acc.process_event(StreamEvent::ContentBlockStart {
            index: 0,
            block_type: "text".into(),
            id: None,
            name: None,
        });
        acc.process_event(StreamEvent::TextDelta {
            index: 0,
            text: "Hello".into(),
        });
        acc.process_event(StreamEvent::TextDelta {
            index: 0,
            text: " world".into(),
        });
        acc.process_event(StreamEvent::ContentBlockStop { index: 0 });
        assert_eq!(acc.content_blocks.len(), 1);
        if let ContentBlock::Text(t) = &acc.content_blocks[0] {
            assert_eq!(t.text, "Hello world");
        } else {
            panic!("expected Text block");
        }
    }

    #[test]
    fn test_accumulator_tool_call_block() {
        let mut acc = StreamAccumulator::default();
        acc.process_event(StreamEvent::ContentBlockStart {
            index: 0,
            block_type: "tool_use".into(),
            id: Some("call_1".into()),
            name: Some("bash".into()),
        });
        acc.process_event(StreamEvent::InputJsonDelta {
            index: 0,
            partial_json: "{\"command\": ".into(),
        });
        acc.process_event(StreamEvent::InputJsonDelta {
            index: 0,
            partial_json: "\"ls\"}".into(),
        });
        acc.process_event(StreamEvent::ContentBlockStop { index: 0 });
        assert_eq!(acc.content_blocks.len(), 1);
        if let ContentBlock::ToolCall(tc) = &acc.content_blocks[0] {
            assert_eq!(tc.id, "call_1");
            assert_eq!(tc.name, "bash");
            assert_eq!(tc.arguments, serde_json::json!({"command": "ls"}));
        } else {
            panic!("expected ToolCall block");
        }
    }

    #[test]
    fn test_accumulator_thinking_block() {
        let mut acc = StreamAccumulator::default();
        acc.process_event(StreamEvent::ContentBlockStart {
            index: 0,
            block_type: "thinking".into(),
            id: None,
            name: None,
        });
        acc.process_event(StreamEvent::ThinkingDelta {
            index: 0,
            thinking: "step 1...".into(),
        });
        acc.process_event(StreamEvent::ContentBlockStop { index: 0 });
        assert_eq!(acc.content_blocks.len(), 1);
        if let ContentBlock::Thinking(t) = &acc.content_blocks[0] {
            assert_eq!(t.thinking, "step 1...");
        } else {
            panic!("expected Thinking block");
        }
    }

    #[test]
    fn test_accumulator_message_delta_stop_reason() {
        let mut acc = StreamAccumulator::default();
        assert!(!acc.has_finish_reason);
        acc.process_event(StreamEvent::MessageDelta {
            stop_reason: Some(StopReason::ToolUse),
            usage: None,
        });
        assert_eq!(acc.stop_reason, Some(StopReason::ToolUse));
        assert!(acc.has_finish_reason);
    }

    #[test]
    fn test_accumulator_message_delta_usage() {
        let mut acc = StreamAccumulator::default();
        let usage = Usage {
            input: 10,
            output: 20,
            total_tokens: 30,
            ..Usage::default()
        };
        acc.process_event(StreamEvent::MessageDelta {
            stop_reason: None,
            usage: Some(usage.clone()),
        });
        assert_eq!(acc.usage.input, 10);
        assert_eq!(acc.usage.output, 20);
    }

    #[test]
    fn test_accumulator_unknown_block_type_ignored() {
        let mut acc = StreamAccumulator::default();
        acc.process_event(StreamEvent::ContentBlockStart {
            index: 0,
            block_type: "unknown".into(),
            id: None,
            name: None,
        });
        acc.process_event(StreamEvent::ContentBlockStop { index: 0 });
        assert!(acc.content_blocks.is_empty());
    }
}
