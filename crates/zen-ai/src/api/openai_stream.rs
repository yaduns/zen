use crate::catalog::Catalog;
use crate::provider::{
    MaxTokensField, OpenAiCompletionsCompat, SessionAffinityFormat, ThinkingFormat,
};
use crate::types::{
    AssistantMessage, AssistantMessageEvent, ContentBlock, ContentBlockList, Context,
    ErrorStopReason, Message, Model, NonErrorStopReason, StopReason, StreamFn, TextContent,
    ThinkingContent, ThinkingLevel, ToolCall, ToolChoice, ToolDefinition, ToolFunction, Usage,
};
use crate::utils::error_body::{extract_error_body, extract_readable_error_message};
use eventsource_stream::Eventsource;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio_util::sync::CancellationToken;

/// Optional configuration parameters for the stream() function, controlling non-functional behavior of the API call (authentication, timeouts, tool-selection policy, thinking mode, etc.).
#[derive(Default, Clone)]
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

    #[error("API error: status {status}: {message}")]
    ApiError {
        status: u16,
        /// Human-readable error summary (prefers the provider's `error.message`, falls back to the raw body).
        message: String,
    },

    #[error("SSE parse error: {0}")]
    SSEError(String),

    #[error("Incomplete stream: {0}")]
    IncompleteStream(String),

    #[error("Request cancelled")]
    Cancelled,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
struct Request {
    /// Model ID (e.g. `"deepseek-v4-flash"`).
    model: String,
    /// Message array.
    messages: Vec<RequestMessage>,
    /// Tool definition list, set only when `context.tools` is non-empty and the message history contains tool calls/results.
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
    /// Tool selection policy (in the OpenAI protocol shape, converted via `tool_choice_to_request`).
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    /// Maximum output token count (used when `maxTokensField == "max_tokens"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
    /// Maximum output token count (used when `maxTokensField == "max_completion_tokens"`, for OpenAI o-series/gpt-5).
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u64>,
    /// Whether to use streaming responses. Always `true` for zen.
    stream: bool,
    /// Streaming options, including `include_usage`. Not serialized when `supportsUsageInStreaming == false`.
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
    /// Reasoning effort (API string value, converted from ThinkingLevel; subject to dual gating by `supportsReasoningEffort` and
    /// `model.reasoning`).
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    /// Thinking toggle for DeepSeek-family endpoints (`thinking: {type: "enabled"/"disabled"}`), appended based on the
    /// `compat.thinkingFormat` format family (aligned with pi openai-completions.ts:664-673);
    /// Not sent in openai format (opencode-go).
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingParam>,
    /// Session affinity cache key (OpenAI `prompt_cache_key` request field); omitted entirely when None.
    /// Value = `StreamConfig.session_id` truncated by `clamp_openai_prompt_cache_key` (≤64 code points),
    /// so the server routes requests for the same session to the same cache shard (aligns with pi buildParams parameter).
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
}

/// Internal structure of the `thinking` toggle parameter (DeepSeek-family endpoints): `{type: "enabled"}` or `{type: "disabled"}`.
#[derive(Debug, Clone, Serialize)]
struct ThinkingParam {
    r#type: String,
}

#[derive(Debug, Clone, Serialize)]
struct StreamOptionsParam {
    /// Whether to return usage information in the final chunk, gated by `compat.supportsUsageInStreaming` (default `true`).
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
    /// Developer message (replaces `system` when `supportsDeveloperRole` is enabled and the model supports reasoning).
    Developer {
        /// Developer instruction content.
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
    /// Fallback thinking field (opencode-go endpoints actually return `reasoning`, a non-standard name).
    #[serde(default)]
    reasoning: Option<String>,
    /// Fallback reasoning field (some compatible endpoints return `reasoning_text`).
    #[serde(default)]
    reasoning_text: Option<String>,
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
/// # Edge Cases
/// This function is a low-level direct-call entry point and **does not append provider attribution headers** (such as the
/// `x-opencode-session` header enforced by the opencode gateway)—it must go through the app assembly layer (`zen_app::session::provider_attribution`)
/// or provide them yourself in `StreamConfig.headers`; for the api key, `StreamConfig.api_key` takes precedence,
/// Falls back to `Catalog::api_key`.
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

    // 1. API Key (`cfg.api_key` takes precedence; fall back to `Catalog::current().api_key(model)` as a data-backed fallback;
    //    No longer reads env by provider name—see `api_key_for_request`)
    let api_key = api_key_for_request(model, &cfg).ok_or(StreamError::NoApiKey)?;

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

    // Merge provider self-declared headers + caller-explicit headers (same-name: the latter wins; single-value headers, see merge_request_headers)
    for (k, v) in &merge_request_headers(model, &cfg) {
        req = req.header(k.as_str(), v.as_str());
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
        // Log the full raw body at debug level (visible with RUST_LOG=debug); Display only gives the readable summary,
        // to avoid printing the entire JSON to the terminal (a single TUI line would be truncated; see the wrapping fix in assistant_message/ui).
        tracing::debug!(status, body = %body, "[zen-ai] HTTP error response body");
        let message = extract_readable_error_message(&body);
        return Err(StreamError::ApiError { status, message });
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

/// Gets `model.compat` (optional); returns `None` when absent, letting each accessor apply the protocol default.
fn compat_of(model: &Model) -> Option<&OpenAiCompletionsCompat> {
    model.compat.as_ref()
}

/// Called whenever `build_request_body` constructs a request: resolves `model.compat.thinking_format` (users can
/// change it in models.json) into a thinking-format family, so DeepSeek-family endpoints no longer guess based on the provider name.
/// Defaults to `ThinkingFormat::OpenAi` (reasoning_effort only; does not send the thinking toggle).
fn thinking_format_of(model: &Model) -> ThinkingFormat {
    compat_of(model)
        .and_then(|compat| compat.thinking_format)
        .unwrap_or(ThinkingFormat::OpenAi)
}

/// Called during request construction: determines which field name to use for `max_tokens` (`max_tokens` or `max_completion_tokens`),
/// Supports standard compatible endpoints and both OpenAI o-series and gpt-5 conventions. Defaults to `MaxTokensField::MaxTokens`.
fn max_tokens_field_of(model: &Model) -> MaxTokensField {
    compat_of(model)
        .and_then(|compat| compat.max_tokens_field)
        .unwrap_or_default()
}

/// Called during request construction: determines whether to send `stream_options:{include_usage:true}` (unsupported by endpoints without usage).
/// Defaults to `true` (preserving zen's existing behavior).
fn supports_usage_in_streaming(model: &Model) -> bool {
    compat_of(model)
        .and_then(|compat| compat.supports_usage_in_streaming)
        .unwrap_or(true)
}

/// Request construction call: Determines whether the system prompt uses the `developer` role instead (`model.reasoning` and when the endpoint supports it).
/// Defaults to `false` (conservatively use `system`, consistent with existing behavior).
fn supports_developer_role(model: &Model) -> bool {
    compat_of(model)
        .and_then(|compat| compat.supports_developer_role)
        .unwrap_or(false)
}

/// Request construction call: Determines whether to send `reasoning_effort` (endpoints that do not support this parameter will error if it is sent).
/// Defaults to `true`.
fn supports_reasoning_effort(model: &Model) -> bool {
    compat_of(model)
        .and_then(|compat| compat.supports_reasoning_effort)
        .unwrap_or(true)
}

/// Request construction call: Determines whether to append the DeepSeek-family `thinking:{type}` toggle based on `compat.thinking_format`.
/// `reasoning_effort` is the converted API string (`None` = disabled). If not `model.reasoning` or not DeepSeek
/// format → `None` (aligns with the pi deepseek branch, which requires `model.reasoning`).
fn build_thinking_param(model: &Model, reasoning_effort: Option<&str>) -> Option<ThinkingParam> {
    if !model.reasoning {
        return None;
    }
    match thinking_format_of(model) {
        ThinkingFormat::DeepSeek => Some(ThinkingParam {
            r#type: if reasoning_effort.is_some() {
                "enabled".to_string()
            } else {
                "disabled".to_string()
            },
        }),
        ThinkingFormat::OpenAi => None,
    }
}

/// Request construction call: Produces the message for the system prompt (`system` or `developer` role),
/// Determined by `model.reasoning && compat.supports_developer_role`.
fn system_request_message(model: &Model, prompt: &str) -> RequestMessage {
    if model.reasoning && supports_developer_role(model) {
        RequestMessage::Developer {
            content: prompt.to_string(),
        }
    } else {
        RequestMessage::System {
            content: prompt.to_string(),
        }
    }
}

/// Converts the internal `ToolChoice` to the OpenAI protocol shape (`"auto"`/`"none"`/`"required"` or
/// `{"type":"function","function":{"name":...}}`）——`ToolChoice`'s derive serialization is externally tagged
/// (`"Auto"` / `{"Function":...}`) cannot be sent directly, so they are explicitly converted at the request boundary.
fn tool_choice_to_request(choice: &ToolChoice) -> serde_json::Value {
    match choice {
        ToolChoice::Auto => serde_json::Value::String("auto".to_string()),
        ToolChoice::None => serde_json::Value::String("none".to_string()),
        ToolChoice::Required => serde_json::Value::String("required".to_string()),
        ToolChoice::Function { name } => serde_json::json!({
            "type": "function",
            "function": { "name": name },
        }),
    }
}

/// Request header merge call: based on `compat.send_session_affinity_headers` / `session_affinity_format`
/// Generates session affinity headers. Header-name literals follow pi `buildHeaders` (`openai-completions.ts:551-560`):
/// When `sessionId && sendSessionAffinityHeaders`, `sessionAffinityFormat == "openrouter"` → only
/// `x-session-id`; otherwise, `openai` additionally includes `session_id`; both cases include `x-client-request-id`/
/// `x-session-affinity`. If `session_id` is empty/missing or the switch is disabled → empty map (do not fabricate a session ID).
fn session_affinity_headers(model: &Model, session_id: Option<&str>) -> HashMap<String, String> {
    let mut headers = HashMap::new();
    let Some(compat) = compat_of(model) else {
        return headers;
    };
    if !compat.send_session_affinity_headers.unwrap_or(false) {
        return headers;
    }
    let Some(session_id) = session_id.filter(|id| !id.is_empty()) else {
        return headers;
    };
    match compat.session_affinity_format.unwrap_or_default() {
        SessionAffinityFormat::Openrouter => {
            headers.insert("x-session-id".to_string(), session_id.to_string());
        }
        SessionAffinityFormat::Openai => {
            headers.insert("session_id".to_string(), session_id.to_string());
            headers.insert("x-client-request-id".to_string(), session_id.to_string());
            headers.insert("x-session-affinity".to_string(), session_id.to_string());
        }
        SessionAffinityFormat::OpenaiNosession => {
            headers.insert("x-client-request-id".to_string(), session_id.to_string());
            headers.insert("x-session-affinity".to_string(), session_id.to_string());
        }
    }
    headers
}

/// Called before `stream()` sends the request: resolves the Bearer key for this request—`cfg.api_key` (injected by the app/agent loop, preferred)
/// Otherwise fall back to `Catalog::current().api_key(model)` (so `stream_collect` direct calls and compaction/branch summary paths
/// have no gap; the data source is the same one used by the selection layer, including the user's `models.json`). If both are empty → `None`, reported by the call site
/// `StreamError::NoApiKey`。
/// **Keyless endpoint decision**: the v1 request layer always requires a Bearer key; it does not implement "do not send Authorization when apiKey is absent";
/// The "keyless endpoint" wording for `Catalog::api_key` only describes its None semantics (see open question 3).
fn api_key_for_request(model: &Model, cfg: &StreamConfig) -> Option<String> {
    cfg.api_key
        .clone()
        .or_else(|| Catalog::current().api_key(model))
}

fn build_request_body(model: &Model, context: &Context, cfg: &StreamConfig) -> Request {
    let mut messages = Vec::new();

    // System prompt as the first message (the `developer`/`system` role is determined by compat and model.reasoning)
    if let Some(ref prompt) = context.system_prompt {
        messages.push(system_request_message(model, prompt));
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

    // reasoning_effort dual gating: model must support reasoning and the endpoint must accept the parameter (aligned with the pi conditional branch).
    let reasoning_effort = if model.reasoning && supports_reasoning_effort(model) {
        cfg.reasoning_effort.and_then(thinking_level_to_api_value)
    } else {
        None
    };

    // Session affinity cache key: sent whenever `session_id` is present (pi's baseUrl whitelist condition gate
    // `baseUrl.includes("api.openai.com") || cacheRetention=="long"` dropped alongside multi-provider quirks—
    // zen's default endpoint is not openai.com; mechanical copying = never sends = dead code; endpoint's parameter acceptance has been verified on real hardware
    // tolerated for 200, see tasks/T-AI-02-6-implemented.md "Implementation prerequisites"). When None, the whole thing is not serialized,
    // zero change to the default-path request body.
    let prompt_cache_key =
        crate::api::prompt_cache::clamp_openai_prompt_cache_key(cfg.session_id.as_deref());

    // Thinking switch: appended according to compat.thinkingFormat (enabled for DeepSeek models → {type:"enabled"},
    // Off → {type:"disabled"}; not sent for openai formats such as opencode-go).
    let thinking = build_thinking_param(model, reasoning_effort.as_deref());

    // The max_tokens field name is determined by compat.maxTokensField (one of two; the other is None).
    let max_tokens_value = cfg.max_tokens.or(Some(model.max_tokens));
    let (max_tokens, max_completion_tokens) = match max_tokens_field_of(model) {
        MaxTokensField::MaxTokens => (max_tokens_value, None),
        MaxTokensField::MaxCompletionTokens => (None, max_tokens_value),
    };

    // stream_options is gated by compat.supportsUsageInStreaming (defaults to true).
    let stream_options = if supports_usage_in_streaming(model) {
        Some(StreamOptionsParam {
            include_usage: true,
        })
    } else {
        None
    };

    Request {
        model: model.id.clone(),
        messages,
        tools,
        tool_choice: cfg.tool_choice.as_ref().map(tool_choice_to_request),
        max_tokens,
        max_completion_tokens,
        stream: true,
        stream_options,
        temperature: cfg.temperature,
        top_p: None,
        stop: None,
        reasoning_effort,
        thinking,
        prompt_cache_key,
    }
}

/// Merge the HTTP headers to add to this request: start with `model.headers` (static catalog headers), then layer in compat session affinity headers,
/// Then layer in the caller's explicit headers—**later values with the same name override**. Why merge into a map first instead of calling `.header()` twice:
/// reqwest `RequestBuilder::header()` uses `HeaderMap::append`; setting the same header twice sends **multiple values** instead of overriding;
/// After merging into a map, each key is set only once, so override semantics hold. Returning an empty map = no additional headers.
/// Provider attribution headers (e.g., opencode `x-opencode-session`) are not generated at the request layer, but by the app assembly layer via
/// `cfg.headers` injection (see `zen-app::session::provider_attribution`).
fn merge_request_headers(model: &Model, cfg: &StreamConfig) -> HashMap<String, String> {
    let mut merged = model.headers.clone().unwrap_or_default();
    merged.extend(session_affinity_headers(model, cfg.session_id.as_deref()));
    if let Some(headers) = &cfg.headers {
        for (key, value) in headers {
            merged.insert(key.clone(), value.clone());
        }
    }
    merged
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

/// Parse a single SSE chunk's JSON and convert it to a `StreamEvent` domain event.
///
/// Convert OpenAI's chunk format (`choices[0].delta.{content,reasoning_content,reasoning,reasoning_text,tool_calls}`)
/// Mapped into the `StreamEvent` enum for `StreamAccumulator` to consume.
/// `provider` is passed in for thinking signature normalization special-casing (opencode-go maps `reasoning` hit to the canonical name
/// `reasoning_content`, aligned with pi openai-completions.ts:412-416).
///
/// Handles:
/// - First chunk (`choices[0].delta.role === "assistant"`) → `ContentBlockStart { block_type: "text" }`
/// - Regular chunk (`delta.content` non-empty) → `TextDelta`
/// - thinking chunk (first non-empty of `delta.reasoning_content` / `reasoning` / `reasoning_text`) → `ThinkingDelta`
/// - First tool call chunk (`delta.tool_calls[0].id` non-empty) → `ContentBlockStart { block_type: "tool_use" }` + `InputJsonDelta`
/// - Tool call incremental chunk (`delta.tool_calls[0].function.arguments` non-empty) → `InputJsonDelta`
/// - Final chunk (`choices[0].finish_reason` non-empty) → `ContentBlockStop` + `MessageDelta`
/// - usage chunk (`usage` non-empty, may exist independently of choices) → `MessageDelta`
///
/// Note: `usage` is processed before `choices` because some APIs return usage as a separate chunk with `choices: []`.
///
/// Combined with Pi's internal functions such as `ensureTextBlock` / `ensureThinkingBlock` / `finishBlock`
/// chunk parsing logic.
fn parse_chunk(chunk: &serde_json::Value, provider: &str) -> Vec<StreamEvent> {
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

    // Thinking delta: try reasoning_content / reasoning / reasoning_text in order, take the first non-empty field
    // (aligned with pi openai-completions.ts:394-407; opencode-go endpoints actually return delta.reasoning,
    // not reading it silently drops thinking deltas — defect A of T-thinking-invisible-fix).
    const REASONING_FIELDS: [&str; 3] = ["reasoning_content", "reasoning", "reasoning_text"];
    let thinking_field = REASONING_FIELDS
        .iter()
        .copied()
        .find(|f| delta[*f].as_str().is_some_and(|s| !s.is_empty()));
    if let Some(field) = thinking_field {
        let thinking = delta[field].as_str().unwrap_or_default().to_string();
        // Same signature normalization as pi: opencode-go maps reasoning hit to canonical name reasoning_content
        // (openai-completions.ts:412-416; when writing back to history at the end, the endpoint expects the canonical name, lines 1009-1013).
        let signature = if provider == "opencode-go" && field == "reasoning" {
            "reasoning_content".to_string()
        } else {
            field.to_string()
        };
        events.push(StreamEvent::ThinkingDelta {
            index: idx,
            thinking,
            signature,
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
///
/// `cache_read` dual-field fallback (aligned with pi's `parseChunkUsage`, openai-completions.ts:1191):
/// OpenAI shape `prompt_tokens_details.cached_tokens` takes priority, DeepSeek official shape's top-level
/// `prompt_cache_hit_tokens` as fallback. **Nullish semantics: explicit 0 does not fall back** — when `cached_tokens` is 0,
/// do not read `prompt_cache_hit_tokens`; Rust uses a `.or_else` chain (`.as_u64()` returns None for missing,
/// Some(0) for explicit 0, naturally matching pi's `??`), **forbidden** to write as two chained short-circuiting `.unwrap_or(0)`.
fn parse_usage(usage: &serde_json::Value) -> (u64, u64, u64, u64, Option<u64>) {
    let input = usage["prompt_tokens"].as_u64().unwrap_or(0);
    let output = usage["completion_tokens"].as_u64().unwrap_or(0);
    let total = usage["total_tokens"].as_u64().unwrap_or(0);
    let cache_read = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .or_else(|| usage["prompt_cache_hit_tokens"].as_u64())
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
#[derive(Clone)]
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
    // Thinking/reasoning content delta. `signature` is the response field name that was hit (canonical name after the opencode-go special case)
    /// `reasoning_content`）。
    ThinkingDelta {
        index: usize,
        thinking: String,
        signature: String,
    },
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
/// - `content_blocks` stores only finalized complete blocks (written in ascending internal index order on `ContentBlockStop`)
/// - `partial_*` maps store only incomplete data, indexed by **internal block index** (unified allocator, eliminating the text/tool index 0 conflict)
/// - `process_event()` is a pure accumulating operation and returns no events; boundary event pushes are handled by `handle_events`
#[derive(Default)]
struct StreamAccumulator {
    /// Final output content blocks (only finalized blocks are here). finalize appends in ascending internal index order, preserving array order
    content_blocks: Vec<ContentBlock>,
    /// Next available internal block index (monotonically increasing; the internal index is the content array position)
    next_block_seq: usize,
    /// Internal index of the text block (API `content` stream), lazily allocated on the first text delta
    content_seq: Option<usize>,
    /// Internal index of the thinking block (API `reasoning_content`/`reasoning`/`reasoning_text` stream), lazily allocated on the first thinking delta
    thinking_seq: Option<usize>,
    /// Thinking block signature (field name of the first ThinkingDelta; opencode-go normalized to reasoning_content).
    /// Effective on first delta, later deltas do not override (aligned with pi ensureThinkingBlock binding the first signature).
    thinking_signature: Option<String>,
    /// API tool index → internal index (each multi-tool call gets its own independent seq, not conflicting with text/thinking)
    tool_seqs: HashMap<usize, usize>,
    /// Incomplete text indexed by internal index
    partial_text: HashMap<usize, String>,
    /// Incomplete thinking content indexed by internal index
    partial_thinking: HashMap<usize, String>,
    /// Incomplete tool call argument JSON fragments indexed by internal index
    partial_json: HashMap<usize, String>,
    /// Block type (text/thinking/tool_use) indexed by internal index
    block_types: HashMap<usize, String>,
    /// Tool call IDs indexed by internal index
    tool_use_ids: HashMap<usize, String>,
    /// Tool names indexed by internal index
    tool_use_names: HashMap<usize, String>,
    /// Stop reason (set by `MessageDelta` events)
    stop_reason: Option<StopReason>,
    /// Usage statistics (set by `MessageDelta` events)
    usage: Usage,
    /// Whether a finish_reason has been received
    has_finish_reason: bool,
}

impl StreamAccumulator {
    /// Allocate the next internal block index.
    fn alloc_seq(&mut self) -> usize {
        let seq = self.next_block_seq;
        self.next_block_seq += 1;
        seq
    }

    /// Resolve the internal index for the text stream (lazily allocated).
    fn resolve_content(&mut self) -> usize {
        if let Some(seq) = self.content_seq {
            seq
        } else {
            let seq = self.alloc_seq();
            self.content_seq = Some(seq);
            seq
        }
    }

    /// Resolve the internal index of the thinking stream (allocated lazily).
    fn resolve_thinking(&mut self) -> usize {
        if let Some(seq) = self.thinking_seq {
            seq
        } else {
            let seq = self.alloc_seq();
            self.thinking_seq = Some(seq);
            seq
        }
    }

    /// Map an API tool index to an internal index (allocated on first appearance).
    fn resolve_tool(&mut self, api_index: usize) -> usize {
        if let Some(seq) = self.tool_seqs.get(&api_index) {
            *seq
        } else {
            let seq = self.alloc_seq();
            self.tool_seqs.insert(api_index, seq);
            seq
        }
    }

    /// Internal index of the text block (None when no text delta occurred).
    fn content_seq_value(&self) -> Option<usize> {
        self.content_seq
    }

    /// Internal index of the thinking block (None when no thinking delta occurred).
    fn thinking_seq_value(&self) -> Option<usize> {
        self.thinking_seq
    }

    /// Consumes a `StreamEvent` and updates the accumulator's internal state.
    /// Emits no events — event delivery is handled by `send()` in `sse_event_loop` / `handle_events`.
    fn process_event(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::ContentBlockStart {
                index,
                block_type,
                id,
                name,
            } => {
                // Tool blocks register their seq at start; text/thinking start events are not sent in the real stream
                // (blocks are built lazily); this branch is only for tests that construct events directly.
                let seq = match block_type.as_str() {
                    "tool_use" => self.resolve_tool(index),
                    "text" => self.resolve_content(),
                    "thinking" => self.resolve_thinking(),
                    _ => self.alloc_seq(), // unknown: discard at finalize time if the type is unrecognized
                };
                self.block_types.insert(seq, block_type);
                // Only update id/name if non-empty. Some providers (e.g. DeepSeek)
                // send subsequent chunks with id="" and name="" which would
                // overwrite the correct values from the first chunk.
                if let Some(id) = id {
                    if !id.is_empty() {
                        self.tool_use_ids.insert(seq, id);
                    }
                }
                if let Some(name) = name {
                    if !name.is_empty() {
                        self.tool_use_names.insert(seq, name);
                    }
                }
            }

            StreamEvent::TextDelta { index: _, text } => {
                let seq = self.resolve_content();
                // Lazily create a text block if not already started at this index.
                // This matches Pi's ensureTextBlock() pattern: text blocks are created
                // when text content actually arrives, not when role: "assistant" is seen.
                self.block_types
                    .entry(seq)
                    .or_insert_with(|| "text".to_string());
                self.partial_text.entry(seq).or_default().push_str(&text);
            }

            StreamEvent::ThinkingDelta {
                index: _,
                thinking,
                signature,
            } => {
                let seq = self.resolve_thinking();
                self.block_types
                    .entry(seq)
                    .or_insert_with(|| "thinking".to_string());
                if self.thinking_signature.is_none() {
                    self.thinking_signature = Some(signature);
                }
                self.partial_thinking
                    .entry(seq)
                    .or_default()
                    .push_str(&thinking);
            }

            StreamEvent::InputJsonDelta {
                index,
                partial_json,
            } => {
                let seq = self.resolve_tool(index);
                self.partial_json
                    .entry(seq)
                    .or_default()
                    .push_str(&partial_json);
            }

            StreamEvent::ContentBlockStop { index: _ } => {
                // Semantics change (T-AI-02-1-fix): current `parse_chunk` emits this event only once at finish_reason, semantics =
                // finalize **all** active blocks (fixes the bug where multi-tool calls only finalized index 0).
                // If mid-stream single-block completion semantics are introduced later, per-index block finalization must be restored.
                self.finalize_all();
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

    /// Terminal finalization: finalize all still-active blocks into `content_blocks` in ascending internal index order,
    /// returning a list of `(internal index, complete block)` for pushing the End event. Ascending internal index equals content array order.
    fn finalize_all(&mut self) -> Vec<(usize, ContentBlock)> {
        let mut seqs: Vec<usize> = self.block_types.keys().copied().collect();
        seqs.sort_unstable();
        let mut out = Vec::with_capacity(seqs.len());
        for seq in seqs {
            let block_type = self.block_types.remove(&seq).unwrap_or_default();
            let block = match block_type.as_str() {
                "text" => {
                    let text = self.partial_text.remove(&seq).unwrap_or_default();
                    ContentBlock::Text(TextContent {
                        text,
                        text_signature: None,
                    })
                }
                "thinking" => {
                    let thinking = self.partial_thinking.remove(&seq).unwrap_or_default();
                    ContentBlock::Thinking(ThinkingContent {
                        thinking,
                        thinking_signature: self.thinking_signature.clone(),
                        redacted: None,
                    })
                }
                "tool_use" => {
                    let id = self.tool_use_ids.remove(&seq).unwrap_or_default();
                    let name = self.tool_use_names.remove(&seq).unwrap_or_default();
                    let json = self.partial_json.remove(&seq).unwrap_or_default();
                    let arguments = serde_json::from_str(&json)
                        .unwrap_or(serde_json::Value::Object(Default::default()));
                    ContentBlock::ToolCall(ToolCall {
                        id,
                        name,
                        arguments,
                        thought_signature: None,
                    })
                }
                _ => continue,
            };
            self.content_blocks.push(block.clone());
            out.push((seq, block));
        }
        out
    }

    /// Build the current partial `AssistantMessage` snapshot (borrowed clone; does not consume the accumulator).
    /// content = finalized blocks (in order) + active blocks in ascending internal index order.
    /// Partial field used for incremental events (Text/Thinking/Toolcall Start/Delta/End).
    fn partial_snapshot(&self, model: &Model, response_id: Option<String>) -> AssistantMessage {
        let mut seqs: Vec<usize> = self.block_types.keys().copied().collect();
        seqs.sort_unstable();
        let mut content = self.content_blocks.clone();
        for seq in seqs {
            let block = match self.block_types.get(&seq).map(|t| t.as_str()) {
                Some("text") => ContentBlock::Text(TextContent {
                    text: self.partial_text.get(&seq).cloned().unwrap_or_default(),
                    text_signature: None,
                }),
                Some("thinking") => ContentBlock::Thinking(ThinkingContent {
                    thinking: self.partial_thinking.get(&seq).cloned().unwrap_or_default(),
                    thinking_signature: self.thinking_signature.clone(),
                    redacted: None,
                }),
                Some("tool_use") => ContentBlock::ToolCall(ToolCall {
                    id: self.tool_use_ids.get(&seq).cloned().unwrap_or_default(),
                    name: self.tool_use_names.get(&seq).cloned().unwrap_or_default(),
                    arguments: serde_json::from_str(
                        self.partial_json
                            .get(&seq)
                            .map(|s| s.as_str())
                            .unwrap_or("{}"),
                    )
                    .unwrap_or(serde_json::Value::Object(Default::default())),
                    thought_signature: None,
                }),
                _ => continue,
            };
            content.push(block);
        }
        AssistantMessage {
            content,
            api: model.api.to_string(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            response_model: None,
            response_id,
            usage: self.usage.clone(),
            stop_reason: self.stop_reason.clone().unwrap_or(StopReason::Stop),
            error_message: None,
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }
}

/// Process a batch of `StreamEvent`s (results of `parse_chunk` from a single SSE chunk):
/// Push incremental `AssistantMessageEvent`s at block boundaries (lazy Start / per-block Delta / batched End),
/// and accumulate into `accumulator`. Called per chunk by the Phase 3 loop of `sse_event_loop`.
///
/// Ordering convention between pushing and accumulating (aligned with pi's `stream.push` model, openai-completions.ts:228/251/282/294):
/// - Lazily emit `TextStart`/`ThinkingStart` before the first text/thinking delta (partial includes empty block, aligned with `ensureTextBlock`);
/// - Emit `ToolcallStart` after the tool block's `ContentBlockStart` (partial includes empty tool block);
/// - After accumulating each Delta, emit the corresponding Delta event (partial includes the post-accumulation snapshot);
/// - On the finish batch (`ContentBlockStop`), finalize all active blocks and emit End for each (partial includes the complete blocks).
fn handle_events(
    events: &[StreamEvent],
    accumulator: &mut StreamAccumulator,
    sender: &tokio::sync::mpsc::UnboundedSender<AssistantMessageEvent>,
    model: &Model,
    response_id: &Option<String>,
) {
    for ev in events {
        // Lazy start: first delta of a text/thinking stream (internal index not yet assigned = block's first appearance)
        match ev {
            StreamEvent::TextDelta { .. } if accumulator.content_seq_value().is_none() => {
                let seq = accumulator.resolve_content();
                accumulator
                    .block_types
                    .entry(seq)
                    .or_insert_with(|| "text".to_string());
                let partial = accumulator.partial_snapshot(model, response_id.clone());
                let _ = sender.send(AssistantMessageEvent::TextStart {
                    content_index: seq as u32,
                    partial,
                });
            }
            StreamEvent::ThinkingDelta { .. } if accumulator.thinking_seq_value().is_none() => {
                let seq = accumulator.resolve_thinking();
                accumulator
                    .block_types
                    .entry(seq)
                    .or_insert_with(|| "thinking".to_string());
                let partial = accumulator.partial_snapshot(model, response_id.clone());
                let _ = sender.send(AssistantMessageEvent::ThinkingStart {
                    content_index: seq as u32,
                    partial,
                });
            }
            _ => {}
        }

        match ev {
            StreamEvent::ContentBlockStart { index, .. } => {
                // Only tool blocks have real start events; text/thinking start events are not emitted on the real stream (lazy),
                // defensively avoid constructing events directly in tests without pushing ToolcallStart.
                accumulator.process_event(ev.clone());
                if let StreamEvent::ContentBlockStart { block_type, .. } = ev {
                    if block_type == "tool_use" {
                        let seq = accumulator.resolve_tool(*index);
                        let partial = accumulator.partial_snapshot(model, response_id.clone());
                        let _ = sender.send(AssistantMessageEvent::ToolcallStart {
                            content_index: seq as u32,
                            partial,
                        });
                    }
                }
            }
            StreamEvent::TextDelta { text, .. } => {
                accumulator.process_event(ev.clone());
                let seq = accumulator.resolve_content();
                let partial = accumulator.partial_snapshot(model, response_id.clone());
                let _ = sender.send(AssistantMessageEvent::TextDelta {
                    content_index: seq as u32,
                    delta: text.clone(),
                    partial,
                });
            }
            StreamEvent::ThinkingDelta { thinking, .. } => {
                accumulator.process_event(ev.clone());
                let seq = accumulator.resolve_thinking();
                let partial = accumulator.partial_snapshot(model, response_id.clone());
                let _ = sender.send(AssistantMessageEvent::ThinkingDelta {
                    content_index: seq as u32,
                    delta: thinking.clone(),
                    partial,
                });
            }
            StreamEvent::InputJsonDelta {
                index,
                partial_json,
            } => {
                accumulator.process_event(ev.clone());
                let seq = accumulator.resolve_tool(*index);
                let partial = accumulator.partial_snapshot(model, response_id.clone());
                let _ = sender.send(AssistantMessageEvent::ToolcallDelta {
                    content_index: seq as u32,
                    delta: partial_json.clone(),
                    partial,
                });
            }
            StreamEvent::ContentBlockStop { .. } => {
                // Finish batch: finalize all active blocks and emit End for each (partial includes the complete blocks),
                // Event order = ascending internal index = content array order.
                let finalized = accumulator.finalize_all();
                for (seq, block) in finalized {
                    let partial = accumulator.partial_snapshot(model, response_id.clone());
                    match block {
                        ContentBlock::Text(t) => {
                            let _ = sender.send(AssistantMessageEvent::TextEnd {
                                content_index: seq as u32,
                                content: t.text,
                                partial,
                            });
                        }
                        ContentBlock::Thinking(t) => {
                            let _ = sender.send(AssistantMessageEvent::ThinkingEnd {
                                content_index: seq as u32,
                                content: t.thinking,
                                partial,
                            });
                        }
                        ContentBlock::ToolCall(tc) => {
                            let _ = sender.send(AssistantMessageEvent::ToolcallEnd {
                                content_index: seq as u32,
                                tool_call: tc,
                                partial,
                            });
                        }
                        ContentBlock::Image(_) => {
                            // No image streaming in v1, defensively ignored
                        }
                    }
                }
            }
            StreamEvent::MessageDelta { stop_reason, usage } => {
                // Sync metadata into the snapshot (delivered uniformly on Done), do not emit events
                if let Some(sr) = stop_reason {
                    accumulator.stop_reason = Some(sr.clone());
                }
                if let Some(u) = usage {
                    accumulator.usage = u.clone();
                }
                accumulator.process_event(ev.clone());
            }
        }
    }
}

// ── Event stream background task ────────────────────────────────

/// Background SSE event loop.
/// Runs in a dedicated tokio task, reading the SSE event stream from the HTTP response body.
/// Internal flow:
/// Reconciles the final `stop_reason` with `error_message` (pure-function extraction from `sse_event_loop` Phase 4a, for unit tests).
///
/// User cancelled (`cancelled`) and no finish_reason received → `Aborted` (aligned with interactive-cancel-fix acceptance
/// that partials are finalized as aborted); the UI's Aborted branch in `assistant_message.rs` maps "Request was aborted"
/// to a clean "Operation aborted"); error present but not cancelled and no finish_reason → `Error`;
/// finish_reason already received → keep the accumulator's original reason and write the error to error_message (if cancellation lands
/// right after finish_reason, keep the original reason without re-labeling); no error → accumulator's original reason.
fn finalize_stop_reason(
    error_message: &Option<String>,
    accumulator: &StreamAccumulator,
    cancelled: bool,
) -> (StopReason, Option<String>) {
    match error_message {
        Some(msg) if accumulator.has_finish_reason => (
            accumulator.stop_reason.clone().unwrap_or(StopReason::Stop),
            Some(msg.clone()),
        ),
        Some(msg) if cancelled => (StopReason::Aborted, Some(msg.clone())),
        Some(msg) => (StopReason::Error, Some(msg.clone())),
        None => (
            accumulator.stop_reason.clone().unwrap_or(StopReason::Stop),
            None,
        ),
    }
}

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
        api: model.api.to_string(),
        provider: model.provider.clone(),
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
            // 3a. User cancelled → set abort info and exit the loop.
            //     "Request was aborted" is the standard text that the UI Aborted branch (assistant_message.rs) maps to
            //     "Operation aborted"; the old implementation (T-AI-02-1) used "Request cancelled by user"
            //     and Phase 4 finalized it as Error, causing cancellations to render as "Error: Request cancelled by user"
            //     on two lines — see the appendix in tasks/interactive-cancel-fix-plan.md for the fix that finalizes cancellation as Aborted.
            _ = cancel.cancelled() => {
                error_message = Some("Request was aborted".to_string());
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

                        // 3f. Parse chunk into StreamEvents and push incremental events at block boundaries (Start/Delta/End)
                        //     and accumulate into the accumulator (T-AI-02-1-fix: source push for the streaming typewriter effect;
                        //     provider used for thinking signature normalization special-casing, T-thinking-invisible-fix)
                        let events = parse_chunk(&json, &model.provider);
                        handle_events(&events, &mut accumulator, &sender, model, &response_id);
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

    // 4a. Reconcile stop_reason with error_message (pure function finalize_stop_reason, for unit tests):
    //     - User cancelled and no finish_reason → Aborted (aligns with cancel-fix acceptance "partial finalized as aborted")
    //     - has finish_reason but also an error → keep finish_reason, write the error into error_message
    //     - Error present and no finish_reason (not cancelled) → termination reason is Error
    //     - normal termination → use the accumulator's stop_reason
    let (stop_reason, error_msg) =
        finalize_stop_reason(&error_message, &accumulator, cancel.is_cancelled());

    // 4b. Assemble the final message
    let mut msg = AssistantMessage {
        content,
        api: model.api.to_string(),
        provider: model.provider.clone(),
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

    #[test]
    fn test_parse_usage_deepseek_shape_fallback() {
        // DeepSeek official shape: top-level prompt_cache_hit_tokens (no prompt_tokens_details)
        let json = serde_json::json!({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_cache_hit_tokens": 20
        });
        let (_, _, cache_read, _, _) = parse_usage(&json);
        assert_eq!(
            cache_read, 20,
            "DeepSeek shape should fall back to prompt_cache_hit_tokens"
        );
    }

    #[test]
    fn test_parse_usage_explicit_zero_no_fallback() {
        // nullish semantics: when cached_tokens is explicitly 0, do not fall back to prompt_cache_hit_tokens (same as pi's `??`)
        let json = serde_json::json!({
            "prompt_tokens_details": { "cached_tokens": 0 },
            "prompt_cache_hit_tokens": 999
        });
        let (_, _, cache_read, _, _) = parse_usage(&json);
        assert_eq!(cache_read, 0, "Explicit 0 does not fall back");
    }

    #[test]
    fn test_parse_usage_openai_shape_precedence() {
        // both shapes present: OpenAI shape takes priority (pi `cached_tokens ?? hit` chain order)
        let json = serde_json::json!({
            "prompt_tokens_details": { "cached_tokens": 20 },
            "prompt_cache_hit_tokens": 999
        });
        let (_, _, cache_read, _, _) = parse_usage(&json);
        assert_eq!(
            cache_read, 20,
            "OpenAI shape takes precedence over DeepSeek shape"
        );
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
            compat: None,
        }
    }

    /// Sample model with compat (T-AI-02-10: all request-side behavior is driven by `model.compat`).
    fn model_with_compat(compat: OpenAiCompletionsCompat) -> Model {
        Model {
            compat: Some(compat),
            ..sample_model()
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
        assert_eq!(req.tool_choice, Some(serde_json::json!("auto")));
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

    // ── prompt_cache_key（T-AI-02-6，pi-102）──

    #[test]
    fn test_build_request_body_prompt_cache_key_absent_by_default() {
        // default config (session_id = None) → parameter not sent, zero change to default-path request body
        let req = build_request_body(&sample_model(), &sample_context(), &StreamConfig::default());
        assert!(req.prompt_cache_key.is_none());
        let json = serde_json::to_value(&req).unwrap();
        assert!(
            json.get("prompt_cache_key").is_none(),
            "When session_id is omitted, prompt_cache_key must not be serialized (serialized key name must be snake_case)"
        );
    }

    #[test]
    fn test_build_request_body_prompt_cache_key_from_session_id() {
        let cfg = StreamConfig {
            session_id: Some("session-123".into()),
            ..Default::default()
        };
        let req = build_request_body(&sample_model(), &sample_context(), &cfg);
        assert_eq!(req.prompt_cache_key.as_deref(), Some("session-123"));
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["prompt_cache_key"], "session-123");
    }

    #[test]
    fn test_build_request_body_prompt_cache_key_truncated() {
        // over-long session_id → truncated to 64 code points then sent
        let long_id = "s".repeat(100);
        let cfg = StreamConfig {
            session_id: Some(long_id.clone()),
            ..Default::default()
        };
        let req = build_request_body(&sample_model(), &sample_context(), &cfg);
        let key = req
            .prompt_cache_key
            .as_deref()
            .expect("Send when session_id exists");
        assert_eq!(key.len(), 64);
        assert_eq!(key, "s".repeat(64));
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["prompt_cache_key"].as_str().unwrap().len(), 64);
    }

    #[test]
    fn test_build_request_body_prompt_cache_key_multibyte_truncation() {
        // multi-byte session_id: truncate by code point, never split a surrogate pair
        let cfg = StreamConfig {
            session_id: Some("😀".repeat(70)),
            ..Default::default()
        };
        let req = build_request_body(&sample_model(), &sample_context(), &cfg);
        let key = req.prompt_cache_key.expect("Send when session_id exists");
        assert_eq!(key, "😀".repeat(64));
    }

    // ── Request header merging (model.headers + compat session affinity + cfg.headers)──

    #[test]
    fn test_merge_request_headers_model_static_then_caller_override() {
        let model = Model {
            headers: Some(HashMap::from([
                ("x-opencode-client".to_string(), "zen".to_string()),
                ("X-Same".to_string(), "model".to_string()),
            ])),
            ..sample_model()
        };
        let cfg = StreamConfig {
            headers: Some(HashMap::from([(
                "X-Same".to_string(),
                "caller".to_string(),
            )])),
            ..Default::default()
        };
        let merged = merge_request_headers(&model, &cfg);
        assert_eq!(merged.len(), 2);
        assert_eq!(
            merged.get("x-opencode-client").map(String::as_str),
            Some("zen")
        );
        // Same-name overrides become a single value (not appended as dual values) — this is the reason this helper exists
        assert_eq!(merged.get("X-Same").map(String::as_str), Some("caller"));
    }

    #[test]
    fn test_merge_request_headers_no_sources_is_empty() {
        assert!(merge_request_headers(&sample_model(), &StreamConfig::default()).is_empty());
    }

    #[test]
    fn test_merge_request_headers_session_affinity_openai() {
        let model = model_with_compat(OpenAiCompletionsCompat {
            send_session_affinity_headers: Some(true),
            session_affinity_format: Some(SessionAffinityFormat::Openai),
            ..Default::default()
        });
        let cfg = StreamConfig {
            session_id: Some("s-1".into()),
            ..Default::default()
        };
        let merged = merge_request_headers(&model, &cfg);
        assert_eq!(merged.get("session_id").map(String::as_str), Some("s-1"));
        assert_eq!(
            merged.get("x-client-request-id").map(String::as_str),
            Some("s-1")
        );
        assert_eq!(
            merged.get("x-session-affinity").map(String::as_str),
            Some("s-1")
        );
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn test_merge_request_headers_session_affinity_openai_nosession() {
        let model = model_with_compat(OpenAiCompletionsCompat {
            send_session_affinity_headers: Some(true),
            session_affinity_format: Some(SessionAffinityFormat::OpenaiNosession),
            ..Default::default()
        });
        let cfg = StreamConfig {
            session_id: Some("s-1".into()),
            ..Default::default()
        };
        let merged = merge_request_headers(&model, &cfg);
        assert!(
            !merged.contains_key("session_id"),
            "nosession does not send session_id"
        );
        assert_eq!(
            merged.get("x-client-request-id").map(String::as_str),
            Some("s-1")
        );
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_merge_request_headers_session_affinity_openrouter() {
        let model = model_with_compat(OpenAiCompletionsCompat {
            send_session_affinity_headers: Some(true),
            session_affinity_format: Some(SessionAffinityFormat::Openrouter),
            ..Default::default()
        });
        let cfg = StreamConfig {
            session_id: Some("s-1".into()),
            ..Default::default()
        };
        let merged = merge_request_headers(&model, &cfg);
        assert_eq!(merged.get("x-session-id").map(String::as_str), Some("s-1"));
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn test_merge_request_headers_session_affinity_disabled_by_default() {
        // sendSessionAffinityHeaders not enabled in compat → do not send any affinity headers
        let cfg = StreamConfig {
            session_id: Some("s-1".into()),
            ..Default::default()
        };
        assert!(merge_request_headers(&sample_model(), &cfg).is_empty());
    }

    #[test]
    fn test_merge_request_headers_session_affinity_empty_session_skipped() {
        let model = model_with_compat(OpenAiCompletionsCompat {
            send_session_affinity_headers: Some(true),
            ..Default::default()
        });
        for session in [None, Some("")] {
            let cfg = StreamConfig {
                session_id: session.map(str::to_string),
                ..Default::default()
            };
            assert!(
                merge_request_headers(&model, &cfg).is_empty(),
                "Empty/missing session must not fabricate affinity header"
            );
        }
    }

    #[test]
    fn test_merge_request_headers_caller_overrides_affinity_same_key() {
        let model = model_with_compat(OpenAiCompletionsCompat {
            send_session_affinity_headers: Some(true),
            session_affinity_format: Some(SessionAffinityFormat::Openai),
            ..Default::default()
        });
        let cfg = StreamConfig {
            session_id: Some("s-1".into()),
            headers: Some(HashMap::from([(
                "x-session-affinity".to_string(),
                "caller".to_string(),
            )])),
            ..Default::default()
        };
        let merged = merge_request_headers(&model, &cfg);
        assert_eq!(
            merged.get("x-session-affinity").map(String::as_str),
            Some("caller"),
            "Caller explicit header overrides compat affinity header (single value)"
        );
        assert_eq!(merged.len(), 3);
    }

    // ── thinking toggle (T-AI-02-10: appended according to `model.compat.thinking_format`; no longer based on the provider name)──

    #[test]
    fn test_build_request_body_thinking_medium_deepseek() {
        let model = model_with_compat(OpenAiCompletionsCompat {
            thinking_format: Some(ThinkingFormat::DeepSeek),
            ..Default::default()
        });
        let cfg = StreamConfig {
            reasoning_effort: Some(ThinkingLevel::Medium),
            ..Default::default()
        };
        let req = build_request_body(&model, &sample_context(), &cfg);
        assert_eq!(req.reasoning_effort.as_deref(), Some("medium"));
        let t = req
            .thinking
            .as_ref()
            .expect("DeepSeek Medium should include thinking switch");
        assert_eq!(t.r#type, "enabled");
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["thinking"]["type"], "enabled");
        assert_eq!(json["reasoning_effort"], "medium");
    }

    #[test]
    fn test_build_request_body_thinking_off_deepseek() {
        // Off → thinking:{type:"disabled"}, do not send reasoning_effort
        let model = model_with_compat(OpenAiCompletionsCompat {
            thinking_format: Some(ThinkingFormat::DeepSeek),
            ..Default::default()
        });
        let req = build_request_body(&model, &sample_context(), &StreamConfig::default());
        assert!(req.reasoning_effort.is_none());
        let t = req
            .thinking
            .as_ref()
            .expect("DeepSeek Off should include thinking switch");
        assert_eq!(t.r#type, "disabled");
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["thinking"]["type"], "disabled");
        assert!(json.get("reasoning_effort").is_none());
    }

    #[test]
    fn test_build_request_body_thinking_openai_no_switch() {
        // openai format (also a default) → reasoning_effort only; do not send the thinking toggle
        let model = model_with_compat(OpenAiCompletionsCompat {
            thinking_format: Some(ThinkingFormat::OpenAi),
            ..Default::default()
        });
        let cfg = StreamConfig {
            reasoning_effort: Some(ThinkingLevel::Medium),
            ..Default::default()
        };
        let req = build_request_body(&model, &sample_context(), &cfg);
        assert_eq!(req.reasoning_effort.as_deref(), Some("medium"));
        assert!(
            req.thinking.is_none(),
            "openai format does not send thinking switch"
        );
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("thinking").is_none());
        assert_eq!(json["reasoning_effort"], "medium");
    }

    #[test]
    fn test_build_request_body_thinking_deepseek_non_reasoning_no_switch() {
        // model.reasoning=false → do not send thinking even for the DeepSeek format (the pi branch requires model.reasoning)
        let model = Model {
            reasoning: false,
            ..model_with_compat(OpenAiCompletionsCompat {
                thinking_format: Some(ThinkingFormat::DeepSeek),
                ..Default::default()
            })
        };
        let req = build_request_body(&model, &sample_context(), &StreamConfig::default());
        assert!(req.thinking.is_none());
    }

    // ── compat request handling (T-AI-02-10)──

    #[test]
    fn test_max_tokens_field_default_is_max_tokens() {
        let req = build_request_body(&sample_model(), &sample_context(), &StreamConfig::default());
        assert_eq!(req.max_tokens, Some(8192));
        assert!(req.max_completion_tokens.is_none());
    }

    #[test]
    fn test_max_tokens_field_max_completion_tokens() {
        let model = model_with_compat(OpenAiCompletionsCompat {
            max_tokens_field: Some(MaxTokensField::MaxCompletionTokens),
            ..Default::default()
        });
        let req = build_request_body(&model, &sample_context(), &StreamConfig::default());
        assert!(req.max_tokens.is_none());
        assert_eq!(req.max_completion_tokens, Some(8192));
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("max_tokens").is_none());
        assert_eq!(json["max_completion_tokens"], 8192);
    }

    #[test]
    fn test_stream_options_disabled_when_supports_usage_false() {
        let model = model_with_compat(OpenAiCompletionsCompat {
            supports_usage_in_streaming: Some(false),
            ..Default::default()
        });
        let req = build_request_body(&model, &sample_context(), &StreamConfig::default());
        assert!(req.stream_options.is_none());
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("stream_options").is_none());
    }

    #[test]
    fn test_stream_options_default_enabled() {
        let req = build_request_body(&sample_model(), &sample_context(), &StreamConfig::default());
        assert!(req.stream_options.is_some());
    }

    #[test]
    fn test_developer_role_when_reasoning_and_supported() {
        let model = model_with_compat(OpenAiCompletionsCompat {
            supports_developer_role: Some(true),
            ..Default::default()
        });
        let req = build_request_body(&model, &sample_context(), &StreamConfig::default());
        assert!(matches!(req.messages[0], RequestMessage::Developer { .. }));
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["messages"][0]["role"], "developer");
    }

    #[test]
    fn test_system_role_by_default() {
        let req = build_request_body(&sample_model(), &sample_context(), &StreamConfig::default());
        assert!(matches!(req.messages[0], RequestMessage::System { .. }));
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["messages"][0]["role"], "system");
    }

    #[test]
    fn test_developer_role_not_used_when_reasoning_false() {
        let model = Model {
            reasoning: false,
            ..model_with_compat(OpenAiCompletionsCompat {
                supports_developer_role: Some(true),
                ..Default::default()
            })
        };
        let req = build_request_body(&model, &sample_context(), &StreamConfig::default());
        assert!(matches!(req.messages[0], RequestMessage::System { .. }));
    }

    #[test]
    fn test_reasoning_effort_suppressed_when_unsupported() {
        let model = model_with_compat(OpenAiCompletionsCompat {
            supports_reasoning_effort: Some(false),
            ..Default::default()
        });
        let cfg = StreamConfig {
            reasoning_effort: Some(ThinkingLevel::High),
            ..Default::default()
        };
        let req = build_request_body(&model, &sample_context(), &cfg);
        assert!(req.reasoning_effort.is_none());
    }

    #[test]
    fn test_reasoning_effort_suppressed_when_model_not_reasoning() {
        let model = Model {
            reasoning: false,
            ..sample_model()
        };
        let cfg = StreamConfig {
            reasoning_effort: Some(ThinkingLevel::High),
            ..Default::default()
        };
        let req = build_request_body(&model, &sample_context(), &cfg);
        assert!(req.reasoning_effort.is_none());
    }

    // ── api_key_for_request (prefers cfg.api_key, falls back to Catalog)──

    #[test]
    fn test_api_key_prefers_config() {
        let cfg = StreamConfig {
            api_key: Some("cfg-key".into()),
            ..Default::default()
        };
        assert_eq!(
            api_key_for_request(&sample_model(), &cfg).as_deref(),
            Some("cfg-key")
        );
    }

    #[test]
    fn test_api_key_none_when_no_config_and_unknown_provider() {
        // sample_model.provider = "deepseek" is not in the built-in catalog → Catalog::api_key returns None
        assert!(api_key_for_request(&sample_model(), &StreamConfig::default()).is_none());
    }

    // ── serialization lock (prevents camelCase regression)──

    #[test]
    fn test_request_body_serialization_uses_snake_case_keys() {
        let model = model_with_compat(OpenAiCompletionsCompat {
            max_tokens_field: Some(MaxTokensField::MaxCompletionTokens),
            ..Default::default()
        });
        let cfg = StreamConfig {
            tool_choice: Some(ToolChoice::Auto),
            session_id: Some("s".into()),
            reasoning_effort: Some(ThinkingLevel::High),
            ..Default::default()
        };
        let req = build_request_body(&model, &sample_context(), &cfg);
        let json = serde_json::to_value(&req).unwrap();
        let obj = json.as_object().unwrap();
        for key in [
            "max_completion_tokens",
            "tool_choice",
            "stream_options",
            "prompt_cache_key",
            "reasoning_effort",
        ] {
            assert!(obj.contains_key(key), "Should contain snake_case key {key}");
        }
        assert_eq!(json["stream_options"]["include_usage"], true);
        for bad in [
            "maxTokens",
            "maxCompletionTokens",
            "toolChoice",
            "streamOptions",
            "includeUsage",
            "promptCacheKey",
            "reasoningEffort",
        ] {
            assert!(
                !obj.contains_key(bad),
                "Must not contain camelCase key {bad}"
            );
        }
    }

    #[test]
    fn test_tool_choice_serializes_openai_shape() {
        assert_eq!(
            tool_choice_to_request(&ToolChoice::Auto),
            serde_json::json!("auto")
        );
        assert_eq!(
            tool_choice_to_request(&ToolChoice::None),
            serde_json::json!("none")
        );
        assert_eq!(
            tool_choice_to_request(&ToolChoice::Required),
            serde_json::json!("required")
        );
        assert_eq!(
            tool_choice_to_request(&ToolChoice::Function {
                name: "read".into()
            }),
            serde_json::json!({"type": "function", "function": {"name": "read"}})
        );
    }

    // ═════════════════════════════════════════════════════════════
    //  parse_chunk
    // ═════════════════════════════════════════════════════════════

    #[test]
    fn test_parse_chunk_empty() {
        let events = parse_chunk(&serde_json::json!({}), "opencode-go");
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
        let events = parse_chunk(&chunk, "opencode-go");
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
        let events = parse_chunk(&chunk, "opencode-go");
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
        // DeepSeek-style form (huoshan): reasoning_content → ThinkingDelta, signature = field name
        let chunk = serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": { "reasoning_content": "thinking..." },
                "finish_reason": null
            }]
        });
        let events = parse_chunk(&chunk, "huoshan");
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            StreamEvent::ThinkingDelta {
                thinking: ref t,
                signature: ref s,
                ..
            } if t == "thinking..." && s == "reasoning_content"
        ));
    }

    #[test]
    fn test_parse_chunk_thinking_reasoning_field() {
        // opencode-go endpoint returns delta.reasoning (non-standard name): readable + signature normalized to canonical name
        let chunk = serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": { "reasoning": "thinking..." },
                "finish_reason": null
            }]
        });
        let events = parse_chunk(&chunk, "opencode-go");
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            StreamEvent::ThinkingDelta {
                thinking: ref t,
                signature: ref s,
                ..
            } if t == "thinking..." && s == "reasoning_content"
        ));
    }

    #[test]
    fn test_parse_chunk_thinking_reasoning_text() {
        // Third fallback field reasoning_text: signature = field name (not normalized)
        let chunk = serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": { "reasoning_text": "thinking..." },
                "finish_reason": null
            }]
        });
        let events = parse_chunk(&chunk, "opencode-go");
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            StreamEvent::ThinkingDelta {
                thinking: ref t,
                signature: ref s,
                ..
            } if t == "thinking..." && s == "reasoning_text"
        ));
    }

    #[test]
    fn test_parse_chunk_thinking_field_priority() {
        // Three fields co-occur (e.g. chutes.ai returns both reasoning_content and reasoning): take the first non-null to avoid duplication
        let chunk = serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": {
                    "reasoning_content": "A",
                    "reasoning": "B",
                    "reasoning_text": "C"
                },
                "finish_reason": null
            }]
        });
        let events = parse_chunk(&chunk, "opencode-go");
        assert_eq!(events.len(), 1);
        if let StreamEvent::ThinkingDelta {
            thinking,
            signature,
            ..
        } = &events[0]
        {
            assert_eq!(thinking, "A");
            assert_eq!(signature, "reasoning_content");
        } else {
            panic!("expected ThinkingDelta");
        }
    }

    #[test]
    fn test_parse_chunk_thinking_reasoning_field_non_opencode_provider_not_normalized() {
        // Non-opencode-go provider hit reasoning: keep raw field name as signature
        let chunk = serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": { "reasoning": "thinking..." },
                "finish_reason": null
            }]
        });
        let events = parse_chunk(&chunk, "huoshan");
        assert!(matches!(
            events[0],
            StreamEvent::ThinkingDelta {
                thinking: ref t,
                signature: ref s,
                ..
            } if t == "thinking..." && s == "reasoning"
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
        let events = parse_chunk(&chunk, "opencode-go");
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
        let events = parse_chunk(&chunk, "opencode-go");
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
        let events = parse_chunk(&chunk, "opencode-go");
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
        let events = parse_chunk(&chunk, "opencode-go");
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
        let events = parse_chunk(&chunk, "opencode-go");
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
            signature: "reasoning_content".into(),
        });
        acc.process_event(StreamEvent::ContentBlockStop { index: 0 });
        assert_eq!(acc.content_blocks.len(), 1);
        if let ContentBlock::Thinking(t) = &acc.content_blocks[0] {
            assert_eq!(t.thinking, "step 1...");
            assert_eq!(t.thinking_signature.as_deref(), Some("reasoning_content"));
        } else {
            panic!("expected Thinking block");
        }
    }

    #[test]
    fn test_accumulator_thinking_signature_first_wins() {
        // First ThinkingDelta's signature takes effect; later ones do not override (aligned with pi ensureThinkingBlock binding the first signature)
        let mut acc = StreamAccumulator::default();
        acc.process_event(StreamEvent::ThinkingDelta {
            index: 0,
            thinking: "a".into(),
            signature: "reasoning".into(),
        });
        acc.process_event(StreamEvent::ThinkingDelta {
            index: 0,
            thinking: "b".into(),
            signature: "reasoning_text".into(),
        });
        acc.process_event(StreamEvent::ContentBlockStop { index: 0 });
        assert_eq!(acc.content_blocks.len(), 1);
        if let ContentBlock::Thinking(t) = &acc.content_blocks[0] {
            assert_eq!(t.thinking, "ab");
            assert_eq!(t.thinking_signature.as_deref(), Some("reasoning"));
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

    // ═════════════════════════════════════════════════════════════
    //  handle_events: incremental event push (T-AI-02-1-fix, chunk-level integration)
    // ═════════════════════════════════════════════════════════════

    // ── finalize_stop_reason (interactive-cancel-fix backfill: cancellation finalized as Aborted) ──

    #[test]
    fn finalize_stop_reason_cancel_without_finish_is_aborted() {
        // User cancelled and no finish_reason received → Aborted (aligns with cancel-fix acceptance "partial finalized as aborted")
        let acc = StreamAccumulator::default();
        let (reason, msg) =
            finalize_stop_reason(&Some("Request was aborted".to_string()), &acc, true);
        assert_eq!(reason, StopReason::Aborted);
        assert_eq!(msg.as_deref(), Some("Request was aborted"));
    }

    #[test]
    fn finalize_stop_reason_cancel_after_finish_keeps_original_reason() {
        // Cancellation lands right after finish_reason: keep the accumulator's original reason, write the error to error_message, do not re-label
        let mut acc = StreamAccumulator::default();
        acc.stop_reason = Some(StopReason::Stop);
        acc.has_finish_reason = true;
        let (reason, msg) =
            finalize_stop_reason(&Some("Request was aborted".to_string()), &acc, true);
        assert_eq!(reason, StopReason::Stop);
        assert_eq!(msg.as_deref(), Some("Request was aborted"));
    }

    #[test]
    fn finalize_stop_reason_error_without_finish_is_error() {
        // Non-cancellation error (e.g. SSE transport-layer error) and no finish_reason → Error
        let acc = StreamAccumulator::default();
        let (reason, msg) =
            finalize_stop_reason(&Some("SSE stream error: boom".to_string()), &acc, false);
        assert_eq!(reason, StopReason::Error);
        assert_eq!(msg.as_deref(), Some("SSE stream error: boom"));
    }

    #[test]
    fn finalize_stop_reason_no_error_uses_accumulator_reason() {
        // Normal completion: no error → accumulator's original reason + no error_message
        let mut acc = StreamAccumulator::default();
        acc.stop_reason = Some(StopReason::ToolUse);
        let (reason, msg) = finalize_stop_reason(&None, &acc, false);
        assert_eq!(reason, StopReason::ToolUse);
        assert!(msg.is_none());
    }

    #[test]
    fn finalize_stop_reason_error_with_finish_keeps_reason() {
        // Error with finish_reason (e.g. last chunk failed): keep the original reason, write the error to error_message
        let mut acc = StreamAccumulator::default();
        acc.stop_reason = Some(StopReason::Stop);
        acc.has_finish_reason = true;
        let (reason, msg) = finalize_stop_reason(&Some("tail error".to_string()), &acc, false);
        assert_eq!(reason, StopReason::Stop);
        assert_eq!(msg.as_deref(), Some("tail error"));
    }

    /// Run a set of JSON chunks through `parse_chunk` + `handle_events`; return the collected incremental events and the accumulator's final state.
    /// Enter from real chunk JSON (no manually constructed StreamEvents), covering `parse_chunk`'s path.
    fn run_chunks(chunks: &[serde_json::Value]) -> (Vec<AssistantMessageEvent>, StreamAccumulator) {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AssistantMessageEvent>();
        let mut acc = StreamAccumulator::default();
        let model = sample_model();
        let rid: Option<String> = None;
        for c in chunks {
            let events = parse_chunk(c, &model.provider);
            handle_events(&events, &mut acc, &tx, &model, &rid);
        }
        let mut got = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            got.push(ev);
        }
        (got, acc)
    }

    fn chunk_content(text: &str) -> serde_json::Value {
        serde_json::json!({"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]})
    }

    fn chunk_thinking(text: &str) -> serde_json::Value {
        serde_json::json!({"choices": [{"index": 0, "delta": {"reasoning_content": text}, "finish_reason": null}]})
    }

    fn chunk_finish() -> serde_json::Value {
        serde_json::json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    }

    fn chunk_tool_start(index: u64, id: &str, name: &str) -> serde_json::Value {
        serde_json::json!({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": index, "id": id, "type": "function", "function": {"name": name, "arguments": ""}}]}, "finish_reason": null}]})
    }

    fn chunk_tool_args(index: u64, args: &str) -> serde_json::Value {
        serde_json::json!({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": index, "function": {"arguments": args}}]}, "finish_reason": null}]})
    }

    #[test]
    fn test_stream_text_incremental_and_consistent() {
        // Plain text stream: lazy TextStart → TextDelta* → TextEnd at finish; final state matches manual calculation
        let (evs, acc) = run_chunks(&[
            chunk_content("Hello"),
            chunk_content(" world"),
            chunk_finish(),
        ]);
        assert_eq!(evs.len(), 4, "TextStart + 2×delta + TextEnd");
        assert!(matches!(
            evs[0],
            AssistantMessageEvent::TextStart {
                content_index: 0,
                ..
            }
        ));
        assert!(matches!(
            &evs[1],
            AssistantMessageEvent::TextDelta { delta, .. } if delta == "Hello"
        ));
        assert!(matches!(
            &evs[2],
            AssistantMessageEvent::TextDelta { delta, .. } if delta == " world"
        ));
        match &evs[3] {
            AssistantMessageEvent::TextEnd {
                content_index: 0,
                content,
                ..
            } => {
                assert_eq!(content, "Hello world");
            }
            _ => panic!("expected TextEnd"),
        }
        // Final state: 1 text block, complete content, stop_reason settled
        assert_eq!(acc.content_blocks.len(), 1);
        match &acc.content_blocks[0] {
            ContentBlock::Text(t) => assert_eq!(t.text, "Hello world"),
            _ => panic!("expected Text block"),
        }
        assert_eq!(acc.stop_reason, Some(StopReason::Stop));
        assert!(acc.has_finish_reason);
    }

    #[test]
    fn test_stream_thinking_then_text_order() {
        // Mixed thinking + text: block order = first-appearance order (thinking first), final state has both blocks in the correct order
        let (evs, acc) = run_chunks(&[
            chunk_thinking("step 1"),
            chunk_thinking(" step 2"),
            chunk_content("Answer"),
            chunk_finish(),
        ]);
        // Event sequence: ThinkingStart, ThinkingDelta, ThinkingDelta, TextStart, TextDelta, ThinkingEnd, TextEnd
        assert!(matches!(
            evs[0],
            AssistantMessageEvent::ThinkingStart {
                content_index: 0,
                ..
            }
        ));
        assert!(matches!(
            &evs[1],
            AssistantMessageEvent::ThinkingDelta { delta, .. } if delta == "step 1"
        ));
        assert!(matches!(
            evs[3],
            AssistantMessageEvent::TextStart {
                content_index: 1,
                ..
            }
        ));
        // End events in ascending internal index order: thinking(0) before text(1)
        assert!(matches!(
            evs[5],
            AssistantMessageEvent::ThinkingEnd {
                content_index: 0,
                ..
            }
        ));
        assert!(matches!(
            evs[6],
            AssistantMessageEvent::TextEnd {
                content_index: 1,
                ..
            }
        ));
        // Final state: two blocks, thinking first; thinking_signature backfilled (chunk comes from reasoning_content → field name)
        assert_eq!(acc.content_blocks.len(), 2);
        match (&acc.content_blocks[0], &acc.content_blocks[1]) {
            (ContentBlock::Thinking(t), ContentBlock::Text(x)) => {
                assert_eq!(t.thinking, "step 1 step 2");
                assert_eq!(t.thinking_signature.as_deref(), Some("reasoning_content"));
                assert_eq!(x.text, "Answer");
            }
            _ => panic!("expected [Thinking, Text]"),
        }
    }

    #[test]
    fn test_stream_multi_tool_all_finalized() {
        // Multi-tool call index 0/1/2: all finalize, three ToolcallEnd, final state contains 3 complete ToolCalls
        let (evs, acc) = run_chunks(&[
            chunk_tool_start(0, "call_1", "bash"),
            chunk_tool_start(1, "call_2", "read"),
            chunk_tool_start(2, "call_3", "write"),
            chunk_tool_args(0, "{\"command\": "),
            chunk_tool_args(0, "\"ls\"}"),
            chunk_tool_args(1, "{\"path\": "),
            chunk_tool_args(1, "\"/a\"}"),
            chunk_tool_args(2, "{\"path\": \"/b\", "),
            chunk_tool_args(2, "\"content\": \"x\"}"),
            chunk_finish(),
        ]);
        let starts = evs
            .iter()
            .filter(|e| matches!(e, AssistantMessageEvent::ToolcallStart { .. }))
            .count();
        let deltas = evs
            .iter()
            .filter(|e| matches!(e, AssistantMessageEvent::ToolcallDelta { .. }))
            .count();
        let ends = evs
            .iter()
            .filter(|e| matches!(e, AssistantMessageEvent::ToolcallEnd { .. }))
            .count();
        assert_eq!(starts, 3, "One ToolcallStart for each of the three tools");
        assert_eq!(
            deltas, 6,
            "One ToolcallDelta for each of 6 argument increments"
        );
        assert_eq!(ends, 3, "All three tools ToolcallEnd (defect 6 regression)");
        assert_eq!(
            acc.content_blocks.len(),
            3,
            "All multi-tool final states settled (defect 6 regression)"
        );
        let ids: Vec<&str> = acc
            .content_blocks
            .iter()
            .map(|b| match b {
                ContentBlock::ToolCall(tc) => tc.id.as_str(),
                _ => "",
            })
            .collect();
        assert_eq!(ids, vec!["call_1", "call_2", "call_3"]);
        // Full parameters for each tool
        let tc2 = match &acc.content_blocks[1] {
            ContentBlock::ToolCall(tc) => tc,
            _ => panic!(),
        };
        assert_eq!(tc2.name, "read");
        assert_eq!(tc2.arguments, serde_json::json!({"path": "/a"}));
        let tc3 = match &acc.content_blocks[2] {
            ContentBlock::ToolCall(tc) => tc,
            _ => panic!(),
        };
        assert_eq!(
            tc3.arguments,
            serde_json::json!({"path": "/b", "content": "x"})
        );
    }

    #[test]
    fn test_stream_text_tool_mixed_keeps_text() {
        // text and tool_use mixed in the same stream: text not lost, two blocks in correct order (defect 5 regression: index 0 conflict)
        let (evs, acc) = run_chunks(&[
            chunk_content("Here "),
            chunk_tool_start(0, "call_1", "bash"),
            chunk_tool_args(0, "{\"a\": "),
            chunk_content("is result"),
            chunk_tool_args(0, "1}"),
            chunk_finish(),
        ]);
        // Event sequence: TextStart, TextDelta, ToolcallStart, ToolcallDelta, TextDelta, ToolcallDelta, TextEnd, ToolcallEnd
        assert!(matches!(
            evs[0],
            AssistantMessageEvent::TextStart {
                content_index: 0,
                ..
            }
        ));
        assert!(matches!(
            &evs[1],
            AssistantMessageEvent::TextDelta { delta, .. } if delta == "Here "
        ));
        assert!(matches!(
            evs[2],
            AssistantMessageEvent::ToolcallStart {
                content_index: 1,
                ..
            }
        ));
        assert!(matches!(
            &evs[4],
            AssistantMessageEvent::TextDelta { delta, .. } if delta == "is result"
        ));
        assert!(matches!(
            evs[6],
            AssistantMessageEvent::TextEnd {
                content_index: 0,
                ..
            }
        ));
        assert!(matches!(
            evs[7],
            AssistantMessageEvent::ToolcallEnd {
                content_index: 1,
                ..
            }
        ));
        // Final state: text not lost, Text first, ToolCall after (internal indices 0/1)
        assert_eq!(
            acc.content_blocks.len(),
            2,
            "One block each for text and tool (defect 5 regression)"
        );
        match (&acc.content_blocks[0], &acc.content_blocks[1]) {
            (ContentBlock::Text(t), ContentBlock::ToolCall(tc)) => {
                assert_eq!(t.text, "Here is result");
                assert_eq!(tc.id, "call_1");
                assert_eq!(tc.arguments, serde_json::json!({"a": 1}));
            }
            _ => panic!("expected [Text, ToolCall]"),
        }
    }

    #[test]
    fn test_stream_delta_count_matches() {
        // Event count ≈ number of deltas in chunk (regression guard against silently swallowed events)
        let (evs, _acc) = run_chunks(&[
            chunk_thinking("t1"),
            chunk_content("c1"),
            chunk_tool_start(0, "call_1", "bash"),
            chunk_tool_args(0, "{}"),
            chunk_finish(),
        ]);
        let delta_events = evs
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    AssistantMessageEvent::TextDelta { .. }
                        | AssistantMessageEvent::ThinkingDelta { .. }
                        | AssistantMessageEvent::ToolcallDelta { .. }
                )
            })
            .count();
        assert_eq!(
            delta_events, 3,
            "1 thinking + 1 text + 1 tool args = 3 deltas"
        );
        assert_eq!(evs.len(), 9, "Start×3 + Delta×3 + End×3");
    }

    #[test]
    fn test_stream_partial_text_monotonic() {
        // Each Delta's partial text accumulates monotonically with no rollback
        let (evs, _acc) = run_chunks(&[chunk_content("a"), chunk_content("b"), chunk_content("c")]);
        let texts: Vec<String> = evs
            .iter()
            .filter_map(|e| match e {
                AssistantMessageEvent::TextDelta { partial, .. } => Some(partial_text(partial)),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["a", "ab", "abc"]);
    }

    #[test]
    fn test_stream_abort_keeps_pushed_partial() {
        // Stream not finalized (simulated abort): already-pushed events and partial remain visible, no End, no Done
        let (evs, acc) = run_chunks(&[chunk_content("Hello"), chunk_content(" world")]);
        assert_eq!(evs.len(), 3, "TextStart + 2×delta, no End/Done");
        assert!(!evs
            .iter()
            .any(|e| matches!(e, AssistantMessageEvent::TextEnd { .. })));
        // Accumulator still holds active blocks; a partial snapshot can be built at any time (partial already seen in the UI before abort is preserved)
        assert!(
            acc.content_blocks.is_empty(),
            "Not finalized; content_blocks empty"
        );
        let model = sample_model();
        let snap = acc.partial_snapshot(&model, None);
        assert_eq!(snap.content.len(), 1);
        match &snap.content[0] {
            ContentBlock::Text(t) => assert_eq!(t.text, "Hello world"),
            _ => panic!("expected Text block in partial"),
        }
    }

    /// Extract the text block content from the partial message (test helper).
    fn partial_text(msg: &AssistantMessage) -> String {
        msg.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}
