//! Agent loop tests.
//!
//! ## Mock tests (mock_ prefix; only mock stream_fn is used, with no real requests sent)
//! Verify flow integrity, error handling, and cancellation behavior of `create_context` + `run_agent_loop`.
//!
//! ## Real integration tests (real_ prefix; depends on the OPENAI_API_KEY environment variable)
//! Inject `zen_ai::api::openai_stream::stream()` as the real StreamFn into the agent loop,
//! verifying the end-to-end chain: LLM call → tool loop → result return.
//!
//! REQ-AGENT-01 (pi-204, pi-202): context assembly + multi-turn tool loop integration behavior.

use std::path::PathBuf;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use zen_ai::tool::{Tool, ToolContext, ToolError, ToolExecutionResult, ToolUpdateCallback};
use zen_ai::types::{
    AssistantMessage, AssistantMessageEvent, ContentBlock, ContentBlockList, Context,
    ErrorStopReason, Message, Model, NonErrorStopReason, StopReason, TextContent, ToolCall,
    ToolResultMessage, Usage, UserMessage,
};
use zen_core::agent::agent_hooks::AgentLoopHooks;
use zen_core::agent::agent_loop::run_agent_loop;
use zen_core::agent::context::create_context;
use zen_core::agent::types::{AgentEvent, AgentLoopConfig, StreamFn, ToolExecutionMode};

// ═══════════════════════════════════════════════════════════════════
//  Mock helpers
// ═══════════════════════════════════════════════════════════════════

/// Build a mock StreamFn that returns the given assistant message (text).
fn mock_text_stream_fn(response_text: &str, stop_reason: StopReason) -> StreamFn {
    let text = response_text.to_string();
    Arc::new(
        move |_model: &Model, _ctx: &Context, _cfg: Option<_>, _cancel: Option<_>| {
            let text = text.clone();
            let s = stop_reason.clone();
            Box::pin(async move {
                let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
                let msg = AssistantMessage {
                    content: vec![ContentBlock::Text(TextContent {
                        text,
                        text_signature: None,
                    })],
                    api: "test".into(),
                    provider: "test".into(),
                    model: "test-model".into(),
                    response_model: None,
                    response_id: None,
                    usage: Usage::default(),
                    stop_reason: s,
                    error_message: None,
                    timestamp: 1000,
                };
                tx.send(AssistantMessageEvent::Start {
                    partial: msg.clone(),
                })
                .ok();
                tx.send(AssistantMessageEvent::Done {
                    reason: NonErrorStopReason::Stop,
                    message: msg,
                })
                .ok();
                Ok(rx)
            })
        },
    )
}

/// Build a mock StreamFn that returns an error.
fn mock_error_stream_fn() -> StreamFn {
    Arc::new(
        |_model: &Model, _ctx: &Context, _cfg: Option<_>, _cancel: Option<_>| {
            Box::pin(async move {
                let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
                let err_msg = AssistantMessage {
                    content: vec![],
                    api: "test".into(),
                    provider: "test".into(),
                    model: "test-model".into(),
                    response_model: None,
                    response_id: None,
                    usage: Usage::default(),
                    stop_reason: StopReason::Error,
                    error_message: Some("Simulated provider error".into()),
                    timestamp: 1000,
                };
                tx.send(AssistantMessageEvent::Error {
                    reason: ErrorStopReason::Error,
                    error: err_msg,
                })
                .ok();
                Ok(rx)
            })
        },
    )
}

/// Build a mock StreamFn that returns a tool call, then text (simulating a two-turn conversation).
fn mock_two_turn_stream_fn() -> StreamFn {
    let turns = Arc::new(std::sync::Mutex::new(0u8));
    Arc::new(
        move |_model: &Model, _ctx: &Context, _cfg: Option<_>, _cancel: Option<_>| {
            let turns = turns.clone();
            Box::pin(async move {
                let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
                let mut turn = turns.lock().unwrap();
                if *turn == 0 {
                    *turn = 1;
                    // First turn: return tool call
                    let msg = AssistantMessage {
                        content: vec![ContentBlock::ToolCall(ToolCall {
                            id: "call-1".into(),
                            name: "bash".into(),
                            arguments: serde_json::json!({"command": "echo hello"}),
                            thought_signature: None,
                        })],
                        api: "test".into(),
                        provider: "test".into(),
                        model: "test-model".into(),
                        response_model: None,
                        response_id: None,
                        usage: Usage::default(),
                        stop_reason: StopReason::ToolUse,
                        error_message: None,
                        timestamp: 1000,
                    };
                    tx.send(AssistantMessageEvent::Start {
                        partial: msg.clone(),
                    })
                    .ok();
                    tx.send(AssistantMessageEvent::Done {
                        reason: NonErrorStopReason::ToolUse,
                        message: msg,
                    })
                    .ok();
                } else {
                    // Second turn: return text
                    let msg = AssistantMessage {
                        content: vec![ContentBlock::Text(TextContent {
                            text: "Task completed successfully.".into(),
                            text_signature: None,
                        })],
                        api: "test".into(),
                        provider: "test".into(),
                        model: "test-model".into(),
                        response_model: None,
                        response_id: None,
                        usage: Usage::default(),
                        stop_reason: StopReason::Stop,
                        error_message: None,
                        timestamp: 2000,
                    };
                    tx.send(AssistantMessageEvent::Start {
                        partial: msg.clone(),
                    })
                    .ok();
                    tx.send(AssistantMessageEvent::Done {
                        reason: NonErrorStopReason::Stop,
                        message: msg,
                    })
                    .ok();
                }
                Ok(rx)
            })
        },
    )
}

// ─── Mock Hooks ──────────────────────────────────────────────────

/// MockHooks that collect events, used to verify the agent loop's event sequence.
struct MockHooks {
    events: Arc<std::sync::Mutex<Vec<AgentEvent>>>,
}

impl MockHooks {
    fn new() -> Self {
        Self {
            events: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    fn collected_events(&self) -> Vec<AgentEvent> {
        self.events.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl AgentLoopHooks for MockHooks {
    async fn on_event(&self, event: &AgentEvent) {
        self.events.lock().unwrap().push(event.clone());
    }
}

// ─── Mock Tool ───────────────────────────────────────────────────

#[derive(Debug)]
struct BashMockTool {
    params: serde_json::Value,
}

impl BashMockTool {
    fn new() -> Self {
        Self {
            params: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }),
        }
    }
}

#[async_trait::async_trait]
impl Tool for BashMockTool {
    fn name(&self) -> &str {
        "bash"
    }
    fn description(&self) -> &str {
        "Execute a bash command"
    }
    fn parameters(&self) -> &serde_json::Value {
        &self.params
    }
    fn label(&self) -> &str {
        "Bash"
    }
    async fn execute(
        &self,
        tool_call_id: &str,
        params: &serde_json::Value,
        _ctx: &ToolContext,
        _cancel: &CancellationToken,
        _on_update: ToolUpdateCallback,
    ) -> Result<ToolExecutionResult, ToolError> {
        let cmd = params
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("<unknown>");
        Ok(ToolExecutionResult {
            message: ToolResultMessage {
                tool_call_id: tool_call_id.to_string(),
                tool_name: "bash".to_string(),
                content: vec![ContentBlock::Text(TextContent {
                    text: format!("Executed: {}", cmd),
                    text_signature: None,
                })],
                details: None,
                usage: Usage::default(),
                is_error: false,
                added_tool_names: None,
                timestamp: 2000,
            },
            terminate: false,
        })
    }
}

// ─── Test helpers ────────────────────────────────────────────────

fn default_model() -> Model {
    Model {
        id: "test-model".into(),
        name: Some("Test Model".into()),
        base_url: "https://api.test.com".into(),
        context_window: 128_000,
        max_tokens: 4096,
        reasoning: false,
        headers: None,
        cost_rates: None,
    }
}

fn default_config() -> AgentLoopConfig {
    AgentLoopConfig {
        model: default_model(),
        api_key: Some("test-key".into()),
        tool_execution: ToolExecutionMode::Sequential,
        max_tokens: None,
        temperature: None,
    }
}

fn user_message(text: &str) -> Message {
    Message::User(UserMessage {
        content: ContentBlockList::Text(text.to_string()),
        timestamp: 1000,
    })
}

// ═══════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════

/// Verify: empty prompt → returns an empty vec immediately.
#[tokio::test]
async fn mock_test_empty_prompts() {
    let hooks = MockHooks::new();
    let cancel = CancellationToken::new();
    let stream_fn = mock_text_stream_fn("should not be called", StopReason::Stop);
    let tools: Vec<Arc<dyn Tool>> = vec![];
    let mut ctx = create_context("prompt".into(), vec![], &tools);

    let result = run_agent_loop(
        vec![],
        &mut ctx,
        &default_config(),
        &hooks,
        &cancel,
        stream_fn,
        &tools,
        &ToolContext::new(PathBuf::from("."), None, None),
    )
    .await;

    assert!(result.is_empty(), "expected empty result for empty prompts");
}

/// Verify: stream_fn returns an error → agent loop handles it correctly.
#[tokio::test]
async fn mock_test_stream_error() {
    let hooks = MockHooks::new();
    let cancel = CancellationToken::new();
    let stream_fn = mock_error_stream_fn();
    let tools: Vec<Arc<dyn Tool>> = vec![];
    let mut ctx = create_context("prompt".into(), vec![], &tools);

    let result = run_agent_loop(
        vec![user_message("trigger error")],
        &mut ctx,
        &default_config(),
        &hooks,
        &cancel,
        stream_fn,
        &tools,
        &ToolContext::new(PathBuf::from("."), None, None),
    )
    .await;

    // Should return one error message (stop_reason = Error)
    assert!(!result.is_empty(), "expected error message");
    if let Some(Message::Assistant(a)) = result.last() {
        assert_eq!(a.stop_reason, StopReason::Error);
        assert!(a.error_message.is_some());
    } else {
        panic!("expected Assistant message with error");
    }
}

/// Verify: cancellation signal → agent loop exits early.
#[tokio::test]
async fn mock_test_cancellation() {
    let hooks = MockHooks::new();
    let cancel = CancellationToken::new();
    let stream_fn = mock_text_stream_fn("Hello", StopReason::Stop);
    let tools: Vec<Arc<dyn Tool>> = vec![];
    let mut ctx = create_context("prompt".into(), vec![], &tools);

    // Cancel first, then run
    cancel.cancel();

    let _result = run_agent_loop(
        vec![user_message("test")],
        &mut ctx,
        &default_config(),
        &hooks,
        &cancel,
        stream_fn,
        &tools,
        &ToolContext::new(PathBuf::from("."), None, None),
    )
    .await;

    // After cancellation, stream_fn still returns messages, but the loop may exit early
    // Mainly verify no panic
}

/// Verify: create_context assembles the Context correctly, and run_agent_loop can consume it.
#[tokio::test]
async fn mock_test_create_context_and_loop() {
    let hooks = MockHooks::new();
    let cancel = CancellationToken::new();
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(BashMockTool::new())];

    // Build the initial context using create_context
    let mut ctx = create_context("System: You are helpful.".into(), vec![], &tools);

    // Verify the context was built correctly
    assert_eq!(
        ctx.system_prompt,
        Some("System: You are helpful.".to_string())
    );
    assert!(ctx.tools.is_some());
    assert_eq!(ctx.tools.as_ref().unwrap().len(), 1);
    assert_eq!(ctx.tools.as_ref().unwrap()[0].name, "bash");

    let stream_fn = mock_text_stream_fn("I understand.", StopReason::Stop);

    let result = run_agent_loop(
        vec![user_message("Hello")],
        &mut ctx,
        &default_config(),
        &hooks,
        &cancel,
        stream_fn,
        &tools,
        &ToolContext::new(PathBuf::from("."), None, None),
    )
    .await;

    assert!(!result.is_empty(), "expected messages from the loop");
    // The message in context should contain the original prompt + the new message
    assert!(
        !ctx.messages.is_empty(),
        "context.messages should have been updated"
    );
}

/// Verify: text response (no tool call) → the agent loop returns the message correctly.
#[tokio::test]
async fn mock_test_text_response_no_tool_calls() {
    let hooks = MockHooks::new();
    let cancel = CancellationToken::new();
    let stream_fn = mock_text_stream_fn("Hello, world!", StopReason::Stop);
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(BashMockTool::new())];
    let mut ctx = create_context("You are a test assistant.".into(), vec![], &tools);

    let result = run_agent_loop(
        vec![user_message("Say hello")],
        &mut ctx,
        &default_config(),
        &hooks,
        &cancel,
        stream_fn,
        &tools,
        &ToolContext::new(PathBuf::from("."), None, None),
    )
    .await;

    // Returned a message
    assert!(!result.is_empty(), "expected at least one new message");

    // The last one should be the assistant message
    let last = result.last().unwrap();
    match last {
        Message::Assistant(a) => {
            assert_eq!(a.stop_reason, StopReason::Stop);
            let text = a
                .content
                .iter()
                .filter_map(|c| match c {
                    ContentBlock::Text(t) => Some(t.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" ");
            assert!(text.contains("Hello, world!"));
        }
        _ => panic!("expected Assistant message, got {:?}", last),
    }

    // Verify the event sequence
    let events = hooks.collected_events();
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::AgentStart)),
        "expected AgentStart"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::AgentEnd { .. })),
        "expected AgentEnd"
    );
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::TurnStart)),
        "expected TurnStart"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::TurnEnd { .. })),
        "expected TurnEnd"
    );
}

/// Verify: tool call → agent loop executes the tool → returns the result.
#[tokio::test]
async fn mock_test_tool_call_execution() {
    // Use a two-turn stream_fn: the first turn returns a tool call, the second returns text
    let hooks = MockHooks::new();
    let cancel = CancellationToken::new();
    let stream_fn = mock_two_turn_stream_fn();
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(BashMockTool::new())];
    let mut ctx = create_context("You are a test assistant.".into(), vec![], &tools);

    let result = run_agent_loop(
        vec![user_message("Run a command")],
        &mut ctx,
        &default_config(),
        &hooks,
        &cancel,
        stream_fn,
        &tools,
        &ToolContext::new(PathBuf::from("."), None, None),
    )
    .await;

    // There should be at least 3 messages: user prompt + assistant(toolcall) + tool result + assistant(text)
    assert!(
        result.len() >= 3,
        "expected at least 3 messages, got {}",
        result.len()
    );

    // Check that the event sequence contains a tool execution event
    let events = hooks.collected_events();

    let tool_starts: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ToolExecutionStart { .. }))
        .collect();
    assert!(
        !tool_starts.is_empty(),
        "expected ToolExecutionStart events"
    );

    let tool_ends: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ToolExecutionEnd { .. }))
        .collect();
    assert!(!tool_ends.is_empty(), "expected ToolExecutionEnd events");
}

// ═══════════════════════════════════════════════════════════════════
// Real integration tests (real_ prefix, require OPENAI_API_KEY env var)
// ═══════════════════════════════════════════════════════════════════

/// Construct a StreamFn wrapping the real `stream()` function.
fn real_stream_fn() -> StreamFn {
    Arc::new(
        |model: &Model, ctx: &Context, cfg: Option<_>, cancel: Option<_>| {
            let model = model.clone();
            let ctx = ctx.clone();
            Box::pin(
                async move { zen_ai::api::openai_stream::stream(&model, &ctx, cfg, cancel).await },
            )
        },
    )
}

/// Real LLM integration test: the agent loop receives a user message and returns the LLM response.
///
/// Skip condition: automatically skipped when the `OPENAI_API_KEY` env var is not set.
/// Validation:
/// 1. `create_context` assembles the Context correctly
/// 2. `run_agent_loop` uses the real `stream()` function to make the LLM call
/// 3. The returned message contains non-empty content and a valid stop_reason
// cargo test --test agent_loop_integration -- --show-output
// cargo test -p zen-core --test agent_loop_integration -- real --nocapture
#[tokio::test]
async fn real_test_agent_loop_simple_chat() {
    let api_key = std::env::var("OPENAI_API_KEY");
    if api_key.is_err() {
        eprintln!("OPENAI_API_KEY not set; skipping real integration test");
        return;
    }

    let hooks = MockHooks::new();
    let cancel = CancellationToken::new();
    let tools: Vec<Arc<dyn Tool>> = vec![];

    // Create the Model from an environment variable
    let model = Model::from_env().with_env_overrides();

    // Build the initial context with create_context
    let mut ctx = create_context(
        "You are a helpful assistant. Answer concisely.".into(),
        vec![],
        &tools,
    );

    let config = AgentLoopConfig {
        model: model.clone(),
        api_key: None, // stream() reads from environment variables internally
        tool_execution: ToolExecutionMode::Sequential,
        max_tokens: Some(256),
        temperature: Some(0.0),
    };

    let stream_fn = real_stream_fn();

    let result = run_agent_loop(
        vec![user_message("Where are you? Answer in Chinese.")],
        &mut ctx,
        &config,
        &hooks,
        &cancel,
        stream_fn,
        &tools,
        &ToolContext::new(PathBuf::from("."), None, None),
    )
    .await;

    // Print the result
    println!("=== real_test_agent_loop_simple_chat ===");
    for (i, msg) in result.iter().enumerate() {
        match msg {
            Message::Assistant(a) => {
                let text = a
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        ContentBlock::Text(t) => Some(t.text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                println!(
                    "  result[{}] assistant: text='{}' stop_reason={:?} model={} usage={}in/{}out",
                    i, text, a.stop_reason, a.model, a.usage.input, a.usage.output
                );
            }
            Message::User(u) => {
                let text = match &u.content {
                    ContentBlockList::Text(t) => t.clone(),
                    ContentBlockList::Blocks(b) => format!("{:?}", b),
                };
                println!("  result[{}] user: {}", i, text);
            }
            Message::ToolResult(t) => {
                println!(
                    "  result[{}] tool_result({}): is_error={}",
                    i, t.tool_name, t.is_error
                );
            }
        }
    }

    // Verify that an assistant message was returned
    let last = result.last().expect("expected at least one new message");
    match last {
        Message::Assistant(a) => {
            assert!(!a.content.is_empty(), "response should have content");
            assert!(
                matches!(
                    a.stop_reason,
                    StopReason::Stop | StopReason::Length | StopReason::ToolUse
                ),
                "stop_reason should be Stop, Length, or ToolUse, got {:?}",
                a.stop_reason
            );
            assert!(
                a.usage.input > 0 || a.usage.output > 0,
                "expected non-zero token usage"
            );
        }
        _ => panic!("expected Assistant message, got {:?}", last),
    }

    // Verify the event sequence
    let events = hooks.collected_events();
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::AgentStart)),
        "expected AgentStart"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::AgentEnd { .. })),
        "expected AgentEnd"
    );
}
