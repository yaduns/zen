//! Real integration tests (real_ prefix, require the OPENAI_API_KEY environment variable).
//!
//! Inject `zen_ai::api::openai_stream::stream()` as the real StreamFn into the agent loop,
//! verifying the end-to-end chain: LLM call → tool loop → result return.
//!
//! Run:
//! ```bash
//! cargo test -p zen-core --test agent_loop_real_tools -- --nocapture
//! ```
//!
//! REQ-AGENT-01 (pi-204, pi-202): context assembly + multi-turn tool loop integration behavior.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use zen_ai::tool::{Tool, ToolContext};
use zen_ai::types::{
    ContentBlock, ContentBlockList, Context, Message, Model, StopReason, ToolResultMessage, Usage,
};
use zen_core::harness::agent_hooks::AgentLoopHooks;
use zen_core::harness::agent_loop::run_agent_loop;
use zen_core::harness::context::create_context;
use zen_core::harness::types::ShouldStopAfterTurnContext;
use zen_core::harness::types::{AgentEvent, AgentLoopConfig, StreamFn, ToolExecutionMode};
use zen_core::tools::create_coding_tools;

// ─── Mock Hooks ──────────────────────────────────────────────────

/// MockHooks that collect events, used to verify the agent loop's event sequence.
/// Supports real-time event printing (`verbose`) and a max-turns limit (`max_turns`).
struct MockHooks {
    events: Arc<std::sync::Mutex<Vec<AgentEvent>>>,
    /// Whether to print events to stderr in real time (unbuffered, visible during tests)
    verbose: bool,
    /// Max turn count; once exceeded, `should_stop_after_turn` returns true
    max_turns: u32,
    /// Current turn counter
    turn_count: Arc<AtomicU32>,
}

impl MockHooks {
    fn new() -> Self {
        Self {
            events: Arc::new(std::sync::Mutex::new(Vec::new())),
            verbose: false,
            max_turns: 50,
            turn_count: Arc::new(AtomicU32::new(0)),
        }
    }

    fn with_verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    fn with_max_turns(mut self, n: u32) -> Self {
        self.max_turns = n;
        self
    }

    #[allow(dead_code)]
    fn collected_events(&self) -> Vec<AgentEvent> {
        self.events.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl AgentLoopHooks for MockHooks {
    async fn on_event(&self, event: &AgentEvent) {
        self.events.lock().unwrap().push(event.clone());
        if self.verbose {
            match event {
                AgentEvent::AgentStart => eprintln!("[hook] AgentStart"),
                AgentEvent::AgentEnd { .. } => eprintln!("[hook] AgentEnd"),
                AgentEvent::TurnStart => eprintln!("[hook] TurnStart"),
                AgentEvent::TurnEnd { .. } => eprintln!("[hook] TurnEnd"),
                AgentEvent::MessageStart { .. } => eprintln!("[hook] MessageStart"),
                AgentEvent::MessageUpdate { .. } => eprintln!("[hook] MessageUpdate"),
                AgentEvent::MessageEnd { .. } => eprintln!("[hook] MessageEnd"),
                AgentEvent::ToolExecutionStart { tool_name, .. } => {
                    eprintln!("[hook] ToolExecutionStart: tool={}", tool_name);
                }
                AgentEvent::ToolExecutionEnd { .. } => eprintln!("[hook] ToolExecutionEnd"),
                _ => eprintln!("[hook] Other"),
            }
        }
    }

    async fn should_stop_after_turn(&self, _context: &ShouldStopAfterTurnContext) -> bool {
        let turn = self.turn_count.fetch_add(1, Ordering::SeqCst) + 1;
        if turn >= self.max_turns {
            eprintln!(
                "[hook] should_stop_after_turn: reached max_turns={}",
                self.max_turns
            );
            true
        } else {
            false
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────

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

fn user_message(text: &str) -> Message {
    Message::User(zen_ai::types::UserMessage {
        content: ContentBlockList::Text(text.to_string()),
        timestamp: 1000,
    })
}

// ═══════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════

/// Real integration test: multi-turn tool calls — first use bash to list files, then use read to read file contents.
///
/// Skip condition: automatically skipped when the `OPENAI_API_KEY` env var is not set.
///
/// The loop in `agent_loop_inner` automatically feeds tool results back to the LLM for the next turn,
/// until the LLM returns plain text to finish. A single `run_agent_loop` here runs all the turns.
///
/// Run:
/// ```bash
/// cargo test -p zen-core --test agent_loop_real_tools -- real_multi_tool --nocapture
/// ```
#[tokio::test]
async fn real_test_multi_tool_chain() {
    let api_key = std::env::var("OPENAI_API_KEY");
    if api_key.is_err() {
        eprintln!("OPENAI_API_KEY not set; skipping real integration test");
        return;
    }

    let hooks = MockHooks::new().with_verbose().with_max_turns(5);
    let cancel = CancellationToken::new();

    let registry = create_coding_tools();
    eprintln!("Registered tools: {:?}", registry.names());
    let tools: Vec<Arc<dyn Tool>> = registry.tools;

    let model = Model::from_env().with_env_overrides();
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let mut ctx = create_context(
        "You are a helpful assistant with access to bash, read, write, and edit tools. \
         Answer concisely."
            .into(),
        vec![],
        &tools,
    );

    let config = AgentLoopConfig {
        model: model.clone(),
        api_key: None,
        tool_execution: ToolExecutionMode::Sequential,
        max_tokens: Some(1024),
        temperature: Some(0.0),
    };

    let stream_fn = real_stream_fn();

    // Single run_agent_loop: agent_loop_inner handles the multiple turns automatically,
    // should_stop_after_turn caps it at 5 turns.
    let all_messages = run_agent_loop(
        vec![user_message(
            "List the files in the current project directory, and if `Cargo.toml` exists, tell me the contents of its `dependencies` section.",
        )],
        &mut ctx,
        &config,
        &hooks,
        &cancel,
        stream_fn.clone(),
        &ToolContext::new(project_root.clone(), None, None),
    )
    .await;

    // ── print full conversation ──
    // println!("=== real_test_multi_tool_chain ===");
    // println!("Project root: {}", project_root.display());
    // println!("Total messages: {}", all_messages.len());

    let mut tool_call_count = 0usize;
    let mut tool_names_used = Vec::new();

    for (i, msg) in all_messages.iter().enumerate() {
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
                let has_tool_calls = a
                    .content
                    .iter()
                    .any(|c| matches!(c, ContentBlock::ToolCall(_)));
                if has_tool_calls {
                    for block in &a.content {
                        if let ContentBlock::ToolCall(tc) = block {
                            tool_call_count += 1;
                            tool_names_used.push(tc.name.clone());
                            println!("  [{}] tool_call: {} → {}", i, tc.name, tc.arguments);
                        }
                    }
                }
                println!(
                    "  [{}] assistant: tc={} text='{}' stop={:?}",
                    i,
                    has_tool_calls,
                    if text.len() > 200 {
                        format!("{}…", &text[..200])
                    } else {
                        text
                    },
                    a.stop_reason,
                );
            }
            Message::ToolResult(t) => {
                let text = t
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        ContentBlock::Text(tc) => Some(tc.text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                println!(
                    "  [{}] tool_result({}): err={} out='{}'",
                    i,
                    t.tool_name,
                    t.is_error,
                    if text.len() > 200 {
                        format!("{}…", &text[..200])
                    } else {
                        text
                    }
                );
            }
            Message::User(u) => {
                let text = match &u.content {
                    ContentBlockList::Text(t) => t.clone(),
                    ContentBlockList::Blocks(b) => format!("{:?}", b),
                };
                println!(
                    "  [{}] user: {}",
                    i,
                    if text.len() > 100 {
                        format!("{}…", &text[..100])
                    } else {
                        text
                    }
                );
            }
        }
    }

    // ── Print the event sequence ──
    let events = hooks.collected_events();
    // println!();
    // println!("=== Event sequence ({} total) ===", events.len());
    // for (i, e) in events.iter().enumerate() {
    //     let label = match e {
    //         AgentEvent::AgentStart => "AgentStart",
    //         AgentEvent::AgentEnd { .. } => "AgentEnd",
    //         AgentEvent::TurnStart => "TurnStart",
    //         AgentEvent::TurnEnd { .. } => "TurnEnd",
    //         AgentEvent::MessageStart { .. } => "MessageStart",
    //         AgentEvent::MessageUpdate { .. } => "MessageUpdate",
    //         AgentEvent::MessageEnd { .. } => "MessageEnd",
    //         AgentEvent::ToolExecutionStart { .. } => "ToolExecutionStart",
    //         AgentEvent::ToolExecutionEnd { .. } => "ToolExecutionEnd",
    //         _ => "Other",
    //     };
    //     println!("  event[{}] {}", i, label);
    // }

    // ── Assertions ──
    assert!(!all_messages.is_empty(), "expected at least one message");

    // Count turns from the event sequence
    // let turn_starts: Vec<_> = events
    //     .iter()
    //     .filter(|e| matches!(e, AgentEvent::TurnStart))
    //     .collect();
    // println!("Turns: {} (from event sequence)", turn_starts.len());

    // Verify that a tool was called at least once
    assert!(
        tool_call_count >= 1,
        "expected at least 1 tool call, got {}",
        tool_call_count
    );

    let has_bash = tool_names_used.iter().any(|n| n == "bash");
    assert!(has_bash, "expected bash tool to be called");
    // Note: some models get stuck on the same tool during multi-turn tool calls.
    // Here we don't require read to be called; we only verify the multi-turn loop mechanism works.
    if tool_names_used.iter().any(|n| n == "read") {
        println!("✓ read tool was called in a later turn");
    }

    // Verify that all tool results succeeded
    for msg in &all_messages {
        if let Message::ToolResult(t) = msg {
            assert!(!t.is_error, "tool {} should succeed", t.tool_name);
        }
    }

    // Verify the last message is an assistant message (with text) or a tool result (success)
    let last = all_messages.last().unwrap();
    match last {
        Message::Assistant(a) => {
            let text = a
                .content
                .iter()
                .filter_map(|c| {
                    if let ContentBlock::Text(t) = c {
                        Some(t.text.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
            let has_tc = a
                .content
                .iter()
                .any(|c| matches!(c, ContentBlock::ToolCall(_)));
            if !has_tc {
                assert!(
                    !text.is_empty(),
                    "final assistant response should have content"
                );
            }
        }
        Message::ToolResult(t) => {
            // When the loop hits the max turns, the last message may be a tool result
            assert!(!t.is_error, "final tool result should succeed");
        }
        _ => {
            panic!(
                "expected final Assistant or ToolResult message, got {:?}",
                last
            );
        }
    }

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

/// Real integration test: use real tools (bash/read/write/edit) to have the agent list the file structure of the current project directory.
///
/// Skip condition: automatically skipped when the `OPENAI_API_KEY` env var is not set.
///
/// Validation:
/// 1. The tool registry returned by `create_coding_tools()` is correctly injected into the agent loop
/// 2. The agent can call the bash tool to run `ls -la` or similar commands
/// 3. Tool results are correctly injected into the subsequent LLM context
/// 4. Returns a structured directory description in the end
///
/// Run:
/// ```bash
/// cargo test -p zen-core --test agent_loop_real_tools -- --nocapture
/// ```
#[tokio::test]
async fn real_test_tools_list_directory() {
    let api_key = std::env::var("OPENAI_API_KEY");
    if api_key.is_err() {
        eprintln!("OPENAI_API_KEY not set; skipping real integration test");
        return;
    }

    let hooks = MockHooks::new();
    let cancel = CancellationToken::new();

    let registry = create_coding_tools();
    println!("Registered tools: {:?}", registry.names());
    let tools: Vec<Arc<dyn Tool>> = registry.tools;

    let model = Model::from_env().with_env_overrides();
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let mut ctx = create_context(
        "You are a helpful assistant with access to bash, read, write, and edit tools. \
         Answer concisely."
            .into(),
        vec![],
        &tools,
    );

    let config = AgentLoopConfig {
        model: model.clone(),
        api_key: None,
        tool_execution: ToolExecutionMode::Sequential,
        max_tokens: Some(512),
        temperature: Some(0.0),
    };

    let stream_fn = real_stream_fn();

    let result = run_agent_loop(
        vec![user_message(
            "List the files in the current project directory using bash. \
             Use `ls -la` to show file details. Then describe what you see.",
        )],
        &mut ctx,
        &config,
        &hooks,
        &cancel,
        stream_fn,
        &ToolContext::new(project_root.clone(), None, None),
    )
    .await;

    // ── print full conversation ──
    println!("=== real_test_tools_list_directory ===");
    println!("Project root: {}", project_root.display());
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
                let has_tool_calls = a
                    .content
                    .iter()
                    .any(|c| matches!(c, ContentBlock::ToolCall(_)));
                println!(
                    "  result[{}] assistant: tool_calls={} text='{}' stop_reason={:?} usage={}in/{}out",
                    i,
                    has_tool_calls,
                    if text.len() > 200 {
                        format!("{}…", &text[..200])
                    } else {
                        text
                    },
                    a.stop_reason,
                    a.usage.input,
                    a.usage.output
                );
            }
            Message::ToolResult(t) => {
                let text = t
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        ContentBlock::Text(tc) => Some(tc.text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                println!(
                    "  result[{}] tool_result({}): is_error={} output='{}'",
                    i,
                    t.tool_name,
                    t.is_error,
                    if text.len() > 300 {
                        format!("{}…", &text[..300])
                    } else {
                        text
                    }
                );
            }
            Message::User(u) => {
                let text = match &u.content {
                    ContentBlockList::Text(t) => t.clone(),
                    ContentBlockList::Blocks(b) => format!("{:?}", b),
                };
                println!(
                    "  result[{}] user: {}",
                    i,
                    if text.len() > 100 {
                        format!("{}…", &text[..100])
                    } else {
                        text
                    }
                );
            }
        }
    }

    // ── Assertions ──
    assert!(!result.is_empty(), "expected at least one new message");

    let has_assistant = result.iter().any(|m| matches!(m, Message::Assistant(_)));
    assert!(has_assistant, "expected at least one assistant message");

    let last = result.last().unwrap();
    match last {
        Message::Assistant(a) => {
            assert!(
                matches!(
                    a.stop_reason,
                    StopReason::Stop | StopReason::Length | StopReason::ToolUse
                ),
                "stop_reason should be Stop, Length, or ToolUse, got {:?}",
                a.stop_reason
            );
            let text = a
                .content
                .iter()
                .filter_map(|c| match c {
                    ContentBlock::Text(t) => Some(t.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" ");
            let has_tool_calls = a
                .content
                .iter()
                .any(|c| matches!(c, ContentBlock::ToolCall(_)));
            if !has_tool_calls {
                assert!(
                    !text.is_empty(),
                    "final assistant text response should have content"
                );
            }
        }
        Message::ToolResult(t) => {
            assert!(!t.is_error, "tool execution should succeed");
            assert_eq!(t.tool_name, "bash", "expected bash tool");
            let text = t
                .content
                .iter()
                .filter_map(|c| match c {
                    ContentBlock::Text(tc) => Some(tc.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" ");
            assert!(
                text.contains("Cargo.toml") || text.contains("src"),
                "tool output should contain project files, got: {}",
                text
            );
        }
        _ => {
            panic!(
                "expected final Assistant or ToolResult message, got {:?}",
                last
            );
        }
    }
}
