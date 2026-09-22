//! Shared helpers for integration tests (real LLM tests: `agent_session_persistence_real` / `session_agent_runtime_real`).
//!
//! Provides: API key env var name for the default provider, turn-limiting hooks, real stream fn, AgentState/Agent construction,
//! assistant plain-text extraction. Shared by both real test files to avoid duplicated scaffolding.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use zen_ai::tool::ToolExecutionMode;
use zen_ai::types::StreamFn;
use zen_ai::types::{ContentBlock, ContentBlockList, Context, Message, Model, ThinkingLevel};
use zen_core::harness::agent_hooks::AgentLoopHooks;
use zen_core::harness::engine::Agent;
use zen_core::harness::types::{AgentState, ShouldStopAfterTurnContext};

/// Returns the required API key environment variable name for the default provider (`ZEN_PROVIDER`, default opencode-go).
pub fn provider_key_var() -> &'static str {
    match zen_ai::provider::default_provider_name().as_str() {
        "opencode-go" => "OPENCODE_API_KEY",
        _ => "ARK_API_KEY",
    }
}

/// Lightweight hooks: cap the max number of turns for the real LLM (prevents tool-less multi-turn loops).
pub struct MockHooks {
    max_turns: u32,
    turn_count: Arc<AtomicU32>,
}

impl MockHooks {
    pub fn new() -> Self {
        Self {
            max_turns: 50,
            turn_count: Arc::new(AtomicU32::new(0)),
        }
    }

    pub fn with_max_turns(mut self, n: u32) -> Self {
        self.max_turns = n;
        self
    }
}

#[async_trait::async_trait]
impl AgentLoopHooks for MockHooks {
    async fn should_stop_after_turn(&self, _context: &ShouldStopAfterTurnContext) -> bool {
        let turn = self.turn_count.fetch_add(1, Ordering::SeqCst) + 1;
        turn >= self.max_turns
    }
}

/// StreamFn wrapping the real `stream()` function (mirrors `real_stream_fn` in `session_persistence_real.rs`).
pub fn real_stream_fn() -> StreamFn {
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

/// Build an AgentState for the engine (empty tools; fixed system_prompt; messages = restored history).
pub fn agent_state(system_prompt: &str, history: Vec<Message>) -> AgentState {
    AgentState {
        system_prompt: system_prompt.to_string(),
        model: Model::from_env(),
        thinking_level: ThinkingLevel::Off,
        tools: vec![],
        messages: history,
        is_run: false,
        streaming_message: None,
        pending_tool_calls: HashSet::new(),
        error_message: None,
    }
}

/// Build an engine with the real stream fn + turn-limiting hooks.
pub fn new_agent(base_prompt: &str, history: Vec<Message>) -> Agent {
    Agent::new(
        agent_state(base_prompt, history),
        ToolExecutionMode::Sequential,
        Arc::new(MockHooks::new().with_max_turns(3)),
        real_stream_fn(),
        None, // T-AI-02-6: integration tests do not carry the session affinity key
    )
}

/// Return the plain text of the last assistant message (concatenated Text blocks).
pub fn last_assistant_text(messages: &[Message]) -> String {
    for msg in messages.iter().rev() {
        if let Message::Assistant(a) = msg {
            return a
                .content
                .iter()
                .filter_map(|c| match c {
                    ContentBlock::Text(t) => Some(t.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" ");
        }
    }
    String::new()
}

/// Print all messages returned by the LLM (role + text/tool-call content) to aid debugging in real tests.
///
/// Outputs to stderr (`eprintln!`); view alongside `--nocapture`.
pub fn print_messages(tag: &str, messages: &[Message]) {
    eprintln!("── {tag} ({} messages) ──", messages.len());
    for (i, msg) in messages.iter().enumerate() {
        match msg {
            Message::User(u) => {
                eprintln!("  [{i}] user: {}", content_block_list_text(&u.content));
            }
            Message::Assistant(a) => {
                let text = block_text(&a.content);
                let tool_calls: Vec<&str> = a
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        ContentBlock::ToolCall(tc) => Some(tc.name.as_str()),
                        _ => None,
                    })
                    .collect();
                if tool_calls.is_empty() {
                    eprintln!("  [{i}] assistant: {text}");
                } else {
                    eprintln!(
                        "  [{i}] assistant: {text} [tool_calls: {}]",
                        tool_calls.join(", ")
                    );
                }
            }
            Message::ToolResult(t) => {
                let text = block_text(&t.content);
                let err = if t.is_error { " (is_error)" } else { "" };
                eprintln!("  [{i}] tool_result({}){err}: {text}", t.tool_name);
            }
        }
    }
}

fn content_block_list_text(c: &ContentBlockList) -> String {
    match c {
        ContentBlockList::Text(s) => s.clone(),
        ContentBlockList::Blocks(blocks) => block_text(blocks),
    }
}

fn block_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|c| match c {
            ContentBlock::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}
