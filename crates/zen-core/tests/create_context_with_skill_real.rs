//! Real integration tests (real_ prefix; depend on the API key environment variables of the selected default provider).
//!
//! Covers the "Skills and System Prompt" section of Phase 1 Step 5:
//! - REQ-AGENT-05 (T-AG-05-2): skill system — `load_skills` loads SKILL.md → `format_skills_for_system_prompt`
//!   formats it into the system prompt → the LLM actually follows the skills in the agent loop.
//! - REQ-AGENT-05 (T-AG-05-1): end-to-end system prompt — `create_context` assembles a constraint-bearing system prompt → the LLM executes it in the loop.
//!   (The assembly logic of `build_system_prompt` itself is covered by unit tests within the crate; this test verifies its downstream effects.)
//!
//! Run:
//! ```bash
//! cargo test -p zen-core --test create_context_with_skill_real -- --nocapture
//! ```

use std::collections::HashSet;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use zen_ai::tool::ToolExecutionMode;
use zen_ai::types::StreamFn;
use zen_ai::types::{ContentBlock, Context, Message, Model, ThinkingLevel};
use zen_core::harness::agent_hooks::AgentLoopHooks;
use zen_core::harness::engine::Agent;
use zen_core::harness::types::{AgentEvent, AgentState, ShouldStopAfterTurnContext};
use zen_core::skills::format::format_skills_for_system_prompt;
use zen_core::skills::load::load_skills;

/// Returns the required API key environment variable name for the default provider (`ZEN_PROVIDER`, default opencode-go).
fn provider_key_var() -> &'static str {
    match zen_ai::provider::default_provider_name().as_str() {
        "opencode-go" => "OPENCODE_API_KEY",
        _ => "ARK_API_KEY",
    }
}

// ─── Mock Hooks (lightweight collectors, following the conventions in agent_loop_real_tools.rs) ───

struct MockHooks {
    events: Arc<std::sync::Mutex<Vec<AgentEvent>>>,
    max_turns: u32,
    turn_count: Arc<AtomicU32>,
}

impl MockHooks {
    fn new() -> Self {
        Self {
            events: Arc::new(std::sync::Mutex::new(Vec::new())),
            max_turns: 50,
            turn_count: Arc::new(AtomicU32::new(0)),
        }
    }

    fn with_max_turns(mut self, n: u32) -> Self {
        self.max_turns = n;
        self
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

    async fn should_stop_after_turn(&self, _context: &ShouldStopAfterTurnContext) -> bool {
        let turn = self.turn_count.fetch_add(1, Ordering::SeqCst) + 1;
        turn >= self.max_turns
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────

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

/// Build an AgentState for engine tests (empty tools, empty history; system_prompt provided by the caller).
///
/// The engine's `Agent::prompt` creates a `Context` on the fly from the state (`create_context`) and drives the low-level loop.
fn agent_state(system_prompt: String) -> AgentState {
    AgentState {
        system_prompt,
        model: Model::from_env(),
        thinking_level: ThinkingLevel::Off,
        tools: vec![],
        messages: vec![],
        is_run: false,
        streaming_message: None,
        pending_tool_calls: HashSet::new(),
        error_message: None,
    }
}

/// Return the plain text of the last assistant message (concatenated Text blocks).
fn last_assistant_text(messages: &[Message]) -> String {
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

// ═══════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════

/// Real integration test: load SKILL.md skills and verify that the LLM follows them in the agent loop.
///
/// Skip condition: automatically skipped when the API key env var of the selected default provider is not set.
///
/// Validation:
/// 1. `load_skills` discovers and loads SKILL.md from a temp directory (name/description/content are correct).
/// 2. `format_skills_for_system_prompt` generates an `<available_skills>` block and appends it to the system prompt.
/// 3. Constraints in the skill description are followed by the LLM (the final answer contains the `<<SKILLED>>` end token).
#[ignore]
#[tokio::test]
async fn real_test_skills_loaded_and_followed_by_llm() {
    let key_var = provider_key_var();
    let api_key = std::env::var(key_var);
    if api_key.is_err() {
        eprintln!("{} not set; skipping real integration test", key_var);
        return;
    }

    // 1) Temp directory + SKILL.md (minimal YAML frontmatter: description is required)
    let dir = tempfile::tempdir().expect("tempdir");
    let skill_dir = dir.path().join("concise-reply");
    std::fs::create_dir_all(&skill_dir).expect("create skill dir");
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\ndescription: Always end your answer with the exact token <<SKILLED>>.\n---\n\n\
         End every reply with the literal token <<SKILLED>> on its own line.\n",
    )
    .expect("write SKILL.md");

    // 2) Load the skills
    let result = load_skills(&[skill_dir]).await;
    assert_eq!(
        result.skills.len(),
        1,
        "expected exactly 1 skill, got {}",
        result.skills.len()
    );
    let skill = &result.skills[0];
    assert_eq!(
        skill.name, "concise-reply",
        "name should derive from parent dir"
    );
    assert!(skill.description.contains("<<SKILLED>>"));
    assert!(skill.content.contains("<<SKILLED>>"));
    assert!(!skill.disable_model_invocation);

    // 3) Format into the system prompt
    let skills_block = format_skills_for_system_prompt(&result.skills);
    assert!(skills_block.contains("<available_skills>"));
    assert!(skills_block.contains("concise-reply"));

    let system_prompt = format!(
        "You are a helpful assistant.\n\
         Follow the available skills when the task matches their description.\n\n{skills_block}"
    );

    // 4) Engine (empty tools, pure skill compliance): the Agent constructs a Context from state internally and drives the low-level loop
    let agent = Agent::new(
        agent_state(system_prompt),
        ToolExecutionMode::Sequential,
        Arc::new(MockHooks::new().with_max_turns(3)),
        real_stream_fn(),
    );
    let all_messages = agent.prompt("Hello! What is 2 + 2?").await;

    // 5) Assert: the skill end token appears in the final answer
    let final_text = last_assistant_text(&all_messages);
    eprintln!("final assistant: {final_text}");
    assert!(
        final_text.contains("<<SKILLED>>"),
        "skill token <<SKILLED>> missing from final answer: '{final_text}'"
    );
}

/// Real integration test: a constraint-bearing system prompt is executed by the LLM in the agent loop.
///
/// Skip condition: automatically skipped when the API key env var of the selected default provider is not set.
///
/// Validation:
/// 1. `create_context` assembles a system prompt carrying a strong constraint ("answer only BANANA").
/// 2. The LLM follows the system prompt (the final answer contains BANANA).
/// 3. No tool calls occur at any point (guaranteed both by the system prompt prohibition and the empty tool list).
#[ignore]
#[tokio::test]
async fn real_test_system_prompt_constraint_honored() {
    let key_var = provider_key_var();
    let api_key = std::env::var(key_var);
    if api_key.is_err() {
        eprintln!("{} not set; skipping real integration test", key_var);
        return;
    }

    // Keep an external Arc reference to hooks to inspect events after the run (assert no tool calls)
    let hooks = Arc::new(MockHooks::new().with_max_turns(3));
    let agent = Agent::new(
        agent_state(
            "Do not use any tools. Answer with exactly the single word: BANANA".to_string(),
        ),
        ToolExecutionMode::Sequential,
        hooks.clone(),
        real_stream_fn(),
    );
    let all_messages = agent.prompt("Say hello.").await;

    // Assert: the answer contains BANANA (tolerating case/punctuation variants)
    let final_text = last_assistant_text(&all_messages);
    eprintln!("final assistant: {final_text}");
    assert!(
        final_text.to_uppercase().contains("BANANA"),
        "system-prompt constraint (BANANA) not honored: '{final_text}'"
    );

    // Assert: no tool calls occur at any point
    let events = hooks.collected_events();
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolExecutionStart { .. })),
        "expected no tool calls, but ToolExecutionStart occurred"
    );
}
