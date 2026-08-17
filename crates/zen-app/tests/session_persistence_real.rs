//! Real integration tests (real_ prefix; depend on the API key environment variables of the selected default provider).
//!
//! Covers the "session tree and persistence" part of Phase 1 Step 5 (JSONL lives in zen-app, so this test goes in zen-app/tests):
//! - REQ-AGENT-03 (T-AG-03-1): session tree structure (SessionTree construction + tree navigation).
//! - REQ-AGENT-03 (T-AG-03-2): `build_context` extracts message history from the tree.
//! - REQ-AGENT-03 (T-AG-03-3): JSONL persistence (create/append/read/fork).
//!
//! Full pipeline: real LLM produces a session → persist to JSONL → read back and rebuild the tree → build_context → real LLM follow-up.
//!
//! Run:
//! ```bash
//! cargo test -p zen-app --test session_persistence_real -- --nocapture
//! ```

use std::collections::HashSet;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use zen_ai::tool::ToolExecutionMode;
use zen_ai::types::{ContentBlock, Context, Message, Model, ThinkingLevel};
use zen_app::session::jsonl::{
    append_entry, create_session_file, create_session_id, create_timestamp, get_entries_to_fork,
    read_session_file, CreateSessionOptions, ForkPosition,
};
use zen_core::harness::agent_hooks::AgentLoopHooks;
use zen_core::harness::engine::Agent;
use zen_core::harness::types::{AgentEvent, AgentState, ShouldStopAfterTurnContext, StreamFn};
use zen_core::session::context::build_context;
use zen_core::session::tree::{SessionTree, SessionTreeEntry};

/// Returns the required API key environment variable name for the default provider (`ZEN_PROVIDER`, default opencode-go).
fn provider_key_var() -> &'static str {
    match zen_ai::provider::default_provider_name().as_str() {
        "opencode-go" => "OPENCODE_API_KEY",
        _ => "ARK_API_KEY",
    }
}

// ─── Mock Hooks (lightweight collectors, following zen-core integration test conventions) ──────────────

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

/// Build an AgentState for engine tests (empty tools; fixed system_prompt, messages is the restored history).
///
/// The engine's `Agent::prompt` creates a `Context` on the fly from the state (`create_context`) and drives the low-level loop.
fn agent_state(system_prompt: &str, history: Vec<Message>) -> AgentState {
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

/// Convert a set of messages into a chained (parent_id sequentially linked) session tree Message entries.
/// Reuse zen-app's JSONL id/timestamp generation functions (T-AG-03-3).
fn messages_to_entries(messages: &[Message]) -> Vec<SessionTreeEntry> {
    let mut entries = Vec::with_capacity(messages.len());
    let mut prev: Option<String> = None;
    for msg in messages {
        let id = create_session_id(None);
        let entry = SessionTreeEntry::Message {
            id: id.clone(),
            parent_id: prev.clone(),
            timestamp: create_timestamp(),
            message: msg.clone(),
        };
        prev = Some(id);
        entries.push(entry);
    }
    entries
}

/// Run one real LLM turn with `history` as the existing context, appending `prompt` (string; the Agent constructs the user message internally),
/// and return the messages added this turn. Create a fresh Agent each turn (state.messages = restored history); no reuse across turns.
async fn run_turn_with_history(
    base_prompt: &str,
    history: Vec<Message>,
    prompt: &str,
) -> Vec<Message> {
    let mut agent = Agent::new(
        agent_state(base_prompt, history),
        ToolExecutionMode::Sequential,
        Arc::new(MockHooks::new().with_max_turns(3)),
        real_stream_fn(),
        CancellationToken::new(),
    );
    agent.prompt(prompt).await
}

/// Run one round with a real LLM on empty history.
async fn run_turn(base_prompt: &str, prompt: &str) -> Vec<Message> {
    run_turn_with_history(base_prompt, Vec::new(), prompt).await
}

// ═══════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════

/// Real integration test: session tree ↔ JSONL persistence ↔ build_context restore → cross-turn LLM continuity.
///
/// Skip condition: automatically skipped when the API key env var of the selected default provider is not set.
///
/// Validation:
/// 1. Real LLM round 1 remembers the passphrase ORANGE.
/// 2. `create_session_file` + `append_entry` write the message history into JSONL.
/// 3. `read_session_file` reads back entries and last_leaf_id (counts match).
/// 4. `SessionTree::new` rebuilds the tree → `build_context` restores the message history.
/// 5. Real LLM round 2 follows up based on the restored context and can answer ORANGE (context not lost).
#[tokio::test]
async fn real_test_session_persist_recover_continuity() {
    let key_var = provider_key_var();
    let api_key = std::env::var(key_var);
    if api_key.is_err() {
        eprintln!("{} not set; skipping real integration test", key_var);
        return;
    }

    let base_prompt = "You are a helpful assistant. Answer concisely.";

    // ── Round 1: establish session history (remember the passphrase) ──
    let turn1 = run_turn(
        base_prompt,
        "Remember the codeword ORANGE. Reply with only the word 'remembered'.",
    )
    .await;
    let t1_text = last_assistant_text(&turn1);
    eprintln!("turn1 assistant: {t1_text}");
    assert!(
        !t1_text.is_empty(),
        "turn 1 should produce an assistant reply"
    );
    assert!(
        turn1.len() >= 2,
        "turn 1 should include user + assistant, got {}",
        turn1.len()
    );

    // ── Persist to JSONL (temp session root directory) ──
    let dir = tempfile::tempdir().expect("tempdir");
    let sessions_root = dir.path().to_path_buf();
    let entries = messages_to_entries(&turn1);
    let path = create_session_file(
        &sessions_root,
        "project",
        CreateSessionOptions {
            session_id: None,
            parent_session_path: None,
            metadata: None,
        },
    )
    .expect("create session file");
    for e in &entries {
        append_entry(&path, e).expect("append entry");
    }

    // ── Read back + rebuild tree + build_context ──
    let (_header, read_entries, last_leaf_id) =
        read_session_file(&path).expect("read session file");
    assert_eq!(
        read_entries.len(),
        entries.len(),
        "entries should round-trip"
    );
    assert!(last_leaf_id.is_some(), "last_leaf_id should be derived");

    let tree = SessionTree::new(read_entries, last_leaf_id);
    let session_ctx = build_context(&tree, None);
    assert!(
        !session_ctx.messages.is_empty(),
        "build_context should recover history"
    );

    // ── Round 2: ask for the passphrase based on the restored context ──
    let turn2 = run_turn_with_history(
        base_prompt,
        session_ctx.messages.clone(),
        "What is the codeword?",
    )
    .await;
    let t2_text = last_assistant_text(&turn2);
    eprintln!("turn2 assistant: {t2_text}");
    assert!(
        t2_text.to_uppercase().contains("ORANGE"),
        "recovered context lost the codeword: '{t2_text}'"
    );
}

/// Real integration test: fork branch semantics (JSONL `get_entries_to_fork` + branch context restoration + LLM continuity).
///
/// Skip condition: automatically skipped when the API key env var of the selected default provider is not set.
///
/// Validation:
/// 1. Round 1 remembers ORANGE, round 2 continues remembering PURPLE → chain [u1, a1, u2, a2].
/// 2. `ForkPosition::Before` fork at u2 → branch contains only [u1, a1] (excluding u2 and later).
/// 3. Rebuild tree from branch → build_context → follow-up question in round 3: should remember ORANGE (branch history is before the fork point).
#[tokio::test]
async fn real_test_session_fork_roundtrip() {
    let key_var = provider_key_var();
    let api_key = std::env::var(key_var);
    if api_key.is_err() {
        eprintln!("{} not set; skipping real integration test", key_var);
        return;
    }

    let base_prompt = "You are a helpful assistant. Answer concisely.";

    // Round 1: remember ORANGE
    let turn1 = run_turn(
        base_prompt,
        "Remember the codeword ORANGE. Reply with only the word 'remembered'.",
    )
    .await;

    // Round 2: continue remembering PURPLE based on round 1 context
    let turn2 = run_turn_with_history(
        base_prompt,
        turn1.clone(),
        "Now also remember the codeword PURPLE. Reply with only the word 'remembered'.",
    )
    .await;

    // Full chain [u1, a1, u2, a2]
    let full_chain: Vec<Message> = turn1.iter().chain(turn2.iter()).cloned().collect();
    let entries = messages_to_entries(&full_chain);
    assert!(
        entries.len() >= 4,
        "expected at least 4 entries (u1,a1,u2,a2), got {}",
        entries.len()
    );

    // Locate the 2nd user message (u2) as the fork target
    let user_indices: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter_map(|(i, e)| match e {
            SessionTreeEntry::Message {
                message: Message::User(_),
                ..
            } => Some(i),
            _ => None,
        })
        .collect();
    assert!(user_indices.len() >= 2, "need at least 2 user messages");
    let u2_id = entries[user_indices[1]].id().to_string();

    // Fork (Before): branch = full chain before the fork point ([u1, a1]), excluding u2 and later
    let fork_entries =
        get_entries_to_fork(&entries, Some(&u2_id), ForkPosition::Before).expect("fork");
    assert!(
        fork_entries.iter().all(|e| e.id() != u2_id),
        "fork branch must not include u2"
    );
    assert!(
        fork_entries.len() < entries.len(),
        "fork branch should be a strict subset"
    );
    assert!(
        !fork_entries.is_empty(),
        "fork branch should keep the first turn"
    );

    // Rebuild fork branch tree → build_context → follow-up question in round 3
    let last_id = fork_entries.last().expect("non-empty").id().to_string();
    let fork_tree = SessionTree::new(fork_entries, Some(last_id));
    let fork_ctx = build_context(&fork_tree, None);

    let turn3 = run_turn_with_history(
        base_prompt,
        fork_ctx.messages.clone(),
        "Which codeword do you remember? Name only one.",
    )
    .await;
    let t3_text = last_assistant_text(&turn3);
    eprintln!("fork branch assistant: {t3_text}");
    let upper = t3_text.to_uppercase();
    assert!(
        upper.contains("ORANGE"),
        "fork branch lost the first-turn codeword: '{t3_text}'"
    );
    if upper.contains("PURPLE") {
        eprintln!("WARNING: fork branch mentioned PURPLE (post-fork message leaked)");
    }
}
