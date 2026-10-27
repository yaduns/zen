//! Flag example: `--no-session` — do not save the session (temporary session).
//! Align with `SessionManager::in_memory`: in-memory session, not persisted to JSONL.
//! Run: `cargo run -p zen-app --example example_no_session -- "Hello"`

#[path = "_support/common.rs"]
mod common;

use common::*;
use zen_ai::types::ThinkingLevel;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let prompt = std::env::args().nth(1).unwrap_or_else(|| "Hello".into());

    // --no-session semantics: use the in_memory store (no file writes); see example_session_dir for the file-session counterpart.
    let _store = in_memory_store(&cwd_str());

    let mut session = make_session(
        common::default_model(),
        "You are a helpful assistant.".into(),
        vec![],
        ThinkingLevel::Off,
    );
    run_and_print(&mut session, &prompt).await
}
