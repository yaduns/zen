//! Flag example: `--session-dir <dir>` — specify the session storage directory.
//! Use `SessionManager::create(cwd, sessions_root, opts)` to create a file session in the given directory and use it as the store.
//! Note: Phase 1 history injection is not wired (see example_fork notes), so the LLM's first-turn context has empty history;
//! This example demonstrates the "specify session root directory, persist JSONL to disk" capability.
//! Run: `cargo run -p zen-app --example example_session_dir -- /tmp/zen-session-dir "hello"`

#[path = "_support/common.rs"]
mod common;

use std::path::PathBuf;
use std::sync::Arc;

use common::*;
use zen_app::session::agent_session::SessionStore;
use zen_app::session::jsonl::CreateSessionOptions;
use zen_app::session::session_manager::SessionManager;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/zen-session-dir".into());
    let prompt = std::env::args().nth(2).unwrap_or_else(|| "Hello".into());
    let cwd = cwd_str();

    // --session-dir semantics: point sessions_root at dir, create (and persist) a file session
    let manager = SessionManager::create(
        &cwd,
        PathBuf::from(&dir).as_path(),
        CreateSessionOptions {
            session_id: None,
            parent_session_path: None,
            metadata: None,
        },
    )?;
    eprintln!(
        "[example] Session root dir: {dir}; session_id: {}",
        manager.session_id()
    );

    let store: Arc<dyn SessionStore> = Arc::new(manager);
    run_with_store(store, &prompt, "You are a helpful assistant.").await
}
