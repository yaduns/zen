//! Real integration test: AgentSession production object-level closed loop (T-AP-01-1 / T-AP-01-4).
//!
//! Complementary to `session_persistence_real.rs` (jsonl / SessionTree **primitive-level**, test assembles manually):
//! This file goes through **production objects** — `SessionManager::create` → `build_context` injects history →
//! `AgentSession::new` → `prompt` → **resident subscription auto-persists** (MessageEnd → append_message) →
//! `open` reopens and restores → LLM continuity.
//!
//! Skip condition: the API key env var for the default provider is unset.

mod common;

use std::sync::Arc;
use zen_app::session::agent_session::{AgentSession, SessionStore};
use zen_app::session::jsonl::{read_session_file, CreateSessionOptions};
use zen_app::session::session_manager::SessionManager;

const BASE_PROMPT: &str = "You are a helpful assistant. Answer concisely.";

/// Real closed loop: AgentSession::prompt → resident subscription auto-persist → reopen restore → continuity.
#[tokio::test]
async fn real_test_agent_session_persist_and_resume_continuity() {
    let key_var = common::provider_key_var();
    if std::env::var(key_var).is_err() {
        eprintln!("{} not set; skipping real integration test", key_var);
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();

    // 1) Create session (real file, write header)
    let manager = Arc::new(
        SessionManager::create(
            "proj",
            &root,
            CreateSessionOptions {
                session_id: None,
                parent_session_path: None,
                metadata: None,
            },
        )
        .expect("create"),
    );
    let store: Arc<dyn SessionStore> = manager.clone();

    // 2) Empty history → AgentSession (persistent subscription: MessageEnd → append_message)
    let mut session = AgentSession::new(common::new_agent(BASE_PROMPT, vec![]), store);

    // 3) Real prompt: remember the passphrase
    let turn1 = session
        .prompt("Remember the codeword ORANGE. Reply with only the word 'remembered'.")
        .await
        .expect("prompt");
    common::print_messages("turn1", &turn1);
    let t1 = common::last_assistant_text(&turn1);
    assert!(!t1.is_empty(), "turn 1 should produce an assistant reply");
    drop(session);

    // 4) Persistent subscription auto-persists: MessageEnd → SessionManager::append_message (same Arc instance)
    let entries = manager.get_entries();
    assert!(
        entries.len() >= 2,
        "user + assistant should be auto-persisted, got {}",
        entries.len()
    );
    let file = manager.get_session_file().expect("persisted file");
    let (_h, file_entries, _leaf) = read_session_file(&file).expect("read back");
    assert_eq!(
        file_entries.len(),
        entries.len(),
        "file entries match in-memory tree"
    );

    // 5) Reopen and restore: open → build_context → new AgentSession → continuity
    let reopened = SessionManager::open(&file, Some("proj"), &root).expect("open");
    let history = reopened.build_context().messages;
    assert!(!history.is_empty(), "recovered history should not be empty");

    let mut session2 = AgentSession::new(common::new_agent(BASE_PROMPT, history), Arc::new(reopened));
    let turn2 = session2
        .prompt("What is the codeword? Reply with only the word.")
        .await
        .expect("prompt 2");
    common::print_messages("turn2 (resumed)", &turn2);
    let t2 = common::last_assistant_text(&turn2);
    assert!(
        t2.to_uppercase().contains("ORANGE"),
        "recovered context lost the codeword: '{t2}'"
    );
}

/// Real restore: the same session file is restored via `continue_recent` and used for follow-up questions (cross-instance continuity).
#[tokio::test]
async fn real_test_agent_session_continue_recent_continuity() {
    let key_var = common::provider_key_var();
    if std::env::var(key_var).is_err() {
        eprintln!("{} not set; skipping real integration test", key_var);
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();

    // Round 1: remember the passphrase
    {
        let manager = Arc::new(
            SessionManager::create(
                "proj",
                &root,
                CreateSessionOptions {
                    session_id: None,
                    parent_session_path: None,
                    metadata: None,
                },
            )
            .expect("create"),
        );
        let store: Arc<dyn SessionStore> = manager.clone();
        let mut session = AgentSession::new(common::new_agent(BASE_PROMPT, vec![]), store);
        let turn1 = session
            .prompt("Remember the codeword BLUE. Reply with only the word 'remembered'.")
            .await
            .expect("prompt");
        common::print_messages("turn1", &turn1);
        assert!(!common::last_assistant_text(&turn1).is_empty());
    } // manager/session dropped, only the on-disk file remains

    // restore the most recent session via continue_recent
    let resumed = Arc::new(SessionManager::continue_recent("proj", &root).expect("continue"));
    assert_eq!(resumed.get_cwd(), "proj");
    assert!(!resumed.get_entries().is_empty(), "entries restored from disk");

    // rebuild AgentSession from the restored history, follow-up continuity
    let history = resumed.build_context().messages;
    let store: Arc<dyn SessionStore> = resumed.clone();
    let mut session2 = AgentSession::new(common::new_agent(BASE_PROMPT, history), store);
    let turn2 = session2
        .prompt("What is the codeword? Reply with only the word.")
        .await
        .expect("prompt 2");
    common::print_messages("turn2 (continue_recent)", &turn2);
    let t2 = common::last_assistant_text(&turn2);
    assert!(
        t2.to_uppercase().contains("BLUE"),
        "continue_recent lost the codeword: '{t2}'"
    );
}
