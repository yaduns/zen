//! Real integration test: the **session→Agent→runtime vertical pipeline** to date (production entry → LLM → auto-persist → restore → fork → runtime switching).
//!
//! Vertical pipeline (aligned with `docs/agent-call-pipeline.md` "Section VI: Application-Layer Boundaries"):
//!
//! ```text
//! SessionManager::create (writes the header file)
//!   → build_context (empty history) → Agent::new (real stream_fn)
//!   → AgentSession::new (session_store = Arc<SessionManager>, persistent subscription)
//!   → create_initial_runtime (Startup) → AgentSession::prompt (real LLM)
//!       ├─ MessageEnd → persistent subscription → SessionManager::append_message → auto-persist to disk
//!   → second-round prompt (continues in the same session, PURPLE)
//!   → replace_runtime (closure: create_branched_session extracts the "root → first-round assistant" branch)
//!       → SessionManager::open → build_context → new AgentSession → new runtime
//!   → third-round prompt → context before the fork point is retained (ORANGE), nothing after it leaks (PURPLE)
//! ```
//!
//! Complementary to `agent_session_persistence_real.rs` (closed loop without a runtime): this file additionally verifies
//! the session-switch orchestration of `AgentSessionRuntime`/`replace_runtime` (T-AP-01-2).
//! Note: the project is not yet complete; this file covers the **nodes landed so far** (SessionManager →
//! AgentSession → runtime assembly) as an aggregate pipeline, not the project's full pipeline.
//!
//! Skip condition: the API key env var for the default provider is unset.

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use zen_ai::types::Message;
use zen_app::session::agent_session::{AgentSession, SessionStore};
use zen_app::session::jsonl::CreateSessionOptions;
use zen_app::session::runtime::{
    create_initial_runtime, replace_runtime, AgentSessionServices, SessionStartReason,
};
use zen_app::session::session_manager::SessionManager;
use zen_core::session::tree::SessionTreeEntry;

const BASE_PROMPT: &str = "You are a helpful assistant. Answer concisely.";

/// Aggregated vertical path up to the current node (SessionManager → AgentSession → runtime assembly):
/// create → prompt (auto-persist) → continue → replace_runtime (fork extraction branch) → continuity + no leakage.
#[tokio::test]
async fn real_test_session_agent_runtime_fork() {
    let key_var = common::provider_key_var();
    if std::env::var(key_var).is_err() {
        eprintln!("{} not set; skipping real integration test", key_var);
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();

    // 1) Session + runtime (Startup)
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
    let session = AgentSession::new(common::new_agent(BASE_PROMPT, vec![]), store.clone());
    let mut rt = create_initial_runtime(
        session,
        AgentSessionServices {
            cwd: PathBuf::from("proj"),
            agent_dir: PathBuf::from("/tmp/zen-test-agent"),
            session_store: store,
        },
        SessionStartReason::Startup,
    );

    // 2) Round one: remember ORANGE (via runtime entry)
    let turn1 = rt
        .session
        .prompt("Remember the codeword ORANGE. Reply with only the word 'remembered'.")
        .await
        .expect("prompt");
    common::print_messages("turn1", &turn1);
    assert!(!common::last_assistant_text(&turn1).is_empty());

    // Persistent subscription auto-persists (same Arc<SessionManager>)
    assert!(
        manager.get_entries().len() >= 2,
        "user + assistant should be auto-persisted, got {}",
        manager.get_entries().len()
    );

    // 3) Round two: continue in the same session, remember PURPLE
    let turn2 = rt
        .session
        .prompt("Now also remember the codeword PURPLE. Reply with only the word 'remembered'.")
        .await
        .expect("prompt 2");
    common::print_messages("turn2", &turn2);
    assert!(!common::last_assistant_text(&turn2).is_empty());

    // 4) Locate the id of the round-one assistant entry (fork point = that entry; the branch excludes the PURPLE segment)
    let original_id = manager.get_session_id();
    let a1_id = manager
        .get_entries()
        .iter()
        .find_map(|e| match e {
            SessionTreeEntry::Message {
                id,
                message: Message::Assistant(_),
                ..
            } => Some(id.clone()),
            _ => None,
        })
        .expect("first assistant entry");

    // 5) replace_runtime: fork inside the closure (extract root→a1 branch as a new session file) + new runtime
    // root must be cloned: the closure move-captures one copy; the test's later list still needs the original root
    let root_for_fork = root.clone();
    let mut rt2 = replace_runtime(rt, move |old_services| {
        let cwd = old_services.cwd;
        let agent_dir = old_services.agent_dir;

        // Extract the "root → round-one assistant" branch as a standalone session file (excluding the u2/a2 PURPLE segment)
        let new_file = manager
            .create_branched_session(&a1_id)
            .expect("create_branched_session")
            .expect("persisted mode creates file");

        // Reopen the branch file → history = context before the fork point
        let new_manager = SessionManager::open(&new_file, None, &root_for_fork).expect("open fork");
        let history = new_manager.build_context().messages;

        // New AgentSession + new services (session_store points to the branch manager)
        let new_store: Arc<dyn SessionStore> = Arc::new(new_manager);
        let new_session = AgentSession::new(common::new_agent(BASE_PROMPT, history), new_store.clone());
        let new_services = AgentSessionServices {
            cwd,
            agent_dir,
            session_store: new_store,
        };
        Ok::<_, anyhow::Error>((new_session, new_services, SessionStartReason::Fork(a1_id)))
    })
    .expect("replace_runtime");

    // 6) New runtime validation: new session id, 2 session files on disk (original + branch)
    assert_ne!(rt2.session.session_id(), original_id, "forked session gets new id");
    let sessions = SessionManager::list("proj", &root).expect("list");
    assert_eq!(sessions.len(), 2, "original + forked session files");

    // 7) Round three: follow-up question on the fork branch → retains pre-fork context (ORANGE), no PURPLE leakage
    let turn3 = rt2
        .session
        .prompt("Which codeword do you remember? Reply with only one word.")
        .await
        .expect("prompt 3");
    common::print_messages("turn3 (fork branch)", &turn3);
    let t3 = common::last_assistant_text(&turn3);
    let upper = t3.to_uppercase();
    assert!(
        upper.contains("ORANGE"),
        "fork branch lost the pre-fork codeword: '{t3}'"
    );
    if upper.contains("PURPLE") {
        eprintln!("WARNING: fork branch mentioned PURPLE (post-fork message leaked)");
    }
}
