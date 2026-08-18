//! Integration test: SessionManager lifecycle (real fs, **does not access LLM**).
//!
//! Covers the cross-instance contract of T-AP-01-4—the parts unit tests (same-process objects) cannot test:
//! - create → append → persist → reopen with open (cross-instance persistence roundtrip)
//! - continue_recent restores the most recent session
//! - fork_from copies history across directories
//! - create_branched_session extracts branch path
//! - branch / reset_leaf semantics
//! - build_context state derivation (thinking_level / model)
//! - list / list_all listing
//!
//! Real-LLM fork continuity verification is in `session_agent_runtime_real.rs` (this file does not depend on an API key and is CI-stable).

use std::path::{Path, PathBuf};
use zen_ai::types::{ContentBlockList, Message, UserMessage};
use zen_app::session::jsonl::{read_session_file, CreateSessionOptions};
use zen_app::session::session_manager::SessionManager;

// ─── Test helpers ───────────────────────────────────────────────

/// Create a unique test directory under the system temp directory (std-only).
fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "zen_it_lifecycle_{}_{}",
        name,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn user_message(text: &str) -> Message {
    Message::User(UserMessage {
        content: ContentBlockList::Text(text.to_string()),
        timestamp: 0,
    })
}

fn create_in(root: &Path, cwd: &str, id: Option<&str>) -> SessionManager {
    SessionManager::create(
        cwd,
        root,
        CreateSessionOptions {
            session_id: id.map(String::from),
            parent_session_path: None,
            metadata: None,
        },
    )
    .expect("create should succeed")
}

// ─── Cross-instance persistence roundtrip ──────────────────────────────────────

#[test]
fn test_persist_roundtrip_across_instances() {
    let root = temp_dir("roundtrip");
    let mgr = create_in(&root, "proj", Some("sess1"));
    mgr.append_message(&user_message("hello"));
    mgr.append_message(&user_message("world"));
    let path = mgr.get_session_file().expect("persisted file");
    assert!(path.exists());
    drop(mgr); // Destroy the instance; verify rebuild from disk only

    let reopened = SessionManager::open(&path, None, &root).expect("open");
    assert_eq!(reopened.get_session_id(), "sess1");
    assert_eq!(reopened.get_entries().len(), 2);
    let _ = std::fs::remove_dir_all(&root);
}

// ─── continue_recent ─────────────────────────────────────────────

#[test]
fn test_continue_recent_resumes_latest() {
    let root = temp_dir("continue");
    let mgr = create_in(&root, "proj", Some("s1"));
    mgr.append_message(&user_message("hi"));
    drop(mgr);

    let resumed = SessionManager::continue_recent("proj", &root).expect("continue");
    assert_eq!(resumed.get_session_id(), "s1");
    assert_eq!(resumed.get_entries().len(), 1);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn test_continue_recent_creates_when_empty() {
    let root = temp_dir("continue_empty");
    let mgr = SessionManager::continue_recent("proj", &root).expect("continue");
    assert!(mgr.get_session_file().is_some());
    assert!(mgr.get_session_file().unwrap().exists());
    let _ = std::fs::remove_dir_all(&root);
}

// ─── fork_from ───────────────────────────────────────────────────

#[test]
fn test_fork_from_copies_history_to_new_cwd() {
    let root = temp_dir("fork");
    let src = create_in(&root, "/src", Some("src1"));
    src.append_message(&user_message("one"));
    src.append_message(&user_message("two"));
    let src_path = src.get_session_file().expect("source file");

    let forked = SessionManager::fork_from(
        &src_path,
        "/target",
        &root,
        CreateSessionOptions {
            session_id: None,
            parent_session_path: None,
            metadata: None,
        },
    )
    .expect("fork_from");
    assert_eq!(forked.get_cwd(), "/target");
    assert_eq!(forked.get_entries().len(), 2);
    assert_ne!(forked.get_session_id(), "src1");
    let _ = std::fs::remove_dir_all(&root);
}

// ─── create_branched_session (data-level branch extraction) ────────────────────

#[test]
fn test_create_branched_session_extracts_path() {
    let root = temp_dir("branch");
    let mgr = create_in(&root, "proj", Some("s1"));
    let m1 = mgr.append_message(&user_message("one"));
    mgr.append_message(&user_message("two"));
    mgr.append_message(&user_message("three"));

    // Extract the branch from root → second message
    let m2 = mgr.get_entries()[1].id().to_string();
    let new_file = mgr
        .create_branched_session(&m2)
        .expect("create_branched_session")
        .expect("persisted mode creates file");

    let (_h, entries, _leaf) = read_session_file(&new_file).expect("read fork file");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].id(), m1);
    assert_eq!(entries[1].id(), m2);
    let _ = std::fs::remove_dir_all(&root);
}

// ─── branch / reset_leaf semantics ────────────────────────────────────

#[test]
fn test_branch_and_reset_leaf_semantics() {
    let mgr = SessionManager::in_memory("proj");
    let m1 = mgr.append_message(&user_message("one"));
    mgr.append_message(&user_message("two"));

    mgr.branch(&m1).expect("branch");
    assert_eq!(mgr.get_leaf_id().as_deref(), Some(m1.as_str()));
    let m3 = mgr.append_message(&user_message("three"));
    assert_eq!(mgr.get_entry(&m3).unwrap().parent_id(), Some(m1.as_str()));

    mgr.reset_leaf();
    assert!(mgr.get_leaf_id().is_none());
    let m4 = mgr.append_message(&user_message("four"));
    assert_eq!(mgr.get_entry(&m4).unwrap().parent_id(), None);
}

// ─── build_context state derivation ──────────────────────────────────────

#[test]
fn test_build_context_derives_thinking_and_model() {
    let mgr = SessionManager::in_memory("proj");
    mgr.append_message(&user_message("hello"));
    mgr.append_thinking_level_change("high");
    mgr.append_model_change("deepseek", "deepseek-v4-flash");

    let ctx = mgr.build_context();
    assert_eq!(ctx.thinking_level, "high");
    assert!(
        !ctx.messages.is_empty(),
        "user message should be recovered into context"
    );
    let model = ctx.model.expect("model derived from model_change");
    assert_eq!(model.provider, "deepseek");
    assert_eq!(model.model_id, "deepseek-v4-flash");
}

// ─── list / list_all ─────────────────────────────────────────────

#[test]
fn test_list_and_list_all() {
    let root = temp_dir("list");
    create_in(&root, "/a", Some("s1"));
    create_in(&root, "/b", Some("s2"));

    assert_eq!(SessionManager::list("/a", &root).expect("list a").len(), 1);
    assert_eq!(SessionManager::list("/b", &root).expect("list b").len(), 1);
    assert_eq!(SessionManager::list_all(&root).expect("list_all").len(), 2);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn test_list_empty_dir_returns_empty() {
    let root = temp_dir("list_empty");
    assert!(SessionManager::list("/proj", &root).expect("list").is_empty());
    let _ = std::fs::remove_dir_all(&root);
}
