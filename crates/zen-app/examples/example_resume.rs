//! Flag examples: `--resume` / `--continue` — resume the most recent session and **continue with history**.
//!
//! Demo (real LLM):
//!   1. Create a session, append one user message (persisted to JSONL), then drop (simulating exit)
//!   2. `SessionManager::continue_recent` reopens that session (restoring tree + header)
//!   3. `build_context()` restores history + `set_messages` injects → LLM sees prior conversation
//!   4. Send a follow-up question; the LLM answers based on the restored history
//!
//! Run:
//!   # Use a temp directory (auto-cleaned after run)
//!   cargo run -p zen-app --example example_resume -- "What was the code phrase I just said?"
//!
//!   # Specify a directory (files persist, inspectable)
//!   cargo run -p zen-app --example example_resume -- --session-dir /tmp/zen-resume-demo "What was the code phrase I just said?"

#[path = "_support/common.rs"]
mod common;

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use zen_ai::api::stream_for;
use zen_ai::types::{ContentBlockList, Message, Model, ThinkingLevel, UserMessage};
use zen_app::config::settings::SettingsManager;
use zen_app::session::agent_session::SessionStore;
use zen_app::session::jsonl::CreateSessionOptions;
use zen_app::session::services::{create_agent_session, NoopHooks};
use zen_app::session::session_manager::SessionManager;
use zen_core::env::ShellConfig;

#[derive(Parser)]
#[command(name = "example_resume")]
struct Args {
    /// Session storage directory (temp directory if unspecified, auto-cleaned after run)
    #[arg(long)]
    session_dir: Option<PathBuf>,
    /// Prompt sent to the LLM
    #[arg(trailing_var_arg = true)]
    prompt: Vec<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let prompt = if args.prompt.is_empty() {
        "What was the passphrase I just said?".to_string()
    } else {
        args.prompt.join(" ")
    };

    // Session root: --session-dir specified (persistent, inspectable) or temp directory (auto-cleaned)
    let td;
    let root = match args.session_dir {
        Some(d) => d,
        None => {
            td = tempfile::tempdir()?;
            td.path().join("sessions")
        }
    };
    let cwd = common::cwd_str();

    eprintln!("[example] sessions directory: {}", root.display());

    // 1. Create a session and append history (persisted to JSONL), then drop (simulating process exit)
    {
        let manager = SessionManager::create(
            &cwd,
            &root,
            CreateSessionOptions {
                session_id: None,
                parent_session_path: None,
                metadata: None,
            },
        )?;
        manager.append_message(&Message::User(UserMessage {
            content: ContentBlockList::Text("Please remember the passphrase: blue parrot.".into()),
            timestamp: 0,
        }));
        eprintln!(
            "[example] Created session {} and appended 1 history message, then drop (simulating exit)",
            manager.session_id()
        );
    } // manager is dropped here; the file is already flushed to disk

    // 2. continue_recent: restore the most recent session under cwd (restoring tree + header)
    let resumed = SessionManager::continue_recent(&cwd, &root)?;
    eprintln!("[example] Restored session {}", resumed.session_id());

    // 3. build_context restores history + set_messages injects (aligned with pi)
    let ctx = resumed.build_context();
    eprintln!("[example] Restored {} history messages", ctx.messages.len());

    let model = Model::from_env();
    let stream_fn = stream_for(model.api).expect("protocol not implemented");
    let mut session = create_agent_session(
        model,
        "You are a helpful assistant.".into(),
        vec![],
        ThinkingLevel::Off,
        Arc::new(NoopHooks),
        stream_fn,
        Arc::new(resumed) as Arc<dyn SessionStore>,
        ShellConfig::default(),
        Arc::new(SettingsManager::empty()),
        Vec::new(),
        Vec::new(),
    );
    session.set_messages(ctx.messages); // ← history replay

    // 4. Real LLM call: the LLM should answer based on the restored history ("blue parrot")
    common::run_and_print(&mut session, &prompt).await
}
