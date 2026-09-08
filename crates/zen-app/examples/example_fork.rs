//! Flag example: `--fork <source>` — fork from an existing session and **continue with its history**.
//!
//! Demo (real LLM):
//!   1. Create a source session and append a user message (persisted to JSONL)
//!   2. `SessionManager::fork_from` forks → the new session carries the source history
//!   3. `build_context()` restores history + `set_messages` injects → LLM sees prior conversation
//!   4. Send a follow-up question; the LLM answers based on the forked history
//!
//! Run:
//!   # Use a temp directory (auto-cleaned after run)
//!   cargo run -p zen-app --example example_fork -- "What was the passphrase I mentioned earlier?"
//!
//!   # Specify a directory (files persist, inspectable)
//!   cargo run -p zen-app --example example_fork -- --session-dir /tmp/zen-fork-demo "What was the passphrase I mentioned earlier?"

#[path = "_support/common.rs"]
mod common;

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use zen_ai::api::stream_for;
use zen_ai::types::{ContentBlockList, Message, Model, ThinkingLevel, UserMessage};
use zen_app::session::agent_session::SessionStore;
use zen_app::session::jsonl::CreateSessionOptions;
use zen_app::session::services::{create_agent_session, NoopHooks};
use zen_app::session::session_manager::SessionManager;
use zen_core::env::ShellConfig;

#[derive(Parser)]
#[command(name = "example_fork")]
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

    // 1. Create a source session and append a history entry (persisted to JSONL)
    let source = SessionManager::create(
        &cwd,
        &root,
        CreateSessionOptions {
            session_id: None,
            parent_session_path: None,
            metadata: None,
        },
    )?;
    source.append_message(&Message::User(UserMessage {
        content: ContentBlockList::Text("Please remember the passphrase: blue parrot.".into()),
        timestamp: 0,
    }));
    eprintln!(
        "[example] Appended 1 history message to source session {}",
        source.session_id()
    );

    // 2. Fork from the source session (fork_from copies all entries to a new file with a new id)
    let source_path = source.get_session_file().expect("File session should have a path");
    let forked = SessionManager::fork_from(
        &source_path,
        &cwd,
        &root,
        CreateSessionOptions {
            session_id: None,
            parent_session_path: None,
            metadata: None,
        },
    )?;
    let forked_id = forked.session_id();
    eprintln!("[example] Forked new session {forked_id}");

    // 3. build_context restores history + set_messages injects (aligned with pi)
    let ctx = forked.build_context();
    eprintln!("[example] Forked session restored {} history messages", ctx.messages.len());

    let model = Model::from_env();
    let stream_fn = stream_for(model.api).expect("protocol not implemented");
    let mut session = create_agent_session(
        model,
        "You are a helpful assistant.".into(),
        vec![],
        ThinkingLevel::Off,
        Arc::new(NoopHooks),
        stream_fn,
        Arc::new(forked) as Arc<dyn SessionStore>,
        ShellConfig::default(),
        Vec::new(),
        Vec::new(),
    );
    session.set_messages(ctx.messages); // ← history replay

    // 4. Real LLM call: the LLM should answer based on the forked history ("blue parrot")
    common::run_and_print(&mut session, &prompt).await
}
