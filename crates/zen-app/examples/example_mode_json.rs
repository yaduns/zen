//! Flag example: `--mode json` (JSON event stream in print mode).
//! Goes directly through the production print outlet `run_print_mode(Mode::Json, ...)`: each AgentEvent is serialized to one JSON line.
//! Construct `AgentSessionRuntime` first (using the existing public assembly API).
//! Run: `cargo run -p zen-app --example example_mode_json -- "Hello, introduce yourself in one sentence"`

#[path = "_support/common.rs"]
mod common;

use std::path::PathBuf;
use std::sync::Arc;

use common::*;
use zen_ai::api::stream_for;
use zen_ai::types::{Model, ThinkingLevel};
use zen_app::cli::Mode;
use zen_app::config::get_agent_dir;
use zen_app::config::settings::SettingsManager;
use zen_app::modes::print_mode::run_print_mode;
use zen_app::session::agent_session::SessionStore;
use zen_app::session::runtime::{create_initial_runtime, AgentSessionServices, SessionStartReason};
use zen_app::session::services::{create_agent_session, NoopHooks};
use zen_app::session::session_manager::SessionManager;
use zen_core::env::ShellConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let prompt = std::env::args().nth(1).unwrap_or_else(|| "Hello".into());

    let model = Model::from_env();
    let stream_fn = stream_for(model.api).expect("protocol not implemented");
    let cwd = cwd_str();
    let store: Arc<dyn SessionStore> = Arc::new(SessionManager::in_memory(&cwd));

    let session = create_agent_session(
        model,
        "You are a helpful assistant.".into(),
        vec![],
        ThinkingLevel::Off,
        Arc::new(NoopHooks),
        stream_fn,
        store.clone(),
        ShellConfig::default(),
        Arc::new(SettingsManager::empty()),
        Vec::new(),
        Vec::new(),
    );

    let services = AgentSessionServices {
        cwd: PathBuf::from(&cwd),
        agent_dir: get_agent_dir()?,
        session_store: store,
        registry: zen_core::process::ProcessRegistry::new(),
    };
    let runtime =
        create_initial_runtime(session, services, SessionStartReason::Startup, Vec::new());

    // --mode json: use the print outlet; events emitted one JSON per line.
    let exit = run_print_mode(runtime, Mode::Json, vec![], Some(prompt), vec![]).await?;
    eprintln!("[example] run_print_mode exit code: {exit}");
    Ok(())
}
