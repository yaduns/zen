//! Shared boilerplate for zen examples: 2–3-step starter encapsulating "build a session + run one real LLM turn",
//! reused by each feature-flag example. Uses only the existing public APIs of zen-app / zen-core / zen-ai; no source changes.
//! Each example includes this via `#[path = "common.rs"] mod common;`; unused helpers are globally
//! allowed in this file to avoid dead_code warnings for any individual example.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Arc;

use zen_ai::api::stream_for;
use zen_ai::tool::Tool;
use zen_ai::types::{ContentBlock, Message, Model, ThinkingLevel};
use zen_app::session::agent_session::{AgentSession, SessionStore};
use zen_app::session::services::{create_agent_session, NoopHooks};
use zen_app::session::session_manager::SessionManager;
use zen_core::env::ShellConfig;

/// Current working directory as a string.
pub fn cwd_str() -> String {
    std::env::current_dir()
        .expect("current dir")
        .to_string_lossy()
        .into_owned()
}

/// Construct an in-memory session store (SessionManager::in_memory implements SessionStore).
pub fn in_memory_store(cwd: &str) -> Arc<dyn SessionStore> {
    Arc::new(SessionManager::in_memory(cwd))
}

/// Shortcut to build an in-memory store from cwd only.
pub fn store() -> Arc<dyn SessionStore> {
    in_memory_store(&cwd_str())
}

/// Core assembly (langchain createAgent equivalent): model/prompt/tools/thinking + empty hooks + in-memory store.
/// Protocol stream implementation is determined automatically by the `stream_for(model.api)` binding table.
pub fn make_session(
    model: Model,
    system_prompt: String,
    tools: Vec<Arc<dyn Tool>>,
    thinking: ThinkingLevel,
) -> AgentSession {
    let stream_fn =
        stream_for(model.api).expect("Protocol pointed to by model.api is not implemented; see stream_for binding table");
    create_agent_session(
        model,
        system_prompt,
        tools,
        thinking,
        Arc::new(NoopHooks),
        stream_fn,
        store(),
        ShellConfig::default(),
        Vec::new(),
        Vec::new(),
    )
}

/// Run one real LLM prompt and print all of the assistant's text blocks.
pub async fn run_and_print(session: &mut AgentSession, prompt: &str) -> anyhow::Result<()> {
    let replies = session.prompt(prompt).await?;
    for msg in replies {
        if let Message::Assistant(assistant) = msg {
            for block in assistant.content {
                if let ContentBlock::Text(text) = block {
                    let s = if text.text.ends_with('\n') {
                        text.text.clone()
                    } else {
                        format!("{}\n", text.text)
                    };
                    print!("{s}");
                }
            }
        }
    }
    Ok(())
}

/// Default model (parsed from env) — starting model for most examples.
pub fn default_model() -> Model {
    Model::from_env()
}

/// Build a session with the given session store and run one real LLM round (for `store switching` examples like session_dir / resume / fork).
/// Note: in Phase 1 `create_agent_session` starts from empty history (history injection is not wired), so the store is only used for persistence/tree structure.
pub async fn run_with_store(
    store: Arc<dyn SessionStore>,
    prompt: &str,
    base_prompt: &str,
) -> anyhow::Result<()> {
    let model = Model::from_env();
    let stream_fn = stream_for(model.api).expect("protocol not implemented");
    let mut session = create_agent_session(
        model,
        base_prompt.to_string(),
        vec![],
        ThinkingLevel::Off,
        Arc::new(NoopHooks),
        stream_fn,
        store,
        ShellConfig::default(),
        Vec::new(),
        Vec::new(),
    );
    run_and_print(&mut session, prompt).await
}

/// (Tool factory) Assemble the tool list from params; empty tools string → no tools (matches --no-tools semantics).
/// Tool filtering (allow/block lists) is not wired in Phase 1's `create_initial_runtime_from_cli`; this only demonstrates
/// common usage of `ToolFactory` (create_all / create_readonly).
pub fn tool_registry_tools(which: &str) -> Vec<Arc<dyn Tool>> {
    use zen_ai::tool::ToolContext;
    use zen_core::tools::factory::ToolFactory;
    let factory = ToolFactory::new(ToolContext::new(PathBuf::from(cwd_str()), None, None));
    let reg = match which {
        "readonly" => factory.create_readonly(),
        _ => factory.create_all(),
    };
    reg.tools
}
