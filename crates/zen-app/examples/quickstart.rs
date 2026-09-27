//! zen quickstart example — minimal "build AgentSession + real LLM call" pipeline.
//!
//! langchain `createAgent` style, three steps:
//!   1. Pick a model + get the protocol stream implementation (`stream_for(model.api)` binding table decides automatically, no assembly from scratch)
//!   2. Assemble the session (`create_agent_session` = langchain's `createAgent` counterpart)
//!   3. Call `session.prompt(...)` for one real LLM call
//!
//! This is the instructional starting point of the main pipeline; `main.rs`'s `create_initial_runtime_from_cli` merely wraps
//! an extra layer of CLI glue around it (CLI customization for model/tools/system prompt/store), not the main pipeline itself.
//!
//! Run (requires a provider API key, e.g. `export OPENCODE_API_KEY=xxx`):
//! ```sh
//! cargo run -p zen-app --example quickstart -- "Hello, introduce yourself in one sentence"
//! ```

use std::sync::Arc;

use zen_ai::api::stream_for;
use zen_ai::types::{ContentBlock, Message, Model, ThinkingLevel};
use zen_app::config::settings::SettingsManager;
use zen_app::session::agent_session::SessionStore;
use zen_app::session::services::{create_agent_session, NoopHooks};
use zen_app::session::session_manager::SessionManager;
use zen_core::env::ShellConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "hello, briefly introduce yourself".to_string());

    // 1. Pick model + protocol (protocol stream implementation determined by binding table: OpenAiCompletions → openai_completions_stream)
    let model = Model::from_env();
    let stream_fn =
        stream_for(model.api).expect("Protocol pointed to by model.api is not implemented; see stream_for binding table");

    // 2. Assemble session (create_agent_session = createAgent; here: empty tools + empty hooks + in-memory session)
    let cwd = std::env::current_dir()?;
    let cwd_str = cwd.to_string_lossy().into_owned();
    let store: Arc<dyn SessionStore> = Arc::new(SessionManager::in_memory(&cwd_str));

    let mut session = create_agent_session(
        model,
        "You are a concise, helpful coding assistant.".to_string(),
        vec![], // No tools; use ToolFactory::create_all() if tools are needed (see crates/zen-core/src/tools/factory.rs)
        ThinkingLevel::Off,
        Arc::new(NoopHooks),
        stream_fn,
        store,
        ShellConfig::default(),
        Arc::new(SettingsManager::empty()),
        Vec::new(),
        Vec::new(),
    );

    // 3. Real LLM call: send one round of prompt, print the assistant's text reply
    let replies = session.prompt(&prompt, &[]).await?;
    for msg in replies {
        if let Message::Assistant(assistant) = msg {
            for block in assistant.content {
                if let ContentBlock::Text(text) = block {
                    println!(
                        "{}{}",
                        text.text,
                        if text.text.ends_with('\n') { "" } else { "\n" }
                    );
                }
            }
        }
    }
    Ok(())
}
