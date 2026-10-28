//! Flag example: `--system-prompt` — replace the default with a custom system prompt.
//! Run: `cargo run -p zen-app --example example_system_prompt -- "You are a SQL expert." "write a SQL query against the users table"`

#[path = "_support/common.rs"]
mod common;

use common::*;
use zen_ai::types::ThinkingLevel;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    common::init_catalog();
    let custom = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "You are a helpful assistant.".into());
    let prompt = std::env::args().nth(2).unwrap_or_else(|| "Hello".into());

    // --system-prompt semantics: use the custom prompt as AgentState.system_prompt
    let mut session = make_session(common::default_model(), custom, vec![], ThinkingLevel::Off);
    run_and_print(&mut session, &prompt).await
}
