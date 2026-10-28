//! Flag example: `--append-system-prompt` — append text to the end of the system prompt.
//! Run: `cargo run -p zen-app --example example_append_system_prompt -- "Always answer in Chinese." "Hello"`

#[path = "_support/common.rs"]
mod common;

use common::*;
use zen_ai::types::ThinkingLevel;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    common::init_catalog();
    let extra = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    let prompt = std::env::args().nth(2).unwrap_or_else(|| "Hello".into());

    // --append-system-prompt semantics: base prompt + appended text (once / can join multiple times)
    let system_prompt =
        format!("You are a helpful assistant.\n\nAdditional instructions:\n{extra}");

    let mut session = make_session(
        common::default_model(),
        system_prompt,
        vec![],
        ThinkingLevel::Off,
    );
    run_and_print(&mut session, &prompt).await
}
