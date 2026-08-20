//! Flag example: `--thinking` — set the thinking level (ThinkingLevel).
//! Run: `cargo run -p zen-app --example example_thinking -- high "solve a complex reasoning problem"`

#[path = "_support/common.rs"]
mod common;

use common::*;
use zen_ai::types::ThinkingLevel;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let level = std::env::args().nth(1).unwrap_or_else(|| "off".into());
    let prompt = std::env::args().nth(2).unwrap_or_else(|| "Hello".into());

    // --thinking semantics: parse the CLI value into ThinkingLevel and inject into the session
    let thinking = match level.as_str() {
        "off" => ThinkingLevel::Off,
        "minimal" => ThinkingLevel::Minimal,
        "low" => ThinkingLevel::Low,
        "medium" => ThinkingLevel::Medium,
        "high" => ThinkingLevel::High,
        "xhigh" => ThinkingLevel::XHigh,
        "max" => ThinkingLevel::Max,
        other => anyhow::bail!(
            "invalid thinking level: {other} (valid: off/minimal/low/medium/high/xhigh/max)"
        ),
    };

    let mut session = make_session(
        common::default_model(),
        "You are a helpful assistant.".into(),
        vec![],
        thinking,
    );
    run_and_print(&mut session, &prompt).await
}
