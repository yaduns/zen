//! Flag example: `--provider` — rebuild the model by provider name (Model::from_provider).
//! Run: `cargo run -p zen-app --example example_provider -- huoshan "Describe Volcano Ark in one sentence"`

#[path = "_support/common.rs"]
mod common;

use common::*;
use zen_ai::provider::resolve_provider;
use zen_ai::types::ThinkingLevel;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    common::init_catalog();
    let provider = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "opencode-go".into());
    let prompt = std::env::args().nth(2).unwrap_or_else(|| "Hello".into());

    // --provider semantics: resolve_provider(name) → Model::from_provider
    let provider_box = resolve_provider(&provider)
        .ok_or_else(|| anyhow::anyhow!("unknown provider: {provider}"))?;
    let model = zen_ai::types::Model::from_provider(provider_box.as_ref());

    let mut session = make_session(
        model,
        "You are a helpful assistant.".into(),
        vec![],
        ThinkingLevel::Off,
    );
    run_and_print(&mut session, &prompt).await
}
