//! Flag example: `--model` — override model id (change id based on the default model).
//! Run: `cargo run -p zen-app --example example_model -- my-model "Hello"`

#[path = "_support/common.rs"]
mod common;

use common::*;
use zen_ai::types::{Model, ThinkingLevel};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    common::init_catalog();
    let model_id = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "deepseek-v4-flash".into());
    let prompt = std::env::args().nth(2).unwrap_or_else(|| "Hello".into());

    // --model semantics (id override): change id after Model::from_env() (--model also supports provider/id and id:thinking shorthand)
    let mut model = Model::from_env();
    model.id = model_id;

    let mut session = make_session(
        model,
        "You are a helpful assistant.".into(),
        vec![],
        ThinkingLevel::Off,
    );
    run_and_print(&mut session, &prompt).await
}
