//! Flag example: `--list-models` — list available models (this example demonstrates printing the provider→model mapping).
//! Run: `cargo run -p zen-app --example example_list_models -- [search string]`

fn main() -> anyhow::Result<()> {
    let q = std::env::args().nth(1);

    // --list-models semantics: list selectable models. Phase 1 models come from “provider + each provider's default id”
    // (env: OPENCODE_MODEL / ARK_MODEL, default deepseek-v4-flash).
    let providers = ["opencode-go", "huoshan"];
    for name in providers {
        let model = match zen_ai::provider::resolve_provider(name) {
            Some(p) => zen_ai::types::Model::from_provider(p.as_ref()),
            None => continue,
        };
        let line = format!(
            "{:<14} {:<24} api={:?}",
            model.provider, model.id, model.api
        );
        if let Some(q) = &q {
            if line.contains(q.as_str()) {
                println!("{line}");
            }
        } else {
            println!("{line}");
        }
    }
    Ok(())
}
