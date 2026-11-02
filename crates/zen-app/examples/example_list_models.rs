//! Flag example: `--list-models` — list available models (this example demonstrates printing the provider→model mapping).
//! Run: `cargo run -p zen-app --example example_list_models -- [search string]`
//!
//! Data source is `Catalog::current().models()` — the same enumeration `/model` and `/scoped-models` use,
//! so providers marked `"listed": false` (e.g. the legacy `huoshan` record) do **not** appear here.
//! They remain resolvable by id (`--provider huoshan`), which this listing intentionally does not show.

fn main() -> anyhow::Result<()> {
    let q = std::env::args().nth(1);

    for model in zen_ai::catalog::Catalog::current().models() {
        let line = format!(
            "{:<14} {:<24} api={:?}",
            model.provider, model.id, model.api
        );
        match &q {
            Some(q) if !line.contains(q.as_str()) => continue,
            _ => println!("{line}"),
        }
    }
    Ok(())
}
