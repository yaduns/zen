//! Flag example: `--export <path>` — export session as HTML.
//! Note: `--export` currently bails in main (`--export is not supported yet (Phase 2)`); Phase 1 does not implement HTML export.
//! This example reproduces main's fast-path behavior: when --export is detected, report that it is not yet supported.
//! Run: `cargo run -p zen-app --example example_export -- out.html`

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "out.html".into());
    eprintln!("--export {path} is not supported yet (Phase 2)");
    std::process::exit(1); // align with main's bail
}
