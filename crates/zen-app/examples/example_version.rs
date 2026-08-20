//! Flag example: `--version` — print the version and exit (corresponds to the main fast path).
//! Run: `cargo run -p zen-app --example example_version --`

fn main() {
    println!("{}", zen_app::config::VERSION);
}
