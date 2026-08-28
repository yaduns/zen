//! Flag example: `--tools` / `--exclude-tools` / `--no-tools` / `--no-builtin-tools` — enable/disable tools.
//!
//! Usage (aligned with real CLI flags):
//!   cargo run -p zen-app --example example_tools -- --tools read,bash "list files"
//!   cargo run -p zen-app --example example_tools -- --exclude-tools bash "list files"
//!   cargo run -p zen-app --example example_tools -- --no-tools "what is 1+1"
//!   cargo run -p zen-app --example example_tools -- --no-builtin-tools "what is 1+1"

#[path = "_support/common.rs"]
mod common;

use clap::Parser;
use common::*;
use zen_ai::types::ThinkingLevel;

/// Simulated CLI tool-selection flags (only for this example; does not modify existing code).
#[derive(Parser)]
#[command(name = "example_tools", about = "Demonstrate tool subset selection")]
struct Args {
    /// Whitelist of enabled tools (comma-separated)
    #[arg(long, value_delimiter = ',')]
    tools: Vec<String>,
    /// Disabled tool blacklist (comma-separated)
    #[arg(long, value_delimiter = ',')]
    exclude_tools: Vec<String>,
    /// Do not use any tools
    #[arg(long)]
    no_tools: bool,
    /// Do not use built-in tools (read only; aligns with --no-builtin-tools semantics)
    #[arg(long)]
    no_builtin_tools: bool,
    /// Prompt to send to the LLM (positional argument)
    prompt: Vec<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let prompt = if args.prompt.is_empty() {
        "List files in the current directory".to_string()
    } else {
        args.prompt.join(" ")
    };

    // Select the tool subset based on the flags
    let tools = if args.no_tools {
        vec![]
    } else if args.no_builtin_tools {
        // read only (aligns with --no-builtin-tools semantics)
        tool_registry_tools("readonly")
    } else if !args.tools.is_empty() {
        // --tools whitelist filtering
        let all = tool_registry_tools("all");
        all.into_iter()
            .filter(|t| args.tools.iter().any(|name| t.name() == name.as_str()))
            .collect()
    } else if !args.exclude_tools.is_empty() {
        // --exclude-tools blacklist filtering
        let all = tool_registry_tools("all");
        all.into_iter()
            .filter(|t| {
                !args
                    .exclude_tools
                    .iter()
                    .any(|name| t.name() == name.as_str())
            })
            .collect()
    } else {
        tool_registry_tools("all")
    };

    eprintln!(
        "[example] Tool count: {} ({})",
        tools.len(),
        if tools.is_empty() {
            "No tools".into()
        } else {
            tools
                .iter()
                .map(|t| t.name())
                .collect::<Vec<_>>()
                .join(", ")
        }
    );

    let mut session = make_session(
        default_model(),
        "You are a helpful coding assistant. Use tools to answer.".into(),
        tools,
        ThinkingLevel::Off,
    );
    run_and_print(&mut session, &prompt).await
}
