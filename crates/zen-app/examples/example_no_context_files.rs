//! Flag example: `--no-context-files` — skip discovery and injection of AGENTS.md/CLAUDE.md.
//! Use `build_system_prompt` + `load_project_context_files` (both existing public APIs) to demonstrate whether loaded or not,
//! whether the <project_context> block appears in the system prompt.
//! Run: `cargo run -p zen-app --example example_no_context_files --`

#[path = "_support/common.rs"]
mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use common::*;
use zen_app::config::get_agent_dir;
use zen_app::system_prompt::load_project_context_files;
use zen_core::harness::system_prompt::{build_system_prompt, BuildSystemPromptOptions};

fn main() -> anyhow::Result<()> {
    common::init_catalog();
    let agent_dir = get_agent_dir().unwrap_or_else(|_| PathBuf::from(""));
    let cwd = PathBuf::from(cwd_str());

    let with = build_system_prompt(&BuildSystemPromptOptions {
        custom_prompt: None,
        selected_tools: None,
        tool_snippets: BTreeMap::new(),
        prompt_guidelines: vec![],
        append_system_prompt: None,
        cwd: cwd_str(),
        skills: vec![],
        context_files: load_project_context_files(&cwd, &agent_dir),
    });

    let without = build_system_prompt(&BuildSystemPromptOptions {
        custom_prompt: None,
        selected_tools: None,
        tool_snippets: BTreeMap::new(),
        prompt_guidelines: vec![],
        append_system_prompt: None,
        cwd: cwd_str(),
        skills: vec![],
        context_files: vec![], // --no-context-files: do not load any project context
    });

    let has_ctx = with.contains("<project_context>");
    let has_ctx_off = without.contains("<project_context>");
    println!("[example] Project context loaded by default: {has_ctx}; after --no-context-files: {has_ctx_off}");
    Ok(())
}
