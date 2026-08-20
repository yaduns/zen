//! Flag example: `--skills <dir>` — load a skills directory and inject into the system prompt (`load_skills` + `build_system_prompt`).
//! Note: `load_skills` and `build_system_prompt` are existing zen-core public APIs (T-AG-05-2);
//! The app layer only lacks the glue to wire `--skills` into `create_initial_runtime_from_cli` (T-AP-03-3).
//! Run:
//!   cargo run -p zen-app --example example_skills -- <directory containing SKILL.md> "please answer per the skill instructions"

#[path = "_support/common.rs"]
mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use common::*;
use zen_ai::types::ThinkingLevel;
use zen_core::harness::system_prompt::{build_system_prompt, BuildSystemPromptOptions};
use zen_core::skills::load::load_skills;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let skill_dir = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: example_skills <skill-dir> [prompt]"))?;
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "Please answer according to the skills you have".into());

    // --skills semantics: load_skills reads SKILL.md → inject into BuildSystemPromptOptions.skills
    let loaded = load_skills(&[PathBuf::from(&skill_dir)]).await;
    let skill_count = loaded.skills.len();
    eprintln!(
        "[example] Loaded {} skills, {} diagnostics",
        skill_count,
        loaded.diagnostics.len()
    );

    let system_prompt = build_system_prompt(&BuildSystemPromptOptions {
        custom_prompt: None,
        selected_tools: None,
        tool_snippets: BTreeMap::new(),
        prompt_guidelines: vec![],
        append_system_prompt: None,
        cwd: cwd_str(),
        skills: loaded.skills, // moved by value (Skill does not implement Clone)
        context_files: vec![],
    });

    let mut session = make_session(
        common::default_model(),
        system_prompt,
        vec![],
        ThinkingLevel::Off,
    );
    run_and_print(&mut session, &prompt).await
}
