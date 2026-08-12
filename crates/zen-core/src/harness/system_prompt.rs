use crate::skills::format::format_skills_for_system_prompt;
use crate::skills::skill::Skill;
use std::collections::BTreeMap;
use zen_ai::tool::Tool;

// All runtime inputs for a single build_system_prompt call. Every field is an assembly input with a single-call lifetime; no serializable business data.
struct BuildSystemPromptOptions {
    /// When non-empty, replaces the default identity + tools + guidelines template
    custom_prompt: Option<String>,
    /// Tool names that should be included; defaults to ["read","bash","edit","write"] when None
    /// Tools in selected_tools appear in the json tools array
    /// Tools with snippets are listed in the system prompt's available tools
    /// The selected_tools set is a superset of the set of tools with snippets
    selected_tools: Option<Vec<String>>,
    /// Tool name -> one-line description; only tools with a snippet appear in Available tools; upstream is collect_tool_prompt_metadata aggregating Tool::prompt_snippet()
    tool_snippets: BTreeMap<String, String>,
    /// Additional guideline entries appended after the default guidelines (deduplicated, trimmed); upstream is collect_tool_prompt_metadata aggregating Tool::prompt_guidelines()
    prompt_guidelines: Vec<String>,
    /// Text appended to the end of the system prompt
    append_system_prompt: Option<String>,
    /// Working directory
    cwd: String,
    /// Injected skills
    skills: Vec<Skill>,
    /// Project context files (AGENTS.md/CLAUDE.md); loaded by the caller via load_project_context_files then injected, assembled into <project_context> / <project_instructions> blocks
    context_files: Vec<ContextFile>,
}

// Path and content of a project context file (e.g. AGENTS.md), loaded by load_project_context_files from cwd up through ancestor directories, injected into BuildSystemPromptOptions.context_files, and assembled by build_system_prompt into <project_instructions>.
struct ContextFile {
    // File path (relative to cwd or absolute)
    path: String,
    // Full file content
    content: String,
}

// Aggregated output of collect_tool_prompt_metadata, produced once and passed by the caller into BuildSystemPromptOptions.
struct ToolPromptMetadata {
    // Tool name -> one-line summary; includes only tools that provided a snippet
    tool_snippets: BTreeMap<String, String>,
    // Flat list of per-tool usage guidelines (may contain duplicates; build_system_prompt deduplicates)
    prompt_guidelines: Vec<String>,
}

/// Assemble the system prompt (full zen-core assembler):
/// Identity text + Available tools (tools with snippets, in selected_tools order) + Guidelines + skills block + project context + extra content + cwd.
///
/// Behavioral details (ref: pi-306 buildSystemPrompt):
/// - When `custom_prompt` is non-empty, take the replacement branch: use custom_prompt directly, append append_system_prompt,
///   Project context, skill blocks, append `\nCurrent working directory: <cwd>` at the end; no default identity/tool/guideline sections are generated.
/// - Default branch: identity text + tool list (selected tools present only in `tool_snippets`; `(none)` if none) + guidelines
///   (default "Be concise…" "Show file paths…" + prompt_guidelines, deduplicated) + append_system_prompt
///   + project context (`context_files` joined into `<project_context>` / `<project_instructions>`) + skills block + cwd.
/// - Formatting of the skills block is provided by `format_skills_for_system_prompt` from **T-AG-05-2**; this function calls it directly.
/// - **Loading** of AGENTS.md is done by the caller (T-AP-03-1)'s `load_project_context_files`; this function only assembles.
/// - The "pi/zen docs" path paragraph (config-layer path) is still appended by the zen-app upper layer (T-AP-03-1), not in this function.
///
/// Returns: the fully assembled system prompt string (never empty; contains at least identity/replacement text and the cwd line).
/// When `context_files` is non-empty, it is joined into `<project_context>`; otherwise that block is skipped.
pub fn build_system_prompt(options: &BuildSystemPromptOptions) -> String {
    let prompt_cwd = options.cwd.replace('\\', "/");

    let append_section = options
        .append_system_prompt
        .as_ref()
        .map(|s| format!("\n\n{}", s))
        .unwrap_or_default();

    let has_read = options
        .selected_tools
        .as_ref()
        .map_or(true, |tools| tools.iter().any(|t| t == "read"));

    // Project context block (<project_context> / <project_instructions>)
    let project_context = format_project_context(&options.context_files);

    // Replacement branch: custom_prompt non-empty
    if let Some(custom_prompt) = &options.custom_prompt {
        let mut prompt = custom_prompt.clone();
        prompt.push_str(&append_section);
        prompt.push_str(&project_context);
        if has_read && !options.skills.is_empty() {
            prompt.push_str(&format_skills_for_system_prompt(&options.skills));
        }
        prompt.push_str(&format!("\nCurrent working directory: {}", prompt_cwd));
        return prompt;
    }

    // Default branch
    let identity = "You are an expert coding assistant operating inside zen, a coding agent harness. You help users by reading
 files, executing commands, editing code, and writing new files.";

    let tools = options
        .selected_tools
        .clone()
        .unwrap_or_else(|| vec!["read".into(), "bash".into(), "edit".into(), "write".into()]);
    let visible_tools: Vec<&str> = tools
        .iter()
        .filter(|name| options.tool_snippets.contains_key(*name))
        .map(|s| s.as_str())
        .collect();
    let tools_list = if visible_tools.is_empty() {
        "(none)".to_string()
    } else {
        visible_tools
            .iter()
            .map(|name| format!("- {}: {}", name, options.tool_snippets[*name]))
            .collect::<Vec<_>>()
            .join("\n")
    };

    // Guidelines: deduplicated
    let mut guidelines = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut add_guideline = |g: &str| {
        if seen.insert(g.to_string()) {
            guidelines.push(g.to_string());
        }
    };

    let has_bash = tools.iter().any(|t| t == "bash");
    let has_grep = tools.iter().any(|t| t == "grep");
    let has_find = tools.iter().any(|t| t == "find");
    let has_ls = tools.iter().any(|t| t == "ls");

    if has_bash && !has_grep && !has_find && !has_ls {
        add_guideline("Use bash for file operations like ls, rg, find");
    }

    for g in &options.prompt_guidelines {
        let normalized = g.trim();
        if !normalized.is_empty() {
            add_guideline(normalized);
        }
    }

    add_guideline("Be concise in your responses");
    add_guideline("Show file paths clearly when working with files");

    let guidelines_text = guidelines
        .iter()
        .map(|g| format!("- {}", g))
        .collect::<Vec<_>>()
        .join("\n");

    let mut prompt = format!(
           "{}\n\nAvailable tools:\n{}\n\nIn addition to the tools above, you may have access to other custom tools depending on the
 project.\n\nGuidelines:\n{}",
           identity, tools_list, guidelines_text
       );

    prompt.push_str(&append_section);
    prompt.push_str(&project_context);
    if has_read && !options.skills.is_empty() {
        prompt.push_str(&format_skills_for_system_prompt(&options.skills));
    }
    prompt.push_str(&format!("\nCurrent working directory: {}", prompt_cwd));
    prompt
}

/// Assembles context_files into a `<project_context>` block.
/// Returns an empty string for empty input.
fn format_project_context(context_files: &[ContextFile]) -> String {
    if context_files.is_empty() {
        return String::new();
    }
    let mut block = String::from("\n\n<project_context>\n\n");
    block.push_str("Project-specific instructions and guidelines:\n\n");
    for file in context_files {
        block.push_str(&format!(
            "<project_instructions path=\"{}\">\n{}\n</project_instructions>\n\n",
            file.path, file.content
        ));
    }
    block.push_str("</project_context>\n");
    block
}

/// Aggregates system prompt metadata from the active tool set: one-line tool summaries + usage guidelines.
///
/// Corresponds to the aggregation logic of pi `AgentSession._rebuildSystemPrompt`: iterates over `tools`,
/// collecting each tool's `Tool::prompt_snippet()` (skipped if `None`) and `Tool::prompt_guidelines()`.
/// The resulting `ToolPromptMetadata` is filled into `BuildSystemPromptOptions.tool_snippets` / `prompt_guidelines` by the caller.
///
/// # Parameters
/// - `tools`: slice of active tools (`&dyn Tool`).
///
/// # Returns
/// The aggregated `ToolPromptMetadata`. Empty input returns an empty map/list.
pub fn collect_tool_prompt_metadata(tools: &[&dyn Tool]) -> ToolPromptMetadata {
    let mut tool_snippets = BTreeMap::new();
    let mut prompt_guidelines = Vec::new();

    for tool in tools {
        if let Some(snippet) = tool.prompt_snippet() {
            tool_snippets.insert(tool.name().to_string(), snippet.to_string());
        }
        prompt_guidelines.extend(tool.prompt_guidelines().iter().map(|g| g.to_string()));
    }

    ToolPromptMetadata {
        tool_snippets,
        prompt_guidelines,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::json;
    use zen_ai::tool::{ToolContext, ToolError, ToolExecutionResult, ToolUpdateCallback};

    // ─── MockTool (test double for collect_tool_prompt_metadata) ──────────
    #[derive(Debug)]
    struct MockTool {
        name: String,
        snippet: Option<String>,
        guidelines: &'static [&'static str],
        params: serde_json::Value,
    }

    impl MockTool {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                snippet: None,
                guidelines: &[],
                params: json!({ "type": "object" }),
            }
        }
        fn with_snippet(mut self, snippet: &str) -> Self {
            self.snippet = Some(snippet.to_string());
            self
        }
        fn with_guidelines(mut self, guidelines: &'static [&'static str]) -> Self {
            self.guidelines = guidelines;
            self
        }
    }

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "a test tool"
        }
        fn parameters(&self) -> &serde_json::Value {
            &self.params
        }
        fn label(&self) -> &str {
            &self.name
        }
        fn prompt_snippet(&self) -> Option<&str> {
            self.snippet.as_deref()
        }
        fn prompt_guidelines(&self) -> &'static [&'static str] {
            self.guidelines
        }
        async fn execute(
            &self,
            _tool_call_id: &str,
            _params: &serde_json::Value,
            _cancel: &tokio_util::sync::CancellationToken,
            _on_update: ToolUpdateCallback,
        ) -> Result<ToolExecutionResult, ToolError> {
            unimplemented!("not used in these tests")
        }
    }

    // ─── helpers ────────────────────────────────────────────────
    // Skill fields constructed per design (name/description/content/file_path/disable_model_invocation) assumption; adjust if inconsistent with the actual definition.
    fn make_skill(name: &str, disable: bool) -> Skill {
        Skill {
            name: name.to_string(),
            description: format!("does {}", name),
            content: "skill content".to_string(),
            file_path: format!("/skills/{}/SKILL.md", name),
            disable_model_invocation: disable,
        }
    }

    fn make_context_file(path: &str, content: &str) -> ContextFile {
        ContextFile {
            path: path.to_string(),
            content: content.to_string(),
        }
    }

    fn default_options(cwd: &str) -> BuildSystemPromptOptions {
        BuildSystemPromptOptions {
            custom_prompt: None,
            selected_tools: None,
            tool_snippets: BTreeMap::new(),
            prompt_guidelines: Vec::new(),
            append_system_prompt: None,
            cwd: cwd.to_string(),
            skills: Vec::new(),
            context_files: Vec::new(),
        }
    }

    // ─── format_project_context ─────────────────────────────────
    #[test]
    fn test_format_project_context_empty() {
        assert_eq!(format_project_context(&[]), "");
    }

    #[test]
    fn test_format_project_context_single_file() {
        let block = format_project_context(&[make_context_file("AGENTS.md", "rules here")]);
        assert!(block.contains("<project_context>"));
        assert!(block.contains("<project_instructions path=\"AGENTS.md\">"));
        assert!(block.contains("rules here"));
        assert!(block.contains("</project_instructions>"));
        assert!(block.contains("</project_context>"));
    }

    #[test]
    fn test_format_project_context_multiple_files() {
        let files = vec![
            make_context_file("global.md", "global rules"),
            make_context_file("project.md", "project rules"),
        ];
        let block = format_project_context(&files);
        assert!(block.contains("path=\"global.md\""));
        assert!(block.contains("global rules"));
        assert!(block.contains("path=\"project.md\""));
        assert!(block.contains("project rules"));
    }

    // ─── build_system_prompt: default branch ───────────────────────
    #[test]
    fn test_build_system_prompt_default_identity_and_cwd() {
        let out = build_system_prompt(&default_options("/proj"));
        assert!(out.contains("You are an expert coding assistant operating inside zen"));
        assert!(out.contains("Available tools:"));
        assert!(out.contains("Guidelines:"));
        assert!(out.contains("Current working directory: /proj"));
    }

    #[test]
    fn test_build_system_prompt_no_snippets_shows_none() {
        let out = build_system_prompt(&default_options("/p"));
        assert!(out.contains("Available tools:\n(none)"));
    }

    #[test]
    fn test_build_system_prompt_visible_tools_only_with_snippet() {
        let mut opts = default_options("/p");
        opts.selected_tools = Some(vec!["read".into(), "bash".into()]);
        opts.tool_snippets
            .insert("read".into(), "Read file contents".into());
        // bash has no snippet → absent; read has one → present
        let out = build_system_prompt(&opts);
        assert!(out.contains("- read: Read file contents"));
        assert!(!out.contains("- bash:"));
    }

    #[test]
    fn test_build_system_prompt_default_tools_with_snippets() {
        let mut opts = default_options("/p");
        for (name, snip) in [
            ("read", "Read file contents"),
            ("bash", "Execute bash commands"),
            ("edit", "Make precise file edits"),
            ("write", "Create or overwrite files"),
        ] {
            opts.tool_snippets.insert(name.into(), snip.into());
        }
        let out = build_system_prompt(&opts);
        assert!(out.contains("- read: Read file contents"));
        assert!(out.contains("- write: Create or overwrite files"));
        assert!(!out.contains("(none)"));
    }

    #[test]
    fn test_build_system_prompt_cwd_backslash_normalized() {
        let out = build_system_prompt(&default_options("C:\\proj\\dir"));
        assert!(out.contains("Current working directory: C:/proj/dir"));
    }

    #[test]
    fn test_build_system_prompt_guidelines_dedup_and_trim() {
        let mut opts = default_options("/p");
        opts.prompt_guidelines = vec![
            "  Be concise in your responses  ".to_string(), // duplicates the default, trim + dedupe
            "   ".to_string(),                              // blank, skip
            "Always run tests".to_string(),
        ];
        let out = build_system_prompt(&opts);
        assert_eq!(out.matches("Be concise in your responses").count(), 1);
        assert!(out.contains("- Always run tests"));
    }

    // ─── build_system_prompt: custom_prompt branch ────────────────
    #[test]
    fn test_build_system_prompt_custom_prompt_replaces_default() {
        let mut opts = default_options("/p");
        opts.custom_prompt = Some("CUSTOM INSTRUCTIONS".to_string());
        let out = build_system_prompt(&opts);
        assert!(out.contains("CUSTOM INSTRUCTIONS"));
        assert!(out.contains("Current working directory: /p"));
        assert!(!out.contains("Available tools:"));
        assert!(!out.contains("You are an expert coding assistant"));
    }

    #[test]
    fn test_build_system_prompt_custom_prompt_with_append_and_context_and_skills() {
        let mut opts = default_options("/p");
        opts.custom_prompt = Some("CUSTOM".to_string());
        opts.append_system_prompt = Some("APPENDED".to_string());
        opts.context_files = vec![make_context_file("AGENTS.md", "rules")];
        opts.skills = vec![make_skill("foo", false)];
        let out = build_system_prompt(&opts);
        assert!(out.contains("\n\nAPPENDED"));
        assert!(out.contains("<project_instructions path=\"AGENTS.md\">"));
        assert!(out.contains("<available_skills>"));
        assert!(out.contains("foo"));
    }

    // ─── build_system_prompt: additional blocks ───────────────────
    #[test]
    fn test_build_system_prompt_appends_system_prompt() {
        let mut opts = default_options("/p");
        opts.append_system_prompt = Some("EXTRA RULES".to_string());
        let out = build_system_prompt(&opts);
        assert!(out.contains("\n\nEXTRA RULES"));
    }

    #[test]
    fn test_build_system_prompt_project_context_omitted_when_empty() {
        let out = build_system_prompt(&default_options("/p"));
        assert!(!out.contains("<project_context>"));
    }

    #[test]
    fn test_build_system_prompt_skills_block_in_default_branch() {
        let mut opts = default_options("/p");
        opts.skills = vec![make_skill("foo", false)];
        let out = build_system_prompt(&opts);
        assert!(out.contains("<available_skills>"));
        assert!(out.contains("foo"));
    }

    #[test]
    fn test_build_system_prompt_disabled_skill_excluded() {
        let mut opts = default_options("/p");
        opts.skills = vec![make_skill("visible", false), make_skill("hidden", true)];
        let out = build_system_prompt(&opts);
        assert!(out.contains("visible"));
        assert!(!out.contains("hidden"));
    }

    // ─── collect_tool_prompt_metadata ───────────────────────────
    fn collect(tools: &[MockTool]) -> ToolPromptMetadata {
        let refs: Vec<&dyn Tool> = tools.iter().map(|t| t as &dyn Tool).collect();
        collect_tool_prompt_metadata(&refs)
    }

    #[test]
    fn test_collect_tool_prompt_metadata_empty() {
        let meta = collect(&[]);
        assert!(meta.tool_snippets.is_empty());
        assert!(meta.prompt_guidelines.is_empty());
    }

    #[test]
    fn test_collect_tool_prompt_metadata_snippet_only() {
        let meta = collect(&[MockTool::new("read").with_snippet("Read file contents")]);
        assert_eq!(
            meta.tool_snippets.get("read").map(|s| s.as_str()),
            Some("Read file contents")
        );
        assert!(meta.prompt_guidelines.is_empty());
    }

    #[test]
    fn test_collect_tool_prompt_metadata_no_snippet_skipped() {
        let meta = collect(&[MockTool::new("bash")]);
        assert!(!meta.tool_snippets.contains_key("bash"));
    }

    #[test]
    fn test_collect_tool_prompt_metadata_guidelines_collected() {
        let meta = collect(&[MockTool::new("read").with_guidelines(&["Use read instead of cat"])]);
        assert_eq!(meta.prompt_guidelines, vec!["Use read instead of cat"]);
    }

    #[test]
    fn test_collect_tool_prompt_metadata_multiple_tools_order() {
        let meta = collect(&[
            MockTool::new("read")
                .with_snippet("Read")
                .with_guidelines(&["g1", "g2"]),
            MockTool::new("bash").with_snippet("Bash"),
        ]);
        assert_eq!(meta.tool_snippets.len(), 2);
        assert_eq!(
            meta.tool_snippets.get("read").map(|s| s.as_str()),
            Some("Read")
        );
        assert_eq!(
            meta.tool_snippets.get("bash").map(|s| s.as_str()),
            Some("Bash")
        );
        assert_eq!(meta.prompt_guidelines, vec!["g1", "g2"]);
    }
}
