use std::collections::HashSet;
use std::path::Path;
use zen_core::harness::system_prompt::{BuildSystemPromptOptions, ContextFile};

/// Input paths for the zen docs paragraph (main doc / additional docs / examples, three items),
/// Actual path resolution is provided by the caller via get_package_dir; this type only carries the paths.
pub struct ZenDocsPaths {
    /// Main doc path (README), used for the first line of the docs paragraph
    pub readme: String,
    /// Additional docs directory path (docs/), second line of the docs paragraph
    pub docs: String,
    /// examples directory path (examples/), third line of the docs paragraph
    pub examples: String,
}

/// Candidate file names for each directory (per REQ-APP-03: lowercase variants only)
const CONTEXT_FILE_CANDIDATES: [&str; 2] = ["AGENTS.md", "CLAUDE.md"];

/// Finds and reads the first existing candidate context file in a single directory.
/// Fields are pub: constructed directly as a struct literal, returning the domain type `Option<ContextFile>` (matching pi's return-null semantics).
fn load_context_file_from_dir(dir: &Path) -> Option<ContextFile> {
    for filename in CONTEXT_FILE_CANDIDATES {
        let file_path = dir.join(filename);
        match std::fs::read_to_string(&file_path) {
            Ok(content) => {
                return Some(ContextFile {
                    path: file_path.to_string_lossy().into_owned(),
                    content,
                })
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {} // not found → try the next candidate
            Err(e) => tracing::warn!(
                "could not read project context file {}: {}",
                file_path.display(),
                e
            ),
        }
    }
    None
}

/// Loads project context files (AGENTS.md/CLAUDE.md) from cwd's ancestor directories plus the global agentDir, returning an ordered ContextFile list.
/// Order: the global agentDir's AGENTS.md first, then each level's AGENTS.md from the ancestor nearest the root down to cwd (matching pi: global first,
/// ancestor results are unshifted so root-nearest come first); duplicate paths are deduped (HashSet).
/// Candidate file names are `AGENTS.md`/`CLAUDE.md` per the REQ-APP-03 note (pi also checks the uppercase variants `AGENTS.MD`/`CLAUDE.MD`; case-insensitive
/// on the filesystem, so they may be kept as needed at implementation time; the requirements doc takes precedence, default is lowercase only).
/// Files that fail to read are skipped with `tracing::warn!` (matching pi's warn-and-continue; this function returns no Result — it is a best-effort contract).
/// Edge cases: cwd/agent_dir should be normalized paths; an empty directory (no readable candidates) returns an empty Vec.
// Consumers: currently = `build_app_system_prompt` in this module; planned external consumers = T-AP-01-1 (direct reuse by AgentSession), T-AP-08-3
// (REQ-APP-03 explicitly names this function as a deliverable, hence pub; if no external consumers are confirmed later, it can be made private)
pub fn load_project_context_files(cwd: &Path, agent_dir: &Path) -> Vec<ContextFile> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut files: Vec<ContextFile> = Vec::new();

    // ① global agentDir first
    if let Some(file) = load_context_file_from_dir(agent_dir) {
        seen.insert(file.path.clone());
        files.push(file);
    }

    // ② collect in order from cwd upward (cwd → root), then reverse (matching pi's unshift order, O(n))
    let mut ancestors: Vec<ContextFile> = Vec::new();
    for dir in cwd.ancestors() {
        if let Some(file) = load_context_file_from_dir(dir) {
            if seen.insert(file.path.clone()) {
                ancestors.push(file);
            }
        }
    }
    ancestors.reverse();

    files.extend(ancestors);
    files
}

/// Application-layer system prompt assembler: load project context → inject `core.context_files` → append the "zen docs" paragraph in the default branch → call
/// `zen_core::harness::system_prompt::build_system_prompt`。
/// Behavior:
/// - Calls `load_project_context_files` with `core.cwd` and `agent_dir`, overwriting `core.context_files` with the result;
/// - When `core.custom_prompt` is None (default branch), merges `build_zen_docs_paragraph(docs)` with `core.append_system_prompt`
///   into `docs_para + "\n\n" + user_append` and writes it to `core.append_system_prompt`;
///   when `custom_prompt` is non-empty, the docs paragraph is **not** injected
/// - `core` is passed by value and modified in place; no Clone needed.
///
/// Returns: the full system prompt string (never empty). This function has no failure path (loading is best-effort, assembly is pure), so it returns no Result.
pub fn build_app_system_prompt(
    mut core: BuildSystemPromptOptions,
    agent_dir: &Path,
    docs: &ZenDocsPaths,
) -> String {
    // ① load project context and overwrite core.context_files (best-effort: read failures are skipped and do not block assembly)
    core.context_files = load_project_context_files(Path::new(&core.cwd), agent_dir);

    // ② default branch (custom_prompt is None): fold the docs paragraph into the append slot
    // People using pi/zen are likely to ask about pi/zen itself, so a pi/zen self-introduction document was added
    if core.custom_prompt.is_none() {
        let docs_para = build_zen_docs_paragraph(docs);
        let user_append = core.append_system_prompt.take().unwrap_or_default();
        core.append_system_prompt = Some(format!("{docs_para}\n\n{user_append}"));
    }
    // custom_prompt non-empty: do not inject the docs section; keep append as-is (the zen-core custom branch still concatenates it)

    // ③ Hand off to zen-core for full assembly (returns String, no failure path)
    zen_core::harness::system_prompt::build_system_prompt(&core)
}

/// Assemble the "pi/zen documentation" path section (default branch only; aligns with pi's "Pi documentation" section text; the zen version changes pi to zen in the identity).
/// Private function: called only internally by `build_app_system_prompt`; not marked pub.
fn build_zen_docs_paragraph(docs: &ZenDocsPaths) -> String {
    format!(
        "zen documentation (read only when the user asks about zen itself, its SDK, extensions, themes, skills, or TUI):\n\
- Main documentation: {}\n\
- Additional docs: {}\n\
- Examples: {} (extensions, custom tools, SDK)\n\
- When reading pi docs or examples, resolve docs/... under Additional docs and examples/... under Examples, not the current working directory\n\
- When asked about: extensions (docs/extensions.md, examples/extensions/), themes (docs/themes.md), skills (docs/skills.md), prompt templates (docs/prompt-templates.md), TUI components (docs/tui.md), keybindings (docs/keybindings.md), SDK integrations (docs/sdk.md), custom providers (docs/custom-provider.md), adding models (docs/models.md), pi packages (docs/packages.md)\n\
- When working on pi topics, read the docs and examples, and follow .md cross-references before implementing\n\
- Always read pi .md files completely and follow links to related docs (e.g., tui.md for TUI API details)",
        docs.readme, docs.docs, docs.examples,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::{Path, PathBuf};

    /// Temp-dir helper: creates a unique directory under the system temp dir and cleans it up on Drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("zen_tap03_{}_{}", name, std::process::id()));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
        fn write(&self, rel: &str, content: &str) -> PathBuf {
            let p = self.0.join(rel);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(&p, content).unwrap();
            p
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// Compress (path, content) into comparable tuples and keep only entries under root
    /// (Tolerates machine-level external context files such as /AGENTS.md and /tmp/AGENTS.md on CI/dev machines).
    fn entries_under(root: &Path, files: &[ContextFile]) -> Vec<(String, String)> {
        files
            .iter()
            .filter(|f| Path::new(&f.path).starts_with(root))
            .map(|f| (f.path.clone(), f.content.clone()))
            .collect()
    }

    fn docs() -> ZenDocsPaths {
        ZenDocsPaths {
            readme: "/docs/README.md".into(),
            docs: "/docs".into(),
            examples: "/examples".into(),
        }
    }

    // ── load_project_context_files ────────────────────────────────

    #[test]
    fn test_load_context_empty() {
        let root = TempDir::new("empty");
        let cwd = root.path().join("nested");
        let agent = root.path().join("agent");
        fs::create_dir_all(&cwd).unwrap();
        fs::create_dir_all(&agent).unwrap();

        let files = load_project_context_files(&cwd, &agent);
        assert!(entries_under(root.path(), &files).is_empty());
    }

    #[test]
    fn test_load_context_global_first_then_ancestors_rootmost_first() {
        let root = TempDir::new("order");
        let agent = root.path().join("agent");
        let cwd = root.path().join("proj").join("sub");
        fs::create_dir_all(&agent).unwrap();
        fs::create_dir_all(&cwd).unwrap();
        root.write("agent/AGENTS.md", "global");
        root.write("proj/AGENTS.md", "mid");
        root.write("proj/sub/AGENTS.md", "near");

        let files = load_project_context_files(&cwd, &agent);
        let got: Vec<String> = entries_under(root.path(), &files)
            .into_iter()
            .map(|(p, _)| p)
            .collect();
        // Expected order: [global, highest ancestor proj, nearest cwd itself proj/sub]
        assert_eq!(got.len(), 3, "got: {got:?}");
        assert!(got[0].ends_with("agent/AGENTS.md"), "got: {got:?}");
        assert!(got[1].ends_with("proj/AGENTS.md"), "got: {got:?}");
        assert!(got[2].ends_with("proj/sub/AGENTS.md"), "got: {got:?}");
    }

    #[test]
    fn test_load_context_dedup_global_inside_ancestry() {
        let root = TempDir::new("dedup");
        // agentDir has the same path as an ancestor directory: the same file must not be loaded twice
        let agent = root.path().join("proj");
        let cwd = root.path().join("proj").join("sub");
        fs::create_dir_all(&cwd).unwrap();
        root.write("proj/AGENTS.md", "X");

        let files = load_project_context_files(&cwd, &agent);
        let got: Vec<String> = entries_under(root.path(), &files)
            .into_iter()
            .map(|(p, _)| p)
            .collect();
        assert_eq!(got.len(), 1, "got: {got:?}");
        assert!(got[0].ends_with("proj/AGENTS.md"));
    }

    #[test]
    fn test_load_context_claude_md_fallback_and_precedence() {
        let root = TempDir::new("claude");
        // cwd directory has only CLAUDE.md (fallback when AGENTS.md is absent); agent directory has both (AGENTS.md takes precedence)
        let cwd = root.path().join("onlyclaude");
        let agent = root.path().join("both");
        fs::create_dir_all(&cwd).unwrap();
        fs::create_dir_all(&agent).unwrap();
        root.write("onlyclaude/CLAUDE.md", "c");
        root.write("both/AGENTS.md", "a");
        root.write("both/CLAUDE.md", "c");

        let files = load_project_context_files(&cwd, &agent);
        let got = entries_under(root.path(), &files);
        assert_eq!(got.len(), 2, "got: {got:?}");
        assert!(got[0].0.ends_with("both/AGENTS.md"), "got: {got:?}");
        assert_eq!(got[0].1, "a");
        assert!(got[1].0.ends_with("onlyclaude/CLAUDE.md"), "got: {got:?}");
        assert_eq!(got[1].1, "c");
    }

    // ── build_app_system_prompt ───────────────────────────────────

    fn base_options(cwd: String) -> BuildSystemPromptOptions {
        BuildSystemPromptOptions {
            custom_prompt: None,
            selected_tools: None,
            tool_snippets: BTreeMap::new(),
            prompt_guidelines: Vec::new(),
            append_system_prompt: None,
            cwd,
            skills: Vec::new(),
            context_files: Vec::new(),
        }
    }

    #[test]
    fn test_build_app_prompt_default_branch_injects_context_and_docs() {
        let root = TempDir::new("app_default");
        let cwd = root.path().join("proj");
        fs::create_dir_all(&cwd).unwrap();
        root.write("proj/AGENTS.md", "project rules");
        let mut opts = base_options(cwd.to_string_lossy().into_owned());
        opts.append_system_prompt = Some("USER APPEND".into());

        let out = build_app_system_prompt(opts, root.path(), &docs());

        assert!(out.contains("zen documentation (read only when the user asks about zen itself"));
        assert!(out.contains("USER APPEND"));
        assert!(out.contains("<project_instructions path="));
        assert!(out.contains("project rules"));
        // docs section comes before user append (pi: docs → appendSection → project_context order)
        assert!(out.find("zen documentation").unwrap() < out.find("USER APPEND").unwrap());
        assert!(out.contains("Current working directory:"));
    }

    #[test]
    fn test_build_app_prompt_custom_branch_omits_docs_keeps_append() {
        let root = TempDir::new("app_custom");
        let mut opts = base_options(root.path().to_string_lossy().into_owned());
        opts.custom_prompt = Some("CUSTOM INSTRUCTIONS".into());
        opts.append_system_prompt = Some("APPEND".into());

        let out = build_app_system_prompt(opts, root.path(), &docs());

        assert!(out.contains("CUSTOM INSTRUCTIONS"));
        assert!(!out.contains("zen documentation"));
        assert!(out.contains("APPEND"));
        assert!(out.contains("Current working directory:"));
    }

    // ── build_zen_docs_paragraph ──────────────────────────────────

    #[test]
    fn test_build_zen_docs_paragraph_contains_paths() {
        let para = build_zen_docs_paragraph(&docs());
        assert!(para.contains("/docs/README.md"));
        assert!(para.contains("/docs"));
        assert!(para.contains("/examples"));
        // Each bullet is on its own line (aligned with pi, no hanging continuation lines)
        assert!(para.contains(
            "- Main documentation: /docs/README.md\n- Additional docs: /docs\n- Examples: /examples"
        ));
    }
}
