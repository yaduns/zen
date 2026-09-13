//! Resource loading assembly layer: unified loading entry point for skills/prompt templates/themes/context files/system prompts.
//!
//! REQ-APP-05 (pi-318): resource loader.
//! Application-layer assembly entry-point shape: pi `DefaultResourceLoader` (stateful class + reload) is reduced in v1 to
//! one-shot assembly function `load_resources` — extension system Drop, settings hot reload does not reload resources, interactive
//! theme hot reload uses the ThemeController watcher's independent channel; stateful cache and reload have no consumers (T-AP-05-3
//! Implementation note 5.2 resolution table). Skills/templates/context delegate to existing loaders respectively (T-AP-03-3/03-4/03-1),
//! Theme assembly is in the `themes` submodule; SYSTEM.md/APPEND_SYSTEM.md discovery is inlined in this layer.

use std::path::{Path, PathBuf};

use zen_core::harness::system_prompt::ContextFile;
use zen_core::prompt_templates::PromptTemplateDiagnostic;

use crate::config::CONFIG_DIR_NAME;
use crate::prompt_templates::LoadedPromptTemplate;
use crate::skills::SkillLoadDiagnostic;
use crate::system_prompt::load_project_context_files;
use crate::{prompt_templates, skills};

pub mod themes;

pub use themes::{LoadedTheme, ThemeLoadDiagnostic};

/// Assembly input for `load_resources` (mirrors `LoadSkillsOptions`/`LoadPromptTemplatesOptions`).
/// Pure input parameters, no behavior.
pub struct LoadResourcesOptions {
    /// Session working directory (base for project resource directory `<cwd>/.zen/*`).
    pub cwd: PathBuf,
    /// Global agent config directory (base for `<agentDir>/*`; output of T-AP-02-1 `get_agent_dir`).
    pub agent_dir: PathBuf,
    /// Explicit skill paths (files or directories); CLI `--skill` + settings.skills are merged by the caller and passed in.
    pub skill_paths: Vec<PathBuf>,
    /// Explicit template paths (files or directories); CLI `--prompt-template` + settings.prompts are merged by the caller and passed in.
    pub prompt_paths: Vec<PathBuf>,
    /// Explicit theme paths (directories or `.json` files); CLI `--theme` + settings.themes are merged by the caller and passed in.
    pub theme_paths: Vec<PathBuf>,
    /// When true, skip default skill directories (explicit paths are still kept; pi `noSkills ? mergePaths(cli, additional)` semantics).
    pub no_skills: bool,
    /// When true, skip the entire template loading path (including default directories and explicit paths; aligns with the pi `noPromptTemplates` branch).
    pub no_prompt_templates: bool,
    /// When true, skip default theme directories (explicit paths are still kept; pi `noThemes` semantics).
    pub no_themes: bool,
    /// When true, do not load project context files (AGENTS.md/CLAUDE.md).
    pub no_context_files: bool,
    /// Value provided by CLI `--system-prompt`: non-empty takes precedence over SYSTEM.md discovery; an explicit empty string short-circuits (no discovery and no injection,
    /// aligns with pi `!input`).
    pub system_prompt: Option<String>,
    /// Repeatable values for CLI `--append-system-prompt`: non-empty entries take precedence over APPEND_SYSTEM.md discovery; empty entries are removed.
    pub append_system_prompt: Vec<String>,
}

/// One-shot output aggregate for `load_resources` — session assembly (services.rs) takes all resources from the same result
/// (aligns with pi getSkills/getPrompts/getThemes/getAgentsFiles/getSystemPrompt/getAppendSystemPrompt
/// six-facet summary), "warnings do not interrupt" semantics (each group may be partially empty; diagnostics and artifacts may both be non-empty).
pub struct LoadResourcesResult {
    /// Skills (delegates to T-AP-03-3 `load_skills`; already deduplicated + source preserved).
    pub skills: Vec<crate::skills::LoadedSkill>,
    /// Templates (after delegating to T-AP-03-4 `load_prompt_templates`, **deduplicate** by first occurrence of name — aligns with pi
    /// `dedupePrompts`; the `path` field lands incrementally with T-AP-03-4).
    pub prompts: Vec<LoadedPromptTemplate>,
    /// Themes (delegates to `themes::load_themes`; name deduplication + collision diagnostics; current consumer = T-AP-11-1, reserved).
    pub themes: Vec<LoadedTheme>,
    /// Project context files (delegates to T-AP-03-1 `load_project_context_files`; empty when `no_context_files`).
    pub context_files: Vec<ContextFile>,
    /// Resolved system prompt (non-empty CLI value or SYSTEM.md file content; explicit empty string / neither present → None).
    pub system_prompt: Option<String>,
    /// Resolved append prompt (each non-empty CLI entry or the single APPEND_SYSTEM.md entry; neither → empty).
    pub append_system_prompt: Vec<String>,
    /// Unified aggregation of five-way diagnostics (skills/templates/template collisions/themes).
    pub diagnostics: Vec<ResourceLoadDiagnostic>,
}

/// Cross-resource unified diagnostic tag — resolved via pi `ResourceDiagnostic.type` string discriminant: `"warning"`/`"collision"`
/// Forwards existing diagnostic types by resource (T-AP-03-3/03-4 outputs + Theme/PromptCollision added by this task),
/// Consumers (services.rs logs, T-AP-11-1 displays) iterate over a single list.
#[derive(Debug)]
pub enum ResourceLoadDiagnostic {
    /// Skill loading warnings/collisions (forwarded from T-AP-03-3).
    Skill(SkillLoadDiagnostic),
    /// Template loading warnings (forwarded from T-AP-03-4 → zen-core).
    Prompt(PromptTemplateDiagnostic),
    /// Same-name template collisions (derived in this task; aligns with pi `dedupePrompts` diagnostics).
    PromptCollision(PromptCollision),
    /// Theme loading warnings/collisions (this task's `resources::themes`).
    Theme(ThemeLoadDiagnostic),
}

/// Same-name template collision diagnostic payload (mirrors `SkillCollision`, the prompt form of pi `ResourceCollision`,
/// resolved by `resourceType: "prompt"`); winner/loser paths are taken directly from `LoadedPromptTemplate.path`
/// (T-AP-03-4 option B incremental field), with no reverse inference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptCollision {
    /// Template name of the collision.
    pub name: String,
    /// First-occurrence path (`LoadedPromptTemplate.path`).
    pub winner_path: String,
    /// Later-occurrence path (discarded).
    pub loser_path: String,
}

/// The session assembly factory (services.rs `create_initial_runtime_from_cli`) calls this once at startup, converging skills, prompt
/// templates, themes, AGENTS.md context, and system prompt — five resource paths — into a session-consumable resource set in one pass — "one-stop shop:"
/// everything needed is loaded, failures do not interrupt", eliminating repeated directory discovery and deduplication by each consumer.
///
/// Behavior (aligns with the assembly semantics of pi `DefaultResourceLoader.reload()`; stateless, one-shot):
/// - Skills/templates/context/themes each delegate to existing loaders (T-AP-03-3/03-4/03-1 + `themes::load_themes`),
///   no_* flags control each path's default directories (explicit paths are always kept; pi `noX ? mergePaths(cli, additional) : ...` semantics);
/// - Templates are **truly deduplicated** by first occurrence of name (later occurrences removed + `PromptCollision` diagnostic; aligns with pi `dedupePrompts`,
///   slash completion list contains no duplicates; routing still hits the first occurrence);
/// - System prompt: non-empty CLI `system_prompt` takes precedence (explicit empty string short-circuits: no file discovery and no injection; aligns with pi `!input`),
///   otherwise discover SYSTEM.md; append prompt: non-empty CLI entries take precedence (empty entries removed); all empty/not provided → discover
///   APPEND_SYSTEM.md; each entry goes through `resolve_prompt_input` (path → read content / inline text as-is; on read failure fall back to original text);
/// - Diagnostics are non-fatal: five-way diagnostics are aggregated into `ResourceLoadDiagnostic`; artifacts and diagnostics may both be non-empty.
///
/// # Parameters
/// - `options`: assembly input (cwd/agent_dir/three explicit path inputs/four no_* flags/system prompt sources).
///
/// # Returns
/// `LoadResourcesResult` (skills/templates/themes/context/system prompt + unified diagnostics).
pub async fn load_resources(options: &LoadResourcesOptions) -> LoadResourcesResult {
    let mut diagnostics = Vec::new();

    // 1. Skills (always loaded; no_skills disables default directories, explicit paths kept)
    let skills_result = skills::load_skills(&skills::LoadSkillsOptions {
        cwd: options.cwd.clone(),
        agent_dir: options.agent_dir.clone(),
        skill_paths: options.skill_paths.clone(),
        include_defaults: !options.no_skills,
    })
    .await;
    let skills = skills_result.skills;
    diagnostics.extend(
        skills_result
            .diagnostics
            .into_iter()
            .map(ResourceLoadDiagnostic::Skill),
    );

    // 2. Templates (no_prompt_templates disables the whole path; aligns with pi branch)
    let mut prompts = Vec::new();
    if !options.no_prompt_templates {
        let prompts_result = prompt_templates::load_prompt_templates(
            &prompt_templates::LoadPromptTemplatesOptions {
                cwd: options.cwd.clone(),
                agent_dir: options.agent_dir.clone(),
                prompt_paths: options.prompt_paths.clone(),
                include_defaults: !options.no_prompt_templates,
            },
        )
        .await;
        diagnostics.extend(
            prompts_result
                .diagnostics
                .into_iter()
                .map(ResourceLoadDiagnostic::Prompt),
        );
        let (deduped, collisions) = dedupe_prompts(prompts_result.prompt_templates);
        prompts = deduped;
        diagnostics.extend(
            collisions
                .into_iter()
                .map(ResourceLoadDiagnostic::PromptCollision),
        );
    }

    // 3. Themes (no_themes disables default directories, explicit paths kept; synchronous implementation calls directly)
    let mut themes = Vec::new();
    if !options.no_themes {
        let themes_result = themes::load_themes(&themes::LoadThemesOptions {
            cwd: options.cwd.clone(),
            agent_dir: options.agent_dir.clone(),
            theme_paths: options.theme_paths.clone(),
            include_defaults: !options.no_themes,
        });
        themes = themes_result.themes;
        diagnostics.extend(
            themes_result
                .diagnostics
                .into_iter()
                .map(ResourceLoadDiagnostic::Theme),
        );
    }

    // 4. Project context files (AGENTS.md/CLAUDE.md)
    let context_files = if options.no_context_files {
        Vec::new()
    } else {
        load_project_context_files(&options.cwd, &options.agent_dir)
    };

    // 5. System prompt: non-empty CLI value takes precedence; explicit empty string short-circuits (aligns with pi `"" ?? discover` = "" → undefined);
    //    None → SYSTEM.md discovery
    let system_prompt = match &options.system_prompt {
        Some(s) if !s.is_empty() => Some(resolve_prompt_input(s, "system prompt")),
        Some(_) => None, // Explicit empty string: neither discover nor inject
        None => discover_system_prompt_file(&options.cwd, &options.agent_dir)
            .map(|p| resolve_prompt_input(&p.to_string_lossy(), "system prompt")),
    };

    // 6. Append prompt: non-empty CLI entries take priority (empty entries dropped, matching pi filter s !== undefined);
    //    All empty/not provided → APPEND_SYSTEM.md discovery
    let append_system_prompt = if options.append_system_prompt.is_empty() {
        discover_append_system_prompt_file(&options.cwd, &options.agent_dir)
            .map(|p| resolve_prompt_input(&p.to_string_lossy(), "append system prompt"))
            .into_iter()
            .collect()
    } else {
        options
            .append_system_prompt
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| resolve_prompt_input(s, "append system prompt"))
            .collect()
    };

    LoadResourcesResult {
        skills,
        prompts,
        themes,
        context_files,
        system_prompt,
        append_system_prompt,
        diagnostics,
    }
}

/// Called inside load_resources when the CLI does not provide --system-prompt: locates the user's SYSTEM.md in the project or global location
/// Custom system prompt file — "project first, then global" (pi `discoverSystemPromptFile`;
/// v1 has no project trust mechanism, so project files always apply).
///
/// Order: `<cwd>/.zen/SYSTEM.md` (hit if present) → `<agentDir>/SYSTEM.md` → None.
/// # Returns
/// Matched file path (`None` = neither location exists).
fn discover_system_prompt_file(cwd: &Path, agent_dir: &Path) -> Option<PathBuf> {
    let project = cwd.join(CONFIG_DIR_NAME).join("SYSTEM.md");
    if project.exists() {
        return Some(project);
    }
    let global = agent_dir.join("SYSTEM.md");
    if global.exists() {
        return Some(global);
    }
    None
}

/// Called inside load_resources when the CLI does not provide --append-system-prompt: locates APPEND_SYSTEM.md
/// Append system prompt file (pi `discoverAppendSystemPromptFile`, same order as above).
/// # Returns
/// Matched file path (`None` = neither location exists).
fn discover_append_system_prompt_file(cwd: &Path, agent_dir: &Path) -> Option<PathBuf> {
    let project = cwd.join(CONFIG_DIR_NAME).join("APPEND_SYSTEM.md");
    if project.exists() {
        return Some(project);
    }
    let global = agent_dir.join("APPEND_SYSTEM.md");
    if global.exists() {
        return Some(global);
    }
    None
}

/// Called inside load_resources for each input entry of the system prompt/append prompt: distinguishes "path to an existing file (read file contents)"
/// from "inline text (used as-is)" — pi `resolvePromptInput`'s "path first, fall back to original text on read failure" semantics.
///
/// Behavior: `input` is a path to an existing file → read file contents (utf-8); read failure → `tracing::warn!` and return
/// the original `input` string (matching pi console.error + return input); non-existing path/inline text → return as-is.
/// Empty-string/empty checks are short-circuited uniformly by the caller (load_resources) (matching pi `!input` → undefined); this function does not handle them again.
/// # Parameters
/// - `input`: prompt entry provided by the CLI or a discovered file path string (non-empty);
/// - `description`: resource description for diagnostic messages (e.g. "system prompt", matching pi warning text).
/// # Returns
/// Final prompt text (`String`).
fn resolve_prompt_input(input: &str, description: &str) -> String {
    let path = Path::new(input);
    if path.exists() {
        match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(e) => {
                tracing::warn!("could not read {description} file {input}: {e}");
                input.to_string()
            }
        }
    } else {
        input.to_string()
    }
}

/// Called inside load_resources after template loading completes: **dedupes** by first occurrence of name and produces collision diagnostics — matching pi
/// `dedupePrompts` (`Array.from(seen.values())` actually removes duplicates, not just diagnostics): the slash-completion list
/// (build_slash_command_list builds commands per template) and the interactive template list no longer contain duplicates.
/// # Parameters
/// - `prompts`: `load_prompt_templates` result in original order (assembly order global > project > explicit), consumed by value.
/// # Returns
/// `(deduped template list, collision diagnostics)` — dedupe keeps the first occurrence; later ones go into diagnostics (winner/loser paths taken from
/// `LoadedPromptTemplate.path`, an incremental field from T-AP-03-4).
fn dedupe_prompts(
    prompts: Vec<LoadedPromptTemplate>,
) -> (Vec<LoadedPromptTemplate>, Vec<PromptCollision>) {
    let mut seen: Vec<LoadedPromptTemplate> = Vec::new();
    let mut collisions = Vec::new();
    for p in prompts {
        let name = p.template.name.clone();
        if let Some(existing) = seen.iter().find(|e| e.template.name == name) {
            collisions.push(PromptCollision {
                name,
                winner_path: existing.path.clone(),
                loser_path: p.path.clone(),
            });
        } else {
            seen.push(p);
        }
    }
    (seen, collisions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt_templates::PromptTemplateSource;
    use crate::skills::LoadedSkill;
    use zen_core::prompt_templates::PromptTemplate;

    fn make_template(name: &str, path: &str) -> LoadedPromptTemplate {
        LoadedPromptTemplate {
            template: PromptTemplate {
                name: name.to_string(),
                description: String::new(),
                argument_hint: None,
                content: "body".to_string(),
            },
            source: PromptTemplateSource::User,
            path: path.to_string(),
        }
    }

    // ─── dedupe_prompts (pure function) ────────────────────────────────

    #[test]
    fn test_dedupe_prompts_collision_keeps_first() {
        let (deduped, collisions) = dedupe_prompts(vec![
            make_template("clash", "/a/clash.md"),
            make_template("clash", "/b/clash.md"),
            make_template("other", "/c/other.md"),
        ]);
        assert_eq!(deduped.len(), 2);
        assert_eq!(deduped[0].template.name, "clash");
        assert_eq!(deduped[0].path, "/a/clash.md"); // keep first occurrence
        assert_eq!(deduped[1].template.name, "other");
        assert_eq!(collisions.len(), 1);
        assert_eq!(collisions[0].name, "clash");
        assert_eq!(collisions[0].winner_path, "/a/clash.md");
        assert_eq!(collisions[0].loser_path, "/b/clash.md");
    }

    #[test]
    fn test_dedupe_prompts_no_collisions() {
        let (deduped, collisions) = dedupe_prompts(vec![
            make_template("a", "/a.md"),
            make_template("b", "/b.md"),
        ]);
        assert_eq!(deduped.len(), 2);
        assert!(collisions.is_empty());
    }

    // ─── discover_* / resolve_prompt_input (pure functions + temp files) ───

    #[test]
    fn test_discover_system_prompt_file_project_wins_over_global() {
        let tmp = tempfile::tempdir().unwrap();
        let agent_dir = tmp.path().join("agent");
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(&agent_dir).unwrap();
        std::fs::create_dir_all(cwd.join(CONFIG_DIR_NAME)).unwrap();
        std::fs::write(agent_dir.join("SYSTEM.md"), "global").unwrap();
        std::fs::write(cwd.join(CONFIG_DIR_NAME).join("SYSTEM.md"), "project").unwrap();

        let found = discover_system_prompt_file(&cwd, &agent_dir).unwrap();
        assert_eq!(found, cwd.join(CONFIG_DIR_NAME).join("SYSTEM.md"));
    }

    #[test]
    fn test_discover_system_prompt_file_global_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        let agent_dir = tmp.path().join("agent");
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(&agent_dir).unwrap();
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::write(agent_dir.join("SYSTEM.md"), "global").unwrap();

        let found = discover_system_prompt_file(&cwd, &agent_dir).unwrap();
        assert_eq!(found, agent_dir.join("SYSTEM.md"));
    }

    #[test]
    fn test_discover_system_prompt_file_none() {
        let tmp = tempfile::tempdir().unwrap();
        let agent_dir = tmp.path().join("agent");
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(&agent_dir).unwrap();
        std::fs::create_dir_all(&cwd).unwrap();
        assert!(discover_system_prompt_file(&cwd, &agent_dir).is_none());
        assert!(discover_append_system_prompt_file(&cwd, &agent_dir).is_none());
    }

    #[test]
    fn test_discover_append_system_prompt_file() {
        let tmp = tempfile::tempdir().unwrap();
        let agent_dir = tmp.path().join("agent");
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(cwd.join(CONFIG_DIR_NAME)).unwrap();
        std::fs::write(cwd.join(CONFIG_DIR_NAME).join("APPEND_SYSTEM.md"), "append").unwrap();

        let found = discover_append_system_prompt_file(&cwd, &agent_dir).unwrap();
        assert_eq!(found, cwd.join(CONFIG_DIR_NAME).join("APPEND_SYSTEM.md"));
    }

    #[test]
    fn test_resolve_prompt_input_reads_existing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let f = tmp.path().join("sp.md");
        std::fs::write(&f, "file content").unwrap();
        assert_eq!(
            resolve_prompt_input(&f.to_string_lossy(), "system prompt"),
            "file content"
        );
    }

    #[test]
    fn test_resolve_prompt_input_inline_text_unchanged() {
        assert_eq!(
            resolve_prompt_input("--system-prompt inline text", "system prompt"),
            "--system-prompt inline text"
        );
        // non-existing path → return as-is (pi treats non-existent as inline text)
        assert_eq!(
            resolve_prompt_input("/nonexistent/path.md", "system prompt"),
            "/nonexistent/path.md"
        );
    }

    // ─── load_resources integration (tempdir + real files) ────────────────

    /// Real built-in theme JSON (same source as themes.rs tests: all 52 color fields, replacing name yields a minimal valid fixture).
    const BUILTIN_DARK_JSON: &str = include_str!("../../../zen-tui/src/theme/builtin_dark.json");

    fn theme_json(name: &str) -> String {
        BUILTIN_DARK_JSON.replace("\"name\": \"dark\"", &format!("\"name\": \"{name}\""))
    }

    /// Builds a complete fixture directory containing skills/prompts/themes/SYSTEM.md/AGENTS.md.
    fn setup_workspace(tmp: &tempfile::TempDir) -> (PathBuf, PathBuf) {
        let agent_dir = tmp.path().join("agent");
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(agent_dir.join("skills")).unwrap();
        std::fs::create_dir_all(agent_dir.join("prompts")).unwrap();
        std::fs::create_dir_all(agent_dir.join("themes")).unwrap();
        std::fs::create_dir_all(cwd.join(CONFIG_DIR_NAME)).unwrap();
        std::fs::write(
            agent_dir.join("skills").join("SKILL.md"),
            "---\nname: s1\ndescription: d\n---\nbody",
        )
        .unwrap();
        std::fs::write(
            agent_dir.join("prompts").join("p1.md"),
            "---\ndescription: d\n---\nbody $1",
        )
        .unwrap();
        std::fs::write(agent_dir.join("themes").join("t1.json"), theme_json("t1")).unwrap();
        std::fs::write(agent_dir.join("SYSTEM.md"), "global sp").unwrap();
        std::fs::write(cwd.join(CONFIG_DIR_NAME).join("SYSTEM.md"), "project sp").unwrap();
        std::fs::write(cwd.join("AGENTS.md"), "project instructions").unwrap();
        (agent_dir, cwd)
    }

    #[tokio::test]
    async fn test_load_resources_full_workspace() {
        let tmp = tempfile::tempdir().unwrap();
        let (agent_dir, cwd) = setup_workspace(&tmp);

        let result = load_resources(&LoadResourcesOptions {
            cwd: cwd.clone(),
            agent_dir: agent_dir.clone(),
            skill_paths: Vec::new(),
            prompt_paths: Vec::new(),
            theme_paths: Vec::new(),
            no_skills: false,
            no_prompt_templates: false,
            no_themes: false,
            no_context_files: false,
            system_prompt: None,
            append_system_prompt: Vec::new(),
        })
        .await;

        // skills/templates/themes all loaded
        assert_eq!(result.skills.len(), 1);
        assert_eq!(result.skills[0].skill.name, "s1");
        assert_eq!(result.prompts.len(), 1);
        assert_eq!(result.prompts[0].template.name, "p1");
        assert_eq!(result.themes.len(), 1);
        assert_eq!(result.themes[0].theme.name, "t1");
        // context file: cwd's AGENTS.md is hit
        assert!(result
            .context_files
            .iter()
            .any(|c| c.content.contains("project instructions")));
        // system prompt: project SYSTEM.md takes priority over global
        assert_eq!(result.system_prompt.as_deref(), Some("project sp"));
        assert!(result.append_system_prompt.is_empty());
        assert!(result.diagnostics.is_empty());
    }

    #[tokio::test]
    async fn test_load_resources_cli_values_beat_discovery() {
        let tmp = tempfile::tempdir().unwrap();
        let (agent_dir, cwd) = setup_workspace(&tmp);

        let result = load_resources(&LoadResourcesOptions {
            cwd: cwd.clone(),
            agent_dir: agent_dir.clone(),
            skill_paths: Vec::new(),
            prompt_paths: Vec::new(),
            theme_paths: Vec::new(),
            no_skills: false,
            no_prompt_templates: false,
            no_themes: false,
            no_context_files: false,
            system_prompt: Some("cli sp".to_string()),
            append_system_prompt: vec!["cli append".to_string()],
        })
        .await;

        // CLI value takes priority over SYSTEM.md discovery
        assert_eq!(result.system_prompt.as_deref(), Some("cli sp"));
        assert_eq!(result.append_system_prompt, vec!["cli append"]);
    }

    #[tokio::test]
    async fn test_load_resources_empty_string_short_circuits() {
        let tmp = tempfile::tempdir().unwrap();
        let (agent_dir, cwd) = setup_workspace(&tmp);

        let result = load_resources(&LoadResourcesOptions {
            cwd: cwd.clone(),
            agent_dir: agent_dir.clone(),
            skill_paths: Vec::new(),
            prompt_paths: Vec::new(),
            theme_paths: Vec::new(),
            no_skills: false,
            no_prompt_templates: false,
            no_themes: false,
            no_context_files: false,
            // explicit empty string: no file discovery and no injection (matching pi `!input` short-circuit)
            system_prompt: Some(String::new()),
            append_system_prompt: vec![String::new(), "keep me".to_string()],
        })
        .await;

        assert_eq!(result.system_prompt, None); // empty string short-circuits, SYSTEM.md contents not injected
        assert_eq!(result.append_system_prompt, vec!["keep me"]); // empty entries dropped
    }

    #[tokio::test]
    async fn test_load_resources_no_flags_disable_respectively() {
        let tmp = tempfile::tempdir().unwrap();
        let (agent_dir, cwd) = setup_workspace(&tmp);

        let result = load_resources(&LoadResourcesOptions {
            cwd: cwd.clone(),
            agent_dir: agent_dir.clone(),
            skill_paths: Vec::new(),
            prompt_paths: Vec::new(),
            theme_paths: Vec::new(),
            no_skills: true,
            no_prompt_templates: true,
            no_themes: true,
            no_context_files: true,
            system_prompt: None,
            append_system_prompt: Vec::new(),
        })
        .await;

        assert!(result.skills.is_empty());
        assert!(result.prompts.is_empty());
        assert!(result.themes.is_empty());
        assert!(result.context_files.is_empty());
    }

    #[tokio::test]
    async fn test_load_resources_prompt_collision_diagnosed() {
        let tmp = tempfile::tempdir().unwrap();
        let (agent_dir, cwd) = setup_workspace(&tmp);
        // add a same-name template in the project dir → global loads first (User first occurrence), project version removed by collision
        std::fs::create_dir_all(cwd.join(CONFIG_DIR_NAME).join("prompts")).unwrap();
        std::fs::write(
            cwd.join(CONFIG_DIR_NAME).join("prompts").join("p1.md"),
            "---\ndescription: d\n---\nproject body",
        )
        .unwrap();

        let result = load_resources(&LoadResourcesOptions {
            cwd: cwd.clone(),
            agent_dir: agent_dir.clone(),
            skill_paths: Vec::new(),
            prompt_paths: Vec::new(),
            theme_paths: Vec::new(),
            no_skills: false,
            no_prompt_templates: false,
            no_themes: false,
            no_context_files: false,
            system_prompt: None,
            append_system_prompt: Vec::new(),
        })
        .await;

        // after dedupe only the global p1 remains; collision diagnostics carry winner/loser paths
        assert_eq!(result.prompts.len(), 1);
        assert_eq!(
            result.prompts[0].path,
            agent_dir.join("prompts/p1.md").to_string_lossy()
        );
        let collision = result
            .diagnostics
            .iter()
            .find_map(|d| match d {
                ResourceLoadDiagnostic::PromptCollision(c) => Some(c),
                _ => None,
            })
            .expect("Should have prompt collision diagnostics");
        assert_eq!(
            collision.winner_path,
            agent_dir.join("prompts/p1.md").to_string_lossy()
        );
        assert_eq!(
            collision.loser_path,
            cwd.join(CONFIG_DIR_NAME)
                .join("prompts")
                .join("p1.md")
                .to_string_lossy()
        );
    }

    #[tokio::test]
    async fn test_load_resources_explicit_paths_loaded_as_path_source() {
        let tmp = tempfile::tempdir().unwrap();
        let (agent_dir, cwd) = setup_workspace(&tmp);
        let extra_skill = tmp.path().join("extra-skill.md");
        std::fs::write(&extra_skill, "---\nname: s2\ndescription: d\n---\nbody").unwrap();

        let result = load_resources(&LoadResourcesOptions {
            cwd: cwd.clone(),
            agent_dir: agent_dir.clone(),
            skill_paths: vec![extra_skill.clone()],
            prompt_paths: Vec::new(),
            theme_paths: Vec::new(),
            no_skills: false,
            no_prompt_templates: true,
            no_themes: true,
            no_context_files: true,
            system_prompt: None,
            append_system_prompt: Vec::new(),
        })
        .await;

        assert_eq!(result.skills.len(), 2); // default + explicit
        assert!(result.skills.iter().any(|s: &LoadedSkill| s.source
            == crate::skills::SkillSource::Path
            && s.skill.name == "s2"));
    }
}
