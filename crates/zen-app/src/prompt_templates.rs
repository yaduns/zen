//! Prompt template management assembly layer: multi-directory assembly + source tracing + `/name args` routing expansion.
//!
//! REQ-APP-03 (pi-307): prompt template management.
//! Application-layer assembly fills in the remaining shape: the pure-function core (`parse_command_args`/`substitute_args`) and explicit path loading are already
//! covered by zen-core (T-AG-05-3); this module supplies the remaining capabilities of pi-307 — default directories
//! (`<agentDir>/prompts` + `<cwd>/.zen/prompts`) + explicit path assembly, source tracing
//! (lightweight `PromptTemplateSource`, full SourceInfo belongs to T-AP-05-4), `/name args` routing expansion
//! (`expand_prompt_template`, designated landing point per the T-AG-05-3 seam).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use regex::Regex;
use zen_core::paths::{resolve_path, PathInputOptions};
use zen_core::prompt_templates::{
    load_prompt_templates as core_load_prompt_templates, parse_command_args, substitute_args,
    LoadPromptTemplatesResult as CoreLoadResult, PromptTemplate, PromptTemplateDiagnostic,
};

use crate::config::CONFIG_DIR_NAME;

/// Lightweight source-tracing identity for template origins — resolves the scope semantics of pi's `createSyntheticSourceInfo`.
///
/// v1 has no extension system; pi's `source` is always `"local"` and `origin` is always `"top-level"`, so the only real semantics left is the source level
/// (user/project/temporary path); `baseDir` is derived by consumers from the load input path and does not occupy a field.
/// The full `SourceInfo` resource model (scope/origin/baseDir) is spec-assigned to **T-AP-05-4** (pi-319),
/// At that point T-AP-05-3 resource loader performs the unifying `PromptTemplateSource → SourceInfo` mapping.
/// an in-memory assembly artifact, not participating in serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptTemplateSource {
    /// Templates under the default directory `<agentDir>/prompts` (pi scope `"user"`).
    User,
    /// Templates under the default directory `<cwd>/.zen/prompts` (pi scope `"project"`).
    Project,
    /// Templates loaded from explicit paths (CLI `--prompt-template` / settings.prompts extension directories) (pi scope `"temporary"`).
    Path,
}

/// A loaded template + its source + source file path—pi `PromptTemplate` (including sourceInfo/filePath) in zen's
/// assembly form: zen-core `PromptTemplate` (pure domain data, finalized by T-AG-05-3, no path field) + application-layer source tracing.
/// `path` is the T-AP-03-4 seam-note option B addition (`LoadedPromptTemplate` has no per-file path → extend core result
/// to carry the path): for T-AP-05-3 collision diagnostics and T-AP-05-4 source tracing; `baseDir` is derived by the consumer from the path's parent directory,
/// and does not occupy a field.
#[derive(Debug)]
pub struct LoadedPromptTemplate {
    /// Template loaded by zen-core (name/description/argument_hint/content).
    pub template: PromptTemplate,
    /// Template origin (default directory user/project / explicit path).
    pub source: PromptTemplateSource,
    /// Template source file path (for display/diagnostics/source tracing; not used for re-addressing)—default directory = `<dir>/<name>.md`,
    /// explicit file = the file itself (attached by `merge_batch` per batch input path).
    pub path: String,
}

/// Assembly input for `load_prompt_templates` (field-by-field alignment with pi's `LoadPromptTemplatesOptions`).
/// Pure input parameters, no behavior.
pub struct LoadPromptTemplatesOptions {
    /// Session working directory (base for the project template directory `<cwd>/.zen/prompts` + base for explicit path resolution).
    pub cwd: PathBuf,
    /// Global agent configuration directory (base for `<agentDir>/prompts`; produced by T-AP-02-1 `get_agent_dir`).
    pub agent_dir: PathBuf,
    /// Explicit template path (file or directory); CLI `--prompt-template` + settings.prompts are provided by the caller
    /// (T-AP-05-3/assembly chain) merged in.
    pub prompt_paths: Vec<PathBuf>,
    /// When true, loads the default directories `<agentDir>/prompts` (User) + `<cwd>/.zen/prompts` (Project).
    pub include_defaults: bool,
}

/// Aggregate return from `load_prompt_templates`—supports the "warn without aborting" semantics: core loading converts I/O/parsing failures
/// into non-interrupting diagnostics; this result forwards those diagnostics (reuses the core `PromptTemplateDiagnostic`; the application layer adds no new diagnostic
/// categories); successfully loaded templates follow load order (this order determines routing priority; see the `expand_prompt_template` doc comment).
#[derive(Debug)]
pub struct LoadPromptTemplatesResult {
    /// Successfully loaded templates with their source attached; order = global → project → explicit path (the core sorts by name byte order within the same source);
    /// Entries with the same name are not deduplicated (T-AP-05-3 performs collision diagnostics on the consumer side).
    pub prompt_templates: Vec<LoadedPromptTemplate>,
    /// Non-interrupting warnings from core loading (FileInfoFailed/ListFailed/ReadFailed/ParseFailed, may be empty); forwarded unchanged.
    pub diagnostics: Vec<PromptTemplateDiagnostic>,
}

/// The slash command expansion path (T-AP-03-2) is called before user input is submitted to the agent: if the text has the form `/templateName args` and
/// a template with the same name is found in the template list, substitute the parameters into the template body to produce the final prompt—expand known / commands and leave unknown
/// ones unchanged. It is a pure function, performs no file I/O, and does not depend on session state.
///
/// Behavior (aligned with pi-307 `expandPromptTemplate`; routing uses first-match order):
/// - Text does not start with `/` → return unchanged (pi `if (!text.startsWith("/")) return text`);
/// - Matches `^/([^\s]+)(?:\s+([\s\S]*))?$` exactly: group 1 = template name (non-whitespace string), group 2 = parameter string
///   (may be empty or contain newlines—`[\s\S]` semantics; equivalent to `(?s:.*)` in Rust); no match → return unchanged;
/// - Template hit (`templates.iter().find(|t| t.name == template_name)`) →
///   `substitute_args(&template.content, &parse_command_args(args_string))`；
/// - No match → return unchanged (template-name matching against `name` is case-sensitive, consistent with pi).
///
/// # Parameters
/// - `text`: full user input (may include a leading `/`);
/// - `templates`: list of loaded templates (slice of the session's `Vec<PromptTemplate>`; on name conflicts
///   list order determines priority — assembly order global > project > explicit, first occurrence wins; see the T-AP-03-4 implementation notes for the rationale).
///
/// # Returns
/// The substituted template body (`String`) on a match; the original text (`text.to_string()`) on no match or for non-template input.
pub fn expand_prompt_template(text: &str, templates: &[PromptTemplate]) -> String {
    if !text.starts_with('/') {
        return text.to_string();
    }

    let captures = match expansion_pattern().captures(text) {
        Some(captures) => captures,
        None => return text.to_string(),
    };

    let template_name = &captures[1];
    let args_string = captures.get(2).map(|m| m.as_str()).unwrap_or("");

    let Some(template) = templates.iter().find(|t| t.name == template_name) else {
        return text.to_string();
    };

    substitute_args(&template.content, &parse_command_args(args_string))
}

/// Matching pattern for expanding `/name args` routes (aligned with pi-307 `expandPromptTemplate` regex):
/// `^/(templateName)(?:\s+(argString))?$`—the template name contains no whitespace; the argument string may be empty or contain newlines (`[\s\S]` is an always-matching character class,
/// equivalent to `(?s:.*)`). The non-greedy match is anchored to the entire string (`^`/`$`).
fn expansion_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(r"^\/([^\s]+)(?:\s+([\s\S]*))?$").expect("static expansion pattern must compile")
        // Static pattern; a failure is a programming error (following the zen-core substitution_pattern precedent)
    })
}

/// Common entry point for assembling the session template manifest: the resource loader (T-AP-05-3) and slash-command completion (T-AP-03-2) call this when collecting
/// the prompt templates configured by the user, consolidating the default directories and explicit paths into a sourced template set in one pass—
/// "Covers all global, project, and CLI-specified templates while tracking their sources."
///
/// Behavior (aligned with pi-307 `loadPromptTemplates`, uses the core explicit-path loader, and mirrors the assembly structure in skills.rs):
/// - When `include_defaults` is set, first load `<agentDir>/prompts` (source=User) and `<cwd>/.zen/prompts`
///   （source=Project，`CONFIG_DIR_NAME` = `.zen`）；
/// - Then process `prompt_paths` in order (files or directories, source=Path): explicit paths are handled via
///   `zen_core::paths::resolve_path` (relative to cwd, `trim: true`) resolves the path, then delegates to the core
///   `load_prompt_templates`——silently skips missing paths, ignores non-`.md` files, and records a core diagnostic on read failure (pi-307
///   Same semantics: no application-layer Warning; aligns with pi rather than skills);
/// - **No deduplication**: templates with the same name coexist in the result (pi-307 behavior); routing priority = result order
///   (global > project > explicit; see the `expand_prompt_template` doc comment); collision diagnostics are handled by the T-AP-05-3 consumer
///   (pi `dedupePrompts` is in resource-loader).
/// - Directory scanning/sorting/symlink handling is all delegated to the core (T-AG-05-3 semantics: non-recursive `.md`, byte-order sorting, broken-link skipping).
///
/// # Parameters
/// - `options`: assembly inputs (cwd/agent_dir/prompt_paths/include_defaults).
///
/// # Returns
/// Forwards the template list with source metadata + core loading diagnostics (both may be non-empty).
pub async fn load_prompt_templates(
    options: &LoadPromptTemplatesOptions,
) -> LoadPromptTemplatesResult {
    let mut result = LoadPromptTemplatesResult {
        prompt_templates: Vec::new(),
        diagnostics: Vec::new(),
    };

    if options.include_defaults {
        // Default directory ①: <agentDir>/prompts (User)
        let user_dir = options.agent_dir.join("prompts");
        merge_batch(
            core_load_prompt_templates(std::slice::from_ref(&user_dir)).await,
            PromptTemplateSource::User,
            &user_dir,
            &mut result,
        );

        // Default directory ②: <cwd>/.zen/prompts (Project)
        let project_dir = options.cwd.join(CONFIG_DIR_NAME).join("prompts");
        merge_batch(
            core_load_prompt_templates(std::slice::from_ref(&project_dir)).await,
            PromptTemplateSource::Project,
            &project_dir,
            &mut result,
        );
    }

    // Explicit paths (source is always Path, aligned with the pi promptPaths branch); core processing silently skips missing paths/ignores non-md files/
    // Record diagnostics on IO failure; this layer only resolves paths and attaches source metadata
    for raw_path in &options.prompt_paths {
        let resolved = match resolve_path(
            &raw_path.to_string_lossy(),
            &options.cwd,
            &PathInputOptions {
                trim: true,
                ..PathInputOptions::default()
            },
        ) {
            Ok(path) => path,
            Err(_) => continue, // Skip silently on path resolution failure (e.g. no home directory), aligning with the explicit path branch of skills.rs
        };
        merge_batch(
            core_load_prompt_templates(std::slice::from_ref(&resolved)).await,
            PromptTemplateSource::Path,
            &resolved,
            &mut result,
        );
    }

    result
}

/// Derive the source file path for a single template (T-AP-03-4 seam-note option B: core result carries path)—
/// input is a directory → `<dir>/<name>.md` (core names by name = file_stem); input is a file → the file itself.
/// For display/diagnostics/source tracing only; not used for re-addressing.
fn derive_template_path(input: &Path, name: &str) -> String {
    if input.is_dir() {
        input
            .join(format!("{name}.md"))
            .to_string_lossy()
            .into_owned()
    } else {
        input.to_string_lossy().into_owned()
    }
}

/// Merge one core loading result into the assembly result: attach source metadata to templates in batches (path-prefix comparison from pi `getSourceInfo`
/// The determination is replaced in zen by "source known at call time", see T-AP-03-4 implementation note 5.2) and source file path; diagnostics are forwarded as-is.
fn merge_batch(
    core: CoreLoadResult,
    source: PromptTemplateSource,
    input_path: &Path,
    result: &mut LoadPromptTemplatesResult,
) {
    for template in core.prompt_templates {
        let path = derive_template_path(input_path, &template.name);
        result.prompt_templates.push(LoadedPromptTemplate {
            template,
            source,
            path,
        });
    }
    result.diagnostics.extend(core.diagnostics);
}

#[cfg(test)]
mod tests {
    use super::*;
    use zen_core::prompt_templates::PromptTemplateDiagnosticCode;

    fn make_template(name: &str, content: &str) -> PromptTemplate {
        PromptTemplate {
            name: name.to_string(),
            description: String::new(),
            argument_hint: None,
            content: content.to_string(),
        }
    }

    // ─── expand_prompt_template (pure function, no IO) ─────────────────

    #[test]
    fn test_expand_non_slash_text_unchanged() {
        assert_eq!(expand_prompt_template("hello world", &[]), "hello world");
    }

    #[test]
    fn test_expand_unknown_template_passthrough() {
        let templates = [make_template("review", "Review $1")];
        assert_eq!(
            expand_prompt_template("/missing a b", &templates),
            "/missing a b"
        );
    }

    #[test]
    fn test_expand_plain_name_no_args() {
        let templates = [make_template("review", "Review the diff")];
        assert_eq!(
            expand_prompt_template("/review", &templates),
            "Review the diff"
        );
    }

    #[test]
    fn test_expand_with_positional_args() {
        let templates = [make_template("fix", "Fix $1 for $2")];
        assert_eq!(
            expand_prompt_template("/fix foo bar", &templates),
            "Fix foo for bar"
        );
    }

    #[test]
    fn test_expand_args_quoted_groups() {
        let templates = [make_template("run", "run $1")];
        assert_eq!(
            expand_prompt_template("/run \"two words\"", &templates),
            "run two words"
        );
    }

    #[test]
    fn test_expand_default_placeholder_when_arg_missing() {
        let templates = [make_template("greet", "Hi ${1:-there}")];
        assert_eq!(expand_prompt_template("/greet", &templates), "Hi there");
        assert_eq!(expand_prompt_template("/greet zen", &templates), "Hi zen");
    }

    #[test]
    fn test_expand_arg_value_not_recursively_substituted() {
        // Single-pass semantics: placeholder forms in argument values remain literal (core substitute_args behavior)
        let templates = [make_template("t", "$1")];
        assert_eq!(expand_prompt_template("/t \"$2\"", &templates), "$2");
    }

    #[test]
    fn test_expand_trailing_space_yields_empty_args() {
        let templates = [make_template("t", "[$@]")];
        assert_eq!(expand_prompt_template("/t ", &templates), "[]");
    }

    #[test]
    fn test_expand_first_match_wins_routing_priority() {
        // Templates with the same name coexist: first match in list order (assembly order: global > project > explicit)
        let templates = [
            make_template("x", "user version"),
            make_template("x", "project version"),
        ];
        assert_eq!(expand_prompt_template("/x", &templates), "user version");
    }

    #[test]
    fn test_expand_case_sensitive_name() {
        let templates = [make_template("Foo", "content")];
        assert_eq!(expand_prompt_template("/foo", &templates), "/foo");
    }

    #[test]
    fn test_expand_lone_slash_not_a_template() {
        let templates = [make_template("", "content")];
        assert_eq!(expand_prompt_template("/", &templates), "/");
    }

    #[test]
    fn test_expand_args_may_contain_newlines() {
        // Argument strings may span lines (pi `[\s\S]*` semantics)
        let templates = [make_template("multi", "got: $@")];
        assert_eq!(
            expand_prompt_template("/multi line1\nline2", &templates),
            "got: line1 line2"
        );
    }

    // ─── load_prompt_templates integration (tempdir + real files) ─────────

    #[tokio::test]
    async fn test_load_defaults_user_and_project() {
        let tmp = tempfile::tempdir().unwrap();
        let agent_dir = tmp.path().join("agent");
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(agent_dir.join("prompts")).unwrap();
        std::fs::create_dir_all(cwd.join(CONFIG_DIR_NAME).join("prompts")).unwrap();
        std::fs::write(
            agent_dir.join("prompts").join("global.md"),
            "---\ndescription: from user\n---\nbody $1",
        )
        .unwrap();
        std::fs::write(
            cwd.join(CONFIG_DIR_NAME).join("prompts").join("local.md"),
            "---\ndescription: from project\n---\nbody",
        )
        .unwrap();

        let opts = LoadPromptTemplatesOptions {
            cwd: cwd.clone(),
            agent_dir: agent_dir.clone(),
            prompt_paths: Vec::new(),
            include_defaults: true,
        };
        let result = load_prompt_templates(&opts).await;
        assert_eq!(result.prompt_templates.len(), 2);
        assert!(result.diagnostics.is_empty());

        let user = result
            .prompt_templates
            .iter()
            .find(|p| p.template.name == "global")
            .unwrap();
        assert_eq!(user.source, PromptTemplateSource::User);
        assert_eq!(
            user.path,
            agent_dir
                .join("prompts")
                .join("global.md")
                .to_string_lossy()
        );
        let project = result
            .prompt_templates
            .iter()
            .find(|p| p.template.name == "local")
            .unwrap();
        assert_eq!(project.source, PromptTemplateSource::Project);
        assert_eq!(
            project.path,
            cwd.join(CONFIG_DIR_NAME)
                .join("prompts")
                .join("local.md")
                .to_string_lossy()
        );
    }

    #[tokio::test]
    async fn test_load_explicit_dir_and_file() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(&cwd).unwrap();
        let extra_dir = tmp.path().join("extra");
        std::fs::create_dir_all(&extra_dir).unwrap();
        std::fs::write(extra_dir.join("extra.md"), "---\n---\nextra body").unwrap();
        let single_file = tmp.path().join("single.md");
        std::fs::write(&single_file, "---\n---\nsingle body").unwrap();

        let opts = LoadPromptTemplatesOptions {
            cwd: cwd.clone(),
            agent_dir: tmp.path().to_path_buf(),
            prompt_paths: vec![extra_dir.clone(), single_file.clone()],
            include_defaults: false,
        };
        let result = load_prompt_templates(&opts).await;
        assert_eq!(result.prompt_templates.len(), 2);
        assert!(result
            .prompt_templates
            .iter()
            .all(|p| p.source == PromptTemplateSource::Path));
        let extra = result
            .prompt_templates
            .iter()
            .find(|p| p.template.name == "extra")
            .unwrap();
        assert_eq!(extra.path, extra_dir.join("extra.md").to_string_lossy());
        let single = result
            .prompt_templates
            .iter()
            .find(|p| p.template.name == "single")
            .unwrap();
        assert_eq!(single.path, single_file.to_string_lossy());
        assert!(result.diagnostics.is_empty());
    }

    #[tokio::test]
    async fn test_load_explicit_path_resolved_against_cwd() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(&cwd).unwrap();
        // Relative paths are resolved via cwd (aligned with pi resolvePath(..., cwd, {trim:true}))
        std::fs::write(cwd.join("rel.md"), "---\n---\nrel body").unwrap();

        let opts = LoadPromptTemplatesOptions {
            cwd: cwd.clone(),
            agent_dir: tmp.path().to_path_buf(),
            prompt_paths: vec![PathBuf::from("rel.md")],
            include_defaults: false,
        };
        let result = load_prompt_templates(&opts).await;
        assert_eq!(result.prompt_templates.len(), 1);
        assert_eq!(result.prompt_templates[0].template.name, "rel");
        assert_eq!(
            result.prompt_templates[0].source,
            PromptTemplateSource::Path
        );
    }

    #[tokio::test]
    async fn test_load_missing_and_non_md_explicit_silently_skipped() {
        // pi-307 semantics: silently skip missing paths, ignore non-.md files, and emit no diagnostics (align with pi rather than skills)
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(&cwd).unwrap();
        let non_md = tmp.path().join("note.txt");
        std::fs::write(&non_md, "hello").unwrap();

        let opts = LoadPromptTemplatesOptions {
            cwd: cwd.clone(),
            agent_dir: tmp.path().to_path_buf(),
            prompt_paths: vec![tmp.path().join("missing"), non_md],
            include_defaults: false,
        };
        let result = load_prompt_templates(&opts).await;
        assert!(result.prompt_templates.is_empty());
        assert!(result.diagnostics.is_empty());
    }

    #[tokio::test]
    async fn test_load_ordering_global_project_explicit() {
        // Same-named templates coexist across three sources: result order = global → project → explicit; the first routing match uses the global version
        let tmp = tempfile::tempdir().unwrap();
        let agent_dir = tmp.path().join("agent");
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(agent_dir.join("prompts")).unwrap();
        std::fs::create_dir_all(cwd.join(CONFIG_DIR_NAME).join("prompts")).unwrap();
        let extra_dir = tmp.path().join("extra");
        std::fs::create_dir_all(&extra_dir).unwrap();
        for (dir, body) in [
            (agent_dir.join("prompts"), "global body"),
            (cwd.join(CONFIG_DIR_NAME).join("prompts"), "project body"),
            (extra_dir.clone(), "explicit body"),
        ] {
            std::fs::write(dir.join("clash.md"), body).unwrap();
        }

        let opts = LoadPromptTemplatesOptions {
            cwd: cwd.clone(),
            agent_dir: agent_dir.clone(),
            prompt_paths: vec![extra_dir],
            include_defaults: true,
        };
        let result = load_prompt_templates(&opts).await;
        assert_eq!(result.prompt_templates.len(), 3); // No deduplication; all three sources coexist
        let sources: Vec<_> = result.prompt_templates.iter().map(|p| p.source).collect();
        assert_eq!(
            sources,
            vec![
                PromptTemplateSource::User,
                PromptTemplateSource::Project,
                PromptTemplateSource::Path
            ]
        );
        let templates: Vec<PromptTemplate> = result
            .prompt_templates
            .iter()
            .map(|p| p.template.clone())
            .collect();
        assert_eq!(expand_prompt_template("/clash", &templates), "global body");
    }

    #[tokio::test]
    async fn test_load_no_defaults_empty_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(&cwd).unwrap();

        let opts = LoadPromptTemplatesOptions {
            cwd: cwd.clone(),
            agent_dir: tmp.path().to_path_buf(),
            prompt_paths: Vec::new(),
            include_defaults: false,
        };
        let result = load_prompt_templates(&opts).await;
        assert!(result.prompt_templates.is_empty());
        assert!(result.diagnostics.is_empty());
    }

    #[tokio::test]
    async fn test_load_malformed_frontmatter_diagnostic_forwarded() {
        // Core parse failures (ParseFailed) are forwarded unchanged by this layer without interrupting the remaining loads
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(&cwd).unwrap();
        let extra_dir = tmp.path().join("extra");
        std::fs::create_dir_all(&extra_dir).unwrap();
        std::fs::write(extra_dir.join("bad.md"), "---\nnot a kv line\n---\nbody").unwrap();
        std::fs::write(extra_dir.join("good.md"), "---\n---\ngood body").unwrap();

        let opts = LoadPromptTemplatesOptions {
            cwd: cwd.clone(),
            agent_dir: tmp.path().to_path_buf(),
            prompt_paths: vec![extra_dir],
            include_defaults: false,
        };
        let result = load_prompt_templates(&opts).await;
        assert_eq!(result.prompt_templates.len(), 1);
        assert_eq!(result.prompt_templates[0].template.name, "good");
        assert_eq!(result.diagnostics.len(), 1);
        assert_eq!(
            result.diagnostics[0].code,
            PromptTemplateDiagnosticCode::ParseFailed
        );
    }
}
