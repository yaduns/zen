// REQ-APP-01 (pi-302, pi-334): startup command-line overrides replayed on every session switch
// Reference: reference/pi-original/packages/coding-agent/src/main.ts (the `createRuntime` closure captures `parsed`
// and is re-run by AgentSessionRuntime on every session replacement)
// Implementation notes: tasks/T-AP-01-6-implemented.md

//! Command-line override snapshot for session assembly.
//!
//! The snapshot is built once at process startup (from the parsed CLI) and stored on the runtime; every session
//! assembled later in this process (`/new`, `/resume`, `/fork`, `/clone`) replays it, so "the command-line
//! settings given at launch" hold for *every* session — this is zen's equivalent of pi's `createRuntime`
//! closure capturing `parsed`. Without it a read-only session (`--tools read,...`) silently regains the
//! default writable tool set after one session switch.

use crate::cli::{parse_thinking_level, CliArgs};
use crate::config::defaults::DEFAULT_THINKING_LEVEL;
use zen_ai::provider::resolve_provider;
use zen_ai::types::{Model, ThinkingLevel};
use zen_core::tools::factory::ToolSelection;

/// The command-line overrides that were in effect when this process started.
///
/// Consumers: `session::services` (assembly: both the startup entry and the session-switch entry) and
/// `session::runtime` (held on [`crate::session::runtime::AgentSessionRuntime`] and replayed on switches).
///
/// Semantics: an empty/`None`/`false` field means "this launch specified no override for that knob", so the
/// assembly falls back to session restore and settings. Therefore [`Default`] must be equivalent to "no CLI
/// override at all" (default built-in tool group, default resource discovery, model/thinking from the
/// session-restore chain).
///
/// Not serialized: pure runtime data (never written to the session JSONL, never merged into settings).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SessionAssemblyOptions {
    /// Tool selection (`--tools`/`--exclude-tools`/`--no-tools`/`--no-builtin-tools`).
    /// [`ToolSelection::default`] = the default built-in group `[read, bash, edit, write]`.
    pub tool_selection: ToolSelection,
    /// `--provider`/`--model` override, already resolved to a concrete model (`provider/id` and `id:thinking`
    /// shorthands are consumed here). `None` = the launch did not override the model.
    pub cli_model: Option<Model>,
    /// `--thinking` or the `--model "id:thinking"` shorthand. Encoding matters: when the CLI *did* specify
    /// thinking but the shorthand suffix is unparsable (e.g. `--model "id:bogus"`), this is
    /// `Some(DEFAULT_THINKING_LEVEL)` rather than `None`, because the old chain gated session/settings
    /// fallback on `args.model.contains(':')` — storing `None` would let the session value win again.
    pub cli_thinking_level: Option<ThinkingLevel>,
    /// `--system-prompt` (raw value; the empty-string short-circuit is `load_resources`' contract).
    pub system_prompt: Option<String>,
    /// `--append-system-prompt` (repeatable; empty = no override).
    pub append_system_prompt: Vec<String>,
    /// `--skill` (repeatable, kept as the raw strings the user typed; path resolution stays in
    /// `load_resources`, i.e. relative paths resolve against the *target* session cwd).
    pub skills: Vec<String>,
    /// `--prompt-template` (repeatable).
    pub prompt_templates: Vec<String>,
    /// `--theme` (repeatable).
    pub themes: Vec<String>,
    /// `--no-skills`.
    pub no_skills: bool,
    /// `--no-prompt-templates`.
    pub no_prompt_templates: bool,
    /// `--no-themes`.
    pub no_themes: bool,
    /// `--no-context-files` (AGENTS.md/CLAUDE.md discovery).
    pub no_context_files: bool,
}

impl SessionAssemblyOptions {
    /// Startup assembly calls this once (from `create_initial_runtime_from_cli`) to snapshot the parsed CLI.
    ///
    /// User-visible problem it solves: "a read-only session turns writable after one session switch" — from now
    /// on every switch replays this snapshot instead of falling back to the default tool group.
    ///
    /// # Parameters
    /// - `args`: the already-parsed CLI (this function only projects it; it introduces no new flag semantics)
    ///
    /// # Behavior
    /// - Tools: `tools`/`exclude_tools`/`no_tools`/`no_builtin_tools` map straight into [`ToolSelection`]
    ///   (selection semantics stay in `ToolFactory`);
    /// - Model/thinking: `Some` only when the CLI specified them (see [`resolve_cli_model_and_thinking`]);
    /// - Resources: `--skill`/`--prompt-template`/`--theme`/`--no-*`/`--system-prompt`/`--append-system-prompt`
    ///   are copied verbatim;
    /// - CLI knobs unrelated to assembly (`positional`/`session_dir`/`print`/`mode`/`export`/`api_key`, …) are
    ///   **not** part of the snapshot.
    ///
    /// # Boundaries
    /// Not a strict pure function: no file IO and no `Result`, but it reads environment variables and the model
    /// catalog (`resolve_provider`/`Model::from_env`) and may warn on stderr for an unknown provider. Unit tests
    /// must serialize environment access via the `ENV_LOCK`/`EnvVarGuard` helpers used in `session::services`.
    pub fn from_cli_args(args: &CliArgs) -> Self {
        let (cli_model, cli_thinking_level) = resolve_cli_model_and_thinking(args);
        Self {
            tool_selection: ToolSelection {
                tools: args.tools.clone(),
                exclude_tools: args.exclude_tools.clone(),
                no_tools: args.no_tools,
                no_builtin_tools: args.no_builtin_tools,
            },
            cli_model,
            cli_thinking_level,
            system_prompt: args.system_prompt.clone(),
            append_system_prompt: args.append_system_prompt.clone(),
            skills: args.skills.clone(),
            prompt_templates: args.prompt_templates.clone(),
            themes: args.themes.clone(),
            no_skills: args.no_skills,
            no_prompt_templates: args.no_prompt_templates,
            no_themes: args.no_themes,
            no_context_files: args.no_context_files,
        }
    }
}

/// Extract the model/thinking overrides from the CLI (only the parts the user explicitly specified).
///
/// Called by [`SessionAssemblyOptions::from_cli_args`]; replaces the old `resolve_model` shape (which always
/// returned a model and let the assembly chain decide precedence).
///
/// # Returns
/// `(cli_model, cli_thinking_level)`:
/// - `--provider` or `--model` present → `Some(Model)` (`provider/id` and `id:thinking` shorthands are consumed
///   here; an unknown `--provider` warns and falls back to the environment default, still `Some`, matching the
///   old chain which stopped consulting the session in that case);
/// - thinking has three cases: `--thinking <level>` → `Some(level)` (highest priority); `--model "id:<parsable>"`
///   → `Some(level)`; `--model "id:<unparsable>"` → **`Some(DEFAULT_THINKING_LEVEL)`** — the old chain used
///   `args.model.contains(':')` to mean "the CLI specified thinking", so an invalid suffix must still block the
///   session/settings fallback (result = default level); neither present → `None`.
///
/// # Boundaries
/// Reads environment variables and the catalog, and warns on stderr for unknown providers — not a strict pure
/// function. `Model::from_env` itself no longer panics (it falls back to the first catalog model).
fn resolve_cli_model_and_thinking(args: &CliArgs) -> (Option<Model>, Option<ThinkingLevel>) {
    let mut model: Option<Model> = match &args.provider {
        Some(provider_name) => match resolve_provider(provider_name) {
            Some(provider) => Some(Model::from_provider(provider.as_ref())),
            None => {
                eprintln!("Warning: unknown provider '{provider_name}', using default");
                Some(Model::from_env())
            }
        },
        None => None,
    };

    let mut shorthand_thinking: Option<ThinkingLevel> = None;
    let mut shorthand_present = false;
    if let Some(model_arg) = &args.model {
        // `--model` has three forms: provider/id, id, id:thinking
        let (provider_part, id_part) = match model_arg.split_once('/') {
            Some((provider, id)) => (Some(provider), id),
            None => (None, model_arg.as_str()),
        };
        let (id, thinking_part) = match id_part.split_once(':') {
            Some((id, suffix)) => (id, Some(suffix)),
            None => (id_part, None),
        };

        let mut resolved = match provider_part {
            Some(provider_name) => match resolve_provider(provider_name) {
                Some(provider) => Model::from_provider(provider.as_ref()),
                None => {
                    eprintln!("Warning: unknown provider '{provider_name}', using default");
                    model.clone().unwrap_or_else(Model::from_env)
                }
            },
            None => model.clone().unwrap_or_else(Model::from_env),
        };
        resolved.id = id.to_string();
        model = Some(resolved);

        if let Some(suffix) = thinking_part {
            shorthand_present = true;
            shorthand_thinking = parse_thinking_level(suffix).ok();
        }
    }

    let thinking_level = match args.thinking {
        Some(level) => Some(level),
        // The CLI specified thinking through the shorthand: keep the (possibly defaulted) value so the
        // session/settings fallback stays blocked, mirroring the old `contains(':')` guard.
        None if shorthand_present => Some(shorthand_thinking.unwrap_or(DEFAULT_THINKING_LEVEL)),
        None => None,
    };

    (model, thinking_level)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Test helper: construct `CliArgs` (same convention as `cli::args` tests).
    fn parse(args: &[&str]) -> CliArgs {
        let mut full: Vec<&str> = Vec::with_capacity(args.len() + 1);
        full.push("zen");
        full.extend_from_slice(args);
        CliArgs::try_parse_from(full).unwrap()
    }

    #[test]
    fn empty_cli_yields_default_snapshot() {
        let options = SessionAssemblyOptions::from_cli_args(&parse(&[]));
        assert_eq!(
            options,
            SessionAssemblyOptions::default(),
            "no CLI flags must be equivalent to `Default` (no override at all)"
        );
    }

    #[test]
    fn tool_flags_map_into_tool_selection() {
        let options =
            SessionAssemblyOptions::from_cli_args(&parse(&["--tools", "read,grep,find,ls"]));
        assert_eq!(
            options.tool_selection.tools,
            vec!["read", "grep", "find", "ls"]
        );

        let options =
            SessionAssemblyOptions::from_cli_args(&parse(&["--exclude-tools", "bash,edit"]));
        assert_eq!(options.tool_selection.exclude_tools, vec!["bash", "edit"]);

        let options = SessionAssemblyOptions::from_cli_args(&parse(&["--no-tools"]));
        assert!(options.tool_selection.no_tools);

        let options = SessionAssemblyOptions::from_cli_args(&parse(&["--no-builtin-tools"]));
        assert!(options.tool_selection.no_builtin_tools);
    }

    #[test]
    fn resource_flags_map_into_snapshot() {
        let options = SessionAssemblyOptions::from_cli_args(&parse(&[
            "--system-prompt",
            "be terse",
            "--append-system-prompt",
            "extra",
            "--skill",
            "./my-skill",
            "--prompt-template",
            "./tpl.md",
            "--theme",
            "./theme.json",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
            "--no-context-files",
        ]));
        assert_eq!(options.system_prompt.as_deref(), Some("be terse"));
        assert_eq!(options.append_system_prompt, vec!["extra"]);
        // Relative CLI paths stay raw (resolved later against the target session cwd) — the documented deviation.
        assert_eq!(options.skills, vec!["./my-skill"]);
        assert_eq!(options.prompt_templates, vec!["./tpl.md"]);
        assert_eq!(options.themes, vec!["./theme.json"]);
        assert!(options.no_skills);
        assert!(options.no_prompt_templates);
        assert!(options.no_themes);
        assert!(options.no_context_files);
    }

    #[test]
    fn model_flag_resolves_provider_and_id() {
        let options =
            SessionAssemblyOptions::from_cli_args(&parse(&["--model", "huoshan/ark-model-x"]));
        let model = options.cli_model.expect("--model must produce an override");
        assert_eq!(model.provider, "huoshan");
        assert_eq!(model.id, "ark-model-x");
        assert_eq!(
            options.cli_thinking_level, None,
            "no thinking in the flag → no thinking override"
        );
    }

    #[test]
    fn thinking_flag_and_shorthand_map_into_snapshot() {
        let options = SessionAssemblyOptions::from_cli_args(&parse(&["--model", "m:high"]));
        assert_eq!(options.cli_thinking_level, Some(ThinkingLevel::High));

        let options = SessionAssemblyOptions::from_cli_args(&parse(&[
            "--model",
            "m:high",
            "--thinking",
            "off",
        ]));
        assert_eq!(
            options.cli_thinking_level,
            Some(ThinkingLevel::Off),
            "--thinking wins over the model shorthand"
        );
    }

    #[test]
    fn invalid_thinking_shorthand_still_blocks_fallback() {
        // Regression: the old chain treated "model string contains ':'" as "CLI specified thinking", so
        // `--model id:bogus` ended at DEFAULT_THINKING_LEVEL instead of falling back to the session value.
        let options = SessionAssemblyOptions::from_cli_args(&parse(&["--model", "id:bogus"]));
        assert_eq!(
            options.cli_thinking_level,
            Some(DEFAULT_THINKING_LEVEL),
            "unparsable shorthand must encode as Some(default), not None"
        );
    }

    #[test]
    fn unknown_provider_warns_and_still_overrides_with_env_default() {
        let options = SessionAssemblyOptions::from_cli_args(&parse(&["--provider", "nonexistent"]));
        let model = options.cli_model.expect(
            "an explicit (unknown) provider still blocks session restore, like the old chain",
        );
        assert!(!model.provider.is_empty());
    }
}
