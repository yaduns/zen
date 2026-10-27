// REQ-APP-08 (pi-330): rewrite argument parsing with clap
// See reference/pi-original/packages/coding-agent/src/cli/args.ts
// Implementation notes in tasks/T-AP-08-1-implemented.md

use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use zen_ai::types::ThinkingLevel;

/// Output mode: output format for print mode (text = final text only, json = one JSON line per event).
/// Corresponds to pi `Mode` (originally `"text" | "json" | "rpc"`; rpc removed per the REQ-APP-08 simplification).
/// clap `ValueEnum` generates values from variant names in kebab-case by default: `Text → "text"`, `Json → "json"`,
/// exactly matching pi's values, so no explicit rename is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Mode {
    /// Output final text only (default)
    Text,
    /// One JSON line per event
    Json,
}

/// AI coding assistant with read, bash, edit, write tools（zen）
///
/// Typed contract for the process command-line arguments
///
/// Long flag names are auto kebab-cased from field names (`no_session` → `--no-session`,
/// `session_id` → `--session-id`）
#[derive(Debug, Parser)]
#[command(
    name = "zen",
    version,
    disable_version_flag = true,
    after_help = "Examples:
  zen                                Start interactive mode
  zen \"List all .ts files in src/\"
  zen -p \"List all .ts files in src/\"
  zen --mode json -p \"prompt\"      Print mode with JSON event stream
  zen --continue \"What did we discuss?\"
  zen --name \"Refactor auth module\" \"Help me refactor this code\"
  zen --model deepseek-v4-flash \"Help me refactor\"
  zen --tools read,grep -p \"Review the code in src/\"

Environment Variables:
  ZEN_PROVIDER                  Default provider (opencode-go | huoshan)
  ZEN_CODING_AGENT_DIR          Config directory (default ~/.zen/agent)
  ZEN_CODING_AGENT_SESSION_DIR  Session storage directory (overridden by --session-dir)
  OPENCODE_API_KEY              OpenCode Go API key
  ARK_API_KEY                   Volcano Ark (huoshan) API key

Provider/model overrides (base URL, model id, max tokens) live in:
  <ZEN_CODING_AGENT_DIR>/models.json"
)]
pub struct CliArgs {
    /// Provider name (defaults to the env var ZEN_PROVIDER, see zen_ai::provider)
    #[arg(long)]
    pub provider: Option<String>,
    /// Model identifier (raw string, including `provider/id` or `:thinking` shorthand; shorthand parsing belongs to T-AP-08-3)
    #[arg(long)]
    pub model: Option<String>,
    /// API key (defaults to the provider's own env var)
    #[arg(long)]
    pub api_key: Option<String>,
    /// System prompt (default: coding assistant prompt)
    #[arg(long)]
    pub system_prompt: Option<String>,
    /// Text appended to the system prompt (repeatable)
    #[arg(long)]
    pub append_system_prompt: Vec<String>,
    /// Thinking level: off, minimal, low, medium, high, xhigh, max
    #[arg(long, value_parser = parse_thinking_level)]
    pub thinking: Option<ThinkingLevel>,
    /// Continue the previous session (`continue` is a Rust keyword, so the field name has a `_flag` suffix)
    #[arg(long = "continue", short = 'c')]
    pub continue_flag: bool,
    /// Select a session to resume
    #[arg(long, short = 'r')]
    pub resume: bool,
    /// Print the version and exit (clap's automatic version flag is disabled; printed by main)
    #[arg(long, short = 'v')]
    pub version: bool,
    /// Output mode: text (default) or json
    #[arg(long)]
    pub mode: Option<Mode>,
    /// Session display name
    #[arg(long, short = 'n')]
    pub name: Option<String>,
    /// Do not save the session (ephemeral session)
    #[arg(long)]
    pub no_session: bool,
    /// Use the given session file path or partial UUID
    #[arg(long)]
    pub session: Option<String>,
    /// Use the exact project session ID (create it if absent)
    #[arg(long)]
    pub session_id: Option<String>,
    /// Fork a new session from the given session file path or partial UUID
    #[arg(long)]
    pub fork: Option<String>,
    /// Session storage directory (--session-dir > env var > default; precedence belongs to T-AP-08-3)
    #[arg(long)]
    pub session_dir: Option<PathBuf>,
    /// Model modes cycled by Ctrl+P (comma-separated, consumed by T-AP-11)
    #[arg(long, value_delimiter = ',')]
    pub models: Vec<String>,
    /// Whitelist of enabled tools (comma-separated; pi's multi-char short flag `-xt` is unsupported, long flag only)
    #[arg(long, short = 't', value_delimiter = ',', value_parser = parse_trimmed)]
    pub tools: Vec<String>,
    /// Disabled tool blacklist (comma-separated)
    #[arg(long, value_delimiter = ',', value_parser = parse_trimmed)]
    pub exclude_tools: Vec<String>,
    /// Disable all tools by default (pi's multi-char short flag `-nt` is unsupported, long flag only)
    #[arg(long)]
    pub no_tools: bool,
    /// Disable built-in tools by default (pi's multi-char short flag `-nbt` is unsupported, long flag only)
    #[arg(long)]
    pub no_builtin_tools: bool,
    /// Non-interactive mode: process the prompt and exit
    #[arg(long, short = 'p')]
    pub print: bool,
    /// Export the session as HTML and exit (consumption belongs to T-AP-08-3, which decides whether v1 implements it)
    #[arg(long)]
    pub export: Option<PathBuf>,
    /// Disable skill discovery and loading (pi's multi-char short flag `-ns` is unsupported, long flag only)
    #[arg(long)]
    pub no_skills: bool,
    /// Load the given skill file or directory (repeatable; pi's flag is singular `--skill`)
    #[arg(long = "skill")]
    pub skills: Vec<String>,
    /// Load the given prompt template file or directory (repeatable; pi's flag is singular `--prompt-template`)
    #[arg(long = "prompt-template")]
    pub prompt_templates: Vec<String>,
    /// Disable prompt template discovery and loading (pi's multi-char short flag `-np` is unsupported, long flag only)
    #[arg(long)]
    pub no_prompt_templates: bool,
    /// Load the given theme file or directory (repeatable; pi's flag is singular `--theme`)
    #[arg(long = "theme")]
    pub themes: Vec<String>,
    /// Disable theme discovery and loading
    #[arg(long)]
    pub no_themes: bool,
    /// Disable AGENTS.md / CLAUDE.md discovery and loading (T-AP-08-3 skips load_project_context_files accordingly)
    #[arg(long)]
    pub no_context_files: bool,
    /// List available models (optional fuzzy search string): absent=None, bare flag=Some(None), with value=Some(Some(s)),
    /// matching pi's `string | true` three-state
    #[arg(long)]
    pub list_models: Option<Option<String>>,
    /// Disable network operations during startup
    #[arg(long)]
    pub offline: bool,
    /// Force verbose startup output (overrides the quietStartup setting)
    #[arg(long)]
    pub verbose: bool,
    /// Positional arguments: mix of messages and `@file` references (`@`-prefix classification is done by `messages()`/`file_args()`)
    #[arg(value_name = "ARG")]
    pub positional: Vec<String>,
}

impl CliArgs {
    /// Extract messages from the positional arguments (those without the `@` prefix), preserving original order.
    pub fn messages(&self) -> Vec<&str> {
        self.positional
            .iter()
            .filter(|arg| !arg.starts_with('@'))
            .map(String::as_str)
            .collect()
    }

    /// Extract `@file` references from the positional arguments (path strings with the `@` prefix removed).
    /// Note: classification and prefix stripping only
    pub fn file_args(&self) -> Vec<&str> {
        self.positional
            .iter()
            .filter(|arg| arg.starts_with('@'))
            .map(|arg| &arg[1..])
            .collect()
    }
}

/// Parse thinking level values: strict lowercase match against the 7 valid values of zen-ai `ThinkingLevel`
/// （off/minimal/low/medium/high/xhigh/max）
///
/// # Errors
/// Invalid value → Err (message containing the list of valid values).
///
/// # Consumers
/// - This file's `#[arg(value_parser = parse_thinking_level)]` (--thinking flag)
/// - T-AP-08-3 (main.rs: --model "id:thinking" shorthand parsing, reused via `cli::parse_thinking_level`)
pub fn parse_thinking_level(s: &str) -> Result<ThinkingLevel, String> {
    const VALID: &str = "off, minimal, low, medium, high, xhigh, max";
    match s {
        "off" => Ok(ThinkingLevel::Off),
        "minimal" => Ok(ThinkingLevel::Minimal),
        "low" => Ok(ThinkingLevel::Low),
        "medium" => Ok(ThinkingLevel::Medium),
        "high" => Ok(ThinkingLevel::High),
        "xhigh" => Ok(ThinkingLevel::XHigh),
        "max" => Ok(ThinkingLevel::Max),
        _ => Err(format!(
            "invalid thinking level \"{s}\". Valid values: {VALID}"
        )),
    }
}

/// Single-value sanitization for comma-separated lists: trim leading/trailing whitespace; empty values error.
fn parse_trimmed(s: &str) -> Result<String, String> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        Err("empty value".to_string())
    } else {
        Ok(trimmed.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::error::ErrorKind;
    use std::path::Path;

    /// Test helper: pass an argument string (without program name) to clap for parsing.
    fn parse(args: &[&str]) -> Result<CliArgs, clap::Error> {
        let mut full: Vec<&str> = Vec::with_capacity(args.len() + 1);
        full.push("zen");
        full.extend_from_slice(args);
        CliArgs::try_parse_from(full)
    }

    // ── Mode values ───────────────────────────────────────────

    #[test]
    fn test_mode_accepts_text_and_json() {
        assert_eq!(parse(&["--mode", "text"]).unwrap().mode, Some(Mode::Text));
        assert_eq!(parse(&["--mode", "json"]).unwrap().mode, Some(Mode::Json));
        // Unspecified → None (default text output determined by print mode)
        assert_eq!(parse(&[]).unwrap().mode, None);
    }

    #[test]
    fn test_mode_rejects_rpc_and_case() {
        // rpc removed by simplification per REQ-APP-08; TEXT casing invalid (lowercase only)
        assert!(parse(&["--mode", "rpc"]).is_err());
        assert!(parse(&["--mode", "TEXT"]).is_err());
    }

    // ── thinking levels ───────────────────────────────────────

    #[test]
    fn test_thinking_accepts_all_levels() {
        for (cli, expected) in [
            ("off", ThinkingLevel::Off),
            ("minimal", ThinkingLevel::Minimal),
            ("low", ThinkingLevel::Low),
            ("medium", ThinkingLevel::Medium),
            ("high", ThinkingLevel::High),
            ("xhigh", ThinkingLevel::XHigh),
            ("max", ThinkingLevel::Max),
        ] {
            let parsed = parse(&["--thinking", cli]).unwrap();
            assert_eq!(parsed.thinking, Some(expected), "thinking level {cli}");
        }
    }

    #[test]
    fn test_thinking_invalid_rejected() {
        // pi only warns and continues on invalid values; clap value_parser hard-errors (behavior difference, see docs)
        let err = parse(&["--thinking", "Off"]).unwrap_err();
        assert!(
            err.to_string().contains("off, minimal"),
            "Error message should include the list of valid values: {err}"
        );
        assert!(parse(&["--thinking", "invalid"]).is_err());
    }

    // ── Positional argument classification (messages / @file) ─

    #[test]
    fn test_positional_messages_and_files_interleaved() {
        let parsed = parse(&["-p", "@a.md", "msg1", "@b.md", "msg2"]).unwrap();
        assert_eq!(parsed.messages(), vec!["msg1", "msg2"]);
        assert_eq!(parsed.file_args(), vec!["a.md", "b.md"]);
        // Pure @file arguments also all go to file_args (strip @ prefix)
        let parsed = parse(&["@prompt.md"]).unwrap();
        assert!(parsed.messages().is_empty());
        assert_eq!(parsed.file_args(), vec!["prompt.md"]);
    }

    // ── Short flags ───────────────────────────────────────────

    #[test]
    fn test_short_flags() {
        let parsed = parse(&["-p", "-c", "-r", "-n", "sess", "-v"]).unwrap();
        assert!(parsed.print);
        assert!(parsed.continue_flag);
        assert!(parsed.resume);
        assert_eq!(parsed.name.as_deref(), Some("sess"));
        assert!(parsed.version);
    }

    #[test]
    fn test_continue_long_form() {
        assert!(parse(&["--continue"]).unwrap().continue_flag);
    }

    #[test]
    fn test_multi_char_short_parses_as_name_value() {
        // pi's `-nt` (--no-tools alias) is unsupported by clap: `-nt` is parsed as `-n t` (name="t").
        // Correct usage is the long flag `--no-tools`. This test locks in that behavior to prevent misuse regressions.
        let parsed = parse(&["-nt"]).unwrap();
        assert_eq!(parsed.name.as_deref(), Some("t"));
        assert!(!parsed.no_tools);
        assert!(parse(&["--no-tools"]).unwrap().no_tools);
    }

    // ── Comma-separated lists ─────────────────────────────────

    #[test]
    fn test_tools_comma_delimited_and_trimmed() {
        let parsed = parse(&["--tools", "read,grep"]).unwrap();
        assert_eq!(parsed.tools, vec!["read", "grep"]);
        // value_delimiter splits on commas first, then trims each item
        let parsed = parse(&["--tools", " a , b "]).unwrap();
        assert_eq!(parsed.tools, vec!["a", "b"]);
        // -t can be repeated (Vec defaults to Append)
        let parsed = parse(&["-t", "read", "-t", "grep"]).unwrap();
        assert_eq!(parsed.tools, vec!["read", "grep"]);
    }

    #[test]
    fn test_tools_empty_value_rejected() {
        // Matches pi's filter(non-empty): empty items error directly in zen (pi silently drops them)
        assert!(parse(&["--tools", ""]).is_err());
        assert!(parse(&["--tools", "a,,b"]).is_err());
    }

    #[test]
    fn test_models_comma_delimited() {
        let parsed = parse(&["--models", "claude-sonnet,claude-haiku"]).unwrap();
        assert_eq!(parsed.models, vec!["claude-sonnet", "claude-haiku"]);
    }

    // ── list-models tri-state ─────────────────────────────────

    #[test]
    fn test_list_models_three_states() {
        assert_eq!(parse(&[]).unwrap().list_models, None);
        assert_eq!(parse(&["--list-models"]).unwrap().list_models, Some(None));
        assert_eq!(
            parse(&["--list-models", "deepseek"]).unwrap().list_models,
            Some(Some("deepseek".to_string()))
        );
    }

    // ── Repeatable flags ──────────────────────────────────────

    #[test]
    fn test_append_system_prompt_repeatable() {
        let parsed =
            parse(&["--append-system-prompt", "a", "--append-system-prompt", "b"]).unwrap();
        assert_eq!(parsed.append_system_prompt, vec!["a", "b"]);
    }

    #[test]
    fn test_repeatable_resource_flags_use_singular_long_names() {
        // pi's flags are singular --skill/--theme/--prompt-template, unlike the plural field names
        let parsed = parse(&[
            "--skill",
            "s1",
            "--skill",
            "s2",
            "--theme",
            "t1",
            "--prompt-template",
            "p1",
        ])
        .unwrap();
        assert_eq!(parsed.skills, vec!["s1", "s2"]);
        assert_eq!(parsed.themes, vec!["t1"]);
        assert_eq!(parsed.prompt_templates, vec!["p1"]);
        // Plural form --skills is not accepted
        assert!(parse(&["--skills", "s1"]).is_err());
    }

    // ── Session-related ───────────────────────────────────────

    #[test]
    fn test_session_flags() {
        let parsed = parse(&[
            "--session-id",
            "abc",
            "--fork",
            "x",
            "--session-dir",
            "/tmp/s",
            "--no-session",
        ])
        .unwrap();
        assert_eq!(parsed.session_id.as_deref(), Some("abc"));
        assert_eq!(parsed.fork.as_deref(), Some("x"));
        assert_eq!(parsed.session_dir.as_deref(), Some(Path::new("/tmp/s")));
        assert!(parsed.no_session);
    }

    #[test]
    fn test_export_flag() {
        let parsed = parse(&["--export", "out.html"]).unwrap();
        assert_eq!(parsed.export.as_deref(), Some(Path::new("out.html")));
    }

    // ── provider / model / api-key ─────────────────────────────

    #[test]
    fn test_provider_model_api_key() {
        let parsed = parse(&[
            "--provider",
            "huoshan",
            "--model",
            "deepseek-v4-flash",
            "--api-key",
            "k",
        ])
        .unwrap();
        assert_eq!(parsed.provider.as_deref(), Some("huoshan"));
        assert_eq!(parsed.model.as_deref(), Some("deepseek-v4-flash"));
        assert_eq!(parsed.api_key.as_deref(), Some("k"));
    }

    #[test]
    fn test_system_prompt_flags() {
        let parsed = parse(&["--system-prompt", "sp", "--append-system-prompt", "a"]).unwrap();
        assert_eq!(parsed.system_prompt.as_deref(), Some("sp"));
        assert_eq!(parsed.append_system_prompt, vec!["a"]);
    }

    // ── bool switch group ─────────────────────────────────────

    #[test]
    fn test_boolean_flags() {
        let parsed = parse(&[
            "--no-tools",
            "--no-builtin-tools",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
            "--no-context-files",
            "--offline",
            "--verbose",
        ])
        .unwrap();
        assert!(parsed.no_tools);
        assert!(parsed.no_builtin_tools);
        assert!(parsed.no_skills);
        assert!(parsed.no_prompt_templates);
        assert!(parsed.no_themes);
        assert!(parsed.no_context_files);
        assert!(parsed.offline);
        assert!(parsed.verbose);
    }

    // ── print forms ───────────────────────────────────────────

    #[test]
    fn test_print_with_and_without_message() {
        let parsed = parse(&["-p", "hello"]).unwrap();
        assert!(parsed.print);
        assert_eq!(parsed.messages(), vec!["hello"]);
        // Bare -p: no message (T-AP-08-2 uses stdin merge)
        let parsed = parse(&["-p"]).unwrap();
        assert!(parsed.print);
        assert!(parsed.positional.is_empty());
    }

    // ── Version / help / unknown flags ────────────────────────

    #[test]
    fn test_version_flag() {
        assert!(parse(&["-v"]).unwrap().version);
        assert!(parse(&["--version"]).unwrap().version);
    }

    #[test]
    fn test_help_flag_exits_with_help() {
        // clap auto-generates -h/--help: try_parse_from returns DisplayHelp error
        let err = parse(&["-h"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::DisplayHelp);
        assert_eq!(
            parse(&["--help"]).unwrap_err().kind(),
            ErrorKind::DisplayHelp
        );
    }

    #[test]
    fn test_unknown_flag_rejected() {
        // v1 has no extension system: unknown long flags are no longer collected into unknownFlags like pi; they error directly
        assert!(parse(&["--bogus"]).is_err());
        assert!(parse(&["-x"]).is_err());
    }
}
