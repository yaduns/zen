//! Slash command system: registry + parsing + expansion + completion assembly.
//!
//! REQ-APP-03 (pi-308): slash commands (registration + parsing + execution + completion).
//! Command routing/data/expansion belongs to this module (pure functions + static registry); command action execution stays in
//! interactive mode (`modes/interactive/mod.rs`, mode-specific — opening the settings panel/quitting depend on UI state);
//! The expansion engine and resource loading are provided by upstream tasks (T-AG-05-3 zen-core replacement engine, T-AP-03-3 `load_skills`,
//! T-AP-03-4 `expand_prompt_template`); this module only does composition and assembly.
//! Implementation notes in tasks/T-AP-03-2-implemented.md.

use crate::prompt_templates::expand_prompt_template;
use std::path::Path;

use zen_core::prompt_templates::PromptTemplate;
use zen_core::skills::skill::Skill;
use zen_tui::autocomplete::SlashCommand;

/// Static registry entry for a built-in slash command — `name` is the routing key for dispatch and completion;
/// `description`/`argument_hint` serve the completion dropdown and help display.
/// Pure `&'static str` static data (`'static` ensures the registry can be referenced as a constant, zero allocation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuiltinSlashCommand {
    /// Command name (without `/`); `resolve_builtin` matches exactly on it (case-sensitive, aligned with pi `text === "/xxx"`).
    pub name: &'static str,
    /// Main body of the completion dropdown description (aligned with pi BUILTIN_SLASH_COMMANDS copy).
    pub description: &'static str,
    /// Completion parameter hint (joined into the dropdown as `${hint} — ${desc}`); None = no parameter hint.
    pub argument_hint: Option<&'static str>,
}

/// v1 built-in command registry — the **single source of truth** for dispatch and completion (interactive mode no longer hardcodes command strings).
/// 12 entries = the v1 subset of pi's 22; dropped entries (login/logout/trust/share/changelog/
/// scoped-models/export/import/copy/reload, etc.) are not registered, not completed, not recognized by dispatch; for their fate see
/// See the internal design notes for the BUILTIN_SLASH_COMMANDS data model.
pub const BUILTIN_SLASH_COMMANDS: &[BuiltinSlashCommand] = &[
    BuiltinSlashCommand {
        name: "settings",
        description: "Open settings menu",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "model",
        description: "Select model (opens selector UI)",
        argument_hint: Some("<provider/model>"),
    },
    BuiltinSlashCommand {
        name: "name",
        description: "Set session display name",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "session",
        description: "Show session info and stats",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "hotkeys",
        description: "Show all keyboard shortcuts",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "fork",
        description: "Create a new fork from a previous user message",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "clone",
        description: "Duplicate the current session at the current position",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "tree",
        description: "Navigate session tree (switch branches)",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "new",
        description: "Start a new session",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "compact",
        description: "Manually compact the session context",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "resume",
        description: "Resume a different session",
        argument_hint: None,
    },
    BuiltinSlashCommand {
        name: "quit",
        description: "Quit zen",
        argument_hint: None,
    },
];

/// Borrowed-result of `parse_slash_command`: `name` is the command name (without `/`),
/// `args` is the raw argument string after the command name (may be empty); both fields borrow from the input text, zero-copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParsedSlashCommand<'a> {
    /// Non-empty string before the first whitespace (e.g. `/skill:refactor` → `"skill:refactor"`).
    pub name: &'a str,
    /// All remaining text after the first whitespace (preserving internal whitespace/newlines, aligned with pi `text.slice(spaceIndex + 1)`);
    /// empty string if no whitespace.
    pub args: &'a str,
}

/// Slash command system (T-AP-03-2), first step shared by interactive dispatch and session expansion: split `/name args`
/// raw input into (command name, argument string) — "one slash apart, each takes what it needs". Pure function: no registry lookup, no file I/O.
///
/// Behavior (aligned with pi interactive-mode's `text.slice(1)` + first-whitespace split semantics):
/// - Does not start with `/` → None (ordinary message);
/// - Empty after stripping the leading `/` → None (bare `/` or `/ `);
/// - First character is whitespace (e.g. `/ foo`) → None (empty name produces nothing);
/// - Otherwise name = substring before the first whitespace, args = all remaining text after it (empty string if no whitespace).
///
/// # Parameters
/// - `text`: full user input.
///
/// # Returns
/// Borrowed `ParsedSlashCommand`; returns None for non-command input.
pub fn parse_slash_command(text: &str) -> Option<ParsedSlashCommand<'_>> {
    let rest = text.strip_prefix('/')?;
    if rest.is_empty() {
        return None;
    }
    match rest.char_indices().find(|(_, c)| c.is_whitespace()) {
        Some((idx, ch)) => {
            let (name, after) = rest.split_at(idx);
            if name.is_empty() {
                return None; // "/ foo": first character is whitespace, empty name produces nothing
            }
            Some(ParsedSlashCommand {
                name,
                args: &after[ch.len_utf8()..],
            })
        }
        None => Some(ParsedSlashCommand {
            name: rest,
            args: "",
        }),
    }
}

/// Shared by interactive dispatch and completion validation: look up the built-in registry by name — "our command, or a template/skill/outsider".
/// On hit, returns the registry entry (`&'static`; dispatch branches on `name`, completion displays its metadata),
/// on miss returns None (caller falls back to the session expansion path).
///
/// # Parameters
/// - `name`: command name produced by `parse_slash_command` (without `/`, e.g. `"skill:refactor"`).
///
/// # Returns
/// Registry entry reference; returns None if unregistered (no distinction between "template/skill/unknown" — the caller routes all to expansion).
pub fn resolve_builtin(name: &str) -> Option<&'static BuiltinSlashCommand> {
    BUILTIN_SLASH_COMMANDS.iter().find(|c| c.name == name)
}

/// Pre-submit prompt preprocessing for session (sink of seam T-AP-03-4 consumed by T-AP-03-2): expand
/// `/skill:name args` and `/name args` in user input into the final prompt — "skill wrapped as a block, template with params filled,
/// everything else passed through unchanged." Pure function; called by AgentSession's prompt/steer/follow_up before delegating to the engine.
///
/// Behavior (aligned with pi agent-session `_expandSkillCommand` → `expandPromptTemplate` order chain):
/// - Skill expansion first (`/skill:name args`): look up by name in `skills` (case-sensitive); hit → assemble
///   `<skill name="{name}" location="{file_path}">\nReferences are relative to {base_dir}.\n\n{content}\n</skill>`
///   (base_dir = parent dir of file_path; content is zen-core `Skill.content`, already stripped body),
///   append `\n\n{args}` after the block if args present; miss → return input as-is (pi pass-through);
/// - Then template expansion on the result (`expand_prompt_template`, T-AP-03-4): skill blocks never start with `/` so they always pass through,
///   hence equivalent to "template expansion directly on non-skill input"; ordering semantics exactly match pi.
///
/// # Parameters
/// - `text`: full user input;
/// - `skills`: skill set assembled for the session (`AgentSession::skills()`);
/// - `templates`: template set assembled for the session (`AgentSession::templates()`).
///
/// # Returns
/// The expanded prompt text; on no hit, returns input unchanged (`text.to_string()`).
pub fn expand_prompt_input(text: &str, skills: &[Skill], templates: &[PromptTemplate]) -> String {
    let expanded = expand_skill_command(text, skills);
    expand_prompt_template(&expanded, templates)
}

/// `/skill:name args` expansion (private, only called by `expand_prompt_input`) — aligned with pi
/// `_expandSkillCommand`: on skill hit, assemble a single `<skill>` block (distinct from the system prompt `<available_skills>`
/// block in zen-core skills/format.rs; different scenarios so no duplication), on miss/non-skill input return as-is.
fn expand_skill_command(text: &str, skills: &[Skill]) -> String {
    if !text.starts_with("/skill:") {
        return text.to_string();
    }
    let rest = &text["/skill:".len()..];
    // Split on first whitespace (aligned with this module's parse convention; pi uses indexOf(" ") for space only, zen relaxed to any whitespace)
    let (name, args) = match rest.char_indices().find(|(_, c)| c.is_whitespace()) {
        Some((idx, ch)) => (&rest[..idx], &rest[idx + ch.len_utf8()..]),
        None => (rest, ""),
    };
    let args = args.trim();
    let Some(skill) = skills.iter().find(|s| s.name == name) else {
        return text.to_string(); // Unknown skill passes through (pi pass-through)
    };
    let base_dir = Path::new(&skill.file_path)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut block = format!(
        "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{}\n</skill>",
        skill.name,
        skill.file_path,
        base_dir,
        skill.content.trim(), // Explicit .trim() semantics from pi (T-AP-03-5 trims everything; kept here defensively)
    );
    if !args.is_empty() {
        block.push_str("\n\n");
        block.push_str(args);
    }
    block
}

/// Interactive-mode completion assembly (interim): converge built-in commands + templates + skills into a zen-tui completable command list —
/// "One-screen dropdown, all three categories as candidates." Called by interactive init before constructing `CombinedAutocompleteProvider`
/// (T-TU-03-3 declares that feeding the command list belongs to this task).
///
/// Behavior (aligned with pi interactive `createBaseAutocompleteProvider`'s builtin + template + skill three sources):
/// - Order: built-in (`BUILTIN_SLASH_COMMANDS`) → templates → skills;
/// - Templates: name/description/argument_hint come from `PromptTemplate` (no duplicate T-AP-03-4 loading,
///   consumes the list already assembled into the session);
/// - Skills: command name is `skill:{name}` (aligned with pi `skill:${skill.name}`), description from
///   `Skill.description`, no argument_hint; v1 skill commands are always enabled (settings'
///   `enableSkillCommands` was already listed as a dropped v1 field in T-AP-02-3);
/// - All `get_argument_completions = None` (argument completion callbacks only exist for pi model/login; dropped in v1).
///
/// # Parameters
/// - `templates`: assembled templates (`AgentSession::templates()`);
/// - `skills`: assembled skills (`AgentSession::skills()`).
///
/// # Returns
/// `zen_tui::autocomplete::SlashCommand` list (built-in + templates + skills, in order).
pub fn build_slash_command_list(
    templates: &[PromptTemplate],
    skills: &[Skill],
) -> Vec<SlashCommand> {
    let mut commands =
        Vec::with_capacity(BUILTIN_SLASH_COMMANDS.len() + templates.len() + skills.len());
    for builtin in BUILTIN_SLASH_COMMANDS {
        commands.push(SlashCommand::new(
            builtin.name,
            Some(builtin.description.to_string()),
            builtin.argument_hint.map(|s| s.to_string()),
            None,
        ));
    }
    for template in templates {
        commands.push(SlashCommand::new(
            template.name.clone(),
            Some(template.description.clone()),
            template.argument_hint.clone(),
            None,
        ));
    }
    for skill in skills {
        commands.push(SlashCommand::new(
            format!("skill:{}", skill.name),
            Some(skill.description.clone()),
            None,
            None,
        ));
    }
    commands
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skill(name: &str, file_path: &str, content: &str) -> Skill {
        Skill {
            name: name.to_string(),
            description: format!("desc of {name}"),
            content: content.to_string(),
            file_path: file_path.to_string(),
            disable_model_invocation: false,
        }
    }

    fn template(name: &str, content: &str) -> PromptTemplate {
        PromptTemplate {
            name: name.to_string(),
            description: format!("desc of {name}"),
            argument_hint: Some("hint".to_string()),
            content: content.to_string(),
        }
    }

    // ─── parse_slash_command ─────────────────────────────────────

    #[test]
    fn test_parse_non_slash_is_none() {
        assert_eq!(parse_slash_command("hello"), None);
        assert_eq!(parse_slash_command("  hello"), None);
    }

    #[test]
    fn test_parse_lone_slash_is_none() {
        assert_eq!(parse_slash_command("/"), None);
        assert_eq!(parse_slash_command("/ "), None);
    }

    #[test]
    fn test_parse_leading_space_after_slash_is_none() {
        // "/ foo": empty name produces nothing
        assert_eq!(parse_slash_command("/ foo"), None);
    }

    #[test]
    fn test_parse_name_no_args() {
        let p = parse_slash_command("/quit").unwrap();
        assert_eq!(p.name, "quit");
        assert_eq!(p.args, "");
    }

    #[test]
    fn test_parse_name_with_args() {
        let p = parse_slash_command("/model claude/sonnet").unwrap();
        assert_eq!(p.name, "model");
        assert_eq!(p.args, "claude/sonnet");
    }

    #[test]
    fn test_parse_skill_command_name() {
        let p = parse_slash_command("/skill:refactor foo bar").unwrap();
        assert_eq!(p.name, "skill:refactor");
        assert_eq!(p.args, "foo bar");
    }

    #[test]
    fn test_parse_double_space_keeps_rest_in_args() {
        // args = everything after the first whitespace (including subsequent whitespace), aligned with pi slice(spaceIndex + 1)
        let p = parse_slash_command("/cmd  a  b").unwrap();
        assert_eq!(p.name, "cmd");
        assert_eq!(p.args, " a  b");
    }

    #[test]
    fn test_parse_tab_separator() {
        let p = parse_slash_command("/cmd\targ").unwrap();
        assert_eq!(p.name, "cmd");
        assert_eq!(p.args, "arg");
    }

    // ─── resolve_builtin ─────────────────────────────────────────

    #[test]
    fn test_resolve_builtin_known() {
        let c = resolve_builtin("settings").unwrap();
        assert_eq!(c.name, "settings");
        assert!(c.description.contains("settings"));
    }

    #[test]
    fn test_resolve_builtin_unknown_none() {
        assert_eq!(resolve_builtin("skill:foo"), None);
        assert_eq!(resolve_builtin("bogus"), None);
    }

    #[test]
    fn test_resolve_builtin_case_sensitive() {
        assert_eq!(resolve_builtin("Quit"), None);
        assert_eq!(resolve_builtin("quit").unwrap().name, "quit");
    }

    #[test]
    fn test_registry_no_duplicate_names() {
        // The registry is the single source of truth for dispatch: names must not be duplicated, otherwise resolve_builtin finds the first occurrence and is ambiguous
        let mut names: Vec<&str> = BUILTIN_SLASH_COMMANDS.iter().map(|c| c.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), BUILTIN_SLASH_COMMANDS.len());
    }

    // ─── expand_prompt_input (skill branch) ────────────────────────

    #[test]
    fn test_expand_non_slash_passthrough() {
        let skills = [skill("refactor", "/s/refactor/SKILL.md", "body")];
        let templates = [template("review", "Review $1")];
        assert_eq!(
            expand_prompt_input("just a message", &skills, &templates),
            "just a message"
        );
    }

    #[test]
    fn test_expand_skill_block_format() {
        let skills = [skill("refactor", "/s/refactor/SKILL.md", "do the thing")];
        assert_eq!(
            expand_prompt_input("/skill:refactor", &skills, &[]),
            "<skill name=\"refactor\" location=\"/s/refactor/SKILL.md\">\n\
             References are relative to /s/refactor.\n\n\
             do the thing\n</skill>"
        );
    }

    #[test]
    fn test_expand_skill_with_args_appended() {
        let skills = [skill("refactor", "/s/refactor/SKILL.md", "body")];
        let expanded = expand_prompt_input("/skill:refactor foo bar", &skills, &[]);
        assert!(
            expanded.ends_with("\n</skill>\n\nfoo bar"),
            "got: {expanded}"
        );
    }

    #[test]
    fn test_expand_unknown_skill_passthrough() {
        let skills = [skill("refactor", "/s/refactor/SKILL.md", "body")];
        assert_eq!(
            expand_prompt_input("/skill:nope args", &skills, &[]),
            "/skill:nope args"
        );
    }

    #[test]
    fn test_expand_skill_content_trimmed_inside_block() {
        let skills = [skill("s", "/s/SKILL.md", "  padded body  ")];
        let expanded = expand_prompt_input("/skill:s", &skills, &[]);
        assert!(expanded.contains("\n\npadded body\n</skill>"));
        assert!(!expanded.contains("padded body  \n</skill>"));
    }

    #[test]
    fn test_expand_skill_with_no_parent_dir() {
        let skills = [skill("s", "SKILL.md", "body")];
        let expanded = expand_prompt_input("/skill:s", &skills, &[]);
        assert!(expanded.contains("References are relative to .\n"));
    }

    // ─── expand_prompt_input (template branch) ─────────────────────────

    #[test]
    fn test_expand_template_with_args() {
        let templates = [template("review", "Review $1 for $2")];
        assert_eq!(
            expand_prompt_input("/review pr#1 style", &[], &templates),
            "Review pr#1 for style"
        );
    }

    #[test]
    fn test_expand_unknown_template_passthrough() {
        let templates = [template("review", "Review $1")];
        assert_eq!(
            expand_prompt_input("/missing a", &[], &templates),
            "/missing a"
        );
    }

    #[test]
    fn test_expand_skill_then_template_order() {
        // Skill blocks don't start with /, template expansion always passes through — the ordering chain matches pi
        let skills = [skill("refactor", "/s/refactor/SKILL.md", "body")];
        let templates = [template("refactor", "should not apply")];
        let expanded = expand_prompt_input("/skill:refactor", &skills, &templates);
        assert!(expanded.starts_with("<skill name=\"refactor\""));
        assert!(!expanded.contains("should not apply"));
    }

    // ─── build_slash_command_list ────────────────────────────────

    #[test]
    fn test_command_list_empty_resources_only_builtin() {
        let list = build_slash_command_list(&[], &[]);
        assert_eq!(list.len(), BUILTIN_SLASH_COMMANDS.len());
    }

    #[tokio::test]
    async fn test_command_list_contains_templates_and_skills_via_provider() {
        // SlashCommand fields are private (zen-tui only constructs via new); end-to-end verified through CombinedAutocompleteProvider
        // 's' command-name completion branch (filter_commands → AutocompleteItem.value()):
        // both template and skill command names appear in the candidates, with the skill: prefix.
        use std::path::PathBuf;
        use tokio_util::sync::CancellationToken;
        use zen_tui::autocomplete::{AutocompleteProvider, CombinedAutocompleteProvider};

        let templates = [template("review", "body")];
        let skills = [skill("refactor", "/s/refactor/SKILL.md", "body")];
        let list = build_slash_command_list(&templates, &skills);
        assert_eq!(list.len(), BUILTIN_SLASH_COMMANDS.len() + 2);
        let provider = CombinedAutocompleteProvider::new(list, PathBuf::from("/tmp"), false);

        // prefix /s → candidates like settings/session/skill:refactor
        let sugg = provider
            .get_suggestions(&["/s".to_string()], 0, 2, false, &CancellationToken::new())
            .await
            .expect("/s prefix should match built-in and skill commands");
        let values: Vec<&str> = sugg.items().iter().map(|i| i.value()).collect();
        assert!(values.contains(&"skill:refactor"), "got: {values:?}");
        assert!(values.contains(&"settings"), "got: {values:?}");

        // prefix /r → the review template
        let sugg = provider
            .get_suggestions(&["/r".to_string()], 0, 2, false, &CancellationToken::new())
            .await
            .expect("/r prefix should match template commands");
        let values: Vec<&str> = sugg.items().iter().map(|i| i.value()).collect();
        assert!(values.contains(&"review"), "got: {values:?}");
    }
}
