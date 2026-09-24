use crate::resources::ResourceLoadDiagnostic;
use crate::resources::ThemeLoadDiagnostic;
use crate::skills::SkillLoadDiagnostic;
use crossterm::style::{Color, ResetColor, SetForegroundColor};

pub enum DiagnosticSeverity {
    Info,
    Warning,
    Error,
}

pub struct RuntimeDiagnostic {
    severity: DiagnosticSeverity, // Determines output color and prefix
    message: String,              // Human-readable single-line message (to_runtime_diagnostics guarantees one line, no embedded newlines)
}

// File: crates/zen-app/src/observability/diagnostics.rs
/// Structured formatting of resource diagnostics — on interactive startup, converts skill/template/theme load warnings and same-name collisions collected during assembly
/// into line-based text that shows at a glance "who wins and who is dropped" (same collision grouping layout as pi `formatDiagnostics`),
/// Users no longer face silently swallowed errors: same-name resource conflicts are grouped into blocks by name first-occurrence order, each block has a line `  "name" collision:` followed by
/// `    ✓ winner_path` and each `    ✗ loser_path (skipped)` (✓/✗ lines indented 4 spaces, aligned with pi); non-collision diagnostics with path → two lines
/// `  path` + `    message`; no path → one line `  message`.
///
/// # Parameters
/// - `diagnostics`: unified resource diagnostic list (`resources::ResourceLoadDiagnostic`, including Skill/Prompt/
///   PromptCollision/Theme four variants); collisions and warnings interleaved, output stably in occurrence order.
///
/// # Returns
/// Display-line list (one String per line, no ANSI colors — color is decided by the display carrier: print side uses fixed ANSI,
/// interactive side draws directly with the `StatusSpec::Warning` semantic color and sanitizes via `strip_ansi`); empty list = no diagnostics.
///
/// # Bounds
/// - Collision grouping key is name (grouped uniformly by name across resource types; pi groups by per-category lists, zen's lists are already unified,
///   same name across types (e.g. a skill and prompt with the same name) merges into one group — v1 has no such conflict scenario, registered as an accepted deviation);
/// - Returns **plain text** (`✓`/`✗` are literal markers, not coloring), ensuring interactive status lines do not go through `strip_ansi`
///   lose structure; segment-level coloring of ✓ success color / ✗ warning color is a pi rich-display detail that the v1 status-line carrier cannot express, so it is omitted.
///
/// # Consumers
/// `modes/interactive/mod.rs` (init resource diagnostic warning stream, wired by this task); print side via
/// `to_runtime_diagnostics` is flattened and then passed to `report_runtime_diagnostics`.
pub fn format_resource_diagnostics(diagnostics: &[ResourceLoadDiagnostic]) -> Vec<String> {
    let (groups, others) = partition(diagnostics);
    render(&groups, &others)
}

/// Normalized grouping of collision payloads — name first-occurrence order; winner is the first item in the group (pi `collisionList[0]` semantics,
/// same-name first occurrence is kept), each collision contributes one loser.
struct CollisionGroup<'a> {
    name: &'a str,
    winner_path: &'a str,
    losers: Vec<&'a str>,
}

/// Normalized view of non-collision diagnostics — in v1 each warning payload always carries path (`Some`);
/// `Option` carries the shape of pi's "no path → single-line message" branch (defensive against future payload variants).
struct OtherDiagnostic<'a> {
    path: Option<&'a str>,
    message: &'a str,
}

/// One exhaustive classification: collisions grouped by name (same name across resource types merged — documented accepted deviation),
/// the rest collected in order. When a new `ResourceLoadDiagnostic` variant is added, the compiler forces adding a branch.
fn partition(
    diagnostics: &[ResourceLoadDiagnostic],
) -> (Vec<CollisionGroup<'_>>, Vec<OtherDiagnostic<'_>>) {
    let mut groups: Vec<CollisionGroup> = Vec::new();
    let mut others: Vec<OtherDiagnostic> = Vec::new();
    for d in diagnostics {
        match d {
            ResourceLoadDiagnostic::Skill(SkillLoadDiagnostic::Collision(c)) => {
                push_collision(&mut groups, &c.name, &c.winner_path, &c.loser_path);
            }
            ResourceLoadDiagnostic::PromptCollision(c) => {
                push_collision(&mut groups, &c.name, &c.winner_path, &c.loser_path);
            }
            ResourceLoadDiagnostic::Theme(ThemeLoadDiagnostic::Collision(c)) => {
                push_collision(&mut groups, &c.name, &c.winner_path, &c.loser_path);
            }
            ResourceLoadDiagnostic::Skill(SkillLoadDiagnostic::Warning { message, path }) => {
                others.push(OtherDiagnostic {
                    path: Some(path),
                    message,
                });
            }
            ResourceLoadDiagnostic::Prompt(d) => {
                others.push(OtherDiagnostic {
                    path: Some(&d.path),
                    message: &d.message,
                });
            }
            ResourceLoadDiagnostic::Theme(ThemeLoadDiagnostic::Warning { message, path }) => {
                others.push(OtherDiagnostic {
                    path: Some(path),
                    message,
                });
            }
        }
    }
    (groups, others)
}

/// Add collision to group: if a same-name group is hit, only append loser (group winner stays the first item); otherwise create a new one at the group end (first-occurrence order).
fn push_collision<'a>(
    groups: &mut Vec<CollisionGroup<'a>>,
    name: &'a str,
    winner_path: &'a str,
    loser_path: &'a str,
) {
    match groups.iter_mut().find(|g| g.name == name) {
        Some(group) => group.losers.push(loser_path),
        None => groups.push(CollisionGroup {
            name,
            winner_path,
            losers: vec![loser_path],
        }),
    }
}

/// Two-section rendering (aligned with pi `formatDiagnostics` layout): all collision blocks come first by name first-occurrence order —
/// `  "name" collision:` header line + `    ✓ winner` (first item in group) + each `    ✗ loser (skipped)`;
/// Non-collisions by occurrence order — with path two lines `  path` + `    message`, without path one line `  message`.
/// Message text is folded via `single_line`, ensuring "one String per line" is not broken by embedded newlines.
fn render(groups: &[CollisionGroup<'_>], others: &[OtherDiagnostic<'_>]) -> Vec<String> {
    let mut lines = Vec::new();
    for group in groups {
        lines.push(format!("  \"{}\" collision:", group.name));
        lines.push(format!("    ✓ {}", group.winner_path));
        for loser in &group.losers {
            lines.push(format!("    ✗ {loser} (skipped)"));
        }
    }
    for other in others {
        match other.path {
            Some(path) => {
                lines.push(format!("  {path}"));
                lines.push(format!("    {}", single_line(other.message.to_string())));
            }
            None => lines.push(format!("  {}", single_line(other.message.to_string()))),
        }
    }
    lines
}

// File: crates/zen-app/src/observability/diagnostics.rs
/// Flatten structured resource diagnostics into runtime diagnostics — on print-mode startup, compress the collision/warning list into individually outputtable
/// single-line warnings (the flat form of pi `runtime.diagnostics`), so stderr reporting does not need to know the collision grouping structure:
/// Each `ResourceLoadDiagnostic` → one `RuntimeDiagnostic::Warning`, with a single-line message —
/// Collision looks like `"{name}" collision: {winner_path} (kept), {loser_path} (skipped)`,
/// Warning looks like `{message} ({path})` (path always has a value, all v1 diagnostic types carry it; the no-path branch is defensive).
///
/// # Parameters
/// - `diagnostics`: unified resource diagnostic list (same input as `format_resource_diagnostics`).
///
/// # Returns
/// Flat runtime diagnostics; order matches input, all `Warning` level (v1 resource diagnostics have no info/error variants).
///
/// # Consumers
/// `main.rs` (fed to `report_runtime_diagnostics` after print-branch assembly).
pub fn to_runtime_diagnostics(diagnostics: &[ResourceLoadDiagnostic]) -> Vec<RuntimeDiagnostic> {
    diagnostics
        .iter()
        .map(|d| RuntimeDiagnostic {
            severity: DiagnosticSeverity::Warning,
            message: flatten_diagnostic_message(d),
        })
        .collect()
}

/// Single resource diagnostic → single-line message (the message-assembly leg of [`to_runtime_diagnostics`]).
/// Collision payloads (SkillCollision/PromptCollision/ThemeCollision have isomorphic fields) uniformly go through
/// [`collision_message`]; warning payloads (including the struct form of `PromptTemplateDiagnostic`)
/// uniformly go through [`warning_message`]. match is exhaustive: adding a new resource diagnostic variant forces the compiler to require a branch.
fn flatten_diagnostic_message(diagnostic: &ResourceLoadDiagnostic) -> String {
    match diagnostic {
        ResourceLoadDiagnostic::Skill(d) => match d {
            SkillLoadDiagnostic::Warning { message, path } => warning_message(message, path),
            SkillLoadDiagnostic::Collision(c) => {
                collision_message(&c.name, &c.winner_path, &c.loser_path)
            }
        },
        ResourceLoadDiagnostic::Prompt(d) => warning_message(&d.message, &d.path),
        ResourceLoadDiagnostic::PromptCollision(c) => {
            collision_message(&c.name, &c.winner_path, &c.loser_path)
        }
        ResourceLoadDiagnostic::Theme(d) => match d {
            ThemeLoadDiagnostic::Warning { message, path } => warning_message(message, path),
            ThemeLoadDiagnostic::Collision(c) => {
                collision_message(&c.name, &c.winner_path, &c.loser_path)
            }
        },
    }
}

/// Warning form: `{message} ({path})` — all v1 diagnostic types carry path
/// (`PromptTemplateDiagnostic` uses struct fields, Skill/Theme use enum named fields).
fn warning_message(message: &str, path: &str) -> String {
    single_line(format!("{message} ({path})"))
}

/// Collision form: `"{name}" collision: {winner_path} (kept), {loser_path} (skipped)` —
/// Text markers align with the `✓ kept / ✗ skipped` semantics of [`format_resource_diagnostics`].
fn collision_message(name: &str, winner_path: &str, loser_path: &str) -> String {
    single_line(format!(
        "\"{name}\" collision: {winner_path} (kept), {loser_path} (skipped)"
    ))
}

/// Single-line guarantee (`RuntimeDiagnostic.message` field contract): diagnostic message sources (io error text,
/// frontmatter parse errors, etc.) may theoretically contain newlines; fold them to spaces to prevent the "exactly one line per entry"
/// stderr layout from being broken by embedded newlines; the normal path has no newlines, so probe first to avoid unnecessary allocation.
fn single_line(message: String) -> String {
    if message.contains(['\r', '\n']) {
        message.replace(['\r', '\n'], " ")
    } else {
        message
    }
}

// File: crates/zen-app/src/observability/diagnostics.rs
/// stderr output for runtime diagnostics — after print-mode startup assembly, write flat diagnostics one by one to stderr (aligning with pi
/// `reportDiagnostics`'s `console.error` semantics), so users can see setup/resource issues before model calls:
/// One line per entry: error level in red with `Error: ` prefix, warning level in yellow with `Warning: ` prefix, info level dim with no prefix;
/// Colors use fixed ANSI (crossterm `SetForegroundColor(Color::{Red, Yellow, DarkGrey})` +
/// `ResetColor`), independent of theme (aligning with pi chalk constants).
///
/// # Parameters
/// - `diagnostics`: flat runtime diagnostics (`to_runtime_diagnostics` output; empty list = no output).
/// - `w`: output writer (caller passes `std::io::stderr()`; parameterized for unit testing).
///
/// # Returns
/// `std::io::Result<()>` (write failures are propagated, not swallowed).
///
/// # Bounds
/// - A single message does **not** embed newlines (guaranteed by `to_runtime_diagnostics`); each entry is exactly one line;
/// - Whether there is an error-level diagnostic is determined by the caller with `iter().any(|d| d.severity == DiagnosticSeverity::Error)`
///   (the exit(1) branch in pi main.ts); this function does not decide whether to exit.
///
/// # Consumers
/// `main.rs` (print branch).
pub fn report_runtime_diagnostics(
    diagnostics: &[RuntimeDiagnostic],
    w: &mut dyn std::io::Write,
) -> std::io::Result<()> {
    for diagnostic in diagnostics {
        let (color, prefix) = match diagnostic.severity {
            DiagnosticSeverity::Error => (Color::Red, "Error: "),
            DiagnosticSeverity::Warning => (Color::Yellow, "Warning: "),
            DiagnosticSeverity::Info => (Color::DarkGrey, ""),
        };
        writeln!(
            w,
            "{}{prefix}{}{}",
            SetForegroundColor(color),
            diagnostic.message,
            ResetColor
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::themes::ThemeCollision;
    use crate::resources::PromptCollision;
    use crate::skills::SkillCollision;
    use zen_core::prompt_templates::{PromptTemplateDiagnostic, PromptTemplateDiagnosticCode};

    // ── Fixtures: four variants × (collision/warning) ────────────────────────────

    fn skill_collision(name: &str, winner: &str, loser: &str) -> ResourceLoadDiagnostic {
        ResourceLoadDiagnostic::Skill(SkillLoadDiagnostic::Collision(SkillCollision {
            name: name.to_string(),
            winner_path: winner.to_string(),
            loser_path: loser.to_string(),
        }))
    }

    fn skill_warning(message: &str, path: &str) -> ResourceLoadDiagnostic {
        ResourceLoadDiagnostic::Skill(SkillLoadDiagnostic::Warning {
            message: message.to_string(),
            path: path.to_string(),
        })
    }

    fn prompt_collision(name: &str, winner: &str, loser: &str) -> ResourceLoadDiagnostic {
        ResourceLoadDiagnostic::PromptCollision(PromptCollision {
            name: name.to_string(),
            winner_path: winner.to_string(),
            loser_path: loser.to_string(),
        })
    }

    fn prompt_warning(message: &str, path: &str) -> ResourceLoadDiagnostic {
        ResourceLoadDiagnostic::Prompt(PromptTemplateDiagnostic {
            code: PromptTemplateDiagnosticCode::ReadFailed,
            message: message.to_string(),
            path: path.to_string(),
        })
    }

    fn theme_collision(name: &str, winner: &str, loser: &str) -> ResourceLoadDiagnostic {
        ResourceLoadDiagnostic::Theme(ThemeLoadDiagnostic::Collision(ThemeCollision {
            name: name.to_string(),
            winner_path: winner.to_string(),
            loser_path: loser.to_string(),
        }))
    }

    fn theme_warning(message: &str, path: &str) -> ResourceLoadDiagnostic {
        ResourceLoadDiagnostic::Theme(ThemeLoadDiagnostic::Warning {
            message: message.to_string(),
            path: path.to_string(),
        })
    }

    // ── format_resource_diagnostics ─────────────────────────────

    #[test]
    fn test_format_resource_diagnostics_empty_input() {
        assert!(format_resource_diagnostics(&[]).is_empty());
    }

    #[test]
    fn test_format_resource_diagnostics_groups_collisions_by_name() {
        let lines = format_resource_diagnostics(&[
            skill_collision("refactor", "/a/SKILL.md", "/b/SKILL.md"),
            theme_collision("dark", "/t1.json", "/t2.json"),
        ]);
        assert_eq!(
            lines,
            vec![
                "  \"refactor\" collision:",
                "    ✓ /a/SKILL.md",
                "    ✗ /b/SKILL.md (skipped)",
                "  \"dark\" collision:",
                "    ✓ /t1.json",
                "    ✗ /t2.json (skipped)",
            ]
        );
    }

    #[test]
    fn test_format_resource_diagnostics_same_name_across_types_merged() {
        // Same names across resource types are merged into one group: the winner is the first entry, and the two losers are appended in occurrence order
        let lines = format_resource_diagnostics(&[
            skill_collision("clash", "/s/SKILL.md", "/s2/SKILL.md"),
            prompt_collision("clash", "/p1.md", "/p2.md"),
        ]);
        assert_eq!(
            lines,
            vec![
                "  \"clash\" collision:",
                "    ✓ /s/SKILL.md",
                "    ✗ /s2/SKILL.md (skipped)",
                "    ✗ /p2.md (skipped)",
            ]
        );
    }

    #[test]
    fn test_format_resource_diagnostics_multi_loser_same_name() {
        let lines = format_resource_diagnostics(&[
            skill_collision("x", "/w1", "/l1"),
            skill_collision("x", "/w2", "/l2"),
            skill_collision("x", "/w3", "/l3"),
        ]);
        assert_eq!(
            lines,
            vec![
                "  \"x\" collision:",
                "    ✓ /w1",
                "    ✗ /l1 (skipped)",
                "    ✗ /l2 (skipped)",
                "    ✗ /l3 (skipped)",
            ]
        );
    }

    #[test]
    fn test_format_resource_diagnostics_warning_with_path_two_lines() {
        let lines =
            format_resource_diagnostics(&[skill_warning("failed to read skill", "/s/SKILL.md")]);
        assert_eq!(lines, vec!["  /s/SKILL.md", "    failed to read skill"]);
    }

    #[test]
    fn test_format_resource_diagnostics_warning_message_folds_newlines() {
        let lines =
            format_resource_diagnostics(&[prompt_warning("bad frontmatter\nline2", "/p.md")]);
        assert_eq!(lines, vec!["  /p.md", "    bad frontmatter line2"]);
    }

    #[test]
    fn test_format_resource_diagnostics_collisions_before_others_keep_order() {
        // Collision blocks (by first occurrence of name) always come first, non-collisions follow in occurrence order; interleaving does not affect group order
        let lines = format_resource_diagnostics(&[
            skill_collision("a", "/w1", "/l1"),
            theme_warning("theme warn", "/t.json"),
            prompt_collision("b", "/w2", "/l2"),
            skill_warning("skill warn", "/s/SKILL.md"),
        ]);
        assert_eq!(
            lines,
            vec![
                "  \"a\" collision:",
                "    ✓ /w1",
                "    ✗ /l1 (skipped)",
                "  \"b\" collision:",
                "    ✓ /w2",
                "    ✗ /l2 (skipped)",
                "  /t.json",
                "    theme warn",
                "  /s/SKILL.md",
                "    skill warn",
            ]
        );
    }

    // ── to_runtime_diagnostics ──────────────────────────────────

    #[test]
    fn test_to_runtime_diagnostics_empty_input() {
        assert!(to_runtime_diagnostics(&[]).is_empty());
    }

    #[test]
    fn test_to_runtime_diagnostics_all_warning_severity() {
        let diags = to_runtime_diagnostics(&[
            skill_warning("w1", "/s"),
            skill_collision("c1", "/w", "/l"),
            prompt_warning("w2", "/p"),
            prompt_collision("c2", "/w", "/l"),
            theme_warning("w3", "/t"),
            theme_collision("c3", "/w", "/l"),
        ]);
        assert_eq!(diags.len(), 6);
        for d in &diags {
            assert!(matches!(d.severity, DiagnosticSeverity::Warning));
        }
    }

    #[test]
    fn test_to_runtime_diagnostics_message_shapes() {
        let diags = to_runtime_diagnostics(&[
            skill_warning("bad skill", "/s/SKILL.md"),
            skill_collision("refactor", "/a/SKILL.md", "/b/SKILL.md"),
            theme_collision("dark", "/t1.json", "/t2.json"),
        ]);
        assert_eq!(diags[0].message, "bad skill (/s/SKILL.md)");
        assert_eq!(
            diags[1].message,
            "\"refactor\" collision: /a/SKILL.md (kept), /b/SKILL.md (skipped)"
        );
        assert_eq!(
            diags[2].message,
            "\"dark\" collision: /t1.json (kept), /t2.json (skipped)"
        );
    }

    #[test]
    fn test_to_runtime_diagnostics_preserves_order_and_folds_newlines() {
        let diags = to_runtime_diagnostics(&[
            theme_warning("first", "/t1.json"),
            prompt_warning("failed\nagain", "/p.md"),
        ]);
        assert_eq!(diags.len(), 2);
        assert_eq!(diags[0].message, "first (/t1.json)");
        assert_eq!(diags[1].message, "failed again (/p.md)");
    }

    // ── report_runtime_diagnostics ──────────────────────────────

    fn rt(severity: DiagnosticSeverity, message: &str) -> RuntimeDiagnostic {
        RuntimeDiagnostic {
            severity,
            message: message.to_string(),
        }
    }

    #[test]
    fn test_report_runtime_diagnostics_empty_input() {
        let mut out = Vec::new();
        report_runtime_diagnostics(&[], &mut out).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_report_runtime_diagnostics_warning_yellow_prefixed() {
        let mut out = Vec::new();
        report_runtime_diagnostics(
            &[rt(DiagnosticSeverity::Warning, "resource issue")],
            &mut out,
        )
        .unwrap();
        // This crossterm version maps Color::Yellow to 256-color code 38;5;11 (not basic color 33)
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "\u{1b}[38;5;11mWarning: resource issue\u{1b}[0m\n"
        );
    }

    #[test]
    fn test_report_runtime_diagnostics_error_red_prefixed() {
        let mut out = Vec::new();
        report_runtime_diagnostics(&[rt(DiagnosticSeverity::Error, "fatal")], &mut out).unwrap();
        // This crossterm version maps Color::Red to 256-color code 38;5;9 (not basic color 31)
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "\u{1b}[38;5;9mError: fatal\u{1b}[0m\n"
        );
    }

    #[test]
    fn test_report_runtime_diagnostics_info_dim_no_prefix() {
        let mut out = Vec::new();
        report_runtime_diagnostics(&[rt(DiagnosticSeverity::Info, "note")], &mut out).unwrap();
        // This crossterm version maps Color::DarkGrey to 256-color code 38;5;8 (not basic color 90)
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "\u{1b}[38;5;8mnote\u{1b}[0m\n"
        );
    }

    #[test]
    fn test_report_runtime_diagnostics_multiple_one_line_each() {
        let mut out = Vec::new();
        report_runtime_diagnostics(
            &[
                rt(DiagnosticSeverity::Warning, "w"),
                rt(DiagnosticSeverity::Info, "i"),
                rt(DiagnosticSeverity::Error, "e"),
            ],
            &mut out,
        )
        .unwrap();
        let s = String::from_utf8(out).unwrap();
        assert_eq!(s.lines().count(), 3);
        assert!(s.contains("Warning: w"));
        assert!(s.contains("i"));
        assert!(s.contains("Error: e"));
    }

    // ── Private helpers: single_line / warning_message / collision_message ──

    #[test]
    fn test_single_line_passthrough_without_newline() {
        assert_eq!(single_line("plain message".to_string()), "plain message");
    }

    #[test]
    fn test_single_line_folds_lf() {
        assert_eq!(single_line("a\nb".to_string()), "a b");
    }

    #[test]
    fn test_single_line_folds_cr() {
        assert_eq!(single_line("a\rb".to_string()), "a b");
    }

    #[test]
    fn test_single_line_crlf_characterizes_double_space() {
        // Known behavior: replace(['\r','\n'], " ") replaces CR and LF separately → double space.
        // Deviates from the doc contract's single-space intent of "fold to spaces" (implementation flaw); lock the behavior to prevent regression,
        // Fix is left to the implementer (if changed: first fold CRLF to LF or split by line and join).
        assert_eq!(single_line("a\r\nb".to_string()), "a  b");
    }

    #[test]
    fn test_single_line_folds_multiple_lines_to_spaces() {
        assert_eq!(single_line("l1\nl2\nl3".to_string()), "l1 l2 l3");
    }

    #[test]
    fn test_warning_message_appends_path_and_folds() {
        assert_eq!(warning_message("bad file", "/x/y.md"), "bad file (/x/y.md)");
        assert_eq!(warning_message("bad\nfile", "/x"), "bad file (/x)");
    }

    #[test]
    fn test_collision_message_shape() {
        assert_eq!(
            collision_message("refactor", "/a/SKILL.md", "/b/SKILL.md"),
            "\"refactor\" collision: /a/SKILL.md (kept), /b/SKILL.md (skipped)"
        );
    }

    // ── Private helpers: push_collision / partition / render ───────────

    #[test]
    fn test_push_collision_new_group_then_appends_loser_keeping_winner() {
        let mut groups: Vec<CollisionGroup> = Vec::new();
        push_collision(&mut groups, "x", "/w1", "/l1");
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].name, "x");
        assert_eq!(groups[0].winner_path, "/w1");
        assert_eq!(groups[0].losers, vec!["/l1"]);

        push_collision(&mut groups, "x", "/w2", "/l2");
        assert_eq!(groups.len(), 1, "Same-name hit should append loser instead of creating a new group");
        assert_eq!(groups[0].winner_path, "/w1", "Group winner remains first");
        assert_eq!(groups[0].losers, vec!["/l1", "/l2"]);

        push_collision(&mut groups, "y", "/w3", "/l3");
        assert_eq!(groups.len(), 2, "New name should create a new group (first-seen order)");
        assert_eq!(groups[1].name, "y");
    }

    #[test]
    fn test_partition_groups_collisions_and_keeps_others_in_order() {
        // The partition output borrows the input array, so it must be bound with let first to ensure the borrow period (CollisionGroup contains &'a str)
        let diagnostics = [
            skill_warning("w1", "/s1"),
            skill_collision("a", "/w1", "/l1"),
            prompt_collision("a", "/w2", "/l2"),
            theme_warning("w2", "/t1"),
        ];
        let (groups, others) = partition(&diagnostics);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].name, "a");
        assert_eq!(groups[0].winner_path, "/w1");
        assert_eq!(groups[0].losers, vec!["/l1", "/l2"]);
        assert_eq!(others.len(), 2);
        assert_eq!(others[0].message, "w1");
        assert_eq!(others[0].path, Some("/s1"));
        assert_eq!(others[1].message, "w2");
        assert_eq!(others[1].path, Some("/t1"));
    }

    #[test]
    fn test_render_other_without_path_single_line() {
        // partition always produces Some(path) (all v1 warning payloads carry it); the None branch is covered by direct construction
        let lines = render(
            &[],
            &[OtherDiagnostic {
                path: None,
                message: "no path",
            }],
        );
        assert_eq!(lines, vec!["  no path"]);
    }
}
