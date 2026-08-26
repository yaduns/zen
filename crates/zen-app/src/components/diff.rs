// REQ-APP-11 (pi-358): diff — tool-result diff text rendering (line-level coloring + inline word-level diff).
// Corresponds to original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/diff.ts.
// Inline word-level diff uses the workspace's existing `similar` crate TextDiff::from_words (same family as zen-core edit_diff,
// no new dependency); returns styled segment lines (not ANSI strings), written to ScreenBuffer by the caller.

use crossterm::style::Attribute;
use similar::{ChangeTag, TextDiff};
use zen_tui::highlight::StyledSegment;
use zen_tui::screen::Style;
use zen_tui::theme::{ColorToken, Theme};

/// Parses a diff line to extract prefix/line number/content; format `[+-\s]<line number> <content>` (line number may be empty),
/// Returns None on mismatch. Consumed only within the module.
fn parse_diff_line(line: &str) -> Option<(char, String, &str)> {
    let mut chars = line.chars();
    let prefix = chars.next()?;
    if prefix != '+' && prefix != '-' && !prefix.is_whitespace() {
        return None;
    }
    let rest = chars.as_str();
    // `\s*` (line number may be empty) followed by one space, then content
    let num_end = rest
        .char_indices()
        .find(|(_, c)| c.is_whitespace())
        .map(|(i, _)| i)
        .unwrap_or(rest.len());
    let line_num = rest[..num_end].to_string();
    let after = rest[num_end..].trim_start_matches(' ');
    Some((prefix, line_num, after))
}

/// Replaces tabs with 3 spaces (aligned with pi replaceTabs, unifies rendered column width).
fn replace_tabs(text: &str) -> String {
    text.replace('\t', "   ")
}

/// When exactly one deletion and one addition, performs word-level diff on the content and applies inverse video to changed tokens; leading whitespace in the first segment is not highlighted with inverse video (aligned with pi).
/// Returns (deleted line segments, added line segments).
fn render_intra_line_diff(
    old_content: &str,
    new_content: &str,
    removed_style: Style,
    added_style: Style,
) -> (Vec<StyledSegment>, Vec<StyledSegment>) {
    fn inverse_of(style: &Style) -> Style {
        let mut attrs = style.attributes.clone();
        if !attrs.contains(&Attribute::Reverse) {
            attrs.push(Attribute::Reverse);
        }
        Style {
            fg: style.fg,
            bg: style.bg,
            attributes: attrs,
        }
    }

    let old_content = replace_tabs(old_content);
    let new_content = replace_tabs(new_content);

    let diff = TextDiff::from_words(&old_content, &new_content);

    let mut removed: Vec<StyledSegment> = Vec::new();
    let mut added: Vec<StyledSegment> = Vec::new();
    let mut first_removed = true;
    let mut first_added = true;

    for change in diff.iter_all_changes() {
        let value = change.value().to_string();
        match change.tag() {
            ChangeTag::Equal => {
                removed.push(StyledSegment {
                    style: removed_style.clone(),
                    text: value.clone(),
                });
                added.push(StyledSegment {
                    style: added_style.clone(),
                    text: value,
                });
            }
            ChangeTag::Delete => {
                if first_removed {
                    first_removed = false;
                    let leading = value.len() - value.trim_start_matches(' ').len();
                    if leading > 0 {
                        removed.push(StyledSegment {
                            style: removed_style.clone(),
                            text: value[..leading].to_string(),
                        });
                    }
                    let rest = &value[leading..];
                    if !rest.is_empty() {
                        removed.push(StyledSegment {
                            style: inverse_of(&removed_style),
                            text: rest.to_string(),
                        });
                    }
                } else {
                    removed.push(StyledSegment {
                        style: inverse_of(&removed_style),
                        text: value,
                    });
                }
            }
            ChangeTag::Insert => {
                if first_added {
                    first_added = false;
                    let leading = value.len() - value.trim_start_matches(' ').len();
                    if leading > 0 {
                        added.push(StyledSegment {
                            style: added_style.clone(),
                            text: value[..leading].to_string(),
                        });
                    }
                    let rest = &value[leading..];
                    if !rest.is_empty() {
                        added.push(StyledSegment {
                            style: inverse_of(&added_style),
                            text: rest.to_string(),
                        });
                    }
                } else {
                    added.push(StyledSegment {
                        style: inverse_of(&added_style),
                        text: value,
                    });
                }
            }
        }
    }

    (removed, added)
}

/// Renders diff text as styled segment lines with line numbers/colors (for diff display in tool results) —
/// Parses `[+- ]<line number> <content>` lines; deleted lines red/added lines green/context dim;
/// When exactly a deletion/addition pair, performs word-level diff on the content (`similar::TextDiff::from_words`); changed tokens use inverse video.
/// Returns a set of segment lines (not just ANSI strings), written to the buffer by the caller.
///
/// Consumers: T-AP-11-1 (tool-execution diff output); reusable by print mode.
pub fn render_diff(diff_text: &str, theme: &Theme) -> Vec<Vec<StyledSegment>> {
    let removed_style = Style::fg(theme.fg(ColorToken::ToolDiffRemoved));
    let added_style = Style::fg(theme.fg(ColorToken::ToolDiffAdded));
    let context_style = Style::fg(theme.fg(ColorToken::ToolDiffContext));

    if diff_text.is_empty() {
        return Vec::new();
    }
    let lines: Vec<&str> = diff_text.split('\n').collect();
    let mut result: Vec<Vec<StyledSegment>> = Vec::new();

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let Some(parsed) = parse_diff_line(line) else {
            result.push(vec![StyledSegment {
                style: context_style.clone(),
                text: line.to_string(),
            }]);
            i += 1;
            continue;
        };

        if parsed.0 == '-' {
            // Collect consecutive deleted lines
            let mut removed_lines: Vec<(String, &str)> = Vec::new();
            while i < lines.len() {
                let Some(p) = parse_diff_line(lines[i]) else {
                    break;
                };
                if p.0 != '-' {
                    break;
                }
                removed_lines.push((p.1, p.2));
                i += 1;
            }
            // Collect consecutive added lines
            let mut added_lines: Vec<(String, &str)> = Vec::new();
            while i < lines.len() {
                let Some(p) = parse_diff_line(lines[i]) else {
                    break;
                };
                if p.0 != '+' {
                    break;
                }
                added_lines.push((p.1, p.2));
                i += 1;
            }

            if removed_lines.len() == 1 && added_lines.len() == 1 {
                let (removed_num, removed_content) = &removed_lines[0];
                let (added_num, added_content) = &added_lines[0];
                let (rsegs, asegs) = render_intra_line_diff(
                    removed_content,
                    added_content,
                    removed_style.clone(),
                    added_style.clone(),
                );
                let mut rline = vec![StyledSegment {
                    style: removed_style.clone(),
                    text: format!("-{removed_num} "),
                }];
                rline.extend(rsegs);
                let mut aline = vec![StyledSegment {
                    style: added_style.clone(),
                    text: format!("+{added_num} "),
                }];
                aline.extend(asegs);
                result.push(rline);
                result.push(aline);
            } else {
                for (num, content) in &removed_lines {
                    result.push(vec![StyledSegment {
                        style: removed_style.clone(),
                        text: format!("-{num} {}", replace_tabs(content)),
                    }]);
                }
                for (num, content) in &added_lines {
                    result.push(vec![StyledSegment {
                        style: added_style.clone(),
                        text: format!("+{num} {}", replace_tabs(content)),
                    }]);
                }
            }
        } else if parsed.0 == '+' {
            result.push(vec![StyledSegment {
                style: added_style.clone(),
                text: format!("+{} {}", parsed.1, replace_tabs(parsed.2)),
            }]);
            i += 1;
        } else {
            result.push(vec![StyledSegment {
                style: context_style.clone(),
                text: format!(" {} {}", parsed.1, replace_tabs(parsed.2)),
            }]);
            i += 1;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn theme() -> &'static Theme {
        zen_tui::theme::builtin_dark_theme()
    }

    fn plain_text(row: &[StyledSegment]) -> String {
        row.iter().map(|s| s.text.as_str()).collect()
    }

    #[test]
    fn empty_diff_renders_nothing() {
        assert!(render_diff("", theme()).is_empty());
    }

    #[test]
    fn unmatchable_lines_are_context() {
        let rows = render_diff("no prefix here\n=separator=", theme());
        assert_eq!(rows.len(), 2);
        assert_eq!(plain_text(&rows[0]), "no prefix here");
        assert_eq!(plain_text(&rows[1]), "=separator=");
    }

    #[test]
    fn context_lines_keep_prefix_and_number() {
        let rows = render_diff(" 10 keep", theme());
        assert_eq!(plain_text(&rows[0]), " 10 keep");
        assert!(rows[0][0].style.fg.is_some());
    }

    #[test]
    fn standalone_added_line() {
        let rows = render_diff("+3 added", theme());
        assert_eq!(plain_text(&rows[0]), "+3 added");
        // Added line = ToolDiffAdded foreground
        let expected = Style::fg(theme().fg(ColorToken::ToolDiffAdded));
        assert_eq!(rows[0][0].style, expected);
    }

    #[test]
    fn removed_then_added_blocks_are_ordered() {
        let diff = "-1 a\n-2 b\n+3 c\n+4 d";
        let rows = render_diff(diff, theme());
        assert_eq!(rows.len(), 4);
        assert_eq!(plain_text(&rows[0]), "-1 a");
        assert_eq!(plain_text(&rows[1]), "-2 b");
        assert_eq!(plain_text(&rows[2]), "+3 c");
        assert_eq!(plain_text(&rows[3]), "+4 d");
    }

    #[test]
    fn single_pair_gets_intra_line_diff() {
        // Word-level diff: b → c inverse video; a remains normal
        let diff = "-1 a b\n+1 a c";
        let rows = render_diff(diff, theme());
        assert_eq!(rows.len(), 2);
        let removed: Vec<&str> = rows[0].iter().map(|s| s.text.as_str()).collect();
        // "-1 " + "a " + "b" (inverse video)
        assert_eq!(removed.join(""), "-1 a b");
        assert!(rows[0]
            .iter()
            .any(|s| s.text == "b" && !s.style.attributes.is_empty()));
        let added: Vec<&str> = rows[1].iter().map(|s| s.text.as_str()).collect();
        assert_eq!(added.join(""), "+1 a c");
        assert!(rows[1]
            .iter()
            .any(|s| s.text == "c" && !s.style.attributes.is_empty()));
    }

    #[test]
    fn tabs_expanded_to_three_spaces() {
        let rows = render_diff(" 10 a\tb", theme());
        assert_eq!(plain_text(&rows[0]), " 10 a   b");
    }

    #[test]
    fn no_number_diff_lines() {
        let rows = render_diff("- content", theme());
        assert_eq!(plain_text(&rows[0]), "- content");
    }
}
