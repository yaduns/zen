// REQ-APP-11 (pi-358): branch-summary-message — branch summary message.
// Corresponds to original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/branch-summary-message.ts.
// Render with shared FoldRow/paint_fold_rows (customMessageBg box family, unified visuals).

use crate::components::{paint_fold_rows, FoldRow, Component};
use crate::keybindings::AppAction;
use crossterm::style::Attribute;
use zen_tui::highlight::StyledSegment;
use zen_tui::keybindings::KeybindingsManager;
use zen_tui::markdown::MarkdownTheme;
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{BgToken, ColorToken, Theme};

/// Branch summary message: customMessageBg background + `[branch]` label; when collapsed, a single-line hint for the expand key,
/// when expanded shows "**Branch Summary**" + markdown summary. Consumer: T-AP-11-1 (after fork).
pub struct BranchSummaryMessageComponent<'a> {
    summary: String,
    expanded: bool,
    theme: &'a Theme,
    kb: &'a KeybindingsManager<AppAction>,
}

impl<'a> BranchSummaryMessageComponent<'a> {
    /// Constructed with summary text, theme, and app keymap (collapsed hint key = app.tools.expand).
    pub fn new(summary: String, theme: &'a Theme, kb: &'a KeybindingsManager<AppAction>) -> Self {
        BranchSummaryMessageComponent {
            summary,
            expanded: false,
            theme,
            kb,
        }
    }

    /// Toggle expanded state (where `app.tools.expand` takes effect).
    pub fn set_expanded(&mut self, expanded: bool) {
        self.expanded = expanded;
    }

    /// Current expanded state.
    pub fn is_expanded(&self) -> bool {
        self.expanded
    }

    fn plan(&self) -> Vec<FoldRow> {
        let theme = self.theme;
        let md = self.markdown_theme();
        let mut rows = vec![FoldRow::Label {
            text: "[branch]".to_string(),
            style: theme.style(ColorToken::CustomMessageLabel, None, &[Attribute::Bold]),
        }];
        rows.push(FoldRow::Gap);

        if self.expanded {
            let header = "**Branch Summary**\n\n";
            rows.push(FoldRow::Markdown {
                theme: md,
                text: format!("{header}{}", self.summary),
            });
        } else {
            let key = crate::components::keybinding_hints::key_text(self.kb, AppAction::ToolsExpand);
            rows.push(FoldRow::Segments {
                segments: vec![
                    StyledSegment {
                        style: Style::fg(theme.fg(ColorToken::CustomMessageText)),
                        text: "Branch summary (".to_string(),
                    },
                    StyledSegment {
                        style: Style::fg(theme.fg(ColorToken::Dim)),
                        text: key,
                    },
                    StyledSegment {
                        style: Style::fg(theme.fg(ColorToken::CustomMessageText)),
                        text: " to expand)".to_string(),
                    },
                ],
            });
        }
        rows
    }

    /// Base markdown theme (text baseline = customMessageText).
    fn markdown_theme(&self) -> MarkdownTheme {
        let mut md = self.theme.markdown_theme();
        md.text = Style::fg(self.theme.fg(ColorToken::CustomMessageText));
        md
    }
}

impl Component for BranchSummaryMessageComponent<'_> {
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        let bg = Style::bg(self.theme.bg(BgToken::CustomMessageBg));
        let rows = self.plan();
        paint_fold_rows(buf, x, y, width, height, &rows, 1, Some(&bg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keybindings::app_default_bindings;
    use std::collections::HashMap;

    fn theme() -> &'static Theme {
        zen_tui::theme::builtin_dark_theme()
    }

    fn kb() -> KeybindingsManager<AppAction> {
        KeybindingsManager::new(app_default_bindings(), HashMap::new())
    }

    fn flush_text(buf: &mut ScreenBuffer) -> String {
        // flush output contains crossterm diff commands (MoveTo/SetColors/Print per cell);
        // here control bytes are stripped to restore "visible text" (printable chars, order preserved) for contains assertions
        let mut out = Vec::new();
        buf.flush(&mut out).unwrap();
        crate::components::strip_ansi(&String::from_utf8_lossy(&out))
            .chars()
            .filter(|c| !c.is_control())
            .collect()
    }


    #[test]
    fn collapsed_shows_label_and_expand_hint() {
        let kb = kb();
        let mut comp = BranchSummaryMessageComponent::new("the summary".into(), theme(), &kb);
        let mut buf = ScreenBuffer::new(60, 10);
        let painted = comp.render(&mut buf, 0, 0, 60, 10);
        assert!(painted >= 4);
        let text = flush_text(&mut buf);
        assert!(text.contains("[branch]"));
        assert!(text.contains("Branch summary"));
        assert!(text.contains("ctrl+o"));
        assert!(!text.contains("the summary"));
    }

    #[test]
    fn expanded_shows_markdown_summary() {
        let kb = kb();
        let mut comp = BranchSummaryMessageComponent::new("the summary".into(), theme(), &kb);
        comp.set_expanded(true);
        assert!(comp.is_expanded());
        let mut buf = ScreenBuffer::new(60, 10);
        comp.render(&mut buf, 0, 0, 60, 10);
        assert!(flush_text(&mut buf).contains("the summary"));
    }
}