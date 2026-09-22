// REQ-APP-11 (pi-358): compaction-summary-message — compaction summary message.
// Corresponding original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/compaction-summary-message.ts.
// Render with shared FoldRow/paint_fold_rows (customMessageBg box family, unified visuals).

use crate::components::{paint_fold_rows, Component, FoldRow};
use crate::keybindings::AppAction;
use crossterm::style::Attribute;
use zen_tui::highlight::StyledSegment;
use zen_tui::keybindings::KeybindingsManager;
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{BgToken, ColorToken, Theme};

/// integer → thousands-separated string (pi `toLocaleString()` counterpart; pure function, for collapsed text).
fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    let bytes = digits.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(b as char);
    }
    out
}

/// Compaction summary message: uses the customMessageBg style family; collapsed line shows "Compacted from N tokens (… to expand)",
/// When expanded: title + markdown summary. Consumer: T-AP-11-1 (after context compaction).
pub struct CompactionSummaryMessageComponent<'a> {
    summary: String,
    tokens_before: u64,
    expanded: bool,
    theme: &'a Theme,
    kb: &'a KeybindingsManager<AppAction>,
}

impl<'a> CompactionSummaryMessageComponent<'a> {
    /// Constructed with summary, pre-compaction token count, theme, and app keymap.
    pub fn new(
        summary: String,
        tokens_before: u64,
        theme: &'a Theme,
        kb: &'a KeybindingsManager<AppAction>,
    ) -> Self {
        CompactionSummaryMessageComponent {
            summary,
            tokens_before,
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

    /// Pre-compaction token count (data for display text).
    pub fn tokens_before(&self) -> u64 {
        self.tokens_before
    }

    fn plan(&self) -> Vec<FoldRow> {
        let theme = self.theme;
        let token_str = thousands(self.tokens_before);
        let mut rows = vec![FoldRow::Label {
            text: "[compaction]".to_string(),
            style: theme.style(ColorToken::CustomMessageLabel, None, &[Attribute::Bold]),
        }];
        rows.push(FoldRow::Gap);

        if self.expanded {
            let header = format!("**Compacted from {token_str} tokens**\n\n");
            let mut md = theme.markdown_theme();
            md.text = Style::fg(theme.fg(ColorToken::CustomMessageText));
            rows.push(FoldRow::Markdown {
                theme: md,
                text: format!("{header}{}", self.summary),
            });
        } else {
            let key =
                crate::components::keybinding_hints::key_text(self.kb, AppAction::ToolsExpand);
            rows.push(FoldRow::Segments {
                segments: vec![
                    StyledSegment {
                        style: Style::fg(theme.fg(ColorToken::CustomMessageText)),
                        text: format!("Compacted from {token_str} tokens ("),
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
}

impl Component for CompactionSummaryMessageComponent<'_> {
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
        zen_core::ansi::strip_ansi(&String::from_utf8_lossy(&out))
            .chars()
            .filter(|c| !c.is_control())
            .collect()
    }

    #[test]
    fn thousands_groups_digits() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1000), "1,000");
        assert_eq!(thousands(1234567), "1,234,567");
    }

    #[test]
    fn collapsed_shows_token_count_and_hint() {
        let kb = kb();
        let mut comp = CompactionSummaryMessageComponent::new("sum".into(), 1500000, theme(), &kb);
        let mut buf = ScreenBuffer::new(60, 10);
        comp.render(&mut buf, 0, 0, 60, 10);
        let text = flush_text(&mut buf);
        assert!(text.contains("[compaction]"));
        assert!(text.contains("Compacted from 1,500,000 tokens"));
        assert!(text.contains("ctrl+o"));
        assert!(!text.contains("sum"));
    }

    #[test]
    fn expanded_shows_summary() {
        let kb = kb();
        let mut comp = CompactionSummaryMessageComponent::new("the sum".into(), 1200, theme(), &kb);
        comp.set_expanded(true);
        let mut buf = ScreenBuffer::new(60, 10);
        comp.render(&mut buf, 0, 0, 60, 10);
        let text = flush_text(&mut buf);
        assert!(text.contains("the sum"));
        assert!(text.contains("Compacted from 1,200 tokens"));
    }

    #[test]
    fn accessors() {
        let kb = kb();
        let mut comp = CompactionSummaryMessageComponent::new("s".into(), 42, theme(), &kb);
        assert!(!comp.is_expanded());
        assert_eq!(comp.tokens_before(), 42);
        comp.set_expanded(true);
        assert!(comp.is_expanded());
    }
}
