// REQ-APP-11 (pi-358): custom-message — custom message rendering (zen-core CustomMessage entry).
// Corresponding original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/custom-message.ts.
// v1 has no extension renderer: customRenderer is removed (see T-AP-11-2-implemented.md integration 5); only default layout is done
// (customMessageBg background + `[custom_type]` tag + text markdown).

use crate::components::{paint_fold_rows, Component, FoldRow};
use crossterm::style::Attribute;
use zen_tui::markdown::MarkdownTheme;
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{BgToken, ColorToken, Theme};

/// Custom message (zen-core `SessionTreeEntry::CustomMessage`) rendering: customMessageBg background +
/// `[custom_type]` tag + text content markdown (customMessageText). v1 has no extension renderer,
/// `customRenderer` parameter is removed (see T-AP-11-2-implemented.md integration 5).
/// Consumer: T-AP-11-1/skill system (custom-type messages such as skill invocation feedback).
pub struct CustomMessageComponent<'a> {
    custom_type: String,
    text: String,
    theme: &'a Theme,
}

impl<'a> CustomMessageComponent<'a> {
    /// Constructed with custom type + text content and theme (text may be a plain string or concatenated content blocks, normalized by the caller).
    pub fn new(custom_type: String, text: String, theme: &'a Theme) -> Self {
        CustomMessageComponent {
            custom_type,
            text,
            theme,
        }
    }

    /// Expanded state remains API-compatible (pi passes setExpanded to the renderer; v1 has no renderer, always expanded to show full text).
    pub fn set_expanded(&mut self, _expanded: bool) {}

    /// Custom type identifier.
    pub fn custom_type(&self) -> &str {
        &self.custom_type
    }

    fn plan(&self) -> Vec<FoldRow> {
        let theme = self.theme;
        let mut md: MarkdownTheme = theme.markdown_theme();
        md.text = Style::fg(theme.fg(ColorToken::CustomMessageText));
        vec![
            FoldRow::Label {
                text: format!("[{}]", self.custom_type),
                style: theme.style(ColorToken::CustomMessageLabel, None, &[Attribute::Bold]),
            },
            FoldRow::Gap,
            FoldRow::Markdown {
                theme: md,
                text: self.text.clone(),
            },
        ]
    }
}

impl Component for CustomMessageComponent<'_> {
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        let bg = Style::bg(self.theme.bg(BgToken::CustomMessageBg));
        let rows = self.plan();
        paint_fold_rows(buf, x, y, width, height, &rows, 1, Some(&bg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn theme() -> &'static Theme {
        zen_tui::theme::builtin_dark_theme()
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
    fn renders_label_and_content() {
        let mut comp = CustomMessageComponent::new("skill".into(), "custom body".into(), theme());
        assert_eq!(comp.custom_type(), "skill");
        let mut buf = ScreenBuffer::new(60, 10);
        let painted = comp.render(&mut buf, 0, 0, 60, 10);
        assert!(painted >= 3);
        let text = flush_text(&mut buf);
        assert!(text.contains("[skill]"));
        assert!(text.contains("custom body"));
    }

    #[test]
    fn set_expanded_is_noop() {
        let mut comp = CustomMessageComponent::new("t".into(), "x".into(), theme());
        comp.set_expanded(true);
        let mut buf = ScreenBuffer::new(20, 10);
        assert!(comp.render(&mut buf, 0, 0, 20, 10) >= 3);
    }

    #[test]
    fn empty_text_shows_label_only_extra_padding() {
        let mut comp = CustomMessageComponent::new("t".into(), String::new(), theme());
        let mut buf = ScreenBuffer::new(20, 10);
        let painted = comp.render(&mut buf, 0, 0, 20, 10);
        assert!(painted >= 2);
    }
}
