// REQ-APP-11 (pi-358): user-message — user message rendering (theme background box + markdown).
// Corresponds to original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/user-message.ts.
// pi's Box+Markdown composition (addChild rebuild) is rendered inline in zen as "rect background fill + paragraph lines written to buffer".

use crate::components::render::write_segment_row;
use zen_tui::components::Component;
use zen_tui::markdown::MarkdownTheme;
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{BgToken, ColorToken, Theme};

/// User message rendering: theme userMessageBg background box + markdown content (userMessageText foreground) —
/// T-AP-11-1 renders after the user sends a message (prompt echo).
///
/// Layout: 1 line padding top/bottom (including background) + markdown lines; `output_pad` columns of padding left/right.
pub struct UserMessageComponent {
    text: String,
    output_pad: u16,
    bg: Style,
    markdown_theme: MarkdownTheme,
}

impl UserMessageComponent {
    /// Constructed from text, theme, and output left padding (background = userMessageBg; content baseline = userMessageText).
    pub fn new(text: String, theme: &Theme, output_pad: u16) -> Self {
        let bg = Style::bg(theme.bg(BgToken::UserMessageBg));
        let mut md = theme.markdown_theme();
        md.text = Style::fg(theme.fg(ColorToken::UserMessageText));
        UserMessageComponent {
            text,
            output_pad,
            bg,
            markdown_theme: md,
        }
    }

    /// Update output left padding (symmetric left/right; affects content width and reflow).
    pub fn set_output_pad(&mut self, padding: u16) {
        self.output_pad = padding;
    }

    /// Update text content (for streaming echo/subsequent corrections).
    pub fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into();
    }

    /// Get the raw text.
    pub fn text(&self) -> &str {
        &self.text
    }
}

impl Component for UserMessageComponent {
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        let content_width = width.saturating_sub(self.output_pad * 2).max(1) as usize;
        let rows =
            zen_tui::markdown::render_markdown(&self.markdown_theme, &self.text, content_width);

        // total height = top/bottom padding 1 each + markdown line count
        let total = 2u16.saturating_add(rows.len() as u16);
        let painted = total.min(height);
        if painted == 0 {
            return 0;
        }

        // background fills the entire drawn area (including padding lines)
        buf.fill(x, y, width, painted, &self.bg);

        // Content lines: start at y+1, start from output_pad on the left (text cells merge into box background userMsgBg)
        let content_rows = painted.saturating_sub(2) as usize;
        for (i, row) in rows.iter().take(content_rows).enumerate() {
            write_segment_row(
                buf,
                x + self.output_pad,
                y + 1 + i as u16,
                width - self.output_pad,
                row,
                self.bg.bg,
            );
        }

        painted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zen_tui::components::Component;

    fn theme() -> &'static Theme {
        zen_tui::theme::builtin_dark_theme()
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
    fn renders_text_with_padding_rows() {
        let mut comp = UserMessageComponent::new("hello user".into(), theme(), 1);
        let mut buf = ScreenBuffer::new(30, 10);
        let painted = comp.render(&mut buf, 0, 0, 30, 10);
        assert_eq!(painted, 3, "top pad + 1 md row + bottom pad");
        assert!(flush_text(&mut buf).contains("hello user"));
    }

    #[test]
    fn wraps_long_text() {
        let mut comp = UserMessageComponent::new(
            "this is a very long user message that should wrap into multiple rows".into(),
            theme(),
            1,
        );
        let mut buf = ScreenBuffer::new(30, 20);
        let painted = comp.render(&mut buf, 0, 0, 30, 20);
        assert!(painted > 3, "long text should wrap");
    }

    #[test]
    fn empty_text_still_backgrounds() {
        let mut comp = UserMessageComponent::new(String::new(), theme(), 1);
        let mut buf = ScreenBuffer::new(30, 10);
        let painted = comp.render(&mut buf, 0, 0, 30, 10);
        assert_eq!(painted, 2, "just the two padding rows with bg");
    }

    #[test]
    fn clamps_to_height_budget() {
        let mut comp = UserMessageComponent::new("line1\nline2\nline3\nline4".into(), theme(), 0);
        let mut buf = ScreenBuffer::new(30, 10);
        // height budget 3 → top pad + 1 content line + bottom pad
        let painted = comp.render(&mut buf, 0, 0, 30, 3);
        assert_eq!(painted, 3);
    }

    #[test]
    fn set_text_and_pad_update_rendering() {
        let mut comp = UserMessageComponent::new("old".into(), theme(), 0);
        comp.set_text("new");
        comp.set_output_pad(2);
        assert_eq!(comp.text(), "new");
        assert_eq!(comp.output_pad, 2);
        let mut buf = ScreenBuffer::new(30, 10);
        let painted = comp.render(&mut buf, 0, 0, 30, 10);
        assert!(painted >= 1);
        assert!(flush_text(&mut buf).contains("new"));
    }

    #[test]
    fn text_cells_carry_user_message_bg() {
        // Box background merge: glyph cell bg = userMsgBg (no longer shows default background through), consistent with the containing block
        let mut comp = UserMessageComponent::new("hello box".into(), theme(), 1);
        let mut buf = ScreenBuffer::new(30, 10);
        comp.render(&mut buf, 0, 0, 30, 10);
        let expected = theme().bg(BgToken::UserMessageBg);
        let row = buf.line_segments(1); // Content lines (line 0 = top padding)
        let text_seg = row
            .iter()
            .find(|s| s.text.contains("hello box"))
            .expect("text segment");
        assert_eq!(
            text_seg.style.bg,
            Some(expected),
            "user message glyphs should land on userMsgBg"
        );
    }
}
