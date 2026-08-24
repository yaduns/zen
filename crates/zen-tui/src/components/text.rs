use crate::screen::{ScreenBuffer, Style};

/// Multi-line text display — \t expansion (wrap::normalize_tabs)
/// → word_wrap_line wrapping → per-line left/right padding + content style;
/// When bg is present, fill the entire line (including padding) as background and pad to full width (buf.fill);
/// top/bottom padding_y rows (also background-filled when bg is present). Empty/all-whitespace text is not rendered
pub struct Text {
    text: String,      // Content (empty/all-whitespace → 0 rows)
    style: Style,      // Content style
    padding_x: u16,    // Horizontal padding
    padding_y: u16,    // Vertical padding
    bg: Option<Style>, // Full-line background
}

impl Text {
    /// Constructs from content, style, and margins.
    /// Color functions are replaced by Style. Consumers: T-AP-11-1 (chat text), T-AP-11-2 (custom messages).
    pub fn new(text: impl Into<String>, style: Style, padding_x: u16, padding_y: u16) -> Self {
        Text {
            text: text.into(),
            style,
            padding_x,
            padding_y,
            bg: None,
        }
    }

    /// Updates content and triggers redraw (the counterpart of setText; no self-cache, only sets fields).
    pub fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into();
    }

    /// Updates content style (the style side of the setCustomBgFn counterpart).
    pub fn set_style(&mut self, style: Style) {
        self.style = style;
    }

    /// Sets full-line background (None = no background); with a background, the line is padded to full width and background-filled.
    pub fn set_bg(&mut self, bg: Option<Style>) {
        self.bg = bg;
    }

    /// Sets left/right and top/bottom padding.
    pub fn set_padding(&mut self, padding_x: u16, padding_y: u16) {
        self.padding_x = padding_x;
        self.padding_y = padding_y;
    }
}

impl super::Component for Text {
    /// Empty/whitespace text is not rendered (0 rows); otherwise tab expansion → wrapping → per-line padding + content style,
    /// When bg is present, fill the entire line background; top/bottom padding_y lines (filled with bg if present).
    /// Clipping semantics: total height = 2·padding_y + wrapped line count; actual lines painted this time = min(total height, height),
    /// When budget is insufficient, clip from the tail (bottom padding first, then content lines); the return value comes from the same source as painted line count.
    /// Consumers: T-AP-11-1 (chat text), T-AP-11-2 (custom message).
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        // Empty/all-whitespace text is not rendered
        if self.text.trim().is_empty() {
            return 0;
        }

        // Tab expansion → wrapping
        let normalized = crate::wrap::normalize_tabs(&self.text);
        // When width < padding_x * 2, return 0, .max(1): ensure at least 1 column to avoid word_wrap_line receiving width=0 and degrading to empty slices
        let content_width = width.saturating_sub(self.padding_x * 2).max(1) as usize;
        let lines = crate::wrap::word_wrap_line(&normalized, content_width);

        // In a terminal, one line is exactly 1 line high—there is no CSS-like line-height concept; each character cell occupies 1 terminal line.
        // Total budget and actual painted line count: a single cap controls both content and padding overflow (clip from the tail when budget is insufficient),
        // The return value comes from the same source as painting; the caller uses it to stack the next component.
        let pad_y = self.padding_y as usize;
        let total = pad_y * 2 + lines.len();
        let painted = total.min(height as usize);

        // Glyphs inherit the line background color by default (aligned with pi's default semantics of coloring the entire line): when self.style.bg is None
        // merge in the bg background color; when the caller explicitly sets style.bg, respect the explicit value (preserve the ability to use different background colors within one line).
        let glyph_style = if self.style.bg.is_none() {
            Style {
                bg: self.bg.as_ref().and_then(|b| b.bg),
                ..self.style.clone()
            }
        } else {
            self.style.clone()
        };

        // Single loop: segment membership is determined by line number (top padding / content / bottom padding),
        // The cap (painted) is the loop boundary; no per-row guard needs to be scattered around.
        for (row, i) in (y..).zip(0..painted) {
            if let Some(ref bg) = self.bg {
                buf.fill(x, row, width, 1, bg);
            }
            if i >= pad_y && i < pad_y + lines.len() {
                // Content line: lines[i - pad_y] (i < painted ≤ pad_y + lines.len(); the index never goes out of bounds)
                buf.write(
                    x + self.padding_x,
                    row,
                    &glyph_style,
                    &lines[i - pad_y].text,
                );
            }
        }

        painted as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Component;
    use crate::screen::{ScreenBuffer, Style};

    #[test]
    fn text_new_and_setters_update_fields() {
        let mut t = Text::new("hi", Style::default(), 1, 2);
        assert_eq!(t.text, "hi");
        assert_eq!(t.padding_x, 1);
        assert_eq!(t.padding_y, 2);
        assert!(t.bg.is_none());

        t.set_text("hello");
        t.set_style(Style::default());
        t.set_bg(Some(Style::default()));
        t.set_padding(3, 4);
        assert_eq!(t.text, "hello");
        assert_eq!(t.padding_x, 3);
        assert_eq!(t.padding_y, 4);
        assert!(t.bg.is_some());
    }

    #[test]
    fn render_empty_or_whitespace_text_returns_zero() {
        for s in ["", "   ", "\n\t\n  "] {
            let mut t = Text::new(s, Style::default(), 0, 0);
            let mut buf = ScreenBuffer::new(20, 5);
            assert_eq!(t.render(&mut buf, 0, 0, 20, 5), 0, "input: {s:?}");
        }
    }

    #[test]
    fn render_single_line_counts_padding() {
        let mut buf = ScreenBuffer::new(40, 10);
        // No padding → 1 line
        let mut t = Text::new("Hello", Style::default(), 0, 0);
        assert_eq!(t.render(&mut buf, 0, 0, 40, 10), 1);
        // padding_y=1 → 1 top + 1 bottom + 1 content = 3 lines
        let mut t2 = Text::new("Hello", Style::default(), 0, 1);
        assert_eq!(t2.render(&mut buf, 0, 0, 40, 10), 3);
    }

    #[test]
    fn render_clips_when_height_shorter_than_total() {
        let mut buf = ScreenBuffer::new(40, 10);
        // padding_y=2 → total=5; height=2 → clip from the tail, returns 2
        let mut t = Text::new("x", Style::default(), 0, 2);
        assert_eq!(t.render(&mut buf, 0, 0, 40, 2), 2);
        // height=0 → 0
        assert_eq!(t.render(&mut buf, 0, 0, 40, 0), 0);
    }

    #[test]
    fn render_wraps_long_text_into_multiple_lines() {
        let mut buf = ScreenBuffer::new(30, 20);
        let text = "aaaa bbbb cccc dddd";
        // Narrow width (content_width=6) forces wrapping into multiple lines
        let mut t = Text::new(text, Style::default(), 0, 0);
        let painted = t.render(&mut buf, 0, 0, 6, 20);
        assert!(painted >= 2, "narrow width should wrap; got {painted}");
        // Wide enough to fit the entire line → 1 line
        let mut t2 = Text::new(text, Style::default(), 0, 0);
        assert_eq!(t2.render(&mut buf, 0, 0, 40, 20), 1);
    }

    #[test]
    fn render_narrower_than_padding_keeps_one_column() {
        let mut buf = ScreenBuffer::new(30, 20);
        // width=3 < padding_x*2=10 → content_width is clamped to at least 1; it should still render (no panic, nonzero line count)
        let mut t = Text::new("abcdef", Style::default(), 5, 0);
        let painted = t.render(&mut buf, 0, 0, 3, 20);
        assert!(painted >= 1);
    }

    #[test]
    fn render_with_background_keeps_row_count() {
        let mut buf = ScreenBuffer::new(40, 10);
        let mut t = Text::new("hello", Style::default(), 0, 1);
        t.set_bg(Some(Style {
            fg: Some(crossterm::style::Color::Blue),
            ..Style::default()
        }));
        // Returned line count is the same with/without background (padding_y=1 → 3)
        assert_eq!(t.render(&mut buf, 0, 0, 40, 10), 3);
    }

    #[test]
    fn render_with_tabs_expands_and_does_not_panic() {
        let mut buf = ScreenBuffer::new(40, 10);
        let mut t = Text::new("a\tb", Style::default(), 1, 0);
        // After normalize_tabs expands, word_wrap_line renders; cells cannot be read to assert content, so verify no panic and a positive returned line count
        let painted = t.render(&mut buf, 0, 0, 40, 10);
        assert!(painted >= 1);
    }
}
