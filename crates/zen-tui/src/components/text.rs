use std::ops::DerefMut;

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
    /// When bg is present, fill the entire line; top/bottom padding_y lines (filled with bg). Returns the actual line count.
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

        // In a terminal, one line is exactly 1 line high—there is no CSS-style line-height concept; each character cell occupies 1 terminal line
        // Compute the total height of the wrapped text
        let total = (self.padding_y as usize) * 2 + lines.len();
        let mut row = y;

        // Top padding
        for _ in 0..self.padding_y {
            if let Some(ref bg) = self.bg {
                buf.fill(x, row, width, 1, bg);
            }
            row += 1;
        }

        // Content lines
        for chunk in &lines {
            if let Some(ref bg) = self.bg {
                buf.fill(x, row, width, 1, bg);
            }
            buf.write(x + self.padding_x, row, &self.style, &chunk.text);
            row += 1;
        }

        // Bottom padding
        for _ in 0..self.padding_y {
            if let Some(ref bg) = self.bg {
                buf.fill(x, row, width, 1, bg);
            }
            row += 1;
        }

        // height is the allocated upper bound; the return value is the actual usage. The caller uses it to stack the next component.
        total.min(height as usize) as u16
    }
}
