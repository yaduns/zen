use crate::screen::{ScreenBuffer, Style};

/// Single-line truncated text component: take content before the first newline, truncate by display columns + ellipsis, with horizontal/vertical padding.
/// Always one line; if it does not fit, truncate and add an ellipsis. The user sees a "single-line hint bar that does not fear length".
pub struct TruncatedText {
    text: String,   // Content (at render time, take the part before the first newline)
    style: Style,   // Content style
    padding_x: u16, // Horizontal padding
    padding_y: u16, // Vertical padding
}

impl TruncatedText {
    /// Construct. Consumers: T-AP-11-1 (pending-message hint), T-AP-11-2 (single-line info such as session selector).
    pub fn new(text: impl Into<String>, style: Style, padding_x: u16, padding_y: u16) -> Self {
        TruncatedText {
            text: text.into(),
            style,
            padding_x,
            padding_y,
        }
    }

    /// Update content.
    pub fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into()
    }

    /// Update style.
    pub fn set_style(&mut self, style: Style) {
        self.style = style
    }
}

impl super::Component for TruncatedText {
    /// Single-line output: padding_y blank lines + truncated line (left/right padding, padded to width) + padding_y blank lines;
    /// Returns 1 + 2*padding_y (capped at height).
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        // Take content before the first newline
        let first_line = self.text.split('\n').next().unwrap_or("");

        // Content width (at least 1)
        let content_width = width.saturating_sub(self.padding_x * 2).max(1) as usize;

        // Truncate + ellipsis
        let truncated = crate::wrap::truncate_to_width(first_line, content_width, "…");

        let total = 1 + self.padding_y as usize * 2;
        let painted = total.min(height as usize);

        //  zip stops at the shorter one
        for (row, i) in (y..).zip(0..painted) {
            // Clear the entire line (prevent ghosting, and pad to width)
            buf.fill(x, row, width, 1, &Style::default());
            // Content line: when i == padding_y, it is the middle line
            if i == self.padding_y as usize {
                buf.write(x + self.padding_x, row, &self.style, &truncated);
            }
        }

        painted as u16
    }
}
