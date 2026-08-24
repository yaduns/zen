use crate::screen::{ScreenBuffer, Style};

/// Blank-line component. Stateless, no cache.
pub struct Spacer {
    lines: u16, // (number of blank lines to output)
}

impl Spacer {
    /// Construct with a blank-line count. Consumers: T-AP-11-1 (chat spacing), T-AP-11-2.
    pub fn new(lines: u16) -> Self {
        Self { lines }
    }

    /// Update the blank-line count.
    pub fn set_lines(&mut self, lines: u16) {
        self.lines = lines
    }
}

impl super::Component for Spacer {
    /// Output min(lines, height) blank lines (fill with default-style spaces).
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        let painted = self.lines.min(height);
        for row in y..y + painted {
            //  Style::default() is exactly the terminal default color (no foreground/background/attributes), which is precisely the semantics of "blank".
            buf.fill(x, row, width, 1, &Style::default());
        }
        painted
    }
}
