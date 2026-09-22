// REQ-APP-11 (pi-358): DynamicBorder — full-width horizontal-line border.
// Corresponding original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/dynamic-border.ts.

use zen_tui::components::Component;
use zen_tui::screen::{ScreenBuffer, Style};

/// Used by bordered components such as messages/selectors to draw a horizontal-line border across the full width—
/// Renders one line of "─" × width (uses 1 when width is less than 1), colored with the given style.
///
/// Consumers: bash-execution, thinking-selector, session-selector, settings-selector,
/// first-time-setup、tree-selector、user-message-selector、bordered-loader。
pub struct DynamicBorder {
    style: Style,
}

impl DynamicBorder {
    /// Constructed with full-line border style (theme border/BorderAccent, etc., colored via `Theme::fg`).
    pub fn new(style: Style) -> Self {
        Self { style }
    }
}

impl Component for DynamicBorder {
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        if width == 0 || height == 0 {
            return 0;
        }
        let line = "─".repeat(width.max(1) as usize);
        buf.write(x, y, &self.style, &line);
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn renders_full_width_line() {
        let mut buf = ScreenBuffer::new(10, 1);
        let mut b = DynamicBorder::new(Style::default());
        assert_eq!(b.render(&mut buf, 0, 0, 10, 1), 1);
        let text = flush_text(&mut buf);
        // 10 ─ characters
        assert_eq!(text.matches('─').count(), 10);
    }

    #[test]
    fn zero_width_or_height_renders_nothing() {
        let mut buf = ScreenBuffer::new(10, 1);
        let mut b = DynamicBorder::new(Style::default());
        assert_eq!(b.render(&mut buf, 0, 0, 0, 1), 0);
        assert_eq!(b.render(&mut buf, 0, 0, 10, 0), 0);
    }
}
