// REQ-APP-11 (pi-358): bordered-loader — bordered cancellable loader block.
// Corresponds to original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/bordered-loader.ts.
// pi's CancellableLoader subclassing is implemented as composition in zen (CancellableLoader handles Esc/Ctrl+C cancellation itself).

use crate::components::dynamic_border::DynamicBorder;
use crate::components::keybinding_hints::raw_key_hint_segments;
use crate::components::render::write_segment_row;
use tokio_util::sync::CancellationToken;
use zen_tui::components::cancellable_loader::CancellableLoader;
use zen_tui::components::Component;
use zen_tui::highlight::StyledSegment;
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{ColorToken, Theme};

/// Bordered cancellable loader block (top/bottom DynamicBorder + CancellableLoader + cancel-key hint) —
/// shown by T-AP-11-1 while waiting on an external process / long blocking work; Esc/Ctrl+C triggers the cancel callback (built into CancellableLoader).
pub struct BorderedLoader {
    loader: CancellableLoader,
    border: DynamicBorder,
    cancel_hint: Vec<StyledSegment>,
    show_cancel_hint: bool,
}

impl BorderedLoader {
    /// Constructed with cancel token, message, and theme (v1 is always cancellable — if a wait scenario is not cancellable, the caller does not create this component).
    pub fn new(token: CancellationToken, message: impl Into<String>, theme: &Theme) -> Self {
        let spinner_style = theme.style(ColorToken::Accent, None, &[]);
        let message_style = Style::fg(theme.fg(ColorToken::Muted));
        let border = DynamicBorder::new(Style::fg(theme.fg(ColorToken::Border)));
        let cancel_hint = raw_key_hint_segments("esc", "cancel", theme);
        BorderedLoader {
            loader: CancellableLoader::new(token, spinner_style, message_style, message),
            border,
            cancel_hint,
            show_cancel_hint: true,
        }
    }

    /// Register cancel callback (fires on Esc/Ctrl+C; wired up and cleaned up by the caller).
    pub fn set_on_cancel(&mut self, f: impl FnMut() + 'static) {
        self.loader.set_on_cancel(f);
    }

    /// Cancel token handle (for caller-side pre-subscription).
    pub fn cancel_token(&self) -> &CancellationToken {
        self.loader.cancel_token()
    }

    /// Whether cancellation has occurred.
    pub fn is_cancelled(&self) -> bool {
        self.loader.is_cancelled()
    }

    /// Start animation.
    pub fn start(&mut self) {
        self.loader.start();
    }

    /// Stop animation.
    pub fn stop(&mut self) {
        self.loader.stop();
    }

    /// Update message text.
    pub fn set_message(&mut self, message: impl Into<String>) {
        self.loader.set_message(message);
    }

    /// Hide the cancel-key hint line (for extended UI customization; shown by default in v1).
    pub fn set_show_cancel_hint(&mut self, show: bool) {
        self.show_cancel_hint = show;
    }
}

impl Component for BorderedLoader {
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        let mut row = y;
        let mut painted = 0u16;
        let mut remaining = height;

        if painted < height {
            let n = self.border.render(buf, x, row, width, remaining);
            painted += n;
            row += n;
            remaining -= n;
        }
        if painted < height {
            let n = self.loader.render(buf, x, row, width, remaining);
            painted += n;
            row += n;
            remaining -= n;
        }
        if self.show_cancel_hint && painted < height && remaining >= 1 {
            buf.fill(x, row, width, 1, &Style::default());
            write_segment_row(
                buf,
                x + 1,
                row,
                width.saturating_sub(2),
                &self.cancel_hint,
                None,
            );
            painted += 1;
            row += 1;
            remaining -= 1;
        }
        if painted < height {
            let n = self.border.render(buf, x, row, width, remaining);
            painted += n;
        }

        painted
    }

    fn handle_key(&mut self, key: &crossterm::event::KeyEvent) -> bool {
        // Esc/Ctrl+C cancellation handled by CancellableLoader
        self.loader.handle_key(key)
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
        crate::components::strip_ansi(&String::from_utf8_lossy(&out))
            .chars()
            .filter(|c| !c.is_control())
            .collect()
    }

    #[test]
    fn renders_borders_and_message() {
        let mut loader = BorderedLoader::new(CancellationToken::new(), "loading", theme());
        let mut buf = ScreenBuffer::new(30, 6);
        let painted = loader.render(&mut buf, 0, 0, 30, 6);
        assert_eq!(painted, 4, "border + loader + hint + border");
        let text = flush_text(&mut buf);
        assert!(text.contains("loading"));
        assert!(text.contains("cancel"));
    }

    #[test]
    fn cancel_key_triggers_cancelled_state() {
        let token = CancellationToken::new();
        let mut loader = BorderedLoader::new(token.clone(), "x", theme());
        loader.set_on_cancel(|| {});
        assert!(!loader.is_cancelled());
        assert!(loader.handle_key(&crossterm::event::KeyEvent::from(
            crossterm::event::KeyCode::Esc
        )));
        assert!(loader.is_cancelled());
    }

    #[test]
    fn other_keys_not_consumed() {
        let mut loader = BorderedLoader::new(CancellationToken::new(), "x", theme());
        assert!(!loader.handle_key(&crossterm::event::KeyEvent::from(
            crossterm::event::KeyCode::Char('a')
        )));
    }

    #[test]
    fn hide_cancel_hint_renders_three_rows() {
        let mut loader = BorderedLoader::new(CancellationToken::new(), "x", theme());
        loader.set_show_cancel_hint(false);
        let mut buf = ScreenBuffer::new(30, 6);
        assert_eq!(loader.render(&mut buf, 0, 0, 30, 6), 3);
    }
}
