// REQ-APP-11 (pi-358): first-time-setup — first-time setup dialog (theme selection).
// Corresponding original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/first-time-setup.ts.
// analytics step removed (pi-325 telemetry dropped): result only has theme, no sharing option (see T-AP-11-2-implemented.md).

use crate::theme::TerminalTheme;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use zen_tui::components::Component;
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{ColorToken, Theme};

/// First-time setup result (v1 only theme; analytics step removed because pi-325 telemetry was dropped).
pub struct FirstTimeSetupResult {
    /// Selected terminal theme (dark/light).
    pub theme: TerminalTheme,
}

/// Theme options (value = TerminalTheme, label = display name).
const THEME_OPTIONS: [(&str, TerminalTheme); 2] = [
    ("Dark", TerminalTheme::Dark),
    ("Light", TerminalTheme::Light),
];

/// First-time setup dialog: step one "Select theme" (dark/light, preview applies immediately)—shown on first run per T-AP-08-6.
pub struct FirstTimeSetupComponent<'a> {
    theme_index: usize,
    theme: &'a Theme,
    on_theme_preview: Option<Box<dyn FnMut(TerminalTheme)>>,
    on_submit: Option<Box<dyn FnMut(FirstTimeSetupResult)>>,
    on_cancel: Option<Box<dyn FnMut()>>,
}

impl<'a> FirstTimeSetupComponent<'a> {
    /// Constructed with detected terminal theme, rendering theme, and three callbacks; preselects the detected theme.
    pub fn new(
        detected_theme: TerminalTheme,
        theme: &'a Theme,
        on_theme_preview: Option<Box<dyn FnMut(TerminalTheme)>>,
        on_submit: Option<Box<dyn FnMut(FirstTimeSetupResult)>>,
        on_cancel: Option<Box<dyn FnMut()>>,
    ) -> Self {
        let theme_index = THEME_OPTIONS
            .iter()
            .position(|(_, t)| *t == detected_theme)
            .unwrap_or(0);
        FirstTimeSetupComponent {
            theme_index,
            theme,
            on_theme_preview,
            on_submit,
            on_cancel,
        }
    }

    /// Moves the option (up/down) and triggers the preview callback.
    fn move_selection(&mut self, delta: isize) {
        let len = THEME_OPTIONS.len() as isize;
        let next = (self.theme_index as isize + delta).clamp(0, len - 1) as usize;
        if next != self.theme_index {
            self.theme_index = next;
            if let Some(cb) = &mut self.on_theme_preview {
                cb(THEME_OPTIONS[self.theme_index].1);
            }
        }
    }

    /// Currently selected theme.
    pub fn selected_theme(&self) -> TerminalTheme {
        THEME_OPTIONS[self.theme_index].1
    }
}

impl Component for FirstTimeSetupComponent<'_> {
    /// Renders the settings dialog (title + detection info + option list + key hints).
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        let theme = self.theme;
        let accent = Style::fg(theme.fg(ColorToken::Accent));
        let muted = Style::fg(theme.fg(ColorToken::Muted));
        let text = Style::fg(theme.fg(ColorToken::Text));
        let dim = Style::fg(theme.fg(ColorToken::Dim));

        let detected = if self.selected_theme() == TerminalTheme::Dark {
            "dark"
        } else {
            "light"
        };
        let mut lines: Vec<(String, Style, u16)> = Vec::new();
        lines.push((String::new(), Style::default(), 1));
        lines.push((
            "Welcome to Zen, the minimal coding agent.".to_string(),
            accent.clone(),
            1,
        ));
        lines.push((String::new(), Style::default(), 1));
        lines.push((
            format!("Detected system appearance: {detected}"),
            muted.clone(),
            3,
        ));
        lines.push(("Pick a theme.".to_string(), text.clone(), 1));
        lines.push((String::new(), Style::default(), 1));
        for (i, (label, _)) in THEME_OPTIONS.iter().enumerate() {
            let selected = i == self.theme_index;
            let prefix = if selected { "→ " } else { "  " };
            let style = if selected {
                accent.clone()
            } else {
                text.clone()
            };
            lines.push((format!("{prefix}{label}"), style, 0));
        }
        lines.push((String::new(), Style::default(), 1));
        lines.push(("↑↓ navigate    Esc skip".to_string(), dim, 1));

        let mut painted = 0u16;
        for (text, style, indent) in lines {
            if painted >= height {
                break;
            }
            buf.fill(x, y + painted, width, 1, &Style::default());
            buf.write(x + indent, y + painted, &style, &text);
            painted += 1;
        }
        painted
    }

    /// handle_key: ↑/↓/j/k to move, Enter to submit, Esc to skip.
    fn handle_key(&mut self, key: &KeyEvent) -> bool {
        match (key.code, key.modifiers) {
            (KeyCode::Up, _) | (KeyCode::Char('k'), KeyModifiers::NONE) => {
                self.move_selection(-1);
                true
            }
            (KeyCode::Down, _) | (KeyCode::Char('j'), KeyModifiers::NONE) => {
                self.move_selection(1);
                true
            }
            (KeyCode::Enter, _) => {
                let theme = self.selected_theme();
                if let Some(cb) = &mut self.on_submit {
                    cb(FirstTimeSetupResult { theme });
                }
                true
            }
            (KeyCode::Esc, _) | (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                if let Some(cb) = &mut self.on_cancel {
                    cb();
                }
                true
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn theme() -> &'static Theme {
        zen_tui::theme::builtin_dark_theme()
    }

    #[test]
    fn preselected_detected_theme() {
        let comp = FirstTimeSetupComponent::new(TerminalTheme::Light, theme(), None, None, None);
        assert_eq!(comp.selected_theme(), TerminalTheme::Light);
    }

    #[test]
    fn down_moves_and_previews() {
        let previews = Rc::new(RefCell::new(Vec::new()));
        let p = previews.clone();
        let mut comp = FirstTimeSetupComponent::new(
            TerminalTheme::Dark,
            theme(),
            Some(Box::new(move |t| p.borrow_mut().push(t))),
            None,
            None,
        );
        let down = KeyEvent::from(KeyCode::Down);
        assert!(comp.handle_key(&down));
        assert_eq!(comp.selected_theme(), TerminalTheme::Light);
        assert_eq!(*previews.borrow(), vec![TerminalTheme::Light]);
    }

    #[test]
    fn jk_also_move() {
        let mut comp = FirstTimeSetupComponent::new(TerminalTheme::Dark, theme(), None, None, None);
        assert!(comp.handle_key(&KeyEvent::from(KeyCode::Char('j'))));
        assert_eq!(comp.selected_theme(), TerminalTheme::Light);
        assert!(comp.handle_key(&KeyEvent::from(KeyCode::Char('k'))));
        assert_eq!(comp.selected_theme(), TerminalTheme::Dark);
    }

    #[test]
    fn enter_submits_theme() {
        let submitted = Rc::new(RefCell::new(None));
        let s = submitted.clone();
        let mut comp = FirstTimeSetupComponent::new(
            TerminalTheme::Dark,
            theme(),
            None,
            Some(Box::new(move |r| *s.borrow_mut() = Some(r.theme))),
            None,
        );
        assert!(comp.handle_key(&KeyEvent::from(KeyCode::Enter)));
        assert_eq!(*submitted.borrow(), Some(TerminalTheme::Dark));
    }

    #[test]
    fn esc_cancels() {
        let cancelled = Rc::new(RefCell::new(0));
        let c = cancelled.clone();
        let mut comp = FirstTimeSetupComponent::new(
            TerminalTheme::Dark,
            theme(),
            None,
            None,
            Some(Box::new(move || *c.borrow_mut() += 1)),
        );
        assert!(comp.handle_key(&KeyEvent::from(KeyCode::Esc)));
        assert_eq!(*cancelled.borrow(), 1);
    }

    #[test]
    fn unknown_keys_not_consumed() {
        let mut comp = FirstTimeSetupComponent::new(TerminalTheme::Dark, theme(), None, None, None);
        assert!(!comp.handle_key(&KeyEvent::from(KeyCode::Char('a'))));
    }

    #[test]
    fn renders_options() {
        let mut comp = FirstTimeSetupComponent::new(TerminalTheme::Dark, theme(), None, None, None);
        let mut buf = ScreenBuffer::new(60, 16);
        let painted = comp.render(&mut buf, 0, 0, 60, 16);
        assert!(painted >= 10);
        let mut out = Vec::new();
        buf.flush(&mut out).unwrap();
        let text = crate::components::strip_ansi(&String::from_utf8_lossy(&out));
        assert!(text.contains("Zen"));
        assert!(text.contains("Dark"));
        assert!(text.contains("Light"));
    }
}
