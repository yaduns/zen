// REQ-APP-11 (pi-358): user-message-selector — fork start selection overlay.
// Corresponds to original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/user-message-selector.ts.
// Simplified rendering; message text is truncated to a single line by width (never exceeding terminal width).

use crate::components::dynamic_border::DynamicBorder;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use zen_tui::components::Component;
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{ColorToken, Theme};
use zen_tui::wrap::truncate_to_width;

/// List item of user-message-selector (contains entry id and single-line text). Plain data, same subject.
#[derive(Debug, Clone)]
pub struct UserMessageListItem {
    /// Session entry id (selected callback payload).
    pub id: String,
    /// Message text (collapsed to a single line for display).
    pub text: String,
}

/// Fork start-point selection overlay: list of historical user messages + cursor/scroll + "Message n of m" metadata line —
/// T-AP-11-1 select a message when forking/continuing; Enter returns the entry id, Esc cancels.
pub struct UserMessageSelectorComponent {
    messages: Vec<UserMessageListItem>,
    selected_index: usize,
    max_visible: usize,
    theme: Theme,
    on_select: Option<Box<dyn FnMut(String)>>,
    on_cancel: Option<Box<dyn FnMut()>>,
    top_border: DynamicBorder,
    bottom_border: DynamicBorder,
}

impl UserMessageSelectorComponent {
    /// Constructed from the message list, theme, and callback; preselects when `initial_selected_id` matches, otherwise selects the most recent one.
    pub fn new(
        messages: Vec<UserMessageListItem>,
        theme: Theme,
        on_select: Option<Box<dyn FnMut(String)>>,
        on_cancel: Option<Box<dyn FnMut()>>,
        initial_selected_id: Option<String>,
    ) -> Self {
        let selected_index = match &initial_selected_id {
            Some(id) => messages
                .iter()
                .position(|m| &m.id == id)
                .unwrap_or_else(|| messages.len().saturating_sub(1)),
            None => messages.len().saturating_sub(1),
        };
        let border = Style::fg(theme.fg(ColorToken::Border));
        UserMessageSelectorComponent {
            messages,
            selected_index,
            max_visible: 10,
            theme,
            on_select,
            on_cancel,
            top_border: DynamicBorder::new(border.clone()),
            bottom_border: DynamicBorder::new(border),
        }
    }

    /// Currently selected index.
    pub fn selected_index(&self) -> usize {
        self.selected_index
    }

    fn move_selection(&mut self, delta: isize) {
        let len = self.messages.len();
        if len == 0 {
            return;
        }
        self.selected_index =
            ((self.selected_index as isize + delta).rem_euclid(len as isize)) as usize;
    }
}

impl Component for UserMessageSelectorComponent {
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        let theme = &self.theme;
        let accent = Style::fg(theme.fg(ColorToken::Accent));
        let muted = Style::fg(theme.fg(ColorToken::Muted));
        let mut painted = 0u16;
        macro_rules! take {
            ($n:expr) => {{
                let remaining = height - painted;
                if height - painted == 0 {
                    return painted;
                }
                let step = ($n).min(remaining);
                if step > 0 {
                    painted += step;
                }
            }};
        }
        macro_rules! line {
            ($text:expr, $style:expr) => {{
                if height - painted == 0 {
                    return painted;
                }
                buf.fill(x, y + painted, width, 1, &Style::default());
                buf.write(x + 1, y + painted, $style, $text);
                take!(1);
            }};
        }

        if height > painted {
            let n = self
                .top_border
                .render(buf, x, y + painted, width, height - painted);
            take!(n);
        }
        line!("Fork from Message", &accent);
        line!(
            "Select a user message to copy the active path up to that point into a new session",
            &muted
        );
        if height > painted {
            buf.fill(x, y + painted, width, 1, &Style::default());
            take!(1);
        }

        if self.messages.is_empty() {
            line!("No user messages found", &muted);
        } else {
            // Visible window (selection kept centered as much as possible)
            let len = self.messages.len();
            let view = ((height - painted) as usize).min(self.max_visible);
            let half = view / 2;
            let start = if self.selected_index < half {
                0
            } else if self.selected_index + half >= len {
                len.saturating_sub(view).min(len)
            } else {
                self.selected_index - half
            };
            let end = (start + view).min(len);
            for i in start..end {
                if height - painted == 0 {
                    break;
                }
                let msg = &self.messages[i];
                let is_selected = i == self.selected_index;
                buf.fill(x, y + painted, width, 1, &Style::default());
                let cursor = if is_selected { "›" } else { " " };
                let cursor_style = if is_selected {
                    accent.clone()
                } else {
                    Style::default()
                };
                buf.write(x + 1, y + painted, &cursor_style, cursor);
                let normalized = msg.text.split_whitespace().collect::<Vec<_>>().join(" ");
                let max_w = (width as usize).saturating_sub(3);
                let truncated = truncate_to_width(&normalized, max_w, "…");
                let style = if is_selected {
                    Style::fg(theme.fg(ColorToken::Text))
                } else {
                    Style::default()
                };
                buf.write(x + 3, y + painted, &style, &truncated);
                take!(1);
                if is_selected && height > painted {
                    buf.fill(x, y + painted, width, 1, &Style::default());
                    buf.write(
                        x + 3,
                        y + painted,
                        &muted,
                        &format!("Message {} of {}", i + 1, len),
                    );
                    take!(1);
                }
                if height > painted {
                    buf.fill(x, y + painted, width, 1, &Style::default());
                    take!(1);
                }
            }
        }

        if height > painted {
            buf.fill(x, y + painted, width, 1, &Style::default());
            take!(1);
        }
        if height > painted {
            let n = self
                .bottom_border
                .render(buf, x, y + painted, width, height - painted);
            take!(n);
        }
        painted
    }

    fn handle_key(&mut self, key: &KeyEvent) -> bool {
        match (key.code, key.modifiers) {
            (KeyCode::Up, _) => {
                self.move_selection(-1);
                true
            }
            (KeyCode::Down, _) => {
                self.move_selection(1);
                true
            }
            (KeyCode::Enter, _) => {
                if let Some(msg) = self.messages.get(self.selected_index) {
                    let id = msg.id.clone();
                    if let Some(cb) = &mut self.on_select {
                        cb(id);
                    }
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

    fn theme() -> Theme {
        zen_tui::theme::builtin_dark_theme().clone()
    }

    fn items() -> Vec<UserMessageListItem> {
        vec![
            UserMessageListItem {
                id: "e1".into(),
                text: "first message".into(),
            },
            UserMessageListItem {
                id: "e2".into(),
                text: "second message".into(),
            },
        ]
    }

    #[test]
    fn defaults_to_most_recent() {
        let comp = UserMessageSelectorComponent::new(items(), theme(), None, None, None);
        assert_eq!(comp.selected_index(), 1);
    }

    #[test]
    fn initial_selected_id_preselects() {
        let comp =
            UserMessageSelectorComponent::new(items(), theme(), None, None, Some("e1".into()));
        assert_eq!(comp.selected_index(), 0);
    }

    #[test]
    fn up_down_navigate() {
        let mut comp = UserMessageSelectorComponent::new(items(), theme(), None, None, None);
        let up = KeyEvent::from(KeyCode::Up);
        assert!(comp.handle_key(&up));
        assert_eq!(comp.selected_index(), 0);
        let down = KeyEvent::from(KeyCode::Down);
        assert!(comp.handle_key(&down));
        assert_eq!(comp.selected_index(), 1);
    }

    #[test]
    fn enter_selects_id() {
        let selected = Rc::new(RefCell::new(None::<String>));
        let selected_captured = selected.clone();
        let mut comp = UserMessageSelectorComponent::new(
            items(),
            theme(),
            Some(Box::new(move |id| {
                *selected_captured.borrow_mut() = Some(id)
            })),
            None,
            None,
        );
        comp.move_selection(-1); // to e1
        let enter = KeyEvent::from(KeyCode::Enter);
        assert!(comp.handle_key(&enter));
        assert_eq!(*selected.borrow(), Some("e1".to_string()));
    }

    #[test]
    fn esc_cancels() {
        let cancelled = Rc::new(RefCell::new(0));
        let cancelled_captured = cancelled.clone();
        let mut comp = UserMessageSelectorComponent::new(
            items(),
            theme(),
            None,
            Some(Box::new(move || *cancelled_captured.borrow_mut() += 1)),
            None,
        );
        assert!(comp.handle_key(&KeyEvent::from(KeyCode::Esc)));
        assert_eq!(*cancelled.borrow(), 1);
    }

    #[test]
    fn renders_list() {
        let mut comp = UserMessageSelectorComponent::new(items(), theme(), None, None, None);
        let mut buf = ScreenBuffer::new(80, 20);
        let painted = comp.render(&mut buf, 0, 0, 80, 20);
        assert!(painted >= 8);
        let mut out = Vec::new();
        buf.flush(&mut out).unwrap();
        let text = crate::components::strip_ansi(&String::from_utf8_lossy(&out));
        assert!(text.contains("Fork from Message"));
        assert!(text.contains("first message"));
        assert!(text.contains("Message 2 of 2"));
    }
}
