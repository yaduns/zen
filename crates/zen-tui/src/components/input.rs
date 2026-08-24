use crate::screen::{ScreenBuffer, Style};
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use unicode_segmentation::UnicodeSegmentation;

/// Single-line input—"> " prompt + text window (scroll horizontally when content exceeds the width:
/// Three-stage cursor positioning: near-start, near-end, and centered; the window is derived from the text via wrap::slice_by_column) + reverse-video block cursor (Style.attributes=[Reverse]);
/// When focused, records rendering coordinates and the window origin for cursor_position() to locate the hardware cursor.
/// Unwired behaviors (not needed in the v1 search scenario): Ctrl+W/K/U, yank/yank-pop, Ctrl+Z—assigned to T-TU-03-8/-7/-5, respectively,
/// In handle_key, a no-op falls through. esc/cancel semantics = Esc/Ctrl+C triggers on_escape
pub struct Input {
    value: String,                           // Input content
    cursor: usize,                           // Cursor byte offset, always at a grapheme boundary (insert/delete advances by grapheme)
    focused: bool,                           // Reverse-video cursor + allow hardware cursor positioning
    on_submit: Option<Box<dyn FnMut(&str)>>, // Enter submits (current value as argument)
    on_escape: Option<Box<dyn FnMut()>>,     // Esc/Ctrl+C cancels
    scroll_start_col: usize, // Horizontal scroll window start column (value-column units, excluding the prompt; recorded by render)
}

impl Input {
    /// Constructor (prompt is fixed `"> "`).
    /// Consumers: T-AP-11-2 (session/settings-selector search box), T-AP-08-6.
    pub fn new() -> Self {
        Input {
            value: String::new(),
            cursor: 0,
            focused: true,
            on_submit: None,
            on_escape: None,
            scroll_start_col: 0,
        }
    }

    /// Current value.
    pub fn get_value(&self) -> &str {
        &self.value
    }

    /// Sets the value and clamps the cursor within the end boundary.
    pub fn set_value(&mut self, value: impl Into<String>) {
        self.value = value.into();
        self.cursor = self.value.len();
    }

    /// Set the focused state (when focused, render an inverse-video cursor and allow cursor_position to position the hardware cursor).
    pub fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
    }

    /// Key entry point: insert printable characters (no Ctrl/Alt); backspace/delete before the cursor; ←→/line start/end;
    /// Enter submits (on_submit); Esc/Ctrl+C cancels (on_escape); Ctrl+W/K/U (delete word/line)
    /// Along with Ctrl+Y (yank) and Ctrl+Z (undo), a no-op — kill-ring/word navigation/undo are respectively
    /// owned by T-TU-03-7/-8/-5 and will be hooked up here then. Returns whether the key was consumed (printable characters and navigation-type consumption,
    /// no-op keys are not consumed). paste uses a separate handle_paste entry point.
    pub fn handle_key(&mut self, key: &KeyEvent) -> bool {
        if key.kind != KeyEventKind::Press {
            return false;
        }

        // Get the start position of the grapheme cluster before pos
        fn prev_grapheme(s: &str, pos: usize) -> usize {
            s[..pos]
                .graphemes(true)
                .last()
                // g.len(), because Chinese characters are not one byte
                .map(|g| pos - g.len())
                .unwrap_or(0)
        }
        // Get the start position of the grapheme cluster after pos
        fn next_grapheme(s: &str, pos: usize) -> usize {
            s[pos..]
                .graphemes(true)
                .next()
                .map(|g| pos + g.len())
                .unwrap_or(s.len())
        }

        // KeyModifiers bit flags: CONTROL, ALT, and SHIFT can be combined
        match (key.code, key.modifiers) {
            (KeyCode::Enter, _) => {
                if let Some(ref mut cb) = self.on_submit {
                    cb(&self.value);
                }
                true
            }
            (KeyCode::Esc, _) | (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                if let Some(ref mut cb) = self.on_escape {
                    cb();
                }
                true
            }
            (KeyCode::Backspace, _) => {
                if self.cursor > 0 {
                    let prev = prev_grapheme(&self.value, self.cursor);
                    self.value.drain(prev..self.cursor);
                    self.cursor = prev;
                }
                true
            }
            (KeyCode::Delete, _) => {
                if self.cursor < self.value.len() {
                    let next = next_grapheme(&self.value, self.cursor);
                    self.value.drain(self.cursor..next);
                }
                true
            }
            (KeyCode::Left, _) => {
                if self.cursor > 0 {
                    self.cursor = prev_grapheme(&self.value, self.cursor);
                }
                true
            }
            (KeyCode::Right, _) => {
                self.cursor = next_grapheme(&self.value, self.cursor);
                true
            }
            (KeyCode::Home, _) => {
                self.cursor = 0;
                true
            }
            (KeyCode::End, _) => {
                self.cursor = self.value.len();
                true
            }
            (KeyCode::Char(c), KeyModifiers::NONE | KeyModifiers::SHIFT) => {
                self.value.insert(self.cursor, c);
                // Here c is of type char; both the len_utf8 method on char and the len method on String can determine grapheme boundaries
                self.cursor += c.len_utf8();
                true
            }
            // Ctrl+W/K/U/Y/Z — no-op (to be hooked up in T-TU-03-7/-8/-5)
            (
                KeyCode::Char('w' | 'k' | 'u' | 'y' | 'z' | 'a' | 'e' | 'b' | 'f'),
                KeyModifiers::CONTROL,
            ) => false,
            _ => false,
        }
    }

    /// Paste entry point (crossterm `Event::Paste` is delivered as a whole, with no frame-wise stitching): clean up line feeds/carriage returns,
    /// Expand tabs to 4 spaces and insert at the cursor (aligns with pi handlePaste semantics).
    pub fn handle_paste(&mut self, text: &str) {
        let cleaned = text
            .replace('\n', "")
            .replace('\r', "")
            .replace('\t', "    ");

        self.value.insert_str(self.cursor, &cleaned);
        self.cursor += cleaned.len();
    }

    /// Register a submit callback (counterpart of onSubmit; takes the current value).
    pub fn set_on_submit(&mut self, f: impl FnMut(&str) + 'static) {
        self.on_submit = Some(Box::new(f));
    }

    /// Register a cancel callback (counterpart of onEscape).
    pub fn set_on_escape(&mut self, f: impl FnMut() + 'static) {
        self.on_escape = Some(Box::new(f));
    }

    /// Hardware cursor coordinates **within the component** (column, row): column = 2 columns for the prompt `"> "` + the cursor's display column within the scrolling window,
    /// row = 0 (single-line component). Uses the same coordinate system as the editor's `cursor_position` (component-relative coordinates),
    /// The caller (T-AP-11-1) converts this to the actual screen position using the rectangle origin at render time.
    /// Before render, `scroll_start_col` is 0; returns the position after the prompt, (2, 0).
    pub fn cursor_position(&self) -> (u16, u16) {
        // Display column before the cursor (within value, in column units)
        let cursor_col = crate::screen::visible_width(&self.value[..self.cursor]);
        // Cursor column within the window = cursor column − window start (recorded by render, in value column units)
        let in_window = cursor_col.saturating_sub(self.scroll_start_col);
        // Fixed 2-column offset for the prompt `"> "`
        (2 + in_window as u16, 0)
    }
}

impl Default for Input {
    /// Same as `new()`.
    fn default() -> Self {
        Self::new()
    }
}

impl super::Component for Input {
    /// Single-line rendering: `"> "` prompt + visible window (display content directly if it fits; when too wide, locate the start in three segments based on cursor position,
    /// align with pi scroll logic), the cursor appears as a reverse-video block (`Style.attributes=[Reverse]`; when at the end of the line, reverse-video a space),
    /// Pad the remaining spaces at the end of the line to the full width. Record the window's starting column (in value-column units) for `cursor_position()`.
    /// Returns 1 (has content) or 0 (does not render when the available width is ≤ prompt width or height is 0).
    /// Consumers: T-AP-11-2 (session/settings-selector search box), T-AP-08-6.
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        const PROMPT: &str = "> ";
        let prompt_w = crate::screen::visible_width(PROMPT);

        // Extremely narrow/zero height: do not render (align with pi's `availableWidth <= 0` branch; the same applies when the height budget is 0)
        let available = (width as usize).saturating_sub(prompt_w);
        if available == 0 || height == 0 {
            return 0;
        }

        let value = &self.value;
        let total = crate::screen::visible_width(value);

        // ── Visible window: display directly if it fits / horizontal scrolling when too wide (pi three-segment positioning: use 0 when the cursor is in the near segment,
        //    align the near-tail segment to the right and center the middle; reserve one column for the reverse-video space when the cursor is at the end of the line)──
        let (visible_text, cursor_byte) = if total < available {
            // Everything fits: display the full text directly, with no scrolling
            self.scroll_start_col = 0;
            (value.to_string(), self.cursor)
        } else {
            let scroll_w = if self.cursor == value.len() {
                available - 1
            } else {
                available
            };
            let cursor_col = crate::screen::visible_width(&value[..self.cursor]);
            let half = scroll_w / 2;
            let start_col = if cursor_col < half {
                0
            } else if cursor_col > total.saturating_sub(half) {
                total.saturating_sub(scroll_w)
            } else {
                cursor_col.saturating_sub(half)
            };
            self.scroll_start_col = start_col;
            let visible_text = crate::wrap::slice_by_column(value, start_col, scroll_w);
            let before = crate::wrap::slice_by_column(
                value,
                start_col,
                cursor_col.saturating_sub(start_col),
            );
            (visible_text, before.len())
        };

        // ── Drawing: clear the line (pad to full width) → prompt → prefix → cursor block (reverse video) → suffix ──
        buf.fill(x, y, width, 1, &Style::default());
        buf.write(x, y, &Style::default(), PROMPT);

        let text_x = x + prompt_w as u16;
        let text_style = Style::default();
        // Render a reverse-video block cursor when focused (matches the field documentation "render a reverse-video cursor when focused");
        // When unfocused, write the cursor glyph as plain text (no block).
        let block_style = if self.focused {
            Style {
                attributes: vec![crossterm::style::Attribute::Reverse],
                ..Style::default()
            }
        } else {
            Style::default()
        };

        // Glyph covered by the cursor block: a reverse-video space when the cursor is at the end of the window (matches pi `atCursor ?? " "`)
        let (prefix, at_cursor, suffix) = if cursor_byte >= visible_text.len() {
            (visible_text.as_str(), " ", "")
        } else {
            let at = visible_text[cursor_byte..]
                .graphemes(true)
                .next()
                .unwrap_or("");
            let at_end = cursor_byte + at.len();
            (&visible_text[..cursor_byte], at, &visible_text[at_end..])
        };

        buf.write(text_x, y, &text_style, prefix);
        let block_x = text_x + crate::screen::visible_width(prefix) as u16;
        buf.write(block_x, y, &block_style, at_cursor);
        let suffix_x = block_x + crate::screen::visible_width(at_cursor) as u16;
        buf.write(suffix_x, y, &text_style, suffix);

        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Component;
    use crate::screen::{ScreenBuffer, Style};
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn new_defaults() {
        let i = Input::new();
        assert_eq!(i.value, "");
        assert_eq!(i.cursor, 0);
        assert!(i.focused);
        assert_eq!(i.scroll_start_col, 0);
        // Default == new()
        let d = Input::default();
        assert_eq!(d.value, "");
        assert_eq!(d.get_value(), "");
    }

    #[test]
    fn set_value_moves_cursor_to_end() {
        let mut i = Input::new();
        i.set_value("abc");
        assert_eq!(i.value, "abc");
        assert_eq!(i.cursor, 3);
        // Byte count: each CJK character in the sample is 3 bytes
        i.set_value("你好");
        assert_eq!(i.cursor, 6);
        assert_eq!(i.get_value(), "你好");
    }

    #[test]
    fn set_focused_toggles() {
        let mut i = Input::new();
        i.set_focused(false);
        assert!(!i.focused);
        i.set_focused(true);
        assert!(i.focused);
    }

    #[test]
    fn handle_key_ignores_non_press_events() {
        let mut i = Input::new();
        let mut key = KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE);
        key.kind = KeyEventKind::Release;
        assert!(!i.handle_key(&key));
        assert_eq!(i.value, "");
        key.kind = KeyEventKind::Repeat;
        assert!(!i.handle_key(&key));
        assert_eq!(i.value, "");
    }

    #[test]
    fn handle_key_inserts_characters_at_cursor() {
        let mut i = Input::new();
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Char('a'))));
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Char('b'))));
        assert_eq!(i.value, "ab");
        assert_eq!(i.cursor, 2);
        // Middle insertion: move the cursor back, then insert again
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Left)));
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Left)));
        assert_eq!(i.cursor, 0);
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Char('X'))));
        assert_eq!(i.value, "Xab");
        assert_eq!(i.cursor, 1);
    }

    #[test]
    fn handle_key_moves_by_grapheme_over_cjk() {
        let mut i = Input::new();
        i.set_value("你a好"); // Bytes 3+1+3 = 7
        assert_eq!(i.cursor, 7);
        // Move left by grapheme cluster: 'B' → 'a' → 'A' → 0
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Left)));
        assert_eq!(i.cursor, 4);
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Left)));
        assert_eq!(i.cursor, 3);
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Left)));
        assert_eq!(i.cursor, 0);
        // Move right back to
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Right)));
        assert_eq!(i.cursor, 3);
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Right)));
        assert_eq!(i.cursor, 4);
    }

    #[test]
    fn handle_key_backspace_deletes_previous_grapheme() {
        let mut i = Input::new();
        i.set_value("abcd");
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Backspace)));
        assert_eq!(i.value, "abc");
        assert_eq!(i.cursor, 3);
        // When the cursor is at 0, backspace is a no-op but is still consumed
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Home)));
        assert_eq!(i.cursor, 0);
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Backspace)));
        assert_eq!(i.value, "abc");
        assert_eq!(i.cursor, 0);
    }

    #[test]
    fn handle_key_backspace_deletes_cjk_grapheme() {
        let mut i = Input::new();
        i.set_value("你a好");
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Backspace)));
        assert_eq!(i.value, "你a");
        assert_eq!(i.cursor, 4);
    }

    #[test]
    fn handle_key_delete_removes_next_grapheme() {
        let mut i = Input::new();
        i.set_value("abcd");
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Home)));
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Delete)));
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Delete)));
        assert_eq!(i.value, "cd");
        assert_eq!(i.cursor, 0);
    }

    #[test]
    fn handle_key_home_end() {
        let mut i = Input::new();
        i.set_value("abc"); // cursor 3
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Home)));
        assert_eq!(i.cursor, 0);
        assert!(i.handle_key(&KeyEvent::from(KeyCode::End)));
        assert_eq!(i.cursor, 3);
    }

    #[test]
    fn handle_key_enter_triggers_on_submit_with_value() {
        let submitted = Rc::new(RefCell::new(Vec::new()));
        let s = submitted.clone();
        let mut i = Input::new();
        i.set_on_submit(move |v| s.borrow_mut().push(v.to_string()));
        i.set_value("hello");
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Enter)));
        assert_eq!(*submitted.borrow(), vec!["hello".to_string()]);
    }

    #[test]
    fn handle_key_esc_and_ctrl_c_trigger_on_escape() {
        let escaped = Rc::new(RefCell::new(0));
        let e = escaped.clone();
        let mut i = Input::new();
        i.set_on_escape(move || *e.borrow_mut() += 1);
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Esc)));
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(i.handle_key(&ctrl_c));
        assert_eq!(*escaped.borrow(), 2);
    }

    #[test]
    fn handle_key_ctrl_noops_not_consumed() {
        let mut i = Input::new();
        i.set_value("abc");
        for c in ['w', 'k', 'u', 'y', 'z', 'a', 'e', 'b', 'f'] {
            let key = KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL);
            assert!(!i.handle_key(&key), "ctrl+{c} should be no-op");
        }
        assert_eq!(i.value, "abc");
    }

    #[test]
    fn handle_key_unknown_keys_not_consumed() {
        let mut i = Input::new();
        i.set_value("abc");
        let f1 = KeyEvent::new(KeyCode::F(1), KeyModifiers::NONE);
        let ctrl_alt = KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL | KeyModifiers::ALT);
        assert!(!i.handle_key(&f1));
        assert!(!i.handle_key(&ctrl_alt));
        assert_eq!(i.value, "abc");
    }

    #[test]
    fn handle_paste_strips_newlines_and_expands_tabs() {
        let mut i = Input::new();
        i.handle_paste("a\nb\tc\r");
        assert_eq!(i.value, "ab    c");
        assert_eq!(i.cursor, i.value.len());
    }

    #[test]
    fn handle_paste_inserts_at_cursor() {
        let mut i = Input::new();
        i.set_value("ac");
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Left)));
        i.handle_paste("b");
        assert_eq!(i.value, "abc");
        assert_eq!(i.cursor, 2);
    }

    #[test]
    fn cursor_position_without_render() {
        let mut i = Input::new();
        assert_eq!(i.cursor_position(), (2, 0));
        i.set_value("ab"); // 2-column cursor → 2+2=4
        assert_eq!(i.cursor_position(), (4, 0));
    }

    #[test]
    fn cursor_position_accounts_scroll_start() {
        let mut i = Input::new();
        i.set_value("abcdef");
        assert!(i.handle_key(&KeyEvent::from(KeyCode::Home))); // cursor 0
        i.scroll_start_col = 3;
        // Cursor column 0 − starting point 3 → 0 → 2
        assert_eq!(i.cursor_position(), (2, 0));
        assert!(i.handle_key(&KeyEvent::from(KeyCode::End))); // cursor 6
        // 6 − 3 = 3 → 2+3=5
        assert_eq!(i.cursor_position(), (5, 0));
    }

    #[test]
    fn render_returns_zero_when_too_narrow_or_zero_height() {
        let mut buf = ScreenBuffer::new(20, 2);
        let mut i = Input::new();
        // available = width - 2(prompt) = 0
        assert_eq!(i.render(&mut buf, 0, 0, 2, 2), 0);
        // height 0
        assert_eq!(i.render(&mut buf, 0, 0, 20, 0), 0);
    }

    #[test]
    fn render_wide_no_scroll_returns_one() {
        let mut buf = ScreenBuffer::new(20, 2);
        let mut i = Input::new();
        i.set_value("ab");
        assert_eq!(i.render(&mut buf, 0, 0, 20, 2), 1);
        assert_eq!(i.scroll_start_col, 0);
    }

    #[test]
    fn render_narrow_scrolls_window_and_tracks_cursor() {
        let mut buf = ScreenBuffer::new(20, 2);
        let mut i = Input::new();
        i.set_value("abcdefghijklmnop"); // 16 columns
        // width=5 → available=3; cursor at end of line → scroll_w=2, start flush right at 14
        assert_eq!(i.render(&mut buf, 0, 0, 5, 2), 1);
        assert_eq!(i.scroll_start_col, 14);
        // cursor column 16 − 14 = 2 → prompt 2 + 2 = 4
        assert_eq!(i.cursor_position(), (4, 0));
    }

    #[test]
    fn render_unfocused_still_returns_one() {
        let mut buf = ScreenBuffer::new(20, 2);
        let mut i = Input::new();
        i.set_focused(false);
        i.set_value("x");
        assert_eq!(i.render(&mut buf, 0, 0, 20, 2), 1);
    }
}
