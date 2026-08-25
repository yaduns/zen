// REQ-TUI-01 (pi-413): Editor component — input intake: keys / paste / action routing.
// The keybinding→action mapping uses inline defaults (aligned with pi TUI_KEYBINDINGS) and is replaced by the keybinding definition table after T-TU-03-1 is implemented
// (A cross-reference comment is left inline); this module is the hook point for T-TU-03-5 (undo) to add a snapshot outside handle_action.
use super::{Editor, EditorAction};

impl Editor {
    /// Handle one key press. Only responds to `KeyEventKind::Press` (Release/Repeat ignored, per T-TU-03-1 semantics or filtered in this method).
    /// Dispatch rules: printable characters (with Shift, no Ctrl/Alt modifiers) → insert_char; otherwise consult the key binding table (inline default,
    /// replaced by its key definition table once T-TU-03-1 lands); on `EditorAction` match → `handle_action`; app-level actions like Ctrl+C
    /// are not routed to the editor (handled upstream, aligning with pi tui.input.copy "let parent handle").
    /// Enter semantics (aligned with pi): plain Enter=Submit; Shift+Enter / Ctrl+J=NewLine; terminals have no Shift+Enter,
    /// Inline the backslash fallback when supported (backspace+newline/submit when the character before the cursor is `\`), following inner's default keybindings.
    /// Consumer: T-AP-11-1 (routes events to the focused component).
    pub fn handle_key(&mut self, key: crossterm::event::KeyEvent) {
        use crossterm::event::{KeyCode, KeyEventKind, KeyModifiers};

        // Only respond to Press events
        if key.kind != KeyEventKind::Press {
            return;
        }

        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);

        match key.code {
            // ── Printable characters (no Ctrl/Alt modifiers; Shift not excluded) ──
            // insert_char exits history browsing itself; no need to repeat here
            KeyCode::Char(c) if !ctrl && !alt => {
                self.insert_char(c);
            }

            // ── Enter family (aligned with pi) ──
            // Ctrl+Enter is unbound (pi binds only plain enter=submit, shift+enter/ctrl+j=newline)
            KeyCode::Enter if ctrl => {}
            // Shift+Enter = NewLine (includes `\` fallback: backspace + submit)
            KeyCode::Enter if shift => self.handle_newline(),
            // Plain Enter = Submit (pi: disableSubmit checked first, `\` falls back to backspace + newline)
            KeyCode::Enter => {
                if self.disable_submit {
                    return;
                }
                if self.char_before_cursor_is_backslash() {
                    self.backspace();
                    self.newline();
                    return;
                }
                self.submit();
            }
            // Ctrl+J = NewLine (terminals typically lack Shift+Enter, includes `\` fallback)
            KeyCode::Char('j') if ctrl => self.handle_newline(),

            // ── Ctrl+character actions ──
            KeyCode::Char('a') if ctrl => self.handle_action(EditorAction::MoveToLineStart),
            KeyCode::Char('e') if ctrl => self.handle_action(EditorAction::MoveToLineEnd),
            KeyCode::Char('b') if ctrl => self.handle_action(EditorAction::MoveCursorLeft),
            KeyCode::Char('f') if ctrl => self.handle_action(EditorAction::MoveCursorRight),
            KeyCode::Char('w') if ctrl => self.handle_action(EditorAction::DeleteWordBackward),
            KeyCode::Char('u') if ctrl => self.handle_action(EditorAction::DeleteToLineStart),
            KeyCode::Char('k') if ctrl => self.handle_action(EditorAction::DeleteToLineEnd),
            KeyCode::Char('d') if ctrl => self.handle_action(EditorAction::DeleteCharForward),
            KeyCode::Char('y') if ctrl => self.handle_action(EditorAction::Yank),

            // ── Alt+character actions (aligned with pi TUI_KEYBINDINGS defaults: alt+b/f/d/y) ──
            KeyCode::Char('b') if alt => self.handle_action(EditorAction::MoveCursorWordLeft),
            KeyCode::Char('f') if alt => self.handle_action(EditorAction::MoveCursorWordRight),
            KeyCode::Char('d') if alt => self.handle_action(EditorAction::DeleteWordForward),
            KeyCode::Char('y') if alt => self.handle_action(EditorAction::YankPop),

            // ── Arrow keys ──
            KeyCode::Up => self.handle_action(EditorAction::MoveCursorUp),
            KeyCode::Down => self.handle_action(EditorAction::MoveCursorDown),
            KeyCode::Left if !ctrl && !alt => self.handle_action(EditorAction::MoveCursorLeft),
            KeyCode::Right if !ctrl && !alt => self.handle_action(EditorAction::MoveCursorRight),
            KeyCode::Left if ctrl || alt => self.handle_action(EditorAction::MoveCursorWordLeft),
            KeyCode::Right if ctrl || alt => self.handle_action(EditorAction::MoveCursorWordRight),

            // ── Editing keys (modifier guards: alt+backspace/delete = word delete, aligned with pi) ──
            KeyCode::Backspace if alt => self.handle_action(EditorAction::DeleteWordBackward),
            KeyCode::Backspace => self.handle_action(EditorAction::DeleteCharBackward),
            KeyCode::Delete if alt => self.handle_action(EditorAction::DeleteWordForward),
            KeyCode::Delete => self.handle_action(EditorAction::DeleteCharForward),
            KeyCode::Home => self.handle_action(EditorAction::MoveToLineStart),
            KeyCode::End => self.handle_action(EditorAction::MoveToLineEnd),
            KeyCode::PageUp => self.handle_action(EditorAction::PageUp),
            KeyCode::PageDown => self.handle_action(EditorAction::PageDown),

            // ── Unmapped keys: ignore, pass to parent ──
            _ => {}
        }
    }

    /// Handle one bracketed-paste: normalize newlines/tabs, filter non-printable chars, insert the whole block atomically.
    /// Consumer: T-AP-11-1 (routes Event::Paste).
    pub fn handle_paste(&mut self, text: &str) {
        // Normalize newlines + expand tabs (aligned with pi normalizeText: **expand tabs before filtering**,
        // otherwise \t is a control character and would be dropped by the filter below, making the expansion dead code)
        let cleaned: String = text
            .replace("\r\n", "\n")
            .replace('\r', "\n")
            .replace('\t', "    ");

        // Filter non-printable characters (keep \n and ASCII ≥32 / non-ASCII; aligned with pi: char === "\n" || charCodeAt(0) >= 32)
        let filtered: String = cleaned
            .chars()
            .filter(|c| *c == '\n' || (*c as u32) >= 32)
            .collect();

        if filtered.is_empty() {
            return;
        }

        self.exit_history_browsing();
        self.insert_text(&filtered);
    }

    /// Dispatch editing behavior by action. **Unified entry point for all mutating operations** (T-TU-03-5 adds undo snapshot at this outer layer).
    /// Unimplemented actions (Undo/Yank/YankPop/Tab/Jump*) are documented no-ops; see the EditorAction table in the Data Model section.
    /// Consumer: called by `handle_key`; unit tests call directly; T-TU-03-1 key mapping tests consume.
    pub fn handle_action(&mut self, action: EditorAction) {
        match action {
            // ── Movement (first/last visible line transfer: history browsing / line start & end, aligned with pi moveCursorUp/Down three branches) ──
            EditorAction::MoveCursorUp => {
                if self.is_on_first_visual_line()
                    && (self.is_editor_empty()
                        || self.history_index > -1
                        || self.state.cursor_col == 0)
                {
                    self.navigate_history(-1);
                } else if self.is_on_first_visual_line() {
                    self.move_to_line_start();
                } else {
                    self.move_vertical(-1);
                }
            }
            EditorAction::MoveCursorDown => {
                if self.history_index > -1 && self.is_on_last_visual_line() {
                    self.navigate_history(1);
                } else if self.is_on_last_visual_line() {
                    self.move_to_line_end();
                } else {
                    self.move_vertical(1);
                }
            }
            EditorAction::MoveCursorLeft => self.move_char(-1),
            EditorAction::MoveCursorRight => self.move_char(1),
            EditorAction::MoveCursorWordLeft => self.move_word(false),
            EditorAction::MoveCursorWordRight => self.move_word(true),
            EditorAction::MoveToLineStart => self.move_to_line_start(),
            EditorAction::MoveToLineEnd => self.move_to_line_end(),
            EditorAction::PageUp => self.page_scroll(false),
            EditorAction::PageDown => self.page_scroll(true),

            // ── Editing (exit history browsing) ──
            EditorAction::DeleteCharBackward => {
                self.exit_history_browsing();
                self.backspace();
            }
            EditorAction::DeleteCharForward => {
                self.exit_history_browsing();
                self.delete_forward();
            }
            EditorAction::DeleteWordBackward => {
                self.exit_history_browsing();
                self.delete_word_backward();
            }
            EditorAction::DeleteWordForward => {
                self.exit_history_browsing();
                self.delete_word_forward();
            }
            EditorAction::DeleteToLineStart => {
                self.exit_history_browsing();
                self.delete_to_line_start();
            }
            EditorAction::DeleteToLineEnd => {
                self.exit_history_browsing();
                self.delete_to_line_end();
            }
            EditorAction::NewLine => {
                self.exit_history_browsing();
                self.newline();
            }
            EditorAction::Submit => {
                // disable_submit: once set, Submit does not trigger (aligned with the disableSubmit check in pi submit branch)
                if !self.disable_submit {
                    self.submit();
                }
            }

            // ── Unimplemented (reserved for T-TU-03-5/-7/-3) ──
            EditorAction::Yank => { /* no-op */ }
            EditorAction::YankPop => { /* no-op */ }
            EditorAction::Undo => { /* no-op */ }
            EditorAction::Tab => { /* no-op */ }

            // ── Jump mode (reserved for T-TU-03) ──
            EditorAction::JumpForward => { /* no-op */ }
            EditorAction::JumpBackward => { /* no-op */ }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::Editor;
    use super::super::EditorAction;
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

    // ── handle_key ──

    #[test]
    fn test_handle_key_inserts_char() {
        let mut e = Editor::new(0);
        e.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE));
        e.handle_key(KeyEvent::new(KeyCode::Char('b'), KeyModifiers::NONE));
        assert_eq!(e.text(), "ab");
        assert_eq!(e.cursor(), (0, 2));
    }

    #[test]
    fn test_handle_key_inserts_char_with_shift() {
        let mut e = Editor::new(0);
        e.handle_key(KeyEvent::new(KeyCode::Char('A'), KeyModifiers::SHIFT));
        assert_eq!(e.text(), "A");
    }

    #[test]
    fn test_handle_key_ignores_release_event() {
        let mut e = Editor::new(0);
        let key = KeyEvent {
            code: KeyCode::Char('a'),
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Release,
            state: crossterm::event::KeyEventState::NONE,
        };
        e.handle_key(key);
        assert_eq!(e.text(), "");
    }

    #[test]
    fn test_handle_key_enter_submits() {
        let mut e = Editor::new(0);
        let mut submitted = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        {
            let s = submitted.clone();
            e.on_submit = Some(Box::new(move |text: &str| {
                s.borrow_mut().push(text.to_string());
            }));
        }
        e.insert_text_at_cursor("hello");
        e.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(*submitted.borrow(), vec!["hello"]);
        assert_eq!(e.text(), "");
    }

    #[test]
    fn test_handle_key_enter_disabled_submit() {
        let mut e = Editor::new(0);
        e.disable_submit = true;
        e.insert_text_at_cursor("hello");
        e.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        // Should not submit, text unchanged
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_handle_key_enter_backslash_newline_instead_of_submit() {
        let mut e = Editor::new(0);
        let mut submitted = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        {
            let s = submitted.clone();
            e.on_submit = Some(Box::new(move |text: &str| {
                s.borrow_mut().push(text.to_string());
            }));
        }
        e.insert_text_at_cursor("hello\\");
        e.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        // Backslash removed, then newline inserted (not submit)
        assert!(submitted.borrow().is_empty());
        assert_eq!(e.text(), "hello\n");
    }

    #[test]
    fn test_handle_key_shift_enter_newline() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT));
        assert_eq!(e.text(), "hello\n");
    }

    #[test]
    fn test_handle_key_ctrl_j_newline() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::CONTROL));
        assert_eq!(e.text(), "hello\n");
    }

    #[test]
    fn test_handle_key_ctrl_enter_ignored() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL));
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_handle_key_ctrl_a_move_to_line_start() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL));
        assert_eq!(e.cursor(), (0, 0));
    }

    #[test]
    fn test_handle_key_ctrl_e_move_to_line_end() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.move_char(-3); // move left from end
        e.handle_key(KeyEvent::new(KeyCode::Char('e'), KeyModifiers::CONTROL));
        assert_eq!(e.cursor(), (0, 5));
    }

    #[test]
    fn test_handle_key_ctrl_b_move_left() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.handle_key(KeyEvent::new(KeyCode::Char('b'), KeyModifiers::CONTROL));
        assert_eq!(e.cursor(), (0, 4));
    }

    #[test]
    fn test_handle_key_ctrl_f_move_right() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.move_char(-3);
        e.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::CONTROL));
        assert_eq!(e.cursor(), (0, 3));
    }

    #[test]
    fn test_handle_key_backspace() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(e.text(), "hell");
    }

    #[test]
    fn test_handle_key_delete() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.move_char(-3);
        e.handle_key(KeyEvent::new(KeyCode::Delete, KeyModifiers::NONE));
        assert_eq!(e.text(), "helo");
    }

    #[test]
    fn test_handle_key_unmapped_key_ignored() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.handle_key(KeyEvent::new(KeyCode::F(1), KeyModifiers::NONE));
        assert_eq!(e.text(), "hello");
    }

    // ── handle_paste ──

    #[test]
    fn test_handle_paste_simple() {
        let mut e = Editor::new(0);
        e.handle_paste("hello world");
        assert_eq!(e.text(), "hello world");
    }

    #[test]
    fn test_handle_paste_normalizes_crlf() {
        let mut e = Editor::new(0);
        e.handle_paste("line1\r\nline2\rline3");
        assert_eq!(e.text(), "line1\nline2\nline3");
    }

    #[test]
    fn test_handle_paste_expands_tabs() {
        let mut e = Editor::new(0);
        e.handle_paste("a\tb");
        assert_eq!(e.text(), "a    b");
    }

    #[test]
    fn test_handle_paste_filters_control_chars() {
        let mut e = Editor::new(0);
        e.handle_paste("he\x01llo\x02");
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_handle_paste_preserves_newline() {
        let mut e = Editor::new(0);
        e.handle_paste("a\nb");
        assert_eq!(e.text(), "a\nb");
    }

    #[test]
    fn test_handle_paste_empty_is_noop() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("existing");
        e.handle_paste("");
        assert_eq!(e.text(), "existing");
    }

    #[test]
    fn test_handle_paste_only_control_chars_is_noop() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("existing");
        e.handle_paste("\x01\x02\x03");
        assert_eq!(e.text(), "existing");
    }

    #[test]
    fn test_handle_paste_at_cursor_position() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello world");
        e.move_char(-5); // cursor before "world"
        e.handle_paste("beautiful ");
        assert_eq!(e.text(), "hello beautiful world");
    }

    // ── handle_action: movement ──

    #[test]
    fn test_handle_action_move_cursor_left() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::MoveCursorLeft);
        assert_eq!(e.cursor(), (0, 4));
    }

    #[test]
    fn test_handle_action_move_cursor_right() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.move_char(-3);
        e.handle_action(EditorAction::MoveCursorRight);
        assert_eq!(e.cursor(), (0, 3));
    }

    #[test]
    fn test_handle_action_move_to_line_start() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::MoveToLineStart);
        assert_eq!(e.cursor(), (0, 0));
    }

    #[test]
    fn test_handle_action_move_to_line_end() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.move_char(-3);
        e.handle_action(EditorAction::MoveToLineEnd);
        assert_eq!(e.cursor(), (0, 5));
    }

    // ── handle_action: editing ──

    #[test]
    fn test_handle_action_delete_char_backward() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::DeleteCharBackward);
        assert_eq!(e.text(), "hell");
    }

    #[test]
    fn test_handle_action_delete_char_forward() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.move_char(-3);
        e.handle_action(EditorAction::DeleteCharForward);
        assert_eq!(e.text(), "helo");
    }

    #[test]
    fn test_handle_action_delete_word_backward() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello world");
        e.handle_action(EditorAction::DeleteWordBackward);
        assert_eq!(e.text(), "hello ");
    }

    #[test]
    fn test_handle_action_delete_word_forward() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello world");
        e.move_char(-5);
        e.handle_action(EditorAction::DeleteWordForward);
        assert_eq!(e.text(), "hello ");
    }

    #[test]
    fn test_handle_action_delete_to_line_start() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello world");
        e.move_char(-5);
        e.handle_action(EditorAction::DeleteToLineStart);
        assert_eq!(e.text(), "world");
    }

    #[test]
    fn test_handle_action_delete_to_line_end() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello world");
        e.move_char(-5);
        e.handle_action(EditorAction::DeleteToLineEnd);
        assert_eq!(e.text(), "hello ");
    }

    #[test]
    fn test_handle_action_newline() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::NewLine);
        assert_eq!(e.text(), "hello\n");
    }

    #[test]
    fn test_handle_action_submit() {
        let mut e = Editor::new(0);
        let submitted = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        {
            let s = submitted.clone();
            e.on_submit = Some(Box::new(move |text: &str| {
                s.borrow_mut().push(text.to_string());
            }));
        }
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::Submit);
        assert_eq!(*submitted.borrow(), vec!["hello"]);
    }

    #[test]
    fn test_handle_action_submit_disabled() {
        let mut e = Editor::new(0);
        e.disable_submit = true;
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::Submit);
        assert_eq!(e.text(), "hello");
    }

    // ── handle_action: editing exits history browsing ──

    #[test]
    fn test_handle_action_delete_exits_history_browsing() {
        let mut e = Editor::new(0);
        e.add_to_history("old");
        e.insert_text_at_cursor("draft");
        e.navigate_history(-1); // enter browsing
        assert_ne!(e.history_index, -1);

        e.handle_action(EditorAction::DeleteCharBackward);
        assert_eq!(e.history_index, -1);
    }

    #[test]
    fn test_handle_action_newline_exits_history_browsing() {
        let mut e = Editor::new(0);
        e.add_to_history("old");
        e.insert_text_at_cursor("draft");
        e.navigate_history(-1);

        e.handle_action(EditorAction::NewLine);
        assert_eq!(e.history_index, -1);
    }

    // ── handle_action: no-op actions don't panic ──

    #[test]
    fn test_handle_action_yank_noop() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::Yank);
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_handle_action_yank_pop_noop() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::YankPop);
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_handle_action_undo_noop() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::Undo);
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_handle_action_tab_noop() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::Tab);
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_handle_action_jump_forward_noop() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::JumpForward);
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_handle_action_jump_backward_noop() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::JumpBackward);
        assert_eq!(e.text(), "hello");
    }

    // ── handle_action: MoveCursorUp/Down with history ──

    #[test]
    fn test_handle_action_move_up_on_empty_editor_enters_history() {
        let mut e = Editor::new(0);
        e.add_to_history("hist entry");
        // Editor is empty, cursor at (0,0)
        e.handle_action(EditorAction::MoveCursorUp);
        assert_eq!(e.text(), "hist entry");
    }

    #[test]
    fn test_handle_action_move_down_exits_history() {
        let mut e = Editor::new(0);
        e.add_to_history("hist entry");
        e.insert_text_at_cursor("draft");
        e.navigate_history(-1); // enter history
        assert_eq!(e.text(), "hist entry");

        e.handle_action(EditorAction::MoveCursorDown);
        // Should restore draft
        assert_eq!(e.text(), "draft");
    }

    #[test]
    fn test_handle_action_move_up_at_col0_enters_history() {
        let mut e = Editor::new(0);
        e.add_to_history("hist");
        e.insert_text_at_cursor("hello");
        // cursor at (0, 5), move to start
        e.handle_action(EditorAction::MoveToLineStart);
        assert_eq!(e.cursor(), (0, 0));

        // Now at col 0 on first visual line → should enter history
        e.handle_action(EditorAction::MoveCursorUp);
        assert_eq!(e.text(), "hist");
    }
}
