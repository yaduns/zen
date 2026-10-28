// REQ-TUI-01 (pi-413): Editor component — input intake: keys / paste / action routing.
// Key→action mapping is routed through the action-domain key registry (table built by T-TU-03-1; injected at construction by T-AP-06-2 with user overrides layered on),
// Aligns with the kb.matches order of pi editor.ts handleInput; undo (T-TU-03-5) snapshot pushes happen inside each edit operation
// (text.rs/history.rs; see their header comments); the action layer only routes EditorAction::Undo to undo().
use super::{Editor, EditorAction};
use crate::keys::KeyId;

impl Editor {
    /// Handles a single key press. Only responds to `KeyEventKind::Press` (Release/Repeat ignored).
    /// Dispatch rules (aligned with the table-driven order of pi editor.ts handleInput): printable characters (including Shift, without Ctrl/Alt modifiers)
    /// → `insert_char` direct insertion (does not go through the action channel; the registry is unaware of printable characters); other keys go through `KeyId::from_key_event`
    /// after normalization, look up the action-domain key registry (injected at construction; user overrides take effect here — T-AP-06-2): `NewLine` before `Submit`
    /// (pi order; terminal fallback for `\`+Enter follows action handling: in the NewLine branch = backspace+submit, in the Submit branch =
    /// backspace+newline); other actions, when hit in definition order, are forwarded to `handle_action` (unimplemented variants are documented no-ops);
    /// If no action matches, the key is not consumed and is left to the upper layer for further routing. Application-level actions such as Ctrl+C/Ctrl+D are not handled here
    /// (handled by the upper-layer CustomEditor/TUI, aligned with pi tui.input.copy "let parent handle").
    /// Consumer: T-AP-11-1 (routes events to the focused component).
    pub fn handle_key(&mut self, key: crossterm::event::KeyEvent) {
        use crossterm::event::{KeyCode, KeyEventKind, KeyModifiers};

        // Only respond to Press events
        if key.kind != KeyEventKind::Press {
            return;
        }

        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);

        // ── Printable characters (without Ctrl/Alt modifiers; Shift not excluded) — direct insertion, bypassing the registry ──
        // insert_char exits history browsing itself; no need to repeat here
        if let KeyCode::Char(c) = key.code {
            if !ctrl && !alt {
                self.insert_char(c);
                return;
            }
        }

        // ── Action routing: after normalization to KeyId, look up the registry (user overrides take effect via the injected table, T-AP-06-2) ──
        let Some(kid) = KeyId::from_key_event(key) else {
            return;
        };

        // NewLine before Submit (aligned with pi handleInput order; default keys do not conflict; order alignment preserves refactor safety).
        // Terminal fallback for `\`+Enter is handled with actions: in handle_newline = backspace+submit, in this Submit branch = backspace+newline.
        if self.keybindings.matches(&kid, EditorAction::NewLine) {
            self.handle_newline();
            return;
        }
        if self.keybindings.matches(&kid, EditorAction::Submit) {
            if self.disable_submit {
                return;
            }
            // Fallback when the terminal lacks Shift+Enter support: if the character before the cursor is `\`, backspace+newline instead of submit (same as pi)
            if self.char_before_cursor_is_backslash() {
                self.backspace();
                self.newline();
                return;
            }
            self.submit();
            return;
        }

        // Other actions: hit one by one in definition order (multiple actions for the same key follow registry order, first match wins, same as pi)
        const ROUTED: [EditorAction; 22] = [
            EditorAction::MoveCursorUp,
            EditorAction::MoveCursorDown,
            EditorAction::MoveCursorLeft,
            EditorAction::MoveCursorRight,
            EditorAction::MoveCursorWordLeft,
            EditorAction::MoveCursorWordRight,
            EditorAction::MoveToLineStart,
            EditorAction::MoveToLineEnd,
            EditorAction::JumpForward,
            EditorAction::JumpBackward,
            EditorAction::PageUp,
            EditorAction::PageDown,
            EditorAction::DeleteCharBackward,
            EditorAction::DeleteCharForward,
            EditorAction::DeleteWordBackward,
            EditorAction::DeleteWordForward,
            EditorAction::DeleteToLineStart,
            EditorAction::DeleteToLineEnd,
            EditorAction::Yank,
            EditorAction::YankPop,
            EditorAction::Undo,
            EditorAction::Tab,
        ];
        for action in ROUTED {
            if self.keybindings.matches(&kid, action) {
                self.handle_action(action);
                return;
            }
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

    /// Dispatch editing behavior by action. **Centralized handling for all mutating operations**; mutating operations push undo snapshots internally according to pi's conditions.
    /// Unimplemented actions (Yank/YankPop/Tab/Jump*) are documented no-ops; see the EditorAction table in "Data Model".
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
                self.kill_with_capture(|e| e.delete_word_backward());
            }
            EditorAction::DeleteWordForward => {
                self.exit_history_browsing();
                self.kill_with_capture(|e| e.delete_word_forward());
            }
            EditorAction::DeleteToLineStart => {
                self.exit_history_browsing();
                self.kill_with_capture(|e| e.delete_to_line_start());
            }
            EditorAction::DeleteToLineEnd => {
                self.exit_history_browsing();
                self.kill_with_capture(|e| e.delete_to_line_end());
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

            // ── Kill ring (T-TU-03-7): Ctrl+Y inserts the last kill, Alt+Y walks back through it ──
            EditorAction::Yank => self.yank(),
            EditorAction::YankPop => self.yank_pop(),
            // Tab is handled by the autocomplete layer (zen-app UI), not by the editor core.
            EditorAction::Tab => { /* no-op */ }

            // ── Undo (T-TU-03-5): pop the stack to restore the state before the last edit ──
            EditorAction::Undo => self.undo(),

            // ── Jump mode (reserved for T-TU-03) ──
            EditorAction::JumpForward => { /* no-op */ }
            EditorAction::JumpBackward => { /* no-op */ }
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::keybindings::KeybindingsManager;
    use std::collections::HashMap;

    /// Test constructor: default keymap (no user overrides). Test-only — production assembly must pass an injected registry.
    fn e(padding_x: usize) -> Editor {
        Editor::new(
            padding_x,
            KeybindingsManager::new(
                crate::keybindings::editor_default_bindings(),
                HashMap::new(),
            ),
        )
    }
    use super::super::Editor;
    use super::super::EditorAction;
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

    // ── User override routing (T-AP-06-2): injected user overrides take effect via the registry ──

    fn e_with_user(user: HashMap<EditorAction, Vec<KeyId>>) -> Editor {
        Editor::new(0, KeybindingsManager::new(editor_default_bindings(), user))
    }

    // ── handle_key ──

    #[test]
    fn test_handle_key_inserts_char() {
        let mut e = e(0);
        e.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE));
        e.handle_key(KeyEvent::new(KeyCode::Char('b'), KeyModifiers::NONE));
        assert_eq!(e.text(), "ab");
        assert_eq!(e.cursor(), (0, 2));
    }

    #[test]
    fn test_handle_key_inserts_char_with_shift() {
        let mut e = e(0);
        e.handle_key(KeyEvent::new(KeyCode::Char('A'), KeyModifiers::SHIFT));
        assert_eq!(e.text(), "A");
    }

    #[test]
    fn test_handle_key_ignores_release_event() {
        let mut e = e(0);
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
        let mut e = e(0);
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
        let mut e = e(0);
        e.disable_submit = true;
        e.insert_text_at_cursor("hello");
        e.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        // Should not submit, text unchanged
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_handle_key_enter_backslash_newline_instead_of_submit() {
        let mut e = e(0);
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
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT));
        assert_eq!(e.text(), "hello\n");
    }

    #[test]
    fn test_handle_key_ctrl_j_newline() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::CONTROL));
        assert_eq!(e.text(), "hello\n");
    }

    #[test]
    fn test_handle_key_ctrl_enter_ignored() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL));
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_handle_key_ctrl_a_move_to_line_start() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL));
        assert_eq!(e.cursor(), (0, 0));
    }

    #[test]
    fn test_handle_key_ctrl_e_move_to_line_end() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.move_char(-3); // move left from end
        e.handle_key(KeyEvent::new(KeyCode::Char('e'), KeyModifiers::CONTROL));
        assert_eq!(e.cursor(), (0, 5));
    }

    #[test]
    fn test_handle_key_ctrl_b_move_left() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.handle_key(KeyEvent::new(KeyCode::Char('b'), KeyModifiers::CONTROL));
        assert_eq!(e.cursor(), (0, 4));
    }

    #[test]
    fn test_handle_key_ctrl_f_move_right() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.move_char(-3);
        e.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::CONTROL));
        assert_eq!(e.cursor(), (0, 3));
    }

    #[test]
    fn test_handle_key_backspace() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(e.text(), "hell");
    }

    #[test]
    fn test_handle_key_delete() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.move_char(-3);
        e.handle_key(KeyEvent::new(KeyCode::Delete, KeyModifiers::NONE));
        assert_eq!(e.text(), "helo");
    }

    #[test]
    fn test_handle_key_unmapped_key_ignored() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.handle_key(KeyEvent::new(KeyCode::F(1), KeyModifiers::NONE));
        assert_eq!(e.text(), "hello");
    }

    // ── handle_paste ──

    #[test]
    fn test_handle_paste_simple() {
        let mut e = e(0);
        e.handle_paste("hello world");
        assert_eq!(e.text(), "hello world");
    }

    #[test]
    fn test_handle_paste_normalizes_crlf() {
        let mut e = e(0);
        e.handle_paste("line1\r\nline2\rline3");
        assert_eq!(e.text(), "line1\nline2\nline3");
    }

    #[test]
    fn test_handle_paste_expands_tabs() {
        let mut e = e(0);
        e.handle_paste("a\tb");
        assert_eq!(e.text(), "a    b");
    }

    #[test]
    fn test_handle_paste_filters_control_chars() {
        let mut e = e(0);
        e.handle_paste("he\x01llo\x02");
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_handle_paste_preserves_newline() {
        let mut e = e(0);
        e.handle_paste("a\nb");
        assert_eq!(e.text(), "a\nb");
    }

    #[test]
    fn test_handle_paste_empty_is_noop() {
        let mut e = e(0);
        e.insert_text_at_cursor("existing");
        e.handle_paste("");
        assert_eq!(e.text(), "existing");
    }

    #[test]
    fn test_handle_paste_only_control_chars_is_noop() {
        let mut e = e(0);
        e.insert_text_at_cursor("existing");
        e.handle_paste("\x01\x02\x03");
        assert_eq!(e.text(), "existing");
    }

    #[test]
    fn test_handle_paste_at_cursor_position() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello world");
        e.move_char(-5); // cursor before "world"
        e.handle_paste("beautiful ");
        assert_eq!(e.text(), "hello beautiful world");
    }

    // ── handle_action: movement ──

    #[test]
    fn test_handle_action_move_cursor_left() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::MoveCursorLeft);
        assert_eq!(e.cursor(), (0, 4));
    }

    #[test]
    fn test_handle_action_move_cursor_right() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.move_char(-3);
        e.handle_action(EditorAction::MoveCursorRight);
        assert_eq!(e.cursor(), (0, 3));
    }

    #[test]
    fn test_handle_action_move_to_line_start() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::MoveToLineStart);
        assert_eq!(e.cursor(), (0, 0));
    }

    #[test]
    fn test_handle_action_move_to_line_end() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.move_char(-3);
        e.handle_action(EditorAction::MoveToLineEnd);
        assert_eq!(e.cursor(), (0, 5));
    }

    // ── handle_action: editing ──

    #[test]
    fn test_handle_action_delete_char_backward() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::DeleteCharBackward);
        assert_eq!(e.text(), "hell");
    }

    #[test]
    fn test_handle_action_delete_char_forward() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.move_char(-3);
        e.handle_action(EditorAction::DeleteCharForward);
        assert_eq!(e.text(), "helo");
    }

    #[test]
    fn test_handle_action_delete_word_backward() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello world");
        e.handle_action(EditorAction::DeleteWordBackward);
        assert_eq!(e.text(), "hello ");
    }

    #[test]
    fn test_handle_action_delete_word_forward() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello world");
        e.move_char(-5);
        e.handle_action(EditorAction::DeleteWordForward);
        assert_eq!(e.text(), "hello ");
    }

    #[test]
    fn test_handle_action_delete_to_line_start() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello world");
        e.move_char(-5);
        e.handle_action(EditorAction::DeleteToLineStart);
        assert_eq!(e.text(), "world");
    }

    #[test]
    fn test_handle_action_delete_to_line_end() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello world");
        e.move_char(-5);
        e.handle_action(EditorAction::DeleteToLineEnd);
        assert_eq!(e.text(), "hello ");
    }

    #[test]
    fn test_handle_action_newline() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::NewLine);
        assert_eq!(e.text(), "hello\n");
    }

    #[test]
    fn test_handle_action_submit() {
        let mut e = e(0);
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
        let mut e = e(0);
        e.disable_submit = true;
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::Submit);
        assert_eq!(e.text(), "hello");
    }

    // ── handle_action: editing exits history browsing ──

    #[test]
    fn test_handle_action_delete_exits_history_browsing() {
        let mut e = e(0);
        e.add_to_history("old");
        e.insert_text_at_cursor("draft");
        e.navigate_history(-1); // enter browsing
        assert_ne!(e.history_index, -1);

        e.handle_action(EditorAction::DeleteCharBackward);
        assert_eq!(e.history_index, -1);
    }

    #[test]
    fn test_handle_action_newline_exits_history_browsing() {
        let mut e = e(0);
        e.add_to_history("old");
        e.insert_text_at_cursor("draft");
        e.navigate_history(-1);

        e.handle_action(EditorAction::NewLine);
        assert_eq!(e.history_index, -1);
    }

    // ── handle_action: no-op actions don't panic ──

    #[test]
    fn test_handle_action_yank_noop() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::Yank);
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_handle_action_yank_pop_noop() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::YankPop);
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_handle_action_undo_restores() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::Undo);
        assert_eq!(e.text(), "");
    }

    #[test]
    fn test_handle_key_ctrl_minus_routes_to_undo() {
        let mut e = e(0);
        e.insert_char('a');
        assert_eq!(e.undo_stack.len(), 1);

        // The inline default Ctrl+- (undo) keybinding routes to EditorAction::Undo
        e.handle_key(KeyEvent::new(KeyCode::Char('-'), KeyModifiers::CONTROL));
        assert_eq!(e.text(), "");
    }

    #[test]
    fn test_handle_action_tab_noop() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::Tab);
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_handle_action_jump_forward_noop() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::JumpForward);
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_handle_action_jump_backward_noop() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        e.handle_action(EditorAction::JumpBackward);
        assert_eq!(e.text(), "hello");
    }

    // ── handle_action: MoveCursorUp/Down with history ──

    #[test]
    fn test_handle_action_move_up_on_empty_editor_enters_history() {
        let mut e = e(0);
        e.add_to_history("hist entry");
        // Editor is empty, cursor at (0,0)
        e.handle_action(EditorAction::MoveCursorUp);
        assert_eq!(e.text(), "hist entry");
    }

    #[test]
    fn test_handle_action_move_down_exits_history() {
        let mut e = e(0);
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
        let mut e = e(0);
        e.add_to_history("hist");
        e.insert_text_at_cursor("hello");
        // cursor at (0, 5), move to start
        e.handle_action(EditorAction::MoveToLineStart);
        assert_eq!(e.cursor(), (0, 0));

        // Now at col 0 on first visual line → should enter history
        e.handle_action(EditorAction::MoveCursorUp);
        assert_eq!(e.text(), "hist");
    }

    // ── User override routing (T-AP-06-2): injected user overrides take effect via the registry ──

    use crate::keybindings::editor_default_bindings;
    use crate::keys::KeyId;

    #[test]
    fn user_override_reroutes_action_and_releases_default() {
        // MoveCursorLeft overridden to ctrl+o: ctrl+o takes effect; default left/ctrl+b no longer works
        let mut ed = e_with_user(HashMap::from([(
            EditorAction::MoveCursorLeft,
            vec![KeyId::parse("ctrl+o").unwrap()],
        )]));
        ed.set_text("ab");
        ed.handle_key(KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL));
        assert_eq!(
            ed.cursor(),
            (0, 1),
            "ctrl+o should match the overridden MoveCursorLeft"
        );
        ed.handle_key(KeyEvent::new(KeyCode::Char('b'), KeyModifiers::CONTROL));
        assert_eq!(
            ed.cursor(),
            (0, 1),
            "Default ctrl+b was overridden and released, no longer matches"
        );
        ed.handle_key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
        assert_eq!(
            ed.cursor(),
            (0, 1),
            "Default left was overridden and released, no longer matches"
        );
    }

    #[test]
    fn user_override_submit_and_newline_keep_backslash_fallbacks() {
        // Submit overridden to ctrl+m: enter no longer submits (the `\` fallback follows the action as well)
        let mut ed = e_with_user(HashMap::from([(
            EditorAction::Submit,
            vec![KeyId::parse("ctrl+m").unwrap()],
        )]));
        ed.set_text("ab\\");
        ed.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        // enter is not bound to Submit: the `\` fallback does not trigger and does not submit — enter does nothing (cursor does not move)
        assert_eq!(ed.cursor(), (0, 3));
        // ctrl+m hits the overridden Submit: `\` fallback = backspace + newline
        ed.handle_key(KeyEvent::new(KeyCode::Char('m'), KeyModifiers::CONTROL));
        assert_eq!(ed.text(), "ab\n");
    }

    #[test]
    fn user_unbind_leaves_key_unconsumed() {
        // Empty Vec = explicit unbind: the default undo ctrl+- no longer matches
        let mut ed = e_with_user(HashMap::from([(EditorAction::Undo, Vec::new())]));
        ed.set_text("ab");
        ed.handle_key(KeyEvent::new(KeyCode::Char('-'), KeyModifiers::CONTROL));
        assert_eq!(
            ed.text(),
            "ab",
            "After unbinding, ctrl+- must not trigger undo (undo should restore to a)"
        );
    }
}
