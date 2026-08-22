// REQ-TUI-01 (pi-413): editor component — submit, history navigation, and session state (history browse/exit/backslash fallback).
// Consumed by input.rs (Enter family, Submit action, ↑↓ history movement) and text.rs (edit operations exiting browse),
// hence the pub(super) methods; set_text_without_history is used only inside this module and stays private.
use super::{Editor, EditorState};

impl Editor {
    /// Submit: take full text trim(), clear state and history browsing state, call on_change("") then on_submit(text).
    /// (No paste marker; expansion step omitted; undo/kill cleanup is hooked up by T-TU-03-5/-7.)
    pub(super) fn submit(&mut self) {
        let text = self.text().trim().to_string();

        // Clear editor state and browsing state (aligned with pi submitValue: exitHistoryBrowsing clears history_index/draft)
        self.state = EditorState::default();
        self.exit_history_browsing();
        self.preferred_visual_col = None;
        self.snapped_from_cursor_col = None;
        self.scroll_offset = 0;

        // Call on_change("") first (notify clear), then on_submit(text) (pass content)
        if let Some(cb) = &mut self.on_change {
            cb("");
        }
        if let Some(cb) = &mut self.on_submit {
            cb(&text);
        }
    }

    /// History browsing: direction=-1 older (up), +1 newer (down). Index semantics: 0=most recent, 1=older
    /// (History prepended, aligned with pi navigateHistory's `newIndex = historyIndex - direction`).
    /// On first entering browsing (any direction leaving -1), stash the current draft; on switching back to -1, restore the draft.
    /// Switching/restoring both trigger on_change (aligned with pi setTextInternal/draft restore path).
    pub(super) fn navigate_history(&mut self, direction: i8) {
        if self.history.is_empty() {
            return;
        }

        let max_index = self.history.len() as isize - 1;

        // Align with pi: newIndex = historyIndex - direction (up = -1 → index increases = older)
        let new_index = self.history_index - direction as isize;

        // Clamp out-of-bounds
        if new_index < -1 || new_index > max_index {
            return;
        }

        // First entering browsing (first move out of -1, regardless of direction): stash the current draft
        if self.history_index == -1 && new_index >= 0 {
            self.history_draft = Some(self.state.clone());
        }

        self.history_index = new_index;

        if self.history_index == -1 {
            // Switch back to not-browsed: restore draft (align with pi restore path, reset cursor/scroll state)
            if let Some(draft) = self.history_draft.take() {
                self.state = draft;
                self.preferred_visual_col = None;
                self.snapped_from_cursor_col = None;
                self.scroll_offset = 0;
            } else {
                // No draft (abnormal path): clear, align with pi setTextInternal("")
                self.set_text_without_history("");
            }
            self.emit_change();
        } else {
            // Show history entry (cursor at end of line, aligning with setText behavior of pi navigateHistory)
            let text = self.history[self.history_index as usize].clone();
            self.set_text_without_history(&text);
            self.emit_change();
        }
    }

    /// Exit history browsing state: called when an edit action starts,
    /// preventing subsequent ↑/↓ from misinterpreting the edited state as history browsing.
    pub(super) fn exit_history_browsing(&mut self) {
        self.history_index = -1;
        self.history_draft = None;
    }

    /// set_text internal version: does not trigger add_to_history, avoiding duplicate pushes while browsing history.
    /// History entries are already first-class text in the editor (normalized during insert/paste — no \r, no \t),
    /// no further normalization needed (align with pi setTextInternal: display as-is, ensuring display == storage).
    fn set_text_without_history(&mut self, text: &str) {
        self.state.lines = text.split('\n').map(str::to_string).collect();
        if self.state.lines.is_empty() {
            self.state.lines.push(String::new());
        }
        // Move cursor to end of last line (aligning with setText behavior of pi navigateHistory)
        self.state.cursor_line = self.state.lines.len() - 1;
        self.state.cursor_col = self.state.lines[self.state.cursor_line].len();
        self.preferred_visual_col = None;
        self.snapped_from_cursor_col = None;
        self.scroll_offset = 0; // Align with pi setTextInternal reset of scrollOffset
    }

    /// Shared newline entry point for Shift+Enter / Ctrl+J: when the char before the cursor is `\` and submit is not disabled
    /// Backspace + submit (aligning with pi shouldSubmitOnBackslashEnter → backspace + submitValue),
    /// Otherwise insert a newline normally. Triggered by input::handle_key's NewLine.
    pub(super) fn handle_newline(&mut self) {
        if !self.disable_submit && self.char_before_cursor_is_backslash() {
            self.backspace();
            self.submit();
        } else {
            self.newline();
        }
    }

    /// Whether the char before the cursor is a backslash (terminal fallback check for `\` + Enter, aligned with pi shouldSubmitOnBackslashEnter).
    /// cursor_col is always at a grapheme boundary: the previous byte is the last byte of the previous grapheme; a continuation byte of a multi-byte grapheme is never b'\\', so the check is safe.
    pub(super) fn char_before_cursor_is_backslash(&self) -> bool {
        self.state.cursor_col > 0
            && self.state.lines[self.state.cursor_line]
                .as_bytes()
                .get(self.state.cursor_col - 1)
                == Some(&b'\\')
    }

    /// Whether the editor is empty (single blank line), aligned with pi isEditorEmpty (↑ history navigation check in input::handle_action).
    pub(super) fn is_editor_empty(&self) -> bool {
        self.state.lines.len() == 1 && self.state.lines[0].is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::super::Editor;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Helper: create a shared capture vec for callbacks.
    fn capture() -> Rc<RefCell<Vec<String>>> {
        Rc::new(RefCell::new(Vec::new()))
    }

    // ── submit ──

    #[test]
    fn test_submit_trims_text_and_calls_on_submit() {
        let mut editor = Editor::new(0);
        let submitted = capture();
        {
            let s = submitted.clone();
            editor.on_submit = Some(Box::new(move |text: &str| {
                s.borrow_mut().push(text.to_string());
            }));
        }
        editor.insert_text_at_cursor("  hello world  ");
        editor.submit();
        assert_eq!(*submitted.borrow(), vec!["hello world"]);
    }

    #[test]
    fn test_submit_clears_editor_state() {
        let mut editor = Editor::new(0);
        editor.on_submit = Some(Box::new(|_| {}));
        editor.insert_text_at_cursor("some text");
        editor.submit();
        assert_eq!(editor.text(), "");
        assert_eq!(editor.cursor(), (0, 0));
    }

    #[test]
    fn test_submit_calls_on_change_with_empty_before_on_submit() {
        let mut editor = Editor::new(0);
        let changes = capture();
        let submitted = capture();
        {
            let c = changes.clone();
            editor.on_change = Some(Box::new(move |text: &str| {
                c.borrow_mut().push(text.to_string());
            }));
            let s = submitted.clone();
            editor.on_submit = Some(Box::new(move |text: &str| {
                s.borrow_mut().push(text.to_string());
            }));
        }
        editor.insert_text_at_cursor("hello");
        // Clear changes from insert
        changes.borrow_mut().clear();

        editor.submit();
        // on_change fires with empty string before on_submit
        assert_eq!(changes.borrow().len(), 1);
        assert_eq!(changes.borrow()[0], "");
        assert_eq!(*submitted.borrow(), vec!["hello"]);
    }

    #[test]
    fn test_submit_resets_history_browsing() {
        let mut editor = Editor::new(0);
        editor.on_submit = Some(Box::new(|_| {}));
        editor.add_to_history("old1");
        editor.add_to_history("old2");
        editor.insert_text_at_cursor("current");

        // Enter history browsing
        editor.navigate_history(-1); // go to older
        assert_ne!(editor.history_index, -1);

        editor.submit();
        assert_eq!(editor.history_index, -1);
        assert!(editor.history_draft.is_none());
    }

    #[test]
    fn test_submit_with_multiline_text() {
        let mut editor = Editor::new(0);
        let submitted = capture();
        {
            let s = submitted.clone();
            editor.on_submit = Some(Box::new(move |text: &str| {
                s.borrow_mut().push(text.to_string());
            }));
        }
        editor.insert_text_at_cursor("line1\nline2\n  line3  ");
        editor.submit();
        // trim() on multiline preserves internal whitespace
        assert_eq!(*submitted.borrow(), vec!["line1\nline2\n  line3"]);
    }

    // ── navigate_history ──

    #[test]
    fn test_navigate_history_empty_does_nothing() {
        let mut editor = Editor::new(0);
        editor.insert_text_at_cursor("hello");
        editor.navigate_history(-1);
        // Should not change text since history is empty
        assert_eq!(editor.text(), "hello");
    }

    #[test]
    fn test_navigate_history_go_to_older() {
        let mut editor = Editor::new(0);
        editor.add_to_history("first");
        editor.add_to_history("second");
        editor.insert_text_at_cursor("current");

        editor.navigate_history(-1); // direction -1 = older → index 0 = "second"
        assert_eq!(editor.text(), "second");
        assert_eq!(editor.history_index, 0);
    }

    #[test]
    fn test_navigate_history_go_to_older_twice() {
        let mut editor = Editor::new(0);
        editor.add_to_history("first");
        editor.add_to_history("second");
        editor.insert_text_at_cursor("current");

        editor.navigate_history(-1); // index 0 = "second"
        editor.navigate_history(-1); // index 1 = "first"
        assert_eq!(editor.text(), "first");
        assert_eq!(editor.history_index, 1);
    }

    #[test]
    fn test_navigate_history_clamp_at_oldest() {
        let mut editor = Editor::new(0);
        editor.add_to_history("only");
        editor.insert_text_at_cursor("current");

        editor.navigate_history(-1); // index 0 = "only"
        editor.navigate_history(-1); // already at max, should clamp
        assert_eq!(editor.text(), "only");
        assert_eq!(editor.history_index, 0);
    }

    #[test]
    fn test_navigate_history_go_back_to_newer() {
        let mut editor = Editor::new(0);
        editor.add_to_history("first");
        editor.add_to_history("second");
        editor.insert_text_at_cursor("current");

        editor.navigate_history(-1); // index 0 = "second"
        editor.navigate_history(-1); // index 1 = "first"
        editor.navigate_history(1); // direction +1 = newer → index 0 = "second"
        assert_eq!(editor.text(), "second");
        assert_eq!(editor.history_index, 0);
    }

    #[test]
    fn test_navigate_history_restore_draft_when_going_back_to_minus_1() {
        let mut editor = Editor::new(0);
        editor.add_to_history("old");
        editor.insert_text_at_cursor("my draft");

        editor.navigate_history(-1); // enter browsing, draft saved
        assert_eq!(editor.text(), "old");

        editor.navigate_history(1); // go back to -1, restore draft
        assert_eq!(editor.text(), "my draft");
        assert_eq!(editor.history_index, -1);
    }

    #[test]
    fn test_navigate_history_clamp_past_minus_1() {
        let mut editor = Editor::new(0);
        editor.add_to_history("old");
        editor.insert_text_at_cursor("current");

        // Already at -1, going newer should clamp
        editor.navigate_history(1);
        assert_eq!(editor.text(), "current");
        assert_eq!(editor.history_index, -1);
    }

    #[test]
    fn test_navigate_history_triggers_on_change() {
        let mut editor = Editor::new(0);
        let changes = capture();
        {
            let c = changes.clone();
            editor.on_change = Some(Box::new(move |text: &str| {
                c.borrow_mut().push(text.to_string());
            }));
        }
        editor.add_to_history("hist");
        editor.insert_text_at_cursor("draft");
        changes.borrow_mut().clear();

        editor.navigate_history(-1);
        assert_eq!(*changes.borrow(), vec!["hist"]);
    }

    #[test]
    fn test_navigate_history_cursor_at_end_of_history_entry() {
        let mut editor = Editor::new(0);
        editor.add_to_history("hello");
        editor.insert_text_at_cursor("draft");

        editor.navigate_history(-1);
        assert_eq!(editor.cursor(), (0, 5)); // cursor at end of "hello"
    }

    #[test]
    fn test_navigate_history_multiline_entry() {
        let mut editor = Editor::new(0);
        // Manually add multiline history entry
        editor.history.push("line1\nline2".to_string());
        editor.insert_text_at_cursor("draft");

        editor.navigate_history(-1);
        assert_eq!(editor.text(), "line1\nline2");
        assert_eq!(editor.cursor(), (1, 5)); // cursor at end of last line
    }

    // ── exit_history_browsing ──

    #[test]
    fn test_exit_history_browsing_resets_state() {
        let mut editor = Editor::new(0);
        editor.add_to_history("old");
        editor.insert_text_at_cursor("draft");
        editor.navigate_history(-1);
        assert_ne!(editor.history_index, -1);

        editor.exit_history_browsing();
        assert_eq!(editor.history_index, -1);
        assert!(editor.history_draft.is_none());
    }

    #[test]
    fn test_exit_history_browsing_when_not_browsing() {
        let mut editor = Editor::new(0);
        // Should be a no-op
        editor.exit_history_browsing();
        assert_eq!(editor.history_index, -1);
        assert!(editor.history_draft.is_none());
    }

    // ── handle_newline ──

    #[test]
    fn test_handle_newline_inserts_newline_normally() {
        let mut editor = Editor::new(0);
        editor.insert_text_at_cursor("hello");
        editor.handle_newline();
        assert_eq!(editor.text(), "hello\n");
    }

    #[test]
    fn test_handle_newline_backslash_triggers_submit() {
        let mut editor = Editor::new(0);
        let submitted = capture();
        {
            let s = submitted.clone();
            editor.on_submit = Some(Box::new(move |text: &str| {
                s.borrow_mut().push(text.to_string());
            }));
        }
        editor.insert_text_at_cursor("hello\\");
        editor.handle_newline();
        // Backslash removed, then submitted
        assert_eq!(*submitted.borrow(), vec!["hello"]);
        assert_eq!(editor.text(), "");
    }

    #[test]
    fn test_handle_newline_backslash_no_submit_when_disabled() {
        let mut editor = Editor::new(0);
        editor.disable_submit = true;
        let submitted = capture();
        {
            let s = submitted.clone();
            editor.on_submit = Some(Box::new(move |text: &str| {
                s.borrow_mut().push(text.to_string());
            }));
        }
        editor.insert_text_at_cursor("hello\\");
        editor.handle_newline();
        // submit disabled, so newline inserted instead
        assert!(submitted.borrow().is_empty());
        assert_eq!(editor.text(), "hello\\\n");
    }

    #[test]
    fn test_handle_newline_no_backslash_inserts_newline() {
        let mut editor = Editor::new(0);
        let submitted = capture();
        {
            let s = submitted.clone();
            editor.on_submit = Some(Box::new(move |text: &str| {
                s.borrow_mut().push(text.to_string());
            }));
        }
        editor.insert_text_at_cursor("hello");
        editor.handle_newline();
        assert!(submitted.borrow().is_empty());
        assert_eq!(editor.text(), "hello\n");
    }

    // ── char_before_cursor_is_backslash ──

    #[test]
    fn test_char_before_cursor_is_backslash_true() {
        let mut editor = Editor::new(0);
        editor.insert_text_at_cursor("abc\\");
        assert!(editor.char_before_cursor_is_backslash());
    }

    #[test]
    fn test_char_before_cursor_is_backslash_false() {
        let mut editor = Editor::new(0);
        editor.insert_text_at_cursor("abc");
        assert!(!editor.char_before_cursor_is_backslash());
    }

    #[test]
    fn test_char_before_cursor_is_backslash_at_start() {
        let editor = Editor::new(0);
        // Empty editor, cursor at (0,0)
        assert!(!editor.char_before_cursor_is_backslash());
    }

    // ── is_editor_empty ──

    #[test]
    fn test_is_editor_empty_true() {
        let editor = Editor::new(0);
        assert!(editor.is_editor_empty());
    }

    #[test]
    fn test_is_editor_empty_false_with_text() {
        let mut editor = Editor::new(0);
        editor.insert_text_at_cursor("hello");
        assert!(!editor.is_editor_empty());
    }

    #[test]
    fn test_is_editor_empty_false_with_multiple_lines() {
        let mut editor = Editor::new(0);
        editor.insert_text_at_cursor("\n");
        assert!(!editor.is_editor_empty());
    }

    // ── set_text_without_history (private, tested indirectly via navigate_history) ──

    #[test]
    fn test_set_text_without_history_via_navigate_resets_scroll() {
        let mut editor = Editor::new(0);
        editor.scroll_offset = 5;
        editor.add_to_history("hist");
        editor.insert_text_at_cursor("draft");

        editor.navigate_history(-1);
        assert_eq!(editor.scroll_offset, 0);
    }

    #[test]
    fn test_set_text_without_history_via_navigate_resets_preferred_col() {
        let mut editor = Editor::new(0);
        editor.preferred_visual_col = Some(42);
        editor.snapped_from_cursor_col = Some(10);
        editor.add_to_history("hist");
        editor.insert_text_at_cursor("draft");

        editor.navigate_history(-1);
        assert!(editor.preferred_visual_col.is_none());
        assert!(editor.snapped_from_cursor_col.is_none());
    }
}
