// REQ-TUI-03 (pi-406): Editor undo stack — snapshot primitive + merged word/character state + undo execution.
// The generic `UndoStack<S>` (pi undo-stack.ts) is implemented in Rust using `Vec<EditorState>` (EditorState is already Clone,
// with equivalent clone-on-push semantics), without introducing a wrapper type; snapshots store only text+cursor, while view state (scroll_offset/
// preferred_visual_col/snapped_from_cursor_col etc.) is not pushed onto the stack and is rebuilt according to pi semantics during undo.
// Edit operations in text.rs/history.rs/mod.rs conditionally invoke snapshot pushes per pi (see tasks/T-TU-03-5-implemented.md
// of the "snapshot push-point list"), this module provides only primitive methods + a merge-state enum.
use super::Editor;

/// Previous input action category (aligned with the semantics of the pi `lastAction` union).
/// T-TU-03-5 uses only `TypeWord` for word-character insertion merge detection; `kill`/`yank` variants are not implemented due to T-TU-03-7 Drop.
/// Merge detection only cares whether it is a word-character insertion, so all non-insertion paths in this round set `None`
/// (Setting `"kill"` for delete operations is fully equivalent to pi for undo behavior; with no kill ring (T-TU-03-7 Drop), no extension is needed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LastAction {
    /// The previous action was a word-character insertion (`insert_char` set): consecutive word characters are merged into one undo unit.
    TypeWord,
}

impl Editor {
    /// Clone the current EditorState onto the top of the undo stack (save a rollback point before an edit changes the text; aligned with pi's
    /// pushUndoSnapshot/pushUndo "push snapshot before change" convention). Store only the text + cursor (EditorState),
    /// View state is not pushed onto the stack—reconstructed according to pi semantics during undo. Callers: the 9 editing operations in text.rs,
    /// set_text in mod.rs (push only if the content differs), and navigate_history in history.rs (on first entering browse mode)
    /// Call before actually modifying state. Stack capacity is unlimited, consistent with pi (cleared on submit, with no trimming).
    /// Shared across submodules (all editing operations in text.rs/history.rs/mod.rs call it), so it is pub(super).
    pub(super) fn push_undo_snapshot(&mut self) {
        self.undo_stack.push(self.state.clone());
    }

    /// Perform one undo: exit history browsing → pop the stack to restore EditorState → reset last_action and
    /// preferred_visual_col/snapped_from_cursor_col → emit_change (align with pi undo:
    /// An empty stack returns directly with no side effects and does not trigger on_change; pi only resets preferredVisualCol,
    /// zen's submit/navigate convention likewise resets snapped_from_cursor_col; reset it as well to avoid
    /// stale snapping state). scroll_offset is not explicitly reset; render clamps it to the cursor (matching pi, which does not reset it).
    /// Consumer: the EditorAction::Undo branch of input.rs::handle_action (user presses Ctrl+-), so pub(super).
    pub(super) fn undo(&mut self) {
        self.exit_history_browsing();
        let Some(snapshot) = self.undo_stack.pop() else {
            return;
        };
        self.state = snapshot;
        self.last_action = None;
        self.preferred_visual_col = None;
        self.snapped_from_cursor_col = None;
        self.emit_change();
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
    use super::super::{Editor, EditorAction};

    // ── Basic undo stack semantics (driven through the public entry point handle_action) ──

    #[test]
    fn test_undo_empty_stack_noop() {
        let mut e = e(0);
        // Empty stack: no operation, no on_change
        e.handle_action(EditorAction::Undo);
        assert_eq!(e.text(), "");
        assert_eq!(e.undo_stack.len(), 0);
    }

    #[test]
    fn test_undo_restores_text_and_cursor() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        // Move the cursor to the start of the line, then insert (creating a second undo unit)
        e.handle_action(EditorAction::MoveToLineStart);
        e.insert_char('X');
        assert_eq!(e.text(), "Xhello");

        // Undo the insertion of X
        e.handle_action(EditorAction::Undo);
        assert_eq!(e.text(), "hello");
        assert_eq!(e.cursor(), (0, 0));

        // Undo the entire insertion again
        e.handle_action(EditorAction::Undo);
        assert_eq!(e.text(), "");
    }

    #[test]
    fn test_undo_exits_history_browsing() {
        let mut e = e(0);
        e.add_to_history("old");
        e.insert_text_at_cursor("draft");
        e.navigate_history(-1);
        assert_ne!(e.history_index, -1);

        e.handle_action(EditorAction::Undo);
        assert_eq!(e.history_index, -1);
        assert!(e.history_draft.is_none());
    }

    #[test]
    fn test_undo_fires_on_change() {
        let mut e = e(0);
        let changes = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        {
            let c = changes.clone();
            e.on_change = Some(Box::new(move |text: &str| {
                c.borrow_mut().push(text.to_string());
            }));
        }
        e.insert_char('a');
        changes.borrow_mut().clear();

        e.handle_action(EditorAction::Undo);
        assert_eq!(*changes.borrow(), vec![""]);
    }

    #[test]
    fn test_undo_resets_view_state() {
        let mut e = e(0);
        e.insert_char('a');
        e.preferred_visual_col = Some(7);
        e.snapped_from_cursor_col = Some(3);

        e.handle_action(EditorAction::Undo);
        assert!(e.preferred_visual_col.is_none());
        assert!(e.snapped_from_cursor_col.is_none());
    }

    // ── Word-character merging (fish style, aligned with pi insertCharacter) ──

    #[test]
    fn test_undo_coalesces_consecutive_word_chars() {
        let mut e = e(0);
        // Consecutive word characters: one undo unit
        for c in "hello".chars() {
            e.insert_char(c);
        }
        assert_eq!(e.undo_stack.len(), 1);

        e.handle_action(EditorAction::Undo);
        assert_eq!(e.text(), "");
    }

    #[test]
    fn test_undo_space_is_separate_undoable() {
        let mut e = e(0);
        for c in "hello ".chars() {
            e.insert_char(c);
        }
        // 'h'..'o' merged into one unit; the space captures the state before itself → two units
        assert_eq!(e.undo_stack.len(), 2);

        // First undo: removes the space + subsequent word (here, the space itself)
        e.handle_action(EditorAction::Undo);
        assert_eq!(e.text(), "hello");
        // Second undo: removes hello
        e.handle_action(EditorAction::Undo);
        assert_eq!(e.text(), "");
    }

    #[test]
    fn test_undo_cursor_move_breaks_coalescing() {
        let mut e = e(0);
        for c in "hel".chars() {
            e.insert_char(c);
        }
        // Continue typing after moving the cursor: a new undo unit
        e.handle_action(EditorAction::MoveToLineStart);
        e.insert_char('X');
        assert_eq!(e.undo_stack.len(), 2);

        e.handle_action(EditorAction::Undo);
        assert_eq!(e.text(), "hel");
        e.handle_action(EditorAction::Undo);
        assert_eq!(e.text(), "");
    }

    // ── Each editing operation pushes a snapshot ──

    #[test]
    fn test_undo_delete_operations() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello world");

        // Ctrl+U (delete to start of line): cursor at end of line → delete the entire line; Ctrl+- restores (including cursor position)
        e.handle_action(EditorAction::DeleteToLineStart);
        assert_eq!(e.text(), "");
        e.handle_action(EditorAction::Undo);
        assert_eq!(e.text(), "hello world");
        assert_eq!(e.cursor(), (0, 11));

        // Move the cursor to the start of the line, Ctrl+K (delete to end of line) → delete all; Ctrl+- restores
        e.handle_action(EditorAction::MoveToLineStart);
        e.handle_action(EditorAction::DeleteToLineEnd);
        assert_eq!(e.text(), "");
        e.handle_action(EditorAction::Undo);
        assert_eq!(e.text(), "hello world");
        assert_eq!(e.cursor(), (0, 0));
    }

    #[test]
    fn test_undo_noop_delete_does_not_push() {
        let mut e = e(0);
        // (0,0) Empty editor, Backspace: no actual deletion → no snapshot is pushed (aligns with pi's no-op guard)
        e.handle_action(EditorAction::DeleteCharBackward);
        assert_eq!(e.undo_stack.len(), 0);
    }

    #[test]
    fn test_undo_paste_is_atomic() {
        let mut e = e(0);
        e.handle_paste("a\nb");
        assert_eq!(e.text(), "a\nb");
        assert_eq!(e.undo_stack.len(), 1);

        e.handle_action(EditorAction::Undo);
        assert_eq!(e.text(), "");
    }

    #[test]
    fn test_undo_newline_is_atomic() {
        let mut e = e(0);
        e.insert_text_at_cursor("abc");
        e.handle_action(EditorAction::NewLine);
        assert_eq!(e.undo_stack.len(), 2);

        e.handle_action(EditorAction::Undo);
        assert_eq!(e.text(), "abc");
    }

    #[test]
    fn test_submit_clears_undo_stack() {
        let mut e = e(0);
        e.on_submit = Some(Box::new(|_| {}));
        e.insert_char('a');
        assert_eq!(e.undo_stack.len(), 1);

        e.handle_action(EditorAction::Submit);
        assert_eq!(e.undo_stack.len(), 0);

        // Undo after commit: no operation
        e.handle_action(EditorAction::Undo);
        assert_eq!(e.text(), "");
    }

    #[test]
    fn test_set_text_different_pushes_snapshot() {
        let mut e = e(0);
        e.set_text("first version");
        assert!(!e.text().is_empty());
        assert_eq!(e.undo_stack.len(), 1);

        // Programmatic changes are undoable
        e.handle_action(EditorAction::Undo);
        assert_eq!(e.text(), "");
    }

    #[test]
    fn test_set_text_same_content_no_push() {
        let mut e = e(0);
        e.set_text("");
        assert_eq!(e.undo_stack.len(), 0);

        e.insert_char('a');
        let n = e.undo_stack.len();
        // Set the same content: not pushed onto the stack
        e.set_text("a");
        assert_eq!(e.undo_stack.len(), n);
    }

    #[test]
    fn test_navigate_history_entry_protects_draft() {
        let mut e = e(0);
        e.add_to_history("old");
        e.insert_text_at_cursor("my draft");
        let n = e.undo_stack.len();

        e.navigate_history(-1);
        assert_eq!(e.text(), "old");
        // First entry into browse mode: a rollback point was pushed for the original document
        assert_eq!(e.undo_stack.len(), n + 1);

        e.handle_action(EditorAction::Undo);
        assert_eq!(e.text(), "my draft");
        assert_eq!(e.history_index, -1);
    }

    #[test]
    fn test_undo_multiline_mid_edit() {
        let mut e = e(0);
        e.insert_text_at_cursor("a\nb");
        // After insertion, the cursor moves from the end of the last line (1,1) → the start of the line (1,0); insert X
        e.handle_action(EditorAction::MoveToLineStart);
        e.insert_char('X');
        assert_eq!(e.text(), "a\nXb");
        assert_eq!(e.cursor(), (1, 1));

        e.handle_action(EditorAction::Undo);
        assert_eq!(e.text(), "a\nb");
        assert_eq!(e.cursor(), (1, 0));
    }
}
