// REQ-TUI-03 (pi-406): Emacs-style kill ring backing `Ctrl+Y` (yank) and `Alt+Y` (yank-pop).
//
// Scope decision: the kill ring is captured at the *action* boundary instead of inside every delete
// primitive. Each kill action snapshots the buffer before and after, and the removed slice is derived
// from the common prefix/suffix — the same result for single-line and multi-line kills, without
// threading return values through `text.rs`. `YankPop` undoes the previous yank (its snapshot is
// already on the undo stack) and yanks the previous ring entry, which is behaviourally equivalent to
// Emacs' replace-the-last-yank for the cases the editor can reach.
use super::{Editor, LastAction};

/// Upper bound on retained kills. Beyond this the oldest entry is dropped; Emacs' default is 60.
const KILL_RING_CAP: usize = 60;

impl Editor {
    /// Append a removed slice to the kill ring, dropping the oldest entry past the cap.
    pub(super) fn kill_ring_push(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if self.kill_ring.len() == KILL_RING_CAP {
            self.kill_ring.remove(0);
        }
        self.kill_ring.push(text.to_string());
        self.yank_index = self.kill_ring.len() - 1;
    }

    /// Run a delete operation and record whatever it removed on the kill ring.
    ///
    /// `kill` must only delete text (no insertion); the removed slice is derived from the buffer
    /// before/after, so multi-line joins (`Ctrl+U` at column 0) are captured as well.
    pub(super) fn kill_with_capture<F: FnOnce(&mut Self)>(&mut self, kill: F) {
        let before = self.text();
        kill(self);
        let after = self.text();
        if before != after {
            if let Some(removed) = removed_segment(&before, &after) {
                self.kill_ring_push(&removed);
            }
        }
    }

    /// Insert the most recent kill at the cursor (`Ctrl+Y`).
    ///
    /// No-op on an empty ring. Marks `last_action` as `Yank` so that an immediately following
    /// [`Self::yank_pop`] knows it may undo the insertion.
    pub(super) fn yank(&mut self) {
        let Some(text) = self.kill_ring.last().cloned() else {
            return;
        };
        self.insert_text(&text);
        self.yank_index = self.kill_ring.len() - 1;
        self.last_action = Some(LastAction::Yank);
    }

    /// Replace the text inserted by the previous yank with the next older ring entry (`Alt+Y`).
    ///
    /// Requires the previous action to have been a yank (so an undo snapshot of the pre-yank buffer
    /// is on top of the stack) and at least two ring entries; otherwise it is a no-op, matching
    /// Emacs' "yank-pop after anything else does nothing" rule.
    pub(super) fn yank_pop(&mut self) {
        if self.last_action != Some(LastAction::Yank) || self.kill_ring.len() < 2 {
            return;
        }
        // Undo the previous yank, then insert the entry before the one currently yanked.
        self.undo();
        let next = if self.yank_index == 0 {
            self.kill_ring.len() - 1
        } else {
            self.yank_index - 1
        };
        let text = self.kill_ring[next].clone();
        self.insert_text(&text);
        self.yank_index = next;
        self.last_action = Some(LastAction::Yank);
    }

    /// Read-only view of the kill ring, oldest first (tests and diagnostics).
    pub fn kill_ring(&self) -> &[String] {
        &self.kill_ring
    }
}

/// The contiguous slice that turns `before` into `after`, or `None` when `after` is not a pure
/// deletion (which would mean a caller misused [`Editor::kill_with_capture`]).
fn removed_segment(before: &str, after: &str) -> Option<String> {
    if before.len() <= after.len() {
        return None;
    }
    let (b, a) = (before.as_bytes(), after.as_bytes());

    // Longest common prefix, trimmed back to a char boundary.
    let mut prefix = 0;
    while prefix < a.len() && b[prefix] == a[prefix] {
        prefix += 1;
    }
    while !before.is_char_boundary(prefix) {
        prefix -= 1;
    }

    // Longest common suffix that does not overlap the prefix.
    let mut suffix = 0;
    while suffix < a.len() - prefix && b[b.len() - 1 - suffix] == a[a.len() - 1 - suffix] {
        suffix += 1;
    }
    while !before.is_char_boundary(before.len() - suffix) {
        suffix -= 1;
    }

    let removed = &before[prefix..before.len() - suffix];
    if removed.is_empty() {
        None
    } else {
        Some(removed.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keybindings::{editor_default_bindings, KeybindingsManager};
    use std::collections::HashMap;

    /// Test helper: editor with the default keymap and the given initial text.
    fn test_editor(text: &str) -> Editor {
        let mut e = Editor::new(
            0,
            KeybindingsManager::new(editor_default_bindings(), HashMap::new()),
        );
        e.set_text(text);
        e
    }

    /// Kill `Ctrl+K` at the cursor and return the ring.
    fn kill_to_end(editor: &mut Editor) {
        editor.kill_with_capture(|e| e.delete_to_line_end());
    }

    #[test]
    fn kill_ring_captures_deleted_text() {
        let mut e = test_editor("hello world");
        e.set_cursor_col(5);
        kill_to_end(&mut e);
        assert_eq!(e.text(), "hello");
        assert_eq!(e.kill_ring(), [" world"]);
    }

    #[test]
    fn yank_inserts_most_recent_kill() {
        let mut e = test_editor("hello world");
        e.set_cursor_col(5);
        kill_to_end(&mut e);
        e.set_cursor_col(0);
        e.yank();
        assert_eq!(e.text(), " worldhello");
    }

    #[test]
    fn yank_is_noop_with_empty_ring() {
        let mut e = test_editor("abc");
        e.yank();
        assert_eq!(e.text(), "abc");
    }

    #[test]
    fn yank_pop_replaces_previous_yank() {
        let mut e = test_editor("one two");
        // Two kills: "one" then " two"
        e.set_cursor_col(3);
        e.kill_with_capture(|ed| ed.delete_to_line_end()); // ring: [" two"]
        e.set_cursor_col(0);
        e.kill_with_capture(|ed| ed.delete_to_line_start()); // col 0 → no-op capture
        e.set_cursor_col(3);
        e.kill_with_capture(|ed| ed.delete_to_line_start()); // ring: [" two", "one"]
        assert_eq!(e.text(), "");
        e.yank(); // inserts "one"
        assert_eq!(e.text(), "one");
        e.yank_pop(); // replaces with " two"
        assert_eq!(e.text(), " two");
    }

    #[test]
    fn yank_pop_without_prior_yank_is_noop() {
        let mut e = test_editor("abc");
        e.kill_ring_push("xyz");
        e.kill_ring_push("uvw");
        e.yank_pop();
        assert_eq!(e.text(), "abc", "yank-pop only follows a yank");
    }

    #[test]
    fn kill_ring_captures_multiline_join() {
        let mut e = test_editor("ab\ncd");
        e.set_cursor_col(0);
        e.set_cursor(1, 0);
        e.kill_with_capture(|ed| ed.delete_to_line_start()); // joins with previous line
        assert_eq!(e.text(), "abcd");
        assert_eq!(e.kill_ring(), ["\n"]);
    }

    #[test]
    fn kill_ring_is_capped() {
        let mut e = test_editor("");
        for i in 0..KILL_RING_CAP + 5 {
            e.kill_ring_push(&format!("k{i}"));
        }
        assert_eq!(e.kill_ring().len(), KILL_RING_CAP);
        assert_eq!(e.kill_ring().last().unwrap(), "k64");
        assert_eq!(e.kill_ring().first().unwrap(), "k5");
    }
}
