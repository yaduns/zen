// REQ-TUI-01 (pi-413): Editor component — text editing operations (insert/delete/line break).
// All via mutating state.lines and triggering on_change; consumed by input.rs (action routing) and history.rs (backslash fallback),
// so methods are pub(super); snap_cursor_to_grapheme_boundary is only used within this module and stays private.
// Undo (T-TU-03-5): each operation pushes a snapshot per pi's condition before actually mutating state (no-ops don't push),
// and resets the last_action merge state; insert_char implements fish-style word-character merging.
use super::{Editor, LastAction};
use unicode_segmentation::UnicodeSegmentation;

impl Editor {
    /// Insert a single grapheme at the cursor and move the cursor right; when the inserted character merges with the preceding character into a single grapheme (combining characters),
    /// snap to the end of the merged grapheme (maintaining the "cursor is always at a grapheme boundary" invariant). Triggers on_change.
    /// Undo coalescing (fish style, aligned with pi insertCharacter): consecutive word characters merge into one undo unit;
    /// whitespace captures state ahead of itself (one undo also removes the word after the space); each space is individually undoable.
    pub(super) fn insert_char(&mut self, c: char) {
        self.exit_history_browsing();

        // Merge decision: only whitespace (or after a non-insert operation) pushes a new snapshot (aligned with pi:
        // isWhitespaceChar(char) || lastAction !== "type-word" → pushUndoSnapshot）
        if c.is_whitespace() || self.last_action != Some(LastAction::TypeWord) {
            self.push_undo_snapshot();
        }
        self.last_action = Some(LastAction::TypeWord);

        let col = self.state.cursor_col;
        {
            let line = &mut self.state.lines[self.state.cursor_line];
            // cursor_col is always at a grapheme (and thus character) boundary; insert inserts at the boundary
            line.insert(col, c);
        }

        // Move the cursor right by the byte length of one character, then snap to the grapheme boundary (combining characters merged via insert may cross the boundary)
        self.set_cursor_col(col + c.len_utf8());
        self.snap_cursor_to_grapheme_boundary();

        // Fire the change callback
        self.emit_change();
    }

    /// Snap the cursor to the end of its containing grapheme, maintaining the "cursor is always at a grapheme boundary" invariant.
    /// No-op when the cursor is already at the start of a grapheme or at end of line; only advances when a combining character merge leaves the cursor mid-grapheme.
    fn snap_cursor_to_grapheme_boundary(&mut self) {
        let col = self.state.cursor_col;
        let line = &self.state.lines[self.state.cursor_line];
        // Already at end of line or exactly at a grapheme start: no snapping needed
        if col == line.len() || line.grapheme_indices(true).any(|(bo, _)| bo == col) {
            return;
        }
        // col falls inside a grapheme: snap to the end of that grapheme
        let snapped = line
            .grapheme_indices(true)
            .find(|(bo, g)| *bo <= col && col < bo + g.len())
            .map(|(bo, g)| bo + g.len())
            .unwrap_or(line.len());
        self.set_cursor_col(snapped);
    }

    /// Backspace: delete the grapheme before the cursor; at line start, merge with the previous line (cursor moves to the end of the previous line). Triggers on_change.
    /// Undo: push a snapshot before an actual deletion (in-line deletion or cross-line merge); no-op (line start) doesn't push (aligned with pi).
    pub(super) fn backspace(&mut self) {
        self.exit_history_browsing();
        self.last_action = None;

        let col = self.state.cursor_col;

        if col > 0 {
            self.push_undo_snapshot();
            // Collect only (byte offset, grapheme byte length); no &str borrow
            let graphemes: Vec<(usize, usize)> = self.state.lines[self.state.cursor_line]
                .grapheme_indices(true)
                .map(|(bo, g)| (bo, g.len()))
                .collect();

            if let Some(&(g_start, g_len)) = graphemes.iter().rfind(|(bo, _)| *bo < col) {
                let g_end = g_start + g_len;
                // graphemes no longer borrow self, so mutable borrow is safe
                let line = &mut self.state.lines[self.state.cursor_line];
                line.replace_range(g_start..g_end, "");
                self.set_cursor_col(g_start);
            }

            self.emit_change();
        } else if self.state.cursor_line > 0 {
            // ── Backspace at line start: merge with previous line ──
            self.push_undo_snapshot();
            let current_line = self.state.lines.remove(self.state.cursor_line);
            self.state.cursor_line -= 1;
            let prev_len = self.state.lines[self.state.cursor_line].len();
            self.state.lines[self.state.cursor_line].push_str(&current_line);
            self.set_cursor_col(prev_len);

            self.emit_change();
        }
        // Start of first line: no-op
    }

    /// Delete forward: delete the grapheme at (or containing) the cursor; at line end, merge with the next line. Triggers on_change.
    /// Undo: push a snapshot before an actual deletion; no-op (line end) doesn't push (aligned with pi).
    pub(super) fn delete_forward(&mut self) {
        self.exit_history_browsing();
        self.last_action = None;

        let line_len = self.state.lines[self.state.cursor_line].len();
        let col = self.state.cursor_col;

        if col < line_len {
            // ── Delete forward in line: delete the grapheme at (or containing) the cursor ──
            self.push_undo_snapshot();
            // Condition = grapheme end passes col: delete the grapheme starting at col when the cursor is at a boundary,
            // delete the containing grapheme when the cursor is inside one (symmetric to backspace's rfind(bo < col)).
            let graphemes: Vec<(usize, usize)> = self.state.lines[self.state.cursor_line]
                .grapheme_indices(true)
                .map(|(bo, g)| (bo, g.len()))
                .collect();

            if let Some(&(g_start, g_len)) = graphemes.iter().find(|(bo, len)| *bo + *len > col) {
                let g_end = g_start + g_len;
                let line = &mut self.state.lines[self.state.cursor_line];
                line.replace_range(g_start..g_end, "");
                // Cursor stays put (following graphemes shift left); if the original cursor was inside the deleted grapheme (merge case), snap to the new boundary
            }

            self.snap_cursor_to_grapheme_boundary();
            self.emit_change();
        } else if self.state.cursor_line + 1 < self.state.lines.len() {
            // ── Delete forward at line end: merge with next line ──
            self.push_undo_snapshot();
            let next_line = self.state.lines.remove(self.state.cursor_line + 1);
            self.state.lines[self.state.cursor_line].push_str(&next_line);
            // Cursor stays put (remains at the old line end, i.e. the merge point)

            self.emit_change();
        }
        // End of last line: no-op
    }

    /// Insert multi-line text at the cursor (shared by enter/paste/yank); newlines split lines at the cursor.
    /// Normalize newlines (\r\n/\r → \n) and tabs (\t → 4 spaces, aligned with pi normalizeText),
    /// so this function is the sole insertion entry point; callers need not normalize. Triggers on_change.
    /// Undo atomicity: non-empty insertion always pushes a single snapshot (aligned with pi's skipUndoCoalescing atomic path: paste /
    /// insertTextAtCursor); a single undo restores the state before the entire insertion.
    pub(super) fn insert_text(&mut self, text: &str) {
        self.exit_history_browsing();
        self.last_action = None;
        // Empty text produces no actual change: no snapshot is pushed (aligned with pi insertTextAtCursor's `if (!text) return`)
        if !text.is_empty() {
            self.push_undo_snapshot();
        }

        let col = self.state.cursor_col;
        let normalized = normalize_text(text);
        let parts: Vec<&str> = normalized.split('\n').collect();

        if parts.len() == 1 {
            // ── Single line: insert directly ──
            let line = &mut self.state.lines[self.state.cursor_line];
            line.insert_str(col, parts[0]);
            self.set_cursor_col(col + parts[0].len());
        } else {
            // ── Multiple lines: split and insert ──
            let rest = self.state.lines[self.state.cursor_line][col..].to_string();
            self.state.lines[self.state.cursor_line].truncate(col);

            // First segment: append to end of current line
            self.state.lines[self.state.cursor_line].push_str(parts[0]);

            // Middle segment: insert line by line
            for part in &parts[1..parts.len() - 1] {
                self.state.cursor_line += 1;
                self.state
                    .lines
                    .insert(self.state.cursor_line, part.to_string());
            }

            // Last segment: new line = last segment + original second half
            self.state.cursor_line += 1;
            let last = format!("{}{}", parts[parts.len() - 1], rest);
            self.state.lines.insert(self.state.cursor_line, last);

            // Cursor sits after the last segment's original content (i.e., before the original second half)
            self.set_cursor_col(parts[parts.len() - 1].len());
        }

        // If the inserted content merges with a neighboring character into a single grapheme (combining char), snap to keep the grapheme-boundary invariant
        self.snap_cursor_to_grapheme_boundary();
        self.emit_change();
    }

    /// Line split: truncate the current line at the cursor; the second half becomes a new line and the cursor moves to the new line's start. Triggers on_change.
    /// Undo: always pushes a snapshot (aligned with pi addNewLine); a single undo reverts the entire newline.
    pub(super) fn newline(&mut self) {
        self.exit_history_browsing();
        self.last_action = None;
        self.push_undo_snapshot();

        let col = self.state.cursor_col;
        let line = &mut self.state.lines[self.state.cursor_line];

        // Truncate out the second half
        let rest = line[col..].to_string();
        line.truncate(col);

        // Insert the new line
        self.state.cursor_line += 1;
        self.state.lines.insert(self.state.cursor_line, rest);

        // Move the cursor to the new line's start
        self.set_cursor_col(0);

        self.emit_change();
    }

    /// Delete from cursor to line start (everything before the cursor); at line start, merge into the previous line (aligns with pi deleteToStartOfLine's col==0 cross-line branch).
    /// Trigger on_change. Undo: push a snapshot before actual deletion; a no-op (at the start of the first line) does not push; the kill ring is integrated by T-TU-03-7.
    pub(super) fn delete_to_line_start(&mut self) {
        self.exit_history_browsing();
        self.last_action = None;

        let col = self.state.cursor_col;
        if col > 0 {
            // ── In-line: delete everything before the cursor ──
            self.push_undo_snapshot();
            let line = &mut self.state.lines[self.state.cursor_line];
            line.replace_range(..col, "");
            self.set_cursor_col(0);
        } else if self.state.cursor_line > 0 {
            // ── Start of line: merge into previous line (the deleted newline is recorded by the kill ring, T-TU-03-7) ──
            self.push_undo_snapshot();
            let current_line = self.state.lines.remove(self.state.cursor_line);
            self.state.cursor_line -= 1;
            let prev_len = self.state.lines[self.state.cursor_line].len();
            self.state.lines[self.state.cursor_line].push_str(&current_line);
            self.set_cursor_col(prev_len);
        }
        // First line start: no-op (pi still triggers onChange; keep consistent)

        self.emit_change();
    }

    /// Delete from cursor to line end; at line end, merge into the next line (aligns with pi deleteToEndOfLine). Triggers on_change.
    /// Undo: push a snapshot before an actual deletion (truncation or cross-line merge); no-op doesn't push.
    pub(super) fn delete_to_line_end(&mut self) {
        self.exit_history_browsing();
        self.last_action = None;

        let col = self.state.cursor_col;
        let line_len = self.state.lines[self.state.cursor_line].len();

        if col >= line_len {
            // Line end: merge with the next line (aligns with pi deleteToEndOfLine cross-line semantics)
            if self.state.cursor_line + 1 < self.state.lines.len() {
                self.push_undo_snapshot();
                let next_line = self.state.lines.remove(self.state.cursor_line + 1);
                self.state.lines[self.state.cursor_line].push_str(&next_line);
                self.emit_change();
            }
            return;
        }

        self.push_undo_snapshot();
        let line = &mut self.state.lines[self.state.cursor_line];
        line.truncate(col);

        self.emit_change();
    }

    /// Delete one word backward (before the cursor); at line start, merge into the previous line. Word boundary reuses `move_word_backward`
    /// (aligns with pi structure: deleteWordBackwards calls moveWordBackwards to get the boundary); deletion and movement
    /// Always uses the same word-boundary semantics. Triggers on_change. Undo: push snapshot before actual deletion; kill ring hooked up in T-TU-03-7.
    pub(super) fn delete_word_backward(&mut self) {
        self.exit_history_browsing();
        self.last_action = None;

        let col = self.state.cursor_col;

        if col == 0 {
            // Line start: merge into the previous line (aligns with pi deleteWordBackwards's col==0 branch)
            if self.state.cursor_line > 0 {
                self.push_undo_snapshot();
                let current_line = self.state.lines.remove(self.state.cursor_line);
                self.state.cursor_line -= 1;
                let prev_len = self.state.lines[self.state.cursor_line].len();
                self.state.lines[self.state.cursor_line].push_str(&current_line);
                self.set_cursor_col(prev_len);
                self.emit_change();
            }
            return;
        }

        // Use move_word_backward to find the word boundary (aligns with pi: moveWordBackwards gets deleteFrom);
        // if the word fills the entire line before the cursor (and it's not the first line), it wraps across lines — deletion semantics must not cross lines,
        // so restore the current line and set the boundary to line start (the whole prefix is the word; delete it all).
        self.push_undo_snapshot();
        let old_line = self.state.cursor_line;
        self.move_word_backward();
        let delete_from = if self.state.cursor_line == old_line {
            self.state.cursor_col
        } else {
            self.state.cursor_line = old_line;
            0
        };
        self.set_cursor_col(col);

        let line = &mut self.state.lines[self.state.cursor_line];
        line.replace_range(delete_from..col, "");
        self.set_cursor_col(delete_from);

        self.emit_change();
    }

    /// Delete one word forward (after the cursor); at line end, merge into the next line. Word boundary reuses `move_word_forward`
    /// (aligns with pi structure: deleteWordForward calls moveWordForwards for the boundary) delete and move
    /// Always uses the same word-boundary semantics. Triggers on_change. Undo: push snapshot before actual deletion; kill ring hooked up in T-TU-03-7.
    pub(super) fn delete_word_forward(&mut self) {
        self.exit_history_browsing();
        self.last_action = None;

        let col = self.state.cursor_col;
        let line_len = self.state.lines[self.state.cursor_line].len();

        if col >= line_len {
            // End of line: merge into next line (aligns with pi deleteWordForward's cross-line branch)
            if self.state.cursor_line + 1 < self.state.lines.len() {
                self.push_undo_snapshot();
                let next_line = self.state.lines.remove(self.state.cursor_line + 1);
                self.state.lines[self.state.cursor_line].push_str(&next_line);
                self.emit_change();
            }
            return;
        }

        // Use move_word_forward to find the word boundary (aligns with pi: moveWordForwards obtains deleteTo);
        // If the word/whitespace extends to end of line, it wraps across lines — delete semantics don't cross lines,
        // so restore the current line and set the boundary to end of line (delete the whole segment from cursor to end of line).
        self.push_undo_snapshot();
        let old_line = self.state.cursor_line;
        self.move_word_forward();
        let delete_to = if self.state.cursor_line == old_line && self.state.cursor_col > col {
            self.state.cursor_col
        } else {
            self.state.cursor_line = old_line;
            line_len
        };
        self.set_cursor_col(col);

        let line = &mut self.state.lines[self.state.cursor_line];
        line.replace_range(col..delete_to, "");
        // Cursor stays put (the following graphemes shift forward)

        self.emit_change();
    }
}

/// Normalize inserted text: \r\n and \r → \n, \t → 4 spaces (aligned with pi `normalizeText`).
/// Called by `insert_text` as the sole insertion entry point; callers need not normalize themselves;
/// placed in this module with the text-edit domain (Plan A: not merged into wrap.rs).
fn normalize_text(text: &str) -> String {
    text.replace("\r\n", "\n")
        .replace('\r', "\n")
        .replace('\t', "    ")
}

#[cfg(test)]
mod tests {
    use super::super::Editor;

    // ── insert_char ──

    #[test]
    fn test_insert_char_basic() {
        let mut e = Editor::new(0);
        e.insert_char('a');
        e.insert_char('b');
        assert_eq!(e.text(), "ab");
        assert_eq!(e.cursor(), (0, 2));
    }

    #[test]
    fn test_insert_char_mid_line() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("ac");
        e.set_cursor_col(1);
        e.insert_char('b');
        assert_eq!(e.text(), "abc");
        assert_eq!(e.cursor(), (0, 2));
    }

    #[test]
    fn test_insert_char_multibyte() {
        let mut e = Editor::new(0);
        e.insert_char('你');
        assert_eq!(e.text(), "你");
        assert_eq!(e.cursor(), (0, 3)); // 3 UTF-8 bytes
    }

    #[test]
    fn test_insert_char_exits_history_browsing() {
        let mut e = Editor::new(0);
        e.add_to_history("old");
        e.insert_text_at_cursor("draft");
        e.navigate_history(-1);
        assert_ne!(e.history_index, -1);
        e.insert_char('x');
        assert_eq!(e.history_index, -1);
    }

    // ── backspace ──

    #[test]
    fn test_backspace_basic() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("abc");
        e.backspace();
        assert_eq!(e.text(), "ab");
        assert_eq!(e.cursor(), (0, 2));
    }

    #[test]
    fn test_backspace_multibyte() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("a你b");
        e.set_cursor_col(4); // after the CJK char (bytes 1..4)
        e.backspace();
        assert_eq!(e.text(), "ab");
        assert_eq!(e.cursor(), (0, 1));
    }

    #[test]
    fn test_backspace_at_start_of_line_merges() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello\nworld");
        e.set_cursor_col(0);
        e.state.cursor_line = 1;
        e.backspace();
        assert_eq!(e.text(), "helloworld");
        assert_eq!(e.cursor(), (0, 5));
    }

    #[test]
    fn test_backspace_at_first_line_first_col_noop() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.set_cursor_col(0);
        e.backspace();
        assert_eq!(e.text(), "hello");
        assert_eq!(e.cursor(), (0, 0));
    }

    #[test]
    fn test_backspace_exits_history_browsing() {
        let mut e = Editor::new(0);
        e.add_to_history("old");
        e.insert_text_at_cursor("draft");
        e.navigate_history(-1);
        e.backspace();
        assert_eq!(e.history_index, -1);
    }

    // ── delete_forward ──

    #[test]
    fn test_delete_forward_basic() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("abc");
        e.set_cursor_col(1);
        e.delete_forward();
        assert_eq!(e.text(), "ac");
        assert_eq!(e.cursor(), (0, 1));
    }

    #[test]
    fn test_delete_forward_at_end_merges_lines() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello\nworld");
        e.set_cursor_col(5);
        e.state.cursor_line = 0;
        e.delete_forward();
        assert_eq!(e.text(), "helloworld");
        assert_eq!(e.cursor(), (0, 5));
    }

    #[test]
    fn test_delete_forward_at_end_of_last_line_noop() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        // cursor at end
        e.delete_forward();
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_delete_forward_multibyte() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("a你b");
        e.set_cursor_col(1); // before the CJK char
        e.delete_forward();
        assert_eq!(e.text(), "ab");
    }

    #[test]
    fn test_delete_forward_exits_history_browsing() {
        let mut e = Editor::new(0);
        e.add_to_history("old");
        e.insert_text_at_cursor("draft");
        e.navigate_history(-1);
        e.delete_forward();
        assert_eq!(e.history_index, -1);
    }

    // ── insert_text ──

    #[test]
    fn test_insert_text_single_line() {
        let mut e = Editor::new(0);
        e.insert_text("hello");
        assert_eq!(e.text(), "hello");
        assert_eq!(e.cursor(), (0, 5));
    }

    #[test]
    fn test_insert_text_multiline() {
        let mut e = Editor::new(0);
        e.insert_text("a\nb\nc");
        assert_eq!(e.text(), "a\nb\nc");
        assert_eq!(e.cursor(), (2, 1));
    }

    #[test]
    fn test_insert_text_mid_line_splits() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello world");
        e.set_cursor_col(6); // before "world"
        e.insert_text("beautiful ");
        assert_eq!(e.text(), "hello beautiful world");
    }

    #[test]
    fn test_insert_text_multiline_mid_line() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("abcdef");
        e.set_cursor_col(3); // after "abc"
        e.insert_text("\nXY");
        // "abc" + "\n" + "XY" + "def"
        assert_eq!(e.text(), "abc\nXYdef");
        assert_eq!(e.cursor(), (1, 2));
    }

    #[test]
    fn test_insert_text_normalizes_crlf() {
        let mut e = Editor::new(0);
        e.insert_text("a\r\nb");
        assert_eq!(e.text(), "a\nb");
    }

    #[test]
    fn test_insert_text_normalizes_tabs() {
        let mut e = Editor::new(0);
        e.insert_text("a\tb");
        assert_eq!(e.text(), "a    b");
    }

    #[test]
    fn test_insert_text_exits_history_browsing() {
        let mut e = Editor::new(0);
        e.add_to_history("old");
        e.insert_text_at_cursor("draft");
        e.navigate_history(-1);
        e.insert_text("x");
        assert_eq!(e.history_index, -1);
    }

    // ── newline ──

    #[test]
    fn test_newline_mid_line() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello world");
        e.set_cursor_col(5);
        e.newline();
        assert_eq!(e.text(), "hello\n world");
        assert_eq!(e.cursor(), (1, 0));
    }

    #[test]
    fn test_newline_at_end() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.newline();
        assert_eq!(e.text(), "hello\n");
        assert_eq!(e.cursor(), (1, 0));
    }

    #[test]
    fn test_newline_at_start() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.set_cursor_col(0);
        e.newline();
        assert_eq!(e.text(), "\nhello");
        assert_eq!(e.cursor(), (1, 0));
    }

    #[test]
    fn test_newline_exits_history_browsing() {
        let mut e = Editor::new(0);
        e.add_to_history("old");
        e.insert_text_at_cursor("draft");
        e.navigate_history(-1);
        e.newline();
        assert_eq!(e.history_index, -1);
    }

    // ── delete_to_line_start ──

    #[test]
    fn test_delete_to_line_start_mid_line() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello world");
        e.set_cursor_col(6);
        e.delete_to_line_start();
        assert_eq!(e.text(), "world");
        assert_eq!(e.cursor(), (0, 0));
    }

    #[test]
    fn test_delete_to_line_start_at_start_merges() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello\nworld");
        e.state.cursor_line = 1;
        e.set_cursor_col(0);
        e.delete_to_line_start();
        assert_eq!(e.text(), "helloworld");
        assert_eq!(e.cursor(), (0, 5));
    }

    #[test]
    fn test_delete_to_line_start_first_line_noop() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.set_cursor_col(0);
        e.delete_to_line_start();
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_delete_to_line_start_exits_history_browsing() {
        let mut e = Editor::new(0);
        e.add_to_history("old");
        e.insert_text_at_cursor("draft");
        e.navigate_history(-1);
        e.delete_to_line_start();
        assert_eq!(e.history_index, -1);
    }

    // ── delete_to_line_end ──

    #[test]
    fn test_delete_to_line_end_mid_line() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello world");
        e.set_cursor_col(5);
        e.delete_to_line_end();
        assert_eq!(e.text(), "hello");
        assert_eq!(e.cursor(), (0, 5));
    }

    #[test]
    fn test_delete_to_line_end_at_end_merges() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello\nworld");
        e.set_cursor_col(5);
        e.state.cursor_line = 0;
        e.delete_to_line_end();
        assert_eq!(e.text(), "helloworld");
    }

    #[test]
    fn test_delete_to_line_end_last_line_noop() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        // cursor at end
        e.delete_to_line_end();
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_delete_to_line_end_exits_history_browsing() {
        let mut e = Editor::new(0);
        e.add_to_history("old");
        e.insert_text_at_cursor("draft");
        e.navigate_history(-1);
        e.delete_to_line_end();
        assert_eq!(e.history_index, -1);
    }

    // ── delete_word_backward ──

    #[test]
    fn test_delete_word_backward_basic() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello world");
        e.delete_word_backward();
        assert_eq!(e.text(), "hello ");
        assert_eq!(e.cursor(), (0, 6));
    }

    #[test]
    fn test_delete_word_backward_at_start_merges() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello\nworld");
        e.state.cursor_line = 1;
        e.set_cursor_col(0);
        e.delete_word_backward();
        assert_eq!(e.text(), "helloworld");
        assert_eq!(e.cursor(), (0, 5));
    }

    #[test]
    fn test_delete_word_backward_first_line_noop() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        e.set_cursor_col(0);
        e.delete_word_backward();
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_delete_word_backward_exits_history_browsing() {
        let mut e = Editor::new(0);
        e.add_to_history("old");
        e.insert_text_at_cursor("draft");
        e.navigate_history(-1);
        e.delete_word_backward();
        assert_eq!(e.history_index, -1);
    }

    // ── delete_word_forward ──

    #[test]
    fn test_delete_word_forward_basic() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello world");
        e.set_cursor_col(0);
        e.delete_word_forward();
        assert_eq!(e.text(), "world");
    }

    #[test]
    fn test_delete_word_forward_at_end_merges() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello\nworld");
        e.set_cursor_col(5);
        e.state.cursor_line = 0;
        e.delete_word_forward();
        assert_eq!(e.text(), "helloworld");
    }

    #[test]
    fn test_delete_word_forward_last_line_noop() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        // cursor at end
        e.delete_word_forward();
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_delete_word_forward_exits_history_browsing() {
        let mut e = Editor::new(0);
        e.add_to_history("old");
        e.insert_text_at_cursor("draft");
        e.navigate_history(-1);
        e.delete_word_forward();
        assert_eq!(e.history_index, -1);
    }

    // ── normalize_text (private) ──

    #[test]
    fn test_normalize_text_crlf() {
        assert_eq!(super::normalize_text("a\r\nb"), "a\nb");
    }

    #[test]
    fn test_normalize_text_cr() {
        assert_eq!(super::normalize_text("a\rb"), "a\nb");
    }

    #[test]
    fn test_normalize_text_tab() {
        assert_eq!(super::normalize_text("a\tb"), "a    b");
    }

    #[test]
    fn test_normalize_text_combined() {
        assert_eq!(super::normalize_text("a\r\nb\tc"), "a\nb    c");
    }

    #[test]
    fn test_normalize_text_empty() {
        assert_eq!(super::normalize_text(""), "");
    }
}
