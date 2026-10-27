// REQ-TUI-01 (pi-413): Editor component — cursor movement (grapheme/word/line start & end/vertical movement across wrapped lines + visual line mapping and sticky column).
// Consumed by input.rs (arrow-key/word-movement actions, ↑↓ first/last line transitions) and text.rs (word deletion boundary), hence `pub(super)` on the related methods;
// cursor_visual_col / build_visual_line_map / find_current_visual_line are only used within this module and stay private.
// Undo (T-TU-03-5): Each movement function sets last_action = None (matching lastAction = null throughout pi),
// Make the next insertion after movement start a new undo unit; do not put the reset in set_cursor_col (it will be overwritten when reused by insert_char).
use super::word_navigation::{find_word_backward, find_word_forward};
use super::{Editor, VisualLine};
use crate::wrap::word_wrap_line;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

impl Editor {
    /// Move cursor left/right by grapheme (cross-line wrap: right at line end → next line start, left at line start → previous line end).
    /// At line end the coordinates also record preferred_visual_col (matching pi moveCursor's last-line case).
    /// Reset last_action: input after movement starts a new undo unit (aligned with pi moveCursor).
    pub(super) fn move_char(&mut self, delta_col: i32) {
        self.last_action = None;
        if delta_col == 0 {
            return;
        }

        let line = &self.state.lines[self.state.cursor_line];
        let graphemes: Vec<(usize, &str)> = line.grapheme_indices(true).collect();
        let g_count = graphemes.len();

        // current cursor index within the grapheme sequence (cursor_col is a byte offset and needs conversion)
        let current_g_idx = graphemes
            .iter()
            .position(|(byte_offset, _)| *byte_offset == self.state.cursor_col)
            .unwrap_or(g_count); // at line end (cursor_col == line.len()) → g_count

        let new_g_idx = current_g_idx as i32 + delta_col;

        if new_g_idx < 0 {
            // ── move left at line start → previous line end ──
            if self.state.cursor_line > 0 {
                self.state.cursor_line -= 1;
                let prev_line = &self.state.lines[self.state.cursor_line];
                let prev_len = prev_line.len();
                self.set_cursor_col(prev_len); // line end = byte length
            }
            // First line start: no move
        } else if new_g_idx as usize > g_count {
            // ── Line end move → next line start ──
            if self.state.cursor_line + 1 < self.state.lines.len() {
                self.state.cursor_line += 1;
                self.set_cursor_col(0);
            }
            // Last line end: no move
        } else {
            // ── Movement within the same line ──
            let target_g_idx = new_g_idx as usize;
            if target_g_idx == g_count {
                // Move to line end: record preferred_visual_col (aligned with pi moveCursor last-line case)
                self.state.cursor_col = line.len();
                self.preferred_visual_col = Some(self.cursor_visual_col());
                self.snapped_from_cursor_col = None;
            } else {
                self.set_cursor_col(graphemes[target_g_idx].0);
            }
        }
    }

    /// Returns the cursor's visual column within the current line (accumulating grapheme widths, used by preferred_visual_col).
    fn cursor_visual_col(&self) -> usize {
        let line = &self.state.lines[self.state.cursor_line];
        line.grapheme_indices(true)
            .take_while(|(byte_offset, _)| *byte_offset < self.state.cursor_col)
            .map(|(_, g)| UnicodeWidthStr::width(g))
            .sum()
    }

    /// Move cursor by word boundaries (forward/backward; line start/end crossing aligns with pi moveWordBackwards/Forwards).
    /// Word-boundary detection delegates to `word_navigation::find_word_*` (T-TU-03-8, pi-407 semantics:
    /// UAX#29 word segmentation + intra-word ASCII punctuation breakpoints).
    pub(super) fn move_word(&mut self, forward: bool) {
        if forward {
            self.move_word_forward();
        } else {
            self.move_word_backward();
        }
    }

    /// Move forward by word: lands at the current word end (before punctuation) or the next word end (pi `findWordForward` semantics,
    /// e.g. `hello world` from 0 moves to 5, not the next word start 6); when the cursor is at line end, cross to the next line start
    /// (matches pi `moveWordForwards` line-boundary check; do not continue finding words after crossing lines).
    /// Consumed by text::delete_word_forward to get the deletion boundary, hence `pub(super)`.
    /// Reset last_action (aligned with pi moveWordForwards).
    pub(super) fn move_word_forward(&mut self) {
        self.last_action = None;
        let line = &self.state.lines[self.state.cursor_line];
        let line_len = line.len();

        // Line end → cross to next line start (matches pi moveWordForwards line-boundary branch; no movement at last line end)
        if self.state.cursor_col >= line_len {
            if self.state.cursor_line + 1 < self.state.lines.len() {
                self.state.cursor_line += 1;
                self.set_cursor_col(0);
            }
            return;
        }

        let new_col = find_word_forward(line, self.state.cursor_col);
        self.set_cursor_col(new_col);
    }

    /// Move backward by word: lands at the previous word start or after intra-word punctuation (pi `findWordBackward` semantics,
    /// `foo.bar` from word end moves back to `bar` start, then to `.`); when the cursor is at line start, cross to the previous line end
    /// (matches pi `moveWordBackwards` line-boundary check; do not continue finding words after crossing lines).
    /// Consumed by text::delete_word_backward to get the deletion boundary, hence `pub(super)`.
    /// Reset last_action (aligned with pi moveWordBackwards).
    pub(super) fn move_word_backward(&mut self) {
        self.last_action = None;
        let line = &self.state.lines[self.state.cursor_line];

        // Line start → cross to previous line end (matches pi moveWordBackwards line-boundary branch; no movement at first line start)
        if self.state.cursor_col == 0 {
            if self.state.cursor_line > 0 {
                self.state.cursor_line -= 1;
                let prev_len = self.state.lines[self.state.cursor_line].len();
                self.set_cursor_col(prev_len);
            }
            return;
        }

        let new_col = find_word_backward(line, self.state.cursor_col);
        self.set_cursor_col(new_col);
    }

    /// Move cursor to logical line start/end (explicit set_cursor_col, resets sticky state).
    /// Reset last_action (aligned with pi moveToLineStart/moveToLineEnd).
    pub(super) fn move_to_line_start(&mut self) {
        self.last_action = None;
        self.set_cursor_col(0);
    }
    pub(super) fn move_to_line_end(&mut self) {
        self.last_action = None;
        let len = self.state.lines[self.state.cursor_line].len();
        self.set_cursor_col(len);
    }

    /// Move up/down by visual line (±1 visual line), crossing wrapped segments. Implements sticky column (simplified form of pi
    /// `computeVerticalMoveColumn` decision table):
    /// - Target too short (stops at line end) → lock preferred_visual_col, so the next move preserves the column;
    /// - Target long enough and lands back on preferred → clear preferred, re-derive from the new cursor column next time.
    ///   Do not snap to or skip atomic segments (zen has no paste-marker; wide text uses 2-column CJK, and the cursor always lands at the start of a grapheme cluster).
    ///   Reset last_action (aligned with the vertical branch of pi moveCursor/pageScroll).
    pub(super) fn move_vertical(&mut self, delta: i32) {
        self.last_action = None;
        if delta == 0 {
            return;
        }

        let width = self.last_width;
        if width == 0 {
            return;
        }

        let map = self.build_visual_line_map(width);
        if map.is_empty() {
            return;
        }

        // ── 1. Find current visual line ──
        let current_vis = self.find_current_visual_line(&map);

        // ── 2. Compute target visual line (clamped) ──
        let target_vis = (current_vis as i32 + delta).clamp(0, map.len() as i32 - 1) as usize;
        if target_vis == current_vis {
            return; // At top/bottom, no move
        }

        // ── 3. Determine target column (sticky column decision) ──
        let target_vl = &map[target_vis];
        let target_line = &self.state.lines[target_vl.logical_line];
        // Byte range of the target visual line (start_col and length in the same unit)
        let vl_start = target_vl.start_col;
        let vl_end = vl_start + target_vl.length;

        // Graphemes in the target visual line (byte offset, byte length, display width)
        let target_graphemes: Vec<(usize, usize, usize)> = target_line
            .grapheme_indices(true)
            .skip_while(|(bo, _)| *bo < vl_start)
            .take_while(|(bo, _)| *bo < vl_end)
            .map(|(bo, g)| (bo, g.len(), UnicodeWidthStr::width(g)))
            .collect();

        // sticky column: prefer the already-locked preferred, otherwise take the current visual column (aligned with pi column P)
        let preferred = self
            .preferred_visual_col
            .unwrap_or_else(|| self.cursor_visual_col());

        // ── 4. Find the position closest to preferred in the target line ──
        let mut visual_col = 0usize;
        let mut best_bo = vl_start;

        for &(bo, _len, g_width) in &target_graphemes {
            if visual_col >= preferred {
                break;
            }
            best_bo = bo;
            visual_col += g_width;
        }

        // preferred exceeds the target line: stop at line end
        let final_bo = if visual_col < preferred {
            vl_end
        } else {
            best_bo
        };

        // ── 5. Update state ──
        self.state.cursor_line = target_vl.logical_line;
        self.state.cursor_col = final_bo;

        // Sticky column lifecycle (aligned with pi decision table):
        // - Target was long enough, snapped back to preferred → fulfilled, clear (re-derive from new cursor column next time)
        // - Target too short (stopped at line end) → lock preferred, to keep the column on the next move
        if visual_col < preferred {
            self.preferred_visual_col = Some(preferred);
        } else {
            self.preferred_visual_col = None;
        }
        self.snapped_from_cursor_col = None;
    }

    /// Page scroll: move vertically by page height (max(5, 30% of viewport height)) (aligned with pi pageScroll);
    /// page height taken from `last_height` (viewport height from the most recent render).
    pub(super) fn page_scroll(&mut self, down: bool) {
        let viewport_height = self.last_height;
        let page_size = std::cmp::max(5, viewport_height * 30 / 100) as i32;
        let delta = if down { page_size } else { -page_size };
        self.move_vertical(delta);
    }

    /// Whether the cursor is on the first visual line (aligned with pi isOnFirstVisualLine), used by input::handle_action for the ↑ transition decision.
    pub(super) fn is_on_first_visual_line(&self) -> bool {
        let map = self.build_visual_line_map(self.last_width);
        self.find_current_visual_line(&map) == 0
    }

    /// Whether the cursor is on the last visual line (aligned with pi isOnLastVisualLine), used by input::handle_action for the ↓ transition decision.
    pub(super) fn is_on_last_visual_line(&self) -> bool {
        let map = self.build_visual_line_map(self.last_width);
        self.find_current_visual_line(&map) == map.len().saturating_sub(1)
    }

    /// Build logical-line → visual-line map (aligned with pi buildVisualLineMap), used by move_vertical/page_scroll.
    /// length is the byte-range length (end_index - start_index), same unit as start_col/cursor_col.
    fn build_visual_line_map(&self, width: usize) -> Vec<VisualLine> {
        let mut map = Vec::new();

        for (line_idx, line) in self.state.lines.iter().enumerate() {
            // Wrap width is provided by the caller (move_vertical passes last_width = layout_width); use it directly, no padding subtracted
            let chunks = word_wrap_line(line, width);

            for chunk in &chunks {
                map.push(VisualLine {
                    logical_line: line_idx,
                    start_col: chunk.start_index,
                    length: chunk.end_index - chunk.start_index,
                });
            }
        }

        map
    }

    /// Find the visual line index containing the cursor (aligned with pi findCurrentVisualLine).
    fn find_current_visual_line(&self, map: &[VisualLine]) -> usize {
        let cursor_line = self.state.cursor_line;
        let cursor_col = self.state.cursor_col;

        for (vis_idx, vl) in map.iter().enumerate() {
            if vl.logical_line != cursor_line {
                continue;
            }
            // Cursor falls within this visual line's [start_col, start_col + length] range
            // The end position (start_col + length) belongs to this line (cursor at line end)
            if cursor_col >= vl.start_col && cursor_col <= vl.start_col + vl.length {
                return vis_idx;
            }
        }

        // Fallback: return the last visual line of this logical line
        map.iter()
            .rposition(|vl| vl.logical_line == cursor_line)
            .unwrap_or(0)
    }

    /// Set cursor column and reset preferred/snapped state (mandatory for all horizontal moves/edits).
    /// Consumed by text.rs (insert/delete positioning) and the move_char/move_word series, hence pub(super).
    pub(super) fn set_cursor_col(&mut self, col: usize) {
        self.state.cursor_col = col;
        self.preferred_visual_col = None;
        self.snapped_from_cursor_col = None;
    }
}

#[cfg(test)]
mod tests {
    /// CJK sample kept intentionally: this test verifies wide-character / multibyte behavior.
    const CJK_SAMPLE: &str = "你好";
    const CJK_SAMPLE_2: &str = "a你b";

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
    use super::*;

    /// Construct an editor with the given line text and cursor (line, col).
    /// tests is a descendant module of editor, so it can access Editor's private fields directly (same privileges as the impl in cursor.rs).
    /// last_width/last_height are used by the visual line system of move_vertical/page_scroll.
    fn editor_with(
        lines: &[&str],
        cursor_line: usize,
        cursor_col: usize,
        last_width: usize,
        last_height: usize,
    ) -> Editor {
        let mut e = e(0);
        e.state.lines = lines.iter().map(|s| s.to_string()).collect();
        if e.state.lines.is_empty() {
            e.state.lines.push(String::new());
        }
        e.state.cursor_line = cursor_line;
        e.state.cursor_col = cursor_col;
        e.last_width = last_width;
        e.last_height = last_height;
        e
    }

    // ── move_char ──

    #[test]
    fn test_move_char_ascii_step() {
        let mut e = editor_with(&["abcd"], 0, 0, 0, 0);
        e.move_char(1);
        assert_eq!(e.cursor(), (0, 1));
        e.move_char(1);
        assert_eq!(e.cursor(), (0, 2));
    }

    #[test]
    fn test_move_char_wraps_down_to_next_line() {
        let mut e = editor_with(&["abc", "def"], 0, 3, 0, 0); // End of line 0
        e.move_char(1);
        assert_eq!(e.cursor(), (1, 0));
    }

    #[test]
    fn test_move_char_wraps_up_to_prev_line() {
        let mut e = editor_with(&["abc", "def"], 1, 0, 0, 0); // Start of line 1
        e.move_char(-1);
        assert_eq!(e.cursor(), (0, 3));
    }

    #[test]
    fn test_move_char_stays_at_bounds() {
        // Moving left from start of first line does nothing
        let mut e = editor_with(&["abc", "def"], 0, 0, 0, 0);
        e.move_char(-1);
        assert_eq!(e.cursor(), (0, 0));
        // Moving right from end of last line does nothing
        let mut e = editor_with(&["abc", "def"], 1, 3, 0, 0);
        e.move_char(1);
        assert_eq!(e.cursor(), (1, 3));
    }

    #[test]
    fn test_move_char_multibyte_grapheme_step() {
        // Two CJK chars = 2 three-byte graphemes; the cursor must step by grapheme, never split one in half
        let mut e = editor_with(&[CJK_SAMPLE], 0, 0, 0, 0);
        e.move_char(1);
        assert_eq!(e.cursor(), (0, 3));
        e.move_char(1); // At end of line: record preferred_visual_col = 4 (each grapheme is 2 columns, accumulated visual column)
        assert_eq!(e.cursor(), (0, 6));
        assert_eq!(e.preferred_visual_col, Some(4));
        assert_eq!(e.snapped_from_cursor_col, None);
    }

    #[test]
    fn test_move_char_zero_delta_noop() {
        let mut e = editor_with(&["abcd"], 0, 2, 0, 0);
        e.move_char(0);
        assert_eq!(e.cursor(), (0, 2));
    }

    #[test]
    fn test_move_char_to_line_end_records_preferred_col() {
        let mut e = editor_with(&["abcd"], 0, 0, 0, 0);
        e.move_char(4); // 4 graphemes, falls straight into the end-of-line branch
        assert_eq!(e.cursor(), (0, 4));
        assert_eq!(e.preferred_visual_col, Some(4));
        assert_eq!(e.snapped_from_cursor_col, None);
    }

    // ── cursor_visual_col ──

    #[test]
    fn test_cursor_visual_col_ascii() {
        let mut e = editor_with(&["abcd"], 0, 2, 0, 0);
        assert_eq!(e.cursor_visual_col(), 2);
    }

    #[test]
    fn test_cursor_visual_col_cjk() {
        // "a" + one CJK char + "b": the CJK char occupies 2 columns; with the cursor after it, the visual column is 1+2=3
        let mut e = editor_with(&[CJK_SAMPLE_2], 0, 4, 0, 0);
        assert_eq!(e.cursor_visual_col(), 3);
    }

    // ── move_word / move_word_forward / move_word_backward ──

    #[test]
    fn test_move_word_forward_lands_at_word_end() {
        let mut e = editor_with(&["hello world foo"], 0, 0, 0, 0);
        e.move_word(true);
        assert_eq!(e.cursor(), (0, 5)); // "hello" word end (before punctuation/whitespace, pi landing semantics)
        e.move_word(true);
        assert_eq!(e.cursor(), (0, 11)); // "world" word end (skip whitespace)
    }

    #[test]
    fn test_move_word_forward_with_leading_space_before_word() {
        let mut e = editor_with(&["  ab cd"], 0, 2, 0, 0); // Cursor at the start of the word "ab"
        e.move_word(true);
        assert_eq!(e.cursor(), (0, 4)); // "ab" word end
    }

    #[test]
    fn test_move_word_forward_at_doc_end_stays() {
        let mut e = editor_with(&["hello"], 0, 0, 0, 0);
        e.move_word(true);
        assert_eq!(e.cursor(), (0, 5)); // Last line end (for a single-word line, landing at word end is line end)
        e.move_word(true);
        assert_eq!(e.cursor(), (0, 5)); // Already at the end, no movement
    }

    #[test]
    fn test_move_word_forward_wraps_to_next_line() {
        let mut e = editor_with(&["hello", "world"], 0, 0, 0, 0);
        e.move_word(true);
        assert_eq!(e.cursor(), (0, 5)); // On the first line, landing at word end is line end (in-line navigation, no line crossing)
        e.move_word(true);
        assert_eq!(e.cursor(), (1, 0)); // Line end → cross to next line start (matches pi moveWordForwards)
    }

    #[test]
    fn test_move_word_forward_stops_at_in_word_punctuation() {
        // pi-407 intra-word ASCII punctuation breakpoints: foo.bar splits into foo / . / bar
        let mut e = editor_with(&["foo.bar"], 0, 0, 0, 0);
        e.move_word(true);
        assert_eq!(e.cursor(), (0, 3)); // "foo" word end (before '.')
        e.move_word(true);
        assert_eq!(e.cursor(), (0, 4)); // '.' segment
        e.move_word(true);
        assert_eq!(e.cursor(), (0, 7)); // "bar" word end
    }

    #[test]
    fn test_move_word_backward_to_word_start() {
        let mut e = editor_with(&["hello world"], 0, 11, 0, 0); // End of line
        e.move_word(false);
        assert_eq!(e.cursor(), (0, 6)); // start of "world"
    }

    #[test]
    fn test_move_word_backward_skips_trailing_space() {
        let mut e = editor_with(&["ab  "], 0, 4, 0, 0); // End of line (after the two spaces)
        e.move_word(false);
        assert_eq!(e.cursor(), (0, 0)); // Skip past the whitespace back to the start of the word
    }

    #[test]
    fn test_move_word_backward_wraps_to_prev_line() {
        let mut e = editor_with(&["hello", "world"], 1, 0, 0, 0);
        e.move_word(false);
        assert_eq!(e.cursor(), (0, 5)); // Line start → cross to previous line end (matches pi moveWordBackwards)
    }

    #[test]
    fn test_move_word_backward_stops_at_in_word_punctuation() {
        // pi-407 intra-word punctuation breakpoints: from "bar" word end move back to "bar" start, then '.', then "foo" start
        let mut e = editor_with(&["foo.bar"], 0, 7, 0, 0);
        e.move_word(false);
        assert_eq!(e.cursor(), (0, 4)); // "bar" word start
        e.move_word(false);
        assert_eq!(e.cursor(), (0, 3)); // '.' segment
        e.move_word(false);
        assert_eq!(e.cursor(), (0, 0)); // Start of the word "foo"
    }

    // ── move_to_line_start / move_to_line_end ──

    #[test]
    fn test_move_to_line_start_resets_sticky() {
        let mut e = editor_with(&["abcd"], 0, 2, 0, 0);
        e.preferred_visual_col = Some(5);
        e.snapped_from_cursor_col = Some(1);
        e.move_to_line_start();
        assert_eq!(e.cursor(), (0, 0));
        assert_eq!(e.preferred_visual_col, None);
        assert_eq!(e.snapped_from_cursor_col, None);
    }

    #[test]
    fn test_move_to_line_end() {
        let mut e = editor_with(&["abcd"], 0, 1, 0, 0);
        e.move_to_line_end();
        assert_eq!(e.cursor(), (0, 4));
    }

    // ── move_vertical (sticky column / visual lines) ──

    #[test]
    fn test_move_vertical_simple_line_step() {
        let mut e = editor_with(&["ab", "cd"], 0, 0, 10, 0);
        e.move_vertical(1);
        assert_eq!(e.cursor(), (1, 0));
        assert_eq!(e.preferred_visual_col, None); // Target line is long enough, landing back on it, clear the sticky column
    }

    #[test]
    fn test_move_vertical_sticky_column_keeps_preferred() {
        // Target row is too short → stop at line end and lock preferred_visual_col
        let mut e = editor_with(&["abcd", "xy"], 0, 4, 10, 0);
        e.move_vertical(1);
        assert_eq!(e.cursor(), (1, 2));
        assert_eq!(e.preferred_visual_col, Some(4));

        // Move up again: resolve with locked col 4; col 4 unreachable in "abcd" → fall back to col 3 and clear
        e.move_vertical(-1);
        assert_eq!(e.cursor(), (0, 3));
        assert_eq!(e.preferred_visual_col, None);
    }

    #[test]
    fn test_move_vertical_cjk_wide_column() {
        // Cursor after the CJK char (visual col 2); next row "ab" max 2 cols → stop at 'b' (col 1)
        let mut e = editor_with(&[CJK_SAMPLE, "ab"], 0, 3, 10, 0);
        e.move_vertical(1);
        assert_eq!(e.cursor(), (1, 1));
    }

    #[test]
    fn test_move_vertical_within_wrapped_visual_lines() {
        // "abc def" width 5 → visual rows [0,4) "abc " and [4,7) "def"
        let mut e = editor_with(&["abc def"], 0, 0, 5, 0);
        e.move_vertical(1); // Move down to the second wrap segment of the same logical line
        assert_eq!(e.cursor(), (0, 4));

        let mut e = editor_with(&["abc def"], 0, 6, 5, 0);
        e.move_vertical(-1); // Move back up to end of first segment; target too short → lock preferred = 6
        assert_eq!(e.cursor(), (0, 4));
        assert_eq!(e.preferred_visual_col, Some(6));
    }

    #[test]
    fn test_move_vertical_clamped_at_bottom_stays() {
        let mut e = editor_with(&["ab", "cd"], 1, 0, 10, 0);
        e.move_vertical(1); // Already on the last visual row
        assert_eq!(e.cursor(), (1, 0));
    }

    #[test]
    fn test_move_vertical_delta_zero_noop() {
        let mut e = editor_with(&["ab", "cd"], 0, 1, 10, 0);
        e.move_vertical(0);
        assert_eq!(e.cursor(), (0, 1));
    }

    // ── page_scroll ──

    #[test]
    fn test_page_scroll_down_and_up() {
        // Height 30 → page height = max(5, 30*30/100) = 9
        let lines: Vec<&str> = (0..12).map(|_| "ab").collect();
        let mut e = editor_with(&lines, 0, 0, 10, 30);
        e.page_scroll(true);
        assert_eq!(e.cursor(), (9, 0));
        e.page_scroll(false);
        assert_eq!(e.cursor(), (0, 0));
    }

    #[test]
    fn test_page_scroll_small_viewport_page_min_5() {
        // Height 4 → page height = max(5, 1) = 5
        let lines: Vec<&str> = (0..7).map(|_| "ab").collect();
        let mut e = editor_with(&lines, 0, 0, 10, 4);
        e.page_scroll(true);
        assert_eq!(e.cursor(), (5, 0));
    }

    // ── is_on_first_visual_line / is_on_last_visual_line ──

    #[test]
    fn test_visual_line_bounds_single_line() {
        let mut e = editor_with(&["ab"], 0, 0, 10, 0);
        assert!(e.is_on_first_visual_line());
        assert!(e.is_on_last_visual_line());
    }

    #[test]
    fn test_visual_line_bounds_wrapped_multi_line() {
        // "abc def" width 5 wraps into two segments: cursor at 0 → first; at 6 → last
        let mut e = editor_with(&["abc def"], 0, 0, 5, 0);
        assert!(e.is_on_first_visual_line());
        assert!(!e.is_on_last_visual_line());

        let mut e = editor_with(&["abc def"], 0, 6, 5, 0);
        assert!(!e.is_on_first_visual_line());
        assert!(e.is_on_last_visual_line());

        // Multiple logical lines: first line → first; last line → last
        let mut e = editor_with(&["ab", "cd"], 1, 0, 10, 0);
        assert!(!e.is_on_first_visual_line());
        assert!(e.is_on_last_visual_line());
    }

    // ── build_visual_line_map ──

    #[test]
    fn test_build_visual_line_map_simple() {
        let e = editor_with(&["ab", "cdefg"], 0, 0, 3, 0);
        let map = e.build_visual_line_map(3);
        assert_eq!(map.len(), 3);
        assert_eq!(
            (map[0].logical_line, map[0].start_col, map[0].length),
            (0, 0, 2)
        );
        assert_eq!(
            (map[1].logical_line, map[1].start_col, map[1].length),
            (1, 0, 3)
        );
        assert_eq!(
            (map[2].logical_line, map[2].start_col, map[2].length),
            (1, 3, 2)
        );
    }

    #[test]
    fn test_build_visual_line_map_empty_line() {
        let e = editor_with(&["", "ab"], 0, 0, 10, 0);
        let map = e.build_visual_line_map(10);
        assert_eq!(map.len(), 2);
        assert_eq!((map[0].logical_line, map[0].length), (0, 0)); // Empty line produces an empty visual row
        assert_eq!((map[1].logical_line, map[1].length), (1, 2));
    }

    // ── find_current_visual_line ──

    #[test]
    fn test_find_current_visual_line_middle_and_end() {
        let e = editor_with(&["cdefg"], 0, 0, 3, 0);
        let map = e.build_visual_line_map(3); // [(0,0,3), (0,3,2)]

        let mut e = editor_with(&["cdefg"], 0, 0, 3, 0);
        assert_eq!(e.find_current_visual_line(&map), 0);
        let mut e = editor_with(&["cdefg"], 0, 3, 3, 0); // Wrap point boundary belongs to the first segment
        assert_eq!(e.find_current_visual_line(&map), 0);
        let mut e = editor_with(&["cdefg"], 0, 4, 3, 0);
        assert_eq!(e.find_current_visual_line(&map), 1);
        let mut e = editor_with(&["cdefg"], 0, 5, 3, 0); // Line end belongs to the last segment
        assert_eq!(e.find_current_visual_line(&map), 1);
    }

    // ── set_cursor_col ──

    #[test]
    fn test_set_cursor_col_clears_sticky() {
        let mut e = editor_with(&["abcd"], 0, 1, 0, 0);
        e.preferred_visual_col = Some(3);
        e.snapped_from_cursor_col = Some(1);
        e.set_cursor_col(2);
        assert_eq!(e.cursor(), (0, 2));
        assert_eq!(e.preferred_visual_col, None);
        assert_eq!(e.snapped_from_cursor_col, None);
    }
}
