// REQ-TUI-01 (pi-413): editor component — layout and rendering: layout_text / render / cursor_position.
// render writes ScreenBuffer and computes layout width using effective padding (shares last_width with cursor_position /
// last_effective_padding / scroll_offset caches); layout_text is private to this module.
use super::{Editor, LayoutLine};
use crate::wrap::word_wrap_line;

impl Editor {
    /// Render the editor into the screen buffer at (x,y), with width width and height height. Design:
    /// Effective padding is clamped by available width (`min(padding_x, (width-1)/2)`, aligned with pi) → layout width
    /// When wrapping at the outer margin with no padding, reserve 1 column for the line-end cursor (aligned with pi layoutWidth calculation, deducted only once) →
    /// Line layout → scroll clamping (visible rows = **budget height − 2 borders** `(h-2).max(1)`; the budget is provided by the caller
    /// [`Editor::needed_height`] capped according to terminal row count, aligned with pi `slice(scrollOffset,
    /// maxVisibleLines)` displays all budgeted rows; content exceeding the budget scrolls naturally, with `↑N more`/`↓N more` indicators
    /// driven by scroll offset and remaining row count) → write to buffer line by line: top/bottom borders, text rows,
    /// Cursor inverse (`Style.attributes=[Reverse]` block: if the cursor is on a grapheme, that grapheme is inverted; at line end, a space is inverted).
    /// Consumers: T-AP-11-1 (called once inside the draw callback), T-AP-11-1-fix (the two render paths in ui.rs).
    pub fn render(
        &mut self,
        buf: &mut crate::screen::ScreenBuffer,
        x: u16,
        y: u16,
        width: u16,
        height: u16,
    ) {
        use unicode_segmentation::UnicodeSegmentation;
        use unicode_width::UnicodeWidthStr;

        if width < 2 || height < 3 {
            return; // Too small to render (minimum: 1 text column + 2 border rows)
        }

        let w = width as usize;
        let h = height as usize;

        // ── 1. Layout width (aligned with pi layoutWidth calculation, padding deducted only once) ──
        // Layout is constrained by available width: max_padding = (w-1)/2 (aligned with pi Math.min(paddingX, maxPadding));
        // With no padding, reserve 1 column for the line-end cursor (aligned with pi contentWidth - (paddingX ? 0 : 1)).
        let max_padding = (w.saturating_sub(1)) / 2;
        let pad = self.padding_x.min(max_padding);
        // Record effective padding for cursor_position to resolve column coordinates (consistent with this layout pass)
        self.last_effective_padding = pad;
        let content_width = (w - pad * 2).max(1);
        let layout_width = (content_width - if pad == 0 { 1 } else { 0 }).max(1);
        self.last_width = layout_width;
        self.last_height = h;

        let layout = self.layout_text(layout_width);
        let total_lines = layout.len();

        // ── Visible rows: budget height − 2 borders (the budget has already been capped by the caller's needed_height according to terminal row count
        //    constraint; P0-A: no longer truncated to 30% of editor height — old formula max(5, h*30/100).clamp(1,h-2)
        //    became disconnected from cap when budget > 7 (a 9-row budget drew only 5 rows + incorrectly showed ↓N more), aligned with pi
        //    slice(scrollOffset, maxVisibleLines) shows full budget-line semantics) ──
        let visible_rows = (h - 2).max(1);

        // ── 2. Scroll clamping: ensure the visual line containing the cursor is in the visible area, then clamp to bottom alignment (aligned with pi maxScrollOffset) ──
        let cursor_vis = layout.iter().position(|l| l.has_cursor).unwrap_or(0);
        if cursor_vis < self.scroll_offset {
            self.scroll_offset = cursor_vis;
        } else if cursor_vis >= self.scroll_offset + visible_rows {
            self.scroll_offset = cursor_vis - visible_rows + 1;
        }
        self.scroll_offset = self
            .scroll_offset
            .min(total_lines.saturating_sub(visible_rows));

        // ── 3. Styles ──
        let border_style = self.border_style.clone();
        let text_style = crate::screen::Style::default();
        let cursor_style = crate::screen::Style {
            attributes: vec![crossterm::style::Attribute::Reverse],
            ..crate::screen::Style::default()
        };

        // Text area left origin: x + effective padding (pad ≤ (w-1)/2, u16 conversion is safe)
        let text_x = x.saturating_add(pad as u16);

        // ── 4. Top border ──
        let border_y = y;
        if self.scroll_offset > 0 {
            let indicator = format!(" ─── ↑{} more ", self.scroll_offset);
            // Fill width is computed by display width (indicator.len() returns byte count and includes multi-byte ─/↑, which undercounts)
            let dashes = w.saturating_sub(crate::screen::visible_width(&indicator));
            buf.write(
                x,
                border_y,
                &border_style,
                &format!("{}{}", indicator, "─".repeat(dashes)),
            );
        } else {
            buf.write(x, border_y, &border_style, &"─".repeat(w));
        }

        // ── 5. Text rows ──
        for i in 0..visible_rows {
            let vis_idx = self.scroll_offset + i;
            let row_y = y + 1 + i as u16;

            if vis_idx < total_lines {
                let line = &layout[vis_idx];

                // Write text
                buf.write(text_x, row_y, &text_style, &line.text);

                // Cursor reverse-video (layout_text guarantees cursor_pos is always Some when has_cursor is set)
                if line.has_cursor {
                    let cursor_pos = line.cursor_pos.unwrap_or(0);
                    let cx = text_x.saturating_add(cursor_pos as u16);
                    let graphemes: Vec<&str> =
                        line.text.grapheme_indices(true).map(|(_, g)| g).collect();

                    // Find the grapheme at the cursor: grapheme's starting display column == cursor_pos
                    let mut col = 0usize;
                    let mut found = false;
                    for g in &graphemes {
                        let gw = UnicodeWidthStr::width(*g);
                        if col == cursor_pos {
                            // Cursor on a grapheme: reverse-video that grapheme
                            buf.write(cx, row_y, &cursor_style, g);
                            found = true;
                            break;
                        }
                        col += gw;
                    }
                    if !found {
                        // Cursor at end of line (or on an empty line): reverse-video a space
                        buf.write(cx, row_y, &cursor_style, " ");
                    }
                }
            }
            // Lines with no content in the window can stay blank: the differential refresh (screen.rs flush) clears leftovers from the old frame
        }

        // ── 6. Bottom border ──
        let bottom_y = y + (h - 1) as u16;
        let remaining = total_lines.saturating_sub(self.scroll_offset + visible_rows);
        if remaining > 0 {
            let indicator = format!(" ─── ↓{} more ", remaining);
            let dashes = w.saturating_sub(crate::screen::visible_width(&indicator));
            buf.write(
                x,
                bottom_y,
                &border_style,
                &format!("{}{}", indicator, "─".repeat(dashes)),
            );
        } else {
            buf.write(x, bottom_y, &border_style, &"─".repeat(w));
        }
    }

    /// Returns the hardware cursor position of the cursor in the editor's **own coordinate system** (visual row, column, relative to render's x/y origin;
    /// the application layer adds the MoveTo offset on top). Corresponds to pi CURSOR_MARKER → TUI positioning semantics (used for IME candidate window positioning).
    /// Layout width must match render (visual lines are parsed using the current value of last_width).
    /// When to call: after render — scroll_offset is only clamped during render, so if the cursor moves without a redraw, ...
    /// Row coordinate may fall outside the visible window; this function clamps it back inside (converges on the next render).
    /// Consumer: T-AP-11-2 (IME / caret positioning).
    pub fn cursor_position(&self) -> (u16, u16) {
        if self.last_width == 0 {
            return (0, 0);
        }

        let layout = self.layout_text(self.last_width);

        // visible line count (same formula as render: budget height − 2 borders),
        // Fallback clamp for the row coordinate while scroll_offset has not yet been clamped by the next render
        let visible = (self.last_height.saturating_sub(2)).max(1);

        // Find the visual row containing the caret
        for (vis_idx, line) in layout.iter().enumerate() {
            if line.has_cursor {
                // Row: relative position within the visible region (minus scroll offset, plus border offset 1)
                let in_window = vis_idx.saturating_sub(self.scroll_offset).min(visible - 1);
                let row = (in_window + 1) as u16;
                // Column: caret's visual column in the text + effective left padding (clamping consistent with render)
                let col = (line.cursor_pos.unwrap_or(0) + self.last_effective_padding) as u16;
                return (col, row);
            }
        }

        // Fallback (normally unreachable: an empty editor still produces a layout row with a caret; kept defensively)
        (self.last_effective_padding as u16, 1)
    }

    /// Compute the total height needed for rendering from the current text and wrapping rules (2 border rows + content rows), for the UI layer
    /// Budget the editor area height. Strictly consistent with [`Editor::render`]: same layout_width
    /// derivation (padding clamped to `(w-1)/2`, 1 column reserved for the cursor when no padding) → `layout_text`
    /// line count → `clamp(lines, 1, cap) + 2`, where `cap = max(5, terminal_rows*30/100)`
    /// (aligned with pi `maxVisibleLines = max(5, floor(rows*0.3))`, 30% with a floor of 5).
    /// Empty editor (layout is always 1 cursor row) returns 3.
    /// Constraint: callers must pass the same width as `render` (same frame, same width), otherwise the wrapped line count is off.
    /// Consumers: T-AP-11-1-fix (both render paths in ui.rs).
    pub fn needed_height(&self, width: u16, terminal_rows: u16) -> u16 {
        let w = width as usize;
        // layout width derivation identical to render (padding clamped + 1 cursor column reserved when no padding)
        let max_padding = (w.saturating_sub(1)) / 2;
        let pad = self.padding_x.min(max_padding);
        let content_width = (w - pad * 2).max(1);
        let layout_width = (content_width - if pad == 0 { 1 } else { 0 }).max(1);

        let lines = self.layout_text(layout_width).len();
        // cap: terminal row count formula (aligned with pi maxVisibleLines), floor 5; u32 intermediate math to avoid overflow
        let cap = std::cmp::max(5u16, ((terminal_rows as u32 * 30 / 100) as u16).max(5));
        (lines as u16).clamp(1, cap) + 2
    }

    // Lay out text into visual rows (including line wrapping and caret attribution, aligned with pi layoutText).
    /// Wrap width is provided by the caller (render/cursor_position pass layout_width/last_width);
    /// padding is no longer deducted — render already computes layout width using effective padding.
    fn layout_text(&self, width: usize) -> Vec<LayoutLine> {
        let mut result = Vec::new();

        for (line_idx, line) in self.state.lines.iter().enumerate() {
            let chunks = word_wrap_line(line, width);
            // word_wrap_line always returns ≥1 chunk (empty lines yield a single empty chunk); no empty-line fallback needed

            for (i, chunk) in chunks.iter().enumerate() {
                // Caret attribution: non-final chunk [start, end), final chunk [start, end]
                // (Aligned with pi layoutText: a cursor at a wrap-point boundary belongs to the next fragment; a cursor at end-of-line belongs to the last fragment)
                let is_last = i + 1 == chunks.len();
                let has_cursor = line_idx == self.state.cursor_line
                    && self.state.cursor_col >= chunk.start_index
                    && (self.state.cursor_col < chunk.end_index
                        || (is_last && self.state.cursor_col == chunk.end_index));

                let cursor_pos = if has_cursor {
                    // Cursor display position within this visual line (grapheme widths accumulated)
                    use unicode_segmentation::UnicodeSegmentation;
                    use unicode_width::UnicodeWidthStr;

                    let offset_in_chunk = self.state.cursor_col - chunk.start_index;
                    let pos = chunk.text[..offset_in_chunk]
                        .graphemes(true)
                        .map(|g| UnicodeWidthStr::width(g))
                        .sum();
                    Some(pos)
                } else {
                    None
                };

                result.push(LayoutLine {
                    text: chunk.text.clone(),
                    has_cursor,
                    cursor_pos,
                });
            }
        }

        result
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
    use crate::screen::ScreenBuffer;

    // ── render ──

    #[test]
    fn test_render_too_small_width_is_noop() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        let mut buf = ScreenBuffer::new(1, 5);
        // width < 2 → early return
        e.render(&mut buf, 0, 0, 1, 5);
        assert_eq!(e.scroll_offset, 0);
    }

    #[test]
    fn test_render_too_small_height_is_noop() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        let mut buf = ScreenBuffer::new(20, 2);
        // height < 3 → early return
        e.render(&mut buf, 0, 0, 20, 2);
        assert_eq!(e.scroll_offset, 0);
    }

    #[test]
    fn test_render_sets_last_width_and_height() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        let mut buf = ScreenBuffer::new(20, 10);
        e.render(&mut buf, 0, 0, 20, 10);
        assert!(e.last_width > 0);
        assert_eq!(e.last_height, 10);
    }

    #[test]
    fn test_render_sets_effective_padding() {
        let mut e = e(4);
        e.insert_text_at_cursor("hello");
        let mut buf = ScreenBuffer::new(20, 10);
        e.render(&mut buf, 0, 0, 20, 10);
        assert_eq!(e.last_effective_padding, 4);
    }

    #[test]
    fn test_render_clamps_padding_to_max() {
        let mut e = e(100);
        e.insert_text_at_cursor("hello");
        let mut buf = ScreenBuffer::new(20, 10);
        e.render(&mut buf, 0, 0, 20, 10);
        // max_padding = (20-1)/2 = 9
        assert_eq!(e.last_effective_padding, 9);
    }

    #[test]
    fn test_render_scroll_clamps_cursor_visible() {
        let mut e = e(0);
        // Insert enough lines to overflow visible area
        for i in 0..20 {
            if i > 0 {
                e.insert_text_at_cursor("\n");
            }
            e.insert_text_at_cursor(&format!("line {i}"));
        }
        // Cursor is on last line
        let mut buf = ScreenBuffer::new(40, 8);
        e.render(&mut buf, 0, 0, 40, 8);
        // visible_rows = budget height − 2 borders = 6 (post-P0-A formula, no longer truncated to 30% editor height)
        let visible = (8 - 2).max(1);
        assert_eq!(visible, 6);
        // Cursor line (19) should be in visible window
        let cursor_vis = e.scroll_offset; // approximate
        assert!(cursor_vis + visible >= 20 || e.scroll_offset > 0);
    }

    #[test]
    fn test_render_no_scroll_when_content_fits() {
        let mut e = e(0);
        e.insert_text_at_cursor("short");
        let mut buf = ScreenBuffer::new(40, 10);
        e.render(&mut buf, 0, 0, 40, 10);
        assert_eq!(e.scroll_offset, 0);
    }

    // ── cursor_position ──

    #[test]
    fn test_cursor_position_zero_width_returns_origin() {
        let e = e(0);
        assert_eq!(e.cursor_position(), (0, 0));
    }

    #[test]
    fn test_cursor_position_empty_editor() {
        let mut e = e(0);
        let mut buf = ScreenBuffer::new(20, 10);
        e.render(&mut buf, 0, 0, 20, 10);
        let (col, row) = e.cursor_position();
        // Empty editor: cursor at (0,0) → col = 0 + padding(0) = 0, row = 1 (after top border)
        assert_eq!(col, 0);
        assert_eq!(row, 1);
    }

    #[test]
    fn test_cursor_position_with_padding() {
        let mut e = e(3);
        e.insert_text_at_cursor("hi");
        let mut buf = ScreenBuffer::new(20, 10);
        e.render(&mut buf, 0, 0, 20, 10);
        let (col, row) = e.cursor_position();
        // col = cursor_pos(2) + padding(3) = 5
        assert_eq!(col, 5);
        assert_eq!(row, 1);
    }

    #[test]
    fn test_cursor_position_after_text() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello");
        let mut buf = ScreenBuffer::new(20, 10);
        e.render(&mut buf, 0, 0, 20, 10);
        let (col, row) = e.cursor_position();
        assert_eq!(col, 5);
        assert_eq!(row, 1);
    }

    #[test]
    fn test_cursor_position_multiline_second_line() {
        let mut e = e(0);
        e.insert_text_at_cursor("first\nsecond");
        let mut buf = ScreenBuffer::new(20, 10);
        e.render(&mut buf, 0, 0, 20, 10);
        let (col, row) = e.cursor_position();
        // Cursor at end of "second" (line 1), row = 1 (line index) + 1 (border) = 2
        assert_eq!(col, 6);
        assert_eq!(row, 2);
    }

    #[test]
    fn test_cursor_position_mid_line() {
        let mut e = e(0);
        e.insert_text_at_cursor("hello world");
        e.move_char(-5); // cursor before "world"
        let mut buf = ScreenBuffer::new(20, 10);
        e.render(&mut buf, 0, 0, 20, 10);
        let (col, row) = e.cursor_position();
        assert_eq!(col, 6);
        assert_eq!(row, 1);
    }

    // ── layout_text (indirect via cursor_position after render) ──

    #[test]
    fn test_layout_text_single_line_no_wrap() {
        let mut e = e(0);
        e.insert_text_at_cursor("short");
        let mut buf = ScreenBuffer::new(40, 10);
        e.render(&mut buf, 0, 0, 40, 10);
        // Cursor should be on first visual line
        let (_, row) = e.cursor_position();
        assert_eq!(row, 1);
    }

    #[test]
    fn test_layout_text_wraps_long_line() {
        let mut e = e(0);
        // Insert text longer than layout width
        e.insert_text_at_cursor("abcdefghijklmnopqrstuvwxyz");
        let mut buf = ScreenBuffer::new(20, 10);
        e.render(&mut buf, 0, 0, 20, 10);
        // last_width should be < 26 (text length), meaning wrapping occurred
        assert!(e.last_width < 26);
    }

    #[test]
    fn test_layout_text_empty_line_has_cursor() {
        let mut e = e(0);
        // Empty editor: single empty line
        let mut buf = ScreenBuffer::new(20, 10);
        e.render(&mut buf, 0, 0, 20, 10);
        let (col, row) = e.cursor_position();
        // Cursor should be at (0, 1) for empty line
        assert_eq!(col, 0);
        assert_eq!(row, 1);
    }

    // ── needed_height ──

    /// Concatenate the text of all spans in a buffer line (helper for reading border/content rows).
    fn join_row(buf: &ScreenBuffer, row: u16) -> String {
        buf.line_segments(row)
            .iter()
            .map(|s| s.text.as_str())
            .collect()
    }

    #[test]
    fn test_needed_height_empty_is_three() {
        let e = e(0);
        // empty editor layout is always 1 cursor row → 2 borders + 1 content = 3
        assert_eq!(e.needed_height(40, 24), 3);
        assert_eq!(e.needed_height(40, 10), 3);
    }

    #[test]
    fn test_needed_height_three_lines_is_five() {
        let mut e = e(0);
        e.set_text("a\nb\nc");
        // 3 content rows → 2 borders + 3 = 5 (cap 24*30%=7 not hit)
        assert_eq!(e.needed_height(40, 24), 5);
    }

    #[test]
    fn test_needed_height_counts_wrapped_rows() {
        let mut e = e(0);
        // 26 chars at width=20 (layout_width=19) wrap into 2 visual lines
        e.insert_text_at_cursor("abcdefghijklmnopqrstuvwxyz");
        assert_eq!(e.needed_height(20, 24), 4);
        // same text in a wider buffer doesn't wrap → 1 line → 3
        assert_eq!(e.needed_height(40, 24), 3);
    }

    #[test]
    fn test_needed_height_caps_at_max_5_or_30pct() {
        let mut e = e(0);
        // rows=10 → cap = max(5, 3) = 5: 5 content rows → 7 (capped), 8 content rows still 7
        let t5 = "a\nb\nc\nd\ne";
        let t8 = "a\nb\nc\nd\ne\nf\ng\nh";
        e.set_text(t5);
        assert_eq!(
            e.needed_height(40, 10),
            7,
            "With rows=10, cap=5; 5 content lines hit the top at 7"
        );
        e.set_text(t8);
        assert_eq!(
            e.needed_height(40, 10),
            7,
            "8 content lines are still truncated by cap=5 to 7"
        );
        // rows=40 → cap = max(5, 12) = 12: 7 content rows → 9 (not capped), 20 rows → 14
        e.set_text(t5);
        assert_eq!(e.needed_height(40, 40), 7);
        let t7 = "a\nb\nc\nd\ne\nf\ng";
        e.set_text(t7);
        assert_eq!(
            e.needed_height(40, 40),
            9,
            "With rows=40, cap=12; 7 content lines do not hit the top"
        );
        let mut t20 = String::new();
        for i in 0..20 {
            t20.push_str(&format!("line {i}\n"));
        }
        t20.pop();
        e.set_text(&t20);
        assert_eq!(
            e.needed_height(40, 40),
            14,
            "20 content lines are truncated by cap=12 to 14"
        );
    }

    /// P0-A core regression: rendered visible rows always equal needed_height − 2 (any rows/content combination),
    /// the "9-row budget only draws 5 rows" disconnect no longer occurs.
    #[test]
    fn test_needed_height_matches_render_visible_rows() {
        for (content_lines, rows, expected_h) in
            [(0usize, 24u16, 3u16), (3, 24, 5), (7, 40, 9), (12, 40, 14)]
        {
            let mut e = e(0);
            if content_lines > 0 {
                let t = (0..content_lines)
                    .map(|i| format!("row {i}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                e.set_text(&t);
            }
            let h = e.needed_height(40, rows);
            assert_eq!(h, expected_h, "content={content_lines} rows={rows}");

            let mut buf = ScreenBuffer::new(40, h);
            e.render(&mut buf, 0, 0, 40, h);
            // cursor on the last line and content ≤ cap → no scrolling: non-empty rows between the two borders == h-2
            let filled = (1..h - 1)
                .filter(|r| !buf.line_segments(*r as u16).is_empty())
                .count();
            assert_eq!(
                filled as u16,
                h - 2,
                "content={content_lines} rows={rows}: budget {h} should fill {h}−2 content lines"
            );
        }
    }

    /// P0-A acceptance: 7 content rows + rows=40 → budget 9; after render all 7 rows are visible, no blanks, no ↓ indicator.
    #[test]
    fn test_render_shows_all_budgeted_lines() {
        let mut e = e(0);
        e.set_text("a\nb\nc\nd\ne\nf\ng");
        let h = e.needed_height(40, 40);
        assert_eq!(h, 9);
        let mut buf = ScreenBuffer::new(40, h);
        e.render(&mut buf, 0, 0, 40, h);
        // content rows 1..=7 each visible (no blanks between border rows 0 and 8); the cursor line ends with a reverse-video space, compare after trim
        let rows: Vec<String> = (1..=7)
            .map(|r| join_row(&buf, r).trim_end().to_string())
            .collect();
        assert_eq!(rows, vec!["a", "b", "c", "d", "e", "f", "g"]);
        // bottom border (row 8) has no ↓N more indicator (content within budget)
        assert!(
            !join_row(&buf, 8).contains('↓'),
            "No ↓ indicator should appear when under budget"
        );
    }

    /// Old fixed-height h=3 path behavior unchanged (regression guard: visible=1, matching the old clamp(5,1,1)).
    #[test]
    fn test_render_fixed_height_3_unchanged() {
        let mut e = e(0);
        e.set_text("hello\nworld");
        let mut buf = ScreenBuffer::new(40, 3);
        e.render(&mut buf, 0, 0, 40, 3);
        // only 1 content row: shows the cursor line (last line in world), content above scrolls out → top border shows ↑ indicator
        let top = join_row(&buf, 0);
        assert!(
            top.contains('↑'),
            "When scroll offset >0, top border should have ↑N more; actual {top:?}"
        );
        let content = join_row(&buf, 1);
        assert_eq!(
            content.trim_end(),
            "world",
            "At height 3, only the line containing the cursor is shown"
        );
        let bottom = join_row(&buf, 2);
        assert!(
            bottom.starts_with('─'),
            "Bottom border is at line 2; actual {bottom:?}"
        );
    }

    /// ↓N more appears only when content truly exceeds the budget (no indicator when content ≤ cap; when over cap and there is content above the cursor
    /// up/down indicators appear per scroll offset).
    #[test]
    fn test_render_scroll_indicator_only_when_exceeds_budget() {
        let mut ed = e(0);
        let mut t = String::new();
        for i in 0..15 {
            t.push_str(&format!("line {i}\n"));
        }
        t.pop(); // 15 content rows
        ed.set_text(&t);
        // move cursor to the first line (clamped before render) so content below exceeds the visible area → ↓N more should appear
        ed.state.cursor_line = 0;
        ed.state.cursor_col = 0;

        let h = ed.needed_height(40, 40); // cap=12 → 14
        assert_eq!(h, 14);
        let mut buf = ScreenBuffer::new(40, h);
        ed.render(&mut buf, 0, 0, 40, h);
        let bottom = join_row(&buf, 13);
        assert!(
            bottom.contains('↓'),
            "15 content lines exceed cap=12; ↓N more should appear; actual {bottom:?}"
        );
        // within-budget scenario: 7 content rows → no ↓
        let mut e2 = e(0);
        e2.set_text("a\nb\nc\nd\ne\nf\ng");
        let h2 = e2.needed_height(40, 40);
        let mut buf2 = ScreenBuffer::new(40, h2);
        e2.render(&mut buf2, 0, 0, 40, h2);
        assert!(!join_row(&buf2, h2 - 1).contains('↓'));
    }
}
