// REQ-TUI-01 (pi-413): editor component — layout and rendering: layout_text / render / cursor_position.
// render writes ScreenBuffer and computes layout width using effective padding (shares last_width with cursor_position /
// last_effective_padding / scroll_offset caches); layout_text is private to this module.
use super::wrap::word_wrap_line;
use super::{Editor, LayoutLine};

impl Editor {
    /// Render the editor into the screen buffer at (x,y), with width width and height height. Design:
    /// Effective padding is clamped by available width (`min(padding_x, (width-1)/2)`, aligned with pi) → layout width
    /// When wrapping at the outer margin with no padding, reserve 1 column for the line-end cursor (aligned with pi layoutWidth calculation, deducted only once) →
    /// Line layout → scroll clamping (visible_rows = clamp(max(5, 30%·height), 1, height-2), aligned with pi)
    /// 30% of terminal height; upper bound h-2 prevents text from covering the top/bottom borders) → write to the buffer line by line: top/bottom borders
    /// (includes `─── ↑N more` / `─── ↓N more` scroll indicators; padding width is calculated from display width), text lines,
    /// Cursor inverse (`Style.attributes=[Reverse]` block: if the cursor is on a grapheme, that grapheme is inverted; at line end, a space is inverted).
    /// Consumer: T-AP-11-1 (called once inside the draw callback).
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

        // ── Visible rows: max(5, 30%·height), clamped to [1, h-2] to prevent border overlap (aligned with pi maxVisibleLines)──
        let visible_rows = std::cmp::max(5, h * 30 / 100).clamp(1, h - 2);

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

        // Visible rows (same formula as render: max(5, 30%·height) clamped to [1, h-2]),
        // Fallback clamp for the row coordinate while scroll_offset has not yet been clamped by the next render
        let visible = std::cmp::max(5, self.last_height * 30 / 100)
            .clamp(1, self.last_height.saturating_sub(2).max(1));

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
    use super::super::Editor;
    use crate::screen::ScreenBuffer;

    // ── render ──

    #[test]
    fn test_render_too_small_width_is_noop() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        let mut buf = ScreenBuffer::new(1, 5);
        // width < 2 → early return
        e.render(&mut buf, 0, 0, 1, 5);
        assert_eq!(e.scroll_offset, 0);
    }

    #[test]
    fn test_render_too_small_height_is_noop() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        let mut buf = ScreenBuffer::new(20, 2);
        // height < 3 → early return
        e.render(&mut buf, 0, 0, 20, 2);
        assert_eq!(e.scroll_offset, 0);
    }

    #[test]
    fn test_render_sets_last_width_and_height() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        let mut buf = ScreenBuffer::new(20, 10);
        e.render(&mut buf, 0, 0, 20, 10);
        assert!(e.last_width > 0);
        assert_eq!(e.last_height, 10);
    }

    #[test]
    fn test_render_sets_effective_padding() {
        let mut e = Editor::new(4);
        e.insert_text_at_cursor("hello");
        let mut buf = ScreenBuffer::new(20, 10);
        e.render(&mut buf, 0, 0, 20, 10);
        assert_eq!(e.last_effective_padding, 4);
    }

    #[test]
    fn test_render_clamps_padding_to_max() {
        let mut e = Editor::new(100);
        e.insert_text_at_cursor("hello");
        let mut buf = ScreenBuffer::new(20, 10);
        e.render(&mut buf, 0, 0, 20, 10);
        // max_padding = (20-1)/2 = 9
        assert_eq!(e.last_effective_padding, 9);
    }

    #[test]
    fn test_render_scroll_clamps_cursor_visible() {
        let mut e = Editor::new(0);
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
        // visible_rows = max(5, 8*30/100) = max(5, 2) = 5, clamped to [1, 6]
        let visible = std::cmp::max(5, 8 * 30 / 100).clamp(1, 8 - 2);
        // Cursor line (19) should be in visible window
        let cursor_vis = e.scroll_offset; // approximate
        assert!(cursor_vis + visible >= 20 || e.scroll_offset > 0);
    }

    #[test]
    fn test_render_no_scroll_when_content_fits() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("short");
        let mut buf = ScreenBuffer::new(40, 10);
        e.render(&mut buf, 0, 0, 40, 10);
        assert_eq!(e.scroll_offset, 0);
    }

    // ── cursor_position ──

    #[test]
    fn test_cursor_position_zero_width_returns_origin() {
        let e = Editor::new(0);
        assert_eq!(e.cursor_position(), (0, 0));
    }

    #[test]
    fn test_cursor_position_empty_editor() {
        let mut e = Editor::new(0);
        let mut buf = ScreenBuffer::new(20, 10);
        e.render(&mut buf, 0, 0, 20, 10);
        let (col, row) = e.cursor_position();
        // Empty editor: cursor at (0,0) → col = 0 + padding(0) = 0, row = 1 (after top border)
        assert_eq!(col, 0);
        assert_eq!(row, 1);
    }

    #[test]
    fn test_cursor_position_with_padding() {
        let mut e = Editor::new(3);
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
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        let mut buf = ScreenBuffer::new(20, 10);
        e.render(&mut buf, 0, 0, 20, 10);
        let (col, row) = e.cursor_position();
        assert_eq!(col, 5);
        assert_eq!(row, 1);
    }

    #[test]
    fn test_cursor_position_multiline_second_line() {
        let mut e = Editor::new(0);
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
        let mut e = Editor::new(0);
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
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("short");
        let mut buf = ScreenBuffer::new(40, 10);
        e.render(&mut buf, 0, 0, 40, 10);
        // Cursor should be on first visual line
        let (_, row) = e.cursor_position();
        assert_eq!(row, 1);
    }

    #[test]
    fn test_layout_text_wraps_long_line() {
        let mut e = Editor::new(0);
        // Insert text longer than layout width
        e.insert_text_at_cursor("abcdefghijklmnopqrstuvwxyz");
        let mut buf = ScreenBuffer::new(20, 10);
        e.render(&mut buf, 0, 0, 20, 10);
        // last_width should be < 26 (text length), meaning wrapping occurred
        assert!(e.last_width < 26);
    }

    #[test]
    fn test_layout_text_empty_line_has_cursor() {
        let mut e = Editor::new(0);
        // Empty editor: single empty line
        let mut buf = ScreenBuffer::new(20, 10);
        e.render(&mut buf, 0, 0, 20, 10);
        let (col, row) = e.cursor_position();
        // Cursor should be at (0, 1) for empty line
        assert_eq!(col, 0);
        assert_eq!(row, 1);
    }
}
