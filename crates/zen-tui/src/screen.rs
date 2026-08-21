use crate::terminal::TerminalError;
use crossterm::queue;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// Describes the foreground color, background color, and text attribute set (bold/underline, etc.) of a cell. Colors are represented as crossterm Color tags, so the theme system (T-AP-11-3) only needs to resolve theme colors into crossterm Color to write them to the buffer; this type only carries "already-resolved concrete styles", not theme rendering logic. Whether a separate Style type is needed (instead of reusing crossterm's ContentStyle) is determined at implementation time — if crossterm's ContentStyle suffices, a type alias can replace this struct, in which case delete this design item and note it in the file.
#[derive(Default, Clone, PartialEq, Debug)]
pub struct Style {
    fg: Option<crossterm::style::Color>, // Foreground color; None means inherit terminal default
    bg: Option<crossterm::style::Color>, // Background color; None means inherit terminal default
    attributes: Vec<crossterm::style::Attribute>, // Text attributes; whether to use a Vec is decided at implementation time; must be Default-able (empty) = no styles
}

/// A single position cell in the screen grid, carrying the display content and style of that column/row.
/// Blank cells are expressed with Option (rather than a separate "blank" marker) so differential refresh can determine "this cell needs to be cleared".
/// kind distinguishes a normal space, a wide-character start cell, and a wide-character continuation cell — continuation cells matter for differential refresh and overwrite rules: a wide character is written once at its start cell, its continuation cells only hold space and are never printed, and cannot be overwritten by single-column writes (width-aware writers won't step past them).
#[derive(Default, Clone, PartialEq, Debug)]
struct Cell {
    kind: CellKind,
    content: Option<String>, // Content of this cell; None/blank = no content (corresponds to the clear state)
    style: Style,            // Cell style; Default yields the blank style
}

#[derive(Clone, Default, PartialEq, Debug)]
enum CellKind {
    #[default]
    Empty,
    Char,
    Continuation,
}

/// Memory buffer for the whole frame. Holds the current frame grid and size; components write content into the current frame via write;
/// flush compares the current frame with the previous one cell by cell, generating crossterm commands only for changed cells (differential refresh); after writing, the current frame is promoted to become the new previous frame.
/// resize handles terminal size changes (including clearing extra rows when content shrinks).
pub struct ScreenBuffer {
    cells: Vec<Cell>,    // Current frame grid; flattened vs nested is an implementation choice, indexed using width
    width: u16,          // Current frame column count
    height: u16,         // Current frame row count
    previous: Vec<Cell>, // Snapshot of the previous frame, used for differential refresh comparison; guaranteed to be the same size as cells
}

impl ScreenBuffer {
    /// Create an empty buffer with the given size.
    pub fn new(width: u16, height: u16) -> Self {
        let size = (width as usize) * (height as usize);
        Self {
            // vec![Cell::default(); size] internally constructs one value then clones it size times
            cells: vec![Cell::default(); size],
            width,
            height,
            previous: vec![Cell::default(); size],
        }
    }

    /// Resize the buffer; when shrinking, clear out-of-bounds content; when expanding, fill with blank cells.
    pub fn resize(&mut self, width: u16, height: u16) {
        if width == self.width && height == self.height {
            return; // Size unchanged, no-op
        }

        let old_width = self.width as usize;
        let old_height = self.height as usize;
        let new_width = width as usize;
        let new_height = height as usize;
        let new_size = new_width * new_height;

        // Build a new buffer, copying only the overlapping region's existing content
        let mut new_cells = vec![Cell::default(); new_size];
        let copy_rows = old_height.min(new_height);
        let copy_cols = old_width.min(new_width);
        for row in 0..copy_rows {
            let old_offset = row * old_width;
            let new_offset = row * new_width;
            for col in 0..copy_cols {
                new_cells[new_offset + col] = self.cells[old_offset + col].clone();
            }
        }

        self.cells = new_cells;
        self.width = width;
        self.height = height;
        // Reset previous to all defaults → next flush() treats every cell as "changed", forcing a full-screen redraw
        self.previous = vec![Cell::default(); new_size];
    }

    /// Clear the current frame (set all cells to the blank style), for use before a redraw.
    pub fn clear(&mut self) {
        self.cells.fill(Cell::default());
    }

    /// Return buffer dimensions (columns, rows) for the draw callback to lay out (top padding, width distribution, etc.).
    /// Consumer: this task's `draw` callback; planned consumers: T-TU-01-4 (component layout), T-AP-11-2 (app components) — reuse uniformly when it recurs.
    pub fn size(&self) -> (u16, u16) {
        (self.width, self.height)
    }

    /// Write text at (x,y) with the given style; out-of-range coordinates are silently ignored (or the implementation may choose to panic — must be documented).
    /// Width-aware: place text by display column width using unicode-width; wide characters (CJK/fullwidth) occupy 2 columns and mark a continuation cell;
    /// When text exceeds the right boundary, truncate by display width without cutting a wide character in half (extra half cell dropped).
    /// Single-line positioning primitive, deliberately no line wrapping
    pub fn write(&mut self, x: u16, y: u16, style: &Style, text: &str) {
        let w = self.width as usize;
        let h = self.height as usize;
        let x = x as usize;
        let y = y as usize;

        // Silently ignore out-of-range: y exceeds height or x exceeds width
        if y >= h || x >= w {
            return;
        }

        let row_offset = y * w;
        let mut col = x;

        for ch in text.chars() {
            let char_width = match ch.width() {
                Some(w) if w > 0 => w,
                _ => continue, // Zero-width chars (control characters, combining marks, etc.) are skipped; they occupy no cell
            };

            // Truncation never splits a wide char: stop when the remaining space is less than a full character width (discard the extra half-cell)
            if col + char_width > w {
                break;
            }

            // --- Orphaned continuation-cell cleanup (three scenarios, see table above) ---

            // Scenario 1: current cell is a continuation cell of some wide char → clear that wide char's starting cell
            // Look back at most 1 cell (max wide char width is 2) to find the starting cell with Char(width=2)
            if col > 0 && matches!(self.cells[row_offset + col].kind, CellKind::Continuation) {
                let prev = col - 1;
                if matches!(self.cells[row_offset + prev].kind, CellKind::Char)
                    && self.cells[row_offset + prev]
                        .content
                        .as_ref()
                        .map_or(false, |c| c.width() == 2)
                {
                    self.cells[row_offset + prev] = Cell::default();
                }
            }

            // Scenario 2+3: current cell is the starting cell of some wide char → clear its continuation cell (only left over by narrow chars; overwriting a wide char needs no extra clearing)
            // Also covers the edge case where clearing the starting cell in scenario 1 makes the continuation cell the new starting cell
            if matches!(self.cells[row_offset + col].kind, CellKind::Char)
                && self.cells[row_offset + col]
                    .content
                    .as_ref()
                    .map_or(false, |c| c.width() == 2)
                && col + 1 < w
            {
                self.cells[row_offset + col + 1] = Cell::default();
            }

            // --- Place new character ---
            self.cells[row_offset + col] = Cell {
                kind: CellKind::Char,
                content: Some(ch.to_string()),
                style: style.clone(),
            };

            // Wide char continuation cell: occupies the cell but prints nothing, content is None
            if char_width == 2 {
                self.cells[row_offset + col + 1] = Cell {
                    kind: CellKind::Continuation,
                    content: None,
                    style: Style::default(),
                };
            }

            col += char_width;
        }
    }

    /// Differential refresh: compare the current frame with the previous frame cell by cell, writing crossterm commands (MoveTo + style + content) for changed cells to `out`;
    /// Cells whose content became empty/cleared are written as blanks to overwrite; afterwards promote the current frame to the previous frame. Returns io errors.
    /// Consumer: T-TU-01-1 (called after each render in the main loop).
    pub fn flush<W: std::io::Write>(&mut self, out: &mut W) -> Result<(), TerminalError> {
        let w = self.width as usize;
        let h = self.height as usize;
        let size = w * h;

        // ── Pass 1: mark dirty cells ──
        // When a wide char's starting cell is cleared, its continuation cell still lingers on the terminal (the terminal doesn't know it belonged to the vanished wide char);
        // the continuation cell must be force-marked dirty as well, so that pass 2 outputs a blank to overwrite it.
        let mut dirty = vec![false; size];
        for i in 0..size {
            if self.cells[i] != self.previous[i] {
                dirty[i] = true;
                // If a wide char's starting cell changed (cleared or replaced), propagate to its continuation cell
                if matches!(self.previous[i].kind, CellKind::Char)
                    && self.previous[i]
                        .content
                        .as_ref()
                        .map_or(false, |c| c.width() == 2)
                    && i + 1 < size
                {
                    dirty[i + 1] = true;
                }
            }
        }

        // ── Pass 2: output changed cells ──
        // Without optimization:  SetColors SetAttr Print('A') SetColors SetAttr Print('B') SetColors SetAttr Print('C')
        // With optimization:  SetColors SetAttr Print('A') Print('B') Print('C')
        // Style of the previous character
        let mut prev_style = Style::default();
        let mut style_dirty = true; // First cell forcibly outputs style

        for row in 0..h {
            let row_offset = row * w;
            for col in 0..w {
                let i = row_offset + col;
                if !dirty[i] {
                    continue;
                }

                let cell = &self.cells[i];

                // Skip continuation cells — continuation cells only occupy space, never print; if marked dirty due to orphan propagation, just write blanks over them below
                // But continuation cells themselves emit no content command (they have no content)
                if matches!(cell.kind, CellKind::Continuation) {
                    // Orphan continuation cell: must write spaces on the terminal to cover the residual right half of a wide character
                    queue!(
                        out,
                        crossterm::cursor::MoveTo(col as u16, row as u16),
                        crossterm::style::Print(' ')
                    )
                    .map_err(TerminalError::Io)?;
                    continue;
                }

                // Style change detection: only emit SetColors + SetAttribute commands when the style changes
                if cell.style != prev_style || style_dirty {
                    queue!(
                        out,
                        crossterm::style::SetColors(crossterm::style::Colors {
                            foreground: cell.style.fg,
                            background: cell.style.bg,
                        })
                    )
                    .map_err(TerminalError::Io)?;

                    // Attribute switching: Reset first, then Set one by one (Vec<Attribute> cannot do an incremental diff)
                    if style_dirty || cell.style.attributes != prev_style.attributes {
                        queue!(
                            out,
                            crossterm::style::SetAttribute(crossterm::style::Attribute::Reset)
                        )
                        .map_err(TerminalError::Io)?;
                        for attr in &cell.style.attributes {
                            queue!(out, crossterm::style::SetAttribute(*attr))
                                .map_err(TerminalError::Io)?;
                        }
                    }

                    prev_style = cell.style.clone();
                    style_dirty = false;
                }

                // Content: Char outputs the character, Empty outputs a blank
                let ch = cell
                    .content
                    .as_deref()
                    .and_then(|s| s.chars().next())
                    .unwrap_or(' ');

                queue!(
                    out,
                    crossterm::cursor::MoveTo(col as u16, row as u16),
                    crossterm::style::Print(ch)
                )
                .map_err(TerminalError::Io)?;
            }
        }

        // queue! only writes to the buffer without flush; at the end a single flush() sends everything in one syscall
        out.flush().map_err(TerminalError::Io)?;

        // ── Promote current frame to previous frame ──
        self.previous = self.cells.clone();

        Ok(())
    }
}

/// Returns the terminal display width of a string (CJK/full-width counted as 2 columns, control characters as 0 columns).
/// Consumer: this task's `write`/`flush`; **planned consumer: T-TU-03-6 (pi-408 utils, counterpart of `visibleWidth`)**.
pub fn visible_width(text: &str) -> usize {
    unicode_width::UnicodeWidthStr::width(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::style::Color;

    /// Accesses the cell at (x,y) in the buffer (private field, visible within the test module).
    fn cell(buf: &ScreenBuffer, x: usize, y: usize) -> &Cell {
        let w = buf.width as usize;
        &buf.cells[y * w + x]
    }

    fn color_style() -> Style {
        Style {
            fg: Some(Color::Red),
            bg: None,
            attributes: vec![],
        }
    }

    // ── new ──

    #[test]
    fn test_new_buffer_has_given_size_and_default_cells() {
        let buf = ScreenBuffer::new(4, 3);
        assert_eq!(buf.width, 4);
        assert_eq!(buf.height, 3);
        assert_eq!(buf.cells.len(), 12);
        assert_eq!(buf.previous.len(), 12);
        // All-default cell: Empty / no content / default style
        assert!(buf.cells.iter().all(|c| c == &Cell::default()));
        assert!(buf.previous.iter().all(|c| c == &Cell::default()));
    }

    #[test]
    fn test_new_zero_size_buffer() {
        let buf = ScreenBuffer::new(0, 0);
        assert!(buf.cells.is_empty());
        assert!(buf.previous.is_empty());
    }

    // ── resize ──

    #[test]
    fn test_resize_same_dimensions_is_noop() {
        let mut buf = ScreenBuffer::new(3, 3);
        buf.write(0, 0, &color_style(), "A");
        buf.flush(&mut Vec::new()).unwrap(); // previous synced to the current frame
        let snapshot = buf.previous.clone();

        buf.resize(3, 3);
        // Same size: content and previous both kept as-is (unlike a real resize, which resets previous to force a redraw)
        assert_eq!(cell(&buf, 0, 0).content.as_deref(), Some("A"));
        assert_eq!(buf.previous, snapshot);
        assert_eq!(buf.previous, buf.cells);
    }

    #[test]
    fn test_resize_grow_preserves_top_left_content() {
        let mut buf = ScreenBuffer::new(2, 2);
        buf.write(0, 0, &color_style(), "A");
        buf.write(1, 1, &color_style(), "B");
        buf.resize(4, 4);
        assert_eq!(buf.width, 4);
        assert_eq!(buf.height, 4);
        assert_eq!(cell(&buf, 0, 0).content.as_deref(), Some("A"));
        assert_eq!(cell(&buf, 1, 1).content.as_deref(), Some("B"));
        // Newly extended region stays blank
        assert_eq!(cell(&buf, 3, 3).kind, CellKind::Empty);
        // previous reset → next flush redraws the whole screen
        assert!(buf.previous.iter().all(|c| c == &Cell::default()));
    }

    #[test]
    fn test_resize_shrink_drops_outside_content() {
        let mut buf = ScreenBuffer::new(4, 4);
        buf.write(3, 3, &color_style(), "A"); // Corner (will be cropped)
        buf.write(1, 1, &color_style(), "B"); // Interior (should be preserved)
        buf.resize(2, 2);
        assert_eq!(cell(&buf, 1, 1).content.as_deref(), Some("B"));
        // Out-of-bounds content is no longer in the new buffer
        assert!(buf.cells.iter().all(|c| c
            != &Cell {
                content: Some("A".to_string()),
                ..Cell::default()
            }));
    }

    // ── clear ──

    #[test]
    fn test_clear_empties_all_cells() {
        let mut buf = ScreenBuffer::new(3, 3);
        buf.write(0, 0, &color_style(), "A");
        buf.write(2, 2, &color_style(), "中");
        buf.clear();
        assert!(buf.cells.iter().all(|c| c == &Cell::default()));
    }

    // ── write: narrow character ──

    #[test]
    fn test_write_places_narrow_char_with_style() {
        let mut buf = ScreenBuffer::new(4, 2);
        let style = color_style();
        buf.write(1, 0, &style, "A");
        let c = cell(&buf, 1, 0);
        assert_eq!(c.kind, CellKind::Char);
        assert_eq!(c.content.as_deref(), Some("A"));
        assert_eq!(c.style, style);
        // Adjacent cells unaffected
        assert_eq!(cell(&buf, 0, 0).kind, CellKind::Empty);
    }

    #[test]
    fn test_write_out_of_bounds_ignored() {
        let mut buf = ScreenBuffer::new(3, 3);
        buf.write(3, 0, &color_style(), "A"); // x == width
        buf.write(0, 3, &color_style(), "B"); // y == height
        buf.write(100, 100, &color_style(), "C");
        assert!(buf.cells.iter().all(|c| c == &Cell::default()));
    }

    #[test]
    fn test_write_skips_zero_width_control_char() {
        let mut buf = ScreenBuffer::new(4, 1);
        buf.write(0, 0, &color_style(), "\t"); // Control characters have width()==None
        assert!(buf.cells.iter().all(|c| c == &Cell::default()));
    }

    // ── write: wide char ──

    #[test]
    fn test_write_wide_char_marks_continuation_cell() {
        let mut buf = ScreenBuffer::new(4, 1);
        buf.write(0, 0, &color_style(), "中");
        let lead = cell(&buf, 0, 0);
        assert_eq!(lead.kind, CellKind::Char);
        assert_eq!(lead.content.as_deref(), Some("中"));
        let cont = cell(&buf, 1, 0);
        assert_eq!(cont.kind, CellKind::Continuation);
        assert_eq!(cont.content, None);
    }

    #[test]
    fn test_write_truncates_wide_char_at_right_edge_without_splitting() {
        // width 4, starting column 3, only 1 column left for the wide char (needs 2) → entire wide char is dropped
        let mut buf = ScreenBuffer::new(4, 1);
        buf.write(3, 0, &color_style(), "中");
        // neither the start cell nor the continuation cell should be written
        assert_eq!(cell(&buf, 3, 0).kind, CellKind::Empty);
    }

    #[test]
    fn test_write_truncates_ascii_at_right_edge() {
        // width 3, starting column 2, only 1 character fits
        let mut buf = ScreenBuffer::new(3, 1);
        buf.write(2, 0, &color_style(), "ABC");
        assert_eq!(cell(&buf, 2, 0).content.as_deref(), Some("A"));
        assert_eq!(cell(&buf, 0, 0).kind, CellKind::Empty);
    }

    // ── write: overwrite / continuation cell cleanup ──

    #[test]
    fn test_write_narrow_over_wide_start_clears_continuation() {
        // scenario 2: narrow char overwrites a wide char's start cell → continuation cell is cleared as well
        let mut buf = ScreenBuffer::new(4, 1);
        buf.write(0, 0, &color_style(), "中");
        assert_eq!(cell(&buf, 1, 0).kind, CellKind::Continuation);

        buf.write(0, 0, &color_style(), "B");
        assert_eq!(cell(&buf, 0, 0).content.as_deref(), Some("B"));
        assert_eq!(cell(&buf, 1, 0).kind, CellKind::Empty);
    }

    #[test]
    fn test_write_over_continuation_clears_wide_start() {
        // scenario 1: narrow char lands on a continuation cell → wide char's start cell is cleared
        let mut buf = ScreenBuffer::new(4, 1);
        buf.write(0, 0, &color_style(), "中");
        buf.write(1, 0, &color_style(), "B");
        assert_eq!(cell(&buf, 0, 0).kind, CellKind::Empty);
        assert_eq!(cell(&buf, 1, 0).content.as_deref(), Some("B"));
    }

    // ── visible_width ──

    #[test]
    fn test_visible_width_measures_by_display_columns() {
        assert_eq!(visible_width("abc"), 3);
        assert_eq!(visible_width("中文"), 4); // 2 columns per wide char
        assert_eq!(visible_width(""), 0);
        assert_eq!(visible_width("a中"), 3);
        assert_eq!(visible_width("\u{200b}"), 0); // zero-width char
    }

    // ── flush ──

    #[test]
    fn test_flush_emits_changed_cells_once() {
        let mut buf = ScreenBuffer::new(3, 1);
        buf.write(0, 0, &color_style(), "A");

        let mut out = Vec::new();
        let res = buf.flush(&mut out);
        assert!(res.is_ok());
        assert!(out.contains(&b'A'));

        // previous has been synced to the current frame
        assert_eq!(buf.previous, buf.cells);
    }

    #[test]
    fn test_flush_unchanged_buffer_produces_no_output() {
        let mut buf = ScreenBuffer::new(3, 1);
        buf.write(0, 0, &color_style(), "A");
        let mut first = Vec::new();
        buf.flush(&mut first).unwrap();
        assert!(first.contains(&b'A'));

        // no changes → the second flush produces no bytes
        let mut second = Vec::new();
        let res = buf.flush(&mut second);
        assert!(res.is_ok());
        assert!(second.is_empty());
    }

    #[test]
    fn test_flush_emits_wide_char_and_skips_continuation_print() {
        let mut buf = ScreenBuffer::new(4, 1);
        buf.write(0, 0, &color_style(), "中");

        let mut out = Vec::new();
        buf.flush(&mut out).unwrap();
        // the wide char itself is emitted
        assert!(out.windows("中".len()).any(|w| w == "中".as_bytes()));
    }

    #[test]
    fn test_flush_cleared_wide_char_propagates_to_continuation() {
        let mut buf = ScreenBuffer::new(4, 1);
        // first frame renders a wide char
        buf.write(0, 0, &color_style(), "中");
        let mut first = Vec::new();
        buf.flush(&mut first).unwrap();

        // second frame clears that cell → the wide char's start cell is cleared, the continuation cell is propagated as dirty
        buf.clear();
        let mut second = Vec::new();
        buf.flush(&mut second).unwrap();
        // the continuation cell should be emitted as a space (the wide char's right half remaining on the terminal)
        // at least the second frame produces output (two overwrites: start cell + continuation cell)
        assert!(!second.is_empty());
    }

    #[test]
    fn test_flush_resize_forces_full_redraw() {
        let mut buf = ScreenBuffer::new(2, 1);
        buf.write(0, 0, &color_style(), "A");
        buf.flush(&mut Vec::new()).unwrap();
        // previous matches the current state
        assert_eq!(buf.previous, buf.cells);

        // resize resets previous → even if the content is unchanged, a full-screen redraw is performed
        buf.resize(4, 1);
        let mut out = Vec::new();
        buf.flush(&mut out).unwrap();
        assert!(out.contains(&b'A'));
    }
}
