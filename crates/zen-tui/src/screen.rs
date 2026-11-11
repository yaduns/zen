use crate::highlight::StyledSegment;
use crate::terminal::TerminalError;
use crossterm::queue;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// Describes the foreground color, background color, and text attribute set (bold/underline, etc.) of a cell. Colors are represented as crossterm Color tags, so the theme system (T-AP-11-3) only needs to resolve theme colors into crossterm Color to write them to the buffer; this type only carries "already-resolved concrete styles", not theme rendering logic. Whether a separate Style type is needed (instead of reusing crossterm's ContentStyle) is determined at implementation time — if crossterm's ContentStyle suffices, a type alias can replace this struct, in which case delete this design item and note it in the file.
#[derive(Default, Clone, PartialEq, Debug)]
pub struct Style {
    pub fg: Option<crossterm::style::Color>, // Foreground color; None means inherit terminal default
    pub bg: Option<crossterm::style::Color>, // Background color; None means inherit terminal default
    pub attributes: Vec<crossterm::style::Attribute>, // Text attributes; whether to use a Vec is decided at implementation time; must be Default-able (empty) = no styles
}

impl Style {
    /// Foreground color constructor—quickly constructs a Style containing only a foreground color for component/theme systems.
    pub fn fg(color: crossterm::style::Color) -> Self {
        Style {
            fg: Some(color),
            bg: None,
            attributes: Vec::new(),
        }
    }

    /// Background color constructor—quickly constructs a Style containing only a background color for component/theme systems.
    pub fn bg(color: crossterm::style::Color) -> Self {
        Style {
            fg: None,
            bg: Some(color),
            attributes: Vec::new(),
        }
    }

    /// Full constructor—constructs a Style containing foreground + background + attributes for the theme system.
    pub fn new(
        fg: Option<crossterm::style::Color>,
        bg: Option<crossterm::style::Color>,
        attributes: Vec<crossterm::style::Attribute>,
    ) -> Self {
        Style { fg, bg, attributes }
    }
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

/// Unified entry point of the render target (REQ-TUI-01 / REQ-APP-11, long-row-clip-fix): components write lines into "a target" instead of being bound to "the screen" — aligned with pi
/// Essence of `Component.render(width): string[]` (height is not a contract; line count is determined by content).
///
/// Only two methods (intentionally minimal): outside these two, no component in the repo touches any surface of the render target (verified: no calls to `size()`/`resize()`/
/// `line_segments()`; components use the input `width` to wrap lines themselves).
///
/// Two implementations: [`ScreenBuffer`] (fixed-height grid, serving **screen-bounded** consumers such as the status line/editor/footer/panels/alt-screen windows) and
/// [`LineBuffer`] (grows in height on demand, serving **line-stream** consumers such as `ChatRow`). Both are fixed-width horizontally, and write semantics are cell-for-cell identical
/// (both share the same cell-writing kernel [`write_row_cells`]), so the same component written into both targets yields the same visual result.
///
/// ⚠️ `fill`'s `height` is the number of rows to fill; the caller's budget `height` **must not** be passed in directly: the budget for the line-stream path is
/// `u16::MAX`; treating it as the `fill` height would fill tens of thousands of blank rows in [`LineBuffer`]. Verified: except for the grid-specific `Container`, all other components
/// pass a `fill` height that is always `1` or `painted` (the number of content rows).
///
/// Consumer: all `Component::render` implementations in `zen-tui`/`zen-app` (35 impls = 34 production + 1 test),
/// `zen-app` row rendering (`render_row_lines`).
pub trait RenderTarget {
    /// Positioned single-line write: width-aware (CJK occupies two cells), right edge truncates by display width without splitting wide characters, out-of-bounds coordinates are silently ignored.
    /// Semantics are **cell-for-cell identical** to [`ScreenBuffer::write`] (same kernel).
    fn write(&mut self, x: u16, y: u16, style: &Style, text: &str);

    /// Rectangular fill (background/blank rows): equivalent to writing same-style spaces row by row; wide-character boundary handling matches [`RenderTarget::write`].
    fn fill(&mut self, x: u16, y: u16, width: u16, height: u16, style: &Style);
}

/// Memory buffer for the whole frame. Holds the current frame grid and size; components write content into the current frame via write;
/// flush compares the current frame with the previous one cell by cell, generating crossterm commands only for changed cells (differential refresh); after writing, the current frame is promoted to become the new previous frame.
/// resize handles terminal size changes (including clearing extra rows when content shrinks).
pub struct ScreenBuffer {
    cells: Vec<Cell>, // Current frame grid; flattened vs nested is an implementation choice, indexed using width
    width: u16,       // Current frame column count
    height: u16,      // Current frame row count
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

    /// Extract an already-drawn row's content into a list of styled segments: in scrollback mode, message rows are first drawn into a temporary ScreenBuffer
    /// rendered, then converted by this method into `Vec<StyledSegment>` and pushed into the row stream — bridging "components draw onto a grid" and "canvas row stream"
    /// two rendering targets, with zero changes to the components themselves. Adjacent consecutive cells with the same style merge into one segment; interior space segments are kept
    /// (needed by the fixed-area placeholder bridge); trailing all-empty rows are dropped; wide-char continuation cells (placeholders, not printed) are skipped; out-of-range rows return an empty Vec.
    /// # Parameters
    /// - `row`: target row index (0-based; ≥ height returns empty)
    /// # Consumers
    /// zen-app `modes/interactive/ui.rs::render_scrollback`
    pub fn line_segments(&self, row: u16) -> Vec<StyledSegment> {
        let w = self.width as usize;
        if row as usize >= self.height as usize {
            return Vec::new();
        }
        let offset = row as usize * w;
        row_segments(&self.cells[offset..offset + w], w)
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
        write_row_cells(
            &mut self.cells[row_offset..row_offset + w],
            w,
            x,
            style,
            text,
        );
    }

    /// Fill a width×height rectangle starting at (x,y) with style (full-row background fill).
    /// Semantics = write styled spaces cell by cell: out-of-bounds silently ignored, wide-char awareness and orphan-cell cleanup identical to [`Self::write`]
    /// (overwriting a wide char's start cell clears its continuation cells too; overwriting a continuation cell clears its start cell — no half wide chars left).
    /// Implementation delegates to per-row [`Self::write`], reusing its orphan-cell cleanup logic rather than reimplementing it here.
    /// no-op when width/height is 0; out-of-bounds parts of the rectangle are clipped.
    /// Consumers: T-TU-01-4 components (Box/Text/TruncatedText background fill and placeholder rows).
    pub fn fill(&mut self, x: u16, y: u16, width: u16, height: u16, style: &Style) {
        if width == 0 || height == 0 {
            return;
        }
        let row = " ".repeat(width as usize);
        for dy in 0..height {
            self.write(x, y + dy, style, &row);
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
        // Without optimization:  SetColors SetAttr Print('cjk1') SetColors SetAttr Print('cjk2') SetColors SetAttr Print('cjk3')
        // With optimization:  SetColors SetAttr Print('cjk1') Print('cjk2') Print('cjk3')
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
                    // Orphan continuation cell: needs to write whitespace to the terminal to overwrite the remaining right half of a wide character.
                    // Only when the starting cell of the wide character associated with this continuation cell **has no printable content in the current frame** (cleared/replaced with empty or in
                    // farther right) before overwriting—when the starting cell has just printed a wide character, its right half is already occupied by that wide character,
                    // printing spaces again will corrupt the display (on a fully dirty first frame, wide characters become "x x x");
                    // Skip when the starting cell is this frame's Char (width 2).
                    let start_painted = col > 0
                        && matches!(self.cells[row_offset + col - 1].kind, CellKind::Char)
                        && self.cells[row_offset + col - 1]
                            .content
                            .as_ref()
                            .map_or(false, |c| c.width() == 2);
                    if !start_painted {
                        queue!(
                            out,
                            crossterm::cursor::MoveTo(col as u16, row as u16),
                            crossterm::style::Print(' ')
                        )
                        .map_err(TerminalError::Io)?;
                    }
                    continue;
                }

                // Style change detection: emit SetColors + SetAttribute commands only when style changes.
                // Order-sensitive: Reset first, then SetColors, then Set one by one (Reset is ESC[0m, which clears previous
                // set colors; SetColors must come after Reset for colors to survive).
                if cell.style != prev_style || style_dirty {
                    // Attribute switching: Reset first to clear residue, then SetColors to set colors, finally Set one by one (Vec<Attribute>
                    // Cannot do incremental diff; emits Reset+attributes only on attribute change/first cell).
                    if style_dirty || cell.style.attributes != prev_style.attributes {
                        queue!(
                            out,
                            crossterm::style::SetAttribute(crossterm::style::Attribute::Reset)
                        )
                        .map_err(TerminalError::Io)?;
                    }

                    queue!(
                        out,
                        crossterm::style::SetColors(crossterm::style::Colors {
                            foreground: cell.style.fg,
                            background: cell.style.bg,
                        })
                    )
                    .map_err(TerminalError::Io)?;

                    if style_dirty || cell.style.attributes != prev_style.attributes {
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

        // PF-06: no flush here. `queue!` only writes into `out` (the frame's `Vec<u8>` in both render paths); the whole
        // frame leaves in a single `write_all` at the end of the frame (`Terminal::flush_out`). Flushing here would
        // put the diff on the wire in pieces — exactly the chunked delivery PF-06 removes.

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

impl RenderTarget for ScreenBuffer {
    fn write(&mut self, x: u16, y: u16, style: &Style, text: &str) {
        ScreenBuffer::write(self, x, y, style, text);
    }

    fn fill(&mut self, x: u16, y: u16, width: u16, height: u16, style: &Style) {
        ScreenBuffer::fill(self, x, y, width, height, style);
    }
}

/// Line-stream render target (REQ-TUI-01 / REQ-APP-11, long-row-clip-fix): when PF-04's line cache misses, `render_row_lines` uses it to catch the component output's
/// entire line stream—the problem solved is that a 700-line assistant message must be fully displayed, not that a height cap should be chosen for the grid.
///
/// Line count has **no upper limit and no budget parameter** (aligned with pi: line count is determined by content). Writing to `y` beyond the current line count automatically appends blank rows; rows never written are just empty `Vec`
/// (they occupy no width space), so memory ≈ actually drawn rows × width × `Cell`, and no `previous` mirror is needed (line streams do not do per-frame diff; that is
/// `ScrollbackCanvas`'s job).
///
/// Width is still fixed (constructor parameter `width`); [`LineBuffer::into_lines`] output is **already truncated to width and blank rows are filtered out**—exactly
/// the precondition contract for `LineCache` and `ScrollbackCanvas::push_line_pre_truncated`.
///
/// Consumer: `zen-app` `modes::interactive::ui::render_row_lines`。
pub struct LineBuffer {
    width: u16,
    /// Row set that grows in height on demand; an empty `Vec` = the row exists but has no allocated cells (equivalent to all defaults).
    rows: Vec<Vec<Cell>>,
}

impl LineBuffer {
    /// Create an empty line buffer with width `width` (0 rows; rows are appended on demand as writes occur).
    /// Consumer: `zen-app` `render_row_lines`。
    pub fn new(width: u16) -> LineBuffer {
        LineBuffer {
            width,
            rows: Vec::new(),
        }
    }

    /// Current line count (= deepest row written by the component + 1; `usize`, the line-stream side is not affected by the count boundary of the component's `u16` return value).
    /// Consumer: diagnostics/probes and unit tests for `render_row_lines`.
    pub fn line_count(&self) -> usize {
        self.rows.len()
    }

    /// Take the full row set: adjacent same-style cells merged into segments, internal spaces preserved, wide-character continuation cells skipped, **blank rows filtered out**
    /// (same extraction logic `row_segments` as [`ScreenBuffer::line_segments`]).
    /// Consumer: `render_row_lines` (the result goes directly into `LineCache` and the line stream).
    pub fn into_lines(self) -> Vec<Vec<StyledSegment>> {
        let width = self.width as usize;
        self.rows
            .into_iter()
            .map(|cells| {
                if cells.is_empty() {
                    Vec::new()
                } else {
                    row_segments(&cells, width)
                }
            })
            .filter(|segs| !segs.is_empty())
            .collect()
    }

    /// Ensure row `y` has cells (grow height + allocate by width when the row is first written).
    fn ensure_row(&mut self, y: usize) {
        if y >= self.rows.len() {
            self.rows.resize(y + 1, Vec::new());
        }
        if self.rows[y].is_empty() {
            self.rows[y] = vec![Cell::default(); self.width as usize];
        }
    }
}

impl RenderTarget for LineBuffer {
    fn write(&mut self, x: u16, y: u16, style: &Style, text: &str) {
        let w = self.width as usize;
        if x as usize >= w {
            return;
        }
        self.ensure_row(y as usize);
        write_row_cells(&mut self.rows[y as usize], w, x as usize, style, text);
    }

    fn fill(&mut self, x: u16, y: u16, width: u16, height: u16, style: &Style) {
        if width == 0 || height == 0 {
            return;
        }
        let row = " ".repeat(width as usize);
        for dy in 0..height {
            let yy = y as usize + dy as usize;
            if yy > u16::MAX as usize {
                break; // Line number exceeds the range expressible by u16: the line-stream side does not accept a larger y (unreachable in practice)
            }
            self.write(x, yy as u16, style, &row);
        }
    }
}

/// Single-line cell-writing kernel: shared by [`ScreenBuffer::write`] and [`LineBuffer`] — width-aware, wide characters occupy two cells, right-edge truncation does not split wide characters,
/// cleans up orphaned continuation cells when overwriting wide characters. `cells` must be **a full row** (length = `width`); out-of-bounds `x` is filtered by the caller.
fn write_row_cells(cells: &mut [Cell], width: usize, x: usize, style: &Style, text: &str) {
    let mut col = x;

    for ch in text.chars() {
        let char_width = match ch.width() {
            Some(w) if w > 0 => w,
            _ => continue, // Zero-width chars (control characters, combining marks, etc.) are skipped; they occupy no cell
        };

        // Truncation never splits a wide char: stop when the remaining space is less than a full character width (discard the extra half-cell)
        if col + char_width > width {
            break;
        }

        // --- Orphaned continuation-cell cleanup (three scenarios, see the CellKind table above) ---

        // Scenario 1: current cell is a continuation cell of some wide char → clear that wide char's starting cell
        if col > 0 && matches!(cells[col].kind, CellKind::Continuation) {
            let prev = col - 1;
            if matches!(cells[prev].kind, CellKind::Char)
                && cells[prev].content.as_ref().is_some_and(|c| c.width() == 2)
            {
                cells[prev] = Cell::default();
            }
        }

        // Scenario 2+3: current cell is the starting cell of some wide char → clear its continuation cell
        if matches!(cells[col].kind, CellKind::Char)
            && cells[col].content.as_ref().is_some_and(|c| c.width() == 2)
            && col + 1 < width
        {
            cells[col + 1] = Cell::default();
        }

        // --- Place new character ---
        cells[col] = Cell {
            kind: CellKind::Char,
            content: Some(ch.to_string()),
            style: style.clone(),
        };

        // Wide char continuation cell: occupies the cell but prints nothing, content is None
        if char_width == 2 {
            cells[col + 1] = Cell {
                kind: CellKind::Continuation,
                content: None,
                style: Style::default(),
            };
        }

        col += char_width;
    }
}

/// Row-extraction kernel: shared by [`ScreenBuffer::line_segments`] and [`LineBuffer::into_lines`] — adjacent same-style cells merged, in-row spaces preserved,
/// wide-character continuation cells skipped, rows with no content return an empty `Vec`. `cells` must be **a full row** (length = `width`).
fn row_segments(cells: &[Cell], width: usize) -> Vec<StyledSegment> {
    let w = width.min(cells.len());

    // 1. Determine extent: the column after the last "non-empty" cell (explicit space / wide-char start or continuation).
    let end = cells[..w]
        .iter()
        .rposition(|c| !matches!(c.kind, CellKind::Empty))
        .map_or(0, |i| i + 1);
    if end == 0 {
        return Vec::new();
    }

    // 2. Scan columns in order, merging adjacent segments with the same style.
    let mut out: Vec<StyledSegment> = Vec::new();
    let mut cur_style: Option<Style> = None;
    let mut cur_text = String::new();
    for cell in &cells[..end] {
        match cell.kind {
            CellKind::Char => {
                let ch = cell
                    .content
                    .as_deref()
                    .and_then(|s| s.chars().next())
                    .unwrap_or(' ');
                if cur_style.as_ref() == Some(&cell.style) {
                    cur_text.push(ch);
                } else {
                    if let Some(s) = cur_style.take() {
                        if !cur_text.is_empty() {
                            out.push(StyledSegment {
                                style: s,
                                text: std::mem::take(&mut cur_text),
                            });
                        }
                    }
                    cur_style = Some(cell.style.clone());
                    cur_text.push(ch);
                }
            }
            CellKind::Continuation => {
                // Wide-char continuation cell: the start cell already covers two columns, so skip it (do not print the placeholder).
            }
            CellKind::Empty => {
                // Interior gap: append a space to the current segment to preserve position; an empty segment starts with the default style.
                if cur_style.is_some() {
                    cur_text.push(' ');
                } else {
                    cur_style = Some(Style::default());
                    cur_text.push(' ');
                }
            }
        }
    }
    if let Some(s) = cur_style.take() {
        if !cur_text.is_empty() {
            out.push(StyledSegment {
                style: s,
                text: cur_text,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    /// CJK sample kept intentionally: this test verifies wide-character / multibyte behavior.
    const CJK_SAMPLE: &str = "中";
    const CJK_SAMPLE_2: &str = "中文";
    const CJK_SAMPLE_3: &str = "a中";
    const CJK_SAMPLE_4: &str = "中x";
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
        buf.write(2, 2, &color_style(), CJK_SAMPLE);
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
        buf.write(0, 0, &color_style(), CJK_SAMPLE);
        let lead = cell(&buf, 0, 0);
        assert_eq!(lead.kind, CellKind::Char);
        assert_eq!(lead.content.as_deref(), Some(CJK_SAMPLE));
        let cont = cell(&buf, 1, 0);
        assert_eq!(cont.kind, CellKind::Continuation);
        assert_eq!(cont.content, None);
    }

    #[test]
    fn test_write_truncates_wide_char_at_right_edge_without_splitting() {
        // width 4, starting column 3, only 1 column left for the wide char (needs 2) → entire wide char is dropped
        let mut buf = ScreenBuffer::new(4, 1);
        buf.write(3, 0, &color_style(), CJK_SAMPLE);
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
        buf.write(0, 0, &color_style(), CJK_SAMPLE);
        assert_eq!(cell(&buf, 1, 0).kind, CellKind::Continuation);

        buf.write(0, 0, &color_style(), "B");
        assert_eq!(cell(&buf, 0, 0).content.as_deref(), Some("B"));
        assert_eq!(cell(&buf, 1, 0).kind, CellKind::Empty);
    }

    #[test]
    fn test_write_over_continuation_clears_wide_start() {
        // scenario 1: narrow char lands on a continuation cell → wide char's start cell is cleared
        let mut buf = ScreenBuffer::new(4, 1);
        buf.write(0, 0, &color_style(), CJK_SAMPLE);
        buf.write(1, 0, &color_style(), "B");
        assert_eq!(cell(&buf, 0, 0).kind, CellKind::Empty);
        assert_eq!(cell(&buf, 1, 0).content.as_deref(), Some("B"));
    }

    // ── fill ──

    #[test]
    fn test_fill_writes_styled_spaces_in_rect() {
        let mut buf = ScreenBuffer::new(4, 3);
        let style = color_style();
        buf.fill(1, 0, 2, 2, &style);
        // (1,0)(2,0)(1,1)(2,1) filled with styled spaces
        for (x, y) in [(1, 0), (2, 0), (1, 1), (2, 1)] {
            let c = cell(&buf, x, y);
            assert_eq!(c.kind, CellKind::Char);
            assert_eq!(c.content.as_deref(), Some(" "));
            assert_eq!(c.style, style);
        }
        // Outside the rectangle unaffected
        assert_eq!(cell(&buf, 0, 0).kind, CellKind::Empty);
        assert_eq!(cell(&buf, 3, 0).kind, CellKind::Empty);
        assert_eq!(cell(&buf, 0, 2).kind, CellKind::Empty);
    }

    #[test]
    fn test_fill_zero_size_is_noop() {
        let mut buf = ScreenBuffer::new(3, 3);
        buf.write(0, 0, &color_style(), "A");
        buf.fill(0, 0, 0, 2, &color_style());
        buf.fill(1, 1, 2, 0, &color_style());
        assert_eq!(cell(&buf, 0, 0).content.as_deref(), Some("A"));
        assert!(buf
            .cells
            .iter()
            .enumerate()
            .all(|(i, c)| i == 0 || c == &Cell::default()));
    }

    #[test]
    fn test_fill_out_of_bounds_ignored() {
        let mut buf = ScreenBuffer::new(3, 3);
        buf.fill(3, 0, 2, 2, &color_style()); // x == width
        buf.fill(0, 3, 2, 2, &color_style()); // y == height
        assert!(buf.cells.iter().all(|c| c == &Cell::default()));
    }

    #[test]
    fn test_fill_over_wide_char_start_clears_continuation() {
        let mut buf = ScreenBuffer::new(4, 1);
        buf.write(0, 0, &color_style(), CJK_SAMPLE);
        assert_eq!(cell(&buf, 1, 0).kind, CellKind::Continuation);

        // Fill overwrites a wide char's start cell → continuation cells cleared too; both cells become fill spaces
        buf.fill(0, 0, 1, 1, &color_style());
        assert_eq!(cell(&buf, 0, 0).content.as_deref(), Some(" "));
        assert_eq!(cell(&buf, 1, 0).kind, CellKind::Empty);
    }

    #[test]
    fn test_fill_over_continuation_clears_wide_start() {
        let mut buf = ScreenBuffer::new(4, 1);
        buf.write(0, 0, &color_style(), CJK_SAMPLE);
        // Fill overwrites a continuation cell → wide char's start cell cleared too (no half wide char left)
        buf.fill(1, 0, 1, 1, &color_style());
        assert_eq!(cell(&buf, 0, 0).kind, CellKind::Empty);
        assert_eq!(cell(&buf, 1, 0).content.as_deref(), Some(" "));
    }

    #[test]
    fn test_fill_rect_clips_at_buffer_edge() {
        let mut buf = ScreenBuffer::new(3, 2);
        buf.fill(1, 1, 10, 10, &color_style()); // Out-of-bounds parts clipped
        assert_eq!(cell(&buf, 2, 1).content.as_deref(), Some(" "));
        assert_eq!(cell(&buf, 0, 0).kind, CellKind::Empty);
    }

    // ── visible_width ──

    #[test]
    fn test_visible_width_measures_by_display_columns() {
        assert_eq!(visible_width("abc"), 3);
        assert_eq!(visible_width(CJK_SAMPLE_2), 4); // 2 columns per wide char
        assert_eq!(visible_width(""), 0);
        assert_eq!(visible_width(CJK_SAMPLE_3), 3);
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
    fn test_flush_reset_precedes_colors() {
        // Order contract: Reset (ESC[0m) must precede SetColors, otherwise colors are cleared by reset (regression protection for historical defect:
        // previous SetColors→Reset order caused TUI full color to fail, only attributes survived — T-TU-02-1-fix).
        let mut buf = ScreenBuffer::new(3, 1);
        buf.write(0, 0, &color_style(), "A");
        let mut out = Vec::new();
        buf.flush(&mut out).unwrap();
        let text = String::from_utf8_lossy(&out);
        let reset = text.find("\x1b[0m").expect("Should contain Reset sequence");
        let color = text
            .find("\x1b[38;5;9m")
            .or_else(|| text.find("\x1b[91m"))
            .or_else(|| text.find("\x1b[31m"))
            .expect("Should contain foreground color sequence");
        assert!(reset < color, "Reset must precede color command: {text:?}");
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
        buf.write(0, 0, &color_style(), CJK_SAMPLE);

        let mut out = Vec::new();
        buf.flush(&mut out).unwrap();
        // the wide char itself is emitted
        assert!(out
            .windows(CJK_SAMPLE.len())
            .any(|w| w == CJK_SAMPLE.as_bytes()));
    }

    #[test]
    fn test_flush_cleared_wide_char_propagates_to_continuation() {
        let mut buf = ScreenBuffer::new(4, 1);
        // first frame renders a wide char
        buf.write(0, 0, &color_style(), CJK_SAMPLE);
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

    // ── line_segments ──

    #[test]
    fn test_line_segments_empty_and_out_of_bounds() {
        let buf = ScreenBuffer::new(5, 3);
        // All-empty row → empty Vec
        assert!(buf.line_segments(0).is_empty());
        // Out-of-range row → empty Vec
        assert!(buf.line_segments(3).is_empty());
    }

    #[test]
    fn test_line_segments_merges_same_style_segments() {
        let mut buf = ScreenBuffer::new(10, 1);
        let red = color_style();
        buf.write(0, 0, &red, "ab");
        buf.write(2, 0, &Style::default(), "cd"); // Adjacent, no gap
        let segs = buf.line_segments(0);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].text, "ab");
        assert_eq!(segs[0].style, red);
        assert_eq!(segs[1].text, "cd");
    }

    #[test]
    fn test_line_segments_trailing_empties_dropped() {
        let mut buf = ScreenBuffer::new(8, 1);
        let red = color_style();
        buf.write(0, 0, &red, "ab"); // Last 6 columns all empty
        let segs = buf.line_segments(0);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "ab");
    }

    #[test]
    fn test_line_segments_interior_gap_preserved_as_spaces() {
        let mut buf = ScreenBuffer::new(10, 1);
        let red = color_style();
        buf.write(0, 0, &red, "a");
        buf.write(5, 0, &Style::default(), "b"); // Middle 1..5 empty
        let segs = buf.line_segments(0);
        let joined: String = segs.iter().map(|s| s.text.clone()).collect();
        assert_eq!(
            joined, "a    b",
            "Interior gaps should be preserved as spaces"
        );
    }

    #[test]
    fn test_line_segments_wide_char_not_duplicated() {
        let mut buf = ScreenBuffer::new(6, 1);
        let red = color_style();
        buf.write(0, 0, &red, CJK_SAMPLE); // Middle + continuation cell
        buf.write(2, 0, &Style::default(), "x");
        let segs = buf.line_segments(0);
        let joined: String = segs.iter().map(|s| s.text.clone()).collect();
        assert_eq!(
            joined, CJK_SAMPLE_4,
            "Wide character continuation cells should be skipped without producing placeholders"
        );
    }

    #[test]
    fn test_line_segments_multiline_isolated() {
        let mut buf = ScreenBuffer::new(5, 2);
        buf.write(0, 0, &color_style(), "A");
        buf.write(0, 1, &color_style(), "B");
        let row0: String = buf
            .line_segments(0)
            .iter()
            .map(|s| s.text.clone())
            .collect();
        let row1: String = buf
            .line_segments(1)
            .iter()
            .map(|s| s.text.clone())
            .collect();
        assert_eq!(row0, "A");
        assert_eq!(row1, "B");
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

    // ── LineBuffer（REQ-TUI-01 / REQ-APP-11，long-row-clip-fix）──

    /// Shared drawing sequence for diff cases: two preconditions (see test comments below)—
    /// ① all y < grid height; ② when comparing, the grid side also filters blank rows.
    fn paint_sequence(target: &mut dyn RenderTarget, width: u16) {
        let bg = Style::bg(crossterm::style::Color::Blue);
        let fg = Style::fg(crossterm::style::Color::Red);
        // full-row background → in-row gaps → wide characters → right-edge truncation → multi-row fill
        target.fill(0, 0, width, 1, &bg);
        target.write(0, 0, &fg, "hello");
        target.write(0, 1, &fg, CJK_SAMPLE_2);
        target.write(6, 2, &fg, "x");
        target.write(width - 2, 3, &fg, "abcdefgh");
        target.fill(0, 4, width, 2, &bg);
        target.write(2, 6, &fg, "tail");
    }

    #[test]
    fn test_line_buffer_matches_screen_buffer_for_the_same_paint_sequence() {
        // Differential test (the single most important one for LineBuffer): the same paint sequence must produce
        // identical lines on both targets ⇒ "no second rendering logic".
        // Two prerequisites are baked into `paint_sequence`:
        //   ① every y is < the grid height (20) — the grid silently drops out-of-range y while LineBuffer grows,
        //      and *that difference is the whole point of the fix*, so the diff can only compare inside the grid;
        //   ② the grid side applies the same `!segs.is_empty()` filter as `into_lines` (grid `line_segments` keeps
        //      empty rows as empty Vecs, `into_lines` drops them).
        const W: u16 = 20;
        const H: u16 = 20;
        let mut grid = ScreenBuffer::new(W, H);
        let mut lines = LineBuffer::new(W);
        paint_sequence(&mut grid, W);
        paint_sequence(&mut lines, W);

        let grid_rows: Vec<Vec<StyledSegment>> = (0..H)
            .map(|r| grid.line_segments(r))
            .filter(|segs| !segs.is_empty())
            .collect();
        assert_eq!(grid_rows, lines.into_lines());
    }

    #[test]
    fn test_line_buffer_fill_densifies_only_the_rows_it_writes() {
        // Locks the precondition of RenderTarget::fill's doc warning: a small rect grows the buffer only up to the
        // rows it actually touches (not to some huge budget).
        let mut lines = LineBuffer::new(10);
        lines.fill(0, 500, 10, 2, &color_style());
        assert_eq!(lines.line_count(), 502);
        assert_eq!(lines.into_lines().len(), 2);
    }

    #[test]
    fn test_line_buffer_grows_without_a_height_budget() {
        // No height parameter exists: writing far down just grows the buffer (the anti-fixed-height-cap property).
        let mut lines = LineBuffer::new(10);
        assert_eq!(lines.line_count(), 0);
        lines.write(0, 2000, &color_style(), "deep");
        assert_eq!(lines.line_count(), 2001);
        let only = lines.into_lines();
        assert_eq!(only.len(), 1, "untouched rows are empty and filtered out");
        assert_eq!(only[0][0].text, "deep");
    }

    #[test]
    fn test_line_buffer_filters_blank_rows_like_the_grid_path() {
        // Existing convention (shared with the pre-PF-04 streaming path): rows that were **never written** are dropped
        // from the line set. Note the contrast (also locked here): a row that was only `fill`ed with spaces IS content
        // (spaces are explicit cells), and both targets keep it.
        let mut lines = LineBuffer::new(6);
        lines.write(0, 0, &color_style(), "a");
        // row 1 untouched → filtered out
        lines.write(0, 2, &color_style(), "b");
        let out = lines.into_lines();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0][0].text, "a");
        assert_eq!(out[1][0].text, "b");

        let mut filled = LineBuffer::new(6);
        filled.fill(0, 0, 6, 1, &Style::default());
        assert_eq!(filled.into_lines().len(), 1);
    }

    #[test]
    fn test_line_buffer_lines_never_exceed_width() {
        // PF-04 contract: cached/pushed lines must already fit the width (no per-character width scan downstream).
        let mut lines = LineBuffer::new(8);
        lines.write(0, 0, &color_style(), "abcdefghijkl");
        lines.write(0, 1, &color_style(), "中文中文中文");
        for row in lines.into_lines() {
            let cols: usize = row.iter().map(|s| visible_width(&s.text)).sum();
            assert!(cols <= 8, "line wider than width: {cols}");
        }
    }
}
