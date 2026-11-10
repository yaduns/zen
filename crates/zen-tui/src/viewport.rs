// REQ-TUI-01 (pi-409): Terminal scrollback render model — content line stream + viewport advancement.
// See doRender main logic in reference/pi-original/packages/tui/src/tui.ts (viewportTop/prevViewportTop
// state + "\r\n".repeat(scroll) advancement + line-level diffing + extra-lines screen clear + synchronized output wrapping),
// converged per zen requirements into ScrollbackCanvas: the draw callback pushes this frame's content lines into `lines`, and flush diffs them against the previous frame
// Line-level diff: push the portion that overflows the screen through real terminal scrolling into scrollback — the user can scroll the terminal to review history,
// instead of only seeing the last screen as with cell-level fixed-line diffs. Remaining visible lines / bottom fixed region are repainted line by line.
// Complementary to `ScreenBuffer` (cell-level grid): this type is the rendering target for "line-level + terminal-scroll carrying".
// Pure runtime canvas, no serialization needs (implementation choices for T-TU-01-5 open questions 1/2/3 are explained in the comments below).
//
// **PF-05 (line-stream convergence)**: lines are stored as shared handles ([`LineSegs`]), not owned vectors. pi keeps its diff baseline as a
// plain reference alias (`this.previousLines = newLines`, `tui.ts:1616`), i.e. O(N) pointer moves; carrying `Vec<Vec<StyledSegment>>` here forced
// two deep copies per frame (row cache → canvas at `ui.rs:2236`, canvas → baseline in `flush`) — a porting regression this module now reverts
// ([`ScrollbackCanvas::promote_frame`] + `Rc::ptr_eq` in the row diff).

use crate::highlight::{LineSegs, StyledSegment};
use crate::screen::Style;
use crate::terminal::TerminalError;
use crossterm::queue;
use std::io::Write;
use std::rc::Rc;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// Line-stream canvas: in scrollback mode the interactive mode pushes each frame's content here line by line; on flush `TuiLoop`
/// handles "line-level diff + real overflow scrolling", so history lands in the terminal scrollback — the user can scroll the terminal to review,
/// instead of only seeing the last screen as with cell-level fixed-line diffs. The sole output target of the draw callback.
///
/// Complementary to `ScreenBuffer` (cell-level grid, covered by fixed-line diffs): this type is the rendering target for "line-level + terminal-scroll carrying",
/// owned by `TuiLoop::run_scrollback` and preserved across frames (`prev_lines`/`prev_viewport_top` are the diff base);
/// before each frame's draw, `TuiLoop` calls [`Self::clear`] to clear this frame's line stream.
///
/// **(choices for open questions 1/2)**: scroll advancement uses pi's `"\r\n".repeat(scroll)` for real line-by-line scrolling
/// (crossterm `ScrollUp` is not used; scroll region semantics introduce unexpected ghosting in chat scenarios);
/// the entire diff output is wrapped in crossterm `BeginSynchronizedUpdate`/`EndSynchronizedUpdate` (`\x1b[?2026h/l`)
/// to reduce flicker on large-frame refreshes. **(open question 3)**: fixed-region ghosting is handled via `\x1b[2K` line clear + repaint;
/// `first_changed < prev_viewport_top → full repaint` (pi fullRender); see [`Self::flush`] for the boundary.
pub struct ScrollbackCanvas {
    /// All content lines for this frame (including the fixed region's trailing lines); call `clear()` before draw.
    ///
    /// Shared handles (PF-05): the draw side pushes the very same allocations its row caches hold, so a cache hit costs one refcount bump.
    lines: Vec<LineSegs>,
    /// Previous frame's line-stream snapshot (line-level diff base; lines scrolled out of the terminal no longer participate in the diff).
    /// Promoted from `lines` by [`Self::promote_frame`] (a `swap`, never a copy).
    prev_lines: Vec<LineSegs>,
    /// Content row number of the current viewport top (terminal row = `content_row - viewport_top`).
    viewport_top: usize,
    /// Previous frame's viewport top (to decide whether to advance more / do a full repaint).
    prev_viewport_top: usize,
    /// Line-level truncation width (aligned with the column-width truncation semantics of `ScreenBuffer::write`).
    width: u16,
    /// Current terminal row count (for flush; kept in sync with the terminal via `set_size`).
    height: u16,
    /// Terminal row count of the previous frame (for size-change detection; `set_size` already resets the diff here, so height is constant inside flush).
    prev_height: u16,
    /// Content row index of the hardware cursor after the previous frame's render (used to compute physical displacement for the diff; aligns with pi's `hardwareCursorRow`).
    hardware_cursor_row: usize,
    /// This frame's "IME cursor anchor": (content line number, absolute column). Set every frame by draw side (editor rendering location),
    /// flush finalization positions the (hidden) hardware cursor here for system IME spelling string/candidate window anchoring
    /// (aligned with pi CURSOR_MARKER → positionHardwareCursor semantics); None when no focus/no editor.
    cursor_hint: Option<(usize, usize)>,
    /// PF-01: rows of the bottom fixed region (status + editor + dropdown + footer) that the draw side actually
    /// pushed into this frame's line stream. Registered by `set_fixed_rows`; consumed by frame metrics so that
    /// "did the fixed region get rewritten?" is answerable (and, from PF-06 on, as the anchored region's height).
    fixed_rows: usize,
    /// PF-01: content rows actually written by the most recent flush, as an inclusive `[first, last]` span.
    /// Reset to `None` at the start of every flush so a no-change frame never reports a stale span.
    last_repaint_span: Option<(usize, usize)>,
    /// PF-05: net advancement of the viewport top during the most recent flush (= content rows that left the screen into the scrollback).
    /// Reset to 0 at the start of every flush (not only in `clear`/`set_size`: a frame that returns early would otherwise republish the previous
    /// frame's value). See [`Self::last_scroll_rows`] for the exact semantics, including the deliberate 0 on the full-redraw path.
    last_scroll_rows: usize,
}

impl ScrollbackCanvas {
    /// Construct an empty canvas with the given per-line truncation width and terminal row count.
    /// Consumer: `TuiLoop::run_scrollback` (zen-tui assembly side).
    pub fn new(width: u16, height: u16) -> Self {
        ScrollbackCanvas {
            lines: Vec::new(),
            prev_lines: Vec::new(),
            viewport_top: 0,
            prev_viewport_top: 0,
            width,
            height,
            prev_height: height,
            hardware_cursor_row: 0,
            cursor_hint: None,
            fixed_rows: 0,
            last_repaint_span: None,
            last_scroll_rows: 0,
        }
    }

    /// Clear this frame's line stream; call before each draw (aligns with `render_frame`'s `buffer.clear()` contract);
    /// `prev_lines`/`prev_viewport_top` are kept as the previous frame's diff baseline.
    /// IME cursor anchor is also reset (re-set every frame by draw side, terminal-ime-anchor-fix), and so is the
    /// last repaint span (a cleared stream cannot describe the previous frame's writes any more).
    pub fn clear(&mut self) {
        self.lines.clear();
        self.cursor_hint = None;
        self.last_repaint_span = None;
        self.last_scroll_rows = 0;
    }

    /// Append a screen line to the end of the line stream (shared entry point for consecutive message-area lines and fixed-area tail lines).
    /// Styles/text of each segment in the line are preserved as-is; if total width exceeds `width`, truncate by column width (never split wide characters).
    /// Consumer: `InteractiveUi::render_scrollback`.
    pub fn push_line(&mut self, line: Vec<StyledSegment>) {
        self.lines.push(Rc::new(truncate_line(line, self.width)));
    }

    /// Append a line that the caller has **already truncated** to the canvas width — the cache-hit path, where the whole point is to
    /// skip the per-character width scan that [`ScrollbackCanvas::push_line`] performs on every segment.
    ///
    /// Precondition (caller's duty): every line must already fit `width()`. A cached row line satisfies this because the render width is
    /// part of the cache key (`CacheKey::width`) — rendering at another width simply misses and re-renders. Unlike `push_line`, an
    /// over-wide line is **not** truncated here (that would hide a broken caller and desynchronise the line geometry); the
    /// non-truncating precedent already exists in [`ScrollbackCanvas::push_text`].
    /// Consumer: zen-app `modes::interactive::ui::push_row_lines` (cache hit); planned PF-06 (fixed-region cached rows).
    pub fn push_line_pre_truncated(&mut self, line: LineSegs) {
        self.lines.push(line);
    }

    /// Convenience entry: push plain text as a single-segment line (default/specified style; used for fixed-area space padding, etc.).
    /// Consumer: `render_scrollback` (fixed-area placeholder lines, pending lines).
    pub fn push_text(&mut self, text: impl Into<String>, style: Style) {
        self.lines.push(Rc::new(vec![StyledSegment {
            style,
            text: text.into(),
        }]));
    }

    /// Total content rows this frame (= `lines.len()`); used by TuiLoop to decide whether to trigger scroll advance.
    pub fn content_height(&self) -> usize {
        self.lines.len()
    }

    /// Read back the **current frame's** line stream by line number (REQ-TUI-01 / REQ-APP-11, long-row-clip-fix): acceptance/diagnostics need to answer
    /// whether the tail of this 700-line long answer really made it into the canvas, whereas `content_height` only gives the total and cannot show content.
    ///
    /// ⚠️ Read window = after `draw`, **before** `flush`: `promote_frame()` at the end of `flush` swaps the current frame's line stream into `prev_lines`
    /// and clears `lines` (PF-05 invariant) ⇒ always `None` after flush, `content_height()` always 0.
    /// **Deliberately does not fall back to `prev_lines`**: falling back would hide this invariant, letting a test that casually adds a flush silently read the previous frame.
    ///
    /// Edge case: `index` out of bounds ⇒ `None`. Returns a shared handle (`LineSegs`, PF-05); callers must treat it as read-only.
    /// **pub, but the only current consumer is tests** (`zen-app`'s long-line acceptance test + this module's unit tests); the production path does not read line-stream contents (flush iterates the internal `lines` directly).
    pub fn line(&self, index: usize) -> Option<&LineSegs> {
        self.lines.get(index)
    }

    /// Set this frame's IME cursor anchor (editor caret's content line number + absolute column; pass None when no focus).
    /// flush finalization positions the (hidden) hardware cursor at this anchor, so system IME spelling string/candidate window
    /// draw in place (aligned with pi CURSOR_MARKER → positionHardwareCursor, see terminal-ime-anchor-fix).
    pub fn set_cursor_hint(&mut self, pos: Option<(usize, usize)>) {
        self.cursor_hint = pos;
    }

    /// Current number of content lines already pushed (`lines.len()`; when called before push, this is the content line number of the next line).
    /// Consumer: the draw side records the content line number corresponding to a given line at the push boundary (for IME cursor anchor row calculation).
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Current IME cursor anchor (read by tests and future consumers).
    pub fn cursor_hint(&self) -> Option<(usize, usize)> {
        self.cursor_hint
    }

    /// Register how many rows at the tail of this frame's line stream belong to the bottom fixed region
    /// (status + editor + dropdown + footer).
    ///
    /// Pass the rows **actually pushed** — `zen-app` computes them as `last_painted.map_or(0, |r| r + 1)` —
    /// not the theoretical `fixed_h` upper bound: the fixed buffer is allocated for the worst case, so its
    /// trailing rows are often never painted (footer without data returns early, dropdown inactive, …), and an
    /// upper bound would make "did the fixed region get rewritten?" look better than it is.
    ///
    /// Consumer: `zen-app` `InteractiveUi::render_scrollback` (every frame); read back by frame metrics, and by
    /// PF-06 as the height of the bottom anchored region.
    pub fn set_fixed_rows(&mut self, rows: usize) {
        self.fixed_rows = rows;
    }

    /// Rows registered by the draw side via [`Self::set_fixed_rows`] for the current frame.
    ///
    /// Boundary: reset to 0 by [`Self::set_size`] on a geometry change; a frame that bails out before drawing
    /// (e.g. zero width) leaves the previous value in place, but such a frame produces no meaningful metrics
    /// either (the line stream is empty).
    pub fn fixed_rows(&self) -> usize {
        self.fixed_rows
    }

    /// How many content rows **left the screen into the scrollback** during the most recent flush, i.e. the net advance of the viewport top.
    ///
    /// Scenario: a streaming frame appends `k` lines while the tail `F` rows sit at the bottom; the tail keeps its *target screen rows*, so the
    /// real advancement is performed by the terminal while the write loop walks past the bottom of the screen. The local `"\r\n".repeat(scroll)`
    /// count is **0 in the assembled case** (`F >= 1` ⇒ `first = prev_len - F` ⇒ not an append-start frame ⇒ `move_row <= prev_viewport_bottom`),
    /// which is why this accessor reports the viewport delta instead: on the diff path the delta equals `k`, the quantity tests and PF-06 can assert.
    ///
    /// Boundaries: (1) 0 when the content still fits the screen (viewport top stays 0) — expected, do not assert `k` there;
    /// (2) **0 on the full-redraw path by fiat** — there the advancement is the terminal's natural overflow while replaying the whole stream, not
    /// canvas advancement (a deliberate inconsistency; do not read it as a per-frame advancement budget);
    /// (3) reset at flush entry (next to `last_repaint_span = None`) and by [`Self::clear`] / [`Self::set_size`].
    /// Consumers: unit tests (append-scroll assertions); planned PF-06 (assertion input only — PF-06 computes its own `height - F` budget).
    pub fn last_scroll_rows(&self) -> usize {
        self.last_scroll_rows
    }

    /// Content rows **written** by the most recent flush, as an inclusive `[first, last]` span.
    ///
    /// `None` when the flush found nothing to write (reset at flush entry, so a no-change frame can never
    /// leak the previous frame's span). A full redraw writes every content row, hence `[0, line_total - 1]` —
    /// this is "rows written", not "visible repaint window".
    ///
    /// Consumers: PF-05 ("an append writes only the new rows") and PF-06 (fixed-region write count, derived
    /// together with `line_total` / `fixed_rows`).
    pub fn last_repaint_span(&self) -> Option<(usize, usize)> {
        self.last_repaint_span
    }

    /// Canvas column width (per-line truncation / single-line measured width); lets the app side build component render buffers per screen width.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Canvas row height (terminal rows; `set_size` syncs with the terminal each frame).
    /// Consumer: T-AP-11-1-fix (`render_scrollback`'s editor cap calculation needs terminal rows,
    /// previously only had a `width()` accessor).
    pub fn height(&self) -> u16 {
        self.height
    }

    /// Size sync: returns whether a change occurred. On change, clears the previous frame's diff and viewport state so the next
    /// flush performs a full redraw (aligns with pi's `widthChanged`/`heightChanged → fullRender`).
    pub(crate) fn set_size(&mut self, width: u16, height: u16) -> bool {
        let changed = width != self.width || height != self.height;
        self.width = width;
        self.height = height;
        if changed {
            self.prev_lines.clear();
            self.prev_viewport_top = 0;
            self.viewport_top = 0;
            self.hardware_cursor_row = 0;
            self.cursor_hint = None;
            // PF-01: the fixed-region registration and the repaint span belong to the old geometry; the next
            // flush is a full redraw, so a stale value would mislead the frame metrics.
            self.fixed_rows = 0;
            self.last_repaint_span = None;
            self.last_scroll_rows = 0;
        }
        changed
    }

    /// Per-line diff flush: diff this frame's `lines` against the previous frame line by line, pushing off-screen content into
    /// scrollback via real terminal scrolling, and redrawing visible lines / bottom fixed area. After writing, promote `lines` to `prev_lines` and update
    /// the viewport state.
    ///
    /// # Algorithm (aligned with pi's `doRender` main path; kitty image/overlay/cursor-marker handling removed)
    /// 1. First frame / after size change / changed region above the previous viewport (already scrolled away) / content shrank overall -> full redraw;
    /// 2. Otherwise line-level diff: if the target line is below the previous viewport bottom, first move the cursor to the screen bottom, then advance into scrollback with `"\r\n".repeat(scroll)`
    ///    then move the cursor to the first changed line and emit each line with `\r`/`\r\n` + `\x1b[2K` + content;
    /// 3. If old content has more lines than new content, clear the extra lines and move back; wrap everything in synchronized output.
    ///
    /// # Cost model (PF-05)
    /// The diff walks `0..max(prev_len, new_len)` (iteration count is O(rows), unchanged), but each iteration is a `Rc::ptr_eq` for rows the
    /// row cache served unchanged and only falls back to a content comparison when a handle changed ([`changed_range`]). The previous frame's
    /// stream is promoted with [`Self::promote_frame`] (`mem::swap`, zero allocations/copies) — pi does the same thing with
    /// `this.previousLines = newLines` (`tui.ts:1616`).
    pub(crate) fn flush<W: Write>(&mut self, out: &mut W) -> Result<(), TerminalError> {
        let height = (self.height as usize).max(1);
        let width = self.width as usize;
        let new_len = self.lines.len();
        let prev_len = self.prev_lines.len();
        // PF-01: this span always describes the *current* flush only; clearing it here means the early-return
        // paths below (nothing changed, or a redraw that fails) can never leave a stale window behind.
        self.last_repaint_span = None;
        // PF-05: same rule for the advancement metric — reset at the entry (not only in `clear`/`set_size`) so a frame that returns early
        // or takes the full-redraw path can never republish the previous frame's value.
        self.last_scroll_rows = 0;
        let entry_viewport_top = self.prev_viewport_top;

        // First / last changed content row (`ptr_eq` fast path, content fallback; index-wise, missing rows count as changed).
        let changed = changed_range(&self.prev_lines, &self.lines);

        let Some((first, last)) = changed else {
            // No visible change: promote anyway (semantically a no-op, keeps the "after flush `lines` is empty" invariant) and return.
            self.promote_frame();
            return Ok(());
        };

        // Full redraw path: first frame / changed region above the previous viewport (already scrolled away) / content shrank overall.
        if needs_full_render(
            changed,
            &self.prev_lines,
            &self.lines,
            self.prev_viewport_top,
        ) {
            self.full_render(out, width, height)?;
            return Ok(());
        }

        // ── Line-level diff (aligned with pi main path) ──────────────────────────────────
        let appended = new_len > prev_len;
        let append_start = appended && first == prev_len && prev_len > 0;
        // In the append case the "target line" is first-1 (first line only wraps, not cleared), otherwise it is first.
        let move_row = if append_start { first - 1 } else { first };

        out.write_all(b"\x1b[?2026h")?; // Begin synchronized output

        let mut prev_vpt = self.prev_viewport_top;
        let mut hw = self.hardware_cursor_row;
        let prev_viewport_bottom = prev_vpt + height - 1;

        // Target line is below the previous viewport bottom: first move to the screen bottom, then advance into scrollback (`"\r\n".repeat(scroll)`,
        // aligned with pi open issue 1's choice -- real per-line scrolling, history stays in scrollback).
        if move_row > prev_viewport_bottom {
            let current_screen_row = hw.saturating_sub(prev_vpt).min(height - 1);
            let move_to_bottom = (height - 1).saturating_sub(current_screen_row);
            if move_to_bottom > 0 {
                out.write_all(format!("\x1b[{move_to_bottom}B").as_bytes())?;
            }
            let scroll = move_row - prev_viewport_bottom;
            out.write_all("\r\n".repeat(scroll).as_bytes())?;
            prev_vpt += scroll;
            hw = move_row;
        }

        // Move the cursor to the screen row of the first changed line (lineDiff = targetScreenRow - currentScreenRow).
        let target_screen_row = move_row.saturating_sub(prev_vpt);
        let current_screen_row = hw.saturating_sub(prev_vpt);
        let line_diff = target_screen_row as isize - current_screen_row as isize;
        if line_diff > 0 {
            out.write_all(format!("\x1b[{line_diff}B").as_bytes())?;
        } else if line_diff < 0 {
            out.write_all(format!("\x1b[{}A", -line_diff).as_bytes())?;
        }
        out.write_all(if append_start { b"\r\n" } else { b"\r" })?;

        // Render only the changed lines (first..=renderEnd), clearing each line with \x1b[2K to prevent ghosting (open issue 3).
        let render_end = last.min(new_len - 1);
        // PF-01: rows written by this flush (inclusive span, in content-row coordinates).
        self.last_repaint_span = Some((first, render_end));
        for i in first..=render_end {
            if i > first {
                out.write_all(b"\r\n")?;
            }
            out.write_all(b"\x1b[2K")?;
            write_segments(out, &self.lines[i], width).map_err(TerminalError::Io)?;
        }
        let mut final_cursor_row = render_end;

        // Old content has more lines than new content: clear extra lines and move back (aligned with pi).
        if prev_len > new_len {
            if render_end < new_len - 1 {
                let move_down = new_len - 1 - render_end;
                out.write_all(format!("\x1b[{move_down}B").as_bytes())?;
                final_cursor_row = new_len - 1;
            }
            let extra = prev_len - new_len;
            for _ in new_len..prev_len {
                out.write_all(b"\r\n\x1b[2K")?;
            }
            if extra > 0 {
                out.write_all(format!("\x1b[{extra}A").as_bytes())?;
            }
        }

        out.write_all(b"\x1b[?2026l")?; // End synchronized output
        out.flush().map_err(TerminalError::Io)?;

        // ── Finalize state update ──
        self.hardware_cursor_row = final_cursor_row;
        self.prev_viewport_top = prev_vpt.max(final_cursor_row.saturating_sub(height - 1));
        self.viewport_top = self.prev_viewport_top;
        self.promote_frame();
        self.prev_height = height as u16;
        // PF-05: content rows that left the screen this frame = net viewport advance (equals `k` for an append of `k` rows).
        self.last_scroll_rows = self.prev_viewport_top.saturating_sub(entry_viewport_top);
        // IME cursor anchor: after content and state finalization, position the (hidden) hardware cursor at the editor caret (line diff
        // finalization is on the last line; the anchor row may move up/down; overrides final_cursor_row for the next frame's diff).
        self.position_cursor_hint(out)?;
        Ok(())
    }

    /// Position the (hidden) hardware cursor at this frame's cursor anchor (editor caret): relative row movement (`\x1b[nA/B`)
    /// + absolute column (`\x1b[nG`). Positioning only; no Show is sent (the cursor stays hidden, and the system IME still anchors at its position
    /// for the composition string/candidate window). After positioning, sync `hardware_cursor_row` for the next frame's line diff.
    /// Align with pi `positionHardwareCursor` (reference/pi-original/packages/tui/src/tui.ts).
    fn position_cursor_hint<W: Write>(&mut self, out: &mut W) -> Result<(), TerminalError> {
        let Some((row, col)) = self.cursor_hint else {
            return Ok(());
        };
        let delta = row as isize - self.hardware_cursor_row as isize;
        if delta > 0 {
            out.write_all(format!("\x1b[{delta}B").as_bytes())?;
        } else if delta < 0 {
            out.write_all(format!("\x1b[{}A", -delta).as_bytes())?;
        }
        out.write_all(format!("\x1b[{}G", col + 1).as_bytes())?;
        out.flush().map_err(TerminalError::Io)?;
        self.hardware_cursor_row = row;
        Ok(())
    }

    /// Full redraw: clear screen (keeping scrollback) + home, then emit each line with `\r\n` + `\x1b[2K` + content.
    /// When content exceeds the screen the terminal scrolls naturally; the last `height` lines stay visible, earlier ones go into scrollback.
    /// After writing, reset viewport and hardware cursor state. Aligned with pi `fullRender` (without `\x1b[3J` clearing scrollback).
    fn full_render<W: Write>(
        &mut self,
        out: &mut W,
        width: usize,
        height: usize,
    ) -> Result<(), TerminalError> {
        out.write_all(b"\x1b[?2026h")?;
        out.write_all(b"\x1b[2J\x1b[H")?; // clear screen + home (does not clear scrollback)
        for (i, line) in self.lines.iter().enumerate() {
            if i > 0 {
                out.write_all(b"\r\n")?;
            }
            out.write_all(b"\x1b[2K")?;
            write_segments(out, line, width).map_err(TerminalError::Io)?;
        }
        out.write_all(b"\x1b[?2026l")?;
        out.flush().map_err(TerminalError::Io)?;

        let new_len = self.lines.len();
        self.hardware_cursor_row = new_len.saturating_sub(1);
        self.prev_viewport_top = new_len.saturating_sub(height);
        self.viewport_top = self.prev_viewport_top;
        self.promote_frame();
        self.prev_height = height as u16;
        // PF-01: a full redraw writes **every** content row (cleared screen + one `\x1b[2K` + line per row; the
        // overflow scrolls into scrollback), so the span covers the whole line stream rather than just the
        // visible window. Empty stream ⇒ nothing written ⇒ None.
        self.last_repaint_span = new_len.checked_sub(1).map(|last| (0, last));
        // PF-05: `last_scroll_rows` stays 0 here **by fiat** — the advancement on this path is the terminal's natural overflow while the whole
        // stream is replayed, not canvas advancement (do not read it as a per-frame advancement budget).
        // IME cursor anchor: the full redraw path also positions after finalization (terminal-ime-anchor-fix).
        self.position_cursor_hint(out)?;
        Ok(())
    }

    /// Promote this frame's line stream to next frame's diff baseline: the Rust equivalent of pi's `this.previousLines = newLines`.
    ///
    /// `mem::swap` + `clear` hands the whole buffer over (capacity swapped, not dropped) — zero allocations, zero copies. Do **not** use
    /// `mem::take`: it replaces `lines` with `Vec::new()`, losing the capacity, so the next frame re-grows it and the "a cache-hit frame allocates
    /// nothing" property is lost. Do not `extend_from_slice` either (N refcount ops, and promote stops being O(1)).
    /// Residual: dropping the old baseline is N refcount decrements (no allocation; measured by `bench_promote_cost`).
    /// Safe on the no-change early-return path: that path leaves `hardware_cursor_row` / `prev_viewport_top` / `prev_height` untouched, i.e. they
    /// already agree with the stream, so promoting an identical stream is a semantic no-op.
    /// ⚠️ Ordering constraint with the frame metrics: `TuiLoop::run_scrollback` reads `content_height()` **after** flush, so `app_loop` captures
    /// `line_total` before calling flush. Never "fix" that by making `lines` survive the promote.
    fn promote_frame(&mut self) {
        std::mem::swap(&mut self.prev_lines, &mut self.lines);
        // `lines` now holds the previous baseline; keep its capacity for the next frame's push loop.
        self.lines.clear();
    }
}

/// Are two line handles the same line? Pointer equality first, content equality as the fallback.
///
/// The scrollback canvas rebuilds its stream from the row caches every frame, so an unchanged row arrives as the very same `Rc` (cache hit)
/// ⇒ one pointer compare. A re-rendered row gets a fresh `Rc`, which only disqualifies the fast path: the content comparison then decides, so a
/// row that re-rendered to identical text is still reported unchanged (same result as pi's line-string comparison — this saves *work*, not
/// *writes*).
fn same_line(a: &LineSegs, b: &LineSegs) -> bool {
    Rc::ptr_eq(a, b) || a == b
}

/// First and last content row that differ between the previous frame and this one, or `None` when the streams are equal.
///
/// Iterates `0..max(prev.len(), new.len())`; a missing row on either side counts as a change, so growth and shrink are both detected here.
/// Cost: iteration count is O(rows); what PF-05 changes is the per-iteration cost — `ptr_eq` for unchanged rows, content comparison only when a
/// handle changed.
/// ⚠️ Implementation trap: never express this as `prev.get(i) != new.get(i)` — `Option<&LineSegs>::ne` compiles and silently delegates to `Rc`'s
/// *content* `PartialEq`, bypassing [`same_line`] and restoring the old full-order comparison ("not faster, and nobody can tell why").
/// Boundary: content differences only. Rows that must be rewritten merely because a growth shifted them stay the caller's concern (`flush` extends
/// to the end when the tail moved).
fn changed_range(prev: &[LineSegs], new: &[LineSegs]) -> Option<(usize, usize)> {
    let max_len = prev.len().max(new.len());
    let mut first: Option<usize> = None;
    let mut last = 0usize;
    for i in 0..max_len {
        let unchanged = match (prev.get(i), new.get(i)) {
            (Some(old), Some(new)) => same_line(old, new),
            _ => false,
        };
        if !unchanged {
            if first.is_none() {
                first = Some(i);
            }
            last = i;
        }
    }
    first.map(|first| (first, last))
}

/// Does this frame have to take the full-redraw path instead of the row diff?
///
/// The cheap row diff assumes the screen still shows the previous frame's rows at the same positions. That stops being true when there is no
/// baseline yet (first frame), when the content shrank below the changed rows (the change is "all deletions", nothing left to repaint), or when
/// the first changed row is **above** the previous viewport top — those rows are already in the scrollback and can only be repaired by replaying
/// the whole stream (pi `tui.ts:1457`, kept verbatim per the performance plan §PF-05 item 3).
/// Takes the already-computed diff plus both slices (slice lengths are read internally): four same-typed `usize` parameters would let a swapped
/// argument compile silently, and `changed` is passed in so the caller does not pay for a second scan.
/// Boundary: a resize never reaches here — `set_size` clears the baseline, so `prev.is_empty()` routes to the full redraw anyway.
fn needs_full_render(
    changed: Option<(usize, usize)>,
    prev: &[LineSegs],
    new: &[LineSegs],
    prev_viewport_top: usize,
) -> bool {
    let Some((first, _)) = changed else {
        return false;
    };
    prev.is_empty() || first < prev_viewport_top || first >= new.len()
}

/// Truncate a line to the canvas column width: count display columns across segments; if remaining space is less than a wide char's full width, drop it (never split in half).
fn truncate_line(line: Vec<StyledSegment>, width: u16) -> Vec<StyledSegment> {
    let width = width as usize;
    let mut out = Vec::new();
    let mut used = 0usize;
    for mut seg in line {
        let seg_w = UnicodeWidthStr::width(seg.text.as_str());
        if used + seg_w <= width {
            out.push(seg);
            used += seg_w;
            continue;
        }
        if used >= width {
            break;
        }
        let mut text = String::new();
        for ch in seg.text.chars() {
            let cw = UnicodeWidthChar::width(ch).unwrap_or(0);
            if used + cw > width {
                break;
            }
            text.push(ch);
            used += cw;
        }
        seg.text = text;
        out.push(seg);
        break;
    }
    out
}

/// Write a style segment on one line as crossterm commands (SetColors + Reset + SetAttribute + Print(text)),
/// Segment content exceeding `width` is not written (truncated per character by column width, without splitting wide chars; lines already truncated by push
/// normally won't trigger this — it is a defensive check for non-truncating entry points such as push_text).
fn write_segments<W: Write>(
    out: &mut W,
    line: &[StyledSegment],
    width: usize,
) -> std::io::Result<()> {
    let mut used = 0usize;
    for seg in line {
        if used >= width {
            break;
        }
        // Order-sensitive: first Reset to clear residual attributes/colors, then SetColors to set colors, and finally Set for each attribute—
        // A Reset (ESC[0m) after SetColors also clears the newly set colors (colors are lost, attributes survive).
        queue!(
            out,
            crossterm::style::SetAttribute(crossterm::style::Attribute::Reset),
            crossterm::style::SetColors(crossterm::style::Colors {
                foreground: seg.style.fg,
                background: seg.style.bg,
            }),
        )?;
        for attr in &seg.style.attributes {
            queue!(out, crossterm::style::SetAttribute(*attr))?;
        }
        let mut written = String::new();
        for ch in seg.text.chars() {
            let cw = UnicodeWidthChar::width(ch).unwrap_or(0);
            if used + cw > width {
                break;
            }
            written.push(ch);
            used += cw;
        }
        if !written.is_empty() {
            out.write_all(written.as_bytes())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// CJK sample kept intentionally: this test verifies wide-character / multibyte behavior.
    const CJK_SAMPLE: &str = "ab中";
    use super::*;
    use crossterm::style::Color;

    fn seg(text: &str, style: Style) -> StyledSegment {
        StyledSegment {
            style,
            text: text.to_string(),
        }
    }
    fn plain(text: &str) -> StyledSegment {
        seg(text, Style::default())
    }
    fn red(text: &str) -> StyledSegment {
        seg(text, Style::fg(Color::Red))
    }

    // ── new / push / content_height / clear ──

    #[test]
    fn test_new_is_empty() {
        let c = ScrollbackCanvas::new(40, 10);
        assert_eq!(c.content_height(), 0);
        assert_eq!(c.width(), 40);
        assert!(c.lines.is_empty());
        assert!(c.prev_lines.is_empty());
        assert_eq!(c.prev_viewport_top, 0);
    }

    #[test]
    fn test_push_and_clear() {
        let mut c = ScrollbackCanvas::new(40, 10);
        c.push_line(vec![plain("hi")]);
        c.push_text("x", Style::default());
        assert_eq!(c.content_height(), 2);
        c.clear();
        assert_eq!(c.content_height(), 0);
        // Can push again after clear and rebuild
        c.push_line(vec![plain("again")]);
        assert_eq!(c.content_height(), 1);
    }

    #[test]
    fn test_line_reads_only_the_current_frame_stream() {
        // long-row-clip-fix acceptance vehicle: the line stream is readable after draw and BEFORE flush;
        // `flush` promotes the stream to `prev_lines` and clears `lines` (PF-05 invariant), so after flush
        // the accessor must return None on every index — it deliberately does not fall back to `prev_lines`.
        let mut c = ScrollbackCanvas::new(20, 5);
        c.push_line(vec![plain("first")]);
        c.push_line(vec![plain("second")]);
        assert!(c.line(2).is_none(), "out of range ⇒ None");
        assert_eq!(c.line(0).unwrap()[0].text, "first");
        assert_eq!(c.line(1).unwrap()[0].text, "second");

        c.flush(&mut Vec::new()).unwrap();
        assert_eq!(c.content_height(), 0);
        assert!(c.line(0).is_none());
        assert!(c.line(1).is_none());
    }

    #[test]
    fn test_push_line_truncates_to_width_and_keeps_wide_chars() {
        let mut c = ScrollbackCanvas::new(4, 10);
        c.push_line(vec![plain("abcdef")]);
        assert_eq!(c.lines[0].len(), 1);
        assert_eq!(c.lines[0][0].text, "abcd");

        // Wide chars: ab(2) + one CJK char(2) = 4 exactly fills
        let mut c2 = ScrollbackCanvas::new(4, 10);
        c2.push_line(vec![plain(CJK_SAMPLE)]);
        assert_eq!(c2.lines[0][0].text, CJK_SAMPLE);

        // 3 columns can't fit a 2-column CJK char: don't split it in half, drop it entirely
        let mut c3 = ScrollbackCanvas::new(3, 10);
        c3.push_line(vec![plain(CJK_SAMPLE)]);
        assert_eq!(c3.lines[0][0].text, "ab");
    }

    /// PF-05 measurement (ignored): the per-line diff scan in `flush` **after** the shared-handle change.
    ///
    /// History: PF-04 measured this probe at 0.07 ms / 2500 lines and it was documented as "exactly the `old != new` total-order comparison
    /// over a full history"; `docs/performance-optimization-plan.md` quotes that number. PF-05 replaced the comparison with a `Rc::ptr_eq`
    /// fast path, so the same code now times a **different quantity** at a different order of magnitude (the plan's quoted figure must be
    /// re-recorded, otherwise someone will later cite "still 0.07 ms after PF-05" as evidence).
    /// Run with: `cargo test -p zen-tui --release -- --ignored --nocapture bench_flush_diff`
    #[test]
    #[ignore = "manual timing check (release): cargo test -p zen-tui --release -- --ignored"]
    fn bench_flush_diff_after_row_cache() {
        let (width, rows, lines) = (80u16, 40u16, 2500usize);
        let mut canvas = ScrollbackCanvas::new(width, rows);
        // Shared handles, re-pushed identically every frame: this is the all-hit frame (the row cache serves the same allocations).
        let base: Vec<LineSegs> = (0..lines)
            .map(|i| {
                Rc::new(vec![
                    plain(&format!("line {i}: ")),
                    seg("inline_code", Style::fg(crossterm::style::Color::Yellow)),
                    plain(" the session tree is the core of the product, "),
                    seg("every message", Style::fg(crossterm::style::Color::Cyan)),
                ])
            })
            .collect();
        for line in &base {
            canvas.push_line_pre_truncated(line.clone());
        }
        let mut sink: Vec<u8> = Vec::new();
        // First flush establishes `prev_lines` (the diff base); afterwards the streams are identical = the all-hit frame.
        canvas
            .flush(&mut sink)
            .expect("flush into a Vec never fails");
        let iterations = 10u32;
        let started = std::time::Instant::now();
        for _ in 0..iterations {
            canvas.clear();
            for line in &base {
                canvas.push_line_pre_truncated(line.clone());
            }
            sink.clear();
            canvas
                .flush(&mut sink)
                .expect("flush into a Vec never fails");
        }
        let per = (started.elapsed() / iterations).as_secs_f64() * 1000.0;
        println!("bench_flush_diff: lines={lines} per_flush={per:.3}ms");
        if !cfg!(debug_assertions) {
            // Sanity ceiling only (the real budget lives with the frame metrics, not this probe).
            assert!(per <= 16.0, "diff scan for {lines} lines took {per:.3}ms");
        }
    }

    /// PF-05 measurement (ignored): the two shapes the diff baseline can take — a deep copy vs a shared-handle bump.
    ///
    /// Step 0 of the task: the canvas used to keep its diff baseline by `prev_lines = lines.clone()` (one deep copy of the whole history per
    /// frame, the same shape as the row cache → canvas copy in `ui.rs::push_row_lines`). This probe times the two alternatives on a realistic
    /// 2500-line stream, so the implementation record has a measured "cost of the two clones" instead of an estimate.
    /// Run with: `cargo test -p zen-tui --release -- --ignored --nocapture bench_promote_cost`
    #[test]
    #[ignore = "manual timing check (release): cargo test -p zen-tui --release -- --ignored"]
    fn bench_promote_cost() {
        let lines: Vec<Vec<StyledSegment>> = (0..2500)
            .map(|i| {
                vec![
                    plain(&format!("line {i}: ")),
                    seg("inline_code", Style::fg(crossterm::style::Color::Yellow)),
                    plain(" the session tree is the core of the product, "),
                    seg("every message", Style::fg(crossterm::style::Color::Cyan)),
                ]
            })
            .collect();
        let shared: Vec<LineSegs> = lines.iter().cloned().map(Rc::new).collect();
        let iterations = 20u32;
        let started = std::time::Instant::now();
        for _ in 0..iterations {
            let copy = lines.clone();
            std::hint::black_box(&copy);
        }
        let deep = (started.elapsed() / iterations).as_secs_f64() * 1000.0;
        let started = std::time::Instant::now();
        for _ in 0..iterations {
            let bump = shared.clone();
            std::hint::black_box(&bump);
        }
        let shallow = (started.elapsed() / iterations).as_secs_f64() * 1000.0;
        println!(
            "bench_promote_cost: lines={} deep_clone={deep:.3}ms handle_bump={shallow:.3}ms ratio={:.1}x",
            lines.len(),
            deep / shallow
        );
        if !cfg!(debug_assertions) {
            assert!(
                shallow <= deep,
                "a handle bump must never be slower than a deep copy"
            );
        }
    }

    /// PF-05 measurement (ignored): one appended line on top of a 2500-line history — the frame shape the task is about.
    ///
    /// The draw side rebuilds the whole stream every frame (that is the draw contract), so the rebuild happens outside the timer and only
    /// `flush` is measured: baseline promote (`mem::swap`) + diff scan (`ptr_eq`) + the one written line.
    /// Run with: `cargo test -p zen-tui --release -- --ignored --nocapture bench_append_frame_flush`
    #[test]
    #[ignore = "manual timing check (release): cargo test -p zen-tui --release -- --ignored"]
    fn bench_append_frame_flush() {
        let (width, rows, lines) = (80u16, 40u16, 2500usize);
        let mut canvas = ScrollbackCanvas::new(width, rows);
        let base: Vec<LineSegs> = (0..lines)
            .map(|i| {
                Rc::new(vec![
                    plain(&format!("line {i}: ")),
                    seg("inline_code", Style::fg(crossterm::style::Color::Yellow)),
                    plain(" the session tree is the core of the product, "),
                    seg("every message", Style::fg(crossterm::style::Color::Cyan)),
                ])
            })
            .collect();
        for line in &base {
            canvas.push_line_pre_truncated(line.clone());
        }
        let mut sink: Vec<u8> = Vec::new();
        canvas
            .flush(&mut sink)
            .expect("flush into a Vec never fails");
        let iterations = 10u32;
        let mut total = std::time::Duration::ZERO;
        for k in 0..iterations {
            // Rebuild the frame (untimed): identical history + one appended line.
            canvas.clear();
            for line in &base {
                canvas.push_line_pre_truncated(line.clone());
            }
            // Append cumulatively: each measured frame adds one row to the previous frame's stream (a real append, not "rewrite the last line").
            for extra in 0..=k {
                canvas.push_line_pre_truncated(Rc::new(vec![plain(&format!("appended {extra}"))]));
            }
            sink.clear();
            let started = std::time::Instant::now();
            canvas
                .flush(&mut sink)
                .expect("flush into a Vec never fails");
            total += started.elapsed();
        }
        let per = (total / iterations).as_secs_f64() * 1000.0;
        println!(
            "bench_append_frame_flush: lines={lines} per_flush={per:.3}ms bytes={} span={:?} scroll_rows={}",
            sink.len(),
            canvas.last_repaint_span(),
            canvas.last_scroll_rows()
        );
        if !cfg!(debug_assertions) {
            assert!(per <= 16.0, "append frame flush took {per:.3}ms");
        }
    }

    // ── flush: full repaint (first frame) ──

    #[test]
    fn test_push_line_pre_truncated_keeps_the_line_verbatim() {
        // Cache-hit path: the caller already truncated to the cached width, so this entry must NOT scan/truncate again
        // (that per-character scan is exactly what PF-04 removes from the steady-state frame).
        let mut c = ScrollbackCanvas::new(4, 10);
        c.push_line_pre_truncated(Rc::new(vec![plain("abcd"), red("e")]));
        assert_eq!(c.content_height(), 1);
        assert_eq!(c.lines[0].len(), 2, "segments preserved as-is");
        assert_eq!(c.lines[0][0].text, "abcd");
        assert_eq!(
            c.lines[0][1].text, "e",
            "not truncated here (caller's duty)"
        );
    }

    #[test]
    fn test_first_flush_writes_all_lines_and_sets_viewport() {
        let mut c = ScrollbackCanvas::new(10, 3);
        for i in 0..5 {
            c.push_line(vec![plain(&format!("line{i}"))]);
        }
        let mut out = Vec::new();
        c.flush(&mut out).unwrap();
        assert!(
            out.windows(5).any(|w| w == b"line0"),
            "First frame should write all lines"
        );
        assert!(out.windows(5).any(|w| w == b"line4"));
        assert_eq!(
            c.prev_viewport_top, 2,
            "height=3, 5 lines → viewport top=5-3=2"
        );
        assert_eq!(c.prev_lines.len(), 5);
        assert_eq!(c.hardware_cursor_row, 4);
    }

    #[test]
    fn test_first_flush_fits_within_height_viewport_zero() {
        let mut c = ScrollbackCanvas::new(10, 5);
        c.push_line(vec![plain("a")]);
        c.push_line(vec![plain("b")]);
        c.flush(&mut Vec::new()).unwrap();
        assert_eq!(
            c.prev_viewport_top, 0,
            "Content does not exceed screen → viewport top=0"
        );
    }

    // ── flush: diff (no change / append / scroll beyond screen) ──

    #[test]
    fn test_no_change_flush_produces_no_output() {
        let mut c = ScrollbackCanvas::new(10, 3);
        c.push_line(vec![plain("a")]);
        c.flush(&mut Vec::new()).unwrap();
        // PF-05: a flush promotes the stream and leaves `lines` empty for the next draw, so "no change" means the draw side pushes the
        // identical stream again (not "flush twice without drawing").
        c.clear();
        c.push_line(vec![plain("a")]);
        let mut out = Vec::new();
        c.flush(&mut out).unwrap();
        assert!(out.is_empty(), "Diff should output no bytes when unchanged");
        assert_eq!(c.last_repaint_span(), None);
        assert_eq!(c.last_scroll_rows(), 0);
        assert_eq!(
            c.content_height(),
            0,
            "flush clears the stream for the next draw"
        );
        assert_eq!(
            c.prev_lines.len(),
            1,
            "the baseline is what survives a flush"
        );
    }

    #[test]
    fn test_growth_within_height_appends_new_line() {
        let mut c = ScrollbackCanvas::new(10, 5);
        c.push_line(vec![plain("a")]);
        c.flush(&mut Vec::new()).unwrap();
        c.clear();
        c.push_line(vec![plain("a")]);
        c.push_line(vec![plain("b")]);
        let mut out = Vec::new();
        c.flush(&mut out).unwrap();
        assert!(
            out.windows(1).any(|w| w == b"b"),
            "Diff should output newly added lines"
        );
        assert_eq!(
            c.prev_viewport_top, 0,
            "Viewport top unchanged when content fits screen"
        );
    }

    #[test]
    fn test_growth_beyond_height_scrolls_viewport() {
        let mut c = ScrollbackCanvas::new(10, 2);
        c.push_line(vec![plain("r0")]);
        c.push_line(vec![plain("r1")]);
        c.flush(&mut Vec::new()).unwrap();
        assert_eq!(c.prev_viewport_top, 0);

        // Append third line → content 3 > height 2 → push scrollback, viewport top = 1
        c.clear();
        for i in 0..3 {
            c.push_line(vec![plain(&format!("r{i}"))]);
        }
        let mut out = Vec::new();
        c.flush(&mut out).unwrap();
        assert!(
            out.windows(2).any(|w| w == b"r2"),
            "Latest line should be visible"
        );
        assert_eq!(c.prev_viewport_top, 1);
        assert!(
            out.windows(2).any(|w| w == b"\r\n"),
            "应含真实滚动推进 \\r\\n"
        );
    }

    #[test]
    fn test_flush_overflow_keeps_increasing_viewport() {
        let mut c = ScrollbackCanvas::new(10, 3);
        // Grows step by step 3 → 4 → 5, viewport top advances accordingly
        let mut lines: Vec<String> = (0..3).map(|i| format!("l{i}")).collect();
        for line in lines.iter() {
            c.push_line(vec![plain(line)]);
        }
        c.flush(&mut Vec::new()).unwrap();
        assert_eq!(c.prev_viewport_top, 0);
        for n in 4..=5 {
            lines.push(format!("l{}", n - 1));
            c.clear();
            for l in &lines {
                c.push_line(vec![plain(l)]);
            }
            c.flush(&mut Vec::new()).unwrap();
            assert_eq!(
                c.prev_viewport_top,
                n - 3,
                "At line {n}, viewport top should be {}-3",
                n - 3
            );
        }
    }

    // ── flush: size change → full repaint ──

    #[test]
    fn test_set_size_change_forces_full_rerender() {
        let mut c = ScrollbackCanvas::new(10, 3);
        c.push_line(vec![plain("a")]);
        c.flush(&mut Vec::new()).unwrap();
        // Width change: reset previous frame diff → next flush does full repaint (prev_lines cleared)
        assert!(c.set_size(15, 3));
        c.clear();
        c.push_line(vec![plain("a")]);
        let mut out = Vec::new();
        c.flush(&mut out).unwrap();
        assert!(out.windows(1).any(|w| w == b"a"));
    }

    #[test]
    fn test_set_size_same_is_noop() {
        let mut c = ScrollbackCanvas::new(10, 3);
        assert!(!c.set_size(10, 3));
        // If unchanged, prev_lines is kept; the next flush still takes the diff path (no output)
        c.push_line(vec![plain("a")]);
        c.flush(&mut Vec::new()).unwrap();
        c.set_size(10, 3);
        c.clear();
        c.push_line(vec![plain("a")]);
        let mut out = Vec::new();
        c.flush(&mut out).unwrap();
        assert!(out.is_empty());
    }

    // ── Style segment output includes color commands ──

    #[test]
    fn test_flush_writes_style_commands_and_text() {
        let mut c = ScrollbackCanvas::new(10, 3);
        c.push_line(vec![red("R"), plain("x")]);
        let mut out = Vec::new();
        c.flush(&mut out).unwrap();
        // Contains at least the Red foreground color (crossterm 256-color encoding consistency check) and text
        let text = String::from_utf8_lossy(&out);
        assert!(
            text.contains("\x1b[38;5;9m") || text.contains("31"),
            "Should contain foreground color command; actual: {text:?}"
        );
        assert!(out.windows(1).any(|w| w == b"R"));
        assert!(out.windows(1).any(|w| w == b"x"));
    }

    // ── IME cursor anchor (terminal-ime-anchor-fix) ──

    #[test]
    fn test_cursor_hint_set_and_clear_and_line_count() {
        let mut c = ScrollbackCanvas::new(10, 5);
        assert_eq!(c.cursor_hint(), None);
        c.push_line(vec![plain("a")]);
        assert_eq!(
            c.line_count(),
            1,
            "line_count reflects the number of pushed lines"
        );
        c.set_cursor_hint(Some((1, 2)));
        assert_eq!(c.cursor_hint(), Some((1, 2)));
        // clear() resets the anchor (re-set by draw each frame)
        c.clear();
        assert_eq!(c.cursor_hint(), None);
        assert_eq!(c.line_count(), 0, "clear clears the line stream");
    }

    #[test]
    fn test_cursor_hint_reset_by_set_size() {
        let mut c = ScrollbackCanvas::new(10, 5);
        c.set_cursor_hint(Some((1, 2)));
        assert!(c.set_size(12, 5));
        assert_eq!(c.cursor_hint(), None, "Resize clears anchors");
    }

    #[test]
    fn test_flush_no_hint_writes_no_cursor_moves() {
        // With no anchor, flush emits no cursor positioning sequence (CSI parameters: 1-3 digits + A/B/G terminator)
        let mut c = ScrollbackCanvas::new(10, 4);
        c.push_line(vec![plain("a")]);
        let mut out = Vec::new();
        c.flush(&mut out).unwrap();
        let has_position = |t: &str| {
            let b = t.as_bytes();
            let mut i = 0;
            while i < b.len() {
                if b[i] == 0x1b && b.get(i + 1) == Some(&b'[') {
                    let mut j = i + 2;
                    while j < b.len() && b[j].is_ascii_digit() {
                        j += 1;
                    }
                    if matches!(b.get(j), Some(b'A' | b'B' | b'G')) {
                        return true;
                    }
                }
                i += 1;
            }
            false
        };
        let text = String::from_utf8_lossy(&out);
        assert!(
            !has_position(&text),
            "Without an anchor, no positioning sequence should be output: {text:?}"
        );
    }

    #[test]
    fn test_flush_with_hint_moves_row_and_column() {
        // Content has 3 lines, height=3: after full redraw, hardware_cursor_row=2; anchor (1, 2) → \x1b[1A + \x1b[3G
        let mut c = ScrollbackCanvas::new(10, 3);
        c.push_line(vec![plain("r0")]);
        c.push_line(vec![plain("r1")]);
        c.push_line(vec![plain("r2")]);
        c.set_cursor_hint(Some((1, 2)));
        let mut out = Vec::new();
        c.flush(&mut out).unwrap();
        let text = String::from_utf8_lossy(&out);
        assert!(text.contains("\x1b[1A"), "Should move up 1 line: {text:?}");
        assert!(
            text.contains("\x1b[3G"),
            "Should move to absolute column 3 (0-based 2+1): {text:?}"
        );
        assert_eq!(
            c.hardware_cursor_row, 1,
            "Anchor line synchronized as diff baseline"
        );
    }

    #[test]
    fn test_flush_with_hint_same_row_only_column_move() {
        // Anchor row equals final_cursor_row: only column positioning is emitted (no A/B)
        let mut c = ScrollbackCanvas::new(10, 3);
        c.push_line(vec![plain("a")]);
        c.push_line(vec![plain("b")]);
        c.push_line(vec![plain("c")]);
        let mut first = Vec::new();
        c.flush(&mut first).unwrap(); // Previous frame: full redraw, cursor row 2
        c.clear();
        for i in 0..3 {
            c.push_line(vec![plain(&format!("r{i}"))]);
        }
        // Changed rows = 0..2 (final_cursor_row = 2), anchor row 2: only column positioning is needed
        c.set_cursor_hint(Some((2, 5)));
        let mut out = Vec::new();
        c.flush(&mut out).unwrap();
        let text = String::from_utf8_lossy(&out);
        assert!(
            text.contains("\x1b[6G"),
            "Should contain column positioning: {text:?}"
        );
        assert!(
            !text.contains("\x1b[1A") && !text.contains("\x1b[1B"),
            "Same anchor line should not cause line movement: {text:?}"
        );
        assert_eq!(c.hardware_cursor_row, 2);
    }

    #[test]
    fn test_full_render_path_also_positions_cursor_hint() {
        // First frame (full redraw) with anchor: finalization positioning takes effect
        let mut c = ScrollbackCanvas::new(10, 3);
        c.push_line(vec![plain("r0")]);
        c.push_line(vec![plain("r1")]);
        c.set_cursor_hint(Some((0, 0)));
        let mut out = Vec::new();
        c.flush(&mut out).unwrap();
        let text = String::from_utf8_lossy(&out);
        // After full redraw, hardware_cursor_row=1, anchor row 0 → move up 1 row + column 1
        assert!(text.contains("\x1b[1A"));
        assert!(text.contains("\x1b[1G"));
        assert_eq!(c.hardware_cursor_row, 0);
    }

    #[test]
    fn test_flush_reset_precedes_colors() {
        // Ordering contract: Reset (ESC[0m) must precede SetColors; otherwise, the reset clears the newly set colors
        // (Attributes survive, colors are lost—guards against reproducing the historical defect T-TU-02-1-fix).
        let mut c = ScrollbackCanvas::new(10, 3);
        c.push_line(vec![red("R")]);
        let mut out = Vec::new();
        c.flush(&mut out).unwrap();
        let text = String::from_utf8_lossy(&out);
        let reset = text.find("\x1b[0m").expect("Should contain Reset sequence");
        let color = text
            .find("\x1b[38;5;9m")
            .or_else(|| text.find("\x1b[91m"))
            .or_else(|| text.find("\x1b[31m"))
            .expect("Should contain foreground color sequence");
        assert!(reset < color, "Reset must precede color command: {text:?}");
    }

    // ── PF-01: repaint span + fixed-row registration ──

    #[test]
    fn repaint_span_of_first_flush_covers_every_line() {
        // First flush takes the full-redraw path; a full redraw writes every content row
        // (screen clear + one \x1b[2K + line per row), so the span is the whole stream.
        let mut c = ScrollbackCanvas::new(20, 3);
        for i in 0..5 {
            c.push_line(vec![plain(&format!("line{i}"))]);
        }
        let mut out = Vec::new();
        c.flush(&mut out).unwrap();
        assert_eq!(c.last_repaint_span(), Some((0, 4)));
    }

    #[test]
    fn repaint_span_is_none_when_nothing_changed() {
        // Steady-state frame: flush early-returns, and the span must not leak the previous frame's window.
        let mut c = ScrollbackCanvas::new(20, 5);
        c.push_line(vec![plain("a")]);
        c.push_line(vec![plain("b")]);
        let mut out = Vec::new();
        c.flush(&mut out).unwrap();
        assert_eq!(c.last_repaint_span(), Some((0, 1)));
        // PF-05: the draw side rebuilds the (identical) stream before the next flush.
        c.clear();
        c.push_line(vec![plain("a")]);
        c.push_line(vec![plain("b")]);
        out.clear();
        c.flush(&mut out).unwrap();
        assert!(out.is_empty(), "steady-state frame must write nothing");
        assert_eq!(
            c.last_repaint_span(),
            None,
            "no-change frame must not report a stale span"
        );
    }

    #[test]
    fn repaint_span_of_append_starts_at_the_new_line() {
        let mut c = ScrollbackCanvas::new(20, 5);
        c.push_line(vec![plain("a")]);
        let mut out = Vec::new();
        c.flush(&mut out).unwrap();
        c.clear();
        c.push_line(vec![plain("a")]);
        c.push_line(vec![plain("b")]);
        out.clear();
        c.flush(&mut out).unwrap();
        assert_eq!(c.last_repaint_span(), Some((1, 1)));
    }

    #[test]
    fn repaint_span_of_empty_stream_is_none() {
        let mut c = ScrollbackCanvas::new(20, 3);
        let mut out = Vec::new();
        c.flush(&mut out).unwrap();
        assert_eq!(c.last_repaint_span(), None);
    }

    #[test]
    fn clear_and_set_size_reset_the_repaint_span() {
        let mut c = ScrollbackCanvas::new(20, 5);
        c.push_line(vec![plain("a")]);
        let mut out = Vec::new();
        c.flush(&mut out).unwrap();
        assert!(c.last_repaint_span().is_some());
        c.clear();
        assert_eq!(c.last_repaint_span(), None);
        // Re-arm it with a real change (pushing the identical line again would make the diff empty),
        // then a geometry change must drop it as well.
        c.push_line(vec![plain("b")]);
        c.flush(&mut out).unwrap();
        assert!(c.last_repaint_span().is_some());
        c.set_size(30, 6);
        assert_eq!(c.last_repaint_span(), None);
    }

    #[test]
    fn fixed_rows_round_trip_and_reset_on_size_change() {
        let mut c = ScrollbackCanvas::new(40, 10);
        assert_eq!(c.fixed_rows(), 0);
        c.set_fixed_rows(7);
        assert_eq!(c.fixed_rows(), 7);
        // Same geometry → keeps the registration; a real resize drops it (new geometry, new layout).
        assert!(!c.set_size(40, 10));
        assert_eq!(c.fixed_rows(), 7);
        assert!(c.set_size(41, 10));
        assert_eq!(c.fixed_rows(), 0);
    }

    // ── PF-05: shared handles, copy-free promote, converged write span ──

    fn shared(text: &str) -> LineSegs {
        Rc::new(vec![plain(text)])
    }

    #[test]
    fn promote_frame_aliases_the_stream_and_empties_it_for_the_next_draw() {
        // The Rust equivalent of pi's `this.previousLines = newLines`: the baseline shares the very same allocations (no deep copy), and
        // `lines` is left empty (with its capacity) for the next draw.
        let line = shared("a");
        let mut c = ScrollbackCanvas::new(20, 5);
        c.push_line_pre_truncated(line.clone());
        c.flush(&mut Vec::new()).unwrap();
        assert_eq!(c.prev_lines.len(), 1);
        assert!(
            Rc::ptr_eq(&c.prev_lines[0], &line),
            "promote must alias the pushed handle, not copy it"
        );
        assert_eq!(c.content_height(), 0, "lines is cleared for the next draw");
    }

    #[test]
    fn same_line_uses_pointer_equality_then_content_equality() {
        let a = shared("a");
        let same_allocation = a.clone();
        let same_content = shared("a");
        assert!(same_line(&a, &same_allocation));
        assert!(
            same_line(&a, &same_content),
            "a re-rendered row with identical content is still the same line"
        );
        assert!(!same_line(&a, &shared("b")));
    }

    #[test]
    fn changed_range_reports_first_and_last_content_difference() {
        let (a, b, c) = (shared("a"), shared("b"), shared("c"));
        let one = std::slice::from_ref(&a);
        assert_eq!(changed_range(&[], &[]), None);
        assert_eq!(changed_range(one, one), None);
        // Re-rendered but identical content: not a change.
        assert_eq!(changed_range(one, &[shared("a")]), None);
        // Append / mid change / shrink (the vanished row counts as changed).
        assert_eq!(changed_range(one, &[a.clone(), b.clone()]), Some((1, 1)));
        assert_eq!(
            changed_range(&[a.clone(), b.clone()], &[a.clone(), c.clone()]),
            Some((1, 1))
        );
        assert_eq!(changed_range(&[a.clone(), b.clone()], &[a]), Some((1, 1)));
    }

    #[test]
    fn needs_full_render_covers_first_frame_shrink_and_change_above_viewport() {
        let (a, b) = (shared("a"), shared("b"));
        let empty: [LineSegs; 0] = [];
        let one = [a.clone()];
        let two = [a.clone(), b.clone()];
        // First frame (no baseline).
        assert!(needs_full_render(Some((0, 0)), &empty, &two, 0));
        // Plain append inside the viewport.
        assert!(!needs_full_render(Some((1, 1)), &one, &two, 0));
        // Change above the previous viewport top → replay the whole stream (pi `tui.ts:1457`).
        assert!(needs_full_render(
            Some((1, 1)),
            &two,
            &[a.clone(), shared("c")],
            2
        ));
        // All deletions (`first >= new.len()`).
        assert!(needs_full_render(Some((1, 1)), &two, &one, 0));
        // No diff at all never triggers a full render.
        assert!(!needs_full_render(None, &two, &two, 99));
    }

    #[test]
    fn append_frame_does_not_reprint_history_and_advances_by_the_new_rows() {
        let mut c = ScrollbackCanvas::new(20, 3);
        let base: Vec<LineSegs> = (0..5).map(|i| shared(&format!("hist{i}"))).collect();
        for line in &base {
            c.push_line_pre_truncated(line.clone());
        }
        c.flush(&mut Vec::new()).unwrap();

        c.clear();
        for line in &base {
            c.push_line_pre_truncated(line.clone());
        }
        c.push_line_pre_truncated(shared("brand new"));
        let mut out = Vec::new();
        c.flush(&mut out).unwrap();

        let text = String::from_utf8_lossy(&out);
        assert!(
            !text.contains("hist0"),
            "an append frame must not reprint history: {text:?}"
        );
        assert!(text.contains("brand new"));
        assert_eq!(c.last_repaint_span(), Some((5, 5)));
        assert_eq!(
            c.last_scroll_rows(),
            1,
            "5 rows history + 1 appended, height 3"
        );
    }

    #[test]
    fn append_bytes_are_independent_of_history_length() {
        fn append_bytes(history: usize) -> usize {
            let mut c = ScrollbackCanvas::new(40, 3);
            let base: Vec<LineSegs> = (0..history).map(|i| shared(&format!("h{i}"))).collect();
            for line in &base {
                c.push_line_pre_truncated(line.clone());
            }
            c.flush(&mut Vec::new()).unwrap();
            c.clear();
            for line in &base {
                c.push_line_pre_truncated(line.clone());
            }
            c.push_line_pre_truncated(shared("appended row"));
            let mut out = Vec::new();
            c.flush(&mut out).unwrap();
            assert_eq!(c.last_repaint_span(), Some((history, history)));
            out.len()
        }
        let small = append_bytes(200);
        let large = append_bytes(2500);
        let ratio = large as f64 / small as f64;
        assert!(
            (0.9..=1.1).contains(&ratio),
            "append bytes must not scale with history: 200 rows = {small}B, 2500 rows = {large}B"
        );
    }

    #[test]
    fn anchored_tail_append_writes_only_new_rows_plus_the_tail() {
        // The assembled shape (interactive mode): `F` fixed-region rows trail the message area, so the growth does **not** start at
        // `prev_len` — `append_start` is false, yet the write span is exactly `k + F` rows and the advancement is `k`.
        let (height, tail_rows, new_rows) = (10u16, 3usize, 2usize);
        // Enough message rows that the viewport is already pinned to the bottom (`vpt = len - height`) in both frames — otherwise the
        // first `k` rows are consumed by filling the screen and the advancement is smaller than `k`.
        let messages: Vec<LineSegs> = (0..20).map(|i| shared(&format!("msg{i}"))).collect();
        let tail: Vec<LineSegs> = (0..tail_rows)
            .map(|i| shared(&format!("tail{i}")))
            .collect();
        let mut c = ScrollbackCanvas::new(20, height);
        for line in messages.iter().chain(tail.iter()) {
            c.push_line_pre_truncated(line.clone());
        }
        c.flush(&mut Vec::new()).unwrap();

        c.clear();
        for line in &messages {
            c.push_line_pre_truncated(line.clone());
        }
        for i in 0..new_rows {
            c.push_line_pre_truncated(shared(&format!("new{i}")));
        }
        for line in &tail {
            c.push_line_pre_truncated(line.clone());
        }
        let line_total = c.content_height();
        let mut out = Vec::new();
        c.flush(&mut out).unwrap();
        let span = c.last_repaint_span().expect("growth must write something");
        assert_eq!(
            span,
            (messages.len(), line_total - 1),
            "write span = insertion point..end = k + F rows (the minimum before PF-06)"
        );
        assert_eq!(
            span.1 - span.0 + 1,
            new_rows + tail_rows,
            "exactly the new rows plus the shifted tail"
        );
        assert_eq!(c.last_scroll_rows(), new_rows);
    }

    #[test]
    fn non_overflow_append_does_not_advance_the_viewport() {
        // Content still fits the screen: nothing leaves for the scrollback, so the advancement is 0 (do not assert `k` here).
        let mut c = ScrollbackCanvas::new(20, 10);
        c.push_line_pre_truncated(shared("a"));
        c.flush(&mut Vec::new()).unwrap();
        c.clear();
        c.push_line_pre_truncated(shared("a"));
        c.push_line_pre_truncated(shared("b"));
        c.flush(&mut Vec::new()).unwrap();
        assert_eq!(c.last_scroll_rows(), 0);
    }

    #[test]
    fn full_redraw_reports_zero_advancement_by_fiat() {
        let mut c = ScrollbackCanvas::new(20, 2);
        for i in 0..5 {
            c.push_line_pre_truncated(shared(&format!("l{i}")));
        }
        c.flush(&mut Vec::new()).unwrap();
        assert_eq!(c.last_repaint_span(), Some((0, 4)));
        assert_eq!(
            c.last_scroll_rows(),
            0,
            "full redraw reports 0 by fiat: its advancement is terminal overflow, not canvas advancement"
        );
    }

    #[test]
    fn clear_and_set_size_reset_the_advancement_report() {
        let mut c = ScrollbackCanvas::new(20, 2);
        let base: Vec<LineSegs> = (0..4).map(|i| shared(&format!("l{i}"))).collect();
        for line in &base {
            c.push_line_pre_truncated(line.clone());
        }
        // First flush is a full redraw ⇒ 0 by fiat.
        c.flush(&mut Vec::new()).unwrap();
        assert_eq!(c.last_scroll_rows(), 0);
        // An append frame on the diff path does advance.
        c.clear();
        for line in &base {
            c.push_line_pre_truncated(line.clone());
        }
        c.push_line_pre_truncated(shared("x"));
        c.flush(&mut Vec::new()).unwrap();
        assert!(c.last_scroll_rows() > 0);
        c.clear();
        assert_eq!(c.last_scroll_rows(), 0, "clear resets the report");
        c.push_line_pre_truncated(shared("x"));
        c.flush(&mut Vec::new()).unwrap();
        c.set_size(30, 4);
        assert_eq!(c.last_scroll_rows(), 0, "a resize resets the report");
    }
}
