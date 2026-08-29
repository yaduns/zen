// REQ-TUI-01 (pi-409): Terminal scrollback render model — content line stream + viewport advancement.
// See doRender main logic in reference/pi-original/packages/tui/src/tui.ts (viewportTop/prevViewportTop
// state + "\r\n".repeat(scroll) advancement + line-level diffing + extra-lines screen clear + synchronized output wrapping),
// converged per zen requirements into ScrollbackCanvas: the draw callback pushes this frame's content lines into `lines`, and flush diffs them against the previous frame
// Line-level diff: push the portion that overflows the screen through real terminal scrolling into scrollback — the user can scroll the terminal to review history,
// instead of only seeing the last screen as with cell-level fixed-line diffs. Remaining visible lines / bottom fixed region are repainted line by line.
// Complementary to `ScreenBuffer` (cell-level grid): this type is the rendering target for "line-level + terminal-scroll carrying".
// Pure runtime canvas, no serialization needs (implementation choices for T-TU-01-5 open questions 1/2/3 are explained in the comments below).

use crate::highlight::StyledSegment;
use crate::screen::Style;
use crate::terminal::TerminalError;
use crossterm::queue;
use std::io::Write;
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
    lines: Vec<Vec<StyledSegment>>,
    /// Previous frame's line-stream snapshot (line-level diff base; lines scrolled out of the terminal no longer participate in the diff).
    prev_lines: Vec<Vec<StyledSegment>>,
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
        }
    }

    /// Clear this frame's line stream; call before each draw (aligns with `render_frame`'s `buffer.clear()` contract);
    /// `prev_lines`/`prev_viewport_top` are kept as the previous frame's diff baseline.
    pub fn clear(&mut self) {
        self.lines.clear();
    }

    /// Append a screen line to the end of the line stream (shared entry point for consecutive message-area lines and fixed-area tail lines).
    /// Styles/text of each segment in the line are preserved as-is; if total width exceeds `width`, truncate by column width (never split wide characters).
    /// Consumer: `InteractiveUi::render_scrollback`.
    pub fn push_line(&mut self, line: Vec<StyledSegment>) {
        self.lines.push(truncate_line(line, self.width));
    }

    /// Convenience entry: push plain text as a single-segment line (default/specified style; used for fixed-area space padding, etc.).
    /// Consumer: `render_scrollback` (fixed-area placeholder lines, pending lines).
    pub fn push_text(&mut self, text: impl Into<String>, style: Style) {
        self.lines.push(vec![StyledSegment {
            style,
            text: text.into(),
        }]);
    }

    /// Total content rows this frame (= `lines.len()`); used by TuiLoop to decide whether to trigger scroll advance.
    pub fn content_height(&self) -> usize {
        self.lines.len()
    }

    /// Canvas column width (per-line truncation / single-line measured width); lets the app side build component render buffers per screen width.
    pub fn width(&self) -> u16 {
        self.width
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
    pub(crate) fn flush<W: Write>(&mut self, out: &mut W) -> Result<(), TerminalError> {
        let height = (self.height as usize).max(1);
        let width = self.width as usize;
        let new_len = self.lines.len();
        let prev_len = self.prev_lines.len();

        // Find first / last changed lines (total-order comparison of Vec<StyledSegment>).
        let max_lines = new_len.max(prev_len);
        let mut first: Option<usize> = None;
        let mut last: Option<usize> = None;
        for i in 0..max_lines {
            let old = self.prev_lines.get(i);
            let new = self.lines.get(i);
            if old != new {
                if first.is_none() {
                    first = Some(i);
                }
                last = Some(i);
            }
        }

        let Some(first) = first else {
            // No visible change: keep previous frame state (leave prev_lines untouched).
            return Ok(());
        };
        let last = last.unwrap();

        // Full redraw path: first frame / changed region above the previous viewport (already scrolled away) / content shrank overall.
        if prev_len == 0 || first < self.prev_viewport_top || first >= new_len {
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
        self.prev_lines = self.lines.clone();
        self.prev_height = height as u16;
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
        self.prev_lines = self.lines.clone();
        self.prev_height = height as u16;
        Ok(())
    }
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
    fn test_push_line_truncates_to_width_and_keeps_wide_chars() {
        let mut c = ScrollbackCanvas::new(4, 10);
        c.push_line(vec![plain("abcdef")]);
        assert_eq!(c.lines[0].len(), 1);
        assert_eq!(c.lines[0][0].text, "abcd");

        // Wide chars: ab(2) + CJK char(2) = 4 exactly fills
        let mut c2 = ScrollbackCanvas::new(4, 10);
        c2.push_line(vec![plain("ab中")]);
        assert_eq!(c2.lines[0][0].text, "ab中");

        // 3 columns can't fit a wide char: don't split it in half, drop it entirely
        let mut c3 = ScrollbackCanvas::new(3, 10);
        c3.push_line(vec![plain("ab中")]);
        assert_eq!(c3.lines[0][0].text, "ab");
    }

    // ── flush: full repaint (first frame) ──

    #[test]
    fn test_first_flush_writes_all_lines_and_sets_viewport() {
        let mut c = ScrollbackCanvas::new(10, 3);
        for i in 0..5 {
            c.push_line(vec![plain(&format!("line{i}"))]);
        }
        let mut out = Vec::new();
        c.flush(&mut out).unwrap();
        assert!(out.windows(5).any(|w| w == b"line0"), "First frame should write all lines");
        assert!(out.windows(5).any(|w| w == b"line4"));
        assert_eq!(c.prev_viewport_top, 2, "height=3, 5 lines → viewport top=5-3=2");
        assert_eq!(c.prev_lines.len(), 5);
        assert_eq!(c.hardware_cursor_row, 4);
    }

    #[test]
    fn test_first_flush_fits_within_height_viewport_zero() {
        let mut c = ScrollbackCanvas::new(10, 5);
        c.push_line(vec![plain("a")]);
        c.push_line(vec![plain("b")]);
        c.flush(&mut Vec::new()).unwrap();
        assert_eq!(c.prev_viewport_top, 0, "Content does not exceed screen → viewport top=0");
    }

    // ── flush: diff (no change / append / scroll beyond screen) ──

    #[test]
    fn test_no_change_flush_produces_no_output() {
        let mut c = ScrollbackCanvas::new(10, 3);
        c.push_line(vec![plain("a")]);
        c.flush(&mut Vec::new()).unwrap();
        let mut out = Vec::new();
        c.flush(&mut out).unwrap();
        assert!(out.is_empty(), "Diff should output no bytes when unchanged");
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
        assert!(out.windows(1).any(|w| w == b"b"), "Diff should output newly added lines");
        assert_eq!(c.prev_viewport_top, 0, "Viewport top unchanged when content fits screen");
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
        assert!(out.windows(2).any(|w| w == b"r2"), "Latest line should be visible");
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
}
