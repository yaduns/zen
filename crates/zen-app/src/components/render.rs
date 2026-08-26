// REQ-APP-11 (pi-358): Shared render helpers for message/tool components — mixed-style line write buffer + markdown segment line write buffer.
// The pure markdown conversion module (T-TU-02-1) produces `Vec<Vec<StyledSegment>>`; the screen diff model (T-TU-01-2)
// only provides single-style `ScreenBuffer::write`; this module is the "draw segment by segment" bridge between the two, consumed only by this crate's components.

use zen_tui::highlight::StyledSegment;
use zen_tui::markdown::{render_markdown, MarkdownTheme};
use zen_tui::screen::Style;
use zen_tui::screen::{visible_width, ScreenBuffer};

/// Shared low-level backing for message/tool components writing mixed-style lines into the screen buffer — multi-segment styling per line (markdown segments,
/// key-hint combos) written segment by segment, truncated within the width, no line wrapping. Consumed only by this crate's components (not marked pub).
///
/// `width` is the available display width; segment text accumulates by display width, stopping once exceeded (later segments are not written).
/// Returns the number of columns actually written (≤ width).
pub(crate) fn write_segment_row(
    buf: &mut ScreenBuffer,
    x: u16,
    y: u16,
    width: u16,
    segments: &[StyledSegment],
) -> u16 {
    let mut col: u16 = 0;
    for seg in segments {
        let seg_width = visible_width(&seg.text) as u16;
        if col + seg_width > width {
            // Partial fit: truncate-write by remaining width, then terminate the whole line
            let remaining = width - col;
            buf.write(x + col, y, &seg.style, &seg.text);
            return col + remaining;
        }
        buf.write(x + col, y, &seg.style, &seg.text);
        col += seg_width;
    }
    col
}

/// Message components (assistant/user/custom/branch, etc.) write markdown render results segment by segment after leading whitespace —
/// "draw markdown text into the current line region". The caller invokes it inside render() after determining text and theme;
/// returns the number of lines actually written (≤ the passed height budget); excess is silently truncated.
///
/// - `padding_x` is the left padding; each line starts with padding_x spaces, then writes segments; content width =
///   `width - padding_x*2` (clamped to at least 1 column, matching zen-tui Text component semantics).
/// - `theme` is the already-resolved MarkdownTheme (the thinking block uses a derived copy of `Theme::markdown_theme()`,
///   overriding foreground and italic).
#[allow(clippy::too_many_arguments)]
pub fn write_markdown(
    buf: &mut ScreenBuffer,
    x: u16,
    y: u16,
    width: u16,
    height: u16,
    theme: &MarkdownTheme,
    text: &str,
    padding_x: u16,
) -> u16 {
    if width == 0 || height == 0 {
        return 0;
    }
    let content_width = width.saturating_sub(padding_x * 2).max(1) as usize;
    let rows = render_markdown(theme, text, content_width);
    let painted = (rows.len() as u16).min(height);

    for (i, row) in rows.iter().take(painted as usize).enumerate() {
        let row_y = y + i as u16;
        // Left padding (default-style spaces to avoid residual background)
        let pad_style = Style::default();
        buf.write(x, row_y, &pad_style, &" ".repeat(padding_x as usize));
        write_segment_row(buf, x + padding_x, row_y, width - padding_x, row);
    }
    painted
}

#[cfg(test)]
mod tests {
    use super::*;
    use zen_tui::screen::Style;

    fn seg(style: Style, text: &str) -> StyledSegment {
        StyledSegment {
            style,
            text: text.to_string(),
        }
    }

    fn flush_text(buf: &mut ScreenBuffer) -> String {
        // flush output contains crossterm diff commands (MoveTo/SetColors/Print per cell);
        // here control bytes are stripped to restore "visible text" (printable chars, order preserved) for contains assertions
        let mut out = Vec::new();
        buf.flush(&mut out).unwrap();
        crate::components::strip_ansi(&String::from_utf8_lossy(&out))
            .chars()
            .filter(|c| !c.is_control())
            .collect()
    }

    #[test]
    fn write_segment_row_writes_segments_left_to_right() {
        let mut buf = ScreenBuffer::new(20, 1);
        let s = Style::default();
        let segments = vec![seg(s.clone(), "ab"), seg(s.clone(), "cd")];
        let used = write_segment_row(&mut buf, 0, 0, 20, &segments);
        assert_eq!(used, 4);
        assert!(flush_text(&mut buf).contains("abcd"));
    }

    #[test]
    fn write_segment_row_clips_at_width() {
        let mut buf = ScreenBuffer::new(20, 1);
        let s = Style::default();
        let segments = vec![seg(s.clone(), "abc"), seg(s.clone(), "defg")];
        // Width 5: abc(3) + remaining 2 can't fit defg → only writes abc, de is truncated by remaining width (ScreenBuffer column truncation)
        let used = write_segment_row(&mut buf, 0, 0, 5, &segments);
        assert_eq!(used, 5);
        let text = flush_text(&mut buf);
        assert!(text.contains("abc"));
    }

    #[test]
    fn write_segment_row_no_segments_returns_zero() {
        let mut buf = ScreenBuffer::new(20, 1);
        assert_eq!(write_segment_row(&mut buf, 0, 0, 20, &[]), 0);
    }

    #[test]
    fn write_markdown_pads_left_and_renders_rows() {
        let mut buf = ScreenBuffer::new(30, 10);
        let theme = MarkdownTheme::default();
        let painted = write_markdown(&mut buf, 0, 0, 30, 10, &theme, "hello **world**", 2);
        assert!(painted >= 1);
        let text = flush_text(&mut buf);
        assert!(text.contains("hello"));
        assert!(text.contains("world"));
    }

    #[test]
    fn write_markdown_wraps_long_text() {
        let mut buf = ScreenBuffer::new(30, 20);
        let theme = MarkdownTheme::default();
        let long = "aaaa bbbb cccc dddd eeee ffff";
        let painted_wide = write_markdown(&mut buf, 0, 0, 30, 20, &theme, long, 0);
        let mut buf2 = ScreenBuffer::new(30, 20);
        let painted_narrow = write_markdown(&mut buf2, 0, 0, 8, 20, &theme, long, 0);
        assert!(
            painted_narrow > painted_wide,
            "narrower should wrap more rows"
        );
    }

    #[test]
    fn write_markdown_respects_height_budget() {
        let mut buf = ScreenBuffer::new(30, 20);
        let theme = MarkdownTheme::default();
        let text = "line one\n\nline two\n\nline three";
        // three paragraphs (including blank lines) → at least 3 lines; height budget 2 → only 2 lines rendered
        let painted = write_markdown(&mut buf, 0, 0, 30, 2, &theme, text, 0);
        assert_eq!(painted, 2);
    }

    #[test]
    fn write_markdown_empty_text_renders_nothing() {
        let mut buf = ScreenBuffer::new(30, 10);
        let theme = MarkdownTheme::default();
        assert_eq!(write_markdown(&mut buf, 0, 0, 30, 10, &theme, "", 0), 0);
        assert_eq!(write_markdown(&mut buf, 0, 0, 0, 10, &theme, "x", 0), 0);
    }
}
