// REQ-APP-11 (pi-358): visual-truncate — tail-truncate by visual line count.
// Corresponds to original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/visual-truncate.ts.
// Wrapping semantics align with zen-tui wrap::word_wrap_line (same width calculation as the Text component),
// Returns **content lines** (without padding); the caller writes the buffer with the same padding_x, ensuring truncation matches display.

use zen_tui::wrap::{normalize_tabs, word_wrap_line};

/// Tail-truncation result: visible lines and hidden line count.
pub struct VisualTruncateResult {
    /// Visible visual lines (content text, excluding left/right padding).
    pub lines: Vec<String>,
    /// Number of skipped visual lines.
    pub skipped: usize,
}

/// Called when bash/tool execution preview tail-truncates oversized output to N visual lines —
/// "Drop what doesn't fit, keep only the bottom N lines"; returns visible lines and hidden line count.
///
/// After tab expansion, each logical line wraps at content width `width - padding_x*2` (minimum 1 column),
/// Same semantics as zen-tui `Text` component; when total visual lines exceed the limit, take the last `max_visual_lines` lines.
/// Empty text returns an empty result; when `max_visual_lines = 0`, there are no visible lines (skipped = total).
///
/// Consumers: bash-execution, tool-execution (PREVIEW_LINES=20).
pub fn truncate_to_visual_lines(
    text: &str,
    max_visual_lines: usize,
    width: u16,
    padding_x: u16,
) -> VisualTruncateResult {
    let normalized = normalize_tabs(text);
    if normalized.is_empty() {
        return VisualTruncateResult {
            lines: Vec::new(),
            skipped: 0,
        };
    }

    let content_width = width.saturating_sub(padding_x * 2).max(1) as usize;

    // Wrap each logical line → all visual lines
    let mut visual: Vec<String> = Vec::new();
    for line in normalized.split('\n') {
        for chunk in word_wrap_line(line, content_width) {
            visual.push(chunk.text);
        }
    }

    let total = visual.len();
    if total <= max_visual_lines {
        return VisualTruncateResult {
            lines: visual,
            skipped: 0,
        };
    }

    let start = total - max_visual_lines;
    let skipped = start;
    VisualTruncateResult {
        lines: visual.split_off(start),
        skipped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_no_truncation() {
        let r = truncate_to_visual_lines("hello", 20, 40, 0);
        assert_eq!(r.lines, vec!["hello"]);
        assert_eq!(r.skipped, 0);
    }

    #[test]
    fn empty_text_returns_empty() {
        let r = truncate_to_visual_lines("", 5, 40, 0);
        assert!(r.lines.is_empty());
        assert_eq!(r.skipped, 0);
    }

    #[test]
    fn keeps_tail_lines_when_over_limit() {
        let r = truncate_to_visual_lines("a\nb\nc\nd", 2, 40, 0);
        assert_eq!(r.lines, vec!["c", "d"]);
        assert_eq!(r.skipped, 2);
    }

    #[test]
    fn wraps_wide_lines_and_counts_visual_rows() {
        // Content width 8: each line wraps into 2 visual lines (e.g. "aaaa " + "bbbb")
        let r = truncate_to_visual_lines("aaaa bbbb cccc dddd", 3, 8, 0);
        assert_eq!(r.lines.len(), 3);
        assert_eq!(r.skipped, 1);
    }

    #[test]
    fn padding_reduces_content_width() {
        // padding_x = 2 → content width 4 → more wrapping
        let no_pad = truncate_to_visual_lines("aaaa bbbb", 5, 8, 0);
        let pad = truncate_to_visual_lines("aaaa bbbb", 5, 8, 2);
        assert!(pad.lines.len() > no_pad.lines.len());
    }

    #[test]
    fn max_visual_lines_zero_keeps_nothing() {
        let r = truncate_to_visual_lines("a\nb", 0, 40, 0);
        assert!(r.lines.is_empty());
        assert_eq!(r.skipped, 2);
    }
}
