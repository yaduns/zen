// REQ-TUI-03 (pi-407): word navigation — boundary calculation for word-wise movement/deletion.
// Intl.Segmenter (word granularity) → unicode-segmentation `split_word_bounds` (UAX #29);
// Intra-word ASCII punctuation re-segmentation uses crate::wrap::is_punctuation_char (T-TU-03-6).
// Verified (node v24 Intl.Segmenter vs unicode-segmentation 1.13): all ASCII cases have identical segmentation results
// (`foo.bar`/`foo:bar`/`don't`/`foo_bar` stay as one segment; `foo-bar`/`foo...bar`/`path/to/file` are segmented),
// Only CJK Han grouping differs—node ICU groups by dictionary ("hello" "world"), while this module segments per character
// (standard UAX #29 behavior), recorded as an accepted deviation.
// See reference/pi-original/packages/tui/src/word-navigation.ts
use crate::wrap::is_punctuation_char;
use unicode_segmentation::UnicodeSegmentation;

/// Whether a word segment is word-like (contains letters/digits), approximating Intl.Segmenter's `isWordLike`.
/// Whitespace/punctuation segments are always false; Han characters such as "hello" are alphanumeric ✓.
fn is_word_like(segment: &str) -> bool {
    segment.chars().any(|c| c.is_alphanumeric())
}

/// Perform UAX #29 word segmentation on text and mark word-like, returning a list of (segment text, is_word_like).
/// Segment text is a borrowed slice of the original string, not copied.
fn word_segments(text: &str) -> Vec<(&str, bool)> {
    text.split_word_bounds()
        .map(|seg| (seg, is_word_like(seg)))
        .collect()
}

/// Called by editor cursor movement/word deletion when the user presses Ctrl+← (or Ctrl+Backspace to get the deletion boundary),
/// Compute the cursor position after moving back one word from `cursor`—like stepping back word by word,
/// Intra-word punctuation (`.`, `/`, `:`, etc.) are also stopping points; `foo.bar` is traversed in three segments.
///
/// Pure function, does not change any state. `cursor == 0` always returns 0; when `cursor > text.len()` is out of bounds
/// clamp to `text.len()` first (callers guarantee no out-of-bounds; clamping is defensive).
/// Semantics (matches pi `findWordBackward`): first skip trailing whitespace; then take the last UAX#29 word segment—
/// If the word segment is word-like (contains letters/digits), within it use ASCII punctuation (`is_punctuation_char`)
/// take the position after the last punctuation as the stopping point (if no punctuation, skip the whole segment); if it is a punctuation segment, skip the whole segment;
/// If no segments remain, stay in place. CJK is segmented per character (registered deviation from pi's ICU dictionary grouping).
pub(super) fn find_word_backward(text: &str, cursor: usize) -> usize {
    if cursor == 0 {
        return 0;
    }
    let cursor = cursor.min(text.len());
    let mut segments = word_segments(&text[..cursor]);
    let mut new_cursor = cursor;

    // Skip trailing whitespace segment (aligned with pi "Skip trailing whitespace")
    while let Some((seg, word_like)) = segments.last() {
        if *word_like || !seg.chars().all(|c| c.is_whitespace()) {
            break;
        }
        let len = seg.len();
        segments.pop();
        new_cursor -= len;
    }

    let Some((last, word_like)) = segments.last() else {
        return new_cursor;
    };

    if *word_like {
        // Re-split within a word segment by ASCII punctuation: stop at the character after the last punctuation (aligned with pi
        // `PUNCTUATION_REGEX` re-split — UAX#29 keeps `foo.bar`/`foo:bar`/`don't` as a single segment);
        // If no punctuation, skip the whole segment. Punctuation is ASCII single-byte; byte offset is the stop-distance basis.
        match last
            .char_indices()
            .rev()
            .find(|(_, c)| is_punctuation_char(*c))
        {
            Some((punct_idx, _)) => new_cursor -= last.len() - (punct_idx + 1),
            None => new_cursor -= last.len(),
        }
    } else {
        // Punctuation segment (non-word-like, non-whitespace): skip the whole segment (aligned with pi "Skip non-word non-whitespace run")
        while let Some((seg, word_like)) = segments.last() {
            if *word_like || seg.chars().all(|c| c.is_whitespace()) {
                break;
            }
            let len = seg.len();
            segments.pop();
            new_cursor -= len;
        }
    }

    new_cursor
}

/// Called by editor cursor movement when the user presses Ctrl+→; computes the cursor position after advancing one word from `cursor` —
/// Symmetric to `find_word_backward`: advance word by word like pressing "next step", landing at the current word end (before punctuation)
/// or the next word end; for `foo.bar`, pressing once from the word start lands at the end of `foo` (before `.`).
///
/// Pure function. If `cursor >= text.len()`, always returns `text.len()`.
/// Semantics (aligned with pi `findWordForward`): first skip leading whitespace; then take the first word segment—word-like segment
/// Stop at the first ASCII punctuation within the segment (if none, skip the whole segment, i.e., stop at segment end); skip punctuation segments entirely.
pub(super) fn find_word_forward(text: &str, cursor: usize) -> usize {
    if cursor >= text.len() {
        return text.len();
    }
    let mut segments = word_segments(&text[cursor..]).into_iter();
    let mut new_cursor = cursor;

    // Skip leading whitespace segment (aligned with pi "Skip leading whitespace")
    loop {
        match segments.next() {
            Some((seg, word_like)) if !word_like && seg.chars().all(|c| c.is_whitespace()) => {
                new_cursor += seg.len();
            }
            Some((seg, word_like)) => {
                if word_like {
                    // Stop at the first ASCII punctuation within the segment (if none, skip whole segment = stop at segment end)
                    match seg.char_indices().find(|(_, c)| is_punctuation_char(*c)) {
                        Some((idx, _)) => new_cursor += idx,
                        None => new_cursor += seg.len(),
                    }
                } else {
                    // Punctuation segment: skip the whole segment (aligned with pi "Skip non-word non-whitespace run")
                    new_cursor += seg.len();
                    for (seg2, word_like2) in segments {
                        if word_like2 || seg2.chars().all(|c| c.is_whitespace()) {
                            break;
                        }
                        new_cursor += seg2.len();
                    }
                }
                break;
            }
            None => break,
        }
    }

    new_cursor
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── find_word_backward (migrated from pi word-navigation.test.ts backward cases, ASCII 1:1) ──

    #[test]
    fn test_backward_basic_words() {
        let text = "hello world";
        assert_eq!(find_word_backward(text, 11), 6);
        assert_eq!(find_word_backward(text, 6), 0);
    }

    #[test]
    fn test_backward_dotted() {
        let text = "foo.bar";
        assert_eq!(find_word_backward(text, 7), 4);
        assert_eq!(find_word_backward(text, 4), 3);
        assert_eq!(find_word_backward(text, 3), 0);
    }

    #[test]
    fn test_backward_colon() {
        let text = "foo:bar";
        assert_eq!(find_word_backward(text, 7), 4);
        assert_eq!(find_word_backward(text, 4), 3);
        assert_eq!(find_word_backward(text, 3), 0);
    }

    #[test]
    fn test_backward_path() {
        let text = "path/to/file";
        assert_eq!(find_word_backward(text, 12), 8);
        assert_eq!(find_word_backward(text, 8), 7);
        // "/to" is a single word-like segment and "/" is an intra-word punctuation break point
        assert_eq!(find_word_backward(text, 7), 5);
        assert_eq!(find_word_backward(text, 5), 4);
        assert_eq!(find_word_backward(text, 4), 0);
    }

    #[test]
    fn test_backward_whitespace_at_boundaries() {
        let text = "  hello  ";
        assert_eq!(find_word_backward(text, 9), 2);
        assert_eq!(find_word_backward(text, 2), 0);
    }

    #[test]
    fn test_backward_punctuation_run() {
        let text = "foo...bar";
        assert_eq!(find_word_backward(text, 9), 6);
        assert_eq!(find_word_backward(text, 6), 3);
        assert_eq!(find_word_backward(text, 3), 0);
    }

    #[test]
    fn test_backward_cursor_at_zero_returns_zero() {
        assert_eq!(find_word_backward("hello", 0), 0);
    }

    #[test]
    fn test_backward_hyphen_split_by_uax29() {
        // Both UAX#29 and Intl.Segmenter split foo-bar into [foo][-][bar] — "-" is a separate punctuation segment
        let text = "foo-bar";
        assert_eq!(find_word_backward(text, 7), 4);
        assert_eq!(find_word_backward(text, 4), 3);
        assert_eq!(find_word_backward(text, 3), 0);
    }

    #[test]
    fn test_backward_underscore_kept_in_word() {
        // "_" is not in PUNCTUATION_REGEX: foo_bar is skipped as a whole segment (T-TU-03-6 ruling)
        let text = "foo_bar";
        assert_eq!(find_word_backward(text, 7), 0);
    }

    #[test]
    fn test_backward_cjk_per_char() {
        // Registered deviation: unicode-segmentation segments per character (pi ICU dictionary grouping gives "hello" "world"),
        // so backward from the start of "test" crosses only one character (boundary) instead of pi's two characters (world).
        let text = "你好世界 test";
        assert_eq!(find_word_backward(text, 17), 13); // Back to the start of "test" (aligned with pi 9→5)
        assert_eq!(find_word_backward(text, 13), 9); // Backward one character to the start of "boundary" (pi is 5→2, deviation)
        assert_eq!(find_word_backward(text, 9), 6); // Backward one character to the start of "world"
        assert_eq!(find_word_backward(text, 6), 3); // Backward one character to the start of "good"
        assert_eq!(find_word_backward(text, 3), 0);
    }

    #[test]
    fn test_backward_cursor_out_of_bounds_clamped() {
        assert_eq!(find_word_backward("hello", 99), 0);
    }

    // ── find_word_forward (migrated from pi word-navigation.test.ts forward cases, ASCII 1:1) ──

    #[test]
    fn test_forward_basic_words() {
        let text = "hello world";
        assert_eq!(find_word_forward(text, 0), 5);
        assert_eq!(find_word_forward(text, 5), 11);
    }

    #[test]
    fn test_forward_dotted() {
        let text = "foo.bar";
        assert_eq!(find_word_forward(text, 0), 3);
        assert_eq!(find_word_forward(text, 3), 4);
        assert_eq!(find_word_forward(text, 4), 7);
    }

    #[test]
    fn test_forward_colon() {
        let text = "foo:bar";
        assert_eq!(find_word_forward(text, 0), 3);
        assert_eq!(find_word_forward(text, 3), 4);
        assert_eq!(find_word_forward(text, 4), 7);
    }

    #[test]
    fn test_forward_path() {
        let text = "path/to/file";
        assert_eq!(find_word_forward(text, 0), 4);
        assert_eq!(find_word_forward(text, 4), 5);
        assert_eq!(find_word_forward(text, 5), 7);
        assert_eq!(find_word_forward(text, 7), 8);
        assert_eq!(find_word_forward(text, 8), 12);
    }

    #[test]
    fn test_forward_whitespace_at_boundaries() {
        let text = "  hello  ";
        assert_eq!(find_word_forward(text, 0), 7);
        assert_eq!(find_word_forward(text, 7), 9);
    }

    #[test]
    fn test_forward_punctuation_run() {
        let text = "foo...bar";
        assert_eq!(find_word_forward(text, 0), 3);
        assert_eq!(find_word_forward(text, 3), 6);
        assert_eq!(find_word_forward(text, 6), 9);
    }

    #[test]
    fn test_forward_cursor_at_end_returns_end() {
        assert_eq!(find_word_forward("hello", 5), 5);
    }

    #[test]
    fn test_forward_hyphen_split_by_uax29() {
        let text = "foo-bar";
        assert_eq!(find_word_forward(text, 0), 3);
        assert_eq!(find_word_forward(text, 3), 4);
        assert_eq!(find_word_forward(text, 4), 7);
    }

    #[test]
    fn test_forward_underscore_kept_in_word() {
        let text = "foo_bar";
        assert_eq!(find_word_forward(text, 0), 7);
    }

    #[test]
    fn test_forward_cjk_per_char_walks_to_end() {
        // Per-character segmentation (registered deviation): each forward press advances one character, eventually reaching end of line
        let text = "你好世界 test";
        let mut pos = 0;
        while pos < text.len() {
            let next = find_word_forward(text, pos);
            assert!(next > pos);
            pos = next;
        }
        assert_eq!(pos, text.len());
    }

    #[test]
    fn test_forward_apostrophe_kept_in_word() {
        // "don't" is a single segment (consistent with UAX#29/Intl), ' is intra-word — forward stops before ', pressing again skips the ' segment to word end
        let text = "don't";
        assert_eq!(find_word_forward(text, 0), 3);
        assert_eq!(find_word_forward(text, 3), 4);
        assert_eq!(find_word_forward(text, 4), 5);
    }

    // ── is_word_like (derived predicate) ──

    #[test]
    fn test_is_word_like() {
        assert!(is_word_like("hello"));
        assert!(is_word_like("foo.bar"));
        assert!(is_word_like("你好"));
        assert!(is_word_like("123"));
        assert!(!is_word_like("  "));
        assert!(!is_word_like("."));
        assert!(!is_word_like("/"));
        assert!(!is_word_like(""));
    }
}
