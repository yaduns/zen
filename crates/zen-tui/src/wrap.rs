// REQ-TUI-01 (pi-413): width-aware text processing family — word wrapping / truncation / column slicing / tab expansion.
// The original word-wrapping logic was moved in from editor/wrap.rs (T-TU-01-3): consumers have expanded from "editor/ only" to
// editor/ + components/ (T-TU-01-4), moved up to a crate-level module per the T-TU-01-4 integration plan.
// Planned consumers: T-TU-02-1 (markdown wrapping/truncation), T-TU-03-6 (pi-408 utils.ts claim,
// this family is its actual successor in this repository).
// Width measurement uniformly goes through screen::visible_width (unicode-width), not reimplemented here.
// TextChunk/word_wrap_line promoted from pub(super) to pub (components reads the text field);
// make_chunk/is_cjk reduced to module-private. Text styling always goes through Style (zen text does not embed ANSI),
// so no ANSI-aware branch from pi utils.ts is implemented (extractAnsiCode etc. are resolved with T-TU-03-6).
use crate::screen::visible_width;
use unicode_segmentation::UnicodeSegmentation;

/// Text fragment: a contiguous slice of the source text (text == line[start_index..end_index], byte offset).
/// Produced by `word_wrap_line`; consumed by both the editor (cursor column mapping) and components (display line text).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextChunk {
    pub text: String,       // Fragment text (visible text after wrapping; display is the buffer's source text)
    pub start_index: usize, // Starting byte offset in the source text (UTF-8, same unit as cursor_col/grapheme_indices)
    pub end_index: usize,   // Ending byte offset in the source text (exclusive; used for cursor attribution / range checks)
}

/// Wrap a single line at word boundaries: prefer word boundaries within max_width, allow breaks between adjacent CJK, and split overwide atomic segments into separate chunks.
/// **A fragment is a contiguous slice of the source text** (text == line[start_index..end_index], byte offset); whitespace belongs to the end of the previous fragment,
/// Never rebuild text — ensures displayed text matches the buffer character-for-character and cursor column mapping is linear.
/// Edge cases: an empty line or max_width = 0 returns a single empty chunk (preserving the caller's invariant of "at least 1 visual line per logical line");
/// If the whole line fits, return a single chunk; if a single grapheme's column width > max_width (extremely narrow terminal), it becomes its own chunk (1 column visual overflow, acceptable).
/// Consumers: editor::render::layout_text, editor::cursor::build_visual_line_map,
/// T-TU-01-4 components::text (text wrapping), components::settings_list (description wrapping), T-TU-02-1.
pub fn word_wrap_line(line: &str, max_width: usize) -> Vec<TextChunk> {
    // Degenerate input: return a single empty slice,
    // ensuring the caller's "every logical line >= 1 visual line" invariant holds.
    if line.is_empty() || max_width == 0 {
        return vec![TextChunk {
            text: String::new(),
            start_index: 0,
            end_index: 0,
        }];
    }

    // Whole row fits: return a single slice (aligns with pi fast path).
    if visible_width(line) <= max_width {
        return vec![TextChunk {
            text: line.to_string(),
            start_index: 0,
            end_index: line.len(),
        }];
    }

    // Grapheme sequence (byte offset, grapheme text); the whole pipeline only "slices", never "joins".
    let segments: Vec<(usize, &str)> = line.grapheme_indices(true).collect();

    let mut chunks: Vec<TextChunk> = Vec::new();
    let mut current_width = 0usize; // Cumulative display width of the current slice
    let mut chunk_start = 0usize; // Starting byte offset of the current slice in the original text

    // Wrap opportunity: a position where a line break is allowed (after the last run of whitespace / after a CJK adjacency boundary), recorded by the previous iteration at the end of "advance".
    // opp_index: byte offset of the opportunity; opp_width: accumulated width at the opportunity (used for backtracking).
    let mut opp_index: Option<usize> = None;
    let mut opp_width = 0usize;

    for (i, (char_index, grapheme)) in segments.iter().enumerate() {
        let g_width = visible_width(grapheme);
        let is_ws = grapheme.chars().all(|c| c.is_whitespace());

        // ── Bounds check (before advancing) ──
        if current_width + g_width > max_width {
            // Prefer backtracking to the previous wrap opportunity (only viable if the remainder + current grapheme fits)
            match opp_index {
                Some(opp) if current_width - opp_width + g_width <= max_width => {
                    chunks.push(make_chunk(line, chunk_start, opp));
                    chunk_start = opp;
                    current_width -= opp_width;
                }
                // No available opportunity, or backtracking still doesn't fit (e.g. wide character): force a break at the current position.
                // Skip when chunk_start == char_index (avoids empty slices).
                _ if chunk_start < *char_index => {
                    chunks.push(make_chunk(line, chunk_start, *char_index));
                    chunk_start = *char_index;
                    current_width = 0;
                }
                _ => {}
            }
            opp_index = None;
        }

        // ── A single atomic segment wider than max_width (extremely narrow terminal: max_width=1 and grapheme column width=2) ──
        // pi re-slices paste-marker atomic segments recursively by grapheme; zen has no markers, so the sole over-wide atomic segment is already
        // a single grapheme and cannot be subdivided further (recursion wouldn't converge) — make it an independent slice (1-column visual overflow, acceptable).
        if g_width > max_width {
            chunks.push(make_chunk(line, *char_index, *char_index + grapheme.len()));
            chunk_start = *char_index + grapheme.len();
            current_width = 0;
            opp_index = None;
            continue;
        }

        // ── Advance current grapheme ──
        current_width += g_width;

        // ── Record wrap opportunity ──
        // (1) whitespace followed by non-whitespace (consecutive spaces merge: break point goes after the following space)
        // (2) either adjacent side is CJK (CJK allows a break at any adjacency boundary)
        if let Some((next_index, next_grapheme)) = segments.get(i + 1) {
            let next_is_ws = next_grapheme.chars().all(|c| c.is_whitespace());
            if is_ws && !next_is_ws {
                opp_index = Some(*next_index);
                opp_width = current_width;
            } else if !is_ws && !next_is_ws {
                let cjk = is_cjk(grapheme.chars().next().unwrap_or('\0'));
                let next_cjk = is_cjk(next_grapheme.chars().next().unwrap_or('\0'));
                if cjk || next_cjk {
                    opp_index = Some(*next_index);
                    opp_width = current_width;
                }
            }
        }
    }

    // Push the remaining content as the final slice; chunk_start already at end-of-row means all content was fully closed above (over-wide final grapheme becomes its own slice),
    // so no empty slice is produced (mirrors the chunk_start < char_index guard in the forced-break branch).
    if chunk_start < line.len() {
        chunks.push(make_chunk(line, chunk_start, line.len()));
    }

    chunks
}

/// Generates slice chunks for the [start,end) byte range of the source text (a private helper for word_wrap_line).
fn make_chunk(line: &str, start: usize, end: usize) -> TextChunk {
    TextChunk {
        text: line[start..end].to_string(),
        start_index: start,
        end_index: end,
    }
}

/// Truncate and append ellipsis when it does not fit
/// Truncate text by display columns (grapheme-aware: never split a wide character/emoji; width counted by `visible_width` columns).
/// If the entire text fits within max_width, **return it unchanged** (no ellipsis, matching pi `truncateToWidth` semantics);
/// If it does not fit, keep the longest fitting grapheme prefix and append ellipsis; the ellipsis is also width-constrained,
/// when max_width cannot fit the full ellipsis, keep only its prefix.
/// Edge cases: max_width = 0 or empty text returns an empty string; a wide character falling within the keep boundary is kept whole,
/// otherwise it is discarded whole (never split in half).
/// Consumers: T-TU-01-4 TruncatedText (pending messages), SelectList (main column/scroll info),
/// SettingsList (values/hints), T-TU-02-1 (markdown overflow handling).
pub fn truncate_to_width(text: &str, max_width: usize, ellipsis: &str) -> String {
    if max_width == 0 || text.is_empty() {
        return String::new();
    }
    // Whole text fits: return unchanged (no ellipsis)
    if visible_width(text) <= max_width {
        return text.to_string();
    }

    let ellipsis_width = visible_width(ellipsis);
    if ellipsis_width >= max_width {
        // Even the ellipsis itself does not fit: keep only its visible prefix
        return truncate_graphemes(ellipsis, max_width);
    }

    // Keep the longest prefix of max_width - ellipsis_width columns + ellipsis
    let keep = max_width - ellipsis_width;
    let mut result = String::new();
    let mut width = 0usize;
    for g in text.graphemes(true) {
        let w = visible_width(g);
        if width + w > keep {
            break;
        }
        result.push_str(g);
        width += w;
    }
    result.push_str(ellipsis);
    result
}

/// Take the longest grapheme prefix by display columns (private helper for truncate_to_width; never split a wide character).
fn truncate_graphemes(text: &str, max_width: usize) -> String {
    let mut result = String::new();
    let mut width = 0usize;
    for g in text.graphemes(true) {
        let w = visible_width(g);
        if width + w > max_width {
            break;
        }
        result.push_str(g);
        width += w;
    }
    result
}

/// The "visible window" during single-line horizontal scrolling
/// Return the text in line's display column range [start_col, start_col+length) (grapheme-aware)
/// Wide characters intersecting the window boundary are **skipped whole** (never split a wide character):
/// Wide characters starting before the window are not kept, nor are those whose left edge is inside the window but right edge extends beyond it.
/// Edge cases: length = 0 returns an empty string; when the window is entirely past the text width, returns an empty string (iteration naturally exhausts).
/// Consumer: T-TU-01-4 Input (horizontal-scroll visible window).
pub fn slice_by_column(line: &str, start_col: usize, length: usize) -> String {
    if length == 0 {
        return String::new();
    }
    let end_col = start_col + length;
    let mut result = String::new();
    let mut col = 0usize;
    for g in line.graphemes(true) {
        let w = visible_width(g);
        // Keep only what falls entirely within the window; discard wide characters crossing the boundary whole
        if col >= start_col && col + w <= end_col {
            result.push_str(g);
        }
        col += w;
        if col >= end_col {
            break;
        }
    }
    result
}

/// Expand `\t` to 3 spaces for terminal tab stops
/// may break logical line wrapping; the display layer uniformly expands it to a fixed width, with width semantics consistent with `visible_width`).
/// Consumer: T-TU-01-4 components::text (normalization before Text rendering).
pub fn normalize_tabs(text: &str) -> String {
    text.replace('\t', "   ")
}

/// Whether the char is a CJK line-break character (aligns with pi's `cjkBreakRegex`: Han / Hiragana / Katakana / Hangul / Bopomofo).
/// Grapheme clusters are identified by their first char (the base of a combined cluster); blocks approximate Unicode Script_Extensions,
/// avoiding the unicode-script dependency; differences from pi are limited to narrow edge cases (e.g. pi doesn't break before full-width Latin letters).
fn is_cjk(c: char) -> bool {
    matches!(c,
        // Han (Unified Ideographs + Extensions A–H + Compatibility block)
        '\u{3400}'..='\u{4DBF}'
        | '\u{4E00}'..='\u{9FFF}'
        | '\u{F900}'..='\u{FAFF}'
        | '\u{20000}'..='\u{2A6DF}'
        | '\u{2A700}'..='\u{2B73F}'
        | '\u{2B740}'..='\u{2B81F}'
        | '\u{2B820}'..='\u{2CEAF}'
        | '\u{2CEB0}'..='\u{2EBEF}'
        | '\u{30000}'..='\u{3134F}'
        | '\u{2F800}'..='\u{2FA1F}'
        // CJK Symbols and Punctuation (mostly Han script, including 、。〈〉 etc.)
        | '\u{3000}'..='\u{303F}'
        // Hiragana / Katakana
        | '\u{3040}'..='\u{309F}'
        | '\u{30A0}'..='\u{30FF}'
        | '\u{31F0}'..='\u{31FF}'
        // Hangul (Jamo + Syllables)
        | '\u{1100}'..='\u{11FF}'
        | '\u{A960}'..='\u{A97F}'
        | '\u{AC00}'..='\u{D7AF}'
        | '\u{D7B0}'..='\u{D7FF}'
        // Bopomofo
        | '\u{3100}'..='\u{312F}'
        | '\u{31A0}'..='\u{31BF}'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── word_wrap_line (moved from editor/wrap.rs, T-TU-01-3) ──

    #[test]
    fn test_word_wrap_empty_line() {
        let chunks = word_wrap_line("", 10);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "");
        assert_eq!(chunks[0].start_index, 0);
        assert_eq!(chunks[0].end_index, 0);
    }

    #[test]
    fn test_word_wrap_zero_width() {
        let chunks = word_wrap_line("hello", 0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "");
    }

    #[test]
    fn test_word_wrap_fits_in_width() {
        let chunks = word_wrap_line("hello", 10);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hello");
        assert_eq!(chunks[0].start_index, 0);
        assert_eq!(chunks[0].end_index, 5);
    }

    #[test]
    fn test_word_wrap_at_word_boundary() {
        let chunks = word_wrap_line("hello world", 6);
        // "hello " fits in 6, "world" goes to next chunk
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "hello ");
        assert_eq!(chunks[1].text, "world");
    }

    #[test]
    fn test_word_wrap_multiple_words() {
        let chunks = word_wrap_line("ab cd ef gh", 5);
        // "ab cd" = 5 → chunk 0
        // " ef " → chunk 1 (leading space + "ef" + space)
        // "gh" → chunk 2
        assert!(chunks.len() >= 2);
        // Verify continuity: chunks cover the full line
        assert_eq!(chunks[0].start_index, 0);
        assert_eq!(chunks.last().unwrap().end_index, 11);
    }

    #[test]
    fn test_word_wrap_long_word_no_boundary() {
        let chunks = word_wrap_line("abcdefghij", 3);
        // No word boundaries, forced breaks
        assert!(chunks.len() > 1);
        // Verify all chunks are contiguous
        for i in 0..chunks.len() - 1 {
            assert_eq!(chunks[i].end_index, chunks[i + 1].start_index);
        }
    }

    #[test]
    fn test_word_wrap_cjk_breakable() {
        // CJK characters are breakable between any two
        let chunks = word_wrap_line("你好世界", 2);
        // Each CJK char is width 2, so each gets its own chunk at max_width=2
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].text, "你");
        assert_eq!(chunks[1].text, "好");
        assert_eq!(chunks[2].text, "世");
        assert_eq!(chunks[3].text, "界");
    }

    #[test]
    fn test_word_wrap_cjk_with_ascii() {
        let chunks = word_wrap_line("ab你好cd", 4);
        // "ab" = 2 + one CJK char = 2 → total 4, fits
        // But CJK boundary allows break after "ab" if needed
        // Actually: "ab" (2) + CJK (2) = 4 fits, then CJK (2) + "cd" (2) = 4
        assert!(!chunks.is_empty());
        // Verify continuity
        assert_eq!(chunks[0].start_index, 0);
        assert_eq!(chunks.last().unwrap().end_index, "ab你好cd".len());
    }

    #[test]
    fn test_word_wrap_wide_char_exceeds_width() {
        // A CJK char (width 2) in max_width=1
        let chunks = word_wrap_line("你", 1);
        // Wide char exceeds width, gets its own chunk (visual overflow)
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "你");
    }

    #[test]
    fn test_word_wrap_preserves_whitespace() {
        let chunks = word_wrap_line("hello   world", 6);
        // "hello " (trailing spaces go with previous word)
        assert_eq!(chunks[0].text, "hello ");
    }

    #[test]
    fn test_word_wrap_only_whitespace() {
        let chunks = word_wrap_line("   ", 10);
        // Whitespace-only line fits in width
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "   ");
    }

    #[test]
    fn test_word_wrap_continuity() {
        // Verify chunks are contiguous slices of the original
        let line = "the quick brown fox jumps over the lazy dog";
        let chunks = word_wrap_line(line, 10);
        assert_eq!(chunks[0].start_index, 0);
        assert_eq!(chunks.last().unwrap().end_index, line.len());
        for i in 0..chunks.len() - 1 {
            assert_eq!(chunks[i].end_index, chunks[i + 1].start_index);
        }
    }

    #[test]
    fn test_word_wrap_chunk_text_matches_slice() {
        let line = "hello world test";
        let chunks = word_wrap_line(line, 8);
        for chunk in &chunks {
            assert_eq!(chunk.text, &line[chunk.start_index..chunk.end_index]);
        }
    }

    // ── is_cjk (moved from editor/wrap.rs) ──

    #[test]
    fn test_is_cjk_han() {
        assert!(is_cjk('你'));
        assert!(is_cjk('好'));
        assert!(is_cjk('中'));
    }

    #[test]
    fn test_is_cjk_hiragana() {
        assert!(is_cjk('あ'));
        assert!(is_cjk('い'));
    }

    #[test]
    fn test_is_cjk_katakana() {
        assert!(is_cjk('ア'));
        assert!(is_cjk('イ'));
    }

    #[test]
    fn test_is_cjk_hangul() {
        assert!(is_cjk('가'));
        assert!(is_cjk('나'));
    }

    #[test]
    fn test_is_cjk_bopomofo() {
        assert!(is_cjk('ㄅ'));
        assert!(is_cjk('ㄆ'));
    }

    #[test]
    fn test_is_cjk_cjk_symbols() {
        assert!(is_cjk('、'));
        assert!(is_cjk('。'));
    }

    #[test]
    fn test_is_cjk_ascii_false() {
        assert!(!is_cjk('a'));
        assert!(!is_cjk('Z'));
        assert!(!is_cjk('0'));
        assert!(!is_cjk(' '));
    }

    #[test]
    fn test_is_cjk_latin_extended_false() {
        assert!(!is_cjk('é'));
        assert!(!is_cjk('ñ'));
    }

    // ── make_chunk (private, tested indirectly via word_wrap_line) ──

    #[test]
    fn test_make_chunk_via_wrap_indices() {
        let chunks = word_wrap_line("abc", 10);
        assert_eq!(chunks[0].start_index, 0);
        assert_eq!(chunks[0].end_index, 3);
        assert_eq!(chunks[0].text, "abc");
    }

    // ── truncate_to_width ──

    #[test]
    fn test_truncate_fits_returns_original_without_ellipsis() {
        assert_eq!(truncate_to_width("hello", 10, "..."), "hello");
        assert_eq!(truncate_to_width("hello world", 11, "..."), "hello world");
    }

    #[test]
    fn test_truncate_overflow_appends_ellipsis() {
        assert_eq!(truncate_to_width("hello world", 8, "..."), "hello...");
        assert_eq!(truncate_to_width("hello world", 11, "..."), "hello world");
    }

    #[test]
    fn test_truncate_zero_width_or_empty() {
        assert_eq!(truncate_to_width("hello", 0, "..."), "");
        assert_eq!(truncate_to_width("", 10, "..."), "");
    }

    #[test]
    fn test_truncate_does_not_split_wide_char() {
        // "ABCD" each wide 2; max_width 5 → keep 2 columns = "A", otherwise it would split in half
        assert_eq!(truncate_to_width("一二三四", 5, "..."), "一...");
        // max_width 7 → keep 4 = "AB"; keep 3 cannot fit two wide characters (4 > 3), so only "A" remains
        assert_eq!(truncate_to_width("一二三四", 7, "..."), "一二...");
        assert_eq!(truncate_to_width("一二三四", 6, "..."), "一...");
    }

    #[test]
    fn test_truncate_ellipsis_wider_than_max() {
        // ellipsis "……" (width 2, each "…" width 1) ≥ max_width 1 → keep only the ellipsis prefix (grapheme-aware)
        assert_eq!(truncate_to_width("abcdef", 1, "……"), "…");
        // max 2 exactly fits the full ellipsis
        assert_eq!(truncate_to_width("abcdef", 2, "……"), "……");
    }

    #[test]
    fn test_truncate_exact_width_keeps_whole_wide_at_boundary() {
        // "Aabcde" width 7 > max 6 → keep 3; A(2)+a(1) exactly fills, wide character kept whole, b truncated
        assert_eq!(truncate_to_width("你abcde", 6, "..."), "你a...");
    }

    // ── slice_by_column ──

    #[test]
    fn test_slice_by_column_ascii_window() {
        assert_eq!(slice_by_column("hello world", 0, 5), "hello");
        assert_eq!(slice_by_column("hello world", 6, 5), "world");
    }

    #[test]
    fn test_slice_by_column_zero_length() {
        assert_eq!(slice_by_column("hello", 0, 0), "");
    }

    #[test]
    fn test_slice_window_past_text_width_returns_empty() {
        assert_eq!(slice_by_column("ab", 5, 3), "");
    }

    #[test]
    fn test_slice_by_column_skips_wide_char_crossing_window() {
        // "aXb": a at column 0, X at columns 1-2, b at column 3. Window [1,2) → does X fall entirely within the window?
        // X starts at column 1 and ends at column 3; its right edge exceeds end_col=2 → discarded whole → empty
        assert_eq!(slice_by_column("a中b", 1, 1), "");
        // Window [1,3) → X starts at 1 and ends at 3, entirely inside → kept
        assert_eq!(slice_by_column("a中b", 1, 2), "中");
        // Window [3,4) → b
        assert_eq!(slice_by_column("a中b", 3, 1), "b");
    }

    #[test]
    fn test_slice_by_column_start_mid_wide_char_skips_it() {
        // Window starts in the middle of a wide character (start_col=2, X occupies 1-2) → X's start column 1 < start_col,
        // so it is not kept; only its right half remains in window [2,3) → empty (matching pi strict semantics: discard whole characters crossing the boundary)
        assert_eq!(slice_by_column("a中b", 2, 1), "");
        // b immediately adjacent to window [3,4) is kept whole
        assert_eq!(slice_by_column("a中b", 3, 1), "b");
    }

    #[test]
    fn test_slice_by_column_full_line_context() {
        // Covers the typical Input windowing scenario: wide characters fully preserved in the middle of the window
        assert_eq!(slice_by_column("你好世界", 2, 4), "好世");
    }

    // ── normalize_tabs ──

    #[test]
    fn test_normalize_tabs_expands_to_three_spaces() {
        assert_eq!(normalize_tabs("\t"), "   ");
        assert_eq!(normalize_tabs("a\tb"), "a   b");
    }

    #[test]
    fn test_normalize_tabs_no_tabs_unchanged() {
        assert_eq!(normalize_tabs("no tabs here"), "no tabs here");
        assert_eq!(normalize_tabs(""), "");
    }
}
