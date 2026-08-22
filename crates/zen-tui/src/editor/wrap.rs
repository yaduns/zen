// REQ-TUI-01 (pi-413): editor component — pure word-wrap functions (word_wrap_line / make_chunk / is_cjk).
// No Editor state dependency; independently testable; promote to pub when needed for T-TU-02-1 (markdown wrapping).
// Consumers: render::layout_text, cursor::build_visual_line_map (via `use super::wrap::...`).
use super::TextChunk;
use unicode_segmentation::UnicodeSegmentation;

/// Wrap a single line at word boundaries: within maxWidth prefer word boundaries, adjacent CJK may break, over-wide atomic segments become their own chunks.
/// **A fragment is a contiguous slice of the source text** (text == line[start_index..end_index], byte offset); whitespace belongs to the end of the previous fragment,
/// Never rebuild text — ensure displayed text matches the buffer character-for-character and cursor column mapping is linear
/// Consumers: render::layout_text, cursor::build_visual_line_map; no external consumers, visibility pub(super).
/// Promote to pub later if needed (T-TU-02-1 markdown wrapping).
pub(super) fn word_wrap_line(line: &str, max_width: usize) -> Vec<TextChunk> {
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
    if crate::screen::visible_width(line) <= max_width {
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
        let g_width = crate::screen::visible_width(grapheme);
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

        // ── A single atomic segment wider than maxWidth (extremely narrow terminal: max_width=1 and grapheme column width=2) ──
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

    // ── word_wrap_line ──

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
        assert!(chunks.len() >= 1);
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

    // ── is_cjk ──

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
}
