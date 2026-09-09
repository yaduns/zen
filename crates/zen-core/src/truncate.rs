// REQ-AGENT-06 (pi-231, pi-354): Shared truncation utility—limits tool output using dual line-count/byte-count caps.
// Original TS counterpart: reference/pi-original/packages/agent/src/harness/utils/truncate.ts (pi-231)
// and reference/pi-original/packages/coding-agent/src/core/tools/truncate.ts (pi-354,
// Export-equivalent copy—zen's single shared module covers both; T-AP-04-8 recommends marking it Drop and merging it into this task).
// Placement exception (top-level rather than env/): see the placement-exception note under "REQ prefix → crate mapping" in AGENTS.md and
// tasks/T-AG-06-3-implemented.md 6.1—Of the 5 consumers, only 1 belongs to the env domain and is structurally equivalent to process.rs.

use std::borrow::Cow;

/// Default line-count limit (2000 lines)—pi `DEFAULT_MAX_LINES`.
pub const DEFAULT_MAX_LINES: usize = 2000;

/// Default byte limit (50KB)—pi `DEFAULT_MAX_BYTES`.
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;

/// Character limit per line for grep matches—pi `GREP_MAX_LINE_LENGTH`.
/// Consumer: grep tool (T-AP-04-5; `grep_files` calls `truncate_line` for each match/context line).
pub const GREP_MAX_LINE_LENGTH: usize = 500;

/// Reason truncation was triggered (pi `truncatedBy: "lines" | "bytes"`; no truncation is represented by the outer `Option`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationKind {
    /// The output line limit is exceeded before the byte limit.
    Lines,
    /// The output byte limit is exceeded before the line limit.
    Bytes,
}

/// Upper-limit parameter pair for a single truncation call (pi `TruncationOptions` optional fields + `??` defaults
/// In Rust, this resolves to the `Default` impl: `truncate_head(content, TruncationOptions::default())`
/// Equivalent to pi's `truncateHead(content)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncationOptions {
    /// Maximum line count; defaults to [`DEFAULT_MAX_LINES`].
    pub max_lines: usize,
    /// Maximum byte count; defaults to [`DEFAULT_MAX_BYTES`].
    pub max_bytes: usize,
}

impl Default for TruncationOptions {
    fn default() -> Self {
        Self {
            max_lines: DEFAULT_MAX_LINES,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

/// Complete result report for a single head/tail truncation call (field-by-field aligned with pi `TruncationResult`).
///
/// The read/bash tools map these fields into the details metadata (with the same structure as pi); `output_lines`/`output_bytes`
/// Although it can be derived from `content`, it is already computed during the algorithm, so the field is retained to prevent consumers from rescanning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TruncationResult {
    /// Content after truncation; the original content when not truncated (including the trailing newline).
    pub content: String,
    /// Whether truncation occurred.
    pub truncated: bool,
    /// The limit that was reached; `None` when not truncated.
    pub truncated_by: Option<TruncationKind>,
    /// Total number of lines in the original content (an empty string counts as 0 lines, and a trailing newline does not add a line).
    pub total_lines: usize,
    /// Total size of the original content in bytes (UTF-8).
    pub total_bytes: usize,
    /// Number of lines output after truncation.
    pub output_lines: usize,
    /// Number of bytes output after truncation (= `content.len()`).
    pub output_bytes: usize,
    /// Whether the last line was partially truncated (`true` only when tail truncation cuts a single over-limit line).
    pub last_line_partial: bool,
    /// Whether the first line exceeds the byte limit and cannot be output at all (head truncation only; the pi read tool uses this to generate a prompt).
    pub first_line_exceeds_limit: bool,
    /// Effective line limit for this operation (echo).
    pub max_lines: usize,
    /// Effective byte limit for this operation (echo).
    pub max_bytes: usize,
}

/// [`truncate_line`] result—the truncated single-line text and whether truncation occurred
/// (pi returns an inline object `{text, wasTruncated}`; Rust requires a named type).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TruncatedLine {
    /// The truncated single-line text (the original line if not truncated).
    pub text: String,
    /// Whether truncation occurred.
    pub was_truncated: bool,
}

/// When viewing tool output from the beginning (read/ls/find-style results where the beginning matters), cap output that exceeds the limit—
/// Count from the beginning and discard anything that does not fit, preserving the first `max_lines` lines while keeping the total size within `max_bytes`,
/// The line and byte limits are independent; whichever is reached first applies.
/// Never return partial lines: include only complete lines; if the first line alone exceeds the byte limit, return empty content
/// and set [`TruncationResult::first_line_exceeds_limit`] (the pi read tool uses it to generate a "file exceeds the limit" message).
///
/// # Parameters
/// - `content`: raw text; returned as-is when within limits (`truncated = false`, trailing newline preserved).
/// - `options`: line/byte limits; defaults to [`DEFAULT_MAX_LINES`]/[`DEFAULT_MAX_BYTES`].
///
/// # Line counting semantics
/// Frames by `'\n'`; a trailing newline does not count as an extra line, and an empty string counts as 0 lines (pi-354 `splitLinesForCounting`
/// semantics, not pi-231 bare `split` semantics; see tasks/T-AG-06-3-implemented.md for the decision).
/// Do not use `str::lines()` (it strips trailing `\r`, unlike pi, which frames only on `\n`).
pub fn truncate_head(content: &str, options: TruncationOptions) -> TruncationResult {
    let total_bytes = content.len();
    let lines = split_lines_for_counting(content);
    let total_lines = lines.len();

    // Both line count and byte count within limits → return as-is
    if total_lines <= options.max_lines && total_bytes <= options.max_bytes {
        return unchanged_result(content, total_lines, total_bytes, options);
    }

    // The first line alone exceeds the byte limit → empty content + first_line_exceeds_limit (pi read hint semantics)
    if lines[0].len() > options.max_bytes {
        return TruncationResult {
            content: String::new(),
            truncated: true,
            truncated_by: Some(TruncationKind::Bytes),
            total_lines,
            total_bytes,
            output_lines: 0,
            output_bytes: 0,
            last_line_partial: false,
            first_line_exceeds_limit: true,
            max_lines: options.max_lines,
            max_bytes: options.max_bytes,
        };
    }

    // Collect complete lines from the start (byte overhead = line bytes + 1 byte for the newline on each non-first line)
    let mut output: Vec<&str> = Vec::new();
    let mut output_bytes: usize = 0;
    let mut truncated_by = TruncationKind::Lines;

    for (i, &line) in lines.iter().enumerate().take(options.max_lines) {
        let line_cost = line.len() + usize::from(i > 0);
        if output_bytes + line_cost > options.max_bytes {
            truncated_by = TruncationKind::Bytes;
            break;
        }
        output.push(line);
        output_bytes += line_cost;
    }

    // The line-count limit is reached first (the loop naturally fills maxLines without exceeding the byte limit; matches pi's finalization check)
    if output.len() >= options.max_lines && output_bytes <= options.max_bytes {
        truncated_by = TruncationKind::Lines;
    }

    let content_out = output.join("\n");
    TruncationResult {
        output_bytes: content_out.len(),
        content: content_out,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines: output.len(),
        last_line_partial: false,
        first_line_exceeds_limit: false,
        max_lines: options.max_lines,
        max_bytes: options.max_bytes,
    }
}

/// bash output capping—errors and the final result are at the end; "count from the end and truncate what does not fit",
/// Keep the last `max_lines` lines with total bytes not exceeding `max_bytes`; the first limit reached takes precedence.
/// Edge case (the only place where pi permits a partial line): when not even one line fits and the last line itself exceeds the byte limit,
/// take up to `max_bytes` bytes from the end of that line (aligning multibyte characters to character boundaries without splitting a character),
/// and set [`TruncationResult::last_line_partial`]—ensuring non-empty input never returns empty content.
///
/// # Parameters
/// - `content`: raw text; returned as-is when within limits (`truncated = false`, trailing newline preserved).
/// - `options`: line/byte limits; defaults to [`DEFAULT_MAX_LINES`]/[`DEFAULT_MAX_BYTES`].
///
/// # Line counting semantics
/// Same as [`truncate_head`] (pi-354 semantics: a trailing newline does not count as an extra line, and an empty string counts as 0 lines).
pub fn truncate_tail(content: &str, options: TruncationOptions) -> TruncationResult {
    let total_bytes = content.len();
    let lines = split_lines_for_counting(content);
    let total_lines = lines.len();

    // Both line count and byte count within limits → return as-is
    if total_lines <= options.max_lines && total_bytes <= options.max_bytes {
        return unchanged_result(content, total_lines, total_bytes, options);
    }

    // Accumulate lines backward from the tail (push in reverse order, then reverse at the end to restore forward order)
    let mut output: Vec<Cow<'_, str>> = Vec::new();
    let mut output_bytes: usize = 0;
    let mut truncated_by = TruncationKind::Lines;
    let mut last_line_partial = false;

    for &line in lines.iter().rev() {
        if output.len() >= options.max_lines {
            break; // Line limit reached
        }
        // Byte overhead = line bytes + (1 byte for the newline when the accumulated content is nonempty)
        let line_cost = line.len() + usize::from(!output.is_empty());
        if output_bytes + line_cost > options.max_bytes {
            truncated_by = TruncationKind::Bytes;
            // No lines have been accumulated yet → the current line (the last line of the original content) itself exceeds the limit:
            // Take a trailing portion of the line, ensuring it is nonempty (the only place where pi permits a partial line)
            if output.is_empty() {
                let partial = truncate_bytes_from_end(line, options.max_bytes);
                output_bytes = partial.len();
                output.push(Cow::Owned(partial));
                last_line_partial = true;
            }
            break;
        }
        output_bytes += line_cost;
        output.push(Cow::Borrowed(line));
    }
    output.reverse(); // Convert the reverse-order push result to forward order

    // The line-count limit is reached first (aligned with pi's finalization check; when max_lines ≤ 1 and the final line exceeds the limit)
    // pi relabels Bytes as Lines—this behavior is faithfully preserved here
    if output.len() >= options.max_lines && output_bytes <= options.max_bytes {
        truncated_by = TruncationKind::Lines;
    }

    let content_out = output.join("\n");
    let output_lines = output.len();
    TruncationResult {
        output_bytes: content_out.len(),
        content: content_out,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines,
        last_line_partial,
        first_line_exceeds_limit: false,
        max_lines: options.max_lines,
        max_bytes: options.max_bytes,
    }
}

/// Single-line trimming before displaying grep-matched lines—"take the visible window when a single line is too wide":
/// More than `max_chars` characters (Unicode scalar values; equivalent to pi's UTF-16 `length` case)
/// Truncate to `max_chars` and append the `... [truncated]` suffix, allowing the caller to annotate details accordingly.
/// Consumer: grep tool (pi grep.ts calls it for each matching line; T-AP-04-5 `grep_files` is already integrated).
///
/// # Parameters
/// - `line`: Single-line text (excluding the newline).
/// - `max_chars`: character limit; callers explicitly pass [`GREP_MAX_LINE_LENGTH`] (Rust has no default parameters).
pub fn truncate_line(line: &str, max_chars: usize) -> TruncatedLine {
    if line.chars().count() <= max_chars {
        return TruncatedLine {
            text: line.to_string(),
            was_truncated: false,
        };
    }
    TruncatedLine {
        text: format!(
            "{}... [truncated]",
            line.chars().take(max_chars).collect::<String>()
        ),
        was_truncated: true,
    }
}

/// Convert byte counts from truncation stats into human-readable sizes (for tool details/prompt text):
/// `< 1024` → `"NB"`; `< 1MB` → `"N.NKB"`; otherwise `"N.NMB"` (matching pi's three-tier `formatSize` format).
///
/// Current consumers: truncation messages for find (T-AP-04-4), grep (T-AP-04-5), ls (T-AP-04-6);
/// The read/bash tools can also use this to align their outputSize text with pi.
pub fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Map [`TruncationResult`] to pi camelCase JSON in tool details (aligned with pi `TruncationResult`
/// interface field names; `content` is intentionally omitted—the body is already output as the tool message, avoiding up to 50KB of text being stored twice in details,
/// session JSONL persistence would amplify disk usage).
///
/// Consumers: find tool (`details.truncation`) and grep tool (`details.truncation`); this task (T-AP-04-5)
/// Move the original private implementation from find.rs up to a shared view (it naturally belongs with the shared truncation types).
pub fn truncation_to_json(t: &TruncationResult) -> serde_json::Value {
    let truncated_by = match t.truncated_by {
        Some(TruncationKind::Lines) => serde_json::Value::String("lines".into()),
        Some(TruncationKind::Bytes) => serde_json::Value::String("bytes".into()),
        None => serde_json::Value::Null,
    };
    serde_json::json!({
        "truncated": t.truncated,
        "truncatedBy": truncated_by,
        "totalLines": t.total_lines,
        "totalBytes": t.total_bytes,
        "outputLines": t.output_lines,
        "outputBytes": t.output_bytes,
        "lastLinePartial": t.last_line_partial,
        "firstLineExceedsLimit": t.first_line_exceeds_limit,
        "maxLines": t.max_lines,
        "maxBytes": t.max_bytes,
    })
}

/// Frame on `'\n'` and preprocess for line counting (matching pi-354 `splitLinesForCounting`):
/// Empty string → 0 lines; a trailing empty line produced by the final newline is not counted (`"a\n"` → `["a"]`, `"\n"` → `[""]`).
/// Do not use `str::lines()`—it strips trailing `\r`, unlike pi, which frames only on `\n`.
fn split_lines_for_counting(content: &str) -> Vec<&str> {
    if content.is_empty() {
        return Vec::new();
    }
    let mut lines: Vec<&str> = content.split('\n').collect();
    if content.ends_with('\n') {
        lines.pop();
    }
    lines
}

/// Take up to `max_bytes` bytes from the end of `s`, with the starting point aligned to a character boundary
/// (Rust form of pi `truncateStringToBytesFromEnd`, based on pi-354's byte-boundary scanning approach):
/// Return the original string if the whole string fits within the limit; otherwise, scan forward from `len - max_bytes`, skipping UTF-8 continuation bytes
/// Cut at a character boundary without producing a partial character.
///
/// Two intentional differences from pi (both are existing contracts for zen consumers; see the original `tail_bytes_utf8`):
/// - `max_bytes == 0` → empty string (same as pi);
/// - Empty after boundary scanning (the final character itself exceeds `max_bytes`, e.g., a 4-byte character with a 2-byte limit) →
///   Return the final character (allowing a slight overage), preserving the “non-empty input never returns empty content” guarantee;
///   pi returns an empty string in this degenerate case.
fn truncate_bytes_from_end(s: &str, max_bytes: usize) -> String {
    if max_bytes == 0 {
        return String::new();
    }
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut start = s.len() - max_bytes;
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    let tail = &s[start..];
    if tail.is_empty() {
        // Fallback: the final character itself does not fit → return that character, ensuring non-empty (allowing a slight overage)
        return s
            .chars()
            .next_back()
            .map(|c| c.to_string())
            .unwrap_or_default();
    }
    tail.to_string()
}

/// Consistent result when not truncated (`content` is preserved as-is, and stats are backfilled).
fn unchanged_result(
    content: &str,
    total_lines: usize,
    total_bytes: usize,
    options: TruncationOptions,
) -> TruncationResult {
    TruncationResult {
        content: content.to_string(),
        truncated: false,
        truncated_by: None,
        total_lines,
        total_bytes,
        output_lines: total_lines,
        output_bytes: total_bytes,
        last_line_partial: false,
        first_line_exceeds_limit: false,
        max_lines: options.max_lines,
        max_bytes: options.max_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── format_size ──────────────────────────────────────────────

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(0), "0B");
        assert_eq!(format_size(512), "512B");
        assert_eq!(format_size(1023), "1023B");
    }

    #[test]
    fn test_format_size_kb() {
        assert_eq!(format_size(1024), "1.0KB");
        assert_eq!(format_size(51200), "50.0KB");
        assert_eq!(format_size(1536), "1.5KB");
    }

    #[test]
    fn test_format_size_mb() {
        assert_eq!(format_size(1024 * 1024), "1.0MB");
        assert_eq!(format_size(1572864), "1.5MB");
    }

    // ── truncation_to_json ────────────────────────────────────────

    #[test]
    fn test_truncation_to_json_camel_case_keys_and_omits_content() {
        let r = truncate_head(
            "a\nb\nc\n",
            TruncationOptions {
                max_lines: 1,
                ..Default::default()
            },
        );
        let json = truncation_to_json(&r);
        assert_eq!(json["truncated"], true);
        assert_eq!(json["truncatedBy"], "lines");
        assert_eq!(json["maxLines"], 1);
        assert_eq!(json["maxBytes"], DEFAULT_MAX_BYTES);
        // content is intentionally omitted (to prevent duplicate storage in JSONL); all other statistics fields are present
        assert!(json.get("content").is_none());
        assert!(json.get("totalLines").is_some());
        assert!(json.get("totalBytes").is_some());
        assert!(json.get("outputLines").is_some());
        assert!(json.get("outputBytes").is_some());
        assert!(json.get("lastLinePartial").is_some());
        assert!(json.get("firstLineExceedsLimit").is_some());
    }

    #[test]
    fn test_truncation_to_json_untrucated_uses_null_truncated_by() {
        let r = truncate_head("hello\n", TruncationOptions::default());
        let json = truncation_to_json(&r);
        assert_eq!(json["truncated"], false);
        assert_eq!(json["truncatedBy"], serde_json::Value::Null);
    }

    // ── truncate_line ────────────────────────────────────────────

    #[test]
    fn test_truncate_line_within_limit() {
        let r = truncate_line("hello", 10);
        assert_eq!(r.text, "hello");
        assert!(!r.was_truncated);
    }

    #[test]
    fn test_truncate_line_exact_boundary() {
        let r = truncate_line("abcde", 5);
        assert_eq!(r.text, "abcde");
        assert!(!r.was_truncated);
    }

    #[test]
    fn test_truncate_line_truncates_with_suffix() {
        let line = "a".repeat(600);
        let r = truncate_line(&line, GREP_MAX_LINE_LENGTH);
        assert!(r.was_truncated);
        assert_eq!(r.text.len(), GREP_MAX_LINE_LENGTH + "... [truncated]".len());
        assert!(r.text.starts_with(&"a".repeat(GREP_MAX_LINE_LENGTH)));
        assert!(r.text.ends_with("... [truncated]"));
    }

    #[test]
    fn test_truncate_line_counts_chars_not_bytes() {
        // Each Chinese character occupies 3 bytes but counts as 1 character
        let r = truncate_line("中中中中中", 3);
        assert!(r.was_truncated);
        assert_eq!(r.text, "中中中... [truncated]");
    }

    // ── truncate_head ────────────────────────────────────────────

    #[test]
    fn test_head_within_limits_unchanged() {
        let r = truncate_head("a\nb\nc\n", TruncationOptions::default());
        assert!(!r.truncated);
        assert_eq!(r.truncated_by, None);
        // Return the content unchanged when not truncated (including the trailing newline)
        assert_eq!(r.content, "a\nb\nc\n");
        assert_eq!(r.total_lines, 3);
        assert_eq!(r.output_lines, 3);
        assert_eq!(r.total_bytes, 6);
        assert_eq!(r.output_bytes, 6);
        assert_eq!(r.max_lines, DEFAULT_MAX_LINES);
        assert_eq!(r.max_bytes, DEFAULT_MAX_BYTES);
    }

    #[test]
    fn test_head_empty_content() {
        let r = truncate_head("", TruncationOptions::default());
        assert!(!r.truncated);
        assert_eq!(r.content, "");
        // pi-354 Semantics: an empty string counts as 0 lines
        assert_eq!(r.total_lines, 0);
        assert_eq!(r.output_lines, 0);
    }

    #[test]
    fn test_head_trailing_newline_counts_no_extra_line() {
        // "a\nb\n" → ["a", "b"], not 3 lines
        let r = truncate_head("a\nb\n", TruncationOptions::default());
        assert_eq!(r.total_lines, 2);
    }

    #[test]
    fn test_head_truncated_by_lines() {
        let r = truncate_head(
            "a\nb\nc\nd\ne",
            TruncationOptions {
                max_lines: 2,
                max_bytes: 1024,
            },
        );
        assert!(r.truncated);
        assert_eq!(r.truncated_by, Some(TruncationKind::Lines));
        assert_eq!(r.content, "a\nb");
        assert_eq!(r.total_lines, 5);
        assert_eq!(r.output_lines, 2);
    }

    #[test]
    fn test_head_exactly_at_line_limit() {
        let r = truncate_head(
            "line1\nline2",
            TruncationOptions {
                max_lines: 2,
                max_bytes: 1024,
            },
        );
        assert!(!r.truncated);
        assert_eq!(r.content, "line1\nline2");
    }

    #[test]
    fn test_head_truncated_by_bytes_mid_collection() {
        // 10 bytes per line; max_bytes=25 → line 3 plus the newline exceeds the limit, so retain the first 2 complete lines
        let r = truncate_head(
            "aaaaaaaaaa\nbbbbbbbbbb\ncccccccccc\ndddddddddd",
            TruncationOptions {
                max_lines: 100,
                max_bytes: 25,
            },
        );
        assert!(r.truncated);
        assert_eq!(r.truncated_by, Some(TruncationKind::Bytes));
        // Byte overhead: 10 + (10+1) = 21 ≤ 25; adding the third line adds 11 → 32 > 25
        assert_eq!(r.content, "aaaaaaaaaa\nbbbbbbbbbb");
        assert_eq!(r.output_bytes, 21);
        assert_eq!(r.output_lines, 2);
    }

    #[test]
    fn test_head_first_line_exceeds_byte_limit() {
        let r = truncate_head(
            "aaaaaaaaaaaaaaaaaaaa\nb\nc",
            TruncationOptions {
                max_lines: 100,
                max_bytes: 10,
            },
        );
        assert!(r.truncated);
        assert_eq!(r.truncated_by, Some(TruncationKind::Bytes));
        assert_eq!(r.content, "");
        assert!(r.first_line_exceeds_limit);
        assert_eq!(r.output_lines, 0);
        assert!(!r.last_line_partial);
    }

    #[test]
    fn test_head_never_splits_multibyte_chars() {
        // 2 Chinese characters per line (6 bytes); max_bytes=14 → retain the first 2 lines in full (6 + 7 = 13),
        // The 3rd line does not fit and is discarded entirely, without splitting characters
        let content = "中中\n中中\n中中";
        let r = truncate_head(
            content,
            TruncationOptions {
                max_lines: 100,
                max_bytes: 14,
            },
        );
        assert!(r.truncated);
        assert_eq!(r.truncated_by, Some(TruncationKind::Bytes));
        assert_eq!(r.content, "中中\n中中");
        assert_eq!(r.output_bytes, 13);
    }

    #[test]
    fn test_head_max_lines_zero_degenerate() {
        let r = truncate_head(
            "a\nb",
            TruncationOptions {
                max_lines: 0,
                max_bytes: 100,
            },
        );
        assert!(r.truncated);
        assert_eq!(r.content, "");
        assert_eq!(r.truncated_by, Some(TruncationKind::Lines));
    }

    #[test]
    fn test_head_custom_options_echoed() {
        let r = truncate_head("x", TruncationOptions::default());
        assert_eq!(r.max_lines, 2000);
        assert_eq!(r.max_bytes, 50 * 1024);
    }

    // ── truncate_tail ────────────────────────────────────────────

    #[test]
    fn test_tail_within_limits_unchanged() {
        let r = truncate_tail("a\nb\nc\n", TruncationOptions::default());
        assert!(!r.truncated);
        assert_eq!(r.content, "a\nb\nc\n");
        assert_eq!(r.truncated_by, None);
    }

    #[test]
    fn test_tail_empty_input() {
        let r = truncate_tail("", TruncationOptions::default());
        assert!(!r.truncated);
        assert_eq!(r.content, "");
        assert_eq!(r.total_lines, 0);
        assert_eq!(r.output_lines, 0);
    }

    #[test]
    fn test_tail_trailing_newline_within_limits() {
        // Strip the trailing blank line produced by the final newline before counting; doesn't affect the "not truncated" determination
        let r = truncate_tail("a\nb\n", TruncationOptions::default());
        assert!(!r.truncated);
        assert_eq!(r.content, "a\nb\n");
    }

    #[test]
    fn test_tail_keeps_last_lines() {
        let content = (1..=5)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let r = truncate_tail(
            &content,
            TruncationOptions {
                max_lines: 3,
                max_bytes: 10_000,
            },
        );
        assert!(r.truncated);
        assert_eq!(r.truncated_by, Some(TruncationKind::Lines));
        assert_eq!(r.content, "line3\nline4\nline5");
        assert_eq!(r.total_lines, 5);
        assert_eq!(r.output_lines, 3);
    }

    #[test]
    fn test_tail_byte_limit_aligns_to_line_start() {
        // 4 characters per line + newline; max_bytes=10 → retain complete lines from the end and align to the line start
        let r = truncate_tail(
            "aaaa\nbbbb\ncccc\ndddd\n",
            TruncationOptions {
                max_lines: 100,
                max_bytes: 10,
            },
        );
        assert!(r.truncated);
        assert_eq!(r.truncated_by, Some(TruncationKind::Bytes));
        // Tail accumulation: dddd(4) → cccc(4+1=5, cumulative 9) → bbbb requires 5, cumulative 14 > 10 → stop
        assert_eq!(r.content, "cccc\ndddd");
        assert_eq!(r.output_bytes, 9);
    }

    #[test]
    fn test_tail_single_line_partial_tail() {
        // A single line exceeds the byte limit → return a trailing partial segment of that line (the only place pi allows a partial line)
        let r = truncate_tail(
            "abcdefghij",
            TruncationOptions {
                max_lines: 10,
                max_bytes: 4,
            },
        );
        assert!(r.truncated);
        assert_eq!(r.content, "ghij");
        assert!(r.last_line_partial);
        assert_eq!(r.output_lines, 1);
    }

    #[test]
    fn test_tail_never_empty_for_nonempty_input() {
        // With max_bytes=1 a non-empty tail is still returned (fallback guarantee)
        let r = truncate_tail(
            "abcdefghij",
            TruncationOptions {
                max_lines: 10,
                max_bytes: 1,
            },
        );
        assert!(r.truncated);
        assert!(!r.content.is_empty());
        assert_eq!(r.content, "j");
        assert!(r.last_line_partial);
    }

    #[test]
    fn test_tail_large_last_line_keeps_partial() {
        // The final line is 10000 bytes and exceeds max_bytes=100 → retain the last 100 bytes of the final line
        let input = format!("short\n{}", "x".repeat(10000));
        let r = truncate_tail(
            &input,
            TruncationOptions {
                max_lines: 100,
                max_bytes: 100,
            },
        );
        assert!(r.truncated);
        assert!(!r.content.is_empty());
        assert_eq!(r.content.len(), 100);
        assert!(r.content.chars().all(|c| c == 'x'));
        assert_eq!(r.total_lines, 2);
        assert_eq!(r.output_lines, 1);
        assert!(r.last_line_partial);
    }

    #[test]
    fn test_tail_lines_and_bytes_both_exceed_bytes_wins() {
        // 10 lines "line0".."line9", max_lines=5, max_bytes=20:
        // pi's line-by-line accumulation algorithm collects from the tail—line9(5) → line8(6, 11) → line7(6, 17) →
        // line6 needs 6, making the total 23 > 20 → the byte limit is reached first, retaining 3 lines, truncated_by = Bytes.
        // (The legacy bash.rs two-stage implementation trims lines before bytes and reports Lines; the shared pi algorithm is the behavioral source of truth.)
        let input = (0..10)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let r = truncate_tail(
            &input,
            TruncationOptions {
                max_lines: 5,
                max_bytes: 20,
            },
        );
        assert!(r.truncated);
        assert_eq!(r.truncated_by, Some(TruncationKind::Bytes));
        assert_eq!(r.content, "line7\nline8\nline9");
        assert_eq!(r.output_lines, 3);
        assert_eq!(r.total_lines, 10);
    }

    #[test]
    fn test_tail_max_lines_one_giant_line_pi_relabel_quirk() {
        // Edge case in pi's finalization logic: when max_lines=1 and the last line exceeds the limit,
        // After accepting the partial line, output.len()(1) >= max_lines(1) and the byte count is within the limit → Bytes is relabeled as Lines.
        // Preserve pi's behavior faithfully (consumers currently do not read truncated_by, so there is no practical impact).
        let r = truncate_tail(
            &"x".repeat(100),
            TruncationOptions {
                max_lines: 1,
                max_bytes: 50,
            },
        );
        assert!(r.truncated);
        assert_eq!(r.truncated_by, Some(TruncationKind::Lines));
        assert_eq!(r.content.len(), 50);
        assert!(r.last_line_partial);
    }

    // ── truncate_bytes_from_end (private helper; indirectly covered via truncate_tail + directly tested)──

    #[test]
    fn test_tail_bytes_multibyte_safe() {
        // é takes 2 bytes; max_bytes=5 exactly fits "éllo", without splitting a multi-byte character
        let r = truncate_tail(
            "héllo",
            TruncationOptions {
                max_lines: 10,
                max_bytes: 5,
            },
        );
        assert_eq!(r.content, "éllo");
        assert!(r.last_line_partial);
    }

    #[test]
    fn test_tail_bytes_boundary_scan_matches_char_collection() {
        // "abX" (5 bytes) max_bytes=3 → the boundary scan lands on the lead byte of the CJK char 'X' → "X"
        let r = truncate_tail(
            "ab中",
            TruncationOptions {
                max_lines: 10,
                max_bytes: 3,
            },
        );
        assert_eq!(r.content, "中");
    }

    #[test]
    fn test_tail_zero_byte_limit_returns_empty() {
        // max_bytes=0 is an explicit convention: return empty (covers the never-empty fallback and aligns with the original tail_bytes_utf8)
        let r = truncate_tail(
            "abc",
            TruncationOptions {
                max_lines: 10,
                max_bytes: 0,
            },
        );
        assert!(r.truncated);
        assert_eq!(r.content, "");
    }

    #[test]
    fn test_tail_single_char_exceeds_limit_fallback() {
        // 🦀 occupies 4 bytes and does not fit in max_bytes=2 → fall back to returning the last character (slight over-limit allowed,
        // zen extension: in this degraded scenario, pi returns an empty string; see truncate_bytes_from_end docs)
        let r = truncate_tail(
            "🦀",
            TruncationOptions {
                max_lines: 10,
                max_bytes: 2,
            },
        );
        assert!(r.truncated);
        assert_eq!(r.content, "🦀");
        assert!(r.last_line_partial);
    }
}
