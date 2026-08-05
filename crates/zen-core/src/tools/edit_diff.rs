use similar::{ChangeTag, TextDiff};
use std::borrow::Cow;
use std::path::Path;

/// Detect the line-ending type of the text.
pub fn detect_line_ending(content: &str) -> &str {
    let crlf = content.find("\r\n");
    let lf = content.find('\n');
    match (crlf, lf) {
        (Some(crlf_idx), Some(lf_idx)) => {
            if crlf_idx < lf_idx {
                "\r\n"
            } else {
                "\n"
            }
        }
        _ => "\n",
    }
}

/// Normalize all line endings to LF.
pub fn normalize_to_lf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

/// Restore LF line endings to the specified type.
pub fn restore_line_endings(text: &str, ending: &str) -> String {
    if ending == "\r\n" {
        text.replace('\n', "\r\n")
    } else {
        text.to_string()
    }
}

/// Normalize text for fuzzy matching.
pub fn normalize_for_fuzzy_match(text: &str) -> Cow<'_, str> {
    use unicode_normalization::UnicodeNormalization;
    let nfkc: String = text.nfc().collect();
    let trimmed: String = nfkc
        .lines()
        .map(|line| line.trim_end())
        .collect::<Vec<_>>()
        .join("\n");
    let s = trimmed
        .chars()
        .map(|c| match c {
            '\u{2018}'..='\u{201B}' => '\'',
            '\u{201C}'..='\u{201F}' => '"',
            '\u{2010}'..='\u{2015}' | '\u{2212}' => '-',
            '\u{00A0}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
            c if ('\u{2002}'..='\u{200A}').contains(&c) => ' ',
            _ => c,
        })
        .collect::<String>();
    if s == text {
        Cow::Borrowed(text)
    } else {
        Cow::Owned(s)
    }
}

/// Fuzzy matching result.
#[derive(Debug, Clone)]
pub(crate) struct FuzzyMatchResult {
    pub index: usize,
    pub match_length: usize,
    pub used_fuzzy_match: bool,
}

/// Find the target text in the text: try exact matching first, then fuzzy matching if it fails.
pub(crate) fn fuzzy_find_text(content: &str, old_text: &str) -> Option<FuzzyMatchResult> {
    if old_text.is_empty() {
        return None;
    }
    if let Some(idx) = content.find(old_text) {
        return Some(FuzzyMatchResult {
            index: idx,
            match_length: old_text.len(),
            used_fuzzy_match: false,
        });
    }
    let fuzzy_content = normalize_for_fuzzy_match(content);
    let fuzzy_old = normalize_for_fuzzy_match(old_text);
    let fuzzy_idx = fuzzy_content.find(&*fuzzy_old)?;
    Some(FuzzyMatchResult {
        index: fuzzy_idx,
        match_length: fuzzy_old.len(),
        used_fuzzy_match: true,
    })
}

/// Strip the UTF-8 BOM.
pub fn strip_bom(content: &str) -> (String, &str) {
    if let Some(rest) = content.strip_prefix('\u{FEFF}') {
        ("\u{FEFF}".to_string(), rest)
    } else {
        (String::new(), content)
    }
}

/// A single edit operation.
#[derive(Debug, Clone)]
pub struct Edit {
    pub old_text: String,
    pub new_text: String,
}

/// Result of applying an edit operation.
#[derive(Debug, Clone)]
pub struct AppliedEditsResult {
    pub base_content: String,
    pub new_content: String,
}

/// Details of the edit tool execution result.
#[derive(Debug, Clone)]
pub struct EditToolDetails {
    pub diff: String,
    pub patch: String,
    pub first_changed_line: Option<usize>,
}

/// Return value of the compute_edits_diff preview.
#[derive(Debug, Clone)]
pub struct EditDiffDetails {
    pub diff: String,
    pub first_changed_line: Option<usize>,
}

/// Error type for edit operations.
#[derive(Debug, thiserror::Error)]
pub enum EditError {
    #[error("Could not find the exact text in {path}.")]
    NotFound {
        path: String,
        edit_index: usize,
        total_edits: usize,
    },
    #[error("Found {occurrences} occurrences of the text in {path}.")]
    Duplicate {
        path: String,
        edit_index: usize,
        total_edits: usize,
        occurrences: usize,
    },
    #[error("oldText must not be empty in {path}.")]
    EmptyOldText {
        path: String,
        edit_index: usize,
        total_edits: usize,
    },
    #[error("No changes made to {path}.")]
    NoChange { path: String, total_edits: usize },
    #[error("edits[{edit_index_a}] and edits[{edit_index_b}] overlap in {path}.")]
    Overlap {
        path: String,
        edit_index_a: usize,
        edit_index_b: usize,
    },
    #[error("Could not edit file: {path}. {reason}")]
    FileAccess { path: String, reason: String },
    #[error("Failed to read file '{path}': {source}")]
    FileRead {
        path: String,
        source: std::io::Error,
    },
    #[error("Failed to write file '{path}': {source}")]
    FileWrite {
        path: String,
        source: std::io::Error,
    },
    #[error("Cannot preserve unchanged lines: {reason}")]
    CantPreserveLines { reason: String },
}

struct MatchedEdit {
    edit_index: usize,
    match_index: usize,
    match_length: usize,
    new_text: String,
}

#[derive(Clone)]
struct TextReplacement {
    match_index: usize,
    match_length: usize,
    new_text: String,
}
struct LineSpan {
    start: usize,
    end: usize,
}

fn count_occurrences(content: &str, old_text: &str) -> usize {
    let fuzzy_content = normalize_for_fuzzy_match(content);
    let fuzzy_old = normalize_for_fuzzy_match(old_text);
    if fuzzy_old.is_empty() {
        return 0;
    }
    fuzzy_content
        .as_ref()
        .match_indices(fuzzy_old.as_ref())
        .count()
}

fn split_lines_with_endings(content: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let bytes = content.as_bytes();
    for i in 0..bytes.len() {
        if bytes[i] == b'\n' {
            lines.push(&content[start..=i]);
            start = i + 1;
        }
    }
    if start < content.len() {
        lines.push(&content[start..]);
    }
    lines
}

fn get_line_spans(content: &str) -> Vec<LineSpan> {
    let mut offset = 0;
    split_lines_with_endings(content)
        .into_iter()
        .map(|line| {
            let span = LineSpan {
                start: offset,
                end: offset + line.len(),
            };
            offset = span.end;
            span
        })
        .collect()
}

fn get_replacement_line_range(lines: &[LineSpan], replacement: &TextReplacement) -> (usize, usize) {
    let rep_start = replacement.match_index;
    let rep_end = replacement.match_index + replacement.match_length;
    let mut start_line = lines.len();
    for (i, line) in lines.iter().enumerate() {
        if rep_start >= line.start && rep_start < line.end {
            start_line = i;
            break;
        }
    }
    let mut end_line = start_line;
    while end_line < lines.len() && lines[end_line].end < rep_end {
        end_line += 1;
    }
    if end_line >= lines.len() {
        end_line = lines.len() - 1;
    }
    (start_line, end_line + 1)
}

fn apply_replacements(content: &str, replacements: &[TextReplacement], offset: usize) -> String {
    let mut result = content.to_string();
    for r in replacements.iter().rev() {
        let mi = r.match_index - offset;
        result = format!(
            "{}{}{}",
            &result[..mi],
            r.new_text,
            &result[mi + r.match_length..]
        );
    }
    result
}

fn apply_replacements_preserving_unchanged_lines(
    original_content: &str,
    base_content: &str,
    replacements: &[TextReplacement],
) -> Result<String, EditError> {
    let original_lines = split_lines_with_endings(original_content);
    let base_lines = get_line_spans(base_content);
    if original_lines.len() != base_lines.len() {
        return Err(EditError::CantPreserveLines {
            reason: format!(
                "original has {} lines, base has {} lines",
                original_lines.len(),
                base_lines.len()
            ),
        });
    }
    let mut sorted: Vec<&TextReplacement> = replacements.iter().collect();
    sorted.sort_by_key(|r| r.match_index);
    let mut groups: Vec<(usize, usize, Vec<TextReplacement>)> = Vec::new();
    for r in &sorted {
        let range = get_replacement_line_range(&base_lines, r);
        if let Some(last) = groups.last_mut() {
            if range.0 < last.1 {
                last.1 = last.1.max(range.1);
                last.2.push((*r).clone());
                continue;
            }
        }
        groups.push((range.0, range.1, vec![(*r).clone()]));
    }
    let mut orig_idx = 0;
    let mut result = String::new();
    for (start_line, end_line, group_repls) in &groups {
        for line in &original_lines[orig_idx..*start_line] {
            result.push_str(line);
        }
        let gs = base_lines[*start_line].start;
        let ge = base_lines[*end_line - 1].end;
        result.push_str(&apply_replacements(&base_content[gs..ge], group_repls, gs));
        orig_idx = *end_line;
    }
    for line in &original_lines[orig_idx..] {
        result.push_str(line);
    }
    Ok(result)
}

/// Apply one or more edit operations to LF-normalized content.
pub fn apply_edits_to_normalized_content(
    normalized_content: &str,
    edits: &[Edit],
    path: &str,
) -> Result<AppliedEditsResult, EditError> {
    let normalized_edits: Vec<Edit> = edits
        .iter()
        .map(|e| Edit {
            old_text: normalize_to_lf(&e.old_text),
            new_text: normalize_to_lf(&e.new_text),
        })
        .collect();
    for (i, e) in normalized_edits.iter().enumerate() {
        if e.old_text.is_empty() {
            return Err(EditError::EmptyOldText {
                path: path.into(),
                edit_index: i,
                total_edits: normalized_edits.len(),
            });
        }
    }
    let initial_matches: Vec<Option<FuzzyMatchResult>> = normalized_edits
        .iter()
        .map(|e| fuzzy_find_text(normalized_content, &e.old_text))
        .collect();
    let used_fuzzy = initial_matches
        .iter()
        .any(|m| m.as_ref().map_or(false, |m| m.used_fuzzy_match));
    let repl_base: Cow<'_, str> = if used_fuzzy {
        normalize_for_fuzzy_match(normalized_content)
    } else {
        Cow::Borrowed(normalized_content)
    };
    let mut matched: Vec<MatchedEdit> = Vec::new();
    for (i, e) in normalized_edits.iter().enumerate() {
        let m = fuzzy_find_text(&repl_base, &e.old_text).ok_or(EditError::NotFound {
            path: path.into(),
            edit_index: i,
            total_edits: normalized_edits.len(),
        })?;
        let occ = count_occurrences(&repl_base, &e.old_text);
        if occ > 1 {
            return Err(EditError::Duplicate {
                path: path.into(),
                edit_index: i,
                total_edits: normalized_edits.len(),
                occurrences: occ,
            });
        }
        matched.push(MatchedEdit {
            edit_index: i,
            match_index: m.index,
            match_length: m.match_length,
            new_text: e.new_text.clone(),
        });
    }
    matched.sort_by_key(|m| m.match_index);
    for i in 1..matched.len() {
        let p = &matched[i - 1];
        let c = &matched[i];
        if p.match_index + p.match_length > c.match_index {
            return Err(EditError::Overlap {
                path: path.into(),
                edit_index_a: p.edit_index,
                edit_index_b: c.edit_index,
            });
        }
    }
    let repls: Vec<TextReplacement> = matched
        .iter()
        .map(|m| TextReplacement {
            match_index: m.match_index,
            match_length: m.match_length,
            new_text: m.new_text.clone(),
        })
        .collect();
    let base = normalized_content.to_string();
    let new = if used_fuzzy {
        apply_replacements_preserving_unchanged_lines(normalized_content, &repl_base, &repls)?
    } else {
        apply_replacements(&repl_base, &repls, 0)
    };
    if base == new {
        return Err(EditError::NoChange {
            path: path.into(),
            total_edits: normalized_edits.len(),
        });
    }
    Ok(AppliedEditsResult {
        base_content: base,
        new_content: new,
    })
}

/// Generate a diff string for display.
pub fn generate_diff_string(
    old_content: &str,
    new_content: &str,
    context_lines: usize,
) -> (String, Option<usize>) {
    let diff = TextDiff::from_lines(old_content, new_content);
    let changes: Vec<_> = diff.iter_all_changes().collect();
    let old_lines: Vec<&str> = old_content.lines().collect();
    let new_lines: Vec<&str> = new_content.lines().collect();
    let max_ln = old_lines.len().max(new_lines.len());
    let w = format!("{}", max_ln).len();
    let mut out: Vec<String> = Vec::new();
    let mut old_ln = 1;
    let mut new_ln = 1;
    let mut first_changed: Option<usize> = None;
    let mut i = 0;
    while i < changes.len() {
        let c = changes[i];
        if c.tag() == ChangeTag::Equal {
            let eq_cnt = changes[i..]
                .iter()
                .take_while(|c| c.tag() == ChangeTag::Equal)
                .count();
            let has_lead = i > 0
                && (changes[i - 1].tag() == ChangeTag::Delete
                    || changes[i - 1].tag() == ChangeTag::Insert);
            let has_trail = i + eq_cnt < changes.len()
                && (changes[i + eq_cnt].tag() == ChangeTag::Delete
                    || changes[i + eq_cnt].tag() == ChangeTag::Insert);
            if has_lead || has_trail {
                let raw: Vec<&str> = changes[i..i + eq_cnt]
                    .iter()
                    .map(|c| c.value().trim_end_matches('\n'))
                    .collect();
                if has_lead && has_trail {
                    if raw.len() <= context_lines * 2 {
                        for line in &raw {
                            out.push(format!(" {:>w$} {}", old_ln, line, w = w));
                            old_ln += 1;
                            new_ln += 1;
                        }
                    } else {
                        let lead = &raw[..context_lines.min(raw.len())];
                        let trail = &raw[raw.len().saturating_sub(context_lines)..];
                        let skipped = raw.len() - lead.len() - trail.len();
                        for line in lead {
                            out.push(format!(" {:>w$} {}", old_ln, line, w = w));
                            old_ln += 1;
                            new_ln += 1;
                        }
                        out.push(format!(" {:>w$} ...", "", w = w));
                        old_ln += skipped;
                        new_ln += skipped;
                        for line in trail {
                            out.push(format!(" {:>w$} {}", old_ln, line, w = w));
                            old_ln += 1;
                            new_ln += 1;
                        }
                    }
                } else if has_lead {
                    let shown = raw.len().min(context_lines);
                    for line in &raw[..shown] {
                        out.push(format!(" {:>w$} {}", old_ln, line, w = w));
                        old_ln += 1;
                        new_ln += 1;
                    }
                    let skipped = raw.len() - shown;
                    if skipped > 0 {
                        out.push(format!(" {:>w$} ...", "", w = w));
                        old_ln += skipped;
                        new_ln += skipped;
                    }
                } else {
                    let skipped = raw.len().saturating_sub(context_lines);
                    if skipped > 0 {
                        out.push(format!(" {:>w$} ...", "", w = w));
                        old_ln += skipped;
                        new_ln += skipped;
                    }
                    for line in &raw[skipped..] {
                        out.push(format!(" {:>w$} {}", old_ln, line, w = w));
                        old_ln += 1;
                        new_ln += 1;
                    }
                }
            } else {
                old_ln += eq_cnt;
                new_ln += eq_cnt;
            }
            i += eq_cnt;
        } else if c.tag() == ChangeTag::Delete {
            if first_changed.is_none() {
                first_changed = Some(new_ln);
            }
            let del_cnt = changes[i..]
                .iter()
                .take_while(|c| c.tag() == ChangeTag::Delete)
                .count();
            for j in 0..del_cnt {
                let line = changes[i + j].value().trim_end_matches('\n');
                out.push(format!("-{:>w$} {}", old_ln, line, w = w));
                old_ln += 1;
            }
            i += del_cnt;
        } else if c.tag() == ChangeTag::Insert {
            if first_changed.is_none() {
                first_changed = Some(new_ln);
            }
            let ins_cnt = changes[i..]
                .iter()
                .take_while(|c| c.tag() == ChangeTag::Insert)
                .count();
            for j in 0..ins_cnt {
                let line = changes[i + j].value().trim_end_matches('\n');
                out.push(format!("+{:>w$} {}", new_ln, line, w = w));
                new_ln += 1;
            }
            i += ins_cnt;
        }
    }
    (out.join("\n"), first_changed)
}

/// Generate a diff in standard unified patch format.
pub fn generate_unified_patch(
    _path: &str,
    old_content: &str,
    new_content: &str,
    _context_lines: usize,
) -> String {
    let text_diff = similar::TextDiff::from_lines(old_content, new_content);
    let diff = similar::udiff::UnifiedDiff::from_text_diff(&text_diff);
    diff.to_string()
}

/// Compute a diff preview for one or more edit operations.
pub async fn compute_edits_diff(
    path: &str,
    edits: &[Edit],
    cwd: &Path,
) -> Result<EditDiffDetails, EditError> {
    let abs = crate::tools::path_utils::resolve_to_cwd(path, cwd);
    let _meta = tokio::fs::metadata(&abs)
        .await
        .map_err(|e| EditError::FileAccess {
            path: path.into(),
            reason: format!("Cannot access file: {}", e),
        })?;
    let raw = tokio::fs::read_to_string(&abs)
        .await
        .map_err(|e| EditError::FileRead {
            path: path.into(),
            source: e,
        })?;
    let (_bom, content) = strip_bom(&raw);
    let normalized = normalize_to_lf(content);
    let result = apply_edits_to_normalized_content(&normalized, edits, path)?;
    let (diff, fcl) = generate_diff_string(&result.base_content, &result.new_content, 4);
    Ok(EditDiffDetails {
        diff,
        first_changed_line: fcl,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_line_ending_lf() {
        assert_eq!(detect_line_ending("a\nb\n"), "\n");
    }
    #[test]
    fn test_detect_line_ending_crlf() {
        assert_eq!(detect_line_ending("a\r\nb\r\n"), "\r\n");
    }
    #[test]
    fn test_detect_line_ending_mixed() {
        assert_eq!(detect_line_ending("a\r\nb\n"), "\r\n");
    }
    #[test]
    fn test_detect_line_ending_empty() {
        assert_eq!(detect_line_ending(""), "\n");
    }

    #[test]
    fn test_normalize_to_lf_crlf() {
        assert_eq!(normalize_to_lf("a\r\nb"), "a\nb");
    }
    #[test]
    fn test_normalize_to_lf_mixed() {
        assert_eq!(normalize_to_lf("a\r\nb\nc\rd"), "a\nb\nc\nd");
    }

    #[test]
    fn test_restore_line_endings_to_crlf() {
        assert_eq!(restore_line_endings("a\nb", "\r\n"), "a\r\nb");
    }
    #[test]
    fn test_restore_line_endings_keep_lf() {
        assert_eq!(restore_line_endings("a\nb", "\n"), "a\nb");
    }

    #[test]
    fn test_normalize_fuzzy_smart_quotes() {
        assert_eq!(normalize_for_fuzzy_match("\u{2018}x\u{2019}"), "'x'");
    }
    #[test]
    fn test_normalize_fuzzy_dashes() {
        assert_eq!(normalize_for_fuzzy_match("\u{2013}x\u{2014}"), "-x-");
    }
    #[test]
    fn test_normalize_fuzzy_no_change() {
        assert!(matches!(normalize_for_fuzzy_match("hi"), Cow::Borrowed(_)));
    }
    #[test]
    fn test_normalize_fuzzy_trailing_ws() {
        assert_eq!(normalize_for_fuzzy_match("a  \nb  "), "a\nb");
    }

    #[test]
    fn test_fuzzy_find_exact() {
        let r = fuzzy_find_text("hello", "hello").unwrap();
        assert!(!r.used_fuzzy_match);
        assert_eq!(r.index, 0);
    }
    #[test]
    fn test_fuzzy_find_no_match() {
        assert!(fuzzy_find_text("hello", "xyz").is_none());
    }
    #[test]
    fn test_fuzzy_find_empty() {
        assert!(fuzzy_find_text("hello", "").is_none());
    }

    #[test]
    fn test_strip_bom_present() {
        let (b, t) = strip_bom("\u{FEFF}a");
        assert_eq!(b, "\u{FEFF}");
        assert_eq!(t, "a");
    }
    #[test]
    fn test_strip_bom_absent() {
        let (b, t) = strip_bom("a");
        assert_eq!(b, "");
        assert_eq!(t, "a");
    }

    #[test]
    fn test_apply_edits_single() {
        let edits = vec![Edit {
            old_text: "hello".into(),
            new_text: "hi".into(),
        }];
        let r = apply_edits_to_normalized_content("hello world", &edits, "t.rs").unwrap();
        assert_eq!(r.new_content, "hi world");
    }
    #[test]
    fn test_apply_edits_not_found() {
        let edits = vec![Edit {
            old_text: "xyz".into(),
            new_text: "abc".into(),
        }];
        assert!(matches!(
            apply_edits_to_normalized_content("hello", &edits, "t.rs"),
            Err(EditError::NotFound { .. })
        ));
    }
    #[test]
    fn test_apply_edits_no_change() {
        let edits = vec![Edit {
            old_text: "hello".into(),
            new_text: "hello".into(),
        }];
        assert!(matches!(
            apply_edits_to_normalized_content("hello", &edits, "t.rs"),
            Err(EditError::NoChange { .. })
        ));
    }
    #[test]
    fn test_apply_edits_multiple() {
        let edits = vec![
            Edit {
                old_text: "aaa".into(),
                new_text: "AAA".into(),
            },
            Edit {
                old_text: "ccc".into(),
                new_text: "CCC".into(),
            },
        ];
        let r = apply_edits_to_normalized_content("aaa\nbbb\nccc", &edits, "t.rs").unwrap();
        assert_eq!(r.new_content, "AAA\nbbb\nCCC");
    }

    #[test]
    fn test_generate_diff_no_changes() {
        let (d, l) = generate_diff_string("a\nb", "a\nb", 4);
        assert!(d.is_empty());
        assert!(l.is_none());
    }
    #[test]
    fn test_generate_diff_with_changes() {
        let (d, l) = generate_diff_string("a\nb", "a\nc", 4);
        assert!(!d.is_empty());
        assert!(l.is_some());
    }
}
