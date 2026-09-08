//! Frontmatter parsing: unified splitting and key-value parsing for `---\nkey: value\n---` blocks in markdown files.
//!
//! REQ-APP-03 (pi-338): frontmatter parsing (skill/command metadata).
//! Consolidated form (T-AP-03-5): unifies zen-core's previously scattered frontmatter capabilities
//! (`util::extract_frontmatter` + key-value line parsing in skills/load.rs + two-key parsing in prompt_templates.rs),
//! Consumers (skills / prompt_templates) project their metadata by key via `parse_frontmatter`, no longer hand-writing line loops.

use std::fmt;

/// Generic frontmatter parsing error—a narrowed form of "invalid YAML" under the minimal YAML subset.
/// Consumers (skills / prompt_templates) convert this error into their respective `ParseFailed` diagnostics,
/// Path context is assembled by the consumer; this error only carries the malformed line.
#[derive(Debug)]
pub enum FrontmatterError {
    /// A malformed line exists in the frontmatter block that is non-empty, non-comment, and has no `:`.
    MalformedLine { line: String },
}

impl fmt::Display for FrontmatterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FrontmatterError::MalformedLine { line } => {
                // Message format aligns with existing consumer assertions (both skills/prompt_templates use contains("malformed"))
                write!(f, "malformed frontmatter line: {line:?}")
            }
        }
    }
}

impl std::error::Error for FrontmatterError {}

/// Complete result of one frontmatter parse—an order-preserving key-value pair list + body.
/// Rust generic form of pi `ParsedFrontmatter<T>`: `T` generic and `Record<string, unknown>` index signature
/// Resolved into `Vec<(String, String)>`; consumers project by their own key sets (skills three keys / templates two keys).
/// Pure in-memory view; not involved in serialization.
#[derive(Debug)]
pub struct ParsedFrontmatter {
    /// Order-preserving key-value pairs (both key and value are trimmed, value is stripped if wrapped in matching quotes, comments/blank lines are skipped).
    pub frontmatter: Vec<(String, String)>,
    /// Body—fully trimmed when frontmatter exists; normalized full text when there is no frontmatter (or it is unclosed).
    pub body: String,
}

/// Parse frontmatter key-value pairs and body from raw markdown—skill loading (T-AG-05-2), prompt templates (T-AG-05-3)
/// and other consumers feed in raw file content to get both "metadata + body" at once, avoiding repeated hand-written block splitting in each consumer.
///
/// Returns empty key-value pairs + normalized full text when there is no frontmatter or `---` is unclosed (aligned with pi parseFrontmatter);
/// Returns `Err(FrontmatterError::MalformedLine)` if a non-comment, colon-less malformed line exists in the block
/// (consumers convert it into their respective `ParseFailed` diagnostics).
/// Behavior aligned with pi: CRLF/CR uniformly normalized, body fully trimmed when frontmatter exists, comment lines (starting with `#`) ignored,
/// value stripped when wrapped in matching single/double quotes. The minimal subset does not cover full YAML syntax (block scalars/arrays/anchors); omitted in v1.
pub fn parse_frontmatter(content: &str) -> Result<ParsedFrontmatter, FrontmatterError> {
    // CRLF/CR normalization (pi normalizeNewlines): splitting and body are both based on normalized text
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    let (yaml_block, body) = extract_frontmatter(&normalized);
    match yaml_block {
        Some(block) => Ok(ParsedFrontmatter {
            frontmatter: parse_key_value_pairs(block)?,
            body: body.to_string(),
        }),
        None => Ok(ParsedFrontmatter {
            frontmatter: Vec::new(),
            body: normalized,
        }),
    }
}

/// Strip frontmatter and take only the body—for `/skill:` command expansion (T-AP-03-2, pi agent-session.ts:1311
/// `stripFrontmatter(content).trim()` scenario) and other consumers that only need the body.
///
/// Internally calls `parse_frontmatter` and takes `.body`: returns normalized full text when there is no frontmatter, and also returns `Err` for malformed lines
/// (aligned with pi stripFrontmatter throwing on invalid YAML); the body is already fully trimmed when "frontmatter exists",
/// so consumers do not need to trim again.
pub fn strip_frontmatter(content: &str) -> Result<String, FrontmatterError> {
    Ok(parse_frontmatter(content)?.body)
}

/// After newline normalization (CRLF/CR → LF), if the content starts with `---` and has a closing `\n---`, split out the frontmatter YAML block and
/// body; otherwise `(None, full text)`. This is the first step of `parse_frontmatter`, with behavior migrated from `util::extract_frontmatter`
/// migrated in and upgraded—completes pi extractFrontmatter's "body fully trimmed when frontmatter exists" semantics
/// (CRLF normalization is done by the caller `parse_frontmatter`; the temporary patch in prompt_templates.rs has been removed).
fn extract_frontmatter(content: &str) -> (Option<&str>, &str) {
    if !content.starts_with("---") {
        return (None, content);
    }
    let rest = &content[3..]; // Skip leading ---
    match rest.find("\n---") {
        Some(idx) => {
            // yaml_block: content after the leading --- and before the closing marker (leading newline stripped)
            let yaml_block = rest[..idx].trim_start_matches('\n');
            // Body: fully trimmed after the closing marker (pi `slice(endIndex + 4).trim()` semantics)
            let body = rest[idx + 4..].trim();
            (Some(yaml_block), body)
        }
        None => (None, content),
    }
}

/// Parse the frontmatter YAML block line by line into order-preserving key-value pairs—the unified syntax surface of the minimal YAML subset (aligned with pi yaml parsing's
/// scalar behavior): blank lines and `#` comment lines are ignored, `key: value` is split at the first `:` (value may contain colons), key/value are both
/// trimmed, and value is stripped if wrapped in matching single/double quotes. Lines that are non-comment and have no `:` return `Err(MalformedLine)`.
/// Consumers (skills three keys / templates two keys) project by their own key sets, no longer hand-writing line loops individually.
fn parse_key_value_pairs(yaml_block: &str) -> Result<Vec<(String, String)>, FrontmatterError> {
    let mut pairs = Vec::new();
    for line in yaml_block.lines() {
        let line = line.trim();
        // Blank lines and comment lines are ignored (aligned with pi yaml parsing: comment-only frontmatter → empty object)
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // split at the first ':': the value may contain colons (e.g. description text)
        let Some((key, value)) = line.split_once(':') else {
            return Err(FrontmatterError::MalformedLine {
                line: line.to_string(),
            });
        };
        let value = strip_quotes(value.trim());
        pairs.push((key.trim().to_string(), value.to_string()));
    }
    Ok(pairs)
}

/// Strip matching single/double quotes from the start and end of value (pi yaml string semantics).
/// Only simple leading/trailing stripping: does not support `\"` escapes (full yaml escape semantics omitted in v1, difference documented).
/// Byte-index safe: the first/last bytes are single-byte ASCII quotes, so split points are always at character boundaries.
fn strip_quotes(value: &str) -> &str {
    if value.len() >= 2 {
        let bytes = value.as_bytes();
        let first = bytes[0];
        let last = bytes[value.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return &value[1..value.len() - 1];
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── extract_frontmatter (migrated from skills/load.rs, behavior preserved + body full-trim upgrade) ───

    #[test]
    fn test_extract_frontmatter_no_frontmatter() {
        let content = "just text\nno frontmatter";
        let (block, body) = extract_frontmatter(content);
        assert_eq!(block, None);
        assert_eq!(body, content);
    }

    #[test]
    fn test_extract_frontmatter_with_block() {
        let content = "---\nname: foo\ndescription: bar\n---\nbody text";
        let (block, body) = extract_frontmatter(content);
        assert_eq!(block, Some("name: foo\ndescription: bar"));
        assert_eq!(body, "body text");
    }

    #[test]
    fn test_extract_frontmatter_unclosed() {
        let content = "---\nname: foo\nno closing marker";
        let (block, body) = extract_frontmatter(content);
        assert_eq!(block, None);
        assert_eq!(body, content);
    }

    #[test]
    fn test_extract_frontmatter_leading_newline_trimmed() {
        let content = "---\n\nname: foo\n---\nbody";
        let (block, body) = extract_frontmatter(content);
        assert_eq!(block, Some("name: foo"));
        assert_eq!(body, "body");
    }

    #[test]
    fn test_extract_frontmatter_body_fully_trimmed() {
        // Upgrade semantics: body is fully trimmed when frontmatter exists (pi `slice(endIndex+4).trim()`)
        let content = "---\nname: foo\n---\n\n\n  body text  \n\n";
        let (block, body) = extract_frontmatter(content);
        assert_eq!(block, Some("name: foo"));
        assert_eq!(body, "body text");
    }

    // ─── parse_key_value_pairs ─────────────────────────────────

    #[test]
    fn test_parse_key_value_pairs_all_keys() {
        let pairs = parse_key_value_pairs(
            "name: foo\ndescription: does foo\ndisable-model-invocation: true",
        )
        .unwrap();
        assert_eq!(
            pairs,
            vec![
                ("name".to_string(), "foo".to_string()),
                ("description".to_string(), "does foo".to_string()),
                ("disable-model-invocation".to_string(), "true".to_string()),
            ]
        );
    }

    #[test]
    fn test_parse_key_value_pairs_quotes_stripped() {
        // pi test: double/single quote stripping
        let pairs =
            parse_key_value_pairs("name: \"skill-name\"\ndescription: 'A desc'\nfoo-bar: value")
                .unwrap();
        assert_eq!(
            pairs,
            vec![
                ("name".to_string(), "skill-name".to_string()),
                ("description".to_string(), "A desc".to_string()),
                ("foo-bar".to_string(), "value".to_string()),
            ]
        );
    }

    #[test]
    fn test_parse_key_value_pairs_value_with_colon() {
        let pairs = parse_key_value_pairs("description: uses a: b and c").unwrap();
        assert_eq!(
            pairs,
            vec![("description".to_string(), "uses a: b and c".to_string())]
        );
    }

    #[test]
    fn test_parse_key_value_pairs_trim_values() {
        let pairs = parse_key_value_pairs("  name:  foo  \n description: bar ").unwrap();
        assert_eq!(
            pairs,
            vec![
                ("name".to_string(), "foo".to_string()),
                ("description".to_string(), "bar".to_string()),
            ]
        );
    }

    #[test]
    fn test_parse_key_value_pairs_empty_lines_skipped() {
        let pairs = parse_key_value_pairs("\n\nname: foo\n\n").unwrap();
        assert_eq!(pairs, vec![("name".to_string(), "foo".to_string())]);
    }

    #[test]
    fn test_parse_key_value_pairs_comment_lines_ignored() {
        // pi test: comment-only frontmatter → empty object (zen's old implementation reported malformed for comment lines; fixed)
        let pairs = parse_key_value_pairs("# just a comment\n\n# another").unwrap();
        assert!(pairs.is_empty());

        let pairs = parse_key_value_pairs("# header\nname: foo\n# footer").unwrap();
        assert_eq!(pairs, vec![("name".to_string(), "foo".to_string())]);
    }

    #[test]
    fn test_parse_key_value_pairs_malformed_line_errors() {
        let err = parse_key_value_pairs("name: foo\nthis has no colon").unwrap_err();
        assert!(err.to_string().contains("malformed"));
    }

    #[test]
    fn test_parse_key_value_pairs_inline_comment_not_special() {
        // Minimal subset: inline `#` is not a comment (only leading `#` is ignored), value is preserved as-is—differs from pi yaml, documented
        let pairs = parse_key_value_pairs("name: foo # not a comment here").unwrap();
        assert_eq!(
            pairs,
            vec![("name".to_string(), "foo # not a comment here".to_string())]
        );
    }

    // ─── parse_frontmatter (aligned with all migratable cases in pi frontmatter.test.ts) ───

    #[test]
    fn test_parse_frontmatter_parses_keys_strips_quotes_returns_body() {
        let input =
            "---\nname: \"skill-name\"\ndescription: 'A desc'\nfoo-bar: value\n---\n\nBody text";
        let parsed = parse_frontmatter(input).unwrap();
        assert_eq!(
            parsed.frontmatter,
            vec![
                ("name".to_string(), "skill-name".to_string()),
                ("description".to_string(), "A desc".to_string()),
                ("foo-bar".to_string(), "value".to_string()),
            ]
        );
        assert_eq!(parsed.body, "Body text");
    }

    #[test]
    fn test_parse_frontmatter_normalizes_newlines_and_handles_crlf() {
        let input = "---\r\nname: test\r\n---\r\nLine one\r\nLine two";
        let parsed = parse_frontmatter(input).unwrap();
        assert_eq!(
            parsed.frontmatter,
            vec![("name".to_string(), "test".to_string())]
        );
        assert_eq!(parsed.body, "Line one\nLine two");
    }

    #[test]
    fn test_parse_frontmatter_returns_original_when_missing_or_unterminated() {
        let no_frontmatter = "Just text\nsecond line";
        let result_no_frontmatter = parse_frontmatter(no_frontmatter).unwrap();
        assert!(result_no_frontmatter.frontmatter.is_empty());
        assert_eq!(result_no_frontmatter.body, "Just text\nsecond line");

        let missing_end = "---\nname: test\nBody without terminator";
        let result_missing_end = parse_frontmatter(missing_end).unwrap();
        assert!(result_missing_end.frontmatter.is_empty());
        assert_eq!(
            result_missing_end.body,
            "---\nname: test\nBody without terminator"
        );
    }

    #[test]
    fn test_parse_frontmatter_empty_or_comment_only_is_empty() {
        let input = "---\n# just a comment\n---\nBody";
        let parsed = parse_frontmatter(input).unwrap();
        assert!(parsed.frontmatter.is_empty());
        assert_eq!(parsed.body, "Body");
    }

    #[test]
    fn test_parse_frontmatter_malformed_line_errors() {
        let err = parse_frontmatter("---\nno colon here\n---\nbody").unwrap_err();
        assert!(err.to_string().contains("malformed"));
    }

    #[test]
    fn test_parse_frontmatter_crlf_normalized_no_frontmatter() {
        // Returns normalized full text when there is no frontmatter (pi semantics)
        let parsed = parse_frontmatter("line one\r\nline two").unwrap();
        assert_eq!(parsed.body, "line one\nline two");
    }

    // ─── strip_frontmatter ─────────────────────────────────────

    #[test]
    fn test_strip_frontmatter_removes_and_trims_body() {
        let input = "---\nkey: value\n---\n\nBody\n";
        assert_eq!(strip_frontmatter(input).unwrap(), "Body");
    }

    #[test]
    fn test_strip_frontmatter_returns_body_when_no_frontmatter() {
        let input = "\n  No frontmatter body  \n";
        // pi semantics: returns original text when there is no frontmatter (no trim)
        assert_eq!(
            strip_frontmatter(input).unwrap(),
            "\n  No frontmatter body  \n"
        );
    }

    #[test]
    fn test_strip_frontmatter_malformed_errors() {
        assert!(strip_frontmatter("---\nno colon here\n---\nbody").is_err());
    }
}
