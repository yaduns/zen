/// A valid JSON escape character (the legal character after a backslash).
const VALID_JSON_ESCAPES: &[char] = &['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'];

/// Determines whether a character is a JSON control character (U+0000–U+001F).
fn is_control_character(c: char) -> bool {
    c as u32 <= 0x1F
}

/// Converts a control character to a JSON escape sequence and appends it to `output`.
fn escape_control_character(c: char, output: &mut String) {
    match c {
        '\u{0008}' => output.push_str("\\b"), // backspace
        '\u{000C}' => output.push_str("\\f"), // form feed
        '\n' => output.push_str("\\n"),       // line feed
        '\r' => output.push_str("\\r"),       // carriage return
        '\t' => output.push_str("\\t"),       // tab
        other => {
            // Other control characters use the \uXXXX form
            output.push_str(&format!("\\u{:04x}", other as u32));
        }
    }
}

/// Repair escape issues in corrupted JSON string literals.
///
/// Apply the following fixes in order:
/// 1. Inside string values, escape raw control characters (U+0000–U+001F) as valid JSON escape sequences
///    (`\b`/`\f`/`\n`/`\r`/`\t` or `\uXXXX`).
/// 2. Inside string values, for `\` followed by an invalid escape character (not `"`, `\`, `/`, `b`, `f`, `n`, `r`, `t`, `u`)
///    case, double the `\` to `\\` (i.e., unescape it), so later JSON parsers treat it as a normal backslash.
/// 3. For truncated sequences where `\u` is followed by fewer than 4 hex digits, keep them as-is (do not double, because `\u` itself is a valid escape prefix,
///    truncation is handled by partial JSON parsing).
///
/// # Parameters
///
/// * `json` — JSON string that may contain invalid escapes or control characters.
///
/// # Return Value
///
/// Repaired JSON string. If no repair is needed, returns a clone of the original string.
///
/// # Edge cases
///
/// - Empty string: returns `""`.
/// - No string values (JSON structure only): return as-is.
/// - String ends with `\` (truncated escape sequence): double the trailing `\` to `\\`.
/// - Valid escape sequences (`\n`, `\\`, `\"`, `\uXXXX`, etc.): keep unchanged.
pub fn repair_json(json: &str) -> String {
    // Preallocate capacity to avoid frequent reallocation
    let mut repaired = String::with_capacity(json.len());
    let mut in_string = false;
    let chars: Vec<char> = json.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let ch = chars[i];

        // When not inside a string value, copy characters directly, only tracking double-quote state
        if !in_string {
            repaired.push(ch);
            if ch == '"' {
                in_string = true;
            }
            i += 1;
            continue;
        }

        // Inside a string value: exit the string on a closing double quote
        if ch == '"' {
            repaired.push('"');
            in_string = false;
            i += 1;
            continue;
        }

        // Handle backslash escapes
        if ch == '\\' {
            // Case: isolated backslash at end of string (truncated escape sequence)
            if i + 1 >= len {
                repaired.push_str("\\\\");
                i += 1;
                continue;
            }

            let next = chars[i + 1];

            // Case: \u followed by 4 hex digits → valid Unicode escape, keep as-is
            if next == 'u'
                && i + 5 < len
                && chars[i + 2].is_ascii_hexdigit()
                && chars[i + 3].is_ascii_hexdigit()
                && chars[i + 4].is_ascii_hexdigit()
                && chars[i + 5].is_ascii_hexdigit()
            {
                repaired.push('\\');
                repaired.push('u');
                repaired.push(chars[i + 2]);
                repaired.push(chars[i + 3]);
                repaired.push(chars[i + 4]);
                repaired.push(chars[i + 5]);
                i += 6; // Skip \u + 4 digits
                continue;
            }
            // If fewer than 4 hex digits follow \u, do not double here; fall through to the logic below:
            // 'u' is in VALID_JSON_ESCAPES, so \u is kept as-is.

            // Case: valid escape sequence (\n, \\, \", \t, \uXXXX, etc.) → keep as-is
            if VALID_JSON_ESCAPES.contains(&next) {
                repaired.push('\\');
                repaired.push(next);
                i += 2; // Skip backslash and escaped character
                continue;
            }

            // Case: invalid escape sequence → double the backslash to unescape it
            repaired.push_str("\\\\");
            i += 1;
            continue;
        }

        // Control character → escape as a valid JSON escape sequence
        if is_control_character(ch) {
            escape_control_character(ch, &mut repaired);
        } else {
            repaired.push(ch);
        }
        i += 1;
    }

    repaired
}

/// Try to parse a JSON string; if it fails, repair first and retry.
///
/// First try `serde_json::from_str`. If it fails and `repair_json` produced a different string,
/// retry parsing with the repaired string. If it still fails after repair, return the original error.
///
/// # Type parameters
///
/// * `T` — target type, must implement `serde::de::DeserializeOwned`.
///
/// # Parameters
///
/// * `json` — JSON string that may contain control characters or invalid escapes.
///
/// # Return Value
///
/// Returns `Ok(T)` on success, `Err(serde_json::Error)` on failure.
///
/// # Edge cases
///
/// - Valid JSON: parses successfully directly; `repair_json` is not called.
/// - Only control-character issues: succeeds after repair.
/// - Structurally corrupted (e.g., missing closing brackets): repair logic still cannot fix it, returns an error.
pub fn parse_json_with_repair<T>(json_str: &str) -> Result<T, serde_json::Error>
where
    T: serde::de::DeserializeOwned,
{
    // 1. Try direct parsing first
    match serde_json::from_str(json_str) {
        Ok(value) => Ok(value),
        Err(err) => {
            // 2. Repair after failure
            let repaired = repair_json(json_str);
            // 3. Retry only if repair produced a different string
            if repaired != json_str {
                serde_json::from_str(&repaired)
            } else {
                // 4. Repair made no change; return the original error
                Err(err)
            }
        }
    }
}

/// Try to parse a possibly incomplete streaming JSON fragment; always returns a valid value.
///
/// This function is the core of LLM streaming tool-call parsing, handling JSON increments arriving step by step.
/// Parsing strategy (by priority):
///
/// 1. Try direct parsing with `parse_json_with_repair`.
/// 2. If that fails, try partial JSON parsing (fill in missing closing quotes/brackets/braces and retry).
/// 3. If it still fails, return `T::default()` (for `serde_json::Value`, this is `Value::Null`).
///
/// # Type parameters
///
/// * `T` — target type, must implement `DeserializeOwned`. Defaults to `serde_json::Value`.
///
/// # Parameters
///
/// * `partial_json` — JSON fragment arriving in a stream; may be `None` or an empty string.
///
/// # Return Value
///
/// Returns `T` on successful parsing; on failure, returns the default empty value for `T` (`serde_json::Value::Object({})` or the type's default value).
///
/// # Edge cases
///
/// - `None` or an empty string: returns `T::default()`.
/// - Complete JSON: parses successfully directly.
/// - Truncated JSON (e.g., `{"key": "val`): try completion and parse; if it still fails, return an empty object.
/// - Corrupted JSON (with invalid escapes): repair first, then parse.
pub fn parse_streaming_json<T>(partial_json: Option<&str>) -> T
where
    T: serde::de::DeserializeOwned + Default,
{
    // Step 0: None or empty string → return default value
    let json_str = match partial_json {
        Some(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return T::default();
            }
            trimmed
        }
        None => return T::default(),
    };

    // Step 1: Try `parse_json_with_repair`
    if let Ok(value) = parse_json_with_repair::<T>(json_str) {
        return value;
    }

    // Step 2: Try completing the incomplete JSON, then parse
    let completed = complete_partial_json(json_str);
    if let Ok(value) = serde_json::from_str::<T>(&completed) {
        return value;
    }

    // Step 3: Repair escapes first, then complete the incomplete JSON and parse
    let repaired = repair_json(json_str);
    let completed_repaired = complete_partial_json(&repaired);
    if let Ok(value) = serde_json::from_str::<T>(&completed_repaired) {
        return value;
    }

    // Step 4: All failed → return default value
    T::default()
}

// ---------------------------------------------------------------------------
// Partial JSON completion
// ---------------------------------------------------------------------------

/// Complete an incomplete JSON fragment by automatically adding missing closing quotes, braces, and brackets.
///
/// Core helper for `parse_streaming_json`; handles JSON truncated in LLM streaming output.
/// It does not perform full parsing; instead it scans the character stream, tracking string state and nesting depth,
/// and automatically appends the necessary closing characters when input ends.
///
/// # Strategy
///
/// 1. Scan character by character, tracking `in_string` (whether inside a string) and the nesting stack (`{` / `[`).
/// 2. At end of input, if still inside a string, append the closing double quote `"`.
/// 3. Remove trailing redundant commas (where a `,` is followed by no valid value).
/// 4. Append closing brackets/braces in reverse order of the nesting stack.
///
/// # Parameters
///
/// * `input` — possibly incomplete JSON fragment.
///
/// # Return Value
///
/// The completed JSON string. It may still not be valid JSON (e.g., `{"key":` with no value),
/// in which case the caller must fall back to a default value.
///
/// # Edge cases
///
/// - Complete JSON: returned unchanged.
/// - `{"key": "val` → `{"key": "val"}` (completes quote and brace)
/// - `{"key": [1, 2` → `{"key": [1, 2]}` (completes bracket and brace)
/// - `{"key": "val",` → `{"key": "val"}` (removes trailing comma, then appends brace)
/// - `{"key":` → `{"key":}` (appends brace, but value is missing, so parsing will fail)
///
fn complete_partial_json(input: &str) -> String {
    let mut output = String::with_capacity(input.len() + 8);
    // Nesting stack: stores expected closing characters (`}` or `]`)
    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escape_next = false;

    for ch in input.chars() {
        if escape_next {
            output.push(ch);
            escape_next = false;
            continue;
        }

        if in_string {
            output.push(ch);
            if ch == '\\' {
                escape_next = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '{' => {
                stack.push('}');
                output.push(ch);
            }
            '[' => {
                stack.push(']');
                output.push(ch);
            }
            '}' => {
                if stack.last() == Some(&'}') {
                    stack.pop();
                }
                output.push(ch);
            }
            ']' => {
                if stack.last() == Some(&']') {
                    stack.pop();
                }
                output.push(ch);
            }
            '"' => {
                in_string = true;
                output.push(ch);
            }
            _ => {
                output.push(ch);
            }
        }
    }

    // 1. If the string is not closed, append the closing double quote
    if in_string {
        output.push('"');
    }

    // 2. Remove trailing redundant comma (trim whitespace first, then delete comma)
    let trimmed = output.as_str().trim_end();
    let trimmed_no_comma = trimmed.trim_end_matches(',');
    output.truncate(trimmed_no_comma.len());

    // 3. Close the nesting stack in reverse order
    while let Some(&expected) = stack.last() {
        output.push(expected);
        stack.pop();
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    // ---------- is_control_character ----------

    #[test]
    fn test_is_control_character_true_for_boundary_and_common_controls() {
        assert!(is_control_character('\u{0000}'));
        assert!(is_control_character('\u{001F}'));
        assert!(is_control_character('\n'));
    }

    #[test]
    fn test_is_control_character_false_for_printable_boundary() {
        assert!(!is_control_character(' ')); // U+0020, just outside the boundary
        assert!(!is_control_character('a'));
    }

    // ---------- escape_control_character ----------

    #[test]
    fn test_escape_control_character_named_escapes() {
        let cases = [
            ('\u{0008}', "\\b"),
            ('\u{000C}', "\\f"),
            ('\n', "\\n"),
            ('\r', "\\r"),
            ('\t', "\\t"),
        ];
        for (input, expected) in cases {
            let mut out = String::new();
            escape_control_character(input, &mut out);
            assert_eq!(out, expected);
        }
    }

    #[test]
    fn test_escape_control_character_falls_back_to_unicode_escape() {
        let mut out = String::new();
        escape_control_character('\u{0001}', &mut out);
        assert_eq!(out, "\\u0001");
    }

    // ---------- repair_json ----------

    #[test]
    fn test_repair_json_empty_string() {
        assert_eq!(repair_json(""), "");
    }

    #[test]
    fn test_repair_json_already_valid_is_unchanged() {
        let input = "{\"key\":\"value\"}";
        assert_eq!(repair_json(input), input);
    }

    #[test]
    fn test_repair_json_raw_control_char_in_string_gets_escaped() {
        // A literal newline is embedded directly inside a string value (invalid JSON, but common in bad output)
        let input = "{\"key\":\"a\nb\"}";
        assert_eq!(repair_json(input), "{\"key\":\"a\\nb\"}");
    }

    #[test]
    fn test_repair_json_raw_control_char_outside_string_is_untouched() {
        // Outside string values, the function does not handle control characters
        let input = "{\n\"key\":\"value\"}";
        assert_eq!(repair_json(input), input);
    }

    #[test]
    fn test_repair_json_valid_escape_sequences_are_idempotent() {
        // Validate each valid escape sequence individually; keep it unchanged without doubling or modification
        let cases = [
            "{\"key\":\"a\\nb\"}",     // \n
            "{\"key\":\"a\\tb\"}",     // \t
            "{\"key\":\"a\\\\b\"}",    // \\ escaped backslash
            "{\"key\":\"a\\\"b\"}",    // \" escaped quote
            "{\"key\":\"a\\u00e9b\"}", // \uXXXX
        ];
        for input in cases {
            assert_eq!(repair_json(input), input);
        }
    }

    #[test]
    fn test_repair_json_illegal_escape_is_doubled() {
        // \q is not a valid escape; the backslash should be doubled to cancel the escape
        let input = "{\"key\":\"a\\qb\"}";
        assert_eq!(repair_json(input), "{\"key\":\"a\\\\qb\"}");
    }

    #[test]
    fn test_repair_json_truncated_unicode_escape_kept_as_is() {
        // \u followed by fewer than 4 hex digits; keep as-is (do not double)
        let input = "{\"key\":\"a\\u0";
        assert_eq!(repair_json(input), input);
    }

    #[test]
    fn test_repair_json_trailing_backslash_is_doubled() {
        let input = "{\"key\":\"a\\";
        assert_eq!(repair_json(input), "{\"key\":\"a\\\\");
    }

    // ---------- parse_json_with_repair ----------

    #[test]
    fn test_parse_json_with_repair_valid_json_parses_directly() {
        let input = "{\"key\":\"value\"}";
        let value: Value = parse_json_with_repair(input).unwrap();
        assert_eq!(value, serde_json::json!({"key": "value"}));
    }

    #[test]
    fn test_parse_json_with_repair_recovers_illegal_escape() {
        let input = "{\"key\":\"a\\qb\"}";
        let value: Value = parse_json_with_repair(input).unwrap();
        assert_eq!(value, serde_json::json!({"key": "a\\qb"}));
    }

    #[test]
    fn test_parse_json_with_repair_structurally_broken_returns_err() {
        let input = "{\"key\":\"value\""; // Missing closing brace; repair logic cannot fix it
        let result: Result<Value, _> = parse_json_with_repair(input);
        assert!(result.is_err());
    }

    // ---------- complete_partial_json ----------

    #[test]
    fn test_complete_partial_json_already_complete_is_unchanged() {
        let input = "{\"key\":\"value\"}";
        assert_eq!(complete_partial_json(input), input);
    }

    #[test]
    fn test_complete_partial_json_closes_unterminated_string() {
        let input = "{\"key\": \"val";
        assert_eq!(complete_partial_json(input), "{\"key\": \"val\"}");
    }

    #[test]
    fn test_complete_partial_json_closes_unterminated_array() {
        let input = "{\"key\": [1, 2";
        assert_eq!(complete_partial_json(input), "{\"key\": [1, 2]}");
    }

    #[test]
    fn test_complete_partial_json_strips_trailing_comma() {
        let input = "{\"key\": \"val\",";
        assert_eq!(complete_partial_json(input), "{\"key\": \"val\"}");
    }

    #[test]
    fn test_complete_partial_json_escaped_quote_does_not_end_string_early() {
        let input = "{\"key\": \"a\\\"b";
        assert_eq!(complete_partial_json(input), "{\"key\": \"a\\\"b\"}");
    }

    // ---------- parse_streaming_json ----------

    #[test]
    fn test_parse_streaming_json_none_returns_default() {
        let value: Value = parse_streaming_json(None);
        assert_eq!(value, Value::default());
    }

    #[test]
    fn test_parse_streaming_json_blank_string_returns_default() {
        let value: Value = parse_streaming_json(Some("   "));
        assert_eq!(value, Value::default());
    }

    #[test]
    fn test_parse_streaming_json_complete_json_parses_directly() {
        let value: Value = parse_streaming_json(Some("{\"key\":\"value\"}"));
        assert_eq!(value, serde_json::json!({"key": "value"}));
    }

    #[test]
    fn test_parse_streaming_json_truncated_json_is_completed_then_parsed() {
        let value: Value = parse_streaming_json(Some("{\"key\": \"val"));
        assert_eq!(value, serde_json::json!({"key": "val"}));
    }

    #[test]
    fn test_parse_streaming_json_illegal_escape_and_truncated_falls_back_to_repair() {
        let value: Value = parse_streaming_json(Some("{\"key\": \"a\\qb"));
        assert_eq!(value, serde_json::json!({"key": "a\\qb"}));
    }

    #[test]
    fn test_parse_streaming_json_unrecoverable_input_returns_default() {
        let value: Value = parse_streaming_json(Some("not json at all {"));
        assert_eq!(value, Value::default());
    }
}
