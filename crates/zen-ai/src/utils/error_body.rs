/// Extracts response body text from an HTTP error response.
/// # Parameters
/// - `response`: HTTP error response (non-2xx).
/// - `max_bytes`: maximum number of bytes to read; bodies longer than this are truncated. Default recommendation is 4000.
/// # Return Value
/// Response body text (truncated), or `None` if it cannot be read.
pub async fn extract_error_body(response: reqwest::Response, max_bytes: usize) -> Option<String> {
    let bytes = response.bytes().await.ok()?;
    // bytes: &[u8] — raw HTTP response body
    // May be UTF-8 text, but may also contain non-UTF-8 bytes
    // We don't want to lose the entire error message because of one bad byte

    if bytes.is_empty() {
        return None;
    }
    let truncated = &bytes[..bytes.len().min(max_bytes)];

    // Best-effort decode of a &[u8] slice to &str; invalid UTF-8 bytes are replaced with U+FFFD instead of erroring
    // .into_owned() ensures an owned String is returned in either case.
    Some(String::from_utf8_lossy(truncated).into_owned())
}

/// Extracts the `error.message` field from an OpenAI-format error response JSON.
/// OpenAI-compatible API error responses usually have the format:
/// ```json
/// { "error": { "message": "Incorrect API key", "type": "authentication_error", "code": "invalid_api_key" } }
/// ```
pub fn extract_openai_error_message(body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    value
        .get("error")?
        .get("message")?
        .as_str()
        .map(String::from)
}

/// Extracts a "readable message" from an error body, for `StreamError::ApiError`'s Display—
/// "when the gateway returns 400, don't dump the entire JSON to the terminal; if `error.message` can be extracted, show it".
/// Prefers OpenAI `error.message`; some gateways stuff upstream JSON verbatim into the message string, in which case parse one more layer;
/// if unavailable/empty, falls back to the raw body (prefer noise over losing information).
/// Consumer: `openai_stream` constructs `StreamError::ApiError` (the full body is also retained in debug logs).
pub fn extract_readable_error_message(body: &str) -> String {
    let fallback = || body.trim().to_string();
    let Some(message) = extract_openai_error_message(body) else {
        return fallback();
    };
    let message = message.trim();
    if message.is_empty() {
        return fallback();
    }
    // Nested JSON: when message itself has the form `{"error":{"message":"..."}}`, parse one more layer
    if let Some(inner) = extract_openai_error_message(message) {
        let inner = inner.trim();
        if !inner.is_empty() {
            return inner.to_string();
        }
    }
    message.to_string()
}

/// Formats HTTP error information into a readable string.
/// # Format
/// - With `body`: `"<status>: <body>"` (or with prefix `"<prefix> (<status>): <body>"`)
/// - Without `body`: `"<status>: <message>"` (or with prefix `"<prefix> (<status>): <message>"`)
/// - With neither `status` nor `body`: returns `message` directly
pub fn format_error_message(
    status: u16,
    body: Option<&str>,
    message: &str,
    prefix: Option<&str>,
) -> String {
    match (prefix, body) {
        (Some(p), Some(b)) => format!("{} ({}): {}", p, status, b),
        (Some(p), None) => format!("{} ({}): {}", p, status, message),
        (None, Some(b)) => format!("{}: {}", status, b),
        (None, None) => format!("{}: {}", status, message),
    }
}

/// Truncates error text, replacing the part beyond `max_chars` with a `[truncated N chars]` suffix.
/// # Parameters
/// - `text`: original text.
/// - `max_chars`: maximum number of characters.
pub fn truncate_error_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let truncated: String = text.chars().take(max_chars).collect();
    format!(
        "{}... [truncated {} chars]",
        truncated,
        text.chars().count() - max_chars
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── truncate_error_text ──

    #[test]
    fn test_truncate_short_text() {
        assert_eq!(truncate_error_text("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_exact_length() {
        assert_eq!(truncate_error_text("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_long_text() {
        let result = truncate_error_text("a very long error message", 10);
        assert_eq!(result, "a very lon... [truncated 15 chars]");
    }

    #[test]
    fn test_truncate_empty_string() {
        assert_eq!(truncate_error_text("", 10), "");
    }

    #[test]
    fn test_truncate_zero_max_chars() {
        let result = truncate_error_text("hello", 0);
        assert_eq!(result, "... [truncated 5 chars]");
    }

    #[test]
    fn test_truncate_unicode_multi_byte() {
        // Each Chinese character is multibyte, but chars().count() counts characters
        let result = truncate_error_text("你好世界，这是一个测试", 4);
        assert_eq!(result, "你好世界... [truncated 7 chars]");
    }

    // ── extract_openai_error_message ──

    #[test]
    fn test_extract_message_found() {
        let body = r#"{"error": {"message": "Incorrect API key", "type": "authentication_error"}}"#;
        assert_eq!(
            extract_openai_error_message(body),
            Some("Incorrect API key".to_string())
        );
    }

    #[test]
    fn test_extract_message_missing_error_field() {
        let body = r#"{"status": "error"}"#;
        assert_eq!(extract_openai_error_message(body), None);
    }

    #[test]
    fn test_extract_message_missing_message_field() {
        let body = r#"{"error": {"code": "rate_limit"}}"#;
        assert_eq!(extract_openai_error_message(body), None);
    }

    #[test]
    fn test_extract_message_invalid_json() {
        assert_eq!(extract_openai_error_message("not json"), None);
    }

    #[test]
    fn test_extract_message_empty_string() {
        assert_eq!(extract_openai_error_message(""), None);
    }

    #[test]
    fn test_extract_message_non_string_value() {
        // message field is present but is not a string
        let body = r#"{"error": {"message": 123}}"#;
        assert_eq!(extract_openai_error_message(body), None);
    }

    #[test]
    fn test_extract_message_nested_deepseek_format() {
        // DeepSeek sometimes returns a nested error structure
        let body = r#"{"error": {"message": "Rate limit reached"}}"#;
        assert_eq!(
            extract_openai_error_message(body),
            Some("Rate limit reached".to_string())
        );
    }

    // ── extract_readable_error_message ──

    #[test]
    fn readable_prefers_error_message() {
        let body = r#"{"error":{"param":null,"type":"invalid_request_error","code":"invalid_request_error","message":"Ups... bad model"}}"#;
        assert_eq!(extract_readable_error_message(body), "Ups... bad model");
    }

    #[test]
    fn readable_unwraps_nested_json_message() {
        // Gateway stuffs upstream JSON into the message string: parse one more layer
        let inner = r#"{"error":{"message":"upstream rejected field X"}}"#;
        let body = serde_json::json!({ "error": { "message": inner } }).to_string();
        assert_eq!(
            extract_readable_error_message(&body),
            "upstream rejected field X"
        );
    }

    #[test]
    fn readable_falls_back_to_trimmed_body() {
        assert_eq!(extract_readable_error_message("  not json  "), "not json");
        // error.message is an empty string → fall back to body
        let body = r#"{"error":{"message":"   "}}"#;
        assert_eq!(extract_readable_error_message(body), body);
    }

    // ── format_error_message ──

    #[test]
    fn test_format_with_body_no_prefix() {
        let result = format_error_message(403, Some("rate limit"), "error", None);
        assert_eq!(result, "403: rate limit");
    }

    #[test]
    fn test_format_without_body_no_prefix() {
        let result = format_error_message(403, None, "Forbidden", None);
        assert_eq!(result, "403: Forbidden");
    }

    #[test]
    fn test_format_with_body_with_prefix() {
        let result = format_error_message(401, Some("bad key"), "error", Some("DeepSeek"));
        assert_eq!(result, "DeepSeek (401): bad key");
    }

    #[test]
    fn test_format_without_body_with_prefix() {
        let result = format_error_message(500, None, "Internal error", Some("DeepSeek"));
        assert_eq!(result, "DeepSeek (500): Internal error");
    }

    // ── extract_error_body ──
    // Depends on a real HTTP response from reqwest::Response, so it cannot be covered by pure unit tests.
    // Recommend verifying by constructing a mock server in integration tests.
}
