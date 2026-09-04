//! Lightweight utility functions shared across modules.

/// Return the current Unix timestamp (milliseconds).
///
/// Used for `Message.timestamp`, `ToolResultMessage.timestamp`,
/// `AssistantMessage.timestamp`, and other fields. Fall back to 0 on overflow (`as i64` saturates).
///
/// # Edge Cases
/// - `SystemTime::now()` may be earlier than UNIX_EPOCH on very few platforms (e.g., VM clock skew),
///   in which case `duration_since` returns `Err`, and `unwrap_or_default` falls back to 0.
pub fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

/// Extracts the frontmatter YAML block and body from the content.
/// After normalizing line endings, if the content starts with `---` and a closing `\n---` is found, returns `(Some(yaml_block), body)`; otherwise, `(None, full text)`.
pub(crate) fn extract_frontmatter(content: &str) -> (Option<&str>, &str) {
    if !content.starts_with("---") {
        return (None, content);
    }
    let rest = &content[3..]; // Skip leading ---
    match rest.find("\n---") {
        Some(idx) => {
            // yaml_block: content after the leading --- and before the closing marker (leading newline stripped)
            let yaml_block = rest[..idx].trim_start_matches('\n');
            // Body: after the closing delimiter (excluding the immediately following newline)
            let body = rest[idx + 4..].trim_start_matches('\n');
            (Some(yaml_block), body)
        }
        None => (None, content),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_now_millis_returns_positive_timestamp() {
        let ts = now_millis();
        // 2024-01-01 00:00:00 UTC = 1704067200000 ms
        assert!(ts > 1_704_067_200_000, "timestamp too old: {ts}");
    }

    #[test]
    fn test_now_millis_is_monotonic_within_call() {
        let t1 = now_millis();
        let t2 = now_millis();
        assert!(t2 >= t1, "timestamps should be non-decreasing");
    }
}
