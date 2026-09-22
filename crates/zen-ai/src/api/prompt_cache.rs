// REQ-AI-02 (pi-102): prompt cache key truncation
// See reference/pi-original/packages/ai/src/api/openai-prompt-cache.ts (11 lines, key truncation only;
// cached_tokens parsing is handled by openai_stream::parse_usage, T-AI-02-1).

/// Maximum code point length of `prompt_cache_key` (aligned with pi `OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH = 64`).
/// pub is only for unit-test assertions and doc readability; no external consumers.
pub const OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH: usize = 64;

/// When a session initiates an LLM request, truncate the session id into a `prompt_cache_key` accepted by the provider—
/// "grind an oversized key down to a length the lock can swallow", ensuring multiple rounds of requests in the same session carry the same stable key to hit the prefix cache.
///
/// Behavior: `None` passes through `None` (the request construction side therefore does not send this parameter); length ≤ 64 Unicode code points is returned as-is;
/// if too long, truncate by code point to the first 64 (matching the code-point semantics of pi's `Array.from(key).slice(0, 64)`—surrogate pairs/CJK
/// characters are not split in half, so byte truncation or grapheme segmentation **must not** be used). `Some("")` returns the empty string as-is (consistent with pi behavior,
/// not converted to None, not dropped).
///
/// # Consumers
/// `openai_stream::build_request_body` (same crate, single request construction point) + unit tests in this file.
pub fn clamp_openai_prompt_cache_key(key: Option<&str>) -> Option<String> {
    key.map(|k| {
        k.chars()
            .take(OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH)
            .collect::<String>()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_none_passthrough() {
        assert_eq!(clamp_openai_prompt_cache_key(None), None);
    }

    #[test]
    fn test_clamp_short_returns_unchanged() {
        let key = "session-abc-123";
        assert_eq!(
            clamp_openai_prompt_cache_key(Some(key)).as_deref(),
            Some(key)
        );
    }

    #[test]
    fn test_clamp_exact_64_unchanged() {
        let key = "a".repeat(OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH);
        assert_eq!(clamp_openai_prompt_cache_key(Some(&key)).unwrap(), key);
    }

    #[test]
    fn test_clamp_long_truncates_to_64() {
        let key = "a".repeat(100);
        let out = clamp_openai_prompt_cache_key(Some(&key)).unwrap();
        assert_eq!(out, "a".repeat(OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH));
    }

    #[test]
    fn test_clamp_multibyte_counts_code_points_not_bytes() {
        // 64 emoji (4 bytes each): truncated by code point → all 64 preserved, no surrogate pair split in half;
        // byte truncation would cut the 64th character in half (at the 128-byte boundary).
        let key = "😀".repeat(64);
        assert_eq!(
            clamp_openai_prompt_cache_key(Some(&key)).unwrap(),
            "😀".repeat(64)
        );
        // oversized multibyte: truncate to 64 code points without producing lone surrogates
        let key2 = "😀".repeat(80);
        let out2 = clamp_openai_prompt_cache_key(Some(&key2)).unwrap();
        assert_eq!(out2, "😀".repeat(64));
        assert_eq!(out2.chars().count(), OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH);
    }

    #[test]
    fn test_clamp_empty_string_passthrough() {
        // pi behavior: Some("") returns the empty string as-is (not converted to None, not dropped)
        assert_eq!(clamp_openai_prompt_cache_key(Some("")), Some(String::new()));
    }
}
