// REQ-AI-01 (pi-108): context overflow detection
//
// Determine whether the AssistantMessage returned by the LLM failed due to context overflow.
// Supports three detection paths:
//   1. Error message pattern matching (most providers)
//   2. Silent overflow (z.ai style)
//   3. Length-truncation overflow (Xiaomi MiMo style)

use std::sync::LazyLock;

use regex::Regex;

use crate::types::{AssistantMessage, StopReason};

/// Set of regex patterns for context overflow error messages.
static OVERFLOW_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
           Regex::new(r"prompt is too long").unwrap(),
           Regex::new(r"request_too_large").unwrap(),
           Regex::new(r"input is too long for requested model").unwrap(),
           Regex::new(r"exceeds the context window").unwrap(),
           Regex::new(r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))").unwrap(),
           Regex::new(r"input token count.*exceeds the maximum").unwrap(),
           Regex::new(r"maximum prompt length is \d+").unwrap(),
           Regex::new(r"reduce the length of the messages").unwrap(),
           Regex::new(r"maximum context length is \d+ tokens").unwrap(),
           Regex::new(r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?").unwrap(),
           Regex::new(r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)").unwrap(),
           Regex::new(r"exceeds the limit of \d+").unwrap(),
           Regex::new(r"exceeds the available context size").unwrap(),
           Regex::new(r"greater than the context length").unwrap(),
           Regex::new(r"context window exceeds limit").unwrap(),
           Regex::new(r"exceeded model token limit").unwrap(),
           Regex::new(r"too large for model with \d+ maximum context length").unwrap(),
           Regex::new(r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?").unwrap(),
           Regex::new(r"model_context_window_exceeded").unwrap(),
           Regex::new(r"prompt too long; exceeded (?:max )?context length").unwrap(),
           Regex::new(r"range of input length should be").unwrap(),
           Regex::new(r"context[_ ]length[_ ]exceeded").unwrap(),
           Regex::new(r"too many tokens").unwrap(),
           Regex::new(r"token limit exceeded").unwrap(),
           Regex::new(r"^4(?:00|13)\s*(?:status code)?\s*\(no body\)").unwrap(),
       ]
});

/// Set of non-overflow error patterns (used to rule out false positives).
static NON_OVERFLOW_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"^(Throttling error|Service unavailable):").unwrap(),
        Regex::new(r"rate limit").unwrap(),
        Regex::new(r"too many requests").unwrap(),
    ]
});

// ============================================================
// overflow.rs — context overflow detection
// ============================================================

/// Detect whether `AssistantMessage` failed due to context overflow.
///
/// Supports three detection paths:
///
/// 1. **Error message pattern matching**: when `stop_reason` is `Error` and `error_message` is present,
///    check whether it matches any pattern in `OVERFLOW_PATTERNS` and does not match `NON_OVERFLOW_PATTERNS`.
/// 2. **Silent overflow** (z.ai style): when `stop_reason` is `Stop` and `context_window` is present,
///    check `usage.input + cache_read > context_window`.
/// 3. **Length-truncation overflow** (Xiaomi MiMo style): when `stop_reason` is `Length`, `context_window` is present, and
///    `usage.output` is 0, check `input + cache_read >= context_window * 0.99`.
///
/// # Parameters
/// - `message`: reference to the assistant message to check.
/// - `context_window`: optional context window size (in tokens), used to detect silent overflow and truncation overflow.
///
/// # Return Value
/// Returns `true` if the message indicates context overflow, otherwise `false`.
///
/// # Edge cases
/// - Always returns `false` when `stop_reason` is not `Error`/`Stop`/`Length`.
/// - Skips path 1 when `error_message` is `None`.
/// - Skips paths 2 and 3 when `context_window` is `None`.
pub fn is_context_overflow(message: &AssistantMessage, context_window: Option<usize>) -> bool {
    // Path 1: error message pattern matching
    if message.stop_reason == StopReason::Error {
        if let Some(ref error_msg) = message.error_message {
            let is_non_overflow = NON_OVERFLOW_PATTERNS.iter().any(|p| p.is_match(error_msg));
            let is_overflow = OVERFLOW_PATTERNS.iter().any(|p| p.is_match(error_msg));
            if is_overflow && !is_non_overflow {
                return true;
            }
        }
    }

    // Path 2: silent overflow (z.ai style)
    if let Some(cw) = context_window {
        if message.stop_reason == StopReason::Stop {
            let input_tokens = message.usage.input + message.usage.cache_read;
            if input_tokens > cw as u64 {
                return true;
            }
        }
    }

    // Path 3: length-truncation overflow (Xiaomi MiMo style)
    if let Some(cw) = context_window {
        if message.stop_reason == StopReason::Length && message.usage.output == 0 {
            let input_tokens = message.usage.input + message.usage.cache_read;
            if input_tokens as f64 >= cw as f64 * 0.99 {
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Cost, Usage};

    /// Quickly construct an AssistantMessage, caring only about fields relevant to overflow detection.
    fn make_msg(
        stop_reason: StopReason,
        error_message: Option<&str>,
        input: u64,
        output: u64,
        cache_read: u64,
    ) -> AssistantMessage {
        AssistantMessage {
            content: vec![],
            api: String::new(),
            provider: String::new(),
            model: String::new(),
            response_model: None,
            response_id: None,
            usage: Usage {
                total_tokens: 0,
                input,
                output,
                cache_read,
                cache_write: 0,
                cost: Cost {
                    input: 0.0,
                    output: 0.0,
                    cache_read: 0.0,
                    cache_write: 0.0,
                    total: 0.0,
                },
                cache_write_1h: None,
                reasoning: None,
            },
            stop_reason,
            error_message: error_message.map(|s| s.to_string()),
            timestamp: 0,
        }
    }

    // ── Path 1: error message pattern matching ───────────────────────────────

    #[test]
    fn test_overflow_detected_by_error_message() {
        let msg = make_msg(StopReason::Error, Some("prompt is too long: 213462 tokens > 200000 maximum"), 0, 0, 0);
        assert!(is_context_overflow(&msg, None));
    }

    #[test]
    fn test_overflow_detected_by_exceeds_context_window() {
        let msg = make_msg(StopReason::Error, Some("Your input exceeds the context window of this model"), 0, 0, 0);
        assert!(is_context_overflow(&msg, None));
    }

    #[test]
    fn test_overflow_detected_by_maximum_context_length() {
        let msg = make_msg(
            StopReason::Error,
            Some("Requested token count exceeds the model's maximum context length of 131072 tokens"),
            0, 0, 0,
        );
        assert!(is_context_overflow(&msg, None));
    }

    #[test]
    fn test_overflow_detected_by_input_token_count() {
        let msg = make_msg(
            StopReason::Error,
            Some("The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)"),
            0, 0, 0,
        );
        assert!(is_context_overflow(&msg, None));
    }

    #[test]
    fn test_overflow_detected_by_too_many_tokens() {
        let msg = make_msg(StopReason::Error, Some("too many tokens"), 0, 0, 0);
        assert!(is_context_overflow(&msg, None));
    }

    #[test]
    fn test_overflow_not_detected_when_error_message_mismatches() {
        let msg = make_msg(
            StopReason::Error,
            Some("An unexpected error occurred during processing"),
            0, 0, 0,
        );
        assert!(!is_context_overflow(&msg, None));
    }

    #[test]
    fn test_overflow_not_detected_for_rate_limit() {
        // rate limit matches NON_OVERFLOW_PATTERNS, should be excluded
        let msg = make_msg(
            StopReason::Error,
            Some("rate limit exceeded, please try again later"),
            0, 0, 0,
        );
        assert!(!is_context_overflow(&msg, None));
    }

    #[test]
    fn test_overflow_not_detected_for_throttling_error() {
        // Throttling error matches NON_OVERFLOW_PATTERNS
        let msg = make_msg(
            StopReason::Error,
            Some("Throttling error: Too many tokens, please wait before trying again."),
            0, 0, 0,
        );
        assert!(!is_context_overflow(&msg, None));
    }

    #[test]
    fn test_overflow_not_detected_when_error_message_is_none() {
        let msg = make_msg(StopReason::Error, None, 0, 0, 0);
        assert!(!is_context_overflow(&msg, None));
    }

    #[test]
    fn test_overflow_not_detected_for_non_error_stop_reason() {
        // StopReason::Stop does not trigger path 1
        let msg = make_msg(StopReason::Stop, Some("prompt is too long"), 0, 0, 0);
        assert!(!is_context_overflow(&msg, None));
    }

    // ── Path 2: silent overflow ───────────────────────────────────────

    #[test]
    fn test_silent_overflow_detected() {
        // input + cache_read = 110 > 100 (context_window)
        let msg = make_msg(StopReason::Stop, None, 100, 0, 10);
        assert!(is_context_overflow(&msg, Some(100)));
    }

    #[test]
    fn test_silent_overflow_not_detected_when_input_within_window() {
        // input + cache_read = 90 <= 100
        let msg = make_msg(StopReason::Stop, None, 80, 0, 10);
        assert!(!is_context_overflow(&msg, Some(100)));
    }

    #[test]
    fn test_silent_overflow_not_detected_without_context_window() {
        // Skips path 2 when context_window = None
        let msg = make_msg(StopReason::Stop, None, 200, 0, 0);
        assert!(!is_context_overflow(&msg, None));
    }

    #[test]
    fn test_silent_overflow_not_detected_for_non_stop() {
        // Only StopReason::Stop triggers path 2
        let msg = make_msg(StopReason::ToolUse, None, 200, 0, 0);
        assert!(!is_context_overflow(&msg, Some(100)));
    }

    // ── Path 3: length-truncation overflow ───────────────────────────────────

    #[test]
    fn test_length_truncation_overflow_detected() {
        // output=0, input=100 >= 100*0.99
        let msg = make_msg(StopReason::Length, None, 100, 0, 0);
        assert!(is_context_overflow(&msg, Some(100)));
    }

    #[test]
    fn test_length_truncation_overflow_at_boundary() {
        // output=0, input=99 >= 100*0.99 (99 >= 99.0)
        let msg = make_msg(StopReason::Length, None, 99, 0, 0);
        assert!(is_context_overflow(&msg, Some(100)));
    }

    #[test]
    fn test_length_truncation_not_detected_when_below_threshold() {
        // output=0, input=98 < 100*0.99 (98 < 99.0)
        let msg = make_msg(StopReason::Length, None, 98, 0, 0);
        assert!(!is_context_overflow(&msg, Some(100)));
    }

    #[test]
    fn test_length_truncation_not_detected_when_output_nonzero() {
        // Skips path 3 when output > 0
        let msg = make_msg(StopReason::Length, None, 100, 10, 0);
        assert!(!is_context_overflow(&msg, Some(100)));
    }

    #[test]
    fn test_length_truncation_not_detected_without_context_window() {
        // Skips path 3 when context_window = None
        let msg = make_msg(StopReason::Length, None, 100, 0, 0);
        assert!(!is_context_overflow(&msg, None));
    }

    // ── Combined ───────────────────────────────────────────────────

    #[test]
    fn test_no_overflow_for_stop_with_small_input() {
        // Matches no path
        let msg = make_msg(StopReason::Stop, None, 10, 20, 0);
        assert!(!is_context_overflow(&msg, Some(100)));
    }

    #[test]
    fn test_no_overflow_for_tool_use() {
        // ToolUse matches no path
        let msg = make_msg(StopReason::ToolUse, None, 0, 0, 0);
        assert!(!is_context_overflow(&msg, Some(100)));
    }

    #[test]
    fn test_overflow_with_cache_read_considered() {
        // Path 2: input + cache_read computed together
        let msg = make_msg(StopReason::Stop, None, 60, 0, 50);  // 110 > 100
        assert!(is_context_overflow(&msg, Some(100)));
    }

    #[test]
    fn test_error_message_matches_overflow_but_not_excluded() {
        // Error message matches OVERFLOW_PATTERNS but not NON_OVERFLOW_PATTERNS
        let msg = make_msg(
            StopReason::Error,
            Some("range of input length should be [1, 128000]"),
            0, 0, 0,
        );
        assert!(is_context_overflow(&msg, None));
    }

    #[test]
    fn test_combined_overflow_paths_precedence() {
        // Returns true early when path 1 matches, without checking paths 2/3
        let msg = make_msg(
            StopReason::Error,
            Some("prompt is too long"),
            5, 0, 0,  // Small input, but the error message matches
        );
        assert!(is_context_overflow(&msg, Some(100)));
    }
}

