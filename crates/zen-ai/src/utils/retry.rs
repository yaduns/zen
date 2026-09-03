use regex::Regex;
use std::future::Future;
use std::sync::OnceLock;
use tokio_util::sync::CancellationToken;

use crate::types::{AssistantMessage, StopReason};

/// Retry policy configuration.
/// Controls retry enable/disable, max retry count, and the base delay for exponential backoff.
/// When `enabled` is `false`, `retry_assistant_call` returns the first result of `produce()` directly,
#[derive(Clone, Copy)]
pub struct RetryPolicy {
    pub enabled: bool,
    pub max_retries: u32,
    pub base_delay_ms: u64,
}

/// Non-retryable error patterns. Matches non-transient errors such as quota/billing/subscription limits.
fn non_retryable_pattern() -> &'static Regex {
    // Initialized at program startup
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        // Compile the regex once per call and cache it for subsequent calls
        Regex::new(r"(?i)GoUsageLimitError|FreeUsageLimitError|Monthly usage limit reached|available balance|insufficient_quota|out of budget|quota exceeded|billing")
            .expect("invalid non-retryable pattern")
    })
}

/// Retryable error patterns. Matches temporary errors such as provider overload / network timeouts / 5xx.
fn retryable_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(r"(?i)overloaded|rate.?limit|too many requests|429|500|502|503|504|524|service.?unavailable|server.?error|internal.?error|provider.?returned.?error|network.?error|connection.?error|connection.?refused|connection.?lost|other side closed|fetch failed|upstream.?connect|reset before headers|socket hang up|socket connection was closed|timed? out|timeout|terminated|websocket.?closed|websocket.?error|ended without|stream ended before message_stop|stream ended before a terminal response event|http2 request did not get a response|retry delay|you can retry your request|try your request again|please retry your request|ResourceExhausted")
            .expect("invalid retryable pattern")
    })
}

/// Retries the `produce` call with exponential backoff until it succeeds, retries are exhausted, or it is cancelled.
///
/// Retry progress is emitted via `tracing` events (target: `"ai::retry"`); callers need not inject callbacks.
/// Retry logs become visible once `tracing_subscriber` is set up at the application layer (see T-OB-01-1).
///
/// - If the `AssistantMessage` returned by `produce()` has `stop_reason` of `Stop`/`Length`/`ToolUse`,
///   it is treated as a success and returned immediately.
/// - `stop_reason` of `Aborted` is treated as a cancellation: no retry, return directly (the cancel signal may fire
///   inside `produce()` or during the backoff sleep).
/// - If `stop_reason` is `Error`, use `is_retryable_assistant_error` to determine whether it is retryable.
///   If not retryable, return that error message immediately.
/// - If retryable and `max_retries` is not exceeded, perform an exponential backoff sleep (`baseDelayMs * 2^(attempt-1)`),
///   then check whether `cancel` has fired before retrying. If already cancelled, change `stop_reason` to `Aborted` and return.
/// - If `cancel` fires during the sleep, the sleep is interrupted early; likewise change `stop_reason` to `Aborted` and return.
///
/// When `policy` is `None` or `policy.enabled == false`, return the result of `produce()` directly.
///
/// # Parameters
/// - `produce`: async closure that generates an `AssistantMessage`. Called again on every retry.
/// - `policy`: retry policy. No retry when `None` or `enabled == false`.
/// - `cancel`: optional cancellation token. When cancelled, the retry loop terminates immediately and returns a message with `stop_reason == Aborted`.
///
/// # Return Value
/// The final `AssistantMessage`: a successful response, a non-retryable error, a cancellation, or an error after retries are exhausted.
pub async fn retry_assistant_call<F, Fut>(
    produce: F,
    policy: Option<RetryPolicy>,
    cancel: Option<CancellationToken>,
) -> AssistantMessage
where
    F: Fn() -> Fut,
    Fut: Future<Output = AssistantMessage>,
{
    // If policy exists and enabled is true, take max_retries; otherwise use 0 (no retry)
    let max_attempts = policy
        .as_ref()
        .filter(|p| p.enabled) // Keep if the inner value matches the condition, otherwise set to None
        .map_or(0, |p| p.max_retries); // Some(p) => p.max_retries,None => 0,

    let mut attempt = 0u32;

    loop {
        let response = produce().await;

        // Abort → return directly
        if response.stop_reason == StopReason::Aborted {
            return response;
        }

        // Success (Stop / Length / ToolUse) → return directly
        if response.stop_reason != StopReason::Error {
            return response;
        }

        // Non-retryable or retries exhausted → return the current error
        if attempt >= max_attempts || !is_retryable_assistant_error(&response) {
            return response;
        }

        // Retryable: exponential backoff
        attempt += 1;
        let delay_ms = policy.as_ref().unwrap().base_delay_ms * 2u64.pow(attempt - 1);

        tracing::info!(
            target: "ai::retry",
            attempt,
            max_attempts,
            delay_ms,
            error = response.error_message.as_deref().unwrap_or("unknown"),
            "retrying"
        );

        // Sleep (cancellation-aware)
        let sleep = tokio::time::sleep(std::time::Duration::from_millis(delay_ms));
        // There is a cancel button; sleep and the button click race to see which completes first;
        // If there is no cancel button, just wait for sleep to finish. (A cancel button existing does not mean the cancel action has occurred.)
        if let Some(ref cancel) = cancel {
            tokio::select! {
                // Sleep completed normally
                _ = sleep => {}
                // Cancellation signal triggered
                _ = cancel.cancelled() => {
                    return AssistantMessage {
                        stop_reason: StopReason::Aborted,
                        error_message: None,
                        ..response
                    };
                }
            }
        } else {
            sleep.await;
        }
    }
}

/// Determines whether a failed `AssistantMessage` is a retryable transient error.
///
/// Rules:
/// 1. If `stop_reason` is not `Error` or `error_message` is `None`, return `false`.
/// 2. If `error_message` matches `NON_RETRYABLE_PATTERN` (quota/billing/subscription limits, etc.), return `false`.
/// 3. If `error_message` matches `RETRYABLE_PATTERN` (provider overload/network timeout/5xx, etc.), return `true`.
/// 4. If none of the above match, return `false` (unknown error type, do not retry).
///
/// Counterpart to Pi's `isRetryableAssistantError` function.
///
/// # Parameters
/// - `message`: the failed `AssistantMessage`; must have `stop_reason == Error`.
///
/// # Return Value
/// Returns `true` if it is a retryable transient error, otherwise `false`.
pub fn is_retryable_assistant_error(message: &AssistantMessage) -> bool {
    if message.stop_reason != StopReason::Error {
        return false;
    }
    let error_msg = match message.error_message.as_deref() {
        Some(msg) => msg,
        None => return false,
    };
    if non_retryable_pattern().is_match(error_msg) {
        return false;
    }
    retryable_pattern().is_match(error_msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Cost, Usage};

    // ── helpers ──

    fn make_assistant_msg(
        stop_reason: StopReason,
        error_message: Option<&str>,
    ) -> AssistantMessage {
        AssistantMessage {
            content: vec![],
            api: String::new(),
            provider: String::new(),
            model: String::new(),
            response_model: None,
            response_id: None,
            usage: Usage {
                input: 0,
                output: 0,
                cache_read: 0,
                cache_write: 0,
                cache_write_1h: None,
                reasoning: None,
                total_tokens: 0,
                cost: Cost {
                    input: 0.0,
                    output: 0.0,
                    cache_read: 0.0,
                    cache_write: 0.0,
                    total: 0.0,
                },
            },
            stop_reason,
            error_message: error_message.map(String::from),
            timestamp: 0,
        }
    }

    fn make_stop_msg() -> AssistantMessage {
        make_assistant_msg(StopReason::Stop, None)
    }

    fn make_error_msg(text: &str) -> AssistantMessage {
        make_assistant_msg(StopReason::Error, Some(text))
    }

    fn make_aborted_msg() -> AssistantMessage {
        make_assistant_msg(StopReason::Aborted, None)
    }

    fn make_tool_use_msg() -> AssistantMessage {
        make_assistant_msg(StopReason::ToolUse, None)
    }

    fn make_length_msg() -> AssistantMessage {
        make_assistant_msg(StopReason::Length, None)
    }

    // ── is_retryable_assistant_error ──

    #[test]
    fn test_non_error_stop_reason_is_not_retryable() {
        assert!(!is_retryable_assistant_error(&make_stop_msg()));
        assert!(!is_retryable_assistant_error(&make_aborted_msg()));
        assert!(!is_retryable_assistant_error(&make_tool_use_msg()));
        assert!(!is_retryable_assistant_error(&make_length_msg()));
    }

    #[test]
    fn test_error_without_message_is_not_retryable() {
        let msg = make_assistant_msg(StopReason::Error, None);
        assert!(!is_retryable_assistant_error(&msg));
    }

    #[test]
    fn test_error_with_empty_string_is_not_retryable() {
        let msg = make_error_msg("");
        assert!(!is_retryable_assistant_error(&msg));
    }

    #[test]
    fn test_unknown_error_is_not_retryable() {
        let msg = make_error_msg("Some completely unknown error that doesn't match any pattern");
        assert!(!is_retryable_assistant_error(&msg));
    }

    // ── non-retryable patterns ──

    #[test]
    fn test_non_retryable_go_usage_limit_error() {
        assert!(!is_retryable_assistant_error(&make_error_msg(
            "GoUsageLimitError: you've hit your free tier limit"
        )));
    }

    #[test]
    fn test_non_retryable_free_usage_limit_error() {
        assert!(!is_retryable_assistant_error(&make_error_msg(
            "FreeUsageLimitError: upgrade to continue"
        )));
    }

    #[test]
    fn test_non_retryable_monthly_usage_limit() {
        assert!(!is_retryable_assistant_error(&make_error_msg(
            "Monthly usage limit reached, please wait"
        )));
    }

    #[test]
    fn test_non_retryable_insufficient_quota() {
        assert!(!is_retryable_assistant_error(&make_error_msg(
            "insufficient_quota"
        )));
    }

    #[test]
    fn test_non_retryable_out_of_budget() {
        assert!(!is_retryable_assistant_error(&make_error_msg(
            "out of budget"
        )));
    }

    #[test]
    fn test_non_retryable_quota_exceeded() {
        assert!(!is_retryable_assistant_error(&make_error_msg(
            "quota exceeded for this API key"
        )));
    }

    #[test]
    fn test_non_retryable_billing() {
        assert!(!is_retryable_assistant_error(&make_error_msg(
            "billing error: payment required"
        )));
    }

    #[test]
    fn test_non_retryable_available_balance() {
        assert!(!is_retryable_assistant_error(&make_error_msg(
            "no available balance, please add funds"
        )));
    }

    // ── retryable patterns ──

    #[test]
    fn test_retryable_overloaded() {
        assert!(is_retryable_assistant_error(&make_error_msg(
            "upstream service is overloaded"
        )));
    }

    #[test]
    fn test_retryable_rate_limit() {
        assert!(is_retryable_assistant_error(&make_error_msg(
            "rate limit exceeded"
        )));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "rate_limit exceeded"
        )));
    }

    #[test]
    fn test_retryable_too_many_requests() {
        assert!(is_retryable_assistant_error(&make_error_msg(
            "too many requests"
        )));
    }

    #[test]
    fn test_retryable_http_status_codes() {
        assert!(is_retryable_assistant_error(&make_error_msg("HTTP 429")));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "HTTP 500 Internal Server Error"
        )));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "502 Bad Gateway"
        )));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "503 Service Unavailable"
        )));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "504 Gateway Timeout"
        )));
        assert!(is_retryable_assistant_error(&make_error_msg("524")));
    }

    #[test]
    fn test_retryable_service_unavailable() {
        assert!(is_retryable_assistant_error(&make_error_msg(
            "service unavailable"
        )));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "service_unavailable"
        )));
    }

    #[test]
    fn test_retryable_server_error() {
        assert!(is_retryable_assistant_error(&make_error_msg(
            "server error"
        )));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "server_error"
        )));
    }

    #[test]
    fn test_retryable_internal_error() {
        assert!(is_retryable_assistant_error(&make_error_msg(
            "internal error"
        )));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "internal_error"
        )));
    }

    #[test]
    fn test_retryable_provider_returned_error() {
        assert!(is_retryable_assistant_error(&make_error_msg(
            "provider returned error"
        )));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "provider_returned_error"
        )));
    }

    #[test]
    fn test_retryable_network_errors() {
        assert!(is_retryable_assistant_error(&make_error_msg(
            "network error"
        )));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "network_error"
        )));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "connection error"
        )));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "connection refused"
        )));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "connection lost"
        )));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "connection_lost"
        )));
    }

    #[test]
    fn test_retryable_fetch_failed() {
        assert!(is_retryable_assistant_error(&make_error_msg(
            "fetch failed"
        )));
    }

    #[test]
    fn test_retryable_socket_errors() {
        assert!(is_retryable_assistant_error(&make_error_msg(
            "socket hang up"
        )));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "socket connection was closed"
        )));
    }

    #[test]
    fn test_retryable_timeout() {
        assert!(is_retryable_assistant_error(&make_error_msg("timeout")));
        assert!(is_retryable_assistant_error(&make_error_msg("timed out")));
    }

    #[test]
    fn test_retryable_terminated() {
        assert!(is_retryable_assistant_error(&make_error_msg("terminated")));
    }

    #[test]
    fn test_retryable_websocket_errors() {
        assert!(is_retryable_assistant_error(&make_error_msg(
            "websocket closed"
        )));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "websocket error"
        )));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "websocket_error"
        )));
    }

    #[test]
    fn test_retryable_stream_ended_errors() {
        assert!(is_retryable_assistant_error(&make_error_msg(
            "stream ended without message_stop"
        )));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "ended without"
        )));
    }

    #[test]
    fn test_retryable_http2_error() {
        assert!(is_retryable_assistant_error(&make_error_msg(
            "http2 request did not get a response"
        )));
    }

    #[test]
    fn test_retryable_retry_delay() {
        assert!(is_retryable_assistant_error(&make_error_msg(
            "retry delay cap reached"
        )));
    }

    #[test]
    fn test_retryable_retry_suggestion() {
        assert!(is_retryable_assistant_error(&make_error_msg(
            "you can retry your request"
        )));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "please retry your request"
        )));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "try your request again"
        )));
    }

    #[test]
    fn test_retryable_resource_exhausted() {
        assert!(is_retryable_assistant_error(&make_error_msg(
            "ResourceExhausted"
        )));
    }

    #[test]
    fn test_retryable_upstream_connect() {
        assert!(is_retryable_assistant_error(&make_error_msg(
            "upstream connect error"
        )));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "upstream_connect"
        )));
    }

    #[test]
    fn test_retryable_reset_before_headers() {
        assert!(is_retryable_assistant_error(&make_error_msg(
            "reset before headers"
        )));
    }

    #[test]
    fn test_retryable_other_side_closed() {
        assert!(is_retryable_assistant_error(&make_error_msg(
            "other side closed"
        )));
    }

    // ── case insensitivity ──

    #[test]
    fn test_case_insensitive_retryable() {
        assert!(is_retryable_assistant_error(&make_error_msg("TIMEOUT")));
        assert!(is_retryable_assistant_error(&make_error_msg("Overloaded")));
        assert!(is_retryable_assistant_error(&make_error_msg(
            "RATE_LIMIT Exceeded"
        )));
    }

    #[test]
    fn test_case_insensitive_non_retryable() {
        assert!(!is_retryable_assistant_error(&make_error_msg(
            "Insufficient_Quota"
        )));
        assert!(!is_retryable_assistant_error(&make_error_msg("BILLING")));
    }

    // ── non-retryable takes priority over retryable ──

    #[test]
    fn test_non_retryable_priority_over_retryable() {
        // Message contains both "billing" (non-retryable) and "timeout" (retryable)
        let msg = make_error_msg("billing timeout error");
        assert!(!is_retryable_assistant_error(&msg));
    }

    // ── retry_assistant_call ──

    #[tokio::test]
    async fn test_retry_policy_disabled() {
        let call_count = std::sync::atomic::AtomicU32::new(0);
        let result = retry_assistant_call(
            || {
                call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::future::ready(make_stop_msg())
            },
            Some(RetryPolicy {
                enabled: false,
                max_retries: 5,
                base_delay_ms: 1000,
            }),
            None,
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_retry_policy_none() {
        let call_count = std::sync::atomic::AtomicU32::new(0);
        let result = retry_assistant_call(
            || {
                call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::future::ready(make_stop_msg())
            },
            None,
            None,
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_success_on_first_call() {
        let result = retry_assistant_call(
            || std::future::ready(make_stop_msg()),
            Some(RetryPolicy {
                enabled: true,
                max_retries: 3,
                base_delay_ms: 1000,
            }),
            None,
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Stop);
    }

    #[tokio::test]
    async fn test_aborted_not_retried() {
        let call_count = std::sync::atomic::AtomicU32::new(0);
        let result = retry_assistant_call(
            || {
                call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::future::ready(make_aborted_msg())
            },
            Some(RetryPolicy {
                enabled: true,
                max_retries: 3,
                base_delay_ms: 1000,
            }),
            None,
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Aborted);
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_non_retryable_error_not_retried() {
        let call_count = std::sync::atomic::AtomicU32::new(0);
        let result = retry_assistant_call(
            || {
                call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::future::ready(make_error_msg("insufficient_quota"))
            },
            Some(RetryPolicy {
                enabled: true,
                max_retries: 3,
                base_delay_ms: 1000,
            }),
            None,
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_retryable_error_then_success() {
        let call_count = std::sync::atomic::AtomicU32::new(0);
        let result = retry_assistant_call(
            || {
                let n = call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n == 0 {
                    std::future::ready(make_error_msg("timeout"))
                } else {
                    std::future::ready(make_stop_msg())
                }
            },
            Some(RetryPolicy {
                enabled: true,
                max_retries: 3,
                base_delay_ms: 1, // 1ms so the test passes quickly.
            }),
            None,
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_retry_exhausted() {
        let call_count = std::sync::atomic::AtomicU32::new(0);
        let result = retry_assistant_call(
            || {
                call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::future::ready(make_error_msg("timeout"))
            },
            Some(RetryPolicy {
                enabled: true,
                max_retries: 2,
                base_delay_ms: 1, // 1ms so the test passes quickly.
            }),
            None,
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Error);
        // Initial call + 2 retries = 3 attempts
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_cancellation_during_sleep() {
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();

        // Trigger cancellation after retry enters sleep: spawn an async task first, then await retry
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            cancel_clone.cancel();
        });

        let result = retry_assistant_call(
            || std::future::ready(make_error_msg("timeout")),
            Some(RetryPolicy {
                enabled: true,
                max_retries: 3,
                base_delay_ms: 10_000, // 10s, ensuring cancellation lands before sleep finishes
            }),
            Some(cancel),
        )
        .await;

        assert_eq!(result.stop_reason, StopReason::Aborted);
        assert_eq!(result.error_message, None);
    }

    #[tokio::test]
    async fn test_zero_retries() {
        let call_count = std::sync::atomic::AtomicU32::new(0);
        let result = retry_assistant_call(
            || {
                call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::future::ready(make_error_msg("timeout"))
            },
            Some(RetryPolicy {
                enabled: true,
                max_retries: 0,
                base_delay_ms: 1000,
            }),
            None,
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_tool_use_not_retried() {
        let call_count = std::sync::atomic::AtomicU32::new(0);
        let result = retry_assistant_call(
            || {
                call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::future::ready(make_tool_use_msg())
            },
            Some(RetryPolicy {
                enabled: true,
                max_retries: 3,
                base_delay_ms: 1000,
            }),
            None,
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::ToolUse);
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_length_not_retried() {
        let call_count = std::sync::atomic::AtomicU32::new(0);
        let result = retry_assistant_call(
            || {
                call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::future::ready(make_length_msg())
            },
            Some(RetryPolicy {
                enabled: true,
                max_retries: 3,
                base_delay_ms: 1000,
            }),
            None,
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Length);
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_unknown_error_not_retried() {
        let call_count = std::sync::atomic::AtomicU32::new(0);
        let result = retry_assistant_call(
            || {
                call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::future::ready(make_error_msg(
                    "some weird error that doesn't match any pattern",
                ))
            },
            Some(RetryPolicy {
                enabled: true,
                max_retries: 3,
                base_delay_ms: 1000,
            }),
            None,
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
