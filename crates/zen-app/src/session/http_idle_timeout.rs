// REQ-AI-02 (pi-101 / pi httpIdleTimeoutMs): app-layer HTTP idle-timeout injection into the request boundary
//
//! Injection point for pi's `httpIdleTimeoutMs` (HTTP-body idle watchdog), mirroring `sdk.ts`'s `streamFn`, which reads
//! the setting per request (`reference/pi-original/packages/coding-agent/src/core/sdk.ts:302-308`) and passes it to the
//! provider SDK.
//!
//! Layering: the zen-ai request layer owns the mechanism (`StreamConfig.idle_timeout_ms` → the byte-level
//! `IdleBodyStream` watchdog inside `stream()`); the settings value lives in the app layer. This decorator is the
//! single place where the two meet, so every LLM call that goes through the session's `StreamFn` (agent turns,
//! compaction, branch summaries) gets the watchdog without any of them knowing about settings.
//!
//! Caller precedence follows pi (`options?.timeoutMs ?? settings…`): a `StreamConfig` that already carries
//! `idle_timeout_ms` keeps it — including `Some(0)`, which explicitly disables the watchdog for that call.

use std::sync::Arc;

use zen_ai::types::StreamFn;

/// Wrap `inner` so that every request carries the resolved HTTP idle timeout.
///
/// # Parameters
/// - `inner`: the stream implementation being decorated (typically already wrapped by
///   [`crate::session::provider_attribution::with_provider_attribution`]).
/// - `idle_timeout_ms`: resolved setting, already folded by
///   [`crate::config::defaults::resolve_http_idle_timeout_ms`] (`None` = disabled; `Some(ms)` = byte-level idle limit).
///
/// # Behavior
/// - A `StreamConfig` whose `idle_timeout_ms` is `None` (unset) gets this value written into it.
/// - A caller-supplied `idle_timeout_ms` (including `Some(0)` = explicit disable) wins, aligned with pi's
///   `options?.timeoutMs ?? settings` precedence.
/// - `cfg == None` (no request config from the caller) is materialized with the value, so the watchdog is never
///   silently skipped.
/// - `idle_timeout_ms == None` leaves the request untouched (watchdog disabled by setting).
///
/// # Consumers
/// `session::services::create_agent_session` (assembly); unit tests below.
pub(crate) fn with_http_idle_timeout(inner: StreamFn, idle_timeout_ms: Option<u64>) -> StreamFn {
    Arc::new(move |model, context, cfg, cancel| {
        let inner = inner.clone();
        let model = model.clone();
        let context = context.clone();
        let cfg = match idle_timeout_ms {
            None => cfg,
            Some(idle) => {
                let mut config = cfg.unwrap_or_default();
                if config.idle_timeout_ms.is_none() {
                    config.idle_timeout_ms = Some(idle);
                }
                Some(config)
            }
        };
        Box::pin(async move { inner(&model, &context, cfg, cancel).await })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tokio_util::sync::CancellationToken;
    use zen_ai::api::openai_stream::{StreamConfig, StreamError};
    use zen_ai::types::{AssistantMessageEvent, Context, Model};

    /// Records the `StreamConfig` each request was made with (the decorator's only observable effect).
    fn recording_stream_fn(seen: Arc<Mutex<Vec<Option<StreamConfig>>>>) -> StreamFn {
        Arc::new(
            move |_model: &Model,
                  _context: &Context,
                  cfg: Option<StreamConfig>,
                  _cancel: Option<CancellationToken>| {
                let seen = Arc::clone(&seen);
                Box::pin(async move {
                    seen.lock().unwrap().push(cfg);
                    let (_tx, rx) = tokio::sync::mpsc::unbounded_channel::<AssistantMessageEvent>();
                    Ok::<_, StreamError>(rx)
                })
            },
        )
    }

    fn test_model() -> Model {
        Model {
            id: "m".into(),
            name: None,
            provider: "opencode-go".into(),
            api: zen_ai::provider::Api::OpenAiCompletions,
            base_url: "https://example.invalid".into(),
            context_window: 1000,
            max_tokens: 100,
            reasoning: false,
            headers: None,
            cost_rates: None,
            compat: None,
        }
    }

    fn context() -> Context {
        Context {
            system_prompt: None,
            messages: Vec::new(),
            tools: None,
        }
    }

    async fn run(fn_: &StreamFn, cfg: Option<StreamConfig>) -> Option<StreamConfig> {
        let _ = fn_(&test_model(), &context(), cfg, None).await;
        None
    }

    #[tokio::test]
    async fn decorator_writes_the_setting_into_an_unset_config() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let wrapped = with_http_idle_timeout(recording_stream_fn(Arc::clone(&seen)), Some(300_000));

        run(&wrapped, Some(StreamConfig::default())).await;

        let recorded = seen.lock().unwrap().clone();
        assert_eq!(recorded.len(), 1);
        assert_eq!(
            recorded[0].as_ref().expect("cfg present").idle_timeout_ms,
            Some(300_000)
        );
    }

    #[tokio::test]
    async fn decorator_preserves_a_caller_supplied_value() {
        // pi precedence: `options?.timeoutMs ?? settings…` — the caller wins.
        let seen = Arc::new(Mutex::new(Vec::new()));
        let wrapped = with_http_idle_timeout(recording_stream_fn(Arc::clone(&seen)), Some(300_000));
        let caller = StreamConfig {
            idle_timeout_ms: Some(1_000),
            ..Default::default()
        };

        run(&wrapped, Some(caller)).await;

        assert_eq!(
            seen.lock().unwrap()[0].as_ref().unwrap().idle_timeout_ms,
            Some(1_000)
        );
    }

    #[tokio::test]
    async fn decorator_materializes_a_missing_config() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let wrapped = with_http_idle_timeout(recording_stream_fn(Arc::clone(&seen)), Some(60_000));

        run(&wrapped, None).await;

        assert_eq!(
            seen.lock().unwrap()[0]
                .as_ref()
                .expect("cfg materialized")
                .idle_timeout_ms,
            Some(60_000)
        );
    }

    #[tokio::test]
    async fn disabled_setting_leaves_the_request_untouched() {
        // `httpIdleTimeoutMs: 0` → pi maps it to "effectively infinite"; zen expresses the same as "no watchdog".
        let seen = Arc::new(Mutex::new(Vec::new()));
        let wrapped = with_http_idle_timeout(recording_stream_fn(Arc::clone(&seen)), None);

        run(&wrapped, Some(StreamConfig::default())).await;

        let recorded = seen.lock().unwrap().clone();
        assert_eq!(recorded[0].as_ref().unwrap().idle_timeout_ms, None);
    }
}
