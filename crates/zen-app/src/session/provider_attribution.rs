// REQ-AI-02 (pi-101 / pi provider-attribution): app-layer provider attribution header injection
//
//! Provider attribution header injection point (matches pi `coding-agent/src/core/provider-attribution.ts` +
//! `sdk.ts::transformHeaders`）。
//!
//! T-AI-02-10 moves the trigger point for "provider self-describing request headers" from the zen-ai request layer
//! (`resolve_provider` in `api/openai_stream.rs::merge_request_headers`) to the app assembly layer:
//! the request layer stays pure data (`model.headers` + compat session affinity + `cfg.headers`),
//! dynamic attribution headers (e.g. `x-opencode-session` enforced by the opencode gateway) are wrapped by this module one layer outside `StreamFn`,
//! evaluated per request from the current `model` + `cfg.session_id`, then merged into `StreamConfig.headers`.
//!
//! Transitional: the provider source is currently `resolve_provider` (built-in catalog); after T-AI-02-11 injects the single `Catalog`
//! instance, switch to that instance's provider (seam already registered, see `tasks/T-AI-02-10-implemented.md`).

use std::collections::HashMap;
use std::sync::Arc;

use zen_ai::provider::resolve_provider;
use zen_ai::types::{Model, StreamFn};

/// Compute provider self-describing attribution headers for a request (e.g. opencode's `x-opencode-session`/`x-opencode-client`).
///
/// the provider implementation (`Provider::request_headers`) decides whether to include headers; unknown provider / no session context
/// returns an empty map (does not fabricate a session id).
fn provider_attribution_headers(
    model: &Model,
    session_id: Option<&str>,
) -> HashMap<String, String> {
    resolve_provider(&model.provider)
        .and_then(|provider| provider.request_headers(model, session_id))
        .unwrap_or_default()
}

/// Called during assembly: wraps the caller-provided `StreamFn` with a decorator that injects provider attribution headers on every request.
///
/// Attribution headers are merged **before** the caller's `cfg.headers` (caller explicit headers take precedence); when there are no headers, `cfg` is left unchanged.
/// This way all sessions assembled through this layer (CLI print/interactive, examples) get the opencode session headers;
/// direct low-level zen-ai calls (`stream`/`stream_collect`) do not include the headers and must provide them via `cfg.headers`.
pub(crate) fn with_provider_attribution(inner: StreamFn) -> StreamFn {
    Arc::new(move |model, context, cfg, cancel| {
        let inner = inner.clone();
        let model = model.clone();
        let context = context.clone();
        let cfg = cfg.map(|mut config| {
            let mut merged = provider_attribution_headers(&model, config.session_id.as_deref());
            if let Some(caller) = config.headers.take() {
                for (key, value) in caller {
                    merged.insert(key, value);
                }
            }
            if !merged.is_empty() {
                config.headers = Some(merged);
            }
            config
        });
        Box::pin(async move { inner(&model, &context, cfg, cancel).await })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tokio_util::sync::CancellationToken;
    use zen_ai::api::openai_stream::{StreamConfig, StreamError};
    use zen_ai::types::{AssistantMessageEvent, Context};

    fn opencode_model() -> Model {
        zen_ai::catalog::Catalog::embedded()
            .model("opencode-go", "deepseek-v4-flash")
            .expect("Built-in catalog contains opencode-go/deepseek-v4-flash")
    }

    #[test]
    fn test_attribution_headers_opencode_with_session() {
        let headers = provider_attribution_headers(&opencode_model(), Some("s-1"));
        assert_eq!(
            headers.get("x-opencode-session").map(String::as_str),
            Some("s-1")
        );
        assert_eq!(
            headers.get("x-opencode-client").map(String::as_str),
            Some("zen")
        );
    }

    #[test]
    fn test_attribution_headers_opencode_without_session() {
        for session in [None, Some("")] {
            let headers = provider_attribution_headers(&opencode_model(), session);
            assert!(!headers.contains_key("x-opencode-session"));
            // static client identification headers are still sent with the directory as usual
            assert_eq!(
                headers.get("x-opencode-client").map(String::as_str),
                Some("zen")
            );
        }
    }

    #[test]
    fn test_attribution_headers_unknown_provider_is_empty() {
        let model = Model {
            provider: "unknown-provider".into(),
            ..opencode_model()
        };
        assert!(provider_attribution_headers(&model, Some("s-1")).is_empty());
    }

    /// Records the headers received by a request; returns `(wrapped, captured)`.
    #[allow(clippy::type_complexity)]
    fn recording_stream_fn() -> (
        StreamFn,
        Arc<Mutex<Option<Option<HashMap<String, String>>>>>,
    ) {
        let captured: Arc<Mutex<Option<Option<HashMap<String, String>>>>> =
            Arc::new(Mutex::new(None));
        let cap = captured.clone();
        let inner: StreamFn = Arc::new(move |_model, _ctx, cfg, _cancel| {
            *cap.lock().unwrap() = Some(cfg.and_then(|c| c.headers));
            Box::pin(async move {
                let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<AssistantMessageEvent>();
                drop(tx);
                Ok::<_, StreamError>(rx)
            })
        });
        (with_provider_attribution(inner), captured)
    }

    async fn call_once(wrapped: &StreamFn, model: &Model, cfg: StreamConfig) {
        let ctx = Context {
            system_prompt: None,
            messages: vec![],
            tools: None,
        };
        let _ = wrapped(model, &ctx, Some(cfg), Some(CancellationToken::new())).await;
    }

    #[tokio::test]
    async fn test_wrapper_injects_attribution_headers() {
        let (wrapped, captured) = recording_stream_fn();
        call_once(
            &wrapped,
            &opencode_model(),
            StreamConfig {
                session_id: Some("s-1".into()),
                ..Default::default()
            },
        )
        .await;
        let headers = captured.lock().unwrap().clone().expect("captured cfg");
        let headers = headers.expect("session present should have header");
        assert_eq!(
            headers.get("x-opencode-session").map(String::as_str),
            Some("s-1")
        );
    }

    #[tokio::test]
    async fn test_wrapper_caller_headers_override_attribution() {
        let (wrapped, captured) = recording_stream_fn();
        call_once(
            &wrapped,
            &opencode_model(),
            StreamConfig {
                session_id: Some("s-1".into()),
                headers: Some(HashMap::from([(
                    "x-opencode-session".to_string(),
                    "caller".to_string(),
                )])),
                ..Default::default()
            },
        )
        .await;
        let headers = captured
            .lock()
            .unwrap()
            .clone()
            .expect("captured cfg")
            .expect("Has header");
        assert_eq!(
            headers.get("x-opencode-session").map(String::as_str),
            Some("caller")
        );
    }

    #[tokio::test]
    async fn test_wrapper_leaves_headers_none_when_nothing_to_add() {
        let (wrapped, captured) = recording_stream_fn();
        let model = Model {
            provider: "unknown-provider".into(),
            ..opencode_model()
        };
        call_once(&wrapped, &model, StreamConfig::default()).await;
        assert!(
            captured
                .lock()
                .unwrap()
                .clone()
                .expect("captured cfg")
                .is_none(),
            "When no attribution header and no caller header, headers remain None"
        );
    }
}
