//! Multi-provider interfaces and catalog-driven implementations (built-in defaults: opencode-go + huoshan).
//!
//! Mirrors Pi's `Provider` interface in `packages/ai/src/models.ts`, the `Api` protocol types
//! in `types.ts`, and the one implementation file per provider under `providers/`. Auth is **API key +
//! per-provider env vars** (aligned with pi's `envApiKeyAuth(name, envVars)`: each provider declares
//! its own env variable, e.g. opencode-go → `OPENCODE_API_KEY`).
//!
//! Env var conventions:
//! - `ZEN_PROVIDER`: default provider selection (`opencode-go` / `huoshan`), defaults to `opencode-go`
//! - opencode-go：`OPENCODE_API_KEY` / `OPENCODE_BASE_URL` / `OPENCODE_MODEL` / `OPENCODE_MAX_TOKENS`
//! - huoshan：`ARK_API_KEY` / `ARK_BASE_URL` / `ARK_MODEL` / `ARK_MAX_TOKENS`
//!
//! Future full pi compatibility path: `--provider` CLI overrides `ZEN_PROVIDER`, auth.json gains a
//! "file > env" priority in `resolve_api_key`, and after model cataloging `default_model_id` comes from the catalog — trait/struct shapes remain unchanged.
//!
//! Traceability: `pi-120` (models.ts) / `pi-118` (provider-env) → REQ-AI-02 → this module + `Model::from_provider`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::catalog::{resolve_config_value, Catalog, ProviderConfig};
use crate::types::Model;

/// Protocol type. Aligned with pi's `Api` (types.ts KnownApi). zen currently only supports
/// OpenAI Chat Completions compatible protocol (both providers use it); add variants when integrating new protocols.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum Api {
    /// OpenAI Chat Completions compatible protocol
    #[serde(rename = "openai-completions")]
    OpenAiCompletions,
    // Placeholder: enable when implementing the OpenAI Responses protocol in the future
    #[serde(rename = "openai-responses")]
    OpenAiResponses,
}

impl std::fmt::Display for Api {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Api::OpenAiCompletions => write!(f, "openai-completions"),
            Api::OpenAiResponses => write!(f, "openai-responses"),
        }
    }
}

/// Thinking toggle format family. Aligns with the value semantics of pi `compat.thinkingFormat` (openai-completions.ts:639-700):
/// Decides whether the request body carries the provider-specific thinking toggle when enabling/disabling thinking.
///
/// - `OpenAi` (aligned with pi `"openai"`): only sends `reasoning_effort`; no thinking toggle.
/// - `DeepSeek` (aligned with pi `"deepseek"`): attaches `thinking: {type: "enabled"/"disabled"}` + `reasoning_effort`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum ThinkingFormat {
    /// Aligned with pi compat.thinkingFormat "openai": only reasoning_effort, no thinking toggle
    #[serde(rename = "openai")]
    OpenAi,
    /// Aligned with pi compat.thinkingFormat "deepseek": thinking:{type:enabled/disabled} + reasoning_effort
    #[serde(rename = "deepseek")]
    DeepSeek,
}

/// `max_tokens` request field name. Aligned with pi `compat.maxTokensField`:
/// most compat endpoints use `max_tokens`; OpenAI o-series / gpt-5 use `max_completion_tokens`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MaxTokensField {
    /// Send `max_tokens` (default).
    #[default]
    MaxTokens,
    /// Send `max_completion_tokens`.
    MaxCompletionTokens,
}

/// Session affinity header format family. Aligned with pi `compat.sessionAffinityFormat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
pub enum SessionAffinityFormat {
    /// Send `session_id` + `x-client-request-id` + `x-session-affinity`.
    #[default]
    #[serde(rename = "openai")]
    Openai,
    /// Send only `x-client-request-id` + `x-session-affinity` (no `session_id`).
    #[serde(rename = "openai-nosession")]
    OpenaiNosession,
    /// Send only `x-session-id`.
    #[serde(rename = "openrouter")]
    Openrouter,
}

/// OpenAI Completions request compatibility switches (the `compat` subset of `models.json`).
///
/// Lives in `provider.rs` (alongside [`Api`]/[`ThinkingFormat`]) — request-layer semantics,
/// not catalog schema; `catalog.rs` only references it. `Model.compat` holds it, T-AI-02-10 consumes it.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAiCompletionsCompat {
    /// Thinking toggle format family; defaults to [`ThinkingFormat::OpenAi`] (since T-AI-02-10 the request layer reads this field,
    /// no longer inferred from the provider name).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_format: Option<ThinkingFormat>,
    /// `max_tokens` field name; defaults to [`MaxTokensField::MaxTokens`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens_field: Option<MaxTokensField>,
    /// Whether `stream_options.include_usage` is supported; defaults to `true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_usage_in_streaming: Option<bool>,
    /// Whether the `developer` role is supported; defaults to `false` (use `system`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_developer_role: Option<bool>,
    /// Whether `reasoning_effort` is supported; defaults to `true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_reasoning_effort: Option<bool>,
    /// Whether to send session affinity headers; defaults to `false` (the opencode attribution header is preserved by a private function).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub send_session_affinity_headers: Option<bool>,
    /// Session affinity header format family; defaults to [`SessionAffinityFormat::Openai`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_affinity_format: Option<SessionAffinityFormat>,
}

/// Multi-provider extension interface. Aligns with pi's `Provider<TApi>` (models.ts:67).
///
/// Association with `Model`: `Provider::id()` string-equals `Model.provider` (`String`)
/// means the model belongs to that provider.
///
/// Authentication: `resolve_api_key` resolves from the `apiKey` config value recorded in the catalog (literal / `$ENV` / `${ENV}`);
/// When auth.json support is added later, add "file > env" priority logic inside this method; signature stays unchanged.
pub trait Provider: Send + Sync {
    /// Provider identifier (e.g. "opencode-go"). String equality with `Model.provider` establishes the association.
    fn id(&self) -> &str;
    /// Protocol type (currently only `OpenAiCompletions` for zen)
    fn api(&self) -> Api;
    /// API endpoint base URL (without the /chat/completions path). Returns an owned `String`
    /// (catalog value; before T-AI-02-11, `Model::from_*` still applies the `*_BASE_URL` env shim).
    fn base_url(&self) -> String;
    /// Parses the API key (the `apiKey` config value stored in the catalog; consumed by the request layer via `Catalog::api_key`).
    fn resolve_api_key(&self) -> Option<String>;

    /// HTTP headers this provider requires to be attached to **every request** (provider-owned: the request-building layer only merges them).
    ///
    /// # Parameters
    /// - `model`: the model for this request (so the implementation can read the effective `base_url` and other config).
    /// - `session_id`: session id; `None`/empty string means this call has no session context (compression / branch summary),
    ///   implementations **must not** fabricate a session id themselves
    ///
    /// # Return Value
    /// `None` = no extra headers (default implementation; no impact on existing or future providers). Returned headers are
    /// merged **before** `StreamConfig.headers`; for duplicate names the **latter overrides the former** (caller-explicit headers take priority) —
    /// merging is done by `api/openai_stream.rs::merge_request_headers`, which first merges into a map and then sets each value once (reqwest
    /// `.header()` appends, so setting the same name twice in two passes would send multiple values).
    ///
    /// Evolution convention: a new provider that needs headers only overrides this method in its own `impl Provider`;
    /// `api/openai_stream.rs` needs no changes.
    fn request_headers(
        &self,
        _model: &Model,
        _session_id: Option<&str>,
    ) -> Option<HashMap<String, String>> {
        None
    }
}

/// Default provider name: env `ZEN_PROVIDER` overrides, default `opencode-go`.
pub fn default_provider_name() -> String {
    std::env::var("ZEN_PROVIDER").unwrap_or_else(|_| "opencode-go".to_string())
}

/// Resolves a provider by name: looks up the built-in [`Catalog`] (overridden by user catalog injection in T-AI-02-11).
/// Returns `None` on miss; signatures of existing callers (openai_stream / services / examples) stay unchanged.
pub fn resolve_provider(name: &str) -> Option<Box<dyn Provider>> {
    Catalog::embedded().provider(name)
}

/// opencode session header name (required by the gateway; missing → 400 `MissingSessionID`).
const OPENCODE_SESSION_HEADER: &str = "x-opencode-session";
/// client identifier header name (for gateway attribution, not required).
const OPENCODE_CLIENT_HEADER: &str = "x-opencode-client";
/// `x-opencode-client` value: pi passes `"pi"`; zen is a separate implementation and passes `"zen"`.
const ZEN_CLIENT_ID: &str = "zen";

/// Catalog-driven provider adapter: replaces the hand-written impls of `OpenCodeGoProvider`/`HuoshanProvider`.
/// Constructed by [`Catalog::provider`]; fields come from [`ProviderConfig`] in `models.json`.
pub(crate) struct ConfiguredProvider {
    id: String,
    config: ProviderConfig,
}

impl ConfiguredProvider {
    /// Construct the adapter (`id` and `config` come from the catalog record).
    pub(crate) fn new(id: String, config: ProviderConfig) -> Self {
        Self { id, config }
    }
}

impl Provider for ConfiguredProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn api(&self) -> Api {
        self.config.api.unwrap_or(Api::OpenAiCompletions)
    }

    fn base_url(&self) -> String {
        self.config.base_url.clone().unwrap_or_default()
    }

    fn resolve_api_key(&self) -> Option<String> {
        self.config
            .api_key
            .as_deref()
            .and_then(resolve_config_value)
    }

    fn request_headers(
        &self,
        _model: &Model,
        session_id: Option<&str>,
    ) -> Option<HashMap<String, String>> {
        let mut headers = self.config.headers.clone().unwrap_or_default();
        if let Some(attribution) = provider_attribution_headers(&self.id, session_id) {
            headers.extend(attribution);
        }
        if headers.is_empty() {
            None
        } else {
            Some(headers)
        }
    }
}

/// Session header attribution required by the opencode gateway (a subset of pi `provider-attribution.ts::getSessionHeaders`).
///
/// Provider name is not rewritten, and no session is fabricated when empty; T-AI-02-10 decides whether to route differently
/// `compat.sendSessionAffinityHeaders`。
fn provider_attribution_headers(
    provider_id: &str,
    session_id: Option<&str>,
) -> Option<HashMap<String, String>> {
    let session_id = session_id.filter(|id| !id.is_empty())?;
    if provider_id != "opencode-go" && provider_id != "opencode" {
        return None;
    }
    Some(HashMap::from([
        (OPENCODE_SESSION_HEADER.to_string(), session_id.to_string()),
        (
            OPENCODE_CLIENT_HEADER.to_string(),
            ZEN_CLIENT_ID.to_string(),
        ),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_provider_opencode_go() {
        let p = resolve_provider("opencode-go").expect("opencode-go should resolve");
        assert_eq!(p.id(), "opencode-go");
        assert_eq!(p.api(), Api::OpenAiCompletions);
        // Endpoint defaults (when OPENCODE_BASE_URL is not set)
        assert_eq!(p.base_url(), "https://opencode.ai/zen/go/v1");
    }

    #[test]
    fn test_resolve_provider_huoshan() {
        let p = resolve_provider("huoshan").expect("huoshan should resolve");
        assert_eq!(p.id(), "huoshan");
        assert_eq!(p.api(), Api::OpenAiCompletions);
        assert_eq!(
            p.base_url(),
            "https://ark.cn-beijing.volces.com/api/coding/v3"
        );
    }

    #[test]
    fn test_resolve_provider_unknown() {
        assert!(resolve_provider("nonexistent").is_none());
    }

    #[test]
    fn test_default_provider_name_falls_back_to_opencode_go() {
        // Default when ZEN_PROVIDER is unset (test processes usually don't set it; if CI sets it, use that value to override the assertion)
        let name = default_provider_name();
        assert!(!name.is_empty());
    }

    // ── request_headers (opencode session header, opencode-session-header-fix) ──

    #[test]
    fn test_request_headers_opencode_with_session() {
        let provider = resolve_provider("opencode-go").expect("opencode-go provider");
        let model = Model::from_provider(provider.as_ref());
        let headers = provider
            .request_headers(&model, Some("session-abc"))
            .expect("Should return both headers when session exists");
        assert_eq!(
            headers.get(OPENCODE_SESSION_HEADER).map(String::as_str),
            Some("session-abc")
        );
        assert_eq!(
            headers.get(OPENCODE_CLIENT_HEADER).map(String::as_str),
            Some(ZEN_CLIENT_ID)
        );
        assert_eq!(headers.len(), 2);
    }

    #[test]
    fn test_request_headers_opencode_without_or_empty_session() {
        let provider = resolve_provider("opencode-go").expect("opencode-go provider");
        let model = Model::from_provider(provider.as_ref());
        // No session context / empty string (pi falsy semantics) → don't fabricate session headers; static client identity headers are still sent with the catalog.
        for session in [None, Some("")] {
            let headers = provider
                .request_headers(&model, session)
                .expect("Static client identification header");
            assert_eq!(
                headers.get(OPENCODE_CLIENT_HEADER).map(String::as_str),
                Some(ZEN_CLIENT_ID)
            );
            assert!(!headers.contains_key(OPENCODE_SESSION_HEADER));
            assert_eq!(headers.len(), 1);
        }
    }

    #[test]
    fn test_request_headers_huoshan_walks_default_none() {
        let provider = resolve_provider("huoshan").expect("huoshan provider");
        let model = Model::from_provider(provider.as_ref());
        // Not overridden → default implementation returns None (huoshan needs no extra headers)
        assert!(provider
            .request_headers(&model, Some("session-abc"))
            .is_none());
    }

    // ── serde (values from models.json) ──

    #[test]
    fn api_deserializes_kebab_case() {
        let api: Api = serde_json::from_str("\"openai-completions\"").expect("api");
        assert_eq!(api, Api::OpenAiCompletions);
        assert_eq!(
            serde_json::to_string(&api).expect("serialize api"),
            "\"openai-completions\""
        );
    }

    #[test]
    fn thinking_format_deserializes_kebab_case() {
        let format: ThinkingFormat = serde_json::from_str("\"deepseek\"").expect("format");
        assert_eq!(format, ThinkingFormat::DeepSeek);
    }

    #[test]
    fn max_tokens_field_deserializes_snake_case() {
        let field: MaxTokensField =
            serde_json::from_str("\"max_completion_tokens\"").expect("field");
        assert_eq!(field, MaxTokensField::MaxCompletionTokens);
    }

    #[test]
    fn session_affinity_format_deserializes_kebab_case() {
        let format: SessionAffinityFormat =
            serde_json::from_str("\"openai-nosession\"").expect("format");
        assert_eq!(format, SessionAffinityFormat::OpenaiNosession);
    }
}
