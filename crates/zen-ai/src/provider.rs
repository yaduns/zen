//! Multi-provider interface and built-in implementations (Phase 1: opencode-go default + huoshan).
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

use crate::types::Model;
use crate::utils::provider_env::get_provider_env_value;

/// Protocol type. Aligned with pi's `Api` (types.ts KnownApi). zen currently only supports
/// OpenAI Chat Completions compatible protocol (both providers use it); add variants when integrating new protocols.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum Api {
    /// OpenAI Chat Completions compatible protocol
    OpenAiCompletions,
    // Placeholder: enable when implementing the OpenAI Responses protocol in the future
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingFormat {
    /// Aligned with pi compat.thinkingFormat "openai": only reasoning_effort, no thinking toggle
    OpenAi,
    /// Aligned with pi compat.thinkingFormat "deepseek": thinking:{type:enabled/disabled} + reasoning_effort
    DeepSeek,
}

/// Called by `build_request_body` before each turn in interactive mode: resolves the thinking-toggle format family based on the provider name,
/// Determines whether to append the `thinking:{type:...}` parameter to the request body (DeepSeek-family endpoints require an explicit toggle to enable/disable thinking).
/// Unknown providers always fall back to `OpenAi` (reasoning_effort only, matching pi detectCompat's default branch,
/// opencode-completions.ts:1304——opencode-go classifies a baseUrl starting with opencode.ai as openai format).
pub fn thinking_format_for_provider(provider_name: &str) -> ThinkingFormat {
    match provider_name {
        "huoshan" => ThinkingFormat::DeepSeek,
        _ => ThinkingFormat::OpenAi,
    }
}

/// Multi-provider extension interface. Aligns with pi's `Provider<TApi>` (models.ts:67).
///
/// Association with `Model`: `Provider::id()` string-equals `Model.provider` (`String`)
/// means the model belongs to that provider.
///
/// Auth: `resolve_api_key` currently reads from the provider's own env var (e.g. `OPENCODE_API_KEY`);
/// When auth.json support is added later, add "file > env" priority logic inside this method; signature stays unchanged.
pub trait Provider: Send + Sync {
    /// Provider identifier (e.g. "opencode-go"). String equality with `Model.provider` establishes the association.
    fn id(&self) -> &str;
    /// Protocol type (currently only `OpenAiCompletions` for zen)
    fn api(&self) -> Api;
    /// API endpoint base URL (without the /chat/completions path). Returns an owned `String`
    /// to support env overrides (e.g. `OPENCODE_BASE_URL`).
    fn base_url(&self) -> String;
    /// Resolve the API key (read from the provider's own env var)
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

/// Resolve a provider by name; returns `None` for unknown names.
/// The future `--provider` CLI arg (T-AP-08-1) reuses this function.
pub fn resolve_provider(name: &str) -> Option<Box<dyn Provider>> {
    match name {
        "opencode-go" => Some(Box::new(OpenCodeGoProvider)),
        "huoshan" => Some(Box::new(HuoshanProvider)),
        _ => None,
    }
}

/// opencode session header name (required by the gateway; missing → 400 `MissingSessionID`).
const OPENCODE_SESSION_HEADER: &str = "x-opencode-session";
/// client identifier header name (for gateway attribution, not required).
const OPENCODE_CLIENT_HEADER: &str = "x-opencode-client";
/// `x-opencode-client` value: pi passes `"pi"`; zen is a separate implementation and passes `"zen"`.
const ZEN_CLIENT_ID: &str = "zen";

/// OpenCode Go (default provider). Key: `OPENCODE_API_KEY`; endpoint: `https://opencode.ai/zen/go/v1`
/// (overridable via `OPENCODE_BASE_URL`).
pub struct OpenCodeGoProvider;

impl Provider for OpenCodeGoProvider {
    fn id(&self) -> &str {
        "opencode-go"
    }

    fn api(&self) -> Api {
        Api::OpenAiCompletions
    }

    fn base_url(&self) -> String {
        get_provider_env_value("OPENCODE_BASE_URL", None)
            .unwrap_or_else(|| "https://opencode.ai/zen/go/v1".to_string())
    }

    fn resolve_api_key(&self) -> Option<String> {
        get_provider_env_value("OPENCODE_API_KEY", None)
    }

    /// The opencode gateway requires a session header; if missing, returns 400 `MissingSessionID`.
    /// Aligns with pi `provider-attribution.ts:getSessionHeaders` (`if (!sessionId) return undefined`;
    /// empty string is also falsy → not sent).
    /// **Registered deviation**: pi also checks `provider === "opencode"` (old name) and `baseUrl.hostname === "opencode.ai"`
    /// Two branches—the zen registry is closed and has no old-name alias, so judging by provider name (this impl) is already equivalent; neither branch is implemented;
    /// The branding/telemetry headers from `getDefaultAttributionHeaders` belong to multi-provider attribution and remain omitted per the scope.md omission table.
    fn request_headers(
        &self,
        _model: &Model,
        session_id: Option<&str>,
    ) -> Option<HashMap<String, String>> {
        let session_id = session_id.filter(|id| !id.is_empty())?;
        Some(HashMap::from([
            (OPENCODE_SESSION_HEADER.to_string(), session_id.to_string()),
            (
                OPENCODE_CLIENT_HEADER.to_string(),
                ZEN_CLIENT_ID.to_string(),
            ),
        ]))
    }
}

/// Volcengine Ark (huoshan). Key: `ARK_API_KEY`; endpoint: `https://ark.cn-beijing.volces.com/api/coding/v3`
/// (overridable via `ARK_BASE_URL`).
pub struct HuoshanProvider;

impl Provider for HuoshanProvider {
    fn id(&self) -> &str {
        "huoshan"
    }

    fn api(&self) -> Api {
        Api::OpenAiCompletions
    }

    fn base_url(&self) -> String {
        get_provider_env_value("ARK_BASE_URL", None)
            .unwrap_or_else(|| "https://ark.cn-beijing.volces.com/api/coding/v3".to_string())
    }

    fn resolve_api_key(&self) -> Option<String> {
        get_provider_env_value("ARK_API_KEY", None)
    }
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

    // ── thinking_format_for_provider ──

    #[test]
    fn test_thinking_format_huoshan_is_deepseek() {
        // Volcano Engine Ark DeepSeek-family endpoints require an explicit thinking:{type} toggle
        assert_eq!(
            thinking_format_for_provider("huoshan"),
            ThinkingFormat::DeepSeek
        );
    }

    #[test]
    fn test_thinking_format_opencode_go_is_openai() {
        // opencode-go matches pi detectCompat: openai format, reasoning_effort only
        assert_eq!(
            thinking_format_for_provider("opencode-go"),
            ThinkingFormat::OpenAi
        );
    }

    #[test]
    fn test_thinking_format_unknown_falls_back_openai() {
        assert_eq!(
            thinking_format_for_provider("nonexistent"),
            ThinkingFormat::OpenAi
        );
    }

    // ── request_headers (opencode session header, opencode-session-header-fix) ──

    #[test]
    fn test_request_headers_opencode_with_session() {
        let provider = OpenCodeGoProvider;
        let model = Model::from_provider(&provider);
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
        let provider = OpenCodeGoProvider;
        let model = Model::from_provider(&provider);
        // No session context / empty string (pi falsy semantics) → do not fabricate; return None
        assert!(provider.request_headers(&model, None).is_none());
        assert!(provider.request_headers(&model, Some("")).is_none());
    }

    #[test]
    fn test_request_headers_huoshan_walks_default_none() {
        let provider = HuoshanProvider;
        let model = Model::from_provider(&provider);
        // Not overridden → default implementation returns None (huoshan needs no extra headers)
        assert!(provider
            .request_headers(&model, Some("session-abc"))
            .is_none());
    }
}
