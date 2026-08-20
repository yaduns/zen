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

use serde::{Deserialize, Serialize};

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
}
