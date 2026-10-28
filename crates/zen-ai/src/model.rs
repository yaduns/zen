use std::collections::HashMap;

use crate::catalog::Catalog;
use crate::provider::Provider;
use crate::types::{Cost, CostRates, Model};

impl Model {
    /// Selects the default model from the current catalog based on `ZEN_PROVIDER` (default provider `opencode-go`, model
    /// `deepseek-v4-flash`）。
    ///
    /// provider selection: env `ZEN_PROVIDER` (default `opencode-go`) → the current catalog [`Catalog::current`]'s
    /// default model (includes user `models.json` overrides if installed during `main` startup; embedded catalog for library/tests).
    ///
    /// # Fallback behaviour (never panics)
    /// When the selected provider is missing from the catalog (e.g. `ZEN_PROVIDER` names a provider
    /// that only exists in a `models.json` that was never installed), the first model of the catalog is
    /// used and a `tracing::warn` is emitted. Library code must not abort the process on a
    /// configuration mistake; callers that need the error use [`Self::try_from_env`].
    pub fn from_env() -> Self {
        match Self::try_from_env() {
            Ok(model) => model,
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    "default provider is not in the model catalog; falling back to the first catalog model"
                );
                Self::first_available().unwrap_or_else(legacy_default_model)
            }
        }
    }

    /// Non-panicking, error-reporting variant of [`Self::from_env`].
    ///
    /// # Errors
    /// [`CatalogError::UnknownProvider`] when the selected provider is absent from the catalog,
    /// [`CatalogError::NoDefaultModel`] when it is present but declares no models.
    pub fn try_from_env() -> Result<Self, crate::catalog::CatalogError> {
        Catalog::current().default_model()
    }

    /// First model declared by the current catalog (the fallback used by [`Self::from_env`]).
    pub fn first_available() -> Option<Self> {
        Catalog::current().models().into_iter().next()
    }

    /// Constructs a model for the selected provider: takes the first model of that provider in the current catalog.
    ///
    /// If the catalog has no such provider or no models, falls back to the existing defaults (`deepseek-v4-flash` + 1M context + 384k output)
    /// and logs `tracing::warn` (unknown providers are already returned as `None` by the selection layer `resolve_provider`).
    pub fn from_provider(provider: &dyn Provider) -> Self {
        Catalog::current()
            .first_model(provider.id())
            .unwrap_or_else(|| {
                tracing::warn!(
                    provider = provider.id(),
                    "provider not found in models.json catalog; using legacy defaults"
                );
                legacy_fallback_model(provider)
            })
    }

    /// Set model ID.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Set the model name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the API base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set the context window size.
    pub fn with_context_window(mut self, tokens: u64) -> Self {
        self.context_window = tokens;
        self
    }

    /// Set the maximum number of output tokens.
    pub fn with_max_tokens(mut self, tokens: u64) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set whether reasoning is supported.
    pub fn with_reasoning(mut self, reasoning: bool) -> Self {
        self.reasoning = reasoning;
        self
    }

    /// Set additional HTTP request headers.
    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers = Some(headers);
        self
    }

    /// Set token cost rates.
    pub fn with_cost_rates(mut self, cost_rates: CostRates) -> Self {
        self.cost_rates = Some(cost_rates);
        self
    }

    pub fn api_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    /// Compute the actual cost from token usage and rates.
    /// # Return Value
    /// The computed `Cost` struct. Returns an all-zero `Cost` if `cost_rates` is `None`.
    pub fn calculate_cost(
        &self,
        input_tokens: u64,
        output_tokens: u64,
        cache_read_tokens: u64,
        cache_write_tokens: u64,
    ) -> crate::types::Cost {
        match &self.cost_rates {
            Some(cost_rates) => {
                let input = input_tokens as f64 / 1_000_000.0 * cost_rates.input;
                let output = output_tokens as f64 / 1_000_000.0 * cost_rates.output;
                let cache_read = cache_read_tokens as f64 / 1_000_000.0 * cost_rates.cache_read;
                let cache_write = cache_write_tokens as f64 / 1_000_000.0 * cost_rates.cache_write;
                let total = input + output + cache_read + cache_write;

                Cost {
                    input,
                    output,
                    cache_read,
                    cache_write,
                    total,
                }
            }
            None => Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
        }
    }
}

/// Compatibility fallback for unknown providers: keeps the T-AI-02-4-era defaults (1M context / 384k output).
/// Normally not triggered under the existing catalog (`resolve_provider` and `from_provider` read the same catalog); only when
/// a provider exists in the catalog but declares no models is this a defensive fallback.
/// Hard-coded last-resort model used when even the catalog is empty (unreachable with the embedded
/// catalog, kept so `from_env` never panics).
fn legacy_default_model() -> Model {
    Model {
        id: "deepseek-v4-flash".to_string(),
        name: None,
        provider: "opencode-go".to_string(),
        api: crate::provider::Api::OpenAiCompletions,
        base_url: "https://opencode.ai/zen/go/v1".to_string(),
        context_window: 1_000_000,
        max_tokens: 384_000,
        reasoning: true,
        headers: None,
        cost_rates: Some(CostRates {
            input: 0.5,
            output: 2.0,
            cache_read: 0.1,
            cache_write: 0.5,
        }),
        compat: None,
    }
}

fn legacy_fallback_model(provider: &dyn Provider) -> Model {
    Model {
        id: "deepseek-v4-flash".to_string(),
        name: None,
        provider: provider.id().to_string(),
        api: provider.api(),
        base_url: provider.base_url(),
        context_window: 1_000_000,
        max_tokens: 384_000,
        reasoning: true,
        headers: None,
        cost_rates: Some(CostRates {
            input: 0.5,
            output: 2.0,
            cache_read: 0.1,
            cache_write: 0.5,
        }),
        compat: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::Api;

    // ── helper ──

    fn default_model() -> Model {
        Model {
            id: "test-model".to_string(),
            name: None,
            provider: "deepseek".to_string(),
            api: Api::OpenAiCompletions,
            base_url: "https://api.test.com".to_string(),
            context_window: 4096,
            max_tokens: 1024,
            reasoning: true,
            headers: None,
            cost_rates: Some(CostRates {
                input: 1.0,
                output: 2.0,
                cache_read: 0.5,
                cache_write: 0.5,
            }),
            compat: None,
        }
    }

    // ── default model selection (from_env / try_from_env) ──

    /// Serialises the env-mutating tests below (they read `ZEN_PROVIDER`).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Restores `ZEN_PROVIDER` on drop.
    struct EnvGuard(Option<String>);
    impl EnvGuard {
        fn set(v: &str) -> Self {
            let old = std::env::var("ZEN_PROVIDER").ok();
            std::env::set_var("ZEN_PROVIDER", v);
            Self(old)
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.0 {
                Some(v) => std::env::set_var("ZEN_PROVIDER", v),
                None => std::env::remove_var("ZEN_PROVIDER"),
            }
        }
    }

    #[test]
    fn test_try_from_env_reports_unknown_provider() {
        let _l = ENV_LOCK.lock().unwrap();
        let _g = EnvGuard::set("provider-that-does-not-exist");
        let err = Model::try_from_env().expect_err("unknown provider must be an error");
        assert!(
            matches!(err, crate::catalog::CatalogError::UnknownProvider(_)),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_from_env_falls_back_instead_of_panicking() {
        // Regression: `from_env` used to panic ("unknown default provider"), aborting library users
        // and examples when ZEN_PROVIDER named a provider that is not in the catalog.
        let _l = ENV_LOCK.lock().unwrap();
        let _g = EnvGuard::set("provider-that-does-not-exist");
        let model = Model::from_env();
        assert_ne!(model.provider, "provider-that-does-not-exist");
        assert!(
            Catalog::current()
                .models()
                .iter()
                .any(|m| m.provider == model.provider && m.id == model.id),
            "fallback model must come from the catalog: {}/{}",
            model.provider,
            model.id
        );
    }

    #[test]
    fn test_first_available_returns_a_catalog_model() {
        let model = Model::first_available().expect("embedded catalog is non-empty");
        assert!(!model.provider.is_empty() && !model.id.is_empty());
    }

    // ── builder methods ──

    #[test]
    fn test_with_id() {
        let m = default_model().with_id("gpt-4");
        assert_eq!(m.id, "gpt-4");
    }

    #[test]
    fn test_with_name() {
        let m = default_model().with_name("GPT-4");
        assert_eq!(m.name, Some("GPT-4".to_string()));
    }

    #[test]
    fn test_with_base_url() {
        let m = default_model().with_base_url("https://other.api.com");
        assert_eq!(m.base_url, "https://other.api.com");
    }

    #[test]
    fn test_with_context_window() {
        let m = default_model().with_context_window(128_000);
        assert_eq!(m.context_window, 128_000);
    }

    #[test]
    fn test_with_max_tokens() {
        let m = default_model().with_max_tokens(4096);
        assert_eq!(m.max_tokens, 4096);
    }

    #[test]
    fn test_with_reasoning() {
        let m = default_model().with_reasoning(false);
        assert!(!m.reasoning);
    }

    #[test]
    fn test_with_headers() {
        let headers = HashMap::from([("X-Custom".to_string(), "value".to_string())]);
        let m = default_model().with_headers(headers.clone());
        assert_eq!(m.headers, Some(headers));
    }

    #[test]
    fn test_with_cost_rates() {
        let rates = CostRates {
            input: 2.0,
            output: 8.0,
            cache_read: 0.2,
            cache_write: 1.0,
        };
        let m = default_model().with_cost_rates(rates.clone());
        assert_eq!(m.cost_rates, Some(rates));
    }

    #[test]
    fn test_builder_chaining() {
        let m = default_model()
            .with_id("chain-model")
            .with_name("Chain")
            .with_base_url("https://chain.api")
            .with_context_window(8192)
            .with_max_tokens(2048)
            .with_reasoning(false);

        assert_eq!(m.id, "chain-model");
        assert_eq!(m.name, Some("Chain".to_string()));
        assert_eq!(m.base_url, "https://chain.api");
        assert_eq!(m.context_window, 8192);
        assert_eq!(m.max_tokens, 2048);
        assert!(!m.reasoning);
    }

    // ── calculate_cost ──

    #[test]
    fn test_calculate_cost_typical() {
        let m = default_model(); // rates: input=1.0, output=2.0, cache_read=0.5, cache_write=0.5
        let cost = m.calculate_cost(1_000_000, 500_000, 200_000, 100_000);

        assert!((cost.input - 1.0).abs() < 1e-10); // 1,000,000 / 1,000,000 * 1.0
        assert!((cost.output - 1.0).abs() < 1e-10); // 500,000 / 1,000,000 * 2.0
        assert!((cost.cache_read - 0.1).abs() < 1e-10); // 200,000 / 1,000,000 * 0.5
        assert!((cost.cache_write - 0.05).abs() < 1e-10); // 100,000 / 1,000,000 * 0.5
        assert!((cost.total - 2.15).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_cost_zero_tokens() {
        let m = default_model();
        let cost = m.calculate_cost(0, 0, 0, 0);

        assert!((cost.input - 0.0).abs() < 1e-10);
        assert!((cost.output - 0.0).abs() < 1e-10);
        assert!((cost.cache_read - 0.0).abs() < 1e-10);
        assert!((cost.cache_write - 0.0).abs() < 1e-10);
        assert!((cost.total - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_cost_no_rates() {
        let m = Model {
            cost_rates: None,
            ..default_model()
        };
        let cost = m.calculate_cost(1_000_000, 500_000, 200_000, 100_000);

        assert!((cost.input - 0.0).abs() < 1e-10);
        assert!((cost.output - 0.0).abs() < 1e-10);
        assert!((cost.cache_read - 0.0).abs() < 1e-10);
        assert!((cost.cache_write - 0.0).abs() < 1e-10);
        assert!((cost.total - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_cost_large_values() {
        let m = default_model();
        let cost = m.calculate_cost(
            20_000_000_000,
            20_000_000_000,
            20_000_000_000,
            20_000_000_000,
        );

        assert!((cost.input - 20000.0).abs() < 1e-6); // 20,000,000,000 / 1,000,000 * 1.0
        assert!((cost.output - 40000.0).abs() < 1e-6); // 20,000,000,000 / 1,000,000 * 2.0
        assert!((cost.cache_read - 10000.0).abs() < 1e-6); // 20,000,000,000 / 1,000,000 * 0.5
        assert!((cost.cache_write - 10000.0).abs() < 1e-6); // 20,000,000,000 / 1,000,000 * 0.5
        assert!((cost.total - 80000.0).abs() < 1e-6);
    }

    // ── from_env / from_provider ──

    #[test]
    fn from_provider_reads_catalog_model() {
        let provider = crate::provider::resolve_provider("huoshan").expect("huoshan provider");
        let model = Model::from_provider(provider.as_ref());
        assert_eq!(model.provider, "huoshan");
        assert_eq!(model.context_window, 1_000_000);
        assert!(model.compat.is_some());
    }

    #[test]
    fn from_provider_unknown_uses_legacy_defaults() {
        struct UnknownProvider;

        impl Provider for UnknownProvider {
            fn id(&self) -> &str {
                "unknown"
            }
            fn api(&self) -> crate::provider::Api {
                crate::provider::Api::OpenAiCompletions
            }
            fn base_url(&self) -> String {
                "https://unknown.test".to_string()
            }
            fn resolve_api_key(&self) -> Option<String> {
                None
            }
        }

        let model = Model::from_provider(&UnknownProvider);
        assert_eq!(model.id, "deepseek-v4-flash");
        assert_eq!(model.provider, "unknown");
        assert_eq!(model.base_url, "https://unknown.test");
        assert_eq!(model.context_window, 1_000_000);
        assert_eq!(model.max_tokens, 384_000);
    }
}
