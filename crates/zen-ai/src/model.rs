use std::collections::HashMap;
use std::env;

use crate::provider::{default_provider_name, resolve_provider, Provider};
use crate::types::{Cost, CostRates, Model};

impl Model {
    /// Create the default model config from environment variables (default provider opencode-go, model deepseek-v4-flash).
    ///
    /// Provider selection: env `ZEN_PROVIDER` (default `opencode-go`); model id / base_url /
    /// max_tokens are read from per-provider env vars (see the env var conventions in the `provider` module).
    pub fn from_env() -> Self {
        let name = default_provider_name();
        let provider =
            resolve_provider(&name).unwrap_or_else(|| panic!("unknown default provider: {}", name));
        Self::from_provider(provider.as_ref())
    }

    /// Build the model for the selected provider.
    ///
    /// Model id and max_tokens are read from per-provider env vars (`OPENCODE_MODEL`/
    /// `ARK_MODEL`, `OPENCODE_MAX_TOKENS`/`ARK_MAX_TOKENS`), defaulting to
    /// `deepseek-v4-flash` and 384000 respectively (aligned with DeepSeek-V4-Flash max output length); base_url
    /// comes from `provider.base_url()` (which internally reads
    /// `OPENCODE_BASE_URL`/`ARK_BASE_URL` and falls back to the default endpoint); the `provider` field is set to `provider.id()`.
    pub fn from_provider(provider: &dyn Provider) -> Self {
        Self {
            id: provider_model_id(provider.id()).unwrap_or_else(|| "deepseek-v4-flash".to_string()),
            provider: provider.id().to_string(),
            api: provider.api(),
            base_url: provider.base_url(),
            max_tokens: provider_max_tokens(provider.id()),
            context_window: 65536,
            reasoning: true,
            name: None,
            headers: None,
            cost_rates: Some(CostRates {
                input: 0.5,
                output: 2.0,
                cache_read: 0.1,
                cache_write: 0.5,
            }),
        }
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

/// Read the model id env var per provider (`OPENCODE_MODEL`/`ARK_MODEL`); returns None for unknown providers.
fn provider_model_id(provider: &str) -> Option<String> {
    match provider {
        "opencode-go" => env::var("OPENCODE_MODEL").ok(),
        "huoshan" => env::var("ARK_MODEL").ok(),
        _ => None,
    }
}

/// Read the max_tokens env var per provider (`OPENCODE_MAX_TOKENS`/`ARK_MAX_TOKENS`);
/// defaults to 384000 when unset or unparsable (aligned with DeepSeek-V4-Flash max output length).
fn provider_max_tokens(provider: &str) -> u64 {
    let var = match provider {
        "opencode-go" => "OPENCODE_MAX_TOKENS",
        "huoshan" => "ARK_MAX_TOKENS",
        _ => return 384_000,
    };
    env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(384_000)
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
        }
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
    // Depends on process env vars (ZEN_PROVIDER / OPENCODE_* / ARK_*), not a pure function. For testing,
    // Recommend explicitly setting env vars inside the test function and using the serial_test crate for serialization,
    // to prevent interference between parallel tests.
    // Not implemented yet; will add once serial_test or a mocking approach is introduced.
}
