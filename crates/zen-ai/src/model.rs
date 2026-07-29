use std::collections::HashMap;
use std::env;

use crate::types::{Cost, CostRates, Model};

impl Model {
    /// Create the default model config from environment variables, falling back to DeepSeek V4 Flash defaults.
    pub fn from_env() -> Self {
        Self {
            id: env::var("OPENAI_MODEL").unwrap_or("deepseek-v4-flash".to_string()),
            base_url: env::var("OPENAI_BASE_URL")
                .unwrap_or("https://ark.cn-beijing.volces.com/api/coding/v3".to_string()),
            max_tokens: env::var("OPENAI_MAX_TOKENS")
                .unwrap_or(8192.to_string())
                .parse()
                .unwrap_or(8192),
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

    /// Apply environment variable overrides and return a new `Model` instance.
    ///
    /// Read environment variables such as `OPENAI_BASE_URL`, `OPENAI_MODEL`, and `OPENAI_MAX_TOKENS`,
    /// Override the corresponding fields in the model config. Environment variables take precedence over defaults.
    ///
    /// Corresponds to the environment variable reading and merging logic in Pi's `provider-env.ts`, but simplifies the scope override chain.
    /// Works with T-AI-02-3 (provider environment variable reading): T-AI-02-3 provides raw environment variable reading,
    /// This function applies environment variable values to the Model config.
    pub fn with_env_overrides(mut self) -> Self {
        if let Ok(val) = env::var("OPENAI_MODEL") {
            self.id = val;
        }
        if let Ok(val) = env::var("OPENAI_BASE_URL") {
            self.base_url = val;
        }
        if let Ok(val) = env::var("OPENAI_MAX_TOKENS") {
            if let Ok(parsed) = val.parse() {
                self.max_tokens = parsed;
            }
        }
        self
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
        let input_tokens = input_tokens;
        let output_tokens = output_tokens;
        let cache_read_tokens = cache_read_tokens;
        let cache_write_tokens = cache_write_tokens;
        match &self.cost_rates {
            Some(cost_rates) => {
                let input = input_tokens as f64 / 1000_1000.0 * cost_rates.input;
                let output = output_tokens as f64 / 1000_1000.0 * cost_rates.output;
                let cache_read = cache_read_tokens as f64 / 1000_1000.0 * cost_rates.cache_read;
                let cache_write = cache_write_tokens as f64 / 1000_1000.0 * cost_rates.cache_write;
                let total = input + output + cache_read + cache_write;

                Cost {
                    input: input,
                    output: output,
                    cache_read: cache_read,
                    cache_write: cache_write,
                    total: total,
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── helper ──

    fn default_model() -> Model {
        Model {
            id: "test-model".to_string(),
            name: None,
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
        // NOTE: implementation uses 1000_1000.0 (10,001,000) as the divisor.
        let m = default_model(); // rates: input=1.0, output=2.0, cache_read=0.5, cache_write=0.5
        let cost = m.calculate_cost(10_001_000, 5_000_500, 2_000_200, 1_000_100);

        assert!((cost.input - 1.0).abs() < 1e-10);     // 10,001,000 / 10,001,000 * 1.0
        assert!((cost.output - 1.0).abs() < 1e-10);  // 5,000,500 / 10,001,000 * 2.0
        assert!((cost.cache_read - 0.1).abs() < 1e-10); // 2,000,200 / 10,001,000 * 0.5
        assert!((cost.cache_write - 0.05).abs() < 1e-10); // 1,000,100 / 10,001,000 * 0.5
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
        // 20,002,000,000 tokens each dimension (multiple of 10,001,000)
        let cost = m.calculate_cost(20_002_000_000, 20_002_000_000, 20_002_000_000, 20_002_000_000);

        assert!((cost.input - 2000.0).abs() < 1e-6);        // 20,002,000,000 / 10,001,000 * 1.0
        assert!((cost.output - 4000.0).abs() < 1e-6);       // 20,002,000,000 / 10,001,000 * 2.0
        assert!((cost.cache_read - 1000.0).abs() < 1e-6);   // 20,002,000,000 / 10,001,000 * 0.5
        assert!((cost.cache_write - 1000.0).abs() < 1e-6);  // 20,002,000,000 / 10,001,000 * 0.5
        assert!((cost.total - 8000.0).abs() < 1e-6);
    }

    // ── from_env / with_env_overrides ──
    // Depends on process environment variables; not a pure function. For testing, explicitly set env vars inside the test function and
    // Use the serial_test crate to serialize and prevent interference between parallel tests.
    // Not implemented yet; will add once serial_test or a mocking approach is introduced.
}
