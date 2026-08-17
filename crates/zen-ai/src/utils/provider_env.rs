//! Prefer values from the config file, fall back to environment variables

use std::collections::HashMap;
use std::env;

/// List of known provider environment variable names, used for config discovery and diagnostics.
/// Per-provider variables (aligned with pi `envApiKeyAuth`); no global `OPENAI_*`.
pub const KNOWN_PROVIDER_ENV_VARS: &[&str] = &[
    // zen selection layer
    "ZEN_PROVIDER",
    // opencode-go (default provider)
    "OPENCODE_API_KEY",
    "OPENCODE_BASE_URL",
    "OPENCODE_MODEL",
    "OPENCODE_MAX_TOKENS",
    // huoshan (Volcano Engine Ark)
    "ARK_API_KEY",
    "ARK_BASE_URL",
    "ARK_MODEL",
    "ARK_MAX_TOKENS",
];

/// Config file takes precedence, provider environment variables as fallback,
pub fn get_provider_env_value(
    name: &str,
    env_overrides: Option<&HashMap<String, String>>,
) -> Option<String> {
    env_overrides
        .and_then(|map| map.get(name).cloned())
        .or_else(|| env::var(name).ok())
}

/// Check whether a provider is configured (at least one required environment variable is set).
/// Iterate the `required_vars` list, checking whether each variable exists in `env_overrides` or the process environment.
/// Used to check config completeness at startup.
///
/// # Parameters
/// - `required_vars`: list of required provider environment variable names.
/// - `env_overrides`: optional scoped override.
///
/// # Return Value
/// Returns `true` if at least one required variable exists, otherwise returns `false`.
pub fn is_provider_configured(
    required_vars: &[&str],
    env_overrides: Option<&HashMap<String, String>>,
) -> bool {
    required_vars
        .iter()
        .any(|name| get_provider_env_value(name, env_overrides).is_some())
}

/// |map| map.get(key Collect all known provider environment variables and return a key-value pair collection.
/// Used for debug output, config diagnostics, and logging.
/// For security, the value of `OPENAI_API_KEY` in the returned collection is truncated (only the first 4 characters are kept).
/// # Parameters
/// - `env_overrides`: optional scoped override.
/// # Return Value
/// All set provider environment variables and their values (API keys truncated).
pub fn get_all_provider_env_values(
    env_overrides: Option<&HashMap<String, String>>,
) -> HashMap<String, String> {
    KNOWN_PROVIDER_ENV_VARS
        .iter()
        .filter_map(|name| {
            let value = get_provider_env_value(name, env_overrides)?;
            let value = if name.ends_with("API_KEY") {
                truncate_sensitive_value(&value)
            } else {
                value
            };
            Some((name.to_string(), value))
        })
        .collect()
}

/// Truncate sensitive values such as API keys, keeping only the first 4 characters and replacing the rest with `****`.
///
/// Used for log and diagnostic output to avoid leaking full credentials.
///
/// # Parameters
/// - `value`: the original value.
///
/// # Return Value
/// Truncated safe string (e.g. `"sk-1****"`); if shorter than 4 characters, kept in full.
fn truncate_sensitive_value(value: &str) -> String {
    if value.len() <= 4 {
        value.to_string()
    } else {
        format!("{}****", &value[..4])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── get_provider_env_value ──

    #[test]
    fn test_override_takes_precedence() {
        let mut overrides = HashMap::new();
        overrides.insert("ARK_API_KEY".to_string(), "override-key".to_string());
        // Even if the real environment variable exists, the override should take precedence
        let result = get_provider_env_value("ARK_API_KEY", Some(&overrides));
        assert_eq!(result, Some("override-key".to_string()));
    }

    #[test]
    fn test_override_returns_none_for_unset_key() {
        let overrides = HashMap::new();
        let result = get_provider_env_value("NONEXISTENT_VAR", Some(&overrides));
        assert_eq!(result, None);
    }

    #[test]
    fn test_no_override_falls_to_env() {
        // Use an environment variable name that almost certainly doesn't exist
        let result = get_provider_env_value("THIS_ENV_VAR_SHOULD_NOT_EXIST_8A3F2B", None);
        assert_eq!(result, None);
    }

    // ── is_provider_configured ──

    #[test]
    fn test_is_configured_true_when_override_present() {
        let mut overrides = HashMap::new();
        overrides.insert("OPENCODE_API_KEY".to_string(), "key".to_string());
        assert!(is_provider_configured(
            &["OPENCODE_API_KEY"],
            Some(&overrides)
        ));
    }

    #[test]
    fn test_is_configured_any_match() {
        let mut overrides = HashMap::new();
        overrides.insert("ARK_BASE_URL".to_string(), "url".to_string());
        // Return true as soon as one matches
        assert!(is_provider_configured(
            &["ARK_API_KEY", "ARK_BASE_URL"],
            Some(&overrides),
        ));
    }

    #[test]
    fn test_is_configured_empty_required_list() {
        let overrides = HashMap::new();
        // An empty list means no required variables; return false
        assert!(!is_provider_configured(&[], Some(&overrides)));
    }

    // ── truncate_sensitive_value ──

    #[test]
    fn test_truncate_long_value() {
        assert_eq!(truncate_sensitive_value("sk-1234567890"), "sk-1****");
    }

    #[test]
    fn test_truncate_exactly_four_chars() {
        assert_eq!(truncate_sensitive_value("abcd"), "abcd");
    }

    #[test]
    fn test_truncate_less_than_four_chars() {
        assert_eq!(truncate_sensitive_value("ab"), "ab");
    }

    #[test]
    fn test_truncate_empty_string() {
        assert_eq!(truncate_sensitive_value(""), "");
    }

    #[test]
    fn test_truncate_single_char() {
        assert_eq!(truncate_sensitive_value("a"), "a");
    }

    // ── get_all_provider_env_values ──
    // Depends on process environment variables, not a pure function; not tested for now.
    // Recommend supplementing this after introducing serial_test or a mocking approach.
}
