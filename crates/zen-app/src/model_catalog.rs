// REQ-V2-02 (pi-366, pi-368, pi-369): built-in model catalog (interim) — placeholder implementation for T-V2-02-1.
//
// This module is the data source for REQ-V2-02 "model selection": it collects the provider default models currently known to zen into
// an enumerable list consumed by the `/model` selector, the `/scoped-models` panel and cycle switching.
//
// **interim note**: T-V2-02-4 first delivered `/model` availability with an "enumerable model set"; **T-AI-02-11 has switched the data source
// from the hardcoded `BUILT_IN_PROVIDERS` table to the process-level catalog `Catalog::current()`** (making user models.json visible as well).
// When T-V2-02-1 takes over, it owns this module's public surface/naming (`resolve_current_model`/`list_available_models`) and
// per-model capability bits; it does not revive model-config/model-registry/model-runtime dropped from scope, and does not do remote catalog refresh.
//
// Traceability chain: pi-366 (list-models) / pi-368 (model-search) / pi-369 (model-selector) → REQ-V2-02
// → this module (T-V2-02-4 interim) + T-V2-02-1 (takes over).

use zen_ai::catalog::Catalog;
use zen_ai::types::Model;

/// Enumerate available models — "there must be a candidate list when `/model` opens". Data source is the process-level catalog
/// [`Catalog::current`] (`main` loads user `models.json` overrides at startup; if not installed, built-in catalog),
/// Sorted by `(provider, id)` to ensure determinism.
/// Consumers: `/model`, `/scoped-models` in `modes/interactive` and `app.model.cycle*`.
/// **T-V2-02-1 takeover point**: function names/public surface and `resolve_current_model`/`list_available_models` are decided there;
/// This task (T-AI-02-11) only switches the data source from the hardcoded `BUILT_IN_PROVIDERS` table to the catalog.
pub(crate) fn built_in_models() -> Vec<Model> {
    Catalog::current().models()
}

/// Canonical full model id: `"<provider>/<id>"` — used uniformly for selector row identity, cycle set keys and `SelectorSelected` payloads.
/// Consumers: the two model selection components + selection routing in `modes/interactive`.
pub(crate) fn full_id(model: &Model) -> String {
    format!("{}/{}", model.provider, model.id)
}

/// Look up and clone a model in the catalog by `"provider/id"` (or by provider + id given separately).
/// Returns `None` on no match (callers report "model not found" instead of panicking).
/// Consumers: selection routing in `modes/interactive` and exact `--model` matching.
pub(crate) fn find_model(models: &[Model], provider: &str, id: &str) -> Option<Model> {
    models
        .iter()
        .find(|m| m.provider == provider && m.id == id)
        .cloned()
}

/// Exact model reference matching per pi's `findExactModelReferenceMatch` semantics —
/// "an explicit form like `/model huoshan/deepseek-v4-flash` should hit directly without opening the selector".
/// Match order (case-insensitive): ① full `provider/id`; ② `provider` + `id` containing `/`; ③ bare `id`.
/// Returns as soon as a step matches **exactly one** model; multiple matches (ambiguous) or zero matches return `None` (caller falls back to opening the selector).
/// Consumers: the exact-match branch of `/model <term>` in `modes/interactive`.
pub(crate) fn find_exact_model_reference(models: &[Model], reference: &str) -> Option<Model> {
    let trimmed = reference.trim();
    if trimmed.is_empty() {
        return None;
    }
    let normalized = trimmed.to_lowercase();

    // ① full provider/id
    let canonical: Vec<&Model> = models
        .iter()
        .filter(|m| format!("{}/{}", m.provider, m.id).to_lowercase() == normalized)
        .collect();
    if canonical.len() == 1 {
        return Some(canonical[0].clone());
    }
    if canonical.len() > 1 {
        return None;
    }

    // ② provider with / + id
    if let Some((provider, id)) = trimmed.split_once('/') {
        let provider = provider.trim();
        let id = id.trim();
        if !provider.is_empty() && !id.is_empty() {
            let matches: Vec<&Model> = models
                .iter()
                .filter(|m| {
                    m.provider.to_lowercase() == provider.to_lowercase()
                        && m.id.to_lowercase() == id.to_lowercase()
                })
                .collect();
            if matches.len() == 1 {
                return Some(matches[0].clone());
            }
            if matches.len() > 1 {
                return None;
            }
        }
    }

    // ③ bare id
    let id_matches: Vec<&Model> = models
        .iter()
        .filter(|m| m.id.to_lowercase() == normalized)
        .collect();
    if id_matches.len() == 1 {
        Some(id_matches[0].clone())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests that serialize access to ZEN_PROVIDER (process-level env is not thread-safe).
    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn built_in_models_covers_both_providers() {
        let _g = env_lock();
        let models = built_in_models();
        let providers: Vec<&str> = models.iter().map(|m| m.provider.as_str()).collect();
        assert!(
            providers.contains(&"opencode-go"),
            "missing opencode-go: {providers:?}"
        );
        assert!(
            providers.contains(&"huoshan"),
            "missing huoshan: {providers:?}"
        );
        // every model id is non-empty
        assert!(models.iter().all(|m| !m.id.is_empty()));
    }

    #[test]
    fn full_id_is_provider_slash_id() {
        let m = Model::from_env();
        let fid = full_id(&m);
        assert_eq!(fid, format!("{}/{}", m.provider, m.id));
        assert!(fid.contains('/'));
    }

    #[test]
    fn find_model_hits_and_misses() {
        let _g = env_lock();
        let models = built_in_models();
        let first = models.first().expect("at least one built-in model");
        let found = find_model(&models, &first.provider, &first.id);
        assert_eq!(
            found.as_ref().map(|m| m.id.as_str()),
            Some(first.id.as_str())
        );

        assert!(find_model(&models, &first.provider, "does-not-exist").is_none());
        assert!(find_model(&models, "no-such-provider", &first.id).is_none());
    }

    #[test]
    fn exact_reference_matching_semantics() {
        let a = Model::from_env().with_id("alpha");
        let b = Model::from_env().with_id("beta");
        let models = vec![a.clone(), b.clone()];

        // ① full provider/id (case-insensitive)
        let canonical = format!("{}/{}", a.provider.to_uppercase(), a.id.to_uppercase());
        assert_eq!(
            find_exact_model_reference(&models, &canonical).map(|m| m.id),
            Some("alpha".to_string())
        );
        // ② provider with / + id
        assert_eq!(
            find_exact_model_reference(&models, &format!("{}/{}", b.provider, b.id)).map(|m| m.id),
            Some("beta".to_string())
        );
        // ③ bare id unique
        assert_eq!(
            find_exact_model_reference(&models, "alpha").map(|m| m.id),
            Some("alpha".to_string())
        );
        // no match / empty string
        assert!(find_exact_model_reference(&models, "gamma").is_none());
        assert!(find_exact_model_reference(&models, "   ").is_none());
        // ambiguous (duplicate provider+id) → neither bare id nor canonical matches
        let dup = vec![a.clone(), Model::from_env().with_id("alpha")];
        assert!(find_exact_model_reference(&dup, "alpha").is_none());
    }
}
