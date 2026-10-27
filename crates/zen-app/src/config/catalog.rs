// REQ-AI-02 (pi-120): application-layer model catalog assembly — loading user models.json and process-level installation
//
// Traceability chain: pi-120 (models.ts) / pi-118 (provider-env) → REQ-AI-02 (post-v1 revisions T-AI-02-9~11)
// → this module (T-AI-02-11). See tasks/T-AI-02-11-implemented.md for implementation notes.

//! Loading and process-level installation of user `models.json` (T-AI-02-11).
//!
//! `zen-ai`'s [`Catalog`] only handles parsing and merging of "built-in catalog + optional user file"; **where the file is** belongs to the application layer
//! policy (`zen-ai` does not read `~/.zen`). This module wires the two together:
//! - [`load_catalog_for_path`]: fault-tolerant loading (bad file falls back to built-in catalog + returns diagnostics), unit-testable, no global state;
//! - [`init_user_catalog`]: called once at startup, installs the result as the process-wide unique catalog (for `main` only).
//!
//! Synchronous IO strategy: one-time read at startup (one of the two options at the T-AI-02-9 seam) — files are small and this happens before TUI setup,
//! no `spawn_blocking`.

use std::path::Path;

use zen_ai::catalog::{Catalog, CatalogError};

use super::get_models_path;

/// Loads the catalog from the given path; a bad file neither panics nor is silent: returns "usable catalog + diagnostics".
///
/// "A typo of one comma in a hand-written user models.json should not prevent zen from starting at all" — if the file exists but read/parse/semantic validation
/// fails, return the built-in catalog and hand the error to the caller for output. Path is `None` or file does not exist → built-in catalog only.
///
/// # Parameters
/// - `path`: user `models.json` path (`None` → built-in catalog only).
///
/// # Return Value
/// `(catalog, diagnostics)`: diagnostics `Some` means it fell back to the built-in catalog; the caller is responsible for warning display.
/// **Diagnostics can only be `CatalogError::Io` / `Parse` / `Invalid`** (the output surface of `Catalog::load`),
/// never query-type variants such as `UnknownProvider`/`UnknownModel`/`NoDefaultModel`.
///
/// # Consumers
/// `init_user_catalog` (current); T-V2-02-2 (list models) / T-V2-02-3 (config selection) planned to hook in.
pub fn load_catalog_for_path(path: Option<&Path>) -> (Catalog, Option<CatalogError>) {
    match Catalog::load(path) {
        Ok(catalog) => (catalog, None),
        Err(err) => (Catalog::embedded().clone(), Some(err)),
    }
}

/// Called once at startup: parses `<agentDir>/models.json` and installs it as the process-level catalog.
///
/// "On the user's next startup, the modified models.json takes effect" — path unavailable (no home) → keep built-in catalog (log warn);
/// bad file → output diagnostics (`tracing::error!` + stderr), then fall back to built-in catalog (**do not install a degraded copy**,
/// `Catalog::current()` naturally falls back to `embedded()` when not installed), without aborting the process.
///
/// # Bounds
/// This function is **application startup assembly (for `main` only)** and has the side effect of writing directly to stderr; it is not a general-purpose library function. For composable
/// the "load + self-handle diagnostics" semantics, use `load_catalog_for_path`.
///
/// Installation is idempotent (first install via `OnceLock` wins); there is a single call site.
///
/// # Consumers
/// `main` (the only call site).
pub fn init_user_catalog() {
    let path = match get_models_path() {
        Ok(path) => path,
        Err(err) => {
            tracing::warn!("models.json path unavailable ({err}); using built-in catalog");
            return;
        }
    };
    let (catalog, diagnostic) = load_catalog_for_path(Some(&path));
    if let Some(err) = diagnostic {
        tracing::error!(path = %path.display(), "invalid models.json: {err}");
        // Interactive mode shows this warning at startup; print mode also stops at stderr, then falls back to the built-in catalog.
        eprintln!(
            "Warning: invalid models.json ({}): {err}; using built-in catalog",
            path.display()
        );
        // Do not install a degraded copy: `Catalog::current()` falls back to `embedded()` when not installed, preserving
        // the invariant "not installed ⟺ current() pointer equals embedded()" (for easier testing and clearer semantics).
        return;
    }
    let _ = Catalog::install(catalog);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    /// Unique temporary directory (std-only; follows existing precedent in session/config tests).
    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "zen_config_catalog_{}_{}",
            name,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(path: &Path, content: &str) {
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
    }

    #[test]
    fn none_path_yields_embedded_without_diagnostic() {
        let (catalog, diagnostic) = load_catalog_for_path(None);
        assert!(diagnostic.is_none());
        assert!(catalog.model("opencode-go", "deepseek-v4-flash").is_some());
        assert!(catalog.model("huoshan", "deepseek-v4-flash").is_some());
    }

    #[test]
    fn missing_file_yields_embedded_without_diagnostic() {
        let dir = temp_dir("missing");
        let path = dir.join("models.json");
        let (catalog, diagnostic) = load_catalog_for_path(Some(&path));
        assert!(diagnostic.is_none(), "a missing file is not an error");
        assert!(catalog.model("opencode-go", "deepseek-v4-flash").is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn valid_user_file_is_merged() {
        let dir = temp_dir("valid");
        let path = dir.join("models.json");
        write(
            &path,
            r#"{
                // 用户自定义 provider（JSONC 容忍）
                "providers": {
                    "custom": {
                        "api": "openai-completions",
                        "baseUrl": "https://custom.example/v1",
                        "apiKey": "literal-key",
                        "models": [{ "id": "custom-model" }]
                    }
                }
            }"#,
        );
        let (catalog, diagnostic) = load_catalog_for_path(Some(&path));
        assert!(diagnostic.is_none());
        // Built-in + user merge
        assert!(catalog.model("opencode-go", "deepseek-v4-flash").is_some());
        assert!(catalog.model("custom", "custom-model").is_some());
        assert_eq!(
            catalog.api_key_for_provider("custom").as_deref(),
            Some("literal-key")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn broken_json_falls_back_to_embedded_with_diagnostic() {
        let dir = temp_dir("broken");
        let path = dir.join("models.json");
        write(&path, "{ this is not json ");
        let (catalog, diagnostic) = load_catalog_for_path(Some(&path));
        assert!(matches!(diagnostic, Some(CatalogError::Parse { .. })));
        // Fallback to built-in catalog remains usable
        assert!(catalog.model("opencode-go", "deepseek-v4-flash").is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn invalid_schema_falls_back_with_diagnostic() {
        let dir = temp_dir("invalid");
        let path = dir.join("models.json");
        // Empty id triggers semantic validation failure
        write(
            &path,
            r#"{ "providers": { "p": { "models": [{ "id": "" }] } } }"#,
        );
        let (catalog, diagnostic) = load_catalog_for_path(Some(&path));
        assert!(matches!(diagnostic, Some(CatalogError::Invalid(_))));
        assert!(catalog.model("opencode-go", "deepseek-v4-flash").is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
