// REQ-AI-02 (pi-120 / T-AI-02-11): process-level Catalog singleton (install/current) — standalone integration test binary.
//
// placed in a separate file (= separate test binary): `OnceLock` cannot be reset after installation; if combined with library unit tests for "not installed == embedded"
// the same process would fail randomly due to execution order. This file only tests the single deterministic sequence "first not installed, then installed".

use zen_ai::catalog::Catalog;

#[test]
fn catalog_singleton_install_and_current() {
    // Precondition: this binary has not installed yet → current falls back to embedded
    assert!(std::ptr::eq(Catalog::current(), Catalog::embedded()));

    // build a directory with user overrides (custom provider)
    let dir = std::env::temp_dir().join(format!("zen_catalog_install_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("models.json");
    std::fs::write(
        &path,
        r#"{
            "providers": {
                "custom": {
                    "api": "openai-completions",
                    "baseUrl": "https://custom.test/v1",
                    "apiKey": "literal",
                    "models": [{ "id": "custom-model" }]
                }
            }
        }"#,
    )
    .unwrap();
    let user = Catalog::load(Some(&path)).expect("load user catalog");

    // first installation succeeds; repeated installation (idempotent) is ignored
    assert!(Catalog::install(user), "First install should succeed");
    assert!(
        !Catalog::install(Catalog::embedded().clone()),
        "Repeated install should be ignored"
    );

    // current now returns the installed instance: includes the user provider and no longer points to embedded
    assert!(!std::ptr::eq(Catalog::current(), Catalog::embedded()));
    assert!(Catalog::current().model("custom", "custom-model").is_some());
    assert_eq!(
        Catalog::current().api_key_for_provider("custom").as_deref(),
        Some("literal")
    );

    let _ = std::fs::remove_dir_all(&dir);
}
