// REQ-AI-02 (pi-120 / T-AI-02-11): integration verification of `init_user_catalog` startup assembly in `main`.
//
// Standalone test binary (`OnceLock` cannot be reset after set + process env vars are mutually exclusive); tests only a single startup sequence.

use zen_ai::catalog::Catalog;

#[test]
fn init_user_catalog_installs_user_models_json() {
    let dir = std::env::temp_dir().join(format!("zen_app_catalog_init_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("models.json"),
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
    // Points to a temp agent directory (`<agentDir>/models.json` takes effect)
    std::env::set_var("ZEN_CODING_AGENT_DIR", &dir);

    zen_app::config::catalog::init_user_catalog();

    // User provider/model has been loaded into the process-level catalog
    assert!(Catalog::current().model("custom", "custom-model").is_some());
    assert_eq!(
        Catalog::current().api_key_for_provider("custom").as_deref(),
        Some("literal")
    );

    std::env::remove_var("ZEN_CODING_AGENT_DIR");
    let _ = std::fs::remove_dir_all(&dir);
}
