//! `models.json` provider/model catalog (REQ-AI-02 / T-AI-02-9).
//!
//! Move "which providers exist, where endpoints are, how keys are fetched, which models exist, what compat flags are" out of Rust hardcoding
//! into a user-editable `models.json`. Schema is a subset of pi's `model-config.ts` (`providers` map +
//! `baseUrl/apiKey/api/headers/compat/models/modelOverrides`); does **not** introduce `auth.json`/OAuth/
//! `models-store.json` remote refresh.
//!
//! Catalog source: built-in default catalog (compile-time `include_str!("../assets/models.json")`) ← user
//! `<agentDir>/models.json` override. `T-AI-02-11` injects the user file path and owns the single `Catalog` instance.

use std::collections::HashMap;
use std::path::Path;
use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use crate::provider::{Api, ConfiguredProvider, OpenAiCompletionsCompat, Provider};
use crate::types::{CostRates, Model};
use crate::utils::provider_env::get_provider_env_value;

/// Built-in default catalog (embedded at compile time).
const DEFAULT_MODELS_JSON: &str = include_str!("../assets/models.json");

/// Schema default for `ModelDef.contextWindow` (aligned with pi `ModelDefinition`).
pub const DEFAULT_CONTEXT_WINDOW: u64 = 128_000;
/// Schema default for `ModelDef.maxTokens` (aligned with pi `ModelDefinition`).
pub const DEFAULT_MAX_TOKENS: u64 = 16_384;

/// Built-in catalog singleton. Parse/validation failure can only mean `assets/models.json` was corrupted;
/// locked by `tests::embedded_catalog_parses` — the AGENTS.md rule 4 debug exception allows one panic.
static EMBEDDED: LazyLock<Catalog> = LazyLock::new(|| {
    let config = parse_config(DEFAULT_MODELS_JSON, "<embedded>")
        .expect("embedded assets/models.json must parse (test: embedded_catalog_parses)");
    Catalog::from_config(config)
        .expect("embedded assets/models.json must pass validation (test: embedded_catalog_parses)")
});

/// Catalog load/lookup error. Library-boundary type, not serialized.
#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    /// User `models.json` exists but could not be read.
    #[error("failed to read models.json at {path}: {source}")]
    Io {
        /// Path where the error occurred (for diagnostics).
        path: String,
        /// Underlying IO error.
        #[source]
        source: std::io::Error,
    },
    /// JSON/JSONC parsing failed.
    #[error("failed to parse models.json at {path}: {source}")]
    Parse {
        /// Path where the error occurred (user file or `<embedded>`).
        path: String,
        /// Underlying JSON error.
        #[source]
        source: serde_json::Error,
    },
    /// Schema semantic error (empty id, model definitions missing baseUrl/api, contextWindow/maxTokens ≤ 0, etc.).
    #[error("invalid models.json: {0}")]
    Invalid(String),
    /// Default provider or explicitly requested one does not exist.
    #[error("unknown provider: {0}")]
    UnknownProvider(String),
    /// Exact model query did not match.
    #[error("provider {provider} has no model {model}")]
    UnknownModel {
        /// provider id。
        provider: String,
        /// model id。
        model: String,
    },
    /// Provider exists but has no models.
    #[error("provider {0} has no default model")]
    NoDefaultModel(String),
}

/// Root object of `models.json`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsConfig {
    /// Provider records; key = provider id (equal to the `Model.provider` string).
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,
}

/// A provider record; provider-level fields apply to all its models, model-level fields can override them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderConfig {
    /// Human-readable name; defaults to id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Protocol type; required when defining a model (at provider or model level).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<Api>,
    /// Endpoint base URL (without `/chat/completions`); required when defining a model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Key configuration value template (literal / `$VAR` / `${VAR}`); resolved keys are not stored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Static headers per request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Provider-level compatibility switches, overridable at model level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<OpenAiCompletionsCompat>,
    /// Model definitions; if absent, only provider-level fields are overridden.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<ModelDef>>,
    /// Patches for existing models; key = model id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_overrides: Option<HashMap<String, ModelOverride>>,
}

/// A single model definition; `Catalog` materializes it into [`Model`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDef {
    /// Model id (non-empty), used as the API's `model` field.
    pub id: String,
    /// Human-readable name; defaults to `None` (zen `Model.name` is an `Option`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Overrides the provider-level `api`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<Api>,
    /// Overrides the provider-level `base_url`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Whether thinking/reasoning is supported; defaults to `false`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,
    /// Context window; schema default `128_000`. Built-in records must set `1_000_000` explicitly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    /// Max output tokens; schema default `16_384`. Built-in records must set `384_000` explicitly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// Model-level static headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Token cost rates (JSON field `cost` → Rust `Model.cost_rates`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<CostRates>,
    /// Model-level compatibility flags, overriding provider-level ones.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<OpenAiCompletionsCompat>,
}

/// Patch for an existing model; no `id`/`api`/`baseUrl` (does not create a new identity, aligned with pi `ModelOverrideSchema`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelOverride {
    /// Overrides the model name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Overrides the reasoning flag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,
    /// Overrides the context window (must be > 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    /// Overrides the output token limit (must be > 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// Overrides model headers key by key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Overrides token cost rates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<CostRates>,
    /// Overrides compatibility flags field by field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<OpenAiCompletionsCompat>,
}

/// Merged read-only provider catalog (built-in defaults ← user `models.json`).
///
/// Not part of serde; holds no resolved secrets/connections. Models are materialized into [`Model`] on demand at query time.
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    providers: HashMap<String, ProviderConfig>,
}

impl Catalog {
    /// Built-in default catalog: `opencode-go` / `huoshan` records available out of the box.
    ///
    /// Current consumers: `provider::resolve_provider`, `Model::from_env`/`from_provider`.
    /// User catalog injection (`load`) is handled by T-AI-02-11.
    pub fn embedded() -> &'static Catalog {
        &EMBEDDED
    }

    /// Load the catalog: first take the built-in default catalog, then layer the optional user `models.json` on top.
    ///
    /// # Parameters
    /// - `user_path`: user file path; `None` or file not found → use only the built-in catalog.
    ///
    /// # Errors
    /// File exists but read/parse/semantic validation fails → [`CatalogError`]; no silent fallback.
    ///
    /// # Merge semantics (aligned with pi's `applyModelsJson`)
    /// - provider-level `name`/`baseUrl`/`apiKey`/`api`: user values override built-in values;
    ///   `headers`/`compat` merged key-by-key / field-by-field;
    /// - `models`: for the same id, merge with "user non-empty fields override built-in fields", otherwise append;
    /// - `modelOverrides`: patch models field-by-field by id.
    ///
    /// Current consumers: T-AI-02-11 (application-layer assembly).
    pub fn load(user_path: Option<&Path>) -> Result<Catalog, CatalogError> {
        let mut config = parse_config(DEFAULT_MODELS_JSON, "<embedded>")?;
        if let Some(path) = user_path {
            match std::fs::read_to_string(path) {
                Ok(content) => {
                    let user = parse_config(&content, &path.display().to_string())?;
                    merge_config(&mut config, user);
                }
                Err(source) if source.kind() == std::io::ErrorKind::NotFound => {}
                Err(source) => {
                    return Err(CatalogError::Io {
                        path: path.display().to_string(),
                        source,
                    });
                }
            }
        }
        Catalog::from_config(config)
    }

    /// Look up a provider and materialize it into a [`Provider`] implementation; returns `None` if not found.
    ///
    /// Current consumer: this crate's `provider::resolve_provider`, hence `pub(crate)`.
    pub(crate) fn provider(&self, id: &str) -> Option<Box<dyn Provider>> {
        let config = self.providers.get(id)?.clone();
        Some(Box::new(ConfiguredProvider::new(id.to_string(), config)))
    }

    /// Enumerate all models in the catalog (provider × model materialized), sorted by `(provider, id)` for determinism.
    ///
    /// Current consumers: T-AI-02-11 (replacing zen-app `model_catalog::built_in_models`) and
    /// T-V2-02-1/2 (list models).
    pub fn models(&self) -> Vec<Model> {
        let mut out = Vec::new();
        for (provider, config) in &self.providers {
            for def in config.models.as_deref().unwrap_or(&[]) {
                out.push(materialize_model(provider, config, def));
            }
        }
        out.sort_by(|a, b| {
            (a.provider.as_str(), a.id.as_str()).cmp(&(b.provider.as_str(), b.id.as_str()))
        });
        out
    }

    /// Look up a model exactly by `(provider, id)`; returns `None` if not found (callers report not found, no panic).
    ///
    /// Current consumer: T-AI-02-11 (`resolve_model` / session restore).
    pub fn model(&self, provider: &str, id: &str) -> Option<Model> {
        let config = self.providers.get(provider)?;
        let def = config.models.as_deref()?.iter().find(|def| def.id == id)?;
        Some(materialize_model(provider, config, def))
    }

    /// Get the first model of a provider (by declaration order in `models.json`, not lexicographic).
    ///
    /// Current consumer: `Model::from_provider` (compat shim before T-AI-02-11).
    pub(crate) fn first_model(&self, provider: &str) -> Option<Model> {
        let config = self.providers.get(provider)?;
        let def = config.models.as_deref()?.first()?;
        Some(materialize_model(provider, config, def))
    }

    /// Get the default model: the provider is determined by `ZEN_PROVIDER` (default `opencode-go`), the model is that provider's first entry.
    ///
    /// Current consumers: `Model::from_env` (this task) and T-AI-02-11 (flag-free startup).
    pub fn default_model(&self) -> Result<Model, CatalogError> {
        let provider = crate::provider::default_provider_name();
        let config = self
            .providers
            .get(&provider)
            .ok_or_else(|| CatalogError::UnknownProvider(provider.clone()))?;
        let def = config
            .models
            .as_ref()
            .and_then(|models| models.first())
            .ok_or_else(|| CatalogError::NoDefaultModel(provider.clone()))?;
        Ok(materialize_model(&provider, config, def))
    }

    /// Resolve the API key for a model's provider (`apiKey` config value: literal / `$ENV` / `${ENV}`).
    /// Returns `None` if `apiKey` is absent or env is missing.
    ///
    /// **No consumers?** Consumer: request-layer api-key fallback (`openai_stream::api_key_for_request`).
    /// Note: the request layer always requires a Bearer key; `None` is treated as `StreamError::NoApiKey`; v1 does not support
    /// keyless endpoints (custom providers must configure `apiKey`), see open question 3 in `tasks/T-AI-02-10-implemented.md`.
    pub fn api_key(&self, model: &Model) -> Option<String> {
        let config = self.providers.get(&model.provider)?;
        let raw = config.api_key.as_deref()?;
        resolve_config_value(raw)
    }

    /// Build the catalog from the resolved config and run schema semantic validation.
    fn from_config(config: ModelsConfig) -> Result<Catalog, CatalogError> {
        validate_config(&config)?;
        Ok(Catalog {
            providers: config.providers,
        })
    }
}

/// Strip JSONC comments and trailing commas (Rust character-by-character implementation of pi-341 `stripJsonComments`).
///
/// Handle only `//` line comments, `/* */` block comments, and trailing commas; `//`/`,` inside string literals are left untouched;
/// Insert a space where a comment was removed to prevent `1/*c*/2` from becoming `12`.
fn strip_json_comments(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    let mut in_string = false;
    while i < chars.len() {
        let c = chars[i];
        if in_string {
            out.push(c);
            if c == '\\' {
                if i + 1 < chars.len() {
                    out.push(chars[i + 1]);
                    i += 2;
                    continue;
                }
            } else if c == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if c == '"' {
            in_string = true;
            out.push(c);
            i += 1;
            continue;
        }
        if c == '/' && i + 1 < chars.len() && chars[i + 1] == '/' {
            push_space_if_needed(&mut out);
            i += 2;
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '/' && i + 1 < chars.len() && chars[i + 1] == '*' {
            push_space_if_needed(&mut out);
            i += 2;
            while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                i += 1;
            }
            i = (i + 2).min(chars.len());
            continue;
        }
        out.push(c);
        i += 1;
    }
    remove_trailing_commas(&out)
}

/// Replace an extracted comment with a space to keep adjacent tokens from sticking together.
fn push_space_if_needed(out: &mut String) {
    if !out.chars().last().is_some_and(char::is_whitespace) {
        out.push(' ');
    }
}

/// Remove trailing commas at the end of JSON objects/arrays (leave string literals untouched).
fn remove_trailing_commas(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    let mut in_string = false;
    while i < chars.len() {
        let c = chars[i];
        if in_string {
            out.push(c);
            if c == '\\' {
                if i + 1 < chars.len() {
                    out.push(chars[i + 1]);
                    i += 2;
                    continue;
                }
            } else if c == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if c == '"' {
            in_string = true;
            out.push(c);
            i += 1;
            continue;
        }
        if c == ',' {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && (chars[j] == '}' || chars[j] == ']') {
                i += 1;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

/// Parse a config value template: literal, `$VAR`, `${VAR}` (`$$` → `$`, `$!` → `!`).
///
/// If any referenced env var is missing or empty → `None` (empty string is not treated as a value). A trailing `!command` returns `None`.
/// Current consumers: `Catalog::api_key` and `ConfiguredProvider::resolve_api_key` (`pub(crate)`).
pub(crate) fn resolve_config_value(raw: &str) -> Option<String> {
    if raw.starts_with('!') {
        return None;
    }
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '$' {
            out.push(c);
            continue;
        }
        match chars.peek().copied() {
            Some('$') => {
                chars.next();
                out.push('$');
            }
            Some('!') => {
                chars.next();
                out.push('!');
            }
            Some('{') => {
                chars.next();
                let mut name = String::new();
                let mut closed = false;
                for ch in chars.by_ref() {
                    if ch == '}' {
                        closed = true;
                        break;
                    }
                    name.push(ch);
                }
                if closed && is_env_name(&name) {
                    out.push_str(&env_value(&name)?);
                } else {
                    out.push_str("${");
                    out.push_str(&name);
                    if closed {
                        out.push('}');
                    }
                }
            }
            Some(next) if is_env_start(next) => {
                let mut name = String::new();
                while let Some(&next) = chars.peek() {
                    if is_env_continue(next) {
                        name.push(next);
                        chars.next();
                    } else {
                        break;
                    }
                }
                out.push_str(&env_value(&name)?);
            }
            _ => out.push('$'),
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Whether an environment variable name is valid (`[A-Za-z_][A-Za-z0-9_]*`).
fn is_env_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) if is_env_start(first) => chars.all(is_env_continue),
        _ => false,
    }
}

/// First character of an environment variable name.
fn is_env_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

/// Subsequent characters of an environment variable name.
fn is_env_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Read an environment variable; missing or empty → `None`.
fn env_value(name: &str) -> Option<String> {
    get_provider_env_value(name, None).filter(|value| !value.is_empty())
}

/// Parse a JSON/JSONC text fragment into [`ModelsConfig`].
fn parse_config(content: &str, path: &str) -> Result<ModelsConfig, CatalogError> {
    let stripped = strip_json_comments(content);
    serde_json::from_str(&stripped).map_err(|source| CatalogError::Parse {
        path: path.to_string(),
        source,
    })
}

/// Schema semantic validation: a defined model must resolve to an `api` and a non-empty `baseUrl`; size fields must not be 0.
fn validate_config(config: &ModelsConfig) -> Result<(), CatalogError> {
    for (provider, provider_config) in &config.providers {
        if provider.trim().is_empty() {
            return Err(CatalogError::Invalid(
                "provider id must not be empty".to_string(),
            ));
        }
        for def in provider_config.models.as_deref().unwrap_or(&[]) {
            if def.id.trim().is_empty() {
                return Err(CatalogError::Invalid(format!(
                    "provider {provider}: model id must not be empty"
                )));
            }
            if def.api.or(provider_config.api).is_none() {
                return Err(CatalogError::Invalid(format!(
                    "provider {provider}, model {}: missing \"api\" (set it at provider or model level)",
                    def.id
                )));
            }
            let base_url = def
                .base_url
                .as_deref()
                .or(provider_config.base_url.as_deref());
            if base_url.map(str::is_empty).unwrap_or(true) {
                return Err(CatalogError::Invalid(format!(
                    "provider {provider}, model {}: \"baseUrl\" is required when defining models",
                    def.id
                )));
            }
            if def.context_window == Some(0) {
                return Err(CatalogError::Invalid(format!(
                    "provider {provider}, model {}: invalid contextWindow",
                    def.id
                )));
            }
            if def.max_tokens == Some(0) {
                return Err(CatalogError::Invalid(format!(
                    "provider {provider}, model {}: invalid maxTokens",
                    def.id
                )));
            }
        }
        for (model_id, model_override) in provider_config
            .model_overrides
            .as_ref()
            .into_iter()
            .flatten()
        {
            if model_override.context_window == Some(0) {
                return Err(CatalogError::Invalid(format!(
                    "provider {provider}, model override {model_id}: invalid contextWindow"
                )));
            }
            if model_override.max_tokens == Some(0) {
                return Err(CatalogError::Invalid(format!(
                    "provider {provider}, model override {model_id}: invalid maxTokens"
                )));
            }
        }
    }
    Ok(())
}

/// Overlay user config onto the built-in config.
fn merge_config(base: &mut ModelsConfig, user: ModelsConfig) {
    for (provider, user_config) in user.providers {
        match base.providers.remove(&provider) {
            Some(mut base_config) => {
                merge_provider(&mut base_config, user_config);
                base.providers.insert(provider, base_config);
            }
            None => {
                base.providers.insert(provider, user_config);
            }
        }
    }
}

/// Provider-level merge: scalars overridden by user, `headers`/`compat` merged per key, `models` merged by id.
fn merge_provider(base: &mut ProviderConfig, user: ProviderConfig) {
    if user.name.is_some() {
        base.name = user.name;
    }
    if user.api.is_some() {
        base.api = user.api;
    }
    if user.base_url.is_some() {
        base.base_url = user.base_url;
    }
    if user.api_key.is_some() {
        base.api_key = user.api_key;
    }
    if let Some(headers) = user.headers {
        base.headers
            .get_or_insert_with(HashMap::new)
            .extend(headers);
    }
    base.compat = merge_compat(base.compat.as_ref(), user.compat.as_ref());
    if let Some(user_models) = user.models {
        let base_models = base.models.get_or_insert_with(Vec::new);
        for user_model in user_models {
            if let Some(base_model) = base_models
                .iter_mut()
                .find(|model| model.id == user_model.id)
            {
                merge_model_def(base_model, user_model);
            } else {
                base_models.push(user_model);
            }
        }
    }
    if let Some(user_overrides) = user.model_overrides {
        let base_overrides = base.model_overrides.get_or_insert_with(HashMap::new);
        for (model_id, user_override) in user_overrides {
            if let Some(base_override) = base_overrides.get_mut(&model_id) {
                merge_model_override(base_override, user_override);
            } else {
                base_overrides.insert(model_id, user_override);
            }
        }
    }
}

/// Model definition merge: non-empty user fields override built-in fields (missing fields inherit built-in values, preventing 1M context fallback).
fn merge_model_def(base: &mut ModelDef, user: ModelDef) {
    if user.name.is_some() {
        base.name = user.name;
    }
    if user.api.is_some() {
        base.api = user.api;
    }
    if user.base_url.is_some() {
        base.base_url = user.base_url;
    }
    if user.reasoning.is_some() {
        base.reasoning = user.reasoning;
    }
    if user.context_window.is_some() {
        base.context_window = user.context_window;
    }
    if user.max_tokens.is_some() {
        base.max_tokens = user.max_tokens;
    }
    if let Some(headers) = user.headers {
        base.headers
            .get_or_insert_with(HashMap::new)
            .extend(headers);
    }
    if user.cost.is_some() {
        base.cost = user.cost;
    }
    base.compat = merge_compat(base.compat.as_ref(), user.compat.as_ref());
}

/// Model patch merge: non-empty user fields override built-in patches.
fn merge_model_override(base: &mut ModelOverride, user: ModelOverride) {
    if user.name.is_some() {
        base.name = user.name;
    }
    if user.reasoning.is_some() {
        base.reasoning = user.reasoning;
    }
    if user.context_window.is_some() {
        base.context_window = user.context_window;
    }
    if user.max_tokens.is_some() {
        base.max_tokens = user.max_tokens;
    }
    if let Some(headers) = user.headers {
        base.headers
            .get_or_insert_with(HashMap::new)
            .extend(headers);
    }
    if user.cost.is_some() {
        base.cost = user.cost;
    }
    base.compat = merge_compat(base.compat.as_ref(), user.compat.as_ref());
}

/// Compatibility-switch field-by-field merge: user `Some` takes priority, otherwise inherit from base.
fn merge_compat(
    base: Option<&OpenAiCompletionsCompat>,
    user: Option<&OpenAiCompletionsCompat>,
) -> Option<OpenAiCompletionsCompat> {
    match (base, user) {
        (None, None) => None,
        (Some(base), None) => Some(base.clone()),
        (None, Some(user)) => Some(user.clone()),
        (Some(base), Some(user)) => Some(OpenAiCompletionsCompat {
            thinking_format: user.thinking_format.or(base.thinking_format),
            max_tokens_field: user.max_tokens_field.or(base.max_tokens_field),
            supports_usage_in_streaming: user
                .supports_usage_in_streaming
                .or(base.supports_usage_in_streaming),
            supports_developer_role: user
                .supports_developer_role
                .or(base.supports_developer_role),
            supports_reasoning_effort: user
                .supports_reasoning_effort
                .or(base.supports_reasoning_effort),
            send_session_affinity_headers: user
                .send_session_affinity_headers
                .or(base.send_session_affinity_headers),
            session_affinity_format: user
                .session_affinity_format
                .or(base.session_affinity_format),
        }),
    }
}

/// Materialize a `ModelDef` (with `modelOverrides` applied) into a [`Model`].
fn materialize_model(provider: &str, config: &ProviderConfig, def: &ModelDef) -> Model {
    let override_config = config
        .model_overrides
        .as_ref()
        .and_then(|overrides| overrides.get(&def.id));

    let mut headers = config.headers.clone().unwrap_or_default();
    if let Some(model_headers) = &def.headers {
        headers.extend(model_headers.clone());
    }
    if let Some(override_headers) = override_config.and_then(|item| item.headers.as_ref()) {
        headers.extend(override_headers.clone());
    }

    let compat = merge_compat(config.compat.as_ref(), def.compat.as_ref());
    let compat = merge_compat(
        compat.as_ref(),
        override_config.and_then(|item| item.compat.as_ref()),
    );

    Model {
        id: def.id.clone(),
        name: override_config
            .and_then(|item| item.name.clone())
            .or_else(|| def.name.clone()),
        provider: provider.to_string(),
        api: def.api.or(config.api).unwrap_or(Api::OpenAiCompletions),
        base_url: def
            .base_url
            .clone()
            .or_else(|| config.base_url.clone())
            .unwrap_or_default(),
        context_window: override_config
            .and_then(|item| item.context_window)
            .or(def.context_window)
            .unwrap_or(DEFAULT_CONTEXT_WINDOW),
        max_tokens: override_config
            .and_then(|item| item.max_tokens)
            .or(def.max_tokens)
            .unwrap_or(DEFAULT_MAX_TOKENS),
        reasoning: override_config
            .and_then(|item| item.reasoning)
            .or(def.reasoning)
            .unwrap_or(false),
        headers: if headers.is_empty() {
            None
        } else {
            Some(headers)
        },
        cost_rates: override_config
            .and_then(|item| item.cost.clone())
            .or_else(|| def.cost.clone()),
        compat,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ThinkingFormat;

    /// Process-unique temporary file path (avoids pulling in an extra dev-dependency).
    fn temp_models_path(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "zen-models-{tag}-{}-{nanos}.json",
            std::process::id()
        ))
    }

    // ── Built-in catalog ──

    #[test]
    fn embedded_catalog_parses() {
        let catalog = Catalog::embedded();
        let opencode = catalog
            .model("opencode-go", "deepseek-v4-flash")
            .expect("opencode-go model");
        assert_eq!(opencode.provider, "opencode-go");
        assert_eq!(opencode.id, "deepseek-v4-flash");
        assert_eq!(opencode.base_url, "https://opencode.ai/zen/go/v1");
        assert_eq!(opencode.context_window, 1_000_000);
        assert_eq!(opencode.max_tokens, 384_000);
        assert!(opencode.reasoning);
        assert_eq!(
            opencode.cost_rates.as_ref().map(|rates| rates.input),
            Some(0.5)
        );
        assert_eq!(
            opencode
                .compat
                .as_ref()
                .and_then(|compat| compat.thinking_format),
            Some(ThinkingFormat::OpenAi)
        );

        let huoshan = catalog
            .model("huoshan", "deepseek-v4-flash")
            .expect("huoshan model");
        assert_eq!(
            huoshan.base_url,
            "https://ark.cn-beijing.volces.com/api/coding/v3"
        );
        assert_eq!(
            huoshan
                .compat
                .as_ref()
                .and_then(|compat| compat.thinking_format),
            Some(ThinkingFormat::DeepSeek)
        );
    }

    #[test]
    fn embedded_catalog_models_are_sorted() {
        let models = Catalog::embedded().models();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].provider, "huoshan");
        assert_eq!(models[1].provider, "opencode-go");
    }

    #[test]
    fn default_model_resolves_default_provider() {
        let model = Catalog::embedded().default_model().expect("default model");
        assert!(!model.provider.is_empty());
        assert!(!model.id.is_empty());
        assert_eq!(model.context_window, 1_000_000);
    }

    #[test]
    fn provider_resolves_from_catalog() {
        let catalog = Catalog::embedded();
        let provider = catalog.provider("opencode-go").expect("provider");
        assert_eq!(provider.id(), "opencode-go");
        assert_eq!(provider.base_url(), "https://opencode.ai/zen/go/v1");
        assert!(catalog.provider("missing").is_none());
    }

    // ── load / merge ──

    #[test]
    fn load_missing_user_file_uses_embedded() {
        let path = temp_models_path("missing");
        let catalog = Catalog::load(Some(&path)).expect("load");
        assert!(catalog.model("opencode-go", "deepseek-v4-flash").is_some());
    }

    #[test]
    fn load_none_uses_embedded() {
        let catalog = Catalog::load(None).expect("load");
        assert!(catalog.model("huoshan", "deepseek-v4-flash").is_some());
    }

    #[test]
    fn load_user_file_overrides_provider_and_model() {
        let path = temp_models_path("override");
        std::fs::write(
            &path,
            r#"{
              "providers": {
                "opencode-go": {
                  "baseUrl": "https://example.test/v1",
                  "compat": { "thinkingFormat": "deepseek" },
                  "modelOverrides": { "deepseek-v4-flash": { "maxTokens": 1234 } }
                },
                "my-compat": {
                  "api": "openai-completions",
                  "baseUrl": "https://my.test/v1",
                  "apiKey": "sk-test",
                  "models": [{ "id": "m1" }]
                }
              }
            }"#,
        )
        .expect("write temp models.json");

        let catalog = Catalog::load(Some(&path)).expect("load");
        let existing = catalog
            .model("opencode-go", "deepseek-v4-flash")
            .expect("existing model");
        assert_eq!(existing.base_url, "https://example.test/v1");
        assert_eq!(existing.max_tokens, 1234);
        // Fields not overridden by the user inherit the built-in record (prevents 1M context fallback).
        assert_eq!(existing.context_window, 1_000_000);
        assert_eq!(
            existing
                .compat
                .as_ref()
                .and_then(|compat| compat.thinking_format),
            Some(ThinkingFormat::DeepSeek)
        );

        let custom = catalog.model("my-compat", "m1").expect("custom model");
        assert_eq!(custom.provider, "my-compat");
        assert_eq!(custom.base_url, "https://my.test/v1");
        assert_eq!(custom.context_window, DEFAULT_CONTEXT_WINDOW);
        assert_eq!(custom.max_tokens, DEFAULT_MAX_TOKENS);
        assert_eq!(catalog.api_key(&custom), Some("sk-test".to_string()));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_user_file_tolerates_jsonc() {
        let path = temp_models_path("jsonc");
        std::fs::write(
            &path,
            r#"{
              // line comment
              "providers": {
                "x": {
                  "api": "openai-completions",
                  "baseUrl": "https://x.test",
                  "models": [{ "id": "m" },],
                },
              },
            }"#,
        )
        .expect("write temp jsonc");
        let catalog = Catalog::load(Some(&path)).expect("load");
        assert!(catalog.model("x", "m").is_some());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_rejects_model_without_base_url() {
        let path = temp_models_path("invalid");
        std::fs::write(
            &path,
            r#"{"providers":{"x":{"api":"openai-completions","models":[{"id":"m"}]}}}"#,
        )
        .expect("write temp models.json");
        let err = Catalog::load(Some(&path)).expect_err("must reject");
        assert!(matches!(err, CatalogError::Invalid(_)));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_rejects_model_without_api() {
        let path = temp_models_path("no-api");
        std::fs::write(
            &path,
            r#"{"providers":{"x":{"baseUrl":"https://x.test","models":[{"id":"m"}]}}}"#,
        )
        .expect("write temp models.json");
        let err = Catalog::load(Some(&path)).expect_err("must reject");
        assert!(matches!(err, CatalogError::Invalid(_)));
        let _ = std::fs::remove_file(&path);
    }

    // ── strip_json_comments ──

    #[test]
    fn strip_json_comments_leaves_strings_untouched() {
        let input = r#"{"a": "http://x//y", /* block */ "b": [1,2,], }"#;
        let output = strip_json_comments(input);
        let value: serde_json::Value = serde_json::from_str(&output).expect("valid json");
        assert_eq!(value["a"], "http://x//y");
        assert_eq!(value["b"], serde_json::json!([1, 2]));
    }

    #[test]
    fn strip_json_comments_removes_line_comments() {
        let input = "{\n  // note\n  \"a\": 1\n}";
        assert_eq!(strip_json_comments(input).trim(), "{\n  \n  \"a\": 1\n}");
    }

    // ── resolve_config_value ──

    #[test]
    fn resolve_config_value_literal_and_escape() {
        assert_eq!(resolve_config_value("sk-test"), Some("sk-test".to_string()));
        assert_eq!(resolve_config_value("$$PATH"), Some("$PATH".to_string()));
        assert_eq!(resolve_config_value("$!"), Some("!".to_string()));
        assert_eq!(resolve_config_value(""), None);
    }

    #[test]
    fn resolve_config_value_env_template() {
        let path = std::env::var("PATH").expect("PATH set");
        assert_eq!(resolve_config_value("$PATH"), Some(path.clone()));
        assert_eq!(resolve_config_value("${PATH}"), Some(path));
    }

    #[test]
    fn resolve_config_value_missing_env_is_none() {
        assert_eq!(
            resolve_config_value("$ZEN_THIS_ENV_SHOULD_NOT_EXIST_9F3A"),
            None
        );
        assert_eq!(resolve_config_value("!echo hi"), None);
    }
}
