pub mod defaults;
pub mod settings;

use std::path::PathBuf;
use zen_core::paths::{normalize_path, PathError, PathInputOptions};

// ── Identity constants (module-level pub const, determined at compile time) ─────────────────────
pub const APP_NAME: &str = "zen";
pub const APP_TITLE: &str = "zen";
pub const PACKAGE_NAME: &str = env!("CARGO_PKG_NAME");
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const CONFIG_DIR_NAME: &str = ".zen";
pub const ENV_AGENT_DIR: &str = "ZEN_CODING_AGENT_DIR";
pub const ENV_SESSION_DIR: &str = "ZEN_CODING_AGENT_SESSION_DIR";

// ── Path detection ─────────────────────────────────────────────────────

/// Global agent config directory.
/// Prefers the override value from the environment variable `ENV_AGENT_DIR` (normalized via `normalize_path`, supports `~`/`file://`);
/// otherwise returns `<home>/{CONFIG_DIR_NAME}/agent` (`home` expanded via `normalize_path("~/...")`).
/// # Errors
/// env override missing and home unavailable → `PathError::NoHomeDir`。
pub fn get_agent_dir() -> Result<PathBuf, PathError> {
    // ① env override takes precedence: empty string treated as unset (aligns with pi `if (envDir)` truthiness check)
    if let Some(dir) = std::env::var_os(ENV_AGENT_DIR).filter(|v| !v.is_empty()) {
        return Ok(PathBuf::from(normalize_path(
            // The standard bridge for converting `OsString`/`OsStr` (platform-native strings) to `&str`; the lossy in the name is key
            &dir.to_string_lossy(),
            &PathInputOptions::default(),
        )?));
    }
    // ② default `<home>/.zen/agent`: borrow normalize_path's `~` expansion to get home,
    //    so "home unavailable" is uniformly expressed by PathError::NoHomeDir
    Ok(PathBuf::from(normalize_path(
        &format!("~/{CONFIG_DIR_NAME}/agent"),
        &PathInputOptions::default(),
    )?))
}

/// User settings file path: `<agentDir>/settings.json`.
pub fn get_settings_path() -> Result<PathBuf, PathError> {
    Ok(get_agent_dir()?.join("settings.json"))
}

/// User keybinding config file path: `<agentDir>/keybindings.json` (aligned with pi `KeybindingsManager::create`).
/// Source of truth for user keybinding overrides, consumed by the T-AP-06-2 loader.
/// # Errors
/// Depends on `get_agent_dir`; when env overrides are missing and home is unavailable → `PathError::NoHomeDir`.
pub fn get_keybindings_path() -> Result<PathBuf, PathError> {
    Ok(get_agent_dir()?.join("keybindings.json"))
}

/// Self-installed managed binaries directory: `<agentDir>/bin` (fd/rg etc.).
/// # Consumers
/// T-AP-04-2 (`getShellEnv` prepends bin to PATH, see T-AP-09-3 open question 1)
pub fn get_bin_dir() -> Result<PathBuf, PathError> {
    Ok(get_agent_dir()?.join("bin"))
}

/// Session persistence directory: `<agentDir>/sessions`.
/// # Consumers
/// T-AP-01-4 (session manager create/list/resume/fork)
pub fn get_sessions_dir() -> Result<PathBuf, PathError> {
    Ok(get_agent_dir()?.join("sessions"))
}

/// Diagnostic log file path: `<agentDir>/{APP_NAME}-debug.log` (i.e. `zen-debug.log`).
/// # Consumers
/// REQ-OBS-01 (diagnostic output), T-AP-08-3 (main assembly)
pub fn get_debug_log_path() -> Result<PathBuf, PathError> {
    Ok(get_agent_dir()?.join(format!("{APP_NAME}-debug.log")))
}

/// Custom theme directory: `<agentDir>/themes`.
/// Where users place custom theme JSON files.
/// # Consumers
/// T-AP-11-3 (ThemeController scans custom themes)
pub fn get_custom_themes_dir() -> Result<PathBuf, PathError> {
    Ok(get_agent_dir()?.join("themes"))
}

/// Global prompt template directory: `<agentDir>/prompts`.
/// Used by loader assembly for the User default directory (T-AP-03-4) and resource loading (T-AP-05-3)
/// (T-AP-02-1 deferred annotation placeholder, aligning with pi `getPromptsDir`).
/// # Errors
/// Depends on `get_agent_dir`; when env overrides are missing and home is unavailable → `PathError::NoHomeDir`.
pub fn get_prompts_dir() -> Result<PathBuf, PathError> {
    Ok(get_agent_dir()?.join("prompts"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::sync::Mutex;

    // All functions in this module read the process environment (ENV_AGENT_DIR / HOME); when cargo test runs in parallel by default,
    // set_var calls interfere with each other, so a single static lock serializes tests that touch environment variables (standard idiom, not a mock).
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Test helper: temporarily set an environment variable, restoring the original value when leaving scope (remove it if it was previously unset).
    struct EnvVarGuard {
        key: &'static str,
        old: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let old = std::env::var_os(key);
            std::env::set_var(key, value);
            Self { key, old }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.old {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }

    // ── get_agent_dir: env override branch ──────────────────────────────

    #[test]
    fn test_get_agent_dir_env_override_absolute() {
        let _l = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::set(ENV_AGENT_DIR, "/fake/zen-agent");
        assert_eq!(get_agent_dir().unwrap(), PathBuf::from("/fake/zen-agent"));
    }

    #[cfg(unix)]
    #[test]
    fn test_get_agent_dir_env_override_expands_tilde() {
        let _l = ENV_LOCK.lock().unwrap();
        // env override value is processed by normalize_path: `~` expansion (depends on home::home_dir() reading $HOME)
        let _h = EnvVarGuard::set("HOME", "/fake/home");
        let _g = EnvVarGuard::set(ENV_AGENT_DIR, "~/zen-agent");
        assert_eq!(
            get_agent_dir().unwrap(),
            PathBuf::from("/fake/home/zen-agent")
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_get_agent_dir_env_override_file_url() {
        let _l = ENV_LOCK.lock().unwrap();
        // env override value supports file:// URLs (normalize_path's url crate conversion)
        let _g = EnvVarGuard::set(ENV_AGENT_DIR, "file:///fake/zen-agent");
        assert_eq!(get_agent_dir().unwrap(), PathBuf::from("/fake/zen-agent"));
    }

    // ── get_agent_dir: default branch ──────────────────────────────────

    #[cfg(unix)]
    #[test]
    fn test_get_agent_dir_default_uses_home() {
        let _l = ENV_LOCK.lock().unwrap();
        // ENV_AGENT_DIR unset → `<home>/.zen/agent`; override with $HOME to get a deterministic home.
        // Assume home::home_dir() reads $HOME on unix (see home crate behavior).
        let _g = EnvVarGuard::set("HOME", "/fake/home");
        std::env::remove_var(ENV_AGENT_DIR);
        assert_eq!(
            get_agent_dir().unwrap(),
            PathBuf::from("/fake/home/.zen/agent")
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_get_agent_dir_empty_override_falls_back_to_default() {
        let _l = ENV_LOCK.lock().unwrap();
        // Empty-string env override is treated as "unset" (aligning with pi's truthiness check on `if (envDir)`) -> falls through to the default branch
        let _h = EnvVarGuard::set("HOME", "/fake/home");
        let _g = EnvVarGuard::set(ENV_AGENT_DIR, "");
        assert_eq!(
            get_agent_dir().unwrap(),
            PathBuf::from("/fake/home/.zen/agent")
        );
    }

    // ── Subpath joining (validate each subpath name under the ENV_AGENT_DIR override) ──────

    #[test]
    fn test_get_settings_path_appends_settings_json() {
        let _l = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::set(ENV_AGENT_DIR, "/fake/agent");
        assert_eq!(
            get_settings_path().unwrap(),
            PathBuf::from("/fake/agent/settings.json")
        );
    }

    #[test]
    fn test_get_keybindings_path_appends_keybindings_json() {
        let _l = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::set(ENV_AGENT_DIR, "/fake/agent");
        assert_eq!(
            get_keybindings_path().unwrap(),
            PathBuf::from("/fake/agent/keybindings.json")
        );
    }

    #[test]
    fn test_get_bin_dir_appends_bin() {
        let _l = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::set(ENV_AGENT_DIR, "/fake/agent");
        assert_eq!(get_bin_dir().unwrap(), PathBuf::from("/fake/agent/bin"));
    }

    #[test]
    fn test_get_sessions_dir_appends_sessions() {
        let _l = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::set(ENV_AGENT_DIR, "/fake/agent");
        assert_eq!(
            get_sessions_dir().unwrap(),
            PathBuf::from("/fake/agent/sessions")
        );
    }

    #[test]
    fn test_get_debug_log_path_appends_app_name_debug_log() {
        let _l = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::set(ENV_AGENT_DIR, "/fake/agent");
        // Log name = {APP_NAME}-debug.log, i.e. zen-debug.log
        assert_eq!(
            get_debug_log_path().unwrap(),
            PathBuf::from(format!("/fake/agent/{APP_NAME}-debug.log"))
        );
    }

    #[test]
    fn test_get_prompts_dir_appends_prompts() {
        let _l = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::set(ENV_AGENT_DIR, "/fake/agent");
        assert_eq!(
            get_prompts_dir().unwrap(),
            PathBuf::from("/fake/agent/prompts")
        );
    }
}
