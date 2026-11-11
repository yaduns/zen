//! Configuration surface: identity constants (app name / version / config directory names) and path detection
//! (agent dir, settings, sessions, prompts, themes, keybindings, models) plus the cross-tool shared skill
//! directory discovery (`.agents`).
//!
//! REQ-APP-02 (pi-359): identity constants + path detection.
//! REQ-APP-03 (pi-318): cross-tool shared skill directory discovery (`<dir>/.agents/skills`, read-only) —
//! consumed by `crate::skills::load_skills`; `.agents` never receives writes (`.zen` is the only write target).

pub mod catalog;
pub mod defaults;
pub mod settings;

use std::path::{Path, PathBuf};
use zen_core::paths::{
    canonicalize_path, normalize_path, resolve_path, PathError, PathInputOptions,
};

// ── Identity constants (module-level pub const, determined at compile time) ─────────────────────
pub const APP_NAME: &str = "zen";
pub const APP_TITLE: &str = "zen";
pub const PACKAGE_NAME: &str = env!("CARGO_PKG_NAME");
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const CONFIG_DIR_NAME: &str = ".zen";
/// Cross-tool shared config directory name (`.agents`) — coexists with the tool-private [`CONFIG_DIR_NAME`] (`.zen`):
/// `.zen` remains this tool's only private config location **and the only write target**, while `.agents` is a
/// **read-only** cross-tool shared source (the same convention pi / codex use).
///
/// v1 consumes only its `skills/` child (`<dir>/.agents/skills`); prompts / themes / `SYSTEM.md` / `settings.json`
/// are never read from `.agents` and nothing is ever written there.
///
/// # Consumers
/// [`collect_project_shared_config_dirs`] / [`collect_project_shared_skill_dirs`] / [`shared_skills_dir_for_home`]
/// (this module); T-V2-01-1 (trust ancestor probe) and T-V2-01-2 (untrusted-project load short-circuit) are the
/// planned consumers registered as seams in `docs/pi-refactoring-plan.md` §8.
pub const SHARED_CONFIG_DIR_NAME: &str = ".agents";
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

/// User-level shared skill directory for an explicitly supplied home directory (`<home>/.agents/skills`).
///
/// Path-joining primitive for callers that already hold a home value (production: the real home; tests: a tempdir),
/// so that resolving the cross-tool shared skill directory never requires reading an environment variable.
///
/// # Parameters
/// - `home`: home directory base (no existence check, no filesystem access).
///
/// # Consumers
/// `crate::skills::shared_skill_dirs` (when the assembly input carries `home_dir`); [`get_user_shared_skills_dir`].
pub fn shared_skills_dir_for_home(home: &Path) -> PathBuf {
    home.join(SHARED_CONFIG_DIR_NAME).join("skills")
}

/// User-level shared skill directory `~/.agents/skills` (real-home variant) — the "cross-tool global skills"
/// segment of skill assembly, so that a single `~/.agents/skills` directory serves zen / pi / codex at once.
///
/// Borrows `normalize_path("~/...")` for home expansion (same technique as [`get_agent_dir`]: an unresolvable `~`
/// is uniformly reported as `PathError::NoHomeDir`).
///
/// # Errors
/// Home directory unavailable → `PathError::NoHomeDir` (the caller silently skips the segment, aligning with pi
/// not aborting skill loading when HOME is missing).
///
/// # Consumers
/// `crate::skills::shared_skill_dirs` (when `LoadSkillsOptions::home_dir` is `None`).
pub fn get_user_shared_skills_dir() -> Result<PathBuf, PathError> {
    Ok(PathBuf::from(normalize_path(
        &format!("~/{SHARED_CONFIG_DIR_NAME}/skills"),
        &PathInputOptions::default(),
    )?))
}

/// Ancestor chain of cross-tool shared config directories (`.agents` itself, **not** its `skills` child) — the
/// single upward-walk primitive shared by skill discovery and (later) trust probing.
///
/// Walks from `cwd` upward collecting `<dir>/.agents`, nearest first; stops **at** `cutoff` (inclusive) when given,
/// otherwise walks up to the filesystem root. The input is normalized through `zen_core::paths::resolve_path`
/// (aligning with pi `resolve(startDir)`).
///
/// # Parameters
/// - `cwd`: session working directory (relative paths resolve against the process cwd);
/// - `cutoff`: stop boundary — skill discovery passes the git repository root (pi `collectAncestorAgentsSkillDirs`),
///   trust probing passes `None` (pi `has_trust_requiring_project_resources` walks to the filesystem root).
///
/// # Returns
/// `<dir>/.agents` for every level; an empty vec when `cwd` cannot be resolved (never panics).
///
/// # Boundaries
/// Performs **no existence filtering and no `~/.agents` exclusion** — both are caller semantics (exclusion lives in
/// `crate::skills::shared_skill_dirs`). Hence the returned list **may contain `~/.agents`** (when `cwd` sits under
/// home); direct consumers must exclude it themselves (pi `trust-manager.ts:194` has the same requirement).
///
/// # Consumers
/// [`collect_project_shared_skill_dirs`] (this module); planned consumer T-V2-01-1 (`cutoff = None`).
pub fn collect_project_shared_config_dirs(cwd: &Path, cutoff: Option<&Path>) -> Vec<PathBuf> {
    let start = resolve_shared_walk_start(cwd);
    walk_shared_config_dirs(&start, cutoff)
}

/// Project-level shared skill directory chain `<dir>/.agents/skills` (cwd → git repository root) — the
/// "cross-tool project skills" segment of skill assembly, so that one repo-root `.agents/skills` is shared by
/// every tool operating inside that repo.
///
/// = [`collect_project_shared_config_dirs`] with `cutoff = find_git_repo_root(cwd)`, each level `join("skills")`
/// (aligning with pi `join(dir, ".agents", "skills")`). Existence filtering and `~/.agents` exclusion are likewise
/// left to the caller (a missing directory loads nothing; exclusion lives in `crate::skills::shared_skill_dirs`),
/// so the returned list may contain `~/.agents/skills` when `cwd` is under home.
///
/// # Consumers
/// `crate::skills::shared_skill_dirs` (current sole consumer).
pub fn collect_project_shared_skill_dirs(cwd: &Path) -> Vec<PathBuf> {
    let start = resolve_shared_walk_start(cwd);
    let git_root = find_git_repo_root(&start);
    walk_shared_config_dirs(&start, git_root.as_deref())
        .into_iter()
        .map(|dir| dir.join("skills"))
        .collect()
}

/// Finds the git repository root at or above `start` (criterion: `<dir>/.git` **exists**; whether it is a directory
/// or a worktree `gitdir:` file is irrelevant, and `HEAD` is not inspected).
///
/// Aligns with pi `findGitRepoRoot` (`package-manager.ts:427-438`). Deliberately duplicates
/// `output::footer_data::find_git_paths` (T-AP-06-3): the latter parses worktree `gitdir:` / `commondir` and requires
/// `HEAD` to exist — merging them would leak the footer's "no HEAD → give up" condition into skill discovery
/// (a repo without HEAD would silently lose its `.agents` skills). pi also ships both implementations.
///
/// # Returns
/// The deepest directory containing `.git`; `None` when the walk reaches the filesystem root without a hit.
fn find_git_repo_root(start: &Path) -> Option<PathBuf> {
    let mut dir = start;
    loop {
        if dir.join(".git").exists() {
            return Some(dir.to_path_buf());
        }
        dir = dir.parent()?;
    }
}

/// Normalizes the walk start point: relative `cwd` resolves against the process cwd, `~` expands via
/// `normalize_path` (pi `resolve(startDir)` equivalent). Falls back to the raw path when normalization fails
/// (e.g. no home for a `~` input) — the walk then simply yields the levels it can see, never panicking.
fn resolve_shared_walk_start(cwd: &Path) -> PathBuf {
    let base = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    resolve_path(&cwd.to_string_lossy(), &base, &PathInputOptions::default())
        .unwrap_or_else(|_| cwd.to_path_buf())
}

/// Walks `start` upward pushing `<dir>/.agents` per level, stopping **after** pushing `cutoff` (inclusive) or at the
/// filesystem root when `cutoff` is `None`/not an ancestor. Comparison uses `canonicalize_path` on both sides so a
/// symlinked `/var` vs `/private/var` (macOS tempdirs) still matches.
fn walk_shared_config_dirs(start: &Path, cutoff: Option<&Path>) -> Vec<PathBuf> {
    let stop = cutoff.map(canonicalize_path);
    let mut dirs = Vec::new();
    let mut dir = start.to_path_buf();
    loop {
        dirs.push(dir.join(SHARED_CONFIG_DIR_NAME));
        if let Some(stop) = &stop {
            if canonicalize_path(&dir) == *stop {
                break;
            }
        }
        let Some(parent) = dir.parent().map(Path::to_path_buf) else {
            break;
        };
        dir = parent;
    }
    dirs
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

/// User model catalog file path: `<agentDir>/models.json`.
///
/// Startup wiring and config commands need to know where the user's hand-written model catalog is — `init_user_catalog` reads it to decide which
/// user providers/models to overlay; T-V2-02-2/3 read it for listing and edit validation. A missing file is not an error
/// (`Catalog::load` treats it as "no user overrides" and uses only the built-in catalog).
/// # Errors
/// Depends on `get_agent_dir`; when env overrides are missing and home is unavailable → `PathError::NoHomeDir`.
/// # Consumers
/// `config::catalog::init_user_catalog` (current); T-V2-02-2 (list models) / T-V2-02-3 (config selection) planned to hook in.
pub fn get_models_path() -> Result<PathBuf, PathError> {
    Ok(get_agent_dir()?.join("models.json"))
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

    // ── Cross-tool shared skill directories (`.agents`) ──────────────────

    #[test]
    fn test_shared_skills_dir_for_home_is_pure_join() {
        // No env, no IO: just `<home>/.agents/skills`
        assert_eq!(
            shared_skills_dir_for_home(Path::new("/h")),
            PathBuf::from("/h/.agents/skills")
        );
    }

    #[test]
    fn test_get_user_shared_skills_dir_ends_with_shared_skills() {
        // Deliberately env-free: only the shape is asserted, so the test cannot race with other env-touching tests.
        if let Ok(dir) = get_user_shared_skills_dir() {
            assert!(dir.ends_with(".agents/skills"), "got {dir:?}");
        }
        // Err(NoHomeDir) → the real-home branch is untestable in this environment, nothing to assert
    }

    #[test]
    fn test_find_git_repo_root_picks_deepest() {
        let tmp = tempfile::tempdir().unwrap();
        let outer = tmp.path().join("outer");
        let inner = outer.join("inner");
        let deep = inner.join("pkg").join("src");
        std::fs::create_dir_all(&deep).unwrap();
        std::fs::create_dir_all(inner.join(".git")).unwrap();
        std::fs::create_dir_all(outer.join(".git")).unwrap();
        // Deepest `.git` wins (pi walks upward and returns the first hit)
        assert_eq!(find_git_repo_root(&deep), Some(inner));
        assert_eq!(find_git_repo_root(&outer), Some(outer.clone()));
    }

    #[test]
    fn test_find_git_repo_root_none_outside_repo() {
        let tmp = tempfile::tempdir().unwrap();
        let leaf = tmp.path().join("no-repo");
        std::fs::create_dir_all(&leaf).unwrap();
        // tmpdirs live outside any git repo (the test harness never creates .git in the components above)
        assert_eq!(find_git_repo_root(&leaf), None);
    }

    #[test]
    fn test_collect_project_shared_config_dirs_stops_at_cutoff() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("repo");
        let cwd = repo.join("pkg").join("src");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        std::fs::create_dir_all(&cwd).unwrap();

        let dirs = collect_project_shared_config_dirs(&cwd, Some(&repo));
        assert_eq!(
            dirs,
            vec![
                cwd.join(".agents"),
                repo.join("pkg").join(".agents"),
                repo.join(".agents"),
            ]
        );
    }

    #[test]
    fn test_collect_project_shared_config_dirs_walks_to_root_without_cutoff() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path().join("a").join("b");
        std::fs::create_dir_all(&cwd).unwrap();

        let dirs = collect_project_shared_config_dirs(&cwd, None);
        // One entry per path component (the root level `/` also pushes `/.agents`), nearest first
        assert_eq!(dirs.len(), cwd.components().count());
        assert_eq!(dirs.first(), Some(&cwd.join(".agents")));
        assert_eq!(dirs.last(), Some(&PathBuf::from("/.agents")));
    }

    #[test]
    fn test_collect_project_shared_skill_dirs_appends_skills_and_cuts_at_git_root() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("repo");
        let cwd = repo.join("sub");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        std::fs::create_dir_all(&cwd).unwrap();

        let dirs = collect_project_shared_skill_dirs(&cwd);
        assert_eq!(
            dirs,
            vec![
                cwd.join(".agents").join("skills"),
                repo.join(".agents").join("skills")
            ]
        );
    }
}
