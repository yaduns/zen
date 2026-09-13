//! Theme loading assembly layer: multi-directory assembly + source tracing + dedupe collisions.
//!
//! REQ-APP-05 (pi-318): resource loader (themes side).
//! Application-layer assembly completion form: single-file parsing is already covered by zen-tui (`load_theme_from_json`, T-AP-11-3 data model),
//! this module fills in the remaining capabilities of pi `DefaultResourceLoader.loadThemes` + `dedupeThemes` — default directories
//! (`<agentDir>/themes` + `<cwd>/.zen/themes`) + explicit paths (directories/`.json` files) assembly,
//! source tracing (lightweight `ThemeSource`; full SourceInfo belongs to T-AP-05-4), name dedupe and collision diagnostics;
//! mirrors the skills.rs (T-AP-03-3) assembly form. The per-directory scan primitive `load_themes_from_dir` is also used by
//! theme/mod.rs `load_custom_themes_from_dir` by delegation (eliminating duplicate scan logic).

use std::path::{Path, PathBuf};

use zen_core::paths::{resolve_path, PathInputOptions};
use zen_tui::theme::{self, Theme};

use crate::config::CONFIG_DIR_NAME;

/// Lightweight tracing identifier for theme origin — resolves the scope semantics of pi `createSyntheticSourceInfo`.
///
/// v1 has no extension system; pi's `source` is always `"local"` and `origin` is always `"top-level"`, so the only real semantics left is the source level
/// (user/project/temporary path); `baseDir` is derived by the consumer from the parent directory of `theme.source_path` and is not a field.
/// The full `SourceInfo` resource model (scope/origin/baseDir) is canonically owned by **T-AP-05-4** (pi-319).
/// an in-memory assembly artifact, not participating in serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThemeSource {
    /// Theme under the default directory `<agentDir>/themes` (pi scope `"user"`).
    User,
    /// Theme under the default directory `<cwd>/.zen/themes` (pi scope `"project"`).
    Project,
    /// Theme loaded from an explicit path (CLI `--theme` / settings.themes extension directory) (pi scope `"temporary"`).
    Path,
}

/// A loaded theme plus its origin — the zen assembly form of pi `Theme` (with sourceInfo/sourcePath):
/// zen-tui `Theme` (with name/source_path, finalized in T-AP-11-3) + application-layer source tracing.
/// `baseDir` is derived by the consumer from the parent directory of `theme.source_path` and is not a field.
#[derive(Debug)]
pub struct LoadedTheme {
    /// Theme already parsed by zen-tui (name/colors/source_path).
    pub theme: Theme,
    /// Theme origin (default directory user/project / explicit path).
    pub source: ThemeSource,
}

/// Payload of a same-name theme collision diagnostic — the theme-specific form of pi `ResourceCollision`, with `resourceType` always resolved as `"theme"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThemeCollision {
    /// Colliding theme name.
    pub name: String,
    /// Earlier-loaded path (first occurrence of name kept; if source_path is missing use `"<builtin>"`, matching pi `?? "<builtin>"`).
    pub winner_path: String,
    /// Later-loaded path (discarded; if source_path is missing use `"<builtin>"`).
    pub loser_path: String,
}

/// Non-fatal theme-load diagnostic — pi `ResourceDiagnostic`'s `type` string discriminant resolved into a Rust enum.
/// Note: pi's warning/error categories (reload-level "explicit path does not exist" uses `type:"error"`, inside loadThemes
/// uses warning) are unified as `Warning` in zen (following the `SkillLoadDiagnostic` precedent — T-AP-03-3 already mapped pi
/// reload-level errors to Warning), the two differ only in message text.
#[derive(Debug)]
pub enum ThemeLoadDiagnostic {
    /// Path/load warning (theme may not be loaded; pi `"warning"`/`"error"` both map here).
    Warning {
        /// Human-readable message.
        message: String,
        /// Associated path.
        path: String,
    },
    /// Same-name collision (later loader is discarded; pi type `"collision"`).
    Collision(ThemeCollision),
}

/// Return aggregate of `load_themes` — "warnings do not abort" semantics: failure of some themes does not affect the rest; diagnostics and themes may both be non-empty.
#[derive(Debug, Default)]
pub struct LoadThemesResult {
    /// Themes successfully loaded and deduplicated (in first-occurrence order of name).
    pub themes: Vec<LoadedTheme>,
    /// Load warnings and collision diagnostics.
    pub diagnostics: Vec<ThemeLoadDiagnostic>,
}

/// Assembly input for `load_themes` (mirrors `LoadSkillsOptions`). Pure input parameters, no behavior.
pub struct LoadThemesOptions {
    /// Session working directory (base for project theme directory `<cwd>/.zen/themes` + resolution base for explicit paths).
    pub cwd: PathBuf,
    /// Global agent config directory (base for `<agentDir>/themes`; produced by T-AP-02-1 `get_agent_dir`).
    pub agent_dir: PathBuf,
    /// Explicit theme paths (directory or `.json` file); CLI `--theme` + settings.themes are provided by the caller
    /// (T-AP-05-3/assembly chain) merged in.
    pub theme_paths: Vec<PathBuf>,
    /// When true, load the default directories `<agentDir>/themes` (User) + `<cwd>/.zen/themes` (Project).
    pub include_defaults: bool,
}

/// Assemble the theme list from all configuration locations — shared source for the resource loader (T-AP-05-3) and later interactive (T-AP-11-1 registration
/// /theme list): converge "default directories + explicit paths" once into a deduplicated theme set — "theme supermarket stocking:
/// bring in global, project, and specified ones; for duplicate names keep only the first-arrived".
///
/// Behavior (matches pi `DefaultResourceLoader.loadThemes` + `dedupeThemes`, mirrors the skills.rs assembly shape):
/// - When `include_defaults`, first load `<agentDir>/themes` (source=User) and `<cwd>/.zen/themes`
///   （source=Project，`CONFIG_DIR_NAME` = `.zen`）；
/// - Then process `theme_paths` in order (directory or `.json` file, source=Path): explicit paths are
///   resolved via `zen_core::paths::resolve_path` (relative to cwd, `trim: true`) — path-resolution failures are silently skipped;
///   does not exist → Warning "theme path does not exist" and continue; directory → scan each directory; `.json` file →
///   single-file load; otherwise → Warning "theme path is not a json file";
/// - Deduplicate by name keeping the first occurrence; later arrivals get a Collision diagnostic (winner/loser use `theme.source_path`,
///   missing `"<builtin>"`);
/// - Synchronous implementation (std::fs + in-memory parsing, no tokio dependency; matches the existing synchronous-loading precedent in theme/mod.rs;
///   the orchestration-side load_resources is async and can call it directly internally).
///
/// # Parameters
/// - `options`: assembly input (cwd/agent_dir/theme_paths/include_defaults).
///
/// # Returns
/// Deduplicated `LoadedTheme` list + non-aborting diagnostics.
pub fn load_themes(options: &LoadThemesOptions) -> LoadThemesResult {
    let mut result = LoadThemesResult::default();

    if options.include_defaults {
        // Default directory ①: <agentDir>/themes (User)
        load_themes_from_dir(
            &options.agent_dir.join("themes"),
            ThemeSource::User,
            &mut result.themes,
            &mut result.diagnostics,
        );
        // Default directory ②: <cwd>/.zen/themes (Project)
        load_themes_from_dir(
            &options.cwd.join(CONFIG_DIR_NAME).join("themes"),
            ThemeSource::Project,
            &mut result.themes,
            &mut result.diagnostics,
        );
    }

    // Explicit paths (source always Path): directory → scan; .json file → single file; other/missing → Warning
    for raw_path in &options.theme_paths {
        let resolved = match resolve_path(
            &raw_path.to_string_lossy(),
            &options.cwd,
            &PathInputOptions {
                trim: true,
                ..PathInputOptions::default()
            },
        ) {
            Ok(path) => path,
            Err(_) => continue, // Skip silently on path resolution failure (e.g. no home directory), aligning with the explicit path branch of skills.rs
        };

        if !resolved.exists() {
            result.diagnostics.push(ThemeLoadDiagnostic::Warning {
                message: "theme path does not exist".to_string(),
                path: resolved.display().to_string(),
            });
            continue;
        }

        if resolved.is_dir() {
            load_themes_from_dir(
                &resolved,
                ThemeSource::Path,
                &mut result.themes,
                &mut result.diagnostics,
            );
        } else if resolved.extension().is_some_and(|e| e == "json") {
            load_theme_from_file(
                &resolved,
                ThemeSource::Path,
                &mut result.themes,
                &mut result.diagnostics,
            );
        } else {
            result.diagnostics.push(ThemeLoadDiagnostic::Warning {
                message: "theme path is not a json file".to_string(),
                path: resolved.display().to_string(),
            });
        }
    }

    // Deduplicate + collision diagnostics (first occurrence of name kept)
    let (themes, collisions) = dedupe_themes(result.themes);
    result.themes = themes;
    result.diagnostics.extend(collisions);

    result
}

/// Per-directory theme scanner shared by the resource loader and ThemeController (delegated in theme/mod.rs): load all
/// `*.json` themes under a directory into the output set — "move goods into warehouse: move one directory at a time; bad boxes get warnings and do not block the way".
///
/// Behavior (matches pi `loadThemesFromDir`): if the directory does not exist, return immediately; symlinks are followed and determined via `metadata`
/// (load only when the link target is a file; broken links/targets that are directories are skipped); non-`.json` skipped; read/parse failures recorded as
/// `ThemeLoadDiagnostic::Warning` (path=file or directory) and continue; successful entries go into `out` as `(theme, source)`.
///
/// # Parameters
/// - `dir`: directory to scan;
/// - `source`: source layer of themes in this directory (default directories User/Project, explicit directory Path);
/// - `out`/`diagnostics`: output carriers (owned by caller; both `load_themes` and theme/mod.rs reuse this primitive).
pub(crate) fn load_themes_from_dir(
    dir: &Path,
    source: ThemeSource,
    out: &mut Vec<LoadedTheme>,
    diagnostics: &mut Vec<ThemeLoadDiagnostic>,
) {
    if !dir.exists() {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            diagnostics.push(ThemeLoadDiagnostic::Warning {
                message: format!("failed to read theme directory: {e}"),
                path: dir.display().to_string(),
            });
            return;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        // Symlinks are followed and determined via metadata (matches pi loadThemesFromDir: entry.isSymbolicLink → statSync)
        let is_file = match std::fs::metadata(&path) {
            Ok(m) => m.is_file(),
            Err(_) => continue, // Skip broken/unreadable entries
        };
        if !is_file {
            continue;
        }
        if path.extension().is_some_and(|e| e == "json") {
            load_theme_from_file(&path, source, out, diagnostics);
        }
    }
}

/// Single `.json` theme file load — read file + parse with zen-tui `load_theme_from_json`,
/// failures recorded as Warning (path=file) and continue; successful entries go into `out` as `(theme, source)`.
fn load_theme_from_file(
    file_path: &Path,
    source: ThemeSource,
    out: &mut Vec<LoadedTheme>,
    diagnostics: &mut Vec<ThemeLoadDiagnostic>,
) {
    match std::fs::read_to_string(file_path) {
        Ok(json) => match theme::load_theme_from_json(&json, Some(file_path.to_path_buf())) {
            Ok(theme) => out.push(LoadedTheme { theme, source }),
            Err(e) => diagnostics.push(ThemeLoadDiagnostic::Warning {
                message: format!("failed to load theme: {e}"),
                path: file_path.display().to_string(),
            }),
        },
        Err(e) => diagnostics.push(ThemeLoadDiagnostic::Warning {
            message: format!("failed to read theme: {e}"),
            path: file_path.display().to_string(),
        }),
    }
}

/// Deduplicate themes with the same name (matches pi `dedupeThemes`) — keep first occurrence of name; later arrivals get a Collision diagnostic and are removed.
fn dedupe_themes(themes: Vec<LoadedTheme>) -> (Vec<LoadedTheme>, Vec<ThemeLoadDiagnostic>) {
    let mut seen: Vec<LoadedTheme> = Vec::new();
    let mut diagnostics = Vec::new();
    for t in themes {
        let name = t.theme.name.clone();
        if let Some(existing) = seen.iter().find(|e| e.theme.name == name) {
            diagnostics.push(ThemeLoadDiagnostic::Collision(ThemeCollision {
                name,
                winner_path: source_path_string(&existing.theme),
                loser_path: source_path_string(&t.theme),
            }));
        } else {
            seen.push(t);
        }
    }
    (seen, diagnostics)
}

/// Theme display path — if `source_path` is missing (e.g. future built-in themes), use `"<builtin>"` (matches pi `?? "<builtin>"`).
fn source_path_string(theme: &Theme) -> String {
    theme
        .source_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "<builtin>".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real built-in theme JSON (all 52 color fields, aligned field-by-field with ThemeJsonColors) —
    /// based on zen-tui builtin_dark.json with name replaced, avoiding drift from a handwritten field list.
    const BUILTIN_DARK_JSON: &str = include_str!("../../../zen-tui/src/theme/builtin_dark.json");

    /// Valid theme JSON (name customizable) — "built-in theme with renamed name" is the minimal valid fixture.
    fn minimal_theme_json(name: &str) -> String {
        BUILTIN_DARK_JSON.replace("\"name\": \"dark\"", &format!("\"name\": \"{name}\""))
    }

    // ─── load_themes_from_dir ────────────────────────────────────

    #[test]
    fn test_load_themes_from_dir_empty_and_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let mut out = Vec::new();
        let mut diags = Vec::new();
        // Nonexistent directory → return immediately
        load_themes_from_dir(
            &tmp.path().join("missing"),
            ThemeSource::User,
            &mut out,
            &mut diags,
        );
        assert!(out.is_empty());
        assert!(diags.is_empty());
        // Exists but is an empty directory
        let empty = tmp.path().join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        load_themes_from_dir(&empty, ThemeSource::User, &mut out, &mut diags);
        assert!(out.is_empty());
        assert!(diags.is_empty());
    }

    #[test]
    fn test_load_themes_from_dir_filters_non_json_and_invalid() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("themes");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("good.json"), minimal_theme_json("good")).unwrap();
        std::fs::write(dir.join("bad.json"), "not json").unwrap();
        std::fs::write(dir.join("notes.txt"), "hello").unwrap();
        std::fs::create_dir_all(dir.join("subdir.json")).unwrap(); // Directory with same-named .json → skip

        let mut out = Vec::new();
        let mut diags = Vec::new();
        load_themes_from_dir(&dir, ThemeSource::User, &mut out, &mut diags);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].theme.name, "good");
        assert_eq!(out[0].source, ThemeSource::User);
        assert_eq!(diags.len(), 1); // bad.json parse failure
        assert!(
            matches!(&diags[0], ThemeLoadDiagnostic::Warning { message, .. }
            if message.contains("failed to load theme"))
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_load_themes_from_dir_follows_symlink_json() {
        use std::os::unix::fs::symlink;
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("themes");
        std::fs::create_dir_all(&dir).unwrap();
        let real = tmp.path().join("real.json");
        std::fs::write(&real, minimal_theme_json("linked")).unwrap();
        symlink(&real, dir.join("linked.json")).unwrap(); // Link named .json → follow and load

        let mut out = Vec::new();
        let mut diags = Vec::new();
        load_themes_from_dir(&dir, ThemeSource::User, &mut out, &mut diags);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].theme.name, "linked");
        assert!(diags.is_empty());
    }

    // ─── load_themes integration (tempdir + real files) ──────────────────

    #[test]
    fn test_load_themes_defaults_user_and_project() {
        let tmp = tempfile::tempdir().unwrap();
        let agent_dir = tmp.path().join("agent");
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(agent_dir.join("themes")).unwrap();
        std::fs::create_dir_all(cwd.join(CONFIG_DIR_NAME).join("themes")).unwrap();
        std::fs::write(
            agent_dir.join("themes").join("user.json"),
            minimal_theme_json("user-theme"),
        )
        .unwrap();
        std::fs::write(
            cwd.join(CONFIG_DIR_NAME).join("themes").join("proj.json"),
            minimal_theme_json("proj-theme"),
        )
        .unwrap();

        let result = load_themes(&LoadThemesOptions {
            cwd: cwd.clone(),
            agent_dir: agent_dir.clone(),
            theme_paths: Vec::new(),
            include_defaults: true,
        });
        assert_eq!(result.themes.len(), 2);
        assert!(result.diagnostics.is_empty());
        let user = result
            .themes
            .iter()
            .find(|t| t.theme.name == "user-theme")
            .unwrap();
        assert_eq!(user.source, ThemeSource::User);
        let proj = result
            .themes
            .iter()
            .find(|t| t.theme.name == "proj-theme")
            .unwrap();
        assert_eq!(proj.source, ThemeSource::Project);
    }

    #[test]
    fn test_load_themes_explicit_dir_and_file() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(&cwd).unwrap();
        let extra_dir = tmp.path().join("extra");
        std::fs::create_dir_all(&extra_dir).unwrap();
        std::fs::write(
            extra_dir.join("extra.json"),
            minimal_theme_json("extra-theme"),
        )
        .unwrap();
        let single = tmp.path().join("single.json");
        std::fs::write(&single, minimal_theme_json("single-theme")).unwrap();

        let result = load_themes(&LoadThemesOptions {
            cwd: cwd.clone(),
            agent_dir: tmp.path().to_path_buf(),
            theme_paths: vec![extra_dir, single],
            include_defaults: false,
        });
        assert_eq!(result.themes.len(), 2);
        assert!(result.themes.iter().all(|t| t.source == ThemeSource::Path));
        assert!(result.themes.iter().any(|t| t.theme.name == "extra-theme"));
        assert!(result.themes.iter().any(|t| t.theme.name == "single-theme"));
        assert!(result.diagnostics.is_empty());
    }

    #[test]
    fn test_load_themes_explicit_path_resolved_against_cwd() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(&cwd).unwrap();
        // Relative paths resolved via cwd (matches skills.rs explicit-path branch resolve_path(trim:true))
        std::fs::write(cwd.join("rel.json"), minimal_theme_json("rel-theme")).unwrap();

        let result = load_themes(&LoadThemesOptions {
            cwd: cwd.clone(),
            agent_dir: tmp.path().to_path_buf(),
            theme_paths: vec![PathBuf::from("rel.json")],
            include_defaults: false,
        });
        assert_eq!(result.themes.len(), 1);
        assert_eq!(result.themes[0].theme.name, "rel-theme");
        assert_eq!(result.themes[0].source, ThemeSource::Path);
    }

    #[test]
    fn test_load_themes_nonexistent_and_non_json_warnings() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(&cwd).unwrap();
        let non_json = tmp.path().join("theme.txt");
        std::fs::write(&non_json, "hello").unwrap();

        let result = load_themes(&LoadThemesOptions {
            cwd: cwd.clone(),
            agent_dir: tmp.path().to_path_buf(),
            theme_paths: vec![tmp.path().join("missing"), non_json],
            include_defaults: false,
        });
        assert!(result.themes.is_empty());
        assert_eq!(result.diagnostics.len(), 2);
        assert!(result.diagnostics.iter().any(|d| matches!(d,
            ThemeLoadDiagnostic::Warning { message, .. } if message.contains("does not exist"))));
        assert!(result.diagnostics.iter().any(|d| matches!(d,
            ThemeLoadDiagnostic::Warning { message, .. } if message.contains("not a json file"))));
    }

    #[test]
    fn test_load_themes_collision_default_wins_over_explicit() {
        let tmp = tempfile::tempdir().unwrap();
        let agent_dir = tmp.path().join("agent");
        let cwd = tmp.path().join("cwd");
        std::fs::create_dir_all(agent_dir.join("themes")).unwrap();
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::write(
            agent_dir.join("themes").join("clash.json"),
            minimal_theme_json("clash"),
        )
        .unwrap();
        let extra_dir = tmp.path().join("extra");
        std::fs::create_dir_all(&extra_dir).unwrap();
        std::fs::write(extra_dir.join("clash.json"), minimal_theme_json("clash")).unwrap();

        let result = load_themes(&LoadThemesOptions {
            cwd: cwd.clone(),
            agent_dir: agent_dir.clone(),
            theme_paths: vec![extra_dir],
            include_defaults: true,
        });
        assert_eq!(result.themes.len(), 1);
        assert_eq!(result.themes[0].source, ThemeSource::User); // first occurrence (User) wins
        assert_eq!(result.diagnostics.len(), 1);
        match &result.diagnostics[0] {
            ThemeLoadDiagnostic::Collision(c) => {
                assert_eq!(c.name, "clash");
                assert!(c.winner_path.ends_with("agent/themes/clash.json"));
                assert!(c.loser_path.ends_with("extra/clash.json"));
            }
            _ => panic!("expected collision"),
        }
    }
}
