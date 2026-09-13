// REQ-APP-11 (pi-357): theme controller — lifecycle management, terminal background detection, auto theme, file-watch hot reload.
// Reference: reference/pi-original/packages/coding-agent/src/modes/interactive/theme/theme-controller.ts
//
// Design notes:
// - ThemeController holds the current theme, terminal background type, and change callback
// - Supports auto theme ("lightThemeName/darkThemeName" format)
// - File watching uses the notify crate (synchronous API, wrapped by tokio spawn_blocking)
// - Built-in themes + custom themes (~/.zen/agent/themes/*.json)

use std::path::{Path, PathBuf};

use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use thiserror::Error;
use zen_tui::theme::{self, Theme, ThemeError};

/// Theme controller error.
#[derive(Debug, Error)]
pub enum ControllerError {
    /// Theme loading/parsing error.
    #[error(transparent)]
    Theme(#[from] ThemeError),
    /// File watching error.
    #[error("file watcher error: {0}")]
    Watcher(#[from] notify::Error),
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Terminal background type, used for auto theme selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalTheme {
    /// Dark terminal.
    Dark,
    /// Light terminal.
    Light,
}

/// Theme lifecycle manager.
/// Holds the current theme, settings reference, and change callback. Manages theme switching, auto sync, and file watching.
/// Corresponds to TS `InteractiveThemeController`.
pub struct ThemeController {
    /// Currently active theme.
    current_theme: Theme,
    /// Detected terminal background type.
    terminal_theme: TerminalTheme,
    /// Whether terminal color scheme synchronization is enabled.
    auto_sync_enabled: bool,
    /// File watcher (notify crate).
    watcher: Option<RecommendedWatcher>,
    /// Theme change callback.
    on_change: Option<Box<dyn Fn() + Send>>,
    /// Custom theme directory path.
    themes_dir: Option<PathBuf>,
}

impl ThemeController {
    /// Create the theme controller —
    /// Called at startup by the interactive-mode main entry point (T-AP-11-1).
    /// Read theme preference from settings, detect terminal background, and load the initial theme.
    /// # Parameters
    /// - `settings_theme`: theme setting from settings (`--theme` argument or settings.json)
    /// - `themes_dir`: custom theme directory (`~/.zen/agent/themes/`)
    /// - `on_change`: callback on theme change (triggers UI redraw)
    pub fn new(
        settings_theme: Option<&str>,
        themes_dir: Option<PathBuf>,
        on_change: Option<Box<dyn Fn() + Send>>,
    ) -> Result<Self, ControllerError> {
        let terminal_theme = detect_terminal_theme();
        let mut controller = ThemeController {
            current_theme: theme::builtin_dark_theme().clone(),
            terminal_theme,
            auto_sync_enabled: false,
            watcher: None,
            on_change,
            themes_dir,
        };
        controller.apply_from_settings(settings_theme)?;
        Ok(controller)
    }

    /// Load theme based on theme setting in settings —
    /// Supports auto theme ("light/dark" format) to automatically select based on terminal background.
    /// Corresponds to TS `applyFromSettings`.
    pub fn apply_from_settings(
        &mut self,
        settings_theme: Option<&str>,
    ) -> Result<(), ControllerError> {
        if let Some(setting) = settings_theme {
            // Check whether it is in auto theme format ("lightThemeName/darkThemeName")
            if let Some(auto) = parse_auto_theme_setting(setting) {
                self.auto_sync_enabled = true;
                let theme_name = match self.terminal_theme {
                    TerminalTheme::Light => &auto.light_theme,
                    TerminalTheme::Dark => &auto.dark_theme,
                };
                self.apply_theme_name(theme_name, true)?;
                return Ok(());
            }

            // Regular theme name
            self.auto_sync_enabled = false;
            self.apply_theme_name(setting, true)?;
            return Ok(());
        }

        // No setting; use terminal background detection
        self.auto_sync_enabled = false;
        let theme_name = match self.terminal_theme {
            TerminalTheme::Light => "light",
            TerminalTheme::Dark => "dark",
        };
        self.apply_theme_name(theme_name, false)?;
        Ok(())
    }

    /// Called when the user switches themes via the /theme command —
    /// Find theme by name (built-in → custom file), switch to it, and notify callback.
    /// Corresponds to TS `setThemeName`.
    pub fn set_theme(&mut self, name: &str) -> Result<(), ControllerError> {
        self.auto_sync_enabled = false;
        self.apply_theme_name(name, true)?;
        Ok(())
    }

    /// Called when the settings selector previews a theme —
    /// Temporarily switch theme for preview without persisting to settings.
    pub fn preview(&mut self, name: &str) -> Result<(), ControllerError> {
        self.apply_theme_name(name, false)?;
        Ok(())
    }

    /// Get a read-only reference to the currently active theme —
    /// Components use this to get theme colors during rendering.
    pub fn current(&self) -> &Theme {
        &self.current_theme
    }

    /// Get a mutable reference to the currently active theme —
    /// For scenarios that need to modify the theme directly.
    pub fn current_mut(&mut self) -> &mut Theme {
        &mut self.current_theme
    }

    /// Get the terminal background type.
    pub fn terminal_theme(&self) -> TerminalTheme {
        self.terminal_theme
    }

    /// Whether auto sync is enabled.
    pub fn is_auto_sync_enabled(&self) -> bool {
        self.auto_sync_enabled
    }

    /// Start file watching —
    /// Enable notify watching on the custom theme directory; automatically reload and trigger callback when theme files change.
    /// Built-in themes are not watched.
    pub fn start_watcher(&mut self) -> Result<(), ControllerError> {
        self.stop_watcher();

        let themes_dir = match &self.themes_dir {
            Some(dir) => dir.clone(),
            None => return Ok(()), // No custom theme directory; do not watch
        };

        // Watch only custom themes (not built-in)
        let current_name = self.current_theme.name.clone();
        if current_name == "dark" || current_name == "light" {
            return Ok(()); // Do not watch built-in themes
        }

        // Check whether the directory exists
        if !themes_dir.exists() {
            return Ok(());
        }

        let on_change = self.on_change.take();
        let theme_name = current_name.clone();
        let themes_dir_clone = themes_dir.clone();

        // Wrap callback in Arc<Mutex> to share it in the closure
        let on_change_shared = std::sync::Arc::new(std::sync::Mutex::new(on_change));
        let on_change_for_watcher = on_change_shared.clone();

        let mut watcher =
            notify::recommended_watcher(move |res: Result<notify::Event, notify::Error>| {
                if let Ok(event) = res {
                    // Only care about modify events
                    if event.kind.is_modify() {
                        // Check whether it is the current theme file
                        for path in &event.paths {
                            let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned());
                            if stem.as_deref() == Some(theme_name.as_str()) {
                                // Try to reload the theme
                                if let Ok(json) = std::fs::read_to_string(path) {
                                    if theme::load_theme_from_json(&json, Some(path.clone()))
                                        .is_ok()
                                    {
                                        // Theme is valid; trigger callback
                                        if let Ok(guard) = on_change_for_watcher.lock() {
                                            if let Some(ref cb) = *guard {
                                                cb();
                                            }
                                        }
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
            })?;

        watcher.watch(&themes_dir_clone, RecursiveMode::NonRecursive)?;
        self.watcher = Some(watcher);
        // Store the shared callback back (if possible)
        if let Ok(inner) = std::sync::Arc::try_unwrap(on_change_shared) {
            if let Ok(cb) = inner.into_inner() {
                self.on_change = cb;
            }
        }

        Ok(())
    }

    /// Stop file watching —
    /// Clean up notify watcher resources.
    pub fn stop_watcher(&mut self) {
        if let Some(watcher) = self.watcher.take() {
            drop(watcher);
        }
    }

    /// Load all themes in the custom theme directory —
    /// Scan `~/.zen/agent/themes/*.json`, load each one, and collect them.
    /// Invalid topics are skipped (no error); returns the list of valid topics.
    pub fn load_custom_themes(&self) -> Vec<Theme> {
        let themes_dir = match &self.themes_dir {
            Some(dir) => dir,
            None => return Vec::new(),
        };
        load_custom_themes_from_dir(themes_dir)
    }

    /// Get all available theme names (built-in + custom).
    pub fn available_theme_names(&self) -> Vec<String> {
        let mut names: Vec<String> = theme::builtin_theme_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        for custom in self.load_custom_themes() {
            if !names.contains(&custom.name) {
                names.push(custom.name);
            }
        }
        names.sort();
        names
    }

    /// Internal method: apply theme by name.
    fn apply_theme_name(&mut self, name: &str, notify: bool) -> Result<(), ControllerError> {
        // First check built-in themes
        if let Some(builtin) = theme::get_builtin_theme(name) {
            self.current_theme = builtin.clone();
            if notify {
                self.notify_changed();
            }
            return Ok(());
        }

        // Then check custom themes
        if let Some(themes_dir) = &self.themes_dir {
            let theme_path = themes_dir.join(format!("{name}.json"));
            if theme_path.exists() {
                let json = std::fs::read_to_string(&theme_path)?;
                let loaded = theme::load_theme_from_json(&json, Some(theme_path))?;
                self.current_theme = loaded;
                if notify {
                    self.notify_changed();
                }
                return Ok(());
            }
        }

        // Theme not found; fall back to dark
        self.current_theme = theme::builtin_dark_theme().clone();
        if notify {
            self.notify_changed();
        }
        Ok(())
    }

    /// Notify theme change.
    fn notify_changed(&self) {
        if let Some(ref cb) = self.on_change {
            cb();
        }
    }
}

impl Drop for ThemeController {
    fn drop(&mut self) {
        self.stop_watcher();
    }
}

// ── Auto theme parsing ───────────────────────────────────────────────

/// Auto theme setting parsing result.
struct AutoThemeSetting {
    light_theme: String,
    dark_theme: String,
}

/// Parse auto theme setting —
/// Format: "lightThemeName/darkThemeName" (e.g., "light/dark").
/// Returns None if the format is invalid.
fn parse_auto_theme_setting(setting: &str) -> Option<AutoThemeSetting> {
    let slash_index = setting.find('/')?;
    // Allow only one '/'
    if setting[slash_index + 1..].contains('/') {
        return None;
    }
    let light_theme = setting[..slash_index].trim().to_string();
    let dark_theme = setting[slash_index + 1..].trim().to_string();
    if light_theme.is_empty() || dark_theme.is_empty() {
        return None;
    }
    Some(AutoThemeSetting {
        light_theme,
        dark_theme,
    })
}

// ── Terminal background detection ─────────────────────────────────

/// Detect terminal background type —
/// Read the COLORFGBG environment variable to infer light/dark.
/// Return the detection result.
pub fn detect_terminal_theme() -> TerminalTheme {
    detect_terminal_theme_from_env(std::env::var("COLORFGBG").ok().as_deref())
}

/// Detect the terminal background type from the COLORFGBG environment variable value.
/// COLORFGBG format: "fg;bg" or "fg;bg;bg"; bg values 0-7 are dark, 8-15 are light.
fn detect_terminal_theme_from_env(colorfgbg: Option<&str>) -> TerminalTheme {
    if let Some(val) = colorfgbg {
        // Take the number after the last semicolon
        if let Some(bg_str) = val.rsplit(';').next() {
            if let Ok(bg) = bg_str.trim().parse::<u8>() {
                // 0-7 are dark backgrounds, 8-15 are light backgrounds
                // But for most terminals bg=0 (black) is dark
                return if bg < 8 {
                    TerminalTheme::Dark
                } else {
                    TerminalTheme::Light
                };
            }
        }
    }
    // Default dark
    TerminalTheme::Dark
}

// ── Custom theme loading ────────────────────────────────────────────────

/// Load all custom themes from a directory—
/// Scan `*.json`, load each one, and collect them.
/// Invalid topics are skipped (no error); returns the list of valid topics.
/// Implementation delegates to `resources::themes::load_themes_from_dir` (T-AP-05-3 unified primitive) and discards diagnostics
/// (ThemeController consumers only look at available theme names); behavioral equivalence (follow symlinks/skip broken links) is accepted based on
/// "visible theme set unchanged" as acceptance; see tasks/T-AP-05-3-implemented.md "Integration with existing code" review ②.
pub fn load_custom_themes_from_dir(themes_dir: &Path) -> Vec<Theme> {
    let mut out = Vec::new();
    let mut diagnostics = Vec::new();
    crate::resources::themes::load_themes_from_dir(
        themes_dir,
        crate::resources::themes::ThemeSource::Path,
        &mut out,
        &mut diagnostics,
    );
    out.into_iter().map(|loaded| loaded.theme).collect()
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_terminal_theme_dark() {
        assert_eq!(
            detect_terminal_theme_from_env(Some("15;0")),
            TerminalTheme::Dark
        );
        assert_eq!(
            detect_terminal_theme_from_env(Some("0;0")),
            TerminalTheme::Dark
        );
        assert_eq!(
            detect_terminal_theme_from_env(Some("15;7")),
            TerminalTheme::Dark
        );
    }

    #[test]
    fn test_detect_terminal_theme_light() {
        assert_eq!(
            detect_terminal_theme_from_env(Some("0;15")),
            TerminalTheme::Light
        );
        assert_eq!(
            detect_terminal_theme_from_env(Some("0;252")),
            TerminalTheme::Light
        );
    }

    #[test]
    fn test_detect_terminal_theme_default() {
        assert_eq!(detect_terminal_theme_from_env(None), TerminalTheme::Dark);
        assert_eq!(
            detect_terminal_theme_from_env(Some("")),
            TerminalTheme::Dark
        );
        assert_eq!(
            detect_terminal_theme_from_env(Some("invalid")),
            TerminalTheme::Dark
        );
    }

    #[test]
    fn test_parse_auto_theme_setting_valid() {
        let result = parse_auto_theme_setting("light/dark");
        assert!(result.is_some());
        let setting = result.unwrap();
        assert_eq!(setting.light_theme, "light");
        assert_eq!(setting.dark_theme, "dark");
    }

    #[test]
    fn test_parse_auto_theme_setting_with_spaces() {
        let result = parse_auto_theme_setting(" my-light / my-dark ");
        assert!(result.is_some());
        let setting = result.unwrap();
        assert_eq!(setting.light_theme, "my-light");
        assert_eq!(setting.dark_theme, "my-dark");
    }

    #[test]
    fn test_parse_auto_theme_setting_no_slash() {
        assert!(parse_auto_theme_setting("dark").is_none());
    }

    #[test]
    fn test_parse_auto_theme_setting_multiple_slashes() {
        assert!(parse_auto_theme_setting("a/b/c").is_none());
    }

    #[test]
    fn test_parse_auto_theme_setting_empty_parts() {
        assert!(parse_auto_theme_setting("/dark").is_none());
        assert!(parse_auto_theme_setting("light/").is_none());
    }

    #[test]
    fn test_theme_controller_new_with_builtin() {
        let controller = ThemeController::new(Some("dark"), None, None).unwrap();
        assert_eq!(controller.current().name, "dark");
    }

    #[test]
    fn test_theme_controller_new_with_auto_theme() {
        // Simulate a dark terminal
        std::env::set_var("COLORFGBG", "15;0");
        let controller = ThemeController::new(Some("light/dark"), None, None).unwrap();
        assert_eq!(controller.current().name, "dark");
        std::env::remove_var("COLORFGBG");
    }

    #[test]
    fn test_theme_controller_set_theme() {
        let mut controller = ThemeController::new(Some("dark"), None, None).unwrap();
        controller.set_theme("light").unwrap();
        assert_eq!(controller.current().name, "light");
    }

    #[test]
    fn test_theme_controller_available_themes() {
        let controller = ThemeController::new(Some("dark"), None, None).unwrap();
        let names = controller.available_theme_names();
        assert!(names.contains(&"dark".to_string()));
        assert!(names.contains(&"light".to_string()));
    }

    #[test]
    fn test_load_custom_themes_from_nonexistent_dir() {
        let themes = load_custom_themes_from_dir(Path::new("/nonexistent/path"));
        assert!(themes.is_empty());
    }
}
