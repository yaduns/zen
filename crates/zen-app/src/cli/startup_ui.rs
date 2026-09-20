// REQ-APP-08 (pi-335 startup-ui.ts) — Startup UI: first-time setup + generic option selector + startup TUI shell.
// Corresponds to the original TS: reference/pi-original/packages/coding-agent/src/cli/startup-ui.ts
// `createStartupTui`/`shouldRunFirstTimeSetup`/`showFirstTimeSetup`/`showStartupSelector<T>`
// in zen form. See tasks/T-AP-08-6-implemented.md for implementation notes.
//
// v1 scope trimming and resolutions:
// - `isOfficialDistribution` (always true for the single zen distribution), `areExperimentalFeaturesEnabled`
//   (no experimental switch), `setEnableAnalytics` (pi-325 telemetry dropped), `applyDetectedStartupTheme`
//   (the startup UI is brief and theme assembly already selects by detection), `clearStartupTui` (the 25ms delay has no behavioral value,
//   TuiLoop::stop already restores the terminal) → dropped/resolved in v1, see the "Ecosystem Reuse Verification Conclusions" in the implementation notes;
// - `showStartupInput` (text input prompt) has no consumer in v1 (project trust v2); the capability slot is registered as
//   REQ-V2-01/REQ-V2-02 reserved, no pub fn implemented;
// - pi's `showStartupSelector(settingsManager, ...)` assembles via the settings theme preference → v1 goes through
//   `startup_theme(None)` (terminal detection fallback), registered as an accepted deviation (see "Open Question 2" in the implementation notes).

use crate::components::first_time_setup::{FirstTimeSetupComponent, FirstTimeSetupResult};
use crate::config::settings::SettingsManager;
use crate::config::ENV_AGENT_DIR;
use crate::theme::{detect_terminal_theme, ControllerError, TerminalTheme};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use thiserror::Error;
use tokio_util::sync::CancellationToken;
use zen_tui::app_loop::{EventOutcome, TuiError, TuiLoop};
use zen_tui::components::Component;
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{ColorToken, Theme};

/// Error aggregation for the startup UI library surface (both the TuiLoop and theme surfaces).
/// `run_startup_tui` returns `TuiError` directly and `startup_theme` returns `ControllerError` directly;
/// this type unifies the errors of the pub entry points (`show_first_time_setup`/`show_startup_selector`).
/// Follows the `SessionPickerError` precedent; its Session variant (session IO callbacks) has no corresponding scenario in this module,
/// so that type is not reused directly.
#[derive(Debug, Error)]
pub enum StartupUiError {
    /// TUI loop start/run/stop errors.
    #[error(transparent)]
    Tui(#[from] TuiError),
    /// Theme controller construction error.
    #[error(transparent)]
    Theme(#[from] ControllerError),
}

/// First-time setup gate: main calls it before assembling interactive mode, deciding whether to show the first-time setup dialog —
/// "the user is running zen for the first time (settings.json not yet created) and has not customized the agent directory via an environment variable,
/// so we should first ask whether they want a dark or light theme, instead of going straight into a session".
/// Rules (aligned with pi's `shouldRunFirstTimeSetup`; the official-distribution/experimental gates
/// are always true in zen and dropped): `ENV_AGENT_DIR` set → false; `settings_path` already exists → false;
/// otherwise → true.
/// # Parameters
/// - `settings_path`: global settings file path (`config::get_settings_path()` resolution result);
///   only checked for existence, contents not read or written.
/// # Consumers
/// main.rs (node 2.5 first-time setup wiring).
pub fn should_run_first_time_setup(settings_path: &Path) -> bool {
    // An env override is treated as an "advanced user custom directory", so no first-time setup is shown (aligned with pi's `process.env[ENV_AGENT_DIR]`);
    // an empty string is treated as unset (aligned with `config::get_agent_dir`'s `filter(!is_empty)` semantics).
    let env_agent_dir = std::env::var_os(ENV_AGENT_DIR)
        .map(|v| v.to_string_lossy().into_owned())
        .filter(|v| !v.is_empty());
    first_time_setup_gate(env_agent_dir.as_deref(), settings_path.exists())
}

/// Private pure core: the testable base of `should_run_first_time_setup` (env reading extracted; process-level env
/// is inconvenient for parallel unit tests) —
/// "combines the two veto conditions 'agent directory taken over by an environment variable' and 'settings file already exists' into a boolean gate".
fn first_time_setup_gate(env_agent_dir: Option<&str>, settings_exists: bool) -> bool {
    if env_agent_dir.is_some() {
        return false;
    }
    !settings_exists
}

/// Startup TUI theme assembly: selects the initial theme by the settings theme preference (terminal background detection if none) —
/// the single source of truth for "what color scheme the startup dialog uses", shared by first-time setup/startup selector/session picker.
/// Wraps `ThemeController::new(settings_theme, None, None)` and clones `current()`;
/// on theme miss falls back to the built-in dark (ThemeController semantics).
/// # Parameters
/// - `settings_theme`: effective theme preference from settings (`SettingsEffective.theme`, `/` path forms
///   already filtered); None = detect from the terminal background.
/// # Consumers
/// `run_picker_inner` (T-AP-08-5 refactor, passes None), `show_first_time_setup` (None),
/// `show_startup_selector` (None, see the deviation registration in "Ecosystem Reuse Verification Conclusions").
pub(crate) fn startup_theme(settings_theme: Option<&str>) -> Result<Theme, ControllerError> {
    let controller = crate::theme::ThemeController::new(settings_theme, None, None)?;
    Ok(controller.current().clone())
}

/// Startup TUI main loop runner (the landing point of the T-AP-08-5 seam): runs a full-screen component's
/// on_event/draw closures through the TuiLoop lifecycle and restores the terminal —
/// "startup-phase dialogs (session picker/first-time setup/Continue-Cancel) share one base of 'spin up a TUI, run the component,
/// stop and restore the terminal', instead of writing it once per dialog".
/// Extracted from the inline shell in `run_picker_inner`; three-phase error propagation start/run/stop (if run errors,
/// stop and restore the terminal first, then propagate, aligned with the existing implementation).
/// # Parameters
/// - `cancel`: cancellation handle created by the caller; the component's cancel callback cancels it to make run return.
/// - `on_event`/`draw`: closures with the same signature passed through to `TuiLoop::run` (components shared via `Rc<RefCell<..>>`,
///   see the existing `run_picker_inner`).
/// # Constraints
/// Must be called inside a `tokio::task::LocalSet` context (TuiLoop/components are !Send).
/// # Consumers
/// `show_first_time_setup`/`show_startup_selector` (this module), `run_picker_inner`
/// (session_picker.rs refactor).
pub(crate) async fn run_startup_tui(
    cancel: &CancellationToken,
    on_event: impl FnMut(crossterm::event::Event) -> EventOutcome,
    draw: impl FnMut(&mut ScreenBuffer),
) -> Result<(), TuiError> {
    let mut tui = TuiLoop::new()?;
    tui.start()?;
    let run_result = tui.run(cancel, on_event, draw).await;
    // Both run and stop are executed (restore the terminal first even if run errors, then propagate the error)
    let stop_result = tui.stop();
    run_result?;
    stop_result?;
    Ok(())
}

/// First-time setup dialog: main calls it after `should_run_first_time_setup` hits; runs
/// `FirstTimeSetupComponent` and persists the theme choice —
/// "on the first launch of zen, let the user pick a dark/light theme; pressing Enter writes the choice into settings.json
/// (won't ask again next launch), pressing Esc skips without writing".
/// Inside LocalSet: `startup_theme(None)` (Theme for rendering; internally already picks the dark/light built-in
/// theme via terminal detection) → the component's `detected_theme: TerminalTheme` comes from a **separate**
/// `detect_terminal_theme()` call (same source and value as the detection inside `startup_theme` — two reads
/// of `COLORFGBG` are deterministically identical; `startup_theme` only returns `Theme`, not `TerminalTheme`,
/// therefore component arguments are taken separately) → construct component (on_submit writes
/// `settings.update_global(|d| d.theme = Some(name))`，name = "dark"/"light"；
/// on_cancel does not write) → `run_startup_tui`.
/// # Parameters
/// - `settings`: start settings manager (main uses `SettingsManager::load(cwd, agent_dir)`
///   passed in; writes go through `update_global` auto diff + atomic disk write; write failures are not thrown back, errors are enqueued).
/// # Consumers
/// main.rs（node 2.5）。
pub async fn show_first_time_setup(settings: &SettingsManager) -> Result<(), StartupUiError> {
    // local-scope !Send (Rc<RefCell> components shared across closures), following the LocalSet precedent of select_session.
    let local = tokio::task::LocalSet::new();
    local.run_until(run_setup_inner(settings)).await
}

/// actual body of `show_first_time_setup` (executes within LocalSet context; holds !Send component state).
async fn run_setup_inner(settings: &SettingsManager) -> Result<(), StartupUiError> {
    let detected_theme = detect_terminal_theme();
    let theme = startup_theme(None)?;

    // shared result cell + cancel handle (submit/cancel callbacks write and cancel → run loop exits)
    let result_cell: Rc<RefCell<Option<Option<TerminalTheme>>>> = Rc::new(RefCell::new(None));
    let cancel = CancellationToken::new();

    let comp_cell = Rc::new(RefCell::new(FirstTimeSetupComponent::new(
        detected_theme,
        &theme,
        // preview callback: v1 dialog does not do full-board real-time recoloring (component holds fixed &Theme); wiring retained for
        // v2 use if component API is upgraded (see implementation notes "Open Question 1").
        Some(Box::new(|_t: TerminalTheme| {})),
        Some(Box::new({
            let result_cell = Rc::clone(&result_cell);
            let cancel = cancel.clone();
            move |r: FirstTimeSetupResult| {
                *result_cell.borrow_mut() = Some(Some(r.theme));
                cancel.cancel();
            }
        })),
        Some(Box::new({
            let result_cell = Rc::clone(&result_cell);
            let cancel = cancel.clone();
            move || {
                *result_cell.borrow_mut() = Some(None);
                cancel.cancel();
            }
        })),
    )));

    run_startup_tui(
        &cancel,
        {
            let comp_cell = Rc::clone(&comp_cell);
            move |event| {
                match event {
                    crossterm::event::Event::Key(key) => {
                        // Defensive filtering of Release events (always Press when enhanced protocol REPORT_EVENT_TYPES is not enabled)
                        if key.kind == crossterm::event::KeyEventKind::Release {
                            return EventOutcome::Continue;
                        }
                        let mut comp = comp_cell.borrow_mut();
                        comp.handle_key(&key);
                        EventOutcome::Continue
                    }
                    _ => EventOutcome::Continue,
                }
            }
        },
        {
            let comp_cell = Rc::clone(&comp_cell);
            move |buf: &mut ScreenBuffer| {
                let (w, h) = buf.size();
                let mut comp = comp_cell.borrow_mut();
                let _ = comp.render(buf, 0, 0, w, h);
            }
        },
    )
    .await?;

    // persistence (submit writes theme; Esc does not write)
    let result = result_cell.borrow_mut().take();
    if let Some(Some(theme)) = result {
        persist_first_time_theme(settings, theme);
    }
    Ok(())
}

/// first-time setup theme persistence: write selected TerminalTheme into global settings (`theme` field,
/// name = "dark"/"light") — auto diff + atomic disk write; write failures are enqueued, not thrown back (SettingsManager
/// semantics). Split into a private function to facilitate unit testing the persistence segment (without going through TuiLoop).
fn persist_first_time_theme(settings: &SettingsManager, theme: TerminalTheme) {
    let name = theme_setting_name(theme);
    settings.update_global(|d| d.theme = Some(name.to_string()));
}

/// TerminalTheme → settings.json `theme` field value ("dark"/"light").
fn theme_setting_name(theme: TerminalTheme) -> &'static str {
    match theme {
        TerminalTheme::Dark => "dark",
        TerminalTheme::Light => "light",
    }
}

/// start generic option selector: run a full-screen option list letting the user pick one or cancel —
/// "At startup the user must choose one of several options (e.g. choose Continue or Cancel when session cwd is invalid),
/// selecting returns the corresponding value, Esc returns None."
/// within LocalSet: `startup_theme(None)` + `StartupSelectorComponent<T>` (↑/↓ move,
/// Enter returns selected value, Esc returns None) → `run_startup_tui`.
/// # Parameters
/// - `title`: top explanatory text (rendered line by line within the component when multiline).
/// - `options`: `(label, value)` list; value type `T` is determined by the caller.
/// # Returns
/// `Ok(Some(value))` = selected; `Ok(None)` = Esc cancel. v1's only consumer (main.rs
/// Continue/Cancel) calls with `T = Option<String>` (Some=fallback cwd / None=Cancel),
/// `Some(Some(cwd))` → reopen, `Some(None)`/`None` → exit.
/// # Consumers
/// main.rs (missing-cwd interactive branch); reserved for REQ-V2-02 (model selection) / REQ-V2-01
/// (project trust).
pub async fn show_startup_selector<T: 'static>(
    title: &str,
    options: Vec<(&'static str, T)>,
) -> Result<Option<T>, StartupUiError> {
    let local = tokio::task::LocalSet::new();
    local.run_until(run_selector_inner(title, options)).await
}

/// actual body of `show_startup_selector` (executes within LocalSet context; holds !Send component state).
async fn run_selector_inner<T: 'static>(
    title: &str,
    options: Vec<(&'static str, T)>,
) -> Result<Option<T>, StartupUiError> {
    let theme = startup_theme(None)?;
    let labels: Vec<&'static str> = options.iter().map(|(l, _)| *l).collect();

    // option value is taken by the on_select callback by index (swap_remove moves out T); result is written to shared cell.
    let options_cell: Rc<RefCell<Vec<(&'static str, T)>>> = Rc::new(RefCell::new(options));
    let result_cell: Rc<RefCell<Option<Option<T>>>> = Rc::new(RefCell::new(None));
    let cancel = CancellationToken::new();

    let comp_cell = Rc::new(RefCell::new(StartupSelectorComponent::new(
        title.to_string(),
        labels,
        &theme,
        Box::new({
            let options_cell = Rc::clone(&options_cell);
            let result_cell = Rc::clone(&result_cell);
            let cancel = cancel.clone();
            move |idx: usize| {
                let value = options_cell.borrow_mut().swap_remove(idx).1;
                *result_cell.borrow_mut() = Some(Some(value));
                cancel.cancel();
            }
        }),
        Box::new({
            let result_cell = Rc::clone(&result_cell);
            let cancel = cancel.clone();
            move || {
                *result_cell.borrow_mut() = Some(None);
                cancel.cancel();
            }
        }),
    )));

    run_startup_tui(
        &cancel,
        {
            let comp_cell = Rc::clone(&comp_cell);
            move |event| match event {
                crossterm::event::Event::Key(key) => {
                    if key.kind == crossterm::event::KeyEventKind::Release {
                        return EventOutcome::Continue;
                    }
                    let mut comp = comp_cell.borrow_mut();
                    comp.handle_key(&key);
                    EventOutcome::Continue
                }
                _ => EventOutcome::Continue,
            }
        },
        {
            let comp_cell = Rc::clone(&comp_cell);
            move |buf: &mut ScreenBuffer| {
                let (w, h) = buf.size();
                let mut comp = comp_cell.borrow_mut();
                let _ = comp.render(buf, 0, 0, w, h);
            }
        },
    )
    .await?;

    let result = result_cell.borrow_mut().take();
    Ok(result.unwrap_or(None))
}

/// startup option selector component (private implementation detail): title + option list (↑/↓/j/k move,
/// Enter submits index, Esc cancels). Keybindings hardcoded (following `FirstTimeSetupComponent` precedent;
/// startup state has no user keybinding surface; SelectList + select_kb wiring left for v2 model search).
struct StartupSelectorComponent<'a> {
    title: String,
    labels: Vec<&'static str>,
    selected: usize,
    theme: &'a Theme,
    on_select: Option<Box<dyn FnMut(usize)>>,
    on_cancel: Option<Box<dyn FnMut()>>,
}

impl<'a> StartupSelectorComponent<'a> {
    fn new(
        title: String,
        labels: Vec<&'static str>,
        theme: &'a Theme,
        on_select: Box<dyn FnMut(usize)>,
        on_cancel: Box<dyn FnMut()>,
    ) -> Self {
        StartupSelectorComponent {
            title,
            labels,
            selected: 0,
            theme,
            on_select: Some(on_select),
            on_cancel: Some(on_cancel),
        }
    }

    /// move option (up/down), clamped within bounds.
    fn move_selection(&mut self, delta: isize) {
        let len = self.labels.len() as isize;
        if len == 0 {
            return;
        }
        let next = (self.selected as isize + delta).clamp(0, len - 1) as usize;
        self.selected = next;
    }
}

impl Component for StartupSelectorComponent<'_> {
    /// render option dialog (multiline title + option list + key hints).
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        let theme = self.theme;
        let accent = Style::fg(theme.fg(ColorToken::Accent));
        let muted = Style::fg(theme.fg(ColorToken::Muted));
        let text = Style::fg(theme.fg(ColorToken::Text));
        let dim = Style::fg(theme.fg(ColorToken::Dim));

        let mut lines: Vec<(String, Style)> = Vec::new();
        // split title by line (prompt text may contain multiple lines)
        for line in self.title.lines() {
            lines.push((line.to_string(), muted.clone()));
        }
        lines.push((String::new(), Style::default()));
        for (i, label) in self.labels.iter().enumerate() {
            let selected = i == self.selected;
            let prefix = if selected { "→ " } else { "  " };
            let style = if selected {
                accent.clone()
            } else {
                text.clone()
            };
            lines.push((format!("{prefix}{label}"), style));
        }
        lines.push((String::new(), Style::default()));
        lines.push(("↑↓ navigate    Esc cancel".to_string(), dim));

        let mut painted = 0u16;
        for (text, style) in lines {
            if painted >= height {
                break;
            }
            buf.fill(x, y + painted, width, 1, &Style::default());
            buf.write(x + 1, y + painted, &style, &text);
            painted += 1;
        }
        painted
    }

    /// handle_key: ↑/↓/j/k move, Enter submit, Esc cancel.
    fn handle_key(&mut self, key: &KeyEvent) -> bool {
        match (key.code, key.modifiers) {
            (KeyCode::Up, _) | (KeyCode::Char('k'), KeyModifiers::NONE) => {
                self.move_selection(-1);
                true
            }
            (KeyCode::Down, _) | (KeyCode::Char('j'), KeyModifiers::NONE) => {
                self.move_selection(1);
                true
            }
            (KeyCode::Enter, _) => {
                if let Some(cb) = &mut self.on_select {
                    cb(self.selected);
                }
                true
            }
            (KeyCode::Esc, _) | (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                if let Some(cb) = &mut self.on_cancel {
                    cb();
                }
                true
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn theme() -> &'static Theme {
        zen_tui::theme::builtin_dark_theme()
    }

    // ── first_time_setup_gate (pure function; env split out to avoid races) ────────────

    #[test]
    fn gate_returns_false_when_env_agent_dir_set() {
        assert!(!first_time_setup_gate(Some("/custom/agent"), false));
        assert!(!first_time_setup_gate(Some("/custom/agent"), true));
    }

    #[test]
    fn gate_returns_true_only_when_no_env_and_no_settings() {
        assert!(first_time_setup_gate(None, false), "First launch with no settings → true");
        assert!(
            !first_time_setup_gate(None, true),
            "Existing settings.json → false"
        );
    }

    #[test]
    fn gate_empty_env_string_treated_as_unset_by_caller_filter() {
        // empty-string filtering happens in should_run_first_time_setup (filter !is_empty); gate only sees
        // None/Some non-empty; here it suffices to lock down gate semantics for None.
        assert!(first_time_setup_gate(None, false));
    }

    // ── should_run_first_time_setup (path branch; env reading and gate tested separately) ─

    #[test]
    fn should_run_first_time_setup_false_when_settings_exists() {
        let dir =
            std::env::temp_dir().join(format!("zen_startup_ui_exists_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let settings_path = dir.join("settings.json");
        // does not exist → true
        assert!(should_run_first_time_setup(&settings_path));
        // after creation → false
        std::fs::write(&settings_path, "{}").unwrap();
        assert!(!should_run_first_time_setup(&settings_path));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── startup_theme (theme resolution/fallback; None branch falls back to detection) ───────

    #[test]
    fn startup_theme_explicit_builtin() {
        let theme = startup_theme(Some("dark")).unwrap();
        assert_eq!(theme.name, "dark");
        let theme = startup_theme(Some("light")).unwrap();
        assert_eq!(theme.name, "light");
    }

    #[test]
    fn startup_theme_unknown_name_falls_back_to_dark() {
        let theme = startup_theme(Some("nonexistent")).unwrap();
        assert_eq!(
            theme.name, "dark",
            "Unknown theme falls back to built-in dark (ThemeController semantics)"
        );
    }

    #[test]
    fn startup_theme_none_uses_detected_builtin() {
        // no preference → choose built-in dark/light theme based on terminal background detection; detection result depends on environment,
        // only assert it falls within the built-in set (immune to COLORFGBG concurrency races).
        let theme = startup_theme(None).unwrap();
        assert!(
            theme.name == "dark" || theme.name == "light",
            "Detection fallback should be the built-in theme (actual {})",
            theme.name
        );
    }

    // ── StartupSelectorComponent (pure component keybindings, no TuiLoop) ─────────

    #[test]
    fn selector_down_up_moves_and_clamps() {
        let mut comp = StartupSelectorComponent::new(
            "title".to_string(),
            vec!["A", "B", "C"],
            theme(),
            Box::new(|_| {}),
            Box::new(|| {}),
        );
        comp.handle_key(&KeyEvent::from(KeyCode::Down));
        assert_eq!(comp.selected, 1);
        comp.handle_key(&KeyEvent::from(KeyCode::Down));
        assert_eq!(comp.selected, 2);
        comp.handle_key(&KeyEvent::from(KeyCode::Down));
        assert_eq!(comp.selected, 2, "Bottom clamp");
        comp.handle_key(&KeyEvent::from(KeyCode::Up));
        assert_eq!(comp.selected, 1);
        comp.handle_key(&KeyEvent::from(KeyCode::Up));
        comp.handle_key(&KeyEvent::from(KeyCode::Up));
        assert_eq!(comp.selected, 0, "Top clamp");
    }

    #[test]
    fn selector_jk_also_move() {
        let mut comp = StartupSelectorComponent::new(
            "title".to_string(),
            vec!["A", "B"],
            theme(),
            Box::new(|_| {}),
            Box::new(|| {}),
        );
        assert!(comp.handle_key(&KeyEvent::from(KeyCode::Char('j'))));
        assert_eq!(comp.selected, 1);
        assert!(comp.handle_key(&KeyEvent::from(KeyCode::Char('k'))));
        assert_eq!(comp.selected, 0);
    }

    #[test]
    fn selector_enter_submits_selected_index() {
        let submitted = Rc::new(RefCell::new(None));
        let s = submitted.clone();
        let mut comp = StartupSelectorComponent::new(
            "title".to_string(),
            vec!["A", "B", "C"],
            theme(),
            Box::new(move |i| *s.borrow_mut() = Some(i)),
            Box::new(|| {}),
        );
        comp.handle_key(&KeyEvent::from(KeyCode::Down));
        assert!(comp.handle_key(&KeyEvent::from(KeyCode::Enter)));
        assert_eq!(*submitted.borrow(), Some(1));
    }

    #[test]
    fn selector_esc_cancels() {
        let cancelled = Rc::new(RefCell::new(0));
        let c = cancelled.clone();
        let mut comp = StartupSelectorComponent::new(
            "title".to_string(),
            vec!["A"],
            theme(),
            Box::new(|_| {}),
            Box::new(move || *c.borrow_mut() += 1),
        );
        assert!(comp.handle_key(&KeyEvent::from(KeyCode::Esc)));
        assert_eq!(*cancelled.borrow(), 1);
    }

    #[test]
    fn selector_unknown_keys_not_consumed() {
        let mut comp = StartupSelectorComponent::new(
            "title".to_string(),
            vec!["A"],
            theme(),
            Box::new(|_| {}),
            Box::new(|| {}),
        );
        assert!(!comp.handle_key(&KeyEvent::from(KeyCode::Char('a'))));
    }

    #[test]
    fn selector_renders_title_and_options() {
        let mut comp = StartupSelectorComponent::new(
            "Pick one\nsecond line".to_string(),
            vec!["Continue", "Cancel"],
            theme(),
            Box::new(|_| {}),
            Box::new(|| {}),
        );
        let mut buf = ScreenBuffer::new(60, 12);
        let painted = comp.render(&mut buf, 0, 0, 60, 12);
        assert!(painted >= 7);
        let mut out = Vec::new();
        buf.flush(&mut out).unwrap();
        let text = crate::components::strip_ansi(&String::from_utf8_lossy(&out));
        assert!(text.contains("Pick one"));
        assert!(text.contains("Continue"));
        assert!(text.contains("Cancel"));
        assert!(text.contains("→ Continue"), "First item, when preselected, should have an arrow");
    }

    // ── persist_first_time_theme (persistence segment, tempdir disk-write assertions) ────

    #[test]
    fn persist_first_time_theme_writes_settings_json() {
        let guard = tempfile::tempdir().unwrap();
        let agent_dir = guard.path().join("agent");
        let settings = SettingsManager::load(guard.path(), &agent_dir);
        persist_first_time_theme(&settings, TerminalTheme::Light);
        // update_global auto diff + atomic disk write (parent directories auto-created)
        let content = std::fs::read_to_string(agent_dir.join("settings.json")).unwrap();
        assert!(
            content.contains("\"theme\": \"light\""),
            "settings.json should contain theme=light (actual {content})"
        );
        // in-memory view sync
        assert_eq!(settings.effective().theme.as_deref(), Some("light"));
    }

    #[test]
    fn theme_setting_name_mapping() {
        assert_eq!(theme_setting_name(TerminalTheme::Dark), "dark");
        assert_eq!(theme_setting_name(TerminalTheme::Light), "light");
    }
}
