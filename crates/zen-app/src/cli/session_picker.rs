// REQ-APP-08 (pi-334 session-picker.ts) — launch session picker.
// Corresponds to original TS: reference/pi-original/packages/coding-agent/src/cli/session-picker.ts
// `selectSession` (startup TUI + SessionSelectorComponent + KeybindingsManager assembly).
// Implementation notes: see tasks/T-AP-08-5-implemented.md.
//
// Differences from pi:
// - pi depends on pi-335 `createStartupTui` (startup UI framework, T-AP-08-6); zen delivers in this task
//   a minimal startup TUI shell (no scrollback, single-content TuiLoop); T-AP-08-6 uses this as the extension point;
// - Lazy loader + progress callback → preload `Vec<SessionListing>` before mounting (the component loader is
//   synchronous callback; T-AP-11-2 notes "async preload completes before construction"), progress display semantics preserved
//   (`SessionListProgress` field still exists), data source becomes a one-time snapshot—recorded as an accepted deviation.

use crate::components::session_selector::{SessionListLoader, SessionSelectorComponent};
use crate::keybindings::app_default_bindings;
use crate::session::jsonl::delete_session;
use crate::session::session_listing::{SessionListProgress, SessionListing};
use crate::session::session_manager::SessionManager;
use crate::theme::{ControllerError, ThemeController};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use thiserror::Error;
use tokio_util::sync::CancellationToken;
use zen_tui::app_loop::{EventOutcome, TuiError, TuiLoop};
use zen_tui::components::Component;
use zen_tui::keybindings::KeybindingsManager;
use zen_tui::screen::ScreenBuffer;

/// Startup picker error aggregation (TuiLoop / theme / session IO).
/// No Io variant: `select_session(current, all, root)` receives the preloaded list; inside the function only
/// Rename/delete callbacks touch files via `SessionManager::open`/`delete_session` (mapped to Session variant).
#[derive(Debug, Error)]
pub enum SessionPickerError {
    /// TUI loop start/run/stop errors.
    #[error(transparent)]
    Tui(#[from] TuiError),
    /// Theme controller construction error.
    #[error(transparent)]
    Theme(#[from] ControllerError),
    /// Session list/open errors (rename/delete callbacks).
    #[error(transparent)]
    Session(#[from] crate::session::session_manager::SessionManagerError),
}

/// Launch session picker (--resume flag): run a startup TUI without scrollback, mount
/// SessionSelectorComponent——
/// "User runs zen --resume in the terminal, a session list pops up for them to choose one to continue."
/// Preload current/all `SessionListing` sources → assemble theme/keymap → run TuiLoop
/// SessionSelectorComponent (on_select→return path, on_cancel→None, rename/delete
/// callbacks wired to `append_session_info`/`delete_session`) → finish by restoring the terminal.
/// Returning `None` means the user canceled (main prints "No session selected" + exit(0) like pi).
/// `sessions_root` lets rename/delete callbacks `open` session files (sessions root directory, `--session-dir`
/// override or default `get_sessions_dir()`, closing the T-AP-02-3 session_dir seam).
/// # Consumers
/// main.rs `select_session_manager` (--resume branch).
pub async fn select_session(
    current: Vec<SessionListing>,
    all: Vec<SessionListing>,
    sessions_root: &Path,
) -> Result<Option<String>, SessionPickerError> {
    // Local-scope !Send (Rc<RefCell> component shared across closures), following the interactive run LocalSet precedent.
    let local = tokio::task::LocalSet::new();
    local
        .run_until(run_picker_inner(current, all, sessions_root))
        .await
}

/// Actual body of `select_session` (executes in LocalSet context; holds !Send component state).
async fn run_picker_inner(
    current: Vec<SessionListing>,
    all: Vec<SessionListing>,
    sessions_root: &Path,
) -> Result<Option<String>, SessionPickerError> {
    // 1. theme + keybinding table assembly
    let controller = ThemeController::new(None, None, None)?;
    let theme = controller.current().clone();
    let kb = KeybindingsManager::new(app_default_bindings(), HashMap::new());

    // 2. List snapshot cell (delete callback removes entries from the cell → loader rereads)
    let current_cell: Rc<RefCell<Vec<SessionListing>>> = Rc::new(RefCell::new(current));
    let all_cell: Rc<RefCell<Vec<SessionListing>>> = Rc::new(RefCell::new(all));
    let loader_current = {
        let cell = Rc::clone(&current_cell);
        Box::new(move |_p: &mut dyn FnMut(SessionListProgress)| cell.borrow().clone())
            as SessionListLoader
    };
    let loader_all = {
        let cell = Rc::clone(&all_cell);
        Box::new(move |_p: &mut dyn FnMut(SessionListProgress)| cell.borrow().clone())
            as SessionListLoader
    };

    // 3. Shared result cell + cancel handle (select/cancel callbacks write and cancel → run loop exits)
    let result_cell: Arc<Mutex<Option<Option<String>>>> = Arc::new(Mutex::new(None));
    let cancel = CancellationToken::new();

    // 4. Delete/rename callbacks (shared cell + root)
    let root_for_open = sessions_root.to_path_buf();
    let on_delete = {
        let current_cell = Rc::clone(&current_cell);
        let all_cell = Rc::clone(&all_cell);
        Some(Box::new(move |path: String| {
            let _ = delete_session(Path::new(&path));
            current_cell
                .borrow_mut()
                .retain(|l| l.path.to_string_lossy() != path);
            all_cell
                .borrow_mut()
                .retain(|l| l.path.to_string_lossy() != path);
        }) as Box<dyn FnMut(String)>)
    };
    let rename_session = {
        let root = root_for_open.clone();
        Some(Box::new(move |path: String, name: Option<&str>| {
            if let Ok(mgr) = SessionManager::open(Path::new(&path), None, &root) {
                mgr.append_session_info(name);
            }
        }) as Box<dyn FnMut(String, Option<&str>)>)
    };

    // 5. Picker component (Rc<RefCell> shared across on_event/draw closures)
    let selector_cell = Rc::new(RefCell::new(SessionSelectorComponent::new(
        loader_current,
        loader_all,
        None,
        theme.clone(),
        kb,
        Box::new({
            let result_cell = Arc::clone(&result_cell);
            let cancel = cancel.clone();
            move |path: String| {
                *result_cell.lock().unwrap() = Some(Some(path));
                cancel.cancel();
            }
        }),
        Box::new({
            let result_cell = Arc::clone(&result_cell);
            let cancel = cancel.clone();
            move || {
                *result_cell.lock().unwrap() = Some(None);
                cancel.cancel();
            }
        }),
        rename_session,
        on_delete,
    )));

    // 6. start TUI shell (no scrollback, single content; T-AP-08-6 extension point)
    let mut tui = TuiLoop::new()?;
    tui.start()?;

    let run_result = {
        let on_event_cell = Rc::clone(&selector_cell);
        let draw_cell = Rc::clone(&selector_cell);
        let cancel = cancel.clone();
        tui.run(
            &cancel,
            move |event| {
                match event {
                    crossterm::event::Event::Key(key) => {
                        // Defensive filtering of Release events (always Press when enhanced protocol REPORT_EVENT_TYPES is not enabled)
                        if key.kind == crossterm::event::KeyEventKind::Release {
                            return EventOutcome::Continue;
                        }
                        let mut comp = on_event_cell.borrow_mut();
                        comp.handle_key(&key);
                        EventOutcome::Continue
                    }
                    _ => EventOutcome::Continue,
                }
            },
            move |buf: &mut ScreenBuffer| {
                let (w, h) = buf.size();
                let mut comp = draw_cell.borrow_mut();
                let _ = comp.render(buf, 0, 0, w, h);
            },
        )
        .await
    };
    // Both run and stop are executed (restore the terminal first even if run errors, then propagate the error)
    let stop_result = tui.stop();
    run_result?;
    stop_result?;

    // 7. Result (on_select/on_cancel already wrote to shared cell; callbacks happen before run returns)
    let result = result_cell.lock().unwrap().take();
    Ok(result.unwrap_or(None))
}
