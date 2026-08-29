// REQ-APP-11 (pi-355): interactive mode main coordinator — assembly and dispatch on the main task side.
// See reference/pi-original/packages/coding-agent/src/modes/interactive/interactive-mode.ts
// Architecture: dual tasks + channels. This file = business side (main task): holds AgentSessionRuntime, subscribes to agent events,
// registers signals, consumes user input via input_rx, pushes UI changes via ui_event_tx; rendering state lives in the UI task
// (ui.rs `InteractiveUi`). UI events are applied uniformly in the draw frame (on_wake only enqueues + declares redraw intent),
// see the assembly comments in `init`.
// Implementation notes in tasks/T-AP-11-1-implemented.md.

pub mod ui;

pub(crate) use crate::keybindings::{app_default_bindings, AppAction};

use crate::components::footer::{add_usage_to_totals, FooterData, UsageTotals};
use crate::components::settings_selector::{DeliveryMode, DoubleEscapeAction, SettingsConfig};
use crate::components::tree_selector::FilterMode;
use crate::modes::interactive::ui::InteractiveUi;
use crate::modes::print_mode::stop_error_message;
use crate::session::agent_session::{AgentSession, AgentSessionError};
use crate::session::runtime::AgentSessionRuntime;
use crate::session::session_manager::SessionManagerError;
use crate::theme::{ControllerError, ThemeController};
use serde_json::Value;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use zen_ai::tool::ToolExecutionResult;
use zen_ai::types::{ImageContent, Message, ThinkingLevel};
use zen_core::harness::types::AgentEvent;
use zen_tui::app_loop::{EventOutcome, TuiError, TuiLoop, WakeOutcome};
use zen_tui::keybindings::KeybindingsManager;
use zen_tui::keys::{
    is_apple_terminal_session, native_shift_pressed, normalize_apple_terminal_enter,
};
use zen_tui::theme::Theme;

/// Startup configuration carrier for run_interactive_mode (aligned with pi `InteractiveModeOptions` v1 subset),
/// consumed once; no special serialization constraints.
pub struct InteractiveModeOptions {
    /// First message (main has already assembled @file contents, T-AP-08-2).
    initial_message: Option<String>,
    /// Always an empty Vec in Phase 2 (same convention as print_mode).
    initial_images: Vec<ImageContent>,
    /// Additional prompts after initial (`args.messages()`).
    messages: Vec<String>,
    /// Force verbose startup (overrides quietStartup; affects header display until v1 settings land).
    verbose: bool,
}

/// UI change messages from the main task → UI task: the agent event bridge (`bridge_agent_event`) and business actions
/// produced by (status/footer/queue updates), applied to the data snapshot on the UI side by `InteractiveUi::apply`.
/// Sent within the synchronous callback via `UnboundedSender`; **must not be lost** (unbounded means no blocking when full).
/// Cloned for enqueueing in the UI task's on_wake (`Arc<Mutex<VecDeque<UiEvent>>>`).
#[allow(clippy::large_enum_variant)] // The Message snapshot variant is fairly large, which is acceptable internally to the channel design
#[derive(Debug, Clone)]
pub(crate) enum UiEvent {
    /// Status line (Working/Retry/Compaction/BranchSummary/Error/Warning; idle = None + ClearStatus).
    Status(StatusSpec),
    /// Clear the current status line.
    ClearStatus,
    /// New message starts: role=user/custom appends a line, role=assistant opens a streaming line
    /// (`streaming_index` points to it; subsequent Update/End replace the snapshot wholesale).
    MessageStart(Message),
    /// Streaming delta: only when an active streaming line exists, replace its message snapshot and request a repaint.
    MessageUpdate(Message),
    /// Message completion: final text + error/termination wording (`error_text` via `print_mode::stop_error_message`;
    /// None for normal termination), replaces the snapshot and closes the streaming line.
    MessageEnd {
        message: Message,
        error_text: Option<String>,
    },
    /// Tool call start: register `tool_indices` and lay out a pending line.
    ToolStart {
        id: String,
        name: String,
        args: Value,
    },
    /// Streaming argument update (`ToolExecutionUpdate`).
    ToolUpdate { id: String, args: Value },
    /// Tool completion: harness-level result (zen_ai::tool::ToolExecutionResult) + error flag;
    /// converted on the UI side into a component render snapshot (text block concatenation + is_error).
    ToolEnd {
        id: String,
        result: ToolExecutionResult,
        is_error: bool,
    },
    /// Queued message snapshot change (follow-up queue).
    PendingChanged(Vec<String>),
    /// Per-frame footer data snapshot (assembled by the main task from AgentSession state).
    Footer(FooterData),
    #[allow(dead_code)] // v1 seam: bash session-state switch (sent explicitly by the main task once T-AP-04-2 lands);
    // the editor body `!` prefix already drives the border color locally via on_change
    BashMode(bool),
    /// Continuous animation toggle (true during the streaming loader → constant 16ms repaints; the render layer, via
    /// `WakeOutcome::Animate` takes effect).
    SetAnimation(bool),
    #[allow(dead_code)] // v1 has no file watcher wired up (sent via ThemeController on_change after T-AP-02-3 lands)
    /// Theme hot reload: replace the owned Theme wholesale and rebuild border colors (v1 seam; apply is implemented).
    ThemeChanged(Theme),
    /// Terminal title (session name + cwd).
    TerminalTitle(String),
    /// Editor content restore (follow-up queue return / batch restore / clear).
    EditorSetText(String),
}

/// status line spec: `InteractiveUi::apply` (`apply_status`) maintains a persistent `StatusIndicator`
/// instance (reused across frames, preserving the Loader clock anchor; animation frames advance across frames); the rendering side only draws, never creates.
/// Errors/warnings are expressed directly via `Status(StatusSpec::Error/Warning)` (render-side semantic colors drawn directly).
#[derive(Debug, Clone)]
pub(crate) enum StatusSpec {
    /// Busy status line (spinner + message, message is updatable).
    Working { message: String },
    /// Request retry countdown (`StatusIndicator::retry` argument).
    #[allow(dead_code)] // v1 has no retry orchestration (StreamFn-level retry deferred to T-AP-04-2)
    Retry {
        attempt: u32,
        max_attempts: u32,
        delay_ms: u64,
    },
    /// Context compression in progress.
    #[allow(dead_code)] // v1 has no compression orchestration (integrate after T-AP-03-2 lands)
    Compaction,
    /// Branch summary generation in progress.
    #[allow(dead_code)] // v1 has no fork/branch digest (wire up after T-AP-08-5 lands)
    BranchSummary,
    /// Error hint line (semantic colors drawn directly).
    Error(String),
    /// Warning hint line (semantic colors drawn directly).
    Warning(String),
}

/// UI task → main task input commands: output of editor/keybinding routing, consumed by the `run()` dispatch loop.
/// Submitted raw text contains a `!`/`!!` prefix (routing decision in `handle_submit`).
#[derive(Debug, PartialEq)]
pub(crate) enum InputCommand {
    /// Editor submit (Enter): plain message / `!` bash / `/` command raw text.
    Submit(String),
    /// Follow-up queueing (Alt+Enter): carries the editor's current text, read and cleared locally by the UI side.
    /// (Implementation extension of the 5.3 baseline table: FollowUp must carry a text payload so the main task can queue it.)
    FollowUp(String),
    /// App-level keybinding hit (parsed by `KeybindingsManager<AppAction>`).
    Action(AppAction),
    /// Esc: context-sensitive handling requires session state (abort stream / abort bash / exit bash mode /
    /// double-Esc selector), so it is decided by the main task rather than the UI side.
    Escape,
    /// Ctrl+C: clear editor; double press (500ms window) quits.
    CtrlC,
    /// Ctrl+D: exit when the editor is empty.
    CtrlD,
}

/// Interactive-mode coordinator errors — the aggregation point for all `Result`s on the main-task side.
/// Follows the thiserror precedent of `ThemeController::ControllerError` inside zen-app (a concrete
/// error type at the library boundary); top-level `run_interactive_mode` then aggregates via anyhow into an exit code.
#[derive(Debug, thiserror::Error)]
pub enum InteractiveModeError {
    /// Agent session errors (`session.prompt`/`steer` call chain, underlying `AgentLoopError`).
    #[error(transparent)]
    Agent(#[from] AgentSessionError),
    /// TUI loop / terminal errors (`TuiLoop` start/stop/four-way loop returns).
    #[error(transparent)]
    Tui(#[from] TuiError),
    /// UI task join failure (task panicked, converged at `ui_task` teardown).
    #[error("UI task failed: {0}")]
    UiTask(#[from] tokio::task::JoinError),
    /// Theme controller errors (`ThemeController::new` / watcher).
    #[error(transparent)]
    Theme(#[from] ControllerError),
    /// Session management error (resume hints on shutdown paths; seam reserved for T-AP-08-5).
    #[error(transparent)]
    Session(#[from] SessionManagerError),
    /// Signal registration errors (`tokio::signal` failure, `register_signal_handlers`).
    #[error("signal setup failed: {0}")]
    Signal(#[from] std::io::Error),
}

/// Interactive session driver (main-task side): holds AgentSessionRuntime, registers agent subscriptions and signal handling,
/// consumes user input via input_rx, pushes UI changes via ui_event_tx; run() is the "initial message + input dispatch" main loop.
///
/// **v1 known limitation**: while `handle_submit`'s prompt inline await is in flight, the main task does not consume input (queued);
/// input is dispatched in order after the turn ends — "steering while streaming" requires the Agent to support concurrent prompt/steer,
/// so this version falls back to idle dispatch (`is_run()` branch retained); see tasks/T-AP-11-1-implemented.md "Open Issues".
pub struct InteractiveMode {
    /// Sole source of session/agent/service.
    runtime: AgentSessionRuntime,
    /// Startup configuration.
    options: InteractiveModeOptions,
    /// Exit signal source (shared clone/child between the UI task and the signal task).
    cancel: CancellationToken,
    /// Input channel sender (created and held by new(), handed off to the UI task's on_event closure in init).
    input_tx: Option<mpsc::UnboundedSender<InputCommand>>,
    /// Input command receiver (consumed by the main-task dispatch loop).
    input_rx: mpsc::UnboundedReceiver<InputCommand>,
    /// UI change sender (main task → UI task; the ui_event_rx consumed at the scan head lives on the UI-task side).
    ui_event_tx: mpsc::UnboundedSender<UiEvent>,
    /// UI event receiver (handed off in init as the UI task's fourth external wakeup).
    ui_event_rx: Option<mpsc::UnboundedReceiver<UiEvent>>,
    /// UI task handle (for join; the task runs the TuiLoop four-way loop).
    ui_task: Option<tokio::task::JoinHandle<Result<(), TuiError>>>,
    /// Theme controller (Option because there is no Default placeholder; lazily constructed in init).
    theme_controller: Option<ThemeController>,
    /// Agent subscription/unsubscription handles (registered in init, called in shutdown/stop).
    subscription: Option<Box<dyn FnOnce() + Send>>,
    /// T-AP-11-2 type: v1 uses default values; real settings land after T-AP-02-3.
    settings: SettingsConfig,
    /// Idempotency guard (pre-check in init/run).
    is_initialized: bool,
    /// Idempotency guard (pre-check in shutdown/stop).
    is_shutting_down: bool,
    /// Double-press Esc detection (500ms window).
    last_escape_at: Option<Instant>,
    /// Double-press Ctrl+C detection (500ms window, aligned with pi handleCtrlC).
    last_ctrl_c_at: Option<Instant>,
    /// Exit request (CtrlD idle / CtrlC double-press / /quit → break the dispatch loop).
    exit_requested: bool,
    /// Footer usage accumulators (mutable within subscription callbacks; std Mutex guarantees callback thread safety).
    usage_totals: Arc<Mutex<UsageTotals>>,
    /// SIGTERM/SIGHUP listener task (register/unregister lifecycle pair).
    signal_task: Option<tokio::task::JoinHandle<()>>,
    /// Exit code written by the signal task (0 = no signal received; SIGTERM→143 / SIGHUP→129).
    signal_exit_code: Arc<AtomicI32>,
}

impl InteractiveMode {
    /// Constructs the coordinator from the assembled runtime and startup config.
    /// Only field initialization and channel creation; no IO/UI actions (idempotency guaranteed by init/run).
    pub fn new(runtime: AgentSessionRuntime, options: InteractiveModeOptions) -> Self {
        let (ui_event_tx, ui_event_rx) = mpsc::unbounded_channel();
        let (input_tx, input_rx) = mpsc::unbounded_channel();
        InteractiveMode {
            runtime,
            options,
            cancel: CancellationToken::new(),
            input_tx: Some(input_tx),
            input_rx,
            ui_event_tx,
            ui_event_rx: Some(ui_event_rx),
            ui_task: None,
            theme_controller: None,
            subscription: None,
            settings: default_settings(),
            is_initialized: false,
            is_shutting_down: false,
            last_escape_at: None,
            last_ctrl_c_at: None,
            exit_requested: false,
            usage_totals: Arc::new(Mutex::new(UsageTotals::default())),
            signal_task: None,
            signal_exit_code: Arc::new(AtomicI32::new(0)),
        }
    }

    /// Initialize the interactive session: create theme, container state, register signal and agent subscriptions, assemble and spawn the UI task.
    /// Idempotent (guarded by `is_initialized`); repeated calls are no-ops.
    ///
    /// # Behavior
    /// 1. `ThemeController::new` (T-AP-11-3) gets the current theme; v1 does not enable file watching
    ///    (themes_dir=None, hot-reload seam = `UiEvent::ThemeChanged`, wired in when T-AP-02-3 lands)
    /// 2. Construct `KeybindingsManager::new(app_default_bindings())` (v1 default table only)
    /// 3. Spawn UI task: `TuiLoop` + `Rc<RefCell<InteractiveUi>>`, running a four-way loop—
    ///    - on_event (!Send): key event → handle_key → `InputCommand` sent to the main task via input_tx
    ///    - fourth-way wake (Send): enqueue `UiEvent` into `Arc<Mutex<VecDeque>>` + declare `WakeOutcome`;
    ///      actual apply is executed uniformly in the draw frame (reads the same queue), avoiding !Send closure cross-thread issues
    ///    - draw (!Send): drain queue → apply → render
    /// 4. `register_signal_handlers`（SIGTERM/SIGHUP）+ `subscribe_to_agent`
    /// 5. Send initial terminal title/footer rendering
    pub async fn init(&mut self) -> Result<(), InteractiveModeError> {
        if self.is_initialized {
            return Ok(());
        }

        // 1. Theme controller: v1 has no custom theme directory or settings (wired after T-AP-02-3 lands); picks a builtin theme based on terminal background.
        let controller = ThemeController::new(None, None, None)?;
        let theme = controller.current().clone();
        self.theme_controller = Some(controller);

        // 2. Keybinding table (v1 default table only).
        let kb = KeybindingsManager::new(app_default_bindings(), HashMap::new());

        // 3. UI task assembly (dual tasks + channels).
        //    - `ui_event_rx`: receiver for the fourth-way external wake (main task pushes via ui_event_tx);
        //    - `input_tx`: channel for sending key commands from the UI task's on_event closure.
        let ui_event_rx = self
            .ui_event_rx
            .take()
            .expect("init idempotent guard ensures the UI event receiver created by new() is handed over only once");
        let input_tx = self
            .input_tx
            .take()
            .expect("init idempotency guard ensures the input sender created by new() is handed off only once");
        let cwd = self.runtime.session.cwd().to_string();
        let ui_cell = Rc::new(RefCell::new(InteractiveUi::new(theme, kb, cwd)));
        // Suspend event queue: on_wake (Send closure) enqueues, draw (!Send closure) dequeues — accessed sequentially within the UI task.
        let pending: Arc<Mutex<VecDeque<UiEvent>>> = Arc::new(Mutex::new(VecDeque::new()));
        let pending_wake = Arc::clone(&pending);
        let pending_draw = Arc::clone(&pending);
        let cancel = self.cancel.child_token();
        let ui_on_event = Rc::clone(&ui_cell);
        let ui_draw = Rc::clone(&ui_cell);
        // UI task is scheduled via spawn_local within a LocalSet (closure captures the !Send ui_cell, see the run() comment)
        self.ui_task = Some(tokio::task::spawn_local(async move {
            let mut tui = TuiLoop::new()?;
            tui.start()?;
            tui.set_external_wakeup(
                ui_event_rx,
                Box::new(move |ev: &UiEvent| {
                    // Synchronous callback: only enqueues + declares a repaint intent, must not await; apply is executed uniformly by the draw frame
                    pending_wake
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push_back(ev.clone());
                    match ev {
                        UiEvent::SetAnimation(v) => WakeOutcome::Animate(*v),
                        _ => WakeOutcome::Render,
                    }
                }),
            );
            // Apple Terminal session detection: evaluated once outside the closure and cached (env query done once on demand, T-AP-11-1-fix)
            let is_apple_terminal = is_apple_terminal_session();
            tui.run_scrollback(
                &cancel,
                move |event| {
                    match event {
                        crossterm::event::Event::Key(key) => {
                            // Defensive filtering: when the enhanced protocol has not enabled REPORT_EVENT_TYPES, always Press;
                            // reject any Release leaking in (T-AP-11-1-fix design decision 1 — even if release events are
                            // produced via other means in the future, they won't cause double-triggering).
                            if key.kind == crossterm::event::KeyEventKind::Release {
                                return EventOutcome::Continue;
                            }
                            // Apple Terminal fallback: this terminal doesn't send CSI-u, rewrite bare `\r`+Shift to
                            // Enter+SHIFT (the editor's existing mapping then inserts a newline). Zero overhead pass-through on non-Apple sessions
                            // (native_shift_pressed is macOS-only and is_apple_terminal is checked first here).
                            let key = if is_apple_terminal {
                                normalize_apple_terminal_enter(key, true, native_shift_pressed())
                            } else {
                                key
                            };
                            let mut ui = ui_on_event.borrow_mut();
                            if let Some(cmd) = ui.handle_key(&key) {
                                let _ = input_tx.send(cmd);
                            }
                        }
                        // Bracketed paste (Terminal::start already enables EnableBracketedPaste): the whole segment goes into the editor.
                        // Before the fix, Event::Paste was silently dropped (T-AP-11-1-fix design decision 2).
                        crossterm::event::Event::Paste(text) => {
                            let mut ui = ui_on_event.borrow_mut();
                            ui.handle_paste(&text);
                        }
                        // Other events (Resize is handled inside run_loop, Focus etc. are ignored)
                        _ => {}
                    }
                    EventOutcome::Continue
                },
                move |canvas| {
                    // first consume pending events, then render the current frame (content line stream + fixed bottom area; scrollback holds off-screen history)
                    let mut ui = ui_draw.borrow_mut();
                    loop {
                        let ev = pending_draw
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .pop_front();
                        match ev {
                            Some(ev) => ui.apply(ev),
                            None => break,
                        }
                    }
                    ui.render_scrollback(canvas);
                },
            )
            .await
        }));

        // 4. Signal registration + agent subscription (start the UI first, then attach listeners, so events have a view ready to receive them).
        self.register_signal_handlers();
        let unsub = self.subscribe_to_agent();
        self.subscription = Some(unsub);

        // 5. Initial render: terminal title (session name + cwd) + footer baseline; emit session diagnostics when verbose.
        let title = format!("zen · {}", self.runtime.session.cwd());
        let _ = self.ui_event_tx.send(UiEvent::TerminalTitle(title));
        self.push_footer();
        if self.options.verbose {
            tracing::info!(
                model = %self.runtime.session.model().id,
                cwd = %self.runtime.session.cwd(),
                "[interactive] session ready"
            );
        }

        self.is_initialized = true;
        Ok(())
    }

    /// Main interactive loop: after handling the initial message, enters the dispatch loop until an exit signal/command.
    /// User-visible effects: press Enter after startup to send a message, the agent streams frame by frame to the screen, Ctrl+C/D,
    /// Esc, slash commands, and `!` bash are all dispatched within the loop.
    ///
    /// # Behavior
    /// 1. `init()` (idempotent) → sends `options.initial_message` and `messages`
    /// 2. `dispatch_loop`: `select!` dispatches `InputCommand` and cancellation signals
    /// 3. Regardless of whether the loop ends normally or with an error, `shutdown(from_signal)` runs for cleanup (ensures UI/terminal/subscription teardown)
    pub async fn run(&mut self) -> Result<(), InteractiveModeError> {
        // The UI task depends on !Send state (`Rc<RefCell<InteractiveUi>>`, see the "Send notes" in ui.rs),
        // so it cannot use tokio::spawn directly — the entire run executes within a LocalSet: init starts the UI task with spawn_local,
        // and while run_inner awaits (prompt/dispatch/join), the LocalSet keeps polling the UI task, so rendering never blocks.
        let local = tokio::task::LocalSet::new();
        local.run_until(self.run_inner()).await
    }

    /// The actual body of `run` (executed within the LocalSet context created by `run`).
    async fn run_inner(&mut self) -> Result<(), InteractiveModeError> {
        self.init().await?;

        // initial_images: always empty in v1 (main always passes an empty Vec, same convention as print_mode); @file images belong to Phase 2.
        if !self.options.initial_images.is_empty() {
            tracing::debug!(
                count = self.options.initial_images.len(),
                "[interactive] initial images ignored in v1 (image support pending)"
            );
        }

        let loop_result = async {
            if let Some(msg) = self.options.initial_message.clone() {
                self.handle_submit(msg).await?;
            }
            let messages = self.options.messages.clone();
            for msg in messages {
                self.handle_submit(msg).await?;
            }
            self.dispatch_loop().await
        }
        .await;

        // Non-zero exit code from a signal path → shutdown(from_signal=true) (no resume hint printed)
        let from_signal = self.signal_exit_code.load(Ordering::Relaxed) != 0;
        let shutdown_result = self.shutdown(from_signal).await;
        loop_result?;
        shutdown_result?;
        Ok(())
    }

    /// Input dispatch loop: two-way select on `input_rx.recv()` / `cancel.cancelled()`.
    /// CtrlD (idle), CtrlC double-tap, /quit, cancel signal → break (then shutdown cleanup).
    async fn dispatch_loop(&mut self) -> Result<(), InteractiveModeError> {
        loop {
            tokio::select! {
                cmd = self.input_rx.recv() => {
                    match cmd {
                        Some(InputCommand::Submit(text)) => {
                            // Single-turn prompt error → report error in status line and continue (do not break the interactive session)
                            if let Err(e) = self.handle_submit(text).await {
                                self.show_error(format!("{e}"));
                            }
                        }
                        Some(InputCommand::FollowUp(text)) => {
                            if self.runtime.session.is_run() {
                                self.runtime.session.follow_up(&text);
                                self.push_pending();
                            } else {
                                // Alt+Enter is equivalent to Enter when idle (aligned with pi handleFollowUp's non-streaming branch)
                                self.handle_submit(text).await?;
                            }
                        }
                        Some(InputCommand::Action(action)) => self.handle_app_action(action),
                        Some(InputCommand::Escape) => self.handle_escape(),
                        Some(InputCommand::CtrlC) => self.handle_ctrl_c(),
                        Some(InputCommand::CtrlD) => {
                            if self.runtime.session.is_run() {
                                self.runtime.session.abort();
                                self.show_status("Aborting...");
                            } else {
                                self.exit_requested = true;
                            }
                        }
                        None => break, // UI task has already ended (channel closed)
                    }
                    if self.exit_requested {
                        break;
                    }
                }
                _ = self.cancel.cancelled() => break, // Signal task / shutdown trigger
            }
        }
        Ok(())
    }

    /// Teardown (synchronous part): cancel triggers UI task exit, deregisters signals and agent subscription.
    /// Idempotent; **does not join the UI task** (the join is done asynchronously by `shutdown`, ensuring terminal restoration happens before returning).
    pub fn stop(&mut self) -> Result<(), InteractiveModeError> {
        if !self.is_shutting_down {
            self.cancel.cancel();
            self.unregister_signal_handlers();
            if let Some(sub) = self.subscription.take() {
                sub();
            }
        }
        Ok(())
    }

    /// Graceful exit (pi `shutdown`): `registry.kill_all()` terminates registered detached child process trees
    /// (T-AP-09-3, same as print_mode), stop UI (cancel + join), unsubscribe; `from_signal` path
    /// Do not print the resume hint.
    /// Unlike pi: **does not `process::exit`**; after returning, `run_interactive_mode` converges the exit code,
    /// keeping the library layer testable.
    async fn shutdown(&mut self, from_signal: bool) -> Result<(), InteractiveModeError> {
        if self.is_shutting_down {
            return Ok(());
        }
        self.is_shutting_down = true;

        // 1. Terminate registered detached child processes (idempotent; no-op if the signal task already called kill_all).
        self.runtime.services.registry.kill_all();

        // 2. Stop UI: cancel triggers the TuiLoop cancellation branch to exit → join (terminal restoration happens in the UI task via tui.stop()).
        self.cancel.cancel();
        self.unregister_signal_handlers();
        if let Some(sub) = self.subscription.take() {
            sub();
        }
        if let Some(task) = self.ui_task.take() {
            task.await??;
        }

        // 3. Resume hint (the signal path does not print, aligning with pi shutdown's fromSignal branch; wording aligns with
        //    `--resume` semantics — the path for `--session <path>` requires T-AP-08-5's session tree to land before it can be assembled).
        if !from_signal {
            eprintln!("To resume this session, run: zen --resume");
        }
        Ok(())
    }

    /// Register SIGTERM/SIGHUP handlers (tokio signal): on signal arrival → spawn task
    /// `registry.kill_all()` + write exit code + `cancel.cancel()` (the main task's dispatch loop exits on cancellation,
    /// then finalizes via shutdown(from_signal) — aligns with print_mode `await_signal_exit_code` semantics).
    /// SIGINT is not registered here (Ctrl+C goes through editor keybindings, aligned with pi).
    fn register_signal_handlers(&mut self) {
        let registry = self.runtime.services.registry.clone();
        let cancel = self.cancel.clone();
        let exit_code = Arc::clone(&self.signal_exit_code);
        self.signal_task = Some(tokio::spawn(async move {
            #[cfg(unix)]
            {
                use tokio::signal::unix::{signal, SignalKind};
                let mut sigterm = match signal(SignalKind::terminate()) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(error = %e, "[interactive] register SIGTERM failed");
                        return;
                    }
                };
                let mut sighup = match signal(SignalKind::hangup()) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(error = %e, "[interactive] register SIGHUP failed");
                        return;
                    }
                };
                let code = tokio::select! {
                    _ = sigterm.recv() => 143, // 128 + 15 (SIGTERM)
                    _ = sighup.recv() => 129,  // 128 + 1 (SIGHUP)
                };
                registry.kill_all();
                exit_code.store(code, Ordering::Relaxed);
                cancel.cancel();
            }
            #[cfg(not(unix))]
            {
                let _ = (registry, cancel, exit_code);
            }
        }));
    }

    /// Deregister signal handlers (abort listening tasks → signal stream drops accordingly).
    fn unregister_signal_handlers(&mut self) {
        if let Some(task) = self.signal_task.take() {
            task.abort();
        }
    }

    /// Subscribe to agent events: `runtime.session.subscribe` (T-AP-01-1 passthrough, callback synchronous
    /// `Fn(&AgentEvent) + Send + Sync`) → `bridge_agent_event`. Returns an unsubscribe handle.
    ///
    /// **Deviation from stub signature notes**: the subscription closure is `'static + Send + Sync` and cannot capture `&mut self`
    /// to call `handle_agent_event(&mut self, ...)` — the mapping logic is extracted into the free function `bridge_agent_event`;
    /// the closure only captures channel/usage accumulator/FooterCtx snapshots (all 'static and clonable).
    fn subscribe_to_agent(&mut self) -> Box<dyn FnOnce() + Send> {
        let tx = self.ui_event_tx.clone();
        let usage = Arc::clone(&self.usage_totals);
        let ctx = FooterCtx::from_session(&self.runtime.session, &self.settings);
        self.runtime
            .session
            .subscribe(Box::new(move |event: &AgentEvent| {
                bridge_agent_event(event, &tx, &usage, &ctx);
            }))
    }

    /// Main-task-side footer refresh (e.g., after switching thinking level): assemble and push based on current session state + usage accumulator.
    fn push_footer(&self) {
        let ctx = FooterCtx::from_session(&self.runtime.session, &self.settings);
        let usage = self.usage_totals.lock().unwrap_or_else(|e| e.into_inner());
        let _ = self.ui_event_tx.send(UiEvent::Footer(ctx.footer(&usage)));
    }

    /// Main-task-side pending queue refresh (follow-up text snapshot → UI pending line).
    fn push_pending(&self) {
        let rows = self.runtime.session.follow_up_messages();
        let _ = self.ui_event_tx.send(UiEvent::PendingChanged(rows));
    }

    /// App-level keybinding action dispatch (the onAction table of pi `setupKeyHandlers`).
    /// v1 coverage: Clear/Exit/Interrupt fallback, ThinkingCycle/Toggle, ToolsExpand, MessageCopy,
    /// FollowUp/Dequeue; session selector/tree actions (session.*/tree.*/fork) reserve seams —
    /// activated when T-AP-08-5 lands; currently emits a hint message.
    fn handle_app_action(&mut self, action: AppAction) {
        match action {
            // These actions are intercepted on the UI side by editor callbacks/keybindings (see ui.rs new assembly) and are theoretically unreachable; fallbacks align with each semantic.
            AppAction::Clear => self.handle_ctrl_c(),
            AppAction::Interrupt => self.handle_escape(),
            AppAction::Exit => self.exit_requested = true,
            // v1 covered actions
            AppAction::ThinkingCycle => self.cycle_thinking(),
            AppAction::ThinkingToggle => {
                self.show_warning("Toggle thinking blocks: pending settings (T-AP-02-3)")
            }
            AppAction::ToolsExpand => {} // Locally toggled fold set on the UI side (consumed by ui.rs handle_key)
            AppAction::MessageCopy => {
                self.show_warning("Copy message to clipboard: not available in v1")
            }
            AppAction::MessageFollowUp => {} // Already converted to a FollowUp(text) command on the UI side
            AppAction::MessageDequeue => {
                self.show_warning("No queued messages to restore (queue pop pending)")
            }
            // Session selector/tree actions: activate after T-AP-08-5 lands
            _ => self.show_warning("Not available in v1 (session selectors land in T-AP-08-5)"),
        }
    }

    /// Esc context-sensitive handling (pi onEscape): abort while streaming; idle records double-click window
    /// (500ms, no tree/fork selector in v1; double-click shows a hint).
    fn handle_escape(&mut self) {
        let now = Instant::now();
        let is_double = self
            .last_escape_at
            .is_some_and(|t| now.duration_since(t) < Duration::from_millis(500));
        self.last_escape_at = Some(now);
        if self.runtime.session.is_run() {
            self.runtime.session.abort();
            self.show_status("Aborting...");
        } else if is_double {
            self.show_warning("Double-Esc (tree/fork) is not available in v1");
        }
    }

    /// Ctrl+C (pi handleCtrlC): double press within 500ms → exit; single press → clear editor.
    fn handle_ctrl_c(&mut self) {
        let now = Instant::now();
        let is_double = self
            .last_ctrl_c_at
            .is_some_and(|t| now.duration_since(t) < Duration::from_millis(500));
        self.last_ctrl_c_at = Some(now);
        if is_double {
            self.exit_requested = true;
        } else {
            // Clear the editor (aligned with pi clearEditor; timestamp recorded on press, double-press detection matches pi)
            let _ = self.ui_event_tx.send(UiEvent::EditorSetText(String::new()));
        }
    }

    /// Cycle through thinking levels (pi cycleThinkingLevel): rotate through available levels from settings;
    /// persistence (append_thinking_level_change) and footer refresh are handled by set_thinking_level.
    fn cycle_thinking(&mut self) {
        let current = self.runtime.session.thinking_level();
        let levels = &self.settings.available_thinking_levels;
        if levels.is_empty() {
            return;
        }
        let next = match levels.iter().position(|l| *l == current) {
            Some(i) => levels[(i + 1) % levels.len()],
            None => levels[0],
        };
        self.runtime.session.set_thinking_level(next);
        self.push_footer();
        self.show_status(format!("Thinking: {next}"));
    }

    /// Editor submit dispatch (pi `setupEditorSubmitHandler`): `/` commands (v1 inline subset +
    /// T-AP-03-2 command table seam), `!`/`!!` bash prefixes, plain messages → `session.prompt`.
    async fn handle_submit(&mut self, text: String) -> Result<(), InteractiveModeError> {
        if text.trim().is_empty() {
            return Ok(());
        }
        match classify_submit(&text) {
            SubmitKind::Bash {
                command,
                exclude_from_context,
            } => {
                self.handle_bash_command(command, exclude_from_context)
                    .await
            }
            SubmitKind::Command(cmd) => {
                self.handle_inline_command(&cmd).await;
                Ok(())
            }
            SubmitKind::Message => {
                if self.runtime.session.is_run() {
                    // Submissions during streaming go through steer (when the v1 dispatch loop blocks inside prompt, inputs are queued,
                    // fallback for idle dispatch; this branch takes effect once Agent supports concurrency after)
                    self.runtime.session.steer(&text);
                    Ok(())
                } else {
                    self.push_footer();
                    self.runtime.session.prompt(&text).await.map(|_| ())?;
                    self.push_footer();
                    Ok(())
                }
            }
        }
    }

    /// Inline slash commands (v1 subset: /settings /compact /new /quit; integration seam for the T-AP-03-2 command table).
    async fn handle_inline_command(&mut self, cmd: &str) {
        match cmd.trim() {
            "/settings" => self.show_warning("Settings manager pending (T-AP-02-3)"),
            "/compact" => self.show_warning("Context compaction pending"),
            "/new" => self.show_warning("New session pending (T-AP-08-5)"),
            "/quit" => self.exit_requested = true,
            _ => self.show_warning(format!("Unknown command: {cmd}")),
        }
    }

    /// `!`/`!!` bash mode: spawn a bash session and render `bash_execution` rows.
    /// **Top open item**: bash execution engine (T-AP-04-2) not yet landed — this function currently falls back to an "unimplemented" notice,
    /// wire in here once T-AP-04-2 is done (output capture/truncation/cancellation/exit code backfill into `ChatRow::Bash`).
    async fn handle_bash_command(
        &mut self,
        command: String,
        exclude_from_context: bool,
    ) -> Result<(), InteractiveModeError> {
        let _ = exclude_from_context; // Flag to be used for exclude_from_context when T-AP-04-2 lands
        self.show_warning(format!("Bash execution pending (T-AP-04-2): {command}"));
        Ok(())
    }

    /// Status line shortcuts: `show_status`/`show_error`/`show_warning` three states
    /// (`UiEvent::Status(StatusSpec)`; Working carries a spinner; Error/Warning drawn directly with semantic colors).
    fn show_status(&mut self, message: impl Into<String>) {
        let _ = self.ui_event_tx.send(UiEvent::Status(StatusSpec::Working {
            message: message.into(),
        }));
    }

    /// Error status line (`StatusSpec::Error`).
    fn show_error(&mut self, message: impl Into<String>) {
        let _ = self
            .ui_event_tx
            .send(UiEvent::Status(StatusSpec::Error(message.into())));
    }

    /// Warning status line (`StatusSpec::Warning`).
    fn show_warning(&mut self, message: impl Into<String>) {
        let _ = self
            .ui_event_tx
            .send(UiEvent::Status(StatusSpec::Warning(message.into())));
    }
}

/// Interactive mode entry: invoked by main.rs when starting with `zen -i` — turns the already-assembled runtime and startup arguments
/// into a complete interactive session: startup banner → initial message → input dispatch loop → graceful exit code.
/// Interactive counterpart of the print mode entry (aligned with pi main.ts `new InteractiveMode(runtime, opts).run()`).
/// # Behavior
/// 1. Construct `InteractiveMode` (assemble options) and `run()`
/// 2. After `run()` returns, call `stop()` to finalize (idempotent; actual UI join/unsubscribe completes in `run()` shutdown)
/// 3. On error return Err (aggregated via anyhow); on normal/signal exit return exit code (0 = normal, 143/129 = shutdown signal)
/// # Consumers
/// `crates/zen-app/src/main.rs` (node 5 mode dispatch, replacing the current stub)
pub async fn run_interactive_mode(
    runtime: AgentSessionRuntime,
    initial_message: Option<String>,
    initial_images: Vec<ImageContent>,
    messages: Vec<String>,
) -> anyhow::Result<i32> {
    let options = InteractiveModeOptions {
        initial_message,
        initial_images,
        messages,
        verbose: false,
    };
    let mut mode = InteractiveMode::new(runtime, options);
    mode.run().await?;
    mode.stop()?;
    Ok(mode.signal_exit_code.load(Ordering::Relaxed))
}

/// Three destinations for submitted input (pure function, easy to unit test).
enum SubmitKind {
    /// `!`/`!!` bash commands (exclude_from_context = `!!` prefix).
    Bash {
        command: String,
        exclude_from_context: bool,
    },
    /// `/xxx` inline command raw text (including the leading `/`).
    Command(String),
    /// Normal message → session.prompt / steer.
    Message,
}

/// Classify submitted text (bash prefix takes priority over command/message; `!!` matches the double prefix first).
fn classify_submit(text: &str) -> SubmitKind {
    if let Some(rest) = text.strip_prefix("!!") {
        return SubmitKind::Bash {
            command: rest.trim().to_string(),
            exclude_from_context: true,
        };
    }
    if let Some(rest) = text.strip_prefix('!') {
        return SubmitKind::Bash {
            command: rest.trim().to_string(),
            exclude_from_context: false,
        };
    }
    if text.starts_with('/') {
        return SubmitKind::Command(text.to_string());
    }
    SubmitKind::Message
}

/// v1 default settings (aligned with pi: editorPaddingX defaults to 0; replace with real loading once T-AP-02-3 lands).
fn default_settings() -> SettingsConfig {
    SettingsConfig {
        auto_compact: false,
        steering_mode: DeliveryMode::OneAtATime,
        follow_up_mode: DeliveryMode::All,
        enable_skill_commands: true,
        show_hardware_cursor: false,
        editor_padding_x: 0,
        output_pad: 1,
        autocomplete_max_visible: 10,
        clear_on_shrink: false,
        hide_thinking_block: false,
        show_cache_miss_notices: true,
        quiet_startup: false,
        double_escape_action: DoubleEscapeAction::Fork,
        tree_filter_mode: FilterMode::Default,
        // thinking defaults to Off repo-wide (aligned with pi session default; settings only holds the panel's current value,
        // runtime authority is in AgentState—see T-AP-02-3 notes).
        thinking_level: ThinkingLevel::Off,
        available_thinking_levels: vec![
            ThinkingLevel::Off,
            ThinkingLevel::Low,
            ThinkingLevel::Medium,
        ],
        current_theme: "dark".into(),
        available_themes: vec!["dark".into(), "light".into()],
    }
}

/// Assemble the immutable context snapshot needed by the footer inside the subscription callback (model/thinking level/cwd do not change with the event stream during the session run;
/// v1 has no model switching; thinking level switching refreshes via the main task's `push_footer`).
#[derive(Clone)]
struct FooterCtx {
    /// Working directory.
    pwd: String,
    /// Model id.
    model_id: String,
    /// Thinking level.
    thinking_level: ThinkingLevel,
    /// Context window token count.
    context_window: u64,
    /// Auto-compaction switch (footer `(auto)` marker).
    auto_compact: bool,
}

impl FooterCtx {
    /// Built from session and settings snapshots.
    fn from_session(session: &AgentSession, settings: &SettingsConfig) -> Self {
        FooterCtx {
            pwd: session.cwd().to_string(),
            model_id: session.model().id.clone(),
            thinking_level: session.thinking_level(),
            context_window: session.model().context_window,
            auto_compact: settings.auto_compact,
        }
    }

    /// Assemble `FooterData`: context ratio = prompt tokens (input+cache_read+cache_write) / window;
    /// cache hit rate = cache_read / prompt tokens.
    fn footer(&self, usage: &UsageTotals) -> FooterData {
        let prompt_tokens = usage.input + usage.cache_read + usage.cache_write;
        let context_percent = if self.context_window > 0 {
            Some((prompt_tokens as f64 / self.context_window as f64) * 100.0)
        } else {
            None
        };
        let cache_hit_rate = if prompt_tokens > 0 {
            Some(usage.cache_read as f64 / prompt_tokens as f64 * 100.0)
        } else {
            None
        };
        FooterData {
            pwd: self.pwd.clone(),
            git_branch: None,   // After T-AP-06-3 lands, hook up git branch
            session_name: None, // Hook up session name after T-AP-08-5 lands
            model_id: self.model_id.clone(),
            thinking_level: Some(self.thinking_level),
            usage: *usage,
            context_percent,
            context_window: self.context_window,
            cache_hit_rate,
            auto_compact_enabled: self.auto_compact,
        }
    }
}

/// Agent event bridge: maps `AgentEvent` to `UiEvent` and delivers them to the UI task via `ui_event_tx`.
/// No awaiting inside the sync callback — `UnboundedSender` never blocks when full. Each event carries footer invalidation
/// (pi `footer.invalidate()`); `AgentStart` → Working status line + animation, `AgentEnd` → clear +
/// stop animation; `MessageEnd` error/abort → error text via `print_mode::stop_error_message`
/// Delivered with `MessageEnd.error_text` (rendered as an Info line on the UI side).
fn bridge_agent_event(
    event: &AgentEvent,
    tx: &mpsc::UnboundedSender<UiEvent>,
    usage: &Mutex<UsageTotals>,
    ctx: &FooterCtx,
) {
    match event {
        AgentEvent::AgentStart => {
            let _ = tx.send(UiEvent::Status(StatusSpec::Working {
                message: "Working...".into(),
            }));
            let _ = tx.send(UiEvent::SetAnimation(true));
        }
        AgentEvent::AgentEnd { .. } => {
            let _ = tx.send(UiEvent::ClearStatus);
            let _ = tx.send(UiEvent::SetAnimation(false));
        }
        AgentEvent::MessageStart { message } => {
            if !is_tool_result(message) {
                let _ = tx.send(UiEvent::MessageStart(message.clone()));
            }
        }
        AgentEvent::MessageUpdate { message, .. } => {
            if !is_tool_result(message) {
                let _ = tx.send(UiEvent::MessageUpdate(message.clone()));
            }
        }
        AgentEvent::MessageEnd { message } => {
            // Usage accumulation (assistant / toolResult each carry usage)
            match message {
                Message::Assistant(a) => {
                    if let Ok(mut u) = usage.lock() {
                        add_usage_to_totals(&mut u, &a.usage);
                    }
                }
                Message::ToolResult(t) => {
                    if let Some(u) = &t.usage {
                        if let Ok(mut totals) = usage.lock() {
                            add_usage_to_totals(&mut totals, u);
                        }
                    }
                }
                Message::User(_) => {}
            }
            let error_text = match message {
                Message::Assistant(a) => stop_error_message(a),
                _ => None,
            };
            if !is_tool_result(message) {
                let _ = tx.send(UiEvent::MessageEnd {
                    message: message.clone(),
                    error_text,
                });
            }
            // Footer invalidation per event (pi footer.invalidate())
            let totals = usage.lock().unwrap_or_else(|e| e.into_inner());
            let _ = tx.send(UiEvent::Footer(ctx.footer(&totals)));
        }
        AgentEvent::ToolExecutionStart {
            tool_call_id,
            tool_name,
            args,
        } => {
            let _ = tx.send(UiEvent::ToolStart {
                id: tool_call_id.clone(),
                name: tool_name.clone(),
                args: args.clone(),
            });
        }
        AgentEvent::ToolExecutionUpdate {
            tool_call_id, args, ..
        } => {
            let _ = tx.send(UiEvent::ToolUpdate {
                id: tool_call_id.clone(),
                args: args.clone(),
            });
        }
        AgentEvent::ToolExecutionEnd {
            tool_call_id,
            result,
            is_error,
            ..
        } => {
            let _ = tx.send(UiEvent::ToolEnd {
                id: tool_call_id.clone(),
                result: result.clone(),
                is_error: *is_error,
            });
        }
        AgentEvent::TurnStart | AgentEvent::TurnEnd { .. } => {}
    }
}

/// ToolResult messages do not go through the message line (rendered by Tool* events).
fn is_tool_result(message: &Message) -> bool {
    matches!(message, Message::ToolResult(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::agent_session::SessionStore;
    use std::collections::HashSet;
    use std::sync::Mutex as StdMutex;
    use zen_ai::tool::ToolExecutionMode;
    use zen_ai::types::{
        AssistantMessage, ContentBlock, StopReason, StreamFn, TextContent, ToolResultMessage, Usage,
    };
    use zen_core::harness::agent_hooks::AgentLoopHooks;
    use zen_core::harness::types::AgentState;

    // ── Runtime fixtures (aligned with session/runtime.rs test conventions, for convenient construction of InteractiveMode) ──

    #[derive(Default)]
    struct MockStore {
        session_id_val: String,
        cwd_val: String,
    }

    impl SessionStore for MockStore {
        fn append_message(&self, _message: &Message) {}
        fn append_model_change(&self, _provider: &str, _model_id: &str) {}
        fn append_thinking_level_change(&self, _level: &str) {}
        fn session_id(&self) -> String {
            self.session_id_val.clone()
        }
        fn cwd(&self) -> &str {
            &self.cwd_val
        }
    }

    struct MockHooks;

    #[async_trait::async_trait]
    impl AgentLoopHooks for MockHooks {
        async fn on_event(&self, _event: &AgentEvent) {}
    }

    fn test_model() -> zen_ai::types::Model {
        zen_ai::types::Model {
            id: "test-model".into(),
            name: None,
            provider: "test".into(),
            api: zen_ai::provider::Api::OpenAiCompletions,
            base_url: "http://localhost".into(),
            context_window: 8192,
            max_tokens: 1024,
            reasoning: false,
            headers: None,
            cost_rates: None,
        }
    }

    fn mock_stream_fn() -> StreamFn {
        Arc::new(
            |_model: &zen_ai::types::Model,
             _ctx: &zen_ai::types::Context,
             _cfg: Option<zen_ai::api::openai_stream::StreamConfig>,
             _cancel: Option<CancellationToken>| {
                Box::pin(async move {
                    let (_tx, rx) = tokio::sync::mpsc::unbounded_channel::<
                        zen_ai::types::AssistantMessageEvent,
                    >();
                    Ok::<_, zen_ai::api::openai_stream::StreamError>(rx)
                })
            },
        )
    }

    fn agent_state() -> AgentState {
        AgentState {
            system_prompt: "test".into(),
            model: test_model(),
            thinking_level: ThinkingLevel::Off,
            tools: vec![],
            messages: vec![],
            is_run: false,
            streaming_message: None,
            pending_tool_calls: HashSet::new(),
            error_message: None,
        }
    }

    fn test_agent() -> zen_core::harness::engine::Agent {
        zen_core::harness::engine::Agent::new(
            agent_state(),
            ToolExecutionMode::Sequential,
            Arc::new(MockHooks),
            mock_stream_fn(),
            CancellationToken::new(),
        )
    }

    fn test_runtime() -> AgentSessionRuntime {
        let store: Arc<dyn SessionStore> = Arc::new(MockStore {
            session_id_val: "s1".into(),
            cwd_val: "/work".into(),
        });
        let session = AgentSession::new(test_agent(), Arc::clone(&store));
        let services = crate::session::runtime::AgentSessionServices {
            cwd: "/work".into(),
            agent_dir: "/agent".into(),
            session_store: store,
            registry: zen_core::process::ProcessRegistry::new(),
        };
        crate::session::runtime::create_initial_runtime(
            session,
            services,
            crate::session::runtime::SessionStartReason::Startup,
        )
    }

    // ── classify_submit ─────────────────────────────────────────

    #[test]
    fn test_classify_submit_message() {
        assert!(matches!(classify_submit("hello"), SubmitKind::Message));
        assert!(matches!(classify_submit(" normal  "), SubmitKind::Message));
    }

    #[test]
    fn test_classify_submit_bash_prefixes() {
        assert!(matches!(
            classify_submit("!ls -la"),
            SubmitKind::Bash { command, exclude_from_context: false } if command == "ls -la"
        ));
        assert!(matches!(
            classify_submit("!!secret"),
            SubmitKind::Bash { command, exclude_from_context: true } if command == "secret"
        ));
    }

    #[test]
    fn test_classify_submit_command() {
        assert!(matches!(
            classify_submit("/quit"),
            SubmitKind::Command(c) if c == "/quit"
        ));
    }

    // ── default_settings / FooterCtx ─────────────────────────────

    #[test]
    fn test_default_settings_sane_baseline() {
        let s = default_settings();
        assert_eq!(s.editor_padding_x, 0);
        assert!(!s.auto_compact);
        assert!(!s.available_thinking_levels.is_empty());
    }

    #[test]
    fn test_footer_ctx_computes_percent_and_cache_hit() {
        let usage = UsageTotals {
            input: 100,
            output: 50,
            cache_read: 900,
            cache_write: 0,
            cost: 0.0,
        };
        let ctx = FooterCtx {
            pwd: "/work".into(),
            model_id: "m".into(),
            thinking_level: ThinkingLevel::Off,
            context_window: 2000,
            auto_compact: true,
        };
        let data = ctx.footer(&usage);
        // prompt = 1000/2000 = 50%；cache hit = 900/1000 = 90%
        assert!((data.context_percent.unwrap() - 50.0).abs() < 0.01);
        assert!((data.cache_hit_rate.unwrap() - 90.0).abs() < 0.01);
        assert!(data.auto_compact_enabled);
        assert_eq!(data.model_id, "m");
    }

    // ── bridge_agent_event (events → UiEvent channel) ───────────────────

    fn channel() -> (
        mpsc::UnboundedSender<UiEvent>,
        mpsc::UnboundedReceiver<UiEvent>,
    ) {
        mpsc::unbounded_channel()
    }

    fn assistant_with_usage(msg: &str, usage: Usage) -> AssistantMessage {
        AssistantMessage {
            content: vec![ContentBlock::Text(TextContent {
                text: msg.into(),
                text_signature: None,
            })],
            api: "openai-completions".into(),
            provider: "test".into(),
            model: "m".into(),
            response_model: None,
            response_id: None,
            usage,
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 0,
        }
    }

    fn mk_usage(i: u64, o: u64, cr: u64) -> Usage {
        Usage {
            input: i,
            output: o,
            cache_read: cr,
            cache_write: 0,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: i + o + cr,
            cost: zen_ai::types::Cost::default(),
        }
    }

    fn test_footer_ctx() -> FooterCtx {
        FooterCtx {
            pwd: "/work".into(),
            model_id: "m".into(),
            thinking_level: ThinkingLevel::Off,
            context_window: 8192,
            auto_compact: false,
        }
    }

    #[test]
    fn test_bridge_agent_start_sets_working_and_animation() {
        let (tx, mut rx) = channel();
        let usage = StdMutex::new(UsageTotals::default());
        bridge_agent_event(&AgentEvent::AgentStart, &tx, &usage, &test_footer_ctx());
        assert!(matches!(
            rx.try_recv(),
            Ok(UiEvent::Status(StatusSpec::Working { .. }))
        ));
        assert!(matches!(rx.try_recv(), Ok(UiEvent::SetAnimation(true))));
    }

    #[test]
    fn test_bridge_message_end_accumulates_usage_and_pushes_footer() {
        let (tx, mut rx) = channel();
        let usage = StdMutex::new(UsageTotals::default());
        bridge_agent_event(
            &AgentEvent::MessageEnd {
                message: Message::Assistant(assistant_with_usage("hi", mk_usage(100, 20, 30))),
            },
            &tx,
            &usage,
            &test_footer_ctx(),
        );
        // MessageEnd delivered
        assert!(matches!(rx.try_recv(), Ok(UiEvent::MessageEnd { .. })));
        // Usage accumulation + carried by footer
        let totals = usage.lock().unwrap();
        assert_eq!(totals.input, 100);
        assert_eq!(totals.output, 20);
        assert_eq!(totals.cache_read, 30);
        assert!(matches!(rx.try_recv(), Ok(UiEvent::Footer(_))));
    }

    #[test]
    fn test_bridge_error_message_end_attaches_error_text() {
        let (tx, mut rx) = channel();
        let usage = StdMutex::new(UsageTotals::default());
        let mut a = assistant_with_usage("x", Usage::default());
        a.stop_reason = StopReason::Error;
        a.error_message = Some("provider 500".into());
        bridge_agent_event(
            &AgentEvent::MessageEnd {
                message: Message::Assistant(a),
            },
            &tx,
            &usage,
            &test_footer_ctx(),
        );
        assert!(matches!(
            rx.try_recv(),
            Ok(UiEvent::MessageEnd { error_text: Some(e), .. }) if e == "provider 500"
        ));
    }

    #[test]
    fn test_bridge_tool_events_map_to_ui_events() {
        let (tx, mut rx) = channel();
        let usage = StdMutex::new(UsageTotals::default());
        bridge_agent_event(
            &AgentEvent::ToolExecutionStart {
                tool_call_id: "t1".into(),
                tool_name: "bash".into(),
                args: serde_json::json!({"command": "ls"}),
            },
            &tx,
            &usage,
            &test_footer_ctx(),
        );
        assert!(matches!(
            rx.try_recv(),
            Ok(UiEvent::ToolStart { id, .. }) if id == "t1"
        ));

        let harness_result = ToolExecutionResult {
            message: ToolResultMessage {
                tool_call_id: "t1".into(),
                tool_name: "bash".into(),
                content: vec![ContentBlock::Text(TextContent {
                    text: "ok".into(),
                    text_signature: None,
                })],
                details: None,
                usage: None,
                is_error: false,
                added_tool_names: None,
                timestamp: 0,
            },
            terminate: false,
        };
        bridge_agent_event(
            &AgentEvent::ToolExecutionEnd {
                tool_call_id: "t1".into(),
                tool_name: "bash".into(),
                result: harness_result,
                is_error: false,
            },
            &tx,
            &usage,
            &test_footer_ctx(),
        );
        assert!(matches!(
            rx.try_recv(),
            Ok(UiEvent::ToolEnd { id, is_error, .. }) if id == "t1" && !is_error
        ));
    }

    #[test]
    fn test_bridge_tool_result_message_skipped_for_chat_rows() {
        let (tx, mut rx) = channel();
        let usage = StdMutex::new(UsageTotals::default());
        bridge_agent_event(
            &AgentEvent::MessageStart {
                message: Message::ToolResult(ToolResultMessage {
                    tool_call_id: "t1".into(),
                    tool_name: "bash".into(),
                    content: vec![],
                    details: None,
                    usage: None,
                    is_error: false,
                    added_tool_names: None,
                    timestamp: 0,
                }),
            },
            &tx,
            &usage,
            &test_footer_ctx(),
        );
        // ToolResult does not go through message lines (no MessageStart event)
        assert!(rx.try_recv().is_err());
    }

    // ── InteractiveMode::new ─────────────────────────────────────

    fn options() -> InteractiveModeOptions {
        InteractiveModeOptions {
            initial_message: None,
            initial_images: vec![],
            messages: vec![],
            verbose: false,
        }
    }

    #[test]
    fn test_new_initializes_channels_and_guards() {
        let mut mode = InteractiveMode::new(test_runtime(), options());
        assert!(!mode.is_initialized);
        assert!(!mode.is_shutting_down);
        assert!(mode.input_tx.is_some(), "channel sender pending handoff at init");
        assert!(mode.ui_event_rx.is_some());
        assert!(mode.ui_task.is_none());
        assert!(mode.subscription.is_none());
        // channel is send/receive capable (send side on input_tx, receive side on input_rx)
        mode.input_tx
            .as_ref()
            .unwrap()
            .send(InputCommand::Submit("hi".into()))
            .unwrap();
        assert!(matches!(
            mode.input_rx.try_recv(),
            Ok(InputCommand::Submit(t)) if t == "hi"
        ));
        // cancel token initially not canceled
        assert!(!mode.cancel.is_cancelled());
    }
}
