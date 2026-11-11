use crate::metrics::{CountingWriter, FrameProbe, FrameStats, DEFAULT_BUDGET};
use crate::screen::ScreenBuffer;
use crate::terminal::{Terminal, TerminalError};
use crate::viewport::ScrollbackCanvas;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

/// Default animation cadence while the animation tick is armed: 80 ms (aligned with pi loader `DEFAULT_INTERVAL_MS`).
///
/// Single source of truth: in pi the loader's frame-derivation period and its `requestRender` cadence are the same knob; zen hands
/// the pacing to the loop, so the two places must reference this constant or they drift (the spinner would appear to stall or skip).
/// Consumers: [`TuiLoop::set_animation`]'s default arming period (zen-app maps `AgentStart` → `Animate(Some(this))` via
/// `modes::interactive::animation_wake_intent`), [`crate::components::Loader`]'s default interval.
pub const DEFAULT_ANIMATION_INTERVAL: Duration = Duration::from_millis(80);

/// Floor for an animation period (a zero period would make the deadline "always due" = busy loop).
const MIN_ANIMATION_INTERVAL: Duration = Duration::from_millis(1);

/// TUI main loop orchestrator. Owns the output canvas (Terminal + ScreenBuffer) and render state (request merge flag, animation
/// clock, throttle timestamp, lifecycle flags).
/// External contract aligned with zen TUI's caller perspective:
/// Application calls start() to enter raw mode → run()/run_scrollback() drive the event loop → stop() restores the terminal.
/// **Repaint policy (PF-03, aligned with pi `requestRender`/`scheduleRender`)**: there is no free-running ticker — a frame is painted
/// only when something asked for it (terminal event, external wake, or an armed animation deadline). With nothing armed the loop does
/// not wake at all, so "model is thinking, nothing on screen changes" costs zero frames instead of 8 fps.
/// Events are forwarded as-is via the on_event callback (key parsing semantics belong to T-TU-03);
/// Redraws write the current frame to the buffer via the draw callback; the loop performs a unified flush of differential output.
/// The loop handles Resize itself (buffer.resize + redraw request).
pub struct TuiLoop {
    terminal: Terminal,                // Output sink; start/stop enter/exit raw mode
    buffer: ScreenBuffer,              // Current frame buffer; resize/flush inside run
    render_requested: bool, // Render merge flag: set by request_render() from loop callbacks, cleared by finish_frame()
    animation: Option<AnimationClock>, // Armed animation cadence (Some = periodic frames, None = no animation); replaces the former `animation_enabled` bool
    last_render_at: Instant, // Completion time of the last painted frame; the throttling baseline for min_render_interval
    min_render_interval: Duration, // Minimum gap between two painted frames (merge floor), default 16ms (pi MIN_RENDER_INTERVAL_MS)
    started: bool,                 // Set after start(); used for precondition checks before run()
    stopped: bool,                 // Lifecycle termination flag
    wake_stream: Option<Pin<Box<dyn futures::Stream<Item = WakeOutcome> + Send>>>, // /// Fourth external wake path (T-AP-11-1 patch): the "message → WakeOutcome" stream registered at assembly; None if unregistered. Items are already erased to `WakeOutcome`, so TuiLoop needs no generics.
    stats: Arc<FrameStats>, // PF-01: per-frame metrics accumulator (records nothing unless ZEN_TUI_TRACE_FRAMES is set); shared with the app layer for the exit summary.
}

pub enum EventOutcome {
    Continue,
    Stop,
}

#[derive(Debug, thiserror::Error)]
pub enum TuiError {
    #[error("terminal error: {0}")]
    Terminal(#[from] TerminalError),
    #[error("failed to read terminal event: {0}")]
    EventStream(std::io::Error),
    #[error("TuiLoop::run() called before start()")]
    NotStarted,
}

/// Rendering intent declared by an external wake: `Render` = request one frame; `Animate(Some(interval))` = arm the animation
/// cadence, `Animate(None)` = disarm it.
///
/// The payload became an `Option<Duration>` in PF-03: the repetition rate belongs to the intent (pi's loader owns
/// `DEFAULT_INTERVAL_MS`), so a wake source that animates at a non-default rate can say so instead of hard-coding 16 ms inside the loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeOutcome {
    /// Repaint once as needed after an external event (merged into the throttle floor).
    Render,
    /// Arm (`Some(period)`) or disarm (`None`) the animation cadence — the zen counterpart of pi's loader
    /// `setInterval`/`clearInterval`. Arming also requests a frame, mirroring pi's «loader.start → updateDisplay → requestRender»,
    /// so the spinner does not wait a whole period for its first frame.
    /// Periods shorter than the merge floor are raised to it (see [`next_frame_deadline`]).
    Animate(Option<Duration>),
}

/// Animation cadence owned by the loop: an armed period plus the next-due instant.
///
/// Wrapping both facts in one `Option` (the field is `Option<AnimationClock>`) makes "armed implies a deadline" a type invariant;
/// the former `animation_enabled: bool` could describe an inconsistent state (armed without a deadline) and needed the free-running
/// tick to notice it was armed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AnimationClock {
    interval: Duration, // Period between two animation frames
    next_at: Instant,   // Instant at which the next frame becomes due
}

impl AnimationClock {
    /// Arm the clock: the first deadline is `now + interval`. A zero period is clamped to 1 ms — a zero interval is always due,
    /// which would spin the loop.
    fn armed(interval: Duration, now: Instant) -> AnimationClock {
        let interval = interval.max(MIN_ANIMATION_INTERVAL);
        AnimationClock {
            interval,
            next_at: now + interval,
        }
    }

    /// Re-anchor by one period: the frame just finished painting at `now`, so the next one is due at `now + interval`.
    ///
    /// This is a re-anchor, **not** an accumulation (intentional deviation from pi's fixed-phase `setInterval`): slow frames must
    /// not queue up catch-up frames. The period therefore stays `frame cost + interval` (124 ms cost ⇒ 204 ms cadence), whereas a
    /// fixed phase would degenerate into back-to-back rendering (124 ms ⇒ 100% CPU).
    fn advance_to(&mut self, now: Instant) {
        self.next_at = now + self.interval;
    }

    /// Next-due instant (read by [`TuiLoop::frame_deadline`]).
    fn deadline(&self) -> Instant {
        self.next_at
    }
}

impl TuiLoop {
    /// Create the main loop at the current terminal size (Terminal::new + ScreenBuffer::new(cols, rows)); does not enter raw mode.
    /// Consumer: T-AP-11-1 (interactive main entry).
    pub fn new() -> Result<TuiLoop, TuiError> {
        let terminal = Terminal::new()?;
        let (cols, rows) = crossterm::terminal::size().unwrap_or((80, 24));
        let buffer = ScreenBuffer::new(cols, rows);

        Ok(TuiLoop {
            terminal,
            buffer,
            render_requested: false,
            animation: None,
            last_render_at: Instant::now(),
            min_render_interval: Duration::from_millis(16), // Aligned with pi MIN_RENDER_INTERVAL_MS
            started: false,
            stopped: false,
            wake_stream: None,
            stats: Arc::new(FrameStats::new(DEFAULT_BUDGET)),
        })
    }

    /// Frame metrics accumulated by this loop (shared `Arc`). Read once by the application layer on exit to
    /// print the summary; nothing is recorded unless `ZEN_TUI_TRACE_FRAMES` was set at construction.
    ///
    /// **Do not call this inside the frame closures**: they only need `&this.stats` (a `Deref` borrow) — going
    /// through this accessor there would hand out a temporary `Arc` that drops while still borrowed.
    ///
    /// Consumer: `zen-app` `modes/interactive/mod.rs` (after `tui.stop()`).
    pub fn frame_stats(&self) -> Arc<FrameStats> {
        Arc::clone(&self.stats)
    }

    /// Enter raw mode and hide the cursor (Terminal::start), request first-frame render.
    /// Idempotent; repeated calls have no side effects. Bracketed paste/keyboard enhancement protocols (EnableBracketedPaste /
    /// PushKeyboardEnhancementFlags) are already enabled along with raw mode in [`Terminal::start`],
    /// and symmetrically disabled in [`Terminal::stop`] (T-AP-11-1-fix closed the loop; the former "open issue" is resolved).
    pub fn start(&mut self) -> Result<(), TuiError> {
        if self.started {
            return Ok(());
        }
        self.terminal.start()?;
        self.started = true;
        self.request_render(); // First frame: draw callback renders the initial screen
        Ok(())
    }

    /// Exits interaction and hands the terminal back to the shell: clear residue + restore raw mode/cursor/protocol (Terminal::stop). Idempotent.
    /// Also for apps temporarily exiting raw mode while running (e.g. suspending the TUI during T-AP-11-4 external editor integration, after `stop()`
    /// launch the editor, then `start()` to restore).
    /// Handover step (terminal-exit-handover-fix): first clear the cursor line via [`Terminal::handover_to_shell`]
    /// (input box) and fixed area below the cursor (footer/status line), move the cursor to a new line at column 0, then restore raw mode
    /// — align with pi `Tui.stop()`'s "last content line + \r\n" (reference/pi-original/packages/tui/src/tui.ts).
    /// Without this step, the shell takes over from the leftover cursor column; the prompt and later output are misaligned (stretched lines/extra spaces).
    /// Consumer: T-AP-11-1.
    pub fn stop(&mut self) -> Result<(), TuiError> {
        if self.stopped {
            return Ok(());
        }
        // Handover write failure (e.g. terminal dead EIO) does not block restoration — align with pi exit path ignoring write errors.
        let _ = self.terminal.handover_to_shell();
        self.terminal.stop()?;
        self.started = false;
        self.stopped = true;
        Ok(())
    }

    /// Request one frame (set the merge flag).
    ///
    /// Only valid from inside a loop callback (`on_event` / `on_wake` / `frame`): the frame deadline is computed before `select!`,
    /// so setting this flag from another task or thread neither wakes the loop nor recomputes the deadline — such a request is
    /// silently dropped. (Before PF-03 it only worked because the free-running 16 ms tick happened to read the flag.)
    /// Cross-task triggers must go through [`TuiLoop::set_external_wakeup`] or the cancellation token.
    /// Multiple requests within `min_render_interval` merge into one frame.
    pub fn request_render(&mut self) {
        self.render_requested = true
    }

    /// Arm (`Some(period)`) or disarm (`None`) the animation cadence from code; arming also requests the first frame.
    ///
    /// Same semantics as [`WakeOutcome::Animate`]; it is the loop's programming surface for hosts that drive it without a wake
    /// channel (zen-app itself goes through the wake channel). Re-arming re-anchors the clock (pi `setIndicator → start()`: frame
    /// sequence and cadence restart); a zero period is clamped to 1 ms so the deadline can never be "always due".
    pub fn set_animation(&mut self, interval: Option<Duration>) {
        self.set_animation_at(interval, Instant::now())
    }

    /// [`TuiLoop::set_animation`] with an injected clock reading (unit tests; production always goes through the wrapper).
    fn set_animation_at(&mut self, interval: Option<Duration>, now: Instant) {
        self.animation = interval.map(|period| AnimationClock::armed(period, now));
        if self.animation.is_some() {
            // Arming is a visible state change: paint it now instead of waiting a whole period (pi: loader.start → requestRender).
            self.request_render();
        }
    }

    /// Set the minimum gap between two painted frames (default 16ms, aligned with pi `MIN_RENDER_INTERVAL_MS`).
    ///
    /// Semantics changed in PF-03: this is no longer "the tick period" (there is no ticker) but the merge floor for
    /// request-driven frames, measured from the previous frame's **completion** time. It is unrelated to [`DEFAULT_BUDGET`]
    /// (the per-frame cost budget) — do not conflate the two.
    pub fn set_min_render_interval(&mut self, interval: Duration) {
        self.min_render_interval = interval
    }

    /// TUI main event loop (cell-level diff mode): multiplexes four async sources: polled event stream / conditional frame timer /
    /// cancellation signal / external wakeup.
    /// Prerequisite: [`TuiLoop::start`] must be called first; see `run_loop` for the exit path.
    /// draw renders the current frame into `ScreenBuffer` (`run_loop` calls `render_frame` each frame to output diffs).
    /// Default path for interactive mode; use [`TuiLoop::run_scrollback`] for scrollback mode.
    pub async fn run<E, D>(
        &mut self,
        cancel: &CancellationToken,
        on_event: E,
        draw: D,
    ) -> Result<(), TuiError>
    where
        E: FnMut(crossterm::event::Event) -> EventOutcome,
        D: FnMut(&mut ScreenBuffer),
    {
        let mut draw = draw;
        self.run_loop(
            cancel,
            on_event,
            // Cell-level diff frame: render into buffer → flush emits cell-by-cell diffs (render_frame semantics remain unchanged).
            |this| this.render_frame(&mut draw),
            // In cell-level diff mode, Resize is handled by ScreenBuffer::resize (reset the diff baseline → full-screen redraw).
            |this, cols, rows| this.buffer.resize(cols, rows),
        )
        .await
    }

    /// Drives the TUI main loop in scrollback rendering mode: uses the same four-way select as [`TuiLoop::run`] (events/conditional
    /// frame timer/cancel/external wakeup), but the draw contract changes from 「render into the ScreenBuffer grid」 to 「push content rows into
    /// `ScrollbackCanvas` — change per-frame diff output to line-level diffing + real terminal scrolling when content overflows the screen (write
    /// scrollback), allowing chat history to be viewed via native terminal scrolling. Coexists with `run` and does not affect it by default.
    /// # Behavior
    /// 1. Use the same pre-validation as `run` (call `start()` first; return NotStarted for non-tty)
    /// 2. Reuse the private `run_loop` extracted from `run` (event forwarding/merge-bit/Resize semantics remain unchanged)
    /// 3. Per frame: `canvas.clear()` → `draw(&mut canvas)` → `canvas.flush` (line-level diffing + advancement + terminal write);
    ///    On Resize, synchronize the canvas size within the frame and trigger a full redraw
    /// # Consumers
    /// zen-app `modes/interactive/mod.rs` (T-AP-11-1 init; wires up scrollback mode)
    pub async fn run_scrollback<E, D>(
        &mut self,
        cancel: &CancellationToken,
        on_event: E,
        draw: D,
    ) -> Result<(), TuiError>
    where
        E: FnMut(crossterm::event::Event) -> EventOutcome,
        D: FnMut(&mut ScrollbackCanvas),
    {
        let mut draw = draw;
        let (cols, rows) = (self.terminal.columns(), self.terminal.rows());
        let mut canvas = ScrollbackCanvas::new(cols, rows);
        self.run_loop(
            cancel,
            on_event,
            // scrollback frame: synchronize the terminal size first each frame (if changed → reset diff baseline → full redraw),
            // Then clear the current frame's line stream, draw, and flush line-level diffs to the terminal.
            // PF-01: the probe brackets clear+draw (draw cost) and the flush (flush cost); the size sync above stays
            // outside the measurement because it is geometry bookkeeping, not rendering.
            |this| {
                let (cols, rows) = (this.terminal.columns(), this.terminal.rows());
                canvas.set_size(cols, rows);
                // PF-06: open the frame (clears the frame buffer) and remember the delivery-counter baseline so
                // `FrameMetrics::writes` reports this frame's deliveries, not a cumulative total.
                this.terminal.begin_frame();
                let writes_before = this.terminal.write_calls();
                let mut probe = FrameProbe::start(&this.stats);
                canvas.clear();
                (draw)(&mut canvas);
                probe.mark_draw();
                // PF-05: `flush` promotes the stream with `mem::swap` and leaves `lines` empty, so `content_height()` must be read
                // **before** the flush (otherwise `FrameMetrics::line_total` would be 0 every frame and PF-01's `written_fixed`
                // derivation would collapse the fixed-region interval to the empty set).
                let line_total = canvas.content_height();
                let (result, bytes) = {
                    let mut out = CountingWriter::new(this.terminal.out_mut());
                    let result = canvas.flush(&mut out).map_err(TuiError::from);
                    (result, out.bytes())
                };
                // PF-06: the frame's only delivery point — one `write_all` for everything the canvas just produced
                // (line diff + `?2026h/l` + IME cursor positioning). It must run **before** `probe.finish`, whose
                // `flush` metric is `elapsed - draw` frozen at that moment; a later delivery would not be measured.
                let delivered = this.terminal.flush_out().map_err(TuiError::from);
                probe.finish(
                    bytes,
                    canvas.last_repaint_span(),
                    line_total,
                    canvas.fixed_rows(),
                    (this.terminal.write_calls() - writes_before) as usize,
                );
                result?;
                delivered?;
                Ok(())
            },
            // scrollback mode handles sizing/full redraw within the frame (canvas size sync); this callback is a no-op.
            |_, _, _| {},
        )
        .await
    }

    /// Shared skeleton for the TUI main event loop: four asynchronous sources (event stream polling / conditional frame timer / cancellation
    /// signal / external wakeup),
    /// Drives `frame` (per-frame rendering) and `on_resize` (size changes). `run`/`run_scrollback` only provide different
    /// frame and resize callbacks; event forwarding / merge flag / throttle floor semantics are identical in both modes.
    /// # Prerequisites
    /// [`TuiLoop::start`] must be called first; precondition checks occur before any IO (EventStream/sleep), so no real terminal is required.
    /// # Exit paths
    /// - `on_event` returns `EventOutcome::Stop`;
    /// - The event stream ends (stdin is closed);
    /// - Cancellation via `cancel`;
    /// - The external wakeup channel is disconnected.
    ///
    /// Unified return Ok; error paths (terminal IO / underlying event stream errors) return Err.
    async fn run_loop<E, F, R>(
        &mut self,
        cancel: &CancellationToken,
        mut on_event: E,
        mut frame: F,
        mut on_resize: R,
    ) -> Result<(), TuiError>
    where
        E: FnMut(crossterm::event::Event) -> EventOutcome,
        F: FnMut(&mut Self) -> Result<(), TuiError>,
        R: FnMut(&mut Self, u16, u16),
    {
        // ── Precondition checks: start() must have entered raw mode first (cursor already hidden),
        //    Otherwise event loop / rendering is meaningless — reject execution outright ──
        if !self.started {
            return Err(TuiError::NotStarted);
        }

        // Event source: crossterm's async event stream. Internally spawns a thread polling stdin,
        // parsing terminal input into structured Event values (KeyEvent/Resize/Paste/Focus…) and yielding them one by one
        use crossterm::event::EventStream;
        // Importing the StreamExt trait is required to call .next() on EventStream for the next event
        use futures::StreamExt;

        let mut event_stream = EventStream::new();

        // Frame pacing (PF-03): there is no free-running ticker. Each iteration derives the next frame deadline from the current
        // state (pending request / armed animation clock, `None` = nothing to paint) and waits for exactly that instant; while
        // nothing is armed the timer branch stays pending, so an idle loop costs zero wakeups instead of 62.5 per second.

        // ── First-frame render: draw one frame before entering the loop to avoid a blank screen ──
        // start() already set the merge flag via request_render(); honor it here directly
        frame(&mut *self)?; // draw writes to the buffer → flush emits the diff; on failure, loop startup fails entirely
        self.finish_frame(); // First frame painted: clear the merge flag + stamp the completion time (and re-anchor any animation)

        loop {
            // Computed before `select!` so the immutable borrow ends here (the branches below need `&mut self`).
            let deadline = self.frame_deadline();
            // ── Monitor four asynchronous sources concurrently: the first ready source runs (tokio uses random fairness, with no source favored)──
            tokio::select! {
                // ① Event stream: user input arrives (key press, mouse, paste, window resize, etc.)
                maybe_event = event_stream.next() => {
                    match maybe_event {
                        Some(Ok(event)) => {
                            // Resize is a "window size changed" event; the loop must handle it itself:
                            // Pass to on_resize (cell-level → buffer.resize; scrollback → synchronize canvas dimensions)
                            if let crossterm::event::Event::Resize(cols, rows) = event {
                                on_resize(&mut *self, cols, rows);
                                self.request_render(); // Size changed → must redraw
                            }

                            // Other events (and Resize itself) are passed to the application unchanged;
                            // Key semantics (shortcut mapping etc., part of T-TU-03) are resolved in on_event
                            match on_event(event) {
                                EventOutcome::Stop => break,
                                // Event handling may have mutated application state → set the merge flag; the next iteration
                                // recomputes the deadline from it (aligns with pi handleInput → requestRender contract)
                                EventOutcome::Continue => self.request_render(),
                            }
                        }
                        Some(Err(e)) => return Err(TuiError::EventStream(e)), // underlying IO error in the event stream, terminate the loop
                        None => break, // event stream ended (stdin closed), exit normally
                    }
                }

                // (2) conditional frame timer: fires only at the instant derived from the current state (nothing armed ⇒ never).
                //     Render coalescing: high-frequency events do not produce one frame per event — they all collapse into this one
                //     frame, which never comes sooner than `min_render_interval` after the previous frame finished.
                _ = sleep_until_opt(deadline) => {
                    frame(&mut *self)?; // Render the current frame → flush the diff output
                    self.finish_frame(); // clear the merge flag, stamp the time, re-anchor the animation cadence
                }

                // (3) cancellation signal: triggered when the caller invokes cancel.cancel()
                _ = cancel.cancelled() => {
                    break; // graceful exit: only break the loop; do not restore the terminal (restoring is the caller stop()'s responsibility)
                }
                // ④ External wake (T-AP-11-1 patch): UI changes pushed by the main task during streaming (no key input).
                   //    When unregistered it stays pending forever (never sleeps this path), so the existing three paths are unaffected.
                   maybe_wake = self.next_wake() => {
                       match maybe_wake {
                           Some(outcome) => self.apply_wake_outcome(outcome),
                           None => break, // Wake channel disconnected: coordinator ends, exits along with cancellation
                       }
                   }
            }
        }

        // loop exit (Stop / stream end / cancel) uniformly returns Ok; error paths already returned Err above
        Ok(())
    }

    /// Internal: perform one render — call `draw` to render the current frame into the buffer, then `buffer.flush(terminal.out_mut())`
    /// differentially output to the terminal. Private, called only by `run`.
    ///
    /// PF-01: measured like the scrollback frame, except that the cell-level path has no line stream — it reports
    /// `repaint_span: None`, `line_total` = buffer rows and `line_fixed` = 0 (the fixed region only exists in
    /// scrollback mode).
    fn render_frame<D>(&mut self, draw: &mut D) -> Result<(), TuiError>
    where
        D: FnMut(&mut ScreenBuffer),
    {
        // PF-06: open the frame (clears the frame buffer) + record the delivery-counter baseline.
        self.terminal.begin_frame();
        let writes_before = self.terminal.write_calls();
        let mut probe = FrameProbe::start(&self.stats);
        // Start each frame from a clean frame: cells not redrawn are judged as "content→empty" in the subsequent flush diff and erased,
        // Components need not self-clear (the zen-grid-level counterpart of pi composite layer clear-extra-lines);
        // Identical cells across two frames are still not written after diff, so differential efficiency is unaffected.
        self.buffer.clear();
        (draw)(&mut self.buffer);
        probe.mark_draw();
        let (result, bytes) = {
            let mut out = CountingWriter::new(self.terminal.out_mut());
            let result = self.buffer.flush(&mut out);
            (result, out.bytes())
        };
        // PF-06: single delivery point, before `probe.finish` (see `run_scrollback` for the ordering rationale).
        let delivered = self.terminal.flush_out();
        let (_, rows) = self.buffer.size();
        probe.finish(
            bytes,
            None,
            rows as usize,
            0,
            (self.terminal.write_calls() - writes_before) as usize,
        );
        result?;
        delivered?;
        Ok(())
    }

    /// Fourth external wake channel (T-AP-11-1 patch): during streaming (no keypresses), the interactive coordinator must also deliver
    /// agent events to the render layer — register a generic channel; when a message arrives inside the run() loop, call `on_wake`
    /// (synchronous callback, must not await; generally used to apply changes + declare redraw intent via WakeOutcome).
    /// R is specified by the caller at the assembly site (zen-app uses `UiEvent`); zen-tui is unaware of the concrete type.
    /// # Consumers
    /// zen-app `modes/interactive/ui.rs` (UI task assembly).
    pub fn set_external_wakeup<R: Send + 'static>(
        &mut self,
        rx: tokio::sync::mpsc::UnboundedReceiver<R>,
        on_wake: Box<dyn FnMut(&R) -> WakeOutcome + Send>,
    ) {
        // Type erasure: fold (receiver + synchronous callback) into `Stream<Item = WakeOutcome>`.
        // One unfold iteration per message: recv().await → on_wake(&msg) (synchronous, must-not-await
        // contract, guaranteed by the callback itself) → yield render intent; receiver/callback carried into the next round via state.
        self.wake_stream = Some(Box::pin(futures::stream::unfold(
            (rx, on_wake),
            |(mut rx, mut on_wake)| async move {
                let msg = rx.recv().await?; // Channel closed (all Senders dropped) → None → stream ends
                Some(((on_wake)(&msg), (rx, on_wake)))
            },
        )));
    }

    /// Take the next wake intent; never ready when the fourth channel is unregistered (equivalent to infinite pending).
    async fn next_wake(&mut self) -> Option<WakeOutcome> {
        use futures::StreamExt;
        let Some(stream) = &mut self.wake_stream else {
            return std::future::pending::<Option<WakeOutcome>>().await;
        };
        stream.next().await
    }

    /// Set render flags per wake intent: Render → merge into one frame; Animate(v) → arm/disarm the animation cadence.
    fn apply_wake_outcome(&mut self, outcome: WakeOutcome) {
        match outcome {
            WakeOutcome::Render => self.request_render(),
            WakeOutcome::Animate(interval) => self.set_animation(interval),
        }
    }

    /// Next instant at which a frame must be painted; `None` = nothing armed (the loop stays asleep this round).
    ///
    /// Instance wrapper of [`next_frame_deadline`]: pulls the four inputs out of the fields so the decision itself stays a pure
    /// function that unit tests can drive with injected instants.
    fn frame_deadline(&self) -> Option<Instant> {
        next_frame_deadline(
            self.last_render_at,
            self.min_render_interval,
            self.render_requested,
            self.animation.map(|clock| clock.deadline()),
        )
    }

    /// Book-keeping after a frame was painted: clear the merge flag, stamp the completion time, re-anchor the animation clock.
    fn finish_frame(&mut self) {
        self.finish_frame_at(Instant::now())
    }

    /// [`TuiLoop::finish_frame`] with an injected clock reading (unit tests).
    ///
    /// Three steps in this order: clear the merge flag → `last_render_at = now` (the **completion** anchor, deliberately different
    /// from pi's pre-render anchor: the cadence becomes `frame cost + interval` and slow frames never compound) → advance the
    /// animation clock to `now + interval` (re-anchor; never catch up on missed periods).
    ///
    /// Invariant asserted by the tests (exactly this, never `> Instant::now()` — a slow CI pause would fake it): after this call an
    /// armed animation is due exactly at `now + interval`.
    ///
    /// ⚠️ PF-08 interface note: clearing **after** drawing is currently equivalent to pi's "clear first, draw, re-schedule",
    /// because `frame(&mut *self)` holds the exclusive borrow for the whole call. If PF-08 moves rendering off the event loop that
    /// premise breaks, and this must become "clear first + re-check after the frame", otherwise requests arriving mid-frame are lost.
    fn finish_frame_at(&mut self, now: Instant) {
        self.render_requested = false;
        self.last_render_at = now;
        if let Some(clock) = self.animation.as_mut() {
            clock.advance_to(now);
        }
    }
}

/// Decide when the next frame must be painted; `None` = nothing to paint (the loop must not wake at all).
///
/// Rules, in this order:
/// 1. merge floor `floor = last_render_at + min_render_interval` (the earliest a request-driven frame may be painted);
/// 2. a pending request wants `floor` (already in the past ⇒ due immediately); an armed animation wants
///    `max(animation_deadline, floor)`, so a custom period shorter than the floor cannot cause a repaint storm;
/// 3. return the earliest of the armed candidates; both disarmed ⇒ `None`.
///
/// Pure (no clock reads, no IO), which is what makes "cadence = frame cost + interval" and "idle ⇒ zero frames" assertable, and is
/// where a future streaming throttle would plug in (see the PF-08 seam note).
fn next_frame_deadline(
    last_render_at: Instant,
    min_render_interval: Duration,
    render_requested: bool,
    animation_deadline: Option<Instant>,
) -> Option<Instant> {
    let floor = last_render_at + min_render_interval;
    let requested = render_requested.then_some(floor);
    let animated = animation_deadline.map(|at| at.max(floor));
    match (requested, animated) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// Wait until `deadline`; with `None`, never wake — the conditional-timer branch that replaced the old free-running ticker.
///
/// Must stay a free function: the other `select!` branches in [`TuiLoop::run_loop`] hold `&mut self`, so a method here would not
/// borrow-check. A deadline already in the past returns immediately. This is the single place where the std clock domain converts to
/// tokio's (`from_std`); everything else in this module stays `std::time::Instant`, so the policy functions remain runtime-free
/// (corollary: unit tests here must not use `tokio::time::pause()` — that would only pause the tokio clock while `std::Instant`
/// keeps running).
async fn sleep_until_opt(deadline: Option<Instant>) {
    match deadline {
        Some(at) => tokio::time::sleep_until(tokio::time::Instant::from_std(at)).await,
        None => std::future::pending::<()>().await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: most methods write commands to the real stdout (the test harness captures output, so the terminal is not polluted).
    // Environment-dependent points: crossterm::terminal::size() (falls back to 80x24 on non-tty) and
    // enable_raw_mode (fails on non-tty) — the relevant tests branch on the environment and do not assert specific return values.
    // Intentionally untested: the full loop of run() with started=true (requires real terminal stdio + raw mode,
    // falls under integration testing); this file only tests parts independent of external state (precondition validation, small state machine methods).

    #[test]
    fn test_new_initial_state() {
        let loop_ = TuiLoop::new().unwrap();
        // default state: raw not enabled, no redraw request, no animation, default 16ms merge floor
        assert!(!loop_.render_requested);
        assert!(loop_.animation.is_none());
        assert!(!loop_.started);
        assert!(!loop_.stopped);
        assert_eq!(loop_.min_render_interval, Duration::from_millis(16));
        // Nothing armed ⇒ no frame is due at all (the idle loop stays asleep)
        assert_eq!(loop_.frame_deadline(), None);
        // buffer size comes from the current terminal (falls back to 80x24 on non-tty), always > 0
        let (cols, rows) = loop_.buffer.size();
        assert!(cols > 0 && rows > 0);
    }

    #[test]
    fn test_start_enters_raw_mode_and_requests_render() {
        let mut loop_ = TuiLoop::new().unwrap();
        match loop_.start() {
            Ok(()) => {
                // with tty: raw mode is enabled and a first-frame render is requested, and it is idempotent (calling start again succeeds directly)
                assert!(loop_.started);
                assert!(loop_.render_requested);
                loop_.start().unwrap();
                assert!(loop_.started);
            }
            Err(TuiError::Terminal(TerminalError::RawMode {
                action: "enter", ..
            })) => {
                // non-tty: enable_raw_mode fails, flag is not set
                assert!(!loop_.started);
                assert!(!loop_.render_requested);
            }
            Err(e) => panic!("unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_stop_marks_stopped_and_is_idempotent() {
        let mut loop_ = TuiLoop::new().unwrap();
        // stop works without a prior start (Terminal::stop is a no-op for raw mode that was never enabled),
        // plural state is recorded correctly
        assert!(loop_.stop().is_ok());
        assert!(!loop_.started);
        assert!(loop_.stopped);
        // idempotent: stop again is still Ok and the state is unchanged
        loop_.stop().unwrap();
        assert!(loop_.stopped);
    }

    #[test]
    fn test_start_stop_cycle_restarts() {
        let mut loop_ = TuiLoop::new().unwrap();
        if loop_.start().is_ok() {
            // with tty: full start → stop → start roundtrip (external editor suspend/resume scenario)
            loop_.stop().unwrap();
            assert!(!loop_.started);
            assert!(loop_.stopped);
            loop_.start().unwrap();
            assert!(loop_.started);
        }
        // non-tty: skip when start fails, to avoid asserting side effects on raw mode state
    }

    #[test]
    fn test_request_render_sets_flag() {
        let mut loop_ = TuiLoop::new().unwrap();
        assert!(!loop_.render_requested);
        loop_.request_render();
        assert!(loop_.render_requested);
        // repeated requests only set the flag without causing other state changes
        loop_.request_render();
        assert!(loop_.render_requested);
    }

    #[test]
    fn test_set_animation_arms_and_disarms() {
        let mut loop_ = TuiLoop::new().unwrap();
        let t0 = Instant::now();
        assert!(loop_.animation.is_none());
        // Arming: the clock is due one period later, and the first frame is requested immediately
        loop_.set_animation_at(Some(DEFAULT_ANIMATION_INTERVAL), t0);
        assert_eq!(
            loop_.animation.map(|clock| clock.deadline()),
            Some(t0 + DEFAULT_ANIMATION_INTERVAL)
        );
        assert!(loop_.render_requested);
        // Disarming: no clock left, and a pending request is left untouched
        loop_.render_requested = false;
        loop_.set_animation_at(None, t0);
        assert!(loop_.animation.is_none());
        assert!(!loop_.render_requested);
    }

    #[test]
    fn test_rearming_animation_reanchors_like_pi_set_indicator() {
        // pi `setIndicator → start()`: arming again restarts the cadence from the new instant with the new period.
        let mut loop_ = TuiLoop::new().unwrap();
        let t0 = Instant::now();
        loop_.last_render_at = t0;
        loop_.set_animation_at(Some(Duration::from_millis(80)), t0);
        loop_.set_animation_at(Some(Duration::from_millis(50)), t0 + Duration::from_secs(1));
        // Neither the old 80 ms period nor the old anchor survives the re-arm.
        assert_eq!(
            loop_.animation.map(|clock| clock.deadline()),
            Some(t0 + Duration::from_secs(1) + Duration::from_millis(50))
        );
        // The first frame still goes out immediately (arming requested it): the merge floor, not the new clock, is due first.
        assert_eq!(loop_.frame_deadline(), Some(t0 + Duration::from_millis(16)));
    }

    #[test]
    fn test_set_min_render_interval_overrides_default() {
        let mut loop_ = TuiLoop::new().unwrap();
        // default 16ms
        assert_eq!(loop_.min_render_interval, Duration::from_millis(16));
        // override with an arbitrary value (0 = no merge floor at all: every request paints on the next iteration)
        loop_.set_min_render_interval(Duration::from_millis(100));
        assert_eq!(loop_.min_render_interval, Duration::from_millis(100));
        loop_.set_min_render_interval(Duration::ZERO);
        assert_eq!(loop_.min_render_interval, Duration::ZERO);
    }

    // ── PF-03: request-driven scheduling (no free-running ticker) ──

    #[test]
    fn test_next_frame_deadline_rules() {
        let t0 = Instant::now();
        let floor = Duration::from_millis(16);
        // Nothing armed → nothing to paint (the loop does not wake at all)
        assert_eq!(next_frame_deadline(t0, floor, false, None), None);
        // Pending request only → merged at the throttle floor
        assert_eq!(next_frame_deadline(t0, floor, true, None), Some(t0 + floor));
        // Animation only (period well above the floor) → the animation deadline wins
        assert_eq!(
            next_frame_deadline(t0, floor, false, Some(t0 + DEFAULT_ANIMATION_INTERVAL)),
            Some(t0 + DEFAULT_ANIMATION_INTERVAL)
        );
        // Both armed → the earlier one
        assert_eq!(
            next_frame_deadline(t0, floor, true, Some(t0 + DEFAULT_ANIMATION_INTERVAL)),
            Some(t0 + floor)
        );
        // A sub-floor animation period is raised to the floor (a tiny custom period cannot cause a repaint storm)
        assert_eq!(
            next_frame_deadline(t0, floor, false, Some(t0 + Duration::from_millis(1))),
            Some(t0 + floor)
        );
    }

    #[test]
    fn test_animation_zero_period_is_clamped() {
        // A zero period would always be due (= busy loop) → clamped to the 1 ms floor
        let t0 = Instant::now();
        let clock = AnimationClock::armed(Duration::ZERO, t0);
        assert_eq!(clock.deadline(), t0 + MIN_ANIMATION_INTERVAL);
    }

    #[test]
    fn test_advance_to_reanchors_and_does_not_catch_up() {
        let t0 = Instant::now();
        let mut clock = AnimationClock::armed(DEFAULT_ANIMATION_INTERVAL, t0);
        // A frame that took 124 ms completes at t0+124 ⇒ next due 80 ms after that (204 ms) — not at the fixed phase
        // (160 ms) and never as a batch of catch-up frames (240 ms).
        clock.advance_to(t0 + Duration::from_millis(124));
        assert_eq!(
            clock.deadline(),
            t0 + Duration::from_millis(124) + DEFAULT_ANIMATION_INTERVAL
        );
    }

    #[test]
    fn test_finish_frame_at_reanchors_to_exact_deadline() {
        let mut loop_ = TuiLoop::new().unwrap();
        let t0 = Instant::now();
        loop_.last_render_at = t0;
        loop_.set_animation_at(Some(DEFAULT_ANIMATION_INTERVAL), t0);
        loop_.render_requested = true;
        loop_.finish_frame_at(t0 + Duration::from_millis(124));
        assert!(!loop_.render_requested);
        assert_eq!(loop_.last_render_at, t0 + Duration::from_millis(124));
        // Exact value (not `> Instant::now()`): armed animation is due `now + interval` after the frame, so the loop
        // cannot wake up already-due (which would be a busy loop).
        assert_eq!(
            loop_.frame_deadline(),
            Some(t0 + Duration::from_millis(124) + DEFAULT_ANIMATION_INTERVAL)
        );
    }

    #[test]
    fn test_finish_frame_disarmed_leaves_nothing_due() {
        let mut loop_ = TuiLoop::new().unwrap();
        let t0 = Instant::now();
        loop_.last_render_at = t0;
        loop_.render_requested = true;
        loop_.finish_frame_at(t0 + Duration::from_millis(5));
        assert!(loop_.animation.is_none());
        assert_eq!(loop_.frame_deadline(), None);
    }

    /// Frames painted inside `window` when every frame costs `cost` and only the animation drives repaints.
    /// Mirrors the loop: the due instant comes from [`next_frame_deadline`], then the clock is re-anchored at completion.
    fn simulated_animation_frames(window: Duration, cost: Duration) -> u64 {
        let t0 = Instant::now();
        let end = t0 + window;
        let mut last_render_at = t0;
        let mut clock = AnimationClock::armed(DEFAULT_ANIMATION_INTERVAL, t0);
        let mut frames = 0u64;
        while let Some(due) = next_frame_deadline(
            last_render_at,
            Duration::from_millis(16),
            false,
            Some(clock.deadline()),
        ) {
            if due > end {
                break;
            }
            frames += 1;
            last_render_at = due + cost; // the frame completes `cost` after it became due
            clock.advance_to(last_render_at);
        }
        frames
    }

    #[test]
    fn test_animation_cadence_is_cost_plus_interval() {
        let window = Duration::from_secs(60);
        // Cadence = frame cost + 80 ms ⇒ 80 / 204 / 100 ms per frame ⇒ 750 / 294 / 600 frames in 60 s.
        assert_eq!(simulated_animation_frames(window, Duration::ZERO), 750);
        assert_eq!(
            simulated_animation_frames(window, Duration::from_millis(124)),
            294
        );
        assert_eq!(
            simulated_animation_frames(window, Duration::from_millis(20)),
            600
        );
        // For comparison: the removed free-running 16 ms ticker painted this many frames in the same window.
        assert_eq!(window.as_millis() / 16, 3750);
    }

    /// Runtime smoke of the frame pacing through the **real** loop (not just the pure decision): verifies the wiring that
    /// animation-armed ⇒ ≈1 frame per 80 ms, and disarmed ⇒ zero frames (idle loop does not wake).
    ///
    /// Requires a real tty (raw mode); run it manually under a pty:
    /// `script -q /dev/null cargo test -p zen-tui tuiloop_frame_pacing_smoke -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "requires a real tty (raw mode); run under a pty via `script`"]
    async fn tuiloop_frame_pacing_smoke() {
        use std::cell::Cell;
        use std::rc::Rc;

        let mut tui = TuiLoop::new().unwrap();
        tui.start().unwrap();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<WakeOutcome>();
        tui.set_external_wakeup(rx, Box::new(|outcome: &WakeOutcome| *outcome));
        let cancel = CancellationToken::new();

        let drawn = Rc::new(Cell::new(0u32));
        let drawn_in_frame = Rc::clone(&drawn);
        let cancel_driver = cancel.clone();
        let local = tokio::task::LocalSet::new();
        let (armed_frames, idle_frames) = local
            .run_until(async move {
                let run = tui.run_scrollback(
                    &cancel,
                    |_| EventOutcome::Continue,
                    move |_| drawn_in_frame.set(drawn_in_frame.get() + 1),
                );
                let driver = async move {
                    // Let run() paint its first frame, then arm the cadence for 1 s and measure.
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    tx.send(WakeOutcome::Animate(Some(DEFAULT_ANIMATION_INTERVAL)))
                        .unwrap();
                    tokio::time::sleep(Duration::from_millis(1000)).await;
                    let armed = drawn.get();
                    // Disarm: an idle loop must not paint (nor wake) at all.
                    tx.send(WakeOutcome::Animate(None)).unwrap();
                    tokio::time::sleep(Duration::from_millis(1000)).await;
                    let idle = drawn.get() - armed;
                    cancel_driver.cancel();
                    (armed, idle)
                };
                let (run_result, measured) = tokio::join!(run, driver);
                run_result.unwrap();
                // `run` no longer borrows the loop here, so the terminal can be restored inside the local set.
                tui.stop().unwrap();
                measured
            })
            .await;

        println!(
            "pacing smoke: armed(1s) = {armed_frames} frames, idle(1s) = {idle_frames} frames"
        );
        // 80 ms cadence ⇒ 12–13 frames per second (allow scheduler jitter on both sides).
        assert!(
            (8..=20).contains(&armed_frames),
            "armed 1s painted {armed_frames} frames, expected ~12 (80 ms cadence)"
        );
        assert!(
            idle_frames <= 1,
            "idle 1s painted {idle_frames} frames, expected 0 (no free-running ticker)"
        );
    }

    #[tokio::test]
    async fn test_sleep_until_opt_is_pending_when_nothing_is_armed() {
        // Nothing armed ⇒ the timer branch must never become ready (an idle loop costs zero wakeups).
        let pending = tokio::time::timeout(Duration::from_millis(10), sleep_until_opt(None)).await;
        assert!(pending.is_err(), "None must never wake the loop");
        // A deadline already in the past is due immediately.
        let past = Some(Instant::now() - Duration::from_millis(1));
        let ready = tokio::time::timeout(Duration::from_millis(500), sleep_until_opt(past)).await;
        assert!(ready.is_ok(), "a past deadline must be due immediately");
    }

    #[test]
    fn test_render_frame_invokes_draw_and_flushes() {
        let mut loop_ = TuiLoop::new().unwrap();
        let mut draw_called = false;
        let mut seen_size = (0, 0);
        let result = loop_.render_frame(&mut |buf: &mut ScreenBuffer| {
            draw_called = true;
            seen_size = buf.size();
            // write content into the buffer (blank style inferred via Default::default())
            buf.write(0, 0, &Default::default(), "ok");
        });
        assert!(result.is_ok());
        assert!(draw_called);
        assert_eq!(seen_size, loop_.buffer.size());
        // flush promotes the current frame to previous frame so it can render again (idempotency smoke test)
        assert!(loop_.render_frame(&mut |_| {}).is_ok());
    }

    #[test]
    fn test_frame_delivers_exactly_once_and_after_the_canvas_flush() {
        // PF-06 (T8): one frame ⇒ exactly one delivery, and the delivery counter must be read **after**
        // `flush_out` (otherwise `FrameMetrics::writes` would be 0). Enabling the accumulator for this loop makes
        // the recorded metric observable without a tty.
        let mut loop_ = TuiLoop::new().unwrap();
        loop_.stats = Arc::new(FrameStats::new_with_enabled(
            crate::metrics::DEFAULT_BUDGET,
            true,
        ));
        let before = loop_.terminal.write_calls();

        loop_
            .render_frame(&mut |buf: &mut ScreenBuffer| {
                buf.write(0, 0, &Default::default(), "frame one");
            })
            .unwrap();
        assert_eq!(
            loop_.terminal.write_calls(),
            before + 1,
            "a frame must reach the terminal exactly once"
        );

        let summary = loop_.stats.summary();
        assert!(
            summary.contains("1 frames"),
            "the frame must be recorded: {summary}"
        );
        assert!(
            summary.contains("avg 1.00 writes/frame"),
            "writes must be counted after flush_out (ordering trap): {summary}"
        );

        // A second frame is a second delivery (the counter is per frame, not cumulative).
        loop_.render_frame(&mut |_| {}).unwrap();
        assert_eq!(loop_.terminal.write_calls(), before + 2);
    }

    #[test]
    fn test_frame_stats_is_the_same_arc_and_records_nothing_by_default() {
        // PF-01: `frame_stats()` hands out a clone of the loop's `Arc` (same allocation), and unless
        // ZEN_TUI_TRACE_FRAMES is set at construction the accumulator stays empty — frame rendering must
        // not start costing a clock read just because the accessor exists.
        let mut loop_ = TuiLoop::new().unwrap();
        let stats = loop_.frame_stats();
        assert!(Arc::ptr_eq(&stats, &loop_.frame_stats()));
        let _ = loop_.render_frame(&mut |_| {});
        if !stats.enabled() {
            assert_eq!(stats.frames(), 0);
            assert_eq!(stats.bytes_total(), 0);
        }
    }

    #[tokio::test]
    async fn test_run_without_start_returns_not_started() {
        let mut loop_ = TuiLoop::new().unwrap();
        let cancel = CancellationToken::new();
        // precondition checks run before any IO (EventStream/interval) —
        // hence this error branch can be verified without a real terminal
        let result = loop_.run(&cancel, |_| EventOutcome::Continue, |_| {}).await;
        assert!(matches!(result, Err(TuiError::NotStarted)));
    }

    #[tokio::test]
    async fn test_run_scrollback_without_start_returns_not_started() {
        let mut loop_ = TuiLoop::new().unwrap();
        let cancel = CancellationToken::new();
        // Scrollback mode reuses the same run_loop preflight validation: reject before start; no real terminal required.
        let result = loop_
            .run_scrollback(&cancel, |_| EventOutcome::Continue, |_| {})
            .await;
        assert!(matches!(result, Err(TuiError::NotStarted)));
    }

    // ── Fourth external wake (T-AP-11-1 patch): flag transitions and channel delivery ──

    #[test]
    fn test_apply_wake_outcome_sets_flags() {
        let mut loop_ = TuiLoop::new().unwrap();
        // By default neither state is set
        assert!(!loop_.render_requested);
        assert!(loop_.animation.is_none());
        // Render → request one frame
        loop_.apply_wake_outcome(WakeOutcome::Render);
        assert!(loop_.render_requested);
        assert!(
            loop_.animation.is_none(),
            "Render does not affect animation state"
        );
        // Animate(Some(period)) → arm the cadence
        loop_.apply_wake_outcome(WakeOutcome::Animate(Some(DEFAULT_ANIMATION_INTERVAL)));
        assert_eq!(
            loop_.animation.map(|clock| clock.interval),
            Some(DEFAULT_ANIMATION_INTERVAL)
        );
        // Animate(None) → disarm (the merge flag stays set)
        loop_.apply_wake_outcome(WakeOutcome::Animate(None));
        assert!(loop_.animation.is_none());
    }

    #[tokio::test]
    async fn test_external_wakeup_delivers_messages_and_outcomes() {
        let mut loop_ = TuiLoop::new().unwrap();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<u8>();
        // Generic channel: zen-tui is unaware of the concrete type (here u8 verifies the callback still yields intents after erasure)
        loop_.set_external_wakeup(
            rx,
            Box::new(|msg: &u8| match msg {
                1 => WakeOutcome::Render,
                2 => WakeOutcome::Animate(Some(DEFAULT_ANIMATION_INTERVAL)),
                _ => WakeOutcome::Animate(None),
            }),
        );
        tx.send(1).unwrap();
        assert_eq!(loop_.next_wake().await, Some(WakeOutcome::Render));
        tx.send(2).unwrap();
        assert_eq!(
            loop_.next_wake().await,
            Some(WakeOutcome::Animate(Some(DEFAULT_ANIMATION_INTERVAL)))
        );
        tx.send(3).unwrap();
        assert_eq!(loop_.next_wake().await, Some(WakeOutcome::Animate(None)));
    }

    #[tokio::test]
    async fn test_external_wakeup_ends_when_channel_closed() {
        let mut loop_ = TuiLoop::new().unwrap();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<u8>();
        loop_.set_external_wakeup(rx, Box::new(|_: &u8| WakeOutcome::Render));
        // Send one message to verify the path, then close the sender → stream ends (None)
        tx.send(9).unwrap();
        assert_eq!(loop_.next_wake().await, Some(WakeOutcome::Render));
        drop(tx);
        assert_eq!(
            loop_.next_wake().await,
            None,
            "Wakeup stream ends after channel disconnect"
        );
    }

    #[tokio::test]
    async fn test_external_wakeup_unregistered_is_pending_forever() {
        // Fourth channel unregistered: next_wake is always pending (in-flight; verify with timeout that it doesn't return immediately)
        let mut loop_ = TuiLoop::new().unwrap();
        let result =
            tokio::time::timeout(std::time::Duration::from_millis(10), loop_.next_wake()).await;
        assert!(
            result.is_err(),
            "Should never become ready when unregistered (timeout expires)"
        );
    }
}
