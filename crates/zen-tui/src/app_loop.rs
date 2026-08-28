use crate::screen::ScreenBuffer;
use crate::terminal::{Terminal, TerminalError};
use std::pin::Pin;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// TUI main loop orchestrator. Owns the output canvas (Terminal + ScreenBuffer) and render state (merge flag, animation flag, render throttle timer, start/stop flags).
/// External contract aligned with zen TUI's caller perspective:
/// Application calls start() to enter raw mode → run() runs the event loop (forwards events + timer tick drives differential refresh) → stop() restores the terminal.
/// Events are forwarded as-is via the on_event callback (key parsing semantics belong to T-TU-03);
/// Redraws write the current frame to the buffer via the draw callback; the loop performs a unified flush of differential output.
/// The loop handles Resize itself (buffer.resize + redraw request).
pub struct TuiLoop {
    terminal: Terminal,                 // Output sink; start/stop enter/exit raw mode
    buffer: ScreenBuffer,               // Current frame buffer; resize/flush inside run
    render_requested: bool,             // Render merge flag: set by request_render(), reset after being consumed by tick
    animation_enabled: bool, // Timed refresh mode: when set, tick continuously repaints (loader animation) — no event driving needed
    last_render_at: std::time::Instant, // Last render timestamp, for coalescing/throttling reference (mainly for debugging/stats when tick interval is fixed)
    min_render_interval: std::time::Duration, // Minimum render interval, default 16ms; the tick interval takes this value
    started: bool,                      // Set after start(); used for precondition checks before run()
    stopped: bool,                      // Lifecycle termination flag
    wake_stream: Option<Pin<Box<dyn futures::Stream<Item = WakeOutcome> + Send>>>, // /// Fourth external wake path (T-AP-11-1 patch): the "message → WakeOutcome" stream registered at assembly; None if unregistered. Items are already erased to `WakeOutcome`, so TuiLoop needs no generics.
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

/// Rendering intent for external wakes: `Render` = request a repaint; `Animate(bool)` = turn continuous animation on/off
/// (loader scenarios); both set the merged bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeOutcome {
    /// Repaint once as needed after an external event.
    Render,
    /// Turn on/off 16ms continuous repaint (streaming loader animation).
    Animate(bool),
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
            animation_enabled: false,
            last_render_at: std::time::Instant::now(),
            min_render_interval: Duration::from_millis(16), // Aligned with pi MIN_RENDER_INTERVAL_MS
            started: false,
            stopped: false,
            wake_stream: None,
        })
    }

    /// Enter raw mode and hide the cursor (Terminal::start), request first-frame render.
    /// Idempotent; repeated calls have no side effects. Bracketed paste / keyboard protocol enhancements (EnableBracketedPaste / KeyboardEnhancementFlags)
    /// Whether to enable it in this method: see "Open Issues".
    pub fn start(&mut self) -> Result<(), TuiError> {
        if self.started {
            return Ok(());
        }
        self.terminal.start()?;
        self.started = true;
        self.request_render(); // First frame: draw callback renders the initial screen
        Ok(())
    }

    /// Exits raw mode and restores the cursor (Terminal::stop), restoring terminal state. Idempotent.
    /// Also for apps temporarily exiting raw mode while running (e.g. suspending the TUI during T-AP-11-4 external editor integration, after `stop()`
    /// launch the editor, then `start()` to restore).
    /// Consumer: T-AP-11-1.
    pub fn stop(&mut self) -> Result<(), TuiError> {
        if self.stopped {
            return Ok(());
        }
        self.terminal.stop()?;
        self.started = false;
        self.stopped = true;
        Ok(())
    }

    /// Request a repaint (set the coalescing flag). Multiple calls within a single tick merge into one render; the minimum interval is
    /// controlled by `set_min_render_interval` (default 16ms).
    pub fn request_render(&mut self) {
        self.render_requested = true
    }

    /// Toggle "timed refresh animation mode": when enabled, tick keeps triggering repaints even without events/requests
    /// (loader animation scenario — the loader component's owner sets it while active, resets it when the loader ends).
    pub fn set_animation(&mut self, enabled: bool) {
        self.animation_enabled = enabled
    }

    /// Set the minimum render interval (i.e. the tick period), default 16ms (aligned with pi `MIN_RENDER_INTERVAL_MS`).
    /// Consumer: T-AP-11-1 (optional tuning; can keep the default value and mark as "experimental" when there is no consumer).
    pub fn set_min_render_interval(&mut self, interval: Duration) {
        self.min_render_interval = interval
    }

    /// Main TUI event loop: polls three async sources—event stream / periodic tick / cancellation signal—to drive diff-based refreshes.
    /// Prerequisite: [`TuiLoop::start`] must be called first; see inline comments for exit paths.
    pub async fn run<E, D>(
        &mut self,
        cancel: &CancellationToken,
        mut on_event: E,
        mut draw: D,
    ) -> Result<(), TuiError>
    where
        E: FnMut(crossterm::event::Event) -> EventOutcome,
        D: FnMut(&mut ScreenBuffer),
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

        // Timer: fires a tick every min_render_interval (default 16ms);
        // serves as the metronome for "render coalescing" and the "loader animation"
        let mut tick = tokio::time::interval(self.min_render_interval);
        // When ticks are missed due to event floods (e.g. holding down a key), Delay lets the backlog fire late,
        // rather than compensating for all missed ticks at once — avoids render storms
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        // ── First-frame render: draw one frame before entering the loop to avoid a blank screen ──
        // start() already set the coalescing flag via request_render(); honor it here directly
        self.render_frame(&mut draw)?; // draw writes to the buffer → flush emits the diff; on failure, loop startup fails entirely
        self.render_requested = false; // First frame drawn; clear the coalescing flag
        self.last_render_at = std::time::Instant::now(); // Record this render's timestamp for throttling reference

        loop {
            // ── Listen to all three async sources concurrently: whichever becomes ready first executes (tokio's randomized fairness does not favor any one source)──
            tokio::select! {
                // ① Event stream: user input arrives (key press, mouse, paste, window resize, etc.)
                maybe_event = event_stream.next() => {
                    match maybe_event {
                        Some(Ok(event)) => {
                            // Resize is a "window size changed" event; the loop must handle it itself:
                            // Resize the buffer grid size. Note that this matches by value, but the Event is not consumed—
                            // The cols/rows bound by Resize are Copy types, so matching does not move the scrutinee,
                            // The code below can still forward event unchanged to on_event
                            if let crossterm::event::Event::Resize(cols, rows) = event {
                                self.buffer.resize(cols, rows);
                                // resize resets the diff buffer's previous frame → the next flush naturally performs a full-screen redraw
                                self.request_render(); // Size changed → must redraw
                            }

                            // Other events (and Resize itself) are passed to the application unchanged;
                            // Key semantics (shortcut mapping etc., part of T-TU-03) are resolved in on_event
                            match on_event(event) {
                                EventOutcome::Stop => break,
                                // Event handling may have mutated application state → set the coalescing flag,
                                // to be honored on the next tick (aligns with pi handleInput → requestRender contract)
                                EventOutcome::Continue => self.request_render(),
                            }
                        }
                        Some(Err(e)) => return Err(TuiError::EventStream(e)), // underlying IO error in the event stream, terminate the loop
                        None => break, // event stream ended (stdin closed), exit normally
                    }
                }

                // (2) periodic tick: render the ticker
                _ = tick.tick() => {
                    // actually draw only when a redraw was requested or animation mode is on; other ticks idle —
                    // this is render coalescing: high-frequency events do not produce one frame per event; screen updates are coalesced at tick granularity (≤16ms latency)
                    if self.render_requested || self.animation_enabled {
                        self.render_frame(&mut draw)?; // draw renders the current frame → flush outputs cell-by-cell diffs
                        self.render_requested = false; // frame already realized, clear the coalescing flag (the next event will set it again)
                        self.last_render_at = std::time::Instant::now(); // record the render time
                    }
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
    fn render_frame<D>(&mut self, draw: &mut D) -> Result<(), TuiError>
    where
        D: FnMut(&mut ScreenBuffer),
    {
        // Start each frame from a clean frame: cells not redrawn are judged as "content→empty" in the subsequent flush diff and erased,
        // Components need not self-clear (the zen-grid-level counterpart of pi composite layer clear-extra-lines);
        // Identical cells across two frames are still not written after diff, so differential efficiency is unaffected.
        self.buffer.clear();
        (draw)(&mut self.buffer);
        self.buffer.flush(self.terminal.out_mut())?;
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

    /// Set render flags per wake intent: Render → merge into one redraw; Animate(v) → toggle 16ms continuous rendering.
    fn apply_wake_outcome(&mut self, outcome: WakeOutcome) {
        match outcome {
            WakeOutcome::Render => self.request_render(),
            WakeOutcome::Animate(v) => self.animation_enabled = v,
        }
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
        // default state: raw not enabled, no redraw request, no animation, default 16ms interval
        assert!(!loop_.render_requested);
        assert!(!loop_.animation_enabled);
        assert!(!loop_.started);
        assert!(!loop_.stopped);
        assert_eq!(loop_.min_render_interval, Duration::from_millis(16));
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
    fn test_set_animation_toggles_flag() {
        let mut loop_ = TuiLoop::new().unwrap();
        assert!(!loop_.animation_enabled);
        loop_.set_animation(true);
        assert!(loop_.animation_enabled);
        loop_.set_animation(false);
        assert!(!loop_.animation_enabled);
    }

    #[test]
    fn test_set_min_render_interval_overrides_default() {
        let mut loop_ = TuiLoop::new().unwrap();
        // default 16ms
        assert_eq!(loop_.min_render_interval, Duration::from_millis(16));
        // override with an arbitrary value (including 0 — tokio interval's panic boundary only triggers at runtime, here we only verify storage)
        loop_.set_min_render_interval(Duration::from_millis(100));
        assert_eq!(loop_.min_render_interval, Duration::from_millis(100));
        loop_.set_min_render_interval(Duration::ZERO);
        assert_eq!(loop_.min_render_interval, Duration::ZERO);
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

    #[tokio::test]
    async fn test_run_without_start_returns_not_started() {
        let mut loop_ = TuiLoop::new().unwrap();
        let cancel = CancellationToken::new();
        // precondition checks run before any IO (EventStream/interval) —
        // hence this error branch can be verified without a real terminal
        let result = loop_.run(&cancel, |_| EventOutcome::Continue, |_| {}).await;
        assert!(matches!(result, Err(TuiError::NotStarted)));
    }

    // ── Fourth external wake (T-AP-11-1 patch): flag transitions and channel delivery ──

    #[test]
    fn test_apply_wake_outcome_sets_flags() {
        let mut loop_ = TuiLoop::new().unwrap();
        // By default neither flag is set
        assert!(!loop_.render_requested);
        assert!(!loop_.animation_enabled);
        // Render → request one redraw
        loop_.apply_wake_outcome(WakeOutcome::Render);
        assert!(loop_.render_requested);
        assert!(!loop_.animation_enabled, "Render does not affect animation state");
        // Animate(true) → start continuous animation
        loop_.apply_wake_outcome(WakeOutcome::Animate(true));
        assert!(loop_.animation_enabled);
        // Animate(false) → stop continuous animation (redraw flag stays set)
        loop_.apply_wake_outcome(WakeOutcome::Animate(false));
        assert!(!loop_.animation_enabled);
    }

    #[tokio::test]
    async fn test_external_wakeup_delivers_messages_and_outcomes() {
        let mut loop_ = TuiLoop::new().unwrap();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<u8>();
        // Generic channel: zen-tui is unaware of the concrete type (here u8 verifies the callback still yields intents after erasure)
        loop_.set_external_wakeup(
            rx,
            Box::new(|msg: &u8| {
                if *msg == 1 {
                    WakeOutcome::Render
                } else {
                    WakeOutcome::Animate(*msg == 2)
                }
            }),
        );
        tx.send(1).unwrap();
        assert_eq!(loop_.next_wake().await, Some(WakeOutcome::Render));
        tx.send(2).unwrap();
        assert_eq!(loop_.next_wake().await, Some(WakeOutcome::Animate(true)));
        tx.send(3).unwrap();
        assert_eq!(loop_.next_wake().await, Some(WakeOutcome::Animate(false)));
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
        assert_eq!(loop_.next_wake().await, None, "Wakeup stream ends after channel disconnect");
    }

    #[tokio::test]
    async fn test_external_wakeup_unregistered_is_pending_forever() {
        // Fourth channel unregistered: next_wake is always pending (in-flight; verify with timeout that it doesn't return immediately)
        let mut loop_ = TuiLoop::new().unwrap();
        let result =
            tokio::time::timeout(std::time::Duration::from_millis(10), loop_.next_wake()).await;
        assert!(result.is_err(), "Should never become ready when unregistered (timeout expires)");
    }
}
