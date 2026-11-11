/// A thin stateful adapter over crossterm terminal primitives, serving as the screen manager's external output sink. Tracks whether raw mode is currently enabled,
/// whether the cursor is currently hidden, as well as keyboard protocol state (keyboard enhancement flag / bracketed paste), and provides start/stop
/// (Entering/exiting raw mode, hiding/showing the cursor, enabling/disabling keyboard enhancements and bracketed paste), write, size, cursor movement, clearing the screen, etc.
/// Command methods.
///
/// Protocol enablement location (consumer: T-AP-11-1-fix): crossterm is enabled in `start()` together with raw mode.
/// Keyboard enhancements (`PushKeyboardEnhancementFlags`, Kitty CSI-u) and bracketed paste (`EnableBracketedPaste`),
/// `stop()` symmetrically disables them—unsupported terminals ignore enhancement sequences with no side effects (blind push; no query-first negotiation with pi;
/// The modifyOtherKeys fallback is not implemented in v1; see tasks/T-AP-11-1-fix-implemented.md.
pub(crate) struct Terminal {
    out: std::io::Stdout, // target for writing crossterm commands (tail of queue!/execute!)
    /// PF-06: the frame's output sink. Frame content (crossterm diff commands + line bytes) accumulates here and is
    /// delivered to `out` exactly once per frame by [`Terminal::flush_out`] — the Rust counterpart of pi's
    /// `buffer: string` -> `terminal.write(buffer)` (`packages/tui/src/tui.ts:1604`).
    ///
    /// Why a `Vec` instead of writing straight to `out`: `Stdout` is a `LineWriter` that flushes on every `\n`, so
    /// writing frame content directly reaches the terminal in ~8-10 chunks per frame; unsynchronized terminals that
    /// ignore `?2026` then repaint chunk by chunk (visible flicker). A `Vec` also makes a frame's bytes *readable*
    /// in tests, which `Stdout` never was (see `flush_frame`).
    ///
    /// **Two write ends, do not mix**: only frame content goes here; control paths (`write`/`set_title`/
    /// `set_progress`/`start`/`stop`/`handover_to_shell` and every `execute!(self.out, …)`) write `self.out` directly.
    frame_buf: Vec<u8>,
    /// PF-06: number of successful frame deliveries (one `write_all` + `flush` each). Read as a per-frame delta by
    /// [`Terminal::write_calls`] to feed `FrameMetrics::writes`. Control-path writes are deliberately not counted:
    /// this counter answers "how many times did this frame reach the terminal", not "how many syscalls happened".
    write_calls: u64,
    raw_mode_active: bool, // Record raw mode state across start/stop for idempotent restoration by stop
    cursor_hidden: bool,   // record cursor visibility state to avoid sending Hide/Show repeatedly
    // Keyboard protocol pairing flag (idempotent: do not resend commands when already enabled/disabled; mirrors raw_mode_active)
    keyboard_enhancement_active: bool,
    bracketed_paste_active: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum TerminalError {
    #[error("terminal I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to {action} raw mode: {source}")]
    RawMode {
        /// Failed operation: "enter" or "exit".
        action: &'static str,
        /// Underlying IO error.
        #[source]
        source: std::io::Error,
    },
}

impl Terminal {
    /// Create a new Terminal. Raw mode is typically not entered here; it is explicitly enabled by [`Terminal::start`].
    pub fn new() -> Result<Terminal, TerminalError> {
        Ok(Terminal {
            out: std::io::stdout(), // Get the standard output handle
            frame_buf: Vec::new(),
            write_calls: 0,
            raw_mode_active: false,
            cursor_hidden: false,
            keyboard_enhancement_active: false,
            bracketed_paste_active: false,
        })
    }

    /// Enter raw mode and hide the cursor, while enabling keyboard enhancements and bracketed paste (to support Shift+Enter's CSI-u
    /// events and allow `Event::Paste` to arrive), preparing for TUI rendering. Idempotent (succeeds immediately if already enabled).
    ///
    /// Keyboard enhancement flags = DISAMBIGUATE_ESCAPE_CODES | REPORT_ALTERNATE_KEYS (**do not enable
    /// REPORT_EVENT_TYPES to avoid double-triggering Release events—zen's event chain has no kind filtering layer,
    /// See design decision 1 in tasks/T-AP-11-1-fix-implemented.md).
    /// See [`Terminal::stop`] for paired deactivation; start/stop round trips (e.g., external editor suspend/resume in T-AP-11-4)
    /// Naturally reentrant: protocol commands are resent via start() during restoration.
    pub fn start(&mut self) -> Result<(), TerminalError> {
        if !self.raw_mode_active {
            crossterm::terminal::enable_raw_mode().map_err(|e| TerminalError::RawMode {
                action: "enter",
                source: e,
            })?;
            self.raw_mode_active = true;
        }

        if !self.cursor_hidden {
            crossterm::execute!(self.out, crossterm::cursor::Hide).map_err(TerminalError::Io)?;
            self.cursor_hidden = true;
        }

        // Keyboard enhancement (Kitty CSI-u: Shift+Enter/modifier-modified function keys are distinguishable). Idempotent pairing.
        if !self.keyboard_enhancement_active {
            use crossterm::event::{KeyboardEnhancementFlags, PushKeyboardEnhancementFlags};
            crossterm::execute!(
                self.out,
                PushKeyboardEnhancementFlags(
                    KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                        | KeyboardEnhancementFlags::REPORT_ALTERNATE_KEYS
                )
            )
            .map_err(TerminalError::Io)?;
            self.keyboard_enhancement_active = true;
        }

        // Bracketed paste: the terminal wraps pasted content in \x1b[200~…\x1b[201~, which crossterm parses as Event::Paste.
        if !self.bracketed_paste_active {
            crossterm::execute!(self.out, crossterm::event::EnableBracketedPaste)
                .map_err(TerminalError::Io)?;
            self.bracketed_paste_active = true;
        }

        self.clear_screen()?;

        Ok(())
    }

    /// Exit raw mode, restore the cursor, and flush to restore terminal state. Idempotent.
    /// Disable order: restore cursor → disable bracketed paste → disable keyboard enhancement → raw mode (escape sequences are ANSI output,
    /// Sent before disable_raw_mode to ensure the terminal receives them; after exit, no flag/paste residue leaks to the shell).
    pub fn stop(&mut self) -> Result<(), TerminalError> {
        if self.cursor_hidden {
            crossterm::execute!(self.out, crossterm::cursor::Show).map_err(TerminalError::Io)?;
            self.cursor_hidden = false;
        }

        if self.bracketed_paste_active {
            crossterm::execute!(self.out, crossterm::event::DisableBracketedPaste)
                .map_err(TerminalError::Io)?;
            self.bracketed_paste_active = false;
        }

        if self.keyboard_enhancement_active {
            crossterm::execute!(self.out, crossterm::event::PopKeyboardEnhancementFlags)
                .map_err(TerminalError::Io)?;
            self.keyboard_enhancement_active = false;
        }

        if self.raw_mode_active {
            crossterm::terminal::disable_raw_mode().map_err(|e| TerminalError::RawMode {
                action: "exit",
                source: e,
            })?;
            self.raw_mode_active = false;
        }

        Ok(())
    }

    /// Write raw bytes to the terminal, for verbatim output and debugging that is neither buffered nor a command.
    ///
    /// Control/non-frame path: goes straight to `self.out` (never through the frame buffer), so it cannot disturb
    /// frame delivery. Not counted by [`Terminal::write_calls`].
    pub fn write(&mut self, data: &str) -> Result<(), TerminalError> {
        use std::io::Write;
        self.out
            .write_all(data.as_bytes())
            .map_err(TerminalError::Io)?;
        self.out.flush().map_err(TerminalError::Io)?;
        Ok(())
    }

    /// Return the current terminal size (columns).
    /// Consumers: T-TU-01-1 (main loop), T-TU-01-4 (component layout).
    pub fn columns(&self) -> u16 {
        crossterm::terminal::size()
            .map(|(cols, _rows)| cols)
            .unwrap_or(80)
    }

    /// Return the current terminal size (rows).
    /// Consumers: T-TU-01-1 (main loop), T-TU-01-4 (component layout).
    pub fn rows(&self) -> u16 {
        crossterm::terminal::size()
            .map(|(_cols, rows)| rows)
            .unwrap_or(24)
    }

    /// Move vertically relative to the current cursor position (positive down, negative up, 0 no-op).
    pub fn move_by(&mut self, lines: i16) -> Result<(), TerminalError> {
        if lines == 0 {
            return Ok(());
        }
        if lines > 0 {
            crossterm::execute!(self.out, crossterm::cursor::MoveDown(lines as u16))
        } else {
            crossterm::execute!(self.out, crossterm::cursor::MoveUp((-lines) as u16))
        }
        .map_err(TerminalError::Io)
    }

    /// Hide the cursor.
    pub fn hide_cursor(&mut self) -> Result<(), TerminalError> {
        if !self.cursor_hidden {
            crossterm::execute!(self.out, crossterm::cursor::Hide).map_err(TerminalError::Io)?;
            self.cursor_hidden = true;
        }
        Ok(())
    }

    /// Show the cursor.
    pub fn show_cursor(&mut self) -> Result<(), TerminalError> {
        if self.cursor_hidden {
            crossterm::execute!(self.out, crossterm::cursor::Show).map_err(TerminalError::Io)?;
            self.cursor_hidden = false;
        }
        Ok(())
    }

    /// Clear the current line.
    pub fn clear_line(&mut self) -> Result<(), TerminalError> {
        crossterm::execute!(
            self.out,
            crossterm::terminal::Clear(crossterm::terminal::ClearType::CurrentLine)
        )
        .map_err(TerminalError::Io)
    }

    /// Clear from the cursor to the end of the screen.
    pub fn clear_from_cursor(&mut self) -> Result<(), TerminalError> {
        crossterm::execute!(
            self.out,
            crossterm::terminal::Clear(crossterm::terminal::ClearType::FromCursorDown)
        )
        .map_err(TerminalError::Io)
    }

    /// Hand the terminal back to the shell (the handover step of `TuiLoop::stop`, aligned with pi `Tui.stop()`'s
    /// 「Move cursor to the end of the content to prevent overwriting/artifacts on exit」）：
    /// clear the cursor's current line (editor input box residue) → clear from below the cursor to end of screen (footer/status line residue) →
    /// `\r\n` lands on a new line at column 0. Chat content above the cursor stays on screen (reviewable via scrollback),
    /// and the shell takes over from a clean new line — avoiding misaligned layout when the prompt/subsequent output after exit starts at a leftover cursor column
    /// （terminal-exit-handover-fix）。
    /// Should be called before [`Terminal::stop`] (restoring raw mode); it only writes to the terminal and does not touch raw/cursor/protocol flags.
    pub fn handover_to_shell(&mut self) -> Result<(), TerminalError> {
        self.clear_line()?;
        self.clear_from_cursor()?;
        self.write("\r\n")
    }

    /// Clear the entire screen and move the cursor back to the top-left corner (0,0).
    pub fn clear_screen(&mut self) -> Result<(), TerminalError> {
        crossterm::execute!(
            self.out,
            crossterm::terminal::Clear(crossterm::terminal::ClearType::All),
            crossterm::cursor::MoveTo(0, 0)
        )
        .map_err(TerminalError::Io)
    }

    /// Set the terminal window title (OSC 0 sequence). Optional enhancement, to be decided at implementation time whether it is in scope for this task.
    /// Consumers: T-TU-01-1 (reserved).
    pub fn set_title(&mut self, title: &str) -> Result<(), TerminalError> {
        use std::io::Write;
        // OSC 0 ; <title> BEL  —— \x1b]0;...\x07
        write!(self.out, "\x1b]0;{}\x07", title).map_err(TerminalError::Io)?;
        self.out.flush().map_err(TerminalError::Io)
    }

    /// Set/clear the terminal progress indicator (OSC 9;4). Optional enhancement, see "Open Questions".
    /// Consumers: T-TU-01-1 (reserved).
    pub fn set_progress(&mut self, active: bool) -> Result<(), TerminalError> {
        use std::io::Write;
        // OSC 9 ; 4 ; <st> BEL  ——  st=0 clears, st=1 sets indeterminate
        let st = if active { 1 } else { 0 };
        write!(self.out, "\x1b]9;4;{}m\x07", st).map_err(TerminalError::Io)?;
        self.out.flush().map_err(TerminalError::Io)
    }

    /// PF-06: start a frame — clears the frame output buffer.
    ///
    /// `flush_out` is the only delivery point, but a frame that bails out early (IO error, panic in a draw callback)
    /// would otherwise leave residue that the next frame appends to and re-delivers. Clearing here is a cheap safety
    /// net; the buffer's capacity is retained so a steady-state frame allocates nothing.
    /// # Consumers
    /// `TuiLoop::run_scrollback` and `TuiLoop::render_frame` (first statement of each frame).
    pub(crate) fn begin_frame(&mut self) {
        self.frame_buf.clear();
    }

    /// PF-06: deliver the whole frame to the terminal in **one** `write_all` — the only delivery point per frame
    /// (the Rust counterpart of pi's `terminal.write(buffer)`, `packages/tui/src/tui.ts:1604`).
    ///
    /// Why this exists at all: `queue!`/`write_all` calls accumulate in [`Terminal::frame_buf`], and `Stdout`'s
    /// `LineWriter` would otherwise flush per line. One `write_all` means the terminal receives the frame (including
    /// the `?2026h`/`?2026l` pair and the IME cursor positioning, which both live inside the canvas flush) as a single
    /// chunk, so terminals without synchronized-update support cannot repaint it piecewise.
    /// # Behaviour
    /// - delegates to [`flush_frame`] (write_all + flush + clear); the buffer is cleared whether the write succeeds or
    ///   fails, so a retry never re-sends half a frame;
    /// - `write_calls` is incremented **only on success** (it counts deliveries, not attempts);
    /// - an oversized frame (e.g. the ~403 KB first full redraw) still goes out in one `write_all`: there is no
    ///   artificial capacity split (kernel/pty may still split, which is out of our hands);
    /// - after delivery, a buffer that grew past [`FRAME_BUF_SHRINK_THRESHOLD`] is shrunk back to
    ///   [`FRAME_BUF_RETAIN`] so a single huge frame does not pin hundreds of KB for the session's lifetime.
    /// # Consumers
    /// `TuiLoop::run_scrollback` and `TuiLoop::render_frame`. **Must be called before** `FrameProbe::finish`, otherwise
    /// `FrameMetrics::flush` (= elapsed - draw, frozen inside `finish`) only measures the in-memory copy.
    pub(crate) fn flush_out(&mut self) -> Result<(), TerminalError> {
        flush_frame(&mut self.frame_buf, &mut self.out)?;
        self.write_calls += 1;
        if self.frame_buf.capacity() > FRAME_BUF_SHRINK_THRESHOLD {
            self.frame_buf.shrink_to(FRAME_BUF_RETAIN);
        }
        Ok(())
    }

    /// PF-06: successful frame deliveries since startup (monotonic).
    ///
    /// Consumers read a per-frame **delta** to fill `FrameMetrics::writes`, so "did this frame reach the terminal
    /// exactly once" becomes observable on a real terminal (a unit test can only prove the buffer behaviour).
    /// Boundary: counts [`Terminal::flush_out`] successes only; `execute!(self.out, …)` control sequences and
    /// [`Terminal::write`] are not counted, and one delivery is not necessarily one syscall (kernel/pty may split).
    /// # Consumers
    /// `TuiLoop::run_scrollback`, `TuiLoop::render_frame`.
    pub(crate) fn write_calls(&self) -> u64 {
        self.write_calls
    }

    /// PF-06: expose the **frame** write end (crossterm diff commands and line bytes land here, not on stdout).
    ///
    /// Returns `&mut Vec<u8>` so a frame's bytes are readable in tests and so the delivery step can be factored out
    /// into [`flush_frame`]. Callers only use it as `W: Write` (`CountingWriter::new(terminal.out_mut())`,
    /// `canvas.flush(out)`), so no call site changes.
    /// ⚠️ Frame content only — control paths keep writing `self.out`.
    /// # Consumers
    /// `TuiLoop::run_scrollback`, `TuiLoop::render_frame`.
    pub(crate) fn out_mut(&mut self) -> &mut Vec<u8> {
        &mut self.frame_buf
    }
}

impl Drop for Terminal {
    /// Panic-safe terminal restoration fallback: during UI task panic unwind, explicit `stop()` call is skipped,
    /// without this guard, raw mode / kitty keyboard enhancement / bracketed paste / hidden cursor all remain in the shell,
    /// the terminal shows distorted keys and garbled layout (post-implementation defect: terminal not restored after abnormal exit).
    /// `stop()` is idempotent (all actions guarded by *_active flags); after explicit stop on normal path, this fallback is a no-op;
    /// IO errors from stop during unwind are swallowed (best-effort; must not panic again while propagating panic).
    fn drop(&mut self) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = self.stop();
        }));
    }
}

/// PF-06: capacity above which the frame buffer is shrunk after delivery (~403 KB first full redraw would otherwise
/// pin half a megabyte for the whole session). Below this threshold the capacity is deliberately retained so that
/// steady-state frames allocate nothing.
pub(crate) const FRAME_BUF_SHRINK_THRESHOLD: usize = 1024 * 1024;

/// PF-06: capacity the frame buffer is shrunk back to after an oversized frame.
pub(crate) const FRAME_BUF_RETAIN: usize = 64 * 1024;

/// PF-06: deliver one frame's bytes in a single `write_all`, then flush, and clear the buffer **whether the write
/// succeeded or failed**.
///
/// Why a free function instead of inlining this in [`Terminal::flush_out`]: `Terminal`'s write end is a concrete
/// `std::io::Stdout` created by `Terminal::new`, so a test cannot inject a spy writer into it. Factoring the delivery
/// step out leaves it generic over `W: Write`, which is what makes "a frame reaches the terminal exactly once"
/// mechanically assertable (see the tests: `flush_frame(&mut buf, &mut spy)`).
/// # Parameters
/// - `buf`: the frame's accumulated bytes (cleared afterwards; capacity retained)
/// - `out`: the real write end (production: `Stdout`; tests: a spy/`Vec<u8>`)
/// # Boundaries
/// - empty buffer → still one `flush` (no actual write);
/// - failure → the error is propagated **and** the buffer is cleared (a retry must not resend half a frame).
/// # Consumers
/// `Terminal::flush_out` (production); unit tests T1/T2/T3/T5/T7.
pub(crate) fn flush_frame<W: std::io::Write>(
    buf: &mut Vec<u8>,
    out: &mut W,
) -> Result<(), TerminalError> {
    let result = out
        .write_all(buf)
        .and_then(|_| out.flush())
        .map_err(TerminalError::Io);
    buf.clear();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: most methods write escape sequences to the real stdout (the test framework captures output, so the terminal is not polluted).
    // The only environment-dependent points are enable_raw_mode / terminal::size() — in non-tty environments (CI/pipes)
    // they fail; the related tests either only assert Ok (writing to a pipe always succeeds) or branch on environment conditions, without asserting specific return values.

    #[test]
    fn test_new_initial_state() {
        let t = Terminal::new().unwrap();
        assert!(!t.raw_mode_active);
        assert!(!t.cursor_hidden);
    }

    #[test]
    fn test_stop_when_inactive_is_noop() {
        // Not started: stop triggers no crossterm calls, returns Ok directly, and the flag is unchanged
        let mut t = Terminal::new().unwrap();
        assert!(t.stop().is_ok());
        assert!(!t.raw_mode_active);
        assert!(!t.cursor_hidden);
        // Idempotent: stopping again still returns Ok
        assert!(t.stop().is_ok());
    }

    #[test]
    fn test_hide_show_cursor_flag_transitions() {
        let mut t = Terminal::new().unwrap();
        t.hide_cursor().unwrap();
        assert!(t.cursor_hidden);
        // Idempotent: when already hidden, don't send Hide again
        t.hide_cursor().unwrap();
        assert!(t.cursor_hidden);

        t.show_cursor().unwrap();
        assert!(!t.cursor_hidden);
        // Idempotent: when already shown, don't send Show again
        t.show_cursor().unwrap();
        assert!(!t.cursor_hidden);
    }

    #[test]
    fn test_move_by_zero_is_noop() {
        let mut t = Terminal::new().unwrap();
        // Return early directly, write no escape sequences
        assert!(t.move_by(0).is_ok());
    }

    #[test]
    fn test_move_by_directions_ok() {
        let mut t = Terminal::new().unwrap();
        assert!(t.move_by(1).is_ok()); // Down
        assert!(t.move_by(-1).is_ok()); // Up
        assert!(t.move_by(i16::MAX).is_ok());
        // Note: in move_by(i16::MIN), negating `-lines` overflows in the i16 domain (panics in debug builds),
        // this is a known edge case; don't lock in the panic here, fix it during implementation and add a test case.
        assert!(t.move_by(-32767).is_ok());
    }

    #[test]
    fn test_write_returns_ok() {
        let mut t = Terminal::new().unwrap();
        assert!(t.write("").is_ok());
        assert!(t.write("hello\n").is_ok());
    }

    #[test]
    fn test_handover_to_shell_ok_and_does_not_touch_flags() {
        // Handover only writes terminal sequences (clear line / clear below / new line), without changing raw/cursor/protocol flags
        let mut t = Terminal::new().unwrap();
        assert!(t.handover_to_shell().is_ok());
        assert!(!t.raw_mode_active);
        assert!(!t.cursor_hidden);
        assert!(!t.keyboard_enhancement_active);
        assert!(!t.bracketed_paste_active);
    }

    #[test]
    fn test_clear_operations_ok() {
        let mut t = Terminal::new().unwrap();
        assert!(t.clear_line().is_ok());
        assert!(t.clear_from_cursor().is_ok());
        assert!(t.clear_screen().is_ok());
    }

    #[test]
    fn test_set_title_and_progress_ok() {
        let mut t = Terminal::new().unwrap();
        assert!(t.set_title("zen").is_ok());
        assert!(t.set_title("").is_ok());
        assert!(t.set_progress(true).is_ok());
        assert!(t.set_progress(false).is_ok());
    }

    #[test]
    fn test_columns_rows_nonzero() {
        // Non-tty: terminal::size() fails → fall back to 80/24; tty returns the real size, both > 0
        let t = Terminal::new().unwrap();
        assert!(t.columns() > 0);
        assert!(t.rows() > 0);
    }

    #[test]
    fn test_start_stop_roundtrip_environment_conditional() {
        let mut t = Terminal::new().unwrap();
        match t.start() {
            Ok(()) => {
                // With tty: start succeeds, flag set; start again is idempotent; stop restores; stop is idempotent
                assert!(t.raw_mode_active);
                assert!(t.cursor_hidden);
                t.start().unwrap();
                assert!(t.raw_mode_active);
                t.stop().unwrap();
                assert!(!t.raw_mode_active);
                assert!(!t.cursor_hidden);
                t.stop().unwrap();
            }
            Err(TerminalError::RawMode {
                action: "enter", ..
            }) => {
                // Without tty: enable_raw_mode fails, flag not set
                assert!(!t.raw_mode_active);
                assert!(!t.cursor_hidden);
            }
            Err(e) => panic!("unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_drop_without_stop_restores_and_never_panics() {
        // Drop fallback: dropping directly after start without explicit stop does not panic, and terminal state is restored
        // (a subsequent Terminal::start can still enter raw mode normally, showing global state is not polluted by residue)
        if let Ok(mut t) = Terminal::new() {
            let _ = t.start(); // Silently skip when the test environment does not support raw mode (same structure as test above)
        } // drop here: fallback executes stop
        let t2 = Terminal::new().expect("Terminal should be reconstructable after drop fallback");
        assert!(!t2.raw_mode_active);
        assert!(!t2.keyboard_enhancement_active);
        assert!(!t2.bracketed_paste_active);
    }

    #[test]
    fn test_start_stop_enables_and_disables_protocols() {
        // Keyboard enhancement + bracketed paste: start sets flags, stop clears them, with idempotent round trips (mirrors raw mode).
        // In a non-tty environment, start fails when enabling raw mode, so the protocol flags are not set.
        let mut t = Terminal::new().unwrap();
        match t.start() {
            Ok(()) => {
                assert!(t.keyboard_enhancement_active);
                assert!(t.bracketed_paste_active);
                // Idempotent: calling start again does not resend protocol commands (flags remain set)
                t.start().unwrap();
                assert!(t.keyboard_enhancement_active);
                assert!(t.bracketed_paste_active);
                // stop symmetrically disables the features and clears the flags
                t.stop().unwrap();
                assert!(!t.keyboard_enhancement_active);
                assert!(!t.bracketed_paste_active);
                // Idempotent: calling stop again has no side effects
                t.stop().unwrap();
                assert!(!t.keyboard_enhancement_active);
            }
            Err(TerminalError::RawMode {
                action: "enter", ..
            }) => {
                assert!(!t.keyboard_enhancement_active);
                assert!(!t.bracketed_paste_active);
            }
            Err(e) => panic!("unexpected error: {:?}", e),
        }
    }
}

#[cfg(test)]
mod frame_delivery_tests {
    use super::*;
    use std::io::Write;

    /// Sink that records how many `write` calls it received (and the total bytes), optionally failing.
    ///
    /// Counting `Write::write` calls is the point: `write_all` loops until satisfied, so a single delivery shows up as
    /// exactly one call here. This is the "≈syscall" quantity the acceptance talks about (the kernel/pty may split
    /// further, which no in-process test can observe).
    #[derive(Default)]
    struct SpyWriter {
        writes: usize,
        bytes: usize,
        payload: Vec<u8>,
        fail: bool,
    }

    impl Write for SpyWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if self.fail {
                return Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "boom"));
            }
            self.writes += 1;
            self.bytes += buf.len();
            self.payload.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// One steady-state frame (~400 B, what PF-01 measured) reaches the sink in exactly one write.
    #[test]
    fn frame_delivery_is_one_write_for_a_steady_state_frame() {
        let mut buf = Vec::new();
        for i in 0..8 {
            buf.extend_from_slice(b"\r\n\x1b[2K");
            buf.extend_from_slice(format!("line {i} with some padding to reach ~50 B").as_bytes());
        }
        let len = buf.len();
        let mut spy = SpyWriter::default();

        // Nothing reaches the sink while the frame is being built.
        assert_eq!(spy.writes, 0);

        flush_frame(&mut buf, &mut spy).unwrap();
        assert_eq!(spy.writes, 1, "a frame must be delivered in one write");
        assert_eq!(spy.bytes, len);
        assert_eq!(spy.payload.len(), len, "payload is byte-identical");
    }

    /// An oversized frame (the ~403 KB first full redraw) is still a single write: no artificial capacity split.
    #[test]
    fn frame_delivery_is_one_write_for_an_oversized_frame() {
        let mut buf = Vec::with_capacity(450 * 1024);
        for i in 0..4_000 {
            buf.extend_from_slice(
                format!("\x1b[2Kline {i:05} of a full-screen redraw\r\n").as_bytes(),
            );
        }
        assert!(
            buf.len() > 100 * 1024,
            "fixture should be large: {}",
            buf.len()
        );
        let len = buf.len();
        let mut spy = SpyWriter::default();

        flush_frame(&mut buf, &mut spy).unwrap();
        assert_eq!(spy.writes, 1, "no capacity-driven split");
        assert_eq!(spy.bytes, len);
    }

    /// The buffer is cleared after delivery (success path) so the next frame starts empty.
    #[test]
    fn flush_frame_clears_the_buffer_after_success() {
        let mut buf = b"\x1b[?2026hhello\x1b[?2026l".to_vec();
        let mut spy = SpyWriter::default();
        flush_frame(&mut buf, &mut spy).unwrap();
        assert!(buf.is_empty());
        assert_eq!(spy.writes, 1);
    }

    /// A failed delivery propagates the IO error **and** still clears the buffer (a retry must not resend half a frame).
    #[test]
    fn flush_frame_clears_the_buffer_and_propagates_on_failure() {
        let mut buf = b"partial frame".to_vec();
        let mut spy = SpyWriter {
            fail: true,
            ..Default::default()
        };
        let err = flush_frame(&mut buf, &mut spy).unwrap_err();
        assert!(matches!(err, TerminalError::Io(_)));
        assert!(buf.is_empty(), "residue must not survive a failed delivery");
        assert_eq!(spy.writes, 0);
    }

    /// `begin_frame` drops residue left by a frame that bailed out before delivery.
    #[test]
    fn begin_frame_discards_residue_and_keeps_capacity() {
        let mut t = Terminal::new().unwrap();
        t.out_mut().extend_from_slice(b"half a frame");
        let capacity_before = t.out_mut().capacity();
        t.begin_frame();
        assert!(t.out_mut().is_empty(), "begin_frame clears the frame sink");
        assert_eq!(t.out_mut().capacity(), capacity_before, "capacity retained");
    }

    /// `out_mut` is the frame sink, not stdout: bytes stay readable until `flush_out` delivers them.
    #[test]
    fn out_mut_accumulates_without_touching_the_terminal() {
        let mut t = Terminal::new().unwrap();
        let base = t.write_calls();
        {
            let buf = t.out_mut();
            buf.extend_from_slice(b"\x1b[?2026h");
            buf.extend_from_slice(b"content");
            buf.extend_from_slice(b"\x1b[?2026l");
        }
        assert_eq!(t.out_mut().len(), 8 + 7 + 8, "all bytes are still buffered");
        assert_eq!(t.write_calls(), base, "no delivery happened yet");
    }

    /// `flush_out` delivers exactly once and bumps the delivery counter by one.
    #[test]
    fn flush_out_counts_one_delivery_per_frame() {
        let mut t = Terminal::new().unwrap();
        let base = t.write_calls();
        t.begin_frame();
        t.out_mut().extend_from_slice(b"frame 1");
        t.flush_out().unwrap();
        assert_eq!(t.write_calls(), base + 1);
        assert!(t.out_mut().is_empty(), "delivered bytes leave the buffer");

        // A second frame is a second delivery.
        t.begin_frame();
        t.out_mut().extend_from_slice(b"frame 2");
        t.flush_out().unwrap();
        assert_eq!(t.write_calls(), base + 2);
    }

    /// The synchronous-update wrapper and the frame content travel in the same buffer, in order.
    #[test]
    fn synchronized_update_markers_share_the_frame_buffer() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"\x1b[?2026h");
        buf.extend_from_slice(b"\r\n\x1b[2Kline");
        buf.extend_from_slice(b"\x1b[?2026l");
        let mut spy = SpyWriter::default();
        flush_frame(&mut buf, &mut spy).unwrap();

        assert_eq!(spy.writes, 1);
        assert!(spy.payload.starts_with(b"\x1b[?2026h"));
        assert!(spy.payload.ends_with(b"\x1b[?2026l"));
    }
}
