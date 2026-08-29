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
    out: std::io::Stdout,  // target for writing crossterm commands (tail of queue!/execute!)
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

    /// Expose the stdout write end so the main loop can write out ScreenBuffer::flush diff commands. Internal crate seam, not public.
    /// Consumer: this task's `render_frame` (sole consumer).
    pub(crate) fn out_mut(&mut self) -> &mut std::io::Stdout {
        &mut self.out
    }
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
