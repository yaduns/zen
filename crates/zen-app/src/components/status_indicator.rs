// REQ-APP-11 (pi-358): status-indicator — status line at the bottom of the chat area (working/retry/compaction/branch summary).
// Corresponds to the original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/status-indicator.ts.
// pi's Loader subclass family (different colors/copy) is implemented in zen as a "Kind discriminator + Loader delegation" combination;
// The retry countdown goes through CountdownTimer (elapsed-driven, render-frame tick, see its file header).

use crate::components::countdown_timer::CountdownTimer;
use crate::components::keybinding_hints::key_text;
use crate::keybindings::AppAction;
use zen_tui::components::loader::Loader;
use zen_tui::components::Component;
use zen_tui::keybindings::KeybindingsManager;
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{ColorToken, Theme};

/// Status-line indicator kind (pi StatusIndicatorKind). Pure discriminator enum, not serialized.
pub enum StatusIndicatorKind {
    /// Agent busy (spinner + task message).
    Working,
    /// Request retry (countdown copy).
    Retry,
    /// Context compression in progress.
    Compaction,
    /// Branch summary generation in progress.
    BranchSummary,
}

/// Compaction trigger reason (selects the message text; pure discriminant enum). derive side: StatusSpec payload requires Debug+Clone
/// (Copy, then pass by value via `*reason`, avoiding a clone); one-to-one mapping with zen-core `CompactionReason` via From
/// （T-AP-04-3）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionStatusReason {
    /// User-initiated /compact.
    Manual,
    /// Automatic compaction when threshold reached.
    Threshold,
    /// Forced compaction on context overflow.
    Overflow,
}

/// Status line at the bottom of the chat area: spinner + single-line message; animation frames and styles fixed to working semantics (accent spinner + muted text).
/// T-AP-11-1 creates the corresponding instance for each phase ("agent busy/retry/compaction/branch summary") and draws it into the message area.
///
/// Advances the internal countdown within the render frame (Retry only); cancellation is routed via key handling by the upper layer (T-AP-11-1), this component only displays.
/// Retry parameters for the retry status line (held by the retry status line; used to refresh the "N seconds" copy each render frame).
struct RetryState {
    attempt: u32,
    max_attempts: u32,
    retry_key: String,
}

pub struct StatusIndicator {
    kind: StatusIndicatorKind,
    loader: Loader,
    countdown: Option<CountdownTimer>,
    retry_state: Option<RetryState>,
    last_retry_secs: u32,
    started: bool,
}

impl StatusIndicator {
    /// Working status line (message fixed / updated via set_message); consumer T-AP-11-1.
    pub fn working(message: impl Into<String>, theme: &Theme) -> Self {
        Self::new(StatusIndicatorKind::Working, theme, message.into(), None)
    }

    /// Retry countdown status line: "Retrying (n/m) in Ns... (hint key to cancel)",
    /// Holds a CountdownTimer internally to refresh the message every second. Consumer: T-AP-11-1 (LLM call retry).
    pub fn retry(
        attempt: u32,
        max_attempts: u32,
        delay_ms: u64,
        kb: &KeybindingsManager<AppAction>,
        theme: &Theme,
    ) -> Self {
        // Self-reference avoidance: tick callback never reaches loader — render frames read countdown remaining seconds and refresh the message
        // (see render; CountdownTimer callback is a no-op, on expiry only sets done internally).
        let retry_key = key_text(kb, AppAction::Interrupt);
        let initial = delay_ms.div_ceil(1000) as u32;
        let mut indicator = Self::new(
            StatusIndicatorKind::Retry,
            theme,
            retry_message(attempt, max_attempts, initial, &retry_key),
            None,
        );
        let mut countdown = CountdownTimer::new(delay_ms, Box::new(|_| {}), Box::new(|| {}));
        countdown.start();
        indicator.countdown = Some(countdown);
        indicator.retry_state = Some(RetryState {
            attempt,
            max_attempts,
            retry_key,
        });
        indicator.last_retry_secs = initial;
        indicator
    }

    /// Compressed status line (reason determines the wording). Consumer: T-AP-11-1.
    pub fn compaction(
        reason: CompactionStatusReason,
        kb: &KeybindingsManager<AppAction>,
        theme: &Theme,
    ) -> Self {
        let cancel_hint = format!("({} to cancel)", key_text(kb, AppAction::Interrupt));
        let label = match reason {
            CompactionStatusReason::Manual => format!("Compacting context... {cancel_hint}"),
            CompactionStatusReason::Threshold => {
                format!("Auto-compacting... {cancel_hint}")
            }
            CompactionStatusReason::Overflow => {
                format!("Context overflow detected, auto-compacting... {cancel_hint}")
            }
        };
        Self::new(StatusIndicatorKind::Compaction, theme, label, None)
    }

    /// Branch summary status line. Consumer: T-AP-11-1 (during summary generation after fork).
    pub fn branch_summary(kb: &KeybindingsManager<AppAction>, theme: &Theme) -> Self {
        let label = format!(
            "Summarizing branch... ({} to cancel)",
            key_text(kb, AppAction::Interrupt)
        );
        Self::new(StatusIndicatorKind::BranchSummary, theme, label, None)
    }

    fn new(kind: StatusIndicatorKind, theme: &Theme, message: String, _extra: Option<()>) -> Self {
        let spinner_style = match kind {
            StatusIndicatorKind::Retry => theme.style(ColorToken::Warning, None, &[]),
            _ => theme.style(ColorToken::Accent, None, &[]),
        };
        let message_style = Style::fg(theme.fg(ColorToken::Muted));
        let mut loader = Loader::new(spinner_style, message_style, message);
        loader.start();
        StatusIndicator {
            kind,
            loader,
            countdown: None,
            retry_state: None,
            last_retry_secs: 0,
            started: true,
        }
    }

    /// Category accessor.
    pub fn kind(&self) -> &StatusIndicatorKind {
        &self.kind
    }

    /// Update the message wording (Alignment: fixed message state changes wording).
    pub fn set_message(&mut self, message: impl Into<String>) {
        self.loader.set_message(message);
    }

    /// Stop countdown and animation (cleanup exclusive to Retry); after stopping, rendering no longer advances frames.
    pub fn dispose(&mut self) {
        if let Some(cd) = &mut self.countdown {
            cd.dispose();
        }
        self.retry_state = None;
        self.loader.stop();
        self.started = false;
    }
}

/// Retry wording (render frames refresh by remaining seconds).
fn retry_message(attempt: u32, max_attempts: u32, seconds: u32, retry_key: &str) -> String {
    format!("Retrying ({attempt}/{max_attempts}) in {seconds}s... ({retry_key} to cancel)")
}

impl Component for StatusIndicator {
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        // The render frame is the countdown driver (seconds on the Retry status line are refreshed here)
        if let Some(cd) = &mut self.countdown {
            cd.tick();
            if let Some(state) = &self.retry_state {
                let seconds = cd.remaining_seconds();
                if seconds != self.last_retry_secs {
                    self.last_retry_secs = seconds;
                    let msg =
                        retry_message(state.attempt, state.max_attempts, seconds, &state.retry_key);
                    self.loader.set_message(msg);
                }
            }
        }
        self.loader.render(buf, x, y, width, height)
    }

    fn handle_key(&mut self, _key: &crossterm::event::KeyEvent) -> bool {
        // Cancellation is routed by the upper layer (T-AP-11-1) to the agent's CancellationToken; this component consumes no keypress.
        false
    }
}

/// Idle placeholder status line — two blank lines fill the width to keep the bottom height stable. Consumer: T-AP-11-1.
pub struct IdleStatus;

impl Component for IdleStatus {
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        let style = Style::default();
        let painted = 2u16.min(height);
        for row in 0..painted {
            buf.fill(x, y + row, width, 1, &style);
        }
        painted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keybindings::app_default_bindings;
    use std::collections::HashMap;

    fn theme() -> &'static Theme {
        zen_tui::theme::builtin_dark_theme()
    }

    fn kb() -> KeybindingsManager<AppAction> {
        KeybindingsManager::new(app_default_bindings(), HashMap::new())
    }

    fn flush_text(buf: &mut ScreenBuffer) -> String {
        // flush output contains crossterm diff commands (MoveTo/SetColors/Print per cell);
        // here control bytes are stripped to restore "visible text" (printable chars, order preserved) for contains assertions
        let mut out = Vec::new();
        buf.flush(&mut out).unwrap();
        crate::components::strip_ansi(&String::from_utf8_lossy(&out))
            .chars()
            .filter(|c| !c.is_control())
            .collect()
    }

    #[test]
    fn working_renders_message() {
        let mut status = StatusIndicator::working("Working on it", theme());
        let mut buf = ScreenBuffer::new(40, 3);
        let rows = status.render(&mut buf, 0, 0, 40, 3);
        assert!(rows >= 1);
        assert!(flush_text(&mut buf).contains("Working on it"));
    }

    #[test]
    fn retry_shows_attempt_and_interrupt_hint() {
        let kb = kb();
        let mut status = StatusIndicator::retry(2, 3, 5000, &kb, theme());
        let mut buf = ScreenBuffer::new(60, 3);
        status.render(&mut buf, 0, 0, 60, 3);
        let text = flush_text(&mut buf);
        assert!(text.contains("Retrying (2/3)"));
        assert!(text.contains("esc"), "interrupt hint missing in: {text}");
    }

    #[test]
    fn compaction_messages_by_reason() {
        let kb = kb();
        let manual = StatusIndicator::compaction(CompactionStatusReason::Manual, &kb, theme());
        let overflow = StatusIndicator::compaction(CompactionStatusReason::Overflow, &kb, theme());
        let mut buf = ScreenBuffer::new(80, 3);
        let mut m = manual;
        m.render(&mut buf, 0, 0, 80, 3);
        assert!(flush_text(&mut buf).contains("Compacting context"));
        let mut buf2 = ScreenBuffer::new(80, 3);
        let mut o = overflow;
        o.render(&mut buf2, 0, 0, 80, 3);
        assert!(flush_text(&mut buf2).contains("Context overflow detected"));
    }

    #[test]
    fn branch_summary_renders() {
        let kb = kb();
        let mut status = StatusIndicator::branch_summary(&kb, theme());
        let mut buf = ScreenBuffer::new(80, 3);
        status.render(&mut buf, 0, 0, 80, 3);
        assert!(flush_text(&mut buf).contains("Summarizing branch"));
    }

    #[test]
    fn dispose_stops_animation_and_countdown() {
        let kb = kb();
        let mut status = StatusIndicator::retry(1, 3, 5000, &kb, theme());
        assert!(status.countdown.as_ref().unwrap().is_running());
        status.dispose();
        assert!(!status.started);
        assert!(!status.countdown.as_ref().unwrap().is_running());
    }

    #[test]
    fn idle_renders_two_blank_rows() {
        let mut idle = IdleStatus;
        let mut buf = ScreenBuffer::new(20, 2);
        assert_eq!(idle.render(&mut buf, 0, 0, 20, 2), 2);
        assert!(flush_text(&mut buf).contains(' '));
    }

    #[test]
    fn kind_accessor() {
        let s = StatusIndicator::working("x", theme());
        assert!(matches!(s.kind(), StatusIndicatorKind::Working));
    }

    #[test]
    fn spinner_advances_on_long_lived_instance() {
        // Contract: animation frames are derived from the clock crossing frames — a second render of the same instance (reused across frames) must move off the first frame.
        // If the caller rebuilds the instance every frame (started_at re-anchored each frame → elapsed≈0), this test fails.
        let mut status = StatusIndicator::working("Working on it", theme());
        // Just constructed: elapsed≈0 → first frame ⠋
        let mut buf1 = ScreenBuffer::new(40, 3);
        status.render(&mut buf1, 0, 0, 40, 3);
        let first = flush_text(&mut buf1);
        assert!(first.contains('⠋'), "first frame should be ⠋, got {first}");
        // Long-lived: 250ms > 80ms/frame → (250/80)%10 = 3 → not the first frame (a slow machine only pushes it later; always non-zero)
        std::thread::sleep(std::time::Duration::from_millis(250));
        let mut buf2 = ScreenBuffer::new(40, 3);
        status.render(&mut buf2, 0, 0, 40, 3);
        let second = flush_text(&mut buf2);
        assert!(
            !second.contains('⠋'),
            "spinner stuck on first frame (clock not advancing): {second}"
        );
    }
}
