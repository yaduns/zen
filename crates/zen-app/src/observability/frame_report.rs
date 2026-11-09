// PF-01 (REQ-TUI-01 observability): frame-metrics summary output — prints the metrics collected by `zen-tui` to stderr when interactive mode exits.
// Plan doc: docs/performance-optimization-plan.md §6.2; task notes: tasks/PF-01-implemented.md
//
// Separation of concerns: `zen-tui` owns the counters and the formatting (`FrameStats::summary`), the
// application layer owns *whether and where* to print. Keeping the print out of zen-tui means print mode
// (whose stdout is taken over by `output::raw_stdout`) never gains a stray line.
use zen_tui::metrics::FrameStats;

/// Print the frame-metrics summary to **stderr** on interactive shutdown.
///
/// Prints nothing unless the accumulator was enabled at loop construction (`ZEN_TUI_TRACE_FRAMES=1`), so a
/// normal exit stays byte-for-byte unchanged. stderr is deliberate: stdout is redirected or taken over in both
/// modes (print mode's raw stdout, an interactive session piped through `tee`), and `diagnostics` already
/// reports to stderr.
///
/// Consumer: `modes/interactive/mod.rs` (UI task, right after `tui.stop()` restored the terminal).
pub fn report_frame_stats(stats: &FrameStats) {
    if let Some(text) = frame_summary_text(stats) {
        eprintln!("{text}");
    }
}

/// Pure decision behind [`report_frame_stats`]: the text to print, or `None` when the accumulator is off.
///
/// Split out so the "off ⇒ silent" contract is testable without capturing a process-global stderr: the
/// accumulator's `enabled` flag (a snapshot taken at loop construction) decides, not the live environment.
///
/// Consumer: [`report_frame_stats`] + tests.
pub(crate) fn frame_summary_text(stats: &FrameStats) -> Option<String> {
    stats.enabled().then(|| stats.summary())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn disabled_accumulator_yields_no_report() {
        let stats = FrameStats::new_with_enabled(Duration::from_millis(16), false);
        assert!(
            frame_summary_text(&stats).is_none(),
            "a loop constructed with metrics off must not print anything on exit"
        );
        report_frame_stats(&stats); // must be a no-op, not a panic
    }

    #[test]
    fn enabled_accumulator_yields_the_summary_text() {
        let stats = FrameStats::new_with_enabled(Duration::from_millis(16), true);
        let text = frame_summary_text(&stats).expect("enabled accumulator must produce a report");
        // Nothing recorded yet → the summary must still explain how to collect data instead of printing an
        // empty block (the "no samples" contract of FrameStats::summary).
        assert!(
            text.contains("no frame recorded"),
            "unexpected report: {text}"
        );
        assert!(
            text.contains(zen_tui::metrics::TRACE_ENV),
            "unexpected report: {text}"
        );
    }
}
