// PF-01 (REQ-TUI-01 observability): frame metrics — per-frame draw/flush cost, bytes written to the terminal, the written content-row span,
// the line-stream total and the fixed region's actual row count; the process-wide accumulator reports p50/p95/p99 and the exit summary.
// Plan doc: docs/performance-optimization-plan.md §PF-01; task notes: tasks/PF-01-implemented.md
//
// Design constraints (rulings from the task notes):
// - Off by default (`ZEN_TUI_TRACE_FRAMES`); when off it takes no `Instant`, allocates nothing and samples nothing — the overhead is one `bool` check.
// - The accumulator uses `Mutex<StatsInner>` with a short critical section (the plan doc originally said "atomic counters"; ring sampling cannot be expressed with a single atomic, registered as a deviation).
// - Frame numbers are assigned by the accumulator inside `record` (a `FrameProbe` is created per frame, so its own counter is always 0).
// - `repaint_span` uses `Option<(usize, usize)>` instead of `Range` (`Range` is only `Clone`, not `Copy`, which would break `FrameMetrics: Copy`).
use std::collections::VecDeque;
use std::io::{self, Write};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Environment variable that turns frame metrics on (`1` / `true` / `yes`; anything else is off).
///
/// Diagnostics switch: the interactive UI is started by a human, so the collection path is opt-in and
/// must never cost anything when nobody asked for it. Consumer: [`frame_trace_enabled`] / [`FrameStats::new`].
pub const TRACE_ENV: &str = "ZEN_TUI_TRACE_FRAMES";

/// Default per-frame cost budget: 16 ms, i.e. one 60 fps display refresh (`draw + flush` above it counts as over budget).
///
/// Unrelated to the repaint cadence: PF-03 removed the 16 ms ticker, and the animation cadence is
/// [`crate::app_loop::DEFAULT_ANIMATION_INTERVAL`] (80 ms). Do not move this value when the cadence changes.
pub const DEFAULT_BUDGET: Duration = Duration::from_millis(16);

/// Default ring capacity per series (≈ 128 KB for three `Duration` series: ~22 min at 6 fps, ~2.3 min at 60 fps).
const DEFAULT_SAMPLE_CAPACITY: usize = 8192;

/// Ask once per process whether frame metrics are enabled.
///
/// The interactive entry point asks this while constructing the loop, so "the user did not enable
/// diagnostics" costs one environment read per process and nothing per frame afterwards. The value is
/// cached in a `OnceLock` (same process never re-reads the environment); invalid values are treated as off
/// instead of being guessed at.
///
/// Consumers: [`FrameStats::new`] (stores the answer in [`FrameStats::enabled`]).
pub fn frame_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| frame_trace_enabled_from(std::env::var(TRACE_ENV).ok().as_deref()))
}

/// Pure form of the environment-value judgement (`None` = variable unset ⇒ off).
///
/// Exists because [`frame_trace_enabled`] caches in a `OnceLock` and therefore cannot be flipped inside a
/// test process: the tests drive this function instead, and `frame_trace_enabled` delegates to it so both
/// paths can never disagree.
///
/// Consumer: unit tests + [`frame_trace_enabled`].
pub(crate) fn frame_trace_enabled_from(raw: Option<&str>) -> bool {
    match raw {
        Some(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes")
        }
        None => false,
    }
}

/// One frame's measurements, filled by the frame loop right after the flush and consumed by
/// [`FrameStats::record`] (plus test assertions and downstream acceptance checks).
///
/// Pure runtime data: not serialized, `Copy` so that recording/asserting never has to clone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameMetrics {
    /// Frame sequence number starting at 1. **Assigned by the accumulator** while recording
    /// ([`FrameStats::record`] overwrites whatever the caller put here), so frames that bail out before
    /// recording never consume a number and `frame_no` always matches [`FrameStats::frames`].
    pub frame_no: u64,
    /// Cost of `canvas.clear()` + `draw(&mut canvas)`: building the frame content (markdown, components,
    /// line stream). Excludes terminal size sync and excludes the flush.
    pub draw: Duration,
    /// Cost of the diff + terminal write (`ScrollbackCanvas::flush` / `ScreenBuffer::flush`).
    pub flush: Duration,
    /// Bytes handed to the terminal this frame (counted by [`CountingWriter`]).
    ///
    /// **Not comparable across render paths**: the cell-level path (`TuiLoop::run`) carries crossterm
    /// cursor/style sequences, while the scrollback path carries `\r\n` advancement plus one `\x1b[2K` per
    /// written line. Byte-per-appended-line assertions only make sense on the scrollback path.
    pub bytes_written: u64,
    /// Content rows actually written this frame, as an inclusive `[first, last]` span.
    ///
    /// `None` when nothing changed (and on the cell-level path, which has no line span). Defined as
    /// **rows written**, not "visible repaint window": a full redraw writes every line, so it reports
    /// `[0, line_total - 1]`.
    pub repaint_span: Option<(usize, usize)>,
    /// Total content rows of this frame's line stream (`ScrollbackCanvas::content_height`; on the cell-level
    /// path, the buffer row count).
    pub line_total: usize,
    /// Rows of the bottom fixed region (status + editor + dropdown + footer) that were **actually pushed**
    /// into the line stream — not the theoretical `fixed_h` upper bound; unpainted trailing rows do not count.
    ///
    /// Before PF-06 this is the tail length of the line stream; after PF-06 it becomes the height of the
    /// bottom anchored region (`BottomRegion`).
    pub line_fixed: usize,
}

impl FrameMetrics {
    /// Whole-frame cost (`draw + flush`), the quantity the performance plan's p50/p95 thresholds refer to.
    ///
    /// Consumer: [`FrameStats::record`] (budget check + percentile sampling) and [`FrameStats::summary`].
    pub fn total(&self) -> Duration {
        self.draw + self.flush
    }
}

/// Process-wide frame accumulator: counters, ring-sampled percentiles and the human-readable summary.
///
/// Shared through `Arc` between the frame loop (writer) and the application layer (exit summary), hence the
/// interior mutability. The critical section only pushes into a ring and adds to counters.
pub struct FrameStats {
    budget: Duration,
    enabled: bool,
    capacity: usize,
    inner: Mutex<StatsInner>,
}

/// Mutable part of [`FrameStats`]: kept private so callers cannot hold the lock across frames.
#[derive(Default)]
struct StatsInner {
    next_frame_no: u64,
    frames: u64,
    over_budget: u64,
    bytes_total: u64,
    line_total_sum: u64,
    line_fixed_sum: u64,
    totals: VecDeque<Duration>,
    draws: VecDeque<Duration>,
    flushes: VecDeque<Duration>,
}

impl FrameStats {
    /// Create the accumulator for a run, reading [`frame_trace_enabled`] once and storing it as
    /// [`Self::enabled`] (so the per-frame path never re-reads the environment).
    ///
    /// `budget` is the single-frame cost budget used for the over-budget counter: it is **independent of the
    /// tick interval** — it stays 16 ms even after the tick cadence changes (PF-03), otherwise the
    /// over-budget ratio would stop meaning anything.
    ///
    /// Consumer: `app_loop.rs` (`TuiLoop::new`, which wraps the result in an `Arc`).
    pub fn new(budget: Duration) -> FrameStats {
        Self::with_options(budget, frame_trace_enabled(), DEFAULT_SAMPLE_CAPACITY)
    }

    /// Explicit-switch constructor for tests: needs to exercise both the enabled and disabled paths without
    /// touching the process environment (which [`frame_trace_enabled`] caches permanently).
    ///
    /// Consumer: unit tests of this module / `app_loop.rs`, and `zen-app`'s frame-report tests (which cannot see
    /// `cfg(test)` items of a dependency, hence a real `pub` item documented as a test seam).
    pub fn new_with_enabled(budget: Duration, enabled: bool) -> FrameStats {
        Self::with_options(budget, enabled, DEFAULT_SAMPLE_CAPACITY)
    }

    /// Constructor with an explicit ring capacity, so tests can prove the "oldest sample is overwritten"
    /// behaviour without pushing 8192 frames.
    ///
    /// Consumer: unit tests of this module (test-only, hence `cfg(test)`).
    #[cfg(test)]
    pub(crate) fn new_with_capacity(
        budget: Duration,
        enabled: bool,
        capacity: usize,
    ) -> FrameStats {
        Self::with_options(budget, enabled, capacity.max(1))
    }

    /// Shared constructor; `capacity` is clamped by callers.
    fn with_options(budget: Duration, enabled: bool, capacity: usize) -> FrameStats {
        FrameStats {
            budget,
            enabled,
            capacity,
            inner: Mutex::new(StatsInner::default()),
        }
    }

    /// Whether this accumulator records anything (a snapshot of [`frame_trace_enabled`] taken at construction).
    ///
    /// Consumers: [`FrameProbe::start`] (skip timing entirely when off) and the application-layer summary
    /// reporter (print nothing when off).
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// The per-frame budget used for over-budget counting (independent of the tick interval, see [`Self::new`]).
    pub fn budget(&self) -> Duration {
        self.budget
    }

    /// Record one frame: assigns the frame number, accumulates counters and pushes one sample per series.
    ///
    /// `frame_no` is **overwritten** here (whatever the caller put in it is ignored) so that numbering stays
    /// consistent with [`Self::frames`] — only recorded frames consume a number. No-op when disabled.
    ///
    /// Consumer: [`FrameProbe::finish`].
    pub fn record(&self, metrics: &mut FrameMetrics) {
        if !self.enabled {
            return;
        }
        let mut inner = self.lock();
        inner.next_frame_no += 1;
        metrics.frame_no = inner.next_frame_no;
        inner.frames += 1;
        let total = metrics.total();
        if total > self.budget {
            inner.over_budget += 1;
        }
        inner.bytes_total += metrics.bytes_written;
        inner.line_total_sum += metrics.line_total as u64;
        inner.line_fixed_sum += metrics.line_fixed as u64;
        push_sample(&mut inner.totals, total, self.capacity);
        push_sample(&mut inner.draws, metrics.draw, self.capacity);
        push_sample(&mut inner.flushes, metrics.flush, self.capacity);
    }

    /// Number of frames recorded so far.
    ///
    /// **Window assertions must use the difference**, e.g. `let base = stats.frames(); …; assert!(stats.frames() - base <= 750)`
    /// — this is a process-cumulative counter, so `assert!(stats.frames() <= 750)` would fail on any long session.
    pub fn frames(&self) -> u64 {
        self.lock().frames
    }

    /// Total bytes written to the terminal across recorded frames (the core metric for "did the fixed region
    /// get rewritten?" checks: a steady-state frame must contribute 0).
    pub fn bytes_total(&self) -> u64 {
        self.lock().bytes_total
    }

    /// Frames whose `draw + flush` **exceeded** the budget (equal-to-budget is not counted).
    pub fn over_budget(&self) -> u64 {
        self.lock().over_budget
    }

    /// Multi-line human-readable summary for the exit report.
    ///
    /// Frame cost is reported as `draw + flush` p50/p95/p99 (the same quantity the performance plan's
    /// thresholds use), with the `draw` / `flush` medians listed separately for locating a regression.
    /// Returns a single explanatory line — never an empty string, never a panic — when nothing was sampled,
    /// so tests can call it on a disabled accumulator; whether it is *printed* is the caller's decision.
    ///
    /// Consumer: `zen-app` `observability::frame_report::report_frame_stats`.
    pub fn summary(&self) -> String {
        let inner = self.lock();
        if inner.frames == 0 {
            return format!("frame metrics: no frame recorded (set {TRACE_ENV}=1 to collect)");
        }
        let frames = inner.frames;
        let over_ratio = inner.over_budget as f64 * 100.0 / frames as f64;
        let totals = sorted(&inner.totals);
        let draws = sorted(&inner.draws);
        let flushes = sorted(&inner.flushes);
        let mut out = String::new();
        out.push_str(&format!(
            "frame metrics ({TRACE_ENV}=1): {frames} frames, {} over budget {} ({:.1}%)\n",
            inner.over_budget,
            format_ms(self.budget),
            over_ratio
        ));
        out.push_str(&format!(
            "  frame total (draw+flush): p50 {}  p95 {}  p99 {}\n",
            format_ms(percentile(&totals, 50.0)),
            format_ms(percentile(&totals, 95.0)),
            format_ms(percentile(&totals, 99.0)),
        ));
        out.push_str(&format!(
            "  draw median {}  flush median {}  (samples {})\n",
            format_ms(percentile(&draws, 50.0)),
            format_ms(percentile(&flushes, 50.0)),
            totals.len(),
        ));
        out.push_str(&format!(
            "  bytes written: total {}, avg {} B/frame\n",
            inner.bytes_total,
            inner.bytes_total / frames,
        ));
        out.push_str(&format!(
            "  lines: avg {:.1} total, avg {:.1} fixed",
            inner.line_total_sum as f64 / frames as f64,
            inner.line_fixed_sum as f64 / frames as f64,
        ));
        out
    }

    /// Number of samples currently held per series (≤ capacity). Test-only seam for the ring bound.
    #[cfg(test)]
    pub(crate) fn sample_len(&self) -> usize {
        self.lock().totals.len()
    }

    /// Lock helper: metrics must never poison the UI loop, so a poisoned lock is recovered rather than
    /// propagated (the payload is plain counters; there is no invariant a panic could have broken).
    fn lock(&self) -> std::sync::MutexGuard<'_, StatsInner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Push one sample into a bounded ring, dropping the oldest when full.
fn push_sample(series: &mut VecDeque<Duration>, value: Duration, capacity: usize) {
    if series.len() == capacity {
        series.pop_front();
    }
    series.push_back(value);
}

/// Sorted copy of a series (percentiles need an ordered slice; the ring itself stays insertion-ordered).
fn sorted(series: &VecDeque<Duration>) -> Vec<Duration> {
    let mut values: Vec<Duration> = series.iter().copied().collect();
    values.sort_unstable();
    values
}

/// Nearest-rank percentile over an already sorted slice (`p` in `0.0..=100.0`); empty ⇒ `Duration::ZERO`.
fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[index]
}

/// Milliseconds with two decimals — readable in a terminal summary without pulling a formatting crate.
fn format_ms(value: Duration) -> String {
    format!("{:.2}ms", value.as_secs_f64() * 1000.0)
}

/// Per-frame timing probe shared by both render paths (`TuiLoop::run` and `TuiLoop::run_scrollback`).
///
/// Exists so the "disabled ⇒ take no clock at all" decision lives in one place instead of being duplicated
/// into both frame closures. Usage: [`Self::start`] → [`Self::mark_draw`] → [`Self::finish`].
pub(crate) struct FrameProbe<'a> {
    stats: Option<&'a FrameStats>,
    started_at: Option<Instant>,
    draw: Duration,
}

impl<'a> FrameProbe<'a> {
    /// Begin a frame. When the accumulator is disabled nothing is timed (`Instant::now` is not even called).
    ///
    /// Call it *before* `canvas.clear()` so the clear is part of the draw cost, and after the terminal size
    /// sync (which is not a rendering cost).
    ///
    /// Note: the frame number is **not** allocated here — the probe is created per frame, so a counter on it
    /// would always read 0; [`FrameStats::record`] assigns it instead.
    pub(crate) fn start(stats: &'a FrameStats) -> FrameProbe<'a> {
        if stats.enabled() {
            FrameProbe {
                stats: Some(stats),
                started_at: Some(Instant::now()),
                draw: Duration::ZERO,
            }
        } else {
            FrameProbe {
                stats: None,
                started_at: None,
                draw: Duration::ZERO,
            }
        }
    }

    /// Close the draw phase (after `draw(&mut canvas)`, before the flush).
    pub(crate) fn mark_draw(&mut self) {
        if let Some(started_at) = self.started_at {
            self.draw = started_at.elapsed();
        }
    }

    /// Close the frame and record it. `line_fixed` must be the rows **actually pushed**, not the theoretical
    /// `fixed_h` bound.
    pub(crate) fn finish(
        self,
        bytes_written: u64,
        repaint_span: Option<(usize, usize)>,
        line_total: usize,
        line_fixed: usize,
    ) {
        let (Some(stats), Some(started_at)) = (self.stats, self.started_at) else {
            return;
        };
        let elapsed = started_at.elapsed();
        let mut metrics = FrameMetrics {
            frame_no: 0, // assigned by record()
            draw: self.draw.min(elapsed),
            flush: elapsed.saturating_sub(self.draw),
            bytes_written,
            repaint_span,
            line_total,
            line_fixed,
        };
        stats.record(&mut metrics);
    }
}

/// Byte-counting writer wrapper: forwards every write to the terminal and adds the length to a counter.
///
/// Why a wrapper instead of returning the byte count from the flush functions: `ScrollbackCanvas::flush` and
/// `ScreenBuffer::flush` are already generic over `W: Write` and are called by 20+ tests with the old
/// signature, so wrapping keeps the API untouched. It is **always** installed (no `if enabled` branch, which
/// would duplicate both frame closures): when metrics are off the only extra work is one `u64` addition per
/// write, negligible next to the syscall.
pub(crate) struct CountingWriter<'a, W: Write> {
    inner: &'a mut W,
    bytes: u64,
}

impl<'a, W: Write> CountingWriter<'a, W> {
    /// Wrap the terminal output (frame closures do `let mut out = CountingWriter::new(terminal.out_mut());`).
    pub(crate) fn new(inner: &'a mut W) -> CountingWriter<'a, W> {
        CountingWriter { inner, bytes: 0 }
    }

    /// Bytes written so far (read once after the flush; `flush()` itself is not counted).
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl<W: Write> Write for CountingWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(buf)?;
        self.bytes += written as u64;
        Ok(written)
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        // Delegate straight to the inner `write_all` (not to `self.write`) so a single call counts once.
        self.inner.write_all(buf)?;
        self.bytes += buf.len() as u64;
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test double that records writes and flush calls so forwarding can be asserted.
    #[derive(Default)]
    struct SpyWriter {
        data: Vec<u8>,
        flushes: usize,
    }

    impl Write for SpyWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.data.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            self.flushes += 1;
            Ok(())
        }
    }

    fn metrics(draw_ms: u64, flush_ms: u64) -> FrameMetrics {
        FrameMetrics {
            frame_no: 0,
            draw: Duration::from_millis(draw_ms),
            flush: Duration::from_millis(flush_ms),
            bytes_written: 100,
            repaint_span: Some((0, 9)),
            line_total: 10,
            line_fixed: 3,
        }
    }

    #[test]
    fn trace_env_parsing_accepts_documented_truthy_values_only() {
        for raw in ["1", "true", "yes", " TRUE ", "Yes\n"] {
            assert!(
                frame_trace_enabled_from(Some(raw)),
                "{raw:?} should enable frame metrics"
            );
        }
        for raw in [None, Some("0"), Some(""), Some("on"), Some("no"), Some("2")] {
            assert!(
                !frame_trace_enabled_from(raw),
                "{raw:?} must be treated as off"
            );
        }
    }

    #[test]
    fn disabled_accumulator_records_nothing() {
        let stats = FrameStats::new_with_enabled(DEFAULT_BUDGET, false);
        assert!(!stats.enabled());
        let mut m = metrics(1, 1);
        stats.record(&mut m);
        assert_eq!(stats.frames(), 0);
        assert_eq!(stats.bytes_total(), 0);
        assert_eq!(stats.over_budget(), 0);
        assert_eq!(stats.sample_len(), 0);
        // Frame number stays untouched: a bailed-out frame must not consume one.
        assert_eq!(m.frame_no, 0);
    }

    #[test]
    fn record_assigns_sequential_frame_numbers_and_accumulates() {
        let stats = FrameStats::new_with_enabled(DEFAULT_BUDGET, true);
        let mut first = metrics(1, 1);
        stats.record(&mut first);
        let mut second = metrics(2, 2);
        stats.record(&mut second);
        assert_eq!(first.frame_no, 1);
        assert_eq!(second.frame_no, 2);
        assert_eq!(stats.frames(), 2);
        assert_eq!(stats.bytes_total(), 200);
        assert_eq!(stats.sample_len(), 2);
    }

    #[test]
    fn record_overwrites_a_caller_supplied_frame_number() {
        let stats = FrameStats::new_with_enabled(DEFAULT_BUDGET, true);
        let mut m = metrics(1, 1);
        m.frame_no = 99;
        stats.record(&mut m);
        assert_eq!(m.frame_no, 1, "frame numbers are owned by the accumulator");
    }

    #[test]
    fn over_budget_counts_only_frames_strictly_above_budget() {
        let budget = Duration::from_millis(16);
        let stats = FrameStats::new_with_enabled(budget, true);
        // draw + flush == budget exactly → not over budget.
        let mut at_budget = FrameMetrics {
            draw: Duration::from_millis(10),
            flush: Duration::from_millis(6),
            ..metrics(0, 0)
        };
        stats.record(&mut at_budget);
        // one nanosecond above → over budget.
        let mut above = FrameMetrics {
            draw: Duration::from_millis(10),
            flush: Duration::from_millis(6) + Duration::from_nanos(1),
            ..metrics(0, 0)
        };
        stats.record(&mut above);
        assert_eq!(stats.over_budget(), 1);
        assert_eq!(stats.frames(), 2);
    }

    #[test]
    fn ring_drops_the_oldest_samples_when_full() {
        let stats = FrameStats::new_with_capacity(DEFAULT_BUDGET, true, 3);
        for i in 1..=5u64 {
            let mut m = metrics(i, 0);
            stats.record(&mut m);
        }
        assert_eq!(stats.frames(), 5);
        assert_eq!(stats.sample_len(), 3);
        // Oldest dropped: the surviving totals are 3ms, 4ms, 5ms → p50 = 4.00ms.
        let summary = stats.summary();
        assert!(
            summary.contains("p50 4.00ms"),
            "unexpected summary: {summary}"
        );
    }

    #[test]
    fn summary_reports_totals_draw_flush_and_bytes() {
        let budget = Duration::from_millis(16);
        let stats = FrameStats::new_with_enabled(budget, true);
        for (draw, flush) in [(1u64, 1u64), (2, 2), (3, 3), (4, 4)] {
            let mut m = FrameMetrics {
                bytes_written: 1000,
                ..metrics(draw, flush)
            };
            stats.record(&mut m);
        }
        let summary = stats.summary();
        // totals: 2,4,6,8 ms → p50 = 4ms, p99 = 8ms; draw median 2ms, flush median 2ms.
        assert!(
            summary.contains("4 frames"),
            "unexpected summary: {summary}"
        );
        assert!(
            summary.contains("p50 4.00ms"),
            "unexpected summary: {summary}"
        );
        assert!(
            summary.contains("p99 8.00ms"),
            "unexpected summary: {summary}"
        );
        assert!(
            summary.contains("draw median 2.00ms"),
            "unexpected summary: {summary}"
        );
        assert!(
            summary.contains("avg 1000 B/frame"),
            "unexpected summary: {summary}"
        );
        assert!(
            summary.contains("avg 10.0 total"),
            "unexpected summary: {summary}"
        );
        assert!(
            summary.contains("avg 3.0 fixed"),
            "unexpected summary: {summary}"
        );
    }

    #[test]
    fn summary_without_samples_explains_how_to_enable() {
        let stats = FrameStats::new_with_enabled(DEFAULT_BUDGET, false);
        let summary = stats.summary();
        assert!(summary.contains(TRACE_ENV), "unexpected summary: {summary}");
        assert_eq!(summary.lines().count(), 1);
    }

    #[test]
    fn counting_writer_counts_bytes_and_forwards_content() {
        let mut spy = SpyWriter::default();
        {
            let mut writer = CountingWriter::new(&mut spy);
            writer.write_all(b"hello ").unwrap();
            writer.write_all(b"world").unwrap();
            assert_eq!(writer.bytes(), 11);
        }
        assert_eq!(spy.data, b"hello world");
    }

    #[test]
    fn counting_writer_flush_is_forwarded_but_not_counted() {
        let mut spy = SpyWriter::default();
        {
            let mut writer = CountingWriter::new(&mut spy);
            writer.write_all(b"abc").unwrap();
            writer.flush().unwrap();
            assert_eq!(writer.bytes(), 3, "flush must not add bytes");
        }
        assert_eq!(spy.flushes, 1);
    }

    #[test]
    fn probe_records_nothing_when_disabled() {
        let stats = FrameStats::new_with_enabled(DEFAULT_BUDGET, false);
        let mut probe = FrameProbe::start(&stats);
        probe.mark_draw();
        probe.finish(42, Some((0, 1)), 2, 1);
        assert_eq!(stats.frames(), 0);
    }

    #[test]
    fn probe_records_one_frame_with_the_passed_shape() {
        let stats = FrameStats::new_with_enabled(Duration::from_secs(60), true);
        let mut probe = FrameProbe::start(&stats);
        probe.mark_draw();
        probe.finish(1234, Some((5, 9)), 20, 4);
        assert_eq!(stats.frames(), 1);
        assert_eq!(stats.bytes_total(), 1234);
        let summary = stats.summary();
        assert!(
            summary.contains("avg 20.0 total"),
            "unexpected summary: {summary}"
        );
        assert!(
            summary.contains("avg 4.0 fixed"),
            "unexpected summary: {summary}"
        );
    }

    #[test]
    fn percentile_uses_nearest_rank() {
        let values: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
        assert_eq!(percentile(&values, 50.0), Duration::from_millis(50));
        assert_eq!(percentile(&values, 95.0), Duration::from_millis(95));
        assert_eq!(percentile(&values, 99.0), Duration::from_millis(99));
        assert_eq!(percentile(&values, 100.0), Duration::from_millis(100));
        assert_eq!(percentile(&[], 50.0), Duration::ZERO);
        assert_eq!(
            percentile(&[Duration::from_millis(7)], 99.0),
            Duration::from_millis(7)
        );
    }

    #[test]
    fn total_sums_draw_and_flush() {
        let m = metrics(3, 4);
        assert_eq!(m.total(), Duration::from_millis(7));
    }
}
