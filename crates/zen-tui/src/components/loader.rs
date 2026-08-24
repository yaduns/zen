// REQ-TUI-01 (pi-413): Loader — spinner + single-line message component
// See reference/pi-original/packages/tui/src/components/loader.ts
// Animation is derived from started_at and elapsed/interval (pi's setInterval + ui.requestRender back-reference
// have all been eliminated: repaint scheduling belongs to TuiLoop animation mode; the component has no cyclic back-reference and no style cache).

use crate::screen::{ScreenBuffer, Style};
use std::time::{Duration, Instant};

/// Default animation frames (aligned with pi `DEFAULT_FRAMES`, ten braille frames).
const DEFAULT_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
/// Default frame interval in milliseconds (aligned with pi `DEFAULT_INTERVAL_MS`).
const DEFAULT_INTERVAL_MS: u64 = 80;

/// Spinner + single-line message component. Animation frames are derived from started_at and elapsed/interval (periodic repaint is driven by TuiLoop animation mode,
/// the application layer is responsible for set_animation(true/false); Loader has no cyclic back-reference); set_indicator resets the frame table/interval and reanchors the clock.
pub struct Loader {
    message: String,             // Display message (default "Loading...")
    spinner_style: Style,        // spinner style
    message_style: Style,        // message style
    frames: Vec<String>,         // Frame table (default DEFAULT_FRAMES, nine braille segments)
    interval: Duration,          // Frame interval (default DEFAULT_INTERVAL_MS = 80ms)
    started_at: Option<Instant>, // Animation clock anchor; set by start(), cleared by stop() (frame 0 when not running)
}

/// Spinner indicator configuration (pi `LoaderIndicatorOptions`); frames as an empty array = hide the indicator.
pub struct LoaderIndicatorOptions {
    pub frames: Option<Vec<String>>,
    pub interval_ms: Option<u64>,
}

impl Loader {
    /// Constructor (default frames/interval; pi constructor params spinnerColorFn/messageColorFn are replaced by
    /// two Styles, and message defaults to "Loading...").
    /// Consumers: T-AP-11-2 (bash-execution/bordered-loader), T-AP-11-1 (status line).
    pub fn new(spinner_style: Style, message_style: Style, message: impl Into<String>) -> Self {
        Loader {
            message: message.into(),
            spinner_style,
            message_style,
            frames: DEFAULT_FRAMES.iter().map(|s| s.to_string()).collect(),
            interval: Duration::from_millis(DEFAULT_INTERVAL_MS),
            started_at: None,
        }
    }

    /// Start animation: set started_at (aligned with pi start: update display + restart animation).
    /// Frame sequence is derived from elapsed/interval at render time, so reanchoring the clock is enough to return to frame 0.
    pub fn start(&mut self) {
        self.started_at = Some(Instant::now());
    }

    /// Stop animation: clear started_at (aligned with pi stop: clear timer). Rendering still shows the first frame.
    pub fn stop(&mut self) {
        self.started_at = None;
    }

    /// Whether running (animation advancing).
    pub fn is_running(&self) -> bool {
        self.started_at.is_some()
    }

    /// Update the message field; takes effect on the next rendered frame. **Does not actively trigger repaint** — inside pi `setMessage`
    /// updateDisplay + requestRender (TUI back-reference) have been eliminated in zen; repaint scheduling belongs to the application layer.
    pub fn set_message(&mut self, message: impl Into<String>) {
        self.message = message.into();
    }

    /// Set indicator (setIndicator counterpart): update frame table/interval and reset animation clock;
    /// frames as an empty array = hide spinner and show only the message; None = restore defaults.
    /// Aligned with pi: reanchor the clock so the frame sequence restarts from 0 (pi's setIndicator → start).
    pub fn set_indicator(&mut self, indicator: Option<&LoaderIndicatorOptions>) {
        match indicator {
            Some(opts) => {
                self.frames = match &opts.frames {
                    // Some(empty array) also follows this → hide indicator (show only message)
                    Some(f) => f.clone(),
                    None => DEFAULT_FRAMES.iter().map(|s| s.to_string()).collect(),
                };
                let ms = opts
                    .interval_ms
                    .filter(|&v| v > 0)
                    .unwrap_or(DEFAULT_INTERVAL_MS);
                self.interval = Duration::from_millis(ms);
            }
            None => {
                self.frames = DEFAULT_FRAMES.iter().map(|s| s.to_string()).collect();
                self.interval = Duration::from_millis(DEFAULT_INTERVAL_MS);
            }
        }
        // Reanchor animation clock: equivalent to start(); frame sequence restarts from 0
        self.started_at = Some(Instant::now());
    }

    /// Update spinner and message style fields (no direct pi counterpart; used when switching themes).
    /// No cache, so no invalidation needed (pi's style cache invalidation semantics have no counterpart in zen).
    pub fn set_styles(&mut self, spinner_style: Style, message_style: Style) {
        self.spinner_style = spinner_style;
        self.message_style = message_style;
    }
}

impl super::Component for Loader {
    /// Render one line: spinner frame (advance by elapsed/interval modulo while running; use frame 0 when not running,
    /// first frame) + space + message; when frames is empty (hide indicator), draw only the message (align pi empty frame → no indicator).
    /// Frame grapheme is a narrow character, no special width handling needed. Returns 1 (returns 0 when height is 0).
    /// The leading blank line of pi render is not emitted (spacing is handled by the container/Spacer).
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        if height == 0 {
            return 0;
        }
        let _ = width; // Width truncation is silently handled by ScreenBuffer::write

        // Current frame index: while running, advance by (elapsed / interval) % frame count; when not running, use frame 0.
        // When frames is empty, always use 0 (get will return None → do not draw the indicator).
        let idx = if self.frames.is_empty() {
            0
        } else {
            match self.started_at {
                Some(t0) => {
                    let period = self.interval.as_millis().max(1) as u64;
                    let elapsed_ms = t0.elapsed().as_millis() as u64;
                    (elapsed_ms / period) % self.frames.len() as u64
                }
                None => 0,
            }
        };

        let mut col = x;
        if let Some(frame) = self.frames.get(idx as usize) {
            // spinner frame (narrow grapheme occupies 1 column) + space separator + message
            buf.write(col, y, &self.spinner_style, frame);
            col += 1;
            buf.write(col, y, &Style::default(), " ");
            col += 1;
        }
        buf.write(col, y, &self.message_style, &self.message);
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Component;

    #[test]
    fn new_defaults() {
        let l = Loader::new(Style::default(), Style::default(), "Loading...");
        assert_eq!(l.frames.len(), DEFAULT_FRAMES.len());
        assert_eq!(l.interval, Duration::from_millis(DEFAULT_INTERVAL_MS));
        assert!(!l.is_running());
    }

    #[test]
    fn start_stop_is_running() {
        let mut l = Loader::new(Style::default(), Style::default(), "hi");
        assert!(!l.is_running());
        l.start();
        assert!(l.is_running());
        l.stop();
        assert!(!l.is_running());
    }

    #[test]
    fn set_message_updates() {
        let mut l = Loader::new(Style::default(), Style::default(), "Loading...");
        l.set_message("Working");
        assert_eq!(l.message, "Working");
    }

    #[test]
    fn set_styles_updates_fields() {
        let s1 = Style {
            fg: Some(crossterm::style::Color::Red),
            ..Style::default()
        };
        let s2 = Style {
            fg: Some(crossterm::style::Color::Blue),
            ..Style::default()
        };
        let mut l = Loader::new(Style::default(), Style::default(), "x");
        l.set_styles(s1.clone(), s2.clone());
        assert_eq!(l.spinner_style, s1);
        assert_eq!(l.message_style, s2);
    }

    #[test]
    fn set_indicator_none_restores_defaults_and_starts() {
        let mut l = Loader::new(Style::default(), Style::default(), "x");
        // Break it first, then restore
        l.frames = vec!["a".to_string()];
        l.interval = Duration::from_millis(999);
        l.set_indicator(None);
        assert_eq!(l.frames.len(), DEFAULT_FRAMES.len());
        assert_eq!(l.interval, Duration::from_millis(DEFAULT_INTERVAL_MS));
        // set_indicator re-anchors the animation clock → running
        assert!(l.is_running());
    }

    #[test]
    fn set_indicator_empty_frames_hides_indicator() {
        let mut l = Loader::new(Style::default(), Style::default(), "msg");
        let opts = LoaderIndicatorOptions {
            frames: Some(vec![]),
            interval_ms: None,
        };
        l.set_indicator(Some(&opts));
        assert!(l.frames.is_empty());
        assert!(l.is_running());
    }

    #[test]
    fn set_indicator_custom_frames_interval() {
        let mut l = Loader::new(Style::default(), Style::default(), "x");
        let opts = LoaderIndicatorOptions {
            frames: Some(vec!["a".to_string(), "b".to_string()]),
            interval_ms: Some(50),
        };
        l.set_indicator(Some(&opts));
        assert_eq!(l.frames, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(l.interval, Duration::from_millis(50));
    }

    #[test]
    fn set_indicator_zero_interval_falls_back_to_default() {
        let mut l = Loader::new(Style::default(), Style::default(), "x");
        let opts = LoaderIndicatorOptions {
            frames: Some(vec!["a".to_string()]),
            interval_ms: Some(0), // ≤0 → default
        };
        l.set_indicator(Some(&opts));
        assert_eq!(l.interval, Duration::from_millis(DEFAULT_INTERVAL_MS));
    }

    #[test]
    fn render_not_running_and_hidden_return_one_line() {
        // Not started: still renders first frame + message and returns 1 line (no public cell read API; only assert line count and running state)
        let mut buf = ScreenBuffer::new(20, 1);
        let mut l = Loader::new(Style::default(), Style::default(), "Wait");
        l.frames = vec!["⠋".to_string()];
        assert_eq!(l.render(&mut buf, 0, 0, 20, 1), 1);
        // Hidden indicator (empty frame list): still returns 1 line
        let mut buf2 = ScreenBuffer::new(20, 1);
        l.set_indicator(Some(&LoaderIndicatorOptions {
            frames: Some(vec![]),
            interval_ms: None,
        }));
        assert_eq!(l.render(&mut buf2, 0, 0, 20, 1), 1);
    }

    #[test]
    fn render_height_zero_returns_zero() {
        let mut buf = ScreenBuffer::new(10, 1);
        let mut l = Loader::new(Style::default(), Style::default(), "x");
        assert_eq!(l.render(&mut buf, 0, 0, 10, 0), 0);
    }
}
