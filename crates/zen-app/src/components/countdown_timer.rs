// REQ-APP-11 (pi-358): CountdownTimer — retry countdown (elapsed-driven, no thread/timer).
// Corresponds to original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/countdown-timer.ts.
// Implementation revision: pi uses setInterval with a callback every second; zen follows Loader's existing pattern (elapsed derivation, redraw handled by
// TuiLoop animation mode) and instead recomputes remaining seconds on each render tick() — no thread or Send bounds needed,
// triggers on_tick once at each one-second boundary and on_expire at zero. request_render parameter is eliminated (render loop is the driver).

use std::time::{Duration, Instant};

/// Retry countdown: decrements every second and invokes callbacks — the 'N seconds remaining' text is refreshed by on_tick; on_expire is called on expiry.
///
/// Consumers: RetryStatusIndicator (retry status line), T-AP-11-1 (LLM call retry display).
/// Render-driven contract: the owner calls [`CountdownTimer::tick`] on every render frame;
/// Remaining seconds are rounded up (ceil); construction enters the 'not started' state, and timing starts after `start()`.
pub struct CountdownTimer {
    timeout: Duration,
    started_at: Option<Instant>, // None = not started/stopped
    last_remaining: u32, // Remaining seconds at last callback (avoid repeated callbacks every second)
    done: bool,
    on_tick: Option<Box<dyn FnMut(u32)>>, // Remaining-seconds change callback (±1s boundary)
    on_expire: Option<Box<dyn FnMut()>>,  // Zero callback (fired only once)
}

impl CountdownTimer {
    /// Construct with total duration and two callbacks; `request_render` is eliminated in zen (render loop is the driver; see file header).
    pub fn new(timeout_ms: u64, on_tick: Box<dyn FnMut(u32)>, on_expire: Box<dyn FnMut()>) -> Self {
        CountdownTimer {
            timeout: Duration::from_millis(timeout_ms),
            started_at: None,
            last_remaining: 0,
            done: false,
            on_tick: Some(on_tick),
            on_expire: Some(on_expire),
        }
    }

    /// Start countdown: reset anchor and completion flag. Call after first construction to enter timing state (idempotent semantics guaranteed by caller).
    pub fn start(&mut self) {
        self.started_at = Some(Instant::now());
        self.last_remaining = self.remaining_seconds();
        self.done = false;
        if let Some(cb) = &mut self.on_tick {
            cb(self.last_remaining);
        }
    }

    /// Compute remaining seconds against the current clock (rounded up; returns total duration seconds if not started).
    pub fn remaining_seconds(&self) -> u32 {
        match self.started_at {
            Some(t0) => {
                let elapsed = t0.elapsed();
                if elapsed >= self.timeout {
                    0
                } else {
                    let left = self.timeout - elapsed;
                    left.as_millis().div_ceil(1000) as u32
                }
            }
            None => self.timeout.as_millis().div_ceil(1000) as u32,
        }
    }

    /// Render-frame driver point: called on every render; internally triggers on_tick when remaining seconds cross a boundary,
    /// and triggers on_expire once at zero. Returns whether it has reached zero (caller can clean up accordingly).
    pub fn tick(&mut self) -> bool {
        if self.started_at.is_none() || self.done {
            return self.done;
        }
        let remaining = self.remaining_seconds();
        if remaining != self.last_remaining {
            self.last_remaining = remaining;
            if let Some(cb) = &mut self.on_tick {
                cb(remaining);
            }
        }
        if remaining == 0 {
            self.done = true;
            if let Some(cb) = &mut self.on_expire {
                cb();
            }
            return true;
        }
        false
    }

    /// Stop timer (idempotent): clear anchor and callbacks; no further tick triggers (aligned with pi dispose semantics).
    pub fn dispose(&mut self) {
        self.started_at = None;
        self.done = true;
        self.on_tick = None;
        self.on_expire = None;
    }

    /// Whether still timing (returns false if not started/stopped/already zero).
    pub fn is_running(&self) -> bool {
        self.started_at.is_some() && !self.done
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn remaining_seconds_ceils_before_start() {
        let mut t = CountdownTimer::new(1500, Box::new(|_| {}), Box::new(|| {}));
        assert_eq!(t.remaining_seconds(), 2);
        t.dispose();
        let t2 = CountdownTimer::new(1000, Box::new(|_| {}), Box::new(|| {}));
        assert_eq!(t2.remaining_seconds(), 1);
    }

    #[test]
    fn start_fires_initial_tick() {
        let ticks = Rc::new(RefCell::new(Vec::new()));
        let t = ticks.clone();
        let mut timer = CountdownTimer::new(
            3000,
            Box::new(move |s| t.borrow_mut().push(s)),
            Box::new(|| {}),
        );
        timer.start();
        assert_eq!(*ticks.borrow(), vec![3]);
        assert!(timer.is_running());
    }

    #[test]
    fn dispose_stops_and_clears_callbacks() {
        let expired = Rc::new(RefCell::new(0));
        let e = expired.clone();
        let mut timer = CountdownTimer::new(
            100,
            Box::new(|_| {}),
            Box::new(move || *e.borrow_mut() += 1),
        );
        timer.start();
        timer.dispose();
        assert!(!timer.is_running());
        // After dispose, tick no longer triggers any callbacks
        timer.tick();
        assert_eq!(*expired.borrow(), 0);
    }

    #[test]
    fn tick_after_deadline_expires_once() {
        let expired = Rc::new(RefCell::new(0));
        let e = expired.clone();
        let mut timer =
            CountdownTimer::new(1, Box::new(|_| {}), Box::new(move || *e.borrow_mut() += 1));
        timer.start();
        // Artificially move anchor earlier to force expiry
        timer.started_at = Some(Instant::now() - Duration::from_secs(10));
        assert!(timer.tick());
        assert_eq!(*expired.borrow(), 1);
        // Second tick no longer fires
        assert!(timer.tick());
        assert_eq!(*expired.borrow(), 1);
    }
}
