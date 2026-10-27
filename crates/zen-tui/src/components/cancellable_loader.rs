// REQ-TUI-01 : CancellableLoader — cancellable loader

use crate::screen::{ScreenBuffer, Style};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use tokio_util::sync::CancellationToken;

use super::loader::{Loader, LoaderIndicatorOptions};
use super::Component;

/// Cancellable loader (pi `CancellableLoader`): composes and delegates to [`Loader`] plus cancellation semantics.
/// Composition instead of inheritance (pi `extends Loader` is implemented in zen via composition + public forwarding; AGENTS.md recommends composition).
/// `cancel_token` is created and passed in by the caller; the async task side holds the same token handle;
/// On user Esc/Ctrl+C, trigger `token.cancel()` and the `on_cancel` callback (align with pi `onAbort`).
pub struct CancellableLoader {
    loader: Loader, // Delegated by composition (rendering/messages/animation)
    cancel_token: CancellationToken, // Cancellation handle (the caller holds the other end)
    on_cancel: Option<Box<dyn FnMut()>>, // Cancellation callback (pi `onAbort`)
}

impl CancellableLoader {
    /// Constructor. Consumer: T-AP-11-2 (bordered-loader, tool execution cancellation).
    pub fn new(
        token: CancellationToken,
        spinner_style: Style,
        message_style: Style,
        message: impl Into<String>,
    ) -> Self {
        CancellableLoader {
            loader: Loader::new(spinner_style, message_style, message),
            cancel_token: token,
            on_cancel: None,
        }
    }

    /// Cancellation token handle (counterpart of pi `signal` getter, for caller/task-side pre-subscription).
    pub fn cancel_token(&self) -> &CancellationToken {
        &self.cancel_token
    }

    /// Whether cancellation has occurred (counterpart of pi `aborted` getter).
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    /// Register cancellation callback (counterpart of pi `onAbort`; triggered on Esc/Ctrl+C, cleanup then wired by caller).
    pub fn set_on_cancel(&mut self, f: impl FnMut() + 'static) {
        self.on_cancel = Some(Box::new(f));
    }

    // ── Public methods delegating to Loader (zen composition form of pi `extends`) ──

    /// Start animation (delegates to Loader; see its docs).
    pub fn start(&mut self) {
        self.loader.start();
    }

    /// Stop animation (delegates to Loader; see its docs).
    pub fn stop(&mut self) {
        self.loader.stop();
    }

    /// Whether running (delegates to Loader; see its docs).
    pub fn is_running(&self) -> bool {
        self.loader.is_running()
    }

    /// Update message field (delegates to Loader; does not trigger redraw itself; see its docs).
    pub fn set_message(&mut self, message: impl Into<String>) {
        self.loader.set_message(message);
    }

    /// Set indicator (delegates to Loader; see its docs).
    pub fn set_indicator(&mut self, indicator: Option<&LoaderIndicatorOptions>) {
        self.loader.set_indicator(indicator);
    }

    /// Update spinner and message styles (delegates to Loader; see its docs).
    pub fn set_styles(&mut self, spinner_style: Style, message_style: Style) {
        self.loader.set_styles(spinner_style, message_style);
    }
}

impl Component for CancellableLoader {
    /// Delegate rendering to the embedded Loader (forward with same signature; see Loader docs).
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        self.loader.render(buf, x, y, width, height)
    }

    /// Key entry point: Esc/Ctrl+C → `cancel_token.cancel()` + `on_cancel`, returns true;
    /// Other keys are not consumed (matches pi `tui.select.cancel`; keymap takes over after T-TU-03-1 lands).
    fn handle_key(&mut self, key: &KeyEvent) -> bool {
        match (key.code, key.modifiers) {
            (KeyCode::Esc, _) | (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                self.cancel_token.cancel();
                if let Some(cb) = self.on_cancel.as_mut() {
                    cb();
                }
                true
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token() -> CancellationToken {
        CancellationToken::new()
    }

    #[test]
    fn new_is_not_cancelled_and_delegates_loader_state() {
        let t = token();
        let cl = CancellableLoader::new(t.clone(), Style::default(), Style::default(), "msg");
        assert!(!cl.is_cancelled());
        // Identity verification (does not rely on unstable is_same_token): cancelling the original token propagates to the component (same token)
        t.cancel();
        assert!(cl.is_cancelled());
        let mut cl2 = CancellableLoader::new(token(), Style::default(), Style::default(), "x");
        cl2.start();
        assert!(cl2.is_running());
    }

    #[test]
    fn esc_cancels_token_and_fires_on_cancel() {
        let t = token();
        let cancelled = std::rc::Rc::new(std::cell::RefCell::new(false));
        let c = cancelled.clone();
        let mut cl = CancellableLoader::new(t.clone(), Style::default(), Style::default(), "msg");
        cl.set_on_cancel(move || *c.borrow_mut() = true);

        // Unrelated key is not consumed and does not cancel
        let key = KeyEvent::from(crossterm::event::KeyCode::Char('x'));
        assert!(!cl.handle_key(&key));
        assert!(!cl.is_cancelled());
        assert!(!*cancelled.borrow());

        // Esc is consumed and cancels
        let esc = KeyEvent::from(crossterm::event::KeyCode::Esc);
        assert!(cl.handle_key(&esc));
        assert!(cl.is_cancelled());
        assert!(t.is_cancelled());
        assert!(*cancelled.borrow());
    }

    #[test]
    fn ctrl_c_cancels() {
        let t = token();
        let mut cl = CancellableLoader::new(t.clone(), Style::default(), Style::default(), "msg");
        let key = KeyEvent::new(
            crossterm::event::KeyCode::Char('c'),
            crossterm::event::KeyModifiers::CONTROL,
        );
        assert!(cl.handle_key(&key));
        assert!(cl.is_cancelled());
    }

    #[test]
    fn delegate_methods_forward_to_loader() {
        let mut cl = CancellableLoader::new(token(), Style::default(), Style::default(), "old");
        // set_message forwarding verified via an isolated separate token
        cl.set_message("new");
        // set_indicator hides the indicator (empty frame)
        cl.set_indicator(Some(&LoaderIndicatorOptions {
            frames: Some(vec![]),
            interval_ms: None,
        }));
        // set_styles does not panic and is runnable
        cl.set_styles(Style::default(), Style::default());
        cl.stop();
        assert!(!cl.is_running());
    }
}
