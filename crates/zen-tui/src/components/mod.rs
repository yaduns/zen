pub mod cancellable_loader;
pub mod container;
pub mod input;
pub mod loader;
pub mod select_list;
pub mod settings_list;
pub mod spacer;
pub mod text;
pub mod truncated_text;

use crate::screen::ScreenBuffer;
use crossterm::event::KeyEvent;

/// Unified abstraction for common components
/// `render` draws the component into the screen buffer within the given rectangle and returns the number of rows actually occupied (≤ height),
/// for row-based stacked layout by containers (Box) and the chat composition (T-AP-11-1); out-of-bounds content outside the rectangle is left to
/// `ScreenBuffer::write` for silent truncation.
/// `handle_key` returns true if the key is consumed (short-circuiting later components/application layer); default is not consumed.
/// `invalidate` clears internal caches for scenarios such as theme switching; current components have no cross-frame cache, so default is no-op.
/// Consumers: all components in this task; planned consumers: T-AP-11-2 (application components implementing this trait),
/// T-AP-11-1 (composition), T-AP-08-6 (startup UI).
pub trait Component {
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16;
    fn handle_key(&mut self, _key: &KeyEvent) -> bool {
        false
    }
    fn invalidate(&mut self) {}
}
