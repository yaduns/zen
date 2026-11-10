pub mod cache;
pub mod cancellable_loader;
pub mod container;
pub mod input;
pub mod loader;
pub mod select_list;
pub mod settings_list;
pub mod spacer;
pub mod text;
pub mod truncated_text;

use crate::screen::RenderTarget;
use crossterm::event::KeyEvent;

/// Unified abstraction for common components (REQ-TUI-01; since long-row-clip-fix changed to "render-target-oriented"):
/// `render` draws the component into any [`RenderTarget`] and returns the **number of lines occupied** (≤ `height`) for stacked layout budgeting; out-of-bounds content is clipped by the target itself.
///
/// **Two kinds of targets**: `ScreenBuffer` (fixed-height grid, bounded screen consumer) and `LineBuffer` (grows on demand, line-stream consumer).
/// `height` is only the **caller's budget**, not the component's contract: the screen side passes the remaining screen lines, and the line-stream side passes `u16::MAX` (= the "no height limit" of pi `render(width)`).
/// Components should treat it only as a stopping condition and must not hard-code any line cap into themselves (this is exactly why the old "300-line cap for a single row" was retired).
///
/// `handle_key` returns true if the key is consumed (short-circuiting later components/application layer); default is not consumed.
/// `invalidate` clears internal caches for scenarios such as theme switching. Components themselves still hold no cross-frame cache
/// (they are constructed transiently per line); the cross-frame cache lives on the UI side as a per-row [`cache::LineCache`] (PF-04).
/// Side-effect note: `render` is not guaranteed to be idempotent (`StatusIndicator` advances its countdown), so it should be called only once per visible frame.
/// Consumers: all components (`zen-tui` + `zen-app`); line-stream entry point `zen-app` `render_row_lines`.
pub trait Component {
    fn render(
        &mut self,
        target: &mut dyn RenderTarget,
        x: u16,
        y: u16,
        width: u16,
        height: u16,
    ) -> u16;
    fn handle_key(&mut self, _key: &KeyEvent) -> bool {
        false
    }
    fn invalidate(&mut self) {}
}
