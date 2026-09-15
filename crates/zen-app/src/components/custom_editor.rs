// REQ-APP-11 (pi-358): custom-editor — chat input editor (app-level keybinding routing + zen-tui Editor delegation).
// Corresponding original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/custom-editor.ts.
// pi's Editor subclass is implemented in zen as a composition of "own a zen-tui Editor + key interception + delegation";
// app-level actions are routed via `KeybindingsManager<AppAction>` (T-AP-11-2 keybindings.rs).

use crate::keybindings::AppAction;
use std::collections::HashMap;
use zen_tui::components::Component;
use zen_tui::editor::{Editor, EditorAction};
use zen_tui::keybindings::KeybindingsManager;
use zen_tui::keys::KeyId;
use zen_tui::screen::{ScreenBuffer, Style};

/// Chat input editor: mounts app-level key routing (interrupt/exit/clear/tools.expand etc.) on top of the zen-tui `Editor`——
/// Shared by T-AP-11-1 main input; when Esc hits Interrupt and no completion is active, call on_escape; Ctrl+D with an
/// empty editor goes to on_exit; other app actions are matched one by one against the registry, on a hit the callback fires and short-circuits; on miss, fall back to Editor.
pub struct CustomEditor {
    editor: Editor,
    kb: KeybindingsManager<AppAction>,
    action_handlers: HashMap<AppAction, Box<dyn FnMut()>>,
    on_escape: Option<Box<dyn FnMut()>>,
    on_exit: Option<Box<dyn FnMut()>>,
}

impl CustomEditor {
    /// Constructed with initial left padding, the app keymap, and the editor-domain keymap. `editor_kb` is passed through to the inner zen-tui `Editor`.
    /// (T-AP-06-2 constructor injection: user overrides enter editor action routing here; default table `editor_default_bindings()`).
    pub fn new(
        padding_x: usize,
        kb: KeybindingsManager<AppAction>,
        editor_kb: KeybindingsManager<EditorAction>,
    ) -> Self {
        CustomEditor {
            editor: Editor::new(padding_x, editor_kb),
            kb,
            action_handlers: HashMap::new(),
            on_escape: None,
            on_exit: None,
        }
    }

    /// Register action handlers (non-special actions such as app.interrupt). Consumer: T-AP-11-1.
    pub fn on_action(&mut self, action: AppAction, handler: Box<dyn FnMut()>) {
        self.action_handlers.insert(action, handler);
    }

    /// Override default Esc/exit behavior (dynamic replacement, pi onEscape counterpart: takes precedence when Esc hits Interrupt).
    pub fn set_on_escape(&mut self, f: Box<dyn FnMut()>) {
        self.on_escape = Some(f);
    }

    /// Override empty-editor Ctrl+D behavior (pi onCtrlD counterpart).
    pub fn set_on_exit(&mut self, f: Box<dyn FnMut()>) {
        self.on_exit = Some(f);
    }

    /// Passthrough to Editor: current full text.
    pub fn text(&self) -> String {
        self.editor.text()
    }

    /// Passthrough Editor: line references (autocomplete provider input, consumed by autocomplete-wiring-fix).
    pub fn lines(&self) -> &[String] {
        self.editor.lines()
    }

    /// Passthrough Editor: logical cursor (line, column) (autocomplete provider input, consumed by autocomplete-wiring-fix).
    pub fn cursor(&self) -> (usize, usize) {
        self.editor.cursor()
    }

    /// Passthrough Editor: programmatically set logical cursor (positioned after `apply_completion` backfill, consumed by autocomplete-wiring-fix).
    pub fn set_cursor(&mut self, line: usize, col: usize) {
        self.editor.set_cursor(line, col);
    }

    /// Passthrough to Editor: overwrite text and reset cursor (programmatic set).
    pub fn set_text(&mut self, text: &str) {
        self.editor.set_text(text);
    }

    /// Passthrough to Editor: replace border style (app-layer focus color change etc.).
    pub fn set_border_style(&mut self, style: Style) {
        self.editor.set_border_style(style);
    }

    /// Passthrough to Editor: horizontal padding.
    pub fn set_padding_x(&mut self, padding: usize) {
        self.editor.set_padding_x(padding);
    }

    /// Passthrough to Editor: submit history (called after each successful submit).
    pub fn add_to_history(&mut self, text: &str) {
        self.editor.add_to_history(text);
    }

    /// Passthrough to Editor: register submit callback.
    pub fn set_on_submit(&mut self, f: impl FnMut(&str) + 'static) {
        self.editor.set_on_submit(f);
    }

    /// Passthrough to Editor: register text change callback.
    pub fn set_on_change(&mut self, f: impl FnMut(&str) + 'static) {
        self.editor.set_on_change(f);
    }

    /// Passthrough to Editor: focused state.
    pub fn set_focused(&mut self, focused: bool) {
        self.editor.set_focused(focused);
    }

    /// Passthrough to Editor: read focused state.
    pub fn is_focused(&self) -> bool {
        self.editor.is_focused()
    }

    /// Whether autocomplete is active (no completion state inside the v1 editor — T-TU-03-3 completion is handled by T-AP-11-1 as an overlay,
    /// Always false here; after autocomplete-wiring-fix this flag becomes **moot**—when completion is active, `InteractiveUi::`
    /// handle_key` already has all keys intercepted by the overlay before calling this editor, so Esc cannot reach the Interrupt branch;
    /// if completion is moved down into the editor in the future, extend this flag).
    pub fn is_showing_autocomplete(&self) -> bool {
        false
    }

    /// Interactive editor render entry (called by T-AP-11-1 inside the draw callback).
    pub fn render_editor(
        &mut self,
        buf: &mut ScreenBuffer,
        x: u16,
        y: u16,
        width: u16,
        height: u16,
    ) {
        self.editor.render(buf, x, y, width, height);
    }

    /// Compute the total height required for rendering from current text and wrapping rules (borders 2 + content lines, capped at
    /// `max(5, terminal_rows*30/100)` — dynamic budget for editor area height at the UI layer
    /// (T-AP-11-1-fix; width semantics strictly consistent with render, see `Editor::needed_height`).
    pub fn needed_height(&self, width: u16, terminal_rows: u16) -> u16 {
        self.editor.needed_height(width, terminal_rows)
    }

    /// Bracketed paste entry: delegate to Editor for newline normalization / tab expansion / non-printable filtering, then insert the whole block.
    /// Consumer: T-AP-11-1-fix (`InteractiveUi::handle_paste` forwards; on_event routes
    /// `Event::Paste`）。
    pub fn handle_paste(&mut self, text: &str) {
        self.editor.handle_paste(text);
    }

    /// Hardware cursor positioning (delegates to Editor; used by T-AP-11-1 IME positioning).
    pub fn cursor_position(&self) -> (u16, u16) {
        self.editor.cursor_position()
    }

    /// Editor action passthrough (send EditorAction via this when the command side needs it).
    pub fn handle_action(&mut self, action: EditorAction) {
        self.editor.handle_action(action);
    }
}

impl Component for CustomEditor {
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        if width < 2 || height < 3 {
            return 0;
        }
        let height_before = height;
        self.editor.render(buf, x, y, width, height);
        height_before
    }

    fn handle_key(&mut self, key: &crossterm::event::KeyEvent) -> bool {
        // 1. app actions (before the editor's inline mapping — aligns with pi CustomEditor.handleInput's app-first)
        if let Some(kid) = KeyId::from_key_event(*key) {
            if self.kb.matches(&kid, AppAction::Interrupt) && !self.is_showing_autocomplete() {
                if let Some(handler) = self.on_escape.as_mut() {
                    handler();
                    return true;
                }
                if let Some(handler) = self.action_handlers.get_mut(&AppAction::Interrupt) {
                    handler();
                    return true;
                }
                // No handler: fall back to the editor (Esc edit semantics)
                self.editor.handle_key(*key);
                return true;
            }
            if self.kb.matches(&kid, AppAction::Exit) && self.editor.text().is_empty() {
                if let Some(handler) = self.on_exit.as_mut() {
                    handler();
                    return true;
                }
                if let Some(handler) = self.action_handlers.get_mut(&AppAction::Exit) {
                    handler();
                    return true;
                }
                return false;
            }

            // Remaining registered actions (Interrupt/Exit handled specially)
            let mut consumed = false;
            for (action, handler) in self.action_handlers.iter_mut() {
                if *action != AppAction::Interrupt
                    && *action != AppAction::Exit
                    && self.kb.matches(&kid, *action)
                {
                    handler();
                    consumed = true;
                    break;
                }
            }
            if consumed {
                return true;
            }
        }

        // 2. Fall back to the editor
        self.editor.handle_key(*key);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keybindings::app_default_bindings;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;
    use zen_tui::keybindings::editor_default_bindings;

    fn kb() -> KeybindingsManager<AppAction> {
        KeybindingsManager::new(app_default_bindings(), HashMap::new())
    }

    /// Editor domain keybinding table (default, no user overrides).
    fn editor_kb() -> KeybindingsManager<EditorAction> {
        KeybindingsManager::new(editor_default_bindings(), HashMap::new())
    }

    fn press(code: crossterm::event::KeyCode) -> crossterm::event::KeyEvent {
        crossterm::event::KeyEvent::new(code, crossterm::event::KeyModifiers::NONE)
    }

    fn ctrl(c: char) -> crossterm::event::KeyEvent {
        crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char(c),
            crossterm::event::KeyModifiers::CONTROL,
        )
    }

    #[test]
    fn printable_keys_go_to_editor() {
        let mut editor = CustomEditor::new(0, kb(), editor_kb());
        assert!(editor.handle_key(&press(crossterm::event::KeyCode::Char('a'))));
        assert!(editor.handle_key(&press(crossterm::event::KeyCode::Char('b'))));
        assert_eq!(editor.text(), "ab");
    }

    #[test]
    fn tools_expand_routes_to_handler() {
        let mut editor = CustomEditor::new(0, kb(), editor_kb());
        let fired = Rc::new(RefCell::new(0));
        let f = fired.clone();
        editor.on_action(
            AppAction::ToolsExpand,
            Box::new(move || *f.borrow_mut() += 1),
        );
        assert!(editor.handle_key(&ctrl('o')));
        assert_eq!(*fired.borrow(), 1);
    }

    #[test]
    fn esc_fires_on_escape() {
        let mutated = Rc::new(RefCell::new(0));
        let m = mutated.clone();
        let mut editor = CustomEditor::new(0, kb(), editor_kb());
        editor.set_on_escape(Box::new(move || *m.borrow_mut() += 1));
        assert!(editor.handle_key(&press(crossterm::event::KeyCode::Esc)));
        assert_eq!(*mutated.borrow(), 1);
    }

    #[test]
    fn exit_only_when_empty() {
        // Empty editor: Ctrl+D → on_exit
        let exited = Rc::new(RefCell::new(0));
        let e = exited.clone();
        let mut editor = CustomEditor::new(0, kb(), editor_kb());
        editor.set_on_exit(Box::new(move || *e.borrow_mut() += 1));
        assert!(editor.handle_key(&ctrl('d')));
        assert_eq!(*exited.borrow(), 1);

        // Non-empty editor: Ctrl+D falls to the editor (delete previous char, does not trigger on_exit)
        let mut editor2 = CustomEditor::new(0, kb(), editor_kb());
        editor2.set_text("ab");
        let _ = editor2.handle_key(&press(crossterm::event::KeyCode::Home));
        assert!(editor2.handle_key(&ctrl('d')));
        assert_eq!(editor2.text(), "b");
    }

    #[test]
    fn delegates_text_and_history() {
        let mut editor = CustomEditor::new(2, kb(), editor_kb());
        editor.set_text("hello");
        assert_eq!(editor.text(), "hello");
        editor.add_to_history("hello");
        editor.set_padding_x(4);
        editor.set_border_style(Style::default());
        editor.set_focused(true);
        assert!(editor.is_focused());
    }

    #[test]
    fn render_returns_height() {
        let mut editor = CustomEditor::new(0, kb(), editor_kb());
        editor.set_text("x");
        let mut buf = ScreenBuffer::new(40, 5);
        let painted = editor.render(&mut buf, 0, 0, 40, 5);
        assert!(painted >= 3, "editor renders border + text + border");
    }

    #[test]
    fn needed_height_delegates_to_editor() {
        let mut editor = CustomEditor::new(0, kb(), editor_kb());
        // empty -> 3; 3 short lines -> 5; rows=40 gives cap=12, 7 lines of content -> 9
        assert_eq!(editor.needed_height(40, 24), 3);
        editor.set_text("a\nb\nc");
        assert_eq!(editor.needed_height(40, 24), 5);
        editor.set_text("a\nb\nc\nd\ne\nf\ng");
        assert_eq!(editor.needed_height(40, 40), 9);
        // semantics match rendering: after rendering at the returned height, all content lines are visible (no disconnect)
        let h = editor.needed_height(40, 40);
        let mut buf = ScreenBuffer::new(40, h);
        editor.render_editor(&mut buf, 0, 0, 40, h);
        let filled = (1..h - 1)
            .filter(|r| !buf.line_segments(*r as u16).is_empty())
            .count();
        assert_eq!(filled as u16, h - 2);
    }

    #[test]
    fn handle_paste_delegates_to_editor() {
        let mut editor = CustomEditor::new(0, kb(), editor_kb());
        editor.handle_paste("line1\r\nline2");
        assert_eq!(editor.text(), "line1\nline2", "Delegates newline normalization semantics to Editor");
        // tab expansion + non-printable filtering also take effect
        let mut editor2 = CustomEditor::new(0, kb(), editor_kb());
        editor2.handle_paste("a\tb\x01c");
        assert_eq!(editor2.text(), "a    bc");
        // empty paste is a no-op
        let mut editor3 = CustomEditor::new(0, kb(), editor_kb());
        editor3.set_text("keep");
        editor3.handle_paste("");
        assert_eq!(editor3.text(), "keep");
    }
}
