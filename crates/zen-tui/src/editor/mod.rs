// Editor component — data types and public API (accessors/callbacks).
// Internal operations split into submodules by domain: cursor.rs (cursor movement), text.rs (editing operations),
// history.rs (commit/history/session state), input.rs (keypress/paste/action routing),
// render.rs (layout/rendering/cursor positioning), undo.rs (undo stack, T-TU-03-5).
// Shared layout types (LayoutLine/VisualLine) live here for submodules to reference via `use super::`;
// TextChunk and word-wrapping logic have been moved to crate::wrap (the width-text family) per the T-TU-01-4 integration plan,
// see `tasks/T-TU-01-4-implemented.md`; `impl Editor` blocks in submodules can access this module's private fields.
use crate::screen::Style;

mod cursor;
mod history;
mod input;
mod render;
mod text;
mod undo;

use undo::LastAction;

/// Editor text core state: multi-line text + logical cursor (row/column)
#[derive(Clone)]
pub struct EditorState {
    lines: Vec<String>, // Multi-line text; always ≥1 line (empty edit = [""]), invariant maintained by edit operations
    cursor_line: usize, // Logical line index of the cursor, always < lines.len()
    cursor_col: usize, // Byte offset of the cursor within its logical line (UTF-8, always on a grapheme boundary, ≤ line length); explicit set_cursor_col resets the sticky state
}

impl Default for EditorState {
    fn default() -> Self {
        Self {
            // Always ≥1 line (empty edit = [""])
            lines: vec![String::new()],
            cursor_line: 0,
            cursor_col: 0,
        }
    }
}

// Key (KeyEvent) → action mapping is handled by T-TU-03-1 (keybindings); T-TU-01-3 handle_action only consumes actions;
// The default trigger keys accompanying each variant doc are the defaults of pi TUI_KEYBINDINGS, used for alignment with the T-TU-03-1 keybinding table and
// T-TU-01-3 handle_key's inline default routing; the mapping is not defined here.
/// Exhaustive set of actions the editor can respond to; a stable contract between the editor input semantics and the key layer.
///
/// All variants are payload-free unit variants (printable characters don't go through the action channel; `handle_key` dispatches insertion directly);
/// See each variant's doc comment for the action ↔ default trigger key correspondence (data source: pi `TUI_KEYBINDINGS`).
/// Unwired behavior variants (`Undo`/`Yank`/`YankPop`/`Tab`) are reserved for later tasks (T-TU-03-5/-7/-3);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EditorAction {
    /// Move cursor up one visual row; at the first visible row, fall through to history scrolling / line start (default `↑`).
    MoveCursorUp,
    /// Move cursor down one visual row; at the last visible row, fall through to history scrolling / line end (default `↓`).
    MoveCursorDown,
    /// Move cursor left one grapheme; at line start, jump to the end of the previous line (default `←` / `Ctrl+B`).
    MoveCursorLeft,
    /// Move cursor right one grapheme; at line end, jump to the start of the next line (default `→` / `Ctrl+F`).
    MoveCursorRight,
    /// Move cursor left by word boundary (default `Alt+←` / `Ctrl+←` / `Alt+B`).
    MoveCursorWordLeft,
    /// Move cursor right by word boundary (default `Alt+→` / `Ctrl+→` / `Alt+F`).
    MoveCursorWordRight,
    /// Move cursor to the start of the logical line (default `Home` / `Ctrl+A`).
    MoveToLineStart,
    /// Move cursor to end of logical line (default `End` / `Ctrl+E`).
    MoveToLineEnd,
    /// Enter "jump forward to character" mode: wait for the next printable character, then jump to its first occurrence after the cursor (default `Ctrl+]`).
    JumpForward,
    /// Enter "jump backward to character" mode: wait for the next printable character, then jump to its nearest occurrence before the cursor (default `Ctrl+Alt+]`).
    JumpBackward,
    /// Scroll up one page (page height = `max(5, 30% of viewport height)`); cursor moves with visual rows (default `PageUp`).
    PageUp,
    /// Scroll down one page (page height as above) (default `PageDown`).
    PageDown,
    /// Delete one grapheme before the cursor; at line start, merge with the previous line (default `Backspace`).
    DeleteCharBackward,
    /// Delete one grapheme at the cursor; at line end, merge with the next line (default `Delete` / `Ctrl+D`).
    DeleteCharForward,
    /// Delete one word backward (before cursor); handle word boundary at line start / merge across lines (default `Ctrl+W` / `Alt+Backspace`).
    DeleteWordBackward,
    /// Delete one word forward (after cursor); handle word boundary at line end / merge across lines (default `Alt+D` / `Alt+Delete`).
    DeleteWordForward,
    /// Delete the entire segment from cursor to line start (default `Ctrl+U`).
    DeleteToLineStart,
    /// Delete the entire segment from cursor to line end (default `Ctrl+K`).
    DeleteToLineEnd,
    /// Paste back the most recent cut text from the kill ring (default `Ctrl+Y`; behavior hooked up by T-TU-03-7).
    Yank,
    /// Rotate the kill ring (only valid immediately after yank/yank-pop) (default `Alt+Y`; T-TU-03-7).
    YankPop,
    /// Undo the most recent edit: pop the stack and restore the text + cursor from before the last edit (default `Ctrl+-`).
    Undo,
    /// Insert a newline at the cursor (default `Shift+Enter` / `Ctrl+J`).
    NewLine,
    /// Submit current input (trim full text, trigger on_submit, then clear) (TS: `tui.input.submit`, default `Enter`).
    Submit,
    /// Tab / autocomplete trigger (default `Tab`; behavior wired in via T-TU-03-3).
    Tab,
}

/// Editor component itself. Held by the application layer after construction; render writes the buffer, handle_key/handle_paste process input, handle_action dispatches actions.
/// Callbacks (on_submit/on_change) and focused are set by the application layer.
pub struct Editor {
    state: EditorState,   // text + cursor
    focused: bool,        // Set/cleared by the application layer on focus change; affects cursor highlight and cursor_position semantics
    scroll_offset: usize, // Offset of the first visible visual row; clamped to the cursor during render
    last_width: usize,    // Cached width of the most recent layout (cursor navigation must match wrap width; updated on render
    last_height: usize,   // Viewport height of the most recent render
    // zen-specific: snapshot of the effective padding_x clamp value used by render. render lays out text with min(padding_x, (w-1)/2),
    // this value cannot be recovered from last_width; cursor_position column resolution (IME positioning) must match render.
    last_effective_padding: usize,
    padding_x: usize, // Initial value passed via constructor; adjustable later via set_padding_x. Constrained by available width during layout
    border_style: Style, // Border/scroll indicator styles; replaceable by the application layer
    history: Vec<String>, // Submitted input history (deduplicated, capped at 100)
    history_index: isize, // -1 = not browsing; 0 = most recent, 1 = older, ...
    history_draft: Option<EditorState>, // Draft stashed when first entering history browsing; restored on exit
    preferred_visual_col: Option<usize>, // Sticky column; only valid during vertical movement (e.g. 9th character in the current row, cursor moves down to 6th character in row 2, then 16th in row 3: cursor should end at the last character of row 2, i.e. the 9th character of row 3)
    snapped_from_cursor_col: Option<usize>, // Original cursor column before snapping to an atomic segment (unbreakable atomic rows break the snapped_from_cursor_col memory; in that case snapped_from_cursor_col takes precedence)
    undo_stack: Vec<EditorState>, // Undo stack (T-TU-03-5, pi-406): before an edit operation mutates state, push a clone of EditorState; Undo pops to restore; submit clears. Snapshots exclude view state (scroll/sticky etc.), which is rebuilt per pi semantics on undo
    last_action: Option<LastAction>, // Last input action category (aligned with pi lastAction; T-TU-03-5 only uses TypeWord to decide word/char insertion merging; kill/yank extended by T-TU-03-7)
    on_submit: Option<Box<dyn FnMut(&str)>>, // Submit callback; receives the expanded text after trim (no marker in this task, i.e. trim of the full text)
    on_change: Option<Box<dyn FnMut(&str)>>, // Called on every text change
    disable_submit: bool,                    // When set, Submit is not triggered
}

/// Rendered layout line: one line of visible text + whether it carries the cursor (with its position within the line when it does).
/// Generated by `render::layout_text`, consumed by `render::render`.
struct LayoutLine {
    text: String,              // Display text of this visual line (a wrap fragment or the whole line)
    has_cursor: bool,          // Whether this line carries the cursor
    cursor_pos: Option<usize>, // Cursor position within this line's text (Some when has_cursor = true)
}

/// Visual line mapping entry: the logical line corresponding to this visual line, and the [start_col, start_col+length) range within it.
/// Generated by `cursor::build_visual_line_map`, consumed by `move_vertical`/`find_current_visual_line`/`page_scroll`.
/// start_col and length are both in **bytes** (consistent with cursor_col/word_wrap_line),
/// length = chunk.end_index - chunk.start_index; an end-of-line cursor falls at start_col+length.
struct VisualLine {
    logical_line: usize, // Index of the owning logical line (index into state.lines)
    start_col: usize,    // Starting byte offset within the logical line
    length: usize,       // Byte range length covered by this visual line (last line allows cursor at start_col+length)
}

impl Editor {
    /// Construct the editor with empty text (single blank line, cursor at (0,0)).
    /// Parameter `padding_x`: initial left/right padding column count (default 0, aligned with pi `options.paddingX`); can be passed at construction,
    /// and later adjusted via `set_padding_x`.
    /// Consumer: T-AP-11-2 (CustomEditor construction, passes settingsManager.getEditorPaddingX()).
    pub fn new(padding_x: usize) -> Self {
        Self {
            state: EditorState::default(),
            focused: false,
            scroll_offset: 0,
            last_width: 0,
            last_height: 0,
            last_effective_padding: 0,
            padding_x,
            border_style: Style::default(),
            history: Vec::new(),
            history_index: -1,
            history_draft: None,
            preferred_visual_col: None,
            snapped_from_cursor_col: None,
            undo_stack: Vec::new(),
            last_action: None,
            on_submit: None,
            on_change: None,
            disable_submit: false,
        }
    }

    /// Returns the full text (lines joined by \n). Consumers read the snapshot; modification doesn't go through this path.
    /// Consumers: T-AP-11-2 (bash mode detection, pre-commit value retrieval), T-AP-11-1, T-TU-03-3 (completion snapshot).
    pub fn text(&self) -> String {
        self.state.lines.join("\n")
    }

    /// Replace the text and reset the cursor to the end of the line (programmatic setting: initial value, external editor write-back T-AP-11-4, history restore).
    /// Normalize newlines (\r\n/\r → \n) and expand tabs (→ 4 spaces, aligning with pi normalizeText).
    /// If content differs from the current text, push an undo snapshot and reset the coalescing state (aligned with pi setText: programmatic changes are undoable;
    /// identical content, e.g. set_text("") on the initial value, is a no-op and is not pushed).
    /// Consumers: T-AP-11-2, T-AP-11-4.
    pub fn set_text(&mut self, text: &str) {
        let normalized = text
            .replace("\r\n", "\n")
            .replace('\r', "\n")
            .replace('\t', "    ");
        self.last_action = None;
        // Aligned with pi setText:
        if self.text() != normalized {
            self.push_undo_snapshot();
        }
        self.state.lines = normalized.split('\n').map(str::to_string).collect();
        if self.state.lines.is_empty() {
            self.state.lines.push(String::new());
        }
        // Move the cursor to the end of the last line
        self.state.cursor_line = self.state.lines.len() - 1;
        self.state.cursor_col = self.state.lines[self.state.cursor_line].len();
        self.preferred_visual_col = None;
        self.snapped_from_cursor_col = None;
        self.scroll_offset = 0;

        self.emit_change();
    }

    /// Return references to each line (input for the autocomplete provider).
    /// Consumer: T-TU-03-3 (getSuggestions/applyCompletion input).
    pub fn lines(&self) -> &[String] {
        &self.state.lines
    }

    /// Return the logical cursor position (line, column).
    /// Consumers: T-TU-03-3, T-AP-11-2 (IME), unit tests.
    pub fn cursor(&self) -> (usize, usize) {
        (self.state.cursor_line, self.state.cursor_col)
    }

    /// Programmatically set the logical cursor (line, column); out-of-range values are clamped to valid bounds (line to last line, column to that line's length).
    /// Use together with `set_text`: after `apply_completion` (autocompletion selection backfill), position the cursor via `AppliedCompletion`'s `cursor_position`
    /// (T-TU-03-3 contract, consumed by the assembly layer's autocomplete-wiring-fix).
    /// Consumer: autocomplete-wiring-fix (`InteractiveUi::apply_selection`).
    pub fn set_cursor(&mut self, line: usize, col: usize) {
        let line = line.min(self.state.lines.len().saturating_sub(1));
        self.state.cursor_line = line;
        self.state.cursor_col = col.min(self.state.lines[line].len());
        self.preferred_visual_col = None;
        self.snapped_from_cursor_col = None;
    }

    /// Programmatically insert text at the cursor (multi-line supported). Undo atomicity: a single call = a single snapshot (T-TU-03-5,
    /// insert_text always pushes internally; one undo restores the entire state before the insertion).
    /// Consumers: T-AP-11-2, T-AP-11-4.
    pub fn insert_text_at_cursor(&mut self, text: &str) {
        self.insert_text(text);
    }

    /// Add a committed input to history (trim then dedupe, cap 100, aligned with pi addToHistory).
    /// Consumer: T-AP-11-2 (called after each successful commit).
    pub fn add_to_history(&mut self, text: &str) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return;
        }

        // Dedupe: if identical to the most recent entry, don't push again (aligned with pi: history[0] dedupe)
        if self.history.first().map_or(false, |last| last == trimmed) {
            return;
        }

        // Insert at front: history[0] = most recent (aligned with pi `unshift`, so index semantics "0=most recent, 1=older" hold)
        self.history.insert(0, trimmed.to_string());

        // Cap 100: evict the oldest at the tail (aligned with pi `pop`)
        if self.history.len() > 100 {
            self.history.pop();
        }
    }

    /// Set left/right padding (affects layout and wrap width).
    /// Consumer: T-AP-11-2 (user configuration getEditorPaddingX).
    pub fn set_padding_x(&mut self, padding: usize) {
        self.padding_x = padding;
    }

    /// Replace border style (focus color change at app layer, etc.).
    /// Consumer: T-AP-11-2 (corresponds to updateEditorBorderColor).
    pub fn set_border_style(&mut self, style: crate::screen::Style) {
        self.border_style = style;
    }

    /// Trigger on_change callback, passing the current full text.
    /// Shared across submodules (text.rs editing operations and history.rs commit/history browsing both call it), hence `pub(super)`.
    pub(super) fn emit_change(&mut self) {
        if let Some(cb) = &mut self.on_change {
            let text = self.state.lines.join("\n");
            cb(&text);
        }
    }

    /// Register submit callback (triggered by Enter; argument is the trimmed full text).
    /// Consumers: T-AP-11-2 (CustomEditor pass-through), T-AP-11-1.
    pub fn set_on_submit(&mut self, f: impl FnMut(&str) + 'static) {
        self.on_submit = Some(Box::new(f));
    }

    /// Register text-change callback (triggered after each edit operation; argument is the current full text).
    /// Consumers: T-AP-11-2 (CustomEditor pass-through), T-AP-11-1.
    pub fn set_on_change(&mut self, f: impl FnMut(&str) + 'static) {
        self.on_change = Some(Box::new(f));
    }

    /// Set focus state (when focused, renders the inverse block cursor and allows cursor positioning; called by the application layer on focus changes).
    /// Consumers: T-AP-11-2 (CustomEditor pass-through), T-AP-11-1.
    pub fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
    }

    /// Current focus state.
    pub fn is_focused(&self) -> bool {
        self.focused
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn capture() -> Rc<RefCell<Vec<String>>> {
        Rc::new(RefCell::new(Vec::new()))
    }

    // ── EditorState::default ──

    #[test]
    fn test_editor_state_default_has_single_empty_line() {
        let state = EditorState::default();
        assert_eq!(state.lines, vec![""]);
        assert_eq!(state.cursor_line, 0);
        assert_eq!(state.cursor_col, 0);
    }

    // ── Editor::new ──

    #[test]
    fn test_new_creates_empty_editor() {
        let e = Editor::new(0);
        assert_eq!(e.text(), "");
        assert_eq!(e.cursor(), (0, 0));
        assert_eq!(e.lines(), &[""]);
    }

    #[test]
    fn test_new_with_padding() {
        let e = Editor::new(4);
        assert_eq!(e.padding_x, 4);
    }

    // ── Editor::text ──

    #[test]
    fn test_text_empty() {
        let e = Editor::new(0);
        assert_eq!(e.text(), "");
    }

    #[test]
    fn test_text_single_line() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_text_multiline_joined_by_newline() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("a\nb\nc");
        assert_eq!(e.text(), "a\nb\nc");
    }

    // ── Editor::set_text ──

    #[test]
    fn test_set_text_simple() {
        let mut e = Editor::new(0);
        e.set_text("hello world");
        assert_eq!(e.text(), "hello world");
        assert_eq!(e.cursor(), (0, 11));
    }

    #[test]
    fn test_set_text_normalizes_crlf() {
        let mut e = Editor::new(0);
        e.set_text("line1\r\nline2");
        assert_eq!(e.text(), "line1\nline2");
    }

    #[test]
    fn test_set_text_normalizes_cr() {
        let mut e = Editor::new(0);
        e.set_text("line1\rline2");
        assert_eq!(e.text(), "line1\nline2");
    }

    #[test]
    fn test_set_text_expands_tabs() {
        let mut e = Editor::new(0);
        e.set_text("a\tb");
        assert_eq!(e.text(), "a    b");
    }

    #[test]
    fn test_set_text_multiline_cursor_at_end() {
        let mut e = Editor::new(0);
        e.set_text("abc\ndef");
        assert_eq!(e.cursor(), (1, 3));
    }

    #[test]
    fn test_set_text_resets_scroll_and_sticky() {
        let mut e = Editor::new(0);
        e.scroll_offset = 10;
        e.preferred_visual_col = Some(5);
        e.snapped_from_cursor_col = Some(3);
        e.set_text("new");
        assert_eq!(e.scroll_offset, 0);
        assert!(e.preferred_visual_col.is_none());
        assert!(e.snapped_from_cursor_col.is_none());
    }

    #[test]
    fn test_set_text_triggers_on_change() {
        let mut e = Editor::new(0);
        let changes = capture();
        {
            let c = changes.clone();
            e.on_change = Some(Box::new(move |text: &str| {
                c.borrow_mut().push(text.to_string());
            }));
        }
        e.set_text("hello");
        assert_eq!(*changes.borrow(), vec!["hello"]);
    }

    #[test]
    fn test_set_text_empty_string() {
        let mut e = Editor::new(0);
        e.set_text("");
        assert_eq!(e.text(), "");
        assert_eq!(e.lines(), &[""]);
    }

    // ── Editor::lines ──

    #[test]
    fn test_lines_empty() {
        let e = Editor::new(0);
        assert_eq!(e.lines().len(), 1);
        assert_eq!(e.lines()[0], "");
    }

    #[test]
    fn test_lines_multiline() {
        let mut e = Editor::new(0);
        e.set_text("a\nb\nc");
        assert_eq!(e.lines(), &["a", "b", "c"]);
    }

    // ── Editor::cursor ──

    #[test]
    fn test_cursor_initial() {
        let e = Editor::new(0);
        assert_eq!(e.cursor(), (0, 0));
    }

    #[test]
    fn test_cursor_after_insert() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        assert_eq!(e.cursor(), (0, 5));
    }

    // ── Editor::insert_text_at_cursor ──

    #[test]
    fn test_insert_text_at_cursor_single_line() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn test_insert_text_at_cursor_multiline() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("a\nb\nc");
        assert_eq!(e.text(), "a\nb\nc");
        assert_eq!(e.cursor(), (2, 1));
    }

    #[test]
    fn test_insert_text_at_cursor_mid_line() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello world");
        // Move cursor to after "hello "
        e.move_char(-5); // cursor at (0, 6)
        e.insert_text_at_cursor("beautiful ");
        assert_eq!(e.text(), "hello beautiful world");
    }

    // ── Editor::add_to_history ──

    #[test]
    fn test_add_to_history_basic() {
        let mut e = Editor::new(0);
        e.add_to_history("first");
        e.add_to_history("second");
        // history[0] = most recent
        assert_eq!(e.history[0], "second");
        assert_eq!(e.history[1], "first");
    }

    #[test]
    fn test_add_to_history_trims_whitespace() {
        let mut e = Editor::new(0);
        e.add_to_history("  hello  ");
        assert_eq!(e.history[0], "hello");
    }

    #[test]
    fn test_add_to_history_skips_empty() {
        let mut e = Editor::new(0);
        e.add_to_history("");
        e.add_to_history("   ");
        assert!(e.history.is_empty());
    }

    #[test]
    fn test_add_to_history_deduplicates_recent() {
        let mut e = Editor::new(0);
        e.add_to_history("same");
        e.add_to_history("same");
        assert_eq!(e.history.len(), 1);
    }

    #[test]
    fn test_add_to_history_allows_non_consecutive_duplicates() {
        let mut e = Editor::new(0);
        e.add_to_history("a");
        e.add_to_history("b");
        e.add_to_history("a"); // not consecutive with first "a"
        assert_eq!(e.history.len(), 3);
    }

    #[test]
    fn test_add_to_history_limit_100() {
        let mut e = Editor::new(0);
        for i in 0..105 {
            e.add_to_history(&format!("entry-{i}"));
        }
        assert_eq!(e.history.len(), 100);
        // Most recent is first
        assert_eq!(e.history[0], "entry-104");
        // Oldest surviving is entry-5 (entries 0-4 were popped)
        assert_eq!(e.history[99], "entry-5");
    }

    // ── Editor::set_padding_x ──

    #[test]
    fn test_set_padding_x() {
        let mut e = Editor::new(0);
        e.set_padding_x(8);
        assert_eq!(e.padding_x, 8);
    }

    // ── Editor::set_border_style ──

    #[test]
    fn test_set_border_style() {
        let mut e = Editor::new(0);
        let style = Style::default();
        e.set_border_style(style);
        // No panic; border_style is set (fields are private, so we just verify no crash)
    }

    // ── Editor::emit_change ──

    #[test]
    fn test_emit_change_fires_callback() {
        let mut e = Editor::new(0);
        let changes = capture();
        {
            let c = changes.clone();
            e.on_change = Some(Box::new(move |text: &str| {
                c.borrow_mut().push(text.to_string());
            }));
        }
        e.insert_text_at_cursor("hello");
        e.emit_change();
        // insert_text_at_cursor already fired once, emit_change fires again
        assert_eq!(changes.borrow().len(), 2);
        assert_eq!(changes.borrow()[1], "hello");
    }

    #[test]
    fn test_emit_change_no_callback_is_noop() {
        let mut e = Editor::new(0);
        e.insert_text_at_cursor("hello");
        // No on_change set; should not panic
        e.emit_change();
    }

    #[test]
    fn test_emit_change_multiline() {
        let mut e = Editor::new(0);
        let changes = capture();
        {
            let c = changes.clone();
            e.on_change = Some(Box::new(move |text: &str| {
                c.borrow_mut().push(text.to_string());
            }));
        }
        e.set_text("a\nb");
        e.emit_change();
        assert_eq!(changes.borrow().last().unwrap(), "a\nb");
    }
}
