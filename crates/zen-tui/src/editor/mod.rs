use crate::screen::Style;
use unicode_segmentation::UnicodeSegmentation;

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
    /// Undo the most recent edit (default `Ctrl+-`; behavior is integrated by T-TU-03-5).
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
    on_submit: Option<Box<dyn FnMut(&str)>>, // Submit callback; receives the expanded text after trim (no marker in this task, i.e. trim of the full text)
    on_change: Option<Box<dyn FnMut(&str)>>, // Called on every text change
    disable_submit: bool,                    // When set, Submit is not triggered
}

/// Text fragment: a contiguous slice of the source text (text == line[start_index..end_index], byte offset).
struct TextChunk {
    text: String,       // Fragment text (visible text after wrapping; display is the buffer's source text)
    start_index: usize, // Starting byte offset in the source text (UTF-8, same unit as cursor_col/grapheme_indices)
    end_index: usize,   // Ending byte offset in the source text (exclusive; used for cursor attribution / range checks)
}

/// Rendered layout line: one line of visible text + whether it carries the cursor (with its position within the line when it does).
struct LayoutLine {
    text: String,              // Display text of this visual line (a wrap fragment or the whole line)
    has_cursor: bool,          // Whether this line carries the cursor
    cursor_pos: Option<usize>, // Cursor position within this line's text (Some when has_cursor = true)
}

/// Visual line mapping entry: the logical line corresponding to this visual line, and the [start_col, start_col+length) range within it.
/// Generated by `build_visual_line_map`, consumed by `move_vertical`/`find_current_visual_line`/`page_scroll`.
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
            on_submit: None,
            on_change: None,
            disable_submit: false,
        }
    }

    /// Returns the full text (lines joined by \n). Consumers read the snapshot; modification doesn't go through this path.
    pub fn text(&self) -> String {
        self.state.lines.join("\n")
    }

    /// Replace the text and reset the cursor to the end of the line (programmatic setting: initial value, external editor write-back T-AP-11-4, history restore).
    /// Normalize newlines (\r\n/\r → \n) and expand tabs (→ 4 spaces, aligning with pi normalizeText).
    pub fn set_text(&mut self, text: &str) {
        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
        self.state.lines = normalized
            .split('\n')
            .map(|s| s.replace('\t', "    "))
            .collect();
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
    pub fn lines(&self) -> &[String] {
        &self.state.lines
    }

    /// Return the logical cursor position (line, column).
    pub fn cursor(&self) -> (usize, usize) {
        (self.state.cursor_line, self.state.cursor_col)
    }

    /// Programmatically insert text at the cursor (supports multiple lines). Atomic with respect to undo.
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

    /// Handle one key press. Only responds to `KeyEventKind::Press` (Release/Repeat ignored, per T-TU-03-1 semantics or filtered in this method).
    /// Dispatch rules: printable characters (with Shift, no Ctrl/Alt modifiers) → insert_char; otherwise consult the key binding table (inline default,
    /// replaced by its key definition table once T-TU-03-1 lands); on `EditorAction` match → `handle_action`; app-level actions like Ctrl+C
    /// are not routed to the editor (handled upstream, aligning with pi tui.input.copy "let parent handle").
    /// Enter semantics (aligned with pi): plain Enter=Submit; Shift+Enter / Ctrl+J=NewLine; terminals have no Shift+Enter,
    /// Backslash fallback when supported (backspace + newline/submit when the character before the cursor is `\`), inlined per inner default keybindings
    pub fn handle_key(&mut self, key: crossterm::event::KeyEvent) {
        use crossterm::event::{KeyCode, KeyEventKind, KeyModifiers};

        // Only respond to Press events
        if key.kind != KeyEventKind::Press {
            return;
        }

        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);

        match key.code {
            // ── Printable characters (no Ctrl/Alt modifiers; Shift not excluded) ──
            // insert_char exits history browsing itself; no need to repeat here
            KeyCode::Char(c) if !ctrl && !alt => {
                self.insert_char(c);
            }

            // ── Enter family (aligned with pi) ──
            // Ctrl+Enter is unbound (pi binds only plain enter=submit, shift+enter/ctrl+j=newline)
            KeyCode::Enter if ctrl => {}
            // Shift+Enter = NewLine (includes `\` fallback: backspace + submit)
            KeyCode::Enter if shift => self.handle_newline(),
            // Plain Enter = Submit (pi: disableSubmit checked first, `\` falls back to backspace + newline)
            KeyCode::Enter => {
                if self.disable_submit {
                    return;
                }
                if self.char_before_cursor_is_backslash() {
                    self.backspace();
                    self.newline();
                    return;
                }
                self.submit();
            }
            // Ctrl+J = NewLine (terminals typically lack Shift+Enter, includes `\` fallback)
            KeyCode::Char('j') if ctrl => self.handle_newline(),

            // ── Ctrl+character actions ──
            KeyCode::Char('a') if ctrl => self.handle_action(EditorAction::MoveToLineStart),
            KeyCode::Char('e') if ctrl => self.handle_action(EditorAction::MoveToLineEnd),
            KeyCode::Char('b') if ctrl => self.handle_action(EditorAction::MoveCursorLeft),
            KeyCode::Char('f') if ctrl => self.handle_action(EditorAction::MoveCursorRight),
            KeyCode::Char('w') if ctrl => self.handle_action(EditorAction::DeleteWordBackward),
            KeyCode::Char('u') if ctrl => self.handle_action(EditorAction::DeleteToLineStart),
            KeyCode::Char('k') if ctrl => self.handle_action(EditorAction::DeleteToLineEnd),
            KeyCode::Char('d') if ctrl => self.handle_action(EditorAction::DeleteCharForward),
            KeyCode::Char('y') if ctrl => self.handle_action(EditorAction::Yank),

            // ── Alt+character actions (aligned with pi TUI_KEYBINDINGS defaults: alt+b/f/d/y) ──
            KeyCode::Char('b') if alt => self.handle_action(EditorAction::MoveCursorWordLeft),
            KeyCode::Char('f') if alt => self.handle_action(EditorAction::MoveCursorWordRight),
            KeyCode::Char('d') if alt => self.handle_action(EditorAction::DeleteWordForward),
            KeyCode::Char('y') if alt => self.handle_action(EditorAction::YankPop),

            // ── Arrow keys ──
            KeyCode::Up => self.handle_action(EditorAction::MoveCursorUp),
            KeyCode::Down => self.handle_action(EditorAction::MoveCursorDown),
            KeyCode::Left if !ctrl && !alt => self.handle_action(EditorAction::MoveCursorLeft),
            KeyCode::Right if !ctrl && !alt => self.handle_action(EditorAction::MoveCursorRight),
            KeyCode::Left if ctrl || alt => self.handle_action(EditorAction::MoveCursorWordLeft),
            KeyCode::Right if ctrl || alt => self.handle_action(EditorAction::MoveCursorWordRight),

            // ── Editing keys (modifier guards: alt+backspace/delete = word delete, aligned with pi) ──
            KeyCode::Backspace if alt => self.handle_action(EditorAction::DeleteWordBackward),
            KeyCode::Backspace => self.handle_action(EditorAction::DeleteCharBackward),
            KeyCode::Delete if alt => self.handle_action(EditorAction::DeleteWordForward),
            KeyCode::Delete => self.handle_action(EditorAction::DeleteCharForward),
            KeyCode::Home => self.handle_action(EditorAction::MoveToLineStart),
            KeyCode::End => self.handle_action(EditorAction::MoveToLineEnd),
            KeyCode::PageUp => self.handle_action(EditorAction::PageUp),
            KeyCode::PageDown => self.handle_action(EditorAction::PageDown),

            // ── Unmapped keys: ignore, pass to parent ──
            _ => {}
        }
    }

    /// Handle one bracketed-paste: normalize newlines/tabs, filter non-printable chars, insert the whole block atomically.
    pub fn handle_paste(&mut self, text: &str) {
        // Normalize newlines + expand tabs (aligned with pi normalizeText: **expand tabs before filtering**,
        // otherwise \t is a control character and would be dropped by the filter below, making the expansion dead code)
        let cleaned: String = text
            .replace("\r\n", "\n")
            .replace('\r', "\n")
            .replace('\t', "    ");

        // Filter non-printable characters (keep \n and ASCII ≥32 / non-ASCII; aligned with pi: char === "\n" || charCodeAt(0) >= 32)
        let filtered: String = cleaned
            .chars()
            .filter(|c| *c == '\n' || (*c as u32) >= 32)
            .collect();

        if filtered.is_empty() {
            return;
        }

        self.exit_history_browsing();
        self.insert_text(&filtered);
    }

    /// Whether the char before the cursor is a backslash (terminal fallback check for `\` + Enter, aligned with pi shouldSubmitOnBackslashEnter).
    /// cursor_col is always at a grapheme boundary: the previous byte is the last byte of the previous grapheme; a continuation byte of a multi-byte grapheme is never b'\\', so the check is safe.
    fn char_before_cursor_is_backslash(&self) -> bool {
        self.state.cursor_col > 0
            && self.state.lines[self.state.cursor_line]
                .as_bytes()
                .get(self.state.cursor_col - 1)
                == Some(&b'\\')
    }

    /// Shared newline entry point for Shift+Enter / Ctrl+J: when the char before the cursor is `\` and submit is not disabled
    /// Backspace + submit (aligning with pi shouldSubmitOnBackslashEnter → backspace + submitValue),
    /// Otherwise insert a newline normally.
    fn handle_newline(&mut self) {
        if !self.disable_submit && self.char_before_cursor_is_backslash() {
            self.backspace();
            self.submit();
        } else {
            self.newline();
        }
    }

    /// Whether the editor is empty (a single blank line), matching pi isEditorEmpty (used for cursorUp history recall checks).
    fn is_editor_empty(&self) -> bool {
        self.state.lines.len() == 1 && self.state.lines[0].is_empty()
    }

    /// Whether the cursor is on the first visual line (matching pi isOnFirstVisualLine).
    fn is_on_first_visual_line(&self) -> bool {
        let map = self.build_visual_line_map(self.last_width);
        self.find_current_visual_line(&map) == 0
    }

    /// Whether the cursor is on the last visual line (matching pi isOnLastVisualLine).
    fn is_on_last_visual_line(&self) -> bool {
        let map = self.build_visual_line_map(self.last_width);
        self.find_current_visual_line(&map) == map.len().saturating_sub(1)
    }

    /// Dispatch editing behavior by action. **The single entry point for all mutating operations**.
    pub fn handle_action(&mut self, action: EditorAction) {
        match action {
            // ── Movement (first/last visible line transfer: history browsing / line start & end, aligned with pi moveCursorUp/Down three branches) ──
            EditorAction::MoveCursorUp => {
                if self.is_on_first_visual_line()
                    && (self.is_editor_empty()
                        || self.history_index > -1
                        || self.state.cursor_col == 0)
                {
                    self.navigate_history(-1);
                } else if self.is_on_first_visual_line() {
                    self.move_to_line_start();
                } else {
                    self.move_vertical(-1);
                }
            }
            EditorAction::MoveCursorDown => {
                if self.history_index > -1 && self.is_on_last_visual_line() {
                    self.navigate_history(1);
                } else if self.is_on_last_visual_line() {
                    self.move_to_line_end();
                } else {
                    self.move_vertical(1);
                }
            }
            EditorAction::MoveCursorLeft => self.move_char(-1),
            EditorAction::MoveCursorRight => self.move_char(1),
            EditorAction::MoveCursorWordLeft => self.move_word(false),
            EditorAction::MoveCursorWordRight => self.move_word(true),
            EditorAction::MoveToLineStart => self.move_to_line_start(),
            EditorAction::MoveToLineEnd => self.move_to_line_end(),
            EditorAction::PageUp => self.page_scroll(false),
            EditorAction::PageDown => self.page_scroll(true),

            // ── Editing (exit history browsing) ──
            EditorAction::DeleteCharBackward => {
                self.exit_history_browsing();
                self.backspace();
            }
            EditorAction::DeleteCharForward => {
                self.exit_history_browsing();
                self.delete_forward();
            }
            EditorAction::DeleteWordBackward => {
                self.exit_history_browsing();
                self.delete_word_backward();
            }
            EditorAction::DeleteWordForward => {
                self.exit_history_browsing();
                self.delete_word_forward();
            }
            EditorAction::DeleteToLineStart => {
                self.exit_history_browsing();
                self.delete_to_line_start();
            }
            EditorAction::DeleteToLineEnd => {
                self.exit_history_browsing();
                self.delete_to_line_end();
            }
            EditorAction::NewLine => {
                self.exit_history_browsing();
                self.newline();
            }
            EditorAction::Submit => {
                // disable_submit: once set, Submit does not trigger (aligned with the disableSubmit check in pi submit branch)
                if !self.disable_submit {
                    self.submit();
                }
            }

            // ── Unimplemented (reserved for T-TU-03-5/-7/-3) ──
            EditorAction::Yank => { /* no-op */ }
            EditorAction::YankPop => { /* no-op */ }
            EditorAction::Undo => { /* no-op */ }
            EditorAction::Tab => { /* no-op */ }

            // ── Jump mode (reserved for T-TU-03) ──
            EditorAction::JumpForward => { /* no-op */ }
            EditorAction::JumpBackward => { /* no-op */ }
        }
    }

    /// Render the editor into the screen buffer at (x,y), with width width and height height. Design:
    /// Effective padding is clamped by available width (`min(padding_x, (width-1)/2)`, aligned with pi) → layout width
    /// When wrapping at the outer margin with no padding, reserve 1 column for the line-end cursor (aligned with pi layoutWidth calculation, deducted only once) →
    /// Line layout → scroll clamping (visible_rows = clamp(max(5, 30%·height), 1, height-2), aligned with pi)
    /// 30% of terminal height; upper bound h-2 prevents text from covering the top/bottom borders) → write to the buffer line by line: top/bottom borders
    /// (includes `─── ↑N more` / `─── ↓N more` scroll indicators; padding width is calculated from display width), text lines,
    /// Cursor inverse (`Style.attributes=[Reverse]` block: if the cursor is on a grapheme, that grapheme is inverted; at line end, a space is inverted).
    /// Consumer: T-AP-11-1 (called once inside the draw callback).
    pub fn render(
        &mut self,
        buf: &mut crate::screen::ScreenBuffer,
        x: u16,
        y: u16,
        width: u16,
        height: u16,
    ) {
        use unicode_segmentation::UnicodeSegmentation;
        use unicode_width::UnicodeWidthStr;

        if width < 2 || height < 3 {
            return; // Too small to render (minimum: 1 text column + 2 border rows)
        }

        let w = width as usize;
        let h = height as usize;

        // ── 1. Layout width (aligned with pi layoutWidth calculation, padding deducted only once) ──
        // Layout is constrained by available width: max_padding = (w-1)/2 (aligned with pi Math.min(paddingX, maxPadding));
        // With no padding, reserve 1 column for the line-end cursor (aligned with pi contentWidth - (paddingX ? 0 : 1)).
        let max_padding = (w.saturating_sub(1)) / 2;
        let pad = self.padding_x.min(max_padding);
        // Record effective padding for cursor_position to resolve column coordinates (consistent with this layout pass)
        self.last_effective_padding = pad;
        let content_width = (w - pad * 2).max(1);
        let layout_width = (content_width - if pad == 0 { 1 } else { 0 }).max(1);
        self.last_width = layout_width;
        self.last_height = h;

        let layout = self.layout_text(layout_width);
        let total_lines = layout.len();

        // ── Visible rows: max(5, 30%·height), clamped to [1, h-2] to prevent border overlap (aligned with pi maxVisibleLines)──
        let visible_rows = std::cmp::max(5, h * 30 / 100).clamp(1, h - 2);

        // ── 2. Scroll clamping: ensure the visual line containing the cursor is in the visible area, then clamp to bottom alignment (aligned with pi maxScrollOffset) ──
        let cursor_vis = layout.iter().position(|l| l.has_cursor).unwrap_or(0);
        if cursor_vis < self.scroll_offset {
            self.scroll_offset = cursor_vis;
        } else if cursor_vis >= self.scroll_offset + visible_rows {
            self.scroll_offset = cursor_vis - visible_rows + 1;
        }
        self.scroll_offset = self
            .scroll_offset
            .min(total_lines.saturating_sub(visible_rows));

        // ── 3. Styles ──
        let border_style = self.border_style.clone();
        let text_style = Style::default();
        let cursor_style = Style {
            attributes: vec![crossterm::style::Attribute::Reverse],
            ..Style::default()
        };

        // Text area left origin: x + effective padding (pad ≤ (w-1)/2, u16 conversion is safe)
        let text_x = x.saturating_add(pad as u16);

        // ── 4. Top border ──
        let border_y = y;
        if self.scroll_offset > 0 {
            let indicator = format!(" ─── ↑{} more ", self.scroll_offset);
            // Fill width is computed by display width (indicator.len() returns byte count and includes multi-byte ─/↑, which undercounts)
            let dashes = w.saturating_sub(crate::screen::visible_width(&indicator));
            buf.write(
                x,
                border_y,
                &border_style,
                &format!("{}{}", indicator, "─".repeat(dashes)),
            );
        } else {
            buf.write(x, border_y, &border_style, &"─".repeat(w));
        }

        // ── 5. Text rows ──
        for i in 0..visible_rows {
            let vis_idx = self.scroll_offset + i;
            let row_y = y + 1 + i as u16;

            if vis_idx < total_lines {
                let line = &layout[vis_idx];

                // Write text
                buf.write(text_x, row_y, &text_style, &line.text);

                // Cursor reverse-video (layout_text guarantees cursor_pos is always Some when has_cursor is set)
                if line.has_cursor {
                    let cursor_pos = line.cursor_pos.unwrap_or(0);
                    let cx = text_x.saturating_add(cursor_pos as u16);
                    let graphemes: Vec<&str> =
                        line.text.grapheme_indices(true).map(|(_, g)| g).collect();

                    // Find the grapheme at the cursor: grapheme's starting display column == cursor_pos
                    let mut col = 0usize;
                    let mut found = false;
                    for g in &graphemes {
                        let gw = UnicodeWidthStr::width(*g);
                        if col == cursor_pos {
                            // Cursor on a grapheme: reverse-video that grapheme
                            buf.write(cx, row_y, &cursor_style, g);
                            found = true;
                            break;
                        }
                        col += gw;
                    }
                    if !found {
                        // Cursor at end of line (or on an empty line): reverse-video a space
                        buf.write(cx, row_y, &cursor_style, " ");
                    }
                }
            }
            // Lines with no content in the window can stay blank: the differential refresh (screen.rs flush) clears leftovers from the old frame
        }

        // ── 6. Bottom border ──
        let bottom_y = y + (h - 1) as u16;
        let remaining = total_lines.saturating_sub(self.scroll_offset + visible_rows);
        if remaining > 0 {
            let indicator = format!(" ─── ↓{} more ", remaining);
            let dashes = w.saturating_sub(crate::screen::visible_width(&indicator));
            buf.write(
                x,
                bottom_y,
                &border_style,
                &format!("{}{}", indicator, "─".repeat(dashes)),
            );
        } else {
            buf.write(x, bottom_y, &border_style, &"─".repeat(w));
        }
    }

    /// Returns the hardware cursor position of the cursor in the editor's **own coordinate system** (visual row, column, relative to render's x/y origin;
    /// the application layer adds the MoveTo offset on top). Corresponds to pi CURSOR_MARKER → TUI positioning semantics (used for IME candidate window positioning).
    /// Layout width must match render (visual lines are parsed using the current value of last_width).
    /// When to call: after render — scroll_offset is only clamped during render, so if the cursor moves without a redraw, ...
    /// Row coordinate may fall outside the visible window; this function clamps it back inside (converges on the next render).
    /// Consumer: T-AP-11-2 (IME / caret positioning).
    pub fn cursor_position(&self) -> (u16, u16) {
        if self.last_width == 0 {
            return (0, 0);
        }

        let layout = self.layout_text(self.last_width);

        // Visible rows (same formula as render: max(5, 30%·height) clamped to [1, h-2]),
        // Fallback clamp for the row coordinate while scroll_offset has not yet been clamped by the next render
        let visible = std::cmp::max(5, self.last_height * 30 / 100)
            .clamp(1, self.last_height.saturating_sub(2).max(1));

        // Find the visual row containing the caret
        for (vis_idx, line) in layout.iter().enumerate() {
            if line.has_cursor {
                // Row: relative position within the visible region (minus scroll offset, plus border offset 1)
                let in_window = vis_idx.saturating_sub(self.scroll_offset).min(visible - 1);
                let row = (in_window + 1) as u16;
                // Column: caret's visual column in the text + effective left padding (clamping consistent with render)
                let col = (line.cursor_pos.unwrap_or(0) + self.last_effective_padding) as u16;
                return (col, row);
            }
        }

        // Fallback (normally unreachable: an empty editor still produces a layout row with a caret; kept defensively)
        (self.last_effective_padding as u16, 1)
    }

    // ── Internal operations (private; called only by handle_action/handle_key/handle_paste and this module's tests; not marked pub) ──

    /// Insert a single grapheme at the cursor and move the cursor right; when the inserted character merges with the preceding character into a single grapheme (combining characters),
    /// snap to the end of the merged grapheme (maintaining the "cursor is always at a grapheme boundary" invariant). Triggers on_change.
    /// Undo coalescing reserved; see T-TU-03-5.
    fn insert_char(&mut self, c: char) {
        self.exit_history_browsing();

        let col = self.state.cursor_col;
        {
            let line = &mut self.state.lines[self.state.cursor_line];
            // cursor_col is always at a grapheme (and thus character) boundary; insert inserts at the boundary
            line.insert(col, c);
        }

        // Move the cursor right by the byte length of one character, then snap to the grapheme boundary (combining characters merged via insert may cross the boundary)
        self.set_cursor_col(col + c.len_utf8());
        self.snap_cursor_to_grapheme_boundary();

        // Fire the change callback
        self.emit_change();
    }

    /// Exit history browsing state: called when an edit action starts,
    /// preventing subsequent ↑/↓ from misinterpreting the edited state as history browsing.
    fn exit_history_browsing(&mut self) {
        self.history_index = -1;
        self.history_draft = None;
    }

    /// Snap the cursor to the end of its containing grapheme, maintaining the "cursor is always at a grapheme boundary" invariant.
    /// No-op when the cursor is already at the start of a grapheme or at end of line; only advances when a combining character merge leaves the cursor mid-grapheme.
    fn snap_cursor_to_grapheme_boundary(&mut self) {
        use unicode_segmentation::UnicodeSegmentation;

        let col = self.state.cursor_col;
        let line = &self.state.lines[self.state.cursor_line];
        // Already at end of line or exactly at a grapheme start: no snapping needed
        if col == line.len() || line.grapheme_indices(true).any(|(bo, _)| bo == col) {
            return;
        }
        // col falls inside a grapheme: snap to the end of that grapheme
        let snapped = line
            .grapheme_indices(true)
            .find(|(bo, g)| *bo <= col && col < bo + g.len())
            .map(|(bo, g)| bo + g.len())
            .unwrap_or(line.len());
        self.set_cursor_col(snapped);
    }

    /// Trigger on_change callback, passing the current full text.
    fn emit_change(&mut self) {
        if let Some(cb) = &mut self.on_change {
            let text = self.state.lines.join("\n");
            cb(&text);
        }
    }

    /// Backspace: delete the grapheme before the cursor; at line start, merge with the previous line (cursor moves to the end of the previous line). Triggers on_change.
    fn backspace(&mut self) {
        self.exit_history_browsing();

        let col = self.state.cursor_col;

        if col > 0 {
            // Collect only (byte offset, grapheme byte length); no &str borrow
            let graphemes: Vec<(usize, usize)> = self.state.lines[self.state.cursor_line]
                .grapheme_indices(true)
                .map(|(bo, g)| (bo, g.len()))
                .collect();

            if let Some(&(g_start, g_len)) = graphemes.iter().rfind(|(bo, _)| *bo < col) {
                let g_end = g_start + g_len;
                // graphemes no longer borrow self, so mutable borrow is safe
                let line = &mut self.state.lines[self.state.cursor_line];
                line.replace_range(g_start..g_end, "");
                self.set_cursor_col(g_start);
            }

            self.emit_change();
        } else if self.state.cursor_line > 0 {
            // ── Backspace at line start: merge with previous line ──
            let current_line = self.state.lines.remove(self.state.cursor_line);
            self.state.cursor_line -= 1;
            let prev_len = self.state.lines[self.state.cursor_line].len();
            self.state.lines[self.state.cursor_line].push_str(&current_line);
            self.set_cursor_col(prev_len);

            self.emit_change();
        }
        // Start of first line: no-op
    }

    /// Delete forward: delete the grapheme at (or containing) the cursor; at line end, merge with the next line. Triggers on_change.
    fn delete_forward(&mut self) {
        use unicode_segmentation::UnicodeSegmentation;

        self.exit_history_browsing();

        let line_len = self.state.lines[self.state.cursor_line].len();
        let col = self.state.cursor_col;

        if col < line_len {
            // ── Delete forward in line: delete the grapheme at (or containing) the cursor ──
            // Condition = grapheme end passes col: delete the grapheme starting at col when the cursor is at a boundary,
            // delete the containing grapheme when the cursor is inside one (symmetric to backspace's rfind(bo < col)).
            let graphemes: Vec<(usize, usize)> = self.state.lines[self.state.cursor_line]
                .grapheme_indices(true)
                .map(|(bo, g)| (bo, g.len()))
                .collect();

            if let Some(&(g_start, g_len)) = graphemes.iter().find(|(bo, len)| *bo + *len > col) {
                let g_end = g_start + g_len;
                let line = &mut self.state.lines[self.state.cursor_line];
                line.replace_range(g_start..g_end, "");
                // Cursor stays put (following graphemes shift left); if the original cursor was inside the deleted grapheme (merge case), snap to the new boundary
            }

            self.snap_cursor_to_grapheme_boundary();
            self.emit_change();
        } else if self.state.cursor_line + 1 < self.state.lines.len() {
            // ── Delete forward at line end: merge with next line ──
            let next_line = self.state.lines.remove(self.state.cursor_line + 1);
            self.state.lines[self.state.cursor_line].push_str(&next_line);
            // Cursor stays put (remains at the old line end, i.e. the merge point)

            self.emit_change();
        }
        // End of last line: no-op
    }

    /// Insert multi-line text at the cursor (shared by enter/paste/yank); newlines split lines at the cursor.
    /// Normalize newlines (\r\n/\r → \n) and tabs (\t → 4 spaces, aligned with pi normalizeText),
    /// so this function is the sole insertion entry point; callers need not normalize. Triggers on_change.
    fn insert_text(&mut self, text: &str) {
        self.exit_history_browsing();

        let col = self.state.cursor_col;
        let normalized = normalize_text(text);
        let parts: Vec<&str> = normalized.split('\n').collect();

        if parts.len() == 1 {
            // ── Single line: insert directly ──
            let line = &mut self.state.lines[self.state.cursor_line];
            line.insert_str(col, parts[0]);
            self.set_cursor_col(col + parts[0].len());
        } else {
            // ── Multiple lines: split and insert ──
            let rest = self.state.lines[self.state.cursor_line][col..].to_string();
            self.state.lines[self.state.cursor_line].truncate(col);

            // First segment: append to end of current line
            self.state.lines[self.state.cursor_line].push_str(parts[0]);

            // Middle segment: insert line by line
            for part in &parts[1..parts.len() - 1] {
                self.state.cursor_line += 1;
                self.state
                    .lines
                    .insert(self.state.cursor_line, part.to_string());
            }

            // Last segment: new line = last segment + original second half
            self.state.cursor_line += 1;
            let last = format!("{}{}", parts[parts.len() - 1], rest);
            self.state.lines.insert(self.state.cursor_line, last);

            // Cursor sits after the last segment's original content (i.e., before the original second half)
            self.set_cursor_col(parts[parts.len() - 1].len());
        }

        // If the inserted content merges with a neighboring character into a single grapheme (combining char), snap to keep the grapheme-boundary invariant
        self.snap_cursor_to_grapheme_boundary();
        self.emit_change();
    }

    /// Line split: truncate the current line at the cursor; the second half becomes a new line and the cursor moves to the new line's start. Triggers on_change.
    fn newline(&mut self) {
        self.exit_history_browsing();

        let col = self.state.cursor_col;
        let line = &mut self.state.lines[self.state.cursor_line];

        // Truncate out the second half
        let rest = line[col..].to_string();
        line.truncate(col);

        // Insert the new line
        self.state.cursor_line += 1;
        self.state.lines.insert(self.state.cursor_line, rest);

        // Move the cursor to the new line's start
        self.set_cursor_col(0);

        self.emit_change();
    }

    /// Delete from cursor to line start (everything before the cursor); at line start, merge into the previous line (aligns with pi deleteToStartOfLine's col==0 cross-line branch).
    /// Trigger on_change. kill ring push is wired in by T-TU-03-7.
    fn delete_to_line_start(&mut self) {
        self.exit_history_browsing();

        let col = self.state.cursor_col;
        if col > 0 {
            // ── In-line: delete everything before the cursor ──
            let line = &mut self.state.lines[self.state.cursor_line];
            line.replace_range(..col, "");
            self.set_cursor_col(0);
        } else if self.state.cursor_line > 0 {
            // ── Start of line: merge into previous line (the deleted newline is recorded by the kill ring, T-TU-03-7) ──
            let current_line = self.state.lines.remove(self.state.cursor_line);
            self.state.cursor_line -= 1;
            let prev_len = self.state.lines[self.state.cursor_line].len();
            self.state.lines[self.state.cursor_line].push_str(&current_line);
            self.set_cursor_col(prev_len);
        }
        // First line start: no-op (pi still triggers onChange; keep consistent)

        self.emit_change();
    }

    /// Delete from cursor to line end; at line end, merge into the next line (aligns with pi deleteToEndOfLine). Triggers on_change.
    fn delete_to_line_end(&mut self) {
        self.exit_history_browsing();

        let col = self.state.cursor_col;
        let line_len = self.state.lines[self.state.cursor_line].len();

        if col >= line_len {
            // Line end: merge with the next line (aligns with pi deleteToEndOfLine cross-line semantics)
            if self.state.cursor_line + 1 < self.state.lines.len() {
                let next_line = self.state.lines.remove(self.state.cursor_line + 1);
                self.state.lines[self.state.cursor_line].push_str(&next_line);
                self.emit_change();
            }
            return;
        }

        let line = &mut self.state.lines[self.state.cursor_line];
        line.truncate(col);

        self.emit_change();
    }

    /// Delete one word backward (before the cursor); at line start, merge into the previous line. Word boundary reuses `move_word_backward`
    /// (aligns with pi structure: deleteWordBackwards calls moveWordBackwards to get the boundary); deletion and movement
    /// Always use the same word-boundary semantics. Fires on_change. Kill ring is wired in via T-TU-03-7.
    fn delete_word_backward(&mut self) {
        self.exit_history_browsing();

        let col = self.state.cursor_col;

        if col == 0 {
            // Line start: merge into the previous line (aligns with pi deleteWordBackwards's col==0 branch)
            if self.state.cursor_line > 0 {
                let current_line = self.state.lines.remove(self.state.cursor_line);
                self.state.cursor_line -= 1;
                let prev_len = self.state.lines[self.state.cursor_line].len();
                self.state.lines[self.state.cursor_line].push_str(&current_line);
                self.set_cursor_col(prev_len);
                self.emit_change();
            }
            return;
        }

        // Use move_word_backward to find the word boundary (aligns with pi: moveWordBackwards gets deleteFrom);
        // if the word fills the entire line before the cursor (and it's not the first line), it wraps across lines — deletion semantics must not cross lines,
        // so restore the current line and set the boundary to line start (the whole prefix is the word; delete it all).
        let old_line = self.state.cursor_line;
        self.move_word_backward();
        let delete_from = if self.state.cursor_line == old_line {
            self.state.cursor_col
        } else {
            self.state.cursor_line = old_line;
            0
        };
        self.set_cursor_col(col);

        let line = &mut self.state.lines[self.state.cursor_line];
        line.replace_range(delete_from..col, "");
        self.set_cursor_col(delete_from);

        self.emit_change();
    }

    /// Delete one word forward (after the cursor); at line end, merge into the next line. Word boundary reuses `move_word_forward`
    /// (aligns with pi structure: deleteWordForward calls moveWordForwards for the boundary) delete and move
    /// Always use the same word-boundary semantics. Fires on_change. Kill ring is wired in via T-TU-03-7.
    fn delete_word_forward(&mut self) {
        self.exit_history_browsing();

        let col = self.state.cursor_col;
        let line_len = self.state.lines[self.state.cursor_line].len();

        if col >= line_len {
            // End of line: merge into next line (aligns with pi deleteWordForward's cross-line branch)
            if self.state.cursor_line + 1 < self.state.lines.len() {
                let next_line = self.state.lines.remove(self.state.cursor_line + 1);
                self.state.lines[self.state.cursor_line].push_str(&next_line);
                self.emit_change();
            }
            return;
        }

        // Use move_word_forward to find the word boundary (aligns with pi: moveWordForwards obtains deleteTo);
        // If the word/whitespace extends to end of line, it wraps across lines — delete semantics don't cross lines,
        // so restore the current line and set the boundary to end of line (delete the whole segment from cursor to end of line).
        let old_line = self.state.cursor_line;
        self.move_word_forward();
        let delete_to = if self.state.cursor_line == old_line && self.state.cursor_col > col {
            self.state.cursor_col
        } else {
            self.state.cursor_line = old_line;
            line_len
        };
        self.set_cursor_col(col);

        let line = &mut self.state.lines[self.state.cursor_line];
        line.replace_range(col..delete_to, "");
        // Cursor stays put (the following graphemes shift forward)

        self.emit_change();
    }

    /// Submit: take full text trim(), clear state and history browsing state, call on_change("") then on_submit(text).
    /// (No paste marker; expansion step omitted; undo/kill cleanup is hooked up by T-TU-03-5/-7.)
    fn submit(&mut self) {
        let text = self.text().trim().to_string();

        // Clear editor state and browsing state (aligned with pi submitValue: exitHistoryBrowsing clears history_index/draft)
        self.state = EditorState::default();
        self.exit_history_browsing();
        self.preferred_visual_col = None;
        self.snapped_from_cursor_col = None;
        self.scroll_offset = 0;

        // Call on_change("") first (notify clear), then on_submit(text) (pass content)
        if let Some(cb) = &mut self.on_change {
            cb("");
        }
        if let Some(cb) = &mut self.on_submit {
            cb(&text);
        }
    }

    /// History browsing: direction=-1 older (up), +1 newer (down). Index semantics: 0=most recent, 1=older
    /// (History prepended, aligned with pi navigateHistory's `newIndex = historyIndex - direction`).
    /// On first entering browsing (any direction leaving -1), stash the current draft; on switching back to -1, restore the draft.
    /// Switching/restoring both trigger on_change (aligned with pi setTextInternal/draft restore path).
    fn navigate_history(&mut self, direction: i8) {
        if self.history.is_empty() {
            return;
        }

        let max_index = self.history.len() as isize - 1;

        // Align with pi: newIndex = historyIndex - direction (up = -1 → index increases = older)
        let new_index = self.history_index - direction as isize;

        // Clamp out-of-bounds
        if new_index < -1 || new_index > max_index {
            return;
        }

        // First entering browsing (first move out of -1, regardless of direction): stash the current draft
        if self.history_index == -1 && new_index >= 0 {
            self.history_draft = Some(self.state.clone());
        }

        self.history_index = new_index;

        if self.history_index == -1 {
            // Switch back to not-browsed: restore draft (align with pi restore path, reset cursor/scroll state)
            if let Some(draft) = self.history_draft.take() {
                self.state = draft;
                self.preferred_visual_col = None;
                self.snapped_from_cursor_col = None;
                self.scroll_offset = 0;
            } else {
                // No draft (abnormal path): clear, align with pi setTextInternal("")
                self.set_text_without_history("");
            }
            self.emit_change();
        } else {
            // Show history entry (cursor at end of line, aligning with setText behavior of pi navigateHistory)
            let text = self.history[self.history_index as usize].clone();
            self.set_text_without_history(&text);
            self.emit_change();
        }
    }

    /// set_text internal version: does not trigger add_to_history, avoiding duplicate pushes while browsing history.
    /// History entries are already first-class text in the editor (normalized during insert/paste — no \r, no \t),
    /// no further normalization needed (align with pi setTextInternal: display as-is, ensuring display == storage).
    fn set_text_without_history(&mut self, text: &str) {
        self.state.lines = text.split('\n').map(str::to_string).collect();
        if self.state.lines.is_empty() {
            self.state.lines.push(String::new());
        }
        // Move cursor to end of last line (aligning with setText behavior of pi navigateHistory)
        self.state.cursor_line = self.state.lines.len() - 1;
        self.state.cursor_col = self.state.lines[self.state.cursor_line].len();
        self.preferred_visual_col = None;
        self.snapped_from_cursor_col = None;
        self.scroll_offset = 0; // Align with pi setTextInternal reset of scrollOffset
    }

    /// Move cursor left/right by grapheme (cross-line wrap: right at line end → next line start, left at line start → previous line end).
    /// At line end the coordinates also record preferred_visual_col (matching pi moveCursor's last-line case).
    fn move_char(&mut self, delta_col: i32) {
        if delta_col == 0 {
            return;
        }

        let line = &self.state.lines[self.state.cursor_line];
        let graphemes: Vec<(usize, &str)> = line.grapheme_indices(true).collect();
        let g_count = graphemes.len();

        // current cursor index within the grapheme sequence (cursor_col is a byte offset and needs conversion)
        let current_g_idx = graphemes
            .iter()
            .position(|(byte_offset, _)| *byte_offset == self.state.cursor_col)
            .unwrap_or(g_count); // at line end (cursor_col == line.len()) → g_count

        let new_g_idx = current_g_idx as i32 + delta_col;

        if new_g_idx < 0 {
            // ── move left at line start → previous line end ──
            if self.state.cursor_line > 0 {
                self.state.cursor_line -= 1;
                let prev_line = &self.state.lines[self.state.cursor_line];
                let prev_len = prev_line.len();
                self.set_cursor_col(prev_len); // line end = byte length
            }
            // First line start: no move
        } else if new_g_idx as usize > g_count {
            // ── Line end move → next line start ──
            if self.state.cursor_line + 1 < self.state.lines.len() {
                self.state.cursor_line += 1;
                self.set_cursor_col(0);
            }
            // Last line end: no move
        } else {
            // ── Movement within the same line ──
            let target_g_idx = new_g_idx as usize;
            if target_g_idx == g_count {
                // Move to line end: record preferred_visual_col (aligned with pi moveCursor last-line case)
                self.state.cursor_col = line.len();
                self.preferred_visual_col = Some(self.cursor_visual_col());
                self.snapped_from_cursor_col = None;
            } else {
                self.set_cursor_col(graphemes[target_g_idx].0);
            }
        }
    }

    // Returns the cursor's visual column on the current line (accumulated grapheme widths, used by preferred_visual_col).
    fn cursor_visual_col(&self) -> usize {
        use unicode_segmentation::UnicodeSegmentation;
        use unicode_width::UnicodeWidthStr;

        let line = &self.state.lines[self.state.cursor_line];
        line.grapheme_indices(true)
            .take_while(|(byte_offset, _)| *byte_offset < self.state.cursor_col)
            .map(|(_, g)| unicode_width::UnicodeWidthStr::width(g))
            .sum()
    }

    /// Move cursor by word boundaries (forward/backward; line start/end crossing aligns with pi moveWordBackwards/Forwards).
    /// Word boundary detection currently uses an inline minimal implementation (unicode-segmentation word boundaries); to be replaced once T-TU-03-8 lands.
    fn move_word(&mut self, forward: bool) {
        use unicode_segmentation::UnicodeSegmentation;

        if forward {
            self.move_word_forward();
        } else {
            self.move_word_backward();
        }
    }

    /// Jump forward by word: skip past current word end → skip whitespace → stop at the start of the next word. Wraps across lines.
    fn move_word_forward(&mut self) {
        use unicode_segmentation::UnicodeSegmentation;

        let line = &self.state.lines[self.state.cursor_line];
        let graphemes: Vec<(usize, &str)> = line.grapheme_indices(true).collect();
        let g_count = graphemes.len();

        // Current grapheme index
        let mut g_idx = graphemes
            .iter()
            .position(|(bo, _)| *bo == self.state.cursor_col)
            .unwrap_or(g_count);

        // Step 1: skip remaining graphemes of the current word (non-whitespace)
        while g_idx < g_count {
            let g = graphemes[g_idx].1;
            if g.chars().all(|c| c.is_whitespace()) {
                break;
            }
            g_idx += 1;
        }

        // Step 2: skip whitespace
        while g_idx < g_count {
            let g = graphemes[g_idx].1;
            if !g.chars().all(|c| c.is_whitespace()) {
                break;
            }
            g_idx += 1;
        }

        if g_idx < g_count {
            // Stop at the start of the next word on the same line
            self.set_cursor_col(graphemes[g_idx].0);
        } else {
            // Reached end of line → wrap to start of next line, keep searching for a word
            if self.state.cursor_line + 1 < self.state.lines.len() {
                self.state.cursor_line += 1;
                self.set_cursor_col(0);
                // Recursion: the next line may be entirely whitespace
                self.move_word_forward();
            } else {
                // End of last line: stop at line end
                self.set_cursor_col(line.len());
            }
        }
    }

    /// Jump backward by word: skip whitespace backward → skip the previous word → stop at that word's start. Wraps across lines.
    fn move_word_backward(&mut self) {
        use unicode_segmentation::UnicodeSegmentation;

        let line = &self.state.lines[self.state.cursor_line];
        let graphemes: Vec<(usize, &str)> = line.grapheme_indices(true).collect();

        // Current grapheme index
        let mut g_idx = graphemes
            .iter()
            .position(|(bo, _)| *bo == self.state.cursor_col)
            .unwrap_or(graphemes.len());

        // Step 1: skip whitespace backward
        while g_idx > 0 {
            let g = graphemes[g_idx - 1].1;
            if !g.chars().all(|c| c.is_whitespace()) {
                break;
            }
            g_idx -= 1;
        }

        // Step 2: skip the previous word backward
        while g_idx > 0 {
            let g = graphemes[g_idx - 1].1;
            if g.chars().all(|c| c.is_whitespace()) {
                break;
            }
            g_idx -= 1;
        }

        if g_idx > 0 || self.state.cursor_line == 0 {
            // Stop at word start (or start of first line)
            self.set_cursor_col(graphemes.get(g_idx).map(|(bo, _)| *bo).unwrap_or(0));
        } else {
            // Reached start of line → wrap to end of previous line, keep searching for a word
            self.state.cursor_line -= 1;
            let prev_len = self.state.lines[self.state.cursor_line].len();
            self.set_cursor_col(prev_len);
            self.move_word_backward();
        }
    }

    /// Move cursor to logical line start/end (explicit set_cursor_col, resets sticky state).
    fn move_to_line_start(&mut self) {
        self.set_cursor_col(0);
    }
    fn move_to_line_end(&mut self) {
        let len = self.state.lines[self.state.cursor_line].len();
        self.set_cursor_col(len);
    }

    /// Move up/down by visual line (±1 visual line), crossing wrapped segments. Implements sticky column (simplified form of pi
    /// `computeVerticalMoveColumn` decision table):
    /// - Target too short (stops at line end) → lock preferred_visual_col, so the next move preserves the column;
    /// - Target long enough and lands back on preferred → clear preferred, re-derive from the new cursor column next time.
    /// No atomic-segment snapping/skipping (zen has no paste-marker; a wide segment is 2-column CJK, and the cursor always lands at the grapheme start).
    fn move_vertical(&mut self, delta: i32) {
        use unicode_segmentation::UnicodeSegmentation;
        use unicode_width::UnicodeWidthStr;

        if delta == 0 {
            return;
        }

        let width = self.last_width;
        if width == 0 {
            return;
        }

        let map = self.build_visual_line_map(width);
        if map.is_empty() {
            return;
        }

        // ── 1. Find current visual line ──
        let current_vis = self.find_current_visual_line(&map);

        // ── 2. Compute target visual line (clamped) ──
        let target_vis = (current_vis as i32 + delta).clamp(0, map.len() as i32 - 1) as usize;
        if target_vis == current_vis {
            return; // At top/bottom, no move
        }

        // ── 3. Determine target column (sticky column decision) ──
        let target_vl = &map[target_vis];
        let target_line = &self.state.lines[target_vl.logical_line];
        // Byte range of the target visual line (start_col and length in the same unit)
        let vl_start = target_vl.start_col;
        let vl_end = vl_start + target_vl.length;

        // Graphemes in the target visual line (byte offset, byte length, display width)
        let target_graphemes: Vec<(usize, usize, usize)> = target_line
            .grapheme_indices(true)
            .skip_while(|(bo, _)| *bo < vl_start)
            .take_while(|(bo, _)| *bo < vl_end)
            .map(|(bo, g)| (bo, g.len(), UnicodeWidthStr::width(g)))
            .collect();

        // sticky column: prefer the already-locked preferred, otherwise take the current visual column (aligned with pi column P)
        let preferred = self
            .preferred_visual_col
            .unwrap_or_else(|| self.cursor_visual_col());

        // ── 4. Find the position closest to preferred in the target line ──
        let mut visual_col = 0usize;
        let mut best_bo = vl_start;

        for &(bo, _len, g_width) in &target_graphemes {
            if visual_col >= preferred {
                break;
            }
            best_bo = bo;
            visual_col += g_width;
        }

        // preferred exceeds the target line: stop at line end
        let final_bo = if visual_col < preferred {
            vl_end
        } else {
            best_bo
        };

        // ── 5. Update state ──
        self.state.cursor_line = target_vl.logical_line;
        self.state.cursor_col = final_bo;

        // Sticky column lifecycle (aligned with pi decision table):
        // - Target was long enough, snapped back to preferred → fulfilled, clear (re-derive from new cursor column next time)
        // - Target too short (stopped at line end) → lock preferred, to keep the column on the next move
        if visual_col < preferred {
            self.preferred_visual_col = Some(preferred);
        } else {
            self.preferred_visual_col = None;
        }
        self.snapped_from_cursor_col = None;
    }

    /// Page scroll: move vertically by page height (max(5, 30% of viewport height)) (aligned with pi pageScroll);
    /// page height taken from `last_height` (viewport height from the most recent render).
    fn page_scroll(&mut self, down: bool) {
        let viewport_height = self.last_height;
        let page_size = std::cmp::max(5, viewport_height * 30 / 100) as i32;
        let delta = if down { page_size } else { -page_size };
        self.move_vertical(delta);
    }

    // Lay out text into visual rows (including line wrapping and caret attribution, aligned with pi layoutText).
    fn layout_text(&self, width: usize) -> Vec<LayoutLine> {
        let mut result = Vec::new();

        for (line_idx, line) in self.state.lines.iter().enumerate() {
            // The wrap width is supplied by the caller (render already computes layout_width from effective padding); use it directly here without subtracting padding again
            let chunks = word_wrap_line(line, width);
            // word_wrap_line always returns ≥1 chunk (empty lines yield a single empty chunk); no empty-line fallback needed

            for (i, chunk) in chunks.iter().enumerate() {
                // Caret attribution: non-final chunk [start, end), final chunk [start, end]
                // (Aligned with pi layoutText: a cursor at a wrap-point boundary belongs to the next fragment; a cursor at end-of-line belongs to the last fragment)
                let is_last = i + 1 == chunks.len();
                let has_cursor = line_idx == self.state.cursor_line
                    && self.state.cursor_col >= chunk.start_index
                    && (self.state.cursor_col < chunk.end_index
                        || (is_last && self.state.cursor_col == chunk.end_index));

                let cursor_pos = if has_cursor {
                    // Cursor display position within this visual line (grapheme widths accumulated)
                    use unicode_segmentation::UnicodeSegmentation;
                    use unicode_width::UnicodeWidthStr;

                    let offset_in_chunk = self.state.cursor_col - chunk.start_index;
                    let pos = chunk.text[..offset_in_chunk]
                        .graphemes(true)
                        .map(|g| UnicodeWidthStr::width(g))
                        .sum();
                    Some(pos)
                } else {
                    None
                };

                result.push(LayoutLine {
                    text: chunk.text.clone(),
                    has_cursor,
                    cursor_pos,
                });
            }
        }

        result
    }

    /// Build logical-line → visual-line map (aligned with pi buildVisualLineMap), used by move_vertical/page_scroll.
    /// length is the byte-range length (end_index - start_index), same unit as start_col/cursor_col.
    fn build_visual_line_map(&self, width: usize) -> Vec<VisualLine> {
        let mut map = Vec::new();

        for (line_idx, line) in self.state.lines.iter().enumerate() {
            // Wrap width is provided by the caller (move_vertical passes last_width = layout_width); use it directly, no padding subtracted
            let chunks = word_wrap_line(line, width);

            for chunk in &chunks {
                map.push(VisualLine {
                    logical_line: line_idx,
                    start_col: chunk.start_index,
                    length: chunk.end_index - chunk.start_index,
                });
            }
        }

        map
    }

    /// Find the visual line index containing the cursor (aligned with pi findCurrentVisualLine).
    fn find_current_visual_line(&self, map: &[VisualLine]) -> usize {
        let cursor_line = self.state.cursor_line;
        let cursor_col = self.state.cursor_col;

        for (vis_idx, vl) in map.iter().enumerate() {
            if vl.logical_line != cursor_line {
                continue;
            }
            // Cursor falls within this visual line's [start_col, start_col + length] range
            // The end position (start_col + length) belongs to this line (cursor at line end)
            if cursor_col >= vl.start_col && cursor_col <= vl.start_col + vl.length {
                return vis_idx;
            }
        }

        // Fallback: return the last visual line of this logical line
        map.iter()
            .rposition(|vl| vl.logical_line == cursor_line)
            .unwrap_or(0)
    }

    /// Set cursor column and reset preferred/snapped state (mandatory for all horizontal moves/edits).
    fn set_cursor_col(&mut self, col: usize) {
        self.state.cursor_col = col;
        self.preferred_visual_col = None;
        self.snapped_from_cursor_col = None;
    }
}

// ── Module-level pure functions ──

/// Normalize inserted text: \r\n and \r → \n, \t → 4 spaces (aligned with pi `normalizeText`).
/// Called by `insert_text`, the sole insertion entry point; callers need not normalize themselves.
fn normalize_text(text: &str) -> String {
    text.replace("\r\n", "\n")
        .replace('\r', "\n")
        .replace('\t', "    ")
}

/// Wrap a single line at word boundaries: within maxWidth prefer word boundaries, adjacent CJK may break, over-wide atomic segments become their own chunks.
/// **A fragment is a contiguous slice of the source text** (text == line[start_index..end_index], byte offset); whitespace belongs to the end of the previous fragment,
/// Never rebuild text — ensure displayed text matches the buffer character-for-character and cursor column mapping is linear
/// Consumers: only this module's layout_text/build_visual_line_map; no external consumers, visibility kept module-private.
/// Promote to pub later if needed (T-TU-02-1 markdown wrapping).
fn word_wrap_line(line: &str, max_width: usize) -> Vec<TextChunk> {
    // Degenerate input: return a single empty slice,
    // ensuring the caller's "every logical line >= 1 visual line" invariant holds.
    if line.is_empty() || max_width == 0 {
        return vec![TextChunk {
            text: String::new(),
            start_index: 0,
            end_index: 0,
        }];
    }

    // Whole row fits: return a single slice (aligns with pi fast path).
    if crate::screen::visible_width(line) <= max_width {
        return vec![TextChunk {
            text: line.to_string(),
            start_index: 0,
            end_index: line.len(),
        }];
    }

    // Grapheme sequence (byte offset, grapheme text); the whole pipeline only "slices", never "joins".
    let segments: Vec<(usize, &str)> = line.grapheme_indices(true).collect();

    let mut chunks: Vec<TextChunk> = Vec::new();
    let mut current_width = 0usize; // Cumulative display width of the current slice
    let mut chunk_start = 0usize; // Starting byte offset of the current slice in the original text

    // Wrap opportunity: a position where a line break is allowed (after the last run of whitespace / after a CJK adjacency boundary), recorded by the previous iteration at the end of "advance".
    // opp_index: byte offset of the opportunity; opp_width: accumulated width at the opportunity (used for backtracking).
    let mut opp_index: Option<usize> = None;
    let mut opp_width = 0usize;

    for (i, (char_index, grapheme)) in segments.iter().enumerate() {
        let g_width = crate::screen::visible_width(grapheme);
        let is_ws = grapheme.chars().all(|c| c.is_whitespace());

        // ── Bounds check (before advancing) ──
        if current_width + g_width > max_width {
            // Prefer backtracking to the previous wrap opportunity (only viable if the remainder + current grapheme fits)
            match opp_index {
                Some(opp) if current_width - opp_width + g_width <= max_width => {
                    chunks.push(make_chunk(line, chunk_start, opp));
                    chunk_start = opp;
                    current_width -= opp_width;
                }
                // No available opportunity, or backtracking still doesn't fit (e.g. wide character): force a break at the current position.
                // Skip when chunk_start == char_index (avoids empty slices).
                _ if chunk_start < *char_index => {
                    chunks.push(make_chunk(line, chunk_start, *char_index));
                    chunk_start = *char_index;
                    current_width = 0;
                }
                _ => {}
            }
            opp_index = None;
        }

        // ── A single atomic segment wider than maxWidth (extremely narrow terminal: max_width=1 and grapheme column width=2) ──
        // pi re-slices paste-marker atomic segments recursively by grapheme; zen has no markers, so the sole over-wide atomic segment is already
        // a single grapheme and cannot be subdivided further (recursion wouldn't converge) — make it an independent slice (1-column visual overflow, acceptable).
        if g_width > max_width {
            chunks.push(make_chunk(line, *char_index, *char_index + grapheme.len()));
            chunk_start = *char_index + grapheme.len();
            current_width = 0;
            opp_index = None;
            continue;
        }

        // ── Advance current grapheme ──
        current_width += g_width;

        // ── Record wrap opportunity ──
        // (1) whitespace followed by non-whitespace (consecutive spaces merge: break point goes after the following space)
        // (2) either adjacent side is CJK (CJK allows a break at any adjacency boundary)
        if let Some((next_index, next_grapheme)) = segments.get(i + 1) {
            let next_is_ws = next_grapheme.chars().all(|c| c.is_whitespace());
            if is_ws && !next_is_ws {
                opp_index = Some(*next_index);
                opp_width = current_width;
            } else if !is_ws && !next_is_ws {
                let cjk = is_cjk(grapheme.chars().next().unwrap_or('\0'));
                let next_cjk = is_cjk(next_grapheme.chars().next().unwrap_or('\0'));
                if cjk || next_cjk {
                    opp_index = Some(*next_index);
                    opp_width = current_width;
                }
            }
        }
    }

    // Push the remaining content as the final slice; chunk_start already at end-of-row means all content was fully closed above (over-wide final grapheme becomes its own slice),
    // so no empty slice is produced (mirrors the chunk_start < char_index guard in the forced-break branch).
    if chunk_start < line.len() {
        chunks.push(make_chunk(line, chunk_start, line.len()));
    }

    chunks
}

/// Generate chunk slices for the original text's byte range [start,end).
fn make_chunk(line: &str, start: usize, end: usize) -> TextChunk {
    TextChunk {
        text: line[start..end].to_string(),
        start_index: start,
        end_index: end,
    }
}

/// Whether the char is a CJK line-break character (aligns with pi's `cjkBreakRegex`: Han / Hiragana / Katakana / Hangul / Bopomofo).
/// Grapheme clusters are identified by their first char (the base of a combined cluster); blocks approximate Unicode Script_Extensions,
/// avoiding the unicode-script dependency; differences from pi are limited to narrow edge cases (e.g. pi doesn't break before full-width Latin letters).
fn is_cjk(c: char) -> bool {
    matches!(c,
        // Han (Unified Ideographs + Extensions A–H + Compatibility block)
        '\u{3400}'..='\u{4DBF}'
        | '\u{4E00}'..='\u{9FFF}'
        | '\u{F900}'..='\u{FAFF}'
        | '\u{20000}'..='\u{2A6DF}'
        | '\u{2A700}'..='\u{2B73F}'
        | '\u{2B740}'..='\u{2B81F}'
        | '\u{2B820}'..='\u{2CEAF}'
        | '\u{2CEB0}'..='\u{2EBEF}'
        | '\u{30000}'..='\u{3134F}'
        | '\u{2F800}'..='\u{2FA1F}'
        // CJK Symbols and Punctuation (mostly Han script, including 、。〈〉 etc.)
        | '\u{3000}'..='\u{303F}'
        // Hiragana / Katakana
        | '\u{3040}'..='\u{309F}'
        | '\u{30A0}'..='\u{30FF}'
        | '\u{31F0}'..='\u{31FF}'
        // Hangul (Jamo + Syllables)
        | '\u{1100}'..='\u{11FF}'
        | '\u{A960}'..='\u{A97F}'
        | '\u{AC00}'..='\u{D7AF}'
        | '\u{D7B0}'..='\u{D7FF}'
        // Bopomofo
        | '\u{3100}'..='\u{312F}'
        | '\u{31A0}'..='\u{31BF}'
    )
}
