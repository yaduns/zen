use crate::keybindings::KeybindingsManager;
use crate::keys::KeyId;
use crate::screen::{ScreenBuffer, Style};
use crossterm::event::KeyEvent;

/// Input semantics contract for selection-type components
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SelectAction {
    MoveUp,   // ↑: move selection up, wrap from top to last item
    MoveDown, // ↓: move selection down, wrap from bottom to first item
    Confirm, // Enter: activate selected item (SelectList fires on_select / SettingsList cycles value or opens submenu)
    Cancel,  // Esc/Ctrl+C: cancel (on_cancel)
}

/// Filterable single-select list; holds all items plus a filtered double buffer, selected index, scroll window (selection centered, start..end sliding),
/// Two-column layout (description column shown when width > 40); selected prefix hardcoded "→ "/" "
pub struct SelectList {
    items: Vec<SelectItem>, // All items; filtering and selection are based on it
    filtered_items: Vec<SelectItem>, // Filtered display view; set_filter recomputes and resets selection to 0
    selected_index: usize, // Selected index, always < filtered_items.len() (stays 0 for empty list)
    max_visible: usize,    // Max rows of the scroll window (passed at construction)
    theme: SelectListTheme, // Resolved styles
    layout: SelectListLayoutOptions, // Column width constraints + main column truncation callback
    on_select: Option<Box<dyn FnMut(&SelectItem)>>, // Fired on Confirm (not fired when nothing is selected)
    on_cancel: Option<Box<dyn FnMut()>>,            // Fired on Cancel
    on_selection_change: Option<Box<dyn FnMut(&SelectItem)>>, // Fired on navigation (↑↓)
    // SelectAction-domain keybinding registry (T-AP-06-2 constructor injection): authoritative source for handle_key action routing
    keybindings: KeybindingsManager<SelectAction>,
}

/// Filterable list item data
#[derive(Clone)]
pub struct SelectItem {
    value: String, // Filter match key (prefix, case-insensitive) and identifier for the selection callback
    label: String, // Display name (get_display_value prefers label, falls back to value when empty)
    description: Option<String>, // Description column shown in wide rows (line breaks/whitespace normalized to a single line)
}

impl SelectItem {
    /// Constructs a list item (called when building items for T-AP-11-2's thinking-selector / T-AP-08-6 config selection).
    /// When `label` is empty, display falls back to `value`; `description` is the description column shown on wide lines.
    pub fn new(
        value: impl Into<String>,
        label: impl Into<String>,
        description: Option<String>,
    ) -> Self {
        SelectItem {
            value: value.into(),
            label: label.into(),
            description,
        }
    }

    /// Match key/callback identifier (read access: get the selected item's identity in the on_select callback).
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Display name (falls back to value when empty, consistent with the render path).
    pub fn label(&self) -> &str {
        if self.label.is_empty() {
            &self.value
        } else {
            &self.label
        }
    }

    /// Description column text.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

/// Resolved styles for each table region (
#[derive(Clone)]
pub struct SelectListTheme {
    selected_style: Style,    // Full-row style for the selected row
    description_style: Style, // Description column style
    scroll_info_style: Style, // Scroll info
    no_match_style: Style,    // No-match hint row style
}

impl SelectListTheme {
    /// Constructs the selection list theme.
    pub fn new(
        selected_style: Style,
        description_style: Style,
        scroll_info_style: Style,
        no_match_style: Style,
    ) -> Self {
        SelectListTheme {
            selected_style,
            description_style,
            scroll_info_style,
            no_match_style,
        }
    }
}

/// Primary column layout constraints and custom truncation callback
pub struct SelectListLayoutOptions {
    pub min_primary_column_width: Option<usize>, // Lower bound of primary column width (default 32; if both are set, min/max are swapped and clamped)
    pub max_primary_column_width: Option<usize>, // Upper bound of primary column width
    pub truncate_primary: Option<Box<dyn FnMut(&SelectListTruncatePrimaryContext) -> String>>, // Custom primary column truncation (default wrap::truncate_to_width)
}

impl Default for SelectListLayoutOptions {
    fn default() -> Self {
        SelectListLayoutOptions {
            min_primary_column_width: None,
            max_primary_column_width: None,
            truncate_primary: None,
        }
    }
}

/// Primary column truncation callback args (pi `SelectListTruncatePrimaryContext`).
pub struct SelectListTruncatePrimaryContext<'a> {
    pub text: &'a str,
    pub max_width: usize,
    pub column_width: usize,
    pub item: &'a SelectItem,
    pub is_selected: bool,
}

impl SelectList {
    /// Construct. Parameter `keybindings`: SelectAction-domain keybinding registry (T-AP-06-2 constructor injection)—the default table uses
    /// `select_default_bindings()` for construction (user overrides can be layered on); `KeybindingsManager` is Clone.
    /// Consumers: T-AP-11-2 (settings-selector), T-AP-08-6 (config/model selection).
    pub fn new(
        items: Vec<SelectItem>,
        max_visible: usize,
        theme: SelectListTheme,
        layout: SelectListLayoutOptions,
        keybindings: KeybindingsManager<SelectAction>,
    ) -> Self {
        let filtered_items = items.clone();
        SelectList {
            items,
            filtered_items,
            selected_index: 0,
            max_visible,
            theme,
            layout,
            on_select: None,
            on_cancel: None,
            on_selection_change: None,
            keybindings,
        }
    }

    /// Replace all items and reset filtering and selection (no direct pi equivalent; used when settings-selector rebuilds).
    pub fn set_items(&mut self, items: Vec<SelectItem>) {
        self.items = items.clone();
        self.filtered_items = items;
        self.selected_index = 0;
    }

    /// Set filter by value prefix (case-insensitive), reset selection to 0.
    pub fn set_filter(&mut self, filter: &str) {
        let filtered_items: Vec<SelectItem> = self
            .items
            .iter()
            .filter(|item| {
                item.value.len() >= filter.len()
                    && item
                        .value
                        .get(..filter.len())
                        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(filter))
            })
            .cloned()
            .collect();
        self.filtered_items = filtered_items;
        self.selected_index = 0;
    }

    /// Clamp the selected index when setting (out-of-range values clamped into range).
    pub fn set_selected_index(&mut self, index: usize) {
        self.selected_index = if self.filtered_items.is_empty() {
            0
        } else {
            index.min(self.filtered_items.len() - 1)
        };
    }

    /// Current selected item (after filtering).
    pub fn get_selected_item(&self) -> Option<&SelectItem> {
        self.filtered_items.get(self.selected_index)
    }

    /// Register selection callback (fired on Enter; counterpart of onSelect, closure type FnMut supports multiple calls).
    pub fn set_on_select(&mut self, f: impl FnMut(&SelectItem) + 'static) {
        self.on_select = Some(Box::new(f))
    }

    /// Register cancel callback (Esc/Ctrl+C; counterpart of onCancel).
    pub fn set_on_cancel(&mut self, f: impl FnMut() + 'static) {
        self.on_cancel = Some(Box::new(f))
    }

    /// Register selection-change callback (fired on ↑↓ navigation; counterpart of onSelectionChange).
    pub fn set_on_selection_change(&mut self, f: impl FnMut(&SelectItem) + 'static) {
        self.on_selection_change = Some(Box::new(f))
    }

    /// Consume one selection action. Returns whether it was "consumed by this component":
    /// - MoveUp/MoveDown: consumed when there are filtered items (even if the selection callback is absent; only not consumed when the list is empty);
    /// - Confirm: consumed whenever there is a selected item (regardless of whether on_select is wired; not consumed when there is no selected item);
    /// - Cancel: always consumed (regardless of whether on_cancel is wired).
    /// Note: consumed ≠ callback executed — when the callback is absent the action is still considered taken over by the list; the caller must ensure correct wiring.
    pub fn handle_action(&mut self, action: SelectAction) -> bool {
        let len = self.filtered_items.len();

        match action {
            SelectAction::MoveUp => {
                if len == 0 {
                    return false;
                }
                self.selected_index = if self.selected_index == 0 {
                    len - 1
                } else {
                    self.selected_index - 1
                };
                // The more idiomatic form is: if let Some(cb) = &mut self.on_selection_change {
                if let Some(ref mut cb) = self.on_selection_change {
                    cb(&self.filtered_items[self.selected_index]);
                }
                true
            }
            SelectAction::MoveDown => {
                if len == 0 {
                    return false;
                }
                self.selected_index = if self.selected_index >= len - 1 {
                    0
                } else {
                    self.selected_index + 1
                };
                if let Some(ref mut cb) = self.on_selection_change {
                    cb(&self.filtered_items[self.selected_index]);
                }
                true
            }
            SelectAction::Confirm => {
                if let Some(item) = self.filtered_items.get(self.selected_index) {
                    if let Some(ref mut cb) = self.on_select {
                        cb(item);
                    }
                    true
                } else {
                    false
                }
            }
            SelectAction::Cancel => {
                if let Some(ref mut cb) = self.on_cancel {
                    cb();
                }
                true
            }
        }
    }
}

impl super::Component for SelectList {
    /// Empty after filtering → no_match-styled message `  No matching commands`;
    /// otherwise render the visible window (selection centered, `start..end` sliding), the description column (two columns when width > 40),
    /// and append an `(n/total)` line when scrolling is needed (scroll_info style, truncated to width). Returns the actual number of lines.
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        let width = width as usize;
        let height = height as usize;
        if height == 0 {
            // No usable line height: nothing to render, return 0 lines directly
            return 0;
        }

        // Empty list: no_match hint
        if self.filtered_items.is_empty() {
            buf.fill(x, y, width as u16, 1, &Style::default());
            buf.write(x, y, &self.theme.no_match_style, "  No matching commands");
            return 1;
        }

        let len = self.filtered_items.len();
        // max_vis: number of lines visible this time, clamped by the three of: configured viewport cap / list length / available height
        let max_vis = self.max_visible.min(len).min(height);
        // needs_scroll: whether to append the (n/total) scroll info at the end (only when the item count exceeds the configured viewport cap)
        let needs_scroll = len > self.max_visible;

        // Visible window [start, end): keep the selection as centered as possible
        // The three-way decision is equivalent to pi's clamp(selected_index - half, 0, len - max_vis):
        //   (1) selected is far enough toward the front → window snaps to the first item (start = 0)
        //   (2) selected is near the end → window bottom sticks to the end (start = len - max_vis)
        //   (3) otherwise → selection centered (start = selected_index - half)
        // end is clamped to len with min to guarantee no out-of-bounds
        let half = max_vis / 2;
        let start = if self.selected_index < half {
            0
        } else if self.selected_index + half >= len {
            len.saturating_sub(max_vis)
        } else {
            self.selected_index - half
        };
        let end = (start + max_vis).min(len);

        // Two-column layout: the description column is shown only when width > 40 (matching pi's width > 40 threshold)
        let show_desc = width > 40;
        let primary_width = if show_desc {
            let min_w = self.layout.min_primary_column_width.unwrap_or(32);
            let max_w = self.layout.max_primary_column_width.unwrap_or(width / 2);
            // The main column width is clamped between the main-column lower bound min_w and the upper bound width/2, then finally clamped by min(width) as a fallback,
            // Prevent callers from setting the lower bound too high so the primary column width doesn't exceed the full row width
            max_w.min(width / 2).max(min_w).min(width)
        } else {
            // In non-dual-column layouts the primary column spans the full width
            width
        };
        // desc_width uses saturating_sub instead of a bare subtraction to avoid a usize underflow panic caused by upstream layout config
        let desc_width = if show_desc {
            width.saturating_sub(primary_width)
        } else {
            0
        };

        // row: the current row being written (incrementing from y); painted: rows actually written this frame (return value, for container stacking layout)
        let mut row = y;
        let mut painted = 0u16;

        // Render each item
        for i in start..end {
            // Safety guard: if the row index exceeds the available height, terminate this frame (the visual window is guaranteed by max_vis; this is a fallback)
            if row >= y + height as u16 {
                break;
            }

            let item = &self.filtered_items[i];
            let is_selected = i == self.selected_index;

            // On each redraw, first clear the whole row to the default style, overwriting leftover content from the previous frame
            buf.fill(x, row, width as u16, 1, &Style::default());

            // Selected row prefix is hard-coded "→ " with the selected style applied; non-selected rows get two leading spaces and no style (aligned with pi)
            let (prefix, style) = if is_selected {
                ("→ ", &self.theme.selected_style)
            } else {
                ("  ", &Style::default())
            };
            buf.write(x, row, style, prefix);

            let display = if item.label.is_empty() {
                &item.value
            } else {
                &item.label
            };
            // Primary column text truncation: if the caller injected a truncate_primary callback, truncate via that callback; otherwise
            // fall back to default truncation (wrap::truncate_to_width; max_width = primary_width - 2 to leave room for the prefix)
            // take/put back: the closure is FnMut and borrows self.layout as &mut, so it must be temporarily moved out to break
            // the borrow conflict with item (&mut borrowed from self.filtered_items); after the call it must be put back unchanged, otherwise the injected callback will be swallowed
            let mut truncate_cb = self.layout.truncate_primary.take();
            let truncated = if let Some(ref mut cb) = truncate_cb {
                let ctx = SelectListTruncatePrimaryContext {
                    text: display,
                    max_width: primary_width.saturating_sub(2),
                    column_width: primary_width,
                    item,
                    is_selected,
                };
                cb(&ctx)
            } else {
                crate::wrap::truncate_to_width(display, primary_width.saturating_sub(2), "…")
            };
            self.layout.truncate_primary = truncate_cb;
            buf.write(x + 2, row, style, &truncated);

            if show_desc {
                if let Some(ref desc) = item.description {
                    // Description column: first strip newlines/CRs to normalize to a single line (aligned with pi normalizeToSingleLine),
                    // then truncate to desc_width, to avoid multi-line strings being written into a single-row grid and breaking the layout
                    let desc_clean = desc.replace(['\r', '\n'], "");
                    let desc_truncated =
                        crate::wrap::truncate_to_width(&desc_clean, desc_width, "…");
                    // The description column is always placed to the right of the primary column (x + primary_width), left-aligned alongside it
                    buf.write(
                        x + primary_width as u16,
                        row,
                        &self.theme.description_style,
                        &desc_truncated,
                    );
                }
            }

            row += 1;
            painted += 1;
        }

        // Scroll info (n/total): append a row (scroll_info style) only when scrolling is needed and there are remaining rows at the end,
        // The row number uses selected_index + 1 (1-based, consistent with pi); truncated to the full row width
        if needs_scroll && row < y + height as u16 {
            let info = format!("({}/{})", self.selected_index + 1, len);
            let truncated = crate::wrap::truncate_to_width(&info, width, "…");
            buf.fill(x, row, width as u16, 1, &Style::default());
            buf.write(x, row, &self.theme.scroll_info_style, &truncated);
            painted += 1;
        }

        painted
    }

    /// Registry-driven keybindings (default ↑↓/Enter/Esc/Ctrl+C; user overrides take effect via the injected table—T-AP-06-2):
    /// After events are normalized to `KeyId`, on a hit in MoveUp/MoveDown/Confirm/Cancel order, dispatch to `handle_action`;
    /// If no action matches, do not consume; let the upper layer continue routing.
    fn handle_key(&mut self, key: &KeyEvent) -> bool {
        let Some(kid) = KeyId::from_key_event(*key) else {
            return false;
        };
        for action in [
            SelectAction::MoveUp,
            SelectAction::MoveDown,
            SelectAction::Confirm,
            SelectAction::Cancel,
        ] {
            if self.keybindings.matches(&kid, action) {
                return self.handle_action(action);
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {

    /// Test constructor: default SelectAction keymap (no user overrides).
    fn kb() -> KeybindingsManager<SelectAction> {
        KeybindingsManager::new(
            crate::keybindings::select_default_bindings(),
            HashMap::new(),
        )
    }
    use super::*;
    use crate::components::Component;
    use crate::screen::{ScreenBuffer, Style};
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    fn item(value: &str) -> SelectItem {
        SelectItem {
            value: value.to_string(),
            label: value.to_string(),
            description: None,
        }
    }

    fn items(vs: &[&str]) -> Vec<SelectItem> {
        vs.iter().map(|v| item(v)).collect()
    }

    fn theme() -> SelectListTheme {
        SelectListTheme {
            selected_style: Style::default(),
            description_style: Style::default(),
            scroll_info_style: Style::default(),
            no_match_style: Style::default(),
        }
    }

    fn layout() -> SelectListLayoutOptions {
        SelectListLayoutOptions {
            min_primary_column_width: None,
            max_primary_column_width: None,
            truncate_primary: None,
        }
    }

    #[test]
    fn new_mirrors_items_and_starts_at_first() {
        let mut l = SelectList::new(items(&["a", "b", "c"]), 3, theme(), layout(), kb());
        assert_eq!(l.items.len(), 3);
        assert_eq!(l.filtered_items.len(), 3);
        assert_eq!(l.selected_index, 0);
        assert_eq!(l.get_selected_item().map(|i| i.value.as_str()), Some("a"));
    }

    #[test]
    fn set_items_resets_selection_and_filtered() {
        let mut l = SelectList::new(items(&["a", "b", "c"]), 3, theme(), layout(), kb());
        l.set_selected_index(2);
        l.set_items(items(&["x", "y"]));
        assert_eq!(l.selected_index, 0);
        assert_eq!(l.filtered_items.len(), 2);
        assert_eq!(l.get_selected_item().map(|i| i.value.as_str()), Some("x"));
    }

    #[test]
    fn set_filter_prefix_case_insensitive_and_resets_selection() {
        let mut l = SelectList::new(
            items(&["Alpha", "beta", "gamma"]),
            3,
            theme(),
            layout(),
            kb(),
        );
        l.set_selected_index(2);
        l.set_filter("al");
        assert_eq!(l.filtered_items.len(), 1);
        assert_eq!(l.filtered_items[0].value, "Alpha");
        assert_eq!(l.selected_index, 0); // Reset selection on filter
        l.set_filter("B");
        assert_eq!(l.filtered_items[0].value, "beta");
        l.set_filter("");
        assert_eq!(l.filtered_items.len(), 3);
        l.set_filter("zz");
        assert!(l.filtered_items.is_empty());
    }

    #[test]
    fn set_filter_safe_across_multi_byte_boundary() {
        let mut l = SelectList::new(items(&["你好", "再见", "abc"]), 3, theme(), layout(), kb());
        // Valid prefix (full grapheme) hit
        l.set_filter("你");
        assert_eq!(l.filtered_items.len(), 1);
        assert_eq!(l.filtered_items[0].value, "你好");
        // Prefixes not at grapheme boundaries must not panic nor falsely match CJK items (but may match ASCII items)
        l.set_filter("a");
        assert_eq!(l.filtered_items.len(), 1);
        assert_eq!(l.filtered_items[0].value, "abc"); // CJK text: byte 0 is not a char boundary → excluded, only "abc" matches
        l.set_filter("中");
        assert!(l.filtered_items.is_empty());
    }

    #[test]
    fn set_selected_index_clamps_and_empty_keeps_zero() {
        let mut l = SelectList::new(items(&["a", "b", "c"]), 3, theme(), layout(), kb());
        l.set_selected_index(10);
        assert_eq!(l.selected_index, 2);
        l.set_selected_index(0);
        assert_eq!(l.selected_index, 0);
        let mut empty = SelectList::new(vec![], 3, theme(), layout(), kb());
        empty.set_selected_index(5);
        assert_eq!(empty.selected_index, 0);
    }

    #[test]
    fn get_selected_item_none_when_empty() {
        let l = SelectList::new(vec![], 3, theme(), layout(), kb());
        assert!(l.get_selected_item().is_none());
    }

    #[test]
    fn handle_action_move_wraps_around() {
        let mut l = SelectList::new(items(&["a", "b", "c"]), 3, theme(), layout(), kb());
        assert!(l.handle_action(SelectAction::MoveDown));
        assert_eq!(l.selected_index, 1);
        assert!(l.handle_action(SelectAction::MoveDown));
        assert_eq!(l.selected_index, 2);
        // Wrap around from bottom to first item
        assert!(l.handle_action(SelectAction::MoveDown));
        assert_eq!(l.selected_index, 0);
        // Wrap around from top to last item
        assert!(l.handle_action(SelectAction::MoveUp));
        assert_eq!(l.selected_index, 2);
    }

    #[test]
    fn handle_action_move_empty_not_consumed() {
        let mut l = SelectList::new(vec![], 3, theme(), layout(), kb());
        assert!(!l.handle_action(SelectAction::MoveUp));
        assert!(!l.handle_action(SelectAction::MoveDown));
        assert_eq!(l.selected_index, 0);
    }

    #[test]
    fn handle_action_confirm_fires_on_select_and_consumes() {
        let selected = Rc::new(RefCell::new(Vec::new()));
        let s = selected.clone();
        let mut l = SelectList::new(items(&["a", "b", "c"]), 3, theme(), layout(), kb());
        l.set_on_select(move |item| s.borrow_mut().push(item.value.clone()));
        l.set_selected_index(1);
        assert!(l.handle_action(SelectAction::Confirm));
        assert_eq!(*selected.borrow(), vec!["b".to_string()]);
    }

    #[test]
    fn handle_action_confirm_without_callback_still_consumes() {
        let mut l = SelectList::new(items(&["a"]), 3, theme(), layout(), kb());
        assert!(l.handle_action(SelectAction::Confirm)); // Has selection → consume (callback absent)
    }

    #[test]
    fn handle_action_confirm_empty_not_consumed() {
        let mut l = SelectList::new(vec![], 3, theme(), layout(), kb());
        assert!(!l.handle_action(SelectAction::Confirm));
    }

    #[test]
    fn handle_action_cancel_always_consumed() {
        let mut l = SelectList::new(items(&["a"]), 3, theme(), layout(), kb());
        assert!(l.handle_action(SelectAction::Cancel)); // Also consumed without a callback
        let cancelled = Rc::new(RefCell::new(0));
        let c = cancelled.clone();
        let mut l2 = SelectList::new(items(&["a"]), 3, theme(), layout(), kb());
        l2.set_on_cancel(move || *c.borrow_mut() += 1);
        assert!(l2.handle_action(SelectAction::Cancel));
        assert_eq!(*cancelled.borrow(), 1);
    }

    #[test]
    fn handle_action_navigation_fires_on_selection_change() {
        let seen = Rc::new(RefCell::new(Vec::new()));
        let s = seen.clone();
        let mut l = SelectList::new(items(&["a", "b", "c"]), 3, theme(), layout(), kb());
        l.set_on_selection_change(move |item| s.borrow_mut().push(item.value.clone()));
        assert!(l.handle_action(SelectAction::MoveDown));
        assert!(l.handle_action(SelectAction::MoveDown));
        assert_eq!(*seen.borrow(), vec!["b".to_string(), "c".to_string()]);
    }

    #[test]
    fn handle_key_maps_to_actions() {
        let selected = Rc::new(RefCell::new(Vec::new()));
        let cancelled = Rc::new(RefCell::new(0));
        let s = selected.clone();
        let c = cancelled.clone();
        let mut l = SelectList::new(items(&["a", "b", "c"]), 3, theme(), layout(), kb());
        l.set_on_select(move |item| s.borrow_mut().push(item.value.clone()));
        l.set_on_cancel(move || *c.borrow_mut() += 1);

        assert!(l.handle_key(&KeyEvent::from(KeyCode::Down)));
        assert_eq!(l.selected_index, 1);
        assert!(l.handle_key(&KeyEvent::from(KeyCode::Up)));
        assert_eq!(l.selected_index, 0);
        assert!(l.handle_key(&KeyEvent::from(KeyCode::Enter)));
        assert_eq!(*selected.borrow(), vec!["a".to_string()]);
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(l.handle_key(&ctrl_c));
        assert_eq!(*cancelled.borrow(), 1);
        assert!(l.handle_key(&KeyEvent::from(KeyCode::Esc)));
        assert_eq!(*cancelled.borrow(), 2);
        // Other keys are not consumed
        assert!(!l.handle_key(&KeyEvent::from(KeyCode::Char('x'))));
    }

    #[test]
    fn render_height_zero_returns_zero() {
        let mut buf = ScreenBuffer::new(40, 10);
        let mut l = SelectList::new(items(&["a", "b"]), 3, theme(), layout(), kb());
        assert_eq!(l.render(&mut buf, 0, 0, 40, 0), 0);
    }

    #[test]
    fn render_empty_shows_no_match_row() {
        let mut buf = ScreenBuffer::new(40, 10);
        let mut l = SelectList::new(vec![], 3, theme(), layout(), kb());
        assert_eq!(l.render(&mut buf, 0, 0, 40, 10), 1);
    }

    #[test]
    fn render_counts_window_and_scroll_info() {
        let mut buf = ScreenBuffer::new(40, 10);
        // 5 items, max_visible=3 → window of 3 rows + 1 scroll-info row = 4
        let mut l = SelectList::new(
            items(&["a", "b", "c", "d", "e"]),
            3,
            theme(),
            layout(),
            kb(),
        );
        assert_eq!(l.render(&mut buf, 0, 0, 40, 10), 4);
    }

    #[test]
    fn render_window_clamped_by_height() {
        let mut buf = ScreenBuffer::new(40, 10);
        // 5 items, max_visible=5, height=2 → window of 2 rows; item count ≤ max_visible → no scroll info
        let mut l = SelectList::new(
            items(&["a", "b", "c", "d", "e"]),
            5,
            theme(),
            layout(),
            kb(),
        );
        assert_eq!(l.render(&mut buf, 0, 0, 40, 2), 2);
    }

    #[test]
    fn render_scroll_info_omitted_when_no_room() {
        let mut buf = ScreenBuffer::new(40, 10);
        // 10 items, max_visible=5, height=3 → window fills 3 rows, no scroll-info row
        let mut l = SelectList::new(
            items(&["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]),
            5,
            theme(),
            layout(),
            kb(),
        );
        assert_eq!(l.render(&mut buf, 0, 0, 40, 3), 3);
    }

    #[test]
    fn render_uses_truncate_primary_and_keeps_callback_after_render() {
        // Verify take/restore: both render callbacks remain usable afterward (otherwise the second render falls back to default truncation and the call count halves);
        // Also check the context args (primary-column truncation callback contract).
        let ctxs = Rc::new(RefCell::new(Vec::new()));
        let c = ctxs.clone();
        let layout = SelectListLayoutOptions {
            min_primary_column_width: None,
            max_primary_column_width: None,
            truncate_primary: Some(Box::new(move |ctx: &SelectListTruncatePrimaryContext| {
                c.borrow_mut().push((
                    ctx.text.to_string(),
                    ctx.max_width,
                    ctx.column_width,
                    ctx.is_selected,
                ));
                "CUSTOM".to_string()
            })),
        };
        let mut l = SelectList::new(items(&["a", "b", "c"]), 3, theme(), layout, kb());
        let mut buf = ScreenBuffer::new(80, 10);
        assert_eq!(l.render(&mut buf, 0, 0, 80, 10), 3);
        assert_eq!(l.render(&mut buf, 0, 0, 80, 10), 3);
        let got = ctxs.borrow();
        assert_eq!(
            got.len(),
            6,
            "Two renders × 3 items → callbacks are reused (take/put-back works)"
        );
        // width=80 > 40 → two columns; primary = clamp(max(32, 40)) = 40; max_width = 40-2 = 38
        assert_eq!(got[0], ("a".to_string(), 38, 40, true)); // First item selected
        assert_eq!(got[1], ("b".to_string(), 38, 40, false));
    }

    #[test]
    fn render_narrow_width_no_desc_single_column() {
        let mut buf = ScreenBuffer::new(40, 10);
        let mut l = SelectList::new(items(&["a", "b", "c"]), 3, theme(), layout(), kb());
        // width=30 ≤ 40 → not two-column; primary column spans the full width, still renders 3 items
        assert_eq!(l.render(&mut buf, 0, 0, 30, 10), 3);
    }

    // ── User override routing (T-AP-06-2) ──

    #[test]
    fn user_override_reroutes_select_action() {
        use crate::keys::KeyId;
        let mut user = HashMap::new();
        user.insert(
            SelectAction::MoveDown,
            vec![KeyId::parse("ctrl+n").unwrap()],
        );
        let mut l = SelectList::new(
            vec![item("a")],
            5,
            theme(),
            layout(),
            KeybindingsManager::new(crate::keybindings::select_default_bindings(), user),
        );
        // Default down is disabled; ctrl+n matches MoveDown
        assert!(!l.handle_key(&KeyEvent::from(KeyCode::Down)));
        assert!(l.handle_key(&KeyEvent::new(KeyCode::Char('n'), KeyModifiers::CONTROL)));
    }
}
