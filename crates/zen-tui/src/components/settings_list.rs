// REQ-TUI-01 (pi-413): Settings row list (SettingsList)—scrolling window, value cycling, submenus, optional search
// See reference/pi-original/packages/tui/src/components/settings-list.ts
// Shares SelectAction with SelectList (selection-action contract); embeds Input for the search row (composition rather than inheritance).

use crate::screen::{ScreenBuffer, Style};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::input::Input;
use super::select_list::SelectAction;
use super::Component;

/// Submenu factory: takes the current value and produces a subcomponent (pi `SettingItem.submenu`).
/// Split the type alias to satisfy clippy `type_complexity`.
pub type SubmenuFactory = Box<dyn FnMut(&str) -> Box<dyn Component> + 'static>;

/// Search matcher: takes (item, query string) and returns whether it matches (an injection point for pi's default fuzzy matching).
pub type ItemFilter = Box<dyn FnMut(&SettingItem, &str) -> bool + 'static>;

/// Value-change callback: takes (id, new_value).
pub type ValueChange = Box<dyn FnMut(&str, &str)>;

/// Setting item (pi `SettingItem`; fields have identical names and semantics).
/// `submenu` is a **submenu factory**: takes the current value and produces a child component; closing the child component is explicitly driven by the caller via
/// `SettingsList::done_submenu` (pi's done callback injection is replaced with an explicit method,
/// See the SettingsList role description in "Data Model"—inversion of control, avoiding component self-handshaking).
pub struct SettingItem {
    pub id: String, // Unique identifier (update_value / on_change payload)
    pub label: String, // Display name in the left column
    pub description: Option<String>, // Wrapped and displayed below the list when selected
    pub current_value: String, // Value displayed in the right column; updated when activated
    /// If provided, Enter/Space cycles through values (aligned with pi `values`).
    pub values: Option<Vec<String>>,
    /// If provided, Enter/Space opens that submenu (aligned with pi `submenu`).
    pub submenu: Option<SubmenuFactory>,
}

/// Sets the list theme (pi `SettingsListTheme`; selected states for label/value are represented by two styles,
/// `cursor_marker` is the prefix string for the selected row, aligned with pi's hard-coded `"→ "`.
pub struct SettingsListTheme {
    pub label_style: Style,
    pub label_selected_style: Style,
    pub value_style: Style,
    pub value_selected_style: Style,
    pub description_style: Style,
    pub hint_style: Style,
    pub cursor_marker: String,
}

/// Constructs options (pi `SettingsListOptions`).
/// `filter` is the search matcher: use it when Ok; when None, default to lowercase label substring matching
/// (After fuzzy matching in T-TU-03-4 is implemented, callers inject a replacement; see "Open Questions").
pub struct SettingsListOptions {
    pub enable_search: bool,
    pub filter: Option<ItemFilter>,
}

/// Sets up the row list (pi `SettingsList`): scrolling window, value cycling, submenus, and optional search.
///
/// **Structural revision (relative to the task document)**: the display view uses `filtered: Vec<usize>` (indices into `items`)
/// Replaces the `filtered_items: Vec<SettingItem>` listed in the document. Reason: `SettingItem.submenu` is
/// `Box<dyn FnMut>`, which is **not Clone**, so items cannot be cloned to produce a filtered copy;
/// Using index references preserves the "filtered view" semantics, avoids the cloning requirement, and eliminates the need to synchronize two copies during update_value/activation.
/// When search is disabled, `filtered` always contains all indices (0..n); both branches access items through it.
pub struct SettingsList {
    items: Vec<SettingItem>, // Full settings set (sole owner, includes the submenu closure)
    filtered: Vec<usize>, // Display view: sequence of indices into items (after search filtering; all indices when search is disabled)
    selected_index: usize, // Selected index, always < filtered.len() (stays 0 for an empty list)
    max_visible: usize, // Viewport row limit
    theme: SettingsListTheme, // Style set
    on_change: ValueChange, // Value change callback (id, new_value)
    on_cancel: Box<dyn FnMut()>, // Cancel callback
    search_enabled: bool, // Search toggle (finalized from options at construction)
    search_input: Option<Input>, // Embedded search box (Some when enable_search is enabled)
    filter: Option<ItemFilter>, // Search matcher (passes through options.filter)
    submenu_component: Option<Box<dyn Component>>, // Active submenu (when Some, render/handle_key only route through it)
    submenu_item_index: Option<usize>, // Selected position before opening the submenu (restored on close and used to locate the updated item)
}

impl SettingsList {
    /// Construct; `on_change` fires when the value changes; `on_cancel` fires on Esc/Ctrl+C
    /// (matches pi's constructor parameters). Consumers: T-AP-11-2 (settings-selector), T-AP-08-6.
    pub fn new(
        items: Vec<SettingItem>,
        max_visible: usize,
        theme: SettingsListTheme,
        on_change: ValueChange,
        on_cancel: Box<dyn FnMut() + 'static>,
        options: SettingsListOptions,
    ) -> Self {
        let filtered = (0..items.len()).collect();
        let search_input = if options.enable_search {
            Some(Input::new())
        } else {
            None
        };
        SettingsList {
            items,
            filtered,
            selected_index: 0,
            max_visible,
            theme,
            on_change,
            on_cancel,
            search_enabled: options.enable_search,
            search_input,
            filter: options.filter,
            submenu_component: None,
            submenu_item_index: None,
        }
    }

    /// Updates the current_value of an item with the given id (equivalent to pi `updateValue`).
    /// Updates data only (items is the sole owner) without triggering on_change.
    pub fn update_value(&mut self, id: &str, new_value: &str) {
        if let Some(item) = self.items.iter_mut().find(|i| i.id == id) {
            item.current_value = new_value.to_string();
        }
    }

    /// Consumes the selection action (same semantics as SelectList::handle_action; also triggered by Confirm
    /// activate_item: cycles through values or calls open_submenu via the submenu factory).
    /// Returns whether the action was consumed (MoveUp/MoveDown treat an empty list as not consumed).
    pub fn handle_action(&mut self, action: SelectAction) -> bool {
        let len = self.filtered.len();
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
                true
            }
            SelectAction::Confirm => {
                self.activate_item();
                true
            }
            SelectAction::Cancel => {
                (self.on_cancel)();
                true
            }
        }
    }

    /// Opens the submenu: sets the submenu state and switches to the child component on the next render (called internally by the activation flow,
    /// Exposed for application-layer preset scenarios).
    pub fn open_submenu(&mut self, component: Box<dyn Component>) {
        // Record the selection before opening (the position in filtered) for restoration on close
        self.submenu_item_index = Some(self.selected_index);
        self.submenu_component = Some(component);
    }

    /// Close the submenu (wired by the caller from the child component's on_select/on_cancel):
    /// When `selected` is Some, update the corresponding item's current_value and trigger on_change(id, value);
    /// Restore the selection from before the submenu was opened.
    pub fn done_submenu(&mut self, selected: Option<String>) {
        let pos = self.submenu_item_index.take();
        self.submenu_component = None;
        // Restore the selection from before opening
        if let Some(p) = pos {
            self.selected_index = p;
        }
        // Update the value of the item whose submenu was opened and notify of the change
        if let Some(value) = selected {
            if let Some(p) = pos {
                let item_idx = self.filtered.get(p).copied();
                if let Some(item_idx) = item_idx {
                    if let Some(item) = self.items.get_mut(item_idx) {
                        item.current_value = value.clone();
                        let id = item.id.clone();
                        (self.on_change)(&id, &value);
                    }
                }
            }
        }
    }

    /// Delegate focus to the embedded search box (aligning focused propagation with pi session-selector);
    /// No-op when search is disabled.
    pub fn set_focused(&mut self, focused: bool) {
        if let Some(input) = self.search_input.as_mut() {
            input.set_focused(focused);
        }
    }

    /// Activate the current item: open the submenu if a submenu factory exists; otherwise, cycle to the next value if values exist
    /// (If the current value does not match, jump to the first item; align with pi `activateItem`.) For an empty list/no activation semantics, no-op.
    fn activate_item(&mut self) {
        let Some(&item_idx) = self.filtered.get(self.selected_index) else {
            return;
        };
        let Some(item) = self.items.get_mut(item_idx) else {
            return;
        };

        if let Some(factory) = item.submenu.as_mut() {
            // Separate borrows: item borrows the self.items field, independent of the self.submenu_* fields.
            let cv = item.current_value.clone();
            self.submenu_item_index = Some(self.selected_index);
            let comp = factory(&cv);
            self.submenu_component = Some(comp);
            return;
        }
        if let Some(values) = item.values.as_ref() {
            if !values.is_empty() {
                let cur = item.current_value.clone();
                // If the current value does not match, jump to the first item (align with pi: when indexOf is -1, next = values[0])
                let next_idx = match values.iter().position(|v| v == &cur) {
                    Some(i) => (i + 1) % values.len(),
                    None => 0,
                };
                let new_value = values[next_idx].clone();
                item.current_value = new_value.clone();
                let id = item.id.clone();
                (self.on_change)(&id, &new_value);
            }
        }
    }

    /// Recompute the filtered index sequence based on the search box contents and reset the selection to 0.
    /// The matcher uses the injected `self.filter` first; otherwise, it defaults to lowercase containment matching on label (align with pi's default).
    fn apply_filter(&mut self, query: &str) {
        let q = query.to_lowercase();
        // Separate borrows: match uses &mut self.filter, iteration uses &self.items; the fields are independent.
        let filtered = if let Some(f) = self.filter.as_mut() {
            self.items
                .iter()
                .enumerate()
                .filter(|(_, item)| f(item, query))
                .map(|(i, _)| i)
                .collect()
        } else {
            self.items
                .iter()
                .enumerate()
                .filter(|(_, item)| item.label.to_lowercase().contains(&q))
                .map(|(i, _)| i)
                .collect()
        };
        self.filtered = filtered;
        self.selected_index = 0;
    }

    /// Best-effort writes the bottom hint line at row (return unchanged if it does not fit) and returns the next line after writing.
    /// Content varies depending on whether search is enabled (align with pi `addHintLine`).
    fn write_hint_line(
        &self,
        buf: &mut ScreenBuffer,
        x: u16,
        width: u16,
        bottom: u16,
        row: u16,
    ) -> u16 {
        if row >= bottom {
            return row;
        }
        let hint = if self.search_enabled {
            "  Type to search · Enter/Space to change · Esc to cancel"
        } else {
            "  Enter/Space to change · Esc to cancel"
        };
        buf.fill(x, row, width, 1, &Style::default());
        buf.write(x, row, &self.theme.hint_style, hint);
        row + 1
    }
}

impl super::Component for SettingsList {
    /// When a submenu is active → render only the subcomponent; otherwise render the main list: search row (Input + blank line),
    /// Empty/no-match notice, scrolling window rows (pad label to min(30, maximum label width) + truncate value),
    /// Scrolling info line, selected-item description (wrapped), and hint line. Keep all content within the given rectangle, truncate on overflow, and return the actual line count.
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        // Submenu active: delegate the entire frame to the child component (pi-isomorphic)
        if let Some(comp) = self.submenu_component.as_mut() {
            return comp.render(buf, x, y, width, height);
        }

        let height = height as usize;
        if height == 0 {
            return 0;
        }
        let bottom = y + height as u16; // Exclusive lower bound
        let mut row = y;
        let mut painted = 0u16;

        // ── Search row (Input) ＋ Blank-line separator ──
        if self.search_enabled {
            if let Some(input) = self.search_input.as_mut() {
                if row < bottom {
                    let used = input.render(buf, x, row, width, bottom - row);
                    row += used;
                    painted += used;
                }
            }
            if row < bottom {
                buf.fill(x, row, width, 1, &Style::default());
                row += 1;
                painted += 1;
            }
        }

        // ── No settings available / Filter has no matches ──
        if self.items.is_empty() {
            if row < bottom {
                buf.fill(x, row, width, 1, &Style::default());
                buf.write(x, row, &self.theme.hint_style, "  No settings available");
                row += 1;
                painted += 1;
            }
            let nr = self.write_hint_line(buf, x, width, bottom, row);
            painted += nr - row;
            return painted;
        }
        if self.filtered.is_empty() {
            if row < bottom {
                buf.fill(x, row, width, 1, &Style::default());
                buf.write(x, row, &self.theme.hint_style, "  No matching settings");
                row += 1;
                painted += 1;
            }
            let nr = self.write_hint_line(buf, x, width, bottom, row);
            painted += nr - row;
            return painted;
        }

        // ── Scrolling window [start, end): center the selection when possible (same algorithm as SelectList) ──
        let len = self.filtered.len();
        let max_vis = self.max_visible.min(len);
        let half = max_vis / 2;
        let start = if self.selected_index < half {
            0
        } else if self.selected_index + half >= len {
            len.saturating_sub(max_vis)
        } else {
            self.selected_index - half
        };
        let end = (start + max_vis).min(len);

        // Label alignment width: min(30, maximum display width of all item labels) (aligned with pi)
        let max_label_w = self
            .items
            .iter()
            .map(|it| crate::screen::visible_width(&it.label))
            .max()
            .unwrap_or(0)
            .min(30);

        for pos in start..end {
            if row >= bottom {
                break;
            }
            let item = &self.items[self.filtered[pos]];
            let is_selected = pos == self.selected_index;

            // Reset the entire line to default, overwriting remnants from the previous frame
            buf.fill(x, row, width, 1, &Style::default());

            // Selected-row prefix (cursor_marker / two spaces)
            let marker = if is_selected {
                &self.theme.cursor_marker
            } else {
                "  "
            };
            let prefix_w = crate::screen::visible_width(marker);
            buf.write(x, row, &Style::default(), marker);

            // Left-column label: pad to max_label_w to align with the value column
            let label_w = crate::screen::visible_width(&item.label);
            let pad = max_label_w.saturating_sub(label_w);
            let padded = format!("{}{}", item.label, " ".repeat(pad));
            let label_style = if is_selected {
                &self.theme.label_selected_style
            } else {
                &self.theme.label_style
            };
            buf.write(x + prefix_w as u16, row, label_style, &padded);

            // Right-column current value: after label + 2 spaces; truncate if too wide
            let used = prefix_w + max_label_w + 2;
            let value_w = (width as usize).saturating_sub(used + 2);
            let value = crate::wrap::truncate_to_width(&item.current_value, value_w, "…");
            let value_style = if is_selected {
                &self.theme.value_selected_style
            } else {
                &self.theme.value_style
            };
            buf.write(x + used as u16, row, value_style, &value);

            row += 1;
            painted += 1;
        }

        // ── Scroll info (requires scrolling and has remaining lines at the end)──
        if len > self.max_visible && row < bottom {
            let info = format!("  ({}/{})", self.selected_index + 1, len);
            buf.fill(x, row, width, 1, &Style::default());
            buf.write(x, row, &self.theme.hint_style, &info);
            row += 1;
            painted += 1;
        }

        // ── Selected item description (wrapped below the list; separated by a blank line)──
        if let Some(&item_idx) = self.filtered.get(self.selected_index) {
            if let Some(item) = self.items.get(item_idx) {
                if let Some(desc) = &item.description {
                    if !desc.trim().is_empty() {
                        if row < bottom {
                            buf.fill(x, row, width, 1, &Style::default());
                            row += 1;
                            painted += 1;
                        }
                        let wrap_w = (width as usize).saturating_sub(4);
                        for chunk in crate::wrap::word_wrap_line(desc, wrap_w) {
                            if row >= bottom {
                                break;
                            }
                            let line = format!("  {}", chunk.text);
                            buf.write(x, row, &self.theme.description_style, &line);
                            row += 1;
                            painted += 1;
                        }
                    }
                }
            }
        }

        // ── Bottom hint line ──
        let nr = self.write_hint_line(buf, x, width, bottom, row);
        painted += nr - row;

        painted
    }

    /// Inline default keybindings: ↑↓/Enter(/Space)/Esc/Ctrl+C use the selection path; remaining keys when search is enabled
    /// Delegate to search_input (aligned with pi sanitize; press Space to enter the search box) and recalculate the filter;
    /// When a submenu is active, delegate all keys to the child component. Other keys are not consumed.
    fn handle_key(&mut self, key: &KeyEvent) -> bool {
        // Submenu active: delegate all keys (mirrors pi)
        if let Some(comp) = self.submenu_component.as_mut() {
            return comp.handle_key(key);
        }
        match (key.code, key.modifiers) {
            (KeyCode::Up, _) => self.handle_action(SelectAction::MoveUp),
            (KeyCode::Down, _) => self.handle_action(SelectAction::MoveDown),
            (KeyCode::Enter, _) | (KeyCode::Char(' '), _) => {
                self.handle_action(SelectAction::Confirm)
            }
            (KeyCode::Esc, _) | (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                self.handle_action(SelectAction::Cancel)
            }
            _ => {
                // Search not enabled: do not consume
                if !self.search_enabled {
                    return false;
                }
                // Delegate to the embedded search box (borrow to finish, then retrieve the term, avoiding a conflict with apply_filter's &mut self)
                let consumed = if let Some(input) = self.search_input.as_mut() {
                    input.handle_key(key)
                } else {
                    false
                };
                let query = self
                    .search_input
                    .as_ref()
                    .map(|i| i.get_value().to_string())
                    .unwrap_or_default();
                self.apply_filter(&query);
                consumed
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn item(id: &str, label: &str, values: Option<&[&str]>) -> SettingItem {
        SettingItem {
            id: id.to_string(),
            label: label.to_string(),
            description: None,
            current_value: values.map(|v| v[0].to_string()).unwrap_or_default(),
            values: values.map(|v| v.iter().map(|s| s.to_string()).collect()),
            submenu: None,
        }
    }

    fn theme() -> SettingsListTheme {
        SettingsListTheme {
            label_style: Style::default(),
            label_selected_style: Style::default(),
            value_style: Style::default(),
            value_selected_style: Style::default(),
            description_style: Style::default(),
            hint_style: Style::default(),
            cursor_marker: "→ ".to_string(),
        }
    }

    fn options() -> SettingsListOptions {
        SettingsListOptions {
            enable_search: false,
            filter: None,
        }
    }

    fn list_of(n: usize) -> SettingsList {
        let items: Vec<SettingItem> = (0..n)
            .map(|i| item(&format!("id{i}"), &format!("L{i}"), Some(&["v"])))
            .collect();
        SettingsList::new(
            items,
            n,
            theme(),
            Box::new(|_, _| {}),
            Box::new(|| {}),
            options(),
        )
    }

    #[test]
    fn navigation_wraps_and_empty_not_consumed() {
        let mut list = list_of(3);
        assert!(list.handle_action(SelectAction::MoveDown));
        assert_eq!(list.selected_index, 1);
        assert!(list.handle_action(SelectAction::MoveDown));
        assert_eq!(list.selected_index, 2);
        // Wrap around from bottom to first item
        assert!(list.handle_action(SelectAction::MoveDown));
        assert_eq!(list.selected_index, 0);
        // Wrap around from top to last item
        assert!(list.handle_action(SelectAction::MoveUp));
        assert_eq!(list.selected_index, 2);

        // Treat empty-list navigation as unconsumed
        let mut empty = SettingsList::new(
            vec![],
            5,
            theme(),
            Box::new(|_, _| {}),
            Box::new(|| {}),
            options(),
        );
        assert!(!empty.handle_action(SelectAction::MoveUp));
        assert!(!empty.handle_action(SelectAction::MoveDown));
    }

    #[test]
    fn confirm_cycles_values_and_fires_on_change() {
        let items = vec![item("a", "Alpha", Some(&["x", "y", "z"]))];
        let calls = Rc::new(RefCell::new(Vec::new()));
        let c = calls.clone();
        let mut list = SettingsList::new(
            items,
            5,
            theme(),
            Box::new(move |id, v| c.borrow_mut().push((id.to_string(), v.to_string()))),
            Box::new(|| {}),
            options(),
        );
        assert!(list.handle_action(SelectAction::Confirm));
        assert_eq!(list.items[0].current_value, "y");
        // Confirm again → cycle to z
        assert!(list.handle_action(SelectAction::Confirm));
        assert_eq!(list.items[0].current_value, "z");
        let got = calls.borrow();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0], ("a".to_string(), "y".to_string()));
        assert_eq!(got[1], ("a".to_string(), "z".to_string()));
    }

    #[test]
    fn cancel_fires_on_cancel() {
        let cancelled = Rc::new(RefCell::new(false));
        let c = cancelled.clone();
        let mut list = SettingsList::new(
            list_of(2).items,
            5,
            theme(),
            Box::new(|_, _| {}),
            Box::new(move || *c.borrow_mut() = true),
            options(),
        );
        assert!(list.handle_action(SelectAction::Cancel));
        assert!(*cancelled.borrow());
    }

    #[test]
    fn done_submenu_updates_value_and_restores_index() {
        let items = vec![item("a", "A", None), item("b", "B", None)];
        let calls = Rc::new(RefCell::new(Vec::new()));
        let c = calls.clone();
        let mut list = SettingsList::new(
            items,
            5,
            theme(),
            Box::new(move |id, v| c.borrow_mut().push((id.to_string(), v.to_string()))),
            Box::new(|| {}),
            options(),
        );
        // Select position 1, open the submenu, then close with selected
        list.selected_index = 1;
        list.open_submenu(Box::new(crate::components::spacer::Spacer::new(0)));
        assert!(list.submenu_component.is_some());
        list.done_submenu(Some("hello".to_string()));
        assert!(list.submenu_component.is_none());
        assert_eq!(list.items[1].current_value, "hello");
        assert_eq!(list.selected_index, 1);
        let got = calls.borrow();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0], ("b".to_string(), "hello".to_string()));
    }

    #[test]
    fn done_submenu_without_selection_restores_only() {
        let mut list = list_of(2);
        list.selected_index = 1;
        list.open_submenu(Box::new(crate::components::spacer::Spacer::new(0)));
        list.done_submenu(None);
        assert!(list.submenu_component.is_none());
        assert_eq!(list.selected_index, 1);
        // No change (list_of's initial current_value is "v")
        assert_eq!(list.items[1].current_value, "v");
    }

    #[test]
    fn update_value_updates_item() {
        let mut list = list_of(2);
        list.update_value("id0", "new");
        assert_eq!(list.items[0].current_value, "new");
        // Ignore a nonexistent id
        list.update_value("nope", "x");
        assert_eq!(list.items[1].current_value, "v");
    }

    #[test]
    fn apply_filter_narrows_filtered_indices() {
        let items = vec![
            item("1", "Alpha", None),
            item("2", "Beta", None),
            item("3", "gamma", None),
        ];
        let mut list = SettingsList::new(
            items,
            5,
            theme(),
            Box::new(|_, _| {}),
            Box::new(|| {}),
            options(),
        );
        list.apply_filter("al");
        // Case-insensitive: Alpha matches
        assert_eq!(list.filtered, vec![0]);
        list.apply_filter("");
        assert_eq!(list.filtered, vec![0, 1, 2]);
        // Reset the selection
        assert_eq!(list.selected_index, 0);
    }
}
