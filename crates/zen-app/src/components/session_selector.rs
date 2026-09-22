// REQ-APP-11 (pi-358): session-selector — session selection overlay (search/sort/name filter/rename/delete).
// Mirrors the original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/session-selector.ts.
// Implementation revision (vs the documented signature): the loader is a **synchronous callback** (the component has no async render path, see stub "Group D" decision),
// so the `request_render` parameter is resolved away (ready at construction/load-complete, no redraw hook needed); rename/delete waits are handled by upper-level callbacks.
// Keybindings: select keys (↑↓/Enter/Esc/Ctrl+C) inline by default (aligned with zen-tui SelectList conventions),
// session.* application actions are routed via `KeybindingsManager<AppAction>` (app_kb).

use crate::components::dynamic_border::DynamicBorder;
use crate::components::session_selector_search::{filter_and_sort_sessions, NameFilter, SortMode};
use crate::keybindings::AppAction;
use crate::session::session_listing::{SessionListProgress, SessionListing};
use chrono::Utc;
use zen_tui::components::input::Input;
use zen_tui::components::select_list::SelectAction;
use zen_tui::components::Component;
use zen_tui::keybindings::KeybindingsManager;
use zen_tui::keys::KeyId;
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{ColorToken, Theme};

/// Synchronous loader callback for session selection (current directory / all directories; with progress callback).
/// Reason for the synchronous form: the component has no async render path; if T-AP-08-5/T-AP-11-1 need async preloading, it completes before construction.
pub type SessionListLoader =
    Box<dyn FnMut(&mut dyn FnMut(SessionListProgress)) -> Vec<SessionListing>>;

/// Selection scope (pi SessionScope): current = sessions of the current cwd, all = sessions of all project directories.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SessionScope {
    Current,
    All,
}

/// Session list state in interactive mode (list / rename).
enum Mode {
    List,
    Rename,
}

/// Relative time display (pi formatSessionDate: now/5m/2h/3d/2w/3mo/1y).
fn format_session_date(modified: chrono::DateTime<Utc>) -> String {
    let now = Utc::now();
    let diff = now.signed_duration_since(modified);
    let diff_min = diff.num_minutes();
    let diff_hours = diff.num_hours();
    let diff_days = diff.num_days();
    if diff_min < 1 {
        return "now".to_string();
    }
    if diff_min < 60 {
        return format!("{diff_min}m");
    }
    if diff_hours < 24 {
        return format!("{diff_hours}h");
    }
    if diff_days < 7 {
        return format!("{diff_days}d");
    }
    if diff_days < 30 {
        return format!("{}w", diff_days / 7);
    }
    if diff_days < 365 {
        return format!("{}mo", diff_days / 30);
    }
    format!("{}y", diff_days / 365)
}

/// Home prefix folded to `~` (for path display).
fn shorten_path(path: &str, home: Option<&str>) -> String {
    match home {
        Some(h) if !h.is_empty() && path.starts_with(h) => path.replacen(h, "~", 1),
        _ => path.to_string(),
    }
}

/// Session selection overlay: search box + sort/filter toggles + list + rename (optional) and delete —
/// Shared by T-AP-08-5 (--session startup selection) and T-AP-11-1 (/sessions navigation).
///
/// Scope switch (current/all) triggers a reload; Esc cancels, Enter confirms and returns the path.
#[allow(clippy::type_complexity)] // Callback fields (Box<dyn FnMut> composition)
pub struct SessionSelectorComponent {
    current_loader: SessionListLoader,
    all_loader: SessionListLoader,
    current_session_path: Option<String>,
    theme: Theme,
    app_kb: KeybindingsManager<AppAction>,
    rename_session: Option<Box<dyn FnMut(String, Option<&str>)>>,
    on_delete: Option<Box<dyn FnMut(String)>>,
    on_select: Box<dyn FnMut(String)>,
    on_cancel: Box<dyn FnMut()>,

    scope: SessionScope,
    sort_mode: SortMode,
    name_filter: NameFilter,
    show_path: bool,
    sessions: Vec<SessionListing>,
    loaded: bool,
    progress: SessionListProgress,
    query: String,
    selected_index: usize,
    confirming_delete: bool,
    status_message: Option<String>,

    mode: Mode,
    rename_input: Input,
    rename_target_path: Option<String>,
    // Focus passthrough (for IME cursor positioning; aligns with pi Focusable).
    focused: bool,
    search: Input,
    top_border: DynamicBorder,
    bottom_border: DynamicBorder,
}

impl SessionSelectorComponent {
    /// Construct the selector: pass in current/all loaders, current session path, theme, app keymap, and callbacks.
    /// The corresponding UI behavior is disabled when `rename_session`/`on_delete` is None.
    /// **Owned (T-AP-08-5)**: theme and app keybinding map are held by value (cloned by the constructor),
    /// Allows the component to be stored in `InteractiveUi` mount fields (avoids self-referential borrow, following the SettingsSelectorComponent precedent).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn new(
        current_loader: SessionListLoader,
        all_loader: SessionListLoader,
        current_session_path: Option<String>,
        theme: Theme,
        app_kb: KeybindingsManager<AppAction>,
        on_select: Box<dyn FnMut(String)>,
        on_cancel: Box<dyn FnMut()>,
        rename_session: Option<Box<dyn FnMut(String, Option<&str>)>>,
        on_delete: Option<Box<dyn FnMut(String)>>,
    ) -> Self {
        let accent = Style::fg(theme.fg(ColorToken::Accent));
        let mut comp = SessionSelectorComponent {
            current_loader,
            all_loader,
            current_session_path,
            theme,
            app_kb,
            rename_session,
            on_delete,
            on_select,
            on_cancel,
            scope: SessionScope::Current,
            sort_mode: SortMode::Threaded,
            name_filter: NameFilter::All,
            show_path: false,
            sessions: Vec::new(),
            loaded: false,
            progress: SessionListProgress {
                loaded: 0,
                total: 0,
            },
            query: String::new(),
            selected_index: 0,
            confirming_delete: false,
            status_message: None,
            mode: Mode::List,
            rename_input: Input::new(),
            rename_target_path: None,
            focused: true,
            search: Input::new(),
            top_border: DynamicBorder::new(accent.clone()),
            bottom_border: DynamicBorder::new(accent),
        };
        comp.reload();
        comp
    }

    /// Reload sessions for the current scope (called on construction/scope switch).
    fn reload(&mut self) {
        let loader = match self.scope {
            SessionScope::Current => &mut self.current_loader,
            SessionScope::All => &mut self.all_loader,
        };
        let mut on_progress = |p: SessionListProgress| {
            self.progress = p;
        };
        let sessions = loader(&mut on_progress);
        self.sessions = sessions;
        self.loaded = true;
        self.selected_index = 0;
        self.confirming_delete = false;
    }

    /// Switch selection scope (current/all) and reload; called by the T-AP-08-5/T-AP-11-1 scope switch UI.
    pub fn set_scope(&mut self, scope: SessionScope) {
        if self.scope != scope {
            self.scope = scope;
            self.reload();
        }
    }

    fn visible_sessions(&self) -> Vec<&SessionListing> {
        filter_and_sort_sessions_raw(
            &self.sessions,
            &self.query,
            self.sort_mode,
            self.name_filter,
        )
    }

    /// Currently selected session (after filtering; None if none).
    fn selected(&self) -> Option<&SessionListing> {
        self.visible_sessions().get(self.selected_index).copied()
    }

    fn move_selection(&mut self, delta: isize) -> bool {
        let len = self.visible_sessions().len();
        if len == 0 {
            return false;
        }
        self.selected_index =
            ((self.selected_index as isize + delta).rem_euclid(len as isize)) as usize;
        true
    }

    /// Public accessor for the current session list focus (used by T-AP-08-5 to set focus).
    pub fn session_list_focused(&self) -> bool {
        self.focused
    }

    /// Set focus (aligns with pi Focusable semantics; in rename mode, focus moves to the input box).
    pub fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
        self.rename_input
            .set_focused(focused && matches!(self.mode, Mode::Rename));
    }

    /// Update the search query and reset the selection (pi search box input path).
    fn set_query(&mut self, query: String) {
        self.query = query;
        self.selected_index = 0;
        self.confirming_delete = false;
    }

    /// Delete the current selection (two-step confirmation: first press sets the confirm state and prompts, second press executes).
    fn delete_selected(&mut self) {
        let Some(selected) = self
            .selected()
            .map(|l| l.path.to_string_lossy().into_owned())
        else {
            return;
        };
        if let Some(on_delete) = &mut self.on_delete {
            if self.confirming_delete {
                on_delete(selected);
                self.confirming_delete = false;
                self.status_message = None;
            } else {
                self.confirming_delete = true;
                self.status_message = Some("Press again to confirm delete".to_string());
            }
        }
    }

    /// Current selection scope (default current; T-AP-08-5 may change to all to list sessions across all projects).
    pub fn scope(&self) -> &SessionScope {
        &self.scope
    }

    fn handle_select_action(&mut self, action: SelectAction) -> bool {
        match action {
            SelectAction::MoveUp => {
                self.move_selection(-1) && {
                    self.confirming_delete = false;
                    true
                }
            }
            SelectAction::MoveDown => {
                self.move_selection(1) && {
                    self.confirming_delete = false;
                    true
                }
            }
            SelectAction::Confirm => {
                if let Some(selected) = self.selected() {
                    let path = selected.path.to_string_lossy().into_owned();
                    // Trigger the callback after the borrow ends
                    let on_select = &mut self.on_select;
                    on_select(path);
                    true
                } else {
                    false
                }
            }
            SelectAction::Cancel => {
                (self.on_cancel)();
                true
            }
        }
    }

    fn handle_app_action(&mut self, action: AppAction) -> bool {
        match action {
            AppAction::SessionToggleSort => {
                self.sort_mode = match self.sort_mode {
                    SortMode::Threaded => SortMode::Recent,
                    SortMode::Recent => SortMode::Relevance,
                    SortMode::Relevance => SortMode::Threaded,
                };
                self.confirming_delete = false;
                true
            }
            AppAction::SessionToggleNamedFilter => {
                self.name_filter = match self.name_filter {
                    NameFilter::All => NameFilter::Named,
                    NameFilter::Named => NameFilter::All,
                };
                self.selected_index = 0;
                true
            }
            AppAction::SessionTogglePath => {
                self.show_path = !self.show_path;
                true
            }
            AppAction::SessionRename => {
                if self.rename_session.is_some() {
                    if let Some(selected) = self.selected() {
                        let path = selected.path.to_string_lossy().into_owned();
                        let name = selected.name.clone().unwrap_or_default();
                        self.rename_target_path = Some(path);
                        self.rename_input.set_value(name);
                        self.mode = Mode::Rename;
                    }
                }
                true
            }
            AppAction::SessionDelete => {
                if self.on_delete.is_some() {
                    self.delete_selected();
                }
                true
            }
            AppAction::SessionDeleteNoninvasive => {
                // Deletion when the query is empty (non-destructive delete)
                if self.query.trim().is_empty() && self.on_delete.is_some() {
                    self.delete_selected();
                }
                true
            }
            _ => false,
        }
    }
}

/// Filter + sort (over borrowed lists; delegates to the owned version in the search module).
fn filter_and_sort_sessions_raw<'x>(
    sessions: &'x [SessionListing],
    query: &str,
    sort: SortMode,
    name_filter: NameFilter,
) -> Vec<&'x SessionListing> {
    let owned: Vec<SessionListing> = sessions.to_vec();
    let out = filter_and_sort_sessions(owned, query, sort, name_filter);
    // Recover references by id while preserving order (ids are unique)
    sessions
        .iter()
        .filter(|l| out.iter().any(|o| o.id == l.id))
        .collect()
}

impl Component for SessionSelectorComponent {
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        let theme = &self.theme;
        let mut painted = 0u16;
        // Row advancement macro: remaining always equals height - painted (avoids manual bookkeeping; the macro only advances painted)
        macro_rules! take {
            ($n:expr) => {{
                let remaining = height - painted;
                if remaining == 0 {
                    return painted;
                }
                let step = ($n).min(remaining);
                if step > 0 {
                    painted += step;
                }
            }};
        }

        // 1. Top padding + border + padding
        if height > painted {
            buf.fill(x, y + painted, width, 1, &Style::default());
            take!(1);
        }
        if height > painted {
            let n = self
                .top_border
                .render(buf, x, y + painted, width, height - painted);
            take!(n);
        }
        if height > painted {
            buf.fill(x, y + painted, width, 1, &Style::default());
            take!(1);
        }

        // 2. Header row (scope + sort + filter + loading)
        if height > painted {
            let scope_label = match self.scope {
                SessionScope::All => "all",
                SessionScope::Current => "current",
            };
            let sort_label = match self.sort_mode {
                SortMode::Threaded => "threaded",
                SortMode::Recent => "recent",
                SortMode::Relevance => "relevance",
            };
            let named = if self.name_filter == NameFilter::Named {
                "• named"
            } else {
                ""
            };
            let loading = if self.loaded { "" } else { " (loading…)" };
            let title = format!("{scope_label} sessions — {sort_label}{named}{loading}");
            buf.fill(x, y + painted, width, 1, &Style::default());
            buf.write(
                x + 1,
                y + painted,
                &Style::fg(theme.fg(ColorToken::Accent)),
                &title,
            );
            take!(1);
        }

        // 3. Status bar (progress / status message / result count)
        if height > painted {
            let line = if let Some(msg) = &self.status_message {
                msg.clone()
            } else {
                let total = self.sessions.len();
                let shown = self.visible_sessions().len();
                if self.loaded {
                    format!("{shown}/{total} sessions")
                } else {
                    format!("loading {}/{}", self.progress.loaded, self.progress.total)
                }
            };
            buf.fill(x, y + painted, width, 1, &Style::default());
            buf.write(
                x + 1,
                y + painted,
                &Style::fg(theme.fg(ColorToken::Muted)),
                &line,
            );
            take!(1);
        }

        if matches!(self.mode, Mode::Rename) {
            // Rename state: prompt + input box
            if height > painted {
                let hint = "Rename session (enter to save)";
                buf.fill(x, y + painted, width, 1, &Style::default());
                buf.write(
                    x + 1,
                    y + painted,
                    &Style::fg(theme.fg(ColorToken::Muted)),
                    hint,
                );
                take!(1);
            }
            if height > painted {
                let n = self
                    .rename_input
                    .render(buf, x, y + painted, width, height - painted);
                take!(n);
            }
        } else {
            // 4. Search box
            if height > painted {
                self.search.set_value(self.query.clone());
                let n = self
                    .search
                    .render(buf, x, y + painted, width, height - painted);
                take!(n);
            }

            // 5. List (scrolling window)
            let visible = self.visible_sessions();
            let total = visible.len();
            if total > 0 {
                let list_height = (height - painted) as usize;
                let max_vis = list_height;
                let half = max_vis / 2;
                let start = if self.selected_index < half {
                    0
                } else if self.selected_index + half >= total {
                    total.saturating_sub(max_vis).min(total)
                } else {
                    self.selected_index - half
                };
                let end = (start + max_vis).min(total);
                let home = std::env::var("HOME").ok();
                for (i, listing) in visible.iter().enumerate().take(end.max(start)).skip(start) {
                    if height - painted == 0 {
                        break;
                    }
                    let is_selected = i == self.selected_index;
                    buf.fill(x, y + painted, width, 1, &Style::default());
                    let (prefix, style) = if is_selected {
                        ("→ ", Style::fg(theme.fg(ColorToken::Accent)))
                    } else {
                        ("  ", Style::default())
                    };
                    buf.write(x, y + painted, &style, prefix);
                    let date = format_session_date(listing.modified);
                    let display_path = if self.show_path {
                        shorten_path(&listing.cwd, home.as_deref())
                    } else {
                        listing.cwd.clone()
                    };
                    let name = listing.name.as_deref().unwrap_or(&listing.id);
                    let line = format!("{date} {name}  [{display_path}]");
                    buf.write(x + 2, y + painted, &style, &line);
                    // Marker for the currently selected session
                    if is_selected
                        && self
                            .current_session_path
                            .as_deref()
                            .is_some_and(|p| *listing.path.to_string_lossy() == *p)
                    {
                        buf.write(
                            x + width.saturating_sub(8),
                            y + painted,
                            &style,
                            " (active)",
                        );
                    }
                    take!(1);
                }
            } else {
                if height > painted {
                    buf.fill(x, y + painted, width, 1, &Style::default());
                    buf.write(
                        x + 1,
                        y + painted,
                        &Style::fg(theme.fg(ColorToken::Muted)),
                        "  No sessions",
                    );
                    take!(1);
                }
            }
        }

        // 6. Bottom padding + border
        if height > painted {
            buf.fill(x, y + painted, width, 1, &Style::default());
            take!(1);
        }
        if height > painted {
            let n = self
                .bottom_border
                .render(buf, x, y + painted, width, height - painted);
            take!(n);
        }

        painted
    }

    fn handle_key(&mut self, key: &crossterm::event::KeyEvent) -> bool {
        // Rename mode: Esc exits rename, everything else goes to the input box
        if matches!(self.mode, Mode::Rename) {
            let is_escape = matches!(
                (key.code, key.modifiers),
                (crossterm::event::KeyCode::Esc, _)
                    | (
                        crossterm::event::KeyCode::Char('c'),
                        crossterm::event::KeyModifiers::CONTROL,
                    )
            );
            if is_escape {
                self.mode = Mode::List;
                return true;
            }
            let consumed = self.rename_input.handle_key(key);
            if consumed {
                // Enter submits the rename
                if key.code == crossterm::event::KeyCode::Enter {
                    let path = self.rename_target_path.clone().unwrap_or_default();
                    let new_name = self.rename_input.get_value().to_string();
                    if let Some(cb) = &mut self.rename_session {
                        cb(path, Some(&new_name));
                    }
                    self.mode = Mode::List;
                }
            }
            return consumed;
        }

        // List mode
        // 1. app actions
        if let Some(kid) = KeyId::from_key_event(*key) {
            for action in [
                AppAction::SessionToggleSort,
                AppAction::SessionToggleNamedFilter,
                AppAction::SessionTogglePath,
                AppAction::SessionRename,
                AppAction::SessionDelete,
                AppAction::SessionDeleteNoninvasive,
            ] {
                if self.app_kb.matches(&kid, action) {
                    return self.handle_app_action(action);
                }
            }
        }

        // 2. select action (inline default keys: ↑↓/Enter/Esc/Ctrl+C)
        let action = match (key.code, key.modifiers) {
            (crossterm::event::KeyCode::Up, _) => Some(SelectAction::MoveUp),
            (crossterm::event::KeyCode::Down, _) => Some(SelectAction::MoveDown),
            (crossterm::event::KeyCode::Enter, _) => Some(SelectAction::Confirm),
            (crossterm::event::KeyCode::Esc, _)
            | (crossterm::event::KeyCode::Char('c'), crossterm::event::KeyModifiers::CONTROL) => {
                Some(SelectAction::Cancel)
            }
            _ => None,
        };
        if let Some(action) = action {
            return self.handle_select_action(action);
        }

        // 3. Search input
        let consumed_search = self.search.handle_key(key);
        if consumed_search {
            // Enter in the search box does not submit the selection (keeps list Enter semantics), only Esc returns; all others update the query
            self.set_query(self.search.get_value().to_string());
            if key.code == crossterm::event::KeyCode::Esc {
                // Esc already took the select Cancel branch, so it never reaches here
            }
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keybindings::app_default_bindings;
    use std::collections::HashMap;

    fn theme() -> Theme {
        zen_tui::theme::builtin_dark_theme().clone()
    }

    fn kb() -> KeybindingsManager<AppAction> {
        KeybindingsManager::new(app_default_bindings(), HashMap::new())
    }

    fn listing(id: &str, name: Option<&str>, text: &str, cwd: &str) -> SessionListing {
        SessionListing::new(
            id.into(),
            std::path::PathBuf::from(format!("/tmp/{id}.jsonl")),
            cwd.into(),
            name.map(|s| s.to_string()),
            None,
            chrono::Utc::now(),
            chrono::Utc::now(),
            1,
            "first".into(),
            text.into(),
        )
    }

    fn loader(items: Vec<SessionListing>) -> SessionListLoader {
        Box::new(move |_progress| items.clone())
    }

    fn esc() -> crossterm::event::KeyEvent {
        crossterm::event::KeyEvent::from(crossterm::event::KeyCode::Esc)
    }
    fn enter() -> crossterm::event::KeyEvent {
        crossterm::event::KeyEvent::from(crossterm::event::KeyCode::Enter)
    }
    fn down() -> crossterm::event::KeyEvent {
        crossterm::event::KeyEvent::from(crossterm::event::KeyCode::Down)
    }

    #[test]
    fn enter_selects_first_session() {
        let selected = std::rc::Rc::new(std::cell::RefCell::new(None::<String>));
        let selected_captured = selected.clone();
        let kb = kb();
        let mut comp = SessionSelectorComponent::new(
            loader(vec![
                listing("s1", None, "text a", "/a"),
                listing("s2", None, "text b", "/b"),
            ]),
            loader(vec![]),
            None,
            theme(),
            kb.clone(),
            Box::new(move |p| *selected_captured.borrow_mut() = Some(p)),
            Box::new(|| {}),
            None,
            None,
        );
        assert!(comp.handle_key(&enter()));
        assert_eq!(*selected.borrow(), Some("/tmp/s1.jsonl".to_string()));
    }

    #[test]
    fn esc_cancels() {
        let cancelled = std::rc::Rc::new(std::cell::RefCell::new(0));
        let cancelled_captured = cancelled.clone();
        let kb = kb();
        let mut comp = SessionSelectorComponent::new(
            loader(vec![listing("s1", None, "t", "/")]),
            loader(vec![]),
            None,
            theme(),
            kb.clone(),
            Box::new(|_| {}),
            Box::new(move || *cancelled_captured.borrow_mut() += 1),
            None,
            None,
        );
        assert!(comp.handle_key(&esc()));
        assert_eq!(*cancelled.borrow(), 1);
    }

    #[test]
    fn down_moves_selection() {
        let kb = kb();
        let mut comp = SessionSelectorComponent::new(
            loader(vec![
                listing("s1", None, "a", "/"),
                listing("s2", None, "b", "/"),
            ]),
            loader(vec![]),
            None,
            theme(),
            kb.clone(),
            Box::new(|_| {}),
            Box::new(|| {}),
            None,
            None,
        );
        assert!(comp.handle_key(&down()));
        assert_eq!(comp.selected().unwrap().id, "s2");
    }

    #[test]
    fn delete_requires_two_presses() {
        let deleted = std::rc::Rc::new(std::cell::RefCell::new(None::<String>));
        let deleted_captured = deleted.clone();
        let kb = kb();
        let mut comp = SessionSelectorComponent::new(
            loader(vec![listing("s1", None, "a", "/")]),
            loader(vec![]),
            None,
            theme(),
            kb.clone(),
            Box::new(|_| {}),
            Box::new(|| {}),
            None,
            Some(Box::new(move |p| *deleted_captured.borrow_mut() = Some(p))),
        );
        let ctrl_d = crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('d'),
            crossterm::event::KeyModifiers::CONTROL,
        );
        // First press: enter confirmation state
        assert!(comp.handle_key(&ctrl_d));
        assert!(deleted.borrow().is_none());
        assert!(comp.confirming_delete);
        // Second press: perform deletion
        assert!(comp.handle_key(&ctrl_d));
        assert_eq!(*deleted.borrow(), Some("/tmp/s1.jsonl".to_string()));
    }

    #[test]
    fn toggle_sort_cycles_modes() {
        let kb = kb();
        let mut comp = SessionSelectorComponent::new(
            loader(vec![listing("s1", None, "a", "/")]),
            loader(vec![]),
            None,
            theme(),
            kb.clone(),
            Box::new(|_| {}),
            Box::new(|| {}),
            None,
            None,
        );
        let ctrl_s = crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('s'),
            crossterm::event::KeyModifiers::CONTROL,
        );
        assert!(comp.handle_key(&ctrl_s));
        assert_eq!(comp.sort_mode, SortMode::Recent);
        assert!(comp.handle_key(&ctrl_s));
        assert_eq!(comp.sort_mode, SortMode::Relevance);
        assert!(comp.handle_key(&ctrl_s));
        assert_eq!(comp.sort_mode, SortMode::Threaded);
    }

    #[test]
    fn search_filters_list() {
        let kb = kb();
        let mut comp = SessionSelectorComponent::new(
            loader(vec![
                listing("s1", None, "rust refactor", "/a"),
                listing("s2", None, "unrelated", "/b"),
            ]),
            loader(vec![]),
            None,
            theme(),
            kb.clone(),
            Box::new(|_| {}),
            Box::new(|| {}),
            None,
            None,
        );
        // Type "rust" (printed characters go to the search box verbatim)
        for c in 'r'.to_string().chars() {
            let _ = c;
        }
        for c in "rust".chars() {
            assert!(comp.handle_key(&crossterm::event::KeyEvent::from(
                crossterm::event::KeyCode::Char(c)
            )));
        }
        let visible = comp.visible_sessions();
        assert_eq!(visible.len(), 1, "filtered to rust session");
        assert_eq!(visible[0].id, "s1");
    }

    #[test]
    fn renders_header_search_and_list() {
        let kb = kb();
        let mut comp = SessionSelectorComponent::new(
            loader(vec![
                listing("s1", Some("first"), "text", "/a"),
                listing("s2", None, "text", "/b"),
            ]),
            loader(vec![]),
            None,
            theme(),
            kb.clone(),
            Box::new(|_| {}),
            Box::new(|| {}),
            None,
            None,
        );
        let mut buf = ScreenBuffer::new(80, 20);
        let painted = comp.render(&mut buf, 0, 0, 80, 20);
        assert!(painted >= 6);
        let mut out = Vec::new();
        buf.flush(&mut out).unwrap();
        let text = zen_core::ansi::strip_ansi(&String::from_utf8_lossy(&out));
        assert!(text.contains("current sessions"));
        assert!(text.contains("first")); // s1 has a name → show the name
        assert!(text.contains("s2")); // s2 has no name → show the id
    }

    #[test]
    fn rename_mode_flow() {
        let renamed = std::rc::Rc::new(std::cell::RefCell::new(None::<(String, String)>));
        let renamed_captured = renamed.clone();
        let kb = kb();
        let mut comp = SessionSelectorComponent::new(
            loader(vec![listing("s1", Some("old"), "t", "/")]),
            loader(vec![]),
            None,
            theme(),
            kb.clone(),
            Box::new(|_| {}),
            Box::new(|| {}),
            Some(Box::new(move |path, name| {
                *renamed_captured.borrow_mut() = Some((path, name.unwrap_or("").to_string()))
            })),
            None,
        );
        let ctrl_r = crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('r'),
            crossterm::event::KeyModifiers::CONTROL,
        );
        assert!(comp.handle_key(&ctrl_r));
        assert!(matches!(comp.mode, Mode::Rename));
        assert!(comp.handle_key(&esc()));
        assert!(matches!(comp.mode, Mode::List));
        assert!(renamed.borrow().is_none());
    }

    #[test]
    fn session_list_focused_toggle() {
        let kb = kb();
        let mut comp = SessionSelectorComponent::new(
            loader(vec![]),
            loader(vec![]),
            None,
            theme(),
            kb.clone(),
            Box::new(|_| {}),
            Box::new(|| {}),
            None,
            None,
        );
        assert!(comp.session_list_focused());
        comp.set_focused(false);
        assert!(!comp.session_list_focused());
    }
}
