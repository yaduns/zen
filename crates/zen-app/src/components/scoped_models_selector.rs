// REQ-V2-02 (pi-369, pi-368): scoped-models-selector — Ctrl+P cycle set configuration panel (T-V2-02-4).
// Corresponding original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/scoped-models-selector.ts
// and model-search.ts (`getModelSearchText` corpus).
//
// Responsibility: user opens this panel via `/scoped-models` to check/uncheck/order the model subset in the Ctrl+P cycle;
// Changes are session-only; only `app.models.save` writes settings (footer shows `(unsaved)`).
// Navigation/confirm/cancel routed via injected `select_kb` (SelectAction domain); panel actions via `app_kb` (AppAction domain)
// routing — both keymaps are constructor-injected, so user overrides (T-AP-06-2) take effect here.

use crossterm::event::KeyEvent;
use crossterm::style::Attribute;
use zen_ai::types::Model;
use zen_tui::components::input::Input;
use zen_tui::components::select_list::SelectAction;
use zen_tui::components::Component;
use zen_tui::keybindings::KeybindingsManager;
use zen_tui::keys::KeyId;
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{ColorToken, Theme};

use crate::components::dynamic_border::DynamicBorder;
use crate::keybindings::AppAction;
use crate::model_catalog::full_id;

/// Ctrl+P cycle set: `None` = all enabled (no filtering); `Some(ids)` = explicit ordered subset (order is cycle order).
/// Reuse pi `EnabledIds` semantics; do not create new enum/struct.
pub type EnabledModelIds = Option<Vec<String>>;

/// Three upstream callbacks for scoped-models panel (change / persist / cancel).
pub struct ScopedModelsCallbacks {
    /// Triggered when enabled set or order changes (session-only, not persisted).
    pub on_change: Box<dyn FnMut(EnabledModelIds)>,
    /// Triggered on `app.models.save` (writes settings).
    pub on_persist: Box<dyn FnMut(EnabledModelIds)>,
    /// Esc / Cancel cancels.
    pub on_cancel: Box<dyn FnMut()>,
}

impl Default for ScopedModelsCallbacks {
    fn default() -> Self {
        ScopedModelsCallbacks {
            on_change: Box::new(|_| {}),
            on_persist: Box::new(|_| {}),
            on_cancel: Box::new(|| {}),
        }
    }
}

// ── Pure helpers (aligned item-by-item with pi scoped-models-selector.ts; module-private, consumed only by this file) ──

/// Whether id is enabled (`None` = all enabled → always true).
fn is_enabled(enabled: &EnabledModelIds, id: &str) -> bool {
    enabled
        .as_ref()
        .is_none_or(|ids| ids.iter().any(|x| x == id))
}

/// Toggle enabled state of a single id: starting from `None` gives `[id]`; remove if present, append to end if absent.
fn toggle_enabled(enabled: EnabledModelIds, id: &str) -> EnabledModelIds {
    match enabled {
        None => Some(vec![id.to_string()]),
        Some(mut ids) => {
            if let Some(pos) = ids.iter().position(|x| x == id) {
                ids.remove(pos);
            } else {
                ids.push(id.to_string());
            }
            Some(ids)
        }
    }
}

/// Batch-enable `target_ids` (use `all_ids` when `None`); if enabled set covers all, normalize to `None` (all enabled).
fn enable_all(
    enabled: EnabledModelIds,
    all_ids: &[String],
    target_ids: Option<&[String]>,
) -> EnabledModelIds {
    let Some(mut current) = enabled else {
        return None; // Already all enabled
    };
    let targets = target_ids.unwrap_or(all_ids);
    for id in targets {
        if !current.iter().any(|x| x == id) {
            current.push(id.clone());
        }
    }
    if current.len() == all_ids.len() {
        None
    } else {
        Some(current)
    }
}

/// Batch-clear `target_ids` (clear all when `None`); starting from `None`, filter by targets (keep the rest).
fn clear_all(
    enabled: EnabledModelIds,
    all_ids: &[String],
    target_ids: Option<&[String]>,
) -> EnabledModelIds {
    match enabled {
        None => match target_ids {
            Some(targets) => Some(
                all_ids
                    .iter()
                    .filter(|id| !targets.iter().any(|t| t == *id))
                    .cloned()
                    .collect(),
            ),
            None => Some(Vec::new()),
        },
        Some(ids) => {
            let targets: Vec<String> = match target_ids {
                Some(t) => t.to_vec(),
                None => ids.clone(),
            };
            Some(
                ids.into_iter()
                    .filter(|id| !targets.iter().any(|t| t == id))
                    .collect(),
            )
        }
    }
}

/// Move id up/down one position within enabled set (`delta` = ∓1); return unchanged when `None` or out of bounds.
fn move_enabled(enabled: EnabledModelIds, id: &str, delta: i32) -> EnabledModelIds {
    let mut list = enabled?;
    let Some(index) = list.iter().position(|x| x == id) else {
        return Some(list);
    };
    let new_index = index as i32 + delta;
    if new_index < 0 || new_index as usize >= list.len() {
        return Some(list);
    }
    list.swap(index, new_index as usize);
    Some(list)
}

/// List display order: enabled-set ids first in order, remaining after in `all_ids` order.
fn sorted_ids(enabled: &EnabledModelIds, all_ids: &[String]) -> Vec<String> {
    match enabled {
        None => all_ids.to_vec(),
        Some(ids) => {
            let mut out = ids.clone();
            out.extend(
                all_ids
                    .iter()
                    .filter(|id| !ids.iter().any(|x| &x == id))
                    .cloned(),
            );
            out
        }
    }
}

/// scoped-models fuzzy search corpus (pi `getModelSearchText`: `id provider provider/id …`).
fn search_text(model: &Model) -> String {
    let id = &model.id;
    let provider = &model.provider;
    let name = model
        .name
        .as_deref()
        .map(|n| format!(" {n}"))
        .unwrap_or_default();
    format!("{id} {provider} {provider}/{id} {provider} {id}{name}")
}

/// Filter: empty query returns all in original order; otherwise ascending by fuzzy score (convert `fuzzy_filter` borrowed view to `models` indices).
fn filter_indices(models: &[Model], query: &str) -> Vec<usize> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return (0..models.len()).collect();
    }
    let mut scored: Vec<(usize, f64)> = models
        .iter()
        .enumerate()
        .filter_map(|(i, m)| zen_tui::fuzzy::fuzzy_match(trimmed, &search_text(m)).map(|s| (i, s)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(i, _)| i).collect()
}

/// Ctrl+P cycle set configuration panel (T-V2-02-4): opened by `/scoped-models` — "pick a few frequently used models into the cycle,
/// and decide their order." Enter toggles enabled; `app.models.*` enables all/clears/all by provider/moves up/down,
/// `app.models.save` persists; changes are session-only until saved.
pub struct ScopedModelsSelectorComponent {
    models: Vec<Model>,
    enabled: EnabledModelIds,
    filtered_indices: Vec<usize>,
    selected_index: usize,
    search: Input,
    select_kb: KeybindingsManager<SelectAction>,
    app_kb: KeybindingsManager<AppAction>,
    focused: bool,
    is_dirty: bool,
    callbacks: ScopedModelsCallbacks,
    max_visible: usize,
    theme: Theme,
    top_border: DynamicBorder,
    bottom_border: DynamicBorder,
}

impl ScopedModelsSelectorComponent {
    /// Construct panel: given full model set, current enabled set, theme, dual-domain keymaps, and three callbacks.
    /// `enabled = None` means all enabled (list shows no check/cross).
    /// `select_kb` routes ↑↓/Enter/Esc/Ctrl+C; `app_kb` routes `app.models.*` actions.
    pub fn new(
        models: Vec<Model>,
        enabled: EnabledModelIds,
        theme: &Theme,
        select_kb: KeybindingsManager<SelectAction>,
        app_kb: KeybindingsManager<AppAction>,
        callbacks: ScopedModelsCallbacks,
    ) -> Self {
        let mut search = Input::new();
        search.set_focused(true);
        let mut comp = ScopedModelsSelectorComponent {
            filtered_indices: filter_indices(&models, ""),
            models,
            enabled,
            selected_index: 0,
            search,
            select_kb,
            app_kb,
            focused: true,
            is_dirty: false,
            callbacks,
            max_visible: 8,
            top_border: DynamicBorder::new(Style::fg(theme.fg(ColorToken::BorderAccent))),
            bottom_border: DynamicBorder::new(Style::fg(theme.fg(ColorToken::BorderAccent))),
            theme: theme.clone(),
        };
        comp.clamp_selected();
        comp
    }

    /// Current enabled set (including unsaved changes). Consumer: `InteractiveUi`/main task reads and tests.
    pub fn enabled(&self) -> EnabledModelIds {
        self.enabled.clone()
    }

    /// Whether there are unsaved changes (footer `(unsaved)`). Consumer: tests.
    pub fn is_dirty(&self) -> bool {
        self.is_dirty
    }

    /// Set focus (pass through to inner search box). Consumer: `InteractiveUi`.
    pub fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
        self.search.set_focused(focused);
    }

    /// Canonical ids of all models (list order = catalog order).
    fn all_ids(&self) -> Vec<String> {
        self.models.iter().map(full_id).collect()
    }

    /// `full_id` of selected item in current filtered view.
    fn selected_full_id(&self) -> Option<String> {
        let idx = *self.filtered_indices.get(self.selected_index)?;
        Some(full_id(&self.models[idx]))
    }

    /// All `full_id`s in current filtered view (target set for `app.models.enableAll/clearAll`).
    fn filtered_ids(&self) -> Vec<String> {
        self.filtered_indices
            .iter()
            .map(|&i| full_id(&self.models[i]))
            .collect()
    }

    /// Recompute filtered view (called after query changes/enabled-set order changes) and rebuild display order.
    fn rebuild(&mut self) {
        // Display order = sorted_ids (enabled order first); then apply fuzzy filter, preserving enabled order.
        let query = self.search.get_value().to_string();
        let display = sorted_ids(&self.enabled, &self.all_ids());
        if query.trim().is_empty() {
            self.filtered_indices = display
                .iter()
                .filter_map(|id| {
                    let (provider, mid) = id.split_once('/')?;
                    self.models
                        .iter()
                        .position(|m| m.provider == provider && m.id == mid)
                })
                .collect();
        } else {
            let mut scored: Vec<(usize, f64)> = display
                .iter()
                .filter_map(|id| {
                    let (provider, mid) = id.split_once('/')?;
                    let idx = self
                        .models
                        .iter()
                        .position(|m| m.provider == provider && m.id == mid)?;
                    zen_tui::fuzzy::fuzzy_match(query.trim(), &search_text(&self.models[idx]))
                        .map(|s| (idx, s))
                })
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            self.filtered_indices = scored.into_iter().map(|(i, _)| i).collect();
        }
        self.clamp_selected();
    }

    /// Clamp selected index to filtered view range.
    fn clamp_selected(&mut self) {
        let len = self.filtered_indices.len();
        self.selected_index = if len == 0 {
            0
        } else {
            self.selected_index.min(len - 1)
        };
    }

    /// ↑↓ wrap-around movement.
    fn move_selection(&mut self, down: bool) {
        let len = self.filtered_indices.len();
        if len == 0 {
            return;
        }
        if down {
            self.selected_index = (self.selected_index + 1) % len;
        } else {
            self.selected_index = (self.selected_index + len - 1) % len;
        }
    }

    fn notify_change(&mut self) {
        (self.callbacks.on_change)(self.enabled.clone());
    }

    /// `app.models.toggleProvider`: batch enable/disable all ids of the current item's provider.
    fn toggle_provider(&mut self) {
        let Some(current) = self.selected_full_id() else {
            return;
        };
        let Some(provider) = current.split_once('/').map(|(p, _)| p.to_string()) else {
            return;
        };
        let provider_ids: Vec<String> = self
            .all_ids()
            .into_iter()
            .filter(|id| id.split_once('/').is_some_and(|(p, _)| p == provider))
            .collect();
        let all_enabled = provider_ids.iter().all(|id| is_enabled(&self.enabled, id));
        let all_ids = self.all_ids();
        self.enabled = if all_enabled {
            clear_all(self.enabled.take(), &all_ids, Some(&provider_ids))
        } else {
            enable_all(self.enabled.take(), &all_ids, Some(&provider_ids))
        };
        self.is_dirty = true;
        self.rebuild();
        self.notify_change();
    }

    /// Style group shared by text and keybindings.
    fn accent_bold(&self) -> Style {
        self.theme
            .style(ColorToken::Accent, None, &[Attribute::Bold])
    }
}

impl Component for ScopedModelsSelectorComponent {
    /// Render panel: border + title/hint + search box + checkbox list + scroll info + footer keybindings and count.
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        if width == 0 || height == 0 {
            return 0;
        }
        let muted = Style::fg(self.theme.fg(ColorToken::Muted));
        let dim = Style::fg(self.theme.fg(ColorToken::Dim));
        let accent = Style::fg(self.theme.fg(ColorToken::Accent));
        let line_style = Style::default();

        let mut row = y;
        let mut remaining = height;
        let mut painted = 0u16;

        // Top border
        if remaining > 0 {
            let n = self.top_border.render(buf, x, row, width, remaining);
            painted += n;
            row += n;
            remaining = remaining.saturating_sub(n);
        }
        // Title
        if remaining > 0 {
            buf.fill(x, row, width, 1, &line_style);
            buf.write(x + 1, row, &self.accent_bold(), "Model Configuration");
            painted += 1;
            row += 1;
            remaining -= 1;
        }
        // Hint
        if remaining > 0 {
            let save_key = key_first(&self.app_kb, AppAction::ScopedModelsSave);
            let hint = format!("Session-only. {save_key} to save to settings.");
            buf.fill(x, row, width, 1, &line_style);
            buf.write(x + 1, row, &muted, &hint);
            painted += 1;
            row += 1;
            remaining -= 1;
        }
        // Empty line
        if remaining > 0 {
            buf.fill(x, row, width, 1, &line_style);
            painted += 1;
            row += 1;
            remaining -= 1;
        }
        // Search box
        if remaining > 0 {
            let n = self.search.render(buf, x, row, width, remaining);
            painted += n;
            row += n;
            remaining = remaining.saturating_sub(n);
        }
        // Empty line
        if remaining > 0 {
            buf.fill(x, row, width, 1, &line_style);
            painted += 1;
            row += 1;
            remaining -= 1;
        }

        // List (scroll window, selected item centered)
        if remaining > 0 {
            let total = self.filtered_indices.len();
            if total == 0 {
                buf.fill(x, row, width, 1, &line_style);
                buf.write(x + 2, row, &muted, "No matching models");
                painted += 1;
                row += 1;
                remaining -= 1;
            } else {
                let max_visible = self.max_visible;
                let start = self
                    .selected_index
                    .saturating_sub(max_visible / 2)
                    .min(total.saturating_sub(max_visible));
                let end = (start + max_visible).min(total);
                let all_enabled = self.enabled.is_none();
                for i in start..end {
                    if remaining == 0 {
                        break;
                    }
                    let idx = self.filtered_indices[i];
                    let model = &self.models[idx];
                    let id = full_id(model);
                    let selected = i == self.selected_index;
                    let prefix = if selected { "→ " } else { "  " };
                    let mut text = format!("{prefix}{}", model.id);
                    text.push_str(&format!(" [{}]", model.provider));
                    if !all_enabled {
                        text.push_str(if is_enabled(&self.enabled, &id) {
                            " ✓"
                        } else {
                            " ✗"
                        });
                    }
                    buf.fill(x, row, width, 1, &line_style);
                    let style = if selected {
                        accent.clone()
                    } else {
                        Style::default()
                    };
                    buf.write(x + 1, row, &style, &text);
                    painted += 1;
                    row += 1;
                    remaining -= 1;
                }
            }
        }
        // Scroll info + selected item name
        if remaining > 0 {
            let total = self.filtered_indices.len();
            if total > 0 {
                let name = self
                    .filtered_indices
                    .get(self.selected_index)
                    .map(|&i| {
                        self.models[i]
                            .name
                            .clone()
                            .unwrap_or_else(|| self.models[i].id.clone())
                    })
                    .unwrap_or_default();
                let info = format!("  ({}/{})  Name: {}", self.selected_index + 1, total, name);
                buf.fill(x, row, width, 1, &line_style);
                buf.write(x + 1, row, &muted, &info);
                painted += 1;
                row += 1;
                remaining -= 1;
            }
        }
        // Empty line
        if remaining > 0 {
            buf.fill(x, row, width, 1, &line_style);
            painted += 1;
            row += 1;
            remaining -= 1;
        }
        // Footer keybindings + count
        if remaining > 0 {
            let confirm = key_first(&self.select_kb, SelectAction::Confirm);
            let enable_all = key_first(&self.app_kb, AppAction::ScopedModelsEnableAll);
            let clear_all = key_first(&self.app_kb, AppAction::ScopedModelsClearAll);
            let provider = key_first(&self.app_kb, AppAction::ScopedModelsToggleProvider);
            let up = key_first(&self.app_kb, AppAction::ScopedModelsReorderUp);
            let down = key_first(&self.app_kb, AppAction::ScopedModelsReorderDown);
            let save = key_first(&self.app_kb, AppAction::ScopedModelsSave);
            let enabled_count = self
                .enabled
                .as_ref()
                .map(|ids| ids.len())
                .unwrap_or(self.models.len());
            let all_enabled = self.enabled.is_none();
            let count_text = if all_enabled {
                "all enabled".to_string()
            } else {
                format!("{enabled_count}/{} enabled", self.models.len())
            };
            let footer = format!(
                "  {confirm} toggle · {enable_all} all · {clear_all} clear · {provider} provider · {up}/{down} reorder · {save} save · {count_text}"
            );
            let style = dim.clone();
            buf.fill(x, row, width, 1, &line_style);
            buf.write(x + 1, row, &style, &footer);
            if self.is_dirty {
                // Dirty marker (warning semantics) drawn at visible line end
                let used = zen_tui::screen::visible_width(&footer) as u16;
                if x + 1 + used + 1 < x + width {
                    buf.write(
                        x + 1 + used,
                        row,
                        &Style::fg(self.theme.fg(ColorToken::Warning)),
                        " (unsaved)",
                    );
                }
            }
            painted += 1;
            row += 1;
            remaining -= 1;
        }
        // Bottom border
        if remaining > 0 {
            let n = self.bottom_border.render(buf, x, row, width, remaining);
            painted += n;
        }
        painted
    }

    /// Key routing: `select_kb` (Up/Down/Confirm/Cancel) + `app_kb` (save/enableAll/clearAll/
    /// toggleProvider/reorderUp/reorderDown) + remaining characters go to search box. Returns whether consumed.
    fn handle_key(&mut self, key: &KeyEvent) -> bool {
        let Some(kid) = KeyId::from_key_event(*key) else {
            return false;
        };

        if self.select_kb.matches(&kid, SelectAction::MoveUp) {
            self.move_selection(false);
            return true;
        }
        if self.select_kb.matches(&kid, SelectAction::MoveDown) {
            self.move_selection(true);
            return true;
        }
        if self.select_kb.matches(&kid, SelectAction::Confirm) {
            if let Some(id) = self.selected_full_id() {
                self.enabled = toggle_enabled(self.enabled.take(), &id);
                self.is_dirty = true;
                self.rebuild();
                self.notify_change();
            }
            return true;
        }
        if self.select_kb.matches(&kid, SelectAction::Cancel) {
            (self.callbacks.on_cancel)();
            return true;
        }
        if self.app_kb.matches(&kid, AppAction::ScopedModelsSave) {
            let enabled = self.enabled.clone();
            (self.callbacks.on_persist)(enabled);
            self.is_dirty = false;
            return true;
        }
        if self.app_kb.matches(&kid, AppAction::ScopedModelsEnableAll) {
            let targets = if self.search.get_value().trim().is_empty() {
                None
            } else {
                Some(self.filtered_ids())
            };
            let all_ids = self.all_ids();
            self.enabled = enable_all(self.enabled.take(), &all_ids, targets.as_deref());
            self.is_dirty = true;
            self.rebuild();
            self.notify_change();
            return true;
        }
        if self.app_kb.matches(&kid, AppAction::ScopedModelsClearAll) {
            let targets = if self.search.get_value().trim().is_empty() {
                None
            } else {
                Some(self.filtered_ids())
            };
            let all_ids = self.all_ids();
            self.enabled = clear_all(self.enabled.take(), &all_ids, targets.as_deref());
            self.is_dirty = true;
            self.rebuild();
            self.notify_change();
            return true;
        }
        if self
            .app_kb
            .matches(&kid, AppAction::ScopedModelsToggleProvider)
        {
            self.toggle_provider();
            return true;
        }
        if self.app_kb.matches(&kid, AppAction::ScopedModelsReorderUp) {
            if let Some(id) = self.selected_full_id() {
                if is_enabled(&self.enabled, &id) {
                    self.enabled = move_enabled(self.enabled.take(), &id, -1);
                    self.selected_index = self.selected_index.saturating_sub(1);
                    self.is_dirty = true;
                    self.rebuild();
                    self.notify_change();
                }
            }
            return true;
        }
        if self
            .app_kb
            .matches(&kid, AppAction::ScopedModelsReorderDown)
        {
            if let Some(id) = self.selected_full_id() {
                if is_enabled(&self.enabled, &id) {
                    self.enabled = move_enabled(self.enabled.take(), &id, 1);
                    self.selected_index = (self.selected_index + 1)
                        .min(self.filtered_indices.len().saturating_sub(1));
                    self.is_dirty = true;
                    self.rebuild();
                    self.notify_change();
                }
            }
            return true;
        }

        // Remaining keys go to search box (edit/backspace/left/right); recompute filtering after value changes
        if self.search.handle_key(key) {
            self.rebuild();
            return true;
        }
        false
    }
}

/// Get display string of the action's first effective key (empty string if unbound).
fn key_first<A: Clone + Eq + std::hash::Hash>(kb: &KeybindingsManager<A>, action: A) -> String {
    kb.get_keys(action)
        .first()
        .map(|k| k.to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn theme() -> &'static Theme {
        zen_tui::theme::builtin_dark_theme()
    }

    fn select_kb() -> KeybindingsManager<SelectAction> {
        KeybindingsManager::new(
            zen_tui::keybindings::select_default_bindings(),
            std::collections::HashMap::new(),
        )
    }

    fn app_kb() -> KeybindingsManager<AppAction> {
        KeybindingsManager::new(
            crate::keybindings::app_default_bindings(),
            std::collections::HashMap::new(),
        )
    }

    fn models() -> Vec<Model> {
        vec![
            Model::from_env().with_id("m-a").with_name("Alpha"),
            Model::from_env().with_id("m-b"),
            Model::from_env().with_id("m-c"),
        ]
    }

    fn ids(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    // ── Pure helpers ──

    #[test]
    fn is_enabled_semantics() {
        assert!(is_enabled(&None, "a"));
        assert!(is_enabled(&Some(ids(&["a"])), "a"));
        assert!(!is_enabled(&Some(ids(&["a"])), "b"));
    }

    #[test]
    fn toggle_first_starts_singleton() {
        assert_eq!(toggle_enabled(None, "a"), Some(ids(&["a"])));
    }

    #[test]
    fn toggle_adds_and_removes() {
        assert_eq!(
            toggle_enabled(Some(ids(&["a"])), "b"),
            Some(ids(&["a", "b"]))
        );
        assert_eq!(
            toggle_enabled(Some(ids(&["a", "b"])), "a"),
            Some(ids(&["b"]))
        );
    }

    #[test]
    fn enable_all_and_saturation() {
        let all = ids(&["a", "b"]);
        assert_eq!(enable_all(None, &all, None), None, "Already fully enabled stays fully enabled");
        assert_eq!(
            enable_all(Some(ids(&["a"])), &all, None),
            None,
            "Fully enabled after padding"
        );
        assert_eq!(
            enable_all(Some(ids(&["a"])), &all, Some(&ids(&["b"]))),
            None
        );
    }

    #[test]
    fn clear_all_paths() {
        let all = ids(&["a", "b", "c"]);
        assert_eq!(clear_all(None, &all, None), Some(Vec::new()));
        assert_eq!(
            clear_all(None, &all, Some(&ids(&["b"]))),
            Some(ids(&["a", "c"]))
        );
        assert_eq!(
            clear_all(Some(ids(&["a", "b"])), &all, Some(&ids(&["a"]))),
            Some(ids(&["b"]))
        );
    }

    #[test]
    fn move_enabled_bounds_and_none() {
        assert_eq!(move_enabled(None, "a", -1), None);
        assert_eq!(
            move_enabled(Some(ids(&["a", "b"])), "a", -1),
            Some(ids(&["a", "b"])),
            "Out of range left as-is"
        );
        assert_eq!(
            move_enabled(Some(ids(&["a", "b"])), "a", 1),
            Some(ids(&["b", "a"]))
        );
        assert_eq!(
            move_enabled(Some(ids(&["a", "b"])), "z", 1),
            Some(ids(&["a", "b"])),
            "Not in the set left as-is"
        );
    }

    #[test]
    fn sorted_ids_enabled_first() {
        let all = ids(&["a", "b", "c"]);
        assert_eq!(sorted_ids(&None, &all), all);
        assert_eq!(
            sorted_ids(&Some(ids(&["c", "a"])), &all),
            ids(&["c", "a", "b"])
        );
    }

    #[test]
    fn search_text_has_front_loaded_provider() {
        let m = Model::from_env().with_id("m-x");
        let t = search_text(&m);
        // pi getModelSearchText shape: `id provider provider/id …` (provider prefix comes first)
        assert!(t.starts_with("m-x "));
        assert!(t.contains(&format!("{}/m-x", m.provider)));
    }

    #[test]
    fn filter_indices_empty_query_all_in_order() {
        let ms = models();
        assert_eq!(filter_indices(&ms, ""), vec![0, 1, 2]);
    }

    // ── Components ──

    fn component(
        enabled: EnabledModelIds,
        on_change: Rc<RefCell<Vec<EnabledModelIds>>>,
        on_persist: Rc<RefCell<Vec<EnabledModelIds>>>,
    ) -> ScopedModelsSelectorComponent {
        let change = on_change.clone();
        let persist = on_persist.clone();
        ScopedModelsSelectorComponent::new(
            models(),
            enabled,
            theme(),
            select_kb(),
            app_kb(),
            ScopedModelsCallbacks {
                on_change: Box::new(move |e| change.borrow_mut().push(e)),
                on_persist: Box::new(move |e| persist.borrow_mut().push(e)),
                on_cancel: Box::new(|| {}),
            },
        )
    }

    #[test]
    fn enter_toggles_and_marks_dirty_and_notifies() {
        let changes = Rc::new(RefCell::new(Vec::new()));
        let persists = Rc::new(RefCell::new(Vec::new()));
        let mut comp = component(None, changes.clone(), persists);
        let enter = KeyEvent::from(crossterm::event::KeyCode::Enter);
        assert!(comp.handle_key(&enter));
        assert!(comp.is_dirty());
        let after = comp.enabled();
        assert!(
            matches!(after, Some(ref v) if v.len() == 1),
            "First toggle starts from [id]"
        );
        assert_eq!(changes.borrow().len(), 1);
    }

    #[test]
    fn save_persists_and_clears_dirty() {
        let changes = Rc::new(RefCell::new(Vec::new()));
        let persists = Rc::new(RefCell::new(Vec::new()));
        let mut comp = component(Some(ids(&["x"])), changes, persists.clone());
        // app.models.save defaults to ctrl+s
        let key = KeyEvent::new(
            crossterm::event::KeyCode::Char('s'),
            crossterm::event::KeyModifiers::CONTROL,
        );
        assert!(comp.handle_key(&key));
        assert!(!comp.is_dirty());
        assert_eq!(persists.borrow().len(), 1);
    }

    #[test]
    fn esc_cancels() {
        let cancelled = Rc::new(RefCell::new(0));
        let c = cancelled.clone();
        let mut comp = ScopedModelsSelectorComponent::new(
            models(),
            None,
            theme(),
            select_kb(),
            app_kb(),
            ScopedModelsCallbacks {
                on_change: Box::new(|_| {}),
                on_persist: Box::new(|_| {}),
                on_cancel: Box::new(move || *c.borrow_mut() += 1),
            },
        );
        let esc = KeyEvent::from(crossterm::event::KeyCode::Esc);
        assert!(comp.handle_key(&esc));
        assert_eq!(*cancelled.borrow(), 1);
    }

    #[test]
    fn navigation_wraps() {
        let mut comp = component(
            None,
            Rc::new(RefCell::new(vec![])),
            Rc::new(RefCell::new(vec![])),
        );
        let up = KeyEvent::from(crossterm::event::KeyCode::Up);
        let down = KeyEvent::from(crossterm::event::KeyCode::Down);
        assert!(comp.handle_key(&down));
        assert_eq!(comp.selected_index, 1);
        assert!(comp.handle_key(&up));
        assert_eq!(comp.selected_index, 0);
        assert!(comp.handle_key(&up));
        assert_eq!(comp.selected_index, 2, "Top wrap-around to last item");
    }

    #[test]
    fn render_does_not_panic_and_shows_header() {
        let mut comp = component(
            None,
            Rc::new(RefCell::new(vec![])),
            Rc::new(RefCell::new(vec![])),
        );
        let mut buf = ScreenBuffer::new(80, 20);
        let painted = comp.render(&mut buf, 0, 0, 80, 20);
        assert!(painted > 0);
        let mut out = Vec::new();
        buf.flush(&mut out).unwrap();
        let text = zen_core::ansi::strip_ansi(&String::from_utf8_lossy(&out));
        assert!(text.contains("Model Configuration"));
    }

    #[test]
    fn typing_filters_and_unknown_key_not_consumed() {
        let mut comp = component(
            None,
            Rc::new(RefCell::new(vec![])),
            Rc::new(RefCell::new(vec![])),
        );
        let f5 = KeyEvent::from(crossterm::event::KeyCode::F(5));
        assert!(!comp.handle_key(&f5), "Unbound keys are not consumed");
        for ch in "m-a".chars() {
            let k = KeyEvent::from(crossterm::event::KeyCode::Char(ch));
            comp.handle_key(&k);
        }
        assert_eq!(comp.search.get_value(), "m-a");
        assert_eq!(comp.filtered_indices.len(), 1);
    }
}
