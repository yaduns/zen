// REQ-V2-02 (pi-369, pi-368): model-selector — /model overlay (search + scope switching + selection) T-V2-02-4.
// Corresponds to original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/model-selector.ts
// And model-search.ts (`getModelSelectorSearchText` corpus).
//
// Responsibility: interactive-mode `/model` (or `app.model.select`) popup — type a few characters in a long model list to find the target,
// ↑↓ to select, Enter to switch to it. Scope can switch all/scoped (Tab, when a Ctrl+P cycle subset exists); current model has ✓.
// Navigation/confirm/cancel are routed through the injected `select_kb` (SelectAction domain); user overrides (T-AP-06-2) take effect here.
// No remote catalog refresh (REQ-V2-02 form (a) static catalog; `ModelRuntime.refresh`-related logic discarded).

use crossterm::event::KeyEvent;
use zen_ai::types::Model;
use zen_tui::components::input::Input;
use zen_tui::components::select_list::SelectAction;
use zen_tui::components::Component;
use zen_tui::keybindings::KeybindingsManager;
use zen_tui::keys::KeyId;
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{ColorToken, Theme};

use crate::components::dynamic_border::DynamicBorder;
use crate::components::scoped_models_selector::EnabledModelIds;

/// Model selection scope expanded by `/model`: all models, or the Ctrl+P cycle subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelScope {
    /// All models.
    All,
    /// scoped-models subset (initially used when a subset exists, aligned with pi).
    Scoped,
}

/// Fuzzy search corpus for the model selector (pi `getModelSelectorSearchText`): provider prefix is prepended,
/// so exact `provider/model` queries rank before a proxy provider's bare id. Private: consumed only in this file.
fn selector_search_text(model: &Model) -> String {
    let id = &model.id;
    let provider = &model.provider;
    let name = model
        .name
        .as_deref()
        .map(|n| format!(" {n}"))
        .unwrap_or_default();
    format!("{provider} {provider}/{id} {provider} {id}{name}")
}

/// Whether two models are the same (provider + id equal).
fn same_model(a: &Model, b: &Model) -> bool {
    a.provider == b.provider && a.id == b.id
}

/// Performs fuzzy filtering on the given candidate index set (empty query returns all in original order); returns candidate index list.
fn filter_indices(models: &[Model], active: &[usize], query: &str) -> Vec<usize> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return active.to_vec();
    }
    let mut scored: Vec<(usize, f64)> = active
        .iter()
        .filter_map(|&i| {
            zen_tui::fuzzy::fuzzy_match(trimmed, &selector_search_text(&models[i])).map(|s| (i, s))
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(i, _)| i).collect()
}

/// Model selection overlay (T-V2-02-4): search box + filtered list + all/scoped scope switching + current model checkmark.
pub struct ModelSelectorComponent {
    models: Vec<Model>,
    scoped_ids: EnabledModelIds,
    filtered_indices: Vec<usize>,
    selected_index: usize,
    current: Option<Model>,
    scope: ModelScope,
    search: Input,
    select_kb: KeybindingsManager<SelectAction>,
    focused: bool,
    max_visible: usize,
    on_select: Box<dyn FnMut(Model)>,
    on_cancel: Box<dyn FnMut()>,
    theme: Theme,
    top_border: DynamicBorder,
    bottom_border: DynamicBorder,
}

impl ModelSelectorComponent {
    /// Constructs the selector: given available models (T-V2-02-1 catalog output), scoped subset, current model, and callbacks;
    /// Preselects the current model; when `scoped_ids` is non-empty, initial scope is `Scoped`.
    /// # Parameters
    /// - `models`: all available models (empty displays "No models available" in the list).
    /// - `scoped_ids`: Ctrl+P cycle subset (`None`/empty = no scope switching, always `All`).
    /// - `current`: current session model (used for checkmark and preselection; no checkmark when `None`).
    /// - `theme`: color lookup; `select_kb`: SelectAction domain keybindings (user overrides take effect here).
    /// - `on_select`: Enter selection (carries the full `Model`; ui.rs converts it to `SelectorSelected("provider/id")`);
    ///   `on_cancel`: Esc/Ctrl+C cancels.
    /// - `initial_search`: initial search term passed by `/model <term>` (when no exact match); if Some non-empty, prefill the search box,
    ///   list filters immediately (aligned with pi `initialSearchInput`); None/empty gives an empty search box and preselects the current model.
    #[allow(clippy::too_many_arguments)] // Needed to complete the injection surface (data/theme/keybindings/fallback/callbacks)
    pub fn new(
        models: Vec<Model>,
        scoped_ids: EnabledModelIds,
        current: Option<Model>,
        theme: &Theme,
        select_kb: KeybindingsManager<SelectAction>,
        initial_search: Option<&str>,
        on_select: Box<dyn FnMut(Model)>,
        on_cancel: Box<dyn FnMut()>,
    ) -> Self {
        let has_scoped = scoped_ids
            .as_ref()
            .map(|ids| !ids.is_empty())
            .unwrap_or(false);
        let scope = if has_scoped {
            ModelScope::Scoped
        } else {
            ModelScope::All
        };
        let mut search = Input::new();
        search.set_focused(true);
        let trimmed_initial = initial_search.map(str::trim).filter(|s| !s.is_empty());
        if let Some(term) = trimmed_initial {
            search.set_value(term);
        }
        let mut comp = ModelSelectorComponent {
            models,
            scoped_ids,
            filtered_indices: Vec::new(),
            selected_index: 0,
            current,
            scope,
            search,
            select_kb,
            focused: true,
            max_visible: 10,
            on_select,
            on_cancel,
            theme: theme.clone(),
            top_border: DynamicBorder::new(Style::fg(theme.fg(ColorToken::BorderAccent))),
            bottom_border: DynamicBorder::new(Style::fg(theme.fg(ColorToken::BorderAccent))),
        };
        comp.rebuild();
        if trimmed_initial.is_some() {
            // Initial search: selection lands on first filtered result (does not preselect current model; aligned with pi filterModels behavior)
            comp.clamp_selected();
        } else {
            comp.preselect_current();
        }
        comp
    }

    /// Replace model set and subset with new snapshot (re-entry after T-V2-02-1 catalog refresh; recompute filter and clamp selection).
    pub fn set_models(&mut self, models: Vec<Model>, scoped_ids: EnabledModelIds) {
        self.models = models;
        self.scoped_ids = scoped_ids;
        if self.scoped_ids.as_ref().is_none_or(|ids| ids.is_empty()) {
            self.scope = ModelScope::All;
        }
        self.rebuild();
        self.preselect_current();
    }

    /// Current scope (`All`/`Scoped`); for tests and UI assertions.
    pub fn scope(&self) -> ModelScope {
        self.scope
    }

    /// Toggle scope (Tab): no-op when scoped subset is empty. Consumer: component `handle_key`.
    pub fn set_scope(&mut self, scope: ModelScope) {
        if scope == ModelScope::Scoped && self.scoped_ids.as_ref().is_none_or(|ids| ids.is_empty())
        {
            return;
        }
        self.scope = scope;
        self.rebuild();
        self.preselect_current();
    }

    /// Set focus (passed through to inner search box for IME cursor positioning). Consumer: `InteractiveUi`.
    pub fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
        self.search.set_focused(focused);
    }

    /// Current text in inner search box. Consumer: tests and debug assertions.
    pub fn search_value(&self) -> &str {
        self.search.get_value()
    }

    /// Currently selected model (`None` when filtering yields no items). Consumer: `InteractiveUi` Enter handling and tests.
    pub fn selected_model(&self) -> Option<&Model> {
        let idx = *self.filtered_indices.get(self.selected_index)?;
        self.models.get(idx)
    }

    /// All candidate indices in scope (`All` = full set; `Scoped` = subset in `scoped_ids` order).
    fn active_indices(&self) -> Vec<usize> {
        match (self.scope, &self.scoped_ids) {
            (ModelScope::Scoped, Some(ids)) => ids
                .iter()
                .filter_map(|id| {
                    let (provider, mid) = id.split_once('/')?;
                    self.models
                        .iter()
                        .position(|m| m.provider == provider && m.id == mid)
                })
                .collect(),
            _ => (0..self.models.len()).collect(),
        }
    }

    /// Recompute filtered view and clamp selection.
    fn rebuild(&mut self) {
        let active = self.active_indices();
        let query = self.search.get_value().to_string();
        self.filtered_indices = filter_indices(&self.models, &active, &query);
        self.clamp_selected();
    }

    fn clamp_selected(&mut self) {
        let len = self.filtered_indices.len();
        self.selected_index = if len == 0 {
            0
        } else {
            self.selected_index.min(len - 1)
        };
    }

    /// Preselect current model (if in current scope); otherwise clamp to start of scope.
    fn preselect_current(&mut self) {
        if let Some(current) = &self.current {
            if let Some(pos) = self
                .filtered_indices
                .iter()
                .position(|&i| same_model(&self.models[i], current))
            {
                self.selected_index = pos;
                return;
            }
        }
        self.clamp_selected();
    }

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

    /// Scope hint text; without a subset, shows "only configured providers"; with a subset, hints Tab to switch.
    fn scope_banner(&self) -> String {
        if self.scoped_ids.as_ref().is_none_or(|ids| ids.is_empty()) {
            "Only showing models from configured providers.".to_string()
        } else {
            let active = match self.scope {
                ModelScope::All => "all",
                ModelScope::Scoped => "scoped",
            };
            format!("Scope: all | scoped  (active: {active}; Tab to switch)")
        }
    }
}

impl Component for ModelSelectorComponent {
    /// Render overlay: border + scope hint + search box + filtered list (current model ✓) + selected item name.
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        if width == 0 || height == 0 {
            return 0;
        }
        let muted = Style::fg(self.theme.fg(ColorToken::Muted));
        let dim = Style::fg(self.theme.fg(ColorToken::Dim));
        let success = Style::fg(self.theme.fg(ColorToken::Success));
        let accent = Style::fg(self.theme.fg(ColorToken::Accent));
        let line_style = Style::default();

        let mut row = y;
        let mut remaining = height;
        let mut painted = 0u16;

        if remaining > 0 {
            let n = self.top_border.render(buf, x, row, width, remaining);
            painted += n;
            row += n;
            remaining = remaining.saturating_sub(n);
        }
        if remaining > 0 {
            buf.fill(x, row, width, 1, &line_style);
            buf.write(x + 1, row, &muted, &self.scope_banner());
            painted += 1;
            row += 1;
            remaining -= 1;
        }
        if remaining > 0 {
            buf.fill(x, row, width, 1, &line_style);
            painted += 1;
            row += 1;
            remaining -= 1;
        }
        if remaining > 0 {
            let n = self.search.render(buf, x, row, width, remaining);
            painted += n;
            row += n;
            remaining = remaining.saturating_sub(n);
        }
        if remaining > 0 {
            buf.fill(x, row, width, 1, &line_style);
            painted += 1;
            row += 1;
            remaining -= 1;
        }

        // List
        if remaining > 0 && !self.models.is_empty() {
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
                for i in start..end {
                    if remaining == 0 {
                        break;
                    }
                    let model = &self.models[self.filtered_indices[i]];
                    let selected = i == self.selected_index;
                    let is_current = self.current.as_ref().is_some_and(|c| same_model(c, model));
                    let prefix = if selected { "→ " } else { "  " };
                    let mut text = format!("{prefix}{}", model.id);
                    text.push_str(&format!(" [{}]", model.provider));
                    if is_current {
                        text.push_str(" ✓");
                    }
                    buf.fill(x, row, width, 1, &line_style);
                    let style = if selected {
                        accent.clone()
                    } else {
                        Style::default()
                    };
                    buf.write(x + 1, row, &style, &text);
                    if is_current {
                        // Override the trailing "✓" with the success semantic color
                        let text_len = zen_tui::screen::visible_width(&text) as u16;
                        let end_col = x + 1 + text_len;
                        if end_col > x + 1 {
                            buf.write(end_col - 1, row, &success, "✓");
                        }
                    }
                    painted += 1;
                    row += 1;
                    remaining -= 1;
                }
                if remaining > 0 && (start > 0 || end < total) {
                    let info = format!("  ({}/{})", self.selected_index + 1, total);
                    buf.fill(x, row, width, 1, &line_style);
                    buf.write(x + 1, row, &muted, &info);
                    painted += 1;
                    row += 1;
                    remaining -= 1;
                }
            }
        } else if remaining > 0 {
            buf.fill(x, row, width, 1, &line_style);
            buf.write(x + 2, row, &muted, "No models available");
            painted += 1;
            row += 1;
            remaining -= 1;
        }

        // Selected item name
        if remaining > 0 {
            if let Some(model) = self.selected_model() {
                let name = model.name.clone().unwrap_or_else(|| model.id.clone());
                buf.fill(x, row, width, 1, &line_style);
                buf.write(x + 1, row, &muted, &format!("  Model Name: {name}"));
                painted += 1;
                row += 1;
                remaining -= 1;
            }
        }
        if remaining > 0 {
            buf.fill(x, row, width, 1, &line_style);
            buf.write(x + 1, row, &dim, "  Enter select · Esc cancel");
            painted += 1;
            row += 1;
            remaining -= 1;
        }
        if remaining > 0 {
            let n = self.bottom_border.render(buf, x, row, width, remaining);
            painted += n;
        }
        painted
    }

    /// Key routing: Tab = toggle scope; SelectAction::Up/Down/Confirm/Cancel; other characters go to search box. Returns whether consumed.
    fn handle_key(&mut self, key: &KeyEvent) -> bool {
        let Some(kid) = KeyId::from_key_event(*key) else {
            return false;
        };

        // Tab: toggle scope (pi uses tui.input.tab; here match Tab key directly to avoid app-domain coupling)
        if key.code == crossterm::event::KeyCode::Tab {
            let next = if self.scope == ModelScope::All {
                ModelScope::Scoped
            } else {
                ModelScope::All
            };
            self.set_scope(next);
            return true;
        }
        if self.select_kb.matches(&kid, SelectAction::MoveUp) {
            self.move_selection(false);
            return true;
        }
        if self.select_kb.matches(&kid, SelectAction::MoveDown) {
            self.move_selection(true);
            return true;
        }
        if self.select_kb.matches(&kid, SelectAction::Confirm) {
            if let Some(model) = self.selected_model().cloned() {
                (self.on_select)(model);
            }
            return true;
        }
        if self.select_kb.matches(&kid, SelectAction::Cancel) {
            (self.on_cancel)();
            return true;
        }

        if self.search.handle_key(key) {
            self.rebuild();
            return true;
        }
        false
    }
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

    #[allow(clippy::type_complexity)] // Shared unit returned by select/cancel in tests; type is complex but locally readable
    fn component(
        scoped: EnabledModelIds,
        current: Option<Model>,
    ) -> (
        ModelSelectorComponent,
        Rc<RefCell<Option<Model>>>,
        Rc<RefCell<usize>>,
    ) {
        let picked = Rc::new(RefCell::new(None));
        let cancelled = Rc::new(RefCell::new(0));
        let p = picked.clone();
        let c = cancelled.clone();
        let comp = ModelSelectorComponent::new(
            models(),
            scoped,
            current,
            theme(),
            select_kb(),
            None,
            Box::new(move |m| *p.borrow_mut() = Some(m)),
            Box::new(move || *c.borrow_mut() += 1),
        );
        (comp, picked, cancelled)
    }

    #[test]
    fn search_text_provider_first() {
        let m = Model::from_env().with_id("m-x");
        let t = selector_search_text(&m);
        assert!(
            t.starts_with(&m.provider),
            "provider prefix should come first: {t}"
        );
        assert!(t.contains(&format!("{}/m-x", m.provider)));
    }

    #[test]
    fn default_scope_all_without_scoped() {
        let (comp, _, _) = component(None, None);
        assert_eq!(comp.scope(), ModelScope::All);
    }

    #[test]
    fn default_scope_scoped_with_scoped() {
        let (comp, _, _) = component(Some(ids(&["opencode-go/m-b"])), None);
        assert_eq!(comp.scope(), ModelScope::Scoped);
    }

    #[test]
    fn preselects_current_model() {
        let current = models().into_iter().nth(1);
        let (comp, _, _) = component(None, current);
        assert_eq!(
            comp.selected_model().map(|m| m.id.clone()),
            Some("m-b".to_string())
        );
    }

    #[test]
    fn enter_selects_model_and_esc_cancels() {
        let (mut comp, picked, cancelled) = component(None, None);
        let enter = KeyEvent::from(crossterm::event::KeyCode::Enter);
        assert!(comp.handle_key(&enter));
        assert!(picked.borrow().is_some());
        let esc = KeyEvent::from(crossterm::event::KeyCode::Esc);
        assert!(comp.handle_key(&esc));
        assert_eq!(*cancelled.borrow(), 1);
    }

    #[test]
    fn tab_toggles_scope_and_noop_without_scoped() {
        let (mut comp, _, _) = component(None, None);
        let tab = KeyEvent::from(crossterm::event::KeyCode::Tab);
        assert!(comp.handle_key(&tab));
        assert_eq!(
            comp.scope(),
            ModelScope::All,
            "No change when there is no subset"
        );

        let (mut comp2, _, _) = component(Some(ids(&["opencode-go/m-b"])), None);
        assert_eq!(comp2.scope(), ModelScope::Scoped);
        comp2.handle_key(&tab);
        assert_eq!(comp2.scope(), ModelScope::All);
    }

    #[test]
    fn navigation_wraps() {
        let (mut comp, _, _) = component(None, None);
        let down = KeyEvent::from(crossterm::event::KeyCode::Down);
        let up = KeyEvent::from(crossterm::event::KeyCode::Up);
        comp.handle_key(&down);
        assert_eq!(
            comp.selected_model().map(|m| m.id.clone()),
            Some("m-b".into())
        );
        comp.handle_key(&up);
        comp.handle_key(&up);
        assert_eq!(
            comp.selected_model().map(|m| m.id.clone()),
            Some("m-c".into()),
            "Top wrap-around"
        );
    }

    #[test]
    fn typing_filters_and_shows_no_match() {
        let (mut comp, _, _) = component(None, None);
        for ch in "zzz".chars() {
            comp.handle_key(&KeyEvent::from(crossterm::event::KeyCode::Char(ch)));
        }
        assert_eq!(comp.filtered_indices.len(), 0);
        assert!(comp.selected_model().is_none());
    }

    #[test]
    fn set_models_recomputes_and_clears_scope_when_scoped_gone() {
        let (mut comp, _, _) = component(Some(ids(&["opencode-go/m-b"])), None);
        assert_eq!(comp.scope(), ModelScope::Scoped);
        comp.set_models(models(), None);
        assert_eq!(comp.scope(), ModelScope::All);
    }

    #[test]
    fn render_shows_list_and_selected_name() {
        let (mut comp, _, _) = component(None, None);
        let mut buf = ScreenBuffer::new(80, 20);
        let painted = comp.render(&mut buf, 0, 0, 80, 20);
        assert!(painted > 0);
        let mut out = Vec::new();
        buf.flush(&mut out).unwrap();
        let text = zen_core::ansi::strip_ansi(&String::from_utf8_lossy(&out));
        assert!(text.contains("m-a"));
        assert!(text.contains("Model Name:"));
    }

    #[test]
    fn initial_search_prefills_and_filters() {
        let picked = Rc::new(RefCell::new(None));
        let p = picked.clone();
        let comp = ModelSelectorComponent::new(
            models(),
            None,
            None,
            theme(),
            select_kb(),
            Some("m-b"),
            Box::new(move |m| *p.borrow_mut() = Some(m)),
            Box::new(|| {}),
        );
        assert_eq!(comp.search_value(), "m-b");
        assert_eq!(comp.filtered_indices.len(), 1);
        assert_eq!(
            comp.selected_model().map(|m| m.id.clone()),
            Some("m-b".into())
        );
    }

    #[test]
    fn initial_search_blank_is_ignored() {
        let comp = ModelSelectorComponent::new(
            models(),
            None,
            None,
            theme(),
            select_kb(),
            Some("   "),
            Box::new(|_| {}),
            Box::new(|| {}),
        );
        assert_eq!(comp.search_value(), "");
        assert_eq!(comp.filtered_indices.len(), 3);
    }

    #[test]
    fn render_empty_models_shows_message() {
        let mut comp = ModelSelectorComponent::new(
            Vec::new(),
            None,
            None,
            theme(),
            select_kb(),
            None,
            Box::new(|_| {}),
            Box::new(|| {}),
        );
        let mut buf = ScreenBuffer::new(40, 12);
        comp.render(&mut buf, 0, 0, 40, 12);
        let mut out = Vec::new();
        buf.flush(&mut out).unwrap();
        let text = zen_core::ansi::strip_ansi(&String::from_utf8_lossy(&out));
        assert!(text.contains("No models available"));
    }
}
