// REQ-APP-11 (pi-358): thinking-selector — thinking level selection popup.
// Corresponds to the original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/thinking-selector.ts.
// Composes SelectList (T-TU-01-4) + DynamicBorder; keys (↑↓/Enter/Esc) come from the injected SelectAction-domain
// registry routing (T-AP-06-2 constructor injection; user overrides take effect through this).

use crate::components::dynamic_border::DynamicBorder;
use zen_ai::types::ThinkingLevel;
use zen_tui::components::select_list::{
    SelectAction, SelectItem, SelectList, SelectListLayoutOptions,
};
use zen_tui::components::Component;
use zen_tui::keybindings::KeybindingsManager;
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{ColorToken, Theme};

/// Thinking level → display description (aligned with pi LEVEL_DESCRIPTIONS, for the description column of the selection list).
fn level_description(level: ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Off => "No reasoning",
        ThinkingLevel::Minimal => "Very brief reasoning (~1k tokens)",
        ThinkingLevel::Low => "Light reasoning (~2k tokens)",
        ThinkingLevel::Medium => "Moderate reasoning (~8k tokens)",
        ThinkingLevel::High => "Deep reasoning (~16k tokens)",
        ThinkingLevel::XHigh => "Extra-high reasoning (~32k tokens)",
        ThinkingLevel::Max => "Maximum reasoning",
    }
}

/// Resolve a selection item value (lowercase level string, e.g. "low") back to ThinkingLevel; returns None for unknown values.
/// Consumer: the on_select callback converts the selected item back to ThinkingLevel.
pub fn thinking_level_from_str(s: &str) -> Option<ThinkingLevel> {
    match s {
        "off" => Some(ThinkingLevel::Off),
        "minimal" => Some(ThinkingLevel::Minimal),
        "low" => Some(ThinkingLevel::Low),
        "medium" => Some(ThinkingLevel::Medium),
        "high" => Some(ThinkingLevel::High),
        "xhigh" => Some(ThinkingLevel::XHigh),
        "max" => Some(ThinkingLevel::Max),
        _ => None,
    }
}

/// Thinking level selection popup (DynamicBorder + SelectList(available_levels, each item with description) + preselecting the current level) —
/// T-AP-11-1 shown on the /thinking command; Confirm invokes on_select(level), Cancel invokes on_cancel.
pub struct ThinkingSelectorComponent {
    select_list: Option<SelectList>,
    top_border: DynamicBorder,
    bottom_border: DynamicBorder,
}

impl ThinkingSelectorComponent {
    /// Constructs from the current level, available-levels list, theme, and SelectAction-domain keybinding map; pre-selects the current level entry.
    /// `select_kb` is passed through to the inner `SelectList` (T-AP-06-2 constructor injection: user overrides enter the select-action routing through this).
    pub fn new(
        current_level: ThinkingLevel,
        available_levels: Vec<ThinkingLevel>,
        theme: &Theme,
        select_kb: KeybindingsManager<SelectAction>,
        on_select: Box<dyn FnMut(ThinkingLevel)>,
        on_cancel: Box<dyn FnMut()>,
    ) -> Self {
        let items: Vec<SelectItem> = available_levels
            .iter()
            .map(|level| {
                SelectItem::new(
                    level_to_value(*level),
                    level_to_value(*level),
                    Some(level_description(*level).to_string()),
                )
            })
            .collect();

        let layout = SelectListLayoutOptions {
            min_primary_column_width: Some(12),
            max_primary_column_width: Some(32),
            truncate_primary: None,
        };

        let mut select_list = SelectList::new(
            items.clone(),
            items.len().max(1),
            theme.select_list_theme(),
            layout,
            select_kb,
        );

        // Preselect the current level
        let current_value = level_to_value(current_level);
        if let Some(idx) = items.iter().position(|i| i.value() == current_value) {
            select_list.set_selected_index(idx);
        }

        {
            let mut on_select = on_select;
            select_list.set_on_select(move |item| {
                if let Some(level) = thinking_level_from_str(item.value()) {
                    on_select(level);
                }
            });
            select_list.set_on_cancel(on_cancel);
        }

        ThinkingSelectorComponent {
            select_list: Some(select_list),
            top_border: DynamicBorder::new(Style::fg(theme.fg(ColorToken::Border))),
            bottom_border: DynamicBorder::new(Style::fg(theme.fg(ColorToken::Border))),
        }
    }
}

/// ThinkingLevel → selection item value string (lowercase, aligned with serde values).
fn level_to_value(level: ThinkingLevel) -> String {
    level.to_string().to_ascii_lowercase()
}

impl Component for ThinkingSelectorComponent {
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        let mut row = y;
        let mut painted = 0u16;
        let mut remaining = height;

        if let Some(list) = &mut self.select_list {
            painted += self.top_border.render(buf, x, row, width, remaining);
            row += painted;
            remaining = remaining.saturating_sub(painted);

            if remaining > 0 {
                let n = list.render(buf, x, row, width, remaining);
                painted += n;
                row += n;
                remaining = remaining.saturating_sub(n);
            }

            if remaining > 0 {
                painted += self.bottom_border.render(buf, x, row, width, remaining);
            }
        }
        painted
    }

    fn handle_key(&mut self, key: &crossterm::event::KeyEvent) -> bool {
        match self.select_list {
            Some(ref mut list) => {
                // SelectList's Component handle_key routes ↑↓/Enter/Esc/Ctrl+C inline → SelectAction
                list.handle_key(key)
            }
            None => false,
        }
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

    /// Test constructor: default SelectAction keymap (no user overrides).
    fn select_kb() -> KeybindingsManager<SelectAction> {
        KeybindingsManager::new(
            zen_tui::keybindings::select_default_bindings(),
            std::collections::HashMap::new(),
        )
    }

    #[test]
    fn from_str_roundtrip_all_levels() {
        for level in [
            ThinkingLevel::Off,
            ThinkingLevel::Minimal,
            ThinkingLevel::Low,
            ThinkingLevel::Medium,
            ThinkingLevel::High,
            ThinkingLevel::XHigh,
            ThinkingLevel::Max,
        ] {
            let s = level_to_value(level);
            assert_eq!(thinking_level_from_str(&s), Some(level), "value {s:?}");
        }
        assert_eq!(thinking_level_from_str("unknown"), None);
    }

    #[test]
    fn new_preselected_current_level() {
        let mut comp = ThinkingSelectorComponent::new(
            ThinkingLevel::High,
            vec![ThinkingLevel::Off, ThinkingLevel::Low, ThinkingLevel::High],
            theme(),
            select_kb(),
            Box::new(|_| {}),
            Box::new(|| {}),
        );
        let list = comp.select_list.as_mut().unwrap();
        // Preselecting the current level → the selected item is High (value "high")
        assert_eq!(list.get_selected_item().unwrap().value(), "high");
        assert_eq!(list.get_selected_item().unwrap().label(), "high");
        assert_eq!(
            list.get_selected_item().unwrap().description(),
            Some("Deep reasoning (~16k tokens)")
        );
    }

    #[test]
    fn render_lays_out_border_list_border() {
        let mut comp = ThinkingSelectorComponent::new(
            ThinkingLevel::Off,
            vec![ThinkingLevel::Off, ThinkingLevel::Low],
            theme(),
            select_kb(),
            Box::new(|_| {}),
            Box::new(|| {}),
        );
        let mut buf = ScreenBuffer::new(50, 10);
        let painted = comp.render(&mut buf, 0, 0, 50, 10);
        assert!(painted >= 2, "border + list + border");
        let mut out = Vec::new();
        buf.flush(&mut out).unwrap();
        let text = zen_core::ansi::strip_ansi(&String::from_utf8_lossy(&out));
        assert!(text.contains("low"));
    }

    #[test]
    fn handle_key_enter_fires_on_select() {
        let selected = Rc::new(RefCell::new(None));
        let s = selected.clone();
        let mut comp = ThinkingSelectorComponent::new(
            ThinkingLevel::Off,
            vec![ThinkingLevel::Low, ThinkingLevel::Medium],
            theme(),
            select_kb(),
            Box::new(move |level| *s.borrow_mut() = Some(level)),
            Box::new(|| {}),
        );
        // Select Medium (index 1) then Enter
        comp.select_list.as_mut().unwrap().set_selected_index(1);
        let enter = crossterm::event::KeyEvent::from(crossterm::event::KeyCode::Enter);
        assert!(comp.handle_key(&enter));
        assert_eq!(*selected.borrow(), Some(ThinkingLevel::Medium));
    }

    #[test]
    fn handle_key_esc_fires_on_cancel() {
        let cancelled = Rc::new(RefCell::new(0));
        let c = cancelled.clone();
        let mut comp = ThinkingSelectorComponent::new(
            ThinkingLevel::Off,
            vec![ThinkingLevel::Off],
            theme(),
            select_kb(),
            Box::new(|_| {}),
            Box::new(move || *c.borrow_mut() += 1),
        );
        let esc = crossterm::event::KeyEvent::from(crossterm::event::KeyCode::Esc);
        assert!(comp.handle_key(&esc));
        assert_eq!(*cancelled.borrow(), 1);
    }

    #[test]
    fn level_to_value_lowercases() {
        assert_eq!(level_to_value(ThinkingLevel::XHigh), "xhigh");
        assert_eq!(level_to_value(ThinkingLevel::Off), "off");
    }

    #[test]
    fn handle_key_ignores_unknown_key() {
        let mut comp = ThinkingSelectorComponent::new(
            ThinkingLevel::Off,
            vec![ThinkingLevel::Off],
            theme(),
            select_kb(),
            Box::new(|_| {}),
            Box::new(|| {}),
        );
        let a = crossterm::event::KeyEvent::from(crossterm::event::KeyCode::Char('a'));
        assert!(!comp.handle_key(&a));
    }
}
