// REQ-APP-11 (pi-358): settings-selector — settings selection popup.
// Corresponds to original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/settings-selector.ts.
// v1 settings items are trimmed per the stub's "pi field resolution table" (transport/httpIdleTimeout/images/telemetry/trust/
// changelog/warnings/terminal-progress are not included). Submenus (thinking/theme) are driven via SettingsList's
// SubmenuFactory + explicit done_submenu (the anti-self-handshake design of T-TU-01-4): the submenu's selected value is first written to
// the shared cell, and SettingsSelectorComponent drives done_submenu after handle_key returns (the value is updated inside the list
// and on_change is triggered).

use crate::components::dynamic_border::DynamicBorder;
use crate::components::thinking_selector::thinking_level_from_str;
use crate::components::tree_selector::{filter_mode_from_str, filter_mode_to_str, FilterMode};
use crossterm::event::KeyEvent;
use std::cell::RefCell;
use std::rc::Rc;
use zen_ai::types::ThinkingLevel;
use zen_tui::components::select_list::SelectListTheme;
use zen_tui::components::select_list::{
    SelectAction, SelectItem, SelectList, SelectListLayoutOptions,
};
use zen_tui::components::settings_list::{
    SettingItem, SettingsList, SettingsListOptions, SettingsListTheme, ValueChange,
};
use zen_tui::components::Component;
use zen_tui::keybindings::KeybindingsManager;
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{ColorToken, Theme};

/// Message delivery mode (pi "all" | "one-at-a-time"; discriminator for settings item values).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryMode {
    /// Deliver all queued messages at once.
    All,
    /// Deliver one at a time, waiting for the response.
    OneAtATime,
}

impl DeliveryMode {
    fn from_str(s: &str) -> DeliveryMode {
        if s == "all" {
            DeliveryMode::All
        } else {
            DeliveryMode::OneAtATime
        }
    }
    fn to_str(self) -> &'static str {
        match self {
            DeliveryMode::All => "all",
            DeliveryMode::OneAtATime => "one-at-a-time",
        }
    }
}

/// Double-Esc action (pi "fork" | "tree" | "none"). Pure discriminating enum, consumed by T-AP-11-1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoubleEscapeAction {
    /// Double-Esc opens the session tree.
    Tree,
    /// Double-Esc forks the session.
    Fork,
    /// No action.
    None,
}

impl DoubleEscapeAction {
    fn from_str(s: &str) -> DoubleEscapeAction {
        match s {
            "tree" => DoubleEscapeAction::Tree,
            "fork" => DoubleEscapeAction::Fork,
            _ => DoubleEscapeAction::None,
        }
    }
    fn to_str(self) -> &'static str {
        match self {
            DoubleEscapeAction::Tree => "tree",
            DoubleEscapeAction::Fork => "fork",
            DoubleEscapeAction::None => "none",
        }
    }
}

/// Input snapshot for the settings selector — "all editable items when entering the settings page"; the component only reads it to generate the list.
/// Pure data, not serialized (persistence is handled by the T-AP-02-3 settings manager; this type is its display projection).
/// Debug/Clone: required by UiEvent (OpenSettings payload) (T-AP-02-3-fix).
#[derive(Debug, Clone)]
pub struct SettingsConfig {
    /// Auto-compaction switch (footer `(auto)` marker).
    pub auto_compact: bool,
    /// Message delivery mode (AgentSession::steer semantics).
    pub steering_mode: DeliveryMode,
    /// Message delivery mode (AgentSession::follow_up semantics).
    pub follow_up_mode: DeliveryMode,
    /// Skill command toggle (zen-core skills).
    pub enable_skill_commands: bool,
    /// Hardware cursor vs inverted block cursor (editor rendering).
    pub show_hardware_cursor: bool,
    /// Input editor left padding (0-3).
    pub editor_padding_x: usize,
    /// Message output left padding (0 or 1).
    pub output_pad: u16,
    /// Number of visible autocomplete entries.
    pub autocomplete_max_visible: usize,
    /// Clear screen when the terminal shrinks.
    pub clear_on_shrink: bool,
    /// Collapse thinking in assistant messages.
    pub hide_thinking_block: bool,
    /// Cache-miss summary hint.
    pub show_cache_miss_notices: bool,
    /// Enable quiet mode on startup (T-AP-08-6).
    pub quiet_startup: bool,
    /// Double-Esc action.
    pub double_escape_action: DoubleEscapeAction,
    /// Default tree filter mode.
    pub tree_filter_mode: FilterMode,
    /// Current thinking level.
    pub thinking_level: ThinkingLevel,
    /// Available thinking levels.
    pub available_thinking_levels: Vec<ThinkingLevel>,
    /// Current theme name.
    pub current_theme: String,
    /// Available theme names.
    pub available_themes: Vec<String>,
}

/// Change channels mirroring `SettingsConfig` one-to-one (one `Box<dyn FnMut>` per item), plus `on_cancel`.
/// `Default` provides no-op handlers, so constructors only need to override the items of interest.
pub struct SettingsCallbacks {
    pub on_auto_compact_change: Box<dyn FnMut(bool)>,
    pub on_steering_mode_change: Box<dyn FnMut(DeliveryMode)>,
    pub on_follow_up_mode_change: Box<dyn FnMut(DeliveryMode)>,
    pub on_enable_skill_commands_change: Box<dyn FnMut(bool)>,
    pub on_show_hardware_cursor_change: Box<dyn FnMut(bool)>,
    pub on_editor_padding_x_change: Box<dyn FnMut(usize)>,
    pub on_output_pad_change: Box<dyn FnMut(u16)>,
    pub on_autocomplete_max_visible_change: Box<dyn FnMut(usize)>,
    pub on_clear_on_shrink_change: Box<dyn FnMut(bool)>,
    pub on_hide_thinking_block_change: Box<dyn FnMut(bool)>,
    pub on_show_cache_miss_notices_change: Box<dyn FnMut(bool)>,
    pub on_quiet_startup_change: Box<dyn FnMut(bool)>,
    pub on_double_escape_action_change: Box<dyn FnMut(DoubleEscapeAction)>,
    pub on_tree_filter_mode_change: Box<dyn FnMut(FilterMode)>,
    pub on_thinking_level_change: Box<dyn FnMut(ThinkingLevel)>,
    pub on_theme_change: Box<dyn FnMut(String)>,
    pub on_cancel: Box<dyn FnMut()>,
}

impl Default for SettingsCallbacks {
    /// Full set of no-op handlers (constructor overrides each callback of interest).
    fn default() -> Self {
        fn noop<T: 'static>() -> Box<dyn FnMut(T)> {
            Box::new(|_| {})
        }
        SettingsCallbacks {
            on_auto_compact_change: noop(),
            on_steering_mode_change: noop(),
            on_follow_up_mode_change: noop(),
            on_enable_skill_commands_change: noop(),
            on_show_hardware_cursor_change: noop(),
            on_editor_padding_x_change: noop(),
            on_output_pad_change: noop(),
            on_autocomplete_max_visible_change: noop(),
            on_clear_on_shrink_change: noop(),
            on_hide_thinking_block_change: noop(),
            on_show_cache_miss_notices_change: noop(),
            on_quiet_startup_change: noop(),
            on_double_escape_action_change: noop(),
            on_tree_filter_mode_change: noop(),
            on_thinking_level_change: noop(),
            on_theme_change: noop(),
            on_cancel: Box::new(|| {}),
        }
    }
}

/// Submenu close signal (shared cell): None = close not requested; Some(None) = canceled; Some(Some(v)) = selected value.
type SubmenuClose = Rc<RefCell<Option<Option<String>>>>;

/// Build a submenu (title + optional description + SelectList); on select/cancel, write to the close signal.
/// parameters align with the pi make_submenu family + keybinding map injection (T-AP-06-2); the extra parameters are needed for assembly-surface completeness.
#[allow(clippy::too_many_arguments)]
fn make_submenu(
    title: String,
    description: Option<String>,
    items: Vec<SelectItem>,
    current_value: &str,
    list_theme: SelectListTheme,
    accent_style: Style,
    select_kb: KeybindingsManager<SelectAction>,
    close: SubmenuClose,
) -> Box<dyn Component> {
    let layout = SelectListLayoutOptions {
        min_primary_column_width: Some(12),
        max_primary_column_width: Some(32),
        truncate_primary: None,
    };
    let mut list = SelectList::new(
        items.clone(),
        items.len().max(1),
        list_theme,
        layout,
        select_kb,
    );
    if let Some(idx) = items.iter().position(|i| i.value() == current_value) {
        list.set_selected_index(idx);
    }
    {
        let close = close.clone();
        list.set_on_select(move |item| {
            *close.borrow_mut() = Some(Some(item.value().to_string()));
        });
    }
    {
        let close = close.clone();
        list.set_on_cancel(move || {
            *close.borrow_mut() = Some(None);
        });
    }
    Box::new(SelectSubmenu {
        title,
        description,
        list,
        theme: accent_style,
    })
}

/// Assembled in make_submenu (title line + blank line + list).
struct SelectSubmenu {
    title: String,
    description: Option<String>,
    list: SelectList,
    theme: Style,
}

impl Component for SelectSubmenu {
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        let mut row = y;
        let mut painted = 0u16;
        let mut remaining = height;

        if remaining > 0 {
            buf.fill(x, row, width, 1, &Style::default());
            buf.write(x + 1, row, &self.theme, &self.title);
            painted += 1;
            row += 1;
            remaining -= 1;
        }
        if let Some(desc) = &self.description {
            if remaining > 0 {
                buf.fill(x, row, width, 1, &Style::default());
                buf.write(
                    x + 1,
                    row,
                    &Style::fg(crossterm::style::Color::DarkGrey),
                    desc,
                );
                painted += 1;
                row += 1;
                remaining -= 1;
            }
            if remaining > 0 {
                buf.fill(x, row, width, 1, &Style::default());
                painted += 1;
                row += 1;
                remaining -= 1;
            }
        }
        if remaining > 0 {
            let n = self.list.render(buf, x, row, width, remaining);
            painted += n;
        }
        painted
    }

    fn handle_key(&mut self, key: &KeyEvent) -> bool {
        self.list.handle_key(key)
    }
}

/// Thinking level submenu entry (value = lowercase level + description).
fn thinking_items(levels: &[ThinkingLevel]) -> Vec<SelectItem> {
    levels
        .iter()
        .map(|level| {
            let value = level.to_string().to_ascii_lowercase();
            SelectItem::new(
                value.clone(),
                value,
                Some(thinking_description(*level).to_string()),
            )
        })
        .collect()
}

fn thinking_description(level: ThinkingLevel) -> &'static str {
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

/// Settings selection popup: SettingsList lays out all settings (values cycle; thinking/theme open submenus)—
/// Shared by T-AP-08-6 (startup config) and T-AP-11-1 (/settings); driven by config snapshot + per-item callbacks,
/// component does not hold the settings store.
pub struct SettingsSelectorComponent {
    list: SettingsList,
    close: SubmenuClose,
    top_border: DynamicBorder,
    bottom_border: DynamicBorder,
}

impl SettingsSelectorComponent {
    /// Constructs from a config snapshot, callback bundle, theme, settings-list theme, and SelectAction-domain keybinding map; list items and
    /// the on_change mapping are built inside the constructor. `select_kb` is passed through to `SettingsList` and the submenu `SelectList`
    /// (T-AP-06-2 constructor injection: user overrides enter the select-action routing through this).
    #[allow(clippy::vec_init_then_push)] // Settings items built one by one in pi order, readability first
    pub fn new(
        config: SettingsConfig,
        callbacks: SettingsCallbacks,
        theme: &Theme,
        settings_theme: SettingsListTheme,
        select_kb: KeybindingsManager<SelectAction>,
    ) -> Self {
        let close: SubmenuClose = Rc::new(RefCell::new(None));
        let list_theme = theme.select_list_theme();
        let accent_style = Style::fg(theme.fg(ColorToken::Accent));
        let mut callbacks = callbacks;
        let on_cancel = std::mem::replace(&mut callbacks.on_cancel, Box::new(|| {}));

        // ── Build settings items (values loop / submenu factory) ──
        let mut items: Vec<SettingItem> = Vec::new();

        items.push(SettingItem {
            id: "autocompact".into(),
            label: "Auto-compact".into(),
            description: Some("Automatically compact context when it gets too large".into()),
            current_value: bool_str(config.auto_compact),
            values: Some(bool_values()),
            submenu: None,
        });
        items.push(SettingItem {
            id: "steering-mode".into(),
            label: "Steering mode".into(),
            description: Some(
                "Enter while streaming queues steering messages. 'one-at-a-time': deliver one, wait for response. 'all': deliver all at once."
                    .into(),
            ),
            current_value: config.steering_mode.to_str().into(),
            values: Some(vec!["one-at-a-time".into(), "all".into()]),
            submenu: None,
        });
        items.push(SettingItem {
            id: "follow-up-mode".into(),
            label: "Follow-up mode".into(),
            description: Some(
                "After-enter queues follow-up messages until agent stops. 'one-at-a-time': deliver one, wait for response. 'all': deliver all at once."
                    .into(),
            ),
            current_value: config.follow_up_mode.to_str().into(),
            values: Some(vec!["one-at-a-time".into(), "all".into()]),
            submenu: None,
        });
        items.push(SettingItem {
            id: "skill-commands".into(),
            label: "Skill commands".into(),
            description: Some("Register skills as /skill:name commands".into()),
            current_value: bool_str(config.enable_skill_commands),
            values: Some(bool_values()),
            submenu: None,
        });
        items.push(SettingItem {
            id: "show-hardware-cursor".into(),
            label: "Show hardware cursor".into(),
            description: Some(
                "Show the terminal cursor while still positioning it for IME support".into(),
            ),
            current_value: bool_str(config.show_hardware_cursor),
            values: Some(bool_values()),
            submenu: None,
        });
        items.push(SettingItem {
            id: "editor-padding".into(),
            label: "Editor padding".into(),
            description: Some("Horizontal padding for input editor (0-3)".into()),
            current_value: config.editor_padding_x.to_string(),
            values: Some(vec!["0".into(), "1".into(), "2".into(), "3".into()]),
            submenu: None,
        });
        items.push(SettingItem {
            id: "output-padding".into(),
            label: "Output padding".into(),
            description: Some(
                "Horizontal padding for user messages, assistant messages, and thinking".into(),
            ),
            current_value: config.output_pad.to_string(),
            values: Some(vec!["0".into(), "1".into()]),
            submenu: None,
        });
        items.push(SettingItem {
            id: "autocomplete-max-visible".into(),
            label: "Autocomplete max items".into(),
            description: Some("Max visible items in autocomplete dropdown (3-20)".into()),
            current_value: config.autocomplete_max_visible.to_string(),
            values: Some(vec![
                "3".into(),
                "5".into(),
                "7".into(),
                "10".into(),
                "15".into(),
                "20".into(),
            ]),
            submenu: None,
        });
        items.push(SettingItem {
            id: "clear-on-shrink".into(),
            label: "Clear on shrink".into(),
            description: Some("Clear empty rows when content shrinks (may cause flicker)".into()),
            current_value: bool_str(config.clear_on_shrink),
            values: Some(bool_values()),
            submenu: None,
        });
        items.push(SettingItem {
            id: "hide-thinking".into(),
            label: "Hide thinking".into(),
            description: Some("Hide thinking blocks in assistant responses".into()),
            current_value: bool_str(config.hide_thinking_block),
            values: Some(bool_values()),
            submenu: None,
        });
        items.push(SettingItem {
            id: "cache-miss-notices".into(),
            label: "Cache miss notices".into(),
            description: Some("Show transcript notices for significant prompt-cache misses".into()),
            current_value: bool_str(config.show_cache_miss_notices),
            values: Some(bool_values()),
            submenu: None,
        });
        items.push(SettingItem {
            id: "quiet-startup".into(),
            label: "Quiet startup".into(),
            description: Some("Disable verbose printing at startup".into()),
            current_value: bool_str(config.quiet_startup),
            values: Some(bool_values()),
            submenu: None,
        });
        items.push(SettingItem {
            id: "double-escape-action".into(),
            label: "Double-escape action".into(),
            description: Some("Action when pressing Escape twice with empty editor".into()),
            current_value: config.double_escape_action.to_str().into(),
            values: Some(vec!["tree".into(), "fork".into(), "none".into()]),
            submenu: None,
        });
        items.push(SettingItem {
            id: "tree-filter-mode".into(),
            label: "Tree filter mode".into(),
            description: Some("Default filter when opening /tree".into()),
            current_value: filter_mode_to_str(config.tree_filter_mode).into(),
            values: Some(vec![
                "default".into(),
                "no-tools".into(),
                "user-only".into(),
                "labeled-only".into(),
                "all".into(),
            ]),
            submenu: None,
        });

        // thinking submenu
        {
            let close = close.clone();
            let lt = list_theme.clone();
            let accent = accent_style.clone();
            let skb = select_kb.clone();
            let current = config.thinking_level.to_string().to_ascii_lowercase();
            let levels = config.available_thinking_levels.clone();
            items.push(SettingItem {
                id: "thinking".into(),
                label: "Thinking level".into(),
                description: Some("Reasoning depth for thinking-capable models".into()),
                current_value: current.clone(),
                values: None,
                submenu: Some(Box::new(move |_cv: &str| {
                    make_submenu(
                        "Thinking Level".into(),
                        Some("Select reasoning depth for thinking-capable models".into()),
                        thinking_items(&levels),
                        &current,
                        lt.clone(),
                        accent.clone(),
                        skb.clone(),
                        close.clone(),
                    )
                })),
            });
        }
        // theme submenu
        {
            let lt = list_theme.clone();
            let accent = accent_style.clone();
            let skb = select_kb.clone();
            let themes = config.available_themes.clone();
            let current_theme = config.current_theme.clone();
            let close = close.clone();
            items.push(SettingItem {
                id: "theme".into(),
                label: "Theme".into(),
                description: Some("Color theme for the interface".into()),
                current_value: current_theme.clone(),
                values: None,
                submenu: Some(Box::new(move |_cv: &str| {
                    let items: Vec<SelectItem> = themes
                        .iter()
                        .map(|t| SelectItem::new(t.clone(), t.clone(), None))
                        .collect();
                    make_submenu(
                        "Theme".into(),
                        None,
                        items,
                        &current_theme,
                        lt.clone(),
                        accent.clone(),
                        skb.clone(),
                        close.clone(),
                    )
                })),
            });
        }

        let on_change: ValueChange = {
            let mut callbacks = callbacks;
            Box::new(move |id: &str, value: &str| match id {
                "autocompact" => (callbacks.on_auto_compact_change)(value == "true"),
                "steering-mode" => {
                    (callbacks.on_steering_mode_change)(DeliveryMode::from_str(value))
                }
                "follow-up-mode" => {
                    (callbacks.on_follow_up_mode_change)(DeliveryMode::from_str(value))
                }
                "skill-commands" => (callbacks.on_enable_skill_commands_change)(value == "true"),
                "show-hardware-cursor" => {
                    (callbacks.on_show_hardware_cursor_change)(value == "true")
                }
                "editor-padding" => {
                    (callbacks.on_editor_padding_x_change)(value.parse().unwrap_or(0))
                }
                "output-padding" => (callbacks.on_output_pad_change)(value.parse().unwrap_or(0)),
                "autocomplete-max-visible" => {
                    (callbacks.on_autocomplete_max_visible_change)(value.parse().unwrap_or(5))
                }
                "clear-on-shrink" => (callbacks.on_clear_on_shrink_change)(value == "true"),
                "hide-thinking" => (callbacks.on_hide_thinking_block_change)(value == "true"),
                "cache-miss-notices" => {
                    (callbacks.on_show_cache_miss_notices_change)(value == "true")
                }
                "quiet-startup" => (callbacks.on_quiet_startup_change)(value == "true"),
                "double-escape-action" => {
                    (callbacks.on_double_escape_action_change)(DoubleEscapeAction::from_str(value))
                }
                "tree-filter-mode" => {
                    (callbacks.on_tree_filter_mode_change)(filter_mode_from_str(value))
                }
                "thinking" => {
                    if let Some(level) = thinking_level_from_str(value) {
                        (callbacks.on_thinking_level_change)(level);
                    }
                }
                "theme" => (callbacks.on_theme_change)(value.to_string()),
                _ => {}
            })
        };

        let list = SettingsList::new(
            items,
            10,
            settings_theme,
            on_change,
            on_cancel,
            SettingsListOptions {
                enable_search: false,
                filter: None,
            },
            select_kb,
        );

        SettingsSelectorComponent {
            list,
            close,
            top_border: DynamicBorder::new(Style::fg(theme.fg(ColorToken::Border))),
            bottom_border: DynamicBorder::new(Style::fg(theme.fg(ColorToken::Border))),
        }
    }
}

fn bool_str(b: bool) -> String {
    if b {
        "true".into()
    } else {
        "false".into()
    }
}

fn bool_values() -> Vec<String> {
    vec!["true".into(), "false".into()]
}

impl Component for SettingsSelectorComponent {
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        let mut row = y;
        let mut painted = 0u16;
        let mut remaining = height;

        if remaining > 0 {
            let n = self.top_border.render(buf, x, row, width, remaining);
            painted += n;
            row += n;
            remaining -= n;
        }
        if remaining > 0 {
            let n = self.list.render(buf, x, row, width, remaining);
            painted += n;
            row += n;
            remaining -= n;
        }
        if remaining > 0 {
            let n = self.bottom_border.render(buf, x, row, width, remaining);
            painted += n;
        }
        painted
    }

    fn handle_key(&mut self, key: &KeyEvent) -> bool {
        let consumed = self.list.handle_key(key);
        if consumed {
            // Submenu close signal: written by submenu on_select/on_cancel, explicitly drives done_submenu here
            let pending = self.close.borrow_mut().take();
            if let Some(value) = pending {
                self.list.done_submenu(value);
            }
        }
        consumed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use zen_tui::components::select_list::SelectItem;

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

    fn config() -> SettingsConfig {
        SettingsConfig {
            auto_compact: true,
            steering_mode: DeliveryMode::OneAtATime,
            follow_up_mode: DeliveryMode::All,
            enable_skill_commands: true,
            show_hardware_cursor: false,
            editor_padding_x: 0,
            output_pad: 1,
            autocomplete_max_visible: 10,
            clear_on_shrink: false,
            hide_thinking_block: false,
            show_cache_miss_notices: true,
            quiet_startup: false,
            double_escape_action: DoubleEscapeAction::Fork,
            tree_filter_mode: FilterMode::Default,
            thinking_level: ThinkingLevel::Off,
            available_thinking_levels: vec![
                ThinkingLevel::Off,
                ThinkingLevel::Low,
                ThinkingLevel::Medium,
            ],
            current_theme: "dark".into(),
            available_themes: vec!["dark".into(), "light".into()],
        }
    }

    fn flush_text(buf: &mut ScreenBuffer) -> String {
        // flush output contains crossterm diff commands (MoveTo/SetColors/Print per cell);
        // here control bytes are stripped to restore "visible text" (printable chars, order preserved) for contains assertions
        let mut out = Vec::new();
        buf.flush(&mut out).unwrap();
        crate::components::strip_ansi(&String::from_utf8_lossy(&out))
            .chars()
            .filter(|c| !c.is_control())
            .collect()
    }

    #[test]
    fn delivery_mode_roundtrip() {
        assert_eq!(DeliveryMode::from_str("all"), DeliveryMode::All);
        assert_eq!(
            DeliveryMode::from_str("one-at-a-time"),
            DeliveryMode::OneAtATime
        );
        assert_eq!(DeliveryMode::All.to_str(), "all");
    }

    #[test]
    fn double_escape_roundtrip() {
        assert_eq!(
            DoubleEscapeAction::from_str("tree"),
            DoubleEscapeAction::Tree
        );
        assert_eq!(
            DoubleEscapeAction::from_str("fork"),
            DoubleEscapeAction::Fork
        );
        assert_eq!(
            DoubleEscapeAction::from_str("none"),
            DoubleEscapeAction::None
        );
        assert_eq!(DoubleEscapeAction::None.to_str(), "none");
    }

    #[test]
    fn renders_items_and_values() {
        let mut comp = SettingsSelectorComponent::new(
            config(),
            SettingsCallbacks::default(),
            theme(),
            theme().settings_list_theme(),
            select_kb(),
        );
        let mut buf = ScreenBuffer::new(80, 20);
        let painted = comp.render(&mut buf, 0, 0, 80, 20);
        assert!(painted >= 4);
        let text = flush_text(&mut buf);
        assert!(text.contains("Auto-compact"));
        assert!(text.contains("Steering mode"));
    }

    #[test]
    fn enter_cycles_bool_setting_and_fires_callback() {
        let changed = Rc::new(RefCell::new(false));
        let c = changed.clone();
        let callbacks = SettingsCallbacks {
            on_auto_compact_change: Box::new(move |v| *c.borrow_mut() = v),
            ..Default::default()
        };
        let mut comp = SettingsSelectorComponent::new(
            config(),
            callbacks,
            theme(),
            theme().settings_list_theme(),
            select_kb(),
        );
        // First item selected by default (Auto-compact true → cycle to "false")
        let enter = crossterm::event::KeyEvent::from(crossterm::event::KeyCode::Enter);
        assert!(comp.handle_key(&enter));
        assert!(!*changed.borrow());
    }

    #[test]
    fn esc_fires_on_cancel() {
        let cancelled = Rc::new(RefCell::new(0));
        let c = cancelled.clone();
        let callbacks = SettingsCallbacks {
            on_cancel: Box::new(move || *c.borrow_mut() += 1),
            ..Default::default()
        };
        let mut comp = SettingsSelectorComponent::new(
            config(),
            callbacks,
            theme(),
            theme().settings_list_theme(),
            select_kb(),
        );
        let esc = crossterm::event::KeyEvent::from(crossterm::event::KeyCode::Esc);
        assert!(comp.handle_key(&esc));
        assert_eq!(*cancelled.borrow(), 1);
    }

    #[test]
    fn thinking_submenu_select_updates_level_and_closes() {
        let level = Rc::new(RefCell::new(None));
        let l = level.clone();
        let callbacks = SettingsCallbacks {
            on_thinking_level_change: Box::new(move |v| *l.borrow_mut() = Some(v)),
            ..Default::default()
        };
        let mut comp = SettingsSelectorComponent::new(
            config(),
            callbacks,
            theme(),
            theme().settings_list_theme(),
            select_kb(),
        );
        // Navigate to the "thinking" item (index 14) and press Enter to open the submenu
        for _ in 0..14 {
            assert!(comp.handle_key(&crossterm::event::KeyEvent::from(
                crossterm::event::KeyCode::Down
            )));
        }
        let enter = crossterm::event::KeyEvent::from(crossterm::event::KeyCode::Enter);
        assert!(comp.handle_key(&enter));
        // submenu open: currently Off (first item) → Down to Low (index 1) then Enter
        assert!(comp.handle_key(&crossterm::event::KeyEvent::from(
            crossterm::event::KeyCode::Down
        )));
        assert!(comp.handle_key(&enter));
        assert_eq!(*level.borrow(), Some(ThinkingLevel::Low));
        // Submenu closed (no longer rendered on the next frame)
    }

    #[test]
    fn theme_submenu_select_fires_theme_change() {
        let theme_name = Rc::new(RefCell::new(String::new()));
        let t = theme_name.clone();
        let callbacks = SettingsCallbacks {
            on_theme_change: Box::new(move |v| *t.borrow_mut() = v),
            ..Default::default()
        };
        let mut comp = SettingsSelectorComponent::new(
            config(),
            callbacks,
            theme(),
            theme().settings_list_theme(),
            select_kb(),
        );
        // Navigate to the "theme" item (index 15)
        for _ in 0..15 {
            assert!(comp.handle_key(&crossterm::event::KeyEvent::from(
                crossterm::event::KeyCode::Down
            )));
        }
        let enter = crossterm::event::KeyEvent::from(crossterm::event::KeyCode::Enter);
        assert!(comp.handle_key(&enter));
        // Select "light" (index 1)
        assert!(comp.handle_key(&crossterm::event::KeyEvent::from(
            crossterm::event::KeyCode::Down
        )));
        assert!(comp.handle_key(&enter));
        assert_eq!(*theme_name.borrow(), "light");
    }

    #[test]
    fn submenu_cancel_closes_without_change() {
        let level = Rc::new(RefCell::new(None));
        let l = level.clone();
        let callbacks = SettingsCallbacks {
            on_thinking_level_change: Box::new(move |v| *l.borrow_mut() = Some(v)),
            ..Default::default()
        };
        let mut comp = SettingsSelectorComponent::new(
            config(),
            callbacks,
            theme(),
            theme().settings_list_theme(),
            select_kb(),
        );
        for _ in 0..14 {
            assert!(comp.handle_key(&crossterm::event::KeyEvent::from(
                crossterm::event::KeyCode::Down
            )));
        }
        let enter = crossterm::event::KeyEvent::from(crossterm::event::KeyCode::Enter);
        assert!(comp.handle_key(&enter));
        // Esc cancels the submenu
        let esc = crossterm::event::KeyEvent::from(crossterm::event::KeyCode::Esc);
        assert!(comp.handle_key(&esc));
        assert!(level.borrow().is_none());
    }

    #[test]
    fn select_item_label_helpers() {
        let item = SelectItem::new("v", "", Some("d".into()));
        assert_eq!(item.label(), "v");
        assert_eq!(item.value(), "v");
        assert_eq!(item.description(), Some("d"));
    }
}
