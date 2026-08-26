// REQ-APP-11 (pi-358): keybinding-hints — formats "action → currently effective key" as readable hints.
// Corresponds to original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/keybinding-hints.ts.
// Difference: pi relies on global getKeybindings(), zen explicitly passes `&KeybindingsManager<AppAction>`;
// pi returns ANSI-colored strings, zen returns `StyledSegment` segments (merged into the line buffer by the component).

use crate::keybindings::AppAction;
use zen_tui::highlight::StyledSegment;
use zen_tui::keybindings::KeybindingsManager;
use zen_tui::screen::Style;
use zen_tui::theme::{ColorToken, Theme};

/// Called when footer/execution components print action bindings as readable strings like "ctrl+o" —
/// Formats a key chain (combo keys separated by "/") as display text; when capitalize=true, capitalizes the first letter (keyDisplayText).
/// On darwin, "alt" → "option" (aligned with pi formatKeyPart).
pub fn format_key_text(key: &str, capitalize: bool) -> String {
    fn format_part(part: &str, capitalize: bool) -> String {
        let display = if cfg!(target_os = "macos") && part.eq_ignore_ascii_case("alt") {
            "option"
        } else {
            part
        };
        if capitalize {
            let mut chars = display.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        } else {
            display.to_string()
        }
    }
    key.split('/')
        .map(|k| {
            k.split('+')
                .map(|part| format_part(part, capitalize))
                .collect::<Vec<_>>()
                .join("+")
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn format_keys(keys: &[zen_tui::keys::KeyId], capitalize: bool) -> String {
    if keys.is_empty() {
        return String::new();
    }
    let joined = keys
        .iter()
        .map(|k| k.to_string())
        .collect::<Vec<_>>()
        .join("/");
    format_key_text(&joined, capitalize)
}

/// Called when a component writes "action → currently effective key" hints to the status line — looks up the KeybindingsManager resolution table,
/// Returns an empty string when unbound. Consumers: bash-execution/status-indicator hint text, etc.
pub fn key_text(kb: &KeybindingsManager<AppAction>, action: AppAction) -> String {
    format_keys(kb.get_keys(action), false)
}

/// Same as `key_text`, but capitalizes the key name's first letter — used by settings-selector follow-up description text.
pub fn key_display_text(kb: &KeybindingsManager<AppAction>, action: AppAction) -> String {
    format_keys(kb.get_keys(action), true)
}

/// Called when a component assembles the two styled segments "key dim + description muted" into one line — returns two StyledSegments,
/// merged into the line buffer by the caller. Consumers: bash-execution's "… to expand", bordered-loader's "cancel" hint.
pub fn key_hint_segments(
    kb: &KeybindingsManager<AppAction>,
    action: AppAction,
    description: &str,
    theme: &Theme,
) -> Vec<StyledSegment> {
    let key = key_text(kb, action);
    mut_segments(&key, description, theme)
}

/// Same as `key_hint_segments`, but the key is a raw string (not via registry) — used by first-time-setup's "↑↓ navigate" hint.
pub fn raw_key_hint_segments(key: &str, description: &str, theme: &Theme) -> Vec<StyledSegment> {
    mut_segments(key, description, theme)
}

fn mut_segments(key: &str, description: &str, theme: &Theme) -> Vec<StyledSegment> {
    let mut segs = Vec::new();
    if !key.is_empty() {
        segs.push(StyledSegment {
            style: Style::fg(theme.fg(ColorToken::Dim)),
            text: format!("{key} "),
        });
    }
    segs.push(StyledSegment {
        style: Style::fg(theme.fg(ColorToken::Muted)),
        text: description.to_string(),
    });
    segs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keybindings::app_default_bindings;
    use std::collections::HashMap;

    fn kb() -> KeybindingsManager<AppAction> {
        KeybindingsManager::new(app_default_bindings(), HashMap::new())
    }

    #[test]
    fn format_key_text_plain_and_combo() {
        assert_eq!(format_key_text("ctrl+o", false), "ctrl+o");
        assert_eq!(format_key_text("ctrl+o", true), "Ctrl+O");
        assert_eq!(format_key_text("esc", false), "esc");
    }

    #[test]
    fn format_key_text_slash_chain() {
        // On darwin, alt → option (aligned with pi formatKeyPart)
        if cfg!(target_os = "macos") {
            assert_eq!(
                format_key_text("ctrl+left/alt+left", false),
                "ctrl+left/option+left"
            );
            assert_eq!(
                format_key_text("ctrl+left/alt+left", true),
                "Ctrl+Left/Option+Left"
            );
        } else {
            assert_eq!(
                format_key_text("ctrl+left/alt+left", false),
                "ctrl+left/alt+left"
            );
            assert_eq!(
                format_key_text("ctrl+left/alt+left", true),
                "Ctrl+Left/Alt+Left"
            );
        }
    }

    #[test]
    fn key_text_resolves_managed_keys() {
        assert_eq!(key_text(&kb(), AppAction::ToolsExpand), "ctrl+o");
        assert_eq!(key_text(&kb(), AppAction::Interrupt), "esc");
    }

    #[test]
    fn key_display_text_capitalizes() {
        assert_eq!(key_display_text(&kb(), AppAction::ToolsExpand), "Ctrl+O");
    }

    #[test]
    fn key_hint_segments_make_two_segments() {
        let theme = zen_tui::theme::builtin_dark_theme();
        let segs = key_hint_segments(&kb(), AppAction::ToolsExpand, "to expand", theme);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].text, "ctrl+o ");
        assert_eq!(segs[1].text, "to expand");
    }

    #[test]
    fn raw_key_hint_segments_skips_empty_key_part() {
        let theme = zen_tui::theme::builtin_dark_theme();
        // Empty key: still produces the description segment (prevents the hint line from disappearing in the onboarding hint)
        let segs = raw_key_hint_segments("", "navigate", theme);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "navigate");
    }
}
