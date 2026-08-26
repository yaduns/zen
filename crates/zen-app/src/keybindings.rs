// REQ-APP-11 (pi-358, pi-322): app-level shortcut actions — AppAction + default key table.
// Boundary convention: this module defines the action enum and default key table (T-AP-11-2's custom-editor / keybinding-hints need it immediately);
// T-AP-06-2 (configurable shortcuts) only does "user override injection + persistence", reusing this module without redefining.
// not included in v1: app.model.* / app.models.* / app.session.new|tree|fork|resume (T-AP-11-1 mode-level commands,
// variants added then), app.suspend (deferred), app.editor.external (T-AP-11-4).
// See reference/pi-original/packages/coding-agent/src/core/keybindings.ts (compare defaultKeys).

use zen_tui::keybindings::KeybindingDefinition;
use zen_tui::keys::KeyId;

/// Application-level shortcut action identities (v1 subset of pi `AppKeybinding`).
///
/// custom-editor (app.interrupt/exit/clear etc.), keybinding-hints (app.tools.expand etc.) and
/// the selectors (session/tree actions) route keys via `KeybindingsManager<AppAction>`; variants are pure discriminators,
/// not a tagged union, not serialized (user overrides persisted by T-AP-06-2 as action id strings; if serde is needed then,
/// use `rename_all = "kebab-case"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AppAction {
    /// Cancel/abort the current operation (Esc). custom-editor and status-indicator/bash-execution hints.
    Interrupt,
    /// Clear the editor (Ctrl+C).
    Clear,
    /// Exit when the editor is empty (Ctrl+D). custom-editor empty-editor route.
    Exit,
    /// Cycle through thinking levels (Shift+Tab). Planned consumer T-AP-11-1.
    ThinkingCycle,
    /// Toggle thinking block display (Ctrl+T). Planned consumer T-AP-11-1.
    ThinkingToggle,
    /// Toggle tool/bash output expansion (Ctrl+O). bash/tool/branch/compaction/skill expansion hints.
    ToolsExpand,
    /// Session picker: toggle name filter (Ctrl+N).
    SessionToggleNamedFilter,
    /// Session picker: toggle path display (Ctrl+P).
    SessionTogglePath,
    /// Session picker: toggle sort mode (Ctrl+S).
    SessionToggleSort,
    /// Session picker: enter rename (Ctrl+R).
    SessionRename,
    /// Session picker: delete session (Ctrl+D).
    SessionDelete,
    /// Session selector: delete session when query is empty (Ctrl+Backspace).
    SessionDeleteNoninvasive,
    /// Session tree: collapse branch or move up (Ctrl+Left / Alt+Left, platform ordering aligned with pi).
    TreeFoldOrUp,
    /// Session tree: expand branch or move down (Ctrl+Right / Alt+Right).
    TreeUnfoldOrDown,
    /// Session tree: edit tag (Shift+L).
    TreeEditLabel,
    /// Session tree: toggle tag timestamp display (Shift+T).
    TreeToggleLabelTimestamp,
    /// Session tree: default filter (Ctrl+D).
    TreeFilterDefault,
    /// Session tree: hide tool results filter (Ctrl+T).
    TreeFilterNoTools,
    /// Session tree: user messages only filter (Ctrl+U).
    TreeFilterUserOnly,
    /// Session tree: tag entries only filter (Ctrl+L).
    TreeFilterLabeledOnly,
    /// Session tree: show all entries filter (Ctrl+A).
    TreeFilterAll,
    /// Session tree: cycle filters forward (Ctrl+O).
    TreeFilterCycleForward,
    /// Session tree: cycle filters backward (Ctrl+I).
    TreeFilterCycleBackward,
    /// Copy message to clipboard (Ctrl+X). Planned consumer T-AP-11-1.
    MessageCopy,
    /// Queue follow-up message (Alt+Enter). Settings-selector description text.
    MessageFollowUp,
    /// Restore queued message (Alt+Up). Planned consumer T-AP-11-1.
    MessageDequeue,
    /// Paste clipboard image (Ctrl+V, Windows Alt+V). No images in v1, reserved for T-AP-11-1 to decide.
    ClipboardPasteImage,
}

/// Build a single keybinding definition (action + default key string + description).
fn keybind(
    action: AppAction,
    default_keys: &[&str],
    description: &str,
) -> KeybindingDefinition<AppAction> {
    KeybindingDefinition {
        action,
        default_keys: default_keys
            .iter()
            .filter_map(|k| KeyId::parse(k))
            .collect(),
        description: Some(description.to_string()),
    }
}

/// App-level action default key table (aligned with pi app-keybindings.ts defaultKeys).
///
/// Shared between T-AP-11-2's custom-editor / keybinding-hints and T-AP-11-1's mode-level commands;
/// T-AP-06-2 injects user overrides on top of it (`KeybindingsManager::set_user_bindings`), not redefined.
/// Covers all `AppAction` variants.
pub fn app_default_bindings() -> Vec<KeybindingDefinition<AppAction>> {
    vec![
        keybind(AppAction::Interrupt, &["esc"], "Cancel or abort"),
        keybind(AppAction::Clear, &["ctrl+c"], "Clear editor"),
        keybind(AppAction::Exit, &["ctrl+d"], "Exit when editor is empty"),
        keybind(
            AppAction::ThinkingCycle,
            &["shift+tab"],
            "Cycle thinking level",
        ),
        keybind(
            AppAction::ThinkingToggle,
            &["ctrl+t"],
            "Toggle thinking blocks",
        ),
        keybind(AppAction::ToolsExpand, &["ctrl+o"], "Toggle tool output"),
        keybind(
            AppAction::SessionToggleNamedFilter,
            &["ctrl+n"],
            "Toggle named session filter",
        ),
        keybind(
            AppAction::SessionTogglePath,
            &["ctrl+p"],
            "Toggle session path display",
        ),
        keybind(
            AppAction::SessionToggleSort,
            &["ctrl+s"],
            "Toggle session sort mode",
        ),
        keybind(AppAction::SessionRename, &["ctrl+r"], "Rename session"),
        keybind(AppAction::SessionDelete, &["ctrl+d"], "Delete session"),
        keybind(
            AppAction::SessionDeleteNoninvasive,
            &["ctrl+backspace"],
            "Delete session when query is empty",
        ),
        keybind(
            AppAction::TreeFoldOrUp,
            &["ctrl+left", "alt+left"],
            "Fold tree branch or move up",
        ),
        keybind(
            AppAction::TreeUnfoldOrDown,
            &["ctrl+right", "alt+right"],
            "Unfold tree branch or move down",
        ),
        keybind(AppAction::TreeEditLabel, &["shift+l"], "Edit tree label"),
        keybind(
            AppAction::TreeToggleLabelTimestamp,
            &["shift+t"],
            "Toggle tree label timestamps",
        ),
        keybind(
            AppAction::TreeFilterDefault,
            &["ctrl+d"],
            "Tree filter: default view",
        ),
        keybind(
            AppAction::TreeFilterNoTools,
            &["ctrl+t"],
            "Tree filter: hide tool results",
        ),
        keybind(
            AppAction::TreeFilterUserOnly,
            &["ctrl+u"],
            "Tree filter: user messages only",
        ),
        keybind(
            AppAction::TreeFilterLabeledOnly,
            &["ctrl+l"],
            "Tree filter: labeled entries only",
        ),
        keybind(
            AppAction::TreeFilterAll,
            &["ctrl+a"],
            "Tree filter: show all entries",
        ),
        keybind(
            AppAction::TreeFilterCycleForward,
            &["ctrl+o"],
            "Tree filter: cycle forward",
        ),
        keybind(
            AppAction::TreeFilterCycleBackward,
            &["ctrl+i"],
            "Tree filter: cycle backward",
        ),
        keybind(
            AppAction::MessageCopy,
            &["ctrl+x"],
            "Copy message to clipboard",
        ),
        keybind(
            AppAction::MessageFollowUp,
            &["alt+enter"],
            "Queue follow-up message",
        ),
        keybind(
            AppAction::MessageDequeue,
            &["alt+up"],
            "Restore queued messages",
        ),
        keybind(
            AppAction::ClipboardPasteImage,
            &["ctrl+v"],
            "Paste image from clipboard",
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use zen_tui::keybindings::KeybindingsManager;

    fn variants() -> Vec<AppAction> {
        use AppAction::*;
        vec![
            Interrupt,
            Clear,
            Exit,
            ThinkingCycle,
            ThinkingToggle,
            ToolsExpand,
            SessionToggleNamedFilter,
            SessionTogglePath,
            SessionToggleSort,
            SessionRename,
            SessionDelete,
            SessionDeleteNoninvasive,
            TreeFoldOrUp,
            TreeUnfoldOrDown,
            TreeEditLabel,
            TreeToggleLabelTimestamp,
            TreeFilterDefault,
            TreeFilterNoTools,
            TreeFilterUserOnly,
            TreeFilterLabeledOnly,
            TreeFilterAll,
            TreeFilterCycleForward,
            TreeFilterCycleBackward,
            MessageCopy,
            MessageFollowUp,
            MessageDequeue,
            ClipboardPasteImage,
        ]
    }

    #[test]
    fn defaults_cover_all_variants() {
        let defs = app_default_bindings();
        let covered: std::collections::HashSet<_> = defs.iter().map(|d| d.action).collect();
        for v in variants() {
            assert!(covered.contains(&v), "variant {v:?} missing from defaults");
        }
        assert_eq!(covered.len(), variants().len());
    }

    #[test]
    fn defaults_parse_all_keys() {
        for d in app_default_bindings() {
            assert!(
                !d.default_keys.is_empty(),
                "{:?} has no default keys",
                d.action
            );
        }
    }

    #[test]
    fn interrupt_defaults_to_esc_and_tools_expand_to_ctrl_o() {
        let kb = KeybindingsManager::new(app_default_bindings(), Default::default());
        let esc = KeyId::parse("esc").unwrap();
        let ctrl_o = KeyId::parse("ctrl+o").unwrap();
        assert!(kb.matches(&esc, AppAction::Interrupt));
        assert!(kb.matches(&ctrl_o, AppAction::ToolsExpand));
        assert!(kb.matches(&ctrl_o, AppAction::TreeFilterCycleForward));
    }

    #[test]
    fn descriptions_are_present() {
        for d in app_default_bindings() {
            assert!(d.description.is_some(), "{:?} has no description", d.action);
        }
    }

    #[test]
    fn set_user_bindings_overrides_default() {
        let mut kb = KeybindingsManager::new(app_default_bindings(), Default::default());
        let custom = KeyId::parse("f2").unwrap();
        let mut user = std::collections::HashMap::new();
        user.insert(AppAction::Interrupt, vec![custom]);
        kb.set_user_bindings(user);
        // After the override, esc no longer matches Interrupt
        let esc = KeyId::parse("esc").unwrap();
        assert!(!kb.matches(&esc, AppAction::Interrupt));
        assert!(kb.matches(&custom, AppAction::Interrupt));
    }
}
