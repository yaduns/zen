// REQ-APP-11 (pi-358, pi-322): application-level shortcut actions — AppAction + default keymap + user override loading (T-AP-06-2).
// Boundary convention: this module defines the action enum and default key table (T-AP-11-2's custom-editor / keybinding-hints need it immediately);
// T-AP-06-2 (configurable keybindings) lands "user override loading (keybindings.json) + injection" in this module, reusing the
// zen-tui registry (KeybindingsManager) instead of redefining it.
// pi canonical ids not included in v1 (app.model.* / app.models.* / app.session.new|tree|fork|resume /
// app.suspend / app.editor.external, etc.) are represented as None entries in the id tables (reserved items; silently skipped if present in user config),
// see the *_ID_TABLE entries below and tasks/T-AP-06-2-implemented.md for the full list and ownership.
// Reference: reference/pi-original/packages/coding-agent/src/core/keybindings.ts (defaultKeys comparison,
// config loading/type validation aligns with pi toKeybindingsConfig; migration table/reload/on-disk ordering dropped in v1, see task doc).

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use zen_tui::components::select_list::SelectAction;
use zen_tui::editor::EditorAction;
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

// ── User override loading (T-AP-06-2, counterpart of pi KeybindingsManager.create/loadFromFile) ──────────

/// `app.*` domain id table: `(config id, action)`; `None` = pi spec id not included in v1 (**placeholder**, silently skipped).
/// Maintenance convention: add AppAction variant → add Some row to table + exhaustive unit test lock (the match body in `*_from_id` is
/// `&str`, so exhaustiveness is not enforced at compile time); once the pi id exists, change the placeholder to Some.
const APP_ID_TABLE: &[(&str, Option<AppAction>)] = &[
    // ── Included (Some) ──
    ("app.interrupt", Some(AppAction::Interrupt)),
    ("app.clear", Some(AppAction::Clear)),
    ("app.exit", Some(AppAction::Exit)),
    ("app.thinking.cycle", Some(AppAction::ThinkingCycle)),
    ("app.thinking.toggle", Some(AppAction::ThinkingToggle)),
    ("app.tools.expand", Some(AppAction::ToolsExpand)),
    (
        "app.session.toggleNamedFilter",
        Some(AppAction::SessionToggleNamedFilter),
    ),
    ("app.session.togglePath", Some(AppAction::SessionTogglePath)),
    ("app.session.toggleSort", Some(AppAction::SessionToggleSort)),
    ("app.session.rename", Some(AppAction::SessionRename)),
    ("app.session.delete", Some(AppAction::SessionDelete)),
    (
        "app.session.deleteNoninvasive",
        Some(AppAction::SessionDeleteNoninvasive),
    ),
    ("app.tree.foldOrUp", Some(AppAction::TreeFoldOrUp)),
    ("app.tree.unfoldOrDown", Some(AppAction::TreeUnfoldOrDown)),
    ("app.tree.editLabel", Some(AppAction::TreeEditLabel)),
    (
        "app.tree.toggleLabelTimestamp",
        Some(AppAction::TreeToggleLabelTimestamp),
    ),
    (
        "app.tree.filter.default",
        Some(AppAction::TreeFilterDefault),
    ),
    (
        "app.tree.filter.noTools",
        Some(AppAction::TreeFilterNoTools),
    ),
    (
        "app.tree.filter.userOnly",
        Some(AppAction::TreeFilterUserOnly),
    ),
    (
        "app.tree.filter.labeledOnly",
        Some(AppAction::TreeFilterLabeledOnly),
    ),
    ("app.tree.filter.all", Some(AppAction::TreeFilterAll)),
    (
        "app.tree.filter.cycleForward",
        Some(AppAction::TreeFilterCycleForward),
    ),
    (
        "app.tree.filter.cycleBackward",
        Some(AppAction::TreeFilterCycleBackward),
    ),
    ("app.message.copy", Some(AppAction::MessageCopy)),
    ("app.message.followUp", Some(AppAction::MessageFollowUp)),
    ("app.message.dequeue", Some(AppAction::MessageDequeue)),
    (
        "app.clipboard.pasteImage",
        Some(AppAction::ClipboardPasteImage),
    ),
    // ── Reserved items (None, not included in v1; consumers T-AP-11-1 / T-AP-11-4 will switch to Some later) ──
    ("app.suspend", None),
    ("app.model.cycleForward", None),
    ("app.model.cycleBackward", None),
    ("app.model.select", None),
    ("app.editor.external", None),
    ("app.session.new", None),
    ("app.session.tree", None),
    ("app.session.fork", None),
    ("app.session.resume", None),
    ("app.models.save", None),
    ("app.models.enableAll", None),
    ("app.models.clearAll", None),
    ("app.models.toggleProvider", None),
    ("app.models.reorderUp", None),
    ("app.models.reorderDown", None),
];

/// `tui.editor.*` / `tui.input.*` domain id table (EditorAction). Only placeholder is `tui.input.copy`
/// (zen editor has no selection copy).
const EDITOR_ID_TABLE: &[(&str, Option<EditorAction>)] = &[
    ("tui.editor.cursorUp", Some(EditorAction::MoveCursorUp)),
    ("tui.editor.cursorDown", Some(EditorAction::MoveCursorDown)),
    ("tui.editor.cursorLeft", Some(EditorAction::MoveCursorLeft)),
    (
        "tui.editor.cursorRight",
        Some(EditorAction::MoveCursorRight),
    ),
    (
        "tui.editor.cursorWordLeft",
        Some(EditorAction::MoveCursorWordLeft),
    ),
    (
        "tui.editor.cursorWordRight",
        Some(EditorAction::MoveCursorWordRight),
    ),
    (
        "tui.editor.cursorLineStart",
        Some(EditorAction::MoveToLineStart),
    ),
    (
        "tui.editor.cursorLineEnd",
        Some(EditorAction::MoveToLineEnd),
    ),
    ("tui.editor.jumpForward", Some(EditorAction::JumpForward)),
    ("tui.editor.jumpBackward", Some(EditorAction::JumpBackward)),
    ("tui.editor.pageUp", Some(EditorAction::PageUp)),
    ("tui.editor.pageDown", Some(EditorAction::PageDown)),
    (
        "tui.editor.deleteCharBackward",
        Some(EditorAction::DeleteCharBackward),
    ),
    (
        "tui.editor.deleteCharForward",
        Some(EditorAction::DeleteCharForward),
    ),
    (
        "tui.editor.deleteWordBackward",
        Some(EditorAction::DeleteWordBackward),
    ),
    (
        "tui.editor.deleteWordForward",
        Some(EditorAction::DeleteWordForward),
    ),
    (
        "tui.editor.deleteToLineStart",
        Some(EditorAction::DeleteToLineStart),
    ),
    (
        "tui.editor.deleteToLineEnd",
        Some(EditorAction::DeleteToLineEnd),
    ),
    ("tui.editor.yank", Some(EditorAction::Yank)),
    ("tui.editor.yankPop", Some(EditorAction::YankPop)),
    ("tui.editor.undo", Some(EditorAction::Undo)),
    ("tui.input.newLine", Some(EditorAction::NewLine)),
    ("tui.input.submit", Some(EditorAction::Submit)),
    ("tui.input.tab", Some(EditorAction::Tab)),
    // ── Reserved ──
    ("tui.input.copy", None),
];

/// `tui.select.*` domain id table (SelectAction). Placeholders are `tui.select.pageUp`/`pageDown`
/// (zen SelectList has no paging).
const SELECT_ID_TABLE: &[(&str, Option<SelectAction>)] = &[
    ("tui.select.up", Some(SelectAction::MoveUp)),
    ("tui.select.down", Some(SelectAction::MoveDown)),
    ("tui.select.confirm", Some(SelectAction::Confirm)),
    ("tui.select.cancel", Some(SelectAction::Cancel)),
    // ── Reserved ──
    ("tui.select.pageUp", None),
    ("tui.select.pageDown", None),
];

/// Config key `app.*` id → AppAction (pi AppKeybinding spec name, e.g. `"app.thinking.cycle"`);
/// called when the loader dispatches the app prefix segment. None placeholders in the table are not matched here; the loader silently skips them.
/// Consumers: `load_user_keybindings` internal dispatch + unit tests in this module (module-private, not marked pub).
fn app_action_from_id(id: &str) -> Option<AppAction> {
    APP_ID_TABLE
        .iter()
        .find(|(key, _)| *key == id)
        .and_then(|(_, action)| *action)
}

/// Config key `tui.editor.*`/`tui.input.*` id → EditorAction (e.g. `"tui.editor.cursorUp"`/
/// `"tui.input.submit"`); called when the loader dispatches the tui. prefix segment. None placeholders in the table (`tui.input.copy`)
/// are not matched here; the loader silently skips them.
/// Consumers: internal dispatch in `load_user_keybindings` + this module's unit tests.
fn editor_action_from_id(id: &str) -> Option<EditorAction> {
    EDITOR_ID_TABLE
        .iter()
        .find(|(key, _)| *key == id)
        .and_then(|(_, action)| *action)
}

/// Config key `tui.select.*` id → SelectAction (e.g. `"tui.select.confirm"`); None placeholders in the table
/// (`tui.select.pageUp`/`pageDown`) are not matched here; the loader silently skips them.
/// Consumers: internal dispatch in `load_user_keybindings` + this module's unit tests.
fn select_action_from_id(id: &str) -> Option<SelectAction> {
    SELECT_ID_TABLE
        .iter()
        .find(|(key, _)| *key == id)
        .and_then(|(_, action)| *action)
}

/// Determines whether an id is a placeholder in the id table (a pi spec id not included in v1, e.g. `"app.session.fork"`);
/// the loader calls it when `*_from_id` misses, to distinguish "placeholder (silently skipped)" from "genuinely unknown (UnknownAction diagnostic)".
/// Consumers: internal dispatch in `load_user_keybindings` + this module's unit tests.
fn is_reserved_pi_id(id: &str) -> bool {
    APP_ID_TABLE
        .iter()
        .filter(|(_, action)| action.is_none())
        .map(|(key, _)| *key)
        .chain(
            EDITOR_ID_TABLE
                .iter()
                .filter(|(_, action)| action.is_none())
                .map(|(key, _)| *key),
        )
        .chain(
            SELECT_ID_TABLE
                .iter()
                .filter(|(_, action)| action.is_none())
                .map(|(key, _)| *key),
        )
        .any(|key| key == id)
}

/// Product of one `load_user_keybindings` assembly: user override tables for the three action domains. Called by interactive init
/// destructured into the second argument of the three `KeybindingsManager` constructors, then its purpose is fulfilled (a short-lived assembly value, not long-lived state).
/// In-memory only, no serialization (JSON parsing completes within the load step, no intermediate DTO persisted).
#[derive(Debug, Default)]
pub struct UserKeybindings {
    /// App-domain user overrides (empty table = no overrides, everything falls back to defaults).
    pub app: HashMap<AppAction, Vec<KeyId>>,
    /// Editor-domain user overrides.
    pub editor: HashMap<EditorAction, Vec<KeyId>>,
    /// Selection-component-domain user overrides.
    pub select: HashMap<SelectAction, Vec<KeyId>>,
}

/// User-visible issues from the load step (init surfaces each via `show_warning`), aligning with the settings "bad files are not silent" precedent;
/// The corresponding pi path silently ignores them, zen surfaces them as diagnostics (deliberate divergence, see "ecosystem reuse verification conclusion" in the task doc).
/// In-memory diagnostics only, no serialization.
#[derive(Debug, Clone, PartialEq)]
pub enum KeybindingLoadDiagnostic {
    /// File exists but could not be read (permissions etc.). A missing file is **not** a diagnostic (normally no customization).
    Io {
        /// Config file path.
        path: std::path::PathBuf,
        /// Error summary (`std::io::Error` message).
        message: String,
    },
    /// JSON parse failure or non-object top level (pi also skips but silently; zen surfaces it).
    Parse {
        /// Config file path.
        path: std::path::PathBuf,
        /// Parser error message or non-object explanation.
        message: String,
    },
    /// Config key not in any of the three domain id tables and not a reserved item (non-canonical name such as a typo); skip the whole entry.
    UnknownAction {
        /// Unrecognized action id.
        id: String,
    },
    /// Value is a key string but `KeyId::parse` failed; skip only that key, keep the other valid keys in the entry.
    InvalidKey {
        /// Action id.
        id: String,
        /// Unparseable key string.
        key: String,
    },
    /// Value is neither a string nor an all-string array (numbers/nested arrays/objects; pi `toKeybindingsConfig` also skips these).
    InvalidEntry {
        /// Action id.
        id: String,
    },
}

impl fmt::Display for KeybindingLoadDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, message } => write!(f, "read {}: {}", path.display(), message),
            Self::Parse { path, message } => write!(f, "parse {}: {}", path.display(), message),
            Self::UnknownAction { id } => {
                write!(f, "unknown keybinding action `{id}` (entry ignored)")
            }
            Self::InvalidKey { id, key } => {
                write!(f, "invalid key `{key}` for `{id}` (key ignored)")
            }
            Self::InvalidEntry { id } => write!(
                f,
                "invalid value for `{id}`: expected a key string or array of key strings (entry ignored)"
            ),
        }
    }
}

/// Load user overrides from `<agent_dir>/keybindings.json` into three domain action-override tables — "a mistyped key must not make the
/// app fail to start": any bad entry degrades to skip + diagnostic, never blocking startup. Called once during interactive init startup assembly
/// (T-AP-06-2); the output feeds directly into `KeybindingsManager::new` for each of the three domains.
/// Behavior: file missing → empty table, no diagnostics; read failure/bad JSON/non-object top level → empty table + Io/Parse diagnostics; per-entry validation
/// (key = action id in the three-domain id table, value = string or all-string array) later normalized via `KeyId::parse`; invalid key strings /
/// invalid value types are skipped and recorded as InvalidKey/InvalidEntry diagnostics; **ids outside the table** record UnknownAction, **reserved entries in the table**
/// (canonical pi spec ids not yet included in v1) are silently skipped without diagnostics (to avoid pi config migration noise, see the mapping table section of the task doc).
/// all key strings in an entry invalid → empty Vec in table = explicit unbinding (net effect equivalent to pi garbage key strings never matching).
/// Returns (three-domain override table, diagnostic list); no Result — a failed load degrades to an empty table, success/failure is conveyed via diagnostics.
/// Consumer: T-AP-11-1 (interactive init assembly).
pub fn load_user_keybindings(path: &Path) -> (UserKeybindings, Vec<KeybindingLoadDiagnostic>) {
    let mut out = UserKeybindings::default();
    let mut diagnostics = Vec::new();

    // file not exists = normally no customization (same semantics as pi existsSync), not a diagnostic
    if !path.exists() {
        return (out, diagnostics);
    }
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) => {
            diagnostics.push(KeybindingLoadDiagnostic::Io {
                path: path.to_path_buf(),
                message: err.to_string(),
            });
            return (out, diagnostics);
        }
    };
    let parsed: Value = match serde_json::from_str(&raw) {
        Ok(parsed) => parsed,
        Err(err) => {
            diagnostics.push(KeybindingLoadDiagnostic::Parse {
                path: path.to_path_buf(),
                message: err.to_string(),
            });
            return (out, diagnostics);
        }
    };
    let Some(entries) = parsed.as_object() else {
        diagnostics.push(KeybindingLoadDiagnostic::Parse {
            path: path.to_path_buf(),
            message: "top-level value is not a JSON object".to_string(),
        });
        return (out, diagnostics);
    };

    for (id, value) in entries {
        // value type validation (aligned with pi toKeybindingsConfig: string or all-string array, others skipped)
        let keys: Option<Vec<&str>> = match value {
            Value::String(s) => Some(vec![s.as_str()]),
            Value::Array(items) if items.iter().all(Value::is_string) => {
                Some(items.iter().filter_map(Value::as_str).collect())
            }
            _ => None,
        };
        let Some(keys) = keys else {
            diagnostics.push(KeybindingLoadDiagnostic::InvalidEntry { id: id.clone() });
            continue;
        };
        // key string parsing: invalid key strings skip that key and record a diagnostic, the rest are kept
        let mut parsed_keys = Vec::new();
        for key in keys {
            match KeyId::parse(key) {
                Some(kid) => parsed_keys.push(kid),
                None => diagnostics.push(KeybindingLoadDiagnostic::InvalidKey {
                    id: id.clone(),
                    key: key.to_string(),
                }),
            }
        }
        // three-domain dispatch; reserved entries silent, ids outside the table diagnosed
        if let Some(action) = app_action_from_id(id) {
            out.app.insert(action, parsed_keys);
        } else if let Some(action) = editor_action_from_id(id) {
            out.editor.insert(action, parsed_keys);
        } else if let Some(action) = select_action_from_id(id) {
            out.select.insert(action, parsed_keys);
        } else if is_reserved_pi_id(id) {
            // canonical pi spec ids not yet included in v1: valid config intent (to be honored later), silently skipped without diagnostics
        } else {
            diagnostics.push(KeybindingLoadDiagnostic::UnknownAction { id: id.clone() });
        }
    }

    (out, diagnostics)
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

    // ── exhaustive id table coverage (T-AP-06-2 maintenance convention: new variants must add a table row in sync) ──

    #[test]
    fn app_id_table_some_entries_cover_all_variants_exactly_once() {
        use AppAction::*;
        let all = [
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
        ];
        for variant in all {
            let hits: Vec<_> = APP_ID_TABLE
                .iter()
                .filter(|(_, a)| *a == Some(variant))
                .collect();
            assert_eq!(hits.len(), 1, "{variant:?} should appear exactly once in APP_ID_TABLE");
        }
        let some = APP_ID_TABLE.iter().filter(|(_, a)| a.is_some()).count();
        assert_eq!(some, 27);
    }

    #[test]
    fn editor_id_table_some_entries_cover_all_editor_actions_exactly_once() {
        use EditorAction::*;
        let all = [
            MoveCursorUp,
            MoveCursorDown,
            MoveCursorLeft,
            MoveCursorRight,
            MoveCursorWordLeft,
            MoveCursorWordRight,
            MoveToLineStart,
            MoveToLineEnd,
            JumpForward,
            JumpBackward,
            PageUp,
            PageDown,
            DeleteCharBackward,
            DeleteCharForward,
            DeleteWordBackward,
            DeleteWordForward,
            DeleteToLineStart,
            DeleteToLineEnd,
            Yank,
            YankPop,
            Undo,
            NewLine,
            Submit,
            Tab,
        ];
        for variant in all {
            let hits = EDITOR_ID_TABLE
                .iter()
                .filter(|(_, a)| *a == Some(variant))
                .count();
            assert_eq!(hits, 1, "{variant:?} should appear exactly once in EDITOR_ID_TABLE");
        }
        let some = EDITOR_ID_TABLE.iter().filter(|(_, a)| a.is_some()).count();
        assert_eq!(some, 24);
    }

    #[test]
    fn select_id_table_some_entries_cover_all_select_actions_exactly_once() {
        use SelectAction::*;
        let all = [MoveUp, MoveDown, Confirm, Cancel];
        for variant in all {
            let hits = SELECT_ID_TABLE
                .iter()
                .filter(|(_, a)| *a == Some(variant))
                .count();
            assert_eq!(hits, 1, "{variant:?} should appear exactly once in SELECT_ID_TABLE");
        }
    }

    #[test]
    fn reserved_entries_match_pi_only_id_manifest() {
        // Reserved item list: app 15 + editor 1 + select 2 = 18 (matches the task doc's "reserved item list")
        let reserved =
            |t: &[(&str, Option<AppAction>)]| t.iter().filter(|(_, a)| a.is_none()).count();
        assert_eq!(reserved(APP_ID_TABLE), 15);
        // editor/select table element types differ, assert separately
        assert_eq!(
            EDITOR_ID_TABLE.iter().filter(|(_, a)| a.is_none()).count(),
            1
        );
        assert_eq!(
            SELECT_ID_TABLE.iter().filter(|(_, a)| a.is_none()).count(),
            2
        );
        assert!(is_reserved_pi_id("app.session.fork"));
        assert!(is_reserved_pi_id("app.models.toggleProvider"));
        assert!(is_reserved_pi_id("tui.input.copy"));
        assert!(is_reserved_pi_id("tui.select.pageUp"));
        assert!(!is_reserved_pi_id("app.interrupt"));
        assert!(!is_reserved_pi_id("bogus.id"));
    }

    #[test]
    fn from_id_spot_checks() {
        assert_eq!(
            app_action_from_id("app.thinking.cycle"),
            Some(AppAction::ThinkingCycle)
        );
        assert_eq!(
            editor_action_from_id("tui.editor.cursorLineStart"),
            Some(EditorAction::MoveToLineStart)
        );
        assert_eq!(
            editor_action_from_id("tui.input.tab"),
            Some(EditorAction::Tab)
        );
        assert_eq!(
            select_action_from_id("tui.select.cancel"),
            Some(SelectAction::Cancel)
        );
        assert_eq!(app_action_from_id("app.nope"), None);
        // reserved entries produce no action
        assert_eq!(app_action_from_id("app.session.fork"), None);
    }

    // ── load_user_keybindings ──

    use tempfile::TempDir;

    /// Write a temporary keybindings.json and return its path.
    fn write_config(content: &str) -> (TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keybindings.json");
        std::fs::write(&path, content).unwrap();
        (dir, path)
    }

    #[test]
    fn load_missing_file_yields_empty_and_no_diagnostics() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent.json");
        let (out, diags) = load_user_keybindings(&path);
        assert!(out.app.is_empty() && out.editor.is_empty() && out.select.is_empty());
        assert!(diags.is_empty());
    }

    #[test]
    fn load_valid_config_dispatches_three_domains() {
        let (_dir, path) = write_config(
            r#"{
                "app.interrupt": "ctrl+e",
                "tui.editor.cursorUp": ["ctrl+k", "alt+k"],
                "tui.select.confirm": "ctrl+enter"
            }"#,
        );
        let (out, diags) = load_user_keybindings(&path);
        assert!(diags.is_empty(), "no diagnostics, actual {diags:?}");
        assert_eq!(
            out.app.get(&AppAction::Interrupt).unwrap()[0],
            KeyId::parse("ctrl+e").unwrap()
        );
        assert_eq!(
            out.editor.get(&EditorAction::MoveCursorUp).unwrap().len(),
            2
        );
        assert_eq!(
            out.select.get(&SelectAction::Confirm).unwrap()[0],
            KeyId::parse("ctrl+enter").unwrap()
        );
    }

    #[test]
    fn load_bad_json_and_non_object_yield_parse_diagnostic() {
        let (_d, p) = write_config("{not json");
        let (out, diags) = load_user_keybindings(&p);
        assert!(out.app.is_empty());
        assert!(matches!(diags[0], KeybindingLoadDiagnostic::Parse { .. }));

        let (_d, p) = write_config("[1, 2]");
        let (out, diags) = load_user_keybindings(&p);
        assert!(out.app.is_empty());
        assert!(matches!(diags[0], KeybindingLoadDiagnostic::Parse { .. }));
    }

    #[test]
    fn load_unknown_action_yields_diagnostic_and_reserved_is_silent() {
        let (_d, p) = write_config(
            r#"{ "app.nope": "f1", "app.session.fork": "f2", "app.interrupt": "ctrl+q" }"#,
        );
        let (out, diags) = load_user_keybindings(&p);
        // reserved entry app.session.fork silent; app.nope records UnknownAction; app.interrupt normally enters the table
        assert_eq!(diags.len(), 1);
        assert_eq!(
            diags[0],
            KeybindingLoadDiagnostic::UnknownAction {
                id: "app.nope".to_string()
            }
        );
        assert!(out.app.contains_key(&AppAction::Interrupt));
        assert!(!out.app.contains_key(&AppAction::SessionRename));
    }

    #[test]
    fn load_invalid_key_skips_key_and_keeps_valid_ones() {
        let (_d, p) = write_config(r#"{ "app.clear": ["ctrl+bad~", "f4"] }"#);
        let (out, diags) = load_user_keybindings(&p);
        assert_eq!(diags.len(), 1);
        assert!(matches!(
            diags[0],
            KeybindingLoadDiagnostic::InvalidKey { .. }
        ));
        assert_eq!(
            out.app.get(&AppAction::Clear).unwrap(),
            &[KeyId::parse("f4").unwrap()]
        );
    }

    #[test]
    fn load_invalid_entry_value_yields_diagnostic() {
        let (_d, p) = write_config(r#"{ "app.clear": 42, "app.exit": ["enter", [1]] }"#);
        let (out, diags) = load_user_keybindings(&p);
        assert_eq!(diags.len(), 2);
        assert!(matches!(
            diags[0],
            KeybindingLoadDiagnostic::InvalidEntry { .. }
        ));
        assert!(matches!(
            diags[1],
            KeybindingLoadDiagnostic::InvalidEntry { .. }
        ));
        assert!(out.app.is_empty());
    }

    #[test]
    fn load_all_keys_invalid_results_in_explicit_unbind() {
        // aligned with pi net effect: garbage key string list replaces default → that action cannot be triggered; zen expresses explicit unbinding with an empty Vec
        let (_d, p) = write_config(r#"{ "app.tools.expand": ["bad key"] }"#);
        let (out, diags) = load_user_keybindings(&p);
        assert_eq!(diags.len(), 1);
        assert_eq!(out.app.get(&AppAction::ToolsExpand), Some(&Vec::new()));
    }

    #[test]
    fn diagnostic_display_is_human_readable() {
        let d = KeybindingLoadDiagnostic::UnknownAction { id: "x".into() };
        assert_eq!(
            d.to_string(),
            "unknown keybinding action `x` (entry ignored)"
        );
    }
}
