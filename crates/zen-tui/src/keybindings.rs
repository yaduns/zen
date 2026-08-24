// REQ-TUI-03 (pi-403): Shortcut mapping—a registry of actions ↔ key sets (including user overrides and conflict detection).
// Corresponds to the original TS implementation: reference/pi-original/packages/tui/src/keybindings.ts.
// Raw TTY byte-string matching is handled after normalization by the KeyId (keys.rs) from T-TU-03-2,
// This module only performs "KeyId ↔ action registry" equality matching and manages overrides/conflicts; there is no global singleton, and the application layer explicitly constructs and injects the manager.
use crate::components::select_list::SelectAction;
use crate::editor::EditorAction;
use crate::keys::KeyId;
use std::collections::HashMap;
use std::hash::Hash;

/// Metadata for a single action's "default keys + description", serving as a construction input unit for the registry.
#[derive(Debug, Clone, PartialEq)]
pub struct KeybindingDefinition<A> {
    /// Action identity (instantiated as either EditorAction / SelectAction).
    pub action: A,
    /// The action's default binding keys (multiple allowed; empty = no default binding).
    pub default_keys: Vec<KeyId>,
    /// Display text (keybinding hints / footer), which may be None.
    pub description: Option<String>,
}

/// A record of a user-override conflict: multiple actions claim the same key simultaneously, used to notify the user/configuration UI that “the key is ambiguous”.
#[derive(Debug, Clone, PartialEq)]
pub struct KeybindingConflict<A> {
    /// The key involved in the conflict.
    pub key: KeyId,
    /// List of actions claiming the key (a conflict requires >1 item).
    pub actions: Vec<A>,
}

/// Shortcut registry—merges user overrides with the default table into an action → effective key mapping and detects key-claim conflicts within user overrides.
///
/// Key routing for the editor/selection component (`matches`) and keybinding-hints queries (`get_keys`, etc.) all go through this table;
/// `rebuild()` rebuilds the `resolved`/`conflicts` caches after `new` and `set_user_bindings`; this is the only mutation point.
pub struct KeybindingsManager<A> {
    /// Default definition table (indexed by action).
    definitions: HashMap<A, KeybindingDefinition<A>>,
    /// User binding overrides (may be an empty table = all defaults); persistence is handled by the app layer T-AP-06-2; this layer only holds runtime overrides.
    user_bindings: HashMap<A, Vec<KeyId>>,
    /// Merged table of effective keybindings (cached and maintained by `rebuild`).
    resolved: HashMap<A, Vec<KeyId>>,
    /// List of conflicts where multiple actions claim the same key within user overrides (cached).
    conflicts: Vec<KeybindingConflict<A>>,
}

/// Normalize a “key set” by deduplicating it while preserving order (aligns with pi normalizeKeys: convert the container to Vec, deduplicate, and preserve the original order).
/// Internal use: normalize input before merging default/user bindings; map `keys` to an empty Vec when it is an `Option` (unbound).
/// Consumer: internal to this module (rebuild / merging); not exposed externally.
fn normalize_keys(keys: Option<Vec<KeyId>>) -> Vec<KeyId> {
    let mut seen = std::collections::HashSet::new();
    keys.unwrap_or_default()
        .into_iter()
        .filter(|k| seen.insert(*k))
        .collect()
}

impl<A: Clone + Eq + Hash> KeybindingsManager<A> {
    /// Construct a keybinding registry from the default definition table + user overrides, then immediately rebuild the resolution and conflict tables.
    /// `defaults` contains all action definitions for the action scope (editor/selection widget); `user_bindings` is a nullable user override table
    /// (If empty, all actions fall back to their default bindings.) After construction, `matches`/`get_keys`/`get_conflicts` are immediately queryable.
    /// Boundary: entries in `user_bindings` that reference actions not defined in `defaults` are ignored (aligns with pi: skip if not in definitions).
    /// Consumers: planned consumers T-AP-11-2 (app layer builds the registries for editor/select components) and T-AP-06-2 (injects user config).
    pub fn new(
        defaults: Vec<KeybindingDefinition<A>>,
        user_bindings: HashMap<A, Vec<KeyId>>,
    ) -> Self {
        let definitions = defaults
            .into_iter()
            .map(|d| (d.action.clone(), d))
            .collect::<HashMap<_, _>>();
        let mut manager = Self {
            definitions,
            user_bindings,
            resolved: HashMap::new(),
            conflicts: Vec::new(),
        };
        manager.rebuild();
        manager
    }

    /// Rebuild the resolution and conflict tables: user overrides take precedence over defaults; record a conflict when multiple user actions claim the same key. Internal mutation point.
    fn rebuild(&mut self) {
        self.resolved.clear();
        self.conflicts.clear();

        // Conflicts arise only from contention between user overrides (aligns with pi: overlapping defaults do not produce conflicts)
        let mut user_claims: HashMap<KeyId, Vec<A>> = HashMap::new();
        for (action, keys) in &self.user_bindings {
            if !self.definitions.contains_key(action) {
                continue;
            }
            for key in normalize_keys(Some(keys.clone())) {
                user_claims.entry(key).or_default().push(action.clone());
            }
        }
        for (key, actions) in user_claims {
            if actions.len() > 1 {
                self.conflicts.push(KeybindingConflict { key, actions });
            }
        }

        // Effective keys per action: use the user override when present (empty Vec = explicit unbinding); otherwise, use the default
        for (id, definition) in &self.definitions {
            let keys = match self.user_bindings.get(id) {
                Some(user_keys) => normalize_keys(Some(user_keys.clone())),
                None => normalize_keys(Some(definition.default_keys.clone())),
            };
            self.resolved.insert(id.clone(), keys);
        }
    }

    /// Determines whether a normalized key matches a target action (called when the editor/selection component routes a key to an action).
    /// `key` is the `KeyId` normalized from crossterm `KeyEvent` in T-TU-03-2; `action` is the target action.
    /// Returns whether the key is in the action's active key set (true if it matches any). If no action matches, the caller forwards it up the chain.
    /// Consumers: editor::handle_key and each handle_key in components (replacing inline default keybindings) — planned consumers.
    pub fn matches(&self, key: &KeyId, action: A) -> bool {
        self.resolved
            .get(&action)
            .is_some_and(|keys| keys.contains(key))
    }

    /// Queries the currently active key list for an action (resolved, including user overrides). Used by keybinding hints / footer to display the current bindings.
    /// Returns a slice retrieved from the `resolved` cache; returns an empty slice for unbound actions.
    /// Consumers: planned consumer T-AP-11-3 (the keybinding-hints component retrieves the current key), and keybinding-hints-style UI.
    pub fn get_keys(&self, action: A) -> &[KeyId] {
        self.resolved
            .get(&action)
            .map_or(&[], |keys| keys.as_slice())
    }

    /// Queries the default definition metadata for an action (default key + description). Used by the configuration UI to display default values and help text.
    /// Returning `None` means the action is not in the registry (undefined).
    /// Consumers: planned consumers T-AP-06-2 (display default bindings), config/help UI.
    pub fn get_definition(&self, action: A) -> Option<&KeybindingDefinition<A>> {
        self.definitions.get(&action)
    }

    /// Returns the list of key conflicts caused by user overrides (the same key is claimed by multiple actions). Used for validation/notifications before saving the configuration.
    /// Returns a borrowed slice of the cached `conflicts`; empty when there are no conflicts.
    /// Consumers: planned consumer T-AP-06-2 (detects ambiguity before saving user keybindings).
    pub fn get_conflicts(&self) -> &[KeybindingConflict<A>] {
        &self.conflicts
    }

    /// Replaces all user override bindings and rebuilds the resolution/conflict tables (called when the user changes the keybinding configuration).
    /// An empty override `HashMap` clears all user overrides and reverts everything to the defaults. `get_user_bindings`/`get_resolved_bindings`
    /// Companion read of current state. Consumers: planned consumer T-AP-06-2 (called after saving new config).
    pub fn set_user_bindings(&mut self, user_bindings: HashMap<A, Vec<KeyId>>) {
        self.user_bindings = user_bindings;
        self.rebuild();
    }

    /// Returns a read-only view of the current user override binding table. Use with `set_user_bindings` for read-modify-write round trips.
    /// Consumers: planned consumer T-AP-06-2.
    pub fn get_user_bindings(&self) -> &HashMap<A, Vec<KeyId>> {
        &self.user_bindings
    }

    /// Returns the complete resolved binding table (effective keys after merging defaults with user overrides), used by the keybinding-hints/footer overview.
    /// Returns a newly cloned `HashMap` to avoid exposing mutable references to the internal cache externally.
    /// Consumer: planned consumer T-AP-11-3 (full keybinding-hints overview).
    pub fn get_resolved_bindings(&self) -> HashMap<A, Vec<KeyId>> {
        self.resolved.clone()
    }
}

/// Parses default binding literals into `KeyId`; the literals form an editing-time static constant table, with validity enforced by the default table tests below,
/// A panic here only indicates that a developer specified an invalid default key name (a developer diagnostic, not a runtime input error).
fn kb(s: &str) -> KeyId {
    KeyId::parse(s)
        .expect("Static default binding literals must be parseable (locked by the editor/select_default_bindings test)")
}

/// Builds the default definition table for the editor action domain (aligned with the tui.editor.*/tui.input.* entries in pi TUI_KEYBINDINGS,
/// The action enum is `EditorAction` from T-TU-01-3).
/// Key values in the table have been normalized by `KeyId::parse` into `KeyId` literals, to be used by T-AP-11-2 as
/// The defaults input for `KeybindingsManager<EditorAction>::new`. Returns all `EditorAction` variants.
/// Current consumers: none (consumed during assembly by T-AP-11-2); this table is also the authoritative source for replacing the inline default keybindings in editor::handle_key.
pub fn editor_default_bindings() -> Vec<KeybindingDefinition<EditorAction>> {
    vec![
        def(EditorAction::MoveCursorUp, &["up"], "Move cursor up"),
        def(EditorAction::MoveCursorDown, &["down"], "Move cursor down"),
        def(
            EditorAction::MoveCursorLeft,
            &["left", "ctrl+b"],
            "Move cursor left",
        ),
        def(
            EditorAction::MoveCursorRight,
            &["right", "ctrl+f"],
            "Move cursor right",
        ),
        def(
            EditorAction::MoveCursorWordLeft,
            &["alt+left", "ctrl+left", "alt+b"],
            "Move cursor word left",
        ),
        def(
            EditorAction::MoveCursorWordRight,
            &["alt+right", "ctrl+right", "alt+f"],
            "Move cursor word right",
        ),
        def(
            EditorAction::MoveToLineStart,
            &["home", "ctrl+a"],
            "Move to line start",
        ),
        def(
            EditorAction::MoveToLineEnd,
            &["end", "ctrl+e"],
            "Move to line end",
        ),
        def(
            EditorAction::JumpForward,
            &["ctrl+]"],
            "Jump forward to character",
        ),
        def(
            EditorAction::JumpBackward,
            &["ctrl+alt+]"],
            "Jump backward to character",
        ),
        def(EditorAction::PageUp, &["pageup"], "Page up"),
        def(EditorAction::PageDown, &["pagedown"], "Page down"),
        def(
            EditorAction::DeleteCharBackward,
            &["backspace"],
            "Delete character backward",
        ),
        def(
            EditorAction::DeleteCharForward,
            &["delete", "ctrl+d"],
            "Delete character forward",
        ),
        def(
            EditorAction::DeleteWordBackward,
            &["ctrl+w", "alt+backspace"],
            "Delete word backward",
        ),
        def(
            EditorAction::DeleteWordForward,
            &["alt+d", "alt+delete"],
            "Delete word forward",
        ),
        def(
            EditorAction::DeleteToLineStart,
            &["ctrl+u"],
            "Delete to line start",
        ),
        def(
            EditorAction::DeleteToLineEnd,
            &["ctrl+k"],
            "Delete to line end",
        ),
        def(EditorAction::Yank, &["ctrl+y"], "Yank"),
        def(EditorAction::YankPop, &["alt+y"], "Yank pop"),
        def(EditorAction::Undo, &["ctrl+-"], "Undo"),
        def(
            EditorAction::NewLine,
            &["shift+enter", "ctrl+j"],
            "Insert newline",
        ),
        def(EditorAction::Submit, &["enter"], "Submit input"),
        def(EditorAction::Tab, &["tab"], "Tab / autocomplete"),
    ]
}

/// Constructs the default definition table for the "select component action domain" (aligned with the tui.select.* entries in pi TUI_KEYBINDINGS,
/// The action enum is `SelectAction` from T-TU-01-4).
/// For use by T-AP-11-2 as the defaults input for `KeybindingsManager<SelectAction>::new`,
/// Replaces the inline default keybindings in SelectList/SettingsList. Returns all four `SelectAction` variants.
/// Current consumers: none (consumed during assembly by T-AP-11-2).
pub fn select_default_bindings() -> Vec<KeybindingDefinition<SelectAction>> {
    vec![
        def(SelectAction::MoveUp, &["up"], "Move selection up"),
        def(SelectAction::MoveDown, &["down"], "Move selection down"),
        def(SelectAction::Confirm, &["enter"], "Confirm selection"),
        // The default pi key is "escape"; the canonical key name in keys.rs is "esc", so use the canonical name
        def(SelectAction::Cancel, &["esc", "ctrl+c"], "Cancel selection"),
    ]
}

/// Default table construction helper: combines an action + static literal key into `KeybindingDefinition`.
fn def<A>(action: A, default_keys: &[&str], description: &str) -> KeybindingDefinition<A> {
    KeybindingDefinition {
        action,
        default_keys: default_keys.iter().map(|s| kb(s)).collect(),
        description: Some(description.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn keys(ids: &[&str]) -> Vec<KeyId> {
        ids.iter().map(|s| KeyId::parse(s).unwrap()).collect()
    }

    fn editor_manager(user: HashMap<EditorAction, Vec<KeyId>>) -> KeybindingsManager<EditorAction> {
        KeybindingsManager::new(editor_default_bindings(), user)
    }

    fn k(s: &str) -> KeyId {
        KeyId::parse(s).unwrap()
    }

    // ── normalize_keys ──

    #[test]
    fn test_normalize_keys_dedups_preserving_order() {
        assert_eq!(
            normalize_keys(Some(vec![k("ctrl+b"), k("ctrl+b"), k("up"), k("ctrl+b")])),
            vec![k("ctrl+b"), k("up")]
        );
    }

    #[test]
    fn test_normalize_keys_none_is_empty() {
        assert!(normalize_keys(None).is_empty());
    }

    // ── new / default bindings ──

    #[test]
    fn test_new_matches_default_keys() {
        let m = editor_manager(HashMap::new());
        assert!(m.matches(&k("ctrl+b"), EditorAction::MoveCursorLeft));
        assert!(m.matches(&k("left"), EditorAction::MoveCursorLeft));
        assert!(!m.matches(&k("ctrl+f"), EditorAction::MoveCursorLeft));
        assert!(m.matches(&k("ctrl+f"), EditorAction::MoveCursorRight));
    }

    #[test]
    fn test_new_no_user_conflicts() {
        // Pure default tables produce no conflicts (pi: default overlaps do not report conflicts).
        assert!(editor_manager(HashMap::new()).get_conflicts().is_empty());
    }

    #[test]
    fn test_unknown_action_outside_defaults_is_ignored() {
        // Simulate an undefined action being overridden with a "pseudo-action" outside the EditorAction domain: a HashMap key = EditorAction cannot
        // represent an undefined action, so this verifies the manager's skip path for invalid keys in user_bindings—
        // With empty defaults, any user key is "undefined" and is ignored entirely.
        let user = HashMap::from([(EditorAction::Submit, keys(&["ctrl+x"]))]);
        let m: KeybindingsManager<EditorAction> = KeybindingsManager::new(Vec::new(), user);
        assert!(m.get_conflicts().is_empty());
        assert!(m.get_resolved_bindings().is_empty());
    }

    // ── get_keys / get_definition ──

    #[test]
    fn test_get_keys_returns_resolved_defaults_in_order() {
        let m = editor_manager(HashMap::new());
        assert_eq!(
            m.get_keys(EditorAction::DeleteWordBackward),
            keys(&["ctrl+w", "alt+backspace"])
        );
        assert_eq!(m.get_keys(EditorAction::Undo), keys(&["ctrl+-"]));
    }

    #[test]
    fn test_get_keys_empty_for_undefined_action() {
        let m: KeybindingsManager<EditorAction> =
            KeybindingsManager::new(Vec::new(), HashMap::new());
        assert!(m.get_keys(EditorAction::Tab).is_empty());
    }

    #[test]
    fn test_get_definition_returns_metadata() {
        let m = editor_manager(HashMap::new());
        let d = m.get_definition(EditorAction::MoveCursorRight).unwrap();
        assert_eq!(d.default_keys, keys(&["right", "ctrl+f"]));
        assert_eq!(d.description.as_deref(), Some("Move cursor right"));
        assert!(m.get_definition(EditorAction::JumpForward).is_some());
    }

    // ── user overrides ──

    #[test]
    fn test_user_override_replaces_default() {
        let mut user = HashMap::new();
        user.insert(EditorAction::MoveCursorLeft, keys(&["ctrl+o"]));
        let m = editor_manager(user);
        assert!(m.matches(&k("ctrl+o"), EditorAction::MoveCursorLeft));
        assert!(!m.matches(&k("left"), EditorAction::MoveCursorLeft)); // The default is replaced.
        assert!(!m.matches(&k("ctrl+b"), EditorAction::MoveCursorLeft));
    }

    #[test]
    fn test_user_override_explicit_unbind() {
        // An empty Vec = explicit unbinding: it takes effect even when a default exists.
        let mut user = HashMap::new();
        user.insert(EditorAction::MoveCursorUp, Vec::new());
        let m = editor_manager(user);
        assert!(!m.matches(&k("up"), EditorAction::MoveCursorUp));
        assert!(m.get_keys(EditorAction::MoveCursorUp).is_empty());
    }

    #[test]
    fn test_user_override_dedup() {
        let mut user = HashMap::new();
        user.insert(EditorAction::Submit, keys(&["enter", "enter", "ctrl+m"]));
        let m = editor_manager(user);
        assert_eq!(m.get_keys(EditorAction::Submit), keys(&["enter", "ctrl+m"]));
    }

    #[test]
    fn test_set_user_bindings_recomputes() {
        let mut m = editor_manager(HashMap::new());
        assert!(m.matches(&k("up"), EditorAction::MoveCursorUp));
        let mut user = HashMap::new();
        user.insert(EditorAction::MoveCursorUp, keys(&["ctrl+n"]));
        m.set_user_bindings(user);
        assert!(m.matches(&k("ctrl+n"), EditorAction::MoveCursorUp));
        assert!(!m.matches(&k("up"), EditorAction::MoveCursorUp));
    }

    #[test]
    fn test_get_user_bindings_roundtrip() {
        let mut user = HashMap::new();
        user.insert(EditorAction::Yank, keys(&["ctrl+y"]));
        let mut m = editor_manager(user);
        assert_eq!(
            m.get_user_bindings()
                .get(&EditorAction::Yank)
                .map(|v| v.clone()),
            Some(keys(&["ctrl+y"]))
        );
        m.set_user_bindings(HashMap::new());
        assert!(m.get_user_bindings().is_empty());
    }

    #[test]
    fn test_get_resolved_bindings_snapshot() {
        let m = editor_manager(HashMap::new());
        let snapshot = m.get_resolved_bindings();
        assert_eq!(snapshot.len(), 24);
        assert_eq!(
            snapshot.get(&EditorAction::MoveCursorLeft).unwrap(),
            &keys(&["left", "ctrl+b"])
        );
    }

    // ── conflict detection ──

    #[test]
    fn test_conflict_detected_for_shared_user_key() {
        // Do the two actions in the Defaults table share up? No. Construct user overrides so both MoveCursorUp and MoveCursorDown bind ctrl+k
        let mut user = HashMap::new();
        user.insert(EditorAction::MoveCursorUp, keys(&["ctrl+k"]));
        user.insert(EditorAction::MoveCursorDown, keys(&["ctrl+k"]));
        let m = editor_manager(user);
        let conflicts = m.get_conflicts();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].key, k("ctrl+k"));
        assert_eq!(conflicts[0].actions.len(), 2);
        assert!(conflicts[0].actions.contains(&EditorAction::MoveCursorUp));
        assert!(conflicts[0].actions.contains(&EditorAction::MoveCursorDown));
        // The conflict does not affect their respective matches (both match)
        assert!(m.matches(&k("ctrl+k"), EditorAction::MoveCursorUp));
        assert!(m.matches(&k("ctrl+k"), EditorAction::MoveCursorDown));
    }

    #[test]
    fn test_conflict_cleared_after_set_user_bindings() {
        let mut m = editor_manager(HashMap::new());
        let mut user = HashMap::new();
        user.insert(EditorAction::Yank, keys(&["ctrl+y"]));
        user.insert(EditorAction::YankPop, keys(&["ctrl+y"]));
        m.set_user_bindings(user);
        assert_eq!(m.get_conflicts().len(), 1);
        m.set_user_bindings(HashMap::new());
        assert!(m.get_conflicts().is_empty());
    }

    #[test]
    fn test_default_overlap_does_not_conflict() {
        // The user did not bind shift+enter; NewLine has shift+enter by default, but no other default action binds the same key → no conflict
        assert!(editor_manager(HashMap::new()).get_conflicts().is_empty());
    }

    // ── Default table integrity ──

    #[test]
    fn test_editor_default_bindings_exhaustive_and_valid() {
        let table = editor_default_bindings();
        // Covers all EditorAction variants (24), with no duplicate actions
        assert_eq!(table.len(), 24);
        let mut seen = std::collections::HashSet::new();
        for d in &table {
            assert!(seen.insert(d.action.clone()), "Duplicate action {:?}", d.action);
            assert!(!d.default_keys.is_empty());
        }
    }

    #[test]
    fn test_select_default_bindings_complete() {
        let table = select_default_bindings();
        assert_eq!(table.len(), 4);
        let actions: Vec<_> = table.iter().map(|d| d.action).collect();
        assert!(actions.contains(&SelectAction::MoveUp));
        assert!(actions.contains(&SelectAction::MoveDown));
        assert!(actions.contains(&SelectAction::Confirm));
        assert!(actions.contains(&SelectAction::Cancel));
        // Cancel default keys: esc + ctrl+c (pi's original "escape" is normalized to "esc")
        let cancel = table
            .iter()
            .find(|d| d.action == SelectAction::Cancel)
            .unwrap();
        assert_eq!(cancel.default_keys, keys(&["esc", "ctrl+c"]));
    }

    // ── Integration with KeyEvent normalization (matches consumes KeyId; KeyId can come from from_key_event)──

    #[test]
    fn test_matches_with_event_derived_key_id() {
        use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
        let event = KeyEvent::new(KeyCode::Char('b'), KeyModifiers::CONTROL);
        let id = KeyId::from_key_event(event).unwrap();
        let m = editor_manager(HashMap::new());
        assert!(m.matches(&id, EditorAction::MoveCursorLeft));
        // Repeat is also normalized via from_key_event to the same KeyId
        let repeat = KeyEvent {
            code: KeyCode::Char('b'),
            modifiers: KeyModifiers::CONTROL,
            kind: KeyEventKind::Repeat,
            state: crossterm::event::KeyEventState::NONE,
        };
        assert!(m.matches(
            &KeyId::from_key_event(repeat).unwrap(),
            EditorAction::MoveCursorLeft
        ));
    }
}
