use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use std::fmt::Display;

/// Normalized identity of a single keypress — wraps crossterm's KeyCode (base key, including char/function/navigation keys) and KeyModifiers bit flags (modifiers);
/// it is the key used by the shortcut registry (T-TU-03-1) for equality matching, the literal writable in the default binding table, and the source displayed by keybinding-hints.
/// Strips the kind/state noise from a crossterm KeyEvent (keeping only Press semantics: code+modifiers), making equality checks stable.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct KeyId {
    code: crossterm::event::KeyCode,
    modifiers: crossterm::event::KeyModifiers,
}

impl KeyId {
    /// Quick check whether a crossterm event matches a given `KeyId` (composition of `KeyId::from_key_event` + equality comparison).
    /// Lets editor/component key routing decide directly when the target `KeyId` is known, skipping the manual normalize/compare two steps.
    /// Consumers: planned consumer `editor::handle_key` (T-TU-01-3), component `handle_key` — equivalent to registry `matches`; use either one.
    pub fn parse(s: &str) -> Option<KeyId> {
        let parts: Vec<&str> = s.split('+').collect();
        if parts.is_empty() {
            return None;
        }

        let (mods, code_parts) = parts.split_at(parts.len() - 1);
        let code_str = code_parts[0];

        let mut modifiers = KeyModifiers::empty();
        for m in mods {
            match m.to_ascii_lowercase().as_str() {
                "ctrl" => modifiers |= KeyModifiers::CONTROL,
                "alt" => modifiers |= KeyModifiers::ALT,
                "shift" => modifiers |= KeyModifiers::SHIFT,
                "super" => modifiers |= KeyModifiers::SUPER,
                _ => return None,
            }
        }

        let code = match code_str.to_ascii_lowercase().as_str() {
            "enter" => KeyCode::Enter,
            "esc" => KeyCode::Esc,
            // Shift+Tab: crossterm normalizes it to a standalone KeyCode::BackTab (not Tab+SHIFT),
            // parsing "tab" with SHIFT yields BackTab directly, so the event side can match it (T-TU-02-1-fix).
            "tab" if modifiers.contains(KeyModifiers::SHIFT) => KeyCode::BackTab,
            "tab" => KeyCode::Tab,
            "backspace" => KeyCode::Backspace,
            "delete" => KeyCode::Delete,
            "up" => KeyCode::Up,
            "down" => KeyCode::Down,
            "left" => KeyCode::Left,
            "right" => KeyCode::Right,
            "home" => KeyCode::Home,
            "end" => KeyCode::End,
            "pageup" => KeyCode::PageUp,
            "pagedown" => KeyCode::PageDown,
            "insert" => KeyCode::Insert,
            "space" => KeyCode::Char(' '),
            "null" => KeyCode::Null,
            s if s.len() > 1 && s.starts_with('f') => {
                let n: u8 = s[1..].parse().ok()?;
                if (1..=12).contains(&n) {
                    KeyCode::F(n)
                } else {
                    return None;
                }
            }
            _ => {
                if code_str.len() == 1 {
                    let c = code_str.chars().next()?;
                    if modifiers.contains(KeyModifiers::SHIFT) {
                        KeyCode::Char(c.to_ascii_uppercase())
                    } else if c.is_ascii_uppercase() {
                        modifiers |= KeyModifiers::SHIFT;
                        KeyCode::Char(c)
                    } else {
                        KeyCode::Char(c)
                    }
                } else {
                    return None;
                }
            }
        };

        Some(KeyId { code, modifiers })
    }

    /// Quick check whether a crossterm event matches a given `KeyId` (composition of `KeyId::from_key_event` + equality comparison).
    /// Lets editor/component key routing decide directly when the target `KeyId` is known, skipping the manual normalize/compare two steps.
    /// Consumers: planned consumer `editor::handle_key` (T-TU-01-3), component `handle_key` — equivalent to registry `matches`; use either one.
    pub fn matches(event: crossterm::event::KeyEvent, key_id: &KeyId) -> bool {
        Self::from_key_event(event).as_ref() == Some(key_id)
    }

    /// Normalizes one physical keypress (crossterm `KeyEvent`) into a `KeyId` for equality matching in the registry.
    /// Called by the editor/main loop when a `KeyEvent` arrives (only accepts `KeyEventKind::Press`; `Repeat` is normalized as Press, `Release` returns `None`),
    /// the normalized result is the key for `KeybindingsManager::matches`. `None` means the event cannot be expressed as a nameable key (e.g. `Release`),
    /// in which case the caller should ignore it and wait for the next event. Boundary: relies on crossterm's already-normalized `KeyEvent` (uppercase ⇔ SHIFT invariant, legacy/kitty resolved).
    /// Consumers: planned consumers `editor::handle_key` (T-TU-01-3), autocomplete (T-TU-03-3) — both convert events to `KeyId` uniformly via this function.
    pub fn from_key_event(event: crossterm::event::KeyEvent) -> Option<KeyId> {
        match event.kind {
            KeyEventKind::Press | KeyEventKind::Repeat => {
                let mut modifiers = event.modifiers;
                // Shift+Tab is a standalone KeyCode::BackTab in crossterm; on some terminals/versions the event modifier bit
                // is missing, so SHIFT is added uniformly to match the "shift+tab" binding (parse produces BackTab+SHIFT).
                if event.code == KeyCode::BackTab {
                    modifiers |= KeyModifiers::SHIFT;
                }
                Some(KeyId {
                    code: event.code,
                    modifiers,
                })
            }
            KeyEventKind::Release => None,
        }
    }
}

/// Apple Terminal normalization filter (aligned with pi `normalizeAppleTerminalInput`):
///
/// Only when `is_apple_terminal` and the event is **unmodified Enter** (Press with no Ctrl/Alt/Shift/Super
/// modifier) and native Shift is held, rewrite the event as `Enter+SHIFT`—the editor's existing mapping
/// (`Shift+Enter → NewLine`) then inserts a newline; all other events are returned unchanged.
///
/// Pure function: `shift_pressed` is injected by the caller (see [`is_apple_terminal_session`] and
/// [`native_shift_pressed`]), making unit testing easier. Background: Apple Terminal does not send CSI-u enhanced sequences,
/// Shift+Enter sends only a bare `\r`; this function rewrites the padded-back Shift modifier state into the event, equivalent to pi rewriting
/// the bare `\r` as `\x1b[13;2u`.
/// See reference/pi-original/packages/tui/src/terminal.ts `normalizeAppleTerminalInput`.
pub fn normalize_apple_terminal_enter(
    key: KeyEvent,
    is_apple_terminal: bool,
    shift_pressed: bool,
) -> KeyEvent {
    if is_apple_terminal
        && shift_pressed
        && key.kind == KeyEventKind::Press
        && key.code == KeyCode::Enter
        && key.modifiers.is_empty()
    {
        KeyEvent {
            code: KeyCode::Enter,
            modifiers: KeyModifiers::SHIFT,
            kind: key.kind,
            state: key.state,
        }
    } else {
        key
    }
}

/// Whether this is an Apple Terminal session (aligned with pi `isAppleTerminalSession`):
/// macOS and `TERM_PROGRAM=Apple_Terminal`. Always false on non-macOS.
/// Consumer: T-AP-11-1-fix (evaluated once outside the interactive `on_event` closure and cached).
pub fn is_apple_terminal_session() -> bool {
    cfg!(target_os = "macos") && std::env::var("TERM_PROGRAM").as_deref() == Ok("Apple_Terminal")
}

/// Queries the local Shift modifier pressed state. Only macOS has a real implementation (Carbon FFI, via
/// [`crate::native_modifiers::native_shift_pressed`] queries `GetCurrentKeyModifiers`,
/// the maskShift bit covers left/right Shift); always false on non-macOS
/// (aligned with pi native helper load-failure fallback to false; see native-modifiers.ts).
/// Consumer: T-AP-11-1-fix (Apple Terminal normalization filter inside interactive `on_event`).
#[cfg(target_os = "macos")]
pub fn native_shift_pressed() -> bool {
    crate::native_modifiers::native_shift_pressed()
}

/// Non-macOS always-false placeholder (callers need no cfg branch).
#[cfg(not(target_os = "macos"))]
pub fn native_shift_pressed() -> bool {
    false
}

/// Writes a `KeyId` back into a human-readable binding string (`"ctrl+b"`, `"alt+shift+P"`), for keybinding-hints/footer to show the current key to the user.
/// The inverse operation of `parse` (any valid `KeyId` can be written back, but the literal order may be normalized to a canonical modifier order).
/// Implemented via `std::fmt::Display`, joining modifiers with `+` followed by the base key name.
impl Display for KeyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // BackTab is shift+tab: written out in full to avoid duplicating "shift+" in the prefix loop.
        if self.code == KeyCode::BackTab {
            return write!(f, "shift+tab");
        }
        if self.modifiers.contains(KeyModifiers::CONTROL) {
            write!(f, "ctrl+")?;
        }
        if self.modifiers.contains(KeyModifiers::ALT) {
            write!(f, "alt+")?;
        }
        if self.modifiers.contains(KeyModifiers::SHIFT) {
            write!(f, "shift+")?;
        }
        if self.modifiers.contains(KeyModifiers::SUPER) {
            write!(f, "super+")?;
        }

        match self.code {
            KeyCode::Char(' ') => write!(f, "space"),
            KeyCode::Char(c) => write!(f, "{}", c),
            KeyCode::Enter => write!(f, "enter"),
            KeyCode::Esc => write!(f, "esc"),
            KeyCode::Tab => write!(f, "tab"),
            KeyCode::Backspace => write!(f, "backspace"),
            KeyCode::Delete => write!(f, "delete"),
            KeyCode::Up => write!(f, "up"),
            KeyCode::Down => write!(f, "down"),
            KeyCode::Left => write!(f, "left"),
            KeyCode::Right => write!(f, "right"),
            KeyCode::Home => write!(f, "home"),
            KeyCode::End => write!(f, "end"),
            KeyCode::PageUp => write!(f, "pageup"),
            KeyCode::PageDown => write!(f, "pagedown"),
            KeyCode::Insert => write!(f, "insert"),
            KeyCode::F(n) => write!(f, "f{}", n),
            KeyCode::Null => write!(f, "null"),
            _ => write!(f, "?"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyEvent, KeyEventState};

    /// Construct a KeyEvent with the given kind/state (defaults to Press, NONE state).
    fn event(code: KeyCode, mods: KeyModifiers, kind: KeyEventKind) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: mods,
            kind,
            state: KeyEventState::NONE,
        }
    }

    fn press(code: KeyCode, mods: KeyModifiers) -> KeyEvent {
        event(code, mods, KeyEventKind::Press)
    }

    // ── from_key_event ──

    #[test]
    fn test_from_key_event_press_yields_key_id() {
        let k = KeyId::from_key_event(press(KeyCode::Char('b'), KeyModifiers::CONTROL)).unwrap();
        assert_eq!(k.code, KeyCode::Char('b'));
        assert_eq!(k.modifiers, KeyModifiers::CONTROL);
    }

    #[test]
    fn test_from_key_event_repeat_treated_as_press() {
        let k = KeyId::from_key_event(event(KeyCode::Up, KeyModifiers::NONE, KeyEventKind::Repeat));
        assert!(k.is_some());
        assert_eq!(k.unwrap().code, KeyCode::Up);
    }

    #[test]
    fn test_from_key_event_release_is_none() {
        let k = KeyId::from_key_event(event(
            KeyCode::Char('x'),
            KeyModifiers::NONE,
            KeyEventKind::Release,
        ));
        assert!(k.is_none());
    }

    #[test]
    fn test_from_key_event_preserves_modifiers() {
        let mods = KeyModifiers::CONTROL | KeyModifiers::SHIFT;
        let k = KeyId::from_key_event(press(KeyCode::Char('p'), mods)).unwrap();
        assert_eq!(k.modifiers, mods);
    }

    // ── parse: single characters and modifiers ──

    #[test]
    fn test_parse_plain_char_no_modifiers() {
        assert_eq!(
            KeyId::parse("a"),
            Some(KeyId {
                code: KeyCode::Char('a'),
                modifiers: KeyModifiers::NONE
            })
        );
    }

    #[test]
    fn test_parse_ctrl_char() {
        assert_eq!(
            KeyId::parse("ctrl+b"),
            Some(KeyId {
                code: KeyCode::Char('b'),
                modifiers: KeyModifiers::CONTROL
            })
        );
    }

    #[test]
    fn test_parse_alt_char() {
        assert_eq!(
            KeyId::parse("alt+f"),
            Some(KeyId {
                code: KeyCode::Char('f'),
                modifiers: KeyModifiers::ALT
            })
        );
    }

    #[test]
    fn test_parse_combined_modifiers() {
        assert_eq!(
            KeyId::parse("ctrl+alt+]"),
            Some(KeyId {
                code: KeyCode::Char(']'),
                modifiers: KeyModifiers::CONTROL | KeyModifiers::ALT,
            })
        );
    }

    #[test]
    fn test_parse_shift_normalizes_char_to_uppercase() {
        // Follow crossterm's 'uppercase Char ⇔ SHIFT' invariant: shift+lowercase → uppercase+SHIFT
        assert_eq!(
            KeyId::parse("shift+a"),
            Some(KeyId {
                code: KeyCode::Char('A'),
                modifiers: KeyModifiers::SHIFT
            })
        );
    }

    #[test]
    fn test_parse_uppercase_char_gains_shift() {
        // Follow crossterm's 'uppercase Char ⇔ SHIFT' invariant: uppercase chars get SHIFT added automatically
        assert_eq!(
            KeyId::parse("A"),
            Some(KeyId {
                code: KeyCode::Char('A'),
                modifiers: KeyModifiers::SHIFT
            })
        );
    }

    #[test]
    fn test_parse_super_modifier() {
        assert_eq!(
            KeyId::parse("super+k").unwrap().modifiers,
            KeyModifiers::SUPER
        );
    }

    #[test]
    fn test_parse_is_case_insensitive_on_modifiers() {
        // Modifier names are case-insensitive (use lowercase chars to avoid interference from uppercase→SHIFT)
        assert_eq!(KeyId::parse("CTRL+b"), KeyId::parse("ctrl+b"));
        assert_eq!(KeyId::parse("Ctrl+b"), KeyId::parse("ctrl+b"));
        assert_eq!(KeyId::parse("ALT+f"), KeyId::parse("alt+f"));
        // Uppercase chars trigger SHIFT normalization: CTRL+B ≡ ctrl+shift+b
        assert_eq!(KeyId::parse("CTRL+B"), KeyId::parse("ctrl+shift+b"));
    }

    #[test]
    fn test_parse_shift_modifier() {
        assert_eq!(
            KeyId::parse("shift+p").unwrap().modifiers,
            KeyModifiers::SHIFT
        );
    }

    #[test]
    fn test_parse_space_with_modifier() {
        assert_eq!(
            KeyId::parse("ctrl+space"),
            Some(KeyId {
                code: KeyCode::Char(' '),
                modifiers: KeyModifiers::CONTROL
            })
        );
    }

    // ── parse: special key names ──

    #[test]
    fn test_parse_special_keys() {
        let cases = [
            ("enter", KeyCode::Enter),
            ("esc", KeyCode::Esc),
            ("tab", KeyCode::Tab),
            ("backspace", KeyCode::Backspace),
            ("delete", KeyCode::Delete),
            ("up", KeyCode::Up),
            ("down", KeyCode::Down),
            ("left", KeyCode::Left),
            ("right", KeyCode::Right),
            ("home", KeyCode::Home),
            ("end", KeyCode::End),
            ("pageup", KeyCode::PageUp),
            ("pagedown", KeyCode::PageDown),
            ("insert", KeyCode::Insert),
            ("null", KeyCode::Null),
        ];
        for (s, code) in cases {
            assert_eq!(
                KeyId::parse(s),
                Some(KeyId {
                    code,
                    modifiers: KeyModifiers::NONE
                }),
                "case {s:?}"
            );
        }
    }

    #[test]
    fn test_parse_shift_tab_is_backtab() {
        // shift+tab → BackTab+SHIFT (same shape as the crossterm event side, so the binding matches)
        assert_eq!(
            KeyId::parse("shift+tab"),
            Some(KeyId {
                code: KeyCode::BackTab,
                modifiers: KeyModifiers::SHIFT
            })
        );
        // plain tab without modifiers is unaffected
        assert_eq!(
            KeyId::parse("tab"),
            Some(KeyId {
                code: KeyCode::Tab,
                modifiers: KeyModifiers::NONE
            })
        );
    }

    #[test]
    fn test_backtab_display_and_event_match_roundtrip() {
        // Display write-back
        let id = KeyId::parse("shift+tab").unwrap();
        assert_eq!(id.to_string(), "shift+tab");
        // real event (BackTab, terminal form with the modifier bit missing) → from_key_event normalizes by adding SHIFT → binding matches
        let ev = KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE);
        assert!(KeyId::matches(ev, &id), "BackTab event should match the shift+tab binding");
        // BackTab with the SHIFT modifier also matches
        let ev2 = KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT);
        assert!(KeyId::matches(ev2, &id));
    }

    #[test]
    fn test_parse_letter_f_is_not_function_key() {
        // Letter f alone is a key by itself (the f-key branch should only match f1..f12)
        assert_eq!(
            KeyId::parse("f"),
            Some(KeyId {
                code: KeyCode::Char('f'),
                modifiers: KeyModifiers::NONE
            })
        );
        assert_eq!(
            KeyId::parse("ctrl+f"),
            Some(KeyId {
                code: KeyCode::Char('f'),
                modifiers: KeyModifiers::CONTROL
            })
        );
    }

    #[test]
    fn test_parse_space_key() {
        assert_eq!(
            KeyId::parse("space"),
            Some(KeyId {
                code: KeyCode::Char(' '),
                modifiers: KeyModifiers::NONE
            })
        );
    }

    #[test]
    fn test_parse_function_keys_1_to_12() {
        for n in 1..=12u8 {
            let s = format!("f{n}");
            assert_eq!(
                KeyId::parse(&s),
                Some(KeyId {
                    code: KeyCode::F(n),
                    modifiers: KeyModifiers::NONE
                }),
                "case {s:?}"
            );
        }
    }

    #[test]
    fn test_parse_function_key_case_insensitive() {
        assert_eq!(KeyId::parse("F5"), KeyId::parse("f5"));
    }

    #[test]
    fn test_parse_page_keys_with_modifier() {
        assert_eq!(
            KeyId::parse("ctrl+pageup"),
            Some(KeyId {
                code: KeyCode::PageUp,
                modifiers: KeyModifiers::CONTROL
            })
        );
    }

    // ── parse: invalid input ──

    #[test]
    fn test_parse_rejects_empty_string() {
        assert!(KeyId::parse("").is_none());
    }

    #[test]
    fn test_parse_rejects_unknown_key_name() {
        assert!(KeyId::parse("unknownkey").is_none());
        assert!(KeyId::parse("longkeyname").is_none());
    }

    #[test]
    fn test_parse_rejects_unknown_modifier() {
        assert!(KeyId::parse("meta+x").is_none());
        assert!(KeyId::parse("hyper+x").is_none());
    }

    #[test]
    fn test_parse_rejects_function_key_out_of_range() {
        assert!(KeyId::parse("f0").is_none());
        assert!(KeyId::parse("f13").is_none());
    }

    #[test]
    fn test_parse_rejects_multi_char_non_key() {
        assert!(KeyId::parse("xy").is_none());
    }

    // ── matches ──

    #[test]
    fn test_matches_press_hit() {
        let id = KeyId::parse("ctrl+b").unwrap();
        assert!(KeyId::matches(
            press(KeyCode::Char('b'), KeyModifiers::CONTROL),
            &id
        ));
    }

    #[test]
    fn test_matches_repeat_hit() {
        let id = KeyId::parse("up").unwrap();
        assert!(KeyId::matches(
            event(KeyCode::Up, KeyModifiers::NONE, KeyEventKind::Repeat),
            &id
        ));
    }

    #[test]
    fn test_matches_modifier_mismatch_misses() {
        let id = KeyId::parse("ctrl+b").unwrap();
        assert!(!KeyId::matches(
            press(KeyCode::Char('b'), KeyModifiers::ALT),
            &id
        ));
        assert!(!KeyId::matches(
            press(KeyCode::Char('b'), KeyModifiers::NONE),
            &id
        ));
    }

    #[test]
    fn test_matches_code_mismatch_misses() {
        let id = KeyId::parse("ctrl+b").unwrap();
        assert!(!KeyId::matches(
            press(KeyCode::Char('c'), KeyModifiers::CONTROL),
            &id
        ));
    }

    #[test]
    fn test_matches_release_misses() {
        let id = KeyId::parse("x").unwrap();
        assert!(!KeyId::matches(
            event(
                KeyCode::Char('x'),
                KeyModifiers::NONE,
                KeyEventKind::Release
            ),
            &id
        ));
    }

    // ── Display ──

    #[test]
    fn test_display_plain_char() {
        assert_eq!(format!("{}", KeyId::parse("a").unwrap()), "a");
    }
    #[test]
    fn test_display_modifier_order_is_ctrl_alt_shift() {
        // SHIFT normalization uppercases the char, displayed as "P"
        assert_eq!(
            format!("{}", KeyId::parse("alt+shift+ctrl+p").unwrap()),
            "ctrl+alt+shift+P"
        );
    }

    #[test]
    fn test_display_special_keys() {
        assert_eq!(format!("{}", KeyId::parse("enter").unwrap()), "enter");
        assert_eq!(format!("{}", KeyId::parse("space").unwrap()), "space");
        assert_eq!(format!("{}", KeyId::parse("pageup").unwrap()), "pageup");
        assert_eq!(format!("{}", KeyId::parse("f5").unwrap()), "f5");
        assert_eq!(
            format!("{}", KeyId::parse("backspace").unwrap()),
            "backspace"
        );
    }

    #[test]
    fn test_display_ctrl_combo() {
        assert_eq!(format!("{}", KeyId::parse("ctrl+b").unwrap()), "ctrl+b");
    }

    // ── normalize_apple_terminal_enter ──

    /// Bare Enter (Press, no modifiers) + Apple session + Shift held → rewrite to Enter+SHIFT
    #[test]
    fn test_normalize_apple_terminal_enter_rewrites_bare_enter() {
        let bare = event(KeyCode::Enter, KeyModifiers::NONE, KeyEventKind::Press);
        let out = normalize_apple_terminal_enter(bare, true, true);
        assert_eq!(out.code, KeyCode::Enter);
        assert_eq!(out.modifiers, KeyModifiers::SHIFT);
        assert_eq!(out.kind, KeyEventKind::Press);
    }

    /// Bare Enter but Shift not held → pass through unchanged (preserve submit semantics)
    #[test]
    fn test_normalize_apple_terminal_enter_no_shift_passthrough() {
        let bare = event(KeyCode::Enter, KeyModifiers::NONE, KeyEventKind::Press);
        let out = normalize_apple_terminal_enter(bare, true, false);
        assert_eq!(out.modifiers, KeyModifiers::NONE);
    }

    /// Non-Apple session stays unchanged even if Shift is held (no rewrite; handled by CSI-u or the editing layer)
    #[test]
    fn test_normalize_non_apple_passthrough() {
        let bare = event(KeyCode::Enter, KeyModifiers::NONE, KeyEventKind::Press);
        let out = normalize_apple_terminal_enter(bare, false, true);
        assert_eq!(out.modifiers, KeyModifiers::NONE);
    }

    /// Enter with modifiers (Ctrl/Alt, etc.) is not rewritten—rewriting only compensates for the missing Shift on bare \r
    #[test]
    fn test_normalize_ignores_modified_enter() {
        for mods in [
            KeyModifiers::CONTROL,
            KeyModifiers::ALT,
            KeyModifiers::SHIFT,
            KeyModifiers::CONTROL | KeyModifiers::SHIFT,
        ] {
            let k = event(KeyCode::Enter, mods, KeyEventKind::Press);
            let out = normalize_apple_terminal_enter(k, true, true);
            assert_eq!(out.modifiers, mods, "Modifiers {mods:?} should be preserved as-is");
        }
    }

    /// Non-Enter keys are not rewritten
    #[test]
    fn test_normalize_non_enter_passthrough() {
        let k = event(KeyCode::Char('x'), KeyModifiers::NONE, KeyEventKind::Press);
        let out = normalize_apple_terminal_enter(k, true, true);
        assert_eq!(out.code, KeyCode::Char('x'));
        assert_eq!(out.modifiers, KeyModifiers::NONE);
    }

    /// Release events are not rewritten (defensive: even if they slip in, no SHIFT variant is produced)
    #[test]
    fn test_normalize_ignores_release_event() {
        let k = event(KeyCode::Enter, KeyModifiers::NONE, KeyEventKind::Release);
        let out = normalize_apple_terminal_enter(k, true, true);
        assert_eq!(out.kind, KeyEventKind::Release);
        assert_eq!(out.modifiers, KeyModifiers::NONE);
    }

    // ── is_apple_terminal_session / native_shift_pressed (shape smoke) ──

    #[test]
    fn test_is_apple_terminal_session_returns_bool() {
        // Environment-dependent (TERM_PROGRAM): only assert true/false shape, do not lock the value
        let _v = is_apple_terminal_session();
    }

    #[test]
    fn test_native_shift_pressed_returns_bool() {
        // Always false on non-macOS; macOS queries real key state (CI has no keyboard context, so do not lock the value)
        let v = native_shift_pressed();
        assert!(v == true || v == false);
    }
}
