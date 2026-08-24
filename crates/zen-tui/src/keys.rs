use crossterm::event::{KeyCode, KeyEventKind, KeyModifiers};
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
            KeyEventKind::Press | KeyEventKind::Repeat => Some(KeyId {
                code: event.code,
                modifiers: event.modifiers,
            }),
            KeyEventKind::Release => None,
        }
    }
}

/// Writes a `KeyId` back into a human-readable binding string (`"ctrl+b"`, `"alt+shift+P"`), for keybinding-hints/footer to show the current key to the user.
/// The inverse operation of `parse` (any valid `KeyId` can be written back, but the literal order may be normalized to a canonical modifier order).
/// Implemented via `std::fmt::Display`, joining modifiers with `+` followed by the base key name.
impl Display for KeyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
}
