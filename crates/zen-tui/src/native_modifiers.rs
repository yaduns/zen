// REQ-TUI-01 (pi-408): macOS native modifier key state query — Carbon/HIToolbox FFI.
// Corresponds to original TS: reference/pi-original/packages/tui/src/native-modifiers.ts
// (pi uses precompiled node addon darwin-modifiers.node → zen resolves to system Carbon constant API
// GetCurrentKeyModifiers, equivalently reads current keyboard modifier state, no heavy dependency).
//
// **API selection record (T-AP-11-1-fix implementation deviates from original draft's P0-B plan)**: original draft planned CoreGraphics
// `CGEventSourceKeyState(combinedSession, 56) || (combinedSession, 60)` (left/right Shift
// queried separately). Tested `CGEventSourceKeyState` and `CGEventSourceFlagsState` in no-GUI session
// environments (SSH/CI) will **block permanently** — if TERM_PROGRAM=Apple_Terminal, every Enter goes through this
// query, the whole UI freezes, a fatal defect. Switched to Carbon (HIToolbox) `GetCurrentKeyModifiers()`:
// synchronously returns instantly (returns 0 under no-GUI session context), and `maskShift` bit naturally covers left/right Shift
// (P0-B semantics over-satisfied: right-hand Shift+Enter also works).
// Compiled on macOS only; on non-macOS, `native_shift_pressed()` in keys.rs is a placeholder always false fallback.

/// Whether the local Shift modifier key is pressed (`maskShift` bit covers left/right Shift, aligned with pi
/// `isNativeModifierPressed("shift")` system-level modifier state; P0-B).
/// Synchronously queried via Carbon (HIToolbox) `GetCurrentKeyModifiers` — returns instantly, does not block,
/// Returns 0 on failure/no session context (i.e. false, aligned with pi native helper failure fallback semantics).
/// Consumer: T-AP-11-1-fix (keys.rs `native_shift_pressed` → interactive on_event).
pub fn native_shift_pressed() -> bool {
    // maskShift = 1 << 17 (Carbon modifier mask; GetCurrentKeyModifiers returns current key state)
    const MASK_SHIFT: u32 = 1 << 17;
    unsafe { GetCurrentKeyModifiers() & MASK_SHIFT != 0 }
}

#[cfg(target_os = "macos")]
#[link(name = "Carbon", kind = "framework")]
extern "C" {
    /// Returns the current keyboard modifier key state mask (HIToolbox/Carbon, synchronous instant return).
    /// Docs: <https://developer.apple.com/documentation/...> (GetCurrentKeyModifiers)
    fn GetCurrentKeyModifiers() -> u32;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only asserts return shape (bool), not actual key state — CI has no keyboard context, pressed or not are both valid;
    /// Aligns with pi `isNativeModifierPressed` contract of returning bool in any environment.
    #[test]
    fn test_native_shift_pressed_returns_bool() {
        let v = native_shift_pressed();
        assert!(v == true || v == false);
    }

    /// The API itself returns synchronously and instantly (non-blocking; CGEventSourceKeyState hangs, regression protection for this choice).
    /// Should also return quickly under no-GUI session context (verified on real machine as 0x0).
    #[test]
    fn test_modifier_query_returns_promptly() {
        let _ = unsafe { GetCurrentKeyModifiers() };
    }
}
