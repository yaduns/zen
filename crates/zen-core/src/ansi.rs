// REQ-APP-09 (pi-336): ANSI escape handling—strip ANSI escape sequences from text (canonical module)
// Reference reference/pi-original/packages/coding-agent/src/utils/ansi.ts (pi-336,
// chalk strip-ansi port: ansiRegex + stripAnsi + fast path without ESC/0x9B)
// Migration absorbed from inline strip_ansi in crates/zen-app/src/components/mod.rs (T-AP-11-2 transitional implementation,
// 27 call sites); components/mod.rs removes the private implementation and consumes this module's public surface.
// Ecosystem: strip-ansi-escapes (vte parser, ECMA-48, same origin as chalk strip-ansi) provides the stripping mechanism,
// this module keeps only a thin wrapper + pi fast path. Placement exception: REQ-APP-09 → zen-core
// (consumers span the zen-core env/shell_output.rs sanitization pipeline and the zen-app components display layer,
// dependency direction does not allow zen-core→zen-app).
// See tasks/T-AP-09-8-implemented.md for implementation notes

/// Strip ANSI escape sequences (CSI/OSC/DCS, etc.) from text, for the bash streaming output sanitization pipeline (ExecutionEnv sanitize path)
/// and interactive components remove color/cursor control characters before showing process/tool output to users or feeding it into LLM context—colloquially:
/// "decolorize colored text" so plain text reaches the screen/context without terminal-specific noise.
/// Parameter `text` is any valid UTF-8 text (may be lossy-decoded process output); returns a String with escape sequences stripped,
/// result is equivalent to input when there are no escapes. With no ESC (U+001B) or C1 CSI (U+009B), takes the same pi fast path and returns as-is,
/// avoiding unnecessary parsing overhead; otherwise delegates to `strip-ansi-escapes` (vte parser) to strip all VT sequences.
/// Boundary/accepted deviation: C1 CSI (U+009B, including UTF-8 encoding 0xC2 0x9B and raw-byte forms) is, by vte in ground state,
/// discarded as a single-byte execute but **does not enter CSI parsing**—following parameters (e.g. "31m") remain, resulting in half-stripping (locked by measurement; pi's
/// char-level regex strips the whole sequence)—in this pipeline raw C1 bytes are lossy-decoded to U+FFFD rather than U+009B, so the deviation is unreachable,
/// registered as an accepted deviation; isolated/incomplete sequences (e.g. dangling ESC at end of line) follow vte behavior, with unit tests locking actual
/// behavior—residual ESC in the sanitize pipeline is caught and removed by downstream C0 filtering in `sanitize_binary_output`; the component display path
/// (bash_execution streaming cleanup straight to UI) has no such fallback; behavioral differences are locked by their respective unit tests.
pub fn strip_ansi(text: &str) -> String {
    // pi fast path (ansi.ts: stripAnsi): no ESC (7-bit) or C1 CSI (8-bit) introducer means no escapes; return as-is.
    if !text.contains('\u{1b}') && !text.contains('\u{9b}') {
        return text.to_string();
    }
    strip_ansi_escapes::strip_str(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Migrated from the existing 3 assertions in components/bash_execution.rs (T-AP-11-2) ──────

    #[test]
    fn strips_csi_color_sequences() {
        assert_eq!(strip_ansi("a\x1b[31mb\x1b[0mc"), "abc");
    }

    #[test]
    fn strips_osc_until_bel() {
        assert_eq!(strip_ansi("\x1b]0;title\x07x"), "x");
    }

    #[test]
    fn fast_path_returns_plain_unchanged() {
        assert_eq!(strip_ansi("plain"), "plain");
    }

    // ── Known gap fixes (inline implementation had C1 passthrough / ESC\ residual-tail gaps; fixed by the crate) ──

    #[test]
    fn strips_osc_terminated_by_esc_backslash() {
        // OSC terminated by ESC \ ST: the crate strips the entire sequence with no trailing backslash residue (a gap where the inline implementation left a residual tail).
        assert_eq!(strip_ansi("\x1b]0;title\x1b\\x"), "x");
    }

    #[test]
    fn c1_csi_in_utf8_str_is_accepted_deviation() {
        // Accepted deviation (module comment): C1 CSI (U+009B, including UTF-8 encoding 0xC2 0x9B and raw-byte forms) is, by vte,
        // in ground state discarded via execute as a single byte, but **does not enter CSI parsing state**—following "31m" parameters remain (measured
        // locked: unlike pi's char-level regex that strips the whole sequence, C1 sequences are half-stripped). In this pipeline raw C1 bytes are lossy
        // decoded to U+FFFD rather than U+009B; the deviation is unreachable and registered as accepted.
        assert_eq!(strip_ansi("\u{9b}31mred"), "31mred");
    }

    // ── Chinese / multibyte fidelity and interleaved multiple sequences ────────────────────────────────

    #[test]
    fn preserves_multibyte_utf8_around_sequences() {
        assert_eq!(strip_ansi("\x1b[31m你好\x1b[0m"), "你好");
        assert_eq!(strip_ansi("\x1b[1m中文\x1b[0m 🙂"), "中文 🙂");
    }

    #[test]
    fn strips_interleaved_multiple_sequences() {
        assert_eq!(
            strip_ansi("\x1b[32mgreen\x1b[m \x1b[1mbold\x1b[0m!"),
            "green bold!"
        );
    }

    #[test]
    fn strips_dcs_and_other_private_sequences() {
        // DCS sequences (ESC P ... ESC \): vte covers a sequence family beyond pi regex.
        assert_eq!(strip_ansi("a\x1bP~>G\x1b\\b"), "ab");
        // CSI with semicolon parameters and equivalent 8-bit terminator forms.
        assert_eq!(strip_ansi("\x1b[38;2;255;0;0mred\x1b[0m"), "red");
    }

    #[test]
    fn empty_string_stays_empty() {
        assert_eq!(strip_ansi(""), "");
    }

    #[test]
    fn dangling_escape_behavior_is_locked() {
        // Isolated/incomplete sequences (dangling ESC at end of line) behavior is locked by unit tests (component path has no downstream C0 fallback; see module comment).
        // Measured vte behavior: dangling ESC is consumed and discarded (not printed or retained); unterminated OSC (no BEL/ESC\/0x9C)
        // the entire sequence is discarded.
        assert_eq!(strip_ansi("ab\x1b"), "ab");
        assert_eq!(strip_ansi("\x1b]0;title"), "");
    }
}
