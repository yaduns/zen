use std::io::IsTerminal;
use tokio::io::AsyncReadExt;
use zen_ai::types::ImageContent;

/// Complete output of initial message construction — the text and image attachments of the "session's first user message", plus the remaining subsequent prompts after the consumed first message
pub struct InitialMessageResult<'a> {
    /// First prompt after merging the three input sources; None when all three are empty
    pub initial_message: Option<String>,
    /// Image attachments from @file; empty Vec = no images (Phase 1 always empty, filled by T-AP-08-4)
    pub initial_images: Vec<ImageContent>,
    /// Remaining CLI messages other than the consumed first one, in original order; empty Vec when there are no messages/only one message
    pub remaining_messages: Vec<&'a str>,
}

/// Merge stdin content, @file text, and the first CLI message into the session initial prompt, and return the remaining messages.
/// Corresponds to pi `buildInitialMessage` (`parts = [stdin?, fileText?, messages[0]?]` then `parts.join("")`).
///
/// # Implicit constraints (derived from original behavior, must be preserved)
/// - The three input sources are **concatenated directly, with no separator** (e.g. stdin="foo" + file="bar" → "foobar"); do not add newlines/separators.
/// - Only stdin content is trimmed on read (see `read_piped_stdin`); file text and messages are **not trimmed**.
/// - **Only the first** entry in `messages` is consumed into initial_message; the rest are returned as-is in original order in
///   `InitialMessageResult::remaining_messages` (aligned with pi's `parsed.messages.shift()` side effect — Rust uses an explicit return value instead of in-place mutation).
/// - `file_text`/`file_images` participate in merging only when non-empty (aligned with pi `if (fileText)` / `if (fileImages.length > 0)`).
///   This task only defines the shape; real injection is filled by T-AP-08-4 (@file handling, Phase 2); Phase 1 passes `None`/empty Vec.
///
/// # Consumers
/// T-AP-08-3 (main: combines stdin + (from Phase 2) file handling results before calling), T-AP-07-1 (print: obtains the result via the orchestration function).
pub fn build_initial_message<'a>(
    messages: &'a [&str],
    stdin_content: Option<String>,
    file_text: Option<String>,
    file_images: Vec<ImageContent>,
) -> InitialMessageResult<'a> {
    // collect borrows of the non-empty parts
    let mut parts: Vec<&str> = Vec::new();

    // stdin (already trimmed in read_piped_stdin)
    if let Some(ref s) = stdin_content {
        if !s.is_empty() {
            parts.push(s);
        }
    }

    // file_text (effective from Phase 2; Phase 1 passes None)
    if let Some(ref s) = file_text {
        if !s.is_empty() {
            parts.push(s);
        }
    }

    // messages[0] (only the first is consumed)
    if !messages.is_empty() {
        parts.push(messages[0]);
    }

    // direct concatenation, no separator
    let initial_message = if parts.is_empty() {
        None
    } else {
        Some(parts.concat())
    };

    InitialMessageResult {
        initial_message,
        initial_images: file_images,
        // everything from the second onward is kept; empty when there are no messages/only one
        remaining_messages: messages.iter().skip(1).copied().collect(),
    }
}

/// Read all of piped stdin and normalize it (trim whitespace at both ends; return None if empty/whitespace-only).
/// Corresponds to pi `readPipedStdin` (`main.ts`): when `process.stdin.isTTY`, do not read and return undefined;
/// otherwise accumulate data until end, then `data.trim() || undefined`.
///
/// # Edge Cases
/// - stdin is a TTY (interactive terminal) → return `Ok(None)`, **do not read** (avoid blocking on EOF).
/// - Non-TTY but content is empty or whitespace-only → `Ok(None)`.
/// - Non-UTF-8 bytes → replaced via `String::from_utf8_lossy`.
/// - IO errors (read failure, fd closed, etc.) → returned as `std::io::Error`, aggregated by caller (main, anyhow).
/// - Note: this function **reads only once**; after reading, even if called again it is semantically treated as empty; caller (T-AP-08-3) calls it only once.
///
pub async fn read_piped_stdin() -> std::io::Result<Option<String>> {
    if stdin_is_terminal() {
        return Ok(None);
    }

    let mut raw = Vec::new();
    tokio::io::stdin().read_to_end(&mut raw).await?;
    let buf = String::from_utf8_lossy(&raw).into_owned();

    Ok(normalize_stdin(buf))
}

/// Whether the current process stdin is a terminal (TTY).
/// Uses `std::io::stdin().is_terminal()` under the hood (`std::io::IsTerminal`, Rust 1.70+).
pub fn stdin_is_terminal() -> bool {
    std::io::stdin().is_terminal()
}

/// Normalize raw piped stdin content: trim whitespace at both ends; return None if empty/whitespace-only.
/// Extracted as a private pure function so this check can be unit-tested without a real fd.
///
/// # Edge Cases
/// - Trim only both ends (Unicode whitespace, `str::trim`); preserve interior whitespace/newlines.
/// - Empty string or all whitespace → `None` (aligned with pi empty-value semantics).
///
/// # Visibility
/// private: only called internally by `read_piped_stdin` in this file; not exported.
fn normalize_stdin(raw: String) -> Option<String> {
    let trimmed = raw.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(data: &str) -> ImageContent {
        ImageContent {
            data: data.into(),
            mime_type: "image/png".into(),
        }
    }

    // ── build_initial_message ────────────────────────────────

    #[test]
    fn test_build_initial_message_all_empty_returns_none() {
        let r = build_initial_message(&[], None, None, vec![]);
        assert_eq!(r.initial_message, None);
        assert!(r.initial_images.is_empty());
        assert!(r.remaining_messages.is_empty());
    }

    #[test]
    fn test_build_initial_message_stdin_only() {
        let r = build_initial_message(&[], Some("hello".into()), None, vec![]);
        assert_eq!(r.initial_message.as_deref(), Some("hello"));
        assert!(r.remaining_messages.is_empty());
    }

    #[test]
    fn test_build_initial_message_empty_stdin_treated_as_absent() {
        let r = build_initial_message(&[], Some(String::new()), None, vec![]);
        assert_eq!(r.initial_message, None);
    }

    #[test]
    fn test_build_initial_message_whitespace_stdin_kept_verbatim() {
        // Guard condition is is_empty(), not trim: caller must ensure stdin is pre-trimmed (see read_piped_stdin),
        // This pins the implementation detail that 'whitespace-only content without pre-trimming is concatenated as-is'
        let r = build_initial_message(&[], Some("   ".into()), None, vec![]);
        assert_eq!(r.initial_message.as_deref(), Some("   "));
    }

    #[test]
    fn test_build_initial_message_file_only() {
        let r = build_initial_message(&[], None, Some("file content".into()), vec![]);
        assert_eq!(r.initial_message.as_deref(), Some("file content"));
    }

    #[test]
    fn test_build_initial_message_empty_file_text_treated_as_absent() {
        let r = build_initial_message(&[], None, Some(String::new()), vec![]);
        assert_eq!(r.initial_message, None);
    }

    #[test]
    fn test_build_initial_message_single_message() {
        let r = build_initial_message(&["msg"], None, None, vec![]);
        assert_eq!(r.initial_message.as_deref(), Some("msg"));
        assert!(r.remaining_messages.is_empty());
    }

    #[test]
    fn test_build_initial_message_first_message_consumed_rest_kept() {
        let msgs = ["a", "b", "c"];
        let r = build_initial_message(&msgs, None, None, vec![]);
        assert_eq!(r.initial_message.as_deref(), Some("a"));
        assert_eq!(r.remaining_messages, vec!["b", "c"]);
    }

    #[test]
    fn test_build_initial_message_empty_message_pushed_unconditionally() {
        // Message push has no is_empty guard: an empty-string message still makes parts non-empty, initial_message is Some("").
        // In actual paths, clap positional args are never empty strings; this pins the current code behavior.
        let r = build_initial_message(&[""], None, None, vec![]);
        assert_eq!(r.initial_message.as_deref(), Some(""));
        assert!(r.remaining_messages.is_empty());
    }

    #[test]
    fn test_build_initial_message_concatenates_without_separator() {
        // Aligned with pi parts.join(""): direct concatenation, no separator
        let r = build_initial_message(&["msg"], Some("hello".into()), Some("world".into()), vec![]);
        assert_eq!(r.initial_message.as_deref(), Some("helloworldmsg"));
    }

    #[test]
    fn test_build_initial_message_ordering_stdin_file_message() {
        // Fixed order: stdin → file_text → message
        let r = build_initial_message(&["M"], Some("S".into()), Some("F".into()), vec![]);
        assert_eq!(r.initial_message.as_deref(), Some("SFM"));
    }

    #[test]
    fn test_build_initial_message_message_not_trimmed() {
        // Only stdin is trimmed on read; message is kept as-is (including leading/trailing whitespace)
        let r = build_initial_message(&["  msg  "], None, None, vec![]);
        assert_eq!(r.initial_message.as_deref(), Some("  msg  "));
    }

    #[test]
    fn test_build_initial_message_images_passthrough() {
        let imgs = vec![img("aGVsbG8="), img("d29ybGQ=")];
        let r = build_initial_message(&["msg"], None, None, imgs);
        assert_eq!(r.initial_images.len(), 2);
        assert_eq!(r.initial_images[0].data, "aGVsbG8=");
        assert_eq!(r.initial_images[0].mime_type, "image/png");
        assert_eq!(r.initial_images[1].data, "d29ybGQ=");

        // Images are still passed through as-is when stdin/file exists
        let r2 = build_initial_message(&["m"], Some("s".into()), None, vec![img("x")]);
        assert_eq!(r2.initial_message.as_deref(), Some("sm"));
        assert_eq!(r2.initial_images.len(), 1);
    }

    #[test]
    fn test_build_initial_message_no_images_all_empty_is_ok() {
        // No panic when image is empty and all else is empty; result is all empty
        let r = build_initial_message(&[][..], Some(String::new()), None, Vec::new());
        assert_eq!(r.initial_message, None);
        assert!(r.initial_images.is_empty());
    }

    // ── normalize_stdin (private pure function) ──────────────────────

    #[test]
    fn test_normalize_stdin_trims_both_ends() {
        assert_eq!(normalize_stdin("  hello  ".into()), Some("hello".into()));
        assert_eq!(normalize_stdin("\thello\n".into()), Some("hello".into()));
    }

    #[test]
    fn test_normalize_stdin_empty_or_whitespace_returns_none() {
        assert_eq!(normalize_stdin(String::new()), None);
        assert_eq!(normalize_stdin("   ".into()), None);
        assert_eq!(normalize_stdin("\t\n \r".into()), None);
        // str::trim uses Unicode whitespace: full-width space/EM SPACE is also treated as empty
        assert_eq!(normalize_stdin("\u{3000}\u{2003}".into()), None);
    }

    #[test]
    fn test_normalize_stdin_keeps_interior_whitespace() {
        // Trim only both ends; preserve interior whitespace/newlines
        assert_eq!(normalize_stdin("a\nb".into()), Some("a\nb".into()));
        assert_eq!(normalize_stdin("  a  b  ".into()), Some("a  b".into()));
    }

    #[test]
    fn test_normalize_stdin_plain_content() {
        assert_eq!(
            normalize_stdin("hello world".into()),
            Some("hello world".into())
        );
    }

    // ── stdin_is_terminal (environment-dependent, smoke test) ─────────────

    #[test]
    fn test_stdin_is_terminal_matches_std_judgement() {
        // Return value depends on runtime environment (under cargo test it may be a terminal, /dev/null, or a pipe),
        // Only assert it matches the std criterion and does not panic
        assert_eq!(stdin_is_terminal(), std::io::stdin().is_terminal());
    }

    // read_piped_stdin: depends on the real stdin fd (TTY / pipe / /dev/null cannot be stably simulated in unit tests,
    // and when non-TTY without EOF, read_to_end blocks), core trim/empty checks are covered by normalize_stdin, so the IO path is skipped.
}
