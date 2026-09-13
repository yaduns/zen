use crate::session::session_manager::SessionManager;
use std::path::Path;
use std::path::PathBuf;

/// An immutable snapshot of the fact that the session's stored cwd is no longer valid on disk,
/// Constructed once by get_missing_session_cwd_issue when the detection matches;
/// The error payload, error message, and interactive prompt share it to avoid separately collecting session_file/cwd/fallback in three places.
/// All fields are pub (aligned with the pi interface's public properties; T-AP-08-5/08-6 consumers must read session_file/fallback_cwd)
/// Perform “Continue → reopen with fallback”).
#[derive(Debug, Clone, PartialEq)]
pub struct SessionCwdIssue {
    pub session_file: PathBuf, // Session file that produced the issue (the sole construction point guarantees a value)
    pub session_cwd: String,   // The cwd stored in the session header and no longer present on disk
    pub fallback_cwd: String, // Fallback working directory (startup scenario = process current directory), passed through unchanged; this module performs no normalization
}

/// A typed error that lets callers catch it by type, extract the snapshot, and proceed with the interactive retry flow.
/// `issue` is pub—the catch handler (under anyhow, `downcast_ref::<MissingSessionCwdError>()`) extracts the snapshot for Continue→reopen
/// (Rust equivalent of pi `instanceof` + `.issue` property access).
#[derive(Debug, thiserror::Error)]
pub struct MissingSessionCwdError {
    pub issue: SessionCwdIssue,
}

impl std::fmt::Display for MissingSessionCwdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format_missing_session_cwd_error(&self.issue))
    }
}

/// A safety gate before restoring an old session: the caller (the main startup flow, after `select_session_manager`,
/// before assembling the runtime; the future T-AP-08-5 session restoration flow) invokes it after obtaining the underlying session, resolving
/// the user-visible issue where “the cwd stored in the session file no longer exists on this machine, but the tool silently runs off into a phantom directory”—
/// analogous to entry security screening: in-memory sessions skip the check, while on-disk sessions check whether their ticket (stored cwd) is still valid.
///
/// 1. `get_session_file()` is `None` (in-memory session with no persistent cwd to validate) → returns `None`
/// 2. `get_cwd()` is an empty string → `None` (pi `!sessionCwd` guard)
/// 3. cwd exists on disk (`Path::exists()`: follows symlinks, swallows errors and returns false; semantically equivalent
///    to `fs.existsSync`; files/directories/symlinks all count as existing) → `None`
/// 4. Otherwise → `Some(SessionCwdIssue)` (session_file + stored cwd + fallback snapshot)
///
/// # Parameters
/// - `session_manager`: constructed session manager (the sole implementer of pi's structured `SessionCwdSource` interface,
///   use the concrete type directly; see "Ecosystem Reuse Verification Results" for disambiguation registration)
/// - `fallback_cwd`: fallback working directory (startup scenario = the process's current directory), written to the issue as-is,
///   this function does not check it for existence or normalize it}
///
/// # Consumers
/// `main.rs` startup gate (this task); `services.rs` consumes it indirectly via `assert_session_cwd_exists` (this task).
pub fn get_missing_session_cwd_issue(
    session_manager: &SessionManager,
    fallback_cwd: &str,
) -> Option<SessionCwdIssue> {
    // Guard 1: In-memory session (no session_file, no persisted cwd to validate) → return None directly and end the function
    let session_file = session_manager.get_session_file()?;

    // Guard 2:
    // Stored cwd is empty (it was not stored initially) or still exists on disk → allow
    // The directory still exists → the session can safely return to its original project context and continue
    let session_cwd = session_manager.get_cwd();
    if session_cwd.is_empty() || Path::new(session_cwd).exists() {
        return None;
    }

    Some(SessionCwdIssue {
        session_file,
        session_cwd: session_cwd.to_string(),
        fallback_cwd: fallback_cwd.to_string(),
    })
}

/// print-mode error text: called after the startup flow detects a missing cwd issue; renders the issue as a three-line error
/// message (「Stored session working directory does not exist: <cwd>」、「Session file: <path>」、
/// 「Current working directory: <fallback>」）——first identify exactly which line is invalid, then have the caller write to stderr and exit with
/// a non-zero exit code, leaving no room for guesswork.///
/// # Consumers
/// `main.rs` print/interim-interactive branch (this task).
pub fn format_missing_session_cwd_error(issue: &SessionCwdIssue) -> String {
    format!(
           "Stored session working directory does not exist: {}\nSession file: {}\nCurrent working directory: {}",
           issue.session_cwd,
           issue.session_file.display(),
           issue.fallback_cwd,
       )
}

/// interactive-mode confirmation dialog text: startup selector (T-AP-08-6) or session restoration flow (T-AP-08-5)
/// Called after detecting a missing cwd; renders the 「cwd from session file does not exist / <cwd> /
/// (blank line) / continue in current cwd / <fallback>」 prompt, allowing the user to continue (switching to the fallback directory
/// or reopening the session) or cancel—the text itself explains the options; the consequence of choosing Continue is stated in 「continue in current cwd」.
///
/// # Consumers
/// **pub but currently has no consumers; reserved for T-AP-08-6 (startup UI Continue/Cancel selector) and
/// T-AP-08-5 (/resume restore-flow confirmation overlay)**; in the interim, interactive goes through
/// the same error exit as `format_missing_session_cwd_error` (see "Integration with existing code").
pub fn format_missing_session_cwd_prompt(issue: &SessionCwdIssue) -> String {
    format!(
        "cwd from session file does not exist\n{}\n\ncontinue in current cwd\n{}",
        issue.session_cwd, issue.fallback_cwd,
    )
}

/// Safety fuse for the assembly factory: `create_initial_runtime_from_cli` (wired up in this task) and future session recovery flows
/// Called before constructing AgentSession; if the session cwd is missing, it returns `Err` and aborts assembly, preventing a defective runtime
/// from flowing downstream.
///
/// Behavior: `get_missing_session_cwd_issue` matches → `Err(MissingSessionCwdError)`; otherwise `Ok(())`.
///
/// # Parameters
/// - `session_manager`: the constructed session manager (object under inspection)
/// - `fallback_cwd`: fallback working directory. **Constraint: must be the user's current working directory, which must actually exist; must not take the
///   cwd of the object under test**—if fallback happens to be the missing directory under inspection, `issue.fallback_cwd` will write the same
///   Phantom directory; a subsequent 「Continue → reopen with `cwd_override=fallback`」 will return to the missing directory, and the assertion will
///   hit again, creating a retry infinite loop. **The value depends on the call-site semantics**: startup scenario (factory, this task) = the process's current directory
///   `std::env::current_dir()`; resume scenario (T-AP-08-5) = the cwd of the current running session (pi
///   `agent-session-runtime.ts:208`'s `this.cwd`), do not incorrectly retrieve it using the process cwd semantics.
///
/// # Errors
/// `MissingSessionCwdError` (carries the issue snapshot). It can be captured by type on the anyhow aggregate chain via
/// `err.downcast_ref::<MissingSessionCwdError>()` ,
/// After the interactive prompt, reopen the session with `cwd_override = fallback` — this retry pattern is reserved for T-AP-08-5.
///
/// # Consumers
/// `services.rs::create_initial_runtime_from_cli` (this task); T-AP-08-5 (reserved).
pub fn assert_session_cwd_exists(
    session_manager: &SessionManager,
    fallback_cwd: &str,
) -> Result<(), MissingSessionCwdError> {
    match get_missing_session_cwd_issue(session_manager, fallback_cwd) {
        Some(issue) => Err(MissingSessionCwdError { issue }),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::jsonl::CreateSessionOptions;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Test helper: construct a unique temporary root directory (pure std, without adding external dependencies).
    /// Best-effort cleanup in the test case (leftovers in the os temp directory are acceptable).
    fn unique_temp_root(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "zen-session-cwd-test-{}-{}-{}",
            std::process::id(),
            tag,
            nanos
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Test helper: default session creation options (aligned with the main.rs call site).
    fn opts() -> CreateSessionOptions {
        CreateSessionOptions {
            session_id: None,
            parent_session_path: None,
            metadata: None,
        }
    }

    /// Test helper: directly construct the issue snapshot (the fields are private, but this module is a submodule of the defining module).
    fn issue(session_file: &str, session_cwd: &str, fallback: &str) -> SessionCwdIssue {
        SessionCwdIssue {
            session_file: PathBuf::from(session_file),
            session_cwd: session_cwd.to_string(),
            fallback_cwd: fallback.to_string(),
        }
    }

    // ── format_missing_session_cwd_error ─────────────────────────

    #[test]
    fn test_format_error_three_lines_exact_layout() {
        let i = issue("/sessions/abc.jsonl", "/project/ghost", "/current");
        assert_eq!(
            format_missing_session_cwd_error(&i),
            "Stored session working directory does not exist: /project/ghost\n\
             Session file: /sessions/abc.jsonl\n\
             Current working directory: /current"
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_format_error_non_utf8_session_file_uses_lossy_display() {
        // PathBuf::display() uses replacement characters for non-UTF-8 paths instead of panicking (to_string panics)
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;
        let bad = PathBuf::from(OsString::from_vec(b"/sessions/\xff\xfe.jsonl".to_vec()));
        let i = SessionCwdIssue {
            session_file: bad,
            session_cwd: "ghost".to_string(),
            fallback_cwd: "cur".to_string(),
        };
        let out = format_missing_session_cwd_error(&i);
        assert!(
            out.contains('\u{FFFD}'),
            "Non-UTF-8 path should display as lossy: {out}"
        );
        assert_eq!(out.lines().count(), 3);
    }

    // ── format_missing_session_cwd_prompt ────────────────────────

    #[test]
    fn test_format_prompt_exact_layout_with_blank_line() {
        let i = issue("/s/abc.jsonl", "/project/ghost", "/current");
        assert_eq!(
            format_missing_session_cwd_prompt(&i),
            "cwd from session file does not exist\n/project/ghost\n\ncontinue in current cwd\n/current"
        );
    }

    // ── MissingSessionCwdError ───────────────────────────────────

    #[test]
    fn test_error_display_delegates_to_error_formatter() {
        let i = issue("/s/abc.jsonl", "/project/ghost", "/current");
        let err = MissingSessionCwdError { issue: i.clone() };
        assert_eq!(err.to_string(), format_missing_session_cwd_error(&i));
        // The payload snapshot is preserved as-is (the data source for reopening the issue after the T-AP-08-5 downcast)
        assert_eq!(err.issue, i);
    }

    // ── get_missing_session_cwd_issue (depends on a real SessionManager) ──

    #[test]
    fn test_get_issue_none_for_in_memory_session_even_when_cwd_ghost() {
        // Guard 1 precedes the cwd check: in-memory sessions have no session_file, so they pass regardless of cwd validity
        let sm = SessionManager::in_memory("/nonexistent/zen/ghost");
        assert_eq!(get_missing_session_cwd_issue(&sm, "/current"), None);
    }

    #[test]
    fn test_get_issue_none_when_stored_cwd_still_exists() {
        // Guard 3: the directory still exists → the session can safely return to its original project context
        let root = unique_temp_root("exists");
        let sm = SessionManager::create(root.to_str().unwrap(), &root, opts()).unwrap();
        assert_eq!(get_missing_session_cwd_issue(&sm, "/current"), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_get_issue_some_when_stored_cwd_missing() {
        // Guard 4: the stored cwd no longer exists → populate the three snapshot fields (pass fallback through unchanged)
        let root = unique_temp_root("ghost");
        let ghost = root.join("deleted-project");
        let sm = SessionManager::create(ghost.to_str().unwrap(), &root, opts()).unwrap();
        let found = get_missing_session_cwd_issue(&sm, "/current").unwrap();
        assert_eq!(found.session_file, sm.get_session_file().unwrap());
        assert_eq!(found.session_cwd, ghost.to_string_lossy());
        assert_eq!(found.fallback_cwd, "/current");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_get_issue_none_when_stored_cwd_is_empty() {
        // Guard 2: the stored cwd is an empty string → allow (pi `!sessionCwd` guard).
        // Assumption: SessionManager::create allows an empty cwd to be persisted (the header cwd stores an empty string).
        let root = unique_temp_root("empty");
        let sm = SessionManager::create("", &root, opts()).unwrap();
        assert_eq!(get_missing_session_cwd_issue(&sm, "/current"), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    // ── assert_session_cwd_exists ────────────────────────────────

    #[test]
    fn test_assert_ok_when_cwd_exists() {
        let root = unique_temp_root("assert-ok");
        let sm = SessionManager::create(root.to_str().unwrap(), &root, opts()).unwrap();
        assert!(assert_session_cwd_exists(&sm, "/current").is_ok());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_assert_err_when_cwd_missing_with_issue_payload() {
        let root = unique_temp_root("assert-err");
        let ghost = root.join("gone");
        let sm = SessionManager::create(ghost.to_str().unwrap(), &root, opts()).unwrap();
        let err = assert_session_cwd_exists(&sm, "/current").unwrap_err();
        // Retrieve the payload by type: Continue → reopen mode using fallback (the consumer surface reserved for T-AP-08-5)
        assert_eq!(err.issue.session_cwd, ghost.to_string_lossy());
        assert_eq!(err.issue.fallback_cwd, "/current");
        // Display delegates the error message
        assert_eq!(
            err.to_string(),
            format_missing_session_cwd_error(&err.issue)
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
