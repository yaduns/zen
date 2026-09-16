// REQ-APP-08 (pi-360): implement the main entry point main (CLI assembly + mode dispatch)
// See reference/pi-original/packages/coding-agent/src/main.ts
// Implementation notes in tasks/T-AP-08-3-implemented.md (thin-shell main + assembly factory, dual delivery)

use clap::Parser;
use std::path::{Path, PathBuf};
use tracing::info;
use zen_app::cli::{
    build_initial_message, process_file_arguments, read_piped_stdin, stdin_is_terminal, CliArgs,
    Mode,
};
use zen_app::config::{get_sessions_dir, VERSION};
use zen_app::modes::interactive::run_interactive_mode;
use zen_app::modes::print_mode::run_print_mode;
use zen_app::session::jsonl::CreateSessionOptions;
use zen_app::session::services::create_initial_runtime_from_cli;
use zen_app::session::session_manager::SessionManager;
use zen_app::session::{
    format_missing_session_cwd_error, get_missing_session_cwd_issue, SessionCwdIssue,
};

/// Application run mode.
/// Corresponds to pi's `AppMode` (`"print" | "interactive"`; `"rpc"` removed for simplification per REQ-APP-08).
/// `Print(Mode)` carries the output format (text/json); `Interactive` carries none (supported only in Phase 2).
#[derive(Debug, Clone, PartialEq, Eq)]
enum AppMode {
    /// print mode: process the prompt then exit
    Print(Mode),
    /// interactive mode: interactive session
    Interactive,
}

/// zen binary entry point (thin shell: parse → assembly factory → dispatch, three layers).
///
/// All assembly logic (model resolution/tool assembly/system prompt/Agent construction/runtime assembly) is in
/// Done inside `zen_app::session::services::create_initial_runtime_from_cli` (rework point 3/4),
/// main retains only three glue layers: CLI parsing, initial message construction, mode dispatch.
///
/// Flow:
/// 1. `CliArgs::parse()` parses CLI arguments
/// 2. `--version` / `--export` fast paths
/// 3. Mode determination (`--mode` > `--print` > stdin not a TTY > interactive)
/// 4. Read stdin + build initial message (T-AP-08-2)
/// 5. Call the assembly factory to obtain `AgentSessionRuntime` (rework point 3)
/// 6. Dispatch by mode: print → `run_print_mode`, interactive → T-AP-11-1 (Phase 2)
///
/// Exit code: print mode returns the exit code of `run_print_mode`; interactive mode defaults to 0.
///
/// # Errors
/// Returns Err when runtime creation fails (aggregated by anyhow; `#[tokio::main]` prints to stderr and exits with 1).
/// Phase 1 does not implement graceful SIGTERM/SIGHUP shutdown (print is a stub that returns immediately;
/// `CancellationToken` + `tokio::select!` will be added when real print lands in T-AP-07-1, see "Open Issues #4" in the task doc).
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // tracing subscriber setup (REQ-OBS-01 refactor note: inlined directly in main.rs, not wrapped in a separate function;
    // code ownership under T-OB-01-1 Drop). Logs go to stderr (stdout contains only structured output, see T-AP-06-1).
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    // Layer 1: parsing + fast paths + mode determination
    let args = CliArgs::parse();
    info!(
        mode = ?args.mode,
        print = args.print,
        provider = ?args.provider,
        model = ?args.model,
        thinking = ?args.thinking,
        "[main] node 1: CliArgs::parse()"
    );

    // --version: print the version and exit (clap disable_version_flag, needs manual handling)
    if args.version {
        println!("{VERSION}");
        return Ok(());
    }

    // --export: not supported in Phase 1 (HTML export belongs to the T-AP-08-3 extension in Phase 2)
    if args.export.is_some() {
        anyhow::bail!("--export is not supported yet (Phase 2)");
    }

    let app_mode = resolve_app_mode(&args, stdin_is_terminal());
    info!(
        app_mode = ?app_mode,
        "[main] node 2: mode resolution (--mode > --print > non-TTY stdin > interactive)"
    );

    // Layer 2: read stdin + @file handling + initial messages (T-AP-08-2 + T-AP-08-4)
    // Process cwd: @file path resolution (T-AP-08-4) + session selection (--session restores its original cwd)
    let process_cwd = std::env::current_dir()?;
    let stdin_content = read_piped_stdin().await?;
    let messages: Vec<&str> = args.messages();
    let file_args = args.file_args();
    let (file_text, file_images) = if file_args.is_empty() {
        (None, vec![])
    } else {
        match process_file_arguments(&file_args, &process_cwd).await {
            Ok(processed) => (Some(processed.text), processed.images),
            Err(e) => {
                // Align with pi processFileArguments' console.error(chalk.red(...)) + exit(1)
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
    };
    let initial = build_initial_message(&messages, stdin_content, file_text, file_images);
    info!(
        has_initial = initial.initial_message.is_some(),
        remaining = initial.remaining_messages.len(),
        "[main] node 3: read stdin + build initial message"
    );

    // Layer 3: assembly factory (refactoring point 3 — assembly logic extracted to services.rs; session chosen by main)
    let session_manager = select_session_manager(&args, &process_cwd)?;

    // T-AP-01-5 gate: after select_session_manager, before assembly (same location as pi)
    let process_cwd_lossy = process_cwd.to_string_lossy();
    let issue = get_missing_session_cwd_issue(&session_manager, &process_cwd_lossy);

    // session's cwd directory does not exist; exit directly
    if let Some((message, code)) = missing_cwd_disposition(issue.as_ref(), &app_mode) {
        eprintln!("{message}");
        std::process::exit(code);
    }

    // Assembled cwd = session cwd (drives settings/skills/AGENTS.md/tool base dir to migrate with the session)
    let session_cwd = PathBuf::from(session_manager.get_cwd());
    let runtime = create_initial_runtime_from_cli(&args, session_cwd, session_manager).await?;

    info!(
        session_id = runtime.session.session_id(),
        model_id = %runtime.session.model().id,
        "[main] node 4: assemble factory -> AgentSessionRuntime ready"
    );

    // Layer 4: mode dispatch
    match app_mode {
        AppMode::Print(mode) => {
            let exit_code = run_print_mode(
                runtime,
                mode,
                initial
                    .remaining_messages
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                initial.initial_message,
                initial.initial_images,
            )
            .await?;
            info!(exit_code, "[main] node 5: print mode dispatch complete");
            if exit_code != 0 {
                std::process::exit(exit_code);
            }
        }
        AppMode::Interactive => {
            let exit_code = run_interactive_mode(
                runtime,
                initial.initial_message,
                initial.initial_images,
                args.messages().iter().map(|s| s.to_string()).collect(),
            )
            .await?;
            info!(exit_code, "[main] node 5: interactive mode dispatch complete");
            if exit_code != 0 {
                std::process::exit(exit_code);
            }
        }
    }

    Ok(())
}

/// Startup disposition decision for the missing-cwd issue (testable pure function: no IO, no exit).
///
/// Return `Some((stderr message, exit code))` to indicate startup should terminate; `None` to allow it to proceed.
/// During the interim phase, print and interactive both use "error message + exit 1";
/// After T-AP-08-6 goes live, the interactive branch switches to a Continue/Cancel selector flow
/// (Continue → reopen with the fallback); the print branch remains unchanged.
///
/// # Consumers
/// Called only internally within `main` (this file).
fn missing_cwd_disposition(
    issue: Option<&SessionCwdIssue>,
    app_mode: &AppMode,
) -> Option<(String, i32)> {
    let issue = issue?;
    match app_mode {
        AppMode::Print(_) | AppMode::Interactive => {
            Some((format_missing_session_cwd_error(issue), 1))
        }
    }
}

/// Determine the application run mode (print / interactive).
///
/// Decision rules (aligned with pi `resolveAppMode`):
/// 1. `args.mode` explicitly specified → use directly (`Mode::Text` → print "text", `Mode::Json` → print "json")
/// 2. `args.print` flag is true → print "text"
/// 3. stdin is not a TTY (piped input) → print "text"
/// 4. none of the above → interactive
///
/// # Parameters
/// - `args`: parsed CLI arguments
/// - `stdin_is_tty`: whether stdin is a terminal (passed by caller for testability)
///
/// # Consumers
/// Called only internally within `main` (this file).
fn resolve_app_mode(args: &CliArgs, stdin_is_tty: bool) -> AppMode {
    // 1. --mode explicitly specified → use directly
    if let Some(mode) = args.mode {
        return AppMode::Print(mode);
    }

    // 2. --print flag → print "text"
    if args.print {
        return AppMode::Print(Mode::Text);
    }

    // 3. stdin is not a TTY (piped input) → print "text"
    if !stdin_is_tty {
        return AppMode::Print(Mode::Text);
    }

    // 4. default → interactive
    AppMode::Interactive
}

/// Select `SessionManager` based on CLI session flags (reuse `SessionManager`'s existing constructor, don't roll your own).
///
/// # Mapping
/// - `--no-session` → `in_memory` (ephemeral, not persisted)
/// - `--fork <source>` → `fork_from` (fork from source session)
/// - `--resume` / `--continue` → `continue_recent` (resume most recent session, create new if none)
/// - `--session <file>` / `--session-id` → `open` (full path; partial UUID resolution assigned to T-AP-08-5)
/// - default → `create` (new file-backed session, **persisted**, aligned with pi)
/// - `--session-dir <dir>` overrides the session root directory, affecting all the `*_root` values above
///
/// # Consumers
/// Called internally by `main` (this file) only. (Decision: persist/restore model+thinking by default; session by full path only)
fn select_session_manager(args: &CliArgs, cwd: &Path) -> anyhow::Result<SessionManager> {
    let cwd_str = cwd.to_string_lossy();
    let root = match &args.session_dir {
        Some(dir) => dir.clone(),
        None => get_sessions_dir()?,
    };
    let default_opts = CreateSessionOptions {
        session_id: None,
        parent_session_path: None,
        metadata: None,
    };

    if args.no_session {
        return Ok(SessionManager::in_memory(&cwd_str));
    }
    if let Some(fork_src) = &args.fork {
        return Ok(SessionManager::fork_from(
            Path::new(fork_src),
            &cwd_str,
            &root,
            default_opts,
        )?);
    }
    if args.resume || args.continue_flag {
        return Ok(SessionManager::continue_recent(&cwd_str, &root)?);
    }
    if let Some(session) = args.session.as_ref().or(args.session_id.as_ref()) {
        // // T-AP-01-5: Remove the Phase 1 simplification—the header cwd takes effect, and session restoration returns to its original directory
        return Ok(SessionManager::open(Path::new(session), None, &root)?);
    }
    Ok(SessionManager::create(&cwd_str, &root, default_opts)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper: construct CliArgs (aligned with args.rs test conventions).
    fn parse(args: &[&str]) -> CliArgs {
        let mut full: Vec<&str> = Vec::with_capacity(args.len() + 1);
        full.push("zen");
        full.extend_from_slice(args);
        CliArgs::try_parse_from(full).unwrap()
    }

    // ── resolve_app_mode ─────────────────────────────────────────

    #[test]
    fn explicit_mode_wins_over_other_signals() {
        // --mode json explicitly specified → print json (even if stdin is a TTY)
        let args = parse(&["--mode", "json"]);
        assert_eq!(resolve_app_mode(&args, true), AppMode::Print(Mode::Json));
        assert_eq!(resolve_app_mode(&args, false), AppMode::Print(Mode::Json));
    }

    #[test]
    fn print_flag_forces_print_text() {
        let args = parse(&["--print"]);
        assert_eq!(resolve_app_mode(&args, true), AppMode::Print(Mode::Text));
        assert_eq!(resolve_app_mode(&args, false), AppMode::Print(Mode::Text));
    }

    #[test]
    fn piped_stdin_forces_print() {
        // no flags + stdin not a TTY → print text
        let args = parse(&[]);
        assert_eq!(resolve_app_mode(&args, false), AppMode::Print(Mode::Text));
    }

    #[test]
    fn tty_without_flags_defaults_to_interactive() {
        let args = parse(&[]);
        assert_eq!(resolve_app_mode(&args, true), AppMode::Interactive);
    }

    // ── missing_cwd_disposition (T-AP-01-5 gate) ──────────────────

    #[test]
    fn missing_cwd_gate_passes_in_memory_session() {
        // In-memory sessions are exempt (guard 1): get_missing_session_cwd_issue → None → allow through, no exit code produced
        let sm = SessionManager::in_memory("/current");
        let issue = get_missing_session_cwd_issue(&sm, "/current");
        assert_eq!(
            missing_cwd_disposition(issue.as_ref(), &AppMode::Print(Mode::Text)),
            None
        );
        assert_eq!(
            missing_cwd_disposition(issue.as_ref(), &AppMode::Interactive),
            None
        );
    }

    #[test]
    fn missing_cwd_disposition_print_error_exit_1_mentions_session_cwd() {
        let issue = SessionCwdIssue {
            session_file: PathBuf::from("/sessions/abc.jsonl"),
            session_cwd: "/project/ghost".to_string(),
            fallback_cwd: "/current".to_string(),
        };
        let (msg, code) =
            missing_cwd_disposition(Some(&issue), &AppMode::Print(Mode::Text)).expect("should be handled");
        assert_eq!(code, 1, "print branch error exit code aligned with pi exit(1)");
        assert!(
            msg.contains("/project/ghost"),
            "dispose message should include the stored session_cwd"
        );
    }

    #[test]
    fn missing_cwd_disposition_interim_interactive_error_exit_1() {
        let issue = SessionCwdIssue {
            session_file: PathBuf::from("/sessions/abc.jsonl"),
            session_cwd: "/project/ghost".to_string(),
            fallback_cwd: "/current".to_string(),
        };
        let (msg, code) =
            missing_cwd_disposition(Some(&issue), &AppMode::Interactive).expect("should be handled");
        assert_eq!(
            code, 1,
            "interim: interactive exits with the same error as print (before T-AP-08-6 takes over)"
        );
        assert!(msg.contains("does not exist"), "disposition message should contain the error text");
    }
}
