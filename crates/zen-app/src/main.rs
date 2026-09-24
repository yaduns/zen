// REQ-APP-08 (pi-360): implement the main entry point main (CLI assembly + mode dispatch)
// See reference/pi-original/packages/coding-agent/src/main.ts
// Implementation notes in tasks/T-AP-08-3-implemented.md (thin-shell main + assembly factory, dual delivery)

use clap::Parser;
use std::path::{Path, PathBuf};
use tracing::info;
use zen_app::cli::{
    build_initial_message, process_file_arguments, read_piped_stdin, select_session,
    should_run_first_time_setup, show_first_time_setup, show_startup_selector, stdin_is_terminal,
    CliArgs, Mode,
};
use zen_app::config::settings::SettingsManager;
use zen_app::config::{get_agent_dir, get_sessions_dir, get_settings_path, VERSION};
use zen_app::migrations::run_migrations;
use zen_app::modes::interactive::run_interactive_mode;
use zen_app::modes::print_mode::run_print_mode;
use zen_app::observability::diagnostics::{report_runtime_diagnostics, to_runtime_diagnostics};
use zen_app::session::jsonl::CreateSessionOptions;
use zen_app::session::services::create_initial_runtime_from_cli;
use zen_app::session::session_listing::{build_all_session_listings, build_session_listings};
use zen_app::session::session_manager::SessionManager;
use zen_app::session::{
    format_missing_session_cwd_error, format_missing_session_cwd_prompt,
    get_missing_session_cwd_issue, SessionCwdIssue,
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

    // Layer 2.5: startup data migration (T-AP-08-7: session relocation / commands→prompts / key renames).
    // pi counterpart: after resolveAppMode, before session selection — migration first relocates sessions scattered in the root directory,
    // so that node 4's select_session_manager can see them (aligned with pi main.ts runMigrations(cwd)).
    run_migrations(&process_cwd);

    // first-time setup (T-AP-08-6): interactive mode on first run (settings.json does not exist) pops the theme picker
    // choose the dialog and persist — aligned with pi main.ts "before session selection and runtime assembly, make the selected
    // settings apply to the whole flow". get_settings_path failure (no home) → silently skipped (cannot persist
    // i.e. no dialog pops up; a more explicit missing-home error surfaces during later assembly).
    if app_mode == AppMode::Interactive {
        if let Ok(settings_path) = get_settings_path() {
            if should_run_first_time_setup(&settings_path) {
                let startup_settings = SettingsManager::load(&process_cwd, &get_agent_dir()?);
                show_first_time_setup(&startup_settings).await?;
                info!("[main] node 2.5: first-time setup complete (theme persisted)");
            }
        }
    }
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
    let mut session_manager = select_session_manager(&args, &process_cwd).await?;

    // T-AP-01-5 gate: after select_session_manager, before assembly (pi in the same position).
    // print: error message + exit 1; interactive: launch the Continue/Cancel selector (T-AP-08-6),
    // Continue → reopen with fallback cwd, Cancel/Esc → exit(0).
    let process_cwd_lossy = process_cwd.to_string_lossy();
    if let Some(issue) = get_missing_session_cwd_issue(&session_manager, &process_cwd_lossy) {
        match app_mode {
            AppMode::Print(_) => {
                let (message, code) = missing_cwd_print_disposition(&issue);
                eprintln!("{message}");
                std::process::exit(code);
            }
            AppMode::Interactive => {
                let root = resolve_sessions_root(&args)?;
                let choice = show_startup_selector(
                    &format_missing_session_cwd_prompt(&issue),
                    vec![
                        ("Continue", Some(issue.fallback_cwd.clone())),
                        ("Cancel", None),
                    ],
                )
                .await?;
                match choice {
                    // Continue → reopen with fallback cwd (mirrors pi promptForMissingSessionCwd)
                    Some(Some(fallback)) => {
                        session_manager =
                            SessionManager::open(&issue.session_file, Some(&fallback), &root)?;
                    }
                    // Cancel / Esc → exit (mirrors pi process.exit(0))
                    _ => std::process::exit(0),
                }
            }
        }
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
            // Diagnostic output is best-effort: a stderr write failure does not abort startup (unused_must_use silenced)
            let _ = report_runtime_diagnostics(
                &to_runtime_diagnostics(&runtime.diagnostics),
                &mut std::io::stderr(),
            );

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

/// Missing-cwd handling in print mode (testable pure function: no IO, no exit)——
/// "session cwd is invalid and there is no interactive chance (print): render error message + exit 1";
/// The interactive branch is handled by the T-AP-08-6 startup selector (see the main gate).
///
/// # Consumers
/// Called only internally within `main` (this file).
fn missing_cwd_print_disposition(issue: &SessionCwdIssue) -> (String, i32) {
    (format_missing_session_cwd_error(issue), 1)
}

/// Session root resolution: `--session-dir` override or default `get_sessions_dir()`——
/// Shared by `select_session_manager` and the missing-cwd Continue reopen (T-AP-08-6).
///
/// # Consumers
/// Called only internally within `main` (this file).
fn resolve_sessions_root(args: &CliArgs) -> anyhow::Result<PathBuf> {
    match &args.session_dir {
        Some(dir) => Ok(dir.clone()),
        None => Ok(get_sessions_dir()?),
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
/// # Mapping (aligned with pi `selectSessionManager`, T-AP-08-5 semantic fixes)
/// - `--no-session` → `in_memory` (ephemeral, not persisted)
/// - `--fork <source>` → `fork_from`; combined with `--session-id` when the target id already exists → error exit
/// - `--resume` → TUI session selector (`select_session`); cancel → "No session selected" + exit(0);
///   non-TTY stdin falls back to `continue_recent` (v1 decision: TUI requires a terminal)
/// - `--continue` → `continue_recent` (no longer conflated with resume)
/// - `--session-id` → exact lookup by id (`SessionManager::list` match) → hit opens /
///   miss → warning + `create` (session_id injected, aligned with pi's yellow warning)
/// - `--session <file>` → `open` (full path)
/// - default → `create` (new file-backed session, **persisted**, aligned with pi)
/// - `--session-dir <dir>` overrides the session root directory, affecting all the `*_root` values above
///
/// # Consumers
/// Called only internally within `main` (this file).
async fn select_session_manager(args: &CliArgs, cwd: &Path) -> anyhow::Result<SessionManager> {
    let cwd_str = cwd.to_string_lossy();
    let root = resolve_sessions_root(args)?;
    let default_opts = CreateSessionOptions {
        session_id: None,
        parent_session_path: None,
        metadata: None,
    };

    if args.no_session {
        return Ok(SessionManager::in_memory(&cwd_str));
    }

    // --fork: when combined with --session-id, verify the target id is not a duplicate (aligned with pi forkSessionOrExit)
    if let Some(fork_src) = &args.fork {
        if let Some(session_id) = &args.session_id {
            let sessions = SessionManager::list(&cwd_str, &root)?;
            if sessions.iter().any(|s| s.id() == session_id.as_str()) {
                anyhow::bail!("Session already exists with id '{session_id}'");
            }
            return Ok(SessionManager::fork_from(
                Path::new(fork_src),
                &cwd_str,
                &root,
                CreateSessionOptions {
                    session_id: Some(session_id.clone()),
                    parent_session_path: None,
                    metadata: None,
                },
            )?);
        }
        return Ok(SessionManager::fork_from(
            Path::new(fork_src),
            &cwd_str,
            &root,
            default_opts,
        )?);
    }

    // --resume: TUI session selector (pi selectSession)
    if args.resume {
        if !stdin_is_terminal() {
            return Ok(SessionManager::continue_recent(&cwd_str, &root)?);
        }
        let current = build_session_listings(&cwd_str, &root);
        let all = build_all_session_listings(&root);
        let selected = select_session(current, all, &root).await?;
        return match selected {
            Some(path) => Ok(SessionManager::open(Path::new(&path), None, &root)?),
            None => {
                // Aligned with pi: console.log(chalk.dim("No session selected")) + exit(0)
                eprintln!("No session selected");
                std::process::exit(0);
            }
        };
    }

    // --continue: resume the most recent session (create if none), no longer conflated with --resume
    if args.continue_flag {
        return Ok(SessionManager::continue_recent(&cwd_str, &root)?);
    }

    // --session-id: exact lookup by id (partial UUID resolution; T-AP-08-5 fixes the original "open by path" deviation)
    if let Some(session_id) = &args.session_id {
        let sessions = SessionManager::list(&cwd_str, &root)?;
        if let Some(found) = sessions.into_iter().find(|s| s.id() == session_id.as_str()) {
            return Ok(SessionManager::open(found.path(), None, &root)?);
        }
        eprintln!(
            "Warning: No project session found with id '{session_id}'; creating a new session with that id."
        );
        return Ok(SessionManager::create(
            &cwd_str,
            &root,
            CreateSessionOptions {
                session_id: Some(session_id.clone()),
                parent_session_path: None,
                metadata: None,
            },
        )?);
    }

    // --session <file>: open by full path
    if let Some(session) = &args.session {
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

    // ── missing-cwd gate (T-AP-01-5 detection + T-AP-08-6 handling) ────────────

    #[test]
    fn missing_cwd_gate_passes_in_memory_session() {
        // In-memory sessions skip the check (guard 1): get_missing_session_cwd_issue → None → gate passes
        let sm = SessionManager::in_memory("/current");
        assert!(get_missing_session_cwd_issue(&sm, "/current").is_none());
    }

    #[test]
    fn missing_cwd_print_disposition_error_exit_1_mentions_session_cwd() {
        let issue = SessionCwdIssue {
            session_file: PathBuf::from("/sessions/abc.jsonl"),
            session_cwd: "/project/ghost".to_string(),
            fallback_cwd: "/current".to_string(),
        };
        let (msg, code) = missing_cwd_print_disposition(&issue);
        assert_eq!(code, 1, "print branch error exit code aligned with pi exit(1)");
        assert!(
            msg.contains("/project/ghost"),
            "dispose message should include the stored session_cwd"
        );
    }

    // ── resolve_sessions_root（T-AP-08-6）─────────────────────────

    #[test]
    fn resolve_sessions_root_prefers_session_dir_override() {
        let args = parse(&["--session-dir", "/custom/root"]);
        assert_eq!(
            resolve_sessions_root(&args).unwrap(),
            PathBuf::from("/custom/root")
        );
    }

    #[test]
    fn resolve_sessions_root_defaults_to_agent_sessions_dir() {
        // Default branch = get_sessions_dir() (<agentDir>/sessions; env/ENV_AGENT_DIR and
        // home are both irrelevant—assertions only match the trailing `sessions` segment, immune to parallel env races).
        let args = parse(&[]);
        let root = resolve_sessions_root(&args).unwrap();
        assert!(
            root.ends_with("sessions"),
            "default session root = <agentDir>/sessions (actual {root:?})"
        );
    }

    // ── select_session_manager (T-AP-08-5: resume/continue/session-id semantics) ────────

    /// Test helper: temp directory (std-only, following the session_manager test precedent).
    fn test_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("zen_main_ssm_{}_{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn create_session(root: &Path, cwd: &str, id: &str) -> PathBuf {
        let mgr = SessionManager::create(
            cwd,
            root,
            CreateSessionOptions {
                session_id: Some(id.to_string()),
                parent_session_path: None,
                metadata: None,
            },
        )
        .unwrap();
        mgr.get_session_file().unwrap()
    }

    #[tokio::test]
    async fn ssm_continue_opens_latest_or_creates() {
        let root = test_dir("continue");
        let args = parse(&["--continue", "--session-dir", root.to_str().unwrap()]);
        let sm = select_session_manager(&args, Path::new("/proj"))
            .await
            .unwrap();
        // No session → create new
        assert!(!sm.get_session_id().is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn ssm_session_id_hit_opens_existing() {
        let root = test_dir("sessid_hit");
        create_session(&root, "/proj", "abc123");
        let args = parse(&[
            "--session-id",
            "abc123",
            "--session-dir",
            root.to_str().unwrap(),
        ]);
        let sm = select_session_manager(&args, Path::new("/proj"))
            .await
            .unwrap();
        assert_eq!(sm.get_session_id(), "abc123", "open by id on hit");
        assert!(sm.get_session_file().is_some());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn ssm_session_id_miss_creates_with_id() {
        let root = test_dir("sessid_miss");
        let args = parse(&[
            "--session-id",
            "newname",
            "--session-dir",
            root.to_str().unwrap(),
        ]);
        let sm = select_session_manager(&args, Path::new("/proj"))
            .await
            .unwrap();
        assert_eq!(sm.get_session_id(), "newname", "on miss → create injects id");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn ssm_fork_with_existing_session_id_errors() {
        let root = test_dir("fork_conflict");
        let src = create_session(&root, "/src", "src1");
        create_session(&root, "/proj", "abc123");
        let args = parse(&[
            "--fork",
            src.to_str().unwrap(),
            "--session-id",
            "abc123",
            "--session-dir",
            root.to_str().unwrap(),
        ]);
        let err = match select_session_manager(&args, Path::new("/proj")).await {
            Err(e) => e,
            Ok(_) => panic!("should error if the target id already exists"),
        };
        assert!(
            err.to_string().contains("Session already exists"),
            "Actual: {err}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn ssm_session_path_opens_and_no_session_in_memory() {
        let root = test_dir("misc");
        let path = create_session(&root, "/proj", "s1");
        let args = parse(&["--session", path.to_str().unwrap()]);
        let sm = select_session_manager(&args, Path::new("/proj"))
            .await
            .unwrap();
        assert_eq!(sm.get_session_id(), "s1", "open by full path");

        let args = parse(&["--no-session"]);
        let sm = select_session_manager(&args, Path::new("/proj"))
            .await
            .unwrap();
        assert!(sm.get_session_file().is_none(), "no-session in-memory session");
        let _ = std::fs::remove_dir_all(&root);
    }
}
