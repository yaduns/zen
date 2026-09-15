// REQ-APP-06 (pi-323): footer data provider — git branch detection + change watching.
// Corresponds to original TS: reference/pi-original/packages/coding-agent/src/core/footer-data-provider.ts.
// v1 scope trimming: extensionStatuses/availableProviderCount (extension system/multi-provider) omitted;
// setCwd is not implemented (session switching rebuilds the provider via replace_runtime, which is equivalent); onBranchChange callback set
// → tokio::sync::watch channel (single latest value + changed() change notification).
// Pattern: notify (T-AP-11-3 already in workspace) replaces pi-339 fs-watch; background debounce task aligns with pi
// WATCH_DEBOUNCE_MS/FS_WATCH_RETRY_DELAY_MS。
// See tasks/T-AP-06-3-implemented.md for implementation notes.

use notify::{Error as NotifyError, Event as NotifyEvent, RecursiveMode, Watcher};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Branch-change debounce window: git atomic writes (temp file + rename) can produce multiple fs events in an instant,
/// merged into one reparse within a 500ms window (aligning with pi `WATCH_DEBOUNCE_MS`).
const WATCH_DEBOUNCE_MS: Duration = Duration::from_millis(500);
/// Rebuild interval after watcher error (aligning with pi `FS_WATCH_RETRY_DELAY_MS`).
const WATCHER_RETRY_DELAY: Duration = Duration::from_secs(5);
/// Polling interval for WSL `/mnt` paths (aligning with pi `watchFile` interval 1000ms).
const POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Three-state result of parsing HEAD file contents.
/// Corresponds to string checks in pi `resolveGitBranchSync` (`ref: refs/heads/` prefix / `.invalid` sentinel / detached).
enum HeadRef {
    /// Normal branch ref (branch name without `refs/heads/` prefix).
    Branch(String),
    /// Detached HEAD (non-ref content such as a bare commit hash).
    Detached,
    /// `ref: refs/heads/.invalid` sentinel — an intermediate state during git atomic branch switching; requires git fallback resolution.
    Invalid,
}

/// Discovered git metadata paths (`find_git_paths` output + watcher watch targets).
#[derive(Clone, Debug)]
pub(crate) struct GitPaths {
    /// Repository root (work tree root containing cwd); used as `current_dir` when falling back to git for `.invalid`.
    pub(crate) repo_dir: PathBuf,
    /// Shared git directory (normal repo = `.git`; worktree points to main repo `.git` via `commondir`).
    pub(crate) common_git_dir: PathBuf,
    /// HEAD file path (normal repo = `common_git_dir/HEAD`; for worktree it is under the worktree's git dir).
    pub(crate) head_path: PathBuf,
}

/// Parse the raw HEAD file contents into three states (Branch/Detached/Invalid) — pure string checks,
/// used for dispatch by `resolve_git_branch`; the `.invalid` sentinel is a separate state so it can use git fallback,
/// all other non-ref content is treated as detached.
fn parse_head_ref(content: &str) -> HeadRef {
    const PREFIX: &str = "ref: refs/heads/";
    let content = content.trim();
    if let Some(branch) = content.strip_prefix(PREFIX) {
        if branch == ".invalid" {
            HeadRef::Invalid
        } else {
            HeadRef::Branch(branch.to_string())
        }
    } else {
        HeadRef::Detached
    }
}

/// Walk up from cwd level by level to locate git metadata, letting the footer read the branch name without invoking git — normal repos look for
/// `.git/HEAD`; worktree repos jump to the shared git directory via the `gitdir:` file and parse `commondir`.
/// Returning `None` means cwd is not in any git repository (footer shows no branch); if the worktree's `commondir`
/// fails to parse or HEAD does not exist, also return `None` (aligning with pi `findGitPaths`'s give-up branch).
pub(crate) fn find_git_paths(cwd: &Path) -> Option<GitPaths> {
    let mut dir = cwd;
    loop {
        let git_path = dir.join(".git");
        if git_path.exists() {
            // worktree: .git is a file whose content is "gitdir: <path>"
            if git_path.is_file() {
                let content = std::fs::read_to_string(&git_path).ok()?;
                if let Some(git_dir_raw) = content.trim().strip_prefix("gitdir: ") {
                    let git_dir = dir.join(git_dir_raw.trim());
                    let head_path = git_dir.join("HEAD");
                    if !head_path.exists() {
                        return None;
                    }
                    // commondir points to the main repo's .git (needed for reftable scenarios); fall back to gitDir when missing
                    let common_dir_path = git_dir.join("commondir");
                    let common_git_dir = if common_dir_path.exists() {
                        let cd = std::fs::read_to_string(&common_dir_path).ok()?;
                        git_dir.join(cd.trim())
                    } else {
                        git_dir.clone()
                    };
                    return Some(GitPaths {
                        repo_dir: dir.to_path_buf(),
                        common_git_dir,
                        head_path,
                    });
                }
                // .git is a file but not gitdir: content → keep walking up (aligning with pi's fall-through without return)
            } else if git_path.is_dir() {
                let head_path = git_path.join("HEAD");
                if !head_path.exists() {
                    return None;
                }
                return Some(GitPaths {
                    repo_dir: dir.to_path_buf(),
                    common_git_dir: git_path,
                    head_path,
                });
            }
        }
        dir = dir.parent()?;
    }
}

/// Ask the real branch via the git binary (`git --no-optional-locks symbolic-ref --quiet --short HEAD`).
/// Returns `None` when git is unavailable or exits non-zero (detached HEAD); the caller falls back to "detached";
/// `--no-optional-locks` prevents the branch query itself from triggering index locks (aligned with pi `resolveBranchWithGitSync/Async`).
/// `git_bin` is a test seam (inject fake git); production passes "git" via `resolve_branch_with_git`.
fn git_symbolic_ref(git_bin: &str, repo_dir: &Path) -> Option<String> {
    let out = std::process::Command::new(git_bin)
        .args([
            "--no-optional-locks",
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
        ])
        .current_dir(repo_dir)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let branch = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if branch.is_empty() {
        None
    } else {
        Some(branch)
    }
}

/// Only when HEAD is the `refs/heads/.invalid` sentinel (intermediate state of git's atomic branch switch, common in reftable repos)
/// is it called: spawns git to query the real branch name, so the footer doesn't show a repo mid-switch as the wrong branch.
fn resolve_branch_with_git(repo_dir: &Path) -> Option<String> {
    git_symbolic_ref("git", repo_dir)
}

/// Each refresh (initial construction + after file-change debounce) converts git metadata into the branch name the footer displays:
/// HEAD points to `refs/heads/X` and is not .invalid → X; bare commit hash → "detached"; .invalid → delegate to
/// `resolve_branch_with_git`, falling back to "detached" on failure; file read failure → `None` (footer shows no branch).
fn resolve_git_branch(paths: &GitPaths) -> Option<String> {
    let content = std::fs::read_to_string(&paths.head_path).ok()?;
    match parse_head_ref(&content) {
        HeadRef::Branch(b) => Some(b),
        HeadRef::Detached => Some("detached".to_string()),
        HeadRef::Invalid => {
            resolve_branch_with_git(&paths.repo_dir).or(Some("detached".to_string()))
        }
    }
}

/// WSL environment detection (aligned with pi `isWslEnvironment`).
fn is_wsl_environment() -> bool {
    std::env::var_os("WSL_DISTRO_NAME").is_some() || std::env::var_os("WSL_INTEROP").is_some()
}

/// `/mnt/[a-z]...` Windows mount path detection (aligned with pi `isWindowsMountedRepoPath`, case-insensitive).
fn is_windows_mounted_repo_path(repo_dir: &Path) -> bool {
    let s = repo_dir.to_string_lossy();
    if !s.starts_with("/mnt/") {
        return false;
    }
    let bytes = s.as_bytes();
    // "/mnt/" is 5 bytes; byte 6 (index 5) must be a single letter, followed by '/' or end of string
    bytes.get(5).is_some_and(|b| b.is_ascii_alphabetic())
        && (bytes.get(6).is_none() || bytes[6] == b'/')
}

/// On WSL, accessing a Windows mount path (`/mnt/[a-z]`) requires polling fallback — on these filesystems inotify
/// events are unreliable, so branch watching degrades to polling to keep the footer refreshing after `git switch` (aligned with pi `shouldPollGitHead`).
fn should_poll_git_head(repo_dir: &Path) -> bool {
    is_wsl_environment() && is_windows_mounted_repo_path(repo_dir)
}

/// Whether a notify event is relevant to this footer: HEAD file change (including atomic rename) or a change inside the reftable dir.
/// Aligned with pi `setupGitWatcher`'s `filename === "HEAD"` filter and the reftableDir callback's unconditional refresh;
/// when the path is empty (not reported by the backend), conservatively treat it as relevant.
fn event_is_relevant(event: &NotifyEvent, reftable_dir: &Path) -> bool {
    if event.paths.is_empty() {
        return true;
    }
    event.paths.iter().any(|p| {
        p.starts_with(reftable_dir) || p.file_name().is_some_and(|n| n == OsStr::new("HEAD"))
    })
}

/// Watch paths: the directory containing HEAD + the reftable dir if present (aligned with pi `setupGitWatcher`'s two watchers).
fn watch_paths(git_paths: &GitPaths) -> Vec<PathBuf> {
    let head_dir = git_paths
        .head_path
        .parent()
        .unwrap_or(&git_paths.common_git_dir)
        .to_path_buf();
    let mut paths = vec![head_dir];
    let reftable_dir = git_paths.common_git_dir.join("reftable");
    if reftable_dir.is_dir() {
        paths.push(reftable_dir);
    }
    paths
}

/// Event signal for the background refresh task.
enum FsSignal {
    /// Relevant git file change (HEAD/reftable).
    Changed,
    /// Watcher error (aligned with pi `handleGitWatcherError` → rebuild after delay).
    WatcherError,
}

/// Creates (and registers) the git file watcher: RecommendedWatcher on normal platforms, PollWatcher on WSL `/mnt`
/// polling fallback (aligned with pi `shouldPollGitHead`'s watchFile polling).
/// Returns `None` on registration failure (aligned with pi's semantics of giving up when initial setup fails).
fn setup_watcher(
    git_paths: &GitPaths,
    event_tx: &UnboundedSender<FsSignal>,
) -> Option<Box<dyn Watcher + Send>> {
    let reftable_dir = git_paths.common_git_dir.join("reftable");
    let build_callback = |tx: UnboundedSender<FsSignal>| {
        let reftable_dir = reftable_dir.clone();
        move |res: Result<NotifyEvent, NotifyError>| match res {
            Ok(ev) => {
                if event_is_relevant(&ev, &reftable_dir) {
                    let _ = tx.send(FsSignal::Changed);
                }
            }
            Err(_) => {
                let _ = tx.send(FsSignal::WatcherError);
            }
        }
    };
    let poll = should_poll_git_head(&git_paths.repo_dir);
    let mut watcher: Box<dyn Watcher + Send> = if poll {
        let cb = build_callback(event_tx.clone());
        let cfg = notify::Config::default().with_poll_interval(POLL_INTERVAL);
        Box::new(notify::PollWatcher::new(cb, cfg).ok()?)
    } else {
        let cb = build_callback(event_tx.clone());
        Box::new(notify::recommended_watcher(cb).ok()?)
    };
    for path in watch_paths(git_paths) {
        if watcher.watch(&path, RecursiveMode::NonRecursive).is_err() {
            return None;
        }
    }
    Some(watcher)
}

/// Merges consecutive fs events within a 500ms debounce window, then re-resolves once (aligned with pi `scheduleRefresh` +
/// `refreshGitBranchAsync`: more events within the window don't extend the deadline; the debounce is a single refresh).
async fn debounce_and_refresh(
    event_rx: &mut UnboundedReceiver<FsSignal>,
    git_paths: &GitPaths,
    branch_tx: &watch::Sender<Option<String>>,
    cancel: &CancellationToken,
) {
    let sleep = tokio::time::sleep(WATCH_DEBOUNCE_MS);
    tokio::pin!(sleep);
    loop {
        tokio::select! {
            _ = &mut sleep => break,
            _ = event_rx.recv() => {} // More events within the window: keep waiting for the original deadline
            _ = cancel.cancelled() => return,
        }
    }
    // .invalid requires a git subprocess; use spawn_blocking to avoid blocking the runtime
    let paths = git_paths.clone();
    let next = match tokio::task::spawn_blocking(move || resolve_git_branch(&paths)).await {
        Ok(v) => v,
        Err(_) => return, // Task cancelled
    };
    if cancel.is_cancelled() {
        return;
    }
    let current = (*branch_tx.borrow()).clone();
    // Notify only on actual change (aligned with pi notifying branch change only when cachedBranch changes)
    if next != current {
        let _ = branch_tx.send(next);
    }
}

/// Background refresh task: notify callbacks send events through an internal channel → debounce-merge → re-resolve branch → write to watch only on change
/// Consumes channel notifications; on watcher error, rebuilds the watcher after a delay (aligned with pi `FS_WATCH_RETRY_DELAY_MS`).
/// `retry_delay` is a test seam (production passes `WATCHER_RETRY_DELAY`).
/// The local `watcher` is keep-alive storage — held only to keep the notify watch alive (dropping it stops watching),
/// on rebuild, `take()` reads and releases the old value, and the new value is written to be held in the next iteration.
#[allow(clippy::too_many_arguments)]
async fn refresh_loop(
    mut event_rx: UnboundedReceiver<FsSignal>,
    git_paths: GitPaths,
    branch_tx: watch::Sender<Option<String>>,
    cancel: CancellationToken,
    mut watcher: Option<Box<dyn Watcher + Send>>,
    event_tx: UnboundedSender<FsSignal>,
    retry_delay: Duration,
) {
    loop {
        tokio::select! {
            sig = event_rx.recv() => {
                match sig {
                    Some(FsSignal::Changed) => {
                        debounce_and_refresh(&mut event_rx, &git_paths, &branch_tx, &cancel).await;
                    }
                    Some(FsSignal::WatcherError) => {
                        // Treat the old watcher as broken: release it explicitly (aligned with pi clearGitWatchers), rebuild after a delay
                        if watcher.take().is_some() {
                            tracing::debug!(
                                retry_ms = retry_delay.as_millis(),
                                "[footer] git watcher error, rebuilding"
                            );
                        }
                        tokio::select! {
                            _ = tokio::time::sleep(retry_delay) => {
                                watcher = setup_watcher(&git_paths, &event_tx);
                            }
                            _ = cancel.cancelled() => return,
                        }
                    }
                    None => return, // All senders released (watcher dropped when the task ends)
                }
            }
            _ = cancel.cancelled() => return,
        }
    }
}

/// **Single source of truth** for footer branch data — resolves the current branch synchronously and starts file watching on construction; thereafter git metadata
/// changes refresh the watch channel via the background debounce task; consumers (interactive mode) read the latest value synchronously (`git_branch`) and subscribe to
/// changes (`branch_rx`). All fields are pure runtime resources (not serialized), with a lifetime matching the provider,
/// reclaimed by `dispose()`.
pub struct FooterDataProvider {
    /// Channel for the latest branch value (`git_branch()` reads via `Sender::borrow()`; `branch_rx()` hands out via `subscribe()`).
    branch_tx: watch::Sender<Option<String>>,
    /// Background debounce refresh task (holds the notify watcher; released via cancel+abort on dispose).
    refresh_task: Option<JoinHandle<()>>,
    /// Cancellation signal for the refresh task.
    cancel: CancellationToken,
    /// dispose idempotency guard.
    disposed: bool,
}

impl FooterDataProvider {
    /// On interactive-mode init, constructs the branch data provider with the session cwd: resolves the current branch synchronously (in regular repos, only reads HEAD
    /// which is very fast; git is spawned only in the rare .invalid case), registers a notify watcher on the HEAD directory and the reftable
    /// directory, and starts the debounced background refresh task — after which the footer status bar follows user `git switch` automatically.
    /// For a non-repo cwd, constructs an empty provider with no watcher (git_branch is always None, no background task).
    pub fn new(cwd: &Path) -> Self {
        let git_paths = find_git_paths(cwd);
        let initial = git_paths.as_ref().and_then(resolve_git_branch);
        let (branch_tx, _) = watch::channel(initial);
        let cancel = CancellationToken::new();
        let refresh_task = if let Some(paths) = &git_paths {
            let (event_tx, event_rx) = mpsc::unbounded_channel::<FsSignal>();
            // Create the watcher synchronously first (avoiding a race window between initial resolve and watch start), then hand it to the refresh task to hold
            if let Some(watcher) = setup_watcher(paths, &event_tx) {
                let task = tokio::spawn(refresh_loop(
                    event_rx,
                    paths.clone(),
                    branch_tx.clone(),
                    cancel.clone(),
                    Some(watcher),
                    event_tx,
                    WATCHER_RETRY_DELAY,
                ));
                Some(task)
            } else {
                // Watcher registration failed: give up watching (aligned with pi stopping when initial setup fails); branch keeps its initial value
                None
            }
        } else {
            None
        };
        FooterDataProvider {
            branch_tx,
            refresh_task,
            cancel,
            disposed: false,
        }
    }

    /// When assembling `FooterData` each frame, the footer reads the current branch snapshot synchronously (`None` = not in a git repo or read failed;
    /// `Some("detached")` = detached HEAD; otherwise the branch name) — read from the watch cache, no IO and no git calls,
    /// so UI frames don't block (aligned with pi `getGitBranch`'s caching semantics). Consumer: interactive `footer()` assembly.
    pub fn git_branch(&self) -> Option<String> {
        (*self.branch_tx.borrow()).clone()
    }

    /// The interactive main loop registers a branch-change subscription: the returned `watch::Receiver` is used in `dispatch_loop`'s
    /// `select!` with `changed().await`, triggering `push_footer()` to refresh the status bar on branch change — equivalent to pi's
    /// `onBranchChange(cb)` callback subscription, expressed on the Rust side as an event stream (receivers can be cloned freely; multiple consumers share
    /// the same latest value). Consumer: the interactive `dispatch_loop` branch-change arm.
    pub fn branch_rx(&self) -> watch::Receiver<Option<String>> {
        self.branch_tx.subscribe()
    }

    /// Called on interactive-mode shutdown (idempotent): cancels the background refresh task and releases the notify watcher — preventing branch
    /// changes from reaching the destroyed UI after exit (aligned with pi `dispose`; the Rust side also has a `Drop` fallback).
    pub fn dispose(&mut self) {
        if self.disposed {
            return;
        }
        self.disposed = true;
        self.cancel.cancel();
        if let Some(task) = self.refresh_task.take() {
            task.abort();
        }
    }
}

impl Drop for FooterDataProvider {
    fn drop(&mut self) {
        self.dispose();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use notify::event::{CreateKind, EventAttributes, EventKind, ModifyKind, RemoveKind};
    use std::ffi::OsString;
    use std::sync::Mutex as StdMutex;
    use tempfile::TempDir;

    // Env var operations (PATH / WSL detection) are global; serialize them to avoid parallel tests polluting each other (precedent: services.rs)
    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

    /// Temporarily point PATH to `dir` (only that directory, masking real git); restore on Drop.
    struct PathGuard(Option<OsString>);
    impl PathGuard {
        fn set(dir: &Path) -> Self {
            let old = std::env::var_os("PATH");
            std::env::set_var("PATH", dir);
            PathGuard(old)
        }
    }
    impl Drop for PathGuard {
        fn drop(&mut self) {
            match &self.0 {
                Some(v) => std::env::set_var("PATH", v),
                None => std::env::remove_var("PATH"),
            }
        }
    }

    /// Temporarily set/clear WSL environment variables; restore on Drop.
    struct WslEnvGuard {
        old_name: Option<Option<OsString>>,
        old_interop: Option<Option<OsString>>,
    }
    impl WslEnvGuard {
        fn set(name: Option<&str>, interop: Option<&str>) -> Self {
            let old_name = std::env::var_os("WSL_DISTRO_NAME");
            let old_interop = std::env::var_os("WSL_INTEROP");
            match name {
                Some(v) => std::env::set_var("WSL_DISTRO_NAME", v),
                None => std::env::remove_var("WSL_DISTRO_NAME"),
            }
            match interop {
                Some(v) => std::env::set_var("WSL_INTEROP", v),
                None => std::env::remove_var("WSL_INTEROP"),
            }
            WslEnvGuard {
                old_name: Some(old_name),
                old_interop: Some(old_interop),
            }
        }
    }
    impl Drop for WslEnvGuard {
        fn drop(&mut self) {
            if let Some(v) = self.old_name.take() {
                match v {
                    Some(v) => std::env::set_var("WSL_DISTRO_NAME", v),
                    None => std::env::remove_var("WSL_DISTRO_NAME"),
                }
            }
            if let Some(v) = self.old_interop.take() {
                match v {
                    Some(v) => std::env::set_var("WSL_INTEROP", v),
                    None => std::env::remove_var("WSL_INTEROP"),
                }
            }
        }
    }

    /// Regular repository fixture: `<root>/repo/.git/HEAD = ref: refs/heads/main`.
    fn repo_fixture() -> TempDir {
        let dir = tempfile::tempdir().unwrap();
        let git = dir.path().join(".git");
        std::fs::create_dir_all(&git).unwrap();
        std::fs::write(git.join("HEAD"), "ref: refs/heads/main\n").unwrap();
        dir
    }

    /// Write an executable fake git script under `dir` (unix): exit code `exit_code`, output `stdout`.
    #[cfg(unix)]
    fn write_fake_git(dir: &Path, exit_code: i32, stdout: &str) {
        use std::os::unix::fs::PermissionsExt;
        let script = format!(
            "#!/bin/sh\nprintf '%s' '{}'\nexit {}\n",
            stdout.replace('\'', "'\\''"),
            exit_code
        );
        let path = dir.join("git");
        std::fs::write(&path, script).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    // ── parse_head_ref ────────────────────────────────────────────────

    #[test]
    fn parse_head_branch() {
        assert!(matches!(
            parse_head_ref("ref: refs/heads/main\n"),
            HeadRef::Branch(b) if b == "main"
        ));
    }

    #[test]
    fn parse_head_branch_with_slash() {
        assert!(matches!(
            parse_head_ref("ref: refs/heads/feature/foo\n"),
            HeadRef::Branch(b) if b == "feature/foo"
        ));
    }

    #[test]
    fn parse_head_invalid_sentinel() {
        assert!(matches!(
            parse_head_ref("ref: refs/heads/.invalid\n"),
            HeadRef::Invalid
        ));
    }

    #[test]
    fn parse_head_detached_hash() {
        assert!(matches!(
            parse_head_ref("9fceb02d0ae598e95dc970b74767f19372d61af8\n"),
            HeadRef::Detached
        ));
    }

    #[test]
    fn parse_head_empty_or_blank_is_detached() {
        assert!(matches!(parse_head_ref(""), HeadRef::Detached));
        assert!(matches!(parse_head_ref("   \n"), HeadRef::Detached));
    }

    // ── find_git_paths ─────────────────────────────────────────────────

    #[test]
    fn find_paths_regular_repo_from_nested() {
        let dir = repo_fixture();
        let nested = dir.path().join("src/nested");
        std::fs::create_dir_all(&nested).unwrap();
        let p = find_git_paths(&nested).expect("Should discover repository");
        assert_eq!(p.repo_dir, dir.path());
        assert_eq!(p.common_git_dir, dir.path().join(".git"));
        assert_eq!(p.head_path, dir.path().join(".git/HEAD"));
    }

    #[test]
    fn find_paths_non_repo_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        assert!(find_git_paths(dir.path()).is_none());
    }

    #[test]
    fn find_paths_worktree_resolves_commondir() {
        let root = tempfile::tempdir().unwrap();
        let repo = root.path().join("repo");
        let git = repo.join(".git");
        let git_dir = git.join("worktrees/src");
        std::fs::create_dir_all(&git_dir).unwrap();
        std::fs::write(git_dir.join("HEAD"), "ref: refs/heads/.invalid\n").unwrap();
        std::fs::write(git_dir.join("commondir"), "../..\n").unwrap();
        std::fs::create_dir_all(git.join("reftable")).unwrap();
        std::fs::write(git.join("reftable/tables.list"), "0\n").unwrap();
        let wt = root.path().join("worktree");
        std::fs::create_dir_all(&wt).unwrap();
        let git_dir_rel = format!("{}", git_dir.display());
        std::fs::write(wt.join(".git"), format!("gitdir: {git_dir_rel}\n")).unwrap();

        let p = find_git_paths(&wt).expect("Should discover worktree");
        assert_eq!(p.repo_dir, wt);
        // commondir "../.." resolved relative to git_dir, then canonicalized to the main repository .git
        assert_eq!(
            std::fs::canonicalize(&p.common_git_dir).unwrap(),
            std::fs::canonicalize(&git).unwrap()
        );
        assert_eq!(p.head_path, git_dir.join("HEAD"));
    }

    #[test]
    fn find_paths_worktree_without_commondir_uses_gitdir() {
        let root = tempfile::tempdir().unwrap();
        let git = root.path().join("repo/.git");
        let git_dir = git.join("worktrees/src");
        std::fs::create_dir_all(&git_dir).unwrap();
        std::fs::write(git_dir.join("HEAD"), "ref: refs/heads/main\n").unwrap();
        let wt = root.path().join("worktree");
        std::fs::create_dir_all(&wt).unwrap();
        let git_dir_rel = format!("{}", git_dir.display());
        std::fs::write(wt.join(".git"), format!("gitdir: {git_dir_rel}\n")).unwrap();

        let p = find_git_paths(&wt).expect("Should discover worktree");
        assert_eq!(
            std::fs::canonicalize(&p.common_git_dir).unwrap(),
            std::fs::canonicalize(&git_dir).unwrap()
        );
    }

    #[test]
    fn find_paths_git_file_not_gitdir_walks_up() {
        let root = tempfile::tempdir().unwrap();
        // Outer layer has a real repository; inner .git is a file but its content is not a gitdir:
        let repo = root.path().join("repo");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        std::fs::write(repo.join(".git/HEAD"), "ref: refs/heads/main\n").unwrap();
        let inner = repo.join("sub");
        std::fs::create_dir_all(&inner).unwrap();
        std::fs::write(inner.join(".git"), "not a gitdir marker\n").unwrap();
        let p = find_git_paths(&inner).expect("Should ascend to outer repository");
        assert_eq!(p.repo_dir, repo);
    }

    #[test]
    fn find_paths_git_dir_without_head_returns_none() {
        let root = tempfile::tempdir().unwrap();
        let repo = root.path().join("repo");
        std::fs::create_dir_all(repo.join(".git")).unwrap(); // No HEAD
        assert!(find_git_paths(&repo).is_none());
    }

    // ── resolve_git_branch (path without the git binary) ────────────────────────

    #[test]
    fn resolve_branch_reads_head_directly() {
        let dir = repo_fixture();
        let paths = find_git_paths(dir.path()).unwrap();
        assert_eq!(resolve_git_branch(&paths).as_deref(), Some("main"));
    }

    #[test]
    fn resolve_branch_detached() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        std::fs::write(
            dir.path().join(".git/HEAD"),
            "9fceb02d0ae598e95dc970b74767f19372d61af8\n",
        )
        .unwrap();
        let paths = find_git_paths(dir.path()).unwrap();
        assert_eq!(resolve_git_branch(&paths).as_deref(), Some("detached"));
    }

    // ── .invalid sentinel → git fallback (fake git script, does not touch real git) ────────────

    #[cfg(unix)]
    #[test]
    fn invalid_sentinel_resolves_via_git() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".git/HEAD"), "ref: refs/heads/.invalid\n").unwrap();
        let fake = tempfile::tempdir().unwrap();
        write_fake_git(fake.path(), 0, "main\n");
        let _path = PathGuard::set(fake.path());
        let paths = find_git_paths(dir.path()).unwrap();
        assert_eq!(resolve_git_branch(&paths).as_deref(), Some("main"));
    }

    #[cfg(unix)]
    #[test]
    fn invalid_sentinel_git_detached_falls_back() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".git/HEAD"), "ref: refs/heads/.invalid\n").unwrap();
        let fake = tempfile::tempdir().unwrap();
        write_fake_git(fake.path(), 1, "");
        let _path = PathGuard::set(fake.path());
        let paths = find_git_paths(dir.path()).unwrap();
        assert_eq!(resolve_git_branch(&paths).as_deref(), Some("detached"));
    }

    #[cfg(unix)]
    #[test]
    fn invalid_sentinel_no_git_falls_back() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".git/HEAD"), "ref: refs/heads/.invalid\n").unwrap();
        let empty = tempfile::tempdir().unwrap(); // No git in the directory
        let _path = PathGuard::set(empty.path());
        let paths = find_git_paths(dir.path()).unwrap();
        assert_eq!(resolve_git_branch(&paths).as_deref(), Some("detached"));
    }

    // ── WSL / mount path detection ─────────────────────────────────────────

    #[test]
    fn windows_mounted_paths() {
        assert!(is_windows_mounted_repo_path(Path::new("/mnt/c/Users/x")));
        assert!(is_windows_mounted_repo_path(Path::new("/mnt/d")));
        assert!(!is_windows_mounted_repo_path(Path::new("/home/user")));
        assert!(!is_windows_mounted_repo_path(Path::new("/mnt/ab/c")));
        assert!(!is_windows_mounted_repo_path(Path::new("/mnt/")));
        assert!(!is_windows_mounted_repo_path(Path::new("mnt/c")));
    }

    #[test]
    fn wsl_environment_detection() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // No WSL variables by default
        let _g = WslEnvGuard::set(None, None);
        assert!(!is_wsl_environment());
        drop(_g);
        let _g = WslEnvGuard::set(Some("Debian"), None);
        assert!(is_wsl_environment());
        drop(_g);
        let _g = WslEnvGuard::set(None, Some("1"));
        assert!(is_wsl_environment());
    }

    #[test]
    fn poll_head_only_in_wsl_mounted() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _g = WslEnvGuard::set(Some("Debian"), None);
        assert!(should_poll_git_head(Path::new("/mnt/c/Users/x")));
        assert!(!should_poll_git_head(Path::new("/home/user/proj")));
        drop(_g);
        let _g = WslEnvGuard::set(None, None);
        assert!(!should_poll_git_head(Path::new("/mnt/c/Users/x")));
    }

    // ── event_is_relevant (notify event filtering) ─────────────────────────

    fn ev(kind: EventKind, paths: Vec<&str>) -> NotifyEvent {
        NotifyEvent {
            kind,
            paths: paths.into_iter().map(PathBuf::from).collect(),
            attrs: EventAttributes::default(),
        }
    }

    #[test]
    fn event_relevance_filters_non_head() {
        let reftable = Path::new("/repo/.git/reftable");
        let modify = EventKind::Modify(ModifyKind::Any);
        assert!(event_is_relevant(
            &ev(modify, vec!["/repo/.git/HEAD"]),
            reftable
        ));
        assert!(!event_is_relevant(
            &ev(modify, vec!["/repo/.git/index"]),
            reftable
        ));
        assert!(!event_is_relevant(
            &ev(modify, vec!["/repo/.git/config"]),
            reftable
        ));
        assert!(event_is_relevant(
            &ev(modify, vec!["/repo/.git/reftable/tables.list"]),
            reftable
        ));
        assert!(event_is_relevant(
            &ev(EventKind::Create(CreateKind::Any), vec!["/repo/.git/HEAD"]),
            reftable
        ));
        assert!(event_is_relevant(
            &ev(EventKind::Remove(RemoveKind::Any), vec!["/repo/.git/HEAD"]),
            reftable
        ));
        // Empty path (not reported by backend) is conservatively treated as relevant
        assert!(event_is_relevant(&ev(modify, vec![]), reftable));
    }

    // ── FooterDataProvider integration (real notify watcher + debounce task) ──────────

    #[tokio::test]
    async fn provider_reports_branch_in_repo() {
        let dir = repo_fixture();
        let mut p = FooterDataProvider::new(dir.path());
        assert_eq!(p.git_branch().as_deref(), Some("main"));
        p.dispose();
        assert!(p.git_branch().is_some()); // Cached value is still readable after dispose
    }

    #[tokio::test]
    async fn provider_none_outside_repo() {
        let dir = tempfile::tempdir().unwrap();
        let mut p = FooterDataProvider::new(dir.path());
        assert_eq!(p.git_branch(), None);
        p.dispose();
    }

    #[tokio::test]
    async fn provider_reports_detached() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        std::fs::write(
            dir.path().join(".git/HEAD"),
            "9fceb02d0ae598e95dc970b74767f19372d61af8\n",
        )
        .unwrap();
        let mut p = FooterDataProvider::new(dir.path());
        assert_eq!(p.git_branch().as_deref(), Some("detached"));
        p.dispose();
    }

    #[tokio::test]
    async fn provider_notifies_on_branch_change() {
        let dir = repo_fixture();
        let mut p = FooterDataProvider::new(dir.path());
        let mut rx = p.branch_rx();
        // Write HEAD → change detected → watch notification
        std::fs::write(dir.path().join(".git/HEAD"), "ref: refs/heads/dev\n").unwrap();
        tokio::time::timeout(Duration::from_secs(5), rx.changed())
            .await
            .expect("Branch change notification timed out")
            .expect("sender alive; changed should not be Err");
        assert_eq!(p.git_branch().as_deref(), Some("dev"));
        assert_eq!((*rx.borrow()).as_deref(), Some("dev"));
        p.dispose();
    }

    #[tokio::test]
    async fn provider_no_notify_on_same_branch() {
        let dir = repo_fixture();
        let mut p = FooterDataProvider::new(dir.path());
        let mut rx = p.branch_rx();
        // Rewrite the same HEAD (redundant event from git atomic write)
        std::fs::write(dir.path().join(".git/HEAD"), "ref: refs/heads/main\n").unwrap();
        // Wait for debounce + refresh window to pass
        tokio::time::sleep(Duration::from_millis(1200)).await;
        let timed_out = tokio::time::timeout(Duration::from_millis(100), rx.changed())
            .await
            .is_err();
        assert!(timed_out, "Same branch should not trigger change notification");
        p.dispose();
    }

    #[tokio::test]
    async fn provider_debounces_rapid_updates_to_single_notify() {
        let dir = repo_fixture();
        let mut p = FooterDataProvider::new(dir.path());
        let mut rx = p.branch_rx();
        // Rapidly write multiple branches (simulating git atomic writes) — should coalesce into one refresh within the debounce window
        for b in ["dev", "fix", "hotfix"] {
            std::fs::write(
                dir.path().join(".git/HEAD"),
                format!("ref: refs/heads/{b}\n"),
            )
            .unwrap();
        }
        let first = tokio::time::timeout(Duration::from_secs(5), rx.changed())
            .await
            .expect("First change notification timed out");
        assert!(first.is_ok());
        // Coalesced refresh reads the last write
        assert_eq!(p.git_branch().as_deref(), Some("hotfix"));
        // No second notification within the debounce window (same-branch re-resolution does not send)
        tokio::time::sleep(Duration::from_millis(1200)).await;
        let second = tokio::time::timeout(Duration::from_millis(100), rx.changed())
            .await
            .is_err();
        assert!(second, "Debounce should coalesce into a single notification");
        p.dispose();
    }

    #[tokio::test]
    async fn provider_dispose_stops_notifications() {
        let dir = repo_fixture();
        let mut p = FooterDataProvider::new(dir.path());
        let mut rx = p.branch_rx();
        p.dispose();
        std::fs::write(
            dir.path().join(".git/HEAD"),
            "ref: refs/heads/after-dispose\n",
        )
        .unwrap();
        tokio::time::sleep(Duration::from_millis(1200)).await;
        let timed_out = tokio::time::timeout(Duration::from_millis(100), rx.changed())
            .await
            .is_err();
        assert!(timed_out, "Should no longer notify after dispose");
    }

    #[tokio::test]
    async fn refresh_loop_retries_watcher_after_error() {
        let dir = repo_fixture();
        let git_paths = find_git_paths(dir.path()).unwrap();
        let (event_tx, event_rx) = mpsc::unbounded_channel::<FsSignal>();
        let (branch_tx, mut rx) = watch::channel(Some("main".to_string()));
        let cancel = CancellationToken::new();
        let watcher = setup_watcher(&git_paths, &event_tx).expect("watcher should register successfully");
        let task = tokio::spawn(refresh_loop(
            event_rx,
            git_paths.clone(),
            branch_tx,
            cancel.clone(),
            Some(watcher),
            event_tx.clone(),
            Duration::from_millis(50), // Short retry delay (production 5s)
        ));
        // Inject watcher error → task drops the old watcher and rebuilds after a delay
        let _ = event_tx.send(FsSignal::WatcherError);
        tokio::time::sleep(Duration::from_millis(150)).await;
        // After successful rebuild, new branch changes should be observed
        std::fs::write(
            dir.path().join(".git/HEAD"),
            "ref: refs/heads/retry-branch\n",
        )
        .unwrap();
        tokio::time::timeout(Duration::from_secs(5), rx.changed())
            .await
            .expect("No change detected after rebuild")
            .expect("sender alive; changed should not be Err");
        assert_eq!((*rx.borrow()).as_deref(), Some("retry-branch"));
        cancel.cancel();
        task.abort();
    }
}
