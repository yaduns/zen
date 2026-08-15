use std::time::Duration;
use tokio::io::AsyncReadExt;

/// Grace period (ms) that stdout/stderr remains idle after exit, aligned with pi `EXIT_STDIO_GRACE_MS = 100`.
const EXIT_STDIO_GRACE_MS: u64 = 100;

/// Process-level detached child process pid registry (module-private).
/// `LazyLock` initializes on first access; `Mutex` ensures concurrency safety; `HashSet` provides O(1) deduplicated registration.
// About Mutex:
// 1. Make static contents mutable. static is immutable; cannot directly &mut the HashSet inside. Mutex provides interior mutability: get &mut HashSet via lock()
// 2. Multi-thread concurrency safety.
// 3. Uses std::sync::Mutex (blocking lock), not tokio::sync::Mutex. This is intentional — track/untrack are both synchronous, O(1) short operations
// and never cross .await while holding the lock (does not block the async runtime), so no async mutex is needed; std::sync::Mutex is zero-overhead and simple.
static REGISTRY: std::sync::LazyLock<std::sync::Mutex<std::collections::HashSet<u32>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashSet::new()));

/// Reads the next chunk from `reader` (may be `None`, meaning the stream is disabled or already ended).
///
/// When `None`, returns a future that never completes, to be used with the outer `select!`'s `if !..._eof`
/// guard (the guard guarantees `reader` is always `Some` when actually polled; this branch is only a defensive fallback).
async fn read_chunk<R: tokio::io::AsyncRead + Unpin>(
    reader: Option<&mut R>,
    buf: &mut [u8],
) -> std::io::Result<usize> {
    match reader {
        Some(r) => r.read(buf).await,
        // std::future::pending() returns a future that never completes (always Pending); awaiting it means "hang forever, never get a result".
        // So when reader is None, the entire read_chunk: reads nothing, returns nothing, and just waits indefinitely.
        //
        // tokio::select! polls all branches concurrently; whoever becomes Ready first wins. If read_chunk(None, ...) were not suspending but returned immediately (even returning Ok(0) pretending to be EOF), the consequences would be catastrophic:
        // - on every select! poll, this branch becomes Ready immediately -> wins immediately;
        // - so it wins every time, and the other branches (stderr, child.wait(), the phase-two 100ms timer) never get a turn — either spinning in an infinite loop or starving other signals.
        // So for the case of "no stream to read", the correct behavior is: yield no result, and let the race continue to others. pending() exists precisely for this — a future that is never Ready, "holding a slot but not competing" in the race.
        None => std::future::pending().await,
    }
}

/// stdout/stderr chunk callback type: called on each raw byte chunk received; infallible contract (must not panic).
/// `pub(crate)`: The `ExecutionEnv::exec` in `env/mod.rs` (T-AP-04-2) also uses it to declare the byte callback, providing a single source of truth.
pub(crate) type ChunkCallback = dyn Fn(&[u8]) + Send + Sync;

/// Waits for the child process to exit while draining its stdout/stderr pipes, and does not hang on the inherited stdio handles held by detached descendants
/// (restores the pi#5303 fix semantics, see
/// reference/pi-original/packages/coding-agent/src/utils/child-process.ts for
/// `waitForChildProcess`）。
///
/// # Behavior
/// Drains the `child`'s stdout/stderr pipes to EOF, concurrently invoking the corresponding callbacks to deliver raw bytes
/// (Rust pipes are exclusive; the callback is the equivalent of the caller separately attaching a `data` listener in pi; the caller decides
/// decode/accumulate strategy). Wait until the process `exit`s; after `exit`:
/// - both pipes reached EOF → return the exit code immediately;
/// - otherwise start a 100ms grace timer, **re-armed on every received chunk** — descendants still actively writing
///   keep this function reading (no trailing output is lost), while handles held silently are still released after the grace period
///   (no permanent hang).
///
/// # Parameters
/// - `child`: a process spawned with piped stdout/stderr (`stdout`/`stderr` being
///   `None` is treated as "already finished", matching pi's initial `child.stdout === null`;
///   recommended to spawn with `process_group(0)` on unix so the caller can terminate the whole process tree).
/// - `on_stdout`/`on_stderr`: optional callbacks with an infallible contract (must not panic), invoked
///   on each received raw byte chunk; `None` means discard that stream.
///
/// # Returns
/// - `Ok(Some(code))`: exit code of a normal exit;
/// - `Ok(None)`: no exit code (e.g. terminated by signal), matching pi's `number | null`;
/// - `Err(io::Error)`: fatal error (e.g. `wait()` failure). Consumers may coalesce per pi semantics;
///   (exec.ts maps errors to code 1).
///
/// # Edge Cases
/// - Callbacks may still be invoked briefly after the grace timer is released (bytes already queued within the race window);
///   callers must not assume "no more callbacks after return".
pub async fn wait_for_child_process(
    child: &mut tokio::process::Child,
    on_stdout: Option<&ChunkCallback>,
    on_stderr: Option<&ChunkCallback>,
) -> Result<Option<i32>, std::io::Error> {
    // Take the pipe handles; `None` (e.g. `Stdio::null`) is treated as "already finished", matching pi's initial `stdout === null`.
    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();
    let mut stdout_eof = stdout.is_none();
    let mut stderr_eof = stderr.is_none();
    let mut stdout_buf = [0u8; 8192];
    let mut stderr_buf = [0u8; 8192];

    // Phase 1: drain both pipes while waiting for the process to exit (matching pi's `exit` event). `wait()`
    // only waits for process termination, not pipe closure, so we must read concurrently to avoid the child filling the pipe and deadlocking;
    // buffered data / handles held by descendants may remain after exit — handled in phase 2.
    let status = loop {
        tokio::select! {
            r = read_chunk(stdout.as_mut(), &mut stdout_buf), if !stdout_eof => {
                match r {
                    Ok(0) => {
                        stdout_eof = true; // What's modified is the bool flag. Its only consumer is the select! branch guard `if !stdout_eof` — it tells the loop "this stream is finished, stop polling this branch."
                        stdout = None; // Set Option<ChildStdout> to None, dropping the real pipe read handle (ChildStdout), which closes the read end on the parent side and releases the corresponding file descriptor. This is the actual resource cleanup.
                    }
                    Ok(n) => {
                        if let Some(cb) = on_stdout {
                            cb(&stdout_buf[..n]);
                        }
                    }
                    Err(e) => return Err(e),
                }
            }
            r = read_chunk(stderr.as_mut(), &mut stderr_buf), if !stderr_eof => {
                match r {
                    Ok(0) => {
                        stderr_eof = true;
                        stderr = None;
                    }
                    Ok(n) => {
                        if let Some(cb) = on_stderr {
                            cb(&stderr_buf[..n]);
                        }
                    }
                    Err(e) => return Err(e),
                }
            }
            status = child.wait() => break status,
        }
    };
    let status = status?; // wait() failure → fatal error
    let exit_code = status.code(); // Terminated by signal → None (matching pi `number | null`)

    // After exit both pipes are already EOF → return immediately, no grace delay is introduced (matching pi's close semantics).
    if stdout_eof && stderr_eof {
        return Ok(exit_code);
    }

    // Phase two: sliding idle grace. A fresh 100ms timer is created each loop iteration — any arriving data chunk causes the loop
    // to restart (equivalent to re-arming the timer), so a descendant that is still actively writing keeps this function reading
    // (no trailing output is lost); a descendant silently holding the handle is released after 100ms of idle in that round (no permanent hang).
    loop {
        if stdout_eof && stderr_eof {
            return Ok(exit_code);
        }
        let idle = tokio::time::sleep(Duration::from_millis(EXIT_STDIO_GRACE_MS));
        tokio::pin!(idle);
        tokio::select! {
            r = read_chunk(stdout.as_mut(), &mut stdout_buf), if !stdout_eof => {
                match r {
                    Ok(0) => {
                        stdout_eof = true;
                        stdout = None;
                    }
                    Ok(n) => {
                        if let Some(cb) = on_stdout {
                            cb(&stdout_buf[..n]);
                        }
                    }
                    Err(e) => return Err(e),
                }
            }
            r = read_chunk(stderr.as_mut(), &mut stderr_buf), if !stderr_eof => {
                match r {
                    Ok(0) => {
                        stderr_eof = true;
                        stderr = None;
                    }
                    Ok(n) => {
                        if let Some(cb) = on_stderr {
                            cb(&stderr_buf[..n]);
                        }
                    }
                    Err(e) => return Err(e),
                }
            }
            // The select! branch uses &mut idle, i.e. "poll this future through a mutable reference".
            // But polling a !Unpin future requires Pin — you can't just &mut it directly.
            // tokio::pin! does exactly this: it pins the !Unpin Sleep in place, yielding Pin<&mut Sleep>,
            // and Pin<&mut Sleep> is Unpin (because &mut references are always Unpin),
            // so &mut idle can be used as the select! branch's future and polled normally.
            //
            // Grace period elapsed, no new data arrived within these 100ms → the pipe is judged fully idle → finish up with the exit code obtained in phase one
            _ = &mut idle => return Ok(exit_code),
        }
    }
}

/// Register a detached child process pid (aligned with pi `trackDetachedChildPid`).
///
/// # Behavior
/// Add `pid` to the process-level registry `REGISTRY` (`REGISTRY.lock().insert(pid)`); once registered, the pid
/// will be batch-terminated by `kill_tracked_detached_children` when the parent receives a shutdown signal (SIGHUP/SIGTERM).
/// The registry is process-global (`LazyLock<Mutex<HashSet<u32>>>`) and idempotent: registering the same pid repeatedly has no side effects.
///
/// # Parameters
/// - `pid`: the pid of a child started in its own process group (unix `process_group(0)`, aligning with pi `detached: true`)
///   child process pid, ensuring `kill_process_tree` can terminate the whole tree by process group.
///
/// # Edge Cases
/// - Only registers; does not validate pid validity. Callers must register immediately after successful spawn and, after the child exits normally, use
///   `untrack_detached_child_pid` to remove it, preventing pid reuse from killing unrelated processes by mistake.
pub fn track_detached_child_pid(pid: u32) {
    // Poisoned-lock recovery: if a thread panics while holding the lock, `.lock()` returns Err; use `into_inner()` to recover
    // the internal data and continue registering instead of unwrap-panicking — this is the shutdown cleanup path; registration must never be lost due to a panic,
    // otherwise the detached child process will not be killed when the parent exits (AGENTS.md rule 4).
    REGISTRY
        .lock()
        // into_inner() recovers the internal HashSet for continued use — registration/cleanup paths must "best-effort"; prefer continuing over panicking
        .unwrap_or_else(|e| e.into_inner())
        .insert(pid);
}

/// Remove a pid from the registry (aligned with pi `untrackDetachedChildPid`).
///
/// # Behavior
/// Remove `pid` from the process-level registry `REGISTRY` (`REGISTRY.lock().remove(&pid)`) so it is no longer
/// terminated by `kill_tracked_detached_children`. Idempotent: no side effects when the pid is not in the registry.
///
/// # Parameters
/// - `pid`: child process pid to remove (usually a process that has already exited normally).
///
/// # Edge Cases
/// - Must be called after the child process exits; otherwise the pid may be reused by the system, leading to killing unrelated processes later.
pub fn untrack_detached_child_pid(pid: u32) {
    REGISTRY
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&pid);
}

/// Terminate all registered detached child process trees and clear the registry (aligned with pi `killTrackedDetachedChildren`).
///
/// # Behavior
/// Lock `REGISTRY`, take a snapshot, then `clear()`, and for each pid in the snapshot call
/// `crate::env::shell::kill_process_tree` (unix sends SIGKILL to process group `-pid`; windows spawns
/// `taskkill /T /F`）。
///
/// # When to Call
/// Called from the parent process's signal (SIGHUP/SIGTERM) handlers: print-mode receives SIGTERM/SIGHUP,
/// on interactive-mode terminal loss/close and RPC shutdown (aligning with pi's signal handling per mode).
///
/// # Edge Cases
/// - `kill_process_tree` is best-effort and does not throw (the process may have already exited); this function likewise returns no error.
/// - After the call the registry is empty; subsequent `track_detached_child_pid` calls start registering again.
pub fn kill_tracked_detached_children() {
    // Take snapshot and clear first, then kill one by one outside the lock — avoid holding the lock for long (kill is an external syscall)
    let pids: Vec<u32> = {
        let mut reg = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
        let pids = reg.iter().copied().collect();
        reg.clear();
        pids
    };
    for pid in pids {
        crate::env::shell::kill_process_tree(pid); // Depends on the T-AG environment shell task; this call is left pending implementation until it lands
    }
}

/// For tests: serialize all tests that touch the global `REGISTRY`.
///
/// `REGISTRY` is a process-level global static; Rust's parallel test execution would let tests overwrite each other's registrations,
/// so all tests that touch the registry (this module's track/untrack/kill_tracked tests, and
/// the detached-tracking test for `ExecutionEnv::exec` in `env/mod.rs`—the new consumer added by T-AP-04-2)
/// must be serialized through **the same lock**, otherwise each other's kill_tracked may kill the other's processes,
/// or the other's registration may cause the "registry should be empty" assertion to fail due to a race.
#[cfg(test)]
pub(crate) fn registry_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::LazyLock<std::sync::Mutex<()>> =
        std::sync::LazyLock::new(|| std::sync::Mutex::new(()));
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    /// Constructs the accumulating callback: returns (accumulation buffer, callback object); pass it in as `Some(cb.as_ref())` when invoking.
    fn collector() -> (Arc<Mutex<Vec<u8>>>, Box<ChunkCallback>) {
        let shared = Arc::new(Mutex::new(Vec::new()));
        let captured = shared.clone();
        let cb = Box::new(move |chunk: &[u8]| captured.lock().unwrap().extend_from_slice(chunk));
        (shared, cb)
    }

    // ── Normal path ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_waits_and_returns_exit_code() {
        let mut child = tokio::process::Command::new("sh")
            .arg("-c")
            .arg("exit 3")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn");
        let code = wait_for_child_process(&mut child, None, None)
            .await
            .expect("wait ok");
        assert_eq!(code, Some(3));
    }

    #[tokio::test]
    async fn test_captures_stdout_and_stderr() {
        let mut child = tokio::process::Command::new("sh")
            .arg("-c")
            .arg("printf 'out\\n'; printf 'err\\n' >&2")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn");
        let (out, out_cb) = collector();
        let (err, err_cb) = collector();
        let code = wait_for_child_process(&mut child, Some(out_cb.as_ref()), Some(err_cb.as_ref()))
            .await
            .expect("wait ok");
        assert_eq!(code, Some(0));
        assert_eq!(&*out.lock().unwrap(), b"out\n");
        assert_eq!(&*err.lock().unwrap(), b"err\n");
    }

    #[tokio::test]
    async fn test_returns_none_when_killed_by_signal() {
        let mut child = tokio::process::Command::new("sh")
            .arg("-c")
            .arg("kill -9 $$")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn");
        let code = wait_for_child_process(&mut child, None, None)
            .await
            .expect("wait ok");
        // On signal termination tokio's ExitStatus::code() is None (on unix) → matching pi `number | null`
        assert_eq!(code, None);
    }

    // ── Regression pi#5303: handles still active/silently held after exit ─────────────

    #[tokio::test]
    async fn test_captures_output_emitted_after_exit_while_detached_holds_stdout() {
        // The shell exits immediately, but a background sub-shell writes TICK every 50ms, the last one far after
        // the 100ms grace; if the grace is not re-armed, the trailing output gets truncated (the original pi#5303 bug).
        let command =
            "printf 'HEAD\\n'; ( for i in 1 2 3 4 5 6; do sleep 0.05; printf \"TICK$i\\n\"; done ) &";
        let mut child = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn");
        let (out, out_cb) = collector();
        let code = wait_for_child_process(&mut child, Some(out_cb.as_ref()), None)
            .await
            .expect("wait ok");
        assert_eq!(code, Some(0));
        let output = String::from_utf8_lossy(&out.lock().unwrap()).into_owned();
        assert!(output.contains("HEAD"), "HEAD missing: {output:?}");
        assert!(output.contains("TICK6"), "TICK6 missing: {output:?}");
    }

    #[tokio::test]
    async fn test_resolves_promptly_when_detached_holds_stdout_quietly() {
        // After the shell exits, the background sleeper inherits the stdout pipe and keeps it open indefinitely but writes no more data;
        // release must rely on the idle grace timeout rather than hanging on the open handle.
        let command = "printf 'DONE\\n'; ( sleep 30 ) &";
        let mut child = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn");
        let (out, out_cb) = collector();
        let start = std::time::Instant::now();
        let code = wait_for_child_process(&mut child, Some(out_cb.as_ref()), None)
            .await
            .expect("wait ok");
        let elapsed = start.elapsed();
        assert_eq!(code, Some(0));
        let output = String::from_utf8_lossy(&out.lock().unwrap()).into_owned();
        assert!(output.contains("DONE"));
        // Don't wait for the 30s sleeper: the idle grace should release within one second
        assert!(
            elapsed < Duration::from_secs(2),
            "took too long: {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn test_null_streams_treated_as_ended() {
        // stdout/stderr are Stdio::null (matching pi's `child.stdout === null`):
        // treat as already exited, only await the exit code.
        let mut child = tokio::process::Command::new("sh")
            .arg("-c")
            .arg("exit 0")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn");
        let code = wait_for_child_process(&mut child, None, None)
            .await
            .expect("wait ok");
        assert_eq!(code, Some(0));
    }

    #[tokio::test]
    async fn test_large_output_not_truncated() {
        // Large output: after exit the pipe buffers a lot of data, all of which must be read (re-arming guarantees no truncation).
        let n = 200_000;
        let mut child = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(format!("head -c {n} /dev/zero | tr '\\0' 'a'"))
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn");
        let (out, out_cb) = collector();
        let code = wait_for_child_process(&mut child, Some(out_cb.as_ref()), None)
            .await
            .expect("wait ok");
        assert_eq!(code, Some(0));
        assert_eq!(out.lock().unwrap().len(), n);
    }

    // ── detached child process registry (track / untrack / kill_tracked) ────────
    // REGISTRY is a process-level global static: parallel Rust tests will overwrite each other's registrations,
    // so all tests that touch the registry are serialized first via registry_guard() (a module-level shared lock;
    // the exec tracking test in env/mod.rs also reuses it via crate::process::registry_guard());
    // and use synthetic values far exceeding real pids (pid limits: Linux ~2^22=4194304, macOS 99999)
    // Avoid accidentally killing the real process group.

    #[test]
    fn test_track_detached_pid_dedups_and_records() {
        let _g = registry_guard();
        let (p1, p2) = (100_000_000u32, 100_000_001u32);
        untrack_detached_child_pid(p1); // Defensive cleanup of registrations possibly left by the previous test case
        untrack_detached_child_pid(p2);

        track_detached_child_pid(p1);
        track_detached_child_pid(p1); // Idempotent: duplicate registration still results in 1 entry
        track_detached_child_pid(p2);

        let reg = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(reg.len(), 2, "duplicate pid should be registered only once");
        assert!(reg.contains(&p1));
        assert!(reg.contains(&p2));
        drop(reg);

        // Clean up and keep the registry empty
        untrack_detached_child_pid(p1);
        untrack_detached_child_pid(p2);
    }

    #[test]
    fn test_untrack_detached_pid_removes_idempotently() {
        let _g = registry_guard();
        let (p1, p2) = (100_000_002u32, 100_000_003u32);
        untrack_detached_child_pid(p1);
        untrack_detached_child_pid(p2);

        track_detached_child_pid(p1);
        untrack_detached_child_pid(p1);
        let reg = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
        assert!(!reg.contains(&p1));
        drop(reg);

        // Idempotent: duplicate removal and removal of a nonexistent pid both have no side effects
        untrack_detached_child_pid(p1);
        untrack_detached_child_pid(p2);
        let reg = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
        assert!(reg.is_empty());
        drop(reg);
    }

    #[test]
    fn test_kill_tracked_clears_registry_and_survives_dead_pids() {
        let _g = registry_guard();
        let (p1, p2) = (100_000_004u32, 100_000_005u32);
        untrack_detached_child_pid(p1);
        untrack_detached_child_pid(p2);

        track_detached_child_pid(p1);
        track_detached_child_pid(p2);

        // Nonexistent process group: kill_process_tree is best-effort and does not throw
        kill_tracked_detached_children();

        let reg = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
        assert!(reg.is_empty(), "registry should be empty after kill_tracked");
        drop(reg);

        // Can re-register after clearing
        track_detached_child_pid(p1);
        let reg = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
        assert!(reg.contains(&p1));
        drop(reg);
        untrack_detached_child_pid(p1);
    }

    #[cfg(unix)]
    #[test]
    fn test_kill_tracked_terminates_real_detached_process() {
        use std::os::unix::process::CommandExt;

        let _g = registry_guard();

        // Start a long-sleeping process in a separate process group (aligns with pi detached: true + process_group(0))
        let mut child = std::process::Command::new("sh")
            .arg("-c")
            .arg("sleep 30")
            .process_group(0)
            .spawn()
            .expect("spawn detached sleeper");
        let pid = child.id();

        track_detached_child_pid(pid);
        kill_tracked_detached_children();

        // Reap the child process and assert it was terminated by SIGKILL (avoid leaving zombies)
        let mut status: libc::c_int = 0;
        let reaped = unsafe { libc::waitpid(pid as libc::pid_t, &mut status, 0) };
        assert_eq!(
            reaped, pid as libc::pid_t,
            "should reap killed detached child processes"
        );
        assert!(libc::WIFSIGNALED(status), "child process should be terminated by signal");
        assert_eq!(libc::WTERMSIG(status), libc::SIGKILL);
    }
}
