pub mod shell;
pub mod shell_output;

use crate::env::shell::{kill_process_tree, resolve_shell, ShellTransport};
use crate::process::{wait_for_child_process, ChunkCallback, ProcessRegistry};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

/// Shell override config for the bash/exec execution path (matches pi `shellPath`/`shellCommandPrefix`).
/// Pure parameter carrier: `shell_path` overrides the default shell binary; `command_prefix` is prepended to every command
/// (assembles `{prefix}\n{command}`, matching `resolvedCommand` in pi `agent-session.ts`);
/// All None = system default (e.g. `sh`).
/// Consumers: `BashTool` (zen-core tools), `AgentSession::execute_bash` (zen-app).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ShellConfig {
    /// Custom shell path (None = system default); `~` expansion is done at the assembly layer via `paths::resolve_path`
    pub shell_path: Option<PathBuf>,
    /// Prefix string prepended to each bash command (None = no prefix)
    pub command_prefix: Option<String>,
}

/// Binds one session to a platform-agnostic shell execution environment, holding cwd, an optional custom shell path, and the session's base environment variables;
/// exec runs commands in the platform self-discovered shell using this environment as context.
pub struct ExecutionEnv {
    /// Session working directory; exec's relative cwd option resolves against it
    base_dir: PathBuf,
    /// Custom shell path (e.g. user-configured); validated lazily at exec time
    shell_path: Option<PathBuf>,
    /// Session base environment variable overrides, layered on top of the process environment, with lower precedence than ExecOptions.env
    base_env: Option<HashMap<String, String>>,
    /// detached child-process registry; when `Some`, exec tracks/untracks the child-process pid,
    /// Used by the parent process to batch-terminate child processes upon receiving a shutdown signal; no tracking when `None`.
    registry: Option<Arc<ProcessRegistry>>,
}

/// All options for a single exec call. Lifetimes to one invocation
pub struct ExecOptions {
    /// Command working directory; relative paths resolve against ExecutionEnv.cwd, default is cwd
    cwd: Option<PathBuf>,
    /// Additional env var overrides, highest priority (process env < base_env < env)
    env: Option<HashMap<String, String>>,
    /// Timeout; on expiry the process tree is killed and ExecutionError::Timeout is returned
    timeout: Option<Duration>,
    /// Cancellation token; on cancellation the process tree is killed and ExecutionError::Aborted is returned
    cancellation: Option<CancellationToken>,
    /// stdout text chunk callback (infallible contract; callers must not panic)
    on_stdout: Option<Box<dyn Fn(&str) + Send + Sync>>,
    /// stderr text chunk callback (same as above)
    on_stderr: Option<Box<dyn Fn(&str) + Send + Sync>>,
}

/// Accumulated result of one exec, used as the Result::Ok payload
#[derive(Debug)]
pub struct ExecOutput {
    /// Accumulated stdout (UTF-8)
    stdout: String,
    /// Accumulated stderr (UTF-8)
    stderr: String,
    /// Process exit code; 0 when terminated by signal (ExitStatus::code() is None)
    exit_code: i32,
}

#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    /// Command interrupted by a cancellation token (CancellationToken)
    #[error("command aborted")]
    Aborted,

    /// Command execution timed out; {0} is the timeout duration
    #[error("command timed out after {0:?}")]
    Timeout(Duration),

    /// Shell not found or unusable; {0} is the shell path or reason
    #[error("shell unavailable: {0}")]
    ShellUnavailable(String),

    /// Failed to spawn child process; the underlying io::Error is preserved as the error chain cause via #[source]
    #[error("failed to spawn command: {0}")]
    SpawnError(#[source] std::io::Error),

    /// Other unclassified error; {0} is the detail
    #[error("unknown execution error: {0}")]
    Unknown(String),
}

impl ExecutionEnv {
    /// Create an execution environment.
    ///
    /// # Parameters
    /// - `registry`: detached child-process registry; when `Some`, exec tracks/untracks the child-process pid.
    pub fn new(
        base_dir: PathBuf,
        shell_path: Option<PathBuf>,
        base_env: Option<HashMap<String, String>>,
        registry: Option<Arc<ProcessRegistry>>,
    ) -> Self {
        Self {
            base_dir,
            shell_path,
            base_env,
            registry,
        }
    }

    /// Execute a command in the environment's bound shell (platform-agnostic).
    ///
    /// Behavior (see pi-229 `NodeExecutionEnv.exec` and pi `createLocalBashOperations().exec`):
    /// - The shell is discovered via `resolve_shell(self.shell_path)`; relative `options.cwd` resolves against `self.cwd`.
    /// - **cwd precheck**: resolved working directory does not exist → `ExecutionError::Unknown` (message aligned with pi's
    ///   `fsAccess(cwd, F_OK)` check).
    /// - The child process is started in its own process group (unix `tokio::process::Command::process_group(0)`),
    ///   Let `kill_process_tree` terminate the entire process tree (aligns with pi's `detached` + `killProcessTree`).
    /// - **detached tracking**: after spawn succeeds call `track_detached_child_pid(pid)`; on end/cancel/timeout paths
    ///   call `untrack_detached_child_pid(pid)`, ensuring the parent-process close signal can clean up the whole tree of long-running commands.
    /// - Env vars = process environment + `self.base_env` + `options.env` (the latter takes highest priority).
    /// - **5303 wait semantics (pi#5303 regression)**: the wait path uses `process::wait_for_child_process`—
    ///   drains stdout/stderr while waiting for the process to exit; after exit, still-writing detached descendants
    ///   are read with a 100 ms idle grace re-armed on every data chunk (tail output not lost), while descendants
    ///   silently holding handles are released after the grace (no permanent hang). The original "`join!` reads to EOF then wait"
    ///   would hang forever when detached descendants hold the pipes, so it has been removed.
    /// - `options.on_stdout` / `options.on_stderr` are callbacks with UTF-8 text chunks (infallible contract);
    ///   callback granularity is **raw byte chunks (≤8KB)**, not line-by-line; multi-byte characters truncated across chunks are
    ///   concatenated via incremental decoding (`StreamDecoder`), producing no replacement characters.
    /// - `options.timeout` expired → terminate the process tree and return `ExecutionError::Timeout`.
    /// - `options.cancellation` triggered → terminate the process tree and return `ExecutionError::Aborted`.
    /// - Return accumulated stdout/stderr and exit code; exit code is 0 on signal termination (aligns with pi's `code ?? 0`).
    ///
    /// # Parameters
    /// - `command`: the shell command string to execute.
    /// - `options`: optional arguments for this call (**passed by value, single-use**—the
    ///   callback Boxes of `on_stdout`/`on_stderr` are moved into the byte callback closures to satisfy `'static`, so callers must not reuse them after exec returns).
    ///
    /// # Returns
    /// Returns `ExecOutput` on success; on failure classified by `ExecutionError` (timeout/aborted/shell_unavailable/spawn_error/unknown).
    pub async fn exec(
        &self,
        command: &str,
        mut options: ExecOptions,
    ) -> Result<ExecOutput, ExecutionError> {
        // 0) Precheck: already cancelled → return Aborted immediately (aligns with pi's abortSignal check at the start)
        if options
            .cancellation
            .as_ref()
            .is_some_and(|ct| ct.is_cancelled())
        {
            return Err(ExecutionError::Aborted);
        }

        // 1) Discover shell (ShellUnavailable propagated via ?)
        let shell_config = resolve_shell(self.shell_path.as_deref())?;

        // 2) Resolve cwd: relative paths are based on self.cwd (join is also correct for absolute paths)
        let cwd = match &options.cwd {
            Some(rel) => self.base_dir.join(rel),
            None => self.base_dir.clone(),
        };

        // 2.5) cwd existence precheck (aligned with pi createLocalBashOperations' fsAccess(cwd, F_OK))
        if !cwd.exists() {
            return Err(ExecutionError::Unknown(format!(
                "Working directory does not exist: {}\nCannot execute bash commands.",
                cwd.display()
            )));
        }

        // 3) Assemble child process
        let command_from_stdin = shell_config.transport == ShellTransport::Stdin;
        let mut cmd = tokio::process::Command::new(&shell_config.shell);
        cmd.args(&shell_config.args);
        if !command_from_stdin {
            cmd.arg(command); // argv transport: command passed as the last argument
        }
        cmd.current_dir(&cwd);
        // Env = inherit parent + self.base_env + options.env (the latter overrides the former)
        if let Some(base) = &self.base_env {
            cmd.envs(base);
        }
        if let Some(extra) = &options.env {
            cmd.envs(extra);
        }
        // unix: detached process group so kill_process_tree can terminate the whole process tree
        #[cfg(unix)]
        cmd.process_group(0);
        if command_from_stdin {
            cmd.stdin(std::process::Stdio::piped());
        } else {
            cmd.stdin(std::process::Stdio::null());
        }
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        // 4) start; pid is usable right after start (for kill/untrack, avoid touching child again in select branches)
        let mut child = cmd.spawn().map_err(ExecutionError::SpawnError)?;
        let pid = child.id();

        // 4.5) detached tracking: register pid to ensure the parent-process close signal cleans the whole tree (aligned with pi trackDetachedChildPid)
        if let (Some(pid), Some(reg)) = (pid, &self.registry) {
            reg.track(pid);
        }

        // 5) stdin transport: write the command in the background and close it (avoids mutual deadlock when child output fills the pipe)
        if command_from_stdin {
            if let Some(mut stdin) = child.stdin.take() {
                let command = command.to_string();
                tokio::spawn(async move {
                    let _ = stdin.write_all(command.as_bytes()).await;
                    let _ = stdin.shutdown().await;
                });
            }
        }

        // 6) streaming decode + accumulation + user callbacks. wait_for_child_process only gives raw byte callbacks,
        //    so create one StreamDecoder each for stdout/stderr (incremental UTF-8, multi-byte characters split across chunks concatenated correctly);
        //    within the callback, lock to decode, then unlock before invoking the user on_stdout/on_stderr (never hold a lock across await).
        //    The byte callback must be 'static (Box<dyn Fn + Send + Sync>), therefore:
        //    - The decoder is held by the closure as a clone of `Arc<Mutex<...>>` (exec keeps another handle to read the final result);
        //    - The user's on_stdout/on_stderr Boxes are moved from options into the closure (hence options is passed by value).
        let stdout_decoder = Arc::new(std::sync::Mutex::new(StreamDecoder::new()));
        let stderr_decoder = Arc::new(std::sync::Mutex::new(StreamDecoder::new()));
        let stdout_user = options.on_stdout.take(); // Option<Box<dyn Fn(&str) + Send + Sync>>
        let stderr_user = options.on_stderr.take();
        let stdout_cb: Box<ChunkCallback> = {
            let dec = stdout_decoder.clone();
            Box::new(move |chunk: &[u8]| {
                let text = dec.lock().unwrap_or_else(|e| e.into_inner()).feed(chunk);
                if let Some(cb) = &stdout_user {
                    cb(&text);
                }
            })
        };
        let stderr_cb: Box<ChunkCallback> = {
            let dec = stderr_decoder.clone();
            Box::new(move |chunk: &[u8]| {
                let text = dec.lock().unwrap_or_else(|e| e.into_inner()).feed(chunk);
                if let Some(cb) = &stderr_user {
                    cb(&text);
                }
            })
        };

        // 7) Wait: wait_for_child_process drains the pipes + waits for exit (including the 5303 tail grace period);
        //    select listens for cancellation/timeout (both branches only use the pid, not the child, to avoid conflicting with the wait branch's &mut child).
        let deadline = options.timeout.map(|d| tokio::time::Instant::now() + d);
        let timeout_duration = options.timeout;
        let outcome: Result<Result<Option<i32>, std::io::Error>, ExecutionError> = tokio::select! {
            wait_result = wait_for_child_process(
                &mut child,
                Some(stdout_cb.as_ref()),
                Some(stderr_cb.as_ref()),
            ) => Ok(wait_result),
            _ = async {
                match &options.cancellation {
                    Some(ct) => ct.cancelled().await,
                    None => std::future::pending::<()>().await,
                }
            } => {
                if let Some(pid) = pid {
                    kill_process_tree(pid);
                }
                Err(ExecutionError::Aborted)
            }
            _ = async {
                match deadline {
                    Some(dl) => tokio::time::sleep_until(dl).await,
                    None => std::future::pending::<()>().await,
                }
            } => {
                if let Some(pid) = pid {
                    kill_process_tree(pid);
                }
                Err(ExecutionError::Timeout(timeout_duration.unwrap_or_default()))
            }
        };

        // 8) Untrack on all paths: the detached registration only covers the process lifetime
        //    (normal exit / cancellation / timeout / wait failure all end the registration, preventing accidental kill via pid reuse).
        if let (Some(pid), Some(reg)) = (pid, &self.registry) {
            reg.untrack(pid);
        }

        // 9) Map results: wait failure → SpawnError; the cancel/timeout branches first reap the child process (to avoid zombies) then return the error.
        let exit_code: i32 = match outcome {
            Ok(Ok(code)) => code.unwrap_or(0), // wait_for_child_process returns None when terminated by a signal → 0
            Ok(Err(io_err)) => return Err(ExecutionError::SpawnError(io_err)),
            Err(err) => {
                let _ = child.wait().await; // reap to avoid zombies
                return Err(err);
            }
        };

        // 10) Final decoding: flush any remaining incomplete UTF-8 prefix (lossy) and extract the accumulated text.
        let stdout = {
            let mut dec = stdout_decoder.lock().unwrap_or_else(|e| e.into_inner());
            dec.finish();
            dec.take_text()
        };
        let stderr = {
            let mut dec = stderr_decoder.lock().unwrap_or_else(|e| e.into_inner());
            dec.finish();
            dec.take_text()
        };

        Ok(ExecOutput {
            stdout,
            stderr,
            exit_code,
        })
    }
}

/// Streaming UTF-8 decoder: incrementally decodes a raw byte stream crossing chunk boundaries into text, correctly joining multi-byte characters split across chunks
/// (matching the incremental semantics of pi `TextDecoder.decode(data, { stream: true })`, avoiding the replacement
/// characters that `from_utf8_lossy` produces at chunk boundaries when decoding chunks independently).
///
/// Each `feed` keeps any possibly incomplete character prefix at the chunk tail in `pending`, which is prepended to the next chunk before decoding;
/// after feeding all bytes, call `finish` to flush the remaining prefix lossily. In-memory state only, `Send`.
struct StreamDecoder {
    /// Incomplete UTF-8 character prefix left over from the previous chunk (0..=3 bytes)
    pending: Vec<u8>,
    /// Decoded complete text
    text: String,
}

impl StreamDecoder {
    fn new() -> Self {
        Self {
            pending: Vec::new(),
            text: String::new(),
        }
    }

    /// Feed a chunk of raw bytes: prepend `pending` first, isolate the trailing truncated prefix for the next chunk,
    /// decode the remaining bytes as UTF-8 lossy and append them to `text`, returning the newly decoded text for this chunk (for streaming callback forwarding).
    fn feed(&mut self, chunk: &[u8]) -> String {
        let mut buf = std::mem::take(&mut self.pending);
        buf.extend_from_slice(chunk);
        let partial = trailing_partial_len(&buf);
        if partial > 0 {
            self.pending = buf[buf.len() - partial..].to_vec();
            buf.truncate(buf.len() - partial);
        }
        let decoded = String::from_utf8_lossy(&buf);
        self.text.push_str(&decoded);
        decoded.into_owned()
    }

    /// Cleanup: append any leftover incomplete prefix to `text` lossily (mirroring the flush at the end of pi stream decoding).
    fn finish(&mut self) {
        let buf = std::mem::take(&mut self.pending);
        let decoded = String::from_utf8_lossy(&buf);
        self.text.push_str(&decoded);
    }

    /// Return all decoded text (clears internal state).
    fn take_text(&mut self) -> String {
        std::mem::take(&mut self.text)
    }
}

/// Return the byte count of the incomplete UTF-8 character prefix at the tail of `buf` (0..=3).
/// When a multibyte character is split at a block boundary, its prefix must be carried to the next block before decoding, avoiding U+FFFD.
///
/// Implementation: scan backward from the tail to the current character's lead byte (skipping continuation bytes 0x80..=0xBF);
/// if the byte count from that lead to the block tail is less than the character's expected length, it is a truncated prefix; return the number of bytes to carry.
fn trailing_partial_len(buf: &[u8]) -> usize {
    let mut tail = 0usize;
    let mut i = buf.len();
    while i > 0 {
        i -= 1;
        tail += 1;
        let b = buf[i];
        // Found a lead byte (anything outside 0x80..=0xBF)
        if b & 0xC0 != 0x80 {
            let expected = match b {
                0x00..=0x7F => 1, // ASCII
                0xC0..=0xDF => 2, // 2 bytes
                0xE0..=0xEF => 3, // 3 bytes
                0xF0..=0xF7 => 4, // 4 bytes
                _ => 0,           // Invalid lead (0xF8..=0xFF): not treated as a pending prefix
            };
            // expected is 0 (invalid byte) or tail >= expected (character complete) → nothing to carry
            return if expected > tail { tail } else { 0 };
        }
    }
    // Block is all continuations (invalid UTF-8): not treated as a pending prefix
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;

    /// Build an ExecOptions with all fields None/empty (default argument for most exec tests).
    fn empty_options() -> ExecOptions {
        ExecOptions {
            cwd: None,
            env: None,
            timeout: None,
            cancellation: None,
            on_stdout: None,
            on_stderr: None,
        }
    }

    // ── ExecutionEnv::new ─────────────────────────────────────────

    #[test]
    fn test_execution_env_new_stores_fields() {
        let env = ExecutionEnv::new(
            PathBuf::from("/tmp/zen-env-test"),
            Some(PathBuf::from("/bin/bash")),
            Some(HashMap::from([("FOO".to_string(), "bar".to_string())])),
            None,
        );
        assert_eq!(env.base_dir, PathBuf::from("/tmp/zen-env-test"));
        assert_eq!(env.shell_path, Some(PathBuf::from("/bin/bash")));
        assert_eq!(
            env.base_env.as_ref().unwrap().get("FOO"),
            Some(&"bar".to_string())
        );
    }

    // ── ExecutionError display ─────────────────────────────────────

    #[test]
    fn test_execution_error_display() {
        assert_eq!(ExecutionError::Aborted.to_string(), "command aborted");
        assert_eq!(
            ExecutionError::Timeout(Duration::from_secs(3)).to_string(),
            "command timed out after 3s"
        );
        assert_eq!(
            ExecutionError::ShellUnavailable("/bin/zzz".into()).to_string(),
            "shell unavailable: /bin/zzz"
        );
        assert_eq!(
            ExecutionError::Unknown("boom".into()).to_string(),
            "unknown execution error: boom"
        );
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "nope");
        assert!(ExecutionError::SpawnError(io_err)
            .to_string()
            .starts_with("failed to spawn command"));
    }

    // ── StreamDecoder / trailing_partial_len (incremental UTF-8 decoding, pure functions) ──

    #[test]
    fn test_trailing_partial_len_ascii_is_complete() {
        assert_eq!(trailing_partial_len(b""), 0);
        assert_eq!(trailing_partial_len(b"hello"), 0);
        assert_eq!(trailing_partial_len(b"a\nb\n"), 0);
    }

    #[test]
    fn test_trailing_partial_len_full_multibyte_is_complete() {
        // Complete multibyte character at the tail → nothing to carry
        assert_eq!(trailing_partial_len("é".as_bytes()), 0); // 2 bytes
        assert_eq!(trailing_partial_len("中".as_bytes()), 0); // 3 bytes
        assert_eq!(trailing_partial_len("🦀".as_bytes()), 0); // 4 bytes
                                                              // Complete character followed by ASCII at the tail → nothing to carry
        assert_eq!(trailing_partial_len("中!".as_bytes()), 0);
    }

    #[test]
    fn test_trailing_partial_len_truncated_multibyte() {
        // One CJK char = E4 B8 AD; only E4 B8 given (missing AD) → carry 2 bytes
        assert_eq!(trailing_partial_len(&[0xE4, 0xB8]), 2);
        // Only E4 given → carry 1 byte
        assert_eq!(trailing_partial_len(&[0xE4]), 1);
        // é = C3 A9; only C3 given → carry 1
        assert_eq!(trailing_partial_len(&[0xC3]), 1);
        // 🦀 = F0 9F A6 80; only F0 9F A6 given → carry 3
        assert_eq!(trailing_partial_len(&[0xF0, 0x9F, 0xA6]), 3);
        // ASCII at the front, truncated prefix at the tail → carry only the tail
        assert_eq!(trailing_partial_len(b"hi\xE4\xB8"), 2);
    }

    #[test]
    fn test_trailing_partial_len_ignores_invalid_sequences() {
        // 0xFF is an invalid lead → not treated as a pending prefix
        assert_eq!(trailing_partial_len(&[0xFF]), 0);
        // Block is all continuations (invalid) → 0
        assert_eq!(trailing_partial_len(&[0x80, 0x80]), 0);
    }

    #[test]
    fn test_stream_decoder_single_chunk() {
        let mut d = StreamDecoder::new();
        assert_eq!(d.feed(b"hello world"), "hello world");
        assert_eq!(d.take_text(), "hello world");
    }

    #[test]
    fn test_stream_decoder_empty_feed() {
        let mut d = StreamDecoder::new();
        assert_eq!(d.feed(b""), "");
        assert_eq!(d.take_text(), "");
    }

    #[test]
    fn test_stream_decoder_multibyte_spanning_chunks() {
        // Split a 3-byte CJK char (E4 B8 AD) into [E4 B8] + [AD]; cross-chunk joined decoding must not produce U+FFFD
        let mut d = StreamDecoder::new();
        assert_eq!(d.feed(&[0xE4, 0xB8]), ""); // Incomplete → this chunk emits no text
        assert_eq!(d.feed(&[0xAD]), "中");
        assert_eq!(d.take_text(), "中");
    }

    #[test]
    fn test_stream_decoder_three_chunk_split() {
        // 4-byte character split into three segments: F0 | 9F A6 | 80
        let mut d = StreamDecoder::new();
        assert_eq!(d.feed(&[0xF0]), "");
        assert_eq!(d.feed(&[0x9F, 0xA6]), "");
        assert_eq!(d.feed(&[0x80]), "🦀");
        assert_eq!(d.take_text(), "🦀");
    }

    #[test]
    fn test_stream_decoder_lossy_invalid_byte() {
        // 0xFF illegal byte → U+FFFD (consistent with per-line lossy semantics)
        let mut d = StreamDecoder::new();
        assert_eq!(d.feed(b"foo\xFF\n"), "foo\u{FFFD}\n");
    }

    #[test]
    fn test_stream_decoder_finish_flushes_pending() {
        // Leftover incomplete prefix is finalized lossily at finish
        let mut d = StreamDecoder::new();
        assert_eq!(d.feed(&[0xE4, 0xB8]), "");
        d.finish();
        assert_eq!(d.take_text(), "\u{FFFD}"); // Incomplete 2-byte sequence → single replacement character
    }

    // ── exec (real shell; local resolve_shell should resolve to /bin/bash) ──

    #[tokio::test]
    async fn test_exec_echo_stdout() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        let out = env
            .exec("echo hello", empty_options())
            .await
            .expect("exec ok");
        assert_eq!(out.stdout, "hello\n");
        assert_eq!(out.stderr, "");
        assert_eq!(out.exit_code, 0);
    }

    #[tokio::test]
    async fn test_exec_empty_command() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        let out = env.exec("", empty_options()).await.expect("exec ok");
        assert_eq!(out.stdout, "");
        assert_eq!(out.stderr, "");
        assert_eq!(out.exit_code, 0);
    }

    #[tokio::test]
    async fn test_exec_stdout_stderr_separation() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        let out = env
            .exec("echo to-out; echo to-err >&2", empty_options())
            .await
            .expect("exec ok");
        assert_eq!(out.stdout, "to-out\n");
        assert_eq!(out.stderr, "to-err\n");
        assert_eq!(out.exit_code, 0);
    }

    #[tokio::test]
    async fn test_exec_nonzero_exit_code() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        let out = env.exec("exit 3", empty_options()).await.expect("exec ok");
        assert_eq!(out.exit_code, 3);
    }

    #[tokio::test]
    async fn test_exec_on_stdout_callback() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        let received = Arc::new(Mutex::new(String::new()));
        let got = received.clone();
        let opts = ExecOptions {
            on_stdout: Some(Box::new(move |s: &str| got.lock().unwrap().push_str(s))),
            ..empty_options()
        };
        let out = env.exec("echo hello", opts).await.expect("exec ok");
        assert_eq!(out.stdout, "hello\n");
        assert_eq!(*received.lock().unwrap(), "hello\n");
    }

    #[tokio::test]
    async fn test_exec_base_env_and_override() {
        let env = ExecutionEnv::new(
            PathBuf::from("/"),
            None,
            Some(HashMap::from([("FOO".to_string(), "bar".to_string())])),
            None,
        );
        // only base_env takes effect
        let out = env
            .exec("printf '%s' \"$FOO\"", empty_options())
            .await
            .expect("exec ok");
        assert_eq!(out.stdout, "bar");
        // options.env overrides base_env
        let opts = ExecOptions {
            env: Some(HashMap::from([("FOO".to_string(), "baz".to_string())])),
            ..empty_options()
        };
        let out = env
            .exec("printf '%s' \"$FOO\"", opts)
            .await
            .expect("exec ok");
        assert_eq!(out.stdout, "baz");
    }

    #[tokio::test]
    async fn test_exec_default_cwd_is_base_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = tokio::fs::canonicalize(dir.path())
            .await
            .expect("canonicalize");
        let env = ExecutionEnv::new(base.clone(), None, None, None);
        let out = env.exec("pwd", empty_options()).await.expect("exec ok");
        assert_eq!(out.stdout.trim_end(), base.to_str().unwrap());
        assert_eq!(out.exit_code, 0);
    }

    #[tokio::test]
    async fn test_exec_cwd_relative_resolves_against_base_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = tokio::fs::canonicalize(dir.path())
            .await
            .expect("canonicalize");
        let sub = base.join("sub");
        tokio::fs::create_dir_all(&sub).await.expect("create sub");
        let env = ExecutionEnv::new(base, None, None, None);
        let opts = ExecOptions {
            cwd: Some(PathBuf::from("sub")),
            ..empty_options()
        };
        let out = env.exec("pwd", opts).await.expect("exec ok");
        assert_eq!(out.stdout.trim_end(), sub.to_str().unwrap());
    }

    #[tokio::test]
    async fn test_exec_absolute_cwd_ignores_base_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let abs = tokio::fs::canonicalize(dir.path())
            .await
            .expect("canonicalize");
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        let opts = ExecOptions {
            cwd: Some(abs.clone()),
            ..empty_options()
        };
        let out = env.exec("pwd", opts).await.expect("exec ok");
        assert_eq!(out.stdout.trim_end(), abs.to_str().unwrap());
    }

    #[tokio::test]
    async fn test_exec_precancelled_aborts() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        let ct = CancellationToken::new();
        ct.cancel();
        let opts = ExecOptions {
            cancellation: Some(ct),
            ..empty_options()
        };
        let res = env.exec("echo hello", opts).await;
        assert!(matches!(res, Err(ExecutionError::Aborted)));
    }

    #[tokio::test]
    async fn test_exec_cancelled_during_execution() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        let ct = CancellationToken::new();
        let ct2 = ct.clone();
        let canceller = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            ct2.cancel();
        });
        let opts = ExecOptions {
            cancellation: Some(ct),
            ..empty_options()
        };
        let res = env.exec("sleep 5", opts).await;
        canceller.await.expect("canceller task");
        assert!(matches!(res, Err(ExecutionError::Aborted)));
    }

    #[tokio::test]
    async fn test_exec_timeout() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        let opts = ExecOptions {
            timeout: Some(Duration::from_millis(200)),
            ..empty_options()
        };
        let res = env.exec("sleep 2", opts).await;
        assert!(matches!(res, Err(ExecutionError::Timeout(_))));
    }

    #[tokio::test]
    async fn test_exec_shell_unavailable() {
        // Assumption: resolve_shell returns ShellUnavailable for a nonexistent custom shell path (aligns with pi getShellConfig)
        let env = ExecutionEnv::new(
            PathBuf::from("/"),
            Some(PathBuf::from("/nonexistent/shell")),
            None,
            None,
        );
        let res = env.exec("echo hi", empty_options()).await;
        assert!(matches!(res, Err(ExecutionError::ShellUnavailable(_))));
    }

    // ── cwd precheck ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_exec_cwd_not_exists_returns_unknown() {
        // Resolved working directory does not exist → Unknown (aligned with pi createLocalBashOperations fsAccess F_OK)
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("does-not-exist");
        let env = ExecutionEnv::new(missing, None, None, None);
        let res = env.exec("echo hi", empty_options()).await;
        match res {
            Err(ExecutionError::Unknown(msg)) => {
                assert!(
                    msg.contains("Working directory does not exist"),
                    "msg: {msg}"
                );
                assert!(msg.contains("Cannot execute bash commands"), "msg: {msg}");
            }
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_exec_relative_cwd_not_exists_returns_unknown() {
        // Relative cwd resolving to a nonexistent path is likewise blocked by the precheck
        let dir = tempfile::tempdir().expect("tempdir");
        let env = ExecutionEnv::new(dir.path().to_path_buf(), None, None, None);
        let opts = ExecOptions {
            cwd: Some(PathBuf::from("nope")),
            ..empty_options()
        };
        let res = env.exec("echo hi", opts).await;
        assert!(matches!(res, Err(ExecutionError::Unknown(_))));
    }

    // ── 5303 regression (detached descendants holding pipes don't hang, tail output not lost) ────

    #[tokio::test]
    async fn test_exec_detached_descendant_holding_stdout_does_not_hang() {
        // Regression pi#5303: after the shell exits, background subshells inherit the stdout/stderr pipes and silently keep them open.
        // The old implementation (join! read then wait) would hang forever; wait_for_child_process should release after a 100ms idle grace period.
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        let start = std::time::Instant::now();
        let out = env
            .exec("printf 'DONE\\n'; ( sleep 2 ) &", empty_options())
            .await
            .expect("exec ok");
        let elapsed = start.elapsed();
        assert_eq!(out.exit_code, 0);
        assert!(out.stdout.contains("DONE"), "stdout: {:?}", out.stdout);
        assert!(
            elapsed < std::time::Duration::from_secs(3),
            "exec should not be hung by detached descendants: {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn test_exec_captures_output_after_exit_from_detached_writer() {
        // The other half of the pi#5303 regression: after the shell exits, a background descendant keeps writing large amounts of output to the inherited stdout,
        // all of which must be captured (no tail loss).
        // Use a "large volume continuous burst" (yes|head writing 50000 lines) instead of timed ticks: data flows continuously → the grace
        // timer is re-armed on every chunk and never fires due to idleness, making it immune to scheduling stretching under parallel load (previously 20ms×15 ticks
        // occasionally lost the tail when background sleeps were CPU-starved while the full env test suite ran in parallel).
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        let out = env
            .exec(
                "printf 'HEAD\\n'; ( yes 'TAIL-LINE' | head -n 50000 ) &",
                empty_options(),
            )
            .await
            .expect("exec ok");
        assert!(out.stdout.contains("HEAD"), "stdout: {:?}", out.stdout);
        assert_eq!(
            out.stdout.matches("TAIL-LINE").count(),
            50_000,
            "tail output must not be lost; stdout len: {}",
            out.stdout.len()
        );
        assert_eq!(out.exit_code, 0);
    }

    #[tokio::test]
    async fn test_exec_on_stdout_chunk_granularity_within_tick_sequence() {
        // Callbacks are delivered per chunk: multi-line output may be accumulated by the callback (a chunk can merge multiple lines, but no content is lost).
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        let received = Arc::new(Mutex::new(String::new()));
        let got = received.clone();
        let opts = ExecOptions {
            on_stdout: Some(Box::new(move |s: &str| got.lock().unwrap().push_str(s))),
            ..empty_options()
        };
        let out = env
            .exec("printf 'a\\nb\\nc\\n'", opts)
            .await
            .expect("exec ok");
        assert_eq!(out.stdout, "a\nb\nc\n");
        assert_eq!(*received.lock().unwrap(), "a\nb\nc\n");
    }

    // ── detached tracking (kill_all can clean up exec's entire child-process tree) ────

    #[tokio::test]
    async fn test_exec_tracks_detached_child_for_shutdown_kill() {
        // Verify that exec registers child processes in ProcessRegistry: simulate a parent-process shutdown-signal call
        // After registry.kill_all(), exec's long-running child processes should be SIGKILLed across the entire tree,
        // exec returns on its own when the token is not cancelled (signal termination → exit_code 0).
        //
        // If tracking is not wired up (exec never tracks), kill_all cannot kill it, and exec will keep running
        // `sleep 30`; assertion fails as soon as the timeout fires.
        let registry = crate::process::ProcessRegistry::new();

        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, Some(registry.clone()));
        let ct = CancellationToken::new();
        let opts = ExecOptions {
            cancellation: Some(ct.clone()),
            ..empty_options()
        };
        let mut task = tokio::spawn(async move { env.exec("sleep 30", opts).await });

        // wait for spawn + registration to complete
        tokio::time::sleep(Duration::from_millis(300)).await;

        // Simulate parent shutdown signal: batch-kill all registered detached process trees
        registry.kill_all();

        // If tracking is wired: process is SIGKILLed → wait_for_child_process returns (token not canceled) →
        // exec returns Ok(exit_code 0) on its own; if it does not return within 3s, kill_all did not kill it.
        let res = tokio::time::timeout(Duration::from_secs(3), &mut task).await;
        match res {
            Ok(Ok(Ok(out))) => assert_eq!(out.exit_code, 0, "signal termination should map to exit_code 0"),
            Ok(Ok(Err(e))) => panic!("exec should not error (token not cancelled): {e}"),
            Ok(Err(_)) => panic!("exec task panic"),
            Err(_elapsed) => {
                // kill_all ineffective → tracing not wired up: cancel finalization and fail
                ct.cancel();
                let _ = task.await;
                panic!("kill_all failed to terminate exec's child process (detached tracking not wired?)");
            }
        }
    }
}
