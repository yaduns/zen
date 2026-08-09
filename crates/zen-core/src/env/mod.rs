pub mod shell;
pub mod shell_output;

use crate::env::shell::{kill_process_tree, resolve_shell, ShellTransport};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
use tokio_util::sync::CancellationToken;

/// Binds one session to a platform-agnostic shell execution environment, holding cwd, an optional custom shell path, and the session's base environment variables;
/// exec runs commands in the platform self-discovered shell using this environment as context.
struct ExecutionEnv {
    /// Session working directory; exec's relative cwd option resolves against it
    base_dir: PathBuf,
    /// Custom shell path (e.g. user-configured); validated lazily at exec time
    shell_path: Option<PathBuf>,
    /// Session base environment variable overrides, layered on top of the process environment, with lower precedence than ExecOptions.env
    base_env: Option<HashMap<String, String>>,
}

/// All options for a single exec call. Lifetimes to one invocation
struct ExecOptions {
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
struct ExecOutput {
    /// Accumulated stdout (UTF-8)
    stdout: String,
    /// Accumulated stderr (UTF-8)
    stderr: String,
    /// Process exit code; 0 when terminated by signal (ExitStatus::code() is None)
    exit_code: i32,
}

#[derive(Debug, thiserror::Error)]
enum ExecutionError {
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
    pub fn new(
        base_dir: PathBuf,
        shell_path: Option<PathBuf>,
        base_env: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            base_dir,
            shell_path,
            base_env,
        }
    }

    /// Execute a command in the environment's bound shell (platform-agnostic).
    ///
    /// Behavior (see pi-229 `NodeExecutionEnv.exec`):
    /// - The shell is discovered via `resolve_shell(self.shell_path)`; relative `options.cwd` resolves against `self.cwd`.
    /// - The child process is started in its own process group (unix `tokio::process::Command::process_group(0)`),
    ///   Let `kill_process_tree` terminate the entire process tree (aligns with pi's `detached` + `killProcessTree`).
    /// - Env vars = process environment + `self.base_env` + `options.env` (the latter takes highest priority).
    /// - `options.on_stdout` / `options.on_stderr` call back with UTF-8 text chunks (infallible contract).
    /// - `options.timeout` expired → terminate the process tree and return `ExecutionError::Timeout`.
    /// - `options.cancellation` triggered → terminate the process tree and return `ExecutionError::Aborted`.
    /// - Return accumulated stdout/stderr and exit code; exit code is 0 on signal termination (aligns with pi's `code ?? 0`).
    ///
    /// # Parameters
    /// - `command`: the shell command string to execute.
    /// - `options`: options for this invocation.
    ///
    /// # Returns
    /// Returns `ExecOutput` on success; on failure classified by `ExecutionError` (timeout/aborted/shell_unavailable/spawn_error/unknown).
    pub async fn exec(
        &self,
        command: &str,
        options: &ExecOptions,
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

        // 4) Start; pid is available after starting (for kill, avoiding touching child again in the select branch)
        let mut child = cmd.spawn().map_err(ExecutionError::SpawnError)?;
        let pid = child.id();

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

        // 6) Take output pipes and build read futures
        let stdout_reader = tokio::io::BufReader::new(
            child
                .stdout
                .take()
                .ok_or_else(|| ExecutionError::Unknown("stdout pipe unavailable".into()))?,
        );
        let stderr_reader = tokio::io::BufReader::new(
            child
                .stderr
                .take()
                .ok_or_else(|| ExecutionError::Unknown("stderr pipe unavailable".into()))?,
        );
        let mut stdout = String::new();
        let mut stderr = String::new();
        let read_stdout = read_stream(stdout_reader, &mut stdout, options.on_stdout.as_deref());
        let read_stderr = read_stream(stderr_reader, &mut stderr, options.on_stderr.as_deref());

        // 7) Concurrently: read streams to completion + wait for exit; select listens for cancel/timeout (both branches use only pid and do not touch child,
        //    avoiding conflicts with &mut child in the wait branch)
        let deadline = options.timeout.map(|d| tokio::time::Instant::now() + d);
        let timeout_duration = options.timeout;
        let outcome = tokio::select! {
            status = async {
                let _ = tokio::join!(read_stdout, read_stderr);
                child.wait().await.map_err(ExecutionError::SpawnError)
            } => status,
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

        // 8) Cancel/timeout branch first waits for child process exit (reap, avoid zombies) before returning an error
        let status = match outcome {
            Ok(status) => status,
            Err(err) => {
                let _ = child.wait().await;
                return Err(err);
            }
        };

        Ok(ExecOutput {
            stdout,
            stderr,
            exit_code: status.code().unwrap_or(0), // When terminated by signal, code() is None → 0
        })
    }
}

/// Read an output stream line by line: accumulate into `out` and trigger the callback with UTF-8 text chunks (if provided).
/// Non-UTF-8 bytes are handled as replacement characters (matching the lossy semantics of pi `setEncoding("utf8")`);
/// Ends on EOF or stream error; best effort, no error is thrown.
async fn read_stream<R: tokio::io::AsyncBufRead + Unpin>(
    mut reader: R,
    out: &mut String,
    callback: Option<&(dyn Fn(&str) + Send + Sync)>,
) {
    let mut buf: Vec<u8> = Vec::new();
    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf).await {
            Ok(0) => break, // EOF
            Ok(_) => {
                let text = String::from_utf8_lossy(&buf);
                out.push_str(&text);
                if let Some(cb) = callback {
                    cb(&text);
                }
            }
            Err(_) => break, // Read error: abandon this stream
        }
    }
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

    // ── read_stream (pure function, in-memory stream) ──────────────────────────────

    #[tokio::test]
    async fn test_read_stream_accumulates_lines() {
        let data: &[u8] = b"a\nb\nc\n";
        let mut out = String::new();
        read_stream(tokio::io::BufReader::new(data), &mut out, None).await;
        assert_eq!(out, "a\nb\nc\n");
    }

    #[tokio::test]
    async fn test_read_stream_accumulates_without_trailing_newline() {
        let data: &[u8] = b"a\nb";
        let mut out = String::new();
        read_stream(tokio::io::BufReader::new(data), &mut out, None).await;
        assert_eq!(out, "a\nb");
    }

    #[tokio::test]
    async fn test_read_stream_empty_input() {
        let data: &[u8] = b"";
        let mut out = String::new();
        read_stream(tokio::io::BufReader::new(data), &mut out, None).await;
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn test_read_stream_callback_receives_chunks() {
        let data: &[u8] = b"x\ny\n";
        let received = Arc::new(Mutex::new(String::new()));
        let got = received.clone();
        let cb = move |s: &str| got.lock().unwrap().push_str(s);
        let mut out = String::new();
        read_stream(tokio::io::BufReader::new(data), &mut out, Some(&cb)).await;
        assert_eq!(out, "x\ny\n");
        assert_eq!(*received.lock().unwrap(), "x\ny\n");
    }

    #[tokio::test]
    async fn test_read_stream_lossy_utf8() {
        // 0xff is an invalid UTF-8 leading byte → from_utf8_lossy replaces it with U+FFFD
        let data: &[u8] = &[0x66, 0x6f, 0x6f, 0xff, 0x0a]; // b"foo" + 0xff + b"\n"
        let mut out = String::new();
        read_stream(tokio::io::BufReader::new(data), &mut out, None).await;
        assert_eq!(out, "foo\u{FFFD}\n");
    }

    // ── exec (real shell; local resolve_shell should resolve to /bin/bash) ──

    #[tokio::test]
    async fn test_exec_echo_stdout() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None);
        let out = env
            .exec("echo hello", &empty_options())
            .await
            .expect("exec ok");
        assert_eq!(out.stdout, "hello\n");
        assert_eq!(out.stderr, "");
        assert_eq!(out.exit_code, 0);
    }

    #[tokio::test]
    async fn test_exec_empty_command() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None);
        let out = env.exec("", &empty_options()).await.expect("exec ok");
        assert_eq!(out.stdout, "");
        assert_eq!(out.stderr, "");
        assert_eq!(out.exit_code, 0);
    }

    #[tokio::test]
    async fn test_exec_stdout_stderr_separation() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None);
        let out = env
            .exec("echo to-out; echo to-err >&2", &empty_options())
            .await
            .expect("exec ok");
        assert_eq!(out.stdout, "to-out\n");
        assert_eq!(out.stderr, "to-err\n");
        assert_eq!(out.exit_code, 0);
    }

    #[tokio::test]
    async fn test_exec_nonzero_exit_code() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None);
        let out = env.exec("exit 3", &empty_options()).await.expect("exec ok");
        assert_eq!(out.exit_code, 3);
    }

    #[tokio::test]
    async fn test_exec_on_stdout_callback() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None);
        let received = Arc::new(Mutex::new(String::new()));
        let got = received.clone();
        let opts = ExecOptions {
            on_stdout: Some(Box::new(move |s: &str| got.lock().unwrap().push_str(s))),
            ..empty_options()
        };
        let out = env.exec("echo hello", &opts).await.expect("exec ok");
        assert_eq!(out.stdout, "hello\n");
        assert_eq!(*received.lock().unwrap(), "hello\n");
    }

    #[tokio::test]
    async fn test_exec_base_env_and_override() {
        let env = ExecutionEnv::new(
            PathBuf::from("/"),
            None,
            Some(HashMap::from([("FOO".to_string(), "bar".to_string())])),
        );
        // only base_env takes effect
        let out = env
            .exec("printf '%s' \"$FOO\"", &empty_options())
            .await
            .expect("exec ok");
        assert_eq!(out.stdout, "bar");
        // options.env overrides base_env
        let opts = ExecOptions {
            env: Some(HashMap::from([("FOO".to_string(), "baz".to_string())])),
            ..empty_options()
        };
        let out = env
            .exec("printf '%s' \"$FOO\"", &opts)
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
        let env = ExecutionEnv::new(base.clone(), None, None);
        let out = env.exec("pwd", &empty_options()).await.expect("exec ok");
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
        let env = ExecutionEnv::new(base, None, None);
        let opts = ExecOptions {
            cwd: Some(PathBuf::from("sub")),
            ..empty_options()
        };
        let out = env.exec("pwd", &opts).await.expect("exec ok");
        assert_eq!(out.stdout.trim_end(), sub.to_str().unwrap());
    }

    #[tokio::test]
    async fn test_exec_absolute_cwd_ignores_base_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let abs = tokio::fs::canonicalize(dir.path())
            .await
            .expect("canonicalize");
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None);
        let opts = ExecOptions {
            cwd: Some(abs.clone()),
            ..empty_options()
        };
        let out = env.exec("pwd", &opts).await.expect("exec ok");
        assert_eq!(out.stdout.trim_end(), abs.to_str().unwrap());
    }

    #[tokio::test]
    async fn test_exec_precancelled_aborts() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None);
        let ct = CancellationToken::new();
        ct.cancel();
        let opts = ExecOptions {
            cancellation: Some(ct),
            ..empty_options()
        };
        let res = env.exec("echo hello", &opts).await;
        assert!(matches!(res, Err(ExecutionError::Aborted)));
    }

    #[tokio::test]
    async fn test_exec_cancelled_during_execution() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None);
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
        let res = env.exec("sleep 5", &opts).await;
        canceller.await.expect("canceller task");
        assert!(matches!(res, Err(ExecutionError::Aborted)));
    }

    #[tokio::test]
    async fn test_exec_timeout() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None);
        let opts = ExecOptions {
            timeout: Some(Duration::from_millis(200)),
            ..empty_options()
        };
        let res = env.exec("sleep 2", &opts).await;
        assert!(matches!(res, Err(ExecutionError::Timeout(_))));
    }

    #[tokio::test]
    async fn test_exec_shell_unavailable() {
        // Assumption: resolve_shell returns ShellUnavailable for a nonexistent custom shell path (aligns with pi getShellConfig)
        let env = ExecutionEnv::new(
            PathBuf::from("/"),
            Some(PathBuf::from("/nonexistent/shell")),
            None,
        );
        let res = env.exec("echo hi", &empty_options()).await;
        assert!(matches!(res, Err(ExecutionError::ShellUnavailable(_))));
    }
}
