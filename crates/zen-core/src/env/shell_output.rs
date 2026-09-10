// All optional arguments for a single execute_shell_with_capture call

use crate::env::{ExecOptions, ExecutionEnv, ExecutionError};
use crate::truncate::{truncate_tail, TruncationOptions, DEFAULT_MAX_BYTES};
use std::collections::HashMap;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

/// Streaming snapshot payload at some point during execution (the `ShellCaptureOptions::on_snapshot` callback argument):
/// the bash tool uses it to assemble throttled partial `on_update`, so that during bash tool execution in an agent session
/// the tail output and truncation state are visible in real time (matching the snapshot contents of `emitOutputUpdate` in pi bash.ts).
pub struct StreamSnapshot {
    /// Display text after tail-buffer truncation (the result of `truncate_tail` on the tail buffer; the full tail when not over the limit).
    pub content: String,
    /// Truncation metadata (derived from the tail buffer; same semantics as [`ShellCaptureResult::truncation`], not cumulative over the full output).
    pub truncation: crate::truncate::TruncationResult,
    /// Path of the full output spilled so far; `None` if not yet overflowed.
    pub full_output_path: Option<PathBuf>,
}

/// Throttled snapshot callback type (`ShellCaptureOptions::on_snapshot`; the bash tool uses it to assemble partial `on_update`).
/// Named alias to avoid the `type_complexity` warning (following the `zen_ai::tool::ToolUpdateCallback` precedent).
pub type StreamSnapshotCallback = Arc<dyn Fn(&StreamSnapshot) + Send + Sync>;

pub struct ShellCaptureOptions {
    // Working directory of the command, passed through as ExecOptions.cwd
    pub cwd: Option<PathBuf>,
    // Additional environment variable overrides, passed through as ExecOptions.env
    pub env: Option<HashMap<String, String>>,
    // Timeout, passed through as ExecOptions.timeout (
    pub timeout: Option<Duration>,
    // Cancellation token, passed through as ExecOptions.cancellation
    pub cancellation: Option<CancellationToken>,
    // Streaming callback for each chunk of sanitized text (\r removed); infallible contract (panic unwinds directly, aligning with ExecOptions.on_stdout)
    pub on_chunk: Arc<dyn Fn(&str) + Send + Sync>,
    /// Whether to run `sanitize_binary_output` + strip `\r` on each chunk. `true` = sanitize (current behavior of the `!/!!` path,
    /// matching pi bash-executor.ts); `false` keeps the raw text (the bash tool passes false, matching the pi bash tool's no-sanitize behavior).
    pub sanitize: bool,
    /// Throttled snapshot callback (`None` = disabled); internally emits [`StreamSnapshot`] from the tail buffer per [`Self::snapshot_interval`].
    /// The callback is a synchronous closure; snapshot collection happens outside the internal lock (take tail text, release lock, then call back), so user code is never invoked while holding the lock.
    pub on_snapshot: Option<StreamSnapshotCallback>,
    /// Trigger interval for [`Self::on_snapshot`] (the bash tool passes `Some(Duration::from_millis(100))`);
    /// not triggered when this is `None` or `on_snapshot` is `None`.
    pub snapshot_interval: Option<Duration>,
}

// Complete result of a capture call returned to the caller
pub struct ShellCaptureResult {
    // Content after tail truncation (truncateTail semantics: keep last 2000 lines / 50KB)
    pub output: String,
    // Process exit code; None when cancelled, Some(code) on normal exit
    pub exit_code: Option<i32>,
    // Whether it was cancelled (exec returned Err(Aborted), or a race where the token was cancelled after execution)
    pub cancelled: bool,
    // Whether tail truncation occurred
    pub truncated: bool,
    // Full-output temp file path (prefix `bash-`, suffix `.log`); exists only on overflow/truncation
    pub full_output_path: Option<PathBuf>,
    // final truncation metadata (already computed internally; exposed for the bash tool to assemble details and hint text)
    pub truncation: crate::truncate::TruncationResult,
}

// ---- Serial write coordination (replaces pi's writeChain promise chain) ----
enum WriteRequest {
    /// Create a temp file (`bash-*.log`) and write initial content
    Create { initial: String },
    /// Append a chunk of text
    Append { text: String },
}

struct SharedCaptureState {
    total_bytes: usize,
    create_sent: bool,
    tail: VecDeque<String>,
    tail_bytes: usize,
    /// Timestamp of the last throttled snapshot; `None` = no snapshot emitted yet (the first snapshot fires immediately).
    last_snapshot_at: Option<Instant>,
}

/// Create temp file + write initial content; on success return the persisted path
async fn create_and_write(initial: &str) -> Result<PathBuf, ExecutionError> {
    use std::io::Write as _;
    let mut tmp = tempfile::Builder::new()
        .prefix("bash-")
        .suffix(".log")
        .tempfile()
        .map_err(|e| ExecutionError::Unknown(format!("create temp file: {e}")))?;
    tmp.write_all(initial.as_bytes())
        .map_err(|e| ExecutionError::Unknown(format!("write temp file: {e}")))?;
    let path = tmp
        .into_temp_path()
        .keep()
        .map_err(|e| ExecutionError::Unknown(format!("persist temp file: {e}")))?;
    Ok(path)
}

/// Append a chunk of text to an existing temp file
async fn append_text(path: &Path, text: &str) -> Result<(), ExecutionError> {
    use tokio::io::AsyncWriteExt;
    let mut f = tokio::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .await
        .map_err(|e| ExecutionError::Unknown(format!("open temp file: {e}")))?;
    f.write_all(text.as_bytes())
        .await
        .map_err(|e| ExecutionError::Unknown(format!("append temp file: {e}")))?;
    // tokio::fs::File has an internal write buffer (`max_buf_size` defaults to 8KB), so small writes stay in the buffer,
    // and drop does not flush to disk (tokio docs explicitly require an explicit flush before drop). Without a flush,
    // data never reaches the kernel page cache, and other fds (e.g. read_to_string) read stale content —
    // under concurrency this manifests as lost appends.
    f.flush()
        .await
        .map_err(|e| ExecutionError::Unknown(format!("flush temp file: {e}")))?;
    Ok(())
}

/// Sanitize binary output text.
///
/// Filter character by character:
/// - Preserve `\t` (0x09), `\n` (0x0a), `\r` (0x0d);
/// - Drop remaining C0 control characters (code points ≤ 0x1f);
/// - Drop U+FFF9..U+FFFB (line-start / line-mid / line-end interlinear annotation chars);
/// - All other characters are kept as-is.
///
/// # Edge Cases
/// - **Do NOT strip `\r`** (`\r` is stripped by each calling pipeline via `.replace(/\r/g, "")`,
///   this function keeps the single responsibility of "control character filtering", so it can be reused by call sites such as render-utils).
/// - Input may be any lossy-decoded text of invalid UTF-8; an empty string returns an empty string.
pub fn sanitize_binary_output(s: &str) -> String {
    s.chars()
        .filter(|c| {
            let code = *c as u32;
            // Whitelist: \t \n \r are kept as-is
            matches!(code, 0x09 | 0x0a | 0x0d)
                   // Others: remove C0 (≤0x1f) and U+FFF9..U+FFFB
                   || (code > 0x1f && !(0xfff9..=0xfffb).contains(&code))
        })
        .collect()
}

/// Execute a command on `ExecutionEnv` and capture output in a streaming manner
///
/// Behavior:
/// - Calls `env.exec` under the hood; both `on_stdout` and `on_stderr` are bound to the same internal handler
///   (preserving the interleaving of stdout/stderr arrival order, consistent with pi).
/// - Per-chunk processing: `total_bytes += chunk.len()` (UTF-8 bytes); sanitize and strip `\r`
///   (`sanitize_binary_output(chunk).replace('\r', "")`); push into the bounded tail buffer
///   `VecDeque<String>` (rolling cap `2 * DEFAULT_MAX_BYTES` = 100KB, popped from the front when exceeded);
///   each sanitized chunk is then passed back via `options.on_chunk`.
/// - **Full output overflow**: when `total_bytes` first exceeds `DEFAULT_MAX_BYTES` (50KB),
///   write "all text so far" to a temp file (`tempfile::Builder`, prefix `bash-`, suffix
///   `.log`); every subsequent chunk is appended; `full_output_path` records that path.
/// - **Serial async write coordination**: sync closures can't await, so use a `tokio::sync::mpsc` channel + a dedicated
///   writer task to serialize the "create temp file / append" operations (replacing pi's `writeChain` promise
///   chain); the outer scope keeps one `tx` for the final write after exec returns, then drops it + awaits
///   the `JoinHandle` to surface write errors.
/// - **Final truncation**: applies `truncate_tail` to the tail buffer (keeps last 2000 lines / 50KB by default,
///   never returns empty content; if a single line exceeds the limit, returns the tail portion of that line). If truncation occurs and no
///   `full_output_path` exists yet, writes the full tail buffer to a temporary file.
///
/// # Return / error semantics
/// - Success (or cancellation) returns `Ok(ShellCaptureResult)`:
///   - exec returns `Err(Aborted)`, or exec succeeds but the token was already cancelled (race) →
///     `cancelled: true`、`exit_code: None`；
///   - otherwise `cancelled: false`, `exit_code: Some(exec_output.exit_code)`.
/// - Other exec errors (`Timeout`/`ShellUnavailable`/`SpawnError`/`Unknown`) propagate unchanged as `Err`.
/// - Temp file creation/append failure → `Err(ExecutionError::Unknown(cause))`.
///
/// # Parameters
/// - `env`: session-level execution environment (defined in T-AG-06-1; this task does not add an impl for it).
/// - `command`: the shell command string to execute.
/// - `options`: options for this capture.
pub async fn execute_shell_with_capture(
    env: &ExecutionEnv,
    command: &str,
    options: &ShellCaptureOptions,
) -> Result<ShellCaptureResult, ExecutionError> {
    const MAX_TAIL_BYTES: usize = 2 * DEFAULT_MAX_BYTES; // 100KB tail buffer rolling cap

    // 1) Shared state: tail buffer / total bytes / file-created flag / full output path
    let state = Arc::new(Mutex::new(SharedCaptureState {
        total_bytes: 0,
        create_sent: false,
        tail: VecDeque::new(),
        tail_bytes: 0,
        last_snapshot_at: None,
    }));
    let full_output_path: Arc<Mutex<Option<PathBuf>>> = Arc::new(Mutex::new(None));

    // 2) mpsc channel + writer task: serializes file creation/appends
    let (tx, mut rx) = tokio::sync::mpsc::channel::<WriteRequest>(1024);
    let path_sink = full_output_path.clone();
    let write_task = tokio::spawn(async move {
        let mut current_path: Option<PathBuf> = None;
        // recv() only returns None when the channel is closed and drained
        while let Some(req) = rx.recv().await {
            let result = match req {
                WriteRequest::Create { initial } => match create_and_write(&initial).await {
                    Ok(p) => {
                        if let Ok(mut guard) = path_sink.lock() {
                            *guard = Some(p.clone());
                        }
                        current_path = Some(p);
                        Ok(())
                    }
                    Err(e) => Err(e),
                },
                WriteRequest::Append { text } => match &current_path {
                    Some(p) => append_text(p, &text).await,
                    None => Ok(()), // Theoretically unreachable: drop it
                },
            };
            if let Err(e) = result {
                return Err(e); // Terminate on write failure; the outer await on the JoinHandle reaps it
            }
        }
        Ok(())
    });

    // 3) Per-chunk internal handler (synchronous closure; can only try_send to queue write operations)
    //    throttled snapshot: inside the lock only check whether due and update the timestamp; snapshot computation and callback happen outside the lock
    //    (take tail text, release lock, then call back, avoiding calling user code while holding the lock — the recommended strategy for open question 1).
    let sanitize = options.sanitize;
    let snapshot_interval = options.snapshot_interval;
    let on_snapshot = options.on_snapshot.clone();
    let on_chunk = options.on_chunk.clone();
    let state_handler = state.clone();
    let tx_handler = tx.clone();
    let path_snapshot = full_output_path.clone();
    let handler: Arc<dyn Fn(&str) + Send + Sync> = Arc::new(move |chunk: &str| {
        let Ok(mut st) = state_handler.lock() else {
            return; // Lock poisoning (should not happen): best effort
        };
        st.total_bytes += chunk.len();
        let text = if sanitize {
            sanitize_binary_output(chunk).replace('\r', "")
        } else {
            chunk.to_string()
        };

        // Full output overflow: create the file on the first overflow (initial content = all text so far), then append each chunk
        if st.total_bytes > DEFAULT_MAX_BYTES {
            if !st.create_sent {
                st.create_sent = true;
                let initial: String = st.tail.iter().cloned().collect::<Vec<_>>().concat() + &text;
                let _ = tx_handler.try_send(WriteRequest::Create { initial });
            } else {
                let _ = tx_handler.try_send(WriteRequest::Append { text: text.clone() });
            }
        }

        // Bounded tail buffer: rolling cap of 100KB, pop from the head when exceeded
        st.tail.push_back(text.clone());
        st.tail_bytes += text.len();
        while st.tail_bytes > MAX_TAIL_BYTES && st.tail.len() > 1 {
            if let Some(removed) = st.tail.pop_front() {
                st.tail_bytes -= removed.len();
            }
        }

        // throttled snapshot check: if due, mark and update the timestamp (computation outside the lock)
        let snapshot_due = match (on_snapshot.as_ref(), snapshot_interval) {
            (Some(_), Some(interval)) => {
                let now = Instant::now();
                let due = st
                    .last_snapshot_at
                    .is_none_or(|t| now.duration_since(t) >= interval);
                if due {
                    st.last_snapshot_at = Some(now);
                }
                due
            }
            _ => false,
        };
        drop(st); // Callback after releasing the lock

        // outside the lock: get tail text + current full output path → build snapshot → callback (no lock held, callback may do anything)
        if snapshot_due {
            let tail_text: String = state_handler
                .lock()
                .map(|st| st.tail.iter().cloned().collect::<Vec<_>>().concat())
                .unwrap_or_default();
            let path = path_snapshot.lock().ok().and_then(|g| g.clone());
            let truncation = truncate_tail(&tail_text, TruncationOptions::default());
            if let Some(cb) = &on_snapshot {
                cb(&StreamSnapshot {
                    content: truncation.content.clone(),
                    truncation,
                    full_output_path: path,
                });
            }
        }
        on_chunk(&text); // Direct call on Arc<dyn Fn>
    });

    // 4) Wrap into two Boxes (stdout/stderr share the same handler, preserving interleaved arrival order)
    let on_stdout: Box<dyn Fn(&str) + Send + Sync> = {
        let h = handler.clone();
        Box::new(move |c: &str| h(c))
    };
    let on_stderr: Box<dyn Fn(&str) + Send + Sync> = {
        let h = handler.clone();
        Box::new(move |c: &str| h(c))
    };

    // 5) Assemble ExecOptions and execute
    let exec_options = ExecOptions {
        cwd: options.cwd.clone(),
        env: options.env.clone(),
        timeout: options.timeout,
        cancellation: options.cancellation.clone(),
        on_stdout: Some(on_stdout),
        on_stderr: Some(on_stderr),
    };
    let exec_result = env.exec(command, exec_options).await;

    // 6) Aggregate tail buffer + final truncation
    let tail_output: String = match state.lock() {
        Ok(st) => st.tail.iter().cloned().collect::<Vec<_>>().concat(),
        Err(_) => String::new(),
    };
    let truncation = truncate_tail(&tail_output, TruncationOptions::default());

    // Truncation occurred and no complete output file yet → write the full tail buffer into the temp file
    let need_final_write = match full_output_path.lock() {
        Ok(g) => truncation.truncated && g.is_none(),
        Err(_) => false,
    };
    if need_final_write {
        let _ = tx
            .send(WriteRequest::Create {
                initial: tail_output,
            })
            .await;
    }

    // 7) Close all senders, wait for the writer to drain and collect errors.
    //    Key point: the channel has three senders — the main tx, and tx_handler captured by the handler closure.
    //    handler holds tx_handler in a local Arc; on_stdout/on_stderr each hold a clone of the handler.
    //    exec consumes ExecOptions by value; on return the Box for on_stdout/on_stderr (and its handler
    //    clone) has already been released, so we only need to drop the local handler and the main tx; the channel then closes and writer recv() returns None.
    drop(handler); // Drop the local handler Arc (holds tx_handler)
    drop(tx); // Drop the main tx → channel closes → writer recv() returns None
    let write_result = match write_task.await {
        Ok(r) => r,
        Err(_) => Err(ExecutionError::Unknown("writer task panicked".into())),
    };
    if let Err(e) = write_result {
        return Err(e); // Temp file creation/append failed
    }

    let full_output_path = match full_output_path.lock() {
        Ok(g) => g.clone(),
        Err(_) => None,
    };

    // 8) Error semantics: Aborted / cancellation race → cancelled; other errors propagate as-is
    let (cancelled, exit_code) = match exec_result {
        Err(ExecutionError::Aborted) => (true, None),
        Err(e) => return Err(e), // Timeout/ShellUnavailable/SpawnError/Unknown propagate as-is
        Ok(out) => {
            let cancelled = options
                .cancellation
                .as_ref()
                .is_some_and(|ct| ct.is_cancelled());
            (
                cancelled,
                if cancelled { None } else { Some(out.exit_code) },
            )
        }
    };

    Ok(ShellCaptureResult {
        output: truncation.content.clone(),
        exit_code,
        cancelled,
        truncated: truncation.truncated,
        full_output_path,
        truncation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::{ExecutionEnv, ExecutionError};
    use crate::truncate::DEFAULT_MAX_LINES;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use tokio_util::sync::CancellationToken;

    /// Construct a default ShellCaptureOptions (no cwd/env/timeout/cancellation, on_chunk is a no-op).
    fn default_options() -> ShellCaptureOptions {
        ShellCaptureOptions {
            cwd: None,
            env: None,
            timeout: None,
            cancellation: None,
            on_chunk: Arc::new(|_s: &str| {}),
            sanitize: true, // sanitize by default (current behavior of the `!/!!` path)
            on_snapshot: None,
            snapshot_interval: None,
        }
    }

    // ── sanitize_binary_output ───────────────────────────────────

    #[test]
    fn test_sanitize_keeps_tab_newline_cr() {
        assert_eq!(sanitize_binary_output("a\tb\nc\rd"), "a\tb\nc\rd");
    }

    #[test]
    fn test_sanitize_strips_c0_control_chars() {
        // Remove everything in \x00..\x1f except \t\n\r
        assert_eq!(sanitize_binary_output("\x00\x01\x07\x0b\x0c\x1f"), "");
        assert_eq!(sanitize_binary_output("a\x00b\x1fc"), "abc");
    }

    #[test]
    fn test_sanitize_strips_interlinear_annotation_chars() {
        // Strip U+FFF9..U+FFFB
        assert_eq!(
            sanitize_binary_output("a\u{fff9}b\u{fffa}c\u{fffb}d"),
            "abcd"
        );
        // Keep U+FFF8 / U+FFFC outside the boundary
        assert_eq!(
            sanitize_binary_output("x\u{fff8}y\u{fffc}z"),
            "x\u{fff8}y\u{fffc}z"
        );
    }

    #[test]
    fn test_sanitize_keeps_visible_text() {
        assert_eq!(
            sanitize_binary_output("hello, world! 中文 🙂"),
            "hello, world! 中文 🙂"
        );
        // Keep both 0x20 (space) and 0x7f (DEL)
        assert_eq!(sanitize_binary_output("\x20\x7f"), "\x20\x7f");
    }

    #[test]
    fn test_sanitize_empty_input() {
        assert_eq!(sanitize_binary_output(""), "");
    }

    #[test]
    fn test_sanitize_mixed_content() {
        // Keep \t\n\r, strip other C0 and annotation characters
        assert_eq!(
            sanitize_binary_output("a\x01b\t\u{fffa}c\nd\r"),
            "ab\tc\nd\r"
        );
    }

    // ── create_and_write / append_text (tempfile isolates resources, deterministically verifiable) ──

    /// Regression test: multiple OS threads (each with its own tokio runtime, sharing the global blocking pool) in parallel
    /// doing create_and_write + append_text + read, covering tokio::fs::File write buffering
    /// concurrency defects where append data was lost without an explicit flush (previously reproducible at ~20% probability).
    #[test]
    fn stress_parallel_create_append() {
        use std::thread;
        let mut bad = 0usize;
        let mut collided = 0usize;
        for round in 0..50 {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    thread::spawn(|| {
                        let rt = tokio::runtime::Runtime::new().expect("rt");
                        rt.block_on(async {
                            let p = create_and_write("hello").await.expect("create");
                            append_text(&p, " world").await.expect("append");
                            let c =
                                std::fs::read_to_string(&p).unwrap_or_else(|_| "<unread>".into());
                            let _ = std::fs::remove_file(&p);
                            (format!("{}", p.display()), c)
                        })
                    })
                })
                .collect();
            let mut round_paths = std::collections::HashSet::new();
            for h in handles {
                let (p, c) = h.join().unwrap();
                if !round_paths.insert(p.clone()) {
                    collided += 1;
                }
                if c != "hello world" {
                    bad += 1;
                    if bad <= 3 {
                        println!("  MISMATCH round={round} path={p} content={c:?}");
                    }
                }
            }
        }
        println!("stress result: bad={bad} collided={collided}");
        assert_eq!(bad, 0, "append lost under parallelism");
    }

    #[tokio::test]
    async fn test_create_and_write_creates_prefixed_log_file() {
        let path = create_and_write("hello").await.expect("create ok");
        let name = path.file_name().and_then(|s| s.to_str()).unwrap();
        assert!(name.starts_with("bash-"), "prefix: {name}");
        assert!(name.ends_with(".log"), "suffix: {name}");
        let content = std::fs::read_to_string(&path).expect("read ok");
        assert_eq!(content, "hello");
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn test_append_text_appends_to_existing_file() {
        let path = create_and_write("hello").await.expect("create ok");
        append_text(&path, " world").await.expect("append ok");
        let content = std::fs::read_to_string(&path).expect("read ok");
        assert_eq!(content, "hello world");
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn test_append_text_missing_file_returns_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("no-such.log");
        let res = append_text(&missing, "x").await;
        assert!(matches!(res, Err(ExecutionError::Unknown(_))));
    }

    // ── execute_shell_with_capture (real shell, same approach as existing env module tests) ──

    #[tokio::test]
    async fn test_execute_capture_basic_echo() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        let res = execute_shell_with_capture(&env, "echo hello", &default_options())
            .await
            .expect("capture ok");
        assert_eq!(res.output, "hello\n");
        assert_eq!(res.exit_code, Some(0));
        assert!(!res.cancelled);
        assert!(!res.truncated);
        assert!(res.full_output_path.is_none());
    }

    #[tokio::test]
    async fn test_execute_capture_nonzero_exit_code() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        let res = execute_shell_with_capture(&env, "exit 3", &default_options())
            .await
            .expect("capture ok");
        assert_eq!(res.exit_code, Some(3));
        assert!(!res.cancelled);
    }

    #[tokio::test]
    async fn test_execute_capture_merges_stdout_and_stderr() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        let res =
            execute_shell_with_capture(&env, "echo to-out; echo to-err >&2", &default_options())
                .await
                .expect("capture ok");
        assert!(res.output.contains("to-out"));
        assert!(res.output.contains("to-err"));
        assert_eq!(res.exit_code, Some(0));
    }

    #[tokio::test]
    async fn test_execute_capture_strips_carriage_returns() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        let res = execute_shell_with_capture(&env, "printf 'a\\r\\nb\\r\\n'", &default_options())
            .await
            .expect("capture ok");
        assert_eq!(res.output, "a\nb\n");
    }

    #[tokio::test]
    async fn test_execute_capture_on_chunk_receives_sanitized_chunks() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        let received = Arc::new(Mutex::new(String::new()));
        let got = received.clone();
        let opts = ShellCaptureOptions {
            on_chunk: Arc::new(move |s: &str| got.lock().unwrap().push_str(s)),
            ..default_options()
        };
        let res = execute_shell_with_capture(&env, "echo hello", &opts)
            .await
            .expect("capture ok");
        assert_eq!(*received.lock().unwrap(), "hello\n");
        assert_eq!(res.output, "hello\n");
    }

    #[tokio::test]
    async fn test_execute_capture_precancelled_returns_cancelled() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        let ct = CancellationToken::new();
        ct.cancel();
        let opts = ShellCaptureOptions {
            cancellation: Some(ct),
            ..default_options()
        };
        let res = execute_shell_with_capture(&env, "echo hello", &opts)
            .await
            .expect("capture ok (cancelled is not an error)");
        assert!(res.cancelled);
        assert_eq!(res.exit_code, None);
    }

    #[tokio::test]
    async fn test_execute_capture_large_output_spills_to_file() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        let res = execute_shell_with_capture(&env, "yes line | head -n 15000", &default_options())
            .await
            .expect("capture ok");
        assert!(res.truncated);
        let path = res.full_output_path.as_ref().expect("spilled path");
        let full = std::fs::read_to_string(path).expect("read full output");
        assert!(full.lines().count() > 10_000);
        assert!(res.output.lines().count() <= DEFAULT_MAX_LINES);
        assert!(res.output.len() < full.len());
        // truncation metadata exposed: total line count (15000 lines < 100KB tail buffer, no rolling trim)
        assert_eq!(res.truncation.total_lines, 15_000);
        assert_eq!(res.truncation.output_lines, DEFAULT_MAX_LINES);
        assert_eq!(res.truncation.content, res.output);
        let _ = std::fs::remove_file(path);
    }

    // ── sanitize switch (T-AP-04-7: the bash tool passes false to keep raw text) ──────────

    #[tokio::test]
    async fn test_execute_capture_sanitize_false_keeps_raw() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        // 0x01 control char + \r: removed on the sanitize path, kept as-is on the raw path
        let raw_opts = ShellCaptureOptions {
            sanitize: false,
            ..default_options()
        };
        let res = execute_shell_with_capture(&env, "printf 'a\\001b\\r\\n'", &raw_opts)
            .await
            .expect("capture ok");
        assert_eq!(res.output, "a\u{1}b\r\n");

        let sanitized =
            execute_shell_with_capture(&env, "printf 'a\\001b\\r\\n'", &default_options())
                .await
                .expect("capture ok");
        assert_eq!(sanitized.output, "ab\n");
    }

    // ── throttled snapshot callback (T-AP-04-7: on_snapshot + snapshot_interval) ──────

    #[tokio::test]
    async fn test_execute_capture_on_snapshot_fires_with_tail() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        let snapshots: std::sync::Arc<std::sync::Mutex<Vec<StreamSnapshot>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let got = snapshots.clone();
        let opts = ShellCaptureOptions {
            on_snapshot: Some(Arc::new(move |s: &StreamSnapshot| {
                got.lock().unwrap().push(StreamSnapshot {
                    content: s.content.clone(),
                    truncation: s.truncation.clone(),
                    full_output_path: s.full_output_path.clone(),
                });
            })),
            snapshot_interval: Some(Duration::ZERO), // fires for every chunk, timing-independent
            ..default_options()
        };
        let res = execute_shell_with_capture(
            &env,
            "for i in 1 2 3; do echo \"line-$i\"; sleep 0.05; done",
            &opts,
        )
        .await
        .expect("capture ok");
        assert_eq!(res.output, "line-1\nline-2\nline-3\n");
        let snaps = snapshots.lock().unwrap();
        assert!(!snaps.is_empty(), "expected >=1 snapshot");
        // snapshot content includes tail text received so far (the first snapshot already contains the first line)
        assert!(snaps.iter().any(|s| s.content.contains("line-1")));
        // small output does not overflow: full_output_path is None in all snapshots
        assert!(snaps.iter().all(|s| s.full_output_path.is_none()));
        // when the snapshot truncation is not truncated, truncated=false
        assert!(snaps.iter().all(|s| !s.truncation.truncated));
    }

    /// Multi-thread flavor: the writer task writing to the temp file must be scheduled concurrently with other workers —
    /// under a current-thread runtime, the read loop in `wait_for_child_process` never really blocks (data is continuously available),
    /// so the spawned writer task is starved and snapshots never see the path; the production runtime is multi-threaded, so the behavior is observable.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_execute_capture_on_snapshot_reports_spill_path() {
        let env = ExecutionEnv::new(PathBuf::from("/"), None, None, None);
        let snapshots: std::sync::Arc<std::sync::Mutex<Vec<StreamSnapshot>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let got = snapshots.clone();
        let opts = ShellCaptureOptions {
            on_snapshot: Some(Arc::new(move |s: &StreamSnapshot| {
                got.lock().unwrap().push(StreamSnapshot {
                    content: s.content.clone(),
                    truncation: s.truncation.clone(),
                    full_output_path: s.full_output_path.clone(),
                });
            })),
            snapshot_interval: Some(Duration::ZERO),
            ..default_options()
        };
        // 160KB output > 50KB overflow threshold: snapshots after overflow should carry the full output path
        let res = execute_shell_with_capture(
            &env,
            "yes xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx | head -n 5000",
            &opts,
        )
        .await
        .expect("capture ok");
        assert!(res.truncated);
        let snaps = snapshots.lock().unwrap();
        assert!(
            snaps.iter().any(|s| s.full_output_path.is_some()),
            "expected a snapshot with spill path, got {}",
            snaps.len()
        );
        // snapshots after overflow should already have truncation marked
        assert!(snaps.iter().any(|s| s.truncation.truncated));
        if let Some(path) = &res.full_output_path {
            let _ = std::fs::remove_file(path);
        }
    }
}
