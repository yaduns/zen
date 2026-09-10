use crate::env::shell_output::{
    execute_shell_with_capture, ShellCaptureOptions, StreamSnapshot, StreamSnapshotCallback,
};
use crate::env::{ExecutionEnv, ExecutionError, ShellConfig};
use crate::truncate::{
    format_size, truncation_to_json, TruncationKind, TruncationResult, DEFAULT_MAX_BYTES,
};
use crate::util::now_millis;
use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use zen_ai::tool::{Tool, ToolContext, ToolError, ToolExecutionResult, ToolUpdateCallback};
use zen_ai::types::{ContentBlock, TextContent, ToolResultMessage, Usage};

/// Bash command execution tool that allows the agent to run commands in a shell.
/// Supports:
/// - Arbitrary shell command execution (via unified execution channel `env::execute_shell_with_capture`, T-AP-04-7)
/// - Combined stdout/stderr capture
/// - Timeout control (via the timeout parameter, in seconds)
/// - Cancellation signal (CancellationToken)
/// - Output tail truncation (defaults to the last 2000 lines / 50KB)
/// - Streaming update callback (on_update, pushes partial output in real time)
///
/// parameter schema (JSON Schema):
/// - `command` (string, required): the bash command to execute
/// - `timeout` (number, optional): Timeout in seconds, no timeout by default
///
/// # Edge Cases
/// - Empty command returns `ToolError::InvalidArguments`.
/// - `timeout` must be a positive number, otherwise returns `ToolError::InvalidArguments`.
/// - When the working directory does not exist, the unified channel's cwd precheck returns an error, passed through as `ToolError::Execution`.
/// - When the command's exit code is non-zero, the output is still returned to the LLM, but the error is flagged via `is_error`.
/// - On timeout, `ToolError::Execution` contains the text "timed out".
/// - On cancellation, returns `ToolError::Cancelled`.
#[derive(Debug)]
pub struct BashTool {
    tool_context: ToolContext, // Bind the session environment at construction time
    shell: ShellConfig,        // Shell override config (shell_path/command_prefix, T-AP-02-3-fix)
}

impl BashTool {
    /// Construct the bash tool with shell override config (None fields fall back to the system default shell / no prefix).
    pub fn new(tool_context: ToolContext, shell: ShellConfig) -> Self {
        Self {
            tool_context,
            shell,
        }
    }

    /// Apply the shell command prefix: `{prefix}\n{command}` (aligned with pi `agent-session.ts`).
    fn resolve_command(&self, command: &str) -> String {
        match &self.shell.command_prefix {
            Some(prefix) if !prefix.is_empty() => format!("{prefix}\n{command}"),
            _ => command.to_string(),
        }
    }
}

/// Tool name: `bash`
/// Description:
/// "Execute a bash command in the current working directory. Returns stdout and
/// stderr. Output is truncated to last 2000 lines or 50KB (whichever is hit
/// first). If truncated, full output is saved to a temp file. Optionally provide
/// a timeout in seconds."
#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command in the current working directory. Returns stdout and \
            stderr. Output is truncated to last 2000 lines or 50KB (whichever is hit \
            first). If truncated, full output is saved to a temp file. Optionally provide \
            a timeout in seconds."
    }

    fn parameters(&self) -> &serde_json::Value {
        static PARAMS: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
        PARAMS.get_or_init(|| {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to execute"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (optional, no default timeout)"
                    }
                },
                "required": ["command"]
            })
        })
    }

    fn label(&self) -> &str {
        "bash"
    }

    fn prompt_snippet(&self) -> Option<&str> {
        // Reference: reference/pi-original/packages/coding-agent/src/core/tools/bash.ts
        Some("Execute bash commands (ls, grep, find, etc.)")
    }

    async fn execute(
        &self,
        tool_call_id: &str,
        params: &serde_json::Value,
        cancel: &CancellationToken,
        on_update: ToolUpdateCallback,
    ) -> Result<ToolExecutionResult, ToolError> {
        let (command, timeout) = parse_bash_args(params)?;
        let resolved = self.resolve_command(&command);
        let (content, exit_code, info, full_output_path) = execute_command(
            &resolved,
            &self.tool_context.working_dir,
            self.shell.shell_path.as_deref(),
            timeout,
            cancel,
            tool_call_id,
            Some(on_update),
        )
        .await?;

        let now = now_millis();

        let is_error = exit_code.map_or(false, |c| c != 0);

        // Truncation hint text (aligned with pi formatOutput's three branches; on truncation full_output_path must have spilled)
        let text = format_bash_output(&content, &info, full_output_path.as_deref());

        let message = ToolResultMessage {
            tool_call_id: tool_call_id.to_string(),
            tool_name: "bash".to_string(),
            content: vec![ContentBlock::Text(TextContent {
                text,
                text_signature: None,
            })],
            details: Some(serde_json::json!({
                "exitCode": exit_code,
                "totalLines": info.total_lines,
                "outputLines": info.output_lines,
                "truncated": info.truncated,
                "fullOutputPath": full_output_path.map(|p| p.to_string_lossy().into_owned()),
            })),
            usage: Some(Usage::default()),
            is_error,
            added_tool_names: None,
            timestamp: now,
        };

        Ok(ToolExecutionResult {
            message,
            terminate: false,
        })
    }
}

/// Truncation hint text assembly (aligned with pi `formatOutput`'s three branches): the bash tool appends to the end of the result text
/// a `[Showing ...]` section so the LLM/user knows the displayed output is the truncated tail rather than the full output, and gives the full
/// output file path. Returns content unchanged when not truncated or no full output path exists.
///
/// Branch semantics (verbatim alignment with pi bash.ts `formatOutput`):
/// - `last_line_partial`: single line exceeds limit, reports "Showing last {output_bytes} of line {end_line}
///   (line is {last line byte count})" — line size derived from the last line of content (equivalent to pi `getLastLineBytes`);
/// - `truncated_by == Lines`: reports "Showing lines {start}-{end} of {total_lines}";
/// - Otherwise (bytes): append `({format_size(DEFAULT_MAX_BYTES)} limit)` text.
fn format_bash_output(
    content: &str,
    truncation: &TruncationResult,
    full_output_path: Option<&std::path::Path>,
) -> String {
    if !truncation.truncated {
        return content.to_string();
    }
    let Some(path) = full_output_path else {
        return content.to_string();
    };
    let start_line = truncation.total_lines - truncation.output_lines + 1;
    let end_line = truncation.total_lines;
    let mut text = content.to_string();
    if truncation.last_line_partial {
        let last_line_size = content.rsplit('\n').next().unwrap_or("").len();
        text.push_str(&format!(
            "\n\n[Showing last {} of line {} (line is {}). Full output: {}]",
            format_size(truncation.output_bytes),
            end_line,
            format_size(last_line_size),
            path.display()
        ));
    } else if truncation.truncated_by == Some(TruncationKind::Lines) {
        text.push_str(&format!(
            "\n\n[Showing lines {}-{} of {}. Full output: {}]",
            start_line,
            end_line,
            truncation.total_lines,
            path.display()
        ));
    } else {
        text.push_str(&format!(
            "\n\n[Showing lines {}-{} of {} ({} limit). Full output: {}]",
            start_line,
            end_line,
            truncation.total_lines,
            format_size(DEFAULT_MAX_BYTES),
            path.display()
        ));
    }
    text
}

/// Unified execution channel error → tool error mapping: after the bash tool migration, `execute_shell_with_capture`'s
/// `ExecutionError` is converted to `ToolError`; cancellation/timeout semantics match the existing contract.
fn map_exec_error(e: ExecutionError) -> ToolError {
    match e {
        ExecutionError::Aborted => ToolError::Cancelled,
        ExecutionError::Timeout(_) => ToolError::Execution("Command timed out".into()),
        other => ToolError::Execution(format!("Command failed: {other}")),
    }
}

/// Extracts command/timeout from the arguments.
/// ```json
///   {
///     "command": "ls -la src/",
///     "timeout": 30
///   }
/// ```
/// # Parameters
/// - `params`: parameter JSON object of the tool call.
///
/// # Return Value
/// Tuple (command, timeout), where timeout is an Option<number of seconds>.
///
/// # Errors
/// - If `command` is missing or not a string, returns `ToolError::InvalidArguments`.
/// - If `command` is an empty string, returns `ToolError::InvalidArguments`.
/// - If `timeout` is not a positive number (or is non-null/undefined), returns `ToolError::InvalidArguments`.
fn parse_bash_args(params: &serde_json::Value) -> Result<(String, Option<f64>), ToolError> {
    let command = params
        .get("command")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            ToolError::InvalidArguments("Missing required 'command' parameter".into())
        })?;

    let timeout = params.get("timeout").and_then(|v| v.as_f64());

    Ok((command.to_string(), timeout))
}

/// Executes a shell command, captures output, and applies tail truncation.
///
/// Migrated to the unified `env` channel (`execute_shell_with_capture`, T-AP-04-7):
/// - Bounded in-memory streaming accumulation + spill to temp file on overflow (full output exposed via `full_output_path`);
/// - 100ms throttled snapshots drive `on_update` streaming partials (aligned with pi `BASH_UPDATE_THROTTLE_MS`);
/// - Inherits `env.exec`'s 5303 wait semantics (detached descendants holding the pipe don't hang) and process tree cleanup;
/// - `sanitize: false` keeps raw text (aligned with the pi bash tool not sanitizing); stdout/stderr arrive interleaved.
///
/// # Parameters
/// - `command`: The command string to execute.
/// - `working_dir`: working directory (base_dir for constructing the execution environment).
/// - `shell_path`: custom shell path (None = system default `sh`).
/// - `timeout`: Timeout in seconds (None = no timeout).
/// - `cancel`: cancellation token.
/// - `tool_call_id`: current tool call id (for assembling partial messages).
/// - `on_update`: streaming update callback (optional; ownership moved so the throttle closure can hold it).
///
/// # Return Value
/// Tuple of four (output content, exit code, truncation info, full output temp file path).
///
/// # Edge Cases
/// - If the cancellation token is triggered during command execution, returns `ToolError::Cancelled`.
/// - After a timeout fires, the process tree is killed and `ToolError::Execution` is returned containing "timed out".
/// - A non-zero exit code is not treated as an error — the output and exit code are both returned to the caller, which decides whether to flag them as an error.
async fn execute_command(
    command: &str,
    working_dir: &std::path::Path,
    shell_path: Option<&std::path::Path>,
    timeout: Option<f64>,
    cancel: &CancellationToken,
    tool_call_id: &str,
    on_update: Option<ToolUpdateCallback>,
) -> Result<(String, Option<i32>, TruncationResult, Option<PathBuf>), ToolError> {
    // 1. Construct the unified execution channel (mirroring the agent_session.rs::execute_bash precedent)
    let env = ExecutionEnv::new(
        working_dir.to_path_buf(),
        shell_path.map(|p| p.to_path_buf()),
        None,
        None,
    );

    // 2. Throttled snapshots (100ms, aligned with pi BASH_UPDATE_THROTTLE_MS) → partial on_update
    let snapshot_interval = Duration::from_millis(100);
    let on_snapshot: Option<StreamSnapshotCallback> = on_update.map(|cb| {
        let tool_call_id = tool_call_id.to_string();
        let cb: StreamSnapshotCallback = Arc::new(move |snap: &StreamSnapshot| {
            let message = ToolResultMessage {
                tool_call_id: tool_call_id.clone(),
                tool_name: "bash".to_string(),
                content: vec![ContentBlock::Text(TextContent {
                    text: snap.content.clone(),
                    text_signature: None,
                })],
                details: Some(serde_json::json!({
                    "truncation": if snap.truncation.truncated {
                        truncation_to_json(&snap.truncation)
                    } else {
                        serde_json::Value::Null
                    },
                    "fullOutputPath": snap
                        .full_output_path
                        .as_ref()
                        .map(|p| p.to_string_lossy().into_owned()),
                })),
                usage: Some(Usage::default()),
                is_error: false,
                added_tool_names: None,
                timestamp: now_millis(),
            };
            cb(ToolExecutionResult {
                message,
                terminate: false,
            });
        });
        cb
    });

    // 3. Assemble options and execute (unified channel)
    let options = ShellCaptureOptions {
        cwd: None, // Default to env.base_dir (= working_dir)
        env: None,
        timeout: timeout.map(Duration::from_secs_f64),
        cancellation: Some(cancel.clone()),
        on_chunk: Arc::new(|_| {}), // This path pushes via on_snapshot throttling and does not consume per-chunk callbacks
        sanitize: false,            // Keep raw text (aligned with the pi bash tool not sanitizing)
        on_snapshot,
        snapshot_interval: Some(snapshot_interval),
    };
    let captured = execute_shell_with_capture(&env, command, &options)
        .await
        .map_err(map_exec_error)?;

    // 4. Semantic mapping: cancellation → Cancelled; all other results passed through as-is
    if captured.cancelled {
        return Err(ToolError::Cancelled);
    }
    Ok((
        captured.output,
        captured.exit_code,
        captured.truncation,
        captured.full_output_path,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Construct a test ToolContext (metadata tests don't care about the specific working_dir value)
    fn test_ctx() -> ToolContext {
        ToolContext::new(std::path::PathBuf::from("/tmp"), None, None)
    }

    /// BashTool with default shell config (all None), reused by metadata tests.
    fn bash_tool() -> BashTool {
        BashTool::new(test_ctx(), ShellConfig::default())
    }
    // ─── parse_bash_args ───────────────────────────────────────────────────

    #[test]
    fn test_parse_bash_args_command_only() {
        let params = json!({ "command": "ls -la" });
        let (cmd, timeout) = parse_bash_args(&params).unwrap();
        assert_eq!(cmd, "ls -la");
        assert_eq!(timeout, None);
    }

    #[test]
    fn test_parse_bash_args_command_with_timeout() {
        let params = json!({ "command": "make build", "timeout": 60.0 });
        let (cmd, timeout) = parse_bash_args(&params).unwrap();
        assert_eq!(cmd, "make build");
        assert_eq!(timeout, Some(60.0));
    }

    #[test]
    fn test_parse_bash_args_timeout_as_integer() {
        // 30 is an integer in JSON; serde_json parses it as i64/u64, and as_f64 can still extract it
        let params = json!({ "command": "echo hi", "timeout": 30 });
        let (_, timeout) = parse_bash_args(&params).unwrap();
        assert_eq!(timeout, Some(30.0));
    }

    #[test]
    fn test_parse_bash_args_missing_command() {
        let params = json!({ "timeout": 10 });
        let err = parse_bash_args(&params).unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
        assert!(err.to_string().contains("command"));
    }

    #[test]
    fn test_parse_bash_args_empty_params() {
        let params = json!({});
        let err = parse_bash_args(&params).unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn test_parse_bash_args_command_not_string() {
        let params = json!({ "command": 42 });
        let err = parse_bash_args(&params).unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn test_parse_bash_args_timeout_as_string_ignored() {
        // timeout is a string, as_f64 returns None, so it should be treated as not provided
        let params = json!({ "command": "echo hi", "timeout": "30" });
        let (_, timeout) = parse_bash_args(&params).unwrap();
        assert_eq!(timeout, None);
    }

    // ─── BashTool metadata ────────────────────────────────────────────────

    #[test]
    fn test_bash_tool_name() {
        let tool = bash_tool();
        assert_eq!(tool.name(), "bash");
    }

    #[test]
    fn test_bash_tool_label() {
        let tool = bash_tool();
        assert_eq!(tool.label(), "bash");
    }

    #[test]
    fn test_bash_tool_description_not_empty() {
        let tool = bash_tool();
        assert!(!tool.description().is_empty());
        assert!(tool.description().contains("truncated"));
    }

    #[test]
    fn test_bash_tool_parameters_schema() {
        let tool = bash_tool();
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["command"].is_object());
        assert_eq!(
            params["required"].as_array().unwrap()[0].as_str().unwrap(),
            "command"
        );
        assert!(params["properties"]["timeout"].is_object());
        assert_eq!(params["required"].as_array().unwrap().len(), 1);
    }

    // ─── resolve_command (shell prefix, T-AP-02-3-fix) ─────────────────────

    #[test]
    fn test_resolve_command_prepends_prefix() {
        let tool = BashTool::new(
            test_ctx(),
            ShellConfig {
                shell_path: None,
                command_prefix: Some("shopt -s expand_aliases".to_string()),
            },
        );
        // Aligned with pi: `{prefix}\n{command}`
        assert_eq!(tool.resolve_command("ls"), "shopt -s expand_aliases\nls");
    }

    #[test]
    fn test_resolve_command_no_or_empty_prefix_passthrough() {
        assert_eq!(bash_tool().resolve_command("ls"), "ls");
        let tool = BashTool::new(
            test_ctx(),
            ShellConfig {
                shell_path: None,
                command_prefix: Some(String::new()),
            },
        );
        assert_eq!(tool.resolve_command("ls"), "ls");
    }

    // ─── execute_command (cancellation only) ────────────────────────────────

    #[tokio::test]
    async fn test_execute_command_cancelled_before_start() {
        let cancel = CancellationToken::new();
        cancel.cancel();

        let result = execute_command(
            "echo hello",
            std::path::Path::new("/tmp"),
            None,
            None,
            &cancel,
            "test-id",
            None,
        )
        .await;
        assert!(matches!(result, Err(ToolError::Cancelled)));
    }

    // ─── execute_command (real shell, unified channel integration) ───────────────────

    #[tokio::test]
    async fn test_execute_command_echo() {
        let cancel = CancellationToken::new();
        let (content, exit_code, info, full_output_path) = execute_command(
            "echo hello",
            std::path::Path::new("/tmp"),
            None,
            None,
            &cancel,
            "test-id",
            None,
        )
        .await
        .expect("exec ok");
        assert_eq!(content, "hello\n");
        assert_eq!(exit_code, Some(0));
        assert!(!info.truncated);
        assert!(full_output_path.is_none());
    }

    #[tokio::test]
    async fn test_execute_command_nonzero_exit() {
        let cancel = CancellationToken::new();
        let (_content, exit_code, _info, _path) = execute_command(
            "exit 3",
            std::path::Path::new("/tmp"),
            None,
            None,
            &cancel,
            "test-id",
            None,
        )
        .await
        .expect("exec ok");
        assert_eq!(exit_code, Some(3));
    }

    #[tokio::test]
    async fn test_execute_command_timeout() {
        let cancel = CancellationToken::new();
        let result = execute_command(
            "sleep 5",
            std::path::Path::new("/tmp"),
            None,
            Some(0.5),
            &cancel,
            "test-id",
            None,
        )
        .await;
        match result {
            Err(ToolError::Execution(msg)) => {
                assert!(msg.contains("timed out"), "msg: {msg}")
            }
            other => panic!("expected Execution timed out, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_execute_command_cancelled_during_execution() {
        let cancel = CancellationToken::new();
        let c2 = cancel.clone();
        let canceller = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            c2.cancel();
        });
        let result = execute_command(
            "sleep 5",
            std::path::Path::new("/tmp"),
            None,
            None,
            &cancel,
            "test-id",
            None,
        )
        .await;
        canceller.await.expect("canceller task");
        assert!(matches!(result, Err(ToolError::Cancelled)));
    }

    #[tokio::test]
    async fn test_execute_command_large_output_truncated_with_notice() {
        // End-to-end via BashTool::execute: truncation hint text and fullOutputPath are assembled in execute()
        let tool = bash_tool();
        let params = json!({ "command": "yes line | head -n 15000" });
        let cancel = CancellationToken::new();
        let result = tool
            .execute("test-id", &params, &cancel, Box::new(|_| {}))
            .await
            .expect("execute ok");
        let msg = result.message;
        let text = match &msg.content[0] {
            ContentBlock::Text(t) => t.text.clone(),
            other => panic!("expected text block, got {other:?}"),
        };
        assert!(
            text.contains("[Showing lines 13001-15000 of 15000. Full output:"),
            "text: {text}"
        );
        let full_output_path = msg
            .details
            .as_ref()
            .and_then(|d| d["fullOutputPath"].as_str())
            .expect("details.fullOutputPath")
            .to_string();
        assert!(text.contains(&full_output_path));
        assert_eq!(msg.details.as_ref().unwrap()["truncated"], true);
        let _ = std::fs::remove_file(&full_output_path);
    }

    #[tokio::test]
    async fn test_execute_command_on_update_receives_partial_snapshots() {
        let cancel = CancellationToken::new();
        let received: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let got = received.clone();
        let on_update: ToolUpdateCallback = Box::new(move |res: ToolExecutionResult| {
            if let Some(ContentBlock::Text(t)) = res.message.content.first() {
                got.lock().unwrap().push(t.text.clone());
            }
        });
        let (_content, exit_code, _info, _path) = execute_command(
            "for i in 1 2 3 4 5 6 7 8; do echo \"line-$i\"; sleep 0.08; done",
            std::path::Path::new("/tmp"),
            None,
            None,
            &cancel,
            "test-id",
            Some(on_update),
        )
        .await
        .expect("exec ok");
        assert_eq!(exit_code, Some(0));
        let parts = received.lock().unwrap();
        assert!(!parts.is_empty(), "expected >=1 partial snapshot");
        assert!(parts.iter().any(|t| t.contains("line-")));
    }

    // ─── format_bash_output (pure function) ─────────────────────────────────

    #[test]
    fn test_format_bash_output_not_truncated_passthrough() {
        let info =
            crate::truncate::truncate_tail("hello", crate::truncate::TruncationOptions::default());
        assert_eq!(
            format_bash_output("hello", &info, Some(std::path::Path::new("/tmp/x.log"))),
            "hello"
        );
    }

    #[test]
    fn test_format_bash_output_lines_branch() {
        let text = (0..3000)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let info = crate::truncate::truncate_tail(
            &text,
            crate::truncate::TruncationOptions {
                max_lines: 2000,
                max_bytes: usize::MAX,
            },
        );
        assert_eq!(info.truncated_by, Some(TruncationKind::Lines));
        let out = format_bash_output(
            &info.content,
            &info,
            Some(std::path::Path::new("/tmp/x.log")),
        );
        assert!(
            out.contains("[Showing lines 1001-3000 of 3000. Full output: /tmp/x.log]"),
            "out: {out}"
        );
    }

    #[test]
    fn test_format_bash_output_bytes_branch() {
        let line = "x".repeat(100);
        let text = (0..3000)
            .map(|_| line.clone())
            .collect::<Vec<_>>()
            .join("\n");
        let info = crate::truncate::truncate_tail(
            &text,
            crate::truncate::TruncationOptions {
                max_lines: usize::MAX,
                max_bytes: 100 * 1024,
            },
        );
        assert_eq!(info.truncated_by, Some(TruncationKind::Bytes));
        let out = format_bash_output(
            &info.content,
            &info,
            Some(std::path::Path::new("/tmp/x.log")),
        );
        assert!(
            out.contains("(50.0KB limit). Full output: /tmp/x.log]"),
            "out: {out}"
        );
    }

    #[test]
    fn test_format_bash_output_last_line_partial_branch() {
        let text = "y".repeat(200_000); // Single line 200KB > 50KB → tail truncation keeps only the line's tail portion
        let info =
            crate::truncate::truncate_tail(&text, crate::truncate::TruncationOptions::default());
        assert!(info.last_line_partial);
        let out = format_bash_output(
            &info.content,
            &info,
            Some(std::path::Path::new("/tmp/x.log")),
        );
        assert!(
            out.starts_with("yyy"),
            "content tail expected, got prefix: {}",
            &out[..20.min(out.len())]
        );
        assert!(out.contains("[Showing last "), "out: {out}");
        assert!(out.contains(" of line 1 (line is "), "out: {out}");
        assert!(out.contains("). Full output: /tmp/x.log]"), "out: {out}");
    }

    #[test]
    fn test_format_bash_output_truncated_without_path_passthrough() {
        let text = (0..3000)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let info = crate::truncate::truncate_tail(
            &text,
            crate::truncate::TruncationOptions {
                max_lines: 2000,
                max_bytes: usize::MAX,
            },
        );
        assert!(info.truncated);
        let out = format_bash_output(&info.content, &info, None);
        assert_eq!(out, info.content);
    }
}
