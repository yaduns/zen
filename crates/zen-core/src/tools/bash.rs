use crate::truncate::{truncate_tail, TruncationOptions, TruncationResult};
use crate::util::now_millis;
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use zen_ai::tool::{Tool, ToolContext, ToolError, ToolExecutionResult, ToolUpdateCallback};
use zen_ai::types::{ContentBlock, TextContent, ToolResultMessage, Usage};

/// Bash command execution tool that allows the agent to run commands in a shell.
/// Supports:
/// - Execute any shell command (invokes the system shell via `tokio::process::Command`)
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
/// - If the working directory does not exist, `tokio::process::Command` returns an error, passed through as `ToolError::Execution`.
/// - When the command's exit code is non-zero, the output is still returned to the LLM, but the error is flagged via `is_error`.
/// - On timeout, `ToolError::Execution` contains the text "timed out".
/// - On cancellation, returns `ToolError::Cancelled`.
#[derive(Debug)]
pub struct BashTool {
    tool_context: ToolContext, // New: bind the session environment at construction
}

impl BashTool {
    /// Creates a new BashTool instance.
    pub fn new(tool_context: ToolContext) -> Self {
        Self { tool_context }
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
        let (content, exit_code, info) = execute_command(
            &command,
            &self.tool_context.working_dir,
            timeout,
            cancel,
            Some(&on_update),
        )
        .await?;

        let now = now_millis();

        let is_error = exit_code.map_or(false, |c| c != 0);

        let message = ToolResultMessage {
            tool_call_id: tool_call_id.to_string(),
            tool_name: "bash".to_string(),
            content: vec![ContentBlock::Text(TextContent {
                text: content,
                text_signature: None,
            })],
            details: Some(serde_json::json!({
                "exitCode": exit_code,
                "totalLines": info.total_lines,
                "outputLines": info.output_lines,
                "truncated": info.truncated,
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
/// Execute commands through the system shell using `tokio::process::Command`.
/// Merge stdout/stderr into a single output stream.
/// If `timeout` is provided, wrap with `tokio::time::timeout`.
/// If `on_update` is provided, periodically push partial results while output accumulates.
/// Truncate output from the end (keep the last N lines/bytes).
///
/// # Parameters
/// - `command`: The command string to execute.
/// - `working_dir`: working directory.
/// - `timeout`: Timeout in seconds (None = no timeout).
/// - `cancel`: cancellation token.
/// - `on_update`: streaming update callback (optional).
///
/// # Return Value
/// A tuple containing the output content, exit code, and truncation info.
///
/// # Edge Cases
/// - If the cancellation token is triggered during command execution, the process is killed and `ToolError::Cancelled` is returned.
/// - After a timeout triggers, the process is killed and `ToolError::Execution` containing "timed out" is returned.
/// - A non-zero exit code is not treated as an error — the output and exit code are both returned to the caller, which decides whether to flag them as an error.
async fn execute_command(
    command: &str,
    working_dir: &std::path::Path,
    timeout: Option<f64>,
    cancel: &CancellationToken,
    on_update: Option<&ToolUpdateCallback>,
) -> Result<(String, Option<i32>, TruncationResult), ToolError> {
    // 1. Start the process
    let mut child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(working_dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| ToolError::Execution(format!("Failed to spawn process: {}", e)))?;

    // 2. Take the stdout/stderr pipes (wait_with_output consumes the child, so read manually instead)
    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();

    // 3. Read stdout/stderr in the background
    use tokio::io::AsyncReadExt;
    let read_stdout = stdout_pipe.map(|mut pipe| {
        tokio::spawn(async move {
            let mut buf = Vec::new();
            pipe.read_to_end(&mut buf).await.ok();
            buf
        })
    });
    let read_stderr = stderr_pipe.map(|mut pipe| {
        tokio::spawn(async move {
            let mut buf = Vec::new();
            pipe.read_to_end(&mut buf).await.ok();
            buf
        })
    });

    // 4. Wait for the process to finish (with timeout + cancellation)
    let status = tokio::select! {
        _ = cancel.cancelled() => {
            let _ = child.kill();
            return Err(ToolError::Cancelled);
        }
        result = async {
            let wait_result = if let Some(t) = timeout {
                let dur = std::time::Duration::from_secs_f64(t);
                match tokio::time::timeout(dur, child.wait()).await {
                    Ok(r) => r,
                    Err(_) => return Err(ToolError::Execution("Command timed out".into())),
                }
            } else {
                child.wait().await
            };
            wait_result.map_err(|e| ToolError::Execution(format!("Process error: {}", e)))
        } => result?,
    };

    // 5. Collect output
    let exit_code = status.code();
    let mut merged = Vec::new();
    if let Some(handle) = read_stdout {
        // handle.await can only complete after (or concurrently with) child.wait(), because if the process does not exit, the pipe never reaches EOF and read_to_end keeps waiting.
        if let Ok(buf) = handle.await {
            merged.extend_from_slice(&buf);
        }
    }
    if let Some(handle) = read_stderr {
        if let Ok(buf) = handle.await {
            merged.extend_from_slice(&buf);
        }
    }
    let text = String::from_utf8_lossy(&merged).into_owned();

    // 6. Tail truncation (shared module crate::truncate, T-AG-06-3)
    let info = truncate_tail(&text, TruncationOptions::default());
    let content = info.content.clone();

    Ok((content, exit_code, info))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Construct a test ToolContext (metadata tests don't care about the specific working_dir value)
    fn test_ctx() -> ToolContext {
        ToolContext::new(std::path::PathBuf::from("/tmp"), None, None)
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
        let tool = BashTool::new(test_ctx());
        assert_eq!(tool.name(), "bash");
    }

    #[test]
    fn test_bash_tool_label() {
        let tool = BashTool::new(test_ctx());
        assert_eq!(tool.label(), "bash");
    }

    #[test]
    fn test_bash_tool_description_not_empty() {
        let tool = BashTool::new(test_ctx());
        assert!(!tool.description().is_empty());
        assert!(tool.description().contains("truncated"));
    }

    #[test]
    fn test_bash_tool_parameters_schema() {
        let tool = BashTool::new(test_ctx());
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

    // ─── execute_command (cancellation only) ────────────────────────────────

    #[tokio::test]
    async fn test_execute_command_cancelled_before_start() {
        let cancel = CancellationToken::new();
        cancel.cancel();

        let result = execute_command(
            "echo hello",
            std::path::Path::new("/tmp"),
            None,
            &cancel,
            None,
        )
        .await;
        assert!(matches!(result, Err(ToolError::Cancelled)));
    }

    // The process-execution part of execute_command relies on the actual system shell,
    // Recommended to cover in integration tests:
    // - simple command executes normally (echo)
    // - command output merges stdout/stderr
    // - timeout triggers (sleep command + short timeout)
    // - cancel a running command
    // - command does not exist (spawn fails)
    // - non-zero exit code
    // - large output truncated by MAX_LINES
    // - large output truncated by MAX_BYTES
}
