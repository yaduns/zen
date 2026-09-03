use crate::tools::file_mutation_queue::with_file_mutation_queue;
use crate::tools::path_utils::resolve_to_cwd;
use crate::util::now_millis;
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use zen_ai::tool::{Tool, ToolContext, ToolError, ToolExecutionResult, ToolUpdateCallback};
use zen_ai::types::{ContentBlock, TextContent, ToolResultMessage, Usage};

/// File write tool, allowing the agent to create or overwrite files.
///
/// Supports:
/// - File writing (UTF-8 encoded, using `tokio::fs::write`)
/// - Automatically creates parent directories (using `tokio::fs::create_dir_all`)
/// - Path resolution (calls `path_utils::resolve_to_cwd`)
/// - Cancellation signal (CancellationToken, checked before/after writing)
///
/// parameter schema (JSON Schema):
/// - `path` (string, required): File path to write (relative or absolute)
/// - `content` (string, required): File content to write
///
/// # Behavior
/// - If the file does not exist, create it.
/// - If the file already exists, **overwrite** its content (no appending).
/// - If the parent directory does not exist, create it automatically (recursively).
/// - After a successful write, return a `Successfully wrote N bytes to <path>` message.
///
/// # Edge Cases
/// - `path` missing or not a string → `ToolError::InvalidArguments`.
/// - `content` missing or not a string → `ToolError::InvalidArguments`.
/// - `path` is an empty string → `ToolError::InvalidArguments`.
/// - Path not writable (insufficient permissions) → `ToolError::Execution`.
/// - Path is a directory → `ToolError::Execution` (behavior of `tokio::fs::write`).
/// - Cancellation triggered → `ToolError::Cancelled`.
/// - Write size exceeds filesystem limits → `ToolError::Execution`.
#[derive(Debug)]
pub struct WriteTool {
    tool_context: ToolContext, // New: bind the session environment at construction
}

impl WriteTool {
    /// Create a new WriteTool instance.
    pub fn new(tool_context: ToolContext) -> Self {
        Self { tool_context }
    }
}

#[async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates the file if it doesn't exist, overwrites \
         if it does. Automatically creates parent directories."
    }

    fn parameters(&self) -> &serde_json::Value {
        static PARAMS: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
        PARAMS.get_or_init(|| {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write (relative or absolute)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            })
        })
    }

    fn label(&self) -> &str {
        "write"
    }

    fn prompt_snippet(&self) -> Option<&str> {
        // Reference: reference/pi-original/packages/coding-agent/src/core/tools/write.ts
        Some("Create or overwrite files")
    }

    fn prompt_guidelines(&self) -> &'static [&'static str] {
        // Reference: reference/pi-original/packages/coding-agent/src/core/tools/write.ts
        &["Use write only for new files or complete rewrites."]
    }

    async fn execute(
        &self,
        tool_call_id: &str,
        params: &serde_json::Value,
        cancel: &CancellationToken,
        _on_update: ToolUpdateCallback,
    ) -> Result<ToolExecutionResult, ToolError> {
        let (file_path, content) = parse_write_args(params)?;

        let absolute_path = resolve_to_cwd(&file_path, self.tool_context.working_dir.as_path())
            .map_err(|e| {
                ToolError::Execution(format!("Failed to resolve path '{file_path}': {e}"))
            })?;

        // let success_msg = write_file(&absolute_path, &content, cancel).await?;
        // File paths are stored in a static, locked HashMap; when operating on a file, acquire the Mutex lock and clone the Arc reference;
        // When the file operation ends, by checking the lock against this file's lock and that the Arc reference count is 2 (including the original Arc and the cloned reference),
        // it is considered that there are no other operations on this file, and the file path is removed from the HashMap.
        // The above is the serialization scheme for multiple simultaneous modifications to individual files, using a LazyLock static variable and Mutex's exclusive lock
        let success_msg = with_file_mutation_queue(&absolute_path, || async {
            write_file(&absolute_path, &content, cancel).await
        })
        .await?;

        let now = now_millis();

        let message = ToolResultMessage {
            tool_call_id: tool_call_id.to_string(),
            tool_name: "write".to_string(),
            content: vec![ContentBlock::Text(TextContent {
                text: success_msg,
                text_signature: None,
            })],
            details: Some(serde_json::json!({
                "path": absolute_path.to_string_lossy(),
                "bytes": content.len(),
            })),
            usage: Some(Usage::default()),
            is_error: false,
            added_tool_names: None,
            timestamp: now,
        };

        Ok(ToolExecutionResult {
            message,
            terminate: false,
        })
    }
}

/// Extract path and content from the arguments.
///
/// ```json
///   {
///     "path": "src/main.rs",
///     "content": "fn main() {}"
///   }
/// ```
///
/// # Parameters
/// - `params`: parameter JSON object of the tool call.
///
/// # Return Value
/// A tuple (path, content), where path is the original path string and content is the content to write.
///
/// # Errors
/// - If `path` is missing or not a string, return `ToolError::InvalidArguments`.
/// - If `content` is missing or not a string, return `ToolError::InvalidArguments`.
/// - If `path` is an empty string, return `ToolError::InvalidArguments`.
fn parse_write_args(params: &serde_json::Value) -> Result<(String, String), ToolError> {
    let file_path = params
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ToolError::InvalidArguments("Missing required 'path' parameter".into()))?;

    if file_path.is_empty() {
        return Err(ToolError::InvalidArguments(
            "'path' parameter must not be empty".into(),
        ));
    }

    let content = params
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            ToolError::InvalidArguments("Missing required 'content' parameter".into())
        })?;

    Ok((file_path.to_string(), content.to_string()))
}

/// Performs the file write operation.
///
/// Performs in order:
/// 1. Check the cancellation signal (`cancel.is_cancelled()`)
/// 2. Create parent directories (`tokio::fs::create_dir_all`)
/// 3. Check the cancellation signal
/// 4. Write the file contents (`tokio::fs::write`)
/// 5. Check the cancellation signal
/// 6. Return a success message
///
/// # Parameters
/// - `absolute_path`: the resolved absolute path.
/// - `content`: the file content to write.
/// - `cancel`: cancellation token.
///
/// # Return Value
/// On success, returns a formatted success message string (e.g. `Successfully wrote 42 bytes to src/main.rs`).
///
/// # Edge Cases
/// - The cancellation signal is checked before each step, but cancellation does not return immediately — it waits for the current IO operation to complete before returning
///   `ToolError::Cancelled` (to avoid releasing resources while IO is in progress; see Pi's `withFileMutationQueue` pattern).
/// - `create_dir_all` does not error when parent directories already exist (idempotent operation).
/// - An empty string still writes successfully (empty file).
async fn write_file(
    absolute_path: &std::path::Path,
    content: &str,
    cancel: &CancellationToken,
) -> Result<String, ToolError> {
    if cancel.is_cancelled() {
        return Err(ToolError::Cancelled);
    }

    // Create parent directories (if they don't exist)
    if let Some(parent) = absolute_path.parent() {
        if !parent.as_os_str().is_empty() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                ToolError::Execution(format!(
                    "Failed to create parent directory '{}': {}",
                    parent.display(),
                    e
                ))
            })?;
        }
    }

    if cancel.is_cancelled() {
        return Err(ToolError::Cancelled);
    }

    // Write the file contents
    let byte_len = content.len();
    tokio::fs::write(absolute_path, content)
        .await
        .map_err(|e| {
            ToolError::Execution(format!(
                "Failed to write file '{}': {}",
                absolute_path.display(),
                e
            ))
        })?;

    if cancel.is_cancelled() {
        return Err(ToolError::Cancelled);
    }

    Ok(format!(
        "Successfully wrote {} bytes to {}",
        byte_len,
        absolute_path.to_string_lossy()
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

    // -- parse_write_args --

    #[test]
    fn test_parse_write_args_normal() {
        let params = json!({
            "path": "src/main.rs",
            "content": "fn main() {}"
        });
        let (path, content) = parse_write_args(&params).unwrap();
        assert_eq!(path, "src/main.rs");
        assert_eq!(content, "fn main() {}");
    }

    #[test]
    fn test_parse_write_args_missing_path() {
        let params = json!({
            "content": "fn main() {}"
        });
        let err = parse_write_args(&params).unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
        assert!(err.to_string().contains("path"));
    }

    #[test]
    fn test_parse_write_args_missing_content() {
        let params = json!({
            "path": "src/main.rs"
        });
        let err = parse_write_args(&params).unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
        assert!(err.to_string().contains("content"));
    }

    #[test]
    fn test_parse_write_args_path_not_string() {
        let params = json!({
            "path": 42,
            "content": "fn main() {}"
        });
        let err = parse_write_args(&params).unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn test_parse_write_args_content_not_string() {
        let params = json!({
            "path": "src/main.rs",
            "content": 42
        });
        let err = parse_write_args(&params).unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn test_parse_write_args_empty_path() {
        let params = json!({
            "path": "",
            "content": "fn main() {}"
        });
        let err = parse_write_args(&params).unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn test_parse_write_args_empty_content() {
        let params = json!({
            "path": "src/main.rs",
            "content": ""
        });
        let (path, content) = parse_write_args(&params).unwrap();
        assert_eq!(path, "src/main.rs");
        assert_eq!(content, "");
    }

    #[test]
    fn test_parse_write_args_extra_fields_ignored() {
        let params = json!({
            "path": "src/main.rs",
            "content": "fn main() {}",
            "extra": "ignored"
        });
        let result = parse_write_args(&params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_write_args_path_null() {
        let params = json!({
            "path": null,
            "content": "fn main() {}"
        });
        let err = parse_write_args(&params).unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn test_parse_write_args_content_null() {
        let params = json!({
            "path": "src/main.rs",
            "content": null
        });
        let err = parse_write_args(&params).unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    // -- WriteTool metadata --

    #[test]
    fn test_write_tool_name() {
        let tool = WriteTool::new(test_ctx());
        assert_eq!(tool.name(), "write");
    }

    #[test]
    fn test_write_tool_label() {
        let tool = WriteTool::new(test_ctx());
        assert_eq!(tool.label(), "write");
    }

    #[test]
    fn test_write_tool_description() {
        let tool = WriteTool::new(test_ctx());
        let desc = tool.description();
        assert!(!desc.is_empty());
        assert!(desc.contains("Write content"));
        assert!(desc.contains("overwrites"));
        assert!(desc.contains("parent directories"));
    }

    #[test]
    fn test_write_tool_parameters_schema() {
        let tool = WriteTool::new(test_ctx());
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["path"].is_object());
        assert!(params["properties"]["content"].is_object());
        assert_eq!(params["required"].as_array().unwrap().len(), 2);
        assert!(params["required"]
            .as_array()
            .unwrap()
            .contains(&json!("path")));
        assert!(params["required"]
            .as_array()
            .unwrap()
            .contains(&json!("content")));
    }

    #[test]
    fn test_write_tool_parameters_path_type() {
        let tool = WriteTool::new(test_ctx());
        let params = tool.parameters();
        assert_eq!(params["properties"]["path"]["type"], "string");
    }

    #[test]
    fn test_write_tool_parameters_content_type() {
        let tool = WriteTool::new(test_ctx());
        let params = tool.parameters();
        assert_eq!(params["properties"]["content"]["type"], "string");
    }

    // -- write_file (cancellation) --

    #[tokio::test]
    async fn test_write_file_cancelled() {
        let cancel = CancellationToken::new();
        cancel.cancel();
        let path = std::path::Path::new("/tmp/nonexistent/write_test.txt");
        let result = write_file(path, "hello", &cancel).await;
        assert!(matches!(result, Err(ToolError::Cancelled)));
    }

    // -- execute (cancellation) --

    #[tokio::test]
    async fn test_execute_cancelled() {
        let tool = WriteTool::new(test_ctx());
        let cancel = CancellationToken::new();
        cancel.cancel();
        let ctx = ToolContext::new(std::path::PathBuf::from("/tmp"), None, None);
        let params = json!({
            "path": "test.txt",
            "content": "hello"
        });
        let result = tool
            .execute("call-1", &params, &cancel, Box::new(|_| {}))
            .await;
        assert!(matches!(result, Err(ToolError::Cancelled)));
    }

    // -- execute (invalid arguments) --

    #[tokio::test]
    async fn test_execute_missing_path() {
        let tool = WriteTool::new(test_ctx());
        let cancel = CancellationToken::new();
        let ctx = ToolContext::new(std::path::PathBuf::from("/tmp"), None, None);
        let params = json!({
            "content": "hello"
        });
        let result = tool
            .execute("call-1", &params, &cancel, Box::new(|_| {}))
            .await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn test_execute_missing_content() {
        let tool = WriteTool::new(test_ctx());
        let cancel = CancellationToken::new();
        let ctx = ToolContext::new(std::path::PathBuf::from("/tmp"), None, None);
        let params = json!({
            "path": "test.txt"
        });
        let result = tool
            .execute("call-1", &params, &cancel, Box::new(|_| {}))
            .await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    // -- write_file (integration) --

    #[tokio::test]
    async fn test_write_file_success() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("subdir/test.txt");
        let result = write_file(&file_path, "hello world", &CancellationToken::new())
            .await
            .unwrap();
        assert!(result.contains("Successfully wrote"));
        assert!(result.contains("11 bytes"));
        assert!(std::fs::read_to_string(&file_path).unwrap() == "hello world");
    }

    #[tokio::test]
    async fn test_write_file_empty_content() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("empty.txt");
        let result = write_file(&file_path, "", &CancellationToken::new())
            .await
            .unwrap();
        assert!(result.contains("0 bytes"));
        assert!(std::fs::read_to_string(&file_path).unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_write_file_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("overwrite.txt");
        // Write first content
        std::fs::write(&file_path, "original").unwrap();
        // Overwrite
        write_file(&file_path, "updated", &CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "updated");
    }

    #[tokio::test]
    async fn test_write_file_creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("a/b/c/d/deep.txt");
        write_file(&file_path, "deep", &CancellationToken::new())
            .await
            .unwrap();
        assert!(file_path.exists());
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "deep");
    }

    // -- execute (integration) --

    #[tokio::test]
    async fn test_execute_success() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("output.txt");
        let file_path_str = file_path.to_string_lossy().to_string();
        let ctx = ToolContext::new(dir.path().to_path_buf(), None, None);
        let tool = WriteTool::new(ctx);
        let params = json!({
            "path": "output.txt",
            "content": "file content"
        });
        let result = tool
            .execute(
                "call-1",
                &params,
                &CancellationToken::new(),
                Box::new(|_| {}),
            )
            .await
            .unwrap();
        assert!(!result.terminate);
        assert_eq!(result.message.tool_name, "write");
        assert_eq!(result.message.tool_call_id, "call-1");
        assert!(!result.message.is_error);
        // Verify content
        let text = &result.message.content[0];
        if let ContentBlock::Text(tc) = text {
            assert!(tc.text.contains("Successfully wrote"));
            assert!(tc.text.contains(&file_path_str));
        } else {
            panic!("Expected Text content block");
        }
        // Verify file was written
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "file content");
        // Verify details
        let details = result.message.details.unwrap();
        assert_eq!(details["bytes"], 12);
    }
}
