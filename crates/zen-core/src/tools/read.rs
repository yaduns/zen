use crate::tools::path_utils::resolve_read_path_async;
use crate::truncate::{truncate_head, TruncationOptions};
use crate::util::now_millis;
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use zen_ai::tool::{Tool, ToolContext, ToolError, ToolExecutionResult, ToolUpdateCallback};
use zen_ai::types::{ContentBlock, TextContent, ToolResultMessage, Usage};

/// Return of read_file_content: content and line statistics after offset/limit window and shared truncation kernel.
struct ReadOutput {
    // Final content after truncation (complete lines within the window)
    content: String,
    // Total lines in the entire file (details.totalLines; note: not window line count)
    total_lines: usize,
    // Number of lines output after truncation (details.readLines)
    output_lines: usize,
    // Byte count of truncated content (details.bytes)
    content_len: usize,
}

/// File reading tool that allows the agent to read file contents.
/// Supports:
/// - Text file reading (UTF-8 encoded)
/// - Line range selection (offset/limit, 1-indexed)
/// - Automatic truncation (default 2000 lines / 50KB, whichever is hit first)
/// - Path resolution (calls path_utils::resolve_read_path_async)
///
/// Phase 1 does not support image file reading (see T-AG-02-8, Phase 2).
///
/// parameter schema (JSON Schema):
/// - `path` (string, required): file path to read (relative or absolute)
/// - `offset` (number, optional): starting line number (1-indexed), defaults to the beginning of the file
/// - `limit` (number, optional): maximum number of lines to read, defaults to no limit (subject to truncation)
#[derive(Debug)]
pub struct ReadTool {
    tool_context: ToolContext, // New: bind the session environment at construction
}

impl ReadTool {
    /// Creates a new ReadTool instance.
    pub fn new(tool_context: ToolContext) -> Self {
        Self { tool_context }
    }
}

/// Tool name: `read`
///
/// Parameter description is returned via the tool's description(), with the following content:
/// "Read the contents of a file. Supports text files and images (jpg, png, gif,
/// webp, bmp). Images are sent as attachments. For text files, output is
/// truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit
/// for large files. When you need the full file, continue with offset until complete."
#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Read the contents of a file. Supports text files and images (jpg, png, gif, \
            webp, bmp). Images are sent as attachments. For text files, output is \
            truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit \
            for large files. When you need the full file, continue with offset until complete."
    }

    fn parameters(&self) -> &serde_json::Value {
        static PARAMS: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
        PARAMS.get_or_init(|| {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read (relative or absolute)"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Starting line number (1-indexed); defaults to the beginning of the file"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum lines to read; unlimited by default (subject to truncation limit)"
                    }
                },
                "required": ["path"]
            })
        })
    }

    fn label(&self) -> &str {
        "read"
    }

    fn prompt_snippet(&self) -> Option<&str> {
        Some("Read file contents")
    }

    fn prompt_guidelines(&self) -> &'static [&'static str] {
        // Reference: reference/pi-original/packages/coding-agent/src/core/tools/read.ts
        &["Use read to examine files instead of cat or sed."]
    }

    async fn execute(
        &self,
        tool_call_id: &str,
        params: &serde_json::Value,
        cancel: &CancellationToken,
        _on_update: ToolUpdateCallback,
    ) -> Result<ToolExecutionResult, ToolError> {
        let (file_path, offset, limit) = parse_read_args(params)?;

        let resolved = resolve_read_path_async(file_path, &self.tool_context.working_dir)
            .await
            .map_err(|e| {
                ToolError::Execution(format!("Failed to resolve read path '{file_path}': {e}"))
            })?;
        let info = read_file_content(&resolved, offset, limit, cancel).await?;

        let now = now_millis();

        let message = ToolResultMessage {
            tool_call_id: tool_call_id.to_string(),
            tool_name: "read".to_string(),
            content: vec![ContentBlock::Text(TextContent {
                text: info.content,
                text_signature: None,
            })],
            details: Some(serde_json::json!({
                "totalLines": info.total_lines,
                "readLines": info.output_lines,
                "offset": offset.unwrap_or(1),
                "bytes": info.content_len,
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

/// Extracts path/offset/limit from the arguments.
/// ```json
///   {
///     "path": "src/main.rs",
///     "offset": 10,
///     "limit": 50
///   }
/// ```
/// # Parameters
/// - `params`: parameter JSON object of the tool call.
///
/// # Return Value
/// Triple (path, offset, limit), where offset and limit are Options.
///
/// # Errors
/// If `path` is missing or not a string, returns `ToolError::InvalidArguments`.
/// If `offset` or `limit` is not a positive integer (or is non-null/undefined), returns `ToolError::InvalidArguments`.
fn parse_read_args(
    params: &serde_json::Value,
) -> Result<(&str, Option<usize>, Option<usize>), ToolError> {
    let file_path = params
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ToolError::InvalidArguments("Missing required 'path' parameter".into()))?;

    let offset = params
        .get("offset")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let limit = params
        .get("limit")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    Ok((file_path, offset, limit))
}

/// Reads a text file, applies offset/limit window and shared truncation kernel (crate::truncate, T-AG-06-3).
///
/// # Parameters
/// - `absolute_path`: the resolved absolute path.
/// - `offset`: starting line number (1-indexed); None means from the beginning of the file.
/// - `limit`: maximum number of lines; None means no limit (but subject to truncation).
/// - `cancel`: cancellation token.
///
/// # Return Value
/// Window content (finalized by shared `truncate_head`) + file line statistics.
///
/// # Edge Cases
/// - Returns an error if offset exceeds the total number of lines in the file.
/// - Binary files return garbled text after reading — this is expected behavior; the LLM may decide to use the bash tool to handle them.
/// - Returns `ToolError::Execution` if the file does not exist or is not readable.
/// - Truncation semantics: offset/limit window slicing itself does not count as truncation; if window content exceeds 2000 lines / 50KB
///   (whichever comes first) it is finalized by shared `truncate_head`, keeping only complete lines
///   (when the first line alone exceeds the limit, returns empty content; `first_line_exceeds_limit` can be used for later prompts).
async fn read_file_content(
    absolute_path: &std::path::Path,
    offset: Option<usize>,
    limit: Option<usize>,
    cancel: &CancellationToken,
) -> Result<ReadOutput, ToolError> {
    if cancel.is_cancelled() {
        return Err(ToolError::Cancelled);
    }

    let content = tokio::fs::read_to_string(absolute_path)
        .await
        .map_err(|e| {
            ToolError::Execution(format!(
                "Failed to read file '{}': {}",
                absolute_path.display(),
                e
            ))
        })?;

    let total_lines = content.lines().count();

    // offset/limit window slicing (1-indexed → 0-indexed; slicing itself does not count as truncation)
    let start = offset.unwrap_or(1).saturating_sub(1).min(total_lines);
    let available = total_lines - start;
    let requested = limit.unwrap_or(available).min(available);
    let window: String = content
        .lines()
        .skip(start)
        .take(requested)
        .collect::<Vec<_>>()
        .join("\n");

    // Window content then passes through shared truncation kernel (2000 lines / 50KB, whichever comes first, keeping only complete lines)
    let result = truncate_head(&window, TruncationOptions::default());

    Ok(ReadOutput {
        content_len: result.content.len(),
        content: result.content,
        total_lines,
        output_lines: result.output_lines,
    })
}
