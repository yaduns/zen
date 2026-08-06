use crate::tools::path_utils::resolve_read_path_async;
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use zen_ai::tool::{Tool, ToolContext, ToolError, ToolExecutionResult, ToolUpdateCallback};
use zen_ai::types::{ContentBlock, TextContent, ToolResultMessage, Usage};

const MAX_LINES: usize = 2000;
const MAX_BYTES: usize = 50 * 1024;

struct TruncationInfo {
    // Truncated content
    content: String,
    // Whether truncation occurred
    truncated: bool,
    // truncation reason (lines/bytes), None when truncated=false
    truncated_by: Option<TruncationKind>,
    // total line count of the original content
    total_lines: usize,
    // line count of the truncated output
    output_lines: usize,
    // whether the first line exceeds the byte limit (making it completely impossible to display)
    first_line_exceeds_limit: bool,
}

enum TruncationKind {
    // content line count exceeds max_lines
    Lines,
    // content byte count exceeds max_bytes
    Bytes,
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
pub struct ReadTool;

impl ReadTool {
    /// Creates a new ReadTool instance.
    pub fn new() -> Self {
        Self
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

    async fn execute(
        &self,
        tool_call_id: &str,
        params: &serde_json::Value,
        ctx: &ToolContext,
        cancel: &CancellationToken,
        _on_update: ToolUpdateCallback,
    ) -> Result<ToolExecutionResult, ToolError> {
        let (file_path, offset, limit) = parse_read_args(params)?;

        let resolved = resolve_read_path_async(file_path, &ctx.working_dir).await;
        let (content, info) = read_file_content(&resolved, offset, limit, cancel).await?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let message = ToolResultMessage {
            tool_call_id: tool_call_id.to_string(),
            tool_name: "read".to_string(),
            content: vec![ContentBlock::Text(TextContent {
                text: content,
                text_signature: None,
            })],
            details: Some(serde_json::json!({
                "totalLines": info.total_lines,
                "readLines": info.output_lines,
                "offset": offset.unwrap_or(1),
                "bytes": info.content.len(),
            })),
            usage: Usage::default(),
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

/// Read a text file, applying offset/limit and truncation.
///
/// # Parameters
/// - `absolute_path`: the resolved absolute path.
/// - `offset`: starting line number (1-indexed); None means from the beginning of the file.
/// - `limit`: maximum number of lines; None means no limit (but subject to truncation).
/// - `cancel`: cancellation token.
///
/// # Return Value
/// Tuple containing the file content and truncation info.
///
/// # Edge Cases
/// - Returns an error if offset exceeds the total number of lines in the file.
/// - Binary files return garbled text after reading — this is expected behavior; the LLM may decide to use the bash tool to handle them.
/// - Returns `ToolError::Execution` if the file does not exist or is not readable.
async fn read_file_content(
    absolute_path: &std::path::Path,
    offset: Option<usize>,
    limit: Option<usize>,
    cancel: &CancellationToken,
) -> Result<(String, TruncationInfo), ToolError> {
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

    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();

    let start = offset.unwrap_or(1).saturating_sub(1).min(total_lines);
    let available = total_lines - start;
    let requested = limit.unwrap_or(available).min(available);
    let truncated_lines = requested.min(MAX_LINES);

    let selected: Vec<&str> = lines[start..start + truncated_lines].to_vec();
    let mut result = selected.join("\n");

    let truncated_by_bytes = result.len() > MAX_BYTES;
    if truncated_by_bytes {
        result.truncate(MAX_BYTES);
        if let Some(pos) = result.rfind('\n') {
            result.truncate(pos);
        }
    }

    let truncated_by_lines = truncated_lines < available;
    let truncated = truncated_by_lines || truncated_by_bytes;

    let info = TruncationInfo {
        content: result.clone(),
        truncated,
        truncated_by: if truncated {
            Some(if truncated_by_lines {
                TruncationKind::Lines
            } else {
                TruncationKind::Bytes
            })
        } else {
            None
        },
        total_lines,
        output_lines: result.lines().count(),
        // Raw content size exceeds 50KB, triggering byte truncation; content is empty after truncation; both conditions together mean: the file's first line alone exceeds 50KB.
        first_line_exceeds_limit: truncated_by_bytes && result.is_empty(),
    };

    Ok((result, info))
}
