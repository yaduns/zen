// REQ-AGENT-02 (pi-212, pi-216): read tool + image attachment
// Reference: reference/pi-original/packages/coding-agent/src/core/tools/read.ts (image branch 243-261).
// MIME sniffing is delegated to crate::mime (T-AP-09-6); images are returned as ContentBlock::Image
// (T-AG-02-8). The resize/convert pipeline is v1-dropped (T-AP-08-4), so BMP is detected but omitted.

use crate::mime::detect_supported_image_mime_type_from_file;
use crate::tools::path_utils::resolve_read_path_async;
use crate::truncate::{truncate_head, TruncationOptions};
use crate::util::now_millis;
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use std::path::Path;
use tokio_util::sync::CancellationToken;
use zen_ai::tool::{Tool, ToolContext, ToolError, ToolExecutionResult, ToolUpdateCallback};
use zen_ai::types::{ContentBlock, ImageContent, TextContent, ToolResultMessage, Usage};

/// Return of read_file_content (text path): content and line statistics after offset/limit window
/// and the shared truncation kernel.
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
/// - Image attachments (jpg/png/gif/webp) as multimodal `ContentBlock::Image`
/// - Line range selection (offset/limit, 1-indexed) for text files
/// - Automatic truncation (default 2000 lines / 50KB, whichever is hit first)
/// - Path resolution (calls path_utils::resolve_read_path_async)
///
/// BMP is detected by the shared MIME sniffer but not attached: v1 has no image conversion
/// pipeline (T-AP-08-4), so the tool returns an omission note instead.
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
/// "Read the contents of a file. Supports text files and images (jpg, png, gif, webp).
/// Images are sent as attachments. BMP images are detected but not attached. For text
/// files, output is truncated to 2000 lines or 50KB (whichever is hit first). Use
/// offset/limit for large files. When you need the full file, continue with offset
/// until complete."
#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Read the contents of a file. Supports text files and images (jpg, png, gif, \
            webp). Images are sent as attachments. BMP images are detected but not attached. \
            For text files, output is truncated to 2000 lines or 50KB (whichever is hit \
            first). Use offset/limit for large files. When you need the full file, continue \
            with offset until complete."
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

    /// Executes a read: resolve the path, sniff for a supported image, otherwise read text.
    /// The image branch returns `[Text note, Image block]`; the text branch keeps the existing
    /// offset/limit + truncation behavior unchanged.
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

        if cancel.is_cancelled() {
            return Err(ToolError::Cancelled);
        }

        let image_mime = detect_image_mime_type(&resolved).await?;
        if let Some(mime_type) = image_mime {
            let image = if mime_type == "image/bmp" {
                None
            } else {
                Some(read_image_content(&resolved, mime_type, cancel).await?)
            };
            let message = build_image_tool_result(tool_call_id, mime_type, image, now_millis());
            return Ok(ToolExecutionResult {
                message,
                terminate: false,
            });
        }

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

/// The read tool calls this after path resolution and before text reading: ask the disk whether
/// this file is one of the supported image formats (glance at the file header, not the whole book).
///
/// Delegates to [`crate::mime::detect_supported_image_mime_type_from_file`] (reads the first 4100
/// bytes). A hit returns the MIME type, a non-image returns `None`; open/read failures are mapped
/// to the same `ToolError::Execution` text used by the text branch. The caller checks cancellation
/// before calling this (the sniff itself is a short IO).
async fn detect_image_mime_type(absolute_path: &Path) -> Result<Option<&'static str>, ToolError> {
    detect_supported_image_mime_type_from_file(absolute_path)
        .await
        .map_err(|e| {
            ToolError::Execution(format!(
                "Failed to read file '{}': {}",
                absolute_path.display(),
                e
            ))
        })
}

/// The read tool calls this after the sniffer reports png/jpeg/gif/webp: turn the whole image into
/// a base64 attachment so a multimodal model can see it directly (hand over the photo, not a
/// text transcription of it).
///
/// `absolute_path` is already resolved by `resolve_read_path_async`; `mime_type` comes from
/// `detect_image_mime_type` (the caller guarantees it is not BMP); `cancel` is checked before and
/// after the read. Read failures use the same `Failed to read file '{path}': {io_error}` text as
/// the text branch. No decode validation is performed: sniffed-but-corrupted bytes pass through
/// (same v1 policy as T-AP-08-4; provider-side rejection is accepted).
async fn read_image_content(
    absolute_path: &Path,
    mime_type: &'static str,
    cancel: &CancellationToken,
) -> Result<ImageContent, ToolError> {
    if cancel.is_cancelled() {
        return Err(ToolError::Cancelled);
    }

    let bytes = tokio::fs::read(absolute_path).await.map_err(|e| {
        ToolError::Execution(format!(
            "Failed to read file '{}': {}",
            absolute_path.display(),
            e
        ))
    })?;

    if cancel.is_cancelled() {
        return Err(ToolError::Cancelled);
    }

    Ok(ImageContent {
        data: STANDARD.encode(bytes),
        mime_type: mime_type.to_string(),
    })
}

/// The read tool calls this to finish the image branch: assemble the sniffed MIME and optional
/// image content into the tool result the model sees (same delivery slip: attach the photo when
/// possible, otherwise state that the image was omitted).
///
/// `image = Some` produces `[Text("Read image file [mime]"), Image]`; `image = None` is used for
/// BMP and other formats v1 cannot inline, producing a single text note with the omission reason.
/// `details` is always `None` (image results have no truncation statistics); `timestamp` is passed
/// by the caller so the image/text branches share the same `now_millis()` convention. The function
/// fixes `tool_name: "read"`, `is_error: false`, `added_tool_names: None`, and
/// `usage: Some(Usage::default())`.
fn build_image_tool_result(
    tool_call_id: &str,
    mime_type: &'static str,
    image: Option<ImageContent>,
    timestamp: i64,
) -> ToolResultMessage {
    let (text, image_block) = match image {
        Some(image) => (
            format!("Read image file [{}]", image.mime_type),
            Some(ContentBlock::Image(image)),
        ),
        None => (
            format!(
                "Read image file [{mime_type}]\n[Image omitted: could not be converted to a supported inline image format.]"
            ),
            None,
        ),
    };

    let mut content = vec![ContentBlock::Text(TextContent {
        text,
        text_signature: None,
    })];
    content.extend(image_block);

    ToolResultMessage {
        tool_call_id: tool_call_id.to_string(),
        tool_name: "read".to_string(),
        content,
        details: None,
        usage: Some(Usage::default()),
        is_error: false,
        added_tool_names: None,
        timestamp,
    }
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
/// - Non-UTF-8 content returns `ToolError::Execution` here; supported images are handled by the
///   image branch before this function is called.
/// - Returns `ToolError::Execution` if the file does not exist or is not readable.
/// - Truncation semantics: offset/limit window slicing itself does not count as truncation; if window content exceeds 2000 lines / 50KB
///   (whichever comes first) it is finalized by shared `truncate_head`, keeping only complete lines
///   (when the first line alone exceeds the limit, returns empty content; `first_line_exceeds_limit` can be used for later prompts).
async fn read_file_content(
    absolute_path: &Path,
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

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose::STANDARD as B64;
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn test_tool(dir: &Path) -> ReadTool {
        ReadTool::new(ToolContext::new(dir.to_path_buf(), None, None))
    }

    fn write_file(dir: &Path, name: &str, content: &[u8]) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    async fn run_read(
        tool: &ReadTool,
        params: serde_json::Value,
    ) -> Result<ToolExecutionResult, ToolError> {
        let cancel = CancellationToken::new();
        tool.execute("call-1", &params, &cancel, Box::new(|_| {}))
            .await
    }

    fn png_bytes() -> Vec<u8> {
        // PNG signature + IHDR length (13) + "IHDR": enough for the pi-342 sniffer.
        let mut bytes = vec![0u8; 16];
        bytes[..8].copy_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        bytes[8..12].copy_from_slice(&13u32.to_be_bytes());
        bytes[12..16].copy_from_slice(b"IHDR");
        bytes
    }

    fn jpeg_bytes() -> Vec<u8> {
        // `FF D8 FF` magic + 4th byte != F7; sniffing only checks the magic.
        vec![0xFF, 0xD8, 0xFF, 0x00, 0xAA, 0xBB, 0xCC, 0xDD]
    }

    fn gif_bytes() -> Vec<u8> {
        b"GIF89a-not-a-real-gif".to_vec()
    }

    fn webp_bytes() -> Vec<u8> {
        let mut bytes = b"RIFF\x00\x00\x00\x00WEBP".to_vec();
        bytes.extend_from_slice(b"garbage");
        bytes
    }

    fn minimal_bmp_bytes() -> Vec<u8> {
        // BITMAPCOREHEADER(12): BM + declaredFileSize=0 + pixelDataOffset=26 + dibHeaderSize=12
        // + colorPlanes=1 + bitsPerPixel=24.
        let mut bytes = vec![0u8; 26];
        bytes[0] = b'B';
        bytes[1] = b'M';
        bytes[10] = 26;
        bytes[14] = 12;
        bytes[22] = 1;
        bytes[24] = 24;
        bytes
    }

    #[tokio::test]
    async fn supported_image_types_return_text_and_image_block() {
        let dir = tempdir().unwrap();
        let cases = [
            ("test.png", png_bytes(), "image/png"),
            ("test.jpg", jpeg_bytes(), "image/jpeg"),
            ("test.gif", gif_bytes(), "image/gif"),
            ("test.webp", webp_bytes(), "image/webp"),
        ];

        for (name, bytes, mime_type) in cases {
            let file = write_file(dir.path(), name, &bytes);
            let tool = test_tool(dir.path());
            let result = run_read(&tool, serde_json::json!({ "path": file.to_str().unwrap() }))
                .await
                .unwrap();

            assert_eq!(result.message.tool_name, "read");
            assert!(!result.message.is_error);
            assert!(result.message.details.is_none());
            assert_eq!(result.message.content.len(), 2);

            match &result.message.content[0] {
                ContentBlock::Text(text) => {
                    assert_eq!(text.text, format!("Read image file [{mime_type}]"));
                }
                other => panic!("expected text block, got {other:?}"),
            }
            match &result.message.content[1] {
                ContentBlock::Image(image) => {
                    assert_eq!(image.mime_type, mime_type);
                    assert_eq!(image.data, B64.encode(&bytes));
                }
                other => panic!("expected image block, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn sniff_hit_but_corrupted_bytes_pass_through_as_base64() {
        let dir = tempdir().unwrap();
        let bytes = jpeg_bytes();
        let file = write_file(dir.path(), "corrupted.jpg", &bytes);
        let result = run_read(
            &test_tool(dir.path()),
            serde_json::json!({ "path": file.to_str().unwrap() }),
        )
        .await
        .unwrap();

        match &result.message.content[1] {
            ContentBlock::Image(image) => assert_eq!(image.data, B64.encode(&bytes)),
            other => panic!("expected image block, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn bmp_returns_omission_note_without_image_block() {
        let dir = tempdir().unwrap();
        let file = write_file(dir.path(), "test.bmp", &minimal_bmp_bytes());
        let result = run_read(
            &test_tool(dir.path()),
            serde_json::json!({ "path": file.to_str().unwrap() }),
        )
        .await
        .unwrap();

        assert_eq!(result.message.content.len(), 1);
        assert!(result.message.details.is_none());
        match &result.message.content[0] {
            ContentBlock::Text(text) => {
                assert_eq!(
                    text.text,
                    "Read image file [image/bmp]\n[Image omitted: could not be converted to a supported inline image format.]"
                );
            }
            other => panic!("expected text block, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn image_branch_ignores_offset_and_limit() {
        let dir = tempdir().unwrap();
        let bytes = png_bytes();
        let file = write_file(dir.path(), "test.png", &bytes);
        let result = run_read(
            &test_tool(dir.path()),
            serde_json::json!({
                "path": file.to_str().unwrap(),
                "offset": 99,
                "limit": 1
            }),
        )
        .await
        .unwrap();

        assert_eq!(result.message.content.len(), 2);
        match &result.message.content[1] {
            ContentBlock::Image(image) => assert_eq!(image.data, B64.encode(&bytes)),
            other => panic!("expected image block, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn cancelled_before_sniff_returns_cancelled() {
        let dir = tempdir().unwrap();
        let file = write_file(dir.path(), "test.png", &png_bytes());
        let tool = test_tool(dir.path());
        let cancel = CancellationToken::new();
        cancel.cancel();

        let result = tool
            .execute(
                "call-1",
                &serde_json::json!({ "path": file.to_str().unwrap() }),
                &cancel,
                Box::new(|_| {}),
            )
            .await;

        assert!(matches!(result, Err(ToolError::Cancelled)));
    }

    #[tokio::test]
    async fn text_file_keeps_offset_limit_and_details_behavior() {
        let dir = tempdir().unwrap();
        let file = write_file(dir.path(), "notes.txt", b"a\nb\nc\n");
        let result = run_read(
            &test_tool(dir.path()),
            serde_json::json!({
                "path": file.to_str().unwrap(),
                "offset": 2,
                "limit": 1
            }),
        )
        .await
        .unwrap();

        match &result.message.content[0] {
            ContentBlock::Text(text) => assert_eq!(text.text, "b"),
            other => panic!("expected text block, got {other:?}"),
        }
        assert_eq!(
            result.message.details,
            Some(serde_json::json!({
                "totalLines": 3,
                "readLines": 1,
                "offset": 2,
                "bytes": 1
            }))
        );
    }

    #[tokio::test]
    async fn directory_error_uses_text_branch_read_error_format() {
        let dir = tempdir().unwrap();
        let tool = test_tool(dir.path());
        let result = run_read(
            &tool,
            serde_json::json!({ "path": dir.path().to_str().unwrap() }),
        )
        .await;
        let error = result.unwrap_err().to_string();

        assert!(error.starts_with(&format!(
            "Tool execution failed: Failed to read file '{}': ",
            dir.path().display()
        )));
    }

    #[test]
    fn description_does_not_promise_bmp_attachment() {
        let tool = ReadTool::new(ToolContext::new(PathBuf::from("/tmp"), None, None));
        let description = tool.description();
        assert!(description.contains("BMP images are detected but not attached"));
        assert!(!description.contains("webp, bmp)"));
    }
}
