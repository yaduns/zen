// REQ-AGENT-02 (pi-214, pi-215): edit tool
//
// File editing tool, allows the agent to perform precise text replacement in files (oldText → newText).
// Relies on capabilities provided by the shared edit_diff module (pi-215): fuzzy matching, line ending handling, diff generation, etc.
//
// Reference Pi original implementation:
// - reference/pi-original/packages/coding-agent/src/core/tools/edit.ts (127 lines)
// - reference/pi-original/packages/coding-agent/src/core/tools/edit-diff.ts (500 lines)

use crate::tools::edit_diff::{
    apply_edits_to_normalized_content, detect_line_ending, normalize_to_lf, restore_line_endings,
    strip_bom, Edit, EditError, EditToolDetails,
};
use crate::tools::path_utils::resolve_to_cwd;
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use zen_ai::tool::{Tool, ToolContext, ToolError, ToolExecutionResult, ToolUpdateCallback};
use zen_ai::types::{ContentBlock, TextContent, ToolResultMessage, Usage};

/// File editing tool, allows the agent to perform precise text replacement in files.
///
/// Each edits entry specifies an `oldText` (must be unique and an exact match in the file) and
/// the replacement `newText`. All edits match against the same original file content; replacements are applied in reverse
/// order to keep offsets stable.
///
/// # Parameter Schema
///
/// ```json
/// {
///   "type": "object",
///   "properties": {
///     "path": {
///       "type": "string",
///       "description": "Path to the file to edit (relative or absolute)"
///     },
///     "edits": {
///       "type": "array",
///       "description": "One or more targeted replacements, each matched against the original file",
///       "items": {
///         "type": "object",
///         "properties": {
///           "oldText": { "type": "string", "description": "Exact text to match, must be unique" },
///           "newText": { "type": "string", "description": "Replacement text" }
///         },
///         "required": ["oldText", "newText"]
///       }
///     }
///   },
///   "required": ["path", "edits"]
/// }
/// ```
///
/// # Execution flow
///
/// 1. Parse parameters (path, edits list)
/// 2. Resolve path relative to working directory (`resolve_to_cwd`)
/// 3. Check that the file exists and is readable/writable
/// 4. Read file content (UTF-8)
/// 5. Strip BOM (strip before matching, restore when writing back)
/// 6. Detect line ending type (CRLF or LF), normalize to LF for matching
/// 7. Perform fuzzy match lookup for all edits (exact match first, Unicode normalization match on failure)
/// 8. Validation: empty oldText, duplicate matches, and overlapping edits are all treated as errors
/// 9. Apply replacements in reverse order
/// 10. Restore BOM and original line endings
/// 11. Write back to file
/// 12. Generate diff and unified patch display info
///
/// # Edge Cases
///
/// - `path` missing or not a string → `ToolError::InvalidArguments`
/// - `edits` missing, not an array, or empty → `ToolError::InvalidArguments`
/// - `oldText` or `newText` missing or not a string in any edit → `ToolError::InvalidArguments`
/// - File does not exist or is read-only → `ToolError::Execution` (includes `EditError::FileAccess` details)
/// - Cancellation triggered → `ToolError::Cancelled` (checked before/after each IO step, does not interrupt in-flight IO)
/// - Multiple edits overlap → `ToolError::Execution` (includes `EditError::Overlap` details)
/// - oldText appears multiple times → `ToolError::Execution` (includes `EditError::Duplicate` details)
/// - oldText not found → `ToolError::Execution` (includes `EditError::NotFound` details)
/// - No content change after replacement → `ToolError::Execution` (includes `EditError::NoChange` details)
///
/// # Design decisions
///
/// - No pluggable `EditOperations` abstraction is provided (Phase 1 uses direct `tokio::fs`),
///   add it via T-AP-04-2 when needed.
/// - Does not include TUI rendering components (`EditRenderState`/`EditCallRenderComponent`);
///   rendering logic belongs to zen-tui/zen-app.
#[derive(Debug)]
pub struct EditTool {
    tool_context: ToolContext, // New: bind the session environment at construction
}

impl EditTool {
    /// Create a new EditTool instance.
    pub fn new(tool_context: ToolContext) -> Self {
        Self { tool_context }
    }
}

#[async_trait]
impl Tool for EditTool {
    /// Tool name: `"edit"`.
    fn name(&self) -> &str {
        "edit"
    }

    /// Tool description, visible to the LLM, used to decide whether to call this tool.
    ///
    /// Emphasizes exact text replacement, uniqueness requirement, non-overlap constraint.
    fn description(&self) -> &str {
        "Edit a single file using exact text replacement. Every edits[].oldText must \
            match a unique, non-overlapping region of the original file. If two changes \
            affect the same block or nearby lines, merge them into one edit instead of \
            emitting overlapping edits. Do not include large unchanged regions just to \
            connect distant changes."
    }

    /// Tool parameters JSON Schema.
    ///
    /// Lazily initialized via `OnceLock`, avoiding reallocation on every call.
    /// The schema defines two required fields:
    /// - `path` (string): path of the file to edit.
    /// - `edits` (array): list of edit operations, each element containing `oldText` and `newText`.
    fn parameters(&self) -> &serde_json::Value {
        static PARAMS: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
        PARAMS.get_or_init(|| {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file to edit" },
                    "edits": {
                        "type": "array",
                        "description": "One or more targeted replacements.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": { "type": "string", "description": "Text to match" },
                                "newText": { "type": "string", "description": "Replacement text" }
                            },
                            "required": ["oldText", "newText"]
                        }
                    }
                },
                "required": ["path", "edits"]
            })
        })
    }

    /// Human-readable tool label, `"edit"`.
    fn label(&self) -> &str {
        "edit"
    }

    fn prompt_snippet(&self) -> Option<&str> {
        // Reference: reference/pi-original/packages/coding-agent/src/core/tools/edit.ts
        Some("Make precise file edits with exact text replacement, including multiple disjoint edits in one call")
    }

    fn prompt_guidelines(&self) -> &'static [&'static str] {
        // Reference: reference/pi-original/packages/coding-agent/src/core/tools/edit.ts
        &[
            "Use edit for precise changes (edits[].oldText must match exactly)",
            "When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls",
            "Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
            "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.",
        ]
    }

    /// Execute an edit tool call.
    ///
    /// # Parameters
    ///
    /// - `tool_call_id`: tool call ID, used to match the tool result message.
    /// - `params`: parameter JSON object processed by `prepare_arguments`,
    ///   containing `path` (string) and `edits` (array of {oldText, newText}).
    /// - `ctx`: runtime context (working directory, etc.) used to resolve relative paths.
    /// - `cancel`: cancellation token; the tool should stop as soon as possible when a cancellation signal is received.
    /// - `_on_update`: callback (currently unused; kept for interface compatibility).
    ///
    /// # Flow
    ///
    /// 1. `parse_edit_args` parses arguments → `EditError` mapped to `ToolError::InvalidArguments`
    /// 2. `resolve_to_cwd` resolves the path
    /// 3. `edit_file` performs the file edit → `EditError` mapped to `ToolError::Execution`
    /// 4. Assemble `ToolResultMessage` (including diff/patch details)
    ///
    /// # Return Value
    ///
    /// On success returns a `ToolExecutionResult`, whose `message.content` contains the success message text.
    /// `message.details` contains `diff`, `patch`, and `firstChangedLine`.
    /// `terminate` is always `false` (edit tools do not trigger loop termination).
    ///
    /// # Errors
    ///
    /// - Argument parsing failure → `ToolError::InvalidArguments`
    /// - File does not exist / read-only / IO error → `ToolError::Execution`
    /// - Edit validation failure (not found / duplicate / overlap / no change) → `ToolError::Execution`
    /// - Cancellation triggered → `ToolError::Cancelled`
    async fn execute(
        &self,
        tool_call_id: &str,
        params: &serde_json::Value,
        cancel: &CancellationToken,
        _on_update: ToolUpdateCallback,
    ) -> Result<ToolExecutionResult, ToolError> {
        let (file_path, edits) = parse_edit_args(params)?;
        let abs = resolve_to_cwd(&file_path, self.tool_context.working_dir.as_path());
        let details = edit_file(&abs, &edits, cancel)
            .await
            .map_err(|e| map_edit_error(e, &file_path))?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        let msg = ToolResultMessage {
            tool_call_id: tool_call_id.into(),
            tool_name: "edit".into(),
            content: vec![ContentBlock::Text(TextContent {
                text: format!(
                    "Successfully replaced {} block(s) in {}.",
                    edits.len(),
                    file_path
                ),
                text_signature: None,
            })],
            details: Some(serde_json::json!({
                "diff": details.diff, "patch": details.patch,
                "firstChangedLine": details.first_changed_line,
            })),
            usage: Some(Usage::default()),
            is_error: false,
            added_tool_names: None,
            timestamp: now,
        };
        Ok(ToolExecutionResult {
            message: msg,
            terminate: false,
        })
    }
}

/// Maps `EditError` to `ToolError`.
///
/// Edit-related validation errors (NotFound, Duplicate, Overlap, etc.) directly use `err.to_string()`
/// as the error message. File IO errors use a more specific formatted message.
///
/// # Parameters
///
/// - `err`: The edit error.
/// - `_path`: File path (not currently used directly; kept for future extension).
///
/// # Return Value
///
/// The mapped `ToolError::Execution`.
fn map_edit_error(err: EditError, _path: &str) -> ToolError {
    match &err {
        EditError::NotFound { .. }
        | EditError::Duplicate { .. }
        | EditError::EmptyOldText { .. }
        | EditError::NoChange { .. }
        | EditError::Overlap { .. }
        | EditError::CantPreserveLines { .. } => ToolError::Execution(err.to_string()),
        EditError::FileAccess { reason, .. } => {
            ToolError::Execution(format!("Could not edit file: {}", reason))
        }
        EditError::FileRead { source, .. } => {
            ToolError::Execution(format!("Failed to read file: {}", source))
        }
        EditError::FileWrite { source, .. } => {
            ToolError::Execution(format!("Failed to write file: {}", source))
        }
    }
}

/// Extracts `path` and `edits` from the tool call arguments.
///
/// Expected JSON format:
///
/// ```json
/// {
///   "path": "src/main.rs",
///   "edits": [
///     { "oldText": "fn old()", "newText": "fn new()" },
///     { "oldText": "// comment", "newText": "/* comment */" }
///   ]
/// }
/// ```
///
/// # Parameters
///
/// - `params`: parameter JSON object of the tool call.
///
/// # Return Value
///
/// A tuple `(path, edits)`, where `path` is the raw path string and `edits` is a list of `Edit`.
///
/// # Errors
///
/// - `path` is missing, not a string, or empty → `ToolError::InvalidArguments`
/// - `edits` missing, not an array, or empty → `ToolError::InvalidArguments`
/// - In each edit, `oldText` is missing or not a string → `ToolError::InvalidArguments`
/// - In each edit, `newText` is missing or not a string → `ToolError::InvalidArguments`
///
/// # Edge Cases
///
/// - Extra arguments are ignored (no error).
/// - `oldText` and `newText` support empty strings (an empty `newText` deletes the matched text;
///   An empty `oldText` will later be detected as an `EmptyOldText` error in `apply_edits_to_normalized_content`).
fn parse_edit_args(params: &serde_json::Value) -> Result<(String, Vec<Edit>), ToolError> {
    let file_path = params
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ToolError::InvalidArguments("Missing 'path'".into()))?;
    if file_path.is_empty() {
        return Err(ToolError::InvalidArguments(
            "'path' must not be empty".into(),
        ));
    }
    let arr = params
        .get("edits")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ToolError::InvalidArguments("'edits' must be a non-empty array".into()))?;
    if arr.is_empty() {
        return Err(ToolError::InvalidArguments(
            "'edits' must not be empty".into(),
        ));
    }
    let edits: Vec<Edit> = arr
        .iter()
        .map(|item| {
            let old = item
                .get("oldText")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidArguments("Each edit needs 'oldText'".into()))?;
            let new = item
                .get("newText")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidArguments("Each edit needs 'newText'".into()))?;
            Ok(Edit {
                old_text: old.into(),
                new_text: new.into(),
            })
        })
        .collect::<Result<_, _>>()?;
    Ok((file_path.into(), edits))
}

/// Core function that performs the file edit operation.
///
/// Performs the following steps in order, checking the cancellation signal before and after each step:
///
/// 1. **Check cancellation signal**: if already cancelled, return `EditError::FileAccess`.
/// 2. **Check file existence**: `tokio::fs::metadata`; return `FileAccess` if the file does not exist or is not accessible.
/// 3. **Check cancellation signal**.
/// 4. **Read file**: `tokio::fs::read_to_string`, UTF-8 encoded.
/// 5. **Check cancellation signal**.
/// 6. **BOM handling**: `strip_bom`, strips the BOM so matching works (the LLM does not include the invisible BOM in oldText).
/// 7. **Line ending detection**: `detect_line_ending`, records the original line ending type (CRLF or LF).
/// 8. **LF normalization**: `normalize_to_lf`, converts to LF uniformly for matching.
/// 9. **Apply edits**: `apply_edits_to_normalized_content`, performs fuzzy matching, validation, and application.
/// 10. **Check cancellation signal**.
/// 11. **Restore line endings**: `restore_line_endings`, restores the original line endings.
/// 12. **Write back to file**: `tokio::fs::write`, prefixed with the BOM.
/// 13. **Check cancellation signal**.
/// 14. **Generate diff**: `generate_diff_string` + `generate_unified_patch`.
///
/// # Parameters
///
/// - `abs`: the resolved absolute path.
/// - `edits`: list of edit operations.
/// - `cancel`: cancellation token.
///
/// # Return Value
///
/// Returns `EditToolDetails` on success, containing the diff string, unified patch, and the first changed line number.
///
/// # Errors
///
/// Returns `EditError`, mapped to `ToolError` by the caller's `map_edit_error`.
///
/// # Edge Cases
///
/// - Cancellation is checked before and after each IO step but does not interrupt an in-flight IO operation —
///   `Cancelled` is only returned after the current IO completes, avoiding releasing resources mid-IO.
/// - Returns a specific `FileAccess` error when the file does not exist, rather than `FileRead`.
/// - Empty files (0 bytes) can be edited normally (oldText must match the empty string, but `EmptyOldText` blocks an empty oldText).
/// - Files containing only a BOM: after the BOM is stripped the content is empty; behavior depends on `apply_edits_to_normalized_content`.
async fn edit_file(
    abs: &std::path::Path,
    edits: &[Edit],
    cancel: &CancellationToken,
) -> Result<EditToolDetails, EditError> {
    if cancel.is_cancelled() {
        return Err(EditError::FileAccess {
            path: abs.to_string_lossy().into(),
            reason: "Cancelled".into(),
        });
    }
    let _meta = tokio::fs::metadata(abs)
        .await
        .map_err(|e| EditError::FileAccess {
            path: abs.to_string_lossy().into(),
            reason: e.to_string(),
        })?;
    if cancel.is_cancelled() {
        return Err(EditError::FileAccess {
            path: abs.to_string_lossy().into(),
            reason: "Cancelled".into(),
        });
    }
    let raw = tokio::fs::read_to_string(abs)
        .await
        .map_err(|e| EditError::FileRead {
            path: abs.to_string_lossy().into(),
            source: e,
        })?;
    if cancel.is_cancelled() {
        return Err(EditError::FileAccess {
            path: abs.to_string_lossy().into(),
            reason: "Cancelled".into(),
        });
    }
    let (bom, content) = strip_bom(&raw);
    let ending = detect_line_ending(content);
    let normalized = normalize_to_lf(content);
    let result = apply_edits_to_normalized_content(&normalized, edits, &abs.to_string_lossy())?;
    if cancel.is_cancelled() {
        return Err(EditError::FileAccess {
            path: abs.to_string_lossy().into(),
            reason: "Cancelled".into(),
        });
    }
    let final_content = bom + &restore_line_endings(&result.new_content, ending);
    tokio::fs::write(abs, &final_content)
        .await
        .map_err(|e| EditError::FileWrite {
            path: abs.to_string_lossy().into(),
            source: e,
        })?;
    if cancel.is_cancelled() {
        return Err(EditError::FileAccess {
            path: abs.to_string_lossy().into(),
            reason: "Cancelled".into(),
        });
    }
    let (diff, fcl) =
        crate::tools::edit_diff::generate_diff_string(&result.base_content, &result.new_content, 4);
    let patch = crate::tools::edit_diff::generate_unified_patch(
        &abs.to_string_lossy(),
        &result.base_content,
        &result.new_content,
        4,
    );
    Ok(EditToolDetails {
        diff,
        patch,
        first_changed_line: fcl,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Unit test coverage: argument parsing, tool metadata, file edit success/failure.
    // Integration tests (filesystem IO) use tempfile to create temporary directories.

    /// Construct a test ToolContext (metadata tests don't care about the specific working_dir value)
    fn test_ctx() -> ToolContext {
        ToolContext::new(std::path::PathBuf::from("/tmp"), None, None)
    }

    #[test]
    fn test_parse_edit_args_normal() {
        let (p, e) =
            parse_edit_args(&json!({"path":"x","edits":[{"oldText":"a","newText":"b"}]})).unwrap();
        assert_eq!(p, "x");
        assert_eq!(e.len(), 1);
    }
    #[test]
    fn test_parse_edit_args_missing_path() {
        assert!(matches!(
            parse_edit_args(&json!({"edits":[]})),
            Err(ToolError::InvalidArguments(_))
        ));
    }
    #[test]
    fn test_parse_edit_args_empty_edits() {
        assert!(matches!(
            parse_edit_args(&json!({"path":"x","edits":[]})),
            Err(ToolError::InvalidArguments(_))
        ));
    }
    #[test]
    fn test_edit_tool_name() {
        assert_eq!(EditTool::new(test_ctx()).name(), "edit");
    }
    #[test]
    fn test_edit_tool_label() {
        assert_eq!(EditTool::new(test_ctx()).label(), "edit");
    }

    #[tokio::test]
    async fn test_edit_file_success() {
        let dir = tempfile::tempdir().unwrap();
        let fp = dir.path().join("t.txt");
        std::fs::write(&fp, "hello world").unwrap();
        let r = edit_file(
            &fp,
            &[Edit {
                old_text: "hello".into(),
                new_text: "hi".into(),
            }],
            &CancellationToken::new(),
        )
        .await
        .unwrap();
        assert!(!r.diff.is_empty());
        assert_eq!(std::fs::read_to_string(&fp).unwrap(), "hi world");
    }
    #[tokio::test]
    async fn test_edit_file_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let r = edit_file(&dir.path().join("nope.txt"), &[], &CancellationToken::new()).await;
        assert!(r.is_err());
    }
    #[tokio::test]
    async fn test_execute_success() {
        let dir = tempfile::tempdir().unwrap();
        let fp = dir.path().join("t.txt");
        std::fs::write(&fp, "hello world").unwrap();
        let ctx = ToolContext::new(dir.path().into(), None, None);
        let tool = EditTool::new(ctx);
        let r = tool
            .execute(
                "c1",
                &json!({"path":"t.txt","edits":[{"oldText":"hello","newText":"hi"}]}),
                &CancellationToken::new(),
                Box::new(|_| {}),
            )
            .await
            .unwrap();
        assert!(!r.terminate);
        assert_eq!(r.message.tool_name, "edit");
        assert_eq!(std::fs::read_to_string(&fp).unwrap(), "hi world");
    }
}
