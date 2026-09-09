// REQ-APP-04 (pi-350): ls tool.
// list directory contents: entries sorted alphabetically (case-insensitive), directory names have a `/` suffix, dotfiles included,
// truncated by entry count (default 500) and byte limit (50KB).
// corresponds to original TS: reference/pi-original/packages/coding-agent/src/core/tools/ls.ts.
// paradigm notes (see tasks/T-AP-04-6-implemented.md 'Ecosystem Reuse Verification Conclusions'):
// - readdir/stat directly consume `std::fs` (same as pi's direct node:fs reads, no subprocesses, unlike find's fd and grep's rg);
// - `LsOperations` pluggable interface (SSH remote proxy) dropped in v1 (no remote execution architecture, same as FindOperations/GrepOperations);
// - `formatLsCall`/`formatLsResult` render functions were eliminated by the T-AP-11-2 unified fallback layout (v1 has no renderer system),
//   entryLimit/truncation warnings are embedded in the content text as notices, so UI display via text output loses nothing;
// - `wrapToolDefinition` has been absorbed into the `Tool` trait method surface (T-AP-04-1 decision), with no wrapper layer.
// behavior mapping: pi `entries.sort((a,b) => a.toLowerCase().localeCompare(b.toLowerCase()))` →
// `sort_entries_case_insensitive` (`to_lowercase().cmp` stable sort; non-ASCII locale differences recorded as accepted deviations);
// pi `limit ?? DEFAULT_LIMIT` (no clamp, unlike grep's max(1)) → `limit.unwrap_or(DEFAULT_LIMIT)`;
// pi `continue`s on per-entry stat failure (skips entries that cannot be stat'ed) → skip on `metadata` failure;
// pi `truncateHead(rawOutput, { maxLines: MAX_SAFE_INTEGER })` → `truncate_head` (only the byte limit takes effect).

use crate::tools::path_utils::resolve_to_cwd;
use crate::truncate::{
    format_size, truncate_head, truncation_to_json, TruncationOptions, DEFAULT_MAX_BYTES,
};
use crate::util::now_millis;
use async_trait::async_trait;
use std::path::Path;
use tokio_util::sync::CancellationToken;
use zen_ai::tool::{Tool, ToolContext, ToolError, ToolExecutionResult, ToolUpdateCallback};
use zen_ai::types::{ContentBlock, TextContent, ToolResultMessage, Usage};

/// ls entry count limit (pi `DEFAULT_LIMIT = 500`).
const DEFAULT_LIMIT: usize = 500;

/// raw output scanned by `read_dir_entries` (private, not exported).
#[derive(Debug)]
struct LsScan {
    /// formatted entry name (directories have a `/` suffix, no path prefix).
    entries: Vec<String>,
    /// whether the limit is reached (collected count >= limit, aligned with pi `results.length >= effectiveLimit`).
    entry_limit_reached: bool,
}

/// final output assembled by `assemble_ls_result` (private, not exported).
struct LsResultAssembly {
    /// final content (including appended notices section; empty result is pi's "(empty directory)").
    content: String,
    /// details metadata (`entryLimitReached`/`truncation`, pi `LsToolDetails` eliminated; `None` when both absent).
    details: Option<serde_json::Value>,
}

/// Directory listing tool, invoked by the LLM when the agent needs to 'see what is in a directory (respecting dotfiles, no recursion needed)'—
/// User-visible effect: returns entry names sorted alphabetically (case-insensitive), directory names with a `/` suffix, and when over the limit (default 500 entries /
/// 50KB, whichever comes first), the body includes a `[..entries limit reached..]`-style notice, aligned with the pi ls tool behavior.
/// Parameter Schema (JSON Schema, static definition isomorphic to the read/find tools):
/// - `path` (string, optional): directory to list, defaults to the current working directory
/// - `limit` (number, optional): maximum number of entries to return, defaults to [`DEFAULT_LIMIT`]
///
/// Implemented with `std::fs::read_dir`/`std::fs::metadata` (same as pi's direct node:fs reads, no subprocesses),
/// see tasks/T-AP-04-6-implemented.md 'Ecosystem Reuse Verification Conclusions'.
#[derive(Debug)]
pub struct LsTool {
    tool_context: ToolContext,
}

impl LsTool {
    /// Construct an ls tool instance with the session environment (called by factory `create(ToolName::Ls)`; same structure as FindTool::new).
    pub fn new(tool_context: ToolContext) -> Self {
        Self { tool_context }
    }
}

#[async_trait]
impl Tool for LsTool {
    /// Tool name `"ls"` (aligned with pi tool name and `ToolName::Ls`).
    fn name(&self) -> &str {
        "ls"
    }

    /// LLM-visible description: list directory contents, sort alphabetically, directory names with a `/` suffix, include dotfiles, output truncated to
    /// `DEFAULT_LIMIT` entries or `DEFAULT_MAX_BYTES/1024` KB (whichever comes first)—wording matches pi verbatim
    /// `createLsToolDefinition.description`。
    fn description(&self) -> &str {
        "List directory contents. Returns entries sorted alphabetically, with '/' suffix for \
            directories. Includes dotfiles. Output is truncated to 500 entries or 50KB (whichever \
            is hit first)."
    }

    /// Parameter JSON Schema (both path/limit optional, **no required**, aligned with pi lsSchema), OnceLock static cache,
    /// Aligned with the read/find tools.
    fn parameters(&self) -> &serde_json::Value {
        static PARAMS: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
        PARAMS.get_or_init(|| {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list (default: current directory)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of entries to return (default: 500)"
                    }
                }
            })
        })
    }

    /// Label `"ls"`.
    fn label(&self) -> &str {
        "ls"
    }

    /// System prompt summary "List directory contents" (aligned with pi promptSnippet).
    fn prompt_snippet(&self) -> Option<&str> {
        Some("List directory contents")
    }

    /// Execute ls: parse args → `effective_limit = limit ?? DEFAULT_LIMIT` (no clamp, aligned with pi) →
    /// `resolve_to_cwd` resolves directory (default `.`) → existence precheck (not found → "Path not found: ...") →
    /// directory precheck (exists but not a directory → "Not a directory: ...") → run synchronous scan inside `spawn_blocking`
    /// (`read_dir_entries`) → `tokio::select!` race cancellation → cancellation recheck → assemble via `assemble_ls_result` →
    /// `ToolResultMessage`。
    /// Cancellation semantics: cancel already cancelled or cancelled mid-scan → `ToolError::Cancelled` (when cancelled mid-scan
    /// `read_dir_entries` breaks early to release the blocking thread and discards the result); `is_error=false`; details contain
    /// `entryLimitReached`/`truncation` (mapped from pi `LsToolDetails`; `None` when both absent).
    /// Edge case: directory exists but readdir fails (permissions, etc.) → `ToolError::Execution` ("Cannot read directory: ...");
    /// Skip entries that cannot be stat'ed (aligned with pi `catch { continue }`).
    async fn execute(
        &self,
        tool_call_id: &str,
        params: &serde_json::Value,
        cancel: &CancellationToken,
        _on_update: ToolUpdateCallback,
    ) -> Result<ToolExecutionResult, ToolError> {
        let (path, limit) = parse_ls_args(params);
        let effective_limit = limit.unwrap_or(DEFAULT_LIMIT);

        let dir_path = resolve_to_cwd(
            path.as_deref().unwrap_or("."),
            self.tool_context.working_dir.as_path(),
        )
        .map_err(|e| ToolError::Execution(format!("Failed to resolve ls path: {e}")))?;

        // Existence precheck (pi uses pathExists via `ops.exists`; if absent → report Path not found directly)
        if !dir_path.try_exists().unwrap_or(false) {
            return Err(ToolError::Execution(format!(
                "Path not found: {}",
                dir_path.display()
            )));
        }
        // Directory precheck (pi uses isDirectory via `ops.stat`; exists but not a directory → Not a directory)
        if !dir_path.is_dir() {
            return Err(ToolError::Execution(format!(
                "Not a directory: {}",
                dir_path.display()
            )));
        }

        let cancel_blocking = cancel.clone();
        let handle = tokio::task::spawn_blocking(move || {
            read_dir_entries(&dir_path, effective_limit, &cancel_blocking)
        });

        let joined = tokio::select! {
            _ = cancel.cancelled() => {
                // Don't abort: blocking tasks carry their own cancel check; the next entry breaks and self-terminates quickly
                return Err(ToolError::Cancelled);
            }
            joined = handle => joined,
        };
        let scan = match joined {
            Ok(Ok(scan)) => scan,
            Ok(Err(e)) => return Err(e),
            Err(join_err) => {
                return Err(ToolError::Execution(format!("ls task failed: {join_err}")));
            }
        };

        // Cancellation recheck: read_dir_entries may return normally at the same time the cancel signal arrives (early break); discard the result in that case
        if cancel.is_cancelled() {
            return Err(ToolError::Cancelled);
        }

        let assembly = assemble_ls_result(scan, effective_limit);

        let now = now_millis();
        let message = ToolResultMessage {
            tool_call_id: tool_call_id.to_string(),
            tool_name: "ls".to_string(),
            content: vec![ContentBlock::Text(TextContent {
                text: assembly.content,
                text_signature: None,
            })],
            details: assembly.details,
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

/// Extract (path, limit) from tool argument JSON. Treat non-string path or non-`u64` limit as default
/// (pi rejects invalid types via schema validation; zen follows find `parse_find_args` lenient parsing);
/// Both fields are optional. **No required fields** (unlike find/grep—ls has no pattern).
fn parse_ls_args(params: &serde_json::Value) -> (Option<String>, Option<usize>) {
    let path = params
        .get("path")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let limit = params
        .get("limit")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    (path, limit)
}

/// Sort entry names from readdir alphabetically (case-insensitive); called by `read_dir_entries` before per-entry stat—
/// User-visible effect: `ls` output matches pi, with "B.txt before a.txt, case-insensitive".
/// Semantics: stable sort by `a.to_lowercase().cmp(&b.to_lowercase())` (Rust `sort_by` is stable; entries with equal lowercase
/// keep readdir original order, aligned with JS stable sort + `localeCompare` returning 0). Non-ASCII
/// locale sort differences are recorded as accepted deviations (see tasks/T-AP-04-6-implemented.md "Resolved Decisions").
fn sort_entries_case_insensitive(entries: &mut [String]) {
    entries.sort_by_key(|s| s.to_lowercase());
}

/// Scan directory contents: `std::fs::read_dir` reads entry names (failure → `ToolError::Execution` "Cannot read
/// directory: {e}", aligned with pi `reject(new Error(\`Cannot read directory: ${e.message}\`))`) →
/// sort via `sort_entries_case_insensitive` → per-entry `std::fs::metadata`: directory → append `/` suffix to entry name,
/// stat failure → skip entry (aligned with pi `catch { continue }`); stop when entry count reaches `limit`
/// (`entry_limit_reached = collected_count >= limit`, aligned with pi `results.length >= effectiveLimit`,
/// **with limit=0, the first entry immediately reaches the limit and returns an empty set**). Check `cancel.is_cancelled()` per entry; if cancelled, break early and return
/// partial/empty result (converted to `ToolError::Cancelled` by the execute-layer cancellation recheck). Synchronous function; wrapped inside execute by
/// `spawn_blocking` execution.
fn read_dir_entries(
    dir_path: &Path,
    limit: usize,
    cancel: &CancellationToken,
) -> Result<LsScan, ToolError> {
    let read_dir = std::fs::read_dir(dir_path)
        .map_err(|e| ToolError::Execution(format!("Cannot read directory: {e}")))?;

    let mut names: Vec<String> = Vec::new();
    for entry in read_dir {
        match entry {
            Ok(e) => names.push(e.file_name().to_string_lossy().to_string()),
            // Skip single-entry read failures (platform-specific; pi's one-shot readdir has no such form), consistent with find single-entry traversal failures
            Err(_) => continue,
        }
    }
    sort_entries_case_insensitive(&mut names);

    let mut results: Vec<String> = Vec::new();
    let mut entry_limit_reached = false;
    for name in names {
        if cancel.is_cancelled() {
            break;
        }
        if results.len() >= limit {
            entry_limit_reached = true;
            break;
        }

        let full_path = dir_path.join(&name);
        let mut suffix = "";
        match std::fs::metadata(&full_path) {
            Ok(md) => {
                if md.is_dir() {
                    suffix = "/";
                }
            }
            // Skip entries that cannot be stat'ed (aligned with pi `catch { continue }`; permissions/dangling links, etc.)
            Err(_) => continue,
        }
        results.push(format!("{name}{suffix}"));
    }

    Ok(LsScan {
        entries: results,
        entry_limit_reached,
    })
}

/// Assemble scan output into final output: `entries.join("\n")` → `truncate_head` (`max_lines: usize::MAX`,
/// only the `DEFAULT_MAX_BYTES` byte limit applies, aligned with the equivalent pi call) → append notices (a `\n\n[..]` section, verbatim aligned
/// pi：`entryLimitReached` → "{limit} entries limit reached. Use limit={limit*2} for more"；
/// `truncation.truncated` → "{format_size(DEFAULT_MAX_BYTES)} limit reached"; multiple messages joined with ". ") →
/// details JSON (`entryLimitReached`/`truncation` are omitted from details when absent; `truncation` uses shared
/// `crate::truncate::truncation_to_json`; content intentionally omitted to prevent duplicate storage in JSONL, same decision as find/grep).
/// **Empty result (`entries.is_empty()`)** → content is pi's "(empty directory)", details=None—
/// Note this branch also swallows notices/details when entry_limit_reached is already set (limit=0 and directory non-empty),
/// aligned with pi's early `if (results.length === 0)` branch (see tasks/T-AP-04-6-implemented.md "Resolved Decisions").
fn assemble_ls_result(scan: LsScan, limit: usize) -> LsResultAssembly {
    if scan.entries.is_empty() {
        return LsResultAssembly {
            content: "(empty directory)".to_string(),
            details: None,
        };
    }

    let raw_output = scan.entries.join("\n");
    let truncation = truncate_head(
        &raw_output,
        TruncationOptions {
            max_lines: usize::MAX,
            max_bytes: DEFAULT_MAX_BYTES,
        },
    );

    let mut notices: Vec<String> = Vec::new();
    if scan.entry_limit_reached {
        notices.push(format!(
            "{limit} entries limit reached. Use limit={} for more",
            limit * 2
        ));
    }
    if truncation.truncated {
        notices.push(format!("{} limit reached", format_size(DEFAULT_MAX_BYTES)));
    }

    let mut details = serde_json::Map::new();
    if scan.entry_limit_reached {
        details.insert("entryLimitReached".into(), serde_json::json!(limit));
    }
    if truncation.truncated {
        // Intentionally omit the content field (already emitted as message body; avoids duplicating up to 50KB of text in details, which would amplify in session JSONL persistence)
        details.insert("truncation".into(), truncation_to_json(&truncation));
    }
    let details = if details.is_empty() {
        None
    } else {
        Some(serde_json::Value::Object(details))
    };

    // Move content only after all borrows of truncation are done
    let mut result_output = truncation.content;
    if !notices.is_empty() {
        result_output.push_str("\n\n[");
        result_output.push_str(&notices.join(". "));
        result_output.push(']');
    }

    LsResultAssembly {
        content: result_output,
        details,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn test_ctx(dir: &std::path::Path) -> ToolContext {
        ToolContext::new(dir.to_path_buf(), None, None)
    }

    /// Create a standard directory tree:
    ///   a.txt B.txt .hidden .gitignore(containing *.txt) sub/nested.log
    /// Note: ls does not honor .gitignore (pi reads directly with fsReaddir, no filtering semantics); all entries are listed.
    fn build_tree(dir: &std::path::Path) {
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        for f in ["a.txt", "B.txt", ".hidden", ".gitignore"] {
            std::fs::write(dir.join(f), "x").unwrap();
        }
        std::fs::write(dir.join("sub/nested.log"), "x").unwrap();
        std::fs::write(dir.join(".gitignore"), "*.txt\n").unwrap();
    }

    // ─── parse_ls_args ────────────────────────────────────────────────

    #[test]
    fn parse_ls_args_empty() {
        let (path, limit) = parse_ls_args(&serde_json::json!({}));
        assert_eq!(path, None);
        assert_eq!(limit, None);
    }

    #[test]
    fn parse_ls_args_both() {
        let (path, limit) = parse_ls_args(&serde_json::json!({ "path": "src", "limit": 50 }));
        assert_eq!(path.as_deref(), Some("src"));
        assert_eq!(limit, Some(50));
    }

    #[test]
    fn parse_ls_args_path_only() {
        let (path, limit) = parse_ls_args(&serde_json::json!({ "path": "src" }));
        assert_eq!(path.as_deref(), Some("src"));
        assert_eq!(limit, None);
    }

    #[test]
    fn parse_ls_args_limit_only() {
        let (path, limit) = parse_ls_args(&serde_json::json!({ "limit": 10 }));
        assert_eq!(path, None);
        assert_eq!(limit, Some(10));
    }

    #[test]
    fn parse_ls_args_non_numeric_limit_treated_as_none() {
        let (path, limit) = parse_ls_args(&serde_json::json!({ "limit": "lots" }));
        assert_eq!(path, None);
        assert_eq!(limit, None);
    }

    // ─── sort_entries_case_insensitive ────────────────────────────────

    #[test]
    fn sort_is_case_insensitive_alphabetical() {
        let mut entries = vec![
            "B.txt".to_string(),
            "a.txt".to_string(),
            "C.txt".to_string(),
            "b.txt".to_string(),
        ];
        sort_entries_case_insensitive(&mut entries);
        assert_eq!(entries, vec!["a.txt", "B.txt", "b.txt", "C.txt"]);
    }

    #[test]
    fn sort_is_stable_for_case_ties() {
        // Same case (equal lowercase) → keep original order (aligned with JS stable sort)
        let mut entries = vec!["a.txt".to_string(), "A.txt".to_string()];
        sort_entries_case_insensitive(&mut entries);
        assert_eq!(entries, vec!["a.txt", "A.txt"]);
    }

    #[test]
    fn sort_handles_dotfiles_and_digits() {
        let mut entries = vec![
            "sub".to_string(),
            ".hidden".to_string(),
            ".gitignore".to_string(),
            "2nd.txt".to_string(),
            "10th.txt".to_string(),
        ];
        sort_entries_case_insensitive(&mut entries);
        // Code point order: '.'(0x2E) < digits < letters; '.gitignore' vs '.hidden' by 'g' < 'h'
        assert_eq!(
            entries,
            vec![".gitignore", ".hidden", "10th.txt", "2nd.txt", "sub",]
        );
    }

    #[test]
    fn sort_already_sorted_unchanged() {
        let mut entries = vec!["a.txt".to_string(), "b.txt".to_string()];
        sort_entries_case_insensitive(&mut entries);
        assert_eq!(entries, vec!["a.txt", "b.txt"]);
    }

    // ─── read_dir_entries (tempdir integration) ──────────────────────────────

    #[test]
    fn read_dir_entries_lists_sorted_with_dir_suffix() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let cancel = CancellationToken::new();
        let scan = read_dir_entries(dir.path(), 500, &cancel).unwrap();
        // Case-insensitive sorting; directory sub has `/` suffix; .gitignore content does not affect listing (ls has no gitignore semantics)
        assert_eq!(
            scan.entries,
            vec![".gitignore", ".hidden", "a.txt", "B.txt", "sub/"]
        );
        assert!(!scan.entry_limit_reached);
    }

    #[test]
    fn read_dir_entries_includes_dotfiles() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let cancel = CancellationToken::new();
        let scan = read_dir_entries(dir.path(), 500, &cancel).unwrap();
        assert!(scan.entries.iter().any(|e| e == ".hidden"));
        assert!(scan.entries.iter().any(|e| e == ".gitignore"));
    }

    #[test]
    fn read_dir_entries_ignores_gitignore_content() {
        // ls differs from find/grep: does not honor .gitignore (pi fsReaddir reads directly); *.txt is still listed
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let cancel = CancellationToken::new();
        let scan = read_dir_entries(dir.path(), 500, &cancel).unwrap();
        assert!(scan.entries.iter().any(|e| e == "a.txt"));
        assert!(scan.entries.iter().any(|e| e == "B.txt"));
    }

    #[test]
    fn read_dir_entries_limit_reached_returns_exactly_limit_entries() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let cancel = CancellationToken::new();
        let scan = read_dir_entries(dir.path(), 3, &cancel).unwrap();
        assert_eq!(scan.entries.len(), 3);
        assert!(scan.entry_limit_reached);
    }

    #[test]
    fn read_dir_entries_limit_zero_returns_empty_with_reached() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let cancel = CancellationToken::new();
        // limit=0: first entry immediately reaches limit (aligned with pi top check `results.length >= effectiveLimit`); returns empty set
        let scan = read_dir_entries(dir.path(), 0, &cancel).unwrap();
        assert!(scan.entries.is_empty());
        assert!(scan.entry_limit_reached);
    }

    #[test]
    fn read_dir_entries_empty_directory_not_reached() {
        let dir = tempdir().unwrap();
        let cancel = CancellationToken::new();
        let scan = read_dir_entries(dir.path(), 500, &cancel).unwrap();
        assert!(scan.entries.is_empty());
        assert!(!scan.entry_limit_reached);
    }

    #[test]
    fn read_dir_entries_nonexistent_directory_errors() {
        let cancel = CancellationToken::new();
        let err =
            read_dir_entries(Path::new("/nonexistent/zen-ls-test"), 500, &cancel).unwrap_err();
        assert!(matches!(err, ToolError::Execution(_)));
        assert!(err.to_string().contains("Cannot read directory"));
    }

    #[test]
    fn read_dir_entries_cancelled_returns_early() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let cancel = CancellationToken::new();
        cancel.cancel();
        // Cancelled: break at the first entry, return an empty result (execute layer rechecks and converts to Cancelled)
        let scan = read_dir_entries(dir.path(), 500, &cancel).unwrap();
        assert!(scan.entries.is_empty());
        assert!(!scan.entry_limit_reached);
    }

    #[cfg(unix)]
    #[test]
    fn read_dir_entries_skips_unstatable_entries() {
        use std::os::unix::fs::symlink;
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("ok.txt"), "x").unwrap();
        // Dangling symlink: metadata fails → skip (aligned with pi `catch { continue }`)
        symlink(dir.path().join("missing-target"), dir.path().join("broken")).unwrap();
        let cancel = CancellationToken::new();
        let scan = read_dir_entries(dir.path(), 500, &cancel).unwrap();
        assert_eq!(scan.entries, vec!["ok.txt"]);
        assert!(!scan.entry_limit_reached);
    }

    // ─── assemble_ls_result ───────────────────────────────────────────

    #[test]
    fn assemble_empty_returns_empty_directory() {
        let assembly = assemble_ls_result(
            LsScan {
                entries: vec![],
                entry_limit_reached: false,
            },
            500,
        );
        assert_eq!(assembly.content, "(empty directory)");
        assert!(assembly.details.is_none());
    }

    #[test]
    fn assemble_empty_with_limit_reached_returns_empty_directory() {
        // limit=0 and directory non-empty: entry_limit_reached is set, but pi's empty-result early branch swallows notices/details
        let assembly = assemble_ls_result(
            LsScan {
                entries: vec![],
                entry_limit_reached: true,
            },
            0,
        );
        assert_eq!(assembly.content, "(empty directory)");
        assert!(assembly.details.is_none());
    }

    #[test]
    fn assemble_within_limit_no_notices_no_details() {
        let assembly = assemble_ls_result(
            LsScan {
                entries: vec!["a.txt".into(), "B.txt".into(), "sub/".into()],
                entry_limit_reached: false,
            },
            500,
        );
        assert_eq!(assembly.content, "a.txt\nB.txt\nsub/");
        assert!(assembly.details.is_none());
    }

    #[test]
    fn assemble_limit_reached_appends_notice_and_details() {
        let assembly = assemble_ls_result(
            LsScan {
                entries: vec!["a.txt".into(), "b.txt".into()],
                entry_limit_reached: true,
            },
            2,
        );
        assert_eq!(
            assembly.content,
            "a.txt\nb.txt\n\n[2 entries limit reached. Use limit=4 for more]"
        );
        let details = assembly.details.unwrap();
        assert_eq!(details["entryLimitReached"], 2);
        assert!(details.get("truncation").is_none());
    }

    #[test]
    fn assemble_truncated_appends_size_notice_and_truncation_details() {
        // Over 50KB → truncation notice + truncation details (6000 entries × ~12B ≈ 72KB > 50KB)
        let long: Vec<String> = (0..6000).map(|i| format!("file_{i}.ts")).collect();
        let assembly = assemble_ls_result(
            LsScan {
                entries: long.clone(),
                entry_limit_reached: false,
            },
            500,
        );
        let truncated_note = format!("[{} limit reached]", format_size(DEFAULT_MAX_BYTES));
        assert!(
            assembly.content.ends_with(&truncated_note),
            "content should end with truncation notice, got: ...{}",
            &assembly.content[assembly.content.len().saturating_sub(120)..]
        );
        let details = assembly.details.expect("truncation details present");
        let trunc = &details["truncation"];
        assert_eq!(trunc["truncated"], true);
        assert_eq!(trunc["truncatedBy"], "bytes");
        assert_eq!(trunc["maxBytes"], DEFAULT_MAX_BYTES);
        assert!(details.get("entryLimitReached").is_none());
    }

    // ─── execute (end-to-end) ──────────────────────────────────────────

    #[tokio::test]
    async fn execute_lists_directory_content() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let tool = LsTool::new(test_ctx(dir.path()));
        let cancel = CancellationToken::new();
        let result = tool
            .execute(
                "call-1",
                &serde_json::json!({ "path": "." }),
                &cancel,
                Box::new(|_| {}),
            )
            .await
            .unwrap();
        assert!(!result.terminate);
        assert!(!result.message.is_error);
        assert_eq!(result.message.tool_name, "ls");
        assert_eq!(result.message.tool_call_id, "call-1");
        let text = &result.message.content[0];
        match text {
            ContentBlock::Text(t) => {
                assert!(t.text.contains("a.txt"), "got: {}", t.text);
                assert!(t.text.contains("sub/"), "got: {}", t.text);
                assert!(t.text.contains(".hidden"), "got: {}", t.text);
            }
            _ => panic!("expected text content"),
        }
        assert!(result.message.details.is_none());
    }

    #[tokio::test]
    async fn execute_empty_directory_returns_empty_directory() {
        let dir = tempdir().unwrap();
        let tool = LsTool::new(test_ctx(dir.path()));
        let cancel = CancellationToken::new();
        let result = tool
            .execute(
                "call-1",
                &serde_json::json!({ "path": "." }),
                &cancel,
                Box::new(|_| {}),
            )
            .await
            .unwrap();
        match &result.message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t.text, "(empty directory)"),
            _ => panic!("expected text content"),
        }
        assert!(result.message.details.is_none());
    }

    #[tokio::test]
    async fn execute_missing_path_errors() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let tool = LsTool::new(test_ctx(dir.path()));
        let cancel = CancellationToken::new();
        let err = tool
            .execute(
                "call-1",
                &serde_json::json!({ "path": "does-not-exist" }),
                &cancel,
                Box::new(|_| {}),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Execution(_)));
        assert!(err.to_string().contains("Path not found"));
    }

    #[tokio::test]
    async fn execute_file_path_is_not_a_directory() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let tool = LsTool::new(test_ctx(dir.path()));
        let cancel = CancellationToken::new();
        let err = tool
            .execute(
                "call-1",
                &serde_json::json!({ "path": "a.txt" }),
                &cancel,
                Box::new(|_| {}),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Execution(_)));
        assert!(err.to_string().contains("Not a directory"));
    }

    #[tokio::test]
    async fn execute_limit_respected() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let tool = LsTool::new(test_ctx(dir.path()));
        let cancel = CancellationToken::new();
        let result = tool
            .execute(
                "call-1",
                &serde_json::json!({ "path": ".", "limit": 2 }),
                &cancel,
                Box::new(|_| {}),
            )
            .await
            .unwrap();
        match &result.message.content[0] {
            ContentBlock::Text(t) => {
                // First 2 entries after sorting + notices section
                assert_eq!(
                    t.text,
                    ".gitignore\n.hidden\n\n[2 entries limit reached. Use limit=4 for more]"
                );
            }
            _ => panic!("expected text content"),
        }
        let details = result.message.details.as_ref().expect("details present");
        assert_eq!(details["entryLimitReached"], 2);
    }

    #[tokio::test]
    async fn execute_cancelled_before_start() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let tool = LsTool::new(test_ctx(dir.path()));
        let cancel = CancellationToken::new();
        cancel.cancel();
        let err = tool
            .execute(
                "call-1",
                &serde_json::json!({ "path": "." }),
                &cancel,
                Box::new(|_| {}),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Cancelled));
    }

    #[tokio::test]
    async fn execute_path_resolved_against_working_dir() {
        // path defaults to '.' → working_dir; relative subdirectory path → resolve relative to working_dir
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let tool = LsTool::new(test_ctx(dir.path()));
        let cancel = CancellationToken::new();
        let result = tool
            .execute(
                "call-1",
                &serde_json::json!({ "path": "sub" }),
                &cancel,
                Box::new(|_| {}),
            )
            .await
            .unwrap();
        match &result.message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t.text, "nested.log"),
            _ => panic!("expected text content"),
        }
    }

    #[test]
    fn tool_metadata_matches_pi() {
        let tool = LsTool::new(ToolContext::new(PathBuf::from("/tmp"), None, None));
        assert_eq!(tool.name(), "ls");
        assert_eq!(tool.label(), "ls");
        assert_eq!(tool.description(), "List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. Includes dotfiles. Output is truncated to 500 entries or 50KB (whichever is hit first).");
        assert_eq!(tool.prompt_snippet(), Some("List directory contents"));
        let params = tool.parameters();
        assert!(
            params.get("required").is_none(),
            "ls schema has no required"
        );
        assert!(params["properties"]["path"]["type"].is_string());
        assert!(params["properties"]["limit"]["type"].is_string());
    }
}
