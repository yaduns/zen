// REQ-APP-04 (pi-349): grep tool.
// Search directories/files for matching lines by regex/literal, return `path:line: text` (supports context lines), respect .gitignore,
// truncated by match count and byte limit.
// Corresponds to original TS: reference/pi-original/packages/coding-agent/src/core/tools/grep.ts.
// Paradigm replacement (see tasks/T-AP-04-5-implemented.md "Ecosystem Reuse Verification Conclusions"):
// - rg subprocess + ensureTool mechanism → `ignore::WalkBuilder` (gitignore-aware traversal) + `regex` crate (line-by-line matching,
//   same BurntSushi engine as rg, nearly identical flavor);
// - `GrepOperations` pluggable interface (SSH remote proxy) dropped in v1 (no remote execution architecture, same as FindOperations);
// - `formatGrepCall`/`formatGrepResult` render functions have been eliminated by T-AP-11-2 unified fallback formatting (v1 has no renderer system),
//   limit/truncation/long-line truncation warnings are embedded in content text as a notices section, so UI text output does not lose them.
// Behavior mapping: pi `--hidden` → `WalkBuilder::hidden(false)` (include hidden files);
// pi `--ignore-case` → `RegexBuilder::case_insensitive`；pi `--fixed-strings` → `regex::escape`；
// pi `--glob` → reuse find `compile_find_glob` (with `/` full path / without `/` basename, same candidates as find);
// Binary detection → `looks_binary` skips via first 8KB NUL probe (rg default heuristic);
// Single-pass "read file→match line by line" replaces pi's two-stage "rg reports positions→reread lines" (rg needs streaming because it is an external process).

use crate::tools::find::compile_find_glob;
use crate::tools::path_utils::resolve_to_cwd;
use crate::truncate::{
    format_size, truncate_head, truncate_line, truncation_to_json, TruncationOptions,
    DEFAULT_MAX_BYTES, GREP_MAX_LINE_LENGTH,
};
use crate::util::now_millis;
use async_trait::async_trait;
use globset::GlobMatcher;
use std::path::Path;
use tokio_util::sync::CancellationToken;
use zen_ai::tool::{Tool, ToolContext, ToolError, ToolExecutionResult, ToolUpdateCallback};
use zen_ai::types::{ContentBlock, TextContent, ToolResultMessage, Usage};

/// Maximum number of grep matches (pi `DEFAULT_LIMIT = 100`).
const DEFAULT_LIMIT: usize = 100;

/// Parsed grep tool parameters (pi `GrepToolInput` resolved; private, not exported).
#[derive(Debug)]
struct GrepArgs {
    /// Regex or literal pattern.
    pattern: String,
    /// Search directory or file (defaults to current working directory).
    path: Option<String>,
    /// Filter files by glob.
    glob: Option<String>,
    /// Case-insensitive (pi `--ignore-case`).
    ignore_case: bool,
    /// Treat pattern as literal (pi `--fixed-strings`).
    literal: bool,
    /// Number of lines to show before and after each match (negatives clamped to 0).
    context: usize,
    /// Maximum number of matches (default None; `effective_limit = max(1, limit ?? 100)` expanded by execute).
    limit: Option<usize>,
}

/// Compiled glob file filter (private, not exported). `full_path` determines candidate string:
/// Patterns containing `/` match the file's absolute posix path; patterns without `/` match the file name (same decision as find `compile_find_glob`).
#[derive(Debug)]
struct GlobFilter {
    matcher: GlobMatcher,
    full_path: bool,
}

/// Raw output of `grep_files` single-pass scan (private, not exported).
#[derive(Debug)]
struct GrepScan {
    /// Formatted output lines (before byte truncation).
    output_lines: Vec<String>,
    /// Actual number of matches (≤ limit).
    match_count: usize,
    /// Whether limit was reached (collected full limit or subsequent matches were stopped, aligned with pi `matchCount >= effectiveLimit`).
    limit_reached: bool,
    /// Whether any match/context line was truncated by `truncate_line`.
    lines_truncated: bool,
}

/// Final output assembled by `assemble_grep_result` (private, not exported).
struct GrepResultAssembly {
    /// Final content (including appended notices section; empty result is pi's "No matches found").
    content: String,
    /// details metadata (`matchLimitReached`/`truncation`/`linesTruncated`, pi `GrepToolDetails` resolved;
    /// `None` when all are absent).
    details: Option<serde_json::Value>,
}

/// File content search tool; the LLM triggers it when the agent needs to "search files for text by regex/literal (respecting .gitignore)" —
/// User-visible effect: returns matching lines as `path:line: text` (when `context>0`, appends context lines prefixed with `path-line- `),
/// When over limit (default 100 matches / 50KB, whichever comes first), body appends a `[..matches limit reached..]`-style notice; long lines truncated to
/// 500 characters, aligned with pi grep tool behavior.
/// Parameter Schema (JSON Schema, static definition isomorphic to the read/find tools):
/// - `pattern` (string, required): regex or literal pattern
/// - `path` (string, optional): search directory or file, defaults to current working directory
/// - `glob` (string, optional): filter files by glob, e.g. `'*.ts'`, `'**/*.spec.ts'`
/// - `ignoreCase` (boolean, optional): case-insensitive, default false
/// - `literal` (boolean, optional): treat pattern as literal instead of regex, default false
/// - `context` (number, optional): lines to show before and after each match, default 0
/// - `limit` (number, optional): maximum number of matches, default 100
///
/// Search kernel uses `ignore::WalkBuilder` (gitignore-aware, includes hidden files, does not follow symlinks) to scan file by file, line by line
/// `regex::is_match` replaces pi's rg subprocess (see tasks/T-AP-04-5-implemented.md "Ecosystem Reuse Verification Conclusions").
#[derive(Debug)]
pub struct GrepTool {
    tool_context: ToolContext,
}

impl GrepTool {
    /// Construct a grep tool instance from the session environment (called by factory `create(ToolName::Grep)`; same structure as FindTool::new).
    pub fn new(tool_context: ToolContext) -> Self {
        Self { tool_context }
    }
}

#[async_trait]
impl Tool for GrepTool {
    /// Tool name `"grep"` (aligned with pi tool name and `ToolName::Grep`).
    fn name(&self) -> &str {
        "grep"
    }

    /// LLM-visible description: search file contents, return matching lines with path and line number, respect .gitignore, output truncated to
    /// `DEFAULT_LIMIT` matches or `DEFAULT_MAX_BYTES/1024` KB (whichever comes first), long lines truncated to
    /// `GREP_MAX_LINE_LENGTH` characters—wording matches pi `createGrepToolDefinition.description` verbatim.
    fn description(&self) -> &str {
        "Search file contents for a pattern. Returns matching lines with file paths and line \
            numbers. Respects .gitignore. Output is truncated to 100 matches or 50KB (whichever \
            is hit first). Long lines are truncated to 500 chars."
    }

    /// Parameter JSON Schema (pattern required; path/glob/ignoreCase/literal/context/limit optional),
    /// OnceLock static cache, aligned with read/find tools.
    fn parameters(&self) -> &serde_json::Value {
        static PARAMS: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
        PARAMS.get_or_init(|| {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern (regex or literal string)"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search (default: current directory)"
                    },
                    "glob": {
                        "type": "string",
                        "description": "Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'"
                    },
                    "ignoreCase": {
                        "type": "boolean",
                        "description": "Case-insensitive search (default: false)"
                    },
                    "literal": {
                        "type": "boolean",
                        "description": "Treat pattern as literal string instead of regex (default: false)"
                    },
                    "context": {
                        "type": "number",
                        "description": "Number of lines to show before and after each match (default: 0)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of matches to return (default: 100)"
                    }
                },
                "required": ["pattern"]
            })
        })
    }

    /// Label `"grep"`.
    fn label(&self) -> &str {
        "grep"
    }

    /// System prompt summary "Search file contents for patterns (respects .gitignore)" (aligned with pi promptSnippet).
    fn prompt_snippet(&self) -> Option<&str> {
        Some("Search file contents for patterns (respects .gitignore)")
    }

    /// Execute grep: parse parameters → compute `effective_limit = max(1, limit ?? 100)` / `context = max(0, context)`
    /// → `resolve_to_cwd` resolves search path (default `.`) → existence precheck → compile regex (`RegexBuilder::case_insensitive`
    /// / `regex::escape`) and glob (reuse find `compile_find_glob`) → run synchronous search kernel inside `spawn_blocking`
    /// (`grep_files`) → `tokio::select!` race cancellation → cancellation recheck → `assemble_grep_result` assembly →
    /// `ToolResultMessage`。
    /// Cancellation semantics: cancel already cancelled or cancelled mid-scan → `ToolError::Cancelled` (when cancelled mid-scan, `grep_files`
    /// breaks early to release the blocking thread; result unused); `is_error=false`; details contains `matchLimitReached`/`truncation`/
    /// `linesTruncated` (pi `GrepToolDetails` mapping; `None` when all absent).
    /// Boundary: search path does not exist → `ToolError::Execution` ("Path not found: ..."); invalid regex/empty pattern →
    /// `ToolError::InvalidArguments` (aligns with rg rejecting empty pattern/bad regex); invalid glob → `ToolError::InvalidArguments`.
    async fn execute(
        &self,
        tool_call_id: &str,
        params: &serde_json::Value,
        cancel: &CancellationToken,
        _on_update: ToolUpdateCallback,
    ) -> Result<ToolExecutionResult, ToolError> {
        let GrepArgs {
            pattern,
            path,
            glob,
            ignore_case,
            literal,
            context,
            limit,
        } = parse_grep_args(params)?;
        let effective_limit = limit.unwrap_or(DEFAULT_LIMIT).max(1);

        let search_path = resolve_to_cwd(
            path.as_deref().unwrap_or("."),
            self.tool_context.working_dir.as_path(),
        )
        .map_err(|e| ToolError::Execution(format!("Failed to resolve grep path: {e}")))?;

        // Search path existence precheck (pi throws via `ops.isDirectory` fsStat; zen uses a unified precheck)
        if !search_path.try_exists().unwrap_or(false) {
            return Err(ToolError::Execution(format!(
                "Path not found: {}",
                search_path.display()
            )));
        }

        let pattern = compile_grep_pattern(&pattern, ignore_case, literal)?;
        let glob = compile_grep_glob(glob.as_deref())?;
        let is_dir = search_path.is_dir();

        let cancel_blocking = cancel.clone();
        let handle = tokio::task::spawn_blocking(move || {
            grep_files(
                &search_path,
                is_dir,
                &pattern,
                glob.as_ref(),
                context,
                effective_limit,
                &cancel_blocking,
            )
        });

        let joined = tokio::select! {
            _ = cancel.cancelled() => {
                // Don't abort: blocking tasks carry their own cancel check; the next entry breaks and self-terminates quickly
                return Err(ToolError::Cancelled);
            }
            joined = handle => joined,
        };
        let scan = match joined {
            Ok(scan) => scan,
            Err(join_err) => {
                return Err(ToolError::Execution(format!(
                    "grep task failed: {join_err}"
                )));
            }
        };

        // Cancellation recheck: `grep_files` may return normally at the same time the cancellation signal arrives (early break); in that case the result is discarded
        if cancel.is_cancelled() {
            return Err(ToolError::Cancelled);
        }

        let assembly = assemble_grep_result(scan, effective_limit);

        let now = now_millis();
        let message = ToolResultMessage {
            tool_call_id: tool_call_id.to_string(),
            tool_name: "grep".to_string(),
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

/// Extract grep arguments from tool argument JSON. Missing/non-string pattern → `ToolError::InvalidArguments`;
/// context defaults to 0; negative/non-integer values collapse to 0 (`as_u64` returns None for negatives); limit defaults to None
/// (`effective_limit = max(1, limit ?? 100)` expanded by execute); ignoreCase/literal default to false.
fn parse_grep_args(params: &serde_json::Value) -> Result<GrepArgs, ToolError> {
    let pattern = params
        .get("pattern")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ToolError::InvalidArguments("Missing required 'pattern' parameter".into()))?
        .to_string();
    let path = params
        .get("path")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let glob = params
        .get("glob")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let ignore_case = params
        .get("ignoreCase")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let literal = params
        .get("literal")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let context = params
        .get("context")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(0);
    let limit = params
        .get("limit")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    Ok(GrepArgs {
        pattern,
        path,
        glob,
        ignore_case,
        literal,
        context,
        limit,
    })
}

/// Compile user pattern into per-line matcher: when `literal=true`, first `regex::escape(pattern)` (pi `--fixed-strings`
/// semantics); when `ignore_case=true`, `RegexBuilder::case_insensitive(true)`; regex still uses default Unicode semantics
/// (consistent with rg's same underlying engine). Empty pattern or compile failure → `ToolError::InvalidArguments` (aligns with rg rejecting empty/bad regex).
fn compile_grep_pattern(
    pattern: &str,
    ignore_case: bool,
    literal: bool,
) -> Result<regex::Regex, ToolError> {
    if pattern.is_empty() {
        return Err(ToolError::InvalidArguments(
            "Empty pattern is not allowed".into(),
        ));
    }
    let effective = if literal {
        regex::escape(pattern)
    } else {
        pattern.to_string()
    };
    regex::RegexBuilder::new(&effective)
        .case_insensitive(ignore_case)
        .build()
        .map_err(|e| ToolError::InvalidArguments(format!("Invalid regex pattern '{pattern}': {e}")))
}

/// Compile `glob` argument into file filter matcher: `None` → `Ok(None)` (no filtering); `Some` → reuse
/// `find::compile_find_glob` to get matcher (`literal_separator(true)`; semantics: contains `/` = full path, no `/` = basename,
/// same as find—when it contains `/`, candidate is file's absolute posix path; when it does not, candidate is file name).
/// Invalid glob → `ToolError::InvalidArguments`.
fn compile_grep_glob(glob: Option<&str>) -> Result<Option<GlobFilter>, ToolError> {
    match glob {
        None => Ok(None),
        Some(g) => {
            let (matcher, full_path) = compile_find_glob(g)?;
            Ok(Some(GlobFilter { matcher, full_path }))
        }
    }
}

/// Convert matched file path to output prefix: search root is directory → posix path relative to search root (strip prefix + posix-ify, no trailing
/// `/`); search root is single file → file name (pi `formatPath`'s `isDirectory ? relative : basename` branch).
/// Non-UTF-8 file name uses `to_string_lossy` (rg output is likewise lossy).
fn format_grep_path(file_path: &Path, search_path: &Path, is_dir: bool) -> String {
    if is_dir {
        let rel = file_path
            .strip_prefix(search_path)
            .unwrap_or(file_path)
            .to_string_lossy()
            .to_string();
        // Fallback branch (when strip_prefix fails): remove the leading separator from the absolute path
        let rel = rel
            .trim_start_matches(std::path::MAIN_SEPARATOR)
            .to_string();
        rel.replace(std::path::MAIN_SEPARATOR, "/")
    } else {
        file_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default()
    }
}

/// Format a single output line: match line `"{path}:{line_number}: {text}"`, context line `"{path}-{line_number}- {text}"`
/// (pi dual format, `-` separator distinguishes match/context). Line text has already had `\r` and trailing `\n` removed by upper layer (single-pass line reading naturally satisfies
/// pi's `replace(/\r/g,"")` + `replace(/\n$/,"")`). Returns formatted line (long-line truncation is done in `grep_files` layer,
/// this function does not truncate).
fn format_grep_line(
    path: &str,
    line_number: usize,
    line_text: &str,
    is_match_line: bool,
) -> String {
    if is_match_line {
        format!("{path}:{line_number}: {line_text}")
    } else {
        format!("{path}-{line_number}- {line_text}")
    }
}

/// Binary file detection, called by `grep_files` when agent searches directories containing binary files—user-visible effect: rg skips by default
/// binary content (NUL heuristic); zen single-pass scan also skips it, preventing lossy-decoded binary noise from mixing into match results.
/// Detection rule: check only first 8KB (all if shorter); containing a NUL byte (`0x00`) means binary (aligns with rg's primary heuristic);
/// Plain text/UTF-8 files normally return false. Takes byte slice input; caller invokes before `from_utf8_lossy` decoding.
fn looks_binary(content: &[u8]) -> bool {
    let probe = &content[..content.len().min(8 * 1024)];
    probe.contains(&0x00)
}

/// Traverse the search root (`ignore::WalkBuilder`: `hidden(false)` includes hidden entries; `require_git(false)` is respected everywhere)
/// .gitignore; by default does not follow symlinks; when search root is a single file, traverse that file directly—ignore `Walk` for a non-directory root
/// produces one entry as-is, verified in ignore 0.4.32 `run_one`), for each file: first pass glob filter
/// (skip if `glob` matcher does not match, aligning with pi via rg `--glob` searching only allowed files), then read raw bytes and
/// `looks_binary` detection (skip on hit, aligning with rg's default binary detection) → `from_utf8_lossy` decode
/// → normalize CRLF (`\r\n`/`\r` → `\n`, aligning with pi `getFileLines` normalization) → split by `\n` → line by line
/// `regex.is_match`; for matched lines, take [line-context, line+context] closed interval context window per `context`, line by line
/// `truncate_line(line, GREP_MAX_LINE_LENGTH)` (set `lines_truncated=true` if truncated, aligning with pi for each
/// match/context line truncation) and `format_grep_line` into `output_lines`; stop once match count reaches `limit`
/// (`limit_reached = match_count >= limit`, aligning with pi `matchCount >= effectiveLimit` stopping child),
/// subsequent files/lines are not scanned. Each entry checks `cancel.is_cancelled()`; if cancelled, break early and return partial results
/// (converted to `ToolError::Cancelled` by execute-layer cancellation recheck). **Synchronous with no internal error source** (parsing/compilation already lifted to execute,
/// single-entry read failure is skipped without interruption—consistent with find's `single-entry traversal failure skip`; pi's “rg reports match then re-read fails outputs
/// `(unable to read file)` marker” does not occur under single-pass model; files that fail to read simply produce no matches, see “Adjudicated Decisions”),
/// therefore returns no `Result` (different shape from `find_files`' `Result` intentionally). Wrapped in `spawn_blocking` inside execute.
fn grep_files(
    search_path: &Path,
    is_dir: bool,
    pattern: &regex::Regex,
    glob: Option<&GlobFilter>,
    context: usize,
    limit: usize,
    cancel: &CancellationToken,
) -> GrepScan {
    let walker = ignore::WalkBuilder::new(search_path)
        .hidden(false) // Include hidden files (aligns with rg --hidden)
        .require_git(false) // Respect .gitignore everywhere (consistent with find, see “Adjudicated Decisions”)
        .build();

    let mut output_lines: Vec<String> = Vec::new();
    let mut match_count = 0usize;
    let mut limit_reached = false;
    let mut lines_truncated = false;

    for entry in walker {
        if cancel.is_cancelled() {
            break;
        }
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue, // Skip single-entry traversal failure (permissions, etc.) without interruption, aligning with find/skills/load.rs
        };
        // Search files only: skip directory entries (including search root itself)
        if entry.file_type().is_none_or(|ft| ft.is_dir()) {
            continue;
        }
        let entry_path = entry.path();

        // glob file filtering (reuses find `compile_find_glob` semantics: contains `/` matches absolute posix path; otherwise matches file name)
        if let Some(filter) = glob {
            let matched = if filter.full_path {
                filter
                    .matcher
                    .is_match(to_posix_string(&entry_path.to_string_lossy()))
            } else {
                filter
                    .matcher
                    .is_match(entry.file_name().to_string_lossy().as_ref())
            };
            if !matched {
                continue;
            }
        }

        // Read file → binary detection → lossy decode
        let bytes = match std::fs::read(entry_path) {
            Ok(b) => b,
            Err(_) => continue, // Skip on read failure (adjudicated: no match; pi's unable-to-read marker does not apply in single-pass model)
        };
        if looks_binary(&bytes) {
            continue;
        }
        let content = String::from_utf8_lossy(&bytes);
        // Normalize CRLF (aligning with pi `getFileLines`: `\r\n`/`\r` → `\n`)
        let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
        let lines: Vec<&str> = normalized.split('\n').collect();
        let path_prefix = format_grep_path(entry_path, search_path, is_dir);

        for (idx, line) in lines.iter().enumerate() {
            if !pattern.is_match(line) {
                continue;
            }
            // Check limit first: after collecting full limit, stop at next match (aligning with pi `matchCount >= effectiveLimit` stopping child)
            if match_count >= limit {
                limit_reached = true;
                return GrepScan {
                    output_lines,
                    match_count,
                    limit_reached,
                    lines_truncated,
                };
            }
            match_count += 1;
            let line_number = idx + 1;

            if context > 0 {
                let start = line_number.saturating_sub(context).max(1);
                let end = (line_number + context).min(lines.len());
                for cur in start..=end {
                    let is_match_line = cur == line_number;
                    let truncated = truncate_line(lines[cur - 1], GREP_MAX_LINE_LENGTH);
                    if truncated.was_truncated {
                        lines_truncated = true;
                    }
                    output_lines.push(format_grep_line(
                        &path_prefix,
                        cur,
                        &truncated.text,
                        is_match_line,
                    ));
                }
            } else {
                let truncated = truncate_line(line, GREP_MAX_LINE_LENGTH);
                if truncated.was_truncated {
                    lines_truncated = true;
                }
                output_lines.push(format_grep_line(
                    &path_prefix,
                    line_number,
                    &truncated.text,
                    true,
                ));
            }
        }
    }

    // Reaching exactly the limit (no more entries to traverse) also counts as limit reached (aligning with pi `matchCount >= effectiveLimit`)
    limit_reached = limit_reached || match_count >= limit;

    GrepScan {
        output_lines,
        match_count,
        limit_reached,
        lines_truncated,
    }
}

/// Assemble scan artifacts into final output: `output_lines.join("\n")` → `truncate_head` (`max_lines: usize::MAX`,
/// only the `DEFAULT_MAX_BYTES` byte limit applies, aligned with the equivalent pi call) → append notices (a `\n\n[..]` section, verbatim aligned
/// pi：`matchLimitReached` → "{limit} matches limit reached. Use limit={limit*2} for more, or refine
/// pattern"；`truncation.truncated` → "{format_size(DEFAULT_MAX_BYTES)} limit reached"；
/// `linesTruncated` → "Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. Use read tool to see
/// full lines"; multiple entries joined with ". ") → details JSON (`matchLimitReached`/`linesTruncated`/`truncation`
/// each omitted from details when absent; `truncation` uses shared `crate::truncate::truncation_to_json`, intentionally omitting
/// content to prevent JSONL double storage, same as find decision). **Empty result (match_count==0)** → content is pi's
/// "No matches found", details=None (aligning with pi `matchCount === 0` early branch).
fn assemble_grep_result(scan: GrepScan, limit: usize) -> GrepResultAssembly {
    if scan.match_count == 0 {
        return GrepResultAssembly {
            content: "No matches found".to_string(),
            details: None,
        };
    }

    let raw_output = scan.output_lines.join("\n");
    let truncation = truncate_head(
        &raw_output,
        TruncationOptions {
            max_lines: usize::MAX,
            max_bytes: DEFAULT_MAX_BYTES,
        },
    );

    let mut notices: Vec<String> = Vec::new();
    if scan.limit_reached {
        notices.push(format!(
            "{limit} matches limit reached. Use limit={} for more, or refine pattern",
            limit * 2
        ));
    }
    if truncation.truncated {
        notices.push(format!("{} limit reached", format_size(DEFAULT_MAX_BYTES)));
    }
    if scan.lines_truncated {
        notices.push(format!(
            "Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. Use read tool to see full lines"
        ));
    }

    let mut details = serde_json::Map::new();
    if scan.limit_reached {
        details.insert("matchLimitReached".into(), serde_json::json!(limit));
    }
    if truncation.truncated {
        // Intentionally omit the content field (already emitted as message body; avoids duplicating up to 50KB of text in details, which would amplify in session JSONL persistence)
        details.insert("truncation".into(), truncation_to_json(&truncation));
    }
    if scan.lines_truncated {
        details.insert("linesTruncated".into(), serde_json::json!(true));
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

    GrepResultAssembly {
        content: result_output,
        details,
    }
}

/// Convert the path to POSIX style (replace only the platform separator; on Unix, `MAIN_SEPARATOR == '/'` makes this a no-op,
/// aligning it with pi `toPosixPath = value.split(path.sep).join("/")`).
fn to_posix_string(s: &str) -> String {
    s.replace(std::path::MAIN_SEPARATOR, "/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn test_ctx(dir: &std::path::Path) -> ToolContext {
        ToolContext::new(dir.to_path_buf(), None, None)
    }

    /// Build a standard search tree (case-sensitive content, with .gitignore and binary files):
    ///   a.txt("hello world") b.txt("foo bar") .hidden.txt("hello hidden")
    ///   multi.txt("line one\nhello world\nline three") c.log("hello log"→ignored by gitignore)
    ///   sub/d.txt("HELLO there") sub/deep.txt("goodbye") sub/binary.dat("hello\x00binary")
    ///   .gitignore("*.log")
    fn build_tree(dir: &std::path::Path) {
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("a.txt"), "hello world").unwrap();
        std::fs::write(dir.join("b.txt"), "foo bar").unwrap();
        std::fs::write(dir.join(".hidden.txt"), "hello hidden").unwrap();
        std::fs::write(dir.join("multi.txt"), "line one\nhello world\nline three").unwrap();
        std::fs::write(dir.join("c.log"), "hello log").unwrap();
        std::fs::write(dir.join(".gitignore"), "*.log\n").unwrap();
        std::fs::write(dir.join("sub/d.txt"), "HELLO there").unwrap();
        std::fs::write(dir.join("sub/deep.txt"), "goodbye").unwrap();
        std::fs::write(dir.join("sub/binary.dat"), b"hello\x00binary").unwrap();
    }

    /// Return scan artifact output lines sorted (walk order across files is nondeterministic; assertions do not depend on order).
    fn sorted_output(scan: &GrepScan) -> Vec<String> {
        let mut v = scan.output_lines.clone();
        v.sort_unstable();
        v
    }

    // ─── parse_grep_args ────────────────────────────────────────────────

    #[test]
    fn parse_grep_args_pattern_only_defaults() {
        let args = parse_grep_args(&serde_json::json!({ "pattern": "hello" })).unwrap();
        assert_eq!(args.pattern, "hello");
        assert_eq!(args.path, None);
        assert_eq!(args.glob, None);
        assert!(!args.ignore_case);
        assert!(!args.literal);
        assert_eq!(args.context, 0);
        assert_eq!(args.limit, None);
    }

    #[test]
    fn parse_grep_args_full() {
        let args = parse_grep_args(&serde_json::json!({
            "pattern": "HELLO", "path": "src", "glob": "*.rs",
            "ignoreCase": true, "literal": true, "context": 2, "limit": 50
        }))
        .unwrap();
        assert_eq!(args.pattern, "HELLO");
        assert_eq!(args.path.as_deref(), Some("src"));
        assert_eq!(args.glob.as_deref(), Some("*.rs"));
        assert!(args.ignore_case);
        assert!(args.literal);
        assert_eq!(args.context, 2);
        assert_eq!(args.limit, Some(50));
    }

    #[test]
    fn parse_grep_args_missing_pattern_is_invalid() {
        let err = parse_grep_args(&serde_json::json!({ "path": "src" })).unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn parse_grep_args_pattern_not_string_is_invalid() {
        let err = parse_grep_args(&serde_json::json!({ "pattern": 42 })).unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn parse_grep_args_non_numeric_limit_and_context_fall_back() {
        // limit non-numeric → None; context non-numeric → 0 (aligned with pi `??` semantics: invalid value falls back to default)
        let args = parse_grep_args(&serde_json::json!({
            "pattern": "hello", "limit": "lots", "context": "wide"
        }))
        .unwrap();
        assert_eq!(args.limit, None);
        assert_eq!(args.context, 0);
    }

    #[test]
    fn parse_grep_args_negative_context_collapses_to_zero() {
        // pi `context && context > 0 ? context : 0`: negative numbers converge to 0 (as_u64 returns None for negative numbers)
        let args = parse_grep_args(&serde_json::json!({
            "pattern": "hello", "context": -2
        }))
        .unwrap();
        assert_eq!(args.context, 0);
    }

    #[test]
    fn parse_grep_args_non_boolean_flags_default_false() {
        let args = parse_grep_args(&serde_json::json!({
            "pattern": "hello", "ignoreCase": "yes", "literal": 1
        }))
        .unwrap();
        assert!(!args.ignore_case);
        assert!(!args.literal);
    }

    // ─── compile_grep_pattern ───────────────────────────────────────────

    #[test]
    fn compile_grep_pattern_simple_regex() {
        let re = compile_grep_pattern("he.llo", false, false).unwrap();
        assert!(re.is_match("heXllo"));
        assert!(!re.is_match("hexxllo"));
    }

    #[test]
    fn compile_grep_pattern_literal_escapes_meta() {
        // when literal=true, `.` is escaped (aligned with pi --fixed-strings)
        let re = compile_grep_pattern("he.llo", false, true).unwrap();
        assert!(re.is_match("he.llo"));
        assert!(!re.is_match("hello"));
    }

    #[test]
    fn compile_grep_pattern_ignore_case() {
        let re = compile_grep_pattern("hello", true, false).unwrap();
        assert!(re.is_match("HELLO"));
        assert!(re.is_match("Hello"));
    }

    #[test]
    fn compile_grep_pattern_invalid_regex_is_invalid_argument() {
        let err = compile_grep_pattern("(", false, false).unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn compile_grep_pattern_empty_pattern_is_invalid_argument() {
        // aligned with rg: empty pattern reports an error (rg "regex parse error: empty pattern")
        let err = compile_grep_pattern("", false, false).unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
        let err = compile_grep_pattern("", false, true).unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    // ─── compile_grep_glob ──────────────────────────────────────────────

    #[test]
    fn compile_grep_glob_none_yields_none() {
        assert!(compile_grep_glob(None).unwrap().is_none());
    }

    #[test]
    fn compile_grep_glob_basename_matches_file_name() {
        let filter = compile_grep_glob(Some("*.txt")).unwrap().unwrap();
        assert!(!filter.full_path);
        assert!(filter.matcher.is_match("a.txt"));
        assert!(!filter.matcher.is_match("a.txt.bak"));
        // basename pattern does not cross `/` (literal_separator)
        assert!(!filter.matcher.is_match("sub/a.txt"));
    }

    #[test]
    fn compile_grep_glob_full_path_matches_absolute_path() {
        let filter = compile_grep_glob(Some("sub/*")).unwrap().unwrap();
        assert!(filter.full_path);
        assert!(filter.matcher.is_match("/work/proj/sub/deep.txt"));
        assert!(!filter.matcher.is_match("/work/proj/a.txt"));
    }

    #[test]
    fn compile_grep_glob_invalid_is_invalid_argument() {
        let err = compile_grep_glob(Some("[")).unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    // ─── format_grep_path ───────────────────────────────────────────────

    #[test]
    fn format_grep_path_dir_root_relativizes() {
        let root = Path::new("/tmp/proj");
        assert_eq!(
            format_grep_path(Path::new("/tmp/proj/src/a.ts"), root, true),
            "src/a.ts"
        );
        assert_eq!(
            format_grep_path(Path::new("/tmp/proj/a.ts"), root, true),
            "a.ts"
        );
    }

    #[test]
    fn format_grep_path_single_file_root_uses_basename() {
        // search root is a single file (is_dir=false) → filename (pi `formatPath` basename branch)
        let root = Path::new("/tmp/proj/a.ts");
        assert_eq!(
            format_grep_path(Path::new("/tmp/proj/a.ts"), root, false),
            "a.ts"
        );
    }

    // ─── format_grep_line ───────────────────────────────────────────────

    #[test]
    fn format_grep_line_match_uses_colon() {
        assert_eq!(
            format_grep_line("a.txt", 3, "hello world", true),
            "a.txt:3: hello world"
        );
    }

    #[test]
    fn format_grep_line_context_uses_dash() {
        assert_eq!(
            format_grep_line("a.txt", 2, "line one", false),
            "a.txt-2- line one"
        );
    }

    // ─── looks_binary ───────────────────────────────────────────────────

    #[test]
    fn looks_binary_plain_text_is_false() {
        assert!(!looks_binary(b"hello world\nplain text\n"));
    }

    #[test]
    fn looks_binary_nul_byte_is_true() {
        assert!(looks_binary(b"hello\x00binary"));
    }

    #[test]
    fn looks_binary_only_probes_first_8kb() {
        // NUL at 9KB does not trigger (aligned with rg's first 8KB detection window)
        let mut content = vec![b'a'; 8 * 1024 + 10];
        content[8 * 1024 + 5] = 0x00;
        assert!(!looks_binary(&content));
    }

    #[test]
    fn looks_binary_empty_is_false() {
        assert!(!looks_binary(b""));
    }

    // ─── assemble_grep_result ───────────────────────────────────────────

    #[test]
    fn assemble_empty_returns_no_matches_found() {
        let assembly = assemble_grep_result(
            GrepScan {
                output_lines: vec![],
                match_count: 0,
                limit_reached: false,
                lines_truncated: false,
            },
            100,
        );
        assert_eq!(assembly.content, "No matches found");
        assert!(assembly.details.is_none());
    }

    #[test]
    fn assemble_within_limit_no_notices_no_details() {
        let assembly = assemble_grep_result(
            GrepScan {
                output_lines: vec!["a.rs:1: x".into(), "b.rs:2: y".into()],
                match_count: 2,
                limit_reached: false,
                lines_truncated: false,
            },
            100,
        );
        assert_eq!(assembly.content, "a.rs:1: x\nb.rs:2: y");
        assert!(assembly.details.is_none());
    }

    #[test]
    fn assemble_limit_reached_appends_notice_and_details() {
        let assembly = assemble_grep_result(
            GrepScan {
                output_lines: vec!["a.rs:1: x".into()],
                match_count: 1,
                limit_reached: true,
                lines_truncated: false,
            },
            1,
        );
        assert_eq!(
            assembly.content,
            "a.rs:1: x\n\n[1 matches limit reached. Use limit=2 for more, or refine pattern]"
        );
        let details = assembly.details.unwrap();
        assert_eq!(details["matchLimitReached"], 1);
        assert!(details.get("linesTruncated").is_none());
        assert!(details.get("truncation").is_none());
    }

    #[test]
    fn assemble_lines_truncated_appends_notice_and_details() {
        let assembly = assemble_grep_result(
            GrepScan {
                output_lines: vec!["a.rs:1: x".into()],
                match_count: 1,
                limit_reached: false,
                lines_truncated: true,
            },
            100,
        );
        assert_eq!(
            assembly.content,
            "a.rs:1: x\n\n[Some lines truncated to 500 chars. Use read tool to see full lines]"
        );
        let details = assembly.details.unwrap();
        assert_eq!(details["linesTruncated"], true);
        assert!(details.get("matchLimitReached").is_none());
    }

    #[test]
    fn assemble_truncated_appends_size_notice_and_truncation_details() {
        // over 50KB → truncation notice + truncation details (4000 lines × ~15B ≈ 60KB > 50KB)
        let long: Vec<String> = (0..4000)
            .map(|i| format!("file_{i}.rs:{i}: xxxxxxxxxxxxxx"))
            .collect();
        let assembly = assemble_grep_result(
            GrepScan {
                output_lines: long.clone(),
                match_count: 4000,
                limit_reached: false,
                lines_truncated: false,
            },
            100,
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
        assert!(details.get("matchLimitReached").is_none());
        assert!(details.get("linesTruncated").is_none());
    }

    // ─── grep_files (tempdir integration) ──────────────────────────────────────

    #[test]
    fn grep_files_basic_match() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let re = compile_grep_pattern("hello", false, false).unwrap();
        let cancel = CancellationToken::new();
        let scan = grep_files(dir.path(), true, &re, None, 0, 100, &cancel);
        // .hidden included; c.log excluded by .gitignore; binary.dat skipped; sub/d.txt case mismatch
        assert_eq!(
            sorted_output(&scan),
            vec![
                ".hidden.txt:1: hello hidden",
                "a.txt:1: hello world",
                "multi.txt:2: hello world",
            ]
        );
        assert_eq!(scan.match_count, 3);
        assert!(!scan.limit_reached);
        assert!(!scan.lines_truncated);
    }

    #[test]
    fn grep_files_ignore_case() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let re = compile_grep_pattern("HELLO", true, false).unwrap();
        let cancel = CancellationToken::new();
        let scan = grep_files(dir.path(), true, &re, None, 0, 100, &cancel);
        assert_eq!(
            sorted_output(&scan),
            vec![
                ".hidden.txt:1: hello hidden",
                "a.txt:1: hello world",
                "multi.txt:2: hello world",
                "sub/d.txt:1: HELLO there",
            ]
        );
    }

    #[test]
    fn grep_files_literal_matches_literal_text_not_regex() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let cancel = CancellationToken::new();
        // literal=true: `..` is literal, `he..o` does not match `hello`
        let re = compile_grep_pattern("he..o", false, true).unwrap();
        let scan = grep_files(dir.path(), true, &re, None, 0, 100, &cancel);
        assert_eq!(scan.match_count, 0);
        assert!(scan.output_lines.is_empty());
        // same pattern as regex: `..` matches any two characters → hits hello
        let re = compile_grep_pattern("he..o", false, false).unwrap();
        let scan = grep_files(dir.path(), true, &re, None, 0, 100, &cancel);
        assert_eq!(scan.match_count, 3);
    }

    #[test]
    fn grep_files_respects_gitignore() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let re = compile_grep_pattern("hello", false, false).unwrap();
        let cancel = CancellationToken::new();
        let scan = grep_files(dir.path(), true, &re, None, 0, 100, &cancel);
        // c.log contains "hello log" but is ignored by *.log (require_git(false): also takes effect in non-git repos)
        assert!(
            scan.output_lines.iter().all(|l| !l.starts_with("c.log")),
            "c.log should be ignored, got {:?}",
            scan.output_lines
        );
    }

    #[test]
    fn grep_files_glob_filters_files() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let re = compile_grep_pattern("hello", false, false).unwrap();
        let cancel = CancellationToken::new();
        // basename glob: only searches multi.txt
        let glob = compile_grep_glob(Some("multi.txt")).unwrap();
        let scan = grep_files(dir.path(), true, &re, glob.as_ref(), 0, 100, &cancel);
        assert_eq!(sorted_output(&scan), vec!["multi.txt:2: hello world"]);

        // full-path glob: only searches files under sub
        let glob = compile_grep_glob(Some("sub/*")).unwrap();
        let re2 = compile_grep_pattern("goodbye", false, false).unwrap();
        let scan = grep_files(dir.path(), true, &re2, glob.as_ref(), 0, 100, &cancel);
        assert_eq!(sorted_output(&scan), vec!["sub/deep.txt:1: goodbye"]);
    }

    #[test]
    fn grep_files_context_blocks() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let re = compile_grep_pattern("world", false, false).unwrap();
        let cancel = CancellationToken::new();
        let scan = grep_files(dir.path(), true, &re, None, 1, 100, &cancel);
        // a.txt single line has no context lines (window clamp); multi.txt matches line 2 → 1 line before and after
        // note: when sorting, `-`(0x2D) comes before `:`(0x3A), so context lines sort before matching lines
        assert_eq!(
            sorted_output(&scan),
            vec![
                "a.txt:1: hello world",
                "multi.txt-1- line one",
                "multi.txt-3- line three",
                "multi.txt:2: hello world",
            ]
        );
    }

    #[test]
    fn grep_files_limit_stops_at_limit() {
        let dir = tempdir().unwrap();
        // three matches in a single file, limit=1 → only 1 result collected
        std::fs::write(dir.path().join("many.txt"), "hello\nhello\nhello").unwrap();
        let re = compile_grep_pattern("hello", false, false).unwrap();
        let cancel = CancellationToken::new();
        let scan = grep_files(dir.path(), true, &re, None, 0, 1, &cancel);
        assert_eq!(scan.output_lines, vec!["many.txt:1: hello"]);
        assert_eq!(scan.match_count, 1);
        assert!(scan.limit_reached);
    }

    #[test]
    fn grep_files_limit_exact_fill_also_marks_reached() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("many.txt"), "hello\nhello\nhello").unwrap();
        let re = compile_grep_pattern("hello", false, false).unwrap();
        let cancel = CancellationToken::new();
        // exactly 3 matches + limit=3 → collects all 3 and reaches the limit (aligned with pi `matchCount >= effectiveLimit`)
        let scan = grep_files(dir.path(), true, &re, None, 0, 3, &cancel);
        assert_eq!(scan.match_count, 3);
        assert!(scan.limit_reached);
    }

    #[test]
    fn grep_files_single_file_root_uses_basename() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let file = dir.path().join("a.txt");
        let re = compile_grep_pattern("hello", false, false).unwrap();
        let cancel = CancellationToken::new();
        // search root is a single file (is_dir=false) → prefix is the filename
        let scan = grep_files(&file, false, &re, None, 0, 100, &cancel);
        assert_eq!(scan.output_lines, vec!["a.txt:1: hello world"]);
    }

    #[test]
    fn grep_files_skips_binary_files() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let re = compile_grep_pattern("hello", false, false).unwrap();
        let cancel = CancellationToken::new();
        let scan = grep_files(dir.path(), true, &re, None, 0, 100, &cancel);
        // sub/binary.dat contains "hello" text but contains NUL → skipped
        assert!(
            scan.output_lines
                .iter()
                .all(|l| !l.starts_with("sub/binary.dat")),
            "binary.dat should be skipped, got {:?}",
            scan.output_lines
        );
    }

    #[test]
    fn grep_files_long_lines_truncated() {
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join("long.txt"),
            format!("MATCH{}", "x".repeat(600)),
        )
        .unwrap();
        let re = compile_grep_pattern("MATCH", false, false).unwrap();
        let cancel = CancellationToken::new();
        let scan = grep_files(dir.path(), true, &re, None, 0, 100, &cancel);
        assert_eq!(scan.output_lines.len(), 1);
        assert!(scan.output_lines[0].starts_with("long.txt:1: MATCH"));
        assert!(scan.output_lines[0].ends_with("... [truncated]"));
        assert!(scan.lines_truncated);
    }

    #[test]
    fn grep_files_cancelled_returns_early() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let re = compile_grep_pattern("hello", false, false).unwrap();
        let cancel = CancellationToken::new();
        cancel.cancel();
        // Cancelled: break at the first entry, return an empty result (execute layer rechecks and converts to Cancelled)
        let scan = grep_files(dir.path(), true, &re, None, 0, 100, &cancel);
        assert!(scan.output_lines.is_empty());
        assert_eq!(scan.match_count, 0);
    }

    #[test]
    fn grep_files_crlf_normalized_before_matching() {
        let dir = tempdir().unwrap();
        // CRLF line endings: neither matches nor output lines contain \r (aligned with pi getFileLines normalization)
        std::fs::write(dir.path().join("win.txt"), "hello\r\nworld\r\n").unwrap();
        let re = compile_grep_pattern("world", false, false).unwrap();
        let cancel = CancellationToken::new();
        let scan = grep_files(dir.path(), true, &re, None, 0, 100, &cancel);
        assert_eq!(scan.output_lines, vec!["win.txt:2: world"]);
    }

    // ─── execute (end-to-end) ──────────────────────────────────────────

    #[tokio::test]
    async fn execute_returns_matching_lines() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let tool = GrepTool::new(test_ctx(dir.path()));
        let cancel = CancellationToken::new();
        let result = tool
            .execute(
                "call-1",
                &serde_json::json!({ "pattern": "hello", "path": "." }),
                &cancel,
                Box::new(|_| {}),
            )
            .await
            .unwrap();
        assert!(!result.terminate);
        assert!(!result.message.is_error);
        assert_eq!(result.message.tool_name, "grep");
        assert_eq!(result.message.tool_call_id, "call-1");
        match &result.message.content[0] {
            ContentBlock::Text(t) => {
                assert!(t.text.contains("a.txt:1: hello world"), "got: {}", t.text);
                assert!(
                    t.text.contains(".hidden.txt:1: hello hidden"),
                    "got: {}",
                    t.text
                );
            }
            _ => panic!("expected text content"),
        }
    }

    #[tokio::test]
    async fn execute_no_match_returns_no_matches_found() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let tool = GrepTool::new(test_ctx(dir.path()));
        let cancel = CancellationToken::new();
        let result = tool
            .execute(
                "call-1",
                &serde_json::json!({ "pattern": "zzz-no-such" }),
                &cancel,
                Box::new(|_| {}),
            )
            .await
            .unwrap();
        match &result.message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t.text, "No matches found"),
            _ => panic!("expected text content"),
        }
        assert!(result.message.details.is_none());
    }

    #[tokio::test]
    async fn execute_missing_path_errors() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let tool = GrepTool::new(test_ctx(dir.path()));
        let cancel = CancellationToken::new();
        let err = tool
            .execute(
                "call-1",
                &serde_json::json!({ "pattern": "hello", "path": "does-not-exist" }),
                &cancel,
                Box::new(|_| {}),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Execution(_)));
    }

    #[tokio::test]
    async fn execute_missing_pattern_is_invalid() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let tool = GrepTool::new(test_ctx(dir.path()));
        let cancel = CancellationToken::new();
        let err = tool
            .execute("call-1", &serde_json::json!({}), &cancel, Box::new(|_| {}))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[tokio::test]
    async fn execute_invalid_regex_is_invalid_argument() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let tool = GrepTool::new(test_ctx(dir.path()));
        let cancel = CancellationToken::new();
        let err = tool
            .execute(
                "call-1",
                &serde_json::json!({ "pattern": "(" }),
                &cancel,
                Box::new(|_| {}),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[tokio::test]
    async fn execute_cancelled_before_start() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let tool = GrepTool::new(test_ctx(dir.path()));
        let cancel = CancellationToken::new();
        cancel.cancel();
        let err = tool
            .execute(
                "call-1",
                &serde_json::json!({ "pattern": "hello" }),
                &cancel,
                Box::new(|_| {}),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Cancelled));
    }

    #[tokio::test]
    async fn execute_search_path_is_resolved_against_working_dir() {
        // path defaults to '.' → working_dir; if path is a relative subdirectory → resolved relative to working_dir;
        // when search root is a subdirectory, output is relative to that subdirectory (aligned with pi `formatPath` isDirectory → relative)
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let tool = GrepTool::new(test_ctx(dir.path()));
        let cancel = CancellationToken::new();
        let result = tool
            .execute(
                "call-1",
                &serde_json::json!({ "pattern": "goodbye", "path": "sub" }),
                &cancel,
                Box::new(|_| {}),
            )
            .await
            .unwrap();
        match &result.message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t.text, "deep.txt:1: goodbye"),
            _ => panic!("expected text content"),
        }
    }

    #[test]
    fn tool_metadata_matches_pi() {
        let tool = GrepTool::new(ToolContext::new(PathBuf::from("/tmp"), None, None));
        assert_eq!(tool.name(), "grep");
        assert_eq!(tool.label(), "grep");
        assert_eq!(
            tool.description(),
            "Search file contents for a pattern. Returns matching lines with file paths and line numbers. Respects .gitignore. Output is truncated to 100 matches or 50KB (whichever is hit first). Long lines are truncated to 500 chars."
        );
        assert_eq!(
            tool.prompt_snippet(),
            Some("Search file contents for patterns (respects .gitignore)")
        );
        let params = tool.parameters();
        assert_eq!(params["required"][0], "pattern");
        assert!(params["properties"]["pattern"]["type"].is_string());
        assert!(params["properties"]["context"]["type"].is_string());
        assert!(params["properties"]["limit"]["type"].is_string());
    }
}
