// REQ-APP-04 (pi-348): find tool.
// Search files by glob pattern, return a list of paths relative to the search directory, respect .gitignore, and truncate by result count and byte limit.
// Corresponds to original TS: reference/pi-original/packages/coding-agent/src/core/tools/find.ts.
// Paradigm replacement (see tasks/T-AP-04-4-implemented.md "Ecosystem Reuse Verification Conclusions"):
// - fd subprocess + ensureTool mechanism → `ignore::WalkBuilder` (gitignore-aware traversal) + `globset` (glob matching);
// - `FindOperations` pluggable interface (SSH remote proxy) dropped in v1 (no remote execution architecture);
// - `formatFindCall`/`formatFindResult` rendering functions were eliminated by the unified fallback layout in T-AP-11-2 (v1 has no renderer system),
//   limit/truncation warnings are embedded into the content text as a notices section, and are not lost when the UI displays text output.
// Behavior mapping: pi `--hidden` (include hidden files) → `WalkBuilder::hidden(false)`;
// Net effect of pi `--no-require-git` (respect .gitignore everywhere) → `WalkBuilder::require_git(false)`;
// Do not follow symlinks → keep the `follow_links(false)` default.

use crate::tools::path_utils::resolve_to_cwd;
use crate::truncate::{
    format_size, truncate_head, truncation_to_json, TruncationOptions, DEFAULT_MAX_BYTES,
};
use crate::util::now_millis;
use async_trait::async_trait;
use globset::{GlobBuilder, GlobMatcher};
use std::path::Path;
use tokio_util::sync::CancellationToken;
use zen_ai::tool::{Tool, ToolContext, ToolError, ToolExecutionResult, ToolUpdateCallback};
use zen_ai::types::{ContentBlock, TextContent, ToolResultMessage, Usage};

/// find result count limit (pi `DEFAULT_LIMIT = 1000`, fd `--max-results` semantics).
const DEFAULT_LIMIT: usize = 1000;

/// Raw output of `find_files` traversal matches (private, not exported).
#[derive(Debug)]
struct FindMatches {
    /// Posix path relative to the search root (directory entries have a trailing `/`).
    paths: Vec<String>,
    /// Whether the limit is reached (collected count >= limit, matching pi `relativized.length >= effectiveLimit`).
    limit_reached: bool,
}

/// Final output assembled by `assemble_find_result` (private, not exported).
struct FindResultAssembly {
    /// Final content (includes appended notices section; when there are no results, pi's "No files found matching pattern").
    content: String,
    /// details metadata (`truncation`/`resultLimitReached`, resolved from pi `FindToolDetails`; `None` when both are absent).
    details: Option<serde_json::Value>,
}

/// File search tool, invoked by the LLM when the agent needs to "find files by glob pattern (respecting .gitignore)" —
/// User-visible effect: returns a list of matching paths relative to the search directory; when over limit (default 1000 entries / 50KB, whichever comes first), the body appends
/// a notice like `[... results limit reached...]`, aligned with pi find tool behavior.
/// Parameter Schema (JSON Schema, static definition isomorphic to the read tool):
/// - `pattern` (string, required): glob pattern, such as `'*.ts'`, `'**/*.json'`, `'src/**/*.spec.ts'`
/// - `path` (string, optional): search directory, defaults to the current working directory
/// - `limit` (number, optional): maximum number of results, defaults to [`DEFAULT_LIMIT`]
///
/// The search core uses `ignore::WalkBuilder` (gitignore-aware, includes hidden files, does not follow symlinks) +
/// `globset` matching replaces pi's fd subprocess (see tasks/T-AP-04-4-implemented.md "Ecosystem Reuse Verification Conclusions").
#[derive(Debug)]
pub struct FindTool {
    tool_context: ToolContext,
}

impl FindTool {
    /// Construct a find tool instance with the session environment (called by factory `create(ToolName::Find)`; isomorphic to ReadTool::new).
    pub fn new(tool_context: ToolContext) -> Self {
        Self { tool_context }
    }
}

#[async_trait]
impl Tool for FindTool {
    /// Tool name `"find"` (aligned with pi tool name and `ToolName::Find`).
    fn name(&self) -> &str {
        "find"
    }

    /// LLM-visible description: search files by glob pattern, return paths relative to the search directory, respect .gitignore,
    /// output is truncated to `DEFAULT_LIMIT` results or `DEFAULT_MAX_BYTES/1024` KB (whichever comes first) —
    /// Wording aligned with pi `createFindToolDefinition.description`.
    fn description(&self) -> &str {
        "Search for files by glob pattern. Returns matching file paths relative to the search \
            directory. Respects .gitignore. Output is truncated to 1000 results or 50KB \
            (whichever is hit first)."
    }

    /// Parameter JSON Schema (pattern required, path/limit optional), statically cached with OnceLock, aligned with the read tool.
    fn parameters(&self) -> &serde_json::Value {
        static PARAMS: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
        PARAMS.get_or_init(|| {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: current directory)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of results (default: 1000)"
                    }
                },
                "required": ["pattern"]
            })
        })
    }

    /// Label `"find"`.
    fn label(&self) -> &str {
        "find"
    }

    /// System prompt summary "Find files by glob pattern (respects .gitignore)" (aligned with pi promptSnippet).
    fn prompt_snippet(&self) -> Option<&str> {
        Some("Find files by glob pattern (respects .gitignore)")
    }

    /// Execute find: parse arguments → `resolve_to_cwd` resolves the search directory (default `.`) → precheck directory existence →
    /// run the synchronous search core (`find_files`) inside `spawn_blocking` → `tokio::select!` race cancellation →
    /// cancellation recheck → `assemble_find_result` assembly → `ToolResultMessage`.
    /// Cancellation semantics: cancel already cancelled or cancelled mid-traversal → `ToolError::Cancelled` (when cancelled mid-traversal, `find_files`
    /// breaks early to release the blocking thread, and the result is discarded); `is_error=false`; details contain `truncation`/`resultLimitReached`
    /// (mapped from pi `FindToolDetails`; `None` when both are absent).
    /// Edge cases: search path does not exist → `ToolError::Execution` ("Path not found: ..."); invalid glob pattern →
    /// `ToolError::InvalidArguments`。
    async fn execute(
        &self,
        tool_call_id: &str,
        params: &serde_json::Value,
        cancel: &CancellationToken,
        _on_update: ToolUpdateCallback,
    ) -> Result<ToolExecutionResult, ToolError> {
        let (pattern, search_dir, limit) = parse_find_args(params)?;
        let effective_limit = limit.unwrap_or(DEFAULT_LIMIT);

        let search_path = resolve_to_cwd(
            search_dir.as_deref().unwrap_or("."),
            self.tool_context.working_dir.as_path(),
        )
        .map_err(|e| ToolError::Execution(format!("Failed to resolve find path: {e}")))?;

        // Precheck search path existence (pi's custom ops path does pathExists; the fd path lets fd report errors, zen uniformly prechecks)
        if !search_path.try_exists().unwrap_or(false) {
            return Err(ToolError::Execution(format!(
                "Path not found: {}",
                search_path.display()
            )));
        }

        let cancel_blocking = cancel.clone();
        let handle = tokio::task::spawn_blocking(move || {
            find_files(&search_path, &pattern, effective_limit, &cancel_blocking)
        });

        let joined = tokio::select! {
            _ = cancel.cancelled() => {
                // Don't abort: blocking tasks carry their own cancel check; the next entry breaks and self-terminates quickly
                return Err(ToolError::Cancelled);
            }
            joined = handle => joined,
        };
        let matches = match joined {
            Ok(Ok(m)) => m,
            Ok(Err(e)) => return Err(e),
            Err(join_err) => {
                return Err(ToolError::Execution(format!(
                    "find task failed: {join_err}"
                )));
            }
        };

        // Cancellation recheck: find_files may return normally at the same time the cancellation signal arrives (early break); in this case the result is discarded
        if cancel.is_cancelled() {
            return Err(ToolError::Cancelled);
        }

        let assembly = assemble_find_result(matches, effective_limit);

        let now = now_millis();
        let message = ToolResultMessage {
            tool_call_id: tool_call_id.to_string(),
            tool_name: "find".to_string(),
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

/// Extract (pattern, path, limit) from tool argument JSON. Missing/non-string pattern, non-positive-integer limit →
/// `ToolError::InvalidArguments`; path/limit default to None (defaults are expanded by the caller per pi `??` semantics).
fn parse_find_args(
    params: &serde_json::Value,
) -> Result<(String, Option<String>, Option<usize>), ToolError> {
    let pattern = params
        .get("pattern")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ToolError::InvalidArguments("Missing required 'pattern' parameter".into()))?
        .to_string();
    let path = params
        .get("path")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let limit = params
        .get("limit")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    Ok((pattern, path, limit))
}

/// Compile the user glob pattern into a globset matcher and determine the matching mode:
/// - Pattern contains `/` → full-path mode (pi `--full-path`), candidate = absolute path posix string; if the pattern does not start with `/` or `**/`
///   and when not `**`, prepend `**/` (same as pi, so `src/**/*.spec.ts` can match absolute paths under any root);
/// - pattern does not contain `/` → basename pattern (pi defaults to fd --glob), candidates = entry filenames.
///
/// `literal_separator(true)` makes `*`/`?` not cross `/` (globset crosses by default, unlike fd semantics, so it must be explicitly enabled).
/// Compilation failure (invalid glob) → `ToolError::InvalidArguments`.
/// Also reused by the grep tool's (T-AP-04-5) `--glob` file filtering (candidates are likewise absolute posix file paths).
pub(crate) fn compile_find_glob(pattern: &str) -> Result<(GlobMatcher, bool), ToolError> {
    let full_path = pattern.contains('/');
    let effective =
        if full_path && !pattern.starts_with('/') && !pattern.starts_with("**/") && pattern != "**"
        {
            format!("**/{pattern}")
        } else {
            pattern.to_string()
        };
    let glob = GlobBuilder::new(&effective)
        .literal_separator(true)
        .build()
        .map_err(|e| {
            ToolError::InvalidArguments(format!("Invalid glob pattern '{pattern}': {e}"))
        })?;
    Ok((glob.compile_matcher(), full_path))
}

/// Relativize walked entry paths to the search root and convert to posix: strip the prefix if it starts with search_path, otherwise (the walk root is
/// search_path, theoretically unreachable, defensive fallback) return the absolute path with leading `/` stripped; append trailing `/` to directory entries
/// (aligned with pi's handling of fd output `hadTrailingSlash`); use `to_string_lossy` for non-UTF-8 filenames
/// (fd output is also lossy, no error).
fn relativize_match(path: &Path, search_path: &Path, is_dir: bool) -> String {
    let rel = path
        .strip_prefix(search_path)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string();
    // Fallback branch (when strip_prefix fails): remove the leading separator from the absolute path
    let rel = rel
        .trim_start_matches(std::path::MAIN_SEPARATOR)
        .to_string();
    let mut rel = rel.replace(std::path::MAIN_SEPARATOR, "/");
    if is_dir && !rel.ends_with('/') {
        rel.push('/');
    }
    rel
}

/// Traverse the search root (`ignore::WalkBuilder`: `hidden(false)` includes hidden entries; `require_git(false)` is respected everywhere)
/// .gitignore, does not follow symlinks by default), match each non-root entry against the pattern from `compile_find_glob`, and on a hit
/// relativize and collect; stop when the entry count reaches `limit` (`limit_reached = collected_count >= limit`, aligned with pi
/// `relativized.length >= effectiveLimit`; when limit=0, the first hit reaches the limit and returns an empty set). Each entry checks
/// `cancel.is_cancelled()`; if cancelled, break early and return partial/empty results (converted by the execute-layer cancellation recheck to
/// `ToolError::Cancelled`). Synchronous function, executed inside `spawn_blocking` in execute.
fn find_files(
    search_path: &Path,
    pattern: &str,
    limit: usize,
    cancel: &CancellationToken,
) -> Result<FindMatches, ToolError> {
    let (matcher, full_path) = compile_find_glob(pattern)?;

    let walker = ignore::WalkBuilder::new(search_path)
        .hidden(false) // Include hidden files (aligned with fd --hidden)
        .require_git(false) // Respect .gitignore everywhere (aligned with the net effect of pi --no-require-git)
        .build();

    let mut paths = Vec::new();
    let mut limit_reached = false;

    for entry in walker {
        if cancel.is_cancelled() {
            break;
        }
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue, // Skip failures on a single walked entry (permissions, etc.) without aborting, aligned with skills/load.rs
        };
        let entry_path = entry.path();
        if entry_path == search_path {
            continue; // Skip the search root itself (fd does not output the root directory)
        }

        let matched = if full_path {
            matcher.is_match(to_posix_string(&entry_path.to_string_lossy()))
        } else {
            matcher.is_match(entry.file_name().to_string_lossy().as_ref())
        };
        if !matched {
            continue;
        }

        if paths.len() >= limit {
            limit_reached = true;
            break;
        }
        let is_dir = entry.file_type().is_some_and(|ft| ft.is_dir());
        paths.push(relativize_match(entry_path, search_path, is_dir));
    }

    // Also considered limit-reached when exactly `limit` entries are collected (no more entries to walk) (aligned with pi `length >= effectiveLimit`)
    limit_reached = limit_reached || paths.len() >= limit;

    Ok(FindMatches {
        paths,
        limit_reached,
    })
}

/// Assemble match results into final output: `paths.join("\n")` → `truncate_head` (`max_lines: usize::MAX`,
/// only the `DEFAULT_MAX_BYTES` byte limit applies) → append notices (`limit` reached notice with
/// "Use limit=<limit*2> for more, or refine pattern" suffix, truncation notice
/// `format_size(DEFAULT_MAX_BYTES) limit reached`, ending with a `\n\n[..]` section, aligned with pi fd path logic)
/// → details JSON (`resultLimitReached`/`truncation` are omitted from details when absent; `truncation` subfields
/// uses the shared `crate::truncate::truncation_to_json`). On empty results (including limit=0), returns pi's
/// "No files found matching pattern" message with details None (aligned with pi's early `if (!output)` branch).
fn assemble_find_result(matches: FindMatches, limit: usize) -> FindResultAssembly {
    if matches.paths.is_empty() {
        return FindResultAssembly {
            content: "No files found matching pattern".to_string(),
            details: None,
        };
    }

    let raw_output = matches.paths.join("\n");
    let truncation = truncate_head(
        &raw_output,
        TruncationOptions {
            max_lines: usize::MAX,
            max_bytes: DEFAULT_MAX_BYTES,
        },
    );

    let mut notices: Vec<String> = Vec::new();
    if matches.limit_reached {
        notices.push(format!(
            "{limit} results limit reached. Use limit={} for more, or refine pattern",
            limit * 2
        ));
    }
    if truncation.truncated {
        notices.push(format!("{} limit reached", format_size(DEFAULT_MAX_BYTES)));
    }

    let mut details = serde_json::Map::new();
    if matches.limit_reached {
        details.insert("resultLimitReached".into(), serde_json::json!(limit));
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

    FindResultAssembly {
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

    /// Build a standard search tree:
    ///   a.ts b.ts c.json .hidden.ts .gitignore(containing *.json)
    ///   sub/d.ts sub/deep/e.rs
    fn build_tree(dir: &std::path::Path) {
        std::fs::create_dir_all(dir.join("sub/deep")).unwrap();
        for f in ["a.ts", "b.ts", "c.json", ".hidden.ts"] {
            std::fs::write(dir.join(f), "x").unwrap();
        }
        std::fs::write(dir.join("sub/d.ts"), "x").unwrap();
        std::fs::write(dir.join("sub/deep/e.rs"), "x").unwrap();
        std::fs::write(dir.join(".gitignore"), "*.json\n").unwrap();
    }

    // ─── parse_find_args ────────────────────────────────────────────────

    #[test]
    fn parse_find_args_pattern_only() {
        let (pattern, path, limit) =
            parse_find_args(&serde_json::json!({ "pattern": "*.ts" })).unwrap();
        assert_eq!(pattern, "*.ts");
        assert_eq!(path, None);
        assert_eq!(limit, None);
    }

    #[test]
    fn parse_find_args_full() {
        let (pattern, path, limit) =
            parse_find_args(&serde_json::json!({ "pattern": "*.ts", "path": "src", "limit": 50 }))
                .unwrap();
        assert_eq!(pattern, "*.ts");
        assert_eq!(path.as_deref(), Some("src"));
        assert_eq!(limit, Some(50));
    }

    #[test]
    fn parse_find_args_missing_pattern_is_invalid() {
        let err = parse_find_args(&serde_json::json!({ "path": "src" })).unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn parse_find_args_pattern_not_string_is_invalid() {
        let err = parse_find_args(&serde_json::json!({ "pattern": 42 })).unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn parse_find_args_non_numeric_limit_treated_as_none() {
        let (_, _, limit) =
            parse_find_args(&serde_json::json!({ "pattern": "*.ts", "limit": "lots" })).unwrap();
        assert_eq!(limit, None);
    }

    // ─── compile_find_glob ──────────────────────────────────────────────

    #[test]
    fn compile_find_glob_basename_mode() {
        let (matcher, full_path) = compile_find_glob("*.ts").unwrap();
        assert!(!full_path);
        assert!(matcher.is_match("a.ts"));
        assert!(matcher.is_match(".hidden.ts"));
        assert!(!matcher.is_match("a.json"));
    }

    #[test]
    fn compile_find_glob_basename_does_not_cross_separator() {
        // literal_separator(true): `*` does not cross `/`; `a/b.rs` does not match `*.rs`
        let (matcher, full_path) = compile_find_glob("*.rs").unwrap();
        assert!(!full_path);
        assert!(!matcher.is_match("a/b.rs"));
    }

    #[test]
    fn compile_find_glob_path_mode_prepends_double_star() {
        let (matcher, full_path) = compile_find_glob("src/**/*.spec.ts").unwrap();
        assert!(full_path);
        // The `**/` prefix lets absolute paths under any root match (aligned with pi `--full-path` + `**/` prefixing)
        assert!(matcher.is_match("/work/proj/src/a/b.spec.ts"));
        assert!(!matcher.is_match("/work/proj/other/a.spec.ts"));
    }

    #[test]
    fn compile_find_glob_path_mode_no_prefix_already_double_star() {
        let (matcher, full_path) = compile_find_glob("**/*.json").unwrap();
        assert!(full_path);
        assert!(matcher.is_match("/work/proj/a.json"));
        assert!(matcher.is_match("/work/proj/sub/a.json"));
    }

    #[test]
    fn compile_find_glob_star_star_alone_matches_everything() {
        // "**" does not contain '/' → basename pattern (aligned with pi `pattern.includes("/")` check); globset treats a bare `**`
        // as a special case matching everything
        let (matcher, full_path) = compile_find_glob("**").unwrap();
        assert!(!full_path);
        assert!(matcher.is_match("a.ts"));
        assert!(matcher.is_match("e.rs"));
        assert!(matcher.is_match(".hidden.ts"));
    }

    #[test]
    fn compile_find_glob_leading_slash_no_prefix() {
        // Starting with `/` means match from the root (pi does not prepend the `**/` prefix)
        let (matcher, full_path) = compile_find_glob("/src/*.ts").unwrap();
        assert!(full_path);
        assert!(matcher.is_match("/src/a.ts"));
        assert!(!matcher.is_match("/work/src/a.ts"));
    }

    #[test]
    fn compile_find_glob_invalid_pattern_is_invalid_argument() {
        let err = compile_find_glob("[").unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    // ─── relativize_match ───────────────────────────────────────────────

    #[test]
    fn relativize_match_strips_search_root() {
        let root = Path::new("/tmp/proj");
        assert_eq!(
            relativize_match(Path::new("/tmp/proj/src/a.ts"), root, false),
            "src/a.ts"
        );
    }

    #[test]
    fn relativize_match_dir_gets_trailing_slash() {
        let root = Path::new("/tmp/proj");
        assert_eq!(
            relativize_match(Path::new("/tmp/proj/sub"), root, true),
            "sub/"
        );
        assert_eq!(
            relativize_match(Path::new("/tmp/proj/sub/deep"), root, true),
            "sub/deep/"
        );
    }

    #[test]
    fn relativize_match_root_itself_is_empty() {
        let root = Path::new("/tmp/proj");
        assert_eq!(relativize_match(Path::new("/tmp/proj"), root, false), "");
    }

    // ─── assemble_find_result ───────────────────────────────────────────

    #[test]
    fn assemble_empty_returns_no_files_found() {
        let assembly = assemble_find_result(
            FindMatches {
                paths: vec![],
                limit_reached: false,
            },
            1000,
        );
        assert_eq!(assembly.content, "No files found matching pattern");
        assert!(assembly.details.is_none());
    }

    #[test]
    fn assemble_limit_zero_returns_no_files_found() {
        // Aligned with pi: empty output returns early with "No files found matching pattern" (fd path `if (!output)` branch)
        let assembly = assemble_find_result(
            FindMatches {
                paths: vec![],
                limit_reached: true,
            },
            0,
        );
        assert_eq!(assembly.content, "No files found matching pattern");
        assert!(assembly.details.is_none());
    }

    #[test]
    fn assemble_within_limit_no_notices_no_details() {
        let assembly = assemble_find_result(
            FindMatches {
                paths: vec!["a.ts".into(), "b.ts".into()],
                limit_reached: false,
            },
            1000,
        );
        assert_eq!(assembly.content, "a.ts\nb.ts");
        assert!(assembly.details.is_none());
    }

    #[test]
    fn assemble_limit_reached_appends_notice_and_details() {
        let assembly = assemble_find_result(
            FindMatches {
                paths: vec!["a.ts".into(), "b.ts".into()],
                limit_reached: true,
            },
            2,
        );
        assert_eq!(
            assembly.content,
            "a.ts\nb.ts\n\n[2 results limit reached. Use limit=4 for more, or refine pattern]"
        );
        let details = assembly.details.unwrap();
        assert_eq!(details["resultLimitReached"], 2);
        assert!(details.get("truncation").is_none());
    }

    #[test]
    fn assemble_truncated_appends_size_notice_and_truncation_details() {
        // Over 50KB → truncation notice + truncation details (6000 lines × ~11B ≈ 66KB > 50KB)
        let long: Vec<String> = (0..6000).map(|i| format!("file_{i}.ts")).collect();
        let assembly = assemble_find_result(
            FindMatches {
                paths: long.clone(),
                limit_reached: false,
            },
            1000,
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
        assert!(details.get("resultLimitReached").is_none());
    }

    // ─── find_files (tempdir integration)──────────────────────────────────────

    #[test]
    fn find_files_basename_glob() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let cancel = CancellationToken::new();
        let found = find_files(dir.path(), "*.ts", 1000, &cancel).unwrap();
        let mut names: Vec<&str> = found.paths.iter().map(|s| s.as_str()).collect();
        names.sort_unstable();
        // .gitignore ignores only *.json; hidden files are included; directories do not match *.ts (sub/ is a directory but its name does not match)
        assert_eq!(names, vec![".hidden.ts", "a.ts", "b.ts", "sub/d.ts"]);
        assert!(!found.limit_reached);
    }

    #[test]
    fn find_files_fullpath_glob() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let cancel = CancellationToken::new();
        let found = find_files(dir.path(), "**/*.rs", 1000, &cancel).unwrap();
        assert_eq!(found.paths, vec!["sub/deep/e.rs"]);
        assert!(!found.limit_reached);
    }

    #[test]
    fn find_files_matches_directories_with_trailing_slash() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let cancel = CancellationToken::new();
        let found = find_files(dir.path(), "sub/*", 1000, &cancel).unwrap();
        let mut names: Vec<&str> = found.paths.iter().map(|s| s.as_str()).collect();
        names.sort_unstable();
        // sub/deep is a directory → trailing `/`; sub/d.ts is a file
        assert_eq!(names, vec!["sub/d.ts", "sub/deep/"]);
    }

    #[test]
    fn find_files_includes_hidden_files() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let cancel = CancellationToken::new();
        // Full-path pattern `**/*.ts` also matches .hidden.ts (--hidden semantics: hidden files are included in the search)
        let found = find_files(dir.path(), "**/*.ts", 1000, &cancel).unwrap();
        assert!(found.paths.contains(&".hidden.ts".to_string()));
    }

    #[test]
    fn find_files_respects_gitignore() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let cancel = CancellationToken::new();
        // .gitignore contains `*.json` → c.json is excluded (require_git(false): takes effect even outside a git repo)
        let found = find_files(dir.path(), "*.json", 1000, &cancel).unwrap();
        assert!(
            found.paths.is_empty(),
            "c.json should be ignored, got {:?}",
            found.paths
        );
    }

    #[test]
    fn find_files_limit_reached_returns_exactly_limit_results() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let cancel = CancellationToken::new();
        let found = find_files(dir.path(), "*.ts", 2, &cancel).unwrap();
        assert_eq!(found.paths.len(), 2);
        assert!(found.limit_reached);
    }

    #[test]
    fn find_files_no_match_empty_result() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let cancel = CancellationToken::new();
        let found = find_files(dir.path(), "*.zig", 1000, &cancel).unwrap();
        assert!(found.paths.is_empty());
        assert!(!found.limit_reached);
    }

    #[test]
    fn find_files_limit_zero_returns_empty_with_limit_reached() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let cancel = CancellationToken::new();
        let found = find_files(dir.path(), "*.ts", 0, &cancel).unwrap();
        assert!(found.paths.is_empty());
        assert!(found.limit_reached);
    }

    #[test]
    fn find_files_cancelled_returns_early() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let cancel = CancellationToken::new();
        cancel.cancel();
        // Cancelled: break at the first entry, return an empty result (execute layer rechecks and converts to Cancelled)
        let found = find_files(dir.path(), "*.ts", 1000, &cancel).unwrap();
        assert!(found.paths.is_empty());
    }

    #[test]
    fn find_files_invalid_glob_propagates_error() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let cancel = CancellationToken::new();
        let err = find_files(dir.path(), "[", 1000, &cancel).unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    // ─── execute (end-to-end) ──────────────────────────────────────────

    #[tokio::test]
    async fn execute_returns_matching_files() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let tool = FindTool::new(test_ctx(dir.path()));
        let cancel = CancellationToken::new();
        let result = tool
            .execute(
                "call-1",
                &serde_json::json!({ "pattern": "*.ts", "path": "." }),
                &cancel,
                Box::new(|_| {}),
            )
            .await
            .unwrap();
        assert!(!result.terminate);
        assert!(!result.message.is_error);
        assert_eq!(result.message.tool_name, "find");
        assert_eq!(result.message.tool_call_id, "call-1");
        let text = &result.message.content[0];
        match text {
            ContentBlock::Text(t) => {
                assert!(t.text.contains("a.ts"), "got: {}", t.text);
                assert!(t.text.contains("b.ts"), "got: {}", t.text);
            }
            _ => panic!("expected text content"),
        }
    }

    #[tokio::test]
    async fn execute_no_match_returns_no_files_found() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let tool = FindTool::new(test_ctx(dir.path()));
        let cancel = CancellationToken::new();
        let result = tool
            .execute(
                "call-1",
                &serde_json::json!({ "pattern": "*.zig" }),
                &cancel,
                Box::new(|_| {}),
            )
            .await
            .unwrap();
        match &result.message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t.text, "No files found matching pattern"),
            _ => panic!("expected text content"),
        }
        assert!(result.message.details.is_none());
    }

    #[tokio::test]
    async fn execute_missing_path_errors() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let tool = FindTool::new(test_ctx(dir.path()));
        let cancel = CancellationToken::new();
        let err = tool
            .execute(
                "call-1",
                &serde_json::json!({ "pattern": "*.ts", "path": "does-not-exist" }),
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
        let tool = FindTool::new(test_ctx(dir.path()));
        let cancel = CancellationToken::new();
        let err = tool
            .execute("call-1", &serde_json::json!({}), &cancel, Box::new(|_| {}))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[tokio::test]
    async fn execute_cancelled_before_start() {
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let tool = FindTool::new(test_ctx(dir.path()));
        let cancel = CancellationToken::new();
        cancel.cancel();
        let err = tool
            .execute(
                "call-1",
                &serde_json::json!({ "pattern": "*.ts" }),
                &cancel,
                Box::new(|_| {}),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Cancelled));
    }

    #[tokio::test]
    async fn execute_search_path_is_resolved_against_working_dir() {
        // path defaults to '.' → working_dir; an absolute/relative subdirectory path → resolved relative to working_dir
        let dir = tempdir().unwrap();
        build_tree(dir.path());
        let tool = FindTool::new(test_ctx(dir.path()));
        let cancel = CancellationToken::new();
        let result = tool
            .execute(
                "call-1",
                &serde_json::json!({ "pattern": "d.ts", "path": "sub" }),
                &cancel,
                Box::new(|_| {}),
            )
            .await
            .unwrap();
        match &result.message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t.text, "d.ts"),
            _ => panic!("expected text content"),
        }
    }

    #[test]
    fn tool_metadata_matches_pi() {
        let tool = FindTool::new(ToolContext::new(PathBuf::from("/tmp"), None, None));
        assert_eq!(tool.name(), "find");
        assert_eq!(tool.label(), "find");
        assert_eq!(tool.description(), "Search for files by glob pattern. Returns matching file paths relative to the search directory. Respects .gitignore. Output is truncated to 1000 results or 50KB (whichever is hit first).");
        assert_eq!(
            tool.prompt_snippet(),
            Some("Find files by glob pattern (respects .gitignore)")
        );
        let params = tool.parameters();
        assert_eq!(params["required"][0], "pattern");
        assert!(params["properties"]["pattern"]["type"].is_string());
    }
}
