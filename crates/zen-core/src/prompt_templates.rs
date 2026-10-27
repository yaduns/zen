use crate::frontmatter;
use regex::{Captures, Regex};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// A reusable prompt parsed from a disk markdown template—name is the unique key for slash routing,
/// content is the body used for parameter substitution before submission; description/argument_hint serve command-list display and completion hints.
/// Immutable after loading and not persisted (no serde requirement)
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    pub name: String, // Filename with .md removed; file_stem semantics are consistent for a.b.md; when implementing, note the .md hidden-file edge case—decide whether to skip and log diagnostics)
    pub description: String, // frontmatter description; if missing, fall back to the first non-empty line truncated to 60 chars + "..."; if still none, empty string
    pub argument_hint: Option<String>, // frontmatter argument-hint; used for slash-completion display (downstream T-AP-03-2/T-TU-03-3)
    pub content: String,               // Body after frontmatter (input to substitution engine)
}

/// Category code for non-fatal loading warnings
#[derive(serde::Serialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PromptTemplateDiagnosticCode {
    FileInfoFailed, // Path metadata probe failed and is not "not found" (not found is silently skipped)
    ListFailed,     // Directory read failed
    ReadFailed,     // File read failed
    ParseFailed, // Malformed frontmatter line (same semantics as the identically named skills code)
}

/// Carrier for a non-fatal loading warning—constructed when the loading entry point hits an IO/parse failure, aggregated into the result for the caller to display or ignore
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptTemplateDiagnostic {
    pub code: PromptTemplateDiagnosticCode, // Category code; mapping rules follow the skills read_skill_file precedent (IO failure → ReadFailed, directory traversal failure → ListFailed, etc.)
    pub message: String, // Human-readable reason including underlying error text; format follows the skills precedent format!("failed to read {}: {e}", path.display()), and parse failures include a path prefix
    pub path: String, // Display-oriented path string (`path.display().to_string()`, not `PathBuf`—diagnostics are only displayed/compared, not used for re-addressing, following the skills precedent)
}

/// Return aggregate for `load_prompt_templates`—the carrier of the "warnings do not interrupt" semantics: failure of some files does not affect the remaining successful items, and both can be non-empty at the same time;
pub struct LoadPromptTemplatesResult {
    pub prompt_templates: Vec<PromptTemplate>, // Successfully loaded templates; multiple input paths follow argument order, entries within a directory are sorted by name byte order; templates with the same name are not deduplicated, routing priority is determined by the consumer T-AP-03-4
    pub diagnostics: Vec<PromptTemplateDiagnostic>, // Non-interrupting warnings (may be empty); ordering is produced as generated, no additional sorting
}

/// Unified construction for IO-type diagnostics (skills precedent message format: `failed to <verb> {path}: {e}`)
fn io_diagnostic(
    code: PromptTemplateDiagnosticCode,
    message: String,
    path: &Path,
) -> PromptTemplateDiagnostic {
    PromptTemplateDiagnostic {
        code,
        message,
        path: path.display().to_string(),
    }
}

/// Single-file load: read raw content → pure assembly; on `ReadFailed` / `ParseFailed`, record a diagnostic and continue
async fn load_template_file(
    file_path: &Path,
    out: &mut Vec<PromptTemplate>,
    diagnostics: &mut Vec<PromptTemplateDiagnostic>,
) {
    match tokio::fs::read_to_string(file_path).await {
        Ok(content) => match build_prompt_template(content, file_path) {
            Ok(template) => out.push(template),
            Err(diagnostic) => diagnostics.push(diagnostic),
        },
        Err(e) => diagnostics.push(io_diagnostic(
            PromptTemplateDiagnosticCode::ReadFailed,
            format!("failed to read {}: {e}", file_path.display()),
            file_path,
        )),
    }
}

/// Directory load: non-recursively scan direct children, only `.md` files, sort by path byte order (= file name byte order within the directory) then load.
/// `readdir` failure → `ListFailed` diagnostic (pi silently swallows it; zen makes it explicit—doc diagnostic code semantics);
/// Symlinks are resolved by following `metadata` (stat semantics); broken links/failures → silently skip (aligning with pi catch/continue).
async fn load_templates_from_dir(
    dir: &Path,
    out: &mut Vec<PromptTemplate>,
    diagnostics: &mut Vec<PromptTemplateDiagnostic>,
) {
    let mut read_dir = match tokio::fs::read_dir(dir).await {
        Ok(read_dir) => read_dir,
        Err(e) => {
            diagnostics.push(io_diagnostic(
                PromptTemplateDiagnosticCode::ListFailed,
                format!("failed to list {}: {e}", dir.display()),
                dir,
            ));
            return;
        }
    };

    // Collect direct children (path + type from `readdir`—does not follow symlinks), sort by path byte order
    let mut entries: Vec<(PathBuf, std::fs::FileType)> = Vec::new();
    loop {
        match read_dir.next_entry().await {
            Ok(Some(entry)) => {
                if let Ok(file_type) = entry.file_type().await {
                    entries.push((entry.path(), file_type));
                }
            }
            Ok(None) => break,
            Err(e) => {
                // Traversal interrupted (same semantic location as pi `readdirSync` one-shot call)
                diagnostics.push(io_diagnostic(
                    PromptTemplateDiagnosticCode::ListFailed,
                    format!("failed to read entries of {}: {e}", dir.display()),
                    dir,
                ));
                break;
            }
        }
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    for (full_path, file_type) in entries {
        // Type check: symlink → follow `metadata` (pi `statSync` branch); regular entries use `readdir`'s built-in `is_file` (pi `entry.isFile()`, zero IO)
        let is_file = if file_type.is_symlink() {
            match tokio::fs::metadata(&full_path).await {
                Ok(meta) => meta.is_file(),
                Err(_) => continue, // Broken symlink: silently skip (same as pi)
            }
        } else {
            file_type.is_file()
        };
        // Only `.md` files (string suffix check on `file_name`, matching pi `endsWith` case-sensitive semantics)
        let is_markdown = full_path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(".md"));
        if !is_file || !is_markdown {
            continue;
        }
        load_template_file(&full_path, out, diagnostics).await;
    }
}

/// Resource loading entry point before session submission of `/name args`: the resource loader (T-AP-05-3) calls it when assembling session resources,
/// Turn user-configured template paths (files or directories, can be mixed) into a routable template list—"silently skip if not found, warn if read fails".
///
/// Behavior (aligning with pi-207; zen resolves `ExecutionEnv` to direct fs, following the skills/load.rs precedent):
/// - Directory input: non-recursively scan direct children, take only `.md` files, sort by name then load (pi `localeCompare` in zen
///   is simplified to byte-order sorting; the difference only affects load order for non-ASCII file names, noted but not matched);
///   Symlink entries within the directory are resolved by following `metadata`; links to files are loaded, broken links are skipped (aligning with pi-307 symlink branch);
/// - File input: load only with `.md` suffix, ignore the rest;
/// - Path does not exist: silently skip; other IO failures: record the corresponding diagnostic and continue (no interruption, no panic);
/// - Recursive directory scanning is **not** this function's responsibility (pi-207 semantics are non-recursive; recursive discovery is a caller convention, determined by T-AP-05-3).
///
/// # Parameters
/// - `paths`: list of paths to load (CLI `--prompt-template` collected values + default directory, resolved by the caller before passing in).
///
/// # Returns
/// Aggregate of successful templates and non-interrupting diagnostics; both can be non-empty at the same time (failure of some files does not affect the rest).
pub async fn load_prompt_templates(paths: &[std::path::PathBuf]) -> LoadPromptTemplatesResult {
    let mut prompt_templates = Vec::new();
    let mut diagnostics = Vec::new();

    for path in paths {
        // Metadata probe: not found → silently skip; other failures → record `FileInfoFailed` diagnostic and continue
        // (pi silently swallows at two try/catch sites; zen makes it explicit per diagnostic-code semantics)
        let metadata = match tokio::fs::metadata(path).await {
            Ok(metadata) => metadata,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => {
                diagnostics.push(io_diagnostic(
                    PromptTemplateDiagnosticCode::FileInfoFailed,
                    format!("failed to stat {}: {e}", path.display()),
                    path,
                ));
                continue;
            }
        };
        if metadata.is_dir() {
            load_templates_from_dir(path, &mut prompt_templates, &mut diagnostics).await;
        } else if metadata.is_file()
            && path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".md"))
        {
            load_template_file(path, &mut prompt_templates, &mut diagnostics).await;
        }
        // Other types (socket/fifo, etc.) are silently ignored (same as pi else branch)
    }

    LoadPromptTemplatesResult {
        prompt_templates,
        diagnostics,
    }
}

/// Pure assembly of a single template file: parse frontmatter metadata from raw markdown and extract the body,
/// Construct `PromptTemplate`—separate IO from parsing so core semantics can be unit-tested without the filesystem (following the skills `build_skill` precedent).
///
/// # Parameters
/// - `content`: raw file content (UTF-8);
/// - `file_path`: template file path (`name` is derived from it and also used in error messages).
///
/// # Returns
/// Return the template on success; malformed frontmatter lines return a `ParseFailed` diagnostic (aggregated by the load entry point; this function does not log).
/// When there is no frontmatter or an unclosed `---`, treat it as "empty frontmatter + full text as body" (aligning with pi `parseFrontmatter`).
fn build_prompt_template(
    content: String,
    file_path: &std::path::Path,
) -> Result<PromptTemplate, PromptTemplateDiagnostic> {
    let path_str = file_path.display().to_string();

    // name: `strip_suffix(".md")` exactly replicates pi `basename(filePath).replace(/\.md$/, "")`.
    // Determination (where the stub comment left a blank): `file_stem` diverges from pi on ".md" files (pi yields "", `file_stem` yields
    // ".md"), so use `strip_suffix`; empty name (".md" file) or unable to derive → skip with `ParseFailed` diagnostic
    let name = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_suffix(".md"))
        .filter(|n| !n.is_empty())
        .map(|n| n.to_string())
        .ok_or_else(|| PromptTemplateDiagnostic {
            code: PromptTemplateDiagnosticCode::ParseFailed,
            message: format!("{path_str}: cannot derive template name"),
            path: path_str.clone(),
        })?;

    // go through the frontmatter module uniformly (T-AP-03-5 consolidation): CRLF normalization and full body trim are lifted into the primitive
    let parsed =
        frontmatter::parse_frontmatter(&content).map_err(|e| PromptTemplateDiagnostic {
            code: PromptTemplateDiagnosticCode::ParseFailed,
            message: format!("{path_str}: {e}"),
            path: path_str.clone(),
        })?;

    // two-key projection: description / argument-hint (generic key-value pairs, unknown keys ignored)
    let (mut description, mut argument_hint) = (String::new(), None);
    for (key, value) in parsed.frontmatter {
        match key.as_str() {
            "description" => description = value,
            // pi `frontmatter["argument-hint"] && {...}` truthiness semantics: empty hint = no hint
            "argument-hint" => {
                argument_hint = if value.is_empty() { None } else { Some(value) };
            }
            _ => {} // Unknown keys are skipped (pi `Record<string,string>` takes only two keys)
        }
    }

    // body already fully trimmed by the primitive (when frontmatter present; otherwise the whole text)
    let body = &parsed.body;

    // description fallback chain: frontmatter missing/empty → first non-empty body line, truncate to 60 chars + "..." (aligning with pi)
    if description.is_empty() {
        if let Some(first_line) = body.lines().find(|line| !line.trim().is_empty()) {
            description = first_line.chars().take(60).collect();
            if first_line.chars().count() > 60 {
                description.push_str("...");
            }
        }
    }

    Ok(PromptTemplate {
        name,
        description,
        argument_hint,
        content: body.to_string(),
    })
}

/// Placeholder substitution before template submission: the slash-command expansion path (T-AP-03-4) calls it after obtaining the template body and split arguments,
/// Produce the final prompt text submitted to the agent—"use the value if present, otherwise fall back to the default".
///
/// Supported forms (superset of pi-307; single-pass regex prevents placeholders inside argument values from being recursively replaced):
/// - `$1`, `$2`…: positional arguments (1-based; out-of-range becomes empty string; `$0` also becomes empty string, aligning with pi `args[-1] ?? ""`);
/// - `$@`, `$ARGUMENTS`: all arguments joined by a single space;
/// - `${N:-default}`: if the Nth argument is missing or empty, substitute `default`; otherwise use the argument value;
/// - `${@:-default}`, `${ARGUMENTS:-default}`: substitute `default` when all arguments are empty;
/// - `${@:N}`: join from the Nth argument to the end with single spaces (bash-style slicing; N is 1-based, 0 is treated as the first argument);
/// - `${@:N:L}`: take L arguments starting from the Nth.
///
/// Note: placeholder-like forms appearing in argument values and default text **will not** be substituted a second time (established behavior of single-pass scanning).
///
/// # Parameters
/// - `content`: template body (`PromptTemplate.content`);
/// - `args`: split argument list (produced by `parse_command_args`).
pub fn substitute_args(content: &str, args: &[String]) -> String {
    let all_args = args.join(" ");

    substitution_pattern()
        .replace_all(content, |caps: &Captures| {
            // Form 1: ${N:-default} / ${@:-default} / ${ARGUMENTS:-default}
            if let Some(target) = caps.get(1) {
                let value: Option<&str> = match target.as_str() {
                    "@" | "ARGUMENTS" => Some(all_args.as_str()),
                    digits => digits
                        .parse::<usize>()
                        .ok()
                        .and_then(|n| n.checked_sub(1)) // ${0:…} → no -1 index (pi args[-1] → undefined)
                        .and_then(|index| args.get(index).map(|s| s.as_str())),
                };
                // pi truthiness check: missing or empty string → default; non-empty string (including "0"/"false") → value
                return match value {
                    Some(v) if !v.is_empty() => v.to_string(),
                    _ => caps[2].to_string(),
                };
            }

            // Form 2: ${@:N} / ${@:N:L}
            if let Some(start_text) = caps.get(3) {
                // N is 1-based; ${@:0} is treated as the first argument (pi: start<0 clamped to 0)—`saturating_sub` rather than `checked_sub`,
                // the latter would wrongly route 0 into the "parse failure" branch and return an empty string; only parse failure (N overflow) returns empty string (pi large-number slice is likewise empty)
                let Some(start) = start_text
                    .as_str()
                    .parse::<usize>()
                    .ok()
                    .map(|n| n.saturating_sub(1))
                else {
                    return String::new();
                };
                let tail: &[String] = args.get(start..).unwrap_or(&[]);
                let selected: Vec<&str> =
                    match caps.get(4).map(|m| m.as_str().parse::<usize>().ok()) {
                        Some(Some(len)) => tail.iter().take(len).map(|s| s.as_str()).collect(),
                        _ => tail.iter().map(|s| s.as_str()).collect(), // No L or L overflow → clamp to the end
                    };
                return selected.join(" ");
            }

            // Form 3: $N / $@ / $ARGUMENTS
            if let Some(simple) = caps.get(5) {
                match simple.as_str() {
                    "ARGUMENTS" | "@" => return all_args.clone(),
                    digits => {
                        return digits
                            .parse::<usize>()
                            .ok()
                            .and_then(|n| n.checked_sub(1))
                            .and_then(|index| args.get(index).map(|s| s.as_str()))
                            .unwrap_or("") // $0 / out of bounds → empty string (pi `args[-1] ?? ""`)
                            .to_string();
                    }
                }
            }

            String::new() // Unreachable: three-way exhaustive
        })
        .into_owned()
}

/// Placeholder substitution modes (aligned with pi-307 three-way choice):
/// ① `${N|@|ARGUMENTS:-default}` ② `${@:N[:L]}` ③ `$N|$@|$ARGUMENTS`
fn substitution_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
           // [0-9] instead of \d: JS \d is ASCII-only, Rust \d defaults to Unicode — tightened to align with pi
           Regex::new(
               r"\$\{([0-9]+|ARGUMENTS|@):-([^}]*)\}|\$\{@:([0-9]+)(?::([0-9]+))?\}|\$(ARGUMENTS|@|[0-9]+)",
           )
           .expect("static substitution pattern must compile") // Static pattern; failure is a programming error (retry.rs precedent)
       })
}

/// Splitting of slash-command arguments: the command expansion path (T-AP-03-4), before feeding `/name followed by the raw argument string` to
/// before `substitute_args`, splitting a line of text into an argument array by shell intuition — "a quoted whole counts as one".
///
/// Behavior (aligned with pi-307): characters inside single/double quotes are preserved as-is (quotes themselves stripped, no escape support — same as pi);
/// Outside quotes, split on whitespace (`\s`, including space/tab/newline); skip empty segments; if a quote is unclosed, the remaining text goes into the current segment.
///
/// # Parameters
/// - `args_string`: raw argument string after the template name (may be an empty string, returns an empty array).
pub fn parse_command_args(args_string: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    // Current quote type (pi inQuote: string | null) — records the opening quote character,
    // Because "inside a quote, only the same quote closes it", a different quote is just a normal character
    let mut in_quote: Option<char> = None;

    for ch in args_string.chars() {
        if let Some(quote) = in_quote {
            if ch == quote {
                in_quote = None; // Same quote → close (strip the quote itself)
            } else {
                current.push(ch); // All characters inside quotes (including whitespace/different quotes/backslashes) are preserved as-is
            }
        } else if ch == '"' || ch == '\'' {
            in_quote = Some(ch); // Opening quote (the quote itself is not added to the buffer)
        } else if ch.is_whitespace() {
            if !current.is_empty() {
                args.push(std::mem::take(&mut current)); // Skip empty segments
            }
        } else {
            current.push(ch);
        }
    }

    // Final flush: if a quote is unclosed, the remaining text is already in current and naturally goes into the last segment
    if !current.is_empty() {
        args.push(current);
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: load_prompt_templates / load_template_file / load_templates_from_dir depend on tokio::fs
    // Real file IO; this file has no mock/test-double infrastructure (aligned with skills/load.rs unit-test precedent), so skipped by convention —
    // Integration verification should be added using a real temporary directory (tempfile + #[tokio::test]).
    // The following cover all pure functions: io_diagnostic / build_prompt_template / substitute_args / parse_command_args.

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    // ── io_diagnostic ────────────────────────────────────────

    #[test]
    fn test_io_diagnostic_uses_display_path() {
        let d = io_diagnostic(
            PromptTemplateDiagnosticCode::ReadFailed,
            "boom".to_string(),
            Path::new("/a/b.md"),
        );
        assert_eq!(
            d,
            PromptTemplateDiagnostic {
                code: PromptTemplateDiagnosticCode::ReadFailed,
                message: "boom".to_string(),
                path: "/a/b.md".to_string(),
            }
        );
    }

    // ── build_prompt_template ────────────────────────────────

    #[test]
    fn test_build_prompt_template_full_frontmatter() {
        let content = "---\ndescription: My desc\nargument-hint: <id>\n---\nBody text".to_string();
        let t = build_prompt_template(content, Path::new("/x/review.md")).unwrap();
        assert_eq!(t.name, "review");
        assert_eq!(t.description, "My desc");
        assert_eq!(t.argument_hint.as_deref(), Some("<id>"));
        assert_eq!(t.content, "Body text");
    }

    #[test]
    fn test_build_prompt_template_no_frontmatter_falls_back_to_first_line() {
        let t = build_prompt_template("Hello world".to_string(), Path::new("/x/t.md")).unwrap();
        assert_eq!(t.name, "t");
        assert_eq!(t.description, "Hello world");
        assert_eq!(t.argument_hint, None);
        assert_eq!(t.content, "Hello world");
    }

    #[test]
    fn test_build_prompt_template_name_strips_md_suffix_only() {
        // pi replace(/\.md$/) removes only the suffix: a.b.md → a.b
        let t = build_prompt_template("x".to_string(), Path::new("/x/a.b.md")).unwrap();
        assert_eq!(t.name, "a.b");
    }

    #[test]
    fn test_build_prompt_template_empty_name_rejected() {
        // A file named ".md" yields an empty name → ParseFailed (stub decision: do not produce an empty route name)
        let d = build_prompt_template("x".to_string(), Path::new("/x/.md")).unwrap_err();
        assert_eq!(d.code, PromptTemplateDiagnosticCode::ParseFailed);
        assert!(d.message.contains("cannot derive template name"));
    }

    #[test]
    fn test_build_prompt_template_missing_filename_rejected() {
        let r = build_prompt_template("x".to_string(), Path::new("/"));
        assert_eq!(
            r.unwrap_err().code,
            PromptTemplateDiagnosticCode::ParseFailed
        );
    }

    #[test]
    fn test_build_prompt_template_malformed_frontmatter_line_rejected() {
        let content = "---\nnot a kv line\n---\nbody".to_string();
        let d = build_prompt_template(content, Path::new("/x/t.md")).unwrap_err();
        assert_eq!(d.code, PromptTemplateDiagnosticCode::ParseFailed);
        assert!(d.message.contains("malformed frontmatter line"));
    }

    #[test]
    fn test_build_prompt_template_unclosed_frontmatter_treated_as_body() {
        // Unclosed --- → (None, full text): no field parsing; description falls back to first line "---"
        let content = "---\ndescription: x\nrest".to_string();
        let t = build_prompt_template(content, Path::new("/x/t.md")).unwrap();
        assert_eq!(t.description, "---");
        assert_eq!(t.content, "---\ndescription: x\nrest");
    }

    #[test]
    fn test_build_prompt_template_crlf_normalized() {
        let content = "---\r\ndescription: d\r\n---\r\nBody\r\nline2".to_string();
        let t = build_prompt_template(content, Path::new("/x/c.md")).unwrap();
        assert_eq!(t.description, "d");
        assert_eq!(t.content, "Body\nline2");
    }

    #[test]
    fn test_build_prompt_template_argument_hint_empty_is_none() {
        // pi truthiness semantics: empty hint = no hint
        let content = "---\nargument-hint:   \n---\nBody".to_string();
        let t = build_prompt_template(content, Path::new("/x/h.md")).unwrap();
        assert_eq!(t.argument_hint, None);
    }

    #[test]
    fn test_build_prompt_template_unknown_keys_ignored() {
        let content = "---\ndescription: d\nunknown: v\n---\nB".to_string();
        let t = build_prompt_template(content, Path::new("/x/t.md")).unwrap();
        assert_eq!(t.description, "d");
        assert_eq!(t.content, "B");
    }

    #[test]
    fn test_build_prompt_template_body_trimmed() {
        // pi parseFrontmatter trims the entire body: leading and trailing whitespace are both removed
        let content = "---\ndescription: d\n---\n\n\n  Body  \n\n".to_string();
        let t = build_prompt_template(content, Path::new("/x/t.md")).unwrap();
        assert_eq!(t.content, "Body");
    }

    #[test]
    fn test_build_prompt_template_value_with_colon() {
        // Split on the first ':': value may contain colons
        let content = "---\ndescription: a: b\n---\nB".to_string();
        let t = build_prompt_template(content, Path::new("/x/t.md")).unwrap();
        assert_eq!(t.description, "a: b");
    }

    #[test]
    fn test_build_prompt_template_description_fallback_truncates_at_60() {
        let content = format!("{}\nsecond", "A".repeat(70));
        let t = build_prompt_template(content, Path::new("/x/t.md")).unwrap();
        assert_eq!(t.description, format!("{}...", "A".repeat(60)));
    }

    #[test]
    fn test_build_prompt_template_description_fallback_exactly_60_no_ellipsis() {
        let line = "B".repeat(60);
        let content = format!("{line}\nx");
        let t = build_prompt_template(content, Path::new("/x/t.md")).unwrap();
        assert_eq!(t.description, line);
    }

    #[test]
    fn test_build_prompt_template_description_fallback_skips_blank_lines() {
        // Empty frontmatter + leading blank lines: fall back to the first non-empty line; body is trimmed as a whole
        let content = "---\n---\n\n   \nReal first\nMore".to_string();
        let t = build_prompt_template(content, Path::new("/x/t.md")).unwrap();
        assert_eq!(t.description, "Real first");
        assert_eq!(t.content, "Real first\nMore");
    }

    #[test]
    fn test_build_prompt_template_empty_description_value_falls_back() {
        let content = "---\ndescription:\n---\nReal first".to_string();
        let t = build_prompt_template(content, Path::new("/x/t.md")).unwrap();
        assert_eq!(t.description, "Real first");
    }

    // ── substitute_args ──────────────────────────────────────

    #[test]
    fn test_substitute_args_positional() {
        assert_eq!(substitute_args("$1 and $2", &args(&["a", "b"])), "a and b");
    }

    #[test]
    fn test_substitute_args_positional_out_of_range_yields_empty() {
        assert_eq!(substitute_args("$3", &args(&["a"])), "");
    }

    #[test]
    fn test_substitute_args_positional_zero_yields_empty() {
        // pi args[-1] ?? "": $0 is also an empty string
        assert_eq!(substitute_args("$0", &args(&["a"])), "");
    }

    #[test]
    fn test_substitute_args_positional_multi_digit_greedy() {
        // $10 greedily matches two digits: the 10th argument, not $1 + "0"
        let ten = args(&["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]);
        assert_eq!(substitute_args("$10", &ten), "10");
    }

    #[test]
    fn test_substitute_args_at_and_arguments_join_all() {
        assert_eq!(
            substitute_args("$@|$ARGUMENTS", &args(&["x", "y"])),
            "x y|x y"
        );
    }

    #[test]
    fn test_substitute_args_default_when_arg_missing() {
        assert_eq!(substitute_args("${1:-fb}", &args(&[])), "fb");
    }

    #[test]
    fn test_substitute_args_default_when_arg_empty() {
        assert_eq!(substitute_args("${1:-fb}", &args(&[""])), "fb");
    }

    #[test]
    fn test_substitute_args_default_skipped_when_value_truthy() {
        // pi truthiness check: any non-empty string is used as the value (including strings like "0")
        assert_eq!(substitute_args("${1:-fb}", &args(&["0"])), "0");
        assert_eq!(substitute_args("${2:-fb}", &args(&["x", "v"])), "v");
    }

    #[test]
    fn test_substitute_args_default_for_all_args_forms() {
        assert_eq!(substitute_args("${@:-none}", &args(&[])), "none");
        assert_eq!(substitute_args("${ARGUMENTS:-none}", &args(&[])), "none");
        assert_eq!(substitute_args("${@:-none}", &args(&["a"])), "a");
    }

    #[test]
    fn test_substitute_args_default_text_not_recursively_substituted() {
        // Single-pass scan: default is used only when the value is missing, and placeholder forms inside it stay literal (not recursively replaced with "b")
        assert_eq!(substitute_args("${9:-$2}", &args(&["a", "b"])), "$2");
    }

    #[test]
    fn test_substitute_args_value_not_recursively_substituted() {
        assert_eq!(substitute_args("$1", &args(&["$2"])), "$2");
    }

    #[test]
    fn test_substitute_args_slice_from_n() {
        assert_eq!(substitute_args("${@:2}", &args(&["a", "b", "c"])), "b c");
    }

    #[test]
    fn test_substitute_args_slice_with_length() {
        assert_eq!(substitute_args("${@:2:1}", &args(&["a", "b", "c"])), "b");
    }

    #[test]
    fn test_substitute_args_slice_zero_clamps_to_all_args() {
        // ${@:0} is treated as the first argument (pi: start<0 clamps to 0) — previously returned an empty string due to early return in checked_sub; fixed
        assert_eq!(substitute_args("${@:0}", &args(&["a", "b"])), "a b");
    }

    #[test]
    fn test_substitute_args_slice_beyond_range_yields_empty() {
        assert_eq!(substitute_args("${@:5}", &args(&["a"])), "");
    }

    #[test]
    fn test_substitute_args_slice_length_zero_yields_empty() {
        assert_eq!(substitute_args("${@:1:0}", &args(&["a", "b"])), "");
    }

    #[test]
    fn test_substitute_args_non_placeholder_dollar_untouched() {
        // If $ is not followed by [0-9]|@|ARGUMENTS, it does not match and stays as-is; empty arguments for $ARGUMENTS become an empty string
        assert_eq!(substitute_args("cost $total", &args(&[])), "cost $total");
        assert_eq!(substitute_args("a $ARGUMENTS b", &args(&[])), "a  b");
    }

    // ── parse_command_args ───────────────────────────────────

    #[test]
    fn test_parse_command_args_empty_returns_empty() {
        assert!(parse_command_args("").is_empty());
    }

    #[test]
    fn test_parse_command_args_splits_on_any_whitespace() {
        assert_eq!(parse_command_args("a b\tc\nd"), vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn test_parse_command_args_empty_segments_skipped() {
        assert_eq!(parse_command_args("  a   b  "), vec!["a", "b"]);
    }

    #[test]
    fn test_parse_command_args_double_quote_groups() {
        assert_eq!(
            parse_command_args("run \"two words\" now"),
            vec!["run", "two words", "now"]
        );
    }

    #[test]
    fn test_parse_command_args_single_quote_groups() {
        assert_eq!(parse_command_args("'a b' c"), vec!["a b", "c"]);
    }

    #[test]
    fn test_parse_command_args_other_quote_inside_is_literal() {
        assert_eq!(parse_command_args("\"a'b\""), vec!["a'b"]);
    }

    #[test]
    fn test_parse_command_args_unclosed_quote_keeps_tail() {
        // Unclosed quote: whitespace inside the quote is not split; the remaining text goes entirely into the last segment
        assert_eq!(parse_command_args("a \"bc d"), vec!["a", "bc d"]);
    }

    #[test]
    fn test_parse_command_args_adjacent_quoted_text_concatenates() {
        assert_eq!(parse_command_args("abc\"def\"ghi"), vec!["abcdefghi"]);
    }
}
