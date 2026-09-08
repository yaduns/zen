use crate::frontmatter;
use crate::skills::skill::Skill;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::path::PathBuf;

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillDiagnosticCode {
    /// fileInfo/canonicalPath failed (not not_found)
    FileInfoFailed,
    /// listDir failed
    ListFailed,
    /// readTextFile failed
    ReadFailed,
    /// frontmatter YAML parsing failed
    ParseFailed,
    /// name/description metadata validation failed
    InvalidMetadata,
}

pub struct SkillDiagnostic {
    /// diagnostic code
    pub code: SkillDiagnosticCode,
    /// human-readable message
    pub message: String,
    /// associated path
    pub path: String,
}

struct FileInfo {
    /// address path
    path: PathBuf,
    /// directory entry name
    name: String,
    /// file/directory/symlink
    kind: FileKind,
}

enum FileKind {
    File,
    Directory,
    Symlink,
}

#[derive(Debug, thiserror::Error)]
enum FileAccessError {
    /// Path does not exist.
    #[error("path not found: {}", path.display())]
    NotFound { path: PathBuf },
    /// Other I/O error; `#[from]` enables automatic conversion via `?`.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Aggregated result of `load_skills`.
pub struct LoadSkillsResult {
    /// Skills loaded successfully
    pub skills: Vec<Skill>,
    /// Warnings produced during loading (non-fatal)
    pub diagnostics: Vec<SkillDiagnostic>,
}

#[derive(Default)]
struct SkillFrontmatter {
    name: Option<String>,
    description: Option<String>,
    disable_model_invocation: bool,
}

/// Load skills from the given directories (**only `SKILL.md`**).
///
/// Recursively traverse with `ignore::WalkBuilder` (gitignore-aware: honors `.gitignore`/`.ignore`/`.fdignore`, skips hidden entries,
/// skips `node_modules`; `follow_links(true)` follows symlinked directories with built-in cycle detection), collecting all entries named `SKILL.md`.
/// For each entry, read asynchronously (`tokio::fs`) and call `load_skill_from_file`. Missing input directories are silently skipped; invalid skills produce diagnostics (no panic, no abort).
///
/// `async fn` (matches zen's tokio runtime + pi's async nature): `ignore::WalkBuilder` traversal is synchronous and fast,
/// a brief synchronous traversal is acceptable; wrap in `tokio::task::spawn_blocking` if stricter behavior is needed. File reads use async `tokio::fs`.
/// # Parameters
/// - `dirs`: list of directories to scan (precomputed and passed in by T-AP-03-3; `load_skills` does no upward search).
/// # Returns
/// Aggregated `skills` and `diagnostics`.
pub async fn load_skills(dirs: &[PathBuf]) -> LoadSkillsResult {
    let mut skills = Vec::new();
    let mut diagnostics = Vec::new();

    for dir in dirs {
        // Silently skip missing input directories
        if !dir.is_dir() {
            continue;
        }
        let walker = ignore::WalkBuilder::new(dir)
            .follow_links(true) // Follow symlinked directories with built-in cycle detection
            .hidden(true) // Skip hidden directories
            .build();
        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue, // Failed to traverse a single entry; skip without aborting
            };
            // Only take files named SKILL.md
            if entry.file_type().is_some_and(|ft| ft.is_file()) && entry.file_name() == "SKILL.md" {
                let (skill, mut diags) = load_skill_from_file(entry.path()).await;
                if let Some(skill) = skill {
                    skills.push(skill);
                }
                diagnostics.append(&mut diags);
            }
        }
    }

    LoadSkillsResult {
        skills,
        diagnostics,
    }
}

/// Read the raw content of SKILL.md. IO errors are mapped to ReadFailed diagnostics.
async fn read_skill_file(file_path: &Path) -> Result<String, SkillDiagnostic> {
    tokio::fs::read_to_string(file_path)
        .await
        .map_err(|e| SkillDiagnostic {
            code: SkillDiagnosticCode::ReadFailed,
            message: format!("failed to read {}: {e}", file_path.display()),
            path: file_path.display().to_string(),
        })
}

/// Derive the skill name from the parent directory name of the path. Returns `None` when derivation is impossible (no parent / directory name not UTF-8),
/// The caller decides how to handle it (record a diagnostic, discard).
fn default_name_from_path(path: &Path) -> Option<String> {
    path.parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .map(|s| s.to_string())
}

/// Pure function: parse frontmatter + validate metadata + construct Skill.
/// Returns `Skill` on success; on failure (ParseFailed / InvalidMetadata) returns a diagnostic.
/// Being a synchronous pure function, it can be unit-tested directly, with no async runtime or filesystem needed.
fn build_skill(content: String, file_path: &Path) -> Result<Skill, SkillDiagnostic> {
    let path_str = file_path.display().to_string();
    let (frontmatter, body) = parse_frontmatter(&content).map_err(|msg| SkillDiagnostic {
        code: SkillDiagnosticCode::ParseFailed,
        message: format!("{path_str}: {msg}"),
        path: path_str.clone(),
    })?;

    let name = match frontmatter.name {
       Some(name) => name,
       None => match default_name_from_path(file_path) {
           Some(name) => name,
           None => {
               return Err(SkillDiagnostic {
                   code: SkillDiagnosticCode::InvalidMetadata,
                   message: format!("{path_str}: cannot derive skill name (missing frontmatter name and invalid path)"),
                   path: path_str,
               })
           }
       },
   };
    let description = frontmatter.description.ok_or_else(|| SkillDiagnostic {
        code: SkillDiagnosticCode::InvalidMetadata,
        message: format!("{path_str}: missing description"),
        path: path_str.clone(),
    })?; // ? Return early from build_skill immediately upon encountering Err

    Ok(Skill {
        name,
        description,
        content: body,
        file_path: path_str,
        disable_model_invocation: frontmatter.disable_model_invocation,
    })
}

/// Load from a single `SKILL.md` file — explicit `.md` skill path (CLI `--skill` pointing to a file, T-AP-03-3 single-file assembly)
/// Called when loading a single file: read raw text (`tokio::fs`) → parse frontmatter → validate metadata → construct `Skill`.
/// Shares the same assembly path as `load_skills` (directory traversal); returns when description is missing or metadata validation fails
/// `(None, diagnostics)`。
/// Returns tuple `(Option<Skill>, Vec<SkillDiagnostic>)` (single use site; no named type).
pub async fn load_skill_from_file(file_path: &Path) -> (Option<Skill>, Vec<SkillDiagnostic>) {
    match read_skill_file(file_path)
        .await
        .and_then(|c| build_skill(c, file_path))
    {
        Ok(skill) => (Some(skill), Vec::new()),
        Err(diag) => (None, vec![diag]),
    }
}

/// Parse the frontmatter of SKILL.md (minimal YAML subset, uniformly via the `frontmatter` module) and the body.
///
/// No YAML crate: only supports the single-line `key: value` form; recognizes name (String), description (String),
/// disable-model-invocation (bool); unknown keys are ignored.
/// Returns empty frontmatter and the full text when the input does not start with `---` or lacks a closing `---`.
/// A malformed `key: value` line returns `Err(malformed line / message)` (converted by the caller into a `ParseFailed` diagnostic).
/// Behavior aligned with pi (T-AP-03-5 convergence): CRLF normalization, comment-line ignoring, and quote stripping are handled by the common layer.
fn parse_frontmatter(content: &str) -> Result<(SkillFrontmatter, String), String> {
    let parsed = frontmatter::parse_frontmatter(content).map_err(|e| e.to_string())?;

    let mut frontmatter = SkillFrontmatter::default();
    for (key, value) in parsed.frontmatter {
        match key.as_str() {
            "name" => frontmatter.name = Some(value),
            "description" => frontmatter.description = Some(value),
            "disable-model-invocation" => frontmatter.disable_model_invocation = value == "true",
            _ => {} // Skip unknown keys
        }
    }

    Ok((frontmatter, parsed.body))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    // Note: Skill / SkillDiagnostic do not implement Debug, so .unwrap()/.unwrap_err() cannot be used;
    // use match uniformly to extract the Ok/Err branches.
    // load_skills / load_skill_from_file / read_skill_file depend on real file IO (tokio::fs + directory traversal)
    // and load_skills is async, so they are skipped per this file's test-scope convention; their core validation logic (build_skill) is already covered by the pure-function tests.

    fn expect_skill_ok(r: Result<Skill, SkillDiagnostic>) -> Skill {
        match r {
            Ok(s) => s,
            Err(_) => panic!("expected Ok"),
        }
    }

    fn expect_skill_err(r: Result<Skill, SkillDiagnostic>) -> SkillDiagnostic {
        match r {
            Err(d) => d,
            Ok(_) => panic!("expected Err"),
        }
    }

    fn expect_parse_fm_err(r: Result<(SkillFrontmatter, String), String>) -> String {
        match r {
            Err(e) => e,
            Ok(_) => panic!("expected Err"),
        }
    }

    // ─── parse_frontmatter ────────────────────────────────────────
    #[test]
    fn test_parse_frontmatter_no_frontmatter() {
        let (fm, body) = parse_frontmatter("just text").unwrap();
        assert!(fm.name.is_none());
        assert!(fm.description.is_none());
        assert_eq!(body, "just text");
    }

    #[test]
    fn test_parse_frontmatter_with_fields() {
        let content = "---\nname: foo\ndescription: bar\n---\nbody text";
        let (fm, body) = parse_frontmatter(content).unwrap();
        assert_eq!(fm.name.as_deref(), Some("foo"));
        assert_eq!(fm.description.as_deref(), Some("bar"));
        assert_eq!(body, "body text");
    }

    #[test]
    fn test_parse_frontmatter_malformed_errors() {
        let err = expect_parse_fm_err(parse_frontmatter("---\nno colon here\n---\nbody"));
        assert!(err.contains("malformed"));
    }

    #[test]
    fn test_parse_frontmatter_comment_lines_ignored() {
        // Aligned with pi: comment-only frontmatter does not error and returns empty metadata (the old zen implementation reported malformed for comment lines)
        let (fm, body) = parse_frontmatter("---\n# just a comment\n---\nBody").unwrap();
        assert!(fm.name.is_none());
        assert!(fm.description.is_none());
        assert_eq!(body, "Body");
    }

    #[test]
    fn test_parse_frontmatter_quotes_stripped() {
        let (fm, _) =
            parse_frontmatter("---\nname: \"foo\"\ndescription: 'bar'\n---\nBody").unwrap();
        assert_eq!(fm.name.as_deref(), Some("foo"));
        assert_eq!(fm.description.as_deref(), Some("bar"));
    }

    #[test]
    fn test_parse_frontmatter_crlf_normalized() {
        // Aligned with pi: CRLF normalization (the old implementation did not normalize on the skills side; fixed after convergence)
        let (fm, body) =
            parse_frontmatter("---\r\nname: foo\r\n---\r\nLine one\r\nLine two").unwrap();
        assert_eq!(fm.name.as_deref(), Some("foo"));
        assert_eq!(body, "Line one\nLine two");
    }

    // ─── default_name_from_path ───────────────────────────────────
    #[test]
    fn test_default_name_from_path_normal() {
        let name = default_name_from_path(Path::new("/skills/foo/SKILL.md"));
        assert_eq!(name.as_deref(), Some("foo"));
    }

    #[test]
    fn test_default_name_from_path_none_for_relative_without_dir() {
        let name = default_name_from_path(Path::new("SKILL.md"));
        assert!(name.is_none());
    }

    // ─── build_skill ──────────────────────────────────────────────
    fn sample_path() -> PathBuf {
        PathBuf::from("/skills/foo/SKILL.md")
    }

    #[test]
    fn test_build_skill_valid() {
        let content = "---\nname: foo\ndescription: does foo\n---\nbody content";
        let skill = expect_skill_ok(build_skill(content.to_string(), &sample_path()));
        assert_eq!(skill.name, "foo");
        assert_eq!(skill.description, "does foo");
        assert_eq!(skill.content, "body content");
        assert!(!skill.disable_model_invocation);
        assert_eq!(skill.file_path, "/skills/foo/SKILL.md");
    }

    #[test]
    fn test_build_skill_name_from_parent_dir() {
        let content = "---\ndescription: does foo\n---\nbody";
        let skill = expect_skill_ok(build_skill(content.to_string(), &sample_path()));
        assert_eq!(skill.name, "foo"); // Missing name in frontmatter → derive from the parent directory name
    }

    #[test]
    fn test_build_skill_disable_model_invocation_true() {
        let content = "---\nname: foo\ndescription: d\ndisable-model-invocation: true\n---\nbody";
        let skill = expect_skill_ok(build_skill(content.to_string(), &sample_path()));
        assert!(skill.disable_model_invocation);
    }

    #[test]
    fn test_build_skill_missing_description_errors() {
        let content = "---\nname: foo\n---\nbody";
        let diag = expect_skill_err(build_skill(content.to_string(), &sample_path()));
        assert!(matches!(diag.code, SkillDiagnosticCode::InvalidMetadata));
        assert!(diag.message.contains("missing description"));
        assert_eq!(diag.path, "/skills/foo/SKILL.md");
    }

    #[test]
    fn test_build_skill_malformed_frontmatter_errors() {
        let content = "---\nbad line\n---\nbody";
        let diag = expect_skill_err(build_skill(content.to_string(), &sample_path()));
        assert!(matches!(diag.code, SkillDiagnosticCode::ParseFailed));
    }

    #[test]
    fn test_build_skill_name_derivation_failure_errors() {
        // No frontmatter and no derivable name from the path (no parent-directory filename) → InvalidMetadata
        let diag = expect_skill_err(build_skill(
            "no frontmatter".to_string(),
            Path::new("SKILL.md"),
        ));
        assert!(matches!(diag.code, SkillDiagnosticCode::InvalidMetadata));
        assert!(diag.message.contains("cannot derive skill name"));
    }

    // ─── SkillDiagnosticCode serde ────────────────────────────────
    #[test]
    fn test_skill_diagnostic_code_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&SkillDiagnosticCode::ParseFailed).unwrap(),
            "\"parse_failed\""
        );
        assert_eq!(
            serde_json::to_string(&SkillDiagnosticCode::InvalidMetadata).unwrap(),
            "\"invalid_metadata\""
        );
    }
}
