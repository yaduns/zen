use std::path::{Component, Path, PathBuf};

/// Optional configuration for path normalization.
///
/// Corresponds to Pi's `PathInputOptions`. `Default` semantics align with TS's
/// `expandTilde ?? true`: `expand_tilde` defaults to `true`; other options are off by default.
pub struct PathInputOptions {
    /// Trim leading/trailing whitespace before normalization (corresponds to `options.trim`).
    pub trim: bool,
    /// Expand leading `~`; defaults to `true` (aligned with `expandTilde ?? true`).
    pub expand_tilde: bool,
    /// Home directory used for `~` expansion; when `None`, fall back to `home::home_dir()` (overridable in tests).
    pub home_dir: Option<PathBuf>,
    /// Strip a leading `@` (CLI `@file` path syntax; strip once).
    pub strip_at_prefix: bool,
    /// Normalize Unicode space variants (`\u{00A0}`, `\u{2000}`-`\u{200A}`, `\u{202F}`, `\u{205F}`, `\u{3000}`) to regular spaces.
    pub normalize_unicode_spaces: bool,
}

impl Default for PathInputOptions {
    fn default() -> Self {
        Self {
            trim: false,
            expand_tilde: true,
            home_dir: None,
            strip_at_prefix: false,
            normalize_unicode_spaces: false,
        }
    }
}

/// Path utility error type.
#[derive(Debug, thiserror::Error)]
pub enum PathError {
    /// Starts with `file://` but cannot be converted to a local path (invalid percent-encoding, non-empty host, etc.).
    #[error("invalid file URL `{0}`: {1}")]
    InvalidFileUrl(String, String),
    /// `~` expansion is needed, but `home_dir` was not injected and `home::home_dir()` returned `None`.
    #[error("cannot determine home directory for `~` expansion")]
    NoHomeDir,
}

/// Normalize the path string.
///
/// Performs the following in order according to `options`: trim → normalize Unicode spaces → strip leading `@` → expand `~` →
/// Converts a `file://` URL to a local path. Returns the normalized string, **not guaranteed to be absolute**, and does not access the filesystem.
///
/// tilde expansion matches only a bare `~` and the `~/` prefix (win32 also `~\`); names like `~draft.md`
/// tilde-prefixed filenames stay literal (aligned with Pi's branch condition).
///
/// `file://` conversion triggers only when the input starts with `file://`; literal `%` sequences in native paths are left as-is.
///
/// # Errors
/// - Expand `~` but home unavailable → `PathError::NoHomeDir`.
/// - Invalid `file://` URL → `PathError::InvalidFileUrl`.
///
/// # Consumers
/// after refactor `zen-core/tools/path_utils.rs` (the original `expand_path` logic is merged into this function); future
/// T-AP-02-1（config）、T-AP-05-3（resource-loader）。
pub fn normalize_path(input: &str, options: &PathInputOptions) -> Result<String, PathError> {
    let mut s = if options.trim {
        input.trim().to_string()
    } else {
        input.to_string()
    };

    if options.normalize_unicode_spaces {
        s = s
            .chars()
            .map(|c| if is_unicode_space(c) { ' ' } else { c })
            .collect();
    }

    if options.strip_at_prefix && s.starts_with('@') {
        s = s[1..].to_string();
    }

    if options.expand_tilde {
        let is_home_self = s == "~";
        let is_home_prefix = s.starts_with("~/") || (cfg!(windows) && s.starts_with("~\\"));
        if is_home_self || is_home_prefix {
            let home = match &options.home_dir {
                Some(h) => h.clone(),
                None => home::home_dir().ok_or(PathError::NoHomeDir)?,
            };
            if is_home_self {
                return Ok(home.to_string_lossy().into_owned());
            }
            let rest = &s[2..];
            return Ok(home.join(rest).to_string_lossy().into_owned());
        }
    }

    if s.starts_with("file://") {
        let path = file_url_to_path(&s)?;
        return Ok(path.to_string_lossy().into_owned());
    }

    Ok(s)
}

/// Resolve the input path to an absolute path.
///
/// First normalize `input` with `options`, then normalize `base_dir` with default options (`base_dir` itself may be
/// a `file://` URL); if the normalized `input` is absolute, normalize it directly, otherwise join it with `base_dir`
/// and normalize after joining (lexically collapses `.`/`..`, no filesystem access, matching `node:path.resolve` semantics).
///
/// # Errors
/// Propagates `normalize_path` errors (`NoHomeDir` / `InvalidFileUrl`).
///
/// # Consumers
/// after refactor `zen-core/tools/path_utils.rs::resolve_to_cwd`; future T-AP-01-1 (AgentSession),
/// T-AP-04-7（render-utils）。
pub fn resolve_path(
    input: &str,
    base_dir: &Path,
    options: &PathInputOptions,
) -> Result<PathBuf, PathError> {
    let normalized = normalize_path(input, options)?;
    let normalized_base =
        normalize_path(&base_dir.to_string_lossy(), &PathInputOptions::default())?;
    let p = Path::new(&normalized);
    let base = Path::new(&normalized_base);
    let joined = if p.is_absolute() {
        p.to_path_buf()
    } else {
        base.join(p)
    };
    Ok(lexical_normalize(&joined))
}

/// Resolve a path to its canonical (real) form, following symlinks.
///
/// On resolution failure (missing target, dangling link, etc.) **fall back to returning the original path**, so callers never crash
/// due to a missing filesystem entry (aligned with Pi `canonicalizePath`).
///
/// # Consumers
/// Future T-AP-05-3 (resource-loader), T-AP-03-3 (skills), T-AP-11-* (interactive components).
pub fn canonicalize_path(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Determine whether a value is a local path.
///
/// `npm:` / `git:` / `github:` / `http:` / `https:` / `ssh:` prefixes are treated as non-local (remote packages/URLs);
/// bare names, relative paths, and `file:` URLs are treated as local (`file:` is resolved by `resolve_path`). Trim before checking.
///
/// # Consumers
/// Future T-AP-08-3 (main `resolveCliPaths`), T-AP-05-3 (resource-loader).
pub fn is_local_path(value: &str) -> bool {
    let trimmed = value.trim();
    !(trimmed.starts_with("npm:")
        || trimmed.starts_with("git:")
        || trimmed.starts_with("github:")
        || trimmed.starts_with("http:")
        || trimmed.starts_with("https:")
        || trimmed.starts_with("ssh:"))
}

/// Compute the path of `file_path` relative to `cwd`.
///
/// Both are first lexically normalized via `resolve_path`. If `file_path` is inside `cwd`, return `Some(relative path)`
/// (`Some(".")` when equal; dot-leading components like `..config` are preserved); if it escapes `cwd` (including parent-directory
/// traversal) or crosses roots, return `None` (aligned with Pi `getCwdRelativePath`'s `isInsideCwd` check, implemented equivalently
/// via `Path::strip_prefix`).
///
/// # Consumers
/// Future T-AP-11-* (interactive-mode).
pub fn get_cwd_relative_path(file_path: &Path, cwd: &Path) -> Result<Option<String>, PathError> {
    let resolved_cwd = resolve_path(&cwd.to_string_lossy(), cwd, &PathInputOptions::default())?;
    let resolved = resolve_path(
        &file_path.to_string_lossy(),
        &resolved_cwd,
        &PathInputOptions::default(),
    )?;
    Ok(match resolved.strip_prefix(&resolved_cwd) {
        Ok(rest) => {
            if rest.as_os_str().is_empty() {
                Some(".".to_string())
            } else {
                Some(rest.to_string_lossy().into_owned())
            }
        }
        Err(_) => None,
    })
}

/// Return a path for display.
///
/// If `file_path` is inside `cwd`, return the relative path, otherwise the absolute path; separators are normalized to
/// forward slash `/` (for cross-platform display, aligned with Pi `formatPathRelativeToCwdOrAbsolute`).
///
/// # Consumers
/// Future T-AP-04-x (app-layer read tool label display).
pub fn format_path_relative_to_cwd_or_absolute(
    file_path: &Path,
    cwd: &Path,
) -> Result<String, PathError> {
    let absolute = resolve_path(
        &file_path.to_string_lossy(),
        cwd,
        &PathInputOptions::default(),
    )?;
    let rel = get_cwd_relative_path(&absolute, cwd)?;
    let s = rel.unwrap_or_else(|| absolute.to_string_lossy().into_owned());
    #[cfg(windows)]
    let s = s.replace('\\', "/");
    Ok(s)
}

/// Determine whether a char is a Unicode space variant that needs normalizing.
///
/// Matches Pi's `UNICODE_SPACES` regex `[\u00A0\u2000-\u200A\u202F\u205F\u3000]`.
fn is_unicode_space(c: char) -> bool {
    matches!(c, '\u{00A0}' | '\u{202F}' | '\u{205F}' | '\u{3000}')
        || ('\u{2000}'..='\u{200A}').contains(&c)
}

/// Convert a `file://` URL to a local path.
///
/// Parsed with the `url` crate (percent-decoding, error on invalid URL, Windows drive-letter mapping),
/// semantics match node `fileURLToPath`. Called by `normalize_path` only when the input starts with `file://`.
fn file_url_to_path(input: &str) -> Result<PathBuf, PathError> {
    let url = url::Url::parse(input)
        .map_err(|e| PathError::InvalidFileUrl(input.to_string(), e.to_string()))?;
    let path = url
        .to_file_path()
        .map_err(|_| PathError::InvalidFileUrl(input.to_string(), "not a file URL".into()))?;
    // Align with node `fileURLToPath`: the URL path must be valid UTF-8 (decodeURIComponent throws
    // URIError on invalid sequences); the url crate's to_file_path is lenient about invalid bytes, so validate here.
    if path.to_str().is_none() {
        return Err(PathError::InvalidFileUrl(
            input.to_string(),
            "path contains invalid UTF-8".into(),
        ));
    }
    Ok(path)
}

/// Lexically normalize a path: collapse `.`, resolve `..` (no filesystem access).
///
/// Aligns with `node:path.resolve` normalization semantics: `..` in an absolute path never escapes the root
/// (`/a/../b` → `/b`, `/..` → `/`); `Path::join`/`PathBuf::from` do not collapse `..`,
/// so resolve_path must normalize explicitly here.
fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                // Pop the previous normal component; `..` is ignored at the root or when empty (aligned with node resolve)
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn home() -> PathBuf {
        home::home_dir().expect("home dir available in test env")
    }

    // ── normalize_path: tilde expansion ────────────────────────────────

    #[test]
    fn test_normalize_path_tilde_self() {
        assert_eq!(
            normalize_path("~", &Default::default()).unwrap(),
            home().to_string_lossy()
        );
    }

    #[test]
    fn test_normalize_path_tilde_prefix() {
        let r = normalize_path("~/file.txt", &Default::default()).unwrap();
        assert_eq!(r, home().join("file.txt").to_string_lossy());
    }

    #[test]
    fn test_normalize_path_tilde_filename_literal() {
        // `~draft.md` is not `~`/`~/`, keep it literal (aligned with pi's tilde-prefixed filenames)
        assert_eq!(
            normalize_path("~draft.md", &Default::default()).unwrap(),
            "~draft.md"
        );
    }

    #[test]
    fn test_normalize_path_home_dir_override() {
        let opts = PathInputOptions {
            home_dir: Some(PathBuf::from("/fake/home")),
            ..Default::default()
        };
        assert_eq!(normalize_path("~", &opts).unwrap(), "/fake/home");
        assert_eq!(normalize_path("~/x", &opts).unwrap(), "/fake/home/x");
    }

    #[test]
    fn test_normalize_path_expand_tilde_disabled() {
        let opts = PathInputOptions {
            expand_tilde: false,
            home_dir: Some(PathBuf::from("/fake/home")),
            ..Default::default()
        };
        assert_eq!(normalize_path("~", &opts).unwrap(), "~");
        assert_eq!(normalize_path("~/x", &opts).unwrap(), "~/x");
    }

    // ── normalize_path: trim / @ stripping / Unicode spaces ──────────────

    #[test]
    fn test_normalize_path_trim() {
        let opts = PathInputOptions {
            trim: true,
            ..Default::default()
        };
        assert_eq!(normalize_path("  /foo/bar  ", &opts).unwrap(), "/foo/bar");
        // no trim by default
        assert_eq!(
            normalize_path("  /foo/bar  ", &Default::default()).unwrap(),
            "  /foo/bar  "
        );
    }

    #[test]
    fn test_normalize_path_strip_at_prefix() {
        let opts = PathInputOptions {
            strip_at_prefix: true,
            ..Default::default()
        };
        assert_eq!(normalize_path("@/foo", &opts).unwrap(), "/foo");
        // strip only once
        assert_eq!(normalize_path("@@/foo", &opts).unwrap(), "@/foo");
        // no stripping when disabled
        assert_eq!(
            normalize_path("@/foo", &Default::default()).unwrap(),
            "@/foo"
        );
    }

    #[test]
    fn test_normalize_path_unicode_spaces() {
        let opts = PathInputOptions {
            normalize_unicode_spaces: true,
            ..Default::default()
        };
        assert_eq!(
            normalize_path("/a\u{00A0}b\u{202F}c\u{3000}d\u{2000}e\u{200A}f", &opts).unwrap(),
            "/a b c d e f"
        );
        // keep as-is when disabled
        assert_eq!(
            normalize_path("a\u{00A0}b", &Default::default()).unwrap(),
            "a\u{00A0}b"
        );
    }

    #[test]
    fn test_normalize_path_at_then_tilde() {
        let opts = PathInputOptions {
            strip_at_prefix: true,
            home_dir: Some(PathBuf::from("/fake/home")),
            ..Default::default()
        };
        // after @ stripping it becomes `~/...`, then tilde expansion applies (aligned with pi expandPath's option combination)
        assert_eq!(normalize_path("@~/dir", &opts).unwrap(), "/fake/home/dir");
    }

    // ── normalize_path / resolve_path：file:// URL ────────────────

    #[test]
    fn test_normalize_path_file_url() {
        assert_eq!(
            normalize_path("file:///home/user/config.toml", &Default::default()).unwrap(),
            "/home/user/config.toml"
        );
    }

    #[test]
    fn test_resolve_path_accepts_file_url_with_spaces() {
        // percent-decoding: `file%20with%20spaces.txt` → `file with spaces.txt`
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("file with spaces.txt");
        let url = url::Url::from_file_path(&file).unwrap().to_string();
        let r = resolve_path(&url, &dir.path().join("base"), &Default::default()).unwrap();
        assert_eq!(r, file);
    }

    #[test]
    fn test_resolve_path_invalid_file_url_errors() {
        let r = resolve_path("file:///%E0%A4%A", Path::new("/tmp"), &Default::default());
        assert!(matches!(r, Err(PathError::InvalidFileUrl(..))));
    }

    #[test]
    fn test_resolve_path_preserves_literal_percent_in_native_paths() {
        // literal `%` in native paths is not URL-decoded
        let dir = tempfile::tempdir().unwrap();
        for name in ["report%2026.md", "foo%2Fbar", "malformed%A.md"] {
            let r = resolve_path(name, &dir.path().join("base"), &Default::default()).unwrap();
            assert_eq!(r, dir.path().join("base").join(name));
        }
    }

    // ── resolve_path: absolute/relative/lexical normalization ───────────

    #[test]
    fn test_resolve_path_absolute_passthrough() {
        let r = resolve_path("/etc/config", Path::new("/some/cwd"), &Default::default()).unwrap();
        assert_eq!(r, PathBuf::from("/etc/config"));
    }

    #[test]
    fn test_resolve_path_relative_against_base() {
        let r = resolve_path(
            "subdir/file.txt",
            Path::new("/some/cwd"),
            &Default::default(),
        )
        .unwrap();
        assert_eq!(r, PathBuf::from("/some/cwd/subdir/file.txt"));
    }

    #[test]
    fn test_resolve_path_file_url_base_dir() {
        // base_dir itself may be a file:// URL (aligned with pi test "resolves relative paths against the base directory")
        let cwd = PathBuf::from("/tmp/pi-paths-cwd");
        let base_url = url::Url::from_directory_path(&cwd).unwrap().to_string();
        let r = resolve_path("subdir/file.txt", Path::new(&base_url), &Default::default()).unwrap();
        assert_eq!(r, cwd.join("subdir/file.txt"));
    }

    #[test]
    fn test_resolve_path_normalizes_dot_dot() {
        // lexically normalize `..` (aligned with node resolve semantics)
        let r = resolve_path("a/../b", Path::new("/x/y"), &Default::default()).unwrap();
        assert_eq!(r, PathBuf::from("/x/y/b"));
        let r2 = resolve_path("../../b", Path::new("/x/y"), &Default::default()).unwrap();
        assert_eq!(r2, PathBuf::from("/b"));
        let r3 = resolve_path("/a/../b", Path::new("/x"), &Default::default()).unwrap();
        assert_eq!(r3, PathBuf::from("/b"));
    }

    #[test]
    fn test_get_cwd_relative_path_after_normalization() {
        // `a/../file.txt` is still inside cwd after lexical normalization
        let cwd = PathBuf::from("/tmp/pi-paths-cwd");
        let r = resolve_path("sub/../file.txt", &cwd, &Default::default()).unwrap();
        assert_eq!(r, cwd.join("file.txt"));
        assert_eq!(
            get_cwd_relative_path(&r, &cwd).unwrap().unwrap(),
            "file.txt"
        );
    }

    // ── canonicalize_path ─────────────────────────────────────────

    #[test]
    fn test_canonicalize_path_regular_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("file.txt");
        std::fs::write(&file, "hello").unwrap();
        let canon = std::fs::canonicalize(&file).unwrap();
        assert_eq!(canonicalize_path(&file), canon);
    }

    #[test]
    fn test_canonicalize_path_fallback_on_missing() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("no-such-file");
        assert_eq!(canonicalize_path(&missing), missing);
    }

    #[cfg(unix)]
    #[test]
    fn test_canonicalize_path_follows_symlink_and_dangling_fallback() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target.txt");
        std::fs::write(&target, "hello").unwrap();
        let link = dir.path().join("link.txt");
        symlink(&target, &link).unwrap();
        assert_eq!(
            canonicalize_path(&link),
            std::fs::canonicalize(&target).unwrap()
        );

        // dangling symlink → fall back to the original path
        let dangling = dir.path().join("dangling.txt");
        symlink(dir.path().join("missing-target"), &dangling).unwrap();
        assert_eq!(canonicalize_path(&dangling), dangling);
    }

    // ── is_local_path ─────────────────────────────────────────────

    #[test]
    fn test_is_local_path_true_for_local() {
        assert!(is_local_path("my-package"));
        assert!(is_local_path("./foo"));
        assert!(is_local_path("file:///tmp/foo"));
        assert!(is_local_path("  ./foo  ")); // check after trimming
    }

    #[test]
    fn test_is_local_path_false_for_remote() {
        assert!(!is_local_path("npm:package"));
        assert!(!is_local_path("git://repo"));
        assert!(!is_local_path("github:owner/repo"));
        assert!(!is_local_path("http://example.com"));
        assert!(!is_local_path("https://example.com"));
        assert!(!is_local_path("ssh://host"));
    }

    // ── get_cwd_relative_path ─────────────────────────────────────

    #[test]
    fn test_get_cwd_relative_path_keeps_dot_leading_components() {
        let cwd = PathBuf::from("/tmp/pi-paths-cwd");
        let file = cwd.join("..config").join("AGENTS.md");
        assert_eq!(
            get_cwd_relative_path(&file, &cwd).unwrap().unwrap(),
            "..config/AGENTS.md"
        );
    }

    #[test]
    fn test_get_cwd_relative_path_rejects_parent_traversal() {
        let cwd = PathBuf::from("/tmp/pi-paths-cwd");
        let file = cwd.join("..").join("AGENTS.md");
        assert_eq!(get_cwd_relative_path(&file, &cwd).unwrap(), None);
    }

    #[test]
    fn test_get_cwd_relative_path_self_is_dot() {
        let cwd = PathBuf::from("/tmp/pi-paths-cwd");
        assert_eq!(get_cwd_relative_path(&cwd, &cwd).unwrap().unwrap(), ".");
    }

    #[test]
    fn test_get_cwd_relative_path_inside() {
        let cwd = PathBuf::from("/tmp/pi-paths-cwd");
        let file = cwd.join("sub").join("a.rs");
        assert_eq!(
            get_cwd_relative_path(&file, &cwd).unwrap().unwrap(),
            "sub/a.rs"
        );
    }

    // ── format_path_relative_to_cwd_or_absolute ───────────────────

    #[test]
    fn test_format_relative_uses_relative_when_inside() {
        let cwd = PathBuf::from("/tmp/pi-paths-cwd");
        let file = cwd.join("sub").join("a.rs");
        assert_eq!(
            format_path_relative_to_cwd_or_absolute(&file, &cwd).unwrap(),
            "sub/a.rs"
        );
    }

    #[test]
    fn test_format_uses_absolute_when_outside() {
        let cwd = PathBuf::from("/tmp/pi-paths-cwd");
        let file = PathBuf::from("/etc/config");
        assert_eq!(
            format_path_relative_to_cwd_or_absolute(&file, &cwd).unwrap(),
            "/etc/config"
        );
    }
}
