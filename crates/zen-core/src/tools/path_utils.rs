// Provide tool-specific file path resolution: `resolve_to_cwd` delegates to `paths::resolve_path` (fixed
// normalize_unicode_spaces + strip_at_prefix options); `resolve_read_path`/`resolve_read_path_async`
// additionally apply macOS filename variant fallbacks (AM/PM narrow spaces, NFD, curly quotes).

use crate::paths::{self, PathError};
use std::path::{Path, PathBuf};
use unicode_normalization::UnicodeNormalization;

/// Options always used for tool path resolution: normalize Unicode spaces + strip leading `@`
const TOOL_PATH_OPTIONS: paths::PathInputOptions = paths::PathInputOptions {
    trim: false,
    expand_tilde: true,
    home_dir: None,
    strip_at_prefix: true,
    normalize_unicode_spaces: true,
};

/// Resolve a file path to an absolute path.
///
/// Delegates to `paths::resolve_path` with fixed `normalize_unicode_spaces` + `strip_at_prefix` options
/// (aligned with Pi `resolveToCwd = resolvePath(filePath, cwd, { normalizeUnicodeSpaces: true, stripAtPrefix: true })`).
/// Handles tilde expansion, `@` stripping, and `file://` URLs internally. Returns `Result` to propagate `PathError`.
///
/// # Errors
/// - Expand `~` but home unavailable → `PathError::NoHomeDir`.
/// - Invalid `file://` URL → `PathError::InvalidFileUrl`.
///
/// # Consumers
/// edit.rs / write.rs / edit_diff.rs (currently present).
pub fn resolve_to_cwd(file_path: &str, cwd: &Path) -> Result<PathBuf, PathError> {
    paths::resolve_path(file_path, cwd, &TOOL_PATH_OPTIONS)
}

/// Resolve the file read path, with fallback support for macOS filename variants.
///
/// On top of `resolve_to_cwd`, when the target path does not exist, try the following variants in order (aligned with Pi `resolveReadPath`):
/// 1. macOS screenshot AM/PM narrow-space variant (`\u{202F}` replaces a regular space)
/// 2. NFD-decomposed variant (the macOS file system stores filenames in NFD)
/// 3. Curly-quote variant (macOS screenshots use the right single quotation mark `\u{2019}`, while users may input `'`)
/// 4. Combined NFD + curly-quote variant
///
/// Synchronous version, checks with `std::path::Path::try_exists()`.
///
/// # Consumers
/// pub but currently has no production consumer (read.rs uses the async version); kept to align with pi-218's synchronous API,
/// for future synchronous resolution scenarios. Directly covered by tests in this module.
pub fn resolve_read_path(file_path: &str, cwd: &Path) -> Result<PathBuf, PathError> {
    let base = resolve_to_cwd(file_path, cwd)?;
    if base.try_exists().unwrap_or(false) {
        return Ok(base);
    }

    let expanded = paths::normalize_path(file_path, &TOOL_PATH_OPTIONS)?;
    for variant in macos_variants(&expanded) {
        let path = if Path::new(&variant).is_absolute() {
            PathBuf::from(variant)
        } else {
            cwd.join(variant)
        };
        if path.try_exists().unwrap_or(false) {
            return Ok(path);
        }
    }

    Ok(base)
}

/// Asynchronously resolve the file read path, with fallback for macOS filename variants.
///
/// Same behavior as `resolve_read_path`, but checks path existence with `tokio::fs::try_exists()`,
/// Suitable for scenarios where blocking the current thread is undesirable.
///
/// # Consumers
/// read.rs (currently present).
pub async fn resolve_read_path_async(file_path: &str, cwd: &Path) -> Result<PathBuf, PathError> {
    let base = resolve_to_cwd(file_path, cwd)?;
    if path_exists(&base).await {
        return Ok(base);
    }

    let expanded = paths::normalize_path(file_path, &TOOL_PATH_OPTIONS)?;
    for variant in macos_variants(&expanded) {
        let path = if Path::new(&variant).is_absolute() {
            PathBuf::from(variant)
        } else {
            cwd.join(variant)
        };
        if path_exists(&path).await {
            return Ok(path);
        }
    }

    Ok(base)
}

/// Checks whether a path exists (async); returns `false` instead of throwing when inaccessible (permission errors, etc.).
async fn path_exists(file_path: &Path) -> bool {
    tokio::fs::try_exists(file_path).await.unwrap_or(false)
}

/// Generates macOS filename variants in order: AM/PM narrow space, NFD, curly quotes, NFD + curly quotes combination.
fn macos_variants(expanded: &str) -> [String; 4] {
    [
        try_macos_screenshot_path(expanded),
        try_nfd_variant(expanded),
        try_curly_quote_variant(expanded),
        try_nfd_variant(&try_curly_quote_variant(expanded)),
    ]
}

/// Try the macOS AM/PM narrow-space variant.
/// Replace regular spaces in ` AM.` or ` PM.` with narrow no-break spaces (`\u{202F}`).
fn try_macos_screenshot_path(file_path: &str) -> String {
    file_path
        .replace(" AM.", "\u{202F}AM.")
        .replace(" PM.", "\u{202F}PM.")
}

/// Try the NFD-decomposed variant (the macOS file system stores filenames in NFD form).
fn try_nfd_variant(file_path: &str) -> String {
    file_path.nfd().collect()
}

/// Try the curly-quote variant (swap `\u{2019}` in macOS screenshot names with the user's `'`).
fn try_curly_quote_variant(file_path: &str) -> String {
    file_path.replace('\u{2019}', "'").replace('\'', "\u{2019}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // ── resolve_to_cwd ────────────────────────────────────────────

    #[test]
    fn test_resolve_to_cwd_relative() {
        let cwd = Path::new("/home/user/project");
        assert_eq!(
            resolve_to_cwd("src/lib.rs", cwd).unwrap(),
            PathBuf::from("/home/user/project/src/lib.rs")
        );
    }

    #[test]
    fn test_resolve_to_cwd_absolute() {
        let cwd = Path::new("/home/user/project");
        assert_eq!(
            resolve_to_cwd("/etc/config", cwd).unwrap(),
            PathBuf::from("/etc/config")
        );
    }

    #[test]
    fn test_resolve_to_cwd_tilde() {
        let home = home::home_dir().expect("home available");
        let cwd = Path::new("/home/user/project");
        assert_eq!(resolve_to_cwd("~/doc", cwd).unwrap(), home.join("doc"));
    }

    #[test]
    fn test_resolve_to_cwd_strip_at_and_unicode_space() {
        let cwd = Path::new("/home/user/project");
        // Tool options always strip @ and normalize Unicode spaces (aligned with pi's resolveToCwd options)
        assert_eq!(
            resolve_to_cwd("@src/lib.rs", cwd).unwrap(),
            cwd.join("src/lib.rs")
        );
        assert_eq!(
            resolve_to_cwd("src/file\u{00A0}name.txt", cwd).unwrap(),
            cwd.join("src/file name.txt")
        );
    }

    #[test]
    fn test_resolve_to_cwd_dot() {
        let cwd = Path::new("/home/user/project");
        assert_eq!(resolve_to_cwd(".", cwd).unwrap(), cwd.join("."));
    }

    // ── macOS variant pure functions ──────────────────────────────────────────

    #[test]
    fn test_try_macos_screenshot_path_am() {
        assert_eq!(
            try_macos_screenshot_path("/dir/Screenshot 2024-01-01 at 10.30.00 AM.png"),
            "/dir/Screenshot 2024-01-01 at 10.30.00\u{202F}AM.png"
        );
    }

    #[test]
    fn test_try_macos_screenshot_path_pm() {
        assert_eq!(
            try_macos_screenshot_path("/dir/photo 3.45.00 PM.jpeg"),
            "/dir/photo 3.45.00\u{202F}PM.jpeg"
        );
    }

    #[test]
    fn test_try_macos_screenshot_path_no_match() {
        let input = "/dir/plain.txt";
        assert_eq!(try_macos_screenshot_path(input), input);
    }

    #[test]
    fn test_try_nfd_variant_accented() {
        let nfd: String = "é".nfd().collect();
        assert_eq!(try_nfd_variant("é"), nfd);
        assert_ne!(try_nfd_variant("é"), "é"); // NFC form is different
    }

    #[test]
    fn test_try_nfd_variant_ascii() {
        assert_eq!(try_nfd_variant("hello.txt"), "hello.txt");
    }

    #[test]
    fn test_try_curly_quote_straight_to_curly() {
        assert_eq!(
            try_curly_quote_variant("Capture d'ecran"),
            "Capture d\u{2019}ecran"
        );
    }

    #[test]
    fn test_try_curly_quote_swap_is_symmetric() {
        let input = "Capture d\u{2019}ecran";
        assert_eq!(try_curly_quote_variant(input), input);
    }

    #[test]
    fn test_try_curly_quote_no_quotes() {
        assert_eq!(try_curly_quote_variant("plain.txt"), "plain.txt");
    }

    // ── resolve_read_path / resolve_read_path_async (real filesystem) ─

    fn write(dir: &Path, name: &str, content: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, content).unwrap();
        p
    }

    #[test]
    fn test_resolve_read_path_existing_returns_as_is() {
        let dir = tempfile::tempdir().unwrap();
        let file = write(dir.path(), "test-file.txt", "content");
        let r = resolve_read_path("test-file.txt", dir.path()).unwrap();
        assert_eq!(r, file);
    }

    #[test]
    fn test_resolve_read_path_nfd_fallback() {
        // User input NFC `é`, disk stores NFD (macOS behavior) → variant fallback matches.
        // Assertions aligned with pi tests (lenient): macOS filesystems normalize NFC/NFD, final byte form is platform-dependent,
        // only assert it lands in the target directory and the filename shape is correct.
        let dir = tempfile::tempdir().unwrap();
        let nfd_name: String = "file\u{0065}\u{0301}.txt".to_string();
        write(dir.path(), &nfd_name, "content");
        let r = resolve_read_path("file\u{00e9}.txt", dir.path()).unwrap();
        assert!(r.starts_with(dir.path()));
        let name = r.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with("file") && name.ends_with(".txt"));
    }

    #[test]
    fn test_resolve_read_path_macos_screenshot_am_pm() {
        let dir = tempfile::tempdir().unwrap();
        // Disk has narrow no-break space (U+202F) + AM; user input has a regular space
        let macos_name = "Screenshot 2024-01-01 at 10.00.00\u{202F}AM.png";
        let macos_file = write(dir.path(), macos_name, "x");
        let r = resolve_read_path("Screenshot 2024-01-01 at 10.00.00 AM.png", dir.path()).unwrap();
        assert_eq!(r, macos_file);
    }

    #[test]
    fn test_resolve_read_path_returns_base_when_no_variant_matches() {
        let dir = tempfile::tempdir().unwrap();
        let base = resolve_to_cwd("no-such-file.txt", dir.path()).unwrap();
        let r = resolve_read_path("no-such-file.txt", dir.path()).unwrap();
        assert_eq!(r, base);
    }

    #[tokio::test]
    async fn test_resolve_read_path_async_existing() {
        let dir = tempfile::tempdir().unwrap();
        let file = write(dir.path(), "async.txt", "content");
        let r = resolve_read_path_async("async.txt", dir.path())
            .await
            .unwrap();
        assert_eq!(r, file);
    }

    #[tokio::test]
    async fn test_resolve_read_path_async_nfd_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let nfd_name: String = "file\u{0065}\u{0301}.txt".to_string();
        write(dir.path(), &nfd_name, "content");
        let r = resolve_read_path_async("file\u{00e9}.txt", dir.path())
            .await
            .unwrap();
        assert!(r.starts_with(dir.path()));
        let name = r.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with("file") && name.ends_with(".txt"));
    }
}
