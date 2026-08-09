use std::path::{Path, PathBuf};
use unicode_normalization::UnicodeNormalization;

/// Normalize the path string.
///
/// Performs the following operations:
/// - Unicode space variants (`\u{00A0}`, `\u{2000}`-`\u{200A}`, `\u{202F}`, `\u{205F}`, `\u{3000}`) → regular space
/// - Strip leading `@` prefix (CLI `@file` path syntax)
/// - Tilde expansion (`~` → home directory, `~/...` → `$HOME/...`)
/// - Conversion from `file://` URL to path
///
/// Does not perform IO and does not check whether the path exists.
///
/// # Parameters
/// - `input`: raw path string.
///
/// # Return Value
/// Normalized path string.
///
/// # Edge Cases
/// - An empty string returns an empty string.
/// - A bare `~` returns the home directory path.
/// - Paths that are not `file://` URLs are not converted.
pub fn expand_path(input: &str) -> String {
    if input.is_empty() {
        return String::new();
    }

    // 1. Unicode space variants → regular space
    let s = normalize_unicode_whitespace(input);

    // 2. Strip leading @
    let s = s.strip_prefix('@').unwrap_or(&s);

    // 3. file:// URL → path
    let s = if let Some(rest) = s.strip_prefix("file://") {
        rest
    } else {
        s
    };

    // 4. Tilde expansion
    if s == "~" {
        return std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
    }
    if let Some(rest) = s.strip_prefix("~/") {
        let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
        return format!("{}/{}", home, rest);
    }

    s.to_string()
}

/// Replace Unicode space variants with regular spaces.
fn normalize_unicode_whitespace(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\u{00A0}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
            c if ('\u{2000}'..='\u{200A}').contains(&c) => ' ',
            _ => c,
        })
        .collect()
}

/// Resolve a path against the specified working directory.
///
/// Handles tilde expansion, absolute path detection, and relative path joining.
/// Returns an absolute path but does not check whether the path exists.
///
/// # Parameters
/// - `file_path`: path to resolve (relative or absolute).
/// - `cwd`: current working directory (absolute path).
///
/// # Return Value
/// Resolved absolute path.
///
/// # Edge Cases
/// - Absolute paths are returned directly without joining `cwd`.
/// - Paths starting with `~` are tilde-expanded first, then processed.
/// - `cwd` should be an absolute path; otherwise behavior is undefined.
pub fn resolve_to_cwd(file_path: &str, cwd: &Path) -> PathBuf {
    let expanded = expand_path(file_path);
    let path = Path::new(&expanded);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}

/// Resolve the file read path, with fallback support for macOS filename variants.
///
/// On top of `resolve_to_cwd`, when the target path does not exist, tries the following variants in order:
/// 1. macOS screenshot AM/PM narrow-space variant (`\u{202F}` replaces a regular space)
/// 2. NFD-decomposed variant (the macOS file system stores filenames in NFD)
/// 3. Curly-quote variant (macOS screenshots use the right single quotation mark `\u{2019}`, while users may input `'`)
/// 4. Combined NFD + curly-quote variant
///
/// Synchronous version, using `std::path::Path::try_exists()` to check.
///
/// # Parameters
/// - `file_path`: path to resolve.
/// - `cwd`: Current working directory.
///
/// # Return Value
/// Resolved absolute path (may not exist; if none of the variants match, returns the original resolution result).
///
/// # Edge Cases
/// - If the original path already exists, returns it directly without trying variants.
/// - If none of the variants exist, returns the original resolution result (does not throw an error).
pub fn resolve_read_path(file_path: &str, cwd: &Path) -> PathBuf {
    let base = resolve_to_cwd(file_path, cwd);
    if base.try_exists().unwrap_or(false) {
        return base;
    }

    let expanded = expand_path(file_path);
    let variants = [
        try_macos_screenshot_path(&expanded),
        try_nfd_variant(&expanded),
        try_curly_quote_variant(&expanded),
        try_nfd_variant(&try_curly_quote_variant(&expanded)),
    ];

    for variant in &variants {
        let path = if Path::new(variant).is_absolute() {
            PathBuf::from(variant)
        } else {
            cwd.join(variant)
        };
        if path.try_exists().unwrap_or(false) {
            return path;
        }
    }

    base
}

/// Asynchronously resolve the file read path, with fallback for macOS filename variants.
///
/// Same behavior as `resolve_read_path`, but uses async `tokio::fs::try_exists()` to check path existence.
/// Suitable for scenarios where blocking the current thread is undesirable.
pub async fn resolve_read_path_async(file_path: &str, cwd: &Path) -> PathBuf {
    let base = resolve_to_cwd(file_path, cwd);
    if path_exists(&base).await {
        return base;
    }

    let expanded = expand_path(file_path);
    let variants = [
        try_macos_screenshot_path(&expanded),
        try_nfd_variant(&expanded),
        try_curly_quote_variant(&expanded),
        try_nfd_variant(&try_curly_quote_variant(&expanded)),
    ];

    for variant in &variants {
        let path = if Path::new(variant).is_absolute() {
            PathBuf::from(variant)
        } else {
            cwd.join(variant)
        };
        if path_exists(&path).await {
            return path;
        }
    }

    base
}

/// Check whether the path exists (async).
///
/// Implemented using `tokio::fs::try_exists()`.
/// If the path is inaccessible (permission error), returns `false` instead of throwing an error.
pub async fn path_exists(file_path: &Path) -> bool {
    tokio::fs::try_exists(file_path).await.unwrap_or(false)
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

    // -- expand_path --

    #[test]
    fn test_expand_path_empty() {
        assert_eq!(expand_path(""), "");
    }

    #[test]
    fn test_expand_path_normal() {
        assert_eq!(expand_path("/foo/bar"), "/foo/bar");
    }

    #[test]
    fn test_expand_path_relative() {
        assert_eq!(expand_path("foo/bar"), "foo/bar");
    }

    #[test]
    fn test_expand_path_tilde_self() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
        assert_eq!(expand_path("~"), home);
    }

    #[test]
    fn test_expand_path_tilde_prefix() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
        let result = expand_path("~/doc/file.txt");
        assert_eq!(result, format!("{}/doc/file.txt", home));
    }

    #[test]
    fn test_expand_path_strip_at_prefix() {
        assert_eq!(expand_path("@/foo/bar"), "/foo/bar");
    }

    #[test]
    fn test_expand_path_file_url() {
        assert_eq!(
            expand_path("file:///home/user/config.toml"),
            "/home/user/config.toml"
        );
    }

    #[test]
    fn test_expand_path_at_and_tilde() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
        let result = expand_path("@~/dir");
        assert_eq!(result, format!("{}/dir", home));
    }

    #[test]
    fn test_expand_path_unicode_whitespace() {
        // \u{00A0} = no-break space, \u{202F} = narrow no-break space
        assert_eq!(
            expand_path("/dir/file\u{00A0}name.txt"),
            "/dir/file name.txt"
        );
        assert_eq!(
            expand_path("/dir/file\u{202F}name.txt"),
            "/dir/file name.txt"
        );
    }

    // -- normalize_unicode_whitespace --

    #[test]
    fn test_normalize_unicode_whitespace_all_variants() {
        assert_eq!(
            normalize_unicode_whitespace("hello\u{00A0}world"),
            "hello world"
        );
        assert_eq!(
            normalize_unicode_whitespace("hello\u{202F}world"),
            "hello world"
        );
        assert_eq!(normalize_unicode_whitespace("a\u{2000}b"), "a b");
        assert_eq!(normalize_unicode_whitespace("a\u{200A}b"), "a b");
        assert_eq!(normalize_unicode_whitespace("a\u{3000}b"), "a b");
    }

    #[test]
    fn test_normalize_unicode_whitespace_no_change() {
        assert_eq!(normalize_unicode_whitespace("hello world"), "hello world");
    }

    // -- resolve_to_cwd --

    #[test]
    fn test_resolve_to_cwd_relative() {
        let cwd = std::path::Path::new("/home/user/project");
        assert_eq!(
            resolve_to_cwd("src/lib.rs", cwd),
            std::path::Path::new("/home/user/project/src/lib.rs")
        );
    }

    #[test]
    fn test_resolve_to_cwd_absolute() {
        let cwd = std::path::Path::new("/home/user/project");
        assert_eq!(
            resolve_to_cwd("/etc/config", cwd),
            std::path::Path::new("/etc/config")
        );
    }

    #[test]
    fn test_resolve_to_cwd_tilde() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
        let cwd = std::path::Path::new("/home/user/project");
        let result = resolve_to_cwd("~/doc", cwd);
        assert_eq!(result, std::path::Path::new(&format!("{}/doc", home)));
    }

    #[test]
    fn test_resolve_to_cwd_dot() {
        let cwd = std::path::Path::new("/home/user/project");
        assert_eq!(resolve_to_cwd(".", cwd), cwd.join("."));
    }

    // -- try_macos_screenshot_path --

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

    // -- try_nfd_variant --

    #[test]
    fn test_try_nfd_variant_accented() {
        // "é" (U+00E9) decomposed to "e" + combining acute (U+0065 U+0301)
        let nfd: String = "é".nfd().collect();
        assert_eq!(try_nfd_variant("é"), nfd);
        assert_ne!(try_nfd_variant("é"), "é"); // NFC form is different
    }

    #[test]
    fn test_try_nfd_variant_ascii() {
        assert_eq!(try_nfd_variant("hello.txt"), "hello.txt");
    }

    // -- try_curly_quote_variant --

    #[test]
    fn test_try_curly_quote_straight_to_curly() {
        // Straight apostrophe -> curly right single quote
        assert_eq!(
            try_curly_quote_variant("Capture d'ecran"),
            "Capture d\u{2019}ecran"
        );
    }

    #[test]
    fn test_try_curly_quote_swap_is_symmetric() {
        // The function swaps both directions, so output equals input for pure-curly input.
        let input = "Capture d\u{2019}ecran";
        // first replace turns curly -> straight, second replace turns straight -> curly
        assert_eq!(try_curly_quote_variant(input), input);
    }

    #[test]
    fn test_try_curly_quote_no_quotes() {
        let input = "plain.txt";
        assert_eq!(try_curly_quote_variant(input), input);
    }

    // -- resolve_read_path / resolve_read_path_async / path_exists --
    // Depends on filesystem IO (Path::try_exists / tokio::fs::try_exists),
    // This module has no mock facilities; skip tests that depend on pure IO.
    // Downstream tool integration tests (T-AG-02-3~T-AG-02-6) cover these functions' actual filesystem behavior.
}
