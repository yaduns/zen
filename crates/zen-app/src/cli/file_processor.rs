// REQ-APP-08 (pi-331): @file argument handling
// See reference/pi-original/packages/coding-agent/src/cli/file-processor.ts
// Image byte sniffing has moved to zen_core::mime (T-AP-09-6, canonical module pi-342 semantics); this file consumes its public API
// detect_supported_image_mime_type (byte-level — read the whole content once, then sniff; eliminates TOCTOU).
// Implementation notes in tasks/T-AP-09-6-implemented.md
// at-attachment-wiring-fix: extract private classify_file_arg (three-state) as a shared classifier for both CLI/interactive consumers,
// Add expand_at_references (interactive submission @path expansion, a new zen capability; see tasks/at-attachment-wiring-fix-implemented.md).

use std::path::{Path, PathBuf};

use base64::{engine::general_purpose::STANDARD, Engine as _};
use zen_ai::types::ImageContent;
use zen_core::mime::detect_supported_image_mime_type;
use zen_core::tools::path_utils::resolve_read_path_async;
use zen_tui::autocomplete::find_at_refs;

/// Full artifacts of `@file` batch processing (handling of all `@file` references in one CLI run).
///
/// Corresponds to pi's `processFileArguments` return `{ text, images }`: `text` is the
/// `<file name="...">` wrapper block (concatenated in file_args order); **since at-attachment-wiring-fix** it is also
/// reused by `expand_at_references` as the return carrier—here `text` = full expanded text (complete message after in-place replacement).
/// `images` is the list of inlineable image attachments. It does not cross API/JSONL boundaries, so no serde derive is needed.
#[derive(Debug)]
pub struct ProcessedFiles {
    /// `<file>` wrapped blocks of all files concatenated in file_args order; may be an empty string when there are no files / all empty / all images omitted
    pub text: String,
    /// Inline image attachments (base64 data + sniffed MIME); empty Vec = no images
    pub images: Vec<ImageContent>,
}

/// Structured carrier for the two user-visible error surfaces of @file processing.
///
/// Corresponds to pi-331's two red error reports + `exit(1)`; Display message aligns character-for-character with pi (excluding colors — colors are
/// a terminal presentation concern, decided when main prints, continuing the "main prints + exits" precedent of `missing_cwd_disposition`).
#[derive(Debug, thiserror::Error)]
pub enum FileProcessError {
    /// Path resolution failure (`~` has no home, invalid `file://` URL) or metadata retrieval failure (including permission denied),
    /// aligned with pi `access()`'s single error message.
    #[error("File not found: {0}")]
    NotFound(PathBuf),
    /// File read failure (text or image bytes), aligned with pi `readFile` failure error.
    #[error("Could not read file {0}: {1}")]
    ReadFailed(PathBuf, std::io::Error),
}

/// Handle all `@file` references in CLI positional arguments: inject text file contents wrapped in `<file name="...">` blocks
/// into the initial prompt text, attach inline-capable image files as message attachments, for main to pass back via T-AP-08-2's
/// `build_initial_message` file_text/file_images parameters (aligned with pi `processFileArguments`).
///
/// # Behavioral constraints (derived from pi-331's original logic, must be preserved)
/// - Each file_arg is resolved via `resolve_read_path_async` to an absolute path (including macOS filename variant fallback);
///   the `<file name>` tag uses the resolved absolute path, not the user's raw input string.
/// - Path resolution failure (`~` has no home, etc.) and metadata retrieval failure (including permission denied) → `NotFound` (aligned with pi's
///   `access()` single message); empty files (`len() == 0`) are **silently skipped**.
/// - **Read full content once**: sniffing and branching share the same byte buffer, eliminating pi's TOCTOU window between
///   "sniffing 4100 bytes" and "reading the full content"; read failure → `ReadFailed` (shared mapping for text/image branches).
/// - Image sniffing (`zen_core::mime::detect_supported_image_mime_type`, pi-342 canonical module, T-AP-09-6):
///   - on png/jpeg/gif/webp hit → pass through base64 as `ImageContent`, append `<file name="{abs}"></file>\n` to text
///     (hints always empty; **no decode validation** — corrupted bytes are attached as-is; pi would hit a decode failure on resize and take the `!ok`
///     branch to omit the message; this divergence is explicitly registered, v1 accepts provider-side rejection);
///   - Match image/bmp (sniffed but no conversion pipeline) → append pi's `!ok` branch omission message to text
///     `<file name="{abs}">[Image omitted: could not be converted to a supported inline image format.]</file>\n`；
///   - No match → text branch: decode with `String::from_utf8_lossy` (Node `readFile(utf-8)` is lossy,
///     Rust `read_to_string` is strict — take lossy per pi semantics; append `<file name="{abs}">\n{content}\n</file>\n` to text
///     (newline layout exact).
/// - `file_args` empty → return empty `ProcessedFiles` (caller can skip the call per pi guard; behavior equivalent).
/// - All files accumulate into `text` in `file_args` order.
///
/// # Consumers
/// `crates/zen-app/src/main.rs` (nodes 2/3, T-AP-08-4 wiring, pi `prepareInitialMessage` counterpart).
pub async fn process_file_arguments(
    file_args: &[&str],
    cwd: &Path,
) -> Result<ProcessedFiles, FileProcessError> {
    let mut text = String::new();
    let mut images: Vec<ImageContent> = Vec::new();

    for file_arg in file_args {
        // Delegates to shared classifier (at-attachment-wiring-fix): Ok(None) = skip empty file (matches pi continue);
        // Err is propagated with existing fatal semantics (main.rs prints + exit(1)).
        if let Some(classified) = classify_file_arg(file_arg, cwd).await? {
            text.push_str(&classified.text_fragment);
            if let Some(image) = classified.image {
                images.push(image);
            }
        }
    }

    Ok(ProcessedFiles { text, images })
}

/// Classification result for a single file reference (produced by `classify_file_arg`, shared by `process_file_arguments` and `expand_at_references`):
/// `text_fragment` is a `<file name="...">…</file>\n` wrapper block (empty reference when an image matches); `image` is an inlineable attachment.
#[derive(Debug)]
struct ClassifiedFile {
    text_fragment: String,
    image: Option<ImageContent>,
}

/// Classify a single file reference (per-file classifier shared by `process_file_arguments` and `expand_at_references`,
/// extracted by at-attachment-wiring-fix): resolve → existence/empty file → read full text once → four-way mime classification,
/// produces the `<file>` wrapper fragment and inlineable attachments.
/// **Three-state return**: `Ok(Some(f))` = classification succeeded; `Ok(None)` = empty file skipped (CLI: do not append / interactive: remove token);
/// `Err(NotFound/ReadFailed)` = fatal and propagated in CLI / keep original token in interactive. Callers consume according to their own policy; do not swallow errors or lose state.
async fn classify_file_arg(
    file_arg: &str,
    cwd: &Path,
) -> Result<Option<ClassifiedFile>, FileProcessError> {
    // 1. Parse absolute path (with macOS filename variant fallback; still returns base when it doesn't exist — existence is decided by metadata)
    let resolved = resolve_read_path_async(file_arg, cwd)
        .await
        .map_err(|_| FileProcessError::NotFound(PathBuf::from(file_arg)))?;
    let abs = resolved.display().to_string();

    // 2. Existence + empty file skip (matches pi access/stat: any metadata error → File not found; size==0 → None)
    let metadata = tokio::fs::metadata(&resolved)
        .await
        .map_err(|_| FileProcessError::NotFound(resolved.clone()))?;
    if metadata.len() == 0 {
        return Ok(None);
    }

    // 3. Read full contents in one go (shared by sniffing and branching; read failure includes the TOCTOU case between sniffing and reading → ReadFailed)
    let bytes = tokio::fs::read(&resolved)
        .await
        .map_err(|e| FileProcessError::ReadFailed(resolved.clone(), e))?;

    // 4. Image sniffing classification (pi-342 semantics, via the zen_core::mime canonical module)
    let (text_fragment, image) = match detect_supported_image_mime_type(&bytes) {
        // Sniff hit but not inlinable (BMP only, no conversion pipeline) → pi `!ok` branch omission message
        Some("image/bmp") => (
            file_ref_wrapper(
                &abs,
                "[Image omitted: could not be converted to a supported inline image format.]",
            ),
            None,
        ),
        // Inlinable (png/jpeg/gif/webp) → pass-through appending raw bytes (no decode validation); hints always empty
        Some(mime) => (
            file_ref_wrapper(&abs, ""),
            Some(ImageContent {
                data: STANDARD.encode(&bytes),
                mime_type: mime.to_string(),
            }),
        ),
        // Text file → lossy decode wrapping (aligned with Node readFile(utf-8) lossy semantics)
        None => {
            let content = String::from_utf8_lossy(&bytes);
            (file_ref_wrapper(&abs, &format!("\n{content}\n")), None)
        }
    };
    Ok(Some(ClassifiedFile {
        text_fragment,
        image,
    }))
}

/// Before interactive submission, expand all `@path` references in the message body into attachment content (at-attachment-wiring-fix, **new zen capability**,
/// not pi-HEAD behavior; oracle = CHANGELOG v0.11.0 "Unified Content References" semantics + this fix's requirements): text file →
/// replace the token in place with a `<file name="...">` wrapper block (span = full `AtRef.raw` text); inlineable image → empty reference at the same position +
/// collect image attachments (in token occurrence order); path missing/unreadable → keep the original `@path` token (do not error and interrupt submission);
/// empty file → remove token (aligns with `processFileArguments` empty-file continue, **only at classifier semantics level**).
/// — "before submission, turn @file into file content; leave bad references as-is so the whole message is not harmed".
/// Parameters: `message` full user message; `cwd` session cwd (base for resolving relative @path). Reuses private `classify_file_arg`
/// (same classifier as CLI `process_file_arguments`; single source for MIME sniffing/wrapper format).
/// Returns `ProcessedFiles`: text = full expanded text (in-place replacement by span; parse first then concatenate to avoid shifting), images = all
/// inlineable attachments (token order). With no `@` references, returns unchanged (text = message, images empty).
/// **Ordering semantics note**: this function runs before the skill/template expansion inside `AgentSession::prompt`—so `@path` in template content
/// is **not** expanded by this function (expansion applies only to the original user input).
///
/// # Consumers
/// `crates/zen-app/src/modes/interactive/mod.rs` (`submit_message`/`FollowUp` branch, at-attachment-wiring-fix).
pub async fn expand_at_references(message: &str, cwd: &Path) -> ProcessedFiles {
    let refs = find_at_refs(message);
    if refs.is_empty() {
        return ProcessedFiles {
            text: message.to_string(),
            images: Vec::new(),
        };
    }

    let mut text = String::new();
    let mut images: Vec<ImageContent> = Vec::new();
    let mut cursor = 0usize;
    for r in &refs {
        // Locate raw after the previous replacement end point (find_at_refs guarantees raw is in occurrence order and non-overlapping; if location fails, fall back to keeping the original text)
        let Some(pos) = message[cursor..].find(&r.raw).map(|p| cursor + p) else {
            text.push_str(&message[cursor..]);
            break;
        };
        text.push_str(&message[cursor..pos]);
        match classify_file_arg(&r.path, cwd).await {
            Ok(Some(classified)) => {
                text.push_str(&classified.text_fragment);
                if let Some(image) = classified.image {
                    images.push(image);
                }
            }
            // Empty file → remove token (do not append content); error → keep original raw (silently, without interrupting submission)
            Ok(None) => {}
            Err(_) => text.push_str(&r.raw),
        }
        cursor = pos + r.raw.len();
    }
    text.push_str(&message[cursor..]);
    ProcessedFiles { text, images }
}

/// Build a single-line `<file name="{abs}">{inner}</file>\n` wrapper reference (exact newline layout, matching pi's `text +=` concatenation in each branch).
fn file_ref_wrapper(abs: &str, inner: &str) -> String {
    format!(r#"<file name="{abs}">{inner}</file>"#) + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose::STANDARD as B64;
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;

    // ── Test helpers: temp file and image byte construction ─────────────────

    fn write_file(dir: &Path, name: &str, content: &[u8]) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, content).unwrap();
        p
    }

    fn jpeg_bytes() -> Vec<u8> {
        // `FF D8 FF` magic bytes + 4th byte != `F7`; content need not be valid JPEG (sniffing only checks the magic)
        vec![0xFF, 0xD8, 0xFF, 0x00, 0xAA, 0xBB, 0xCC, 0xDD]
    }

    fn minimal_bmp_bytes() -> Vec<u8> {
        // BITMAPCOREHEADER(12) minimal valid structure: BM + declaredFileSize=0 + pixelDataOffset=26 + dibHeaderSize=12
        // + colorPlanes=1 + bitsPerPixel=24
        let mut b = vec![0u8; 26];
        b[0] = b'B';
        b[1] = b'M';
        b[10] = 26; // pixelDataOffset（LE）
        b[14] = 12; // dibHeaderSize（LE）
        b[22] = 1; // colorPlanes（LE u16）
        b[24] = 24; // bitsPerPixel（LE u16）
        b
    }

    // ── process_file_arguments: text paths ─────────────────

    #[tokio::test]
    async fn test_empty_file_args_returns_empty() {
        let dir = tempdir().unwrap();
        let r = process_file_arguments(&[], dir.path()).await.unwrap();
        assert_eq!(r.text, "");
        assert!(r.images.is_empty());
    }

    #[tokio::test]
    async fn test_text_file_wrapped_exactly() {
        let dir = tempdir().unwrap();
        let file = write_file(dir.path(), "notes.md", b"hello world");
        let r = process_file_arguments(&["notes.md"], dir.path())
            .await
            .unwrap();
        assert_eq!(
            r.text,
            format!("<file name=\"{}\">\nhello world\n</file>\n", file.display())
        );
        assert!(r.images.is_empty());
    }

    #[tokio::test]
    async fn test_multiple_files_concatenated_in_order() {
        let dir = tempdir().unwrap();
        let a = write_file(dir.path(), "a.md", b"AAA");
        let b = write_file(dir.path(), "b.md", b"BBB");
        let r = process_file_arguments(&["a.md", "b.md"], dir.path())
            .await
            .unwrap();
        assert_eq!(
            r.text,
            format!(
                "<file name=\"{}\">\nAAA\n</file>\n<file name=\"{}\">\nBBB\n</file>\n",
                a.display(),
                b.display()
            )
        );
    }

    #[tokio::test]
    async fn test_empty_file_skipped_silently() {
        let dir = tempdir().unwrap();
        write_file(dir.path(), "empty.txt", b"");
        let a = write_file(dir.path(), "a.md", b"AAA");
        let r = process_file_arguments(&["empty.txt", "a.md"], dir.path())
            .await
            .unwrap();
        // Empty file: no error, no text contribution
        assert_eq!(
            r.text,
            format!("<file name=\"{}\">\nAAA\n</file>\n", a.display())
        );
    }

    #[tokio::test]
    async fn test_missing_file_not_found() {
        let dir = tempdir().unwrap();
        let err = process_file_arguments(&["no-such.md"], dir.path())
            .await
            .unwrap_err();
        match &err {
            FileProcessError::NotFound(path) => {
                assert!(path.ends_with("no-such.md"));
                assert_eq!(
                    err.to_string(),
                    format!("File not found: {}", path.display())
                );
            }
            other => panic!("Expected NotFound, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_directory_read_failed() {
        let dir = tempdir().unwrap();
        // Directory exists and is non-empty → metadata passes, read fails → ReadFailed
        let sub = dir.path().join("subdir");
        std::fs::create_dir(&sub).unwrap();
        let err = process_file_arguments(&["subdir"], dir.path())
            .await
            .unwrap_err();
        assert!(matches!(err, FileProcessError::ReadFailed(_, _)));
        assert!(err.to_string().starts_with("Could not read file"));
    }

    #[tokio::test]
    async fn test_invalid_utf8_lossy() {
        let dir = tempdir().unwrap();
        write_file(dir.path(), "bad.bin", &[0xFF, 0xFE, b'a', b'b']);
        let r = process_file_arguments(&["bad.bin"], dir.path())
            .await
            .unwrap();
        // lossy replacement aligned with Node readFile(utf-8)
        assert!(r.text.contains('\u{FFFD}'));
        assert!(r.text.contains("ab"));
    }

    #[tokio::test]
    async fn test_text_then_image_order_preserved() {
        let dir = tempdir().unwrap();
        write_file(dir.path(), "notes.md", b"text");
        write_file(dir.path(), "pic.jpg", &jpeg_bytes());
        let r = process_file_arguments(&["notes.md", "pic.jpg"], dir.path())
            .await
            .unwrap();
        // text accumulation order: text blocks first, image references after
        assert!(r.text.contains(">text</file>") || r.text.contains("\ntext\n"));
        assert!(r.text.contains("</file>"));
        assert_eq!(r.text.matches("<file name=").count(), 2);
        assert_eq!(r.images.len(), 1);
        // images append order
        assert_eq!(r.images[0].mime_type, "image/jpeg");
    }

    // ── process_file_arguments: image branch ─────────────────

    #[tokio::test]
    async fn test_jpeg_image_attached() {
        let dir = tempdir().unwrap();
        let file = write_file(dir.path(), "pic.jpg", &jpeg_bytes());
        let r = process_file_arguments(&["pic.jpg"], dir.path())
            .await
            .unwrap();
        assert_eq!(r.images.len(), 1);
        assert_eq!(r.images[0].mime_type, "image/jpeg");
        assert_eq!(r.images[0].data, B64.encode(jpeg_bytes()));
        // reference block hints always empty
        assert_eq!(
            r.text,
            format!("<file name=\"{}\"></file>\n", file.display())
        );
    }

    #[tokio::test]
    async fn test_corrupt_image_attached_raw_no_omission() {
        let dir = tempdir().unwrap();
        // `FF D8 FF` magic + garbage bytes: sniff hit but corrupted content → pass-through appending raw bytes (divergence already registered,
        // No ellipsis message produced; pi decode failure after resize goes through the !ok path）
        let corrupt: Vec<u8> = vec![0xFF, 0xD8, 0xFF, 0x00, 0xDE, 0xAD, 0xBE, 0xEF];
        write_file(dir.path(), "broken.jpg", &corrupt);
        let r = process_file_arguments(&["broken.jpg"], dir.path())
            .await
            .unwrap();
        assert_eq!(r.images.len(), 1);
        assert_eq!(r.images[0].data, B64.encode(&corrupt));
        assert!(!r.text.contains("Image omitted"));
    }

    #[tokio::test]
    async fn test_bmp_omission_message() {
        let dir = tempdir().unwrap();
        write_file(dir.path(), "pic.bmp", &minimal_bmp_bytes());
        let r = process_file_arguments(&["pic.bmp"], dir.path())
            .await
            .unwrap();
        // Sniffing hit but not inlinable (no conversion pipeline) → pi `!ok` branch omits the message, not appended
        assert!(r.images.is_empty());
        assert!(r.text.contains(
            "[Image omitted: could not be converted to a supported inline image format.]"
        ));
    }

    // ── expand_at_references: interactive @path expansion (at-attachment-wiring-fix, new zen capability) ──

    #[tokio::test]
    async fn test_expand_at_refs_text_in_place() {
        let dir = tempdir().unwrap();
        let file = write_file(dir.path(), "a.md", b"hello world");
        let r = expand_at_references("see @a.md and done", dir.path()).await;
        // Text file → replace token in place with a `<file>` block; context preserved
        assert_eq!(
            r.text,
            format!(
                "see <file name=\"{}\">\nhello world\n</file>\n and done",
                file.display()
            )
        );
        assert!(r.images.is_empty());
    }

    #[tokio::test]
    async fn test_expand_at_refs_missing_path_keeps_raw() {
        let dir = tempdir().unwrap();
        let r = expand_at_references("see @missing.txt here", dir.path()).await;
        // Path missing → keep original token (do not error and interrupt submission)
        assert_eq!(r.text, "see @missing.txt here");
        assert!(r.images.is_empty());
    }

    #[tokio::test]
    async fn test_expand_at_refs_empty_file_removes_token() {
        let dir = tempdir().unwrap();
        write_file(dir.path(), "empty.txt", b"");
        let r = expand_at_references("see @empty.txt done", dir.path()).await;
        // Empty file skipped (only classifier semantics align with pi continue) → token removed
        assert_eq!(r.text, "see  done");
    }

    #[tokio::test]
    async fn test_expand_at_refs_image_attached_in_place() {
        let dir = tempdir().unwrap();
        let file = write_file(dir.path(), "pic.jpg", &jpeg_bytes());
        let r = expand_at_references("look @pic.jpg end", dir.path()).await;
        // Inlineable image → empty reference at same position + attachment collection
        assert_eq!(r.images.len(), 1);
        assert_eq!(r.images[0].mime_type, "image/jpeg");
        assert_eq!(r.images[0].data, B64.encode(jpeg_bytes()));
        assert_eq!(
            r.text,
            format!("look <file name=\"{}\"></file>\n end", file.display())
        );
    }

    #[tokio::test]
    async fn test_expand_at_refs_no_refs_returns_original() {
        let dir = tempdir().unwrap();
        let r = expand_at_references("no refs here", dir.path()).await;
        assert_eq!(r.text, "no refs here");
        assert!(r.images.is_empty());
    }

    #[tokio::test]
    async fn test_expand_at_refs_multiple_refs_order_preserved() {
        let dir = tempdir().unwrap();
        let a = write_file(dir.path(), "a.md", b"AAA");
        let b = write_file(dir.path(), "b.md", b"BBB");
        let r = expand_at_references("@a.md and @b.md", dir.path()).await;
        // Multiple references expanded in occurrence order; image order matches token order
        assert_eq!(
            r.text,
            format!(
                "<file name=\"{}\">\nAAA\n</file>\n and <file name=\"{}\">\nBBB\n</file>\n",
                a.display(),
                b.display()
            )
        );
        assert!(r.images.is_empty());
    }
}
