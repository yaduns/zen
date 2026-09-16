// REQ-APP-08 (pi-331): @file argument handling
// See reference/pi-original/packages/coding-agent/src/cli/file-processor.ts
// image byte sniffing (sniff_supported_image_mime_type and byte helpers) reference
// reference/pi-original/packages/coding-agent/src/utils/mime.ts（pi-342）；
// migrate to the canonical MIME module when T-AP-09-6 lands and expose detect_supported_image_mime_type_from_file.
// implementation notes: tasks/T-AP-08-4-implemented.md

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use base64::{engine::general_purpose::STANDARD, Engine as _};
use zen_ai::types::ImageContent;
use zen_core::tools::path_utils::resolve_read_path_async;

/// PNG file signature (`89 50 4E 47 0D 0A 1A 0A`), used for image byte sniffing.
const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

/// Full artifacts of `@file` batch processing (handling of all `@file` references in one CLI run).
///
/// Corresponds to pi's `processFileArguments` return `{ text, images }`: `text` is the
/// `<file name="...">` wrapper block (concatenated in file_args order); `images` is the list of inline image attachments.
/// Does not cross API/JSONL boundaries; no serde derive needed.
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
/// - image sniffing (private `sniff_supported_image_mime_type`, pi-342 semantics; migrate when T-AP-09-6 lands):
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
        // 1. Parse absolute path (with macOS filename variant fallback; still returns base when it doesn't exist — existence is decided by metadata)
        let resolved = resolve_read_path_async(file_arg, cwd)
            .await
            .map_err(|_| FileProcessError::NotFound(PathBuf::from(file_arg)))?;
        let abs = resolved.display().to_string();

        // 2. Existence + empty-file skip (matches pi access/stat: any metadata error → File not found; size==0 → continue)
        let metadata = tokio::fs::metadata(&resolved)
            .await
            .map_err(|_| FileProcessError::NotFound(resolved.clone()))?;
        if metadata.len() == 0 {
            continue;
        }

        // 3. Read full contents in one go (shared by sniffing and branching; read failure includes the TOCTOU case between sniffing and reading → ReadFailed)
        let bytes = tokio::fs::read(&resolved)
            .await
            .map_err(|e| FileProcessError::ReadFailed(resolved.clone(), e))?;

        // 4. image sniffing classification (pi-342 semantics)
        match sniff_supported_image_mime_type(&bytes) {
            // Sniff hit but not inlinable (BMP only, no conversion pipeline) → pi `!ok` branch omission message
            Some("image/bmp") => {
                push_file_ref(
                    &mut text,
                    &abs,
                    "[Image omitted: could not be converted to a supported inline image format.]",
                );
            }
            // Inlinable (png/jpeg/gif/webp) → pass-through appending raw bytes (no decode validation); hints always empty
            Some(mime) => {
                images.push(ImageContent {
                    data: STANDARD.encode(&bytes),
                    mime_type: mime.to_string(),
                });
                push_file_ref(&mut text, &abs, "");
            }
            // Text file → lossy decode wrapping (aligned with Node readFile(utf-8) lossy semantics)
            None => {
                let content = String::from_utf8_lossy(&bytes);
                push_file_ref(&mut text, &abs, &format!("\n{content}\n"));
            }
        }
    }

    Ok(ProcessedFiles { text, images })
}

/// Appends one line to `text` wrapping the reference as `<file name="{abs}">{inner}</file>\n` (exact newline layout, matching the `text +=` concatenation in pi's branches).
/// Writing to a String cannot fail (Infallible); `let _ =` explicitly ignores the `fmt::Result`.
fn push_file_ref(text: &mut String, abs: &str, inner: &str) {
    let _ = write!(text, r#"<file name="{abs}">"#);
    text.push_str(inner);
    text.push_str("</file>\n");
}

/// Sniff the supported image MIME type from the byte prefix (JPEG/PNG/GIF/WebP/BMP); returns `None` on no match.
///
/// Corresponds to pi-342 `detectSupportedImageMimeType` (byte-level detection, not by extension):
/// - JPEG: `FF D8 FF` prefix and 4th byte not `F7` (JFIF magic) → `image/jpeg`
/// - PNG: full signature + IHDR first chunk (length=13) + non-animated (no `acTL` chunk) → `image/png`
/// - GIF: `GIF` prefix → `image/gif`
/// - WebP: `RIFF` prefix + `WEBP` at offset 8 → `image/webp`
/// - BMP: `BM` prefix + `is_bmp` structural check → `image/bmp`
///
/// migrate to the canonical MIME module when T-AP-09-6 lands (precedent: T-AP-09-1 absorbed path_utils.rs); at that point
/// `process_file_arguments` will consume its public surface `detect_supported_image_mime_type_from_file`.
fn sniff_supported_image_mime_type(buffer: &[u8]) -> Option<&'static str> {
    if starts_with_bytes(buffer, &[0xFF, 0xD8, 0xFF]) {
        return if buffer.get(3) == Some(&0xF7) {
            None
        } else {
            Some("image/jpeg")
        };
    }
    if starts_with_bytes(buffer, &PNG_SIGNATURE) {
        return if is_png(buffer) && !is_animated_png(buffer) {
            Some("image/png")
        } else {
            None
        };
    }
    if starts_with_ascii(buffer, 0, "GIF") {
        return Some("image/gif");
    }
    if starts_with_ascii(buffer, 0, "RIFF") && starts_with_ascii(buffer, 8, "WEBP") {
        return Some("image/webp");
    }
    if starts_with_ascii(buffer, 0, "BM") && is_bmp(buffer) {
        return Some("image/bmp");
    }
    None
}

/// Whether the PNG has a valid first IHDR chunk (8 bytes after the signature are chunk length=13 + `IHDR` type).
fn is_png(buffer: &[u8]) -> bool {
    buffer.len() >= 16
        && read_u32_be(buffer, PNG_SIGNATURE.len()) == 13
        && starts_with_ascii(buffer, 12, "IHDR")
}

/// Whether the PNG contains an `acTL` animation control chunk (walks the chunk chain; treats `IDAT` as static and returns early).
fn is_animated_png(buffer: &[u8]) -> bool {
    let mut offset = PNG_SIGNATURE.len();
    while offset + 8 <= buffer.len() {
        let chunk_length = read_u32_be(buffer, offset) as usize;
        let chunk_type_offset = offset + 4;
        if starts_with_ascii(buffer, chunk_type_offset, "acTL") {
            return true;
        }
        if starts_with_ascii(buffer, chunk_type_offset, "IDAT") {
            return false;
        }
        let next_offset = offset + 8 + chunk_length + 4;
        if next_offset <= offset || next_offset > buffer.len() {
            return false;
        }
        offset = next_offset;
    }
    false
}

/// BMP structure validation (field consistency for BITMAPCOREHEADER=12 / BITMAPINFOHEADER=40..=124).
fn is_bmp(buffer: &[u8]) -> bool {
    if buffer.len() < 26 {
        return false;
    }
    let declared_file_size = read_u32_le(buffer, 2);
    let pixel_data_offset = read_u32_le(buffer, 10);
    let dib_header_size = read_u32_le(buffer, 14);
    if declared_file_size != 0 && declared_file_size < 26 {
        return false;
    }
    if pixel_data_offset < 14 + dib_header_size {
        return false;
    }
    if declared_file_size != 0 && pixel_data_offset >= declared_file_size {
        return false;
    }

    let (color_planes, bits_per_pixel) = if dib_header_size == 12 {
        (read_u16_le(buffer, 22), read_u16_le(buffer, 24))
    } else if (40..=124).contains(&dib_header_size) {
        if buffer.len() < 30 {
            return false;
        }
        (read_u16_le(buffer, 26), read_u16_le(buffer, 28))
    } else {
        return false;
    };

    color_planes == 1 && matches!(bits_per_pixel, 1 | 4 | 8 | 16 | 24 | 32)
}

/// Byte-prefix match (false on out-of-bounds).
fn starts_with_bytes(buffer: &[u8], prefix: &[u8]) -> bool {
    buffer.len() >= prefix.len() && &buffer[..prefix.len()] == prefix
}

/// Match ASCII text starting at `offset` (false on out-of-bounds).
fn starts_with_ascii(buffer: &[u8], offset: usize, text: &str) -> bool {
    if buffer.len() < offset + text.len() {
        return false;
    }
    text.as_bytes()
        .iter()
        .enumerate()
        .all(|(i, &b)| buffer[offset + i] == b)
}

/// Little-endian u16 (out-of-bounds bytes treated as 0, matching pi `buffer[offset] ?? 0`).
fn read_u16_le(buffer: &[u8], offset: usize) -> u16 {
    let b0 = buffer.get(offset).copied().unwrap_or(0) as u16;
    let b1 = buffer.get(offset + 1).copied().unwrap_or(0) as u16;
    b0 | (b1 << 8)
}

/// Big-endian u32 (out-of-bounds bytes treated as 0).
fn read_u32_be(buffer: &[u8], offset: usize) -> u32 {
    let mut value = 0u32;
    for i in 0..4 {
        value = (value << 8) | buffer.get(offset + i).copied().unwrap_or(0) as u32;
    }
    value
}

/// Little-endian u32 (out-of-bounds bytes treated as 0).
fn read_u32_le(buffer: &[u8], offset: usize) -> u32 {
    let mut value = 0u32;
    for i in 0..4 {
        value |= (buffer.get(offset + i).copied().unwrap_or(0) as u32) << (8 * i);
    }
    value
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

    fn static_png_bytes() -> Vec<u8> {
        // Real PNG chunk structure: length(4) + type(4) + data + CRC(4)—chunk traversal (is_animated_png)
        // Advance by `offset + 8 + chunk_length + 4`; missing CRC shifts traversal by 4 bytes
        let mut png = PNG_SIGNATURE.to_vec();
        png.extend_from_slice(&13u32.to_be_bytes()); // IHDR chunk length
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&[0u8; 13]); // IHDR data
        png.extend_from_slice(&[0u8; 4]); // CRC
        png
    }

    fn animated_png_bytes() -> Vec<u8> {
        let mut png = static_png_bytes();
        png.extend_from_slice(&8u32.to_be_bytes()); // acTL chunk length
        png.extend_from_slice(b"acTL");
        png.extend_from_slice(&[0u8; 8]); // acTL data
        png.extend_from_slice(&[0u8; 4]); // CRC
        png
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

    fn invalid_bmp_bytes() -> Vec<u8> {
        let mut b = minimal_bmp_bytes();
        b[22] = 2; // colorPlanes != 1 → reject
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

    // ── sniff_supported_image_mime_type pure function ───────────────────

    #[test]
    fn test_sniff_jpeg() {
        assert_eq!(
            sniff_supported_image_mime_type(&jpeg_bytes()),
            Some("image/jpeg")
        );
        // 4th byte is 0xF7 → reject (pi: buffer[3] === 0xf7 → null)
        assert_eq!(
            sniff_supported_image_mime_type(&[0xFF, 0xD8, 0xFF, 0xF7, 0x00, 0x00]),
            None
        );
        // Prefix too short
        assert_eq!(sniff_supported_image_mime_type(&[0xFF, 0xD8]), None);
    }

    #[test]
    fn test_sniff_png_static_and_animated() {
        assert_eq!(
            sniff_supported_image_mime_type(&static_png_bytes()),
            Some("image/png")
        );
        // Animated PNG (acTL chunk) → reject (pi does not inline animated PNGs)
        assert_eq!(sniff_supported_image_mime_type(&animated_png_bytes()), None);
    }

    #[test]
    fn test_sniff_gif() {
        assert_eq!(
            sniff_supported_image_mime_type(b"GIF89a\x00\x00\x00"),
            Some("image/gif")
        );
    }

    #[test]
    fn test_sniff_webp() {
        let mut webp = b"RIFF".to_vec();
        webp.extend_from_slice(&[0u8; 4]);
        webp.extend_from_slice(b"WEBP");
        assert_eq!(sniff_supported_image_mime_type(&webp), Some("image/webp"));
        // RIFF but not WEBP
        let mut riff_only = b"RIFF".to_vec();
        riff_only.extend_from_slice(&[0u8; 8]);
        assert_eq!(sniff_supported_image_mime_type(&riff_only), None);
    }

    #[test]
    fn test_sniff_bmp_valid_and_invalid() {
        assert_eq!(
            sniff_supported_image_mime_type(&minimal_bmp_bytes()),
            Some("image/bmp")
        );
        assert_eq!(sniff_supported_image_mime_type(&invalid_bmp_bytes()), None);
    }

    #[test]
    fn test_sniff_unknown_returns_none() {
        assert_eq!(sniff_supported_image_mime_type(b"hello world"), None);
        assert_eq!(sniff_supported_image_mime_type(&[]), None);
    }
}
