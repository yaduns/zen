// REQ-APP-09 (pi-342): MIME type inference — image byte sniffing (canonical module)
// See reference/pi-original/packages/coding-agent/src/utils/mime.ts (pi-342)
// Migrated and absorbed from the private sniff_supported_image_mime_type in crates/zen-app/src/cli/file_processor.rs (T-AP-08-4
// inline implementation, locked by 17 tests); T-AP-08-4 now consumes this module's public surface. Placement exception: REQ-APP-09 → zen-core
// (consumers span T-AP-08-4 and the T-AG-02-8/read tool; the dependency direction does not allow zen-core→zen-app).
// Implementation notes in tasks/T-AP-09-6-implemented.md

use std::path::Path;
use tokio::io::AsyncReadExt;

/// PNG file signature (`89 50 4E 47 0D 0A 1A 0A`), used for image byte sniffing.
const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

/// Single-read upper limit (bytes) for file-level sniffing, aligned with pi `IMAGE_TYPE_SNIFF_BYTES` —
/// Determining an image only needs enough bytes from the file header; no need to read the whole file.
const IMAGE_TYPE_SNIFF_BYTES: usize = 4100;

/// Before attaching a file as a multimodal attachment, the @file handler (T-AP-08-4) and image tool (T-AG-02-8) first look at
/// the first few header bytes to determine whether it can be inlined as a supported image format — return the MIME type on a match, otherwise None
/// (Analogy: glance at the first few header lines to identify the address sign; if unrecognized, treat it as not an image).
///
/// Pure function, no IO; rules align with pi-342 exact semantics: JPEG `FF D8 FF` and the 4th byte is not `0xF7` (reject JPEG-2000) /
/// PNG signature + first IHDR chunk (length=13) + non-animated (`acTL` rejected) / GIF prefix / RIFF+WEBP prefix / BM prefix +
/// BMP structure validation (declaredFileSize, pixelDataOffset, dibHeaderSize∈{12,40..=124}, colorPlanes==1,
/// bpp∈{1,4,8,16,24,32}）。
///
/// Boundary: if `buffer` is too short, return `None` (all reads go through `get`/bounds checks, no panic); if no signature matches, return `None`.
///
/// Current consumer: T-AP-08-4 (`cli/file_processor.rs`). Planned consumer: T-AG-02-8 (image tool).
pub fn detect_supported_image_mime_type(buffer: &[u8]) -> Option<&'static str> {
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

/// When the image tool (T-AG-02-8) or a later read tool only knows the file path and wants to avoid reading the whole file, it reads only the first
/// `IMAGE_TYPE_SNIFF_BYTES` (4100) bytes from disk to determine whether it can be inlined as an image — return the MIME type on a hit, `None` for non-images,
/// If the file cannot be opened/read, return `io::Error` (analogy: glance at the file header at the door; no need to move the whole book inside).
///
/// Delegates to the byte-level `detect_supported_image_mime_type`; performs a **single** read via `tokio::io::AsyncReadExt::read`,
/// takes `&buf[..n]` (tolerating short reads), equivalent to pi's single-read behavior of `fileHandle.read(buffer, 0, 4100, 0)`;
/// `path` is the file path (aligned with pi's `open` contract; relative or absolute, reachability guaranteed by the caller).
///
/// Currently no consumer; pub but reserved for T-AG-02-8 (image tool) (plan note contract `io::Result<Option<String>>`
/// refined with `&'static str`, already registered).
pub async fn detect_supported_image_mime_type_from_file(
    path: &Path,
) -> std::io::Result<Option<&'static str>> {
    let mut file = tokio::fs::File::open(path).await?;
    let mut buffer = vec![0u8; IMAGE_TYPE_SNIFF_BYTES];
    let bytes_read = file.read(&mut buffer).await?;
    Ok(detect_supported_image_mime_type(&buffer[..bytes_read]))
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
    use std::path::PathBuf;
    use tempfile::tempdir;

    // ── Test helper: image byte construction (migrated from T-AP-08-4) ───────────────

    fn write_file(dir: &std::path::Path, name: &str, content: &[u8]) -> PathBuf {
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

    // ── detect_supported_image_mime_type pure function (migrated from T-AP-08-4) ──────────

    #[test]
    fn test_sniff_jpeg() {
        assert_eq!(
            detect_supported_image_mime_type(&jpeg_bytes()),
            Some("image/jpeg")
        );
        // 4th byte is 0xF7 → reject (pi: buffer[3] === 0xf7 → null)
        assert_eq!(
            detect_supported_image_mime_type(&[0xFF, 0xD8, 0xFF, 0xF7, 0x00, 0x00]),
            None
        );
        // Prefix too short
        assert_eq!(detect_supported_image_mime_type(&[0xFF, 0xD8]), None);
    }

    #[test]
    fn test_sniff_png_static_and_animated() {
        assert_eq!(
            detect_supported_image_mime_type(&static_png_bytes()),
            Some("image/png")
        );
        // Animated PNG (acTL chunk) → reject (pi does not inline animated PNGs)
        assert_eq!(
            detect_supported_image_mime_type(&animated_png_bytes()),
            None
        );
    }

    #[test]
    fn test_sniff_gif() {
        assert_eq!(
            detect_supported_image_mime_type(b"GIF89a\x00\x00\x00"),
            Some("image/gif")
        );
    }

    #[test]
    fn test_sniff_webp() {
        let mut webp = b"RIFF".to_vec();
        webp.extend_from_slice(&[0u8; 4]);
        webp.extend_from_slice(b"WEBP");
        assert_eq!(detect_supported_image_mime_type(&webp), Some("image/webp"));
        // RIFF but not WEBP
        let mut riff_only = b"RIFF".to_vec();
        riff_only.extend_from_slice(&[0u8; 8]);
        assert_eq!(detect_supported_image_mime_type(&riff_only), None);
    }

    #[test]
    fn test_sniff_bmp_valid_and_invalid() {
        assert_eq!(
            detect_supported_image_mime_type(&minimal_bmp_bytes()),
            Some("image/bmp")
        );
        assert_eq!(detect_supported_image_mime_type(&invalid_bmp_bytes()), None);
    }

    #[test]
    fn test_sniff_unknown_returns_none() {
        assert_eq!(detect_supported_image_mime_type(b"hello world"), None);
        assert_eq!(detect_supported_image_mime_type(&[]), None);
    }

    // ── detect_supported_image_mime_type_from_file file-level (new in this task) ────────

    #[tokio::test]
    async fn test_from_file_hits_image() {
        let dir = tempdir().unwrap();
        let f = write_file(dir.path(), "pic.jpg", &jpeg_bytes());
        assert_eq!(
            detect_supported_image_mime_type_from_file(&f)
                .await
                .unwrap(),
            Some("image/jpeg")
        );
    }

    #[tokio::test]
    async fn test_from_file_non_image_returns_none() {
        let dir = tempdir().unwrap();
        let f = write_file(dir.path(), "notes.txt", b"hello world");
        assert_eq!(
            detect_supported_image_mime_type_from_file(&f)
                .await
                .unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn test_from_file_missing_path_is_io_error() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("no-such.bin");
        let err = detect_supported_image_mime_type_from_file(&missing)
            .await
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    #[tokio::test]
    async fn test_from_file_small_image_still_detected() {
        // Files smaller than 4100 bytes: after a single short read, `&buf[..n]` still detects correctly
        let dir = tempdir().unwrap();
        let f = write_file(dir.path(), "tiny.gif", b"GIF89a\x00\x00\x00");
        assert_eq!(
            detect_supported_image_mime_type_from_file(&f)
                .await
                .unwrap(),
            Some("image/gif")
        );
    }
}
