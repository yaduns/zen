use std::hash::{Hash, Hasher};

/// Fast deterministic hash that shortens long strings into short base-36 strings.
///
/// Internally uses `std::collections::hash_map::DefaultHasher` (SipHash-1-3),
/// Output is the base-36 encoding of a u64 (lowercase letters + digits), usually 6–10 characters long.
///
/// Note: this function is **not** cryptographically secure and is only used to generate short identifiers
/// (such as tool call ID prefix `fc_xxx`, message ID `msg_xxx`, cache keys).
///
/// Corresponds to Pi's `shortHash(str: string): string`.
///
/// # Parameters
/// - `input`: input string to hash.
///
/// # Return Value
/// Base-36 encoded hash string.
///
/// # Determinism guarantee
/// The Rust standard library guarantees that `DefaultHasher` produces the same output for the same input within the same process,
/// but different versions of the Rust standard library may use different seeding strategies. `short_hash` is only used for
/// shortening identifiers within a single process; hash values are not exchanged across versions or processes (Rust ↔ TS),
/// so this difference is acceptable.
///
/// # Edge cases
/// - Empty string: returns a non-empty string and does not panic.
/// - Any valid UTF-8 string can be used as input, including non-BMP characters such as emoji.
pub fn short_hash(input: &str) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    input.hash(&mut hasher);
    encode_base36(hasher.finish()) // u64 → base-36
}

/// Encodes a u64 as a base-36 lowercase string (semantically consistent with TS `(n >>> 0).toString(36)`).
///
/// # Parameters
/// - `n`: 64-bit unsigned integer to encode.
///
/// # Return Value
/// Base-36 encoded string (digits `0-9` + lowercase letters `a-z`), without leading zeros.
fn encode_base36(mut n: u64) -> String {
    if n == 0 {
        return "0".into();
    }
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    // u64::MAX = 18446744073709551615, base-36 is at most 13 digits
    let mut buf = [0u8; 13];
    let mut i = 13;
    while n > 0 {
        i -= 1;
        buf[i] = DIGITS[(n % 36) as usize];
        n /= 36;
    }
    String::from_utf8(buf[i..].to_vec()).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // encode_base36
    // ============================================================

    #[test]
    fn test_encode_base36_zero() {
        assert_eq!(encode_base36(0), "0");
    }

    #[test]
    fn test_encode_base36_one() {
        assert_eq!(encode_base36(1), "1");
    }

    #[test]
    fn test_encode_base36_ten() {
        assert_eq!(encode_base36(10), "a");
    }

    #[test]
    fn test_encode_base36_thirty_five() {
        assert_eq!(encode_base36(35), "z");
    }

    #[test]
    fn test_encode_base36_thirty_six() {
        assert_eq!(encode_base36(36), "10");
    }

    #[test]
    fn test_encode_base36_large_round() {
        // 36^2 = 1296
        assert_eq!(encode_base36(1296), "100");
    }

    #[test]
    fn test_encode_base36_u64_max() {
        // u64::MAX = 18446744073709551615, should not panic
        let result = encode_base36(u64::MAX);
        assert!(!result.is_empty());
        // Verify it's deterministic
        assert_eq!(encode_base36(u64::MAX), result);
    }

    #[test]
    fn test_encode_base36_deterministic() {
        let cases = [0u64, 1, 42, 100, 65535, 123456789];
        for &n in &cases {
            let first = encode_base36(n);
            let second = encode_base36(n);
            assert_eq!(first, second, "encode_base36({}) is not deterministic", n);
        }
    }

    #[test]
    fn test_encode_base36_no_leading_zero() {
        // No base-36 encoded value (except "0" itself) should start with '0'
        for n in [1u64, 36, 1296, 46656, 1679616, 60466176, u64::MAX] {
            let encoded = encode_base36(n);
            assert!(
                !encoded.starts_with('0'),
                "{} encoded as {} starts with '0'",
                n,
                encoded
            );
        }
    }

    // ============================================================
    // short_hash
    // ============================================================

    #[test]
    fn test_short_hash_empty_string() {
        let result = short_hash("");
        assert!(!result.is_empty(), "empty string hash should not be empty");
    }

    #[test]
    fn test_short_hash_deterministic() {
        let input = "hello world";
        let first = short_hash(input);
        let second = short_hash(input);
        assert_eq!(first, second, "short_hash is not deterministic");
    }

    #[test]
    fn test_short_hash_different_inputs_differ() {
        // Different ASCII strings should (almost certainly) produce different hashes
        let a = short_hash("foo");
        let b = short_hash("bar");
        assert_ne!(a, b, "different inputs produced same hash");
    }

    #[test]
    fn test_short_hash_similar_inputs_differ() {
        // Small differences should produce different hashes
        let a = short_hash("abc");
        let b = short_hash("abd");
        assert_ne!(a, b, "similar inputs produced same hash");
    }

    #[test]
    fn test_short_hash_unicode() {
        // Emoji and non-ASCII characters should not panic
        let result = short_hash("Hello 🙈 World 👋");
        assert!(!result.is_empty(), "unicode hash should not be empty");
        // Deterministic check
        assert_eq!(short_hash("Hello 🙈 World 👋"), result);
    }

    #[test]
    fn test_short_hash_long_string() {
        // Very long input should not panic
        let long = "a".repeat(10_000);
        let result = short_hash(&long);
        assert!(!result.is_empty());
        assert_eq!(short_hash(&long), result);
    }

    #[test]
    fn test_short_hash_case_sensitive() {
        // Case sensitivity matters
        let lower = short_hash("hello");
        let upper = short_hash("HELLO");
        assert_ne!(lower, upper, "case-sensitive inputs should differ");
    }

    #[test]
    fn test_short_hash_output_format() {
        // Output should be base-36: only 0-9 and a-z
        let result = short_hash("test format");
        assert!(
            result.chars().all(|c| c.is_ascii_alphanumeric()),
            "hash '{}' contains non-alphanumeric characters",
            result
        );
        assert!(result
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()));
    }

    #[test]
    fn test_short_hash_reasonable_length() {
        // u64 in base-36 is at most 13 chars, so hash should be ≤ 13
        let result = short_hash("any reasonable length string");
        assert!(
            result.len() <= 13,
            "hash '{}' length {} exceeds 13",
            result,
            result.len()
        );
    }
}
