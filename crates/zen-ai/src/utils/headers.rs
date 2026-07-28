use std::collections::HashMap;

/// Converts `reqwest::header::HeaderMap` to `HashMap<String, String>`.
pub fn headers_to_record(headers: &reqwest::header::HeaderMap) -> HashMap<String, String> {
    let mut result = HashMap::new();
    for (key, value) in headers.iter() {
        // HeaderValue::to_str() returns Err for non-UTF-8 values
        if let Ok(v) = value.to_str() {
            result.insert(key.as_str().to_string(), v.to_string());
        }
    }
    result
}

/// Converts optional provider custom headers into `HashMap<String, String>`, filtering out null values.
pub fn provider_headers_to_record(
    headers: Option<&HashMap<String, Option<String>>>,
) -> Option<HashMap<String, String>> {
    let map = headers?;
    let result: HashMap<String, String> = map
        .iter()
        .filter_map(|(key, value)| {
            // Filter out null values (None), keep entries that have a value
            value.as_ref().map(|v| (key.clone(), v.clone()))
        })
        .collect();
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

    // ── headers_to_record ──────────────────────────────────────

    #[test]
    fn test_headers_to_record_empty() {
        let headers = HeaderMap::new();
        let result = headers_to_record(&headers);
        assert!(result.is_empty(), "empty HeaderMap should produce empty HashMap");
    }

    #[test]
    fn test_headers_to_record_multiple_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("content-type", "application/json".parse().unwrap());
        headers.insert("authorization", "Bearer token123".parse().unwrap());

        let result = headers_to_record(&headers);
        assert_eq!(result.len(), 2);
        assert_eq!(result.get("content-type").unwrap(), "application/json");
        assert_eq!(result.get("authorization").unwrap(), "Bearer token123");
    }

    #[test]
    fn test_headers_to_record_skips_non_utf8_values() {
        let mut headers = HeaderMap::new();
        headers.insert("valid", "hello".parse().unwrap());
        // 0x80 is an ISO-8859-1 extended character; valid for HTTP but not UTF-8
        let invalid = HeaderValue::from_bytes(&[0x80]).unwrap();
        headers.insert(HeaderName::from_static("invalid"), invalid);

        let result = headers_to_record(&headers);
        assert_eq!(result.len(), 1, "non-UTF-8 header should be skipped");
        assert_eq!(result.get("valid").unwrap(), "hello");
        assert!(result.get("invalid").is_none(), "invalid header should not appear");
    }

    // ── provider_headers_to_record ─────────────────────────────

    #[test]
    fn test_provider_headers_to_record_none_input() {
        let result = provider_headers_to_record(None);
        assert_eq!(result, None, "None input should return None");
    }

    #[test]
    fn test_provider_headers_to_record_empty_map() {
        let headers: HashMap<String, Option<String>> = HashMap::new();
        let result = provider_headers_to_record(Some(&headers));
        assert_eq!(result, None, "empty map should return None");
    }

    #[test]
    fn test_provider_headers_to_record_all_null_values() {
        let mut headers: HashMap<String, Option<String>> = HashMap::new();
        headers.insert("a".to_string(), None);
        headers.insert("b".to_string(), None);
        let result = provider_headers_to_record(Some(&headers));
        assert_eq!(result, None, "all null values should return None");
    }

    #[test]
    fn test_provider_headers_to_record_filters_nulls() {
        let mut headers: HashMap<String, Option<String>> = HashMap::new();
        headers.insert("keep".to_string(), Some("value".to_string()));
        headers.insert("drop".to_string(), None);
        headers.insert("also_keep".to_string(), Some("also".to_string()));

        let result = provider_headers_to_record(Some(&headers)).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result.get("keep").unwrap(), "value");
        assert_eq!(result.get("also_keep").unwrap(), "also");
        assert!(result.get("drop").is_none(), "null value should be filtered out");
    }

    #[test]
    fn test_provider_headers_to_record_all_non_null() {
        let mut headers: HashMap<String, Option<String>> = HashMap::new();
        headers.insert("x".to_string(), Some("1".to_string()));
        headers.insert("y".to_string(), Some("2".to_string()));

        let result = provider_headers_to_record(Some(&headers)).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result.get("x").unwrap(), "1");
        assert_eq!(result.get("y").unwrap(), "2");
    }

    #[test]
    fn test_provider_headers_to_record_empty_string_value() {
        let mut headers: HashMap<String, Option<String>> = HashMap::new();
        headers.insert("empty".to_string(), Some(String::new()));

        let result = provider_headers_to_record(Some(&headers)).unwrap();
        assert_eq!(result.len(), 1, "empty string should be kept (not null)");
        assert_eq!(result.get("empty").unwrap(), "");
    }
}

