// REQ-AI-02 (pi-101/pi-106): transport-layer error chain flattening (retry classification + user-visible error text)

//! Error-chain flattening for transport failures.
//!
//! Rust's transport stack (reqwest → hyper → io) hides the actionable cause (`connection closed before message
//! completed`, `connection refused`, …) behind a generic wrapper (`error sending request for url (...)`). Both the
//! retry classifier (wording table) and the user-visible error line need the cause, while `eventsource_stream`'s
//! error type does not implement [`std::error::Error::source`] — hence this module walks the chain by hand.

/// Maximum number of layers walked; a pathological cycle or a very deep wrapper stack must not hang or bloat the text.
const MAX_CHAIN_DEPTH: usize = 8;

/// The gateway wraps "the connection was cut" in an uninformative shell (the user only sees
/// `error sending request for url (...)`, and the retry classifier matches nothing) — this function digs the real
/// cause out: like unboxing a parcel, it peels `source()` layer by layer and joins
/// `connection closed before message completed` into the same line.
///
/// # Behavior
/// - Walks `std::error::Error::source()` from `err` downwards, joining each layer's `Display` with `": "` into one
///   line that the retry regexes can match.
/// - When a layer's `Display` is empty (some hyper 1.x errors are), that layer falls back to its `{:?}` so the cause
///   is never swallowed by an empty string.
/// - Consecutive duplicate texts are collapsed (several libraries wrapping the same sentence); depth is capped at
///   [`MAX_CHAIN_DEPTH`].
///
/// # Edge Cases
/// - `err`'s own `Display` is also empty → returns its `{:?}` (noise is preferable to an empty string).
///
/// # Consumers
/// `crate::api::openai_stream::describe_sse_error` (SSE transport layer),
/// [`crate::api::openai_stream::StreamError::describe`] (failure during the HTTP send phase).
pub fn describe_error_chain(err: &(dyn std::error::Error + 'static)) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(err);

    while let Some(layer) = current {
        if parts.len() >= MAX_CHAIN_DEPTH {
            break;
        }
        let text = layer_text(layer);
        if !text.is_empty() && parts.last() != Some(&text) {
            parts.push(text);
        }
        current = layer.source();
    }

    if parts.is_empty() {
        return format!("{err:?}");
    }
    parts.join(": ")
}

/// Renders one layer: `Display` when non-empty, `Debug` otherwise (hyper 1.x errors can have empty `Display`).
fn layer_text(err: &(dyn std::error::Error + 'static)) -> String {
    let display = err.to_string();
    if display.trim().is_empty() {
        format!("{err:?}")
    } else {
        display
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt;

    /// Fake error with an optional cause, used to build deterministic chains.
    struct Fake {
        text: String,
        /// Debug text used when `text` is empty (simulates a hyper-style empty `Display`).
        debug_text: Option<String>,
        source: Option<Box<Fake>>,
    }

    impl Fake {
        fn leaf(text: &str) -> Self {
            Self {
                text: text.to_string(),
                debug_text: None,
                source: None,
            }
        }

        fn wrap(text: &str, source: Fake) -> Self {
            Self {
                text: text.to_string(),
                debug_text: None,
                source: Some(Box::new(source)),
            }
        }

        /// A layer whose `Display` is empty while `Debug` still carries the cause.
        fn silent(debug_text: &str) -> Self {
            Self {
                text: String::new(),
                debug_text: Some(debug_text.to_string()),
                source: None,
            }
        }
    }

    impl fmt::Display for Fake {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(&self.text)
        }
    }

    impl fmt::Debug for Fake {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match &self.debug_text {
                Some(t) => f.write_str(t),
                None => f.write_str(&self.text),
            }
        }
    }

    impl std::error::Error for Fake {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            self.source
                .as_ref()
                .map(|b| b.as_ref() as &(dyn std::error::Error + 'static))
        }
    }

    #[test]
    fn joins_three_layers_with_colon_separator() {
        let err = Fake::wrap("outer", Fake::wrap("middle", Fake::leaf("root")));
        assert_eq!(describe_error_chain(&err), "outer: middle: root");
    }

    #[test]
    fn single_layer_returns_its_display() {
        let err = Fake::leaf("boom");
        assert_eq!(describe_error_chain(&err), "boom");
    }

    #[test]
    fn consecutive_duplicates_are_collapsed() {
        let err = Fake::wrap("same", Fake::wrap("same", Fake::leaf("root")));
        assert_eq!(describe_error_chain(&err), "same: root");
    }

    #[test]
    fn empty_display_layer_falls_back_to_debug() {
        let err = Fake::wrap("outer", Fake::silent("inner-cause"));
        let text = describe_error_chain(&err);
        assert!(text.starts_with("outer: "), "actual: {text}");
        assert!(text.contains("inner-cause"), "actual: {text}");
    }

    #[test]
    fn top_level_with_empty_display_returns_debug() {
        let err = Fake::silent("only-cause");
        assert!(describe_error_chain(&err).contains("only-cause"));
    }

    #[test]
    fn deep_chain_is_capped() {
        let mut err = Fake::leaf("leaf");
        for i in 0..(MAX_CHAIN_DEPTH + 4) {
            err = Fake::wrap(&format!("layer{i}"), err);
        }
        let text = describe_error_chain(&err);
        assert!(
            text.matches(": ").count() < MAX_CHAIN_DEPTH,
            "actual: {text}"
        );
    }
}
