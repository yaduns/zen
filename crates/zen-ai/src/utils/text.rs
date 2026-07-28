use crate::types::{ContentBlock, ContentBlockList};

/// Extract all text blocks from the message content block slice, join them with `separator`, and return.
///
/// Only filter the `ContentBlock::Text` variant, extract its `text` field, and ignore `Thinking`/`Image`/`ToolCall` variants.
///
/// # Parameters
///
/// * `content` — slice of message content blocks
///
/// * `separator` — separator between blocks, defaults to `"\n"`.
///
/// # Return Value
///
/// The concatenated string of all text block contents. Returns an empty string `""` if there are no text blocks.
///
/// # Edge cases
///
/// - Empty slice: returns `""`.
/// - No `Text` variant (only Thinking/Image/ToolCall): returns `""`.
/// - Single text block: returns that block's content (no separator added).
/// - Consecutive text blocks: extracted in order and joined with the separator.
pub fn content_text(content: &[ContentBlock], separator: &str) -> String {
    content.iter().fold(String::new(), |mut acc, inner| {
        if let ContentBlock::Text(text) = inner {
            if !acc.is_empty() {
                acc.push_str(separator);
            }
            acc.push_str(&text.text);
        }
        acc
    })
}

/// Extract text content from a `ContentBlockList`.
///
/// Wraps the explicit enum match for `ContentBlockList`:
/// - If it is `ContentBlockList::Text(s)`, directly return a clone of `s`.
/// - If it is `ContentBlockList::Blocks(blocks)`, call `content_text` to extract and join.
pub fn content_text_from_list(content: &ContentBlockList, separator: &str) -> String {
    match content {
        ContentBlockList::Text(text) => text.clone(),
        ContentBlockList::Blocks(blocks) => content_text(blocks, separator),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ImageContent, TextContent, ThinkingContent, ToolCall};

    fn make_text(text: &str) -> ContentBlock {
        ContentBlock::Text(TextContent {
            text: text.to_string(),
            text_signature: None,
        })
    }

    fn make_thinking() -> ContentBlock {
        ContentBlock::Thinking(ThinkingContent {
            thinking: "reasoning".to_string(),
            thinking_signature: None,
            redacted: None,
        })
    }

    fn make_image() -> ContentBlock {
        ContentBlock::Image(ImageContent {
            data: "...".to_string(),
            mime_type: "image/png".to_string(),
        })
    }

    fn make_tool_call() -> ContentBlock {
        ContentBlock::ToolCall(ToolCall {
            id: "1".to_string(),
            name: "read".to_string(),
            arguments: serde_json::Value::Object(Default::default()),
            thought_signature: None,
        })
    }

    // content_text tests

    #[test]
    fn extracts_text_from_mixed_blocks() {
        let content = vec![
            make_thinking(),
            make_text("first"),
            make_tool_call(),
            make_text("second"),
        ];
        assert_eq!(content_text(&content, "\n"), "first\nsecond");
    }

    #[test]
    fn supports_custom_separator() {
        let content = vec![make_text("first"), make_text("second")];
        assert_eq!(content_text(&content, ""), "firstsecond");
    }

    #[test]
    fn empty_slice_returns_empty_string() {
        let content: Vec<ContentBlock> = vec![];
        assert_eq!(content_text(&content, "\n"), "");
    }

    #[test]
    fn no_text_blocks_returns_empty_string() {
        let content = vec![make_thinking(), make_image(), make_tool_call()];
        assert_eq!(content_text(&content, "\n"), "");
    }

    #[test]
    fn single_text_block_no_separator_appended() {
        let content = vec![make_text("hello")];
        assert_eq!(content_text(&content, "\n"), "hello");
    }

    #[test]
    fn extracts_text_ignoring_non_text_blocks() {
        // Mixed text and non-text blocks; non-text blocks produce no separator placeholder
        let content = vec![
            make_text("first"),
            make_image(),
            make_text("second"),
            make_thinking(),
            make_text("third"),
        ];
        assert_eq!(content_text(&content, "\n"), "first\nsecond\nthird");
    }

    #[test]
    fn consecutive_text_blocks_joined_with_separator() {
        let content = vec![make_text("a"), make_text("b"), make_text("c")];
        assert_eq!(content_text(&content, ", "), "a, b, c");
    }

    // content_text_from_list tests

    #[test]
    fn from_list_text_variant_returns_clone() {
        let list = ContentBlockList::Text("hello".to_string());
        assert_eq!(content_text_from_list(&list, "\n"), "hello");
    }

    #[test]
    fn from_list_blocks_variant_extracts_text() {
        let list = ContentBlockList::Blocks(vec![
            make_text("first"),
            make_image(),
            make_text("second"),
        ]);
        assert_eq!(content_text_from_list(&list, "\n"), "first\nsecond");
    }

    #[test]
    fn from_list_blocks_empty_returns_empty() {
        let list = ContentBlockList::Blocks(vec![]);
        assert_eq!(content_text_from_list(&list, "\n"), "");
    }

    #[test]
    fn from_list_blocks_no_text_returns_empty() {
        let list = ContentBlockList::Blocks(vec![make_thinking(), make_image()]);
        assert_eq!(content_text_from_list(&list, "\n"), "");
    }
}
