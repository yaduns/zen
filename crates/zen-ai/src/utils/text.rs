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
