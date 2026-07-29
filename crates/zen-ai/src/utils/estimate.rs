use crate::types::{
    ContentBlock, ContentBlockList, Context, Message, StopReason, Tool, Usage,
};

/// Estimated char-to-token ratio: every 4 characters ≈ 1 token.
pub const CHARS_PER_TOKEN: usize = 4;

/// Fixed estimated char count for a single image (~1200 tokens), corresponding to Pi's `ESTIMATED_IMAGE_CHARS = 4800`.
pub const ESTIMATED_IMAGE_CHARS: usize = 4800;

/// Represents the breakdown of the context token estimate, distinguishing "accurate token count reported by the latest assistant usage" from "estimated token count of subsequent messages", used for compression scheduling decisions.
/// 1. How many tokens is the current context in total? (tokens)
/// 2. What is the exact usage returned by the most recent API call? (usage_tokens)
/// 3. How many messages have been added since then, estimated? (trailing_tokens)
pub struct ContextUsageEstimate {
    /// Estimated total context token count (usage_tokens + trailing_tokens, plus prefix if present)
    tokens: usize,
    /// Token count reported by the usage field of the most recent available assistant message (calculate_context_tokens result), or 0 if none
    usage_tokens: usize,
    /// Estimated token count of messages after last_usage_index (equal to tokens if last_usage_index is None),
    trailing_tokens: usize,
    /// Index of the assistant message providing the usage within the message list, or None if none
    last_usage_index: Option<usize>,
}

// Compute total token count from the `Usage` struct.
/// Preferentially use the `total_tokens` field; if it is 0, fall back to
/// `input + output + cache_read + cache_write`。
pub fn calculate_context_tokens(usage: &Usage) -> usize {
    if usage.total_tokens == 0 {
        let total = usage.input + usage.output + usage.cache_read + usage.cache_write;
        total as usize
    } else {
        usage.total_tokens as usize
    }
}

/// Estimate the token count of plain text.
/// Formula: `ceil(text.len() / CHARS_PER_TOKEN)`.
pub fn estimate_text_tokens(text: &str) -> usize {
    text.len().div_ceil(CHARS_PER_TOKEN)
}

/// Estimate the token count of a batch of content blocks (text and image blocks only).
/// For each content block:
/// - `Text`: estimated by character count (`ceil(text.len() / CHARS_PER_TOKEN)`).
/// - `Image`: fixed estimate of `ceil(ESTIMATED_IMAGE_CHARS / CHARS_PER_TOKEN)`.
pub fn estimate_content_blocks_tokens(blocks: &[ContentBlock]) -> usize {
    blocks
        .iter()
        .map(|block| match block {
            ContentBlock::Text(text) => estimate_text_tokens(&text.text),
            ContentBlock::Image(_) => ESTIMATED_IMAGE_CHARS.div_ceil(CHARS_PER_TOKEN),
            ContentBlock::Thinking(thinging_content) => {
                estimate_text_tokens(&thinging_content.thinking)
            }
            ContentBlock::ToolCall(tool_call) => {
                estimate_text_tokens(&tool_call.name)
                    + estimate_text_tokens(&tool_call.arguments.to_string())
            }
        })
        .sum()
}

/// Estimate the token count of a single message.
///
/// Handle each variant of the `Message` enum separately:
/// - `Message::User`: call `estimate_content_blocks_tokens` to estimate the user message content.
/// - `Message::Assistant`: iterate through content blocks, using character count for `Text` and `Thinking` blocks,
///   `ToolCall` blocks are estimated by `name` length + serialized `arguments` length.
/// - `Message::ToolResult`: same as `Message::User`, calls `estimate_content_blocks_tokens`.
pub fn estimate_message_tokens(message: &Message) -> usize {
    match message {
        Message::User(user_message) => match &user_message.content {
            ContentBlockList::Text(text) => estimate_text_tokens(&text),
            ContentBlockList::Blocks(block_list) => estimate_content_blocks_tokens(&block_list),
        },
        Message::Assistant(assistant_message) => {
            estimate_content_blocks_tokens(&assistant_message.content)
        }
        Message::ToolResult(tool_result_message) => {
            estimate_content_blocks_tokens(&tool_result_message.content)
        }
    }
}

/// Estimates the token count of the full context.
///
/// This is the outer main entry point, taking a `Context` reference. Internal logic:
/// 1. Calls `estimate_messages` to get the estimate for the message portion.
/// 2. If `last_usage_index` exists, checks whether any tool result messages after that index introduce new tool names
///    (`added_tool_names`); if so, adds the token estimate for those new tools.
/// 3. If `last_usage_index` does not exist, adds the token estimate of `system_prompt` and `tools` as a prefix.
pub fn estimate_context_tokens(context: &Context) -> ContextUsageEstimate {
    let estimate = estimate_messages(&context.messages);

    if let Some(last_index) = estimate.last_usage_index {
        // Has usage: usage already includes system_prompt + original tools
        // Patch: new tools introduced by subsequent tool results
        let added_names: std::collections::HashSet<String> = context.messages[last_index + 1..]
            .iter()
            .filter_map(|m| {
                if let Message::ToolResult(tr) = m {
                    tr.added_tool_names.as_ref()
                } else {
                    None
                }
            })
            .flatten()
            .cloned()
            .collect();

        let added_tool_tokens = if added_names.is_empty() {
            0
        } else {
            let new_tools: Vec<&Tool> = context
                .tools
                .as_ref()
                .map(|tools| {
                    tools
                        .iter()
                        .filter(|t| added_names.contains(&t.name))
                        .collect()
                })
                .unwrap_or_default();
            if new_tools.is_empty() {
                0
            } else {
                let json = serde_json::to_string(&new_tools).unwrap_or_default();
                estimate_text_tokens(&json)
            }
        };

        ContextUsageEstimate {
            tokens: estimate.tokens + added_tool_tokens,
            usage_tokens: estimate.usage_tokens,
            trailing_tokens: estimate.trailing_tokens + added_tool_tokens,
            last_usage_index: estimate.last_usage_index,
        }
    } else {
        // No usage: system_prompt + tools are not in the messages, patch them in manually
        let system_tokens = context
            .system_prompt
            .as_ref()
            .map(|s| estimate_text_tokens(s))
            .unwrap_or(0);

        let tools_tokens = context
            .tools
            .as_ref()
            .map(|tools| {
                if tools.is_empty() {
                    0
                } else {
                    let json = serde_json::to_string(tools).unwrap_or_default();
                    estimate_text_tokens(&json)
                }
            })
            .unwrap_or(0);

        let prefix_tokens = system_tokens + tools_tokens;

        ContextUsageEstimate {
            tokens: estimate.tokens + prefix_tokens,
            usage_tokens: estimate.usage_tokens,
            trailing_tokens: estimate.trailing_tokens + prefix_tokens,
            last_usage_index: estimate.last_usage_index,
        }
    }
}

/// Estimates only the token count of the message list (without the context's system_prompt and tools).
pub fn estimate_messages(messages: &[Message]) -> ContextUsageEstimate {
    // Find the last assistant message with valid usage
    let mut latest_prefix_ts = i64::MIN;
    let mut usage_info: Option<(usize, &Usage)> = None;

    for (i, message) in messages.iter().enumerate() {
        let msg_ts = match message {
            Message::User(m) => m.timestamp,
            Message::Assistant(m) => m.timestamp,
            Message::ToolResult(m) => m.timestamp,
        };

        if let Message::Assistant(assistant) = message {
            let usage_applies = assistant.timestamp >= latest_prefix_ts;
            if usage_applies
                && assistant.stop_reason != StopReason::Aborted
                && assistant.stop_reason != StopReason::Error
                && calculate_context_tokens(&assistant.usage) > 0
            {
                usage_info = Some((i, &assistant.usage));
            }
        }

        latest_prefix_ts = latest_prefix_ts.max(msg_ts);
    }

    if let Some((idx, usage)) = usage_info {
        let usage_tokens = calculate_context_tokens(usage);
        let trailing_tokens: usize = messages[idx + 1..]
            .iter()
            .map(|m| estimate_message_tokens(m))
            .sum();

        ContextUsageEstimate {
            tokens: usage_tokens + trailing_tokens,
            usage_tokens,
            trailing_tokens,
            last_usage_index: Some(idx),
        }
    } else {
        let tokens: usize = messages.iter().map(|m| estimate_message_tokens(m)).sum();
        ContextUsageEstimate {
            tokens,
            usage_tokens: 0,
            trailing_tokens: tokens,
            last_usage_index: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        AssistantMessage, Cost, ImageContent, TextContent, ThinkingContent, ToolCall,
        ToolResultMessage, UserMessage,
    };
    use serde_json::json;

    // ── calculate_context_tokens ────────────────────────────────

    #[test]
    fn test_calculate_context_tokens_uses_total_tokens_when_nonzero() {
        let usage = Usage {
            total_tokens: 100,
            input: 50,
            output: 30,
            cache_read: 10,
            cache_write: 10,
            cost: Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
            cache_write_1h: None,
            reasoning: None,
        };
        assert_eq!(calculate_context_tokens(&usage), 100);
    }

    #[test]
    fn test_calculate_context_tokens_fallback_when_total_tokens_zero() {
        let usage = Usage {
            total_tokens: 0,
            input: 50,
            output: 30,
            cache_read: 10,
            cache_write: 5,
            cost: Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
            cache_write_1h: None,
            reasoning: None,
        };
        assert_eq!(calculate_context_tokens(&usage), 95); // 50 + 30 + 10 + 5
    }

    #[test]
    fn test_calculate_context_tokens_all_zero() {
        let usage = Usage {
            total_tokens: 0,
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            cost: Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
            cache_write_1h: None,
            reasoning: None,
        };
        assert_eq!(calculate_context_tokens(&usage), 0);
    }

    // ── estimate_text_tokens ───────────────────────────────────

    #[test]
    fn test_estimate_text_tokens_empty() {
        assert_eq!(estimate_text_tokens(""), 0);
    }

    #[test]
    fn test_estimate_text_tokens_exact_multiple() {
        // 8 chars -> ceil(8/4) = 2
        assert_eq!(estimate_text_tokens("12345678"), 2);
    }

    #[test]
    fn test_estimate_text_tokens_not_multiple() {
        // 5 chars -> ceil(5/4) = 2
        assert_eq!(estimate_text_tokens("12345"), 2);
        // 1 char -> ceil(1/4) = 1
        assert_eq!(estimate_text_tokens("1"), 1);
    }

    #[test]
    fn test_estimate_text_tokens_boundary() {
        // 4 chars -> ceil(4/4) = 1
        assert_eq!(estimate_text_tokens("abcd"), 1);
        // 3 chars -> ceil(3/4) = 1
        assert_eq!(estimate_text_tokens("abc"), 1);
    }

    // ── estimate_content_blocks_tokens ──────────────────────────

    #[test]
    fn test_estimate_content_blocks_tokens_empty() {
        assert_eq!(estimate_content_blocks_tokens(&[]), 0);
    }

    #[test]
    fn test_estimate_content_blocks_tokens_text_blocks() {
        let blocks = vec![
            ContentBlock::Text(TextContent {
                text: "hello".to_string(),          // 5 -> ceil(5/4) = 2
                text_signature: None,
            }),
            ContentBlock::Text(TextContent {
                text: "world".to_string(),          // 5 -> ceil(5/4) = 2
                text_signature: None,
            }),
        ];
        assert_eq!(estimate_content_blocks_tokens(&blocks), 4);
    }

    #[test]
    fn test_estimate_content_blocks_tokens_image_blocks() {
        let blocks = vec![
            ContentBlock::Image(ImageContent {
                data: "base64data".to_string(),
                mime_type: "image/png".to_string(),
            }),
            ContentBlock::Image(ImageContent {
                data: "moredata".to_string(),
                mime_type: "image/jpeg".to_string(),
            }),
        ];
        // Each image: 4800/4 = 1200 tokens
        assert_eq!(estimate_content_blocks_tokens(&blocks), 2400);
    }

    #[test]
    fn test_estimate_content_blocks_tokens_mixed() {
        let blocks = vec![
            ContentBlock::Text(TextContent {
                text: "abcd".to_string(),            // 4 -> ceil(4/4) = 1
                text_signature: None,
            }),
            ContentBlock::Image(ImageContent {
                data: "img".to_string(),
                mime_type: "image/png".to_string(),
            }),                                      // 4800/4 = 1200
        ];
        assert_eq!(estimate_content_blocks_tokens(&blocks), 1201);
    }

    #[test]
    fn test_estimate_content_blocks_tokens_thinking() {
        let blocks = vec![ContentBlock::Thinking(ThinkingContent {
            thinking: "deep thought".to_string(),  // 12 -> ceil(12/4) = 3
            thinking_signature: None,
            redacted: None,
        })];
        assert_eq!(estimate_content_blocks_tokens(&blocks), 3);
    }

    #[test]
    fn test_estimate_content_blocks_tokens_tool_call() {
        let blocks = vec![ContentBlock::ToolCall(ToolCall {
            id: "call_1".to_string(),
            name: "read_file".to_string(),    // 8 -> ceil(8/4) = 2
            arguments: json!({"path": "/tmp/test.txt"}),  // ~26 chars -> ceil(26/4) = 7
            thought_signature: None,
        })];
        // 2 + 7 = 9
        assert_eq!(estimate_content_blocks_tokens(&blocks), 9);
    }

    // ── estimate_message_tokens ─────────────────────────────────

    #[test]
    fn test_estimate_message_tokens_user_text() {
        let msg = Message::User(UserMessage {
            content: ContentBlockList::Text("hello world".to_string()),  // 11 -> ceil(11/4) = 3
            timestamp: 1000,
        });
        assert_eq!(estimate_message_tokens(&msg), 3);
    }

    #[test]
    fn test_estimate_message_tokens_user_blocks() {
        let msg = Message::User(UserMessage {
            content: ContentBlockList::Blocks(vec![
                ContentBlock::Text(TextContent {
                    text: "hello".to_string(),     // 5 -> ceil(5/4) = 2
                    text_signature: None,
                }),
                ContentBlock::Image(ImageContent {
                    data: "data".to_string(),
                    mime_type: "image/png".to_string(),
                }),                                 // 4800/4 = 1200
            ]),
            timestamp: 1000,
        });
        assert_eq!(estimate_message_tokens(&msg), 1202);
    }

    #[test]
    fn test_estimate_message_tokens_assistant() {
        let msg = Message::Assistant(AssistantMessage {
            content: vec![
                ContentBlock::Text(TextContent {
                    text: "abcd".to_string(),       // 4 -> 1
                    text_signature: None,
                }),
                ContentBlock::Thinking(ThinkingContent {
                    thinking: "Thinking".to_string(),    // 6 bytes -> ceil(6/4) = 2
                    thinking_signature: None,
                    redacted: None,
                }),
            ],
            timestamp: 2000,
            stop_reason: StopReason::Stop,
            usage: Usage {
                total_tokens: 50,
                input: 40,
                output: 10,
                cache_read: 0,
                cache_write: 0,
                cost: Cost {
                    input: 0.0,
                    output: 0.0,
                    cache_read: 0.0,
                    cache_write: 0.0,
                    total: 0.0,
                },
                cache_write_1h: None,
                reasoning: None,
            },
            api: "openai-completions".to_string(),
            provider: "deepseek".to_string(),
            model: "deepseek-v4-flash".to_string(),
            response_model: None,
            response_id: None,
            error_message: None,
        });
        assert_eq!(estimate_message_tokens(&msg), 3);  // 1 + 2
    }

    #[test]
    fn test_estimate_message_tokens_tool_result() {
        let msg = Message::ToolResult(ToolResultMessage {
            tool_call_id: "call_1".to_string(),
            tool_name: "read".to_string(),
            content: vec![ContentBlock::Text(TextContent {
                text: "file content".to_string(),  // 12 -> ceil(12/4) = 3
                text_signature: None,
            })],
            details: None,
            usage: Usage {
                total_tokens: 0,
                input: 0,
                output: 0,
                cache_read: 0,
                cache_write: 0,
                cost: Cost {
                    input: 0.0,
                    output: 0.0,
                    cache_read: 0.0,
                    cache_write: 0.0,
                    total: 0.0,
                },
                cache_write_1h: None,
                reasoning: None,
            },
            is_error: false,
            timestamp: 3000,
            added_tool_names: None,
        });
        assert_eq!(estimate_message_tokens(&msg), 3);
    }

    // ── estimate_messages ──────────────────────────────────────

    #[test]
    fn test_estimate_messages_empty() {
        let result = estimate_messages(&[]);
        assert_eq!(result.tokens, 0);
        assert_eq!(result.usage_tokens, 0);
        assert_eq!(result.trailing_tokens, 0);
        assert_eq!(result.last_usage_index, None);
    }

    #[test]
    fn test_estimate_messages_no_assistant_usage() {
        let messages = vec![
            Message::User(UserMessage {
                content: ContentBlockList::Text("hello".to_string()),  // 5 -> 2
                timestamp: 1000,
            }),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: "c1".to_string(),
                tool_name: "read".to_string(),
                content: vec![ContentBlock::Text(TextContent {
                    text: "result".to_string(),                        // 6 -> 2
                    text_signature: None,
                })],
                details: None,
                usage: Usage {
                    total_tokens: 0,
                    input: 0,
                    output: 0,
                    cache_read: 0,
                    cache_write: 0,
                    cost: Cost {
                        input: 0.0,
                        output: 0.0,
                        cache_read: 0.0,
                        cache_write: 0.0,
                        total: 0.0,
                    },
                    cache_write_1h: None,
                    reasoning: None,
                },
                is_error: false,
                timestamp: 2000,
                added_tool_names: None,
            }),
        ];
        let result = estimate_messages(&messages);
        assert_eq!(result.tokens, 4);   // 2 + 2
        assert_eq!(result.usage_tokens, 0);
        assert_eq!(result.trailing_tokens, 4);
        assert_eq!(result.last_usage_index, None);
    }

    #[test]
    fn test_estimate_messages_with_usage() {
        let messages = vec![
            Message::User(UserMessage {
                content: ContentBlockList::Text("hi".to_string()),
                timestamp: 1000,
            }),
            Message::Assistant(AssistantMessage {
                content: vec![ContentBlock::Text(TextContent {
                    text: "response".to_string(),
                    text_signature: None,
                })],
                timestamp: 2000,
                stop_reason: StopReason::Stop,
                usage: Usage {
                    total_tokens: 30,
                    input: 20,
                    output: 10,
                    cache_read: 0,
                    cache_write: 0,
                    cost: Cost {
                        input: 0.0,
                        output: 0.0,
                        cache_read: 0.0,
                        cache_write: 0.0,
                        total: 0.0,
                    },
                    cache_write_1h: None,
                    reasoning: None,
                },
                api: "openai".to_string(),
                provider: "deepseek".to_string(),
                model: "m".to_string(),
                response_model: None,
                response_id: None,
                error_message: None,
            }),
            Message::User(UserMessage {
                content: ContentBlockList::Text("follow up".to_string()),  // 9 -> 3
                timestamp: 3000,
            }),
        ];
        let result = estimate_messages(&messages);
        // usage_tokens = 30, trailing_tokens = 3 (follow up), tokens = 33
        assert_eq!(result.usage_tokens, 30);
        assert_eq!(result.trailing_tokens, 3);
        assert_eq!(result.tokens, 33);
        assert_eq!(result.last_usage_index, Some(1));
    }

    #[test]
    fn test_estimate_messages_skips_aborted_usage() {
        let messages = vec![
            Message::Assistant(AssistantMessage {
                content: vec![],
                timestamp: 1000,
                stop_reason: StopReason::Aborted,
                usage: Usage {
                    total_tokens: 100,
                    input: 0,
                    output: 0,
                    cache_read: 0,
                    cache_write: 0,
                    cost: Cost {
                        input: 0.0,
                        output: 0.0,
                        cache_read: 0.0,
                        cache_write: 0.0,
                        total: 0.0,
                    },
                    cache_write_1h: None,
                    reasoning: None,
                },
                api: "o".to_string(),
                provider: "d".to_string(),
                model: "m".to_string(),
                response_model: None,
                response_id: None,
                error_message: None,
            }),
        ];
        let result = estimate_messages(&messages);
        // Aborted assistant should be skipped for usage, falls back to per-message estimation
        assert_eq!(result.last_usage_index, None);
        assert_eq!(result.usage_tokens, 0);
        assert_eq!(result.tokens, 0);  // content is empty
    }

    // ── estimate_context_tokens ─────────────────────────────────

    #[test]
    fn test_estimate_context_tokens_with_system_prompt_and_tools() {
        let context = Context {
            system_prompt: Some("You are a helpful assistant".to_string()),  // 29 -> 8
            messages: vec![
                Message::User(UserMessage {
                    content: ContentBlockList::Text("hello".to_string()),    // 5 -> 2
                    timestamp: 1000,
                }),
            ],
            tools: Some(vec![
                Tool {
                    name: "read".to_string(),
                    description: "Read a file".to_string(),
                    parameters: json!({"type": "object", "properties": {}}),
                },
            ]),
        };
        let result = estimate_context_tokens(&context);
        // No usage -> prefix = system_prompt + tools + messages
        assert!(result.tokens > 10);
        assert_eq!(result.usage_tokens, 0);
        assert_eq!(result.last_usage_index, None);
    }

    #[test]
    fn test_estimate_context_tokens_empty_context() {
        let context = Context {
            system_prompt: None,
            messages: vec![],
            tools: None,
        };
        let result = estimate_context_tokens(&context);
        assert_eq!(result.tokens, 0);
        assert_eq!(result.usage_tokens, 0);
        assert_eq!(result.trailing_tokens, 0);
        assert_eq!(result.last_usage_index, None);
    }
}