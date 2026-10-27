use crate::compaction::FileOperations;
use crate::tools::factory::ToolName;
use std::collections::HashSet;
use zen_ai::types::{ContentBlock, Message};
use zen_ai::utils::text::{content_text, content_text_from_list};

/// Extracts file operations from the assistant message's toolCall blocks (the path argument of the read/write/edit tools) and accumulates them into the buckets.
/// Consumers: `prepare_compaction` (mod.rs), `prepare_branch_entries` (branch.rs, T-AG-04-2)
/// (All consumers are inside the zen-core crate; FileOperations is pub(crate), and pub would trigger private_interfaces.)
pub(crate) fn extract_file_ops_from_message(message: &Message, file_ops: &mut FileOperations) {
    let Message::Assistant(assistant) = message else {
        return;
    };
    for block in &assistant.content {
        let ContentBlock::ToolCall(call) = block else {
            continue;
        };
        // For non-object arguments (including null), `get` returns None, equivalent to pi's `if (!args) continue`
        let Some(path) = call.arguments.get("path").and_then(|v| v.as_str()) else {
            continue;
        };
        if path.is_empty() {
            continue;
        }
        let bucket = match ToolName::from_str(&call.name) {
            Some(ToolName::Read) => &mut file_ops.read,
            Some(ToolName::Write) => &mut file_ops.written,
            Some(ToolName::Edit) => &mut file_ops.edited,
            _ => continue, // Unknown tool names are silently ignored (pi's switch default does nothing)
        };
        bucket.insert(path.to_string());
    }
}

/// Computes the final list from the accumulated buckets: modified = edited ∪ written; read minus modified, then sorted.
/// Consumers: `compact` (mod.rs), `generate_branch_summary` (branch.rs, T-AG-04-2);
/// pub(crate) (all consumers are within the zen-core crate; same visibility rationale as extract).
pub(crate) fn compute_file_lists(file_ops: &FileOperations) -> (Vec<String>, Vec<String>) {
    let modified: HashSet<&String> = file_ops.edited.union(&file_ops.written).collect();
    let mut read_files: Vec<String> = file_ops
        .read
        .iter()
        .filter(|path| !modified.contains(*path))
        .cloned()
        .collect();
    let mut modified_files: Vec<String> = modified.into_iter().cloned().collect();
    read_files.sort();
    modified_files.sort();
    (read_files, modified_files)
}

/// Formats the file list into `<read-files>`/`<modified-files>` appendix sections (empty list returns an empty string).
/// Consumers: `compact` (mod.rs), `generate_branch_summary` (branch.rs, T-AG-04-2);
/// pub(crate) (isolated-pub discipline: no consumers outside the crate).
pub(crate) fn format_file_operations(read_files: &[String], modified_files: &[String]) -> String {
    let mut sections: Vec<String> = Vec::new();
    if !read_files.is_empty() {
        sections.push(format!(
            "<read-files>\n{}\n</read-files>",
            read_files.join("\n")
        ));
    }
    if !modified_files.is_empty() {
        sections.push(format!(
            "<modified-files>\n{}\n</modified-files>",
            modified_files.join("\n")
        ));
    }
    if sections.is_empty() {
        return String::new();
    }
    format!("\n\n{}", sections.join("\n\n"))
}

/// Maximum number of characters for a tool result in the summary (aligned with pi's `TOOL_RESULT_MAX_CHARS`).
const TOOL_RESULT_MAX_CHARS: usize = 2000;

/// Placeholder fallback when serialization fails (aligned with the catch branch of pi's `safeJsonStringify`);
/// pi's `?? "undefined"` branch is dissolved: zen `Value` has no undefined form, and `Null` serializes naturally to "null".
fn safe_json_stringify(value: &serde_json::Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "[unserializable]".to_string())
}

/// Truncates overlong text + "[... N more characters truncated]" suffix (aligned with pi's `truncateForSummary`).
/// Counts by Unicode chars (pi counts UTF-16 code units; identical within the BMP; zen docs say "2000 characters", hence the chars measure).
fn truncate_for_summary(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }
    let truncated_chars = char_count - max_chars;
    let prefix: String = text.chars().take(max_chars).collect();
    format!("{prefix}\n\n[... {truncated_chars} more characters truncated]")
}

/// Serializes the LLM messages into a plain-text conversation (`[User]:`/`[Assistant]:`/`[Assistant thinking]:`/
/// `[Assistant tool calls]:`/`[Tool result]:` lines, with tool results truncated to 2000 characters),
/// The `<conversation>` section for the summary prompt — this is the actual trimming compaction performs on messages.
/// Consumers: `generate_summary`/`generate_turn_prefix_summary` (within the crate);
/// pub(crate) for reuse by T-AG-04-2 (branch-summary conversation serialization) (isolated pub discipline).
pub(crate) fn serialize_conversation(messages: &[Message]) -> String {
    let mut parts: Vec<String> = Vec::new();

    for msg in messages {
        match msg {
            Message::User(user) => {
                // pi: contentText(msg.content, "") — no separator between blocks
                let content = content_text_from_list(&user.content, "");
                if !content.is_empty() {
                    parts.push(format!("[User]: {content}"));
                }
            }
            Message::Assistant(assistant) => {
                let mut thinking_parts: Vec<&str> = Vec::new();
                let mut tool_calls: Vec<String> = Vec::new();

                for block in &assistant.content {
                    match block {
                        ContentBlock::Thinking(thinking) => {
                            thinking_parts.push(&thinking.thinking);
                        }
                        ContentBlock::ToolCall(call) => {
                            // pi: Object.entries(args).map(([k, v]) => `${k}=${safeJsonStringify(v)}`)
                            let args_str = call
                                .arguments
                                .as_object()
                                .map(|obj| {
                                    obj.iter()
                                        .map(|(k, v)| format!("{k}={}", safe_json_stringify(v)))
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                })
                                .unwrap_or_default();
                            tool_calls.push(format!("{}({})", call.name, args_str));
                        }
                        _ => {}
                    }
                }

                if !thinking_parts.is_empty() {
                    parts.push(format!(
                        "[Assistant thinking]: {}",
                        thinking_parts.join("\n")
                    ));
                }
                // pi: msg.content.some(block => block.type === "text") — emit this line if any text block exists
                // (Empty text blocks also yield an empty content line; do not change to a non-empty content check, or this edge behavior is lost)
                if assistant
                    .content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::Text(_)))
                {
                    parts.push(format!(
                        "[Assistant]: {}",
                        content_text(&assistant.content, "\n")
                    ));
                }
                if !tool_calls.is_empty() {
                    parts.push(format!("[Assistant tool calls]: {}", tool_calls.join("; ")));
                }
            }
            Message::ToolResult(result) => {
                let content = content_text(&result.content, "");
                if !content.is_empty() {
                    parts.push(format!(
                        "[Tool result]: {}",
                        truncate_for_summary(&content, TOOL_RESULT_MAX_CHARS)
                    ));
                }
            }
        }
    }

    parts.join("\n\n")
}

// Private: `entries_to_messages_for_summarization` in mod.rs — delegates per entry
// `session_entry_to_context_messages` (existing converter in context.rs) and skips Compaction entries
// (pi getMessageFromEntryForCompaction semantics); `find_valid_cut_points` in cut.rs —
// Valid cut point scan (message and role ∈ {user, assistant, bashExecution*, custom, branchSummary,
// compactionSummary} or branch_summary/custom_message entries; toolResult is never valid).
// *zen Message has no bashExecution/custom roles — these pi roles are represented in zen by custom_message entries/
// User messages; cut point checks exhaustively map the zen variant set (CustomMessage entries are cut points, see pi original logic in same list).

#[cfg(test)]
mod tests {
    /// CJK sample kept intentionally: this test verifies wide-character / multibyte behavior.
    const CJK_SAMPLE: &str = "你好你好你好";
    use super::*;
    use serde_json::json;
    use zen_ai::types::{
        AssistantMessage, ContentBlockList, StopReason, TextContent, ThinkingContent, ToolCall,
        ToolResultMessage, UserMessage,
    };

    // ── construction helpers ─────────────────────────────────────────────

    fn tool_call(name: &str, arguments: serde_json::Value) -> ContentBlock {
        ContentBlock::ToolCall(ToolCall {
            id: format!("call-{name}"),
            name: name.to_string(),
            arguments,
            thought_signature: None,
        })
    }

    fn assistant_message(content: Vec<ContentBlock>) -> Message {
        Message::Assistant(AssistantMessage {
            content,
            api: "openai-completions".to_string(),
            provider: "deepseek".to_string(),
            model: "test-model".to_string(),
            response_model: None,
            response_id: None,
            usage: zen_ai::types::Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 0,
        })
    }

    fn text_block(text: &str) -> ContentBlock {
        ContentBlock::Text(TextContent {
            text: text.to_string(),
            text_signature: None,
        })
    }

    fn thinking_block(text: &str) -> ContentBlock {
        ContentBlock::Thinking(ThinkingContent {
            thinking: text.to_string(),
            thinking_signature: None,
            redacted: None,
        })
    }

    fn tool_result(content: &str) -> Message {
        Message::ToolResult(ToolResultMessage {
            tool_call_id: "call-1".to_string(),
            tool_name: "bash".to_string(),
            content: if content.is_empty() {
                vec![]
            } else {
                vec![text_block(content)]
            },
            details: None,
            usage: None,
            is_error: false,
            added_tool_names: None,
            timestamp: 0,
        })
    }

    fn user_message(content: ContentBlockList) -> Message {
        Message::User(UserMessage {
            content,
            timestamp: 0,
        })
    }

    // ── extract_file_ops_from_message ────────────────────────

    #[test]
    fn test_extract_file_ops_ignores_non_assistant_messages() {
        let mut ops = FileOperations::new();
        extract_file_ops_from_message(&user_message(ContentBlockList::Text("hi".into())), &mut ops);
        extract_file_ops_from_message(&tool_result("out"), &mut ops);
        assert!(ops.read.is_empty());
        assert!(ops.written.is_empty());
        assert!(ops.edited.is_empty());
    }

    #[test]
    fn test_extract_file_ops_buckets_by_tool_name() {
        let mut ops = FileOperations::new();
        extract_file_ops_from_message(
            &assistant_message(vec![
                tool_call("read", json!({"path": "a.ts"})),
                tool_call("write", json!({"path": "b.ts"})),
                tool_call("edit", json!({"path": "c.ts"})),
            ]),
            &mut ops,
        );
        assert_eq!(ops.read, HashSet::from(["a.ts".to_string()]));
        assert_eq!(ops.written, HashSet::from(["b.ts".to_string()]));
        assert_eq!(ops.edited, HashSet::from(["c.ts".to_string()]));
    }

    #[test]
    fn test_extract_file_ops_skips_invalid_paths() {
        // Missing path key / non-string path / empty path string — all silently skipped
        let mut ops = FileOperations::new();
        extract_file_ops_from_message(
            &assistant_message(vec![
                tool_call("read", json!({})),            // no path key
                tool_call("read", json!({"path": 123})), // path is not a string
                tool_call("read", json!({"path": ""})),  // path is an empty string
            ]),
            &mut ops,
        );
        assert!(ops.read.is_empty());
    }

    #[test]
    fn test_extract_file_ops_skips_unknown_tool_name() {
        let mut ops = FileOperations::new();
        extract_file_ops_from_message(
            &assistant_message(vec![tool_call("grep", json!({"path": "x.rs"}))]),
            &mut ops,
        );
        assert!(ops.read.is_empty());
        assert!(ops.written.is_empty());
        assert!(ops.edited.is_empty());
    }

    #[test]
    fn test_extract_file_ops_ignores_non_toolcall_blocks_and_dedupes_paths() {
        let mut ops = FileOperations::new();
        extract_file_ops_from_message(
            &assistant_message(vec![
                text_block("reading"),
                thinking_block("hmm"),
                tool_call("read", json!({"path": "a.ts"})),
                tool_call("read", json!({"path": "a.ts"})),
                tool_call("read", json!({"path": "b.ts"})),
            ]),
            &mut ops,
        );
        // HashSet deduplication: two reads of the same path occupy only one slot
        assert_eq!(
            ops.read,
            HashSet::from(["a.ts".to_string(), "b.ts".to_string()])
        );
    }

    // ── compute_file_lists ───────────────────────────────────

    #[test]
    fn test_compute_file_lists_empty() {
        let (read, modified) = compute_file_lists(&FileOperations::new());
        assert!(read.is_empty());
        assert!(modified.is_empty());
    }

    #[test]
    fn test_compute_file_lists_excludes_modified_from_read() {
        let mut ops = FileOperations::new();
        ops.read = HashSet::from(["a.ts".to_string(), "b.ts".to_string()]);
        ops.written = HashSet::from(["b.ts".to_string()]);
        ops.edited = HashSet::from(["c.ts".to_string()]);
        let (read, modified) = compute_file_lists(&ops);
        // modified = edited ∪ written; read minus modified
        assert_eq!(read, vec!["a.ts".to_string()]);
        assert_eq!(modified, vec!["b.ts".to_string(), "c.ts".to_string()]);
    }

    #[test]
    fn test_compute_file_lists_sorts_output() {
        let mut ops = FileOperations::new();
        ops.read = HashSet::from(["z.ts".to_string(), "a.ts".to_string()]);
        ops.edited = HashSet::from(["m.ts".to_string(), "b.ts".to_string()]);
        let (read, modified) = compute_file_lists(&ops);
        assert_eq!(read, vec!["a.ts".to_string(), "z.ts".to_string()]);
        assert_eq!(modified, vec!["b.ts".to_string(), "m.ts".to_string()]);
    }

    // ── format_file_operations ───────────────────────────────

    #[test]
    fn test_format_file_operations_empty_returns_empty_string() {
        assert_eq!(format_file_operations(&[], &[]), "");
    }

    #[test]
    fn test_format_file_operations_read_only() {
        assert_eq!(
            format_file_operations(&["a.ts".to_string()], &[]),
            "\n\n<read-files>\na.ts\n</read-files>"
        );
    }

    #[test]
    fn test_format_file_operations_modified_only() {
        assert_eq!(
            format_file_operations(&[], &["b.ts".to_string()]),
            "\n\n<modified-files>\nb.ts\n</modified-files>"
        );
    }

    #[test]
    fn test_format_file_operations_joins_both_sections() {
        assert_eq!(
            format_file_operations(&["a.ts".to_string()], &["b.ts".to_string()]),
            "\n\n<read-files>\na.ts\n</read-files>\n\n<modified-files>\nb.ts\n</modified-files>"
        );
    }

    // ── safe_json_stringify (private, unit-tested directly) ──────────────────

    #[test]
    fn test_safe_json_stringify_serializes_values() {
        assert_eq!(safe_json_stringify(&json!({"a": 1})), r#"{"a":1}"#);
        assert_eq!(safe_json_stringify(&json!([1, 2])), "[1,2]");
        assert_eq!(safe_json_stringify(&json!("s")), r#""s""#);
        assert_eq!(safe_json_stringify(&serde_json::Value::Null), "null");
    }

    // ── truncate_for_summary (private, unit-tested directly) ─────────────────

    #[test]
    fn test_truncate_for_summary_within_limit_unchanged() {
        assert_eq!(truncate_for_summary("abc", 4), "abc");
        // Exactly at the limit: no truncation
        assert_eq!(truncate_for_summary("abcd", 4), "abcd");
    }

    #[test]
    fn test_truncate_for_summary_truncates_with_tail() {
        let text = "0123456789"; // 10 characters
        assert_eq!(
            truncate_for_summary(text, 4),
            "0123\n\n[... 6 more characters truncated]"
        );
    }

    #[test]
    fn test_truncate_for_summary_counts_unicode_chars_not_bytes() {
        // Three CJK chars = 6 characters (18 bytes in UTF-8) — counted by characters
        let text = CJK_SAMPLE;
        assert_eq!(
            truncate_for_summary(text, 4),
            "你好你好\n\n[... 2 more characters truncated]"
        );
    }

    // ── serialize_conversation ───────────────────────────────

    #[test]
    fn test_serialize_conversation_empty_returns_empty_string() {
        let messages: Vec<Message> = vec![];
        assert_eq!(serialize_conversation(&messages), "");
    }

    #[test]
    fn test_serialize_conversation_user_line() {
        let messages = vec![user_message(ContentBlockList::Text("hello".into()))];
        assert_eq!(serialize_conversation(&messages), "[User]: hello");
    }

    #[test]
    fn test_serialize_conversation_user_blocks_join_without_separator() {
        // pi: contentText(msg.content, "") — no separator between blocks
        let messages = vec![user_message(ContentBlockList::Blocks(vec![
            text_block("he"),
            text_block("llo"),
        ]))];
        assert_eq!(serialize_conversation(&messages), "[User]: hello");
    }

    #[test]
    fn test_serialize_conversation_empty_user_content_skipped() {
        let messages = vec![user_message(ContentBlockList::Text(String::new()))];
        assert_eq!(serialize_conversation(&messages), "");
    }

    #[test]
    fn test_serialize_conversation_assistant_lines_in_order() {
        // Order: thinking → text → tool calls (each a separate part)
        let messages = vec![assistant_message(vec![
            thinking_block("let me check"),
            text_block("I will read the file"),
            tool_call("read", json!({"path": "a.ts"})),
        ])];
        assert_eq!(
            serialize_conversation(&messages),
            "[Assistant thinking]: let me check\n\n\
             [Assistant]: I will read the file\n\n\
             [Assistant tool calls]: read(path=\"a.ts\")"
        );
    }

    #[test]
    fn test_serialize_conversation_multiple_tool_calls_joined_by_semicolon() {
        let messages = vec![assistant_message(vec![
            tool_call("read", json!({"path": "a.ts"})),
            tool_call("edit", json!({"path": "b.ts"})),
        ])];
        assert_eq!(
            serialize_conversation(&messages),
            "[Assistant tool calls]: read(path=\"a.ts\"); edit(path=\"b.ts\")"
        );
    }

    #[test]
    fn test_serialize_conversation_assistant_without_text_block_has_no_assistant_line() {
        // No text block means no [Assistant]: line is emitted (pi: content.some(type === "text"))
        let messages = vec![assistant_message(vec![tool_call("read", json!({}))])];
        let serialized = serialize_conversation(&messages);
        assert!(!serialized.contains("[Assistant]:"));
        assert!(serialized.contains("[Assistant tool calls]: read()"));
    }

    #[test]
    fn test_serialize_conversation_tool_result_truncated_at_2000_chars() {
        let messages = vec![tool_result(&"x".repeat(2_100))];
        let serialized = serialize_conversation(&messages);
        assert!(serialized.starts_with("[Tool result]: "));
        assert!(serialized.contains("\n\n[... 100 more characters truncated]"));
    }

    #[test]
    fn test_serialize_conversation_short_tool_result_untouched() {
        let messages = vec![tool_result("done")];
        assert_eq!(serialize_conversation(&messages), "[Tool result]: done");
    }

    #[test]
    fn test_serialize_conversation_empty_tool_result_skipped() {
        let messages = vec![tool_result("")];
        assert_eq!(serialize_conversation(&messages), "");
    }

    #[test]
    fn test_serialize_conversation_joins_parts_with_blank_line() {
        let messages = vec![
            user_message(ContentBlockList::Text("a".into())),
            tool_result("b"),
        ];
        assert_eq!(
            serialize_conversation(&messages),
            "[User]: a\n\n[Tool result]: b"
        );
    }
}
