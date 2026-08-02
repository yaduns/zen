use crate::agent::types::{
    AgentMessage, BashExecutionMessage, BranchSummaryMessage, CompactionSummaryMessage,
    CustomAgentMessage,
};
use zen_ai::types::{ContentBlock, ContentBlockList, Message, TextContent, UserMessage};

/// Prefix text wrapping the compaction summary, used for formatting in convertToLlm.
/// Corresponds to `COMPACTION_SUMMARY_PREFIX` in TS.
pub const COMPACTION_SUMMARY_PREFIX: &str = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";

/// Suffix text that wraps the compaction summary.
/// Corresponds to TS `COMPACTION_SUMMARY_SUFFIX`.
pub const COMPACTION_SUMMARY_SUFFIX: &str = "\n</summary>";

/// Prefix text that wraps the branch summary.
/// Corresponds to TS `BRANCH_SUMMARY_PREFIX`.
pub const BRANCH_SUMMARY_PREFIX: &str =
    "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n";

/// Suffix text that wraps the branch summary.
/// Corresponds to TS `BRANCH_SUMMARY_SUFFIX`.
pub const BRANCH_SUMMARY_SUFFIX: &str = "\n</summary>";

/// Converts agent transcript messages to an LLM-compatible message format.
///
/// Conversion rules (inherits original TS semantics):
/// - `AgentMessage::BashExecution`: if `exclude_from_context` is `true`, skip it (do not include in the returned array);
///   otherwise, format it as plain text via `bash_execution_to_text` and return it as a `UserMessage`.
/// - `AgentMessage::Custom`: if `content` is plain text, wrap it as `ContentBlock::Text`; otherwise use it directly.
///   Returned as a `UserMessage`.
/// - `AgentMessage::BranchSummary`: wrap `summary` with `BRANCH_SUMMARY_PREFIX`/`SUFFIX`,
///   Returned as a `UserMessage`.
/// - `AgentMessage::CompactionSummary`: wrap `summary` with `COMPACTION_SUMMARY_PREFIX`/`SUFFIX`,
///   Returned as a `UserMessage`.
/// - `AgentMessage::Llm(Message::User)` / `Message::Assistant` / `Message::ToolResult`: pass through directly.
///
/// - Must not panic. Silently skip unknown variants or meaningless messages.
/// - The returned `Message` array preserves the original order, only skipping excluded messages.
/// - Does not modify the original `messages` slice.
///
/// # Parameters
/// - `messages`: slice of agent transcript messages
///
/// # Return Value
/// Converted LLM message array, with messages marked `excludeFromContext` filtered out.
pub fn convert_to_llm(messages: &[AgentMessage]) -> Vec<Message> {
    messages
        .iter()
        .filter_map(|msg| match msg {
            AgentMessage::BashExecution(bash) => {
                if bash.exclude_from_context {
                    None
                } else {
                    Some(Message::User(UserMessage {
                        content: ContentBlockList::Blocks(vec![ContentBlock::Text(TextContent {
                            text: bash_execution_to_text(bash),
                            text_signature: None,
                        })]),
                        timestamp: bash.timestamp,
                    }))
                }
            }
            AgentMessage::Custom(custom) => {
                let content = match &custom.content {
                    ContentBlockList::Text(_) => custom.content.clone(),
                    ContentBlockList::Blocks(blocks) => ContentBlockList::Blocks(blocks.clone()),
                };
                Some(Message::User(UserMessage {
                    content,
                    timestamp: custom.timestamp,
                }))
            }
            AgentMessage::BranchSummary(summary) => Some(Message::User(UserMessage {
                content: ContentBlockList::Blocks(vec![ContentBlock::Text(TextContent {
                    text: format!(
                        "{}{}{}",
                        BRANCH_SUMMARY_PREFIX, summary.summary, BRANCH_SUMMARY_SUFFIX
                    ),
                    text_signature: None,
                })]),
                timestamp: summary.timestamp,
            })),
            AgentMessage::CompactionSummary(summary) => Some(Message::User(UserMessage {
                content: ContentBlockList::Blocks(vec![ContentBlock::Text(TextContent {
                    text: format!(
                        "{}{}{}",
                        COMPACTION_SUMMARY_PREFIX, summary.summary, COMPACTION_SUMMARY_SUFFIX
                    ),
                    text_signature: None,
                })]),
                timestamp: summary.timestamp,
            })),
            AgentMessage::Llm(llm_msg) => Some(llm_msg.clone()),
        })
        .collect()
}

/// Formats a `BashExecutionMessage` as an LLM-readable plain-text description.
///
/// ```text
/// Ran `{command}`
/// ```
/// {output}
/// ```
/// ```
/// If the command was cancelled, append "\n\n(command cancelled)".
/// If the exit code is non-zero and not `None`, append "\n\nCommand exited with code {exit_code}".
/// If the output was truncated and `full_output_path` exists, append "\n\n[Output truncated. Full output: {full_output_path}]".
/// If the output is an empty string, replace it with "(no output)".
///
/// # Parameters
/// - `msg`: bash execution message
///
/// # Return Value
/// Formatted text string.
pub fn bash_execution_to_text(msg: &BashExecutionMessage) -> String {
    let mut text = format!("Ran `{}`\n", msg.command);
    if msg.output.is_empty() {
        text.push_str("(no output)");
    } else {
        text.push_str(&format!("```\n{}\n```", msg.output));
    }
    if msg.cancelled {
        text.push_str("\n\n(command cancelled)");
    } else if let Some(exit_code) = msg.exit_code {
        if exit_code != 0 {
            text.push_str(&format!("\n\nCommand exited with code {}", exit_code));
        }
    }
    if msg.truncated {
        if let Some(ref path) = msg.full_output_path {
            text.push_str(&format!("\n\n[Output truncated. Full output: {}]", path));
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use zen_ai::types::ContentBlock;

    // ---- bash_execution_to_text ----

    #[test]
    fn test_bash_execution_to_text_with_output() {
        let msg = BashExecutionMessage {
            command: "ls -la".into(),
            output: "file1.txt\nfile2.txt".into(),
            exit_code: Some(0),
            cancelled: false,
            truncated: false,
            full_output_path: None,
            timestamp: 1000,
            exclude_from_context: false,
        };
        let result = bash_execution_to_text(&msg);
        assert_eq!(result, "Ran `ls -la`\n```\nfile1.txt\nfile2.txt\n```");
    }

    #[test]
    fn test_bash_execution_to_text_empty_output() {
        let msg = BashExecutionMessage {
            command: "echo".into(),
            output: "".into(),
            exit_code: Some(0),
            cancelled: false,
            truncated: false,
            full_output_path: None,
            timestamp: 1000,
            exclude_from_context: false,
        };
        let result = bash_execution_to_text(&msg);
        assert_eq!(result, "Ran `echo`\n(no output)");
    }

    #[test]
    fn test_bash_execution_to_text_cancelled() {
        let msg = BashExecutionMessage {
            command: "sleep 10".into(),
            output: "".into(),
            exit_code: None,
            cancelled: true,
            truncated: false,
            full_output_path: None,
            timestamp: 1000,
            exclude_from_context: false,
        };
        let result = bash_execution_to_text(&msg);
        assert!(result.contains("(command cancelled)"));
        // There should be no exit code info when cancelled
        assert!(!result.contains("exit code"));
    }

    #[test]
    fn test_bash_execution_to_text_nonzero_exit() {
        let msg = BashExecutionMessage {
            command: "invalid-command".into(),
            output: "error: not found".into(),
            exit_code: Some(127),
            cancelled: false,
            truncated: false,
            full_output_path: None,
            timestamp: 1000,
            exclude_from_context: false,
        };
        let result = bash_execution_to_text(&msg);
        assert!(result.contains("Command exited with code 127"));
    }

    #[test]
    fn test_bash_execution_to_text_zero_exit_omitted() {
        // Exit code info should not be appended when the exit code is 0
        let msg = BashExecutionMessage {
            command: "true".into(),
            output: "".into(),
            exit_code: Some(0),
            cancelled: false,
            truncated: false,
            full_output_path: None,
            timestamp: 1000,
            exclude_from_context: false,
        };
        let result = bash_execution_to_text(&msg);
        assert!(!result.contains("exit code"));
    }

    #[test]
    fn test_bash_execution_to_text_truncated_with_path() {
        let msg = BashExecutionMessage {
            command: "cat large.log".into(),
            output: "last 10 lines...".into(),
            exit_code: Some(0),
            cancelled: false,
            truncated: true,
            full_output_path: Some("/tmp/zen-xxx.log".into()),
            timestamp: 1000,
            exclude_from_context: false,
        };
        let result = bash_execution_to_text(&msg);
        assert!(result.contains("[Output truncated. Full output: /tmp/zen-xxx.log]"));
    }

    #[test]
    fn test_bash_execution_to_text_truncated_without_path() {
        // Path info should not be appended when truncated is true but full_output_path is None
        let msg = BashExecutionMessage {
            command: "cat large.log".into(),
            output: "last 10 lines...".into(),
            exit_code: Some(0),
            cancelled: false,
            truncated: true,
            full_output_path: None,
            timestamp: 1000,
            exclude_from_context: false,
        };
        let result = bash_execution_to_text(&msg);
        assert!(!result.contains("Output truncated"));
    }

    #[test]
    fn test_bash_execution_to_text_all_conditions() {
        // Timeout cancellation + non-zero exit code + truncation, multiple conditions satisfied simultaneously
        let msg = BashExecutionMessage {
            command: "slow-script".into(),
            output: "partial output".into(),
            exit_code: Some(1),
            cancelled: true,
            truncated: true,
            full_output_path: Some("/tmp/out.log".into()),
            timestamp: 1000,
            exclude_from_context: false,
        };
        let result = bash_execution_to_text(&msg);
        // Ignore exit_code when cancelled; do not output the exit code
        assert!(result.contains("(command cancelled)"));
        assert!(!result.contains("exit code"));
        assert!(result.contains("[Output truncated. Full output: /tmp/out.log]"));
    }

    // ---- convert_to_llm ----

    #[test]
    fn test_convert_to_llm_empty() {
        let result = convert_to_llm(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_convert_to_llm_bash_excluded() {
        let msgs = [AgentMessage::BashExecution(BashExecutionMessage {
            command: "ls".into(),
            output: "file.txt".into(),
            exit_code: Some(0),
            cancelled: false,
            truncated: false,
            full_output_path: None,
            timestamp: 1000,
            exclude_from_context: true,
        })];
        let result = convert_to_llm(&msgs);
        assert!(result.is_empty());
    }

    #[test]
    fn test_convert_to_llm_bash_included() {
        let msgs = [AgentMessage::BashExecution(BashExecutionMessage {
            command: "ls".into(),
            output: "file.txt".into(),
            exit_code: Some(0),
            cancelled: false,
            truncated: false,
            full_output_path: None,
            timestamp: 1000,
            exclude_from_context: false,
        })];
        let result = convert_to_llm(&msgs);
        assert_eq!(result.len(), 1);
        match &result[0] {
            Message::User(u) => {
                let text = match &u.content {
                    ContentBlockList::Text(_) => panic!("expected Blocks"),
                    ContentBlockList::Blocks(blocks) => {
                        assert_eq!(blocks.len(), 1);
                        match &blocks[0] {
                            ContentBlock::Text(t) => t.text.clone(),
                            _ => panic!("expected Text"),
                        }
                    }
                };
                assert!(text.starts_with("Ran `ls`"));
                assert!(text.contains("file.txt"));
            }
            _ => panic!("expected User message"),
        }
    }

    #[test]
    fn test_convert_to_llm_custom_text_content() {
        let msgs = [AgentMessage::Custom(CustomAgentMessage {
            custom_type: "notification".into(),
            content: ContentBlockList::Text("hello world".into()),
            details: None,
            display: true,
            timestamp: 1000,
        })];
        let result = convert_to_llm(&msgs);
        assert_eq!(result.len(), 1);
        match &result[0] {
            Message::User(u) => match &u.content {
                ContentBlockList::Text(t) => assert_eq!(t, "hello world"),
                _ => panic!("expected Text content"),
            },
            _ => panic!("expected User message"),
        }
    }

    #[test]
    fn test_convert_to_llm_custom_blocks_content() {
        let block = ContentBlock::Text(TextContent {
            text: "rich content".into(),
            text_signature: None,
        });
        let msgs = [AgentMessage::Custom(CustomAgentMessage {
            custom_type: "artifact".into(),
            content: ContentBlockList::Blocks(vec![block]),
            details: None,
            display: true,
            timestamp: 1000,
        })];
        let result = convert_to_llm(&msgs);
        assert_eq!(result.len(), 1);
        match &result[0] {
            Message::User(u) => match &u.content {
                ContentBlockList::Blocks(blocks) => {
                    assert_eq!(blocks.len(), 1);
                    match &blocks[0] {
                        ContentBlock::Text(t) => assert_eq!(t.text, "rich content"),
                        _ => panic!("expected Text"),
                    }
                }
                _ => panic!("expected Blocks"),
            },
            _ => panic!("expected User message"),
        }
    }

    #[test]
    fn test_convert_to_llm_branch_summary() {
        let msgs = [AgentMessage::BranchSummary(BranchSummaryMessage {
            summary: "user asked about Rust ownership".into(),
            from_id: "abc-123".into(),
            timestamp: 1000,
        })];
        let result = convert_to_llm(&msgs);
        assert_eq!(result.len(), 1);
        match &result[0] {
            Message::User(u) => match &u.content {
                ContentBlockList::Blocks(blocks) => {
                    assert_eq!(blocks.len(), 1);
                    match &blocks[0] {
                        ContentBlock::Text(t) => {
                            assert!(t.text.starts_with(BRANCH_SUMMARY_PREFIX));
                            assert!(t.text.contains("user asked about Rust ownership"));
                            assert!(t.text.ends_with(BRANCH_SUMMARY_SUFFIX));
                        }
                        _ => panic!("expected Text"),
                    }
                }
                _ => panic!("expected Blocks"),
            },
            _ => panic!("expected User message"),
        }
    }

    #[test]
    fn test_convert_to_llm_compaction_summary() {
        let msgs = [AgentMessage::CompactionSummary(CompactionSummaryMessage {
            summary: "previous discussion about testing".into(),
            tokens_before: 4500,
            timestamp: 1000,
        })];
        let result = convert_to_llm(&msgs);
        assert_eq!(result.len(), 1);
        match &result[0] {
            Message::User(u) => match &u.content {
                ContentBlockList::Blocks(blocks) => {
                    assert_eq!(blocks.len(), 1);
                    match &blocks[0] {
                        ContentBlock::Text(t) => {
                            assert!(t.text.starts_with(COMPACTION_SUMMARY_PREFIX));
                            assert!(t.text.contains("previous discussion about testing"));
                            assert!(t.text.ends_with(COMPACTION_SUMMARY_SUFFIX));
                        }
                        _ => panic!("expected Text"),
                    }
                }
                _ => panic!("expected Blocks"),
            },
            _ => panic!("expected User message"),
        }
    }

    #[test]
    fn test_convert_to_llm_llm_passthrough() {
        let msgs = [AgentMessage::Llm(Message::User(UserMessage {
            content: ContentBlockList::Text("hello".into()),
            timestamp: 1000,
        }))];
        let result = convert_to_llm(&msgs);
        assert_eq!(result.len(), 1);
        match &result[0] {
            Message::User(u) => match &u.content {
                ContentBlockList::Text(t) => assert_eq!(t, "hello"),
                _ => panic!("expected Text"),
            },
            _ => panic!("expected User message"),
        }
    }

    #[test]
    fn test_convert_to_llm_mixed_preserves_order() {
        // Mixed list: excluded bash + pass-through user + retained bash; order should be preserved
        let msgs = vec![
            AgentMessage::BashExecution(BashExecutionMessage {
                command: "secret".into(),
                output: "sensitive data".into(),
                exit_code: Some(0),
                cancelled: false,
                truncated: false,
                full_output_path: None,
                timestamp: 1000,
                exclude_from_context: true,
            }),
            AgentMessage::Llm(Message::User(UserMessage {
                content: ContentBlockList::Text("keep me".into()),
                timestamp: 2000,
            })),
            AgentMessage::BashExecution(BashExecutionMessage {
                command: "ls".into(),
                output: "files".into(),
                exit_code: Some(0),
                cancelled: false,
                truncated: false,
                full_output_path: None,
                timestamp: 3000,
                exclude_from_context: false,
            }),
        ];
        let result = convert_to_llm(&msgs);
        assert_eq!(result.len(), 2);
        // The first should be user "keep me"
        match &result[0] {
            Message::User(u) => match &u.content {
                ContentBlockList::Text(t) => assert_eq!(t, "keep me"),
                _ => panic!("expected Text"),
            },
            _ => panic!("expected User message"),
        }
        // The second should be bash "ls"
        match &result[1] {
            Message::User(u) => match &u.content {
                ContentBlockList::Blocks(blocks) => {
                    assert_eq!(blocks.len(), 1);
                    match &blocks[0] {
                        ContentBlock::Text(t) => assert!(t.text.starts_with("Ran `ls`")),
                        _ => panic!("expected Text"),
                    }
                }
                _ => panic!("expected Blocks"),
            },
            _ => panic!("expected User message"),
        }
    }
}
