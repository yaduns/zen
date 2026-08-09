use crate::session::tree::{SessionContext, SessionModel, SessionTree, SessionTreeEntry};
use std::fmt::Write;
use zen_ai::types::{ContentBlock, ContentBlockList, Message, TextContent, UserMessage};

/// Default context entry transform: handles `Compaction` entries, replacing history messages with a summary.
///
/// Input: pathEntries — ordered list of entries from root to leaf ([entry₀, entry₁, ..., entryₙ])
/// Output: the subset that retains the Compaction summary and drops the compacted old entries
///
/// 1. Find the last Compaction entry; if none → return as-is
/// 2. The result always starts with [compaction, ...all entries after compaction]
/// 3. If compaction.retainedTail exists → stop here and return
/// 4. If compaction.firstKeptEntryId exists → scan pathEntries from the head up to before compaction; once firstKeptEntryId is found, also add the entries between that ID and compaction to the result
pub fn default_context_entry_transform(path_entries: &[SessionTreeEntry]) -> Vec<SessionTreeEntry> {
    // 1. Find the last Compaction entry
    let last_compaction_idx = path_entries
        .iter()
        .rposition(|e| matches!(e, SessionTreeEntry::Compaction { .. }));

    let Some(compaction_idx) = last_compaction_idx else {
        // No Compaction → return as-is
        return path_entries.to_vec();
    };

    let compaction = &path_entries[compaction_idx];

    // 3. If retained_tail exists → return only [compaction, ...entries after it]
    if let SessionTreeEntry::Compaction {
        retained_tail: Some(_),
        ..
    } = compaction
    {
        return path_entries[compaction_idx..].to_vec();
    }

    let mut result = Vec::new();

    // 4. If first_kept_entry_id exists → scan from the head up to before compaction, and once that ID is found add it to the result
    if let SessionTreeEntry::Compaction {
        first_kept_entry_id: Some(ref kept_id),
        ..
    } = compaction
    {
        if let Some(kept_idx) = path_entries[..compaction_idx]
            .iter()
            .position(|e| e.id() == kept_id.as_str())
        {
            result.extend_from_slice(&path_entries[kept_idx..compaction_idx]);
        }
    }

    // 2. Always append [compaction, ...all entries after compaction]
    result.extend_from_slice(&path_entries[compaction_idx..]);
    result
}

/// Build the context entry list: always applies `default_context_entry_transform`, no custom transform accepted.
///
pub fn build_context_entries(path_entries: &[SessionTreeEntry]) -> Vec<SessionTreeEntry> {
    default_context_entry_transform(path_entries)
}

/// Convert a single `SessionTreeEntry` into a set of context messages.
///
/// Corresponds to Pi's `sessionEntryToContextMessages` function.
/// Mapping rules:
/// - `Message` → return `entry.message` directly
/// - `CustomMessage` → create the message via `createCustomMessage`
/// - `Compaction` → create a summary message via `createCompactionSummaryMessage`; if `retained_tail` exists, append it
/// - `BranchSummary` → create a branch summary message via `createBranchSummaryMessage`
/// - `Custom` → return an empty Vec (v1 extension system not implemented; CustomEntry is never produced)
/// - Other types (ThinkingLevelChange/ModelChange/ActiveToolsChange/Label/SessionInfo/Leaf) → return empty
/// entries and index are unused in v1 and retained as reserved parameters; adjacent entries may need to be inspected to determine how to convert a CustomEntry into a message
pub fn session_entry_to_context_messages(
    entry: &SessionTreeEntry,
    index: usize,
    entries: &[SessionTreeEntry],
) -> Vec<Message> {
    match entry {
           SessionTreeEntry::Message { message, .. } => {
               vec![message.clone()]
           }
           SessionTreeEntry::Compaction {
               summary,
               tokens_before,
               retained_tail,
               timestamp,
               ..
           } => {
               let mut msgs = vec![create_compaction_summary_message(
                   summary,
                   *tokens_before,
                   timestamp,
               )];
               if let Some(tail) = retained_tail {
                   msgs.extend(tail.clone());
               }
               msgs
           }
           SessionTreeEntry::BranchSummary {
               summary,
               from_id,
               timestamp,
               ..
           } => {
               vec![create_branch_summary_message(summary, from_id, timestamp)]
           }
           SessionTreeEntry::CustomMessage {
               custom_type,
               content,
               display,
               details,
               timestamp,
               ..
           } => {
               vec![create_custom_message(
                   custom_type,
                   content,
                   *display,
                   details.as_ref(),
                   timestamp,
               )]
           }
           // The Custom → v1 extension is not implemented and produces no message
           SessionTreeEntry::Custom { .. }
           // Other types do not produce context messages
           | SessionTreeEntry::ThinkingLevelChange { .. }
           | SessionTreeEntry::ModelChange { .. }
           | SessionTreeEntry::ActiveToolsChange { .. }
           | SessionTreeEntry::Label { .. }
           | SessionTreeEntry::SessionInfo { .. }
           | SessionTreeEntry::Leaf { .. } => vec![],
       }
}

/// Build the full session context: derive state from path entries (thinking level/model/active tools),
/// apply default context entry transformations, then convert into a message list.
///
pub fn build_session_context(path_entries: &[SessionTreeEntry]) -> SessionContext {
    // 1. Derive state: iterate path_entries and take the most recent value
    let mut thinking_level = String::from("off");
    let mut model: Option<SessionModel> = None;
    let mut active_tool_names: Option<Vec<String>> = None;

    for entry in path_entries {
        match entry {
            SessionTreeEntry::ThinkingLevelChange {
                thinking_level: level,
                ..
            } => {
                thinking_level = level.clone();
            }
            SessionTreeEntry::ModelChange {
                provider, model_id, ..
            } => {
                model = Some(SessionModel {
                    provider: provider.clone(),
                    model_id: model_id.clone(),
                });
            }
            SessionTreeEntry::Message { message, .. } => {
                if let Message::Assistant(assistant_msg) = message {
                    // Assistant messages also carry provider/model; the most recent one overrides
                    model = Some(SessionModel {
                        provider: assistant_msg.provider.clone(),
                        model_id: assistant_msg.model.clone(),
                    });
                }
            }
            SessionTreeEntry::ActiveToolsChange {
                active_tool_names: names,
                ..
            } => {
                active_tool_names = Some(names.clone());
            }
            _ => {}
        }
    }

    // 2. Apply default context entry transformations
    let context_entries = build_context_entries(path_entries);

    // 3. Convert each entry into messages, expanded via flat_map
    let messages: Vec<Message> = context_entries
        .iter()
        .enumerate()
        .flat_map(|(index, entry)| {
            session_entry_to_context_messages(entry, index, &context_entries)
        })
        .collect();

    SessionContext {
        messages,
        thinking_level,
        model,
        active_tool_names,
    }
}

// ─── Message construction helpers ────────────────────────────────────────────────

/// Creates a summary message from a `Compaction` entry (injected into the LLM context as a `UserMessage`).
/// Retains the `summary` text and `tokens_before` information so the LLM can understand the context compaction state.
/// Equivalent to Pi's `createCompactionSummaryMessage`.
pub fn create_compaction_summary_message(
    summary: &str,
    tokens_before: u64,
    timestamp: &str,
) -> Message {
    Message::User(UserMessage {
        content: ContentBlockList::Text(format!(
            "[Context Summarize - Before Compact {} tokens]\n{}",
            tokens_before, summary
        )),
        timestamp: parse_iso_timestamp(timestamp),
    })
}

/// Creates a branch summary message from a `BranchSummary` entry (injected into the LLM context as a `UserMessage`).
/// Retains the `summary` text and `from_id` (the source branch's entry ID) so the LLM can understand the branch origin.
/// Equivalent to Pi's `createBranchSummaryMessage`.
pub fn create_branch_summary_message(summary: &str, from_id: &str, timestamp: &str) -> Message {
    Message::User(UserMessage {
        content: ContentBlockList::Text(format!(
            "[Branch Summarize - from branch entry {}]\n{}",
            from_id, summary
        )),
        timestamp: parse_iso_timestamp(timestamp),
    })
}

/// Create a custom message from a `CustomMessage` entry.
/// Preserve metadata such as `custom_type`, `content`, `display`, and `details`.
/// Corresponds to Pi's `createCustomMessage`.
pub fn create_custom_message(
    custom_type: &str,
    content: &ContentBlockList,
    display: bool,
    details: Option<&serde_json::Value>,
    timestamp: &str,
) -> Message {
    Message::User(UserMessage {
        content: content.clone(),
        timestamp: parse_iso_timestamp(timestamp),
    })
}

/// Parse an ISO 8601 timestamp as Unix milliseconds, falling back to 0 if parsing fails.
fn parse_iso_timestamp(ts: &str) -> i64 {
    chrono::DateTime::parse_from_rfc3339(ts)
        .or_else(|_| {
            chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%dT%H:%M:%S")
                .map(|dt| dt.and_utc().into())
        })
        .map(|dt| dt.timestamp_millis())
        .unwrap_or(0)
}

// ─── buildContext orchestration ────────────────────────────────────────────────

/// Extract message history from the conversation tree and build the LLM context.
///
/// Orchestration steps:
/// 1. Call `tree.get_path_to_root_or_compaction(leaf_id)` to get the path entries
/// 2. Call `build_session_context(&path)` to convert the entries into a `SessionContext`
///
/// The returned `SessionContext` contains:
/// - `messages`: transformed message list (compression summary transformation applied)
/// - `thinking_level` / `model` / `active_tool_names`: session state derived from path entries
///
/// If `leaf_id` is `None`, use `tree.leaf_id`.
/// If `leaf_id` is not in the tree, return an empty context (`messages: []`; other fields use default values).
///
/// # excludeFromContext filtering
///
/// Phase 1 does not filter and returns all messages. Phase 2 is handled by `AgentLoopHooks::filter_messages`
/// (T-AG-01-7) handles filtering in the agent loop, not in this function.
///
/// Corresponds to Pi's `Session.buildContext()` method.
pub fn build_context(tree: &SessionTree, leaf_id: Option<&str>) -> SessionContext {
    let path = tree.get_path_to_root_or_compaction(leaf_id);
    if path.is_empty() {
        return SessionContext {
            messages: vec![],
            thinking_level: String::from("off"),
            model: None,
            active_tool_names: None,
        };
    }
    let owned: Vec<SessionTreeEntry> = path.into_iter().cloned().collect();
    build_session_context(&owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::tree::SessionTree;
    use zen_ai::types::{AssistantMessage, Cost, StopReason, Usage};

    // ─── Helper constructors ────────────────────────────────────────────────────

    fn msg_entry(id: &str, parent: Option<&str>, role: &str) -> SessionTreeEntry {
        let msg = match role {
            "user" => Message::User(UserMessage {
                content: ContentBlockList::Text("hello".into()),
                timestamp: 0,
            }),
            "assistant" => Message::Assistant(AssistantMessage {
                content: vec![],
                api: "test".into(),
                provider: "test".into(),
                model: "test".into(),
                response_model: None,
                response_id: None,
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: 0,
            }),
            _ => unreachable!(),
        };
        SessionTreeEntry::Message {
            id: id.into(),
            parent_id: parent.map(|p| p.into()),
            timestamp: "2024-01-01T00:00:00Z".into(),
            message: msg,
        }
    }

    fn compaction_entry(
        id: &str,
        parent: Option<&str>,
        retained_tail: bool,
        first_kept: Option<&str>,
    ) -> SessionTreeEntry {
        SessionTreeEntry::Compaction {
            id: id.into(),
            parent_id: parent.map(|p| p.into()),
            timestamp: "2024-01-01T00:00:00Z".into(),
            summary: "compacted".into(),
            first_kept_entry_id: first_kept.map(|s| s.into()),
            tokens_before: 100,
            retained_tail: if retained_tail { Some(vec![]) } else { None },
            details: None,
            usage: None,
            from_hook: None,
        }
    }

    fn branch_summary_entry(
        id: &str,
        parent: Option<&str>,
        summary: &str,
        from_id: &str,
    ) -> SessionTreeEntry {
        SessionTreeEntry::BranchSummary {
            id: id.into(),
            parent_id: parent.map(|p| p.into()),
            timestamp: "2024-01-01T00:00:00Z".into(),
            from_id: from_id.into(),
            summary: summary.into(),
            details: None,
            usage: None,
            from_hook: None,
        }
    }

    fn custom_msg_entry(id: &str, parent: Option<&str>, custom_type: &str) -> SessionTreeEntry {
        SessionTreeEntry::CustomMessage {
            id: id.into(),
            parent_id: parent.map(|p| p.into()),
            timestamp: "2024-01-01T00:00:00Z".into(),
            custom_type: custom_type.into(),
            content: ContentBlockList::Text("custom content".into()),
            display: true,
            details: None,
        }
    }

    fn make_tree(entries: Vec<SessionTreeEntry>, leaf_id: Option<&str>) -> SessionTree {
        SessionTree::new(entries, leaf_id.map(|s| s.into()))
    }

    // ─── parse_iso_timestamp ─────────────────────────────────────────

    #[test]
    fn test_parse_iso_timestamp_rfc3339() {
        let ts = parse_iso_timestamp("2024-01-01T00:00:00Z");
        assert_eq!(ts, 1704067200000);
    }

    #[test]
    fn test_parse_iso_timestamp_without_tz() {
        let ts = parse_iso_timestamp("2024-01-01T00:00:00");
        assert_eq!(ts, 1704067200000);
    }

    #[test]
    fn test_parse_iso_timestamp_invalid() {
        let ts = parse_iso_timestamp("not-a-timestamp");
        assert_eq!(ts, 0);
    }

    #[test]
    fn test_parse_iso_timestamp_empty() {
        let ts = parse_iso_timestamp("");
        assert_eq!(ts, 0);
    }

    // ─── create_compaction_summary_message ───────────────────────────

    #[test]
    fn test_create_compaction_summary_message_format() {
        let msg =
            create_compaction_summary_message("summarized content", 500, "2024-01-01T00:00:00Z");
        let text = match &msg {
            Message::User(u) => match &u.content {
                ContentBlockList::Text(t) => t.clone(),
                _ => panic!("expected Text"),
            },
            _ => panic!("expected User"),
        };
        assert!(text.contains("Before Compact 500 tokens"));
        assert!(text.contains("summarized content"));
    }

    #[test]
    fn test_create_compaction_summary_message_timestamp() {
        let msg = create_compaction_summary_message("x", 0, "2024-01-01T00:00:00Z");
        match &msg {
            Message::User(u) => assert_eq!(u.timestamp, 1704067200000),
            _ => panic!("expected User"),
        }
    }

    // ─── create_branch_summary_message ───────────────────────────────

    #[test]
    fn test_create_branch_summary_message_format() {
        let msg =
            create_branch_summary_message("branch summary", "entry-abc", "2024-01-01T00:00:00Z");
        let text = match &msg {
            Message::User(u) => match &u.content {
                ContentBlockList::Text(t) => t.clone(),
                _ => panic!("expected Text"),
            },
            _ => panic!("expected User"),
        };
        assert!(text.contains("from branch entry entry-abc"));
        assert!(text.contains("branch summary"));
    }

    // ─── session_entry_to_context_messages ───────────────────────────

    #[test]
    fn test_session_entry_to_messages_user_message() {
        let entry = msg_entry("e1", None, "user");
        let msgs = session_entry_to_context_messages(&entry, 0, &[]);
        assert_eq!(msgs.len(), 1);
        assert!(matches!(msgs[0], Message::User(_)));
    }

    #[test]
    fn test_session_entry_to_messages_assistant_message() {
        let entry = msg_entry("e1", None, "assistant");
        let msgs = session_entry_to_context_messages(&entry, 0, &[]);
        assert_eq!(msgs.len(), 1);
        assert!(matches!(msgs[0], Message::Assistant(_)));
    }

    #[test]
    fn test_session_entry_to_messages_compaction() {
        let entry = compaction_entry("c1", None, false, None);
        let msgs = session_entry_to_context_messages(&entry, 0, &[]);
        assert_eq!(msgs.len(), 1);
        let text = match &msgs[0] {
            Message::User(u) => match &u.content {
                ContentBlockList::Text(t) => t.clone(),
                _ => panic!("expected Text"),
            },
            _ => panic!("expected User"),
        };
        assert!(text.contains("Before Compact 100 tokens"));
    }

    #[test]
    fn test_session_entry_to_messages_compaction_with_retained_tail() {
        let retained = vec![Message::User(UserMessage {
            content: ContentBlockList::Text("tail msg".into()),
            timestamp: 0,
        })];
        let entry = SessionTreeEntry::Compaction {
            id: "c1".into(),
            parent_id: None,
            timestamp: "2024-01-01T00:00:00Z".into(),
            summary: "sum".into(),
            first_kept_entry_id: None,
            tokens_before: 100,
            retained_tail: Some(retained),
            details: None,
            usage: None,
            from_hook: None,
        };
        let msgs = session_entry_to_context_messages(&entry, 0, &[]);
        assert_eq!(msgs.len(), 2); // summary + retained tail
    }

    #[test]
    fn test_session_entry_to_messages_branch_summary() {
        let entry = branch_summary_entry("b1", None, "branch summary", "old-entry");
        let msgs = session_entry_to_context_messages(&entry, 0, &[]);
        assert_eq!(msgs.len(), 1);
        let text = match &msgs[0] {
            Message::User(u) => match &u.content {
                ContentBlockList::Text(t) => t.clone(),
                _ => panic!("expected Text"),
            },
            _ => panic!("expected User"),
        };
        assert!(text.contains("from branch entry old-entry"));
    }

    #[test]
    fn test_session_entry_to_messages_custom_message() {
        let entry = custom_msg_entry("cm1", None, "skill_invocation");
        let msgs = session_entry_to_context_messages(&entry, 0, &[]);
        assert_eq!(msgs.len(), 1);
        assert!(matches!(msgs[0], Message::User(_)));
    }

    #[test]
    fn test_session_entry_to_messages_custom_entry_returns_empty() {
        let entry = SessionTreeEntry::Custom {
            id: "x".into(),
            parent_id: None,
            timestamp: "".into(),
            custom_type: "test".into(),
            data: None,
        };
        let msgs = session_entry_to_context_messages(&entry, 0, &[]);
        assert!(msgs.is_empty());
    }

    #[test]
    fn test_session_entry_to_messages_non_message_returns_empty() {
        let variants: Vec<SessionTreeEntry> = vec![
            SessionTreeEntry::ThinkingLevelChange {
                id: "t1".into(),
                parent_id: None,
                timestamp: "".into(),
                thinking_level: "off".into(),
            },
            SessionTreeEntry::ModelChange {
                id: "m1".into(),
                parent_id: None,
                timestamp: "".into(),
                provider: "dp".into(),
                model_id: "m".into(),
            },
            SessionTreeEntry::ActiveToolsChange {
                id: "a1".into(),
                parent_id: None,
                timestamp: "".into(),
                active_tool_names: vec![],
            },
            SessionTreeEntry::Label {
                id: "l1".into(),
                parent_id: None,
                timestamp: "".into(),
                target_id: "x".into(),
                label: None,
            },
            SessionTreeEntry::SessionInfo {
                id: "s1".into(),
                parent_id: None,
                timestamp: "".into(),
                name: None,
            },
            SessionTreeEntry::Leaf {
                id: "f1".into(),
                parent_id: None,
                timestamp: "".into(),
                target_id: None,
            },
        ];
        for entry in &variants {
            let msgs = session_entry_to_context_messages(entry, 0, &[]);
            assert!(msgs.is_empty(), "expected empty for {:?}", entry.id());
        }
    }

    // ─── default_context_entry_transform ─────────────────────────────

    #[test]
    fn test_default_transform_no_compaction() {
        let entries = vec![
            msg_entry("e1", None, "user"),
            msg_entry("e2", Some("e1"), "user"),
        ];
        let result = default_context_entry_transform(&entries);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_default_transform_with_compaction_retained_tail() {
        let entries = vec![
            msg_entry("old", None, "user"),
            compaction_entry("c1", Some("old"), true, None),
            msg_entry("new", Some("c1"), "user"),
        ];
        let result = default_context_entry_transform(&entries);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id(), "c1");
        assert_eq!(result[1].id(), "new");
    }

    #[test]
    fn test_default_transform_with_first_kept() {
        let entries = vec![
            msg_entry("a", None, "user"),
            msg_entry("b", Some("a"), "user"),
            compaction_entry("c1", Some("b"), false, Some("b")),
            msg_entry("d", Some("c1"), "user"),
        ];
        let result = default_context_entry_transform(&entries);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].id(), "b");
        assert_eq!(result[1].id(), "c1");
        assert_eq!(result[2].id(), "d");
    }

    #[test]
    fn test_default_transform_compaction_no_tail_no_kept() {
        let entries = vec![
            msg_entry("old", None, "user"),
            compaction_entry("c1", Some("old"), false, None),
            msg_entry("new", Some("c1"), "user"),
        ];
        let result = default_context_entry_transform(&entries);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id(), "c1");
        assert_eq!(result[1].id(), "new");
    }

    #[test]
    fn test_default_transform_empty_input() {
        let result = default_context_entry_transform(&[]);
        assert!(result.is_empty());
    }

    // ─── build_session_context ───────────────────────────────────────

    #[test]
    fn test_build_session_context_empty() {
        let ctx = build_session_context(&[]);
        assert!(ctx.messages.is_empty());
        assert_eq!(ctx.thinking_level, "off");
        assert!(ctx.model.is_none());
        assert!(ctx.active_tool_names.is_none());
    }

    #[test]
    fn test_build_session_context_derives_thinking_level() {
        let entries = vec![SessionTreeEntry::ThinkingLevelChange {
            id: "t1".into(),
            parent_id: None,
            timestamp: "".into(),
            thinking_level: "high".into(),
        }];
        let ctx = build_session_context(&entries);
        assert_eq!(ctx.thinking_level, "high");
    }

    #[test]
    fn test_build_session_context_derives_model() {
        let entries = vec![SessionTreeEntry::ModelChange {
            id: "m1".into(),
            parent_id: None,
            timestamp: "".into(),
            provider: "deepseek".into(),
            model_id: "v4-flash".into(),
        }];
        let ctx = build_session_context(&entries);
        assert!(ctx.model.is_some());
        assert_eq!(ctx.model.as_ref().unwrap().provider, "deepseek");
        assert_eq!(ctx.model.as_ref().unwrap().model_id, "v4-flash");
    }

    #[test]
    fn test_build_session_context_derives_model_from_assistant_message() {
        let usage = Usage::default();
        let assistant_entry = SessionTreeEntry::Message {
            id: "a1".into(),
            parent_id: None,
            timestamp: "".into(),
            message: Message::Assistant(AssistantMessage {
                content: vec![],
                api: "test".into(),
                provider: "openai".into(),
                model: "gpt-4".into(),
                response_model: None,
                response_id: None,
                usage,
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: 0,
            }),
        };
        let ctx = build_session_context(&[assistant_entry]);
        assert!(ctx.model.is_some());
        assert_eq!(ctx.model.as_ref().unwrap().provider, "openai");
        assert_eq!(ctx.model.as_ref().unwrap().model_id, "gpt-4");
    }

    #[test]
    fn test_build_session_context_derives_active_tools() {
        let entries = vec![SessionTreeEntry::ActiveToolsChange {
            id: "at1".into(),
            parent_id: None,
            timestamp: "".into(),
            active_tool_names: vec!["bash".into(), "read".into()],
        }];
        let ctx = build_session_context(&entries);
        assert!(ctx.active_tool_names.is_some());
        assert_eq!(ctx.active_tool_names.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_build_session_context_messages() {
        let entries = vec![
            msg_entry("u1", None, "user"),
            msg_entry("a1", Some("u1"), "assistant"),
        ];
        let ctx = build_session_context(&entries);
        assert_eq!(ctx.messages.len(), 2);
    }

    #[test]
    fn test_build_session_context_applies_compaction_transform() {
        let entries = vec![
            msg_entry("old", None, "user"),
            compaction_entry("c1", Some("old"), false, None),
            msg_entry("new", Some("c1"), "user"),
        ];
        let ctx = build_session_context(&entries);
        assert_eq!(ctx.messages.len(), 2); // compaction summary + new
    }

    // ─── build_context ───────────────────────────────────────────────

    #[test]
    fn test_build_context_empty_tree() {
        let tree = make_tree(vec![], None);
        let ctx = build_context(&tree, None);
        assert!(ctx.messages.is_empty());
    }

    #[test]
    fn test_build_context_with_leaf_id() {
        let entries = vec![
            msg_entry("root", None, "user"),
            msg_entry("leaf", Some("root"), "user"),
        ];
        let tree = make_tree(entries, Some("leaf"));
        let ctx = build_context(&tree, None);
        assert_eq!(ctx.messages.len(), 2);
    }

    #[test]
    fn test_build_context_explicit_leaf_id() {
        let entries = vec![
            msg_entry("root", None, "user"),
            msg_entry("mid", Some("root"), "user"),
            msg_entry("leaf", Some("mid"), "user"),
        ];
        let tree = make_tree(entries, Some("leaf"));
        let ctx = build_context(&tree, Some("mid"));
        assert_eq!(ctx.messages.len(), 2);
    }

    #[test]
    fn test_build_context_leaf_id_not_found() {
        let tree = make_tree(vec![msg_entry("e1", None, "user")], Some("e1"));
        let ctx = build_context(&tree, Some("nonexistent"));
        assert!(ctx.messages.is_empty());
    }
}
