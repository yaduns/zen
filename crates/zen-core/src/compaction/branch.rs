use crate::compaction::summarize::{collect_stream_response, summarization_context};
use crate::compaction::utils::{extract_file_ops_from_message, serialize_conversation};
use crate::compaction::{
    compute_file_lists, format_file_operations, seed_file_ops_from_details, CompactionError,
    FileOperations,
};
use crate::session::context::{
    create_branch_summary_message, create_compaction_summary_message, create_custom_message,
};
use crate::session::tree::{SessionTree, SessionTreeEntry};
use std::collections::HashSet;
use tokio_util::sync::CancellationToken;
use zen_ai::api::openai_stream::StreamConfig;
use zen_ai::types::{Message, Model, StopReason, StreamFn, Usage};
use zen_ai::utils::estimate::estimate_message_tokens;
use zen_ai::utils::retry::retry_assistant_call;
use zen_ai::utils::retry::RetryPolicy;
use zen_ai::utils::text::content_text;

/// Return carrier of collect, describing "what to summarize when leaving the current branch"—abandoned branch segment entries + divergence point with the target path.
/// Caller can use common_ancestor_id to know whether the two branches intersect (None = no common history)
#[derive(Debug)]
pub struct CollectEntriesResult {
    pub entries: Vec<SessionTreeEntry>, // Cloned abandoned-branch-segment entries, chronological order (old→new; collected along parent chain leaf→root then reverse
    //    1. You: help me write a login feature
    //    2. AI: Done, used SMS verification code
    //    ├─ 3. You: SMS is too slow, optimize it
    //    │     └─ 4. AI: switched to plan B            ← you are here now (old leaf)
    //    └─ 5. You: maybe switch to QR-code login               ← you want to jump here (target)
    // 2 is common_ancestor_id; 3 and 4 will be summarized into one sentence and inserted between 2 and 5
    pub common_ancestor_id: Option<String>, // ID of the nearest common ancestor entry between the old-leaf path and target path; None = no intersection (including early-return path where old leaf is None)
}

/// Return carrier of generate, complete output of one branch-summary call
#[derive(Debug)]
pub struct BranchSummaryResult {
    pub summary: String, // BRANCH_SUMMARY_PREAMBLE + LLM summary text + format_file_operations appendix; on empty-message short circuit it is "No content to summarize" (then the latter two lists are empty and usage is None)
    pub usage: Option<Usage>, // Usage of the summary LLM call; None on short-circuit path, always Some on normal path (zero value = provider did not report, following the CompactionResult.usage precedent)
    pub read_files: Vec<String>, // Sorted read-only list (produced by compute_file_lists)
    pub modified_files: Vec<String>, // Sorted modified list (produced by compute_file_lists)
}

/// Return carrier of prepare—an intermediate bundle of "ingredients ready but not fired yet", consumed only by generate and unit tests
pub(crate) struct BranchPreparation {
    pub(crate) messages: Vec<Message>,
    pub(crate) file_ops: FileOperations,
    // Metrics slot on pi BranchPreparation interface: generate does not consume it (pi also does not destructure),
    // Used by unit tests and downstream metrics—under lib build it is only constructed, not read; explicitly exempt from dead_code
    #[allow(dead_code)]
    pub(crate) total_tokens: usize,
}

// Owning file: crates/zen-core/src/compaction/branch.rs

/// pi BRANCH_SUMMARY_PREAMBLE text copied verbatim (AGENTS.md rule 10: prompt text may stay in original form):
/// The "exploration preamble" prepended to the head of the LLM summary text; module-private, consumed only by generate_branch_summary.
const BRANCH_SUMMARY_PREAMBLE: &str =  "The user explored a different conversation branch before returning here.\nSummary of that exploration:\n\n";

/// pi BRANCH_SUMMARY_PROMPT copied verbatim: structured-format instructions for branch summaries
/// (## Goal / ## Constraints & Preferences / ## Progress / ## Key Decisions / ## Next Steps), module-private.
const BRANCH_SUMMARY_PROMPT: &str = r#"Create a structured summary of this conversation branch for context when returning later.

   Use this EXACT format:

   ## Goal
   [What was the user trying to accomplish in this branch?]

   ## Constraints & Preferences
   - [Any constraints, preferences, or requirements mentioned]
   - [Or "(none)" if none were mentioned]

   ## Progress
   ### Done
   - [x] [Completed tasks/changes]

   ### In Progress
   - [ ] [Work that was started but not finished]

   ### Blocked
   - [Issues preventing progress, if any]

   ## Key Decisions
   - **[Decision]**: [Brief rationale]

   ## Next Steps
   1. [What should happen next to continue this work]

   Keep each section concise. Preserve exact file paths, function names, and error messages."#;

/// The first question when tree navigation triggers a branch summary: "which entries must be summarized away when leaving the current branch?"
/// Called by the tree-navigation orchestrator (the /tree flow of T-AP-03-2) before moving the leaf pointer to `target_id`:
/// Collect upward from the old leaf along the parent chain until reaching (exclusive) the nearest common ancestor with the target path,
/// yielding the abandoned branch segment (chronological order) and the fork point, for later consumption by `generate_branch_summary`.
/// Edge case: `old_leaf_id` is None (empty session / no leaf) → always returns an empty result and does not touch the tree (mirrors pi's `if (!oldLeafId)` early return;
/// note that None must not be passed through to `SessionTree::get_path_to_root` — the latter's None means "fall back to the current leaf", which differs from "no old leaf" here).
/// Existence of `target_id` is validated by the caller via `get_entry` first (same shape as pi's navigateTree pre-validation);
/// if missing, this function defensively returns "full-path collection + common_ancestor as None".
/// No error surface: pi throws `SessionError("invalid_session")` when `getEntry` misses midway, whereas zen's in-memory tree's
/// "parent node always exists" invariant makes that unreachable (`get_path_to_root` silently truncates on broken links), so the signature omits Result.
/// Consumers: none yet; reserved for T-AP-03-2 (/tree navigation orchestration); unit tests construct a `SessionTree` directly for coverage.
pub fn collect_entries_for_branch_summary(
    tree: &SessionTree,
    old_leaf_id: Option<&str>,
    target_id: &str,
) -> CollectEntriesResult {
    // None must not be passed through to get_path_to_root — its None means "use the current leaf_id"
    let Some(old_leaf_id) = old_leaf_id else {
        return CollectEntriesResult {
            entries: Vec::new(),
            common_ancestor_id: None,
        };
    };

    // Id set of the old path (root→old leaf) + target path (root→target), passing Some explicitly
    let old_path = tree.get_path_to_root(Some(old_leaf_id));
    let old_path_ids: HashSet<&str> = old_path.iter().map(|entry| entry.id()).collect();
    let target_path = tree.get_path_to_root(Some(target_id));

    // Scan the target path from leaf toward root; the first entry that hits the old path = fork point (deepest common ancestor)
    let common_ancestor_id = target_path
        .iter()
        .rev()
        .find(|entry| old_path_ids.contains(entry.id()))
        .map(|entry| entry.id().to_string());

    // Abandoned segment = the suffix of the old path after the fork point (fork point excluded).
    // No fork point (target-missing defense / empty intersection) → the entire old path,
    // entry-for-entry equivalent (including order) to pi's while loop collecting all the way to the root.
    let fork_index = common_ancestor_id
        .as_deref()
        .and_then(|id| old_path.iter().position(|entry| entry.id() == id));
    let start = fork_index.map_or(0, |index| index + 1);
    let entries: Vec<SessionTreeEntry> = old_path[start..]
        .iter()
        .map(|entry| (*entry).clone())
        .collect();

    CollectEntriesResult {
        entries,
        common_ancestor_id,
    }
}

/// The "prep" step of branch summarization: converts abandoned-branch entries into summary input messages, along with file operations and token accounting —
/// "first see how much work this branch did; if over budget, cut from newest to oldest". Called by `generate_branch_summary` before issuing the LLM call;
/// `token_budget` comes from `context_window - reserve_tokens`; 0 means no budget (pi's default-parameter semantics), collecting everything.
/// Behavior:
/// ① Full seeding: readFiles/modifiedFiles of BranchSummary entries (not from_hook output and with details present)
///    are merged into the read/edited buckets respectively (delegated to `seed_file_ops_from_details`, whose JSON key-name contract matches);
/// ② Iterate newest to oldest: convert each entry to a message via `message_from_entry` (skip None), extract file operations
///    (already counted even if truncated by budget afterwards, matching pi's order of extraction before the budget check), estimate tokens
///    (`estimate_messages` single-message form); when `token_budget > 0` and the cumulative total would exceed it:
///    Compaction/BranchSummary entries are still kept and then truncated when the collected amount < budget×0.9 (f64 comparison matches pi); otherwise truncate immediately.
pub(crate) fn prepare_branch_entries(
    entries: &[SessionTreeEntry],
    token_budget: usize,
) -> BranchPreparation {
    // ① Full seeding (independent of budget): BranchSummary entries that are not hook output and have details present,
    //    readFiles→read, modifiedFiles→edited (key-name contract matches the compaction-side seed helper)
    let mut file_ops = FileOperations::new();
    for entry in entries {
        if let SessionTreeEntry::BranchSummary {
            details: Some(details),
            from_hook,
            ..
        } = entry
        {
            if !from_hook.unwrap_or(false) {
                seed_file_ops_from_details(details, &mut file_ops);
            }
        }
    }

    // ② Collect newest to oldest (push in iteration order, reverse at the end = pi's per-item unshift)
    let mut messages: Vec<Message> = Vec::new();
    let mut total_tokens: usize = 0;
    for entry in entries.iter().rev() {
        let Some(message) = message_from_entry(entry) else {
            continue; // Metadata entries are skipped transparently and do not trigger truncation
        };
        // pi: extraction precedes the budget check — even if this entry triggers truncation, its file operations are already counted
        extract_file_ops_from_message(&message, &mut file_ops);

        let tokens = estimate_message_tokens(&message);
        if token_budget > 0 && total_tokens + tokens > token_budget {
            // Summary-type entries below the 90% watermark are still kept before truncating (preserves context-boundary integrity)
            if matches!(
                entry,
                SessionTreeEntry::Compaction { .. } | SessionTreeEntry::BranchSummary { .. }
            ) && (total_tokens as f64) < token_budget as f64 * 0.9
            {
                messages.push(message);
                total_tokens += tokens;
            }
            break;
        }

        messages.push(message);
        total_tokens += tokens;
    }
    messages.reverse();

    BranchPreparation {
        messages,
        file_ops,
        total_tokens,
    }
}

/// The "fire" step of branch summarization: called when the tree-navigation orchestrator decides to summarize an abandoned branch — prep (`prepare_branch_entries`),
/// serialize the conversation, assemble instructions per the replace/append rules, issue an independent LLM call with `SUMMARIZATION_SYSTEM_PROMPT`,
/// producing summary text of "exploration preamble + LLM summary + file-operations appendix"; the caller uses it to construct
/// a `SessionTreeEntry::BranchSummary` written back to the session tree (details serialization contract see `BranchSummaryResult`).
/// Deliberately different from compaction summaries (`generate_summary`) (consistent with pi's original behavior): instructions use BRANCH_SUMMARY_PROMPT,
/// max_tokens is fixed at 2048 (not proportional to reserve, not min'd with model.max_tokens), no `<previous-summary>` iteration path,
/// and `stream_config.reasoning_effort` is left untouched (pi does not set reasoning on this path).
/// Parameters: `stream_config` is the caller's base config (carrying connection credentials such as api_key/headers); this function only overrides `max_tokens`;
/// when `replace_instructions` is true and `custom_instructions` is present, the default instructions are replaced wholesale; otherwise
/// `custom_instructions` is appended as "Additional focus:"; if neither is present, the default instructions are used; `reserve_tokens` is only used for
/// the input budget (`context_window − reserve_tokens`; falls back to 128_000 when `model.context_window` is 0, matching pi's `|| 128000`).
/// Returns/errors: no messages to summarize → `Ok` (summary = "No content to summarize", zero-cost short circuit);
/// cancel/abort → `CompactionError::Aborted`; call failure → `SummarizationFailed` (the first two codes of pi's BranchSummaryError reused with the same shape).
/// Consumers: none yet; reserved for T-AP-03-2 (/tree navigation orchestration); unit tests cover it with a fake StreamFn (following the summarize.rs test precedent).
// Parameter surface mirrors pi generateBranchSummary(entries, options) flattened (9 params); same precedent as compact(); bundling params into a struct to be evaluated when consumers are wired up
#[allow(clippy::too_many_arguments)]
pub async fn generate_branch_summary(
    entries: &[SessionTreeEntry],
    stream_fn: &StreamFn,
    model: &Model,
    stream_config: StreamConfig,
    custom_instructions: Option<&str>,
    replace_instructions: bool,
    reserve_tokens: u64,
    cancel: Option<CancellationToken>,
    retry: Option<RetryPolicy>,
) -> Result<BranchSummaryResult, CompactionError> {
    // Input budget: contextWindow falls back to 128_000 (matching pi `model.contextWindow || 128000`);
    // saturating resolves pi's negative budget — when reserve > window, JS gets a negative number, which prepareBranchEntries'
    // `tokenBudget > 0` guard degrades to "no budget"; zen saturating to 0 is behaviorally equivalent
    let context_window = if model.context_window == 0 {
        128_000
    } else {
        model.context_window
    };
    let token_budget = context_window.saturating_sub(reserve_tokens) as usize;

    // Prep (total_tokens is only for accounting; pi likewise does not destructure it)
    let BranchPreparation {
        messages, file_ops, ..
    } = prepare_branch_entries(entries, token_budget);

    // Empty-message short circuit: zero-cost Ok — usage is None, both lists are empty (same shape as pi)
    if messages.is_empty() {
        return Ok(BranchSummaryResult {
            summary: "No content to summarize".to_string(),
            usage: None,
            read_files: Vec::new(),
            modified_files: Vec::new(),
        });
    }

    // Three-branch instruction assembly (matching pi:199-205): replace requires both to be present;
    // (true, None) falls through to default — pi's nested if behaves the same (replace with nothing to replace = use default)
    let instructions = match (replace_instructions, custom_instructions) {
        (true, Some(custom)) => custom.to_string(),
        (_, Some(custom)) => format!("{BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: {custom}"),
        _ => BRANCH_SUMMARY_PROMPT.to_string(),
    };
    let conversation_text = serialize_conversation(&messages);
    let prompt_text =
        format!("<conversation>\n{conversation_text}\n</conversation>\n\n{instructions}");

    // Request assembly: the system prompt is shared with the compaction path (single-point contract via summarization_context);
    // max_tokens is fixed at 2048 — not scaled by reserve ratio, not min'd with model.max_tokens (pi passes `{ maxTokens: 2048 }` as-is);
    // reasoning_effort is explicitly cleared — on this path pi's options is a fresh object that never carries a reasoning key,
    // zen's rendering of "don't touch" = set to None (inheriting reasoning from the caller's base would deviate from pi)
    let mut config = stream_config;
    config.max_tokens = Some(2048);
    config.reasoning_effort = None;
    let context = summarization_context(&prompt_text);

    let response = retry_assistant_call(
        || {
            collect_stream_response(
                stream_fn,
                model,
                &context,
                Some(config.clone()),
                cancel.clone(),
            )
        },
        retry,
        cancel.clone(),
    )
    .await;

    match response.stop_reason {
        // pi's BranchSummaryError first two codes are reused verbatim as CompactionError
        StopReason::Aborted => Err(CompactionError::Aborted),
        StopReason::Error => Err(CompactionError::SummarizationFailed(format!(
            "Branch summary failed: {}",
            response.error_message.as_deref().unwrap_or("Unknown error")
        ))),
        _ => {
            // pi: summary = PREAMBLE + text; then file-list computation + appendix concatenation
            let mut summary = format!(
                "{BRANCH_SUMMARY_PREAMBLE}{}",
                content_text(&response.content, "\n")
            );
            let (read_files, modified_files) = compute_file_lists(&file_ops);
            summary.push_str(&format_file_operations(&read_files, &modified_files));
            Ok(BranchSummaryResult {
                summary,
                usage: Some(response.usage),
                read_files,
                modified_files,
            })
        }
    }
}

/// Called by prepare_branch_entries while preparing each entry,
/// Converts a single entry into a summary input message (None = this entry is excluded from the summary).
/// Two deliberate differences from `session_entry_to_context_messages` (hence no direct reuse):
/// ① For Message entries, `Message::ToolResult` is filtered out (branch summaries want only conversational semantics, not tool-result noise);
/// ② For Compaction entries, only the summary message is taken; retained_tail is not appended.
/// CustomMessage/BranchSummary match the generic conversion (single-message output, reusing context.rs's
/// `create_custom_message`/`create_branch_summary_message`); other metadata entries → None.
fn message_from_entry(entry: &SessionTreeEntry) -> Option<Message> {
    match entry {
        // ① Message entries: toolResult does not enter the summary (conversational semantics, no tool-result noise),
        //    User/Assistant are kept as-is — a deliberate difference from the compaction path's "keep all Messages"
        SessionTreeEntry::Message { message, .. } => match message {
            Message::ToolResult(_) => None,
            msg @ (Message::User(_) | Message::Assistant(_)) => Some(msg.clone()),
        },
        // ② Compaction entries: only the summary message is taken; retained_tail is not appended
        //    (pi-227 has no buildContext tail semantics — the tail belongs to "current path reconstruction", not to the "discarded segment narrative")
        SessionTreeEntry::Compaction {
            summary,
            tokens_before,
            timestamp,
            ..
        } => Some(create_compaction_summary_message(
            summary,
            *tokens_before,
            timestamp,
        )),
        // ③ BranchSummary entries: same as the generic conversion (single message)
        SessionTreeEntry::BranchSummary {
            summary,
            from_id,
            timestamp,
            ..
        } => Some(create_branch_summary_message(summary, from_id, timestamp)),
        // ④ CustomMessage entries: same as the generic conversion (single message)
        SessionTreeEntry::CustomMessage {
            custom_type,
            content,
            display,
            details,
            timestamp,
            ..
        } => Some(create_custom_message(
            custom_type,
            content,
            *display,
            details.as_ref(),
            timestamp,
        )),
        // Metadata entries produce no message
        SessionTreeEntry::Custom { .. }
        | SessionTreeEntry::ThinkingLevelChange { .. }
        | SessionTreeEntry::ModelChange { .. }
        | SessionTreeEntry::ActiveToolsChange { .. }
        | SessionTreeEntry::Label { .. }
        | SessionTreeEntry::SessionInfo { .. }
        | SessionTreeEntry::Leaf { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::summarize::SUMMARIZATION_SYSTEM_PROMPT;
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use tokio::sync::mpsc;
    use zen_ai::provider::Api;
    use zen_ai::types::{
        AssistantMessage, AssistantMessageEvent, ContentBlock, ContentBlockList,
        NonErrorStopReason, TextContent, ThinkingLevel, ToolResultMessage, UserMessage,
    };

    // ── Construction helpers ─────────────────────────────────────────

    fn ts() -> String {
        "2024-01-01T00:00:00Z".to_string()
    }

    fn user_message(text: &str) -> Message {
        Message::User(UserMessage {
            content: ContentBlockList::Text(text.to_string()),
            timestamp: 0,
        })
    }

    fn user_entry(id: &str, parent: Option<&str>, text: &str) -> SessionTreeEntry {
        SessionTreeEntry::Message {
            id: id.to_string(),
            parent_id: parent.map(str::to_string),
            timestamp: ts(),
            message: user_message(text),
        }
    }

    fn tool_result_entry(id: &str, parent: Option<&str>) -> SessionTreeEntry {
        SessionTreeEntry::Message {
            id: id.to_string(),
            parent_id: parent.map(str::to_string),
            timestamp: ts(),
            message: Message::ToolResult(ToolResultMessage {
                tool_call_id: "call-1".to_string(),
                tool_name: "read".to_string(),
                content: vec![ContentBlock::Text(TextContent {
                    text: "file body".to_string(),
                    text_signature: None,
                })],
                details: None,
                usage: None,
                is_error: false,
                added_tool_names: None,
                timestamp: 0,
            }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compaction_entry(
        id: &str,
        parent: Option<&str>,
        summary: &str,
        retained_tail: Option<Vec<Message>>,
    ) -> SessionTreeEntry {
        SessionTreeEntry::Compaction {
            id: id.to_string(),
            parent_id: parent.map(str::to_string),
            timestamp: ts(),
            summary: summary.to_string(),
            first_kept_entry_id: None,
            tokens_before: 100,
            retained_tail,
            details: None,
            usage: None,
            from_hook: None,
        }
    }

    fn branch_summary_entry(
        id: &str,
        parent: Option<&str>,
        summary: &str,
        details: Option<serde_json::Value>,
        from_hook: Option<bool>,
    ) -> SessionTreeEntry {
        SessionTreeEntry::BranchSummary {
            id: id.to_string(),
            parent_id: parent.map(str::to_string),
            timestamp: ts(),
            from_id: "from-1".to_string(),
            summary: summary.to_string(),
            details,
            usage: None,
            from_hook,
        }
    }

    fn custom_message_entry(id: &str) -> SessionTreeEntry {
        SessionTreeEntry::CustomMessage {
            id: id.to_string(),
            parent_id: None,
            timestamp: ts(),
            custom_type: "skill".to_string(),
            content: ContentBlockList::Text("custom body".to_string()),
            display: true,
            details: None,
        }
    }

    /// A representative metadata entry (the None branch shares an arm with the other metadata variants; any one suffices)
    fn label_entry(id: &str) -> SessionTreeEntry {
        SessionTreeEntry::Label {
            id: id.to_string(),
            parent_id: None,
            timestamp: ts(),
            target_id: "t".to_string(),
            label: Some("x".to_string()),
        }
    }

    /// Forked tree: e1 ← e2 ← e3 ← e4 (old leaf), with e5 hanging under e2 (target)
    fn forked_tree() -> SessionTree {
        SessionTree::new(
            vec![
                user_entry("e1", None, "1"),
                user_entry("e2", Some("e1"), "2"),
                user_entry("e3", Some("e2"), "3"),
                user_entry("e4", Some("e3"), "4"),
                user_entry("e5", Some("e2"), "5"),
            ],
            Some("e4".to_string()),
        )
    }

    /// Two disjoint trees: main chain e1..e4 + independent root t1
    fn disjoint_tree() -> SessionTree {
        SessionTree::new(
            vec![
                user_entry("e1", None, "1"),
                user_entry("e2", Some("e1"), "2"),
                user_entry("e3", Some("e2"), "3"),
                user_entry("e4", Some("e3"), "4"),
                user_entry("t1", None, "t"),
            ],
            Some("e4".to_string()),
        )
    }

    fn user_text(message: &Message) -> String {
        match message {
            Message::User(u) => match &u.content {
                ContentBlockList::Text(s) => s.clone(),
                ContentBlockList::Blocks(_) => String::new(),
            },
            _ => String::new(),
        }
    }

    fn texts(messages: &[Message]) -> Vec<String> {
        messages.iter().map(user_text).collect()
    }

    fn entry_ids(result: &CollectEntriesResult) -> Vec<String> {
        result
            .entries
            .iter()
            .map(|entry| entry.id().to_string())
            .collect()
    }

    // ── collect_entries_for_branch_summary ────────────────

    #[test]
    fn test_collect_entries_none_old_leaf_returns_empty_ignoring_tree_leaf() {
        // The tree itself has a leaf (e4): None must still early-return and must not fall back to tree.leaf_id
        let tree = forked_tree();
        let result = collect_entries_for_branch_summary(&tree, None, "e5");
        assert!(result.entries.is_empty());
        assert_eq!(result.common_ancestor_id, None);
    }

    #[test]
    fn test_collect_entries_extracts_abandoned_segment_in_chronological_order() {
        let tree = forked_tree();
        let result = collect_entries_for_branch_summary(&tree, Some("e4"), "e5");
        // Fork point e2, discarded segment = [e3, e4] (old→new)
        assert_eq!(result.common_ancestor_id, Some("e2".to_string()));
        assert_eq!(entry_ids(&result), vec!["e3".to_string(), "e4".to_string()]);
    }

    #[test]
    fn test_collect_entries_disjoint_paths_collects_full_old_path() {
        let tree = disjoint_tree();
        let result = collect_entries_for_branch_summary(&tree, Some("e4"), "t1");
        assert_eq!(result.common_ancestor_id, None);
        assert_eq!(
            entry_ids(&result),
            vec![
                "e1".to_string(),
                "e2".to_string(),
                "e3".to_string(),
                "e4".to_string()
            ]
        );
    }

    #[test]
    fn test_collect_entries_old_leaf_is_ancestor_of_target_returns_empty() {
        let tree = forked_tree();
        let result = collect_entries_for_branch_summary(&tree, Some("e2"), "e5");
        // The fork point is the old leaf itself → no discarded segment
        assert_eq!(result.common_ancestor_id, Some("e2".to_string()));
        assert!(result.entries.is_empty());
    }

    #[test]
    fn test_collect_entries_missing_target_defensively_collects_full_path() {
        let tree = forked_tree();
        let result = collect_entries_for_branch_summary(&tree, Some("e4"), "nope");
        // Defensive behavior: target missing → no fork point → the entire old path
        assert_eq!(result.common_ancestor_id, None);
        assert_eq!(
            entry_ids(&result),
            vec![
                "e1".to_string(),
                "e2".to_string(),
                "e3".to_string(),
                "e4".to_string()
            ]
        );
    }

    // ── message_from_entry (private, unit-tested directly) ─────────────

    #[test]
    fn test_message_from_entry_filters_tool_result() {
        let entry = tool_result_entry("t1", None);
        assert!(
            message_from_entry(&entry).is_none(),
            "toolResult not included in branch summary"
        );
    }

    #[test]
    fn test_message_from_entry_keeps_user_and_assistant() {
        let user = user_entry("e1", None, "hello");
        assert_eq!(
            message_from_entry(&user).map(|message| user_text(&message)),
            Some("hello".to_string())
        );
        let assistant = SessionTreeEntry::Message {
            id: "e2".to_string(),
            parent_id: None,
            timestamp: ts(),
            message: Message::Assistant(AssistantMessage {
                content: vec![ContentBlock::Text(TextContent {
                    text: "hi".to_string(),
                    text_signature: None,
                })],
                api: "openai-completions".to_string(),
                provider: "deepseek".to_string(),
                model: "test-model".to_string(),
                response_model: None,
                response_id: None,
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: 0,
            }),
        };
        assert!(message_from_entry(&assistant).is_some());
    }

    #[test]
    fn test_message_from_entry_compaction_drops_retained_tail() {
        let entry = compaction_entry(
            "c1",
            None,
            "prev summary",
            Some(vec![user_message("tail msg")]),
        );
        let message = message_from_entry(&entry).expect("Compaction should produce summary message");
        let text = user_text(&message);
        assert!(text.starts_with("[Context Summarize - Before Compact 100 tokens]"));
        // pi-227 deliberate difference: no retained_tail (the tail belongs to current path reconstruction, not to the discarded segment narrative)
        assert!(!text.contains("tail msg"));
    }

    #[test]
    fn test_message_from_entry_branch_summary_format() {
        let entry = branch_summary_entry("b1", None, "exploration", None, None);
        let message = message_from_entry(&entry).expect("BranchSummary should produce message");
        assert_eq!(
            user_text(&message),
            "[Branch Summarize - from branch entry from-1]\nexploration".to_string()
        );
    }

    #[test]
    fn test_message_from_entry_custom_message_and_metadata_entries() {
        let custom = custom_message_entry("m1");
        let message = message_from_entry(&custom).expect("CustomMessage should produce message");
        assert_eq!(user_text(&message), "custom body".to_string());
        // Metadata entries share the arm → None (Label as representative)
        assert!(message_from_entry(&label_entry("m2")).is_none());
    }

    // ── prepare_branch_entries ────────────────────────

    #[test]
    fn test_prepare_branch_entries_collects_all_without_budget() {
        let entries = vec![
            user_entry("e1", None, "u1"),
            label_entry("m1"),
            user_entry("e2", Some("e1"), "u2"),
            tool_result_entry("t1", Some("e2")),
            user_entry("e3", Some("e2"), "u3"),
        ];
        let prep = prepare_branch_entries(&entries, 0);
        // budget = 0 means unlimited; metadata and toolResult are skipped transparently; order is chronological (old→new)
        assert_eq!(
            texts(&prep.messages),
            vec!["u1".to_string(), "u2".to_string(), "u3".to_string()]
        );
        assert!(prep.total_tokens > 0);
    }

    #[test]
    fn test_prepare_branch_entries_compaction_entry_excludes_tail() {
        let entries = vec![compaction_entry(
            "c1",
            None,
            "prev",
            Some(vec![user_message("tail msg")]),
        )];
        let prep = prepare_branch_entries(&entries, 0);
        assert_eq!(prep.messages.len(), 1);
        assert!(user_text(&prep.messages[0]).starts_with("[Context Summarize"));
        assert!(!user_text(&prep.messages[0]).contains("tail msg"));
    }

    #[test]
    fn test_prepare_branch_entries_seeds_file_ops_from_branch_details() {
        let details = json!({
            "readFiles": ["a.ts", "b.md"],
            "modifiedFiles": ["a.ts", "c.rs"]
        });
        let entries = vec![branch_summary_entry(
            "b1",
            None,
            "earlier",
            Some(details),
            None,
        )];
        let prep = prepare_branch_entries(&entries, 0);
        // from_hook defaulting to None is treated as non-hook → seed
        assert_eq!(
            prep.file_ops.read,
            HashSet::from(["a.ts".to_string(), "b.md".to_string()])
        );
        assert_eq!(
            prep.file_ops.edited,
            HashSet::from(["a.ts".to_string(), "c.rs".to_string()])
        );
        assert!(prep.file_ops.written.is_empty());
        // BranchSummary entries also produce a summary message
        assert_eq!(prep.messages.len(), 1);
    }

    #[test]
    fn test_prepare_branch_entries_skips_seeding_when_from_hook() {
        let details = json!({"readFiles": ["a.ts"]});
        let entries = vec![branch_summary_entry(
            "b1",
            None,
            "earlier",
            Some(details),
            Some(true),
        )];
        let prep = prepare_branch_entries(&entries, 0);
        // pi: seed only when !fromHook
        assert!(prep.file_ops.read.is_empty());
        assert!(prep.file_ops.edited.is_empty());
    }

    #[test]
    fn test_prepare_branch_entries_truncates_plain_entry_over_budget() {
        // 250 characters far exceeds 1 token (estimation is by text length; the assumption is documented)
        let long = "word ".repeat(50);
        let entries = vec![user_entry("e1", None, &long)];
        let prep = prepare_branch_entries(&entries, 1);
        // Ordinary entries get no 0.9 fallback → truncated outright, not one accepted
        assert!(prep.messages.is_empty());
        assert_eq!(prep.total_tokens, 0);
    }

    #[test]
    fn test_prepare_branch_entries_keeps_summary_boundary_at_budget_edge() {
        let summary = "summary text ".repeat(30);
        let entries = vec![compaction_entry("c1", None, &summary, None)];
        let prep = prepare_branch_entries(&entries, 1);
        // Compaction entries are still accepted (then truncated) when the collected amount (0) < budget×0.9
        assert_eq!(prep.messages.len(), 1);
        assert!(user_text(&prep.messages[0]).starts_with("[Context Summarize"));
    }

    // ── generate_branch_summary（async，fake StreamFn）───

    fn test_model(context_window: u64) -> Model {
        Model {
            id: "test-model".to_string(),
            name: None,
            provider: "deepseek".to_string(),
            base_url: "https://example.test".to_string(),
            context_window,
            max_tokens: 0,
            reasoning: false,
            headers: None,
            cost_rates: None,
            api: Api::OpenAiCompletions,
        }
    }

    fn assistant_reply(text: &str, stop_reason: StopReason) -> AssistantMessage {
        AssistantMessage {
            content: if text.is_empty() {
                vec![]
            } else {
                vec![ContentBlock::Text(TextContent {
                    text: text.to_string(),
                    text_signature: None,
                })]
            },
            api: "openai-completions".to_string(),
            provider: "deepseek".to_string(),
            model: "test-model".to_string(),
            response_model: None,
            response_id: None,
            usage: Usage::default(),
            stop_reason,
            error_message: None,
            timestamp: 0,
        }
    }

    fn done_event(text: &str) -> AssistantMessageEvent {
        AssistantMessageEvent::Done {
            reason: NonErrorStopReason::Stop,
            message: assistant_reply(text, StopReason::Stop),
        }
    }

    fn base_config() -> StreamConfig {
        StreamConfig {
            api_key: None,
            headers: None,
            tool_choice: None,
            reasoning_effort: None,
            max_tokens: None,
            temperature: None,
            timeout_ms: None,
            session_id: None,
        }
    }

    type Captured = (String, String, Option<StreamConfig>); // (system_prompt, user prompt text, config)

    /// Fake StreamFn: replays a preset event stream and records the (system_prompt, prompt, config) received on each call.
    /// Follows the existing test-double pattern in summarize.rs (StreamFn as an injected closure, event stream + channel).
    fn fake_stream_fn(
        events: Vec<AssistantMessageEvent>,
        captured: Arc<Mutex<Vec<Captured>>>,
    ) -> StreamFn {
        Arc::new(move |_model, context, config, _cancel| {
            let system = context.system_prompt.clone().unwrap_or_default();
            let prompt = match context.messages.first() {
                Some(Message::User(u)) => match &u.content {
                    ContentBlockList::Text(t) => t.clone(),
                    _ => String::new(),
                },
                _ => String::new(),
            };
            captured
                .lock()
                .unwrap()
                .push((system, prompt, config.as_ref().cloned()));
            let (tx, rx) = mpsc::unbounded_channel();
            for event in &events {
                let _ = tx.send(event.clone());
            }
            drop(tx);
            Box::pin(async move { Ok(rx) })
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_generate(
        entries: &[SessionTreeEntry],
        events: Vec<AssistantMessageEvent>,
        custom: Option<&str>,
        replace: bool,
        reserve_tokens: u64,
        context_window: u64,
    ) -> (
        Result<BranchSummaryResult, CompactionError>,
        Arc<Mutex<Vec<Captured>>>,
    ) {
        let captured: Arc<Mutex<Vec<Captured>>> = Arc::new(Mutex::new(Vec::new()));
        let stream_fn = fake_stream_fn(events, captured.clone());
        let result = generate_branch_summary(
            entries,
            &stream_fn,
            &test_model(context_window),
            base_config(),
            custom,
            replace,
            reserve_tokens,
            None,
            None,
        )
        .await;
        (result, captured)
    }

    #[tokio::test]
    async fn test_generate_branch_summary_empty_entries_short_circuits_without_llm_call() {
        let (result, captured) = run_generate(&[], vec![], None, false, 16_384, 100_000).await;
        let r = result.expect("empty input should short-circuit to Ok");
        assert_eq!(r.summary, "No content to summarize");
        assert!(r.usage.is_none());
        assert!(r.read_files.is_empty());
        assert!(r.modified_files.is_empty());
        assert!(
            captured.lock().unwrap().is_empty(),
            "short-circuit path should not initiate LLM call"
        );
    }

    #[tokio::test]
    async fn test_generate_branch_summary_default_prompt_fixed_max_tokens_and_usage() {
        let entries = vec![user_entry("e1", None, "hello")];
        let mut done = assistant_reply("the summary", StopReason::Stop);
        done.usage.input = 7;
        let (result, captured) = run_generate(
            &entries,
            vec![AssistantMessageEvent::Done {
                reason: NonErrorStopReason::Stop,
                message: done,
            }],
            None,
            false,
            16_384,
            100_000,
        )
        .await;
        let r = result.expect("Stop should succeed");
        assert!(r.summary.starts_with(BRANCH_SUMMARY_PREAMBLE));
        assert!(r.summary.contains("the summary"));
        let usage = r.usage.expect("normal path should carry usage");
        assert_eq!(usage.input, 7);
        let cap = &captured.lock().unwrap()[0];
        assert_eq!(cap.0, SUMMARIZATION_SYSTEM_PROMPT);
        assert!(cap.1.contains("<conversation>\n[User]: hello"));
        assert!(cap.1.contains("Create a structured summary"));
        assert!(!cap.1.contains("Additional focus"));
        let config = cap.2.as_ref().expect("Should carry config");
        // max_tokens is fixed at 2048: not scaled by reserve ratio, not min'd with model.max_tokens
        assert_eq!(config.max_tokens, Some(2048));
        assert!(config.reasoning_effort.is_none());
    }

    #[tokio::test]
    async fn test_generate_branch_summary_appends_custom_instructions() {
        let entries = vec![user_entry("e1", None, "hello")];
        let (result, captured) = run_generate(
            &entries,
            vec![done_event("s")],
            Some("focus on tests"),
            false,
            16_384,
            100_000,
        )
        .await;
        result.expect("Should succeed");
        let cap = &captured.lock().unwrap()[0];
        assert!(cap.1.contains("Additional focus: focus on tests"));
        assert!(cap.1.contains("Create a structured summary"));
    }

    #[tokio::test]
    async fn test_generate_branch_summary_replace_instructions() {
        let entries = vec![user_entry("e1", None, "hello")];
        let (result, captured) = run_generate(
            &entries,
            vec![done_event("s")],
            Some("just do X"),
            true,
            16_384,
            100_000,
        )
        .await;
        result.expect("Should succeed");
        let cap = &captured.lock().unwrap()[0];
        assert!(cap.1.contains("just do X"));
        assert!(
            !cap.1.contains("Create a structured summary"),
            "replacement should not leave default instructions"
        );
        assert!(!cap.1.contains("Additional focus"));
    }

    #[tokio::test]
    async fn test_generate_branch_summary_clears_inherited_reasoning_effort() {
        let mut config = base_config();
        config.reasoning_effort = Some(ThinkingLevel::Off);
        let captured = Arc::new(Mutex::new(Vec::new()));
        let stream_fn = fake_stream_fn(vec![done_event("s")], captured.clone());
        let entries = vec![user_entry("e1", None, "hello")];
        generate_branch_summary(
            &entries,
            &stream_fn,
            &test_model(100_000),
            config,
            None,
            false,
            16_384,
            None,
            None,
        )
        .await
        .expect("Should succeed");
        let sent = captured.lock().unwrap()[0]
            .2
            .as_ref()
            .expect("Should carry config")
            .clone();
        // pi does not set reasoning on this path; zen realizes this as an explicit clear (no inheritance from the caller's base)
        assert!(sent.reasoning_effort.is_none());
        assert_eq!(sent.max_tokens, Some(2048));
    }

    #[tokio::test]
    async fn test_generate_branch_summary_aborted_maps_to_compaction_error() {
        let entries = vec![user_entry("e1", None, "hello")];
        let (result, _) = run_generate(
            &entries,
            vec![AssistantMessageEvent::Done {
                reason: NonErrorStopReason::Stop,
                message: assistant_reply("x", StopReason::Aborted),
            }],
            None,
            false,
            16_384,
            100_000,
        )
        .await;
        assert!(matches!(result, Err(CompactionError::Aborted)));
    }

    #[tokio::test]
    async fn test_generate_branch_summary_error_maps_to_summarization_failed() {
        let entries = vec![user_entry("e1", None, "hello")];
        let mut error_reply = assistant_reply("x", StopReason::Error);
        error_reply.error_message = Some("boom".to_string());
        let (result, _) = run_generate(
            &entries,
            vec![AssistantMessageEvent::Done {
                reason: NonErrorStopReason::Stop,
                message: error_reply,
            }],
            None,
            false,
            16_384,
            100_000,
        )
        .await;
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Branch summary failed: boom"),
            "Actual: {err}"
        );
    }

    #[tokio::test]
    async fn test_generate_branch_summary_context_window_zero_falls_back_to_128k() {
        // Discriminating design: context_window=0 → fallback 128_000 → budget = 128_000-127_999 = 1
        // → truncation → short-circuit; without the fallback, budget = 0 (unlimited) → an LLM call would be made.
        // Precondition: the message's estimated tokens > 1 (about 300 characters, far more).
        let long = "the user explored a branch and did quite a lot of work in it ".repeat(5);
        let entries = vec![user_entry("e1", None, &long)];
        let (result, captured) = run_generate(&entries, vec![], None, false, 127_999, 0).await;
        let r = result.expect("should be Ok");
        assert_eq!(r.summary, "No content to summarize");
        assert!(captured.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_generate_branch_summary_appends_file_operations_section() {
        let details = json!({"readFiles": ["a.ts"], "modifiedFiles": []});
        let entries = vec![branch_summary_entry(
            "b1",
            None,
            "earlier exploration",
            Some(details),
            None,
        )];
        let (result, _) = run_generate(
            &entries,
            vec![done_event("s")],
            None,
            false,
            16_384,
            100_000,
        )
        .await;
        let r = result.expect("Should succeed");
        assert_eq!(r.read_files, vec!["a.ts".to_string()]);
        assert!(r.modified_files.is_empty());
        assert!(r.summary.starts_with(BRANCH_SUMMARY_PREAMBLE));
        // The file appendix should concatenate the list into the summary
        assert!(r.summary.contains("a.ts"), "actual: {}", r.summary);
    }
}
