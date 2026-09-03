use crate::compaction::CutPointResult;
use crate::session::tree::SessionTreeEntry;
use zen_ai::types::Message;
use zen_ai::utils::estimate::estimate_message_tokens;

/// After `find_cut_point` selects a cut point, if the cut point is not a turn start (e.g. falls on an assistant message),
/// it needs to backtrack to find the start entry of that turn so the "turn prefix" can be summarized separately—this function is that backtracker.
/// Scan from `entry_index` **inclusive** toward `start_index`; on hitting a turn start (User message—including pi's
/// bashExecution semantics—or a BranchSummary/CustomMessage entry), return its index; if start is reached without a hit, None.
/// Contract: `start_index <= entry_index < entries.len()` (guaranteed by caller; out-of-bounds panics, matching slice semantics).
/// Current consumers: `find_cut_point` (within crate) and unit tests; implementation confirmed no external consumers
pub(crate) fn find_turn_start_index(
    entries: &[SessionTreeEntry],
    entry_index: usize,
    start_index: usize,
) -> Option<usize> {
    for i in (start_index..=entry_index).rev() {
        match &entries[i] {
            SessionTreeEntry::BranchSummary { .. } | SessionTreeEntry::CustomMessage { .. } => {
                return Some(i);
            }
            SessionTreeEntry::Message { message, .. } => {
                // pi: role === "user" || role === "bashExecution" — both are User in zen
                if matches!(message, Message::User(_)) {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Compaction-preparation cut-point selector: accumulate message tokens from back to front until `keep_recent_tokens` is reached
/// Stop at budget, then fall to a legal cut point that **does not split tool-call pairs** (toolResult is never a cut point),
/// Finally, back off past non-message entries and determine whether an in-progress turn was split—turn-boundary awareness is the soul of this algorithm.
/// `entries[start_index..end_index]` is the window used for cut-point selection (the caller has already trimmed the part before the previous compaction boundary).
/// When no legal cut point exists, returns `first_kept_entry_index = start_index` (keep all, equivalent to no cut).
/// Current consumers: `prepare_compaction` (within crate); the pi-226 named capability surface (findCutPoint) is preserved via the function name,
/// Downgraded to pub(crate) because the output type `CutPointResult` is private to the compaction module (pub triggers private_interfaces),
/// and there are no consumers outside the crate—noted in the implementation docs.
pub(crate) fn find_cut_point(
    entries: &[SessionTreeEntry],
    start_index: usize,
    end_index: usize,
    keep_recent_tokens: usize,
) -> CutPointResult {
    let cut_points = find_valid_cut_points(entries, start_index, end_index);

    if cut_points.is_empty() {
        return CutPointResult {
            first_kept_entry_index: start_index,
            turn_start_index: None,
            is_split_turn: false,
        };
    }

    // ① Accumulate message tokens from back to front (only Message entries count; matches pi `if (entry.type !== "message") continue`)
    let mut accumulated_tokens: usize = 0;
    let mut cut_index = cut_points[0];

    for i in (start_index..end_index).rev() {
        let SessionTreeEntry::Message { message, .. } = &entries[i] else {
            continue;
        };
        accumulated_tokens += estimate_message_tokens(message);
        if accumulated_tokens >= keep_recent_tokens {
            // ② Take the first legal cut point >= i; if none (budget only reached after the last legal cut point),
            //    keep the original value—matching pi's inner for behavior of preserving the existing cutIndex when no hit
            if let Some(&candidate) = cut_points.iter().find(|&&c| c >= i) {
                cut_index = candidate;
            }
            break;
        }
    }

    // ③ Back off past non-message, non-compaction entries: metadata entries immediately before the cut point (model_change, etc.) stay in context with the tail,
    //    and are not folded into the summary; stop on Message/Compaction (matching the two break conditions of pi's while loop)
    while cut_index > start_index {
        match &entries[cut_index - 1] {
            SessionTreeEntry::Message { .. } | SessionTreeEntry::Compaction { .. } => break,
            _ => cut_index -= 1,
        }
    }

    let is_user_message = matches!(
        &entries[cut_index],
        SessionTreeEntry::Message {
            message: Message::User(_),
            ..
        }
    );
    let turn_start_index = if is_user_message {
        None // The cut point is exactly a turn start; no turn is split
    } else {
        find_turn_start_index(entries, cut_index, start_index)
    };

    CutPointResult {
        first_kept_entry_index: cut_index,
        turn_start_index,
        is_split_turn: !is_user_message && turn_start_index.is_some(),
    }
}

/// Legal cut-point scan: indices of entries in `entries[start_index..end_index]` that can serve as cut points (ascending).
/// zen variant mapping:
/// - `Message{User}` ← pi user/bashExecution roles (zen's bash execution messages are carried by User)
/// - `Message{Assistant}` ← pi assistant role
/// - `BranchSummary`/`CustomMessage` entries
/// - `Message{ToolResult}` is never valid (does not split tool-call pairs); metadata/Compaction entries are also invalid
fn find_valid_cut_points(
    entries: &[SessionTreeEntry],
    start_index: usize,
    end_index: usize,
) -> Vec<usize> {
    let mut cut_points = Vec::new();
    for (i, entry) in entries.iter().enumerate().take(end_index).skip(start_index) {
        match entry {
            SessionTreeEntry::Message { message, .. } => match message {
                Message::User(_) | Message::Assistant(_) => cut_points.push(i),
                Message::ToolResult(_) => {} // Never a cut point
            },
            SessionTreeEntry::BranchSummary { .. } | SessionTreeEntry::CustomMessage { .. } => {
                cut_points.push(i)
            }
            _ => {}
        }
    }
    cut_points
}

#[cfg(test)]
mod tests {
    use super::*;
    use zen_ai::types::{
        AssistantMessage, ContentBlock, ContentBlockList, StopReason, TextContent,
        ToolResultMessage, Usage, UserMessage,
    };

    // ── Construction helpers (tokens are budgetable: 4 text chars ≈ 1 token) ────────────

    fn user_entry(id: &str, text: &str) -> SessionTreeEntry {
        SessionTreeEntry::Message {
            id: id.to_string(),
            parent_id: None,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            message: Message::User(UserMessage {
                content: ContentBlockList::Text(text.to_string()),
                timestamp: 0,
            }),
        }
    }

    fn assistant_entry(id: &str, text: &str) -> SessionTreeEntry {
        SessionTreeEntry::Message {
            id: id.to_string(),
            parent_id: None,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            message: Message::Assistant(AssistantMessage {
                content: vec![ContentBlock::Text(TextContent {
                    text: text.to_string(),
                    text_signature: None,
                })],
                api: "test".to_string(),
                provider: "test".to_string(),
                model: "test-model".to_string(),
                response_model: None,
                response_id: None,
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: 0,
            }),
        }
    }

    fn tool_result_entry(id: &str) -> SessionTreeEntry {
        SessionTreeEntry::Message {
            id: id.to_string(),
            parent_id: None,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            message: Message::ToolResult(ToolResultMessage {
                tool_call_id: format!("call-{id}"),
                tool_name: "bash".to_string(),
                content: vec![],
                details: None,
                usage: None,
                is_error: false,
                added_tool_names: None,
                timestamp: 0,
            }),
        }
    }

    fn branch_summary_entry(id: &str) -> SessionTreeEntry {
        SessionTreeEntry::BranchSummary {
            id: id.to_string(),
            parent_id: None,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            from_id: "src".to_string(),
            summary: "branch summary".to_string(),
            details: None,
            usage: None,
            from_hook: None,
        }
    }

    fn compaction_entry(id: &str) -> SessionTreeEntry {
        SessionTreeEntry::Compaction {
            id: id.to_string(),
            parent_id: None,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            summary: "compaction summary".to_string(),
            first_kept_entry_id: None,
            tokens_before: 100,
            retained_tail: None,
            details: None,
            usage: None,
            from_hook: None,
        }
    }

    /// Metadata entry (non-message, non-boundary), used to verify scan-skipping logic
    fn custom_meta_entry(id: &str) -> SessionTreeEntry {
        SessionTreeEntry::Custom {
            id: id.to_string(),
            parent_id: None,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            custom_type: "test".to_string(),
            data: None,
        }
    }

    // ── find_turn_start_index ────────────────────────────────

    #[test]
    fn test_find_turn_start_index_returns_user_at_entry_index() {
        let entries = vec![user_entry("a", "aaaa"), user_entry("b", "aaaa")];
        assert_eq!(find_turn_start_index(&entries, 1, 0), Some(1));
    }

    #[test]
    fn test_find_turn_start_index_walks_back_to_user_over_assistant_and_tool_result() {
        // assistant/toolResult are not turn starts; backtrack all the way to User
        let entries = vec![
            user_entry("a", "aaaa"),
            assistant_entry("b", "aaaa"),
            tool_result_entry("c"),
            assistant_entry("d", "aaaa"),
        ];
        assert_eq!(find_turn_start_index(&entries, 3, 0), Some(0));
    }

    #[test]
    fn test_find_turn_start_index_branch_summary_is_turn_boundary() {
        // BranchSummary is a turn boundary; do not continue upward
        let entries = vec![
            user_entry("a", "aaaa"),
            branch_summary_entry("b"),
            assistant_entry("c", "aaaa"),
        ];
        assert_eq!(find_turn_start_index(&entries, 2, 0), Some(1));
    }

    #[test]
    fn test_find_turn_start_index_skips_metadata_while_walking_back() {
        let entries = vec![
            user_entry("a", "aaaa"),
            custom_meta_entry("b"),
            assistant_entry("c", "aaaa"),
        ];
        assert_eq!(find_turn_start_index(&entries, 2, 0), Some(0));
    }

    #[test]
    fn test_find_turn_start_index_start_bound_is_inclusive() {
        let entries = vec![user_entry("a", "aaaa"), assistant_entry("b", "aaaa")];
        // entry_index == start and that position is User → hit (start inclusive)
        assert_eq!(find_turn_start_index(&entries, 0, 0), Some(0));
        // Range contains only assistant; User is outside start → None (does not cross the lower bound)
        assert_eq!(find_turn_start_index(&entries, 1, 1), None);
    }

    #[test]
    fn test_find_turn_start_index_none_when_no_turn_start_in_range() {
        let entries = vec![
            custom_meta_entry("a"),
            tool_result_entry("b"),
            assistant_entry("c", "aaaa"),
        ];
        assert_eq!(find_turn_start_index(&entries, 2, 0), None);
    }

    // ── find_cut_point ───────────────────────────────────────

    #[test]
    fn test_find_cut_point_empty_window_returns_start() {
        let entries = vec![user_entry("a", "aaaa")];
        let result = find_cut_point(&entries, 1, 1, 0);
        assert_eq!(result.first_kept_entry_index, 1);
        assert_eq!(result.turn_start_index, None);
        assert!(!result.is_split_turn);
    }

    #[test]
    fn test_find_cut_point_no_valid_cut_points_returns_start() {
        // Window contains only toolResult and metadata entries → no legal cut point; keep all
        let entries = vec![tool_result_entry("a"), custom_meta_entry("b")];
        let result = find_cut_point(&entries, 0, 2, 0);
        assert_eq!(result.first_kept_entry_index, 0);
        assert_eq!(result.turn_start_index, None);
        assert!(!result.is_split_turn);
    }

    #[test]
    fn test_find_cut_point_zero_budget_keeps_last_message() {
        // keep=0: the first message accumulated from the tail meets the budget → cut at the last User
        let entries = vec![
            user_entry("a", "aaaa"),
            user_entry("b", "aaaa"),
            user_entry("c", "aaaa"),
        ];
        let result = find_cut_point(&entries, 0, 3, 0);
        assert_eq!(result.first_kept_entry_index, 2);
        assert_eq!(result.turn_start_index, None);
        assert!(!result.is_split_turn);
    }

    #[test]
    fn test_find_cut_point_large_budget_keeps_first_cut_point() {
        // Budget never met → keep the first legal cut point (keep all)
        let entries = vec![
            user_entry("a", "aaaa"),
            user_entry("b", "aaaa"),
            user_entry("c", "aaaa"),
        ];
        let result = find_cut_point(&entries, 0, 3, 10_000);
        assert_eq!(result.first_kept_entry_index, 0);
        assert!(!result.is_split_turn);
    }

    #[test]
    fn test_find_cut_point_budget_lands_on_intermediate_user() {
        // 1 token each, keep=2: i=2 accumulates 1 (insufficient), i=1 accumulates 2 (meets budget) → cut point 1 (User)
        let entries = vec![
            user_entry("a", "aaaa"),
            user_entry("b", "aaaa"),
            user_entry("c", "aaaa"),
        ];
        let result = find_cut_point(&entries, 0, 3, 2);
        assert_eq!(result.first_kept_entry_index, 1);
        assert_eq!(result.turn_start_index, None);
        assert!(!result.is_split_turn);
    }

    #[test]
    fn test_find_cut_point_split_turn_on_assistant_message() {
        // keep=2: cut point lands on assistant (not a turn start) → backtrack to User → split turn
        let entries = vec![
            user_entry("a", "aaaa"),
            assistant_entry("b", "aaaa"),
            user_entry("c", "aaaa"),
        ];
        let result = find_cut_point(&entries, 0, 3, 2);
        assert_eq!(result.first_kept_entry_index, 1);
        assert_eq!(result.turn_start_index, Some(0));
        assert!(result.is_split_turn);
    }

    #[test]
    fn test_find_cut_point_retreats_over_metadata_before_cut() {
        // Cut-point backtracking skips metadata entries (they stay in context with the tail);
        // Lands on Custom metadata → not User → backtrack to User → split
        let entries = vec![
            user_entry("a", "aaaa"),
            custom_meta_entry("b"),
            user_entry("c", "aaaa"),
        ];
        let result = find_cut_point(&entries, 0, 3, 0);
        assert_eq!(result.first_kept_entry_index, 1);
        assert_eq!(result.turn_start_index, Some(0));
        assert!(result.is_split_turn);
    }

    #[test]
    fn test_find_cut_point_retreat_stops_at_compaction() {
        // Backtracking stops on Compaction (does not cross compaction boundaries)
        let entries = vec![
            user_entry("a", "aaaa"),
            compaction_entry("b"),
            user_entry("c", "aaaa"),
        ];
        let result = find_cut_point(&entries, 0, 3, 0);
        assert_eq!(result.first_kept_entry_index, 2);
        assert_eq!(result.turn_start_index, None);
        assert!(!result.is_split_turn);
    }

    #[test]
    fn test_find_cut_point_tool_result_never_selected_as_cut() {
        // toolResult is never a cut point (does not split tool-call pairs);
        // With keep=0, the cut point goes straight to the last User; a toolResult in between is not selected
        let entries = vec![
            user_entry("a", "aaaa"),
            tool_result_entry("b"),
            user_entry("c", "aaaa"),
        ];
        let result = find_cut_point(&entries, 0, 3, 0);
        assert_eq!(result.first_kept_entry_index, 2);
        assert!(!result.is_split_turn);
    }
}
