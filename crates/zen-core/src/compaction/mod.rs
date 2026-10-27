pub mod branch;
pub mod cut;
pub mod summarize;
pub mod utils;

use crate::compaction::cut::find_cut_point;
use crate::compaction::summarize::combine_usage;
use crate::compaction::summarize::generate_summary;
use crate::compaction::summarize::generate_turn_prefix_summary;
use crate::compaction::utils::compute_file_lists;
use crate::compaction::utils::extract_file_ops_from_message;
use crate::compaction::utils::format_file_operations;
use crate::session::context::build_session_context;
use crate::session::context::session_entry_to_context_messages;
use crate::session::tree::SessionTreeEntry;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashSet;
use thiserror::Error;
use tokio_util::sync::CancellationToken;
use zen_ai::api::openai_stream::StreamConfig;
use zen_ai::types::{Message, Model, StreamFn, ThinkingLevel, Usage};
use zen_ai::utils::estimate::estimate_messages;
use zen_ai::utils::retry::RetryPolicy;

/// Configuration container for compaction thresholds and retention policies, shared by should_compact/find_cut_point/compact
#[derive(Debug, Clone, Copy)]
pub struct CompactionSettings {
    pub enabled: bool, // When false, should_compact always returns false (automatic compaction switch)
    pub reserve_tokens: u64, // Tokens reserved for the summary prompt and output; u64 matches Model.max_tokens
    pub keep_recent_tokens: u64, // Approximate token budget for recent context retained after compaction
}

/// Default compaction settings (matching pi `DEFAULT_COMPACTION_SETTINGS`).
pub const DEFAULT_COMPACTION_SETTINGS: CompactionSettings = CompactionSettings {
    enabled: true,
    reserve_tokens: 16_384,
    keep_recent_tokens: 20_000,
};

/// Error surface of the compaction flow
#[derive(Debug, Error)]
pub enum CompactionError {
    #[error("summarization aborted")]
    Aborted, // Summary LLM call has stop_reason = aborted (including cancellation)
    #[error("summarization failed: {0}")]
    SummarizationFailed(String), // Summary LLM call has stop_reason = error
}

/// Compaction trigger reason (introduced by T-AP-04-3 scheduling), corresponding to pi's string literals `"overflow" | "threshold" | "manual"`.
/// Threads through compaction lifecycle events (`AgentEvent::CompactionStart/End` payload), scheduling branches, and UI status text
/// (the three variants map one-to-one to zen-app `CompactionStatusReason`).
/// serde: `lowercase` produces `"overflow"/"threshold"/"manual"`, exactly matching pi literals; event variant names via
/// `AgentEvent` outer `snake_case` produces `"compaction_start"/"compaction_end"`, also matching pi event names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CompactionReason {
    /// LLM returns a context-overflow error, or a successful response's usage exceeds the configured window
    Overflow,
    /// Context tokens exceed the "window − reserve_tokens" threshold
    Threshold,
    /// Manually triggered by user `/compact [instructions]`
    Manual,
}

/// Typed container for the list of files involved in compaction;
/// Serialized by zen-app during persistence as SessionTreeEntry::Compaction.details (Option<Value>).
/// derive(Clone, Serialize): travels with the enum as the AgentEvent::CompactionEnd payload (T-AP-04-3 event surface).
#[derive(Debug, Clone, Serialize)]
pub struct CompactionDetails {
    pub read_files: Vec<String>, // Sorted read-only file list (produced by compute_file_lists, excluding modified files)
    pub modified_files: Vec<String>, // Sorted modified-file list (edited ∪ written)
}

/// Output of compact, used by the caller (T-AP-04-3 scheduling) to construct SessionTreeEntry::Compaction and write it back to the session tree.
/// derive(Clone, Serialize): travels with the enum as the AgentEvent::CompactionEnd payload (T-AP-04-3 event surface).
#[derive(Debug, Clone, Serialize)]
pub struct CompactionResult {
    pub summary: String, // Summary text (including the file operation addendum)
    pub first_kept_entry_id: String, // Retained history start entry id (optional for pi = transitional artifact, required for zen)
    pub tokens_before: u64,          // Estimated context token count before compaction
    pub usage: Option<Usage>, // Summary LLM call usage (two combine operations for split-turn)
    pub retained_tail: Vec<Message>, // Retained tail messages stored directly in the entry (empty = none)
    pub details: CompactionDetails,  // File operation list
}

/// Argument bundle for compact, describing all inputs to one compaction run
pub struct CompactionPreparation {
    first_kept_entry_id: String,         // Retained start entry id
    messages_to_summarize: Vec<Message>, // Historical messages included in the summary (Compaction entries skipped)
    turn_prefix_messages: Vec<Message>, // Turn-prefix messages summarized separately for split-turn (empty otherwise)
    retained_tail: Vec<Message>,        // Retained tail messages
    is_split_turn: bool,                // Whether the cut point splits an in-progress turn
    tokens_before: u64, // Estimated token count before compaction (read via the estimate_messages accessor)
    previous_summary: Option<String>, // Previous compaction summary (for iterative updates)
    file_ops: FileOperations, // File operations accumulated since the previous compaction boundary
    settings: CompactionSettings, // Settings used in this run
}

/// Output of find_cut_point, describing the selected cut point and its turn semantics.
/// pub(crate): find_cut_point in cut.rs is pub(crate) and returns this type,
/// A private field triggers a private_interfaces warning; the field remains accessible within the module.
pub(crate) struct CutPointResult {
    first_kept_entry_index: usize, // Index of the first entry retained after compaction
    turn_start_index: Option<usize>, // Starting entry index of the turn containing the cut point
    is_split_turn: bool,           // Whether the cut point splits an in-progress turn
}

/// File-operation accumulator (read/written/edited buckets), used to generate the summary addendum and details.
#[derive(Default)]
pub(crate) struct FileOperations {
    read: HashSet<String>,    // Paths read via the read tool
    written: HashSet<String>, // Paths written in full via the write tool
    edited: HashSet<String>,  // Paths modified via the edit tool
}

impl FileOperations {
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// The compaction scheduler (T-AP-04-3) runs after each context assembly to determine whether the context is now costly enough to require compaction—
/// In short: compact when the current context no longer fits the window after subtracting the summary reserve.
/// Always returns false when `settings.enabled` is false (disables automatic compaction entirely);
/// The predicate matches pi: `context_tokens > context_window - settings.reserve_tokens`.
/// Consumers: T-AP-04-3 (AgentSession::check_compaction, zen-app/src/session/compaction.rs).
pub fn should_compact(
    context_tokens: usize,
    context_window: u64,
    settings: &CompactionSettings,
) -> bool {
    if !settings.enabled {
        return false;
    }
    // Empty context does not trigger compaction (prepare already returns None for an empty path); non-empty context behavior is unchanged
    context_tokens as u64 > context_window.saturating_sub(settings.reserve_tokens)
}

/// Called by the compaction scheduler after `should_compact` evaluates to true: prepares all inputs for one compaction from the current session path entries—
/// Locates the previous compaction boundary, estimates the pre-compaction token count, selects a cut point, and extracts three message groups (to summarize, turn prefix, retained tail) and file operations.
/// “Ready but not fired”: this function makes no LLM calls and is purely synchronous.
/// `Ok(None)` indicates not applicable (the path is empty, or the last entry is already Compaction—nothing left to compact);
/// `previous_summary` comes from the most recent Compaction entry in the path and supports iterative summary updates.
/// Consumer: T-AP-04-3 (scheduler); directly covered by unit tests.
pub fn prepare_compaction(
    path_entries: &[SessionTreeEntry],
    settings: &CompactionSettings,
) -> Result<Option<CompactionPreparation>, CompactionError> {
    // Empty path, or the last entry is already Compaction — nothing left to compact
    if path_entries.is_empty()
        || matches!(
            path_entries.last(),
            Some(SessionTreeEntry::Compaction { .. })
        )
    {
        return Ok(None);
    }

    // ① Locate the most recent Compaction boundary by scanning backward from the tail
    let prev_compaction_index = path_entries
        .iter()
        .rposition(|entry| matches!(entry, SessionTreeEntry::Compaction { .. }));

    // ② Previous compaction summary + current window start: resolve the entry index for firstKeptEntryId first,
    //    Fall back to prev+1 if id is missing or not found (aligns with the pi findIndex / -1 branch)
    let mut previous_summary: Option<String> = None;
    let mut boundary_start = 0;
    if let Some(prev_index) = prev_compaction_index {
        if let SessionTreeEntry::Compaction {
            summary,
            first_kept_entry_id,
            ..
        } = &path_entries[prev_index]
        {
            previous_summary = Some(summary.clone());
            boundary_start = first_kept_entry_id
                .as_deref()
                .and_then(|id| path_entries.iter().position(|entry| entry.id() == id))
                .unwrap_or(prev_index + 1);
        }
    }
    let boundary_end = path_entries.len();

    // ③ Pre-compaction token estimation: estimate tokens for context messages built from the full path
    //    (Aligns with pi estimateContextTokens(buildSessionContext(pathEntries).messages).tokens;
    //    zen's estimate_messages is the message-level form of pi estimateContextTokens)
    let tokens_before =
        estimate_messages(&build_session_context(path_entries).messages).tokens as u64;

    // ④ Select the cut point (window [boundary_start, boundary_end))
    let cut_point = find_cut_point(
        path_entries,
        boundary_start,
        boundary_end,
        settings.keep_recent_tokens as usize,
    );

    // zen entries always have an id (String is non-optional)
    let first_kept_entry_id = path_entries[cut_point.first_kept_entry_index]
        .id()
        .to_string();

    // ⑤ Three-segment message extraction: for a split turn, the history segment ends at the turn start; otherwise, it ends at the cut point
    let history_end = if cut_point.is_split_turn {
        // find_cut_point contract: is_split_turn ⟹ turn_start_index is Some
        cut_point
            .turn_start_index
            .unwrap_or(cut_point.first_kept_entry_index)
    } else {
        cut_point.first_kept_entry_index
    };
    let messages_to_summarize =
        entries_to_messages_for_summarization(path_entries, boundary_start, history_end);
    let turn_prefix_messages = match cut_point.turn_start_index {
        Some(turn_start) if cut_point.is_split_turn => entries_to_messages_for_summarization(
            path_entries,
            turn_start,
            cut_point.first_kept_entry_index,
        ),
        _ => Vec::new(),
    };
    let retained_tail = entries_to_messages_for_summarization(
        path_entries,
        cut_point.first_kept_entry_index,
        boundary_end,
    );

    // ⑥ File-operation accumulation: seed from the previous compaction details (only seed for non-hook output) → history messages → split-turn turn prefix
    let mut file_ops = FileOperations::new();
    if let Some(prev_index) = prev_compaction_index {
        if let SessionTreeEntry::Compaction {
            details: Some(details),
            from_hook,
            ..
        } = &path_entries[prev_index]
        {
            // pi: !fromHook && details — an absent fromHook (None) is treated as non-hook
            if !from_hook.unwrap_or(false) {
                seed_file_ops_from_details(details, &mut file_ops);
            }
        }
    }
    for message in &messages_to_summarize {
        extract_file_ops_from_message(message, &mut file_ops);
    }
    if cut_point.is_split_turn {
        for message in &turn_prefix_messages {
            extract_file_ops_from_message(message, &mut file_ops);
        }
    }

    Ok(Some(CompactionPreparation {
        first_kept_entry_id,
        messages_to_summarize,
        turn_prefix_messages,
        retained_tail,
        is_split_turn: cut_point.is_split_turn,
        tokens_before,
        previous_summary,
        file_ops,
        settings: *settings,
    }))
}

/// Convert entries in the window [start_index, end_index) into summary input messages:
/// Delegate each entry to session_entry_to_context_messages (the existing converter in context.rs) and skip Compaction entries
/// (Matches pi getMessageFromEntryForCompaction semantics—the Compaction summary is being summarized again and is excluded from the input);
/// Metadata entries are naturally filtered out because the converter returns an empty Vec (aligns with the undefined branch of pi getMessageFromEntry).
fn entries_to_messages_for_summarization(
    path_entries: &[SessionTreeEntry],
    start_index: usize,
    end_index: usize,
) -> Vec<Message> {
    let mut messages = Vec::new();
    for (index, entry) in path_entries
        .iter()
        .enumerate()
        .take(end_index)
        .skip(start_index)
    {
        if matches!(entry, SessionTreeEntry::Compaction { .. }) {
            continue;
        }
        messages.extend(session_entry_to_context_messages(
            entry,
            index,
            path_entries,
        ));
    }
    messages
}

/// Seed the file-operation buckets from the previous compaction entry's details: readFiles → read, modifiedFiles → edited.
/// details is JSON persisted by zen-app; the key-name contract follows the pi JSONL shape (camelCase
/// readFiles/modifiedFiles) — already written in the same shape by T-AP-04-3 `SessionStore::append_compaction`
/// (seam is closed; evidence: tasks/T-AP-04-3-implemented.md, including persist→reopen→seed round-trip tests);
/// Silently skip non-string array elements (pi calls add directly and relies on callers to guarantee this; zen is defensively stricter).
pub(crate) fn seed_file_ops_from_details(details: &Value, file_ops: &mut FileOperations) {
    if let Some(read_files) = details.get("readFiles").and_then(Value::as_array) {
        for file in read_files.iter().filter_map(Value::as_str) {
            file_ops.read.insert(file.to_string());
        }
    }
    if let Some(modified_files) = details.get("modifiedFiles").and_then(Value::as_array) {
        for file in modified_files.iter().filter_map(Value::as_str) {
            file_ops.edited.insert(file.to_string());
        }
    }
}

/// Called by the compaction scheduler once `CompactionPreparation` is ready: run the summary LLM call and assemble the compaction output—
/// For a split turn, summarize the history first, then the turn prefix (up to 0.5×reserve), concatenate both segments, and merge usage;
/// Finally, append the `<read-files>`/`<modified-files>` sections to the summary.
/// `stream_config` is the caller-provided base configuration (with connection settings such as api_key/headers); this function overrides it
/// `max_tokens = min(0.8 × settings.reserve_tokens, model.max_tokens (0 means unlimited))`
/// Aligns with `reasoning_effort` (set only when `model.reasoning` is present and `thinking_level` is not off, matching pi's conditional construction).
/// Cancellation/retry semantics: `cancel` is passed through to stream collection; `retry` uses exponential backoff via zen-ai `retry_assistant_call`.
/// Consumer: T-AP-04-3 (scheduling).
// Mirror the parameter surface of pi compact(...) (8 parameters); reassess the parameter structure when T-AP-04-3 scheduling integration is introduced
#[allow(clippy::too_many_arguments)]
pub async fn compact(
    preparation: CompactionPreparation,
    stream_fn: &StreamFn,
    model: &Model,
    stream_config: StreamConfig,
    custom_instructions: Option<&str>,
    thinking_level: Option<ThinkingLevel>,
    cancel: Option<CancellationToken>,
    retry: Option<RetryPolicy>,
) -> Result<CompactionResult, CompactionError> {
    let CompactionPreparation {
        messages_to_summarize,
        turn_prefix_messages,
        is_split_turn,
        ..
    } = &preparation;

    let mut llm_config = stream_config;
    llm_config.reasoning_effort = if model.reasoning {
        thinking_level.filter(|level| *level != ThinkingLevel::Off)
    } else {
        None
    };

    let (mut summary, usage) = if *is_split_turn && !turn_prefix_messages.is_empty() {
        // split-turn: first summarize history (may be empty → "No prior history."), then summarize the turn prefix, concatenate both sections, and merge usage
        let (history_text, history_usage): (String, Option<Usage>) =
            if messages_to_summarize.is_empty() {
                ("No prior history.".to_string(), None)
            } else {
                let (text, usage) = generate_summary(
                    messages_to_summarize,
                    stream_fn,
                    model,
                    llm_config.clone(),
                    preparation.settings.reserve_tokens,
                    cancel.clone(),
                    custom_instructions,
                    preparation.previous_summary.as_deref(),
                    retry,
                )
                .await?;
                (text, Some(usage))
            };
        let (prefix_text, prefix_usage) = generate_turn_prefix_summary(
            turn_prefix_messages,
            stream_fn,
            model,
            llm_config,
            preparation.settings.reserve_tokens,
            cancel,
            retry,
        )
        .await?;
        let summary = format!(
            "{}\n\n---\n\n**Turn Context (split turn):**\n\n{}",
            history_text, prefix_text
        );
        let usage = match history_usage {
            Some(history) => combine_usage(&history, &prefix_usage),
            None => prefix_usage,
        };
        (summary, usage)
    } else {
        generate_summary(
            messages_to_summarize,
            stream_fn,
            model,
            llm_config,
            preparation.settings.reserve_tokens,
            cancel,
            custom_instructions,
            preparation.previous_summary.as_deref(),
            retry,
        )
        .await?
    };

    // pi: summary += formatFileOperations(readFiles, modifiedFiles) — empty lists result in a no-op concatenation
    let (read_files, modified_files) = compute_file_lists(&preparation.file_ops);
    summary.push_str(&format_file_operations(&read_files, &modified_files));

    Ok(CompactionResult {
        summary,
        first_kept_entry_id: preparation.first_kept_entry_id,
        tokens_before: preparation.tokens_before,
        usage: Some(usage), // zen Usage always exists (zero value = provider did not report it ≈ pi's default)
        retained_tail: preparation.retained_tail,
        details: CompactionDetails {
            read_files,
            modified_files,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use zen_ai::types::{ContentBlockList, UserMessage};

    // ── construction helpers ─────────────────────────────────────────────

    fn user_message(text: &str) -> Message {
        Message::User(UserMessage {
            content: ContentBlockList::Text(text.to_string()),
            timestamp: 0,
        })
    }

    fn user_entry(id: &str, text: &str) -> SessionTreeEntry {
        SessionTreeEntry::Message {
            id: id.to_string(),
            parent_id: None,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            message: user_message(text),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compaction_entry(
        id: &str,
        summary: &str,
        first_kept: Option<&str>,
        details: Option<Value>,
        from_hook: Option<bool>,
    ) -> SessionTreeEntry {
        SessionTreeEntry::Compaction {
            id: id.to_string(),
            parent_id: None,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            summary: summary.to_string(),
            first_kept_entry_id: first_kept.map(str::to_string),
            tokens_before: 100,
            retained_tail: None,
            details,
            usage: None,
            from_hook,
        }
    }

    /// Metadata entries (the Custom converter produces no messages), used to verify summary input filtering
    fn custom_entry(id: &str) -> SessionTreeEntry {
        SessionTreeEntry::Custom {
            id: id.to_string(),
            parent_id: None,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            custom_type: "test".to_string(),
            data: None,
        }
    }

    /// (reserve_tokens, keep_recent_tokens)
    fn settings(reserve_tokens: u64, keep_recent_tokens: u64) -> CompactionSettings {
        CompactionSettings {
            enabled: true,
            reserve_tokens,
            keep_recent_tokens,
        }
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

    /// Union of the three extracted sections (for verifying partition invariants independent of the split point)
    fn all_extracted(prep: &CompactionPreparation) -> Vec<Message> {
        let mut messages = prep.messages_to_summarize.clone();
        messages.extend(prep.turn_prefix_messages.iter().cloned());
        messages.extend(prep.retained_tail.iter().cloned());
        messages
    }

    // ── DEFAULT_COMPACTION_SETTINGS / should_compact ────────

    #[test]
    fn test_default_settings_values() {
        let s = DEFAULT_COMPACTION_SETTINGS;
        assert!(s.enabled);
        assert_eq!(s.reserve_tokens, 16_384);
        assert_eq!(s.keep_recent_tokens, 20_000);
    }

    #[test]
    fn test_should_compact_triggers_over_budget() {
        // 900 > 1000 - 200 = 800
        assert!(should_compact(900, 1_000, &settings(200, 0)));
    }

    #[test]
    fn test_should_compact_not_triggered_under_budget() {
        assert!(!should_compact(700, 1_000, &settings(200, 0)));
    }

    #[test]
    fn test_should_compact_boundary_exact_false() {
        // The predicate is strictly greater-than: exactly equal to the threshold does not trigger it
        assert!(!should_compact(800, 1_000, &settings(200, 0)));
    }

    #[test]
    fn test_should_compact_disabled_returns_false() {
        let s = CompactionSettings {
            enabled: false,
            ..settings(200, 0)
        };
        assert!(!should_compact(usize::MAX, 1_000, &s));
    }

    #[test]
    fn test_should_compact_reserve_exceeds_window_saturates() {
        // When reserve > window, the threshold saturates at 0, so any nonempty context triggers it
        assert!(should_compact(1, 100, &settings(500, 0)));
    }

    // ── prepare_compaction: inapplicable branches ───────────────────────

    #[test]
    fn test_prepare_compaction_empty_path_returns_none() {
        let entries: Vec<SessionTreeEntry> = vec![];
        let prep = prepare_compaction(&entries, &settings(1_000, 0)).unwrap();
        assert!(prep.is_none());
    }

    #[test]
    fn test_prepare_compaction_trailing_compaction_returns_none() {
        let entries = vec![
            user_entry("a", "u-a"),
            compaction_entry("c1", "prev", None, None, None),
        ];
        let prep = prepare_compaction(&entries, &settings(1_000, 0)).unwrap();
        assert!(prep.is_none());
    }

    // ── prepare_compaction: boundary detection and message extraction ────────────────

    #[test]
    fn test_prepare_compaction_extracts_previous_summary_and_keeps_tail() {
        // Single-element window [2,3): the split point can only fall within u-b — the history section is empty, and the tail is [u-b]
        let entries = vec![
            user_entry("a", "u-a"),
            compaction_entry("c1", "prev", None, None, None),
            user_entry("b", "u-b"),
        ];
        let prep = prepare_compaction(&entries, &settings(1_000, 0))
            .unwrap()
            .expect("Non-empty window should be preparable");
        assert_eq!(prep.previous_summary.as_deref(), Some("prev"));
        assert!(prep.messages_to_summarize.is_empty());
        assert_eq!(texts(&prep.retained_tail), vec!["u-b".to_string()]);
        assert_eq!(prep.first_kept_entry_id, "b");
        assert!(prep.tokens_before > 0);
    }

    #[test]
    fn test_prepare_compaction_boundary_resolves_via_first_kept_entry_id() {
        // first_kept = "b" (index 1) → window [1,5); u-a before the boundary does not enter any section
        let entries = vec![
            user_entry("a", "u-a"),
            user_entry("b", "u-b"),
            compaction_entry("c1", "prev", Some("b"), None, None),
            user_entry("c", "u-c"),
            user_entry("d", "u-d"),
        ];
        let prep = prepare_compaction(&entries, &settings(1_000, 0))
            .unwrap()
            .expect("Non-empty window should be preparable");
        assert_eq!(prep.previous_summary.as_deref(), Some("prev"));
        assert_eq!(
            texts(&all_extracted(&prep)),
            vec!["u-b".to_string(), "u-c".to_string(), "u-d".to_string()]
        );
    }

    #[test]
    fn test_prepare_compaction_boundary_falls_back_when_first_kept_id_missing() {
        // first_kept points to a nonexistent id → falls back to prev+1 (matching pi's findIndex -1 branch)
        let entries = vec![
            user_entry("a", "u-a"),
            compaction_entry("c1", "prev", Some("zzz"), None, None),
            user_entry("b", "u-b"),
        ];
        let prep = prepare_compaction(&entries, &settings(1_000, 0))
            .unwrap()
            .expect("Non-empty window should be preparable");
        assert_eq!(prep.previous_summary.as_deref(), Some("prev"));
        assert_eq!(texts(&all_extracted(&prep)), vec!["u-b".to_string()]);
    }

    #[test]
    fn test_prepare_compaction_interior_compaction_entries_skipped() {
        // first_kept points back to "a" → the window contains two Compaction entries, neither of which may enter the summary input
        // (Otherwise, the summary prompt will contain nested “[Context Summarize” markers)
        let entries = vec![
            user_entry("a", "u-a"),
            compaction_entry("c1", "first", None, None, None),
            user_entry("b", "u-b"),
            compaction_entry("c2", "second", Some("a"), None, None),
            user_entry("c", "u-c"),
        ];
        let prep = prepare_compaction(&entries, &settings(1_000, 0))
            .unwrap()
            .expect("Non-empty window should be preparable");
        // previous_summary uses the most recent one (c2)
        assert_eq!(prep.previous_summary.as_deref(), Some("second"));
        let extracted = texts(&all_extracted(&prep));
        assert_eq!(
            extracted,
            vec!["u-a".to_string(), "u-b".to_string(), "u-c".to_string()]
        );
    }

    #[test]
    fn test_prepare_compaction_metadata_entries_produce_no_messages() {
        let entries = vec![
            user_entry("a", "u-a"),
            custom_entry("m1"),
            user_entry("b", "u-b"),
        ];
        let prep = prepare_compaction(&entries, &settings(1_000, 0))
            .unwrap()
            .expect("Non-empty window should be preparable");
        // Matches the full-window conversion result: Custom entries are naturally filtered out
        let expected = texts(&entries_to_messages_for_summarization(&entries, 0, 3));
        assert_eq!(texts(&all_extracted(&prep)), expected);
        assert_eq!(expected.len(), 2);
    }

    #[test]
    fn test_prepare_compaction_segments_partition_full_window() {
        // Split-point invariant: concatenating the three segments == the conversion result for the full window [0, len) (holds for both split and non-split cases)
        let entries = vec![
            user_entry("a", "u-a"),
            custom_entry("m1"),
            user_entry("b", "u-b"),
            user_entry("c", "u-c"),
        ];
        for keep in [0u64, 100_000] {
            let prep = prepare_compaction(&entries, &settings(1_000, keep))
                .unwrap()
                .expect("Non-empty window should be preparable");
            let expected = texts(&entries_to_messages_for_summarization(&entries, 0, 4));
            assert_eq!(texts(&all_extracted(&prep)), expected, "keep={keep}");
        }
    }

    // ── prepare_compaction: file operation seeding ─────────────────────

    #[test]
    fn test_prepare_compaction_seeds_file_ops_from_previous_details() {
        let details = serde_json::json!({
            "readFiles": ["a.ts", "b.md"],
            "modifiedFiles": ["a.ts", "c.rs"]
        });
        let entries = vec![
            user_entry("a", "u-a"),
            compaction_entry("c1", "prev", None, Some(details), None),
            user_entry("b", "u-b"),
        ];
        let prep = prepare_compaction(&entries, &settings(1_000, 0))
            .unwrap()
            .expect("Non-empty window should be preparable");
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
    }

    #[test]
    fn test_prepare_compaction_skips_seeding_when_prev_from_hook() {
        let details = serde_json::json!({"readFiles": ["a.ts"]});
        let entries = vec![
            user_entry("a", "u-a"),
            compaction_entry("c1", "prev", None, Some(details), Some(true)),
            user_entry("b", "u-b"),
        ];
        let prep = prepare_compaction(&entries, &settings(1_000, 0))
            .unwrap()
            .expect("Non-empty window should be preparable");
        // pi: seed only when !fromHook → the hook-produced previous compaction does not accumulate the file list twice
        assert!(prep.file_ops.read.is_empty());
        assert!(prep.file_ops.edited.is_empty());
    }

    // ── seed_file_ops_from_details (private, directly unit-tested) ──────────

    #[test]
    fn test_seed_file_ops_from_details_seeds_buckets() {
        let details = serde_json::json!({
            "readFiles": ["a.ts", "b.md"],
            "modifiedFiles": ["a.ts", "c.rs"]
        });
        let mut ops = FileOperations::new();
        seed_file_ops_from_details(&details, &mut ops);
        assert_eq!(
            ops.read,
            HashSet::from(["a.ts".to_string(), "b.md".to_string()])
        );
        assert_eq!(
            ops.edited,
            HashSet::from(["a.ts".to_string(), "c.rs".to_string()])
        );
        assert!(ops.written.is_empty());
    }

    #[test]
    fn test_seed_file_ops_from_details_skips_non_string_elements() {
        let details = serde_json::json!({"readFiles": [1, "x", null]});
        let mut ops = FileOperations::new();
        seed_file_ops_from_details(&details, &mut ops);
        // Defensive tightening: silently skip non-string elements
        assert_eq!(ops.read, HashSet::from(["x".to_string()]));
    }

    #[test]
    fn test_seed_file_ops_from_details_missing_keys_noop() {
        let mut ops = FileOperations::new();
        seed_file_ops_from_details(&serde_json::json!({}), &mut ops);
        assert!(ops.read.is_empty());
        assert!(ops.written.is_empty());
        assert!(ops.edited.is_empty());
    }

    // ── entries_to_messages_for_summarization (private, directly unit-tested)

    #[test]
    fn test_entries_to_messages_for_summarization_skips_compaction_and_metadata() {
        let entries = vec![
            user_entry("a", "u-a"),
            custom_entry("m1"),
            compaction_entry("c1", "prev", None, None, None),
            user_entry("b", "u-b"),
        ];
        // Window [1,4): neither Custom nor Compaction produces a message; only u-b remains
        let window = entries_to_messages_for_summarization(&entries, 1, 4);
        assert_eq!(texts(&window), vec!["u-b".to_string()]);
        // Window [0,1): only u-a
        let head = entries_to_messages_for_summarization(&entries, 0, 1);
        assert_eq!(texts(&head), vec!["u-a".to_string()]);
    }

    // ── FileOperations / CompactionError ────────────────────

    #[test]
    fn test_file_operations_new_starts_empty() {
        let ops = FileOperations::new();
        assert!(ops.read.is_empty());
        assert!(ops.written.is_empty());
        assert!(ops.edited.is_empty());
    }

    #[test]
    fn test_compaction_error_display() {
        assert_eq!(
            CompactionError::Aborted.to_string(),
            "summarization aborted"
        );
        assert_eq!(
            CompactionError::SummarizationFailed("boom".to_string()).to_string(),
            "summarization failed: boom"
        );
    }

    // compact / generate_summary depend on a real or fake StreamFn (LLM event stream collection),
    // Requires a test-double that constructs a complete AssistantMessage event sequence, beyond this file's unit-test scope—
    // Add branch coverage in zen-core integration tests or after introducing a fake StreamFn
    // (split-turn two-segment concatenation / combine_usage merging / file attachment section / reasoning_effort override).
}
