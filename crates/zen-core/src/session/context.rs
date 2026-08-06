use crate::session::tree::{SessionContext, SessionModel, SessionTreeEntry};
use zen_ai::types::{ContentBlockList, Message, UserMessage};

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
pub fn session_entry_to_context_messages(
    entry: &SessionTreeEntry,
    index: usize,
    entries: &[SessionTreeEntry],
) -> Vec<Message> {
    todo!()
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
