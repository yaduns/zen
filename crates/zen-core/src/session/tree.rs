use serde::{Deserialize, Serialize};
use serde_json::Value;
use zen_ai::types::{ContentBlockList, Message, Usage};

/// Error types for session tree operations.
///
/// Defined at the `zen-core` layer (thiserror), for recoverable errors during tree operations.
/// Storage-layer errors (IO errors) are handled at the `zen-app` layer.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SessionTreeError {
    /// Entry not found
    #[error("entry not found: {0}")]
    EntryNotFound(String),

    /// Invalid fork target
    #[error("invalid fork target: {0}")]
    InvalidForkTarget(String),

    /// Cyclic reference (the parentId chain forms a cycle)
    #[error("circular reference detected")]
    CircularReference,
}

/// Session tree node type with 11 variants. Each node has a unique id and parent_id (pointing to the parent, None means root), plus an ISO 8601 timestamp.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionTreeEntry {
    /// Append user/assistant/tool-result messages to the session tree
    #[serde(rename_all = "camelCase")]
    Message {
        id: String,
        parent_id: Option<String>,
        timestamp: String,
        message: Message,
    },
    /// User changes thinking intensity (off/low/medium/high)
    #[serde(rename_all = "camelCase")]
    ThinkingLevelChange {
        id: String,
        parent_id: Option<String>,
        timestamp: String,
        thinking_level: String,
    },
    /// Record provider name and model ID when the user switches models, for context tracing
    #[serde(rename_all = "camelCase")]
    ModelChange {
        id: String,
        parent_id: Option<String>,
        timestamp: String,
        provider: String,
        model_id: String,
    },
    /// Record the list of active tool names when enabling/disabling toolsets
    #[serde(rename_all = "camelCase")]
    ActiveToolsChange {
        id: String,
        parent_id: Option<String>,
        timestamp: String,
        active_tool_names: Vec<String>,
    },
    /// On context compaction, generate a summary to replace historical messages
    #[serde(rename_all = "camelCase")]
    Compaction {
        id: String,
        parent_id: Option<String>,
        timestamp: String,
        summary: String,
        first_kept_entry_id: Option<String>,
        tokens_before: u64,
        retained_tail: Option<Vec<Message>>,
        details: Option<Value>,
        usage: Option<Usage>,
        from_hook: Option<bool>,
    },
    /// Generate a branch summary when forking a branch, to ease navigation of historical branches in the session tree
    #[serde(rename_all = "camelCase")]
    BranchSummary {
        id: String,
        parent_id: Option<String>,
        timestamp: String,
        from_id: String,
        summary: String,
        details: Option<Value>,
        usage: Option<Usage>,
        from_hook: Option<bool>,
    },
    /// Extended system data, for JSONL format compatibility only; this entry is never actually produced
    #[serde(rename_all = "camelCase")]
    Custom {
        id: String,
        parent_id: Option<String>,
        timestamp: String,
        custom_type: String,
        data: Option<Value>,
    },
    /// Custom message injected into the LLM context. Used for skill invocation feedback in Pi, and by the skill system in zen
    #[serde(rename_all = "camelCase")]
    CustomMessage {
        id: String,
        parent_id: Option<String>,
        timestamp: String,
        custom_type: String,
        content: ContentBlockList,
        display: bool,
        details: Option<Value>,
    },
    /// Tag/name an entry (label: None means removing the tag). Used by the session navigation UI, e.g. naming a checkpoint
    #[serde(rename_all = "camelCase")]
    Label {
        id: String,
        parent_id: Option<String>,
        timestamp: String,
        target_id: String,
        label: Option<String>,
    },
    /// Record the session name (e.g. set via the /name command). Used for session list display
    #[serde(rename_all = "camelCase")]
    SessionInfo {
        id: String,
        parent_id: Option<String>,
        timestamp: String,
        name: Option<String>,
    },
    /// JSONL persistence layer records the current leaf node position. The leaf_id field of the SessionTree struct already serves this role, but LeafEntry is kept in JSONL to support restoring the leaf position without scanning the whole tree
    #[serde(rename_all = "camelCase")]
    Leaf {
        id: String,
        parent_id: Option<String>,
        timestamp: String,
        target_id: Option<String>,
    },
}

/// Holds a set of SessionTreeEntry and tracks the current leaf node ID; provides pure tree navigation functions (no storage IO)
#[derive(Debug, Serialize, Deserialize)]
struct SessionTree {
    /// All entries, in append order
    entries: Vec<SessionTreeEntry>,
    /// Current leaf node ID; None means an empty tree
    leaf_id: Option<String>,
}

/// Return value of build_session_context, containing the built message list and the session state derived from path entries (thinking level, model, active tools)
pub struct SessionContext {
    /// The constructed context message list
    pub messages: Vec<Message>,
    /// Derived from the most recent thinking_level_change entry along the path
    pub thinking_level: String,
    /// Derived from the most recent model_change or message.role == "assistant" entry along the path
    pub model: Option<SessionModel>,
    /// Derived from the most recent active_tools_change entry along the path
    pub active_tool_names: Option<Vec<String>>,
}

/// Describes the provider and model ID used by the model; the internal type of the SessionContext.model field
pub struct SessionModel {
    /// Provider name (e.g. "deepseek")
    pub provider: String,
    /// Model identifier (e.g. "deepseek-v4-flash")
    pub model_id: String,
}

/// Session statistics, computed by SessionTree::get_stats()
struct SessionStats {
    /// Total number of message entries
    message_count: u64,
    /// Cumulative cache hit token count
    cached_tokens: u64,
    /// Cumulative cache miss token count
    uncached_tokens: u64,
    /// Total token count
    total_tokens: u64,
    /// Total cost
    cost_total: f64,
}

// Enumerate all variants, extracting id and parent_id
impl SessionTreeEntry {
    pub(crate) fn id(&self) -> &str {
        match self {
            SessionTreeEntry::Message { id, .. }
            | SessionTreeEntry::ThinkingLevelChange { id, .. }
            | SessionTreeEntry::ModelChange { id, .. }
            | SessionTreeEntry::ActiveToolsChange { id, .. }
            | SessionTreeEntry::Compaction { id, .. }
            | SessionTreeEntry::BranchSummary { id, .. }
            | SessionTreeEntry::Custom { id, .. }
            | SessionTreeEntry::CustomMessage { id, .. }
            | SessionTreeEntry::Label { id, .. }
            | SessionTreeEntry::SessionInfo { id, .. }
            | SessionTreeEntry::Leaf { id, .. } => id.as_str(),
        }
    }
    pub(crate) fn parent_id(&self) -> Option<&str> {
        match self {
            SessionTreeEntry::Message { parent_id, .. }
            | SessionTreeEntry::ThinkingLevelChange { parent_id, .. }
            | SessionTreeEntry::ModelChange { parent_id, .. }
            | SessionTreeEntry::ActiveToolsChange { parent_id, .. }
            | SessionTreeEntry::Compaction { parent_id, .. }
            | SessionTreeEntry::BranchSummary { parent_id, .. }
            | SessionTreeEntry::Custom { parent_id, .. }
            | SessionTreeEntry::CustomMessage { parent_id, .. }
            | SessionTreeEntry::Label { parent_id, .. }
            | SessionTreeEntry::SessionInfo { parent_id, .. }
            | SessionTreeEntry::Leaf { parent_id, .. } => parent_id.as_deref(),
        }
    }
}

/// Session tree: holds a set of SessionTreeEntry and provides pure tree navigation functions.
///
/// Involves no storage IO — all operations run on the in-memory entries slice.
/// The caller (zen-app's SessionManager) is responsible for persisting after modifications.
impl SessionTree {
    /// Walk from the given leaf (or the current leaf_id) up the parentId chain to the root.
    /// Returns an ordered slice of entries from root to leaf (root first, leaf last).
    ///
    /// If `leaf_id` is `None`, `self.leaf_id` is used.
    /// If leaf_id is not in entries, returns an empty slice.
    /// If the chain is broken (parent missing), stop at the break.
    pub fn get_path_to_root(&self, leaf_id: Option<&str>) -> Vec<&SessionTreeEntry> {
        let leaf_id = leaf_id.or(self.leaf_id.as_deref());
        let mut path = Vec::new();
        let mut current = leaf_id.and_then(|id| self.get_entry(id));
        while let Some(entry) = current {
            path.push(entry);
            current = entry.parent_id().and_then(|pid| self.get_entry(pid));
        }
        path.reverse();
        path
    }

    /// Traverse upward from `leaf_id` along the `parent_id` chain, truncating at a `Compaction` boundary.
    ///
    /// Specific logic:
    /// - Start at `leaf_id` and walk upward via `parent_id`
    /// - When a `Compaction` entry is encountered:
    ///   - If `compaction.retained_tail` exists → stop at that compaction (do not continue upward); the compaction itself stays in the path
    ///   - Otherwise set `stop_at_id = compaction.first_kept_entry_id` (or `None`) and continue upward until that ID
    /// - Return an ordered slice from the truncation point (or root) to the leaf
    ///
    /// Corresponds to Pi's `getPathToRootOrCompaction` semantics.
    pub fn get_path_to_root_or_compaction(&self, leaf_id: Option<&str>) -> Vec<&SessionTreeEntry> {
        let leaf_id = leaf_id.or(self.leaf_id.as_deref());
        let mut path = Vec::new();
        let mut stop_at_id: Option<&str> = None;
        let mut current = leaf_id.and_then(|id| self.get_entry(id));
        while let Some(entry) = current {
            path.push(entry);

            // Check whether we've reached stop_at_id (the firstKeptEntryId boundary)
            if let Some(sid) = stop_at_id {
                if entry.id() == sid {
                    break;
                }
            }

            // Boundary handling when a Compaction is encountered
            if let SessionTreeEntry::Compaction {
                retained_tail,
                first_kept_entry_id,
                ..
            } = entry
            {
                if retained_tail.is_some() {
                    // retainedTail present → stop at this compaction, don't continue upward
                    break;
                }
                // otherwise set firstKeptEntryId to the stop point and continue upward until that ID
                stop_at_id = first_kept_entry_id.as_deref();
                if stop_at_id.is_none() {
                    // neither present → stop at this compaction
                    break;
                }
                // continue traversing upward
            }

            current = entry.parent_id().and_then(|pid| self.get_entry(pid));
        }
        path.reverse();
        path
    }

    /// Look up an entry by ID.
    pub fn get_entry(&self, id: &str) -> Option<&SessionTreeEntry> {
        self.entries.iter().find(|entry| entry.id() == id)
    }

    /// Compute session statistics (message count, token totals, total cost).
    ///
    /// message_count — count all Message entries, regardless of role
    /// usage sources — three entry types: AssistantMessage (message.usage where entry is a Message with role == "assistant"), Compaction (entry.usage), BranchSummary (entry.usage)
    /// accumulation condition — only accumulate when usage is present and non-zero
    pub fn get_stats(&self) -> SessionStats {
        let mut message_count = 0u64;
        let mut cached_tokens = 0u64;
        let mut uncached_tokens = 0u64;
        let mut total_tokens = 0u64;
        let mut cost_total = 0.0f64;

        for entry in &self.entries {
            if matches!(entry, SessionTreeEntry::Message { .. }) {
                message_count += 1;
            }

            let usage = match entry {
                SessionTreeEntry::Message { message, .. } => match message {
                    Message::Assistant(msg) => Some(&msg.usage),
                    _ => None,
                },
                SessionTreeEntry::Compaction { usage, .. }
                | SessionTreeEntry::BranchSummary { usage, .. } => usage.as_ref(),
                _ => None,
            };

            let Some(usage) = usage else { continue };
            if usage.input == 0
                && usage.output == 0
                && usage.cache_read == 0
                && usage.cache_write == 0
            {
                continue;
            }

            cached_tokens += usage.cache_read;
            uncached_tokens += usage.input + usage.cache_write;
            total_tokens += usage.input + usage.output + usage.cache_read + usage.cache_write;
            cost_total += usage.cost.total;
        }

        SessionStats {
            message_count,
            cached_tokens,
            uncached_tokens,
            total_tokens,
            cost_total,
        }
    }
}
