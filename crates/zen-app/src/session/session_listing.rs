// REQ-APP-11 (pi-358): session list data model — input view for session-selector / search pure functions.
// naming avoidance: zen-core `SessionTreeEntry::SessionInfo` is the "session naming entry"; this type contains search text,
// This is the session **list** view; named `SessionListing` to avoid ambiguity.
// construction logic (from SessionManager::list / open + JSONL reads) belongs to the caller (T-AP-08-5 / T-AP-11-1),
// this module only defines data shapes and pure data unrelated to display/search.

use std::path::PathBuf;

/// Session list item: input view for the session selector (session-selector) to display and search.
#[derive(Debug, Clone, PartialEq)]
///
/// T-AP-08-5 (--session startup selection) and T-AP-11-1 (/sessions navigation), before opening the selector,
/// Build this snapshot list from `SessionManager::list`/`open` and JSONL files; this type itself is pure data,
/// and holds no IO or mutable state.
///
/// Field semantics align with pi `SessionInfo` (core/session-manager.ts):
/// - `all_messages_text` is the concatenated full session text for the search index (id + name + full text + cwd participate in matching).
pub struct SessionListing {
    /// Session id (JSONL header).
    pub id: String,
    /// Absolute JSONL path; passed back to the upper layer when selected.
    pub path: PathBuf,
    /// Working directory at session creation.
    pub cwd: String,
    /// Session name (the name written in the SessionInfo entry).
    pub name: Option<String>,
    /// Fork source session path (if any).
    pub parent_session_path: Option<String>,
    /// Creation time (for list sorting and date display).
    pub created: chrono::DateTime<chrono::Utc>,
    /// Last modified time ("recent" sort key).
    pub modified: chrono::DateTime<chrono::Utc>,
    /// Number of message entries (reusable for list display).
    pub message_count: usize,
    /// Summary of the first user message (shown on the first line of the list).
    pub first_message: String,
    /// Search index text (concatenated full session text, participates in fuzzy/phrase/regex matching).
    pub all_messages_text: String,
}

/// Session list loading progress: `loaded`/`total` (used for the loading header "12/40" display).
///
/// Reported batch by batch by the loader callback; when `total` is unknown, 0 may be passed first (header displays `loaded`).
pub struct SessionListProgress {
    /// Number of loaded items.
    pub loaded: usize,
    /// Total number of items (0 when unknown).
    pub total: usize,
}

impl SessionListing {
    /// Construct a list item (fields correspond one-to-one; pure data assembly).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        path: PathBuf,
        cwd: String,
        name: Option<String>,
        parent_session_path: Option<String>,
        created: chrono::DateTime<chrono::Utc>,
        modified: chrono::DateTime<chrono::Utc>,
        message_count: usize,
        first_message: String,
        all_messages_text: String,
    ) -> Self {
        SessionListing {
            id,
            path,
            cwd,
            name,
            parent_session_path,
            created,
            modified,
            message_count,
            first_message,
            all_messages_text,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn listing(name: Option<&str>) -> SessionListing {
        SessionListing::new(
            "id1".into(),
            PathBuf::from("/tmp/s1.jsonl"),
            "/tmp".into(),
            name.map(|s| s.to_string()),
            None,
            chrono::Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap(),
            chrono::Utc.with_ymd_and_hms(2025, 1, 2, 0, 0, 0).unwrap(),
            3,
            "first".into(),
            "first hello world".into(),
        )
    }

    #[test]
    fn new_sets_all_fields() {
        let l = listing(Some("my session"));
        assert_eq!(l.id, "id1");
        assert_eq!(l.path, PathBuf::from("/tmp/s1.jsonl"));
        assert_eq!(l.cwd, "/tmp");
        assert_eq!(l.name.as_deref(), Some("my session"));
        assert!(l.parent_session_path.is_none());
        assert_eq!(l.message_count, 3);
        assert_eq!(l.first_message, "first");
        assert_eq!(l.all_messages_text, "first hello world");
        // Time fields are available
        assert!(l.modified > l.created);
    }

    #[test]
    fn new_without_name() {
        let l = listing(None);
        assert!(l.name.is_none());
    }

    #[test]
    fn progress_fields() {
        let p = SessionListProgress {
            loaded: 12,
            total: 40,
        };
        assert_eq!(p.loaded, 12);
        assert_eq!(p.total, 40);
    }
}
