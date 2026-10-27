// REQ-APP-11 (pi-358): session list data model — input view for session-selector / search pure functions.
// naming avoidance: zen-core `SessionTreeEntry::SessionInfo` is the "session naming entry"; this type contains search text,
// This is the session **list** view; named `SessionListing` to avoid ambiguity.
// T-AP-08-5: build logic (from SessionManager::list + JSONL reads → SessionListing snapshot)
// implemented in this file (build functions build_session_listings / build_all_session_listings),
// consumed by preloading in the startup selector (cli/session_picker.rs) and interactive selector (interactive /resume).

use crate::session::jsonl::{list_sessions, read_session_file};
use std::path::{Path, PathBuf};
use zen_ai::types::Message;
use zen_ai::utils::text::{content_text, content_text_from_list};
use zen_core::session::tree::SessionTreeEntry;

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

/// Before the startup/interactive session selector opens, organize the JSONL sessions in the current cwd into a list snapshot—
/// "The --resume selector and /resume panel must list which sessions in the current project are resumable."
/// For metadata from `jsonl::list_sessions(sessions_root, Some(cwd))`, read the full content per entry to derive
/// first message / full-text concatenation / message count / naming; invalid sessions are silently skipped (matching list_sessions fault tolerance).
/// Returns a list sorted by **most recently modified descending** (the threaded default view and recent sorting both use this as the base order,
/// matching pi buildSessionTree's modified-descending root sort); an empty list means no selectable sessions.
/// `sessions_root` is the session root directory (the unified entry point for the CLI `--session-dir` override and the default `get_sessions_dir()`,
/// closing the T-AP-02-3 session_dir seam).
pub fn build_session_listings(cwd: &str, sessions_root: &Path) -> Vec<SessionListing> {
    let Ok(metas) = list_sessions(sessions_root, Some(cwd)) else {
        return Vec::new();
    };
    build_listings_from_metas(metas)
}

/// Isomorphic to `build_session_listings`, but across all project directories—
/// "When the selector switches to All scope, list resumable sessions from all projects."
/// `list_sessions(sessions_root, None)` for all + per-entry derivation; invalid sessions are silently skipped.
pub fn build_all_session_listings(sessions_root: &Path) -> Vec<SessionListing> {
    let Ok(metas) = list_sessions(sessions_root, None) else {
        return Vec::new();
    };
    build_listings_from_metas(metas)
}

/// Metadata list → list snapshot (derive by reading each file; skip corrupted), sorted by modified descending.
fn build_listings_from_metas(
    metas: Vec<crate::session::jsonl::JsonlSessionMetadata>,
) -> Vec<SessionListing> {
    let mut out: Vec<SessionListing> = metas
        .iter()
        .filter_map(|m| session_listing_from_file(&m.path))
        .collect();
    out.sort_by(|a, b| b.modified.cmp(&a.modified));
    out
}

/// Single JSONL session file → list item; returns None on corrupted file/parse failure (silently skipped).
/// Derive from entries: SessionInfo entries provide the name; Message entries accumulate first_message/
/// all_messages_text/message_count (matching the display-field semantics of pi buildSessionInfo);
/// modified = last message activity time; if absent, fall back to header timestamp, then file mtime.
fn session_listing_from_file(path: &Path) -> Option<SessionListing> {
    let (header, entries, _leaf) = read_session_file(path).ok()?;
    let created = chrono::DateTime::parse_from_rfc3339(&header.timestamp)
        .ok()?
        .with_timezone(&chrono::Utc);

    let mut name: Option<String> = None;
    let mut message_count = 0usize;
    let mut first_message = String::new();
    let mut all_messages: Vec<String> = Vec::new();
    let mut last_activity_ms: Option<i64> = None;

    for entry in &entries {
        match entry {
            // Session naming: take the latest session_info (including explicit clear, matching pi `name = entry.name?.trim() || undefined`)
            SessionTreeEntry::SessionInfo { name: n, .. } => {
                let trimmed = n.as_deref().unwrap_or("").trim();
                name = if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                };
            }
            SessionTreeEntry::Message { message, .. } => {
                message_count += 1;
                let (ts, text) = message_text_and_ts(message);
                if let Some(ts) = ts {
                    last_activity_ms = Some(last_activity_ms.map_or(ts, |cur| cur.max(ts)));
                }
                if let Some(text) = text {
                    if !text.is_empty() {
                        all_messages.push(text.clone());
                        if first_message.is_empty() && matches!(message, Message::User(_)) {
                            first_message = text;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let first_message = if first_message.is_empty() {
        "(no messages)".to_string()
    } else {
        first_message
    };
    let all_messages_text = all_messages.join(" ");
    let modified = match last_activity_ms {
        Some(ms) => chrono::DateTime::from_timestamp_millis(ms).unwrap_or(created),
        None => std::fs::metadata(path)
            .and_then(|m| m.modified().map(chrono::DateTime::from))
            .unwrap_or(created),
    };

    Some(SessionListing::new(
        header.id.clone(),
        path.to_path_buf(),
        header.cwd.clone(),
        name,
        header.parent_session.clone(),
        created,
        modified,
        message_count,
        first_message,
        all_messages_text,
    ))
}

/// Message entry → (Unix millisecond timestamp, user/assistant text content); tool results are not included in list text.
fn message_text_and_ts(message: &Message) -> (Option<i64>, Option<String>) {
    match message {
        Message::User(u) => (
            Some(u.timestamp),
            Some(content_text_from_list(&u.content, "")),
        ),
        Message::Assistant(a) => (Some(a.timestamp), Some(content_text(&a.content, ""))),
        Message::ToolResult(_) => (None, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use zen_ai::types::ContentBlockList;

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

    // ── T-AP-08-5: build functions (real filesystem, tempdir follows session_manager precedent) ──────

    /// Create a unique test directory under the system temp directory (std-only).
    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "zen_session_listing_{}_{}",
            name,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn user_message(text: &str) -> Message {
        Message::User(zen_ai::types::UserMessage {
            content: ContentBlockList::Text(text.to_string()),
            // Recent timestamp (milliseconds) +1min: header ts has RFC3339 nanosecond precision, and message ts within the same millisecond
            // may be earlier than the header write time (ms truncation race) — the offset ensures modified >= created
            timestamp: chrono::Utc::now().timestamp_millis() + 60_000,
        })
    }

    fn create_session(
        root: &Path,
        cwd: &str,
        id: &str,
        name: Option<&str>,
        texts: &[&str],
    ) -> PathBuf {
        use crate::session::jsonl::{append_entry, create_session_file, CreateSessionOptions};
        use zen_core::session::tree::SessionTreeEntry;
        let file = create_session_file(
            root,
            cwd,
            CreateSessionOptions {
                session_id: Some(id.to_string()),
                parent_session_path: None,
                metadata: None,
            },
        )
        .expect("create session file");
        let mut leaf: Option<String> = None;
        for t in texts {
            let entry = SessionTreeEntry::Message {
                id: format!("e{}", t.len()),
                parent_id: leaf.clone(),
                timestamp: "2025-01-01T00:00:00Z".to_string(),
                message: user_message(t),
            };
            leaf = Some(entry.id().to_string());
            append_entry(&file, &entry).unwrap();
        }
        if let Some(n) = name {
            let entry = SessionTreeEntry::SessionInfo {
                id: "sess-info-1".to_string(),
                parent_id: leaf,
                timestamp: "2025-01-01T00:00:00Z".to_string(),
                name: Some(n.to_string()),
            };
            append_entry(&file, &entry).unwrap();
        }
        file
    }

    #[test]
    fn build_session_listings_empty_dir_returns_empty() {
        let root = test_dir("empty");
        let out = build_session_listings("/proj", &root);
        assert!(out.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn build_session_listings_derives_fields_and_filters_by_cwd() {
        let root = test_dir("derive");
        create_session(
            &root,
            "/proj",
            "s1",
            Some("  my session  "),
            &["first msg", "second msg"],
        );
        create_session(&root, "/other", "s2", None, &["other msg"]);

        let out = build_session_listings("/proj", &root);
        assert_eq!(out.len(), 1, "list only current cwd sessions");
        let l = &out[0];
        assert_eq!(l.id, "s1");
        assert_eq!(l.cwd, "/proj");
        // name trim + empty string → None
        assert_eq!(l.name.as_deref(), Some("my session"));
        assert_eq!(l.message_count, 2);
        assert_eq!(l.first_message, "first msg");
        assert_eq!(l.all_messages_text, "first msg second msg");
        // time fields available and modified >= created
        assert!(l.modified >= l.created);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn build_all_session_listings_across_cwds_sorted_by_modified_desc() {
        let root = test_dir("all");
        create_session(&root, "/projA", "s1", None, &["a"]);
        // Append a later-timestamp message to s1 (modified increases) → it should sort first
        use crate::session::jsonl::append_entry;
        use zen_core::session::tree::SessionTreeEntry;
        let dir_a = root.join(crate::session::jsonl::encode_cwd("/projA"));
        // Filename looks like {timestamp}-s1.jsonl: scan the directory to find s1's actual file
        let s1_path = std::fs::read_dir(&dir_a)
            .unwrap()
            .find_map(|e| {
                let p = e.unwrap().path();
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.ends_with("s1.jsonl"))
                    .then_some(p)
            })
            .expect("s1 session file");
        append_entry(
            &s1_path,
            &SessionTreeEntry::Message {
                id: "late".to_string(),
                parent_id: None,
                timestamp: "2025-01-01T00:00:00Z".to_string(),
                message: Message::User(zen_ai::types::UserMessage {
                    content: ContentBlockList::Text("later".to_string()),
                    timestamp: chrono::Utc::now().timestamp_millis() + 120_000,
                }),
            },
        )
        .unwrap();
        create_session(&root, "/projB", "s2", None, &["b"]);

        let out = build_all_session_listings(&root);
        assert_eq!(out.len(), 2);
        assert_eq!(
            out[0].cwd, "/projA",
            "s1 later activity timestamp should rank first"
        );
        assert_eq!(out[1].cwd, "/projB");
        assert!(out[0].modified >= out[1].modified);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn build_session_listings_no_messages_uses_no_messages_placeholder() {
        let root = test_dir("nomessages");
        create_session(&root, "/proj", "s1", None, &[]);
        let out = build_session_listings("/proj", &root);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].first_message, "(no messages)");
        assert_eq!(out[0].message_count, 0);
        assert!(out[0].all_messages_text.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn build_session_listings_skips_corrupt_files() {
        let root = test_dir("corrupt");
        create_session(&root, "/proj", "good", None, &["hi"]);
        // Write a bad file directly (first line is not a session header)
        let dir = root.join(crate::session::jsonl::encode_cwd("/proj"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("bad.jsonl"), "not a session\n").unwrap();

        let out = build_session_listings("/proj", &root);
        assert_eq!(out.len(), 1, "bad files silently skipped");
        assert_eq!(out[0].id, "good");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn build_session_listings_latest_session_info_name_wins() {
        let root = test_dir("namewins");
        use crate::session::jsonl::append_entry;
        use zen_core::session::tree::SessionTreeEntry;
        let file = create_session(&root, "/proj", "s1", Some("first name"), &["hi"]);
        // Append a session_info overriding the name (clear semantics: empty name → None)
        append_entry(
            &file,
            &SessionTreeEntry::SessionInfo {
                id: "sess-info-2".to_string(),
                parent_id: None,
                timestamp: "2025-01-02T00:00:00Z".to_string(),
                name: None,
            },
        )
        .unwrap();

        let out = build_session_listings("/proj", &root);
        assert_eq!(out.len(), 1);
        assert!(
            out[0].name.is_none(),
            "latest session_info None overwrites old name"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
