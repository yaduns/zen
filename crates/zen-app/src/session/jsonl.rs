use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use zen_ai::types::Message;
use zen_core::session::tree::SessionTreeEntry;

pub const CURRENT_SESSION_VERSION: u64 = 3;

/// First line of the JSONL file, identifying session metadata. Contains session ID, creation timestamp, working directory, parent session path (for fork tracking), and arbitrary metadata
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionHeader {
    /// Fixed to "session", used for format discrimination
    #[serde(rename = "type")]
    pub(crate) ty: String,
    /// Fixed to 3, used for version compatibility checking
    pub(crate) version: u8,
    /// Session ID (UUIDv7)
    pub(crate) id: String,
    /// ISO 8601 timestamp
    pub(crate) timestamp: String,
    /// Working directory at session creation
    pub(crate) cwd: String,
    /// Path of the forked source session, used for fork tracking
    pub(crate) parent_session: Option<String>,
    /// Arbitrary metadata (e.g. session name, tags, etc.)
    pub(crate) metadata: Option<serde_json::Value>,
}

/// Metadata summary of the session list, read from SessionHeader.
/// Used as the return value of SessionManager::list_sessions(); not written directly to the JSONL file (constructed after reading from the header).
pub struct JsonlSessionMetadata {
    /// Session ID
    pub(crate) id: String,
    /// ISO 8601 creation timestamp
    pub(crate) created_at: String,
    /// Working directory
    pub(crate) cwd: String,
    /// Absolute path to the JSONL file
    pub(crate) path: PathBuf,
    /// Path of the fork source
    pub(crate) parent_session_path: Option<String>,
    /// Header metadata
    pub(crate) metadata: Option<serde_json::Value>,
}

impl JsonlSessionMetadata {
    /// Session ID (read by the bin target via an accessor; the field is pub(crate) and invisible to bin).
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Absolute path of the JSONL file (read by the bin target via an accessor).
    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

/// Error types that can occur during JSONL persistence operations. Derived via thiserror, covering file I/O errors, format errors, entry parsing errors, etc.
#[derive(Debug, thiserror::Error)]
pub enum JsonlError {
    #[error("I/O error: {0}")]
    Io(std::io::Error),
    #[error("invalid session metadata: {0}")]
    InvalidSession(String),
    #[error("invalid entry at line {0}: {1}")]
    InvalidEntry(usize, String),
    #[error("entry not found: {0}")]
    EntryNotFound(String),
    #[error("invalid fork target: {0}")]
    ForkTargetInvalid(String),
}

/// Position where entries are included when forking.
///
/// Exposed to callers (SessionManager / integration tests) so `get_entries_to_fork` can specify the branch split point.
pub enum ForkPosition {
    /// Before the target entry (excluding it; requires the target entry to be a UserMessage)
    Before,
    /// Starting from the target entry (inclusive)
    At,
}

// ─── Utility Functions ────────────────────────────────────────────────────

/// Creates a session ID (short ID from UUIDv7, last 8 chars).
///
/// Corresponds to Pi's `createSessionId` function. In Pi, the UUIDv7 prefix is timestamp-derived and the short ID is taken from the random tail.
/// If the short ID collides (extremely unlikely), retries up to 100 times; if still colliding, returns the full UUIDv7.
pub fn create_session_id(existing_ids: Option<&std::collections::HashSet<String>>) -> String {
    for _ in 0..100 {
        let full = uuid::Uuid::now_v7().to_string();
        // take last 8 chars
        let short = full[full.len() - 8..].to_string();
        // check for collision with existing IDs
        if existing_ids.map_or(true, |ids| !ids.contains(&short)) {
            return short;
        }
    }
    // if colliding 100 times, return the full 36-char string
    uuid::Uuid::now_v7().to_string()
}

/// Creates an ISO 8601 timestamp string.
///
/// Format: `2025-01-15T10:30:00.000Z` (millisecond precision, UTC).
pub fn create_timestamp() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// Encodes a working directory into a filesystem-safe directory name.
///
/// Conversion rule: `/home/user/project` → `--home-user-project--`
/// Strip the leading `/` or `\`, replace path separators and `:` with `-`.
pub fn encode_cwd(cwd: &str) -> String {
    let inner = cwd
        .trim_start_matches(|c| c == '/' || c == '\\')
        .replace('/', "-")
        .replace('\\', "-")
        .replace(':', "-");
    format!("--{}--", inner)
}

/// Create the session JSONL file path.
///
/// Format: `{sessions_root}/{encoded_cwd}/{timestamp}-{session_id}.jsonl`
/// Replace `:` and `.` in the timestamp with `-` (filesystem-safe).
pub fn create_session_file_path(
    sessions_root: &std::path::Path,
    cwd: &str,
    session_id: &str,
    timestamp: &str,
) -> std::path::PathBuf {
    let safe_ts = timestamp.replace(':', "-").replace('.', "-");
    sessions_root
        .join(encode_cwd(cwd))
        .join(format!("{}-{}.jsonl", safe_ts, session_id))
}

// ─── Format encoding/decoding ────────────────────────────────────────────────────

/// Parse the session header from the first line of a JSONL file.
///
/// Validation rules:
/// - must be a valid JSON object
/// - the `type` field must be `"session"`
/// - `version` must be `3`
/// - `id`, `timestamp`, and `cwd` must be non-empty strings
/// - `parentSession` and `metadata` are optional
///
/// Returns `JsonlError::InvalidSession` when validation fails.
pub fn parse_header_line(
    line: &str,
    file_path: &std::path::Path,
) -> Result<SessionHeader, JsonlError> {
    let value: serde_json::Value =
        serde_json::from_str(line).map_err(|e| JsonlError::InvalidSession(e.to_string()))?;

    // validate type
    match value.get("type").and_then(|v| v.as_str()) {
        Some("session") => {}
        _ => {
            return Err(JsonlError::InvalidSession(format!(
                "invalid type in {}: expected 'session'",
                file_path.display()
            )));
        }
    }

    // validate version
    match value.get("version").and_then(|v| v.as_u64()) {
        Some(CURRENT_SESSION_VERSION) => {}
        _ => {
            return Err(JsonlError::InvalidSession(format!(
                "invalid version in {}: expected 3",
                file_path.display()
            )));
        }
    }

    // validate that required fields are non-empty
    for field in &["id", "timestamp", "cwd"] {
        match value.get(field).and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => {}
            _ => {
                return Err(JsonlError::InvalidSession(format!(
                    "missing or empty '{}' in {}",
                    field,
                    file_path.display()
                )));
            }
        }
    }

    // deserialize into SessionHeader
    let header: SessionHeader =
        serde_json::from_value(value).map_err(|e| JsonlError::InvalidSession(e.to_string()))?;

    Ok(header)
}

/// Parse a session entry from a JSONL file line.
///
/// Validation rules:
/// - must be a valid JSON object
/// - the `type` field must be a string
/// - `id` must be a non-empty string
/// - `parentId` may be null/absent/a string
/// - `timestamp` must be a non-empty string
///
/// Returns `JsonlError::InvalidEntry` when validation fails.
pub fn parse_entry_line(
    line: &str,
    line_number: usize,
    file_path: &std::path::Path,
) -> Result<SessionTreeEntry, JsonlError> {
    let value: serde_json::Value = serde_json::from_str(line)
        .map_err(|e| JsonlError::InvalidEntry(line_number, format!("invalid JSON: {}", e)))?;

    // Validate that type is a string
    match value.get("type").and_then(|v| v.as_str()) {
        Some(_) => {}
        None => {
            return Err(JsonlError::InvalidEntry(
                line_number,
                format!("missing or invalid 'type' in {}", file_path.display()),
            ));
        }
    }

    // Validate that id is a non-empty string
    match value.get("id").and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => {}
        _ => {
            return Err(JsonlError::InvalidEntry(
                line_number,
                format!("missing or empty 'id' in {}", file_path.display()),
            ));
        }
    }

    // Validate that timestamp is a non-empty string
    match value.get("timestamp").and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => {}
        _ => {
            return Err(JsonlError::InvalidEntry(
                line_number,
                format!("missing or empty 'timestamp' in {}", file_path.display()),
            ));
        }
    }

    // parentId is optional, no validation needed

    // Deserialize into SessionTreeEntry
    let entry: SessionTreeEntry = serde_json::from_value(value).map_err(|e| {
        JsonlError::InvalidEntry(line_number, format!("deserialization failed: {}", e))
    })?;

    Ok(entry)
}

/// Serialize `SessionHeader` into a JSONL line (with a trailing `\n`).
pub fn serialize_header(header: &SessionHeader) -> String {
    let mut line =
        serde_json::to_string(header).expect("SessionHeader serialization should not fail");
    line.push('\n');
    line
}

/// Serialize `SessionTreeEntry` into a JSONL line (with a trailing `\n`).
pub fn serialize_entry(entry: &SessionTreeEntry) -> String {
    let mut line =
        serde_json::to_string(entry).expect("SessionTreeEntry serialization should not fail");
    line.push('\n');
    line
}

/// Constructs `JsonlSessionMetadata` from a `SessionHeader`.
pub fn header_to_metadata(
    header: &SessionHeader,
    path: std::path::PathBuf,
) -> JsonlSessionMetadata {
    JsonlSessionMetadata {
        id: header.id.clone(),
        created_at: header.timestamp.clone(),
        cwd: header.cwd.clone(),
        path,
        parent_session_path: header.parent_session.clone(),
        metadata: header.metadata.clone(),
    }
}

// ─── File I/O operations ────────────────────────────────────────────────────

/// Creates a new session JSONL file.
///
/// Steps:
/// 1. Ensure the session directory exists (`create_dir_all`)
/// 2. Build the `SessionHeader` and write it as the first line of the file
///
/// Returns the file path.
/// Corresponds to Pi's `JsonlSessionRepo.create` method (simplified, without the `Session` wrapper).
pub fn create_session_file(
    sessions_root: &std::path::Path,
    cwd: &str,
    options: CreateSessionOptions,
) -> Result<std::path::PathBuf, JsonlError> {
    let encoded = encode_cwd(cwd);
    let session_dir = sessions_root.join(&encoded);

    // Scan existing session IDs for collision check
    let existing_ids: std::collections::HashSet<String> = if session_dir.exists() {
        std::fs::read_dir(&session_dir)
            .map_err(JsonlError::Io)?
            .filter_map(|entry| entry.ok())
            .filter_map(|e| {
                e.file_name()
                    .to_str()
                    .and_then(|name| name.strip_suffix(".jsonl"))
                    .and_then(|stem| stem.rsplit('-').next())
                    .map(|id| id.to_string())
            })
            .collect()
    } else {
        std::collections::HashSet::new()
    };

    let session_id = options
        .session_id
        .unwrap_or_else(|| create_session_id(Some(&existing_ids)));
    let timestamp = create_timestamp();
    let file_path = session_dir.join(format!(
        "{}-{}.jsonl",
        timestamp.replace(':', "-").replace('.', "-"),
        session_id
    ));

    // Ensure the directory exists
    std::fs::create_dir_all(&session_dir).map_err(JsonlError::Io)?;

    let header = SessionHeader {
        ty: "session".to_string(),
        version: CURRENT_SESSION_VERSION as u8,
        id: session_id,
        timestamp,
        cwd: cwd.to_string(),
        parent_session: options.parent_session_path,
        metadata: options.metadata,
    };

    let line = serialize_header(&header);
    std::fs::write(&file_path, line).map_err(JsonlError::Io)?;

    Ok(file_path)
}

/// Opens an existing session JSONL file and reads all entries.
///
/// Steps:
/// 1. Read the entire file contents
/// 2. Split by lines and filter out empty lines
/// 3. Parse the first line as `SessionHeader`
/// 4. Parse subsequent lines as `SessionTreeEntry`
/// 5. Return `(header, entries, last_leaf_id)`
///
/// `last_leaf_id` is derived from the `target_id` of the last `Leaf` entry or the `id` of the last non-Leaf entry.
/// Returns `JsonlError::Io` if the file does not exist.
/// Corresponds to Pi's `loadJsonlStorage` function.
pub fn read_session_file(
    file_path: &std::path::Path,
) -> Result<(SessionHeader, Vec<SessionTreeEntry>, Option<String>), JsonlError> {
    let content = std::fs::read_to_string(file_path).map_err(JsonlError::Io)?;
    let mut lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();

    let header_line = lines.first().ok_or_else(|| {
        JsonlError::InvalidSession(format!("empty file: {}", file_path.display()))
    })?;
    let header = parse_header_line(header_line, file_path)?;

    let entries: Vec<SessionTreeEntry> = lines[1..]
        .iter()
        .enumerate()
        .map(|(i, line)| parse_entry_line(line, i + 2, file_path))
        .collect::<Result<Vec<_>, _>>()?;

    // Derive last_leaf_id
    let last_leaf_id = entries
        .iter()
        .rev()
        .find_map(|e| {
            if let SessionTreeEntry::Leaf { target_id, .. } = e {
                target_id.clone()
            } else {
                None
            }
        })
        .or_else(|| {
            entries.iter().rev().find_map(|e| {
                if !matches!(e, SessionTreeEntry::Leaf { .. }) {
                    Some(e.id().to_string())
                } else {
                    None
                }
            })
        });

    Ok((header, entries, last_leaf_id))
}

/// Read the first line of a JSONL file (header only, for fast session list display).
///
/// Steps:
/// 1. Read the first 4096 bytes of the file (enough to cover the header line)
/// 2. Extract the first line
/// 3. Parse it as `SessionHeader`
/// 4. Return `JsonlSessionMetadata`
///
/// Returns `JsonlError::InvalidSession` when the file is empty or the first line is invalid.
pub fn read_session_metadata(
    file_path: &std::path::Path,
) -> Result<JsonlSessionMetadata, JsonlError> {
    let file = std::fs::File::open(file_path).map_err(JsonlError::Io)?;
    let mut reader = BufReader::new(file);
    let mut first_line = String::new();
    reader.read_line(&mut first_line).map_err(JsonlError::Io)?;

    if first_line.trim().is_empty() {
        return Err(JsonlError::InvalidSession(format!(
            "empty file: {}",
            file_path.display()
        )));
    }

    let header = parse_header_line(first_line.trim(), file_path)?;
    Ok(header_to_metadata(&header, file_path.to_path_buf()))
}

/// Append an entry to an existing JSONL file.
///
/// Steps:
/// 1. Serialize the entry to a JSON line
/// 2. Open the file in append mode
/// 3. Write the JSON line + newline
///
/// Returns `JsonlError::Io` if the file does not exist.
pub fn append_entry(
    file_path: &std::path::Path,
    entry: &SessionTreeEntry,
) -> Result<(), JsonlError> {
    let line = serialize_entry(entry);
    // If the file does not exist, naturally returns std::io::Error (NotFound)
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(file_path)
        .map_err(JsonlError::Io)?;
    file.write_all(line.as_bytes()).map_err(JsonlError::Io)?;
    Ok(())
}

/// List all JSONL files in the sessions directory and read their metadata.
///
/// Steps:
/// 1. List all `.jsonl` files under the `{sessions_root}/{encoded_cwd}` directory
/// 2. Call `read_session_metadata` for each file
/// 3. Sort by `created_at` descending
/// 4. Skip invalid session files (silently ignored)
///
/// If `cwd` is specified, only list sessions in that directory; otherwise list all subdirectories under `{sessions_root}/`.
/// If the directory does not exist, return an empty list.
pub fn list_sessions(
    sessions_root: &std::path::Path,
    cwd: Option<&str>,
) -> Result<Vec<JsonlSessionMetadata>, JsonlError> {
    let dirs: Vec<std::path::PathBuf> = match cwd {
        Some(cwd) => {
            let dir = sessions_root.join(encode_cwd(cwd));
            if dir.exists() {
                vec![dir]
            } else {
                return Ok(vec![]);
            }
        }
        None => list_session_dirs(sessions_root)?,
    };

    let mut sessions = Vec::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            if let Ok(metadata) = read_session_metadata(&path) {
                sessions.push(metadata);
            }
            // Silently ignore invalid files
        }
    }

    // Sort by created_at descending
    sessions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(sessions)
}

/// Delete the session JSONL file.
///
/// Silently ignore if the file does not exist (`force` semantics of `remove_file`).
pub fn delete_session(file_path: &std::path::Path) -> Result<(), JsonlError> {
    match std::fs::remove_file(file_path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(JsonlError::Io(e)),
    }
}

/// Compute the list of entries to include when forking.
///
/// If no `entry_id`, return all entries.
/// If `position` is `"at"`, start from that entry (inclusive) up to the root path.
/// If `position` is `"before"` or the default, start from the previous sibling of that entry (exclusive); the target entry must be a `UserMessage`.
///
/// Corresponds to Pi's `getEntriesToFork` function.
pub fn get_entries_to_fork(
    entries: &[SessionTreeEntry],
    entry_id: Option<&str>,
    position: ForkPosition,
) -> Result<Vec<SessionTreeEntry>, JsonlError> {
    let Some(entry_id) = entry_id else {
        return Ok(entries.to_vec());
    };

    // Find the target entry
    let target = entries
        .iter()
        .find(|e| e.id() == entry_id)
        .ok_or_else(|| JsonlError::ForkTargetInvalid(entry_id.to_string()))?;

    // Compute valid leaves
    let effective_leaf_id: Option<&str> = match position {
        ForkPosition::At => Some(target.id()),
        ForkPosition::Before => {
            // The target entry must be a UserMessage
            if !matches!(
                target,
                SessionTreeEntry::Message {
                    message: Message::User(_),
                    ..
                }
            ) {
                return Err(JsonlError::ForkTargetInvalid(format!(
                    "entry {} is not a user message",
                    entry_id
                )));
            }
            target.parent_id()
        }
    };

    // Walk up the parent chain to the root
    let mut path = Vec::new();
    let mut current = effective_leaf_id.and_then(|id| entries.iter().find(|e| e.id() == id));
    while let Some(entry) = current {
        path.push(entry);
        current = entry
            .parent_id()
            .and_then(|pid| entries.iter().find(|e| e.id() == pid));
    }
    path.reverse();

    Ok(path.into_iter().cloned().collect())
}

// ─── Helper types ─────────────────────────────────────────────────

/// Options for creating a session.
pub struct CreateSessionOptions {
    /// Session ID (optional; auto-generated if not specified)
    pub session_id: Option<String>,
    /// Parent session path (optional, used for fork)
    pub parent_session_path: Option<String>,
    /// Arbitrary metadata (optional)
    pub metadata: Option<serde_json::Value>,
}

// ─── Directory enumeration ─────────────────────────────────────────────────

/// List all session directories (non-hidden subdirectories under `{sessions_root}/`).
///
/// Corresponds to Pi's private method `JsonlSessionRepo.listSessionDirs`.
pub fn list_session_dirs(
    sessions_root: &std::path::Path,
) -> Result<Vec<std::path::PathBuf>, JsonlError> {
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(sessions_root).map_err(JsonlError::Io)? {
        let entry = entry.map_err(JsonlError::Io)?;
        let path = entry.path();
        if path.is_dir() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if !name.starts_with('.') {
                    dirs.push(path);
                }
            }
        }
    }
    dirs.sort();
    Ok(dirs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use zen_ai::types::{ContentBlockList, UserMessage};

    // ─── Test helpers ───────────────────────────────────────────────

    /// Creates a unique test directory under the system temp directory (std-only, to avoid introducing tempfile).
    /// Removes the old directory on each call to ensure a clean state.
    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("zen_jsonl_{}_{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn user_msg_entry(id: &str, parent: Option<&str>) -> SessionTreeEntry {
        SessionTreeEntry::Message {
            id: id.to_string(),
            parent_id: parent.map(|s| s.to_string()),
            timestamp: "2025-01-15T10:30:00.000Z".to_string(),
            message: Message::User(UserMessage {
                content: ContentBlockList::Text("hello".to_string()),
                timestamp: 0,
            }),
        }
    }

    fn leaf_entry(id: &str, parent: Option<&str>, target: Option<&str>) -> SessionTreeEntry {
        SessionTreeEntry::Leaf {
            id: id.to_string(),
            parent_id: parent.map(|s| s.to_string()),
            timestamp: "2025-01-15T10:30:00.000Z".to_string(),
            target_id: target.map(|s| s.to_string()),
        }
    }

    // ─── encode_cwd ────────────────────────────────────────────────────

    #[test]
    fn test_encode_cwd_posix() {
        assert_eq!(encode_cwd("/home/user/project"), "--home-user-project--");
    }

    #[test]
    fn test_encode_cwd_leading_backslash() {
        assert_eq!(encode_cwd("\\home\\user"), "--home-user--");
    }

    #[test]
    fn test_encode_cwd_replaces_colon() {
        assert_eq!(encode_cwd("C:/Users/me"), "--C--Users-me--");
    }

    #[test]
    fn test_encode_cwd_empty() {
        assert_eq!(encode_cwd(""), "----");
    }

    // ─── create_session_file_path ─────────────────────────────────────

    #[test]
    fn test_create_session_file_path_encodes_and_sanitizes() {
        let p = create_session_file_path(
            Path::new("/root"),
            "/home/user",
            "abc123",
            "2025-01-15T10:30:00.000Z",
        );
        assert_eq!(
            p,
            PathBuf::from("/root/--home-user--/2025-01-15T10-30-00-000Z-abc123.jsonl")
        );
    }

    // ─── create_session_id ─────────────────────────────────────────────

    #[test]
    fn test_create_session_id_len_is_8() {
        assert_eq!(create_session_id(None).len(), 8);
    }

    #[test]
    fn test_create_session_id_avoids_existing() {
        let existing: std::collections::HashSet<String> =
            std::collections::HashSet::from([create_session_id(None)]);
        let new_id = create_session_id(Some(&existing));
        assert_eq!(new_id.len(), 8);
        assert!(!existing.contains(&new_id));
    }

    // ─── serialize / parse_header_line ────────────────────────────────

    #[test]
    fn test_serialize_header_roundtrip() {
        let header = SessionHeader {
            ty: "session".to_string(),
            version: CURRENT_SESSION_VERSION as u8,
            id: "s1".to_string(),
            timestamp: "2025-01-15T10:30:00.000Z".to_string(),
            cwd: "/home/user".to_string(),
            parent_session: None,
            metadata: None,
        };
        let line = serialize_header(&header);
        assert!(line.ends_with('\n'));
        let parsed = parse_header_line(line.trim(), Path::new("/tmp/s.jsonl")).unwrap();
        assert_eq!(parsed.id, "s1");
        assert_eq!(parsed.version, 3);
        assert_eq!(parsed.cwd, "/home/user");
    }

    #[test]
    fn test_parse_header_line_wrong_type() {
        let line = r#"{"type":"not_session","version":3,"id":"s1","timestamp":"t","cwd":"/"}"#;
        assert!(matches!(
            parse_header_line(line, Path::new("/tmp/s.jsonl")),
            Err(JsonlError::InvalidSession(_))
        ));
    }

    #[test]
    fn test_parse_header_line_wrong_version() {
        let line = r#"{"type":"session","version":4,"id":"s1","timestamp":"t","cwd":"/"}"#;
        assert!(matches!(
            parse_header_line(line, Path::new("/tmp/s.jsonl")),
            Err(JsonlError::InvalidSession(_))
        ));
    }

    #[test]
    fn test_parse_header_line_missing_field() {
        let line = r#"{"type":"session","version":3,"timestamp":"t","cwd":"/"}"#;
        assert!(matches!(
            parse_header_line(line, Path::new("/tmp/s.jsonl")),
            Err(JsonlError::InvalidSession(_))
        ));
    }

    #[test]
    fn test_parse_header_line_invalid_json() {
        assert!(matches!(
            parse_header_line("not json", Path::new("/tmp/s.jsonl")),
            Err(JsonlError::InvalidSession(_))
        ));
    }

    // ─── serialize / parse_entry_line ─────────────────────────────────

    #[test]
    fn test_serialize_entry_roundtrip() {
        let entry = user_msg_entry("m1", Some("root"));
        let line = serialize_entry(&entry);
        assert!(line.ends_with('\n'));
        let parsed = parse_entry_line(line.trim(), 2, Path::new("/tmp/s.jsonl")).unwrap();
        assert_eq!(parsed.id(), "m1");
        assert_eq!(parsed.parent_id(), Some("root"));
    }

    #[test]
    fn test_parse_entry_line_invalid_json() {
        assert!(matches!(
            parse_entry_line("not json", 2, Path::new("/tmp/s.jsonl")),
            Err(JsonlError::InvalidEntry(2, _))
        ));
    }

    #[test]
    fn test_parse_entry_line_missing_type() {
        let line = r#"{"id":"m1","timestamp":"t"}"#;
        assert!(matches!(
            parse_entry_line(line, 2, Path::new("/tmp/s.jsonl")),
            Err(JsonlError::InvalidEntry(2, _))
        ));
    }

    #[test]
    fn test_parse_entry_line_missing_id() {
        let line = r#"{"type":"message","timestamp":"t"}"#;
        assert!(matches!(
            parse_entry_line(line, 2, Path::new("/tmp/s.jsonl")),
            Err(JsonlError::InvalidEntry(2, _))
        ));
    }

    #[test]
    fn test_parse_entry_line_missing_timestamp() {
        let line = r#"{"type":"message","id":"m1"}"#;
        assert!(matches!(
            parse_entry_line(line, 2, Path::new("/tmp/s.jsonl")),
            Err(JsonlError::InvalidEntry(2, _))
        ));
    }

    // ─── header_to_metadata ───────────────────────────────────────────

    #[test]
    fn test_header_to_metadata_copies_fields() {
        let header = SessionHeader {
            ty: "session".to_string(),
            version: CURRENT_SESSION_VERSION as u8,
            id: "s1".to_string(),
            timestamp: "2025-01-15T10:30:00.000Z".to_string(),
            cwd: "/home/user".to_string(),
            parent_session: Some("/old/s.jsonl".to_string()),
            metadata: None,
        };
        let meta = header_to_metadata(&header, PathBuf::from("/root/--home-user--/x.jsonl"));
        assert_eq!(meta.id, "s1");
        assert_eq!(meta.created_at, "2025-01-15T10:30:00.000Z");
        assert_eq!(meta.cwd, "/home/user");
        assert_eq!(meta.path, PathBuf::from("/root/--home-user--/x.jsonl"));
        assert_eq!(meta.parent_session_path.as_deref(), Some("/old/s.jsonl"));
        assert!(meta.metadata.is_none());
    }

    // ─── get_entries_to_fork ──────────────────────────────────────────

    fn fork_chain() -> Vec<SessionTreeEntry> {
        vec![
            user_msg_entry("m0", None),
            user_msg_entry("m1", Some("m0")),
            leaf_entry("leaf1", Some("m1"), None),
        ]
    }

    #[test]
    fn test_get_entries_to_fork_no_entry_id_returns_all() {
        let entries = fork_chain();
        let result = get_entries_to_fork(&entries, None, ForkPosition::Before).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_get_entries_to_fork_at_position() {
        let entries = fork_chain();
        let result = get_entries_to_fork(&entries, Some("m1"), ForkPosition::At).unwrap();
        let ids: Vec<&str> = result.iter().map(|e| e.id()).collect();
        assert_eq!(ids, vec!["m0", "m1"]);
    }

    #[test]
    fn test_get_entries_to_fork_before_position() {
        let entries = fork_chain();
        // m1 is a user message; before that, from m1.parent_id(m0) to root
        let result = get_entries_to_fork(&entries, Some("m1"), ForkPosition::Before).unwrap();
        let ids: Vec<&str> = result.iter().map(|e| e.id()).collect();
        assert_eq!(ids, vec!["m0"]);
    }

    #[test]
    fn test_get_entries_to_fork_before_non_user_message_errors() {
        let entries = fork_chain();
        // leaf1 is not a user message
        assert!(matches!(
            get_entries_to_fork(&entries, Some("leaf1"), ForkPosition::Before),
            Err(JsonlError::ForkTargetInvalid(_))
        ));
    }

    #[test]
    fn test_get_entries_to_fork_entry_not_found_errors() {
        let entries = fork_chain();
        assert!(matches!(
            get_entries_to_fork(&entries, Some("nope"), ForkPosition::At),
            Err(JsonlError::ForkTargetInvalid(_))
        ));
    }

    // ─── File I/O (std temp directory) ───────────────────────────────

    #[test]
    fn test_create_session_file_writes_header() {
        let root = test_dir("create");
        let path = create_session_file(
            &root,
            "/home/user",
            CreateSessionOptions {
                session_id: Some("s1".to_string()),
                parent_session_path: None,
                metadata: None,
            },
        )
        .unwrap();

        assert!(path.exists());
        let content = std::fs::read_to_string(&path).unwrap();
        let header = parse_header_line(content.trim(), &path).unwrap();
        assert_eq!(header.id, "s1");
        assert_eq!(header.cwd, "/home/user");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_append_and_read_session_file_roundtrip() {
        let root = test_dir("append_read");
        let path = create_session_file(
            &root,
            "/home/user",
            CreateSessionOptions {
                session_id: Some("s1".to_string()),
                parent_session_path: None,
                metadata: None,
            },
        )
        .unwrap();

        append_entry(&path, &user_msg_entry("m0", None)).unwrap();
        append_entry(&path, &user_msg_entry("m1", Some("m0"))).unwrap();

        let (header, entries, last_leaf) = read_session_file(&path).unwrap();
        assert_eq!(header.id, "s1");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].id(), "m0");
        assert_eq!(entries[1].id(), "m1");
        // When there are no leaf entries, last_leaf takes the id of the last non-leaf entry
        assert_eq!(last_leaf.as_deref(), Some("m1"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_read_session_file_last_leaf_from_leaf_entry() {
        let root = test_dir("last_leaf");
        let path = create_session_file(
            &root,
            "/home/user",
            CreateSessionOptions {
                session_id: Some("s1".to_string()),
                parent_session_path: None,
                metadata: None,
            },
        )
        .unwrap();
        append_entry(&path, &user_msg_entry("m0", None)).unwrap();
        append_entry(&path, &leaf_entry("leaf1", Some("m0"), Some("m0"))).unwrap();

        let (_, _, last_leaf) = read_session_file(&path).unwrap();
        assert_eq!(last_leaf.as_deref(), Some("m0"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_read_session_metadata_reads_header() {
        let root = test_dir("metadata");
        let path = create_session_file(
            &root,
            "/home/user",
            CreateSessionOptions {
                session_id: Some("s1".to_string()),
                parent_session_path: None,
                metadata: None,
            },
        )
        .unwrap();

        let meta = read_session_metadata(&path).unwrap();
        assert_eq!(meta.id, "s1");
        assert_eq!(meta.path, path);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_read_session_file_not_found_is_io_error() {
        let p = Path::new("/nonexistent/zen/definitely/missing.jsonl");
        assert!(matches!(read_session_file(p), Err(JsonlError::Io(_))));
    }

    #[test]
    fn test_append_entry_missing_file_is_io_error() {
        let p = Path::new("/nonexistent/zen/definitely/missing.jsonl");
        assert!(matches!(
            append_entry(p, &user_msg_entry("m0", None)),
            Err(JsonlError::Io(_))
        ));
    }

    #[test]
    fn test_delete_session_removes_file() {
        let root = test_dir("delete");
        let path = create_session_file(
            &root,
            "/home/user",
            CreateSessionOptions {
                session_id: Some("s1".to_string()),
                parent_session_path: None,
                metadata: None,
            },
        )
        .unwrap();
        assert!(path.exists());
        delete_session(&path).unwrap();
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_delete_session_not_found_is_ok() {
        let p = Path::new("/nonexistent/zen/definitely/missing.jsonl");
        assert!(delete_session(p).is_ok());
    }

    #[test]
    fn test_list_session_dirs_lists_subdirs() {
        let root = test_dir("list_dirs");
        std::fs::create_dir_all(root.join("--a--")).unwrap();
        std::fs::create_dir_all(root.join("--b--")).unwrap();
        std::fs::create_dir_all(root.join(".hidden")).unwrap();

        let dirs = list_session_dirs(&root).unwrap();
        let names: Vec<String> = dirs
            .iter()
            .map(|d| d.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(names, vec!["--a--", "--b--"]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_list_sessions_by_cwd_sorts_desc() {
        let root = test_dir("list_sessions");
        create_session_file(
            &root,
            "/home/user",
            CreateSessionOptions {
                session_id: Some("s1".to_string()),
                parent_session_path: None,
                metadata: None,
            },
        )
        .unwrap();
        create_session_file(
            &root,
            "/home/user",
            CreateSessionOptions {
                session_id: Some("s2".to_string()),
                parent_session_path: None,
                metadata: None,
            },
        )
        .unwrap();

        let sessions = list_sessions(&root, Some("/home/user")).unwrap();
        assert_eq!(sessions.len(), 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_list_sessions_missing_cwd_returns_empty() {
        let root = test_dir("list_empty");
        let sessions = list_sessions(&root, Some("/nonexistent/cwd")).unwrap();
        assert!(sessions.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }
}
