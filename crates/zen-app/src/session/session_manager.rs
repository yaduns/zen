use crate::session::agent_session::SessionStore;
use crate::session::jsonl::{
    append_entry, create_session_file, create_timestamp, get_entries_to_fork, list_sessions,
    read_session_file, CreateSessionOptions, ForkPosition, JsonlError, JsonlSessionMetadata,
    CURRENT_SESSION_VERSION,
};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use zen_ai::types::Message;
use zen_core::compaction::{CompactionDetails, CompactionResult};
use zen_core::session::context::build_session_context;
use zen_core::session::tree::{SessionContext, SessionTree, SessionTreeEntry, SessionTreeError};

/// Session lifecycle manager
pub struct SessionManager {
    cwd: String,                // Immutable: unchanged after construction; source of the &str for session_store::cwd()
    sessions_root: PathBuf,     // Immutable: default get_sessions_dir()
    persist: bool,              // Immutable: in_memory=false
    inner: Mutex<SessionInner>, // Mutable: all state that may change
}

struct SessionInner {
    tree: SessionTree,             // entries + leaf_id
    session_id: String,            // Replaced with a new one on create/open/fork
    session_file: Option<PathBuf>, // None when persist=false
}

/// SessionManager operation error, aggregating underlying JSONL / tree errors and domain validation errors
#[derive(Debug, thiserror::Error)]

pub enum SessionManagerError {
    #[error("session persistence: {0}")]
    Jsonl(#[from] JsonlError), // JSONL persistence error
    #[error("session tree: {0}")]
    Tree(#[from] SessionTreeError), // Session tree errors
    #[error("entry not found: {0}")]
    EntryNotFound(String), // Branch/fork target entry does not exist
    #[error("fork target is not a valid entry type: {0}")]
    ForkTargetInvalid(String), // Fork target is not a UserMessage, etc.
    #[error("invalid session file: {0}")]
    InvalidSession(String), // Session file is not a valid zen session
}

impl SessionManager {
    /// Create a brand new session (persisted to disk).
    ///
    /// # Behavior
    /// 1. Call `jsonl::create_session_file(sessions_root, cwd, options)` to write the header file
    /// 2. Construct an empty `SessionTree::new(vec![], None)`
    /// 3. Return a SessionManager holding `session_id`/`session_file`
    ///
    /// # Parameters
    /// - `cwd`: session working directory (stores the session header)
    /// - `sessions_root`: sessions root directory (defaults to `config::get_sessions_dir()`)
    /// - `options`: session options (id / parent_session_path / metadata), reusing `jsonl::CreateSessionOptions`
    pub fn create(
        cwd: &str,
        sessions_root: &Path,
        options: CreateSessionOptions,
    ) -> Result<Self, SessionManagerError> {
        // 1. Create the JSONL file (write header), return the file path
        let session_file = create_session_file(sessions_root, cwd, options)?;

        // 2. Extract session_id from the file name (format: {timestamp}-{session_id}.jsonl)
        let session_id = session_file
            .file_stem()
            .and_then(|stem| stem.to_str())
            .and_then(|s| s.rsplit('-').next())
            .ok_or_else(|| {
                SessionManagerError::InvalidSession(format!(
                    "cannot extract session id from path: {}",
                    session_file.display()
                ))
            })?
            .to_string();

        // 3. Empty session tree (no entries, no leaves)
        let tree = SessionTree::new(vec![], None);

        // 4. Assemble the SessionManager
        Ok(Self {
            cwd: cwd.to_string(),
            sessions_root: sessions_root.to_path_buf(),
            persist: true,
            inner: Mutex::new(SessionInner {
                tree,
                session_id,
                session_file: Some(session_file),
            }),
        })
    }

    /// Open the specified session file (restore).
    ///
    /// # Behavior
    /// 1. `jsonl::read_session_file(path)` reads header + entries + last_leaf
    /// 2. `SessionTree::new(entries, last_leaf)` rebuilds the tree
    /// 3. Use the header's cwd (or `cwd_override`) as cwd
    /// 4. Validate that the file is a valid zen session (version=3)
    ///
    /// # Parameters
    /// - `path`: session file path
    /// - `cwd_override`: override the cwd in the header (optional)
    /// - `sessions_root`: session root directory (for new branches/new sessions)
    pub fn open(
        path: &Path,
        cwd_override: Option<&str>,
        sessions_root: &Path,
    ) -> Result<Self, SessionManagerError> {
        // 1. Read JSONL (header + entries + last_leaf)
        let (header, entries, last_leaf_id) = read_session_file(path)?;

        // 2. Validate version
        if header.version as u64 != CURRENT_SESSION_VERSION {
            return Err(SessionManagerError::InvalidSession(format!(
                "unsupported session version: {} (expected {CURRENT_SESSION_VERSION})",
                header.version
            )));
        }

        // 3. cwd: caller override takes precedence, otherwise use the one from the header
        let cwd = cwd_override.unwrap_or(&header.cwd).to_string();

        // 4. Rebuild the session tree
        let tree = SessionTree::new(entries, last_leaf_id);

        // 5. Assemble the SessionManager
        Ok(Self {
            cwd,
            sessions_root: sessions_root.to_path_buf(),
            persist: true,
            inner: Mutex::new(SessionInner {
                tree,
                session_id: header.id,
                session_file: Some(path.to_path_buf()),
            }),
        })
    }

    /// Restore the most recent session; if none exists, create a new one.
    ///
    /// # Behavior
    /// 1. Call `jsonl::list_sessions(sessions_root, Some(cwd))` to get the most recent session under that cwd
    /// 2. If found → `open`; otherwise → `create`
    pub fn continue_recent(cwd: &str, sessions_root: &Path) -> Result<Self, SessionManagerError> {
        let sessions = list_sessions(sessions_root, Some(cwd))?;

        match sessions.into_iter().next() {
            Some(latest) => Self::open(&latest.path, Some(cwd), sessions_root),
            None => Self::create(
                cwd,
                sessions_root,
                CreateSessionOptions {
                    session_id: None,
                    parent_session_path: None,
                    metadata: None,
                },
            ),
        }
    }

    /// Create an in-memory session (not persisted to disk).
    ///
    /// # Behavior
    /// Construct an empty SessionManager with `persist=false` and `session_file=None`.
    pub fn in_memory(cwd: &str) -> Self {
        Self {
            cwd: cwd.to_string(),
            sessions_root: PathBuf::new(),
            persist: false,
            inner: Mutex::new(SessionInner {
                tree: SessionTree::new(vec![], None),
                session_id: uuid::Uuid::now_v7().to_string(),
                session_file: None,
            }),
        }
    }

    /// Fork a new session from another project's session file (into the current cwd).
    ///
    /// # Behavior
    /// 1. `jsonl::read_session_file(source_path)` to read the source session entries
    /// 2. Validate that the source session is non-empty and contains a valid header
    /// 3. New header (new id, cwd=target_cwd, parent_session=source_path)
    /// 4. Write all non-header entries from the source session into the new file one by one (`append_entry`)
    /// 5. Return the SessionManager pointing at the new file
    ///
    /// # Parameters
    /// - `source_path`: source session file
    /// - `target_cwd`: working directory for the new session
    /// - `sessions_root`: root directory for the new session
    pub fn fork_from(
        source_path: &Path,
        target_cwd: &str,
        sessions_root: &Path,
        options: CreateSessionOptions,
    ) -> Result<Self, SessionManagerError> {
        // 1. Read source session
        let (_source_header, source_entries, _source_leaf) = read_session_file(source_path)?;

        if source_entries.is_empty() {
            return Err(SessionManagerError::InvalidSession(
                "source session has no entries to fork".into(),
            ));
        }

        // 2. Create new session file
        let new_options = CreateSessionOptions {
            session_id: options.session_id,
            parent_session_path: Some(source_path.display().to_string()),
            metadata: options.metadata,
        };
        let session_file = create_session_file(sessions_root, target_cwd, new_options)?;

        // 3. Write source entries one by one
        let mut tree_entries = Vec::with_capacity(source_entries.len());
        for entry in &source_entries {
            append_entry(&session_file, entry)?;
            tree_entries.push(entry.clone());
        }

        // 4. Rebuild tree + assemble SessionManager
        let last_leaf_id = tree_entries.last().map(|e| e.id().to_string());
        let tree = SessionTree::new(tree_entries, last_leaf_id);

        // 5. Extract session_id
        let session_id = session_file
            .file_stem()
            .and_then(|stem| stem.to_str())
            .and_then(|s| s.rsplit('-').next())
            .ok_or_else(|| {
                SessionManagerError::InvalidSession(format!(
                    "cannot extract session id from path: {}",
                    session_file.display()
                ))
            })?
            .to_string();

        Ok(Self {
            cwd: target_cwd.to_string(),
            sessions_root: sessions_root.to_path_buf(),
            persist: true,
            inner: Mutex::new(SessionInner {
                tree,
                session_id,
                session_file: Some(session_file),
            }),
        })
    }
    // Consumers: T-AP-11-1 (cross-project fork), T-AP-08-3

    // ── Append methods (return entry id; also serve as the concrete implementation source for SessionStore) ─────

    /// Append a session message as a child of the current leaf, advancing the leaf. Returns the entry id.
    ///
    /// # Behavior
    /// 1. Construct `SessionTreeEntry::Message { id, parent_id=leaf_id, timestamp, message c }`
    /// 2. `tree.append(entry)` (set leaf)
    /// 3. On `persist`, call `jsonl::append_entry(session_file, &entry)`
    pub fn append_message(&self, message: &Message) -> String {
        // 1. Inside the lock: pure in-memory operations (ID generation + tree append)
        let (entry_id, entry_to_persist, session_file) = {
            let mut inner = self.inner.lock().unwrap();
            let entry_id = inner.tree.generate_entry_id();
            let parent_id = inner.tree.leaf_id().map(|s| s.to_string());
            let entry = SessionTreeEntry::Message {
                id: entry_id.clone(),
                parent_id,
                timestamp: create_timestamp(),
                message: message.clone(),
            };
            inner.tree.append(entry.clone());
            let file = if self.persist {
                inner.session_file.clone()
            } else {
                None
            };
            (entry_id, entry, file)
        }; // ← lock released

        // 2. Outside the lock: file I/O (doesn't block other threads)
        if let Some(path) = session_file {
            if let Err(e) = append_entry(&path, &entry_to_persist) {
                tracing::warn!("JSONL append failed (entry in memory only): {e}");
            }
        }

        entry_id
    }
    // Consumers: T-AP-01-1 (AgentSession subscription closure via SessionStore), T-AP-11-1 (interactive append)

    /// Append a thinking-level change, returning the entry id.
    pub fn append_thinking_level_change(&self, thinking_level: &str) -> String {
        // 1. Inside the lock: pure in-memory operations (ID generation + tree append)
        let (entry_id, entry_to_persist, session_file) = {
            let mut inner = self.inner.lock().unwrap();
            let entry_id = inner.tree.generate_entry_id();
            let parent_id = inner.tree.leaf_id().map(|s| s.to_string());
            let entry = SessionTreeEntry::ThinkingLevelChange {
                id: entry_id.clone(),
                parent_id,
                timestamp: create_timestamp(),
                thinking_level: thinking_level.to_string(),
            };
            inner.tree.append(entry.clone());
            let file = if self.persist {
                inner.session_file.clone()
            } else {
                None
            };
            (entry_id, entry, file)
        }; // ← lock released

        // 2. Outside the lock: file I/O (doesn't block other threads)
        if let Some(path) = session_file {
            if let Err(e) = append_entry(&path, &entry_to_persist) {
                tracing::warn!("JSONL append failed (entry in memory only): {e}");
            }
        }

        entry_id
    }
    // Consumers: T-AP-01-1 (via SessionStore), T-AP-11-1 (/thinking)

    /// Append a compaction entry and redirect the leaf, returning the entry id (T-AP-04-3 scheduling persistence exit).
    ///
    /// # Parameter mapping (CompactionResult → SessionTreeEntry::Compaction)
    /// - `summary`/`tokens_before` → passed through under the same name;
    /// - `first_kept_entry_id: String` → `Option<String>` (always Some when non-empty; pi optional is a transitional artifact);
    /// - `retained_tail: Vec<Message>` → `Option<Vec<Message>>` (empty Vec → None; entry shape matches pi "no tail");
    /// - `usage` → passed through; `details` → serialized via [`compaction_details_to_json`];
    /// - `from_hook` → `Some(from_hook)` (app-layer compaction is always false).
    pub fn append_compaction(&self, result: &CompactionResult, from_hook: bool) -> String {
        // 1. Inside the lock: pure in-memory operations (ID generation + tree append)
        let (entry_id, entry_to_persist, session_file) = {
            let mut inner = self.inner.lock().unwrap();
            let entry_id = inner.tree.generate_entry_id();
            let parent_id = inner.tree.leaf_id().map(|s| s.to_string());
            let entry = SessionTreeEntry::Compaction {
                id: entry_id.clone(),
                parent_id,
                timestamp: create_timestamp(),
                summary: result.summary.clone(),
                first_kept_entry_id: Some(result.first_kept_entry_id.clone()),
                tokens_before: result.tokens_before,
                retained_tail: (!result.retained_tail.is_empty())
                    .then(|| result.retained_tail.clone()),
                details: Some(compaction_details_to_json(&result.details)),
                usage: result.usage.clone(),
                from_hook: Some(from_hook),
            };
            inner.tree.append(entry.clone());
            let file = if self.persist {
                inner.session_file.clone()
            } else {
                None
            };
            (entry_id, entry, file)
        }; // ← lock released

        // 2. Outside the lock: file I/O (doesn't block other threads)
        if let Some(path) = session_file {
            if let Err(e) = append_entry(&path, &entry_to_persist) {
                tracing::warn!("JSONL append failed (entry in memory only): {e}");
            }
        }

        entry_id
    }
    // Consumer: T-AP-04-3 (via SessionStore::append_compaction)

    /// Append a model change, returning the entry id.
    pub fn append_model_change(&self, provider: &str, model_id: &str) -> String {
        // 1. Inside the lock: pure in-memory operations (ID generation + tree append)
        let (entry_id, entry_to_persist, session_file) = {
            let mut inner = self.inner.lock().unwrap();
            let entry_id = inner.tree.generate_entry_id();
            let parent_id = inner.tree.leaf_id().map(|s| s.to_string());
            let entry = SessionTreeEntry::ModelChange {
                id: entry_id.clone(),
                parent_id,
                timestamp: create_timestamp(),
                provider: provider.to_string(),
                model_id: model_id.to_string(),
            };
            inner.tree.append(entry.clone());
            let file = if self.persist {
                inner.session_file.clone()
            } else {
                None
            };
            (entry_id, entry, file)
        }; // ← lock released

        // 2. Outside the lock: file I/O (doesn't block other threads)
        if let Some(path) = session_file {
            if let Err(e) = append_entry(&path, &entry_to_persist) {
                tracing::warn!("JSONL append failed (entry in memory only): {e}");
            }
        }

        entry_id
    }

    // ── Navigation / read-only access ───────────────────────────────────────────

    /// Current session ID (clone; returns owned String after the SessionStore revision).
    pub fn get_session_id(&self) -> String {
        self.inner.lock().unwrap().session_id.clone()
    }

    /// The current working directory.
    pub fn get_cwd(&self) -> &str {
        &self.cwd
    }
    // Consumers: T-AP-07-1, T-AP-11-1

    /// Path to the current session file (None when persist=false).
    pub fn get_session_file(&self) -> Option<PathBuf> {
        self.inner.lock().unwrap().session_file.clone()
    }

    /// Current leaf entry id.
    pub fn get_leaf_id(&self) -> Option<String> {
        self.inner
            .lock()
            .unwrap()
            .tree
            .leaf_id()
            .map(|s| s.to_string())
    }
    // Consumer: T-AP-11-1 (navigation)

    /// Look up an entry by id.
    pub fn get_entry(&self, id: &str) -> Option<SessionTreeEntry> {
        self.inner
            .lock()
            .unwrap()
            .tree
            .entries()
            .iter()
            .find(|entry| entry.id() == id)
            .cloned()
    }

    /// All session entries (excluding header).
    pub fn get_entries(&self) -> Vec<SessionTreeEntry> {
        self.inner.lock().unwrap().tree.entries().to_vec()
    }

    /// Walk the parent chain from `from_id` (or the current leaf) to the root, returning an ordered entry slice (root first).
    /// Counterpart of Pi's `getBranch`.
    pub fn get_branch(&self, from_id: Option<&str>) -> Vec<SessionTreeEntry> {
        self.inner
            .lock()
            .unwrap()
            .tree
            .get_path_to_root(from_id)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Build the LLM context (delegates to zen-core `build_context(&tree, None)`).
    /// Rebuild AgentState.messages from `session_context.messages` for runtime assembly.
    pub fn build_context(&self) -> SessionContext {
        let inner = self.inner.lock().unwrap();
        let path: Vec<SessionTreeEntry> = inner
            .tree
            .get_path_to_root(None) // None → use the current leaf_id
            .into_iter()
            .cloned()
            .collect();
        build_session_context(&path)
    }

    // ── Branch navigation ──────────────────────────────────────────

    /// Move the leaf pointer to the given entry, so the next append becomes its child (forming a new branch).
    /// Existing entries are not modified or deleted.
    pub fn branch(&self, branch_from_id: &str) -> Result<(), SessionManagerError> {
        self.inner.lock().unwrap().tree.branch(branch_from_id)?;
        Ok(())
    }

    /// Reset the leaf pointer to None (the next append becomes the root, parent_id=None).
    pub fn reset_leaf(&self) {
        self.inner.lock().unwrap().tree.reset_leaf();
    }

    /// Extract the "root → given leaf" path into a new standalone session file (fork a single branch).
    ///
    /// ### Behavior
    /// 1. `jsonl::get_entries_to_fork(entries, Some(leaf_id), ForkPosition::At)` to get the branch
    /// 2. `jsonl::create_session_file` to create the new file (new id, parent_session = current file path)
    /// 3. Write the branch entries one by one via `append_entry`
    /// 4. Return the new session file path (None when persist=false)
    pub fn create_branched_session(
        &self,
        leaf_id: &str,
    ) -> Result<Option<PathBuf>, SessionManagerError> {
        let inner = self.inner.lock().unwrap();
        let entries = inner.tree.entries();

        // 1. Get root -> branch entries for the specified leaf
        let branch_entries = get_entries_to_fork(entries, Some(leaf_id), ForkPosition::At)?;

        if branch_entries.is_empty() {
            return Err(SessionManagerError::ForkTargetInvalid(leaf_id.to_string()));
        }

        // 2. In non-persistent mode, return None directly
        if !self.persist {
            return Ok(None);
        }

        let current_file = inner.session_file.as_ref().ok_or_else(|| {
            SessionManagerError::InvalidSession("no session file to branch from".into())
        })?;

        // 3. Create a new session file (parent_session = current file)
        let new_options = CreateSessionOptions {
            session_id: None,
            parent_session_path: Some(current_file.display().to_string()),
            metadata: None,
        };
        let new_file = create_session_file(&self.sessions_root, &self.cwd, new_options)?;

        // 4. Write branch entries one by one
        for entry in &branch_entries {
            append_entry(&new_file, entry)?;
        }

        Ok(Some(new_file))
    }

    // ── List ──────────────────────────────────────────────────────

    /// List all sessions in the `cwd` session directory (ordered by most recently modified, descending).
    /// Delegates directly to `jsonl::list_sessions(sessions_root, Some(cwd))`.
    pub fn list(
        cwd: &str,
        sessions_root: &Path,
    ) -> Result<Vec<JsonlSessionMetadata>, SessionManagerError> {
        Ok(list_sessions(sessions_root, Some(cwd))?)
    }

    /// List sessions across all project directories (ordered by most recently modified, descending).
    pub fn list_all(
        sessions_root: &Path,
    ) -> Result<Vec<JsonlSessionMetadata>, SessionManagerError> {
        Ok(list_sessions(sessions_root, None)?)
    }
}

/// Implements the `SessionStore` trait, shared by AgentSession / AgentSessionServices
/// via `Arc<dyn SessionStore>`. Methods delegate to the inherent append/read methods; returns are unified to the trait signatures.
///
/// Note: `session_id` returns an owned String (the trait was revised from &str, see open question 1), because
/// session_id is mutable inside the Mutex, so it cannot be borrowed across the lock for the return value.
///
/// Name-collision note: the trait method names (append_message / append_model_change /
/// append_thinking_level_change) collide with the inherent methods of `impl SessionManager` (the inherent returns
/// an entry id `String`, the trait returns `()`). Rust allows them to coexist — calls through `&SessionManager` use the inherent
/// methods (returning the id), calls through `&dyn SessionStore` use the trait methods (returning ()). During implementation, the inherent
/// methods are the real implementation and the trait methods delegate to them (`todo!()` placeholders, internally delegating to the inherent methods at implementation time).
/// Serialize compaction details to entry JSON—**key-name contract camelCase `readFiles`/`modifiedFiles`
/// (same shape as pi JSONL)**: consumer `zen_core::compaction::seed_file_ops_from_details` seeds by these keys
/// file-operation buckets; if written as snake_case, seeding silently fails after reopening the session (missing keys do not error).
/// Acceptance: the round-trip test in session_manager tests (write to disk → reopen → seed) prevents key-name drift.
fn compaction_details_to_json(details: &CompactionDetails) -> serde_json::Value {
    serde_json::json!({
        "readFiles": details.read_files,
        "modifiedFiles": details.modified_files,
    })
}

impl SessionStore for SessionManager {
    /// Append a message. Called via AgentSession's resident subscription (persisted to disk in real time).
    fn append_message(&self, _message: &Message) {
        self.append_message(_message);
    }
    /// Record a model switch.
    fn append_model_change(&self, _provider: &str, _model_id: &str) {
        self.append_model_change(_provider, _model_id);
    }
    /// Record a thinking-level switch.
    fn append_thinking_level_change(&self, _level: &str) {
        self.append_thinking_level_change(_level);
    }
    /// Current session ID (owned, see open question 1).
    fn session_id(&self) -> String {
        self.get_session_id()
    }
    /// The current working directory.
    fn cwd(&self) -> &str {
        &self.cwd
    }
    /// Compaction scheduling reads current-branch entries (T-AP-04-3: input for prepare_compaction and stale usage checks).
    fn branch_entries(&self) -> Vec<SessionTreeEntry> {
        self.get_branch(None)
    }
    /// Write a compaction entry and redirect the leaf (T-AP-04-3). Delegates to the inherent method, returning the entry id.
    fn append_compaction(&self, result: &CompactionResult, from_hook: bool) -> String {
        self.append_compaction(result, from_hook)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::jsonl::encode_cwd;
    use zen_ai::types::{ContentBlockList, UserMessage};

    // ─── Test helpers ───────────────────────────────────────────────

    /// Creates a unique test directory under the system temp directory (std-only, to avoid introducing tempfile).
    /// Removes the old directory on each call to ensure a clean state.
    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "zen_session_manager_{}_{}",
            name,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn user_message(text: &str) -> Message {
        Message::User(UserMessage {
            content: ContentBlockList::Text(text.to_string()),
            timestamp: 0,
        })
    }

    fn create_in(root: &Path, cwd: &str, id: Option<&str>) -> SessionManager {
        SessionManager::create(
            cwd,
            root,
            CreateSessionOptions {
                session_id: id.map(String::from),
                parent_session_path: None,
                metadata: None,
            },
        )
        .expect("create should succeed")
    }

    /// Write a session file containing only a header (timestamp controllable), used to construct an expected "recent session".
    fn write_manual_session(root: &Path, cwd: &str, session_id: &str, timestamp: &str) -> PathBuf {
        let dir = root.join(encode_cwd(cwd));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{session_id}.jsonl"));
        let header = format!(
            r#"{{"type":"session","version":3,"id":"{session_id}","timestamp":"{timestamp}","cwd":"{cwd}"}}"#
        );
        std::fs::write(&path, format!("{header}\n")).unwrap();
        path
    }

    // ─── create ──────────────────────────────────────────────────────

    #[test]
    fn test_create_writes_file_and_sets_fields() {
        let root = test_dir("create");
        let mgr = create_in(&root, "/proj", None);

        let file = mgr.get_session_file().expect("persisted session has file");
        assert!(file.exists(), "session file should be written");
        assert_eq!(mgr.get_cwd(), "/proj");
        assert_eq!(mgr.get_session_id().len(), 8, "auto id is 8-char short id");
        assert!(mgr.get_leaf_id().is_none());
        assert!(mgr.get_entries().is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_create_with_custom_session_id() {
        let root = test_dir("create_custom");
        let mgr = create_in(&root, "/proj", Some("mysess"));

        assert_eq!(mgr.get_session_id(), "mysess");
        let file = mgr.get_session_file().unwrap();
        assert!(file
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .ends_with("mysess.jsonl"));
        let _ = std::fs::remove_dir_all(&root);
    }

    // ─── in_memory ───────────────────────────────────────────────────

    #[test]
    fn test_in_memory_has_no_file_and_no_persist() {
        let mgr = SessionManager::in_memory("/proj");

        assert_eq!(mgr.get_cwd(), "/proj");
        assert!(
            mgr.get_session_file().is_none(),
            "memory session has no file"
        );
        assert!(!mgr.get_session_id().is_empty());
        assert!(mgr.get_entries().is_empty());
        assert!(mgr.get_leaf_id().is_none());
    }

    // ─── open ────────────────────────────────────────────────────────

    #[test]
    fn test_open_restores_session_from_file() {
        let root = test_dir("open");
        let mgr = create_in(&root, "/proj", Some("abc123"));
        mgr.append_message(&user_message("first"));
        let path = mgr.get_session_file().unwrap();

        let reopened = SessionManager::open(&path, None, &root).unwrap();
        assert_eq!(reopened.get_session_id(), "abc123");
        assert_eq!(reopened.get_cwd(), "/proj");
        assert_eq!(reopened.get_entries().len(), 1);
        assert_eq!(reopened.get_leaf_id(), mgr.get_leaf_id());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_open_cwd_override_takes_precedence() {
        let root = test_dir("open_override");
        let mgr = create_in(&root, "/proj", Some("abc123"));
        let path = mgr.get_session_file().unwrap();

        let reopened = SessionManager::open(&path, Some("/other"), &root).unwrap();
        assert_eq!(reopened.get_cwd(), "/other");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_open_missing_file_errors() {
        let root = test_dir("open_missing");
        let result = SessionManager::open(Path::new("/nonexistent/x.jsonl"), None, &root);
        assert!(matches!(result, Err(SessionManagerError::Jsonl(_))));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_open_invalid_version_errors() {
        let root = test_dir("open_badver");
        let path = root.join("bad.jsonl");
        std::fs::write(
            &path,
            r#"{"type":"session","version":2,"id":"x","timestamp":"t","cwd":"/"}"#,
        )
        .unwrap();

        let result = SessionManager::open(&path, None, &root);
        assert!(matches!(result, Err(SessionManagerError::Jsonl(_))));
        let _ = std::fs::remove_dir_all(&root);
    }

    // ─── continue_recent ─────────────────────────────────────────────

    #[test]
    fn test_continue_recent_creates_when_no_sessions() {
        let root = test_dir("continue_recent_empty");
        let mgr = SessionManager::continue_recent("/proj", &root).unwrap();

        assert!(mgr.get_session_file().is_some());
        assert!(mgr.get_session_file().unwrap().exists());
        assert_eq!(mgr.get_cwd(), "/proj");
        assert!(!mgr.get_session_id().is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_continue_recent_opens_latest_session() {
        let root = test_dir("continue_recent");
        write_manual_session(&root, "/proj", "older", "2025-01-01T00:00:00.000Z");
        write_manual_session(&root, "/proj", "newer", "2025-01-02T00:00:00.000Z");

        let mgr = SessionManager::continue_recent("/proj", &root).unwrap();
        assert_eq!(mgr.get_session_id(), "newer");
        assert_eq!(mgr.get_cwd(), "/proj");
        let _ = std::fs::remove_dir_all(&root);
    }

    // ─── fork_from ───────────────────────────────────────────────────

    #[test]
    fn test_fork_from_copies_entries_to_new_file() {
        let root = test_dir("fork_from");
        let src = create_in(&root, "/src", Some("src1"));
        src.append_message(&user_message("hello"));
        src.append_message(&user_message("world"));
        let src_path = src.get_session_file().unwrap();

        let forked = SessionManager::fork_from(
            &src_path,
            "/target",
            &root,
            CreateSessionOptions {
                session_id: None,
                parent_session_path: None,
                metadata: None,
            },
        )
        .unwrap();

        assert_eq!(forked.get_cwd(), "/target");
        assert_eq!(forked.get_entries().len(), 2);
        assert_eq!(
            forked.get_session_id().len(),
            8,
            "forked session gets new id"
        );
        assert_ne!(forked.get_session_id(), src.get_session_id());
        // Leaf advanced to the last source entry
        let leaf = forked.get_leaf_id().unwrap();
        assert_eq!(
            forked.get_entry(&leaf).unwrap().id(),
            forked.get_entries().last().unwrap().id()
        );

        // New file header's parentSession points to the source file
        let new_path = forked.get_session_file().unwrap();
        let first_line = std::fs::read_to_string(&new_path)
            .unwrap()
            .lines()
            .next()
            .unwrap()
            .to_string();
        let v: serde_json::Value = serde_json::from_str(&first_line).unwrap();
        assert_eq!(
            v["parentSession"],
            serde_json::Value::String(src_path.display().to_string())
        );

        // New file readable back: 2 entries
        let (_h, entries, _leaf) = read_session_file(&new_path).unwrap();
        assert_eq!(entries.len(), 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_fork_from_empty_source_errors() {
        let root = test_dir("fork_empty");
        let src = create_in(&root, "/src", Some("src1")); // Header only, no entries
        let src_path = src.get_session_file().unwrap();

        let result = SessionManager::fork_from(
            &src_path,
            "/target",
            &root,
            CreateSessionOptions {
                session_id: None,
                parent_session_path: None,
                metadata: None,
            },
        );
        assert!(matches!(
            result,
            Err(SessionManagerError::InvalidSession(_))
        ));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_fork_from_missing_source_errors() {
        let root = test_dir("fork_missing");
        let result = SessionManager::fork_from(
            Path::new("/nonexistent/source.jsonl"),
            "/target",
            &root,
            CreateSessionOptions {
                session_id: None,
                parent_session_path: None,
                metadata: None,
            },
        );
        assert!(matches!(result, Err(SessionManagerError::Jsonl(_))));
        let _ = std::fs::remove_dir_all(&root);
    }

    // ─── append_* ────────────────────────────────────────────────────

    #[test]
    fn test_append_message_persists_and_advances_leaf() {
        let root = test_dir("append");
        let mgr = create_in(&root, "/proj", Some("s1"));

        let id = mgr.append_message(&user_message("hello"));
        assert_eq!(mgr.get_entries().len(), 1);
        assert_eq!(mgr.get_leaf_id().as_deref(), Some(id.as_str()));
        assert!(mgr.get_entry(&id).is_some());

        // Entry persisted to disk
        let path = mgr.get_session_file().unwrap();
        let (_h, entries, _leaf) = read_session_file(&path).unwrap();
        assert_eq!(entries.len(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_append_chain_builds_branch_path() {
        let mgr = SessionManager::in_memory("/proj");
        let m1 = mgr.append_message(&user_message("one"));
        let m2 = mgr.append_message(&user_message("two"));

        let ids: Vec<String> = mgr
            .get_branch(None)
            .iter()
            .map(|e| e.id().to_string())
            .collect();
        assert_eq!(ids, vec![m1.clone(), m2.clone()]);
        assert_eq!(mgr.get_entry(&m2).unwrap().parent_id(), Some(m1.as_str()));
        assert_eq!(mgr.get_entry(&m1).unwrap().parent_id(), None);
    }

    #[test]
    fn test_append_thinking_level_change_entry() {
        let mgr = SessionManager::in_memory("/proj");
        mgr.append_thinking_level_change("high");

        let entries = mgr.get_entries();
        assert!(matches!(
            &entries[0],
            SessionTreeEntry::ThinkingLevelChange {
                thinking_level,
                ..
            } if thinking_level == "high"
        ));
    }

    #[test]
    fn test_append_model_change_entry() {
        let mgr = SessionManager::in_memory("/proj");
        mgr.append_model_change("deepseek", "deepseek-v4-flash");

        let entries = mgr.get_entries();
        assert!(matches!(
            &entries[0],
            SessionTreeEntry::ModelChange {
                provider,
                model_id,
                ..
            } if provider == "deepseek" && model_id == "deepseek-v4-flash"
        ));
    }

    // ─── Navigation / read-only access ───────────────────────────────────────

    #[test]
    fn test_get_entry_missing_returns_none() {
        let mgr = SessionManager::in_memory("/proj");
        assert!(mgr.get_entry("nope").is_none());
    }

    #[test]
    fn test_get_branch_with_explicit_from_id() {
        let mgr = SessionManager::in_memory("/proj");
        let m1 = mgr.append_message(&user_message("one"));
        mgr.append_message(&user_message("two"));

        let branch = mgr.get_branch(Some(&m1));
        assert_eq!(branch.len(), 1);
        assert_eq!(branch[0].id(), m1);
    }

    #[test]
    fn test_build_context_returns_recovered_messages() {
        let mgr = SessionManager::in_memory("/proj");
        mgr.append_message(&user_message("hello"));
        mgr.append_message(&user_message("world"));

        let ctx = mgr.build_context();
        assert!(
            !ctx.messages.is_empty(),
            "user messages should be recovered"
        );
        assert!(ctx.model.is_none(), "no assistant/model entries yet");
    }

    // ─── Branch navigation ───────────────────────────────────────────────

    #[test]
    fn test_branch_moves_leaf_and_next_append_becomes_child() {
        let mgr = SessionManager::in_memory("/proj");
        let m1 = mgr.append_message(&user_message("one"));
        mgr.append_message(&user_message("two"));

        mgr.branch(&m1).unwrap();
        assert_eq!(mgr.get_leaf_id().as_deref(), Some(m1.as_str()));

        let m3 = mgr.append_message(&user_message("three"));
        assert_eq!(mgr.get_entry(&m3).unwrap().parent_id(), Some(m1.as_str()));
        let ids: Vec<String> = mgr
            .get_branch(None)
            .iter()
            .map(|e| e.id().to_string())
            .collect();
        assert_eq!(ids, vec![m1.clone(), m3.clone()]);
    }

    #[test]
    fn test_branch_unknown_entry_errors() {
        let mgr = SessionManager::in_memory("/proj");
        let result = mgr.branch("nope");
        assert!(matches!(
            result,
            Err(SessionManagerError::Tree(SessionTreeError::EntryNotFound(
                _
            )))
        ));
    }

    #[test]
    fn test_reset_leaf_makes_next_append_root() {
        let mgr = SessionManager::in_memory("/proj");
        mgr.append_message(&user_message("one"));
        mgr.append_message(&user_message("two"));

        mgr.reset_leaf();
        assert!(mgr.get_leaf_id().is_none());

        let m3 = mgr.append_message(&user_message("three"));
        assert_eq!(mgr.get_entry(&m3).unwrap().parent_id(), None);
    }

    // ─── create_branched_session ─────────────────────────────────────

    #[test]
    fn test_create_branched_session_in_memory_returns_none() {
        let mgr = SessionManager::in_memory("/proj");
        let m1 = mgr.append_message(&user_message("one"));

        let result = mgr.create_branched_session(&m1);
        assert!(matches!(result, Ok(None)), "non-persist mode returns None");
    }

    #[test]
    fn test_create_branched_session_creates_new_file_with_path() {
        let root = test_dir("branch_session");
        let mgr = create_in(&root, "/proj", Some("src1"));
        let m1 = mgr.append_message(&user_message("one"));
        let m2 = mgr.append_message(&user_message("two"));

        let new_file = mgr
            .create_branched_session(&m2)
            .unwrap()
            .expect("persisted mode creates file");
        assert!(new_file.exists());

        let (_h, entries, _leaf) = read_session_file(&new_file).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].id(), m1);
        assert_eq!(entries[1].id(), m2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_create_branched_session_unknown_leaf_errors() {
        let root = test_dir("branch_session_err");
        let mgr = create_in(&root, "/proj", Some("src1"));

        let result = mgr.create_branched_session("nope");
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    // ─── list / list_all ─────────────────────────────────────────────

    #[test]
    fn test_list_returns_sessions_for_cwd() {
        let root = test_dir("list");
        create_in(&root, "/proj", Some("s1"));
        create_in(&root, "/proj", Some("s2"));

        let sessions = SessionManager::list("/proj", &root).unwrap();
        assert_eq!(sessions.len(), 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_list_empty_dir_returns_empty() {
        let root = test_dir("list_empty");
        let sessions = SessionManager::list("/proj", &root).unwrap();
        assert!(sessions.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_list_all_returns_sessions_across_cwds() {
        let root = test_dir("list_all");
        create_in(&root, "/projA", Some("s1"));
        create_in(&root, "/projB", Some("s2"));

        let sessions = SessionManager::list_all(&root).unwrap();
        assert_eq!(sessions.len(), 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    // ─── SessionStore trait implementation ─────────────────────────────

    #[test]
    fn test_session_store_trait_delegates_to_manager() {
        let root = test_dir("store");
        let mgr = create_in(&root, "/proj", Some("store1"));
        let store: &dyn SessionStore = &mgr;

        assert_eq!(store.session_id(), "store1");
        assert_eq!(store.cwd(), "/proj");

        store.append_message(&user_message("hello"));
        store.append_model_change("deepseek", "deepseek-v4-flash");
        store.append_thinking_level_change("high");

        // Trait methods are backed by inherent methods on the same tree
        assert_eq!(mgr.get_entries().len(), 3);
        let _ = std::fs::remove_dir_all(&root);
    }

    // ─── append_compaction（T-AP-04-3）───────────────────────────

    fn compaction_result() -> CompactionResult {
        CompactionResult {
            summary: "summary text".to_string(),
            first_kept_entry_id: "e1".to_string(),
            tokens_before: 12_345,
            usage: None,
            retained_tail: vec![],
            details: CompactionDetails {
                read_files: vec!["src/a.rs".to_string(), "src/b.rs".to_string()],
                modified_files: vec!["src/c.rs".to_string()],
            },
        }
    }

    #[test]
    fn test_append_compaction_writes_camelcase_details_and_redirects_leaf() {
        // Key-name contract: details must use camelCase readFiles/modifiedFiles (same shape as pi JSONL),
        // Consumer seed_file_ops_from_details seeds by these keys; if written as snake_case, it silently fails
        let root = test_dir("append_compaction");
        let mgr = create_in(&root, "/proj", None);
        mgr.append_message(&user_message("hi"));
        let result = compaction_result();
        let id = mgr.append_compaction(&result, false);
        assert!(!id.is_empty());
        // Redirect leaf to the compaction entry
        let entries = mgr.get_entries();
        assert_eq!(entries.len(), 2);
        let last = entries.last().unwrap();
        assert!(matches!(last, SessionTreeEntry::Compaction { .. }));
        assert_eq!(mgr.get_leaf_id().as_deref(), Some(id.as_str()));
        // details key-name contract
        if let SessionTreeEntry::Compaction {
            details,
            from_hook,
            first_kept_entry_id,
            ..
        } = last
        {
            let details = details.as_ref().unwrap();
            assert!(
                details.get("readFiles").is_some(),
                "readFiles key must be camelCase"
            );
            assert!(
                details.get("modifiedFiles").is_some(),
                "modifiedFiles key must be camelCase"
            );
            assert!(
                details.get("read_files").is_none(),
                "Must not contain snake_case keys"
            );
            assert_eq!(
                details
                    .get("readFiles")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>(),
                vec!["src/a.rs", "src/b.rs"]
            );
            assert_eq!(first_kept_entry_id.as_deref(), Some("e1"));
            assert_eq!(*from_hook, Some(false));
        } else {
            panic!("expected compaction entry");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_append_compaction_roundtrip_empty_tail_maps_to_none() {
        // Write to disk → reopen: JSONL serialization preserves all entry fields; empty retained_tail Vec → None
        let root = test_dir("compaction_roundtrip");
        let mgr = create_in(&root, "/proj", None);
        mgr.append_message(&user_message("hi"));
        let result = compaction_result();
        mgr.append_compaction(&result, false);
        let file = mgr.get_session_file().unwrap();
        drop(mgr);

        let reopened = SessionManager::open(&file, None, &root).expect("reopen should succeed");
        let entries = reopened.get_entries();
        assert_eq!(entries.len(), 2);
        if let SessionTreeEntry::Compaction {
            summary,
            tokens_before,
            retained_tail,
            details,
            ..
        } = entries.last().unwrap()
        {
            assert_eq!(summary, "summary text");
            assert_eq!(*tokens_before, 12_345);
            assert!(retained_tail.is_none(), "empty retained_tail should be written as None");
            assert!(details.as_ref().unwrap().get("readFiles").is_some());
        } else {
            panic!("expected compaction entry");
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
