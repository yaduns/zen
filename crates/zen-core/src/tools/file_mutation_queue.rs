use crate::paths::canonicalize_path;
use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::LazyLock;
use tokio::sync::Mutex;

/// Async serialization registry keyed by real file path (keyed-lock).
///
/// When the write/edit tools perform a whole-file change ("read old content → compute new content → write back"), concurrent calls on the same file must be queued:
/// otherwise two tools read the same old content and the later write overwrites the former (lost update). This type holds one exclusive lock per registered path:
/// same path serialized, different paths parallel, symlink aliases share the same lock. Production code uses the global singleton; tests may construct isolated instances.
pub(crate) struct FileMutationQueue {
    /// Key (canonical absolute path) → serialization lock for that file; the outer Mutex only briefly guards map access.
    inner: Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>,
}

impl FileMutationQueue {
    /// Construct an empty queue registry.
    ///
    /// Use when tests need mutually isolated instances, or when the caller wants an independent (non-global) serialization scope;
    /// production code normally does not call this directly, but uses the `global()` lazily-initialized singleton via `with_file_mutation_queue`.
    pub(crate) fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    /// Run the serialized operation `f` on the underlying file at `file_path`, returning the result of `f`.
    ///
    /// Behavior: first compute the key (`mutation_key`: absolutize + canonicalize, falling back to the absolute path if missing, so symlink aliases and
    /// different spellings share the same queue); fetch (or create) the `Arc<Mutex<()>>` for that key and hold the lock, `f().await`,
    /// then release the lock and remove the map entry by identity (`Arc::ptr_eq`) if it is still this lock (matching pi's `=== chainedQueue` cleanup).
    ///
    /// The lock covers the **entire** run of `f`, including the tail where `f` waits for in-flight IO to complete before returning — this guarantees pi's semantics that "the queue is not released before a cancelled in-flight write completes": subsequent operations on the same file must wait for the previous operation's write to land.
    /// same file's subsequent operations must wait until the previous operation's write-back is flushed to disk before continuing.
    ///
    /// Edge cases:
    /// - `f` returns `T` (usually `Result<...>`); errors are returned to the caller as-is by `f`, this method does not swallow them.
    /// - It is safe for this future to be dropped (e.g. cancelled) **while waiting for the per-key lock**: `tokio::sync::Mutex::lock` is cancel-safe,
    ///   it gives up without acquiring the lock, no lock leak, no deadlock.
    /// - After `f` returns normally, the map is cleaned up; if `f` panics, the per-key lock is released during unwinding via guard Drop, but the map entry
    ///   cleanup is skipped (a stale unlocked entry remains; a very minor leak on the abnormal path, acceptable; the normal path always cleans up).
    /// - Concurrency: different keys do not block each other.
    pub(crate) async fn with_lock<T, F, Fut>(&self, file_path: &Path, f: F) -> T
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>,
    {
        let key = mutation_key(file_path);

        // Fetch (or create) the lock for this key: clone the Arc then immediately release the outer lock — the outer Mutex only briefly guards map
        // access, is not held across await, so subsequent locking by different keys does not block each other.
        let lock = {
            let mut map = self.inner.lock().await;
            map.entry(key.clone())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };

        // Hold the per-key lock through the entire run of f (including the tail of f waiting for its in-flight IO). tokio Mutex::lock is cancel-safe:
        // if the future is dropped while waiting for the lock it gives up, no leak, no deadlock; if f panics, the guard releases the lock during unwinding
        // via Drop, but the cleanup below is skipped (a stale unlocked entry remains on the abnormal path, acceptable).
        let _guard = lock.lock().await;
        let result = f().await;
        drop(_guard);

        // Cleanup:
        // remove only if the entry is still this lock (ptr_eq) **and** there are no other in-flight holders (strong_count == 2: map 1 + this task 1).
        // Checking ptr_eq alone is insufficient: in pi each registrant replaces the chain head (`set(key, chainedQueue)`), and `===` inequality means
        // someone registered later; this implementation's or_insert does not replace the entry, so later and earlier registrants share the same Arc, and checking only ptr_eq would
        // wrongly delete the entry while someone is queued — a subsequent new task's or_insert would create a new lock, running concurrently with the still-queued holders of the old lock.
        // Count accuracy: clones only happen inside the outer lock (while held, others' clones are excluded by the mutex); others' drops only decrement, never increment,
        // and a drop means that task has finished/been cancelled, so it poses no serialization threat.
        let mut map = self.inner.lock().await;
        let last_registrant = map
            .get(&key)
            .is_some_and(|entry| Arc::ptr_eq(entry, &lock) && Arc::strong_count(entry) == 2);
        if last_registrant {
            map.remove(&key);
        }
        result
    }
}

/// Global file mutation queue singleton (matching pi's module-level `fileMutationQueues`).
///
/// Lazily initialized; within this task it is only used by `with_file_mutation_queue`, not marked `pub`.
fn global() -> &'static FileMutationQueue {
    static QUEUE: LazyLock<FileMutationQueue> = LazyLock::new(FileMutationQueue::new);
    &QUEUE
}

/// Serialize mutation operations on the same underlying file: different files run in parallel, the same file is queued, symlink aliases share the queue.
///
/// The write/edit tools call this function before writing the same file in parallel tool calls, handing over the whole read-modify-write as `f`:
/// it acquires the per-key exclusive lock on the caller's behalf (the lock covers the entire run of `f`, including the tail of a cancelled in-flight write-back) and returns the result of `f`.
/// Equivalent to "one entrance per door, different doors pass independently". Production consumers: `WriteTool::execute` (write.rs),
/// `EditTool::execute` (edit.rs), both wired up by this task.
///
/// Params: `file_path` is the already-resolved absolute path (obtained by the tool layer via `resolve_to_cwd`, consistent with pi's extension doc requirements);
/// `f` is the async operation to serialize. The return value is the output of `f` (in tool scenarios `Result<_, ToolError>`/`Result<_, EditError>`).
///
/// Edge cases: when the path does not exist (new file), queue by absolute path, not affecting creation; see `with_lock` for the cleanup boundary when `f` panics.
pub async fn with_file_mutation_queue<T, F, Fut>(file_path: &Path, f: F) -> T
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = T>,
{
    global().with_lock(file_path, f).await
}

/// Compute the queue key: absolutize (`std::path::absolute`, relative input normalized against cwd) + canonicalize to follow symlinks;
/// fall back to the absolute path when the file/dangling link is missing (reusing `crate::paths::canonicalize_path`).
fn mutation_key(file_path: &Path) -> PathBuf {
    // ① Absolutize: relative input is expanded against the current working directory;
    //    only errors when cwd cannot be obtained (process-level anomaly), in which case it degrades to the raw input — the key is only used for hash grouping,
    //    serialization semantics need not be interrupted by an absolute failure.
    let absolute = std::path::absolute(file_path).unwrap_or_else(|_| file_path.to_path_buf());
    // ② canonicalize follows symlinks (matching pi's `realpath`); when the target is missing (new file/dangling link)
    //    `canonicalize_path` falls back to the passed-in absolute path.
    canonicalize_path(&absolute)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::oneshot;

    // ── mutation_key ────────────────────────────────────────────

    #[test]
    fn test_mutation_key_relative_becomes_absolute() {
        let key = mutation_key(Path::new("rel/name.txt"));
        assert!(key.is_absolute());
        assert!(key.ends_with(Path::new("rel/name.txt")));
    }

    #[test]
    fn test_mutation_key_absolute_passthrough() {
        let dir = tempfile::tempdir().unwrap();
        let abs = dir.path().join("abs.txt");
        assert_eq!(mutation_key(&abs), abs);
    }

    #[test]
    fn test_mutation_key_missing_file_fallback_to_absolute() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("no-such.txt");
        // Fall back to the absolute path when the target is missing (new file), no error, does not affect new-file queuing
        assert_eq!(mutation_key(&missing), missing);
    }

    #[test]
    fn test_mutation_key_existing_file_canonicalizes() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("real.txt");
        std::fs::write(&file, "x").unwrap();
        assert_eq!(mutation_key(&file), std::fs::canonicalize(&file).unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn test_mutation_key_symlink_resolves_to_target() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target.txt");
        let alias = dir.path().join("alias.txt");
        std::fs::write(&target, "x").unwrap();
        symlink(&target, &alias).unwrap();
        assert_eq!(
            mutation_key(&alias),
            std::fs::canonicalize(&target).unwrap()
        );
    }

    #[test]
    fn test_mutation_key_normalizes_dot_dot() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        let file = dir.path().join("same.txt");
        std::fs::write(&file, "x").unwrap();
        let messy = sub.join("..").join("same.txt"); // dir/sub/../same.txt == dir/same.txt
        let clean = dir.path().join("same.txt");
        // Two spellings of the same file (including ..) yield the same queue key via canonicalize (filesystem resolves ..)
        assert_eq!(mutation_key(&messy), mutation_key(&clean));
    }

    // ── FileMutationQueue::new ───────────────────────────────────

    #[tokio::test]
    async fn test_new_creates_empty_registry() {
        let queue = FileMutationQueue::new();
        let map = queue.inner.lock().await;
        assert!(map.is_empty());
    }

    // ── FileMutationQueue::with_lock ─────────────────────────────

    #[tokio::test]
    async fn test_with_lock_returns_f_result() {
        let queue = FileMutationQueue::new();
        let file = Path::new("/tmp/zen-mq-return.txt");
        // Normal return values are passed through as-is
        assert_eq!(queue.with_lock(file, || async { 42 }).await, 42);
        // Error return values are not swallowed either (in tool scenarios the Result is returned to the caller by f itself)
        let err: Result<(), &str> = queue.with_lock(file, || async { Err("boom") }).await;
        assert_eq!(err, Err("boom"));
    }

    #[tokio::test]
    async fn test_with_lock_cleans_up_after_last_completion() {
        let queue = FileMutationQueue::new();
        let file = Path::new("/tmp/zen-mq-cleanup.txt");
        queue.with_lock(file, || async {}).await;
        // After the last registrant completes, the entry is cleaned up (map does not grow unboundedly)
        let map = queue.inner.lock().await;
        assert!(map.is_empty());
    }

    #[tokio::test]
    async fn test_with_lock_serializes_same_file() {
        let queue = Arc::new(FileMutationQueue::new());
        let file = PathBuf::from("/tmp/zen-mq-same.txt");
        let order: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));

        let (first_started_tx, mut first_started_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let (second_started_tx, mut second_started_rx) = oneshot::channel();

        let q = queue.clone();
        let o = order.clone();
        let h1_file = file.clone();
        let h1 = tokio::spawn(async move {
            q.with_lock(&h1_file, || async move {
                o.lock().await.push("first:start");
                first_started_tx.send(()).unwrap();
                release_rx.await.unwrap();
                o.lock().await.push("first:end");
            })
            .await;
        });

        // Wait until first acquires the lock and enters f
        first_started_rx.await.unwrap();

        let q = queue.clone();
        let o = order.clone();
        let h2_file = file.clone();
        let h2 = tokio::spawn(async move {
            q.with_lock(&h2_file, || async move {
                o.lock().await.push("second:start");
                second_started_tx.send(()).unwrap();
                o.lock().await.push("second:end");
            })
            .await;
        });

        // second is queued: must not enter f before first releases
        let started = second_started_rx
            .try_recv()
            .expect_err("second must not start before first is released");
        assert!(matches!(
            started,
            tokio::sync::oneshot::error::TryRecvError::Empty
        ));

        release_tx.send(()).unwrap();
        h1.await.unwrap();
        second_started_rx.await.unwrap();
        h2.await.unwrap();

        let order = order.lock().await;
        assert_eq!(
            order.as_slice(),
            &["first:start", "first:end", "second:start", "second:end"]
        );
    }

    #[tokio::test]
    async fn test_with_lock_different_files_parallel() {
        let queue = Arc::new(FileMutationQueue::new());
        let fa = PathBuf::from("/tmp/zen-mq-a.txt");
        let fb = PathBuf::from("/tmp/zen-mq-b.txt");
        let (a_started_tx, mut a_started_rx) = oneshot::channel();
        let (release_a_tx, release_a_rx) = oneshot::channel();
        let (b_started_tx, mut b_started_rx) = oneshot::channel();

        let q = queue.clone();
        let h_a = tokio::spawn(async move {
            q.with_lock(&fa, || async move {
                a_started_tx.send(()).unwrap();
                release_a_rx.await.unwrap();
            })
            .await;
        });

        // A already holds the lock while suspended
        a_started_rx.await.unwrap();

        let q = queue.clone();
        let h_b = tokio::spawn(async move {
            q.with_lock(&fb, || async move {
                b_started_tx.send(()).unwrap();
            })
            .await;
        });

        // Different files do not block each other: B should already have entered f while A holds the lock
        b_started_rx.await.unwrap();

        release_a_tx.send(()).unwrap();
        h_a.await.unwrap();
        h_b.await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_with_lock_symlink_alias_shares_queue() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target.txt");
        let alias = dir.path().join("alias.txt");
        std::fs::write(&target, "hello").unwrap();
        symlink(&target, &alias).unwrap();

        let queue = Arc::new(FileMutationQueue::new());
        let order: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
        let (t_started_tx, mut t_started_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let (a_started_tx, mut a_started_rx) = oneshot::channel();

        let q = queue.clone();
        let o = order.clone();
        let h1 = tokio::spawn(async move {
            q.with_lock(&target, || async move {
                o.lock().await.push("target:start");
                t_started_tx.send(()).unwrap();
                release_rx.await.unwrap();
                o.lock().await.push("target:end");
            })
            .await;
        });
        t_started_rx.await.unwrap();

        let q = queue.clone();
        let o = order.clone();
        let h2 = tokio::spawn(async move {
            q.with_lock(&alias, || async move {
                o.lock().await.push("alias:start");
                a_started_tx.send(()).unwrap();
                o.lock().await.push("alias:end");
            })
            .await;
        });

        // alias and target share the same queue after canonicalize: alias must not enter f before target releases
        let started = a_started_rx
            .try_recv()
            .expect_err("alias must not start before target is released");
        assert!(matches!(
            started,
            tokio::sync::oneshot::error::TryRecvError::Empty
        ));

        release_tx.send(()).unwrap();
        h1.await.unwrap();
        a_started_rx.await.unwrap();
        h2.await.unwrap();

        let order = order.lock().await;
        assert_eq!(
            order.as_slice(),
            &["target:start", "target:end", "alias:start", "alias:end"]
        );
    }

    #[tokio::test]
    async fn test_with_lock_keeps_entry_while_queued() {
        let queue = Arc::new(FileMutationQueue::new());
        let file = PathBuf::from("/tmp/zen-mq-queued.txt");
        let key = mutation_key(&file);

        let (a_started_tx, mut a_started_rx) = oneshot::channel();
        let (release_a_tx, release_a_rx) = oneshot::channel();
        let (b_entered_tx, mut b_entered_rx) = oneshot::channel();
        let (release_b_tx, release_b_rx) = oneshot::channel();

        let q = queue.clone();
        let h_a_file = file.clone();
        let h_a = tokio::spawn(async move {
            q.with_lock(&h_a_file, || async move {
                a_started_tx.send(()).unwrap();
                release_a_rx.await.unwrap();
            })
            .await;
        });
        // A already holds the lock while suspended
        a_started_rx.await.unwrap();

        let q = queue.clone();
        let h_b_file = file.clone();
        let h_b = tokio::spawn(async move {
            q.with_lock(&h_b_file, || async move {
                b_entered_tx.send(()).unwrap();
                release_b_rx.await.unwrap();
            })
            .await;
        });

        // Wait for B to finish registration (strong_count = map + A + B = 3), confirming B is enqueued
        loop {
            let map = queue.inner.lock().await;
            let registered = map
                .get(&key)
                .is_some_and(|entry| Arc::strong_count(entry) >= 3);
            drop(map);
            if registered {
                break;
            }
            tokio::task::yield_now().await;
        }

        // Release A: when A completes, B is still in flight (registered but not completed) → the entry must be retained. Checking only ptr_eq would incorrectly delete it here,
        // causing subsequent new registrations to create a new lock and run concurrently with the still-in-flight B → lost updates (this test verifies this invariant).
        release_a_tx.send(()).unwrap();
        h_a.await.unwrap();

        // B has acquired the lock and entered f (still in flight, not yet cleaned up); the entry should still be in the map at this point
        b_entered_rx.await.unwrap();
        let map = queue.inner.lock().await;
        assert!(
            map.contains_key(&key),
            "Entry must not be deleted while tasks are in flight"
        );
        drop(map);

        // Release B: B is the last in-flight task → its cleanup removes the entry (map does not grow unboundedly)
        release_b_tx.send(()).unwrap();
        h_b.await.unwrap();
        let map = queue.inner.lock().await;
        assert!(!map.contains_key(&key));
    }

    // ── with_file_mutation_queue (global singleton entry) ─────────────────

    #[tokio::test]
    async fn test_with_file_mutation_queue_serializes_global() {
        // Use a unique temp file to avoid interfering with other tests on the global singleton
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("global.txt");
        let order: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
        let (first_started_tx, mut first_started_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();

        let o = order.clone();
        let h1_file = file.clone();
        let h1 = tokio::spawn(async move {
            with_file_mutation_queue(&h1_file, || async move {
                o.lock().await.push("g1:start");
                first_started_tx.send(()).unwrap();
                release_rx.await.unwrap();
                o.lock().await.push("g1:end");
            })
            .await;
        });
        first_started_rx.await.unwrap();

        let o = order.clone();
        let h2_file = file.clone();
        let h2 = tokio::spawn(async move {
            with_file_mutation_queue(&h2_file, || async move {
                o.lock().await.push("g2:start");
                o.lock().await.push("g2:end");
            })
            .await;
        });

        release_tx.send(()).unwrap();
        h1.await.unwrap();
        h2.await.unwrap();

        let order = order.lock().await;
        assert_eq!(
            order.as_slice(),
            &["g1:start", "g1:end", "g2:start", "g2:end"]
        );
    }
}
