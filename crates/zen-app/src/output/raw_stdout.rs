use tokio::io::AsyncWriteExt;
use tokio::sync::{mpsc, oneshot};

/// The sole serialized writer in print mode that writes structured output to real stdout (fd 1).
/// Internally starts a background writer task holding tokio::io::Stdout, receiving command messages via a FIFO mpsc::UnboundedSender
pub struct RawStdout {
    /// The only channel for sending commands (Write/Flush) to the writer task;
    /// FIFO ensures writes from different callers do not interleave. On struct drop, the sender is released and the writer task ends.
    sender: mpsc::UnboundedSender<RawStdoutMsg>,
}

enum RawStdoutMsg {
    /// Bytes pending write
    Write(Vec<u8>),
    /// Flush barrier: the writer task first processes all prior Writes (FIFO guarantees order), then flush()es and returns the result.
    /// Write order is naturally guaranteed by the FIFO channel, so no extra lock is needed; after the sender is released (struct drop), the writer task ends automatically.
    Flush(oneshot::Sender<std::io::Result<()>>),
}

#[derive(Debug, thiserror::Error)]
pub enum RawStdoutError {
    #[error("raw stdout writer is closed")]
    Closed,
    #[error("stdout io error: {0}")]
    Io(#[from] std::io::Error),
}

impl RawStdout {
    /// Creates a writer: starts a background writer task (holding tokio::io::Stdout).
    /// Must be called within the current tokio runtime context (spawns internally).
    pub fn new() -> RawStdout {
        // 1) Create the FIFO command channel (type corresponds to the field sender: UnboundedSender)
        let (sender, mut rx) = mpsc::unbounded_channel::<RawStdoutMsg>();
        // 2) Spawn the writer task: owns stdout + consumes commands.
        //    Dropping the JoinHandle detaches it: the task is driven only by the sender's lifetime,
        //    and ends naturally after struct drop (sender release), with no explicit join/abort
        tokio::spawn(async move {
            let mut stdout = tokio::io::stdout();
            // write_raw is fire-and-forget, so write errors cannot be returned immediately; record the first error here,
            // and hand it back to the caller on the next Flush (aligned with the docs: "real write errors surface uniformly at flush")
            let mut last_error: Option<std::io::Error> = None;
            while let Some(msg) = rx.recv().await {
                match msg {
                    RawStdoutMsg::Write(bytes) => {
                        // If an error already exists, skip subsequent writes (fd is broken; writing again only repeats a failing syscall)
                        if last_error.is_none() {
                            if let Err(e) = stdout.write_all(&bytes).await {
                                last_error = Some(e);
                            }
                        }
                    }
                    RawStdoutMsg::Flush(tx) => {
                        let result = match last_error.take() {
                            Some(e) => Err(e),
                            None => stdout.flush().await,
                        };
                        // The flush caller may have already given up waiting due to cancellation; send failure can be ignored
                        let _ = tx.send(result);
                    }
                }
            }
            // rx.recv() returns None (all senders released) → the task ends naturally and the stdout handle is released
        });
        RawStdout { sender }
    }

    /// Writes text to real stdout (fd 1) and returns immediately (fire-and-forget), without waiting for the write.
    /// Enqueues only; order is guaranteed by the FIFO channel; text is written as raw UTF-8 bytes without any formatting.
    /// Error: returns `RawStdoutError::Closed` if the writer task has terminated; real write errors surface uniformly at `flush`.
    pub fn write_raw(&self, text: &str) -> Result<(), RawStdoutError> {
        // Aligns with pi writeRawStdout: empty text is not enqueued (empty write is a no-op, avoiding unnecessary channel message allocation)
        if text.is_empty() {
            return Ok(());
        }
        // Enqueue and return only (fire-and-forget): FIFO guarantees cross-call order, raw UTF-8 bytes, and no waiting for disk
        self.sender
            .send(RawStdoutMsg::Write(text.as_bytes().to_vec()))
            .map_err(|_| RawStdoutError::Closed)
    }

    /// Waits for all enqueued writes to complete and flushes the underlying stdout.
    /// On the Rust side, the writer task's `stdout.flush()` replaces direct empty-block writes. Returns the first underlying write error (including BrokenPipe), or Ok(()) on success.
    /// Must be called before print mode exits (finally block); otherwise a fast exit truncates output not yet written.
    pub async fn flush(&self) -> Result<(), RawStdoutError> {
        // ① Send the Flush barrier: all Writes before it are already in the FIFO; when the writer task processes this
        //    message, all prior bytes must already have been write_all'ed (order guaranteed by FIFO)
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(RawStdoutMsg::Flush(tx))
            .map_err(|_| RawStdoutError::Closed)?;
        // ② Wait for the writer task to return the real result; then map the error
        rx.await
            .map_err(|_| RawStdoutError::Closed)?
            .map_err(RawStdoutError::Io)
    }
}

impl Default for RawStdout {
    /// Equivalent to `RawStdout::new()`: no-arg construction that starts the background writer task.
    /// Aligns with clippy `new_without_default`; semantically, Default is the simplest usable instance.
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: the writer task writes to real stdout (fd 1); bytes cannot be captured within the test process for assertions,
    // so tests focus on the public contract: write_raw enqueues and returns, flush barrier round-trip, and writer task survives across rounds.
    // The actual written content appears in the test process stdout when running cargo test (can be checked visually).
    // The `Closed` branch (task terminated) is unreachable via the public API: the task only ends when the sender (with struct drop)
    // Ends after release; by then no method handle is callable — defensive error branch, not unit-tested.

    /// Empty-text write_raw returns Ok directly without enqueueing (matches pi writeRawStdout's empty-write no-op semantics) and does not affect subsequent flush
    #[tokio::test]
    async fn test_write_raw_empty_text_returns_ok() {
        let writer = RawStdout::new();
        assert!(writer.write_raw("").is_ok());
        assert!(writer.flush().await.is_ok());
    }

    /// Basic path: after writing a text segment, flush returns Ok (full round trip of FIFO write + flush barrier)
    #[tokio::test]
    async fn test_write_raw_then_flush_ok() {
        let writer = RawStdout::new();
        assert!(writer.write_raw("hello raw stdout\n").is_ok());
        assert!(writer.flush().await.is_ok());
    }

    /// flush still returns Ok when nothing has been written (empty-queue barrier)
    #[tokio::test]
    async fn test_flush_with_no_writes_ok() {
        let writer = RawStdout::new();
        assert!(writer.flush().await.is_ok());
    }

    /// flush is idempotent: two consecutive flushes both return Ok (writer task stays alive across flushes)
    #[tokio::test]
    async fn test_flush_twice_ok() {
        let writer = RawStdout::new();
        assert!(writer.write_raw("first\n").is_ok());
        assert!(writer.flush().await.is_ok());
        assert!(writer.flush().await.is_ok());
    }

    /// writer task lifetime spans multiple write/flush rounds
    #[tokio::test]
    async fn test_write_after_flush_ok() {
        let writer = RawStdout::new();
        assert!(writer.write_raw("round1\n").is_ok());
        assert!(writer.flush().await.is_ok());
        assert!(writer.write_raw("round2\n").is_ok());
        assert!(writer.flush().await.is_ok());
    }

    /// flush after multiple fire-and-forget writes: the FIFO barrier covers all enqueued writes (ordering guaranteed by the channel)
    #[tokio::test]
    async fn test_many_writes_then_flush_ok() {
        let writer = RawStdout::new();
        for i in 0..100 {
            assert!(writer.write_raw(&format!("line {i}\n")).is_ok());
        }
        assert!(writer.flush().await.is_ok());
    }

    /// Multi-byte UTF-8 text is enqueued and written byte-for-byte as-is without error
    #[tokio::test]
    async fn test_write_raw_multibyte_utf8_ok() {
        let writer = RawStdout::new();
        assert!(writer.write_raw("中文输出，含 emoji 🦀 与重音 é\n").is_ok());
        assert!(writer.flush().await.is_ok());
    }

    /// Large text (100KB) written via write_all does not error
    #[tokio::test]
    async fn test_write_raw_large_text_ok() {
        let writer = RawStdout::new();
        let big = "x".repeat(100 * 1024);
        assert!(writer.write_raw(&big).is_ok());
        assert!(writer.flush().await.is_ok());
    }
}
