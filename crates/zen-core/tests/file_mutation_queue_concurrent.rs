// REQ-AGENT-02 (pi-219): file mutation queue (serialized file writes) concurrency integration tests
//
// Covers: multiple write/edit tool operations concurrently modifying the same underlying file (aligned with pi
// serialization semantics in reference/pi-original/packages/coding-agent/src/core/tools/file-mutation-queue.ts):
//   1. Concurrent writes to the same file → final state is exactly the complete content of one write (no tearing/interleaving)
//   2. Concurrent edits to the same file → read-modify-write is fully serialized, all edits are preserved (no lost updates)
//   3. Mixed concurrent write and edit → final state must be the result of some valid serial order
//   4. Concurrent real path and symlink alias → share the same per-key lock (unix)
//
// No LLM dependency: call Tool::execute directly, manually construct argument JSON.
// Temp file cleanup: tempfile::TempDir recursively deletes on Drop, executed on both success and panic (unwind) paths.

use std::sync::Arc;

use serde_json::json;
use tokio::sync::Barrier;
use tokio_util::sync::CancellationToken;
use zen_ai::tool::{Tool, ToolContext};
use zen_core::tools::edit::EditTool;
use zen_core::tools::file_mutation_queue::with_file_mutation_queue;
use zen_core::tools::write::WriteTool;

fn write_tool(dir: &std::path::Path) -> WriteTool {
    WriteTool::new(ToolContext::new(dir.to_path_buf(), None, None))
}

fn edit_tool(dir: &std::path::Path) -> EditTool {
    EditTool::new(ToolContext::new(dir.to_path_buf(), None, None))
}

// ── 1. Concurrent writes to the same file ─────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_writes_same_file_end_with_one_complete_content() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let n = 6;
    // Contents of different lengths: if two writes interleave on disk, the final state will not match either complete content
    let contents: Vec<String> = (0..n)
        .map(|i| format!("write-{i}-{}", "x".repeat((i + 1) * 256)))
        .collect();
    let barrier = Arc::new(Barrier::new(n));

    let mut handles = Vec::with_capacity(n);
    for content in &contents {
        let tool = write_tool(dir.path());
        let content = content.clone();
        let barrier = barrier.clone();
        handles.push(tokio::spawn(async move {
            barrier.wait().await; // Align start points to maximize the race window
            tool.execute(
                "wc",
                &json!({"path": "target.txt", "content": content}),
                &CancellationToken::new(),
                Box::new(|_| {}),
            )
            .await
        }));
    }
    for handle in handles {
        handle
            .await
            .expect("task panicked")
            .expect("write must succeed");
    }

    let final_content = std::fs::read_to_string(dir.path().join("target.txt")).expect("read back");
    assert!(
        contents.iter().any(|c| *c == final_content),
        "Final state must be exactly the complete content of one write (no tearing/interleaving); actual: {final_content:?}"
    );
}

// ── 2. Concurrent edits to the same file ─────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_edits_same_file_preserve_every_edit() {
    let dir = tempfile::tempdir().expect("create tempdir");
    std::fs::write(dir.path().join("e.txt"), "aaa bbb ccc ddd").expect("seed");

    let replacements = [
        ("aaa", "AAA"),
        ("bbb", "BBB"),
        ("ccc", "CCC"),
        ("ddd", "DDD"),
    ];
    let barrier = Arc::new(Barrier::new(replacements.len()));

    let mut handles = Vec::new();
    for (old, new) in replacements {
        let tool = edit_tool(dir.path());
        let barrier = barrier.clone();
        handles.push(tokio::spawn(async move {
            barrier.wait().await;
            tool.execute(
                "ec",
                &json!({"path": "e.txt", "edits": [{"oldText": old, "newText": new}]}),
                &CancellationToken::new(),
                Box::new(|_| {}),
            )
            .await
        }));
    }
    for handle in handles {
        handle
            .await
            .expect("task panicked")
            .expect("edit must succeed");
    }

    // Lost-update criterion: if one edit reads the original content before another edit writes back,
    // its replacement will be overwritten by the later writer, and the final state will lack the marker
    let final_content = std::fs::read_to_string(dir.path().join("e.txt")).expect("read back");
    assert_eq!(
        final_content, "AAA BBB CCC DDD",
        "All concurrent edits must be preserved (no lost updates)"
    );
}

// ── 3. Mixed concurrent write and edit ─────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_write_and_edit_end_in_valid_serial_outcome() {
    let dir = tempfile::tempdir().expect("create tempdir");
    std::fs::write(dir.path().join("m.txt"), "common base").expect("seed");
    let barrier = Arc::new(Barrier::new(2));

    let wt = write_tool(dir.path());
    let wb = barrier.clone();
    let write_task = tokio::spawn(async move {
        wb.wait().await;
        wt.execute(
            "wc",
            &json!({"path": "m.txt", "content": "common written"}),
            &CancellationToken::new(),
            Box::new(|_| {}),
        )
        .await
    });
    let et = edit_tool(dir.path());
    let edit_task = tokio::spawn(async move {
        barrier.wait().await;
        et.execute(
            "ec",
            &json!({"path": "m.txt", "edits": [{"oldText": "common", "newText": "common edited"}]}),
            &CancellationToken::new(),
            Box::new(|_| {}),
        )
        .await
    });

    write_task
        .await
        .expect("task panicked")
        .expect("write must succeed");
    edit_task
        .await
        .expect("task panicked")
        .expect("edit must succeed");

    // Two valid serial orders:
    //   write→edit: "common written" → "common edited written"
    //   edit→write: "common edited base" → overwritten by write to "common written"
    // Interleaving (without queue) produces "common edited base" (write lost) — excluded
    let final_content = std::fs::read_to_string(dir.path().join("m.txt")).expect("read back");
    assert!(
        final_content == "common written" || final_content == "common edited written",
        "Final state must equal the result of some valid serial order; actual: {final_content:?}"
    );
}

// ── 4. Symlink alias shares queue (unix) ───────────────────────────────

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn symlink_alias_shares_mutation_queue_with_real_path() {
    use std::os::unix::fs::symlink;

    let dir = tempfile::tempdir().expect("create tempdir");
    let real = dir.path().join("real.txt");
    let link = dir.path().join("alias.txt");
    symlink(&real, &link).expect("create symlink");
    std::fs::write(&real, "seed-v1").expect("seed");

    // Task A: acquires lock via real path. Lock action: first notify main "lock acquired", then leave
    // a 150ms lock-holding window (workspace tokio has no time feature, use std::thread::sleep;
    // blocking a single worker under multi_thread runtime is acceptable), finally write v2.
    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel::<()>();
    let real_for_call = real.clone();
    let real_for_write = real.clone();
    let task_a = tokio::spawn(async move {
        with_file_mutation_queue(&real_for_call, move || {
            let entered_tx = entered_tx;
            let path = real_for_write;
            async move {
                entered_tx.send(()).expect("notify main");
                std::thread::sleep(std::time::Duration::from_millis(150));
                std::fs::write(&path, "seed-v2").expect("A write v2");
            }
        })
        .await;
    });

    // Start B only after A confirms lock acquisition — eliminates the "who registers first" scheduling race
    entered_rx.await.expect("A was not enqueued");

    // Task B: goes through edit tool via symlink alias. If the alias shares the queue, B will wait until A
    // releases, then reads v2 and succeeds; if not shared, B will immediately read v1, oldText does not match and fails.
    let dir_for_b = dir.path().to_path_buf();
    let link_str = link.to_string_lossy().to_string();
    let task_b = tokio::spawn(async move {
        let tool = EditTool::new(ToolContext::new(dir_for_b, None, None));
        tool.execute(
               "eb",
               &json!({"path": link_str, "edits": [{"oldText": "seed-v2", "newText": "seed-v2-edited"}]}),
               &CancellationToken::new(),
               Box::new(|_| {}),
           )
           .await
    });

    task_a.await.expect("task A panicked");
    let result = task_b.await.expect("task B panicked").expect(
        "edit on alias path must wait for the lock-holding operation on the real path to complete",
    );
    assert_eq!(result.message.tool_name, "edit");

    let final_content = std::fs::read_to_string(&real).expect("read back");
    assert_eq!(
        final_content, "seed-v2-edited",
        "B must read the v2 content written by A (alias shares the same lock)"
    );
}
