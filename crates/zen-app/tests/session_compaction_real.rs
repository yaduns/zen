//! Compaction scheduling persistence round-trip integration test (T-AP-04-3).
//!
//! Verify that the **details key-name contract** of `SessionStore::append_compaction` (camelCase
//! `readFiles`/`modifiedFiles`) holds end to end:
//! Session A writes a compaction entry → persisted to disk → reopen session B → append message → compact again →
//! The details produced by the new compaction must **seed** the file list registered by the previous compaction entry
//! (`zen_core::compaction::seed_file_ops_from_details` reads it back by this key).
//! If the implementation drifts to snake_case keys, seeding silently fails after reopen, and this test fails.

use std::sync::Arc;

use zen_ai::types::{
    ContentBlock, ContentBlockList, Message, Model, NonErrorStopReason, TextContent, ThinkingLevel,
    Usage, UserMessage,
};
use zen_app::config::settings::SettingsManager;
use zen_app::session::agent_session::SessionStore;
use zen_app::session::jsonl::CreateSessionOptions;
use zen_app::session::services::{create_agent_session, NoopHooks};
use zen_app::session::session_manager::SessionManager;
use zen_core::compaction::{CompactionDetails, CompactionResult};
use zen_core::env::ShellConfig;

fn test_model() -> Model {
    Model {
        id: "deepseek-v4-flash".into(),
        name: None,
        provider: "opencode-go".into(),
        api: zen_ai::provider::Api::OpenAiCompletions,
        base_url: "https://opencode.ai/zen/go/v1".into(),
        context_window: 65536,
        max_tokens: 8192,
        reasoning: true,
        headers: None,
        cost_rates: None,
    }
}

/// StreamFn that produces a fixed text summary (Start + Done)
fn summary_stream_fn(text: &'static str) -> zen_ai::types::StreamFn {
    use zen_ai::api::openai_stream::StreamConfig;
    use zen_ai::types::AssistantMessageEvent;
    Arc::new(
        move |_model: &Model,
              _ctx: &zen_ai::types::Context,
              _cfg: Option<StreamConfig>,
              _cancel: Option<tokio_util::sync::CancellationToken>| {
            Box::pin(async move {
                let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<AssistantMessageEvent>();
                let partial = zen_ai::types::AssistantMessage {
                    content: vec![ContentBlock::Text(TextContent {
                        text: text.to_string(),
                        text_signature: None,
                    })],
                    api: "openai-completions".into(),
                    provider: "test".into(),
                    model: "m".into(),
                    response_model: None,
                    response_id: None,
                    usage: Usage::default(),
                    stop_reason: zen_ai::types::StopReason::Stop,
                    error_message: None,
                    timestamp: 0,
                };
                let _ = tx.send(AssistantMessageEvent::Done {
                    reason: NonErrorStopReason::Stop,
                    message: partial,
                });
                drop(tx);
                Ok::<_, zen_ai::api::openai_stream::StreamError>(rx)
            })
        },
    )
}

fn user_msg(text: &str) -> Message {
    Message::User(UserMessage {
        content: ContentBlockList::Text(text.to_string()),
        timestamp: 0,
    })
}

#[tokio::test]
async fn compaction_details_key_contract_survives_reopen_and_reseeds() {
    let root =
        std::env::temp_dir().join(format!("zen_compaction_roundtrip_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();

    // ── Session A: first compaction, register /proj/a.rs as a read file ──
    let mgr = SessionManager::create(
        "/proj",
        &root,
        CreateSessionOptions {
            session_id: None,
            parent_session_path: None,
            metadata: None,
        },
    )
    .expect("create session");
    let user_id = mgr.append_message(&user_msg("read some files"));
    let first = CompactionResult {
        summary: "first summary".into(),
        first_kept_entry_id: user_id.clone(),
        tokens_before: 1_000,
        usage: None,
        retained_tail: vec![],
        details: CompactionDetails {
            read_files: vec!["/proj/a.rs".to_string()],
            modified_files: vec![],
        },
    };
    mgr.append_compaction(&first, false);
    let file = mgr.get_session_file().expect("persisted");
    drop(mgr);

    // ── Session B: reopen → append message → second compaction ──
    let reopened = SessionManager::open(&file, None, &root).expect("reopen");
    reopened.append_message(&user_msg("more work"));

    let store: Arc<dyn SessionStore> = Arc::new(reopened);
    let session = create_agent_session(
        test_model(),
        "test".into(),
        vec![],
        ThinkingLevel::Off,
        Arc::new(NoopHooks),
        summary_stream_fn("second summary"),
        Arc::clone(&store),
        ShellConfig::default(),
        Arc::new(SettingsManager::empty()),
        vec![],
        vec![],
    );

    // Manual compaction: prepare seeds file-operation buckets from the previous compaction entry details + extracts new messages
    let second = session.compact(None).await.expect("second compaction");

    // Assertion: seeding takes effect—the new compaction's details.read_files contains /proj/a.rs registered previously
    // (If append_compaction writes snake_case keys, seeding silently fails after reopen → this assertion fails)
    assert!(
        second
            .details
            .read_files
            .contains(&"/proj/a.rs".to_string()),
        "seeded read_files should survive reopen, got {:?}",
        second.details.read_files
    );
    assert!(second.summary.contains("second summary"));
    // Another proof that seeding took effect: the summary appendix lists the seeded read files (format_file_operations)
    assert!(second.summary.contains("/proj/a.rs"));

    // Third compaction entry has been persisted and leaf redirected
    let entries = store.branch_entries();
    assert!(matches!(
        entries.last(),
        Some(zen_core::session::tree::SessionTreeEntry::Compaction { .. })
    ));

    let _ = std::fs::remove_dir_all(&root);
}
