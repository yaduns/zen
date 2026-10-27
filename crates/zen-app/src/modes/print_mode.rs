use crate::cli::Mode;
use crate::output::raw_stdout::RawStdout;
use crate::session::agent_session::AgentSession;
use crate::session::runtime::AgentSessionRuntime;
use std::sync::Arc;
use zen_ai::types::{AssistantMessage, ContentBlock, ImageContent, Message};
use zen_core::harness::types::AgentEvent;
use zen_core::process::ProcessRegistry;

/// print (single-run) mode entry point: send prompts → output → exit.
///
/// # Parameters
/// ```text
///   user input
///   ├── stdin (pipe)      ─┐
///   ├── @file file content  ─┼─→ merged into initial_message
///   └── first CLI message   ─┘
///            │
///            └── remaining CLI messages → messages
/// ```
/// - `runtime`: a fully assembled `AgentSessionRuntime` (containing the session and services; the `session` field is borrowed via `&mut` for `prompt`)
/// - `output_mode`: text (final text only) / json (one JSON line per event)
/// - `messages`: additional prompts sent serially after the initial message
/// - `initial_message`: the first message (may include @file content; already assembled by T-AP-08-2)
/// - `initial_images`: images attached to the initial message (empty Vec in Phase 1)
///
/// # Behavior
/// 1. Create `RawStdout`; in json mode subscribe to `AgentSession::subscribe`, serializing each `AgentEvent` to one JSON line and writing it out
/// 2. Race via `tokio::select!`: `print_prompts` (sends initial + messages) vs `await_signal_exit_code` (shutdown signal)
/// 3. Signal arrives first: terminate tracked detached child processes and return the signal exit code (SIGTERM→143 / SIGHUP→129)
/// 4. text mode: last message is an assistant message ending normally → output the text block; error/aborted → report to stderr and return 1
/// 5. finally: unsubscribe + `flush`, return the exit code
///
/// # Return Value
/// Exit codes: 0 = success; 1 = assistant error/abort or prompt threw; 143/129 = shutdown signal received
///
/// # Edge Cases
/// - Even with no initial message and no additional messages, run through once (json mode still subscribes and writes the header/events)
/// - json mode does not change the exit code based on stop reason (only exception paths set 1), aligned with pi
///
/// # Consumers
/// `crates/zen-app/src/main.rs` (node 5 mode dispatch)
pub async fn run_print_mode(
    runtime: AgentSessionRuntime,
    output_mode: Mode,
    messages: Vec<String>,
    initial_message: Option<String>,
    initial_images: Vec<ImageContent>,
) -> anyhow::Result<i32> {
    // at-attachment-wiring-fix: initial message carries image attachment (closes leftover T-AP-08-2/T-AP-08-4
    // "Phase 2 @file image" deferred seam — `AgentSession::prompt` has extended the images slot).
    let writer = Arc::new(RawStdout::new());

    // JSON mode: write to stdout via subscription as events occur; no post-processing needed
    let _unsub = if output_mode == Mode::Json {
        Some(subscribe_json_events(&runtime.session, Arc::clone(&writer)))
    } else {
        None
    };

    // Record the last assistant message (required for text mode)
    let mut last_assistant: Option<AssistantMessage> = None;

    // Send prompts
    let send_prompts = async {
        if let Some(ref text) = initial_message {
            let msgs = runtime.session.prompt(text, &initial_images).await?;
            // Record the last assistant
            for msg in msgs.iter().rev() {
                if let Message::Assistant(a) = msg {
                    last_assistant = Some(a.clone());
                    break;
                }
            }
        }
        for msg in &messages {
            let msgs = runtime.session.prompt(msg, &[]).await?;
            for msg in msgs.iter().rev() {
                if let Message::Assistant(a) = msg {
                    last_assistant = Some(a.clone());
                    break;
                }
            }
        }
        Ok::<(), anyhow::Error>(())
    };

    // race condition
    let registry = runtime.services.registry.clone();
    let exit_code = tokio::select! {
        biased;
        code = await_signal_exit_code(registry) => code,
        res = send_prompts => {
            if let Err(e) = res {
                eprintln!("Error: {e}");
                1
            } else {
                0
            }
        }
    };

    if exit_code != 0 {
        runtime.services.registry.kill_all();
        writer.flush().await?;
        return Ok(exit_code);
    }

    // Text mode: must wait for all prompts to finish before getting the last assistant message output.
    // Json mode does not handle this here, since it was already handled above
    if output_mode == Mode::Text {
        if let Some(assistant) = &last_assistant {
            // Check whether it ended normally
            match assistant.stop_reason {
                zen_ai::types::StopReason::Stop => {
                    // Output text chunk
                    let blocks = assistant_text_blocks(assistant);
                    for block in blocks {
                        writer.write_raw(&block)?;
                    }
                }
                _ => {
                    // error/aborted
                    if let Some(msg) = stop_error_message(assistant) {
                        eprintln!("Error: {msg}");
                    }
                    writer.flush().await?;
                    return Ok(1);
                }
            }
        }
    }

    writer.flush().await?;
    Ok(exit_code)
}

/// Serialize `AgentEvent` into a single JSON line (with a trailing `\n`).
/// Only used for the print mode of --mode json.
/// Pure function: reuses `AgentEvent`'s derived `Serialize` (`#[serde(tag = "type", rename_all = "snake_case")]`).
/// Returns `None` on serialization failure (caller skips this event without breaking the print flow).
///
/// # Parameters
/// - `event`: a single agent lifecycle/message/tool event
///
/// # Consumers
/// `subscribe_json_events` (internal json subscription closure) — extracted as a pure function for unit testing
pub fn serialize_event_line(event: &AgentEvent) -> Option<String> {
    let mut line = serde_json::to_string(event).ok()?;
    line.push('\n');
    Some(line)
}

/// Extract all text content blocks from the assistant message (in order of appearance, filtering out non-`ContentBlock::Text`).
/// Aligned with pi text mode's `content.type === "text"` branch; each text block becomes a separate output unit.
///
/// # Parameters
/// - `assistant`: the last assistant message
///
/// # Return Value
/// Text block list; empty Vec when no text blocks (outputs nothing)
///
/// # Consumers
/// `run_print_mode` (text mode output)
pub fn assistant_text_blocks(assistant: &AssistantMessage) -> Vec<String> {
    assistant
        .content
        .iter()
        .filter_map(|block| {
            if let ContentBlock::Text(text) = block {
                Some(text.text.clone())
            } else {
                None
            }
        })
        .collect()
}

// File: crates/zen-app/src/modes/print_mode.rs

/// Returns the error description written to stderr when an assistant message ends with `Error`/`Aborted`.
/// Prefers `error_message`, otherwise returns `Request {stop_reason}`; returns None on normal termination.
///
/// # Consumers
/// `run_print_mode` (text mode, determines exit code 1 + stderr error)
pub fn stop_error_message(assistant: &AssistantMessage) -> Option<String> {
    match assistant.stop_reason {
        zen_ai::types::StopReason::Error | zen_ai::types::StopReason::Aborted => Some(
            assistant
                .error_message
                .clone()
                .unwrap_or_else(|| format!("Request {:?}", assistant.stop_reason)),
        ),
        _ => None,
    }
}

// The shutdown signal is only awaited racily during a print run; internal to the module, so not marked pub.

/// Waits for the first shutdown signal (SIGTERM/SIGHUP) and returns the corresponding exit code (SIGTERM→143, SIGHUP→129).
/// On receiving the signal, first call `registry.kill_all()` to terminate already-tracked
/// detached child process tree (aligned with pi's `killTrackedDetachedChildren`), then returns the exit code.
///
/// # Parameters
/// - `registry`: registry of detached child processes, used to batch-terminate the child process tree on shutdown signal
///
/// # Edge Cases
/// - Unix only: `SIGHUP` uses `SignalKind::hangup()`, `SIGTERM` uses `SignalKind::terminate()`
///   (the tokio `signal` feature is already enabled in the workspace Cargo.toml)
/// - This future must be placed in the same `tokio::select!` as the prompt flow so the signal wins the race on arrival
///
/// # Return Value
/// 1xx exit code (143 / 129), for `run_print_mode` to return directly
#[cfg(unix)]
async fn await_signal_exit_code(registry: Arc<ProcessRegistry>) -> i32 {
    use tokio::signal::unix::{signal, SignalKind};

    let mut sigterm = signal(SignalKind::terminate()).expect("failed to register SIGTERM handler");
    let mut sighup = signal(SignalKind::hangup()).expect("failed to register SIGHUP handler");

    let exit_code = tokio::select! {
        _ = sigterm.recv() => 143,  // 128 + 15 (SIGTERM)
        _ = sighup.recv() => 129,   // 128 + 1 (SIGHUP)
    };

    // Clean up detached child processes
    registry.kill_all();

    exit_code
}

/// Sets up the json mode event subscription: each `AgentEvent` is serialized via `serialize_event_line` and written to `RawStdout`.
/// The subscription closure is synchronous and returns quickly (fire-and-forget; write errors ignored, per RawStdout semantics).
///
/// # Parameters
/// - `session`: the assembled session (subscribes to engine events transparently; `&self` suffices)
/// - `writer`: the `RawStdout` held by print mode
///
/// # Return Value
/// Engine unsubscribe closure (removes the listener when called; invoked when print ends)
///
/// # Edge Cases
/// - json mode only; text mode does not subscribe
/// - Subscription callbacks run synchronously on the UUID/event thread; slow operations must not be placed in the closure
///
/// # Consumers
/// `run_print_mode` (event output in json mode)
pub fn subscribe_json_events(
    session: &AgentSession,
    writer: Arc<RawStdout>,
) -> Box<dyn FnOnce() + Send> {
    session.subscribe(Box::new(move |event| {
        if let Some(line) = serialize_event_line(event) {
            let _ = writer.write_raw(&line);
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use zen_ai::types::{StopReason, TextContent, ThinkingContent, Usage};

    /// Construct a minimally viable AssistantMessage fixture (specify only case-relevant fields; others take defaults).
    fn make_assistant(
        content: Vec<ContentBlock>,
        stop_reason: StopReason,
        error_message: Option<String>,
    ) -> AssistantMessage {
        AssistantMessage {
            content,
            api: "openai-completions".into(),
            provider: "test".into(),
            model: "test-model".into(),
            response_model: None,
            response_id: None,
            usage: Usage::default(),
            stop_reason,
            error_message,
            timestamp: 0,
        }
    }

    fn text_block(text: &str) -> ContentBlock {
        ContentBlock::Text(TextContent {
            text: text.into(),
            text_signature: None,
        })
    }

    // ── serialize_event_line ─────────────────────────────────────

    #[test]
    fn test_serialize_event_line_unit_variant_ends_with_newline() {
        let line = serialize_event_line(&AgentEvent::AgentStart)
            .expect("AgentStart should be serializable");
        assert!(line.ends_with('\n'), "Line should end with newline");
    }

    #[test]
    fn test_serialize_event_line_uses_snake_case_type_tag() {
        let line = serialize_event_line(&AgentEvent::AgentStart).unwrap();
        assert!(line.contains(r#""type":"agent_start""#));
    }

    // ── assistant_text_blocks ────────────────────────────────────

    #[test]
    fn test_assistant_text_blocks_returns_text_only_in_order() {
        let a = make_assistant(
            vec![
                text_block("first"),
                text_block("second"),
                text_block("third"),
            ],
            StopReason::Stop,
            None,
        );
        assert_eq!(assistant_text_blocks(&a), vec!["first", "second", "third"]);
    }

    #[test]
    fn test_assistant_text_blocks_filters_non_text_blocks() {
        let thinking = ContentBlock::Thinking(ThinkingContent {
            thinking: "Thinking".into(),
            thinking_signature: None,
            redacted: None,
        });
        let img = ContentBlock::Image(ImageContent {
            data: "abc".into(),
            mime_type: "image/png".into(),
        });
        let a = make_assistant(
            vec![
                text_block("lead"),
                thinking,
                text_block("mid"),
                img,
                text_block("tail"),
            ],
            StopReason::Stop,
            None,
        );
        // Keep only Text blocks, preserving order (non-text blocks are filtered out)
        assert_eq!(assistant_text_blocks(&a), vec!["lead", "mid", "tail"]);
    }

    #[test]
    fn test_assistant_text_blocks_empty_when_no_text() {
        let a = make_assistant(vec![], StopReason::Stop, None);
        assert!(assistant_text_blocks(&a).is_empty());
    }

    // ── stop_error_message ───────────────────────────────────────

    #[test]
    fn test_stop_error_message_none_for_stop() {
        let a = make_assistant(vec![], StopReason::Stop, None);
        assert_eq!(stop_error_message(&a), None);
    }

    #[test]
    fn test_stop_error_message_none_for_length_and_tool_use() {
        for reason in [StopReason::Length, StopReason::ToolUse] {
            let a = make_assistant(vec![], reason, None);
            assert_eq!(stop_error_message(&a), None);
        }
    }

    #[test]
    fn test_stop_error_message_uses_error_message_when_present() {
        let a = make_assistant(vec![], StopReason::Error, Some("provider 500".into()));
        assert_eq!(stop_error_message(&a), Some("provider 500".into()));
    }

    #[test]
    fn test_stop_error_message_falls_back_to_request_reason() {
        // When there is no error_message, fall back to "Request {reason}", aligning with pi `errorMessage || "Request ${stopReason}"`
        for reason in [StopReason::Error, StopReason::Aborted] {
            let a = make_assistant(vec![], reason.clone(), None);
            assert_eq!(stop_error_message(&a), Some(format!("Request {reason:?}")));
        }
    }

    // ── The following functions depend on external state/handles, so no unit tests are added here ──
    // - `subscribe_json_events` / `run_print_mode`: require constructing a full AgentSession (Agent engine + StreamFn event stream),
    //   falling under integration testing; `RawStdout::write_raw` writes to real fd 1, so unit tests cannot assert the written content.
    // - `await_signal_exit_code`: private, unix, blocks waiting for a real system signal; requires sending a signal to the own process (libc kill) to trigger,
    //   platform-dependent and fragile; recommended to supplement manually or cover via integration tests.
}
