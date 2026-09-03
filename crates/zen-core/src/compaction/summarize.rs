use crate::compaction::utils::serialize_conversation;
use crate::compaction::CompactionError;
use crate::util::now_millis;
use tokio_util::sync::CancellationToken;
use zen_ai::api::openai_stream::StreamConfig;
use zen_ai::types::{
    AssistantMessage, AssistantMessageEvent, ContentBlockList, Context, Message, Model, StopReason,
    StreamFn, Usage, UserMessage,
};
use zen_ai::utils::retry::{retry_assistant_call, RetryPolicy};
use zen_ai::utils::text::content_text;

/// System prompt for summary tasks (copied from pi `SUMMARIZATION_SYSTEM_PROMPT`); reused by T-AG-04-2.
pub(crate) const SUMMARIZATION_SYSTEM_PROMPT: &str = r#"You are a context summarization assistant. Your task is to read a
 conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

   Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary."#;

/// Initial compaction summary instruction (copied from pi `SUMMARIZATION_PROMPT`; module-private, consumed only by generate_summary).
const SUMMARIZATION_PROMPT: &str = r#"The messages above are a conversation to summarize. Create a structured context checkpoint
 summary that another LLM will use to continue the work.

   Use this EXACT format:

   ## Goal
   [What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

   ## Constraints & Preferences
   - [Any constraints, preferences, or requirements mentioned by user]
   - [Or "(none)" if none were mentioned]

   ## Progress
   ### Done
   - [x] [Completed tasks/changes]

   ### In Progress
   - [ ] [Current work]

   ### Blocked
   - [Issues preventing progress, if any]

   ## Key Decisions
   - **[Decision]**: [Brief rationale]

   ## Next Steps
   1. [Ordered list of what should happen next]

   ## Critical Context
   - [Any data, examples, or references needed to continue]
   - [Or "(none)" if not applicable]

   Keep each section concise. Preserve exact file paths, function names, and error messages."#;

/// Iterative summary update instruction (copied from pi `UPDATE_SUMMARIZATION_PROMPT`; module-private).
const UPDATE_SUMMARIZATION_PROMPT: &str = r#"The messages above are NEW conversation messages to incorporate into the existing
 summary provided in <previous-summary> tags.

   Update the existing structured summary with new information. RULES:
   - PRESERVE all existing information from the previous summary
   - ADD new progress, decisions, and context from the new messages
   - UPDATE the Progress section: move items from "In Progress" to "Done" when completed
   - UPDATE "Next Steps" based on what was accomplished
   - PRESERVE exact file paths, function names, and error messages
   - If something is no longer relevant, you may remove it

   Use this EXACT format:

   ## Goal
   [Preserve existing goals, add new ones if the task expanded]

   ## Constraints & Preferences
   - [Preserve existing, add new ones discovered]

   ## Progress
   ### Done
   - [x] [Include previously done items AND newly completed items]

   ### In Progress
   - [ ] [Current work - update based on progress]

   ### Blocked
   - [Current blockers - remove if resolved]

   ## Key Decisions
   - **[Decision]**: [Brief rationale] (preserve all previous, add new)

   ## Next Steps
   1. [Update based on current state]

   ## Critical Context
   - [Preserve important context, add new if needed]

   Keep each section concise. Preserve exact file paths, function names, and error messages."#;

/// split-turn prefix summary instruction (copied from pi `TURN_PREFIX_SUMMARIZATION_PROMPT`; module-private).
const TURN_PREFIX_SUMMARIZATION_PROMPT: &str = r#"This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work)
 is retained.

   Summarize the prefix to provide context for the retained suffix:

   ## Original Request
   [What did the user ask for in this turn?]

   ## Early Progress
   - [Key decisions and work done in the prefix]

   ## Context for Suffix
   - [Information needed to understand the retained recent work]

   Be concise. Focus on what's needed to understand the kept suffix."#;

/// Generate (or iteratively update) a compaction summary: serialize the message sequence into conversation text and wrap it in `<conversation>` tags
/// (Add a `<previous-summary>` section when `previous_summary` is present), attach the structured summarization instructions, and make a separate LLM call.
/// "Fire" function: max_tokens = min(0.8 × reserve_tokens, model.max_tokens (0 means no limit));
/// `custom_instructions` is appended with "Additional focus:"; prompt text is copied verbatim from pi (AGENTS.md rule 10).
/// `stream_config.reasoning_effort` is preset by the caller (`compact`) according to pi's conditions
/// (model.reasoning and thinking_level ≠ off); this function no longer modifies it.
/// Returns (summary text, Usage for this call); aborted → `CompactionError::Aborted`, error → `SummarizationFailed`.
/// Consumer: `compact` (within the crate); pub(crate) enables reuse by T-AG-04-2 (branch summarization)—consumes the "summarization LLM call primitive".
/// (Consolidate the two pi generateSummary/generateSummaryWithUsage exports: zen always includes Usage; discard it when not needed.)
// Copy the parameter surface of pi generateSummaryWithUsage(...) as-is (9 parameters); reassess the parameter struct when reusing it for T-AG-04-2
#[allow(clippy::too_many_arguments)]
pub(crate) async fn generate_summary(
    current_messages: &[Message],
    stream_fn: &StreamFn,
    model: &Model,
    stream_config: StreamConfig,
    reserve_tokens: u64,
    cancel: Option<CancellationToken>,
    custom_instructions: Option<&str>,
    previous_summary: Option<&str>,
    retry: Option<RetryPolicy>,
) -> Result<(String, Usage), CompactionError> {
    let mut config = stream_config;
    config.max_tokens = Some(summary_max_tokens(reserve_tokens, 0.8, model));

    // pi: previousSummary ? UPDATE_... : SUMMARIZATION_...; custom_instructions is appended as "Additional focus:"
    let base_prompt = if previous_summary.is_some() {
        UPDATE_SUMMARIZATION_PROMPT
    } else {
        SUMMARIZATION_PROMPT
    };
    let base_prompt = match custom_instructions {
        Some(instructions) => format!("{base_prompt}\n\nAdditional focus: {instructions}"),
        None => base_prompt.to_string(),
    };

    let conversation_text = serialize_conversation(current_messages);
    let mut prompt_text = format!("<conversation>\n{conversation_text}\n</conversation>\n\n");
    if let Some(previous) = previous_summary {
        prompt_text.push_str(&format!(
            "<previous-summary>\n{previous}\n</previous-summary>\n\n"
        ));
    }
    prompt_text.push_str(&base_prompt);

    let context = summarization_context(&prompt_text);

    let response = retry_assistant_call(
        || {
            collect_stream_response(
                stream_fn,
                model,
                &context,
                Some(config.clone()),
                cancel.clone(),
            )
        },
        retry,
        cancel.clone(),
    )
    .await;

    finish_summary_response(response, "Summarization aborted", "Summarization failed")
}

// ── Private items (within summarize.rs, not pub)──

/// Initiates a streaming call via StreamFn and collects events until a terminal state.
/// The event stream already includes accumulated partial results (aligned with agent_loop's existing consumption pattern: only the latest state is tracked per event),
/// `Done`/`Error` return typed messages directly; if StreamFn fails to establish the stream or the stream ends without a terminal state → construct an Error terminal-state message.
/// Produces a complete AssistantMessage for `retry_assistant_call` to consume.
async fn collect_stream_response(
    stream_fn: &StreamFn,
    model: &Model,
    context: &Context,
    config: Option<StreamConfig>,
    cancel: Option<CancellationToken>,
) -> AssistantMessage {
    let mut receiver = match stream_fn(model, context, config, cancel).await {
        Ok(receiver) => receiver,
        Err(err) => return error_terminal_message(model, err.to_string()),
    };

    let mut latest: Option<AssistantMessage> = None;
    while let Some(event) = receiver.recv().await {
        match event {
            AssistantMessageEvent::Done { message, .. } => return message,
            AssistantMessageEvent::Error { error, .. } => return error,
            AssistantMessageEvent::Start { partial }
            | AssistantMessageEvent::TextStart { partial, .. }
            | AssistantMessageEvent::TextDelta { partial, .. }
            | AssistantMessageEvent::TextEnd { partial, .. }
            | AssistantMessageEvent::ThinkingStart { partial, .. }
            | AssistantMessageEvent::ThinkingDelta { partial, .. }
            | AssistantMessageEvent::ThinkingEnd { partial, .. }
            | AssistantMessageEvent::ToolcallStart { partial, .. }
            | AssistantMessageEvent::ToolcallDelta { partial, .. }
            | AssistantMessageEvent::ToolcallEnd { partial, .. } => latest = Some(partial),
        }
    }

    // The stream is interrupted before a terminal event (protocol violation) → use an Error terminal state as a fallback and pass it to the retry/error path
    let mut fallback = latest.unwrap_or_else(|| error_terminal_message(model, "empty stream"));
    fallback.stop_reason = StopReason::Error;
    fallback.error_message = Some("stream ended without terminal event".to_string());
    fallback
}

/// Turn-prefix summary for split-turn scenarios (`TURN_PREFIX_SUMMARIZATION_PROMPT`, module-private constant);
/// The max_tokens upper limit is 0.5 × reserve_tokens (aligned with pi). Consumer: `compact` (within the crate).
pub(crate) async fn generate_turn_prefix_summary(
    messages: &[Message],
    stream_fn: &StreamFn,
    model: &Model,
    stream_config: StreamConfig,
    reserve_tokens: u64,
    cancel: Option<CancellationToken>,
    retry: Option<RetryPolicy>,
) -> Result<(String, Usage), CompactionError> {
    let mut config = stream_config;
    config.max_tokens = Some(summary_max_tokens(reserve_tokens, 0.5, model));

    let conversation_text = serialize_conversation(messages);
    let prompt_text =
           format!("<conversation>\n{conversation_text}\n</conversation>\n\n{TURN_PREFIX_SUMMARIZATION_PROMPT}");
    let context = summarization_context(&prompt_text);

    let response = retry_assistant_call(
        || {
            collect_stream_response(
                stream_fn,
                model,
                &context,
                Some(config.clone()),
                cancel.clone(),
            )
        },
        retry,
        cancel.clone(),
    )
    .await;

    finish_summary_response(
        response,
        "Turn prefix summarization aborted",
        "Turn prefix summarization failed",
    )
}

/// Combines the Usage from two summarization calls (split-turn: history segment + turn-prefix segment; aligned with pi `combineUsage`,
/// Includes the "sum if either is present" rule for cache_write_1h/reasoning). Consumer: `compact`.
pub(crate) fn combine_usage(first: &Usage, second: &Usage) -> Usage {
    let sum_if_either = |a: Option<u64>, b: Option<u64>| -> Option<u64> {
        (a.is_some() || b.is_some()).then(|| a.unwrap_or(0) + b.unwrap_or(0))
    };
    Usage {
        input: first.input + second.input,
        output: first.output + second.output,
        cache_read: first.cache_read + second.cache_read,
        cache_write: first.cache_write + second.cache_write,
        cache_write_1h: sum_if_either(first.cache_write_1h, second.cache_write_1h),
        reasoning: sum_if_either(first.reasoning, second.reasoning),
        total_tokens: first.total_tokens + second.total_tokens,
        cost: zen_ai::types::Cost {
            input: first.cost.input + second.cost.input,
            output: first.cost.output + second.cost.output,
            cache_read: first.cost.cache_read + second.cost.cache_read,
            cache_write: first.cost.cache_write + second.cost.cache_write,
            total: first.cost.total + second.cost.total,
        },
    }
}

/// Output limit for summarization calls: the smaller of floor(ratio × reserve_tokens) and model.max_tokens;
/// Treat max_tokens = 0 as unlimited (resolve pi's `Number.POSITIVE_INFINITY` sentinel by conditionally omitting it).
fn summary_max_tokens(reserve_tokens: u64, ratio: f64, model: &Model) -> u64 {
    let budget = (ratio * reserve_tokens as f64).floor() as u64;
    if model.max_tokens > 0 {
        budget.min(model.max_tokens)
    } else {
        budget
    }
}

/// Single-message Context for summarization calls: the system prompt is fixed to SUMMARIZATION_SYSTEM_PROMPT,
/// The sole user message carries the assembled prompt text (equivalent to a single text block in a pi array ↔ zen `ContentBlockList::Text`),
/// Carries no tools (pi completeSimple has no tools).
fn summarization_context(prompt_text: &str) -> Context {
    Context {
        system_prompt: Some(SUMMARIZATION_SYSTEM_PROMPT.to_string()),
        messages: vec![Message::User(UserMessage {
            content: ContentBlockList::Text(prompt_text.to_string()),
            timestamp: now_millis(),
        })],
        tools: None,
    }
}

/// Terminal-state dispatch: aborted → `Aborted` (the stub has no payload; errorMessage details carried by pi are not retained),
/// error → `SummarizationFailed(details)`, with all other stop reasons treated as success
/// (Aligns with pi, which special-cases only the aborted/error branches; text extraction uses contentText with "\n" as the default separator).
fn finish_summary_response(
    response: AssistantMessage,
    aborted_label: &str,
    failed_label: &str,
) -> Result<(String, Usage), CompactionError> {
    match response.stop_reason {
        StopReason::Aborted => {
            let _ = aborted_label; // pi uses this as fallback text; zen Aborted has no payload, so the placeholder prevents accidental removal
            Err(CompactionError::Aborted)
        }
        StopReason::Error => Err(CompactionError::SummarizationFailed(format!(
            "{failed_label}: {}",
            response.error_message.as_deref().unwrap_or("Unknown error")
        ))),
        _ => Ok((content_text(&response.content, "\n"), response.usage)),
    }
}

/// Error terminal-state message when StreamFn fails to create a stream or the stream is empty (mirrors the agent_loop construction).
fn error_terminal_message(model: &Model, message: impl Into<String>) -> AssistantMessage {
    AssistantMessage {
        content: vec![],
        api: model.api.to_string(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        usage: zen_ai::types::Usage::default(),
        stop_reason: StopReason::Error,
        error_message: Some(message.into()),
        timestamp: now_millis(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use tokio::sync::mpsc;
    use zen_ai::api::openai_stream::StreamError;
    use zen_ai::provider::Api;
    use zen_ai::types::{ContentBlock, NonErrorStopReason, TextContent};

    // ── construction helpers ─────────────────────────────────────────────

    fn test_model(max_tokens: u64) -> Model {
        Model {
            id: "test-model".to_string(),
            name: None,
            provider: "deepseek".to_string(),
            base_url: "https://example.test".to_string(),
            context_window: 100_000,
            max_tokens,
            reasoning: false,
            headers: None,
            cost_rates: None,
            api: Api::OpenAiCompletions,
        }
    }

    fn assistant_message(text: &str, stop_reason: StopReason) -> AssistantMessage {
        AssistantMessage {
            content: if text.is_empty() {
                vec![]
            } else {
                vec![ContentBlock::Text(TextContent {
                    text: text.to_string(),
                    text_signature: None,
                })]
            },
            api: "openai-completions".to_string(),
            provider: "deepseek".to_string(),
            model: "test-model".to_string(),
            response_model: None,
            response_id: None,
            usage: Usage::default(),
            stop_reason,
            error_message: None,
            timestamp: 0,
        }
    }

    fn done_event(text: &str) -> AssistantMessageEvent {
        AssistantMessageEvent::Done {
            reason: NonErrorStopReason::Stop,
            message: assistant_message(text, StopReason::Stop),
        }
    }

    fn base_config() -> StreamConfig {
        StreamConfig {
            api_key: None,
            headers: None,
            tool_choice: None,
            reasoning_effort: None,
            max_tokens: None,
            temperature: None,
            timeout_ms: None,
            session_id: None,
        }
    }

    fn user_msg(text: &str) -> Message {
        Message::User(UserMessage {
            content: ContentBlockList::Text(text.to_string()),
            timestamp: 0,
        })
    }

    type Captured = (String, String, Option<u64>); // (system_prompt, user prompt text, config.max_tokens)

    /// Fake StreamFn: replays a predefined event stream and records the (system_prompt, prompt, max_tokens) received on each call.
    /// StreamFn is an injected in-memory closure, not external IO—the existing consumption pattern in this file (event stream + channel) serves as its test double.
    fn fake_stream_fn(
        events: Vec<AssistantMessageEvent>,
        captured: Arc<Mutex<Vec<Captured>>>,
    ) -> StreamFn {
        Arc::new(move |_model, context, config, _cancel| {
            let system = context.system_prompt.clone().unwrap_or_default();
            let prompt = match context.messages.first() {
                Some(Message::User(u)) => match &u.content {
                    ContentBlockList::Text(t) => t.clone(),
                    _ => String::new(),
                },
                _ => String::new(),
            };
            captured.lock().unwrap().push((
                system,
                prompt,
                config.as_ref().and_then(|c| c.max_tokens),
            ));
            let (tx, rx) = mpsc::unbounded_channel();
            for event in &events {
                let _ = tx.send(event.clone());
            }
            drop(tx);
            Box::pin(async move { Ok(rx) })
        })
    }

    /// A StreamFn that always fails to create the stream (protocol-level Err branch)
    fn failing_stream_fn(calls: Arc<AtomicUsize>) -> StreamFn {
        Arc::new(move |_model, _context, _config, _cancel| {
            calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Err(StreamError::NoApiKey) })
        })
    }

    // ── summary_max_tokens (private, directly unit-tested)──────────────────

    #[test]
    fn test_summary_max_tokens_floor_of_ratio() {
        assert_eq!(summary_max_tokens(1_000, 0.8, &test_model(0)), 800);
        // 0.5 × 1001 = 500.5 → floor 500
        assert_eq!(summary_max_tokens(1_001, 0.5, &test_model(0)), 500);
    }

    #[test]
    fn test_summary_max_tokens_capped_by_model() {
        assert_eq!(summary_max_tokens(1_000, 0.8, &test_model(100)), 100);
    }

    #[test]
    fn test_summary_max_tokens_zero_model_max_is_uncapped() {
        // Treat model.max_tokens = 0 as unlimited (resolving pi's POSITIVE_INFINITY sentinel)
        assert_eq!(summary_max_tokens(1_000, 0.8, &test_model(0)), 800);
    }

    // ── summarization_context (private, directly unit-tested)────────────────

    #[test]
    fn test_summarization_context_shape() {
        let context = summarization_context("the prompt");
        assert_eq!(
            context.system_prompt.as_deref(),
            Some(SUMMARIZATION_SYSTEM_PROMPT)
        );
        assert_eq!(context.messages.len(), 1);
        assert!(matches!(context.messages[0], Message::User(_)));
        match &context.messages[0] {
            Message::User(u) => match &u.content {
                ContentBlockList::Text(t) => assert_eq!(t, "the prompt"),
                _ => panic!("should be plain text content"),
            },
            _ => panic!("should be User message"),
        }
        assert!(context.tools.is_none());
    }

    // ── finish_summary_response (private, directly unit-tested)──────────────

    #[test]
    fn test_finish_summary_response_success_joins_text_blocks() {
        let mut response = assistant_message("first", StopReason::Stop);
        response.content.push(ContentBlock::Text(TextContent {
            text: "second".to_string(),
            text_signature: None,
        }));
        response.usage.input = 7;
        let (text, usage) =
            finish_summary_response(response, "aborted", "failed").expect("Stop should succeed");
        assert_eq!(text, "first\nsecond"); // contentText is separated by "\n" by default
        assert_eq!(usage.input, 7);
    }

    #[test]
    fn test_finish_summary_response_aborted_maps_to_compaction_error() {
        let result = finish_summary_response(
            assistant_message("x", StopReason::Aborted),
            "aborted",
            "failed",
        );
        assert!(matches!(result, Err(CompactionError::Aborted)));
    }

    #[test]
    fn test_finish_summary_response_error_includes_error_message() {
        let mut response = assistant_message("x", StopReason::Error);
        response.error_message = Some("boom".to_string());
        let result = finish_summary_response(response, "aborted", "Summarization failed");
        assert_eq!(
            result.unwrap_err().to_string(),
            "summarization failed: Summarization failed: boom"
        );
        // Fall back to "Unknown error" when error_message is missing
        let response = assistant_message("x", StopReason::Error);
        let result = finish_summary_response(response, "aborted", "Summarization failed");
        assert_eq!(
            result.unwrap_err().to_string(),
            "summarization failed: Summarization failed: Unknown error"
        );
    }

    // ── combine_usage ────────────────────────────────────────

    #[test]
    fn test_combine_usage_sums_all_fields() {
        let first = Usage {
            input: 1,
            output: 2,
            cache_read: 3,
            cache_write: 4,
            cache_write_1h: Some(1),
            reasoning: Some(2),
            total_tokens: 10,
            cost: zen_ai::types::Cost {
                input: 1.0,
                output: 2.0,
                cache_read: 3.0,
                cache_write: 4.0,
                total: 10.0,
            },
        };
        let second = Usage {
            input: 10,
            output: 20,
            cache_read: 30,
            cache_write: 40,
            cache_write_1h: Some(2),
            reasoning: None,
            total_tokens: 100,
            cost: zen_ai::types::Cost {
                input: 10.0,
                output: 20.0,
                cache_read: 30.0,
                cache_write: 40.0,
                total: 100.0,
            },
        };
        let combined = combine_usage(&first, &second);
        assert_eq!(combined.input, 11);
        assert_eq!(combined.output, 22);
        assert_eq!(combined.cache_read, 33);
        assert_eq!(combined.cache_write, 44);
        assert_eq!(combined.total_tokens, 110);
        assert_eq!(combined.cache_write_1h, Some(3)); // Sum if any are present
        assert_eq!(combined.reasoning, Some(2)); // Treat None as 0
        assert!((combined.cost.total - 110.0).abs() < f64::EPSILON);
        assert!((combined.cost.input - 11.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_combine_usage_optional_both_none_stays_none() {
        let combined = combine_usage(&Usage::default(), &Usage::default());
        assert_eq!(combined.cache_write_1h, None);
        assert_eq!(combined.reasoning, None);
    }

    // ── error_terminal_message (private, directly unit-tested)───────────────

    #[test]
    fn test_error_terminal_message_maps_model_fields() {
        let message = error_terminal_message(&test_model(42), "something broke");
        assert_eq!(message.stop_reason, StopReason::Error);
        assert_eq!(message.error_message.as_deref(), Some("something broke"));
        assert_eq!(message.model, "test-model");
        assert_eq!(message.provider, "deepseek");
        assert_eq!(message.api, "openai-completions");
        assert!(message.content.is_empty());
    }

    // ── collect_stream_response（async，fake StreamFn）────────

    #[tokio::test]
    async fn test_collect_stream_response_done_returns_message() {
        let stream_fn = fake_stream_fn(
            vec![
                AssistantMessageEvent::Start {
                    partial: assistant_message("partial", StopReason::Stop),
                },
                done_event("final"),
            ],
            Arc::new(Mutex::new(Vec::new())),
        );
        let response = collect_stream_response(
            &stream_fn,
            &test_model(0),
            &summarization_context("p"),
            None,
            None,
        )
        .await;
        assert_eq!(response.stop_reason, StopReason::Stop);
        match &response.content[0] {
            ContentBlock::Text(t) => assert_eq!(t.text, "final"),
            _ => panic!("should be text block"),
        }
    }

    #[tokio::test]
    async fn test_collect_stream_response_stream_fn_failure_yields_error_terminal() {
        let stream_fn = failing_stream_fn(Arc::new(AtomicUsize::new(0)));
        let response = collect_stream_response(
            &stream_fn,
            &test_model(0),
            &summarization_context("p"),
            None,
            None,
        )
        .await;
        assert_eq!(response.stop_reason, StopReason::Error);
        assert_eq!(
            response.error_message.as_deref(),
            Some("API key not configured")
        );
    }

    #[tokio::test]
    async fn test_collect_stream_response_interrupted_stream_falls_back_to_error() {
        // Close the stream after only non-terminal events → construct an Error terminal state from the latest partial (protocol violation fallback)
        let stream_fn = fake_stream_fn(
            vec![AssistantMessageEvent::Start {
                partial: assistant_message("partial", StopReason::Stop),
            }],
            Arc::new(Mutex::new(Vec::new())),
        );
        let response = collect_stream_response(
            &stream_fn,
            &test_model(0),
            &summarization_context("p"),
            None,
            None,
        )
        .await;
        assert_eq!(response.stop_reason, StopReason::Error);
        assert_eq!(
            response.error_message.as_deref(),
            Some("stream ended without terminal event")
        );
    }

    // ── generate_summary (async, fake StreamFn + prompt capture)─

    #[tokio::test]
    async fn test_generate_summary_builds_first_run_prompt_and_overrides_max_tokens() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let stream_fn = fake_stream_fn(vec![done_event("the summary")], captured.clone());
        let messages = vec![user_msg("hello"), assistant_entry_msg()];
        let (text, usage) = generate_summary(
            &messages,
            &stream_fn,
            &test_model(0),
            base_config(),
            1_000,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("Stop should succeed");
        assert_eq!(text, "the summary");
        assert_eq!(usage, Usage::default());
        let (system, prompt, max_tokens) = &captured.lock().unwrap()[0];
        assert_eq!(system, SUMMARIZATION_SYSTEM_PROMPT);
        assert!(prompt.contains("<conversation>\n[User]: hello"));
        assert!(prompt.contains("[Assistant]: hi"));
        assert!(prompt.contains("</conversation>"));
        assert!(prompt.contains("conversation to summarize")); // Initial summarization instruction
        assert!(!prompt.contains("<previous-summary>"));
        assert!(!prompt.contains("Additional focus"));
        // max_tokens = floor(0.8 × 1000) = 800 (model.max_tokens = 0 means no limit)
        assert_eq!(*max_tokens, Some(800));
    }

    #[tokio::test]
    async fn test_generate_summary_appends_custom_instructions() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let stream_fn = fake_stream_fn(vec![done_event("s")], captured.clone());
        generate_summary(
            &[user_msg("hello")],
            &stream_fn,
            &test_model(0),
            base_config(),
            1_000,
            None,
            Some("focus on tests"),
            None,
            None,
        )
        .await
        .expect("Stop should succeed");
        let (_, prompt, _) = &captured.lock().unwrap()[0];
        assert!(prompt.contains("Additional focus: focus on tests"));
    }

    #[tokio::test]
    async fn test_generate_summary_uses_update_prompt_with_previous_summary() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let stream_fn = fake_stream_fn(vec![done_event("s")], captured.clone());
        generate_summary(
            &[user_msg("hello")],
            &stream_fn,
            &test_model(0),
            base_config(),
            1_000,
            None,
            None,
            Some("old summary"),
            None,
        )
        .await
        .expect("Stop should succeed");
        let (_, prompt, _) = &captured.lock().unwrap()[0];
        assert!(prompt.contains("NEW conversation messages")); // Iterative update instruction
        assert!(prompt.contains("<previous-summary>\nold summary\n</previous-summary>"));
    }

    #[tokio::test]
    async fn test_generate_summary_aborted_returns_compaction_error() {
        let stream_fn = fake_stream_fn(
            vec![AssistantMessageEvent::Done {
                reason: NonErrorStopReason::Stop,
                message: assistant_message("x", StopReason::Aborted),
            }],
            Arc::new(Mutex::new(Vec::new())),
        );
        let result = generate_summary(
            &[user_msg("hello")],
            &stream_fn,
            &test_model(0),
            base_config(),
            1_000,
            None,
            None,
            None,
            None,
        )
        .await;
        assert!(matches!(result, Err(CompactionError::Aborted)));
    }

    #[tokio::test]
    async fn test_generate_summary_error_stop_returns_summarization_failed() {
        let mut error_message = assistant_message("x", StopReason::Error);
        error_message.error_message = Some("boom".to_string());
        let stream_fn = fake_stream_fn(
            vec![AssistantMessageEvent::Done {
                reason: NonErrorStopReason::Stop,
                message: error_message,
            }],
            Arc::new(Mutex::new(Vec::new())),
        );
        let result = generate_summary(
            &[user_msg("hello")],
            &stream_fn,
            &test_model(0),
            base_config(),
            1_000,
            None,
            None,
            None,
            None,
        )
        .await;
        assert_eq!(
            result.unwrap_err().to_string(),
            "summarization failed: Summarization failed: boom"
        );
    }

    #[tokio::test]
    async fn test_generate_summary_stream_fn_failure_yields_summarization_failed() {
        // Persistent protocol-layer failure (including retry policy) → Error terminal state → SummarizationFailed (with underlying error details)
        let stream_fn = failing_stream_fn(Arc::new(AtomicUsize::new(0)));
        let result = generate_summary(
            &[user_msg("hello")],
            &stream_fn,
            &test_model(0),
            base_config(),
            1_000,
            None,
            None,
            None,
            Some(RetryPolicy {
                enabled: true,
                max_retries: 2,
                base_delay_ms: 1,
            }),
        )
        .await;
        assert_eq!(
            result.unwrap_err().to_string(),
            "summarization failed: Summarization failed: API key not configured"
        );
    }

    // ── generate_turn_prefix_summary（async）─────────────────

    #[tokio::test]
    async fn test_generate_turn_prefix_summary_uses_half_budget_and_prefix_prompt() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let stream_fn = fake_stream_fn(vec![done_event("prefix summary")], captured.clone());
        let (text, _) = generate_turn_prefix_summary(
            &[user_msg("turn question")],
            &stream_fn,
            &test_model(0),
            base_config(),
            1_000,
            None,
            None,
        )
        .await
        .expect("Stop should succeed");
        assert_eq!(text, "prefix summary");
        let (system, prompt, max_tokens) = &captured.lock().unwrap()[0];
        assert_eq!(system, SUMMARIZATION_SYSTEM_PROMPT);
        assert!(prompt.contains("[User]: turn question"));
        assert!(prompt.contains("PREFIX of a turn")); // Turn prefix instruction
                                                      // max_tokens = floor(0.5 × 1000) = 500 (distinct from generate_summary's 0.8)
        assert_eq!(*max_tokens, Some(500));
    }

    // assistant text message ("hi", 1-token level)
    fn assistant_entry_msg() -> Message {
        Message::Assistant(AssistantMessage {
            content: vec![ContentBlock::Text(TextContent {
                text: "hi".to_string(),
                text_signature: None,
            })],
            api: "openai-completions".to_string(),
            provider: "deepseek".to_string(),
            model: "test-model".to_string(),
            response_model: None,
            response_id: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 0,
        })
    }
}
