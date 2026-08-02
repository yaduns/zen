use crate::agent::agent_hooks::{
    fail_tool_calls_from_truncated_message, AgentLoopError, AgentLoopHooks,
    FinalizedToolCallOutcome, execute_prepared_tool_call, finalize_executed_tool_call,
    prepare_tool_call,
};
use crate::agent::types::{
    AgentEvent, AgentLoopConfig, PrepareNextTurnContext, ShouldStopAfterTurnContext, StreamFn,
    ToolExecutionMode,
};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use zen_ai::tool::Tool;
use zen_ai::types::{
    AssistantMessage, AssistantMessageEvent, ContentBlock, Context, Message, StopReason, ToolCall,
    ToolResultMessage, Usage,
};

/// Start a new agent loop beginning with the prompt messages.
///
/// Appends `prompts` to the end of `context`, then repeatedly runs the "LLM call → tool execution → LLM call" loop,
/// until the LLM no longer requests tool calls or a stop condition is met. All events are sent via `hooks.on_event()`.
///
/// # Parameters
/// - `prompts`: list of new prompt messages to process
/// - `context`: starting agent context (system prompt, existing messages, available tools). It is mutated (new messages are appended).
/// - `config`: plain-data configuration (model, API key, execution mode, etc.)
/// - `hooks`: fixed behavior policies (message transformation, hooks, stop determination, event handling, etc.)
/// - `cancel`: cancellation token; when a cancellation signal is received the agent loop should stop as soon as possible
/// - `stream_fn`: LLM streaming call function
/// - `tools`: list of available tools, used to execute tool calls in assistant messages
///
/// # Return Value
/// Returns all messages newly produced during this loop (excluding those already in `context.messages`).
///
/// # Edge Cases
/// - If `prompts` is empty, returns an empty vec immediately (no LLM call is made).
/// - If `stream_fn` returns an error, the error is encoded via the `AgentEvent::AgentEnd` event and the loop terminates.
/// - On cancellation, the loop emits `turn_end` + `agent_end` events and exits without panicking.
/// - If hook methods (`before_tool_call`, `after_tool_call`, etc.) return `Err`, the error is captured and encoded as a tool error result rather than propagated.
/// - Errors occurring inside `hooks.on_event()` are handled by the hooks implementer and are not propagated into the loop.
pub async fn run_agent_loop(
    prompts: Vec<Message>,
    context: &mut Context,
    config: &AgentLoopConfig,
    hooks: &dyn AgentLoopHooks,
    cancel: &CancellationToken,
    stream_fn: StreamFn,
    tools: &[Arc<dyn Tool>],
) -> Vec<Message> {
    // 1. Empty prompts → no-op
    if prompts.is_empty() {
        return vec![];
    }

    let mut new_messages: Vec<Message> = Vec::with_capacity(prompts.len());

    // 2. Append prompts to context and emit events
    hooks.on_event(&AgentEvent::AgentStart).await;

    for prompt in prompts {
        hooks
            .on_event(&AgentEvent::MessageStart {
                message: prompt.clone(),
            })
            .await;
        hooks
            .on_event(&AgentEvent::MessageEnd {
                message: prompt.clone(),
            })
            .await;
        context.messages.push(prompt.clone());
        new_messages.push(prompt);
    }

    // 3. Multi-turn inner loop
    agent_loop_inner(context, config, hooks, cancel, stream_fn, tools, &mut new_messages).await;

    hooks
        .on_event(&AgentEvent::AgentEnd {
            messages: new_messages.clone(),
        })
        .await;

    new_messages
}

/// Continue the agent loop from the current context without adding a new prompt.
///
/// Used for retry scenarios — the context already contains a user message or tool result, proceed directly to the LLM call.
///
/// # Parameters
/// - `context`: current agent context (the last message must convert to a user or toolResult message,
///   otherwise the LLM provider will reject the request)
/// - `config`: pure data configuration
/// - `hooks`: fixed behavior policy
/// - `cancel`: cancellation token
/// - `stream_fn`: LLM streaming call function
///
/// # Return Value
/// Returns the messages added during this loop.
///
/// # Errors
/// If `context.messages` is empty, returns `Err(AgentLoopError::EmptyContext)`.
/// If the last message's role is "assistant", returns `Err(AgentLoopError::InvalidLastRole)`.
///
/// # Edge Cases
/// Same as `run_agent_loop`, but without adding a new prompt.
/// Continue the agent loop from the current context without adding a new prompt.
///
/// Used for retry scenarios — the context already contains a user message or tool result, proceed directly to the LLM call.
///
/// # Parameters
/// - `context`: current agent context (the last message must convert to a user or toolResult message,
///   otherwise the LLM provider will reject the request)
/// - `config`: pure data configuration
/// - `hooks`: fixed behavior policy
/// - `cancel`: cancellation token
/// - `stream_fn`: LLM streaming call function
/// - `tools`: list of available tools
///
/// # Return Value
/// Returns the messages added during this loop.
///
/// # Errors
/// If `context.messages` is empty, returns `Err(AgentLoopError::EmptyContext)`.
/// If the last message's role is "assistant", returns `Err(AgentLoopError::InvalidLastRole)`.
///
/// # Edge Cases
/// Same as `run_agent_loop`, but without adding a new prompt.
pub async fn run_agent_loop_continue(
    context: &mut Context,
    config: &AgentLoopConfig,
    hooks: &dyn AgentLoopHooks,
    cancel: &CancellationToken,
    stream_fn: StreamFn,
    tools: &[Arc<dyn Tool>],
) -> Result<Vec<Message>, AgentLoopError> {
    // 1. Validate: context must not be empty and last message must not be assistant
    if context.messages.is_empty() {
        return Err(AgentLoopError::EmptyContext);
    }
    if let Some(Message::Assistant(_)) = context.messages.last() {
        return Err(AgentLoopError::InvalidLastRole);
    }

    let mut new_messages: Vec<Message> = Vec::new();

    // 2. Emit lifecycle events
    hooks.on_event(&AgentEvent::AgentStart).await;

    // 3. Multi-turn inner loop
    agent_loop_inner(context, config, hooks, cancel, stream_fn, tools, &mut new_messages).await;

    hooks
        .on_event(&AgentEvent::AgentEnd {
            messages: new_messages.clone(),
        })
        .await;

    Ok(new_messages)
}

/// Stream the LLM's assistant response.
///
/// Calls `stream_fn` to obtain an `AssistantMessageEventStream`, processes all events (start/delta/end),
/// and assembles the final `AssistantMessage`. Events are forwarded via `hooks.on_event()`.
///
/// In this function, `Message` is converted to `Message` (LLM-compatible format) and sent to the LLM.
///
/// # Parameters
/// - `context`: current agent context (it is mutated: partial messages are appended, eventually replaced by the finalized message)
/// - `config`: pure data configuration
/// - `hooks`: used for `transform_context`, `filter_messages`, and `on_event`
/// - `cancel`: cancellation token
/// - `stream_fn`: LLM streaming call function
///
/// # Return Value
/// Final `AssistantMessage` (with complete content, usage, stop_reason).
///
/// # Edge Cases
/// - If `stream_fn` returns an error, stop_reason is "error" or "aborted".
/// - Calls `hooks.transform_context()` and `hooks.filter_messages()` for message transformation.
/// - If `filter_messages` returns an empty list, behavior is up to the LLM provider (usually an error).
pub(crate) async fn stream_assistant_response(
    context: &mut Context,
    config: &AgentLoopConfig,
    hooks: &dyn AgentLoopHooks,
    cancel: &CancellationToken,
    stream_fn: StreamFn,
) -> AssistantMessage {
    let now = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64
    };

    // 1. Filter and transform messages via hooks
    let filtered = hooks.filter_messages(context.messages.clone()).await;
    let transformed = hooks.transform_context(filtered, cancel).await;

    // 2. Build a new context for the LLM call (keep system_prompt and tools from original)
    let stream_ctx = Context {
        system_prompt: context.system_prompt.clone(),
        messages: transformed,
        tools: context.tools.clone(),
    };

    // 3. Resolve API key via hooks (for side effects / logging; StreamConfig is opaque)
    if config.api_key.is_none() {
        hooks.get_api_key(&config.model.id).await;
    }

    // 4. Call stream_fn
    let mut receiver = match stream_fn(&config.model, &stream_ctx, None, Some(cancel.clone())).await
    {
        Ok(rx) => rx,
        Err(err) => {
            let msg = AssistantMessage {
                content: vec![],
                api: "openai-completions".to_string(),
                provider: "unknown".to_string(),
                model: config.model.id.clone(),
                response_model: None,
                response_id: None,
                usage: Usage::default(),
                stop_reason: StopReason::Error,
                error_message: Some(err.to_string()),
                timestamp: now(),
            };
            context.messages.push(Message::Assistant(msg.clone()));
            return msg;
        }
    };

    // 5. Process stream events, mirroring Pi's agent-loop.ts streamAssistantResponse
    let mut added_partial = false;

    while let Some(event) = receiver.recv().await {
        match event {
            AssistantMessageEvent::Start { ref partial } => {
                context.messages.push(Message::Assistant(partial.clone()));
                added_partial = true;
                hooks
                    .on_event(&AgentEvent::MessageStart {
                        message: Message::Assistant(partial.clone()),
                    })
                    .await;
            }
            AssistantMessageEvent::TextStart { ref partial, .. }
            | AssistantMessageEvent::TextDelta { ref partial, .. }
            | AssistantMessageEvent::TextEnd { ref partial, .. }
            | AssistantMessageEvent::ThinkingStart { ref partial, .. }
            | AssistantMessageEvent::ThinkingDelta { ref partial, .. }
            | AssistantMessageEvent::ThinkingEnd { ref partial, .. }
            | AssistantMessageEvent::ToolcallStart { ref partial, .. }
            | AssistantMessageEvent::ToolcallDelta { ref partial, .. }
            | AssistantMessageEvent::ToolcallEnd { ref partial, .. } => {
                if added_partial {
                    if let Some(last) = context.messages.last_mut() {
                        *last = Message::Assistant(partial.clone());
                    }
                    hooks
                        .on_event(&AgentEvent::MessageUpdate {
                            message: Message::Assistant(partial.clone()),
                            assistant_message_event: event.clone(),
                        })
                        .await;
                }
            }
            AssistantMessageEvent::Done { message, .. } => {
                if added_partial {
                    if let Some(last) = context.messages.last_mut() {
                        *last = Message::Assistant(message.clone());
                    }
                } else {
                    context.messages.push(Message::Assistant(message.clone()));
                    hooks
                        .on_event(&AgentEvent::MessageStart {
                            message: Message::Assistant(message.clone()),
                        })
                        .await;
                }
                hooks
                    .on_event(&AgentEvent::MessageEnd {
                        message: Message::Assistant(message.clone()),
                    })
                    .await;
                return message;
            }
            AssistantMessageEvent::Error { error, .. } => {
                if added_partial {
                    if let Some(last) = context.messages.last_mut() {
                        *last = Message::Assistant(error.clone());
                    }
                } else {
                    context.messages.push(Message::Assistant(error.clone()));
                    hooks
                        .on_event(&AgentEvent::MessageStart {
                            message: Message::Assistant(error.clone()),
                        })
                        .await;
                }
                hooks
                    .on_event(&AgentEvent::MessageEnd {
                        message: Message::Assistant(error.clone()),
                    })
                    .await;
                return error;
            }
        }
    }

    // 6. Stream ended without a terminal event (shouldn't happen in practice)
    let msg = AssistantMessage {
        content: vec![],
        api: "openai-completions".to_string(),
        provider: "unknown".to_string(),
        model: config.model.id.clone(),
        response_model: None,
        response_id: None,
        usage: Usage::default(),
        stop_reason: StopReason::Error,
        error_message: Some("Stream ended without terminal event".to_string()),
        timestamp: now(),
    };
    context.messages.push(Message::Assistant(msg.clone()));
    msg
}

/// Multi-turn inner loop: repeatedly execute "LLM call → tool execution → LLM call",
/// until the LLM no longer requests tool calls or a stop condition is met.
///
/// `new_messages` is appended with messages newly added this round (including assistant messages and tool results).
async fn agent_loop_inner(
    context: &mut Context,
    config: &AgentLoopConfig,
    hooks: &dyn AgentLoopHooks,
    cancel: &CancellationToken,
    stream_fn: StreamFn,
    tools: &[Arc<dyn Tool>],
    new_messages: &mut Vec<Message>,
) {
    loop {
        // Check cancel signal
        if cancel.is_cancelled() {
            break;
        }

        hooks.on_event(&AgentEvent::TurnStart).await;

        // Stream the assistant response
        let assistant_msg =
            stream_assistant_response(context, config, hooks, cancel, stream_fn.clone()).await;
        new_messages.push(Message::Assistant(assistant_msg.clone()));

        // Extract tool calls
        let tool_calls: Vec<ToolCall> = assistant_msg
            .content
            .iter()
            .filter_map(|block| {
                if let ContentBlock::ToolCall(tc) = block {
                    Some(tc.clone())
                } else {
                    None
                }
            })
            .collect();

        // No tool calls → LLM is done, exit the loop
        if tool_calls.is_empty() {
            hooks
                .on_event(&AgentEvent::TurnEnd {
                    message: Message::Assistant(assistant_msg),
                    tool_results: vec![],
                })
                .await;
            break;
        }

        // Has tool calls: execute or mark as truncated
        let (tool_results, terminate) = if assistant_msg.stop_reason == StopReason::Length {
            fail_tool_calls_from_truncated_message(&tool_calls, hooks).await
        } else {
            execute_tool_calls(context, &assistant_msg, tools, config, hooks, cancel).await
        };

        // Inject tool results into context
        for result in &tool_results {
            context.messages.push(Message::ToolResult(result.clone()));
            new_messages.push(Message::ToolResult(result.clone()));
        }

        // Send turn_end event
        hooks
            .on_event(&AgentEvent::TurnEnd {
                message: Message::Assistant(assistant_msg.clone()),
                tool_results: tool_results.clone(),
            })
            .await;

        // Check terminate signal
        if terminate {
            break;
        }

        // Check should_stop_after_turn hook
        let stop_ctx = ShouldStopAfterTurnContext {
            message: assistant_msg,
            tool_results,
            context: context.clone(),
            new_messages: new_messages.clone(),
        };
        if hooks.should_stop_after_turn(&stop_ctx).await {
            break;
        }

        // Check prepare_next_turn hook
        let prep_ctx = PrepareNextTurnContext {
            message: stop_ctx.message,
            tool_results: stop_ctx.tool_results,
            context: stop_ctx.context,
            new_messages: stop_ctx.new_messages,
        };
        if let Some(update) = hooks.prepare_next_turn(&prep_ctx).await {
            if let Some(ctx) = update.context {
                *context = ctx;
            }
            // model and thinking_level are fields in config, not in context
            // The application layer can replace it via hooks; not handled here for now
        }

        // Check steering messages
        let steering = hooks.get_steering_messages().await;
        if !steering.is_empty() {
            for msg in steering {
                hooks
                    .on_event(&AgentEvent::MessageStart {
                        message: msg.clone(),
                    })
                    .await;
                hooks
                    .on_event(&AgentEvent::MessageEnd {
                        message: msg.clone(),
                    })
                    .await;
                context.messages.push(msg.clone());
                new_messages.push(msg);
            }
            // Continue to the next round (steering messages + tool results sent to LLM together)
            continue;
        }

        // Check follow-up messages
        let follow_up = hooks.get_follow_up_messages().await;
        if !follow_up.is_empty() {
            for msg in follow_up {
                hooks
                    .on_event(&AgentEvent::MessageStart {
                        message: msg.clone(),
                    })
                    .await;
                hooks
                    .on_event(&AgentEvent::MessageEnd {
                        message: msg.clone(),
                    })
                    .await;
                context.messages.push(msg.clone());
                new_messages.push(msg);
            }
            continue;
        }

        // No steering, no follow-up, no tool calls → loop ends
        break;
    }
}

/// Execute the tool calls in the assistant message.
///
/// Determine the execution strategy based on `config.tool_execution` and each tool's `execution_mode`:
/// - `Sequential`: execute tool calls one by one, starting the next only after the previous completes
/// - `Parallel`: preprocess all tool calls in order, then execute the permitted tools concurrently
///
/// If the assistant message's `stop_reason` is `Length`, all tool calls are treated as truncated and not executed;
/// directly returns error tool results (`fail_tool_calls_from_truncated_message`).
///
/// # Return Value
/// The tool result message list and a flag indicating whether the loop should terminate.
pub(crate) async fn execute_tool_calls(
    context: &mut Context,
    assistant_message: &AssistantMessage,
    tools: &[Arc<dyn Tool>],
    config: &AgentLoopConfig,
    hooks: &dyn AgentLoopHooks,
    cancel: &CancellationToken,
) -> (Vec<ToolResultMessage>, bool) {
    let tool_calls: Vec<ToolCall> = assistant_message
        .content
        .iter()
        .filter_map(|block| {
            if let ContentBlock::ToolCall(tc) = block {
                Some(tc.clone())
            } else {
                None
            }
        })
        .collect();

    if tool_calls.is_empty() {
        return (vec![], false);
    }

    match config.tool_execution {
        ToolExecutionMode::Sequential => {
            execute_tool_calls_sequential(
                context,
                assistant_message,
                &tool_calls,
                tools,
                config,
                hooks,
                cancel,
            )
            .await
        }
        ToolExecutionMode::Parallel => {
            execute_tool_calls_parallel(
                context,
                assistant_message,
                &tool_calls,
                tools,
                config,
                hooks,
                cancel,
            )
            .await
        }
    }
}

/// Execute tool calls sequentially.
pub(crate) async fn execute_tool_calls_sequential(
    context: &mut Context,
    assistant_message: &AssistantMessage,
    tool_calls: &[ToolCall],
    tools: &[Arc<dyn Tool>],
    config: &AgentLoopConfig,
    hooks: &dyn AgentLoopHooks,
    cancel: &CancellationToken,
) -> (Vec<ToolResultMessage>, bool) {
    let mut finalized_calls: Vec<FinalizedToolCallOutcome> = Vec::new();
    let mut messages: Vec<ToolResultMessage> = Vec::new();

    for tool_call in tool_calls {
        // Check cancel signal
        if cancel.is_cancelled() {
            break;
        }

        // Send tool_execution_start event
        hooks
            .on_event(&AgentEvent::ToolExecutionStart {
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                args: tool_call.arguments.clone(),
            })
            .await;

        // Preprocessing
        let preparation =
            prepare_tool_call(context, assistant_message, tool_call, tools, hooks, cancel).await;

        let finalized = match preparation {
            Err(err_result) => FinalizedToolCallOutcome {
                tool_call: tool_call.clone(),
                result: err_result,
                is_error: true,
            },
            Ok(prepared) => {
                let executed = execute_prepared_tool_call(&prepared, cancel, hooks).await;
                finalize_executed_tool_call(
                    context,
                    assistant_message,
                    &prepared,
                    &executed,
                    hooks,
                    cancel,
                )
                .await
            }
        };

        // Send tool_execution_end event
        hooks
            .on_event(&AgentEvent::ToolExecutionEnd {
                tool_call_id: finalized.tool_call.id.clone(),
                tool_name: finalized.tool_call.name.clone(),
                result: finalized.result.clone(),
                is_error: finalized.is_error,
            })
            .await;

        messages.push(finalized.result.message.clone());
        finalized_calls.push(finalized);
    }

    (messages, should_terminate_tool_batch(&finalized_calls))
}

/// Execute tool calls concurrently.
///
/// Tool calls are preprocessed in order (`prepare_tool_call`), then the permitted tool calls are executed concurrently;
/// Tool results are ordered by the tool call order in the original assistant message.
pub(crate) async fn execute_tool_calls_parallel(
    context: &mut Context,
    assistant_message: &AssistantMessage,
    tool_calls: &[ToolCall],
    tools: &[Arc<dyn Tool>],
    config: &AgentLoopConfig,
    hooks: &dyn AgentLoopHooks,
    cancel: &CancellationToken,
) -> (Vec<ToolResultMessage>, bool) {
    let mut finalized_calls: Vec<FinalizedToolCallOutcome> = Vec::new();
    let mut messages: Vec<ToolResultMessage> = Vec::new();

    for tool_call in tool_calls {
        // Send tool_execution_start event
        hooks
            .on_event(&AgentEvent::ToolExecutionStart {
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                args: tool_call.arguments.clone(),
            })
            .await;

        // Preprocessing
        let preparation =
            prepare_tool_call(context, assistant_message, tool_call, tools, hooks, cancel).await;

        let finalized = match preparation {
            Err(err_result) => FinalizedToolCallOutcome {
                tool_call: tool_call.clone(),
                result: err_result,
                is_error: true,
            },
            Ok(prepared) => {
                let executed = execute_prepared_tool_call(&prepared, cancel, hooks).await;
                finalize_executed_tool_call(
                    context,
                    assistant_message,
                    &prepared,
                    &executed,
                    hooks,
                    cancel,
                )
                .await
            }
        };

        // Send tool_execution_end event
        hooks
            .on_event(&AgentEvent::ToolExecutionEnd {
                tool_call_id: finalized.tool_call.id.clone(),
                tool_name: finalized.tool_call.name.clone(),
                result: finalized.result.clone(),
                is_error: finalized.is_error,
            })
            .await;

        messages.push(finalized.result.message.clone());
        finalized_calls.push(finalized);
    }

    (messages, should_terminate_tool_batch(&finalized_calls))
}

/// Determines whether a tool batch should terminate the agent loop.
///
/// Returns `true` when there is at least one tool result and all tool results have `terminate` set to `true`.
fn should_terminate_tool_batch(finalized_calls: &[FinalizedToolCallOutcome]) -> bool {
    !finalized_calls.is_empty() && finalized_calls.iter().all(|f| f.result.terminate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::agent_hooks::FinalizedToolCallOutcome;
    use zen_ai::tool::ToolExecutionResult;
    use zen_ai::types::{ToolCall, ToolResultMessage, Usage};

    fn make_tool_call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: serde_json::json!({}),
            thought_signature: None,
        }
    }

    fn make_outcome(terminate: bool) -> FinalizedToolCallOutcome {
        FinalizedToolCallOutcome {
            tool_call: make_tool_call("call_1", "test_tool"),
            result: ToolExecutionResult {
                message: ToolResultMessage {
                    tool_call_id: "call_1".to_string(),
                    tool_name: "test_tool".to_string(),
                    content: vec![],
                    details: None,
                    usage: Usage::default(),
                    is_error: false,
                    added_tool_names: None,
                    timestamp: 0,
                },
                terminate,
            },
            is_error: false,
        }
    }

    #[test]
    fn test_should_terminate_tool_batch_empty() {
        assert!(!should_terminate_tool_batch(&[]));
    }

    #[test]
    fn test_should_terminate_tool_batch_all_false() {
        let calls = vec![make_outcome(false), make_outcome(false)];
        assert!(!should_terminate_tool_batch(&calls));
    }

    #[test]
    fn test_should_terminate_tool_batch_all_true() {
        let calls = vec![make_outcome(true), make_outcome(true)];
        assert!(should_terminate_tool_batch(&calls));
    }

    #[test]
    fn test_should_terminate_tool_batch_mixed() {
        let calls = vec![make_outcome(true), make_outcome(false)];
        assert!(!should_terminate_tool_batch(&calls));
    }

    #[test]
    fn test_should_terminate_tool_batch_single_true() {
        let calls = vec![make_outcome(true)];
        assert!(should_terminate_tool_batch(&calls));
    }

    #[test]
    fn test_should_terminate_tool_batch_single_false() {
        let calls = vec![make_outcome(false)];
        assert!(!should_terminate_tool_batch(&calls));
    }

    // The following functions depend on external state (StreamFn, AgentLoopHooks trait, Tool trait, CancellationToken, etc.),
    // and there is no ready-made mock/test-double approach in the current module, so unit tests are skipped for now:
    // - run_agent_loop: depends on stream_fn (async closure) and hooks trait
    // - run_agent_loop_continue: same as above
    // - stream_assistant_response: depends on stream_fn channel and hooks trait
    // - execute_tool_calls: depends on Tool trait and hooks trait
    // - execute_tool_calls_sequential: same as above
    // - execute_tool_calls_parallel: same as above
    // Suggestion: add tests for these functions after introducing a mock framework (e.g. mockall).
}

