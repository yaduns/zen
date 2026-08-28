use crate::harness::agent_hooks::{AgentLoopError, AgentLoopHooks};
use crate::harness::agent_loop::{run_agent_loop, run_agent_loop_continue};
use crate::harness::context::create_context;
use crate::harness::types::AgentLoopConfig;
use crate::harness::types::{
    AfterToolCallContext, AfterToolCallResult, AgentEvent, AgentLoopTurnUpdate, AgentState,
    BeforeToolCallContext, BeforeToolCallResult, PrepareNextTurnContext,
    ShouldStopAfterTurnContext,
};
use crate::util::now_millis;
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;
use zen_ai::tool::{Tool, ToolExecutionMode};
use zen_ai::types::StreamFn;
use zen_ai::types::{ContentBlock, ContentBlockList, Message, Model, ThinkingLevel, UserMessage};

/// Stateful, session-level agent engine. Holds AgentState (accumulated across turns) and two message queues: steer and follow-up,
/// exposing prompt/continue_turn/steer/follow_up/set_model/set_tools/abort/wait_for_idle/reset etc. externally;
/// internally implements AgentLoopHooks (queue draining + everything else delegated to the user hooks).
/// Each prompt/continue_turn builds a fresh Context (with tools) and AgentLoopConfig from state (model derived from state.model + scattered settings),
///  syncing newly added messages back into state.messages when done.
pub struct Agent {
    /// system_prompt (a fully assembled string produced at the app layer), model, thinking_level, tools (single source, created per-session by the application layer via ToolFactory and bound to the environment), messages (accumulated across turns); transient fields (is_run/streaming_message/pending_tool_calls/error_message) are maintained by AgentState's start_run/finish_run/apply_event (aligned with pi's MutableAgentState transient fields, plan A)
    state: AgentState,
    tool_execution: ToolExecutionMode,
    /// User-behavior hooks; delegation target for Agent's non-queue methods
    hooks: Arc<dyn AgentLoopHooks>,
    /// Each prompt passes self.stream_fn.clone() to the lower layer
    stream_fn: StreamFn,
    /// abort() = cancel.cancel()
    cancel: CancellationToken,
    /// Consumed by the Agent's own impl AgentLoopHooks
    steer_queue: Arc<Mutex<VecDeque<Message>>>,
    /// Consumed by the Agent's own impl AgentLoopHooks
    follow_up_queue: Arc<Mutex<VecDeque<Message>>>,
    // New Agent fields (initialized in new: next_id: 0, entries: vec![])
    event_listeners: Arc<Mutex<EventListenerList>>,
}

// Engine-private implementation details (not pub)
struct EventListenerList {
    next_id: u64,
    entries: Vec<(u64, Arc<dyn Fn(&AgentEvent) + Send + Sync>)>,
}

#[async_trait]
impl AgentLoopHooks for Agent {
    // ── Engine-owned: queue draining (route A: steer/follow-up are engine-internal concerns, not delegated) ──
    async fn get_steering_messages(&self) -> Vec<Message> {
        self.steer_queue
            .lock()
            .unwrap()
            .pop_front()
            .into_iter()
            .collect() // One per turn
    }
    async fn get_follow_up_messages(&self) -> Vec<Message> {
        self.follow_up_queue
            .lock()
            .unwrap()
            .pop_front()
            .into_iter()
            .collect()
    }

    // ── Everything below delegates to self.hooks (user hooks), so user behavior is not silently overridden by trait defaults ──
    async fn filter_messages(&self, messages: Vec<Message>) -> Vec<Message> {
        self.hooks.filter_messages(messages).await
    }
    async fn transform_context(
        &self,
        messages: Vec<Message>,
        cancel: &CancellationToken,
    ) -> Vec<Message> {
        self.hooks.transform_context(messages, cancel).await
    }
    async fn get_api_key(&self, provider: &str) -> Option<String> {
        self.hooks.get_api_key(provider).await
    }
    async fn should_stop_after_turn(&self, context: &ShouldStopAfterTurnContext) -> bool {
        self.hooks.should_stop_after_turn(context).await
    }
    async fn prepare_next_turn(
        &self,
        context: &PrepareNextTurnContext,
    ) -> Option<AgentLoopTurnUpdate> {
        self.hooks.prepare_next_turn(context).await
    }
    async fn before_tool_call(
        &self,
        context: &BeforeToolCallContext,
        cancel: &CancellationToken,
    ) -> Result<Option<BeforeToolCallResult>, String> {
        self.hooks.before_tool_call(context, cancel).await
    }
    async fn after_tool_call(
        &self,
        context: &AfterToolCallContext,
        cancel: &CancellationToken,
    ) -> Result<Option<AfterToolCallResult>, String> {
        self.hooks.after_tool_call(context, cancel).await
    }
    async fn on_event(&self, event: &AgentEvent) {
        // (1) Snapshot inside the lock (Arc clone, shallow copy only; lock is held only very briefly)
        let snapshot: Vec<Arc<dyn Fn(&AgentEvent) + Send + Sync>> = {
            let guard = self.event_listeners.lock().unwrap();
            guard.entries.iter().map(|(_, l)| Arc::clone(l)).collect()
        };
        // (2) Invoke one by one outside the lock (no lock held; a listener panic does not poison the lock; a listener re-entering subscribe does not deadlock)
        for l in &snapshot {
            l(event);
        }
        // (3) The lock is now released, so delegating to hooks can safely await
        self.hooks.on_event(event).await
    }
}

impl Agent {
    /// Construct the engine (configured once).
    ///
    /// # Parameters
    /// - `state`: route A single-source-of-truth state (`system_prompt` is a finished string assembled by the app layer,
    ///   `tools` is the current tool set, `messages` is the restored history). After construction `state.tools`/`state.model`
    ///   i.e. the sole source.
    /// - `tool_execution`: tool execution mode (aligned with pi's `toolExecution`, the engine's only scattered setting).
    /// - `hooks`: user-behavior hooks (delegation target for non-queue methods)
    /// - `stream_fn`: LLM streaming call function
    /// - `cancel`: cancellation token
    ///
    /// # Edge Cases
    /// - Both queues are created inside `new` and retained as engine fields.
    /// - `Agent` is the `AgentLoopHooks` implementation: no separate hooks wrapper needed.
    pub fn new(
        state: AgentState,
        tool_execution: ToolExecutionMode,
        hooks: Arc<dyn AgentLoopHooks>,
        stream_fn: StreamFn,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            state,
            tool_execution,
            hooks,
            stream_fn,
            cancel,
            // Both queues are created in new and kept as engine fields (aligns with the comment's edge cases).
            steer_queue: Arc::new(Mutex::new(VecDeque::new())),
            follow_up_queue: Arc::new(Mutex::new(VecDeque::new())),
            event_listeners: Arc::new(Mutex::new(EventListenerList {
                next_id: 0,
                entries: Vec::new(),
            })),
        }
    }

    /// Builds this turn's runtime `AgentLoopConfig` (model/reasoning derived from the single source of truth, `state`).
    ///
    /// Rebuilt on the fly from state + scattered settings on every low-level call.
    /// `reasoning` is derived from `state.thinking_level`: `Off` → `None`, otherwise `Some(level)`
    /// Fields: `model`/`reasoning` ← state, `tool_execution` ← engine field, `api_key` fixed to `None`
    /// (low-level goes through `hooks.get_api_key`).
    fn make_config(&self) -> AgentLoopConfig {
        AgentLoopConfig {
            model: self.state.model.clone(),     // ← single source of truth: state
            api_key: None,                       // Low level goes through hooks.get_api_key
            tool_execution: self.tool_execution, // ← engine field (Copy)
            reasoning: match self.state.thinking_level {
                ThinkingLevel::Off => None, // Off → don't send reasoning_effort
                level => Some(level),       // Pass other levels through unchanged
            },
        }
    }

    /// Appends a user prompt and runs one turn (= Pi low-level `Agent::prompt()`).
    ///
    /// # Behavior
    /// 1. Build a fresh `Context` from `self.state`: `create_context(state.system_prompt.clone(),
    ///    state.messages.clone(), &state.tools)` (aligns with Route A's per-turn `AgentContext` snapshot);
    /// 2. Build a fresh `AgentLoopConfig` (`self.make_config()`, model derived from `state.model`);
    /// 3. Call `run_agent_loop(vec![user_msg], &mut ctx, &run_config, self, &self.cancel,
    ///    self.stream_fn.clone())` — pass `&*self` as the hooks argument (`Agent` implements `AgentLoopHooks`).
    ///    `&*self` is a **shared borrow** that lives across `.await`; `ctx`/`config`/`cancel`/`stream_fn` are all owned locals
    ///    cloned from `self` before the call, so they don't conflict with the `&mut self` receiver; the shared borrow ends
    ///    after the await returns, then flow back into `state` (the low-level signature drops `tool_ctx` per the "tool alignment" plan);
    /// 4. Sync back to `self.state`: by default `extend` the returned new messages back into `state.messages`;
    ///    if the low-level replaced `ctx` wholesale mid-run (`prepare_next_turn` returned a `Context` with swapped material, see edge case below),
    ///    take the replaced `ctx` back in full and overwrite `state` with it.
    ///
    /// # Return Value
    /// All messages added this turn (including the user prompt, assistant, and tool results).
    ///
    /// # Edge Cases
    /// - `&mut self`: the engine is exclusively borrowed during the run, so external code cannot call `prompt` concurrently (guaranteed by the Rust compiler).
    /// - `state.messages` is the authoritative accumulator; `ctx`/`run_config` are per-turn snapshots, not held across turns.
    /// - **Swapping materials must be written back in full**: between turns the low-level loop asks the app layer "swap in a new set of materials for the next turn?" (`prepare_next_turn`
    ///   hook). If the app layer returns a full replacement `Context` (typical case: Phase 2 compaction—when the conversation gets long, compress history into a summary,
    ///   prune old messages, or even replace the system prompt), the low-level loop replaces `ctx` wholesale. The engine **must not merely `extend` the new messages back into
    ///   `state`**—it must write the replacement `ctx` (messages / system_prompt / tools) back over `state` in full**,
    ///   otherwise the "rewrite the whole session" operation is silently dropped, and the canonical copy diverges from what the model actually saw.
    ///   v1 has no compaction and this hook usually returns `None`, but the write-back convention must be agreed upon from the start.
    pub async fn prompt(&mut self, text: &str) -> Vec<Message> {
        // 1) Build this turn's snapshot (Route A: derive Context from state each turn)
        let mut ctx = create_context(
            self.state.system_prompt.clone(),
            self.state.messages.clone(),
            &self.state.tools,
        );
        // 2) Build config on the fly (single source for model/reasoning)
        let run_config = self.make_config();
        // 3) User prompt message
        let user_msg = Message::User(UserMessage {
            content: ContentBlockList::Text(text.to_string()),
            timestamp: now_millis(),
        });
        // 4) Run the low-level loop: pass &*self as hooks (Agent implements AgentLoopHooks);
        //    ctx/run_config/cancel/stream_fn are owned locals cloned before the call, no borrow conflict with &mut self
        let new_messages = run_agent_loop(
            vec![user_msg],
            &mut ctx,
            &run_config,
            &*self,
            &self.cancel,
            self.stream_fn.clone(),
        )
        .await;
        // 5) Take ctx back in full → state (messages are the authoritative accumulation; if prepare_next_turn swapped the material,
        //    system_prompt/tools are also recalled, to avoid divergence between the canonical copy and what the model sees)
        self.state.messages = ctx.messages;
        self.state.system_prompt = ctx.system_prompt.unwrap_or_default();
        self.state.tools = ctx.tools.unwrap_or_default();
        new_messages
    }

    /// Runs one more turn from the current transcript (= Pi low-level `Agent::continue()`, `agent.ts:350`).
    /// prompt is "run again with input"; continue is "no input given—let the LLM finish what it hadn't answered"
    /// continue's job is "the transcript already contains a reason for the LLM to respond" (a trailing user or toolResult), so there's no need to craft another
    /// Typical use cases for continue:
    /// 1. Retry: the previous turn ended due to an error/abort, the transcript still ends with user/toolResult (the error is recorded in the assistant message), and you want the LLM to respond again with the same context;
    /// 2. Continuation after an external message is injected: e.g. some stage pushes a user/toolResult message into the context outside the loop, then lets the agent continue processing;
    /// 3. steer/followUp queue is drained
    ///
    /// Same construction flow as `prompt`, but **without appending a user prompt** — calls `run_agent_loop_continue` directly
    /// (requires `state.messages` to be non-empty and the last message to be non-assistant).
    ///
    /// # Errors
    /// - Empty context / last message is assistant → `Err(AgentLoopError)` (returned by the lower layer).
    pub async fn continue_turn(&mut self) -> Result<Vec<Message>, AgentLoopError> {
        // 1) Build a snapshot for this round on the fly (same construction flow as prompt)
        let mut ctx = create_context(
            self.state.system_prompt.clone(),
            self.state.messages.clone(),
            &self.state.tools,
        );
        // 2) Build config on the fly (single source for model/reasoning)
        let run_config = self.make_config();
        // 3) Call the lower-level continue: do not append a user prompt;
        //    Empty context / last message is assistant → Err(AgentLoopError) returned by the lower layer, propagated via ?
        let new_messages = run_agent_loop_continue(
            &mut ctx,
            &run_config,
            &*self,
            &self.cancel,
            self.stream_fn.clone(),
        )
        .await?;
        // 4) Return the whole ctx → state (same return-flow convention as prompt)
        self.state.messages = ctx.messages;
        self.state.system_prompt = ctx.system_prompt.unwrap_or_default();
        self.state.tools = ctx.tools.unwrap_or_default();
        Ok(new_messages)
    }

    /// Queue a user message to be injected before the LLM responds (= route A `steer()`).
    ///
    /// # Behavior
    /// Constructs a user message and pushes it into `steer_queue`; the low-level loop drains it after each tool batch via
    /// `get_steering_messages` (one-at-a-time, matching route A's default).
    ///
    /// # Edge Cases
    /// - `&self`: only locks the `Arc<Mutex>`; safe to call while running (leaves room for future concurrent TUI steering).
    /// - Messages in the queue are injected on the **next** loop iteration; they do not interrupt the current streaming LLM response.
    pub fn steer(&self, text: &str) {
        // Construct user message (aligned with the Message::User construction pattern used across the zen codebase)
        let message = Message::User(UserMessage {
            content: ContentBlockList::Text(text.to_string()),
            timestamp: now_millis(),
        });
        // Push into steer_queue (&self: lock only, safe to call while running)
        if let Ok(mut queue) = self.steer_queue.lock() {
            queue.push_back(message);
        }
    }

    /// Queue a follow-up message (= route A `followUp()`).
    ///
    /// Similar to [`Agent::steer`], but injected after the agent has no more tool calls
    /// (drained via `get_follow_up_messages`), used to continue the conversation.
    pub fn follow_up(&self, text: &str) {
        // Construct user message (aligned with the Message::User construction pattern used across the zen codebase)
        let message = Message::User(UserMessage {
            content: ContentBlockList::Text(text.to_string()),
            timestamp: now_millis(),
        });
        // Push into steer_queue (&self: lock only, safe to call while running)
        if let Ok(mut queue) = self.follow_up_queue.lock() {
            queue.push_back(message);
        }
    }

    /// Set the model (= route A `setModel()`).
    ///
    /// **Only modifies `state.model` (single source of truth)**. `AgentLoopConfig.model` is derived from `state` on the next `prompt`/`continue_turn`
    /// — no persistent dual ownership. Session persistence (appendModelChange) will be wired in when SessionManager lands in Phase 2.
    pub fn set_model(&mut self, model: Model) {
        self.state.model = model;
    }

    /// Replace the tool set (= route A `setTools()`).
    ///
    /// Only updates `self.state.tools` (single source of truth). `Context.tools` is derived from `state` on the next `prompt`,
    /// with no persistent dual ownership.
    pub fn set_tools(&mut self, tools: Vec<Arc<dyn Tool>>) {
        self.state.tools = tools;
    }

    /// Restore session history after construction (mirrors pi's `agent.state.messages = buildSessionContext().messages`).
    ///
    /// Symmetric to `set_model`/`set_tools`: only modifies `state.messages` (single source of truth). Used by the assembly phase (T-AP-08-3
    /// `create_initial_runtime_from_cli`) to load back a persisted/forked session's history; the next `prompt`'s
    /// Context is derived from `state`. For a new session, pass an empty `Vec` for empty history (no forking needed).
    pub fn set_messages(&mut self, messages: Vec<Message>) {
        self.state.messages = messages;
    }

    /// Cancel the current/subsequent run (= route A `abort()`).
    ///
    /// Calls `self.cancel.cancel()` internally. The lower-level loop exits at a safe point after receiving the cancellation signal.
    pub fn abort(&self) {
        self.cancel.cancel();
    }

    /// Wait until the engine is idle (API aligned with route A `waitForIdle()`).
    ///
    /// Rust's `&mut self` gives compile-time mutual exclusion — the engine is always idle externally; empty implementation.
    pub async fn wait_for_idle(&self) {}

    /// Reset session (= route A `reset()`, `agent.ts:326`).
    ///
    /// Clear `state.messages` and transient runtime fields (via `AgentState::reset`,
    /// (including `is_run`/`streaming_message`/`pending_tool_calls`/`error_message`) + the steer/follow_up queues.
    /// config in AgentState (system_prompt/model/thinking_level/tools) is preserved — after reset the same instance can continue `prompt()`.
    pub fn reset(&mut self) {
        self.state.reset();
        self.state.messages = vec![];
        if let Ok(mut queue) = self.steer_queue.lock() {
            queue.clear();
        }
        if let Ok(mut queue) = self.follow_up_queue.lock() {
            queue.clear();
        }
    }

    /// Register an event listener (aligned with pi's low-level `Agent.subscribe`, agent.ts:243).
    ///
    /// Listeners are stored in the engine's internal listener list and cleaned up when the `Agent` is dropped (no global/static table is created).
    /// The engine's own `AgentLoopHooks::on_event` implementation becomes: first synchronously notify all subscribed listeners,
    /// then delegate to the user hooks (`self.hooks.on_event`) — a dual channel of subscribe + hooks, see "Open Issue 1".
    ///
    /// # Return Value
    /// Returns an unsubscribe closure (calling it removes the listener from the list).
    pub fn subscribe(
        &self,
        listener: Box<dyn Fn(&AgentEvent) + Send + Sync>,
    ) -> Box<dyn FnOnce() + Send> {
        // Box<dyn Fn> → Arc<dyn Fn> (unsized coercion, Arc::from converts directly)
        let listener: Arc<dyn Fn(&AgentEvent) + Send + Sync> = Arc::from(listener);
        let mut guard = self.event_listeners.lock().unwrap();
        let id = guard.next_id;
        guard.next_id += 1;
        guard.entries.push((id, listener));
        // Arc::clone does two things: reference count +1 and copy a pointer — adds another owner of the same shared object
        let list = self.event_listeners.clone();
        // Return the unsubscriber function to the subscriber
        // The values of id and list (note: it is an Arc pointer holder) are copied into the closure
        Box::new(move || {
            if let Ok(mut g) = list.lock() {
                g.entries.retain(|(i, _)| *i != id);
            }
        })
    }

    // Consumer: T-AP-01-1 (persistent subscription via AgentSession::new + pass-through in AgentSession::subscribe)

    pub fn state(&self) -> &AgentState {
        &self.state
    }

    pub fn set_thinking_level(&mut self, level: ThinkingLevel) {
        self.state.thinking_level = level;
    }

    /// Read-only snapshot of the steer queue (for displaying pending steer message text in the UI).
    pub fn steering_messages(&self) -> Vec<String> {
        queue_texts(&self.steer_queue)
    }

    /// Read-only snapshot of the follow-up queue (for displaying pending follow-up message text in the UI).
    pub fn follow_up_messages(&self) -> Vec<String> {
        queue_texts(&self.follow_up_queue)
    }
}

/// Helper that takes a plain-text snapshot of all messages in the queue (read-only, no drain).
///
/// Messages currently constructed via `steer`/`follow_up` are all `ContentBlockList::Text`;
/// Defensive handling of the `Blocks` variant — concatenate text blocks, ignore non-text blocks like image/thinking.
fn queue_texts(queue: &Arc<Mutex<VecDeque<Message>>>) -> Vec<String> {
    queue
        .lock()
        .unwrap() // Lock poisoning only occurs when the lock-holding thread panics, consistent with the existing drain path in this file (get_steering_messages)
        .iter()
        .map(|msg| match msg {
            Message::User(user) => match &user.content {
                ContentBlockList::Text(text) => text.clone(),
                ContentBlockList::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text(t) => Some(t.text.clone()),
                        _ => None, // Image/Thinking/ToolCall blocks are not included in text display
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            },
            _ => String::new(), // The queue can only hold User messages; defensive fallback
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use zen_ai::types::{AssistantMessageEvent, Context, TextContent};

    // ── Test helper construction ──

    /// Minimal AgentLoopHooks: records only on_event, other methods use trait defaults.
    /// If a trait method has no default and causes a compile error, add an empty implementation.
    struct MockHooks;

    #[async_trait]
    impl AgentLoopHooks for MockHooks {
        async fn on_event(&self, _event: &AgentEvent) {}
    }

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

    /// Returns a mock StreamFn of an empty stream (tests in this file don't invoke the prompt; this is only a placeholder).
    fn mock_stream_fn() -> StreamFn {
        Arc::new(
            |_model: &Model,
             _ctx: &Context,
             _cfg: Option<zen_ai::api::openai_stream::StreamConfig>,
             _cancel: Option<CancellationToken>| {
                Box::pin(async move {
                    let (_tx, rx): (
                        tokio::sync::mpsc::UnboundedSender<AssistantMessageEvent>,
                        tokio::sync::mpsc::UnboundedReceiver<AssistantMessageEvent>,
                    ) = tokio::sync::mpsc::unbounded_channel();
                    Ok::<_, zen_ai::api::openai_stream::StreamError>(rx)
                })
            },
        )
    }

    fn agent_state() -> AgentState {
        AgentState {
            system_prompt: "test".into(),
            model: test_model(),
            thinking_level: ThinkingLevel::Off,
            tools: vec![],
            messages: vec![],
            is_run: false,
            streaming_message: None,
            pending_tool_calls: HashSet::new(),
            error_message: None,
        }
    }

    fn test_agent_with_cancel(cancel: CancellationToken) -> Agent {
        Agent::new(
            agent_state(),
            ToolExecutionMode::Sequential,
            Arc::new(MockHooks),
            mock_stream_fn(),
            cancel,
        )
    }

    fn test_agent() -> Agent {
        test_agent_with_cancel(CancellationToken::new())
    }

    // ── subscribe / on_event ──

    #[tokio::test]
    async fn test_subscribe_receives_event() {
        let agent = test_agent();
        let received: Arc<Mutex<Vec<AgentEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let recv = received.clone();
        let _unsub = agent.subscribe(Box::new(move |e: &AgentEvent| {
            recv.lock().unwrap().push(e.clone());
        }));

        agent.on_event(&AgentEvent::AgentStart).await;

        assert_eq!(received.lock().unwrap().len(), 1);
        assert!(matches!(
            received.lock().unwrap()[0],
            AgentEvent::AgentStart
        ));
    }

    #[tokio::test]
    async fn test_subscribe_multiple_listeners_all_receive() {
        let agent = test_agent();
        let c1 = Arc::new(AtomicUsize::new(0));
        let c2 = Arc::new(AtomicUsize::new(0));
        let (a, b) = (c1.clone(), c2.clone());
        let _u1 = agent.subscribe(Box::new(move |_: &AgentEvent| {
            a.fetch_add(1, Ordering::SeqCst);
        }));
        let _u2 = agent.subscribe(Box::new(move |_: &AgentEvent| {
            b.fetch_add(1, Ordering::SeqCst);
        }));

        agent.on_event(&AgentEvent::TurnStart).await;

        assert_eq!(c1.load(Ordering::SeqCst), 1);
        assert_eq!(c2.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_subscribe_unsubscribe_stops_events() {
        let agent = test_agent();
        let count = Arc::new(AtomicUsize::new(0));
        let c = count.clone();
        let unsub = agent.subscribe(Box::new(move |_: &AgentEvent| {
            c.fetch_add(1, Ordering::SeqCst);
        }));

        agent.on_event(&AgentEvent::AgentStart).await;
        assert_eq!(count.load(Ordering::SeqCst), 1);

        unsub();
        agent.on_event(&AgentEvent::AgentStart).await;
        assert_eq!(count.load(Ordering::SeqCst), 1, "no events received after unsubscribe");
    }

    #[tokio::test]
    async fn test_subscribe_ids_are_unique() {
        let agent = test_agent();
        let drop_a = Arc::new(AtomicUsize::new(0));
        let drop_b = Arc::new(AtomicUsize::new(0));
        let (da, db) = (drop_a.clone(), drop_b.clone());
        let _ua = agent.subscribe(Box::new(move |_: &AgentEvent| {
            da.fetch_add(1, Ordering::SeqCst);
        }));
        let _ub = agent.subscribe(Box::new(move |_: &AgentEvent| {
            db.fetch_add(1, Ordering::SeqCst);
        }));

        // Unsubscribing the first one should only decrease the first's count, not affecting the second (ids are unique)
        _ua();
        agent.on_event(&AgentEvent::AgentStart).await;
        assert_eq!(drop_a.load(Ordering::SeqCst), 0);
        assert_eq!(drop_b.load(Ordering::SeqCst), 1);
    }

    // ── steer / follow_up / queue snapshot ──

    #[test]
    fn test_steer_enqueues_and_snapshot() {
        let agent = test_agent();
        agent.steer("hello");
        agent.steer("world");
        assert_eq!(
            agent.steering_messages(),
            vec!["hello".to_string(), "world".to_string()]
        );
    }

    #[test]
    fn test_follow_up_enqueues_and_snapshot() {
        let agent = test_agent();
        agent.follow_up("please continue");
        assert_eq!(
            agent.follow_up_messages(),
            vec!["please continue".to_string()]
        );
    }

    #[tokio::test]
    async fn test_get_steering_messages_drains_one() {
        let agent = test_agent();
        agent.steer("a");
        agent.steer("b");

        let drained = agent.get_steering_messages().await;
        assert_eq!(drained.len(), 1, "each round drains one");
        // One message remains undrained
        assert_eq!(agent.steering_messages(), vec!["b".to_string()]);
    }

    #[test]
    fn test_queue_texts_blocks_variant_joins_text() {
        let queue: Arc<Mutex<VecDeque<Message>>> =
            Arc::new(Mutex::new(VecDeque::from(vec![Message::User(
                UserMessage {
                    content: ContentBlockList::Blocks(vec![
                        ContentBlock::Text(TextContent {
                            text: "first".into(),
                            text_signature: None,
                        }),
                        ContentBlock::Text(TextContent {
                            text: "second".into(),
                            text_signature: None,
                        }),
                    ]),
                    timestamp: 0,
                },
            )])));
        assert_eq!(queue_texts(&queue), vec!["first\nsecond".to_string()]);
    }

    // ── set_model / set_thinking_level / set_tools / state ──

    #[test]
    fn test_set_model_updates_single_source() {
        let mut agent = test_agent();
        let m = test_model();
        agent.set_model(m.clone());
        assert_eq!(agent.state().model, m);
    }

    #[test]
    fn test_set_thinking_level_updates_state() {
        let mut agent = test_agent();
        agent.set_thinking_level(ThinkingLevel::High);
        assert_eq!(agent.state().thinking_level, ThinkingLevel::High);
    }

    #[test]
    fn test_set_tools_replaces_state() {
        let mut agent = test_agent();
        agent.set_tools(vec![]);
        assert!(agent.state().tools.is_empty());
    }

    #[test]
    fn test_set_messages_restores_history() {
        let mut agent = test_agent();
        // Initially empty history
        assert!(agent.state().messages.is_empty());
        // Reload persisted session history (align with pi `agent.state.messages = buildContext().messages`)
        let msg = Message::User(UserMessage {
            content: ContentBlockList::Text("History messages".into()),
            timestamp: 0,
        });
        agent.set_messages(vec![msg.clone()]);
        assert_eq!(agent.state().messages, vec![msg]);
        // Empty Vec restores empty history (new session)
        agent.set_messages(vec![]);
        assert!(agent.state().messages.is_empty());
    }

    #[test]
    fn test_state_returns_latest_state() {
        let agent = test_agent();
        assert_eq!(agent.state().system_prompt, "test");
        assert_eq!(agent.state().is_run, false);
    }

    // ── reset ──

    #[tokio::test]
    async fn test_reset_clears_queue_but_preserves_config() {
        let mut agent = test_agent();
        agent.steer("queued");
        let m = test_model();
        agent.set_model(m.clone());

        agent.reset();

        // Configuration is preserved (model is not cleared)
        assert_eq!(agent.state().model, m);
        // Queue is emptied
        assert!(agent.steering_messages().is_empty());
        assert!(agent.follow_up_messages().is_empty());
        // messages is empty
        assert!(agent.state().messages.is_empty());
    }

    // ── abort ──

    #[test]
    fn test_abort_cancels_token() {
        let cancel = CancellationToken::new();
        let agent = test_agent_with_cancel(cancel.clone());
        assert!(!cancel.is_cancelled());
        agent.abort();
        assert!(cancel.is_cancelled());
    }

    // ── make_config (private, testable within this module) ──

    #[test]
    fn test_make_config_reasoning_off_is_none() {
        let agent = test_agent(); // thinking_level = Off
        let cfg = agent.make_config();
        assert_eq!(cfg.reasoning, None, "Off → do not send reasoning_effort");
        assert_eq!(cfg.api_key, None, "api_key fixed to None, uses hooks.get_api_key");
    }

    #[test]
    fn test_make_config_reasoning_level_passthrough() {
        let mut agent = test_agent();
        agent.set_thinking_level(ThinkingLevel::Low);
        let cfg = agent.make_config();
        assert_eq!(cfg.reasoning, Some(ThinkingLevel::Low));
    }

    #[test]
    fn test_make_config_derives_model_and_tool_execution() {
        let agent = test_agent();
        let cfg = agent.make_config();
        assert_eq!(cfg.model, agent.state().model);
        assert_eq!(cfg.tool_execution, ToolExecutionMode::Sequential);
    }
}
