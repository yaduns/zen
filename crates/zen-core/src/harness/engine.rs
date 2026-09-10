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
///
/// **Interactive cancellation fix (Plan D1-D12)**:
/// - D3: `state: Mutex<AgentState>` (single lock point, short-lived std Mutex; shared surface provided by outer `Arc<AgentSession>`),
///   all methods take `&self`, supporting concurrent access from the run task and the main task (aligned with pi MutableAgentState).
/// - D4: `cancel: Mutex<CancellationToken>` replaced on every run = pi `activeRun.abortController`; `abort()` targets the "current" run,
///   when idle it targets the old token (no-op, doesn't pollute the next run).
/// - D7: `start_run` is wired into the `run_with_lifecycle` entry; `finish_run` is backed by `RunGuard` RAII on Drop
///   (unified across normal/panic paths, prevents a panic skipping finish_run → is_run stuck true → session dead).
/// - D11: the three transient fields `streaming_message`/`pending_tool_calls`/`error_message` are registered as v1 gaps
///   (no production readers, `on_event` not wired to `apply_event`; fields kept to align with pi).
pub struct Agent {
    /// system_prompt (fully assembled string from the app layer), model, thinking_level, tools (single source, created per session by the application layer via ToolFactory, bound to environment), messages (accumulated across turns); is_run maintained by start_run/run_with_lifecycle
    state: Mutex<AgentState>,
    tool_execution: ToolExecutionMode,
    /// User-behavior hooks; delegation target for Agent's non-queue methods
    hooks: Arc<dyn AgentLoopHooks>,
    /// Each prompt passes self.stream_fn.clone() to the lower layer
    stream_fn: StreamFn,
    /// D4: cancellation token for the current run (replaced on every run); abort() targets it (idle = old token no-op).
    cancel: Mutex<CancellationToken>,
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

/// In run_with_lifecycle, attach RunGuard after Agent's start_run succeeds; both normal return and panic unwind paths reset is_run on Drop.
/// RunGuard carries AgentState data because Drop needs its finish_run function; otherwise any variable inside run_with_lifecycle could serve as the carrier.
/// Prevents run_agent_loop panicking past finish_run → is_run stuck true → all subsequent runs rejected by the start_run guard
/// (session permanently dead); `unwrap_or_else(into_inner)` guards against lock poisoning (P1-2).
/// 'a cannot outlive the actual lifetime of the borrowed Mutex (otherwise dangling reference, compile error); the guard instance cannot outlive 'a, ensuring finish_run is called on the guard before the Mutex is dropped
struct RunGuard<'a>(&'a Mutex<AgentState>);

impl Drop for RunGuard<'_> {
    fn drop(&mut self) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .finish_run();
    }
}

/// Run entry distinction (prompt appends input / continue does not).
enum RunBody {
    Prompt(Vec<Message>),
    Continue,
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
    ///   i.e. the single source. Wrapped as `Mutex<AgentState>` (single lock point, D3).
    /// - `tool_execution`: tool execution mode (aligned with pi `toolExecution`, the engine's only scattered setting, Copy, read inside the lock).
    /// - `hooks`: user-behavior hooks (delegation target for non-queue methods)
    /// - `stream_fn`: LLM streaming call function
    ///
    /// # Edge Cases
    /// - Both queues are created inside `new` and retained as engine fields.
    /// - `Agent` is the `AgentLoopHooks` implementation: no separate hooks wrapper needed.
    /// - **Does not take a `cancel` argument** (D5): the token is built at each run entry (D4), so a constructor argument is meaningless.
    pub fn new(
        state: AgentState,
        tool_execution: ToolExecutionMode,
        hooks: Arc<dyn AgentLoopHooks>,
        stream_fn: StreamFn,
    ) -> Self {
        Self {
            state: Mutex::new(state),
            tool_execution,
            hooks,
            stream_fn,
            cancel: Mutex::new(CancellationToken::new()), // placeholder; replaced on the first run (D4)
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
    /// **Pure function: no locking, no touching `self`** (avoids re-entering the lock while holding the state guard; std Mutex is not reentrant).
    /// `state` is held by the caller under the lock; `tool_execution` is Copy, so the caller reads fields directly under the lock
    /// (field-level shared borrow, no conflict with the state lock).
    /// `reasoning` is derived from `state.thinking_level`: `Off` → `None`, otherwise `Some(level)`.
    fn make_config(state: &AgentState, tool_execution: ToolExecutionMode) -> AgentLoopConfig {
        AgentLoopConfig {
            model: state.model.clone(), // ← single source of truth: state
            api_key: None,              // Low level goes through hooks.get_api_key
            tool_execution,             // Passed in as parameters instead of reading from `self` (avoids re-entering the lock while holding the state guard)
            reasoning: match state.thinking_level {
                ThinkingLevel::Off => None, // Off → don't send reasoning_effort
                level => Some(level),       // Pass other levels through unchanged
            },
        }
    }

    /// Unified run lifecycle: under lock "guard + token refresh + snapshot" → release lock, run loop → under lock "write-back".
    /// `is_run` is finalized by `RunGuard`'s Drop (unified normal/panic paths, P1-2).
    ///
    /// # Lock Discipline (documented in comments)
    /// 1. Never await or call the loop under a lock; only do snapshot clones / pure logic.
    /// 2. Enforce the three-stage pattern: "snapshot under lock → release lock → await loop → write back under lock".
    /// 3. `make_config` is a pure function + Copy prefetched, eliminating lock re-entry while holding the state guard.
    /// 4. If the `start_run` guard rejects → warn + empty Vec (defensive net; a real prompt run always has at least a user message, so an empty Vec distinguishes it).
    /// 5. Poisoning handled with `unwrap_or_else(|e| e.into_inner())`.
    /// 6. Lock ordering: state→cancel nesting exists; always state→cancel, never the reverse.
    async fn run_with_lifecycle(&self, body: RunBody) -> Vec<Message> {
        let token = CancellationToken::new(); // D4：= pi new AbortController
        let (mut ctx, run_config, _guard) = {
            let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if s.start_run().is_err() {
                // D7 guard (defensive net, unreachable in the normal flow): dispatch_loop single-spawns + sticky guard, at most one run at any instant
                tracing::warn!("run rejected: agent already running");
                return Vec::new();
            }
            *self.cancel.lock().unwrap_or_else(|e| e.into_inner()) = token.clone(); // D4: refresh
            let tool_execution = self.tool_execution; // Copy-read fields under lock (not reentrant, harmless), consumed in place (not added to the tuple)
            let ctx = create_context(s.system_prompt.clone(), s.messages.clone(), &s.tools);
            let cfg = Self::make_config(&s, tool_execution); // Pure function, no re-entry
            let _guard = RunGuard(&self.state); // Attach RAII after start_run succeeds (P1-2)
            (ctx, cfg, _guard)
        }; // Release lock (guard drop); _guard borrows &self.state (shared) and outlives the function
        let new_messages = match body {
            RunBody::Prompt(input) => {
                run_agent_loop(
                    input,
                    &mut ctx,
                    &run_config,
                    self,
                    &token,
                    self.stream_fn.clone(),
                )
                .await
            }
            RunBody::Continue => match run_agent_loop_continue(
                &mut ctx,
                &run_config,
                self,
                &token,
                self.stream_fn.clone(),
            )
            .await
            {
                Ok(msgs) => msgs,
                // Fallback: continue_turn already pre-validates (see below), so Err is theoretically unreachable.
                // Note: run_agent_loop_continue **also performs** EmptyContext/InvalidLastRole validation internally,
                // If someone changes the internal validation in the future without syncing the up-front check, this Err branch catches it — **not dead code, do not delete** (dual-validation redundancy).
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "continue validation failed (continue_turn pre-validates, so theoretically unreachable)"
                    );
                    Vec::new()
                }
            },
        };
        {
            let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
            s.messages = ctx.messages; // Changing the material voids the entire batch
            s.system_prompt = ctx.system_prompt.unwrap_or_default();
            s.tools = ctx.tools.unwrap_or_default();
            // Explicit finish_run is no longer needed: _guard Drop (unified across normal/panic paths) provides an idempotent fallback (P1-2)
        }
        new_messages
    }

    /// Appends a user prompt and runs one turn (= Pi low-level `Agent::prompt()`).
    ///
    /// # Behavior
    /// Go through `run_with_lifecycle` (token refresh + start_run + snapshot → release lock to run loop → reacquire lock to flush back + RunGuard reset).
    ///
    /// # Return Value
    /// All messages added in this round (including user prompt, assistant, tool results). Returns an empty `Vec` when rejected by the guard (defensive net).
    ///
    /// # Edge Cases
    /// - `&self`: after plan D, the engine supports concurrent prompt/steer (single lock point AgentState, std Mutex short lock).
    /// - `state.messages` is the authoritative accumulator; `ctx`/`run_config` are per-turn snapshots, not held across turns.
    /// - **Swapping materials must be written back in full**: between turns the low-level loop asks the app layer "swap in a new set of materials for the next turn?" (`prepare_next_turn`
    ///   hook). If the app layer returns a full replacement `Context` (typical case: Phase 2 compaction—when the conversation gets long, compress history into a summary,
    ///   prune old messages, or even replace the system prompt), the low-level loop replaces `ctx` wholesale. The engine **must not merely `extend` the new messages back into
    ///   `state`**—it must write the replacement `ctx` (messages / system_prompt / tools) back over `state` in full**,
    ///   otherwise the "rewrite the whole session" operation is silently dropped, and the canonical copy diverges from what the model actually saw.
    ///   v1 has no compaction and this hook usually returns `None`, but the write-back convention must be agreed upon from the start.
    pub async fn prompt(&self, text: &str) -> Vec<Message> {
        let user_msg = Message::User(UserMessage {
            content: ContentBlockList::Text(text.to_string()),
            timestamp: now_millis(),
        });
        self.run_with_lifecycle(RunBody::Prompt(vec![user_msg]))
            .await
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
    /// - Empty context / last message is assistant → `Err(AgentLoopError)` (up-front entry validation, single snapshot).
    ///   **The up-front check and the internal check in run_agent_loop_continue must stay in sync** (see the Err fallback comment on the Continue arm of run_with_lifecycle; don't let the two drift apart).
    pub async fn continue_turn(&self) -> Result<Vec<Message>, AgentLoopError> {
        // Entry validation runs before the helper (produces Result; a single snapshot avoids two full clones)
        let msgs = self.messages();
        if msgs.is_empty() {
            return Err(AgentLoopError::EmptyContext);
        }
        if matches!(msgs.last(), Some(Message::Assistant(_))) {
            return Err(AgentLoopError::InvalidLastRole);
        }
        Ok(self.run_with_lifecycle(RunBody::Continue).await)
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
    pub fn set_model(&self, model: Model) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).model = model;
    }

    /// Replace the tool set (= route A `setTools()`).
    ///
    /// Only updates `self.state.tools` (single source of truth). `Context.tools` is derived from `state` on the next `prompt`,
    /// with no persistent dual ownership.
    pub fn set_tools(&self, tools: Vec<Arc<dyn Tool>>) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).tools = tools;
    }

    /// Restore session history after construction (mirrors pi's `agent.state.messages = buildSessionContext().messages`).
    ///
    /// Symmetric to `set_model`/`set_tools`: only modifies `state.messages` (single source of truth). Used by the assembly phase (T-AP-08-3
    /// `create_initial_runtime_from_cli`) to load back a persisted/forked session's history; the next `prompt`'s
    /// Context is derived from `state`. For a new session, pass an empty `Vec` for empty history (no forking needed).
    pub fn set_messages(&self, messages: Vec<Message>) {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .messages = messages;
    }

    /// Cancel the current/subsequent run (= route A `abort()`, plan D4 revision).
    ///
    /// Strike the "current" run token (`cancel: Mutex<CancellationToken>`, refreshed on every run);
    /// When idle, strike the old token (no-op, doesn't pollute the next round).
    pub fn abort(&self) {
        self.cancel
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .cancel();
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
    pub fn reset(&self) {
        {
            let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
            s.reset();
            s.messages = vec![];
        }
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

    /// The sole entry point where the application layer injects compaction lifecycle events into the agent event bus (T-AP-04-3) — compaction does not run
    /// inside the engine loop, but the UI only recognizes a single event channel, so the scheduler broadcasts through here at start/end.
    /// Behavior: synchronously dispatches `event` to the same listener list as `subscribe` (invoked one by one, indistinguishable from engine-emitted
    /// events; see the EventListenerList consumption path for details).
    /// Boundary: for use only by the zen-app scheduling layer; engine events never go through this method (prevents bypassing the state machine),
    /// `AgentState::apply_event` reduces compaction events to no-op.
    pub fn emit(&self, event: AgentEvent) {
        let listeners: Vec<_> = {
            let guard = self
                .event_listeners
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            guard.entries.iter().map(|(_, l)| Arc::clone(l)).collect()
        };
        for listener in listeners {
            listener(&event);
        }
    }

    /// Accessor for the engine stream function used by compaction scheduling (T-AP-04-3) — the summary LLM call must share the same source as the prompt loop
    /// (same auth resolution, same provider endpoint), otherwise a split of "conversation works, compaction fails" occurs.
    /// Behavior: clones and returns the internal stream_fn (`StreamFn: Clone`, same way the engine passes it each prompt round).
    pub fn stream_fn(&self) -> StreamFn {
        self.stream_fn.clone()
    }

    // ── State snapshot accessors (cannot return references while holding the lock, so all return owned snapshots; replaces the original state(), III.3/III.4) ──

    /// Whether a run is in progress (`state.is_run`, maintained by start_run/run_with_lifecycle/RunGuard).
    pub fn is_run(&self) -> bool {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).is_run
    }

    /// Current thinking level snapshot (`state.thinking_level`).
    pub fn thinking_level(&self) -> ThinkingLevel {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .thinking_level
    }

    /// Current model snapshot (`state.model`, owned clone).
    pub fn model(&self) -> Model {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .model
            .clone()
    }

    /// Finalized message history snapshot (`state.messages`, owned clone).
    pub fn messages(&self) -> Vec<Message> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .messages
            .clone()
    }

    /// Current toolset snapshot (`state.tools`, owned clone; shallow `Arc` copy).
    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .tools
            .clone()
    }

    /// Snapshot of the error message from the most recent failed/interrupted assistant turn (`state.error_message`, a vestigial field from D11, currently no writer).
    pub fn error_message(&self) -> Option<String> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .error_message
            .clone()
    }

    pub fn set_thinking_level(&self, level: ThinkingLevel) {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .thinking_level = level;
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
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use zen_ai::types::{AssistantMessage, AssistantMessageEvent, Context, TextContent};

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

    /// Returns a mock StreamFn that emits an empty stream (most tests in this file don't call prompt, they only construct a placeholder).
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

    /// Returns a "blockable/completable" mock StreamFn: on receiving a call it first emits Start (run immediately enters the running state),
    /// if `block` is true it waits — until the cancellation token is cancelled (→ emits `Error{Aborted}`) or `block` becomes false
    /// (→ emits TextDelta + Done and completes normally). Used for abort/per-run token regression and is_run transient wiring tests:
    /// the main task has a chance to observe the running state before deciding how to finish up.
    fn controllable_stream_fn(block: Arc<AtomicBool>) -> StreamFn {
        Arc::new(
            move |_model: &Model,
                  _ctx: &Context,
                  _cfg: Option<zen_ai::api::openai_stream::StreamConfig>,
                  cancel: Option<CancellationToken>| {
                let block = Arc::clone(&block);
                Box::pin(async move {
                    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<AssistantMessageEvent>();
                    let cancel = cancel.unwrap_or_default();
                    let partial = AssistantMessage {
                        content: vec![ContentBlock::Text(TextContent {
                            text: "ok".into(),
                            text_signature: None,
                        })],
                        api: "openai-completions".into(),
                        provider: "opencode-go".into(),
                        model: "deepseek-v4-flash".into(),
                        response_model: None,
                        response_id: None,
                        usage: zen_ai::types::Usage::default(),
                        stop_reason: zen_ai::types::StopReason::Stop,
                        error_message: None,
                        timestamp: 0,
                    };
                    let _ = tx.send(AssistantMessageEvent::Start {
                        partial: partial.clone(),
                    });
                    if block.load(Ordering::SeqCst) {
                        // wait for cancellation (→ Error{Aborted}) or unblocking (→ Done)
                        tokio::select! {
                            _ = cancel.cancelled() => {}
                            _ = async {
                                while block.load(Ordering::SeqCst) {
                                    tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                                }
                            } => {}
                        }
                    }
                    if cancel.is_cancelled() {
                        let _ = tx.send(AssistantMessageEvent::Error {
                            reason: zen_ai::types::ErrorStopReason::Aborted,
                            error: partial,
                        });
                    } else {
                        let _ = tx.send(AssistantMessageEvent::TextDelta {
                            content_index: 0,
                            delta: "ok".into(),
                            partial: partial.clone(),
                        });
                        let _ = tx.send(AssistantMessageEvent::Done {
                            reason: zen_ai::types::NonErrorStopReason::Stop,
                            message: partial,
                        });
                    }
                    drop(tx);
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

    fn test_agent() -> Agent {
        Agent::new(
            agent_state(),
            ToolExecutionMode::Sequential,
            Arc::new(MockHooks),
            mock_stream_fn(),
        )
    }

    /// Builds an engine with a "blockable/completable" StreamFn (for abort/per-run token regression / transient wiring tests).
    fn test_controllable_agent(block: Arc<AtomicBool>) -> Agent {
        Agent::new(
            agent_state(),
            ToolExecutionMode::Sequential,
            Arc::new(MockHooks),
            controllable_stream_fn(block),
        )
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

    // ── set_model / set_thinking_level / set_tools / state snapshot accessors ──

    #[test]
    fn test_set_model_updates_single_source() {
        let agent = test_agent();
        let m = test_model();
        agent.set_model(m.clone());
        assert_eq!(agent.model(), m);
    }

    #[test]
    fn test_set_thinking_level_updates_state() {
        let agent = test_agent();
        agent.set_thinking_level(ThinkingLevel::High);
        assert_eq!(agent.thinking_level(), ThinkingLevel::High);
    }

    #[test]
    fn test_set_tools_replaces_state() {
        let agent = test_agent();
        agent.set_tools(vec![]);
        assert!(agent.tools().is_empty());
    }

    #[test]
    fn test_set_messages_restores_history() {
        let agent = test_agent();
        // Initially empty history
        assert!(agent.messages().is_empty());
        // Reload persisted session history (align with pi `agent.state.messages = buildContext().messages`)
        let msg = Message::User(UserMessage {
            content: ContentBlockList::Text("History messages".into()),
            timestamp: 0,
        });
        agent.set_messages(vec![msg.clone()]);
        assert_eq!(agent.messages(), vec![msg]);
        // Empty Vec restores empty history (new session)
        agent.set_messages(vec![]);
        assert!(agent.messages().is_empty());
    }

    #[test]
    fn test_accessors_snapshot_latest_state() {
        let agent = test_agent();
        assert_eq!(agent.model().id, "deepseek-v4-flash");
        assert_eq!(agent.thinking_level(), ThinkingLevel::Off);
        assert!(!agent.is_run(), "Initially not running");
        assert!(agent.error_message().is_none());
        assert!(agent.messages().is_empty());
        assert!(agent.tools().is_empty());
    }

    // ── reset ──

    #[tokio::test]
    async fn test_reset_clears_queue_but_preserves_config() {
        let agent = test_agent();
        agent.steer("queued");
        let m = test_model();
        agent.set_model(m.clone());

        agent.reset();

        // Configuration is preserved (model is not cleared)
        assert_eq!(agent.model(), m);
        // Queue is emptied
        assert!(agent.steering_messages().is_empty());
        assert!(agent.follow_up_messages().is_empty());
        // messages is empty
        assert!(agent.messages().is_empty());
    }

    // ── abort (scheme D4: invalidate the current run token, no-op when idle) ──

    #[test]
    fn test_abort_idle_is_noop() {
        let agent = test_agent();
        // abort is a no-op when idle (no panic, doesn't pollute subsequent run tokens)
        agent.abort();
        assert!(!agent.is_run());
    }

    /// abort/per-run token regression: abort cancels the current run; then prompting again with a fresh run token (the engine swaps in a new one on each run)
    /// should complete normally (no longer polluted by "one abort cancels forever", D4).
    #[tokio::test]
    async fn test_abort_cancels_active_run_and_next_run_completes() {
        let block = Arc::new(AtomicBool::new(true));
        let agent = Arc::new(test_controllable_agent(Arc::clone(&block)));
        // First round: abort mid-run → this round is cancelled (controllable_stream_fn emits Error{Aborted} for a cancelled token)
        let task_agent = Arc::clone(&agent);
        let spawn1 = tokio::spawn(async move {
            let _ = task_agent.prompt("first").await;
        });
        // Abort after the first round enters running (is_run set, run hangs because block=true)
        let mut entered = false;
        for _ in 0..200 {
            if agent.is_run() {
                entered = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        assert!(entered, "is_run should be true during prompt run (D7 wiring)");
        agent.abort();
        spawn1.await.expect("first round task should not panic");
        assert!(!agent.is_run(), "is_run resets after run ends (RunGuard Drop)");

        // Second round: unblock + engine's fresh internal token (not cancelled) → should complete normally (returns an assistant message)
        block.store(false, Ordering::SeqCst);
        let msgs = agent.prompt("second").await;
        assert!(!msgs.is_empty(), "second round should produce message (normal completion)");
        assert!(agent
            .messages()
            .iter()
            .any(|m| matches!(m, Message::Assistant(_))));
        assert!(!agent.is_run(), "is_run resets after second round");
    }

    /// Transient wiring: `is_run()` is true during prompt, false after it finishes.
    #[tokio::test]
    async fn test_is_run_toggles_during_prompt() {
        let block = Arc::new(AtomicBool::new(true));
        let agent = Arc::new(test_controllable_agent(Arc::clone(&block)));
        assert!(!agent.is_run());
        let task_agent = Arc::clone(&agent);
        let spawned = tokio::spawn(async move {
            let _ = task_agent.prompt("hi").await;
        });
        let mut saw_run = false;
        for _ in 0..200 {
            if agent.is_run() {
                saw_run = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        assert!(saw_run, "is_run=true should be observed during prompt run");
        // Unblock → run completes normally → is_run resets
        block.store(false, Ordering::SeqCst);
        spawned.await.expect("prompt task should not panic");
        assert!(!agent.is_run(), "is_run=false after prompt ends");
    }

    // ── make_config (private pure function, testable within the same module) ──

    #[test]
    fn test_make_config_reasoning_off_is_none() {
        let cfg = Agent::make_config(&agent_state(), ToolExecutionMode::Sequential);
        assert_eq!(cfg.reasoning, None, "Off → do not send reasoning_effort");
        assert_eq!(cfg.api_key, None, "api_key fixed to None, uses hooks.get_api_key");
    }

    #[test]
    fn test_make_config_reasoning_level_passthrough() {
        let mut st = agent_state();
        st.thinking_level = ThinkingLevel::Low;
        let cfg = Agent::make_config(&st, ToolExecutionMode::Sequential);
        assert_eq!(cfg.reasoning, Some(ThinkingLevel::Low));
    }

    #[test]
    fn test_make_config_derives_model_and_tool_execution() {
        let st = agent_state();
        let cfg = Agent::make_config(&st, ToolExecutionMode::Parallel);
        assert_eq!(cfg.model, st.model);
        assert_eq!(cfg.tool_execution, ToolExecutionMode::Parallel);
    }
}
