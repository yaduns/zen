use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use zen_ai::types::{ContentBlockList, Message, Model, ThinkingLevel, UserMessage};
use zen_core::env::shell_output::{execute_shell_with_capture, ShellCaptureOptions};
use zen_core::env::{ExecutionEnv, ExecutionError};
use zen_core::harness::agent_hooks::AgentLoopError;
use zen_core::harness::engine::Agent;
use zen_core::harness::types::AgentEvent;
use zen_core::util::now_millis;

/// Session persistence interface, defining the persistence operations required by AgentSession.
/// Implemented by the SessionManager of T-AP-01-4.
///
/// Does not include Pi's repo injection abstraction (does not distinguish jsonl/memory repos),
/// it is a single trait describing AgentSession's requirements on the persistence layer.
pub trait SessionStore: Send + Sync {
    /// Append a message to the session JSONL.
    /// Corresponds to Pi's `SessionManager.appendMessage`.
    fn append_message(&self, message: &Message);

    /// Record a model-switch event.
    /// Corresponds to Pi `SessionManager.appendModelChange`.
    fn append_model_change(&self, provider: &str, model_id: &str);

    /// Records thinking level switch events.
    /// Corresponds to Pi `SessionManager.appendThinkingLevelChange`.
    fn append_thinking_level_change(&self, level: &str);

    /// Gets the current session ID.
    fn session_id(&self) -> String;

    /// Gets the working directory.
    fn cwd(&self) -> &str;
}
// Consumers: T-AP-01-1 (subscription closure invoked inside AgentSession), T-AP-01-4 (SessionManager implements this trait)

/// Higher-level session abstraction shared across modes.
///
/// Holds a pre-built Agent engine (with AgentState single source of truth + resident persistence subscription), SessionStore
/// Persistence delegate. **All session state (model/thinking_level/tools/messages/is_run)
/// All reads/writes are delegated via `self.agent.state()`; this struct stores no state fields** (aligns with pi
/// `agent.state.*` delegate pattern of `AgentSession`).
///
/// # Lifecycle
/// - Construct (establishes resident persistence subscription) → prompt (may be called multiple times) → drop (subscription cleaned up with the agent)
/// - abort can be called cross-thread during a prompt run (only manipulates the cancel token)
/// - subscribe should be called before the first prompt
pub struct AgentSession {
    agent: Agent,
    session_store: Arc<dyn SessionStore>,
    /// Identity registry for `!!` bash messages (excluded from the LLM context), shared with the engine-side `SessionHooks`
    /// (services.rs, T-AP-04-2-fix) via the same Arc — the key is the value of `exclusion_key`
    /// (user bash = UserMessage timestamp).
    excluded: Arc<Mutex<HashSet<String>>>,
}

impl AgentSession {
    /// Constructs an AgentSession.
    ///
    /// # Parameters
    /// - `agent`: the already-built engine (with the assembled AgentState single source of truth +
    ///   tool_execution + hooks + stream_fn + cancel）
    /// - `session_store`: persistence delegate
    ///
    /// # Behavior
    /// 1. `agent.subscribe(...)` establishes a **resident persistence subscription** (option 1, aligned with pi's `AgentSession`
    ///    which does `agent.subscribe(this._handleAgentEvent)` at construction):
    ///    - `AgentEvent::MessageEnd` → `session_store.append_message` (real-time persistence)
    /// 2. The subscription closure lives in the engine's listener list; **no unsubscribe handle is returned/stored** — it is
    ///    automatically cleaned up when dropped along with the `Agent` (held by value by this
    /// 3. Returns `Self` (internally holds an empty exclusion set; the assembly path uses `new_with_exclusions` to share the exclusion set)
    ///
    /// # Edge Cases
    ///    struct), so no `_subscription` field is needed
    /// - No config struct: only 2 constructor parameters, no need to aggregate (introduce one in Phase 2 if needed)
    /// - The persistence subscription is resident (established at construction, terminated on drop); it is the implementation of AgentSession's "automatic session persistence"
    /// - `new`'s exclusion set is an independent empty set (tests/quick-start path, not participating in loop filtering); real assembly
    ///   goes through `create_agent_session` (services.rs) using `new_with_exclusions`
    pub fn new(agent: Agent, session_store: Arc<dyn SessionStore>) -> Self {
        Self::new_with_exclusions(agent, session_store, Arc::new(Mutex::new(HashSet::new())))
    }

    /// Real assembly constructor (T-AP-04-2-fix): shares the exclusion set with the engine hooks.
    ///
    /// `excluded` must be the **same Arc** as the exclusion set inside the `SessionHooks` injected into the engine (`services.rs::create_agent_session`)
    /// so that identities registered by `execute_bash(!!)` can be read by `filter_messages`
    /// in the loop. `new` uses an empty set (test path, does not participate in filtering).
    pub(crate) fn new_with_exclusions(
        agent: Agent,
        session_store: Arc<dyn SessionStore>,
        excluded: Arc<Mutex<HashSet<String>>>,
    ) -> Self {
        // Establish a persistent subscription (option 1, aligned with pi construction as `agent.subscribe(this._handleAgentEvent)`):
        // message_end → session_store.append_message (persist to disk in real time). Subscription closure lives in the engine's listener list,
        // dropped and auto-cleaned together with the Agent (held by value in this struct); no return/no stored unsubscribe handle.
        let store = session_store.clone();
        // The subscription closure lives in the engine's listener list; unsubscription is left to agent drop (design: no unsubscribe handle stored)
        let _ = agent.subscribe(Box::new(move |event: &AgentEvent| {
            if let AgentEvent::MessageEnd { message } = event {
                store.append_message(message);
            }
        }));
        Self {
            agent,
            session_store,
            excluded,
        }
    }

    /// Send a prompt to the agent and wait for completion.
    ///
    /// # Behavior
    /// 1. Calls `self.agent.prompt(text)` (the engine renews the run token internally; pi's surface takes no args; D2/D4)
    /// 2. A resident persistence subscription writes to disk in real time during runtime (message_end → append_message); the consumer
    ///    Events subscribed via `subscribe` are delivered synchronously
    /// 3. Returns all messages added in this round
    ///
    /// # Errors
    /// - `AgentSessionError::Agent`: agent loop error (underlying `AgentLoopError`; the `prompt` path is always Ok — legacy quirk, out of scope here)
    ///
    /// # Edge Cases
    /// - `&self`: this struct is a lock-free Facade (store/excluded are already Arc-shared), callable concurrently by the run task and the main task
    ///   (&self after the engine is shared; see interactive cancellation fix plan D)
    pub async fn prompt(&self, text: &str) -> Result<Vec<Message>, AgentSessionError> {
        Ok(self.agent.prompt(text).await)
    }
    // Consumers: T-AP-07-1 (print mode prompt), T-AP-11-1 (interactive mode prompt)

    /// Executes a user `!`/`!!` bash command (T-AP-04-2-fix) — a standalone channel outside `AgentLoopHooks`.
    ///
    /// Called when `!command` is submitted in interactive mode: goes through the underlying `execute_shell_with_capture` (T-AP-04-2 channel,
    /// run command (with 5303 wait semantics / cwd pre-check / tail truncation / overflow to temp file), render result as pi
    /// `bashExecutionToText` as `Message::User` appended to the conversation (LLM sees "user ran
    /// a command + output", no orphan tool_call); `!!` (`exclude_from_context`) registers the message's
    /// timestamp into the exclusion set — the next loop's `SessionHooks::filter_messages` will strip it from
    /// the LLM context.
    ///
    /// # Parameters
    /// - `command`: shell command string to execute (without the `!` prefix).
    /// - `exclude_from_context`: `!!` prefix semantics — when true, the result does not enter the LLM context.
    /// - `on_chunk`: streaming callback per chunk of sanitized output text (for incremental UI rendering); None = discard.
    ///
    /// # Return Value
    /// `BashOutcome` (exit code / cancellation / truncation / full output path) for the UI to backfill `ChatRow::Bash`.
    ///
    /// # Errors
    /// - `AgentSessionError::Execution`: low-level execution errors (spawn/wait/temp file etc.).
    ///
    /// # Edge Cases
    /// - `&self`: only appends messages + registers the exclusion set (under lock), no conflict with `&mut self` prompt.
    /// - v1 does not wire up timeout/cancellation/detached registration (keeps the established T-AP-04-2 boundary, see task doc).
    pub async fn execute_bash(
        &self,
        command: &str,
        exclude_from_context: bool,
        on_chunk: Option<Arc<dyn Fn(&str) + Send + Sync>>,
    ) -> Result<BashOutcome, AgentSessionError> {
        let env = ExecutionEnv::new(PathBuf::from(self.cwd()), None, None, None);
        let options = ShellCaptureOptions {
            cwd: None, // default to env.base_dir (= session cwd)
            env: None,
            timeout: None,
            cancellation: None,
            on_chunk: on_chunk.unwrap_or_else(|| Arc::new(|_| {})),
        };
        let captured = execute_shell_with_capture(&env, command, &options).await?;

        let timestamp = now_millis();
        let full_output_text = captured
            .full_output_path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned());
        let message = build_bash_user_message(
            command,
            &captured.output,
            captured.exit_code,
            captured.cancelled,
            captured.truncated,
            full_output_text.as_deref(),
            timestamp,
        );
        self.session_store.append_message(&message);
        if exclude_from_context {
            if let Ok(mut excluded) = self.excluded.lock() {
                excluded.insert(timestamp.to_string());
            }
        }
        Ok(BashOutcome {
            exit_code: captured.exit_code,
            cancelled: captured.cancelled,
            truncated: captured.truncated,
            full_output_path: full_output_text,
        })
    }
    // consumer: T-AP-11-1 (interactive `handle_bash_command`)

    /// Abort the current agent run.
    ///
    /// # Behavior
    /// Calls `self.agent.abort()` (triggering the cancellation token); the low-level loop exits at a safe point.
    ///
    /// # Edge Cases
    /// - `&self`: only manipulates the cancellation token, no conflict with `&mut self` of prompt (safe to call across threads)
    /// - No-op when idle
    pub fn abort(&self) {
        self.agent.abort();
    }

    /// Subscribe to session events.
    ///
    /// # Behavior
    /// **Passes through** `agent.subscribe(listener)` — the listener is registered directly with the engine, not going through
    /// Relay through `AgentSession`; `AgentSession` does not maintain its own listener list. **The caller decides
    /// which `AgentEvent` variants to subscribe to**. The callback is invoked synchronously on the agent's run thread and should return quickly.
    ///
    /// # Return Value
    /// Returns the engine's unsubscribe closure (removes the listener when called).
    pub fn subscribe(
        &self,
        listener: Box<dyn Fn(&AgentEvent) + Send + Sync>,
    ) -> Box<dyn FnOnce() + Send> {
        self.agent.subscribe(listener)
    }

    // ── State accessors (all delegate to agent snapshot accessors, nothing stored) ─────────────

    /// Current model snapshot (delegated via `agent.model()`, owned clone).
    pub fn model(&self) -> Model {
        self.agent.model()
    }
    // Consumers: T-AP-07-1 (print mode footer), T-AP-11-1 (interactive mode footer)

    /// Current thinking level (delegated via `agent.thinking_level()`).
    pub fn thinking_level(&self) -> ThinkingLevel {
        self.agent.thinking_level()
    }

    /// List of currently active tool names (delegated via the `agent.tools()` snapshot; returns an owned `Vec<String>`;
    /// references cannot be borrowed while the lock is held).
    pub fn active_tool_names(&self) -> Vec<String> {
        self.agent
            .tools()
            .iter()
            .map(|tool| tool.name().to_string())
            .collect()
    }

    /// Whether a run is currently in progress (delegated via `agent.is_run()`).
    /// Semantics: not "whether the protocol is streaming", but whether the agent is currently executing a run (busy state).
    pub fn is_run(&self) -> bool {
        self.agent.is_run()
    }

    /// Whether idle (= not running).
    pub fn is_idle(&self) -> bool {
        !self.agent.is_run()
    }

    /// Session ID (read via delegation to `session_store.session_id()`).
    pub fn session_id(&self) -> String {
        self.session_store.session_id()
    }

    /// Working directory (read via delegation to `session_store.cwd()`).
    pub fn cwd(&self) -> &str {
        self.session_store.cwd()
    }

    // ── Model / thinking level management (delegated to engine + persistence) ────────────────────

    /// Set the model.
    ///
    /// # Behavior
    /// 1. First get provider and model_id from `model.provider`/`model.id` (zen-ai's `Model`
    ///    added the `provider` field, filling in the purchased provider value at the source of the persistence chain)
    /// 2. `self.agent.set_model(model)` (only changes `AgentState.model`, single source of truth)
    /// 3. `self.session_store.append_model_change(&provider, &model_id)` persists it
    ///
    /// # Edge Cases
    /// - Must not be called during streaming (caller should ensure idle)
    /// - `&self`: lock-free Facade; state is mutated through the engine lock, callable from any thread
    pub fn set_model(&self, model: Model) {
        let provider = model.provider.clone();
        let model_id = model.id.clone();
        self.agent.set_model(model);
        self.session_store.append_model_change(&provider, &model_id);
    }

    /// Set the thinking level.
    ///
    /// # Behavior
    /// 1. `self.agent.set_thinking_level(level)` (only changes `AgentState.thinking_level`, single source of truth; requires a new engine method)
    /// 2. `self.session_store.append_thinking_level_change(level)` persists it
    ///
    /// # Edge Cases
    /// - Must not be called during streaming (caller should ensure idle)
    /// - `&self`: lock-free Facade (same as set_model)
    pub fn set_thinking_level(&self, level: ThinkingLevel) {
        self.agent.set_thinking_level(level);
        self.session_store
            .append_thinking_level_change(&level.to_string())
    }

    /// Restore session history after construction (mirrors engine `Agent::set_messages`).
    ///
    /// Mutates only the underlying `AgentState.messages` (single source of truth), for use by the assembly phase (`create_initial_runtime_from_cli`)
    /// to seed the history of `SessionManager::build_context()`; pass an empty value for a new session to get empty history.
    ///
    /// # Consumers
    /// `create_initial_runtime_from_cli` (session/fork/resume assembly); `&self` (lock-free Facade).
    pub fn set_messages(&self, messages: Vec<Message>) {
        self.agent.set_messages(messages);
    }

    // ── Queue operations (agent engine queue + read-only UI snapshot) ──────────────

    /// Queue a steer message (injected after tool execution, before the next LLM call).
    ///
    /// # Edge Cases
    /// - `&self`: enqueues only via `Agent::steer`, does not block the prompt
    pub fn steer(&self, text: &str) {
        self.agent.steer(text);
    }

    /// Queue a follow-up message (injected after the agent goes idle).
    ///
    /// # Edge Cases
    /// - `&self`: enqueues only via `Agent::follow_up`, does not block the prompt
    pub fn follow_up(&self, text: &str) {
        self.agent.follow_up(text);
    }

    /// Get a snapshot of the current steer queue (read-only, delegated via engine `steering_messages()`).
    pub fn steering_messages(&self) -> Vec<String> {
        self.agent.steering_messages()
    }

    /// Get a snapshot of the current follow-up queue (read-only, delegated via engine `follow_up_messages()`).
    pub fn follow_up_messages(&self) -> Vec<String> {
        self.agent.follow_up_messages()
    }
}

/// Error type for AgentSession operations.
/// Phase 1 only wraps engine errors; Phase 2 appends more (e.g. "queueing during streaming without specifying deliverAs", etc.).
#[derive(Debug, thiserror::Error)]
pub enum AgentSessionError {
    /// Agent engine error
    #[error("agent error: {0}")]
    Agent(#[from] AgentLoopError),

    /// User bash execution error (`execute_bash` underlying shell channel, wired up in T-AP-04-2-fix)
    #[error("bash execution error: {0}")]
    Execution(#[from] ExecutionError),
}

/// Result snapshot of a one-shot user `!`/`!!` bash execution (T-AP-04-2-fix), for the UI to backfill `ChatRow::Bash`.
/// v1 subset aligned with pi `BashResult` (full output accounting: truncation + temp file path).
pub struct BashOutcome {
    /// Process exit code; None when cancelled, Some(code) on normal exit.
    pub exit_code: Option<i32>,
    /// Whether cancelled (race where the token was cancelled after channel execution).
    pub cancelled: bool,
    /// Whether tail truncation occurred.
    pub truncated: bool,
    /// Full temp file path for output (prefix `bash-`, suffix `.log`); exists only on overflow/truncation
    pub full_output_path: Option<String>,
}

/// Render the user's bash execution result as `Message::User` (aligned with pi's `bashExecutionToText`).
///
/// Text layout: `Ran \`{command}\`\n` + output code block (empty output = "(no output)") +
/// Cancel / non-zero exit code / truncated path suffix. Lets the LLM know "the user ran a command" with no orphan tool_call
/// (design 1 decision point of T-AP-04-2-fix).
///
/// # Parameters
/// - `command`: command string (without the `!` prefix)
/// - `output`: output text after tail truncation (channel exit, may be empty)
/// - `exit_code`: process exit code (`None` if cancelled)
/// - `cancelled`: whether cancelled
/// - `truncated`: whether tail truncation occurred
/// - `full_output_path`: full temp file path for output (only present when truncated)
/// - `timestamp`: message timestamp (ms), also the `!!` exclusion identity (`exclusion_key`)
///
/// # Edge Cases
/// - Pure function, no IO, easy to unit test; called by `execute_bash` after capture.
pub(crate) fn build_bash_user_message(
    command: &str,
    output: &str,
    exit_code: Option<i32>,
    cancelled: bool,
    truncated: bool,
    full_output_path: Option<&str>,
    timestamp: i64,
) -> Message {
    let mut text = format!("Ran `{command}`\n");
    if output.is_empty() {
        text.push_str("(no output)");
    } else {
        text.push_str(&format!("```\n{output}\n```"));
    }
    if cancelled {
        text.push_str("\n\n(command cancelled)");
    } else if let Some(code) = exit_code {
        if code != 0 {
            text.push_str(&format!("\n\nCommand exited with code {code}"));
        }
    }
    if truncated {
        if let Some(path) = full_output_path {
            text.push_str(&format!("\n\n[Output truncated. Full output: {path}]"));
        }
    }
    Message::User(UserMessage {
        content: ContentBlockList::Text(text),
        timestamp,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Mutex;
    use zen_ai::tool::ToolExecutionMode;
    use zen_ai::types::StreamFn;
    use zen_core::harness::agent_hooks::AgentLoopHooks;
    use zen_core::harness::types::AgentState;

    // ── Recording SessionStore mock ──

    #[derive(Default)]
    struct MockStore {
        model_changes: Mutex<Vec<(String, String)>>,
        thinking_changes: Mutex<Vec<String>>,
    }

    impl MockStore {
        fn model_changes(&self) -> Vec<(String, String)> {
            self.model_changes.lock().unwrap().clone()
        }
        fn thinking_changes(&self) -> Vec<String> {
            self.thinking_changes.lock().unwrap().clone()
        }
    }

    impl SessionStore for MockStore {
        fn append_message(&self, _message: &Message) {}
        fn append_model_change(&self, provider: &str, model_id: &str) {
            self.model_changes
                .lock()
                .unwrap()
                .push((provider.to_string(), model_id.to_string()));
        }
        fn append_thinking_level_change(&self, level: &str) {
            self.thinking_changes
                .lock()
                .unwrap()
                .push(level.to_string());
        }
        fn session_id(&self) -> String {
            "session-1".to_string()
        }
        fn cwd(&self) -> &str {
            "/work/dir"
        }
    }

    // ── Agent construction helpers (same test conventions as engine.rs) ──

    struct MockHooks;

    #[async_trait::async_trait]
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

    fn mock_stream_fn() -> StreamFn {
        std::sync::Arc::new(
            |_model: &Model,
             _ctx: &zen_ai::types::Context,
             _cfg: Option<zen_ai::api::openai_stream::StreamConfig>,
             _cancel: Option<tokio_util::sync::CancellationToken>| {
                Box::pin(async move {
                    let (_tx, rx): (
                        tokio::sync::mpsc::UnboundedSender<zen_ai::types::AssistantMessageEvent>,
                        tokio::sync::mpsc::UnboundedReceiver<zen_ai::types::AssistantMessageEvent>,
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

    fn test_agent() -> Agent {
        Agent::new(
            agent_state(),
            ToolExecutionMode::Sequential,
            std::sync::Arc::new(MockHooks),
            mock_stream_fn(),
        )
    }

    fn test_session(store: Arc<MockStore>) -> AgentSession {
        AgentSession::new(test_agent(), store)
    }

    // ── State accessors ──

    #[test]
    fn test_model_delegates_to_agent_state() {
        let m = test_model();
        let store = Arc::new(MockStore::default());
        let session = test_session(store);
        assert_eq!(session.model(), m);
    }

    #[test]
    fn test_thinking_level_and_is_run_idle() {
        let store = Arc::new(MockStore::default());
        let session = test_session(store);
        assert_eq!(session.thinking_level(), ThinkingLevel::Off);
        assert!(!session.is_run(), "Initially not running");
        assert!(session.is_idle(), "Not running means idle");
    }

    #[test]
    fn test_session_id_and_cwd_delegate_to_store() {
        let store = Arc::new(MockStore::default());
        let session = test_session(store);
        assert_eq!(session.session_id(), "session-1");
        assert_eq!(session.cwd(), "/work/dir");
    }

    #[test]
    fn test_active_tool_names_empty_for_no_tools() {
        let store = Arc::new(MockStore::default());
        let session = test_session(store);
        assert!(session.active_tool_names().is_empty());
    }

    // ── set_model / set_thinking_level (persistence + single source of truth) ──

    #[test]
    fn test_set_model_persists_and_updates_state() {
        let store = Arc::new(MockStore::default());
        let session = test_session(store.clone());

        let mut m2 = test_model();
        m2.id = "other-model".into();
        m2.provider = "huoshan".into();
        session.set_model(m2.clone());

        // Persist provider + model_id
        assert_eq!(
            store.model_changes(),
            vec![("huoshan".to_string(), "other-model".to_string())]
        );
        // Single-source update: model() reads the new model
        assert_eq!(session.model(), m2);
    }

    #[test]
    fn test_set_thinking_level_persists_and_updates_state() {
        let store = Arc::new(MockStore::default());
        let session = test_session(store.clone());

        session.set_thinking_level(ThinkingLevel::High);

        // Record one thinking-level change
        assert_eq!(store.thinking_changes().len(), 1);
        assert_eq!(session.thinking_level(), ThinkingLevel::High);
    }

    #[test]
    fn test_set_messages_restores_history_on_delegate() {
        // set_messages delegates to the engine and only mutates state.messages, without triggering any persistence
        let store = Arc::new(MockStore::default());
        let session = test_session(store.clone());

        let msg = Message::User(zen_ai::types::UserMessage {
            content: zen_ai::types::ContentBlockList::Text("History messages".into()),
            timestamp: 0,
        });
        session.set_messages(vec![msg]);
        // An empty restore (new session) neither panics nor triggers persistence
        session.set_messages(vec![]);

        // Delegates to the engine to mutate state directly: does not write model/thinking changes
        assert!(store.model_changes().is_empty());
        assert!(store.thinking_changes().is_empty());
    }

    // ── steer / follow_up / queue snapshot ──

    #[test]
    fn test_steer_and_snapshot() {
        let store = Arc::new(MockStore::default());
        let session = test_session(store);
        session.steer("hello");
        session.steer("world");
        assert_eq!(
            session.steering_messages(),
            vec!["hello".to_string(), "world".to_string()]
        );
    }

    #[test]
    fn test_follow_up_and_snapshot() {
        let store = Arc::new(MockStore::default());
        let session = test_session(store);
        session.follow_up("continue");
        assert_eq!(session.follow_up_messages(), vec!["continue".to_string()]);
    }

    // ── Note: the following methods are not unit-tested here because the agent field is private / depends on the event stream ──
    // - The persistent subscription in `new` (message_end → append_message) **is implemented** (see `new`), but triggering it
    //   requires agent.on_event or a prompt event stream; the agent field is private and unreachable. Recommendation: add an integration test
    // - `prompt`: depends on a full mock StreamFn event stream (AgentStart/MessageUpdate/Done...); recommended to add an integration test
    // - `abort`: the cancel token is a private agent field (Mutex<CancellationToken>), so unit tests cannot assert is_cancelled;
    //   the semantics (bumping the current run token, no-op when idle, not polluting the next round) are covered by abort/per-run regression tests in engine.rs
    // - Event reception in `subscribe`: likewise requires triggering agent.on_event, but agent is private and unreachable; the pass-through behavior itself is already
    //   covered by the subscribe unit tests in engine.rs

    // ── build_bash_user_message (pure function, T-AP-04-2-fix) ──

    #[test]
    fn test_build_bash_user_message_normal_output() {
        let msg =
            build_bash_user_message("ls -la", "file1\nfile2", Some(0), false, false, None, 1000);
        let Message::User(user) = msg else {
            panic!("expected User message");
        };
        let ContentBlockList::Text(text) = user.content else {
            panic!("expected text content");
        };
        assert_eq!(text, "Ran `ls -la`\n```\nfile1\nfile2\n```");
        assert_eq!(user.timestamp, 1000);
    }

    #[test]
    fn test_build_bash_user_message_no_output() {
        let msg = build_bash_user_message("true", "", Some(0), false, false, None, 1);
        let Message::User(user) = msg else {
            panic!("expected User message");
        };
        let ContentBlockList::Text(text) = user.content else {
            panic!("expected text content");
        };
        assert_eq!(text, "Ran `true`\n(no output)");
    }

    #[test]
    fn test_build_bash_user_message_cancelled() {
        let msg = build_bash_user_message("sleep 10", "partial", None, true, false, None, 2);
        let Message::User(user) = msg else {
            panic!("expected User message");
        };
        let ContentBlockList::Text(text) = user.content else {
            panic!("expected text content");
        };
        // Cancellation takes precedence over exit code: no exit code + (command cancelled) suffix
        assert_eq!(
            text,
            "Ran `sleep 10`\n```\npartial\n```\n\n(command cancelled)"
        );
    }

    #[test]
    fn test_build_bash_user_message_nonzero_exit_and_truncated() {
        let msg = build_bash_user_message(
            "make build",
            "error: boom",
            Some(1),
            false,
            true,
            Some("/tmp/bash-abc.log"),
            3,
        );
        let Message::User(user) = msg else {
            panic!("expected User message");
        };
        let ContentBlockList::Text(text) = user.content else {
            panic!("expected text content");
        };
        assert_eq!(
            text,
            concat!(
                "Ran `make build`\n```\nerror: boom\n```\n\nCommand exited with code 1\n\n[",
                "Output truncated. Full output: /tmp/bash-abc.log]"
            )
        );
    }
}
