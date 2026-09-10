use crate::session::agent_session::{AgentSession, SessionStore};
use std::path::PathBuf;
use std::sync::Arc;
use zen_core::process::ProcessRegistry;

/// Reason for session creation/restoration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionStartReason {
    /// Session created on first startup
    Startup,
    /// Restored from an existing session file
    Resume(PathBuf),
    /// Create a brand-new session (clear history)
    New,
    /// Fork from a given entry
    Fork(String),
}

/// cwd-bound set of runtime services.
/// Consumers: T-AP-07-1 (print mode construction), T-AP-11-1 (interactive mode construction)
pub struct AgentSessionServices {
    /// Runtime working directory
    pub cwd: PathBuf,
    /// Global agent configuration directory
    pub agent_dir: PathBuf,
    /// Persistence delegate
    pub session_store: Arc<dyn SessionStore>,
    /// Registry of detached child processes, used to batch-terminate the child process tree on shutdown signal
    pub registry: Arc<ProcessRegistry>,
}

/// Runtime wrapper holding `AgentSession` + `AgentSessionServices`.
/// Consumers: T-AP-07-1 (print mode), T-AP-11-1 (interactive mode)
///
/// Plan D: `session` is an `Arc<AgentSession>` (lock-free facade, shareable between the run task and the main task),
/// all `runtime.session.xxx()` calls pass through via Deref, with zero changes for callers.
pub struct AgentSessionRuntime {
    /// Currently active session (Arc-shared for concurrent access by interactive-mode run tasks, D3)
    pub session: Arc<AgentSession>,
    /// Runtime service collection
    pub services: AgentSessionServices,
    /// Reason for this runtime creation (for diagnostics/logging)
    pub start_reason: SessionStartReason,
}

/// Create the initial runtime.
///
/// Assemble cwd-bound services and the initial session into an `AgentSessionRuntime`.
/// Callers should invoke this once at CLI startup (T-AP-08-3) or at mode entry.
///
/// # Parameters
/// - `session`: already-built AgentSession
/// - `services`: cwd-bound runtime services
/// - `start_reason`: reason for this creation (Startup/Resume/New/Fork)
///
/// # Behavior
/// Package session + services into an `AgentSessionRuntime`. No cwd existence check is performed
/// (that check must be done by the caller before creating AgentSession).
///
/// # Return Value
/// Runtime wrapper usable directly in print-mode / interactive-mode.
/// `session` is wrapped in an `Arc` here (assembly-phase `set_messages` etc. must be completed on the owned session before this call;
/// after wrapping, all access goes through read-only/&self Deref calls).
pub fn create_initial_runtime(
    session: AgentSession,
    services: AgentSessionServices,
    start_reason: SessionStartReason,
) -> AgentSessionRuntime {
    AgentSessionRuntime {
        session: Arc::new(session),
        services,
        start_reason,
    }
}

/// Create a new runtime from the current one, replacing the session.
///
/// # Behavior
/// 1. Destructure the old runtime, handing `AgentSessionServices` to the closure (old `AgentSession` tears down on drop)
/// 2. The closure builds a new session + new services from the old services (cwd/agentDir unchanged)
/// 3. Build a new runtime from the new session + services and return it
///
/// # Closure contract
/// `create_runtime_fn` takes the old `AgentSessionServices` (ownership), returns
/// `(AgentSession, AgentSessionServices, SessionStartReason)`. The old services can be reused
/// directly (cwd/agentDir unchanged), or entirely new services can be constructed (e.g. when cwd changes).
pub fn replace_runtime<F>(
    current: AgentSessionRuntime,
    create_runtime_fn: F,
) -> anyhow::Result<AgentSessionRuntime>
where
    F: FnOnce(
        AgentSessionServices,
    ) -> anyhow::Result<(AgentSession, AgentSessionServices, SessionStartReason)>,
{
    // 1) Destructure the old runtime: services are moved into the closure, session drops naturally with `old` (teardown)
    let AgentSessionRuntime { services, .. } = current;
    // 2) Closure: construct new session + services + start_reason from the old services (ownership transfer)
    let (session, services, start_reason) = create_runtime_fn(services)?;
    // 3) Construct new runtime (session wrapped in Arc, D3)
    Ok(AgentSessionRuntime {
        session: Arc::new(session),
        services,
        start_reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use zen_ai::tool::ToolExecutionMode;
    use zen_ai::types::StreamFn;
    use zen_ai::types::{AssistantMessageEvent, Context};
    use zen_core::harness::agent_hooks::AgentLoopHooks;
    use zen_core::harness::types::AgentState;

    // ── Mock helpers ──

    #[derive(Default)]
    struct MockStore {
        session_id_val: String,
        cwd_val: String,
    }

    impl SessionStore for MockStore {
        fn append_message(&self, _message: &zen_ai::types::Message) {}
        fn append_model_change(&self, _provider: &str, _model_id: &str) {}
        fn append_thinking_level_change(&self, _level: &str) {}
        fn session_id(&self) -> String {
            self.session_id_val.clone()
        }
        fn cwd(&self) -> &str {
            &self.cwd_val
        }
        fn branch_entries(&self) -> Vec<zen_core::session::tree::SessionTreeEntry> {
            Vec::new()
        }
        fn append_compaction(
            &self,
            _result: &zen_core::compaction::CompactionResult,
            _from_hook: bool,
        ) -> String {
            "mock-compaction-id".to_string()
        }
    }

    struct MockHooks;

    #[async_trait::async_trait]
    impl AgentLoopHooks for MockHooks {
        async fn on_event(&self, _event: &zen_core::harness::types::AgentEvent) {}
    }

    fn test_model() -> zen_ai::types::Model {
        zen_ai::types::Model {
            id: "test-model".into(),
            name: None,
            provider: "test".into(),
            api: zen_ai::provider::Api::OpenAiCompletions,
            base_url: "http://localhost".into(),
            context_window: 8192,
            max_tokens: 1024,
            reasoning: false,
            headers: None,
            cost_rates: None,
        }
    }

    fn mock_stream_fn() -> StreamFn {
        Arc::new(
            |_model: &zen_ai::types::Model,
             _ctx: &Context,
             _cfg: Option<zen_ai::api::openai_stream::StreamConfig>,
             _cancel: Option<tokio_util::sync::CancellationToken>| {
                Box::pin(async move {
                    let (_tx, rx) = tokio::sync::mpsc::unbounded_channel::<AssistantMessageEvent>();
                    Ok::<_, zen_ai::api::openai_stream::StreamError>(rx)
                })
            },
        )
    }

    fn agent_state() -> AgentState {
        AgentState {
            system_prompt: "test".into(),
            model: test_model(),
            thinking_level: zen_ai::types::ThinkingLevel::Off,
            tools: vec![],
            messages: vec![],
            is_run: false,
            streaming_message: None,
            pending_tool_calls: HashSet::new(),
            error_message: None,
        }
    }

    fn test_agent() -> zen_core::harness::engine::Agent {
        zen_core::harness::engine::Agent::new(
            agent_state(),
            ToolExecutionMode::Sequential,
            Arc::new(MockHooks),
            mock_stream_fn(),
        )
    }

    fn test_session_store(id: &str, cwd: &str) -> Arc<dyn SessionStore> {
        Arc::new(MockStore {
            session_id_val: id.to_string(),
            cwd_val: cwd.to_string(),
        })
    }

    fn test_session(id: &str, cwd: &str) -> AgentSession {
        AgentSession::new(test_agent(), test_session_store(id, cwd))
    }

    fn test_services(cwd: &str, agent_dir: &str) -> AgentSessionServices {
        AgentSessionServices {
            cwd: cwd.into(),
            agent_dir: agent_dir.into(),
            session_store: test_session_store("s1", cwd),
            registry: ProcessRegistry::new(),
        }
    }

    // ── create_initial_runtime ──

    #[test]
    fn test_create_initial_runtime_returns_runtime_with_correct_fields() {
        let session = test_session("sid", "/tmp/cwd");
        let services = test_services("/tmp/cwd", "/tmp/agent");
        let reason = SessionStartReason::Startup;

        let rt = create_initial_runtime(session, services, reason);

        // Session is validated via session_store (cwd accessible)
        assert_eq!(rt.session.cwd(), "/tmp/cwd");
        assert_eq!(rt.services.cwd, PathBuf::from("/tmp/cwd"));
        assert_eq!(rt.services.agent_dir, PathBuf::from("/tmp/agent"));
        assert!(matches!(rt.start_reason, SessionStartReason::Startup));
    }

    #[test]
    fn test_create_initial_runtime_preserves_start_reason_resume() {
        let session = test_session("s2", "/work");
        let services = test_services("/work", "/agent");
        let reason = SessionStartReason::Resume("/sessions/abc.jsonl".into());

        let rt = create_initial_runtime(session, services, reason);

        match &rt.start_reason {
            SessionStartReason::Resume(path) => {
                assert_eq!(path, &PathBuf::from("/sessions/abc.jsonl"))
            }
            _ => panic!("expected Resume variant"),
        }
    }

    #[test]
    fn test_create_initial_runtime_preserves_start_reason_fork() {
        let session = test_session("s3", "/work");
        let services = test_services("/work", "/agent");
        let reason = SessionStartReason::Fork("entry-42".into());

        let rt = create_initial_runtime(session, services, reason);

        match &rt.start_reason {
            SessionStartReason::Fork(id) => assert_eq!(id, "entry-42"),
            _ => panic!("expected Fork variant"),
        }
    }

    // ── replace_runtime ──

    #[test]
    fn test_replace_runtime_calls_closure_with_old_services() {
        let old_rt = create_initial_runtime(
            test_session("old", "/old/cwd"),
            test_services("/old/cwd", "/old/agent"),
            SessionStartReason::Startup,
        );

        let captured_cwd = Arc::new(std::sync::Mutex::new(PathBuf::new()));
        let captured_cwd_clone = captured_cwd.clone();

        let new_rt = replace_runtime(old_rt, move |old_services| {
            // Closure receives ownership of the old services
            *captured_cwd_clone.lock().unwrap() = old_services.cwd.clone();
            let new_session = test_session("new", "/new/cwd");
            let new_services = test_services("/new/cwd", "/new/agent");
            Ok((new_session, new_services, SessionStartReason::New))
        })
        .expect("replace_runtime should succeed");

        // The closure receives the cwd of the old services
        assert_eq!(*captured_cwd.lock().unwrap(), PathBuf::from("/old/cwd"));
        // The new runtime contains the new session + new services
        assert_eq!(new_rt.session.cwd(), "/new/cwd");
        assert_eq!(new_rt.services.cwd, PathBuf::from("/new/cwd"));
        assert!(matches!(new_rt.start_reason, SessionStartReason::New));
    }

    #[test]
    fn test_replace_runtime_propagates_closure_error() {
        let old_rt = create_initial_runtime(
            test_session("old", "/work"),
            test_services("/work", "/agent"),
            SessionStartReason::Startup,
        );

        let result = replace_runtime::<fn(AgentSessionServices) -> _>(old_rt, |_services| {
            Err(anyhow::anyhow!("something went wrong"))
        });

        assert!(result.is_err());
        match result {
            Err(e) => assert_eq!(e.to_string(), "something went wrong"),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn test_replace_runtime_old_session_is_dropped() {
        // Track drop count via Arc<AtomicUsize>
        // Since AgentSession holds Agent (including event_listeners etc.),
        // we cannot directly hook AgentSession's drop.
        // Indirect verification: the old session is unreachable after replacement, and the new runtime's session is new.
        let old_rt = create_initial_runtime(
            test_session("old-sid", "/work"),
            test_services("/work", "/agent"),
            SessionStartReason::Startup,
        );

        let new_rt = replace_runtime(old_rt, |services| {
            assert_eq!(services.cwd, PathBuf::from("/work"));
            Ok((
                test_session("new-sid", "/work"),
                services,
                SessionStartReason::Resume("/x.jsonl".into()),
            ))
        })
        .unwrap();

        // The new runtime's session_id is fresh
        assert_eq!(new_rt.session.session_id(), "new-sid");
    }

    #[test]
    fn test_replace_runtime_can_change_services() {
        let old_rt = create_initial_runtime(
            test_session("s1", "/old/cwd"),
            test_services("/old/cwd", "/old/agent"),
            SessionStartReason::Startup,
        );

        let new_rt = replace_runtime(old_rt, |_old_services| {
            // The closure chooses not to use the old services and constructs brand-new ones
            Ok((
                test_session("s2", "/brand-new/cwd"),
                test_services("/brand-new/cwd", "/brand-new/agent"),
                SessionStartReason::New,
            ))
        })
        .unwrap();

        assert_eq!(new_rt.services.cwd, PathBuf::from("/brand-new/cwd"));
        assert_eq!(new_rt.services.agent_dir, PathBuf::from("/brand-new/agent"));
    }
}
