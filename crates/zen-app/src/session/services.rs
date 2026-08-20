// REQ-APP-08 (pi-360: thin out the main entry point) + REQ-APP-01 (AgentSessionServices wiring)
// Reference: reference/pi-original/packages/coding-agent/src/core/agent-session-services.ts
// Implementation notes: tasks/T-AP-08-3-implemented.md (modification points 3/4/5/6)

//! Thin wiring factory: extracts the "input args -> runnable AgentSessionRuntime" assembly from main into library-level functions.
//!
//! Corresponds to pi's merged form of `createAgentSessionServices` + `createAgentSessionFromServices`.
//! - `create_agent_session` is the generic wiring layer (counterpart of langchain's `create_agent`),
//!   with all inputs injected (model/system_prompt/tools/hooks/stream_fn); depends on no CLI types,
//!   so print/interactive modes can reuse the same factory.
//! - `create_initial_runtime_from_cli` is CLI-specific thin glue that feeds `CliArgs` parse results into the generic factory.
//! - After SettingsManager / ResourceLoader land in Phase 2, in `create_initial_runtime_from_cli`
//!   just append assembly steps (wrap-up of T-AP-02-3 / T-AP-05-3); main not rewritten.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use crate::cli::{parse_thinking_level, CliArgs};
use crate::config::get_agent_dir;
use crate::session::agent_session::{AgentSession, SessionStore};
use crate::session::runtime::{
    create_initial_runtime, AgentSessionRuntime, AgentSessionServices, SessionStartReason,
};
use crate::session::session_manager::SessionManager;
use crate::system_prompt::load_project_context_files;
use clap::Parser;
use tokio_util::sync::CancellationToken;
use zen_ai::api::stream_for;
use zen_ai::provider::resolve_provider;
use zen_ai::tool::{Tool, ToolContext, ToolExecutionMode};
use zen_ai::types::{ImageContent, Model, ThinkingLevel};
use zen_core::harness::agent_hooks::AgentLoopHooks;
use zen_core::harness::engine::Agent;
use zen_core::harness::system_prompt::{
    build_system_prompt, collect_tool_prompt_metadata, BuildSystemPromptOptions,
};
use zen_core::harness::types::AgentState;
use zen_core::session::tree::{SessionContext, SessionModel};
use zen_core::tools::factory::ToolFactory;

/// Phase 1 default empty hook implementation: all hook methods use `AgentLoopHooks` trait defaults
/// (no filtering, no transformation, no key injection, no tool-call interception, no event handling).
///
/// `pub` so examples / quick starts can directly construct an empty `Arc<dyn AgentLoopHooks>` and inject it
/// `create_agent_session`; modes needing events/transformation (print json / TUI) should pass custom hooks.
/// Note: print mode actually goes through `AgentSession::subscribe` serializing events rather than hooks (see T-AP-07-1).
pub struct NoopHooks;

impl AgentLoopHooks for NoopHooks {}

/// Generic assembly layer: build a runnable `AgentSession` from the resolved inputs.
///
/// This is langchain's `create_agent` counterpart — all parameters are injected externally; the factory only assembles.
///
/// # Parameters
/// - `model`: resolved model (with provider endpoint/protocol/auth)
/// - `system_prompt`: finished system prompt string (concatenated by the caller; this function does not assemble content)
/// - `tools`: list of available tools (created by the caller for the session environment, with cwd bound)
/// - `thinking_level`: thinking level
/// - `hooks`: hooks injected per mode (print serializes events / TUI forwards events / test collects events)
/// - `stream_fn`: protocol stream implementation fetched from the binding table by `model.api` (future protocols swapped in here)
/// - `session_store`: persistence delegate
///
/// # Behavior
/// 1. Assemble AgentState (model + system prompt + tools + empty message history + idle state)
/// 2. Construct the Agent engine (including cancel token)
/// 3. Construct the AgentSession (establishes a resident persistent subscription)
///
/// # Edge Cases
/// - This function touches no CLI / filesystem / environment variables — all inputs are resolved and passed in by the caller.
/// - The returned AgentSession can be consumed directly by print/interactive modes, or loaded into the runtime.
///
/// # Consumers
/// - `create_initial_runtime_from_cli` (same file, CLI glue layer)
/// - T-AP-07-1 (print mode, Phase 2), T-AP-11-1 (interactive mode, Phase 2)
pub fn create_agent_session(
    model: Model,
    system_prompt: String,
    tools: Vec<Arc<dyn Tool>>,
    thinking_level: ThinkingLevel,
    hooks: Arc<dyn AgentLoopHooks>,
    stream_fn: zen_ai::types::StreamFn,
    session_store: Arc<dyn SessionStore>,
) -> AgentSession {
    let state = AgentState {
        system_prompt,
        model,
        thinking_level,
        tools,
        messages: vec![],
        is_run: false,
        streaming_message: None,
        pending_tool_calls: HashSet::new(),
        error_message: None,
    };

    let agent = Agent::new(
        state,
        ToolExecutionMode::Parallel,
        hooks,
        stream_fn,
        CancellationToken::new(),
    );
    AgentSession::new(agent, session_store)
}

/// Thin CLI-specific glue: parse `CliArgs` + the already-selected session manager into model/tools/system prompt, and assemble the full runtime.
///
/// This is main's core call point — main only does parse → (select session) → call this function → dispatch, these layers.
///
/// # Parameters
/// - `args`: parsed CLI arguments
/// - `cwd`: current working directory (obtained by the caller, for test injection)
/// - `session_manager`: **session manager selected by the caller (main)**. Session selection (`in_memory`/`create`/`open`/
///   `continue_recent`/`fork_from`) is the caller's responsibility; this function only "assembles a runnable session from the given manager".
///
/// # Behavior
/// 1. Resolve model and thinking (CLI flags take precedence, otherwise restore from session `build_context`, then defaults)
/// 2. Create the tool factory and assemble the tool list (binds cwd + session_id)
/// 3. Assemble the system prompt (AGENTS.md context + --system-prompt/--append-system-prompt)
/// 4. Look up the protocol stream implementation from the binding table by model.api (stream_for from T-AI-02-8)
/// 5. Build AgentSession via the generic factory create_agent_session, and feed back the `build_context()` session history
///    (`set_messages`, aligned with pi `agent.state.messages = buildSessionContext().messages`; empty for a new session)
/// 6. Assemble AgentSessionServices + create_initial_runtime
///
/// # Return Value
/// The fully assembled `AgentSessionRuntime` (session + services + session store)
///
/// # Edge Cases
/// - Phase 1 injects no skills (append here after T-AP-03-3 lands)
/// - `--session` only supports full paths (partial UUID resolution is T-AP-08-5)
/// - Phase 2's SettingsManager/ResourceLoader append their assembly steps within this function.
///
/// # Consumers
/// T-AP-08-3 (main's node 4 call point, session selected by main)
pub fn create_initial_runtime_from_cli(
    args: &CliArgs,
    cwd: PathBuf,
    session_manager: SessionManager,
) -> anyhow::Result<AgentSessionRuntime> {
    // 0. Restore context from session (model/thinking/history). New session (empty tree) → messages empty, model None, naturally unified.
    let ctx = session_manager.build_context();

    // 1. Resolve model and thinking: CLI flags take precedence, otherwise restore from session ctx, then defaults
    let (mut model, mut thinking_level) = resolve_model(args);
    apply_session_restore(args, &ctx, &mut model, &mut thinking_level);

    // 2. Agent config directory (needed for AGENTS.md loading and default paths)
    let agent_dir = get_agent_dir()?;

    // 3. Session ID (supplied by a manager chosen by the caller; the session_id of the persisted/forked session)
    let cwd_str = cwd.to_string_lossy().into_owned();
    let session_id = session_manager.session_id();

    // 4. Tool factory: bind cwd + session_id
    let tool_context = ToolContext::new(cwd.clone(), Some(session_id.clone()), None);
    let factory = ToolFactory::new(tool_context);
    let tools = factory.create_all().tools;

    // 5. System prompt: AGENTS.md context + CLI override
    let tool_refs: Vec<&dyn Tool> = tools.iter().map(|t| t.as_ref()).collect();
    let meta = collect_tool_prompt_metadata(&tool_refs);
    let system_prompt = build_system_prompt(&BuildSystemPromptOptions {
        custom_prompt: args.system_prompt.clone(),
        selected_tools: None,
        tool_snippets: meta.tool_snippets,
        prompt_guidelines: meta.prompt_guidelines,
        append_system_prompt: if args.append_system_prompt.is_empty() {
            None
        } else {
            Some(args.append_system_prompt.join("\n"))
        },
        cwd: cwd_str,
        skills: vec![],
        context_files: if args.no_context_files {
            vec![]
        } else {
            load_project_context_files(&cwd, &agent_dir)
        },
    });

    // 6. Protocol stream implementation + common assembly (stream_for binding table from T-AI-02-8)
    let model_api = model.api;
    let stream_fn = stream_for(model_api).expect(
        "Protocol referenced by model.api not implemented: currently supports OpenAiCompletions, see T-AI-02-8 stream_for binding table",
    );

    let session_store: Arc<dyn SessionStore> = Arc::new(session_manager);
    let mut session = create_agent_session(
        model,
        system_prompt,
        tools,
        thinking_level,
        Arc::new(NoopHooks),
        stream_fn,
        session_store.clone(),
    );
    // Replay session history (aligned with pi; new session ctx.messages is empty)
    session.set_messages(ctx.messages);

    // 7. Runtime assembly
    let services = AgentSessionServices {
        cwd,
        agent_dir,
        session_store,
    };
    Ok(create_initial_runtime(
        session,
        services,
        SessionStartReason::Startup,
    ))
}

/// When the CLI does not explicitly override provider/model/thinking, restore them from the session `SessionContext`.
/// Aligned with pi `createAgentSession`: the `existingSession.model`/`existingSession.thinkingLevel` fallback chain.
///
/// # Rules
/// - model: restore from `ctx.model` (provider+model_id) only when neither `--provider` nor `--model` is given;
/// - thinking: restore from `ctx.thinking_level` only when neither `--thinking` nor the `model id:thinking` shorthand is given.
///
/// # Consumers
/// Called only internally by `create_initial_runtime_from_cli` (same file).
fn apply_session_restore(
    args: &CliArgs,
    ctx: &SessionContext,
    model: &mut Model,
    thinking: &mut ThinkingLevel,
) {
    // model: not specified via CLI → restore from session
    if args.provider.is_none() && args.model.is_none() {
        if let Some(SessionModel { provider, model_id }) = &ctx.model {
            if let Some(p) = resolve_provider(provider) {
                let mut restored = Model::from_provider(p.as_ref());
                restored.id = model_id.clone();
                *model = restored;
            }
        }
    }
    // thinking: not specified via CLI (incl. model:thinking shorthand) → restore from session
    let has_model_thinking_short = args.model.as_deref().is_some_and(|s| s.contains(':'));
    if args.thinking.is_none() && !has_model_thinking_short {
        if let Ok(level) = parse_thinking_level(&ctx.thinking_level) {
            *thinking = level;
        }
    }
}

/// Resolve the model and thinking level specified on the CLI.
///
/// Rules (aligned with the Phase 1 subset of pi's `resolveCliModel`):
/// - Base model: `Model::from_env()` (`ZEN_PROVIDER` + per-provider env vars)
/// - `--provider <name>`: resolve the provider and rebuild the model (unknown provider → warn, keep the default)
/// - `--model <arg>`: supports three forms——
///   - `provider/id`: provider override + id override
///   - `id`: id override only
///   - `id:thinking`: id override + thinking shorthand (parsed by `parse_thinking_level`)
/// - Thinking level priority: `--thinking` flag > `--model "id:thinking"` shorthand > default `Off`
///
/// # Return Value
/// `(Model, ThinkingLevel)`: the resolved model and thinking level.
///
/// # Consumers
/// Called only internally by `create_initial_runtime_from_cli` (same file).
fn resolve_model(args: &CliArgs) -> (Model, ThinkingLevel) {
    // --provider override: rebuild the model directly from the provider to avoid Model::from_env panicking on an unknown
    // ZEN_PROVIDER env var (aligned with the zen-ai contract; unknown providers go through the warning path)
    let mut model = if let Some(provider_name) = &args.provider {
        match resolve_provider(provider_name) {
            Some(p) => Model::from_provider(p.as_ref()),
            None => {
                eprintln!("Warning: unknown provider '{provider_name}', using default");
                Model::from_env()
            }
        }
    } else {
        Model::from_env()
    };

    // --model override: three forms——provider/id, id, id:thinking
    let mut model_thinking = None;
    if let Some(model_arg) = &args.model {
        // Split provider/id
        let (provider_part, id_part) = match model_arg.split_once('/') {
            Some((p, id)) => (Some(p), id),
            None => (None, model_arg.as_str()),
        };
        // Split id:thinking
        let (id, thinking_part) = match id_part.split_once(':') {
            Some((id, t)) => (id, Some(t)),
            None => (id_part, None),
        };

        if let Some(p) = provider_part {
            match resolve_provider(p) {
                Some(provider) => model = Model::from_provider(provider.as_ref()),
                None => eprintln!("Warning: unknown provider '{p}', using default"),
            }
        }
        model.id = id.to_string();
        if let Some(t) = thinking_part {
            model_thinking = parse_thinking_level(t).ok();
        }
    }

    // thinking level: --thinking flag takes precedence, then the model short name, default Off
    let thinking_level = args
        .thinking
        .or(model_thinking)
        .unwrap_or(ThinkingLevel::Off);

    (model, thinking_level)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper: construct CliArgs (aligned with args.rs test conventions).
    fn parse(args: &[&str]) -> CliArgs {
        let mut full: Vec<&str> = Vec::with_capacity(args.len() + 1);
        full.push("zen");
        full.extend_from_slice(args);
        CliArgs::try_parse_from(full).unwrap()
    }

    // ── resolve_model ────────────────────────────────────────────

    #[test]
    fn model_flag_overrides_id() {
        let args = parse(&["--model", "my-model"]);
        let (model, thinking) = resolve_model(&args);
        assert_eq!(model.id, "my-model");
        assert_eq!(thinking, ThinkingLevel::Off);
    }

    #[test]
    fn model_provider_slash_id_sets_provider_and_id() {
        let args = parse(&["--model", "huoshan/ark-model-x"]);
        let (model, _) = resolve_model(&args);
        assert_eq!(model.provider, "huoshan");
        assert_eq!(model.id, "ark-model-x");
    }

    #[test]
    fn model_thinking_suffix_sets_thinking() {
        let args = parse(&["--model", "my-model:high"]);
        let (model, thinking) = resolve_model(&args);
        assert_eq!(model.id, "my-model");
        assert_eq!(thinking, ThinkingLevel::High);
    }

    #[test]
    fn thinking_flag_overrides_model_suffix() {
        let args = parse(&["--model", "my-model:high", "--thinking", "off"]);
        let (_, thinking) = resolve_model(&args);
        assert_eq!(thinking, ThinkingLevel::Off);
    }

    #[test]
    fn provider_flag_overrides_default_provider() {
        let args = parse(&["--provider", "huoshan"]);
        let (model, _) = resolve_model(&args);
        assert_eq!(model.provider, "huoshan");
    }

    #[test]
    fn unknown_provider_warns_and_falls_back() {
        let args = parse(&["--provider", "nonexistent"]);
        let (model, _) = resolve_model(&args);
        // Fall back to default provider (no panic)
        assert!(!model.provider.is_empty());
    }

    #[test]
    fn default_model_from_env() {
        let args = parse(&[]);
        let (model, thinking) = resolve_model(&args);
        assert!(!model.id.is_empty());
        assert_eq!(thinking, ThinkingLevel::Off);
    }

    /// Smoke test for `create_initial_runtime_from_cli`: no CLI flags (zero config) + fake cwd should assemble a runtime.
    /// No real LLM call is triggered (print stub does not run the prompt).
    #[test]
    fn create_runtime_zero_config_smoke() {
        let args = parse(&[]);
        let cwd = std::env::temp_dir();
        let manager = SessionManager::in_memory(&cwd.to_string_lossy());
        let runtime = create_initial_runtime_from_cli(&args, cwd.clone(), manager)
            .expect("Zero config should be able to assemble runtime");
        // Model has a default value (env or provider default chain); session is accessible
        assert_eq!(runtime.services.cwd, cwd);
        assert_eq!(runtime.session.session_id().len() >= 36, true);
    }

    // ── apply_session_restore ──────────────────────────────────

    /// Construct a session context carrying model/thinking records (directly builds zen-core SessionContext, avoiding file I/O).
    fn restored_ctx(
        provider: &str,
        model_id: &str,
        thinking: &str,
    ) -> zen_core::session::tree::SessionContext {
        zen_core::session::tree::SessionContext {
            messages: vec![],
            thinking_level: thinking.into(),
            model: Some(zen_core::session::tree::SessionModel {
                provider: provider.into(),
                model_id: model_id.into(),
            }),
            active_tool_names: None,
        }
    }

    fn restored_ctx_from_none() -> zen_core::session::tree::SessionContext {
        zen_core::session::tree::SessionContext {
            messages: vec![],
            thinking_level: "off".into(),
            model: None,
            active_tool_names: None,
        }
    }

    #[test]
    fn restore_applies_when_cli_has_no_model_flags() {
        let args = parse(&[]);
        let (mut model, mut thinking) = resolve_model(&args);
        let ctx = restored_ctx("huoshan", "ark-restored-model", "high");

        apply_session_restore(&args, &ctx, &mut model, &mut thinking);

        assert_eq!(model.provider, "huoshan");
        assert_eq!(model.id, "ark-restored-model");
        assert_eq!(thinking, ThinkingLevel::High);
    }

    #[test]
    fn restore_skipped_when_cli_provides_model() {
        // --model explicit → don't restore session model (but still… restore thinking when not given)
        let args = parse(&["--model", "my-model"]);
        let (mut model, mut thinking) = resolve_model(&args);
        let ctx = restored_ctx("huoshan", "ark-restored-model", "high");

        apply_session_restore(&args, &ctx, &mut model, &mut thinking);

        assert_eq!(model.id, "my-model", "CLI --model takes precedence, not overridden");
        assert_eq!(model.provider, "opencode-go", "provider not overridden");
        // thinking not explicitly specified (model shorthand has no colon) → restore
        assert_eq!(thinking, ThinkingLevel::High);
    }

    #[test]
    fn restore_skipped_when_cli_provides_thinking() {
        let args = parse(&["--thinking", "low"]);
        let (mut model, mut thinking) = resolve_model(&args);
        let ctx = restored_ctx("huoshan", "ark", "high");

        apply_session_restore(&args, &ctx, &mut model, &mut thinking);

        assert_eq!(thinking, ThinkingLevel::Low, "CLI --thinking takes precedence");
        // model not provided → restore
        assert_eq!(model.provider, "huoshan");
    }

    #[test]
    fn restore_noop_on_empty_session_context() {
        let args = parse(&[]);
        let (model, thinking) = resolve_model(&args);
        let ctx = restored_ctx_from_none();
        let mut model = model;
        let mut thinking = thinking;

        apply_session_restore(&args, &ctx, &mut model, &mut thinking);

        // New session: no session model/thinking to restore, keep resolve_model defaults (leave untouched)
        assert!(!model.id.is_empty());
        assert_eq!(thinking, ThinkingLevel::Off);
    }
}
