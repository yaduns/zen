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
use std::sync::{Arc, Mutex};

use crate::cli::{parse_thinking_level, CliArgs};
use crate::config::defaults::DEFAULT_THINKING_LEVEL;
use crate::config::get_agent_dir;
use crate::config::settings::{SettingsData, SettingsManager};
use crate::prompt_templates::{load_prompt_templates, LoadPromptTemplatesOptions};
use crate::session::agent_session::{AgentSession, SessionStore};
use crate::session::runtime::{
    create_initial_runtime, AgentSessionRuntime, AgentSessionServices, SessionStartReason,
};
use crate::session::session_manager::SessionManager;
use crate::skills::{load_skills, LoadSkillsOptions};
use crate::system_prompt::load_project_context_files;
use clap::Parser;
use tokio_util::sync::CancellationToken;
use zen_ai::api::stream_for;
use zen_ai::provider::resolve_provider;
use zen_ai::tool::{Tool, ToolContext, ToolExecutionMode};
use zen_ai::types::{ImageContent, Message, Model, ThinkingLevel};
use zen_core::env::ShellConfig;
use zen_core::harness::agent_hooks::AgentLoopHooks;
use zen_core::harness::engine::Agent;
use zen_core::harness::message_filter::filter_excluded;
use zen_core::harness::system_prompt::{
    build_system_prompt, collect_tool_prompt_metadata, BuildSystemPromptOptions,
};
use zen_core::harness::types::{
    AfterToolCallContext, AfterToolCallResult, AgentEvent, AgentLoopTurnUpdate, AgentState,
    BeforeToolCallContext, BeforeToolCallResult, PrepareNextTurnContext,
    ShouldStopAfterTurnContext,
};
use zen_core::paths::{resolve_path, PathInputOptions};
use zen_core::prompt_templates::PromptTemplate;
use zen_core::session::tree::{SessionContext, SessionModel};
use zen_core::skills::skill::Skill;
use zen_core::tools::factory::{ToolFactory, ToolSelection};

/// Phase 1 default empty hook implementation: all hook methods use `AgentLoopHooks` trait defaults
/// (no filtering, no transformation, no key injection, no tool-call interception, no event handling).
///
/// `pub` so examples / quick starts can directly construct an empty `Arc<dyn AgentLoopHooks>` and inject it
/// `create_agent_session`; modes needing events/transformation (print json / TUI) should pass custom hooks.
/// Note: print mode actually goes through `AgentSession::subscribe` serializing events rather than hooks (see T-AP-07-1).
pub struct NoopHooks;

impl AgentLoopHooks for NoopHooks {}

/// Wrapping hooks used for real assembly (T-AP-04-2-fix): merge the caller's hooks with the `excludeFromContext`
/// exclusion set stream.
///
/// `filter_messages` = first run the caller's hooks (NoopHooks = pass-through; print/TUI may pass custom ones)
/// then remove messages hit by the `!!` bash exclusion identity via `zen_core::harness::message_filter::filter_excluded`
/// (registered by `AgentSession::execute_bash`) — policy lives in the hooks layer, no markers added at the data layer
/// (per the ruling in `type-duplication-refactoring-plan.md`). All other methods delegate to `inner`.
///
/// `excluded` shares the same Arc with `AgentSession` (created inside `create_agent_session`).
pub struct SessionHooks {
    inner: Arc<dyn AgentLoopHooks>,
    excluded: Arc<Mutex<HashSet<String>>>,
}

#[async_trait::async_trait]
impl AgentLoopHooks for SessionHooks {
    /// After caller filtering, drop the `!!` bash messages per the exclusion set (T-AP-04-2-fix wiring).
    async fn filter_messages(&self, messages: Vec<Message>) -> Vec<Message> {
        let inner_filtered = self.inner.filter_messages(messages).await;
        let excluded = self.excluded.lock().unwrap();
        filter_excluded(inner_filtered, &excluded)
    }

    async fn transform_context(
        &self,
        messages: Vec<Message>,
        cancel: &CancellationToken,
    ) -> Vec<Message> {
        self.inner.transform_context(messages, cancel).await
    }

    async fn get_api_key(&self, provider: &str) -> Option<String> {
        self.inner.get_api_key(provider).await
    }

    async fn should_stop_after_turn(&self, context: &ShouldStopAfterTurnContext) -> bool {
        self.inner.should_stop_after_turn(context).await
    }

    async fn prepare_next_turn(
        &self,
        context: &PrepareNextTurnContext,
    ) -> Option<AgentLoopTurnUpdate> {
        self.inner.prepare_next_turn(context).await
    }

    async fn get_steering_messages(&self) -> Vec<Message> {
        self.inner.get_steering_messages().await
    }

    async fn get_follow_up_messages(&self) -> Vec<Message> {
        self.inner.get_follow_up_messages().await
    }

    async fn before_tool_call(
        &self,
        context: &BeforeToolCallContext,
        cancel: &CancellationToken,
    ) -> Result<Option<BeforeToolCallResult>, String> {
        self.inner.before_tool_call(context, cancel).await
    }

    async fn after_tool_call(
        &self,
        context: &AfterToolCallContext,
        cancel: &CancellationToken,
    ) -> Result<Option<AfterToolCallResult>, String> {
        self.inner.after_tool_call(context, cancel).await
    }

    async fn on_event(&self, event: &AgentEvent) {
        self.inner.on_event(event).await
    }
}

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
#[allow(clippy::too_many_arguments)] // All assembly-face parameters are inputs to session assembly (model/prompts/tools/thinking/hooks/streaming/storage/shell/skills/templates); grouping them would only hurt readability
pub fn create_agent_session(
    model: Model,
    system_prompt: String,
    tools: Vec<Arc<dyn Tool>>,
    thinking_level: ThinkingLevel,
    hooks: Arc<dyn AgentLoopHooks>,
    stream_fn: zen_ai::types::StreamFn,
    session_store: Arc<dyn SessionStore>,
    shell_config: ShellConfig,
    skills: Vec<Skill>,
    templates: Vec<PromptTemplate>,
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

    // T-AP-04-2-fix: shared exclusion set — one goes into SessionHooks (engine filtering), one into AgentSession
    // (execute_bash registers the `!!` identity); the caller's hooks are wrapped, not replaced (event/transform semantics unchanged).
    let excluded: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));
    let session_hooks: Arc<dyn AgentLoopHooks> = Arc::new(SessionHooks {
        inner: hooks,
        excluded: excluded.clone(),
    });
    let agent = Agent::new(state, ToolExecutionMode::Parallel, session_hooks, stream_fn);
    AgentSession::new_with_exclusions(
        agent,
        session_store,
        excluded,
        shell_config,
        skills,
        templates,
    )
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
/// - Resource loading (skills/prompt templates, T-AP-03-2 interim) is done **inside this function** — T-AP-05-3
///   takes over this assembly section once unified loading lands (see seam backfill in tasks/T-AP-03-2-implemented.md)
/// - `--session` only supports full paths (partial UUID resolution is T-AP-08-5)
/// - Phase 2's SettingsManager/ResourceLoader append their assembly steps within this function.
///
/// # Consumers
/// T-AP-08-3 (main's node 4 call point, session selected by main)
pub async fn create_initial_runtime_from_cli(
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

    // 2.5 Settings assembly (T-AP-02-3 wrap-up): load the two-tier settings, provide default_model/default_thinking_level
    //     fill-in (priority: CLI > session restore > settings > built-in defaults) and shell override config.
    //     A corrupt file degrades to empty + is recorded in the manager error queue (not surfaced at this assembly point; the interactive side's drain_errors handles that).
    let settings = SettingsManager::load(&cwd, &agent_dir);
    let merged = settings.merged();
    // settings fill-in (priority: CLI > session restore > settings > env/default): default_provider
    // Known provider matched → rebuild the model by provider (provider side); default_model only changes the id,
    // default_thinking_level fills in thinking. Extracted as a free function for unit testing (T-AP-02-3 additional fields).
    apply_settings_defaults(args, &ctx, &merged, &mut model, &mut thinking_level);

    // 2.6 Resource loading (T-AP-03-2 interim, taken over by T-AP-05-3 later): skills + prompt templates
    //     (default dir + CLI `--skill`/`--prompt-template` + merged settings.skills/prompts).
    //     `--no-prompt-templates` disables template loading (aligns with pi resource-loader branch);
    //     loading diagnostics are non-fatal (missing/broken files don't abort), log tracing warnings, not surfaced in v1.
    let skill_paths: Vec<PathBuf> = merged
        .skills
        .clone()
        .unwrap_or_default()
        .into_iter()
        .chain(args.skills.iter().cloned())
        .map(PathBuf::from)
        .collect();
    let prompt_paths: Vec<PathBuf> = merged
        .prompts
        .clone()
        .unwrap_or_default()
        .into_iter()
        .chain(args.prompt_templates.iter().cloned())
        .map(PathBuf::from)
        .collect();
    let skills: Vec<Skill>;
    let mut templates: Vec<PromptTemplate> = Vec::new();
    // Skills are always loaded (`--no-prompt-templates` only governs templates); templates are controlled by that flag (aligns with pi resource-loader branch)
    let skills_result = load_skills(&LoadSkillsOptions {
        cwd: cwd.clone(),
        agent_dir: agent_dir.clone(),
        skill_paths,
        include_defaults: true,
    })
    .await;
    skills = skills_result
        .skills
        .into_iter()
        .map(|loaded| loaded.skill)
        .collect();
    for d in &skills_result.diagnostics {
        if let crate::skills::SkillLoadDiagnostic::Warning { message, .. } = d {
            tracing::warn!(target: "zen::resources", "skill load: {message}");
        }
    }
    if !args.no_prompt_templates {
        let prompts_result = load_prompt_templates(&LoadPromptTemplatesOptions {
            cwd: cwd.clone(),
            agent_dir: agent_dir.clone(),
            prompt_paths,
            include_defaults: true,
        })
        .await;
        templates = prompts_result
            .prompt_templates
            .into_iter()
            .map(|loaded| loaded.template)
            .collect();
        for d in &prompts_result.diagnostics {
            tracing::warn!(target: "zen::resources", "prompt load: {}", d.message);
        }
    }

    // 3. Session ID (supplied by a manager chosen by the caller; the session_id of the persisted/forked session)
    let cwd_str = cwd.to_string_lossy().into_owned();
    let session_id = session_manager.session_id();

    // 4. Tool factory: binds cwd + session_id; selection by CLI flags (--tools/--exclude-tools/
    //    --no-tools/--no-builtin-tools, cli-tools-flags-fix — tool selection not landed in T-AP-08-3
    //    injection; pi's `createAgentSession` counterparts for tools/excludeTools/noTools.
    //    Default builtin group is [read, bash, edit, write]; `--tools read` yields a read-only session.
    let tool_context = ToolContext::new(cwd.clone(), Some(session_id.clone()), None);
    // Shell override config (T-AP-02-3-fix, Seam B): shell_path is normalized via paths (`~` expansion,
    // relative paths absolutized against cwd; on parse failure falls back to None = system default shell), command_prefix passed through as-is.
    let shell = ShellConfig {
        shell_path: merged
            .shell_path
            .as_deref()
            .and_then(|p| resolve_path(p, &cwd, &PathInputOptions::default()).ok()),
        command_prefix: merged.shell_command_prefix.clone(),
    };
    let factory = ToolFactory::with_shell(tool_context, shell.clone());
    let selection = ToolSelection {
        tools: args.tools.clone(),
        exclude_tools: args.exclude_tools.clone(),
        no_tools: args.no_tools,
        no_builtin_tools: args.no_builtin_tools,
    };
    let tools = factory.create_with_selection(&selection).tools;

    // 5. System prompt: AGENTS.md context + CLI overrides + skills (assembled and injected per T-AP-03-2, honoring
    //    the "Phase 1 does not inject skills (append here once T-AP-03-3 lands)" note)
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
        skills: skills.clone(),
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
    let session = create_agent_session(
        model,
        system_prompt,
        tools,
        thinking_level,
        Arc::new(NoopHooks),
        stream_fn,
        session_store.clone(),
        shell,
        skills,
        templates,
    );
    // Replay session history (aligned with pi; new session ctx.messages is empty) — call on owned session before Arc::new
    session.set_messages(ctx.messages);

    // 7. Runtime assembly
    let services = AgentSessionServices {
        cwd,
        agent_dir,
        session_store,
        registry: zen_core::process::ProcessRegistry::new(),
    };
    Ok(create_initial_runtime(
        session,
        services,
        SessionStartReason::Startup,
    ))
}

/// Apply the fill-in from settings' default_provider/default_model/default_thinking_level.
///
/// # Priority
/// CLI (--provider/--model/--thinking) > session restore (ctx.model/ctx.thinking_level) >
/// settings > env (ZEN_PROVIDER)/built-in defaults. Applied only when neither CLI nor session restore specifies:
/// `default_provider` hits a known provider → rebuild the model by provider (overriding the env-selected
/// provider), `default_model` only changes the id (consistent with existing T-AP-02-3-fix semantics),
/// unknown provider names are ignored (fall back to the original model, no error).
///
/// # Consumers
/// Called only internally by `create_initial_runtime_from_cli` (same file).
fn apply_settings_defaults(
    args: &CliArgs,
    ctx: &SessionContext,
    merged: &SettingsData,
    model: &mut Model,
    thinking: &mut ThinkingLevel,
) {
    if args.provider.is_none() && args.model.is_none() && ctx.model.is_none() {
        if let Some(provider_name) = merged.default_provider.as_deref() {
            if let Some(provider) = resolve_provider(provider_name) {
                *model = Model::from_provider(provider.as_ref());
            }
        }
        if let Some(default_model) = &merged.default_model {
            model.id = default_model.clone();
        }
    }
    let has_model_thinking_short = args.model.as_deref().is_some_and(|s| s.contains(':'));
    if args.thinking.is_none() && !has_model_thinking_short && ctx.thinking_level.trim().is_empty()
    {
        if let Some(level) = merged.default_thinking_level {
            *thinking = level;
        }
    }
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

    // Thinking level: `--thinking` flag first, then model short name, else DEFAULT_THINKING_LEVEL (requirement ruled High)
    let thinking_level = args
        .thinking
        .or(model_thinking)
        .unwrap_or(DEFAULT_THINKING_LEVEL);

    (model, thinking_level)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zen_ai::types::{ContentBlockList, UserMessage};

    use std::ffi::OsString;
    use std::sync::Mutex;

    /// Test helper: serializes tests touching environment variables (same pattern as config/mod.rs — `resolve_model` reads
    /// `ZEN_PROVIDER`; parallel tests mutating it race with each other).
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Test helper: temporarily removes an environment variable, restoring the original value on scope exit (stays unset if it was originally unset).
    struct EnvVarGuard {
        key: &'static str,
        old: Option<OsString>,
    }

    impl EnvVarGuard {
        fn remove(key: &'static str) -> Self {
            let old = std::env::var_os(key);
            std::env::remove_var(key);
            Self { key, old }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.old {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }

    /// Test helper: construct CliArgs (aligned with args.rs test conventions).
    fn parse(args: &[&str]) -> CliArgs {
        let mut full: Vec<&str> = Vec::with_capacity(args.len() + 1);
        full.push("zen");
        full.extend_from_slice(args);
        CliArgs::try_parse_from(full).unwrap()
    }

    /// Test helper: empty session context (no restore).
    fn empty_ctx() -> SessionContext {
        SessionContext {
            messages: vec![],
            thinking_level: String::new(),
            model: None,
            active_tool_names: None,
        }
    }

    // ── apply_settings_defaults (default_provider fallback, additional fields for T-AP-02-3) ─────

    #[test]
    fn settings_default_thinking_applies_on_fresh_session() {
        // Regression (fix verified by T-AP-02-3-fix feedback): new session ctx.thinking_level is an empty string (no record),
        // settings.defaultThinkingLevel must take effect as the fallback — previously build_session_context faked an empty session
        // as "off", causing this gate to wrongly skip; see fix in zen-core/session/context.rs.
        let args = parse(&[]);
        let ctx = empty_ctx(); // thinking_level = ""
        let merged = SettingsData {
            default_thinking_level: Some(ThinkingLevel::Medium),
            ..Default::default()
        };
        let (mut model, mut thinking) = resolve_model(&args);
        apply_settings_defaults(&args, &ctx, &merged, &mut model, &mut thinking);
        assert_eq!(thinking, ThinkingLevel::Medium);
    }

    #[test]
    fn settings_default_provider_rebuilds_model_when_unset() {
        // Env isolation: assert on the default provider when "ZEN_PROVIDER is unset" (opencode-go); explicitly remove
        // the variable so the test isn't affected by the shell environment (e.g. ZEN_PROVIDER=huoshan) (provider.rs:86 reads env).
        let _l = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::remove("ZEN_PROVIDER");
        let args = parse(&[]);
        let ctx = empty_ctx();
        let merged = SettingsData {
            default_provider: Some("huoshan".into()),
            ..Default::default()
        };
        let (mut model, mut thinking) = resolve_model(&args);
        assert_eq!(model.provider, "opencode-go"); // env default
        apply_settings_defaults(&args, &ctx, &merged, &mut model, &mut thinking);
        // settings override the env-selected provider
        assert_eq!(model.provider, "huoshan");
        assert_eq!(model.api, zen_ai::provider::Api::OpenAiCompletions);
    }

    #[test]
    fn settings_default_provider_ignored_when_cli_provider_set() {
        let args = parse(&["--provider", "huoshan"]);
        let ctx = empty_ctx();
        let merged = SettingsData {
            default_provider: Some("opencode-go".into()),
            ..Default::default()
        };
        let (mut model, mut thinking) = resolve_model(&args);
        apply_settings_defaults(&args, &ctx, &merged, &mut model, &mut thinking);
        // CLI --provider takes precedence; settings do not override
        assert_eq!(model.provider, "huoshan");
    }

    #[test]
    fn settings_default_model_applies_id_within_settings_provider() {
        let args = parse(&[]);
        let ctx = empty_ctx();
        let merged = SettingsData {
            default_provider: Some("huoshan".into()),
            default_model: Some("ark-xyz".into()),
            ..Default::default()
        };
        let (mut model, mut thinking) = resolve_model(&args);
        apply_settings_defaults(&args, &ctx, &merged, &mut model, &mut thinking);
        assert_eq!(model.provider, "huoshan");
        assert_eq!(model.id, "ark-xyz");
    }

    #[test]
    fn settings_ignored_when_session_restored() {
        let args = parse(&[]);
        let ctx = SessionContext {
            model: Some(SessionModel {
                provider: "opencode-go".into(),
                model_id: "deepseek-v4-flash".into(),
            }),
            thinking_level: "high".into(),
            ..empty_ctx()
        };
        let merged = SettingsData {
            default_provider: Some("huoshan".into()),
            default_model: Some("ark-xyz".into()),
            default_thinking_level: Some(ThinkingLevel::Low),
            ..Default::default()
        };
        let (mut model, mut thinking) = resolve_model(&args);
        apply_session_restore(&args, &ctx, &mut model, &mut thinking);
        apply_settings_defaults(&args, &ctx, &merged, &mut model, &mut thinking);
        // session restore takes precedence; settings do not override provider/model/thinking
        assert_eq!(model.provider, "opencode-go");
        assert_eq!(model.id, "deepseek-v4-flash");
        assert_eq!(thinking, ThinkingLevel::High);
    }

    #[test]
    fn settings_unknown_provider_name_ignored() {
        // env isolation: same as settings_default_provider_rebuilds_model_when_unset — assertion based on
        // the default provider when ZEN_PROVIDER is unset.
        let _l = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::remove("ZEN_PROVIDER");
        let args = parse(&[]);
        let ctx = empty_ctx();
        let merged = SettingsData {
            default_provider: Some("not-a-provider".into()),
            default_model: Some("m".into()),
            ..Default::default()
        };
        let (mut model, mut thinking) = resolve_model(&args);
        apply_settings_defaults(&args, &ctx, &merged, &mut model, &mut thinking);
        // unknown provider ignored (fall back to original model); default_model still only changes the id
        assert_eq!(model.provider, "opencode-go");
        assert_eq!(model.id, "m");
    }

    // ── resolve_model ────────────────────────────────────────────

    #[test]
    fn model_flag_overrides_id() {
        let args = parse(&["--model", "my-model"]);
        let (model, thinking) = resolve_model(&args);
        assert_eq!(model.id, "my-model");
        assert_eq!(
            thinking, DEFAULT_THINKING_LEVEL,
            "No thinking specified → requirement decision defaults to High"
        );
    }

    #[test]
    fn model_provider_slash_id_sets_provider_and_id() {
        let args = parse(&["--model", "huoshan/ark-model-x"]);
        let (model, thinking) = resolve_model(&args);
        assert_eq!(model.provider, "huoshan");
        assert_eq!(model.id, "ark-model-x");
        assert_eq!(thinking, DEFAULT_THINKING_LEVEL);
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
        assert_eq!(thinking, DEFAULT_THINKING_LEVEL);
    }

    /// Smoke test for `create_initial_runtime_from_cli`: no CLI flags (zero config) + fake cwd should assemble a runtime.
    /// No real LLM call is triggered (the print stub does not run the prompt); resource loading (T-AP-03-2) scans the real default directories,
    /// silently skipping missing ones without aborting.
    #[tokio::test]
    async fn create_runtime_zero_config_smoke() {
        let args = parse(&[]);
        let cwd = std::env::temp_dir();
        let manager = SessionManager::in_memory(&cwd.to_string_lossy());
        let runtime = create_initial_runtime_from_cli(&args, cwd.clone(), manager)
            .await
            .expect("Zero config should be able to assemble runtime");
        // Model has a default value (env or provider default chain); session is accessible
        assert_eq!(runtime.services.cwd, cwd);
        assert_eq!(runtime.session.session_id().len() >= 36, true);
    }

    // ── Tool selection flags (cli-tools-flags-fix: --tools/--exclude-tools/--no-tools/--no-builtin-tools) ──

    /// Assemble a runtime with the specified CLI flags and return the list of active tool names for the session.
    async fn runtime_tools(args: &[&str]) -> Vec<String> {
        let parsed = parse(args);
        let cwd = std::env::temp_dir();
        let manager = SessionManager::in_memory(&cwd.to_string_lossy());
        let runtime = create_initial_runtime_from_cli(&parsed, cwd, manager)
            .await
            .expect("Assemble runtime");
        runtime
            .session
            .active_tool_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    #[tokio::test]
    async fn cli_no_flags_creates_default_builtin_tools() {
        assert_eq!(
            runtime_tools(&[]).await,
            vec!["read", "bash", "edit", "write"],
            "No flags → default built-in group (aligned with pi defaultActiveToolNames)"
        );
    }

    #[tokio::test]
    async fn cli_no_tools_disables_all_tools() {
        assert!(runtime_tools(&["--no-tools"]).await.is_empty());
    }

    #[tokio::test]
    async fn cli_no_builtin_tools_disables_builtin_set() {
        assert!(
            runtime_tools(&["--no-builtin-tools"]).await.is_empty(),
            "v1 has no extension tools: same effect as --no-tools"
        );
    }

    #[tokio::test]
    async fn cli_tools_allowlist_gives_read_only_session() {
        assert_eq!(
            runtime_tools(&["--tools", "read"]).await,
            vec!["read"],
            "--tools read → read-only tool session (user motivation scenario)"
        );
    }

    #[tokio::test]
    async fn cli_exclude_tools_removes_from_default() {
        assert_eq!(
            runtime_tools(&["--exclude-tools", "bash,write"]).await,
            vec!["read", "edit"],
            "--exclude-tools bash,write → set difference (comma-separated + trim handled in args.rs)"
        );
    }

    #[tokio::test]
    async fn cli_tools_wins_over_no_tools() {
        // Aligned with pi: allow-list takes precedence over no_tools
        assert_eq!(
            runtime_tools(&["--no-tools", "--tools", "read"]).await,
            vec!["read"]
        );
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
            thinking_level: String::new(), // Empty string = no entry (context.rs convention); recovery noop keeps the default
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
        let baseline = resolve_model(&parse(&[])).0; // Baseline: the provider created by from_env in the same process environment
        let (mut model, mut thinking) = resolve_model(&args);
        let ctx = restored_ctx("huoshan", "ark-restored-model", "high");

        apply_session_restore(&args, &ctx, &mut model, &mut thinking);

        assert_eq!(model.id, "my-model", "CLI --model takes precedence, not overridden");
        assert_eq!(model.provider, baseline.provider, "provider not overridden");
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
        assert_eq!(thinking, DEFAULT_THINKING_LEVEL);
    }

    // ── SessionHooks (T-AP-04-2-fix wiring) ──

    /// Recording inner hooks: only records filter_messages input args and applies one marker transform; everything else uses defaults.
    struct RecordingHooks {
        seen: Arc<Mutex<Vec<Vec<Message>>>>,
    }

    #[async_trait::async_trait]
    impl AgentLoopHooks for RecordingHooks {
        async fn filter_messages(&self, messages: Vec<Message>) -> Vec<Message> {
            self.seen.lock().unwrap().push(messages.clone());
            messages
        }
    }

    fn user_msg(text: &str, ts: i64) -> Message {
        Message::User(UserMessage {
            content: ContentBlockList::Text(text.to_string()),
            timestamp: ts,
        })
    }

    #[tokio::test]
    async fn test_session_hooks_filter_delegates_inner_then_excludes() {
        // inner receives the full set (delegated); exclusion-set hits are removed after `!!` bash User timestamps
        let seen = Arc::new(Mutex::new(Vec::new()));
        let inner: Arc<dyn AgentLoopHooks> = Arc::new(RecordingHooks { seen: seen.clone() });
        let excluded = Arc::new(Mutex::new(HashSet::from(["777".to_string()])));
        let hooks = SessionHooks { inner, excluded };

        let msgs = vec![user_msg("secret", 777), user_msg("keep", 1)];
        let result = hooks.filter_messages(msgs.clone()).await;

        assert_eq!(seen.lock().unwrap()[0], msgs, "inner delegation receives full set");
        assert_eq!(result, vec![user_msg("keep", 1)]);
    }

    #[tokio::test]
    async fn test_session_hooks_filter_no_match_keeps_all() {
        let inner: Arc<dyn AgentLoopHooks> = Arc::new(RecordingHooks {
            seen: Arc::new(Mutex::new(Vec::new())),
        });
        let excluded = Arc::new(Mutex::new(HashSet::from(["888".to_string()])));
        let hooks = SessionHooks { inner, excluded };

        let msgs = vec![user_msg("a", 1), user_msg("b", 2)];
        assert_eq!(hooks.filter_messages(msgs.clone()).await, msgs);
    }

    #[tokio::test]
    async fn test_session_hooks_empty_exclusion_passthrough() {
        let inner: Arc<dyn AgentLoopHooks> = Arc::new(RecordingHooks {
            seen: Arc::new(Mutex::new(Vec::new())),
        });
        let excluded = Arc::new(Mutex::new(HashSet::new()));
        let hooks = SessionHooks { inner, excluded };

        let tool_result_msg = Message::ToolResult(zen_ai::types::ToolResultMessage {
            tool_call_id: "call-1".into(),
            tool_name: "bash".into(),
            content: vec![],
            details: None,
            usage: None,
            is_error: false,
            added_tool_names: None,
            timestamp: 0,
        });
        let msgs = vec![tool_result_msg.clone()];
        // empty exclusion set always passes through (filter_excluded guarantee)
        assert_eq!(hooks.filter_messages(msgs.clone()).await, msgs);
    }
}
