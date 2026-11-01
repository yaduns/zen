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

use crate::session::assert_session_cwd_exists;
use anyhow::Context;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::cli::{parse_thinking_level, CliArgs};
use crate::config::defaults::DEFAULT_THINKING_LEVEL;
use crate::config::get_agent_dir;
use crate::config::settings::{SettingsData, SettingsManager};
use crate::prompt_templates::LoadedPromptTemplate;
use crate::resources::{load_resources, LoadResourcesOptions, LoadResourcesResult};
use crate::session::agent_session::{AgentSession, SessionStore};
use crate::session::assembly_options::SessionAssemblyOptions;
use crate::session::provider_attribution::with_provider_attribution;
use crate::session::runtime::{
    create_initial_runtime, AgentSessionRuntime, AgentSessionServices, SessionStartReason,
};
use crate::session::session_manager::SessionManager;
use crate::skills::LoadedSkill;
use tokio_util::sync::CancellationToken;
use zen_ai::api::stream_for;
use zen_ai::catalog::Catalog;
use zen_ai::provider::resolve_provider;
use zen_ai::tool::{Tool, ToolContext, ToolExecutionMode};
use zen_ai::types::{Message, Model, ThinkingLevel};
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

    /// Real `get_api_key` hook (T-AI-02-11): `inner` takes precedence (allows caller overrides); if no match, resolve by provider
    /// name from the current catalog [`Catalog::current`] to resolve `apiKey` (literal / `$ENV`). The result is
    /// `agent_loop::resolve_api_key` writes to `StreamConfig.api_key`, which the request layer uses first.
    async fn get_api_key(&self, provider: &str) -> Option<String> {
        match self.inner.get_api_key(provider).await {
            Some(key) => Some(key),
            None => Catalog::current().api_key_for_provider(provider),
        }
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
    settings: Arc<SettingsManager>,
    skills: Vec<LoadedSkill>,
    templates: Vec<LoadedPromptTemplate>,
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
    // T-AI-02-6: store constructed before agent; session_id readable before Agent::new → threaded through the whole chain
    // (Agent.session_id → AgentLoopConfig → StreamConfig → request body prompt_cache_key)
    let session_id = session_store.session_id();
    // T-AI-02-10: provider attribution headers (e.g. opencode `x-opencode-session`) are injected at the assembly layer
    // (the request layer no longer contains resolve_provider; see `session::provider_attribution`).
    let stream_fn = with_provider_attribution(stream_fn);
    let agent = Agent::new(
        state,
        ToolExecutionMode::Parallel,
        session_hooks,
        stream_fn,
        Some(session_id),
    );
    AgentSession::new_with_exclusions(
        agent,
        session_store,
        excluded,
        shell_config,
        settings,
        skills,
        templates,
    )
}

/// Thin CLI-specific glue: snapshot the parsed CLI, then assemble the runtime through the shared assembler.
///
/// This is main's core call point — main only does parse → (select session) → call this function → dispatch.
///
/// # Parameters
/// - `args`: parsed CLI arguments (consumed **only here**: projected into `SessionAssemblyOptions`)
/// - `cwd`: session working directory (obtained by the caller, for test injection)
/// - `session_manager`: **session manager selected by the caller (main)**. Session selection (`in_memory`/`create`/`open`/
///   `continue_recent`/`fork_from`) is the caller's responsibility; this function only "assembles a runnable session from the given manager".
///
/// # Behavior
/// 1. `assert_session_cwd_exists` (fallback = process cwd) — assembly fuse, same position as pi
/// 2. `get_agent_dir` **once**: the same value feeds `SettingsManager::load` and is passed down to
///    `assemble_runtime` (the assembler never re-reads it, so both see one consistent value)
/// 3. `SessionAssemblyOptions::from_cli_args(args)`: remember this launch's overrides on the runtime so later
///    session switches replay them (a read-only session must stay read-only)
/// 4. Delegate to `assemble_runtime` — the single assembly path shared with interactive session switching
/// 5. `warn_model_configuration` on the assembled model: it must stay on the startup path, because the warning
///    is `eprintln!` and printing during a switch would write into raw-mode TUI output
///
/// # Return Value
/// The fully assembled `AgentSessionRuntime` (session + services + session store + launch-time overrides)
///
/// # Edge Cases
/// - Resource loading (skills/prompt templates/themes/context/system prompt discovery, unified loading in T-AP-05-3) happens
///   inside `assemble_runtime` via `load_assembly_resources`; `load_resources` converges all five resource paths in one call
///   (see tasks/T-AP-05-3-implemented.md for the seam backfill)
/// - `--session` only supports full paths (partial UUID resolution is T-AP-08-5)
///
/// # Consumers
/// T-AP-08-3 (main's node 4 call point, session selected by main)
pub async fn create_initial_runtime_from_cli(
    args: &CliArgs,
    cwd: PathBuf,
    session_manager: SessionManager,
) -> anyhow::Result<AgentSessionRuntime> {
    // T-AP-01-5: assembly fuse (same-line defensive check in pi createAgentSessionRuntime)
    let process_cwd =
        std::env::current_dir().context("failed to resolve process working directory")?;
    assert_session_cwd_exists(&session_manager, &process_cwd.to_string_lossy())?;

    // Agent config directory: resolved once here, shared by settings loading and the assembler (T-AP-01-6).
    let agent_dir = get_agent_dir()?;

    // Settings assembly (T-AP-02-3 wrap-up): load the two-tier settings, provide default_model/default_thinking_level
    // fill-in (priority: CLI > session restore > settings > built-in defaults) and shell override config.
    // A corrupt file degrades to empty + is recorded in the manager error queue (not surfaced at this assembly point; the interactive side's drain_errors handles that).
    let settings = Arc::new(SettingsManager::load(&cwd, &agent_dir));

    // Remember this launch's command-line overrides (T-AP-01-6): the runtime replays them on every later switch,
    // so `zen --tools read,...` stays read-only after `/new` / `/resume`.
    let options = SessionAssemblyOptions::from_cli_args(args);

    let runtime = assemble_runtime(
        cwd,
        agent_dir,
        session_manager,
        settings,
        &options,
        SessionStartReason::Startup,
    )
    .await?;

    // Diagnose a model that the catalog cannot serve (unknown id / missing key) before the first request,
    // instead of letting it surface as an opaque API error after the user typed a prompt. Kept on the startup
    // path only: the warning is `eprintln!`, which would corrupt raw-mode TUI output if emitted during a switch.
    warn_model_configuration(&runtime.session.model());
    Ok(runtime)
}

/// Interactive session-switch assembly (rebuild the runtime after `/resume`/`/new`/`/fork`/`/clone`).
///
/// The target session inherits the **launch-time command-line overrides** carried by the current runtime
/// (`assembly`): tool selection, resource overrides and model/thinking overrides are replayed instead of falling
/// back to the default built-in tool group — the user-visible fix for "a read-only session turns writable after
/// one session switch".
///
/// # Parameters
/// - `cwd`: target session working directory
/// - `agent_dir`: global agent dir, resolved by the caller once (shared with `SettingsManager` and
///   `AgentSessionServices`, avoiding a second env read)
/// - `session_manager`: target session (already opened/created/forked by the caller)
/// - `settings`: reuses the current session's settings manager (not reloaded — documented deviation: pi reloads
///   settings per target cwd; see tasks/T-AP-01-6-implemented.md)
/// - `assembly`: command-line override snapshot copied from the current runtime (`self.runtime.assembly.clone()`)
/// - `start_reason`: `/resume` → `Resume(path)`, `/new` → `New`, `/fork`/`/clone` → `Fork(<forked session file path>)`
///
/// # Return Value
/// A runtime ready for a direct swap.
///
/// # Consumers
/// `modes::interactive` `switch_runtime` (single exit for the five switch commands).
pub(crate) async fn assemble_runtime_from_session_manager(
    cwd: &Path,
    agent_dir: PathBuf,
    session_manager: SessionManager,
    settings: Arc<SettingsManager>,
    assembly: SessionAssemblyOptions,
    start_reason: SessionStartReason,
) -> anyhow::Result<AgentSessionRuntime> {
    assemble_runtime(
        cwd.to_path_buf(),
        agent_dir,
        session_manager,
        settings,
        &assembly,
        start_reason,
    )
    .await
}

/// The single assembly core shared by the startup entry and the session-switch entry:
/// model/thinking → resources → tools → system prompt → protocol stream → runtime construction.
///
/// Having exactly one implementation (fed by the same `options`) is what makes "the launch-time command-line
/// settings hold for every session in this process" true.
///
/// # Parameters
/// - `cwd`: target session working directory (settings scope, skill/AGENTS.md discovery, tool base dir)
/// - `agent_dir`: global agent dir, resolved by the caller — this function never calls `get_agent_dir`, so the
///   settings load and `AgentSessionServices` always see the same value
/// - `session_manager`: the session to assemble
/// - `settings`: settings manager (reused as-is on switches)
/// - `options`: command-line override snapshot (built at startup, copied on switches)
/// - `start_reason`: Startup/Resume/New/Fork
///
/// # Behavior
/// 1. `build_context()` is called **once** — the same `ctx` feeds model/thinking resolution and the final
///    `set_messages` (no duplicate context construction)
/// 2. model/thinking via `resolve_model_and_thinking`; resources via `load_assembly_resources`
/// 3. tools via `options.tool_selection` (never a hard-coded default group)
///
/// # Errors
/// Assembly errors (agent dir resolution is already done by the caller; missing protocol binding, etc.) propagate
/// through `anyhow`.
async fn assemble_runtime(
    cwd: PathBuf,
    agent_dir: PathBuf,
    session_manager: SessionManager,
    settings: Arc<SettingsManager>,
    options: &SessionAssemblyOptions,
    start_reason: SessionStartReason,
) -> anyhow::Result<AgentSessionRuntime> {
    // Session ID (provided by the manager chosen by the caller; session_id of the persisted/forked session)
    let cwd_str = cwd.to_string_lossy().into_owned();
    let session_id = session_manager.session_id();
    let merged = settings.merged();

    // 0. Session context: built once (history for `set_messages` + model/thinking restore inputs)
    let ctx = session_manager.build_context();

    // 1. Model / thinking: CLI override > session restore > settings > built-in default
    let (model, thinking_level) = resolve_model_and_thinking(options, &ctx, &merged);

    // 2. Resources: CLI overrides merged with settings (same loader for startup and switches)
    let resources = load_assembly_resources(&cwd, &agent_dir, &merged, options).await;

    // 3. Tool factory: binds cwd + session_id; the selection comes from the launch-time snapshot
    //    (cli-tools-flags-fix — default builtin group [read, bash, edit, write]; `--tools read` yields a read-only session)
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
    let tools = factory.create_with_selection(&options.tool_selection).tools;

    // 4. System prompt: AGENTS.md context + CLI overrides + skills (assembled and injected per T-AP-03-2)
    let tool_refs: Vec<&dyn Tool> = tools.iter().map(|t| t.as_ref()).collect();
    let meta = collect_tool_prompt_metadata(&tool_refs);
    let system_prompt = build_system_prompt(&BuildSystemPromptOptions {
        custom_prompt: resources.system_prompt,
        selected_tools: None,
        tool_snippets: meta.tool_snippets,
        prompt_guidelines: meta.prompt_guidelines,
        append_system_prompt: if resources.append_system_prompt.is_empty() {
            None
        } else {
            // Aligns with pi agent-session.ts `loaderAppendSystemPrompt.join("\n\n")`
            Some(resources.append_system_prompt.join("\n\n"))
        },
        cwd: cwd_str,
        // T-AP-05-4: session side keeps the Loaded form (source end-to-end); the system prompt only needs domain data → clone once at assembly phase
        skills: resources
            .skills
            .iter()
            .map(|loaded| loaded.skill.clone())
            .collect(),
        context_files: resources.context_files,
    });

    // 5. Protocol stream implementation + common assembly (stream_for binding table from T-AI-02-8)
    let model_api = model.api;
    let stream_fn = stream_for(model_api).expect(
        "Protocol referenced by model.api not implemented: currently supports OpenAiCompletions, see T-AI-02-8 stream_for binding table",
    );

    // Diagnostic logging (topic-loading artifacts currently have no consumers; log for diagnostics only)
    for d in &resources.diagnostics {
        tracing::warn!(target: "zen::resources", "resource load: {d:?}");
    }

    // 6. Session construction (ctx.messages is empty for new sessions) — `ctx` built once above is reused here
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
        Arc::clone(&settings),
        resources.skills,
        resources.prompts,
    );
    session.set_messages(ctx.messages);

    // 7. Runtime assembly (the launch-time overrides ride along for later switches)
    let services = AgentSessionServices {
        cwd,
        agent_dir,
        session_store,
        registry: zen_core::process::ProcessRegistry::new(),
    };
    Ok(create_initial_runtime(
        session,
        services,
        start_reason,
        resources.diagnostics,
        options.clone(),
    ))
}

/// Resolve a provider name declared in `settings.json` (`defaultProvider`), warning when the name is
/// absent from the current catalog.
///
/// A stale `defaultProvider` used to be ignored **silently**, which made "I edited models.json but zen
/// still talks to the old provider" impossible to diagnose. The warning names the catalog providers so the
/// fix (rename the entry, or update settings) is obvious.
///
/// # Returns
/// `Some(provider)` when the name is known; `None` (after warning on stderr) otherwise, so callers keep the
/// previously resolved model.
fn resolve_settings_provider(name: &str) -> Option<Box<dyn zen_ai::provider::Provider>> {
    if let Some(provider) = resolve_provider(name) {
        return Some(provider);
    }
    let mut known: Vec<String> = Catalog::current()
        .models()
        .iter()
        .map(|m| m.provider.clone())
        .collect();
    known.sort();
    known.dedup();
    eprintln!(
        "Warning: settings.json defaultProvider '{name}' is not in the model catalog (known: {}); ignoring it",
        if known.is_empty() {
            "none".to_string()
        } else {
            known.join(", ")
        }
    );
    None
}

/// Configuration problems that will make the first request fail, as human-readable warnings.
///
/// Pure function (no side effects) so the diagnostics are unit-testable; the caller prints them.
/// Two failure modes are covered, both of which are otherwise only visible as an opaque API error:
///
/// 1. **Model not in the catalog** — `settings.json` `defaultModel` (or `--model`) names an id the provider
///    does not declare in `models.json`; the request is sent with the unknown id and the endpoint rejects it.
/// 2. **No API key** — the provider's `apiKey` (literal or `$ENV`) does not resolve; v1 always sends an
///    `Authorization: Bearer` header, so keyless local endpoints must configure a placeholder key.
///
/// # Returns
/// Empty when the model is fully resolvable; otherwise one message per problem.
pub(crate) fn model_config_warnings(model: &Model) -> Vec<String> {
    let catalog = Catalog::current();
    let mut warnings = Vec::new();

    let in_catalog = catalog
        .models()
        .iter()
        .any(|m| m.provider == model.provider && m.id == model.id);
    if !in_catalog {
        warnings.push(format!(
            "model '{}/{}' is not declared in models.json; the endpoint will reject it (add it to the provider's \"models\" list, or fix settings.json defaultModel)",
            model.provider, model.id
        ));
    }

    if catalog.api_key(model).is_none() {
        warnings.push(format!(
            "no API key resolved for provider '{}' (set the environment variable referenced by models.json apiKey; local endpoints can use any placeholder value)",
            model.provider
        ));
    }

    warnings
}

/// Print [`model_config_warnings`] for a runtime-resolved model (one line per problem, on stderr).
///
/// Called once per process at assembly time: the model is not yet used, so the user gets the diagnosis
/// before the first turn rather than as an API error afterwards.
fn warn_model_configuration(model: &Model) {
    for warning in model_config_warnings(model) {
        eprintln!("Warning: {warning}");
    }
}

/// Resolve which model / thinking level a session should use:
/// CLI override > session restore > settings > built-in default.
///
/// Both the startup entry and the session-switch entry go through this function, which is what keeps `--model`
/// in effect after `/new` (previously the switch path ignored the CLI and only consulted session + settings).
///
/// # Parameters
/// - `options`: launch-time command-line overrides
/// - `ctx`: target session context (`ctx.model`/`ctx.thinking_level` are the session-restore inputs)
/// - `merged`: effective settings view (`default_provider`/`default_model`/`default_thinking_level` fill-in)
///
/// # Behavior (boundary semantics are pinned — do not turn these into "settings fill-in")
/// - model:
///   1. `options.cli_model` is `Some` → use it;
///   2. else `ctx.model` is `Some`: a resolvable provider rebuilds provider+id, an **unresolvable** provider keeps
///      `Model::from_env()` (old-chain behavior, no new warning). **Neither case consults settings**
///      (matches the old `ctx.model.is_none()` guard);
///   3. else (`ctx.model` is `None`) settings: `default_provider` and `default_model` are handled **independently** —
///      a known `default_provider` rebuilds the model by provider, and **`default_model` still overrides
///      `model.id` even when `default_provider` is unknown** (warned and ignored by `resolve_settings_provider`);
///      otherwise `Model::from_env()`.
/// - thinking:
///   1. `options.cli_thinking_level` is `Some` → use it (this includes the "unparsable `--model id:bogus` suffix"
///      case, which is stored as `Some(DEFAULT_THINKING_LEVEL)` so the fallback chain stays blocked);
///   2. else a **non-empty** `ctx.thinking_level`: parsable → use it, unparsable → `DEFAULT_THINKING_LEVEL`
///      (old-chain behavior). **Non-empty never consults settings** (matches the old
///      `ctx.thinking_level.trim().is_empty()` guard);
///   3. else (empty string) settings `default_thinking_level` → otherwise `DEFAULT_THINKING_LEVEL`.
///
/// # Boundaries
/// Pure function, no IO; `--api-key` is not handled here (that field has no consumer in zen yet).
/// "ctx value present but unparsable → no settings fill-in" is deliberately aligned with the two old chains; if a
/// future change wants fill-in instead, it must be registered as an intentional divergence, not slipped in here.
fn resolve_model_and_thinking(
    options: &SessionAssemblyOptions,
    ctx: &SessionContext,
    merged: &SettingsData,
) -> (Model, ThinkingLevel) {
    let model = match &options.cli_model {
        Some(cli_model) => cli_model.clone(),
        None => match &ctx.model {
            Some(SessionModel { provider, model_id }) => match resolve_provider(provider) {
                Some(p) => {
                    let mut restored = Model::from_provider(p.as_ref());
                    restored.id = model_id.clone();
                    restored
                }
                None => Model::from_env(),
            },
            None => {
                let mut model = Model::from_env();
                if let Some(provider_name) = merged.default_provider.as_deref() {
                    if let Some(provider) = resolve_settings_provider(provider_name) {
                        model = Model::from_provider(provider.as_ref());
                    }
                }
                // Independent of the provider branch: an unknown default_provider must not swallow default_model.
                if let Some(default_model) = &merged.default_model {
                    model.id = default_model.clone();
                }
                model
            }
        },
    };

    let thinking = match options.cli_thinking_level {
        Some(level) => level,
        None => {
            let session_thinking = ctx.thinking_level.trim();
            if session_thinking.is_empty() {
                merged
                    .default_thinking_level
                    .unwrap_or(DEFAULT_THINKING_LEVEL)
            } else {
                parse_thinking_level(session_thinking).unwrap_or(DEFAULT_THINKING_LEVEL)
            }
        }
    };

    (model, thinking)
}

/// Load skills / prompt templates / themes / system prompt / project context for one assembly: the single place
/// that merges the launch-time CLI overrides with settings into `LoadResourcesOptions`.
///
/// Shared by the startup entry and the session-switch entry, so a switch cannot silently drop `--system-prompt` or
/// `--no-skills` (the same class of bug as the tool-selection deviation).
///
/// # Parameters
/// - `cwd` / `agent_dir`: target session working directory and global agent dir (they follow the target session on
///   a switch)
/// - `merged`: effective settings view (`skills`/`prompts`/`themes` are merged first, CLI values appended after —
///   the existing merge order)
/// - `options`: command-line override snapshot
///
/// # Boundaries
/// Load failures never abort (the existing "warnings do not interrupt" contract lives in `load_resources`;
/// diagnostics are returned inside `LoadResourcesResult.diagnostics`).
async fn load_assembly_resources(
    cwd: &Path,
    agent_dir: &Path,
    merged: &SettingsData,
    options: &SessionAssemblyOptions,
) -> LoadResourcesResult {
    // Resource loading (unified loading taken over by T-AP-05-3): default directories + CLI
    // `--skill`/`--prompt-template`/`--theme` + settings.skills/prompts/themes merged; SYSTEM.md/APPEND_SYSTEM.md
    // discovery is inlined via load_resources, CLI values take precedence. `no_*` flags toggle each path
    // individually (aligned with pi resource-loader branches); loading diagnostics are non-fatal.
    let skill_paths: Vec<PathBuf> = merged
        .skills
        .clone()
        .unwrap_or_default()
        .into_iter()
        .chain(options.skills.iter().cloned())
        .map(PathBuf::from)
        .collect();
    let prompt_paths: Vec<PathBuf> = merged
        .prompts
        .clone()
        .unwrap_or_default()
        .into_iter()
        .chain(options.prompt_templates.iter().cloned())
        .map(PathBuf::from)
        .collect();
    let theme_paths: Vec<PathBuf> = merged
        .themes
        .clone()
        .unwrap_or_default()
        .into_iter()
        .chain(options.themes.iter().cloned())
        .map(PathBuf::from)
        .collect();
    load_resources(&LoadResourcesOptions {
        cwd: cwd.to_path_buf(),
        agent_dir: agent_dir.to_path_buf(),
        skill_paths,
        prompt_paths,
        theme_paths,
        no_skills: options.no_skills,
        no_prompt_templates: options.no_prompt_templates,
        no_themes: options.no_themes,
        no_context_files: options.no_context_files,
        system_prompt: options.system_prompt.clone(),
        append_system_prompt: options.append_system_prompt.clone(),
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
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

        /// Test helper: temporarily set an environment variable (restore the original value when leaving scope).
        fn set(key: &'static str, value: &str) -> Self {
            let old = std::env::var_os(key);
            std::env::set_var(key, value);
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

    /// Test helper: session context carrying model/thinking records (builds a zen-core SessionContext directly, avoiding file IO).
    fn restored_ctx(provider: &str, model_id: &str, thinking: &str) -> SessionContext {
        SessionContext {
            messages: vec![],
            thinking_level: thinking.into(),
            model: Some(zen_core::session::tree::SessionModel {
                provider: provider.into(),
                model_id: model_id.into(),
            }),
            active_tool_names: None,
        }
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

    // ── resolve_model_and_thinking: settings fill-in (was apply_settings_defaults) ─────

    /// Test helper: resolve model/thinking for a session given CLI overrides + session ctx + settings.
    fn resolve_with(
        args: &[&str],
        ctx: &SessionContext,
        merged: &SettingsData,
    ) -> (Model, ThinkingLevel) {
        let options = SessionAssemblyOptions::from_cli_args(&parse(args));
        resolve_model_and_thinking(&options, ctx, merged)
    }

    #[test]
    fn settings_default_thinking_applies_on_fresh_session() {
        // Regression (fix verified by T-AP-02-3-fix feedback): new session ctx.thinking_level is an empty string (no record),
        // settings.defaultThinkingLevel must take effect as the fallback — previously build_session_context faked an empty session
        // as "off", causing this gate to wrongly skip; see fix in zen-core/session/context.rs.
        let ctx = empty_ctx(); // thinking_level = ""
        let merged = SettingsData {
            default_thinking_level: Some(ThinkingLevel::Medium),
            ..Default::default()
        };
        let (_, thinking) = resolve_with(&[], &ctx, &merged);
        assert_eq!(thinking, ThinkingLevel::Medium);
    }

    #[test]
    fn settings_default_provider_rebuilds_model_when_unset() {
        // Env isolation: assert on the default provider when "ZEN_PROVIDER is unset" (opencode-go); explicitly remove
        // the variable so the test isn't affected by the shell environment (e.g. ZEN_PROVIDER=huoshan) (provider.rs:86 reads env).
        let _l = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::remove("ZEN_PROVIDER");
        let ctx = empty_ctx();
        let merged = SettingsData {
            default_provider: Some("huoshan".into()),
            ..Default::default()
        };
        let (model, _) = resolve_with(&[], &ctx, &merged);
        assert_eq!(model.provider, "huoshan");
        assert_eq!(model.api, zen_ai::provider::Api::OpenAiCompletions);
    }

    #[test]
    fn settings_default_provider_ignored_when_cli_provider_set() {
        let ctx = empty_ctx();
        let merged = SettingsData {
            default_provider: Some("opencode-go".into()),
            ..Default::default()
        };
        let (model, _) = resolve_with(&["--provider", "huoshan"], &ctx, &merged);
        // CLI --provider takes precedence; settings do not override
        assert_eq!(model.provider, "huoshan");
    }

    #[test]
    fn settings_default_model_applies_id_within_settings_provider() {
        let ctx = empty_ctx();
        let merged = SettingsData {
            default_provider: Some("huoshan".into()),
            default_model: Some("ark-xyz".into()),
            ..Default::default()
        };
        let (model, _) = resolve_with(&[], &ctx, &merged);
        assert_eq!(model.provider, "huoshan");
        assert_eq!(model.id, "ark-xyz");
    }

    #[test]
    fn settings_ignored_when_session_restored() {
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
        let (model, thinking) = resolve_with(&[], &ctx, &merged);
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
        let ctx = empty_ctx();
        let merged = SettingsData {
            default_provider: Some("not-a-provider".into()),
            default_model: Some("m".into()),
            ..Default::default()
        };
        let (model, _) = resolve_with(&[], &ctx, &merged);
        // The two settings branches are independent: the unknown provider is ignored (fall back to the env-selected
        // provider) while default_model still overrides the id.
        assert_eq!(model.provider, "opencode-go");
        assert_eq!(model.id, "m");
    }

    // ── resolve_model_and_thinking: session restore (was apply_session_restore) ─────

    #[test]
    fn restore_applies_when_cli_has_no_model_flags() {
        let ctx = restored_ctx("huoshan", "ark-restored-model", "high");
        let (model, thinking) = resolve_with(&[], &ctx, &SettingsData::default());
        assert_eq!(model.provider, "huoshan");
        assert_eq!(model.id, "ark-restored-model");
        assert_eq!(thinking, ThinkingLevel::High);
    }

    #[test]
    fn restore_skipped_when_cli_provides_model() {
        // --model explicit → the session model is not restored (thinking still is, since the shorthand has no colon)
        let _l = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::remove("ZEN_PROVIDER");
        let baseline = Model::from_env();
        let ctx = restored_ctx("huoshan", "ark-restored-model", "high");
        let (model, thinking) =
            resolve_with(&["--model", "my-model"], &ctx, &SettingsData::default());
        assert_eq!(model.id, "my-model", "CLI --model takes precedence");
        assert_eq!(model.provider, baseline.provider, "provider not overridden");
        assert_eq!(thinking, ThinkingLevel::High);
    }

    #[test]
    fn restore_skipped_when_cli_provides_thinking() {
        let ctx = restored_ctx("huoshan", "ark", "high");
        let (model, thinking) =
            resolve_with(&["--thinking", "low"], &ctx, &SettingsData::default());
        assert_eq!(
            thinking,
            ThinkingLevel::Low,
            "CLI --thinking takes precedence"
        );
        assert_eq!(model.provider, "huoshan", "model not provided → restore");
    }

    #[test]
    fn restore_noop_on_empty_session_context() {
        let (model, thinking) = resolve_with(&[], &empty_ctx(), &SettingsData::default());
        // New session: no session model/thinking to restore, keep the built-in defaults
        assert!(!model.id.is_empty());
        assert_eq!(thinking, DEFAULT_THINKING_LEVEL);
    }

    #[test]
    fn restore_ctx_model_with_unknown_provider_keeps_env_default_without_settings_fill_in() {
        // Boundary pinned by T-AP-01-6 review: ctx.model present but unparsable must NOT fall through to settings
        // (the old chain gated settings on `ctx.model.is_none()`).
        let _l = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::remove("ZEN_PROVIDER");
        let ctx = SessionContext {
            model: Some(SessionModel {
                provider: "no-such-provider".into(),
                model_id: "whatever".into(),
            }),
            thinking_level: "bogus-level".into(),
            ..empty_ctx()
        };
        let merged = SettingsData {
            default_provider: Some("huoshan".into()),
            default_model: Some("ark-xyz".into()),
            default_thinking_level: Some(ThinkingLevel::Low),
            ..Default::default()
        };
        let (model, thinking) = resolve_with(&[], &ctx, &merged);
        assert_eq!(
            model.provider, "opencode-go",
            "env default, no settings fill-in"
        );
        assert_ne!(
            model.id, "ark-xyz",
            "settings default_model must not fill in"
        );
        assert_eq!(
            thinking, DEFAULT_THINKING_LEVEL,
            "unparsable non-empty session thinking → default, no settings fill-in"
        );
    }

    // ── resolve_model_and_thinking: CLI overrides (was resolve_model) ─────

    #[test]
    fn model_flag_overrides_id() {
        let ctx = empty_ctx();
        let (model, thinking) =
            resolve_with(&["--model", "my-model"], &ctx, &SettingsData::default());
        assert_eq!(model.id, "my-model");
        assert_eq!(
            thinking, DEFAULT_THINKING_LEVEL,
            "No thinking specified → requirement decision defaults to High"
        );
    }

    #[test]
    fn model_provider_slash_id_sets_provider_and_id() {
        let (model, thinking) = resolve_with(
            &["--model", "huoshan/ark-model-x"],
            &empty_ctx(),
            &SettingsData::default(),
        );
        assert_eq!(model.provider, "huoshan");
        assert_eq!(model.id, "ark-model-x");
        assert_eq!(thinking, DEFAULT_THINKING_LEVEL);
    }

    #[test]
    fn model_thinking_suffix_sets_thinking() {
        let (_, thinking) = resolve_with(
            &["--model", "my-model:high"],
            &empty_ctx(),
            &SettingsData::default(),
        );
        assert_eq!(thinking, ThinkingLevel::High);
    }

    #[test]
    fn thinking_flag_overrides_model_suffix() {
        let (_, thinking) = resolve_with(
            &["--model", "my-model:high", "--thinking", "off"],
            &empty_ctx(),
            &SettingsData::default(),
        );
        assert_eq!(thinking, ThinkingLevel::Off);
    }

    #[test]
    fn invalid_thinking_shorthand_blocks_session_restore() {
        // Regression (T-AP-01-6 review): `--model id:bogus` used to end at DEFAULT_THINKING_LEVEL because the old
        // chain gated on `args.model.contains(':')`; storing `None` here would let the session value win again.
        let ctx = restored_ctx("huoshan", "ark", "high");
        let (_, thinking) = resolve_with(
            &["--model", "id:bogus"],
            &ctx,
            &SettingsData {
                default_thinking_level: Some(ThinkingLevel::Low),
                ..Default::default()
            },
        );
        assert_eq!(thinking, DEFAULT_THINKING_LEVEL);
    }

    #[test]
    fn provider_flag_overrides_default_provider() {
        let (model, _) = resolve_with(
            &["--provider", "huoshan"],
            &empty_ctx(),
            &SettingsData::default(),
        );
        assert_eq!(model.provider, "huoshan");
    }

    #[test]
    fn unknown_provider_warns_and_falls_back() {
        let (model, _) = resolve_with(
            &["--provider", "nonexistent"],
            &empty_ctx(),
            &SettingsData::default(),
        );
        // Fall back to default provider (no panic)
        assert!(!model.provider.is_empty());
    }

    #[test]
    fn default_model_from_env() {
        let (model, thinking) = resolve_with(&[], &empty_ctx(), &SettingsData::default());
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
        assert!(runtime.session.session_id().len() >= 36);
    }

    /// T-AP-01-5 wiring test (mirror of the future T-AP-08-5 resume capture branch, now locked down against regressions):
    /// Session storage cwd points to a non-existent directory → create persists to disk → open reopens (header cwd takes effect) →
    /// call the factory with the session cwd → first-line assertion hits → `downcast_ref::<MissingSessionCwdError>()` on the anyhow chain
    /// yields a snapshot, with `issue.fallback_cwd` set to the process cwd (not the ghost directory).
    #[tokio::test]
    async fn create_runtime_asserts_missing_session_cwd() {
        // unique temporary root directory (std-only; open's root param is the same root as create)
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "zen-session-cwd-wiring-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let ghost = root.join("deleted-project");

        let opts = crate::session::jsonl::CreateSessionOptions {
            session_id: None,
            parent_session_path: None,
            metadata: None,
        };
        let created = SessionManager::create(ghost.to_str().unwrap(), &root, opts).unwrap();
        let session_file = created
            .get_session_file()
            .expect("Persisted session must have session_file");

        // reopen with open: no cwd_override passed → header cwd takes effect (T-AP-01-5 integration semantics 1)
        let manager = SessionManager::open(&session_file, None, &root).unwrap();
        let session_cwd = PathBuf::from(manager.get_cwd());

        let args = parse(&[]);
        // AgentSessionRuntime does not derive Debug, so expect_err is unavailable; use match to get the Err
        let err = match create_initial_runtime_from_cli(&args, session_cwd, manager).await {
            Ok(_) => panic!("Session storage cwd missing → assembly must abort"),
            Err(e) => e,
        };

        let missing = err
            .downcast_ref::<crate::session::MissingSessionCwdError>()
            .expect("MissingSessionCwdError should be catchable by type on anyhow chain");
        let process_cwd = std::env::current_dir()
            .expect("Test process cwd should be resolvable")
            .to_string_lossy()
            .to_string();
        assert_eq!(
            missing.issue.fallback_cwd, process_cwd,
            "fallback should be process cwd"
        );
        assert_ne!(
            missing.issue.fallback_cwd,
            ghost.to_string_lossy(),
            "fallback must not write to ghost directory (otherwise reopening the session falls back into a missing-directory infinite loop)"
        );
        assert_eq!(missing.issue.session_file, session_file);
        assert_eq!(missing.issue.session_cwd, ghost.to_string_lossy());

        let _ = std::fs::remove_dir_all(&root);
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

    // ── Session-switch assembly keeps the launch-time CLI overrides (T-AP-01-6) ─────

    /// Assemble through the *session-switch* path (the `/new`·`/resume`·`/fork`·`/clone` exit) with the
    /// launch-time snapshot built from `args`, and return the assembled runtime.
    async fn switch_session(args: &[&str], cwd: &Path, agent_dir: &Path) -> AgentSessionRuntime {
        let parsed = parse(args);
        let assembly = SessionAssemblyOptions::from_cli_args(&parsed);
        let manager = SessionManager::in_memory(&cwd.to_string_lossy());
        let runtime = assemble_runtime_from_session_manager(
            cwd,
            agent_dir.to_path_buf(),
            manager,
            Arc::new(SettingsManager::empty()),
            assembly,
            SessionStartReason::New,
        )
        .await
        .expect("switch-path assembly");
        runtime
    }

    /// User-accident regression: a read-only launch (`--tools read,grep,find,ls`) must stay read-only after a
    /// session switch — previously the switch path hard-coded `ToolSelection::default()` and the model regained
    /// `bash`/`edit`/`write`.
    #[tokio::test]
    async fn switch_path_keeps_cli_tool_allowlist() {
        let cwd = std::env::temp_dir();
        let agent_dir = get_agent_dir().expect("agent dir");
        let runtime = switch_session(&["--tools", "read,grep,find,ls"], &cwd, &agent_dir).await;
        assert_eq!(
            runtime.session.active_tool_names(),
            vec!["read", "grep", "find", "ls"],
            "switching sessions must replay the launch-time --tools allow-list"
        );
    }

    #[tokio::test]
    async fn switch_path_without_cli_overrides_uses_default_group() {
        let cwd = std::env::temp_dir();
        let agent_dir = get_agent_dir().expect("agent dir");
        let runtime = switch_session(&[], &cwd, &agent_dir).await;
        assert_eq!(
            runtime.session.active_tool_names(),
            vec!["read", "bash", "edit", "write"],
            "no launch-time override → default built-in group (pi parity)"
        );
    }

    #[tokio::test]
    async fn switch_path_keeps_cli_resource_overrides() {
        // Isolated cwd + agent_dir so the assertion cannot depend on the machine's ~/.zen skills.
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path().join("cwd");
        let agent_dir = tmp.path().join("agent");
        std::fs::create_dir_all(agent_dir.join("skills")).unwrap();
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::write(
            agent_dir.join("skills").join("SKILL.md"),
            "---\nname: known-skill\ndescription: fixture\n---\nbody",
        )
        .unwrap();

        // Control: default discovery finds the fixture skill.
        let runtime = switch_session(&[], &cwd, &agent_dir).await;
        assert_eq!(
            runtime.session.skills().len(),
            1,
            "control: the fixture skill is discovered by default"
        );

        // `--no-skills` must survive the switch (same class of deviation as --tools).
        let runtime = switch_session(&["--no-skills"], &cwd, &agent_dir).await;
        assert!(
            runtime.session.skills().is_empty(),
            "--no-skills must be replayed on the switch path"
        );
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

        assert_eq!(
            seen.lock().unwrap()[0],
            msgs,
            "inner delegation receives full set"
        );
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

    // ── SessionHooks::get_api_key (T-AI-02-11 real hook) ──

    /// Inner hooks with a fixed API key: only override `get_api_key`; the rest use defaults.
    struct ApiKeyHooks(Option<String>);

    #[async_trait::async_trait]
    impl AgentLoopHooks for ApiKeyHooks {
        async fn get_api_key(&self, _provider: &str) -> Option<String> {
            self.0.clone()
        }
    }

    fn api_key_hooks(inner: Arc<dyn AgentLoopHooks>) -> SessionHooks {
        SessionHooks {
            inner,
            excluded: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    #[tokio::test]
    async fn test_session_hooks_get_api_key_prefers_inner() {
        let inner: Arc<dyn AgentLoopHooks> = Arc::new(ApiKeyHooks(Some("inner-key".into())));
        let hooks = api_key_hooks(inner);
        // if inner hits, return immediately; do not query catalog
        assert_eq!(
            hooks.get_api_key("opencode-go").await,
            Some("inner-key".to_string())
        );
    }

    #[tokio::test]
    async fn test_session_hooks_get_api_key_falls_back_to_catalog() {
        let _l = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::set("OPENCODE_API_KEY", "env-key");
        let inner: Arc<dyn AgentLoopHooks> = Arc::new(ApiKeyHooks(None));
        let hooks = api_key_hooks(inner);
        // no inner → via Catalog::current().api_key_for_provider
        // (the built-in opencode-go apiKey template is `$OPENCODE_API_KEY`)
        assert_eq!(
            hooks.get_api_key("opencode-go").await,
            Some("env-key".to_string())
        );
    }

    #[tokio::test]
    async fn test_session_hooks_get_api_key_unknown_provider_none() {
        let inner: Arc<dyn AgentLoopHooks> = Arc::new(ApiKeyHooks(None));
        let hooks = api_key_hooks(inner);
        // provider not in catalog → None (no panic, no fabrication)
        assert_eq!(hooks.get_api_key("nonexistent").await, None);
    }

    // ── resolve_model_and_thinking: settings fallback when the session ctx is empty ─────

    #[test]
    fn settings_default_thinking_used_when_ctx_empty() {
        let merged = SettingsData {
            default_thinking_level: Some(ThinkingLevel::High),
            ..Default::default()
        };
        let (_, thinking) = resolve_with(&[], &empty_ctx(), &merged);
        assert_eq!(
            thinking,
            ThinkingLevel::High,
            "ctx no thinking → settings fallback"
        );
    }

    #[test]
    fn empty_ctx_falls_back_to_env_default() {
        // env isolation: default provider (opencode-go)
        let _l = ENV_LOCK.lock().unwrap();
        let _g = EnvVarGuard::remove("ZEN_PROVIDER");
        let (model, thinking) = resolve_with(&[], &empty_ctx(), &SettingsData::default());
        assert_eq!(
            model.provider, "opencode-go",
            "no resume/fallback → env default"
        );
        assert_eq!(thinking, DEFAULT_THINKING_LEVEL);
    }

    // ── model_config_warnings / resolve_settings_provider (settings-vs-models.json mismatch) ──

    #[test]
    fn model_config_warnings_flags_model_missing_from_catalog() {
        // settings.json defaultModel / --model naming an id the provider never declares in models.json:
        // the request would be sent with an unknown id, so the user must be warned before the first turn.
        let mut model = Model::from_env();
        model.id = "definitely-not-a-model".to_string();
        let warnings = model_config_warnings(&model);
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("is not declared in models.json")),
            "expected unknown-model warning, got {warnings:?}"
        );
    }

    #[test]
    fn model_config_warnings_flags_missing_api_key() {
        let _l = ENV_LOCK.lock().unwrap();
        // The embedded default provider resolves its key from $OPENCODE_API_KEY; clearing it (before the
        // model is materialised, since the catalog reads env on demand) must produce the key warning.
        let _g = EnvVarGuard::remove("OPENCODE_API_KEY");
        let _p = EnvVarGuard::remove("ZEN_PROVIDER");
        let model = Model::from_env();
        let warnings = model_config_warnings(&model);
        assert!(
            warnings.iter().any(|w| w.contains("no API key resolved")),
            "expected missing-key warning, got {warnings:?}"
        );
    }

    #[test]
    fn model_config_warnings_clean_for_catalog_model_with_key() {
        let _l = ENV_LOCK.lock().unwrap();
        let _k = EnvVarGuard::set("OPENCODE_API_KEY", "test-key");
        let _p = EnvVarGuard::remove("ZEN_PROVIDER");
        let model = Model::from_env();
        assert_eq!(
            model_config_warnings(&model),
            Vec::<String>::new(),
            "a catalog model with a resolvable key must not warn"
        );
    }

    #[test]
    fn resolve_settings_provider_returns_none_for_unknown_name() {
        // A stale settings.json defaultProvider must not abort startup: it is reported and ignored.
        let _l = ENV_LOCK.lock().unwrap();
        assert!(
            resolve_settings_provider("no-such-provider").is_none(),
            "unknown provider name must resolve to None (caller keeps the previous model)"
        );
    }

    #[test]
    fn resolve_settings_provider_returns_catalog_provider() {
        let _l = ENV_LOCK.lock().unwrap();
        let provider =
            resolve_settings_provider("opencode-go").expect("embedded provider resolves");
        assert_eq!(provider.id(), "opencode-go");
    }
}
