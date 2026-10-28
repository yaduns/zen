# zen Examples — CLI flag overrides and quick start

Example programs with real LLM calls. **No existing code is modified**; only public APIs are used.
A model endpoint must be configured before running: add a provider to `~/.zen/agent/models.json`
(any OpenAI-compatible endpoint) and export the variable its `apiKey` field references.

## Quick start

```sh
cargo run -p zen-app --example quickstart -- "introduce yourself in one sentence"
```

## Example list

### Tutorial template

| Example | Overridden flags | Description |
|---------|---------|------|
| `quickstart` | — | 3-step main assembly line tutorial (model→session→prompt), real LLM |

### Models and reasoning

| Example | Overridden flags | Description |
|---------|---------|------|
| `example_provider` | `--provider` | `Model::from_provider(resolve_provider(...))` overrides the default provider |
| `example_model` | `--model` | Overrides the model id (`--model` also supports `provider/id` and `id:thinking` shorthand, see quickstart comments) |
| `example_thinking` | `--thinking` | Sets the thinking level (off/minimal/low/medium/high/xhigh/max) |
| `example_list_models` | `--list-models` | Lists the available provider→model mappings (optionally filtered by a search string) |

### System prompt

| Example | Overridden flags | Description |
|---------|---------|------|
| `example_system_prompt` | `--system-prompt` | Replaces the default identity with a custom prompt |
| `example_append_system_prompt` | `--append-system-prompt` | Appends instructions to the end of the default prompt (repeatable) |
| `example_no_context_files` | `--no-context-files` | Compares with/without injection of project context such as AGENTS.md |
| `example_skills` | `--skills <dir>` | Loads a SKILL.md directory and injects it into the system prompt (`load_skills` + `build_system_prompt`) |

### Tools

| Example | Overridden flags | Description |
|---------|---------|------|
| `example_tools` | `--tools` / `--exclude-tools` / `--no-tools` / `--no-builtin-tools` | Tool subset selection (clap parses the real flags: allowlist/denylist/no tools/no builtin tools) |

### Sessions (persistence / resume / fork)

| Example | Overridden flags | Description |
|---------|---------|------|
| `example_no_session` | `--no-session` | Temporary in-memory session (`in_memory`, not written to disk) |
| `example_session_dir` | `--session-dir` | Specifies the session storage directory, creates a file session and writes it to disk |
| `example_fork` | `--fork` | Forks from the source session → `build_context` restores history → `set_messages` injects → LLM answers based on the forked history |
| `example_resume` | `--resume` / `--continue` | Resumes the most recent session → `build_context` restores history → `set_messages` injects → LLM answers based on the restored history |

### Output and behavior

| Example | Overridden flags | Description |
|---------|---------|------|
| `example_mode_json` | `--mode json` | Goes through the production print outlet `run_print_mode(Mode::Json)`, outputting events as one JSON per line |
| `example_version` | `--version` | Prints the version number |

### Shared boilerplate

| File | Description |
|------|------|
| `_support/common.rs` | Starter functions shared across examples: `make_session` / `run_and_print` / `run_with_store` / `default_model` / `tool_registry_tools`. Suppresses unused warnings with `#![allow(dead_code)]`. |

## Usage

```sh
# Basic (quickstart)
cargo run -p zen-app --example quickstart -- "your prompt"

# Specify provider + model
cargo run -p zen-app --example example_provider -- huoshan "hello huoshan"

# Tool subset
cargo run -p zen-app --example example_tools -- --tools read,bash "list files"   # allowlist
    cargo run -p zen-app --example example_tools -- --exclude-tools bash "list files" # denylist
    cargo run -p zen-app --example example_tools -- --no-tools "what is 1+1"          # no tools
    cargo run -p zen-app --example example_tools -- --no-builtin-tools "what is 1+1"

# Fork with history (demonstrates the "blue parrot" code word surviving the fork)
cargo run -p zen-app --example example_fork -- "what was the code word I just said?"

# Resume with history
cargo run -p zen-app --example example_resume -- "what was the code word I just said?"

# JSON event stream (production print outlet)
cargo run -p zen-app --example example_mode_json -- "introduce yourself in one sentence"

# Load skills
cargo run -p zen-app --example example_skills -- <directory containing SKILL.md> "answer according to the skill"
```

## CLI flag override matrix

| Flag | Example | Status |
|------|---------|------|
| `--provider` | `example_provider` | ✅ |
| `--model` | `example_model` | ✅ |
| `--api-key` | — | ⚠️ Needs wiring (pass the CLI key through to StreamConfig; currently only reads env) |
| `--system-prompt` | `example_system_prompt` | ✅ |
| `--append-system-prompt` | `example_append_system_prompt` | ✅ |
| `--thinking` | `example_thinking` | ✅ |
| `--continue` | `example_resume` | ✅ |
| `--resume` | `example_resume` | ✅ |
| `--version` | `example_version` | ✅ |
| `--mode text/json` | `example_mode_json` | ✅ |
| `--print` | `example_mode_json` (implied) | ✅ |
| `--name` | — | 📋 small flag (session name); usable once wired |
| `--no-session` | `example_no_session` | ✅ |
| `--session` | `example_session_dir` (implied) | ✅ full path (partial UUID lookup deferred to Phase 2) |
| `--session-id` | `example_session_dir` (implied) | ✅ full path |
| `--fork` | `example_fork` | ✅ with history |
| `--session-dir` | `example_session_dir` | ✅ |
| `--models` | — | 📋 Phase 2 (consumed by interactive Ctrl+P list) |
| `--tools` | `example_tools` | ✅ |
| `--exclude-tools` | `example_tools` | ✅ blacklist filtering |
| `--no-tools` | `example_tools` | ✅ |
| `--no-builtin-tools` | `example_tools` | ✅ |
| `--no-skills` | — | 📋 no-op (no skills engine in Phase 1) |
| `--skills` | `example_skills` | ✅ |
| `--prompt-templates` | — | 📋 Phase 2 (template system does not exist) |
| `--no-prompt-templates` | — | 📋 Phase 2 |
| `--themes` | — | 📋 Phase 2 (theme system does not exist) |
| `--no-themes` | — | 📋 Phase 2 |
| `--no-context-files` | `example_no_context_files` | ✅ |
| `--list-models` | `example_list_models` | ✅ |
| `--offline` | — | 📋 small flag (disable network); no corresponding LLM behavior |
| `--verbose` | — | 📋 small flag (verbose startup output); no corresponding LLM behavior |

> **Legend**: ✅ has example · ⚠️ requires code changes · 📋 wire-up only/Phase 2/low priority
