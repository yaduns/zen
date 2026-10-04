// REQ-APP-02 (pi-321): Settings manager — two-level (global/project) settings.json persistence + in-memory override layer (session-level)
// + FieldDiff change-tracking write-back + hot reload.
// The "three levels" in name and fact: two persistent files (global/project) + `apply_overrides` in-memory override layer (not persisted, corresponding to
// CLI args / session-level temporary overrides); merge order (read side) overrides > project > global, write-back isolated by scope.
// See reference/pi-original/packages/coding-agent/src/core/settings-manager.ts.
//
// ⚠️ Iron rule (captured from plan T-TU-02-1-fix): runtime-affecting fields (default_thinking_level, etc.)
// are only a source of initial values here and must not become the authoritative copy at runtime — AgentState is the single authority;
// the settings panel is a one-way view of AgentState.

use std::fmt;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use zen_ai::types::ThinkingLevel;
use zen_core::paths::{resolve_path, PathInputOptions};

use super::defaults;
use super::CONFIG_DIR_NAME;

/// Empty editor double-Esc behavior discriminator (aligned one by one with pi's `doubleEscapeAction` values).
/// serde lowercase: variant → `"fork"` / `"tree"` / `"none"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DoubleEscapeAction {
    /// Double Esc opens session fork
    Fork,
    /// Double Esc opens session tree
    Tree,
    /// No action
    None,
}

/// Compaction toggle and token threshold (Option-ized override shape of pi's `CompactionSettings`).
/// Field aligned with `zen_core::compaction::CompactionSettings`;
/// Default value fallbacks are already handled by defaults.rs (T-AP-02-4: `effective_compaction` folds `DEFAULT_COMPACTION_SETTINGS`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionSettingsPatch {
    /// Auto-compaction toggle (should_compact is always false when false)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Tokens reserved for the summary prompt + output (aligned with Model.max_tokens; pi default 16384)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reserve_tokens: Option<u64>,
    /// Approximate token budget for recent context retained after compaction (pi default 20000)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_recent_tokens: Option<u64>,
}

/// Branch summary settings (pi `BranchSummarySettings`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummarySettingsPatch {
    /// Tokens reserved for the summary prompt + output (pi default 16384)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reserve_tokens: Option<u64>,
    /// When true, skip the "Summarize branch?" confirmation prompt and default to not generating a summary (pi default false)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_prompt: Option<bool>,
}

/// Request retry policy (pi `RetrySettings`; the `retry.provider` sub-object is flattened in,
/// single provider, no per-provider dimension). Default values are already handled by defaults.rs (T-AP-02-4: `effective_retry`)
/// （pi：enabled=true / maxRetries=3 / baseDelayMs=2000 / timeoutMs / maxRetryDelayMs=60000）。
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrySettingsPatch {
    /// Master switch for retries (pi default true)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Retry attempt limit (pi default 3)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,
    /// Base delay in ms for exponential backoff (2s, 4s, 8s…, pi default 2000)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_delay_ms: Option<u64>,
    /// Per-request timeout in ms (SDK/provider request timeout)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// Retry delay cap in ms (aligned with pi `provider.maxRetryDelayMs`, default 60000)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retry_delay_ms: Option<u64>,
}

/// Markdown rendering tweaks (pi `MarkdownSettings`).
/// Consumers: the existing `code_block_indent: String` field in zen-tui `markdown.rs` (default `"  "`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MarkdownSettingsPatch {
    /// Code block indent string (pi default `"  "`)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_block_indent: Option<String>,
}

/// Settings data for a single scope (user preferences persisted in settings.json).
/// serde camelCase: settings.json keys are mutually recognized with pi (`defaultThinkingLevel` /
/// `doubleEscapeAction`, etc.). zen uses a new format with no legacy baggage but keeps pi key names
/// for cross-reference and future tool interoperability. Unknown keys are ignored during deserialization (preserved on the persist side via
/// `serde_json::Value` merging).
/// `skip_serializing_if`: `None` fields serialize as **absent** rather than `null` — this is
/// Serialization representation of "default = unset"; also the FieldDiff deletion semantics (key disappearing = deletion path)
/// ...and the premise that persist only writes modified fields.
///
/// v1 fields dropped (present in pi but not produced by zen): lastChangelogVersion / collapseChangelog /
/// enableInstallTelemetry / enableAnalytics / trackingId /
/// transport / enabledModels / websocketConnectTimeoutMs / httpProxy /
/// packages / extensions / npmCommand / enableSkillCommands / images, and
/// terminal.showImages / terminal.imageWidthCells / terminal.showTerminalProgress /
/// defaultProjectTrust / steeringMode / followUpMode / thinkingBudgets /
/// warnings / treeFilterMode (rationale in tasks/T-AP-02-3-implemented.md).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsData {
    /// Source of AgentState's initial thinking level (assembled by consumers T-AP-11-1 / T-AP-08-3);
    /// the authoritative runtime state lives in AgentState; this field is only the initial value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_thinking_level: Option<ThinkingLevel>,
    /// Fallback when `--model` is unset (CLI --model > settings > built-in default; consumer T-AP-08-3)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    /// Provider fallback when `--provider` is unset (CLI --provider > session restore >
    /// settings.default_provider > ZEN_PROVIDER env > built-in default; consumer T-AP-08-3
    /// assembly — T-AP-02-3 supplementary fields: multi-provider support already implemented in zen-ai (opencode-go/huoshan))
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_provider: Option<String>,
    /// Theme preference (consumer T-AP-11-3 `ThemeController::new` parameter slot;
    /// Path-form entries containing `/` are already ignored by defaults.rs (T-AP-02-4: `effective_theme`, folded into `SettingsEffective.theme`)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub theme: Option<String>,
    /// Session storage directory (consumer T-AP-08-5; same format as `--session-dir`,
    /// normalized via `zen_core::paths` on read)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_dir: Option<String>,
    /// Suppress startup banner (consumer: interactive/print startup path)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quiet_startup: Option<bool>,
    /// Custom shell path (consumer T-AP-04-2 `getShellEnv` deferred item;
    /// supports `~` prefix, normalized via paths on read)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shell_path: Option<String>,
    /// bash command prefix (consumer T-AP-04-2)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shell_command_prefix: Option<String>,
    /// Ctrl+G external editor command, takes precedence over VISUAL/EDITOR (consumer T-AP-11-4, Phase 3)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_editor: Option<String>,
    /// Thinking blocks hidden by default (consumer T-AP-11-2 initial display state;
    /// Ctrl+T run-state toggle belongs to AgentState / key routing)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hide_thinking_block: Option<bool>,
    /// Transcript notice for significant cache miss (consumer T-OB-01-3, Step 6)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_cache_miss_notices: Option<bool>,
    /// Double Esc action in empty editor (consumer T-AP-11-1 key routing)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub double_escape_action: Option<DoubleEscapeAction>,
    /// Input editor horizontal padding (consumer zen-tui editor; clamp 0..=3 already handled by defaults.rs: `clamp_editor_padding_x`)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub editor_padding_x: Option<u32>,
    /// Message output horizontal margin 0/1 (consumer render component)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_pad: Option<u8>,
    /// Max visible completion dropdown items (consumer zen-tui autocomplete; clamp 3..=20 already handled by defaults.rs: `clamp_autocomplete_max_visible`)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub autocomplete_max_visible: Option<u32>,
    /// Hardware cursor during IME positioning (consumer zen-tui editor)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_hardware_cursor: Option<bool>,
    /// Clear extra blank lines when content shrinks (flattened pi `terminal.clearOnShrink`;
    /// consumer zen-tui editor rendering)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clear_on_shrink: Option<bool>,
    /// HTTP idle timeout in milliseconds, 0 = disabled (consumer zen-ai request boundary `StreamConfig.timeout_ms`
    /// same-family parameter slot, T-AP-08-3 assembly point; whether it shares the slot with the overall request timeout is pending decision, see implementation notes "Open Issues")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub http_idle_timeout_ms: Option<u64>,
    /// Compression toggle and token threshold (consumer T-AP-04-3; defaults
    /// `zen_core::compaction::DEFAULT_COMPACTION_SETTINGS`）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compaction: Option<CompactionSettingsPatch>,
    /// Branch summary reserved tokens / skip confirmation (consumer T-AP-04-3 / T-AG-04-2 parameter surface)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch_summary: Option<BranchSummarySettingsPatch>,
    /// Request retry policy (consumer zen-ai `RetryPolicy` assembly)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetrySettingsPatch>,
    /// Rendering tweaks (consumer zen-tui `markdown.rs` existing `code_block_indent` field)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub markdown: Option<MarkdownSettingsPatch>,
    /// Extra directories for local skills (consumer T-AP-03-3 directory discovery)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skills: Option<Vec<String>>,
    /// Extra directories for local prompt templates (consumer T-AP-05-3 directory discovery)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<Vec<String>>,
    /// Extra directories for local themes (consumer T-AP-05-3 directory discovery)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub themes: Option<Vec<String>>,
    /// Ctrl+P model cycle set (`None`=all enabled; `Some`=ordered subset, elements are `"provider/id"`).
    /// Consumer T-V2-02-4 (`/scoped-models` panel persistence + `app.model.cycle*` cycling).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled_models: Option<Vec<String>>,
}

/// Scope discriminator for settings (pure discriminator, not serialized; not part of the settings.json key space).
/// `Display` output aligns with pi scope literals (`"global"` / `"project"`),
/// for `SettingsError` `#[error]` message interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsScope {
    /// Global file: `<agentDir>/settings.json`
    Global,
    /// Project file: `<projectCwd>/.zen/settings.json`
    Project,
}

impl fmt::Display for SettingsScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Global => "global",
            Self::Project => "project",
        })
    }
}

/// Error surface of the settings read/write path (thiserror) — load/reload/update failures **do not panic**;
/// errors go into the `SettingsManager` error queue (pi `recordError` equivalent), consumed by
/// `drain_errors()` for display. zen-app is app-level (anyhow aggregation), but consumers need
/// to distinguish "Parse degraded but continue" from "Write failed, needs a prompt", hence concrete variants.
#[derive(Debug, Error)]
pub enum SettingsError {
    /// File read IO failure (open/read content; file-not-found excluded — first launch with no file = normal empty settings,
    /// aligned with pi's `existsSync` pre-check)
    #[error("failed to read {scope} settings file {path}: {source}")]
    Io {
        /// Scope of the file that errored
        scope: SettingsScope,
        /// Original file path that errored (for diagnostics display)
        path: PathBuf,
        /// Underlying IO error (error chain preserved via `#[source]`)
        #[source]
        source: std::io::Error,
    },
    /// JSON parse/structure validation failure; **non-fatal** (pi tryLoad semantics: corrupt file degrades to empty +
    /// recorded in the error queue, startup not blocked)
    #[error("failed to parse {scope} settings file {path}: {source}")]
    Parse {
        /// Scope of the file that errored
        scope: SettingsScope,
        /// Original file path that errored (for diagnostics display)
        path: PathBuf,
        /// Underlying serde error (error chain preserved via `#[source]`)
        #[source]
        source: serde_json::Error,
    },
    /// Disk write failure (any step of tempfile write + rename atomic replacement);
    /// in-memory state is already updated (this session's changes take effect, just not persisted)
    #[error("failed to write {scope} settings file {path}: {source}")]
    Write {
        /// Scope of the file that errored
        scope: SettingsScope,
        /// Original file path that errored (for diagnostics display)
        path: PathBuf,
        /// Underlying IO error (error chain preserved via `#[source]`)
        #[source]
        source: std::io::Error,
    },
}

/// **Modified path set** obtained by serializing `SettingsData` before and after an update to `serde_json::Value` and diffing recursively
/// (JSON Pointer style, e.g. `/theme`, `/compaction/reserveTokens`).
/// It's a type alias rather than a single-field struct: consumers are only the two free functions in the same file and
/// `update_global`/`update_project`, with no cross-file boundary; the "removed path" semantics are derived by
/// `apply_diff_to_value` checking path absence in `new` — a single-field struct has no invariant to protect.
/// Path segments are always field names (serde camelCase), with no `~` or `/` special characters, so RFC 6901 escaping is unnecessary;
/// arrays are compared as a whole (`skills`/`prompts`/`themes` changes record the top-level path, replaced wholesale);
/// empty Vec = no change (no disk write); the diff is transient within the update call, with no cross-call tracked state.
pub(crate) type FieldDiff = Vec<String>;

/// Compute the set of modified paths between two SettingsData Value views (JSON Pointer style).
/// Inputs are the serialized views of two `SettingsData` values (one before and one after the update);
/// the output paths are consumed by `apply_diff_to_value` for write-back — kept as a free function for independent testing.
/// `Some→None` records a deletion path; `None→Some` records an addition path; nested objects recurse deeper.
pub(crate) fn diff_settings_values(old: &serde_json::Value, new: &serde_json::Value) -> FieldDiff {
    let mut paths = Vec::new();
    diff_values(old, new, "", &mut paths);
    paths
}

/// Recursive diff: when both sides are objects, drill down per key (present only in old = deletion, only in new = addition);
/// for all other types, skip if equal, record the current path if not. `prefix` is the parent path (`""` = root).
fn diff_values(
    old: &serde_json::Value,
    new: &serde_json::Value,
    prefix: &str,
    out: &mut FieldDiff,
) {
    match (old, new) {
        (serde_json::Value::Object(old_map), serde_json::Value::Object(new_map)) => {
            for (key, old_val) in old_map {
                let child = child_path(prefix, key);
                match new_map.get(key) {
                    Some(new_val) => diff_values(old_val, new_val, &child, out),
                    // present in old, absent in new = deletion path (the apply side checks for absence in new and performs remove)
                    None => out.push(child),
                }
            }
            for key in new_map.keys() {
                if !old_map.contains_key(key) {
                    out.push(child_path(prefix, key)); // addition path
                }
            }
        }
        (a, b) if a == b => {}
        // wholesale replacement (including object↔scalar shape changes); a differing root (empty prefix) is a defensive branch,
        // in normal calls both old and new are serialization products of SettingsData (always objects), so this does not occur
        _ => {
            if !prefix.is_empty() {
                out.push(prefix.to_string());
            }
        }
    }
}

/// Join a child segment: `"/key"` for the first segment under the root, `"{parent}/key"` when nested.
fn child_path(prefix: &str, key: &str) -> String {
    if prefix.is_empty() {
        format!("/{key}")
    } else {
        format!("{prefix}/{key}")
    }
}

/// Parse a JSON Pointer path into a segment list (RFC 6901: `~1` → `/`, `~0` → `~`;
/// path segments in this task are always fixed field names; the escaping rules are kept for conformance with the standard semantics).
fn pointer_segments(pointer: &str) -> Vec<String> {
    if pointer.is_empty() {
        return Vec::new();
    }
    pointer[1..]
        .split('/')
        .map(|seg| seg.replace("~1", "/").replace("~0", "~"))
        .collect()
}

/// The write-back half of persist: apply the set of modified paths to the "existing Value of the file",
/// overwriting only keys the user changed and preserving other keys in the file (including keys unknown to this task).
/// If the path exists in `new` → write it (fill in missing intermediate levels as objects); if absent → delete
/// (the `Some→None` deletion path semantics).
pub(crate) fn apply_diff_to_value(
    target: &mut serde_json::Value,
    diff: &[String],
    new: &serde_json::Value,
) {
    for path in diff {
        let segments = pointer_segments(path);
        match new.pointer(path) {
            Some(value) => insert_path(target, &segments, value.clone()),
            None => remove_path(target, &segments),
        }
    }
}

/// Write the value along the path segments: any missing intermediate level that is not an object is replaced/filled in as an object (in the normal update flow,
/// all intermediate levels are serialization products of patch structs; the replacement branch only defends against illegal nesting in hand-edited files).
fn insert_path(target: &mut serde_json::Value, segments: &[String], value: serde_json::Value) {
    if segments.is_empty() {
        *target = value; // root replacement (the current call shape never produces an empty path; defensive branch)
        return;
    }
    if !target.is_object() {
        *target = serde_json::Value::Object(serde_json::Map::new());
    }
    // Previous line already normalized to object, so this branch is physically unreachable; let-else early-return satisfies the borrow checker without unwrap.
    let serde_json::Value::Object(map) = target else {
        return;
    };
    if segments.len() == 1 {
        map.insert(segments[0].clone(), value);
    } else {
        let child = map
            .entry(segments[0].clone())
            .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
        insert_path(child, &segments[1..], value);
    }
}

/// Delete value along a path segment: if any link in the parent chain is absent, treat as already deleted; root path is not deleted (empty-segment defense).
fn remove_path(target: &mut serde_json::Value, segments: &[String]) {
    let Some((last, parents)) = segments.split_last() else {
        return;
    };
    let mut current = target;
    for seg in parents {
        let serde_json::Value::Object(map) = current else {
            return;
        };
        let Some(child) = map.get_mut(seg) else {
            return;
        };
        current = child;
    }
    if let serde_json::Value::Object(map) = current {
        map.remove(last);
    }
}

/// Merge two settings: fields set in `over` override `base` (per-field Option override,
/// nested patch groups recurse to leaf fields); array fields (directory lists) are replaced wholesale, elements not merged.
/// Aligns with pi `deepMergeSettings` (nested objects recurse, scalars: later wins).
fn merge_settings(base: &SettingsData, over: &SettingsData) -> SettingsData {
    SettingsData {
        default_thinking_level: over.default_thinking_level.or(base.default_thinking_level),
        default_model: over.default_model.clone().or(base.default_model.clone()),
        default_provider: over
            .default_provider
            .clone()
            .or(base.default_provider.clone()),
        theme: over.theme.clone().or(base.theme.clone()),
        session_dir: over.session_dir.clone().or(base.session_dir.clone()),
        quiet_startup: over.quiet_startup.or(base.quiet_startup),
        shell_path: over.shell_path.clone().or(base.shell_path.clone()),
        shell_command_prefix: over
            .shell_command_prefix
            .clone()
            .or(base.shell_command_prefix.clone()),
        external_editor: over
            .external_editor
            .clone()
            .or(base.external_editor.clone()),
        hide_thinking_block: over.hide_thinking_block.or(base.hide_thinking_block),
        show_cache_miss_notices: over
            .show_cache_miss_notices
            .or(base.show_cache_miss_notices),
        double_escape_action: over.double_escape_action.or(base.double_escape_action),
        editor_padding_x: over.editor_padding_x.or(base.editor_padding_x),
        output_pad: over.output_pad.or(base.output_pad),
        autocomplete_max_visible: over
            .autocomplete_max_visible
            .or(base.autocomplete_max_visible),
        show_hardware_cursor: over.show_hardware_cursor.or(base.show_hardware_cursor),
        clear_on_shrink: over.clear_on_shrink.or(base.clear_on_shrink),
        http_idle_timeout_ms: over.http_idle_timeout_ms.or(base.http_idle_timeout_ms),
        compaction: merge_compaction(base.compaction.as_ref(), over.compaction.as_ref()),
        branch_summary: merge_branch_summary(
            base.branch_summary.as_ref(),
            over.branch_summary.as_ref(),
        ),
        retry: merge_retry(base.retry.as_ref(), over.retry.as_ref()),
        markdown: merge_markdown(base.markdown.as_ref(), over.markdown.as_ref()),
        skills: over.skills.clone().or(base.skills.clone()),
        prompts: over.prompts.clone().or(base.prompts.clone()),
        themes: over.themes.clone().or(base.themes.clone()),
        enabled_models: over.enabled_models.clone().or(base.enabled_models.clone()),
    }
}

/// Nested-group leaf-level merge: both None → None; otherwise per-leaf `over.or(base)`.
/// An intermediate "empty patch" (`Some` but all None) does not shadow base leaf values (object recursive-merge semantics).
fn merge_compaction(
    base: Option<&CompactionSettingsPatch>,
    over: Option<&CompactionSettingsPatch>,
) -> Option<CompactionSettingsPatch> {
    if base.is_none() && over.is_none() {
        return None;
    }
    let base = base.cloned().unwrap_or_default();
    let over = over.cloned().unwrap_or_default();
    Some(CompactionSettingsPatch {
        enabled: over.enabled.or(base.enabled),
        reserve_tokens: over.reserve_tokens.or(base.reserve_tokens),
        keep_recent_tokens: over.keep_recent_tokens.or(base.keep_recent_tokens),
    })
}

/// Same as `merge_compaction`, applied to the `branch_summary` group.
fn merge_branch_summary(
    base: Option<&BranchSummarySettingsPatch>,
    over: Option<&BranchSummarySettingsPatch>,
) -> Option<BranchSummarySettingsPatch> {
    if base.is_none() && over.is_none() {
        return None;
    }
    let base = base.cloned().unwrap_or_default();
    let over = over.cloned().unwrap_or_default();
    Some(BranchSummarySettingsPatch {
        reserve_tokens: over.reserve_tokens.or(base.reserve_tokens),
        skip_prompt: over.skip_prompt.or(base.skip_prompt),
    })
}

/// Same as `merge_compaction`, applied to the `retry` group (pi `retry.provider` is flattened and merged in).
fn merge_retry(
    base: Option<&RetrySettingsPatch>,
    over: Option<&RetrySettingsPatch>,
) -> Option<RetrySettingsPatch> {
    if base.is_none() && over.is_none() {
        return None;
    }
    let base = base.cloned().unwrap_or_default();
    let over = over.cloned().unwrap_or_default();
    Some(RetrySettingsPatch {
        enabled: over.enabled.or(base.enabled),
        max_retries: over.max_retries.or(base.max_retries),
        base_delay_ms: over.base_delay_ms.or(base.base_delay_ms),
        timeout_ms: over.timeout_ms.or(base.timeout_ms),
        max_retry_delay_ms: over.max_retry_delay_ms.or(base.max_retry_delay_ms),
    })
}

/// Same as `merge_compaction`, applied to the `markdown` group.
fn merge_markdown(
    base: Option<&MarkdownSettingsPatch>,
    over: Option<&MarkdownSettingsPatch>,
) -> Option<MarkdownSettingsPatch> {
    if base.is_none() && over.is_none() {
        return None;
    }
    let base = base.cloned().unwrap_or_default();
    let over = over.cloned().unwrap_or_default();
    Some(MarkdownSettingsPatch {
        code_block_indent: over.code_block_indent.or(base.code_block_indent),
    })
}

/// In-memory state of a single scope: `data` is the strongly typed view (for merged/snapshot/update diff),
/// `raw` is the file's full Value view (includes keys unknown to this task, kept when persisting).
/// Both are produced together by `read_scope`; `raw` is always an object (parse failure degrades to `{}`).
struct ScopeState {
    data: SettingsData,
    raw: serde_json::Value,
    path: PathBuf,
}

/// Manager internal state (interior mutability via `Mutex`, equivalent to pi's `&self` setter semantics).
struct Inner {
    global: ScopeState,
    project: ScopeState,
    /// Project working directory (base for resolving session_dir relative paths).
    project_cwd: PathBuf,
    /// In-memory override layer (injected by `apply_overrides`, not persisted; highest precedence in merge).
    overlay: SettingsData,
    /// Accumulated read/write errors (taken by `drain_errors`, cleared after taking).
    errors: Vec<SettingsError>,
}

impl Inner {
    fn state(&self, scope: SettingsScope) -> &ScopeState {
        match scope {
            SettingsScope::Global => &self.global,
            SettingsScope::Project => &self.project,
        }
    }

    fn state_mut(&mut self, scope: SettingsScope) -> &mut ScopeState {
        match scope {
            SettingsScope::Global => &mut self.global,
            SettingsScope::Project => &mut self.project,
        }
    }

    /// Merged view: global < project < override layer (per-leaf recursive merge of nested groups).
    fn merged_view(&self) -> SettingsData {
        let base = merge_settings(&self.global.data, &self.project.data);
        merge_settings(&base, &self.overlay)
    }
}

/// Main entry assembly: load the two-level settings files into memory, solving "user preferences changed must take effect in this session".
/// A corrupt file does not block startup (degrades to empty + queued as an error, retrieved via `drain_errors`).
/// File not found = normal empty settings (first launch); no error recorded.
pub struct SettingsManager {
    inner: Mutex<Inner>,
}

impl SettingsManager {
    /// Scenario preamble: the main entry assembly (T-AP-08-3) calls this at process startup, reading both files to build the in-memory view.
    /// Global file = `agent_dir/settings.json`, project file = `project_cwd/.zen/settings.json`
    /// (directory name reuses config::CONFIG_DIR_NAME). Parse errors are downgraded and enqueued.
    pub fn load(project_cwd: &Path, agent_dir: &Path) -> SettingsManager {
        let global_path = agent_dir.join("settings.json");
        let project_path = project_cwd.join(CONFIG_DIR_NAME).join("settings.json");
        let mut errors = Vec::new();
        let (global_data, global_raw, global_err) = read_scope(&global_path, SettingsScope::Global);
        let (project_data, project_raw, project_err) =
            read_scope(&project_path, SettingsScope::Project);
        errors.extend(global_err);
        errors.extend(project_err);
        SettingsManager {
            inner: Mutex::new(Inner {
                global: ScopeState {
                    data: global_data,
                    raw: global_raw,
                    path: global_path,
                },
                project: ScopeState {
                    data: project_data,
                    raw: project_raw,
                    path: project_path,
                },
                project_cwd: project_cwd.to_path_buf(),
                overlay: SettingsData::default(),
                errors,
            }),
        }
    }

    /// In-memory empty settings (default-value view, no file mounts) — for test/quick-start paths (`AgentSession::new`);
    /// real assembly always goes through `load` (two-level file view + error queue).
    pub fn empty() -> SettingsManager {
        let empty_scope = |path: PathBuf| ScopeState {
            data: SettingsData::default(),
            raw: serde_json::Value::Object(Default::default()),
            path,
        };
        SettingsManager {
            inner: Mutex::new(Inner {
                global: empty_scope(PathBuf::new()),
                project: empty_scope(PathBuf::new()),
                project_cwd: PathBuf::new(),
                overlay: SettingsData::default(),
                errors: Vec::new(),
            }),
        }
    }

    /// Returns the merged full settings view (overlay > project > global, recursive to leaf fields of nested groups),
    /// for one-shot snapshot reads by each domain. Returns a clone (pi's structuredClone equivalent);
    /// Default value filling does not happen at this layer (T-AP-02-4 effective view, see `effective()`).
    pub fn merged(&self) -> SettingsData {
        self.lock_inner().merged_view()
    }

    /// Get the **effective view** (defaults folded) of all settings fields: `merged()` (all-Option patch form)
    /// folded via `defaults::fold_effective` — pi defaults + clamp + path filtering, producing
    /// `SettingsEffective` — consumers (services assembly / interactive / T-AP-04-3) always go through this
    /// to get effective values, instead of each doing its own `unwrap_or` fallback.
    /// Now handled by defaults.rs (T-AP-02-4).
    pub fn effective(&self) -> defaults::SettingsEffective {
        let merged = self.lock_inner().merged_view();
        defaults::fold_effective(&merged)
    }

    /// Single-scope snapshot (excludes the other level and the overlay) — lets the effective view and diagnostics distinguish
    /// "what the user wrote in the global file" from "what the project overrode".
    pub fn scope_snapshot(&self, scope: SettingsScope) -> SettingsData {
        self.lock_inner().state(scope).data.clone()
    }

    /// CLI/session-level temporary overrides (e.g. --model) injection: affects only the `merged()` view,
    /// never written to disk (global/project in-memory snapshots unchanged). Repeated calls accumulate per layer
    /// (pi's `applyOverrides` equivalent; multiple merges have identical semantics).
    pub fn apply_overrides(&self, overrides: SettingsData) {
        let mut inner = self.lock_inner();
        inner.overlay = merge_settings(&inner.overlay, &overrides);
    }

    /// Settings panel / startup UI saving user changes: modify global-scope fields inside the callback; the manager
    /// diffs out the modified paths and writes them back to the global file (writes only modified fields, merged with the file's existing content;
    /// pi's multiple setXxx equivalent). Write failures don't panic: memory state is still updated (effective for this session),
    /// the error is enqueued (retrieve via `drain_errors`). Empty diff (callback changed nothing) skips the write.
    pub fn update_global(&self, f: impl FnOnce(&mut SettingsData)) {
        self.update_scope(SettingsScope::Global, f);
    }

    /// Same as `update_global`, but targets the project-level file (`cwd/.zen/settings.json`; the directory is
    /// auto-created on write if missing). v1 is always trusted (project-trust v2 was dropped); no write gate.
    pub fn update_project(&self, f: impl FnOnce(&mut SettingsData)) {
        self.update_scope(SettingsScope::Project, f);
    }

    /// Explicit refresh entry point after the user hand-edits settings.json in another terminal/editor (v1 has no file
    /// watching): per scope, "overwrite the in-memory snapshot only if the read succeeds; on failure keep the current value + enqueue the error"
    /// (matches pi's conditional overwrite on reload); a deleted file is treated as empty settings (disk is the source of truth).
    /// In-memory overlay retained (CLI/session overrides remain in effect).
    pub fn reload(&self) {
        let mut inner = self.lock_inner();
        let (global_data, global_raw, global_err) =
            read_scope(&inner.global.path, SettingsScope::Global);
        if let Some(err) = global_err {
            inner.errors.push(err);
        } else {
            inner.global.data = global_data;
            inner.global.raw = global_raw;
        }
        let (project_data, project_raw, project_err) =
            read_scope(&inner.project.path, SettingsScope::Project);
        if let Some(err) = project_err {
            inner.errors.push(err);
        } else {
            inner.project.data = project_data;
            inner.project.raw = project_raw;
        }
    }

    /// Diagnostics/status bar pick up accumulated read/write errors at the right moment and show them to the user, so bad files aren't silently ignored
    /// (cleared on read, pi drainErrors semantics).
    pub fn drain_errors(&self) -> Vec<SettingsError> {
        std::mem::take(&mut self.lock_inner().errors)
    }

    /// Get the session_dir setting and normalize it via `zen_core::paths::resolve_path`
    /// (`~` expansion, relative paths absolutized against the project cwd), addressing the user-visible issue of "relative/~ paths written in settings"
    /// None = unset or resolution failed (e.g. home unavailable, treated as unset;
    /// fallback logic belongs to the caller).
    pub fn session_dir(&self) -> Option<PathBuf> {
        let inner = self.lock_inner();
        let raw = inner.merged_view().session_dir?;
        resolve_path(&raw, &inner.project_cwd, &PathInputOptions::default()).ok()
    }

    /// Shared update skeleton: clone under lock → mutate the clone via callback → serialize diff → mutate memory → atomic write to disk.
    /// Single critical section throughout (disk write is synchronous, file <10KB, low frequency); callback f operates on the clone,
    /// so panics can't corrupt internal state (poison recovery see `lock_inner`).
    fn update_scope(&self, scope: SettingsScope, f: impl FnOnce(&mut SettingsData)) {
        let mut inner = self.lock_inner();
        let old_data = inner.state(scope).data.clone();
        let mut new_data = old_data.clone();
        f(&mut new_data);

        // Serialization is theoretically infallible (no non-string-key maps / no float members); defensive fallback:
        // map to the Write variant and enqueue, in-memory state unchanged
        let old_value = match serde_json::to_value(&old_data) {
            Ok(value) => value,
            Err(e) => {
                let path = inner.state(scope).path.clone();
                inner.errors.push(SettingsError::Write {
                    scope,
                    path,
                    source: std::io::Error::new(std::io::ErrorKind::InvalidData, e),
                });
                return;
            }
        };
        let new_value = match serde_json::to_value(&new_data) {
            Ok(value) => value,
            Err(e) => {
                let path = inner.state(scope).path.clone();
                inner.errors.push(SettingsError::Write {
                    scope,
                    path,
                    source: std::io::Error::new(std::io::ErrorKind::InvalidData, e),
                });
                return;
            }
        };
        let diff = diff_settings_values(&old_value, &new_value);
        if diff.is_empty() {
            return; // callback changed no fields, skip disk write
        }

        // mutate memory first (still effective this session if the write fails), raw evolves in place (= expected file contents,
        // monotonically accumulating including unwritten changes; next write self-heals, avoiding rollback losing changes)
        let path = {
            let state = inner.state_mut(scope);
            state.data = new_data;
            apply_diff_to_value(&mut state.raw, &diff, &new_value);
            state.path.clone()
        };
        if let Err(source) = write_atomic(&path, &inner.state(scope).raw) {
            inner.errors.push(SettingsError::Write {
                scope,
                path,
                source,
            });
        }
    }

    /// Lock acquisition (poison recovery): code under the lock doesn't break invariants (callback operates on the clone, write failures only enqueue),
    /// so state left after a panic is still a valid snapshot; take the data and continue (standard poison-recovery idiom).
    fn lock_inner(&self) -> MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Read a single scope file → (strongly-typed view, full Value, error).
/// File missing = normal empty settings (no error, aligned with pi's `existsSync` pre-check);
/// IO/parse/structure validation failure = degrade to empty + record error (does not block startup).
/// Structural validation via `from_value::<SettingsData>`: unknown keys are ignored and preserved in raw;
/// Mismatched types on known fields or a non-object root count as a corrupt file; both data and raw are demoted
/// (Strict demotion: corrupt files are not rescued via partial; the error is queued and observable).
fn read_scope(
    path: &Path,
    scope: SettingsScope,
) -> (SettingsData, serde_json::Value, Option<SettingsError>) {
    let empty = || {
        (
            SettingsData::default(),
            serde_json::Value::Object(Default::default()),
            None,
        )
    };
    if !path.exists() {
        return empty();
    }
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        // Race between exists and read (file deleted): handled per first-start semantics
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => return empty(),
        Err(source) => {
            return (
                SettingsData::default(),
                serde_json::Value::Object(Default::default()),
                Some(SettingsError::Io {
                    scope,
                    path: path.to_path_buf(),
                    source,
                }),
            );
        }
    };
    let raw: serde_json::Value = match serde_json::from_str(&content) {
        Ok(raw) => raw,
        Err(source) => {
            return (
                SettingsData::default(),
                serde_json::Value::Object(Default::default()),
                Some(SettingsError::Parse {
                    scope,
                    path: path.to_path_buf(),
                    source,
                }),
            );
        }
    };
    match serde_json::from_value::<SettingsData>(raw.clone()) {
        Ok(data) => (data, raw, None),
        Err(source) => (
            SettingsData::default(),
            serde_json::Value::Object(Default::default()),
            Some(SettingsError::Parse {
                scope,
                path: path.to_path_buf(),
                source,
            }),
        ),
    }
}

/// Atomic write: temp file in the same directory + rename replacement (tempfile::persist); a process crash cannot corrupt
/// existing files (satisfies pi proper-lockfile's corruption-prevention goal; cross-process exclusion is still dropped, see the seam note).
/// Missing parent directories are created automatically (project `.zen/` first-write scenario).
fn write_atomic(path: &Path, raw: &serde_json::Value) -> std::io::Result<()> {
    let content = serde_json::to_string_pretty(raw)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    std::fs::create_dir_all(&parent)?;
    let mut tmp = tempfile::NamedTempFile::new_in(&parent)?;
    tmp.write_all(content.as_bytes())?;
    tmp.persist(path).map_err(|persist_err| persist_err.error)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Test helper: temporary agent_dir + project cwd (mutually disjoint, cleaned up when TempDir is dropped).
    struct Fixture {
        _guard: tempfile::TempDir,
        agent_dir: PathBuf,
        project_cwd: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let guard = tempfile::tempdir().unwrap();
            let agent_dir = guard.path().join("agent");
            let project_cwd = guard.path().join("project");
            std::fs::create_dir_all(&agent_dir).unwrap();
            std::fs::create_dir_all(&project_cwd).unwrap();
            Self {
                _guard: guard,
                agent_dir,
                project_cwd,
            }
        }

        fn global_path(&self) -> PathBuf {
            self.agent_dir.join("settings.json")
        }

        fn project_path(&self) -> PathBuf {
            self.project_cwd.join(CONFIG_DIR_NAME).join("settings.json")
        }

        fn load(&self) -> SettingsManager {
            SettingsManager::load(&self.project_cwd, &self.agent_dir)
        }
    }

    // ── diff_settings_values ────────────────────────────────────

    #[test]
    fn test_diff_scalar_change_records_path() {
        let old = json!({"theme": "dark"});
        let new = json!({"theme": "light"});
        assert_eq!(diff_settings_values(&old, &new), vec!["/theme".to_string()]);
    }

    #[test]
    fn test_diff_nested_change_records_two_segment_path() {
        let old = json!({"compaction": {"reserveTokens": 1, "keepRecentTokens": 2}});
        let new = json!({"compaction": {"reserveTokens": 9, "keepRecentTokens": 2}});
        assert_eq!(
            diff_settings_values(&old, &new),
            vec!["/compaction/reserveTokens".to_string()]
        );
    }

    #[test]
    fn test_diff_added_and_removed_keys() {
        let old = json!({"theme": "dark"});
        let new = json!({"defaultModel": "m"});
        assert_eq!(
            diff_settings_values(&old, &new),
            vec!["/theme".to_string(), "/defaultModel".to_string()]
        );
    }

    #[test]
    fn test_diff_array_replaced_whole() {
        let old = json!({"skills": ["a"]});
        let new = json!({"skills": ["a", "b"]});
        // Whole-array comparison: only top-level paths are recorded, no element-level diff
        assert_eq!(
            diff_settings_values(&old, &new),
            vec!["/skills".to_string()]
        );
    }

    #[test]
    fn test_diff_equal_returns_empty() {
        let value = json!({"theme": "dark", "compaction": {"enabled": true}});
        assert!(diff_settings_values(&value, &value).is_empty());
    }

    // ── apply_diff_to_value ─────────────────────────────────────

    #[test]
    fn test_apply_sets_nested_and_creates_intermediates() {
        let mut target = json!({"existing": 1});
        let new = json!({"compaction": {"reserveTokens": 5}});
        apply_diff_to_value(
            &mut target,
            &["/compaction/reserveTokens".to_string()],
            &new,
        );
        assert_eq!(
            target,
            json!({"existing": 1, "compaction": {"reserveTokens": 5}})
        );
    }

    #[test]
    fn test_apply_removes_path_absent_in_new() {
        let mut target = json!({"theme": "dark", "keep": 1});
        let new = json!({"keep": 1});
        apply_diff_to_value(&mut target, &["/theme".to_string()], &new);
        assert_eq!(target, json!({"keep": 1}));
    }

    #[test]
    fn test_apply_preserves_unknown_keys() {
        let mut target = json!({"theme": "dark", "customFutureKey": [1, 2]});
        let new = json!({"theme": "light"});
        apply_diff_to_value(&mut target, &["/theme".to_string()], &new);
        assert_eq!(target, json!({"theme": "light", "customFutureKey": [1, 2]}));
    }

    // ── load / merged / scope_snapshot ──────────────────────────

    #[test]
    fn test_load_missing_files_yields_empty_without_errors() {
        let fx = Fixture::new();
        let manager = fx.load();
        assert_eq!(manager.merged(), SettingsData::default());
        assert!(manager.drain_errors().is_empty());
    }

    #[test]
    fn test_load_merges_project_over_global() {
        let fx = Fixture::new();
        std::fs::write(
            fx.global_path(),
            r#"{"theme": "dark", "quietStartup": true}"#,
        )
        .unwrap();
        std::fs::create_dir_all(fx.project_path().parent().unwrap()).unwrap();
        std::fs::write(fx.project_path(), r#"{"theme": "light"}"#).unwrap();
        let manager = fx.load();
        let merged = manager.merged();
        assert_eq!(merged.theme.as_deref(), Some("light")); // Project overrides global
        assert_eq!(merged.quiet_startup, Some(true)); // Global-only fields are preserved
        assert_eq!(
            manager
                .scope_snapshot(SettingsScope::Global)
                .theme
                .as_deref(),
            Some("dark")
        );
        assert_eq!(
            manager
                .scope_snapshot(SettingsScope::Project)
                .theme
                .as_deref(),
            Some("light")
        );
        assert!(manager.drain_errors().is_empty());
    }

    #[test]
    fn test_load_corrupt_file_degrades_to_empty_with_parse_error() {
        let fx = Fixture::new();
        std::fs::write(fx.global_path(), "{oops").unwrap();
        let manager = fx.load();
        assert_eq!(manager.merged(), SettingsData::default());
        let errors = manager.drain_errors();
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0],
            SettingsError::Parse {
                scope: SettingsScope::Global,
                ..
            }
        ));
    }

    #[test]
    fn test_load_type_mismatch_degrades_with_parse_error() {
        let fx = Fixture::new();
        // theme expects a string, a number is written → from_value structural validation fails
        std::fs::create_dir_all(fx.project_path().parent().unwrap()).unwrap();
        std::fs::write(fx.project_path(), r#"{"theme": 123}"#).unwrap();
        let manager = fx.load();
        assert_eq!(manager.merged(), SettingsData::default());
        let errors = manager.drain_errors();
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0],
            SettingsError::Parse {
                scope: SettingsScope::Project,
                ..
            }
        ));
    }

    #[test]
    fn test_load_io_error_recorded() {
        let fx = Fixture::new();
        // A directory occupies the settings.json path: exists is true but read fails → Io variant (unix semantics)
        std::fs::create_dir(fx.global_path()).unwrap();
        let manager = fx.load();
        let errors = manager.drain_errors();
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0],
            SettingsError::Io {
                scope: SettingsScope::Global,
                ..
            }
        ));
    }

    // ── apply_overrides ─────────────────────────────────────────

    #[test]
    fn test_apply_overrides_only_affects_merged_view() {
        let fx = Fixture::new();
        std::fs::write(fx.global_path(), r#"{"theme": "dark"}"#).unwrap();
        let manager = fx.load();
        let overrides = SettingsData {
            theme: Some("light".to_string()),
            ..Default::default()
        };
        manager.apply_overrides(overrides);
        // The merged view is overridden
        assert_eq!(manager.merged().theme.as_deref(), Some("light"));
        // Neither the single-scope snapshot nor the file is affected
        assert_eq!(
            manager
                .scope_snapshot(SettingsScope::Global)
                .theme
                .as_deref(),
            Some("dark")
        );
        assert_eq!(
            std::fs::read_to_string(fx.global_path()).unwrap(),
            r#"{"theme": "dark"}"#
        );
    }

    // ── update_global / update_project ──────────────────────────

    #[test]
    fn test_update_global_writes_camel_case_pretty_json() {
        let fx = Fixture::new();
        let manager = fx.load();
        manager.update_global(|s| {
            s.default_thinking_level = Some(ThinkingLevel::High);
            s.theme = Some("dark".to_string());
        });
        assert!(manager.drain_errors().is_empty());
        let content = std::fs::read_to_string(fx.global_path()).unwrap();
        assert!(content.contains("\"defaultThinkingLevel\""));
        assert!(content.contains("\"theme\""));
        assert!(content.contains("\"high\"")); // ThinkingLevel serde lowercase
                                               // Written content can be read back
        let manager2 = fx.load();
        assert_eq!(
            manager2.merged().default_thinking_level,
            Some(ThinkingLevel::High)
        );
    }

    #[test]
    fn test_update_global_noop_skips_write() {
        let fx = Fixture::new();
        std::fs::write(fx.global_path(), r#"{"theme": "dark"}"#).unwrap();
        let manager = fx.load();
        manager.update_global(|_| {}); // Empty callback = empty diff
        assert!(manager.drain_errors().is_empty());
        assert_eq!(
            std::fs::read_to_string(fx.global_path()).unwrap(),
            r#"{"theme": "dark"}"#
        );
    }

    #[test]
    fn test_update_preserves_unknown_keys_and_untouched_fields() {
        let fx = Fixture::new();
        std::fs::write(
            fx.global_path(),
            r#"{"theme": "dark", "customFutureKey": 42, "compaction": {"enabled": false}}"#,
        )
        .unwrap();
        let manager = fx.load();
        manager.update_global(|s| s.default_model = Some("m".to_string()));
        assert!(manager.drain_errors().is_empty());
        let raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fx.global_path()).unwrap()).unwrap();
        // Only changed fields are written: unknown keys and untouched fields are preserved verbatim
        assert_eq!(raw["customFutureKey"], json!(42));
        assert_eq!(raw["theme"], json!("dark"));
        assert_eq!(raw["compaction"]["enabled"], json!(false));
        assert_eq!(raw["defaultModel"], json!("m"));
    }

    #[test]
    fn test_update_project_creates_zen_dir() {
        let fx = Fixture::new();
        assert!(!fx.project_path().parent().unwrap().exists()); // .zen/ does not exist yet
        let manager = fx.load();
        manager.update_project(|s| s.theme = Some("light".to_string()));
        assert!(manager.drain_errors().is_empty());
        assert!(fx.project_path().exists());
        let raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fx.project_path()).unwrap()).unwrap();
        assert_eq!(raw["theme"], json!("light"));
    }

    #[test]
    fn test_update_write_failure_queues_error_but_updates_memory() {
        let fx = Fixture::new();
        // A directory occupies the settings.json path: load records Io; persisting onto the directory → Write fails
        std::fs::create_dir(fx.global_path()).unwrap();
        let manager = fx.load();
        assert_eq!(manager.drain_errors().len(), 1); // take the Io error from load first
        manager.update_global(|s| s.default_model = Some("m".to_string()));
        let errors = manager.drain_errors();
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0],
            SettingsError::Write {
                scope: SettingsScope::Global,
                ..
            }
        ));
        // in-memory state is still updated (effective for this session, just not persisted)
        assert_eq!(manager.merged().default_model.as_deref(), Some("m"));
    }

    #[test]
    fn test_update_global_does_not_leak_into_project_file() {
        let fx = Fixture::new();
        let manager = fx.load();
        manager.update_global(|s| s.theme = Some("dark".to_string()));
        manager.update_project(|s| s.theme = Some("light".to_string()));
        assert!(manager.drain_errors().is_empty());
        let global: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fx.global_path()).unwrap()).unwrap();
        let project: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fx.project_path()).unwrap()).unwrap();
        // write-back is isolated per scope, no cross-contamination
        assert_eq!(global["theme"], json!("dark"));
        assert_eq!(project["theme"], json!("light"));
    }

    // ── reload ──────────────────────────────────────────────────

    #[test]
    fn test_reload_picks_up_external_edit() {
        let fx = Fixture::new();
        let manager = fx.load();
        assert_eq!(manager.merged().theme, None);
        std::fs::write(fx.global_path(), r#"{"theme": "solarized"}"#).unwrap();
        manager.reload();
        assert_eq!(manager.merged().theme.as_deref(), Some("solarized"));
        assert!(manager.drain_errors().is_empty());
    }

    #[test]
    fn test_reload_corrupt_file_keeps_old_value_and_records_error() {
        let fx = Fixture::new();
        std::fs::write(fx.global_path(), r#"{"theme": "dark"}"#).unwrap();
        let manager = fx.load();
        std::fs::write(fx.global_path(), "{broken").unwrap();
        manager.reload();
        // keep existing value on read failure
        assert_eq!(manager.merged().theme.as_deref(), Some("dark"));
        let errors = manager.drain_errors();
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0],
            SettingsError::Parse {
                scope: SettingsScope::Global,
                ..
            }
        ));
    }

    #[test]
    fn test_reload_deleted_file_resets_to_empty() {
        let fx = Fixture::new();
        std::fs::write(fx.global_path(), r#"{"theme": "dark"}"#).unwrap();
        let manager = fx.load();
        std::fs::remove_file(fx.global_path()).unwrap();
        manager.reload();
        // file deleted = disk is source of truth, reset to empty settings, not an error
        assert_eq!(manager.merged().theme, None);
        assert!(manager.drain_errors().is_empty());
    }

    // ── session_dir ─────────────────────────────────────────────

    #[test]
    fn test_session_dir_resolves_relative_against_project_cwd() {
        let fx = Fixture::new();
        std::fs::write(fx.global_path(), r#"{"sessionDir": "sessions-store"}"#).unwrap();
        let manager = fx.load();
        assert_eq!(
            manager.session_dir(),
            Some(fx.project_cwd.join("sessions-store"))
        );
    }

    #[test]
    fn test_session_dir_absolute_passthrough_and_unset() {
        let fx = Fixture::new();
        std::fs::write(
            fx.global_path(),
            r#"{"sessionDir": "/tmp/zen-sessions-abs"}"#,
        )
        .unwrap();
        let manager = fx.load();
        assert_eq!(
            manager.session_dir(),
            Some(PathBuf::from("/tmp/zen-sessions-abs"))
        );
        let unset = Fixture::new();
        assert_eq!(unset.load().session_dir(), None);
    }

    // ── Type level: serde / Display / nested merge ──────────────────────

    #[test]
    fn test_settings_data_camel_case_roundtrip_and_none_absent() {
        let data: SettingsData = serde_json::from_str(
            r#"{"defaultThinkingLevel": "high", "compaction": {"reserveTokens": 100}}"#,
        )
        .unwrap();
        assert_eq!(data.default_thinking_level, Some(ThinkingLevel::High));
        assert_eq!(data.compaction.as_ref().unwrap().reserve_tokens, Some(100));
        assert_eq!(data.theme, None);
        // None fields serialize as absent, not null
        let out = serde_json::to_value(&data).unwrap();
        assert!(out.get("theme").is_none());
        assert!(out.get("defaultModel").is_none());
        assert_eq!(out["compaction"]["reserveTokens"], json!(100));
    }

    #[test]
    fn test_default_provider_serde_and_merge() {
        // camelCase roundtrip: defaultProvider key
        let data: SettingsData =
            serde_json::from_str(r#"{"defaultProvider": "huoshan", "defaultModel": "m"}"#).unwrap();
        assert_eq!(data.default_provider.as_deref(), Some("huoshan"));
        let out = serde_json::to_value(&data).unwrap();
        assert_eq!(out["defaultProvider"], json!("huoshan"));
        // merge: overwriter already set → override; not set → keep base value
        let base = SettingsData {
            default_provider: Some("opencode-go".into()),
            ..Default::default()
        };
        let over = SettingsData {
            default_provider: Some("huoshan".into()),
            ..Default::default()
        };
        assert_eq!(
            merge_settings(&base, &over).default_provider.as_deref(),
            Some("huoshan")
        );
        assert_eq!(
            merge_settings(&base, &SettingsData::default())
                .default_provider
                .as_deref(),
            Some("opencode-go")
        );
    }

    #[test]
    fn test_unknown_keys_ignored_on_deserialize() {
        let data: SettingsData =
            serde_json::from_str(r#"{"theme": "dark", "futureField": {"a": 1}}"#).unwrap();
        assert_eq!(data.theme.as_deref(), Some("dark"));
    }

    #[test]
    fn test_double_escape_action_serde_lowercase() {
        assert_eq!(
            serde_json::to_string(&DoubleEscapeAction::Fork).unwrap(),
            "\"fork\""
        );
        assert_eq!(
            serde_json::to_string(&DoubleEscapeAction::Tree).unwrap(),
            "\"tree\""
        );
        assert_eq!(
            serde_json::to_string(&DoubleEscapeAction::None).unwrap(),
            "\"none\""
        );
        let back: DoubleEscapeAction = serde_json::from_str("\"tree\"").unwrap();
        assert_eq!(back, DoubleEscapeAction::Tree);
    }

    #[test]
    fn test_settings_scope_display_matches_pi_literals() {
        assert_eq!(SettingsScope::Global.to_string(), "global");
        assert_eq!(SettingsScope::Project.to_string(), "project");
    }

    #[test]
    fn test_nested_patch_merge_is_leaf_field_level() {
        // global sets reserve_tokens, project sets enabled → leaf fields complement each other, not wholesale replacement
        let base = SettingsData {
            compaction: Some(CompactionSettingsPatch {
                enabled: None,
                reserve_tokens: Some(16384),
                keep_recent_tokens: None,
            }),
            ..Default::default()
        };
        let over = SettingsData {
            compaction: Some(CompactionSettingsPatch {
                enabled: Some(false),
                reserve_tokens: None,
                keep_recent_tokens: None,
            }),
            ..Default::default()
        };
        let merged = merge_settings(&base, &over);
        let compaction = merged.compaction.unwrap();
        assert_eq!(compaction.enabled, Some(false));
        assert_eq!(compaction.reserve_tokens, Some(16384));
    }

    #[test]
    fn test_enabled_models_merge_project_over_global() {
        // T-V2-02-4: enabled_models has the same "whole-array replacement" semantics as skills/prompts/themes —
        // Project-level overrides global; if over is absent, fall back to base.
        let base = SettingsData {
            enabled_models: Some(vec!["a/x".to_string()]),
            ..Default::default()
        };
        let over = SettingsData {
            enabled_models: Some(vec!["b/y".to_string()]),
            ..Default::default()
        };
        assert_eq!(
            merge_settings(&base, &over).enabled_models.as_deref(),
            Some(["b/y".to_string()].as_slice())
        );
        assert_eq!(
            merge_settings(&base, &SettingsData::default())
                .enabled_models
                .as_deref(),
            Some(["a/x".to_string()].as_slice())
        );
    }

    #[test]
    fn test_empty_patch_does_not_shadow_base_leaves() {
        // project wrote an empty patch group {}: does not shadow globally set leaf values (object recursive merge semantics)
        let base = SettingsData {
            compaction: Some(CompactionSettingsPatch {
                enabled: Some(true),
                reserve_tokens: None,
                keep_recent_tokens: None,
            }),
            ..Default::default()
        };
        let over = SettingsData {
            compaction: Some(CompactionSettingsPatch::default()),
            ..Default::default()
        };
        let merged = merge_settings(&base, &over);
        assert_eq!(merged.compaction.unwrap().enabled, Some(true));
    }
}
