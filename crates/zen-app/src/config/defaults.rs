// REQ-APP-02 (pi-314): default definitions — effective values when settings are absent (effective view)
//
// Two-layer defaults principle: domain defaults (zen-core `DEFAULT_COMPACTION_SETTINGS`, zen-ai Model defaults, etc.)
// stay in their respective domain modules (algorithms still work without settings); this file only defines default decisions for the "settings system" and references
// domain constants as the folding base (without copying values). The sole exit for setting defaults = `fold_effective` → `SettingsEffective`,
// Consumers (services assembly / interactive / T-AP-04-3) obtain effective values via `SettingsManager::effective()`,
// Do not write `unwrap_or(<literal>)` fallbacks in consumers again.
// See reference/pi-original/packages/coding-agent/src/core/defaults.ts and settings-manager.ts
// The `??` fallbacks in each getter (this file consolidates defaults scattered across getters into constants + folding functions).

use crate::config::settings::{
    BranchSummarySettingsPatch, CompactionSettingsPatch, DoubleEscapeAction, MarkdownSettingsPatch,
    RetrySettingsPatch, SettingsData,
};
use zen_ai::types::ThinkingLevel;
use zen_core::compaction::{CompactionSettings, DEFAULT_COMPACTION_SETTINGS};

/// Final fallback for thinking level in the session assembly chain and UI panel: when none of CLI/session restore/settings specifies it,
/// services assembly and the interactive panel use this value, ensuring the repo-wide "absent = High" has a single source.
/// Value decision: pi defaults.ts is "medium"; the zen implementation chain had settled on `Off` (services.rs fallback
/// + plan notes have no requirement basis); **in this task the user decided to enable `High` by default**—this constant is the
/// sole hard-coded point; consumers must not write `Off`/`High` literals again.
pub const DEFAULT_THINKING_LEVEL: ThinkingLevel = ThinkingLevel::High;

/// Default horizontal padding for the input editor (pi `getEditorPaddingX`'s `?? 0`).
pub const DEFAULT_EDITOR_PADDING_X: u32 = 0;

/// Default max visible items in the autocomplete dropdown (pi `getAutocompleteMaxVisible`'s `?? 5`).
/// Note: the current interactive value 10 has no decision record; aligned to pi as 5 (see stub open question 2).
pub const DEFAULT_AUTOCOMPLETE_MAX_VISIBLE: u32 = 5;

/// Default horizontal padding for message output (pi `getOutputPad`: unset is treated as 1; only an explicit 0 is 0).
/// T-AP-02-3-fix already states that "the pi default-1 fallback belongs to T-AP-02-4".
pub const DEFAULT_OUTPUT_PAD: u8 = 1;

/// Default reserved tokens for branch summary (pi `getBranchSummarySettings`'s `?? 16384`).
/// Same value as `DEFAULT_COMPACTION_SETTINGS.reserve_tokens`—pi's two settings groups happen to share the same default,
/// but their semantics are independent (compaction vs branch summary) and can evolve separately, so no cross-group borrowing.
pub const DEFAULT_BRANCH_SUMMARY_RESERVE_TOKENS: u64 = 16_384;

/// Default for skipping branch-summary confirmation (pi `?? false`).
pub const DEFAULT_BRANCH_SUMMARY_SKIP_PROMPT: bool = false;

/// Default for the retry master switch (pi `RetrySettings` enabled defaults to true).
pub const DEFAULT_RETRY_ENABLED: bool = true;

/// Default max retry count (pi `?? 3`).
pub const DEFAULT_RETRY_MAX_RETRIES: u32 = 3;

/// Default base delay in ms for exponential backoff (pi `?? 2000`, 2s/4s/8s…).
pub const DEFAULT_RETRY_BASE_DELAY_MS: u64 = 2_000;

/// Default max retry delay in ms (aligned with pi `provider.maxRetryDelayMs` default 60000).
pub const DEFAULT_RETRY_MAX_DELAY_MS: u64 = 60_000;

/// Default Markdown code block indent (pi `MarkdownSettings.codeBlockIndent` defaults to `"  "`).
pub const DEFAULT_MARKDOWN_CODE_BLOCK_INDENT: &str = "  ";

/// Theme wiring filters path-shaped values when reading the preference: after interactive init() reads theme from settings,
/// it uses this to drop any "custom theme file path" form containing `/` (v1 has no file-based theme loading; prevents it being mistaken for a
/// theme name)—when it returns None, ThemeController falls back to a built-in theme based on the terminal background.
/// Aligned with pi `settings-manager.getTheme()`'s `includes("/") ? undefined : theme`.
pub fn effective_theme(theme: Option<&str>) -> Option<&str> {
    theme.filter(|name| !name.contains('/'))
}

/// Clamp before settings values reach the UI (write path): the panel callback passes the new editor_padding_x through this to clamp it
/// to the valid range 0..=3 before persisting, preventing out-of-range values from breaking the editor layout; for read-path folding see `SettingsEffective.editor_padding_x`.
/// Aligned with pi's setter-time `Math.max(0, Math.min(3, …))` (zen uses read-time folding plus write-time clamping as a double safeguard).
pub fn clamp_editor_padding_x(value: u32) -> u32 {
    value.clamp(0, 3)
}

/// Clamp before settings values reach the UI (write path): the panel callback passes the new autocomplete_max_visible through this before persisting
/// to clamp it to the valid range 3..=20 (0 would swallow the dropdown, too large would overflow the screen). Aligned with pi's setter-time
/// `Math.max(3, Math.min(20, …))`。
pub fn clamp_autocomplete_max_visible(value: u32) -> u32 {
    value.clamp(3, 20)
}

/// Folding of the thresholds used by compaction scheduling: before T-AP-04-3 derives scheduling params from `merged().compaction` (an all-Option override form),
/// it uses this to take zen-core `DEFAULT_COMPACTION_SETTINGS` as the base and patch leaf fields
/// one by one, folding out a complete `CompactionSettings`—the scheduler never touches Option.
/// When patch is None (including still None after merging), returns the domain defaults as-is.
pub fn effective_compaction(patch: Option<&CompactionSettingsPatch>) -> CompactionSettings {
    let mut s = DEFAULT_COMPACTION_SETTINGS;
    if let Some(p) = patch {
        if let Some(v) = p.enabled {
            s.enabled = v;
        }
        if let Some(v) = p.reserve_tokens {
            s.reserve_tokens = v;
        }
        if let Some(v) = p.keep_recent_tokens {
            s.keep_recent_tokens = v;
        }
    }
    s
}

/// Effective form of branch summary settings (complete values after folding defaults, consumed directly by scheduling/UI).
/// Differs from the patch form (`BranchSummarySettingsPatch`, all Option): this type's fields have no Option,
/// absent fields are already filled with defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BranchSummarySettings {
    /// Tokens reserved for the summary prompt + output
    pub reserve_tokens: u64,
    /// When true, skips the "Summarize branch?" confirmation prompt and does not generate a summary by default
    pub skip_prompt: bool,
}

/// Folding of branch summary settings: `merged().branch_summary` (all Option) → complete effective values.
/// Returns all defaults when patch is None. Consumer: T-AP-04-3 (compaction scheduling, branch summary parameter surface).
pub fn effective_branch_summary(
    patch: Option<&BranchSummarySettingsPatch>,
) -> BranchSummarySettings {
    let mut s = BranchSummarySettings {
        reserve_tokens: DEFAULT_BRANCH_SUMMARY_RESERVE_TOKENS,
        skip_prompt: DEFAULT_BRANCH_SUMMARY_SKIP_PROMPT,
    };
    if let Some(p) = patch {
        if let Some(v) = p.reserve_tokens {
            s.reserve_tokens = v;
        }
        if let Some(v) = p.skip_prompt {
            s.skip_prompt = v;
        }
    }
    s
}

/// Effective form of the request retry policy (complete values after folding defaults).
/// `timeout_ms` has no pi default (undefined → SDK layer), so it stays `Option` and is passed through (None = left to
/// the assembly point to decide alongside the sibling `StreamConfig.timeout_ms` parameter).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetrySettings {
    /// Retry master switch
    pub enabled: bool,
    /// Max retry count
    pub max_retries: u32,
    /// Exponential backoff base delay in ms
    pub base_delay_ms: u64,
    /// Per-request timeout in ms (no pi default, passed through)
    pub timeout_ms: Option<u64>,
    /// Max retry delay in ms
    pub max_retry_delay_ms: u64,
}

/// Folding of request retry settings: `merged().retry` (all Option) → complete effective values.
pub fn effective_retry(patch: Option<&RetrySettingsPatch>) -> RetrySettings {
    let mut s = RetrySettings {
        enabled: DEFAULT_RETRY_ENABLED,
        max_retries: DEFAULT_RETRY_MAX_RETRIES,
        base_delay_ms: DEFAULT_RETRY_BASE_DELAY_MS,
        timeout_ms: None,
        max_retry_delay_ms: DEFAULT_RETRY_MAX_DELAY_MS,
    };
    if let Some(p) = patch {
        if let Some(v) = p.enabled {
            s.enabled = v;
        }
        if let Some(v) = p.max_retries {
            s.max_retries = v;
        }
        if let Some(v) = p.base_delay_ms {
            s.base_delay_ms = v;
        }
        if let Some(v) = p.timeout_ms {
            s.timeout_ms = Some(v);
        }
        if let Some(v) = p.max_retry_delay_ms {
            s.max_retry_delay_ms = v;
        }
    }
    s
}

/// Effective form of Markdown rendering settings (after folding defaults).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarkdownSettings {
    /// Code block indent string
    pub code_block_indent: String,
}

/// Folding of Markdown settings: `merged().markdown` → complete effective values.
pub fn effective_markdown(patch: Option<&MarkdownSettingsPatch>) -> MarkdownSettings {
    MarkdownSettings {
        code_block_indent: patch
            .and_then(|p| p.code_block_indent.clone())
            .unwrap_or_else(|| DEFAULT_MARKDOWN_CODE_BLOCK_INDENT.to_string()),
    }
}

/// Effective view of all settings fields—the return form of `SettingsManager::effective()`.
/// Differs from the patch form (`SettingsData`, all Option, absent means unset): this struct folds all
/// defaults (fields with a pi default get the folded value; fields without a pi default stay `Option` and pass through), and consumers
/// only read effective values from here instead of each doing `unwrap_or`.
///
/// Fields not included in this struct: fields already dropped in v1 from `SettingsData` (steering/follow_up, etc., see
/// settings.rs comments); `SettingsConfig` (UI-only static data) is outside the settings defaulting scope.
/// Note: does not derive `PartialEq` (a field contains zen-core `CompactionSettings`, which has no PartialEq).
#[derive(Debug, Clone)]
pub struct SettingsEffective {
    /// Source of AgentState's initial thinking level (passed through; when absent the assembly chain falls back to `DEFAULT_THINKING_LEVEL`)
    pub default_thinking_level: Option<ThinkingLevel>,
    /// Fallback for `--model` when absent (passed through)
    pub default_model: Option<String>,
    /// Fallback for `--provider` when absent (passed through)
    pub default_provider: Option<String>,
    /// Theme preference (path forms containing `/` already filtered out; None = no preference, ThemeController falls back)
    pub theme: Option<String>,
    /// Session storage directory (passed through, normalized via paths on read)
    pub session_dir: Option<String>,
    /// Custom shell path (passed through, supports `~` prefix)
    pub shell_path: Option<String>,
    /// bash command prefix (passed through)
    pub shell_command_prefix: Option<String>,
    /// Ctrl+G external editor command (passed through)
    pub external_editor: Option<String>,
    /// Silence startup banner (pi default false)
    pub quiet_startup: bool,
    /// Hide thinking blocks by default (pi default false)
    pub hide_thinking_block: bool,
    /// Transcript notification for significant cache misses (pi default false)
    pub show_cache_miss_notices: bool,
    /// Hardware cursor during IME positioning (pi default false, except env `PI_HARDWARE_CURSOR=1`)
    pub show_hardware_cursor: bool,
    /// Clear extra blank lines when content shrinks (pi default false)
    pub clear_on_shrink: bool,
    /// Double-Esc action on empty editor (pi default `"tree"`)
    pub double_escape_action: DoubleEscapeAction,
    /// Input editor horizontal padding (pi default 0, clamp 0..=3)
    pub editor_padding_x: u32,
    /// Message output horizontal margin (pi default 1; only an explicit 0 is 0)
    pub output_pad: u8,
    /// Max visible items in completion dropdown (pi default 5, clamp 3..=20)
    pub autocomplete_max_visible: u32,
    /// HTTP idle timeout in milliseconds, 0 = disabled (pi default 300000, but whether the wiring point is the same location is pending,
    /// so it stays passed through; see the `http_idle_timeout_ms` comment in settings.rs)
    pub http_idle_timeout_ms: Option<u64>,
    /// Compaction scheduling parameters (folds `DEFAULT_COMPACTION_SETTINGS`)
    pub compaction: CompactionSettings,
    /// Branch summary parameters (folds defaults)
    pub branch_summary: BranchSummarySettings,
    /// Request retry policy (folds defaults)
    pub retry: RetrySettings,
    /// Markdown rendering tweaks (folds defaults)
    pub markdown: MarkdownSettings,
    /// Extra directories for local skills (passed through)
    pub skills: Vec<String>,
    /// Extra directories for local prompt templates (passed through)
    pub prompts: Vec<String>,
    /// Extra directories for local themes (passed through)
    pub themes: Vec<String>,
}

/// Sole folding entry point for the effective settings view: called internally by `SettingsManager::effective()`; consumers
/// must not touch it directly. Takes `merged()` (patch form: overrides > project > global) as input, field by field
/// folds pi defaults + clamp + path filtering to produce the complete effective values consumers need.
pub fn fold_effective(merged: &SettingsData) -> SettingsEffective {
    SettingsEffective {
        default_thinking_level: merged.default_thinking_level,
        default_model: merged.default_model.clone(),
        default_provider: merged.default_provider.clone(),
        theme: effective_theme(merged.theme.as_deref()).map(str::to_string),
        session_dir: merged.session_dir.clone(),
        shell_path: merged.shell_path.clone(),
        shell_command_prefix: merged.shell_command_prefix.clone(),
        external_editor: merged.external_editor.clone(),
        quiet_startup: merged.quiet_startup.unwrap_or(false),
        hide_thinking_block: merged.hide_thinking_block.unwrap_or(false),
        show_cache_miss_notices: merged.show_cache_miss_notices.unwrap_or(false),
        show_hardware_cursor: merged.show_hardware_cursor.unwrap_or(false),
        clear_on_shrink: merged.clear_on_shrink.unwrap_or(false),
        double_escape_action: merged
            .double_escape_action
            .unwrap_or(DoubleEscapeAction::Tree),
        editor_padding_x: clamp_editor_padding_x(
            merged.editor_padding_x.unwrap_or(DEFAULT_EDITOR_PADDING_X),
        ),
        output_pad: merged.output_pad.unwrap_or(DEFAULT_OUTPUT_PAD),
        autocomplete_max_visible: clamp_autocomplete_max_visible(
            merged
                .autocomplete_max_visible
                .unwrap_or(DEFAULT_AUTOCOMPLETE_MAX_VISIBLE),
        ),
        http_idle_timeout_ms: merged.http_idle_timeout_ms,
        compaction: effective_compaction(merged.compaction.as_ref()),
        branch_summary: effective_branch_summary(merged.branch_summary.as_ref()),
        retry: effective_retry(merged.retry.as_ref()),
        markdown: effective_markdown(merged.markdown.as_ref()),
        skills: merged.skills.clone().unwrap_or_default(),
        prompts: merged.prompts.clone().unwrap_or_default(),
        themes: merged.themes.clone().unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::settings::SettingsData;

    // ── Locked constant values (pi-aligned) ────────────────────────────────

    #[test]
    fn thinking_default_is_high_per_decision() {
        assert_eq!(DEFAULT_THINKING_LEVEL, ThinkingLevel::High);
    }

    #[test]
    fn compaction_defaults_follow_zen_core_const() {
        // Reference domain constants rather than copy: values match zen-core (CompactionSettings has no PartialEq, so assert field by field)
        let s = effective_compaction(None);
        assert!(s.enabled);
        assert_eq!(s.reserve_tokens, DEFAULT_COMPACTION_SETTINGS.reserve_tokens);
        assert_eq!(
            s.keep_recent_tokens,
            DEFAULT_COMPACTION_SETTINGS.keep_recent_tokens
        );
    }

    #[test]
    fn branch_summary_defaults_pi_aligned() {
        let s = effective_branch_summary(None);
        assert_eq!(s.reserve_tokens, 16_384);
        assert!(!s.skip_prompt);
    }

    #[test]
    fn retry_defaults_pi_aligned() {
        let s = effective_retry(None);
        assert!(s.enabled);
        assert_eq!(s.max_retries, 3);
        assert_eq!(s.base_delay_ms, 2_000);
        assert_eq!(s.max_retry_delay_ms, 60_000);
        assert_eq!(
            s.timeout_ms, None,
            "timeout_ms has no pi default, passed through"
        );
    }

    #[test]
    fn markdown_default_indent_two_spaces() {
        let s = effective_markdown(None);
        assert_eq!(s.code_block_indent, "  ");
    }

    // ── Folding behavior ──────────────────────────────────────────────

    #[test]
    fn effective_theme_filters_path_forms() {
        assert_eq!(effective_theme(None), None);
        assert_eq!(effective_theme(Some("dark")), Some("dark"));
        assert_eq!(
            effective_theme(Some("")),
            Some(""),
            "empty string contains no /, no special case, matching pi"
        );
        assert_eq!(effective_theme(Some("my-themes/custom.json")), None);
    }

    #[test]
    fn clamps_converge_out_of_range_values() {
        assert_eq!(clamp_editor_padding_x(0), 0);
        assert_eq!(clamp_editor_padding_x(3), 3);
        assert_eq!(clamp_editor_padding_x(7), 3);
        assert_eq!(clamp_autocomplete_max_visible(3), 3);
        assert_eq!(clamp_autocomplete_max_visible(20), 20);
        assert_eq!(clamp_autocomplete_max_visible(1), 3);
        assert_eq!(clamp_autocomplete_max_visible(99), 20);
    }

    #[test]
    fn effective_compaction_overlays_patch_leaves() {
        let patch = CompactionSettingsPatch {
            enabled: Some(false),
            reserve_tokens: None,
            keep_recent_tokens: Some(30_000),
        };
        let s = effective_compaction(Some(&patch));
        assert!(!s.enabled);
        assert_eq!(
            s.reserve_tokens, 16_384,
            "uncovered leaf fields keep their defaults"
        );
        assert_eq!(s.keep_recent_tokens, 30_000);
    }

    #[test]
    fn fold_effective_empty_settings_yields_all_defaults() {
        let eff = fold_effective(&SettingsData::default());
        assert_eq!(
            eff.default_thinking_level, None,
            "settings fields passed through (absent = None)"
        );
        assert_eq!(eff.theme, None);
        assert_eq!(eff.quiet_startup, false);
        assert_eq!(eff.hide_thinking_block, false);
        assert_eq!(eff.show_cache_miss_notices, false);
        assert_eq!(
            eff.double_escape_action,
            DoubleEscapeAction::Tree,
            "pi default tree"
        );
        assert_eq!(eff.editor_padding_x, 0);
        assert_eq!(
            eff.output_pad, 1,
            "pi default 1, T-AP-02-3-fix fallback assigned to this task"
        );
        assert_eq!(
            eff.autocomplete_max_visible, 5,
            "pi default 5 (interactive current value 10 consolidated)"
        );
        assert!(eff.compaction.enabled);
        assert_eq!(
            eff.compaction.reserve_tokens,
            DEFAULT_COMPACTION_SETTINGS.reserve_tokens
        );
        assert_eq!(
            eff.compaction.keep_recent_tokens,
            DEFAULT_COMPACTION_SETTINGS.keep_recent_tokens
        );
        assert_eq!(eff.branch_summary.reserve_tokens, 16_384);
        assert!(eff.retry.enabled);
        assert_eq!(eff.markdown.code_block_indent, "  ");
        assert!(eff.skills.is_empty());
    }

    #[test]
    fn fold_effective_applies_theme_filter_and_clamps() {
        let data = SettingsData {
            theme: Some("my-themes/custom.json".into()),
            editor_padding_x: Some(9),
            autocomplete_max_visible: Some(1),
            output_pad: Some(0),
            ..Default::default()
        };
        let eff = fold_effective(&data);
        assert_eq!(eff.theme, None, "path-like forms are filtered out");
        assert_eq!(eff.editor_padding_x, 3, "clamp 0..=3");
        assert_eq!(eff.autocomplete_max_visible, 3, "clamp 3..=20");
        assert_eq!(
            eff.output_pad, 0,
            "explicit 0 preserved (pi getOutputPad semantics)"
        );
    }
}
