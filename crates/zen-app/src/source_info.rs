//! Resource source information tracking: the full `SourceInfo` model + unified mapping of the three lightweight source enums + completion source labels.
//!
//! REQ-APP-05 (pi-319): resource source information.
//! The assembly chain's `SkillSource`/`PromptTemplateSource`/`ThemeSource` (T-AP-03-3/03-4/05-3) carry only
//! scope semantics (user/project/explicit path); this module supplies the rest of pi-319 — the full `SourceInfo`
//! five-field model (path/source/scope/origin/baseDir), "unified mapping" (`From<*Source> for SourceScope`)
//! and interactive-completion source labels (`source_tag`/`prefix_description`, closing the open issue registered under T-AP-03-2).
//! In-memory assembly artifacts only; not serialized (consistent with the Loaded* type conventions).
//! See tasks/T-AP-05-4-implemented.md for implementation notes.

use std::path::PathBuf;

/// Unified discriminator for resource source scope — the three lightweight source enums (SkillSource/PromptTemplateSource/ThemeSource)
/// converge into this common form. When the interactive completion list adds a source prefix to template/skill descriptions, it first maps the lightweight enum via `From` into this enum,
/// then reads the tag; for debugging provenance, print scope directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceScope {
    /// Loaded from the default directory `<agentDir>/skills|prompts|themes` (pi scope `"user"`).
    User,
    /// Loaded from the default directory `<cwd>/.zen/skills|prompts|themes` (pi scope `"project"`).
    Project,
    /// Loaded from explicit paths (CLI `--skill`/`--prompt-template`/`--theme` / settings extension directories)
    /// （pi scope `"temporary"`）。
    Temporary,
}

/// Discriminator for resource source origin — v1 assembly is always `TopLevel` (local files); `Package` (npm:/git: package sources) is
/// reserved for the REQ-V2-01 extension system. Used when printing origin for debugging provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceOrigin {
    /// Local file / top-level assembly (pi `"top-level"`).
    TopLevel,
    /// npm:/git: package source (pi `"package"`; reserved for the REQ-V2-01 extension system).
    Package,
}

/// Complete provenance of a loaded resource (field-by-field aligned with pi-319 `SourceInfo`) — path, source identifier, scope,
/// origin, and base directory. Resource assembly artifacts (Loaded*'s source + path) are constructed via `SourceInfo::local`;
/// consumed by interactive-completion source prefixes (`source_tag`/`prefix_description`) and debugging provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceInfo {
    /// Resource file path (LoadedSkill.skill.file_path / LoadedPromptTemplate.path /
    /// LoadedTheme.theme.source_path(); builtin themes use the `"<builtin>"` placeholder).
    pub path: String,
    /// Source identifier (always `"local"` in v1; npm:/git: reserved for REQ-V2-01 extension package sources).
    pub source: String,
    /// Source scope (user/project/temporary).
    pub scope: SourceScope,
    /// Source origin kind (always `TopLevel` in v1).
    pub origin: SourceOrigin,
    /// Source root directory (not needed by v1 consumers, always None or filled for debugging; consumed by REQ-V2-01 for short package-source path display).
    pub base_dir: Option<PathBuf>,
}

impl SourceInfo {
    /// The assembly chain builds a "local source origin" for a loaded resource — the v1 equivalent of pi `createSyntheticSourceInfo`:
    /// The loader/autocomplete list assembles the full model from the Loaded* path + lightweight source (first mapped to scope via `From`).
    /// `source` is always `"local"`, `origin` always `TopLevel` (reserved for REQ-V2-01 extension package sources); `base_dir` has no consumers
    /// so pass `None`. Current consumer: `build_slash_command_list` (commands.rs); planned consumer: REQ-V2-01 (package sources).
    pub fn local(path: String, scope: SourceScope, base_dir: Option<PathBuf>) -> SourceInfo {
        SourceInfo {
            path,
            source: "local".to_string(),
            scope,
            origin: SourceOrigin::TopLevel,
            base_dir,
        }
    }

    /// Computes the source tag for a resource command in the interactive-mode autocomplete dropdown — the v1 implementation of pi `getAutocompleteSourceTag`:
    /// Template/skill command descriptions use `[u]/[p]/[t]` prefixes to distinguish user/project/explicit-path sources. v1 source is always `"local"`
    /// so it always returns the scope prefix; the npm:/git: tagged branch is reserved for REQ-V2-01 (the value domain will be extended then).
    ///
    /// **Precondition**: call only on **resource commands that have a source** (templates/skills, with source already attached to Loaded*); builtin
    /// slash commands **do not** call it (matching pi `createBaseAutocompleteProvider`, which only prefixes template/extension/skill
    /// commands with `prefixAutocompleteDescription`, while `BUILTIN_SLASH_COMMANDS` get no prefix) — so the return value is always
    /// a valid tag, with no pi `undefined` branch. Current consumer: `prefix_description` in this module.
    pub fn source_tag(&self) -> String {
        // v1: source is always "local" (locked by the `local()` constructor), so pi's local/auto/cli branch always matches →
        // always returns the scope prefix; the npm:/git: tagged branch is reserved for REQ-V2-01 extension package sources.
        match self.scope {
            SourceScope::User => "u".to_string(),
            SourceScope::Project => "p".to_string(),
            SourceScope::Temporary => "t".to_string(),
        }
    }

    /// Adds a source prefix to an autocomplete description — pi `prefixAutocompleteDescription`: with description → `[u] description`,
    /// no description (None or empty string, matching pi's falsy check) → `[u]`. The interactive-mode command list passes template/skill descriptions
    /// through this prefix before handing them to `SlashCommand::new`.
    ///
    /// **Precondition**: same as `source_tag` — call only on resource commands that have a source, such as templates/skills; builtin commands do not call it
    /// (descriptions pass through as-is, not wrapped in `Some`). Current consumer: `build_slash_command_list` (commands.rs).
    pub fn prefix_description(&self, description: Option<&str>) -> Option<String> {
        let tag = self.source_tag();
        match description {
            Some(desc) if !desc.is_empty() => Some(format!("[{tag}] {desc}")),
            _ => Some(format!("[{tag}]")),
        }
    }
}

// Mapping from the three lightweight source enums → unified scope (closing the "unified mapping" seam; enum types belong to T-AP-03-3/03-4/05-3,
// this task only writes From, no additional impl blocks)
impl From<crate::skills::SkillSource> for SourceScope {
    fn from(source: crate::skills::SkillSource) -> Self {
        match source {
            crate::skills::SkillSource::User => SourceScope::User,
            crate::skills::SkillSource::Project => SourceScope::Project,
            crate::skills::SkillSource::Path => SourceScope::Temporary,
        }
    }
}
impl From<crate::prompt_templates::PromptTemplateSource> for SourceScope {
    fn from(source: crate::prompt_templates::PromptTemplateSource) -> Self {
        match source {
            crate::prompt_templates::PromptTemplateSource::User => SourceScope::User,
            crate::prompt_templates::PromptTemplateSource::Project => SourceScope::Project,
            crate::prompt_templates::PromptTemplateSource::Path => SourceScope::Temporary,
        }
    }
}
impl From<crate::resources::themes::ThemeSource> for SourceScope {
    fn from(source: crate::resources::themes::ThemeSource) -> Self {
        match source {
            crate::resources::themes::ThemeSource::User => SourceScope::User,
            crate::resources::themes::ThemeSource::Project => SourceScope::Project,
            crate::resources::themes::ThemeSource::Path => SourceScope::Temporary,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(scope: SourceScope) -> SourceInfo {
        SourceInfo::local("/p/x.md".to_string(), scope, None)
    }

    // ─── SourceScope mapping (three lightweight enums → unified scope) ─────────────

    #[test]
    fn test_skill_source_mapping() {
        use crate::skills::SkillSource;
        assert_eq!(SourceScope::from(SkillSource::User), SourceScope::User);
        assert_eq!(
            SourceScope::from(SkillSource::Project),
            SourceScope::Project
        );
        assert_eq!(SourceScope::from(SkillSource::Path), SourceScope::Temporary);
    }

    #[test]
    fn test_prompt_template_source_mapping() {
        use crate::prompt_templates::PromptTemplateSource;
        assert_eq!(
            SourceScope::from(PromptTemplateSource::User),
            SourceScope::User
        );
        assert_eq!(
            SourceScope::from(PromptTemplateSource::Project),
            SourceScope::Project
        );
        assert_eq!(
            SourceScope::from(PromptTemplateSource::Path),
            SourceScope::Temporary
        );
    }

    #[test]
    fn test_theme_source_mapping() {
        use crate::resources::themes::ThemeSource;
        assert_eq!(SourceScope::from(ThemeSource::User), SourceScope::User);
        assert_eq!(
            SourceScope::from(ThemeSource::Project),
            SourceScope::Project
        );
        assert_eq!(SourceScope::from(ThemeSource::Path), SourceScope::Temporary);
    }

    // ─── SourceInfo::local (constant-value surface in v1) ───────────────────────────

    #[test]
    fn test_local_constructs_v1_constant_fields() {
        let info = info(SourceScope::User);
        assert_eq!(info.path, "/p/x.md");
        assert_eq!(info.source, "local");
        assert_eq!(info.origin, SourceOrigin::TopLevel);
        assert_eq!(info.base_dir, None);
    }

    #[test]
    fn test_local_preserves_base_dir_when_supplied() {
        let base = PathBuf::from("/agent/skills");
        let info = SourceInfo::local(
            "/agent/skills/a/SKILL.md".to_string(),
            SourceScope::User,
            Some(base.clone()),
        );
        assert_eq!(info.base_dir, Some(base));
    }

    // ─── source_tag (scope prefix) ─────────────────────────────────

    #[test]
    fn test_source_tag_scope_prefixes() {
        assert_eq!(info(SourceScope::User).source_tag(), "u");
        assert_eq!(info(SourceScope::Project).source_tag(), "p");
        assert_eq!(info(SourceScope::Temporary).source_tag(), "t");
    }

    #[test]
    fn test_source_tag_independent_of_path_and_source_field() {
        // v1 source is always "local": even if the field is overwritten by future extensions, the tag still comes from scope (pi local/auto/cli branch)
        let mut i = info(SourceScope::Project);
        i.source = "auto".to_string();
        assert_eq!(i.source_tag(), "p");
    }

    // ─── prefix_description（pi prefixAutocompleteDescription） ───

    #[test]
    fn test_prefix_description_with_description() {
        assert_eq!(
            info(SourceScope::User).prefix_description(Some("Review the diff")),
            Some("[u] Review the diff".to_string())
        );
    }

    #[test]
    fn test_prefix_description_without_description_is_bare_tag() {
        assert_eq!(
            info(SourceScope::Temporary).prefix_description(None),
            Some("[t]".to_string())
        );
    }

    #[test]
    fn test_prefix_description_empty_description_is_bare_tag() {
        // pi falsy check: empty description → tag only (no trailing space)
        assert_eq!(
            info(SourceScope::Project).prefix_description(Some("")),
            Some("[p]".to_string())
        );
    }
}
