use crate::env::ShellConfig;
use crate::tools::bash::BashTool;
use crate::tools::edit::EditTool;
use crate::tools::read::ReadTool;
use crate::tools::write::WriteTool;
use std::collections::HashSet;
use std::sync::Arc;
use zen_ai::tool::{Tool, ToolContext, ToolRegistry};

/// Built-in tool names (aligned with pi `ToolName = "read" | "bash" | "edit" | "write" | "grep" | "find" | "ls"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolName {
    Read,
    Bash,
    Write,
    Edit,
    Grep,
    Find,
    Ls,
}

impl ToolName {
    /// All built-in tool names, in fixed order [read, bash, write, edit, grep, find, ls].
    pub const ALL: [ToolName; 7] = [
        ToolName::Read,
        ToolName::Bash,
        ToolName::Write,
        ToolName::Edit,
        ToolName::Grep,
        ToolName::Find,
        ToolName::Ls,
    ];

    /// Convert to the tool registration name (`Tool::name()` return value / pi's string literal).
    pub fn as_str(&self) -> &'static str {
        match self {
            ToolName::Read => "read",
            ToolName::Bash => "bash",
            ToolName::Write => "write",
            ToolName::Edit => "edit",
            ToolName::Grep => "grep",
            ToolName::Find => "find",
            ToolName::Ls => "ls",
        }
    }
    /// Parse from a string (returns `None` for unknown names; aligned with pi `createTool`'s `default: throw Unknown tool name`).
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "read" => Some(ToolName::Read),
            "bash" => Some(ToolName::Bash),
            "write" => Some(ToolName::Write),
            "edit" => Some(ToolName::Edit),
            "grep" => Some(ToolName::Grep),
            "find" => Some(ToolName::Find),
            "ls" => Some(ToolName::Ls),
            _ => None,
        }
    }
}

/// Create a tool set with an environment by session/cwd (aligned with pi's tool factory).
/// pi: tools are created per session (`createAllTools(cwd, options)`), with tool instances bound to cwd;
/// After aligning zen: instead of globally registering shared stateless tools, create environment-bound tools per session using this factory.
pub struct ToolFactory {
    tool_context: ToolContext,
    /// Shell override config (shell_path/command_prefix, injected per T-AP-02-3-fix; None falls back to defaults)
    shell: ShellConfig,
}

/// Tool selection parameters (aligned with pi's `CreateAgentSessionOptions.tools/excludeTools/noTools`,
/// mapped from the CLI flags `--tools/--exclude-tools/--no-tools/--no-builtin-tools`; see
/// cli-tools-flags-fix). v1 has no extension/custom tools: `no_tools` and `no_builtin_tools`
/// both collapse to "no built-in tools"; the semantic distinction is preserved (custom tools will only be
/// (turned off by `no_tools`).
#[derive(Debug, Clone, Default)]
pub struct ToolSelection {
    /// Allow-list (`--tools`); when non-empty, only tools in the list are created (unknown names silently skipped, aligned with pi).
    pub tools: Vec<String>,
    /// Deny-list (`--exclude-tools`), applied after the allow-list (aligned with pi's "Applies after tools").
    pub exclude_tools: Vec<String>,
    /// Disable all tools (`--no-tools`; when given together with `--tools`, the allow-list takes precedence, aligned with pi).
    pub no_tools: bool,
    /// Disable builtin tool groups (`--no-builtin-tools`; v1 has no extension tools, same effect as `no_tools`).
    pub no_builtin_tools: bool,
}

impl ToolFactory {
    /// Build the factory from session env (no shell override; falls back to system default shell / no prefix).
    pub fn new(tool_context: ToolContext) -> Self {
        Self::with_shell(tool_context, ShellConfig::default())
    }

    /// Build the factory from shell override config (None fields fall back to system default shell / no prefix).
    /// Main assembly (services.rs) feeds in the merged view from settings; `new()` keeps no-shell-override semantics.
    pub fn with_shell(tool_context: ToolContext, shell: ShellConfig) -> Self {
        Self {
            tool_context,
            shell,
        }
    }
    /// Create all built-in tools (full `ToolName` set), injecting the environment into each instance.
    /// Equivalent to `create_with_selection(&ToolSelection::default())` (default builtin groups:
    /// read/bash/edit/write, aligned with pi's `defaultActiveToolNames`).
    pub fn create_all(&self) -> ToolRegistry {
        self.create_with_selection(&ToolSelection::default())
    }

    /// Create a tool set from selection parameters. Aligned with pi's `createAgentSession`:
    /// `initialActiveToolNames =
    ///   (tools ? tools : noTools ? [] : defaultActiveToolNames).filter(!excluded)`：
    /// - `tools` non-empty → only tools in the list; unknown names via `ToolName::from_str` silently skipped (pi only filters among
    ///   already-registered tools; not found → not enabled, no error);
    /// - `tools` empty and `no_tools`/`no_builtin_tools` → empty set;
    /// - otherwise → default builtin groups [read, bash, edit, write] (aligned with pi's `defaultActiveToolNames`);
    /// - `exclude_tools` is set-subtracted from the candidates (applied after `--exclude-tools`).
    /// Unimplemented tools (grep/find/ls; v1 `create` returns None) naturally produce no entries in the list.
    pub fn create_with_selection(&self, selection: &ToolSelection) -> ToolRegistry {
        let excluded: HashSet<ToolName> = selection
            .exclude_tools
            .iter()
            .filter_map(|s| ToolName::from_str(s))
            .collect();
        let candidate: Vec<ToolName> = if !selection.tools.is_empty() {
            selection
                .tools
                .iter()
                .filter_map(|s| ToolName::from_str(s))
                .collect()
        } else if selection.no_tools || selection.no_builtin_tools {
            Vec::new()
        } else {
            vec![
                ToolName::Read,
                ToolName::Bash,
                ToolName::Edit,
                ToolName::Write,
            ]
        };
        let mut registry = ToolRegistry::new();
        for name in candidate {
            if excluded.contains(&name) {
                continue;
            }
            if let Some(tool) = self.create(name) {
                registry.register(tool);
            }
        }
        registry
    }

    /// Create the coding tool group (bash + read + write + edit) — aligned with pi `createCodingTools(cwd)`.
    /// Excludes read-only query tools (grep/find/ls).
    pub fn create_coding(&self) -> ToolRegistry {
        let mut registry = ToolRegistry::new();
        for name in [
            ToolName::Read,
            ToolName::Bash,
            ToolName::Edit,
            ToolName::Write,
        ] {
            if let Some(tool) = self.create(name) {
                registry.register(tool);
            }
        }
        registry
    }

    /// Create the read-only tool group (read + grep + find + ls) — aligned with pi `createReadOnlyTools(cwd)`.
    /// grep/find/ls are not implemented yet (Phase 2), only read is returned (same restriction note as the original create_readonly_tools).
    pub fn create_readonly(&self) -> ToolRegistry {
        let mut registry = ToolRegistry::new();
        for name in [ToolName::Read] {
            if let Some(tool) = self.create(name) {
                registry.register(tool);
            }
        }
        registry
    }

    /// Create a single built-in tool (returns `None` for unknown names, aligned with pi's `Unknown tool name`).
    pub fn create(&self, name: ToolName) -> Option<Arc<dyn Tool>> {
        let tool: Arc<dyn Tool> = match name {
            ToolName::Read => Arc::new(ReadTool::new(self.tool_context.clone())),
            ToolName::Bash => {
                Arc::new(BashTool::new(self.tool_context.clone(), self.shell.clone()))
            }
            ToolName::Edit => Arc::new(EditTool::new(self.tool_context.clone())),
            ToolName::Write => Arc::new(WriteTool::new(self.tool_context.clone())),
            // Grep/Find/Ls tool structs are not implemented yet, cannot be created in v1
            ToolName::Grep | ToolName::Find | ToolName::Ls => return None,
        };
        Some(tool)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_factory() -> ToolFactory {
        ToolFactory::new(ToolContext::new(PathBuf::from("/tmp"), None, None))
    }

    #[test]
    fn tool_name_all_covers_all_variants_in_pi_order() {
        assert_eq!(
            ToolName::ALL,
            [
                ToolName::Read,
                ToolName::Bash,
                ToolName::Write,
                ToolName::Edit,
                ToolName::Grep,
                ToolName::Find,
                ToolName::Ls,
            ]
        );
        assert_eq!(ToolName::ALL.len(), 7);
    }

    #[test]
    fn tool_name_all_has_no_duplicates() {
        for i in 0..ToolName::ALL.len() {
            for j in (i + 1)..ToolName::ALL.len() {
                assert_ne!(ToolName::ALL[i], ToolName::ALL[j]);
            }
        }
    }

    #[test]
    fn tool_name_as_str_maps_to_pi_lowercase_names() {
        assert_eq!(ToolName::Read.as_str(), "read");
        assert_eq!(ToolName::Bash.as_str(), "bash");
        assert_eq!(ToolName::Write.as_str(), "write");
        assert_eq!(ToolName::Edit.as_str(), "edit");
        assert_eq!(ToolName::Grep.as_str(), "grep");
        assert_eq!(ToolName::Find.as_str(), "find");
        assert_eq!(ToolName::Ls.as_str(), "ls");
    }

    #[test]
    fn tool_name_from_str_parses_known_names() {
        assert_eq!(ToolName::from_str("read"), Some(ToolName::Read));
        assert_eq!(ToolName::from_str("bash"), Some(ToolName::Bash));
        assert_eq!(ToolName::from_str("write"), Some(ToolName::Write));
        assert_eq!(ToolName::from_str("edit"), Some(ToolName::Edit));
        assert_eq!(ToolName::from_str("grep"), Some(ToolName::Grep));
        assert_eq!(ToolName::from_str("find"), Some(ToolName::Find));
        assert_eq!(ToolName::from_str("ls"), Some(ToolName::Ls));
    }

    #[test]
    fn tool_name_from_str_rejects_unknown_and_case_mismatch() {
        assert_eq!(ToolName::from_str("exec"), None);
        assert_eq!(ToolName::from_str(""), None);
        assert_eq!(ToolName::from_str("READ"), None);
    }

    #[test]
    fn tool_name_as_str_and_from_str_roundtrip() {
        for name in ToolName::ALL {
            assert_eq!(ToolName::from_str(name.as_str()), Some(name));
        }
    }

    #[test]
    fn factory_create_returns_known_tools_with_matching_names() {
        let factory = test_factory();
        for name in [
            ToolName::Read,
            ToolName::Bash,
            ToolName::Edit,
            ToolName::Write,
        ] {
            let tool = factory
                .create(name)
                .expect("known tool should be creatable");
            assert_eq!(tool.name(), name.as_str());
        }
    }

    #[test]
    fn factory_create_returns_none_for_unimplemented_tools() {
        let factory = test_factory();
        assert!(factory.create(ToolName::Grep).is_none());
        assert!(factory.create(ToolName::Find).is_none());
        assert!(factory.create(ToolName::Ls).is_none());
    }

    #[test]
    fn factory_create_coding_registers_four_tools() {
        let registry = test_factory().create_coding();
        assert_eq!(registry.names(), vec!["read", "bash", "edit", "write"]);
    }

    #[test]
    fn factory_create_readonly_registers_only_read() {
        let registry = test_factory().create_readonly();
        assert_eq!(registry.names(), vec!["read"]);
    }

    #[test]
    fn factory_create_all_currently_registers_coding_group() {
        let registry = test_factory().create_all();
        assert_eq!(registry.names(), vec!["read", "bash", "edit", "write"]);
    }

    // ── create_with_selection（cli-tools-flags-fix：--tools/--exclude-tools/--no-tools/--no-builtin-tools）──

    #[test]
    fn selection_default_equals_create_all() {
        let factory = test_factory();
        assert_eq!(
            factory
                .create_with_selection(&ToolSelection::default())
                .names(),
            factory.create_all().names()
        );
        assert_eq!(
            factory
                .create_with_selection(&ToolSelection::default())
                .names(),
            vec!["read", "bash", "edit", "write"]
        );
    }

    #[test]
    fn selection_allowlist_limits_to_named_tools() {
        let factory = test_factory();
        let sel = ToolSelection {
            tools: vec!["read".into()],
            ..Default::default()
        };
        assert_eq!(
            factory.create_with_selection(&sel).names(),
            vec!["read"],
            "--tools read → read-only session (user motivation scenario)"
        );
    }

    #[test]
    fn selection_allowlist_skips_unknown_names_silently() {
        let factory = test_factory();
        let sel = ToolSelection {
            tools: vec!["read".into(), "does-not-exist".into()],
            ..Default::default()
        };
        // Aligned with pi: unknown names not in registered tools are silently skipped without error
        assert_eq!(factory.create_with_selection(&sel).names(), vec!["read"]);
        // All unknown → empty set
        let all_unknown = ToolSelection {
            tools: vec!["nope".into()],
            ..Default::default()
        };
        assert!(factory
            .create_with_selection(&all_unknown)
            .names()
            .is_empty());
    }

    #[test]
    fn selection_unimplemented_tools_in_allowlist_produce_nothing() {
        // grep/find/ls not implemented in v1 (create returns None): silently produce nothing when listed
        let factory = test_factory();
        let sel = ToolSelection {
            tools: vec!["grep".into()],
            ..Default::default()
        };
        assert!(factory.create_with_selection(&sel).names().is_empty());
    }

    #[test]
    fn selection_exclude_removes_from_default_set() {
        let factory = test_factory();
        let sel = ToolSelection {
            exclude_tools: vec!["bash".into(), "write".into()],
            ..Default::default()
        };
        assert_eq!(
            factory.create_with_selection(&sel).names(),
            vec!["read", "edit"],
            "--exclude-tools bash,write → set difference from default group"
        );
    }

    #[test]
    fn selection_exclude_applied_after_allowlist() {
        let factory = test_factory();
        let sel = ToolSelection {
            tools: vec!["read".into(), "bash".into(), "edit".into()],
            exclude_tools: vec!["bash".into()],
            ..Default::default()
        };
        assert_eq!(
            factory.create_with_selection(&sel).names(),
            vec!["read", "edit"],
            "--tools a,b,c --exclude-tools b → set difference after allowlist (aligned with pi)"
        );
    }

    #[test]
    fn selection_unknown_exclude_is_noop() {
        let factory = test_factory();
        let sel = ToolSelection {
            exclude_tools: vec!["not-a-tool".into()],
            ..Default::default()
        };
        assert_eq!(
            factory.create_with_selection(&sel).names(),
            vec!["read", "bash", "edit", "write"]
        );
    }

    #[test]
    fn selection_no_tools_and_no_builtin_tools_yield_empty() {
        let factory = test_factory();
        let no_tools = ToolSelection {
            no_tools: true,
            ..Default::default()
        };
        let no_builtin = ToolSelection {
            no_builtin_tools: true,
            ..Default::default()
        };
        // v1 has no extension/custom tools: both converge to an empty set
        assert!(factory.create_with_selection(&no_tools).names().is_empty());
        assert!(factory
            .create_with_selection(&no_builtin)
            .names()
            .is_empty());
        // No harm in excluding both
        let both = ToolSelection {
            no_tools: true,
            exclude_tools: vec!["read".into()],
            ..Default::default()
        };
        assert!(factory.create_with_selection(&both).names().is_empty());
    }

    #[test]
    fn selection_allowlist_wins_over_no_tools() {
        // Aligned with pi: `options.tools ? [...] : ...` — allow-list takes precedence over no_tools
        let factory = test_factory();
        let sel = ToolSelection {
            tools: vec!["read".into()],
            no_tools: true,
            ..Default::default()
        };
        assert_eq!(factory.create_with_selection(&sel).names(), vec!["read"]);
    }
}
