use crate::tools::bash::BashTool;
use crate::tools::edit::EditTool;
use crate::tools::read::ReadTool;
use crate::tools::write::WriteTool;
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
}

impl ToolFactory {
    /// Constructs a factory from the session environment.
    pub fn new(tool_context: ToolContext) -> Self {
        Self { tool_context }
    }
    /// Create all built-in tools (full `ToolName` set), injecting the environment into each instance.
    pub fn create_all(&self) -> ToolRegistry {
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
            ToolName::Bash => Arc::new(BashTool::new(self.tool_context.clone())),
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
}
