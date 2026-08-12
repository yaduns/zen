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
