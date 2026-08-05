pub mod bash;
pub mod edit;
pub mod edit_diff;
pub mod path_utils;
pub mod read;
pub mod write;

use std::sync::Arc;
use zen_ai::tool::ToolRegistry;

/// Creates the coding tool set (bash + read + write + edit).
///
/// Contains all file operation tools needed by the agent to edit code.
/// Excludes read-only query tools (grep, find, ls), which are provided by `create_readonly_tools`.
pub fn create_coding_tools() -> ToolRegistry {
    ToolRegistry::from(vec![
        Arc::new(bash::BashTool::new()) as Arc<dyn zen_ai::tool::Tool>,
        Arc::new(read::ReadTool::new()),
        Arc::new(write::WriteTool::new()),
        Arc::new(edit::EditTool::new()),
    ])
}

/// Creates the read-only tool set (read + grep + find + ls).
///
/// Contains only file query tools, not file modification tools.
/// Suitable for read-only mode or scenarios with strict security checks.
///
/// # Current limitations
///
/// grep, find, and ls tools are not yet implemented (T-AP-04-4, T-AP-04-5, T-AP-04-6, Phase 2 tasks),
/// Currently returns only the read tool. This function must be updated once these tools are implemented.
pub fn create_readonly_tools() -> ToolRegistry {
    // TODO: Add after T-AP-04-4 (grep), T-AP-04-5 (find), T-AP-04-6 (ls) are implemented
    ToolRegistry::from(vec![
        Arc::new(read::ReadTool::new()) as Arc<dyn zen_ai::tool::Tool>,
    ])
}

/// Creates all tools (bash + read + write + edit + grep + find + ls).
///
/// Contains all implemented tools, suitable for a full agent session.
///
/// # Current limitations
///
/// grep, find, and ls tools are not yet implemented (Phase 2 tasks); currently contains only bash, read, write, and edit.
pub fn create_all_tools() -> ToolRegistry {
    // TODO: Add after T-AP-04-4 (grep), T-AP-04-5 (find), T-AP-04-6 (ls) are implemented
    ToolRegistry::from(vec![
        Arc::new(bash::BashTool::new()) as Arc<dyn zen_ai::tool::Tool>,
        Arc::new(read::ReadTool::new()),
        Arc::new(write::WriteTool::new()),
        Arc::new(edit::EditTool::new()),
    ])
}
