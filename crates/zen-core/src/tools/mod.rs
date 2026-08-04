use std::sync::Arc;
use zen_ai::tool::ToolRegistry;

/// Creates the coding tool set (bash + read + write + edit).
///
/// # Parameters
/// - `cwd`: current working directory (passed to the tool context).
///
/// # Return Value
/// A `ToolRegistry` containing 4 tools.
pub fn create_coding_tools(cwd: &str) -> ToolRegistry {
    todo!()
}

/// Creates the read-only tool set (read + grep + find + ls).
///
/// # Parameters
/// - `cwd`: Current working directory.
///
/// # Return Value
/// A `ToolRegistry` containing 4 tools.
pub fn create_readonly_tools(cwd: &str) -> ToolRegistry {
    todo!()
}

/// Creates all tools (bash + read + write + edit + grep + find + ls).
///
/// # Parameters
/// - `cwd`: Current working directory.
///
/// # Return Value
/// A `ToolRegistry` containing 7 tools.
pub fn create_all_tools(cwd: &str) -> ToolRegistry {
    todo!()
}
