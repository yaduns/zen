pub mod bash;
pub mod path_utils;
pub mod read;
pub mod write;

use std::sync::Arc;
use zen_ai::tool::ToolRegistry;

/// Creates the coding tool set (bash + read + write + edit).
pub fn create_coding_tools() -> ToolRegistry {
    todo!()
}

/// Creates the read-only tool set (read + grep + find + ls).
pub fn create_readonly_tools() -> ToolRegistry {
    todo!()
}

/// Creates all tools (bash + read + write + edit + grep + find + ls).
pub fn create_all_tools() -> ToolRegistry {
    todo!()
}
