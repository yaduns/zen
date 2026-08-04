//! Tool trait and execution types for the AI layer.
//!
//! This module defines the `Tool` trait (runtime tool execution), its error
//! and callback types, and the `ToolExecutionResult` wrapper. The serializable
//! `ToolDefinition` struct lives in `types.rs` alongside `Context`.
//!
//! REQ-AGENT-02 (pi-210): Tool trait, merged from zen_ai::types::ToolDefinition (data
//! struct) and zen_core::agent::types::AgentTool (trait, eliminated).

use crate::types::{ToolDefinition, ToolResultMessage};
use async_trait::async_trait;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Runtime tool execution callback, used to stream partial results.
/// The callback is only valid during the current `execute()` call — calls after the Future resolves are ignored.
pub type ToolUpdateCallback = Box<dyn Fn(ToolExecutionResult) + Send + Sync>;

/// Tool execution result, containing the message injected into the LLM context and runtime control signals.
///
/// `message` is pure data; it is stored in the session transcript and fed into the LLM context.
/// `terminate` is a runtime control signal telling the agent loop whether to terminate early after the current tool batch.
/// The two are separated because `terminate` does not belong to the message data.
#[derive(Debug, Clone, Serialize)]
pub struct ToolExecutionResult {
    /// Message injected into the LLM context
    pub message: ToolResultMessage,
    /// Whether to terminate the agent loop early (triggered when all tool results have terminate=true)
    pub terminate: bool,
}

/// Runtime context during tool execution.
///
/// Contains the shared state required for tool execution (e.g. working directory, session ID, permissions).
/// Tools are stateless; all external dependencies are passed in at execution time via `ToolContext`.
/// Each tool picks the fields it needs; unused fields are simply ignored.
///
/// Phase 1 contains only `working_dir`; later extended by T-AG-02-2 (session_id, permissions, etc.).
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Current working directory, used for resolving relative paths and executing commands
    pub working_dir: PathBuf,
    pub session_id: Option<String>,
    pub allowed_directories: Option<Vec<PathBuf>>,
}

impl ToolContext {
    /// Creates a new tool context.
    pub fn new(
        working_dir: PathBuf,
        session_id: Option<String>,
        allowed_directories: Option<Vec<PathBuf>>,
    ) -> Self {
        Self {
            working_dir: working_dir,
            session_id: session_id,
            allowed_directories: allowed_directories,
        }
    }
}

/// Tool execution error type.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// Tool execution failed
    #[error("Tool execution failed: {0}")]
    Execution(String),
    /// Argument validation failed
    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),
    /// User cancelled
    #[error("Tool execution cancelled")]
    Cancelled,
}

/// Lightweight wrapper around a tool set.
///
/// Provides lookup by name, conversion to an API definition list, iteration, emptiness checks, etc.
/// Not persisted, not serialized — constructed purely at runtime.
///
/// # Design decisions
/// - Uses `Vec<Arc<dyn Tool>>` instead of `HashMap<String, Arc<dyn Tool>>`,
///   because the tool count is small (< 10), so a linear scan is fine and registration order is preserved.
/// - Provides flexible construction via `FromIterator` and `From<Vec<Arc<dyn Tool>>>`.
///
/// # Edge Cases
/// - Empty registry: `is_empty()` returns true, `get_by_name()` always returns None.
/// - Duplicate tool names: when two tools with the same name are registered, `get_by_name()` returns the first match.
/// - Thread-safe: `Send + Sync` (guaranteed via `Arc<dyn Tool>`).
pub struct ToolRegistry {
    /// Tool list, ordered by registration. `Arc` provides shared ownership, `dyn Tool` provides runtime polymorphism.
    pub tools: Vec<Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Create an empty tool registry.
    pub fn new() -> Self {
        Self { tools: vec![] }
    }

    /// Register a tool.
    ///
    /// # Parameters
    /// - `tool`: the tool instance to register.
    ///
    /// # Edge Cases
    /// - Registering a tool with a duplicate name is allowed (the later one does not overwrite, but `get_by_name` returns the first match).
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.push(tool);
    }

    /// Find a tool by name.
    ///
    /// # Parameters
    /// - `name`: the tool name (compared against `Tool::name()`'s return value).
    ///
    /// # Return Value
    /// The first matching tool found, or `None`.
    ///
    /// # Edge Cases
    /// - The comparison is case-sensitive (consistent with `Tool::name()`'s return value).
    pub fn get_by_name(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.iter().find(|tool| tool.name() == name)
    }

    /// Get the list of all tool names.
    pub fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|tool| tool.name()).collect()
    }

    /// Convert to a list of API definitions (sent to the LLM).
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .map(|tool| tool.to_api_definition())
            .collect()
    }

    /// Number of tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether it is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Iterate over all tools (in registration order).
    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn Tool>> {
        self.tools.iter()
    }
}

impl FromIterator<Arc<dyn Tool>> for ToolRegistry {
    fn from_iter<I: IntoIterator<Item = Arc<dyn Tool>>>(iter: I) -> Self {
        Self {
            tools: iter.into_iter().collect(),
        }
    }
}

impl From<Vec<Arc<dyn Tool>>> for ToolRegistry {
    fn from(tools: Vec<Arc<dyn Tool>>) -> Self {
        Self { tools }
    }
}

/// Runtime tool definition, containing the tool's metadata, parameter schema, and execution logic.
///
/// Merges the former `zen_ai::types::ToolDefinition` (data struct) and `zen_core::agent::types::AgentTool` (trait, now removed).
/// All tools (bash/read/write/edit etc.) implement this trait.
///
/// Contract (inherits Pi's original constraints):
/// - `execute()` should return `Err` on failure rather than encoding the error in `content`.
/// - Continuing to call the `on_update` callback after `execute()` returns has no effect.
/// - `prepare_arguments()` is optional; the default implementation returns the input directly.
#[async_trait]
pub trait Tool: std::fmt::Debug + Send + Sync {
    /// Tool name, used for LLM tool selection and log identification.
    fn name(&self) -> &str;

    /// Tool description, visible to the LLM, used to decide whether to call this tool.
    fn description(&self) -> &str;

    /// Tool parameter JSON Schema, used by the LLM to generate valid arguments.
    fn parameters(&self) -> &serde_json::Value;

    /// Human-readable tool label for UI display.
    fn label(&self) -> &str;

    /// Optional argument preprocessing hook that type-coerces raw arguments before schema validation.
    /// Returns the transformed argument object (must match the Schema defined by `parameters`).
    /// Default implementation returns the input as-is.
    fn prepare_arguments(&self, args: &serde_json::Value) -> Result<serde_json::Value, ToolError> {
        Ok(args.clone())
    }

    /// Executes the tool call.
    ///
    /// # Parameters
    /// - `tool_call_id`: tool call ID, used to match the tool result message.
    /// - `params`: Arguments after processing by `prepare_arguments`.
    /// - `ctx`: Runtime context (working directory, etc.). Tools pick fields as needed; unused fields are ignored.
    /// - `cancel`: cancellation token; the tool should stop as soon as possible when a cancellation signal is received.
    /// - `on_update`: Optional callback for streaming partial execution results.
    ///
    /// # Errors
    /// Returns `Err(ToolError)` to indicate tool execution failure. The error message is converted into the error content of a `ToolResultMessage`.
    async fn execute(
        &self,
        tool_call_id: &str,
        params: &serde_json::Value,
        ctx: &ToolContext,
        cancel: &CancellationToken,
        on_update: ToolUpdateCallback,
    ) -> Result<ToolExecutionResult, ToolError>;

    /// Serialize into an API-usable tool definition (sent to the LLM).
    fn to_api_definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters().clone(),
        }
    }
}
