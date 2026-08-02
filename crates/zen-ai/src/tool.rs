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
    /// - `cancel`: cancellation token; the tool should stop as soon as possible when a cancellation signal is received.
    /// - `on_update`: Optional callback for streaming partial execution results.
    ///
    /// # Errors
    /// Returns `Err(ToolError)` to indicate tool execution failure. The error message is converted into the error content of a `ToolResultMessage`.
    async fn execute(
        &self,
        tool_call_id: &str,
        params: &serde_json::Value,
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
