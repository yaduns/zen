//! This module assembles "data to send to the LLM" into `Context`: `create_context` (system_prompt + messages + tools)
//! and `tools_to_definitions` (Tool → ToolDefinition). Together with `system_prompt` and `prompt_templates`, it belongs to
//! the harness domain (assembling data sent to the LLM), separate from `agent/` (loop/state machine runtime orchestration).

pub mod context;
pub mod system_prompt;
