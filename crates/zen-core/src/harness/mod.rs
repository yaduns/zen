//! `harness` is the unified module for "everything in the agent except the model": the stateful engine (`engine`, Step B),
//! Stateless low-level loop (`agent_loop`), state machine (`state_machine`), hooks (`agent_hooks`),
//! Public types (`types`), and the assembly of "data to send to the LLM" into `Context`
//! `context` / `system_prompt`. This module is no longer separated from "runtime orchestration" — runtime orchestration is part of the harness.

pub mod agent_hooks;
pub mod agent_loop;
pub mod context;
pub mod state_machine;
pub mod system_prompt;
pub mod types;
