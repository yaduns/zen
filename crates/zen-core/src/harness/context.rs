use std::sync::Arc;
use zen_ai::tool::{Tool, ToolContext};
use zen_ai::types::{Context, Message, ToolDefinition};

/// Assemble the agent context (Context).
///
/// Assemble system_prompt, messages, and tools into a `Context` instance.
/// Each `Tool` in `tools` is converted to a `ToolDefinition` via `Tool::to_api_definition()`
/// so it can be serialized and sent to the LLM.
///
/// This function assembles the entry arguments for `run_agent_loop` and `run_agent_loop_continue`,
/// for use by callers (zen-app) before starting the agent loop.
///
/// # Parameters
/// - `system_prompt`: system prompt text
/// - `messages`: message history (from the session tree; the caller has already filtered it via `AgentLoopHooks::filter_messages`)
/// - `tools`: list of available tools (runtime tools implementing the `Tool` trait)
///
/// # Return Value
/// Returns a fully assembled `Context`, ready for `run_agent_loop` or `run_agent_loop_continue`.
///
/// # Edge Cases
/// - If `system_prompt` is an empty string, `Context.system_prompt` is `Some("")` (an empty system prompt is still a valid prompt).
/// - If `tools` is an empty slice, `Context.tools` is `Some(vec![])` (empty tool list; the LLM will not call any tools).
/// - This function does not modify `messages` — it assumes the caller has preprocessed the message list via filtering logic.
/// - This function is pure construction; it performs no IO or async operations.
pub fn create_context(
    system_prompt: String,
    messages: Vec<Message>,
    tools: &[Arc<dyn Tool>],
) -> Context {
    Context {
        system_prompt: Some(system_prompt),
        messages,
        tools: Some(tools_to_definitions(tools)),
    }
}

/// Converts a slice of `Tool` trait objects into a list of `ToolDefinition`s.
///
/// Used internally by `create_context`; also exposed separately for other scenarios (e.g. tool list preview).
///
/// # Parameters
/// - `tools`: runtime tool list
///
/// # Return Value
/// Serializes the tool definitions; each tool's name, description, and parameter schema come from `Tool::to_api_definition()`.
///
/// # Edge Cases
/// - Returns an empty vec if `tools` is an empty slice.
/// - The output order matches the input order.
pub fn tools_to_definitions(tools: &[Arc<dyn Tool>]) -> Vec<ToolDefinition> {
    tools.iter().map(|tool| tool.to_api_definition()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::json;
    use zen_ai::tool::{ToolError, ToolExecutionResult, ToolUpdateCallback};
    use zen_ai::types::{ContentBlockList, UserMessage};

    // ─── MockTool ────────────────────────────────────────────────────────

    #[derive(Debug)]
    struct MockTool {
        name: String,
        description: String,
        params: serde_json::Value,
        label: String,
    }

    impl MockTool {
        fn new(name: &str, description: &str, params: serde_json::Value) -> Self {
            Self {
                name: name.to_string(),
                description: description.to_string(),
                params,
                label: name.to_string(),
            }
        }
    }

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            &self.description
        }
        fn parameters(&self) -> &serde_json::Value {
            &self.params
        }
        fn label(&self) -> &str {
            &self.label
        }
        async fn execute(
            &self,
            _tool_call_id: &str,
            _params: &serde_json::Value,
            _ctx: &ToolContext,
            _cancel: &tokio_util::sync::CancellationToken,
            _on_update: ToolUpdateCallback,
        ) -> Result<ToolExecutionResult, ToolError> {
            unimplemented!("not used in these tests")
        }
    }

    // ─── helpers ────────────────────────────────────────────────────────────

    fn dummy_message() -> Message {
        Message::User(UserMessage {
            content: ContentBlockList::Text("hello".into()),
            timestamp: 1000,
        })
    }

    fn make_tool(name: &str, params: serde_json::Value) -> Arc<dyn Tool> {
        Arc::new(MockTool::new(name, "A test tool", params))
    }

    // ─── tools_to_definitions ──────────────────────────────────────────────

    #[test]
    fn test_tools_to_definitions_empty() {
        let result = tools_to_definitions(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_tools_to_definitions_single_tool() {
        let params = json!({"type": "object", "properties": {"x": {"type": "string"}}});
        let tools = [make_tool("bash", params.clone())];
        let result = tools_to_definitions(&tools);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function.name, "bash");
        assert_eq!(result[0].function.description, "A test tool");
        assert_eq!(result[0].function.parameters, params);
    }

    #[test]
    fn test_tools_to_definitions_preserves_order() {
        let tools = [
            make_tool("bash", json!({})),
            make_tool("read", json!({})),
            make_tool("write", json!({})),
        ];
        let result = tools_to_definitions(&tools);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].function.name, "bash");
        assert_eq!(result[1].function.name, "read");
        assert_eq!(result[2].function.name, "write");
    }

    #[test]
    fn test_tools_to_definitions_maps_parameters_correctly() {
        let params_a = json!({"type": "object", "properties": {"path": {"type": "string"}}});
        let params_b = json!({"type": "object", "properties": {"code": {"type": "string"}}});

        let tools = [
            make_tool("read", params_a.clone()),
            make_tool("edit", params_b.clone()),
        ];
        let result = tools_to_definitions(&tools);

        assert_eq!(result[0].function.name, "read");
        assert_eq!(result[0].function.parameters, params_a);
        assert_eq!(result[1].function.name, "edit");
        assert_eq!(result[1].function.parameters, params_b);
    }

    // ─── create_context ────────────────────────────────────────────────────

    #[test]
    fn test_create_context_normal() {
        let tools = [make_tool("bash", json!({"type": "object"}))];
        let msgs = vec![dummy_message()];

        let ctx = create_context("You are a helpful assistant.".into(), msgs.clone(), &tools);

        assert_eq!(
            ctx.system_prompt,
            Some("You are a helpful assistant.".to_string())
        );
        assert_eq!(ctx.messages, msgs);
        assert!(ctx.tools.is_some());
        assert_eq!(ctx.tools.as_ref().unwrap().len(), 1);
        assert_eq!(ctx.tools.as_ref().unwrap()[0].function.name, "bash");
    }

    #[test]
    fn test_create_context_empty_system_prompt() {
        let tools = [make_tool("bash", json!({}))];
        let ctx = create_context(String::new(), vec![dummy_message()], &tools);

        assert_eq!(ctx.system_prompt, Some(String::new()));
    }

    #[test]
    fn test_create_context_empty_tools() {
        let ctx = create_context("prompt".into(), vec![dummy_message()], &[]);

        assert!(ctx.tools.is_some());
        assert!(ctx.tools.as_ref().unwrap().is_empty());
    }

    #[test]
    fn test_create_context_empty_messages() {
        let tools = [make_tool("bash", json!({}))];
        let ctx = create_context("prompt".into(), vec![], &tools);

        assert!(ctx.messages.is_empty());
        assert_eq!(ctx.tools.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_create_context_preserves_message_order() {
        let msg1 = dummy_message();
        let msg2 = Message::User(UserMessage {
            content: ContentBlockList::Text("world".into()),
            timestamp: 2000,
        });
        let msgs = vec![msg1, msg2];

        let ctx = create_context("prompt".into(), msgs, &[]);

        assert_eq!(ctx.messages.len(), 2);
        match &ctx.messages[0] {
            Message::User(u) => assert_eq!(u.content, ContentBlockList::Text("hello".into())),
            _ => panic!("expected User message"),
        }
        match &ctx.messages[1] {
            Message::User(u) => assert_eq!(u.content, ContentBlockList::Text("world".into())),
            _ => panic!("expected User message"),
        }
    }

    #[test]
    fn test_create_context_multiple_tools() {
        let tools = [
            make_tool("bash", json!({})),
            make_tool("read", json!({})),
            make_tool("edit", json!({})),
        ];
        let ctx = create_context("prompt".into(), vec![], &tools);

        let defs = ctx.tools.unwrap();
        assert_eq!(defs.len(), 3);
        assert_eq!(defs[0].function.name, "bash");
        assert_eq!(defs[1].function.name, "read");
        assert_eq!(defs[2].function.name, "edit");
    }

    #[test]
    fn test_create_context_does_not_mutate_input_slice() {
        // tools is &[Arc<dyn Tool>]; create_context must not modify the tool list
        let tool = make_tool("bash", json!({}));
        let tools = vec![tool];
        let tools_slice: &[Arc<dyn Tool>] = &tools;

        let ctx = create_context("prompt".into(), vec![dummy_message()], tools_slice);

        // tool list is not consumed
        assert_eq!(tools.len(), 1);
        assert_eq!(ctx.tools.as_ref().unwrap().len(), 1);
        assert_eq!(ctx.tools.as_ref().unwrap()[0].function.name, "bash");
    }
}
