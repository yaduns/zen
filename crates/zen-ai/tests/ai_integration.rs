// In tests/ai_integration.rs: integration test

use zen_ai::api::openai_stream::{stream_collect, StreamConfig, StreamError};
use zen_ai::types::{
    ContentBlock, ContentBlockList, Context, Message, Model, StopReason, UserMessage,
};

/// End-to-end LLM call test.
///
/// Reads environment variables, builds a Context, calls stream_collect(), and validates the returned AssistantMessage
/// contains non-empty content and valid usage/stop_reason.
///
/// Skip condition: automatically skipped when the OPENAI_API_KEY environment variable is not set (checked inside the test function body,
/// rather than using the #[ignore] attribute, since the latter cannot be decided at compile time and must be determined at runtime).
#[tokio::test]
async fn test_simple_chat() {
    // Check the environment variable, skip if not set
    let _ =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set; skipping integration test");

    // 1. Create the model from environment variables
    let model = Model::from_env().with_env_overrides();

    // 2. Build a simple Context
    let context = Context {
        system_prompt: Some("You are a helpful assistant. Answer concisely.".to_string()),
        messages: vec![Message::User(UserMessage {
            content: ContentBlockList::Text("Where are you? Answer in Chinese".to_string()),
            timestamp: chrono::Utc::now().timestamp_millis(),
        })],
        tools: None,
    };

    // 3. Call stream_collect
    let result = stream_collect(&model, &context, None, None).await;

    // 4. Validate the result
    let message = result.expect("stream_collect should succeed");
    assert!(!message.content.is_empty(), "response should have content");
    assert!(
        matches!(message.stop_reason, StopReason::Stop | StopReason::Length),
        "stop_reason should be Stop or Length, got {:?}",
        message.stop_reason
    );
    // Print the response content (for manual verification)
    for block in &message.content {
        if let ContentBlock::Text(t) = block {
            println!("Response: {}", t.text);
        }
    }
    println!(
        "Usage: {} in / {} out / {} cached",
        message.usage.input, message.usage.output, message.usage.cache_read
    );
}
