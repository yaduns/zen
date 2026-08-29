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
/// Skip condition: automatically skipped when the API key env var for the selected default provider is not set
/// (detected inside the test function body rather than using the #[ignore] attribute).
#[ignore]
#[tokio::test]
async fn test_simple_chat() {
    // Check the corresponding API key for the default provider (ZEN_PROVIDER, default opencode-go)
    let provider = zen_ai::provider::default_provider_name();
    let key_var = match provider.as_str() {
        "opencode-go" => "OPENCODE_API_KEY",
        _ => "ARK_API_KEY",
    };
    let _ = std::env::var(key_var)
        .unwrap_or_else(|_| panic!("{} not set; skipping integration test", key_var));

    // 1. Create the model from environment variables (constructed for the default provider)
    let model = Model::from_env();

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
