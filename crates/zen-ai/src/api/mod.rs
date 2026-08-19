pub mod openai_stream;

use crate::provider::Api;
use crate::types::StreamFn;

/// Gets the stream implementation bound to a protocol; returns `None` for unimplemented or unregistered protocols.
///
/// # Behavior
/// Exhaustively matches on `api` and returns the `StreamFn` constructor result for the protocol. `OpenAiCompletions`
/// Returns `Some(openai_completions_stream())`; `OpenAiResponses` returns `None` (reserved).
/// The caller should raise a "protocol not implemented" error when receiving `None` (the error type is to be designed during implementation).
///
pub fn stream_for(api: Api) -> Option<StreamFn> {
    match api {
        Api::OpenAiCompletions => Some(openai_stream::openai_completions_stream()),
        Api::OpenAiResponses => None, // Reserved for Phase 2 implementation.
    }
}
