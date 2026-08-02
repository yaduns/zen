use crate::agent::types::StreamFn;
use std::sync::RwLock;
use thiserror;

/// Module-level global default stream function.
/// Set by `set_default_stream_fn` and read by `get_default_stream_fn`.
/// `RwLock::new` is a const fn, so it can initialize a `static` directly.
/// `RwLock` is chosen over `Mutex` because semantically this is "many reads, few writes" (the setter is called only once at startup,
/// and the getter is called frequently in the agent loop); `RwLock` allows multiple reads without blocking.
static DEFAULT_STREAM_FN: RwLock<Option<StreamFn>> = RwLock::new(None);

/// Configures the default LLM streaming function used by the agent loop.
/// When the caller does not explicitly pass `stream_fn`, the agent loop and underlying loop use this default function.
/// Afterwards, the agent loop obtains and uses it directly via `get_default_stream_fn()`.
///
/// # Parameters
/// - `stream_fn`: default stream function to set. Pass `None` to clear the current configuration,
///   subsequent calls to `get_default_stream_fn()` will return `Err(NoDefaultStreamFnError)`.
///
/// # Thread Safety
/// Uses `RwLock` for internal synchronization and can be safely called from multiple threads.
pub fn set_default_stream_fn(stream_fn: Option<StreamFn>) {
    // RwLock::write() returns RwLockWriteGuard<Option<StreamFn>>, which is a smart pointer
    // so *guard dereferences to &mut Option<StreamFn>, i.e. a mutable reference to the inner value
    *DEFAULT_STREAM_FN.write().unwrap() = stream_fn;
}

/// Gets the default LLM streaming function.
///
/// Returns the currently configured default stream function. If not configured via `set_default_stream_fn`,
/// returns `Err(NoDefaultStreamFnError)`.
///
/// # Return Value
/// - `Ok(StreamFn)`: currently configured default stream function (`Arc` clone, low cost).
/// - `Err(NoDefaultStreamFnError)`: no default stream function configured.
///
/// # Thread Safety
/// Uses `RwLock` for internal synchronization and can be safely called from multiple threads.
pub fn get_default_stream_fn() -> Result<StreamFn, NoDefaultStreamFnError> {
    DEFAULT_STREAM_FN
        .read()
        .unwrap()
        .clone()
        .ok_or(NoDefaultStreamFnError)
}

/// Error returned when `get_default_stream_fn()` is called but no default stream function is configured.
#[derive(Debug, Clone, thiserror::Error)]
#[error("No default stream function configured. Pass streamFn explicitly or call set_default_stream_fn().")]
pub struct NoDefaultStreamFnError;

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::Mutex;
    use tokio::sync::mpsc::UnboundedReceiver;
    use tokio_util::sync::CancellationToken;
    use zen_ai::api::openai_stream::StreamConfig;
    use zen_ai::api::openai_stream::StreamError;
    use zen_ai::types::AssistantMessageEvent;
    use zen_ai::types::Context;
    use zen_ai::types::Model;

    /// Test serialization lock to prevent parallel tests from interfering via the shared global `DEFAULT_STREAM_FN`.
    /// Each test function holds the lock for the lifetime of `_lock` and releases it automatically on exit.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Creates a mock StreamFn that does not perform real LLM calls.
    /// The returned stream is closed immediately (no longer emits events), suitable for tests that only verify set/get semantics.
    fn mock_stream_fn() -> StreamFn {
        Arc::new(
            |_: &Model,
             _: &Context,
             _: Option<StreamConfig>,
             _: Option<CancellationToken>| {
                Box::pin(async {
                    let (_tx, rx) = tokio::sync::mpsc::unbounded_channel::<AssistantMessageEvent>();
                    Ok(rx)
                })
            },
        )
    }

    /// Verifies that calling `get_default_stream_fn` when unconfigured returns an error.
    #[test]
    fn test_get_before_set_returns_error() {
        let _lock = TEST_LOCK.lock().unwrap();
        set_default_stream_fn(None);

        let result = get_default_stream_fn();
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "No default stream function configured. Pass streamFn explicitly or call set_default_stream_fn()."
        );
    }

    /// Verifies that after setting, calling `get_default_stream_fn` returns the same Arc.
    #[test]
    fn test_set_and_get_returns_same_arc() {
        let _lock = TEST_LOCK.lock().unwrap();
        set_default_stream_fn(None);

        let fn1 = mock_stream_fn();
        set_default_stream_fn(Some(fn1.clone()));

        let result = get_default_stream_fn();
        assert!(result.is_ok());
        assert!(Arc::ptr_eq(&fn1, &result.unwrap()));
    }

    /// Verifies that setting `None` clears the configured default stream function.
    #[test]
    fn test_set_none_clears_default() {
        let _lock = TEST_LOCK.lock().unwrap();
        set_default_stream_fn(None);

        set_default_stream_fn(Some(mock_stream_fn()));
        assert!(get_default_stream_fn().is_ok());

        set_default_stream_fn(None);
        assert!(get_default_stream_fn().is_err());
    }

    /// Verifies that setting multiple times replaces the previous value.
    #[test]
    fn test_set_twice_replaces_old() {
        let _lock = TEST_LOCK.lock().unwrap();
        set_default_stream_fn(None);

        let fn1 = mock_stream_fn();
        let fn2 = mock_stream_fn();

        set_default_stream_fn(Some(fn1));
        set_default_stream_fn(Some(fn2.clone()));

        let result = get_default_stream_fn().unwrap();
        assert!(Arc::ptr_eq(&fn2, &result));
    }

    /// Verifies the Display implementation of `NoDefaultStreamFnError`.
    #[test]
    fn test_error_display() {
        let err = NoDefaultStreamFnError;
        assert_eq!(
            err.to_string(),
            "No default stream function configured. Pass streamFn explicitly or call set_default_stream_fn()."
        );
    }

    /// Verifies that `NoDefaultStreamFnError` is Clone.
    #[test]
    fn test_error_clone() {
        let err = NoDefaultStreamFnError;
        let cloned = err.clone();
        assert_eq!(err.to_string(), cloned.to_string());
    }

    /// Verifies that `NoDefaultStreamFnError` is Debug.
    #[test]
    fn test_error_debug() {
        let err = NoDefaultStreamFnError;
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("NoDefaultStreamFnError"));
    }
}
