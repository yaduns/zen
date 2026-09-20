pub mod args;
pub mod file_processor;
pub mod initial_message;
pub mod session_picker;
pub mod startup_ui;

pub use args::{parse_thinking_level, CliArgs, Mode};
pub use file_processor::{process_file_arguments, FileProcessError, ProcessedFiles};
pub use initial_message::{
    build_initial_message, read_piped_stdin, stdin_is_terminal, InitialMessageResult,
};
pub use session_picker::{select_session, SessionPickerError};
pub use startup_ui::{
    should_run_first_time_setup, show_first_time_setup, show_startup_selector, StartupUiError,
};
