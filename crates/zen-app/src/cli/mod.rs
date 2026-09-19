pub mod args;
pub mod file_processor;
pub mod initial_message;
pub mod session_picker;

pub use args::{parse_thinking_level, CliArgs, Mode};
pub use file_processor::{process_file_arguments, FileProcessError, ProcessedFiles};
pub use initial_message::{
    build_initial_message, read_piped_stdin, stdin_is_terminal, InitialMessageResult,
};
pub use session_picker::{select_session, SessionPickerError};
