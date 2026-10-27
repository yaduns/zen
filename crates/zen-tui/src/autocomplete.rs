// REQ-TUI-03 (pi-401): Autocomplete — context detection, candidate generation, and text application for the input line (pure logic, no terminal interaction).
// Corresponding original TS: reference/pi-original/packages/tui/src/autocomplete.ts.
// Differences from pi:
// - `fd` external binary → `ignore` crate in-process recursive traversal (at-gitignore-fix: same source as fd/ripgrep,
//   respects .gitignore/.ignore/.git/info/exclude + `--hidden` + `--follow`, matching pi fd default behavior;
//   excludes the entire `.git` tree);
// - `fuzzyFilter` (pi-402 / T-TU-03-4) is wired in: command-name filtering is implemented via `fuzzy::fuzzy_filter`
//   (aligned with pi `fuzzyFilter(commandItems, prefix, i => i.name)`), pub API unchanged;
// - `cursor_col` conventionally holds a character (char) index: the assembly layer (T-AP-11-2) converts the Editor's grapheme column to a character column before passing it in;
//   for plain text the two coincide; converting edge-case text containing combining characters is the assembly layer's responsibility;
// - Home directory resolution uses the workspace's existing `home` crate (aligned with the zen-core/src/paths.rs precedent); the originally planned `dirs` dependency from the task doc is no longer introduced.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use tokio_util::sync::CancellationToken;

/// Completion token delimiters: context-splitting boundaries for paths/attachments (aligned with pi `PATH_DELIMITERS`).
const PATH_DELIMITERS: [char; 5] = [' ', '\t', '"', '\'', '='];

/// One completion candidate — `value` is the full text applied to the line (including @/quotes/trailing / modifiers),
/// `label` is the brief text shown in the dropdown (directories carry a trailing /), `description` is an optional secondary note (for @ completion it shows the full relative path).
/// All three views share one source; the application layer renders whichever it needs.
#[derive(Debug, Clone, PartialEq)]
pub struct AutocompleteItem {
    value: String, // Full completion text applied to the line (including @/quote prefix modifiers; directories end with /)
    label: String, // Dropdown display text: basename + ("/" if directory)
    description: Option<String>, // For @ completion: the full relative display path; otherwise none
}

impl AutocompleteItem {
    /// Construct one completion candidate from three-view data: `value` applied to the line, `label` shown in the dropdown, `description` as optional secondary note.
    /// Consumers: this module's candidate generation (get_file_suggestions / get_fuzzy_file_suggestions / filter_commands),
    /// T-AP-03-2 (constructed when the parameter-completion callback returns candidates).
    pub fn new(
        value: impl Into<String>,
        label: impl Into<String>,
        description: Option<String>,
    ) -> Self {
        Self {
            value: value.into(),
            label: label.into(),
            description,
        }
    }

    /// Get the full completion text applied to the line (including @/quote modifiers; directories end with /).
    /// Consumers: T-AP-11-2's dropdown rendering and quote-dedup decision on apply (.ends_with('"')).
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Get the dropdown display text (name + trailing / for directories).
    /// Consumer: T-AP-11-2 renders the dropdown; directory detection for apply (`.ends_with('/')`).
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Secondary note (None means none).
    /// Consumer: T-AP-11-2 dropdown rendering.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

/// Future returned by the argument-completion callback: resolves to "no candidates (None) or a candidate list"; `+ Send` allows awaiting across threads in async contexts.
/// Lifetime anchor: the future is `'static`—the implementor clones the inputs inside the closure (the `argument_prefix` string,
/// the `CancellationToken` itself is cheaply Cloneable) before constructing the future, avoiding HRTB complexity from borrowing inputs.
pub type ArgumentCompletionFuture =
    Pin<Box<dyn Future<Output = Option<Vec<AutocompleteItem>>> + Send>>;

/// Argument-completion callback of `SlashCommand`: takes the argument segment after the cursor and a cancellation token, returns a boxed future.
/// The `Fn` family must spell out the full parenthesized parameter list (Rust requires `Fn` to take explicit generic parameters, see #29625);
/// `+ Send + Sync` goes after the entire trait path.
pub type ArgumentCompletionCallback =
    Box<dyn Fn(&str, &CancellationToken) -> ArgumentCompletionFuture + Send + Sync>;

/// In-process registry entry for a completable slash command—name participates in command name matching,
/// description and argument_hint are joined into the dropdown hint (if hint exists `${hint} — ${desc}`, if only hint then hint),
/// get_argument_completions is called when the cursor is after `/command ` to produce argument candidates (returns None = no argument completion).
/// Constructed and injected by zen-app's command system (T-AP-03-2 and its upstream); this module only consumes.
pub struct SlashCommand {
    name: String,                                                 // Command name (without /)
    description: Option<String>,                                  // Main body of the dropdown hint
    argument_hint: Option<String>, // Joined with description for display
    get_argument_completions: Option<ArgumentCompletionCallback>, // F(argument_prefix, ct) -> ArgumentCompletionFuture; None = no argument completion
}

impl SlashCommand {
    // Build a completable command entry from registry data.
    /// When `get_argument_completions` is None, the command only completes its name and yields no candidates for the argument segment.
    /// Consumer: T-AP-03-2 (constructed when the command system registers commands; slack command skills etc. plug in through this contract).
    pub fn new(
        name: impl Into<String>,
        description: Option<String>,
        argument_hint: Option<String>,
        get_argument_completions: Option<ArgumentCompletionCallback>,
    ) -> Self {
        Self {
            name: name.into(),
            description,
            argument_hint,
            get_argument_completions,
        }
    }
}

/// Response envelope for get_suggestions—items is the candidate list, prefix is the text segment being matched/will be replaced (e.g. /, src/, @"my folder/).
/// apply_completion cuts len(prefix) chars before the cursor, then inserts item.value.
#[derive(Debug, Clone, PartialEq)]
pub struct AutocompleteSuggestions {
    items: Vec<AutocompleteItem>, // Candidates (already sorted: directories first, alphabetical)
    prefix: String,               // Original text segment before the cursor that gets replaced
}

impl AutocompleteSuggestions {
    /// Builds one completion response from the candidate list + the replaced segment.
    /// Consumers: this module's get_suggestions; the wrapper provider for T-AP-11-2 (the autocompleteProviderWrappers chain).
    pub fn new(items: Vec<AutocompleteItem>, prefix: impl Into<String>) -> Self {
        Self {
            items,
            prefix: prefix.into(),
        }
    }

    /// Gets the candidate list (already sorted: directories first, alphabetical).
    /// Consumer: T-AP-11-2 dropdown rendering.
    pub fn items(&self) -> &[AutocompleteItem] {
        &self.items
    }

    /// Gets the original text segment that was matched / will be replaced.
    /// Consumers: T-AP-11-2 passes it to apply_completion after a selection.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
}

/// Result of apply_completion — the replaced line set and the new cursor position.
#[derive(Debug, Clone, PartialEq)]
pub struct AppliedCompletion {
    lines: Vec<String>, // Replaced line set (only the cursor line is rewritten)
    cursor_line: usize, // New cursor row (usually unchanged)
    cursor_col: usize,  // New cursor column (character index)
}

impl AppliedCompletion {
    /// Gets the replaced line set.
    /// Consumers: T-AP-11-2 writes it back to the editor state.
    pub fn lines(&self) -> &[String] {
        &self.lines
    }

    /// Gets the new cursor position (row, column) — column unit is character index.
    /// Consumers: T-AP-11-2 writes it back to the editor cursor (converted to grapheme columns).
    pub fn cursor_position(&self) -> (usize, usize) {
        (self.cursor_line, self.cursor_col)
    }
}

/// Input completion contract: interactive mode (T-AP-11-2) calls it when the user types a trigger character or presses Tab,
/// gets candidates or None (no candidates) through this trait, and after a selection calls `apply_completion` to write the completed text back to the input line.
/// Abstracted from pi's `AutocompleteProvider`; implementations may be `CombinedAutocompleteProvider` or an app-level wrapper provider
/// (aligning with pi's `autocompleteProviderWrappers` wrapper chain).
/// Requires `Send + Sync`: the provider is stored in the editor slot and invoked in async contexts.
#[async_trait::async_trait]
pub trait AutocompleteProvider: Send + Sync {
    /// Typing these characters naturally triggers completion (e.g. `@`): handled by the application layer (T-AP-11-2) in the character-input branch
    /// match each in turn; on a hit, call `get_suggestions` so completion candidates surface while typing without Tab.
    /// No trigger characters by default; implementations return based on their own prefix semantics (slash `/` triggering is determined separately by the editor's slash context and is not listed here).
    fn trigger_characters(&self) -> &[char] {
        &[]
    }

    /// Recognize the completion context before the cursor and return candidates; return `None` when there are no candidates or the context doesn't match, so the caller dismisses the dropdown without disturbing input.
    /// `lines`/`cursor_line`/`cursor_col` are the current input lines and cursor (`cursor_col` is a character index,
    /// converted from the Editor's grapheme column by the assembly layer); `force` is true when Tab forces a trigger (produce candidates even without an explicit context);
    /// `ct` is a cancellation token; long-running steps like directory walking must honor cancellation and immediately return `None` (aligned with pi signal.aborted).
    async fn get_suggestions(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        force: bool,
        ct: &tokio_util::sync::CancellationToken,
    ) -> Option<AutocompleteSuggestions>;

    /// Apply the selected `item` to the lines: cut the matched segment of length `prefix` from before the cursor and insert `item.value`,
    /// and handle the trailing semantics for each of the four contexts — append a space after command-name completion; do **not** add a space after `@` attachment-directory completion
    /// (a trailing `/` continues completion); command arguments/file paths replace the whole segment; in the quote case, dedupe a closing quote already present after the cursor.
    /// Pure function: does not mutate its inputs, returns new lines and cursor (`AppliedCompletion`). `cursor_col` is a character index.
    fn apply_completion(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        item: &AutocompleteItem,
        prefix: &str,
    ) -> AppliedCompletion;

    /// Whether Tab file completion should trigger: return false for a line-leading slash command (`/cmd` with no space, cursor within the command name),
    /// keeping Tab on command-name matching instead of accidentally entering path completion; true otherwise.
    /// Called from the Tab key branch of T-AP-11-2 for discrimination.
    fn should_trigger_file_completion(
        &self,
        _lines: &[String],
        _cursor_line: usize,
        _cursor_col: usize,
    ) -> bool {
        true
    }
}

/// Aggregate completion provider: constructed at session startup (T-AP-11-2) from the command list + cwd and injected into the editor; the zen-app command system (T-AP-03-2)
/// feeds its command registry into `commands` per its `SlashCommand` contract. Internally combines four candidate sources (command names / command arguments / `@` attachments / plain paths),
/// so a single completion session covers the three user needs — slash commands, attachments, and paths — without the caller having to determine the context.
/// Consumers: T-AP-03-2 / T-AP-11-2 (no in-crate Rust consumers currently; reserved for interactive mode).
pub struct CombinedAutocompleteProvider {
    commands: Vec<SlashCommand>,
    base_path: std::path::PathBuf,
    fuzzy_walk_enabled: bool,
}

impl CombinedAutocompleteProvider {
    /// Construct the aggregate provider: `commands` is the set of completable command entries (name/description/parameter callbacks),
    /// `base_path` is the relative-path resolution base (session cwd); when `fuzzy_walk_enabled=false`, `@` fuzzy file completion is disabled
    /// (aligned with pi `fdPath: null` semantics: without fd, `@` completion is off).
    /// Consumer: T-AP-11-2 (interactive mode assembly).
    pub fn new(
        commands: Vec<SlashCommand>,
        base_path: std::path::PathBuf,
        fuzzy_walk_enabled: bool,
    ) -> Self {
        Self {
            commands,
            base_path,
            fuzzy_walk_enabled,
        }
    }
}

#[async_trait::async_trait]
impl AutocompleteProvider for CombinedAutocompleteProvider {
    fn trigger_characters(&self) -> &[char] {
        // Natural trigger characters (pi `["@", "#"]`; `#` omitted in v1; at-attachment-wiring-fix)
        &['@']
    }

    async fn get_suggestions(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        force: bool,
        ct: &CancellationToken,
    ) -> Option<AutocompleteSuggestions> {
        if lines.is_empty() {
            return None;
        }
        let cursor_line = cursor_line.min(lines.len() - 1);
        let current_line = &lines[cursor_line];
        let byte_col = byte_index_of(current_line, cursor_col);
        let text_before_cursor = &current_line[..byte_col];

        // 1) `@` attachment → fuzzy file candidates (aligned with pi extractAtPrefix branch)
        if let Some(at_prefix) = extract_at_prefix(text_before_cursor) {
            if !self.fuzzy_walk_enabled {
                return None;
            }
            let pp = parse_path_prefix(&at_prefix);
            let items = get_fuzzy_file_suggestions(
                &pp.raw_prefix,
                pp.is_quoted_prefix,
                &self.base_path,
                ct,
            )
            .await;
            if items.is_empty() {
                return None;
            }
            return Some(AutocompleteSuggestions::new(items, at_prefix));
        }

        // 2) leading-slash commands (enter the command branch only on natural trigger; on forced Tab hand off to path completion, aligned with pi)
        if !force && text_before_cursor.starts_with('/') {
            if let Some(space_index) = text_before_cursor.find(' ') {
                // command argument completion: invoke the argument callback for that command
                let command_name = &text_before_cursor[1..space_index];
                let argument_text = &text_before_cursor[space_index + 1..];
                let command = self.commands.iter().find(|c| c.name == command_name);
                match command.and_then(|c| c.get_argument_completions.as_ref()) {
                    Some(callback) => match callback(argument_text, ct).await {
                        Some(items) if !items.is_empty() => {
                            return Some(AutocompleteSuggestions::new(
                                items,
                                argument_text.to_string(),
                            ));
                        }
                        _ => return None,
                    },
                    None => return None,
                }
            } else {
                // command name completion
                let prefix = &text_before_cursor[1..];
                let items = filter_commands(&self.commands, prefix);
                if items.is_empty() {
                    return None;
                }
                return Some(AutocompleteSuggestions::new(
                    items,
                    text_before_cursor.to_string(),
                ));
            }
        }

        // 3) path completion (natural trigger: segment contains /, starts with . or ~/, or an empty segment after a space; under force, any segment yields candidates)
        let path_match = extract_path_prefix(text_before_cursor, force)?;
        let items = get_file_suggestions(&path_match, &self.base_path);
        if items.is_empty() {
            return None;
        }
        Some(AutocompleteSuggestions::new(items, path_match))
    }

    fn apply_completion(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        item: &AutocompleteItem,
        prefix: &str,
    ) -> AppliedCompletion {
        if lines.is_empty() {
            return AppliedCompletion {
                lines: lines.to_vec(),
                cursor_line: 0,
                cursor_col: 0,
            };
        }
        let cursor_line = cursor_line.min(lines.len() - 1);
        let line = &lines[cursor_line];
        let char_count = line.chars().count();
        let cursor_col = cursor_col.min(char_count);
        // text before the replaced segment (character-index split); prefix is a suffix of text_before_cursor, so subtraction is safe
        let before_chars = cursor_col.saturating_sub(prefix.chars().count());
        let before: String = line.chars().take(before_chars).collect();
        let after: String = line.chars().skip(cursor_col).collect();

        // quoted case: value already carries its closing quote and a closing quote exists after the cursor → dedupe (aligned with pi adjustedAfterCursor)
        let is_quoted_prefix = prefix.starts_with('"') || prefix.starts_with("@\"");
        let has_leading_quote_after_cursor = after.starts_with('"');
        let adjusted_after =
            if is_quoted_prefix && item.value().ends_with('"') && has_leading_quote_after_cursor {
                &after[1..] // '"' is 1-byte ASCII, safe
            } else {
                &after
            };

        let value_len = item.value().chars().count();
        let is_directory = item.label().ends_with('/');
        let has_trailing_quote = item.value().ends_with('"');
        let cursor_offset = if is_directory && has_trailing_quote {
            value_len - 1
        } else {
            value_len
        };

        // replacement semantics of the four branches (for directory completion the cursor stays after the trailing separator / before the closing quote, to continue completion)
        let (new_line, new_col) =
            if prefix.starts_with('/') && before.trim().is_empty() && !prefix[1..].contains('/') {
                // command name completion: append a trailing space
                let new_line = format!("{before}/{} {adjusted_after}", item.value());
                (new_line, before_chars + value_len + 2)
            } else if prefix.starts_with('@') {
                // `@` attachment: directory completion adds no space (trailing / continues completion), non-directory adds a space
                let suffix = if is_directory { "" } else { " " };
                let new_line = format!("{before}{}{suffix}{adjusted_after}", item.value());
                (new_line, before_chars + cursor_offset + suffix.len())
            } else {
                // command argument / file path: replace the whole segment
                let new_line = format!("{before}{}{adjusted_after}", item.value());
                (new_line, before_chars + cursor_offset)
            };

        let mut new_lines = lines.to_vec();
        new_lines[cursor_line] = new_line;
        AppliedCompletion {
            lines: new_lines,
            cursor_line,
            cursor_col: new_col,
        }
    }

    fn should_trigger_file_completion(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
    ) -> bool {
        match lines.get(cursor_line) {
            Some(current_line) => {
                let text_before_cursor: String = {
                    let byte_col = byte_index_of(current_line, cursor_col);
                    current_line[..byte_col].to_string()
                };
                let trimmed = text_before_cursor.trim();
                !(trimmed.starts_with('/') && !trimmed.contains(' '))
            }
            None => true,
        }
    }
}

// ────────────────────────────────────────────────────────────────
// module-private struct (structured result returned by a helper, not part of the pub API surface)
// ────────────────────────────────────────────────────────────────

/// parse_path_prefix result — raw_prefix is the actual path fragment with `@`/quote decorations stripped;
/// the two bool flags drive later candidate wrapping (build_completion_value) to decide the decoration form of the completion text.
struct PathPrefix {
    raw_prefix: String,
    is_at_prefix: bool,
    is_quoted_prefix: bool,
}

/// resolve_scoped_fuzzy_query output — base_dir is the actual walk start point, query is the remaining match text within the segment,
/// display_base is used to echo walk results as relative paths.
struct ScopedQuery {
    base_dir: PathBuf,
    query: String,
    display_base: String,
}

/// A single result of walk_directory — `path` is the display path relative to base_dir (`/`-separated, directories ending with `/`),
/// with `is_directory` reused for scoring and candidate wrapping.
struct FuzzyEntry {
    path: String,
    is_directory: bool,
}

// ────────────────────────────────────────────────────────────────
// Module-private helpers
// ────────────────────────────────────────────────────────────────

/// Normalize separators in path text involved in autocomplete to `/`: fold Windows drive letters/user-input `\`, so prefix splitting,
/// candidate echo, and fuzzy matching all work under a single separator semantics (identity pass-through on non-Windows). Aligns with pi `toDisplayPath`.
fn to_display_path(s: &str) -> String {
    s.replace('\\', "/")
}

/// Locate the last completion delimiter (space/Tab/quote/`'`/`=`) in the text before the cursor: `extract_at_prefix`/`extract_path_prefix`
/// cut the "token being completed" from after it — the splitting primitive for path/attachment context detection. Aligns with pi `findLastDelimiter`.
/// Args: `text` — text before the cursor; returns the **byte index** of the last delimiter, or `None` if none exists.
fn find_last_delimiter(text: &str) -> Option<usize> {
    text.char_indices()
        .rev()
        .find(|(_, c)| PATH_DELIMITERS.contains(c))
        .map(|(i, _)| i)
}

/// Detect the start of an unclosed double quote in the text before the cursor: precondition for quoted path completion (`"my folder/…`),
/// only worth taking the quoted branch when currently inside a quote. Aligns with pi `findUnclosedQuoteStart`.
/// Args: `text`; returns the byte index of the unclosed quote, or `None` if quotes are paired or absent.
fn find_unclosed_quote_start(text: &str) -> Option<usize> {
    let mut in_quotes = false;
    let mut quote_start = None;
    for (i, c) in text.char_indices() {
        if c == '"' {
            in_quotes = !in_quotes;
            if in_quotes {
                quote_start = Some(i);
            }
        }
    }
    if in_quotes {
        quote_start
    } else {
        None
    }
}

/// Determine whether an index is the start of a completion token (line start or preceded by a completion delimiter): `@"`/`"` prefix validation
/// relies on it to reject quotes embedded mid-word (e.g. `a"b`). Aligns with pi `isTokenStart`.
/// Args: `text` + `index` (byte index, must be on a char boundary); returns whether it is a token start.
fn is_token_start(text: &str, index: usize) -> bool {
    match text.get(..index) {
        Some(before) => {
            index == 0
                || before
                    .chars()
                    .next_back()
                    .is_some_and(|c| PATH_DELIMITERS.contains(&c))
        }
        None => false,
    }
}

/// Extract the quoted completion prefix (`@"…`/`"…`, including the unclosed quote start as a whole): when the user types `"my fol`, returns `"my fol`,
/// letting `extract_at_prefix`/`extract_path_prefix` detect "quoted context" and hand off to `parse_path_prefix` to split off the bare path.
/// Aligns with pi `extractQuotedPrefix`; returns `None` when the quote edge is not a token start.
fn extract_quoted_prefix(text: &str) -> Option<String> {
    let quote_start = find_unclosed_quote_start(text)?;
    if quote_start > 0 && text[..quote_start].ends_with('@') {
        if !is_token_start(text, quote_start - 1) {
            return None;
        }
        return Some(text[quote_start - 1..].to_string());
    }
    if !is_token_start(text, quote_start) {
        return None;
    }
    Some(text[quote_start..].to_string())
}

/// Split an input prefix into "bare path fragment + modifier flags" as a `PathPrefix`: the four forms `@"a b`/`"a b`/`@a b`/`a b`
/// map to (raw, at, quoted) respectively, so file completion parses only the raw segment and restores `@` and quotes on display per the flags.
/// Aligns with pi `parsePathPrefix`.
fn parse_path_prefix(prefix: &str) -> PathPrefix {
    if let Some(rest) = prefix.strip_prefix("@\"") {
        PathPrefix {
            raw_prefix: rest.to_string(),
            is_at_prefix: true,
            is_quoted_prefix: true,
        }
    } else if let Some(rest) = prefix.strip_prefix('"') {
        PathPrefix {
            raw_prefix: rest.to_string(),
            is_at_prefix: false,
            is_quoted_prefix: true,
        }
    } else if let Some(rest) = prefix.strip_prefix('@') {
        PathPrefix {
            raw_prefix: rest.to_string(),
            is_at_prefix: true,
            is_quoted_prefix: false,
        }
    } else {
        PathPrefix {
            raw_prefix: prefix.to_string(),
            is_at_prefix: false,
            is_quoted_prefix: false,
        }
    }
}

/// Build the completion text applied to the line from the bare relative path plus decoration flags: add quotes when the space-or-quote context calls for it, `@` when in `@`.
/// Aligns with pi `buildCompletionValue`; **resolution**: the `isDirectory` param of the TS function is not used in the assembly
/// (the trailing `/` for directories is pre-appended to the end of `path` by the caller), so it is not received here.
/// Params: `path` is the bare relative path with a trailing `/` for directories; returns the full value with decorations.
fn build_completion_value(path: &str, is_at_prefix: bool, is_quoted_prefix: bool) -> String {
    let needs_quotes = is_quoted_prefix || path.contains(' ');
    let at = if is_at_prefix { "@" } else { "" };
    if !needs_quotes {
        format!("{at}{path}")
    } else {
        format!("{at}\"{path}\"")
    }
}

/// `@` attachment-context detection: when the text before the cursor is inside an `@` (or `@"…`) token, return the prefixed text with the decoration,
/// so that `get_suggestions` takes the `@` fuzzy file completion branch instead of the path/command branch. Aligns with pi `extractAtPrefix`.
/// Returns: the `@"…` / `@…` prefixed text; `None` if not inside an `@` token.
fn extract_at_prefix(text: &str) -> Option<String> {
    if let Some(quoted_prefix) = extract_quoted_prefix(text) {
        if quoted_prefix.starts_with("@\"") {
            return Some(quoted_prefix);
        }
    }
    let last_delimiter = find_last_delimiter(text);
    let token_start = last_delimiter.map_or(0, |i| i + 1);
    if text[token_start..].starts_with('@') {
        return Some(text[token_start..].to_string());
    }
    None
}

/// Whether the text before the cursor is in an `@` attachment completion context (at-attachment-wiring-fix): the last token starts with `@` (or `@"`)
/// at the beginning (token start = line start or after PATH_DELIMITER, reusing the splitting semantics of private `extract_at_prefix`)——
/// Type an @ or continue typing in an @token, and file candidates should appear.
/// Argument: `text_before_cursor` is the text before the cursor on the current line; if it returns true, the caller (zen-app `maybe_trigger_autocomplete`)
/// Call `trigger_autocomplete`.
/// Boundary: `@` embedded in `email@example` → false (does not trigger); `@`/`@"` bare prefix → true.
pub fn is_at_attachment_context(text_before_cursor: &str) -> bool {
    extract_at_prefix(text_before_cursor).is_some()
}

/// Path-prefix detection: extract the "path segment being completed" (after the last separator; in a quoted context, the whole span inside the quotes).
/// With `force=true` (Tab) any segment is returned (possibly an empty string; the caller then checks candidates); otherwise the segment must contain `/`,
/// start with `.`/`~/`, or immediately follow a space to trigger. Aligns with pi `extractPathPrefix`.
/// Returns `None` to mean path completion is not triggered (note: unlike `resolve_scoped_fuzzy_query`, where `None` = full base-directory traversal).
fn extract_path_prefix(text: &str, force: bool) -> Option<String> {
    if let Some(quoted_prefix) = extract_quoted_prefix(text) {
        return Some(quoted_prefix);
    }
    let last_delimiter = find_last_delimiter(text);
    let path_prefix = last_delimiter.map_or(text, |i| &text[i + 1..]);
    if force {
        return Some(path_prefix.to_string());
    }
    if path_prefix.contains('/') || path_prefix.starts_with('.') || path_prefix.starts_with("~/") {
        return Some(path_prefix.to_string());
    }
    if path_prefix.is_empty() && text.ends_with(' ') {
        return Some(path_prefix.to_string());
    }
    None
}

/// Expand a `~`/`~/…` prefix to the home directory: `~/a/b` → `$HOME/a/b`, preserving the trailing `/` (`~/` → `$HOME/`),
/// so file completion can readdir the `~` root while keeping the `~` display form. Aligns with pi `expandHomePath`.
/// Relies on the existing `home` crate in the workspace (`home::home_dir()`); returns unchanged when `~` is not involved.
fn expand_home_path(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/") {
        return match home::home_dir() {
            Some(home) => {
                let mut expanded = to_display_path(&home.join(rest).to_string_lossy());
                if path.ends_with('/') && !expanded.ends_with('/') {
                    expanded.push('/');
                }
                expanded
            }
            None => path.to_string(),
        };
    }
    if path == "~" {
        return home::home_dir()
            .map(|h| to_display_path(&h.to_string_lossy()))
            .unwrap_or_else(|| path.to_string());
    }
    path.to_string()
}

/// Directory name of a display path (aligns with node `path.dirname`: no separator → ".").
fn dirname(p: &str) -> String {
    Path::new(p)
        .parent()
        .map(|d| d.to_string_lossy().into_owned())
        .filter(|d| !d.is_empty())
        .unwrap_or_else(|| ".".to_string())
}

/// Base name of a display path (aligns with node `path.basename`: `.`/`..` returned as-is).
fn basename(p: &str) -> String {
    if p == "." || p == ".." {
        return p.to_string();
    }
    Path::new(p)
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Generate path candidates by readdir'ing the target directory: the raw segment is split into "root/dir/partial name" three states (ending in `/` = false displays directory
/// contents; otherwise "dir name + file prefix" is cut at the last separator), case-insensitive prefix filtering, symlink-directory detection,
/// `./` and `~/` display forms preserved, directories first + alphabetical order. Aligns with pi `getFileSuggestions`.
/// Params: `prefix` is the path segment before the cursor (with `@`/quote decorations; `parse_path_prefix` internally); `base_path` is the relative base.
/// Returns: candidates (value includes suffix, label is name + trailing `/`); returns an empty Vec if the target directory doesn't exist/is unreadable (completion is best-effort).
fn get_file_suggestions(prefix: &str, base_path: &Path) -> Vec<AutocompleteItem> {
    let pp = parse_path_prefix(prefix);
    let raw_prefix = &pp.raw_prefix;
    let is_at_prefix = pp.is_at_prefix;
    let is_quoted_prefix = pp.is_quoted_prefix;

    let mut expanded_prefix = raw_prefix.clone();
    if expanded_prefix.starts_with('~') {
        expanded_prefix = expand_home_path(&expanded_prefix);
    }

    let is_root_prefix = raw_prefix.is_empty()
        || raw_prefix == "./"
        || raw_prefix == "../"
        || raw_prefix == "~"
        || raw_prefix == "~/"
        || raw_prefix == "/"
        || (is_at_prefix && raw_prefix.is_empty());

    // Resolves the traversal directory and filename prefix from the three states: root / directory-end / partial name
    let (search_dir, search_prefix) = if is_root_prefix || raw_prefix.ends_with('/') {
        let dir = if raw_prefix.starts_with('~') || expanded_prefix.starts_with('/') {
            PathBuf::from(&expanded_prefix)
        } else {
            base_path.join(&expanded_prefix)
        };
        (dir, String::new())
    } else {
        let dir_text = dirname(&expanded_prefix);
        let file = basename(&expanded_prefix);
        let dir = if raw_prefix.starts_with('~') || expanded_prefix.starts_with('/') {
            PathBuf::from(dir_text)
        } else {
            base_path.join(dir_text)
        };
        (dir, file)
    };

    let Ok(entries) = std::fs::read_dir(&search_dir) else {
        return Vec::new();
    };

    let display_prefix = to_display_path(raw_prefix);
    let lower_search_prefix = search_prefix.to_lowercase();
    let mut suggestions = Vec::new();

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.to_lowercase().starts_with(&lower_search_prefix) {
            continue;
        }
        // Directory detection follows symlinks (aligned with pi: symlinks to directories are treated as directories; broken/permission-denied as files)
        let mut is_directory = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        if !is_directory && entry.file_type().map(|t| t.is_symlink()).unwrap_or(false) {
            is_directory = std::fs::metadata(entry.path())
                .map(|m| m.is_dir())
                .unwrap_or(false);
        }

        // Candidate echo path (preserves `~/`, `./` display form, aligned with pi displayPrefix branch)
        let relative_path = if display_prefix.ends_with('/') {
            format!("{display_prefix}{name}")
        } else if display_prefix.contains('/') {
            if let Some(home_relative) = display_prefix.strip_prefix("~/") {
                let dir = dirname(home_relative);
                if dir == "." {
                    format!("~/{name}")
                } else {
                    format!("~/{dir}/{name}")
                }
            } else if display_prefix.starts_with('/') {
                let dir = dirname(&display_prefix);
                if dir == "/" {
                    format!("/{name}")
                } else {
                    format!("{dir}/{name}")
                }
            } else {
                let dir = dirname(&display_prefix);
                let mut rel = if dir == "." {
                    name.clone()
                } else {
                    format!("{dir}/{name}")
                };
                if display_prefix.starts_with("./") && !rel.starts_with("./") {
                    rel = format!("./{rel}");
                }
                rel
            }
        } else if display_prefix.starts_with('~') {
            format!("~/{name}")
        } else {
            name.clone()
        };

        let path_value = if is_directory {
            format!("{relative_path}/")
        } else {
            relative_path
        };
        let value = build_completion_value(&path_value, is_at_prefix, is_quoted_prefix);
        let label = if is_directory {
            format!("{name}/")
        } else {
            name
        };
        suggestions.push(AutocompleteItem::new(value, label, None));
    }

    // Directories first, then label alphabetical order (case-insensitive, aligned with pi directory-first + localeCompare)
    suggestions.sort_by(|a, b| {
        let a_dir = a.label.ends_with('/');
        let b_dir = b.label.ends_with('/');
        if a_dir != b_dir {
            return if a_dir {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            };
        }
        a.label.to_lowercase().cmp(&b.label.to_lowercase())
    });
    suggestions
}

/// Scopes an `@` query into a `ScopedQuery` of "traversal base directory + in-segment query + display prefix":
/// `@tui/src/auto` → base=tui/src, query=auto, display=tui/src/; a bare `@` query without `/` returns `None`
/// (the caller falls back to full base-directory traversal). `~/` and absolute paths are resolved via `expand_home_path`.
/// Note: returning `None` does not mean "no trigger"; the caller traverses under base_path with the full query text.
fn resolve_scoped_fuzzy_query(raw_query: &str, base_path: &Path) -> Option<ScopedQuery> {
    let normalized = to_display_path(raw_query);
    let slash_index = normalized.rfind('/')?;
    let display_base = normalized[..=slash_index].to_string();
    let query = normalized[slash_index + 1..].to_string();
    let base_dir = if display_base.starts_with("~/") {
        PathBuf::from(expand_home_path(&display_base))
    } else if display_base.starts_with('/') {
        PathBuf::from(&display_base)
    } else {
        base_path.join(&display_base)
    };
    if !base_dir.is_dir() {
        return None;
    }
    Some(ScopedQuery {
        base_dir,
        query,
        display_base,
    })
}

/// Echoes the relative path returned by walk into a display path with the scoped prefix: root-joined when display_base=`/`, otherwise plain join,
/// ensuring candidate values for scoped completions like `@../outside/nested/a.ts` match the user's input form. Aligned with pi `scopedPathForDisplay`.
fn scoped_path_for_display(display_base: &str, relative_path: &str) -> String {
    let normalized = to_display_path(relative_path);
    if display_base == "/" {
        format!("/{normalized}")
    } else {
        format!("{}{normalized}", to_display_path(display_base))
    }
}

/// Query matching where `/`-separated segments appear in order (replaces fd's full-path regex): `tui/src/auto` requires the path to contain the substrings
/// `tui`, `src`, `auto` in sequence (case-insensitive); an in-process approximation of fd `--full-path` multi-segment semantics.
/// An empty `query`, or one of all-empty segments, always matches.
fn query_matches_path(query: &str, display_path: &str) -> bool {
    let segments: Vec<&str> = query.split('/').filter(|s| !s.is_empty()).collect();
    if segments.is_empty() {
        return true;
    }
    let lower_path = display_path.to_lowercase();
    let mut search_from = 0usize;
    for seg in segments {
        let lower_seg = seg.to_lowercase();
        match lower_path[search_from..].find(&lower_seg) {
            Some(pos) => search_from += pos + lower_seg.len(),
            None => return false,
        }
    }
    true
}

/// In-process directory traversal replaces external fd: recursively collect file and directory candidates from base_dir (hidden entries included as usual, respecting
/// .gitignore/.ignore/.git/info/exclude and excluding the entire .git tree; symlink directories are followed, cycles handled by the walker's
/// built-in protection); when query is non-empty, filter by `/`-segments appearing in order.
/// Align with pi `walkDirectoryWithFd` (fd's `.gitignore` respect — at-gitignore-fix; v1 previously used
/// walkdir without respect, causing ignored directories such as target/ to enter candidates; now uses the same `ignore` crate as fd/ripgrep).
/// `max_results` is the early-stop limit; when `ct` is cancelled, returns the partial results collected so far immediately (the caller short-circuits to empty).
/// Synchronous fs operations: async callers go through `tokio::task::spawn_blocking` or accept short blocking.
fn walk_directory(
    base_dir: &Path,
    query: &str,
    max_results: usize,
    ct: &CancellationToken,
) -> Vec<FuzzyEntry> {
    let mut results = Vec::new();
    // Align with pi's fd arguments (`--type f --type d --follow --hidden --exclude .git …`):
    // follow_links = `--follow`; hidden(true) = `--hidden` (hidden files/directories shown as usual);
    // The rest (ignore/git_ignore/git_exclude/git_global) = fd's default .gitignore-family respect.
    let mut builder = ignore::WalkBuilder::new(base_dir);
    builder
        .follow_links(true)
        .hidden(false) // false = do not ignore hidden entries (align with fd `--hidden`: hidden files/directories shown as usual)
        .ignore(true)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true);
    // Explicitly exclude the entire .git tree (equivalent on the fd side via `--exclude .git/.git/*/.git/**`)
    builder.filter_entry(|entry| {
        !(entry.depth() > 0
            && entry.file_name() == ".git"
            && entry.file_type().map(|t| t.is_dir()).unwrap_or(false))
    });
    let mut walker = builder.build();
    // Skip base_dir itself (fd outputs relative to children, not the root)
    let _ = walker.next();
    while let Some(entry) = walker.next() {
        if ct.is_cancelled() || results.len() >= max_results {
            break;
        }
        let Ok(entry) = entry else { continue };
        let is_directory = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        let Some(relative) = entry.path().strip_prefix(base_dir).ok() else {
            continue;
        };
        let display = to_display_path(&relative.to_string_lossy());
        if !query.is_empty() && !query_matches_path(query, &display) {
            continue;
        }
        results.push(FuzzyEntry {
            path: if is_directory {
                format!("{display}/")
            } else {
                display
            },
            is_directory,
        });
    }
    results
}

/// Scoring of file/directory candidates for `@` queries (exact 100 / prefix 80 / in-segment contains 50 / full-path contains 30, directory +10):
/// Full walk results are sorted by this score to take the top ones, so `@` completion prioritizes exact/prefix/directory hits. Aligned with pi `scoreEntry`.
/// Params: `file_path` display path, `query` in-segment query, `is_directory`; returns 0 = no match (caller filters).
fn score_entry(file_path: &str, query: &str, is_directory: bool) -> u32 {
    let file_name = basename(file_path);
    let lower_file = file_name.to_lowercase();
    let lower_query = query.to_lowercase();
    let score = if lower_file == lower_query {
        100
    } else if lower_file.starts_with(&lower_query) {
        80
    } else if lower_file.contains(&lower_query) {
        50
    } else if file_path.to_lowercase().contains(&lower_query) {
        30
    } else {
        0
    };
    if is_directory && score > 0 {
        score + 10
    } else {
        score
    }
}

/// `@` attachment candidate main entry: `resolve_scoped_fuzzy_query` boundary parsing → `walk_directory` full scan → `score_entry`
/// Filter to top 20 → `scoped_path_for_display` echo → `build_completion_value` wrapping (`@`+quotes)
/// → candidates carry description (full display path). Aligns with pi `getFuzzyFileSuggestions`.
/// Returns empty Vec when `fuzzy_walk_enabled=false` (caller decides) or `ct` is cancelled.
async fn get_fuzzy_file_suggestions(
    query: &str,
    is_quoted_prefix: bool,
    base_path: &Path,
    ct: &CancellationToken,
) -> Vec<AutocompleteItem> {
    if ct.is_cancelled() {
        return Vec::new();
    }
    let scoped = resolve_scoped_fuzzy_query(query, base_path);
    let base_dir = scoped
        .as_ref()
        .map(|s| s.base_dir.clone())
        .unwrap_or_else(|| base_path.to_path_buf());
    let walk_query = scoped
        .as_ref()
        .map(|s| s.query.clone())
        .unwrap_or_else(|| query.to_string());

    // Move synchronous fs traversal into spawn_blocking (a hard prerequisite for non-blocking, at-attachment-wiring-fix): LocalSet !Send context
    // Transfer across threads via owned data (base_dir/walk_query already owned) + ct clone; check ct within the walk loop to stop early,
    // If the spawn_blocking join fails (runtime shutdown, etc.), fall back to an empty result. Keep scoring/sorting on the async side.
    let base_dir_blocking = base_dir.clone();
    let walk_query_blocking = walk_query.clone();
    let ct_blocking = ct.clone();
    let entries = tokio::task::spawn_blocking(move || {
        walk_directory(&base_dir_blocking, &walk_query_blocking, 100, &ct_blocking)
    })
    .await
    .unwrap_or_default();
    if ct.is_cancelled() {
        return Vec::new();
    }

    let mut scored: Vec<(u32, FuzzyEntry)> = entries
        .into_iter()
        .map(|entry| {
            let score = if walk_query.is_empty() {
                1
            } else {
                score_entry(&entry.path, &walk_query, entry.is_directory)
            };
            (score, entry)
        })
        .filter(|(score, _)| *score > 0)
        .collect();
    scored.sort_by_key(|(score, _)| std::cmp::Reverse(*score)); // highest scores first

    let mut suggestions = Vec::with_capacity(scored.len().min(20));
    for (_, entry) in scored.into_iter().take(20) {
        let FuzzyEntry { path, is_directory } = entry;
        let path_without_slash = if is_directory {
            path.trim_end_matches('/')
        } else {
            &path
        };
        let display_path = match &scoped {
            Some(s) => scoped_path_for_display(&s.display_base, path_without_slash),
            None => path_without_slash.to_string(),
        };
        let entry_name = basename(path_without_slash);
        let completion_path = if is_directory {
            format!("{display_path}/")
        } else {
            display_path.clone()
        };
        let value = build_completion_value(&completion_path, true, is_quoted_prefix);
        let label = if is_directory {
            format!("{entry_name}/")
        } else {
            entry_name
        };
        suggestions.push(AutocompleteItem::new(value, label, Some(display_path)));
    }
    suggestions
}

/// Command name candidate filter: filter matching commands from the registry by the typed command prefix before the cursor (excluding `/`), combining description and hint into
/// dropdown text (when hint exists: `${hint} — ${desc}`), returning displayable candidates.
/// **fuzzy matching** (T-TU-03-4 upgrade, aligning with pi `fuzzyFilter(commandItems, prefix, i => i.name)`):
/// prefix characters hit as an ordered subsequence (not necessarily contiguous, case-insensitive), stably sorted by match quality (exact/contiguous matches ranked higher first);
/// empty prefix returns all in original order. This function's signature/return and callers unchanged (zero changes to the command-name branch of `get_suggestions`).
fn filter_commands(commands: &[SlashCommand], prefix: &str) -> Vec<AutocompleteItem> {
    crate::fuzzy::fuzzy_filter(commands, prefix, |cmd| cmd.name.as_str())
        .into_iter()
        .map(|cmd| {
            let full_description = match (&cmd.argument_hint, &cmd.description) {
                (Some(hint), Some(desc)) => format!("{hint} — {desc}"),
                (Some(hint), None) => hint.clone(),
                (None, desc) => desc.clone().unwrap_or_default(),
            };
            AutocompleteItem::new(
                cmd.name.clone(),
                cmd.name.clone(),
                if full_description.is_empty() {
                    None
                } else {
                    Some(full_description)
                },
            )
        })
        .collect()
}

/// Convert a char-index column to a byte offset (cursor column → &str slice boundary). When `char_col` exceeds the line's char count, clamp to end of line.
fn byte_index_of(line: &str, char_col: usize) -> usize {
    line.char_indices()
        .nth(char_col)
        .map(|(i, _)| i)
        .unwrap_or(line.len())
}

#[cfg(test)]
mod tests {
    /// CJK sample kept intentionally: this test verifies wide-character / multibyte behavior.
    const CJK_SAMPLE: &str = "你好 世界";
    const CJK_SAMPLE_2: &str = "你好";
    const CJK_SAMPLE_3: &str = "你好 /src";
    const CJK_SAMPLE_4: &str = "你好 /";
    use super::*;
    use std::fs;

    // ── pure function leaves ──

    #[test]
    fn to_display_path_converts_backslashes() {
        assert_eq!(to_display_path("a\\b/c"), "a/b/c");
        assert_eq!(to_display_path("plain"), "plain");
    }

    #[test]
    fn find_last_delimiter_finds_last_space_tab_quote_apostrophe_equals() {
        assert_eq!(find_last_delimiter("hello world"), Some(5));
        assert_eq!(find_last_delimiter("a=b=c"), Some(3));
        assert_eq!(find_last_delimiter("hello\tworld"), Some(5));
        assert_eq!(find_last_delimiter("no-delim"), None);
        assert_eq!(find_last_delimiter(""), None);
        // space before CJK: byte index falls on the space
        assert_eq!(find_last_delimiter(CJK_SAMPLE), Some(6));
    }

    #[test]
    fn find_unclosed_quote_start_detects_unclosed_only() {
        assert_eq!(find_unclosed_quote_start("no quotes"), None);
        assert_eq!(find_unclosed_quote_start("\"closed\""), None);
        assert_eq!(find_unclosed_quote_start("\"open"), Some(0));
        // one pair closed, then another unclosed (pair 2/5 closed, then 8 unclosed)
        assert_eq!(find_unclosed_quote_start("a \"b \"c \"d"), Some(8));
        assert_eq!(find_unclosed_quote_start("x \"y"), Some(2));
    }

    #[test]
    fn is_token_start_checks_line_start_and_delimiters() {
        assert!(is_token_start("abc", 0));
        assert!(is_token_start("a abc", 2));
        assert!(!is_token_start("abc", 1));
        assert!(is_token_start("\"abc\"", 0));
        assert!(!is_token_start("a\"b\"", 1)); // previous char is not a separator
    }

    #[test]
    fn extract_quoted_prefix_handles_at_and_plain_quotes() {
        assert_eq!(
            extract_quoted_prefix("\"my fol"),
            Some("\"my fol".to_string())
        );
        assert_eq!(
            extract_quoted_prefix("@\"my fol"),
            Some("@\"my fol".to_string())
        );
        assert_eq!(
            extract_quoted_prefix("pre @\"my fol"),
            Some("@\"my fol".to_string())
        );
        assert_eq!(extract_quoted_prefix("a\"b"), None); // fake quote
        assert_eq!(extract_quoted_prefix("plain"), None);
        assert_eq!(extract_quoted_prefix("\"closed\""), None);
    }

    #[test]
    fn parse_path_prefix_maps_four_forms() {
        let p = parse_path_prefix("@\"a b");
        assert_eq!(p.raw_prefix, "a b");
        assert!(p.is_at_prefix);
        assert!(p.is_quoted_prefix);

        let p = parse_path_prefix("\"a b");
        assert_eq!(p.raw_prefix, "a b");
        assert!(!p.is_at_prefix);
        assert!(p.is_quoted_prefix);

        let p = parse_path_prefix("@ab");
        assert_eq!(p.raw_prefix, "ab");
        assert!(p.is_at_prefix);
        assert!(!p.is_quoted_prefix);

        let p = parse_path_prefix("ab");
        assert_eq!(p.raw_prefix, "ab");
        assert!(!p.is_at_prefix);
        assert!(!p.is_quoted_prefix);
    }

    #[test]
    fn build_completion_value_quotes_only_when_needed() {
        assert_eq!(build_completion_value("src", false, false), "src");
        assert_eq!(
            build_completion_value("my folder", false, false),
            "\"my folder\""
        );
        assert_eq!(build_completion_value("src", true, false), "@src");
        assert_eq!(
            build_completion_value("my folder", true, true),
            "@\"my folder\""
        );
        assert_eq!(
            build_completion_value("my folder/", true, false),
            "@\"my folder/\""
        );
    }

    #[test]
    fn extract_at_prefix_detects_at_token() {
        assert_eq!(extract_at_prefix("@"), Some("@".to_string()));
        assert_eq!(
            extract_at_prefix("@file.txt"),
            Some("@file.txt".to_string())
        );
        assert_eq!(extract_at_prefix("hey @file"), Some("@file".to_string()));
        assert_eq!(
            extract_at_prefix("@\"my folder/te"),
            Some("@\"my folder/te".to_string())
        );
        assert_eq!(extract_at_prefix("email@example"), None); // @ is not at a token start
        assert_eq!(extract_at_prefix("plain"), None);
    }

    #[test]
    fn extract_path_prefix_natural_vs_forced() {
        // natural trigger: segment contains /, starts with . or ~/, empty segment after space
        assert_eq!(
            extract_path_prefix("hey src/foo", false),
            Some("src/foo".to_string())
        );
        assert_eq!(extract_path_prefix("./up", false), Some("./up".to_string()));
        assert_eq!(extract_path_prefix("~/x", false), Some("~/x".to_string()));
        assert_eq!(extract_path_prefix("hey ", false), Some("".to_string()));
        assert_eq!(extract_path_prefix("plain", false), None);
        // forced trigger: any segment
        assert_eq!(
            extract_path_prefix("plain", true),
            Some("plain".to_string())
        );
        assert_eq!(extract_path_prefix("hey /", true), Some("/".to_string()));
        // quote context goes to the quote branch
        assert_eq!(
            extract_path_prefix("\"my folder/te", false),
            Some("\"my folder/te".to_string())
        );
    }

    #[test]
    fn expand_home_path_passthrough_and_tilde() {
        assert_eq!(expand_home_path("plain"), "plain");
        assert_eq!(expand_home_path(""), "");
        // when HOME is available, ~ expands to an absolute path
        if let Some(home) = home::home_dir() {
            assert_eq!(
                expand_home_path("~"),
                to_display_path(&home.to_string_lossy())
            );
            assert!(!expand_home_path("~/x").is_empty());
            // Preserve trailing slash
            assert!(expand_home_path("~/x/").ends_with('/'));
        }
    }

    #[test]
    fn dirname_basename_follow_node_semantics() {
        assert_eq!(dirname("a/b"), "a");
        assert_eq!(dirname("b"), ".");
        assert_eq!(dirname("/x"), "/");
        assert_eq!(basename("a/b"), "b");
        assert_eq!(basename("b"), "b");
        assert_eq!(basename("."), ".");
        assert_eq!(basename("src/"), "src");
    }

    #[test]
    fn score_entry_priority_and_dir_bonus() {
        assert_eq!(score_entry("src/", "src", true), 110);
        assert_eq!(score_entry("src.txt", "src", false), 80);
        assert_eq!(score_entry("myfile.txt", "file", false), 50);
        assert_eq!(score_entry("packages/tui/a.ts", "tui", false), 30);
        assert_eq!(score_entry("zzz.txt", "qqq", false), 0);
        // Bonus directory only applies when score > 0
        assert_eq!(score_entry("zzz/", "qqq", true), 0);
        assert_eq!(score_entry("tui/", "tui", true), 110);
    }

    #[test]
    fn query_matches_path_segments_in_order() {
        assert!(query_matches_path(
            "tui/src/auto",
            "packages/tui/src/autocomplete.ts"
        ));
        assert!(!query_matches_path(
            "tui/src/auto",
            "packages/ai/src/autocomplete.ts"
        ));
        assert!(query_matches_path(
            "components/",
            "src/components/Button.tsx"
        ));
        assert!(!query_matches_path("components/", "src/utils/helpers.ts"));
        assert!(query_matches_path("re", "README.md"));
        assert!(query_matches_path("", "anything"));
        assert!(query_matches_path("/", "anything"));
    }

    #[test]
    fn filter_commands_fuzzy_match_and_description_join() {
        let commands = vec![
            SlashCommand::new(
                "model",
                Some("select model".to_string()),
                Some("<p/m>".to_string()),
                None,
            ),
            SlashCommand::new("new", Some("start new session".to_string()), None, None),
            SlashCommand::new("newer", None, None, None),
        ];
        // Under fuzzy matching, assertions consistent with old prefix semantics are kept: exact matches/consecutive hits come first
        let items = filter_commands(&commands, "new");
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].value(), "new");
        assert_eq!(items[1].value(), "newer");
        // If hint exists: `${hint} — ${desc}`
        let model = filter_commands(&commands, "mod");
        assert_eq!(model.len(), 1);
        assert_eq!(model[0].description(), Some("<p/m> — select model"));
        // Empty prefix = all (empty fuzzy_filter query returns everything in original order)
        assert_eq!(filter_commands(&commands, "").len(), 3);
        // No match
        assert!(filter_commands(&commands, "zzz").is_empty());
    }

    #[test]
    fn filter_commands_matches_fuzzy_subsequences() {
        let commands = vec![
            SlashCommand::new("model", None, None, None),
            SlashCommand::new("new", None, None, None),
            SlashCommand::new("newer", None, None, None),
        ];
        // Subsequence hit (not a prefix): n-w-r appears in order in "newer"; does not hit "new" (no r) or "model" (no n)
        let items = filter_commands(&commands, "nwr");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].value(), "newer");
        // Prefix order mismatch but fuzzy hit: n-w-e appears in order in "newer" (n→w→e exists),
        // while "new" has character order n-e-w, so "nwe" does not match "new"
        let items = filter_commands(&commands, "nwe");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].value(), "newer");
        // Exact match takes precedence over subsequence hit: "new" exactly hits "new", "newer" is only a subsequence
        let items = filter_commands(&commands, "new");
        assert_eq!(items[0].value(), "new");
        assert_eq!(items[1].value(), "newer");
    }

    #[test]
    fn byte_index_of_converts_char_col_to_byte() {
        assert_eq!(byte_index_of("abc", 2), 2);
        assert_eq!(byte_index_of(CJK_SAMPLE_2, 1), 3);
        assert_eq!(byte_index_of(CJK_SAMPLE_2, 5), 6); // Beyond → end of line
    }

    // ── Filesystem support (tempdir + std fs) ──

    /// Lay out a `structure` directory tree under a temporary directory.
    fn setup_folder(base: &Path, dirs: &[&str], files: &[&str]) {
        for d in dirs {
            fs::create_dir_all(base.join(d)).unwrap();
        }
        for f in files {
            let path = base.join(f);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, "content").unwrap();
        }
    }

    #[test]
    fn get_file_suggestions_lists_base_dir_and_sorts_dirs_first() {
        let tmp = tempfile::tempdir().unwrap();
        setup_folder(tmp.path(), &["src", "docs"], &["README.md", "src.txt"]);
        let items = get_file_suggestions("", tmp.path());
        // Directories first, alphabetical within group: docs/ src/ come first
        assert_eq!(items[0].value(), "docs/");
        assert_eq!(items[1].value(), "src/");
        let values: Vec<&str> = items.iter().map(|i| i.value()).collect();
        assert!(values.contains(&"README.md"));
        assert!(values.contains(&"src.txt"));
    }

    #[test]
    fn get_file_suggestions_prefix_filter_and_case_insensitive() {
        let tmp = tempfile::tempdir().unwrap();
        setup_folder(tmp.path(), &[], &["README.md", "update.sh"]);
        let items = get_file_suggestions("up", tmp.path());
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].value(), "update.sh");
        let items = get_file_suggestions("re", tmp.path());
        assert_eq!(items[0].value(), "README.md");
    }

    #[test]
    fn get_file_suggestions_preserves_dot_slash_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        setup_folder(tmp.path(), &[], &["update.sh"]);
        let items = get_file_suggestions("./up", tmp.path());
        assert!(items.iter().any(|i| i.value() == "./update.sh"));
        // Directories one by one
        setup_folder(tmp.path(), &["src"], &[]);
        let items = get_file_suggestions("./sr", tmp.path());
        assert!(items.iter().any(|i| i.value() == "./src/"));
    }

    #[test]
    fn get_file_suggestions_quotes_paths_with_spaces() {
        let tmp = tempfile::tempdir().unwrap();
        setup_folder(tmp.path(), &["my folder"], &["my folder/test.txt"]);
        let items = get_file_suggestions("my", tmp.path());
        assert!(items.iter().any(|i| i.value() == "\"my folder/\""));
    }

    #[test]
    fn get_file_suggestions_missing_dir_returns_empty() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(get_file_suggestions("nope/", tmp.path()).is_empty());
    }

    #[test]
    fn resolve_scoped_fuzzy_query_scopes_to_existing_dir() {
        let tmp = tempfile::tempdir().unwrap();
        setup_folder(tmp.path(), &["tui/src"], &[]);
        let scoped = resolve_scoped_fuzzy_query("tui/src/auto", tmp.path()).unwrap();
        assert_eq!(scoped.query, "auto");
        assert_eq!(scoped.display_base, "tui/src/");
        assert_eq!(scoped.base_dir, tmp.path().join("tui/src"));
        // No / → None (traverse from the base directory)
        assert!(resolve_scoped_fuzzy_query("auto", tmp.path()).is_none());
        // Bounding directory does not exist → None
        assert!(resolve_scoped_fuzzy_query("nope/x", tmp.path()).is_none());
    }

    #[test]
    fn walk_directory_excludes_git_includes_hidden_and_dirs_trailing_slash() {
        let tmp = tempfile::tempdir().unwrap();
        setup_folder(
            tmp.path(),
            &[".pi", ".github", ".git", "src/deep"],
            &[
                ".pi/config.json",
                ".github/workflows/ci.yml",
                ".git/config",
                "src/deep/a.ts",
            ],
        );
        let ct = CancellationToken::new();
        let entries = walk_directory(tmp.path(), "", 100, &ct);
        let paths: Vec<&str> = entries.iter().map(|e| e.path.as_str()).collect();
        assert!(paths.contains(&".pi/"));
        assert!(paths.contains(&".github/"));
        assert!(paths.contains(&".pi/config.json"));
        assert!(paths.contains(&"src/deep/a.ts"));
        // Directory with trailing slash
        assert!(
            entries
                .iter()
                .find(|e| e.path == "src/")
                .unwrap()
                .is_directory
        );
        // Exclude the whole .git tree (keep .github)
        assert!(paths.contains(&".github/"));
        assert!(entries
            .iter()
            .all(|e| !(e.path == ".git" || e.path.starts_with(".git/"))));
    }

    #[test]
    fn walk_directory_respects_gitignore_excludes_target() {
        // at-gitignore-fix (pi alignment): directories in .gitignore (e.g. Rust's target/) do not enter candidates —
        // fd (pi) respects .gitignore by default; v1 previously used walkdir without respect, so target files could be selected.
        let tmp = tempfile::tempdir().unwrap();
        setup_folder(
            tmp.path(),
            &["target/debug", "src", ".git", "docs"],
            &[
                ".gitignore",
                "target/debug/zen",
                "src/main.rs",
                ".git/config",
                "docs/guide.md",
            ],
        );
        std::fs::write(tmp.path().join(".gitignore"), "target/\n").unwrap();
        let ct = CancellationToken::new();
        let entries = walk_directory(tmp.path(), "", 100, &ct);
        let paths: Vec<&str> = entries.iter().map(|e| e.path.as_str()).collect();
        assert!(
            entries
                .iter()
                .all(|e| !(e.path == "target" || e.path.starts_with("target/"))),
            "target/ is excluded by .gitignore: {paths:?}"
        );
        assert!(
            !paths.contains(&"target"),
            "target/ directory does not appear: {paths:?}"
        );
        assert!(
            paths.contains(&"src/main.rs"),
            "Unignored files still present: {paths:?}"
        );
        assert!(
            paths.contains(&"docs/guide.md"),
            "Unignored files still present: {paths:?}"
        );
        assert!(
            paths.contains(&".gitignore"),
            ".gitignore itself is visible: {paths:?}"
        );
    }

    #[test]
    fn walk_directory_gitignore_negation_keeps_files() {
        // .gitignore negation patterns (!) keep files explicitly added back (same fd/git semantics);
        // Note: the ignore crate applies .gitignore only when a `.git` directory exists (same behavior as fd/ripgrep,
        // real repositories always have it; this test uses a fixture containing .git).
        let tmp = tempfile::tempdir().unwrap();
        setup_folder(
            tmp.path(),
            &[".git", "src"],
            &[
                ".gitignore",
                "src/important.txt",
                "src/keep.log",
                "src/other.log",
            ],
        );
        std::fs::write(tmp.path().join(".gitignore"), "*.log\n!keep.log\n").unwrap();
        let ct = CancellationToken::new();
        let entries = walk_directory(tmp.path(), "", 100, &ct);
        let paths: Vec<&str> = entries.iter().map(|e| e.path.as_str()).collect();
        assert!(
            entries
                .iter()
                .all(|e| !e.path.starts_with(".git/") && e.path != "src/other.log"),
            "*.log ignored (keep.log excluded), .git excluded: {paths:?}"
        );
        // keep.log restored by a negation pattern is listed; other.log, which no negation matches, is not
        assert!(
            paths.contains(&"src/keep.log"),
            "!keep.log negation preserved: {paths:?}"
        );
        assert!(
            !paths.contains(&"src/other.log"),
            "other.log remains in ignore set: {paths:?}"
        );
        assert!(
            paths.contains(&"src/important.txt"),
            "txt is unaffected: {paths:?}"
        );
    }

    #[test]
    fn walk_directory_cancelled_stops_early() {
        let tmp = tempfile::tempdir().unwrap();
        setup_folder(tmp.path(), &["a"], &["a/b.ts", "a/c.ts", "a/d.ts"]);
        let ct = CancellationToken::new();
        ct.cancel();
        let entries = walk_directory(tmp.path(), "", 100, &ct);
        assert!(entries.is_empty());
    }

    #[tokio::test]
    async fn get_fuzzy_file_suggestions_basic_cases() {
        let tmp = tempfile::tempdir().unwrap();
        let ct = CancellationToken::new();

        // Empty @ → all (@ prefix + directory trailing /)
        setup_folder(tmp.path(), &["src"], &["README.md"]);
        let items = get_fuzzy_file_suggestions("", false, tmp.path(), &ct).await;
        let values: Vec<&str> = items.iter().map(|i| i.value()).collect();
        assert!(values.contains(&"@README.md"));
        assert!(values.contains(&"@src/"));

        // Query with extension
        setup_folder(tmp.path(), &[], &["file.txt"]);
        let items = get_fuzzy_file_suggestions("file.txt", false, tmp.path(), &ct).await;
        assert!(items.iter().any(|i| i.value() == "@file.txt"));

        // Case-insensitive; description is the relative path
        let items = get_fuzzy_file_suggestions("re", false, tmp.path(), &ct).await;
        assert!(items
            .iter()
            .any(|i| i.value() == "@README.md" && i.description() == Some("README.md")));

        // Directories take precedence (src/ ranks above src.txt)
        setup_folder(tmp.path(), &[], &["src.txt"]);
        let items = get_fuzzy_file_suggestions("src", false, tmp.path(), &ct).await;
        assert_eq!(items[0].value(), "@src/");

        // Nested path segments match in order
        setup_folder(
            tmp.path(),
            &["packages/tui/src", "packages/ai/src"],
            &[
                "packages/tui/src/autocomplete.ts",
                "packages/ai/src/autocomplete.ts",
            ],
        );
        let items = get_fuzzy_file_suggestions("tui/src/auto", false, tmp.path(), &ct).await;
        assert!(items
            .iter()
            .any(|i| i.value() == "@packages/tui/src/autocomplete.ts"));
        assert!(!items
            .iter()
            .any(|i| i.value() == "@packages/ai/src/autocomplete.ts"));

        // Cancel → empty
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        let items = get_fuzzy_file_suggestions("", false, tmp.path(), &cancelled).await;
        assert!(items.is_empty());
    }

    #[tokio::test]
    async fn get_fuzzy_file_suggestions_quoted_and_scoped() {
        let tmp = tempfile::tempdir().unwrap();
        let ct = CancellationToken::new();

        // Quote paths containing spaces
        setup_folder(tmp.path(), &["my folder"], &["my folder/test.txt"]);
        let items = get_fuzzy_file_suggestions("my", false, tmp.path(), &ct).await;
        assert!(items.iter().any(|i| i.value() == "@\"my folder/\""));

        // scoped: query ../outside/a under base=cwd → cwd/../outside/nested/alpha.ts
        let root = tempfile::tempdir().unwrap();
        let cwd = root.path().join("cwd");
        fs::create_dir_all(&cwd).unwrap();
        setup_folder(
            root.path(),
            &["outside/nested/deeper"],
            &[
                "outside/nested/alpha.ts",
                "outside/nested/deeper/also-alpha.ts",
                "outside/nested/deeper/zzz.ts",
            ],
        );
        // Scoped relative delimiting with ../outside/a in the query (base is cwd, not root)
        let items = get_fuzzy_file_suggestions("../outside/a", false, &cwd, &ct).await;
        assert!(items
            .iter()
            .any(|i| i.value() == "@../outside/nested/alpha.ts"));
        assert!(items
            .iter()
            .any(|i| i.value() == "@../outside/nested/deeper/also-alpha.ts"));
        assert!(!items
            .iter()
            .any(|i| i.value() == "@../outside/nested/deeper/zzz.ts"));
    }

    // ── CombinedAutocompleteProvider (trait assembly surface) ──

    fn command(name: &str, desc: Option<&str>, hint: Option<&str>) -> SlashCommand {
        SlashCommand::new(name, desc.map(String::from), hint.map(String::from), None)
    }

    #[tokio::test]
    async fn get_suggestions_slash_command_name_and_args() {
        let provider = CombinedAutocompleteProvider::new(
            vec![
                command("model", Some("select model"), Some("<provider/model>")),
                command("compact", Some("compact session"), None),
            ],
            tempfile::tempdir().unwrap().path().to_path_buf(),
            true,
        );
        let ct = CancellationToken::new();
        let lines = vec!["/mod".to_string()];
        let s = provider
            .get_suggestions(&lines, 0, 4, false, &ct)
            .await
            .unwrap();
        assert_eq!(s.prefix(), "/mod");
        assert_eq!(s.items().len(), 1);
        assert_eq!(s.items()[0].value(), "model");

        // Command argument: callback returns candidates (prefix filtering)
        let models: Vec<String> = vec!["deepseek-v4", "deepseek-r1"]
            .into_iter()
            .map(String::from)
            .collect();
        let arg_cmd = SlashCommand::new(
            "model",
            Some("select model".to_string()),
            Some("<p/m>".to_string()),
            Some(Box::new(move |prefix: &str, _ct: &CancellationToken| {
                let models = models.clone();
                let p = prefix.to_string();
                Box::pin(async move {
                    Some(
                        models
                            .into_iter()
                            .filter(|m| m.starts_with(&p))
                            .map(|m| AutocompleteItem::new(m.clone(), m, None))
                            .collect(),
                    )
                })
            })),
        );
        let provider = CombinedAutocompleteProvider::new(
            vec![arg_cmd],
            tempfile::tempdir().unwrap().path().to_path_buf(),
            true,
        );
        let lines = vec!["/model deepseek-r".to_string()];
        let s = provider
            .get_suggestions(&lines, 0, 17, false, &ct)
            .await
            .unwrap();
        assert_eq!(s.prefix(), "deepseek-r");
        assert_eq!(s.items().len(), 1);
        assert_eq!(s.items()[0].value(), "deepseek-r1");
    }

    #[tokio::test]
    async fn get_suggestions_argument_completion_none_returns_null() {
        // Command has no argument callback or callback returns None → not triggered
        let provider = CombinedAutocompleteProvider::new(
            vec![command("model", Some("select model"), None)],
            tempfile::tempdir().unwrap().path().to_path_buf(),
            true,
        );
        let ct = CancellationToken::new();
        let lines = vec!["/model abc".to_string()];
        assert!(provider
            .get_suggestions(&lines, 0, 10, false, &ct)
            .await
            .is_none());
    }

    #[tokio::test]
    async fn get_suggestions_path_completion_natural_and_forced() {
        let tmp = tempfile::tempdir().unwrap();
        setup_folder(tmp.path(), &[], &["update.sh", "utils.ts"]);
        let provider = CombinedAutocompleteProvider::new(vec![], tmp.path().to_path_buf(), true);
        let ct = CancellationToken::new();

        // Natural trigger on path segments containing /
        let lines = vec!["hey ./up".to_string()];
        let s = provider
            .get_suggestions(&lines, 0, 8, false, &ct)
            .await
            .unwrap();
        assert_eq!(s.prefix(), "./up");
        assert!(s.items().iter().any(|i| i.value() == "./update.sh"));

        // Tab forces root-directory candidates after a trailing space
        let lines = vec!["hey ".to_string()];
        let s = provider
            .get_suggestions(&lines, 0, 4, true, &ct)
            .await
            .unwrap();
        assert_eq!(s.prefix(), "");

        // Plain words do not trigger
        let lines = vec!["plain".to_string()];
        assert!(provider
            .get_suggestions(&lines, 0, 5, false, &ct)
            .await
            .is_none());
    }

    #[tokio::test]
    async fn get_suggestions_at_attachment_branch() {
        let tmp = tempfile::tempdir().unwrap();
        setup_folder(tmp.path(), &["src"], &["README.md"]);
        let ct = CancellationToken::new();

        // fuzzy walk enabled: @re → @README.md, prefix contains @
        let provider = CombinedAutocompleteProvider::new(vec![], tmp.path().to_path_buf(), true);
        let lines = vec!["hey @re".to_string()];
        let s = provider
            .get_suggestions(&lines, 0, 7, false, &ct)
            .await
            .unwrap();
        assert_eq!(s.prefix(), "@re");
        assert!(s.items().iter().any(|i| i.value() == "@README.md"));

        // fuzzy walk disabled: @ branch returns None directly
        let provider = CombinedAutocompleteProvider::new(vec![], tmp.path().to_path_buf(), false);
        let lines = vec!["@re".to_string()];
        assert!(provider
            .get_suggestions(&lines, 0, 3, false, &ct)
            .await
            .is_none());
    }

    #[tokio::test]
    async fn get_suggestions_empty_lines_returns_none() {
        let provider = CombinedAutocompleteProvider::new(
            vec![],
            tempfile::tempdir().unwrap().path().to_path_buf(),
            true,
        );
        let ct = CancellationToken::new();
        assert!(provider
            .get_suggestions(&[], 0, 0, false, &ct)
            .await
            .is_none());
    }

    // ── apply_completion: four branches ──

    #[test]
    fn apply_completion_slash_command_appends_space() {
        let provider = CombinedAutocompleteProvider::new(vec![], "/tmp".into(), true);
        // Slash command at line start: complete name + trailing space
        let lines = vec!["/mod".to_string()];
        let item = AutocompleteItem::new("model", "model", None);
        let r = provider.apply_completion(&lines, 0, 4, &item, "/mod");
        assert_eq!(r.lines(), &["/model ".to_string()]);
        assert_eq!(r.cursor_position(), (0, 7));
        // Command name not at line start (e.g. “hey /mod”) is not the slash branch: replace segment, keep remaining text (aligns with pi concatenation semantics)
        let lines = vec!["hey /mod".to_string()];
        let r = provider.apply_completion(&lines, 0, 7, &item, "/mod");
        assert_eq!(r.lines(), &["heymodeld".to_string()]);
    }

    #[test]
    fn apply_completion_at_file_adds_space_but_not_after_dir() {
        let provider = CombinedAutocompleteProvider::new(vec![], "/tmp".into(), true);
        // File completion → trailing space
        let lines = vec!["@file".to_string()];
        let item = AutocompleteItem::new("@file.txt", "file.txt", None);
        let r = provider.apply_completion(&lines, 0, 5, &item, "@file");
        assert_eq!(r.lines(), &["@file.txt ".to_string()]);
        assert_eq!(r.cursor_position(), (0, 10));

        // Directory completion → no trailing space (trailing / continues completion)
        let lines = vec!["@src".to_string()];
        let item = AutocompleteItem::new("@src/", "src/", None);
        let r = provider.apply_completion(&lines, 0, 4, &item, "@src");
        assert_eq!(r.lines(), &["@src/".to_string()]);
        assert_eq!(r.cursor_position(), (0, 5));
    }

    #[test]
    fn apply_completion_file_path_replaces_segment() {
        let provider = CombinedAutocompleteProvider::new(vec![], "/tmp".into(), true);
        let lines = vec!["open ./sr".to_string()];
        let item = AutocompleteItem::new("./src/", "src/", None);
        let r = provider.apply_completion(&lines, 0, 9, &item, "./sr");
        assert_eq!(r.lines(), &["open ./src/".to_string()]);
        assert_eq!(r.cursor_position(), (0, 11));
    }

    #[test]
    fn apply_completion_quoted_prefix_dedups_trailing_quote() {
        let provider = CombinedAutocompleteProvider::new(vec![], "/tmp".into(), true);
        // Cursor before the closing quote: value carries its own quotes → dedupe the closing quote after the cursor
        let line = "\"my folder/te\"".to_string();
        let lines = vec![line];
        let item = AutocompleteItem::new("\"my folder/test.txt\"", "test.txt", None);
        let r = provider.apply_completion(&lines, 0, 13, &item, "\"my folder/te");
        assert_eq!(r.lines(), &["\"my folder/test.txt\"".to_string()]);
    }

    #[test]
    fn should_trigger_file_completion_skips_slash_command_name() {
        let provider = CombinedAutocompleteProvider::new(vec![], "/tmp".into(), true);
        // Inside the leading-slash command name (no space after trim) → false (Tab reserved for command-name matching)
        assert!(!provider.should_trigger_file_completion(&["/model".to_string()], 0, 3));
        // Command name followed by a space (still /cmd with no internal space after trim) → false (aligns with pi trim semantics)
        assert!(!provider.should_trigger_file_completion(&["/model ".to_string()], 0, 7));
        // Command argument segment (contains a space after trim) → true
        assert!(provider.should_trigger_file_completion(&["/model x".to_string()], 0, 8));
        // Plain text → true
        assert!(provider.should_trigger_file_completion(&["hey .".to_string()], 0, 5));
    }

    // ── Cursor semantics: multi-byte text must not be split ──

    #[test]
    fn get_suggestions_multibyte_cursor_not_panicking() {
        // "hello /src": cursor is in the /src segment, so the prefix should be "/src"
        let line = CJK_SAMPLE_3.to_string();
        // Ask for characters of "hello /src" from index 4? — character index 4 maps to a byte offset; convert via byte_index_of and manually verify the helper
        let byte_col = byte_index_of(&line, 4);
        let text_before = &line[..byte_col];
        assert_eq!(text_before, CJK_SAMPLE_4);
        assert_eq!(
            extract_path_prefix(text_before, true),
            Some("/".to_string())
        );
    }

    // ── @ trigger predicate and trigger_characters (at-attachment-wiring-fix)──

    #[test]
    fn is_at_attachment_context_predicate() {
        assert!(is_at_attachment_context("@"));
        assert!(is_at_attachment_context("@file.txt"));
        assert!(is_at_attachment_context("@\"my folder/te"));
        assert!(is_at_attachment_context("hey @file"));
        assert!(!is_at_attachment_context("email@example")); // @ is not at a token start
        assert!(!is_at_attachment_context("plain"));
        assert!(!is_at_attachment_context(""));
    }

    #[test]
    fn trigger_characters_returns_at_only() {
        let provider = CombinedAutocompleteProvider::new(vec![], "/tmp".into(), true);
        // Only declares `@` (pi ["@","#"], `#` omitted in v1)
        assert_eq!(provider.trigger_characters(), &['@']);
    }
}
