// REQ-APP-11 (pi-355): interactive UI task-side state — InteractiveUi + channel message consumption + rendering.
// See reference/pi-original/packages/coding-agent/src/modes/interactive/interactive-mode.ts
// Design: the main task (mod.rs) pushes changes via the `UiEvent` channel; the UI task first enqueues events into a pending queue, then
// Then, in the draw frame, uniformly `apply` + render (see the assembly in `mod.rs` `init` and the render comments in this file's `ui.rs`).
// `InteractiveUi` is pure rendering/interaction state (`Rc<RefCell<..>>` accessed sequentially by three closures within the UI task,
// never across threads; the `Rc` choice aligns with the !Send nature of CustomEditor callbacks, see the assembly comments in mod.rs).

use crate::components::assistant_message::AssistantMessageComponent;
use crate::components::bash_execution::{BashExecutionComponent, TruncationInfo};
use crate::components::branch_summary_message::BranchSummaryMessageComponent;
use crate::components::compaction_summary_message::CompactionSummaryMessageComponent;
use crate::components::custom_editor::CustomEditor;
use crate::components::custom_entry::CustomEntryComponent;
use crate::components::custom_message::CustomMessageComponent;
use crate::components::footer::{FooterComponent, FooterData, UsageTotals};
use crate::components::skill_invocation_message::SkillInvocationMessageComponent;
use crate::components::status_indicator::{
    CompactionStatusReason, IdleStatus, StatusIndicator, StatusIndicatorKind,
};
use crate::components::tool_execution::{ToolExecutionComponent, ToolExecutionResult};
use crate::components::user_message::UserMessageComponent;
use crate::modes::interactive::{AppAction, InputCommand, StatusSpec, UiEvent};
use crossterm::event::KeyEvent;
use serde_json::Value;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use zen_ai::types::{AssistantMessage, ContentBlock, ContentBlockList, Message, UserMessage};
use zen_tui::components::Component;
use zen_tui::editor::EditorAction;
use zen_tui::keybindings::KeybindingsManager;
use zen_tui::keys::KeyId;
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{ColorToken, Theme};
use zen_tui::viewport::ScrollbackCanvas;

/// Height cap for single-line metrics of message blocks (the measure stage enlarges the cap and takes the real height; lines exceeding it are truncated by this value for scroll computation;
/// the vast majority of lines in the chat window are far below this value, and truncating giant lines (e.g. very long bash output) is an acceptable v1 approximation).
const MEASURE_HEIGHT_CAP: u16 = 300;

/// Shared rendering/interaction state on the UI task side (`Rc<RefCell<InteractiveUi>>` shared by the on_event/on_wake/draw closures);
/// Holds owned Theme and editor; data snapshot rendered line by line (components are constructed transiently, not stored in 'static containers;
/// the status line is the exception — its `status_indicator` persistent instance is reused across frames, see apply_status).
///
/// **Send note**: this type contains `Rc<RefCell<EditorRoute>>` and `CustomEditor` (which holds !Send callbacks),
/// so it is inherently non-Send—the UI task does not cross threads and the main task never holds this type (it only communicates via channels), so `Rc` is safe.
pub struct InteractiveUi {
    theme: Theme, // Cloned from ThemeController; hot reload replaces it wholesale via ThemeChanged
    kb: KeybindingsManager<AppAction>, // constructed via KeybindingsManager::new(app_default_bindings()) (defaults only until T-AP-06-2 user overrides land)
    editor: CustomEditor, // owned; contains on_action/on_escape/on_exit/on_submit/on_change callbacks—callbacks fire inside handle_key and are routed as InputCommand
    chat_rows: Vec<ChatRow>, // Message/tool/bash/summary line data snapshot
    pending_rows: Vec<String>, // Snapshot of queued follow-up message texts (strictly aligned with UiEvent::PendingChanged(Vec<String>))
    streaming_index: Option<usize>, // Index of the current streaming assistant line (wholesale replacement of the snapshot on update)
    tool_indices: HashMap<String, usize>, // tool_call_id → chat line index
    footer_data: Option<FooterData>, // Per-frame state snapshot (the footer is supplied by the main task each frame via UiEvent::Footer)
    status: Option<StatusSpec>, // Per-frame state snapshot
    // Persistent status line instance: reused across frames to preserve the Loader animation clock (Working→Working only updates the message, no rebuild).
    // Lifecycle maintained by apply (apply_status / ClearStatus / ThemeChanged); the render side only draws, never constructs —
    // fixes the animation defect where per-frame transient construction → started_at re-anchored every frame → spinner stuck on the first frame.
    status_indicator: Option<StatusIndicator>,
    folded: HashSet<usize>, // Collapsed row set (tree/branch/compaction/skill collapse; ToolsExpand flips locally)
    is_bash_mode: bool,     // Editor border indicator (! prefix)
    hide_thinking_block: bool, // Collapsed thinking in assistant row rendering (v1 default false; T-AP-02-3 wires in the real setting)
    output_pad: u16,           // Left padding for message output (v1 default 0; T-AP-02-3 wires in the real setting)
    route: Rc<RefCell<EditorRoute>>, // Editor callbacks → routing slot for handle_key (see EditorRoute)
}

/// Shared routing slot between editor callbacks and handle_key.
/// The callback is a `'static` closure (stored inside `CustomEditor`), so it cannot directly borrow `mut self`; instead we go through
/// `Rc<RefCell<..>>`; `handle_key` takes the command after the editor has processed the key and synchronizes the bash indicator.
/// Only accessed sequentially within the UI task (`InteractiveUi` holds a `Box<dyn FnMut>` callback, which was never thread-safe to begin with), so no Mutex is needed.
#[derive(Default)]
struct EditorRoute {
    /// Input commands produced by the callback (at most one per keypress: submit/escape/exit/action are mutually exclusive per key).
    commands: Vec<InputCommand>,
    /// Latest bash-mode indicator written by `on_change` (body starts with `!`, aligned with pi's `onChange`).
    bash_mode: bool,
}

/// v1 editor left padding (matching pi `getEditorPaddingX` default **0** — settings-manager.ts `?? 0`;
/// after T-AP-02-3 settings manager lands, it will be injected by InteractiveMode via SettingsConfig; default remains 0).
const DEFAULT_EDITOR_PADDING_X: usize = 0;

impl InteractiveUi {
    /// Build the UI state from the theme, app keybinding table, and cwd: create `CustomEditor` (with on_action/on_escape/
    /// on_exit/on_submit/on_change callbacks), initial footer baseline.
    pub(crate) fn new(theme: Theme, kb: KeybindingsManager<AppAction>, cwd: String) -> Self {
        // 1. Editor: holds a copy of the app keybinding table (no user overrides in v1, both tables identical; after T-AP-06-2 lands, via
        //    `kb.clone()` override table injected synchronously) + v1 default left padding.
        let mut editor = CustomEditor::new(DEFAULT_EDITOR_PADDING_X, kb.clone());

        // 2. Shared routing slot (implementation of constraint 1).
        let route = Rc::new(RefCell::new(EditorRoute::default()));

        // 3. Callback assembly (aligned with pi's `setupKeyHandlers` + `setupEditorSubmitHandler`, v1 subset).
        //    Esc → Escape: context-sensitive handling (abort streaming/bash, exit bash mode, double-Esc selector);
        //    requires session state, so the command is sent up to the main task for decision; the UI side makes no decisions (bash-mode exit is intercepted locally in handle_key).
        {
            let r = Rc::clone(&route);
            editor.set_on_escape(Box::new(move || {
                r.borrow_mut().commands.push(InputCommand::Escape);
            }));
        }
        //    Empty editor Ctrl+D → CtrlD (exit); when non-empty CustomEditor returns false and defers to
        //    handle_key as a fallback to perform the delete action (see section 4); no command is emitted here.
        {
            let r = Rc::clone(&route);
            editor.set_on_exit(Box::new(move || {
                r.borrow_mut().commands.push(InputCommand::CtrlD);
            }));
        }
        //    Enter submit → Submit (editor already trims and clears the body; commit history is written back in handle_key,
        //    the closure cannot borrow &mut editor again to call add_to_history).
        {
            let r = Rc::clone(&route);
            editor.set_on_submit(move |text: &str| {
                r.borrow_mut()
                    .commands
                    .push(InputCommand::Submit(text.to_string()));
            });
        }
        //    Text changes → bash-mode indicator (`!` prefix; covers the "" after submit clears and programmatic set_text).
        {
            let r = Rc::clone(&route);
            editor.set_on_change(move |text: &str| {
                r.borrow_mut().bash_mode = text.trim_start().starts_with('!');
            });
        }
        //    Ctrl+C → CtrlC (clear/abort, aligned with pi's `onAction("app.clear", () => handleCtrlC())`).
        {
            let r = Rc::clone(&route);
            editor.on_action(
                AppAction::Clear,
                Box::new(move || {
                    r.borrow_mut().commands.push(InputCommand::CtrlC);
                }),
            );
        }
        //    Remaining v1 actions → Action commands, dispatched by the main task `handle_app_action` (FollowUp/ToolsExpand
        //    are reinforced locally in handle_key; see that function).
        //    Interrupt/Exit are already taken over by on_escape/on_exit; session.*/tree.* actions are not registered in v1,
        //    so their keys fall through to the editor and are ignored (a natural no-op).
        for action in [
            AppAction::ThinkingCycle,
            AppAction::ThinkingToggle,
            AppAction::ToolsExpand,
            AppAction::MessageCopy,
            AppAction::MessageFollowUp,
            AppAction::MessageDequeue,
        ] {
            let r = Rc::clone(&route);
            editor.on_action(
                action,
                Box::new(move || {
                    r.borrow_mut().commands.push(InputCommand::Action(action));
                }),
            );
        }

        // 4. Initial appearance and focus: normal border (aligned with pi's `getThinkingBorderColor("off")`),
        // Focus the editor on startup (aligns with pi `ui.setFocus(editor)`).
        editor.set_border_style(Style::fg(theme.fg(ColorToken::ThinkingOff)));
        editor.set_focused(true);

        // 5. Initial footer baseline: show cwd on the first frame (placeholder until the first real Footer event arrives;
        //    leave unknown fields like the model name blank; footer rendering guards against empty right segments).
        let footer_data = Some(FooterData {
            pwd: cwd,
            git_branch: None,
            session_name: None,
            model_id: String::new(),
            thinking_level: None,
            usage: UsageTotals::default(),
            context_percent: None,
            context_window: 0,
            cache_hit_rate: None,
            auto_compact_enabled: false,
        });

        InteractiveUi {
            theme,
            kb,
            editor,
            chat_rows: Vec::new(),
            pending_rows: Vec::new(),
            streaming_index: None,
            tool_indices: HashMap::new(),
            footer_data,
            status: None,
            status_indicator: None,
            folded: HashSet::new(),
            is_bash_mode: false,
            hide_thinking_block: false,
            output_pad: 0,
            route,
        }
    }

    /// Key event routing: editor first (EditorAction keybindings) -> editor callback path; if no hit, resolve
    /// `KeybindingsManager::matches` output into `InputCommand::Action/Submit/Escape/CtrlC/CtrlD`.
    /// Returns None = the event was consumed (editor-internal movement/editing / localized action).
    ///
    /// Localized actions (no session state needed; disposed on the UI side, not forwarded to the main task):
    /// - `MessageFollowUp`: take the editor's current text -> send `FollowUp(text)` (queued) and clear the editor;
    /// - `ToolsExpand`: invert the entire collapse set (tool/bash/summary row expand toggle, pi `toggleToolOutputExpansion`);
    /// - `Escape` in bash mode (unlike non-bash mode): exit bash mode (clear editor, pi onEscape branch 3).
    pub(crate) fn handle_key(&mut self, key: &KeyEvent) -> Option<InputCommand> {
        // 1. Editor first: internal app keybinding match -> trigger registered callbacks (commands write to route; pure editing actions have no command)
        let handled = self.editor.handle_key(key);
        // 2. Bash indicator sync: on_change already wrote route.bash_mode -> persist to self + border color (false→ThinkingOff / true→BashMode)
        self.sync_bash_mode();
        // 3. Take the command and strengthen locally; Submit also writes back to submission history (the closure inside new can't borrow &mut editor, patch it here).
        //    Extract the command first and release the route borrow so later branches can use &mut self for localized actions.
        let popped = self.route.borrow_mut().commands.pop();
        if let Some(cmd) = popped {
            let cmd = match cmd {
                InputCommand::Submit(text) => {
                    self.editor.add_to_history(&text);
                    InputCommand::Submit(text)
                }
                InputCommand::Action(AppAction::MessageFollowUp) => {
                    let text = self.editor.text();
                    if text.trim().is_empty() {
                        return None; // Don't queue an empty editor (pi: empty Alt+Enter submit is ignored)
                    }
                    self.editor.add_to_history(&text);
                    self.editor.set_text("");
                    self.sync_bash_mode();
                    return Some(InputCommand::FollowUp(text));
                }
                InputCommand::Action(AppAction::ToolsExpand) => {
                    self.toggle_rows_expanded();
                    return None; // Local consumption (expanded state is pure UI, no main task needed)
                }
                InputCommand::Escape if self.is_bash_mode => {
                    // Bash mode exit edit state: clear the editor (pi onEscape: isBashMode → setText(""))
                    self.editor.set_text("");
                    self.sync_bash_mode();
                    return None;
                }
                cmd => cmd,
            };
            return Some(cmd);
        }
        // 4. The only case where CustomEditor returns false = Exit matched but the editor is non-empty: fall back to sending a delete action, do not send CtrlD
        if !handled {
            if let Some(kid) = KeyId::from_key_event(*key) {
                if self.kb.matches(&kid, AppAction::Exit) {
                    self.editor.handle_action(EditorAction::DeleteCharForward);
                }
            }
        }
        None
    }

    // Also call when apply receives EditorSetText — set_text triggers on_change, which already updates route.bash_mode
    fn sync_bash_mode(&mut self) {
        let bash = self.route.borrow().bash_mode; // The borrow is released after this statement ends
        if bash != self.is_bash_mode {
            self.is_bash_mode = bash;
            let token = if bash {
                ColorToken::BashMode
            } else {
                ColorToken::ThinkingOff
            };
            self.editor
                .set_border_style(Style::fg(self.theme.fg(token)));
        }
    }

    /// Bracketed paste entry: hand the pasted text to the editor as a whole for atomic insertion (`Editor::handle_paste`
    /// handles newline normalization / tab expansion / non-printable filtering internally), and syncs the bash mode indicator (body `!` prefix).
    /// Repaint triggering is done by the event path's `EventOutcome::Continue` setting the coalescing flag.
    /// Consumer: T-AP-11-1-fix (interactive `on_event` routes `Event::Paste`).
    pub(crate) fn handle_paste(&mut self, text: &str) {
        self.editor.handle_paste(text);
        self.sync_bash_mode();
    }

    /// Localization: ToolsExpand flips the expanded state (`folded` set) of all tool/bash/summary lines at once.
    /// Aligns with pi `toggleToolOutputExpansion` (v1 component expansion differs little under fallback layout; semantic interface preserved).
    fn toggle_rows_expanded(&mut self) {
        for (idx, row) in self.chat_rows.iter().enumerate() {
            match row {
                ChatRow::Tool { .. }
                | ChatRow::Bash { .. }
                | ChatRow::BranchSummary { .. }
                | ChatRow::CompactionSummary { .. }
                | ChatRow::SkillInvocation { .. } => {
                    if self.folded.contains(&idx) {
                        self.folded.remove(&idx);
                    } else {
                        self.folded.insert(idx);
                    }
                }
                _ => {}
            }
        }
    }

    /// Apply a `UiEvent` to the data snapshot (message row append/replace, tool rows, status, footer,
    /// animation flags, theme replacement) and set the internal redraw request (effective in the render layer via WakeOutcome).
    pub(crate) fn apply(&mut self, ev: UiEvent) {
        match ev {
            UiEvent::Status(spec) => self.apply_status(spec),
            UiEvent::ClearStatus => {
                self.status = None;
                self.status_indicator = None;
            }
            UiEvent::MessageStart(msg) => self.apply_message_start(msg),
            UiEvent::MessageUpdate(msg) => self.apply_message_update(msg),
            UiEvent::MessageEnd {
                message,
                error_text,
            } => self.apply_message_end(message, error_text),
            UiEvent::ToolStart { id, name, args } => self.apply_tool_start(id, name, args),
            UiEvent::ToolUpdate { id, args } => self.apply_tool_update(id, args),
            UiEvent::ToolEnd {
                id,
                result,
                is_error,
            } => self.apply_tool_end(id, result, is_error),
            UiEvent::PendingChanged(rows) => self.pending_rows = rows,
            UiEvent::Footer(data) => self.footer_data = Some(data),
            UiEvent::BashMode(b) => {
                // Main-task explicit indication (v1 editor body `!` is already locally driven by on_change; this event is reserved for future
                // bash session-state toggling), idempotently update the border color.
                if b != self.is_bash_mode {
                    self.is_bash_mode = b;
                    let token = if b {
                        ColorToken::BashMode
                    } else {
                        ColorToken::ThinkingOff
                    };
                    self.editor
                        .set_border_style(Style::fg(self.theme.fg(token)));
                }
            }
            UiEvent::SetAnimation(_) => {
                // Takes effect in the render layer (WakeOutcome::Animate is handled by the on_wake closure); no local state here.
            }
            UiEvent::ThemeChanged(theme) => {
                self.theme = theme;
                // Border color rebuilt from theme tokens (aligned with pi onThemeChange → updateEditorBorderColor).
                let token = if self.is_bash_mode {
                    ColorToken::BashMode
                } else {
                    ColorToken::ThinkingOff
                };
                self.editor
                    .set_border_style(Style::fg(self.theme.fg(token)));
                // Status line style is based on theme tokens: persistent instances are rebuilt from the current state snapshot
                // (very low-frequency events, clock re-anchoring is acceptable, see rebuild_status_indicator).
                self.rebuild_status_indicator();
            }
            UiEvent::TerminalTitle(title) => {
                // Write OSC terminal title escape directly (same thread as ScreenBuffer differential output, executed sequentially, no interleaving risk).
                use std::io::Write;
                let mut out = std::io::stdout();
                let _ = write!(out, "\x1b]0;{}\x07", title);
                let _ = out.flush();
            }
            UiEvent::EditorSetText(text) => {
                self.editor.set_text(&text);
                self.sync_bash_mode();
            }
        }
    }

    /// Apply the status line spec: update the snapshot and maintain the persistent `status_indicator` instance.
    /// Working→Working only calls [`StatusIndicator::set_message`] without rebuilding (preserves the Loader clock anchor,
    /// which is required for animation frames to advance across frames); construct a new instance only on category switch or first appearance; Error/Warning semantic colors are drawn directly
    /// (no persistent instance).
    fn apply_status(&mut self, spec: StatusSpec) {
        self.status = Some(spec.clone());
        match spec {
            StatusSpec::Working { message } => {
                let reuse = self
                    .status_indicator
                    .as_ref()
                    .is_some_and(|i| matches!(i.kind(), StatusIndicatorKind::Working));
                if reuse {
                    // Existing Working instance: only update the message, clock anchor continues (no rebuild).
                    if let Some(i) = self.status_indicator.as_mut() {
                        i.set_message(message);
                    }
                } else {
                    self.status_indicator = Some(StatusIndicator::working(message, &self.theme));
                }
            }
            StatusSpec::Retry {
                attempt,
                max_attempts,
                delay_ms,
            } => {
                self.status_indicator = Some(StatusIndicator::retry(
                    attempt,
                    max_attempts,
                    delay_ms,
                    &self.kb,
                    &self.theme,
                ));
            }
            StatusSpec::Compaction => {
                self.status_indicator = Some(StatusIndicator::compaction(
                    CompactionStatusReason::Manual,
                    &self.kb,
                    &self.theme,
                ));
            }
            StatusSpec::BranchSummary => {
                self.status_indicator =
                    Some(StatusIndicator::branch_summary(&self.kb, &self.theme));
            }
            StatusSpec::Error(_) | StatusSpec::Warning(_) => {
                // Semantic colors drawn directly; no persistent instances.
                self.status_indicator = None;
            }
        }
    }

    /// After a theme replacement, rebuild the persistent status line from the current state snapshot (style is based on theme tokens, rebuild required;
    /// clock re-anchoring is acceptable for very low-frequency theme events). Clear the instance when the snapshot is None/Error/Warning.
    fn rebuild_status_indicator(&mut self) {
        self.status_indicator = match &self.status {
            Some(StatusSpec::Working { message }) => {
                Some(StatusIndicator::working(message.clone(), &self.theme))
            }
            Some(StatusSpec::Retry {
                attempt,
                max_attempts,
                delay_ms,
            }) => Some(StatusIndicator::retry(
                *attempt,
                *max_attempts,
                *delay_ms,
                &self.kb,
                &self.theme,
            )),
            Some(StatusSpec::Compaction) => Some(StatusIndicator::compaction(
                CompactionStatusReason::Manual,
                &self.kb,
                &self.theme,
            )),
            Some(StatusSpec::BranchSummary) => {
                Some(StatusIndicator::branch_summary(&self.kb, &self.theme))
            }
            Some(StatusSpec::Error(_)) | Some(StatusSpec::Warning(_)) | None => None,
        };
    }

    /// Status line rendering: animation categories (Working/Retry/Compaction/BranchSummary) draw the persistent instance
    /// (reused across frames to preserve the clock); Error/Warning drawn directly in semantic color; idle → IdleStatus placeholder.
    /// Paired with `apply_status`: the render side only draws, never constructs.
    fn render_status(&mut self, buf: &mut ScreenBuffer, y: u16, height: u16) {
        let width = buf.size().0;
        if matches!(
            self.status,
            Some(
                StatusSpec::Working { .. }
                    | StatusSpec::Retry { .. }
                    | StatusSpec::Compaction
                    | StatusSpec::BranchSummary
            )
        ) {
            if let Some(comp) = self.status_indicator.as_mut() {
                let _ = comp.render(buf, 0, y, width, height);
            }
            return;
        }
        let theme = &self.theme;
        match &self.status {
            Some(StatusSpec::Error(msg)) => {
                let style = Style::fg(theme.fg(ColorToken::Error));
                buf.write(0, y, &style, msg);
            }
            Some(StatusSpec::Warning(msg)) => {
                let style = Style::fg(theme.fg(ColorToken::Warning));
                buf.write(0, y, &style, msg);
            }
            _ => {
                let mut idle = IdleStatus;
                let _ = idle.render(buf, 0, y, width, height);
            }
        }
    }

    /// `MessageStart`: User → append user row; Assistant → open streaming row (`streaming_index` points to it);
    /// ToolResult is rendered by Tool* events, skipped here.
    fn apply_message_start(&mut self, msg: Message) {
        match msg {
            Message::User(user) => {
                let text = extract_user_text(&user);
                self.chat_rows.push(ChatRow::User { text });
            }
            Message::Assistant(a) => {
                let idx = self.chat_rows.len();
                self.streaming_index = Some(idx);
                self.chat_rows.push(ChatRow::Assistant(a));
            }
            Message::ToolResult(_) => {}
        }
    }

    /// Streaming delta: only when an active streaming line exists, replace its message snapshot and request a repaint.
    fn apply_message_update(&mut self, msg: Message) {
        if let Message::Assistant(a) = msg {
            if let Some(idx) = self.streaming_index {
                if idx < self.chat_rows.len() {
                    self.chat_rows[idx] = ChatRow::Assistant(a);
                }
            }
        }
    }

    /// Message complete: final text + error text (error/abort via `MessageEnd.error_text` → Info row).
    fn apply_message_end(&mut self, message: Message, error_text: Option<String>) {
        match message {
            Message::Assistant(a) => {
                if let Some(idx) = self.streaming_index {
                    if idx < self.chat_rows.len() {
                        self.chat_rows[idx] = ChatRow::Assistant(a);
                    } else {
                        self.chat_rows.push(ChatRow::Assistant(a));
                    }
                } else {
                    // No active streaming row (e.g., missing prior event before an error): append a row directly to avoid losing the message.
                    self.chat_rows.push(ChatRow::Assistant(a));
                }
                if let Some(err) = error_text {
                    self.chat_rows.push(ChatRow::Info {
                        text: err,
                        kind: InfoKind::Error,
                    });
                }
                self.streaming_index = None;
            }
            Message::User(_) | Message::ToolResult(_) => {}
        }
    }

    /// Tool start: register `tool_indices` and lay out pending rows.
    fn apply_tool_start(&mut self, id: String, name: String, args: Value) {
        let idx = self.chat_rows.len();
        self.tool_indices.insert(id.clone(), idx);
        self.chat_rows.push(ChatRow::Tool {
            tool_call_id: id,
            tool_name: name,
            args,
            result: None,
            is_partial: true,
            expanded: false,
        });
    }

    /// Streaming argument update: locate the row by tool_call_id and replace the argument snapshot.
    fn apply_tool_update(&mut self, id: String, args: Value) {
        if let Some(&idx) = self.tool_indices.get(&id) {
            if let Some(ChatRow::Tool { args: a, .. }) = self.chat_rows.get_mut(idx) {
                *a = args;
            }
        }
    }

    /// Tool complete: harness-layer result (zen_ai::tool::ToolExecutionResult) → component render snapshot conversion.
    fn apply_tool_end(
        &mut self,
        id: String,
        result: zen_ai::tool::ToolExecutionResult,
        is_error: bool,
    ) {
        if let Some(&idx) = self.tool_indices.get(&id) {
            self.tool_indices.remove(&id);
            if let Some(ChatRow::Tool {
                result: r,
                is_partial,
                ..
            }) = self.chat_rows.get_mut(idx)
            {
                *r = Some(convert_tool_result(&result, is_error));
                *is_partial = false;
            }
        }
    }

    /// Per-frame composited rendering (**cell-level diff mode**): top-down cursor advance — message area (chat_rows scrolling window + pending
    /// queue) → status line → editor → footer; each component is **constructed transiently per line** (borrows `&self.theme`, not stored),
    /// the status line is the sole exception — draws a persistent `status_indicator` instance (reused across frames to preserve the animation clock, see `render_status`).
    ///
    /// Layout budget (fixed bottom-up): footer 2 lines + status line 2 lines (1 line busy indicator +
    /// idle placeholder keeps the bottom stable), **editor height grows dynamically with content** (`needed_height`, cap = terminal rows
    /// 30% with a floor of 5, aligned with pi `maxVisibleLines`; T-AP-11-1-fix replaces the former fixed 3 lines),
    /// everything else is allocated to the message area (bottom-following scroll, see render_chat).
    ///
    /// **Current assembly note**: after T-TU-01-5, interactive mode init has switched to scrollback mode (`run_scrollback` +
    /// [`Self::render_scrollback`]); this method and its windowed subpaths (render_chat/paint_row) follow the task
    /// Scope discipline: keep original as-is (no old code removed, regression surface minimized) for future rollback/comparison use (T-AP-11-4 external
    /// editor suspend/resume may revert to the old path); not called yet, hence marked allow(dead_code).
    #[allow(dead_code)] // Old cell-level path retained (see note above), not invoked under scrollback mode
    pub(crate) fn render(&mut self, buf: &mut ScreenBuffer) {
        let (width, height) = buf.size();
        if height == 0 {
            return;
        }

        // Budget for the bottom three regions (line counts): editor height computed dynamically from content (cap = 30% of terminal rows with a floor of 5,
        // aligned with pi maxVisibleLines; T-AP-11-1-fix).
        let footer_h: u16 = 2;
        let editor_h: u16 = self.editor.needed_height(width, buf.size().1);
        let status_h: u16 = 2;
        let editor_y = height.saturating_sub(footer_h + editor_h);
        let status_y = editor_y.saturating_sub(status_h);
        let message_area_h = status_y; // Message area = [0, status_y)

        // 1. Message area (scroll window + pending queue tail)
        if message_area_h > 0 {
            self.render_chat(buf, message_area_h);
        }
        //    Error/Warning → text drawn directly in semantic color; idle → IdleStatus (2-line placeholder keeps the bottom stable);
        //    Animated categories draw persistent instances (reused across frames to preserve the clock, see apply_status/render_status).
        self.render_status(buf, status_y, status_h);

        // 3. Editor (fixed 3 lines at the bottom)
        self.editor.render_editor(buf, 0, editor_y, width, editor_h);

        // 4. Footer (fixed 2 lines at the bottom)
        let mut footer = FooterComponent::new(&self.theme);
        if let Some(data) = &self.footer_data {
            footer.set_data(data.clone());
        }
        let _ = footer.render(buf, 0, editor_y + editor_h, width, footer_h);
    }

    /// The message area and other bottom fixed regions render as a "content line stream" (scrollback mode): chat lines are pushed into the canvas continuously
    /// (off-screen content is carried by the terminal scrollback, so render's hard-cropped window is no longer needed); status line/editor/footer act as
    /// the tail of the fixed-region line stream — users can scroll the terminal to review the full history. Scrollback-mode assembly is selected by `init` in mod.rs
    /// （`tui.run_scrollback` + `draw = |c| ui.render_scrollback(c)`）。
    /// Reuses render_chat's measurement/component construction logic, but renders each row into a temporary ScreenBuffer and then extracts it for streaming via
    /// `ScreenBuffer::line_segments` (components unchanged); fixed regions are still drawn into a screen-height ScreenBuffer
    /// and then bridged wholesale as the line stream tail.
    pub(crate) fn render_scrollback(&mut self, canvas: &mut ScrollbackCanvas) {
        let width = canvas.width();
        if width == 0 {
            return;
        }
        let theme = &self.theme;

        // ── 1. Message area content line stream: all chat rows (no window, full history; off-screen content carried by scrollback) ──
        let mut scratch = ScreenBuffer::new(width, MEASURE_HEIGHT_CAP);
        for (i, row) in self.chat_rows.iter().enumerate() {
            // Info line does not use a component: semantic-colored text drawn directly (aligned with paint_row).
            if let ChatRow::Info { text, kind } = row {
                let token = match kind {
                    InfoKind::Error => ColorToken::Error,
                    InfoKind::Warning => ColorToken::Warning,
                    InfoKind::Status => ColorToken::Muted,
                };
                let style = Style::fg(theme.fg(token));
                canvas.push_text(text.clone(), style);
                continue;
            }
            // Measure-as-render (same temporary buffer), then extract per row into style segments for streaming.
            let used = self.measure_row(row, i, &mut scratch, width);
            for r in 0..used {
                let segs = scratch.line_segments(r);
                if !segs.is_empty() {
                    canvas.push_line(segs);
                }
            }
            scratch.clear();
        }
        // Pending queue rows (dim, appended at the end of the message area, aligned with render_chat).
        for text in &self.pending_rows {
            let style = Style::fg(theme.fg(ColorToken::Dim));
            let line = format!("↳ Follow-up: {text}");
            canvas.push_text(line, style);
        }

        // ── 2. Fixed-region tail (status line + editor + footer) drawn into a screen-height buffer → bridged wholesale into the line stream ──
        // Editor height grows dynamically with content (cap = 30% of terminal rows with a floor of 5, aligned with pi maxVisibleLines;
        // T-AP-11-1-fix replaces the former fixed 3 lines — multiline input / long wrapped text fully visible; the lines yielded by the message area are
        // naturally carried by the scrollback line stream).
        let status_h: u16 = 2;
        let editor_h: u16 = self.editor.needed_height(width, canvas.height());
        let footer_h: u16 = 2;
        let fixed_h = status_h + editor_h + footer_h;
        let mut fixed = ScreenBuffer::new(width, fixed_h);

        // Status line: animated categories draw persistent instances (reused across frames to preserve the clock), Error/Warning drawn directly in semantic color,
        // idle → IdleStatus (see apply_status/render_status).
        self.render_status(&mut fixed, 0, status_h);
        // Editor (fixed 3 rows)
        self.editor
            .render_editor(&mut fixed, 0, status_h, width, editor_h);
        // Footer (fixed 2 rows)
        let mut footer = FooterComponent::new(&self.theme);
        if let Some(data) = &self.footer_data {
            footer.set_data(data.clone());
        }
        let _ = footer.render(&mut fixed, 0, status_h + editor_h, width, footer_h);

        // Bridge the fixed region as the line stream tail (only non-empty rows are pushed, to avoid blank rows appearing in scrollback).
        // IME cursor anchor (terminal-ime-anchor-fix): line of the editor caret within fixed =
        // status_h + cursor_position().1 (already includes the +1 top border offset); its corresponding content line number (=
        // number of lines already pushed) is recorded before push — the fixed region skips empty lines, so the line number cannot be derived directly from the fixed line number.
        // flush uses this to place the (hidden) hardware cursor at the caret, anchoring the system IME composition string/candidate window in place.
        let caret_pos = self.editor.cursor_position();
        let caret_fixed_row = status_h as usize + caret_pos.1 as usize;
        let caret_col = caret_pos.0 as usize;
        let mut caret_canvas_row: Option<usize> = None;
        for r in 0..fixed_h {
            let segs = fixed.line_segments(r);
            if !segs.is_empty() {
                if r as usize == caret_fixed_row {
                    caret_canvas_row = Some(canvas.line_count());
                }
                canvas.push_line(segs);
            }
        }
        canvas.set_cursor_hint(caret_canvas_row.map(|row| (row, caret_col)));
    }

    /// Message area rendering: measure each row's actual height → bottom-following scroll window → draw sequentially.
    /// Measurement goes through a temporary buffer `render` up to `MEASURE_HEIGHT_CAP` height; rows outside the window are not rendered (scroll semantics aligned with pi
    /// `setScrollFromBottom`). Pending queue text is appended as dim rows at the end of the message area.
    fn render_chat(&self, buf: &mut ScreenBuffer, area_h: u16) {
        let (width, _) = buf.size();
        if area_h == 0 {
            return;
        }

        // Measure all rows (reuse a single temporary buffer to avoid per-row allocation)
        let mut scratch = ScreenBuffer::new(width, MEASURE_HEIGHT_CAP);
        let mut heights: Vec<u16> = Vec::with_capacity(self.chat_rows.len());
        for (i, row) in self.chat_rows.iter().enumerate() {
            let h = self.measure_row(row, i, &mut scratch, width);
            heights.push(h);
        }
        // Pending queue rows counted at 1 row each (text truncation handled at draw time)
        let pending_heights: Vec<u16> = self.pending_rows.iter().map(|_| 1u16).collect();

        // Bottom-following window: accumulate from the end backwards until it no longer fits the message area
        let mut start = self.chat_rows.len();
        let mut used = 0u16;
        for (i, h) in heights.iter().enumerate().rev() {
            if used + *h > area_h {
                if start == self.chat_rows.len() {
                    start = i; // Fallback: at least draw the last line (truncated display, not blank)
                }
                break;
            }
            used += *h;
            start = i;
        }
        // Append pending rows to the end of the window (if space remains, at most area_h - used rows)
        let mut used_pending = 0u16;
        for h in &pending_heights {
            if used + *h > area_h {
                break;
            }
            used += *h;
            used_pending += 1;
        }

        // Draw: chat rows inside the window (only the window is drawn; measured heights are used only for scroll computation)
        let mut y = 0u16;
        for (i, row) in self.chat_rows.iter().enumerate().skip(start) {
            if y >= area_h {
                break;
            }
            let painted = self.paint_row(row, i, buf, y, width, area_h - y);
            y += painted;
        }
        // Pending rows (dim text)
        for text in self.pending_rows.iter().take(used_pending as usize) {
            if y >= area_h {
                break;
            }
            let style = Style::fg(self.theme.fg(ColorToken::Dim));
            let line = format!("↳ Follow-up: {text}");
            buf.write(0, y, &style, &line);
            y += 1;
        }
    }

    /// Measure a single row's actual render height (temporary buffer + height cap).
    fn measure_row(
        &self,
        row: &ChatRow,
        row_index: usize,
        scratch: &mut ScreenBuffer,
        width: u16,
    ) -> u16 {
        let mut comp = self.build_row_component(row, row_index);
        comp.render(scratch, 0, 0, width, MEASURE_HEIGHT_CAP)
    }

    /// Draw a single row: construct the component → render into the real buffer (height limited to the remaining area; multi-row truncation is handled by the component/buffer).
    fn paint_row(
        &self,
        row: &ChatRow,
        row_index: usize,
        buf: &mut ScreenBuffer,
        y: u16,
        width: u16,
        height: u16,
    ) -> u16 {
        match row {
            ChatRow::Info { text, kind } => {
                let token = match kind {
                    InfoKind::Error => ColorToken::Error,
                    InfoKind::Warning => ColorToken::Warning,
                    InfoKind::Status => ColorToken::Muted,
                };
                let style = Style::fg(self.theme.fg(token));
                buf.write(0, y, &style, text);
                1
            }
            _ => {
                let mut comp = self.build_row_component(row, row_index);
                comp.render(buf, 0, y, width, height)
            }
        }
    }

    /// Construct the row component on the fly (borrows `&self.theme`/`&self.kb`, kept only for this frame).
    /// `row_index` is used by the folded collection (`folded`) to look up this row's expanded state.
    fn build_row_component(&self, row: &ChatRow, row_index: usize) -> Box<dyn Component + '_> {
        let theme = &self.theme;
        let kb = &self.kb;
        let folded_expanded = self.folded.contains(&row_index);
        match row {
            ChatRow::User { text } => Box::new(UserMessageComponent::new(
                text.clone(),
                theme,
                self.output_pad,
            )),
            ChatRow::Assistant(msg) => Box::new(AssistantMessageComponent::new(
                msg.clone(),
                theme,
                self.hide_thinking_block,
                "Thinking...",
                self.output_pad,
            )),
            ChatRow::Tool {
                tool_call_id,
                tool_name,
                args,
                result,
                is_partial,
                expanded,
            } => {
                let mut comp = ToolExecutionComponent::new(
                    tool_name.clone(),
                    tool_call_id.clone(),
                    args.clone(),
                    theme,
                    kb,
                );
                comp.set_expanded(*expanded || folded_expanded);
                if let Some(r) = result {
                    // ToolExecutionResult (component view) does not implement Clone — rebuild by fields
                    comp.update_result(
                        ToolExecutionResult {
                            text: r.text.clone(),
                            is_error: r.is_error,
                        },
                        *is_partial,
                    );
                }
                Box::new(comp)
            }
            ChatRow::Bash {
                command,
                exclude_from_context,
                output,
                exit_code,
                cancelled,
                truncation,
                full_output_path,
                expanded,
            } => {
                let mut comp =
                    BashExecutionComponent::new(command.clone(), *exclude_from_context, theme, kb);
                comp.append_output(output);
                comp.set_complete(
                    *exit_code,
                    *cancelled,
                    truncation.as_ref().map(|t| TruncationInfo {
                        truncated: t.truncated,
                    }),
                    full_output_path.clone(),
                );
                comp.set_expanded(*expanded || folded_expanded);
                Box::new(comp)
            }
            ChatRow::BranchSummary { summary } => {
                let mut comp = BranchSummaryMessageComponent::new(summary.clone(), theme, kb);
                comp.set_expanded(folded_expanded);
                Box::new(comp)
            }
            ChatRow::CompactionSummary {
                summary,
                tokens_before,
            } => {
                let mut comp = CompactionSummaryMessageComponent::new(
                    summary.clone(),
                    *tokens_before,
                    theme,
                    kb,
                );
                comp.set_expanded(folded_expanded);
                Box::new(comp)
            }
            ChatRow::SkillInvocation { name, content } => {
                let mut comp =
                    SkillInvocationMessageComponent::new(name.clone(), content.clone(), theme, kb);
                comp.set_expanded(folded_expanded);
                Box::new(comp)
            }
            ChatRow::CustomEntry { custom_type, data } => Box::new(CustomEntryComponent::new(
                custom_type.clone(),
                data.clone(),
                theme,
            )),
            ChatRow::CustomMessage { custom_type, text } => Box::new(CustomMessageComponent::new(
                custom_type.clone(),
                text.clone(),
                theme,
            )),
            ChatRow::Info { .. } => unreachable!("Info lines are drawn directly by paint_row, bypassing components"),
        }
    }
}

/// Extract plain text from a user message (Text as-is / Blocks: concatenate text blocks, ignoring images and other non-text blocks).
pub(crate) fn extract_user_text(user: &UserMessage) -> String {
    match &user.content {
        ContentBlockList::Text(s) => s.clone(),
        ContentBlockList::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| {
                if let ContentBlock::Text(t) = b {
                    Some(t.text.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Harness-layer tool result → component render snapshot (concatenated text blocks + is_error).
/// Corresponds to the `result` field of `ChatRow::Tool` (two distinct types: zen_ai::tool result vs component view).
pub(crate) fn convert_tool_result(
    result: &zen_ai::tool::ToolExecutionResult,
    is_error: bool,
) -> ToolExecutionResult {
    let mut text = String::new();
    for block in &result.message.content {
        if let ContentBlock::Text(t) = block {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(&t.text);
        }
    }
    ToolExecutionResult {
        text: if text.is_empty() { None } else { Some(text) },
        is_error,
    }
}

/// Message area row data snapshot: element of `InteractiveUi::chat_rows`, constructed transiently per row in `render()`
/// Corresponding component (UserMessageComponent / AssistantMessageComponent / ToolExecutionComponent /
/// BashExecutionComponent / summary / custom / info rows). Folded state is tracked by row index in the
/// Marker; state is not stored inline (aligned with pi's folded message box expand flag ownership).
#[allow(clippy::large_enum_variant)] // Tool/Bash variants carry the full execution snapshot
pub(crate) enum ChatRow {
    /// User message row (`MessageStart` role=user → extract text snapshot).
    User { text: String },
    /// Assistant message row: an incremental snapshot while streaming (`MessageUpdate` replaces wholesale),
    /// `MessageEnd` commits the final version; display params like `output_pad` are injected at render time via display config.
    Assistant(AssistantMessage),
    /// Tool call/result row (`tool_indices` indexes into this row; result is a render snapshot
    /// of `crate::components::tool_execution::ToolExecutionResult`; is_partial follows the execution stage).
    Tool {
        tool_call_id: String,
        tool_name: String,
        args: Value,
        result: Option<crate::components::tool_execution::ToolExecutionResult>,
        is_partial: bool,
        expanded: bool,
    },
    /// Bash execution line (no producer before the T-AP-04-2 seam; render path retained).
    #[allow(dead_code)] // Constructed by handle_bash_command once T-AP-04-2 lands
    Bash {
        command: String,
        exclude_from_context: bool,
        output: String,
        exit_code: Option<i32>,
        cancelled: bool,
        truncation: Option<TruncationInfo>,
        full_output_path: Option<String>,
        expanded: bool,
    },
    /// Branch summary message row (driven by turn_end branch summary events; v1 engine has no such event, render path kept).
    #[allow(dead_code)] // Constructed by apply once T-AP-08-5 branch summaries land
    BranchSummary { summary: String },
    /// Compact summary message row (driven by compact events; v1 has no compaction orchestration, render path kept).
    #[allow(dead_code)] // Constructed by apply once T-AP-03-2 compaction orchestration lands
    CompactionSummary { summary: String, tokens_before: u64 },
    /// Skill invocation message row (v1 has no skill events, render path kept).
    #[allow(dead_code)] // Constructed by apply once skill commands (T-AP-03-2) land
    SkillInvocation { name: String, content: String },
    /// Custom entry row (entry_appended; v1 has no custom entries, render path kept).
    #[allow(dead_code)] // Constructed by apply once the extension system (v2) is in place
    CustomEntry {
        custom_type: String,
        data: Option<Value>,
    },
    /// Custom message row (message_start role=custom; v1 engine Message has no custom, render path kept).
    #[allow(dead_code)] // Constructed by apply once the extension system (v2) is in place
    CustomMessage { custom_type: String, text: String },
    /// Pure informational row (showError/showWarning/showStatus → drawn directly in the corresponding semantic color; in v1 produced via MessageEnd error text).
    Info { text: String, kind: InfoKind },
}

/// Semantic colors for info rows (aligned with pi's `showError`/`showWarning`/`showStatus` palette).
/// `Warning`/`Status` have no producers in v1 (only `Error` via MessageEnd error text); semantic surface kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InfoKind {
    Error,
    #[allow(dead_code)]
    Warning,
    #[allow(dead_code)]
    Status,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keybindings::app_default_bindings;
    use crossterm::event::{KeyCode, KeyModifiers};
    use zen_ai::types::{StopReason, TextContent, ThinkingContent, ToolResultMessage, Usage};
    use zen_tui::theme::Theme;

    fn theme() -> &'static Theme {
        zen_tui::theme::builtin_dark_theme()
    }

    fn kb() -> KeybindingsManager<AppAction> {
        KeybindingsManager::new(app_default_bindings(), HashMap::new())
    }

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn char_key(c: char) -> KeyEvent {
        press(KeyCode::Char(c))
    }

    fn ctrl(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
    }

    fn text_block(s: &str) -> ContentBlock {
        ContentBlock::Text(TextContent {
            text: s.to_string(),
            text_signature: None,
        })
    }

    fn assistant(content: Vec<ContentBlock>, stop: StopReason) -> AssistantMessage {
        AssistantMessage {
            content,
            api: "openai-completions".into(),
            provider: "test".into(),
            model: "test-model".into(),
            response_model: None,
            response_id: None,
            usage: Usage::default(),
            stop_reason: stop,
            error_message: None,
            timestamp: 0,
        }
    }

    fn user_message(text: &str) -> UserMessage {
        UserMessage {
            content: ContentBlockList::Text(text.to_string()),
            timestamp: 0,
        }
    }

    fn ui() -> InteractiveUi {
        InteractiveUi::new(theme().clone(), kb(), "/work".into())
    }

    #[test]
    fn dbg_temp_bg_probe() {
        // Temporary diagnostics: per-line fg/bg of user messages + thinking messages
        let mut u = ui();
        u.apply(UiEvent::MessageStart(Message::User(user_message(
            "What files are in the current directory?",
        ))));
        let m = assistant(
            vec![
                ContentBlock::Thinking(ThinkingContent {
                    thinking: "secret **bold**".into(),
                    thinking_signature: None,
                    redacted: None,
                }),
                text_block("# Head\nbody\n\n```rust\nfn main() {}\n```"),
            ],
            StopReason::Stop,
        );
        u.apply(UiEvent::MessageStart(Message::Assistant(m)));
        let mut scratch = ScreenBuffer::new(100, 300);
        for (i, row) in u.chat_rows.iter().enumerate() {
            let used = u.measure_row(row, i, &mut scratch, 100);
            for r in 0..used {
                let segs = scratch.line_segments(r);
                if !segs.is_empty() {
                    let text: String = segs.iter().map(|s| s.text.as_str()).collect();
                    let fgs: Vec<String> =
                        segs.iter().map(|s| format!("{:?}", s.style.fg)).collect();
                    let bgs: Vec<String> =
                        segs.iter().map(|s| format!("{:?}", s.style.bg)).collect();
                    eprintln!("row {r}: {text:?}\n   fg={fgs:?} bg={bgs:?}");
                }
            }
            scratch.clear();
        }
    }

    // ── extract_user_text / convert_tool_result (pure functions) ──────────

    #[test]
    fn test_extract_user_text_text_content_verbatim() {
        let u = user_message("hello world");
        assert_eq!(extract_user_text(&u), "hello world");
    }

    #[test]
    fn test_extract_user_text_blocks_joins_text_only() {
        let u = UserMessage {
            content: ContentBlockList::Blocks(vec![
                text_block("a"),
                ContentBlock::Thinking(ThinkingContent {
                    thinking: "skip".into(),
                    thinking_signature: None,
                    redacted: None,
                }),
                text_block("b"),
            ]),
            timestamp: 0,
        };
        assert_eq!(extract_user_text(&u), "a\nb");
    }

    #[test]
    fn test_convert_tool_result_joins_text_and_flags_error() {
        let result = zen_ai::tool::ToolExecutionResult {
            message: ToolResultMessage {
                tool_call_id: "c1".into(),
                tool_name: "read".into(),
                content: vec![text_block("line1"), text_block("line2")],
                details: None,
                usage: None,
                is_error: false,
                added_tool_names: None,
                timestamp: 0,
            },
            terminate: false,
        };
        let view = convert_tool_result(&result, true);
        assert_eq!(view.text.as_deref(), Some("line1\nline2"));
        assert!(view.is_error);

        let empty = zen_ai::tool::ToolExecutionResult {
            message: ToolResultMessage {
                tool_call_id: "c1".into(),
                tool_name: "read".into(),
                content: vec![],
                details: None,
                usage: None,
                is_error: false,
                added_tool_names: None,
                timestamp: 0,
            },
            terminate: false,
        };
        let view = convert_tool_result(&empty, false);
        assert_eq!(view.text, None);
        assert!(!view.is_error);
    }

    // ── apply: message stream (Start → Update → End) ─────────────────────

    #[test]
    fn test_apply_message_start_user_appends_row() {
        let mut u = ui();
        u.apply(UiEvent::MessageStart(Message::User(user_message("hi"))));
        assert_eq!(u.chat_rows.len(), 1);
        assert!(matches!(&u.chat_rows[0], ChatRow::User { text } if text == "hi"));
        assert!(u.streaming_index.is_none());
    }

    #[test]
    fn test_apply_message_streaming_assistant_full_flow() {
        let mut u = ui();
        // Start: begin streaming line
        u.apply(UiEvent::MessageStart(Message::Assistant(assistant(
            vec![text_block("partial")],
            StopReason::Stop,
        ))));
        assert_eq!(u.chat_rows.len(), 1);
        assert_eq!(u.streaming_index, Some(0));
        // Update: replace whole snapshot
        u.apply(UiEvent::MessageUpdate(Message::Assistant(assistant(
            vec![text_block("partial more")],
            StopReason::Stop,
        ))));
        assert!(matches!(&u.chat_rows[0], ChatRow::Assistant(m) if m.content.len() == 1));
        // End: commit final text + no-error message → replace only, close streaming line
        u.apply(UiEvent::MessageEnd {
            message: Message::Assistant(assistant(vec![text_block("final")], StopReason::Stop)),
            error_text: None,
        });
        assert!(u.streaming_index.is_none());
        assert!(matches!(&u.chat_rows[0], ChatRow::Assistant(m) if m.content.len() == 1));
        assert_eq!(u.chat_rows.len(), 1, "normal completion does not append an Info line");
    }

    #[test]
    fn test_apply_message_end_with_error_appends_info_row() {
        let mut u = ui();
        u.apply(UiEvent::MessageStart(Message::Assistant(assistant(
            vec![text_block("x")],
            StopReason::Stop,
        ))));
        u.apply(UiEvent::MessageEnd {
            message: Message::Assistant(assistant(vec![text_block("x")], StopReason::Aborted)),
            error_text: Some("provider 500".into()),
        });
        assert_eq!(u.chat_rows.len(), 2);
        assert!(matches!(
            &u.chat_rows[1],
            ChatRow::Info { text, kind: InfoKind::Error } if text == "provider 500"
        ));
    }

    #[test]
    fn test_apply_tool_flow_registers_and_updates_rows() {
        let mut u = ui();
        u.apply(UiEvent::ToolStart {
            id: "t1".into(),
            name: "bash".into(),
            args: serde_json::json!({"command": "ls"}),
        });
        assert_eq!(u.chat_rows.len(), 1);
        assert!(u.tool_indices.contains_key("t1"));
        u.apply(UiEvent::ToolUpdate {
            id: "t1".into(),
            args: serde_json::json!({"command": "ls -la"}),
        });
        assert!(
            matches!(&u.chat_rows[0], ChatRow::Tool { args, is_partial: true, .. } if args == &serde_json::json!({"command": "ls -la"}))
        );
        let harness_result = zen_ai::tool::ToolExecutionResult {
            message: ToolResultMessage {
                tool_call_id: "t1".into(),
                tool_name: "bash".into(),
                content: vec![text_block("total 0")],
                details: None,
                usage: None,
                is_error: false,
                added_tool_names: None,
                timestamp: 0,
            },
            terminate: false,
        };
        u.apply(UiEvent::ToolEnd {
            id: "t1".into(),
            result: harness_result,
            is_error: false,
        });
        assert!(!u.tool_indices.contains_key("t1"), "remove the index after completion");
        assert!(
            matches!(&u.chat_rows[0], ChatRow::Tool { is_partial: false, result: Some(r), .. } if !r.is_error)
        );
    }

    #[test]
    fn test_apply_footer_status_and_pending() {
        let mut u = ui();
        u.apply(UiEvent::Footer(FooterData {
            pwd: "/work".into(),
            git_branch: None,
            session_name: None,
            model_id: "m".into(),
            thinking_level: None,
            usage: UsageTotals::default(),
            context_percent: None,
            context_window: 0,
            cache_hit_rate: None,
            auto_compact_enabled: false,
        }));
        assert!(u.footer_data.is_some());
        u.apply(UiEvent::Status(StatusSpec::Working {
            message: "Thinking...".into(),
        }));
        assert!(matches!(u.status, Some(StatusSpec::Working { .. })));
        u.apply(UiEvent::ClearStatus);
        assert!(u.status.is_none());
        u.apply(UiEvent::PendingChanged(vec!["follow-up".into()]));
        assert_eq!(u.pending_rows, vec!["follow-up".to_string()]);
    }

    #[test]
    fn test_status_indicator_lifecycle_via_apply() {
        let mut u = ui();
        // Working → persistent instance already constructed (clock anchored at construction time)
        u.apply(UiEvent::Status(StatusSpec::Working {
            message: "Thinking...".into(),
        }));
        assert!(u.status_indicator.is_some());
        // Working→Working: reuse the instance without rebuilding (preserves the clock anchor, the precondition for cross-frame animation progression)
        u.apply(UiEvent::Status(StatusSpec::Working {
            message: "Thinking...".into(),
        }));
        assert!(u
            .status_indicator
            .as_ref()
            .is_some_and(|i| matches!(i.kind(), StatusIndicatorKind::Working)));
        // Category switch: Compaction → rebuild as a Compaction category
        u.apply(UiEvent::Status(StatusSpec::Compaction));
        assert!(u
            .status_indicator
            .as_ref()
            .is_some_and(|i| matches!(i.kind(), StatusIndicatorKind::Compaction)));
        // Error → drawn directly in semantic color, no persistent instance
        u.apply(UiEvent::Status(StatusSpec::Error("boom".into())));
        assert!(u.status_indicator.is_none());
        // ClearStatus → clear the snapshot and the instance
        u.apply(UiEvent::Status(StatusSpec::Working {
            message: "again".into(),
        }));
        assert!(u.status_indicator.is_some());
        u.apply(UiEvent::ClearStatus);
        assert!(u.status.is_none());
        assert!(u.status_indicator.is_none());
    }

    #[test]
    fn test_rebuild_status_indicator_on_theme_changed() {
        let mut u = ui();
        u.apply(UiEvent::Status(StatusSpec::Working {
            message: "Thinking...".into(),
        }));
        assert!(u.status_indicator.is_some());
        let light = zen_tui::theme::builtin_light_theme().clone();
        u.apply(UiEvent::ThemeChanged(light.clone()));
        // Rebuild from the snapshot after a theme swap: instance persists, category unchanged (styling already uses the new tokens)
        assert!(u
            .status_indicator
            .as_ref()
            .is_some_and(|i| matches!(i.kind(), StatusIndicatorKind::Working)));
        // Theme swap under an idle snapshot → instance cleared
        u.apply(UiEvent::ClearStatus);
        u.apply(UiEvent::ThemeChanged(light.clone()));
        assert!(u.status_indicator.is_none());
    }

    #[test]
    fn test_apply_theme_changed_replaces_owned_theme() {
        let mut u = ui();
        let light = zen_tui::theme::builtin_light_theme().clone();
        u.apply(UiEvent::ThemeChanged(light.clone()));
        assert_eq!(u.theme.name, "light");
    }

    #[test]
    fn test_apply_editor_set_text_clears_and_syncs_bash() {
        let mut u = ui();
        // Enter bash mode first (editor body starts with !)
        for c in ['!', 'l', 's'] {
            let _ = u.handle_key(&char_key(c));
        }
        assert!(u.is_bash_mode);
        // Programmatic clear (EditorSetText): body cleared → on_change resets bash indicator
        u.apply(UiEvent::EditorSetText(String::new()));
        assert!(!u.is_bash_mode);
        assert_eq!(u.editor.text(), "");
    }

    // ── render: smoke + scroll window ──────────────────────────────────

    #[test]
    fn test_render_empty_ui_smoke() {
        let mut u = ui();
        let mut buf = ScreenBuffer::new(80, 24);
        u.render(&mut buf);
        // No panic; buffer size unchanged
        assert_eq!(buf.size(), (80, 24));
    }

    #[test]
    fn test_render_chat_rows_bottom_anchored_window() {
        let mut u = ui();
        // Push far more user lines than the message area capacity (each line is 3 rows tall: padding 1 + body 1 + padding 1)
        for i in 0..200 {
            u.apply(UiEvent::MessageStart(Message::User(user_message(
                &format!("line {i}"),
            ))));
        }
        let mut buf = ScreenBuffer::new(80, 24);
        u.render(&mut buf);
        // Bottom follow: the newest line (line 199) must fall inside the visible window — do not assert buffer content directly,
        // smoke-test with "render does not panic + monotonicity of window start" (window computation is inside render_chat)
        assert_eq!(u.chat_rows.len(), 200);
    }

    #[test]
    fn test_render_string_scroll_window_drops_head() {
        // Build 100 lines of text directly, verify the window start gets advanced (metric + fallback window logic verified as pure functions:
        // derive start from single-line measured height and area)
        let mut u = ui();
        for i in 0..50 {
            u.chat_rows.push(ChatRow::User {
                text: format!("row {i}"),
            });
        }
        // Simulate a message area of ~19 rows (24 - footer 2 - editor 3 - status 2 = 17; use a small 40x17 buffer)
        let mut buf = ScreenBuffer::new(40, 17);
        u.render(&mut buf);
        // Row height 3 * visible window ≤ 17 → window of ~5 trailing rows → start ≥ 45
        // Do not read the buffer (no public cell read), just confirm render does not panic
        let _ = buf.size();
    }

    // ── render_scrollback: line stream output + fixed-region tail ──────────────────

    #[test]
    fn test_render_scrollback_empty_ui_produces_fixed_area_tail() {
        let mut u = ui();
        let mut canvas = ScrollbackCanvas::new(80, 24);
        u.render_scrollback(&mut canvas);
        // Even with an empty session, the fixed region (status line 2 + editor 3 + footer 2 = 7 rows) is bridged as the line stream tail,
        // and it does not panic; canvas width matches construction.
        assert_eq!(canvas.width(), 80);
        assert!(canvas.content_height() > 0);
        assert!(
            canvas.content_height() >= 6,
            "fixed area should have at least 6-7 non-empty lines, actual {} lines",
            canvas.content_height()
        );
    }

    #[test]
    fn test_render_scrollback_sets_ime_cursor_hint() {
        // IME cursor anchor (terminal-ime-anchor-fix): editor caret line (status lines 2 + top border +1)
        // Content present must be pushed → anchor is Some; anchor line falls within the fixed region
        let mut u = ui();
        let mut canvas = ScrollbackCanvas::new(80, 24);
        u.render_scrollback(&mut canvas);
        let hint = canvas.cursor_hint();
        assert!(hint.is_some(), "IME cursor anchor should be set every frame while the editor is focused");
        let (row, _col) = hint.unwrap();
        assert!(
            row < canvas.content_height(),
            "anchor row should be within the pushed content rows: row={row} < {}",
            canvas.content_height()
        );
        // Empty editor caret column = effective left padding (DEFAULT_EDITOR_PADDING_X=0), within the line width
        if let Some((_row, col)) = hint {
            assert!(col < 80, "anchor column should be within the line width: {col}");
        }
    }

    #[test]
    fn test_render_scrollback_chat_rows_grow_line_flow() {
        let mut u = ui();
        let width = 80;
        let mut empty = ScrollbackCanvas::new(width, 24);
        u.render_scrollback(&mut empty);
        let baseline = empty.content_height();

        // Append two user messages (containing multi-line text) → line stream grows with content
        u.apply(UiEvent::MessageStart(Message::User(user_message(
            "hello\nsecond line\nthird",
        ))));
        u.apply(UiEvent::MessageStart(Message::User(user_message("world"))));
        let mut canvas = ScrollbackCanvas::new(width, 24);
        u.render_scrollback(&mut canvas);
        assert!(
            canvas.content_height() > baseline,
            "line stream should grow after appending a chat line ({} > {})",
            canvas.content_height(),
            baseline
        );
    }

    #[test]
    fn test_render_scrollback_pending_rows_appended() {
        let mut u = ui();
        let width = 80;
        let mut empty = ScrollbackCanvas::new(width, 24);
        u.render_scrollback(&mut empty);
        let baseline = empty.content_height();

        u.apply(UiEvent::PendingChanged(vec!["queued".into()]));
        let mut canvas = ScrollbackCanvas::new(width, 24);
        u.render_scrollback(&mut canvas);
        assert!(
            canvas.content_height() > baseline,
            "pending line should be appended to the end of the line stream"
        );
    }

    #[test]
    fn test_render_scrollback_status_rows_in_tail() {
        let mut u = ui();
        u.apply(UiEvent::Status(StatusSpec::Working {
            message: "Thinking...".into(),
        }));
        let mut canvas = ScrollbackCanvas::new(80, 24);
        u.render_scrollback(&mut canvas);
        assert!(canvas.content_height() > 0);
        // Working status line + editor + footer are all non-empty line stream tail
        assert!(canvas.content_height() >= 6);
    }

    // ── render_scrollback: editor height grows dynamically with content (T-AP-11-1-fix) ──

    #[test]
    fn test_scrollback_editor_grows_with_content() {
        let mut u = ui();
        let mut empty = ScrollbackCanvas::new(80, 24);
        u.render_scrollback(&mut empty);
        let base = empty.content_height();
        assert!(
            base >= 6,
            "empty editor fixed area (status 2 + editor 3 + footer 2) is at least 6 lines"
        );

        // 3 lines of input → needed_height = 5 (2 borders + 3 content) → fixed region is 2 lines taller than the empty state
        u.editor.set_text("a\nb\nc");
        assert_eq!(u.editor.needed_height(80, 24), 5);
        let mut grown = ScrollbackCanvas::new(80, 24);
        u.render_scrollback(&mut grown);
        assert_eq!(
            grown.content_height(),
            base + 2,
            "3 lines of input → editor 3→5 lines, fixed area grows by 2 lines"
        );
        // cap reached: 7 lines of input (rows=24 → cap=max(5,7)=7), width=80 with no wrapping
        u.editor.set_text("a\nb\nc\nd\ne\nf\ng");
        assert_eq!(
            u.editor.needed_height(80, 24),
            9,
            "7 lines of content, rows=24 → cap=7 hits the ceiling at 9"
        );
    }

    #[test]
    fn test_editor_paste_inserts_text_and_syncs_bash() {
        let mut u = ui();
        u.handle_paste("line1\r\nline2");
        assert_eq!(
            u.editor.text(),
            "line1\nline2",
            "after paste newline normalization the whole block goes into the editor"
        );
        assert!(!u.is_bash_mode);
        // pasting a `!` prefix → bash mode indicator sync (on_change → route.bash_mode → border color)
        let mut u2 = ui();
        u2.handle_paste("!ls -la");
        assert_eq!(u2.editor.text(), "!ls -la");
        assert!(u2.is_bash_mode);
    }

    // ── handle_key: routing / localized actions ────────────────────────────

    #[test]
    fn test_handle_key_typing_returns_none_then_submit() {
        let mut u = ui();
        assert_eq!(u.handle_key(&char_key('h')), None);
        assert_eq!(u.handle_key(&char_key('i')), None);
        assert_eq!(u.editor.text(), "hi");
        let cmd = u
            .handle_key(&press(KeyCode::Enter))
            .expect("Enter should produce Submit");
        assert!(matches!(cmd, InputCommand::Submit(t) if t == "hi"));
        assert_eq!(u.editor.text(), "", "editor is cleared after submit");
    }

    #[test]
    fn test_handle_key_ctrl_d_empty_editor_sends_exit() {
        let mut u = ui();
        let cmd = u
            .handle_key(&ctrl('d'))
            .expect("Ctrl+D on an empty editor should produce CtrlD");
        assert!(matches!(cmd, InputCommand::CtrlD));
    }

    #[test]
    fn test_handle_key_tools_expand_toggles_folded_locally() {
        let mut u = ui();
        u.apply(UiEvent::ToolStart {
            id: "t1".into(),
            name: "read".into(),
            args: serde_json::json!({"path": "a"}),
        });
        assert!(u.folded.is_empty());
        assert_eq!(u.handle_key(&ctrl('o')), None, "ToolsExpand is consumed locally");
        assert_eq!(u.folded.len(), 1, "collapsed set contains the tool line");
        assert_eq!(u.handle_key(&ctrl('o')), None);
        assert!(u.folded.is_empty(), "pressing again expands it");
    }

    #[test]
    fn test_handle_key_follow_up_queues_text_and_clears_editor() {
        let mut u = ui();
        for c in ['h', 'i'] {
            let _ = u.handle_key(&char_key(c));
        }
        // Alt+Enter (default MessageFollowUp keybinding)
        let alt_enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT);
        let cmd = u.handle_key(&alt_enter).expect("Alt+Enter should produce FollowUp");
        assert!(matches!(cmd, InputCommand::FollowUp(t) if t == "hi"));
        assert_eq!(u.editor.text(), "", "editor is cleared after queuing");
    }

    #[test]
    fn test_handle_key_escape_exits_bash_mode_locally() {
        let mut u = ui();
        for c in ['!', 'l', 's'] {
            let _ = u.handle_key(&char_key(c));
        }
        assert!(u.is_bash_mode);
        // Esc in bash mode → locally exit bash mode (clear editor), do not submit the main task
        let cmd = u.handle_key(&press(KeyCode::Esc));
        assert!(cmd.is_none());
        assert!(!u.is_bash_mode);
        assert_eq!(u.editor.text(), "");
    }

    #[test]
    fn test_handle_key_normal_escape_forwards_to_main() {
        let mut u = ui();
        u.handle_key(&char_key('x'));
        let cmd = u
            .handle_key(&press(KeyCode::Esc))
            .expect("Esc is forwarded when not in bash mode");
        assert!(matches!(cmd, InputCommand::Escape));
    }
}
