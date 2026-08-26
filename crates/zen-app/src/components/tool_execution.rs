// REQ-APP-11 (pi-358): tool-execution — tool call/result display block.
// Corresponds to the original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/tool-execution.ts.
// v1 has no renderer system (renderCall/renderResult/renderShell dissolved, see T-AP-11-2-implemented.md integration 5):
// Unified fallback layout (aligned with pi's formatToolExecution): title + argument JSON + text output, background color switches by phase.

use crossterm::style::Attribute;
use serde_json::Value;
use zen_tui::components::Component;
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{BgToken, ColorToken, Theme};

/// Result input view for ToolExecutionComponent (render-only snapshot extracted from zen-ai `ToolResultMessage`).
/// Pure data of the same subject: text output (text blocks in content concatenated), is_error.
pub struct ToolExecutionResult {
    /// Result text output (None if no text).
    pub text: Option<String>,
    /// Whether execution failed (error background color).
    pub is_error: bool,
}

/// Execution phase (determines background color: pending while running / error on failure / success on success).
#[derive(Clone, Copy, PartialEq)]
enum ToolPhase {
    Pending,
    Error,
    Success,
}

/// Tool call/result display block: title + args + output, background color switches by phase (pending/success/error)—
/// T-AP-11-1 driven on ToolExecutionStart/Update/MessageEnd events; v1 has no renderer system,
/// unified fallback layout (aligned with pi's createCallFallback/formatToolExecution).
pub struct ToolExecutionComponent<'a> {
    tool_name: String,
    tool_call_id: String,
    args: Value,
    theme: &'a Theme,
    expanded: bool,
    result: Option<ToolExecutionResult>,
    is_partial: bool,
}

impl<'a> ToolExecutionComponent<'a> {
    /// Constructed from tool name, call id, argument JSON, and theme (initial phase = pending).
    pub fn new(tool_name: String, tool_call_id: String, args: Value, theme: &'a Theme) -> Self {
        ToolExecutionComponent {
            tool_name,
            tool_call_id,
            args,
            theme,
            expanded: false,
            result: None,
            is_partial: true,
        }
    }

    /// Update tool args (streaming partial args scenario).
    pub fn update_args(&mut self, args: Value) {
        self.args = args;
    }

    /// Mark tool execution start (execution state of pi's renderer system; v1 fallback layout does not distinguish, keeps API contract).
    pub fn mark_execution_started(&mut self) {}

    /// Mark args as complete (argsComplete of pi's renderer system; v1 fallback layout does not distinguish, keeps API contract).
    pub fn set_args_complete(&mut self) {}

    /// Toggle expanded state (where app.tools.expand takes effect; under current fallback layout args/output are always shown, semantic interface retained).
    pub fn set_expanded(&mut self, expanded: bool) {
        self.expanded = expanded;
    }

    /// Current expanded state.
    pub fn is_expanded(&self) -> bool {
        self.expanded
    }

    /// Result backfill: set is_partial / is_error / text output, re-layout background and output (pi updateResult).
    pub fn update_result(&mut self, result: ToolExecutionResult, is_partial: bool) {
        self.result = Some(result);
        self.is_partial = is_partial;
    }

    /// Current phase (for rendering background).
    fn phase(&self) -> ToolPhase {
        if self.is_partial {
            ToolPhase::Pending
        } else if self.result.as_ref().is_some_and(|r| r.is_error) {
            ToolPhase::Error
        } else {
            ToolPhase::Success
        }
    }

    /// Tool name.
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    /// Call id.
    pub fn tool_call_id(&self) -> &str {
        &self.tool_call_id
    }

    /// Assemble display text (title + args JSON + text output; aligned with pi formatToolExecution).
    fn format_content(&self) -> String {
        let mut text = self.tool_name.clone();
        if !matches!(self.args, Value::Null) {
            let content = serde_json::to_string_pretty(&self.args).unwrap_or_default();
            if !content.is_empty() {
                text.push_str("\n\n");
                text.push_str(&content);
            }
        }
        if let Some(Some(output)) = self.result.as_ref().map(|r| r.text.as_deref()) {
            if !output.is_empty() {
                text.push('\n');
                text.push_str(output);
            }
        }
        text
    }
}

impl Component for ToolExecutionComponent<'_> {
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        let bg_token = match self.phase() {
            ToolPhase::Pending => BgToken::ToolPendingBg,
            ToolPhase::Error => BgToken::ToolErrorBg,
            ToolPhase::Success => BgToken::ToolSuccessBg,
        };
        let bg = Style::bg(self.theme.bg(bg_token));
        let theme = self.theme;

        // Display text → expand into content lines (padding 1: content width = width - 2)
        let content = self.format_content();
        let content_width = width.saturating_sub(2).max(1) as usize;
        let mut rows: Vec<String> = Vec::new();

        let mut first = true;
        for logical in content.split('\n') {
            let wrapped = if first {
                // First line = title (toolTitle bold), if not wrapped use the whole line directly
                vec![logical.to_string()]
            } else {
                zen_tui::wrap::word_wrap_line(logical, content_width)
                    .into_iter()
                    .map(|c| c.text)
                    .collect()
            };
            rows.extend(wrapped);
            first = false;
        }

        let painted = (rows.len() as u16).min(height);
        if painted == 0 {
            return 0;
        }

        // Background + content line (1-column indent; title uses toolTitle+bold, rest uses toolOutput)
        buf.fill(x, y, width, painted, &bg);
        let title_style = theme.style(ColorToken::ToolTitle, None, &[Attribute::Bold]);
        let output_style = Style::fg(theme.fg(ColorToken::ToolOutput));
        for (i, row_text) in rows.iter().take(painted as usize).enumerate() {
            let style = if i == 0 { &title_style } else { &output_style };
            buf.write(x + 1, y + i as u16, style, row_text);
        }
        painted
    }

    fn handle_key(&mut self, _key: &crossterm::event::KeyEvent) -> bool {
        // Expand/collapse keys are routed by the application layer (T-AP-11-1) (this component's fallback layout always shows everything; no need to consume them).
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn theme() -> &'static Theme {
        zen_tui::theme::builtin_dark_theme()
    }

    fn flush_text(buf: &mut ScreenBuffer) -> String {
        // flush output contains crossterm diff commands (MoveTo/SetColors/Print per cell);
        // here control bytes are stripped to restore "visible text" (printable chars, order preserved) for contains assertions
        let mut out = Vec::new();
        buf.flush(&mut out).unwrap();
        crate::components::strip_ansi(&String::from_utf8_lossy(&out))
            .chars()
            .filter(|c| !c.is_control())
            .collect()
    }

    #[test]
    fn renders_title_args_and_output() {
        let mut comp = ToolExecutionComponent::new(
            "bash".into(),
            "call-1".into(),
            json!({"command": "ls"}),
            theme(),
        );
        comp.update_result(
            ToolExecutionResult {
                text: Some("file1\nfile2".into()),
                is_error: false,
            },
            false,
        );
        let mut buf = ScreenBuffer::new(60, 10);
        let painted = comp.render(&mut buf, 0, 0, 60, 10);
        assert!(painted >= 4);
        let text = flush_text(&mut buf);
        assert!(text.contains("bash"));
        assert!(text.contains("ls"));
        assert!(text.contains("file1"));
        assert!(text.contains("file2"));
    }

    #[test]
    fn null_args_omit_params_block() {
        let mut comp =
            ToolExecutionComponent::new("read".into(), "c2".into(), Value::Null, theme());
        let mut buf = ScreenBuffer::new(60, 10);
        comp.render(&mut buf, 0, 0, 60, 10);
        assert_eq!(comp.format_content(), "read");
    }

    #[test]
    fn accessors() {
        let mut comp = ToolExecutionComponent::new("bash".into(), "c1".into(), json!({}), theme());
        assert_eq!(comp.tool_name(), "bash");
        assert_eq!(comp.tool_call_id(), "c1");
        comp.mark_execution_started();
        comp.set_args_complete();
        comp.set_expanded(true);
        assert!(comp.is_expanded());
    }

    #[test]
    fn update_result_switches_phase() {
        let mut comp = ToolExecutionComponent::new("bash".into(), "c1".into(), json!({}), theme());
        assert!(matches!(comp.phase(), ToolPhase::Pending));
        comp.update_result(
            ToolExecutionResult {
                text: Some("ok".into()),
                is_error: false,
            },
            false,
        );
        assert!(matches!(comp.phase(), ToolPhase::Success));
        comp.update_result(
            ToolExecutionResult {
                text: Some("boom".into()),
                is_error: true,
            },
            false,
        );
        assert!(matches!(comp.phase(), ToolPhase::Error));
    }

    #[test]
    fn handle_key_not_consumed() {
        let mut comp = ToolExecutionComponent::new("bash".into(), "c1".into(), json!({}), theme());
        let a = crossterm::event::KeyEvent::from(crossterm::event::KeyCode::Char('a'));
        assert!(!comp.handle_key(&a));
    }
}
