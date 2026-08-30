// REQ-APP-11 (pi-358): tool-execution — tool call/result display block.
// Corresponds to the original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/tool-execution.ts.
// v1 has no renderer system (renderCall/renderResult/renderShell dissolved, see T-AP-11-2-implemented.md integration 5):
// Unified fallback layout (aligned with pi's formatToolExecution): title + argument JSON + text output, background color switches by phase.
// tool-output-preview-fix: when not expanded, output collapses to the last PREVIEW_LINES lines + a key hint
// (the key is resolved via KeybindingsManager for the current ToolsExpand binding, dim keycap segment, aligned with pi keyHint).

use crate::components::keybinding_hints::key_text;
use crate::keybindings::AppAction;
use crossterm::style::Attribute;
use serde_json::Value;
use zen_tui::components::Component;
use zen_tui::highlight::StyledSegment;
use zen_tui::keybindings::KeybindingsManager;
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{BgToken, ColorToken, Theme};

/// Number of preview lines shown for output when not expanded (aligned with pi `BASH_PREVIEW_LINES = 5`, bash.ts:174; for tail semantics see
/// `slice(-maxVisualLines)` in `truncateToVisualLines`).
const PREVIEW_LINES: usize = 5;

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
    /// Keybinding map: resolves the current key for `AppAction::ToolsExpand` (including user overrides, aligned with pi keyHint),
    /// rendered into the preview hint line (dim keycap segment).
    kb: &'a KeybindingsManager<AppAction>,
    expanded: bool,
    result: Option<ToolExecutionResult>,
    is_partial: bool,
}

impl<'a> ToolExecutionComponent<'a> {
    /// Constructs from tool name, call id, arguments JSON, theme, and keybinding map (initial phase = pending).
    pub fn new(
        tool_name: String,
        tool_call_id: String,
        args: Value,
        theme: &'a Theme,
        kb: &'a KeybindingsManager<AppAction>,
    ) -> Self {
        ToolExecutionComponent {
            tool_name,
            tool_call_id,
            args,
            theme,
            kb,
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

    /// Toggles the expanded state (where app.tools.expand takes effect; when not expanded, output collapses to a PREVIEW_LINES-line preview; when expanded, renders in full).
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
        let mut text = self.header_text();
        if let Some(output) = self.output_text() {
            text.push('\n');
            text.push_str(output);
        }
        text
    }

    /// Heading + arguments JSON segments (always shown, not part of the collapsed preview).
    fn header_text(&self) -> String {
        let mut text = self.tool_name.clone();
        if !matches!(self.args, Value::Null) {
            let content = serde_json::to_string_pretty(&self.args).unwrap_or_default();
            if !content.is_empty() {
                text.push_str("\n\n");
                text.push_str(&content);
            }
        }
        text
    }

    /// Result text output (None when there is no text or it is empty).
    fn output_text(&self) -> Option<&str> {
        self.result
            .as_ref()
            .and_then(|r| r.text.as_deref())
            .filter(|s| !s.is_empty())
    }

    /// Body after the collapsed preview (excluding the hint line): when not expanded and the output exceeds `PREVIEW_LINES` lines, the output takes the **last**
    /// N lines (aligned with the tail semantics of pi `truncateToVisualLines`); when expanded or for short output, returns everything.
    /// Returns `(body, skipped line count)` — when `skipped > 0`, render appends a hint line
    /// `... (N earlier lines, ctrl+o to expand)` (key dim, aligned with pi keyHint).
    fn folded_body(&self) -> (String, usize) {
        let Some(output) = self.output_text() else {
            return (self.format_content(), 0);
        };
        let lines: Vec<&str> = output.lines().collect();
        if self.expanded || lines.len() <= PREVIEW_LINES {
            return (self.format_content(), 0);
        }
        let skipped = lines.len() - PREVIEW_LINES;
        let kept = lines[skipped..].join("\n");
        let mut text = self.header_text();
        text.push('\n');
        text.push_str(&kept);
        (text, skipped)
    }

    /// Background color for the current phase (box background merge: heading/output/hint-line glyph cells share this background from one source, single-point mapping).
    fn bg_token(&self) -> BgToken {
        match self.phase() {
            ToolPhase::Pending => BgToken::ToolPendingBg,
            ToolPhase::Error => BgToken::ToolErrorBg,
            ToolPhase::Success => BgToken::ToolSuccessBg,
        }
    }

    /// Expand/collapse hint segment (`... (N earlier lines, ⌘key to expand)`, key dim),
    /// aligned with pi `keyHint` (keybinding-hints.ts:42-43: dim key + muted description) + bash.ts:250-252.
    /// The segment style carries `self.bg_token()` (box background merge, same source as content lines — glyph cell bg = phase background,
    /// so the hint line does not show the default background; see the T-AP-11-1-fix "box background merge" precedent).
    fn expand_hint_segments(&self, skipped: usize) -> Vec<StyledSegment> {
        let theme = self.theme;
        let bg_token = self.bg_token();
        let key = key_text(self.kb, AppAction::ToolsExpand);
        let muted = theme.style(ColorToken::Muted, Some(bg_token), &[]);
        vec![
            StyledSegment {
                style: muted.clone(),
                text: format!("... ({skipped} earlier lines, "),
            },
            StyledSegment {
                style: theme.style(ColorToken::Dim, Some(bg_token), &[]),
                text: key,
            },
            StyledSegment {
                style: muted,
                text: " to expand)".to_string(),
            },
        ]
    }
}

impl Component for ToolExecutionComponent<'_> {
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        let bg_token = self.bg_token();
        let bg = Style::bg(self.theme.bg(bg_token));

        // Display body (output already collapsed to preview when not expanded) → expand into content lines (padding 1: content width = width - 2)
        let (content, skipped) = self.folded_body();
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

        let has_hint = skipped > 0;
        let total_rows = rows.len() as u16 + u16::from(has_hint);
        let painted = total_rows.min(height);
        if painted == 0 {
            return 0;
        }

        // Background + content line (1-column indent; title uses toolTitle+bold, the rest use toolOutput;
        // text style carries box background bg_token — glyphs match the panel background, fixing glyphs showing the default backdrop color)
        buf.fill(x, y, width, painted, &bg);
        let title_style =
            self.theme
                .style(ColorToken::ToolTitle, Some(bg_token), &[Attribute::Bold]);
        let output_style = self
            .theme
            .style(ColorToken::ToolOutput, Some(bg_token), &[]);
        for (i, row_text) in rows.iter().take(painted as usize).enumerate() {
            let style = if i == 0 { &title_style } else { &output_style };
            buf.write(x + 1, y + i as u16, style, row_text);
        }

        // Preview hint line (`... (N earlier lines, ctrl+o to expand)`, key dim): pad 1 indent,
        // always one line after the content lines; not drawn when the over-height area is clipped by painted.
        if has_hint && (rows.len() as u16) < painted {
            let segs = self.expand_hint_segments(skipped);
            let row_y = y + rows.len() as u16;
            let mut col = x + 1;
            for seg in segs {
                let w = zen_tui::screen::visible_width(&seg.text) as u16;
                if col + w > width {
                    break;
                }
                buf.write(col, row_y, &seg.style, &seg.text);
                col += w;
            }
        }
        painted
    }

    fn handle_key(&mut self, _key: &crossterm::event::KeyEvent) -> bool {
        // The expand/collapse key is routed by the application layer (T-AP-11-1) (this component only switches between preview/full rendering via set_expanded).
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keybindings::app_default_bindings;
    use serde_json::json;
    use std::collections::HashMap;

    fn theme() -> &'static Theme {
        zen_tui::theme::builtin_dark_theme()
    }

    fn kb() -> KeybindingsManager<AppAction> {
        KeybindingsManager::new(app_default_bindings(), HashMap::new())
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
        let kb = kb();
        let mut comp = ToolExecutionComponent::new(
            "bash".into(),
            "call-1".into(),
            json!({"command": "ls"}),
            theme(),
            &kb,
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
        let kb = kb();
        let mut comp =
            ToolExecutionComponent::new("read".into(), "c2".into(), Value::Null, theme(), &kb);
        let mut buf = ScreenBuffer::new(60, 10);
        comp.render(&mut buf, 0, 0, 60, 10);
        assert_eq!(comp.format_content(), "read");
    }

    #[test]
    fn accessors() {
        let kb = kb();
        let mut comp =
            ToolExecutionComponent::new("bash".into(), "c1".into(), json!({}), theme(), &kb);
        assert_eq!(comp.tool_name(), "bash");
        assert_eq!(comp.tool_call_id(), "c1");
        comp.mark_execution_started();
        comp.set_args_complete();
        comp.set_expanded(true);
        assert!(comp.is_expanded());
    }

    #[test]
    fn update_result_switches_phase() {
        let kb = kb();
        let mut comp =
            ToolExecutionComponent::new("bash".into(), "c1".into(), json!({}), theme(), &kb);
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
        let kb = kb();
        let mut comp =
            ToolExecutionComponent::new("bash".into(), "c1".into(), json!({}), theme(), &kb);
        let a = crossterm::event::KeyEvent::from(crossterm::event::KeyCode::Char('a'));
        assert!(!comp.handle_key(&a));
    }

    #[test]
    fn tool_text_cells_carry_phase_bg() {
        // Box background merge: tool panel glyph cell bg = stage background (success=ToolSuccessBg dark green), no longer showing the default backdrop color
        let kb = kb();
        let mut comp =
            ToolExecutionComponent::new("bash".into(), "c1".into(), json!({}), theme(), &kb);
        comp.update_result(
            ToolExecutionResult {
                text: Some("ok".into()),
                is_error: false,
            },
            false,
        );
        assert!(matches!(comp.phase(), ToolPhase::Success));
        let expected = theme().bg(BgToken::ToolSuccessBg);
        let mut buf = ScreenBuffer::new(60, 10);
        comp.render(&mut buf, 0, 0, 60, 10);
        let title_row = buf.line_segments(0);
        assert!(
            title_row
                .iter()
                .any(|s| s.text.contains("bash") && s.style.bg == Some(expected)),
            "Tool panel title glyphs should land on the stage background color"
        );
    }

    // ── Collapsed preview (tool-output-preview-fix: when not expanded, output takes the last N lines + a key hint) ──

    fn ten_line_output() -> String {
        (1..=10)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn unexpanded_long_output_renders_preview_and_hint() {
        // not expanded + 10 lines of output (> PREVIEW_LINES=5) → heading (1) + last 5 lines + hint line (1) = 7 lines
        let kb = kb();
        let mut comp =
            ToolExecutionComponent::new("read".into(), "c1".into(), Value::Null, theme(), &kb);
        comp.update_result(
            ToolExecutionResult {
                text: Some(ten_line_output()),
                is_error: false,
            },
            false,
        );
        assert!(!comp.is_expanded());
        let mut buf = ScreenBuffer::new(60, 30);
        let painted = comp.render(&mut buf, 0, 0, 60, 30);
        assert_eq!(painted, 7, "title + last 5 lines + hint = 7 lines");
        let text = flush_text(&mut buf);
        assert!(
            text.contains("line6") && text.contains("line10"),
            "Keep the last 5 lines"
        );
        // Assert by whole line (line1 is a substring of line10; contains cannot be used)
        let shown: Vec<&str> = text.lines().collect();
        assert!(
            !shown.contains(&"line1") && !shown.contains(&"line5"),
            "Drop the first 5 lines (tail semantics)"
        );
        // Hint line includes skipped count and real keybinding (default binding ctrl+o), aligned with pi keyHint
        assert!(text.contains("5 earlier lines"));
        assert!(text.contains("ctrl+o") && text.contains("to expand"));
    }

    #[test]
    fn expanded_long_output_renders_full() {
        // Expanded → full render (heading 1 + 10 lines), no hint line
        let kb = kb();
        let mut comp =
            ToolExecutionComponent::new("bash".into(), "c1".into(), Value::Null, theme(), &kb);
        comp.set_expanded(true);
        comp.update_result(
            ToolExecutionResult {
                text: Some(ten_line_output()),
                is_error: false,
            },
            false,
        );
        assert!(comp.is_expanded());
        let mut buf = ScreenBuffer::new(60, 30);
        let painted = comp.render(&mut buf, 0, 0, 60, 30);
        assert_eq!(painted, 11, "expanded = title + all 10 lines");
        let text = flush_text(&mut buf);
        assert!(text.contains("line1") && text.contains("line10"));
        assert!(!text.contains("to expand"), "no hint line after expanding");
    }

    #[test]
    fn unexpanded_short_output_renders_full_without_hint() {
        // Collapsed + output ≤ 5 lines → full render, no hint line
        let kb = kb();
        let mut comp =
            ToolExecutionComponent::new("read".into(), "c1".into(), Value::Null, theme(), &kb);
        comp.update_result(
            ToolExecutionResult {
                text: Some("a\nb\nc".into()),
                is_error: false,
            },
            false,
        );
        let mut buf = ScreenBuffer::new(60, 30);
        let painted = comp.render(&mut buf, 0, 0, 60, 30);
        assert_eq!(painted, 4, "title + 3 lines = 4 lines");
        let text = flush_text(&mut buf);
        assert!(text.contains("a") && text.contains("b") && text.contains("c"));
        assert!(!text.contains("to expand"));
    }

    #[test]
    fn preview_hint_row_segments_muted_text_dim_key() {
        // Hint line has three segments: muted text + dim keybinding + muted description (aligned with pi keyHint)
        let kb = kb();
        let mut comp =
            ToolExecutionComponent::new("read".into(), "c1".into(), Value::Null, theme(), &kb);
        comp.update_result(
            ToolExecutionResult {
                text: Some(ten_line_output()),
                is_error: false,
            },
            false,
        );
        let mut buf = ScreenBuffer::new(60, 30);
        comp.render(&mut buf, 0, 0, 60, 30);
        let hint = buf.line_segments(6); // Line 7 = hint line (heading 0 + last 5 lines 1..=5)
        let muted = theme().fg(ColorToken::Muted);
        let dim = theme().fg(ColorToken::Dim);
        assert!(
            hint.iter()
                .any(|s| s.text.contains("earlier lines") && s.style.fg == Some(muted)),
            "hint text segment should use muted foreground"
        );
        assert!(
            hint.iter()
                .any(|s| s.text.trim() == "ctrl+o" && s.style.fg == Some(dim)),
            "key segment should use dim foreground"
        );
        // Box background merge: hint line glyph cell bg = phase background color (success=ToolSuccessBg), does not show through default background
        // (Regression: segment refactor mistakenly used Style::fg to set only foreground — same root defect as T-AP-11-1-fix box background merge)
        let expected_bg = theme().bg(BgToken::ToolSuccessBg);
        assert!(
            hint.iter().any(|s| s.style.bg == Some(expected_bg)),
            "hint line glyph cells should land on the stage background color"
        );
    }
}
