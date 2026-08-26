// REQ-APP-11 (pi-358): assistant-message — full assistant message rendering.
// Corresponds to original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/assistant-message.ts.
// pi's Container child rebuild (updateContent) is simplified in zen to "plan blocks (plan) at render time, then write them out block by block":
// Message snapshot owned (aligned with pi lastMessage); streaming updates are replaced wholesale via update_content.
// OSC133 terminal chunk markers v1 are not emitted (ScreenBuffer diff model has no start-of-line control sequence injection, see open issue 1).

use zen_ai::types::{AssistantMessage, ContentBlock, StopReason};
use zen_tui::components::Component;
use zen_tui::highlight::StyledSegment;
use zen_tui::markdown::{render_markdown, MarkdownTheme};
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{ColorToken, Theme};

/// Plan block: rendering unit the message body is split into (equivalent of pi's addChild sequence).
#[allow(clippy::large_enum_variant)] // Markdown carries the full MarkdownTheme (15 styles)
enum Planned {
    /// Empty line (Gap).
    Gap,
    /// Single-line label (thinking collapse label / stop-reason error line).
    Label { text: String, style: Style },
    /// markdown segment (text block / thinking expanded block, each with independent theme).
    Markdown { text: String, theme: MarkdownTheme },
}

/// Full assistant message rendering component: arranges text / thinking blocks and the stop-reason error line in order—
/// T-AP-11-1 created/updated per assistant message (including streaming updates); the user sees the chat bubble directly.
///
/// Behavior (aligned with pi):
/// - leading empty line when there is visible content; for each consecutive thinking run when thinking is collapsed (hide_thinking_block)
///   Show only one line `hidden_thinking_label` (thinkingText + italic); when expanded, render with derived theme
///   (thinkingText foreground + italic baseline);
/// - consecutive thinking blocks merged into one markdown segment; append an empty line after the segment if further visible content follows;
/// - always append an error line when stop_reason = Length; append an error line for Aborted/Error when there is no toolCall;
/// - toolCall blocks are excluded from the body (rendered separately by ToolExecutionComponent).
pub struct AssistantMessageComponent<'a> {
    message: AssistantMessage,
    hide_thinking_block: bool,
    hidden_thinking_label: String,
    output_pad: u16,
    base_md: MarkdownTheme,
    thinking_md: MarkdownTheme,
    theme: &'a Theme,
}

impl<'a> AssistantMessageComponent<'a> {
    /// Constructed from the message snapshot, theme, and display parameters; `hidden_thinking_label` defaults to "Thinking...".
    pub fn new(
        message: AssistantMessage,
        theme: &'a Theme,
        hide_thinking_block: bool,
        hidden_thinking_label: impl Into<String>,
        output_pad: u16,
    ) -> Self {
        let base_md = theme.markdown_theme();
        let mut thinking_md = base_md.clone();
        thinking_md.text = theme.style(
            ColorToken::ThinkingText,
            None,
            &[crossterm::style::Attribute::Italic],
        );
        AssistantMessageComponent {
            message,
            hide_thinking_block,
            hidden_thinking_label: hidden_thinking_label.into(),
            output_pad,
            base_md,
            thinking_md,
            theme,
        }
    }

    /// During streaming, T-AP-11-1 fully refreshes content with the latest partial snapshot (pi updateContent).
    pub fn update_content(&mut self, message: AssistantMessage) {
        self.message = message;
    }

    /// Toggle thinking collapse (effective point for /settings hide-thinking or app.thinking.toggle).
    pub fn set_hide_thinking_block(&mut self, hide: bool) {
        self.hide_thinking_block = hide;
    }

    /// Update the collapsed label text.
    pub fn set_hidden_thinking_label(&mut self, label: impl Into<String>) {
        self.hidden_thinking_label = label.into();
    }

    /// Update the left padding of the output.
    pub fn set_output_pad(&mut self, padding: u16) {
        self.output_pad = padding;
    }

    /// Reference to the current message snapshot (used by T-AP-11-1 to determine toolCall/stopReason).
    pub fn message(&self) -> &AssistantMessage {
        &self.message
    }

    /// Whether toolCall blocks are present (TS-level hasToolCalls determination; consumer: own stop-reason branch decision).
    fn has_tool_calls(&self) -> bool {
        self.message
            .content
            .iter()
            .any(|c| matches!(c, ContentBlock::ToolCall(_)))
    }

    /// Stop-reason error line (present only when there is no toolCall and reason is not length; always present for length).
    fn stop_line(&self) -> Option<String> {
        match self.message.stop_reason {
            StopReason::Length => Some(
                "Error: Model stopped because it reached the maximum output token limit. The response may be incomplete."
                    .to_string(),
            ),
            StopReason::Aborted if !self.has_tool_calls() => {
                let msg = self.message.error_message.clone().unwrap_or_default();
                Some(if msg.is_empty() || msg == "Request was aborted" {
                    "Operation aborted".to_string()
                } else {
                    msg
                })
            }
            StopReason::Error if !self.has_tool_calls() => {
                let msg = self
                    .message
                    .error_message
                    .clone()
                    .unwrap_or_else(|| "Unknown error".to_string());
                Some(format!("Error: {msg}"))
            }
            _ => None,
        }
    }

    /// Whether a content block is "visible" (non-empty text / non-empty thinking) — pi hasVisibleContent determination.
    fn block_visible(c: &ContentBlock) -> bool {
        match c {
            ContentBlock::Text(t) => !t.text.trim().is_empty(),
            ContentBlock::Thinking(t) => !t.thinking.trim().is_empty(),
            _ => false,
        }
    }

    fn plan_blocks(&self) -> Vec<Planned> {
        let mut plan: Vec<Planned> = Vec::new();

        let has_visible = self.message.content.iter().any(Self::block_visible);
        if has_visible {
            plan.push(Planned::Gap);
        }

        let mut i = 0;
        let content = &self.message.content;
        while i < content.len() {
            match &content[i] {
                ContentBlock::Text(t) if !t.text.trim().is_empty() => {
                    plan.push(Planned::Markdown {
                        text: t.text.trim().to_string(),
                        theme: self.base_md.clone(),
                    });
                    i += 1;
                }
                ContentBlock::Thinking(_) => {
                    let mut run: Vec<String> = Vec::new();
                    while i < content.len() {
                        match &content[i] {
                            ContentBlock::Thinking(t) => {
                                let s = t.thinking.trim();
                                if !s.is_empty() {
                                    run.push(s.to_string());
                                }
                                i += 1;
                            }
                            _ => break,
                        }
                    }
                    if run.is_empty() {
                        continue;
                    }
                    let has_after = content[i..].iter().any(Self::block_visible);
                    if self.hide_thinking_block {
                        let style = self.theme.style(
                            ColorToken::ThinkingText,
                            None,
                            &[crossterm::style::Attribute::Italic],
                        );
                        plan.push(Planned::Label {
                            text: self.hidden_thinking_label.clone(),
                            style,
                        });
                    } else {
                        plan.push(Planned::Markdown {
                            text: run.join("\n\n"),
                            theme: self.thinking_md.clone(),
                        });
                    }
                    if has_after {
                        plan.push(Planned::Gap);
                    }
                }
                _ => {
                    // Image / ToolCall blocks are excluded from body rendering
                    i += 1;
                }
            }
        }

        if let Some(line) = self.stop_line() {
            plan.push(Planned::Gap);
            plan.push(Planned::Label {
                text: line,
                style: Style::fg(self.theme.fg(ColorToken::Error)),
            });
        }

        plan
    }

    /// Write plan blocks into the buffer (shared by render and tests).
    fn paint_plan(
        &self,
        buf: &mut ScreenBuffer,
        x: u16,
        y: u16,
        width: u16,
        height: u16,
        plan: &[Planned],
    ) -> u16 {
        let mut row = y;
        let mut painted = 0u16;
        let mut remaining = height;

        for block in plan {
            if remaining == 0 {
                break;
            }
            match block {
                Planned::Gap => {
                    buf.fill(x, row, width, 1, &Style::default());
                    painted += 1;
                    row += 1;
                    remaining -= 1;
                }
                Planned::Label { text, style } => {
                    buf.fill(x, row, width, 1, &Style::default());
                    buf.write(x + self.output_pad, row, style, text);
                    painted += 1;
                    row += 1;
                    remaining -= 1;
                }
                Planned::Markdown { text, theme } => {
                    let content_width = width.saturating_sub(self.output_pad * 2).max(1) as usize;
                    let rows = render_markdown(theme, text, content_width);
                    if rows.is_empty() {
                        continue;
                    }
                    let n = (rows.len() as u16).min(remaining);
                    for (i, r) in rows.iter().take(n as usize).enumerate() {
                        write_row(buf, x, row + i as u16, width, self.output_pad, r);
                    }
                    painted += n;
                    row += n;
                    remaining -= n;
                }
            }
        }
        painted
    }
}

/// Single-line segment write (left padding + segment sequence), for markdown lines of assistant messages.
fn write_row(
    buf: &mut ScreenBuffer,
    x: u16,
    y: u16,
    width: u16,
    padding: u16,
    segments: &[StyledSegment],
) {
    buf.write(x, y, &Style::default(), &" ".repeat(padding as usize));
    let mut col = x + padding;
    for seg in segments {
        let w = zen_tui::screen::visible_width(&seg.text) as u16;
        if col + w > width {
            buf.write(col, y, &seg.style, &seg.text);
            return;
        }
        buf.write(col, y, &seg.style, &seg.text);
        col += w;
    }
}

impl Component for AssistantMessageComponent<'_> {
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        let plan = self.plan_blocks();
        self.paint_plan(buf, x, y, width, height, &plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zen_ai::types::{AssistantMessage, TextContent, ThinkingContent, Usage};

    fn theme() -> &'static Theme {
        zen_tui::theme::builtin_dark_theme()
    }

    fn msg(content: Vec<ContentBlock>, stop: StopReason, error: Option<&str>) -> AssistantMessage {
        AssistantMessage {
            content,
            api: "openai-completions".into(),
            provider: "deepseek".into(),
            model: "m".into(),
            response_model: None,
            response_id: None,
            usage: Usage::default(),
            stop_reason: stop,
            error_message: error.map(|s| s.to_string()),
            timestamp: 0,
        }
    }

    fn text_block(s: &str) -> ContentBlock {
        ContentBlock::Text(TextContent {
            text: s.to_string(),
            text_signature: None,
        })
    }

    fn thinking_block(s: &str) -> ContentBlock {
        ContentBlock::Thinking(ThinkingContent {
            thinking: s.to_string(),
            thinking_signature: None,
            redacted: None,
        })
    }

    fn flush_text(buf: &mut ScreenBuffer) -> String {
        let mut out = Vec::new();
        buf.flush(&mut out).unwrap();
        crate::components::strip_ansi(&String::from_utf8_lossy(&out))
            .chars()
            .filter(|c| !c.is_control())
            .collect()
    }

    #[test]
    fn empty_message_renders_nothing() {
        let mut comp = AssistantMessageComponent::new(
            msg(vec![], StopReason::Stop, None),
            theme(),
            false,
            "Thinking...",
            1,
        );
        let mut buf = ScreenBuffer::new(40, 10);
        assert_eq!(comp.render(&mut buf, 0, 0, 40, 10), 0);
    }

    #[test]
    fn renders_text_block() {
        let mut comp = AssistantMessageComponent::new(
            msg(vec![text_block("hello world")], StopReason::Stop, None),
            theme(),
            false,
            "Thinking...",
            1,
        );
        let mut buf = ScreenBuffer::new(40, 10);
        let painted = comp.render(&mut buf, 0, 0, 40, 10);
        // gap + 1 md line
        assert_eq!(painted, 2);
        assert!(flush_text(&mut buf).contains("hello world"));
    }

    #[test]
    fn thinking_shown_by_default() {
        let mut comp = AssistantMessageComponent::new(
            msg(
                vec![thinking_block("secret chain of thought")],
                StopReason::Stop,
                None,
            ),
            theme(),
            false,
            "Thinking...",
            0,
        );
        let mut buf = ScreenBuffer::new(60, 10);
        comp.render(&mut buf, 0, 0, 60, 10);
        assert!(flush_text(&mut buf).contains("secret chain of thought"));
    }

    #[test]
    fn thinking_hidden_shows_label() {
        let mut comp = AssistantMessageComponent::new(
            msg(
                vec![thinking_block("secret"), thinking_block(" more")],
                StopReason::Stop,
                None,
            ),
            theme(),
            true,
            "Thinking...",
            0,
        );
        let mut buf = ScreenBuffer::new(40, 10);
        comp.render(&mut buf, 0, 0, 40, 10);
        let text = flush_text(&mut buf);
        assert!(text.contains("Thinking..."));
        assert!(!text.contains("secret"));
    }

    #[test]
    fn length_stop_shows_error_always() {
        // toolCall presence does not suppress the length error line
        let m = msg(
            vec![ContentBlock::ToolCall(zen_ai::types::ToolCall {
                id: "1".into(),
                name: "bash".into(),
                arguments: serde_json::json!({}),
                thought_signature: None,
            })],
            StopReason::Length,
            None,
        );
        let mut comp = AssistantMessageComponent::new(m, theme(), false, "Thinking...", 0);
        let mut buf = ScreenBuffer::new(80, 10);
        comp.render(&mut buf, 0, 0, 80, 10);
        assert!(flush_text(&mut buf).contains("maximum output token"));
    }

    #[test]
    fn aborted_without_tool_calls_shows_abort_message() {
        let m = msg(
            vec![text_block("partial")],
            StopReason::Aborted,
            Some("Request was aborted"),
        );
        let mut comp = AssistantMessageComponent::new(m, theme(), false, "Thinking...", 0);
        let mut buf = ScreenBuffer::new(60, 10);
        comp.render(&mut buf, 0, 0, 60, 10);
        assert!(flush_text(&mut buf).contains("Operation aborted"));
    }

    #[test]
    fn aborted_with_tool_calls_is_silent() {
        let m = msg(
            vec![ContentBlock::ToolCall(zen_ai::types::ToolCall {
                id: "1".into(),
                name: "bash".into(),
                arguments: serde_json::json!({}),
                thought_signature: None,
            })],
            StopReason::Aborted,
            Some("Request was aborted"),
        );
        let mut comp = AssistantMessageComponent::new(m, theme(), false, "Thinking...", 0);
        let mut buf = ScreenBuffer::new(60, 10);
        comp.render(&mut buf, 0, 0, 60, 10);
        assert!(!flush_text(&mut buf).contains("aborted"));
    }

    #[test]
    fn error_stop_shows_error_line() {
        let m = msg(vec![], StopReason::Error, Some("boom"));
        let mut comp = AssistantMessageComponent::new(m, theme(), false, "Thinking...", 0);
        let mut buf = ScreenBuffer::new(60, 10);
        comp.render(&mut buf, 0, 0, 60, 10);
        assert!(flush_text(&mut buf).contains("Error: boom"));
    }

    #[test]
    fn update_content_replaces_snapshot() {
        let mut comp = AssistantMessageComponent::new(
            msg(vec![text_block("old")], StopReason::Stop, None),
            theme(),
            false,
            "Thinking...",
            0,
        );
        comp.update_content(msg(vec![text_block("new text")], StopReason::Stop, None));
        let mut buf = ScreenBuffer::new(60, 10);
        comp.render(&mut buf, 0, 0, 60, 10);
        let text = flush_text(&mut buf);
        assert!(text.contains("new text"));
        assert!(!text.contains("old"));
    }

    #[test]
    fn set_hide_thinking_and_label() {
        let mut comp = AssistantMessageComponent::new(
            msg(vec![thinking_block("x")], StopReason::Stop, None),
            theme(),
            false,
            "Thinking...",
            0,
        );
        comp.set_hide_thinking_block(true);
        comp.set_hidden_thinking_label("Thinking");
        assert!(comp.hide_thinking_block);
        assert_eq!(comp.hidden_thinking_label, "Thinking");
        let mut buf = ScreenBuffer::new(60, 10);
        comp.render(&mut buf, 0, 0, 60, 10);
        let t = flush_text(&mut buf);
        eprintln!("HIDDEN-LABEL flush={t:?}");
        assert!(t.contains("Thinking"));
    }

    #[test]
    fn blocks_respect_height_budget() {
        let mut comp = AssistantMessageComponent::new(
            msg(
                vec![text_block("a\nb\nc\nd\ne"), text_block("f\ng")],
                StopReason::Stop,
                None,
            ),
            theme(),
            false,
            "Thinking...",
            0,
        );
        let mut buf = ScreenBuffer::new(40, 10);
        // Height budget 2 → draw only the first 2 lines (gap + 1 md line)
        let painted = comp.render(&mut buf, 0, 0, 40, 2);
        assert_eq!(painted, 2);
    }
}
