// REQ-APP-11 (pi-358): assistant-message — full assistant message rendering.
// Corresponds to original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/assistant-message.ts.
// pi's Container child rebuild (updateContent) is simplified in zen to "plan blocks (plan) at render time, then write them out block by block":
// Message snapshot owned (aligned with pi lastMessage); streaming updates are replaced wholesale via update_content.
// OSC133 terminal chunk markers v1 are not emitted (ScreenBuffer diff model has no start-of-line control sequence injection, see open issue 1).

use crossterm::style::{Attribute, Color};
use zen_ai::types::{AssistantMessage, ContentBlock, StopReason};
use zen_tui::components::Component;
use zen_tui::highlight::StyledSegment;
use zen_tui::markdown::{render_markdown, render_markdown_with_override, MarkdownTheme};
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{ColorToken, Theme};

/// Plan block: rendering unit the message body is split into (equivalent of pi's addChild sequence).
#[allow(clippy::large_enum_variant)] // Markdown carries the full MarkdownTheme (15 styles)
enum Planned {
    /// Empty line (Gap).
    Gap,
    /// Single-line label (thinking collapse label / stop-reason error line).
    Label { text: String, style: Style },
    /// markdown segment (text block, base theme).
    Markdown { text: String, theme: MarkdownTheme },
    /// thinking expansion segment: carries the base theme (including syntax_theme_name, code blocks inside thinking still get syntax highlighting),
    /// paint_plan renders the text stream with segment-level overrides (thinkingText gray + italic) (T-TU-02-1-fix).
    ThinkingMarkdown { text: String, theme: MarkdownTheme },
}

/// Full assistant message rendering component: arranges text / thinking blocks and the stop-reason error line in order—
/// T-AP-11-1 created/updated per assistant message (including streaming updates); the user sees the chat bubble directly.
///
/// Behavior (aligned with pi):
/// - leading empty line when there is visible content; for each consecutive thinking run when thinking is collapsed (hide_thinking_block)
///   Show only one line of `hidden_thinking_label` (thinkingText + italic); use paragraph-level override when expanded
///   render (text-stream gray + italic; code blocks/structural elements keep original color, aligning with pi default style semantics — T-TU-02-1-fix);
/// - consecutive thinking blocks merged into one markdown segment; append an empty line after the segment if further visible content follows;
/// - always append an error line when stop_reason = Length; append an error line for Aborted/Error when there is no toolCall;
/// - toolCall blocks are excluded from the body (rendered separately by ToolExecutionComponent).
pub struct AssistantMessageComponent<'a> {
    message: AssistantMessage,
    hide_thinking_block: bool,
    hidden_thinking_label: String,
    output_pad: u16,
    base_md: MarkdownTheme,
    /// Recolor thinking foreground (single color ColorToken::ThinkingText, reuse via override closure;
    /// the collapsed label style is also constructed from this, see plan_blocks).
    thinking_fg: Color,
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
        let thinking_fg = theme.fg(ColorToken::ThinkingText);
        AssistantMessageComponent {
            message,
            hide_thinking_block,
            hidden_thinking_label: hidden_thinking_label.into(),
            output_pad,
            base_md,
            thinking_fg,
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
                        let style = Style {
                            fg: Some(self.thinking_fg),
                            bg: None,
                            attributes: vec![Attribute::Italic],
                        };
                        plan.push(Planned::Label {
                            text: self.hidden_thinking_label.clone(),
                            style,
                        });
                    } else {
                        plan.push(Planned::ThinkingMarkdown {
                            text: run.join("\n\n"),
                            theme: self.base_md.clone(),
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
                    // Wrap long text (e.g., API error bodies) by display width to avoid single-line silent truncation by ScreenBuffer
                    let content_width = width.saturating_sub(self.output_pad * 2).max(1) as usize;
                    let chunks = zen_tui::wrap::word_wrap_line(text, content_width);
                    let mut n = 0u16;
                    for chunk in &chunks {
                        if n >= remaining {
                            break;
                        }
                        buf.fill(x, row + n, width, 1, &Style::default());
                        if !chunk.text.is_empty() {
                            buf.write(x + self.output_pad, row + n, style, &chunk.text);
                        }
                        n += 1;
                        painted += 1;
                    }
                    row += n;
                    remaining = remaining.saturating_sub(n);
                }
                Planned::Markdown { text, theme } | Planned::ThinkingMarkdown { text, theme } => {
                    let content_width = width.saturating_sub(self.output_pad * 2).max(1) as usize;
                    let rows = if matches!(block, Planned::ThinkingMarkdown { .. }) {
                        // thinking: paragraph-level override (text-stream gray + italic; code blocks/structural elements keep original color,
                        // see the render_markdown_with_override docs).
                        let ov = thinking_override(self.thinking_fg);
                        render_markdown_with_override(theme, text, content_width, Some(&ov))
                    } else {
                        render_markdown(theme, text, content_width)
                    };
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

/// Thinking paragraph-level recolor closure: foreground fixed to `thinking_fg` (single color thinkingText) and forces italic,
/// other attributes (bold/underline etc.) are preserved — aligns with pi `defaultTextStyle { color, italic }` semantics.
/// Idempotent (reapplying to the same paragraph yields the same result); consumer: `paint_plan` thinking branch.
fn thinking_override(thinking_fg: Color) -> impl Fn(Style) -> Style {
    move |mut s: Style| {
        s.fg = Some(thinking_fg);
        if !s.attributes.contains(&Attribute::Italic) {
            s.attributes.push(Attribute::Italic);
        }
        s
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
        zen_core::ansi::strip_ansi(&String::from_utf8_lossy(&out))
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
    fn long_error_label_wraps_instead_of_clipping() {
        // Render over-wide error text with wrapping (no lost characters): at 40 columns a long sentence should wrap to multiple lines, with trailing marker visible
        let long = format!("API error: status 400: {}TAILMARK", "word ".repeat(40));
        let mut comp = AssistantMessageComponent::new(
            msg(vec![], StopReason::Error, Some(long.as_str())),
            theme(),
            false,
            "Thinking...",
            0,
        );
        let mut buf = ScreenBuffer::new(40, 20);
        let painted = comp.render(&mut buf, 0, 0, 40, 20);
        assert!(painted >= 3, "Long errors should wrap onto multiple lines, actual {painted}");
        assert!(
            flush_text(&mut buf).contains("TAILMARK"),
            "Trailing marker should be visible (not truncated to a single line)"
        );
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

    // ── Thinking paragraph-level override (T-TU-02-1-fix) ──

    /// Get the first segment of row row in the buffer (skip space-placeholder segments).
    fn first_seg(buf: &ScreenBuffer, row: u16) -> zen_tui::highlight::StyledSegment {
        buf.line_segments(row)
            .into_iter()
            .find(|s| !s.text.trim().is_empty())
            .expect("Line should contain a non-empty segment")
    }

    #[test]
    fn thinking_plan_uses_thinkingmarkdown_with_base_theme() {
        let comp = AssistantMessageComponent::new(
            msg(vec![thinking_block("secret")], StopReason::Stop, None),
            theme(),
            false,
            "Thinking...",
            0,
        );
        let plan = comp.plan_blocks();
        let thinking = plan
            .iter()
            .find_map(|p| match p {
                Planned::ThinkingMarkdown { theme, .. } => Some(theme),
                _ => None,
            })
            .expect("Expanding thinking should produce a ThinkingMarkdown segment");
        // Carries the base theme (including syntax_theme_name → code blocks inside thinking are still syntax-colored per theme)
        assert_eq!(thinking.syntax_theme_name, comp.base_md.syntax_theme_name);
    }

    #[test]
    fn thinking_text_flow_gray_italic() {
        let mut comp = AssistantMessageComponent::new(
            msg(
                vec![thinking_block("plain **bold**")],
                StopReason::Stop,
                None,
            ),
            theme(),
            false,
            "Thinking...",
            0,
        );
        let mut buf = ScreenBuffer::new(80, 10);
        comp.render(&mut buf, 0, 0, 80, 10);
        let tt = theme().fg(ColorToken::ThinkingText);
        // Row 1 = thinking text stream (row 0 = Gap)
        let plain = first_seg(&buf, 1);
        assert_eq!(
            plain.style.fg,
            Some(tt),
            "thinking text foreground should be ThinkingText"
        );
        assert!(
            plain.style.attributes.contains(&Attribute::Italic),
            "thinking text should be forced italic"
        );
        let bold = buf
            .line_segments(1)
            .into_iter()
            .find(|s| s.text.contains("bold"))
            .expect("Bold segment");
        assert_eq!(bold.style.fg, Some(tt));
        assert!(bold.style.attributes.contains(&Attribute::Bold));
        assert!(bold.style.attributes.contains(&Attribute::Italic));
    }

    #[test]
    fn thinking_code_block_keeps_syntax_colors() {
        let mut comp = AssistantMessageComponent::new(
            msg(
                vec![thinking_block("```rust\nfn main() {}\n```")],
                StopReason::Stop,
                None,
            ),
            theme(),
            false,
            "Thinking...",
            0,
        );
        let mut buf = ScreenBuffer::new(80, 10);
        comp.render(&mut buf, 0, 0, 80, 10);
        let tt = theme().fg(ColorToken::ThinkingText);
        // Rows 1/2/3 = ```rust / code lines / ``` (row 0 = Gap)
        let code_row = buf.line_segments(2);
        assert!(
            code_row
                .iter()
                .any(|s| s.style.fg.is_some() && s.style.fg != Some(tt)),
            "thinking code block should retain syntax coloring (foreground not ThinkingText)"
        );
        assert!(
            !code_row
                .iter()
                .any(|s| s.style.attributes.contains(&Attribute::Italic)),
            "thinking code block should not have italic overridden"
        );
    }
}
