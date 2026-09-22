// REQ-APP-11 (pi-358): bash-execution — bash command execution display block.
// Corresponds to the original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/bash-execution.ts.
// Output finalization: context truncation (2000 lines/50KB, shared module zen_core::truncate, T-AG-06-3)
// + preview 20 visual lines (visual-truncate). ANSI stripping is a private submodule implementation
// (Use its module once T-AP-09-8 lands.)

use crate::components::dynamic_border::DynamicBorder;
use crate::components::keybinding_hints::key_text;
use crate::components::visual_truncate::truncate_to_visual_lines;
use crate::keybindings::AppAction;
use crossterm::style::Attribute;
use zen_core::truncate::{truncate_tail, TruncationOptions};
use zen_tui::components::loader::Loader;
use zen_tui::components::Component;
use zen_tui::highlight::StyledSegment;
use zen_tui::keybindings::KeybindingsManager;
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{ColorToken, Theme};

/// Preview line limit (collapsed state; aligns with pi PREVIEW_LINES).
const PREVIEW_LINES: usize = 20;

/// Execution status.
enum BashStatus {
    Running,
    Complete {
        exit_code: Option<i32>,
        cancelled: bool,
        /// Whether context truncation occurred (None = truncation info not reported)
        truncated: Option<bool>,
        full_output_path: Option<String>,
    },
}

/// Bash command execution display block: $cmd header + streaming output + border + running/finished status — T-AP-11-1 in tool
/// Called in order at execution start/per-block output/end, so the user sees command output and exit status in real time.
///
/// Tail truncation of output lines (context truncation 2000 lines / 50KB + 20-line preview, width-aware); expanded state shows everything.
pub struct BashExecutionComponent<'a> {
    command: String,
    exclude_from_context: bool,
    theme: &'a Theme,
    kb: &'a KeybindingsManager<AppAction>,
    output_lines: Vec<String>,
    status: BashStatus,
    expanded: bool,
    loader: Loader,
    top_border: DynamicBorder,
    bottom_border: DynamicBorder,
}

impl<'a> BashExecutionComponent<'a> {
    /// Constructed from command, whether excluded from context, theme, and app keymap.
    /// Uses a dim border color when `exclude_from_context` (commands with `!!` prefix).
    pub fn new(
        command: String,
        exclude_from_context: bool,
        theme: &'a Theme,
        kb: &'a KeybindingsManager<AppAction>,
    ) -> Self {
        let color_key = if exclude_from_context {
            ColorToken::Dim
        } else {
            ColorToken::BashMode
        };
        let border_style = Style::fg(theme.fg(color_key));
        let spinner_style = Style::fg(theme.fg(color_key));
        let message_style = Style::fg(theme.fg(ColorToken::Muted));
        let run_message = format!(
            "Running... ({} to cancel)",
            key_text(kb, AppAction::Interrupt)
        );
        let mut loader = Loader::new(spinner_style, message_style, run_message);
        loader.start();
        BashExecutionComponent {
            command,
            exclude_from_context,
            theme,
            kb,
            output_lines: Vec::new(),
            status: BashStatus::Running,
            expanded: false,
            loader,
            top_border: DynamicBorder::new(border_style.clone()),
            bottom_border: DynamicBorder::new(border_style),
        }
    }

    /// Append streaming output block: strip ANSI + CRLF normalization + continuation-line joining, re-layout for display (pi appendOutput).
    pub fn append_output(&mut self, chunk: &str) {
        let clean = zen_core::ansi::strip_ansi(chunk)
            .replace("\r\n", "\n")
            .replace('\r', "\n");
        let new_lines: Vec<&str> = clean.split('\n').collect();
        if !self.output_lines.is_empty() && !new_lines.is_empty() {
            // Continuation-line joining: first block appends to the end of the previous line (pi semantics: incomplete lines continue).
            let last = self.output_lines.len() - 1;
            self.output_lines[last].push_str(new_lines[0]);
            for l in &new_lines[1..] {
                self.output_lines.push(l.to_string());
            }
        } else {
            for l in new_lines {
                self.output_lines.push(l.to_string());
            }
        }
    }

    /// Finalize on execution end: set exit code/cancel/error state and truncation info, stop spinner (pi setComplete).
    pub fn set_complete(
        &mut self,
        exit_code: Option<i32>,
        cancelled: bool,
        truncated: Option<bool>,
        full_output_path: Option<String>,
    ) {
        self.status = BashStatus::Complete {
            exit_code,
            cancelled,
            truncated,
            full_output_path,
        };
        self.loader.stop();
    }

    /// Toggle expanded state (where `app.tools.expand` takes effect).
    pub fn set_expanded(&mut self, expanded: bool) {
        self.expanded = expanded;
    }

    /// Current expanded state.
    pub fn is_expanded(&self) -> bool {
        self.expanded
    }

    /// Full raw output (for creating the BashExecution message). Consumer: T-AP-11-1.
    pub fn get_output(&self) -> String {
        self.output_lines.join("\n")
    }

    /// Command text.
    pub fn get_command(&self) -> &str {
        &self.command
    }

    fn color_key(&self) -> ColorToken {
        if self.exclude_from_context {
            ColorToken::Dim
        } else {
            ColorToken::BashMode
        }
    }

    /// Expand/collapse hint segment (". . . N more lines (ctrl+o to expand)", width-aware).
    fn expand_hint_segments(&self, hidden: usize) -> Vec<StyledSegment> {
        let theme = self.theme;
        let key = key_text(self.kb, AppAction::ToolsExpand);
        let text = if self.expanded {
            "(".to_string()
        } else if hidden > 0 {
            format!("... {hidden} more lines (")
        } else {
            "(".to_string()
        };
        let hint = if self.expanded {
            " to collapse)"
        } else {
            " to expand)"
        };
        vec![
            StyledSegment {
                style: Style::fg(theme.fg(ColorToken::Muted)),
                text,
            },
            StyledSegment {
                style: Style::fg(theme.fg(ColorToken::Dim)),
                text: key,
            },
            StyledSegment {
                style: Style::fg(theme.fg(ColorToken::Muted)),
                text: hint.to_string(),
            },
        ]
    }

    /// Status line (cancel/exit code/truncation hint).
    fn status_segments(&self) -> Vec<Vec<StyledSegment>> {
        let theme = self.theme;
        let mut lines: Vec<Vec<StyledSegment>> = Vec::new();
        let BashStatus::Complete {
            exit_code,
            cancelled,
            truncated,
            full_output_path,
        } = &self.status
        else {
            return lines;
        };

        let mut parts: Vec<StyledSegment> = Vec::new();
        if *cancelled {
            parts.push(StyledSegment {
                style: Style::fg(theme.fg(ColorToken::Warning)),
                text: "(cancelled)".to_string(),
            });
        } else if let Some(code) = exit_code {
            if *code != 0 {
                parts.push(StyledSegment {
                    style: Style::fg(theme.fg(ColorToken::Error)),
                    text: format!("(exit {code})"),
                });
            }
        }
        let was_truncated = truncated.unwrap_or(false);
        if was_truncated {
            if let Some(path) = full_output_path {
                parts.push(StyledSegment {
                    style: Style::fg(theme.fg(ColorToken::Warning)),
                    text: format!(" Output truncated. Full output: {path}"),
                });
            }
        }
        if !parts.is_empty() {
            lines.push(parts);
        }
        lines
    }
}

impl Component for BashExecutionComponent<'_> {
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        let mut row = y;
        let mut painted = 0u16;
        let mut remaining = height;
        let theme = self.theme;

        // 1. Top border
        if painted < height {
            let n = self.top_border.render(buf, x, row, width, remaining);
            painted += n;
            row += n;
            remaining -= n;
        }

        // 2. Command header (blank line + header line, aligned with pi Spacer+Text)
        if painted < height {
            buf.fill(x, row, width, 1, &Style::default());
            painted += 1;
            row += 1;
            remaining -= 1;
        }
        if painted < height {
            // "  $ cmd" (1-column indent + $) + Bold coloring
            let style = theme.style(self.color_key(), None, &[Attribute::Bold]);
            buf.write(x + 1, row, &style, &format!("$ {}", self.command));
            painted += 1;
            row += 1;
            remaining -= 1;
        }

        // 3. Content: running → loader; otherwise output + status
        if painted < height {
            match self.status {
                BashStatus::Running => {
                    let n = self.loader.render(buf, x, row, width, remaining);
                    painted += n;
                    row += n;
                    remaining -= n;
                }
                BashStatus::Complete { .. } => {
                    // Context truncation → available lines (shared module crate::truncate, T-AG-06-3)
                    let full = self.get_output();
                    let available_text = truncate_tail(&full, TruncationOptions::default()).content;
                    let available_lines: Vec<&str> = if available_text.is_empty() {
                        Vec::new()
                    } else {
                        available_text.split('\n').collect()
                    };

                    let hidden_logical = available_lines.len().saturating_sub(PREVIEW_LINES);

                    if !available_lines.is_empty() && painted < height {
                        buf.fill(x, row, width, 1, &Style::default());
                        painted += 1;
                        row += 1;
                        remaining -= 1;
                    }

                    if painted < height {
                        let muted = Style::fg(theme.fg(ColorToken::Muted));
                        if self.expanded {
                            for line in &available_lines {
                                if remaining == 0 {
                                    break;
                                }
                                buf.write(x + 1, row, &muted, line);
                                painted += 1;
                                row += 1;
                                remaining -= 1;
                            }
                        } else {
                            let preview = if hidden_logical > 0 {
                                &available_lines[available_lines.len() - PREVIEW_LINES..]
                            } else {
                                &available_lines[..]
                            };
                            let preview_text = preview.join("\n");
                            let vr =
                                truncate_to_visual_lines(&preview_text, PREVIEW_LINES, width, 1);
                            for line in &vr.lines {
                                if remaining == 0 {
                                    break;
                                }
                                buf.fill(x, row, width, 1, &Style::default());
                                buf.write(x + 1, row, &muted, line);
                                painted += 1;
                                row += 1;
                                remaining -= 1;
                            }
                        }
                    }

                    // Expand/collapse hint (when hidden lines > 0)
                    if hidden_logical > 0 && painted < height {
                        let segs = self.expand_hint_segments(hidden_logical);
                        buf.fill(x, row, width, 1, &Style::default());
                        let mut col = x + 1;
                        for seg in segs {
                            let w = zen_tui::screen::visible_width(&seg.text) as u16;
                            if col + w > width {
                                break;
                            }
                            buf.write(col, row, &seg.style, &seg.text);
                            col += w;
                        }
                        painted += 1;
                        row += 1;
                        remaining -= 1;
                    }

                    // Cancel/exit code/truncation status line
                    for segs in self.status_segments() {
                        if remaining == 0 {
                            break;
                        }
                        buf.fill(x, row, width, 1, &Style::default());
                        let mut col = x + 1;
                        for seg in segs {
                            let w = zen_tui::screen::visible_width(&seg.text) as u16;
                            if col + w > width {
                                break;
                            }
                            buf.write(col, row, &seg.style, &seg.text);
                            col += w;
                        }
                        painted += 1;
                        row += 1;
                        remaining -= 1;
                    }
                }
            }
        }

        // 4. Bottom border
        if painted < height {
            let n = self.bottom_border.render(buf, x, row, width, remaining);
            painted += n;
        }

        painted
    }

    fn handle_key(&mut self, key: &crossterm::event::KeyEvent) -> bool {
        // app.tools.expand → toggle expanded state
        if let Some(kid) = zen_tui::keys::KeyId::from_key_event(*key) {
            if self.kb.matches(&kid, AppAction::ToolsExpand) {
                self.set_expanded(!self.expanded);
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keybindings::app_default_bindings;
    use std::collections::HashMap;
    use zen_core::truncate::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES};

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
        zen_core::ansi::strip_ansi(&String::from_utf8_lossy(&out))
            .chars()
            .filter(|c| !c.is_control())
            .collect()
    }

    #[test]
    fn truncate_tail_limits_lines() {
        let text = (0..3000)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let out = truncate_tail(&text, TruncationOptions::default());
        assert!(out.truncated);
        assert_eq!(out.output_lines, DEFAULT_MAX_LINES);
    }

    #[test]
    fn truncate_tail_limits_bytes() {
        let text = "x".repeat(60 * 1024);
        let out = truncate_tail(&text, TruncationOptions::default());
        assert!(out.truncated);
        assert!(out.content.len() <= DEFAULT_MAX_BYTES);
    }

    #[test]
    fn truncate_tail_no_truncation() {
        let out = truncate_tail("hello", TruncationOptions::default());
        assert!(!out.truncated);
        assert_eq!(out.content, "hello");
    }

    #[test]
    fn running_renders_command_and_loader() {
        let kb = kb();
        let mut comp = BashExecutionComponent::new("echo hi".into(), false, theme(), &kb);
        let mut buf = ScreenBuffer::new(80, 10);
        let painted = comp.render(&mut buf, 0, 0, 80, 10);
        assert!(painted >= 4, "border + header + loader + border");
        let text = flush_text(&mut buf);
        assert!(text.contains("$ echo hi"));
        assert!(text.contains("Running..."));
    }

    #[test]
    fn append_output_normalizes_and_merges() {
        let kb = kb();
        let mut comp = BashExecutionComponent::new("cmd".into(), false, theme(), &kb);
        comp.append_output("part1\r\npart2");
        comp.append_output("cont");
        assert_eq!(comp.get_output(), "part1\npart2cont");
    }

    #[test]
    fn complete_shows_exit_code_and_hidden_lines() {
        let kb = kb();
        let mut comp = BashExecutionComponent::new("cmd".into(), false, theme(), &kb);
        for i in 0..30 {
            comp.append_output(&format!("out line {i}\n"));
        }
        comp.set_complete(Some(1), false, Some(false), None);
        let mut buf = ScreenBuffer::new(100, 40);
        comp.render(&mut buf, 0, 0, 100, 40);
        let text = flush_text(&mut buf);

        assert!(text.contains("(exit 1)"));
        assert!(text.contains("more lines"));
    }

    #[test]
    fn cancelled_shows_cancelled_marker() {
        let kb = kb();
        let mut comp = BashExecutionComponent::new("cmd".into(), false, theme(), &kb);
        comp.append_output("x");
        comp.set_complete(None, true, None, None);
        let mut buf = ScreenBuffer::new(80, 20);
        comp.render(&mut buf, 0, 0, 80, 20);
        assert!(flush_text(&mut buf).contains("(cancelled)"));
    }

    #[test]
    fn truncation_warning_with_path() {
        let kb = kb();
        let mut comp = BashExecutionComponent::new("cmd".into(), false, theme(), &kb);
        comp.append_output("x");
        comp.set_complete(Some(0), false, Some(true), Some("/tmp/out.txt".into()));
        let mut buf = ScreenBuffer::new(120, 20);
        comp.render(&mut buf, 0, 0, 120, 20);
        assert!(flush_text(&mut buf).contains("/tmp/out.txt"));
    }

    #[test]
    fn expand_key_toggles_expanded() {
        let kb = kb();
        let mut comp = BashExecutionComponent::new("cmd".into(), false, theme(), &kb);
        assert!(!comp.is_expanded());
        let ctrl_o = crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('o'),
            crossterm::event::KeyModifiers::CONTROL,
        );
        assert!(comp.handle_key(&ctrl_o));
        assert!(comp.is_expanded());
    }

    #[test]
    fn supports_short_circuit_keys_not_consumed() {
        let kb = kb();
        let mut comp = BashExecutionComponent::new("cmd".into(), false, theme(), &kb);
        let a = crossterm::event::KeyEvent::from(crossterm::event::KeyCode::Char('a'));
        assert!(!comp.handle_key(&a));
    }
}
