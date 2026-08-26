// REQ-APP-11 (pi-358): skill-invocation-message — skill invocation message.
// Corresponding original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/skill-invocation-message.ts.
// Input is a loose (name, content) pair (the zen ParsedSkillBlock type has not landed yet, see T-AP-11-2 open question 2).

use crate::components::{paint_fold_rows, Component, FoldRow};
use crate::keybindings::AppAction;
use crossterm::style::Attribute;
use zen_tui::highlight::StyledSegment;
use zen_tui::keybindings::KeybindingsManager;
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{BgToken, ColorToken, Theme};

/// Skill invocation message (the skill block only; the user message is rendered separately): when collapsed, `[skill] name (… to expand)`,
/// when expanded, label + `**name**` heading + skill content markdown. Consumer: T-AP-11-1 (after the /skill command is delivered).
pub struct SkillInvocationMessageComponent<'a> {
    name: String,
    content: String,
    expanded: bool,
    theme: &'a Theme,
    kb: &'a KeybindingsManager<AppAction>,
}

impl<'a> SkillInvocationMessageComponent<'a> {
    /// Constructs from skill name, content, theme, and app keybinding map (collapse hint key = app.tools.expand).
    pub fn new(
        name: String,
        content: String,
        theme: &'a Theme,
        kb: &'a KeybindingsManager<AppAction>,
    ) -> Self {
        SkillInvocationMessageComponent {
            name,
            content,
            expanded: false,
            theme,
            kb,
        }
    }

    /// Toggle expanded state (where `app.tools.expand` takes effect).
    pub fn set_expanded(&mut self, expanded: bool) {
        self.expanded = expanded;
    }

    /// Current expanded state.
    pub fn is_expanded(&self) -> bool {
        self.expanded
    }

    /// Skill name.
    pub fn name(&self) -> &str {
        &self.name
    }

    fn plan(&self) -> Vec<FoldRow> {
        let theme = self.theme;
        let label_style = theme.style(ColorToken::CustomMessageLabel, None, &[Attribute::Bold]);

        if self.expanded {
            let mut md = theme.markdown_theme();
            md.text = Style::fg(theme.fg(ColorToken::CustomMessageText));
            vec![
                FoldRow::Label {
                    text: "[skill]".to_string(),
                    style: label_style,
                },
                FoldRow::Gap,
                FoldRow::Markdown {
                    theme: md,
                    text: format!("**{}**\n\n{}", self.name, self.content),
                },
            ]
        } else {
            let key =
                crate::components::keybinding_hints::key_text(self.kb, AppAction::ToolsExpand);
            vec![FoldRow::Segments {
                segments: vec![
                    StyledSegment {
                        style: label_style,
                        text: format!("[skill] {}", self.name),
                    },
                    StyledSegment {
                        style: Style::fg(theme.fg(ColorToken::Dim)),
                        text: format!(" ({key} to expand)"),
                    },
                ],
            }]
        }
    }
}

impl Component for SkillInvocationMessageComponent<'_> {
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        let bg = Style::bg(self.theme.bg(BgToken::CustomMessageBg));
        let rows = self.plan();
        paint_fold_rows(buf, x, y, width, height, &rows, 1, Some(&bg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keybindings::app_default_bindings;
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
    fn collapsed_shows_name_and_hint_in_one_line() {
        let kb = kb();
        let mut comp =
            SkillInvocationMessageComponent::new("my-skill".into(), "content".into(), theme(), &kb);
        let mut buf = ScreenBuffer::new(60, 10);
        comp.render(&mut buf, 0, 0, 60, 10);
        let text = flush_text(&mut buf);
        assert!(text.contains("[skill] my-skill"));
        assert!(text.contains("ctrl+o"));
        assert!(!text.contains("content"));
    }

    #[test]
    fn expanded_shows_name_header_and_content() {
        let kb = kb();
        let mut comp = SkillInvocationMessageComponent::new(
            "my-skill".into(),
            "hello content".into(),
            theme(),
            &kb,
        );
        comp.set_expanded(true);
        assert!(comp.is_expanded());
        let mut buf = ScreenBuffer::new(60, 10);
        comp.render(&mut buf, 0, 0, 60, 10);
        let text = flush_text(&mut buf);
        assert!(text.contains("my-skill"));
        assert!(text.contains("hello content"));
    }

    #[test]
    fn accessors() {
        let kb = kb();
        let mut comp = SkillInvocationMessageComponent::new("s".into(), "c".into(), theme(), &kb);
        assert_eq!(comp.name(), "s");
        assert!(!comp.is_expanded());
        comp.set_expanded(true);
        assert!(comp.is_expanded());
    }
}
