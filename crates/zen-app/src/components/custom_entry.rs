// REQ-APP-11 (pi-358): custom-entry — fallback rendering for extension custom session entries.
// Corresponding original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/custom-entry.ts.
// v1 has no extension system: EntryRenderer is removed; only renders "`[customType]` tag + JSON data overview";
// Custom entries are not actually produced (JSONL format compatibility only; see zen-core SessionTreeEntry::Custom comment).

use crate::components::{paint_fold_rows, Component, FoldRow};
use crossterm::style::Attribute;
use serde_json::Value;
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{BgToken, ColorToken, Theme};

/// Fallback rendering for extension custom session entries (`Custom` entry, JSONL compatibility only, not produced in v1):
/// `[customType]` tag + serde_json data overview; renderer is None (v1 has no extension system).
/// Consumer: T-AP-11-1 (when replaying historical sessions and encountering Custom entries).
pub struct CustomEntryComponent<'a> {
    custom_type: String,
    data: Option<Value>,
    theme: &'a Theme,
}

impl<'a> CustomEntryComponent<'a> {
    /// Constructed with custom type and optional JSON data.
    pub fn new(custom_type: String, data: Option<Value>, theme: &'a Theme) -> Self {
        CustomEntryComponent {
            custom_type,
            data,
            theme,
        }
    }

    /// Custom type identifier.
    pub fn custom_type(&self) -> &str {
        &self.custom_type
    }

    /// Whether there is data to display.
    pub fn has_content(&self) -> bool {
        self.data.is_some()
    }

    fn plan(&self) -> Vec<FoldRow> {
        let theme = self.theme;
        let label_style = theme.style(ColorToken::CustomMessageLabel, None, &[Attribute::Bold]);
        match &self.data {
            Some(data) => {
                let text = match data {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                vec![
                    FoldRow::Label {
                        text: format!("[{}]", self.custom_type),
                        style: label_style,
                    },
                    FoldRow::Gap,
                    FoldRow::Segments {
                        segments: vec![zen_tui::highlight::StyledSegment {
                            style: Style::fg(theme.fg(ColorToken::CustomMessageText)),
                            text,
                        }],
                    },
                ]
            }
            None => vec![FoldRow::Label {
                text: format!("[{}]", self.custom_type),
                style: label_style,
            }],
        }
    }
}

impl Component for CustomEntryComponent<'_> {
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        let bg = Style::bg(self.theme.bg(BgToken::CustomMessageBg));
        let rows = self.plan();
        paint_fold_rows(buf, x, y, width, height, &rows, 1, Some(&bg))
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
    fn renders_type_label_with_data() {
        let mut comp =
            CustomEntryComponent::new("checkpoint".into(), Some(json!({"note": "saved"})), theme());
        assert_eq!(comp.custom_type(), "checkpoint");
        assert!(comp.has_content());
        let mut buf = ScreenBuffer::new(60, 10);
        comp.render(&mut buf, 0, 0, 60, 10);
        let text = flush_text(&mut buf);
        assert!(text.contains("[checkpoint]"));
        assert!(text.contains("saved"));
    }

    #[test]
    fn no_data_renders_label_only() {
        let mut comp = CustomEntryComponent::new("checkpoint".into(), None, theme());
        assert!(!comp.has_content());
        let mut buf = ScreenBuffer::new(60, 10);
        let painted = comp.render(&mut buf, 0, 0, 60, 10);
        // label + vertical padding
        assert_eq!(painted, 3);
        assert!(flush_text(&mut buf).contains("[checkpoint]"));
    }

    #[test]
    fn string_data_shown_verbatim() {
        let mut comp = CustomEntryComponent::new("note".into(), Some(json!("plain")), theme());
        let mut buf = ScreenBuffer::new(60, 10);
        comp.render(&mut buf, 0, 0, 60, 10);
        assert!(flush_text(&mut buf).contains("plain"));
    }
}
