// REQ-APP-11 (pi-358): interactive component set — app-layer component registry + shared rendering helpers.
// Unlike pi's `mod.rs` (index.ts counterpart), this module also hosts the row model and painting function
// shared by the fold message boxes (branch/compaction/skill/custom) — unified visuals across all four message types and avoids duplicating the same "background box + label + fold" logic.

pub mod assistant_message;
pub mod bash_execution;
pub mod bordered_loader;
pub mod branch_summary_message;
pub mod compaction_summary_message;
pub mod countdown_timer;
pub mod custom_editor;
pub mod custom_entry;
pub mod custom_message;
pub mod diff;
pub mod dynamic_border;
pub mod first_time_setup;
pub mod footer;
pub mod keybinding_hints;
pub mod render;
pub mod session_selector;
pub mod session_selector_search;
pub mod settings_selector;
pub mod skill_invocation_message;
pub mod status_indicator;
pub mod thinking_selector;
pub mod tool_execution;
pub mod tree_selector;
pub mod user_message;
pub mod user_message_selector;
pub mod visual_truncate;

use crossterm::style::Color;
use zen_tui::highlight::StyledSegment;
use zen_tui::markdown::MarkdownTheme;
use zen_tui::screen::{self, ScreenBuffer, Style};

pub use zen_tui::components::Component;

/// Inline segment sequence write (truncates when too wide), used by FoldRow::Segments etc. `box_bg` is the background of the containing box
/// (merged into bg-None segments, see render::merge_box_bg; pass None for non-boxes).
fn write_segments_row(
    buf: &mut ScreenBuffer,
    x: u16,
    y: u16,
    width: u16,
    segments: &[StyledSegment],
    box_bg: Option<Color>,
) {
    let mut col = x;
    for seg in segments {
        let w = screen::visible_width(&seg.text) as u16;
        let style = render::merge_box_bg(&seg.style, box_bg);
        if col + w > width {
            buf.write(col, y, &style, &seg.text);
            return;
        }
        buf.write(col, y, &style, &seg.text);
        col += w;
    }
}

// strip_ansi has been migrated into the canonical zen-core module (T-AP-09-8, `zen_core::ansi::strip_ansi`, vte parser),
// this file no longer maintains an inline implementation; the behavior differences from the old inline one (C1 0x9B half-strip, dangling ESC dropped) are known-gap fixes,
// locked by zen-core ansi.rs unit tests. Components consume it via `zen_core::ansi::strip_ansi` (bash-execution streaming cleanup,
// flush text restoration in component tests).

/// Row-planning unit for fold message boxes (equivalent of pi's addChild sequence).
///
/// Shared row model for the four message types —
/// (customMessageBg background box family); `paint_fold_rows` does the actual drawing.
#[allow(clippy::large_enum_variant)] // Markdown carries a full MarkdownTheme
pub(crate) enum FoldRow {
    /// Blank line.
    Gap,
    /// Label line (e.g. `[branch]`, `[compaction]`).
    Label { text: String, style: Style },
    /// Single-line segment sequence (fold hint line: text + key-hint segments).
    Segments { segments: Vec<StyledSegment> },
    /// Markdown segment (expanded content).
    Markdown { theme: MarkdownTheme, text: String },
}

/// Paint a fold box row plan line by line: 1 line of background padding above and below (including Box(1,1,…) semantics for all four message types),
/// Each row leaves `pad` columns blank on the left; truncated from the bottom when the height budget is insufficient. Returns the number of rows actually drawn.
#[allow(clippy::too_many_arguments)]
pub(crate) fn paint_fold_rows(
    buf: &mut ScreenBuffer,
    x: u16,
    y: u16,
    width: u16,
    height: u16,
    rows: &[FoldRow],
    pad: u16,
    bg: Option<&Style>,
) -> u16 {
    // Row sequence = [top padding] + rows + [bottom padding];
    // Each row (especially Markdown paragraphs) occupies lines based on actual written height, avoiding content being clipped due to a precomputed line count being too low.
    // box background (provided by bg) covers every line; text cells/whitespace merge into the box background via merge_box_bg —
    // glyphs match the block background (fixes separation between fill and text-cell bg=None, T-TU-02-1-fix).
    let mut painted = 0u16;
    let default_style = Style::default();
    let box_bg: Option<Color> = bg.and_then(|s| s.bg);

    // top padding line (inside the box, with box background)
    if painted < height {
        let fill_style = bg.unwrap_or(&default_style);
        buf.fill(x, y + painted, width, 1, fill_style);
        painted += 1;
    }

    for r in rows {
        if painted >= height {
            break;
        }
        let remaining = height - painted;
        let fill_style = bg.unwrap_or(&default_style);
        buf.fill(x, y + painted, width, 1, fill_style);
        match r {
            FoldRow::Gap => {
                painted += 1;
            }
            FoldRow::Label { text, style } => {
                let style = render::merge_box_bg(style, box_bg);
                buf.write(x + pad, y + painted, &style, text);
                painted += 1;
            }
            FoldRow::Segments { segments } => {
                write_segments_row(buf, x + pad, y + painted, width, segments, box_bg);
                painted += 1;
            }
            FoldRow::Markdown { theme, text } => {
                let n = render::write_markdown(
                    buf,
                    x,
                    y + painted,
                    width,
                    remaining,
                    theme,
                    text,
                    pad,
                    box_bg,
                );
                painted += n;
            }
        }
    }

    // Bottom padding row (if a background is provided, it covers this row)
    if painted < height {
        let fill_style = bg.unwrap_or(&default_style);
        buf.fill(x, y + painted, width, 1, fill_style);
        painted += 1;
    }

    painted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::style::Color;

    fn seg(text: &str) -> StyledSegment {
        StyledSegment {
            style: Style::default(),
            text: text.to_string(),
        }
    }

    #[test]
    fn paint_fold_rows_no_bg_matches_previous_behavior() {
        let rows = vec![
            FoldRow::Gap,
            FoldRow::Label {
                text: "x".into(),
                style: Style::default(),
            },
        ];
        let mut buf = ScreenBuffer::new(20, 6);
        let painted = paint_fold_rows(&mut buf, 0, 0, 20, 6, &rows, 0, None);
        // top padding + 2 content lines + bottom padding
        assert_eq!(painted, 4);
        // no box background: all lines bg None (preserves original behavior)
        for r in 0..painted {
            assert!(
                buf.line_segments(r).iter().all(|s| s.style.bg.is_none()),
                "Without a box background, should not merge into any background (row {r})"
            );
        }
    }

    #[test]
    fn paint_fold_rows_box_merges_bg_into_text_and_padding() {
        // box background: Segments text cells, Gap, top/bottom padding all land on the box background (glyphs no longer show through the default base color)
        let bg = Style::bg(Color::Magenta);
        let rows = vec![
            FoldRow::Gap,
            FoldRow::Segments {
                segments: vec![seg("content")],
            },
        ];
        let mut buf = ScreenBuffer::new(20, 6);
        let painted = paint_fold_rows(&mut buf, 0, 0, 20, 6, &rows, 0, Some(&bg));
        assert_eq!(painted, 4);
        // top padding: box background
        assert_eq!(buf.line_segments(0)[0].style.bg, Some(Color::Magenta));
        // Gap row: box background
        assert_eq!(buf.line_segments(1)[0].style.bg, Some(Color::Magenta));
        // Segments text cell: merged into box background
        assert!(
            buf.line_segments(2)
                .iter()
                .any(|s| s.text.contains("content") && s.style.bg == Some(Color::Magenta)),
            "Text glyphs inside the box should merge into the box background"
        );
        // Bottom padding: box background
        assert_eq!(buf.line_segments(3)[0].style.bg, Some(Color::Magenta));
    }
}
