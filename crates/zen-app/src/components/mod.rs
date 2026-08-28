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

use zen_tui::highlight::StyledSegment;
use zen_tui::markdown::MarkdownTheme;
use zen_tui::screen::{self, ScreenBuffer, Style};

pub use zen_tui::components::Component;

/// Write inline segment sequence (stop when exceeding width), for use by FoldRow::Segments etc.
fn write_segments_row(
    buf: &mut ScreenBuffer,
    x: u16,
    y: u16,
    width: u16,
    segments: &[StyledSegment],
) {
    let mut col = x;
    for seg in segments {
        let w = screen::visible_width(&seg.text) as u16;
        if col + w > width {
            buf.write(col, y, &seg.style, &seg.text);
            return;
        }
        buf.write(col, y, &seg.style, &seg.text);
        col += w;
    }
}

/// Strip ANSI escape sequences (CSI/OSC, etc.). Used to clean streamed output from bash-execution,
/// And flush text restoration for each component test (preserving printable character order). Shared module implementation.
pub(crate) fn strip_ansi(text: &str) -> String {
    let mut out = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\x1b' {
            out.push(c);
            continue;
        }
        // ESC sequence: first skip the introducer characters ([ ] ^ _ P X, etc.), then consume parameter bytes (0x30-0x3F),
        // Stop at the final byte (0x40-0x7E) or a control character; OSC (starting with ]) continues until BEL/control character.
        let ostyle = chars.next();
        let osc = matches!(ostyle, Some(']'));
        for next in chars.by_ref() {
            if next.is_control() {
                break;
            }
            let b = next as u32;
            if osc {
                continue;
            }
            if (0x30..=0x3F).contains(&b) {
                continue; // Parameters
            }
            break; // Final byte
        }
    }
    out
}

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
    let mut painted = 0u16;

    // Top padding line
    if painted < height {
        buf.fill(x, y + painted, width, 1, &Style::default());
        painted += 1;
    }

    for r in rows {
        if painted >= height {
            break;
        }
        let remaining = height - painted;
        // Fill background per line first (align with pi Box whole-box background; per-line to avoid overwriting content below out of bounds)
        if let Some(bg_style) = bg {
            buf.fill(x, y + painted, width, 1, bg_style);
        }
        match r {
            FoldRow::Gap => {
                buf.fill(x, y + painted, width, 1, &Style::default());
                painted += 1;
            }
            FoldRow::Label { text, style } => {
                buf.fill(x, y + painted, width, 1, &Style::default());
                buf.write(x + pad, y + painted, style, text);
                painted += 1;
            }
            FoldRow::Segments { segments } => {
                buf.fill(x, y + painted, width, 1, &Style::default());
                write_segments_row(buf, x + pad, y + painted, width, segments);
                painted += 1;
            }
            FoldRow::Markdown { theme, text } => {
                let n =
                    render::write_markdown(buf, x, y + painted, width, remaining, theme, text, pad);
                painted += n;
            }
        }
    }

    // Bottom padding row (if a background is provided, it covers this row)
    if painted < height {
        if let Some(bg_style) = bg {
            buf.fill(x, y + painted, width, 1, bg_style);
        } else {
            buf.fill(x, y + painted, width, 1, &Style::default());
        }
        painted += 1;
    }

    painted
}
