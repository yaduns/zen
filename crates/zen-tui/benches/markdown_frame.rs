// PF-01 (REQ-TUI-01 observability): markdown frame-render benchmark — five synthetic fixtures feeding the numeric gate in `scripts/perf-gate.sh`.
// Plan doc: docs/performance-optimization-plan.md §6.2; task notes: tasks/PF-01-implemented.md
//
// Why synthetic fixtures instead of a recorded session: the interactive frame cost is dominated by the markdown
// pass over the whole transcript (one `render_markdown` call per message row), so a deterministic fixture with a
// controlled mix of prose and fenced code reproduces the shape of the regression without committing a 169 KB
// session file. Every fixture targets 15–25 KB, matching the measured baseline in the plan document.
//
// Fixture set (aligned with the plan's five tiers):
//   frame_no_code          — prose only (thinking collapsed / no code)
//   frame_few_code         — prose + 3 fenced blocks
//   frame_many_code        — prose + 20 fenced blocks (~1.5 K code lines): the syntect-dominated case
//   frame_thinking_expanded— prose + 15 blocks, largest total (thinking blocks are rendered through the same parser)
//   frame_thinking_hidden  — prose only, same size as the expanded tier's prose half
use criterion::{criterion_group, criterion_main, Criterion};
use zen_tui::markdown::render_markdown;
use zen_tui::theme::builtin_dark_theme;

/// Terminal width used by the measurements (the wrapped-row count depends on it, so it is frozen here).
const WIDTH: usize = 80;

/// One prose paragraph, ~400 bytes. Repeated to reach the target size.
const PARAGRAPH: &str = "\
The session tree is the core of the product: every message hangs off the previous one and a branch is just a \
different leaf. Resuming replays the persisted JSONL, rebuilding the transcript from the entries and their \
parent pointers, so the model never sees a message that the user did not actually produce on this branch.

";

/// One fenced Rust block body (~10 lines): enough for the highlighter to switch scopes several times.
const CODE_BLOCK: &str = "\
```rust
fn handle_frame(&mut self, canvas: &mut ScrollbackCanvas) -> Result<(), TuiError> {
    let (cols, rows) = (self.terminal.columns(), self.terminal.rows());
    canvas.set_size(cols, rows);
    let mut probe = FrameProbe::start(&self.stats);
    canvas.clear();
    (self.draw)(canvas);
    probe.mark_draw();
    canvas.flush(self.terminal.out_mut())?;
    Ok(())
}
```

";

/// Build `paragraphs` prose paragraphs followed by code blocks interleaved every third paragraph.
fn fixture(paragraphs: usize, code_blocks: usize) -> String {
    let mut text =
        String::with_capacity(paragraphs * PARAGRAPH.len() + code_blocks * CODE_BLOCK.len());
    let mut placed = 0usize;
    for i in 0..paragraphs {
        text.push_str(PARAGRAPH);
        // Interleave so that the highlighter is re-entered repeatedly across the document (closer to a real
        // assistant message, where code and prose alternate) instead of one contiguous code tail.
        if i % 3 == 2 && placed < code_blocks {
            text.push_str(CODE_BLOCK);
            placed += 1;
        }
    }
    while placed < code_blocks {
        text.push_str(CODE_BLOCK);
        placed += 1;
    }
    text
}

/// Measure one fixture at `WIDTH`: one `render_markdown` pass is exactly what a frame does per message row.
fn bench_render(c: &mut Criterion, id: &str, text: &str) {
    let theme = builtin_dark_theme().markdown_theme();
    c.bench_function(id, |b| b.iter(|| render_markdown(&theme, text, WIDTH)));
}

fn markdown_frame(c: &mut Criterion) {
    let no_code = fixture(48, 0);
    let few_code = fixture(45, 3);
    let many_code = fixture(30, 20);
    let thinking_expanded = fixture(40, 15);
    let thinking_hidden = fixture(40, 0);

    bench_render(c, "frame_no_code", &no_code);
    bench_render(c, "frame_few_code", &few_code);
    bench_render(c, "frame_many_code", &many_code);
    bench_render(c, "frame_thinking_expanded", &thinking_expanded);
    bench_render(c, "frame_thinking_hidden", &thinking_hidden);
}

criterion_group!(benches, markdown_frame);
criterion_main!(benches);
