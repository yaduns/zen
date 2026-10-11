// REQ-TUI-02 (pi-412): markdown rendering — crossterm + pulldown-cmark + syntax highlighting.
// Pure conversion module: markdown → styled terminal lines (Vec<Vec<StyledSegment>>); no dependency on the TUI main loop, no ScreenBuffer writes,
// reused by interactive message rendering (T-AP-11-2) and print-mode message output.
// Ecosystem paradigm: parsing via pulldown-cmark (Options: tables/strikethrough/tasklists), code-block coloring delegated to
// zen_tui::highlight (T-AP-10-1); wrapping reuses wrap::word_wrap_line (word-boundary/CJK-aware, byte-range back-mapping to style segments);
// pi's ANSI string output model collapses to a Style-segment model (aligned with the hash screen::Style carrier),
// MarkdownOptions (preserve* normalization options) and cache fields (ScreenBuffer differential model re-renders every frame) are collapsed entirely.
use crate::highlight::{self, StyledSegment};
use crate::screen::{visible_width, Style};
use crate::wrap::word_wrap_line;
use crossterm::style::{Attribute, Color};
use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};

/// Static mapping of markdown elements → `screen::Style` (visual configuration carrier).
/// Rendering combines 15 style fields + 1 code-block indent string to derive each token's style;
/// `text` is the plain-text baseline style; `heading` layers Bold on top (h1 additionally gets Underline).
/// Attribute bits for `link` stacked with `underline`; `bold/italic/strikethrough` are the
/// foreground color and attribute source for the corresponding inline element (aligned with pi's level composition: bold uses the brightest foreground, italic uses purple, strikethrough uses gray).
/// Pure data, Clone/PartialEq/Debug-able; not serialized (runtime visual data).
/// `Default` provides a dark theme suited to the zen dark TUI (base16-ocean.dark palette,
/// same family as the default theme in `highlight.rs`); when the T-AP-11-3 theme system is integrated, it will construct this type to populate the field.
#[derive(Clone, PartialEq, Debug)]
pub struct MarkdownTheme {
    /// Plain text baseline style; used when not overridden by inline styles
    pub text: Style,
    /// Heading styles; the renderer overlays Bold+Underline on h1, Bold on h2+
    pub heading: Style,
    /// `**bold**` bold (foreground color + attribute source)
    pub bold: Style,
    /// `*italic*` italic (foreground color + attribute source)
    pub italic: Style,
    /// `~~del~~` strikethrough (foreground color + attribute source)
    pub strikethrough: Style,
    /// Underline (link text overlays its attribute bits)
    pub underline: Style,
    /// Link text; the renderer overlays Underline, and attaches a `link_url` annotation in parentheses when the URL differs from the text
    pub link: Style,
    /// Style for the link URL annotation ` (url)`
    pub link_url: Style,
    /// Inline code `` `code` ``
    pub code: Style,
    /// Code block body (syntax coloring handled by `highlight.rs`; this style provides the fallback foreground;
    /// `bg` is a schema-reserved field not applied by the renderer — code block background = surrounding environment background, aligned with pi)
    pub code_block: Style,
    /// `` ``` `` fence line (including the language tag)
    pub code_block_border: Style,
    /// Quote text (the renderer treats it as the baseline style inside a quote; includes Italic by default)
    pub quote: Style,
    /// Left border `│ ` of a quote
    pub quote_border: Style,
    /// `---` horizontal rule
    pub hr: Style,
    /// List markers (`.` / `-` / `[x]`)
    pub list_bullet: Style,
    /// Prefix indent before each line of a code block (default `"  "`)
    pub code_block_indent: String,
    /// Name of the built-in syntect theme used for code block syntax highlighting (in the same family as `highlight.rs` `DEFAULT_THEME`).
    /// The renderer is self-contained: code block coloring resolves by this name and does not depend on a `Theme` reference (T-AP-11-3 via
    /// `Theme::syntax_theme_name` supplies the dark/light mapping; `markdown_theme()` populates it;
    /// `Default` provides the dark-mode fallback).
    pub syntax_theme_name: String,
}

impl Default for MarkdownTheme {
    /// Provides a set of default markdown visual styles fitting the zen dark TUI — an "out-of-the-box" fallback for the first screen / print mode,
    /// so consumers can render directly without configuring a theme; once T-AP-11-3 theme system lands, it will construct this type instead.
    /// Palette mapping semantics (base16 ocean.dark, same family as `highlight.rs` `DEFAULT_THEME`):
    /// text=base05 body, heading=base0D+Bold blue headings, bold=base07 brightest,
    /// italic=base0E purple, strikethrough=base03 comment gray, link=base0C+Underline cyan,
    /// link_url=base03, code=base08 red, code_block=base05 on base01 (dark background);
    /// quote=base0E+Italic、quote_border=base0C、hr=base03、list_bullet=base0C。
    fn default() -> Self {
        fn fg(r: u8, g: u8, b: u8) -> Style {
            Style {
                fg: Some(Color::Rgb { r, g, b }),
                bg: None,
                attributes: Vec::new(),
            }
        }
        fn fg_attr(r: u8, g: u8, b: u8, attr: Attribute) -> Style {
            Style {
                fg: Some(Color::Rgb { r, g, b }),
                bg: None,
                attributes: vec![attr],
            }
        }
        fn fg_bg(fg_r: u8, fg_g: u8, fg_b: u8, bg_r: u8, bg_g: u8, bg_b: u8) -> Style {
            Style {
                fg: Some(Color::Rgb {
                    r: fg_r,
                    g: fg_g,
                    b: fg_b,
                }),
                bg: Some(Color::Rgb {
                    r: bg_r,
                    g: bg_g,
                    b: bg_b,
                }),
                attributes: Vec::new(),
            }
        }

        Self {
            text: fg(0xc0, 0xc5, 0xce),                                  // base05
            heading: fg_attr(0x8f, 0xa1, 0xb3, Attribute::Bold),         // base0D + Bold
            bold: fg(0xef, 0xf1, 0xf5),                                  // base07
            italic: fg(0xb4, 0x8e, 0xad),                                // base0E
            strikethrough: fg(0x65, 0x73, 0x7e),                         // base03
            underline: fg_attr(0xc0, 0xc5, 0xce, Attribute::Underlined), // base05 + Underline
            link: fg_attr(0x96, 0xb5, 0xb4, Attribute::Underlined),      // base0C + Underline
            link_url: fg(0x65, 0x73, 0x7e),                              // base03
            code: fg(0xbf, 0x61, 0x6a),                                  // base08
            code_block: fg_bg(0xc0, 0xc5, 0xce, 0x34, 0x3d, 0x46),       // base05 on base01
            code_block_border: fg(0x65, 0x73, 0x7e),                     // base03
            quote: fg_attr(0xb4, 0x8e, 0xad, Attribute::Italic),         // base0E + Italic
            quote_border: fg(0x96, 0xb5, 0xb4),                          // base0C
            hr: fg(0x65, 0x73, 0x7e),                                    // base03
            list_bullet: fg(0x96, 0xb5, 0xb4),                           // base0C
            code_block_indent: "  ".to_string(),
            syntax_theme_name: "base16-ocean.dark".to_string(),
        }
    }
}

/// Converts a markdown message into styled terminal lines — chat bubbles are "tailored to fit" before being displayed;
/// Both interactive message rendering (T-AP-11-2) and print-mode message output come here for the fabric before framing/piping.
/// Parameters: `theme` element style map (including the plain-text baseline), `text` raw markdown (line endings normalized by this function,
/// CRLF/CR → LF, for use with `highlight`), `width` available display columns of the content area (excluding outer margin/background).
/// Returns: styled segment lines organized by display line (`Vec<Vec<StyledSegment>>`, each line may contain multiple segments with different styles),
/// word-wrapped to `width` (quote/list prefixes count toward the width); empty input returns an empty Vec; callers then write line by line, segment by segment
/// to a `ScreenBuffer` or convert to ANSI for piping. Supports headings/paragraphs/lists (including nesting, task items, ordered start numbers)/
/// fenced code blocks (syntax highlighted)/blockquotes/tables/inline code/bold/italic/strikethrough/links (with URL annotation)/horizontal rules.
/// Determines whether a line starts a block-level marker (used by `protect_prose_indentation` to decide whether to protect the line's leading whitespace):
/// If the content after stripping leading whitespace starts with these → let pulldown parse it with existing CommonMark block semantics (no protection),
/// avoiding corruption of real markdown structures such as nested lists/headings/blockquotes/fences/indented code. Aligned with marked (pi)'s
/// paragraph rule boundary: marked excludes only hl/heading/blockquote/tag/table from paragraphs; zen here
/// additionally excludes lists/fences/hr to protect nested structures (the difference is only "more conservatively not protecting"; display result matches marked).
/// Parameter: `content` is the line content after stripping leading whitespace.
fn is_block_marker(content: &str) -> bool {
    if content.is_empty() {
        return false;
    }
    let c = content.chars().next().unwrap();
    match c {
        '#' | '>' | '|' => true,
        '`' => content.starts_with("```"),
        '~' => content.starts_with("~~~"),
        '-' | '*' | '_' => {
            // Bullet list marker without whitespace (- foo / * foo) + three or more identical chars (hr --- / *** / ___)
            let rest = &content[1..];
            (content.chars().count() >= 3 && rest.chars().all(|c2| c2 == c))
                || rest.chars().next().is_some_and(|c2| c2.is_whitespace())
        }
        '0'..='9' => {
            // Ordered list: `1. foo` / `1) foo` (digit string followed by . or ) then whitespace)
            let digit_run: usize = content.chars().take_while(|c2| c2.is_ascii_digit()).count();
            let after = &content[digit_run..];
            matches!(after.chars().next(), Some('.') | Some(')'))
                && after[1..]
                    .chars()
                    .next()
                    .is_some_and(|c2| c2.is_whitespace())
        }
        _ => false,
    }
}

/// Protects leading whitespace on pure prose lines (prose-indent-preserve-fix, pi-aligned): pulldown-cmark per CommonMark
/// normalizes away leading whitespace on paragraph continuation/normal lines (`x\n   y` → `y` flush left), whereas pi's marked keeps the paragraph
/// raw text (including leading spaces and single newlines) as a whole, then outputs line by line. Fix: for lines that start with whitespace and whose content does not start with a block marker
/// (pure prose lines), replace leading whitespace with NBSP (pulldown treats it as normal text content and no longer strips it); during rendering
/// `push_text`/`push_cell_text` restore NBSP to spaces.
/// Skips fenced code blocks (``` / ~~~ variants paired by same character) and indented code blocks (block start & ≥4 spaces;
/// indented code blocks continue across lines with ≥4 spaces, and continuation lines also skip protection)—their content is preserved verbatim by pulldown and needs no protection.
/// Parameter: `text` is the original text after CRLF normalization; returns the protected text (line-break layout unchanged).
/// Edge case: genuine NBSP (in the user's original text) is also restored to a space (extremely rare; accepted deviation, noted in push_text).
fn protect_prose_indentation(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 16);
    let mut in_fence: Option<char> = None; // Current fence delimiter character (` or ~)
    let mut in_indented_code = false; // Inside an indented code block (lines with ≥4 spaces remain code and are not prose-protected)
    let mut prev_blank = true; // First line is treated as block start (≥4 spaces = indented code block, no protection)
    let mut first = true;

    for line in text.split('\n') {
        if line.is_empty() {
            out.push('\n');
            prev_blank = true;
            first = false;
            continue;
        }

        let trimmed = line.trim_start();
        let ws_count = line.len() - trimmed.len();
        let content = trimmed;

        // Indented code block continuation: lines with ≥4 spaces are kept as-is (CommonMark indented code continues across lines,
        // not bounded by blank lines—see the handling below where blank lines do not reset in_indented_code)
        if in_indented_code {
            if ws_count >= 4 {
                out.push_str(line);
                out.push('\n');
                prev_blank = false;
                first = false;
                continue;
            }
            in_indented_code = false; // Non-indented line → code block ends, handled as a new block
        }

        // Indented code block start (block start & ≥4 spaces)
        if (first || prev_blank) && ws_count >= 4 {
            in_indented_code = true;
            out.push_str(line);
            out.push('\n');
            prev_blank = false;
            first = false;
            continue;
        }

        // Fence boundary (≤3 indent): a ``` / ~~~ line with the same character as the current fence toggles state;
        // Opens when no fence is active, closes when a fence is active and the same character appears (a different character does not close; CommonMark semantics).
        if ws_count <= 3
            && ((content.starts_with("```") && in_fence != Some('~'))
                || (content.starts_with("~~~") && in_fence != Some('`')))
        {
            let fence_char = if content.starts_with("```") { '`' } else { '~' };
            in_fence = if in_fence == Some(fence_char) {
                None
            } else {
                Some(fence_char)
            };
            out.push_str(line);
            out.push('\n');
            prev_blank = false;
            first = false;
            continue;
        }

        // Code content inside a fence: as-is (pulldown does not parse it as markdown; no protection needed)
        if in_fence.is_some() {
            out.push_str(line);
            out.push('\n');
            prev_blank = false;
            first = false;
            continue;
        }

        // Pure prose line (leading whitespace & not a block marker): leading whitespace → NBSP protection (restored during rendering)
        if ws_count > 0 && !is_block_marker(content) {
            out.push_str(&"\u{00A0}".repeat(ws_count));
            out.push_str(content);
        } else {
            out.push_str(line);
        }
        out.push('\n');
        prev_blank = false;
        first = false;
    }
    out
}

/// Purely synchronous function; no TUI loop dependency, no cross-frame caching.
/// Current consumers: interactive message rendering (T-AP-11-2/assistant_message.rs), write_markdown
/// (shared by user/custom/skill messages in render.rs), print-mode message output.
/// For paragraph-level recoloring, use [`render_markdown_with_override`] (this entry point is byte-for-byte identical to its None variant).
pub fn render_markdown(theme: &MarkdownTheme, text: &str, width: usize) -> Vec<Vec<StyledSegment>> {
    render_markdown_with_override(theme, text, width, None)
}

/// Markdown rendering with paragraph-level style post-processing: assistant_message's thinking block calls it before the full paragraph is displayed,
/// so that text-stream segments within thinking (paragraph/heading/bold/italic/strikethrough/link text/table cell text) are all styled as
/// 「gray + italic」—reproduces pi's text-stream recoloring with `DefaultTextStyle { color, italic }`; when `None`, it is byte-for-byte identical to
/// [`render_markdown`] behavior (the original entry point delegates to this function with None).
/// Parameter: `style_override` paragraph-level style transform (changes only styling, not text/layout/line wrapping), applied to text-stream segments
/// (push_text/push_cell_text: the two plain-text entry points); **does not** apply to code-block syntax-highlighted segments, inline code,
/// fence borders, list markers, blockquote borders, table borders, and other structural elements (aligned with pi: the default style does not recolor code syntax highlighting,
/// Code blocks in thinking retain syntax colors; see the adjudicated decision in T-TU-02-1-fix).
/// Consumer: zen-app `assistant_message.rs` (thinking section, T-TU-02-1-fix).
pub fn render_markdown_with_override(
    theme: &MarkdownTheme,
    text: &str,
    width: usize,
    style_override: Option<&dyn Fn(Style) -> Style>,
) -> Vec<Vec<StyledSegment>> {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    if normalized.is_empty() || width == 0 {
        return Vec::new();
    }

    // pi-aligned (prose-indent-preserve-fix): protect leading whitespace on pure prose lines as NBSP before parsing—
    // pi's marked (breaks:false) paragraph token preserves the original line (including leading spaces and single newlines),
    // pulldown-cmark per CommonMark normalizes away leading whitespace on paragraph continuation lines, causing user-pasted
    // indented code/multiline prompts to lose indentation when displayed (see the softbreak-preserve-fix follow-up registered in tasks);
    // After parsing, during rendering (push_text/push_cell_text), restore NBSP to spaces.
    let protected = protect_prose_indentation(&normalized);

    let options =
        Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS;
    let parser = Parser::new_ext(&protected, options);

    let mut renderer = MarkdownRenderer::new(theme, width, style_override);
    for event in parser {
        renderer.process_event(event);
    }
    renderer.finish()
}

/// State machine for a single `render_markdown(_with_override)` call; not publicly exposed.
/// One mainline plus three buffers: `current_line` (styled segments of the current physical line, flushed via `wrap_segments`
/// into `output` at block end), `code_buf` (accumulates fenced code blocks, highlighted in one pass at end),
/// `table` (accumulates table cells, laid out in one pass at end).
struct MarkdownRenderer<'a, 'b> {
    theme: &'a MarkdownTheme,
    /// Segment-level style post-processing (thinking recoloring closure): applies only to text-stream segments
    /// (push_text/push_cell_text); structural elements/code-block highlighted segments are exempt—see
    /// [`render_markdown_with_override`] documentation.
    style_override: Option<&'b dyn Fn(Style) -> Style>,
    width: usize,

    /// Completed output lines (final result)
    output: Vec<Vec<StyledSegment>>,
    /// Segment currently accumulating (multiple StyledSegments within one line)
    current_line: Vec<StyledSegment>,

    /// Inline style stack: push for Strong/Emphasis/Strikethrough/Link, pop at end
    inline_stack: Vec<InlineStyle>,
    /// Block context stack: quote border / heading baseline / list item marker / ordered list counter
    block_stack: Vec<BlockContext>,

    /// Fenced code block accumulation buffer (highlight in one pass after collection completes)
    code_buf: String,
    code_lang: Option<String>,
    in_code_block: bool,

    /// Table accumulation state (established on Start(Table), consumed on End(Table))
    table: Option<TableState>,
}

/// Inline style element: determines the appended style for a text span (foreground color + attributes); popped when the closing tag is reached.
enum InlineStyle {
    /// `**bold**`
    Bold,
    /// `*italic*`
    Italic,
    /// `~~strikethrough~~`
    Strikethrough,
    /// `[text](url)`: carries the target URL and accumulated text; on completion, compares to decide whether to append the ` (url)` annotation
    Link { dest: String, text: String },
}

/// Block context: determines the line prefix (`│ ` / list marker and indentation) and the text baseline style.
enum BlockContext {
    /// Quote block (per-line prefix `│ `, text baseline = theme.quote)
    Blockquote,
    /// Heading (text baseline = combined heading style; h1 gets Bold+Underline)
    Heading(Style),
    /// List item: indent columns + marker text (including task checkbox suffix) + marker width + whether the marker has already been consumed
    ListItem {
        indent_cols: usize,
        marker_text: String,
        marker_width: usize,
        marker_used: bool,
    },
    /// Ordered list start: records the next ordinal to generate (established by `Tag::List(Some(start))`)
    OrderedList { next: u64 },
}

/// Table accumulation state: header row + data rows + current cell buffer.
#[derive(Default)]
struct TableState {
    header: Vec<Vec<StyledSegment>>,
    rows: Vec<Vec<Vec<StyledSegment>>>,
    current_cell: Vec<StyledSegment>,
    in_header: bool,
}

impl<'a, 'b> MarkdownRenderer<'a, 'b> {
    fn new(
        theme: &'a MarkdownTheme,
        width: usize,
        style_override: Option<&'b dyn Fn(Style) -> Style>,
    ) -> Self {
        Self {
            theme,
            style_override,
            width,
            output: Vec::new(),
            current_line: Vec::new(),
            inline_stack: Vec::new(),
            block_stack: Vec::new(),
            code_buf: String::new(),
            code_lang: None,
            in_code_block: false,
            table: None,
        }
    }

    // ── Event entry ──

    /// Handles a single pulldown-cmark event: block-level Start/End maintain the context stack; text/code events produce styled spans.
    fn process_event(&mut self, event: Event<'_>) {
        match event {
            Event::Start(tag) => self.start_tag(tag),
            Event::End(end) => self.end_tag(end),

            Event::Text(text) => {
                self.record_link_text(&text);
                if self.in_code_block {
                    self.code_buf.push_str(&text);
                } else if self.table.is_some() {
                    self.push_cell_text(&text);
                } else {
                    self.push_text(&text);
                }
            }

            Event::Code(code) => {
                let seg = StyledSegment {
                    style: self.theme.code.clone(),
                    text: code.to_string(),
                };
                self.emit_segment(seg);
            }

            Event::SoftBreak => {
                // pi-aligned (softbreak-preserve-fix): a single newline within a paragraph becomes a **newline** rather than a space—
                // pi's marked (breaks:false) leaves a single `\n` inside the paragraph text token, then via
                // applyTextWithNewlines / wrapTextWithAnsi explicitly split by line and preserve it (including streaming assistant
                // and user messages); zen previously folded SoftBreak into a space per CommonMark default, so pasted into the editor
                // multiline code/prompts were mashed into one line. After alignment: `a\nb` displays as two lines (matching pi), `a  \nb` (two spaces)
                // is still a hard break (HardBreak branch), and `a\n\nb` still yields a blank line between paragraphs (paragraph End).
                if self.table.is_some() {
                    self.push_cell_text(" ");
                } else {
                    self.flush_line();
                }
            }

            Event::HardBreak => {
                if self.table.is_some() {
                    self.push_cell_text(" ");
                } else {
                    self.flush_line();
                }
            }

            Event::Rule => {
                self.maybe_insert_blank_line();
                let hr_width = self.effective_width().clamp(3, 80);
                self.emit_segment(StyledSegment {
                    style: self.theme.hr.clone(),
                    text: "─".repeat(hr_width),
                });
                self.flush_line();
            }

            Event::Html(html) | Event::InlineHtml(html) => {
                if self.in_code_block {
                    self.code_buf.push_str(&html);
                } else if self.table.is_some() {
                    self.push_cell_text(&html);
                } else {
                    self.push_text(&html);
                }
            }

            Event::TaskListMarker(checked) => self.mark_task_item(checked),

            _ => {}
        }
    }

    // ── Block start ──

    fn start_tag(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => self.maybe_insert_blank_line(),

            Tag::Heading { level, .. } => {
                self.maybe_insert_blank_line();
                let mut style = self.theme.heading.clone();
                push_attr(&mut style, Attribute::Bold);
                if level == HeadingLevel::H1 {
                    push_attr(&mut style, Attribute::Underlined);
                }
                self.block_stack.push(BlockContext::Heading(style));
            }

            Tag::CodeBlock(kind) => {
                self.maybe_insert_blank_line();
                self.in_code_block = true;
                self.code_buf.clear();
                self.code_lang = match kind {
                    CodeBlockKind::Fenced(lang) => {
                        let lang_str = lang.to_string();
                        self.emit_segment(StyledSegment {
                            style: self.theme.code_block_border.clone(),
                            text: format!("```{lang_str}"),
                        });
                        self.flush_line();
                        if lang_str.is_empty() {
                            None
                        } else {
                            Some(lang_str)
                        }
                    }
                    CodeBlockKind::Indented => {
                        self.emit_segment(StyledSegment {
                            style: self.theme.code_block_border.clone(),
                            text: "```".to_string(),
                        });
                        self.flush_line();
                        None
                    }
                };
            }

            Tag::BlockQuote(_) => {
                self.maybe_insert_blank_line();
                self.block_stack.push(BlockContext::Blockquote);
            }

            Tag::List(start) => {
                // Compact list items have no Paragraph event; before a nested list starts, flush any residual text from the parent item
                self.flush_line();
                self.maybe_insert_blank_line();
                if let Some(start) = start {
                    self.block_stack
                        .push(BlockContext::OrderedList { next: start });
                }
            }

            Tag::Item => {
                let depth = self.list_depth();
                let bullet = self.next_bullet();
                let marker_width = visible_width(&bullet);
                self.block_stack.push(BlockContext::ListItem {
                    indent_cols: 4 * depth,
                    marker_text: bullet,
                    marker_width,
                    marker_used: false,
                });
            }

            Tag::Table(_) => {
                self.maybe_insert_blank_line();
                self.table = Some(TableState::default());
            }

            Tag::TableHead => {
                if let Some(t) = self.table.as_mut() {
                    t.in_header = true;
                    t.current_cell.clear();
                }
            }

            Tag::TableRow => {
                if let Some(t) = self.table.as_mut() {
                    // New data row: build the row buffer first, then clear the current cell
                    if !t.in_header {
                        t.rows.push(Vec::new());
                    }
                    t.current_cell.clear();
                }
            }

            Tag::TableCell => {
                if let Some(t) = self.table.as_mut() {
                    t.current_cell.clear();
                }
            }

            Tag::Strong => self.inline_stack.push(InlineStyle::Bold),
            Tag::Emphasis => self.inline_stack.push(InlineStyle::Italic),
            Tag::Strikethrough => self.inline_stack.push(InlineStyle::Strikethrough),
            Tag::Link { dest_url, .. } => self.inline_stack.push(InlineStyle::Link {
                dest: dest_url.to_string(),
                text: String::new(),
            }),

            // Image: v1 has no terminal image capability, so alt text flows through as plain text
            Tag::Image { .. } => {}
            _ => {}
        }
    }

    // ── Block end ──

    fn end_tag(&mut self, end: TagEnd) {
        match end {
            TagEnd::Paragraph => self.flush_line(),

            TagEnd::Heading(_) => {
                self.flush_line();
                self.block_stack.pop();
            }

            TagEnd::CodeBlock => {
                self.in_code_block = false;
                let lang_ref = self.code_lang.as_deref();
                let highlighted =
                    highlight::highlight(&self.code_buf, lang_ref, &self.theme.syntax_theme_name);
                let indent = self.theme.code_block_indent.clone();

                for row in highlighted {
                    self.emit_segment(StyledSegment {
                        style: self.code_line_style(Style::default()),
                        text: indent.clone(),
                    });
                    for seg in row {
                        self.emit_segment(StyledSegment {
                            style: self.code_line_style(seg.style),
                            text: seg.text,
                        });
                    }
                    self.flush_line();
                }

                self.emit_segment(StyledSegment {
                    style: self.theme.code_block_border.clone(),
                    text: "```".to_string(),
                });
                self.flush_line();
                self.code_buf.clear();
                self.code_lang = None;
            }

            TagEnd::BlockQuote(_) => {
                self.flush_line();
                self.block_stack.pop();
            }

            TagEnd::Item => {
                self.flush_line();
                // Empty-content list item (e.g. `-` with no body, or containing only a nested list): append a marker-only line
                let marker_unused = matches!(
                    self.block_stack.last(),
                    Some(BlockContext::ListItem {
                        marker_used: false,
                        ..
                    })
                );
                if marker_unused {
                    let prefix = self.context_prefix_segs(true);
                    if !prefix.is_empty() {
                        self.output.push(prefix);
                    }
                    self.mark_item_marker_used();
                }
                self.block_stack.pop();
            }

            TagEnd::List(_) => {
                if matches!(
                    self.block_stack.last(),
                    Some(BlockContext::OrderedList { .. })
                ) {
                    self.block_stack.pop();
                }
            }

            TagEnd::Strong => {
                self.inline_stack.pop();
            }
            TagEnd::Emphasis => {
                self.inline_stack.pop();
            }
            TagEnd::Strikethrough => {
                self.inline_stack.pop();
            }
            TagEnd::Link => {
                if let Some(InlineStyle::Link { dest, text }) = self.inline_stack.pop() {
                    let comparable = dest.trim().strip_prefix("mailto:").unwrap_or(dest.trim());
                    if text.trim() != comparable {
                        self.emit_segment(StyledSegment {
                            style: self.theme.link_url.clone(),
                            text: format!(" ({})", dest.trim()),
                        });
                    }
                }
            }

            TagEnd::TableCell => {
                if let Some(t) = self.table.as_mut() {
                    let cell = std::mem::take(&mut t.current_cell);
                    if t.in_header {
                        t.header.push(cell);
                    } else if let Some(row) = t.rows.last_mut() {
                        row.push(cell);
                    }
                }
            }

            TagEnd::TableHead => {
                if let Some(t) = self.table.as_mut() {
                    t.in_header = false;
                }
            }

            TagEnd::Table => {
                if let Some(t) = self.table.take() {
                    self.render_table(t);
                }
            }

            _ => {}
        }
    }

    // ── Text emission ──

    /// Accumulate text inside a link into the innermost `Link`'s acc, for the end tag to compare whether to append the URL annotation.
    fn record_link_text(&mut self, text: &str) {
        for s in self.inline_stack.iter_mut().rev() {
            if let InlineStyle::Link { text: acc, .. } = s {
                acc.push_str(text);
                break;
            }
        }
    }

    /// Emit a piece of text as a single style segment with the current inline styles; break lines at `\n`.
    /// The segment-level override is applied here (one of the text-stream entry points; see `apply_style_override`).
    /// **prose-indent-preserve-fix**: NBSP used during protection is restored to a normal space here (protection happens before parsing,
    /// so pulldown keeps paragraph continuation leading whitespace as text; restoring here yields the original indentation on pi's display side).
    fn push_text(&mut self, text: &str) {
        let style = self.apply_style_override(self.current_inline_style());
        for (i, part) in text.split('\n').enumerate() {
            if i > 0 {
                self.flush_line();
            }
            if !part.is_empty() {
                self.current_line.push(StyledSegment {
                    style: style.clone(),
                    text: if part.contains('\u{00A0}') {
                        part.replace('\u{00A0}', " ")
                    } else {
                        part.to_string()
                    },
                });
            }
        }
    }

    /// Text inside a table cell: collapse newlines to spaces, emit paragraphs into the current cell buffer.
    /// The segment-level override is applied here (the second text-stream entry point; as a result, cell content is grayed out along with thinking,
    /// Table borders/separators do not pass through here; preserve structural colors).
    /// **prose-indent-preserve-fix**: restore NBSP (protection placeholder) to spaces (consistent with push_text).
    fn push_cell_text(&mut self, text: &str) {
        let style = self.apply_style_override(self.current_inline_style());
        let cleaned = text.replace('\n', " ").replace('\u{00A0}', " ");
        if !cleaned.is_empty() {
            self.emit_segment(StyledSegment {
                style,
                text: cleaned,
            });
        }
    }

    /// Apply paragraph-level style post-processing when `style_override` is not None (recolor thinking);
    /// Return unchanged when None (byte-for-byte identical to `render_markdown`).
    fn apply_style_override(&self, style: Style) -> Style {
        match self.style_override {
            Some(f) => f(style),
            None => style,
        }
    }

    /// Write a paragraph to the current context: into the cell buffer inside a table, otherwise into the current row.
    fn emit_segment(&mut self, seg: StyledSegment) {
        if let Some(t) = self.table.as_mut() {
            t.current_cell.push(seg);
        } else {
            self.current_line.push(seg);
        }
    }

    /// Merge the current block context with the inline style stack into a single Style: baseline (text/quote/heading)
    /// layered with foreground color and attributes from inline elements.
    fn current_inline_style(&self) -> Style {
        let mut style = self.base_style();
        for s in &self.inline_stack {
            self.apply_inline(&mut style, s);
        }
        style
    }

    /// Text baseline style determined by the block context (quote: purple italic; heading: combined style; plain body).
    fn base_style(&self) -> Style {
        match self.block_stack.last() {
            Some(BlockContext::Heading(s)) => s.clone(),
            Some(BlockContext::Blockquote) => self.theme.quote.clone(),
            _ => self.theme.text.clone(),
        }
    }

    /// Inline element style composition: foreground color (if any) override, bitwise-OR of attribute flags, forced style attributes.
    fn apply_inline(&self, style: &mut Style, s: &InlineStyle) {
        match s {
            InlineStyle::Bold => {
                style.fg = self.theme.bold.fg.or(style.fg);
                merge_attrs(style, &self.theme.bold);
                push_attr(style, Attribute::Bold);
            }
            InlineStyle::Italic => {
                style.fg = self.theme.italic.fg.or(style.fg);
                merge_attrs(style, &self.theme.italic);
                push_attr(style, Attribute::Italic);
            }
            InlineStyle::Strikethrough => {
                style.fg = self.theme.strikethrough.fg.or(style.fg);
                merge_attrs(style, &self.theme.strikethrough);
                push_attr(style, Attribute::CrossedOut);
            }
            InlineStyle::Link { .. } => {
                style.fg = self.theme.link.fg.or(style.fg);
                merge_attrs(style, &self.theme.link);
                merge_attrs(style, &self.theme.underline);
                push_attr(style, Attribute::Underlined);
            }
        }
    }

    /// Mark the current list item as consumed (first visual line); subsequent lines degrade to plain indentation.
    fn mark_task_item(&mut self, checked: bool) {
        if let Some(BlockContext::ListItem {
            marker_text,
            marker_width,
            ..
        }) = self.block_stack.last_mut()
        {
            let suffix = if checked { "[x] " } else { "[ ] " };
            marker_text.push_str(suffix);
            *marker_width = visible_width(marker_text);
        }
    }

    /// Next bullet for an ordered list (`N. `); returns `- ` in non-ordered contexts.
    fn next_bullet(&mut self) -> String {
        for ctx in self.block_stack.iter_mut().rev() {
            if let BlockContext::OrderedList { next } = ctx {
                let bullet = format!("{next}. ");
                *next += 1;
                return bullet;
            }
        }
        "- ".to_string()
    }

    /// Number of currently open list item levels (nesting indentation basis).
    fn list_depth(&self) -> usize {
        self.block_stack
            .iter()
            .filter(|c| matches!(c, BlockContext::ListItem { .. }))
            .count()
    }

    /// Whether inside a list item (determines suppression of blank lines between blocks).
    fn in_list_item(&self) -> bool {
        self.block_stack
            .iter()
            .any(|c| matches!(c, BlockContext::ListItem { .. }))
    }

    // ── Line prefix and effective width ──

    /// Line prefix segment for the current block context: quote border `│ ` (quote_border color) + innermost list item's
    /// "indent spaces + bullet segment (list_bullet color) or bullet-width spaces". Outer list items are excluded
    /// (child items reposition from the left margin; parent continuations only affect the parent's own wrapping).
    /// Use the full bullet (e.g. `- `, `1. `, `- [x] `) when `first_line = true` and the innermost item's bullet is unconsumed.
    /// Read-only; does not modify state.
    fn context_prefix_segs(&self, first_line: bool) -> Vec<StyledSegment> {
        let innermost_item = self
            .block_stack
            .iter()
            .rposition(|c| matches!(c, BlockContext::ListItem { .. }));
        let mut segs = Vec::new();
        for (i, ctx) in self.block_stack.iter().enumerate() {
            match ctx {
                BlockContext::Blockquote => {
                    segs.push(StyledSegment {
                        style: self.theme.quote_border.clone(),
                        text: "│ ".to_string(),
                    });
                }
                BlockContext::ListItem {
                    indent_cols,
                    marker_width,
                    marker_text,
                    marker_used,
                } => {
                    if Some(i) != innermost_item {
                        continue;
                    }
                    if *indent_cols > 0 {
                        segs.push(StyledSegment {
                            style: Style::default(),
                            text: " ".repeat(*indent_cols),
                        });
                    }
                    if first_line && !*marker_used {
                        segs.push(StyledSegment {
                            style: self.theme.list_bullet.clone(),
                            text: marker_text.clone(),
                        });
                    } else if *marker_width > 0 {
                        segs.push(StyledSegment {
                            style: Style::default(),
                            text: " ".repeat(*marker_width),
                        });
                    }
                }
                BlockContext::Heading(_) | BlockContext::OrderedList { .. } => {}
            }
        }
        segs
    }

    /// Mark the innermost list item's first-line bullet as consumed.
    fn mark_item_marker_used(&mut self) {
        if let Some(i) = self
            .block_stack
            .iter()
            .rposition(|c| matches!(c, BlockContext::ListItem { .. }))
        {
            if let BlockContext::ListItem { marker_used, .. } = &mut self.block_stack[i] {
                *marker_used = true;
            }
        }
    }

    /// Line prefix styles no longer exist: prefix segments carry their own style (quote_border, list_bullet).
    /// Content effective width: subtract 2 columns per quote level (`│ `).
    fn effective_width(&self) -> usize {
        let quotes = self
            .block_stack
            .iter()
            .filter(|c| matches!(c, BlockContext::Blockquote))
            .count();
        self.width.saturating_sub(2 * quotes)
    }

    // ── Line flush and wrapping ──

    /// Wrap the current line and push into output: first visual line uses the first-line prefix (including list marker), continuation lines use aligned indent.
    fn flush_line(&mut self) {
        if self.current_line.is_empty() {
            return;
        }

        let first_prefix = self.context_prefix_segs(true);
        let rest_prefix = self.context_prefix_segs(false);
        self.mark_item_marker_used();

        let rest_width = rest_prefix
            .iter()
            .map(|s| visible_width(&s.text))
            .sum::<usize>();
        let content_width = self.effective_width().saturating_sub(rest_width);
        let pieces = if content_width == 0 {
            vec![std::mem::take(&mut self.current_line)]
        } else {
            wrap_segments(&self.current_line, content_width)
        };

        for (i, piece) in pieces.iter().enumerate() {
            let prefix = if i == 0 { &first_prefix } else { &rest_prefix };
            let mut line = Vec::new();
            line.extend(prefix.iter().cloned());
            line.extend(piece.iter().cloned());
            self.output.push(line);
        }
        self.current_line.clear();
    }

    /// Inter-block blank line insertion: skip after the first block or right after an existing blank line; skip inside list items (compact lists have no gaps).
    fn maybe_insert_blank_line(&mut self) {
        if self.output.is_empty() || self.in_list_item() {
            return;
        }
        let last_empty = matches!(self.output.last(), Some(l) if l.is_empty());
        if !last_empty {
            self.push_blank_line();
        }
    }

    /// Push a blank line; inside quotes, blank lines carry the `│ ` border (preserves visual quote continuity).
    fn push_blank_line(&mut self) {
        let quote_count = self
            .block_stack
            .iter()
            .filter(|c| matches!(c, BlockContext::Blockquote))
            .count();
        if quote_count == 0 {
            self.output.push(Vec::new());
            return;
        }
        let mut line = Vec::new();
        let mut text = String::new();
        for _ in 0..quote_count {
            text.push_str("│ ");
        }
        line.push(StyledSegment {
            style: self.theme.quote_border.clone(),
            text,
        });
        self.output.push(line);
    }

    /// Render end: flush leftover lines and strip trailing blank lines.
    fn finish(mut self) -> Vec<Vec<StyledSegment>> {
        if !self.current_line.is_empty() {
            self.flush_line();
        }
        while matches!(self.output.last(), Some(l) if l.is_empty()) {
            self.output.pop();
        }
        self.output
    }

    // ── Code block ──

    /// Code line style: spans already colored by syntect keep their foreground; when the foreground is missing, `code_block` is the fallback.
    /// **Background is no longer applied by this function** (T-TU-02-1-fix aligns with pi: code block background = surrounding environment background —
    /// assistant area = terminal default, inside a box = box background merged via render.rs box merge); `code_block.bg`
    /// is kept in the theme schema for future theming panels; the renderer does not consume it currently.
    fn code_line_style(&self, mut style: Style) -> Style {
        if style.fg.is_none() && self.theme.code_block.fg.is_some() {
            style.fg = self.theme.code_block.fg;
        }
        style
    }

    // ── Tables ──

    /// Full table rendering: column widths allocated by natural content width / min word width, `┌─┬─┐` style borders, cell wrapping,
    /// bold header row; falls back to plain text lines joined by ` | ` when extremely narrow.
    fn render_table(&mut self, t: TableState) {
        let cols = t.header.len();
        if cols == 0 {
            return;
        }
        let prefix = self.context_prefix_segs(false);
        let prefix_width: usize = prefix.iter().map(|s| visible_width(&s.text)).sum();
        let width = self.effective_width().saturating_sub(prefix_width);
        if width < 1 {
            return;
        }
        let overhead = 3 * cols + 1;
        let avail = width.saturating_sub(overhead);
        if avail < cols {
            self.fallback_table(t, width);
            return;
        }

        // Per-column natural width (max visible content width) and min word width (longest word ≤30 columns)
        let mut natural = vec![1usize; cols];
        let mut minw = vec![1usize; cols];
        for ci in 0..cols {
            let mut update_col = |cell: &[StyledSegment]| {
                let cell_text = concat_segs(cell);
                natural[ci] = natural[ci].max(visible_width(&cell_text));
                let longest_word = cell_text
                    .split_whitespace()
                    .map(visible_width)
                    .max()
                    .unwrap_or(0)
                    .min(30);
                minw[ci] = minw[ci].max(longest_word.max(1));
            };
            if ci < t.header.len() {
                update_col(&t.header[ci]);
            }
            for row in &t.rows {
                if ci < row.len() {
                    update_col(&row[ci]);
                }
            }
        }

        let total_natural = natural.iter().sum::<usize>() + overhead;
        let widths: Vec<usize> = if total_natural <= width {
            natural
        } else {
            let min_sum = minw.iter().sum::<usize>();
            if min_sum > avail {
                self.fallback_table(t, width);
                return;
            }
            let extra = avail - min_sum;
            let grow_caps: Vec<usize> = (0..cols)
                .map(|c| natural[c].saturating_sub(minw[c]))
                .collect();
            let total_grow: usize = grow_caps.iter().sum();
            let mut widths = minw.clone();
            if total_grow > 0 {
                for c in 0..cols {
                    // total_grow > 0 is already guarded above, checked_div is always Some; defensive fallback to 0
                    widths[c] += extra
                        .saturating_mul(grow_caps[c])
                        .checked_div(total_grow)
                        .unwrap_or(0);
                }
                let allocated: usize = (0..cols).map(|c| widths[c] - minw[c]).sum();
                let leftover = extra.saturating_sub(allocated);
                for _ in 0..leftover {
                    match (0..cols).find(|&c| widths[c] < natural[c].max(1)) {
                        Some(c) => widths[c] += 1,
                        None => break,
                    }
                }
            }
            widths
        };

        let mut lines: Vec<Vec<StyledSegment>> = Vec::new();
        lines.push(border_line(&widths, "┌─", "─┬─", "─┐"));

        // Header row (wrappable, bold)
        let header_cells: Vec<Vec<Vec<StyledSegment>>> = (0..cols)
            .map(|c| wrap_segments(&t.header[c], widths[c]))
            .collect();
        let header_lines = header_cells.iter().map(|l| l.len()).max().unwrap_or(0);
        for li in 0..header_lines {
            lines.push(build_row_line(&header_cells, &widths, li, true, self.theme));
        }
        lines.push(border_line(&widths, "├─", "─┼─", "─┤"));

        // Data rows
        for (ri, row) in t.rows.iter().enumerate() {
            let cells: Vec<Vec<Vec<StyledSegment>>> = (0..cols)
                .map(|c| {
                    if c < row.len() {
                        wrap_segments(&row[c], widths[c])
                    } else {
                        vec![Vec::new()]
                    }
                })
                .collect();
            let cell_lines = cells.iter().map(|l| l.len()).max().unwrap_or(1).max(1);
            for li in 0..cell_lines {
                lines.push(build_row_line(&cells, &widths, li, false, self.theme));
            }
            if ri + 1 < t.rows.len() {
                lines.push(border_line(&widths, "├─", "─┼─", "─┤"));
            }
        }
        lines.push(border_line(&widths, "└─", "─┴─", "─┘"));

        self.push_prefixed_lines(lines);
    }

    /// Extremely narrow fallback: only plain text lines separated by ` | ` (matches pi's raw fallback semantics, no borders).
    fn fallback_table(&mut self, t: TableState, width: usize) {
        let mut lines: Vec<Vec<StyledSegment>> = Vec::new();
        let mut rows: Vec<&Vec<Vec<StyledSegment>>> = vec![&t.header];
        for row in &t.rows {
            rows.push(row);
        }
        for row in rows {
            let text = row
                .iter()
                .map(|cell| concat_segs(cell))
                .collect::<Vec<_>>()
                .join(" | ");
            lines.extend(wrap_segments(
                &[StyledSegment {
                    style: Style::default(),
                    text,
                }],
                width,
            ));
        }
        self.push_prefixed_lines(lines);
    }

    /// Push the full group of already-laid-out lines into output (uniformly prepends the current context prefix, e.g. quote border/list indent).
    fn push_prefixed_lines(&mut self, lines: Vec<Vec<StyledSegment>>) {
        let prefix = self.context_prefix_segs(false);
        for mut line in lines {
            let mut segs = prefix.clone();
            segs.append(&mut line);
            self.output.push(segs);
        }
    }
}

// ── Private pure-function helpers ──

/// Wrap styled-segment lines at visible column word boundaries: concatenate text → `word_wrap_line` yields byte ranges → map back to style segments.
/// (shared by table cells and main lines)
fn wrap_segments(segs: &[StyledSegment], max_width: usize) -> Vec<Vec<StyledSegment>> {
    if max_width == 0 {
        return vec![segs.to_vec()];
    }
    let text: String = concat_segs(segs);
    word_wrap_line(&text, max_width)
        .into_iter()
        .map(|chunk| extract_segments(segs, chunk.start_index, chunk.end_index))
        .collect()
}

/// Extract the segment covering byte range [start, end) from a sequence of style segments (splits never break UTF-8 character boundaries).
fn extract_segments(segs: &[StyledSegment], start: usize, end: usize) -> Vec<StyledSegment> {
    let mut result = Vec::new();
    let mut byte_offset = 0;
    for seg in segs {
        let seg_end = byte_offset + seg.text.len();
        if seg_end <= start {
            byte_offset = seg_end;
            continue;
        }
        if byte_offset >= end {
            break;
        }
        let take_start = start.saturating_sub(byte_offset);
        let take_end = (end - byte_offset).min(seg.text.len());
        if take_start < take_end {
            let text: String = seg
                .text
                .chars()
                .scan(0, |byte_pos, ch| {
                    let ch_start = *byte_pos;
                    *byte_pos += ch.len_utf8();
                    Some((ch_start, ch))
                })
                .filter(|(pos, _)| *pos >= take_start && *pos < take_end)
                .map(|(_, ch)| ch)
                .collect();
            if !text.is_empty() {
                result.push(StyledSegment {
                    style: seg.style.clone(),
                    text,
                });
            }
        }
        byte_offset = seg_end;
    }
    result
}

/// Concatenated text of the styled segments.
fn concat_segs(segs: &[StyledSegment]) -> String {
    segs.iter().map(|s| s.text.as_str()).collect()
}

/// Union of properties, appended.
fn push_attr(style: &mut Style, attr: Attribute) {
    if !style.attributes.contains(&attr) {
        style.attributes.push(attr);
    }
}

/// Merge the source style property union into the target.
fn merge_attrs(target: &mut Style, src: &Style) {
    for attr in &src.attributes {
        push_attr(target, *attr);
    }
}

/// Table border row (`┌──┬──┐` style, `─` repeated per column width).
fn border_line(widths: &[usize], left: &str, mid: &str, right: &str) -> Vec<StyledSegment> {
    let mut text = String::from(left);
    for (i, w) in widths.iter().enumerate() {
        text.push_str(&"─".repeat(*w));
        if i + 1 < widths.len() {
            text.push_str(mid);
        }
    }
    text.push_str(right);
    vec![StyledSegment {
        style: Style::default(),
        text,
    }]
}

/// Table row: `│ ` + each column (pad with spaces up to column width, leave blank for extra rows) + ` │`;
/// Apply bold + `bold` foreground to the header row as a whole.
fn build_row_line(
    cells: &[Vec<Vec<StyledSegment>>],
    widths: &[usize],
    line_idx: usize,
    is_header: bool,
    theme: &MarkdownTheme,
) -> Vec<StyledSegment> {
    let mut line = vec![StyledSegment {
        style: Style::default(),
        text: "│ ".to_string(),
    }];
    for (ci, cell) in cells.iter().enumerate() {
        if ci > 0 {
            line.push(StyledSegment {
                style: Style::default(),
                text: " │ ".to_string(),
            });
        }
        let mut cell_text = if line_idx < cell.len() {
            cell[line_idx].clone()
        } else {
            Vec::new()
        };
        let width = widths[ci];
        let used = visible_width(&concat_segs(&cell_text));
        if used < width {
            cell_text.push(StyledSegment {
                style: Style::default(),
                text: " ".repeat(width - used),
            });
        }
        if is_header {
            for seg in &mut cell_text {
                seg.style.fg = theme.bold.fg.or(seg.style.fg);
                push_attr(&mut seg.style, Attribute::Bold);
            }
        }
        line.extend(cell_text);
    }
    line.push(StyledSegment {
        style: Style::default(),
        text: " │".to_string(),
    });
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(text: &str, width: usize) -> Vec<Vec<StyledSegment>> {
        render_markdown(&MarkdownTheme::default(), text, width)
    }

    /// Concatenated text per row.
    fn lines_text(lines: &[Vec<StyledSegment>]) -> Vec<String> {
        lines.iter().map(|l| concat_segs(l)).collect()
    }

    /// Full text (including prefix).
    fn join(lines: &[Vec<StyledSegment>]) -> String {
        lines_text(lines).join("\n")
    }

    /// The first styled segment whose text contains `needle`.
    fn find_seg<'a>(lines: &'a [Vec<StyledSegment>], needle: &str) -> Option<&'a StyledSegment> {
        lines.iter().flatten().find(|s| s.text.contains(needle))
    }

    // ── Entry point and normalization ──

    #[test]
    fn test_empty_input_returns_empty() {
        assert!(render("", 80).is_empty());
        assert!(render("\r\n", 80).is_empty());
    }

    #[test]
    fn test_zero_width_returns_empty() {
        assert!(render("hello", 0).is_empty());
    }

    #[test]
    fn test_crlf_normalized() {
        // pi alignment (softbreak-preserve-fix): single newline = SoftBreak → **preserved as newline** (consistent with pi,
        // marked paragraph text contains `\n`; wrapTextWithAnsi processes line by line); after `\r` is normalized, no residual characters remain.
        let lines = render("a\r\nb", 80);
        assert_eq!(lines_text(&lines), vec!["a".to_string(), "b".to_string()]);
        // Two trailing spaces + newline = hard line break: produces two lines (pulldown swallows trailing spaces)
        let lines = render("a  \r\nb", 80);
        assert_eq!(lines_text(&lines), vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn test_plain_paragraph_wraps_at_width() {
        // Wrapping preserves the original whitespace (whitespace is attributed to the end of the previous fragment, matching wrap::word_wrap_line semantics)
        let lines = render("hello world foo bar", 10);
        assert_eq!(
            lines_text(&lines),
            vec![
                "hello ".to_string(),
                "world foo ".to_string(),
                "bar".to_string()
            ]
        );
    }

    #[test]
    fn test_blank_between_blocks() {
        let lines = render("a\n\nb", 80);
        assert_eq!(
            lines_text(&lines),
            vec!["a".to_string(), "".to_string(), "b".to_string()]
        );
    }

    // ── prose indentation preservation (prose-indent-preserve-fix, pi-aligned) ──

    #[test]
    fn test_softbreak_keeps_paragraph_continuation_indent() {
        // pi (marked) paragraph preserves original line leading whitespace; zen is consistent after NBSP protection + render restoration
        let lines = render("x\n   y", 80);
        assert_eq!(
            lines_text(&lines),
            vec!["x".to_string(), "   y".to_string()]
        );
    }

    #[test]
    fn test_pasted_match_body_keeps_newline_and_indent() {
        // User-tested scenario: multi-line indentation of pasted Rust code is preserved verbatim (consistent with pi)
        let lines = render(
            "文件 @apps/server/src/llm/mod.rs 中第415~418行的这段代码:\n let err = match outcome {\n                 Ok(value) => return Ok(value),\n                 Err(err) => err,\n             };",
            240,
        );
        assert_eq!(
            lines_text(&lines),
            vec![
                "文件 @apps/server/src/llm/mod.rs 中第415~418行的这段代码:".to_string(),
                " let err = match outcome {".to_string(),
                "                 Ok(value) => return Ok(value),".to_string(),
                "                 Err(err) => err,".to_string(),
                "             };".to_string(),
            ]
        );
    }

    #[test]
    fn test_pasted_comment_lines_keep_indent() {
        // User-tested scenario: /// comment lines preserve 4-space indentation (no wrapping within width)
        let lines = render(
            "文件 @..llm/mod.rs 中第251~259行的这段代码：\nrust\n    /// 构建客户端：用配置超时装配 reqwest::Client。\n    /// 输入：config——LLM 路由配置。\n    pub fn new(config: LlmConfig) {\n        todo!()\n    }",
            140,
        );
        assert_eq!(
            lines_text(&lines),
            vec![
                "文件 @..llm/mod.rs 中第251~259行的这段代码：".to_string(),
                "rust".to_string(),
                "    /// 构建客户端：用配置超时装配 reqwest::Client。".to_string(),
                "    /// 输入：config——LLM 路由配置。".to_string(),
                "    pub fn new(config: LlmConfig) {".to_string(),
                "        todo!()".to_string(),
                "    }".to_string(),
            ]
        );
    }

    #[test]
    fn test_nested_list_untouched_by_indent_protection() {
        // Nested lists (2/4-space indentation + list markers) are still parsed as lists, not polluted by prose protection
        let lines = render("- a\n  - b\n    - c", 80);
        assert_eq!(
            lines_text(&lines),
            vec![
                "- a".to_string(),
                "    - b".to_string(),
                "        - c".to_string(),
            ]
        );
    }

    #[test]
    fn test_fenced_code_indented_content_untouched() {
        // Indented lines inside fenced code stay as-is (protection skips fences)
        let lines = render("```rust\n    let x = 1;\n```", 80);
        let text = lines_text(&lines);
        assert!(
            text.iter().any(|l| l.contains("let x = 1;")),
            "code line: {text:?}"
        );
        assert_eq!(text[0], "```rust");
        assert_eq!(*text.last().unwrap(), "```");
    }

    #[test]
    fn test_indented_code_block_continuation_untouched() {
        // Indented code blocks remain continuous across lines (a second line with ≥4 spaces still belongs to code, not split by prose protection)
        let lines = render("text\n\n    indented code\n    第二行", 80);
        let text = lines_text(&lines);
        // Both lines are in the same code block (``` fence is continuous, no prose line in between)
        let f0 = text.iter().position(|l| l == "```").expect("Opening fence");
        let f1 = text.iter().rposition(|l| l == "```").expect("Closing fence");
        assert_eq!(f1 - f0, 3, "Code block contains two lines: {text:?}");
        assert!(text[f0 + 1].contains("indented code"));
        assert!(text[f0 + 2].contains("第二行"));
    }

    #[test]
    fn test_indented_heading_still_heading() {
        // A heading line with 2-space indentation is still a heading (protection skips block markers)
        let lines = render("para\n  # heading\nnext", 80);
        let text = lines_text(&lines);
        assert!(text.contains(&"heading".to_string()));
        assert!(!text.contains(&"# heading".to_string()));
    }

    #[test]
    fn test_indented_bold_prose_keeps_indent_and_format() {
        // Bold prose with leading indentation: indentation preserved + inline formatting still applies
        let lines = render("para\n  **加粗** 和 `code`", 80);
        assert_eq!(
            lines_text(&lines),
            vec!["para".to_string(), "  加粗 和 code".to_string()]
        );
    }

    // ── Headings ──

    #[test]
    fn test_h1_heading_style_bold_and_underline() {
        let lines = render("# Title", 80);
        let seg = find_seg(&lines, "Title").expect("Title text exists");
        let theme = MarkdownTheme::default();
        assert_eq!(seg.style.fg, theme.heading.fg);
        assert!(seg.style.attributes.contains(&Attribute::Bold));
        assert!(seg.style.attributes.contains(&Attribute::Underlined));
    }

    #[test]
    fn test_h2_heading_style_bold_no_underline() {
        let lines = render("## Title", 80);
        let seg = find_seg(&lines, "Title").expect("Title text exists");
        let theme = MarkdownTheme::default();
        assert!(seg.style.attributes.contains(&Attribute::Bold));
        assert!(!seg.style.attributes.contains(&Attribute::Underlined));
        assert_eq!(seg.style.fg, theme.heading.fg);
    }

    // ── Inline styles ──

    #[test]
    fn test_bold_italic_strikethrough() {
        let lines = render("**b** *i* ~~s~~", 80);
        let theme = MarkdownTheme::default();
        let bold = find_seg(&lines, "b").expect("Bold text");
        assert!(bold.style.attributes.contains(&Attribute::Bold));
        assert_eq!(bold.style.fg, theme.bold.fg);
        let italic = find_seg(&lines, "i").expect("Italic text");
        assert!(italic.style.attributes.contains(&Attribute::Italic));
        assert_eq!(italic.style.fg, theme.italic.fg);
        let strike = find_seg(&lines, "s").expect("Strikethrough text");
        assert!(strike.style.attributes.contains(&Attribute::CrossedOut));
        assert_eq!(strike.style.fg, theme.strikethrough.fg);
    }

    #[test]
    fn test_inline_code_style() {
        let lines = render("run `cargo build` now", 80);
        let seg = find_seg(&lines, "cargo build").expect("Inline code");
        assert_eq!(seg.style.fg, MarkdownTheme::default().code.fg);
    }

    #[test]
    fn test_link_styled_and_url_appended_when_different() {
        let lines = render("[text](https://example.com)", 80);
        let theme = MarkdownTheme::default();
        let text_seg = find_seg(&lines, "text").expect("Link text");
        assert!(text_seg.style.attributes.contains(&Attribute::Underlined));
        assert_eq!(text_seg.style.fg, theme.link.fg);
        let joined = join(&lines);
        assert!(joined.contains("(https://example.com)"));
        assert_eq!(
            find_seg(&lines, "example.com").map(|s| s.style.fg),
            Some(theme.link_url.fg)
        );
    }

    #[test]
    fn test_link_no_url_when_text_equals_href() {
        let lines = render("[https://example.com](https://example.com)", 80);
        assert_eq!(join(&lines), "https://example.com");
    }

    // ── Code block ──

    #[test]
    fn test_code_fence_with_highlight_and_indent() {
        let lines = render("```rust\nfn main() {}\n```", 80);
        let text_lines = lines_text(&lines);
        assert_eq!(text_lines[0], "```rust");
        assert_eq!(text_lines[1], "  fn main() {}");
        assert_eq!(text_lines[2], "```");
        // Syntax highlighting: at least one segment has an explicit foreground color
        assert!(
            lines[1].iter().any(|s| s.style.fg.is_some()),
            "Rust code line should have colored segments"
        );
        // T-TU-02-1-fix: code block background is no longer applied by the renderer (aligned with pi environment background) — all code line span bg are None
        assert!(
            lines[1].iter().all(|s| s.style.bg.is_none()),
            "Code line should not have a panel background (background = environment)"
        );
    }

    #[test]
    fn test_code_block_indented_fence() {
        let lines = render("```\nplain code\n```", 80);
        let text_lines = lines_text(&lines);
        assert_eq!(text_lines[0], "```");
        assert_eq!(text_lines[1], "  plain code");
        assert_eq!(text_lines[2], "```");
    }

    // ── Lists ──

    #[test]
    fn test_unordered_list_bullet_no_blank_between_items() {
        let lines = render("- a\n- b", 80);
        let text_lines = lines_text(&lines);
        assert_eq!(text_lines, vec!["- a".to_string(), "- b".to_string()]);
        let theme = MarkdownTheme::default();
        let marker = &lines[0][0];
        assert_eq!(marker.text, "- ");
        assert_eq!(marker.style.fg, theme.list_bullet.fg);
    }

    #[test]
    fn test_ordered_list_start_number() {
        let lines = render("3. x\n4. y", 80);
        let text_lines = lines_text(&lines);
        assert_eq!(text_lines, vec!["3. x".to_string(), "4. y".to_string()]);
    }

    #[test]
    fn test_nested_list_indent() {
        let lines = render("- a\n  - b", 80);
        assert_eq!(
            lines_text(&lines),
            vec!["- a".to_string(), "    - b".to_string()]
        );
    }

    #[test]
    fn test_task_list_marker() {
        let lines = render("- [x] done\n- [ ] todo", 80);
        let text_lines = lines_text(&lines);
        assert_eq!(
            text_lines,
            vec!["- [x] done".to_string(), "- [ ] todo".to_string()]
        );
    }

    #[test]
    fn test_empty_list_item_renders_bare_marker() {
        let lines = render("- a\n-\n- b", 80);
        assert_eq!(
            lines_text(&lines),
            vec!["- a".to_string(), "- ".to_string(), "- b".to_string()]
        );
    }

    #[test]
    fn test_list_item_wrapping_continuation_alignment() {
        // Wrapping wide content: continuation lines are aligned with marker-width spaces (whitespace attributed to the end of the previous fragment)
        let lines = render("- aaaaaaaa bbbbbbbb cccccccc", 14);
        let text_lines = lines_text(&lines);
        assert_eq!(text_lines[0], "- aaaaaaaa ");
        assert_eq!(text_lines[1], "  bbbbbbbb ");
        assert_eq!(text_lines[2], "  cccccccc");
    }

    // ── Blockquotes ──

    #[test]
    fn test_blockquote_border_and_quote_style() {
        let lines = render("> quote", 80);
        assert_eq!(lines_text(&lines), vec!["│ quote".to_string()]);
        let theme = MarkdownTheme::default();
        assert_eq!(lines[0][0].style.fg, theme.quote_border.fg);
        assert_eq!(lines[0][0].text, "│ ");
        let text_seg = &lines[0][1];
        assert_eq!(text_seg.style.fg, theme.quote.fg);
        assert!(text_seg.style.attributes.contains(&Attribute::Italic));
    }

    #[test]
    fn test_blockquote_blank_line_keeps_border() {
        let lines = render("> a\n>\n> b", 80);
        let text_lines = lines_text(&lines);
        assert_eq!(
            text_lines,
            vec!["│ a".to_string(), "│ ".to_string(), "│ b".to_string()]
        );
    }

    // ── Horizontal rules and tables ──

    #[test]
    fn test_hr() {
        let lines = render("---", 80);
        let theme = MarkdownTheme::default();
        assert!(lines[0]
            .iter()
            .any(|s| s.style.fg == theme.hr.fg && s.text.contains('─')));
    }

    #[test]
    fn test_table_borders_and_cells() {
        let md = "| a | b |\n|---|---|\n| c | d |";
        let lines = render(md, 40);
        let text_lines = lines_text(&lines);
        // Border format aligned with pi: `┌─` + one `─` per column width + `─┬─` + ... + `─┐` (column width 1 → 3 `─`)
        assert!(text_lines.contains(&"┌───┬───┐".to_string()));
        assert!(text_lines.contains(&"│ a │ b │".to_string()));
        assert!(text_lines.contains(&"│ c │ d │".to_string()));
        assert!(text_lines.contains(&"├───┼───┤".to_string()));
        assert!(text_lines.contains(&"└───┴───┘".to_string()));
    }

    #[test]
    fn test_table_wide_cell_shrinks_to_width() {
        // Column widths shrink to available bandwidth; no row overflows
        let md = "| header | c |\n|--------|---|\n| aaaa bbbb cccc dddd | e |";
        let lines = render(md, 20);
        let text_lines = lines_text(&lines);
        assert!(
            text_lines.iter().all(|l| visible_width(l) <= 20),
            "Line too wide: {text_lines:?}"
        );
        assert!(text_lines.iter().any(|l| l.contains("aaaa")));
        assert!(text_lines.iter().any(|l| l.contains('┌')));
    }

    #[test]
    fn test_table_too_narrow_falls_back_to_text() {
        let md = "| a | b |\n|---|---|\n| c | d |";
        let lines = render(md, 8);
        let text_lines = lines_text(&lines);
        assert!(!text_lines.iter().any(|l| l.contains('┌')));
        assert!(text_lines.iter().any(|l| l.contains("a | b")));
    }

    // ── Theme defaults ──

    #[test]
    fn test_default_theme_all_styles_defined() {
        let theme = MarkdownTheme::default();
        assert_eq!(theme.code_block_indent, "  ");
        for (name, style) in [
            ("text", &theme.text),
            ("heading", &theme.heading),
            ("bold", &theme.bold),
            ("italic", &theme.italic),
            ("strikethrough", &theme.strikethrough),
            ("underline", &theme.underline),
            ("link", &theme.link),
            ("link_url", &theme.link_url),
            ("code", &theme.code),
            ("code_block", &theme.code_block),
            ("code_block_border", &theme.code_block_border),
            ("quote", &theme.quote),
            ("quote_border", &theme.quote_border),
            ("hr", &theme.hr),
            ("list_bullet", &theme.list_bullet),
        ] {
            assert!(style.fg.is_some(), "{name} should have foreground color");
        }
        assert!(theme.code_block.bg.is_some(), "code_block should have background color");
    }

    #[test]
    fn test_style_functions_compose_attributes() {
        // Italic nested inside bold: attribute union
        let lines = render("***x***", 80);
        // pulldown: Emphasis(Strong)
        let seg = find_seg(&lines, "x").expect("Nested inline text");
        assert!(seg.style.attributes.contains(&Attribute::Bold));
        assert!(seg.style.attributes.contains(&Attribute::Italic));
    }

    // ── render_markdown_with_override (T-TU-02-1-fix paragraph-level recoloring) ──

    /// Override closure test double: fixed magenta foreground + forced italics (aligned with thinking recoloring semantics).
    fn override_magenta() -> impl Fn(Style) -> Style {
        |mut s: Style| {
            s.fg = Some(Color::Magenta);
            push_attr(&mut s, Attribute::Italic);
            s
        }
    }

    #[test]
    fn test_override_none_equals_render_markdown() {
        // The None case is byte-for-byte identical to render_markdown (a prerequisite for the original entry point to delegate to this function)
        let text = "# T\n\n**b** *i* `c`\n\n- a\n- b\n\n```rust\nfn main() {}\n```\n\n> q\n";
        let theme = MarkdownTheme::default();
        let plain = render_markdown(&theme, text, 60);
        let via_override = render_markdown_with_override(&theme, text, 60, None);
        assert_eq!(plain, via_override, "None form should be byte-for-byte identical");
    }

    #[test]
    fn test_override_applies_to_text_flow_not_code() {
        let text = "para **bold** `inline`\n\n```rust\nfn main() {}\n```";
        let theme = MarkdownTheme::default();
        let rows = render_markdown_with_override(&theme, text, 80, Some(&override_magenta()));
        // Paragraph text: foreground overridden to magenta + italics
        let para = find_seg(&rows, "para").expect("Paragraph text");
        assert_eq!(para.style.fg, Some(Color::Magenta));
        assert!(para.style.attributes.contains(&Attribute::Italic));
        // Bold text: foreground is also overridden (text stream); bold attribute preserved
        let bold = find_seg(&rows, "bold").expect("Bold text");
        assert_eq!(bold.style.fg, Some(Color::Magenta));
        assert!(bold.style.attributes.contains(&Attribute::Bold));
        // Inline code: preserve theme.code color (structural elements are exempt, matching pi codespan semantics)
        let inline = find_seg(&rows, "inline").expect("Inline code");
        assert_eq!(inline.style.fg, theme.code.fg);
        assert!(!inline.style.attributes.contains(&Attribute::Italic));
        // Code block: preserve syntax highlighting (at least one segment has fg that is neither magenta nor None, with no override italics)
        let code_line = rows
            .iter()
            .find(|l| concat_segs(l).contains("fn main"))
            .expect("Code line");
        assert!(
            code_line.iter().any(|s| s.style.fg.is_some()
                && s.style.fg != Some(Color::Magenta)
                && !s.style.attributes.contains(&Attribute::Italic)),
            "thinking code block should preserve syntax highlighting"
        );
        // fence border: preserve the code_block_border color (structural element exemption)
        let fence = find_seg(&rows, "```rust").expect("fence line");
        assert_eq!(fence.style.fg, theme.code_block_border.fg);
    }

    #[test]
    fn test_override_applies_to_heading_text() {
        let theme = MarkdownTheme::default();
        let rows =
            render_markdown_with_override(&theme, "# Title\n\ntext", 60, Some(&override_magenta()));
        let seg = find_seg(&rows, "Title").expect("Heading text");
        assert_eq!(seg.style.fg, Some(Color::Magenta));
        assert!(seg.style.attributes.contains(&Attribute::Bold));
        assert!(seg.style.attributes.contains(&Attribute::Italic));
    }

    #[test]
    fn test_override_applies_to_table_cells_not_borders() {
        let text = "| a | b |\n|---|---|\n| 1 | 2 |";
        let theme = MarkdownTheme::default();
        let rows = render_markdown_with_override(&theme, text, 40, Some(&override_magenta()));
        let joined = join(&rows);
        assert!(joined.contains('┌'), "Table border exists");
        // Data cell content: text stream → overwritten
        assert_eq!(find_seg(&rows, "1").unwrap().style.fg, Some(Color::Magenta));
        assert_eq!(find_seg(&rows, "2").unwrap().style.fg, Some(Color::Magenta));
        // Header cell: build_row_line uses theme.bold for foreground emphasis (matching pi's bold header wrapping),
        // The override color is replaced by the header emphasis color—structural semantics, not the text stream
        assert_eq!(
            find_seg(&rows, "a").unwrap().style.fg,
            Some(theme.bold.fg.unwrap())
        );
        // Border segment: constructed directly with Style::default, bypassing the text stream → not overridden (fg None)
        let border = rows
            .iter()
            .flatten()
            .find(|s| s.text.contains('┌'))
            .expect("Border segment");
        assert_eq!(border.style.fg, None);
    }
}
