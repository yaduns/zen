// REQ-APP-10 : syntax highlighting — syntect implementation (highlight.js is not ported).
// Placement exception: the spec module lands in zen-tui (consumer = zen-tui's markdown renderer T-TU-02-1; the dependency direction forbids landing in zen-app);
// see the placement exception note under "REQ prefix → crate mapping" in AGENTS.md and tasks/T-AP-10-1-implemented.md.
// Ecosystem paradigm: syntect `(Style, &str)` spans map directly onto zen `screen::Style` carriers (text styles always go through Style,
// never embedded ANSI); pi's hljs HTML intermediate layer / renderHighlightedHtml / decodeHtmlEntityAt are dissolved by dropping export-html.
// No counterpart for highlightAuto/languageSubset/ignoreIllegals (fixed parse chain + syntect is inherently fault-tolerant).
// Output is organized by logical line, 1:1 with input lines (an empty line yields an empty Vec), so the renderer can place cells line by line.
use crate::screen::Style;
use crossterm::style::{Attribute, Color};
use syntect::easy::HighlightLines;
use syntect::highlighting::{
    FontStyle, ScopeSelectors, StyleModifier, Theme, ThemeItem, ThemeSettings,
};
use syntect::parsing::{SyntaxReference, SyntaxSet};

/// A continuous run of text with uniform styling—the minimal unit of the highlight output (carries "multiple styled spans within one line").
/// Produced by `highlight` and written cell by cell via `ScreenBuffer::write` by the renderer (markdown renderer),
/// width calculation reuses the existing unicode-width machinery; the renderer need not understand syntax coloring internals.
/// Pure runtime rendering data; not serialized.
#[derive(Debug, Clone, PartialEq)]
pub struct StyledSegment {
    /// Style of this span; `fg` = syntect foreground (RGB), `attributes` = folded font bits,
    /// `bg` is always `None` (design decision: the code block background is laid down uniformly by the display component, no per-token coloring).
    pub style: Style,
    /// Text content of this span (already split by line, newline excluded).
    pub text: String,
}

/// Language alias mapping: languages missing from syntect's built-in syntax set (75) → covered by their (superset) equivalent syntax.
/// Rationale: highlight.js's `typescript` grammar is essentially the `javascript` grammar plus a handful of TS keywords—
/// mapping ts→js gives highlighting strength equivalent to hljs. Of the Rust/TS/Python/Go/JSON/YAML/Markdown requirement list,
/// all but TS are natively in the set (guarded by test_required_language_set_highlightable).
/// We do not introduce a third-party TypeScript.sublime-syntax just for this (it would require vendored assets + a full runtime set rebuild).
const LANG_ALIASES: &[(&str, &str)] = &[
    ("ts", "js"),
    ("typescript", "js"),
    ("tsx", "js"),
    ("jsx", "js"),
];

/// List of supported languages explicitly required by REQ-APP-10 (Rust/TS/Python/Go/JSON/YAML/Markdown).
/// Runtime support comes from native matches in the built-in syntect syntax set (direct token/extension lookup in the lower-level parsing chain) + LANG_ALIASES to fill gaps
/// jointly determined—all rust/python/go/json/yaml/markdown entries in this list are natively matched and require no alias entries,
/// Only `typescript` is mapped via an alias to the JS superset (see LANG_ALIASES).
/// This constant is only an explicit code-level declaration of the requirements list, used by the iteration guard test (test_required_language_set_highlightable),
/// It is not used during parsing; if the requirements list changes in the future (adding/removing languages), update this location + the tests to keep them in sync.
#[cfg(test)]
const REQUIRED_LANGUAGES: [&str; 7] = [
    "rust",
    "typescript",
    "python",
    "go",
    "json",
    "yaml",
    "markdown",
];

/// Syntax set singleton: embeds 160+ grammars (workspace syntect feature `default-syntaxes`),
/// lazily initialized once per process; shared by language resolution (resolve_syntax) and highlighting to avoid duplicate loading.
static SYNTAX_SET: std::sync::OnceLock<SyntaxSet> = std::sync::OnceLock::new();

/// Syntax set accessor: initialized via `SyntaxSet::load_defaults_newlines()` on first call
/// (a heavy one-time load, hosted by a OnceLock lazily), afterwards returns a 'static reference. Consumed by both resolve_syntax and
/// highlight execution — "load once, live in memory, grab it at hand".
fn syntax_set() -> &'static SyntaxSet {
    SYNTAX_SET.get_or_init(SyntaxSet::load_defaults_newlines)
}

// ── Theme-derived syntax palette (T-AP-10-1-fix) ──────────────────────────────────

/// Syntax coloring palette handed from the theme to the highlighter — the single surface through which
/// theme colors enter code blocks (before T-AP-10-1-fix the palette was a hardcoded syntect built-in theme,
/// so the theme's nine `syntax*` tokens never reached the code block).
/// Built by `Theme::syntax_palette()`, carried on `MarkdownTheme`, and turned into a one-shot syntect
/// `Theme` by [`build_syntect_theme`] inside [`highlight`].
/// Each slot is `None` when the theme leaves that token as the terminal default (`""` → `Color::Reset`),
/// in which case no rule is emitted and the text keeps the base/fallback color.
/// Pure runtime rendering data; `Copy` (11 resolved colors), never serialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyntaxPalette {
    /// Comments (`comment.*`, and `punctuation.definition.comment`). pi: `comment`/`doctag` → `fg("syntaxComment")`.
    pub comment: Option<Color>,
    /// Keywords (`keyword.*`, `storage.modifier`). pi: `keyword`/`name` → `fg("syntaxKeyword")`.
    pub keyword: Option<Color>,
    /// Function names (`entity.name.function`, `support.function`). pi: `function`/`title` → `fg("syntaxFunction")`.
    pub function: Option<Color>,
    /// Variables and parameters (`variable.*`). pi: `attr`/`variable`/`params` → `fg("syntaxVariable")`.
    pub variable: Option<Color>,
    /// String and regexp literals (`string.*`). pi: `string`/`regexp` → `fg("syntaxString")`.
    pub string: Option<Color>,
    /// Numeric and constant literals (`constant.numeric`, `constant.language`). pi: `number`/`literal` → `fg("syntaxNumber")`.
    pub number: Option<Color>,
    /// Type names and built-in objects (`entity.name.type`, `support.type`, `storage.type`).
    /// pi: `built_in`/`class`/`type` → `fg("syntaxType")`.
    pub type_name: Option<Color>,
    /// Operators (`keyword.operator`, `punctuation.accessor`). pi: `operator` → `fg("syntaxOperator")`.
    pub operator: Option<Color>,
    /// Structural punctuation and markup tags (`punctuation.*`). pi: `tag`/`punctuation` → `fg("syntaxPunctuation")`.
    pub punctuation: Option<Color>,
    /// Secondary/meta text (`meta.preprocessor`, annotations). pi: `meta` → `fg("muted")` (a non-`syntax*` token).
    pub muted: Option<Color>,
    /// Fallback foreground for text that matches no rule — takes the theme's **`text`** color and is written to
    /// syntect's `ThemeSettings.foreground`. pi's `subst` class is `fg("text")` too, so the two naturally merge
    /// (see the "dissolution-equivalence check" table in `tasks/T-AP-10-1-fix-implemented.md`).
    /// Note: this slot does **not** cover "unrecognized language" — that path returns plain segments and the
    /// renderer fills in `mdCodeBlock` via `markdown.rs::code_line_style`, exactly like pi's `!validLang` branch.
    pub base: Option<Color>,
}

impl Default for SyntaxPalette {
    /// Palette used when no theme is available yet (`MarkdownTheme::default()`, i.e. print mode / first frame fallback) —
    /// base16-ocean.dark's measured colors, so code blocks stay sensibly colored before a theme is wired in.
    /// Honest caveat: this is an **approximation**, not a per-scope reproduction. The previous implementation read
    /// colors from syntect's `ThemeSet` at runtime and could give different scopes different shades; base16-ocean itself
    /// paints `type`, `operator` and `punctuation` with the same default foreground, so 11 slots can only carry
    /// representative colors. Values measured with `Highlighter::style_for_stack` (syntect 5.3):
    /// base #c0c5ce, comment #65737e, keyword #b48ead, function #8fa1b3, variable #bf616a, string #a3be8c,
    /// number #d08770, type_name #c0c5ce, operator #c0c5ce, punctuation #c0c5ce, muted #65737e (base03, secondary gray).
    fn default() -> Self {
        fn rgb(r: u8, g: u8, b: u8) -> Option<Color> {
            Some(Color::Rgb { r, g, b })
        }
        SyntaxPalette {
            comment: rgb(0x65, 0x73, 0x7e),
            keyword: rgb(0xb4, 0x8e, 0xad),
            function: rgb(0x8f, 0xa1, 0xb3),
            variable: rgb(0xbf, 0x61, 0x6a),
            string: rgb(0xa3, 0xbe, 0x8c),
            number: rgb(0xd0, 0x87, 0x70),
            type_name: rgb(0xc0, 0xc5, 0xce),
            operator: rgb(0xc0, 0xc5, 0xce),
            punctuation: rgb(0xc0, 0xc5, 0xce),
            muted: rgb(0x65, 0x73, 0x7e),
            base: rgb(0xc0, 0xc5, 0xce),
        }
    }
}

/// Which palette slot a scope rule paints — lets [`SCOPE_RULES`] stay a plain data table
/// (pi's equivalent is a table of closures `class → (s) => fg(token, s)`; data + enum is the Rust idiomatic form).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SyntaxSlot {
    Comment,
    Keyword,
    Function,
    Variable,
    String,
    Number,
    Type,
    Operator,
    Punctuation,
    Muted,
}

/// Scope-selector → slot table, mirroring pi's highlight.js class mapping (`buildCliHighlightTheme`,
/// `theme.js:711-739`) in syntect's scope vocabulary.
/// Ordering is not load-bearing: syntect arbitrates by selector specificity (`MatchPower`), so
/// `keyword.operator` beats `keyword` and `punctuation.definition.comment` beats `punctuation` regardless of order.
/// Every selector here must parse (locked by `test_scope_rules_all_parse`); entries whose slot is `None` in the
/// palette are simply skipped when the theme is built.
const SCOPE_RULES: &[(&str, SyntaxSlot)] = &[
    ("comment", SyntaxSlot::Comment),
    ("punctuation.definition.comment", SyntaxSlot::Comment),
    ("keyword", SyntaxSlot::Keyword),
    ("storage.modifier", SyntaxSlot::Keyword),
    // Declaration keywords: Sublime grammars express `fn`/`def`/`struct`/`class`/`impl` as
    // `storage.type.*` variants; hljs calls them `keyword`, so they map here.
    ("storage.type.function", SyntaxSlot::Keyword),
    ("storage.type.struct", SyntaxSlot::Keyword),
    ("storage.type.class", SyntaxSlot::Keyword),
    ("storage.type.enum", SyntaxSlot::Keyword),
    ("storage.type.trait", SyntaxSlot::Keyword),
    ("storage.type.impl", SyntaxSlot::Keyword),
    ("storage.type.interface", SyntaxSlot::Keyword),
    // Bare `storage.type`: storage-class keywords in most grammars (`let`, `const`). Known deviation:
    // Rust/C grammar also scopes primitive type names (`i32`) to the same node, so they follow the
    // keyword color; pi (hljs `built_in`) would paint them with `syntaxType` instead. We prefer the
    // keyword reading because declaration keywords vastly outnumber primitive type names in real code.
    ("storage.type", SyntaxSlot::Keyword),
    ("keyword.operator", SyntaxSlot::Operator),
    ("punctuation.accessor", SyntaxSlot::Operator),
    ("entity.name.type", SyntaxSlot::Type),
    ("entity.name.class", SyntaxSlot::Type),
    ("entity.name.struct", SyntaxSlot::Type),
    ("entity.name.enum", SyntaxSlot::Type),
    ("entity.name.trait", SyntaxSlot::Type),
    ("entity.name.interface", SyntaxSlot::Type),
    ("support.type", SyntaxSlot::Type),
    ("support.class", SyntaxSlot::Type),
    ("constant.numeric", SyntaxSlot::Number),
    ("constant.language", SyntaxSlot::Number),
    ("constant.character.escape", SyntaxSlot::Number),
    ("string", SyntaxSlot::String),
    // hljs folds the quotes into the string span, so they take the string color as well.
    ("punctuation.definition.string", SyntaxSlot::String),
    ("constant.other.symbol", SyntaxSlot::String),
    ("entity.name.function", SyntaxSlot::Function),
    ("support.function", SyntaxSlot::Function),
    ("variable.function", SyntaxSlot::Function),
    ("meta.function-call", SyntaxSlot::Function),
    ("entity.name.tag", SyntaxSlot::Keyword),
    ("punctuation", SyntaxSlot::Punctuation),
    ("punctuation.separator", SyntaxSlot::Punctuation),
    ("punctuation.terminator", SyntaxSlot::Punctuation),
    ("punctuation.definition.tag", SyntaxSlot::Punctuation),
    ("variable", SyntaxSlot::Variable),
    ("variable.parameter", SyntaxSlot::Variable),
    ("variable.other", SyntaxSlot::Variable),
    ("meta.object-literal.key", SyntaxSlot::Variable),
    ("meta.preprocessor", SyntaxSlot::Muted),
    ("meta.annotation", SyntaxSlot::Muted),
    ("storage.type.annotation", SyntaxSlot::Muted),
];

/// ANSI 256-color index → sRGB, using pi's own palette tables (`pi-tui/dist/colors.js::indexedToRgb`),
/// so a theme that writes a bare index (`39`) still yields a concrete color for syntect, which only knows RGB.
fn indexed_to_rgb(index: u8) -> (u8, u8, u8) {
    const BASIC: [(u8, u8, u8); 16] = [
        (0, 0, 0),
        (128, 0, 0),
        (0, 128, 0),
        (128, 128, 0),
        (0, 0, 128),
        (128, 0, 128),
        (0, 128, 128),
        (192, 192, 192),
        (128, 128, 128),
        (255, 0, 0),
        (0, 255, 0),
        (255, 255, 0),
        (0, 0, 255),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
    ];
    const CUBE: [u8; 6] = [0, 95, 135, 175, 215, 255];
    match index {
        0..=15 => BASIC[index as usize],
        16..=231 => {
            let i = index - 16;
            (
                CUBE[(i / 36) as usize],
                CUBE[((i % 36) / 6) as usize],
                CUBE[(i % 6) as usize],
            )
        }
        _ => {
            let gray = 8 + (index as u16 - 232) * 10;
            let gray = gray as u8;
            (gray, gray, gray)
        }
    }
}

/// Converts a resolved theme color into syntect's RGB color, or `None` when the theme asked for the terminal
/// default (`Color::Reset`, i.e. `""` in JSON) — such slots emit no rule at all, leaving the base color in charge.
fn to_syntect_color(color: Color) -> Option<syntect::highlighting::Color> {
    let (r, g, b) = match color {
        Color::Reset => return None,
        Color::Rgb { r, g, b } => (r, g, b),
        Color::AnsiValue(index) => indexed_to_rgb(index),
        // Named 16-color variants crossterm exposes (Color::Red, Color::DarkGrey, ...) are not produced by the
        // theme loader (it only yields Rgb/AnsiValue/Reset); map defensively through their ANSI index.
        other => {
            let index = match other {
                Color::Black => 0,
                Color::DarkRed => 1,
                Color::DarkGreen => 2,
                Color::DarkYellow => 3,
                Color::DarkBlue => 4,
                Color::DarkMagenta => 5,
                Color::DarkCyan => 6,
                Color::Grey => 7,
                Color::DarkGrey => 8,
                Color::Red => 9,
                Color::Green => 10,
                Color::Yellow => 11,
                Color::Blue => 12,
                Color::Magenta => 13,
                Color::Cyan => 14,
                Color::White => 15,
                _ => return None,
            };
            indexed_to_rgb(index)
        }
    };
    Some(syntect::highlighting::Color { r, g, b, a: 255 })
}

/// How glyphs that match no rule look — the fallback foreground written into `ThemeSettings.foreground`.
fn slot_color(palette: &SyntaxPalette, slot: SyntaxSlot) -> Option<syntect::highlighting::Color> {
    let color = match slot {
        SyntaxSlot::Comment => palette.comment,
        SyntaxSlot::Keyword => palette.keyword,
        SyntaxSlot::Function => palette.function,
        SyntaxSlot::Variable => palette.variable,
        SyntaxSlot::String => palette.string,
        SyntaxSlot::Number => palette.number,
        SyntaxSlot::Type => palette.type_name,
        SyntaxSlot::Operator => palette.operator,
        SyntaxSlot::Punctuation => palette.punctuation,
        SyntaxSlot::Muted => palette.muted,
    };
    color.and_then(to_syntect_color)
}

/// Language resolution chain (shared by highlight's explicit-language branch and supports_language):
/// Some(lang) → first normalized via LANG_ALIASES (ts/typescript/tsx/jsx → js) → `find_syntax_by_token`
/// (case-insensitive; syntect matches by extension → syntax name internally) → `find_syntax_by_extension`;
/// None → `find_syntax_by_first_line` (shebang/modeline detection).
/// Returns a resident syntax reference; None = unrecognized → caller falls back to plain text (default Style), no error.
fn resolve_syntax(language: Option<&str>, code: &str) -> Option<&'static SyntaxReference> {
    let ss = syntax_set();
    match language {
        Some(lang) => {
            let normalized = LANG_ALIASES
                .iter()
                .find(|(alias, _)| alias.eq_ignore_ascii_case(lang))
                .map(|(_, target)| *target)
                .unwrap_or(lang);
            ss.find_syntax_by_token(normalized)
                .or_else(|| ss.find_syntax_by_extension(normalized))
        }
        None => ss.find_syntax_by_first_line(code),
    }
}

/// Builds a throwaway syntect theme from the palette — the bridge that lets theme tokens actually paint code blocks.
/// `palette.base` becomes the theme's default foreground (glyphs matching no rule), and every [`SCOPE_RULES`] entry whose
/// slot resolves to a color becomes one `ThemeItem` with that foreground and an empty font style (pi's syntax coloring
/// never adds bold/italic, so the synthesized theme does not either).
/// Built per code block instead of cached: a theme is only a few dozen `ThemeItem`s, and caching would have to be keyed
/// by palette (i.e. invalidated on every theme switch) for negligible gain.
/// Selector parse failures are skipped rather than panicking (project rule: no `unwrap`/`expect` outside `main`/tests);
/// the table is a compile-time constant and `test_scope_rules_all_parse` locks that every selector parses.
fn build_syntect_theme(palette: &SyntaxPalette) -> Theme {
    let scopes: Vec<ThemeItem> = SCOPE_RULES
        .iter()
        .filter_map(|(selector, slot)| {
            let foreground = slot_color(palette, *slot)?;
            let scope: ScopeSelectors = selector.parse().ok()?;
            Some(ThemeItem {
                scope,
                style: StyleModifier {
                    foreground: Some(foreground),
                    background: None,
                    font_style: Some(FontStyle::empty()),
                },
            })
        })
        .collect();
    Theme {
        name: Some("zen-syntax".to_string()),
        author: None,
        settings: ThemeSettings {
            foreground: palette.base.and_then(to_syntect_color),
            background: None,
            ..ThemeSettings::default()
        },
        scopes,
    }
}

/// When highlight execution receives a (Style, &str) span from syntect, it translates its display semantics into a
/// screen::Style writable by the screen buffer — "one code garment per span": fg = Color{r,g,b} → crossterm `Color::Rgb`;
/// attributes = FontStyle bit folding → `Attribute::Bold`/`Italic`/`Underlined` (none → empty Vec);
/// bg is always None — design decision per "ecosystem reuse verification conclusions" 5.1-6 (code block backgrounds are laid uniformly by the display component,
/// not colored per token, to avoid conflicts with the rendering layer's block background).
/// `has_base` carries "the theme defined a fallback foreground": when it did not (theme `text` = `""`), syntect falls back to
/// its own default `Color::BLACK` (`highlighting/highlighter.rs:306`, `highlighting/style.rs:443`) and we map that back to
/// `fg: None`, so the renderer keeps the terminal default instead of painting hard black on a dark theme.
fn syntect_style_to_screen(style: &syntect::highlighting::Style, has_base: bool) -> Style {
    let fg = if !has_base && style.foreground == syntect::highlighting::Color::BLACK {
        None
    } else {
        Some(Color::Rgb {
            r: style.foreground.r,
            g: style.foreground.g,
            b: style.foreground.b,
        })
    };
    let mut attributes = Vec::new();
    if style.font_style.contains(FontStyle::BOLD) {
        attributes.push(Attribute::Bold);
    }
    if style.font_style.contains(FontStyle::ITALIC) {
        attributes.push(Attribute::Italic);
    }
    if style.font_style.contains(FontStyle::UNDERLINE) {
        attributes.push(Attribute::Underlined);
    }
    Style {
        fg,
        bg: None,
        attributes,
    }
}

/// The markdown renderer (T-TU-02-1) passes the raw source code and language tag of a code block fence to
/// this function when parsing (the tag from pulldown-cmark's `CodeBlockKind::Fenced`), yielding line-organized colored spans —
/// "plain text walks in, colored lines walk out": each line consists of several (style, text) spans,
/// the rendering layer lays out and wraps them directly and writes them to the buffer reusing screen::Style, with no need to understand syntax coloring internals.
/// Language resolution chain: Some(lang) → `find_syntax_by_token` (case-insensitive, matched by extension/syntax name) →
/// `find_syntax_by_extension`; None → `find_syntax_by_first_line` (shebang); unrecognized always yields
/// default style (no coloring) segments, no error.
/// Theme: `palette` carries the current theme's nine `syntax*` tokens (via `MarkdownTheme.syntax_palette`, built by
/// `Theme::syntax_palette`) and is turned into a syntect theme on the spot — so code blocks follow the zen theme
/// (dark / light / user themes) instead of a fixed built-in palette (T-AP-10-1-fix closes the T-AP-10-1 integration gap).
/// Boundary: split lines by `\n` (per `str::lines` semantics); output line count is 1:1 with input (empty lines yield an empty Vec,
/// ensuring line-by-line alignment in the rendering layer); empty input returns an empty Vec; CRLF/CR is normalized first by the caller (REQ-TUI-02 markdown
/// already commits to normalizing line endings, this function does not handle `\r`); TABs are kept as-is, width expansion is the rendering layer's responsibility.
pub fn highlight(
    code: &str,
    language: Option<&str>,
    palette: &SyntaxPalette,
) -> Vec<Vec<StyledSegment>> {
    let Some(syntax) = resolve_syntax(language, code) else {
        // Unrecognized language / no shebang: plain-text passthrough (one default-style segment per line), does not enter syntect.
        // The renderer then fills in `mdCodeBlock` via `code_line_style` — the same net effect as pi's `!validLang` branch.
        return code
            .lines()
            .map(|line| {
                vec![StyledSegment {
                    style: Style::default(),
                    text: line.to_string(),
                }]
            })
            .collect();
    };

    let theme = build_syntect_theme(palette);
    let has_base = theme.settings.foreground.is_some();
    let mut hl = HighlightLines::new(syntax, &theme);
    // Each line is handed to syntect WITH its terminator: the grammars come from
    // `SyntaxSet::load_defaults_newlines`, where line-scoped contexts are popped by `\n`. Without it,
    // e.g. JavaScript `// comment` never closes and every following line renders as a comment
    // (bug present since T-AP-10-1; fixed here because it makes code blocks visibly wrong).
    code.split_inclusive('\n')
        .map(|line| {
            hl.highlight_line(line, syntax_set())
                .map(|segs| {
                    let mut segs: Vec<StyledSegment> = segs
                        .into_iter()
                        .map(|(s, text)| StyledSegment {
                            style: syntect_style_to_screen(&s, has_base),
                            text: text.to_string(),
                        })
                        .collect();
                    strip_line_terminator(&mut segs);
                    segs
                })
                // syntect only returns fmt::Error when line parameters are insufficient (normally unreachable); degrade to the whole line as plain text,
                // preserving the "1:1 line count" invariant, not letting an error swallow a line
                .unwrap_or_else(|_| {
                    let mut segs = vec![StyledSegment {
                        style: Style::default(),
                        text: line.to_string(),
                    }];
                    strip_line_terminator(&mut segs);
                    segs
                })
        })
        .collect()
}

/// Drops the line terminator syntect echoed back (see the `split_inclusive` note in [`highlight`]), keeping the
/// renderer's contract that a segment never contains a newline; a segment made of nothing but the terminator is removed,
/// which is what turns a blank source line into an empty segment list.
fn strip_line_terminator(segs: &mut Vec<StyledSegment>) {
    if let Some(last) = segs.last_mut() {
        if last.text.ends_with('\n') {
            last.text.pop();
            if last.text.ends_with('\r') {
                last.text.pop();
            }
            if last.text.is_empty() {
                segs.pop();
            }
        }
    }
}

/// Assembly/UI layer (e.g. T-AP-03-2 slash-command language selection, T-AP-11-3 theme preview) needs to answer
/// "can this language label actually be highlighted" when calling this function — list only highlightable languages, to avoid promising
/// a non-existent highlight effect for unknown labels. The resolution chain matches the explicit-language branch of `highlight`
/// (token → extension); a plain fallback never counts as "supported" (aligned with pi `hljs.getLanguage` semantics:
/// unknown languages return undefined / false).
pub fn supports_language(name: &str) -> bool {
    resolve_syntax(Some(name), "").is_some()
}

#[cfg(test)]
mod tests {
    /// CJK sample kept intentionally: this test verifies wide-character / multibyte behavior.
    const CJK_SAMPLE: &str = "/* Comment starts";
    const CJK_SAMPLE_2: &str = "Still in comment */";
    use super::*;

    /// Aggregate all styles across all lines, taking the judgment of whether "an explicit foreground color exists" (plain degradation always yields fg=None).
    fn any_fg_set(row: &[StyledSegment]) -> bool {
        row.iter().any(|s| s.style.fg.is_some())
    }

    /// Concatenate the text of all segments in a line, verifying no text is lost.
    fn row_text(row: &[StyledSegment]) -> String {
        row.iter().map(|s| s.text.as_str()).collect()
    }

    // ── Language resolution chain ──

    #[test]
    fn test_highlight_known_language_colors_keyword() {
        let rows = highlight("fn main() {}", Some("rust"), &SyntaxPalette::default());
        assert_eq!(rows.len(), 1);
        // Keyword is colored: at least one segment has an explicit foreground color (plain degradation yields all fg=None)
        assert!(
            any_fg_set(&rows[0]),
            "Rust highlighting should produce styled spans"
        );
        assert_eq!(
            row_text(&rows[0]),
            "fn main() {}",
            "Concatenating spans should reconstruct the original text"
        );
    }

    #[test]
    fn test_highlight_unknown_language_plain() {
        let rows = highlight(
            "let x = 1;",
            Some("no-such-language-xyz"),
            &SyntaxPalette::default(),
        );
        assert_eq!(rows.len(), 1);
        // Unrecognized language: plain-text passthrough, all default style (no explicit foreground color)
        assert!(!any_fg_set(&rows[0]));
        assert_eq!(row_text(&rows[0]), "let x = 1;");
    }

    #[test]
    fn test_highlight_no_language_shebang_detects_python() {
        let rows = highlight(
            "#!/usr/bin/env python\nprint(1)",
            None,
            &SyntaxPalette::default(),
        );
        assert_eq!(rows.len(), 2);
        // Shebang triggers syntax recognition → second line (the print call) should be colored
        assert!(
            any_fg_set(&rows[1]),
            "shebang should trigger Python syntax detection"
        );
        assert_eq!(row_text(&rows[1]), "print(1)");
    }

    #[test]
    fn test_highlight_no_language_without_shebang_plain() {
        let rows = highlight("just some text", None, &SyntaxPalette::default());
        assert_eq!(rows.len(), 1);
        assert!(
            !any_fg_set(&rows[0]),
            "Without a shebang, it should be treated as plain text"
        );
    }

    // ── Line-count invariant and boundaries ──

    #[test]
    fn test_highlight_empty_input() {
        assert!(highlight("", Some("rust"), &SyntaxPalette::default()).is_empty());
        assert!(highlight("", None, &SyntaxPalette::default()).is_empty());
    }

    #[test]
    fn test_highlight_blank_line_preserved() {
        let rows = highlight("a\n\nb", Some("rust"), &SyntaxPalette::default());
        assert_eq!(rows.len(), 3, "Line count should be 1:1 with input");
        assert!(rows[1].is_empty(), "Blank lines produce a blank-line Vec");
        assert_eq!(row_text(&rows[0]), "a");
        assert_eq!(row_text(&rows[2]), "b");
    }

    #[test]
    fn test_highlight_line_integrity_multiline() {
        let code = "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n";
        let rows = highlight(code, Some("rust"), &SyntaxPalette::default());
        let joined: String = rows
            .iter()
            .map(|r| row_text(r))
            .collect::<Vec<_>>()
            .join("\n");
        // A trailing newline yields no extra empty line under lines() semantics
        assert_eq!(joined, code.trim_end());
        assert_eq!(rows.len(), code.lines().count());
    }

    // ── Cross-line statefulness (HighlightLines state continuation: comments/strings spanning lines) ──

    #[test]
    fn test_highlight_stateful_across_lines() {
        let ss = syntax_set();
        let syntax = ss
            .find_syntax_by_token("rust")
            .expect("Built-in Rust syntax");
        let theme = build_syntect_theme(&SyntaxPalette::default());

        // Context: first line opens a block comment → second line is still inside the comment scope
        let mut hl_ctx = HighlightLines::new(syntax, &theme);
        let l1 = hl_ctx.highlight_line(CJK_SAMPLE, ss).expect("fmt ok");
        let l2 = hl_ctx.highlight_line(CJK_SAMPLE_2, ss).expect("fmt ok");

        // Same line text without context: should be parsed as ordinary code (not comment scope)
        let mut hl_fresh = HighlightLines::new(syntax, &theme);
        let l2_fresh = hl_fresh.highlight_line(CJK_SAMPLE_2, ss).expect("fmt ok");

        // In cross-line context the second line falls in comment scope (style differs from the style of independently parsed code) → proves state continuation
        assert_ne!(
            l2.first().expect("Segment is non-empty").0.foreground,
            l2_fresh.first().expect("Segment is non-empty").0.foreground,
            "Multi-line comments should carry comment coloring across lines (unlike code style parsed independently)"
        );
        // The first line itself is already in comment scope (same color as the second line)
        assert_eq!(
            l1.first().expect("Segment is non-empty").0.foreground,
            l2.first().expect("Segment is non-empty").0.foreground
        );
    }

    // ── syntect_style_to_screen mapping ──

    #[test]
    fn test_style_mapping_fg_and_attributes() {
        let st = syntect::highlighting::Style {
            foreground: syntect::highlighting::Color {
                r: 10,
                g: 20,
                b: 30,
                a: 255,
            },
            background: syntect::highlighting::Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            },
            font_style: FontStyle::BOLD | FontStyle::ITALIC,
        };
        let mapped = syntect_style_to_screen(&st, true);
        assert_eq!(
            mapped.fg,
            Some(Color::Rgb {
                r: 10,
                g: 20,
                b: 30
            }),
            "Foreground color maps to RGB"
        );
        assert_eq!(mapped.bg, None, "bg is always None (design decision)");
        assert_eq!(mapped.attributes.len(), 2);
        assert!(mapped.attributes.contains(&Attribute::Bold));
        assert!(mapped.attributes.contains(&Attribute::Italic));
    }

    // ── supports_language ──

    #[test]
    fn test_supports_language_known() {
        // Both the grammar name (case-insensitive) and the extension (including aliases) match
        for lang in [
            "rust", "Rust", "ts", "py", "json", "yaml", "markdown", "go", "js",
        ] {
            assert!(
                supports_language(lang),
                "Language {lang} should be supported"
            );
        }
    }

    #[test]
    fn test_supports_language_unknown() {
        for lang in ["no-such-lang-123", ""] {
            assert!(
                !supports_language(lang),
                "Unknown language {lang:?} should return false"
            );
        }
    }

    // ── Common language set (task requirement list) ──

    #[test]
    fn test_required_language_set_highlightable() {
        // REQ-APP-10 requires: Rust/TS/Python/Go/JSON/YAML/Markdown (see REQUIRED_LANGUAGES for the list)
        let snippets: std::collections::HashMap<&str, &str> = [
            ("rust", "fn main() {}"),
            ("typescript", "const x: number = 1;"),
            ("python", "def f(x):\n    return x"),
            ("go", "package main"),
            ("json", "{\"a\": 1}"),
            ("yaml", "key: value"),
            ("markdown", "# Title"),
        ]
        .into_iter()
        .collect();

        for lang in REQUIRED_LANGUAGES {
            assert!(
                supports_language(lang),
                "{lang} should be in the supported list"
            );
            let rows = highlight(snippets[lang], Some(lang), &SyntaxPalette::default());
            assert!(
                !rows.is_empty(),
                "Highlighting {lang} should produce at least one line"
            );
        }
    }

    // ── palette-driven theming ──

    /// Two palettes that differ only in their `keyword` slot: the synthesized theme must carry each through to the output.
    fn palette_with_keyword(color: Color) -> SyntaxPalette {
        SyntaxPalette {
            keyword: Some(color),
            ..SyntaxPalette::default()
        }
    }

    /// First foreground used by a span whose text equals `needle`.
    fn fg_of(row: &[StyledSegment], needle: &str) -> Option<Color> {
        row.iter()
            .find(|s| s.text == needle)
            .and_then(|s| s.style.fg)
    }

    #[test]
    fn test_highlight_theme_switches_colors() {
        // Same code, two palettes differing only in the keyword color → token text identical, at least one span recolored.
        // (Before T-AP-10-1-fix this compared two syntect built-ins; now the palette IS the theming input.)
        let red = highlight(
            "fn main() {}",
            Some("rust"),
            &palette_with_keyword(Color::Rgb { r: 255, g: 0, b: 0 }),
        );
        let blue = highlight(
            "fn main() {}",
            Some("rust"),
            &palette_with_keyword(Color::Rgb { r: 0, g: 0, b: 255 }),
        );
        assert!(any_fg_set(&red[0]));
        assert!(any_fg_set(&blue[0]));
        assert_eq!(row_text(&red[0]), row_text(&blue[0]));
        assert_eq!(
            fg_of(&red[0], "fn"),
            Some(Color::Rgb { r: 255, g: 0, b: 0 })
        );
        assert_eq!(
            fg_of(&blue[0], "fn"),
            Some(Color::Rgb { r: 0, g: 0, b: 255 })
        );
    }

    #[test]
    fn test_scope_rules_all_parse() {
        // The table is a compile-time constant parsed at build time; a typo would silently drop a coloring rule,
        // so lock that every selector parses (build_syntect_theme skips parse failures instead of panicking).
        for (selector, _) in SCOPE_RULES {
            assert!(
                selector.parse::<ScopeSelectors>().is_ok(),
                "scope selector {selector:?} must parse"
            );
        }
    }

    #[test]
    fn test_syntax_palette_default_values() {
        // base16-ocean.dark's measured colors — the print/first-frame fallback palette.
        let p = SyntaxPalette::default();
        assert_eq!(
            p.comment,
            Some(Color::Rgb {
                r: 0x65,
                g: 0x73,
                b: 0x7e
            })
        );
        assert_eq!(
            p.keyword,
            Some(Color::Rgb {
                r: 0xb4,
                g: 0x8e,
                b: 0xad
            })
        );
        assert_eq!(
            p.function,
            Some(Color::Rgb {
                r: 0x8f,
                g: 0xa1,
                b: 0xb3
            })
        );
        assert_eq!(
            p.variable,
            Some(Color::Rgb {
                r: 0xbf,
                g: 0x61,
                b: 0x6a
            })
        );
        assert_eq!(
            p.string,
            Some(Color::Rgb {
                r: 0xa3,
                g: 0xbe,
                b: 0x8c
            })
        );
        assert_eq!(
            p.number,
            Some(Color::Rgb {
                r: 0xd0,
                g: 0x87,
                b: 0x70
            })
        );
        assert_eq!(
            p.type_name,
            Some(Color::Rgb {
                r: 0xc0,
                g: 0xc5,
                b: 0xce
            })
        );
        assert_eq!(
            p.operator,
            Some(Color::Rgb {
                r: 0xc0,
                g: 0xc5,
                b: 0xce
            })
        );
        assert_eq!(
            p.punctuation,
            Some(Color::Rgb {
                r: 0xc0,
                g: 0xc5,
                b: 0xce
            })
        );
        assert_eq!(
            p.muted,
            Some(Color::Rgb {
                r: 0x65,
                g: 0x73,
                b: 0x7e
            })
        );
        assert_eq!(
            p.base,
            Some(Color::Rgb {
                r: 0xc0,
                g: 0xc5,
                b: 0xce
            })
        );
    }

    #[test]
    fn test_highlight_maps_tokens_to_palette_slots() {
        // A palette with distinct colors per slot, so the assertion pins down scope→slot mapping, not just
        // "something got colored". JavaScript is used because its grammar scopes plain variables and
        // parameters (`variable.other`/`variable.parameter`), which Rust's grammar does not.
        let palette = SyntaxPalette {
            comment: Some(Color::Rgb { r: 1, g: 1, b: 1 }),
            keyword: Some(Color::Rgb { r: 2, g: 2, b: 2 }),
            function: Some(Color::Rgb { r: 3, g: 3, b: 3 }),
            variable: Some(Color::Rgb { r: 4, g: 4, b: 4 }),
            string: Some(Color::Rgb { r: 5, g: 5, b: 5 }),
            number: Some(Color::Rgb { r: 6, g: 6, b: 6 }),
            type_name: Some(Color::Rgb { r: 7, g: 7, b: 7 }),
            operator: Some(Color::Rgb { r: 8, g: 8, b: 8 }),
            punctuation: Some(Color::Rgb { r: 9, g: 9, b: 9 }),
            muted: Some(Color::Rgb {
                r: 10,
                g: 10,
                b: 10,
            }),
            base: Some(Color::Rgb {
                r: 11,
                g: 11,
                b: 11,
            }),
        };
        let rows = highlight(
            "// c\nfunction g(p) { const y = \"x\"; return p + 2; }",
            Some("js"),
            &palette,
        );
        // comment → comment slot (the `//` punctuation belongs to the comment line scope)
        assert_eq!(fg_of(&rows[0], "//"), Some(Color::Rgb { r: 1, g: 1, b: 1 }));
        assert_eq!(fg_of(&rows[0], " c"), Some(Color::Rgb { r: 1, g: 1, b: 1 }));
        // `function` / `const` → keyword slot
        assert_eq!(
            fg_of(&rows[1], "function"),
            Some(Color::Rgb { r: 2, g: 2, b: 2 })
        );
        assert_eq!(
            fg_of(&rows[1], "const"),
            Some(Color::Rgb { r: 2, g: 2, b: 2 })
        );
        // declared function name → function slot
        assert_eq!(fg_of(&rows[1], "g"), Some(Color::Rgb { r: 3, g: 3, b: 3 }));
        // parameters / variables → variable slot
        assert_eq!(fg_of(&rows[1], "p"), Some(Color::Rgb { r: 4, g: 4, b: 4 }));
        assert_eq!(fg_of(&rows[1], "y"), Some(Color::Rgb { r: 4, g: 4, b: 4 }));
        // string content and its quotes → string slot
        assert_eq!(fg_of(&rows[1], "x"), Some(Color::Rgb { r: 5, g: 5, b: 5 }));
        assert_eq!(fg_of(&rows[1], "\""), Some(Color::Rgb { r: 5, g: 5, b: 5 }));
        // numeric literal → number slot
        assert_eq!(fg_of(&rows[1], "2"), Some(Color::Rgb { r: 6, g: 6, b: 6 }));
        // `+` → operator slot
        assert_eq!(fg_of(&rows[1], "+"), Some(Color::Rgb { r: 8, g: 8, b: 8 }));
        // structural punctuation → punctuation slot
        assert_eq!(fg_of(&rows[1], "{"), Some(Color::Rgb { r: 9, g: 9, b: 9 }));
    }

    #[test]
    fn test_highlight_unmatched_text_uses_base() {
        // Text matching no rule takes `base` (the theme's `text` token), mirroring pi's `subst → fg("text")`.
        // Built by giving the palette ONLY a base color: no rule is emitted, so every glyph is "unmatched"
        // and syntect's default foreground is what paints them.
        let palette = SyntaxPalette {
            comment: None,
            keyword: None,
            function: None,
            variable: None,
            string: None,
            number: None,
            type_name: None,
            operator: None,
            punctuation: None,
            muted: None,
            base: Some(Color::Rgb {
                r: 0x12,
                g: 0x34,
                b: 0x56,
            }),
        };
        let rows = highlight("fn main() {}", Some("rust"), &palette);
        assert_eq!(row_text(&rows[0]), "fn main() {}");
        assert!(
            rows[0].iter().all(|s| s.style.fg
                == Some(Color::Rgb {
                    r: 0x12,
                    g: 0x34,
                    b: 0x56
                })),
            "every unmatched span must fall back to `base`"
        );
    }

    #[test]
    fn test_highlight_without_base_keeps_terminal_default() {
        // Theme `text` = "" (terminal default): no fallback foreground is emitted, and syntect's own BLACK default must be
        // mapped back to `fg: None` so the renderer keeps the terminal default instead of painting hard black.
        let palette = SyntaxPalette {
            base: None,
            keyword: None,
            comment: None,
            function: None,
            variable: None,
            string: None,
            number: None,
            type_name: None,
            operator: None,
            punctuation: None,
            muted: None,
        };
        let rows = highlight("fn main() {}", Some("rust"), &palette);
        assert_eq!(row_text(&rows[0]), "fn main() {}");
        assert!(
            !any_fg_set(&rows[0]),
            "with an empty palette nothing may be painted (no hard black from syntect's default)"
        );
    }

    #[test]
    fn test_style_mapping_black_is_kept_when_base_defined() {
        // `has_base = true` means BLACK is a legitimate theme color, not the sentinel.
        let st = syntect::highlighting::Style {
            foreground: syntect::highlighting::Color::BLACK,
            background: syntect::highlighting::Color::WHITE,
            font_style: FontStyle::empty(),
        };
        assert_eq!(
            syntect_style_to_screen(&st, true).fg,
            Some(Color::Rgb { r: 0, g: 0, b: 0 })
        );
        assert_eq!(syntect_style_to_screen(&st, false).fg, None);
    }

    #[test]
    fn test_indexed_palette_colors_are_converted() {
        // Custom themes may use 256-color indices; syntect only knows RGB, so the palette must be expanded
        // through pi's own palette tables before it reaches the theme.
        assert_eq!(indexed_to_rgb(16), (0, 0, 0));
        assert_eq!(indexed_to_rgb(196), (255, 0, 0));
        assert_eq!(indexed_to_rgb(231), (255, 255, 255));
        assert_eq!(indexed_to_rgb(232), (8, 8, 8));
        assert_eq!(indexed_to_rgb(255), (238, 238, 238));
        assert_eq!(indexed_to_rgb(9), (255, 0, 0));
        let palette = SyntaxPalette {
            keyword: Some(Color::AnsiValue(196)),
            ..SyntaxPalette::default()
        };
        let rows = highlight("fn main() {}", Some("rust"), &palette);
        assert_eq!(
            fg_of(&rows[0], "fn"),
            Some(Color::Rgb { r: 255, g: 0, b: 0 })
        );
    }
}
