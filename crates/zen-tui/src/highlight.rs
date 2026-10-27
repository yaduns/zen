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
use syntect::highlighting::{FontStyle, Theme, ThemeSet};
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

/// Default theme name: a built-in dark theme from the syntect ThemeSet (base16 family), suited to the zen TUI's dark interface.
/// Loaded lazily on first highlight via the THEME_SET singleton; the replacement seam for theme system integration (T-AP-11-3) is described under "Dependencies".
const DEFAULT_THEME: &str = "base16-ocean.dark";

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

/// Theme set singleton: built-in themes batch-loaded lazily (feature `default-themes`),
/// Highlighting obtains the entry by name via `resolve_syntect_theme()` and outputs coloring.
static THEME_SET: std::sync::OnceLock<ThemeSet> = std::sync::OnceLock::new();

/// Fallback theme: used when both DEFAULT_THEME and the built-in theme set are missing (unreachable on the normal path;
/// `ThemeSet::load_defaults()` always includes the base16 family; this is only a defensive fallback).
static FALLBACK_THEME: std::sync::OnceLock<Theme> = std::sync::OnceLock::new();

/// Syntax set accessor: initialized via `SyntaxSet::load_defaults_newlines()` on first call
/// (a heavy one-time load, hosted by a OnceLock lazily), afterwards returns a 'static reference. Consumed by both resolve_syntax and
/// highlight execution — "load once, live in memory, grab it at hand".
fn syntax_set() -> &'static SyntaxSet {
    SYNTAX_SET.get_or_init(SyntaxSet::load_defaults_newlines)
}

/// Theme set accessor: initialized via `ThemeSet::load_defaults()` on first call, afterwards returns a 'static reference,
/// Used by resolve_syntect_theme() to get a theme entry.
fn theme_set() -> &'static ThemeSet {
    THEME_SET.get_or_init(ThemeSet::load_defaults)
}

/// Resolves a syntect built-in theme by name: `highlight` fetches the theme before coloring each code block, letting the 'zen theme name
/// (dark → base16-ocean.dark, provided by `Theme::syntax_theme_name`)' map to deterministic coloring;
/// On miss, falls back to `DEFAULT_THEME`, then to the first built-in theme, and then to a static fallback theme—a theme is always available.
/// Returns a `'static` reference (`ThemeSet` is already a OnceLock singleton, so no extra cache is needed).
/// Module-private—consumed only by `highlight` and its tests.
fn resolve_syntect_theme(name: &str) -> &'static Theme {
    let ts = theme_set();
    ts.themes
        .get(name)
        .or_else(|| ts.themes.get(DEFAULT_THEME))
        .or_else(|| ts.themes.values().next())
        .unwrap_or_else(|| FALLBACK_THEME.get_or_init(Theme::default))
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

/// When highlight execution receives a (Style, &str) span from syntect, it translates its display semantics into a
/// screen::Style writable by the screen buffer — "one code garment per span": fg = Color{r,g,b} → crossterm `Color::Rgb`;
/// attributes = FontStyle bit folding → `Attribute::Bold`/`Italic`/`Underlined` (none → empty Vec);
/// bg is always None — design decision per "ecosystem reuse verification conclusions" 5.1-6 (code block backgrounds are laid uniformly by the display component,
/// not colored per token, to avoid conflicts with the rendering layer's block background).
fn syntect_style_to_screen(style: &syntect::highlighting::Style) -> Style {
    let fg = Some(Color::Rgb {
        r: style.foreground.r,
        g: style.foreground.g,
        b: style.foreground.b,
    });
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
/// Theme: `theme_name` specifies the syntect built-in theme name (`MarkdownTheme.syntax_theme_name`, supplied by
/// `Theme::syntax_theme_name`), and `resolve_syntect_theme` falls back to `DEFAULT_THEME` on miss—
/// Code blocks switch with the zen theme (dark/light) instead of always using dark (T-TU-02-1-fix closes the T-AP-10-1 integration gap).
/// Boundary: split lines by `\n` (per `str::lines` semantics); output line count is 1:1 with input (empty lines yield an empty Vec,
/// ensuring line-by-line alignment in the rendering layer); empty input returns an empty Vec; CRLF/CR is normalized first by the caller (REQ-TUI-02 markdown
/// already commits to normalizing line endings, this function does not handle `\r`); TABs are kept as-is, width expansion is the rendering layer's responsibility.
pub fn highlight(code: &str, language: Option<&str>, theme_name: &str) -> Vec<Vec<StyledSegment>> {
    let Some(syntax) = resolve_syntax(language, code) else {
        // Unrecognized language / no shebang: plain-text passthrough (one default-style segment per line), does not enter syntect
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

    let mut hl = HighlightLines::new(syntax, resolve_syntect_theme(theme_name));
    code.lines()
        .map(|line| {
            hl.highlight_line(line, syntax_set())
                .map(|segs| {
                    segs.into_iter()
                        .map(|(s, text)| StyledSegment {
                            style: syntect_style_to_screen(&s),
                            text: text.to_string(),
                        })
                        .collect()
                })
                // syntect only returns fmt::Error when line parameters are insufficient (normally unreachable); degrade to the whole line as plain text,
                // preserving the "1:1 line count" invariant, not letting an error swallow a line
                .unwrap_or_else(|_| {
                    vec![StyledSegment {
                        style: Style::default(),
                        text: line.to_string(),
                    }]
                })
        })
        .collect()
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
        let rows = highlight("fn main() {}", Some("rust"), DEFAULT_THEME);
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
        let rows = highlight("let x = 1;", Some("no-such-language-xyz"), DEFAULT_THEME);
        assert_eq!(rows.len(), 1);
        // Unrecognized language: plain-text passthrough, all default style (no explicit foreground color)
        assert!(!any_fg_set(&rows[0]));
        assert_eq!(row_text(&rows[0]), "let x = 1;");
    }

    #[test]
    fn test_highlight_no_language_shebang_detects_python() {
        let rows = highlight("#!/usr/bin/env python\nprint(1)", None, DEFAULT_THEME);
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
        let rows = highlight("just some text", None, DEFAULT_THEME);
        assert_eq!(rows.len(), 1);
        assert!(
            !any_fg_set(&rows[0]),
            "Without a shebang, it should be treated as plain text"
        );
    }

    // ── Line-count invariant and boundaries ──

    #[test]
    fn test_highlight_empty_input() {
        assert!(highlight("", Some("rust"), DEFAULT_THEME).is_empty());
        assert!(highlight("", None, DEFAULT_THEME).is_empty());
    }

    #[test]
    fn test_highlight_blank_line_preserved() {
        let rows = highlight("a\n\nb", Some("rust"), DEFAULT_THEME);
        assert_eq!(rows.len(), 3, "Line count should be 1:1 with input");
        assert!(rows[1].is_empty(), "Blank lines produce a blank-line Vec");
        assert_eq!(row_text(&rows[0]), "a");
        assert_eq!(row_text(&rows[2]), "b");
    }

    #[test]
    fn test_highlight_line_integrity_multiline() {
        let code = "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n";
        let rows = highlight(code, Some("rust"), DEFAULT_THEME);
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
        let theme = resolve_syntect_theme(DEFAULT_THEME);

        // Context: first line opens a block comment → second line is still inside the comment scope
        let mut hl_ctx = HighlightLines::new(syntax, theme);
        let l1 = hl_ctx.highlight_line(CJK_SAMPLE, ss).expect("fmt ok");
        let l2 = hl_ctx.highlight_line(CJK_SAMPLE_2, ss).expect("fmt ok");

        // Same line text without context: should be parsed as ordinary code (not comment scope)
        let mut hl_fresh = HighlightLines::new(syntax, theme);
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
        let mapped = syntect_style_to_screen(&st);
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
            let rows = highlight(snippets[lang], Some(lang), DEFAULT_THEME);
            assert!(
                !rows.is_empty(),
                "Highlighting {lang} should produce at least one line"
            );
        }
    }

    // ── theme name resolution ──

    #[test]
    fn test_resolve_syntect_theme_known_and_fallback() {
        // A known theme name resolves to the same entry by name (pointer comparison); an unknown name falls back to DEFAULT_THEME.
        // syntect Theme.name is a display name (not a lookup key), so assert pointer identity.
        let ts = theme_set();
        let dark = resolve_syntect_theme("base16-ocean.dark");
        assert!(std::ptr::eq(
            dark,
            ts.themes.get("base16-ocean.dark").unwrap()
        ));
        let light = resolve_syntect_theme("base16-ocean.light");
        assert!(std::ptr::eq(
            light,
            ts.themes.get("base16-ocean.light").unwrap()
        ));
        let fallback = resolve_syntect_theme("no-such-theme");
        assert!(std::ptr::eq(
            fallback,
            ts.themes.get(DEFAULT_THEME).unwrap()
        ));
    }

    #[test]
    fn test_highlight_theme_switches_colors() {
        // Same code under dark/light themes: text lines are identical, but at least one token segment has a different foreground color
        // (base16 dark/light only swap base00-07 lightness; keywords and similar have the same color; punctuation/whitespace segments differ by theme).
        let dark = highlight("fn main() {}", Some("rust"), "base16-ocean.dark");
        let light = highlight("fn main() {}", Some("rust"), "base16-ocean.light");
        assert!(any_fg_set(&dark[0]));
        assert!(any_fg_set(&light[0]));
        assert_eq!(row_text(&dark[0]), row_text(&light[0]));
        let d_fgs: Vec<_> = dark[0].iter().map(|s| s.style.fg).collect();
        let l_fgs: Vec<_> = light[0].iter().map(|s| s.style.fg).collect();
        assert_ne!(
            d_fgs, l_fgs,
            "At least one token span should be colored differently between dark/light themes"
        );
    }
}
