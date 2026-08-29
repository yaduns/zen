// REQ-APP-11 (pi-357): theme system — color token definitions, JSON loading, variable reference resolution, built-in themes.
// Placement exception: REQ-APP-11 → zen-tui (consumers MarkdownTheme/SelectListTheme/SettingsListTheme/Style all live in zen-tui).
// See reference/pi-original/packages/coding-agent/src/modes/interactive/theme/theme.ts
//
// Design notes:
// - RawColorValue (serde untagged) → ResolvedColor → crossterm::style::Color, three-stage resolution
// - Variable references resolved recursively, with cycle detection
// - Built-in themes embedded at compile time via include_str!, no IO
// - Theme provides fg()/bg()/style() color lookup + markdown_theme()/select_list_theme()/settings_list_theme() to build component Themes

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::OnceLock;

use crossterm::style::{Attribute, Color};
use serde::Deserialize;
use thiserror::Error;

use crate::markdown::MarkdownTheme;
use crate::screen::Style;

// ── Error types ──────────────────────────────────────────────────

/// Theme system error.
#[derive(Debug, Error)]
pub enum ThemeError {
    /// Invalid JSON format.
    #[error("invalid theme JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),
    /// Missing required color field.
    #[error("theme \"{name}\" missing required color: {field}")]
    MissingColor { name: String, field: String },
    /// Invalid color value format (e.g. hex parse failure).
    #[error("invalid color value for \"{field}\": {reason}")]
    InvalidColor { field: String, reason: String },
    /// Variable reference not found.
    #[error("variable reference \"{var}\" not found in theme \"{theme}\"")]
    VarNotFound { theme: String, var: String },
    /// Cyclic variable reference.
    #[error("circular variable reference detected: {chain}")]
    CircularVarRef { chain: String },
    /// Theme name contains the reserved character `/`.
    #[error("theme name \"{name}\" cannot contain '/'")]
    InvalidName { name: String },
    /// Failed to read theme file.
    #[error("failed to read theme file: {0}")]
    Io(#[from] std::io::Error),
}

// ── Raw color value (for serde deserialization) ─────────────────

/// Raw representation of a color value in JSON (`string | number`), used for serde `untagged` deserialization.
/// Converted to `ResolvedColor` after variable reference resolution during the resolve stage.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum RawColorValue {
    /// String value: hex `"#ff0000"` / variable reference `"accent"` / empty string `""`.
    Str(String),
    /// Integer 0-255: 256-color index.
    Int(u8),
}

// ── Resolved color ───────────────────────────────────────────────

/// Color value after variable reference resolution, directly convertible to `crossterm::style::Color`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolvedColor {
    /// RGB color (obtained from hex parsing).
    Rgb(u8, u8, u8),
    /// 256-color index.
    Indexed(u8),
    /// Terminal default color (empty string).
    Default,
}

// ── Foreground color token enum ──────────────────────────────────

/// Foreground color token enum; components look up colors from the theme through this enum.
/// Corresponds to the TS `ThemeColor` union type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorToken {
    /// Primary accent color (logo, selected item, cursor).
    Accent,
    /// Normal border.
    Border,
    /// Highlighted border.
    BorderAccent,
    /// Dimmed border.
    BorderMuted,
    /// Success state.
    Success,
    /// Error state.
    Error,
    /// Warning state.
    Warning,
    /// Secondary/dimmed text.
    Muted,
    /// More dimmed text.
    Dim,
    /// Default text.
    Text,
    /// Thinking block text.
    ThinkingText,
    /// User message text.
    UserMessageText,
    /// Custom message text.
    CustomMessageText,
    /// Custom message label.
    CustomMessageLabel,
    /// Tool title.
    ToolTitle,
    /// Tool output.
    ToolOutput,
    /// Markdown heading.
    MdHeading,
    /// Markdown link.
    MdLink,
    /// Markdown URL。
    MdLinkUrl,
    /// Inline code.
    MdCode,
    /// Code block.
    MdCodeBlock,
    /// Code block border.
    MdCodeBlockBorder,
    /// Quote text.
    MdQuote,
    /// Quote border.
    MdQuoteBorder,
    /// Horizontal rule.
    MdHr,
    /// List marker.
    MdListBullet,
    /// Diff added line.
    ToolDiffAdded,
    /// Diff removed line.
    ToolDiffRemoved,
    /// Diff context.
    ToolDiffContext,
    /// Syntax highlighting: comment.
    SyntaxComment,
    /// Syntax highlighting: keyword.
    SyntaxKeyword,
    /// Syntax highlighting: function name.
    SyntaxFunction,
    /// Syntax highlighting: variable name.
    SyntaxVariable,
    /// Syntax highlighting: string.
    SyntaxString,
    /// Syntax highlighting: number.
    SyntaxNumber,
    /// Syntax highlighting: type name.
    SyntaxType,
    /// Syntax highlighting: operator.
    SyntaxOperator,
    /// Syntax highlighting: punctuation.
    SyntaxPunctuation,
    /// Thinking level border: off.
    ThinkingOff,
    /// Thinking level border: minimal.
    ThinkingMinimal,
    /// Thinking level border: low.
    ThinkingLow,
    /// Thinking level border: medium.
    ThinkingMedium,
    /// Thinking level border: high.
    ThinkingHigh,
    /// Thinking level border: very high.
    ThinkingXhigh,
    /// Thinking level border: max.
    ThinkingMax,
    /// Bash mode border.
    BashMode,
}

// ── Background color token enum ─────────────────────────────────────────────

/// Background color token enum. Corresponds to TS `ThemeBg` union type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BgToken {
    /// Selected item background.
    SelectedBg,
    /// User message background.
    UserMessageBg,
    /// Custom message background.
    CustomMessageBg,
    /// Tool running background.
    ToolPendingBg,
    /// Tool success background.
    ToolSuccessBg,
    /// Tool error background.
    ToolErrorBg,
}

// ── ThemeColors (resolved color set) ──────────────────────────────────

/// Resolved set of all theme color tokens.
/// Each field corresponds to a color token, of type `crossterm::style::Color`.
/// Obtained from ThemeJson's colors section after resolving variable references.
#[derive(Debug, Clone)]
pub struct ThemeColors {
    // ── Core UI ──
    pub accent: Color,
    pub border: Color,
    pub border_accent: Color,
    pub border_muted: Color,
    pub success: Color,
    pub error: Color,
    pub warning: Color,
    pub muted: Color,
    pub dim: Color,
    pub text: Color,
    pub thinking_text: Color,
    // ── Backgrounds ──
    pub selected_bg: Color,
    pub user_message_bg: Color,
    pub user_message_text: Color,
    pub custom_message_bg: Color,
    pub custom_message_text: Color,
    pub custom_message_label: Color,
    pub tool_pending_bg: Color,
    pub tool_success_bg: Color,
    pub tool_error_bg: Color,
    pub tool_title: Color,
    pub tool_output: Color,
    // ── Markdown ──
    pub md_heading: Color,
    pub md_link: Color,
    pub md_link_url: Color,
    pub md_code: Color,
    pub md_code_block: Color,
    pub md_code_block_border: Color,
    pub md_quote: Color,
    pub md_quote_border: Color,
    pub md_hr: Color,
    pub md_list_bullet: Color,
    // ── Tool Diff ──
    pub tool_diff_added: Color,
    pub tool_diff_removed: Color,
    pub tool_diff_context: Color,
    // ── Syntax ──
    pub syntax_comment: Color,
    pub syntax_keyword: Color,
    pub syntax_function: Color,
    pub syntax_variable: Color,
    pub syntax_string: Color,
    pub syntax_number: Color,
    pub syntax_type: Color,
    pub syntax_operator: Color,
    pub syntax_punctuation: Color,
    // ── Thinking levels ──
    pub thinking_off: Color,
    pub thinking_minimal: Color,
    pub thinking_low: Color,
    pub thinking_medium: Color,
    pub thinking_high: Color,
    pub thinking_xhigh: Color,
    pub thinking_max: Color,
    // ── Bash mode ──
    pub bash_mode: Color,
}

// ── Theme (complete theme instance) ──────────────────────────────────────────

/// Complete theme instance holding the name and resolved color set.
/// Provides methods to get colors by token and construct a component Theme.
/// Components get colors via `Theme` methods and do not access `ThemeColors` fields directly.
#[derive(Debug, Clone)]
pub struct Theme {
    /// Theme name.
    pub name: String,
    /// Resolved color set.
    colors: ThemeColors,
    /// Source file path (None for built-in themes).
    source_path: Option<PathBuf>,
}

impl Theme {
    /// Terminal UI components call this method before rendering to get the current theme's foreground color—
    /// Just "give me the accent color"; components do not care whether the color comes from hex or a 256-color index.
    /// Returns the parsed crossterm Color.
    pub fn fg(&self, token: ColorToken) -> Color {
        match token {
            ColorToken::Accent => self.colors.accent,
            ColorToken::Border => self.colors.border,
            ColorToken::BorderAccent => self.colors.border_accent,
            ColorToken::BorderMuted => self.colors.border_muted,
            ColorToken::Success => self.colors.success,
            ColorToken::Error => self.colors.error,
            ColorToken::Warning => self.colors.warning,
            ColorToken::Muted => self.colors.muted,
            ColorToken::Dim => self.colors.dim,
            ColorToken::Text => self.colors.text,
            ColorToken::ThinkingText => self.colors.thinking_text,
            ColorToken::UserMessageText => self.colors.user_message_text,
            ColorToken::CustomMessageText => self.colors.custom_message_text,
            ColorToken::CustomMessageLabel => self.colors.custom_message_label,
            ColorToken::ToolTitle => self.colors.tool_title,
            ColorToken::ToolOutput => self.colors.tool_output,
            ColorToken::MdHeading => self.colors.md_heading,
            ColorToken::MdLink => self.colors.md_link,
            ColorToken::MdLinkUrl => self.colors.md_link_url,
            ColorToken::MdCode => self.colors.md_code,
            ColorToken::MdCodeBlock => self.colors.md_code_block,
            ColorToken::MdCodeBlockBorder => self.colors.md_code_block_border,
            ColorToken::MdQuote => self.colors.md_quote,
            ColorToken::MdQuoteBorder => self.colors.md_quote_border,
            ColorToken::MdHr => self.colors.md_hr,
            ColorToken::MdListBullet => self.colors.md_list_bullet,
            ColorToken::ToolDiffAdded => self.colors.tool_diff_added,
            ColorToken::ToolDiffRemoved => self.colors.tool_diff_removed,
            ColorToken::ToolDiffContext => self.colors.tool_diff_context,
            ColorToken::SyntaxComment => self.colors.syntax_comment,
            ColorToken::SyntaxKeyword => self.colors.syntax_keyword,
            ColorToken::SyntaxFunction => self.colors.syntax_function,
            ColorToken::SyntaxVariable => self.colors.syntax_variable,
            ColorToken::SyntaxString => self.colors.syntax_string,
            ColorToken::SyntaxNumber => self.colors.syntax_number,
            ColorToken::SyntaxType => self.colors.syntax_type,
            ColorToken::SyntaxOperator => self.colors.syntax_operator,
            ColorToken::SyntaxPunctuation => self.colors.syntax_punctuation,
            ColorToken::ThinkingOff => self.colors.thinking_off,
            ColorToken::ThinkingMinimal => self.colors.thinking_minimal,
            ColorToken::ThinkingLow => self.colors.thinking_low,
            ColorToken::ThinkingMedium => self.colors.thinking_medium,
            ColorToken::ThinkingHigh => self.colors.thinking_high,
            ColorToken::ThinkingXhigh => self.colors.thinking_xhigh,
            ColorToken::ThinkingMax => self.colors.thinking_max,
            ColorToken::BashMode => self.colors.bash_mode,
        }
    }

    /// Terminal UI components call this method before rendering to get the current theme's background color—
    /// Just "give me the user message background color".
    /// Returns the parsed crossterm Color.
    pub fn bg(&self, token: BgToken) -> Color {
        match token {
            BgToken::SelectedBg => self.colors.selected_bg,
            BgToken::UserMessageBg => self.colors.user_message_bg,
            BgToken::CustomMessageBg => self.colors.custom_message_bg,
            BgToken::ToolPendingBg => self.colors.tool_pending_bg,
            BgToken::ToolSuccessBg => self.colors.tool_success_bg,
            BgToken::ToolErrorBg => self.colors.tool_error_bg,
        }
    }

    /// Called when a component needs a combined "foreground + background + attributes" style—
    /// Get a complete Style in one line without manually assembling fg/bg/attributes.
    /// `attrs` are extra attributes (Bold/Italic, etc.), layered on top of the returned Style.
    pub fn style(
        &self,
        fg_token: ColorToken,
        bg_token: Option<BgToken>,
        attrs: &[Attribute],
    ) -> Style {
        Style {
            fg: Some(self.fg(fg_token)),
            bg: bg_token.map(|t| self.bg(t)),
            attributes: attrs.to_vec(),
        }
    }

    /// Called by the markdown renderer (T-TU-02-1) when rendering themed markdown—
    /// Constructs a MarkdownTheme from the current theme colors, replacing its Default dark fallback.
    /// Maps the 15 Style fields one by one from theme colors.
    pub fn markdown_theme(&self) -> MarkdownTheme {
        MarkdownTheme {
            text: self.style(ColorToken::Text, None, &[]),
            heading: self.style(ColorToken::MdHeading, None, &[Attribute::Bold]),
            bold: Style {
                fg: Some(self.fg(ColorToken::Text)),
                bg: None,
                attributes: vec![Attribute::Bold],
            },
            italic: Style {
                fg: Some(self.fg(ColorToken::MdHeading)),
                bg: None,
                attributes: vec![Attribute::Italic],
            },
            strikethrough: self.style(ColorToken::Muted, None, &[Attribute::CrossedOut]),
            underline: Style {
                fg: Some(self.fg(ColorToken::Text)),
                bg: None,
                attributes: vec![Attribute::Underlined],
            },
            link: self.style(ColorToken::MdLink, None, &[Attribute::Underlined]),
            link_url: self.style(ColorToken::MdLinkUrl, None, &[]),
            code: self.style(ColorToken::MdCode, None, &[]),
            code_block: self.style(ColorToken::MdCodeBlock, Some(BgToken::ToolPendingBg), &[]),
            code_block_border: self.style(ColorToken::MdCodeBlockBorder, None, &[]),
            quote: Style {
                fg: Some(self.fg(ColorToken::MdQuote)),
                bg: None,
                attributes: vec![Attribute::Italic],
            },
            quote_border: self.style(ColorToken::MdQuoteBorder, None, &[]),
            hr: self.style(ColorToken::MdHr, None, &[]),
            list_bullet: self.style(ColorToken::MdListBullet, None, &[]),
            code_block_indent: "  ".to_string(),
            syntax_theme_name: self.syntax_theme_name().to_string(),
        }
    }

    /// Called by the select list component (T-TU-01-4) during construction—
    /// Constructs a SelectListTheme from the current theme colors.
    pub fn select_list_theme(&self) -> crate::components::select_list::SelectListTheme {
        crate::components::select_list::SelectListTheme::new(
            self.style(ColorToken::Accent, Some(BgToken::SelectedBg), &[]),
            self.style(ColorToken::Muted, None, &[]),
            self.style(ColorToken::Muted, None, &[]),
            self.style(ColorToken::Muted, None, &[]),
        )
    }

    /// Called by the settings list component (T-TU-01-4) during construction—
    /// Constructs a SettingsListTheme from the current theme colors.
    pub fn settings_list_theme(&self) -> crate::components::settings_list::SettingsListTheme {
        crate::components::settings_list::SettingsListTheme {
            label_style: self.style(ColorToken::Text, None, &[]),
            label_selected_style: self.style(ColorToken::Accent, None, &[]),
            value_style: self.style(ColorToken::Muted, None, &[]),
            value_selected_style: self.style(ColorToken::Accent, None, &[]),
            description_style: self.style(ColorToken::Dim, None, &[]),
            hint_style: self.style(ColorToken::Dim, None, &[]),
            cursor_marker: "→ ".to_string(),
        }
    }

    /// Called by the syntax highlighting module (T-AP-10-1) when highlighting code—
    /// Returns the syntect theme name corresponding to the current theme (e.g. "base16-ocean.dark"),
    /// Used to select a built-in syntect theme. v1 maps by theme name: dark → base16-ocean.dark, light → base16-ocean.light.
    pub fn syntax_theme_name(&self) -> &str {
        match self.name.as_str() {
            "light" => "base16-ocean.light",
            _ => "base16-ocean.dark",
        }
    }

    /// Called by the thinking level selector component when rendering the border—
    /// Returns the corresponding border color based on ThinkingLevel.
    pub fn thinking_border_color(&self, level: zen_ai::types::ThinkingLevel) -> Color {
        match level {
            zen_ai::types::ThinkingLevel::Off => self.colors.thinking_off,
            zen_ai::types::ThinkingLevel::Minimal => self.colors.thinking_minimal,
            zen_ai::types::ThinkingLevel::Low => self.colors.thinking_low,
            zen_ai::types::ThinkingLevel::Medium => self.colors.thinking_medium,
            zen_ai::types::ThinkingLevel::High => self.colors.thinking_high,
            zen_ai::types::ThinkingLevel::XHigh => self.colors.thinking_xhigh,
            zen_ai::types::ThinkingLevel::Max => self.colors.thinking_max,
        }
    }

    /// Gets the source file path of the theme (None for built-in themes).
    pub fn source_path(&self) -> Option<&PathBuf> {
        self.source_path.as_ref()
    }
}

// ── ThemeJson (serde deserialization carrier, module-private) ──────────────────────

/// Deserialization carrier for JSON theme files. Converted to `Theme` after parsing.
#[derive(Debug, Deserialize)]
struct ThemeJson {
    /// Theme name (without `/`).
    name: String,
    /// Variable definitions.
    vars: Option<HashMap<String, RawColorValue>>,
    /// Color definitions.
    colors: ThemeJsonColors,
}

/// Color definitions section (the `colors` object in JSON).
/// Each field corresponds to a color token; the value is a hex/variable reference/256-color index/empty string.
#[derive(Debug, Deserialize)]
struct ThemeJsonColors {
    #[serde(rename = "accent")]
    accent: RawColorValue,
    #[serde(rename = "border")]
    border: RawColorValue,
    #[serde(rename = "borderAccent")]
    border_accent: RawColorValue,
    #[serde(rename = "borderMuted")]
    border_muted: RawColorValue,
    #[serde(rename = "success")]
    success: RawColorValue,
    #[serde(rename = "error")]
    error: RawColorValue,
    #[serde(rename = "warning")]
    warning: RawColorValue,
    #[serde(rename = "muted")]
    muted: RawColorValue,
    #[serde(rename = "dim")]
    dim: RawColorValue,
    #[serde(rename = "text")]
    text: RawColorValue,
    #[serde(rename = "thinkingText")]
    thinking_text: RawColorValue,
    #[serde(rename = "selectedBg")]
    selected_bg: RawColorValue,
    #[serde(rename = "userMessageBg")]
    user_message_bg: RawColorValue,
    #[serde(rename = "userMessageText")]
    user_message_text: RawColorValue,
    #[serde(rename = "customMessageBg")]
    custom_message_bg: RawColorValue,
    #[serde(rename = "customMessageText")]
    custom_message_text: RawColorValue,
    #[serde(rename = "customMessageLabel")]
    custom_message_label: RawColorValue,
    #[serde(rename = "toolPendingBg")]
    tool_pending_bg: RawColorValue,
    #[serde(rename = "toolSuccessBg")]
    tool_success_bg: RawColorValue,
    #[serde(rename = "toolErrorBg")]
    tool_error_bg: RawColorValue,
    #[serde(rename = "toolTitle")]
    tool_title: RawColorValue,
    #[serde(rename = "toolOutput")]
    tool_output: RawColorValue,
    #[serde(rename = "mdHeading")]
    md_heading: RawColorValue,
    #[serde(rename = "mdLink")]
    md_link: RawColorValue,
    #[serde(rename = "mdLinkUrl")]
    md_link_url: RawColorValue,
    #[serde(rename = "mdCode")]
    md_code: RawColorValue,
    #[serde(rename = "mdCodeBlock")]
    md_code_block: RawColorValue,
    #[serde(rename = "mdCodeBlockBorder")]
    md_code_block_border: RawColorValue,
    #[serde(rename = "mdQuote")]
    md_quote: RawColorValue,
    #[serde(rename = "mdQuoteBorder")]
    md_quote_border: RawColorValue,
    #[serde(rename = "mdHr")]
    md_hr: RawColorValue,
    #[serde(rename = "mdListBullet")]
    md_list_bullet: RawColorValue,
    #[serde(rename = "toolDiffAdded")]
    tool_diff_added: RawColorValue,
    #[serde(rename = "toolDiffRemoved")]
    tool_diff_removed: RawColorValue,
    #[serde(rename = "toolDiffContext")]
    tool_diff_context: RawColorValue,
    #[serde(rename = "syntaxComment")]
    syntax_comment: RawColorValue,
    #[serde(rename = "syntaxKeyword")]
    syntax_keyword: RawColorValue,
    #[serde(rename = "syntaxFunction")]
    syntax_function: RawColorValue,
    #[serde(rename = "syntaxVariable")]
    syntax_variable: RawColorValue,
    #[serde(rename = "syntaxString")]
    syntax_string: RawColorValue,
    #[serde(rename = "syntaxNumber")]
    syntax_number: RawColorValue,
    #[serde(rename = "syntaxType")]
    syntax_type: RawColorValue,
    #[serde(rename = "syntaxOperator")]
    syntax_operator: RawColorValue,
    #[serde(rename = "syntaxPunctuation")]
    syntax_punctuation: RawColorValue,
    #[serde(rename = "thinkingOff")]
    thinking_off: RawColorValue,
    #[serde(rename = "thinkingMinimal")]
    thinking_minimal: RawColorValue,
    #[serde(rename = "thinkingLow")]
    thinking_low: RawColorValue,
    #[serde(rename = "thinkingMedium")]
    thinking_medium: RawColorValue,
    #[serde(rename = "thinkingHigh")]
    thinking_high: RawColorValue,
    #[serde(rename = "thinkingXhigh")]
    thinking_xhigh: RawColorValue,
    /// Optional; defaults to falling back to thinkingXhigh.
    #[serde(rename = "thinkingMax")]
    thinking_max: Option<RawColorValue>,
    #[serde(rename = "bashMode")]
    bash_mode: RawColorValue,
}

// ── Color parsing utility functions ──────────────────────────────────────────────

/// Recursively resolves variable references—
/// If the value is a string and not hex/empty, treat it as a variable name to look up in vars;
/// Supports chained references (a → b → c) and detects circular references.
/// # Errors
/// Variable not found or circular reference → `ThemeError`.
fn resolve_var_ref(
    value: &RawColorValue,
    vars: &HashMap<String, RawColorValue>,
    visited: &mut HashSet<String>,
) -> Result<ResolvedColor, ThemeError> {
    match value {
        RawColorValue::Int(n) => Ok(ResolvedColor::Indexed(*n)),
        RawColorValue::Str(s) => {
            if s.is_empty() {
                Ok(ResolvedColor::Default)
            } else if s.starts_with('#') {
                parse_hex_color(s).map_err(|e| ThemeError::InvalidColor {
                    field: "color".to_string(),
                    reason: e,
                })
            } else {
                // Variable reference
                if visited.contains(s) {
                    return Err(ThemeError::CircularVarRef {
                        chain: visited.iter().cloned().collect::<Vec<_>>().join(" → ") + " → " + s,
                    });
                }
                visited.insert(s.clone());
                let ref_value = vars.get(s).ok_or_else(|| ThemeError::VarNotFound {
                    theme: String::new(), // Filled in by the caller
                    var: s.clone(),
                })?;
                resolve_var_ref(ref_value, vars, visited)
            }
        }
    }
}

/// Parses a color value into ResolvedColor (directly, without variable reference resolution).
#[allow(dead_code)]
fn resolve_color_direct(value: &RawColorValue) -> Result<ResolvedColor, ThemeError> {
    match value {
        RawColorValue::Int(n) => Ok(ResolvedColor::Indexed(*n)),
        RawColorValue::Str(s) => {
            if s.is_empty() {
                Ok(ResolvedColor::Default)
            } else if s.starts_with('#') {
                parse_hex_color(s).map_err(|e| ThemeError::InvalidColor {
                    field: "color".to_string(),
                    reason: e,
                })
            } else {
                Err(ThemeError::InvalidColor {
                    field: "color".to_string(),
                    reason: format!("expected hex color or empty string, got: {s}"),
                })
            }
        }
    }
}

/// Parses a color value into ResolvedColor (with variable reference resolution).
fn resolve_color(
    value: &RawColorValue,
    vars: &HashMap<String, RawColorValue>,
) -> Result<ResolvedColor, ThemeError> {
    let mut visited = HashSet::new();
    resolve_var_ref(value, vars, &mut visited)
}

/// Converts ResolvedColor to crossterm Color—
/// Rgb → Color::Rgb，Indexed → Color::AnsiValue，Default → Color::Reset。
fn resolved_to_crossterm(resolved: ResolvedColor) -> Color {
    match resolved {
        ResolvedColor::Rgb(r, g, b) => Color::Rgb { r, g, b },
        ResolvedColor::Indexed(n) => Color::AnsiValue(n),
        ResolvedColor::Default => Color::Reset,
    }
}

/// Parses a hex color string into ResolvedColor::Rgb.
/// Supports the `#RRGGBB` format.
fn parse_hex_color(hex: &str) -> Result<ResolvedColor, String> {
    let cleaned = hex.strip_prefix('#').unwrap_or(hex);
    if cleaned.len() != 6 {
        return Err(format!("invalid hex color length: {hex}"));
    }
    let r = u8::from_str_radix(&cleaned[0..2], 16)
        .map_err(|_| format!("invalid hex red component: {hex}"))?;
    let g = u8::from_str_radix(&cleaned[2..4], 16)
        .map_err(|_| format!("invalid hex green component: {hex}"))?;
    let b = u8::from_str_radix(&cleaned[4..6], 16)
        .map_err(|_| format!("invalid hex blue component: {hex}"))?;
    Ok(ResolvedColor::Rgb(r, g, b))
}

// ── Theme loading ──────────────────────────────────────────────────────

/// Loads a theme from a JSON string—
/// Parses JSON, validates required color fields, recursively resolves variable references, and returns Theme.
/// Both built-in and custom themes are loaded through this entry point.
/// # Errors
/// Invalid JSON format, missing required colors, circular variable references, or invalid color values → `ThemeError`.
pub fn load_theme_from_json(json: &str, source_path: Option<PathBuf>) -> Result<Theme, ThemeError> {
    let theme_json: ThemeJson = serde_json::from_str(json)?;
    let name = theme_json.name.clone();

    // Validate that the name does not contain '/'
    if name.contains('/') {
        return Err(ThemeError::InvalidName { name });
    }

    let vars = theme_json.vars.unwrap_or_default();
    let c = &theme_json.colors;

    // Helper closure: parse one color field
    let resolve = |value: &RawColorValue, _field: &str| -> Result<Color, ThemeError> {
        let resolved = resolve_color(value, &vars).map_err(|e| match e {
            ThemeError::VarNotFound { .. } => ThemeError::VarNotFound {
                theme: name.clone(),
                var: match value {
                    RawColorValue::Str(s) => s.clone(),
                    _ => String::new(),
                },
            },
            other => other,
        })?;
        Ok(resolved_to_crossterm(resolved))
    };

    // thinkingMax fallback: use thinkingXhigh when absent
    let thinking_max_value = c.thinking_max.as_ref().unwrap_or(&c.thinking_xhigh);

    let colors = ThemeColors {
        accent: resolve(&c.accent, "accent")?,
        border: resolve(&c.border, "border")?,
        border_accent: resolve(&c.border_accent, "borderAccent")?,
        border_muted: resolve(&c.border_muted, "borderMuted")?,
        success: resolve(&c.success, "success")?,
        error: resolve(&c.error, "error")?,
        warning: resolve(&c.warning, "warning")?,
        muted: resolve(&c.muted, "muted")?,
        dim: resolve(&c.dim, "dim")?,
        text: resolve(&c.text, "text")?,
        thinking_text: resolve(&c.thinking_text, "thinkingText")?,
        selected_bg: resolve(&c.selected_bg, "selectedBg")?,
        user_message_bg: resolve(&c.user_message_bg, "userMessageBg")?,
        user_message_text: resolve(&c.user_message_text, "userMessageText")?,
        custom_message_bg: resolve(&c.custom_message_bg, "customMessageBg")?,
        custom_message_text: resolve(&c.custom_message_text, "customMessageText")?,
        custom_message_label: resolve(&c.custom_message_label, "customMessageLabel")?,
        tool_pending_bg: resolve(&c.tool_pending_bg, "toolPendingBg")?,
        tool_success_bg: resolve(&c.tool_success_bg, "toolSuccessBg")?,
        tool_error_bg: resolve(&c.tool_error_bg, "toolErrorBg")?,
        tool_title: resolve(&c.tool_title, "toolTitle")?,
        tool_output: resolve(&c.tool_output, "toolOutput")?,
        md_heading: resolve(&c.md_heading, "mdHeading")?,
        md_link: resolve(&c.md_link, "mdLink")?,
        md_link_url: resolve(&c.md_link_url, "mdLinkUrl")?,
        md_code: resolve(&c.md_code, "mdCode")?,
        md_code_block: resolve(&c.md_code_block, "mdCodeBlock")?,
        md_code_block_border: resolve(&c.md_code_block_border, "mdCodeBlockBorder")?,
        md_quote: resolve(&c.md_quote, "mdQuote")?,
        md_quote_border: resolve(&c.md_quote_border, "mdQuoteBorder")?,
        md_hr: resolve(&c.md_hr, "mdHr")?,
        md_list_bullet: resolve(&c.md_list_bullet, "mdListBullet")?,
        tool_diff_added: resolve(&c.tool_diff_added, "toolDiffAdded")?,
        tool_diff_removed: resolve(&c.tool_diff_removed, "toolDiffRemoved")?,
        tool_diff_context: resolve(&c.tool_diff_context, "toolDiffContext")?,
        syntax_comment: resolve(&c.syntax_comment, "syntaxComment")?,
        syntax_keyword: resolve(&c.syntax_keyword, "syntaxKeyword")?,
        syntax_function: resolve(&c.syntax_function, "syntaxFunction")?,
        syntax_variable: resolve(&c.syntax_variable, "syntaxVariable")?,
        syntax_string: resolve(&c.syntax_string, "syntaxString")?,
        syntax_number: resolve(&c.syntax_number, "syntaxNumber")?,
        syntax_type: resolve(&c.syntax_type, "syntaxType")?,
        syntax_operator: resolve(&c.syntax_operator, "syntaxOperator")?,
        syntax_punctuation: resolve(&c.syntax_punctuation, "syntaxPunctuation")?,
        thinking_off: resolve(&c.thinking_off, "thinkingOff")?,
        thinking_minimal: resolve(&c.thinking_minimal, "thinkingMinimal")?,
        thinking_low: resolve(&c.thinking_low, "thinkingLow")?,
        thinking_medium: resolve(&c.thinking_medium, "thinkingMedium")?,
        thinking_high: resolve(&c.thinking_high, "thinkingHigh")?,
        thinking_xhigh: resolve(&c.thinking_xhigh, "thinkingXhigh")?,
        thinking_max: resolve(thinking_max_value, "thinkingMax")?,
        bash_mode: resolve(&c.bash_mode, "bashMode")?,
    };

    Ok(Theme {
        name,
        colors,
        source_path,
    })
}

// ── Built-in themes ──────────────────────────────────────────────────────

/// Built-in dark theme JSON (embedded at compile time).
const BUILTIN_DARK_JSON: &str = include_str!("builtin_dark.json");

/// Built-in light theme JSON (embedded at compile time).
const BUILTIN_LIGHT_JSON: &str = include_str!("builtin_light.json");

/// Built-in dark theme (lazy parsing cache).
static BUILTIN_DARK: OnceLock<Theme> = OnceLock::new();

/// Built-in light theme (lazy parsing cache).
static BUILTIN_LIGHT: OnceLock<Theme> = OnceLock::new();

/// Gets the built-in dark theme—
/// Compile-time embedded dark.json; lazily parsed and cached on first call.
/// No IO, no panic; guaranteed always available.
pub fn builtin_dark_theme() -> &'static Theme {
    BUILTIN_DARK.get_or_init(|| {
        load_theme_from_json(BUILTIN_DARK_JSON, None)
            .expect("builtin dark theme JSON is valid (embedded at compile time)")
    })
}

/// Gets the built-in light theme—
/// Compile-time embedded light.json; lazily parsed and cached on first call.
pub fn builtin_light_theme() -> &'static Theme {
    BUILTIN_LIGHT.get_or_init(|| {
        load_theme_from_json(BUILTIN_LIGHT_JSON, None)
            .expect("builtin light theme JSON is valid (embedded at compile time)")
    })
}

/// Looks up a built-in theme by name—
/// "dark" → dark theme, "light" → light theme, otherwise → None.
/// Filesystem lookup for custom themes is handled by ThemeController; this function does no IO.
pub fn get_builtin_theme(name: &str) -> Option<&'static Theme> {
    match name {
        "dark" => Some(builtin_dark_theme()),
        "light" => Some(builtin_light_theme()),
        _ => None,
    }
}

/// Gets all built-in theme names.
pub fn builtin_theme_names() -> &'static [&'static str] {
    &["dark", "light"]
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hex_color_valid() {
        assert!(matches!(
            parse_hex_color("#ff0000"),
            Ok(ResolvedColor::Rgb(255, 0, 0))
        ));
        assert!(matches!(
            parse_hex_color("#00ff00"),
            Ok(ResolvedColor::Rgb(0, 255, 0))
        ));
        assert!(matches!(
            parse_hex_color("#0000ff"),
            Ok(ResolvedColor::Rgb(0, 0, 255))
        ));
        assert!(matches!(
            parse_hex_color("#000000"),
            Ok(ResolvedColor::Rgb(0, 0, 0))
        ));
        assert!(matches!(
            parse_hex_color("#ffffff"),
            Ok(ResolvedColor::Rgb(255, 255, 255))
        ));
    }

    #[test]
    fn test_parse_hex_color_invalid() {
        assert!(parse_hex_color("#fff").is_err());
        assert!(parse_hex_color("#gggggg").is_err());
        assert!(parse_hex_color("red").is_err());
    }

    #[test]
    fn test_resolve_color_direct_hex() {
        let value = RawColorValue::Str("#ff0000".to_string());
        assert!(matches!(
            resolve_color_direct(&value),
            Ok(ResolvedColor::Rgb(255, 0, 0))
        ));
    }

    #[test]
    fn test_resolve_color_direct_empty() {
        let value = RawColorValue::Str(String::new());
        assert!(matches!(
            resolve_color_direct(&value),
            Ok(ResolvedColor::Default)
        ));
    }

    #[test]
    fn test_resolve_color_direct_index() {
        let value = RawColorValue::Int(128);
        assert!(matches!(
            resolve_color_direct(&value),
            Ok(ResolvedColor::Indexed(128))
        ));
    }

    #[test]
    fn test_resolve_var_ref_direct_hex() {
        let vars = HashMap::new();
        let value = RawColorValue::Str("#ff0000".to_string());
        assert!(matches!(
            resolve_color(&value, &vars),
            Ok(ResolvedColor::Rgb(255, 0, 0))
        ));
    }

    #[test]
    fn test_resolve_var_ref_simple() {
        let mut vars = HashMap::new();
        vars.insert(
            "accent".to_string(),
            RawColorValue::Str("#8abeb7".to_string()),
        );
        let value = RawColorValue::Str("accent".to_string());
        assert!(matches!(
            resolve_color(&value, &vars),
            Ok(ResolvedColor::Rgb(0x8a, 0xbe, 0xb7))
        ));
    }

    #[test]
    fn test_resolve_var_ref_chain() {
        let mut vars = HashMap::new();
        vars.insert(
            "base".to_string(),
            RawColorValue::Str("#ff0000".to_string()),
        );
        vars.insert("accent".to_string(), RawColorValue::Str("base".to_string()));
        let value = RawColorValue::Str("accent".to_string());
        assert!(matches!(
            resolve_color(&value, &vars),
            Ok(ResolvedColor::Rgb(255, 0, 0))
        ));
    }

    #[test]
    fn test_resolve_var_ref_circular() {
        let mut vars = HashMap::new();
        vars.insert("a".to_string(), RawColorValue::Str("b".to_string()));
        vars.insert("b".to_string(), RawColorValue::Str("a".to_string()));
        let value = RawColorValue::Str("a".to_string());
        assert!(matches!(
            resolve_color(&value, &vars),
            Err(ThemeError::CircularVarRef { .. })
        ));
    }

    #[test]
    fn test_resolve_var_ref_not_found() {
        let vars = HashMap::new();
        let value = RawColorValue::Str("nonexistent".to_string());
        assert!(matches!(
            resolve_color(&value, &vars),
            Err(ThemeError::VarNotFound { .. })
        ));
    }

    #[test]
    fn test_resolved_to_crossterm() {
        assert_eq!(
            resolved_to_crossterm(ResolvedColor::Rgb(255, 0, 0)),
            Color::Rgb { r: 255, g: 0, b: 0 }
        );
        assert_eq!(
            resolved_to_crossterm(ResolvedColor::Indexed(128)),
            Color::AnsiValue(128)
        );
        assert_eq!(resolved_to_crossterm(ResolvedColor::Default), Color::Reset);
    }

    #[test]
    fn test_load_builtin_dark_theme() {
        let theme = builtin_dark_theme();
        assert_eq!(theme.name, "dark");
        // Verify some known colors
        assert_eq!(
            theme.fg(ColorToken::Accent),
            Color::Rgb {
                r: 0x8a,
                g: 0xbe,
                b: 0xb7
            }
        );
        assert_eq!(
            theme.fg(ColorToken::Error),
            Color::Rgb {
                r: 0xcc,
                g: 0x66,
                b: 0x66
            }
        );
    }

    #[test]
    fn test_load_builtin_light_theme() {
        let theme = builtin_light_theme();
        assert_eq!(theme.name, "light");
        assert_eq!(
            theme.fg(ColorToken::Accent),
            Color::Rgb {
                r: 0x5a,
                g: 0x80,
                b: 0x80
            }
        );
    }

    #[test]
    fn test_get_builtin_theme() {
        assert!(get_builtin_theme("dark").is_some());
        assert!(get_builtin_theme("light").is_some());
        assert!(get_builtin_theme("nonexistent").is_none());
    }

    #[test]
    fn test_theme_fg_all_tokens() {
        let theme = builtin_dark_theme();
        // Ensure every token can resolve to a color (without panicking)
        let tokens = [
            ColorToken::Accent,
            ColorToken::Border,
            ColorToken::BorderAccent,
            ColorToken::BorderMuted,
            ColorToken::Success,
            ColorToken::Error,
            ColorToken::Warning,
            ColorToken::Muted,
            ColorToken::Dim,
            ColorToken::Text,
            ColorToken::ThinkingText,
            ColorToken::UserMessageText,
            ColorToken::CustomMessageText,
            ColorToken::CustomMessageLabel,
            ColorToken::ToolTitle,
            ColorToken::ToolOutput,
            ColorToken::MdHeading,
            ColorToken::MdLink,
            ColorToken::MdLinkUrl,
            ColorToken::MdCode,
            ColorToken::MdCodeBlock,
            ColorToken::MdCodeBlockBorder,
            ColorToken::MdQuote,
            ColorToken::MdQuoteBorder,
            ColorToken::MdHr,
            ColorToken::MdListBullet,
            ColorToken::ToolDiffAdded,
            ColorToken::ToolDiffRemoved,
            ColorToken::ToolDiffContext,
            ColorToken::SyntaxComment,
            ColorToken::SyntaxKeyword,
            ColorToken::SyntaxFunction,
            ColorToken::SyntaxVariable,
            ColorToken::SyntaxString,
            ColorToken::SyntaxNumber,
            ColorToken::SyntaxType,
            ColorToken::SyntaxOperator,
            ColorToken::SyntaxPunctuation,
            ColorToken::ThinkingOff,
            ColorToken::ThinkingMinimal,
            ColorToken::ThinkingLow,
            ColorToken::ThinkingMedium,
            ColorToken::ThinkingHigh,
            ColorToken::ThinkingXhigh,
            ColorToken::ThinkingMax,
            ColorToken::BashMode,
        ];
        for token in tokens {
            let _ = theme.fg(token);
        }
    }

    #[test]
    fn test_theme_bg_all_tokens() {
        let theme = builtin_dark_theme();
        let tokens = [
            BgToken::SelectedBg,
            BgToken::UserMessageBg,
            BgToken::CustomMessageBg,
            BgToken::ToolPendingBg,
            BgToken::ToolSuccessBg,
            BgToken::ToolErrorBg,
        ];
        for token in tokens {
            let _ = theme.bg(token);
        }
    }

    #[test]
    fn test_theme_style() {
        let theme = builtin_dark_theme();
        let style = theme.style(
            ColorToken::Accent,
            Some(BgToken::SelectedBg),
            &[Attribute::Bold],
        );
        assert_eq!(
            style.fg,
            Some(Color::Rgb {
                r: 0x8a,
                g: 0xbe,
                b: 0xb7
            })
        );
        assert_eq!(
            style.bg,
            Some(Color::Rgb {
                r: 0x3a,
                g: 0x3a,
                b: 0x4a
            })
        );
        assert_eq!(style.attributes, vec![Attribute::Bold]);
    }

    #[test]
    fn test_theme_markdown_theme() {
        let theme = builtin_dark_theme();
        let md_theme = theme.markdown_theme();
        // Verify the text style
        assert_eq!(
            md_theme.text.fg,
            Some(Color::Rgb {
                r: 0xd4,
                g: 0xd4,
                b: 0xd4
            })
        );
    }

    #[test]
    fn test_theme_select_list_theme() {
        let theme = builtin_dark_theme();
        let _sl_theme = theme.select_list_theme();
        // Successful construction is enough
    }

    #[test]
    fn test_theme_settings_list_theme() {
        let theme = builtin_dark_theme();
        let _st_theme = theme.settings_list_theme();
        // Successful construction is enough
    }

    #[test]
    fn test_theme_syntax_theme_name() {
        assert_eq!(
            builtin_dark_theme().syntax_theme_name(),
            "base16-ocean.dark"
        );
        assert_eq!(
            builtin_light_theme().syntax_theme_name(),
            "base16-ocean.light"
        );
    }

    #[test]
    fn test_theme_thinking_border_color() {
        let theme = builtin_dark_theme();
        assert_eq!(
            theme.thinking_border_color(zen_ai::types::ThinkingLevel::Off),
            Color::Rgb {
                r: 0x50,
                g: 0x50,
                b: 0x50
            }
        );
        assert_eq!(
            theme.thinking_border_color(zen_ai::types::ThinkingLevel::Max),
            Color::Rgb {
                r: 0xff,
                g: 0x5f,
                b: 0xff
            }
        );
    }

    /// Helper function: generates a JSON theme string for testing (using 256-color indices).
    fn test_theme_json(
        name: &str,
        accent: u8,
        thinking_xhigh: u8,
        include_thinking_max: bool,
    ) -> String {
        let thinking_max_field = if include_thinking_max {
            format!("\"thinkingMax\":{},", thinking_xhigh + 1)
        } else {
            String::new()
        };
        format!(
            concat!(
                "{{\"name\":\"{name}\",\"colors\":{{",
                "\"accent\":{accent},\"border\":1,\"borderAccent\":1,",
                "\"borderMuted\":1,\"success\":1,\"error\":1,",
                "\"warning\":1,\"muted\":1,\"dim\":1,",
                "\"text\":1,\"thinkingText\":1,",
                "\"selectedBg\":1,\"userMessageBg\":1,",
                "\"userMessageText\":1,\"customMessageBg\":1,",
                "\"customMessageText\":1,\"customMessageLabel\":1,",
                "\"toolPendingBg\":1,\"toolSuccessBg\":1,",
                "\"toolErrorBg\":1,\"toolTitle\":1,\"toolOutput\":1,",
                "\"mdHeading\":1,\"mdLink\":1,\"mdLinkUrl\":1,",
                "\"mdCode\":1,\"mdCodeBlock\":1,\"mdCodeBlockBorder\":1,",
                "\"mdQuote\":1,\"mdQuoteBorder\":1,\"mdHr\":1,",
                "\"mdListBullet\":1,",
                "\"toolDiffAdded\":1,\"toolDiffRemoved\":1,",
                "\"toolDiffContext\":1,",
                "\"syntaxComment\":1,\"syntaxKeyword\":1,",
                "\"syntaxFunction\":1,\"syntaxVariable\":1,",
                "\"syntaxString\":1,\"syntaxNumber\":1,",
                "\"syntaxType\":1,\"syntaxOperator\":1,",
                "\"syntaxPunctuation\":1,",
                "\"thinkingOff\":1,\"thinkingMinimal\":2,",
                "\"thinkingLow\":3,\"thinkingMedium\":4,",
                "\"thinkingHigh\":5,\"thinkingXhigh\":{thinking_xhigh},",
                "{thinking_max_field}",
                "\"bashMode\":1}}}}"
            ),
            name = name,
            accent = accent,
            thinking_xhigh = thinking_xhigh,
            thinking_max_field = thinking_max_field,
        )
    }

    #[test]
    fn test_thinking_max_fallback() {
        // Construct a JSON without thinkingMax, verifying fallback to thinkingXhigh
        let json = test_theme_json("test", 1, 6, false);
        let theme = load_theme_from_json(&json, None).unwrap();
        // thinkingMax should fall back to thinkingXhigh's value (6)
        assert_eq!(theme.fg(ColorToken::ThinkingMax), Color::AnsiValue(6));
    }

    #[test]
    fn test_invalid_theme_name() {
        let json = test_theme_json("invalid/name", 1, 1, false);
        assert!(matches!(
            load_theme_from_json(&json, None),
            Err(ThemeError::InvalidName { .. })
        ));
    }

    #[test]
    fn test_load_theme_with_vars() {
        // Use JSON with variable references
        let json = format!(
            concat!(
                "{{\"name\":\"test-vars\",\"vars\":{{\"primary\":\"#aabbcc\",\"secondary\":\"primary\"}},\"colors\":{{",
                "\"accent\":\"secondary\",\"border\":1,\"borderAccent\":1,",
                "\"borderMuted\":1,\"success\":1,\"error\":1,",
                "\"warning\":1,\"muted\":1,\"dim\":1,",
                "\"text\":1,\"thinkingText\":1,",
                "\"selectedBg\":1,\"userMessageBg\":1,",
                "\"userMessageText\":1,\"customMessageBg\":1,",
                "\"customMessageText\":1,\"customMessageLabel\":1,",
                "\"toolPendingBg\":1,\"toolSuccessBg\":1,",
                "\"toolErrorBg\":1,\"toolTitle\":1,\"toolOutput\":1,",
                "\"mdHeading\":1,\"mdLink\":1,\"mdLinkUrl\":1,",
                "\"mdCode\":1,\"mdCodeBlock\":1,\"mdCodeBlockBorder\":1,",
                "\"mdQuote\":1,\"mdQuoteBorder\":1,\"mdHr\":1,",
                "\"mdListBullet\":1,",
                "\"toolDiffAdded\":1,\"toolDiffRemoved\":1,",
                "\"toolDiffContext\":1,",
                "\"syntaxComment\":1,\"syntaxKeyword\":1,",
                "\"syntaxFunction\":1,\"syntaxVariable\":1,",
                "\"syntaxString\":1,\"syntaxNumber\":1,",
                "\"syntaxType\":1,\"syntaxOperator\":1,",
                "\"syntaxPunctuation\":1,",
                "\"thinkingOff\":1,\"thinkingMinimal\":1,",
                "\"thinkingLow\":1,\"thinkingMedium\":1,",
                "\"thinkingHigh\":1,\"thinkingXhigh\":1,",
                "\"bashMode\":1}}}}"
            ),
        );
        let theme = load_theme_from_json(&json, None).unwrap();
        // accent → secondary → primary → #aabbcc
        assert_eq!(
            theme.fg(ColorToken::Accent),
            Color::Rgb {
                r: 0xaa,
                g: 0xbb,
                b: 0xcc
            }
        );
    }

    #[test]
    fn test_load_theme_256_color() {
        let json = test_theme_json("test-256", 128, 1, false);
        let theme = load_theme_from_json(&json, None).unwrap();
        assert_eq!(theme.fg(ColorToken::Accent), Color::AnsiValue(128));
    }
}
