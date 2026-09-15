// REQ-APP-11 (pi-358): footer — bottom status bar.
// Mirrors the original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/footer.ts.
// Attribution note: canonical definitions of `UsageTotals`/`add_usage_to_totals` have moved to output/usage_totals.rs
// (pi-324, landed in T-AP-06-4, closing the fallback migration seam registered by T-AP-11-2); this file retains the footer display domain's
// `compute_usage_totals` (including the latest assistant cache hit rate) and `format_tokens`. git branch/session name
// Supplied by the caller (T-AP-11-1) via `FooterData` (wired in once T-AP-06-3 lands).

use crate::output::usage_totals::{add_usage_to_totals, UsageTotals};
use zen_ai::types::Message;
use zen_core::session::tree::SessionTreeEntry;
use zen_tui::components::Component;
use zen_tui::screen::{ScreenBuffer, Style};
use zen_tui::theme::{ColorToken, Theme};
use zen_tui::wrap::truncate_to_width;

/// Compact formatting for token counts (<1000 as-is; <10k one decimal + k; <1M integer + k; likewise for M) —
/// the footer status bar compresses large numbers to "1.2k / 345k / 2.1M". Consumer: footer rendering.
pub fn format_tokens(count: u64) -> String {
    if count < 1000 {
        return count.to_string();
    }
    if count < 10_000 {
        return format!("{:.1}k", count as f64 / 1000.0);
    }
    if count < 1_000_000 {
        return format!("{}k", count / 1000);
    }
    if count < 10_000_000 {
        return format!("{:.1}M", count as f64 / 1_000_000.0);
    }
    format!("{}M", count / 1_000_000)
}

/// Collapse cwd to `~`/`~/…` (when home is inside home) — the footer's first line shows the current directory. Consumer: footer rendering.
pub fn format_cwd_for_footer(cwd: &str, home: Option<&str>) -> String {
    let Some(home) = home else {
        return cwd.to_string();
    };
    let home = home.trim_end_matches('/');
    if home.is_empty() {
        return cwd.to_string();
    }
    // Pure string prefix check (aligned with pi's path.resolve, no file-existence check; avoids canonicalize failing on non-existent paths).
    if cwd == home {
        return "~".to_string();
    }
    let prefix = format!("{home}/");
    if let Some(rest) = cwd.strip_prefix(&prefix) {
        return format!("~/{rest}");
    }
    cwd.to_string()
}

/// Iterate over session entries, accumulating token usage (summing usage from each of the
/// assistant/toolResult/branch_summary/compaction entries), and return the cache hit rate of the most recent assistant (cacheRead/(input+cacheRead+cacheWrite)).
/// Pure function; called by T-AP-11-1 each frame when assembling FooterData.
pub fn compute_usage_totals(entries: &[SessionTreeEntry]) -> (UsageTotals, Option<f64>) {
    let mut totals = UsageTotals::default();
    let mut latest_cache_hit_rate: Option<f64> = None;
    for entry in entries {
        match entry {
            SessionTreeEntry::Message {
                message: Message::Assistant(a),
                ..
            } => {
                add_usage_to_totals(&mut totals, &a.usage);
                let prompt = a.usage.input + a.usage.cache_read + a.usage.cache_write;
                if prompt > 0 {
                    latest_cache_hit_rate = Some(a.usage.cache_read as f64 / prompt as f64 * 100.0);
                }
            }
            SessionTreeEntry::Message {
                message: Message::ToolResult(t),
                ..
            } => {
                if let Some(usage) = &t.usage {
                    add_usage_to_totals(&mut totals, usage);
                }
            }
            SessionTreeEntry::BranchSummary { usage: Some(u), .. }
            | SessionTreeEntry::Compaction { usage: Some(u), .. } => {
                add_usage_to_totals(&mut totals, u);
            }
            _ => {}
        }
    }
    (totals, latest_cache_hit_rate)
}

/// Data snapshot rendered by the footer each frame (plain data, assembled by T-AP-11-1 each frame from AgentSession).
/// Clone/Debug for T-AP-11-1's UI event queue (UiEvent::Footer cloned across tasks with the message).
#[derive(Debug, Clone)]
pub struct FooterData {
    /// Current working directory (collapsed to `~` before display).
    pub pwd: String,
    /// Git branch name (provided by the upper layer until T-AP-06-3 lands).
    pub git_branch: Option<String>,
    /// Session name (SessionInfo entry).
    pub session_name: Option<String>,
    /// Current provider name (footer displays `(provider) model_id`, aligned with pi; empty string = placeholder stage).
    pub model_provider: String,
    /// Current model id.
    pub model_id: String,
    /// Current thinking level (shown when the model supports thinking).
    pub thinking_level: Option<zen_ai::types::ThinkingLevel>,
    /// Cumulative usage snapshot.
    pub usage: UsageTotals,
    /// Context occupancy percentage (None = unknown, e.g. after compaction).
    pub context_percent: Option<f64>,
    /// Context window token count.
    pub context_window: u64,
    /// Recent assistant cache hit rate (%).
    pub cache_hit_rate: Option<f64>,
    /// Auto-compaction toggle (decides the `(auto)` marker).
    pub auto_compact_enabled: bool,
}

/// Bottom status bar: pwd (with branch/session name) + usage stats + context percentage + right-aligned model name/thinking level
/// T-AP-11-1 renders each frame driven by the FooterData snapshot; data assembly happens outside the component (keeps it testable).
pub struct FooterComponent<'a> {
    theme: &'a Theme,
    data: Option<FooterData>,
}

impl<'a> FooterComponent<'a> {
    /// Constructed from the theme (no data before the first render; renders 0 lines).
    pub fn new(theme: &'a Theme) -> Self {
        FooterComponent { theme, data: None }
    }

    /// Update the current snapshot (called per frame by T-AP-11-1; the component does not render before the first call).
    pub fn set_data(&mut self, data: FooterData) {
        self.data = Some(data);
    }

    /// Whether there is data (whether placeholder height is needed).
    pub fn has_data(&self) -> bool {
        self.data.is_some()
    }
}

impl Component for FooterComponent<'_> {
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        let Some(data) = &self.data else {
            return 0;
        };
        let theme = self.theme;
        let dim = Style::fg(theme.fg(ColorToken::Dim));
        let error = Style::fg(theme.fg(ColorToken::Error));
        let warning = Style::fg(theme.fg(ColorToken::Warning));

        // 1. pwd line (with branch/session name; truncated if too wide)
        let mut pwd_line = format_cwd_for_footer(&data.pwd, std::env::var("HOME").ok().as_deref());
        if let Some(branch) = &data.git_branch {
            pwd_line = format!("{pwd_line} ({branch})");
        }
        if let Some(name) = &data.session_name {
            pwd_line = format!("{pwd_line} • {name}");
        }
        let pwd_line = truncate_to_width(&pwd_line, width as usize, "…");

        // 2. Stats line
        let mut parts: Vec<String> = Vec::new();
        if data.usage.input > 0 {
            parts.push(format!("↑{}", format_tokens(data.usage.input)));
        }
        if data.usage.output > 0 {
            parts.push(format!("↓{}", format_tokens(data.usage.output)));
        }
        if data.usage.cache_read > 0 {
            parts.push(format!("R{}", format_tokens(data.usage.cache_read)));
        }
        if data.usage.cache_write > 0 {
            parts.push(format!("W{}", format_tokens(data.usage.cache_write)));
        }
        if data.usage.cache_read > 0 || data.usage.cache_write > 0 {
            if let Some(hit) = data.cache_hit_rate {
                parts.push(format!("CH{:.1}%", hit));
            }
        }
        if data.usage.cost > 0.0 {
            parts.push(format!("${:.3}", data.usage.cost));
        }
        let context_display = match data.context_percent {
            Some(p) => {
                let auto = if data.auto_compact_enabled {
                    " (auto)"
                } else {
                    ""
                };
                format!("{:.1}%/{}{auto}", p, format_tokens(data.context_window))
            }
            None => {
                let auto = if data.auto_compact_enabled {
                    " (auto)"
                } else {
                    ""
                };
                format!("?/{}{auto}", format_tokens(data.context_window))
            }
        };
        // Color the context ratio beyond thresholds (>90 error, >70 warning)
        let ctx_style = match data.context_percent {
            Some(p) if p > 90.0 => &error,
            Some(p) if p > 70.0 => &warning,
            _ => &dim,
        };
        let stats = parts.join(" ");

        // Right side: provider + model name + thinking level (aligned with pi `(provider) model`; during placeholder, empty provider hides the parentheses)
        let mut right = if data.model_provider.is_empty() {
            data.model_id.clone()
        } else {
            format!("({}) {}", data.model_provider, data.model_id)
        };
        if let Some(level) = &data.thinking_level {
            let lv = level.to_string().to_ascii_lowercase();
            right = if lv == "off" {
                format!("{right} • thinking off")
            } else {
                format!("{right} • {lv}")
            };
        }

        let mut painted = 0u16;
        if height > painted {
            buf.fill(x, y + painted, width, 1, &Style::default());
            buf.write(x, y + painted, &dim, &pwd_line);
            painted += 1;
        }
        if height > painted {
            buf.fill(x, y + painted, width, 1, &Style::default());
            // Write in segments: stats segment dim + context segment threshold color + right-aligned model (advance by columns, avoiding byte/column mixing)
            let mut col = x;
            if !stats.is_empty() {
                buf.write(col, y + painted, &dim, &stats);
                col += stw(&stats) as u16 + 1;
            }
            buf.write(col, y + painted, ctx_style, &context_display);
            col += stw(&context_display) as u16;
            let right_x = x + width.saturating_sub(stw(&right) as u16);
            if right_x > col {
                buf.write(right_x, y + painted, &dim, &right);
            }
            painted += 1;
        }
        painted
    }
}

fn stw(s: &str) -> usize {
    zen_tui::screen::visible_width(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zen_ai::types::{Cost, Usage};

    fn theme() -> &'static Theme {
        zen_tui::theme::builtin_dark_theme()
    }

    fn usage(i: u64, o: u64, cr: u64) -> Usage {
        Usage {
            input: i,
            output: o,
            cache_read: cr,
            cache_write: 0,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: i + o,
            cost: Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
        }
    }

    #[test]
    fn format_tokens_scales() {
        assert_eq!(format_tokens(0), "0");
        assert_eq!(format_tokens(999), "999");
        assert_eq!(format_tokens(1200), "1.2k");
        assert_eq!(format_tokens(345000), "345k");
        assert_eq!(format_tokens(2000000), "2.0M");
    }

    #[test]
    fn format_cwd_in_home() {
        assert_eq!(format_cwd_for_footer("/Users/x", Some("/Users/x")), "~");
        assert_eq!(
            format_cwd_for_footer("/Users/x/proj", Some("/Users/x")),
            "~/proj"
        );
        assert_eq!(format_cwd_for_footer("/other", Some("/Users/x")), "/other");
        assert_eq!(format_cwd_for_footer("/tmp/a", None), "/tmp/a");
    }

    #[test]
    fn compute_usage_totals_sums_entries() {
        // assistant message (with usage)
        let msg = Message::Assistant(zen_ai::types::AssistantMessage {
            content: vec![],
            api: "openai".into(),
            provider: "p".into(),
            model: "m".into(),
            response_model: None,
            response_id: None,
            usage: usage(10, 5, 80),
            stop_reason: zen_ai::types::StopReason::Stop,
            error_message: None,
            timestamp: 0,
        });
        let entries = vec![SessionTreeEntry::Message {
            id: "a1".into(),
            parent_id: None,
            timestamp: "t".into(),
            message: msg,
        }];
        let (totals, hit) = compute_usage_totals(&entries);
        assert_eq!(totals.input, 10);
        assert_eq!(totals.output, 5);
        assert_eq!(totals.cache_read, 80);
        // hit = 80/(10+80+0)*100 = 88.9
        assert!((hit.unwrap() - 88.888).abs() < 0.01);
    }

    #[test]
    fn compute_usage_totals_empty() {
        let (totals, hit) = compute_usage_totals(&[]);
        assert_eq!(totals, UsageTotals::default());
        assert!(hit.is_none());
    }

    #[test]
    fn footer_renders_pwd_and_stats() {
        let mut comp = FooterComponent::new(theme());
        assert!(!comp.has_data());
        comp.set_data(FooterData {
            pwd: "/tmp/proj".into(),
            git_branch: Some("main".into()),
            session_name: None,
            model_provider: "opencode-go".into(),
            model_id: "deepseek-v4-flash".into(),
            thinking_level: None,
            usage: UsageTotals {
                input: 1200,
                output: 500,
                cache_read: 800,
                cache_write: 100,
                cost: 0.0,
            },
            context_percent: Some(42.0),
            context_window: 128_000,
            cache_hit_rate: Some(50.0),
            auto_compact_enabled: true,
        });
        assert!(comp.has_data());
        let mut buf = ScreenBuffer::new(140, 3);
        let painted = comp.render(&mut buf, 0, 0, 140, 3);
        assert_eq!(painted, 2);
        let mut out = Vec::new();
        buf.flush(&mut out).unwrap();
        let text = crate::components::strip_ansi(&String::from_utf8_lossy(&out));
        assert!(text.contains("/tmp/proj"));
        assert!(text.contains("main"));
        assert!(text.contains("(opencode-go) deepseek-v4-flash"));
        assert!(text.contains("↑1.2k"));
        assert!(text.contains("42.0%"));
    }

    #[test]
    fn footer_renders_model_without_provider_placeholder() {
        // Placeholder stage (provider is empty string): do not show empty parentheses, fall back to bare model name
        let mut comp = FooterComponent::new(theme());
        comp.set_data(FooterData {
            pwd: "/tmp/proj".into(),
            git_branch: None,
            session_name: None,
            model_provider: String::new(),
            model_id: "deepseek-v4-flash".into(),
            thinking_level: None,
            usage: UsageTotals::default(),
            context_percent: None,
            context_window: 0,
            cache_hit_rate: None,
            auto_compact_enabled: false,
        });
        let mut buf = ScreenBuffer::new(140, 3);
        comp.render(&mut buf, 0, 0, 140, 3);
        let mut out = Vec::new();
        buf.flush(&mut out).unwrap();
        let text = crate::components::strip_ansi(&String::from_utf8_lossy(&out));
        assert!(text.contains("deepseek-v4-flash"));
        assert!(!text.contains("()"));
    }

    #[test]
    fn footer_no_data_renders_zero() {
        let mut comp = FooterComponent::new(theme());
        let mut buf = ScreenBuffer::new(100, 3);
        assert_eq!(comp.render(&mut buf, 0, 0, 100, 3), 0);
    }
}
