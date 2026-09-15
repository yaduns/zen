// REQ-APP-06 (pi-324): session-level usage statistics — cumulative tokens/cost + per-model attribution breakdown.
// Corresponds to original TS: reference/pi-original/packages/coding-agent/src/core/usage-totals.ts.
// `UsageTotals`/`add_usage_to_totals` moved in from the fallback definitions in components/footer.rs (T-AP-11-2
// registered migration seam, closed by T-AP-06-4); pi `createUsageTotals` factory is eliminated by `Default` derive.
// In the fallback definitions, `last_input` (latest round's input, overwritten each round) and the other five fields' session-monotonic accumulation
// have inconsistent lifetimes; split out in T-AP-06-4 — carried independently by the consumer side (interactive/mod.rs `UsageTurnState`).

use std::collections::BTreeMap;

use zen_ai::types::Usage;
use zen_core::session::tree::SessionTreeEntry;

/// Session-level token/cost accumulator — the shared "how much has this session spent in total" for the footer stats line and /session info command
/// Single source of truth (pi `UsageTotals`, with all five fields aligned item by item). Fields accumulate monotonically via `add_usage_to_totals`,
/// only increase, never decrease; pure runtime display/statistics data, not serialized or persisted to disk.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct UsageTotals {
    /// Cumulative input tokens.
    pub input: u64,
    /// Cumulative output tokens.
    pub output: u64,
    /// Cumulative cache-hit read tokens.
    pub cache_read: u64,
    /// Cumulative cache-write tokens.
    pub cache_write: u64,
    /// Cumulative cost (`usage.cost.total` semantics, accumulated directly as f64 — aligned with pi, display layer uses to_fixed(3)).
    pub cost: f64,
}

/// The footer stats line and /session command feed usage from "a single LLM request/tool summary/compaction summary" into the session
/// ledger posting action — one entry per usage, cumulative fields only increase (pi `addUsageToTotals`).
///
/// Behavior: accumulate the five fields item by item; cost takes `usage.cost.total` (total cost including cache read/write price differences).
/// Parameter `usage` is zen-ai usage for a single request (borrowed to avoid clone).
pub fn add_usage_to_totals(totals: &mut UsageTotals, usage: &Usage) {
    totals.input += usage.input;
    totals.output += usage.output;
    totals.cache_read += usage.cache_read;
    totals.cache_write += usage.cache_write;
    totals.cost += usage.cost.total;
}

/// Single per-model cost-attribution breakdown line — each `key: $x (n tokens)` entry in the /session command's "Cost" section
/// (pi `UsageCostBreakdownEntry`). Pure display data, not serialized.
#[derive(Debug, Clone, PartialEq)]
pub struct UsageCostBreakdownEntry {
    /// Attribution bucket key: assistant is `{provider}/{response_model ?? model}`; tool and summary types
    /// uniformly use `"Tools/summaries"` (preserve pi's original literal).
    pub key: String,
    /// Cumulative cost for this bucket.
    pub cost: f64,
    /// Sum of this bucket's input+output+cache_read+cache_write.
    pub tokens: u64,
}

/// Per-model attribution breakdown for the /session command's "Cost" section: bucket usage from session entries by "who spent the money",
/// so users can see where cost went with multiple models/sources (pi `getUsageCostBreakdown`).
///
/// Bucketing rules (aligned with pi):
/// - assistant message → key = `{provider}/{response_model preferred, fallback model}` (response-model attribution,
///   in routing scenarios such as OpenRouter auto, it lands on the actual model);
/// - toolResult messages with usage, and branch_summary/compaction entries with usage →
///   key = `"Tools/summaries"` (usage not directly attributable to a model is merged separately, ensuring bucket totals match the session ledger).
///
/// Output: filter out all-zero buckets (cost <= 0 and tokens == 0), then sort by cost descending (f64 via `partial_cmp`,
/// NaN treated as equal). Internal bucketing uses `BTreeMap`: unlike pi's insertion-order Map, ties with equal cost are ordered by key
/// lexicographically for stable output — this determinism difference is a deliberate zen choice (HashMap nondeterminism would introduce irreproducible ordering), already
/// registered as a deliberate difference in the task docs.
///
/// The caller decides whether parameter `entries` passes the full set (`SessionManager::get_entries()`, aligned with pi
/// getSessionStats semantics: including compacted-away history) or the branch view (`SessionStore::branch_entries()`).
///
/// Current consumers: none (pub but currently unconsumed; reserved for the unassigned /session command Cost section —
/// the /session execution section is recorded as "deferred" and unassigned in the T-AP-03-2 disposition table, see tasks/T-AP-06-4-implemented.md open issue 1).
pub fn get_usage_cost_breakdown(entries: &[SessionTreeEntry]) -> Vec<UsageCostBreakdownEntry> {
    const TOOLS_SUMMARIES: &str = "Tools/summaries";
    let mut buckets: BTreeMap<String, UsageTotals> = BTreeMap::new();
    for entry in entries {
        let (key, usage) = match entry {
            SessionTreeEntry::Message {
                message: zen_ai::types::Message::Assistant(a),
                ..
            } => (
                format!(
                    "{}/{}",
                    a.provider,
                    a.response_model.as_deref().unwrap_or(&a.model)
                ),
                &a.usage,
            ),
            SessionTreeEntry::Message {
                message: zen_ai::types::Message::ToolResult(t),
                ..
            } => match &t.usage {
                Some(u) => (TOOLS_SUMMARIES.to_string(), u),
                None => continue,
            },
            SessionTreeEntry::BranchSummary { usage: Some(u), .. }
            | SessionTreeEntry::Compaction { usage: Some(u), .. } => {
                (TOOLS_SUMMARIES.to_string(), u)
            }
            _ => continue,
        };
        add_usage_to_totals(buckets.entry(key).or_default(), usage);
    }
    let mut out: Vec<UsageCostBreakdownEntry> = buckets
        .into_iter()
        .map(|(key, totals)| UsageCostBreakdownEntry {
            tokens: totals.input + totals.output + totals.cache_read + totals.cache_write,
            cost: totals.cost,
            key,
        })
        .filter(|e| e.cost > 0.0 || e.tokens > 0)
        .collect();
    out.sort_by(|a, b| {
        b.cost
            .partial_cmp(&a.cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use zen_ai::types::{
        AssistantMessage, ContentBlock, Cost, Message, StopReason, TextContent, ToolResultMessage,
    };

    fn usage(i: u64, o: u64, cr: u64, cw: u64, cost_total: f64) -> Usage {
        Usage {
            input: i,
            output: o,
            cache_read: cr,
            cache_write: cw,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: i + o + cr + cw,
            cost: Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: cost_total,
            },
        }
    }

    fn assistant(provider: &str, model: &str, response_model: Option<&str>, u: Usage) -> Message {
        Message::Assistant(AssistantMessage {
            content: vec![ContentBlock::Text(TextContent {
                text: String::new(),
                text_signature: None,
            })],
            api: "openai-completions".into(),
            provider: provider.into(),
            model: model.into(),
            response_model: response_model.map(Into::into),
            response_id: None,
            usage: u,
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 0,
        })
    }

    fn tool_result(u: Option<Usage>) -> Message {
        Message::ToolResult(ToolResultMessage {
            tool_call_id: "t1".into(),
            tool_name: "bash".into(),
            content: vec![],
            details: None,
            usage: u,
            is_error: false,
            added_tool_names: None,
            timestamp: 0,
        })
    }

    fn msg_entry(message: Message) -> SessionTreeEntry {
        SessionTreeEntry::Message {
            id: "e1".into(),
            parent_id: None,
            timestamp: "t".into(),
            message,
        }
    }

    fn compaction_entry(u: Option<Usage>) -> SessionTreeEntry {
        SessionTreeEntry::Compaction {
            id: "c1".into(),
            parent_id: None,
            timestamp: "t".into(),
            summary: "s".into(),
            first_kept_entry_id: None,
            tokens_before: 0,
            retained_tail: None,
            details: None,
            usage: u,
            from_hook: None,
        }
    }

    fn branch_summary_entry(u: Option<Usage>) -> SessionTreeEntry {
        SessionTreeEntry::BranchSummary {
            id: "b1".into(),
            parent_id: None,
            timestamp: "t".into(),
            from_id: "from".into(),
            summary: "s".into(),
            details: None,
            usage: u,
            from_hook: None,
        }
    }

    // ── UsageTotals / add_usage_to_totals ────────────────────────

    #[test]
    fn default_totals_all_zero() {
        let t = UsageTotals::default();
        assert_eq!(
            t,
            UsageTotals {
                input: 0,
                output: 0,
                cache_read: 0,
                cache_write: 0,
                cost: 0.0
            }
        );
    }

    #[test]
    fn add_usage_accumulates_all_fields() {
        let mut totals = UsageTotals::default();
        add_usage_to_totals(&mut totals, &usage(10, 5, 80, 2, 0.5));
        add_usage_to_totals(&mut totals, &usage(1, 2, 3, 4, 0.25));
        assert_eq!(totals.input, 11);
        assert_eq!(totals.output, 7);
        assert_eq!(totals.cache_read, 83);
        assert_eq!(totals.cache_write, 6);
        assert!((totals.cost - 0.75).abs() < 1e-12);
    }

    #[test]
    fn totals_is_copy_and_comparable() {
        let mut a = UsageTotals::default();
        add_usage_to_totals(&mut a, &usage(1, 1, 0, 0, 0.0));
        let b = a;
        assert_eq!(a, b);
    }

    // ── get_usage_cost_breakdown ─────────────────────────────────

    #[test]
    fn breakdown_keys_assistant_by_response_model_fallback() {
        let entries = vec![
            msg_entry(assistant(
                "p",
                "requested",
                Some("actual"),
                usage(10, 0, 0, 0, 0.1),
            )),
            msg_entry(assistant("p", "plain", None, usage(20, 0, 0, 0, 0.2))),
        ];
        let out = get_usage_cost_breakdown(&entries);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].key, "p/plain"); // cost 0.2 > 0.1, descending puts it first
        assert_eq!(out[1].key, "p/actual"); // response_model takes precedence over model
    }

    #[test]
    fn breakdown_groups_tools_and_summaries_into_one_bucket() {
        let entries = vec![
            msg_entry(tool_result(Some(usage(5, 0, 0, 0, 0.01)))),
            compaction_entry(Some(usage(100, 0, 0, 0, 0.02))),
            branch_summary_entry(Some(usage(50, 0, 0, 0, 0.03))),
        ];
        let out = get_usage_cost_breakdown(&entries);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].key, "Tools/summaries");
        assert!((out[0].cost - 0.06).abs() < 1e-12);
        assert_eq!(out[0].tokens, 155);
    }

    #[test]
    fn breakdown_skips_tool_result_without_usage() {
        let entries = vec![msg_entry(tool_result(None))];
        assert!(get_usage_cost_breakdown(&entries).is_empty());
    }

    #[test]
    fn breakdown_filters_zero_buckets() {
        // assistant has tokens but no cost → keep; compaction all zero → drop
        let entries = vec![
            msg_entry(assistant("p", "m", None, usage(10, 5, 0, 0, 0.0))),
            compaction_entry(Some(usage(0, 0, 0, 0, 0.0))),
        ];
        let out = get_usage_cost_breakdown(&entries);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].key, "p/m");
        assert_eq!(out[0].tokens, 15);
    }

    #[test]
    fn breakdown_empty_entries_yield_empty_vec() {
        assert!(get_usage_cost_breakdown(&[]).is_empty());
    }

    #[test]
    fn breakdown_ignores_non_usage_entry_kinds() {
        // user messages and other entry types produce no bucket
        let user = Message::User(zen_ai::types::UserMessage {
            content: zen_ai::types::ContentBlockList::Text("hi".into()),
            timestamp: 0,
        });
        let entries = vec![msg_entry(user)];
        assert!(get_usage_cost_breakdown(&entries).is_empty());
    }

    #[test]
    fn breakdown_sorts_by_cost_desc_stably() {
        // three buckets with different costs + equal-cost tie (BTreeMap lexicographic order + stable sort → deterministic order)
        let entries = vec![
            msg_entry(assistant("p", "mid", None, usage(1, 0, 0, 0, 0.5))),
            msg_entry(assistant("p", "cheap", None, usage(1, 0, 0, 0, 0.1))),
            msg_entry(assistant("p", "rich", None, usage(1, 0, 0, 0, 0.9))),
            msg_entry(assistant("a", "tie", None, usage(1, 0, 0, 0, 0.5))),
        ];
        let out = get_usage_cost_breakdown(&entries);
        let keys: Vec<&str> = out.iter().map(|e| e.key.as_str()).collect();
        assert_eq!(keys, vec!["p/rich", "a/tie", "p/mid", "p/cheap"]);
    }
}
