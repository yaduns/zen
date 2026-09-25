// REQ-OBS-01 (pi-327): prompt cache hit statistics—compare each turn with previous request to identify cache misses and account for waste.
// Corresponds to original TS: reference/pi-original/packages/coding-agent/src/core/cache-stats.ts.
use std::collections::HashMap;
use zen_ai::types::{AssistantMessage, Message, Model, Usage};
use zen_core::session::tree::SessionTreeEntry;

use crate::components::footer::format_tokens;

/// Prompt-cache TTL: only when idle time reaches this threshold is a miss attributed to 'cache expired'.
/// Aligns with pi `CACHE_TTL_MS` (Anthropic default cache TTL 5 minutes); only the label in `cache_miss_notice_text`
/// branch consumes it; no consumers outside module → private const (pi exports it for interactive label; in zen this logic is consolidated into this module).
const CACHE_TTL_MS: i64 = 5 * 60 * 1000;

/// A single-turn miss below this token count is treated as cache breakpoint granularity noise and not counted.
/// Aligns with pi `NOISE_FLOOR_TOKENS` (private).
const NOISE_FLOOR_TOKENS: u64 = 1024;

/// A counted (above noise floor) cache miss—displayable 'previous prompt content that should have been cache-read but was recomputed at full price this turn'
/// description. Interactive mode detects at message_end, re-injects by position on restore/rebuild, and `/session` command accumulates; pure runtime display data,
/// not serialized or persisted (notices are not persisted, recomputed on rebuild).
#[derive(Debug, Clone, PartialEq)]
pub struct CacheMiss {
    /// Extra tokens paid (`min(prev.prompt_tokens, current prompt_tokens) - cache_read`, always > NOISE_FLOOR_TOKENS).
    pub missed_tokens: u64,
    /// Extra amount paid ($) based on difference between paid price and cache price; when pricing unknown (cache price falls back to 0), equals full recomputation price.
    pub missed_cost: f64,
    /// Idle duration since previous request (ms, `max(0, current timestamp - previous timestamp)`; used by notice label to determine 'TTL expired').
    pub idle_ms: i64,
    /// Whether model switched (`provider/model` differs from previous request; used by notice label to determine 'model switch caused recomputation').
    pub model_changed: bool,
}

/// Session cumulative cache waste summary—`/session` info command shows 'how much extra was paid this session for cache misses'. Pure display data,
/// not serialized. Produced in one pass by `compute_cache_waste` scanning all session entries.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CacheWasteTotals {
    /// Total extra tokens paid.
    pub missed_tokens: u64,
    /// Total extra amount paid ($).
    pub missed_cost: f64,
    /// Number of misses above noise floor.
    pub miss_count: usize,
}

/// Model cache-read rate query—cache miss accounting (pricing fallback in detect/collect/compute functions) needs to know
/// 'cache-read unit price for a provider/model'; caller (interactive mode, passing current session model) provides this query;
/// Corresponds to pi `ModelPriceSource` (projection resolution of `getModel(p, m).cost.cacheRead`—Rust directly returns rate
/// $/million tokens, removing TS-side nesting of projecting `{cost:{cacheRead}}` from model object).
///
/// # Methods
/// - `cache_read_rate(provider, model_id)`: returns cache-read rate for the model ($/million tokens); returns `None` if unknown
///   (caller falls back to 0 per pi `?? 0` semantics—under v1 single provider, old models switched away from have no rate found, billing uses full price, correct).
///
/// # Consumers
/// `detect_cache_miss`/`collect_cache_misses`/`compute_cache_waste` (this module); in v1 implemented by
/// `impl CachePriceSource for Model`; future multi-provider (REQ-V2-02 model cataloging) can be replaced by registry implementation.
pub trait CachePriceSource {
    fn cache_read_rate(&self, provider: &str, model_id: &str) -> Option<f64>;
}

/// Current session model acts as pricing source—under v1 single provider, caller (interactive bridge/restore flow) directly passes `session.model()`:
/// returns `cost_rates.cache_read` only when queried provider+model matches the model itself; mismatch (old model switched away within session
/// model) returns None and falls back to full price, aligning with pi `ModelRuntime` behavior when model not found. v1 accepts deviation: pi's model registry
/// can resolve rates for old switched-away models, zen only knows current model—`missed_cost` will be overestimated for cross-model turns (billed as full recomputation).
impl CachePriceSource for Model {
    fn cache_read_rate(&self, provider: &str, model_id: &str) -> Option<f64> {
        if self.provider == provider && self.id == model_id {
            self.cost_rates.as_ref().map(|c| c.cache_read)
        } else {
            None
        }
    }
}

/// Scanner's sequential session state (private): 'prompt composition of previous request'. Corresponds to pi `PreviousRequest`.
struct PreviousRequest {
    /// Total tokens of previous prompt (input + cache_read + cache_write).
    prompt_tokens: u64,
    /// `provider/model` concatenated key, used for model-switch detection.
    model_key: String,
    /// Previous request timestamp (unix milliseconds).
    timestamp: i64,
    /// Sticky flag: whether any previous request in this scan segment reported cache activity—distinguishes 'total miss on cache-read-only provider' from
    /// miss' and 'provider that never reports cache' (aligns with pi; domain semantics, not a language limitation artifact).
    reported_cache: bool,
}

/// Total prompt tokens for a single request (input + cache_read + cache_write).
fn prompt_tokens(usage: &Usage) -> u64 {
    usage.input + usage.cache_read + usage.cache_write
}

/// Compute cache miss for an assistant message relative to previous request (aligns with pi `detectMiss`). `prev` is None (first turn/
/// after reset) → None; zero-cache turns are counted only if cache activity was reported earlier in scan segment; miss ≤ noise floor → None;
/// `paidTokens == 0` divide-by-zero guard retained (pi handles explicitly; formula derivation shows this branch unreachable when missed>floor—\
/// paid_tokens==0 ⇒ pt==cache_read ⇒ missed≤0, defensive code).
fn detect_miss(
    prev: Option<&PreviousRequest>,
    message: &AssistantMessage,
    prices: &impl CachePriceSource,
) -> Option<CacheMiss> {
    let usage = &message.usage;
    let pt = prompt_tokens(usage);
    let prev = prev?;
    // Zero-cache turns are counted only if cache was reported earlier (total miss on cache-read-only provider vs provider that never reports cache)
    if pt == 0 || (usage.cache_read + usage.cache_write == 0 && !prev.reported_cache) {
        return None;
    }

    let missed_tokens = prev.prompt_tokens.min(pt).saturating_sub(usage.cache_read);
    if missed_tokens <= NOISE_FLOOR_TOKENS {
        return None;
    }

    // Extra cost = missed tokens billed at actual paid price (input/cacheWrite includes write premium) rather than cache-read price.
    // Missed tokens can only fall into input or cacheWrite buckets; paid price is taken directly from this message's cost breakdown.
    let paid_tokens = usage.input + usage.cache_write;
    let paid_per_token = if paid_tokens > 0 {
        (usage.cost.input + usage.cost.cache_write) / paid_tokens as f64
    } else {
        0.0
    };
    let read_per_token = if usage.cache_read > 0 {
        usage.cost.cache_read / usage.cache_read as f64
    } else {
        prices
            .cache_read_rate(&message.provider, &message.model)
            .unwrap_or(0.0)
            / 1_000_000.0
    };

    Some(CacheMiss {
        missed_tokens,
        missed_cost: missed_tokens as f64 * (paid_per_token - read_per_token).max(0.0),
        idle_ms: (message.timestamp - prev.timestamp).max(0),
        model_changed: format!("{}/{}", message.provider, message.model) != prev.model_key,
    })
}

/// Convert an assistant message into previous-request state (aligns with pi `asPreviousRequest`); returns None when prompt is 0.
fn as_previous_request(
    message: &AssistantMessage,
    reported_cache: bool,
) -> Option<PreviousRequest> {
    let pt = prompt_tokens(&message.usage);
    if pt == 0 {
        return None;
    }
    Some(PreviousRequest {
        prompt_tokens: pt,
        model_key: format!("{}/{}", message.provider, message.model),
        timestamp: message.timestamp,
        reported_cache: reported_cache || message.usage.cache_read + message.usage.cache_write > 0,
    })
}

/// Scan output: `prev` (latest request state) + accumulated `totals` + `misses` keyed by entry id of paying message.
struct ScanResult {
    prev: Option<PreviousRequest>,
    totals: CacheWasteTotals,
    misses: HashMap<String, CacheMiss>,
}

/// Single-pass scan over session entries (aligns with pi `scan`): `Compaction`/`BranchSummary` entries reset `prev` (legitimate context
/// change; next turn is new content, not recomputation; model switch is **not** exempt and still counted); assistant messages run `detect_miss` one by one and
/// update `prev`. Misses are keyed by `SessionTreeEntry::Message.id` of paying message (pi keys by message object reference).
fn scan(entries: &[SessionTreeEntry], prices: &impl CachePriceSource) -> ScanResult {
    let mut prev: Option<PreviousRequest> = None;
    let mut totals = CacheWasteTotals::default();
    let mut misses = HashMap::new();
    for entry in entries {
        match entry {
            SessionTreeEntry::Compaction { .. } | SessionTreeEntry::BranchSummary { .. } => {
                prev = None;
                continue;
            }
            SessionTreeEntry::Message {
                id,
                message: Message::Assistant(a),
                ..
            } => {
                if let Some(miss) = detect_miss(prev.as_ref(), a, prices) {
                    totals.missed_tokens += miss.missed_tokens;
                    totals.missed_cost += miss.missed_cost;
                    totals.miss_count += 1;
                    misses.insert(id.clone(), miss);
                }
                let reported = prev.as_ref().is_some_and(|p| p.reported_cache);
                if let Some(new_prev) = as_previous_request(a, reported) {
                    prev = Some(new_prev);
                }
            }
            _ => {}
        }
    }
    ScanResult {
        prev,
        totals,
        misses,
    }
}

/// When a session turn's messages finish, interactive mode determines 'whether this just-completed assistant response is a prompt cache [miss] worth notifying
/// miss'—when user has `show_cache_miss_notices` enabled, this recomputation notice about paying more than previous turn should appear in chat stream
/// (Aligned with pi `detectCacheMiss` + `maybeShowCacheMissNotice`): take the "previous request" prompt from the session entries
/// state and compare it with the current message; only when the miss count (the part of the previous prompt not served from cache this time) exceeds the 1024-token noise floor
/// is it counted as one miss.
///
/// # Parameters
/// - `entries`: session entries (branch view is fine; the function only scans read-only).
/// - `message`: the just-completed assistant message.
/// - `prices`: pricing lookup (v1 passes `&session.model()`).
///
/// # Hard contract (load-bearing wall; do not weaken)
/// **Tail removal**: the AgentSession persistence subscription is registered before the interactive bridge, so by the time `MessageEnd` reaches the bridge the message has already been appended to entries
/// (the opposite ordering from pi's "entries must not contain message") — this function **must** first remove the trailing Message entry equal to `message`
/// (`PartialEq`, already derived on `AssistantMessage`) before scanning; if the removal fails, the scan will treat the message
/// itself as `prev` (`prev = as_previous_request(message)`), yielding `missed_tokens = prompt_tokens -
/// cache_read = input + cache_write`, and >1024 means a false-positive miss. `PartialEq` is safe here as a semantic boundary: the persistence
/// subscription and the bridge receive clones of the **same event payload** (value equality means same identity), not general "semantic equality" — this
/// contract must not be generalized to "remove any entry whose value equals message" (behavior would be undefined if a future historical message had an equal value but a different identity)
/// undefined).
///
/// # Returns
/// `Some(CacheMiss)` = this is a significant miss (above the noise floor); `None` = first turn / after reset / the provider has never reported
/// cache activity / the miss is below the noise floor (aligned with pi `detectMiss`'s undefined return conditions).
///
/// # Bounds
/// - `Compaction`/`BranchSummary` entries reset `prev` (legitimate context change; model switches are **not** exempt and still count);
/// - a "zero-cache turn" counts only if cache activity was previously reported within the scanned segment (a complete miss on a cache-read-only provider vs
///   a provider that never reports cache).
///
/// # Consumers
/// interactive/mod.rs bridge (`AgentEvent::MessageEnd` assistant branch, wired up in this task).
pub fn detect_cache_miss(
    entries: &[SessionTreeEntry],
    message: &AssistantMessage,
    prices: &impl CachePriceSource,
) -> Option<CacheMiss> {
    // Tail removal (hard contract, see doc comment): when the bridge receives MessageEnd the message is already the last entry (the persistence subscription runs first).
    let entries = match entries.last() {
        Some(SessionTreeEntry::Message {
            message: Message::Assistant(last),
            ..
        }) if last == message => &entries[..entries.len() - 1],
        _ => entries,
    };
    detect_miss(scan(entries, prices).prev.as_ref(), message, prices)
}

/// When a session is restored from JSONL / the chat is rebuilt after compaction, interactive mode re-inserts previously occurred cache-miss notices back into the transcript
/// (notices are not persisted, recomputed on demand, aligned with pi `collectCacheMisses` + `renderSessionItems` re-injection): scan all
/// entries and return the association between each significant miss and "the session entry id of the assistant message that paid for it".
///
/// # Parameters
/// - `entries`: session entries (the restore path passes the branch view `branch_entries()` — including compacted-away historical ancestors, with an intact `prev` chain;
///   the branch is the context lineage actually sent to the LLM, which computes turn-by-turn comparison more correctly than the full `entries()` (including fork branches);
///   a deliberate deviation from pi passing the full `getEntries()`; the two are equivalent in a single-branch session).
/// - `prices`: pricing query.
///
/// # Returns
/// `HashMap<String, CacheMiss>`, key = the `SessionTreeEntry::Message.id` of the paying message (pi keys by message **object
/// reference**; under Rust value semantics the entry id is a stable string key); consumers align and insert notices by entry id when restoring/rebuilding.
///
/// # Consumers
/// interactive/mod.rs restore path (two RestoreHistory constructions: init and session switch; wired up in this task).
pub fn collect_cache_misses(
    entries: &[SessionTreeEntry],
    prices: &impl CachePriceSource,
) -> HashMap<String, CacheMiss> {
    scan(entries, prices).misses
}

/// When the `/session` info command shows "how much extra this session paid for cache misses", accumulate the tokens and
/// cost of all significant misses (aligned with pi `computeCacheWaste` + the cacheWaste section of the `/session` command).
///
/// # Parameters
/// - `entries`: session entries (aligned with pi's full `sessionManager.getEntries()` semantics; v1 implementers choose either
///   `branch_entries()` or `entries()`, see the parameter docs of `collect_cache_misses` in this module).
/// - `prices`: pricing query.
///
/// # Returns
/// `CacheWasteTotals` (overpaid tokens / overpaid cost / hit count; all zero when there are no significant misses).
///
/// # Consumers
/// **pub but currently has no consumer** — reserved for <unassigned /session command> (the /session execution section is marked "deferred" and unassigned in the T-AP-03-2 disposition table
/// ; listed as pending assignment alongside the same precedent as `get_usage_cost_breakdown` (T-AP-06-4 open question 1)).
pub fn compute_cache_waste(
    entries: &[SessionTreeEntry],
    prices: &impl CachePriceSource,
) -> CacheWasteTotals {
    scan(entries, prices).totals
}

/// Interactive mode turns a detected miss into a chat notice line (aligned with pi `addCacheMissNotice`'s threshold + label + text
/// section, consolidated into a module pure function shared by the bridge and restore path): a miss too small (<20k tokens and <$0.1) returns `None` without disturbing;
/// otherwise produce a plain-text notice (no ANSI — coloring is done directly by the consumer via the `ChatRow::Info { kind: Warning }` semantic color).
///
/// # Parameters
/// - `miss`: output of `detect_cache_miss`/`collect_cache_misses`.
///
/// # Returns
/// `Some(text)` looks like `Cache miss: 12k tokens re-billed (~$0.03)` — the label depends on `model_changed`
/// （`Cache miss after model switch`）/ `idle_ms >= CACHE_TTL_MS`（`Cache miss after Nm idle`）/
/// with three states, default `Cache miss`; append ` (~$X.XX)` when the cost is ≥$0.01 (not appended below $0.01); the token count goes through `format_tokens`
/// (components/footer.rs) for compact formatting; `None` = below the display threshold (20k/0.1).
///
/// # Display-layer deviation log
/// `format_tokens` differs from pi `formatTokens` (footer.ts:24-30) only in the ≥10M branch: pi uses `Math.round`
/// (12.5M → "13M"), while zen truncates with integer division (12.5M → "12M"); the other branches match branch by branch. Miss sizes are usually <1M,
/// so there is no practical impact — reuse the existing function rather than inlining pi semantics for this.
///
/// # Consumers
/// interactive/mod.rs bridge (MessageEnd notice) and restore path (notice re-injection), wired up in this task.
pub fn cache_miss_notice_text(miss: &CacheMiss) -> Option<String> {
    if miss.missed_tokens < 20_000 && miss.missed_cost < 0.1 {
        return None;
    }
    let cost = if miss.missed_cost >= 0.01 {
        format!(" (~${:.2})", miss.missed_cost)
    } else {
        String::new()
    };
    let re_billed = format!(
        "{} tokens re-billed{}",
        format_tokens(miss.missed_tokens),
        cost
    );
    let label = if miss.model_changed {
        "Cache miss after model switch".to_string()
    } else if miss.idle_ms >= CACHE_TTL_MS {
        // Math.round(idleMs / 60_000) semantics (idle_ms ≥ 0 and ≥ CACHE_TTL_MS, round half up)
        format!(
            "Cache miss after {}m idle",
            (miss.idle_ms + 30_000) / 60_000
        )
    } else {
        "Cache miss".to_string()
    };
    Some(format!("{label}: {re_billed}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use zen_ai::types::{ContentBlock, Cost, StopReason, TextContent};

    /// Test pricing source: constant cache-read rate ($/million tokens), aligned with pi's test `models`.
    struct FixedPriceSource(f64);
    impl CachePriceSource for FixedPriceSource {
        fn cache_read_rate(&self, _provider: &str, _model_id: &str) -> Option<f64> {
            Some(self.0)
        }
    }
    const PRICES: FixedPriceSource = FixedPriceSource(0.3);

    fn zero_cost() -> Cost {
        Cost::default()
    }

    /// Build an assistant message (provider always "test").
    fn assistant(
        input: u64,
        cache_read: u64,
        cache_write: u64,
        cost: Cost,
        model: &str,
        timestamp: i64,
    ) -> AssistantMessage {
        AssistantMessage {
            content: vec![ContentBlock::Text(TextContent {
                text: String::new(),
                text_signature: None,
            })],
            api: "openai-completions".into(),
            provider: "test".into(),
            model: model.into(),
            response_model: None,
            response_id: None,
            usage: Usage {
                input,
                output: 10,
                cache_read,
                cache_write,
                cache_write_1h: None,
                reasoning: None,
                total_tokens: input + cache_read + cache_write,
                cost,
            },
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp,
        }
    }

    /// Wrap an assistant message into a Message entry.
    fn entry(id: &str, msg: AssistantMessage) -> SessionTreeEntry {
        SessionTreeEntry::Message {
            id: id.into(),
            parent_id: None,
            timestamp: String::new(),
            message: Message::Assistant(msg),
        }
    }

    /// Turn 1: fresh 100k cache write ($3.75/M).
    fn turn1() -> AssistantMessage {
        assistant(
            0,
            0,
            100_000,
            Cost {
                cache_write: 0.375,
                ..zero_cost()
            },
            "test-model",
            0,
        )
    }
    /// Turn 2: healthy — all read back from cache at $0.30/M.
    fn turn2() -> AssistantMessage {
        assistant(
            0,
            100_000,
            5_000,
            Cost {
                cache_read: 0.03,
                cache_write: 0.019,
                ..zero_cost()
            },
            "test-model",
            60_000,
        )
    }

    // ── compute_cache_waste ─────────────────────────────────────

    #[test]
    fn waste_accumulates_missed_tokens_and_cost_across_turns() {
        // Turn 3: complete miss; the previous turn's 105k prompt is re-billed at the $3.75/M write price
        let turn3 = assistant(
            0,
            0,
            110_000,
            Cost {
                cache_write: 0.4125,
                ..zero_cost()
            },
            "test-model",
            120_000,
        );
        let totals = compute_cache_waste(
            &[
                entry("t1", turn1()),
                entry("t2", turn2()),
                entry("t3", turn3),
            ],
            &PRICES,
        );
        assert_eq!(totals.missed_tokens, 105_000);
        // 105k × ($3.75 - $0.30)/M
        assert!((totals.missed_cost - 0.36225).abs() < 1e-5);
        assert_eq!(totals.miss_count, 1);
    }

    #[test]
    fn waste_counts_nothing_for_healthy_sessions() {
        let totals = compute_cache_waste(&[entry("t1", turn1()), entry("t2", turn2())], &PRICES);
        assert_eq!(totals.missed_tokens, 0);
        assert_eq!(totals.missed_cost, 0.0);
        assert_eq!(totals.miss_count, 0);
    }

    #[test]
    fn waste_skips_the_turn_after_a_compaction_reset() {
        let reset = SessionTreeEntry::Compaction {
            id: "c".into(),
            parent_id: None,
            timestamp: String::new(),
            summary: "sum".into(),
            first_kept_entry_id: None,
            tokens_before: 0,
            retained_tail: None,
            details: None,
            usage: None,
            from_hook: None,
        };
        let after = assistant(
            0,
            0,
            20_000,
            Cost {
                cache_write: 0.075,
                ..zero_cost()
            },
            "test-model",
            180_000,
        );
        let totals = compute_cache_waste(
            &[entry("t1", turn1()), reset, entry("after", after)],
            &PRICES,
        );
        assert_eq!(totals.missed_tokens, 0);
    }

    #[test]
    fn waste_counts_misses_caused_by_model_switches() {
        let other = assistant(
            0,
            0,
            100_000,
            Cost {
                cache_write: 0.375,
                ..zero_cost()
            },
            "other-model",
            120_000,
        );
        let totals = compute_cache_waste(&[entry("t1", turn1()), entry("other", other)], &PRICES);
        assert_eq!(totals.missed_tokens, 100_000);
        assert_eq!(totals.miss_count, 1);
    }

    #[test]
    fn waste_skips_providers_that_report_no_cache_activity() {
        let a = assistant(100_000, 0, 0, zero_cost(), "test-model", 0);
        let b = assistant(110_000, 0, 0, zero_cost(), "test-model", 60_000);
        let totals = compute_cache_waste(&[entry("a", a), entry("b", b)], &PRICES);
        assert_eq!(totals.missed_tokens, 0);
        assert_eq!(totals.miss_count, 0);
    }

    #[test]
    fn waste_handles_zero_cost_usage_without_panicking() {
        // all-zero cost + real miss (read_per_token = 0/0 does not panic when cache_read > 0)
        let a = assistant(0, 0, 100_000, zero_cost(), "test-model", 0);
        let b = assistant(0, 100_000, 5_000, zero_cost(), "test-model", 60_000);
        let totals = compute_cache_waste(&[entry("a", a), entry("b", b)], &PRICES);
        assert_eq!(totals.missed_tokens, 0); // healthy turn, no miss
    }

    // ── collect_cache_misses ─────────────────────────────────────

    #[test]
    fn collect_maps_counted_misses_to_entry_ids() {
        let miss = assistant(
            0,
            0,
            110_000,
            Cost {
                cache_write: 0.4125,
                ..zero_cost()
            },
            "test-model",
            120_000,
        );
        let misses = collect_cache_misses(
            &[entry("t1", turn1()), entry("t2", turn2()), entry("m", miss)],
            &PRICES,
        );
        assert_eq!(misses.len(), 1);
        let m = misses.get("m").expect("miss keyed by payment entry id");
        assert_eq!(m.missed_tokens, 105_000);
    }

    #[test]
    fn collect_returns_empty_when_no_misses() {
        let misses = collect_cache_misses(&[entry("t1", turn1()), entry("t2", turn2())], &PRICES);
        assert!(misses.is_empty());
    }

    // ── detect_cache_miss ────────────────────────────────────────

    #[test]
    fn detect_miss_on_just_completed_message_with_idle_time() {
        let miss_msg = assistant(
            0,
            0,
            110_000,
            Cost {
                cache_write: 0.4125,
                ..zero_cost()
            },
            "test-model",
            600_000,
        );
        let miss = detect_cache_miss(
            &[entry("t1", turn1()), entry("t2", turn2())],
            &miss_msg,
            &PRICES,
        );
        let miss = miss.expect("Should detect a miss");
        assert_eq!(miss.missed_tokens, 105_000);
        assert!((miss.missed_cost - 0.36225).abs() < 1e-5);
        // 600s - 60s = 540s
        assert_eq!(miss.idle_ms, 540_000);
        assert!(!miss.model_changed);
    }

    #[test]
    fn detect_flags_model_switches_on_misses() {
        let other = assistant(
            0,
            0,
            110_000,
            Cost {
                cache_write: 0.4125,
                ..zero_cost()
            },
            "other-model",
            120_000,
        );
        let miss = detect_cache_miss(
            &[entry("t1", turn1()), entry("t2", turn2())],
            &other,
            &PRICES,
        )
        .expect("Should detect a miss");
        assert_eq!(miss.missed_tokens, 105_000);
        assert!(miss.model_changed);
    }

    #[test]
    fn detect_returns_none_for_healthy_turns() {
        let healthy = assistant(
            0,
            105_000,
            2_000,
            Cost {
                cache_read: 0.0315,
                cache_write: 0.0075,
                ..zero_cost()
            },
            "test-model",
            120_000,
        );
        assert!(detect_cache_miss(
            &[entry("t1", turn1()), entry("t2", turn2())],
            &healthy,
            &PRICES
        )
        .is_none());
    }

    #[test]
    fn detect_returns_none_for_the_first_turn_of_a_session() {
        assert!(detect_cache_miss(&[], &turn1(), &PRICES).is_none());
    }

    #[test]
    fn detect_returns_none_below_noise_floor() {
        // previous prompt 2_000, this turn cache_read 1_500 → miss 500 ≤ 1024
        let prev = assistant(
            0,
            0,
            2_000,
            Cost {
                cache_write: 0.0075,
                ..zero_cost()
            },
            "test-model",
            0,
        );
        let cur = assistant(
            0,
            1_500,
            500,
            Cost {
                cache_read: 0.00045,
                cache_write: 0.001875,
                ..zero_cost()
            },
            "test-model",
            60_000,
        );
        assert!(detect_cache_miss(&[entry("p", prev)], &cur, &PRICES).is_none());
    }

    /// Tail-exclusion hard contract: when entries already contain message itself (zen subscription order—persistence before bridge), must not treat message
    /// as its own prev (otherwise false positive 110_000); should correctly exclude the last entry to get 105_000.
    #[test]
    fn detect_excludes_trailing_self_entry() {
        let miss_msg = assistant(
            0,
            0,
            110_000,
            Cost {
                cache_write: 0.4125,
                ..zero_cost()
            },
            "test-model",
            600_000,
        );
        // message has been appended to entries as the last item (matching the real timing when MessageEnd reaches the bridge)
        let miss = detect_cache_miss(
            &[
                entry("t1", turn1()),
                entry("t2", turn2()),
                entry("self", miss_msg.clone()),
            ],
            &miss_msg,
            &PRICES,
        )
        .expect("Should detect a miss");
        assert_eq!(
            miss.missed_tokens, 105_000,
            "After tail removal prev = turn2, miss 105k"
        );
    }

    #[test]
    fn detect_ignores_non_assistant_trailing_entry() {
        // last entry is not assistant (e.g. defensive case): do not exclude, scan normally
        let miss_msg = assistant(
            0,
            0,
            110_000,
            Cost {
                cache_write: 0.4125,
                ..zero_cost()
            },
            "test-model",
            600_000,
        );
        let trailing_user = SessionTreeEntry::Message {
            id: "u".into(),
            parent_id: None,
            timestamp: String::new(),
            message: Message::User(zen_ai::types::UserMessage {
                content: zen_ai::types::ContentBlockList::Text("hi".into()),
                timestamp: 610_000,
            }),
        };
        let miss = detect_cache_miss(
            &[entry("t1", turn1()), entry("t2", turn2()), trailing_user],
            &miss_msg,
            &PRICES,
        );
        assert_eq!(miss.map(|m| m.missed_tokens), Some(105_000));
    }

    // ── CachePriceSource for Model ───────────────────────────────

    /// Construct a Model with provider/id/cost_rates (matching the runtime.rs test fixture shape).
    fn price_model(
        provider: &str,
        id: &str,
        cost_rates: Option<zen_ai::types::CostRates>,
    ) -> Model {
        Model {
            id: id.into(),
            name: None,
            provider: provider.into(),
            base_url: "http://localhost".into(),
            context_window: 8192,
            max_tokens: 1024,
            reasoning: false,
            headers: None,
            cost_rates,
            api: zen_ai::provider::Api::OpenAiCompletions,
        }
    }

    #[test]
    fn model_price_source_matches_provider_and_id() {
        let model = price_model(
            "test",
            "test-model",
            Some(zen_ai::types::CostRates {
                input: 3.75,
                output: 1.1,
                cache_read: 0.3,
                cache_write: 3.75,
            }),
        );
        assert_eq!(model.cache_read_rate("test", "test-model"), Some(0.3));
        // no match (old model) → None
        assert_eq!(model.cache_read_rate("test", "other-model"), None);
        assert_eq!(model.cache_read_rate("other-provider", "test-model"), None);
    }

    #[test]
    fn model_price_source_none_without_cost_rates() {
        let model = price_model("test", "test-model", None);
        assert_eq!(model.cache_read_rate("test", "test-model"), None);
    }

    // ── cache_miss_notice_text ───────────────────────────────────

    fn miss(missed_tokens: u64, missed_cost: f64, idle_ms: i64, model_changed: bool) -> CacheMiss {
        CacheMiss {
            missed_tokens,
            missed_cost,
            idle_ms,
            model_changed,
        }
    }

    #[test]
    fn notice_none_below_display_threshold() {
        // <20k tokens and <$0.1
        assert!(cache_miss_notice_text(&miss(15_000, 0.05, 0, false)).is_none());
    }

    #[test]
    fn notice_some_when_either_threshold_exceeded() {
        // show if any dimension exceeds threshold
        assert!(cache_miss_notice_text(&miss(25_000, 0.05, 0, false)).is_some());
        assert!(cache_miss_notice_text(&miss(15_000, 0.5, 0, false)).is_some());
    }

    #[test]
    fn notice_default_label_with_cost() {
        let text = cache_miss_notice_text(&miss(105_000, 0.36225, 60_000, false)).expect("Should display");
        assert_eq!(text, "Cache miss: 105k tokens re-billed (~$0.36)");
    }

    #[test]
    fn notice_model_switch_label() {
        let text = cache_miss_notice_text(&miss(105_000, 0.36225, 60_000, true)).expect("Should display");
        assert!(
            text.starts_with("Cache miss after model switch: "),
            "{text}"
        );
    }

    #[test]
    fn notice_idle_label_at_and_above_ttl_boundary() {
        // idle_ms == CACHE_TTL_MS (300_000 = 5m) triggers idle label
        let text =
            cache_miss_notice_text(&miss(105_000, 0.36225, CACHE_TTL_MS, false)).expect("Should display");
        assert_eq!(
            text,
            "Cache miss after 5m idle: 105k tokens re-billed (~$0.36)"
        );
        // 6.5 minutes → Math.round → 7m
        let text = cache_miss_notice_text(&miss(105_000, 0.36225, 390_000, false)).expect("Should display");
        assert_eq!(
            text,
            "Cache miss after 7m idle: 105k tokens re-billed (~$0.36)"
        );
    }

    #[test]
    fn notice_omits_cost_below_one_cent() {
        let text = cache_miss_notice_text(&miss(25_000, 0.005, 0, false)).expect("Should display");
        assert_eq!(text, "Cache miss: 25k tokens re-billed");
    }
}
