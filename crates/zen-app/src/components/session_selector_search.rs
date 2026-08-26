// REQ-APP-11 (pi-358): session-selector-search — pure functions for session search parsing/matching/sorting.
// Corresponds to original TS: reference/pi-original/packages/coding-agent/src/modes/interactive/components/session-selector-search.ts.
// fuzzy matching delegates to zen_tui::fuzzy::fuzzy_match (T-TU-03-4, Option<f64> semantics: hit is Some, lower is better).

use crate::session::session_listing::SessionListing;
use regex::Regex;
use zen_tui::fuzzy::fuzzy_match;

/// Session list sort mode (pi SortMode). Plain discriminant enum, not serialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortMode {
    /// Threaded order (input order; with search, sort by relevance score, matching pi's non-recent branch).
    Threaded,
    /// Most recently modified descending; when searching, only filter and keep original order.
    Recent,
    /// Relevance score ascending (fuzzy/phrase score); ties by modified descending.
    Relevance,
}

/// Name filter (pi NameFilter). Plain discriminant enum, not serialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NameFilter {
    /// All sessions.
    All,
    /// Named sessions only.
    Named,
}

/// Parsed search query result (zen form of pi ParsedSearchQuery: empty / tokenized / regex / invalid).
/// Not a tagged union, not serialized.
#[derive(Debug)]
pub enum SearchQuery {
    /// Empty query (no filtering).
    Empty,
    /// Tokenized mode (fuzzy + phrase tokens).
    Tokens(Vec<SearchToken>),
    /// Regex mode (`re:` prefix, case-insensitive).
    Regex(Regex),
    /// Parse failure (invalid regex, etc.) — query matches no sessions.
    Invalid(String),
}

/// Token kind (pi token.kind). Plain discriminant enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchTokenKind {
    /// Fuzzy match individually.
    Fuzzy,
    /// Whitespace-normalized contains match (quoted phrase).
    Phrase,
}

/// Search token.
#[derive(Debug)]
pub struct SearchToken {
    /// Token type.
    pub kind: SearchTokenKind,
    /// Token raw text (matching key).
    pub value: String,
}

/// Whether session is named (name non-empty after trim). Consumer: NameFilter filtering.
pub fn has_session_name(listing: &SessionListing) -> bool {
    listing
        .name
        .as_deref()
        .is_some_and(|n| !n.trim().is_empty())
}

/// Search index text (matching pi getSessionSearchText).
fn search_text(listing: &SessionListing) -> String {
    format!(
        "{} {} {} {}",
        listing.id,
        listing.name.as_deref().unwrap_or(""),
        listing.all_messages_text,
        listing.cwd
    )
}

fn normalize_whitespace_lower(text: &str) -> String {
    text.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Parse search input: `re:<pattern>` regex mode; quotes wrap into phrase token; unclosed quote falls back to whitespace tokenization;
/// Empty string → Empty. Consumers: filter_and_sort_sessions, session-selector search box.
pub fn parse_search_query(query: &str) -> SearchQuery {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return SearchQuery::Empty;
    }

    // Regex mode: re:<pattern> (case-insensitive)
    if let Some(pattern) = trimmed.strip_prefix("re:") {
        let pattern = pattern.trim();
        if pattern.is_empty() {
            return SearchQuery::Invalid("Empty regex".to_string());
        }
        match Regex::new(pattern) {
            Ok(re) => return SearchQuery::Regex(re),
            Err(e) => return SearchQuery::Invalid(e.to_string()),
        }
    }

    // Tokenized mode: quotes = phrase token
    let mut tokens: Vec<SearchToken> = Vec::new();
    let mut buf = String::new();
    let mut in_quote = false;
    let mut had_unclosed_quote = false;

    for ch in trimmed.chars() {
        if ch == '"' {
            if in_quote {
                // Closing quote → phrase token (pi flush("phrase"))
                flush_phrase(&mut tokens, &mut buf);
                in_quote = false;
            } else {
                flush_fuzzy(&mut tokens, &mut buf);
                in_quote = true;
            }
            continue;
        }
        if !in_quote && ch.is_whitespace() {
            flush_fuzzy(&mut tokens, &mut buf);
            continue;
        }
        buf.push(ch);
    }

    if in_quote {
        had_unclosed_quote = true;
    }

    // Unclosed quote: fall back to whitespace-only tokenization (all fuzzy)
    if had_unclosed_quote {
        return SearchQuery::Tokens(
            trimmed
                .split_whitespace()
                .map(|t| SearchToken {
                    kind: SearchTokenKind::Fuzzy,
                    value: t.to_string(),
                })
                .collect(),
        );
    }

    if in_quote {
        flush_phrase(&mut tokens, &mut buf);
    } else {
        flush_fuzzy(&mut tokens, &mut buf);
    }

    SearchQuery::Tokens(tokens)
}

fn flush_fuzzy(tokens: &mut Vec<SearchToken>, buf: &mut String) {
    let v = buf.trim().to_string();
    buf.clear();
    if !v.is_empty() {
        tokens.push(SearchToken {
            kind: SearchTokenKind::Fuzzy,
            value: v,
        });
    }
}

fn flush_phrase(tokens: &mut Vec<SearchToken>, buf: &mut String) {
    let v = buf.trim().to_string();
    buf.clear();
    if !v.is_empty() {
        tokens.push(SearchToken {
            kind: SearchTokenKind::Phrase,
            value: v,
        });
    }
}

/// Single-session match score: hit returns Some(lower is better), miss returns None (fuzzy semantics aligned with zen fuzzy module).
/// Text = id + name + all_messages_text + cwd (whitespace-normalized).
pub fn match_session(listing: &SessionListing, query: &SearchQuery) -> Option<f64> {
    let text = search_text(listing);

    match query {
        SearchQuery::Empty => Some(0.0),
        SearchQuery::Regex(re) => {
            let pos = re.find(&text)?.start();
            Some(pos as f64 * 0.1)
        }
        SearchQuery::Invalid(_) => None,
        SearchQuery::Tokens(tokens) => {
            if tokens.is_empty() {
                return Some(0.0);
            }
            let mut total = 0.0;
            let mut normalized: Option<String> = None;
            for token in tokens {
                if token.kind == SearchTokenKind::Phrase {
                    if normalized.is_none() {
                        normalized = Some(normalize_whitespace_lower(&text));
                    }
                    let phrase = normalize_whitespace_lower(&token.value);
                    if phrase.is_empty() {
                        continue;
                    }
                    let idx = normalized.as_ref().unwrap().find(&phrase)?;
                    total += idx as f64 * 0.1;
                } else {
                    let score = fuzzy_match(&token.value, &text)?;
                    total += score;
                }
            }
            Some(total)
        }
    }
}

/// Filter + sort session list (threaded = original order; recent = by modified descending; relevance = by score ascending,
/// ties by modified descending); name_filter filters unnamed. Consumers: session-selector and T-AP-08-5.
///
/// Matches pi filterAndSortSessions: empty query only applies name_filter; recent mode search only filters and keeps original order;
/// Other modes (threaded/relevance) sort by score after search, ties by modified descending.
pub fn filter_and_sort_sessions(
    mut listings: Vec<SessionListing>,
    query: &str,
    sort: SortMode,
    name_filter: NameFilter,
) -> Vec<SessionListing> {
    if name_filter == NameFilter::Named {
        listings.retain(has_session_name);
    }

    let trimmed = query.trim();
    if trimmed.is_empty() {
        return listings;
    }

    let parsed = parse_search_query(trimmed);
    if matches!(parsed, SearchQuery::Invalid(_)) {
        return Vec::new();
    }

    // Recent: filter only, keep original order
    if sort == SortMode::Recent {
        listings.retain(|l| match_session(l, &parsed).is_some());
        return listings;
    }

    // Other modes: sort by score after search (ties by modified descending)
    let mut scored: Vec<(SessionListing, f64)> = Vec::new();
    for l in listings {
        if let Some(score) = match_session(&l, &parsed) {
            scored.push((l, score));
        }
    }
    scored.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.0.modified.cmp(&a.0.modified))
    });
    scored.into_iter().map(|(l, _)| l).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::session_listing::SessionListing;
    use chrono::TimeZone;
    use std::path::PathBuf;

    fn listing(
        id: &str,
        name: Option<&str>,
        text: &str,
        cwd: &str,
        modified_day: u32,
    ) -> SessionListing {
        SessionListing::new(
            id.into(),
            PathBuf::from(format!("/tmp/{id}.jsonl")),
            cwd.into(),
            name.map(|s| s.to_string()),
            None,
            chrono::Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap(),
            chrono::Utc
                .with_ymd_and_hms(2025, 1, modified_day, 0, 0, 0)
                .unwrap(),
            1,
            "first".into(),
            text.into(),
        )
    }

    #[test]
    fn has_session_name_trims() {
        assert!(has_session_name(&listing(
            "a",
            Some("  name  "),
            "t",
            "/",
            1
        )));
        assert!(!has_session_name(&listing("a", Some("   "), "t", "/", 1)));
        assert!(!has_session_name(&listing("a", None, "t", "/", 1)));
    }

    #[test]
    fn parse_empty_query() {
        assert!(matches!(parse_search_query(""), SearchQuery::Empty));
        assert!(matches!(parse_search_query("   "), SearchQuery::Empty));
    }

    #[test]
    fn parse_regex_mode() {
        match parse_search_query("re:node") {
            SearchQuery::Regex(_) => {}
            other => panic!("expected regex, got {other:?}"),
        }
        let invalid = parse_search_query("re:");
        assert!(matches!(invalid, SearchQuery::Invalid(_)));
        let bad = parse_search_query("re:[");
        assert!(matches!(bad, SearchQuery::Invalid(_)));
    }

    #[test]
    fn parse_tokens_and_phrases() {
        match parse_search_query("foo \"node cve\" bar") {
            SearchQuery::Tokens(tokens) => {
                assert_eq!(tokens.len(), 3);
                assert_eq!(tokens[0].kind, SearchTokenKind::Fuzzy);
                assert_eq!(tokens[0].value, "foo");
                assert_eq!(tokens[1].kind, SearchTokenKind::Phrase);
                assert_eq!(tokens[1].value, "node cve");
                assert_eq!(tokens[2].value, "bar");
            }
            other => panic!("expected tokens, got {other:?}"),
        }
    }

    #[test]
    fn parse_unclosed_quote_falls_back_to_whitespace() {
        match parse_search_query("foo \"bar") {
            SearchQuery::Tokens(tokens) => {
                assert_eq!(tokens.len(), 2);
                assert!(tokens.iter().all(|t| t.kind == SearchTokenKind::Fuzzy));
            }
            other => panic!("expected tokens, got {other:?}"),
        }
    }

    #[test]
    fn match_phrase_finds_normalized_text() {
        let l = listing("s1", None, "Hello   World  again", "/repo", 1);
        let q = parse_search_query("\"hello world\"");
        let score = match_session(&l, &q);
        assert!(score.is_some());
        // Search text = "s1 hello world again /repo" → phrase hit position 3 → 0.3
        assert_eq!(score, Some(0.30000000000000004));
    }

    #[test]
    fn match_fuzzy_uses_search_text() {
        let l = listing("s1", Some("My sessions"), "Rust refactor notes", "/repo", 1);
        let q = parse_search_query("rust");
        assert!(match_session(&l, &q).is_some());
        let q_miss = parse_search_query("zzzz");
        assert!(match_session(&l, &q_miss).is_none());
    }

    #[test]
    fn match_regex_by_position() {
        let l = listing("s1", None, "alpha node beta node", "/", 1);
        match parse_search_query("re:node") {
            SearchQuery::Regex(_) => {}
            _ => unreachable!(),
        }
        let q = parse_search_query("re:b");
        let score = match_session(&l, &q).unwrap();
        assert!(score > 0.0);
    }

    #[test]
    fn filter_recent_keeps_order() {
        let mut listings = vec![
            listing("b", None, "x", "/", 3),
            listing("a", None, "yy", "/", 1),
        ];
        let out = filter_and_sort_sessions(
            std::mem::take(&mut listings),
            "",
            SortMode::Recent,
            NameFilter::All,
        );
        assert_eq!(out[0].id, "b");
        assert_eq!(out[1].id, "a");
    }

    #[test]
    fn filter_named_only() {
        let listings = vec![
            listing("a", Some("n"), "t", "/", 1),
            listing("b", None, "t", "/", 2),
        ];
        let out = filter_and_sort_sessions(listings, "", SortMode::Recent, NameFilter::Named);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, "a");
    }

    #[test]
    fn filter_invalid_query_returns_empty() {
        let listings = vec![listing("a", None, "t", "/", 1)];
        let out = filter_and_sort_sessions(listings, "re:[", SortMode::Threaded, NameFilter::All);
        assert!(out.is_empty());
    }

    #[test]
    fn filter_threaded_sorts_by_score() {
        // "rust" fuzzy: entries containing "rust" rank first (lower score); non-matches dropped.
        let listings = vec![
            listing("a", None, "jump rust", "/", 1),
            listing("b", None, "nothing here", "/", 2),
        ];
        let out = filter_and_sort_sessions(listings, "rust", SortMode::Threaded, NameFilter::All);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, "a");
    }
}
