// REQ-TUI-03 (pi-402): fuzzy matching — subsequence matching, quality scoring, and filter sorting (pure logic, no terminal access).
// Corresponds to original TS: reference/pi-original/packages/tui/src/fuzzy.ts.
// Differences from pi:
// - The `FuzzyMatch` interface (matches/score) is resolved to `Option<f64>` (None = no match); no result wrapper type is introduced
//   (no consumer needs a composite "match + score" result; see design conclusion in tasks/T-TU-03-4-implemented.md);
// - Word-boundary character set uses `char::is_whitespace()` to cover JS `\s` (minor Unicode table differences are edge-case text);
// - Text is iterated by char: the index basis for position penalty `i * 0.1` differs from JS UTF-16 code units for astral characters
//   (only affects fractional penalty values, not match determination or ordering of ordinary text);
// - `fuzzy_filter`'s `get_text` returns the candidate's own `&str` (borrow constraint); pi's getText can return arbitrary
//   concatenated strings — this module does not support returning temporary concatenated values (consumers prepare their own match-text field).

use std::cmp::Ordering;

/// Single-token fuzzy match: all characters in the query must appear **in order** (not necessarily contiguous) in the text (case-insensitive),
/// Returns a quality score on match, `None` on no match. Every query term in autocomplete command-name/search-box matching is judged by it,
/// Smaller score means better match (sort key).
/// Scoring (aligned with `matchQuery` inside pi `fuzzyMatch`): exact equality -100; each consecutive hit -5n; word-boundary
/// (preceding text character is whitespace/`-_./:`) each hit -10; gap penalty +2 per slot; later hit position penalty +0.1×char index;
/// Empty query always matches with 0; query longer than text does not match. **Alphanumeric swap fallback**: when the query is exactly "letters+digits" or
/// "digits+letters" and the above direct match fails, swap the two parts and try again (e.g. `codex52`→`52codex`); if the swap matches
/// the score is the swapped score +5. `text` is iterated by char (aligned with pi's code-unit semantics; for astral characters the index basis
/// differs slightly from JS UTF-16, affecting only fractional penalty values, not match determination).
/// Consumers: this module's `fuzzy_filter`/`fuzzy_matches`; T-AP-11-2 (interactive-mode session/model selection search).
pub fn fuzzy_match(query: &str, text: &str) -> Option<f64> {
    let query_lower = query.to_lowercase();
    let text_lower = text.to_lowercase();
    if let Some(score) = score_subsequence(&query_lower, &text_lower) {
        return Some(score);
    }
    let swapped = try_swap_alphanumeric(&query_lower)?;
    score_subsequence(&swapped, &text_lower).map(|score| score + 5.0)
}

/// Fuzzy filter + sort: split the query by whitespace and `/` (if `query.trim()` is empty, return all candidates in original order),
/// **all tokens must match**; sort ascending by the sum of token quality scores (stable), and return a reference view of matching candidates.
/// Autocomplete command-name routing (`autocomplete.rs::filter_commands` upgrade seam, aligned with pi
/// `fuzzyFilter(commandItems, prefix, i => i.name)`) and interactive-mode list search share it;
/// `get_text` obtains match text for each candidate — returns the candidate's own `&str` (does not require the candidate to be Clone,
/// `SlashCommand` contains closures and is not Clone). Empty query/pure whitespace/pure slash query: return all in original order (consistent with pi);
/// If no token matches: return an empty Vec; equal-score ties keep original order (stable sort).
/// Consumers: `autocomplete.rs::filter_commands` (upgraded by this task); T-AP-11-2 (reserved).
pub fn fuzzy_filter<'a, T>(
    items: &'a [T],
    query: &str,
    get_text: impl Fn(&T) -> &str,
) -> Vec<&'a T> {
    let tokens = split_tokens(query);
    if tokens.is_empty() {
        return items.iter().collect();
    }
    let mut results: Vec<(&'a T, f64)> = Vec::new();
    for item in items {
        let text = get_text(item);
        let mut total = 0.0;
        let mut all_match = true;
        for token in &tokens {
            let Some(score) = fuzzy_match(token, text) else {
                all_match = false;
                break;
            };
            total += score;
        }
        if all_match {
            results.push((item, total));
        }
    }
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    results.into_iter().map(|(item, _)| item).collect()
}

/// Boolean predicate form of fuzzy matching: after splitting the query by whitespace and `/`, returns true only if **all tokens match**
/// (token semantics are exactly the same as `fuzzy_filter`, just without ranking). Serves the existing `SettingsList` search injection seam
/// — `ItemFilter` is an `FnMut(&SettingItem, &str) -> bool` predicate and cannot express ranking; callers use it to build
/// a filter to get match semantics consistent with Pi settings search. Empty query (including pure whitespace/slash) is always true.
/// Consumers: currently no consumers within crates; reserved for T-AP-08-6 / T-AP-11-2 (SettingsList injects ItemFilter).
pub fn fuzzy_matches(query: &str, text: &str) -> bool {
    split_tokens(query)
        .iter()
        .all(|token| fuzzy_match(token, text).is_some())
}

/// Single subsequence scoring: scan the lowercased query/text as an 'in-order subsequence' and accumulate a quality score;
/// Returns `Some(score)` if all query characters match in order; otherwise `None`. Aligns with the pi `matchQuery` closure
/// (consecutive matches / gap penalty / word boundary / position penalty / exact equality); called by both the primary attempt and swap fallback of `fuzzy_match`.
fn score_subsequence(query_lower: &str, text_lower: &str) -> Option<f64> {
    if query_lower.is_empty() {
        return Some(0.0);
    }
    let query: Vec<char> = query_lower.chars().collect();
    let text: Vec<char> = text_lower.chars().collect();
    if query.len() > text.len() {
        return None;
    }
    let mut query_index = 0;
    let mut score = 0.0f64;
    let mut last_match_index: i64 = -1;
    let mut consecutive_matches = 0i64;
    for (i, &c) in text.iter().enumerate() {
        if query_index == query.len() {
            break;
        }
        if c != query[query_index] {
            continue;
        }
        let i_i64 = i as i64;
        if last_match_index == i_i64 - 1 {
            // Consecutive matches: reward increases stepwise (aligned with pi: after consecutiveMatches++, score -= n*5)
            consecutive_matches += 1;
            score -= (consecutive_matches * 5) as f64;
        } else {
            consecutive_matches = 0;
            if last_match_index >= 0 {
                // Gap penalty: +2 per gap cell between the previous match and this one
                score += ((i_i64 - last_match_index - 1) * 2) as f64;
            }
        }
        // Word-boundary match reward: start of line or previous char is whitespace/`-_./:`
        if i == 0 || is_word_boundary(text[i - 1]) {
            score -= 10.0;
        }
        // Small penalty for later matches
        score += (i as f64) * 0.1;
        last_match_index = i_i64;
        query_index += 1;
    }
    if query_index < query.len() {
        return None;
    }
    if query_lower == text_lower {
        score -= 100.0;
    }
    Some(score)
}

/// Alphanumeric swap: when the query is exactly 'letters+digits' or 'digits+letters', returns the swapped string (`codex52`→`52codex`),
/// otherwise `None`. Aligns with pi's two anchored regexes (`^[a-z]+[0-9]+$` and the reverse); called only on lowercase queries.
fn try_swap_alphanumeric(query_lower: &str) -> Option<String> {
    let chars: Vec<char> = query_lower.chars().collect();
    if chars.is_empty() {
        return None;
    }
    // Accepts only [a-z0-9]+ (the character domain of pi's anchored regexes; other characters are not eligible for swap)
    if !chars
        .iter()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
    {
        return None;
    }
    let first_is_lower = chars[0].is_ascii_lowercase();
    let is_second_kind = |c: char| {
        if first_is_lower {
            c.is_ascii_digit()
        } else {
            c.is_ascii_lowercase()
        }
    };
    // Length of the first segment: consecutive characters of the same kind from the start until another kind is encountered
    let mut first_end = 1;
    while first_end < chars.len() && !is_second_kind(chars[first_end]) {
        first_end += 1;
    }
    // Exactly two segments: the first is non-empty (always holds), the second exists and is entirely the other kind
    if first_end == chars.len() || !chars[first_end..].iter().all(|&c| is_second_kind(c)) {
        return None;
    }
    let (first, second) = chars.split_at(first_end);
    let mut swapped = String::with_capacity(chars.len());
    swapped.extend(second.iter());
    swapped.extend(first.iter());
    Some(swapped)
}

/// Splits the query by whitespace and `/` into a token list (empty tokens removed); an empty string or only separators returns an empty Vec.
/// Aligns with pi `query.trim().split(/[\s/]+/)`; shared by `fuzzy_filter`/`fuzzy_matches`.
fn split_tokens(query: &str) -> Vec<&str> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    trimmed
        .split(|c: char| c.is_whitespace() || c == '/')
        .filter(|token| !token.is_empty())
        .collect()
}

/// Word-boundary character check: whitespace (aligned with JS `\s`) and `-_./:` (aligned with pi regex `[\s\-_./:]`).
fn is_word_boundary(c: char) -> bool {
    c.is_whitespace() || matches!(c, '-' | '_' | '.' | '/' | ':')
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── fuzzy_match: migrated from pi tui/test/fuzzy.test.ts ──

    #[test]
    fn empty_query_matches_anything_with_score_zero() {
        assert_eq!(fuzzy_match("", "anything"), Some(0.0));
    }

    #[test]
    fn query_longer_than_text_does_not_match() {
        assert_eq!(fuzzy_match("longquery", "short"), None);
    }

    #[test]
    fn exact_match_has_good_negative_score() {
        let score = fuzzy_match("test", "test").unwrap();
        assert!(score < 0.0, "exact match should be strongly rewarded");
    }

    #[test]
    fn characters_must_appear_in_order() {
        assert!(
            fuzzy_match("abc", "aXbXc").is_some(),
            "in-order subsequence"
        );
        assert!(
            fuzzy_match("abc", "cba").is_none(),
            "out-of-order must not match"
        );
    }

    #[test]
    fn case_insensitive_matching_both_directions() {
        assert!(fuzzy_match("ABC", "abc").is_some());
        assert!(fuzzy_match("abc", "ABC").is_some());
    }

    #[test]
    fn consecutive_matches_score_better_than_scattered() {
        let consecutive = fuzzy_match("foo", "foobar").unwrap();
        let scattered = fuzzy_match("foo", "f_o_o_bar").unwrap();
        assert!(
            consecutive < scattered,
            "consecutive={consecutive} scattered={scattered}"
        );
    }

    #[test]
    fn word_boundary_matches_score_better() {
        let at_boundary = fuzzy_match("fb", "foo-bar").unwrap();
        let not_at_boundary = fuzzy_match("fb", "afbx").unwrap();
        assert!(
            at_boundary < not_at_boundary,
            "boundary={at_boundary} non-boundary={not_at_boundary}"
        );
    }

    #[test]
    fn matches_swapped_alphanumeric_tokens() {
        // Direct 'codex52' is not in order in 'gpt-5.2-codex'; swapped '52codex' matches (aligned with pi test case)
        assert!(fuzzy_match("codex52", "gpt-5.2-codex").is_some());
    }

    // ── fuzzy_filter: migrated from pi tui/test/fuzzy.test.ts ──

    #[test]
    fn empty_query_returns_all_items_unchanged() {
        let items = vec!["apple", "banana", "cherry"];
        let result = fuzzy_filter(&items, "", |x| *x);
        assert_eq!(result, vec![&items[0], &items[1], &items[2]]);
    }

    #[test]
    fn filters_out_non_matching_items() {
        let items = vec!["apple", "banana", "cherry"];
        let result = fuzzy_filter(&items, "an", |x| *x);
        // 'an' in order: banana (a1,n2) matches; apple has no n, cherry has no a/n, so neither matches
        assert_eq!(result, vec![&items[1]]);
    }

    #[test]
    fn sorts_results_by_match_quality() {
        let items = vec!["a_p_p", "app", "application"];
        let result = fuzzy_filter(&items, "app", |x| *x);
        // Step-by-step calculation aligned with pi's algorithm (at i=0, lastMatchIndex===i-1, i.e. -1===-1, takes the consecutive branch):
        // app (exact equality -100) ranks first; application's consecutive matches after the first character (-5-10-15) beat
        // a_p_p)'s two gaps + word boundaries (-10+2-10+2-10); so the full order is [app, application, a_p_p]
        assert_eq!(result, vec![&items[1], &items[2], &items[0]]);
    }

    #[test]
    fn prioritizes_exact_matches_over_longer_prefix_matches() {
        let items = vec!["clone", "cl"];
        let result = fuzzy_filter(&items, "cl", |x| *x);
        // 'cl' exact equality (-100) comes before 'clone' (subsequence bonus only)
        assert_eq!(result, vec![&items[1], &items[0]]);
    }

    #[test]
    fn works_with_custom_get_text() {
        let items = vec![
            ("foo".to_string(), 1),
            ("bar".to_string(), 2),
            ("foobar".to_string(), 3),
        ];
        let result = fuzzy_filter(&items, "foo", |item| item.0.as_str());
        assert_eq!(result.len(), 2);
        let names: Vec<&str> = result.iter().map(|m| m.0.as_str()).collect();
        assert!(names.contains(&"foo"));
        assert!(names.contains(&"foobar"));
        assert!(!names.contains(&"bar"));
    }

    #[test]
    fn slash_separated_tokens_match_against_reordered_text() {
        // pi test case: after tokenizing query 'openai-codex/gpt-5.5', both tokens are in 'gpt-5.5 openai-codex'
        // match in order (tokens match against the full text; text order is not required)
        let items = vec!["gpt-5.5 openai-codex"];
        let result = fuzzy_filter(&items, "openai-codex/gpt-5.5", |x| *x);
        assert_eq!(result, vec![&items[0]]);
    }

    #[test]
    fn stable_sort_keeps_original_order_on_equal_scores() {
        // 'axb' and 'ayb' score the same for query 'ab' (a boundary -10, b gap +2 same position penalty) → stable original order is preserved
        let items = vec!["axb", "ayb"];
        let result = fuzzy_filter(&items, "ab", |x| *x);
        assert_eq!(result, vec![&items[0], &items[1]]);
    }

    // ── fuzzy_matches ──

    #[test]
    fn matches_single_and_multi_token_queries() {
        assert!(fuzzy_matches("foo", "foobar"));
        assert!(fuzzy_matches("foo bar", "food and bars"));
        assert!(fuzzy_matches("a/b", "xx a yy b"));
        assert!(!fuzzy_matches("zzz", "foo"));
        assert!(!fuzzy_matches("foo zzz", "food"));
    }

    #[test]
    fn matches_blank_query_is_always_true() {
        assert!(fuzzy_matches("", "anything"));
        assert!(fuzzy_matches("  /  ", "anything"));
    }

    #[test]
    fn boolean_predicate_consistent_with_filter() {
        // For the same query, the match set from fuzzy_matches corresponds one-to-one with the filtered result from fuzzy_filter
        let items = vec!["apple", "banana", "cherry"];
        for query in ["", "an", "app", "cl", "zz", "a/b"] {
            let filtered = fuzzy_filter(&items, query, |x| *x);
            for item in &items {
                assert_eq!(
                    filtered.contains(&item),
                    fuzzy_matches(query, item),
                    "query={query:?} item={item:?}"
                );
            }
        }
    }

    // ── private helpers ──

    #[test]
    fn score_subsequence_empty_query_scores_zero() {
        assert_eq!(score_subsequence("", "abc"), Some(0.0));
    }

    #[test]
    fn score_subsequence_query_longer_than_text_is_none() {
        assert_eq!(score_subsequence("abcd", "abc"), None);
    }

    #[test]
    fn score_subsequence_out_of_order_is_none() {
        assert_eq!(score_subsequence("abc", "cba"), None);
    }

    #[test]
    fn score_subsequence_exact_match_applies_big_bonus() {
        let exact = score_subsequence("test", "test").unwrap();
        let subsequence = score_subsequence("test", "xyztxeysxt").unwrap();
        assert!(
            exact < subsequence,
            "exact={exact} subsequence={subsequence}"
        );
    }

    #[test]
    fn swap_alphanumeric_maps_both_directions() {
        assert_eq!(
            try_swap_alphanumeric("codex52"),
            Some("52codex".to_string())
        );
        assert_eq!(
            try_swap_alphanumeric("52codex"),
            Some("codex52".to_string())
        );
        assert_eq!(try_swap_alphanumeric("a1"), Some("1a".to_string()));
        assert_eq!(try_swap_alphanumeric("12ab"), Some("ab12".to_string()));
    }

    #[test]
    fn swap_alphanumeric_rejects_single_segment_and_mixed() {
        assert_eq!(try_swap_alphanumeric("codex"), None); // only one span
        assert_eq!(try_swap_alphanumeric("123"), None); // only one span
        assert_eq!(try_swap_alphanumeric("a1b"), None); // three spans
        assert_eq!(try_swap_alphanumeric("abc123def"), None); // three spans
        assert_eq!(try_swap_alphanumeric(""), None);
        assert_eq!(try_swap_alphanumeric("ab-cd12"), None); // Contains characters outside [a-z0-9]
        assert_eq!(try_swap_alphanumeric("ABC12"), None); // Uppercase is outside the [a-z] domain
    }

    #[test]
    fn split_tokens_splits_on_whitespace_and_slash() {
        assert_eq!(split_tokens(""), Vec::<&str>::new());
        assert_eq!(split_tokens("   "), Vec::<&str>::new());
        assert_eq!(split_tokens("a b"), vec!["a", "b"]);
        assert_eq!(split_tokens("a/b"), vec!["a", "b"]);
        assert_eq!(split_tokens(" a/b\tc "), vec!["a", "b", "c"]);
        assert_eq!(split_tokens("a//b"), vec!["a", "b"]);
    }

    #[test]
    fn is_word_boundary_accepts_delimiters_and_whitespace() {
        for c in [' ', '\t', '\n', '-', '_', '.', '/', ':'] {
            assert!(is_word_boundary(c), "{c:?} should be a boundary");
        }
        for c in ['a', '0', '中', 'x'] {
            assert!(!is_word_boundary(c), "{c:?} should not be a boundary");
        }
    }
}
