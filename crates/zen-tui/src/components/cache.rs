// PF-04 (REQ-TUI-01 render cache): line-level render-cache primitive — single-slot `LineCache` + three-field `CacheKey`.
// Plan doc: docs/performance-optimization-plan.md §PF-04; task notes: tasks/PF-04-implemented.md
//
// Design constraints (rulings from the task notes):
// - Single slot, not a map: a row has exactly one current content, so extra slots would only hoard dead memory.
// - The key is an O(1) comparison (width + content version + global-input generation); **no text hashing** (that would be O(whole history) per frame).
// - It stores lines already truncated to `key.width`, so the hit path can use the non-truncating entry (no per-character width scan per line).
// - `renders()` is the acceptance/diagnostics surface: a steady-state frame's re-render count must be asserted as a difference of 0.
// - Cannot express time-dependent rendering (Loader frame index / CountdownTimer remaining seconds): see LineCache's documentation warning.
use crate::highlight::LineSegs;

/// A row's render fingerprint: this width + this content generation + this global-input generation uniquely identify one set of
/// rendered lines.
///
/// Built by the holder (zen-app's `RowView`) on every lookup; the cache only compares keys and never interprets them.
/// `width` makes resize invalidate automatically (no explicit global flush needed); `content_version` is bumped per row by the UI
/// helpers whenever that row's data changes; `global_epoch` is bumped whenever an input shared by *all* rows changes
/// (theme / thinking collapse / output padding). All three are O(1) comparisons — deliberately **not** a hash of the row text, which
/// would cost O(whole history) per frame.
///
/// Consumer: zen-app `modes::interactive::ui::push_row_lines` (planned: PF-05 line-stream convergence, PF-06 fixed-region caching).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheKey {
    /// Render width (line wrapping / truncation target); a resize therefore misses automatically.
    pub width: u16,
    /// Per-row content generation, bumped by the holder when that row's data changes.
    pub content_version: u64,
    /// Generation of the inputs shared by every row (theme, thinking collapse, output padding).
    pub global_epoch: u64,
}

/// Single-slot cache of one row's rendered lines: the chat stream asks "did this row change?" once per frame, and a hit means
/// **no markdown parsing, no highlighting, no width scanning, no component construction** — the zen counterpart of pi's
/// `Markdown.cachedLines`.
///
/// The stored lines are **already truncated to `key.width`** (they come out of a width-bounded `ScreenBuffer` extraction), which is why
/// a hit may be pushed through the non-truncating canvas entry — a per-character width scan per line is skipped entirely.
/// Single slot on purpose: a row has exactly one current content, so extra slots would only accumulate dead memory.
///
/// ⚠️ **Not expressible here: time-dependent rendering.** A row whose appearance depends on the clock (the bash-row `Loader` spinner,
/// `CountdownTimer` remaining seconds) is frozen by caching, because the animation clock is not part of the key. Today the bash-row
/// spinner is *already* a dead frame (the component is rebuilt every frame, so its `started_at` is re-anchored), hence no regression —
/// but whoever makes it live must add a time dimension to the key (e.g. a `frame_bucket`) or bypass the cache for animated rows.
///
/// Consumers: zen-app `modes::interactive::ui::push_row_lines`; planned PF-05 (reuses the same invalidation semantics) and PF-06.
#[derive(Debug, Default)]
pub struct LineCache {
    /// Key of the stored lines; `None` = empty cache.
    key: Option<CacheKey>,
    /// Rendered lines, already truncated to `key.width` and empty lines already filtered out.
    ///
    /// Shared handles (PF-05): a hit hands these to the canvas by bumping refcounts, so a steady-state frame copies nothing.
    lines: Vec<LineSegs>,
    /// Real renders (= `store` calls = misses); never reset by [`LineCache::invalidate`].
    renders: u64,
}

impl LineCache {
    /// Empty cache (nothing stored, no render counted).
    ///
    /// Consumer: `RowView::default()` in zen-app `modes::interactive::ui`.
    pub fn new() -> LineCache {
        LineCache {
            key: None,
            lines: Vec::new(),
            renders: 0,
        }
    }

    /// Ask for the lines stored under `key`; `None` = miss (the caller must render and [`LineCache::store`]).
    ///
    /// Returns a slice of **shared handles** (PF-05), so pushing them into a canvas costs one refcount bump per line instead of a deep copy —
    /// this is the hot path of every frame and must not allocate.
    /// Consumer: zen-app `push_row_lines`.
    pub fn get(&self, key: &CacheKey) -> Option<&[LineSegs]> {
        match &self.key {
            Some(stored) if stored == key => Some(&self.lines),
            _ => None,
        }
    }

    /// Replace the stored lines with a freshly rendered set and record one real render (the acceptance counter for "a steady-state
    /// frame re-renders nothing").
    ///
    /// Boundary: storing an empty line set is legal — a row can legitimately render to zero lines — and still counts as a render.
    /// Precondition: the caller guarantees the lines are already truncated to `key.width`.
    /// The fresh handles push this row past `same_line`'s pointer fast path, so the next diff compares content for this row only — and writes
    /// it only if the content really changed.
    pub fn store(&mut self, key: CacheKey, lines: Vec<LineSegs>) {
        self.key = Some(key);
        self.lines = lines;
        self.renders += 1;
    }

    /// Drop the stored lines (key → `None`) without touching the render counter.
    ///
    /// Boundary: with `global_epoch` in the key this is no longer needed for theme / thinking-collapse / padding changes (a new epoch
    /// already misses); it stays for "drop everything, forget it" cases and for the PF-05 / PF-06 consumers that cache more than rows.
    pub fn invalidate(&mut self) {
        self.key = None;
        self.lines.clear();
    }

    /// Real renders performed by this cache so far (= `store` calls = cache misses).
    ///
    /// Diagnostics/acceptance surface: PF-04's acceptance is "a steady-state frame adds 0 renders", asserted as a **difference**
    /// (`let base = renders(); …; assert_eq!(renders() - base, 0)`), the same convention as `FrameStats::frames()`.
    /// Consumers: zen-app unit tests (`ui.rs`); no production consumer yet — showing it in the frame summary would need a cross-crate
    /// channel (registered as an open question in `tasks/PF-04-implemented.md`).
    pub fn renders(&self) -> u64 {
        self.renders
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::highlight::StyledSegment;
    use crate::screen::Style;

    fn key(width: u16, version: u64, epoch: u64) -> CacheKey {
        CacheKey {
            width,
            content_version: version,
            global_epoch: epoch,
        }
    }

    fn line(text: &str) -> Vec<StyledSegment> {
        vec![StyledSegment {
            style: Style::default(),
            text: text.to_string(),
        }]
    }

    fn lines(texts: &[&str]) -> Vec<LineSegs> {
        texts.iter().map(|t| std::rc::Rc::new(line(t))).collect()
    }

    #[test]
    fn new_cache_misses_and_counts_nothing() {
        let cache = LineCache::new();
        assert!(cache.get(&key(80, 0, 0)).is_none());
        assert_eq!(cache.renders(), 0);
    }

    #[test]
    fn default_equals_new() {
        // clippy's `new_without_default` requires both; they must agree.
        let a = LineCache::default();
        let b = LineCache::new();
        assert!(a.get(&key(80, 0, 0)).is_none());
        assert_eq!(a.renders(), b.renders());
    }

    #[test]
    fn store_then_get_hits_with_same_lines() {
        let mut cache = LineCache::new();
        let k = key(80, 3, 1);
        let stored = lines(&["hello", "world"]);
        cache.store(k, stored.clone());
        let hit = cache.get(&k).expect("same key must hit");
        assert_eq!(hit.len(), 2);
        assert_eq!(hit[0][0].text, "hello");
        assert_eq!(hit[1][0].text, "world");
        assert_eq!(cache.renders(), 1);
    }

    #[test]
    fn each_key_dimension_invalidates() {
        // The three dimensions must each be part of the comparison — a cache that only compares two of them
        // would show stale rows after a resize / a content update / a theme switch respectively.
        let mut cache = LineCache::new();
        cache.store(key(80, 3, 1), lines(&["a"]));
        assert!(cache.get(&key(80, 3, 1)).is_some());
        assert!(cache.get(&key(40, 3, 1)).is_none(), "width must invalidate");
        assert!(
            cache.get(&key(80, 4, 1)).is_none(),
            "content_version must invalidate"
        );
        assert!(
            cache.get(&key(80, 3, 2)).is_none(),
            "global_epoch must invalidate"
        );
    }

    #[test]
    fn second_store_replaces_and_counts_a_render() {
        let mut cache = LineCache::new();
        let k = key(80, 1, 0);
        cache.store(k, lines(&["old", "rows"]));
        cache.store(k, lines(&["new"]));
        let hit = cache.get(&k).expect("hit");
        assert_eq!(hit.len(), 1, "store replaces, never accumulates");
        assert_eq!(hit[0][0].text, "new");
        assert_eq!(cache.renders(), 2);
    }

    #[test]
    fn invalidate_clears_lines_but_keeps_the_render_counter() {
        let mut cache = LineCache::new();
        let k = key(80, 1, 0);
        cache.store(k, lines(&["a"]));
        cache.invalidate();
        assert!(cache.get(&k).is_none());
        // Difference assertions rely on the counter being cumulative, not reset by invalidation.
        assert_eq!(cache.renders(), 1);
    }

    #[test]
    fn empty_line_set_is_a_legal_render() {
        // A row can legitimately render to zero lines (e.g. a collapsed empty block): it must hit, not re-render.
        let mut cache = LineCache::new();
        let k = key(80, 1, 0);
        cache.store(k, Vec::new());
        assert_eq!(cache.get(&k).map(|l| l.len()), Some(0));
        assert_eq!(cache.renders(), 1);
    }
}
