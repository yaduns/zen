use crate::components::Component;
use crate::screen::{ScreenBuffer, Style};

/// Container component — fill the entire rectangle (including padding) with bg; children render sequentially inside the inset rectangle and stack vertically by returned line count;
/// Empty children → 0 lines; no cache fields (ScreenBuffer diffing is the cache).
/// Container(1,1, custom message background) and a plain container without background (chat composition) are structurally identical.
pub struct Container {
    padding_x: u16,
    padding_y: u16,
    bg: Option<Style>,
    children: Vec<Box<dyn Component>>, // Child components (managed by add_child/remove_child/clear)
}

impl Container {
    /// Constructor; bg is the background style (covers the entire rectangle including padding). Consumer: T-AP-11-2 (custom message background).
    pub fn new(padding_x: u16, padding_y: u16, bg: Option<Style>) -> Self {
        Container {
            padding_x,
            padding_y,
            bg,
            children: Vec::new(),
        }
    }

    /// Append a child component (counterpart of addChild).
    pub fn add_child(&mut self, child: Box<dyn super::Component>) {
        self.children.push(child);
    }

    /// Remove the child component at index (counterpart of removeChild; out-of-bounds ignored).
    pub fn remove_child(&mut self, index: usize) {
        if index < self.children.len() {
            self.children.remove(index);
        }
    }

    /// Clear child components (counterpart of clear).
    pub fn clear(&mut self) {
        self.children.clear();
    }

    /// Update the background style.
    pub fn set_bg(&mut self, bg: Option<Style>) {
        self.bg = bg;
    }
}

impl super::Component for Container {
    /// No children → 0 lines (aligned with pi empty box); otherwise fill the background (including padding), child components according to
    /// Returned line counts are stacked vertically within the indented rectangle; returns the total line count (capped at height).
    fn render(&mut self, buf: &mut ScreenBuffer, x: u16, y: u16, width: u16, height: u16) -> u16 {
        if self.children.is_empty() || height == 0 {
            return 0;
        }

        // Fill the background (full rectangle including padding)
        if let Some(ref bg) = self.bg {
            buf.fill(x, y, width, height, bg);
        }

        // Content rectangle (indented by padding)
        let inner_x = x + self.padding_x;
        let inner_y = y + self.padding_y;
        let inner_w = width.saturating_sub(self.padding_x * 2);
        let inner_h = height.saturating_sub(self.padding_y * 2);

        if inner_w == 0 || inner_h == 0 {
            return 0;
        }

        // Stack child components vertically
        let mut row = inner_y;
        let mut children_rows = 0u16;
        let inner_bottom = inner_y + inner_h;

        for child in &mut self.children {
            if row >= inner_bottom {
                break;
            }
            let remaining = inner_bottom.saturating_sub(row);
            let painted = child.render(buf, inner_x, row, inner_w, remaining);
            row += painted;
            children_rows += painted;
        }

        // Total lines = top padding + child components + bottom padding
        let total = self.padding_y + children_rows + self.padding_y;

        if self.bg.is_some() {
            height
        } else {
            total.min(height)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screen::{ScreenBuffer, Style};
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Test child component: records the rectangle (x, y, width, height) received on each render,
    /// Returns a fixed line count (clamped by the passed height budget, simulating real components' clipping semantics).
    struct RecordingChild {
        rows: u16,
        calls: Rc<RefCell<Vec<(u16, u16, u16, u16)>>>,
    }

    impl Component for RecordingChild {
        fn render(
            &mut self,
            _buf: &mut ScreenBuffer,
            x: u16,
            y: u16,
            width: u16,
            height: u16,
        ) -> u16 {
            self.calls.borrow_mut().push((x, y, width, height));
            self.rows.min(height)
        }
    }

    fn child(rows: u16, calls: &Rc<RefCell<Vec<(u16, u16, u16, u16)>>>) -> RecordingChild {
        RecordingChild {
            rows,
            calls: calls.clone(),
        }
    }

    #[test]
    fn new_defaults_and_children_management() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut c = Container::new(1, 2, None);
        assert_eq!(c.padding_x, 1);
        assert_eq!(c.padding_y, 2);
        assert!(c.bg.is_none());
        assert!(c.children.is_empty());

        c.add_child(Box::new(child(1, &calls)));
        c.add_child(Box::new(child(1, &calls)));
        assert_eq!(c.children.len(), 2);

        c.remove_child(0);
        assert_eq!(c.children.len(), 1);
        // Ignore out of bounds
        c.remove_child(99);
        assert_eq!(c.children.len(), 1);

        c.clear();
        assert_eq!(c.children.len(), 0);
    }

    #[test]
    fn set_bg_updates() {
        let mut c = Container::new(0, 0, None);
        assert!(c.bg.is_none());
        c.set_bg(Some(Style::default()));
        assert!(c.bg.is_some());
        c.set_bg(None);
        assert!(c.bg.is_none());
    }

    #[test]
    fn render_empty_or_zero_height_returns_zero() {
        let mut buf = ScreenBuffer::new(20, 10);
        let mut c = Container::new(0, 0, None);
        // Empty box → 0
        assert_eq!(c.render(&mut buf, 0, 0, 20, 10), 0);
        // Has children but height=0 → 0
        let calls = Rc::new(RefCell::new(Vec::new()));
        c.add_child(Box::new(child(1, &calls)));
        assert_eq!(c.render(&mut buf, 0, 0, 20, 0), 0);
    }

    #[test]
    fn render_single_child_counts_padding_without_bg() {
        let mut buf = ScreenBuffer::new(20, 10);
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut c = Container::new(0, 1, None);
        c.add_child(Box::new(child(1, &calls)));
        // total = 1 (top pad) + 1 (content) + 1 (bottom pad) = 3 (no bg → content determines height)
        assert_eq!(c.render(&mut buf, 0, 0, 20, 10), 3);
        assert_eq!(calls.borrow().len(), 1);
    }

    #[test]
    fn render_without_bg_budget_clamps_child_and_totals_padding() {
        let mut buf = ScreenBuffer::new(20, 10);
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut c = Container::new(0, 1, None);
        c.add_child(Box::new(child(5, &calls)));
        // inner_h = height - 2*padding_y = 3-2 = 1: child gets only a 1-line budget → clamped to 1 line
        // total = 1 (top pad) + 1 + 1 (bottom pad) = 3 = height;
        // Therefore the outer total.min(height) is identity under this cap (children have already been clipped by the budget).
        assert_eq!(c.render(&mut buf, 0, 0, 20, 3), 3);
        assert_eq!(*calls.borrow(), vec![(0, 1, 20, 1)]);
    }

    #[test]
    fn render_with_bg_returns_full_height() {
        let mut buf = ScreenBuffer::new(20, 10);
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut c = Container::new(1, 1, Some(Style::default()));
        c.add_child(Box::new(child(1, &calls)));
        // With bg → fill semantics; returns the full allocated height (even if child content is only 1 line)
        assert_eq!(c.render(&mut buf, 3, 2, 20, 10), 10);
        // Rectangle received by child = inner rectangle after indentation
        assert_eq!(*calls.borrow(), vec![(4, 3, 18, 8)]);
    }

    #[test]
    fn render_inner_collapse_returns_zero() {
        let mut buf = ScreenBuffer::new(20, 10);
        // Width consumed entirely by padding (padding_x*2 = width)
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut c = Container::new(10, 1, None);
        c.add_child(Box::new(child(1, &calls)));
        assert_eq!(c.render(&mut buf, 0, 0, 20, 10), 0);
        assert!(calls.borrow().is_empty());
        // Height consumed entirely by padding
        let mut c2 = Container::new(0, 6, None);
        c2.add_child(Box::new(child(1, &calls)));
        assert_eq!(c2.render(&mut buf, 0, 0, 20, 10), 0);
    }

    #[test]
    fn render_passes_inner_rect_and_stacks_children() {
        let mut buf = ScreenBuffer::new(20, 10);
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut c = Container::new(2, 1, None);
        c.add_child(Box::new(child(1, &calls)));
        c.add_child(Box::new(child(1, &calls)));
        // No bg, content determines height: 1 + (1+1) + 1 = 4
        assert_eq!(c.render(&mut buf, 10, 5, 20, 10), 4);
        // First child component: inner rectangle origin (10+2, 5+1), size (20-4, 10-2)
        // Second child component: y = inner_y + 1 line occupied by the first; height budget decreases
        let got = calls.borrow();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0], (12, 6, 16, 8));
        assert_eq!(got[1], (12, 7, 16, 7));
    }

    #[test]
    fn render_caps_children_at_inner_bottom() {
        let mut buf = ScreenBuffer::new(20, 10);
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut c = Container::new(0, 0, None);
        c.add_child(Box::new(child(3, &calls)));
        c.add_child(Box::new(child(3, &calls)));
        // inner_h = 5: the first uses 3 lines; the second gets only the remaining 2-line budget → clamped to 2
        assert_eq!(c.render(&mut buf, 0, 0, 20, 5), 5);
        let got = calls.borrow();
        assert_eq!(got[0], (0, 0, 20, 5));
        assert_eq!(got[1], (0, 3, 20, 2));
    }

    #[test]
    fn render_reports_total_min_height_without_bg() {
        let mut buf = ScreenBuffer::new(20, 10);
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut c = Container::new(0, 1, None);
        c.add_child(Box::new(child(7, &calls)));
        // child receives inner_h=8 budget, returns 7; total = 1+7+1 = 9 ≤ height=10
        assert_eq!(c.render(&mut buf, 0, 0, 20, 10), 9);
        // Insufficient height budget: height=3 → total clamped to 3
        assert_eq!(c.render(&mut buf, 0, 0, 20, 3), 3);
    }
}
