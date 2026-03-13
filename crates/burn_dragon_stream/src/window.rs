use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TbpttWindow {
    pub unroll_steps: usize,
    pub backprop_steps: usize,
}

impl Default for TbpttWindow {
    fn default() -> Self {
        Self {
            unroll_steps: 1,
            backprop_steps: 1,
        }
    }
}

impl TbpttWindow {
    pub fn new(unroll_steps: usize, backprop_steps: usize) -> Self {
        let unroll_steps = unroll_steps.max(1);
        let backprop_steps = backprop_steps.clamp(1, unroll_steps);
        Self {
            unroll_steps,
            backprop_steps,
        }
    }

    pub fn detach_prefix_steps(self) -> usize {
        self.unroll_steps.saturating_sub(self.backprop_steps)
    }

    pub fn requires_detach(self) -> bool {
        self.backprop_steps < self.unroll_steps
    }
}

#[cfg(test)]
mod tests {
    use super::TbpttWindow;

    #[test]
    fn tbptt_window_clamps_and_reports_detach_prefix() {
        let window = TbpttWindow::new(8, 3);
        assert_eq!(window.unroll_steps, 8);
        assert_eq!(window.backprop_steps, 3);
        assert_eq!(window.detach_prefix_steps(), 5);
        assert!(window.requires_detach());

        let clamped = TbpttWindow::new(4, 8);
        assert_eq!(clamped.unroll_steps, 4);
        assert_eq!(clamped.backprop_steps, 4);
        assert_eq!(clamped.detach_prefix_steps(), 0);
        assert!(!clamped.requires_detach());
    }
}
