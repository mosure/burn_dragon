use crate::FrameFixationTrace;

#[cfg(test)]
use crate::{FixationPoint, FixationSet};

pub trait AutoGazeTeacher {
    fn trace_clip(
        &self,
        frames: &[f32],
        clip_len: usize,
        channels: usize,
        height: usize,
        width: usize,
        k: usize,
    ) -> FrameFixationTrace;
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct MotionHeuristicAutoGazeTeacher {
    pub motion_weight: f32,
    pub detail_weight: f32,
    pub suppression_radius: usize,
    pub salience_floor: f32,
}

#[cfg(test)]
impl Default for MotionHeuristicAutoGazeTeacher {
    fn default() -> Self {
        Self {
            motion_weight: 0.7,
            detail_weight: 0.3,
            suppression_radius: 4,
            salience_floor: 1.0e-4,
        }
    }
}

#[cfg(test)]
impl MotionHeuristicAutoGazeTeacher {
    fn grayscale_frame(
        &self,
        frames: &[f32],
        frame_idx: usize,
        clip_len: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Vec<f32> {
        let frame_area = height * width;
        let frame_stride = channels * frame_area;
        let base = frame_idx * frame_stride;
        let mut out = vec![0.0; frame_area];
        for channel in 0..channels {
            let channel_base = base + channel * frame_area;
            for idx in 0..frame_area {
                out[idx] += frames.get(channel_base + idx).copied().unwrap_or(0.0);
            }
        }
        let denom = channels.max(1) as f32;
        for value in out.iter_mut() {
            *value /= denom;
        }
        debug_assert!(frame_idx < clip_len);
        out
    }

    fn local_detail(&self, frame: &[f32], height: usize, width: usize) -> Vec<f32> {
        let mut out = vec![0.0; height * width];
        for y in 0..height {
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(height.saturating_sub(1));
            for x in 0..width {
                let x0 = x.saturating_sub(1);
                let x1 = (x + 1).min(width.saturating_sub(1));
                let mut sum = 0.0;
                let mut count: f32 = 0.0;
                for yy in y0..=y1 {
                    for xx in x0..=x1 {
                        sum += frame[yy * width + xx];
                        count += 1.0;
                    }
                }
                let idx = y * width + x;
                out[idx] = (frame[idx] - sum / count.max(1.0)).abs();
            }
        }
        out
    }

    fn saliency_map(
        &self,
        frames: &[f32],
        frame_idx: usize,
        clip_len: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Vec<f32> {
        let current = self.grayscale_frame(frames, frame_idx, clip_len, channels, height, width);
        let previous = if frame_idx == 0 {
            current.clone()
        } else {
            self.grayscale_frame(frames, frame_idx - 1, clip_len, channels, height, width)
        };
        let detail = self.local_detail(&current, height, width);
        current
            .iter()
            .zip(previous.iter())
            .zip(detail.iter())
            .map(|((&cur, &prev), &detail_score)| {
                self.motion_weight * (cur - prev).abs() + self.detail_weight * detail_score
            })
            .collect()
    }

    fn greedily_pick_fixations(
        &self,
        mut saliency: Vec<f32>,
        height: usize,
        width: usize,
        k: usize,
    ) -> FixationSet {
        let total_salience: f32 = saliency.iter().sum();
        let max_salience = saliency
            .iter()
            .copied()
            .fold(0.0_f32, f32::max)
            .max(self.salience_floor);
        let mut points = Vec::with_capacity(k.max(1));
        let radius = self.suppression_radius.max(1);

        for _ in 0..k.max(1) {
            let Some((best_idx, best_score)) = saliency
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            else {
                break;
            };
            let x = best_idx % width.max(1);
            let y = best_idx / width.max(1);
            let norm_x = (x as f32 + 0.5) / width.max(1) as f32;
            let norm_y = (y as f32 + 0.5) / height.max(1) as f32;
            let confidence = (best_score / max_salience).clamp(0.0, 1.0);
            let scale = (0.12 + 0.18 * (1.0 - confidence)).clamp(0.08, 0.35);
            points.push(FixationPoint::new(norm_x, norm_y, scale, confidence));

            let y0 = y.saturating_sub(radius);
            let y1 = (y + radius).min(height.saturating_sub(1));
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius).min(width.saturating_sub(1));
            for yy in y0..=y1 {
                for xx in x0..=x1 {
                    saliency[yy * width + xx] = 0.0;
                }
            }
        }

        let stop_probability = if total_salience <= self.salience_floor {
            1.0
        } else {
            (1.0 - (max_salience / total_salience.max(self.salience_floor))).clamp(0.0, 1.0)
        };
        FixationSet::new(points, stop_probability, k)
    }
}

#[cfg(test)]
impl AutoGazeTeacher for MotionHeuristicAutoGazeTeacher {
    fn trace_clip(
        &self,
        frames: &[f32],
        clip_len: usize,
        channels: usize,
        height: usize,
        width: usize,
        k: usize,
    ) -> FrameFixationTrace {
        let mut per_frame = Vec::with_capacity(clip_len.max(1));
        for frame_idx in 0..clip_len.max(1) {
            let saliency = self.saliency_map(frames, frame_idx, clip_len, channels, height, width);
            per_frame.push(self.greedily_pick_fixations(saliency, height, width, k));
        }
        FrameFixationTrace::new(per_frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_teacher_tracks_bright_motion() {
        let clip_len = 3;
        let channels = 1;
        let height = 8;
        let width = 8;
        let mut frames = vec![0.0; clip_len * channels * height * width];
        for frame_idx in 0..clip_len {
            let x = 1 + frame_idx;
            let y = 2 + frame_idx;
            let idx = frame_idx * height * width + y * width + x;
            frames[idx] = 1.0;
        }

        let teacher = MotionHeuristicAutoGazeTeacher::default();
        let trace = teacher.trace_clip(&frames, clip_len, channels, height, width, 1);
        let first = trace.frames[0].top_point();
        let last = trace.frames[2].top_point();

        assert!(first.x < last.x, "expected fixation to move right");
        assert!(first.y < last.y, "expected fixation to move down");
        assert!(last.confidence > 0.0);
    }
}
