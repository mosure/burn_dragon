use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SampleStats {
    pub grid_width: usize,
    pub grid_height: usize,
    pub steps: usize,
    pub state_count: usize,
    pub patch_count_per_frame: usize,
    pub patch_token_count: usize,
    pub mean_entropy_bits: f32,
    pub mean_transition_rate: f32,
    pub active_ratio_mean: f32,
    pub unique_frames: usize,
    pub unique_patch_count: usize,
    pub frame_uniqueness_ratio: f32,
    pub patch_uniqueness_ratio: f32,
    pub gzip_complexity_ratio: f32,
    pub complexity_score: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ComplexityHistogramBin {
    pub lower: f32,
    pub upper: f32,
    pub count: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct CorpusStats {
    pub total_samples: usize,
    pub train_samples: usize,
    pub validation_samples: usize,
    pub total_token_count: usize,
    pub mean_token_count: f32,
    pub mean_entropy_bits: f32,
    pub mean_transition_rate: f32,
    pub mean_active_ratio: f32,
    pub mean_gzip_complexity_ratio: f32,
    pub min_gzip_complexity_ratio: f32,
    pub max_gzip_complexity_ratio: f32,
    pub mean_complexity_score: f32,
    pub min_complexity_score: f32,
    pub max_complexity_score: f32,
    pub family_counts: std::collections::BTreeMap<String, usize>,
    pub complexity_histogram: Vec<ComplexityHistogramBin>,
}

pub fn build_complexity_histogram(scores: &[f32]) -> Vec<ComplexityHistogramBin> {
    let mut bins = (0..10)
        .map(|index| ComplexityHistogramBin {
            lower: index as f32 * 10.0,
            upper: (index + 1) as f32 * 10.0,
            count: 0,
        })
        .collect::<Vec<_>>();
    for &score in scores {
        let clamped = score.clamp(0.0, 100.0);
        let bin_index = ((clamped / 10.0).floor() as usize).min(bins.len().saturating_sub(1));
        bins[bin_index].count += 1;
    }
    bins
}
