use std::collections::HashSet;
use std::io::Write;

use flate2::Compression;
use flate2::write::GzEncoder;
use rand::Rng;
use rand::RngCore;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::config::{
    FloatRangeConfig, NcaComplexityBand, NcaFamilyConfig, NcaFamilyKind, NcaSerializationConfig,
    UsizeRangeConfig, default_rule_filter_for_band,
};
use crate::stats::SampleStats;

#[derive(Debug, Clone)]
pub struct NcaSample {
    pub family_kind: NcaFamilyKind,
    pub complexity_band: NcaComplexityBand,
    pub width: usize,
    pub height: usize,
    pub state_count: usize,
    pub frames: Vec<Vec<u8>>,
    pub rule_seed: Option<u64>,
    pub complexity_filter_matched: bool,
    pub identity_bias: f32,
    pub temperature: f32,
    pub step_stride: usize,
    pub start_step: usize,
    pub gzip_complexity_ratio: f32,
}

#[derive(Debug, Clone)]
struct NeuralStochasticRule {
    conv3_weights: Vec<f32>,
    conv3_bias: [f32; 4],
    conv1_weights: [f32; 16 * 4],
    conv1_bias: [f32; 16],
    conv2_weights: Vec<f32>,
    conv2_bias: Vec<f32>,
    init_logits: Vec<f32>,
}

pub fn generate_sample(
    family: &NcaFamilyConfig,
    serialization: &NcaSerializationConfig,
    rng: &mut StdRng,
) -> NcaSample {
    let mut sample = match family.kind {
        NcaFamilyKind::NeuralStochastic => {
            generate_neural_stochastic_sample(family, serialization, rng)
        }
        NcaFamilyKind::LifeLikeBinary => generate_life_like_sample(family, serialization, rng),
        NcaFamilyKind::Cyclic => generate_cyclic_sample(family, serialization, rng),
        NcaFamilyKind::NeuralTotalistic => {
            generate_neural_totalistic_sample(family, serialization, rng)
        }
    };
    if sample.gzip_complexity_ratio == 0.0 {
        sample.gzip_complexity_ratio = compute_gzip_complexity_ratio(&sample, serialization);
    }
    sample
}

pub fn serialize_sample(sample: &NcaSample, serialization: &NcaSerializationConfig) -> String {
    let patch_size = serialization.patch_size.max(1);
    let patches_per_row = sample.width / patch_size;
    let patches_per_col = sample.height / patch_size;
    let mut out = String::new();
    if serialization.include_observable_header {
        out.push_str(&format!(
            "<NCA family={} w={} h={} states={} patch={} steps={} dT={} start={} gzip={:.4} rule_seed={} matched={}>\n",
            family_kind_label(sample.family_kind),
            sample.width,
            sample.height,
            sample.state_count,
            patch_size,
            sample.frames.len(),
            sample.step_stride,
            sample.start_step,
            sample.gzip_complexity_ratio,
            sample.rule_seed
                .map(|seed| seed.to_string())
                .unwrap_or_else(|| "none".to_string()),
            sample.complexity_filter_matched
        ));
    }
    for (frame_index, frame) in sample.frames.iter().enumerate() {
        out.push_str(&format!("<F{frame_index:02}>"));
        for patch_y in 0..patches_per_col {
            for patch_x in 0..patches_per_row {
                out.push(' ');
                for dy in 0..patch_size {
                    for dx in 0..patch_size {
                        let x = patch_x * patch_size + dx;
                        let y = patch_y * patch_size + dy;
                        let cell = frame[y * sample.width + x] as usize;
                        out.push(base36_digit(cell));
                    }
                }
            }
        }
        out.push('\n');
    }
    out
}

pub fn patch_token_ids(sample: &NcaSample, serialization: &NcaSerializationConfig) -> Vec<u32> {
    let patch_size = serialization.patch_size.max(1);
    let patches_per_row = sample.width / patch_size;
    let patches_per_col = sample.height / patch_size;
    let mut tokens = Vec::with_capacity(sample.frames.len() * patches_per_row * patches_per_col);
    let radix = sample.state_count as u32;
    for frame in &sample.frames {
        for patch_y in 0..patches_per_col {
            for patch_x in 0..patches_per_row {
                let mut value = 0u32;
                for dy in 0..patch_size {
                    for dx in 0..patch_size {
                        let x = patch_x * patch_size + dx;
                        let y = patch_y * patch_size + dy;
                        let cell = frame[y * sample.width + x] as u32;
                        value = value
                            .checked_mul(radix)
                            .and_then(|acc| acc.checked_add(cell))
                            .expect("patch token id should fit in u32");
                    }
                }
                tokens.push(value);
            }
        }
    }
    tokens
}

pub fn compute_gzip_complexity_ratio(
    sample: &NcaSample,
    serialization: &NcaSerializationConfig,
) -> f32 {
    let patch_tokens = patch_token_ids(sample, serialization);
    if patch_tokens.is_empty() {
        return 0.0;
    }
    let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
    for token in &patch_tokens {
        encoder
            .write_all(&token.to_le_bytes())
            .expect("gzip write should succeed");
    }
    let compressed = encoder.finish().expect("gzip finish should succeed");
    let original_size = patch_tokens.len() * std::mem::size_of::<u32>();
    if original_size == 0 {
        0.0
    } else {
        compressed.len() as f32 / original_size as f32
    }
}

pub fn compute_sample_stats(
    sample: &NcaSample,
    serialization: &NcaSerializationConfig,
) -> SampleStats {
    let patch_size = serialization.patch_size.max(1);
    let patch_count_per_frame = (sample.width / patch_size) * (sample.height / patch_size);
    let patch_tokens = patch_token_ids(sample, serialization);
    let mut entropies = Vec::with_capacity(sample.frames.len());
    let mut active_ratios = Vec::with_capacity(sample.frames.len());
    let mut transition_rates = Vec::with_capacity(sample.frames.len().saturating_sub(1));
    let mut unique_frames = HashSet::new();
    let mut unique_patches = HashSet::new();

    for (frame_index, frame) in sample.frames.iter().enumerate() {
        unique_frames.insert(frame.clone());
        entropies.push(frame_entropy_bits(frame, sample.state_count));
        active_ratios
            .push(frame.iter().filter(|&&cell| cell != 0).count() as f32 / frame.len() as f32);
        for patch in frame_patches(frame, sample.width, sample.height, patch_size) {
            unique_patches.insert(patch);
        }
        if frame_index > 0 {
            let previous = &sample.frames[frame_index - 1];
            let changed = previous
                .iter()
                .zip(frame.iter())
                .filter(|(lhs, rhs)| lhs != rhs)
                .count();
            transition_rates.push(changed as f32 / frame.len() as f32);
        }
    }

    let mean_entropy_bits = mean(&entropies);
    let mean_transition_rate = mean(&transition_rates);
    let active_ratio_mean = mean(&active_ratios);
    let unique_frames_count = unique_frames.len();
    let unique_patch_count = unique_patches.len();
    let frame_uniqueness_ratio = unique_frames_count as f32 / sample.frames.len().max(1) as f32;
    let total_patch_slots = patch_count_per_frame.max(1) * sample.frames.len().max(1);
    let patch_uniqueness_ratio = unique_patch_count as f32 / total_patch_slots as f32;
    let gzip_complexity_ratio = if sample.gzip_complexity_ratio > 0.0 {
        sample.gzip_complexity_ratio
    } else {
        compute_gzip_complexity_ratio(sample, serialization)
    };

    SampleStats {
        grid_width: sample.width,
        grid_height: sample.height,
        steps: sample.frames.len(),
        state_count: sample.state_count,
        patch_count_per_frame,
        patch_token_count: patch_tokens.len(),
        mean_entropy_bits,
        mean_transition_rate,
        active_ratio_mean,
        unique_frames: unique_frames_count,
        unique_patch_count,
        frame_uniqueness_ratio,
        patch_uniqueness_ratio,
        gzip_complexity_ratio,
        complexity_score: (gzip_complexity_ratio * 100.0).clamp(0.0, 100.0),
    }
}

fn generate_neural_stochastic_sample(
    family: &NcaFamilyConfig,
    serialization: &NcaSerializationConfig,
    rng: &mut StdRng,
) -> NcaSample {
    let grid_extent = sample_grid_extent(family, rng);
    let width = grid_extent;
    let height = grid_extent;
    let steps = sample_steps(family, rng).max(2);
    let state_count = sample_state_count(family, rng).clamp(2, 10);
    let step_stride = sample_step_stride(family, rng).max(1);
    let start_step = sample_start_step(family, rng);
    let identity_bias = sample_float(
        family
            .identity_bias
            .unwrap_or(FloatRangeConfig { min: 0.0, max: 0.0 }),
        rng,
    );
    let temperature = sample_float(
        family
            .temperature
            .unwrap_or(FloatRangeConfig { min: 0.0, max: 0.0 }),
        rng,
    );
    let filter = family
        .rule_filter
        .clone()
        .unwrap_or_else(|| default_rule_filter_for_band(family.complexity));
    let lower = filter
        .threshold
        .or_else(|| default_rule_filter_for_band(family.complexity).threshold)
        .unwrap_or(0.0);
    let upper = filter
        .upper_bound
        .or_else(|| default_rule_filter_for_band(family.complexity).upper_bound)
        .unwrap_or(1.0);
    let scoring_examples = filter.scoring_examples.unwrap_or(steps).max(1);
    let target = 0.5 * (lower + upper);

    let mut best_distance = f32::INFINITY;
    let mut best_sample = None;

    for _ in 0..filter.max_attempts.max(1) {
        let rule_seed = rng.next_u64();
        let mut rule_rng = StdRng::seed_from_u64(rule_seed);
        let rule = sample_neural_stochastic_rule(state_count, &mut rule_rng);
        let frames = rollout_neural_stochastic(
            &rule,
            width,
            height,
            scoring_examples,
            start_step,
            step_stride,
            identity_bias,
            temperature,
            rule_seed ^ 0x9E37_79B9_7F4A_7C15,
        );
        let scoring_sample = NcaSample {
            family_kind: family.kind,
            complexity_band: family.complexity,
            width,
            height,
            state_count,
            frames,
            rule_seed: Some(rule_seed),
            complexity_filter_matched: false,
            identity_bias,
            temperature,
            step_stride,
            start_step,
            gzip_complexity_ratio: 0.0,
        };
        let scoring_ratio = compute_gzip_complexity_ratio(&scoring_sample, serialization);
        let matched = (lower..=upper).contains(&scoring_ratio);
        let output_frames = if scoring_examples == steps {
            scoring_sample.frames.clone()
        } else {
            rollout_neural_stochastic(
                &rule,
                width,
                height,
                steps,
                start_step,
                step_stride,
                identity_bias,
                temperature,
                rule_seed ^ 0xA24B_AED4_963E_E407,
            )
        };
        let mut candidate = NcaSample {
            family_kind: family.kind,
            complexity_band: family.complexity,
            width,
            height,
            state_count,
            frames: output_frames,
            rule_seed: Some(rule_seed),
            complexity_filter_matched: matched,
            identity_bias,
            temperature,
            step_stride,
            start_step,
            gzip_complexity_ratio: 0.0,
        };
        candidate.gzip_complexity_ratio = compute_gzip_complexity_ratio(&candidate, serialization);
        if matched {
            return candidate;
        }
        let distance = complexity_distance(scoring_ratio, lower, upper, target);
        if distance < best_distance {
            best_distance = distance;
            best_sample = Some(candidate);
        }
    }

    best_sample.expect("neural stochastic rule search should produce a fallback candidate")
}

fn generate_life_like_sample(
    family: &NcaFamilyConfig,
    serialization: &NcaSerializationConfig,
    rng: &mut StdRng,
) -> NcaSample {
    let patch_size = serialization.patch_size.max(1);
    let mut width = sample_grid_extent(family, rng);
    let mut height = sample_grid_extent(family, rng);
    width = width.div_ceil(patch_size) * patch_size;
    height = height.div_ceil(patch_size) * patch_size;
    let steps = sample_steps(family, rng).max(2);
    let initial = random_binary_grid(width, height, rng);
    let birth = sample_mask(family.complexity, rng, true);
    let survive = sample_mask(family.complexity, rng, false);
    let frames = rollout(
        initial,
        width,
        height,
        steps,
        |current, width, height, next| {
            apply_life_like(current, width, height, next, birth, survive);
        },
    );
    NcaSample {
        family_kind: family.kind,
        complexity_band: family.complexity,
        width,
        height,
        state_count: 2,
        frames,
        rule_seed: None,
        complexity_filter_matched: true,
        identity_bias: 0.0,
        temperature: 0.0,
        step_stride: 1,
        start_step: 0,
        gzip_complexity_ratio: 0.0,
    }
}

fn generate_cyclic_sample(
    family: &NcaFamilyConfig,
    serialization: &NcaSerializationConfig,
    rng: &mut StdRng,
) -> NcaSample {
    let patch_size = serialization.patch_size.max(1);
    let mut width = sample_grid_extent(family, rng);
    let mut height = sample_grid_extent(family, rng);
    width = width.div_ceil(patch_size) * patch_size;
    height = height.div_ceil(patch_size) * patch_size;
    let steps = sample_steps(family, rng).max(2);
    let state_count = sample_state_count(family, rng).clamp(3, 16);
    let threshold = match family.complexity {
        NcaComplexityBand::Simple => rng.gen_range(1..=2),
        NcaComplexityBand::Medium => rng.gen_range(2..=4),
        NcaComplexityBand::Complex => rng.gen_range(3..=5),
    };
    let initial = random_state_grid(width, height, state_count, rng);
    let frames = rollout(
        initial,
        width,
        height,
        steps,
        |current, width, height, next| {
            apply_cyclic(current, width, height, next, state_count as u8, threshold);
        },
    );
    NcaSample {
        family_kind: family.kind,
        complexity_band: family.complexity,
        width,
        height,
        state_count,
        frames,
        rule_seed: None,
        complexity_filter_matched: true,
        identity_bias: 0.0,
        temperature: 0.0,
        step_stride: 1,
        start_step: 0,
        gzip_complexity_ratio: 0.0,
    }
}

fn generate_neural_totalistic_sample(
    family: &NcaFamilyConfig,
    serialization: &NcaSerializationConfig,
    rng: &mut StdRng,
) -> NcaSample {
    let patch_size = serialization.patch_size.max(1);
    let mut width = sample_grid_extent(family, rng);
    let mut height = sample_grid_extent(family, rng);
    width = width.div_ceil(patch_size) * patch_size;
    height = height.div_ceil(patch_size) * patch_size;
    let steps = sample_steps(family, rng).max(2);
    let state_count = sample_state_count(family, rng).clamp(4, 16);
    let rule = sample_neural_totalistic_rule(state_count, family.complexity, rng);
    let initial = random_state_grid(width, height, state_count, rng);
    let frames = rollout(
        initial,
        width,
        height,
        steps,
        |current, width, height, next| {
            apply_neural_totalistic(current, width, height, next, &rule);
        },
    );
    NcaSample {
        family_kind: family.kind,
        complexity_band: family.complexity,
        width,
        height,
        state_count,
        frames,
        rule_seed: None,
        complexity_filter_matched: true,
        identity_bias: 0.0,
        temperature: 0.0,
        step_stride: 1,
        start_step: 0,
        gzip_complexity_ratio: 0.0,
    }
}

fn sample_grid_extent(family: &NcaFamilyConfig, rng: &mut StdRng) -> usize {
    let default = match family.kind {
        NcaFamilyKind::NeuralStochastic => UsizeRangeConfig { min: 12, max: 12 },
        _ => match family.complexity {
            NcaComplexityBand::Simple => UsizeRangeConfig { min: 8, max: 12 },
            NcaComplexityBand::Medium => UsizeRangeConfig { min: 12, max: 16 },
            NcaComplexityBand::Complex => UsizeRangeConfig { min: 16, max: 20 },
        },
    };
    sample_range(family.grid_size.unwrap_or(default), rng)
}

fn sample_steps(family: &NcaFamilyConfig, rng: &mut StdRng) -> usize {
    let default = match family.kind {
        NcaFamilyKind::NeuralStochastic => UsizeRangeConfig { min: 10, max: 10 },
        _ => match family.complexity {
            NcaComplexityBand::Simple => UsizeRangeConfig { min: 8, max: 14 },
            NcaComplexityBand::Medium => UsizeRangeConfig { min: 12, max: 20 },
            NcaComplexityBand::Complex => UsizeRangeConfig { min: 16, max: 28 },
        },
    };
    sample_range(family.steps.unwrap_or(default), rng)
}

fn sample_state_count(family: &NcaFamilyConfig, rng: &mut StdRng) -> usize {
    let default = match family.kind {
        NcaFamilyKind::NeuralStochastic => UsizeRangeConfig { min: 10, max: 10 },
        NcaFamilyKind::LifeLikeBinary => UsizeRangeConfig { min: 2, max: 2 },
        NcaFamilyKind::Cyclic => match family.complexity {
            NcaComplexityBand::Simple => UsizeRangeConfig { min: 3, max: 4 },
            NcaComplexityBand::Medium => UsizeRangeConfig { min: 4, max: 6 },
            NcaComplexityBand::Complex => UsizeRangeConfig { min: 6, max: 8 },
        },
        NcaFamilyKind::NeuralTotalistic => match family.complexity {
            NcaComplexityBand::Simple => UsizeRangeConfig { min: 4, max: 6 },
            NcaComplexityBand::Medium => UsizeRangeConfig { min: 6, max: 10 },
            NcaComplexityBand::Complex => UsizeRangeConfig { min: 10, max: 16 },
        },
    };
    sample_range(family.state_count.unwrap_or(default), rng)
}

fn sample_step_stride(family: &NcaFamilyConfig, rng: &mut StdRng) -> usize {
    let default = match family.kind {
        NcaFamilyKind::NeuralStochastic => UsizeRangeConfig { min: 2, max: 2 },
        _ => UsizeRangeConfig { min: 1, max: 1 },
    };
    sample_range(family.step_stride.unwrap_or(default), rng)
}

fn sample_start_step(family: &NcaFamilyConfig, rng: &mut StdRng) -> usize {
    let default = UsizeRangeConfig { min: 0, max: 0 };
    sample_range(family.start_step.unwrap_or(default), rng)
}

fn sample_range(range: UsizeRangeConfig, rng: &mut StdRng) -> usize {
    if range.min == range.max {
        range.min
    } else {
        rng.gen_range(range.min..=range.max)
    }
}

fn sample_float(range: FloatRangeConfig, rng: &mut StdRng) -> f32 {
    if (range.min - range.max).abs() < f32::EPSILON {
        range.min
    } else {
        rng.gen_range(range.min..=range.max)
    }
}

fn random_binary_grid(width: usize, height: usize, rng: &mut StdRng) -> Vec<u8> {
    let density = rng.gen_range(0.15f32..0.45f32);
    (0..width * height)
        .map(|_| u8::from(rng.gen_bool(density as f64)))
        .collect()
}

fn random_state_grid(width: usize, height: usize, state_count: usize, rng: &mut StdRng) -> Vec<u8> {
    (0..width * height)
        .map(|_| rng.gen_range(0..state_count) as u8)
        .collect()
}

fn sample_mask(band: NcaComplexityBand, rng: &mut StdRng, birth: bool) -> u16 {
    let count = match (band, birth) {
        (NcaComplexityBand::Simple, true) => rng.gen_range(1..=2),
        (NcaComplexityBand::Simple, false) => rng.gen_range(2..=3),
        (NcaComplexityBand::Medium, true) => rng.gen_range(2..=3),
        (NcaComplexityBand::Medium, false) => rng.gen_range(2..=4),
        (NcaComplexityBand::Complex, true) => rng.gen_range(2..=4),
        (NcaComplexityBand::Complex, false) => rng.gen_range(3..=5),
    };
    let mut mask = 0u16;
    while mask.count_ones() < count as u32 {
        mask |= 1 << rng.gen_range(0..=8);
    }
    mask
}

fn rollout<F>(
    mut current: Vec<u8>,
    width: usize,
    height: usize,
    steps: usize,
    mut step_fn: F,
) -> Vec<Vec<u8>>
where
    F: FnMut(&[u8], usize, usize, &mut [u8]),
{
    let mut frames = Vec::with_capacity(steps);
    let mut next = vec![0u8; current.len()];
    frames.push(current.clone());
    for _ in 1..steps {
        step_fn(&current, width, height, &mut next);
        frames.push(next.clone());
        std::mem::swap(&mut current, &mut next);
    }
    frames
}

fn sample_neural_stochastic_rule(state_count: usize, rng: &mut StdRng) -> NeuralStochasticRule {
    let mut conv3_weights = vec![0.0f32; 4 * state_count * 9];
    for value in &mut conv3_weights {
        *value = sample_weight(rng, state_count * 9);
    }
    let mut conv3_bias = [0.0f32; 4];
    for value in &mut conv3_bias {
        *value = sample_weight(rng, state_count * 9);
    }
    let mut conv1_weights = [0.0f32; 16 * 4];
    for value in &mut conv1_weights {
        *value = sample_weight(rng, 4);
    }
    let mut conv1_bias = [0.0f32; 16];
    for value in &mut conv1_bias {
        *value = sample_weight(rng, 4);
    }
    let mut conv2_weights = vec![0.0f32; state_count * 16];
    for value in &mut conv2_weights {
        *value = sample_weight(rng, 16);
    }
    let mut conv2_bias = vec![0.0f32; state_count];
    for value in &mut conv2_bias {
        *value = sample_weight(rng, 16);
    }
    let init_logits = (0..state_count)
        .map(|_| sample_weight(rng, state_count))
        .collect::<Vec<_>>();
    NeuralStochasticRule {
        conv3_weights,
        conv3_bias,
        conv1_weights,
        conv1_bias,
        conv2_weights,
        conv2_bias,
        init_logits,
    }
}

fn rollout_neural_stochastic(
    rule: &NeuralStochasticRule,
    width: usize,
    height: usize,
    sampled_steps: usize,
    start_step: usize,
    step_stride: usize,
    identity_bias: f32,
    temperature: f32,
    rollout_seed: u64,
) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(rollout_seed);
    let mut current = init_neural_stochastic_state(rule, width, height, &mut rng);
    let mut next = vec![0u8; current.len()];
    let mut frames = Vec::with_capacity(sampled_steps);
    let total_iterations = start_step.saturating_add(sampled_steps.saturating_mul(step_stride));
    for step in 0..total_iterations {
        if step >= start_step && (step - start_step) % step_stride == 0 {
            frames.push(current.clone());
            if frames.len() == sampled_steps {
                break;
            }
        }
        apply_neural_stochastic(
            rule,
            &current,
            width,
            height,
            &mut next,
            identity_bias,
            temperature,
            &mut rng,
        );
        std::mem::swap(&mut current, &mut next);
    }
    frames
}

fn init_neural_stochastic_state(
    rule: &NeuralStochasticRule,
    width: usize,
    height: usize,
    rng: &mut StdRng,
) -> Vec<u8> {
    let mut state = Vec::with_capacity(width * height);
    for _ in 0..width * height {
        state.push(sample_categorical(&rule.init_logits, rng) as u8);
    }
    state
}

fn apply_neural_stochastic(
    rule: &NeuralStochasticRule,
    current: &[u8],
    width: usize,
    height: usize,
    next: &mut [u8],
    identity_bias: f32,
    temperature: f32,
    rng: &mut StdRng,
) {
    let state_count = rule.conv2_bias.len();
    let mut conv3 = [0.0f32; 4];
    let mut hidden = [0.0f32; 16];
    let mut logits = vec![0.0f32; state_count];

    for y in 0..height {
        for x in 0..width {
            for out_channel in 0..4 {
                let mut sum = rule.conv3_bias[out_channel];
                for ky in 0..3 {
                    for kx in 0..3 {
                        let nx =
                            ((x as isize + kx as isize - 1).rem_euclid(width as isize)) as usize;
                        let ny =
                            ((y as isize + ky as isize - 1).rem_euclid(height as isize)) as usize;
                        let state = current[ny * width + nx] as usize;
                        let weight_idx = ((out_channel * state_count + state) * 9) + ky * 3 + kx;
                        sum += rule.conv3_weights[weight_idx];
                    }
                }
                conv3[out_channel] = sum;
            }

            for hidden_idx in 0..16 {
                let mut sum = rule.conv1_bias[hidden_idx];
                for in_idx in 0..4 {
                    sum += rule.conv1_weights[hidden_idx * 4 + in_idx] * conv3[in_idx];
                }
                hidden[hidden_idx] = sum.max(0.0);
            }

            let current_state = current[y * width + x] as usize;
            for state_idx in 0..state_count {
                let mut sum = rule.conv2_bias[state_idx];
                for hidden_idx in 0..16 {
                    sum += rule.conv2_weights[state_idx * 16 + hidden_idx] * hidden[hidden_idx];
                }
                if state_idx == current_state {
                    sum += identity_bias;
                }
                logits[state_idx] = sum;
            }

            let next_state = if temperature <= 1.0e-6 {
                argmax(&logits)
            } else {
                let mut scaled = logits.clone();
                for logit in &mut scaled {
                    *logit /= temperature;
                }
                sample_categorical(&scaled, rng)
            };
            next[y * width + x] = next_state as u8;
        }
    }
}

fn sample_weight(rng: &mut StdRng, fan_in: usize) -> f32 {
    let scale = (1.0f32 / fan_in.max(1) as f32).sqrt();
    rng.gen_range(-scale..=scale)
}

fn sample_categorical(logits: &[f32], rng: &mut StdRng) -> usize {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0f32;
    let mut weights = Vec::with_capacity(logits.len());
    for &logit in logits {
        let weight = (logit - max_logit).exp();
        total += weight;
        weights.push(weight);
    }
    if !total.is_finite() || total <= 0.0 {
        return argmax(logits);
    }
    let mut target = rng.gen_range(0.0..total);
    for (index, weight) in weights.into_iter().enumerate() {
        target -= weight;
        if target <= 0.0 {
            return index;
        }
    }
    logits.len().saturating_sub(1)
}

fn argmax(values: &[f32]) -> usize {
    let mut best_index = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in values.iter().enumerate() {
        if value > best_value {
            best_value = value;
            best_index = index;
        }
    }
    best_index
}

fn complexity_distance(value: f32, lower: f32, upper: f32, target: f32) -> f32 {
    if value < lower {
        lower - value
    } else if value > upper {
        value - upper
    } else {
        (value - target).abs()
    }
}

fn apply_life_like(
    current: &[u8],
    width: usize,
    height: usize,
    next: &mut [u8],
    birth_mask: u16,
    survive_mask: u16,
) {
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let alive_neighbors = moore_neighbor_count(current, width, height, x, y, 1);
            let alive = current[idx] != 0;
            next[idx] = if alive {
                u8::from(((survive_mask >> alive_neighbors) & 1) != 0)
            } else {
                u8::from(((birth_mask >> alive_neighbors) & 1) != 0)
            };
        }
    }
}

fn apply_cyclic(
    current: &[u8],
    width: usize,
    height: usize,
    next: &mut [u8],
    state_count: u8,
    threshold: usize,
) {
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let state = current[idx];
            let target = (state + 1) % state_count;
            let matches = neighbor_state_count(current, width, height, x, y, target);
            next[idx] = if matches >= threshold { target } else { state };
        }
    }
}

#[derive(Debug, Clone)]
struct NeuralTotalisticRule {
    weights: Vec<Vec<i32>>,
    biases: Vec<i32>,
    stay_bias: i32,
}

fn sample_neural_totalistic_rule(
    state_count: usize,
    band: NcaComplexityBand,
    rng: &mut StdRng,
) -> NeuralTotalisticRule {
    let magnitude = match band {
        NcaComplexityBand::Simple => 2,
        NcaComplexityBand::Medium => 3,
        NcaComplexityBand::Complex => 4,
    };
    let mut weights = vec![vec![0i32; state_count]; state_count];
    for row in &mut weights {
        for weight in row {
            *weight = rng.gen_range(-magnitude..=magnitude);
        }
    }
    let biases = (0..state_count)
        .map(|_| rng.gen_range(-magnitude..=magnitude))
        .collect::<Vec<_>>();
    let stay_bias = match band {
        NcaComplexityBand::Simple => 2,
        NcaComplexityBand::Medium => 1,
        NcaComplexityBand::Complex => 0,
    };
    NeuralTotalisticRule {
        weights,
        biases,
        stay_bias,
    }
}

fn apply_neural_totalistic(
    current: &[u8],
    width: usize,
    height: usize,
    next: &mut [u8],
    rule: &NeuralTotalisticRule,
) {
    let state_count = rule.weights.len();
    let mut histogram = vec![0usize; state_count];
    for y in 0..height {
        for x in 0..width {
            histogram.fill(0);
            let idx = y * width + x;
            let current_state = current[idx] as usize;
            accumulate_neighbor_histogram(current, width, height, x, y, &mut histogram);

            let mut best_state = current_state;
            let mut best_score = i32::MIN;
            for next_state in 0..state_count {
                let mut score = rule.biases[next_state];
                if next_state == current_state {
                    score += rule.stay_bias;
                }
                for (state_idx, &count) in histogram.iter().enumerate() {
                    score += rule.weights[next_state][state_idx] * count as i32;
                }
                if score > best_score {
                    best_score = score;
                    best_state = next_state;
                }
            }
            next[idx] = best_state as u8;
        }
    }
}

fn moore_neighbor_count(
    current: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    target_state: u8,
) -> usize {
    let mut total = 0usize;
    for dy in -1isize..=1 {
        for dx in -1isize..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = ((x as isize + dx).rem_euclid(width as isize)) as usize;
            let ny = ((y as isize + dy).rem_euclid(height as isize)) as usize;
            total += usize::from(current[ny * width + nx] == target_state);
        }
    }
    total
}

fn neighbor_state_count(
    current: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    target_state: u8,
) -> usize {
    moore_neighbor_count(current, width, height, x, y, target_state)
}

fn accumulate_neighbor_histogram(
    current: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    histogram: &mut [usize],
) {
    for dy in -1isize..=1 {
        for dx in -1isize..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = ((x as isize + dx).rem_euclid(width as isize)) as usize;
            let ny = ((y as isize + dy).rem_euclid(height as isize)) as usize;
            histogram[current[ny * width + nx] as usize] += 1;
        }
    }
}

fn frame_entropy_bits(frame: &[u8], state_count: usize) -> f32 {
    if frame.is_empty() || state_count == 0 {
        return 0.0;
    }
    let mut counts = vec![0usize; state_count];
    for &cell in frame {
        counts[cell as usize] += 1;
    }
    let len = frame.len() as f32;
    counts
        .into_iter()
        .filter(|&count| count > 0)
        .map(|count| {
            let p = count as f32 / len;
            -p * p.log2()
        })
        .sum()
}

fn frame_patches(frame: &[u8], width: usize, height: usize, patch_size: usize) -> Vec<String> {
    let mut patches = Vec::new();
    for patch_y in (0..height).step_by(patch_size) {
        for patch_x in (0..width).step_by(patch_size) {
            let mut patch = String::with_capacity(patch_size * patch_size);
            for dy in 0..patch_size {
                for dx in 0..patch_size {
                    let idx = (patch_y + dy) * width + patch_x + dx;
                    patch.push(base36_digit(frame[idx] as usize));
                }
            }
            patches.push(patch);
        }
    }
    patches
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn base36_digit(value: usize) -> char {
    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    DIGITS[value.min(DIGITS.len().saturating_sub(1))] as char
}

fn family_kind_label(kind: NcaFamilyKind) -> &'static str {
    match kind {
        NcaFamilyKind::NeuralStochastic => "neural_stochastic",
        NcaFamilyKind::LifeLikeBinary => "life_like_binary",
        NcaFamilyKind::Cyclic => "cyclic",
        NcaFamilyKind::NeuralTotalistic => "neural_totalistic",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NcaRuleFilterConfig;

    #[test]
    fn neural_stochastic_generation_is_stable_for_fixed_seed() {
        let family = NcaFamilyConfig {
            kind: NcaFamilyKind::NeuralStochastic,
            weight: 1,
            complexity: NcaComplexityBand::Medium,
            grid_size: Some(UsizeRangeConfig { min: 12, max: 12 }),
            steps: Some(UsizeRangeConfig { min: 10, max: 10 }),
            state_count: Some(UsizeRangeConfig { min: 10, max: 10 }),
            step_stride: Some(UsizeRangeConfig { min: 2, max: 2 }),
            start_step: Some(UsizeRangeConfig { min: 0, max: 0 }),
            identity_bias: Some(FloatRangeConfig { min: 0.0, max: 0.0 }),
            temperature: Some(FloatRangeConfig { min: 0.0, max: 0.0 }),
            rule_filter: Some(default_rule_filter_for_band(NcaComplexityBand::Medium)),
        };
        let serialization = NcaSerializationConfig::default();
        let mut rng_a = StdRng::seed_from_u64(1337);
        let mut rng_b = StdRng::seed_from_u64(1337);
        let sample_a = generate_sample(&family, &serialization, &mut rng_a);
        let sample_b = generate_sample(&family, &serialization, &mut rng_b);
        assert_eq!(
            serialize_sample(&sample_a, &serialization),
            serialize_sample(&sample_b, &serialization)
        );
        assert_eq!(sample_a.rule_seed, sample_b.rule_seed);
    }

    #[test]
    fn gzip_complexity_is_finite_and_bounded() {
        let family = NcaFamilyConfig {
            kind: NcaFamilyKind::NeuralStochastic,
            weight: 1,
            complexity: NcaComplexityBand::Complex,
            grid_size: Some(UsizeRangeConfig { min: 12, max: 12 }),
            steps: Some(UsizeRangeConfig { min: 10, max: 10 }),
            state_count: Some(UsizeRangeConfig { min: 10, max: 10 }),
            step_stride: Some(UsizeRangeConfig { min: 2, max: 2 }),
            start_step: Some(UsizeRangeConfig { min: 0, max: 0 }),
            identity_bias: Some(FloatRangeConfig { min: 0.0, max: 0.0 }),
            temperature: Some(FloatRangeConfig { min: 0.0, max: 0.0 }),
            rule_filter: Some(default_rule_filter_for_band(NcaComplexityBand::Complex)),
        };
        let serialization = NcaSerializationConfig::default();
        let mut rng = StdRng::seed_from_u64(4242);
        let sample = generate_sample(&family, &serialization, &mut rng);
        let stats = compute_sample_stats(&sample, &serialization);
        assert!(stats.mean_entropy_bits.is_finite());
        assert!(stats.mean_transition_rate.is_finite());
        assert!(stats.gzip_complexity_ratio.is_finite());
        assert!((0.0..=1.5).contains(&stats.gzip_complexity_ratio));
        assert!(stats.complexity_score.is_finite());
    }

    #[test]
    fn patch_token_ids_are_bounded_by_state_space() {
        let family = NcaFamilyConfig {
            kind: NcaFamilyKind::NeuralStochastic,
            weight: 1,
            complexity: NcaComplexityBand::Medium,
            grid_size: Some(UsizeRangeConfig { min: 12, max: 12 }),
            steps: Some(UsizeRangeConfig { min: 10, max: 10 }),
            state_count: Some(UsizeRangeConfig { min: 10, max: 10 }),
            step_stride: Some(UsizeRangeConfig { min: 2, max: 2 }),
            start_step: Some(UsizeRangeConfig { min: 0, max: 0 }),
            identity_bias: Some(FloatRangeConfig { min: 0.0, max: 0.0 }),
            temperature: Some(FloatRangeConfig { min: 0.0, max: 0.0 }),
            rule_filter: Some(default_rule_filter_for_band(NcaComplexityBand::Medium)),
        };
        let serialization = NcaSerializationConfig::default();
        let mut rng = StdRng::seed_from_u64(7);
        let sample = generate_sample(&family, &serialization, &mut rng);
        let tokens = patch_token_ids(&sample, &serialization);
        assert!(!tokens.is_empty());
        let patch_vocab = (sample.state_count as u32)
            .pow((serialization.patch_size * serialization.patch_size) as u32);
        assert!(tokens.iter().all(|&token| token < patch_vocab));
    }

    #[test]
    fn neural_stochastic_rule_filter_can_score_shorter_rollouts_than_output_examples() {
        let family = NcaFamilyConfig {
            kind: NcaFamilyKind::NeuralStochastic,
            weight: 1,
            complexity: NcaComplexityBand::Medium,
            grid_size: Some(UsizeRangeConfig { min: 12, max: 12 }),
            steps: Some(UsizeRangeConfig { min: 15, max: 15 }),
            state_count: Some(UsizeRangeConfig { min: 10, max: 10 }),
            step_stride: Some(UsizeRangeConfig { min: 2, max: 2 }),
            start_step: Some(UsizeRangeConfig { min: 0, max: 0 }),
            identity_bias: Some(FloatRangeConfig { min: 0.0, max: 0.0 }),
            temperature: Some(FloatRangeConfig { min: 0.0, max: 0.0 }),
            rule_filter: Some(NcaRuleFilterConfig {
                scoring_examples: Some(10),
                ..default_rule_filter_for_band(NcaComplexityBand::Medium)
            }),
        };
        let serialization = NcaSerializationConfig::default();
        let mut rng = StdRng::seed_from_u64(2026);
        let sample = generate_sample(&family, &serialization, &mut rng);
        assert_eq!(sample.frames.len(), 15);
        assert_eq!(sample.step_stride, 2);
        assert_eq!(sample.start_step, 0);
        assert!(sample.gzip_complexity_ratio.is_finite());
    }
}
