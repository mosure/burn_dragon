use std::path::PathBuf;

use burn::module::Module;
use burn_dragon::vision::{
    VisionDragon, VisionPatchEmbedMode, VisionTrainingConfig, VisionTrmGraphConfig,
    load_vision_training_config,
};
use burn_ndarray::NdArray;
use clap::Parser;
use serde::Serialize;

type StatsBackend = NdArray<f32>;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, required = true)]
    config: Vec<PathBuf>,
}

#[derive(Serialize)]
struct VisionModelStatsReport {
    config: Vec<PathBuf>,
    image_size: usize,
    patch_size: usize,
    patch_tokens: usize,
    total_tokens: usize,
    embed_dim: usize,
    projection_dim: usize,
    projection_hidden_dim: usize,
    steps: usize,
    n_head: usize,
    latent_per_head: usize,
    latent_total: usize,
    params: usize,
    patch_embed_macs: u128,
    x_projection_macs: u128,
    attention_qk_macs: u128,
    attention_mix_macs: u128,
    y_projection_macs: u128,
    tail_decode_macs: u128,
    recurrent_step_macs: u128,
    projection_head_macs: u128,
    total_forward_macs_by_step: Vec<u128>,
}

fn main() {
    let args = Args::parse();
    let config = load_vision_training_config(&args.config)
        .unwrap_or_else(|err| panic!("failed to load config overlays {:?}: {err}", args.config));
    let report = build_report(&config, args.config);
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize stats")
    );
}

fn build_report(
    config: &VisionTrainingConfig,
    config_paths: Vec<PathBuf>,
) -> VisionModelStatsReport {
    let device = <StatsBackend as burn::tensor::backend::Backend>::Device::default();
    let vision = config.vision.build();
    let model = VisionDragon::<StatsBackend>::new(vision.clone(), &device);
    let grid = vision.image_size.div_ceil(vision.patch_size);
    let patch_tokens = grid * grid;
    let total_tokens = patch_tokens + usize::from(vision.use_cls_token);
    let latent_per_head = vision.latent_per_head();
    let latent_total = vision.latent_total();

    let x_projection_macs = total_tokens as u128 * vision.embed_dim as u128 * latent_total as u128;
    let attention_qk_macs = total_tokens as u128 * total_tokens as u128 * latent_total as u128;
    let attention_mix_macs = vision.n_head as u128
        * total_tokens as u128
        * total_tokens as u128
        * vision.embed_dim as u128;
    let y_projection_macs = vision.n_head as u128
        * total_tokens as u128
        * vision.embed_dim as u128
        * latent_per_head as u128;
    let tail_decode_macs = total_tokens as u128 * latent_total as u128 * vision.embed_dim as u128;
    let recurrent_step_macs = x_projection_macs
        + attention_qk_macs
        + attention_mix_macs
        + y_projection_macs
        + tail_decode_macs;
    let projection_head_macs = total_tokens as u128
        * (vision.embed_dim as u128 * vision.projection_hidden_dim as u128
            + vision.projection_hidden_dim as u128 * vision.projection_dim as u128);
    let patch_embed_macs = estimate_patch_embed_macs(
        vision.image_size,
        vision.patch_size,
        vision.in_channels,
        vision.embed_dim,
        vision.patch_embed_mode,
        &vision.trm_graph,
    );
    let total_forward_macs_by_step = (1..=vision.steps.max(1))
        .map(|step| patch_embed_macs + recurrent_step_macs * step as u128 + projection_head_macs)
        .collect();

    VisionModelStatsReport {
        config: config_paths,
        image_size: vision.image_size,
        patch_size: vision.patch_size,
        patch_tokens,
        total_tokens,
        embed_dim: vision.embed_dim,
        projection_dim: vision.projection_dim,
        projection_hidden_dim: vision.projection_hidden_dim,
        steps: vision.steps,
        n_head: vision.n_head,
        latent_per_head,
        latent_total,
        params: model.num_params(),
        patch_embed_macs,
        x_projection_macs,
        attention_qk_macs,
        attention_mix_macs,
        y_projection_macs,
        tail_decode_macs,
        recurrent_step_macs,
        projection_head_macs,
        total_forward_macs_by_step,
    }
}

fn estimate_patch_embed_macs(
    image_size: usize,
    patch_size: usize,
    in_channels: usize,
    embed_dim: usize,
    mode: VisionPatchEmbedMode,
    _trm_graph: &VisionTrmGraphConfig,
) -> u128 {
    match mode {
        VisionPatchEmbedMode::Linear => {
            let grid = image_size.div_ceil(patch_size);
            let tokens = grid * grid;
            tokens as u128 * (patch_size * patch_size * in_channels) as u128 * embed_dim as u128
        }
        VisionPatchEmbedMode::Identity => 0,
        VisionPatchEmbedMode::Conv | VisionPatchEmbedMode::ConvNext => {
            let hidden_dim = (embed_dim / 2).max(16).min(embed_dim.max(1));
            let mut h = image_size.max(1);
            let mut w = image_size.max(1);
            let mut channels = in_channels.max(1);
            let mut macs = 0u128;

            if matches!(mode, VisionPatchEmbedMode::ConvNext) {
                macs += patch_embed_stage_macs(h, w, channels, hidden_dim, 1, 2);
                channels = hidden_dim;
            }

            for stride in patch_downsample_strides(patch_size) {
                macs += patch_embed_stage_macs(h, w, channels, hidden_dim, stride, 1);
                h = h.div_ceil(stride);
                w = w.div_ceil(stride);
                channels = hidden_dim;
            }

            macs + (h * w * channels * embed_dim) as u128
        }
    }
}

fn patch_embed_stage_macs(
    h: usize,
    w: usize,
    in_channels: usize,
    out_channels: usize,
    stride: usize,
    blocks: usize,
) -> u128 {
    let kernel = if stride <= 1 { 3 } else { stride };
    let out_h = h.div_ceil(stride.max(1));
    let out_w = w.div_ceil(stride.max(1));
    let mut macs = (out_h * out_w * in_channels * out_channels * kernel * kernel) as u128;
    for _ in 0..blocks.max(1) {
        macs += (out_h * out_w * out_channels * 3 * 3) as u128;
        macs += (out_h * out_w * out_channels * (out_channels * 4)) as u128;
        macs += (out_h * out_w * (out_channels * 4) * out_channels) as u128;
    }
    macs
}

fn patch_downsample_strides(patch_size: usize) -> Vec<usize> {
    let mut remaining = patch_size.max(1);
    let mut strides = Vec::new();
    while remaining > 1 {
        if remaining % 4 == 0 {
            strides.push(4);
            remaining /= 4;
        } else if remaining % 2 == 0 {
            strides.push(2);
            remaining /= 2;
        } else {
            strides.push(remaining);
            remaining = 1;
        }
    }
    strides
}
