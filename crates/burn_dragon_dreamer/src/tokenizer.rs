use crate::checkpoint::{checkpoint_base, load_module_checkpoint, save_module_checkpoint};
use crate::moving_mnist::{
    CachedMovingMnistSplit, TrainBackend, load_autogaze_source, load_crop_teacher_source,
    load_vjepa_source, prepare_run_dir, sample_indices, scalar,
};
use crate::{DragonDreamer, MovingMnistDreamerTrainConfig};
use anyhow::{Context, Result};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn_cuda::CudaDevice;
use burn_dragon_vision::{
    MovingMnistSplit, MovingMnistVideoDataset, MovingMnistVideoDatasetConfig, VisionNormalize,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MovingMnistTokenizerRunSummary {
    pub initial_valid_total: f32,
    pub final_valid_total: f32,
    pub best_valid_total: f32,
    pub final_train_total: f32,
    pub final_valid_current: f32,
    pub final_valid_query: f32,
    pub final_valid_gaze: f32,
    pub final_valid_tokenizer: f32,
    pub final_valid_tokenizer_recon: f32,
    pub final_valid_recon_current: f32,
    pub steps: usize,
    pub run_dir: Option<PathBuf>,
    pub checkpoint_base: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Default)]
struct TokenizerLossSnapshot {
    total: f32,
    current: f32,
    query: f32,
    gaze: f32,
    tokenizer: f32,
    tokenizer_recon: f32,
    recon_current: f32,
}

impl TokenizerLossSnapshot {
    fn add_forward(&mut self, forward: &crate::DreamerForward<TrainBackend>) {
        self.total += scalar(&forward.total);
        self.current += scalar(&forward.current);
        self.query += scalar(&forward.query);
        self.gaze += scalar(&forward.gaze);
        self.tokenizer += scalar(&forward.tokenizer);
        self.tokenizer_recon += scalar(&forward.tokenizer_recon);
        self.recon_current += scalar(&forward.recon_current);
    }

    fn div_scalar(mut self, denom: f32) -> Self {
        let denom = denom.max(1.0);
        self.total /= denom;
        self.current /= denom;
        self.query /= denom;
        self.gaze /= denom;
        self.tokenizer /= denom;
        self.tokenizer_recon /= denom;
        self.recon_current /= denom;
        self
    }
}

pub fn train_moving_mnist_tokenizer(
    mut config: MovingMnistDreamerTrainConfig,
) -> Result<MovingMnistTokenizerRunSummary> {
    let device = std::panic::catch_unwind(std::panic::AssertUnwindSafe(CudaDevice::default))
        .map_err(|_| anyhow::anyhow!("CUDA device initialization failed"))?;
    let run_dir = prepare_run_dir(&config)?;
    if let Some(run_dir) = run_dir.as_ref() {
        fs::write(
            run_dir.join("config.json"),
            serde_json::to_vec_pretty(&config).context("serialize tokenizer config")?,
        )
        .with_context(|| {
            format!(
                "write tokenizer config {}",
                run_dir.join("config.json").display()
            )
        })?;
    }

    let sequence_len = (config.context_len + config.target_len).max(2);
    let train_dataset = MovingMnistVideoDataset::new_from_mnist(MovingMnistVideoDatasetConfig {
        split: MovingMnistSplit::Train,
        frame_size: config.model.frame_size,
        digit_size: 12,
        in_channels: config.model.channels,
        context_len: sequence_len.saturating_sub(1),
        target_len: 1,
        extra_future_frames: 0,
        frame_stride: 1,
        max_records: Some(256),
        normalize: VisionNormalize::new([0.5; 3], [0.5; 3]),
        min_velocity: config.min_velocity,
        max_velocity: config.max_velocity,
        seed: config.train_seed,
    })?;
    let valid_dataset = MovingMnistVideoDataset::new_from_mnist(MovingMnistVideoDatasetConfig {
        split: MovingMnistSplit::Val,
        frame_size: config.model.frame_size,
        digit_size: 12,
        in_channels: config.model.channels,
        context_len: sequence_len.saturating_sub(1),
        target_len: 1,
        extra_future_frames: 0,
        frame_stride: 1,
        max_records: Some(128),
        normalize: VisionNormalize::new([0.5; 3], [0.5; 3]),
        min_velocity: config.min_velocity,
        max_velocity: config.max_velocity,
        seed: config.val_seed,
    })?;

    let train_teacher = load_autogaze_source(&config, MovingMnistSplit::Train)?;
    let valid_teacher = load_autogaze_source(&config, MovingMnistSplit::Val)?;
    let global_teacher = load_vjepa_source(&mut config, &device)?;
    let crop_teacher = load_crop_teacher_source(&mut config, &device)?;
    let train_cache = CachedMovingMnistSplit::build(
        &train_dataset,
        &global_teacher,
        &crop_teacher,
        &train_teacher,
        &config,
        &device,
    );
    let valid_cache = CachedMovingMnistSplit::build(
        &valid_dataset,
        &global_teacher,
        &crop_teacher,
        &valid_teacher,
        &config,
        &device,
    );

    let mut model = DragonDreamer::<TrainBackend>::new(config.model.clone(), &device);
    if let Some(checkpoint) = config.tokenizer_checkpoint.as_ref() {
        model = load_module_checkpoint(model, checkpoint, &device)?;
    }
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(config.weight_decay)
        .init::<TrainBackend, DragonDreamer<TrainBackend>>();

    let initial_valid =
        evaluate_tokenizer_validation(&model, &valid_cache, &config, config.valid_batches);
    let mut best_valid_total = initial_valid.total;
    let mut final_train_total = initial_valid.total;
    let checkpoint_dir = run_dir.as_ref().map(|path| path.join("checkpoint"));
    if let Some(dir) = checkpoint_dir.as_ref() {
        fs::create_dir_all(dir)
            .with_context(|| format!("create tokenizer checkpoint dir {}", dir.display()))?;
    }
    let mut best_checkpoint_base = None;

    for step in 0..config.steps {
        let indices = sample_indices(
            train_cache.len(),
            config.batch_size,
            config
                .train_seed
                .wrapping_add((step as u64).wrapping_mul(0xA24B_AED4_963E_E407)),
        );
        let batch = train_cache.batch(&indices);
        let forward = model.forward_tokenizer_pretrain(
            batch.clip_frames,
            &batch.traces,
            batch.teacher_features,
            batch.crop_teacher_features,
        );
        final_train_total = scalar(&forward.total);
        let grads = GradientsParams::from_grads(forward.total.backward(), &model);
        model = optimizer.step(config.learning_rate, model, grads);

        if (step + 1) % config.log_every.max(1) == 0 {
            println!(
                "tokenizer step={} train_total={:.5} tokenizer={:.5} tok_recon={:.5} recon_cur={:.5}",
                step + 1,
                final_train_total,
                scalar(&forward.tokenizer),
                scalar(&forward.tokenizer_recon),
                scalar(&forward.recon_current),
            );
        }

        if (step + 1) % config.validate_every.max(1) == 0 || step + 1 == config.steps {
            let valid =
                evaluate_tokenizer_validation(&model, &valid_cache, &config, config.valid_batches);
            println!(
                "tokenizer valid step={} total={:.5} current={:.5} query={:.5} gaze={:.5} tok={:.5} tok_recon={:.5} recon_cur={:.5}",
                step + 1,
                valid.total,
                valid.current,
                valid.query,
                valid.gaze,
                valid.tokenizer,
                valid.tokenizer_recon,
                valid.recon_current,
            );
            if valid.total <= best_valid_total {
                best_valid_total = valid.total;
                if let Some(dir) = checkpoint_dir.as_ref() {
                    let base = checkpoint_base(dir, "tokenizer-best");
                    save_module_checkpoint::<TrainBackend, _>(&model, &base)?;
                    best_checkpoint_base = Some(base);
                }
            }
        }
    }

    let final_valid =
        evaluate_tokenizer_validation(&model, &valid_cache, &config, config.valid_batches);
    if let Some(dir) = checkpoint_dir.as_ref() {
        let final_base = checkpoint_base(dir, "tokenizer-final");
        save_module_checkpoint::<TrainBackend, _>(&model, &final_base)?;
        if best_checkpoint_base.is_none() {
            best_checkpoint_base = Some(final_base);
        }
    }

    Ok(MovingMnistTokenizerRunSummary {
        initial_valid_total: initial_valid.total,
        final_valid_total: final_valid.total,
        best_valid_total,
        final_train_total,
        final_valid_current: final_valid.current,
        final_valid_query: final_valid.query,
        final_valid_gaze: final_valid.gaze,
        final_valid_tokenizer: final_valid.tokenizer,
        final_valid_tokenizer_recon: final_valid.tokenizer_recon,
        final_valid_recon_current: final_valid.recon_current,
        steps: config.steps,
        run_dir,
        checkpoint_base: best_checkpoint_base,
    })
}

fn evaluate_tokenizer_validation(
    model: &DragonDreamer<TrainBackend>,
    dataset: &CachedMovingMnistSplit,
    config: &MovingMnistDreamerTrainConfig,
    batches: usize,
) -> TokenizerLossSnapshot {
    let mut totals = TokenizerLossSnapshot::default();
    let count = batches.max(1);
    for batch_idx in 0..count {
        let indices = sample_indices(
            dataset.len(),
            config.batch_size,
            config
                .val_seed
                .wrapping_add((batch_idx as u64).wrapping_mul(0x517C_C1B7_2722_0A95)),
        );
        let batch = dataset.batch(&indices);
        let forward = model.forward_tokenizer_pretrain(
            batch.clip_frames,
            &batch.traces,
            batch.teacher_features,
            batch.crop_teacher_features,
        );
        totals.add_forward(&forward);
    }
    totals.div_scalar(count as f32)
}
