use crate::checkpoint::load_module_checkpoint;
use crate::data::{CachedSequenceSplit, sample_indices};
use crate::run::{prepare_configured_run, save_named_checkpoint, should_run_step};
use crate::runtime::{
    Backend, TrainBackend, cuda_device, train_to_runtime_tensor3, train_to_runtime_tensor5,
};
use crate::tasks::moving_mnist::{
    build_cached_moving_mnist_split, build_moving_mnist_tokenizer_datasets, load_autogaze_source,
    load_crop_teacher_source, load_vjepa_source, uses_crop_teacher, uses_global_teacher,
};
use crate::{DragonDreamer, MovingMnistDreamerTrainConfig};
use anyhow::Result;
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn_dragon_vision::MovingMnistSplit;
use serde::{Deserialize, Serialize};
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
    fn add_values(&mut self, values: &[f32]) {
        self.total += values[0];
        self.current += values[1];
        self.query += values[2];
        self.gaze += values[3];
        self.tokenizer += values[4];
        self.tokenizer_recon += values[5];
        self.recon_current += values[6];
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

fn tokenizer_forward_scalars<B: burn::tensor::backend::Backend>(
    forward: &crate::DreamerForward<B>,
) -> Vec<f32> {
    burn::tensor::Tensor::cat(
        vec![
            forward.total.clone(),
            forward.current.clone(),
            forward.query.clone(),
            forward.gaze.clone(),
            forward.tokenizer.clone(),
            forward.tokenizer_recon.clone(),
            forward.recon_current.clone(),
        ],
        0,
    )
    .detach()
    .to_data()
    .convert::<f32>()
    .into_vec::<f32>()
    .expect("tokenizer scalar pack")
}

pub fn train_moving_mnist_tokenizer(
    mut config: MovingMnistDreamerTrainConfig,
) -> Result<MovingMnistTokenizerRunSummary> {
    let device = cuda_device()?;
    let workspace = prepare_configured_run(config.run_root.as_deref(), &config, "tokenizer")?;
    let run_dir = workspace.run_dir.clone();
    let checkpoint_dir = workspace.checkpoint_dir.clone();

    let datasets = build_moving_mnist_tokenizer_datasets(&config)?;
    let train_dataset = datasets.train;
    let valid_dataset = datasets.valid;

    let train_teacher = load_autogaze_source(&config, MovingMnistSplit::Train, &device)?;
    let valid_teacher = load_autogaze_source(&config, MovingMnistSplit::Val, &device)?;
    let use_global_teacher = uses_global_teacher(&config);
    let use_internal_state_targets = config.model.passive_full_frame
        && (config.model.current_loss_weight > 0.0 || config.model.future_loss_weight > 0.0);
    let use_crop_teacher = uses_crop_teacher(&config);
    if use_internal_state_targets {
        config.model.teacher_dim = config.model.latent_dim.max(1);
    } else if !use_global_teacher {
        config.model.teacher_dim = 1;
    }
    if !use_crop_teacher {
        config.model.crop_teacher_dim = 1;
    }
    let train_global_teacher = if use_global_teacher {
        Some(load_vjepa_source(
            &mut config,
            MovingMnistSplit::Train,
            &device,
        )?)
    } else {
        None
    };
    let valid_global_teacher = if use_global_teacher {
        Some(load_vjepa_source(
            &mut config,
            MovingMnistSplit::Val,
            &device,
        )?)
    } else {
        None
    };
    let train_crop_teacher = if use_crop_teacher {
        Some(load_crop_teacher_source(
            &mut config,
            MovingMnistSplit::Train,
            &device,
        )?)
    } else {
        None
    };
    let valid_crop_teacher = if use_crop_teacher {
        Some(load_crop_teacher_source(
            &mut config,
            MovingMnistSplit::Val,
            &device,
        )?)
    } else {
        None
    };
    let train_cache = build_cached_moving_mnist_split(
        &train_dataset,
        &config,
        &train_teacher,
        train_global_teacher.as_ref(),
        train_crop_teacher.as_ref(),
        &device,
    );
    let valid_cache = build_cached_moving_mnist_split(
        &valid_dataset,
        &config,
        &valid_teacher,
        valid_global_teacher.as_ref(),
        valid_crop_teacher.as_ref(),
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
        evaluate_tokenizer_validation(&model, &valid_cache, &config, config.valid_batches, &device);
    let mut best_valid_total = initial_valid.total;
    let mut final_train_total = initial_valid.total;
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
        let clip_frames = batch.clip_frames;
        let actions = batch.actions;
        let traces = batch.traces;
        let teacher_features = batch.teacher_features;
        let crop_teacher_features = batch.crop_teacher_features;
        let forward = model.forward_tokenizer_pretrain(
            clip_frames,
            &traces,
            actions,
            teacher_features,
            crop_teacher_features,
        );
        let should_read_train_scalars = should_run_step(step, config.steps, config.log_every)
            || should_run_step(step, config.steps, config.validate_every);
        let train_scalars = should_read_train_scalars.then(|| tokenizer_forward_scalars(&forward));
        if let Some(values) = train_scalars.as_ref() {
            final_train_total = values[0];
        }
        let grads = GradientsParams::from_grads(forward.total.backward(), &model);
        model = optimizer.step(config.learning_rate, model, grads);

        if should_run_step(step, config.steps, config.log_every) {
            let values = train_scalars
                .as_ref()
                .expect("log step should have tokenizer scalar pack");
            println!(
                "tokenizer step={} train_total={:.5} tokenizer={:.5} tok_recon={:.5} recon_cur={:.5}",
                step + 1,
                final_train_total,
                values[4],
                values[5],
                values[6],
            );
        }

        if should_run_step(step, config.steps, config.validate_every) {
            let valid = evaluate_tokenizer_validation(
                &model,
                &valid_cache,
                &config,
                config.valid_batches,
                &device,
            );
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
                best_checkpoint_base = save_named_checkpoint::<TrainBackend, _>(
                    &model,
                    checkpoint_dir.as_deref(),
                    "tokenizer-best",
                )?;
            }
        }
    }

    let final_valid =
        evaluate_tokenizer_validation(&model, &valid_cache, &config, config.valid_batches, &device);
    if let Some(final_base) = save_named_checkpoint::<TrainBackend, _>(
        &model,
        checkpoint_dir.as_deref(),
        "tokenizer-final",
    )? {
        if best_checkpoint_base.is_none() {
            best_checkpoint_base = Some(final_base.clone());
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
    dataset: &CachedSequenceSplit<TrainBackend, burn_autogaze::FrameFixationTrace>,
    config: &MovingMnistDreamerTrainConfig,
    batches: usize,
    device: &<TrainBackend as burn::tensor::backend::Backend>::Device,
) -> TokenizerLossSnapshot {
    let mut totals = TokenizerLossSnapshot::default();
    let count = batches.max(1);
    let runtime_model: DragonDreamer<Backend> = model.valid();
    for batch_idx in 0..count {
        let indices = sample_indices(
            dataset.len(),
            config.batch_size,
            config
                .val_seed
                .wrapping_add((batch_idx as u64).wrapping_mul(0x517C_C1B7_2722_0A95)),
        );
        let batch = dataset.batch(&indices);
        let clip_frames = train_to_runtime_tensor5(batch.clip_frames, device);
        let actions = batch
            .actions
            .map(|tensor| train_to_runtime_tensor3(tensor, device));
        let traces = batch.traces;
        let teacher_features = train_to_runtime_tensor3(batch.teacher_features, device);
        let crop_teacher_features = train_to_runtime_tensor3(batch.crop_teacher_features, device);
        let forward = runtime_model.forward_tokenizer_pretrain(
            clip_frames,
            &traces,
            actions,
            teacher_features,
            crop_teacher_features,
        );
        totals.add_values(&tokenizer_forward_scalars(&forward));
    }
    totals.div_scalar(count as f32)
}
