use anyhow::bail;
use burn_dragon_dreamer::{
    DreamerLatentBackend, MovingMnistDreamerTrainConfig, train_moving_mnist,
};
use std::fs;
use std::path::Path;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    ensure_release_build()?;
    let args: Vec<String> = std::env::args().collect();
    let (mut config, offset) = load_config_or_default(&args)?;
    config.artifact_every = config.validate_every;
    if config.artifact_samples == 0 {
        config.artifact_samples = 16;
    }
    if let Some(steps) = args.get(offset) {
        config.steps = steps.parse().unwrap_or(config.steps);
    }
    if let Some(batch_size) = args.get(offset + 1) {
        config.batch_size = batch_size.parse().unwrap_or(config.batch_size);
    }
    if let Some(k_fovea) = args.get(offset + 2) {
        config.model.k_fovea = k_fovea.parse().unwrap_or(config.model.k_fovea);
    }
    if let Some(use_bdh_posterior) = args.get(offset + 3) {
        let enabled = matches!(
            use_bdh_posterior.as_str(),
            "1" | "true" | "True" | "TRUE" | "yes" | "on"
        );
        config.model.use_bdh_posterior = enabled;
        if enabled {
            config.model.latent_backend = DreamerLatentBackend::BdhPooled;
        }
    }
    if let Some(artifact_dir) = args.get(offset + 4) {
        if matches!(artifact_dir.as_str(), "none" | "off" | "false") {
            config.artifact_enabled = false;
            config.artifact_dir = None;
        } else if matches!(artifact_dir.as_str(), "auto" | "default") {
            config.artifact_enabled = true;
            config.artifact_dir = None;
        } else {
            config.artifact_enabled = true;
            config.artifact_dir = Some(PathBuf::from(artifact_dir));
        }
    }
    if let Some(vjepa_checkpoint) = args.get(offset + 5) {
        if !matches!(vjepa_checkpoint.as_str(), "none" | "off" | "false") {
            config.vjepa_checkpoint = Some(PathBuf::from(vjepa_checkpoint));
        }
    }
    if let Some(vjepa_config_paths) = args.get(offset + 6) {
        if !vjepa_config_paths.is_empty()
            && !matches!(vjepa_config_paths.as_str(), "none" | "off" | "false")
        {
            config.vjepa_config_paths = vjepa_config_paths
                .split(',')
                .filter(|entry| !entry.is_empty())
                .map(PathBuf::from)
                .collect();
        }
    }
    if let Some(autogaze_trace_store) = args.get(offset + 7) {
        if !matches!(autogaze_trace_store.as_str(), "none" | "off" | "false") {
            config.autogaze_trace_store = Some(PathBuf::from(autogaze_trace_store));
        }
    }
    if let Some(autogaze_train_trace_store) = args.get(offset + 8) {
        if !matches!(
            autogaze_train_trace_store.as_str(),
            "none" | "off" | "false"
        ) {
            config.autogaze_train_trace_store = Some(PathBuf::from(autogaze_train_trace_store));
        }
    }
    if let Some(autogaze_val_trace_store) = args.get(offset + 9) {
        if !matches!(autogaze_val_trace_store.as_str(), "none" | "off" | "false") {
            config.autogaze_val_trace_store = Some(PathBuf::from(autogaze_val_trace_store));
        }
    }
    if let Some(artifact_future_steps) = args.get(offset + 10) {
        config.artifact_future_steps = artifact_future_steps
            .parse()
            .unwrap_or(config.artifact_future_steps);
    }
    if let Some(latent_backend) = args.get(offset + 11) {
        config.model.latent_backend = match latent_backend.as_str() {
            "transformer" | "transformer_baseline" | "slot_transformer" => {
                DreamerLatentBackend::TransformerBaseline
            }
            "bdh" | "bdh_pooled" => DreamerLatentBackend::BdhPooled,
            "bdh_challenger" | "bdh_slot" | "bdh_slots" => DreamerLatentBackend::BdhChallenger,
            "pooled" | "legacy" => DreamerLatentBackend::Pooled,
            _ => config.model.latent_backend,
        };
    }
    if let Some(slot_grid_size) = args.get(offset + 12) {
        config.model.slot_grid_size = slot_grid_size
            .parse()
            .unwrap_or(config.model.slot_grid_size);
    }
    if let Some(tokenizer_checkpoint) = args.get(offset + 13) {
        if !matches!(tokenizer_checkpoint.as_str(), "none" | "off" | "false") {
            config.tokenizer_checkpoint = Some(PathBuf::from(tokenizer_checkpoint));
        }
    }
    if let Some(freeze_tokenizer) = args.get(offset + 14) {
        config.freeze_tokenizer = matches!(
            freeze_tokenizer.as_str(),
            "1" | "true" | "True" | "TRUE" | "yes" | "on"
        );
    }
    if let Some(dreamer_checkpoint) = args.get(offset + 15) {
        if !matches!(dreamer_checkpoint.as_str(), "none" | "off" | "false") {
            config.dreamer_checkpoint = Some(PathBuf::from(dreamer_checkpoint));
        }
    }
    let summary = train_moving_mnist(config)?;
    println!(
        "moving_mnist_dreamer backend={} initial_valid_total={:.5} final_valid_total={:.5} best_valid_total={:.5} initial_valid_future={:.5} final_valid_future={:.5} best_valid_future={:.5} final_train_total={:.5} final_train_future={:.5} final_valid_recon_current={:.5} final_valid_recon_future={:.5} final_valid_future_psnr={:.2} final_valid_future_fg_iou={:.3} final_valid_future_motion_ratio={:.3} final_valid_future_stop_mean={:.3} final_valid_future_stop_std={:.3} best_future_step={} best_future_total={:.5} best_future_future={:.5} best_future_psnr={:.2} best_future_fg_iou={:.3} best_future_motion_ratio={:.3} best_future_fix_l1={:.3} best_future_quality={:.3} best_future_checkpoint={} best_quality_step={} best_quality_total={:.5} best_quality_future={:.5} best_quality_psnr={:.2} best_quality_fg_iou={:.3} best_quality_motion_ratio={:.3} best_quality_fix_l1={:.3} best_quality_score={:.3} best_quality_checkpoint={} steps={} run_dir={} artifact_dir={}",
        summary.latent_backend,
        summary.initial_valid_total,
        summary.final_valid_total,
        summary.best_valid_total,
        summary.initial_valid_future,
        summary.final_valid_future,
        summary.best_valid_future,
        summary.final_train_total,
        summary.final_train_future,
        summary.final_valid_recon_current,
        summary.final_valid_recon_future,
        summary.final_valid_future_psnr,
        summary.final_valid_future_fg_iou,
        summary.final_valid_future_motion_ratio,
        summary.final_valid_future_stop_mean,
        summary.final_valid_future_stop_std,
        summary.best_future_selection.step,
        summary.best_future_selection.total,
        summary.best_future_selection.future,
        summary.best_future_selection.future_psnr,
        summary.best_future_selection.future_fg_iou,
        summary.best_future_selection.future_motion_ratio,
        summary.best_future_selection.future_fixation_teacher_l1,
        summary.best_future_selection.quality_score,
        summary
            .best_future_selection
            .checkpoint_base
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "none".to_string()),
        summary.best_quality_selection.step,
        summary.best_quality_selection.total,
        summary.best_quality_selection.future,
        summary.best_quality_selection.future_psnr,
        summary.best_quality_selection.future_fg_iou,
        summary.best_quality_selection.future_motion_ratio,
        summary.best_quality_selection.future_fixation_teacher_l1,
        summary.best_quality_selection.quality_score,
        summary
            .best_quality_selection
            .checkpoint_base
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "none".to_string()),
        summary.steps,
        summary
            .run_dir
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "none".to_string()),
        summary
            .artifact_dir
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "none".to_string()),
    );
    Ok(())
}

fn ensure_release_build() -> anyhow::Result<()> {
    if cfg!(debug_assertions) && std::env::var_os("BURN_DRAGON_DREAMER_ALLOW_DEBUG").is_none() {
        bail!(
            "train_moving_mnist must be run with --release for real experiments; set BURN_DRAGON_DREAMER_ALLOW_DEBUG=1 only for local debugging"
        );
    }
    Ok(())
}

fn load_config_or_default(
    args: &[String],
) -> anyhow::Result<(MovingMnistDreamerTrainConfig, usize)> {
    let mut config = MovingMnistDreamerTrainConfig::default();
    let mut offset = 1usize;
    if let Some(first) = args.get(1) {
        let config_path = if first == "--config" {
            offset = 3;
            args.get(2)
        } else if first.ends_with(".toml") {
            offset = 2;
            Some(first)
        } else {
            None
        };
        if let Some(path) = config_path {
            let contents = fs::read_to_string(path)?;
            config = toml::from_str(&contents)?;
            if config.run_root.is_none() {
                config.run_root =
                    Some(Path::new("runs/burn_dragon_dreamer/moving_mnist").to_path_buf());
            }
        }
    }
    Ok((config, offset))
}
