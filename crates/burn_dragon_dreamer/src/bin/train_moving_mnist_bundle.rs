use anyhow::bail;
use burn_dragon_dreamer::{MovingMnistDreamerBundleConfig, train_moving_mnist_bundle};
use std::fs;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    ensure_release_build()?;
    let args: Vec<String> = std::env::args().collect();
    let config_path = match args.get(1) {
        Some(path) if path.ends_with(".toml") => path,
        Some(flag) if flag == "--config" => args
            .get(2)
            .ok_or_else(|| anyhow::anyhow!("missing bundle config path after --config"))?,
        _ => bail!("usage: train_moving_mnist_bundle <bundle.toml>"),
    };
    let contents = fs::read_to_string(config_path)?;
    let mut config: MovingMnistDreamerBundleConfig = toml::from_str(&contents)?;
    if config.run_root.is_none() {
        config.run_root = Some(Path::new("runs/burn_dragon_dreamer/bundles").to_path_buf());
    }

    let summary = train_moving_mnist_bundle(config)?;
    println!(
        "moving_mnist_bundle name={} tokenizer_best_total={:.5} tokenizer_recon={:.5} tokenizer_checkpoint={} dynamics_backend={} dynamics_best_future_step={} dynamics_best_future_psnr={:.2} dynamics_best_future_iou={:.3} dynamics_best_future_motion_ratio={:.3} dynamics_best_quality_step={} dynamics_best_quality_psnr={:.2} dynamics_best_quality_iou={:.3} dynamics_best_quality_motion_ratio={:.3} dynamics_run_dir={}",
        summary.name,
        summary.tokenizer.best_valid_total,
        summary.tokenizer.final_valid_recon_current,
        summary.tokenizer_checkpoint.display(),
        summary.dynamics.latent_backend,
        summary.dynamics.best_future_selection.step,
        summary.dynamics.best_future_selection.future_psnr,
        summary.dynamics.best_future_selection.future_fg_iou,
        summary.dynamics.best_future_selection.future_motion_ratio,
        summary.dynamics.best_quality_selection.step,
        summary.dynamics.best_quality_selection.future_psnr,
        summary.dynamics.best_quality_selection.future_fg_iou,
        summary.dynamics.best_quality_selection.future_motion_ratio,
        summary
            .dynamics
            .run_dir
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "none".to_string()),
    );
    Ok(())
}

fn ensure_release_build() -> anyhow::Result<()> {
    if cfg!(debug_assertions) && std::env::var_os("BURN_DRAGON_DREAMER_ALLOW_DEBUG").is_none() {
        bail!(
            "train_moving_mnist_bundle must be run with --release for real experiments; set BURN_DRAGON_DREAMER_ALLOW_DEBUG=1 only for local debugging"
        );
    }
    Ok(())
}
