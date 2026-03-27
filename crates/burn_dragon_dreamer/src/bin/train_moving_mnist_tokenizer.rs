use anyhow::bail;
use burn_dragon_dreamer::{MovingMnistDreamerTrainConfig, train_moving_mnist_tokenizer};
use std::fs;
use std::path::Path;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    ensure_release_build()?;
    let args: Vec<String> = std::env::args().collect();
    let (mut config, offset) = load_config_or_default(&args)?;
    if let Some(steps) = args.get(offset) {
        config.steps = steps.parse().unwrap_or(config.steps);
    }
    if let Some(batch_size) = args.get(offset + 1) {
        config.batch_size = batch_size.parse().unwrap_or(config.batch_size);
    }
    if let Some(k_fovea) = args.get(offset + 2) {
        config.model.k_fovea = k_fovea.parse().unwrap_or(config.model.k_fovea);
    }
    if let Some(tokenizer_checkpoint) = args.get(offset + 3) {
        if !matches!(tokenizer_checkpoint.as_str(), "none" | "off" | "false") {
            config.tokenizer_checkpoint = Some(PathBuf::from(tokenizer_checkpoint));
        }
    }

    let summary = train_moving_mnist_tokenizer(config)?;
    println!(
        "moving_mnist_tokenizer initial_valid_total={:.5} final_valid_total={:.5} best_valid_total={:.5} final_train_total={:.5} final_valid_current={:.5} final_valid_query={:.5} final_valid_gaze={:.5} final_valid_tokenizer={:.5} final_valid_tokenizer_recon={:.5} final_valid_recon_current={:.5} steps={} run_dir={} checkpoint_base={}",
        summary.initial_valid_total,
        summary.final_valid_total,
        summary.best_valid_total,
        summary.final_train_total,
        summary.final_valid_current,
        summary.final_valid_query,
        summary.final_valid_gaze,
        summary.final_valid_tokenizer,
        summary.final_valid_tokenizer_recon,
        summary.final_valid_recon_current,
        summary.steps,
        summary
            .run_dir
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "none".to_string()),
        summary
            .checkpoint_base
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "none".to_string()),
    );
    Ok(())
}

fn ensure_release_build() -> anyhow::Result<()> {
    if cfg!(debug_assertions) && std::env::var_os("BURN_DRAGON_DREAMER_ALLOW_DEBUG").is_none() {
        bail!(
            "train_moving_mnist_tokenizer must be run with --release for real experiments; set BURN_DRAGON_DREAMER_ALLOW_DEBUG=1 only for local debugging"
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
                config.run_root = Some(
                    Path::new("runs/burn_dragon_dreamer/tokenizer_moving_mnist").to_path_buf(),
                );
            }
        }
    }
    Ok((config, offset))
}
