use crate::train::prelude::*;

pub fn adamw_config_from_optimizer(optimizer_cfg: &OptimizerConfig) -> AdamWConfig {
    let mut config = AdamWConfig::new().with_weight_decay(optimizer_cfg.weight_decay);
    if let Some(clip) = optimizer_cfg.grad_clip_norm {
        config = config.with_grad_clipping(Some(GradientClippingConfig::Norm(clip)));
    } else if let Some(clip) = optimizer_cfg.grad_clip_value {
        config = config.with_grad_clipping(Some(GradientClippingConfig::Value(clip)));
    }
    config
}

pub fn create_run_dir(run_root: &Path) -> Result<(PathBuf, String)> {
    let mut generator = Generator::default();

    for _ in 0..64 {
        let name = generator
            .next()
            .unwrap_or_else(|| "nameless-dragon".to_string());
        let candidate = run_root.join(&name);
        if !candidate.exists() {
            return Ok((candidate, name));
        }
    }

    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| anyhow!("failed to read system time: {err}"))?
        .as_secs();
    let name = format!("run-{suffix}");
    Ok((run_root.join(&name), name))
}

pub fn write_latest_run(run_root: &Path, run_name: &str) -> Result<()> {
    fs::create_dir_all(run_root)
        .with_context(|| format!("failed to create run directory {}", run_root.display()))?;
    let path = run_root.join("latest");
    fs::write(&path, run_name)
        .with_context(|| format!("failed to write latest run {}", path.display()))?;
    Ok(())
}
