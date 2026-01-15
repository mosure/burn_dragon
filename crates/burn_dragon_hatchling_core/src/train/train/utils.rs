use crate::train::prelude::*;

pub(crate) fn build_vocab_only(config: &TrainingConfig) -> Result<()> {
    let dataset = prepare_dataset(&config.dataset, &config.training)?;
    let tokenizer = dataset.tokenizer();
    info!(
        "Tokenizer `{}` ready with {} tokens",
        config.dataset.tokenizer.kind_name(),
        tokenizer.len()
    );
    Ok(())
}

pub(crate) fn prepare_dataset(
    dataset_cfg: &DatasetConfig,
    training: &TrainingHyperparameters,
) -> Result<Arc<Dataset>> {
    let tokenizer_path = dataset_cfg.tokenizer.storage_path(&dataset_cfg.cache_dir);
    let tokenizer_preexists = tokenizer_path
        .as_ref()
        .map(|path| path.is_file())
        .unwrap_or(false);

    let (dataset_enum, dataset_summary) = build_dataset(dataset_cfg, training)?;
    let dataset = Arc::new(dataset_enum);

    let tokenizer = dataset.tokenizer();
    match tokenizer_path {
        Some(path) if tokenizer_preexists => info!(
            "Loaded {} tokenizer with {} tokens from {}",
            dataset_cfg.tokenizer.kind_name(),
            tokenizer.len(),
            path.display()
        ),
        Some(path) => info!(
            "Built {} tokenizer with {} tokens at {}",
            dataset_cfg.tokenizer.kind_name(),
            tokenizer.len(),
            path.display()
        ),
        None => info!(
            "Initialized {} tokenizer with {} tokens (no persistence required)",
            dataset_cfg.tokenizer.kind_name(),
            tokenizer.len()
        ),
    };

    info!("{dataset_summary}");

    Ok(dataset)
}

pub(crate) fn adamw_config_from_optimizer(optimizer_cfg: &OptimizerConfig) -> AdamWConfig {
    let mut config = AdamWConfig::new().with_weight_decay(optimizer_cfg.weight_decay);
    if let Some(clip) = optimizer_cfg.grad_clip_norm {
        config = config.with_grad_clipping(Some(GradientClippingConfig::Norm(clip)));
    } else if let Some(clip) = optimizer_cfg.grad_clip_value {
        config = config.with_grad_clipping(Some(GradientClippingConfig::Value(clip)));
    }
    config
}

pub(crate) fn log_theoretical_profile(config: &BDHConfig, batch: usize, block: usize, backend: &str) {
    let batch = batch as u64;
    let time = block as u64;
    let embed = config.n_embd as u64;
    let latent_per_head = config.latent_per_head() as u64;
    let latent_total = config.latent_total() as u64;
    let heads = config.n_head as u64;
    let bt = batch * time;

    let encoder_matmul = 2 * bt * embed * latent_total;
    let attn_scores = 2 * batch * heads * time * time * latent_per_head;
    let attn_value = 2 * batch * heads * time * time * embed;
    let decoder_matmul = 2 * bt * latent_total * embed;
    let total = encoder_matmul + attn_scores + attn_value + decoder_matmul;

    info!(
        "[train:{backend}] approx forward GFLOPs: total={total_gflops:.2}, encoder={enc:.2}, \
         attn_scores={scores:.2}, attn_value={value:.2}, decoder={dec:.2} (backward ~2x forward)",
        total_gflops = total as f64 / 1e9,
        enc = encoder_matmul as f64 / 1e9,
        scores = attn_scores as f64 / 1e9,
        value = attn_value as f64 / 1e9,
        dec = decoder_matmul as f64 / 1e9,
    );
}

pub(crate) fn create_run_dir(run_root: &Path) -> Result<(PathBuf, String)> {
    let mut generator = Generator::default();

    for _ in 0..64 {
        let name = generator
            .next()
            .unwrap_or_else(|| "nameless-hatchling".to_string());
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

pub(crate) fn write_latest_run(run_root: &Path, run_name: &str) -> Result<()> {
    fs::create_dir_all(run_root)
        .with_context(|| format!("failed to create run directory {}", run_root.display()))?;
    let path = run_root.join("latest");
    fs::write(&path, run_name)
        .with_context(|| format!("failed to write latest run {}", path.display()))?;
    Ok(())
}

#[derive(Serialize)]
pub(crate) struct WebConfigOutput {
    run_name: String,
    block_size: usize,
    overrides: ModelOverrides,
}

pub(crate) fn write_run_config(config: &TrainingConfig, run_dir: &Path, run_name: &str) -> Result<()> {
    fs::create_dir_all(run_dir)
        .with_context(|| format!("failed to create run directory {}", run_dir.display()))?;

    let block_size = config
        .model
        .block_size
        .unwrap_or(config.training.block_size)
        .max(1);
    let output = WebConfigOutput {
        run_name: run_name.to_string(),
        block_size,
        overrides: config.model.clone(),
    };
    let payload =
        serde_json::to_string_pretty(&output).context("failed to serialize web config")?;
    let path = run_dir.join("config.json");
    fs::write(&path, payload)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

