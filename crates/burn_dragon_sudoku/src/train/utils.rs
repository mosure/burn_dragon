use crate::train::prelude::*;

pub fn prepare_dataset(config: &SudokuTrainingConfig) -> Result<(Arc<SudokuDataset>, String)> {
    let (dataset, summary) = SudokuDataset::new(&config.dataset, config.training.batch_size)?;
    let dataset = Arc::new(dataset);
    info!("{summary}");
    Ok((dataset, summary))
}

#[derive(Serialize)]
struct SudokuRunConfigOutput {
    run_name: String,
    training: SudokuTrainingHyperparameters,
    model: SudokuModelConfig,
    dataset: SudokuDatasetConfig,
}

pub fn write_run_config(
    config: &SudokuTrainingConfig,
    run_dir: &Path,
    run_name: &str,
) -> Result<()> {
    fs::create_dir_all(run_dir)
        .with_context(|| format!("failed to create run directory {}", run_dir.display()))?;
    let output = SudokuRunConfigOutput {
        run_name: run_name.to_string(),
        training: config.training.clone(),
        model: config.model.clone(),
        dataset: config.dataset.clone(),
    };
    let payload = serde_json::to_string_pretty(&output).context("serialize run config")?;
    let path = run_dir.join("config.json");
    fs::write(&path, payload).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}
