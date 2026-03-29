use crate::checkpoint::{checkpoint_base, save_module_checkpoint};
use anyhow::{Context, Result};
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn_dragon_train::train::pipeline::{activate_planned_run, plan_run_artifacts};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default)]
pub struct RunWorkspace {
    pub run_dir: Option<PathBuf>,
    pub checkpoint_dir: Option<PathBuf>,
}

pub fn prepare_run_dir(run_root: Option<&Path>) -> Result<Option<PathBuf>> {
    let Some(run_root) = run_root else {
        return Ok(None);
    };
    let planned = plan_run_artifacts(run_root, None)?;
    activate_planned_run(&planned)?;
    Ok(Some(planned.run_dir))
}

pub fn prepare_configured_run<T: Serialize>(
    run_root: Option<&Path>,
    config: &T,
    config_label: &str,
) -> Result<RunWorkspace> {
    let run_dir = prepare_run_dir(run_root)?;
    if let Some(run_dir) = run_dir.as_ref() {
        let config_path = run_dir.join("config.json");
        fs::write(
            &config_path,
            serde_json::to_vec_pretty(config)
                .with_context(|| format!("serialize {config_label} config"))?,
        )
        .with_context(|| format!("write {config_label} config {}", config_path.display()))?;
    }
    let checkpoint_dir = run_dir.as_ref().map(|path| path.join("checkpoint"));
    if let Some(dir) = checkpoint_dir.as_ref() {
        fs::create_dir_all(dir)
            .with_context(|| format!("create {config_label} checkpoint dir {}", dir.display()))?;
    }
    Ok(RunWorkspace {
        run_dir,
        checkpoint_dir,
    })
}

pub fn should_run_step(step_index: usize, total_steps: usize, every: usize) -> bool {
    let step = step_index + 1;
    step % every.max(1) == 0 || step == total_steps
}

pub fn save_named_checkpoint<B, M>(
    model: &M,
    checkpoint_dir: Option<&Path>,
    stem: &str,
) -> Result<Option<PathBuf>>
where
    B: Backend,
    M: Module<B> + Clone,
{
    let Some(checkpoint_dir) = checkpoint_dir else {
        return Ok(None);
    };
    let base = checkpoint_base(checkpoint_dir, stem);
    save_module_checkpoint::<B, _>(model, &base)?;
    Ok(Some(base))
}
