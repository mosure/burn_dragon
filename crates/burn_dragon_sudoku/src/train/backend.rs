use crate::artifacts::write_validation_artifacts;
use crate::checkpoint::write_training_snapshot;
use crate::train::prelude::*;
use crate::train::schedule::{
    SudokuTrainEnvironment, resolve_lr_scheduler, resolve_train_schedule, train_with_scheduler,
};
use crate::train::steps::SudokuTrainer;
use crate::train::utils::{prepare_dataset, write_run_config};
use burn_dragon_core::BDHConfig;
type TrainBackendResult<B> = (
    SudokuTrainer<ValidBackend<B>>,
    PathBuf,
    Arc<SudokuDataset>,
    <B as BackendTrait>::Device,
);

fn resolved_core_config(config: &SudokuTrainingConfig, backend_name: &str) -> BDHConfig {
    config
        .model
        .to_bdh_config_for_backend(backend_name, &config.wgpu)
}

pub fn train_backend<B, Init>(
    config: &SudokuTrainingConfig,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    let (model, run_dir, dataset, device) =
        train_backend_inner::<B, Init>(config, backend_name, init_backend, true)?;

    if config.artifacts.max_samples > 0 {
        write_validation_artifacts(
            &model.model,
            dataset.as_ref(),
            &config.artifacts,
            &config.training,
            &run_dir,
            &device,
        )?;
    }

    Ok(())
}

pub fn train_backend_for_test<B, Init>(
    config: &SudokuTrainingConfig,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    let _ = train_backend_inner::<B, Init>(config, backend_name, init_backend, false)?;
    Ok(())
}

fn train_backend_inner<B, Init>(
    config: &SudokuTrainingConfig,
    backend_name: &str,
    init_backend: Init,
    write_config: bool,
) -> Result<TrainBackendResult<B>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    config.validate()?;

    let device = B::Device::default();
    B::seed(&device, 1337);
    init_backend(&device);

    let (dataset, _summary) = prepare_dataset(config)?;
    let training = &config.training;
    let optimizer_cfg = &config.optimizer;

    let steps_per_epoch = dataset.steps_per_epoch(SudokuSplit::Train);
    let schedule = resolve_train_schedule(training, steps_per_epoch)?;
    let steps_per_epoch = schedule.steps_per_epoch;
    let total_epochs = schedule.total_epochs;
    let total_steps = schedule.total_steps;

    info!(
        "sudoku train schedule: steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, epochs={total_epochs}, source={}",
        schedule.source.as_str()
    );

    let train_loader: Arc<dyn DataLoader<B, SudokuBatch<B>>> =
        Arc::new(SudokuRandomDataLoader::<B>::new(
            Arc::clone(&dataset),
            SudokuSplit::Train,
            &device,
            steps_per_epoch,
            Some(total_steps),
        ));

    let val_steps_per_epoch = dataset.steps_per_epoch(SudokuSplit::Val);
    let valid_steps =
        resolve_valid_steps_per_epoch(total_steps, training.log_frequency, val_steps_per_epoch);

    let valid_device = device.clone();
    let valid_loader: Arc<dyn DataLoader<ValidBackend<B>, SudokuBatch<ValidBackend<B>>>> =
        Arc::new(SudokuRandomDataLoader::<ValidBackend<B>>::new(
            Arc::clone(&dataset),
            SudokuSplit::Val,
            &valid_device,
            valid_steps,
            None,
        ));

    let core_config = resolved_core_config(config, backend_name);
    let model = SudokuSaccadeModel::<B>::new_with_bdh_config(&config.model, core_config, &device);
    let mut trainer = SudokuTrainer::new(model, training.clone(), total_steps);
    let optimizer = adamw_config_from_optimizer(optimizer_cfg).init::<B, SudokuTrainer<B>>();

    let scheduler_iters = match schedule.source {
        ScheduleSource::Epochs => Some(total_steps),
        ScheduleSource::MaxIters => None,
    };
    let scheduler =
        resolve_lr_scheduler(optimizer_cfg, total_steps, scheduler_iters, &config.model)?;

    let run_root = PathBuf::from("runs").join("sudoku");
    let (run_dir, run_name) = create_run_dir(&run_root)?;
    write_latest_run(&run_root, &run_name)?;
    if write_config {
        write_run_config(config, &run_dir, &run_name)?;
        write_training_snapshot(config, &run_dir)?;
    }

    if config.artifacts.max_samples > 0 {
        trainer = trainer.with_validation_artifacts(config.artifacts.clone(), run_dir.clone());
    }

    info!("sudoku run name: {run_name}");

    let context = SudokuTrainEnvironment {
        run_dir: &run_dir,
        run_name: &run_name,
        backend_name,
        training,
        device: &device,
        train_loader,
        valid_loader,
        epochs: total_epochs,
    };

    let trained = match scheduler {
        ResolvedLrScheduler::Constant(lr) => {
            train_with_scheduler(&context, trainer, optimizer, lr)?
        }
        ResolvedLrScheduler::Cosine(schedule) => {
            train_with_scheduler(&context, trainer, optimizer, schedule)?
        }
        ResolvedLrScheduler::Linear(schedule) => {
            train_with_scheduler(&context, trainer, optimizer, schedule)?
        }
        ResolvedLrScheduler::Exponential(schedule) => {
            train_with_scheduler(&context, trainer, optimizer, schedule)?
        }
        ResolvedLrScheduler::Step(schedule) => {
            train_with_scheduler(&context, trainer, optimizer, schedule)?
        }
        ResolvedLrScheduler::Noam(schedule) => {
            train_with_scheduler(&context, trainer, optimizer, schedule)?
        }
    };

    info!("Sudoku training complete on {backend_name}");

    Ok((trained, run_dir, dataset, device))
}

#[cfg(test)]
mod tests {
    use super::resolved_core_config;
    use crate::config::{SudokuModelConfig, load_training_config};
    use std::path::PathBuf;

    fn load_tiny_config() -> crate::config::SudokuTrainingConfig {
        let config_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../config/sudoku/saccade/tiny.toml");
        load_training_config(&[config_path]).expect("load tiny sudoku config")
    }

    #[test]
    fn wgpu_core_override_enables_fused_recurrent_and_rollout() {
        let mut config = load_tiny_config();
        config.model = SudokuModelConfig {
            fused_kernels: false,
            ..SudokuModelConfig::default()
        };
        config.wgpu.training.fused_core_recurrent = Some(true);
        config.wgpu.training.fused_core_rollout = Some(true);

        let core = resolved_core_config(&config, "wgpu");
        assert!(core.fused_kernels.enabled);
        assert!(core.fused_kernels.wgpu_recurrent_kernel);
        assert!(core.fused_kernels.wgpu_rollout_fused);
    }

    #[test]
    fn non_wgpu_backend_keeps_model_default_fused_flags() {
        let mut config = load_tiny_config();
        config.model = SudokuModelConfig {
            fused_kernels: false,
            ..SudokuModelConfig::default()
        };
        config.wgpu.training.fused_core_recurrent = Some(true);
        config.wgpu.training.fused_core_rollout = Some(true);

        let baseline = config.model.to_bdh_config();
        let core = resolved_core_config(&config, "cpu");
        assert_eq!(core.fused_kernels.enabled, baseline.fused_kernels.enabled);
        assert_eq!(
            core.fused_kernels.wgpu_recurrent_kernel,
            baseline.fused_kernels.wgpu_recurrent_kernel
        );
        assert_eq!(
            core.fused_kernels.wgpu_rollout_fused,
            baseline.fused_kernels.wgpu_rollout_fused
        );
    }
}
