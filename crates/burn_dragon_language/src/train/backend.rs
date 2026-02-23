use crate::train::prelude::*;
use crate::train::schedule::{
    TrainEnvironment, resolve_lr_scheduler, resolve_train_schedule, train_with_scheduler,
};
use crate::train::utils::write_run_config;
use std::time::Instant;

pub fn train_backend<B, Init>(
    config: &TrainingConfig,
    dataset: Arc<Dataset>,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    let stage_profile = crate::train::profile::enabled();
    if stage_profile {
        crate::train::profile::reset();
    }
    let train_wall_start = stage_profile.then(Instant::now);

    let device = B::Device::default();
    B::seed(&device, 1337);
    init_backend(&device);

    let training = &config.training;
    let optimizer_cfg = &config.optimizer;

    let mut model_config = build_model_config(&config.model, training.block_size);
    apply_wgpu_fused_core_override(
        &mut model_config,
        backend_name,
        config.wgpu.training.fused_core_recurrent,
        config.wgpu.training.fused_core_rollout,
    );
    let tokenizer = dataset.tokenizer();
    model_config.vocab_size = tokenizer.len();

    let steps_per_epoch = dataset.steps_per_epoch(DatasetSplit::Train);
    let schedule = resolve_train_schedule(training, steps_per_epoch)?;
    let steps_per_epoch = schedule.steps_per_epoch;
    let total_epochs = schedule.total_epochs;
    let total_steps = schedule.total_steps;

    info!(
        "train schedule: steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, epochs={total_epochs}, source={}",
        schedule.source.as_str()
    );
    let train_loader: Arc<dyn DataLoader<B, SequenceBatch<B>>> =
        Arc::new(RandomDataLoader::<B>::new(
            Arc::clone(&dataset),
            DatasetSplit::Train,
            &device,
            steps_per_epoch,
            Some(total_steps),
        ));

    let val_steps_per_epoch = dataset.steps_per_epoch(DatasetSplit::Val);
    let desired_valid_steps = usize::max(1, total_steps / training.log_frequency.max(1));
    let valid_steps = desired_valid_steps.min(val_steps_per_epoch).max(1);

    let valid_device = device.clone();
    let valid_loader: Arc<dyn DataLoader<ValidBackend<B>, SequenceBatch<ValidBackend<B>>>> =
        Arc::new(RandomDataLoader::<ValidBackend<B>>::new(
            Arc::clone(&dataset),
            DatasetSplit::Val,
            &valid_device,
            valid_steps,
            None,
        ));

    let mut model = Some(BDH::<B>::new(model_config.clone(), &device));
    let mut optim = Some(adamw_config_from_optimizer(optimizer_cfg).init::<B, BDH<B>>());
    let scheduler_iters = match schedule.source {
        ScheduleSource::Epochs => Some(total_steps),
        ScheduleSource::MaxIters => None,
    };
    let scheduler =
        resolve_lr_scheduler(optimizer_cfg, total_steps, scheduler_iters, &model_config)?;

    let run_root = PathBuf::from("runs");
    let (run_dir, run_name) = create_run_dir(&run_root)?;
    write_latest_run(&run_root, &run_name)?;
    write_run_config(config, &run_dir, &run_name)?;
    info!("run name: {run_name}");
    let context = TrainEnvironment {
        run_dir: &run_dir,
        run_name: &run_name,
        backend_name,
        training,
        model_config: &model_config,
        device: &device,
        train_loader,
        valid_loader,
        epochs: total_epochs,
    };
    let _model = match scheduler {
        ResolvedLrScheduler::Constant(lr) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            lr,
        )?,
        ResolvedLrScheduler::Cosine(scheduler) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
        )?,
        ResolvedLrScheduler::Linear(scheduler) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
        )?,
        ResolvedLrScheduler::Exponential(scheduler) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
        )?,
        ResolvedLrScheduler::Step(scheduler) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
        )?,
        ResolvedLrScheduler::Noam(scheduler) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
        )?,
    };

    info!("Training complete on {backend_name}");

    if let Some(start) = train_wall_start {
        let elapsed_ns = start.elapsed().as_nanos();
        let snapshot = crate::train::profile::snapshot();
        info!(
            "[stage-profile][training] total_ns={elapsed_ns} dataloader_cpu_ns={} dataloader_tensor_copy_ns={} dataloader_host_to_device_copy_bytes={} host_sync_points={} forward_ns={} loss_backward_ns={} train_steps={}",
            snapshot.dataloader_cpu_ns,
            snapshot.dataloader_tensor_copy_ns,
            snapshot.dataloader_host_to_device_copy_bytes,
            snapshot.host_sync_points,
            snapshot.forward_ns,
            snapshot.loss_backward_ns,
            snapshot.train_steps,
        );
    }

    Ok(())
}
