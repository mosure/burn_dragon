use super::*;

#[test]
fn epochs_schedule_overrides_max_iters() {
    let training = make_training(5, Some(3));
    let schedule = resolve_train_schedule(&training, 4).expect("schedule");

    assert_eq!(schedule.source, TrainScheduleSource::Epochs);
    assert_eq!(schedule.steps_per_epoch, 4);
    assert_eq!(schedule.total_epochs, 3);
    assert_eq!(schedule.total_steps, 12);
    assert_eq!(schedule.total_steps % schedule.steps_per_epoch, 0);
}

#[test]
fn max_iters_schedule_uses_step_limit() {
    let training = make_training(12, None);
    let schedule = resolve_train_schedule(&training, 5).expect("schedule");

    assert_eq!(schedule.source, TrainScheduleSource::MaxIters);
    assert_eq!(schedule.steps_per_epoch, 5);
    assert_eq!(schedule.total_steps, 12);
    assert_eq!(schedule.total_epochs, 3);
}

#[test]
fn vision_schedule_mode_max_iters_overrides_inherited_epochs() {
    let mut training = VisionTrainingHyperparameters::default();
    training.epochs = Some(2);
    training.max_iters = 12;
    training.schedule_mode = Some(crate::VisionTrainScheduleMode::MaxIters);

    let schedule = resolve_vision_train_schedule(&training, 5).expect("schedule");

    assert_eq!(schedule.source, TrainScheduleSource::MaxIters);
    assert_eq!(schedule.steps_per_epoch, 5);
    assert_eq!(schedule.total_steps, 12);
    assert_eq!(schedule.total_epochs, 3);
}
