use crate::train::prelude::*;

pub(crate) static FAST_TRAIN: AtomicBool = AtomicBool::new(false);
pub(crate) const LEJEPA_EPS: f32 = 1e-6;
pub(crate) const SACCADE_EPS: f32 = 1e-6;
pub(crate) const SACCADE_SIGMA_MIN: f32 = 0.03;
pub(crate) const SACCADE_SIGMA_MAX: f32 = 0.5;
pub(crate) const SACCADE_LN_2: f32 = std::f32::consts::LN_2;
pub(crate) const SACCADE_LOD_LOG2_MIN: f32 = -2.0;
pub(crate) const SACCADE_LOD_LOG2_MAX: f32 = 1.0;
pub(crate) const SACCADE_RING_WIDTH: f32 = 0.02;
pub(crate) const SACCADE_RING_INTENSITY: f32 = 2.0;
pub(crate) const SACCADE_RING_OUTER_SCALE: f32 = 2.5;
pub(crate) const SACCADE_RING_OUTER_INTENSITY: f32 = 0.7;
pub(crate) const SACCADE_RING_INNER_COLOR: [f32; 3] = [60.0 / 255.0, 200.0 / 255.0, 1.0];
pub(crate) const SACCADE_VIEW_GAP: usize = 2;
pub(crate) const SACCADE_FOVEA_SUBSAMPLES: usize = 1;
pub(crate) const SACCADE_FOVEA_LOD_WINDOW: f32 = 3.0;
pub(crate) const SACCADE_FOVEA_AA_THRESHOLD: f32 = 1.25;
pub(crate) const SACCADE_FOVEA_SQRT2: f32 = std::f32::consts::SQRT_2;
pub(crate) const SACCADE_FOVEA_PI: f32 = std::f32::consts::PI;
pub(crate) const SACCADE_FOVEA_ERF_A: f32 = 0.147;
pub(crate) const SACCADE_FOVEA_SQRT_PI_OVER_2: f32 = 0.88622692545;

pub(crate) fn fast_train_enabled() -> bool {
    FAST_TRAIN.load(Ordering::Relaxed)
}

pub(crate) type ValidBackend<B> = <B as AutodiffBackend>::InnerBackend;

