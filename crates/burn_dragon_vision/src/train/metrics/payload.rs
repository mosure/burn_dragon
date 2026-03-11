use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{Int, Tensor};
use burn_train::metric::{Adaptor, ItemLazy, LossInput};
use burn_dragon_train::train::metrics::{
    LossValue, MetricsBackend, OptionalScalarValue, ScalarValue,
};

pub const VISION_ROLLOUT_HORIZON_CAPS: [usize; 6] = [1, 2, 4, 8, 12, 24];
pub const VISION_ROLLOUT_HORIZON_COUNT: usize = VISION_ROLLOUT_HORIZON_CAPS.len();

fn sync_float_tensor<B: BackendTrait, const D: usize>(
    tensor: Tensor<B, D>,
) -> Tensor<MetricsBackend, D> {
    Tensor::<MetricsBackend, D>::from_data(tensor.into_data(), &Default::default())
}

fn sync_int_tensor<B: BackendTrait, const D: usize>(
    tensor: Tensor<B, D, Int>,
) -> Tensor<MetricsBackend, D, Int> {
    Tensor::<MetricsBackend, D, Int>::from_data(tensor.into_data(), &Default::default())
}

fn sync_optional_float_tensor<B: BackendTrait, const D: usize>(
    tensor: Option<Tensor<B, D>>,
) -> Option<Tensor<MetricsBackend, D>> {
    tensor.map(sync_float_tensor)
}

fn sync_optional_int_tensor<B: BackendTrait, const D: usize>(
    tensor: Option<Tensor<B, D, Int>>,
) -> Option<Tensor<MetricsBackend, D, Int>> {
    tensor.map(sync_int_tensor)
}

#[derive(Clone)]
pub struct VisionArtifactInput<B: BackendTrait> {
    pub views: Option<Tensor<B, 5>>,
    pub frames: Option<Tensor<B, 5>>,
    pub debug_recon_frames: Option<Tensor<B, 5>>,
    pub patch_norms: Option<Tensor<B, 3>>,
    pub pca_rgb: Option<Tensor<B, 4>>,
    pub posterior_patch_norms_steps: Option<Tensor<B, 4>>,
    pub posterior_pca_rgb_steps: Option<Tensor<B, 5>>,
    pub patch_norms_steps: Option<Tensor<B, 4>>,
    pub pca_rgb_steps: Option<Tensor<B, 5>>,
    pub debug_patch_norms_steps: Option<Tensor<B, 4>>,
    pub debug_pca_rgb_steps: Option<Tensor<B, 5>>,
    pub probe_logits: Option<Tensor<B, 2>>,
    pub labels: Option<Tensor<B, 1, Int>>,
    pub legend: Option<Vec<String>>,
    pub artifact_scale: usize,
    pub prediction_start: Option<usize>,
}

impl<B: BackendTrait> VisionArtifactInput<B> {
    pub fn empty() -> Self {
        Self {
            views: None,
            frames: None,
            debug_recon_frames: None,
            patch_norms: None,
            pca_rgb: None,
            posterior_patch_norms_steps: None,
            posterior_pca_rgb_steps: None,
            patch_norms_steps: None,
            pca_rgb_steps: None,
            debug_patch_norms_steps: None,
            debug_pca_rgb_steps: None,
            probe_logits: None,
            labels: None,
            legend: None,
            artifact_scale: 1,
            prediction_start: None,
        }
    }

    pub fn sync(self) -> VisionArtifactInput<MetricsBackend> {
        VisionArtifactInput {
            views: sync_optional_float_tensor(self.views),
            frames: sync_optional_float_tensor(self.frames),
            debug_recon_frames: sync_optional_float_tensor(self.debug_recon_frames),
            patch_norms: sync_optional_float_tensor(self.patch_norms),
            pca_rgb: sync_optional_float_tensor(self.pca_rgb),
            posterior_patch_norms_steps: sync_optional_float_tensor(
                self.posterior_patch_norms_steps,
            ),
            posterior_pca_rgb_steps: sync_optional_float_tensor(self.posterior_pca_rgb_steps),
            patch_norms_steps: sync_optional_float_tensor(self.patch_norms_steps),
            pca_rgb_steps: sync_optional_float_tensor(self.pca_rgb_steps),
            debug_patch_norms_steps: sync_optional_float_tensor(self.debug_patch_norms_steps),
            debug_pca_rgb_steps: sync_optional_float_tensor(self.debug_pca_rgb_steps),
            probe_logits: sync_optional_float_tensor(self.probe_logits),
            labels: sync_optional_int_tensor(self.labels),
            legend: self.legend,
            artifact_scale: self.artifact_scale,
            prediction_start: self.prediction_start,
        }
    }
}

#[derive(Clone)]
pub struct VisionOutput<B: BackendTrait> {
    loss: Tensor<B, 1>,
    inv_loss: Tensor<B, 1>,
    observe_loss: Tensor<B, 1>,
    mode_separation_ratio: Tensor<B, 1>,
    sigreg_loss: Tensor<B, 1>,
    recon_loss: Tensor<B, 1>,
    recon_psnr_masked: Tensor<B, 1>,
    recon_psnr_full: Tensor<B, 1>,
    policy_loss: Tensor<B, 1>,
    policy_advantage_abs_mean: Tensor<B, 1>,
    policy_advantage_std: Tensor<B, 1>,
    policy_log_prob_mean: Tensor<B, 1>,
    policy_entropy: Tensor<B, 1>,
    policy_action_clamp_rate: Tensor<B, 1>,
    probe_loss: Tensor<B, 1>,
    probe_acc: Tensor<B, 1>,
    rollout_inv_to_horizon: [Tensor<B, 1>; VISION_ROLLOUT_HORIZON_COUNT],
    rollout_state_norm_ratio_to_horizon: [Tensor<B, 1>; VISION_ROLLOUT_HORIZON_COUNT],
    rollout_state_motion_to_horizon: [Tensor<B, 1>; VISION_ROLLOUT_HORIZON_COUNT],
    rollout_com_error_to_h24: Option<Tensor<B, 1>>,
    rollout_velocity_error_to_h24: Option<Tensor<B, 1>>,
    long_rollout_inv_to_horizon: [Option<Tensor<B, 1>>; VISION_ROLLOUT_HORIZON_COUNT],
    long_rollout_state_norm_ratio_to_horizon: [Option<Tensor<B, 1>>; VISION_ROLLOUT_HORIZON_COUNT],
    long_rollout_state_motion_to_horizon: [Option<Tensor<B, 1>>; VISION_ROLLOUT_HORIZON_COUNT],
    long_rollout_com_error_to_h24: Option<Tensor<B, 1>>,
    long_rollout_velocity_error_to_h24: Option<Tensor<B, 1>>,
    artifacts: Option<VisionArtifactInput<B>>,
}

impl<B: BackendTrait> VisionOutput<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        loss: Tensor<B, 1>,
        inv_loss: Tensor<B, 1>,
        observe_loss: Tensor<B, 1>,
        mode_separation_ratio: Tensor<B, 1>,
        sigreg_loss: Tensor<B, 1>,
        recon_loss: Tensor<B, 1>,
        recon_psnr_masked: Tensor<B, 1>,
        recon_psnr_full: Tensor<B, 1>,
        policy_loss: Tensor<B, 1>,
        policy_advantage_abs_mean: Tensor<B, 1>,
        policy_advantage_std: Tensor<B, 1>,
        policy_log_prob_mean: Tensor<B, 1>,
        policy_entropy: Tensor<B, 1>,
        policy_action_clamp_rate: Tensor<B, 1>,
        probe_loss: Tensor<B, 1>,
        probe_acc: Tensor<B, 1>,
        artifacts: Option<VisionArtifactInput<B>>,
    ) -> Self {
        let device = loss.device();
        let rollout_inv_to_horizon = core::array::from_fn(|_| Tensor::<B, 1>::zeros([1], &device));
        let rollout_state_norm_ratio_to_horizon =
            core::array::from_fn(|_| Tensor::<B, 1>::zeros([1], &device));
        let rollout_state_motion_to_horizon =
            core::array::from_fn(|_| Tensor::<B, 1>::zeros([1], &device));
        let rollout_com_error_to_h24 = None;
        let rollout_velocity_error_to_h24 = None;
        let long_rollout_inv_to_horizon = core::array::from_fn(|_| None);
        let long_rollout_state_norm_ratio_to_horizon = core::array::from_fn(|_| None);
        let long_rollout_state_motion_to_horizon = core::array::from_fn(|_| None);
        let long_rollout_com_error_to_h24 = None;
        let long_rollout_velocity_error_to_h24 = None;
        Self {
            loss,
            inv_loss,
            observe_loss,
            mode_separation_ratio,
            sigreg_loss,
            recon_loss,
            recon_psnr_masked,
            recon_psnr_full,
            policy_loss,
            policy_advantage_abs_mean,
            policy_advantage_std,
            policy_log_prob_mean,
            policy_entropy,
            policy_action_clamp_rate,
            probe_loss,
            probe_acc,
            rollout_inv_to_horizon,
            rollout_state_norm_ratio_to_horizon,
            rollout_state_motion_to_horizon,
            rollout_com_error_to_h24,
            rollout_velocity_error_to_h24,
            long_rollout_inv_to_horizon,
            long_rollout_state_norm_ratio_to_horizon,
            long_rollout_state_motion_to_horizon,
            long_rollout_com_error_to_h24,
            long_rollout_velocity_error_to_h24,
            artifacts,
        }
    }

    pub fn with_rollout_horizon_metrics(
        mut self,
        rollout_inv_to_horizon: [Tensor<B, 1>; VISION_ROLLOUT_HORIZON_COUNT],
        rollout_state_norm_ratio_to_horizon: [Tensor<B, 1>; VISION_ROLLOUT_HORIZON_COUNT],
        rollout_state_motion_to_horizon: [Tensor<B, 1>; VISION_ROLLOUT_HORIZON_COUNT],
    ) -> Self {
        self.rollout_inv_to_horizon = rollout_inv_to_horizon;
        self.rollout_state_norm_ratio_to_horizon = rollout_state_norm_ratio_to_horizon;
        self.rollout_state_motion_to_horizon = rollout_state_motion_to_horizon;
        self
    }

    pub fn with_long_rollout_horizon_metrics(
        mut self,
        rollout_inv_to_horizon: [Option<Tensor<B, 1>>; VISION_ROLLOUT_HORIZON_COUNT],
        rollout_state_norm_ratio_to_horizon: [Option<Tensor<B, 1>>; VISION_ROLLOUT_HORIZON_COUNT],
        rollout_state_motion_to_horizon: [Option<Tensor<B, 1>>; VISION_ROLLOUT_HORIZON_COUNT],
    ) -> Self {
        self.long_rollout_inv_to_horizon = rollout_inv_to_horizon;
        self.long_rollout_state_norm_ratio_to_horizon = rollout_state_norm_ratio_to_horizon;
        self.long_rollout_state_motion_to_horizon = rollout_state_motion_to_horizon;
        self
    }

    pub fn with_rollout_kinematics_metrics(
        mut self,
        rollout_com_error_to_h24: Option<Tensor<B, 1>>,
        rollout_velocity_error_to_h24: Option<Tensor<B, 1>>,
    ) -> Self {
        self.rollout_com_error_to_h24 = rollout_com_error_to_h24;
        self.rollout_velocity_error_to_h24 = rollout_velocity_error_to_h24;
        self
    }

    pub fn with_long_rollout_kinematics_metrics(
        mut self,
        rollout_com_error_to_h24: Option<Tensor<B, 1>>,
        rollout_velocity_error_to_h24: Option<Tensor<B, 1>>,
    ) -> Self {
        self.long_rollout_com_error_to_h24 = rollout_com_error_to_h24;
        self.long_rollout_velocity_error_to_h24 = rollout_velocity_error_to_h24;
        self
    }
}

impl<B: BackendTrait> ItemLazy for VisionOutput<B> {
    type ItemSync = VisionOutput<MetricsBackend>;

    fn sync(self) -> Self::ItemSync {
        VisionOutput {
            loss: sync_float_tensor(self.loss),
            inv_loss: sync_float_tensor(self.inv_loss),
            observe_loss: sync_float_tensor(self.observe_loss),
            mode_separation_ratio: sync_float_tensor(self.mode_separation_ratio),
            sigreg_loss: sync_float_tensor(self.sigreg_loss),
            recon_loss: sync_float_tensor(self.recon_loss),
            recon_psnr_masked: sync_float_tensor(self.recon_psnr_masked),
            recon_psnr_full: sync_float_tensor(self.recon_psnr_full),
            policy_loss: sync_float_tensor(self.policy_loss),
            policy_advantage_abs_mean: sync_float_tensor(self.policy_advantage_abs_mean),
            policy_advantage_std: sync_float_tensor(self.policy_advantage_std),
            policy_log_prob_mean: sync_float_tensor(self.policy_log_prob_mean),
            policy_entropy: sync_float_tensor(self.policy_entropy),
            policy_action_clamp_rate: sync_float_tensor(self.policy_action_clamp_rate),
            probe_loss: sync_float_tensor(self.probe_loss),
            probe_acc: sync_float_tensor(self.probe_acc),
            rollout_inv_to_horizon: self.rollout_inv_to_horizon.map(sync_float_tensor),
            rollout_state_norm_ratio_to_horizon: self
                .rollout_state_norm_ratio_to_horizon
                .map(sync_float_tensor),
            rollout_state_motion_to_horizon: self
                .rollout_state_motion_to_horizon
                .map(sync_float_tensor),
            rollout_com_error_to_h24: sync_optional_float_tensor(self.rollout_com_error_to_h24),
            rollout_velocity_error_to_h24: sync_optional_float_tensor(
                self.rollout_velocity_error_to_h24,
            ),
            long_rollout_inv_to_horizon: self
                .long_rollout_inv_to_horizon
                .map(sync_optional_float_tensor),
            long_rollout_state_norm_ratio_to_horizon: self
                .long_rollout_state_norm_ratio_to_horizon
                .map(sync_optional_float_tensor),
            long_rollout_state_motion_to_horizon: self
                .long_rollout_state_motion_to_horizon
                .map(sync_optional_float_tensor),
            long_rollout_com_error_to_h24: sync_optional_float_tensor(
                self.long_rollout_com_error_to_h24,
            ),
            long_rollout_velocity_error_to_h24: sync_optional_float_tensor(
                self.long_rollout_velocity_error_to_h24,
            ),
            artifacts: self.artifacts.map(VisionArtifactInput::sync),
        }
    }
}

impl<B: BackendTrait> Adaptor<LossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<LossValue<B>> for VisionOutput<B> {
    fn adapt(&self) -> LossValue<B> {
        LossValue::new(self.loss.clone())
    }
}

macro_rules! define_scalar_input {
    ($name:ident, scalar) => {
        #[derive(Clone)]
        pub struct $name<B: BackendTrait> {
            value: Tensor<B, 1>,
        }

        impl<B: BackendTrait> $name<B> {
            pub fn new(value: Tensor<B, 1>) -> Self {
                Self { value }
            }
        }

        impl<B: BackendTrait> ScalarValue<B> for $name<B> {
            fn value(&self) -> Tensor<B, 1> {
                self.value.clone()
            }
        }
    };
    ($name:ident, optional) => {
        #[derive(Clone)]
        pub struct $name<B: BackendTrait> {
            value: Option<Tensor<B, 1>>,
        }

        impl<B: BackendTrait> $name<B> {
            pub fn new(value: Option<Tensor<B, 1>>) -> Self {
                Self { value }
            }
        }

        impl<B: BackendTrait> OptionalScalarValue<B> for $name<B> {
            fn value(&self) -> Option<Tensor<B, 1>> {
                self.value.clone()
            }
        }
    };
}

macro_rules! define_scalar_input_const {
    ($name:ident, scalar) => {
        #[derive(Clone)]
        pub struct $name<B: BackendTrait, const INDEX: usize> {
            value: Tensor<B, 1>,
        }

        impl<B: BackendTrait, const INDEX: usize> $name<B, INDEX> {
            pub fn new(value: Tensor<B, 1>) -> Self {
                Self { value }
            }
        }

        impl<B: BackendTrait, const INDEX: usize> ScalarValue<B> for $name<B, INDEX> {
            fn value(&self) -> Tensor<B, 1> {
                self.value.clone()
            }
        }
    };
    ($name:ident, optional) => {
        #[derive(Clone)]
        pub struct $name<B: BackendTrait, const INDEX: usize> {
            value: Option<Tensor<B, 1>>,
        }

        impl<B: BackendTrait, const INDEX: usize> $name<B, INDEX> {
            pub fn new(value: Option<Tensor<B, 1>>) -> Self {
                Self { value }
            }
        }

        impl<B: BackendTrait, const INDEX: usize> OptionalScalarValue<B> for $name<B, INDEX> {
            fn value(&self) -> Option<Tensor<B, 1>> {
                self.value.clone()
            }
        }
    };
}

define_scalar_input!(InvLossInput, scalar);
define_scalar_input!(ObserveLossInput, scalar);
define_scalar_input!(ModeSeparationRatioInput, scalar);
define_scalar_input!(SigRegLossInput, scalar);
define_scalar_input!(ReconLossInput, scalar);
define_scalar_input!(ReconPsnrMaskedInput, scalar);
define_scalar_input!(ReconPsnrFullInput, scalar);
define_scalar_input!(PolicyLossInput, scalar);
define_scalar_input!(AdvantageAbsMeanInput, scalar);
define_scalar_input!(AdvantageStdInput, scalar);
define_scalar_input!(LogProbMeanInput, scalar);
define_scalar_input!(PolicyEntropyInput, scalar);
define_scalar_input!(ActionClampRateInput, scalar);
define_scalar_input!(ProbeLossInput, scalar);
define_scalar_input!(ProbeAccInput, scalar);
define_scalar_input_const!(RolloutInvToHorizonInput, scalar);
define_scalar_input_const!(RolloutStateNormRatioToHorizonInput, scalar);
define_scalar_input_const!(RolloutStateMotionToHorizonInput, scalar);
define_scalar_input_const!(LongRolloutInvToHorizonInput, optional);
define_scalar_input_const!(LongRolloutStateNormRatioToHorizonInput, optional);
define_scalar_input_const!(LongRolloutStateMotionToHorizonInput, optional);
define_scalar_input!(RolloutComErrorToH24Input, optional);
define_scalar_input!(RolloutVelocityErrorToH24Input, optional);
define_scalar_input!(LongRolloutComErrorToH24Input, optional);
define_scalar_input!(LongRolloutVelocityErrorToH24Input, optional);

impl<B: BackendTrait> Adaptor<InvLossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> InvLossInput<B> {
        InvLossInput::new(self.inv_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<ObserveLossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> ObserveLossInput<B> {
        ObserveLossInput::new(self.observe_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<ModeSeparationRatioInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> ModeSeparationRatioInput<B> {
        ModeSeparationRatioInput::new(self.mode_separation_ratio.clone())
    }
}

impl<B: BackendTrait> Adaptor<SigRegLossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> SigRegLossInput<B> {
        SigRegLossInput::new(self.sigreg_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<ReconLossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> ReconLossInput<B> {
        ReconLossInput::new(self.recon_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<ReconPsnrMaskedInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> ReconPsnrMaskedInput<B> {
        ReconPsnrMaskedInput::new(self.recon_psnr_masked.clone())
    }
}

impl<B: BackendTrait> Adaptor<ReconPsnrFullInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> ReconPsnrFullInput<B> {
        ReconPsnrFullInput::new(self.recon_psnr_full.clone())
    }
}

impl<B: BackendTrait> Adaptor<PolicyLossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> PolicyLossInput<B> {
        PolicyLossInput::new(self.policy_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<AdvantageAbsMeanInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> AdvantageAbsMeanInput<B> {
        AdvantageAbsMeanInput::new(self.policy_advantage_abs_mean.clone())
    }
}

impl<B: BackendTrait> Adaptor<AdvantageStdInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> AdvantageStdInput<B> {
        AdvantageStdInput::new(self.policy_advantage_std.clone())
    }
}

impl<B: BackendTrait> Adaptor<LogProbMeanInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> LogProbMeanInput<B> {
        LogProbMeanInput::new(self.policy_log_prob_mean.clone())
    }
}

impl<B: BackendTrait> Adaptor<PolicyEntropyInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> PolicyEntropyInput<B> {
        PolicyEntropyInput::new(self.policy_entropy.clone())
    }
}

impl<B: BackendTrait> Adaptor<ActionClampRateInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> ActionClampRateInput<B> {
        ActionClampRateInput::new(self.policy_action_clamp_rate.clone())
    }
}

impl<B: BackendTrait> Adaptor<ProbeLossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> ProbeLossInput<B> {
        ProbeLossInput::new(self.probe_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<ProbeAccInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> ProbeAccInput<B> {
        ProbeAccInput::new(self.probe_acc.clone())
    }
}

impl<B: BackendTrait, const INDEX: usize> Adaptor<RolloutInvToHorizonInput<B, INDEX>>
    for VisionOutput<B>
{
    fn adapt(&self) -> RolloutInvToHorizonInput<B, INDEX> {
        RolloutInvToHorizonInput::new(self.rollout_inv_to_horizon[INDEX].clone())
    }
}

impl<B: BackendTrait, const INDEX: usize> Adaptor<RolloutStateNormRatioToHorizonInput<B, INDEX>>
    for VisionOutput<B>
{
    fn adapt(&self) -> RolloutStateNormRatioToHorizonInput<B, INDEX> {
        RolloutStateNormRatioToHorizonInput::new(
            self.rollout_state_norm_ratio_to_horizon[INDEX].clone(),
        )
    }
}

impl<B: BackendTrait, const INDEX: usize> Adaptor<RolloutStateMotionToHorizonInput<B, INDEX>>
    for VisionOutput<B>
{
    fn adapt(&self) -> RolloutStateMotionToHorizonInput<B, INDEX> {
        RolloutStateMotionToHorizonInput::new(self.rollout_state_motion_to_horizon[INDEX].clone())
    }
}

impl<B: BackendTrait, const INDEX: usize> Adaptor<LongRolloutInvToHorizonInput<B, INDEX>>
    for VisionOutput<B>
{
    fn adapt(&self) -> LongRolloutInvToHorizonInput<B, INDEX> {
        LongRolloutInvToHorizonInput::new(self.long_rollout_inv_to_horizon[INDEX].clone())
    }
}

impl<B: BackendTrait, const INDEX: usize> Adaptor<LongRolloutStateNormRatioToHorizonInput<B, INDEX>>
    for VisionOutput<B>
{
    fn adapt(&self) -> LongRolloutStateNormRatioToHorizonInput<B, INDEX> {
        LongRolloutStateNormRatioToHorizonInput::new(
            self.long_rollout_state_norm_ratio_to_horizon[INDEX].clone(),
        )
    }
}

impl<B: BackendTrait, const INDEX: usize> Adaptor<LongRolloutStateMotionToHorizonInput<B, INDEX>>
    for VisionOutput<B>
{
    fn adapt(&self) -> LongRolloutStateMotionToHorizonInput<B, INDEX> {
        LongRolloutStateMotionToHorizonInput::new(
            self.long_rollout_state_motion_to_horizon[INDEX].clone(),
        )
    }
}

impl<B: BackendTrait> Adaptor<RolloutComErrorToH24Input<B>> for VisionOutput<B> {
    fn adapt(&self) -> RolloutComErrorToH24Input<B> {
        RolloutComErrorToH24Input::new(self.rollout_com_error_to_h24.clone())
    }
}

impl<B: BackendTrait> Adaptor<RolloutVelocityErrorToH24Input<B>> for VisionOutput<B> {
    fn adapt(&self) -> RolloutVelocityErrorToH24Input<B> {
        RolloutVelocityErrorToH24Input::new(self.rollout_velocity_error_to_h24.clone())
    }
}

impl<B: BackendTrait> Adaptor<LongRolloutComErrorToH24Input<B>> for VisionOutput<B> {
    fn adapt(&self) -> LongRolloutComErrorToH24Input<B> {
        LongRolloutComErrorToH24Input::new(self.long_rollout_com_error_to_h24.clone())
    }
}

impl<B: BackendTrait> Adaptor<LongRolloutVelocityErrorToH24Input<B>> for VisionOutput<B> {
    fn adapt(&self) -> LongRolloutVelocityErrorToH24Input<B> {
        LongRolloutVelocityErrorToH24Input::new(self.long_rollout_velocity_error_to_h24.clone())
    }
}

impl<B: BackendTrait> Adaptor<VisionArtifactInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> VisionArtifactInput<B> {
        self.artifacts
            .clone()
            .unwrap_or_else(VisionArtifactInput::empty)
    }
}

pub struct VisionTrainItem<B: AutodiffBackend> {
    loss: Tensor<MetricsBackend, 1>,
    inv_loss: Tensor<MetricsBackend, 1>,
    observe_loss: Tensor<MetricsBackend, 1>,
    mode_separation_ratio: Tensor<MetricsBackend, 1>,
    sigreg_loss: Tensor<MetricsBackend, 1>,
    recon_loss: Tensor<MetricsBackend, 1>,
    recon_psnr_masked: Tensor<MetricsBackend, 1>,
    recon_psnr_full: Tensor<MetricsBackend, 1>,
    policy_loss: Tensor<MetricsBackend, 1>,
    policy_advantage_abs_mean: Tensor<MetricsBackend, 1>,
    policy_advantage_std: Tensor<MetricsBackend, 1>,
    policy_log_prob_mean: Tensor<MetricsBackend, 1>,
    policy_entropy: Tensor<MetricsBackend, 1>,
    policy_action_clamp_rate: Tensor<MetricsBackend, 1>,
    probe_loss: Tensor<MetricsBackend, 1>,
    probe_acc: Tensor<MetricsBackend, 1>,
    rollout_inv_to_horizon: [Tensor<MetricsBackend, 1>; VISION_ROLLOUT_HORIZON_COUNT],
    rollout_state_norm_ratio_to_horizon: [Tensor<MetricsBackend, 1>; VISION_ROLLOUT_HORIZON_COUNT],
    rollout_state_motion_to_horizon: [Tensor<MetricsBackend, 1>; VISION_ROLLOUT_HORIZON_COUNT],
    _marker: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> VisionTrainItem<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        loss: Tensor<B, 1>,
        inv_loss: Tensor<B, 1>,
        observe_loss: Tensor<B, 1>,
        mode_separation_ratio: Tensor<B, 1>,
        sigreg_loss: Tensor<B, 1>,
        recon_loss: Tensor<B, 1>,
        recon_psnr_masked: Tensor<B, 1>,
        recon_psnr_full: Tensor<B, 1>,
        policy_loss: Tensor<B, 1>,
        policy_advantage_abs_mean: Tensor<B, 1>,
        policy_advantage_std: Tensor<B, 1>,
        policy_log_prob_mean: Tensor<B, 1>,
        policy_entropy: Tensor<B, 1>,
        policy_action_clamp_rate: Tensor<B, 1>,
        probe_loss: Tensor<B, 1>,
        probe_acc: Tensor<B, 1>,
    ) -> Self {
        let metrics_device = <MetricsBackend as BackendTrait>::Device::default();
        let rollout_inv_to_horizon =
            core::array::from_fn(|_| Tensor::<MetricsBackend, 1>::zeros([1], &metrics_device));
        let rollout_state_norm_ratio_to_horizon =
            core::array::from_fn(|_| Tensor::<MetricsBackend, 1>::zeros([1], &metrics_device));
        let rollout_state_motion_to_horizon =
            core::array::from_fn(|_| Tensor::<MetricsBackend, 1>::zeros([1], &metrics_device));
        Self {
            loss: sync_float_tensor(loss.detach().inner()),
            inv_loss: sync_float_tensor(inv_loss.detach().inner()),
            observe_loss: sync_float_tensor(observe_loss.detach().inner()),
            mode_separation_ratio: sync_float_tensor(mode_separation_ratio.detach().inner()),
            sigreg_loss: sync_float_tensor(sigreg_loss.detach().inner()),
            recon_loss: sync_float_tensor(recon_loss.detach().inner()),
            recon_psnr_masked: sync_float_tensor(recon_psnr_masked.detach().inner()),
            recon_psnr_full: sync_float_tensor(recon_psnr_full.detach().inner()),
            policy_loss: sync_float_tensor(policy_loss.detach().inner()),
            policy_advantage_abs_mean: sync_float_tensor(
                policy_advantage_abs_mean.detach().inner(),
            ),
            policy_advantage_std: sync_float_tensor(policy_advantage_std.detach().inner()),
            policy_log_prob_mean: sync_float_tensor(policy_log_prob_mean.detach().inner()),
            policy_entropy: sync_float_tensor(policy_entropy.detach().inner()),
            policy_action_clamp_rate: sync_float_tensor(policy_action_clamp_rate.detach().inner()),
            probe_loss: sync_float_tensor(probe_loss.detach().inner()),
            probe_acc: sync_float_tensor(probe_acc.detach().inner()),
            rollout_inv_to_horizon,
            rollout_state_norm_ratio_to_horizon,
            rollout_state_motion_to_horizon,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_rollout_horizon_metrics(
        mut self,
        rollout_inv_to_horizon: [Tensor<B, 1>; VISION_ROLLOUT_HORIZON_COUNT],
        rollout_state_norm_ratio_to_horizon: [Tensor<B, 1>; VISION_ROLLOUT_HORIZON_COUNT],
        rollout_state_motion_to_horizon: [Tensor<B, 1>; VISION_ROLLOUT_HORIZON_COUNT],
    ) -> Self {
        self.rollout_inv_to_horizon =
            rollout_inv_to_horizon.map(|value| sync_float_tensor(value.detach().inner()));
        self.rollout_state_norm_ratio_to_horizon = rollout_state_norm_ratio_to_horizon
            .map(|value| sync_float_tensor(value.detach().inner()));
        self.rollout_state_motion_to_horizon =
            rollout_state_motion_to_horizon.map(|value| sync_float_tensor(value.detach().inner()));
        self
    }
}

impl<B: AutodiffBackend> ItemLazy for VisionTrainItem<B> {
    type ItemSync = VisionOutput<MetricsBackend>;

    fn sync(self) -> Self::ItemSync {
        VisionOutput {
            loss: self.loss,
            inv_loss: self.inv_loss,
            observe_loss: self.observe_loss,
            mode_separation_ratio: self.mode_separation_ratio,
            sigreg_loss: self.sigreg_loss,
            recon_loss: self.recon_loss,
            recon_psnr_masked: self.recon_psnr_masked,
            recon_psnr_full: self.recon_psnr_full,
            policy_loss: self.policy_loss,
            policy_advantage_abs_mean: self.policy_advantage_abs_mean,
            policy_advantage_std: self.policy_advantage_std,
            policy_log_prob_mean: self.policy_log_prob_mean,
            policy_entropy: self.policy_entropy,
            policy_action_clamp_rate: self.policy_action_clamp_rate,
            probe_loss: self.probe_loss,
            probe_acc: self.probe_acc,
            rollout_inv_to_horizon: self.rollout_inv_to_horizon,
            rollout_state_norm_ratio_to_horizon: self.rollout_state_norm_ratio_to_horizon,
            rollout_state_motion_to_horizon: self.rollout_state_motion_to_horizon,
            rollout_com_error_to_h24: None,
            rollout_velocity_error_to_h24: None,
            long_rollout_inv_to_horizon: core::array::from_fn(|_| None),
            long_rollout_state_norm_ratio_to_horizon: core::array::from_fn(|_| None),
            long_rollout_state_motion_to_horizon: core::array::from_fn(|_| None),
            long_rollout_com_error_to_h24: None,
            long_rollout_velocity_error_to_h24: None,
            artifacts: None,
        }
    }
}
