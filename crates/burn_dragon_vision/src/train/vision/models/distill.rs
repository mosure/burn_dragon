use crate::model::vision::VisionProjectionHead;
use crate::train::prelude::*;

#[cfg(feature = "burn_dino")]
pub(crate) type DistillTeacherModel<B> = DinoVisionTransformer<B>;
#[cfg(not(feature = "burn_dino"))]
#[derive(Debug, Clone)]
pub(crate) struct DistillTeacherModel<B: BackendTrait> {
    _marker: core::marker::PhantomData<B>,
}

fn default_auxiliary_decoder_hidden_dim(shared_dim: usize, target_dim: usize) -> usize {
    shared_dim.max(target_dim).saturating_mul(2).max(1)
}

#[derive(Module, Debug)]
pub(crate) struct VisionTeacherDecoderHead<B: BackendTrait> {
    pub(crate) patch_head: Option<VisionProjectionHead<B>>,
    pub(crate) cls_head: VisionProjectionHead<B>,
}

impl<B: BackendTrait> VisionTeacherDecoderHead<B> {
    fn from_target(
        target: &VisionTeacherTargetConfig,
        shared_dim: usize,
        norm_config: &DragonNormConfig,
        dropout: f64,
        device: &B::Device,
    ) -> Option<Self> {
        if !target.decoder_mode.uses_dedicated_projection() {
            return None;
        }

        let (target_dim, target_patch_tokens) = match &target.teacher {
            VisionTeacherConfig::Features(teacher) => (teacher.feature_dim, teacher.patch_tokens),
            VisionTeacherConfig::Model(teacher) => (
                teacher.feature_dim.unwrap_or(shared_dim),
                teacher.patch_tokens,
            ),
        };
        let hidden_dim = target
            .decoder_hidden_dim
            .unwrap_or_else(|| default_auxiliary_decoder_hidden_dim(shared_dim, target_dim));
        let patch_head =
            matches!(target.target_kind, VisionTeacherTargetKind::PatchAndCls).then(|| {
                VisionProjectionHead::new(
                    shared_dim,
                    hidden_dim,
                    target_dim,
                    dropout,
                    norm_config,
                    device,
                )
            });
        let cls_head = VisionProjectionHead::new(
            shared_dim,
            hidden_dim,
            target_dim,
            dropout,
            norm_config,
            device,
        );

        let _ = target_patch_tokens;

        Some(Self {
            patch_head,
            cls_head,
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct VisionDistillModel<B: BackendTrait> {
    pub(crate) model: VisionDragon<B>,
    pub(crate) loss: VisionDistillationLossConfig,
    pub(crate) teacher: Option<DistillTeacherModel<B>>,
    pub(crate) auxiliary_teacher_heads: Vec<VisionTeacherDecoderHead<B>>,
    pub(crate) auxiliary_teacher_head_names: Vec<String>,
    pub(crate) auxiliary_teacher_head_modes: Vec<VisionTeacherDecoderMode>,
    pub(crate) auxiliary_teacher_head_patch_tokens: Vec<Option<usize>>,
    pub(crate) rollout: VisionRollout,
    pub(crate) rollout_supervision_frames: usize,
    pub(crate) rollout_supervision_stride: usize,
    pub(crate) rollout_supervision_groups: usize,
    pub(crate) rollout_supervision_explicit_steps: Vec<usize>,
    pub(crate) rollout_supervision_explicit_groups: Vec<Vec<usize>>,
    pub(crate) rollout_supervision_include_step1: bool,
    pub(crate) rollout_supervision_power: f32,
    pub(crate) rollout_sampling_power: f32,
    pub(crate) rollout_improvement_weight: f32,
    pub(crate) rollout_improvement_margin: f32,
}

impl<B: BackendTrait> VisionDistillModel<B> {
    pub(crate) fn new(
        model: VisionDragon<B>,
        config: VisionDistillConfig,
        teacher: Option<DistillTeacherModel<B>>,
        rollout: VisionRollout,
        device: &B::Device,
    ) -> Self {
        let auxiliary_teacher_head_names = config
            .teacher_targets
            .iter()
            .filter(|target| target.decoder_mode.uses_dedicated_projection())
            .map(|target| target.name.clone())
            .collect::<Vec<_>>();
        let auxiliary_teacher_head_modes = config
            .teacher_targets
            .iter()
            .filter(|target| target.decoder_mode.uses_dedicated_projection())
            .map(|target| target.decoder_mode)
            .collect::<Vec<_>>();
        let auxiliary_teacher_head_patch_tokens = config
            .teacher_targets
            .iter()
            .filter(|target| target.decoder_mode.uses_dedicated_projection())
            .map(|target| match &target.teacher {
                VisionTeacherConfig::Features(teacher) => teacher.patch_tokens,
                VisionTeacherConfig::Model(teacher) => teacher.patch_tokens,
            })
            .collect::<Vec<_>>();
        let auxiliary_teacher_heads = config
            .teacher_targets
            .iter()
            .filter_map(|target| {
                VisionTeacherDecoderHead::from_target(
                    target,
                    model.projection_dim(),
                    model.normalization_config(),
                    0.0,
                    device,
                )
            })
            .collect::<Vec<_>>();
        let mut rollout_supervision_explicit_steps = config
            .rollout_supervision_explicit_steps
            .into_iter()
            .filter(|step| *step > 0)
            .collect::<Vec<_>>();
        rollout_supervision_explicit_steps.sort_unstable();
        rollout_supervision_explicit_steps.dedup();
        let rollout_supervision_explicit_groups = config
            .rollout_supervision_explicit_groups
            .into_iter()
            .map(|group| {
                let mut group = group
                    .into_iter()
                    .filter(|step| *step > 0)
                    .collect::<Vec<_>>();
                group.sort_unstable();
                group.dedup();
                group
            })
            .filter(|group| !group.is_empty())
            .collect::<Vec<_>>();
        Self {
            model,
            loss: config.loss,
            teacher,
            auxiliary_teacher_heads,
            auxiliary_teacher_head_names,
            auxiliary_teacher_head_modes,
            auxiliary_teacher_head_patch_tokens,
            rollout,
            rollout_supervision_frames: config.rollout_supervision_frames.max(1),
            rollout_supervision_stride: config.rollout_supervision_stride.max(1),
            rollout_supervision_groups: config.rollout_supervision_groups.max(1),
            rollout_supervision_explicit_steps,
            rollout_supervision_explicit_groups,
            rollout_supervision_include_step1: config.rollout_supervision_include_step1,
            rollout_supervision_power: config.rollout_supervision_power.max(0.0),
            rollout_sampling_power: config.rollout_sampling_power.max(0.0),
            rollout_improvement_weight: config.rollout_improvement_weight.max(0.0),
            rollout_improvement_margin: config.rollout_improvement_margin.max(0.0),
        }
    }

    pub(crate) fn auxiliary_teacher_head(
        &self,
        name: &str,
    ) -> Option<(
        &VisionTeacherDecoderHead<B>,
        VisionTeacherDecoderMode,
        Option<usize>,
    )> {
        self.auxiliary_teacher_head_names
            .iter()
            .position(|candidate| candidate == name)
            .and_then(|index| {
                self.auxiliary_teacher_heads.get(index).map(|head| {
                    (
                        head,
                        self.auxiliary_teacher_head_modes[index],
                        self.auxiliary_teacher_head_patch_tokens[index],
                    )
                })
            })
    }
}

#[derive(burn::record::Record)]
pub(crate) struct VisionDistillModelRecord<B: BackendTrait> {
    pub(crate) model: <VisionDragon<B> as Module<B>>::Record,
    pub(crate) loss: <VisionDistillationLossConfig as Module<B>>::Record,
    pub(crate) auxiliary_teacher_heads: <Vec<VisionTeacherDecoderHead<B>> as Module<B>>::Record,
}

impl<B: BackendTrait> Module<B> for VisionDistillModel<B> {
    type Record = VisionDistillModelRecord<B>;

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        let devices = Module::collect_devices(&self.model, devices);
        let devices = Module::<B>::collect_devices(&self.loss, devices);
        Module::collect_devices(&self.auxiliary_teacher_heads, devices)
    }

    fn fork(self, device: &B::Device) -> Self {
        Self {
            model: Module::fork(self.model, device),
            loss: Module::<B>::fork(self.loss, device),
            teacher: self.teacher,
            auxiliary_teacher_heads: Module::fork(self.auxiliary_teacher_heads, device),
            auxiliary_teacher_head_names: self.auxiliary_teacher_head_names,
            auxiliary_teacher_head_modes: self.auxiliary_teacher_head_modes,
            auxiliary_teacher_head_patch_tokens: self.auxiliary_teacher_head_patch_tokens,
            rollout: self.rollout,
            rollout_supervision_frames: self.rollout_supervision_frames,
            rollout_supervision_stride: self.rollout_supervision_stride,
            rollout_supervision_groups: self.rollout_supervision_groups,
            rollout_supervision_explicit_steps: self.rollout_supervision_explicit_steps,
            rollout_supervision_explicit_groups: self.rollout_supervision_explicit_groups,
            rollout_supervision_include_step1: self.rollout_supervision_include_step1,
            rollout_supervision_power: self.rollout_supervision_power,
            rollout_sampling_power: self.rollout_sampling_power,
            rollout_improvement_weight: self.rollout_improvement_weight,
            rollout_improvement_margin: self.rollout_improvement_margin,
        }
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            model: Module::to_device(self.model, device),
            loss: Module::<B>::to_device(self.loss, device),
            teacher: self.teacher,
            auxiliary_teacher_heads: Module::to_device(self.auxiliary_teacher_heads, device),
            auxiliary_teacher_head_names: self.auxiliary_teacher_head_names,
            auxiliary_teacher_head_modes: self.auxiliary_teacher_head_modes,
            auxiliary_teacher_head_patch_tokens: self.auxiliary_teacher_head_patch_tokens,
            rollout: self.rollout,
            rollout_supervision_frames: self.rollout_supervision_frames,
            rollout_supervision_stride: self.rollout_supervision_stride,
            rollout_supervision_groups: self.rollout_supervision_groups,
            rollout_supervision_explicit_steps: self.rollout_supervision_explicit_steps,
            rollout_supervision_explicit_groups: self.rollout_supervision_explicit_groups,
            rollout_supervision_include_step1: self.rollout_supervision_include_step1,
            rollout_supervision_power: self.rollout_supervision_power,
            rollout_sampling_power: self.rollout_sampling_power,
            rollout_improvement_weight: self.rollout_improvement_weight,
            rollout_improvement_margin: self.rollout_improvement_margin,
        }
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, visitor: &mut Visitor) {
        self.model.visit(visitor);
        self.loss.visit(visitor);
        self.auxiliary_teacher_heads.visit(visitor);
    }

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, mapper: &mut Mapper) -> Self {
        Self {
            model: Module::map(self.model, mapper),
            loss: Module::map(self.loss, mapper),
            teacher: self.teacher,
            auxiliary_teacher_heads: Module::map(self.auxiliary_teacher_heads, mapper),
            auxiliary_teacher_head_names: self.auxiliary_teacher_head_names,
            auxiliary_teacher_head_modes: self.auxiliary_teacher_head_modes,
            auxiliary_teacher_head_patch_tokens: self.auxiliary_teacher_head_patch_tokens,
            rollout: self.rollout,
            rollout_supervision_frames: self.rollout_supervision_frames,
            rollout_supervision_stride: self.rollout_supervision_stride,
            rollout_supervision_groups: self.rollout_supervision_groups,
            rollout_supervision_explicit_steps: self.rollout_supervision_explicit_steps,
            rollout_supervision_explicit_groups: self.rollout_supervision_explicit_groups,
            rollout_supervision_include_step1: self.rollout_supervision_include_step1,
            rollout_supervision_power: self.rollout_supervision_power,
            rollout_sampling_power: self.rollout_sampling_power,
            rollout_improvement_weight: self.rollout_improvement_weight,
            rollout_improvement_margin: self.rollout_improvement_margin,
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            model: Module::load_record(self.model, record.model),
            loss: {
                let _: () = record.loss;
                Module::<B>::load_record(self.loss, ())
            },
            teacher: self.teacher,
            auxiliary_teacher_heads: Module::load_record(
                self.auxiliary_teacher_heads,
                record.auxiliary_teacher_heads,
            ),
            auxiliary_teacher_head_names: self.auxiliary_teacher_head_names,
            auxiliary_teacher_head_modes: self.auxiliary_teacher_head_modes,
            auxiliary_teacher_head_patch_tokens: self.auxiliary_teacher_head_patch_tokens,
            rollout: self.rollout,
            rollout_supervision_frames: self.rollout_supervision_frames,
            rollout_supervision_stride: self.rollout_supervision_stride,
            rollout_supervision_groups: self.rollout_supervision_groups,
            rollout_supervision_explicit_steps: self.rollout_supervision_explicit_steps,
            rollout_supervision_explicit_groups: self.rollout_supervision_explicit_groups,
            rollout_supervision_include_step1: self.rollout_supervision_include_step1,
            rollout_supervision_power: self.rollout_supervision_power,
            rollout_sampling_power: self.rollout_sampling_power,
            rollout_improvement_weight: self.rollout_improvement_weight,
            rollout_improvement_margin: self.rollout_improvement_margin,
        }
    }

    fn into_record(self) -> Self::Record {
        VisionDistillModelRecord {
            model: Module::into_record(self.model),
            loss: Module::<B>::into_record(self.loss),
            auxiliary_teacher_heads: Module::into_record(self.auxiliary_teacher_heads),
        }
    }
}

impl<B: AutodiffBackend> AutodiffModule<B> for VisionDistillModel<B> {
    type InnerModule = VisionDistillModel<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        VisionDistillModel {
            model: AutodiffModule::valid(&self.model),
            loss: AutodiffModule::<B>::valid(&self.loss),
            teacher: None,
            auxiliary_teacher_heads: AutodiffModule::valid(&self.auxiliary_teacher_heads),
            auxiliary_teacher_head_names: self.auxiliary_teacher_head_names.clone(),
            auxiliary_teacher_head_modes: self.auxiliary_teacher_head_modes.clone(),
            auxiliary_teacher_head_patch_tokens: self.auxiliary_teacher_head_patch_tokens.clone(),
            rollout: self.rollout,
            rollout_supervision_frames: self.rollout_supervision_frames,
            rollout_supervision_stride: self.rollout_supervision_stride,
            rollout_supervision_groups: self.rollout_supervision_groups,
            rollout_supervision_explicit_steps: self.rollout_supervision_explicit_steps.clone(),
            rollout_supervision_explicit_groups: self.rollout_supervision_explicit_groups.clone(),
            rollout_supervision_include_step1: self.rollout_supervision_include_step1,
            rollout_supervision_power: self.rollout_supervision_power,
            rollout_sampling_power: self.rollout_sampling_power,
            rollout_improvement_weight: self.rollout_improvement_weight,
            rollout_improvement_margin: self.rollout_improvement_margin,
        }
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        VisionDistillModel {
            model: AutodiffModule::from_inner(module.model),
            loss: AutodiffModule::<B>::from_inner(module.loss),
            teacher: None,
            auxiliary_teacher_heads: AutodiffModule::from_inner(module.auxiliary_teacher_heads),
            auxiliary_teacher_head_names: module.auxiliary_teacher_head_names,
            auxiliary_teacher_head_modes: module.auxiliary_teacher_head_modes,
            auxiliary_teacher_head_patch_tokens: module.auxiliary_teacher_head_patch_tokens,
            rollout: module.rollout,
            rollout_supervision_frames: module.rollout_supervision_frames,
            rollout_supervision_stride: module.rollout_supervision_stride,
            rollout_supervision_groups: module.rollout_supervision_groups,
            rollout_supervision_explicit_steps: module.rollout_supervision_explicit_steps,
            rollout_supervision_explicit_groups: module.rollout_supervision_explicit_groups,
            rollout_supervision_include_step1: module.rollout_supervision_include_step1,
            rollout_supervision_power: module.rollout_supervision_power,
            rollout_sampling_power: module.rollout_sampling_power,
            rollout_improvement_weight: module.rollout_improvement_weight,
            rollout_improvement_margin: module.rollout_improvement_margin,
        }
    }
}

impl<B: BackendTrait> core::fmt::Display for VisionDistillModel<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&burn::module::ModuleDisplay::format(
            self,
            burn::module::DisplaySettings::default(),
        ))
    }
}

impl<B: BackendTrait> ModuleDisplayDefault for VisionDistillModel<B> {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("model", &self.model)
            .add("loss", &self.loss)
            .add("auxiliary_teacher_heads", &self.auxiliary_teacher_heads)
            .add(
                "rollout_supervision_frames",
                &self.rollout_supervision_frames,
            )
            .add(
                "rollout_supervision_stride",
                &self.rollout_supervision_stride,
            )
            .add(
                "rollout_supervision_groups",
                &self.rollout_supervision_groups,
            )
            .add(
                "rollout_supervision_explicit_steps",
                &self.rollout_supervision_explicit_steps,
            )
            .add(
                "rollout_supervision_explicit_groups",
                &self.rollout_supervision_explicit_groups,
            )
            .add(
                "rollout_supervision_include_step1",
                &self.rollout_supervision_include_step1,
            )
            .add("rollout_supervision_power", &self.rollout_supervision_power)
            .add("rollout_sampling_power", &self.rollout_sampling_power)
            .add(
                "rollout_improvement_weight",
                &self.rollout_improvement_weight,
            )
            .add(
                "rollout_improvement_margin",
                &self.rollout_improvement_margin,
            )
            .optional()
    }
}

impl<B: BackendTrait> ModuleDisplay for VisionDistillModel<B> {}
