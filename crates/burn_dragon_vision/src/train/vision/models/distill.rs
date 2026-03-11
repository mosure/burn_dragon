use crate::train::prelude::*;

#[cfg(feature = "burn_dino")]
pub(crate) type DistillTeacherModel<B> = DinoVisionTransformer<B>;
#[cfg(not(feature = "burn_dino"))]
#[derive(Debug, Clone)]
pub(crate) struct DistillTeacherModel<B: BackendTrait> {
    _marker: core::marker::PhantomData<B>,
}

#[derive(Debug, Clone)]
pub(crate) struct VisionDistillModel<B: BackendTrait> {
    pub(crate) model: VisionDragon<B>,
    pub(crate) loss: VisionDistillationLossConfig,
    pub(crate) teacher: Option<DistillTeacherModel<B>>,
    pub(crate) rollout: VisionRollout,
    pub(crate) rollout_supervision_frames: usize,
    pub(crate) rollout_supervision_power: f32,
    pub(crate) rollout_sampling_power: f32,
}

impl<B: BackendTrait> VisionDistillModel<B> {
    pub(crate) fn new(
        model: VisionDragon<B>,
        config: VisionDistillConfig,
        teacher: Option<DistillTeacherModel<B>>,
        rollout: VisionRollout,
    ) -> Self {
        Self {
            model,
            loss: config.loss,
            teacher,
            rollout,
            rollout_supervision_frames: config.rollout_supervision_frames.max(1),
            rollout_supervision_power: config.rollout_supervision_power.max(0.0),
            rollout_sampling_power: config.rollout_sampling_power.max(0.0),
        }
    }
}

#[derive(burn::record::Record)]
pub(crate) struct VisionDistillModelRecord<B: BackendTrait> {
    pub(crate) model: <VisionDragon<B> as Module<B>>::Record,
    pub(crate) loss: <VisionDistillationLossConfig as Module<B>>::Record,
}

impl<B: BackendTrait> Module<B> for VisionDistillModel<B> {
    type Record = VisionDistillModelRecord<B>;

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        let devices = Module::collect_devices(&self.model, devices);
        Module::<B>::collect_devices(&self.loss, devices)
    }

    fn fork(self, device: &B::Device) -> Self {
        Self {
            model: Module::fork(self.model, device),
            loss: Module::<B>::fork(self.loss, device),
            teacher: self.teacher,
            rollout: self.rollout,
            rollout_supervision_frames: self.rollout_supervision_frames,
            rollout_supervision_power: self.rollout_supervision_power,
            rollout_sampling_power: self.rollout_sampling_power,
        }
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            model: Module::to_device(self.model, device),
            loss: Module::<B>::to_device(self.loss, device),
            teacher: self.teacher,
            rollout: self.rollout,
            rollout_supervision_frames: self.rollout_supervision_frames,
            rollout_supervision_power: self.rollout_supervision_power,
            rollout_sampling_power: self.rollout_sampling_power,
        }
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, visitor: &mut Visitor) {
        self.model.visit(visitor);
        self.loss.visit(visitor);
    }

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, mapper: &mut Mapper) -> Self {
        Self {
            model: Module::map(self.model, mapper),
            loss: Module::map(self.loss, mapper),
            teacher: self.teacher,
            rollout: self.rollout,
            rollout_supervision_frames: self.rollout_supervision_frames,
            rollout_supervision_power: self.rollout_supervision_power,
            rollout_sampling_power: self.rollout_sampling_power,
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            model: Module::load_record(self.model, record.model),
            loss: Module::<B>::load_record(self.loss, record.loss),
            teacher: self.teacher,
            rollout: self.rollout,
            rollout_supervision_frames: self.rollout_supervision_frames,
            rollout_supervision_power: self.rollout_supervision_power,
            rollout_sampling_power: self.rollout_sampling_power,
        }
    }

    fn into_record(self) -> Self::Record {
        VisionDistillModelRecord {
            model: Module::into_record(self.model),
            loss: Module::<B>::into_record(self.loss),
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
            rollout: self.rollout,
            rollout_supervision_frames: self.rollout_supervision_frames,
            rollout_supervision_power: self.rollout_supervision_power,
            rollout_sampling_power: self.rollout_sampling_power,
        }
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        VisionDistillModel {
            model: AutodiffModule::from_inner(module.model),
            loss: AutodiffModule::<B>::from_inner(module.loss),
            teacher: None,
            rollout: module.rollout,
            rollout_supervision_frames: module.rollout_supervision_frames,
            rollout_supervision_power: module.rollout_supervision_power,
            rollout_sampling_power: module.rollout_sampling_power,
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
            .add(
                "rollout_supervision_frames",
                &self.rollout_supervision_frames,
            )
            .add("rollout_supervision_power", &self.rollout_supervision_power)
            .add("rollout_sampling_power", &self.rollout_sampling_power)
            .optional()
    }
}

impl<B: BackendTrait> ModuleDisplay for VisionDistillModel<B> {}
