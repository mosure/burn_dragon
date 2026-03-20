use burn::module::{
    AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor,
};
use burn::tensor::backend::{AutodiffBackend, Backend};
use serde::{Deserialize, Serialize};

use crate::model::config::{BDHConfig, SequenceKernelKind};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SequenceKernelFamily {
    #[default]
    LinearAttention,
    Rwkv8,
    Mamba1SelectiveSsm,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SequenceTrainingExecutor {
    #[default]
    Reference,
    DenseScoreShortContext,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct SequenceKernelConfig {
    #[serde(default)]
    pub family: SequenceKernelFamily,
    #[serde(default)]
    pub executor: SequenceTrainingExecutor,
}

impl SequenceKernelKind {
    pub fn family(self) -> SequenceKernelFamily {
        match self {
            SequenceKernelKind::BdhLinearAttention
            | SequenceKernelKind::BdhLinearDenseScoreExperimental => {
                SequenceKernelFamily::LinearAttention
            }
            SequenceKernelKind::Rwkv8StateSpaceExperimental => SequenceKernelFamily::Rwkv8,
            SequenceKernelKind::MambaSelectiveSsmExperimental => {
                SequenceKernelFamily::Mamba1SelectiveSsm
            }
        }
    }

    pub fn training_executor(self) -> SequenceTrainingExecutor {
        match self {
            SequenceKernelKind::BdhLinearAttention
            | SequenceKernelKind::Rwkv8StateSpaceExperimental
            | SequenceKernelKind::MambaSelectiveSsmExperimental => {
                SequenceTrainingExecutor::Reference
            }
            SequenceKernelKind::BdhLinearDenseScoreExperimental => {
                SequenceTrainingExecutor::DenseScoreShortContext
            }
        }
    }

    pub fn resolved_config(self) -> SequenceKernelConfig {
        SequenceKernelConfig {
            family: self.family(),
            executor: self.training_executor(),
        }
    }
}

impl SequenceKernelConfig {
    pub fn legacy_kind(self) -> Option<SequenceKernelKind> {
        match (self.family, self.executor) {
            (SequenceKernelFamily::LinearAttention, SequenceTrainingExecutor::Reference) => {
                Some(SequenceKernelKind::BdhLinearAttention)
            }
            (
                SequenceKernelFamily::LinearAttention,
                SequenceTrainingExecutor::DenseScoreShortContext,
            ) => Some(SequenceKernelKind::BdhLinearDenseScoreExperimental),
            (SequenceKernelFamily::Rwkv8, SequenceTrainingExecutor::Reference) => {
                Some(SequenceKernelKind::Rwkv8StateSpaceExperimental)
            }
            (SequenceKernelFamily::Mamba1SelectiveSsm, SequenceTrainingExecutor::Reference) => {
                Some(SequenceKernelKind::MambaSelectiveSsmExperimental)
            }
            _ => None,
        }
    }
}

impl BDHConfig {
    pub fn resolved_sequence_kernel_config(&self) -> SequenceKernelConfig {
        self.sequence_kernel.resolved_config()
    }
}

impl<B: Backend> Module<B> for SequenceKernelFamily {
    type Record = ();

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for SequenceKernelFamily {
    type InnerModule = SequenceKernelFamily;

    fn valid(&self) -> Self::InnerModule {
        *self
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for SequenceKernelFamily {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .set_top_level_type("SequenceKernelFamily")
            .add_formatted(&format!("{self:?}"))
            .optional()
    }
}

impl ModuleDisplay for SequenceKernelFamily {}

impl<B: Backend> Module<B> for SequenceTrainingExecutor {
    type Record = ();

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for SequenceTrainingExecutor {
    type InnerModule = SequenceTrainingExecutor;

    fn valid(&self) -> Self::InnerModule {
        *self
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for SequenceTrainingExecutor {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .set_top_level_type("SequenceTrainingExecutor")
            .add_formatted(&format!("{self:?}"))
            .optional()
    }
}

impl ModuleDisplay for SequenceTrainingExecutor {}

impl<B: Backend> Module<B> for SequenceKernelConfig {
    type Record = ();

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for SequenceKernelConfig {
    type InnerModule = SequenceKernelConfig;

    fn valid(&self) -> Self::InnerModule {
        *self
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for SequenceKernelConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .set_top_level_type("SequenceKernelConfig")
            .add_formatted(&format!(
                "family={:?}, executor={:?}",
                self.family, self.executor
            ))
            .optional()
    }
}

impl ModuleDisplay for SequenceKernelConfig {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_sequence_kernel_kind_round_trips_to_family_executor_split() {
        for kind in [
            SequenceKernelKind::BdhLinearAttention,
            SequenceKernelKind::BdhLinearDenseScoreExperimental,
            SequenceKernelKind::Rwkv8StateSpaceExperimental,
            SequenceKernelKind::MambaSelectiveSsmExperimental,
        ] {
            let resolved = kind.resolved_config();
            assert_eq!(resolved.legacy_kind(), Some(kind));
        }
    }

    #[test]
    fn bdh_config_resolves_legacy_sequence_kernel() {
        let config = BDHConfig {
            sequence_kernel: SequenceKernelKind::BdhLinearDenseScoreExperimental,
            ..Default::default()
        };

        assert_eq!(
            config.resolved_sequence_kernel_config(),
            SequenceKernelConfig {
                family: SequenceKernelFamily::LinearAttention,
                executor: SequenceTrainingExecutor::DenseScoreShortContext,
            }
        );
    }
}
