use burn::module::{AutodiffModule, Devices, Module, ModuleMapper, ModuleVisitor};
use burn::tensor::backend::{AutodiffBackend, Backend};
use serde::{Deserialize, Serialize};

const MHC_EPS: f32 = 1e-6;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ManifoldHyperConnectionCoefficientPolicy {
    #[default]
    StaticSinkhorn,
    DynamicPositive,
}

impl ManifoldHyperConnectionCoefficientPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StaticSinkhorn => "static_sinkhorn",
            Self::DynamicPositive => "dynamic_positive",
        }
    }

    pub const fn uses_dynamic_stream_controller(self) -> bool {
        matches!(self, Self::DynamicPositive)
    }
}

impl<B: Backend> Module<B> for ManifoldHyperConnectionCoefficientPolicy {
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

impl<B: AutodiffBackend> AutodiffModule<B> for ManifoldHyperConnectionCoefficientPolicy {
    type InnerModule = ManifoldHyperConnectionCoefficientPolicy;

    fn valid(&self) -> Self::InnerModule {
        *self
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

/// Configuration for manifold-constrained hyper-connections (mHC).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ManifoldHyperConnectionsConfig {
    pub enabled: bool,
    pub num_streams: usize,
    pub num_views: usize,
    pub last_layers: Option<usize>,
    #[serde(default)]
    pub coefficient_policy: ManifoldHyperConnectionCoefficientPolicy,
    pub mhc_iters: usize,
    pub mhc_tau: f32,
    pub add_branch_out_to_residual: bool,
    pub dropout: f64,
}

impl Default for ManifoldHyperConnectionsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_streams: 1,
            num_views: 1,
            last_layers: None,
            coefficient_policy: ManifoldHyperConnectionCoefficientPolicy::StaticSinkhorn,
            mhc_iters: 10,
            mhc_tau: 0.05,
            add_branch_out_to_residual: true,
            dropout: 0.0,
        }
    }
}

impl ManifoldHyperConnectionsConfig {
    pub fn resolved_num_streams(&self) -> usize {
        self.num_streams.max(1)
    }

    pub fn resolved_num_views(&self) -> usize {
        self.num_views.max(1)
    }

    pub fn resolved_tau(&self) -> f32 {
        self.mhc_tau.max(MHC_EPS)
    }

    pub fn resolved_iters(&self) -> usize {
        self.mhc_iters.max(1)
    }
}
