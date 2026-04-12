use burn::module::{
    AutodiffModule, Content, Devices, Initializer, Module, ModuleDisplay, ModuleDisplayDefault,
    ModuleMapper, ModuleVisitor,
};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Distribution as TensorDistribution, Tensor, TensorData};
use rand::Rng;
use serde::{Deserialize, Serialize};

const CONTROLLED_INIT_STD_CAP: f64 = 0.02;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BdhInitializationKind {
    #[default]
    NearCritical,
    SimpleNormal,
    HeGlorot,
    HeadwiseSemiOrthogonal,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BdhResidualScalingKind {
    #[default]
    FamilyDefault,
    Disabled,
    DepthScaled,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct BdhResidualScalingConfig {
    #[serde(default)]
    pub kind: BdhResidualScalingKind,
    #[serde(default = "default_residual_scaling_gain")]
    pub gain: f64,
}

impl Default for BdhResidualScalingConfig {
    fn default() -> Self {
        Self {
            kind: BdhResidualScalingKind::default(),
            gain: default_residual_scaling_gain(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BdhNeuronGainKind {
    #[default]
    Iid,
    HeavyTailedLogNormal,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BdhTopologyPriorKind {
    #[default]
    Iid,
    ModularBridges,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BdhFiringTargetKind {
    #[default]
    Disabled,
    GaussianEstimate,
    ExplicitThresholds,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct BdhFiringTargetConfig {
    #[serde(default)]
    pub kind: BdhFiringTargetKind,
    #[serde(default = "default_x_firing_target")]
    pub x_target: f64,
    #[serde(default = "default_y_firing_target")]
    pub y_target: f64,
    #[serde(default)]
    pub x_threshold: f64,
    #[serde(default)]
    pub y_threshold: f64,
}

impl Default for BdhFiringTargetConfig {
    fn default() -> Self {
        Self {
            kind: BdhFiringTargetKind::default(),
            x_target: default_x_firing_target(),
            y_target: default_y_firing_target(),
            x_threshold: 0.0,
            y_threshold: 0.0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct BdhTopologyPriorConfig {
    #[serde(default)]
    pub kind: BdhTopologyPriorKind,
    #[serde(default = "default_topology_community_count")]
    pub community_count: usize,
    #[serde(default = "default_topology_bridge_fraction")]
    pub bridge_fraction: f64,
    #[serde(default = "default_topology_intra_community_gain")]
    pub intra_community_gain: f64,
    #[serde(default = "default_topology_inter_community_gain")]
    pub inter_community_gain: f64,
    #[serde(default = "default_topology_bridge_gain")]
    pub bridge_gain: f64,
}

impl Default for BdhTopologyPriorConfig {
    fn default() -> Self {
        Self {
            kind: BdhTopologyPriorKind::default(),
            community_count: default_topology_community_count(),
            bridge_fraction: default_topology_bridge_fraction(),
            intra_community_gain: default_topology_intra_community_gain(),
            inter_community_gain: default_topology_inter_community_gain(),
            bridge_gain: default_topology_bridge_gain(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct BdhNeuronGainConfig {
    #[serde(default)]
    pub kind: BdhNeuronGainKind,
    #[serde(default = "default_neuron_gain_log_sigma")]
    pub log_sigma: f64,
    #[serde(default = "default_neuron_gain_max")]
    pub max_gain: f64,
}

impl Default for BdhNeuronGainConfig {
    fn default() -> Self {
        Self {
            kind: BdhNeuronGainKind::default(),
            log_sigma: default_neuron_gain_log_sigma(),
            max_gain: default_neuron_gain_max(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct BdhInitializationConfig {
    #[serde(default)]
    pub kind: BdhInitializationKind,
    #[serde(default)]
    pub residual_scaling: BdhResidualScalingConfig,
    #[serde(default)]
    pub neuron_gains: BdhNeuronGainConfig,
    #[serde(default)]
    pub topology_prior: BdhTopologyPriorConfig,
    #[serde(default)]
    pub firing_targets: BdhFiringTargetConfig,
    #[serde(default = "default_simple_normal_std")]
    pub simple_normal_std: f64,
}

impl Default for BdhInitializationConfig {
    fn default() -> Self {
        Self {
            kind: BdhInitializationKind::SimpleNormal,
            residual_scaling: BdhResidualScalingConfig {
                kind: BdhResidualScalingKind::DepthScaled,
                ..Default::default()
            },
            neuron_gains: BdhNeuronGainConfig {
                kind: BdhNeuronGainKind::HeavyTailedLogNormal,
                ..Default::default()
            },
            topology_prior: BdhTopologyPriorConfig::default(),
            firing_targets: BdhFiringTargetConfig::default(),
            simple_normal_std: default_simple_normal_std(),
        }
    }
}

impl BdhInitializationConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.simple_normal_std.is_finite() || self.simple_normal_std <= 0.0 {
            return Err(format!(
                "model.initialization.simple_normal_std must be finite and > 0 (got {})",
                self.simple_normal_std
            ));
        }
        if !self.residual_scaling.gain.is_finite() || self.residual_scaling.gain <= 0.0 {
            return Err(format!(
                "model.initialization.residual_scaling.gain must be finite and > 0 (got {})",
                self.residual_scaling.gain
            ));
        }
        if !self.neuron_gains.log_sigma.is_finite() || self.neuron_gains.log_sigma < 0.0 {
            return Err(format!(
                "model.initialization.neuron_gains.log_sigma must be finite and >= 0 (got {})",
                self.neuron_gains.log_sigma
            ));
        }
        if !self.neuron_gains.max_gain.is_finite() || self.neuron_gains.max_gain <= 0.0 {
            return Err(format!(
                "model.initialization.neuron_gains.max_gain must be finite and > 0 (got {})",
                self.neuron_gains.max_gain
            ));
        }
        match self.topology_prior.kind {
            BdhTopologyPriorKind::Iid => {}
            BdhTopologyPriorKind::ModularBridges => {
                if self.topology_prior.community_count == 0 {
                    return Err(
                        "model.initialization.topology_prior.community_count must be > 0".into(),
                    );
                }
                validate_probability_inclusive(
                    self.topology_prior.bridge_fraction,
                    "model.initialization.topology_prior.bridge_fraction",
                )?;
                validate_positive_finite(
                    self.topology_prior.intra_community_gain,
                    "model.initialization.topology_prior.intra_community_gain",
                )?;
                validate_positive_finite(
                    self.topology_prior.inter_community_gain,
                    "model.initialization.topology_prior.inter_community_gain",
                )?;
                validate_positive_finite(
                    self.topology_prior.bridge_gain,
                    "model.initialization.topology_prior.bridge_gain",
                )?;
            }
        }
        match self.firing_targets.kind {
            BdhFiringTargetKind::Disabled => {}
            BdhFiringTargetKind::GaussianEstimate => {
                validate_probability(
                    self.firing_targets.x_target,
                    "model.initialization.firing_targets.x_target",
                )?;
                validate_probability(
                    self.firing_targets.y_target,
                    "model.initialization.firing_targets.y_target",
                )?;
            }
            BdhFiringTargetKind::ExplicitThresholds => {
                validate_finite(
                    self.firing_targets.x_threshold,
                    "model.initialization.firing_targets.x_threshold",
                )?;
                validate_finite(
                    self.firing_targets.y_threshold,
                    "model.initialization.firing_targets.y_threshold",
                )?;
            }
        }
        Ok(())
    }
}

impl<B: Backend> Module<B> for BdhInitializationKind {
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

impl<B: AutodiffBackend> AutodiffModule<B> for BdhInitializationKind {
    type InnerModule = BdhInitializationKind;

    fn valid(&self) -> Self::InnerModule {
        *self
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for BdhInitializationKind {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .set_top_level_type("BdhInitializationKind")
            .add_formatted(&format!("{self:?}"))
            .optional()
    }
}

impl ModuleDisplay for BdhInitializationKind {}

impl<B: Backend> Module<B> for BdhInitializationConfig {
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

impl<B: AutodiffBackend> AutodiffModule<B> for BdhInitializationConfig {
    type InnerModule = BdhInitializationConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for BdhInitializationConfig {
    fn content(&self, content: Content) -> Option<Content> {
        let summary = format!(
            "kind={:?}, residual_scaling={:?}, neuron_gains={:?}, topology_prior={:?}, firing_targets={:?}, simple_normal_std={}",
            self.kind,
            self.residual_scaling.kind,
            self.neuron_gains.kind,
            self.topology_prior.kind,
            self.firing_targets.kind,
            self.simple_normal_std
        );
        content
            .set_top_level_type("BdhInitializationConfig")
            .add_formatted(&summary)
            .optional()
    }
}

impl ModuleDisplay for BdhInitializationConfig {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BdhProjectionRole {
    Encoder,
    EncoderValue,
    Decoder,
    LmHead,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BdhActivationThresholds {
    pub x: f32,
    pub y: f32,
}

impl BdhProjectionRole {
    fn is_residual_branch(self) -> bool {
        matches!(
            self,
            BdhProjectionRole::Encoder
                | BdhProjectionRole::EncoderValue
                | BdhProjectionRole::Decoder
        )
    }

    fn supports_neuron_gain_prior(self) -> bool {
        matches!(
            self,
            BdhProjectionRole::Encoder
                | BdhProjectionRole::EncoderValue
                | BdhProjectionRole::Decoder
        )
    }

    fn supports_topology_prior(self) -> bool {
        matches!(
            self,
            BdhProjectionRole::Encoder
                | BdhProjectionRole::EncoderValue
                | BdhProjectionRole::Decoder
        )
    }
}

pub struct BdhInitializer<'a> {
    config: &'a BdhInitializationConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BdhTopologyLatentAxis {
    Rows,
    Cols,
}

impl<'a> BdhInitializer<'a> {
    pub fn new(config: &'a BdhInitializationConfig) -> Self {
        Self { config }
    }

    pub fn embedding_initializer(&self, width: usize) -> Initializer {
        Initializer::Normal {
            mean: 0.0,
            std: self.embedding_std(width),
        }
    }

    pub fn headwise_projection_tensor<B: Backend>(
        &self,
        role: BdhProjectionRole,
        heads: usize,
        fan_in: usize,
        fan_out: usize,
        residual_depth: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let tensor = match self.config.kind {
            BdhInitializationKind::HeadwiseSemiOrthogonal => {
                let target_std = self.projection_std(role, fan_in, fan_out, residual_depth);
                let mut values = Vec::with_capacity(heads * fan_in * fan_out);
                for _ in 0..heads {
                    values.extend(make_semi_orthogonal_values::<B>(
                        fan_in, fan_out, target_std, device,
                    ));
                }
                Tensor::<B, 3>::from_data(TensorData::new(values, [heads, fan_in, fan_out]), device)
            }
            _ => Tensor::<B, 3>::random(
                [heads, fan_in, fan_out],
                TensorDistribution::Normal(
                    0.0,
                    self.projection_std(role, fan_in, fan_out, residual_depth),
                ),
                device,
            ),
        };

        let tensor = self.apply_headwise_neuron_gains(role, tensor, heads, fan_out, device);
        self.apply_headwise_topology_prior(role, tensor, heads, fan_in, fan_out, device)
    }

    pub fn projection_tensor<B: Backend>(
        &self,
        role: BdhProjectionRole,
        fan_in: usize,
        fan_out: usize,
        residual_depth: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let tensor = match self.config.kind {
            BdhInitializationKind::HeadwiseSemiOrthogonal
                if matches!(role, BdhProjectionRole::Decoder) =>
            {
                Tensor::<B, 2>::from_data(
                    TensorData::new(
                        make_semi_orthogonal_values::<B>(
                            fan_in,
                            fan_out,
                            self.projection_std(role, fan_in, fan_out, residual_depth),
                            device,
                        ),
                        [fan_in, fan_out],
                    ),
                    device,
                )
            }
            _ => Tensor::<B, 2>::random(
                [fan_in, fan_out],
                TensorDistribution::Normal(
                    0.0,
                    self.projection_std(role, fan_in, fan_out, residual_depth),
                ),
                device,
            ),
        };

        let tensor = self.apply_projection_neuron_gains(role, tensor, fan_in, device);
        self.apply_projection_topology_prior(role, tensor, fan_in, fan_out, device)
    }

    pub fn activation_thresholds(
        &self,
        fan_in: usize,
        fan_out: usize,
        residual_depth: usize,
    ) -> BdhActivationThresholds {
        match self.config.firing_targets.kind {
            BdhFiringTargetKind::Disabled => BdhActivationThresholds::default(),
            BdhFiringTargetKind::ExplicitThresholds => BdhActivationThresholds {
                x: self.config.firing_targets.x_threshold as f32,
                y: self.config.firing_targets.y_threshold as f32,
            },
            BdhFiringTargetKind::GaussianEstimate => {
                let x_sigma = self.estimated_preactivation_std(
                    BdhProjectionRole::Encoder,
                    fan_in,
                    fan_out,
                    residual_depth,
                );
                let y_sigma = self.estimated_preactivation_std(
                    BdhProjectionRole::EncoderValue,
                    fan_in,
                    fan_out,
                    residual_depth,
                );
                BdhActivationThresholds {
                    x: (x_sigma * inverse_normal_cdf(1.0 - self.config.firing_targets.x_target))
                        as f32,
                    y: (y_sigma * inverse_normal_cdf(1.0 - self.config.firing_targets.y_target))
                        as f32,
                }
            }
        }
    }

    fn embedding_std(&self, width: usize) -> f64 {
        match self.config.kind {
            BdhInitializationKind::NearCritical | BdhInitializationKind::HeadwiseSemiOrthogonal => {
                near_critical_embedding_std(width)
            }
            BdhInitializationKind::SimpleNormal => self.config.simple_normal_std,
            BdhInitializationKind::HeGlorot => glorot_std(width.max(1), width.max(1)),
        }
    }

    fn projection_std(
        &self,
        role: BdhProjectionRole,
        fan_in: usize,
        fan_out: usize,
        residual_depth: usize,
    ) -> f64 {
        self.base_projection_std(role, fan_in, fan_out)
            * self.residual_scaling_factor(role, residual_depth)
    }

    fn base_projection_std(&self, role: BdhProjectionRole, fan_in: usize, fan_out: usize) -> f64 {
        match self.config.kind {
            BdhInitializationKind::NearCritical | BdhInitializationKind::HeadwiseSemiOrthogonal => {
                near_critical_projection_std(fan_in, fan_out)
            }
            BdhInitializationKind::SimpleNormal => self.config.simple_normal_std,
            BdhInitializationKind::HeGlorot => match role {
                BdhProjectionRole::Encoder | BdhProjectionRole::EncoderValue => he_std(fan_in),
                BdhProjectionRole::Decoder | BdhProjectionRole::LmHead => {
                    glorot_std(fan_in, fan_out)
                }
            },
        }
    }

    fn residual_scaling_factor(&self, role: BdhProjectionRole, residual_depth: usize) -> f64 {
        if !role.is_residual_branch() {
            return 1.0;
        }

        let depth_factor = match self.config.residual_scaling.kind {
            BdhResidualScalingKind::FamilyDefault => {
                if matches!(
                    self.config.kind,
                    BdhInitializationKind::NearCritical
                        | BdhInitializationKind::HeadwiseSemiOrthogonal
                ) {
                    1.0 / (residual_depth.max(1) as f64).sqrt()
                } else {
                    1.0
                }
            }
            BdhResidualScalingKind::Disabled => 1.0,
            BdhResidualScalingKind::DepthScaled => 1.0 / (residual_depth.max(1) as f64).sqrt(),
        };

        depth_factor * self.config.residual_scaling.gain
    }

    fn estimated_preactivation_std(
        &self,
        role: BdhProjectionRole,
        fan_in: usize,
        fan_out: usize,
        residual_depth: usize,
    ) -> f64 {
        (fan_in.max(1) as f64).sqrt() * self.projection_std(role, fan_in, fan_out, residual_depth)
    }

    fn apply_headwise_neuron_gains<B: Backend>(
        &self,
        role: BdhProjectionRole,
        tensor: Tensor<B, 3>,
        heads: usize,
        fan_out: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        if !role.supports_neuron_gain_prior() {
            return tensor;
        }

        let gains = self
            .sample_neuron_gains_2d::<B>(heads, fan_out, device)
            .reshape([heads, 1, fan_out]);
        tensor * gains
    }

    fn apply_projection_neuron_gains<B: Backend>(
        &self,
        role: BdhProjectionRole,
        tensor: Tensor<B, 2>,
        fan_in: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        if !matches!(role, BdhProjectionRole::Decoder) {
            return tensor;
        }

        let gains = self
            .sample_neuron_gains_1d::<B>(fan_in, device)
            .reshape([fan_in, 1]);
        tensor * gains
    }

    fn apply_headwise_topology_prior<B: Backend>(
        &self,
        role: BdhProjectionRole,
        tensor: Tensor<B, 3>,
        heads: usize,
        fan_in: usize,
        fan_out: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        if !role.supports_topology_prior() {
            return tensor;
        }
        let Some(values_2d) =
            self.topology_prior_values(fan_in, fan_out, BdhTopologyLatentAxis::Cols)
        else {
            return tensor;
        };
        let mut values = Vec::with_capacity(heads * values_2d.len());
        for _ in 0..heads {
            values.extend_from_slice(&values_2d);
        }
        let prior =
            Tensor::<B, 3>::from_data(TensorData::new(values, [heads, fan_in, fan_out]), device);
        tensor * prior
    }

    fn apply_projection_topology_prior<B: Backend>(
        &self,
        role: BdhProjectionRole,
        tensor: Tensor<B, 2>,
        fan_in: usize,
        fan_out: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        if !role.supports_topology_prior() {
            return tensor;
        }
        let Some(values) = self.topology_prior_values(fan_in, fan_out, BdhTopologyLatentAxis::Rows)
        else {
            return tensor;
        };
        let prior = Tensor::<B, 2>::from_data(TensorData::new(values, [fan_in, fan_out]), device);
        tensor * prior
    }

    fn topology_prior_values(
        &self,
        rows: usize,
        cols: usize,
        latent_axis: BdhTopologyLatentAxis,
    ) -> Option<Vec<f32>> {
        match self.config.topology_prior.kind {
            BdhTopologyPriorKind::Iid => None,
            BdhTopologyPriorKind::ModularBridges => Some(make_modular_bridge_values(
                rows,
                cols,
                latent_axis,
                &self.config.topology_prior,
            )),
        }
    }

    fn sample_neuron_gains_1d<B: Backend>(&self, count: usize, device: &B::Device) -> Tensor<B, 1> {
        match self.config.neuron_gains.kind {
            BdhNeuronGainKind::Iid => Tensor::<B, 1>::ones([count], device),
            BdhNeuronGainKind::HeavyTailedLogNormal => {
                let gains = Tensor::<B, 1>::random(
                    [count],
                    TensorDistribution::Normal(0.0, self.config.neuron_gains.log_sigma),
                    device,
                )
                .exp()
                .clamp_max(self.config.neuron_gains.max_gain);
                let rms = gains
                    .clone()
                    .powf_scalar(2.0)
                    .mean()
                    .sqrt()
                    .clamp_min(1.0e-6);
                gains.div(rms)
            }
        }
    }

    fn sample_neuron_gains_2d<B: Backend>(
        &self,
        rows: usize,
        cols: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        match self.config.neuron_gains.kind {
            BdhNeuronGainKind::Iid => Tensor::<B, 2>::ones([rows, cols], device),
            BdhNeuronGainKind::HeavyTailedLogNormal => {
                let gains = Tensor::<B, 2>::random(
                    [rows, cols],
                    TensorDistribution::Normal(0.0, self.config.neuron_gains.log_sigma),
                    device,
                )
                .exp()
                .clamp_max(self.config.neuron_gains.max_gain);
                let rms = gains
                    .clone()
                    .powf_scalar(2.0)
                    .mean()
                    .sqrt()
                    .clamp_min(1.0e-6);
                gains.div(rms.reshape([1, 1]))
            }
        }
    }
}

fn default_simple_normal_std() -> f64 {
    0.02
}

fn default_residual_scaling_gain() -> f64 {
    1.0
}

fn default_neuron_gain_log_sigma() -> f64 {
    0.75
}

fn default_neuron_gain_max() -> f64 {
    4.0
}

fn default_topology_community_count() -> usize {
    4
}

fn default_topology_bridge_fraction() -> f64 {
    0.05
}

fn default_topology_intra_community_gain() -> f64 {
    1.5
}

fn default_topology_inter_community_gain() -> f64 {
    0.5
}

fn default_topology_bridge_gain() -> f64 {
    1.0
}

fn default_x_firing_target() -> f64 {
    0.15
}

fn default_y_firing_target() -> f64 {
    0.05
}

pub fn near_critical_embedding_std(width: usize) -> f64 {
    (1.0 / (width.max(1) as f64).sqrt()).min(CONTROLLED_INIT_STD_CAP)
}

pub fn near_critical_projection_std(fan_in: usize, fan_out: usize) -> f64 {
    (1.0 / ((fan_in.max(1) + fan_out.max(1)) as f64).sqrt()).min(CONTROLLED_INIT_STD_CAP)
}

pub fn near_critical_residual_output_std(
    fan_in: usize,
    fan_out: usize,
    residual_depth: usize,
) -> f64 {
    let base = 1.0 / ((fan_in.max(1) + fan_out.max(1)) as f64).sqrt();
    (base / (residual_depth.max(1) as f64).sqrt()).min(CONTROLLED_INIT_STD_CAP)
}

pub fn near_critical_embedding_initializer(width: usize) -> Initializer {
    Initializer::Normal {
        mean: 0.0,
        std: near_critical_embedding_std(width),
    }
}

fn he_std(fan_in: usize) -> f64 {
    (2.0 / fan_in.max(1) as f64).sqrt()
}

fn glorot_std(fan_in: usize, fan_out: usize) -> f64 {
    (2.0 / (fan_in.max(1) + fan_out.max(1)) as f64).sqrt()
}

fn validate_probability(value: f64, field: &str) -> Result<(), String> {
    if !value.is_finite() || !(0.0..1.0).contains(&value) {
        return Err(format!(
            "{field} must be finite and in (0, 1) (got {value})"
        ));
    }
    Ok(())
}

fn validate_probability_inclusive(value: f64, field: &str) -> Result<(), String> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(format!(
            "{field} must be finite and in [0, 1] (got {value})"
        ));
    }
    Ok(())
}

fn validate_finite(value: f64, field: &str) -> Result<(), String> {
    if !value.is_finite() {
        return Err(format!("{field} must be finite (got {value})"));
    }
    Ok(())
}

fn validate_positive_finite(value: f64, field: &str) -> Result<(), String> {
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{field} must be finite and > 0 (got {value})"));
    }
    Ok(())
}

fn inverse_normal_cdf(p: f64) -> f64 {
    let p = p.clamp(1.0e-12, 1.0 - 1.0e-12);
    let a: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    let b: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    let c: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    let d: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;

    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if p > phigh {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }

    let q = p - 0.5;
    let r = q * q;
    (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
}

fn make_semi_orthogonal_values<B: Backend>(
    rows: usize,
    cols: usize,
    target_std: f64,
    device: &B::Device,
) -> Vec<f32> {
    let block = rows.min(cols).max(1);
    let gain = target_std * (block as f64).sqrt();
    let gain = gain as f32;
    let mut values = vec![0.0f32; rows * cols];

    if rows <= cols {
        let mut col_start = 0usize;
        while col_start < cols {
            let block_basis = sample_orthonormal_block::<B>(block, device);
            let cols_this = (cols - col_start).min(block);
            for row in 0..rows {
                for col_offset in 0..cols_this {
                    values[row * cols + col_start + col_offset] =
                        block_basis[row * block + col_offset] * gain;
                }
            }
            col_start += block;
        }
    } else {
        let mut row_start = 0usize;
        while row_start < rows {
            let block_basis = sample_orthonormal_block::<B>(block, device);
            let rows_this = (rows - row_start).min(block);
            for row_offset in 0..rows_this {
                for col in 0..cols {
                    values[(row_start + row_offset) * cols + col] =
                        block_basis[row_offset * block + col] * gain;
                }
            }
            row_start += block;
        }
    }

    values
}

fn sample_orthonormal_block<B: Backend>(size: usize, device: &B::Device) -> Vec<f32> {
    let _ = device;
    let mut rng = rand::thread_rng();
    let mut samples = Vec::with_capacity(size * size);
    while samples.len() < size * size {
        let u1 = rng
            .r#gen::<f64>()
            .clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        let u2 = rng.r#gen::<f64>();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        samples.push((radius * theta.cos()) as f32);
        if samples.len() < size * size {
            samples.push((radius * theta.sin()) as f32);
        }
    }
    orthonormalize_columns(&samples, size)
}

fn orthonormalize_columns(samples: &[f32], size: usize) -> Vec<f32> {
    let mut basis = vec![0.0f64; size * size];
    let mut column = vec![0.0f64; size];
    let eps = 1.0e-8;

    for col in 0..size {
        for row in 0..size {
            column[row] = samples[row * size + col] as f64;
        }

        for prev in 0..col {
            let mut dot = 0.0f64;
            for row in 0..size {
                dot += column[row] * basis[row * size + prev];
            }
            for row in 0..size {
                column[row] -= dot * basis[row * size + prev];
            }
        }

        let mut norm = column.iter().map(|value| value * value).sum::<f64>().sqrt();
        if norm < eps {
            column.fill(0.0);
            column[col] = 1.0;
            for prev in 0..col {
                let mut dot = 0.0f64;
                for row in 0..size {
                    dot += column[row] * basis[row * size + prev];
                }
                for row in 0..size {
                    column[row] -= dot * basis[row * size + prev];
                }
            }
            norm = column.iter().map(|value| value * value).sum::<f64>().sqrt();
            if norm < eps {
                norm = 1.0;
            }
        }

        for row in 0..size {
            basis[row * size + col] = column[row] / norm;
        }
    }

    basis.into_iter().map(|value| value as f32).collect()
}

fn make_modular_bridge_values(
    rows: usize,
    cols: usize,
    latent_axis: BdhTopologyLatentAxis,
    config: &BdhTopologyPriorConfig,
) -> Vec<f32> {
    let rows = rows.max(1);
    let cols = cols.max(1);
    let latent_size = match latent_axis {
        BdhTopologyLatentAxis::Rows => rows,
        BdhTopologyLatentAxis::Cols => cols,
    };
    let bridge_count =
        ((latent_size as f64 * config.bridge_fraction).round() as usize).min(latent_size);
    let community_count = config.community_count.max(1).min(rows.min(cols).max(1));
    let mut values = vec![0.0f32; rows * cols];
    let mut square_sum = 0.0f64;

    for row in 0..rows {
        let row_community = (row * community_count) / rows;
        for col in 0..cols {
            let col_community = (col * community_count) / cols;
            let latent_index = match latent_axis {
                BdhTopologyLatentAxis::Rows => row,
                BdhTopologyLatentAxis::Cols => col,
            };
            let is_bridge = latent_index >= latent_size.saturating_sub(bridge_count);
            let value = if is_bridge {
                config.bridge_gain
            } else if row_community == col_community {
                config.intra_community_gain
            } else {
                config.inter_community_gain
            };
            let idx = row * cols + col;
            values[idx] = value as f32;
            square_sum += value * value;
        }
    }

    let rms = (square_sum / (rows * cols) as f64).sqrt().max(1.0e-12);
    let inv_rms = (1.0 / rms) as f32;
    values.iter_mut().for_each(|value| *value *= inv_rms);
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn embedding_std_caps_small_models_and_scales_large_ones() {
        assert!((near_critical_embedding_std(256) - 0.02).abs() < 1e-12);
        assert!((near_critical_embedding_std(4096) - 0.015625).abs() < 1e-12);
    }

    #[test]
    fn projection_std_caps_small_models_and_scales_large_ones() {
        assert!((near_critical_projection_std(64, 64) - 0.02).abs() < 1e-12);
        assert!((near_critical_projection_std(2048, 2048) - 0.015625).abs() < 1e-12);
    }

    #[test]
    fn residual_output_std_shrinks_with_depth() {
        let shallow = near_critical_residual_output_std(2048, 2048, 1);
        let deep = near_critical_residual_output_std(2048, 2048, 16);
        assert!((deep * 4.0 - shallow).abs() < 1e-12);
    }

    #[test]
    fn initialization_config_rejects_non_positive_simple_normal_std() {
        let config = BdhInitializationConfig {
            kind: BdhInitializationKind::SimpleNormal,
            simple_normal_std: 0.0,
            ..Default::default()
        };
        assert!(
            config
                .validate()
                .expect_err("expected invalid std")
                .contains("simple_normal_std")
        );
    }

    #[test]
    fn initialization_config_rejects_non_positive_residual_scaling_gain() {
        let config = BdhInitializationConfig {
            residual_scaling: BdhResidualScalingConfig {
                kind: BdhResidualScalingKind::DepthScaled,
                gain: 0.0,
            },
            ..Default::default()
        };
        assert!(
            config
                .validate()
                .expect_err("expected invalid residual scaling gain")
                .contains("residual_scaling.gain")
        );
    }

    #[test]
    fn initialization_config_rejects_invalid_gaussian_firing_target_probability() {
        let config = BdhInitializationConfig {
            firing_targets: BdhFiringTargetConfig {
                kind: BdhFiringTargetKind::GaussianEstimate,
                x_target: 1.0,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(
            config
                .validate()
                .expect_err("expected invalid firing target probability")
                .contains("firing_targets.x_target")
        );
    }

    #[test]
    fn he_glorot_initializer_uses_role_specific_scaling() {
        let config = BdhInitializationConfig {
            kind: BdhInitializationKind::HeGlorot,
            ..Default::default()
        };
        let initializer = BdhInitializer::new(&config);
        let encoder_std = initializer.projection_std(BdhProjectionRole::Encoder, 256, 8192, 8);
        let decoder_std = initializer.projection_std(BdhProjectionRole::Decoder, 8192, 256, 8);
        assert!(encoder_std > decoder_std);
    }

    #[test]
    fn he_glorot_family_default_residual_scaling_is_depth_independent() {
        let config = BdhInitializationConfig {
            kind: BdhInitializationKind::HeGlorot,
            residual_scaling: BdhResidualScalingConfig {
                kind: BdhResidualScalingKind::FamilyDefault,
                ..Default::default()
            },
            ..Default::default()
        };
        let initializer = BdhInitializer::new(&config);
        let shallow = initializer.projection_std(BdhProjectionRole::Encoder, 256, 8192, 1);
        let deep = initializer.projection_std(BdhProjectionRole::Encoder, 256, 8192, 16);
        assert!((shallow - deep).abs() < 1e-12);
    }

    #[test]
    fn explicit_depth_scaled_residual_scaling_shrinks_non_critical_families() {
        let config = BdhInitializationConfig {
            kind: BdhInitializationKind::HeGlorot,
            residual_scaling: BdhResidualScalingConfig {
                kind: BdhResidualScalingKind::DepthScaled,
                gain: 1.0,
            },
            ..Default::default()
        };
        let initializer = BdhInitializer::new(&config);
        let shallow = initializer.projection_std(BdhProjectionRole::Encoder, 256, 8192, 1);
        let deep = initializer.projection_std(BdhProjectionRole::Encoder, 256, 8192, 16);
        assert!((deep * 4.0 - shallow).abs() < 1e-12);
    }

    #[test]
    fn gaussian_estimate_firing_targets_produce_ordered_branch_thresholds() {
        let config = BdhInitializationConfig {
            firing_targets: BdhFiringTargetConfig {
                kind: BdhFiringTargetKind::GaussianEstimate,
                x_target: 0.15,
                y_target: 0.05,
                ..Default::default()
            },
            ..Default::default()
        };
        let initializer = BdhInitializer::new(&config);
        let thresholds = initializer.activation_thresholds(256, 8192, 8);
        assert!(thresholds.x.is_finite() && thresholds.x > 0.0);
        assert!(thresholds.y.is_finite() && thresholds.y > thresholds.x);
    }

    #[test]
    fn explicit_firing_thresholds_are_forwarded_verbatim() {
        let config = BdhInitializationConfig {
            firing_targets: BdhFiringTargetConfig {
                kind: BdhFiringTargetKind::ExplicitThresholds,
                x_threshold: 0.25,
                y_threshold: 0.75,
                ..Default::default()
            },
            ..Default::default()
        };
        let initializer = BdhInitializer::new(&config);
        let thresholds = initializer.activation_thresholds(256, 8192, 8);
        assert!((thresholds.x - 0.25).abs() < 1.0e-6);
        assert!((thresholds.y - 0.75).abs() < 1.0e-6);
    }

    #[test]
    fn semi_orthogonal_family_is_backend_seeded_and_finite() {
        let device = <TestBackend as Backend>::Device::default();
        TestBackend::seed(&device, 1234);
        let config = BdhInitializationConfig {
            kind: BdhInitializationKind::HeadwiseSemiOrthogonal,
            ..Default::default()
        };
        let initializer = BdhInitializer::new(&config);
        let tensor = initializer.headwise_projection_tensor::<TestBackend>(
            BdhProjectionRole::Encoder,
            2,
            8,
            24,
            8,
            &device,
        );
        let values = tensor
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("values");
        assert!(values.iter().all(|value| value.is_finite()));
        let max_abs = values.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        assert!(max_abs > 0.0);
    }

    #[test]
    fn heavy_tailed_neuron_gains_increase_column_norm_dispersion() {
        let device = <TestBackend as Backend>::Device::default();
        let iid = BdhInitializationConfig {
            kind: BdhInitializationKind::SimpleNormal,
            simple_normal_std: 0.02,
            ..Default::default()
        };
        let heavy_tailed = BdhInitializationConfig {
            kind: BdhInitializationKind::SimpleNormal,
            neuron_gains: BdhNeuronGainConfig {
                kind: BdhNeuronGainKind::HeavyTailedLogNormal,
                log_sigma: 1.0,
                max_gain: 6.0,
            },
            simple_normal_std: 0.02,
            ..Default::default()
        };

        TestBackend::seed(&device, 1234);
        let iid_tensor = BdhInitializer::new(&iid).headwise_projection_tensor::<TestBackend>(
            BdhProjectionRole::Encoder,
            4,
            64,
            128,
            8,
            &device,
        );
        TestBackend::seed(&device, 1234);
        let heavy_tensor = BdhInitializer::new(&heavy_tailed)
            .headwise_projection_tensor::<TestBackend>(
                BdhProjectionRole::Encoder,
                4,
                64,
                128,
                8,
                &device,
            );

        let iid_values = iid_tensor
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("iid values");
        let heavy_values = heavy_tensor
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("heavy-tailed values");

        let iid_cv = headwise_column_norm_cv(&iid_values, 4, 64, 128);
        let heavy_cv = headwise_column_norm_cv(&heavy_values, 4, 64, 128);
        assert!(heavy_cv > iid_cv);
    }

    #[test]
    fn default_initialization_config_matches_promoted_candidate() {
        let config = BdhInitializationConfig::default();
        assert_eq!(config.kind, BdhInitializationKind::SimpleNormal);
        assert_eq!(
            config.residual_scaling.kind,
            BdhResidualScalingKind::DepthScaled
        );
        assert_eq!(
            config.neuron_gains.kind,
            BdhNeuronGainKind::HeavyTailedLogNormal
        );
        assert_eq!(config.topology_prior.kind, BdhTopologyPriorKind::Iid);
        assert_eq!(config.firing_targets.kind, BdhFiringTargetKind::Disabled);
        assert!((config.simple_normal_std - 0.02).abs() < 1.0e-12);
    }

    #[test]
    fn initialization_config_rejects_zero_topology_community_count() {
        let config = BdhInitializationConfig {
            topology_prior: BdhTopologyPriorConfig {
                kind: BdhTopologyPriorKind::ModularBridges,
                community_count: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(
            config
                .validate()
                .expect_err("expected invalid topology prior")
                .contains("topology_prior.community_count")
        );
    }

    #[test]
    fn modular_bridges_prior_biases_intra_community_weights() {
        let config = BdhInitializationConfig {
            topology_prior: BdhTopologyPriorConfig {
                kind: BdhTopologyPriorKind::ModularBridges,
                community_count: 4,
                bridge_fraction: 0.125,
                intra_community_gain: 1.5,
                inter_community_gain: 0.5,
                bridge_gain: 1.0,
            },
            ..Default::default()
        };
        let initializer = BdhInitializer::new(&config);
        let values = initializer
            .topology_prior_values(8, 16, BdhTopologyLatentAxis::Cols)
            .expect("modular prior values");

        let (same_mean, cross_mean) = community_means(&values, 8, 16, 4, 2, false);
        assert!(same_mean > cross_mean);
        assert!(bridge_columns_are_uniform(&values, 8, 16, 2));
        assert!((matrix_rms(&values) - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn modular_bridges_decoder_prior_biases_matching_row_blocks() {
        let config = BdhInitializationConfig {
            topology_prior: BdhTopologyPriorConfig {
                kind: BdhTopologyPriorKind::ModularBridges,
                community_count: 4,
                bridge_fraction: 0.25,
                intra_community_gain: 1.5,
                inter_community_gain: 0.5,
                bridge_gain: 1.0,
            },
            ..Default::default()
        };
        let initializer = BdhInitializer::new(&config);
        let values = initializer
            .topology_prior_values(16, 8, BdhTopologyLatentAxis::Rows)
            .expect("decoder prior values");

        let (same_mean, cross_mean) = community_means(&values, 16, 8, 4, 4, true);
        assert!(same_mean > cross_mean);
        assert!(bridge_rows_are_uniform(&values, 16, 8, 4));
        assert!((matrix_rms(&values) - 1.0).abs() < 1.0e-5);
    }

    fn headwise_column_norm_cv(values: &[f32], heads: usize, fan_in: usize, fan_out: usize) -> f32 {
        let mut norms = Vec::with_capacity(heads * fan_out);
        for head in 0..heads {
            for col in 0..fan_out {
                let mut sum_sq = 0.0f32;
                for row in 0..fan_in {
                    let idx = head * fan_in * fan_out + row * fan_out + col;
                    let value = values[idx];
                    sum_sq += value * value;
                }
                norms.push(sum_sq.sqrt());
            }
        }

        let mean = norms.iter().copied().sum::<f32>() / norms.len() as f32;
        let variance = norms
            .iter()
            .copied()
            .map(|value| {
                let centered = value - mean;
                centered * centered
            })
            .sum::<f32>()
            / norms.len() as f32;
        variance.sqrt() / mean.max(1.0e-6)
    }

    fn matrix_rms(values: &[f32]) -> f32 {
        (values
            .iter()
            .copied()
            .map(|value| value * value)
            .sum::<f32>()
            / values.len().max(1) as f32)
            .sqrt()
    }

    fn community_means(
        values: &[f32],
        rows: usize,
        cols: usize,
        community_count: usize,
        bridge_count: usize,
        bridge_on_rows: bool,
    ) -> (f32, f32) {
        let mut same_sum = 0.0f32;
        let mut same_count = 0usize;
        let mut cross_sum = 0.0f32;
        let mut cross_count = 0usize;
        for row in 0..rows {
            let row_community = (row * community_count) / rows.max(1);
            for col in 0..cols {
                let latent_index = if bridge_on_rows { row } else { col };
                let latent_size = if bridge_on_rows { rows } else { cols };
                if latent_index >= latent_size.saturating_sub(bridge_count) {
                    continue;
                }
                let col_community = (col * community_count) / cols.max(1);
                let value = values[row * cols + col].abs();
                if row_community == col_community {
                    same_sum += value;
                    same_count += 1;
                } else {
                    cross_sum += value;
                    cross_count += 1;
                }
            }
        }
        (
            same_sum / same_count.max(1) as f32,
            cross_sum / cross_count.max(1) as f32,
        )
    }

    fn bridge_columns_are_uniform(
        values: &[f32],
        rows: usize,
        cols: usize,
        bridge_count: usize,
    ) -> bool {
        for col in cols.saturating_sub(bridge_count)..cols {
            let reference = values[col];
            for row in 1..rows {
                if (values[row * cols + col] - reference).abs() > 1.0e-6 {
                    return false;
                }
            }
        }
        true
    }

    fn bridge_rows_are_uniform(
        values: &[f32],
        rows: usize,
        cols: usize,
        bridge_count: usize,
    ) -> bool {
        for row in rows.saturating_sub(bridge_count)..rows {
            let start = row * cols;
            let reference = values[start];
            for col in 1..cols {
                if (values[start + col] - reference).abs() > 1.0e-6 {
                    return false;
                }
            }
        }
        true
    }
}
