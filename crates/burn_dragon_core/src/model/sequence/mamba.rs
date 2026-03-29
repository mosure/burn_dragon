use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution as TensorDistribution, Tensor, TensorData, activation};
use rand::Rng;
use serde::{Deserialize, Serialize};

use super::config::SequenceMemorySystem;

#[allow(dead_code)]
pub const MAMBA1_UPSTREAM_REPO: &str = "https://github.com/state-spaces/mamba";
#[allow(dead_code)]
pub const MAMBA1_UPSTREAM_COMMIT: &str = "c5afbdf";
#[allow(dead_code)]
pub const MAMBA2_UPSTREAM_REPO: &str = "https://github.com/state-spaces/mamba";

fn default_mamba_d_state() -> usize {
    16
}

fn default_mamba_d_conv() -> usize {
    4
}

fn default_mamba_expand() -> usize {
    2
}

fn default_mamba_dt_min() -> f32 {
    1.0e-3
}

fn default_mamba_dt_max() -> f32 {
    1.0e-1
}

fn default_mamba_dt_scale() -> f32 {
    1.0
}

fn default_mamba_headdim() -> usize {
    128
}

fn default_mamba_ngroups() -> usize {
    1
}

fn default_mamba_a_init_min() -> f32 {
    1.0
}

fn default_mamba_a_init_max() -> f32 {
    16.0
}

fn default_mamba_norm_eps() -> f32 {
    1.0e-5
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct MambaSequenceConfig {
    #[serde(default = "default_mamba_d_state")]
    pub d_state: usize,
    #[serde(default = "default_mamba_d_conv")]
    pub d_conv: usize,
    #[serde(default = "default_mamba_expand")]
    pub expand: usize,
    #[serde(default)]
    pub dt_rank: Option<usize>,
    #[serde(default = "default_mamba_dt_min")]
    pub dt_min: f32,
    #[serde(default = "default_mamba_dt_max")]
    pub dt_max: f32,
    #[serde(default = "default_mamba_dt_scale")]
    pub dt_scale: f32,
    #[serde(default = "default_true")]
    pub conv_bias: bool,
    #[serde(default = "default_true")]
    pub use_fast_path: bool,
    #[serde(default = "default_mamba_headdim")]
    pub headdim: usize,
    #[serde(default = "default_mamba_ngroups")]
    pub ngroups: usize,
    #[serde(default = "default_mamba_a_init_min")]
    pub a_init_min: f32,
    #[serde(default = "default_mamba_a_init_max")]
    pub a_init_max: f32,
    #[serde(default = "default_mamba_norm_eps")]
    pub norm_eps: f32,
}

impl Default for MambaSequenceConfig {
    fn default() -> Self {
        Self {
            d_state: default_mamba_d_state(),
            d_conv: default_mamba_d_conv(),
            expand: default_mamba_expand(),
            dt_rank: None,
            dt_min: default_mamba_dt_min(),
            dt_max: default_mamba_dt_max(),
            dt_scale: default_mamba_dt_scale(),
            conv_bias: default_true(),
            use_fast_path: default_true(),
            headdim: default_mamba_headdim(),
            ngroups: default_mamba_ngroups(),
            a_init_min: default_mamba_a_init_min(),
            a_init_max: default_mamba_a_init_max(),
            norm_eps: default_mamba_norm_eps(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedMambaSequenceConfig {
    pub d_model: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub dt_rank: usize,
    pub dt_min: f32,
    pub dt_max: f32,
    pub dt_scale: f32,
    pub conv_bias: bool,
    pub use_fast_path: bool,
    pub headdim: usize,
    pub ngroups: usize,
    pub nheads: usize,
    pub a_init_min: f32,
    pub a_init_max: f32,
    pub norm_eps: f32,
}

impl ResolvedMambaSequenceConfig {
    pub fn mamba2_conv_dim(self) -> usize {
        self.d_inner + 2 * self.ngroups * self.d_state
    }

    pub fn mamba2_in_proj_dim(self) -> usize {
        2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
    }
}

impl MambaSequenceConfig {
    pub fn validate(
        &self,
        memory_system: SequenceMemorySystem,
        d_model: usize,
    ) -> Result<(), String> {
        if self.d_state == 0 {
            return Err("d_state must be positive".to_string());
        }
        if self.d_conv == 0 {
            return Err("d_conv must be positive".to_string());
        }
        if self.expand == 0 {
            return Err("expand must be positive".to_string());
        }
        if self.dt_min <= 0.0 || !self.dt_min.is_finite() {
            return Err("dt_min must be finite and positive".to_string());
        }
        if self.dt_max < self.dt_min || !self.dt_max.is_finite() {
            return Err("dt_max must be finite and >= dt_min".to_string());
        }
        if self.dt_scale <= 0.0 || !self.dt_scale.is_finite() {
            return Err("dt_scale must be finite and positive".to_string());
        }
        let d_inner = d_model.max(1) * self.expand.max(1);
        if matches!(memory_system, SequenceMemorySystem::Mamba2StateSpaceDuality) {
            if self.headdim == 0 {
                return Err("headdim must be positive for mamba2_state_space_duality".to_string());
            }
            if d_inner % self.headdim != 0 {
                return Err(format!(
                    "mamba2_state_space_duality requires d_inner divisible by headdim (got d_inner={d_inner} headdim={})",
                    self.headdim
                ));
            }
            let nheads = d_inner / self.headdim;
            if self.ngroups == 0 {
                return Err("ngroups must be positive for mamba2_state_space_duality".to_string());
            }
            if nheads % self.ngroups != 0 {
                return Err(format!(
                    "mamba2_state_space_duality requires nheads divisible by ngroups (got nheads={nheads} ngroups={})",
                    self.ngroups
                ));
            }
            if self.a_init_min <= 0.0
                || self.a_init_max < self.a_init_min
                || !self.a_init_min.is_finite()
                || !self.a_init_max.is_finite()
            {
                return Err("a_init range must be finite, positive, and ordered".to_string());
            }
            if self.norm_eps <= 0.0 || !self.norm_eps.is_finite() {
                return Err("norm_eps must be finite and positive".to_string());
            }
        }
        Ok(())
    }

    pub fn resolve(
        &self,
        d_model: usize,
        memory_system: SequenceMemorySystem,
    ) -> ResolvedMambaSequenceConfig {
        self.validate(memory_system, d_model)
            .unwrap_or_else(|message| panic!("{message}"));
        let d_model = d_model.max(1);
        let d_state = self.d_state.max(1);
        let d_conv = self.d_conv.max(1);
        let expand = self.expand.max(1);
        let d_inner = d_model * expand;
        let dt_rank = self.dt_rank.unwrap_or_else(|| d_model.div_ceil(16)).max(1);
        let headdim = self.headdim.max(1);
        let nheads = if matches!(memory_system, SequenceMemorySystem::Mamba2StateSpaceDuality) {
            d_inner / headdim
        } else {
            0
        };
        ResolvedMambaSequenceConfig {
            d_model,
            d_inner,
            d_state,
            d_conv,
            dt_rank,
            dt_min: self.dt_min.max(1.0e-6),
            dt_max: self.dt_max.max(self.dt_min.max(1.0e-6)),
            dt_scale: self.dt_scale.max(1.0e-6),
            conv_bias: self.conv_bias,
            use_fast_path: self.use_fast_path,
            headdim,
            ngroups: self.ngroups.max(1),
            nheads,
            a_init_min: self.a_init_min.max(1.0e-6),
            a_init_max: self.a_init_max.max(self.a_init_min.max(1.0e-6)),
            norm_eps: self.norm_eps.max(1.0e-8),
        }
    }
}

#[derive(Module, Debug)]
pub struct Mamba1SequenceParameters<B: Backend> {
    d_model: usize,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    dt_rank: usize,
    in_proj: Param<Tensor<B, 2>>,
    conv_weight: Param<Tensor<B, 2>>,
    conv_bias: Option<Param<Tensor<B, 1>>>,
    x_proj: Param<Tensor<B, 2>>,
    dt_proj_weight: Param<Tensor<B, 2>>,
    dt_proj_bias: Param<Tensor<B, 1>>,
    a_log: Param<Tensor<B, 2>>,
    d_skip: Param<Tensor<B, 1>>,
    out_proj: Param<Tensor<B, 2>>,
}

impl<B: Backend> Mamba1SequenceParameters<B> {
    pub fn new(config: ResolvedMambaSequenceConfig, device: &B::Device) -> Self {
        let in_std = (1.0 / config.d_model.max(1) as f32).sqrt();
        let out_std = (1.0 / config.d_inner.max(1) as f32).sqrt();
        let conv_std = (1.0 / config.d_conv.max(1) as f32).sqrt();
        let dt_weight_std = (1.0 / config.dt_rank.max(1) as f32).sqrt() * config.dt_scale;
        let dt_target = (config.dt_min * config.dt_max).sqrt().max(1.0e-6);
        let dt_bias = dt_target + (-(-dt_target).exp_m1()).ln();

        let in_proj = Param::from_tensor(Tensor::<B, 2>::random(
            [config.d_model, config.d_inner * 2],
            TensorDistribution::Normal(0.0, in_std as f64),
            device,
        ));
        let conv_weight = Param::from_tensor(Tensor::<B, 2>::random(
            [config.d_inner, config.d_conv],
            TensorDistribution::Normal(0.0, conv_std as f64),
            device,
        ));
        let conv_bias = config
            .conv_bias
            .then(|| Param::from_tensor(Tensor::<B, 1>::zeros([config.d_inner], device)));
        let x_proj = Param::from_tensor(Tensor::<B, 2>::random(
            [config.d_inner, config.dt_rank + config.d_state * 2],
            TensorDistribution::Normal(0.0, out_std as f64),
            device,
        ));
        let dt_proj_weight = Param::from_tensor(Tensor::<B, 2>::random(
            [config.dt_rank, config.d_inner],
            TensorDistribution::Normal(0.0, dt_weight_std as f64),
            device,
        ));
        let dt_proj_bias = Param::from_tensor(Tensor::<B, 1>::from_data(
            TensorData::new(vec![dt_bias; config.d_inner], [config.d_inner]),
            device,
        ));
        let a_values = (0..config.d_inner)
            .flat_map(|_| (1..=config.d_state).map(|value| (value as f32).ln()))
            .collect::<Vec<_>>();
        let a_log = Param::from_tensor(Tensor::<B, 2>::from_data(
            TensorData::new(a_values, [config.d_inner, config.d_state]),
            device,
        ));
        let d_skip = Param::from_tensor(Tensor::<B, 1>::ones([config.d_inner], device));
        let out_proj = Param::from_tensor(Tensor::<B, 2>::random(
            [config.d_inner, config.d_model],
            TensorDistribution::Normal(0.0, out_std as f64),
            device,
        ));

        Self {
            d_model: config.d_model,
            d_inner: config.d_inner,
            d_state: config.d_state,
            d_conv: config.d_conv,
            dt_rank: config.dt_rank,
            in_proj,
            conv_weight,
            conv_bias,
            x_proj,
            dt_proj_weight,
            dt_proj_bias,
            a_log,
            d_skip,
            out_proj,
        }
    }

    pub fn config(&self) -> ResolvedMambaSequenceConfig {
        ResolvedMambaSequenceConfig {
            d_model: self.d_model,
            d_inner: self.d_inner,
            d_state: self.d_state,
            d_conv: self.d_conv,
            dt_rank: self.dt_rank,
            dt_min: default_mamba_dt_min(),
            dt_max: default_mamba_dt_max(),
            dt_scale: default_mamba_dt_scale(),
            conv_bias: self.conv_bias.is_some(),
            use_fast_path: false,
            headdim: default_mamba_headdim(),
            ngroups: default_mamba_ngroups(),
            nheads: 0,
            a_init_min: default_mamba_a_init_min(),
            a_init_max: default_mamba_a_init_max(),
            norm_eps: default_mamba_norm_eps(),
        }
    }

    pub fn in_proj_tensor(&self) -> Tensor<B, 2> {
        self.in_proj.val()
    }

    pub fn conv_weight_tensor(&self) -> Tensor<B, 2> {
        self.conv_weight.val()
    }

    pub fn conv_bias_tensor(&self) -> Option<Tensor<B, 1>> {
        self.conv_bias.as_ref().map(|bias| bias.val())
    }

    pub fn x_proj_tensor(&self) -> Tensor<B, 2> {
        self.x_proj.val()
    }

    pub fn dt_proj_weight_tensor(&self) -> Tensor<B, 2> {
        self.dt_proj_weight.val()
    }

    pub fn dt_proj_bias_tensor(&self) -> Tensor<B, 1> {
        self.dt_proj_bias.val()
    }

    pub fn a_log_tensor(&self) -> Tensor<B, 2> {
        self.a_log.val()
    }

    pub fn d_skip_tensor(&self) -> Tensor<B, 1> {
        self.d_skip.val()
    }

    pub fn out_proj_tensor(&self) -> Tensor<B, 2> {
        self.out_proj.val()
    }
}

#[derive(Module, Debug)]
pub struct Mamba2SequenceParameters<B: Backend> {
    d_model: usize,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    headdim: usize,
    ngroups: usize,
    nheads: usize,
    norm_eps: f32,
    in_proj: Param<Tensor<B, 2>>,
    conv_weight: Param<Tensor<B, 2>>,
    conv_bias: Option<Param<Tensor<B, 1>>>,
    dt_bias: Param<Tensor<B, 1>>,
    a_log: Param<Tensor<B, 1>>,
    d_skip: Param<Tensor<B, 1>>,
    norm_weight: Param<Tensor<B, 1>>,
    out_proj: Param<Tensor<B, 2>>,
}

impl<B: Backend> Mamba2SequenceParameters<B> {
    pub fn new(config: ResolvedMambaSequenceConfig, device: &B::Device) -> Self {
        let in_std = (1.0 / config.d_model.max(1) as f32).sqrt();
        let out_std = (1.0 / config.d_inner.max(1) as f32).sqrt();
        let conv_std = (1.0 / config.d_conv.max(1) as f32).sqrt();
        let mut rng = rand::thread_rng();
        let log_dt_min = config.dt_min.ln();
        let log_dt_max = config.dt_max.ln();
        let dt_bias = (0..config.nheads)
            .map(|_| {
                let sample = rng.gen_range(log_dt_min..=log_dt_max).exp().max(1.0e-4);
                sample + (-(-sample).exp_m1()).ln()
            })
            .collect::<Vec<_>>();
        let a_log = (0..config.nheads)
            .map(|_| rng.gen_range(config.a_init_min..=config.a_init_max).ln())
            .collect::<Vec<_>>();

        let in_proj = Param::from_tensor(Tensor::<B, 2>::random(
            [config.d_model, config.mamba2_in_proj_dim()],
            TensorDistribution::Normal(0.0, in_std as f64),
            device,
        ));
        let conv_weight = Param::from_tensor(Tensor::<B, 2>::random(
            [config.mamba2_conv_dim(), config.d_conv],
            TensorDistribution::Normal(0.0, conv_std as f64),
            device,
        ));
        let conv_bias = config
            .conv_bias
            .then(|| Param::from_tensor(Tensor::<B, 1>::zeros([config.mamba2_conv_dim()], device)));
        let dt_bias = Param::from_tensor(Tensor::<B, 1>::from_data(
            TensorData::new(dt_bias, [config.nheads]),
            device,
        ));
        let a_log = Param::from_tensor(Tensor::<B, 1>::from_data(
            TensorData::new(a_log, [config.nheads]),
            device,
        ));
        let d_skip = Param::from_tensor(Tensor::<B, 1>::ones([config.nheads], device));
        let norm_weight = Param::from_tensor(Tensor::<B, 1>::ones([config.d_inner], device));
        let out_proj = Param::from_tensor(Tensor::<B, 2>::random(
            [config.d_inner, config.d_model],
            TensorDistribution::Normal(0.0, out_std as f64),
            device,
        ));

        Self {
            d_model: config.d_model,
            d_inner: config.d_inner,
            d_state: config.d_state,
            d_conv: config.d_conv,
            headdim: config.headdim,
            ngroups: config.ngroups,
            nheads: config.nheads,
            norm_eps: config.norm_eps,
            in_proj,
            conv_weight,
            conv_bias,
            dt_bias,
            a_log,
            d_skip,
            norm_weight,
            out_proj,
        }
    }

    pub fn config(&self) -> ResolvedMambaSequenceConfig {
        ResolvedMambaSequenceConfig {
            d_model: self.d_model,
            d_inner: self.d_inner,
            d_state: self.d_state,
            d_conv: self.d_conv,
            dt_rank: self.d_model.div_ceil(16),
            dt_min: default_mamba_dt_min(),
            dt_max: default_mamba_dt_max(),
            dt_scale: default_mamba_dt_scale(),
            conv_bias: self.conv_bias.is_some(),
            use_fast_path: false,
            headdim: self.headdim,
            ngroups: self.ngroups,
            nheads: self.nheads,
            a_init_min: default_mamba_a_init_min(),
            a_init_max: default_mamba_a_init_max(),
            norm_eps: self.norm_eps,
        }
    }

    pub fn in_proj_tensor(&self) -> Tensor<B, 2> {
        self.in_proj.val()
    }

    pub fn conv_weight_tensor(&self) -> Tensor<B, 2> {
        self.conv_weight.val()
    }

    pub fn conv_bias_tensor(&self) -> Option<Tensor<B, 1>> {
        self.conv_bias.as_ref().map(|bias| bias.val())
    }

    pub fn dt_bias_tensor(&self) -> Tensor<B, 1> {
        self.dt_bias.val()
    }

    pub fn a_log_tensor(&self) -> Tensor<B, 1> {
        self.a_log.val()
    }

    pub fn d_skip_tensor(&self) -> Tensor<B, 1> {
        self.d_skip.val()
    }

    pub fn norm_weight_tensor(&self) -> Tensor<B, 1> {
        self.norm_weight.val()
    }

    pub fn out_proj_tensor(&self) -> Tensor<B, 2> {
        self.out_proj.val()
    }
}

#[derive(Module, Debug)]
pub struct MambaSequenceParameters<B: Backend> {
    mamba1: Option<Mamba1SequenceParameters<B>>,
    mamba2: Option<Mamba2SequenceParameters<B>>,
}

impl<B: Backend> MambaSequenceParameters<B> {
    pub fn new(
        config: ResolvedMambaSequenceConfig,
        memory_system: SequenceMemorySystem,
        device: &B::Device,
    ) -> Self {
        match memory_system {
            SequenceMemorySystem::Mamba1SelectiveScan => Self {
                mamba1: Some(Mamba1SequenceParameters::new(config, device)),
                mamba2: None,
            },
            SequenceMemorySystem::Mamba2StateSpaceDuality => Self {
                mamba1: None,
                mamba2: Some(Mamba2SequenceParameters::new(config, device)),
            },
            other => panic!("unsupported memory system {other:?} for mamba params"),
        }
    }

    pub fn mamba1(&self) -> Option<&Mamba1SequenceParameters<B>> {
        self.mamba1.as_ref()
    }

    pub fn mamba2(&self) -> Option<&Mamba2SequenceParameters<B>> {
        self.mamba2.as_ref()
    }
}

#[derive(Debug, Clone)]
pub struct MambaReferenceState<B: Backend> {
    pub conv: Tensor<B, 4>,
    pub ssm: Tensor<B, 4>,
}

fn silu<B: Backend, const D: usize>(values: Tensor<B, D>) -> Tensor<B, D> {
    values.clone() * activation::sigmoid(values)
}

pub(crate) fn mamba_depthwise_conv_step_reference<B: Backend>(
    x_t: Tensor<B, 3>,
    conv_state: Tensor<B, 4>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Option<Tensor<B, 1>>,
) -> (Tensor<B, 3>, Tensor<B, 4>) {
    let [batch, views, channels] = x_t.shape().dims::<3>();
    let d_conv = conv_state.shape().dims::<4>()[3];
    let device = x_t.device();
    let conv_tail = if d_conv > 1 {
        conv_state.clone().slice_dim(3, 1..d_conv)
    } else {
        Tensor::<B, 4>::zeros([batch, views, channels, 0], &device)
    };
    let next_conv_state = Tensor::cat(vec![conv_tail, x_t.clone().unsqueeze_dim::<4>(3)], 3);
    let mut u_t = (next_conv_state.clone() * conv_weight.reshape([1, 1, channels, d_conv]))
        .sum_dim(3)
        .reshape([batch, views, channels]);
    if let Some(bias) = conv_bias {
        u_t = u_t + bias.reshape([1, 1, channels]);
    }
    (silu(u_t), next_conv_state)
}

pub(crate) fn mamba1_selective_scan_step_reference<B: Backend>(
    u_t: Tensor<B, 3>,
    z_t: Tensor<B, 3>,
    ssm_state: Tensor<B, 4>,
    params: &Mamba1SequenceParameters<B>,
) -> (Tensor<B, 3>, Tensor<B, 4>) {
    let [batch, views, d_inner] = u_t.shape().dims::<3>();
    let config = params.config();
    let a = params
        .a_log
        .val()
        .exp()
        .neg()
        .reshape([1, 1, config.d_inner, config.d_state]);
    let d_skip = params.d_skip.val().reshape([1, 1, config.d_inner]);

    let x_db = u_t
        .clone()
        .reshape([batch, d_inner])
        .matmul(params.x_proj.val())
        .reshape([batch, config.dt_rank + config.d_state * 2]);
    let dt = activation::softplus(
        x_db.clone()
            .slice_dim(1, 0..config.dt_rank)
            .matmul(params.dt_proj_weight.val())
            .reshape([batch, views, config.d_inner])
            + params.dt_proj_bias.val().reshape([1, 1, config.d_inner]),
        1.0,
    );
    let b_t = x_db
        .clone()
        .slice_dim(1, config.dt_rank..(config.dt_rank + config.d_state))
        .reshape([batch, views, config.d_state]);
    let c_t = x_db
        .slice_dim(
            1,
            (config.dt_rank + config.d_state)..(config.dt_rank + config.d_state * 2),
        )
        .reshape([batch, views, config.d_state]);

    let d_a = (dt.clone().unsqueeze_dim::<4>(3) * a).exp();
    let d_b = dt.clone().unsqueeze_dim::<4>(3) * b_t.clone().unsqueeze_dim::<4>(2);
    let next_ssm_state = ssm_state * d_a + u_t.clone().unsqueeze_dim::<4>(3) * d_b;
    let y_t = (next_ssm_state.clone() * c_t.unsqueeze_dim::<4>(2))
        .sum_dim(3)
        .reshape([batch, views, config.d_inner])
        + d_skip * u_t;
    (y_t * silu(z_t), next_ssm_state)
}

fn repeat_groups_to_heads<B: Backend>(grouped: Tensor<B, 3>, nheads: usize) -> Tensor<B, 3> {
    let [batch, ngroups, d_state] = grouped.shape().dims::<3>();
    assert_eq!(
        nheads % ngroups,
        0,
        "Mamba-2 requires nheads divisible by ngroups"
    );
    grouped
        .reshape([batch, ngroups, 1, d_state])
        .repeat_dim(2, nheads / ngroups)
        .reshape([batch, nheads, d_state])
}

fn mamba2_rmsnorm_gated_reference<B: Backend>(
    y: Tensor<B, 3>,
    z: Tensor<B, 3>,
    weight: Tensor<B, 1>,
    eps: f32,
) -> Tensor<B, 3> {
    let width = weight.shape().dims::<1>()[0];
    let rms = y
        .clone()
        .powf_scalar(2.0)
        .mean_dim(2)
        .add_scalar(eps)
        .sqrt();
    (y / rms) * weight.reshape([1, 1, width]) * silu(z)
}

pub(crate) fn mamba2_state_space_duality_step_reference<B: Backend>(
    x_t: Tensor<B, 3>,
    dt_t: Tensor<B, 2>,
    b_t: Tensor<B, 3>,
    c_t: Tensor<B, 3>,
    ssm_state: Tensor<B, 4>,
    params: &Mamba2SequenceParameters<B>,
) -> (Tensor<B, 3>, Tensor<B, 4>) {
    let [batch, nheads, headdim] = x_t.shape().dims::<3>();
    let config = params.config();
    assert_eq!(nheads, config.nheads);
    assert_eq!(headdim, config.headdim);

    let a = params
        .a_log
        .val()
        .exp()
        .neg()
        .reshape([1, config.nheads, 1, 1]);
    let d_skip = params.d_skip.val().reshape([1, config.nheads, 1]);
    let dt = activation::softplus(dt_t + params.dt_bias.val().reshape([1, config.nheads]), 1.0);
    let b_heads = repeat_groups_to_heads(b_t, config.nheads);
    let c_heads = repeat_groups_to_heads(c_t, config.nheads);

    let decay = (dt.clone().reshape([batch, config.nheads, 1, 1]) * a).exp();
    let input_term = dt.reshape([batch, config.nheads, 1, 1])
        * b_heads.unsqueeze_dim::<4>(2)
        * x_t.clone().unsqueeze_dim::<4>(3);
    let next_ssm_state = ssm_state * decay + input_term;
    let y_t = (next_ssm_state.clone() * c_heads.unsqueeze_dim::<4>(2))
        .sum_dim(3)
        .reshape([batch, config.nheads, config.headdim])
        + x_t * d_skip;
    (y_t, next_ssm_state)
}

fn mamba1_reference<B: Backend>(
    hidden_states: Tensor<B, 4>,
    params: &Mamba1SequenceParameters<B>,
    state: Option<MambaReferenceState<B>>,
) -> (Tensor<B, 4>, MambaReferenceState<B>) {
    let [batch, views, time, dim] = hidden_states.shape().dims::<4>();
    assert_eq!(
        views, 1,
        "Mamba reference path currently expects a single dense stream view"
    );
    assert_eq!(
        dim, params.d_model,
        "hidden dim {} must match mamba d_model {}",
        dim, params.d_model
    );

    let device = hidden_states.device();
    let config = params.config();

    let mut conv_state = match state.as_ref() {
        Some(existing)
            if existing.conv.shape().dims::<4>() == [batch, 1, config.d_inner, config.d_conv] =>
        {
            existing.conv.clone()
        }
        _ => Tensor::<B, 4>::zeros([batch, 1, config.d_inner, config.d_conv], &device),
    };
    let mut ssm_state = match state.as_ref() {
        Some(existing)
            if existing.ssm.shape().dims::<4>() == [batch, 1, config.d_inner, config.d_state] =>
        {
            existing.ssm.clone()
        }
        _ => Tensor::<B, 4>::zeros([batch, 1, config.d_inner, config.d_state], &device),
    };

    let xz = hidden_states
        .clone()
        .reshape([batch * time, config.d_model])
        .matmul(params.in_proj.val())
        .reshape([batch, time, config.d_inner * 2]);
    let x = xz
        .clone()
        .slice_dim(2, 0..config.d_inner)
        .swap_dims(1, 2)
        .reshape([batch, 1, config.d_inner, time]);
    let z = xz
        .slice_dim(2, config.d_inner..(config.d_inner * 2))
        .swap_dims(1, 2)
        .reshape([batch, 1, config.d_inner, time]);

    let mut outputs = Vec::with_capacity(time);

    for step in 0..time {
        let x_t = x
            .clone()
            .slice_dim(3, step..step + 1)
            .reshape([batch, 1, config.d_inner]);
        let z_t = z
            .clone()
            .slice_dim(3, step..step + 1)
            .reshape([batch, 1, config.d_inner]);
        let (u_t, next_conv_state) = mamba_depthwise_conv_step_reference(
            x_t,
            conv_state,
            params.conv_weight.val(),
            params.conv_bias.as_ref().map(|bias| bias.val()),
        );
        conv_state = next_conv_state;
        let (y_t, next_ssm_state) =
            mamba1_selective_scan_step_reference(u_t, z_t, ssm_state, params);
        ssm_state = next_ssm_state;
        let out_t = y_t
            .reshape([batch, config.d_inner])
            .matmul(params.out_proj.val())
            .reshape([batch, 1, 1, config.d_model]);
        outputs.push(out_t);
    }

    (
        Tensor::cat(outputs, 2),
        MambaReferenceState {
            conv: conv_state,
            ssm: ssm_state,
        },
    )
}

fn mamba2_reference<B: Backend>(
    hidden_states: Tensor<B, 4>,
    params: &Mamba2SequenceParameters<B>,
    state: Option<MambaReferenceState<B>>,
) -> (Tensor<B, 4>, MambaReferenceState<B>) {
    let [batch, views, time, dim] = hidden_states.shape().dims::<4>();
    assert_eq!(
        views, 1,
        "Mamba-2 reference path currently expects a single dense stream view"
    );
    assert_eq!(
        dim, params.d_model,
        "hidden dim {} must match mamba2 d_model {}",
        dim, params.d_model
    );

    let config = params.config();
    let device = hidden_states.device();
    let conv_dim = config.mamba2_conv_dim();

    let mut conv_state = match state.as_ref() {
        Some(existing)
            if existing.conv.shape().dims::<4>() == [batch, 1, conv_dim, config.d_conv] =>
        {
            existing.conv.clone()
        }
        _ => Tensor::<B, 4>::zeros([batch, 1, conv_dim, config.d_conv], &device),
    };
    let mut ssm_state = match state.as_ref() {
        Some(existing)
            if existing.ssm.shape().dims::<4>()
                == [batch, config.nheads, config.headdim, config.d_state] =>
        {
            existing.ssm.clone()
        }
        _ => Tensor::<B, 4>::zeros(
            [batch, config.nheads, config.headdim, config.d_state],
            &device,
        ),
    };

    let zxbcdt = hidden_states
        .clone()
        .reshape([batch * time, config.d_model])
        .matmul(params.in_proj.val())
        .reshape([batch, time, config.mamba2_in_proj_dim()]);
    let z = zxbcdt
        .clone()
        .slice_dim(2, 0..config.d_inner)
        .reshape([batch, time, config.d_inner]);
    let xbc = zxbcdt
        .clone()
        .slice_dim(
            2,
            config.d_inner..(config.d_inner + config.mamba2_conv_dim()),
        )
        .reshape([batch, time, config.mamba2_conv_dim()]);
    let dt = zxbcdt
        .slice_dim(
            2,
            (config.d_inner + config.mamba2_conv_dim())..config.mamba2_in_proj_dim(),
        )
        .reshape([batch, time, config.nheads]);

    let mut outputs = Vec::with_capacity(time);
    for step in 0..time {
        let xbc_t = xbc
            .clone()
            .slice_dim(1, step..step + 1)
            .reshape([batch, 1, conv_dim]);
        let z_t = z
            .clone()
            .slice_dim(1, step..step + 1)
            .reshape([batch, 1, config.d_inner]);
        let dt_t = dt
            .clone()
            .slice_dim(1, step..step + 1)
            .reshape([batch, config.nheads]);
        let (xbc_conv_t, next_conv_state) = mamba_depthwise_conv_step_reference(
            xbc_t,
            conv_state,
            params.conv_weight.val(),
            params.conv_bias.as_ref().map(|bias| bias.val()),
        );
        conv_state = next_conv_state;

        let x_t = xbc_conv_t.clone().slice_dim(2, 0..config.d_inner).reshape([
            batch,
            config.nheads,
            config.headdim,
        ]);
        let b_t = xbc_conv_t
            .clone()
            .slice_dim(
                2,
                config.d_inner..(config.d_inner + config.ngroups * config.d_state),
            )
            .reshape([batch, config.ngroups, config.d_state]);
        let c_t = xbc_conv_t
            .slice_dim(
                2,
                (config.d_inner + config.ngroups * config.d_state)..config.mamba2_conv_dim(),
            )
            .reshape([batch, config.ngroups, config.d_state]);

        let (y_t, next_ssm_state) =
            mamba2_state_space_duality_step_reference(x_t, dt_t, b_t, c_t, ssm_state, params);
        ssm_state = next_ssm_state;
        let y_flat = y_t.reshape([batch, 1, config.d_inner]);
        let normed =
            mamba2_rmsnorm_gated_reference(y_flat, z_t, params.norm_weight.val(), params.norm_eps);
        let out_t = normed
            .reshape([batch, config.d_inner])
            .matmul(params.out_proj.val())
            .reshape([batch, 1, 1, config.d_model]);
        outputs.push(out_t);
    }

    (
        Tensor::cat(outputs, 2),
        MambaReferenceState {
            conv: conv_state,
            ssm: ssm_state,
        },
    )
}

pub fn mamba_reference<B: Backend>(
    hidden_states: Tensor<B, 4>,
    params: &MambaSequenceParameters<B>,
    state: Option<MambaReferenceState<B>>,
) -> (Tensor<B, 4>, MambaReferenceState<B>) {
    match (params.mamba1(), params.mamba2()) {
        (Some(mamba1), None) => mamba1_reference(hidden_states, mamba1, state),
        (None, Some(mamba2)) => mamba2_reference(hidden_states, mamba2, state),
        _ => panic!("invalid mamba parameter bundle"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn::tensor::backend::Backend as BackendTrait;
    use burn_ndarray::NdArray;
    use serde::Deserialize;

    type Backend = NdArray<f32>;

    #[derive(Debug, Deserialize)]
    struct MambaFixture {
        upstream_repo: String,
        upstream_commit: String,
        d_model: usize,
        d_state: usize,
        d_conv: usize,
        expand: usize,
        dt_rank: usize,
        batch: usize,
        time: usize,
        hidden: Vec<Vec<Vec<f32>>>,
        in_proj: Vec<Vec<f32>>,
        conv_weight: Vec<Vec<f32>>,
        conv_bias: Vec<f32>,
        x_proj: Vec<Vec<f32>>,
        dt_proj_weight: Vec<Vec<f32>>,
        dt_proj_bias: Vec<f32>,
        a_log: Vec<Vec<f32>>,
        d_skip: Vec<f32>,
        out_proj: Vec<Vec<f32>>,
        expected_output: Vec<Vec<f32>>,
        expected_final_conv_state: Vec<Vec<f32>>,
        expected_final_ssm_state: Vec<Vec<f32>>,
    }

    fn fixture() -> MambaFixture {
        serde_json::from_str(include_str!(
            "../../../tests/data/mamba_c5afbdf_fixture.json"
        ))
        .expect("parse mamba fixture")
    }

    fn flatten2(values: &[Vec<f32>]) -> Vec<f32> {
        values.iter().flat_map(|row| row.iter().copied()).collect()
    }

    fn flatten3(values: &[Vec<Vec<f32>>]) -> Vec<f32> {
        values
            .iter()
            .flat_map(|plane| plane.iter().flat_map(|row| row.iter().copied()))
            .collect()
    }

    fn assert_step_mode_matches_full_sequence(
        config: MambaSequenceConfig,
        memory_system: SequenceMemorySystem,
        d_model: usize,
        batch: usize,
        time: usize,
    ) {
        let device = <Backend as BackendTrait>::Device::default();
        <Backend as BackendTrait>::seed(&device, 7);
        let resolved = config.resolve(d_model, memory_system);
        let params = MambaSequenceParameters::<Backend>::new(resolved, memory_system, &device);
        let hidden = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                (0..(batch * time * d_model))
                    .map(|idx| ((idx % 17) as f32) / 17.0 - 0.25)
                    .collect::<Vec<_>>(),
                [batch, 1, time, d_model],
            ),
            &device,
        );

        let (full_out, full_state) = mamba_reference(hidden.clone(), &params, None);
        let mut outputs = Vec::with_capacity(time);
        let mut state = None;
        for step in 0..time {
            let step_hidden = hidden.clone().slice_dim(2, step..step + 1);
            let (step_out, next_state) = mamba_reference(step_hidden, &params, state);
            outputs.push(step_out);
            state = Some(next_state);
        }
        let step_out = Tensor::cat(outputs, 2);
        let step_state = state.expect("step state");

        let out_diff = step_out.clone().sub(full_out).abs().max().into_scalar();
        let conv_diff = step_state
            .conv
            .clone()
            .sub(full_state.conv)
            .abs()
            .max()
            .into_scalar();
        let ssm_diff = step_state
            .ssm
            .clone()
            .sub(full_state.ssm)
            .abs()
            .max()
            .into_scalar();

        assert!(out_diff <= 1.0e-6, "output diff {out_diff}");
        assert!(conv_diff <= 1.0e-6, "conv diff {conv_diff}");
        assert!(ssm_diff <= 1.0e-6, "ssm diff {ssm_diff}");
    }

    fn assert_chunked_state_matches_full_sequence(
        config: MambaSequenceConfig,
        memory_system: SequenceMemorySystem,
        d_model: usize,
        batch: usize,
        time: usize,
        prefix: usize,
    ) {
        let device = <Backend as BackendTrait>::Device::default();
        <Backend as BackendTrait>::seed(&device, 11);
        let resolved = config.resolve(d_model, memory_system);
        let params = MambaSequenceParameters::<Backend>::new(resolved, memory_system, &device);
        let hidden = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                (0..(batch * time * d_model))
                    .map(|idx| ((idx % 23) as f32) / 23.0 - 0.35)
                    .collect::<Vec<_>>(),
                [batch, 1, time, d_model],
            ),
            &device,
        );

        let (full_out, full_state) = mamba_reference(hidden.clone(), &params, None);
        let (prefix_out, prefix_state) =
            mamba_reference(hidden.clone().slice_dim(2, 0..prefix), &params, None);
        let (suffix_out, suffix_state) = mamba_reference(
            hidden.clone().slice_dim(2, prefix..time),
            &params,
            Some(prefix_state),
        );
        let chunked_out = Tensor::cat(vec![prefix_out, suffix_out], 2);

        let out_diff = chunked_out.clone().sub(full_out).abs().max().into_scalar();
        let conv_diff = suffix_state
            .conv
            .clone()
            .sub(full_state.conv)
            .abs()
            .max()
            .into_scalar();
        let ssm_diff = suffix_state
            .ssm
            .clone()
            .sub(full_state.ssm)
            .abs()
            .max()
            .into_scalar();

        assert!(out_diff <= 1.0e-6, "output diff {out_diff}");
        assert!(conv_diff <= 1.0e-6, "conv diff {conv_diff}");
        assert!(ssm_diff <= 1.0e-6, "ssm diff {ssm_diff}");
    }

    #[test]
    fn mamba1_config_resolves_like_upstream_defaults() {
        let resolved =
            MambaSequenceConfig::default().resolve(256, SequenceMemorySystem::Mamba1SelectiveScan);
        assert_eq!(resolved.d_inner, 512);
        assert_eq!(resolved.d_state, 16);
        assert_eq!(resolved.d_conv, 4);
        assert_eq!(resolved.dt_rank, 16);
    }

    #[test]
    fn mamba2_config_resolves_heads_from_inner_width() {
        let resolved = MambaSequenceConfig {
            headdim: 64,
            ..Default::default()
        }
        .resolve(256, SequenceMemorySystem::Mamba2StateSpaceDuality);
        assert_eq!(resolved.d_inner, 512);
        assert_eq!(resolved.nheads, 8);
        assert_eq!(resolved.ngroups, 1);
    }

    #[test]
    fn mamba2_config_rejects_non_divisible_head_width() {
        let err = MambaSequenceConfig {
            headdim: 96,
            ..Default::default()
        }
        .validate(SequenceMemorySystem::Mamba2StateSpaceDuality, 256)
        .expect_err("expected invalid mamba2 config");
        assert!(err.contains("divisible"));
    }

    #[test]
    fn mamba1_reference_returns_expected_shapes() {
        let device = <Backend as BackendTrait>::Device::default();
        let config =
            MambaSequenceConfig::default().resolve(8, SequenceMemorySystem::Mamba1SelectiveScan);
        let params = MambaSequenceParameters::<Backend>::new(
            config,
            SequenceMemorySystem::Mamba1SelectiveScan,
            &device,
        );
        let hidden = Tensor::<Backend, 4>::zeros([2, 1, 5, 8], &device);

        let (output, state) = mamba_reference(hidden, &params, None);
        assert_eq!(output.shape().dims::<4>(), [2, 1, 5, 8]);
        assert_eq!(
            state.conv.shape().dims::<4>(),
            [2, 1, config.d_inner, config.d_conv]
        );
        assert_eq!(
            state.ssm.shape().dims::<4>(),
            [2, 1, config.d_inner, config.d_state]
        );
    }

    #[test]
    fn mamba2_reference_returns_expected_shapes() {
        let device = <Backend as BackendTrait>::Device::default();
        let config = MambaSequenceConfig {
            headdim: 8,
            ..Default::default()
        }
        .resolve(8, SequenceMemorySystem::Mamba2StateSpaceDuality);
        let params = MambaSequenceParameters::<Backend>::new(
            config,
            SequenceMemorySystem::Mamba2StateSpaceDuality,
            &device,
        );
        let hidden = Tensor::<Backend, 4>::zeros([2, 1, 5, 8], &device);

        let (output, state) = mamba_reference(hidden, &params, None);
        assert_eq!(output.shape().dims::<4>(), [2, 1, 5, 8]);
        assert_eq!(
            state.conv.shape().dims::<4>(),
            [2, 1, config.mamba2_conv_dim(), config.d_conv]
        );
        assert_eq!(
            state.ssm.shape().dims::<4>(),
            [2, config.nheads, config.headdim, config.d_state]
        );
    }

    #[test]
    fn mamba1_reference_matches_pinned_c5afbdf_fixture() {
        let fixture = fixture();
        assert_eq!(fixture.upstream_repo, MAMBA1_UPSTREAM_REPO);
        assert_eq!(fixture.upstream_commit, MAMBA1_UPSTREAM_COMMIT);

        let device = <Backend as BackendTrait>::Device::default();
        let resolved = MambaSequenceConfig {
            d_state: fixture.d_state,
            d_conv: fixture.d_conv,
            expand: fixture.expand,
            dt_rank: Some(fixture.dt_rank),
            ..Default::default()
        }
        .resolve(fixture.d_model, SequenceMemorySystem::Mamba1SelectiveScan);
        let mut params = Mamba1SequenceParameters::<Backend>::new(resolved, &device);
        params.in_proj = Param::from_tensor(Tensor::<Backend, 2>::from_data(
            TensorData::new(
                flatten2(&fixture.in_proj),
                [fixture.d_model, resolved.d_inner * 2],
            ),
            &device,
        ));
        params.conv_weight = Param::from_tensor(Tensor::<Backend, 2>::from_data(
            TensorData::new(
                flatten2(&fixture.conv_weight),
                [resolved.d_inner, resolved.d_conv],
            ),
            &device,
        ));
        params.conv_bias = Some(Param::from_tensor(Tensor::<Backend, 1>::from_data(
            TensorData::new(fixture.conv_bias.clone(), [resolved.d_inner]),
            &device,
        )));
        params.x_proj = Param::from_tensor(Tensor::<Backend, 2>::from_data(
            TensorData::new(
                flatten2(&fixture.x_proj),
                [resolved.d_inner, resolved.dt_rank + resolved.d_state * 2],
            ),
            &device,
        ));
        params.dt_proj_weight = Param::from_tensor(Tensor::<Backend, 2>::from_data(
            TensorData::new(
                flatten2(&fixture.dt_proj_weight),
                [resolved.dt_rank, resolved.d_inner],
            ),
            &device,
        ));
        params.dt_proj_bias = Param::from_tensor(Tensor::<Backend, 1>::from_data(
            TensorData::new(fixture.dt_proj_bias.clone(), [resolved.d_inner]),
            &device,
        ));
        params.a_log = Param::from_tensor(Tensor::<Backend, 2>::from_data(
            TensorData::new(
                flatten2(&fixture.a_log),
                [resolved.d_inner, resolved.d_state],
            ),
            &device,
        ));
        params.d_skip = Param::from_tensor(Tensor::<Backend, 1>::from_data(
            TensorData::new(fixture.d_skip.clone(), [resolved.d_inner]),
            &device,
        ));
        params.out_proj = Param::from_tensor(Tensor::<Backend, 2>::from_data(
            TensorData::new(
                flatten2(&fixture.out_proj),
                [resolved.d_inner, fixture.d_model],
            ),
            &device,
        ));

        let hidden = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                flatten3(&fixture.hidden),
                [fixture.batch, 1, fixture.time, fixture.d_model],
            ),
            &device,
        );
        let wrapped = MambaSequenceParameters {
            mamba1: Some(params),
            mamba2: None,
        };
        let (output, state) = mamba_reference(hidden, &wrapped, None);
        let expected_output = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                flatten3(&vec![fixture.expected_output.clone()]),
                [fixture.batch, 1, fixture.time, fixture.d_model],
            ),
            &device,
        );
        let expected_conv = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                flatten2(&fixture.expected_final_conv_state),
                [fixture.batch, 1, resolved.d_inner, resolved.d_conv],
            ),
            &device,
        );
        let expected_ssm = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                flatten2(&fixture.expected_final_ssm_state),
                [fixture.batch, 1, resolved.d_inner, resolved.d_state],
            ),
            &device,
        );

        let max_output_diff = output
            .clone()
            .sub(expected_output)
            .abs()
            .max()
            .into_scalar();
        let max_conv_diff = state
            .conv
            .clone()
            .sub(expected_conv)
            .abs()
            .max()
            .into_scalar();
        let max_ssm_diff = state
            .ssm
            .clone()
            .sub(expected_ssm)
            .abs()
            .max()
            .into_scalar();

        assert!(max_output_diff <= 1.0e-6, "output diff {max_output_diff}");
        assert!(max_conv_diff <= 1.0e-7, "conv diff {max_conv_diff}");
        assert!(max_ssm_diff <= 1.0e-7, "ssm diff {max_ssm_diff}");
    }

    #[test]
    fn mamba1_reference_step_mode_matches_full_sequence() {
        assert_step_mode_matches_full_sequence(
            MambaSequenceConfig::default(),
            SequenceMemorySystem::Mamba1SelectiveScan,
            8,
            2,
            5,
        );
    }

    #[test]
    fn mamba2_reference_step_mode_matches_full_sequence() {
        assert_step_mode_matches_full_sequence(
            MambaSequenceConfig {
                headdim: 8,
                ..Default::default()
            },
            SequenceMemorySystem::Mamba2StateSpaceDuality,
            8,
            2,
            5,
        );
    }

    #[test]
    fn mamba1_reference_chunked_state_matches_full_sequence() {
        assert_chunked_state_matches_full_sequence(
            MambaSequenceConfig::default(),
            SequenceMemorySystem::Mamba1SelectiveScan,
            8,
            2,
            6,
            2,
        );
    }

    #[test]
    fn mamba2_reference_chunked_state_matches_full_sequence() {
        assert_chunked_state_matches_full_sequence(
            MambaSequenceConfig {
                headdim: 8,
                ..Default::default()
            },
            SequenceMemorySystem::Mamba2StateSpaceDuality,
            8,
            2,
            6,
            2,
        );
    }
}
