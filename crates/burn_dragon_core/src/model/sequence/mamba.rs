use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution as TensorDistribution, Tensor, TensorData, activation};
use serde::{Deserialize, Serialize};

#[cfg_attr(not(test), allow(dead_code))]
pub const MAMBA1_UPSTREAM_REPO: &str = "https://github.com/state-spaces/mamba";
#[cfg_attr(not(test), allow(dead_code))]
pub const MAMBA1_UPSTREAM_COMMIT: &str = "c5afbdf";

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
}

impl MambaSequenceConfig {
    pub fn resolve(&self, d_model: usize) -> ResolvedMambaSequenceConfig {
        let d_state = self.d_state.max(1);
        let d_conv = self.d_conv.max(1);
        let expand = self.expand.max(1);
        let d_inner = d_model.max(1) * expand;
        let dt_rank = self.dt_rank.unwrap_or_else(|| d_model.div_ceil(16)).max(1);
        ResolvedMambaSequenceConfig {
            d_model: d_model.max(1),
            d_inner,
            d_state,
            d_conv,
            dt_rank,
            dt_min: self.dt_min.max(1.0e-6),
            dt_max: self.dt_max.max(self.dt_min.max(1.0e-6)),
            dt_scale: self.dt_scale.max(1.0e-6),
            conv_bias: self.conv_bias,
            use_fast_path: self.use_fast_path,
        }
    }
}

#[derive(Module, Debug)]
pub struct MambaSequenceParameters<B: Backend> {
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

impl<B: Backend> MambaSequenceParameters<B> {
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
    let [batch, views, d_inner] = x_t.shape().dims::<3>();
    let d_conv = conv_state.shape().dims::<4>()[3];
    let device = x_t.device();
    let conv_tail = if d_conv > 1 {
        conv_state.clone().slice_dim(3, 1..d_conv)
    } else {
        Tensor::<B, 4>::zeros([batch, views, d_inner, 0], &device)
    };
    let next_conv_state = Tensor::cat(vec![conv_tail, x_t.clone().unsqueeze_dim::<4>(3)], 3);
    let mut u_t = (next_conv_state.clone() * conv_weight.reshape([1, 1, d_inner, d_conv]))
        .sum_dim(3)
        .reshape([batch, views, d_inner]);
    if let Some(bias) = conv_bias {
        u_t = u_t + bias.reshape([1, 1, d_inner]);
    }
    (silu(u_t), next_conv_state)
}

pub(crate) fn mamba_selective_scan_step_reference<B: Backend>(
    u_t: Tensor<B, 3>,
    z_t: Tensor<B, 3>,
    ssm_state: Tensor<B, 4>,
    params: &MambaSequenceParameters<B>,
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

pub fn mamba_reference<B: Backend>(
    hidden_states: Tensor<B, 4>,
    params: &MambaSequenceParameters<B>,
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
            mamba_selective_scan_step_reference(u_t, z_t, ssm_state, params);
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

    #[test]
    fn mamba_config_resolves_like_upstream_defaults() {
        let resolved = MambaSequenceConfig::default().resolve(256);
        assert_eq!(resolved.d_inner, 512);
        assert_eq!(resolved.d_state, 16);
        assert_eq!(resolved.d_conv, 4);
        assert_eq!(resolved.dt_rank, 16);
    }

    #[test]
    fn mamba_reference_returns_expected_shapes() {
        let device = <Backend as BackendTrait>::Device::default();
        let config = MambaSequenceConfig::default().resolve(8);
        let params = MambaSequenceParameters::<Backend>::new(config, &device);
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
    fn mamba_reference_matches_pinned_c5afbdf_fixture() {
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
        .resolve(fixture.d_model);
        let mut params = MambaSequenceParameters::<Backend>::new(resolved, &device);
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
        let (output, state) = mamba_reference(hidden, &params, None);
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
    fn mamba_reference_step_mode_matches_full_sequence() {
        let device = <Backend as BackendTrait>::Device::default();
        <Backend as BackendTrait>::seed(&device, 7);
        let resolved = MambaSequenceConfig::default().resolve(8);
        let params = MambaSequenceParameters::<Backend>::new(resolved, &device);
        let hidden = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                (0..(2 * 1 * 5 * 8))
                    .map(|idx| ((idx % 17) as f32) / 17.0 - 0.25)
                    .collect::<Vec<_>>(),
                [2, 1, 5, 8],
            ),
            &device,
        );

        let (full_out, full_state) = mamba_reference(hidden.clone(), &params, None);

        let mut outputs = Vec::with_capacity(5);
        let mut state = None;
        for step in 0..5 {
            let step_hidden = hidden.clone().slice_dim(2, step..step + 1);
            let (step_out, next_state) = mamba_reference(step_hidden, &params, state);
            outputs.push(step_out);
            state = Some(next_state);
        }
        let step_out = Tensor::cat(outputs, 2);
        let step_state = state.expect("mamba step state");

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

    #[test]
    fn mamba_reference_chunked_state_matches_full_sequence() {
        let device = <Backend as BackendTrait>::Device::default();
        <Backend as BackendTrait>::seed(&device, 11);
        let resolved = MambaSequenceConfig::default().resolve(8);
        let params = MambaSequenceParameters::<Backend>::new(resolved, &device);
        let hidden = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                (0..(2 * 1 * 6 * 8))
                    .map(|idx| ((idx % 23) as f32) / 23.0 - 0.35)
                    .collect::<Vec<_>>(),
                [2, 1, 6, 8],
            ),
            &device,
        );

        let (full_out, full_state) = mamba_reference(hidden.clone(), &params, None);
        let (prefix_out, prefix_state) =
            mamba_reference(hidden.clone().slice_dim(2, 0..2), &params, None);
        let (suffix_out, suffix_state) = mamba_reference(
            hidden.clone().slice_dim(2, 2..6),
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
    fn mamba_depthwise_conv_step_matches_full_reference_first_step_state() {
        let device = <Backend as BackendTrait>::Device::default();
        <Backend as BackendTrait>::seed(&device, 5);
        let resolved = MambaSequenceConfig::default().resolve(8);
        let params = MambaSequenceParameters::<Backend>::new(resolved, &device);
        let hidden = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                (0..8)
                    .map(|idx| idx as f32 / 8.0 - 0.25)
                    .collect::<Vec<_>>(),
                [1, 1, 1, 8],
            ),
            &device,
        );
        let xz = hidden
            .clone()
            .reshape([1, resolved.d_model])
            .matmul(params.in_proj.val())
            .reshape([1, 1, resolved.d_inner * 2]);
        let x_t = xz
            .clone()
            .slice_dim(2, 0..resolved.d_inner)
            .reshape([1, 1, resolved.d_inner]);
        let z_t = xz
            .slice_dim(2, resolved.d_inner..(resolved.d_inner * 2))
            .reshape([1, 1, resolved.d_inner]);
        let initial_conv =
            Tensor::<Backend, 4>::zeros([1, 1, resolved.d_inner, resolved.d_conv], &device);
        let initial_ssm =
            Tensor::<Backend, 4>::zeros([1, 1, resolved.d_inner, resolved.d_state], &device);

        let (_u_t, helper_conv_state) = mamba_depthwise_conv_step_reference(
            x_t.clone(),
            initial_conv,
            params.conv_weight.val(),
            params.conv_bias.as_ref().map(|bias| bias.val()),
        );
        let (_y_t, helper_ssm_state) = mamba_selective_scan_step_reference(
            mamba_depthwise_conv_step_reference(
                x_t,
                Tensor::<Backend, 4>::zeros([1, 1, resolved.d_inner, resolved.d_conv], &device),
                params.conv_weight.val(),
                params.conv_bias.as_ref().map(|bias| bias.val()),
            )
            .0,
            z_t,
            initial_ssm,
            &params,
        );
        let (_full_out, full_state) = mamba_reference(hidden, &params, None);

        let conv_diff = helper_conv_state
            .sub(full_state.conv)
            .abs()
            .max()
            .into_scalar();
        let ssm_diff = helper_ssm_state
            .sub(full_state.ssm)
            .abs()
            .max()
            .into_scalar();
        assert!(conv_diff <= 1.0e-6, "conv diff {conv_diff}");
        assert!(ssm_diff <= 1.0e-6, "ssm diff {ssm_diff}");
    }

    #[test]
    fn mamba_selective_scan_step_matches_full_reference_first_step_output() {
        let device = <Backend as BackendTrait>::Device::default();
        <Backend as BackendTrait>::seed(&device, 17);
        let resolved = MambaSequenceConfig::default().resolve(8);
        let params = MambaSequenceParameters::<Backend>::new(resolved, &device);
        let hidden = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                (0..8)
                    .map(|idx| idx as f32 / 11.0 - 0.2)
                    .collect::<Vec<_>>(),
                [1, 1, 1, 8],
            ),
            &device,
        );
        let xz = hidden
            .clone()
            .reshape([1, resolved.d_model])
            .matmul(params.in_proj.val())
            .reshape([1, 1, resolved.d_inner * 2]);
        let x_t = xz
            .clone()
            .slice_dim(2, 0..resolved.d_inner)
            .reshape([1, 1, resolved.d_inner]);
        let z_t = xz
            .slice_dim(2, resolved.d_inner..(resolved.d_inner * 2))
            .reshape([1, 1, resolved.d_inner]);
        let (u_t, _conv_state) = mamba_depthwise_conv_step_reference(
            x_t,
            Tensor::<Backend, 4>::zeros([1, 1, resolved.d_inner, resolved.d_conv], &device),
            params.conv_weight.val(),
            params.conv_bias.as_ref().map(|bias| bias.val()),
        );
        let (y_t, helper_ssm_state) = mamba_selective_scan_step_reference(
            u_t,
            z_t,
            Tensor::<Backend, 4>::zeros([1, 1, resolved.d_inner, resolved.d_state], &device),
            &params,
        );
        let helper_out = y_t
            .reshape([1, resolved.d_inner])
            .matmul(params.out_proj.val())
            .reshape([1, 1, 1, resolved.d_model]);
        let (full_out, full_state) = mamba_reference(hidden, &params, None);

        let out_diff = helper_out.sub(full_out).abs().max().into_scalar();
        let ssm_diff = helper_ssm_state
            .sub(full_state.ssm)
            .abs()
            .max()
            .into_scalar();
        assert!(out_diff <= 1.0e-6, "output diff {out_diff}");
        assert!(ssm_diff <= 1.0e-6, "ssm diff {ssm_diff}");
    }
}
