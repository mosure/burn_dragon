use std::sync::{Mutex, OnceLock};

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_dragon_core::{
    BDH, BDHConfig, FusedKernelConfig, MambaSequenceConfig, SequenceKernelConfig,
    SequenceMemorySystem,
};
use burn_dragon_kernel::kernels::sequence::mamba3::forward::{
    Mamba3TensorizedState, tensorized_mamba3_forward,
};
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};

type TestBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

fn init_runtime(device: &<TestBackend as Backend>::Device) {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

fn init_seeded_model(config: BDHConfig) -> BDH<TestBackend> {
    static INIT_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _guard = INIT_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("seed lock");
    let device = <TestBackend as Backend>::Device::default();
    init_runtime(&device);
    <TestBackend as Backend>::seed(&device, 2026);
    BDH::<TestBackend>::new(config, &device)
}

fn tensor_max_abs_diff<const D: usize>(
    lhs: Tensor<TestBackend, D>,
    rhs: Tensor<TestBackend, D>,
) -> f32 {
    let lhs = lhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("lhs vec");
    let rhs = rhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rhs vec");
    lhs.into_iter()
        .zip(rhs)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max)
}

fn deterministic_tensor<const D: usize>(
    device: &<TestBackend as Backend>::Device,
    shape: [usize; D],
    period: usize,
) -> Tensor<TestBackend, D> {
    let len = shape.iter().product::<usize>();
    Tensor::<TestBackend, D>::from_data(
        TensorData::new(
            (0..len)
                .map(|idx| ((idx % period) as f32) / period as f32 - 0.5)
                .collect::<Vec<_>>(),
            shape,
        ),
        device,
    )
}

fn make_tokens(
    device: &<TestBackend as Backend>::Device,
    token_values: Vec<i64>,
    shape: [usize; 2],
) -> Tensor<TestBackend, 2, Int> {
    Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(token_values, shape), device)
}

fn shakespeare_like_mamba3_config() -> MambaSequenceConfig {
    MambaSequenceConfig {
        d_state: 16,
        expand: 2,
        headdim: 64,
        ngroups: 4,
        rope_fraction: 0.5,
        chunk_size: 64,
        use_fast_path: true,
        ..Default::default()
    }
}

#[test]
fn mamba3_kernel_tensorized_token_step_state_matches_full_sequence_on_shakespeare_like_shape() {
    let device = <TestBackend as Backend>::Device::default();
    init_runtime(&device);
    let config = shakespeare_like_mamba3_config()
        .resolve(128, SequenceMemorySystem::Mamba3StateSpaceDuality);
    let batch = 1;
    let time = 64;
    let d_model = 128;
    let hidden = deterministic_tensor(&device, [batch, 1, time, d_model], 257);
    let in_proj = deterministic_tensor(&device, [d_model, config.mamba3_in_proj_dim()], 263);
    let dt_bias = deterministic_tensor(&device, [config.nheads], 269);
    let b_bias = deterministic_tensor(&device, [config.nheads, config.d_state], 271);
    let c_bias = deterministic_tensor(&device, [config.nheads, config.d_state], 277);
    let b_norm_weight = deterministic_tensor(&device, [config.d_state], 281);
    let c_norm_weight = deterministic_tensor(&device, [config.d_state], 283);
    let d_skip = deterministic_tensor(&device, [config.nheads], 293);
    let out_proj = deterministic_tensor(&device, [config.d_inner, d_model], 307);

    let full = tensorized_mamba3_forward(
        hidden.clone(),
        config.d_inner,
        config.d_state,
        config.headdim,
        config.ngroups,
        config.num_rope_angles,
        config.norm_eps,
        config.a_floor,
        config.chunk_size,
        in_proj.clone(),
        dt_bias.clone(),
        b_bias.clone(),
        c_bias.clone(),
        b_norm_weight.clone(),
        c_norm_weight.clone(),
        d_skip.clone(),
        out_proj.clone(),
        None::<Mamba3TensorizedState<TestBackend>>,
    );

    let mut state = None::<Mamba3TensorizedState<TestBackend>>;
    let mut outputs = Vec::with_capacity(time);
    for step in 0..time {
        let step_output = tensorized_mamba3_forward(
            hidden.clone().slice_dim(2, step..step + 1),
            config.d_inner,
            config.d_state,
            config.headdim,
            config.ngroups,
            config.num_rope_angles,
            config.norm_eps,
            config.a_floor,
            config.chunk_size,
            in_proj.clone(),
            dt_bias.clone(),
            b_bias.clone(),
            c_bias.clone(),
            b_norm_weight.clone(),
            c_norm_weight.clone(),
            d_skip.clone(),
            out_proj.clone(),
            state,
        );
        state = Some(step_output.state);
        outputs.push(step_output.context);
    }

    let stepped_context = Tensor::cat(outputs, 2);
    let final_state = state.expect("final token-step state");
    let context_diff = tensor_max_abs_diff(full.context, stepped_context);
    let ssm_diff = tensor_max_abs_diff(full.state.ssm, final_state.ssm);
    let angle_diff = tensor_max_abs_diff(full.state.angle, final_state.angle);
    let k_diff = tensor_max_abs_diff(full.state.k, final_state.k);
    let v_diff = tensor_max_abs_diff(full.state.v, final_state.v);

    assert!(
        context_diff <= 2.0e-3,
        "expected token-step mamba3 tensorized context parity, max diff {context_diff}"
    );
    assert!(
        ssm_diff <= 2.0e-3,
        "expected token-step mamba3 tensorized ssm parity, max diff {ssm_diff}"
    );
    assert!(
        angle_diff <= 2.0e-4,
        "expected token-step mamba3 tensorized angle parity, max diff {angle_diff}"
    );
    assert!(
        k_diff <= 2.0e-4,
        "expected token-step mamba3 tensorized k parity, max diff {k_diff}"
    );
    assert!(
        v_diff <= 2.0e-4,
        "expected token-step mamba3 tensorized v parity, max diff {v_diff}"
    );
}

#[test]
fn mamba3_bdh_full_forward_matches_token_step_recurrence_on_shakespeare_like_shape() {
    let device = <TestBackend as Backend>::Device::default();
    init_runtime(&device);
    let tokens = make_tokens(
        &device,
        (0..64).map(|idx| (idx % 32) as i64).collect::<Vec<_>>(),
        [1, 64],
    );

    let mut config = BDHConfig {
        n_layer: 4,
        n_embd: 128,
        n_head: 4,
        mlp_internal_dim_multiplier: 4,
        vocab_size: 69,
        dropout: 0.0,
        sequence_kernel: SequenceKernelConfig::reference(
            SequenceMemorySystem::Mamba3StateSpaceDuality,
        ),
        fused_kernels: FusedKernelConfig {
            enabled: true,
            ..Default::default()
        },
        ..Default::default()
    };
    config.mamba = shakespeare_like_mamba3_config();
    let model = init_seeded_model(config);

    let logits_full = model.forward(tokens.clone());
    let mut recurrent_state = model.init_state();
    let mut logits_steps = Vec::new();
    for step in 0..tokens.shape().dims::<2>()[1] {
        let step_tokens = tokens.clone().slice_dim(1, step..step + 1);
        logits_steps.push(model.forward_with_state(step_tokens, &mut recurrent_state));
    }
    let logits_stepwise = Tensor::cat(logits_steps, 1);
    let max_diff = tensor_max_abs_diff(logits_full, logits_stepwise);
    assert!(
        max_diff <= 2.0e-3,
        "expected mamba3 BDH full forward and token-step recurrence to match on realistic shape, max diff {max_diff}"
    );
}
