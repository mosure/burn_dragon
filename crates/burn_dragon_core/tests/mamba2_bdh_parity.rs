use std::sync::{Mutex, OnceLock};

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
#[cfg(feature = "cuda")]
use burn_autodiff::Autodiff;
#[cfg(feature = "cuda")]
use burn_cuda::Cuda;
use burn_dragon_core::{
    BDH, BDHConfig, FusedKernelConfig, MambaSequenceConfig, SequenceKernelConfig,
    SequenceMemorySystem,
};
#[cfg(feature = "cuda")]
use burn_dragon_kernel::kernels::sequence::mamba2::forward::{
    CudaShellCoreMode, CudaSsdCoreMode, tensorized_mamba2_forward_custom_backward_with_cuda_modes,
};
use burn_dragon_kernel::kernels::sequence::mamba2::forward::{
    Mamba2TensorizedState, tensorized_mamba2_forward,
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

fn with_mamba2_forward_env<T>(value: &str, f: impl FnOnce() -> T) -> T {
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _guard = ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("env lock");
    let previous = std::env::var("BURN_DRAGON_MAMBA2_TENSORIZED_FORWARD").ok();
    // SAFETY: the lock above serializes env mutation within this test harness.
    unsafe { std::env::set_var("BURN_DRAGON_MAMBA2_TENSORIZED_FORWARD", value) };
    let output = f();
    match previous {
        // SAFETY: the same lock still guards env mutation here.
        Some(previous) => unsafe {
            std::env::set_var("BURN_DRAGON_MAMBA2_TENSORIZED_FORWARD", previous)
        },
        // SAFETY: the same lock still guards env mutation here.
        None => unsafe { std::env::remove_var("BURN_DRAGON_MAMBA2_TENSORIZED_FORWARD") },
    }
    output
}

fn tensor_max_abs_diff_backend<B: Backend, const D: usize>(
    lhs: Tensor<B, D>,
    rhs: Tensor<B, D>,
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

fn tensor_max_abs_diff<const D: usize>(
    lhs: Tensor<TestBackend, D>,
    rhs: Tensor<TestBackend, D>,
) -> f32 {
    tensor_max_abs_diff_backend(lhs, rhs)
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

fn shakespeare_like_mamba2_config() -> MambaSequenceConfig {
    MambaSequenceConfig {
        d_state: 16,
        d_conv: 4,
        expand: 2,
        headdim: 64,
        ngroups: 4,
        ..Default::default()
    }
}

#[test]
fn mamba2_bdh_fast_path_matches_reference_logits_on_shakespeare_like_shape() {
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
            SequenceMemorySystem::Mamba2StateSpaceDuality,
        ),
        fused_kernels: FusedKernelConfig {
            enabled: true,
            ..Default::default()
        },
        ..Default::default()
    };
    config.mamba = MambaSequenceConfig {
        use_fast_path: true,
        ..shakespeare_like_mamba2_config()
    };
    let model = init_seeded_model(config);
    let reference_logits = with_mamba2_forward_env("0", || model.forward(tokens.clone()));
    let fast_logits = with_mamba2_forward_env("1", || model.forward(tokens));
    let max_diff = tensor_max_abs_diff(reference_logits, fast_logits);
    assert!(
        max_diff <= 2.0e-3,
        "expected mamba2 BDH fast path logits to match reference on realistic shape, max diff {max_diff}"
    );
}

#[test]
fn mamba2_kernel_tensorized_chunked_state_matches_full_sequence_on_shakespeare_like_shape() {
    let device = <TestBackend as Backend>::Device::default();
    init_runtime(&device);
    let config = shakespeare_like_mamba2_config()
        .resolve(128, SequenceMemorySystem::Mamba2StateSpaceDuality);
    let batch = 1;
    let time = 64;
    let split = 17;
    let d_model = 128;
    let hidden = deterministic_tensor(&device, [batch, 1, time, d_model], 257);
    let in_proj = deterministic_tensor(&device, [d_model, config.mamba2_in_proj_dim()], 263);
    let conv_weight = deterministic_tensor(&device, [config.mamba2_conv_dim(), config.d_conv], 269);
    let conv_bias = Some(deterministic_tensor(
        &device,
        [config.mamba2_conv_dim()],
        271,
    ));
    let dt_bias = deterministic_tensor(&device, [config.nheads], 277);
    let a_log = deterministic_tensor(&device, [config.nheads], 281);
    let d_skip = deterministic_tensor(&device, [config.nheads], 283);
    let norm_weight = deterministic_tensor(&device, [config.d_inner], 293);
    let out_proj = deterministic_tensor(&device, [config.d_inner, d_model], 307);

    let full = tensorized_mamba2_forward(
        hidden.clone(),
        config.d_inner,
        config.d_state,
        config.d_conv,
        config.headdim,
        config.ngroups,
        in_proj.clone(),
        conv_weight.clone(),
        conv_bias.clone(),
        dt_bias.clone(),
        a_log.clone(),
        d_skip.clone(),
        norm_weight.clone(),
        config.norm_eps,
        out_proj.clone(),
        None::<Mamba2TensorizedState<TestBackend>>,
    );
    let prefix = tensorized_mamba2_forward(
        hidden.clone().slice_dim(2, 0..split),
        config.d_inner,
        config.d_state,
        config.d_conv,
        config.headdim,
        config.ngroups,
        in_proj.clone(),
        conv_weight.clone(),
        conv_bias.clone(),
        dt_bias.clone(),
        a_log.clone(),
        d_skip.clone(),
        norm_weight.clone(),
        config.norm_eps,
        out_proj.clone(),
        None::<Mamba2TensorizedState<TestBackend>>,
    );
    let suffix = tensorized_mamba2_forward(
        hidden.slice_dim(2, split..time),
        config.d_inner,
        config.d_state,
        config.d_conv,
        config.headdim,
        config.ngroups,
        in_proj,
        conv_weight,
        conv_bias,
        dt_bias,
        a_log,
        d_skip,
        norm_weight,
        config.norm_eps,
        out_proj,
        Some(prefix.state),
    );

    let chunked_context = Tensor::cat(vec![prefix.context, suffix.context], 2);
    let context_diff = tensor_max_abs_diff(full.context, chunked_context);
    let conv_diff = tensor_max_abs_diff(full.state.conv, suffix.state.conv);
    let ssm_diff = tensor_max_abs_diff(full.state.ssm, suffix.state.ssm);
    assert!(
        context_diff <= 2.0e-3,
        "expected chunked mamba2 tensorized context parity, max diff {context_diff}"
    );
    assert!(
        conv_diff <= 1.0e-4,
        "expected chunked mamba2 conv parity, max diff {conv_diff}"
    );
    assert!(
        ssm_diff <= 2.0e-3,
        "expected chunked mamba2 ssm parity, max diff {ssm_diff}"
    );
}

#[test]
fn mamba2_kernel_tensorized_token_step_state_matches_full_sequence_on_shakespeare_like_shape() {
    let device = <TestBackend as Backend>::Device::default();
    init_runtime(&device);
    let config = shakespeare_like_mamba2_config()
        .resolve(128, SequenceMemorySystem::Mamba2StateSpaceDuality);
    let batch = 1;
    let time = 64;
    let d_model = 128;
    let hidden = deterministic_tensor(&device, [batch, 1, time, d_model], 257);
    let in_proj = deterministic_tensor(&device, [d_model, config.mamba2_in_proj_dim()], 263);
    let conv_weight = deterministic_tensor(&device, [config.mamba2_conv_dim(), config.d_conv], 269);
    let conv_bias = Some(deterministic_tensor(
        &device,
        [config.mamba2_conv_dim()],
        271,
    ));
    let dt_bias = deterministic_tensor(&device, [config.nheads], 277);
    let a_log = deterministic_tensor(&device, [config.nheads], 281);
    let d_skip = deterministic_tensor(&device, [config.nheads], 283);
    let norm_weight = deterministic_tensor(&device, [config.d_inner], 293);
    let out_proj = deterministic_tensor(&device, [config.d_inner, d_model], 307);

    let full = tensorized_mamba2_forward(
        hidden.clone(),
        config.d_inner,
        config.d_state,
        config.d_conv,
        config.headdim,
        config.ngroups,
        in_proj.clone(),
        conv_weight.clone(),
        conv_bias.clone(),
        dt_bias.clone(),
        a_log.clone(),
        d_skip.clone(),
        norm_weight.clone(),
        config.norm_eps,
        out_proj.clone(),
        None::<Mamba2TensorizedState<TestBackend>>,
    );

    let mut state = None::<Mamba2TensorizedState<TestBackend>>;
    let mut outputs = Vec::with_capacity(time);
    for step in 0..time {
        let step_output = tensorized_mamba2_forward(
            hidden.clone().slice_dim(2, step..step + 1),
            config.d_inner,
            config.d_state,
            config.d_conv,
            config.headdim,
            config.ngroups,
            in_proj.clone(),
            conv_weight.clone(),
            conv_bias.clone(),
            dt_bias.clone(),
            a_log.clone(),
            d_skip.clone(),
            norm_weight.clone(),
            config.norm_eps,
            out_proj.clone(),
            state,
        );
        state = Some(step_output.state);
        outputs.push(step_output.context);
    }

    let stepped_context = Tensor::cat(outputs, 2);
    let final_state = state.expect("final token-step state");
    let context_diff = tensor_max_abs_diff(full.context, stepped_context);
    let conv_diff = tensor_max_abs_diff(full.state.conv, final_state.conv);
    let ssm_diff = tensor_max_abs_diff(full.state.ssm, final_state.ssm);
    assert!(
        context_diff <= 2.0e-3,
        "expected token-step mamba2 tensorized context parity, max diff {context_diff}"
    );
    assert!(
        conv_diff <= 1.0e-4,
        "expected token-step mamba2 conv parity, max diff {conv_diff}"
    );
    assert!(
        ssm_diff <= 2.0e-3,
        "expected token-step mamba2 ssm parity, max diff {ssm_diff}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn mamba2_cuda_shell_fused_token_step_matches_full_sequence_on_shakespeare_like_shape() {
    type CudaBackend = Autodiff<Cuda<f32, i32>>;

    let device = <CudaBackend as Backend>::Device::default();
    let config = shakespeare_like_mamba2_config()
        .resolve(128, SequenceMemorySystem::Mamba2StateSpaceDuality);
    let batch = 1;
    let time = 64;
    let d_model = 128;
    let hidden = Tensor::<CudaBackend, 4>::from_data(
        TensorData::new(
            (0..(batch * time * d_model))
                .map(|idx| ((idx % 257) as f32) / 257.0 - 0.5)
                .collect::<Vec<_>>(),
            [batch, 1, time, d_model],
        ),
        &device,
    );
    let in_proj = Tensor::<CudaBackend, 2>::from_data(
        TensorData::new(
            (0..(d_model * config.mamba2_in_proj_dim()))
                .map(|idx| ((idx % 263) as f32) / 263.0 - 0.5)
                .collect::<Vec<_>>(),
            [d_model, config.mamba2_in_proj_dim()],
        ),
        &device,
    );
    let conv_weight = Tensor::<CudaBackend, 2>::from_data(
        TensorData::new(
            (0..(config.mamba2_conv_dim() * config.d_conv))
                .map(|idx| ((idx % 269) as f32) / 269.0 - 0.5)
                .collect::<Vec<_>>(),
            [config.mamba2_conv_dim(), config.d_conv],
        ),
        &device,
    );
    let conv_bias = Some(Tensor::<CudaBackend, 1>::from_data(
        TensorData::new(
            (0..config.mamba2_conv_dim())
                .map(|idx| ((idx % 271) as f32) / 271.0 - 0.5)
                .collect::<Vec<_>>(),
            [config.mamba2_conv_dim()],
        ),
        &device,
    ));
    let dt_bias = Tensor::<CudaBackend, 1>::from_data(
        TensorData::new(
            (0..config.nheads)
                .map(|idx| ((idx % 277) as f32) / 277.0 - 0.5)
                .collect::<Vec<_>>(),
            [config.nheads],
        ),
        &device,
    );
    let a_log = Tensor::<CudaBackend, 1>::from_data(
        TensorData::new(
            (0..config.nheads)
                .map(|idx| ((idx % 281) as f32) / 281.0 - 0.5)
                .collect::<Vec<_>>(),
            [config.nheads],
        ),
        &device,
    );
    let d_skip = Tensor::<CudaBackend, 1>::from_data(
        TensorData::new(
            (0..config.nheads)
                .map(|idx| ((idx % 283) as f32) / 283.0 - 0.5)
                .collect::<Vec<_>>(),
            [config.nheads],
        ),
        &device,
    );
    let norm_weight = Tensor::<CudaBackend, 1>::from_data(
        TensorData::new(
            (0..config.d_inner)
                .map(|idx| ((idx % 293) as f32) / 293.0 - 0.5)
                .collect::<Vec<_>>(),
            [config.d_inner],
        ),
        &device,
    );
    let out_proj = Tensor::<CudaBackend, 2>::from_data(
        TensorData::new(
            (0..(config.d_inner * d_model))
                .map(|idx| ((idx % 307) as f32) / 307.0 - 0.5)
                .collect::<Vec<_>>(),
            [config.d_inner, d_model],
        ),
        &device,
    );

    let full = tensorized_mamba2_forward_custom_backward_with_cuda_modes(
        hidden.clone(),
        config.d_inner,
        config.d_state,
        config.d_conv,
        config.headdim,
        config.ngroups,
        in_proj.clone(),
        conv_weight.clone(),
        conv_bias.clone(),
        dt_bias.clone(),
        a_log.clone(),
        d_skip.clone(),
        norm_weight.clone(),
        config.norm_eps,
        out_proj.clone(),
        None::<Mamba2TensorizedState<CudaBackend>>,
        CudaSsdCoreMode::ForcedEnabled,
        CudaShellCoreMode::ForcedEnabled,
    )
    .expect("cuda shell-fused mamba2 output");

    let mut state = None::<Mamba2TensorizedState<CudaBackend>>;
    let mut outputs = Vec::with_capacity(time);
    for step in 0..time {
        let step_output = tensorized_mamba2_forward_custom_backward_with_cuda_modes(
            hidden.clone().slice_dim(2, step..step + 1),
            config.d_inner,
            config.d_state,
            config.d_conv,
            config.headdim,
            config.ngroups,
            in_proj.clone(),
            conv_weight.clone(),
            conv_bias.clone(),
            dt_bias.clone(),
            a_log.clone(),
            d_skip.clone(),
            norm_weight.clone(),
            config.norm_eps,
            out_proj.clone(),
            state,
            CudaSsdCoreMode::ForcedEnabled,
            CudaShellCoreMode::ForcedEnabled,
        )
        .expect("cuda shell-fused token-step output");
        state = Some(step_output.state);
        outputs.push(step_output.context);
    }

    let stepped_context = Tensor::cat(outputs, 2);
    let final_state = state.expect("final shell-fused token-step state");

    let full_context = full
        .context
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("full context");
    let stepped_context_vec = stepped_context
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("stepped context");
    let full_ssm = full
        .state
        .ssm
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("full ssm");
    let final_ssm = final_state
        .ssm
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("final ssm");
    let full_conv = full
        .state
        .conv
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("full conv");
    let final_conv = final_state
        .conv
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("final conv");

    let context_diff = full_context
        .iter()
        .zip(stepped_context_vec.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max);
    let ssm_diff = full_ssm
        .iter()
        .zip(final_ssm.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max);
    let conv_diff = full_conv
        .iter()
        .zip(final_conv.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max);

    assert!(
        context_diff <= 3.0e-3,
        "expected shell-fused token-step context parity, max diff {context_diff}"
    );
    assert!(
        ssm_diff <= 3.0e-3,
        "expected shell-fused token-step ssm parity, max diff {ssm_diff}"
    );
    assert!(
        conv_diff <= 1.0e-4,
        "expected shell-fused token-step conv parity, max diff {conv_diff}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn mamba2_cuda_default_training_path_matches_direct_graph_gradients_on_shakespeare_like_shape() {
    type CudaBackend = Autodiff<Cuda<f32, i32>>;

    let device = <CudaBackend as Backend>::Device::default();
    let config = shakespeare_like_mamba2_config()
        .resolve(128, SequenceMemorySystem::Mamba2StateSpaceDuality);
    let batch = 1;
    let time = 32;
    let d_model = 128;

    let hidden_data = TensorData::new(
        (0..(batch * time * d_model))
            .map(|idx| ((idx % 257) as f32) / 257.0 - 0.5)
            .collect::<Vec<_>>(),
        [batch, 1, time, d_model],
    );
    let in_proj_data = TensorData::new(
        (0..(d_model * config.mamba2_in_proj_dim()))
            .map(|idx| ((idx % 263) as f32) / 263.0 - 0.5)
            .collect::<Vec<_>>(),
        [d_model, config.mamba2_in_proj_dim()],
    );
    let conv_weight_data = TensorData::new(
        (0..(config.mamba2_conv_dim() * config.d_conv))
            .map(|idx| ((idx % 269) as f32) / 269.0 - 0.5)
            .collect::<Vec<_>>(),
        [config.mamba2_conv_dim(), config.d_conv],
    );
    let conv_bias_data = TensorData::new(
        (0..config.mamba2_conv_dim())
            .map(|idx| ((idx % 271) as f32) / 271.0 - 0.5)
            .collect::<Vec<_>>(),
        [config.mamba2_conv_dim()],
    );
    let dt_bias_data = TensorData::new(
        (0..config.nheads)
            .map(|idx| ((idx % 277) as f32) / 277.0 - 0.5)
            .collect::<Vec<_>>(),
        [config.nheads],
    );
    let a_log_data = TensorData::new(
        (0..config.nheads)
            .map(|idx| ((idx % 281) as f32) / 281.0 - 0.5)
            .collect::<Vec<_>>(),
        [config.nheads],
    );
    let d_skip_data = TensorData::new(
        (0..config.nheads)
            .map(|idx| ((idx % 283) as f32) / 283.0 - 0.5)
            .collect::<Vec<_>>(),
        [config.nheads],
    );
    let norm_weight_data = TensorData::new(
        (0..config.d_inner)
            .map(|idx| ((idx % 293) as f32) / 293.0 - 0.5)
            .collect::<Vec<_>>(),
        [config.d_inner],
    );
    let out_proj_data = TensorData::new(
        (0..(config.d_inner * d_model))
            .map(|idx| ((idx % 307) as f32) / 307.0 - 0.5)
            .collect::<Vec<_>>(),
        [config.d_inner, d_model],
    );
    let output_weight_data = TensorData::new(
        (0..(batch * time * d_model))
            .map(|idx| ((idx % 311) as f32) / 311.0 - 0.5)
            .collect::<Vec<_>>(),
        [batch, 1, time, d_model],
    );

    let hidden_graph =
        Tensor::<CudaBackend, 4>::from_data(hidden_data.clone(), &device).require_grad();
    let in_proj_graph =
        Tensor::<CudaBackend, 2>::from_data(in_proj_data.clone(), &device).require_grad();
    let conv_weight_graph =
        Tensor::<CudaBackend, 2>::from_data(conv_weight_data.clone(), &device).require_grad();
    let conv_bias_graph =
        Tensor::<CudaBackend, 1>::from_data(conv_bias_data.clone(), &device).require_grad();
    let dt_bias_graph =
        Tensor::<CudaBackend, 1>::from_data(dt_bias_data.clone(), &device).require_grad();
    let a_log_graph =
        Tensor::<CudaBackend, 1>::from_data(a_log_data.clone(), &device).require_grad();
    let d_skip_graph =
        Tensor::<CudaBackend, 1>::from_data(d_skip_data.clone(), &device).require_grad();
    let norm_weight_graph =
        Tensor::<CudaBackend, 1>::from_data(norm_weight_data.clone(), &device).require_grad();
    let out_proj_graph =
        Tensor::<CudaBackend, 2>::from_data(out_proj_data.clone(), &device).require_grad();

    let hidden_fused = Tensor::<CudaBackend, 4>::from_data(hidden_data, &device).require_grad();
    let in_proj_fused = Tensor::<CudaBackend, 2>::from_data(in_proj_data, &device).require_grad();
    let conv_weight_fused =
        Tensor::<CudaBackend, 2>::from_data(conv_weight_data, &device).require_grad();
    let conv_bias_fused =
        Tensor::<CudaBackend, 1>::from_data(conv_bias_data, &device).require_grad();
    let dt_bias_fused = Tensor::<CudaBackend, 1>::from_data(dt_bias_data, &device).require_grad();
    let a_log_fused = Tensor::<CudaBackend, 1>::from_data(a_log_data, &device).require_grad();
    let d_skip_fused = Tensor::<CudaBackend, 1>::from_data(d_skip_data, &device).require_grad();
    let norm_weight_fused =
        Tensor::<CudaBackend, 1>::from_data(norm_weight_data, &device).require_grad();
    let out_proj_fused = Tensor::<CudaBackend, 2>::from_data(out_proj_data, &device).require_grad();

    let graph = burn_dragon_kernel::kernels::sequence::mamba2::forward::tensorized_mamba2_forward_direct_graph(
        hidden_graph.clone(),
        config.d_inner,
        config.d_state,
        config.d_conv,
        config.headdim,
        config.ngroups,
        in_proj_graph.clone(),
        conv_weight_graph.clone(),
        Some(conv_bias_graph.clone()),
        dt_bias_graph.clone(),
        a_log_graph.clone(),
        d_skip_graph.clone(),
        norm_weight_graph.clone(),
        config.norm_eps,
        out_proj_graph.clone(),
        None::<Mamba2TensorizedState<CudaBackend>>,
    );
    let shell_fused = tensorized_mamba2_forward_custom_backward_with_cuda_modes(
        hidden_fused.clone(),
        config.d_inner,
        config.d_state,
        config.d_conv,
        config.headdim,
        config.ngroups,
        in_proj_fused.clone(),
        conv_weight_fused.clone(),
        Some(conv_bias_fused.clone()),
        dt_bias_fused.clone(),
        a_log_fused.clone(),
        d_skip_fused.clone(),
        norm_weight_fused.clone(),
        config.norm_eps,
        out_proj_fused.clone(),
        None::<Mamba2TensorizedState<CudaBackend>>,
        CudaSsdCoreMode::ForcedDisabled,
        CudaShellCoreMode::ForcedEnabled,
    )
    .expect("cuda shell-fused mamba2 output");

    let _ = <CudaBackend as Backend>::sync(&device);
    let output_diff =
        tensor_max_abs_diff_backend(graph.context.clone(), shell_fused.context.clone());
    let conv_state_diff =
        tensor_max_abs_diff_backend(graph.state.conv.clone(), shell_fused.state.conv.clone());
    let ssm_state_diff =
        tensor_max_abs_diff_backend(graph.state.ssm.clone(), shell_fused.state.ssm.clone());

    let output_weights = Tensor::<CudaBackend, 4>::from_data(output_weight_data, &device);
    let graph_grads = (graph.context * output_weights.clone()).sum().backward();
    let shell_fused_grads = (shell_fused.context * output_weights).sum().backward();
    let _ = <CudaBackend as Backend>::sync(&device);

    let hidden_grad_diff = tensor_max_abs_diff_backend(
        hidden_graph.grad(&graph_grads).expect("graph hidden grad"),
        hidden_fused
            .grad(&shell_fused_grads)
            .expect("shell-fused hidden grad"),
    );
    let in_proj_grad_diff = tensor_max_abs_diff_backend(
        in_proj_graph
            .grad(&graph_grads)
            .expect("graph in_proj grad"),
        in_proj_fused
            .grad(&shell_fused_grads)
            .expect("shell-fused in_proj grad"),
    );
    let conv_weight_grad_diff = tensor_max_abs_diff_backend(
        conv_weight_graph
            .grad(&graph_grads)
            .expect("graph conv weight grad"),
        conv_weight_fused
            .grad(&shell_fused_grads)
            .expect("shell-fused conv weight grad"),
    );
    let conv_bias_grad_diff = tensor_max_abs_diff_backend(
        conv_bias_graph
            .grad(&graph_grads)
            .expect("graph conv bias grad"),
        conv_bias_fused
            .grad(&shell_fused_grads)
            .expect("shell-fused conv bias grad"),
    );
    let dt_bias_grad_diff = tensor_max_abs_diff_backend(
        dt_bias_graph
            .grad(&graph_grads)
            .expect("graph dt bias grad"),
        dt_bias_fused
            .grad(&shell_fused_grads)
            .expect("shell-fused dt bias grad"),
    );
    let a_log_grad_diff = tensor_max_abs_diff_backend(
        a_log_graph.grad(&graph_grads).expect("graph a_log grad"),
        a_log_fused
            .grad(&shell_fused_grads)
            .expect("shell-fused a_log grad"),
    );
    let d_skip_grad_diff = tensor_max_abs_diff_backend(
        d_skip_graph.grad(&graph_grads).expect("graph d_skip grad"),
        d_skip_fused
            .grad(&shell_fused_grads)
            .expect("shell-fused d_skip grad"),
    );
    let norm_weight_grad_diff = tensor_max_abs_diff_backend(
        norm_weight_graph
            .grad(&graph_grads)
            .expect("graph norm weight grad"),
        norm_weight_fused
            .grad(&shell_fused_grads)
            .expect("shell-fused norm weight grad"),
    );
    let out_proj_grad_diff = tensor_max_abs_diff_backend(
        out_proj_graph
            .grad(&graph_grads)
            .expect("graph out_proj grad"),
        out_proj_fused
            .grad(&shell_fused_grads)
            .expect("shell-fused out_proj grad"),
    );

    eprintln!(
        "mamba2 realistic cuda parity: output_diff={output_diff} conv_state_diff={conv_state_diff} ssm_state_diff={ssm_state_diff} hidden_grad_diff={hidden_grad_diff} in_proj_grad_diff={in_proj_grad_diff} conv_weight_grad_diff={conv_weight_grad_diff} conv_bias_grad_diff={conv_bias_grad_diff} dt_bias_grad_diff={dt_bias_grad_diff} a_log_grad_diff={a_log_grad_diff} d_skip_grad_diff={d_skip_grad_diff} norm_weight_grad_diff={norm_weight_grad_diff} out_proj_grad_diff={out_proj_grad_diff}"
    );

    assert!(
        output_diff <= 5.0e-3,
        "expected shell-fused output parity on realistic shape, max diff {output_diff}"
    );
    assert!(
        conv_state_diff <= 5.0e-3,
        "expected shell-fused conv state parity on realistic shape, max diff {conv_state_diff}"
    );
    assert!(
        ssm_state_diff <= 5.0e-3,
        "expected shell-fused ssm state parity on realistic shape, max diff {ssm_state_diff}"
    );
    assert!(
        hidden_grad_diff <= 5.0e-3,
        "expected shell-fused hidden grad parity on realistic shape, max diff {hidden_grad_diff}"
    );
    assert!(
        in_proj_grad_diff <= 5.0e-3,
        "expected shell-fused in_proj grad parity on realistic shape, max diff {in_proj_grad_diff}"
    );
    assert!(
        conv_weight_grad_diff <= 5.0e-3,
        "expected shell-fused conv_weight grad parity on realistic shape, max diff {conv_weight_grad_diff}"
    );
    assert!(
        conv_bias_grad_diff <= 5.0e-3,
        "expected shell-fused conv_bias grad parity on realistic shape, max diff {conv_bias_grad_diff}"
    );
    assert!(
        dt_bias_grad_diff <= 5.0e-3,
        "expected shell-fused dt_bias grad parity on realistic shape, max diff {dt_bias_grad_diff}"
    );
    assert!(
        a_log_grad_diff <= 5.0e-3,
        "expected shell-fused a_log grad parity on realistic shape, max diff {a_log_grad_diff}"
    );
    assert!(
        d_skip_grad_diff <= 5.0e-3,
        "expected shell-fused d_skip grad parity on realistic shape, max diff {d_skip_grad_diff}"
    );
    assert!(
        norm_weight_grad_diff <= 5.0e-3,
        "expected shell-fused norm_weight grad parity on realistic shape, max diff {norm_weight_grad_diff}"
    );
    assert!(
        out_proj_grad_diff <= 5.0e-3,
        "expected shell-fused out_proj grad parity on realistic shape, max diff {out_proj_grad_diff}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn mamba2_cuda_fused_ssd_training_path_matches_direct_graph_gradients_on_shakespeare_like_shape() {
    type CudaBackend = Autodiff<Cuda<f32, i32>>;

    let device = <CudaBackend as Backend>::Device::default();
    let config = shakespeare_like_mamba2_config()
        .resolve(128, SequenceMemorySystem::Mamba2StateSpaceDuality);
    let batch = 1;
    let time = 32;
    let d_model = 128;

    let hidden_data = TensorData::new(
        (0..(batch * time * d_model))
            .map(|idx| ((idx % 257) as f32) / 257.0 - 0.5)
            .collect::<Vec<_>>(),
        [batch, 1, time, d_model],
    );
    let in_proj_data = TensorData::new(
        (0..(d_model * config.mamba2_in_proj_dim()))
            .map(|idx| ((idx % 263) as f32) / 263.0 - 0.5)
            .collect::<Vec<_>>(),
        [d_model, config.mamba2_in_proj_dim()],
    );
    let conv_weight_data = TensorData::new(
        (0..(config.mamba2_conv_dim() * config.d_conv))
            .map(|idx| ((idx % 269) as f32) / 269.0 - 0.5)
            .collect::<Vec<_>>(),
        [config.mamba2_conv_dim(), config.d_conv],
    );
    let conv_bias_data = TensorData::new(
        (0..config.mamba2_conv_dim())
            .map(|idx| ((idx % 271) as f32) / 271.0 - 0.5)
            .collect::<Vec<_>>(),
        [config.mamba2_conv_dim()],
    );
    let dt_bias_data = TensorData::new(
        (0..config.nheads)
            .map(|idx| ((idx % 277) as f32) / 277.0 - 0.5)
            .collect::<Vec<_>>(),
        [config.nheads],
    );
    let a_log_data = TensorData::new(
        (0..config.nheads)
            .map(|idx| ((idx % 281) as f32) / 281.0 - 0.5)
            .collect::<Vec<_>>(),
        [config.nheads],
    );
    let d_skip_data = TensorData::new(
        (0..config.nheads)
            .map(|idx| ((idx % 283) as f32) / 283.0 - 0.5)
            .collect::<Vec<_>>(),
        [config.nheads],
    );
    let norm_weight_data = TensorData::new(
        (0..config.d_inner)
            .map(|idx| ((idx % 293) as f32) / 293.0 - 0.5)
            .collect::<Vec<_>>(),
        [config.d_inner],
    );
    let out_proj_data = TensorData::new(
        (0..(config.d_inner * d_model))
            .map(|idx| ((idx % 307) as f32) / 307.0 - 0.5)
            .collect::<Vec<_>>(),
        [config.d_inner, d_model],
    );
    let output_weight_data = TensorData::new(
        (0..(batch * time * d_model))
            .map(|idx| ((idx % 311) as f32) / 311.0 - 0.5)
            .collect::<Vec<_>>(),
        [batch, 1, time, d_model],
    );

    let hidden_graph =
        Tensor::<CudaBackend, 4>::from_data(hidden_data.clone(), &device).require_grad();
    let in_proj_graph =
        Tensor::<CudaBackend, 2>::from_data(in_proj_data.clone(), &device).require_grad();
    let conv_weight_graph =
        Tensor::<CudaBackend, 2>::from_data(conv_weight_data.clone(), &device).require_grad();
    let conv_bias_graph =
        Tensor::<CudaBackend, 1>::from_data(conv_bias_data.clone(), &device).require_grad();
    let dt_bias_graph =
        Tensor::<CudaBackend, 1>::from_data(dt_bias_data.clone(), &device).require_grad();
    let a_log_graph =
        Tensor::<CudaBackend, 1>::from_data(a_log_data.clone(), &device).require_grad();
    let d_skip_graph =
        Tensor::<CudaBackend, 1>::from_data(d_skip_data.clone(), &device).require_grad();
    let norm_weight_graph =
        Tensor::<CudaBackend, 1>::from_data(norm_weight_data.clone(), &device).require_grad();
    let out_proj_graph =
        Tensor::<CudaBackend, 2>::from_data(out_proj_data.clone(), &device).require_grad();

    let hidden_fused = Tensor::<CudaBackend, 4>::from_data(hidden_data, &device).require_grad();
    let in_proj_fused = Tensor::<CudaBackend, 2>::from_data(in_proj_data, &device).require_grad();
    let conv_weight_fused =
        Tensor::<CudaBackend, 2>::from_data(conv_weight_data, &device).require_grad();
    let conv_bias_fused =
        Tensor::<CudaBackend, 1>::from_data(conv_bias_data, &device).require_grad();
    let dt_bias_fused = Tensor::<CudaBackend, 1>::from_data(dt_bias_data, &device).require_grad();
    let a_log_fused = Tensor::<CudaBackend, 1>::from_data(a_log_data, &device).require_grad();
    let d_skip_fused = Tensor::<CudaBackend, 1>::from_data(d_skip_data, &device).require_grad();
    let norm_weight_fused =
        Tensor::<CudaBackend, 1>::from_data(norm_weight_data, &device).require_grad();
    let out_proj_fused = Tensor::<CudaBackend, 2>::from_data(out_proj_data, &device).require_grad();

    let graph = burn_dragon_kernel::kernels::sequence::mamba2::forward::tensorized_mamba2_forward_direct_graph(
        hidden_graph.clone(),
        config.d_inner,
        config.d_state,
        config.d_conv,
        config.headdim,
        config.ngroups,
        in_proj_graph.clone(),
        conv_weight_graph.clone(),
        Some(conv_bias_graph.clone()),
        dt_bias_graph.clone(),
        a_log_graph.clone(),
        d_skip_graph.clone(),
        norm_weight_graph.clone(),
        config.norm_eps,
        out_proj_graph.clone(),
        None::<Mamba2TensorizedState<CudaBackend>>,
    );
    let full_fused = tensorized_mamba2_forward_custom_backward_with_cuda_modes(
        hidden_fused.clone(),
        config.d_inner,
        config.d_state,
        config.d_conv,
        config.headdim,
        config.ngroups,
        in_proj_fused.clone(),
        conv_weight_fused.clone(),
        Some(conv_bias_fused.clone()),
        dt_bias_fused.clone(),
        a_log_fused.clone(),
        d_skip_fused.clone(),
        norm_weight_fused.clone(),
        config.norm_eps,
        out_proj_fused.clone(),
        None::<Mamba2TensorizedState<CudaBackend>>,
        CudaSsdCoreMode::ForcedEnabled,
        CudaShellCoreMode::ForcedEnabled,
    )
    .expect("cuda full-fused mamba2 output");

    let _ = <CudaBackend as Backend>::sync(&device);
    let output_diff =
        tensor_max_abs_diff_backend(graph.context.clone(), full_fused.context.clone());
    let conv_state_diff =
        tensor_max_abs_diff_backend(graph.state.conv.clone(), full_fused.state.conv.clone());
    let ssm_state_diff =
        tensor_max_abs_diff_backend(graph.state.ssm.clone(), full_fused.state.ssm.clone());

    let output_weights = Tensor::<CudaBackend, 4>::from_data(output_weight_data, &device);
    let graph_grads = (graph.context * output_weights.clone()).sum().backward();
    let full_fused_grads = (full_fused.context * output_weights).sum().backward();
    let _ = <CudaBackend as Backend>::sync(&device);

    let hidden_grad_diff = tensor_max_abs_diff_backend(
        hidden_graph.grad(&graph_grads).expect("graph hidden grad"),
        hidden_fused
            .grad(&full_fused_grads)
            .expect("full-fused hidden grad"),
    );
    let in_proj_grad_diff = tensor_max_abs_diff_backend(
        in_proj_graph
            .grad(&graph_grads)
            .expect("graph in_proj grad"),
        in_proj_fused
            .grad(&full_fused_grads)
            .expect("full-fused in_proj grad"),
    );
    let conv_weight_grad_diff = tensor_max_abs_diff_backend(
        conv_weight_graph
            .grad(&graph_grads)
            .expect("graph conv weight grad"),
        conv_weight_fused
            .grad(&full_fused_grads)
            .expect("full-fused conv weight grad"),
    );
    let conv_bias_grad_diff = tensor_max_abs_diff_backend(
        conv_bias_graph
            .grad(&graph_grads)
            .expect("graph conv bias grad"),
        conv_bias_fused
            .grad(&full_fused_grads)
            .expect("full-fused conv bias grad"),
    );
    let dt_bias_grad_diff = tensor_max_abs_diff_backend(
        dt_bias_graph
            .grad(&graph_grads)
            .expect("graph dt bias grad"),
        dt_bias_fused
            .grad(&full_fused_grads)
            .expect("full-fused dt bias grad"),
    );
    let a_log_grad_diff = tensor_max_abs_diff_backend(
        a_log_graph.grad(&graph_grads).expect("graph a_log grad"),
        a_log_fused
            .grad(&full_fused_grads)
            .expect("full-fused a_log grad"),
    );
    let d_skip_grad_diff = tensor_max_abs_diff_backend(
        d_skip_graph.grad(&graph_grads).expect("graph d_skip grad"),
        d_skip_fused
            .grad(&full_fused_grads)
            .expect("full-fused d_skip grad"),
    );
    let norm_weight_grad_diff = tensor_max_abs_diff_backend(
        norm_weight_graph
            .grad(&graph_grads)
            .expect("graph norm weight grad"),
        norm_weight_fused
            .grad(&full_fused_grads)
            .expect("full-fused norm weight grad"),
    );
    let out_proj_grad_diff = tensor_max_abs_diff_backend(
        out_proj_graph
            .grad(&graph_grads)
            .expect("graph out_proj grad"),
        out_proj_fused
            .grad(&full_fused_grads)
            .expect("full-fused out_proj grad"),
    );

    eprintln!(
        "mamba2 realistic cuda fused parity: output_diff={output_diff} conv_state_diff={conv_state_diff} ssm_state_diff={ssm_state_diff} hidden_grad_diff={hidden_grad_diff} in_proj_grad_diff={in_proj_grad_diff} conv_weight_grad_diff={conv_weight_grad_diff} conv_bias_grad_diff={conv_bias_grad_diff} dt_bias_grad_diff={dt_bias_grad_diff} a_log_grad_diff={a_log_grad_diff} d_skip_grad_diff={d_skip_grad_diff} norm_weight_grad_diff={norm_weight_grad_diff} out_proj_grad_diff={out_proj_grad_diff}"
    );

    assert!(
        output_diff <= 5.0e-3,
        "expected full-fused output parity on realistic shape, max diff {output_diff}"
    );
    assert!(
        conv_state_diff <= 5.0e-3,
        "expected full-fused conv state parity on realistic shape, max diff {conv_state_diff}"
    );
    assert!(
        ssm_state_diff <= 5.0e-3,
        "expected full-fused ssm state parity on realistic shape, max diff {ssm_state_diff}"
    );
    assert!(
        hidden_grad_diff <= 5.0e-3,
        "expected full-fused hidden grad parity on realistic shape, max diff {hidden_grad_diff}"
    );
    assert!(
        in_proj_grad_diff <= 5.0e-3,
        "expected full-fused in_proj grad parity on realistic shape, max diff {in_proj_grad_diff}"
    );
    assert!(
        conv_weight_grad_diff <= 5.0e-3,
        "expected full-fused conv_weight grad parity on realistic shape, max diff {conv_weight_grad_diff}"
    );
    assert!(
        conv_bias_grad_diff <= 5.0e-3,
        "expected full-fused conv_bias grad parity on realistic shape, max diff {conv_bias_grad_diff}"
    );
    assert!(
        dt_bias_grad_diff <= 5.0e-3,
        "expected full-fused dt_bias grad parity on realistic shape, max diff {dt_bias_grad_diff}"
    );
    assert!(
        a_log_grad_diff <= 5.0e-3,
        "expected full-fused a_log grad parity on realistic shape, max diff {a_log_grad_diff}"
    );
    assert!(
        d_skip_grad_diff <= 5.0e-3,
        "expected full-fused d_skip grad parity on realistic shape, max diff {d_skip_grad_diff}"
    );
    assert!(
        norm_weight_grad_diff <= 5.0e-3,
        "expected full-fused norm_weight grad parity on realistic shape, max diff {norm_weight_grad_diff}"
    );
    assert!(
        out_proj_grad_diff <= 5.0e-3,
        "expected full-fused out_proj grad parity on realistic shape, max diff {out_proj_grad_diff}"
    );
}
