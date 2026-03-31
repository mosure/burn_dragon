use std::sync::{Mutex, OnceLock};

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
#[cfg(feature = "cuda")]
use burn_autodiff::Autodiff;
#[cfg(feature = "cuda")]
use burn_cuda::Cuda;
use burn_dragon_core::{
    BDH, BDHConfig, FusedKernelConfig, SequenceKernelConfig, SequenceMemorySystem,
};
use burn_dragon_kernel::kernels::sequence::rwkv8::forward::{
    tensorized_rwkv8_forward, tensorized_rwkv8_forward_direct_graph,
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

fn with_rwkv8_env<T>(
    forward: Option<&str>,
    chunk: Option<&str>,
    wrapper: Option<&str>,
    f: impl FnOnce() -> T,
) -> T {
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _guard = ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("env lock");
    let previous_forward = std::env::var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD").ok();
    let previous_chunk = std::env::var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_CHUNK").ok();
    let previous_wrapper = std::env::var("BURN_DRAGON_RWKV8_TENSORIZED_TRAIN_WRAPPER").ok();

    match forward {
        Some(value) => unsafe { std::env::set_var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD", value) },
        None => unsafe { std::env::remove_var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD") },
    }
    match chunk {
        Some(value) => unsafe {
            std::env::set_var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_CHUNK", value)
        },
        None => unsafe { std::env::remove_var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_CHUNK") },
    }
    match wrapper {
        Some(value) => unsafe {
            std::env::set_var("BURN_DRAGON_RWKV8_TENSORIZED_TRAIN_WRAPPER", value)
        },
        None => unsafe { std::env::remove_var("BURN_DRAGON_RWKV8_TENSORIZED_TRAIN_WRAPPER") },
    }

    let output = f();

    match previous_forward {
        Some(value) => unsafe { std::env::set_var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD", value) },
        None => unsafe { std::env::remove_var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD") },
    }
    match previous_chunk {
        Some(value) => unsafe {
            std::env::set_var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_CHUNK", value)
        },
        None => unsafe { std::env::remove_var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_CHUNK") },
    }
    match previous_wrapper {
        Some(value) => unsafe {
            std::env::set_var("BURN_DRAGON_RWKV8_TENSORIZED_TRAIN_WRAPPER", value)
        },
        None => unsafe { std::env::remove_var("BURN_DRAGON_RWKV8_TENSORIZED_TRAIN_WRAPPER") },
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

fn make_tokens(
    device: &<TestBackend as Backend>::Device,
    token_values: Vec<i64>,
    shape: [usize; 2],
) -> Tensor<TestBackend, 2, Int> {
    Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(token_values, shape), device)
}

fn shakespeare_like_rwkv8_config() -> BDHConfig {
    BDHConfig {
        n_layer: 4,
        n_embd: 128,
        n_head: 4,
        mlp_internal_dim_multiplier: 4,
        vocab_size: 69,
        dropout: 0.0,
        sequence_kernel: SequenceKernelConfig::reference(SequenceMemorySystem::Rwkv8StateSpace),
        fused_kernels: FusedKernelConfig {
            enabled: true,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn rwkv8_bdh_full_forward_matches_token_step_recurrence_on_shakespeare_like_shape() {
    let device = <TestBackend as Backend>::Device::default();
    init_runtime(&device);
    let tokens = make_tokens(
        &device,
        (0..64).map(|idx| (idx % 32) as i64).collect::<Vec<_>>(),
        [1, 64],
    );
    let model = init_seeded_model(shakespeare_like_rwkv8_config());

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
        max_diff <= 1.0e-4,
        "expected rwkv8 BDH full forward and token-step recurrence to match, max diff {max_diff}"
    );
}

#[test]
fn rwkv8_bdh_chunked_recurrence_matches_full_forward_on_shakespeare_like_shape() {
    let device = <TestBackend as Backend>::Device::default();
    init_runtime(&device);
    let tokens = make_tokens(
        &device,
        (0..64).map(|idx| (idx % 32) as i64).collect::<Vec<_>>(),
        [1, 64],
    );
    let model = init_seeded_model(shakespeare_like_rwkv8_config());

    let logits_full = model.forward(tokens.clone());
    let mut recurrent_state = model.init_state();
    let splits = [0usize, 19, 41, 64];
    let mut chunked_logits = Vec::new();
    for window in splits.windows(2) {
        chunked_logits.push(model.forward_with_state(
            tokens.clone().slice_dim(1, window[0]..window[1]),
            &mut recurrent_state,
        ));
    }
    let logits_chunked = Tensor::cat(chunked_logits, 1);
    let max_diff = tensor_max_abs_diff(logits_full, logits_chunked);
    assert!(
        max_diff <= 1.0e-4,
        "expected rwkv8 BDH chunked recurrence and full forward to match, max diff {max_diff}"
    );
}

#[test]
fn rwkv8_bdh_chunk_size_sweep_matches_reference_logits_on_shakespeare_like_shape() {
    let device = <TestBackend as Backend>::Device::default();
    init_runtime(&device);
    let tokens = make_tokens(
        &device,
        (0..256).map(|idx| (idx % 64) as i64).collect::<Vec<_>>(),
        [1, 256],
    );
    let model = init_seeded_model(shakespeare_like_rwkv8_config());

    let reference_logits = with_rwkv8_env(Some("0"), None, None, || model.forward(tokens.clone()));
    for chunk in [8usize, 16, 32, 64, 128, 256] {
        let logits = with_rwkv8_env(Some("1"), Some(&chunk.to_string()), Some("0"), || {
            model.forward(tokens.clone())
        });
        let max_diff = tensor_max_abs_diff(reference_logits.clone(), logits);
        assert!(
            max_diff <= 2.0e-4,
            "expected rwkv8 chunk-size parity for chunk {chunk}, max diff {max_diff}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn rwkv8_cuda_default_training_path_matches_direct_graph_gradients_on_shakespeare_like_shape() {
    type CudaBackend = Autodiff<Cuda<f32, i32>>;

    let device = <CudaBackend as Backend>::Device::default();
    let batch = 1;
    let heads = 4;
    let time = 64;
    let latent = 128;
    let embd = 128;

    let query_data = TensorData::new(
        (0..(batch * heads * time * latent))
            .map(|idx| ((idx % 257) as f32) / 257.0 + 0.1)
            .collect::<Vec<_>>(),
        [batch, heads, time, latent],
    );
    let value_data = TensorData::new(
        (0..(batch * time * embd))
            .map(|idx| ((idx % 263) as f32) / 263.0 - 0.5)
            .collect::<Vec<_>>(),
        [batch, 1, time, embd],
    );
    let rho_state_data = TensorData::new(
        (0..(batch * heads * latent * embd))
            .map(|idx| ((idx % 269) as f32) / 269.0 - 0.5)
            .collect::<Vec<_>>(),
        [batch, heads, latent, embd],
    );
    let rho_norm_state_data = TensorData::new(
        (0..(batch * heads * latent))
            .map(|idx| ((idx % 271) as f32) / 271.0 + 0.2)
            .collect::<Vec<_>>(),
        [batch, heads, latent],
    );
    let decay_data = TensorData::new(
        (0..(heads * latent))
            .map(|idx| 0.85 + ((idx % 31) as f32) / 310.0)
            .collect::<Vec<_>>(),
        [1, heads, latent],
    );
    let output_weight_data = TensorData::new(
        (0..(batch * heads * time * embd))
            .map(|idx| ((idx % 277) as f32) / 277.0 - 0.5)
            .collect::<Vec<_>>(),
        [batch, heads, time, embd],
    );

    let query_graph =
        Tensor::<CudaBackend, 4>::from_data(query_data.clone(), &device).require_grad();
    let value_graph =
        Tensor::<CudaBackend, 4>::from_data(value_data.clone(), &device).require_grad();
    let rho_graph = Tensor::<CudaBackend, 4>::from_data(rho_state_data.clone(), &device);
    let rho_norm_graph = Tensor::<CudaBackend, 3>::from_data(rho_norm_state_data.clone(), &device);
    let decay_graph =
        Tensor::<CudaBackend, 3>::from_data(decay_data.clone(), &device).require_grad();

    let query_default = Tensor::<CudaBackend, 4>::from_data(query_data, &device).require_grad();
    let value_default = Tensor::<CudaBackend, 4>::from_data(value_data, &device).require_grad();
    let rho_default = Tensor::<CudaBackend, 4>::from_data(rho_state_data, &device);
    let rho_norm_default = Tensor::<CudaBackend, 3>::from_data(rho_norm_state_data, &device);
    let decay_default = Tensor::<CudaBackend, 3>::from_data(decay_data, &device).require_grad();

    let graph = tensorized_rwkv8_forward_direct_graph(
        query_graph.clone(),
        value_graph.clone(),
        Some(rho_graph),
        Some(rho_norm_graph),
        decay_graph.clone(),
    );
    let default = with_rwkv8_env(Some("1"), None, Some("1"), || {
        tensorized_rwkv8_forward(
            query_default.clone(),
            value_default.clone(),
            Some(rho_default.clone()),
            Some(rho_norm_default.clone()),
            decay_default.clone(),
        )
    });

    let _ = <CudaBackend as Backend>::sync(&device);
    let context_diff = tensor_max_abs_diff_backend(graph.context.clone(), default.context.clone());
    let rho_diff = tensor_max_abs_diff_backend(graph.rho.clone(), default.rho.clone());
    let rho_norm_diff =
        tensor_max_abs_diff_backend(graph.rho_norm.clone(), default.rho_norm.clone());

    let output_weights = Tensor::<CudaBackend, 4>::from_data(output_weight_data, &device);
    let graph_grads = (graph.context * output_weights.clone()).sum().backward();
    let default_grads = (default.context * output_weights).sum().backward();
    let _ = <CudaBackend as Backend>::sync(&device);

    let query_grad_diff = tensor_max_abs_diff_backend(
        query_graph.grad(&graph_grads).expect("graph query grad"),
        query_default
            .grad(&default_grads)
            .expect("default query grad"),
    );
    let value_grad_diff = tensor_max_abs_diff_backend(
        value_graph.grad(&graph_grads).expect("graph value grad"),
        value_default
            .grad(&default_grads)
            .expect("default value grad"),
    );
    let decay_grad_diff = tensor_max_abs_diff_backend(
        decay_graph.grad(&graph_grads).expect("graph decay grad"),
        decay_default
            .grad(&default_grads)
            .expect("default decay grad"),
    );

    println!(
        "rwkv8 realistic cuda parity: context_diff={context_diff} rho_diff={rho_diff} rho_norm_diff={rho_norm_diff} query_grad_diff={query_grad_diff} value_grad_diff={value_grad_diff} decay_grad_diff={decay_grad_diff}"
    );

    assert!(context_diff <= 3.0e-4, "context diff {context_diff}");
    assert!(rho_diff <= 3.0e-4, "rho diff {rho_diff}");
    assert!(rho_norm_diff <= 3.0e-4, "rho_norm diff {rho_norm_diff}");
    assert!(
        query_grad_diff <= 6.0e-4,
        "query grad diff {query_grad_diff}"
    );
    assert!(
        value_grad_diff <= 6.0e-4,
        "value grad diff {value_grad_diff}"
    );
    assert!(
        decay_grad_diff <= 6.0e-4,
        "decay grad diff {decay_grad_diff}"
    );
}
