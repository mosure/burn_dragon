use super::forward::{
    rwkv8_tensorized_chunk_size, tensorized_rwkv8_forward, tensorized_rwkv8_forward_direct_graph,
};
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Tensor, TensorData};
use burn_autodiff::Autodiff;
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};

type WgpuBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type WgpuAutodiffBackend = Autodiff<WgpuBackend>;
#[cfg(feature = "cuda")]
type CudaBackend = CubeBackend<CudaRuntime, f32, i32, u8>;
#[cfg(feature = "cuda")]
type CudaAutodiffBackend = Autodiff<CudaBackend>;

struct EnvVarGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.previous {
            Some(previous) => unsafe { std::env::set_var(self.key, previous) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

fn init_wgpu_runtime(device: &<WgpuBackend as BackendTrait>::Device) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

fn assert_close_backend<B: BackendTrait, const D: usize>(
    lhs: Tensor<B, D>,
    rhs: Tensor<B, D>,
    atol: f32,
    rtol: f32,
) {
    let lhs_data = lhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("lhs vec");
    let rhs_data = rhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rhs vec");
    let mut max_diff = 0.0_f32;
    let mut max_tol = 0.0_f32;
    for (a, b) in lhs_data.iter().zip(rhs_data.iter()) {
        let diff = (a - b).abs();
        let tol = atol + rtol * b.abs();
        if diff > max_diff {
            max_diff = diff;
            max_tol = tol;
        }
    }
    assert!(
        max_diff <= max_tol,
        "max difference {max_diff} exceeds tolerance {max_tol}"
    );
}

fn parity_inputs<B: BackendTrait>(
    device: &B::Device,
) -> (
    Tensor<B, 4>,
    Tensor<B, 4>,
    Tensor<B, 4>,
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 4>,
) {
    let batch = 1;
    let heads = 2;
    let time = 5;
    let latent = 4;
    let embd = 6;

    let query = Tensor::<B, 4>::from_data(
        TensorData::new(
            (0..(batch * heads * time * latent))
                .map(|idx| 0.15 + (idx as f32) * 0.013)
                .collect::<Vec<_>>(),
            [batch, heads, time, latent],
        ),
        device,
    );
    let value = Tensor::<B, 4>::from_data(
        TensorData::new(
            (0..(batch * time * embd))
                .map(|idx| -0.2 + (idx as f32) * 0.017)
                .collect::<Vec<_>>(),
            [batch, 1, time, embd],
        ),
        device,
    );
    let rho_state = Tensor::<B, 4>::from_data(
        TensorData::new(
            (0..(batch * heads * latent * embd))
                .map(|idx| -0.1 + (idx as f32) * 0.009)
                .collect::<Vec<_>>(),
            [batch, heads, latent, embd],
        ),
        device,
    );
    let rho_norm_state = Tensor::<B, 3>::from_data(
        TensorData::new(
            (0..(batch * heads * latent))
                .map(|idx| 0.4 + (idx as f32) * 0.021)
                .collect::<Vec<_>>(),
            [batch, heads, latent],
        ),
        device,
    );
    let decay = Tensor::<B, 3>::from_data(
        TensorData::new(
            vec![0.97, 0.95, 0.93, 0.91, 0.96, 0.94, 0.92, 0.90],
            [1, heads, latent],
        ),
        device,
    );
    let weights = Tensor::<B, 4>::from_data(
        TensorData::new(
            (0..(batch * heads * time * embd))
                .map(|idx| -0.05 + (idx as f32) * 0.011)
                .collect::<Vec<_>>(),
            [batch, heads, time, embd],
        ),
        device,
    );
    (query, value, rho_state, rho_norm_state, decay, weights)
}

fn expected_parity_rho_norm<B: BackendTrait>(device: &B::Device) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(
        TensorData::new(
            vec![
                1.5544561, 1.5578794, 1.5574768, 1.5537275, 2.7866828, 2.7363230, 2.6847200,
                2.6322460,
            ],
            [1, 2, 4],
        ),
        device,
    )
}

#[test]
fn rwkv8_direct_graph_matches_manual_rho_norm_on_wgpu() {
    let device = <WgpuBackend as BackendTrait>::Device::default();
    init_wgpu_runtime(&device);
    let (query, value, rho_state, rho_norm_state, decay, _) = parity_inputs::<WgpuBackend>(&device);
    let output = tensorized_rwkv8_forward_direct_graph(
        query,
        value,
        Some(rho_state),
        Some(rho_norm_state),
        decay,
    );
    assert_close_backend(
        output.rho_norm,
        expected_parity_rho_norm::<WgpuBackend>(&device),
        1.0e-5,
        1.0e-5,
    );
}

#[cfg(feature = "cuda")]
#[test]
fn rwkv8_direct_graph_matches_manual_rho_norm_on_cuda() {
    let device = <CudaBackend as BackendTrait>::Device::default();
    let (query, value, rho_state, rho_norm_state, decay, _) = parity_inputs::<CudaBackend>(&device);
    let output = tensorized_rwkv8_forward_direct_graph(
        query,
        value,
        Some(rho_state),
        Some(rho_norm_state),
        decay,
    );
    assert_close_backend(
        output.rho_norm,
        expected_parity_rho_norm::<CudaBackend>(&device),
        1.0e-5,
        1.0e-5,
    );
}

#[cfg(feature = "cuda")]
#[test]
fn rwkv8_direct_graph_matches_manual_rho_norm_on_cuda_autodiff_forward_only() {
    let device = <CudaAutodiffBackend as BackendTrait>::Device::default();
    let (query, value, rho_state, rho_norm_state, decay, _) =
        parity_inputs::<CudaAutodiffBackend>(&device);
    let output = tensorized_rwkv8_forward_direct_graph(
        query,
        value,
        Some(rho_state),
        Some(rho_norm_state),
        decay,
    );
    assert_close_backend(
        output.rho_norm,
        expected_parity_rho_norm::<CudaAutodiffBackend>(&device),
        1.0e-5,
        1.0e-5,
    );
}

#[cfg(feature = "cuda")]
#[test]
fn rwkv8_direct_graph_matches_manual_rho_norm_on_cuda_after_wrapper_call() {
    let _guard = EnvVarGuard::set("BURN_DRAGON_RWKV8_TENSORIZED_TRAIN_WRAPPER", "1");
    let device = <CudaAutodiffBackend as BackendTrait>::Device::default();
    let (query, value, rho_state, rho_norm_state, decay, _) =
        parity_inputs::<CudaAutodiffBackend>(&device);

    let query = query.require_grad();
    let value = value.require_grad();
    let decay = decay.require_grad();

    let _wrapper = tensorized_rwkv8_forward(
        query.clone(),
        value.clone(),
        Some(rho_state.clone()),
        Some(rho_norm_state.clone()),
        decay.clone(),
    );
    let direct = tensorized_rwkv8_forward_direct_graph(
        query,
        value,
        Some(rho_state),
        Some(rho_norm_state),
        decay,
    );
    assert_close_backend(
        direct.rho_norm,
        expected_parity_rho_norm::<CudaAutodiffBackend>(&device),
        1.0e-5,
        1.0e-5,
    );
}

#[test]
fn tensorized_rwkv8_custom_backward_matches_direct_graph_on_wgpu_autodiff() {
    let _guard = EnvVarGuard::set("BURN_DRAGON_RWKV8_TENSORIZED_TRAIN_WRAPPER", "1");
    let device = <WgpuAutodiffBackend as BackendTrait>::Device::default();
    init_wgpu_runtime(&device);
    let (query, value, rho_state, rho_norm_state, decay, weights) =
        parity_inputs::<WgpuAutodiffBackend>(&device);

    let query = query.require_grad();
    let value = value.require_grad();
    let decay = decay.require_grad();

    let wrapper = tensorized_rwkv8_forward(
        query.clone(),
        value.clone(),
        Some(rho_state.clone()),
        Some(rho_norm_state.clone()),
        decay.clone(),
    );
    let direct = tensorized_rwkv8_forward_direct_graph(
        query.clone(),
        value.clone(),
        Some(rho_state),
        Some(rho_norm_state),
        decay.clone(),
    );

    assert_close_backend(
        wrapper.context.clone(),
        direct.context.clone(),
        1.0e-5,
        1.0e-5,
    );
    assert_close_backend(wrapper.rho.clone(), direct.rho.clone(), 1.0e-5, 1.0e-5);
    assert_close_backend(
        wrapper.rho_norm.clone(),
        direct.rho_norm.clone(),
        1.0e-5,
        1.0e-5,
    );

    let wrapper_grads = (wrapper.context * weights.clone()).sum().backward();
    let direct_grads = (direct.context * weights).sum().backward();

    assert_close_backend(
        query.grad(&wrapper_grads).expect("wrapper query grad"),
        query.grad(&direct_grads).expect("direct query grad"),
        3.0e-4,
        3.0e-4,
    );
    assert_close_backend(
        value.grad(&wrapper_grads).expect("wrapper value grad"),
        value.grad(&direct_grads).expect("direct value grad"),
        3.0e-4,
        3.0e-4,
    );
    assert_close_backend(
        decay.grad(&wrapper_grads).expect("wrapper decay grad"),
        decay.grad(&direct_grads).expect("direct decay grad"),
        3.0e-4,
        3.0e-4,
    );
}

#[cfg(feature = "cuda")]
#[test]
fn tensorized_rwkv8_custom_backward_matches_direct_graph_on_cuda_autodiff() {
    let _guard = EnvVarGuard::set("BURN_DRAGON_RWKV8_TENSORIZED_TRAIN_WRAPPER", "1");
    let device = <CudaAutodiffBackend as BackendTrait>::Device::default();
    let (query, value, rho_state, rho_norm_state, decay, weights) =
        parity_inputs::<CudaAutodiffBackend>(&device);

    let query = query.require_grad();
    let value = value.require_grad();
    let decay = decay.require_grad();

    let wrapper = tensorized_rwkv8_forward(
        query.clone(),
        value.clone(),
        Some(rho_state.clone()),
        Some(rho_norm_state.clone()),
        decay.clone(),
    );
    let direct = tensorized_rwkv8_forward_direct_graph(
        query.clone(),
        value.clone(),
        Some(rho_state),
        Some(rho_norm_state),
        decay.clone(),
    );

    assert_close_backend(
        wrapper.context.clone(),
        direct.context.clone(),
        2.0e-4,
        2.0e-4,
    );
    assert_close_backend(wrapper.rho.clone(), direct.rho.clone(), 2.0e-4, 2.0e-4);
    assert_close_backend(
        wrapper.rho_norm.clone(),
        direct.rho_norm.clone(),
        2.0e-4,
        2.0e-4,
    );

    let wrapper_grads = (wrapper.context * weights.clone()).sum().backward();
    let direct_grads = (direct.context * weights).sum().backward();

    assert_close_backend(
        query.grad(&wrapper_grads).expect("wrapper query grad"),
        query.grad(&direct_grads).expect("direct query grad"),
        4.0e-3,
        4.0e-3,
    );
    assert_close_backend(
        value.grad(&wrapper_grads).expect("wrapper value grad"),
        value.grad(&direct_grads).expect("direct value grad"),
        4.0e-3,
        4.0e-3,
    );
    assert_close_backend(
        decay.grad(&wrapper_grads).expect("wrapper decay grad"),
        decay.grad(&direct_grads).expect("direct decay grad"),
        4.0e-3,
        4.0e-3,
    );
}

#[cfg(feature = "cuda")]
#[test]
fn rwkv8_cuda_default_chunk_size_prefers_128_token_windows() {
    let _chunk_guard = EnvVarGuard::set("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_CHUNK", "0");
    let _threshold_guard = EnvVarGuard::set(
        "BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_SCAN_THRESHOLD_BYTES",
        "4294967296",
    );
    let chunk = rwkv8_tensorized_chunk_size::<CudaBackend>(24, 4, 512, 128, 128);
    assert_eq!(chunk, 128);
}
