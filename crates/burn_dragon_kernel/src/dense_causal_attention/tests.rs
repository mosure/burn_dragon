use super::*;
use burn::tensor::{Distribution, Tensor, TensorData};
use burn_autodiff::Autodiff;
#[cfg(feature = "cuda")]
use burn_cuda::Cuda;
use burn_wgpu::{CubeBackend, RuntimeOptions, graphics};

type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type AutodiffBackendImpl = Autodiff<Backend>;

fn init_runtime(device: &<Backend as BackendTrait>::Device) {
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
    let mut max_lhs = 0.0_f32;
    let mut max_rhs = 0.0_f32;
    for (a, b) in lhs_data.iter().zip(rhs_data.iter()) {
        let diff = (a - b).abs();
        let tol = atol + rtol * b.abs();
        if diff > max_diff {
            max_diff = diff;
            max_tol = tol;
            max_lhs = *a;
            max_rhs = *b;
        }
    }
    assert!(
        max_diff <= max_tol,
        "max difference {max_diff} exceeds tolerance {max_tol} (lhs={max_lhs}, rhs={max_rhs})"
    );
}

fn assert_close(lhs: Tensor<Backend, 4>, rhs: Tensor<Backend, 4>, atol: f32, rtol: f32) {
    assert_close_backend(lhs, rhs, atol, rtol);
}

fn reference_attention(
    query: Tensor<Backend, 4>,
    value: Tensor<Backend, 4>,
    decay: Tensor<Backend, 1>,
) -> Tensor<Backend, 4> {
    dense_causal_attention_reference(query, value, decay)
}

#[test]
fn dense_causal_attention_matches_reference_on_wgpu() {
    let device = <Backend as BackendTrait>::Device::default();
    init_runtime(&device);
    <Backend as BackendTrait>::seed(&device, 17);

    let query =
        Tensor::<Backend, 4>::random([2, 4, 16, 32], Distribution::Normal(0.0, 1.0), &device);
    let value =
        Tensor::<Backend, 4>::random([2, 1, 16, 24], Distribution::Normal(0.0, 1.0), &device);
    let decay = Tensor::<Backend, 1>::from_floats([0.97, 0.93, 0.89, 0.85], &device);

    let fused = try_fused_dense_causal_attention_wgpu::<Backend>(&query, &value, &decay)
        .expect("wgpu dense causal attention");
    let expected = reference_attention(query, value, decay);
    assert_close(fused, expected, 2e-4, 2e-4);
}

#[test]
fn dense_causal_attention_matches_reference_gradients_on_wgpu_autodiff() {
    let device = <AutodiffBackendImpl as BackendTrait>::Device::default();
    init_runtime(&device);

    let query = Tensor::<AutodiffBackendImpl, 4>::from_data(
        TensorData::new(
            (0..24).map(|i| (i as f32) * 0.03 - 0.2).collect(),
            [1, 2, 3, 4],
        ),
        &device,
    )
    .require_grad();
    let value = Tensor::<AutodiffBackendImpl, 4>::from_data(
        TensorData::new(
            (0..18).map(|i| (i as f32) * 0.05 - 0.15).collect(),
            [1, 1, 3, 6],
        ),
        &device,
    )
    .require_grad();
    let decay =
        Tensor::<AutodiffBackendImpl, 1>::from_data(TensorData::new(vec![0.95, 0.9], [2]), &device)
            .require_grad();
    let weights = Tensor::<AutodiffBackendImpl, 4>::from_data(
        TensorData::new(
            (0..36).map(|i| (i as f32) * 0.02 - 0.1).collect(),
            [1, 2, 3, 6],
        ),
        &device,
    );

    let fused =
        try_fused_dense_causal_attention_wgpu::<AutodiffBackendImpl>(&query, &value, &decay)
            .expect("wgpu dense causal attention autodiff");
    let reference = dense_causal_attention_reference(query.clone(), value.clone(), decay.clone());

    let fused_grads = (fused * weights.clone()).sum().backward();
    let reference_grads = (reference * weights).sum().backward();

    assert_close_backend(
        query.grad(&fused_grads).expect("fused query grad"),
        query.grad(&reference_grads).expect("reference query grad"),
        5e-3,
        5e-3,
    );
    assert_close_backend(
        value.grad(&fused_grads).expect("fused value grad"),
        value.grad(&reference_grads).expect("reference value grad"),
        5e-3,
        5e-3,
    );
    assert_close_backend(
        decay.grad(&fused_grads).expect("fused decay grad"),
        decay.grad(&reference_grads).expect("reference decay grad"),
        5e-3,
        5e-3,
    );
}

#[cfg(feature = "cuda")]
#[test]
fn dense_causal_attention_supports_cuda_backend_types() {
    type CudaBackend = Cuda<f32, i32>;
    type CudaAutodiffBackend = Autodiff<CudaBackend>;

    assert!(supports_dense_causal_attention_backend::<CudaBackend>());
    assert!(supports_dense_causal_attention_backend::<CudaAutodiffBackend>());
}

#[cfg(feature = "cuda")]
#[test]
fn dense_causal_attention_matches_reference_on_cuda() {
    type CudaBackend = Cuda<f32, i32>;
    let device = <CudaBackend as BackendTrait>::Device::default();
    <CudaBackend as BackendTrait>::seed(&device, 17);

    let query =
        Tensor::<CudaBackend, 4>::random([1, 2, 8, 16], Distribution::Normal(0.0, 1.0), &device);
    let value =
        Tensor::<CudaBackend, 4>::random([1, 1, 8, 12], Distribution::Normal(0.0, 1.0), &device);
    let decay = Tensor::<CudaBackend, 1>::from_floats([0.97, 0.93], &device);

    let fused = try_fused_dense_causal_attention_wgpu::<CudaBackend>(&query, &value, &decay)
        .expect("cuda dense causal attention");
    let expected = dense_causal_attention_reference(query, value, decay);
    let _ = <CudaBackend as BackendTrait>::sync(&device);
    assert_close_backend(fused, expected, 2e-2, 2e-2);
}

#[cfg(feature = "cuda")]
#[test]
fn dense_causal_attention_matches_reference_gradients_on_cuda_autodiff() {
    type CudaBackend = Cuda<f32, i32>;
    type CudaAutodiffBackend = Autodiff<CudaBackend>;

    let device = <CudaAutodiffBackend as BackendTrait>::Device::default();

    let query = Tensor::<CudaAutodiffBackend, 4>::from_data(
        TensorData::new(
            (0..24).map(|i| (i as f32) * 0.03 - 0.2).collect(),
            [1, 2, 3, 4],
        ),
        &device,
    )
    .require_grad();
    let value = Tensor::<CudaAutodiffBackend, 4>::from_data(
        TensorData::new(
            (0..18).map(|i| (i as f32) * 0.05 - 0.15).collect(),
            [1, 1, 3, 6],
        ),
        &device,
    )
    .require_grad();
    let decay =
        Tensor::<CudaAutodiffBackend, 1>::from_data(TensorData::new(vec![0.95, 0.9], [2]), &device)
            .require_grad();
    let weights = Tensor::<CudaAutodiffBackend, 4>::from_data(
        TensorData::new(
            (0..36).map(|i| (i as f32) * 0.02 - 0.1).collect(),
            [1, 2, 3, 6],
        ),
        &device,
    );

    let fused =
        try_fused_dense_causal_attention_wgpu::<CudaAutodiffBackend>(&query, &value, &decay)
            .expect("cuda dense causal attention autodiff");
    let reference = dense_causal_attention_reference(query.clone(), value.clone(), decay.clone());

    let fused_grads = (fused * weights.clone()).sum().backward();
    let reference_grads = (reference * weights).sum().backward();

    assert_close_backend(
        query.grad(&fused_grads).expect("fused query grad"),
        query.grad(&reference_grads).expect("reference query grad"),
        3e-2,
        3e-2,
    );
    assert_close_backend(
        value.grad(&fused_grads).expect("fused value grad"),
        value.grad(&reference_grads).expect("reference value grad"),
        3e-2,
        3e-2,
    );
    assert_close_backend(
        decay.grad(&fused_grads).expect("fused decay grad"),
        decay.grad(&reference_grads).expect("reference decay grad"),
        3e-2,
        3e-2,
    );
}
