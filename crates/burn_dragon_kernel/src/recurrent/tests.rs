use super::*;
use burn::tensor::{Distribution, Tensor};
use burn_autodiff::Autodiff;
use burn_cubecl::cubecl::Runtime;
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

#[derive(Clone, Copy)]
struct MemorySnapshot {
    reserved: u64,
    in_use: u64,
}

fn memory_snapshot(device: &<Backend as BackendTrait>::Device) -> MemorySnapshot {
    let usage = <WgpuRuntime as Runtime>::client(device)
        .memory_usage()
        .expect("wgpu memory usage");
    MemorySnapshot {
        reserved: usage.bytes_reserved,
        in_use: usage.bytes_in_use,
    }
}

fn assert_memory_growth_bounded(
    label: &str,
    snapshots: &[MemorySnapshot],
    max_reserved_growth: u64,
    max_in_use_growth: u64,
) {
    assert!(!snapshots.is_empty(), "{label}: no memory snapshots");
    let first = snapshots[0];
    let last = snapshots[snapshots.len() - 1];
    let reserved_growth = last.reserved.saturating_sub(first.reserved);
    let in_use_growth = last.in_use.saturating_sub(first.in_use);
    assert!(
        reserved_growth <= max_reserved_growth,
        "{label}: reserved growth {} exceeded {}",
        reserved_growth,
        max_reserved_growth
    );
    assert!(
        in_use_growth <= max_in_use_growth,
        "{label}: in_use growth {} exceeded {}",
        in_use_growth,
        max_in_use_growth
    );
}

fn reference_recurrent<B: BackendTrait>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho: Tensor<B, 4>,
    decay: Tensor<B, 1>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, heads, time, _latent] = query.shape().dims::<4>();
    let value_heads = value.shape().dims::<4>()[1];
    let embd = value.shape().dims::<4>()[3];

    let decay = decay.reshape([1, heads, 1, 1]);
    let mut state = rho;
    let mut outputs: Vec<Tensor<B, 4>> = Vec::with_capacity(time);

    for t in 0..time {
        let q_t = query.clone().slice_dim(2, t..t + 1);
        let h_value = if value_heads == 1 {
            value.clone().slice_dim(1, 0..1)
        } else {
            value.clone().slice_dim(1, 0..heads)
        };
        let v_t = h_value.slice_dim(2, t..t + 1);

        let q_latent = q_t.swap_dims(2, 3);
        let context = (state.clone() * q_latent.clone())
            .sum_dim(2)
            .reshape([batch, heads, 1, embd]);
        outputs.push(context);

        state = (state + q_latent * v_t) * decay.clone();
    }

    (Tensor::cat(outputs, 2), state)
}

fn reference_recurrent_autodiff<B: BackendTrait>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho: Tensor<B, 4>,
    decay: Tensor<B, 1>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    reference_recurrent(query, value, rho, decay)
}

#[test]
fn fused_recurrent_matches_reference_with_decay() {
    let device = <Backend as BackendTrait>::Device::default();
    init_runtime(&device);
    <Backend as BackendTrait>::seed(&device, 7);

    let query =
        Tensor::<Backend, 4>::random([2, 4, 6, 16], Distribution::Normal(0.0, 1.0), &device);
    let value =
        Tensor::<Backend, 4>::random([2, 1, 6, 24], Distribution::Normal(0.0, 1.0), &device);
    let rho = Tensor::<Backend, 4>::random([2, 4, 16, 24], Distribution::Normal(0.0, 1.0), &device);
    let decay_values = [0.95_f32, 0.9, 0.85, 0.8];
    let decay = Tensor::<Backend, 1>::from_floats(decay_values.as_slice(), &device);

    let fused =
        try_fused_recurrent_attention_wgpu::<Backend>(&query, &value, Some(&rho), Some(&decay))
            .expect("wgpu fused recurrent output");
    let (reference_context, reference_rho) = reference_recurrent(query, value, rho, decay);

    assert_close_backend(fused.context, reference_context, 2e-4, 2e-4);
    assert_close_backend(fused.rho, reference_rho, 2e-4, 2e-4);
}

#[test]
fn fused_recurrent_matches_reference_without_decay() {
    let device = <Backend as BackendTrait>::Device::default();
    init_runtime(&device);
    <Backend as BackendTrait>::seed(&device, 11);

    let query = Tensor::<Backend, 4>::random([1, 2, 5, 8], Distribution::Normal(0.0, 1.0), &device);
    let value =
        Tensor::<Backend, 4>::random([1, 2, 5, 10], Distribution::Normal(0.0, 1.0), &device);
    let rho = Tensor::<Backend, 4>::zeros([1, 2, 8, 10], &device);
    let decay = Tensor::<Backend, 1>::ones([2], &device);

    let fused =
        try_fused_recurrent_attention_wgpu::<Backend>(&query, &value, Some(&rho), Some(&decay))
            .expect("wgpu fused recurrent output");
    let (reference_context, reference_rho) = reference_recurrent(query, value, rho, decay);

    assert_close_backend(fused.context, reference_context, 2e-4, 2e-4);
    assert_close_backend(fused.rho, reference_rho, 2e-4, 2e-4);
}

#[test]
fn fused_recurrent_matches_reference_query_value_gradients_on_wgpu_autodiff() {
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
    let rho = Tensor::<AutodiffBackendImpl, 4>::from_data(
        TensorData::new(
            (0..48).map(|i| (i as f32) * 0.01 - 0.08).collect(),
            [1, 2, 4, 6],
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

    let fused = try_fused_recurrent_attention_wgpu::<AutodiffBackendImpl>(
        &query,
        &value,
        Some(&rho),
        Some(&decay),
    )
    .expect("wgpu recurrent autodiff");
    let (reference_context, _) =
        reference_recurrent_autodiff(query.clone(), value.clone(), rho.clone(), decay.clone());

    let fused_grads = (fused.context * weights.clone()).sum().backward();
    let reference_grads = (reference_context * weights).sum().backward();

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
        rho.grad(&fused_grads).expect("fused rho grad"),
        rho.grad(&reference_grads).expect("reference rho grad"),
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

#[test]
fn fused_recurrent_memory_stays_bounded_across_repeated_calls() {
    let device = <Backend as BackendTrait>::Device::default();
    init_runtime(&device);
    <Backend as BackendTrait>::seed(&device, 23);

    let query =
        Tensor::<Backend, 4>::random([2, 4, 16, 8], Distribution::Normal(0.0, 1.0), &device);
    let value =
        Tensor::<Backend, 4>::random([2, 1, 16, 12], Distribution::Normal(0.0, 1.0), &device);
    let decay = Tensor::<Backend, 1>::from_floats([0.9, 0.91, 0.92, 0.93], &device);
    let mut rho = Tensor::<Backend, 4>::zeros([2, 4, 8, 12], &device);

    for _ in 0..2 {
        let output =
            try_fused_recurrent_attention_wgpu::<Backend>(&query, &value, Some(&rho), Some(&decay))
                .expect("fused recurrent");
        rho = output.rho;
    }
    let _ = Backend::sync(&device);
    Backend::memory_cleanup(&device);
    let _ = Backend::sync(&device);

    let mut snapshots = Vec::with_capacity(24);
    for step in 0..32 {
        let output =
            try_fused_recurrent_attention_wgpu::<Backend>(&query, &value, Some(&rho), Some(&decay))
                .expect("fused recurrent");
        rho = output.rho;
        let _ = Backend::sync(&device);
        Backend::memory_cleanup(&device);
        let _ = Backend::sync(&device);
        if step >= 8 {
            snapshots.push(memory_snapshot(&device));
        }
    }

    assert_memory_growth_bounded(
        "wgpu_recurrent",
        &snapshots,
        256 * 1024 * 1024,
        64 * 1024 * 1024,
    );
}

#[cfg(feature = "cuda")]
#[test]
fn recurrent_attention_supports_cuda_backend_types() {
    type CudaBackend = Cuda<f32, i32>;
    type CudaAutodiffBackend = Autodiff<CudaBackend>;

    assert!(supports_backend::<CudaBackend>());
    assert!(supports_backend::<CudaAutodiffBackend>());
}

#[cfg(feature = "cuda")]
#[test]
fn fused_recurrent_matches_reference_with_decay_on_cuda() {
    type CudaBackend = Cuda<f32, i32>;

    let device = <CudaBackend as BackendTrait>::Device::default();
    <CudaBackend as BackendTrait>::seed(&device, 7);

    let query =
        Tensor::<CudaBackend, 4>::random([1, 2, 6, 12], Distribution::Normal(0.0, 1.0), &device);
    let value =
        Tensor::<CudaBackend, 4>::random([1, 1, 6, 10], Distribution::Normal(0.0, 1.0), &device);
    let rho =
        Tensor::<CudaBackend, 4>::random([1, 2, 12, 10], Distribution::Normal(0.0, 1.0), &device);
    let decay = Tensor::<CudaBackend, 1>::from_floats([0.95_f32, 0.9], &device);

    let fused =
        try_fused_recurrent_attention_wgpu::<CudaBackend>(&query, &value, Some(&rho), Some(&decay))
            .expect("cuda fused recurrent output");
    let (reference_context, reference_rho) = reference_recurrent(query, value, rho, decay);

    assert_close_backend(fused.context, reference_context, 2e-2, 2e-2);
    assert_close_backend(fused.rho, reference_rho, 2e-2, 2e-2);
}

#[cfg(feature = "cuda")]
#[test]
fn fused_recurrent_matches_reference_query_value_gradients_on_cuda_autodiff() {
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
    let rho = Tensor::<CudaAutodiffBackend, 4>::from_data(
        TensorData::new(
            (0..48).map(|i| (i as f32) * 0.01 - 0.08).collect(),
            [1, 2, 4, 6],
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

    let fused = try_fused_recurrent_attention_wgpu::<CudaAutodiffBackend>(
        &query,
        &value,
        Some(&rho),
        Some(&decay),
    )
    .expect("cuda recurrent autodiff");
    let (reference_context, _) =
        reference_recurrent_autodiff(query.clone(), value.clone(), rho.clone(), decay.clone());

    let fused_grads = (fused.context * weights.clone()).sum().backward();
    let reference_grads = (reference_context * weights).sum().backward();

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
        rho.grad(&fused_grads).expect("fused rho grad"),
        rho.grad(&reference_grads).expect("reference rho grad"),
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
