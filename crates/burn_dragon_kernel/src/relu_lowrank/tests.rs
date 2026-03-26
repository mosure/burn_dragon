use super::*;
use burn::tensor::{Distribution, Tensor, TensorData};
use burn_autodiff::Autodiff;
#[cfg(feature = "cuda")]
use burn_cuda::Cuda;
use burn_wgpu::{CubeBackend, RuntimeOptions, Wgpu, graphics};

type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type AutodiffBackendImpl = Autodiff<Backend>;
type FusionBackend = Wgpu<f32>;
type FusionAutodiffBackendImpl = Autodiff<FusionBackend>;
#[cfg(feature = "cuda")]
type CudaBackend = Cuda<f32>;
#[cfg(feature = "cuda")]
type CudaAutodiffBackendImpl = Autodiff<CudaBackend>;

fn init_runtime(device: &<Backend as BackendTrait>::Device) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

fn assert_close<const D: usize, B: BackendTrait>(
    lhs: BurnTensor<B, D>,
    rhs: BurnTensor<B, D>,
    atol: f32,
    rtol: f32,
) {
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
    assert_eq!(lhs.len(), rhs.len());
    for (index, (lhs, rhs)) in lhs.into_iter().zip(rhs).enumerate() {
        let diff = (lhs - rhs).abs();
        let limit = atol + rtol * rhs.abs();
        assert!(
            diff <= limit,
            "mismatch at {index}: lhs={lhs}, rhs={rhs}, diff={diff}, limit={limit}"
        );
    }
}

#[test]
fn fused_relu_lowrank_matches_reference_single_stream() {
    let device = <Backend as BackendTrait>::Device::default();
    init_runtime(&device);

    let input = Tensor::<Backend, 4>::random([2, 1, 7, 32], Distribution::Default, &device);
    let weight = Tensor::<Backend, 4>::random([1, 4, 32, 16], Distribution::Default, &device);
    let mask = Tensor::<Backend, 1>::from_floats([1.0; 16], &device).reshape([1, 1, 1, 16]);

    let actual =
        try_fused_relu_lowrank_projection_wgpu(&input, &weight, 0.1, Some(&mask)).expect("fused");
    let expected = lowrank_projection_reference_forward(input, weight, 0.1, Some(mask));
    assert_close(actual, expected, 1.0e-4, 1.0e-4);
}

#[test]
fn fused_relu_lowrank_matches_reference_head_aligned() {
    let device = <Backend as BackendTrait>::Device::default();
    init_runtime(&device);

    let input = Tensor::<Backend, 4>::random([2, 4, 5, 24], Distribution::Default, &device);
    let weight = Tensor::<Backend, 4>::random([1, 4, 24, 12], Distribution::Default, &device);
    let actual =
        try_fused_relu_lowrank_projection_wgpu(&input, &weight, 0.05, None).expect("fused");
    let expected = lowrank_projection_reference_forward(input, weight, 0.05, None);
    assert_close(actual, expected, 1.0e-4, 1.0e-4);
}

#[test]
fn fused_relu_lowrank_matches_reference_single_stream_query_weight_gradients_on_wgpu_autodiff() {
    let device = <AutodiffBackendImpl as BackendTrait>::Device::default();
    init_runtime(&device);

    let input =
        Tensor::<AutodiffBackendImpl, 4>::random([2, 1, 7, 16], Distribution::Default, &device)
            .require_grad();
    let weight =
        Tensor::<AutodiffBackendImpl, 4>::random([1, 4, 16, 12], Distribution::Default, &device)
            .require_grad();
    let mask =
        Tensor::<AutodiffBackendImpl, 1>::from_floats([1.0; 12], &device).reshape([1, 1, 1, 12]);
    let output_weights = Tensor::<AutodiffBackendImpl, 4>::from_data(
        TensorData::new(vec![0.05; 2 * 4 * 7 * 12], [2, 4, 7, 12]),
        &device,
    );

    let fused = try_fused_relu_lowrank_projection_wgpu::<AutodiffBackendImpl>(
        &input,
        &weight,
        0.1,
        Some(&mask),
    )
    .expect("fused autodiff");
    let reference =
        lowrank_projection_reference_forward(input.clone(), weight.clone(), 0.1, Some(mask));

    let fused_grads = (fused * output_weights.clone()).sum().backward();
    let reference_grads = (reference * output_weights).sum().backward();

    let fused_input_grad = input.grad(&fused_grads).expect("fused input grad");
    let reference_input_grad = input.grad(&reference_grads).expect("reference input grad");
    let fused_weight_grad = weight.grad(&fused_grads).expect("fused weight grad");
    let reference_weight_grad = weight
        .grad(&reference_grads)
        .expect("reference weight grad");

    assert_close(fused_input_grad, reference_input_grad, 1.0e-4, 1.0e-4);
    assert_close(fused_weight_grad, reference_weight_grad, 1.0e-4, 1.0e-4);
}

#[test]
fn fused_relu_lowrank_matches_reference_single_stream_query_weight_gradients_on_wgpu_autodiff_long_sequence()
 {
    let device = <AutodiffBackendImpl as BackendTrait>::Device::default();
    init_runtime(&device);

    let input =
        Tensor::<AutodiffBackendImpl, 4>::random([1, 1, 401, 8], Distribution::Default, &device)
            .require_grad();
    let weight =
        Tensor::<AutodiffBackendImpl, 4>::random([1, 4, 8, 6], Distribution::Default, &device)
            .require_grad();
    let output_weights = Tensor::<AutodiffBackendImpl, 4>::from_data(
        TensorData::new(vec![0.02; 4 * 401 * 6], [1, 4, 401, 6]),
        &device,
    );

    let fused =
        try_fused_relu_lowrank_projection_wgpu::<AutodiffBackendImpl>(&input, &weight, 0.05, None)
            .expect("fused autodiff");
    let reference = lowrank_projection_reference_forward(input.clone(), weight.clone(), 0.05, None);

    let fused_grads = (fused * output_weights.clone()).sum().backward();
    let reference_grads = (reference * output_weights).sum().backward();

    let fused_input_grad = input.grad(&fused_grads).expect("fused input grad");
    let reference_input_grad = input.grad(&reference_grads).expect("reference input grad");
    let fused_weight_grad = weight.grad(&fused_grads).expect("fused weight grad");
    let reference_weight_grad = weight
        .grad(&reference_grads)
        .expect("reference weight grad");

    assert_close(fused_input_grad, reference_input_grad, 1.0e-4, 1.0e-4);
    assert_close(fused_weight_grad, reference_weight_grad, 1.0e-4, 1.0e-4);
}

#[test]
fn fused_relu_lowrank_matches_reference_head_aligned_query_weight_gradients_on_wgpu_autodiff() {
    let device = <AutodiffBackendImpl as BackendTrait>::Device::default();
    init_runtime(&device);

    let input =
        Tensor::<AutodiffBackendImpl, 4>::random([2, 4, 5, 12], Distribution::Default, &device)
            .require_grad();
    let weight =
        Tensor::<AutodiffBackendImpl, 4>::random([1, 4, 12, 9], Distribution::Default, &device)
            .require_grad();
    let output_weights = Tensor::<AutodiffBackendImpl, 4>::from_data(
        TensorData::new(vec![0.03; 2 * 4 * 5 * 9], [2, 4, 5, 9]),
        &device,
    );

    let fused =
        try_fused_relu_lowrank_projection_wgpu::<AutodiffBackendImpl>(&input, &weight, 0.05, None)
            .expect("fused autodiff");
    let reference = lowrank_projection_reference_forward(input.clone(), weight.clone(), 0.05, None);

    let fused_grads = (fused * output_weights.clone()).sum().backward();
    let reference_grads = (reference * output_weights).sum().backward();

    let fused_input_grad = input.grad(&fused_grads).expect("fused input grad");
    let reference_input_grad = input.grad(&reference_grads).expect("reference input grad");
    let fused_weight_grad = weight.grad(&fused_grads).expect("fused weight grad");
    let reference_weight_grad = weight
        .grad(&reference_grads)
        .expect("reference weight grad");

    assert_close(fused_input_grad, reference_input_grad, 1.0e-4, 1.0e-4);
    assert_close(fused_weight_grad, reference_weight_grad, 1.0e-4, 1.0e-4);
}

#[test]
fn fused_relu_lowrank_matches_reference_head_aligned_query_weight_gradients_on_wgpu_fusion_autodiff()
 {
    let device = <FusionAutodiffBackendImpl as BackendTrait>::Device::default();
    init_runtime(&device);

    let input = Tensor::<FusionAutodiffBackendImpl, 4>::random(
        [2, 4, 5, 12],
        Distribution::Default,
        &device,
    )
    .require_grad();
    let weight = Tensor::<FusionAutodiffBackendImpl, 4>::random(
        [1, 4, 12, 9],
        Distribution::Default,
        &device,
    )
    .require_grad();
    let output_weights = Tensor::<FusionAutodiffBackendImpl, 4>::from_data(
        TensorData::new(vec![0.03; 2 * 4 * 5 * 9], [2, 4, 5, 9]),
        &device,
    );

    let fused = try_fused_relu_lowrank_projection_wgpu::<FusionAutodiffBackendImpl>(
        &input, &weight, 0.05, None,
    )
    .expect("fused fusion autodiff");
    let reference = lowrank_projection_reference_forward(input.clone(), weight.clone(), 0.05, None);

    let fused_grads = (fused * output_weights.clone()).sum().backward();
    let reference_grads = (reference * output_weights).sum().backward();

    let fused_input_grad = input.grad(&fused_grads).expect("fused input grad");
    let reference_input_grad = input.grad(&reference_grads).expect("reference input grad");
    let fused_weight_grad = weight.grad(&fused_grads).expect("fused weight grad");
    let reference_weight_grad = weight
        .grad(&reference_grads)
        .expect("reference weight grad");

    assert_close(fused_input_grad, reference_input_grad, 1.0e-4, 1.0e-4);
    assert_close(fused_weight_grad, reference_weight_grad, 1.0e-4, 1.0e-4);
}

#[test]
fn fused_relu_lowrank_kernel_tiled_matches_reference_head_aligned_query_weight_gradients_on_wgpu_autodiff_long_sequence()
 {
    let device = <AutodiffBackendImpl as BackendTrait>::Device::default();
    init_runtime(&device);

    let input =
        Tensor::<AutodiffBackendImpl, 4>::random([1, 4, 401, 12], Distribution::Default, &device)
            .require_grad();
    let weight =
        Tensor::<AutodiffBackendImpl, 4>::random([1, 4, 12, 9], Distribution::Default, &device)
            .require_grad();
    let output_weights = Tensor::<AutodiffBackendImpl, 4>::from_data(
        TensorData::new(vec![0.03; 4 * 401 * 9], [1, 4, 401, 9]),
        &device,
    );

    let fused = try_fused_relu_lowrank_projection_wgpu_with_executor::<AutodiffBackendImpl>(
        &input,
        &weight,
        0.05,
        None,
        LowrankGradInputExecutor::KernelTiled,
    )
    .expect("fused tiled autodiff");
    let reference = lowrank_projection_reference_forward(input.clone(), weight.clone(), 0.05, None);

    let fused_grads = (fused * output_weights.clone()).sum().backward();
    let reference_grads = (reference * output_weights).sum().backward();

    let fused_input_grad = input.grad(&fused_grads).expect("fused input grad");
    let reference_input_grad = input.grad(&reference_grads).expect("reference input grad");
    let fused_weight_grad = weight.grad(&fused_grads).expect("fused weight grad");
    let reference_weight_grad = weight
        .grad(&reference_grads)
        .expect("reference weight grad");

    assert_close(fused_input_grad, reference_input_grad, 1.0e-4, 1.0e-4);
    assert_close(fused_weight_grad, reference_weight_grad, 1.0e-4, 1.0e-4);
}

#[test]
fn fused_relu_lowrank_kernel_tiled_matches_reference_head_aligned_query_weight_gradients_on_wgpu_fusion_autodiff_long_sequence()
 {
    let device = <FusionAutodiffBackendImpl as BackendTrait>::Device::default();
    init_runtime(&device);

    let input = Tensor::<FusionAutodiffBackendImpl, 4>::random(
        [1, 4, 401, 12],
        Distribution::Default,
        &device,
    )
    .require_grad();
    let weight = Tensor::<FusionAutodiffBackendImpl, 4>::random(
        [1, 4, 12, 9],
        Distribution::Default,
        &device,
    )
    .require_grad();
    let output_weights = Tensor::<FusionAutodiffBackendImpl, 4>::from_data(
        TensorData::new(vec![0.03; 4 * 401 * 9], [1, 4, 401, 9]),
        &device,
    );

    let fused = try_fused_relu_lowrank_projection_wgpu_with_executor::<FusionAutodiffBackendImpl>(
        &input,
        &weight,
        0.05,
        None,
        LowrankGradInputExecutor::KernelTiled,
    )
    .expect("fused tiled fusion autodiff");
    let reference = lowrank_projection_reference_forward(input.clone(), weight.clone(), 0.05, None);

    let fused_grads = (fused * output_weights.clone()).sum().backward();
    let reference_grads = (reference * output_weights).sum().backward();

    let fused_input_grad = input.grad(&fused_grads).expect("fused input grad");
    let reference_input_grad = input.grad(&reference_grads).expect("reference input grad");
    let fused_weight_grad = weight.grad(&fused_grads).expect("fused weight grad");
    let reference_weight_grad = weight
        .grad(&reference_grads)
        .expect("reference weight grad");

    assert_close(fused_input_grad, reference_input_grad, 1.0e-4, 1.0e-4);
    assert_close(fused_weight_grad, reference_weight_grad, 1.0e-4, 1.0e-4);
}

#[cfg(feature = "cuda")]
#[test]
fn fused_relu_lowrank_supports_cuda_backend_types() {
    assert!(supports_relu_lowrank_projection_backend::<CudaBackend>());
    assert!(supports_relu_lowrank_projection_backend::<
        CudaAutodiffBackendImpl,
    >());
}

#[cfg(feature = "cuda")]
#[test]
fn fused_relu_lowrank_matches_reference_single_stream_on_cuda() {
    let device = <CudaBackend as BackendTrait>::Device::default();
    let input = Tensor::<CudaBackend, 4>::random([2, 1, 7, 32], Distribution::Default, &device);
    let weight = Tensor::<CudaBackend, 4>::random([1, 4, 32, 16], Distribution::Default, &device);
    let mask = Tensor::<CudaBackend, 1>::from_floats([1.0; 16], &device).reshape([1, 1, 1, 16]);

    let actual = try_fused_relu_lowrank_projection_wgpu(&input, &weight, 0.1, Some(&mask))
        .expect("cuda fused");
    let expected = lowrank_projection_reference_forward(input, weight, 0.1, Some(mask));
    assert_close(actual, expected, 2.0e-3, 2.0e-3);
}

#[cfg(feature = "cuda")]
#[test]
fn fused_relu_lowrank_matches_reference_gradients_on_cuda_autodiff() {
    let device = <CudaAutodiffBackendImpl as BackendTrait>::Device::default();

    let input =
        Tensor::<CudaAutodiffBackendImpl, 4>::random([1, 1, 5, 16], Distribution::Default, &device)
            .require_grad();
    let weight =
        Tensor::<CudaAutodiffBackendImpl, 4>::random([1, 2, 16, 8], Distribution::Default, &device)
            .require_grad();
    let mask =
        Tensor::<CudaAutodiffBackendImpl, 1>::from_floats([1.0; 8], &device).reshape([1, 1, 1, 8]);
    let output_weights = Tensor::<CudaAutodiffBackendImpl, 4>::from_data(
        TensorData::new(vec![0.05; 1 * 2 * 5 * 8], [1, 2, 5, 8]),
        &device,
    );

    let fused = try_fused_relu_lowrank_projection_wgpu::<CudaAutodiffBackendImpl>(
        &input,
        &weight,
        0.1,
        Some(&mask),
    )
    .expect("cuda fused autodiff");
    let reference =
        lowrank_projection_reference_forward(input.clone(), weight.clone(), 0.1, Some(mask));

    let fused_grads = (fused * output_weights.clone()).sum().backward();
    let reference_grads = (reference * output_weights).sum().backward();

    let fused_input_grad = input.grad(&fused_grads).expect("fused input grad");
    let reference_input_grad = input.grad(&reference_grads).expect("reference input grad");
    let fused_weight_grad = weight.grad(&fused_grads).expect("fused weight grad");
    let reference_weight_grad = weight
        .grad(&reference_grads)
        .expect("reference weight grad");

    assert_close(fused_input_grad, reference_input_grad, 2.0e-3, 2.0e-3);
    assert_close(fused_weight_grad, reference_weight_grad, 2.0e-3, 2.0e-3);
}
