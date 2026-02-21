#[cfg(not(target_arch = "wasm32"))]
use crate::train::init_wgpu_test_runtime;
use crate::train::prelude::*;
use burn_ndarray::NdArray;

#[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
use burn_cuda::Cuda;
#[cfg(not(target_arch = "wasm32"))]
use burn_wgpu::Wgpu;

fn token_count_for_patch(patch_size: usize) -> usize {
    let image_size = 128usize;
    let patch_size = patch_size.max(1);
    let grid = (image_size / patch_size).max(1);
    grid * grid
}

fn assert_finite(values: &[f32]) {
    assert!(values.iter().all(|value| value.is_finite()));
}

fn run_projection_backend<B: BackendTrait>(
    device: &B::Device,
    config: VisionSaccadeInputProjectionConfig,
    patch_size: usize,
) {
    let embed_dim = 32usize;
    let batch = 2usize;
    let tokens = token_count_for_patch(patch_size);
    let input = Tensor::<B, 3>::random(
        [batch, tokens, embed_dim],
        TensorDistribution::Default,
        device,
    );
    let projection = VisionSaccadeInputProjection::new(embed_dim, patch_size, &config, device);
    let output = projection.forward(input.clone());
    assert_eq!(output.shape().dims::<3>(), input.shape().dims::<3>());
    let data = output
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("output vec");
    assert_finite(&data);
}

fn run_projection_variants<B: BackendTrait>(device: &B::Device, patch_size: usize) {
    run_projection_backend::<B>(
        device,
        VisionSaccadeInputProjectionConfig::Linear,
        patch_size,
    );
    run_projection_backend::<B>(
        device,
        VisionSaccadeInputProjectionConfig::Cnn(VisionSaccadeInputProjectionCnnConfig::default()),
        patch_size,
    );
    run_projection_backend::<B>(
        device,
        VisionSaccadeInputProjectionConfig::RadialMicroVit(
            VisionSaccadeInputProjectionMicroVitConfig::default(),
        ),
        patch_size,
    );
}

#[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
fn cuda_random_kernel_stable() -> bool {
    static STABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *STABLE.get_or_init(|| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let device = burn_cuda::CudaDevice::default();
            let _ = Tensor::<Cuda<f32>, 3>::random([1, 1, 1], TensorDistribution::Default, &device);
            Cuda::<f32>::sync(&device);
        }))
        .is_ok()
    })
}

#[test]
fn input_projection_forward_shapes_across_patch_sizes() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let patch_sizes = [16usize, 32, 64, 128];
    for patch_size in patch_sizes {
        run_projection_variants::<Backend>(&device, patch_size);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn input_projection_forward_shapes_wgpu() {
    type Backend = Wgpu<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    init_wgpu_test_runtime(&device);
    let patch_sizes = [16usize, 32, 64, 128];
    for patch_size in patch_sizes {
        run_projection_variants::<Backend>(&device, patch_size);
    }
}

#[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
#[test]
fn input_projection_forward_shapes_cuda() {
    type Backend = Cuda<f32>;
    if !cuda_random_kernel_stable() {
        return;
    }
    let device = burn_cuda::CudaDevice::default();
    let patch_sizes = [16usize, 32, 64, 128];
    for patch_size in patch_sizes {
        run_projection_variants::<Backend>(&device, patch_size);
    }
}

#[test]
fn input_projection_linear_param_count_matches_formula() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let embed_dim = 32usize;
    let patch_size = 32usize;
    let config = VisionSaccadeInputProjectionConfig::Linear;
    let projection: VisionSaccadeInputProjection<Backend> =
        VisionSaccadeInputProjection::new(embed_dim, patch_size, &config, &device);
    let expected = embed_dim * embed_dim + 3 * embed_dim;
    assert_eq!(projection.param_count(), expected);
}
