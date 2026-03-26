use super::*;
use burn::tensor::TensorData;
use burn::tensor::backend::Backend as BackendTrait;
use burn_cubecl::cubecl::Runtime;
use burn_ndarray::NdArray;
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};

#[test]
fn structured_pyramid_reference_step_preserves_shapes() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let shape = StructuredPyramidShape {
        patch: LocalGridShape2d::new(2, 2),
        coarse: LocalGridShape2d::new(1, 1),
        coarse_stride: 2,
        hub_count: 2,
    };
    let input = StructuredPyramidRhoStepInput {
        patch_query: Tensor::<Backend, 4>::ones([1, 2, 2, 2], &device),
        patch_value: Tensor::<Backend, 4>::ones([1, 4, 2, 2], &device),
        coarse_query: Tensor::<Backend, 4>::ones([1, 2, 1, 1], &device),
        coarse_value: Tensor::<Backend, 4>::ones([1, 4, 1, 1], &device),
        patch_rho: Tensor::<Backend, 5>::zeros([1, 2, 4, 2, 2], &device),
        coarse_rho: Tensor::<Backend, 5>::zeros([1, 2, 4, 1, 1], &device),
        hub_rho: Tensor::<Backend, 4>::zeros([1, 2, 2, 4], &device),
        patch_hub_weights: None,
        coarse_hub_weights: None,
        neighborhood: LocalGridNeighborhood::moore(1),
        decay: Tensor::<Backend, 1>::ones([2], &device),
    };

    let output = reference_structured_pyramid_rho_step(shape, input);
    assert_eq!(output.patch_local_context.shape().dims(), [1, 4, 2, 2]);
    assert_eq!(output.coarse_local_context.shape().dims(), [1, 4, 1, 1]);
    assert_eq!(
        output.patch_from_coarse_context.shape().dims(),
        [1, 4, 2, 2]
    );
    assert_eq!(output.patch_from_hub_context.shape().dims(), [1, 4, 2, 2]);
    assert_eq!(output.coarse_from_hub_context.shape().dims(), [1, 4, 1, 1]);
    assert_eq!(output.next_patch_rho.shape().dims(), [1, 2, 4, 2, 2]);
    assert_eq!(output.next_coarse_rho.shape().dims(), [1, 2, 4, 1, 1]);
    assert_eq!(output.next_hub_rho.shape().dims(), [1, 2, 2, 4]);
}

#[test]
fn structured_pyramid_reference_step_writes_patch_and_coarse_activity_into_hub_bank() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let shape = StructuredPyramidShape {
        patch: LocalGridShape2d::new(2, 2),
        coarse: LocalGridShape2d::new(1, 1),
        coarse_stride: 2,
        hub_count: 2,
    };
    let input = StructuredPyramidRhoStepInput {
        patch_query: Tensor::<Backend, 4>::ones([1, 2, 2, 2], &device),
        patch_value: Tensor::<Backend, 4>::ones([1, 4, 2, 2], &device),
        coarse_query: Tensor::<Backend, 4>::ones([1, 2, 1, 1], &device),
        coarse_value: Tensor::<Backend, 4>::ones([1, 4, 1, 1], &device),
        patch_rho: Tensor::<Backend, 5>::zeros([1, 2, 4, 2, 2], &device),
        coarse_rho: Tensor::<Backend, 5>::zeros([1, 2, 4, 1, 1], &device),
        hub_rho: Tensor::<Backend, 4>::zeros([1, 2, 2, 4], &device),
        patch_hub_weights: None,
        coarse_hub_weights: None,
        neighborhood: LocalGridNeighborhood::moore(1),
        decay: Tensor::<Backend, 1>::ones([2], &device),
    };

    let output = reference_structured_pyramid_rho_step(shape, input);
    let max_abs = output
        .next_hub_rho
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("hub rho")
        .into_iter()
        .map(f32::abs)
        .fold(0.0_f32, f32::max);
    assert!(max_abs > 0.0);
}

#[test]
fn cross_scale_read_tiles_the_coarse_grid_across_patch_space() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let coarse_shape = LocalGridShape2d::new(2, 2);
    let patch_shape = LocalGridShape2d::new(4, 4);
    let coarse_rho = Tensor::<Backend, 5>::from_data(
        TensorData::new(
            vec![
                1.0, 2.0, //
                3.0, 4.0,
            ],
            [1, 1, 1, 2, 2],
        ),
        &device,
    );
    let patch_query = Tensor::<Backend, 4>::ones([1, 1, 4, 4], &device);

    let context = cross_scale_read(coarse_rho, patch_query, coarse_shape, patch_shape, 2);
    assert_eq!(context.shape().dims(), [1, 1, 4, 4]);
    assert_eq!(
        context
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("cross scale context"),
        vec![
            1.0, 2.0, 1.0, 2.0, //
            3.0, 4.0, 3.0, 4.0, //
            1.0, 2.0, 1.0, 2.0, //
            3.0, 4.0, 3.0, 4.0,
        ]
    );
}

#[test]
fn pool_target_major_outer_sums_patch_blocks_without_spatial_roundtrip() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let update = Tensor::<Backend, 4>::from_data(
        TensorData::new(
            vec![
                1.0, 2.0, //
                3.0, 4.0,
            ],
            [1, 4, 1, 1],
        ),
        &device,
    );

    let pooled = pool_target_major_outer(update, LocalGridShape2d::new(2, 2), 2);
    assert_eq!(pooled.shape().dims(), [1, 1, 1, 1]);
    let values = pooled
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("pooled update");
    assert!((values[0] - 10.0).abs() <= 1.0e-6);
}

#[test]
fn pool_target_major_outer_fast_matches_reference_pooling() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let update = Tensor::<Backend, 4>::from_data(
        TensorData::new((0..16).map(|value| value as f32).collect(), [1, 4, 2, 2]),
        &device,
    );
    let patch_shape = LocalGridShape2d::new(2, 2);
    let coarse_shape = LocalGridShape2d::new(1, 1);
    let route = patch_to_coarse_pool_route::<Backend>(1, patch_shape, 2, &device);
    let fast =
        pool_target_major_outer_fast(update.clone(), patch_shape, coarse_shape, 2, Some(route))
            .expect("pooled route");
    let reference = pool_target_major_outer(update, patch_shape, 2);
    assert!(max_abs_diff_any(fast, reference) <= 1.0e-6);
}

#[test]
fn hub_read_matches_manual_weighted_projection() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let hub_rho = Tensor::<Backend, 4>::from_data(
        TensorData::new(
            vec![
                1.0, 2.0, //
                3.0, 4.0,
            ],
            [1, 2, 2, 1],
        ),
        &device,
    );
    let query = Tensor::<Backend, 4>::from_data(
        TensorData::new(
            vec![
                1.0, 0.0, //
                0.0, 1.0,
            ],
            [1, 2, 2, 1],
        ),
        &device,
    );
    let weights = Tensor::<Backend, 4>::from_data(
        TensorData::new(
            vec![
                0.75, 0.20, //
                0.25, 0.80,
            ],
            [1, 2, 2, 1],
        ),
        &device,
    );

    let output = hub_read(hub_rho, query, Some(weights));
    assert_eq!(output.shape().dims(), [1, 1, 2, 1]);
    let values = output
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("hub read output");
    assert!((values[0] - 1.5).abs() <= 1.0e-6);
    assert!((values[1] - 3.6).abs() <= 1.0e-6);
}

#[test]
fn hub_read_pair_matches_separate_hub_reads() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let hub_rho = Tensor::<Backend, 4>::from_data(
        TensorData::new(
            vec![
                1.0, 2.0, //
                3.0, 4.0,
            ],
            [1, 2, 2, 1],
        ),
        &device,
    );
    let patch_query = Tensor::<Backend, 4>::from_data(
        TensorData::new(
            vec![
                1.0, 0.0, //
                0.0, 1.0,
            ],
            [1, 2, 2, 1],
        ),
        &device,
    );
    let coarse_query =
        Tensor::<Backend, 4>::from_data(TensorData::new(vec![0.5, 1.5], [1, 2, 1, 1]), &device);
    let patch_weights = Tensor::<Backend, 4>::from_data(
        TensorData::new(
            vec![
                0.75, 0.20, //
                0.25, 0.80,
            ],
            [1, 2, 2, 1],
        ),
        &device,
    );
    let coarse_weights =
        Tensor::<Backend, 4>::from_data(TensorData::new(vec![0.4, 0.6], [1, 2, 1, 1]), &device);

    let (patch_pair, coarse_pair) = hub_read_pair(
        hub_rho.clone(),
        patch_query.clone(),
        coarse_query.clone(),
        Some(patch_weights.clone()),
        Some(coarse_weights.clone()),
        LocalGridShape2d::new(2, 1),
        LocalGridShape2d::new(1, 1),
    );
    let patch_reference = hub_read(hub_rho.clone(), patch_query, Some(patch_weights));
    let coarse_reference = hub_read(hub_rho, coarse_query, Some(coarse_weights));

    assert!(max_abs_diff_any(patch_pair, patch_reference) <= 1.0e-6);
    assert!(max_abs_diff_any(coarse_pair, coarse_reference) <= 1.0e-6);
}

#[test]
fn update_hub_from_deltas_distributes_multi_hub_delta_uniformly() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let hub_rho = Tensor::<Backend, 4>::zeros([1, 2, 2, 1], &device);
    let patch_delta =
        Tensor::<Backend, 3>::from_data(TensorData::new(vec![2.0, 4.0], [1, 2, 1]), &device);
    let coarse_delta =
        Tensor::<Backend, 3>::from_data(TensorData::new(vec![1.0, 3.0], [1, 2, 1]), &device);
    let decay = Tensor::<Backend, 1>::ones([2], &device);

    let output = update_hub_from_deltas(hub_rho, patch_delta, coarse_delta, 2, decay);
    assert_eq!(output.shape().dims(), [1, 2, 2, 1]);
    assert_eq!(
        output
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("hub rho output"),
        vec![
            1.5, 3.5, //
            1.5, 3.5,
        ]
    );
}

#[test]
fn weighted_global_sum_pair_matches_sum_of_individual_weighted_updates() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();

    let patch_update = Tensor::<Backend, 5>::from_data(
        TensorData::new(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0,
            ],
            [1, 2, 1, 2, 2],
        ),
        &device,
    );
    let coarse_update = Tensor::<Backend, 5>::from_data(
        TensorData::new(
            vec![
                0.5, 1.5, //
                2.5, 3.5,
            ],
            [1, 2, 1, 1, 2],
        ),
        &device,
    );
    let patch_weights = Tensor::<Backend, 4>::from_data(
        TensorData::new(
            vec![
                0.10, 0.20, 0.30, 0.40, //
                0.40, 0.30, 0.20, 0.10,
            ],
            [1, 2, 2, 2],
        ),
        &device,
    );
    let coarse_weights = Tensor::<Backend, 4>::from_data(
        TensorData::new(
            vec![
                0.25, 0.75, //
                0.60, 0.40,
            ],
            [1, 2, 1, 2],
        ),
        &device,
    );

    let combined = weighted_global_sum_pair(
        Some(patch_update.clone()),
        Some(coarse_update.clone()),
        Some(patch_weights.clone()),
        Some(coarse_weights.clone()),
        2,
    )
    .expect("combined update");

    let separate = weighted_global_sum(patch_update, Some(patch_weights), 2)
        .add(weighted_global_sum(coarse_update, Some(coarse_weights), 2));

    assert!(max_abs_diff_any(combined, separate) <= 1.0e-6);
}

#[cfg(not(target_arch = "wasm32"))]
type WgpuBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

#[cfg(not(target_arch = "wasm32"))]
fn init_wgpu_runtime(device: &<WgpuBackend as BackendTrait>::Device) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
struct MemorySnapshot {
    reserved: u64,
    in_use: u64,
}

#[cfg(not(target_arch = "wasm32"))]
fn memory_snapshot(device: &<WgpuBackend as BackendTrait>::Device) -> MemorySnapshot {
    let usage = <WgpuRuntime as Runtime>::client(device).memory_usage();
    MemorySnapshot {
        reserved: usage.bytes_reserved,
        in_use: usage.bytes_in_use,
    }
}

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
fn max_abs_diff<const D: usize>(lhs: Tensor<WgpuBackend, D>, rhs: Tensor<WgpuBackend, D>) -> f32 {
    lhs.sub(rhs)
        .abs()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("diff vec")
        .into_iter()
        .fold(0.0_f32, f32::max)
}

fn max_abs_diff_any<B: BackendTrait, const D: usize>(lhs: Tensor<B, D>, rhs: Tensor<B, D>) -> f32 {
    lhs.sub(rhs)
        .abs()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("diff vec")
        .into_iter()
        .fold(0.0_f32, f32::max)
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn fused_structured_pyramid_matches_reference() {
    let device = <WgpuBackend as BackendTrait>::Device::default();
    init_wgpu_runtime(&device);

    let shape = StructuredPyramidShape {
        patch: LocalGridShape2d::new(4, 4),
        coarse: LocalGridShape2d::new(2, 2),
        coarse_stride: 2,
        hub_count: 2,
    };
    let input = StructuredPyramidRhoStepInput {
        patch_query: Tensor::<WgpuBackend, 4>::random(
            [2, 4, 4, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ),
        patch_value: Tensor::<WgpuBackend, 4>::random(
            [2, 6, 4, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ),
        coarse_query: Tensor::<WgpuBackend, 4>::random(
            [2, 4, 2, 2],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ),
        coarse_value: Tensor::<WgpuBackend, 4>::random(
            [2, 6, 2, 2],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ),
        patch_rho: Tensor::<WgpuBackend, 5>::random(
            [2, 4, 6, 4, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ),
        coarse_rho: Tensor::<WgpuBackend, 5>::random(
            [2, 4, 6, 2, 2],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ),
        hub_rho: Tensor::<WgpuBackend, 4>::random(
            [2, 2, 4, 6],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ),
        patch_hub_weights: None,
        coarse_hub_weights: None,
        neighborhood: LocalGridNeighborhood::moore(1),
        decay: Tensor::<WgpuBackend, 1>::from_floats([0.85, 0.9, 0.95, 0.975], &device),
    };

    let reference = reference_structured_pyramid_rho_step(shape, input.clone());
    let plan = CompiledStructuredPyramidRhoPlan::new(2, 4, 6, shape, input.neighborhood, &device);
    let fused = try_fused_structured_pyramid_rho_step_wgpu_with_plan(shape, input, &plan)
        .expect("structured pyramid fused output");

    assert!(max_abs_diff(fused.patch_local_context, reference.patch_local_context) <= 1.0e-5);
    assert!(max_abs_diff(fused.coarse_local_context, reference.coarse_local_context) <= 1.0e-5);
    assert!(
        max_abs_diff(
            fused.patch_from_coarse_context,
            reference.patch_from_coarse_context
        ) <= 1.0e-5
    );
    assert!(
        max_abs_diff(
            fused.patch_from_hub_context,
            reference.patch_from_hub_context
        ) <= 1.0e-5
    );
    assert!(
        max_abs_diff(
            fused.coarse_from_hub_context,
            reference.coarse_from_hub_context
        ) <= 1.0e-5
    );
    assert!(max_abs_diff(fused.next_patch_rho, reference.next_patch_rho) <= 1.0e-5);
    assert!(max_abs_diff(fused.next_coarse_rho, reference.next_coarse_rho) <= 1.0e-5);
    assert!(max_abs_diff(fused.next_hub_rho, reference.next_hub_rho) <= 1.0e-5);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn fused_structured_pyramid_matches_reference_on_compact_hub_gated_shape() {
    let device = <WgpuBackend as BackendTrait>::Device::default();
    init_wgpu_runtime(&device);
    <WgpuBackend as BackendTrait>::seed(&device, 6_060);

    let shape = StructuredPyramidShape {
        patch: LocalGridShape2d::new(2, 2),
        coarse: LocalGridShape2d::new(1, 1),
        coarse_stride: 2,
        hub_count: 2,
    };
    let patch_hub_weights = Tensor::<WgpuBackend, 4>::random(
        [2, 2, 2, 2],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    )
    .abs();
    let patch_hub_weights =
        patch_hub_weights.clone() / patch_hub_weights.clone().sum_dim(1).add_scalar(1.0e-6);
    let coarse_hub_weights = Tensor::<WgpuBackend, 4>::random(
        [2, 2, 1, 1],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    )
    .abs();
    let coarse_hub_weights =
        coarse_hub_weights.clone() / coarse_hub_weights.clone().sum_dim(1).add_scalar(1.0e-6);
    let input = StructuredPyramidRhoStepInput {
        patch_query: Tensor::<WgpuBackend, 4>::random(
            [2, 2, 2, 2],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ),
        patch_value: Tensor::<WgpuBackend, 4>::random(
            [2, 4, 2, 2],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ),
        coarse_query: Tensor::<WgpuBackend, 4>::random(
            [2, 2, 1, 1],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ),
        coarse_value: Tensor::<WgpuBackend, 4>::random(
            [2, 4, 1, 1],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ),
        patch_rho: Tensor::<WgpuBackend, 5>::random(
            [2, 2, 4, 2, 2],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ),
        coarse_rho: Tensor::<WgpuBackend, 5>::random(
            [2, 2, 4, 1, 1],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ),
        hub_rho: Tensor::<WgpuBackend, 4>::random(
            [2, 2, 2, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ),
        patch_hub_weights: Some(patch_hub_weights),
        coarse_hub_weights: Some(coarse_hub_weights),
        neighborhood: LocalGridNeighborhood::moore(1),
        decay: Tensor::<WgpuBackend, 1>::from_floats([0.85, 0.95], &device),
    };

    let reference = reference_structured_pyramid_rho_step(shape, input.clone());
    let plan = CompiledStructuredPyramidRhoPlan::new(2, 2, 4, shape, input.neighborhood, &device);
    let fused = try_fused_structured_pyramid_rho_step_wgpu_with_plan(shape, input, &plan)
        .expect("structured pyramid fused output");

    assert!(max_abs_diff(fused.patch_local_context, reference.patch_local_context) <= 1.0e-5);
    assert!(max_abs_diff(fused.coarse_local_context, reference.coarse_local_context) <= 1.0e-5);
    assert!(
        max_abs_diff(
            fused.patch_from_coarse_context,
            reference.patch_from_coarse_context
        ) <= 1.0e-5
    );
    assert!(
        max_abs_diff(
            fused.patch_from_hub_context,
            reference.patch_from_hub_context
        ) <= 1.0e-5
    );
    assert!(
        max_abs_diff(
            fused.coarse_from_hub_context,
            reference.coarse_from_hub_context
        ) <= 1.0e-5
    );
    assert!(max_abs_diff(fused.next_patch_rho, reference.next_patch_rho) <= 1.0e-5);
    assert!(max_abs_diff(fused.next_coarse_rho, reference.next_coarse_rho) <= 1.0e-5);
    assert!(max_abs_diff(fused.next_hub_rho, reference.next_hub_rho) <= 1.0e-5);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn fused_structured_pyramid_matches_reference_on_stride3_tiled_cross_scale_shape() {
    let device = <WgpuBackend as BackendTrait>::Device::default();
    init_wgpu_runtime(&device);
    <WgpuBackend as BackendTrait>::seed(&device, 7_070);

    let shape = StructuredPyramidShape {
        patch: LocalGridShape2d::new(6, 6),
        coarse: LocalGridShape2d::new(2, 2),
        coarse_stride: 3,
        hub_count: 2,
    };
    let input = StructuredPyramidRhoStepInput {
        patch_query: Tensor::<WgpuBackend, 4>::random(
            [1, 3, 6, 6],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ),
        patch_value: Tensor::<WgpuBackend, 4>::random(
            [1, 5, 6, 6],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ),
        coarse_query: Tensor::<WgpuBackend, 4>::random(
            [1, 3, 2, 2],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ),
        coarse_value: Tensor::<WgpuBackend, 4>::random(
            [1, 5, 2, 2],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ),
        patch_rho: Tensor::<WgpuBackend, 5>::random(
            [1, 3, 5, 6, 6],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ),
        coarse_rho: Tensor::<WgpuBackend, 5>::random(
            [1, 3, 5, 2, 2],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ),
        hub_rho: Tensor::<WgpuBackend, 4>::random(
            [1, 2, 3, 5],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ),
        patch_hub_weights: None,
        coarse_hub_weights: None,
        neighborhood: LocalGridNeighborhood::moore(1),
        decay: Tensor::<WgpuBackend, 1>::from_floats([0.85, 0.925, 0.975], &device),
    };

    let reference = reference_structured_pyramid_rho_step(shape, input.clone());
    let plan = CompiledStructuredPyramidRhoPlan::new(1, 3, 5, shape, input.neighborhood, &device);
    let fused = try_fused_structured_pyramid_rho_step_wgpu_with_plan(shape, input, &plan)
        .expect("structured pyramid fused output");

    assert!(
        max_abs_diff(
            fused.patch_from_coarse_context,
            reference.patch_from_coarse_context
        ) <= 1.0e-5
    );
    assert!(max_abs_diff(fused.next_patch_rho, reference.next_patch_rho) <= 1.0e-5);
    assert!(max_abs_diff(fused.next_coarse_rho, reference.next_coarse_rho) <= 1.0e-5);
    assert!(max_abs_diff(fused.next_hub_rho, reference.next_hub_rho) <= 1.0e-5);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn structured_pyramid_reference_memory_stays_bounded_across_repeated_calls() {
    let device = <WgpuBackend as BackendTrait>::Device::default();
    init_wgpu_runtime(&device);

    let shape = StructuredPyramidShape {
        patch: LocalGridShape2d::new(4, 4),
        coarse: LocalGridShape2d::new(2, 2),
        coarse_stride: 2,
        hub_count: 2,
    };
    let patch_query = Tensor::<WgpuBackend, 4>::ones([1, 4, 4, 4], &device);
    let patch_value = Tensor::<WgpuBackend, 4>::ones([1, 6, 4, 4], &device);
    let coarse_query = Tensor::<WgpuBackend, 4>::ones([1, 4, 2, 2], &device);
    let coarse_value = Tensor::<WgpuBackend, 4>::ones([1, 6, 2, 2], &device);
    let decay = Tensor::<WgpuBackend, 1>::ones([4], &device);

    let mut patch_rho = Tensor::<WgpuBackend, 5>::zeros([1, 4, 6, 4, 4], &device);
    let mut coarse_rho = Tensor::<WgpuBackend, 5>::zeros([1, 4, 6, 2, 2], &device);
    let mut hub_rho = Tensor::<WgpuBackend, 4>::zeros([1, 2, 4, 6], &device);

    let mut snapshots = Vec::with_capacity(24);
    for step in 0..32 {
        let output = reference_structured_pyramid_rho_step(
            shape,
            StructuredPyramidRhoStepInput {
                patch_query: patch_query.clone(),
                patch_value: patch_value.clone(),
                coarse_query: coarse_query.clone(),
                coarse_value: coarse_value.clone(),
                patch_rho: patch_rho.clone(),
                coarse_rho: coarse_rho.clone(),
                hub_rho: hub_rho.clone(),
                patch_hub_weights: None,
                coarse_hub_weights: None,
                neighborhood: LocalGridNeighborhood::moore(1),
                decay: decay.clone(),
            },
        );
        patch_rho = output.next_patch_rho;
        coarse_rho = output.next_coarse_rho;
        hub_rho = output.next_hub_rho;
        let _ = WgpuBackend::sync(&device);
        WgpuBackend::memory_cleanup(&device);
        let _ = WgpuBackend::sync(&device);
        if step >= 8 {
            snapshots.push(memory_snapshot(&device));
        }
    }

    assert_memory_growth_bounded(
        "structured_pyramid_reference",
        &snapshots,
        256 * 1024 * 1024,
        64 * 1024 * 1024,
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn fused_structured_pyramid_memory_stays_bounded_across_repeated_calls() {
    let device = <WgpuBackend as BackendTrait>::Device::default();
    init_wgpu_runtime(&device);

    let shape = StructuredPyramidShape {
        patch: LocalGridShape2d::new(4, 4),
        coarse: LocalGridShape2d::new(2, 2),
        coarse_stride: 2,
        hub_count: 2,
    };
    let patch_query = Tensor::<WgpuBackend, 4>::ones([1, 4, 4, 4], &device);
    let patch_value = Tensor::<WgpuBackend, 4>::ones([1, 6, 4, 4], &device);
    let coarse_query = Tensor::<WgpuBackend, 4>::ones([1, 4, 2, 2], &device);
    let coarse_value = Tensor::<WgpuBackend, 4>::ones([1, 6, 2, 2], &device);
    let decay = Tensor::<WgpuBackend, 1>::ones([4], &device);
    let plan = CompiledStructuredPyramidRhoPlan::new(
        1,
        4,
        6,
        shape,
        LocalGridNeighborhood::moore(1),
        &device,
    );

    let mut patch_rho = Tensor::<WgpuBackend, 5>::zeros([1, 4, 6, 4, 4], &device);
    let mut coarse_rho = Tensor::<WgpuBackend, 5>::zeros([1, 4, 6, 2, 2], &device);
    let mut hub_rho = Tensor::<WgpuBackend, 4>::zeros([1, 2, 4, 6], &device);

    let mut snapshots = Vec::with_capacity(24);
    for step in 0..32 {
        let output = try_fused_structured_pyramid_rho_step_wgpu_with_plan(
            shape,
            StructuredPyramidRhoStepInput {
                patch_query: patch_query.clone(),
                patch_value: patch_value.clone(),
                coarse_query: coarse_query.clone(),
                coarse_value: coarse_value.clone(),
                patch_rho: patch_rho.clone(),
                coarse_rho: coarse_rho.clone(),
                hub_rho: hub_rho.clone(),
                patch_hub_weights: None,
                coarse_hub_weights: None,
                neighborhood: LocalGridNeighborhood::moore(1),
                decay: decay.clone(),
            },
            &plan,
        )
        .expect("fused structured pyramid output");
        patch_rho = output.next_patch_rho;
        coarse_rho = output.next_coarse_rho;
        hub_rho = output.next_hub_rho;
        let _ = WgpuBackend::sync(&device);
        WgpuBackend::memory_cleanup(&device);
        let _ = WgpuBackend::sync(&device);
        if step >= 8 {
            snapshots.push(memory_snapshot(&device));
        }
    }

    assert_memory_growth_bounded(
        "structured_pyramid_fused",
        &snapshots,
        256 * 1024 * 1024,
        64 * 1024 * 1024,
    );
}
