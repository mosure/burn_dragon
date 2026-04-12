use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_cubecl::cubecl::Runtime;
use burn_ndarray::NdArray;
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};

use super::{
    ManifoldHyperConnectionCoefficientPolicy, ManifoldHyperConnections,
    ManifoldHyperConnectionsConfig, mhc_merge, mhc_passthrough, mhc_passthrough_with_coefficients,
    mhc_split,
};

type TestBackend = NdArray<f32>;
type WgpuBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type WgpuAutodiffBackend = Autodiff<WgpuBackend>;

#[derive(Clone, Copy, Debug)]
struct MemorySnapshot {
    bytes_in_use: u64,
    bytes_reserved: u64,
}

#[cfg(not(target_arch = "wasm32"))]
fn init_wgpu_runtime(device: &<WgpuBackend as Backend>::Device) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn memory_snapshot(device: &<WgpuBackend as Backend>::Device) -> MemorySnapshot {
    let usage = <WgpuRuntime as Runtime>::client(device)
        .memory_usage()
        .expect("wgpu memory usage");
    MemorySnapshot {
        bytes_in_use: usage.bytes_in_use,
        bytes_reserved: usage.bytes_reserved,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn assert_memory_growth_bounded(
    label: &str,
    snapshots: &[MemorySnapshot],
    max_reserved_growth: u64,
    max_in_use_growth: u64,
) {
    assert!(!snapshots.is_empty(), "{label}: expected memory snapshots");
    let first = snapshots.first().expect("first snapshot");
    let last = snapshots.last().expect("last snapshot");
    let reserved_growth = last.bytes_reserved.saturating_sub(first.bytes_reserved);
    let in_use_growth = last.bytes_in_use.saturating_sub(first.bytes_in_use);
    assert!(
        reserved_growth <= max_reserved_growth,
        "{label}: reserved bytes grew by {reserved_growth}, limit {max_reserved_growth}"
    );
    assert!(
        in_use_growth <= max_in_use_growth,
        "{label}: in-use bytes grew by {in_use_growth}, limit {max_in_use_growth}"
    );
}

#[test]
fn mhc_sinkhorn_rows_cols_sum_close_to_one() {
    let device = <TestBackend as Backend>::Device::default();
    let config = ManifoldHyperConnectionsConfig {
        enabled: true,
        num_streams: 3,
        num_views: 2,
        ..Default::default()
    };
    let mhc = ManifoldHyperConnections::<TestBackend>::new(&config, 0, &device);
    let h_res = mhc.coefficients().residual_weights;
    let row_sums = h_res
        .clone()
        .sum_dim(1)
        .to_data()
        .iter::<f32>()
        .collect::<Vec<_>>();
    let col_sums = h_res.sum_dim(0).to_data().iter::<f32>().collect::<Vec<_>>();
    for sum in row_sums.into_iter().chain(col_sums) {
        assert!((sum - 1.0).abs() < 1e-3, "sum not close to 1: {sum}");
    }
}

#[test]
fn mhc_coefficients_report_expected_shapes_and_policy() {
    let device = <TestBackend as Backend>::Device::default();
    let config = ManifoldHyperConnectionsConfig {
        enabled: true,
        num_streams: 2,
        num_views: 3,
        coefficient_policy: ManifoldHyperConnectionCoefficientPolicy::StaticSinkhorn,
        ..Default::default()
    };
    let mhc = ManifoldHyperConnections::<TestBackend>::new(&config, 1, &device);
    let coeffs = mhc.coefficients();
    assert_eq!(
        coeffs.residual_weights.shape().dims::<2>(),
        [config.num_streams, config.num_streams]
    );
    assert_eq!(
        coeffs.branch_input_weights.shape().dims::<2>(),
        [config.num_streams, config.num_views]
    );
    assert_eq!(
        coeffs
            .branch_output_weights
            .expect("branch output weights")
            .shape()
            .dims::<2>(),
        [config.num_views, config.num_streams]
    );
    assert_eq!(
        mhc.coefficient_policy(),
        ManifoldHyperConnectionCoefficientPolicy::StaticSinkhorn
    );
}

#[test]
fn mhc_width_connection_shapes_match_streams_and_views() {
    let device = <TestBackend as Backend>::Device::default();
    let config = ManifoldHyperConnectionsConfig {
        enabled: true,
        num_streams: 2,
        num_views: 2,
        ..Default::default()
    };
    let mhc = ManifoldHyperConnections::<TestBackend>::new(&config, 1, &device);
    let residuals = Tensor::<TestBackend, 4>::zeros([4, config.num_streams, 6, 8], &device);
    let output = mhc.width_connection(residuals);
    assert_eq!(
        output.branch_input.shape().dims::<4>(),
        [4, config.num_views, 6, 8]
    );
    assert_eq!(
        output.residuals_out.shape().dims::<4>(),
        [4, config.num_streams, 6, 8]
    );
    assert_eq!(
        output
            .coefficients
            .branch_output_weights
            .expect("expected beta")
            .shape()
            .dims::<2>(),
        [config.num_views, config.num_streams]
    );
}

#[test]
fn dynamic_positive_stream_coefficients_preserve_positive_constraints_per_token() {
    let device = <TestBackend as Backend>::Device::default();
    let config = ManifoldHyperConnectionsConfig {
        enabled: true,
        num_streams: 2,
        num_views: 1,
        coefficient_policy: ManifoldHyperConnectionCoefficientPolicy::DynamicPositive,
        ..Default::default()
    };
    let mhc =
        ManifoldHyperConnections::<TestBackend>::new_with_dense_dim(&config, 0, Some(4), &device);
    let residuals = Tensor::<TestBackend, 4>::from_data(
        TensorData::new(
            (0..32).map(|idx| idx as f32 / 16.0).collect::<Vec<_>>(),
            [2, 2, 2, 4],
        ),
        &device,
    );

    let coeffs = mhc.stream_coefficients(residuals);
    let residual_rows = coeffs
        .residual_weights
        .clone()
        .sum_dim(3)
        .into_data()
        .to_vec::<f32>()
        .expect("row sums");
    let residual_cols = coeffs
        .residual_weights
        .clone()
        .sum_dim(2)
        .into_data()
        .to_vec::<f32>()
        .expect("col sums");
    let alpha_sums = coeffs
        .branch_input_weights
        .clone()
        .sum_dim(2)
        .into_data()
        .to_vec::<f32>()
        .expect("alpha sums");
    let beta = coeffs.branch_output_weights.clone().expect("beta");
    let beta_sums = beta
        .clone()
        .sum_dim(2)
        .into_data()
        .to_vec::<f32>()
        .expect("beta sums");
    let alpha_values = coeffs
        .branch_input_weights
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("alpha values");
    let beta_values = beta
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("beta values");

    for sum in residual_rows.into_iter().chain(residual_cols) {
        assert!((sum - 1.0).abs() < 1e-3, "sum not close to 1: {sum}");
    }
    for value in alpha_values.into_iter().chain(beta_values) {
        assert!(
            (0.0..=1.0).contains(&value),
            "coefficient should stay in [0, 1]: {value}"
        );
    }
    for sum in alpha_sums.into_iter().chain(beta_sums) {
        assert!(
            sum.is_finite() && sum > 0.0,
            "positive mapping sum should stay finite: {sum}"
        );
    }
}

#[test]
fn dynamic_positive_stream_wrapper_uses_single_bdh_branch() {
    let device = <TestBackend as Backend>::Device::default();
    let config = ManifoldHyperConnectionsConfig {
        enabled: true,
        num_streams: 3,
        num_views: 1,
        coefficient_policy: ManifoldHyperConnectionCoefficientPolicy::DynamicPositive,
        ..Default::default()
    };
    let mhc =
        ManifoldHyperConnections::<TestBackend>::new_with_dense_dim(&config, 1, Some(5), &device);
    let residuals = Tensor::<TestBackend, 4>::zeros([4, config.num_streams, 6, 5], &device);
    let output = mhc.stream_width_connection(residuals.clone());
    assert_eq!(output.branch_input.shape().dims::<4>(), [4, 1, 6, 5]);
    assert_eq!(
        output.residuals_out.shape().dims::<4>(),
        [4, config.num_streams, 6, 5]
    );
    let merged = mhc.stream_depth_connection(
        output.branch_input,
        output.residuals_out,
        &output.coefficients,
    );
    assert_eq!(merged.shape().dims::<4>(), [4, config.num_streams, 6, 5]);
}

#[test]
fn dynamic_positive_stream_coefficients_start_from_nonuniform_static_priors() {
    let device = <TestBackend as Backend>::Device::default();
    let config = ManifoldHyperConnectionsConfig {
        enabled: true,
        num_streams: 3,
        num_views: 1,
        coefficient_policy: ManifoldHyperConnectionCoefficientPolicy::DynamicPositive,
        ..Default::default()
    };
    let mhc =
        ManifoldHyperConnections::<TestBackend>::new_with_dense_dim(&config, 1, Some(4), &device);
    let residuals = Tensor::<TestBackend, 4>::zeros([2, config.num_streams, 2, 4], &device);
    let coeffs = mhc.stream_coefficients(residuals);

    let alpha = coeffs
        .branch_input_weights
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("alpha");
    let beta = coeffs
        .branch_output_weights
        .expect("beta")
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("beta");

    assert!(
        alpha
            .windows(config.num_streams)
            .any(|window| { window.iter().any(|value| (*value - window[0]).abs() > 1e-4) }),
        "alpha should inherit non-uniform static priors"
    );
    assert!(
        beta.windows(config.num_streams)
            .any(|window| { window.iter().any(|value| (*value - window[0]).abs() > 1e-4) }),
        "beta should inherit non-uniform static priors"
    );
}

#[test]
fn dynamic_positive_bootstrap_streams_break_initial_symmetry() {
    let device = <TestBackend as Backend>::Device::default();
    let config = ManifoldHyperConnectionsConfig {
        enabled: true,
        num_streams: 3,
        num_views: 1,
        coefficient_policy: ManifoldHyperConnectionCoefficientPolicy::DynamicPositive,
        ..Default::default()
    };
    let mhc =
        ManifoldHyperConnections::<TestBackend>::new_with_dense_dim(&config, 0, Some(4), &device);
    let residuals = Tensor::<TestBackend, 4>::ones([1, 1, 2, 4], &device);
    let bootstrapped = mhc.bootstrap_streams(residuals);
    let stream0 = bootstrapped
        .clone()
        .slice_dim(1, 0..1)
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("stream0");
    let stream1 = bootstrapped
        .slice_dim(1, 1..2)
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("stream1");

    assert_ne!(
        stream0, stream1,
        "bootstrapped streams should not remain identical"
    );
}

#[test]
fn mhc_width_and_depth_with_explicit_coefficients_match_compatibility_path() {
    let device = <TestBackend as Backend>::Device::default();
    let config = ManifoldHyperConnectionsConfig {
        enabled: true,
        num_streams: 2,
        num_views: 2,
        ..Default::default()
    };
    let mhc = ManifoldHyperConnections::<TestBackend>::new(&config, 1, &device);
    let residuals = Tensor::<TestBackend, 4>::from_data(
        TensorData::new(
            (0..64).map(|idx| idx as f32 / 32.0).collect::<Vec<_>>(),
            [2, 2, 4, 4],
        ),
        &device,
    );

    let coeffs = mhc.coefficients();
    let explicit = mhc.width_connection_with_coefficients(residuals.clone(), &coeffs);
    let compatibility = mhc.width_connection(residuals.clone());
    assert_eq!(
        explicit
            .branch_input
            .clone()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("explicit branch"),
        compatibility
            .branch_input
            .clone()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("compat branch")
    );
    assert_eq!(
        explicit
            .residuals_out
            .clone()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("explicit residuals"),
        compatibility
            .residuals_out
            .clone()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("compat residuals")
    );

    let explicit_depth = mhc.depth_connection_with_coefficients(
        explicit.branch_input,
        explicit.residuals_out,
        &explicit.coefficients,
    );
    let compat_depth = mhc.depth_connection(
        compatibility.branch_input,
        compatibility.residuals_out,
        compatibility.coefficients.branch_output_weights,
    );
    assert_eq!(
        explicit_depth
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("explicit depth"),
        compat_depth
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("compat depth")
    );
}

#[test]
fn mhc_generic_mix_matches_manual_stream_weighting_for_reduce_case() {
    let device = <TestBackend as Backend>::Device::default();
    let config = ManifoldHyperConnectionsConfig {
        enabled: true,
        num_streams: 2,
        num_views: 1,
        dropout: 0.0,
        ..Default::default()
    };
    let mhc = ManifoldHyperConnections::<TestBackend>::new(&config, 0, &device);
    let residuals = Tensor::<TestBackend, 4>::from_data(
        TensorData::new(vec![1.0, 2.0, 10.0, 20.0], [1, 2, 1, 2]),
        &device,
    );
    let coeffs = super::ManifoldHyperConnectionCoefficients {
        residual_weights: Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![1.0, 0.0, 0.0, 1.0], [2, 2]),
            &device,
        ),
        branch_input_weights: Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![0.25, 0.75], [2, 1]),
            &device,
        ),
        branch_output_weights: None,
    };
    let output = mhc.width_connection_with_coefficients(residuals, &coeffs);
    assert_eq!(
        output
            .branch_input
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("branch input"),
        vec![7.75, 15.5]
    );
}

#[test]
fn mhc_passthrough_single_stream_single_view_matches_split_merge_path() {
    let device = <TestBackend as Backend>::Device::default();
    let config = ManifoldHyperConnectionsConfig {
        enabled: true,
        num_streams: 1,
        num_views: 1,
        dropout: 0.0,
        ..Default::default()
    };
    let mhc = ManifoldHyperConnections::<TestBackend>::new(&config, 0, &device);
    let residuals = Tensor::<TestBackend, 4>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]),
        &device,
    );
    let passthrough = mhc_passthrough(Some(&mhc), residuals.clone());
    let (branch_input, residuals_out, beta) = mhc_split(Some(&mhc), residuals);
    let merged = mhc_merge(Some(&mhc), branch_input, residuals_out, beta);
    assert_eq!(
        passthrough
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("passthrough"),
        merged
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("merged")
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn mhc_passthrough_with_explicit_coefficients_matches_one_shot_path() {
    let device = <TestBackend as Backend>::Device::default();
    let config = ManifoldHyperConnectionsConfig {
        enabled: true,
        num_streams: 4,
        num_views: 1,
        dropout: 0.0,
        ..Default::default()
    };
    let mhc = ManifoldHyperConnections::<TestBackend>::new(&config, 0, &device);
    let coefficients = mhc.coefficients();
    let residuals =
        Tensor::<TestBackend, 4>::random([2, 4, 8, 12], Distribution::Normal(0.0, 1.0), &device);

    let one_shot = mhc_passthrough(Some(&mhc), residuals.clone());
    let reused = mhc_passthrough_with_coefficients(Some(&mhc), residuals, Some(&coefficients));
    assert_eq!(
        one_shot
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("one-shot"),
        reused
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("reused")
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn mhc_wgpu_passthrough_memory_stays_bounded_across_repeated_calls() {
    let device = <WgpuBackend as Backend>::Device::default();
    init_wgpu_runtime(&device);
    let config = ManifoldHyperConnectionsConfig {
        enabled: true,
        num_streams: 4,
        num_views: 4,
        dropout: 0.0,
        ..Default::default()
    };
    let mhc = ManifoldHyperConnections::<WgpuBackend>::new(&config, 1, &device);
    let residuals =
        Tensor::<WgpuBackend, 4>::random([2, 4, 32, 48], Distribution::Normal(0.0, 1.0), &device);

    for _ in 0..2 {
        let _ = mhc_passthrough(Some(&mhc), residuals.clone());
        let _ = WgpuBackend::sync(&device);
        WgpuBackend::memory_cleanup(&device);
        let _ = WgpuBackend::sync(&device);
    }

    let mut snapshots = Vec::with_capacity(24);
    for step in 0..32 {
        let _ = mhc_passthrough(Some(&mhc), residuals.clone());
        let _ = WgpuBackend::sync(&device);
        WgpuBackend::memory_cleanup(&device);
        let _ = WgpuBackend::sync(&device);
        if step >= 8 {
            snapshots.push(memory_snapshot(&device));
        }
    }

    assert_memory_growth_bounded(
        "mhc_passthrough",
        &snapshots,
        256 * 1024 * 1024,
        64 * 1024 * 1024,
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn mhc_wgpu_width_and_depth_are_autodiff_stable_after_one_step() {
    let device = <WgpuAutodiffBackend as Backend>::Device::default();
    init_wgpu_runtime(&device);

    let config = ManifoldHyperConnectionsConfig {
        enabled: true,
        num_streams: 4,
        num_views: 1,
        dropout: 0.0,
        ..Default::default()
    };
    let reference = ManifoldHyperConnections::<WgpuAutodiffBackend>::new(&config, 0, &device);
    let cloned = reference
        .clone()
        .load_record(reference.clone().into_record());
    let residuals = Tensor::<WgpuAutodiffBackend, 4>::random(
        [2, 4, 16, 24],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let reference_loss = reference.passthrough(residuals.clone()).mean();
    let cloned_loss = cloned.passthrough(residuals).mean();
    let reference_value = reference_loss
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("reference loss")[0];
    let cloned_value = cloned_loss
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("cloned loss")[0];
    assert!((reference_value - cloned_value).abs() <= 1.0e-5);
}
