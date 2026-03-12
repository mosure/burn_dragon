use super::*;
use crate::compiled_routing::CompiledGraphRouting;
use crate::{GraphCompiledExecutor, GraphCsrAdjacency, GraphTopologyRouting};
#[cfg(not(target_arch = "wasm32"))]
use burn::optim::{AdamWConfig, GradientsParams, LearningRate, Optimizer};
use burn::tensor::TensorData;
use burn::tensor::backend::Backend as BackendTrait;
#[cfg(not(target_arch = "wasm32"))]
use burn::tensor::Distribution;
#[cfg(not(target_arch = "wasm32"))]
use burn_autodiff::Autodiff;
use burn_cubecl::cubecl::Runtime;
use burn_ndarray::NdArray;
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};

type Backend = NdArray<f32>;
type WgpuBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
#[cfg(not(target_arch = "wasm32"))]
type WgpuAutodiffBackend = Autodiff<WgpuBackend>;

fn device() -> <Backend as BackendTrait>::Device {
    <Backend as BackendTrait>::Device::default()
}

fn seeded_model(device: &<Backend as BackendTrait>::Device) -> GraphDragon<Backend> {
    <Backend as BackendTrait>::seed(device, 7);
    GraphDragon::new(
        GraphDragonConfig {
            embed_dim: 4,
            rank: 2,
            value_dim: 4,
            predict_decay: 1.0,
            mode_embeddings: true,
        },
        device,
    )
}

#[test]
fn graph_dragon_config_exposes_paper_dimension_aliases() {
    let config = GraphDragonConfig {
        embed_dim: 96,
        rank: 24,
        value_dim: 64,
        predict_decay: 0.95,
        mode_embeddings: true,
    };

    assert_eq!(config.dense_space_dim(), 96);
    assert_eq!(config.neuron_space_dim(), 24);
    assert_eq!(config.recurrent_value_dim(), 64);
}

#[cfg(not(target_arch = "wasm32"))]
fn wgpu_device() -> <WgpuBackend as BackendTrait>::Device {
    static INIT: std::sync::Once = std::sync::Once::new();
    let device = <WgpuBackend as BackendTrait>::Device::default();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(&device, RuntimeOptions::default());
    });
    device
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
struct MemorySnapshot {
    reserved: u64,
    in_use: u64,
}

#[cfg(not(target_arch = "wasm32"))]
fn wgpu_memory_snapshot(device: &<WgpuBackend as BackendTrait>::Device) -> MemorySnapshot {
    let usage = <WgpuRuntime as Runtime>::client(device).memory_usage();
    MemorySnapshot {
        reserved: usage.bytes_reserved,
        in_use: usage.bytes_in_use,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn graph_rollout_loss<B: burn::tensor::backend::AutodiffBackend>(
    model: &GraphDragon<B>,
    state: GraphTopologyState<B>,
    routing: &GraphTopologyRouting,
    compiled: Option<&CompiledGraphRouting<B>>,
) -> Tensor<B, 1> {
    let state = match compiled {
        Some(compiled) => model
            .rollout_compiled(state, compiled, 3, StructuredStepMode::Predict)
            .expect("compiled rollout should succeed"),
        None => model
            .rollout(state, routing, 3, StructuredStepMode::Predict)
            .expect("reference rollout should succeed"),
    };

    state.node_state().tanh().powf_scalar(2.0).mean()
        + state.cluster_state().tanh().powf_scalar(2.0).mean()
        + state.node_rho().tanh().powf_scalar(2.0).mean()
        + state.cluster_rho().tanh().powf_scalar(2.0).mean()
        + state.global_rho().tanh().powf_scalar(2.0).mean()
}

#[cfg(not(target_arch = "wasm32"))]
fn max_abs_diff<B: BackendTrait, const D: usize>(lhs: Tensor<B, D>, rhs: Tensor<B, D>) -> f32 {
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
    lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs_value, rhs_value)| (*lhs_value - *rhs_value).abs())
        .fold(0.0_f32, f32::max)
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

fn routing() -> GraphTopologyRouting {
    GraphTopologyRouting::new(
        GraphCsrAdjacency::try_from_edges(3, 3, &[(0, 1), (1, 0), (1, 2), (2, 2)])
            .expect("valid node adjacency"),
    )
    .expect("valid routing")
    .with_cluster_assignments(2, &[0, 0, 1])
    .expect("valid cluster assignments")
    .with_node_global_assignments(1, &[0, 0, 0])
    .expect("valid node/global assignments")
    .with_cluster_global_assignments(1, &[0, 0])
    .expect("valid cluster/global assignments")
}

fn observations(
    device: &<Backend as BackendTrait>::Device,
) -> (Tensor<Backend, 3>, Tensor<Backend, 3>) {
    (
        Tensor::<Backend, 3>::from_data(
            TensorData::new(
                vec![
                    1.0, 0.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, 0.0, //
                    0.0, 0.0, 1.0, 0.0,
                ],
                [1, 3, 4],
            ),
            device,
        ),
        Tensor::<Backend, 3>::from_data(
            TensorData::new(
                vec![
                    1.0, 1.0, 0.0, 0.0, //
                    0.0, 1.0, 1.0, 0.0,
                ],
                [1, 2, 4],
            ),
            device,
        ),
    )
}

#[test]
fn graph_dragon_state_from_observations_matches_layout_and_zeros_rho() {
    let device = device();
    let model = seeded_model(&device);
    let (node_obs, cluster_obs) = observations(&device);
    let state = model
        .state_from_observations(&routing(), node_obs.clone(), cluster_obs.clone())
        .expect("state init should succeed");

    assert_eq!(state.layout().node_count, 3);
    assert_eq!(state.layout().cluster_count, 2);
    assert_eq!(state.layout().global_count, 1);
    assert_eq!(state.node_state().shape().dims::<3>(), [1, 3, 4]);
    assert_eq!(state.node_dense_state().shape().dims::<3>(), [1, 3, 4]);
    assert_eq!(state.cluster_state().shape().dims::<3>(), [1, 2, 4]);
    assert_eq!(state.cluster_dense_state().shape().dims::<3>(), [1, 2, 4]);
    assert_eq!(state.node_rho().shape().dims::<4>(), [1, 2, 4, 3]);
    assert_eq!(state.cluster_rho().shape().dims::<4>(), [1, 2, 4, 2]);
    assert_eq!(state.global_rho().shape().dims::<4>(), [1, 1, 2, 4]);
    assert_eq!(
        state
            .node_rho()
            .into_data()
            .to_vec::<f32>()
            .expect("node rho data")
            .iter()
            .sum::<f32>(),
        0.0
    );
    assert_eq!(
        state
            .global_rho()
            .into_data()
            .to_vec::<f32>()
            .expect("global rho data")
            .iter()
            .sum::<f32>(),
        0.0
    );
}

#[test]
fn graph_dragon_state_with_observations_preserves_rho_and_replaces_dense_state() {
    let device = device();
    let model = seeded_model(&device);
    let routing = routing();
    let (node_obs, cluster_obs) = observations(&device);
    let state = model
        .state_from_observations(&routing, node_obs, cluster_obs)
        .expect("state init should succeed");
    let replacement_node = Tensor::<Backend, 3>::ones([1, 3, 4], &device);
    let replacement_cluster = Tensor::<Backend, 3>::ones([1, 2, 4], &device).mul_scalar(2.0);
    let preserved_node_rho = state.node_rho();
    let preserved_cluster_rho = state.cluster_rho();
    let preserved_global_rho = state.global_rho();
    let replaced = model
        .state_with_observations(state, replacement_node.clone(), replacement_cluster.clone())
        .expect("observation replacement should succeed");

    assert_eq!(
        replaced
            .node_state()
            .into_data()
            .to_vec::<f32>()
            .expect("node state"),
        replacement_node
            .into_data()
            .to_vec::<f32>()
            .expect("replacement node")
    );
    assert_eq!(
        replaced
            .cluster_state()
            .into_data()
            .to_vec::<f32>()
            .expect("cluster state"),
        replacement_cluster
            .into_data()
            .to_vec::<f32>()
            .expect("replacement cluster")
    );
    assert_eq!(
        replaced
            .node_rho()
            .into_data()
            .to_vec::<f32>()
            .expect("node rho"),
        preserved_node_rho
            .into_data()
            .to_vec::<f32>()
            .expect("preserved node rho")
    );
    assert_eq!(
        replaced
            .cluster_rho()
            .into_data()
            .to_vec::<f32>()
            .expect("cluster rho"),
        preserved_cluster_rho
            .into_data()
            .to_vec::<f32>()
            .expect("preserved cluster rho")
    );
    assert_eq!(
        replaced
            .global_rho()
            .into_data()
            .to_vec::<f32>()
            .expect("global rho"),
        preserved_global_rho
            .into_data()
            .to_vec::<f32>()
            .expect("preserved global rho")
    );
}

#[test]
fn graph_dragon_mode_embeddings_distinguish_refine_and_predict_when_decay_is_one() {
    let device = device();
    let model = seeded_model(&device);
    let routing = routing();
    let (node_obs, cluster_obs) = observations(&device);
    let state = model
        .state_from_observations(&routing, node_obs, cluster_obs)
        .expect("state init should succeed");

    let refined = model
        .step(state.clone(), &routing, StructuredStepMode::Refine)
        .expect("refine should succeed");
    let predicted = model
        .step(state, &routing, StructuredStepMode::Predict)
        .expect("predict should succeed");

    assert_ne!(
        refined
            .state
            .node_state()
            .into_data()
            .to_vec::<f32>()
            .expect("refined node state"),
        predicted
            .state
            .node_state()
            .into_data()
            .to_vec::<f32>()
            .expect("predicted node state")
    );
}

#[test]
fn graph_dragon_observe_and_rollout_track_temporal_axis() {
    let device = device();
    let model = seeded_model(&device);
    let routing = routing();
    let (node_obs, cluster_obs) = observations(&device);
    let state = model
        .state_from_observations(&routing, node_obs.clone(), cluster_obs.clone())
        .expect("state init should succeed");

    let predicted = model
        .predict(state.clone(), &routing)
        .expect("predict should succeed");
    assert_eq!(predicted.state.temporal_position(), 1);
    assert_eq!(predicted.state.prediction_age(), 1);

    let observed = model
        .observe(predicted.state, &routing, node_obs, cluster_obs)
        .expect("observe should succeed");
    assert_eq!(observed.state.temporal_position(), 1);
    assert_eq!(observed.state.prediction_age(), 0);

    let rolled = model
        .rollout(state, &routing, 3, StructuredStepMode::Predict)
        .expect("rollout should succeed");
    assert_eq!(rolled.temporal_position(), 3);
    assert_eq!(rolled.prediction_age(), 3);
}

#[test]
fn graph_dragon_step_compiled_matches_reference_on_ndarray_backend() {
    let device = device();
    let model = seeded_model(&device);
    let routing = routing();
    let compiled = CompiledGraphRouting::<Backend>::new(routing.clone(), &device);
    let (node_obs, cluster_obs) = observations(&device);
    let state = model
        .state_from_observations(&routing, node_obs, cluster_obs)
        .expect("state init should succeed");

    let reference = model
        .step(state.clone(), &routing, StructuredStepMode::Predict)
        .expect("reference step should succeed");
    let compiled_step = model
        .step_compiled(state, &compiled, StructuredStepMode::Predict)
        .expect("compiled step should succeed");

    assert_eq!(
        reference
            .state
            .node_state()
            .into_data()
            .to_vec::<f32>()
            .expect("reference node state"),
        compiled_step
            .state
            .node_state()
            .into_data()
            .to_vec::<f32>()
            .expect("compiled node state")
    );
    assert_eq!(
        reference
            .state
            .global_rho()
            .into_data()
            .to_vec::<f32>()
            .expect("reference global rho"),
        compiled_step
            .state
            .global_rho()
            .into_data()
            .to_vec::<f32>()
            .expect("compiled global rho")
    );
}

#[test]
fn graph_compiled_executor_matches_direct_compiled_step_on_ndarray_backend() {
    let device = device();
    let routing = routing();
    let model = seeded_model(&device);
    let (node_obs, cluster_obs) = observations(&device);
    let state = model
        .state_from_observations(&routing, node_obs, cluster_obs)
        .expect("state init");
    let compiled = CompiledGraphRouting::<Backend>::new(routing.clone(), &device);
    let executor = GraphCompiledExecutor::<Backend>::new(routing, &device);

    let direct = model
        .step_compiled(state.clone(), &compiled, StructuredStepMode::Predict)
        .expect("direct compiled step");
    let wrapped = model
        .step_with_executor(state, &executor, StructuredStepMode::Predict)
        .expect("executor step");

    assert!(max_abs_diff(direct.state.node_state(), wrapped.state.node_state()) <= 1e-5);
    assert!(max_abs_diff(
        direct.state.cluster_state(),
        wrapped.state.cluster_state()
    ) <= 1e-5);
    assert!(max_abs_diff(direct.state.node_rho(), wrapped.state.node_rho()) <= 1e-5);
}

#[test]
fn compiled_cluster_pool_weights_match_incoming_adjacency_pooling() {
    let device = device();
    let routing = routing();
    let compiled = CompiledGraphRouting::<Backend>::new(routing.clone(), &device);
    let route = compiled
        .node_to_cluster()
        .expect("compiled node to cluster route");
    let source = Tensor::<Backend, 3>::from_data(
        TensorData::new(
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
                5.0, 6.0,
            ],
            [1, 3, 2],
        ),
        &device,
    );

    let pooled_from_incoming = super::support::pool_dense_from_incoming(
        source.clone(),
        route.incoming_adjacency(),
        routing.layout().expect("layout").cluster_count,
    );
    let pooled_from_weights =
        super::support::pool_dense_with_weights(source, route.incoming_pool_weights().clone());

    assert_eq!(
        pooled_from_incoming
            .into_data()
            .to_vec::<f32>()
            .expect("incoming pooled"),
        pooled_from_weights
            .into_data()
            .to_vec::<f32>()
            .expect("weighted pooled")
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn graph_dragon_step_compiled_matches_reference_on_wgpu_backend() {
    let device = wgpu_device();
    <WgpuBackend as BackendTrait>::seed(&device, 7);

    let model = GraphDragon::new(
        GraphDragonConfig {
            embed_dim: 4,
            rank: 2,
            value_dim: 4,
            predict_decay: 0.95,
            mode_embeddings: true,
        },
        &device,
    );
    let routing = GraphTopologyRouting::new(
        GraphCsrAdjacency::try_from_edges(3, 3, &[(0, 1), (1, 0), (1, 2), (2, 2)])
            .expect("valid node adjacency"),
    )
    .expect("valid routing")
    .with_cluster_assignments(2, &[0, 0, 1])
    .expect("valid cluster assignments")
    .with_node_global_assignments(1, &[0, 0, 0])
    .expect("valid node/global assignments")
    .with_cluster_global_assignments(1, &[0, 0])
    .expect("valid cluster/global assignments");
    let compiled = CompiledGraphRouting::<WgpuBackend>::new(routing.clone(), &device);
    let node_obs = Tensor::<WgpuBackend, 3>::from_data(
        TensorData::new(
            vec![
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
            [1, 3, 4],
        ),
        &device,
    );
    let cluster_obs = Tensor::<WgpuBackend, 3>::from_data(
        TensorData::new(
            vec![
                1.0, 1.0, 0.0, 0.0, //
                0.0, 1.0, 1.0, 0.0,
            ],
            [1, 2, 4],
        ),
        &device,
    );
    let state = model
        .state_from_observations(&routing, node_obs, cluster_obs)
        .expect("state init should succeed");

    let reference = model
        .step(state.clone(), &routing, StructuredStepMode::Predict)
        .expect("reference step should succeed");
    let fused = model
        .step_compiled(state, &compiled, StructuredStepMode::Predict)
        .expect("compiled step should succeed");

    let reference_node = reference
        .state
        .node_state()
        .into_data()
        .to_vec::<f32>()
        .expect("reference node state");
    let fused_node = fused
        .state
        .node_state()
        .into_data()
        .to_vec::<f32>()
        .expect("fused node state");
    for (lhs, rhs) in reference_node.iter().zip(fused_node.iter()) {
        assert!(
            (lhs - rhs).abs() <= 3e-4,
            "node diff too large: {lhs} vs {rhs}"
        );
    }

    let reference_global = reference
        .state
        .global_rho()
        .into_data()
        .to_vec::<f32>()
        .expect("reference global rho");
    let fused_global = fused
        .state
        .global_rho()
        .into_data()
        .to_vec::<f32>()
        .expect("fused global rho");
    for (lhs, rhs) in reference_global.iter().zip(fused_global.iter()) {
        assert!(
            (lhs - rhs).abs() <= 4e-4,
            "global rho diff too large: {lhs} vs {rhs}"
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn graph_dragon_rollout_compiled_matches_reference_on_wgpu_backend() {
    let device = wgpu_device();
    <WgpuBackend as BackendTrait>::seed(&device, 11);

    let model = GraphDragon::new(
        GraphDragonConfig {
            embed_dim: 6,
            rank: 3,
            value_dim: 6,
            predict_decay: 0.97,
            mode_embeddings: true,
        },
        &device,
    );
    let routing = GraphTopologyRouting::new(
        GraphCsrAdjacency::try_from_edges(4, 4, &[(0, 0), (0, 1), (1, 2), (2, 1), (2, 3), (3, 0)])
            .expect("valid node adjacency"),
    )
    .expect("valid routing")
    .with_cluster_assignments(2, &[0, 0, 1, 1])
    .expect("valid cluster assignments")
    .with_node_global_assignments(1, &[0, 0, 0, 0])
    .expect("valid node/global assignments")
    .with_cluster_global_assignments(1, &[0, 0])
    .expect("valid cluster/global assignments");
    let compiled = CompiledGraphRouting::<WgpuBackend>::new(routing.clone(), &device);
    let node_obs = Tensor::<WgpuBackend, 3>::from_data(
        TensorData::new(
            vec![
                1.0, 0.0, 0.0, 0.0, 0.5, 0.0, //
                0.0, 1.0, 0.0, 0.0, 0.0, 0.5, //
                0.0, 0.0, 1.0, 0.0, 0.5, 0.5, //
                0.0, 0.0, 0.0, 1.0, 0.5, 1.0,
            ],
            [1, 4, 6],
        ),
        &device,
    );
    let cluster_obs = Tensor::<WgpuBackend, 3>::from_data(
        TensorData::new(
            vec![
                1.0, 1.0, 0.0, 0.0, 0.25, 0.25, //
                0.0, 1.0, 1.0, 0.0, 0.75, 0.75,
            ],
            [1, 2, 6],
        ),
        &device,
    );
    let state = model
        .state_from_observations(&routing, node_obs, cluster_obs)
        .expect("state init should succeed");

    let reference = model
        .rollout(state.clone(), &routing, 3, StructuredStepMode::Predict)
        .expect("reference rollout should succeed");
    let fused = model
        .rollout_compiled(state, &compiled, 3, StructuredStepMode::Predict)
        .expect("compiled rollout should succeed");

    let reference_node = reference
        .node_state()
        .into_data()
        .to_vec::<f32>()
        .expect("reference node state");
    let fused_node = fused
        .node_state()
        .into_data()
        .to_vec::<f32>()
        .expect("fused node state");
    for (lhs, rhs) in reference_node.iter().zip(fused_node.iter()) {
        assert!(
            (lhs - rhs).abs() <= 5e-4,
            "rollout node diff too large: {lhs} vs {rhs}"
        );
    }

    let reference_cluster = reference
        .cluster_rho()
        .into_data()
        .to_vec::<f32>()
        .expect("reference cluster rho");
    let fused_cluster = fused
        .cluster_rho()
        .into_data()
        .to_vec::<f32>()
        .expect("fused cluster rho");
    for (lhs, rhs) in reference_cluster.iter().zip(fused_cluster.iter()) {
        assert!(
            (lhs - rhs).abs() <= 6e-4,
            "rollout cluster rho diff too large: {lhs} vs {rhs}"
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn graph_dragon_rollout_compiled_memory_stays_bounded_on_wgpu_backend() {
    let device = wgpu_device();
    <WgpuBackend as BackendTrait>::seed(&device, 29);

    let model = GraphDragon::new(
        GraphDragonConfig {
            embed_dim: 32,
            rank: 8,
            value_dim: 32,
            predict_decay: 0.97,
            mode_embeddings: true,
        },
        &device,
    );
    let routing = GraphTopologyRouting::new(
        GraphCsrAdjacency::try_from_edges(
            32,
            32,
            &[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 0),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 8),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 12),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 16),
                (20, 21),
                (21, 22),
                (22, 23),
                (23, 20),
                (24, 25),
                (25, 26),
                (26, 27),
                (27, 24),
                (28, 29),
                (29, 30),
                (30, 31),
                (31, 28),
            ],
        )
        .expect("valid node adjacency"),
    )
    .expect("valid routing")
    .with_cluster_assignments(8, &[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7])
    .expect("valid cluster assignments")
    .with_node_global_assignments(2, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    .expect("valid node/global assignments")
    .with_cluster_global_assignments(2, &[0, 0, 1, 1, 0, 0, 1, 1])
    .expect("valid cluster/global assignments");
    let compiled = CompiledGraphRouting::<WgpuBackend>::new(routing.clone(), &device);
    let node_obs = Tensor::<WgpuBackend, 3>::ones([2, 32, 32], &device);
    let cluster_obs = Tensor::<WgpuBackend, 3>::ones([2, 8, 32], &device);
    let mut state = model
        .state_from_observations(&routing, node_obs, cluster_obs)
        .expect("state init");

    for _ in 0..2 {
        state = model
            .rollout_compiled(state, &compiled, 4, StructuredStepMode::Predict)
            .expect("compiled rollout");
    }
    let _ = WgpuBackend::sync(&device);
    WgpuBackend::memory_cleanup(&device);
    let _ = WgpuBackend::sync(&device);

    let mut snapshots = Vec::with_capacity(24);
    for step in 0..32 {
        state = model
            .rollout_compiled(state, &compiled, 4, StructuredStepMode::Predict)
            .expect("compiled rollout");
        let _ = WgpuBackend::sync(&device);
        WgpuBackend::memory_cleanup(&device);
        let _ = WgpuBackend::sync(&device);
        if step >= 8 {
            snapshots.push(wgpu_memory_snapshot(&device));
        }
    }

    assert_memory_growth_bounded(
        "graph_rollout_compiled_wgpu",
        &snapshots,
        256 * 1024 * 1024,
        64 * 1024 * 1024,
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn graph_dragon_rollout_compiled_autodiff_matches_reference_after_one_step() {
    let device = <WgpuAutodiffBackend as BackendTrait>::Device::default();
    burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(&device, RuntimeOptions::default());
    <WgpuAutodiffBackend as BackendTrait>::seed(&device, 1_337);

    let reference = GraphDragon::new(
        GraphDragonConfig {
            embed_dim: 8,
            rank: 4,
            value_dim: 8,
            predict_decay: 0.95,
            mode_embeddings: true,
        },
        &device,
    );
    let fused = GraphDragon::new(
        GraphDragonConfig {
            embed_dim: 8,
            rank: 4,
            value_dim: 8,
            predict_decay: 0.95,
            mode_embeddings: true,
        },
        &device,
    )
    .load_record(reference.clone().into_record());
    let routing = GraphTopologyRouting::new(
        GraphCsrAdjacency::try_from_edges(8, 8, &[(0, 1), (1, 0), (1, 2), (2, 2), (3, 5), (4, 4), (5, 6), (6, 7)])
            .expect("valid node adjacency"),
    )
    .expect("valid routing")
    .with_cluster_assignments(3, &[0, 0, 1, 1, 1, 2, 2, 2])
    .expect("valid cluster assignments")
    .with_node_global_assignments(2, &[0, 0, 0, 0, 1, 1, 1, 1])
    .expect("valid node/global assignments")
    .with_cluster_global_assignments(2, &[0, 0, 1])
    .expect("valid cluster/global assignments");
    let compiled = CompiledGraphRouting::<WgpuAutodiffBackend>::new(routing.clone(), &device);
    let node_obs = Tensor::<WgpuAutodiffBackend, 3>::random(
        [2, 8, 8],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let cluster_obs = Tensor::<WgpuAutodiffBackend, 3>::random(
        [2, 3, 8],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let reference_state = reference
        .state_from_observations(&routing, node_obs.clone(), cluster_obs.clone())
        .expect("reference state");
    let fused_state = fused
        .state_from_observations(&routing, node_obs.clone(), cluster_obs.clone())
        .expect("fused state");

    let mut reference_optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<WgpuAutodiffBackend, GraphDragon<WgpuAutodiffBackend>>();
    let mut fused_optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<WgpuAutodiffBackend, GraphDragon<WgpuAutodiffBackend>>();
    let lr: LearningRate = 1e-3;

    let reference_loss = graph_rollout_loss(&reference, reference_state, &routing, None);
    let reference_loss_value = reference_loss
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("reference loss")[0];
    let reference_grads = GradientsParams::from_grads(reference_loss.backward(), &reference);
    let reference = reference_optimizer.step(lr, reference, reference_grads);

    let fused_loss = graph_rollout_loss(&fused, fused_state, &routing, Some(&compiled));
    let fused_loss_value = fused_loss
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("fused loss")[0];
    let fused_grads = GradientsParams::from_grads(fused_loss.backward(), &fused);
    let fused = fused_optimizer.step(lr, fused, fused_grads);

    assert!((reference_loss_value - fused_loss_value).abs() <= 5e-2);

    let reference_state = reference
        .rollout(
            reference
                .state_from_observations(&routing, node_obs.clone(), cluster_obs.clone())
                .expect("reference state"),
            &routing,
            3,
            StructuredStepMode::Predict,
        )
        .expect("reference rollout");
    let fused_state = fused
        .rollout_compiled(
            fused.state_from_observations(&routing, node_obs, cluster_obs)
                .expect("fused state"),
            &compiled,
            3,
            StructuredStepMode::Predict,
        )
        .expect("fused rollout");

    assert!(max_abs_diff(fused_state.node_state(), reference_state.node_state()) <= 8e-2);
    assert!(max_abs_diff(fused_state.cluster_state(), reference_state.cluster_state()) <= 8e-2);
    assert!(max_abs_diff(fused_state.node_rho(), reference_state.node_rho()) <= 5e-2);
    assert!(max_abs_diff(fused_state.cluster_rho(), reference_state.cluster_rho()) <= 5e-2);
    assert!(max_abs_diff(fused_state.global_rho(), reference_state.global_rho()) <= 5e-2);
}
