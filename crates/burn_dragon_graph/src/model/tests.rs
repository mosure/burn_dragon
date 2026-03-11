use super::*;
use crate::{CompiledGraphRouting, GraphCsrAdjacency, GraphTopologyRouting};
use burn::tensor::TensorData;
use burn::tensor::backend::Backend as BackendTrait;
use burn_ndarray::NdArray;
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};

type Backend = NdArray<f32>;
type WgpuBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

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
