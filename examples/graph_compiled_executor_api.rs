use burn::tensor::{Tensor, TensorData};
use burn_dragon::api::graph::config::GraphTopologyConfig;
use burn_dragon::api::graph::execution::{GraphCompiledExecutor, GraphDragon, GraphDragonConfig};
use burn_dragon::api::graph::routing::{GraphCsrAdjacency, GraphTopologyRouting};
use burn_dragon::api::core::state::StructuredStepMode;
use burn_ndarray::NdArray;

fn main() {
    type Backend = NdArray<f32>;

    let device = <Backend as burn::tensor::backend::Backend>::Device::default();
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

    let _routing_spec = GraphTopologyConfig {
        cluster_count: 2,
        global_count: 1,
        ..Default::default()
    }
    .routing_spec_for(&routing)
    .expect("routing spec");

    let model = GraphDragon::<Backend>::new(
        GraphDragonConfig {
            embed_dim: 4,
            rank: 2,
            value_dim: 4,
            ..Default::default()
        },
        &device,
    );
    let node_obs = Tensor::<Backend, 3>::from_data(
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
    let cluster_obs = Tensor::<Backend, 3>::from_data(
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
        .expect("state init");
    let executor = GraphCompiledExecutor::<Backend>::new(routing, &device);
    let output = model
        .step_with_executor(state, &executor, StructuredStepMode::Predict)
        .expect("compiled graph step");

    println!(
        "node_state={:?}, cluster_state={:?}",
        output.state.node_state().shape().dims::<3>(),
        output.state.cluster_state().shape().dims::<3>()
    );
}
