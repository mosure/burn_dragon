use std::error::Error;
use std::fmt::{Display, Formatter};

use burn::prelude::*;
use burn::tensor::TensorData;
use burn_dragon_core::{
    StructuredStepMode, structured_predict_decay, target_major_decay_add,
    target_major_identity_read, target_major_identity_write, target_major_outer_product,
};

use crate::{
    GraphCsrAdjacency, GraphRoutingError, GraphTopologyRouting, GraphTopologyState,
    state::{rho_from_target_major, rho_to_target_major},
};

/// Inputs for one graph recurrent step.
///
/// Paper mapping:
/// - `node_query` / `cluster_query` are the graph adapter's `x_neuron` projections used to read
///   and write `rho`
/// - `node_value` / `cluster_value` are the dense/write-side values paired with those neuron-space
///   activations
#[derive(Clone)]
pub struct GraphStepInputs<B: Backend> {
    pub node_query: Tensor<B, 3>,
    pub node_value: Tensor<B, 3>,
    pub cluster_query: Tensor<B, 3>,
    pub cluster_value: Tensor<B, 3>,
}

impl<B: Backend> GraphStepInputs<B> {
    pub fn node_x_neuron(&self) -> &Tensor<B, 3> {
        &self.node_query
    }

    pub fn node_write_value(&self) -> &Tensor<B, 3> {
        &self.node_value
    }

    pub fn cluster_x_neuron(&self) -> &Tensor<B, 3> {
        &self.cluster_query
    }

    pub fn cluster_write_value(&self) -> &Tensor<B, 3> {
        &self.cluster_value
    }
}

/// Dense-space recurrent readouts produced from graph `rho` banks during a single step.
#[derive(Clone)]
pub struct GraphStepReadouts<B: Backend> {
    pub node_from_node: Tensor<B, 3>,
    pub node_from_cluster: Option<Tensor<B, 3>>,
    pub node_from_global: Option<Tensor<B, 3>>,
    pub cluster_from_global: Option<Tensor<B, 3>>,
}

impl<B: Backend> GraphStepReadouts<B> {
    pub fn node_local_a_dense(&self) -> &Tensor<B, 3> {
        &self.node_from_node
    }

    pub fn node_context_a_dense(&self) -> Option<&Tensor<B, 3>> {
        self.node_from_cluster.as_ref()
    }

    pub fn node_global_a_dense(&self) -> Option<&Tensor<B, 3>> {
        self.node_from_global.as_ref()
    }

    pub fn cluster_global_a_dense(&self) -> Option<&Tensor<B, 3>> {
        self.cluster_from_global.as_ref()
    }
}

/// Output of one graph recurrent step.
///
/// `state` carries the persistent dense-state + `rho` contract across calls, while `readouts`
/// expose the transient dense-space recurrent reads (`a_dense`) used to form the next activation.
#[derive(Clone)]
pub struct GraphStepOutput<B: Backend> {
    pub state: GraphTopologyState<B>,
    pub readouts: GraphStepReadouts<B>,
}

/// Recurrent decay policy for graph `rho` updates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GraphRhoStepConfig {
    pub predict_decay: f32,
}

impl Default for GraphRhoStepConfig {
    fn default() -> Self {
        Self {
            predict_decay: 0.95,
        }
    }
}

impl GraphRhoStepConfig {
    fn decay_for_mode(self, mode: StructuredStepMode) -> f32 {
        structured_predict_decay(mode, self.predict_decay)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GraphExecutionError {
    Routing(GraphRoutingError),
    InvalidPredictDecay {
        value: f32,
    },
    BatchMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    CountMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    RankMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    ValueDimMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    DenseDimMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl Display for GraphExecutionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Routing(err) => write!(f, "{err}"),
            Self::InvalidPredictDecay { value } => {
                write!(f, "predict decay must be in [0, 1], got {value}")
            }
            Self::BatchMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "batch mismatch for {field}: expected {expected}, got {actual}"
            ),
            Self::CountMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "count mismatch for {field}: expected {expected}, got {actual}"
            ),
            Self::RankMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "rank mismatch for {field}: expected {expected}, got {actual}"
            ),
            Self::ValueDimMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "value-dim mismatch for {field}: expected {expected}, got {actual}"
            ),
            Self::DenseDimMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "dense-dim mismatch for {field}: expected {expected}, got {actual}"
            ),
        }
    }
}

impl Error for GraphExecutionError {}

impl From<GraphRoutingError> for GraphExecutionError {
    fn from(value: GraphRoutingError) -> Self {
        Self::Routing(value)
    }
}

pub fn graph_reference_step<B: Backend>(
    state: GraphTopologyState<B>,
    routing: &GraphTopologyRouting,
    inputs: GraphStepInputs<B>,
    mode: StructuredStepMode,
    config: GraphRhoStepConfig,
) -> Result<GraphStepOutput<B>, GraphExecutionError> {
    validate_inputs(&state, &inputs, config)?;

    let layout = state.layout();
    let node_state = state.node_state();
    let cluster_state = state.cluster_state();
    let node_rho = rho_to_target_major(state.node_rho());
    let cluster_rho = rho_to_target_major(state.cluster_rho());
    let global_rho = state.global_rho();

    let node_from_node = sparse_read(
        &inputs.node_query,
        &node_rho,
        routing.node_neighbors(),
        layout.node_count,
        layout.node_count,
    );
    let node_from_cluster = routing.node_to_cluster().map(|route| {
        sparse_read(
            &inputs.node_query,
            &cluster_rho,
            route,
            layout.node_count,
            layout.cluster_count,
        )
    });
    let node_from_global = routing.node_to_global().map(|route| {
        sparse_read(
            &inputs.node_query,
            &global_rho,
            route,
            layout.node_count,
            layout.global_count,
        )
    });
    let cluster_from_global = routing.cluster_to_global().map(|route| {
        sparse_read(
            &inputs.cluster_query,
            &global_rho,
            route,
            layout.cluster_count,
            layout.global_count,
        )
    });

    let decay = config.decay_for_mode(mode);
    let decay_tensor = Tensor::<B, 1>::from_data(
        TensorData::new(vec![decay], [1]),
        &inputs.node_query.device(),
    );
    let next_node_rho = target_major_identity_write(
        node_rho,
        inputs.node_query.clone(),
        inputs.node_value.clone(),
        decay_tensor.clone(),
    );
    let next_cluster_rho = {
        let self_update = direct_outer(&inputs.cluster_query, &inputs.cluster_value);
        let node_update = routing
            .node_to_cluster()
            .map(|route| {
                sparse_write_aggregate(
                    &inputs.node_query,
                    &inputs.node_value,
                    route,
                    layout.cluster_count,
                )
            })
            .unwrap_or_else(|| {
                zeros_target_major::<B>(
                    layout.cluster_count,
                    inputs.cluster_query.shape().dims::<3>()[2],
                    inputs.cluster_value.shape().dims::<3>()[2],
                    &inputs.cluster_query.device(),
                    inputs.cluster_query.shape().dims::<3>()[0],
                )
            });
        target_major_decay_add(
            cluster_rho,
            self_update.add(node_update),
            decay_tensor.clone(),
        )
    };
    let next_global_rho = {
        let [batch, global_count, rank, value_dim] = global_rho.shape().dims::<4>();
        let mut update =
            Tensor::<B, 4>::zeros([batch, global_count, rank, value_dim], &global_rho.device());
        if let Some(route) = routing.node_to_global() {
            let node_update =
                sparse_write_aggregate(&inputs.node_query, &inputs.node_value, route, global_count);
            update = update.add(node_update);
        }
        if let Some(route) = routing.cluster_to_global() {
            let cluster_update = sparse_write_aggregate(
                &inputs.cluster_query,
                &inputs.cluster_value,
                route,
                global_count,
            );
            update = update.add(cluster_update);
        }
        target_major_decay_add(global_rho, update, decay_tensor.clone()).reshape([
            batch,
            global_count,
            rank,
            value_dim,
        ])
    };

    let mut next_state = GraphTopologyState::from_parts(
        node_state,
        cluster_state,
        rho_from_target_major(next_node_rho),
        rho_from_target_major(next_cluster_rho),
        next_global_rho,
        state.temporal_position(),
        state.prediction_age(),
    );

    let temporal_dt = mode.temporal_dt();
    if temporal_dt > 0 {
        let inner = next_state.into_topology_state();
        next_state = GraphTopologyState::from_topology_state(
            burn_dragon_core::StructuredTopologyState {
                temporal_position: inner.temporal_position.saturating_add(temporal_dt),
                prediction_age: inner.prediction_age.saturating_add(temporal_dt),
                ..inner
            },
            layout,
        );
    } else if mode.resets_prediction_age() {
        let inner = next_state.into_topology_state();
        next_state = GraphTopologyState::from_topology_state(
            burn_dragon_core::StructuredTopologyState {
                prediction_age: 0,
                ..inner
            },
            layout,
        );
    }

    Ok(GraphStepOutput {
        state: next_state,
        readouts: GraphStepReadouts {
            node_from_node,
            node_from_cluster,
            node_from_global,
            cluster_from_global,
        },
    })
}

fn validate_inputs<B: Backend>(
    state: &GraphTopologyState<B>,
    inputs: &GraphStepInputs<B>,
    config: GraphRhoStepConfig,
) -> Result<(), GraphExecutionError> {
    if !(0.0..=1.0).contains(&config.predict_decay) {
        return Err(GraphExecutionError::InvalidPredictDecay {
            value: config.predict_decay,
        });
    }

    let layout = state.layout();
    let [node_batch, node_count, node_rank] = inputs.node_query.shape().dims::<3>();
    let [node_value_batch, node_value_count, node_value_dim] =
        inputs.node_value.shape().dims::<3>();
    let [cluster_batch, cluster_count, cluster_rank] = inputs.cluster_query.shape().dims::<3>();
    let [cluster_value_batch, cluster_value_count, cluster_value_dim] =
        inputs.cluster_value.shape().dims::<3>();

    let [
        state_node_batch,
        state_node_rank,
        state_node_value_dim,
        state_node_count,
    ] = state.node_rho().shape().dims::<4>();
    let [
        state_cluster_batch,
        state_cluster_rank,
        state_cluster_value_dim,
        state_cluster_count,
    ] = state.cluster_rho().shape().dims::<4>();
    let [
        state_global_batch,
        _global_count,
        state_global_rank,
        state_global_value_dim,
    ] = state.global_rho().shape().dims::<4>();

    expect_eq(
        "node_query.batch",
        state_node_batch,
        node_batch,
        GraphExecutionError::BatchMismatch {
            field: "node_query.batch",
            expected: state_node_batch,
            actual: node_batch,
        },
    )?;
    expect_eq(
        "node_value.batch",
        state_node_batch,
        node_value_batch,
        GraphExecutionError::BatchMismatch {
            field: "node_value.batch",
            expected: state_node_batch,
            actual: node_value_batch,
        },
    )?;
    expect_eq(
        "cluster_query.batch",
        state_cluster_batch,
        cluster_batch,
        GraphExecutionError::BatchMismatch {
            field: "cluster_query.batch",
            expected: state_cluster_batch,
            actual: cluster_batch,
        },
    )?;
    expect_eq(
        "cluster_value.batch",
        state_cluster_batch,
        cluster_value_batch,
        GraphExecutionError::BatchMismatch {
            field: "cluster_value.batch",
            expected: state_cluster_batch,
            actual: cluster_value_batch,
        },
    )?;

    expect_eq(
        "node_query.count",
        layout.node_count,
        node_count,
        GraphExecutionError::CountMismatch {
            field: "node_query.count",
            expected: layout.node_count,
            actual: node_count,
        },
    )?;
    expect_eq(
        "node_value.count",
        layout.node_count,
        node_value_count,
        GraphExecutionError::CountMismatch {
            field: "node_value.count",
            expected: layout.node_count,
            actual: node_value_count,
        },
    )?;
    expect_eq(
        "cluster_query.count",
        layout.cluster_count,
        cluster_count,
        GraphExecutionError::CountMismatch {
            field: "cluster_query.count",
            expected: layout.cluster_count,
            actual: cluster_count,
        },
    )?;
    expect_eq(
        "cluster_value.count",
        layout.cluster_count,
        cluster_value_count,
        GraphExecutionError::CountMismatch {
            field: "cluster_value.count",
            expected: layout.cluster_count,
            actual: cluster_value_count,
        },
    )?;

    expect_eq(
        "node_query.rank",
        state_node_rank,
        node_rank,
        GraphExecutionError::RankMismatch {
            field: "node_query.rank",
            expected: state_node_rank,
            actual: node_rank,
        },
    )?;
    expect_eq(
        "cluster_query.rank",
        state_cluster_rank,
        cluster_rank,
        GraphExecutionError::RankMismatch {
            field: "cluster_query.rank",
            expected: state_cluster_rank,
            actual: cluster_rank,
        },
    )?;
    expect_eq(
        "cluster_rho.rank",
        state_node_rank,
        state_cluster_rank,
        GraphExecutionError::RankMismatch {
            field: "cluster_rho.rank",
            expected: state_node_rank,
            actual: state_cluster_rank,
        },
    )?;
    expect_eq(
        "global_rho.rank",
        state_node_rank,
        state_global_rank,
        GraphExecutionError::RankMismatch {
            field: "global_rho.rank",
            expected: state_node_rank,
            actual: state_global_rank,
        },
    )?;

    expect_eq(
        "node_value.value_dim",
        state_node_value_dim,
        node_value_dim,
        GraphExecutionError::ValueDimMismatch {
            field: "node_value.value_dim",
            expected: state_node_value_dim,
            actual: node_value_dim,
        },
    )?;
    expect_eq(
        "cluster_value.value_dim",
        state_cluster_value_dim,
        cluster_value_dim,
        GraphExecutionError::ValueDimMismatch {
            field: "cluster_value.value_dim",
            expected: state_cluster_value_dim,
            actual: cluster_value_dim,
        },
    )?;
    expect_eq(
        "cluster_rho.value_dim",
        state_node_value_dim,
        state_cluster_value_dim,
        GraphExecutionError::ValueDimMismatch {
            field: "cluster_rho.value_dim",
            expected: state_node_value_dim,
            actual: state_cluster_value_dim,
        },
    )?;
    expect_eq(
        "global_rho.value_dim",
        state_node_value_dim,
        state_global_value_dim,
        GraphExecutionError::ValueDimMismatch {
            field: "global_rho.value_dim",
            expected: state_node_value_dim,
            actual: state_global_value_dim,
        },
    )?;
    expect_eq(
        "node_rho.count",
        layout.node_count,
        state_node_count,
        GraphExecutionError::CountMismatch {
            field: "node_rho.count",
            expected: layout.node_count,
            actual: state_node_count,
        },
    )?;
    expect_eq(
        "cluster_rho.count",
        layout.cluster_count,
        state_cluster_count,
        GraphExecutionError::CountMismatch {
            field: "cluster_rho.count",
            expected: layout.cluster_count,
            actual: state_cluster_count,
        },
    )?;
    expect_eq(
        "global_rho.batch",
        state_node_batch,
        state_global_batch,
        GraphExecutionError::BatchMismatch {
            field: "global_rho.batch",
            expected: state_node_batch,
            actual: state_global_batch,
        },
    )?;

    Ok(())
}

fn expect_eq(
    _field: &'static str,
    expected: usize,
    actual: usize,
    err: GraphExecutionError,
) -> Result<(), GraphExecutionError> {
    if expected != actual {
        return Err(err);
    }
    Ok(())
}

fn sparse_read<B: Backend>(
    query: &Tensor<B, 3>,
    memory: &Tensor<B, 4>,
    adjacency: &GraphCsrAdjacency,
    source_count: usize,
    target_count: usize,
) -> Tensor<B, 3> {
    let [batch, _, _rank] = query.shape().dims::<3>();
    let value_dim = memory.shape().dims::<4>()[3];
    debug_assert_eq!(adjacency.source_count(), source_count);
    debug_assert_eq!(adjacency.target_count(), target_count);

    if source_count == 0 {
        return Tensor::<B, 3>::zeros([batch, 0, value_dim], &query.device());
    }

    let mut outputs = Vec::with_capacity(source_count);
    for source in 0..source_count {
        let q_s = query.clone().slice_dim(1, source..source + 1);
        let mut context = Tensor::<B, 3>::zeros([batch, 1, value_dim], &query.device());
        if let Some(targets) = adjacency.neighbors(source) {
            for &target in targets {
                let source_state = memory.clone().slice_dim(1, target..target + 1);
                let msg = target_major_identity_read(q_s.clone(), source_state);
                context = context + msg;
            }
        }
        outputs.push(context);
    }

    if outputs.is_empty() {
        Tensor::<B, 3>::zeros([batch, 0, value_dim], &query.device())
    } else {
        Tensor::cat(outputs, 1).reshape([batch, source_count, value_dim])
    }
}

fn direct_outer<B: Backend>(query: &Tensor<B, 3>, value: &Tensor<B, 3>) -> Tensor<B, 4> {
    target_major_outer_product(query.clone(), value.clone())
}

fn sparse_write_aggregate<B: Backend>(
    query: &Tensor<B, 3>,
    value: &Tensor<B, 3>,
    adjacency: &GraphCsrAdjacency,
    target_count: usize,
) -> Tensor<B, 4> {
    let [batch, source_count, rank] = query.shape().dims::<3>();
    let value_dim = value.shape().dims::<3>()[2];
    let incoming = adjacency.transpose();
    let device = query.device();

    if target_count == 0 {
        return Tensor::<B, 4>::zeros([batch, 0, rank, value_dim], &device);
    }

    let mut outputs = Vec::with_capacity(target_count);
    for target in 0..target_count {
        let mut update = Tensor::<B, 4>::zeros([batch, 1, rank, value_dim], &device);
        if let Some(sources) = incoming.neighbors(target) {
            for &source in sources {
                if source >= source_count {
                    continue;
                }
                let outer = query.clone().slice_dim(1, source..source + 1);
                let outer = target_major_outer_product(
                    outer,
                    value.clone().slice_dim(1, source..source + 1),
                );
                update = update + outer;
            }
        }
        outputs.push(update);
    }

    Tensor::cat(outputs, 1).reshape([batch, target_count, rank, value_dim])
}

fn zeros_target_major<B: Backend>(
    target_count: usize,
    rank: usize,
    value_dim: usize,
    device: &B::Device,
    batch: usize,
) -> Tensor<B, 4> {
    Tensor::<B, 4>::zeros([batch, target_count, rank, value_dim], device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphCsrAdjacency, GraphTopologyRouting};
    use burn::tensor::TensorData;
    use burn::tensor::backend::Backend as BackendTrait;
    use burn_ndarray::NdArray;

    type Backend = NdArray<f32>;

    fn device() -> <Backend as BackendTrait>::Device {
        <Backend as BackendTrait>::Device::default()
    }

    fn approx_eq(data: TensorData, expected: &[f32]) {
        let actual = data.to_vec::<f32>().expect("tensor data should be f32");
        assert_eq!(actual.len(), expected.len());
        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            let diff = (lhs - rhs).abs();
            assert!(diff <= 1e-5, "expected {rhs}, got {lhs}, diff={diff}");
        }
    }

    #[test]
    fn graph_reference_step_routes_sparse_reads_and_writes() {
        let device = device();
        let state = GraphTopologyState::from_parts(
            Tensor::<Backend, 3>::zeros([1, 3, 2], &device),
            Tensor::<Backend, 3>::zeros([1, 2, 2], &device),
            Tensor::<Backend, 4>::from_data(
                TensorData::new(vec![10.0, 20.0, 30.0, 1.0, 2.0, 3.0], [1, 1, 2, 3]),
                &device,
            ),
            Tensor::<Backend, 4>::from_data(
                TensorData::new(vec![100.0, 200.0, 10.0, 20.0], [1, 1, 2, 2]),
                &device,
            ),
            Tensor::<Backend, 4>::from_data(
                TensorData::new(vec![1000.0, 100.0], [1, 1, 1, 2]),
                &device,
            ),
            0,
            0,
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
        let inputs = GraphStepInputs {
            node_query: Tensor::<Backend, 3>::ones([1, 3, 1], &device),
            node_value: Tensor::<Backend, 3>::from_data(
                TensorData::new(vec![1.0, 11.0, 2.0, 12.0, 3.0, 13.0], [1, 3, 2]),
                &device,
            ),
            cluster_query: Tensor::<Backend, 3>::ones([1, 2, 1], &device),
            cluster_value: Tensor::<Backend, 3>::from_data(
                TensorData::new(vec![4.0, 14.0, 5.0, 15.0], [1, 2, 2]),
                &device,
            ),
        };

        let output = graph_reference_step(
            state,
            &routing,
            inputs,
            StructuredStepMode::Predict,
            GraphRhoStepConfig { predict_decay: 0.5 },
        )
        .expect("graph step should succeed");

        approx_eq(
            output.readouts.node_from_node.into_data(),
            &[20.0, 2.0, 40.0, 4.0, 30.0, 3.0],
        );
        approx_eq(
            output
                .readouts
                .node_from_cluster
                .expect("cluster readout")
                .into_data(),
            &[100.0, 10.0, 100.0, 10.0, 200.0, 20.0],
        );
        approx_eq(
            output
                .readouts
                .node_from_global
                .expect("global readout")
                .into_data(),
            &[1000.0, 100.0, 1000.0, 100.0, 1000.0, 100.0],
        );
        approx_eq(
            output
                .readouts
                .cluster_from_global
                .expect("cluster/global readout")
                .into_data(),
            &[1000.0, 100.0, 1000.0, 100.0],
        );

        approx_eq(
            output.state.node_rho().into_data(),
            &[6.0, 12.0, 18.0, 11.5, 13.0, 14.5],
        );
        approx_eq(
            output.state.cluster_rho().into_data(),
            &[57.0, 108.0, 42.0, 38.0],
        );
        approx_eq(output.state.global_rho().into_data(), &[515.0, 115.0]);
        assert_eq!(output.state.temporal_position(), 1);
        assert_eq!(output.state.prediction_age(), 1);
    }

    #[test]
    fn graph_reference_step_mode_controls_time_counters() {
        let device = device();
        let base_state = GraphTopologyState::from_parts(
            Tensor::<Backend, 3>::zeros([1, 1, 1], &device),
            Tensor::<Backend, 3>::zeros([1, 1, 1], &device),
            Tensor::<Backend, 4>::zeros([1, 1, 1, 1], &device),
            Tensor::<Backend, 4>::zeros([1, 1, 1, 1], &device),
            Tensor::<Backend, 4>::zeros([1, 1, 1, 1], &device),
            5,
            2,
        );
        let routing = GraphTopologyRouting::new(
            GraphCsrAdjacency::try_from_edges(1, 1, &[(0, 0)]).expect("valid self edge"),
        )
        .expect("valid routing")
        .with_cluster_assignments(1, &[0])
        .expect("valid cluster assignment")
        .with_node_global_assignments(1, &[0])
        .expect("valid node/global assignment")
        .with_cluster_global_assignments(1, &[0])
        .expect("valid cluster/global assignment");
        let inputs = GraphStepInputs {
            node_query: Tensor::<Backend, 3>::ones([1, 1, 1], &device),
            node_value: Tensor::<Backend, 3>::ones([1, 1, 1], &device),
            cluster_query: Tensor::<Backend, 3>::ones([1, 1, 1], &device),
            cluster_value: Tensor::<Backend, 3>::ones([1, 1, 1], &device),
        };

        let refined = graph_reference_step(
            base_state.clone(),
            &routing,
            inputs.clone(),
            StructuredStepMode::Refine,
            GraphRhoStepConfig::default(),
        )
        .expect("refine step should succeed");
        assert_eq!(refined.state.temporal_position(), 5);
        assert_eq!(refined.state.prediction_age(), 2);

        let predicted = graph_reference_step(
            base_state.clone(),
            &routing,
            inputs.clone(),
            StructuredStepMode::Predict,
            GraphRhoStepConfig::default(),
        )
        .expect("predict step should succeed");
        assert_eq!(predicted.state.temporal_position(), 6);
        assert_eq!(predicted.state.prediction_age(), 3);

        let observed = graph_reference_step(
            base_state,
            &routing,
            inputs,
            StructuredStepMode::Observe,
            GraphRhoStepConfig::default(),
        )
        .expect("observe step should succeed");
        assert_eq!(observed.state.temporal_position(), 5);
        assert_eq!(observed.state.prediction_age(), 0);
    }

    #[test]
    fn graph_reference_step_rejects_rank_mismatch() {
        let device = device();
        let state = GraphTopologyState::from_parts(
            Tensor::<Backend, 3>::zeros([1, 1, 1], &device),
            Tensor::<Backend, 3>::zeros([1, 1, 1], &device),
            Tensor::<Backend, 4>::zeros([1, 1, 1, 1], &device),
            Tensor::<Backend, 4>::zeros([1, 1, 1, 1], &device),
            Tensor::<Backend, 4>::zeros([1, 1, 1, 1], &device),
            0,
            0,
        );
        let routing = GraphTopologyRouting::new(
            GraphCsrAdjacency::try_from_edges(1, 1, &[(0, 0)]).expect("valid self edge"),
        )
        .expect("valid routing");
        let inputs = GraphStepInputs {
            node_query: Tensor::<Backend, 3>::ones([1, 1, 2], &device),
            node_value: Tensor::<Backend, 3>::ones([1, 1, 1], &device),
            cluster_query: Tensor::<Backend, 3>::zeros([1, 1, 1], &device),
            cluster_value: Tensor::<Backend, 3>::zeros([1, 1, 1], &device),
        };

        let err = match graph_reference_step(
            state,
            &routing,
            inputs,
            StructuredStepMode::Predict,
            GraphRhoStepConfig::default(),
        ) {
            Ok(_) => panic!("rank mismatch should fail validation"),
            Err(err) => err,
        };

        assert_eq!(
            err,
            GraphExecutionError::RankMismatch {
                field: "node_query.rank",
                expected: 1,
                actual: 2,
            }
        );
    }
}
