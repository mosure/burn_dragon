use burn::module::Initializer;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{IndexingUpdateOp, activation};
use burn_dragon_core::near_critical_projection_std;

use crate::{GraphCsrAdjacency, GraphExecutionError, GraphTopologyState};

#[derive(Module, Debug)]
pub(super) struct GraphObservationMerger<B: Backend> {
    prior_norm: LayerNorm<B>,
    obs_norm: LayerNorm<B>,
    gate_prior: Linear<B>,
    gate_obs: Linear<B>,
}

impl<B: Backend> GraphObservationMerger<B> {
    pub(super) fn new(embed_dim: usize, device: &B::Device) -> Self {
        Self {
            prior_norm: LayerNormConfig::new(embed_dim.max(1)).init(device),
            obs_norm: LayerNormConfig::new(embed_dim.max(1)).init(device),
            gate_prior: linear(embed_dim.max(1), embed_dim.max(1), device),
            gate_obs: linear(embed_dim.max(1), embed_dim.max(1), device),
        }
    }

    pub(super) fn forward(&self, prior: Tensor<B, 3>, observation: Tensor<B, 3>) -> Tensor<B, 3> {
        let prior_norm = self.prior_norm.forward(prior.clone());
        let obs_norm = self.obs_norm.forward(observation.clone());
        let gate = activation::sigmoid(
            self.gate_prior.forward(prior_norm) + self.gate_obs.forward(obs_norm),
        );
        prior.clone() + gate * (observation - prior)
    }
}

pub(super) fn linear<B: Backend>(fan_in: usize, fan_out: usize, device: &B::Device) -> Linear<B> {
    let std = near_critical_projection_std(fan_in.max(1), fan_out.max(1));
    LinearConfig::new(fan_in.max(1), fan_out.max(1))
        .with_initializer(Initializer::Normal { mean: 0.0, std })
        .init(device)
}

pub(super) fn validate_observation_layout<B: Backend>(
    embed_dim: usize,
    node_count: usize,
    cluster_count: usize,
    node_observation: &Tensor<B, 3>,
    cluster_observation: &Tensor<B, 3>,
) -> Result<(), GraphExecutionError> {
    let [node_batch, observed_nodes, observed_dim] = node_observation.shape().dims::<3>();
    let [cluster_batch, observed_clusters, cluster_dim] = cluster_observation.shape().dims::<3>();
    if node_batch != cluster_batch {
        return Err(GraphExecutionError::BatchMismatch {
            field: "observations.batch",
            expected: node_batch,
            actual: cluster_batch,
        });
    }
    if observed_nodes != node_count {
        return Err(GraphExecutionError::CountMismatch {
            field: "node_observation.count",
            expected: node_count,
            actual: observed_nodes,
        });
    }
    if observed_clusters != cluster_count {
        return Err(GraphExecutionError::CountMismatch {
            field: "cluster_observation.count",
            expected: cluster_count,
            actual: observed_clusters,
        });
    }
    if observed_dim != embed_dim {
        return Err(GraphExecutionError::DenseDimMismatch {
            field: "node_observation.dim",
            expected: embed_dim,
            actual: observed_dim,
        });
    }
    if cluster_dim != embed_dim {
        return Err(GraphExecutionError::DenseDimMismatch {
            field: "cluster_observation.dim",
            expected: embed_dim,
            actual: cluster_dim,
        });
    }
    Ok(())
}

pub(super) fn validate_state_dims<B: Backend>(
    embed_dim: usize,
    rank: usize,
    value_dim: usize,
    state: &GraphTopologyState<B>,
) -> Result<(), GraphExecutionError> {
    let [_, _, node_dim] = state.node_state().shape().dims::<3>();
    let [_, _, cluster_dim] = state.cluster_state().shape().dims::<3>();
    let [_, node_rank, node_value_dim, _] = state.node_rho().shape().dims::<4>();
    let [_, cluster_rank, cluster_value_dim, _] = state.cluster_rho().shape().dims::<4>();
    let [_, _, global_rank, global_value_dim] = state.global_rho().shape().dims::<4>();

    if node_dim != embed_dim {
        return Err(GraphExecutionError::DenseDimMismatch {
            field: "state.node_state.dim",
            expected: embed_dim,
            actual: node_dim,
        });
    }
    if cluster_dim != embed_dim {
        return Err(GraphExecutionError::DenseDimMismatch {
            field: "state.cluster_state.dim",
            expected: embed_dim,
            actual: cluster_dim,
        });
    }
    if node_rank != rank {
        return Err(GraphExecutionError::RankMismatch {
            field: "state.node_rho.rank",
            expected: rank,
            actual: node_rank,
        });
    }
    if cluster_rank != rank {
        return Err(GraphExecutionError::RankMismatch {
            field: "state.cluster_rho.rank",
            expected: rank,
            actual: cluster_rank,
        });
    }
    if global_rank != rank {
        return Err(GraphExecutionError::RankMismatch {
            field: "state.global_rho.rank",
            expected: rank,
            actual: global_rank,
        });
    }
    if node_value_dim != value_dim {
        return Err(GraphExecutionError::ValueDimMismatch {
            field: "state.node_rho.value_dim",
            expected: value_dim,
            actual: node_value_dim,
        });
    }
    if cluster_value_dim != value_dim {
        return Err(GraphExecutionError::ValueDimMismatch {
            field: "state.cluster_rho.value_dim",
            expected: value_dim,
            actual: cluster_value_dim,
        });
    }
    if global_value_dim != value_dim {
        return Err(GraphExecutionError::ValueDimMismatch {
            field: "state.global_rho.value_dim",
            expected: value_dim,
            actual: global_value_dim,
        });
    }
    Ok(())
}

pub(super) fn pool_dense_to_targets<B: Backend>(
    source: Tensor<B, 3>,
    adjacency: &GraphCsrAdjacency,
    target_count: usize,
) -> Tensor<B, 3> {
    let incoming = adjacency.transpose();
    pool_dense_from_incoming(source, &incoming, target_count)
}

pub(super) fn pool_dense_from_incoming<B: Backend>(
    source: Tensor<B, 3>,
    incoming: &GraphCsrAdjacency,
    target_count: usize,
) -> Tensor<B, 3> {
    let [batch, _, dim] = source.shape().dims::<3>();
    if target_count == 0 {
        return Tensor::<B, 3>::zeros([batch, 0, dim], &source.device());
    }

    let mut outputs = Vec::with_capacity(target_count);
    for target in 0..target_count {
        let mut pooled = Tensor::<B, 3>::zeros([batch, 1, dim], &source.device());
        let mut degree = 0usize;
        if let Some(sources) = incoming.neighbors(target) {
            for &source_idx in sources {
                pooled = pooled + source.clone().slice_dim(1, source_idx..source_idx + 1);
                degree += 1;
            }
        }
        if degree > 0 {
            pooled = pooled.div_scalar(degree as f32);
        }
        outputs.push(pooled);
    }

    Tensor::cat(outputs, 1).reshape([batch, target_count, dim])
}

pub(super) fn pool_dense_with_weights<B: Backend>(
    source: Tensor<B, 3>,
    weights: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let [batch, source_count, dim] = source.shape().dims::<3>();
    let [weight_batch, weight_sources, target_count] = weights.shape().dims::<3>();
    assert_eq!(
        weight_batch, 1,
        "pool weights must be compiled in broadcast-ready batch-one layout"
    );
    assert_eq!(
        weight_sources, source_count,
        "pool weights source count must match dense source count"
    );
    if target_count == 0 {
        return Tensor::<B, 3>::zeros([batch, 0, dim], &source.device());
    }
    source
        .swap_dims(1, 2)
        .matmul(weights)
        .swap_dims(1, 2)
}

#[cfg(test)]
pub(super) fn gather_target_major_with_weights<B: Backend>(
    rho: Tensor<B, 4>,
    read_weights: Tensor<B, 3>,
) -> Tensor<B, 4> {
    let [batch, target_count, rank, value_dim] = rho.shape().dims::<4>();
    let [weight_batch, source_count, weight_targets] = read_weights.shape().dims::<3>();
    assert_eq!(
        weight_batch, 1,
        "assignment read weights must be compiled in broadcast-ready batch-one layout"
    );
    assert_eq!(
        weight_targets, target_count,
        "assignment read weights target count must match rho target count"
    );
    read_weights
        .matmul(rho.reshape([batch, target_count, rank * value_dim]))
        .reshape([batch, source_count, rank, value_dim])
}

pub(super) fn select_target_major_assignments<B: Backend>(
    rho: Tensor<B, 4>,
    assignment_targets: Tensor<B, 1, Int>,
) -> Tensor<B, 4> {
    rho.select(1, assignment_targets)
}

#[cfg(test)]
pub(super) fn aggregate_target_major_with_weights<B: Backend>(
    outer: Tensor<B, 4>,
    write_weights: Tensor<B, 3>,
) -> Tensor<B, 4> {
    let [batch, source_count, rank, value_dim] = outer.shape().dims::<4>();
    let [weight_batch, target_count, weight_sources] = write_weights.shape().dims::<3>();
    assert_eq!(
        weight_batch, 1,
        "assignment write weights must be compiled in broadcast-ready batch-one layout"
    );
    assert_eq!(
        weight_sources, source_count,
        "assignment write weights source count must match outer source count"
    );
    write_weights
        .matmul(outer.reshape([batch, source_count, rank * value_dim]))
        .reshape([batch, target_count, rank, value_dim])
}

pub(super) fn select_assign_target_major_assignments<B: Backend>(
    target_major: Tensor<B, 4>,
    outer: Tensor<B, 4>,
    assignment_targets: Tensor<B, 1, Int>,
) -> Tensor<B, 4> {
    target_major.select_assign(1, assignment_targets, outer, IndexingUpdateOp::Add)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn::tensor::backend::Backend as BackendTrait;
    use burn_ndarray::NdArray;

    type Backend = NdArray<f32>;

    #[test]
    fn broadcast_ready_weight_helpers_match_manual_batch_expansion() {
        let device = <Backend as BackendTrait>::Device::default();
        let source = Tensor::<Backend, 3>::from_data(
            TensorData::new(
                vec![
                    1.0, 2.0, 3.0, 4.0, //
                    5.0, 6.0, 7.0, 8.0, //
                    9.0, 10.0, 11.0, 12.0, //
                    13.0, 14.0, 15.0, 16.0,
                ],
                [2, 2, 4],
            ),
            &device,
        );
        let pool_weights = Tensor::<Backend, 3>::from_data(
            TensorData::new(
                vec![
                    1.0, 0.0, //
                    0.5, 0.5,
                ],
                [1, 2, 2],
            ),
            &device,
        );
        let pooled = pool_dense_with_weights(source.clone(), pool_weights.clone());
        let manual_pooled = source
            .clone()
            .swap_dims(1, 2)
            .matmul(pool_weights.clone().repeat_dim(0, 2))
            .swap_dims(1, 2);
        assert_eq!(
            pooled.into_data().to_vec::<f32>().expect("pooled"),
            manual_pooled
                .into_data()
                .to_vec::<f32>()
                .expect("manual pooled")
        );

        let rho = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                (0..2 * 2 * 2 * 3).map(|index| index as f32).collect::<Vec<_>>(),
                [2, 2, 2, 3],
            ),
            &device,
        );
        let read_weights = Tensor::<Backend, 3>::from_data(
            TensorData::new(
                vec![
                    1.0, 0.0, //
                    0.25, 0.75, //
                    0.0, 1.0,
                ],
                [1, 3, 2],
            ),
            &device,
        );
        let gathered = gather_target_major_with_weights(rho.clone(), read_weights.clone());
        let manual_gathered = read_weights
            .clone()
            .repeat_dim(0, 2)
            .matmul(rho.clone().reshape([2, 2, 6]))
            .reshape([2, 3, 2, 3]);
        assert_eq!(
            gathered
                .clone()
                .into_data()
                .to_vec::<f32>()
                .expect("gathered"),
            manual_gathered
                .into_data()
                .to_vec::<f32>()
                .expect("manual gathered")
        );

        let write_weights = Tensor::<Backend, 3>::from_data(
            TensorData::new(
                vec![
                    1.0, 0.0, 0.0, //
                    0.0, 0.5, 0.5,
                ],
                [1, 2, 3],
            ),
            &device,
        );
        let aggregated = aggregate_target_major_with_weights(gathered.clone(), write_weights.clone());
        let manual_aggregated = write_weights
            .clone()
            .repeat_dim(0, 2)
            .matmul(gathered.reshape([2, 3, 6]))
            .reshape([2, 2, 2, 3]);
        assert_eq!(
            aggregated
                .into_data()
                .to_vec::<f32>()
                .expect("aggregated"),
            manual_aggregated
                .into_data()
                .to_vec::<f32>()
                .expect("manual aggregated")
        );
    }

    #[test]
    fn select_target_major_assignments_matches_weight_gather_for_assignment_routes() {
        let device = <Backend as BackendTrait>::Device::default();
        let rho = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                (0..2 * 3 * 2 * 2).map(|index| index as f32).collect::<Vec<_>>(),
                [2, 3, 2, 2],
            ),
            &device,
        );
        let assignment_targets =
            Tensor::<Backend, 1, Int>::from_data(TensorData::new(vec![2_i64, 0, 1], [3]), &device);
        let read_weights = Tensor::<Backend, 3>::from_data(
            TensorData::new(
                vec![
                    0.0, 0.0, 1.0, //
                    1.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0,
                ],
                [1, 3, 3],
            ),
            &device,
        );

        let gathered = select_target_major_assignments(rho.clone(), assignment_targets);
        let weighted = gather_target_major_with_weights(rho, read_weights);

        assert_eq!(
            gathered.into_data().to_vec::<f32>().expect("gathered"),
            weighted.into_data().to_vec::<f32>().expect("weighted"),
        );
    }

    #[test]
    fn select_assign_target_major_assignments_matches_weight_aggregation_for_assignment_routes() {
        let device = <Backend as BackendTrait>::Device::default();
        let base = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                vec![
                    10.0, 11.0, 12.0, 13.0, //
                    20.0, 21.0, 22.0, 23.0, //
                    30.0, 31.0, 32.0, 33.0, //
                    40.0, 41.0, 42.0, 43.0,
                ],
                [1, 4, 2, 2],
            ),
            &device,
        );
        let outer = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                vec![
                    1.0, 2.0, 3.0, 4.0, //
                    5.0, 6.0, 7.0, 8.0, //
                    9.0, 10.0, 11.0, 12.0,
                ],
                [1, 3, 2, 2],
            ),
            &device,
        );
        let assignment_targets =
            Tensor::<Backend, 1, Int>::from_data(TensorData::new(vec![2_i64, 0, 2], [3]), &device);
        let write_weights = Tensor::<Backend, 3>::from_data(
            TensorData::new(
                vec![
                    0.0, 1.0, 0.0, //
                    0.0, 0.0, 0.0, //
                    1.0, 0.0, 1.0, //
                    0.0, 0.0, 0.0,
                ],
                [1, 4, 3],
            ),
            &device,
        );

        let updated = select_assign_target_major_assignments(
            base.clone(),
            outer.clone(),
            assignment_targets,
        );
        let weighted = base + aggregate_target_major_with_weights(outer, write_weights);

        assert_eq!(
            updated.into_data().to_vec::<f32>().expect("updated"),
            weighted.into_data().to_vec::<f32>().expect("weighted"),
        );
    }

}
