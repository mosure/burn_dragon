use burn::module::Initializer;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation;
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
