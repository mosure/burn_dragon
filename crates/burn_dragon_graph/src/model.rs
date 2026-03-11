mod compiled;
mod support;
#[cfg(test)]
mod tests;

use burn::module::{Module, Param};
use burn::nn::{LayerNorm, LayerNormConfig, Linear};
use burn::prelude::*;
use burn::tensor::activation;
use burn_dragon_core::{StructuredStepMode, near_critical_embedding_initializer};
use burn_dragon_wgpu::fused_sparse_graph_rho_attention_wgpu;

use crate::{
    CompiledGraphRouting, GraphExecutionError, GraphRhoStepConfig, GraphStepInputs,
    GraphStepOutput, GraphTopologyRouting, GraphTopologyState, graph_reference_step,
};
use compiled::{GraphCompiledRecurrentOutput, GraphCompiledState, GraphCompiledStepOutput};
use support::{
    GraphObservationMerger, linear, pool_dense_from_incoming, pool_dense_to_targets,
    validate_observation_layout, validate_state_dims,
};

/// Graph adapter configuration over the shared Dragon `rho` contract.
///
/// Paper mapping:
/// - `embed_dim` is the graph dense-space dimension
/// - `rank` is the graph neuron-space width used to read/write `rho`
/// - `value_dim` is the recurrent readout / write-value width emitted by `rho`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GraphDragonConfig {
    pub embed_dim: usize,
    pub rank: usize,
    pub value_dim: usize,
    pub predict_decay: f32,
    pub mode_embeddings: bool,
}

impl Default for GraphDragonConfig {
    fn default() -> Self {
        Self {
            embed_dim: 256,
            rank: 32,
            value_dim: 256,
            predict_decay: 0.95,
            mode_embeddings: true,
        }
    }
}

impl GraphDragonConfig {
    pub fn dense_space_dim(&self) -> usize {
        self.embed_dim.max(1)
    }

    pub fn neuron_space_dim(&self) -> usize {
        self.rank.max(1)
    }

    pub fn recurrent_value_dim(&self) -> usize {
        self.value_dim.max(1)
    }
}

/// Learned graph recurrent adapter built on top of sparse banked `rho`.
///
/// The persistent state contract is:
/// - dense-space node / cluster activations
/// - node / cluster / global `rho` banks
/// - temporal counters that define the recurrent axis for `observe` / `refine` / `predict`
///
/// Unlike the core BDH path, this adapter does not expose a paper-identical `y_gate` / `y_neuron`
/// tensor. It projects dense state into neuron-space write activations, reads dense-space messages
/// from `rho`, and merges those readouts back into the dense activation stream.
#[derive(Module, Debug)]
pub struct GraphDragon<B: Backend> {
    embed_dim: usize,
    rank: usize,
    value_dim: usize,
    predict_decay: f32,
    step_mode_embeddings: Option<Param<Tensor<B, 2>>>,
    node_query: Linear<B>,
    node_value: Linear<B>,
    cluster_query: Linear<B>,
    cluster_value: Linear<B>,
    node_local_out: Linear<B>,
    node_cluster_out: Linear<B>,
    node_global_out: Linear<B>,
    cluster_pool_out: Linear<B>,
    cluster_global_out: Linear<B>,
    node_norm: LayerNorm<B>,
    cluster_norm: LayerNorm<B>,
    node_observation: GraphObservationMerger<B>,
    cluster_observation: GraphObservationMerger<B>,
}

impl<B: Backend> GraphDragon<B> {
    pub fn new(config: GraphDragonConfig, device: &B::Device) -> Self {
        let mode_embeddings = if config.mode_embeddings {
            Some(
                near_critical_embedding_initializer(config.embed_dim.max(1))
                    .init([StructuredStepMode::COUNT, config.embed_dim.max(1)], device),
            )
        } else {
            None
        };

        Self {
            embed_dim: config.embed_dim.max(1),
            rank: config.rank.max(1),
            value_dim: config.value_dim.max(1),
            predict_decay: config.predict_decay,
            step_mode_embeddings: mode_embeddings,
            node_query: linear(config.embed_dim.max(1), config.rank.max(1), device),
            node_value: linear(config.embed_dim.max(1), config.value_dim.max(1), device),
            cluster_query: linear(config.embed_dim.max(1), config.rank.max(1), device),
            cluster_value: linear(config.embed_dim.max(1), config.value_dim.max(1), device),
            node_local_out: linear(config.value_dim.max(1), config.embed_dim.max(1), device),
            node_cluster_out: linear(config.value_dim.max(1), config.embed_dim.max(1), device),
            node_global_out: linear(config.value_dim.max(1), config.embed_dim.max(1), device),
            cluster_pool_out: linear(config.embed_dim.max(1), config.embed_dim.max(1), device),
            cluster_global_out: linear(config.value_dim.max(1), config.embed_dim.max(1), device),
            node_norm: LayerNormConfig::new(config.embed_dim.max(1)).init(device),
            cluster_norm: LayerNormConfig::new(config.embed_dim.max(1)).init(device),
            node_observation: GraphObservationMerger::new(config.embed_dim.max(1), device),
            cluster_observation: GraphObservationMerger::new(config.embed_dim.max(1), device),
        }
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn value_dim(&self) -> usize {
        self.value_dim
    }

    pub fn state_from_observations(
        &self,
        routing: &GraphTopologyRouting,
        node_observation: Tensor<B, 3>,
        cluster_observation: Tensor<B, 3>,
    ) -> Result<GraphTopologyState<B>, GraphExecutionError> {
        let layout = routing.layout()?;
        validate_observation_layout(
            self.embed_dim,
            layout.node_count,
            layout.cluster_count,
            &node_observation,
            &cluster_observation,
        )?;
        let [batch, node_count, _] = node_observation.shape().dims::<3>();
        let cluster_count = cluster_observation.shape().dims::<3>()[1];
        let device = node_observation.device();

        Ok(GraphTopologyState::from_parts(
            node_observation,
            cluster_observation,
            Tensor::<B, 4>::zeros([batch, self.rank, self.value_dim, node_count], &device),
            Tensor::<B, 4>::zeros([batch, self.rank, self.value_dim, cluster_count], &device),
            Tensor::<B, 4>::zeros(
                [batch, layout.global_count, self.rank, self.value_dim],
                &device,
            ),
            0,
            0,
        ))
    }

    pub fn state_with_observations(
        &self,
        state: GraphTopologyState<B>,
        node_observation: Tensor<B, 3>,
        cluster_observation: Tensor<B, 3>,
    ) -> Result<GraphTopologyState<B>, GraphExecutionError> {
        let layout = state.layout();
        validate_observation_layout(
            self.embed_dim,
            layout.node_count,
            layout.cluster_count,
            &node_observation,
            &cluster_observation,
        )?;
        Ok(GraphTopologyState::from_parts(
            node_observation,
            cluster_observation,
            state.node_rho(),
            state.cluster_rho(),
            state.global_rho(),
            state.temporal_position(),
            state.prediction_age(),
        ))
    }

    pub fn observe(
        &self,
        state: GraphTopologyState<B>,
        routing: &GraphTopologyRouting,
        node_observation: Tensor<B, 3>,
        cluster_observation: Tensor<B, 3>,
    ) -> Result<GraphStepOutput<B>, GraphExecutionError> {
        validate_state_dims(self.embed_dim, self.rank, self.value_dim, &state)?;
        let merged_node = self
            .node_observation
            .forward(state.node_state(), node_observation);
        let merged_cluster = self
            .cluster_observation
            .forward(state.cluster_state(), cluster_observation);
        let observed_state = self.state_with_observations(state, merged_node, merged_cluster)?;
        self.step(observed_state, routing, StructuredStepMode::Observe)
    }

    pub fn refine(
        &self,
        state: GraphTopologyState<B>,
        routing: &GraphTopologyRouting,
    ) -> Result<GraphStepOutput<B>, GraphExecutionError> {
        self.step(state, routing, StructuredStepMode::Refine)
    }

    pub fn predict(
        &self,
        state: GraphTopologyState<B>,
        routing: &GraphTopologyRouting,
    ) -> Result<GraphStepOutput<B>, GraphExecutionError> {
        self.step(state, routing, StructuredStepMode::Predict)
    }

    pub fn step(
        &self,
        state: GraphTopologyState<B>,
        routing: &GraphTopologyRouting,
        mode: StructuredStepMode,
    ) -> Result<GraphStepOutput<B>, GraphExecutionError> {
        validate_state_dims(self.embed_dim, self.rank, self.value_dim, &state)?;
        let inputs = self.project_step_inputs(&state, mode);
        let recurrent = graph_reference_step(
            state,
            routing,
            inputs,
            mode,
            GraphRhoStepConfig {
                predict_decay: self.predict_decay,
            },
        )?;
        Ok(self.finish_step_output(recurrent, routing, mode))
    }

    pub fn step_compiled(
        &self,
        state: GraphTopologyState<B>,
        compiled: &CompiledGraphRouting<B>,
        mode: StructuredStepMode,
    ) -> Result<GraphStepOutput<B>, GraphExecutionError> {
        validate_state_dims(self.embed_dim, self.rank, self.value_dim, &state)?;
        let output = self.step_compiled_state_unchecked(
            GraphCompiledState::from_topology_state(state),
            compiled,
            mode,
        )?;
        Ok(GraphStepOutput {
            state: output.state.into_topology_state(),
            readouts: output.readouts,
        })
    }

    pub fn rollout(
        &self,
        mut state: GraphTopologyState<B>,
        routing: &GraphTopologyRouting,
        steps: usize,
        mode: StructuredStepMode,
    ) -> Result<GraphTopologyState<B>, GraphExecutionError> {
        let steps = steps.max(1);
        for _ in 0..steps {
            state = self.step(state, routing, mode)?.state;
        }
        Ok(state)
    }

    pub fn rollout_compiled(
        &self,
        state: GraphTopologyState<B>,
        compiled: &CompiledGraphRouting<B>,
        steps: usize,
        mode: StructuredStepMode,
    ) -> Result<GraphTopologyState<B>, GraphExecutionError> {
        validate_state_dims(self.embed_dim, self.rank, self.value_dim, &state)?;
        let mut state = GraphCompiledState::from_topology_state(state);
        let steps = steps.max(1);
        for _ in 0..steps {
            state = self
                .step_compiled_state_unchecked(state, compiled, mode)?
                .state;
        }
        Ok(state.into_topology_state())
    }

    fn project_step_inputs(
        &self,
        state: &GraphTopologyState<B>,
        mode: StructuredStepMode,
    ) -> GraphStepInputs<B> {
        self.project_step_inputs_from_dense(state.node_state(), state.cluster_state(), mode)
    }

    fn project_step_inputs_from_dense(
        &self,
        node_dense_state: Tensor<B, 3>,
        cluster_dense_state: Tensor<B, 3>,
        mode: StructuredStepMode,
    ) -> GraphStepInputs<B> {
        let node_dense_state = self.apply_step_mode(node_dense_state, mode);
        let cluster_dense_state = self.apply_step_mode(cluster_dense_state, mode);
        let node_x_neuron = activation::relu(self.node_query.forward(node_dense_state.clone()));
        let cluster_x_neuron =
            activation::relu(self.cluster_query.forward(cluster_dense_state.clone()));
        GraphStepInputs {
            node_query: node_x_neuron,
            node_value: self.node_value.forward(node_dense_state),
            cluster_query: cluster_x_neuron,
            cluster_value: self.cluster_value.forward(cluster_dense_state),
        }
    }

    fn step_compiled_state_unchecked(
        &self,
        state: GraphCompiledState<B>,
        compiled: &CompiledGraphRouting<B>,
        mode: StructuredStepMode,
    ) -> Result<GraphCompiledStepOutput<B>, GraphExecutionError> {
        let inputs = self.project_step_inputs_from_dense(
            state.node_state.clone(),
            state.cluster_state.clone(),
            mode,
        );
        let recurrent = if let Some(recurrent) =
            self.try_fused_recurrent_step_compiled(&state, compiled, &inputs, mode)
        {
            recurrent
        } else {
            let reference = graph_reference_step(
                state.into_topology_state(),
                compiled.host(),
                inputs,
                mode,
                GraphRhoStepConfig {
                    predict_decay: self.predict_decay,
                },
            )?;
            GraphCompiledRecurrentOutput {
                state: GraphCompiledState::from_topology_state(reference.state),
                readouts: reference.readouts,
            }
        };
        Ok(self.finish_compiled_step_output(recurrent, compiled, mode))
    }

    fn finish_step_output(
        &self,
        recurrent: GraphStepOutput<B>,
        routing: &GraphTopologyRouting,
        mode: StructuredStepMode,
    ) -> GraphStepOutput<B> {
        let next_node_state = self.merge_node_state(&recurrent, mode);
        let next_cluster_state = self.merge_cluster_state(&recurrent, routing, mode);
        let recurrent_state = recurrent.state;
        let next_state = GraphTopologyState::from_parts(
            next_node_state,
            next_cluster_state,
            recurrent_state.node_rho(),
            recurrent_state.cluster_rho(),
            recurrent_state.global_rho(),
            recurrent_state.temporal_position(),
            recurrent_state.prediction_age(),
        );

        GraphStepOutput {
            state: next_state,
            readouts: recurrent.readouts,
        }
    }

    fn apply_step_mode(&self, state: Tensor<B, 3>, mode: StructuredStepMode) -> Tensor<B, 3> {
        let Some(step_mode_embeddings) = &self.step_mode_embeddings else {
            return state;
        };
        let [batch, time, dim] = state.shape().dims::<3>();
        let bias = step_mode_embeddings
            .val()
            .slice_dim(0, mode.index()..mode.index() + 1)
            .reshape([1, 1, dim])
            .repeat_dim(0, batch)
            .repeat_dim(1, time);
        state + bias
    }

    fn try_fused_recurrent_step_compiled(
        &self,
        state: &GraphCompiledState<B>,
        compiled: &CompiledGraphRouting<B>,
        inputs: &GraphStepInputs<B>,
        mode: StructuredStepMode,
    ) -> Option<GraphCompiledRecurrentOutput<B>> {
        let decay = self.predict_decay_for_mode(mode);

        let node_from_node = fused_sparse_graph_rho_attention_wgpu(
            &inputs.node_query,
            &inputs.node_value,
            Some(&state.node_rho),
            compiled.node_neighbors().csr(),
            decay,
        )
        .ok()?;
        let node_self = fused_sparse_graph_rho_attention_wgpu(
            &inputs.node_query,
            &inputs.node_value,
            Some(&state.node_rho),
            compiled.node_identity().csr(),
            decay,
        )
        .ok()?;

        let cluster_self = compiled.cluster_identity().and_then(|route| {
            fused_sparse_graph_rho_attention_wgpu(
                &inputs.cluster_query,
                &inputs.cluster_value,
                Some(&state.cluster_rho),
                route.csr(),
                decay,
            )
            .ok()
        });
        let node_to_cluster = compiled.node_to_cluster().and_then(|route| {
            fused_sparse_graph_rho_attention_wgpu(
                &inputs.node_query,
                &inputs.node_value,
                Some(&state.cluster_rho),
                route.csr(),
                decay,
            )
            .ok()
        });
        let node_to_global = compiled.node_to_global().and_then(|route| {
            fused_sparse_graph_rho_attention_wgpu(
                &inputs.node_query,
                &inputs.node_value,
                Some(&state.global_rho),
                route.csr(),
                decay,
            )
            .ok()
        });
        let cluster_to_global = compiled.cluster_to_global().and_then(|route| {
            fused_sparse_graph_rho_attention_wgpu(
                &inputs.cluster_query,
                &inputs.cluster_value,
                Some(&state.global_rho),
                route.csr(),
                decay,
            )
            .ok()
        });

        let decayed_cluster_rho = state.cluster_rho.clone().mul_scalar(decay);
        let cluster_rho = match (cluster_self.as_ref(), node_to_cluster.as_ref()) {
            (Some(cluster_self), Some(node_to_cluster)) => {
                cluster_self.rho.clone() + node_to_cluster.rho.clone() - decayed_cluster_rho
            }
            (Some(cluster_self), None) => cluster_self.rho.clone(),
            (None, Some(node_to_cluster)) => node_to_cluster.rho.clone(),
            (None, None) => decayed_cluster_rho,
        };

        let decayed_global_rho = state.global_rho.clone().mul_scalar(decay);
        let global_rho = match (node_to_global.as_ref(), cluster_to_global.as_ref()) {
            (Some(node_to_global), Some(cluster_to_global)) => {
                node_to_global.rho.clone() + cluster_to_global.rho.clone() - decayed_global_rho
            }
            (Some(node_to_global), None) => node_to_global.rho.clone(),
            (None, Some(cluster_to_global)) => cluster_to_global.rho.clone(),
            (None, None) => decayed_global_rho,
        };

        let (temporal_position, prediction_age) =
            self.advance_temporal_counters(state.temporal_position, state.prediction_age, mode);

        Some(GraphCompiledRecurrentOutput {
            state: GraphCompiledState {
                layout: state.layout,
                node_state: state.node_state.clone(),
                cluster_state: state.cluster_state.clone(),
                node_rho: node_self.rho,
                cluster_rho,
                global_rho,
                temporal_position,
                prediction_age,
            },
            readouts: crate::GraphStepReadouts {
                node_from_node: node_from_node.context,
                node_from_cluster: node_to_cluster.map(|output| output.context),
                node_from_global: node_to_global.map(|output| output.context),
                cluster_from_global: cluster_to_global.map(|output| output.context),
            },
        })
    }

    fn finish_compiled_step_output(
        &self,
        recurrent: GraphCompiledRecurrentOutput<B>,
        compiled: &CompiledGraphRouting<B>,
        mode: StructuredStepMode,
    ) -> GraphCompiledStepOutput<B> {
        let next_node_state = self.merge_node_state_dense(
            recurrent.state.node_state.clone(),
            &recurrent.readouts,
            mode,
        );
        let next_cluster_state = self.merge_cluster_state_compiled(
            recurrent.state.cluster_state.clone(),
            recurrent.state.node_state.clone(),
            compiled
                .node_to_cluster()
                .map(|route| route.incoming_adjacency()),
            &recurrent.readouts,
            mode,
        );
        let recurrent_state = recurrent.state;
        let next_state = GraphCompiledState {
            layout: recurrent_state.layout,
            node_state: next_node_state,
            cluster_state: next_cluster_state,
            node_rho: recurrent_state.node_rho,
            cluster_rho: recurrent_state.cluster_rho,
            global_rho: recurrent_state.global_rho,
            temporal_position: recurrent_state.temporal_position,
            prediction_age: recurrent_state.prediction_age,
        };

        GraphCompiledStepOutput {
            state: next_state,
            readouts: recurrent.readouts,
        }
    }

    fn predict_decay_for_mode(&self, mode: StructuredStepMode) -> f32 {
        if mode.temporal_dt() == 0 {
            1.0
        } else {
            self.predict_decay.clamp(0.0, 1.0)
        }
    }

    fn advance_temporal_counters(
        &self,
        temporal_position: usize,
        prediction_age: usize,
        mode: StructuredStepMode,
    ) -> (usize, usize) {
        let temporal_dt = mode.temporal_dt();
        if temporal_dt > 0 {
            (
                temporal_position.saturating_add(temporal_dt),
                prediction_age.saturating_add(temporal_dt),
            )
        } else if mode.resets_prediction_age() {
            (temporal_position, 0)
        } else {
            (temporal_position, prediction_age)
        }
    }

    fn merge_node_state(
        &self,
        recurrent: &GraphStepOutput<B>,
        mode: StructuredStepMode,
    ) -> Tensor<B, 3> {
        self.merge_node_state_dense(recurrent.state.node_state(), &recurrent.readouts, mode)
    }

    fn merge_node_state_dense(
        &self,
        current_node_state: Tensor<B, 3>,
        readouts: &crate::GraphStepReadouts<B>,
        mode: StructuredStepMode,
    ) -> Tensor<B, 3> {
        let current = self.apply_step_mode(current_node_state, mode);
        let mut delta = self.node_local_out.forward(readouts.node_from_node.clone());
        if let Some(node_from_cluster) = &readouts.node_from_cluster {
            delta = delta + self.node_cluster_out.forward(node_from_cluster.clone());
        }
        if let Some(node_from_global) = &readouts.node_from_global {
            delta = delta + self.node_global_out.forward(node_from_global.clone());
        }
        self.node_norm.forward(current + delta)
    }

    fn merge_cluster_state(
        &self,
        recurrent: &GraphStepOutput<B>,
        routing: &GraphTopologyRouting,
        mode: StructuredStepMode,
    ) -> Tensor<B, 3> {
        let current = self.apply_step_mode(recurrent.state.cluster_state(), mode);
        let [batch, cluster_count, _] = current.shape().dims::<3>();
        let device = current.device();
        let mut delta = if let Some(route) = routing.node_to_cluster() {
            let pooled = pool_dense_to_targets(recurrent.state.node_state(), route, cluster_count);
            self.cluster_pool_out.forward(pooled)
        } else {
            Tensor::<B, 3>::zeros([batch, cluster_count, self.embed_dim], &device)
        };
        if let Some(cluster_from_global) = &recurrent.readouts.cluster_from_global {
            delta = delta + self.cluster_global_out.forward(cluster_from_global.clone());
        }
        self.cluster_norm.forward(current + delta)
    }

    fn merge_cluster_state_compiled(
        &self,
        current_cluster_state: Tensor<B, 3>,
        node_state_for_pool: Tensor<B, 3>,
        incoming_node_to_cluster: Option<&crate::GraphCsrAdjacency>,
        readouts: &crate::GraphStepReadouts<B>,
        mode: StructuredStepMode,
    ) -> Tensor<B, 3> {
        let current = self.apply_step_mode(current_cluster_state, mode);
        let [batch, cluster_count, _] = current.shape().dims::<3>();
        let device = current.device();
        let mut delta = if let Some(incoming) = incoming_node_to_cluster {
            let pooled = pool_dense_from_incoming(node_state_for_pool, incoming, cluster_count);
            self.cluster_pool_out.forward(pooled)
        } else {
            Tensor::<B, 3>::zeros([batch, cluster_count, self.embed_dim], &device)
        };
        if let Some(cluster_from_global) = &readouts.cluster_from_global {
            delta = delta + self.cluster_global_out.forward(cluster_from_global.clone());
        }
        self.cluster_norm.forward(current + delta)
    }
}
