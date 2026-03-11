use burn::prelude::*;

use crate::{
    GraphStepReadouts, GraphTopologyLayout, GraphTopologyState,
    state::{rho_from_target_major, rho_to_target_major},
};

#[derive(Clone)]
pub(super) struct GraphCompiledState<B: Backend> {
    pub(super) layout: GraphTopologyLayout,
    pub(super) node_state: Tensor<B, 3>,
    pub(super) cluster_state: Tensor<B, 3>,
    pub(super) node_rho: Tensor<B, 4>,
    pub(super) cluster_rho: Tensor<B, 4>,
    pub(super) global_rho: Tensor<B, 4>,
    pub(super) temporal_position: usize,
    pub(super) prediction_age: usize,
}

#[derive(Clone)]
pub(super) struct GraphCompiledRecurrentOutput<B: Backend> {
    pub(super) state: GraphCompiledState<B>,
    pub(super) readouts: GraphStepReadouts<B>,
}

#[derive(Clone)]
pub(super) struct GraphCompiledStepOutput<B: Backend> {
    pub(super) state: GraphCompiledState<B>,
    pub(super) readouts: GraphStepReadouts<B>,
}

impl<B: Backend> GraphCompiledState<B> {
    pub(super) fn from_topology_state(state: GraphTopologyState<B>) -> Self {
        let layout = state.layout();
        Self {
            layout,
            node_state: state.node_state(),
            cluster_state: state.cluster_state(),
            node_rho: rho_to_target_major(state.node_rho()),
            cluster_rho: rho_to_target_major(state.cluster_rho()),
            global_rho: state.global_rho(),
            temporal_position: state.temporal_position(),
            prediction_age: state.prediction_age(),
        }
    }

    pub(super) fn into_topology_state(self) -> GraphTopologyState<B> {
        GraphTopologyState::from_parts(
            self.node_state,
            self.cluster_state,
            rho_from_target_major(self.node_rho),
            rho_from_target_major(self.cluster_rho),
            self.global_rho,
            self.temporal_position,
            self.prediction_age,
        )
    }
}
