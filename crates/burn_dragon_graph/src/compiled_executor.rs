use burn::prelude::*;

use crate::GraphTopologyRouting;
use crate::compiled_routing::CompiledGraphRouting;

/// Domain-level compiled execution wrapper for graph recurrent rollout.
///
/// This is the preferred library-facing handle for compiled graph execution. It gives callers a
/// smaller graph-native surface while still allowing expert access to the underlying routing plan.
#[derive(Clone)]
pub struct GraphCompiledExecutor<B: Backend> {
    routing: CompiledGraphRouting<B>,
}

impl<B: Backend> GraphCompiledExecutor<B> {
    pub fn new(routing: GraphTopologyRouting, device: &B::Device) -> Self {
        Self {
            routing: CompiledGraphRouting::new(routing, device),
        }
    }

    pub fn host_routing(&self) -> &GraphTopologyRouting {
        self.routing.host()
    }

    pub fn routing(&self) -> &CompiledGraphRouting<B> {
        &self.routing
    }

    pub fn into_routing(self) -> CompiledGraphRouting<B> {
        self.routing
    }
}
