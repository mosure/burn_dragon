use burn::prelude::*;
use burn::tensor::Int;
use burn::tensor::TensorData;
use burn_dragon_kernel::api::graph::SparseGraphCsr;

use crate::{GraphCsrAdjacency, GraphTopologyRouting};

#[derive(Clone)]
pub struct CompiledGraphRoute<B: Backend> {
    adjacency: GraphCsrAdjacency,
    incoming_adjacency: GraphCsrAdjacency,
    incoming_pool_weights: Tensor<B, 3>,
    assignment_targets: Option<Tensor<B, 1, Int>>,
    csr: SparseGraphCsr<B>,
}

impl<B: Backend> CompiledGraphRoute<B> {
    fn new(adjacency: GraphCsrAdjacency, device: &B::Device) -> Self {
        let incoming = adjacency.transpose();
        let assignment_targets = assignment_route_targets(&adjacency, device);
        Self {
            incoming_pool_weights: incoming_pool_weights(&incoming, device),
            assignment_targets,
            csr: SparseGraphCsr::from_usize_slices(
                adjacency.offsets(),
                adjacency.indices(),
                incoming.offsets(),
                incoming.indices(),
                device,
            ),
            incoming_adjacency: incoming,
            adjacency,
        }
    }

    pub fn adjacency(&self) -> &GraphCsrAdjacency {
        &self.adjacency
    }

    pub fn incoming_adjacency(&self) -> &GraphCsrAdjacency {
        &self.incoming_adjacency
    }

    pub fn incoming_pool_weights(&self) -> &Tensor<B, 3> {
        &self.incoming_pool_weights
    }

    pub fn assignment_targets(&self) -> Option<&Tensor<B, 1, Int>> {
        self.assignment_targets.as_ref()
    }

    pub fn csr(&self) -> &SparseGraphCsr<B> {
        &self.csr
    }
}

#[derive(Clone)]
pub struct CompiledGraphRouting<B: Backend> {
    host: GraphTopologyRouting,
    node_identity: CompiledGraphRoute<B>,
    node_neighbors: CompiledGraphRoute<B>,
    cluster_identity: Option<CompiledGraphRoute<B>>,
    node_to_cluster: Option<CompiledGraphRoute<B>>,
    node_to_global: Option<CompiledGraphRoute<B>>,
    cluster_to_global: Option<CompiledGraphRoute<B>>,
}

impl<B: Backend> CompiledGraphRouting<B> {
    pub fn new(host: GraphTopologyRouting, device: &B::Device) -> Self {
        let node_identity = CompiledGraphRoute::new(identity_adjacency(host.node_count()), device);
        let node_neighbors = CompiledGraphRoute::new(host.node_neighbors().clone(), device);
        let cluster_identity = host
            .layout()
            .ok()
            .and_then(|layout| (layout.cluster_count > 0).then_some(layout.cluster_count))
            .map(|cluster_count| {
                CompiledGraphRoute::new(identity_adjacency(cluster_count), device)
            });
        let node_to_cluster = host
            .node_to_cluster()
            .cloned()
            .map(|adj| CompiledGraphRoute::new(adj, device));
        let node_to_global = host
            .node_to_global()
            .cloned()
            .map(|adj| CompiledGraphRoute::new(adj, device));
        let cluster_to_global = host
            .cluster_to_global()
            .cloned()
            .map(|adj| CompiledGraphRoute::new(adj, device));

        Self {
            host,
            node_identity,
            node_neighbors,
            cluster_identity,
            node_to_cluster,
            node_to_global,
            cluster_to_global,
        }
    }

    pub fn host(&self) -> &GraphTopologyRouting {
        &self.host
    }

    pub fn node_identity(&self) -> &CompiledGraphRoute<B> {
        &self.node_identity
    }

    pub fn node_neighbors(&self) -> &CompiledGraphRoute<B> {
        &self.node_neighbors
    }

    pub fn cluster_identity(&self) -> Option<&CompiledGraphRoute<B>> {
        self.cluster_identity.as_ref()
    }

    pub fn node_to_cluster(&self) -> Option<&CompiledGraphRoute<B>> {
        self.node_to_cluster.as_ref()
    }

    pub fn node_to_global(&self) -> Option<&CompiledGraphRoute<B>> {
        self.node_to_global.as_ref()
    }

    pub fn cluster_to_global(&self) -> Option<&CompiledGraphRoute<B>> {
        self.cluster_to_global.as_ref()
    }
}

fn identity_adjacency(count: usize) -> GraphCsrAdjacency {
    let edges = (0..count).map(|index| (index, index)).collect::<Vec<_>>();
    GraphCsrAdjacency::try_from_edges(count, count, &edges).expect("identity graph adjacency")
}

fn incoming_pool_weights<B: Backend>(
    incoming: &GraphCsrAdjacency,
    device: &B::Device,
) -> Tensor<B, 3> {
    let source_count = incoming.target_count();
    let target_count = incoming.source_count();
    let mut weights = vec![0.0_f32; source_count * target_count];

    for target in 0..target_count {
        if let Some(sources) = incoming.neighbors(target) {
            if sources.is_empty() {
                continue;
            }
            let weight = 1.0_f32 / sources.len() as f32;
            for &source in sources {
                weights[source * target_count + target] = weight;
            }
        }
    }

    Tensor::<B, 3>::from_data(
        TensorData::new(weights, [1, source_count, target_count]),
        device,
    )
}

fn assignment_route_targets<B: Backend>(
    adjacency: &GraphCsrAdjacency,
    device: &B::Device,
) -> Option<Tensor<B, 1, Int>> {
    if !(0..adjacency.source_count()).all(|source| adjacency.degree(source) == Some(1)) {
        return None;
    }

    let source_count = adjacency.source_count();
    let mut targets = vec![0_i64; source_count];

    for (source, target_slot) in targets.iter_mut().enumerate().take(source_count) {
        let target = adjacency
            .neighbors(source)
            .and_then(|neighbors| neighbors.first().copied())
            .expect("assignment route requires one target per source");
        *target_slot = target as i64;
    }

    Some(Tensor::<B, 1, Int>::from_data(
        TensorData::new(targets, [source_count]),
        device,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend as BackendTrait;
    use burn_ndarray::NdArray;

    #[test]
    fn compiled_graph_routing_materializes_identity_and_sparse_buffers() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let host = GraphTopologyRouting::new(
            GraphCsrAdjacency::try_from_edges(3, 3, &[(0, 1), (1, 0), (2, 2)])
                .expect("valid node adjacency"),
        )
        .expect("valid routing")
        .with_cluster_assignments(2, &[0, 0, 1])
        .expect("valid cluster assignments")
        .with_node_global_assignments(1, &[0, 0, 0])
        .expect("valid node/global assignments")
        .with_cluster_global_assignments(1, &[0, 0])
        .expect("valid cluster/global assignments");

        let compiled = CompiledGraphRouting::<Backend>::new(host.clone(), &device);

        assert_eq!(compiled.host().layout(), host.layout());
        assert_eq!(
            compiled
                .node_identity()
                .csr()
                .source_offsets()
                .shape()
                .dims::<1>(),
            [host.node_count() + 1]
        );
        assert_eq!(
            compiled
                .node_neighbors()
                .csr()
                .source_indices()
                .shape()
                .dims::<1>()[0],
            host.node_neighbors().edge_count()
        );
        assert_eq!(
            compiled
                .cluster_identity()
                .expect("cluster identity")
                .csr()
                .source_indices()
                .shape()
                .dims::<1>()[0],
            2
        );
        assert_eq!(
            compiled
                .node_to_cluster()
                .expect("node/cluster route")
                .csr()
                .incoming_offsets()
                .shape()
                .dims::<1>(),
            [3]
        );
        assert_eq!(
            compiled
                .node_to_cluster()
                .expect("node/cluster route")
                .assignment_targets()
                .expect("assignment targets")
                .shape()
                .dims::<1>(),
            [3]
        );
    }
}
