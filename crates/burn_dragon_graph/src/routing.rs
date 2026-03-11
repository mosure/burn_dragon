use std::error::Error;
use std::fmt::{Display, Formatter};

use burn_dragon_core::{
    StructuredBankRole, StructuredRouteOperation, StructuredRoutePattern, StructuredRouteSpec,
    StructuredRoutingSpec,
};

use crate::state::GraphTopologyLayout;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphRoutingError {
    InvalidOffsetCount {
        expected: usize,
        actual: usize,
    },
    OffsetsMustStartAtZero {
        actual: usize,
    },
    OffsetsMustEndAtEdgeCount {
        expected: usize,
        actual: usize,
    },
    OffsetsNotMonotonic {
        index: usize,
        previous: usize,
        next: usize,
    },
    SourceOutOfBounds {
        source: usize,
        source_count: usize,
    },
    TargetOutOfBounds {
        target: usize,
        target_count: usize,
    },
    CountMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    MissingRoute {
        route: &'static str,
    },
}

impl Display for GraphRoutingError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOffsetCount { expected, actual } => write!(
                f,
                "invalid CSR offset count: expected {expected}, got {actual}"
            ),
            Self::OffsetsMustStartAtZero { actual } => {
                write!(f, "CSR offsets must start at zero, got {actual}")
            }
            Self::OffsetsMustEndAtEdgeCount { expected, actual } => write!(
                f,
                "CSR offsets must end at edge count: expected {expected}, got {actual}"
            ),
            Self::OffsetsNotMonotonic {
                index,
                previous,
                next,
            } => write!(
                f,
                "CSR offsets must be monotonic: offsets[{index}]={previous} > offsets[{}]={next}",
                index + 1
            ),
            Self::SourceOutOfBounds {
                source,
                source_count,
            } => write!(
                f,
                "CSR source index {source} is out of bounds for {source_count} sources"
            ),
            Self::TargetOutOfBounds {
                target,
                target_count,
            } => write!(
                f,
                "CSR target index {target} is out of bounds for {target_count} targets"
            ),
            Self::CountMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "count mismatch for {field}: expected {expected}, got {actual}"
            ),
            Self::MissingRoute { route } => write!(f, "missing required route: {route}"),
        }
    }
}

impl Error for GraphRoutingError {}

/// Sparse CSR adjacency owned by the graph adapter layer.
///
/// This stays out of `burn_dragon_core` on purpose: core only needs bank roles
/// and route metadata, while graph crates need concrete edge/index structure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphCsrAdjacency {
    source_count: usize,
    target_count: usize,
    offsets: Vec<usize>,
    indices: Vec<usize>,
}

impl GraphCsrAdjacency {
    pub fn try_new(
        source_count: usize,
        target_count: usize,
        offsets: Vec<usize>,
        indices: Vec<usize>,
    ) -> Result<Self, GraphRoutingError> {
        let expected_offsets = source_count + 1;
        if offsets.len() != expected_offsets {
            return Err(GraphRoutingError::InvalidOffsetCount {
                expected: expected_offsets,
                actual: offsets.len(),
            });
        }
        if offsets.first().copied().unwrap_or(usize::MAX) != 0 {
            return Err(GraphRoutingError::OffsetsMustStartAtZero {
                actual: offsets.first().copied().unwrap_or(usize::MAX),
            });
        }
        if offsets.last().copied().unwrap_or(usize::MAX) != indices.len() {
            return Err(GraphRoutingError::OffsetsMustEndAtEdgeCount {
                expected: indices.len(),
                actual: offsets.last().copied().unwrap_or(usize::MAX),
            });
        }
        for (index, window) in offsets.windows(2).enumerate() {
            if window[0] > window[1] {
                return Err(GraphRoutingError::OffsetsNotMonotonic {
                    index,
                    previous: window[0],
                    next: window[1],
                });
            }
        }
        for &target in &indices {
            if target >= target_count {
                return Err(GraphRoutingError::TargetOutOfBounds {
                    target,
                    target_count,
                });
            }
        }
        Ok(Self {
            source_count,
            target_count,
            offsets,
            indices,
        })
    }

    pub fn try_from_edges(
        source_count: usize,
        target_count: usize,
        edges: &[(usize, usize)],
    ) -> Result<Self, GraphRoutingError> {
        let mut buckets = vec![Vec::new(); source_count];
        for &(source, target) in edges {
            if source >= source_count {
                return Err(GraphRoutingError::SourceOutOfBounds {
                    source,
                    source_count,
                });
            }
            if target >= target_count {
                return Err(GraphRoutingError::TargetOutOfBounds {
                    target,
                    target_count,
                });
            }
            buckets[source].push(target);
        }

        let mut offsets = Vec::with_capacity(source_count + 1);
        let mut indices = Vec::new();
        offsets.push(0);
        for bucket in &mut buckets {
            bucket.sort_unstable();
            bucket.dedup();
            indices.extend(bucket.iter().copied());
            offsets.push(indices.len());
        }

        Self::try_new(source_count, target_count, offsets, indices)
    }

    pub fn try_from_assignments(
        target_count: usize,
        assignments: &[usize],
    ) -> Result<Self, GraphRoutingError> {
        let edges = assignments
            .iter()
            .copied()
            .enumerate()
            .map(|(source, target)| (source, target))
            .collect::<Vec<_>>();
        Self::try_from_edges(assignments.len(), target_count, &edges)
    }

    pub fn source_count(&self) -> usize {
        self.source_count
    }

    pub fn target_count(&self) -> usize {
        self.target_count
    }

    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn edge_count(&self) -> usize {
        self.indices.len()
    }

    pub fn degree(&self, source: usize) -> Option<usize> {
        self.neighbors(source).map(|neighbors| neighbors.len())
    }

    pub fn neighbors(&self, source: usize) -> Option<&[usize]> {
        if source >= self.source_count {
            return None;
        }
        let start = self.offsets[source];
        let end = self.offsets[source + 1];
        Some(&self.indices[start..end])
    }

    pub fn has_edge(&self, source: usize, target: usize) -> bool {
        self.neighbors(source)
            .is_some_and(|neighbors| neighbors.binary_search(&target).is_ok())
    }

    pub fn transpose(&self) -> Self {
        let mut counts = vec![0usize; self.target_count];
        for &target in &self.indices {
            counts[target] += 1;
        }

        let mut offsets = Vec::with_capacity(self.target_count + 1);
        offsets.push(0);
        for count in &counts {
            offsets.push(offsets.last().copied().unwrap_or(0) + count);
        }

        let mut indices = vec![0usize; self.indices.len()];
        let mut cursor = offsets[..self.target_count].to_vec();
        for source in 0..self.source_count {
            let start = self.offsets[source];
            let end = self.offsets[source + 1];
            for &target in &self.indices[start..end] {
                let write_index = cursor[target];
                indices[write_index] = source;
                cursor[target] += 1;
            }
        }

        Self {
            source_count: self.target_count,
            target_count: self.source_count,
            offsets,
            indices,
        }
    }
}

/// Concrete graph routing metadata over the generic primary/context/global banks.
///
/// The graph crate owns these node/cluster/global routes and can later map them
/// to graph-specific kernels without pushing CSR semantics into core.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphTopologyRouting {
    node_neighbors: GraphCsrAdjacency,
    node_to_cluster: Option<GraphCsrAdjacency>,
    cluster_to_node: Option<GraphCsrAdjacency>,
    node_to_global: Option<GraphCsrAdjacency>,
    global_to_node: Option<GraphCsrAdjacency>,
    cluster_to_global: Option<GraphCsrAdjacency>,
    global_to_cluster: Option<GraphCsrAdjacency>,
}

impl GraphTopologyRouting {
    pub fn new(node_neighbors: GraphCsrAdjacency) -> Result<Self, GraphRoutingError> {
        if node_neighbors.target_count() != node_neighbors.source_count() {
            return Err(GraphRoutingError::CountMismatch {
                field: "node_neighbors.target_count",
                expected: node_neighbors.source_count(),
                actual: node_neighbors.target_count(),
            });
        }
        Ok(Self {
            node_neighbors,
            node_to_cluster: None,
            cluster_to_node: None,
            node_to_global: None,
            global_to_node: None,
            cluster_to_global: None,
            global_to_cluster: None,
        })
    }

    pub fn with_node_cluster(
        mut self,
        node_to_cluster: GraphCsrAdjacency,
    ) -> Result<Self, GraphRoutingError> {
        self.expect_source_count(
            "node_to_cluster.source_count",
            &node_to_cluster,
            self.node_count(),
        )?;
        self.cluster_to_node = Some(node_to_cluster.transpose());
        self.node_to_cluster = Some(node_to_cluster);
        Ok(self)
    }

    pub fn with_cluster_assignments(
        self,
        cluster_count: usize,
        assignments: &[usize],
    ) -> Result<Self, GraphRoutingError> {
        self.with_node_cluster(GraphCsrAdjacency::try_from_assignments(
            cluster_count,
            assignments,
        )?)
    }

    pub fn with_node_global(
        mut self,
        node_to_global: GraphCsrAdjacency,
    ) -> Result<Self, GraphRoutingError> {
        self.expect_source_count(
            "node_to_global.source_count",
            &node_to_global,
            self.node_count(),
        )?;
        self.global_to_node = Some(node_to_global.transpose());
        self.node_to_global = Some(node_to_global);
        Ok(self)
    }

    pub fn with_node_global_assignments(
        self,
        global_count: usize,
        assignments: &[usize],
    ) -> Result<Self, GraphRoutingError> {
        self.with_node_global(GraphCsrAdjacency::try_from_assignments(
            global_count,
            assignments,
        )?)
    }

    pub fn with_cluster_global(
        mut self,
        cluster_to_global: GraphCsrAdjacency,
    ) -> Result<Self, GraphRoutingError> {
        if let Some(node_to_cluster) = &self.node_to_cluster {
            self.expect_source_count(
                "cluster_to_global.source_count",
                &cluster_to_global,
                node_to_cluster.target_count(),
            )?;
        }
        self.global_to_cluster = Some(cluster_to_global.transpose());
        self.cluster_to_global = Some(cluster_to_global);
        Ok(self)
    }

    pub fn with_cluster_global_assignments(
        self,
        global_count: usize,
        assignments: &[usize],
    ) -> Result<Self, GraphRoutingError> {
        self.with_cluster_global(GraphCsrAdjacency::try_from_assignments(
            global_count,
            assignments,
        )?)
    }

    pub fn node_neighbors(&self) -> &GraphCsrAdjacency {
        &self.node_neighbors
    }

    pub fn node_to_cluster(&self) -> Option<&GraphCsrAdjacency> {
        self.node_to_cluster.as_ref()
    }

    pub fn cluster_to_node(&self) -> Option<&GraphCsrAdjacency> {
        self.cluster_to_node.as_ref()
    }

    pub fn node_to_global(&self) -> Option<&GraphCsrAdjacency> {
        self.node_to_global.as_ref()
    }

    pub fn global_to_node(&self) -> Option<&GraphCsrAdjacency> {
        self.global_to_node.as_ref()
    }

    pub fn cluster_to_global(&self) -> Option<&GraphCsrAdjacency> {
        self.cluster_to_global.as_ref()
    }

    pub fn global_to_cluster(&self) -> Option<&GraphCsrAdjacency> {
        self.global_to_cluster.as_ref()
    }

    pub fn node_count(&self) -> usize {
        self.node_neighbors.source_count()
    }

    pub fn layout(&self) -> Result<GraphTopologyLayout, GraphRoutingError> {
        let node_count = self.node_count();

        if let Some(route) = &self.node_to_cluster {
            self.expect_source_count("node_to_cluster.source_count", route, node_count)?;
        }
        if let Some(route) = &self.cluster_to_node {
            self.expect_target_count("cluster_to_node.target_count", route, node_count)?;
        }
        if let Some(route) = &self.node_to_global {
            self.expect_source_count("node_to_global.source_count", route, node_count)?;
        }
        if let Some(route) = &self.global_to_node {
            self.expect_target_count("global_to_node.target_count", route, node_count)?;
        }

        let cluster_count = self.resolve_count(
            "cluster_count",
            &[
                self.node_to_cluster
                    .as_ref()
                    .map(|route| route.target_count()),
                self.cluster_to_node
                    .as_ref()
                    .map(|route| route.source_count()),
                self.cluster_to_global
                    .as_ref()
                    .map(|route| route.source_count()),
                self.global_to_cluster
                    .as_ref()
                    .map(|route| route.target_count()),
            ],
        )?;

        let global_count = self.resolve_count(
            "global_count",
            &[
                self.node_to_global
                    .as_ref()
                    .map(|route| route.target_count()),
                self.global_to_node
                    .as_ref()
                    .map(|route| route.source_count()),
                self.cluster_to_global
                    .as_ref()
                    .map(|route| route.target_count()),
                self.global_to_cluster
                    .as_ref()
                    .map(|route| route.source_count()),
            ],
        )?;

        Ok(GraphTopologyLayout::new(
            node_count,
            cluster_count.unwrap_or(0),
            global_count.unwrap_or(0),
        ))
    }

    pub fn routing_spec(&self) -> Result<StructuredRoutingSpec, GraphRoutingError> {
        let layout = self.layout()?;
        let mut spec = StructuredRoutingSpec::new().with_route(StructuredRouteSpec::new(
            StructuredBankRole::Primary,
            StructuredBankRole::Primary,
            StructuredRouteOperation::Read,
            StructuredRoutePattern::Sparse,
        ));
        spec = spec.with_route(StructuredRouteSpec::new(
            StructuredBankRole::Primary,
            StructuredBankRole::Primary,
            StructuredRouteOperation::Write,
            StructuredRoutePattern::Identity,
        ));
        if layout.cluster_count > 0 {
            if self.node_to_cluster.is_some() {
                spec = spec.with_route(StructuredRouteSpec::new(
                    StructuredBankRole::Primary,
                    StructuredBankRole::Context,
                    StructuredRouteOperation::Write,
                    StructuredRoutePattern::Sparse,
                ));
            }
            if self.cluster_to_node.is_some() {
                spec = spec.with_route(StructuredRouteSpec::new(
                    StructuredBankRole::Context,
                    StructuredBankRole::Primary,
                    StructuredRouteOperation::Read,
                    StructuredRoutePattern::Sparse,
                ));
            }
        }
        if layout.global_count > 0 {
            if self.node_to_global.is_some() {
                spec = spec.with_route(StructuredRouteSpec::new(
                    StructuredBankRole::Primary,
                    StructuredBankRole::Global,
                    StructuredRouteOperation::Write,
                    StructuredRoutePattern::Sparse,
                ));
            }
            if self.global_to_node.is_some() {
                spec = spec.with_route(StructuredRouteSpec::new(
                    StructuredBankRole::Global,
                    StructuredBankRole::Primary,
                    StructuredRouteOperation::Read,
                    StructuredRoutePattern::Sparse,
                ));
            }
            if self.cluster_to_global.is_some() {
                spec = spec.with_route(StructuredRouteSpec::new(
                    StructuredBankRole::Context,
                    StructuredBankRole::Global,
                    StructuredRouteOperation::Write,
                    StructuredRoutePattern::Sparse,
                ));
            }
            if self.global_to_cluster.is_some() {
                spec = spec.with_route(StructuredRouteSpec::new(
                    StructuredBankRole::Global,
                    StructuredBankRole::Context,
                    StructuredRouteOperation::Read,
                    StructuredRoutePattern::Sparse,
                ));
            }
        }
        Ok(spec)
    }

    fn expect_source_count(
        &self,
        field: &'static str,
        route: &GraphCsrAdjacency,
        expected: usize,
    ) -> Result<(), GraphRoutingError> {
        if route.source_count() != expected {
            return Err(GraphRoutingError::CountMismatch {
                field,
                expected,
                actual: route.source_count(),
            });
        }
        Ok(())
    }

    fn expect_target_count(
        &self,
        field: &'static str,
        route: &GraphCsrAdjacency,
        expected: usize,
    ) -> Result<(), GraphRoutingError> {
        if route.target_count() != expected {
            return Err(GraphRoutingError::CountMismatch {
                field,
                expected,
                actual: route.target_count(),
            });
        }
        Ok(())
    }

    fn resolve_count(
        &self,
        field: &'static str,
        values: &[Option<usize>],
    ) -> Result<Option<usize>, GraphRoutingError> {
        let mut expected = None;
        for value in values.iter().flatten().copied() {
            match expected {
                Some(previous) if previous != value => {
                    return Err(GraphRoutingError::CountMismatch {
                        field,
                        expected: previous,
                        actual: value,
                    });
                }
                None => expected = Some(value),
                Some(_) => {}
            }
        }
        Ok(expected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_from_edges_sorts_dedups_and_transposes() {
        let csr = GraphCsrAdjacency::try_from_edges(3, 3, &[(0, 2), (0, 1), (0, 1), (2, 0)])
            .expect("valid CSR adjacency");

        assert_eq!(csr.offsets(), &[0, 2, 2, 3]);
        assert_eq!(csr.indices(), &[1, 2, 0]);
        assert_eq!(csr.edge_count(), 3);
        assert_eq!(csr.degree(0), Some(2));
        assert_eq!(csr.degree(1), Some(0));
        assert_eq!(csr.neighbors(2), Some(&[0][..]));
        assert!(csr.has_edge(0, 1));
        assert!(!csr.has_edge(1, 2));

        let transpose = csr.transpose();
        assert_eq!(transpose.offsets(), &[0, 1, 2, 3]);
        assert_eq!(transpose.indices(), &[2, 0, 0]);
        assert!(transpose.has_edge(1, 0));
        assert!(transpose.has_edge(0, 2));
    }

    #[test]
    fn graph_topology_routing_derives_layout_and_sparse_routes() {
        let node_neighbors =
            GraphCsrAdjacency::try_from_edges(4, 4, &[(0, 1), (1, 2), (2, 3), (3, 0)])
                .expect("valid node neighbor graph");
        let routing = GraphTopologyRouting::new(node_neighbors)
            .expect("node routing should be valid")
            .with_cluster_assignments(2, &[0, 0, 1, 1])
            .expect("cluster assignments should be valid")
            .with_cluster_global_assignments(1, &[0, 0])
            .expect("global assignments should be valid");

        assert_eq!(routing.layout(), Ok(GraphTopologyLayout::new(4, 2, 1)));
        assert_eq!(
            routing.node_to_cluster().map(GraphCsrAdjacency::edge_count),
            Some(4)
        );
        assert_eq!(
            routing.cluster_to_node().map(GraphCsrAdjacency::edge_count),
            Some(4)
        );
        assert_eq!(
            routing
                .cluster_to_global()
                .map(GraphCsrAdjacency::edge_count),
            Some(2)
        );
        assert_eq!(
            routing
                .global_to_cluster()
                .map(GraphCsrAdjacency::edge_count),
            Some(2)
        );

        let spec = routing
            .routing_spec()
            .expect("routing spec should validate");
        assert!(spec.contains(StructuredRouteSpec::new(
            StructuredBankRole::Primary,
            StructuredBankRole::Primary,
            StructuredRouteOperation::Read,
            StructuredRoutePattern::Sparse,
        )));
        assert!(spec.contains(StructuredRouteSpec::new(
            StructuredBankRole::Primary,
            StructuredBankRole::Context,
            StructuredRouteOperation::Write,
            StructuredRoutePattern::Sparse,
        )));
        assert!(spec.contains(StructuredRouteSpec::new(
            StructuredBankRole::Context,
            StructuredBankRole::Global,
            StructuredRouteOperation::Write,
            StructuredRoutePattern::Sparse,
        )));
        assert!(spec.contains(StructuredRouteSpec::new(
            StructuredBankRole::Global,
            StructuredBankRole::Context,
            StructuredRouteOperation::Read,
            StructuredRoutePattern::Sparse,
        )));
    }

    #[test]
    fn graph_topology_routing_rejects_node_neighbor_count_mismatch() {
        let node_neighbors =
            GraphCsrAdjacency::try_from_edges(3, 2, &[(0, 1), (1, 0)]).expect("valid CSR");
        let err = GraphTopologyRouting::new(node_neighbors)
            .expect_err("node-neighbor routing must be square over nodes");

        assert_eq!(
            err,
            GraphRoutingError::CountMismatch {
                field: "node_neighbors.target_count",
                expected: 3,
                actual: 2,
            }
        );
    }
}
