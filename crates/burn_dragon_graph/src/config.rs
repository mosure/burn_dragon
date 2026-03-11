use burn_dragon_core::{
    StructuredBankRole, StructuredRouteOperation, StructuredRoutePattern, StructuredRouteSpec,
    StructuredRoutingSpec,
};

use crate::routing::{GraphRoutingError, GraphTopologyRouting};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphTopologyConfig {
    pub cluster_count: usize,
    pub global_count: usize,
    pub node_neighbor_pattern: StructuredRoutePattern,
    pub cluster_read_pattern: StructuredRoutePattern,
    pub cluster_write_pattern: StructuredRoutePattern,
    pub global_read_pattern: StructuredRoutePattern,
    pub global_write_pattern: StructuredRoutePattern,
}

impl Default for GraphTopologyConfig {
    fn default() -> Self {
        Self {
            cluster_count: 1,
            global_count: 1,
            node_neighbor_pattern: StructuredRoutePattern::Sparse,
            cluster_read_pattern: StructuredRoutePattern::Broadcast,
            cluster_write_pattern: StructuredRoutePattern::Pool,
            global_read_pattern: StructuredRoutePattern::Broadcast,
            global_write_pattern: StructuredRoutePattern::Pool,
        }
    }
}

impl GraphTopologyConfig {
    pub fn routing_spec(&self) -> StructuredRoutingSpec {
        let mut spec = StructuredRoutingSpec::new()
            .with_route(StructuredRouteSpec::new(
                StructuredBankRole::Primary,
                StructuredBankRole::Primary,
                StructuredRouteOperation::Read,
                self.node_neighbor_pattern,
            ))
            .with_route(StructuredRouteSpec::new(
                StructuredBankRole::Primary,
                StructuredBankRole::Primary,
                StructuredRouteOperation::Write,
                StructuredRoutePattern::Identity,
            ));

        if self.cluster_count > 0 {
            spec = spec
                .with_route(StructuredRouteSpec::new(
                    StructuredBankRole::Context,
                    StructuredBankRole::Primary,
                    StructuredRouteOperation::Read,
                    self.cluster_read_pattern,
                ))
                .with_route(StructuredRouteSpec::new(
                    StructuredBankRole::Primary,
                    StructuredBankRole::Context,
                    StructuredRouteOperation::Write,
                    self.cluster_write_pattern,
                ));
        }

        if self.global_count > 0 {
            spec = spec
                .with_route(StructuredRouteSpec::new(
                    StructuredBankRole::Global,
                    StructuredBankRole::Primary,
                    StructuredRouteOperation::Read,
                    self.global_read_pattern,
                ))
                .with_route(StructuredRouteSpec::new(
                    StructuredBankRole::Primary,
                    StructuredBankRole::Global,
                    StructuredRouteOperation::Write,
                    self.global_write_pattern,
                ));

            if self.cluster_count > 0 {
                spec = spec.with_route(StructuredRouteSpec::new(
                    StructuredBankRole::Context,
                    StructuredBankRole::Global,
                    StructuredRouteOperation::Write,
                    self.global_write_pattern,
                ));
            }
        }

        spec
    }

    pub fn routing_spec_for(
        &self,
        routing: &GraphTopologyRouting,
    ) -> Result<StructuredRoutingSpec, GraphRoutingError> {
        let layout = routing.layout()?;
        if layout.cluster_count != self.cluster_count {
            return Err(GraphRoutingError::CountMismatch {
                field: "cluster_count",
                expected: self.cluster_count,
                actual: layout.cluster_count,
            });
        }
        if layout.global_count != self.global_count {
            return Err(GraphRoutingError::CountMismatch {
                field: "global_count",
                expected: self.global_count,
                actual: layout.global_count,
            });
        }

        if self.cluster_count > 0
            && (routing.node_to_cluster().is_none() || routing.cluster_to_node().is_none())
        {
            return Err(GraphRoutingError::MissingRoute {
                route: "primary<->context",
            });
        }
        if self.global_count > 0
            && routing.node_to_global().is_none()
            && routing.cluster_to_global().is_none()
        {
            return Err(GraphRoutingError::MissingRoute {
                route: "(* -> global) write",
            });
        }
        if self.global_count > 0
            && routing.global_to_node().is_none()
            && routing.global_to_cluster().is_none()
        {
            return Err(GraphRoutingError::MissingRoute {
                route: "(global -> *) read",
            });
        }

        let mut spec = StructuredRoutingSpec::new()
            .with_route(StructuredRouteSpec::new(
                StructuredBankRole::Primary,
                StructuredBankRole::Primary,
                StructuredRouteOperation::Read,
                self.node_neighbor_pattern,
            ))
            .with_route(StructuredRouteSpec::new(
                StructuredBankRole::Primary,
                StructuredBankRole::Primary,
                StructuredRouteOperation::Write,
                StructuredRoutePattern::Identity,
            ));

        if routing.node_to_cluster().is_some() {
            spec = spec.with_route(StructuredRouteSpec::new(
                StructuredBankRole::Primary,
                StructuredBankRole::Context,
                StructuredRouteOperation::Write,
                self.cluster_write_pattern,
            ));
        }
        if routing.cluster_to_node().is_some() {
            spec = spec.with_route(StructuredRouteSpec::new(
                StructuredBankRole::Context,
                StructuredBankRole::Primary,
                StructuredRouteOperation::Read,
                self.cluster_read_pattern,
            ));
        }
        if routing.node_to_global().is_some() {
            spec = spec.with_route(StructuredRouteSpec::new(
                StructuredBankRole::Primary,
                StructuredBankRole::Global,
                StructuredRouteOperation::Write,
                self.global_write_pattern,
            ));
        }
        if routing.global_to_node().is_some() {
            spec = spec.with_route(StructuredRouteSpec::new(
                StructuredBankRole::Global,
                StructuredBankRole::Primary,
                StructuredRouteOperation::Read,
                self.global_read_pattern,
            ));
        }
        if routing.cluster_to_global().is_some() {
            spec = spec.with_route(StructuredRouteSpec::new(
                StructuredBankRole::Context,
                StructuredBankRole::Global,
                StructuredRouteOperation::Write,
                self.global_write_pattern,
            ));
        }
        if routing.global_to_cluster().is_some() {
            spec = spec.with_route(StructuredRouteSpec::new(
                StructuredBankRole::Global,
                StructuredBankRole::Context,
                StructuredRouteOperation::Read,
                self.global_read_pattern,
            ));
        }

        Ok(spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphCsrAdjacency, GraphTopologyRouting};

    #[test]
    fn graph_topology_config_reports_sparse_node_cluster_global_routes() {
        let config = GraphTopologyConfig::default();
        let spec = config.routing_spec();

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
            StructuredRoutePattern::Pool,
        )));
        assert!(spec.contains(StructuredRouteSpec::new(
            StructuredBankRole::Global,
            StructuredBankRole::Primary,
            StructuredRouteOperation::Read,
            StructuredRoutePattern::Broadcast,
        )));
    }

    #[test]
    fn graph_topology_config_maps_sparse_routing_into_route_patterns() {
        let config = GraphTopologyConfig {
            cluster_count: 2,
            ..Default::default()
        };
        let routing = GraphTopologyRouting::new(
            GraphCsrAdjacency::try_from_edges(4, 4, &[(0, 1), (1, 2), (2, 3), (3, 0)])
                .expect("valid node-neighbor CSR"),
        )
        .expect("valid node routing")
        .with_cluster_assignments(2, &[0, 0, 1, 1])
        .expect("valid cluster assignments")
        .with_cluster_global_assignments(1, &[0, 0])
        .expect("valid global assignments");

        let spec = config
            .routing_spec_for(&routing)
            .expect("config should accept matching graph routing");

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
            StructuredRoutePattern::Pool,
        )));
        assert!(spec.contains(StructuredRouteSpec::new(
            StructuredBankRole::Context,
            StructuredBankRole::Primary,
            StructuredRouteOperation::Read,
            StructuredRoutePattern::Broadcast,
        )));
        assert!(spec.contains(StructuredRouteSpec::new(
            StructuredBankRole::Context,
            StructuredBankRole::Global,
            StructuredRouteOperation::Write,
            StructuredRoutePattern::Pool,
        )));
        assert!(spec.contains(StructuredRouteSpec::new(
            StructuredBankRole::Global,
            StructuredBankRole::Context,
            StructuredRouteOperation::Read,
            StructuredRoutePattern::Broadcast,
        )));
    }

    #[test]
    fn graph_topology_config_rejects_cluster_count_mismatch() {
        let config = GraphTopologyConfig {
            cluster_count: 3,
            ..Default::default()
        };
        let routing = GraphTopologyRouting::new(
            GraphCsrAdjacency::try_from_edges(4, 4, &[(0, 1), (1, 0)])
                .expect("valid node-neighbor CSR"),
        )
        .expect("valid node routing")
        .with_cluster_assignments(2, &[0, 0, 1, 1])
        .expect("valid cluster assignments")
        .with_cluster_global_assignments(1, &[0, 0])
        .expect("valid global assignments");

        let err = config
            .routing_spec_for(&routing)
            .expect_err("cluster count mismatch must be rejected");
        assert_eq!(
            err,
            GraphRoutingError::CountMismatch {
                field: "cluster_count",
                expected: 3,
                actual: 2,
            }
        );
    }
}
