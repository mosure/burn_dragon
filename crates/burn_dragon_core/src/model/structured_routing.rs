#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum StructuredBankRole {
    Primary,
    Context,
    Global,
}

impl StructuredBankRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Primary => "primary",
            Self::Context => "context",
            Self::Global => "global",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum StructuredRouteOperation {
    Read,
    Write,
}

impl StructuredRouteOperation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum StructuredRoutePattern {
    Identity,
    Local,
    Pool,
    Broadcast,
    Sparse,
    Dense,
    Grouped,
}

impl StructuredRoutePattern {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Local => "local",
            Self::Pool => "pool",
            Self::Broadcast => "broadcast",
            Self::Sparse => "sparse",
            Self::Dense => "dense",
            Self::Grouped => "grouped",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct StructuredRouteSpec {
    pub source: StructuredBankRole,
    pub target: StructuredBankRole,
    pub operation: StructuredRouteOperation,
    pub pattern: StructuredRoutePattern,
}

impl StructuredRouteSpec {
    pub const fn new(
        source: StructuredBankRole,
        target: StructuredBankRole,
        operation: StructuredRouteOperation,
        pattern: StructuredRoutePattern,
    ) -> Self {
        Self {
            source,
            target,
            operation,
            pattern,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StructuredRoutingSpec {
    routes: Vec<StructuredRouteSpec>,
}

impl StructuredRoutingSpec {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn routes(&self) -> &[StructuredRouteSpec] {
        &self.routes
    }

    pub fn with_route(mut self, route: StructuredRouteSpec) -> Self {
        self.routes.push(route);
        self
    }

    pub fn contains(&self, route: StructuredRouteSpec) -> bool {
        self.routes.contains(&route)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_spec_tracks_routes() {
        let route = StructuredRouteSpec::new(
            StructuredBankRole::Primary,
            StructuredBankRole::Context,
            StructuredRouteOperation::Write,
            StructuredRoutePattern::Pool,
        );
        let spec = StructuredRoutingSpec::new().with_route(route);
        assert!(spec.contains(route));
        assert_eq!(spec.routes().len(), 1);
        assert_eq!(spec.routes()[0].pattern.as_str(), "pool");
    }
}
