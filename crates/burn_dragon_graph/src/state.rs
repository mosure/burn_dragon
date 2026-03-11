use burn::prelude::*;
use burn_dragon_core::{BankedRhoState, StructuredTopologyState};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphTopologyLayout {
    pub node_count: usize,
    pub cluster_count: usize,
    pub global_count: usize,
}

impl GraphTopologyLayout {
    pub const fn new(node_count: usize, cluster_count: usize, global_count: usize) -> Self {
        Self {
            node_count,
            cluster_count,
            global_count,
        }
    }
}

/// Graph adapter view over the generic structured topology state.
///
/// Paper mapping:
/// - `node_state` / `cluster_state` are dense-space activations
/// - `node_rho` / `cluster_rho` / `global_rho` are persistent synaptic state banks
/// - `temporal_position` / `prediction_age` define the recurrent axis for `observe`/`refine`/`predict`
#[derive(Clone)]
pub struct GraphTopologyState<B: Backend> {
    inner: StructuredTopologyState<B>,
    layout: GraphTopologyLayout,
}

impl<B: Backend> GraphTopologyState<B> {
    pub fn from_parts(
        node_state: Tensor<B, 3>,
        cluster_state: Tensor<B, 3>,
        node_rho: Tensor<B, 4>,
        cluster_rho: Tensor<B, 4>,
        global_rho: Tensor<B, 4>,
        temporal_position: usize,
        prediction_age: usize,
    ) -> Self {
        let [batch, node_count, dim] = node_state.shape().dims::<3>();
        let [cluster_batch, cluster_count, cluster_dim] = cluster_state.shape().dims::<3>();
        let [rho_batch, _rank, _value_dim, rho_nodes] = node_rho.shape().dims::<4>();
        let [
            cluster_rho_batch,
            _cluster_rank,
            _cluster_value_dim,
            rho_clusters,
        ] = cluster_rho.shape().dims::<4>();
        let [global_batch, global_count, _global_rank, _global_value_dim] =
            global_rho.shape().dims::<4>();

        assert_eq!(
            cluster_batch, batch,
            "cluster state batch must match node state"
        );
        assert_eq!(cluster_dim, dim, "cluster state dim must match node state");
        assert_eq!(rho_batch, batch, "node rho batch must match node state");
        assert_eq!(
            rho_nodes, node_count,
            "node rho count must match node state"
        );
        assert_eq!(
            cluster_rho_batch, batch,
            "cluster rho batch must match node state"
        );
        assert_eq!(
            rho_clusters, cluster_count,
            "cluster rho count must match cluster state"
        );
        assert_eq!(
            global_batch, batch,
            "global rho batch must match node state"
        );

        let inner = StructuredTopologyState {
            primary_state: pack_state(node_state),
            context_state: pack_state(cluster_state),
            rho: BankedRhoState {
                primary_rho: pack_rho(node_rho),
                context_rho: pack_rho(cluster_rho),
                global_rho,
            },
            temporal_position,
            prediction_age,
        };
        let layout = GraphTopologyLayout::new(node_count, cluster_count, global_count);
        Self { inner, layout }
    }

    pub fn from_topology_state(
        inner: StructuredTopologyState<B>,
        layout: GraphTopologyLayout,
    ) -> Self {
        let [batch, dim, nodes, width] = inner.primary_state().shape().dims::<4>();
        let [context_batch, context_dim, clusters, context_width] =
            inner.context_state().shape().dims::<4>();
        let [rho_batch, _rank, _value_dim, rho_nodes, rho_width] =
            inner.primary_rho().shape().dims::<5>();
        let [
            context_rho_batch,
            _context_rank,
            _context_value_dim,
            rho_clusters,
            rho_context_width,
        ] = inner.context_rho().shape().dims::<5>();
        let [global_batch, global_count, _global_rank, _global_value_dim] =
            inner.global_rho().shape().dims::<4>();

        assert_eq!(
            width, 1,
            "graph primary state must use width=1 packed layout"
        );
        assert_eq!(
            context_width, 1,
            "graph context state must use width=1 packed layout"
        );
        assert_eq!(
            rho_width, 1,
            "graph primary rho must use width=1 packed layout"
        );
        assert_eq!(
            rho_context_width, 1,
            "graph context rho must use width=1 packed layout"
        );
        assert_eq!(batch, context_batch);
        assert_eq!(dim, context_dim);
        assert_eq!(rho_batch, batch);
        assert_eq!(rho_nodes, nodes);
        assert_eq!(context_rho_batch, batch);
        assert_eq!(rho_clusters, clusters);
        assert_eq!(global_batch, batch);
        assert_eq!(nodes, layout.node_count);
        assert_eq!(clusters, layout.cluster_count);
        assert_eq!(global_count, layout.global_count);

        Self { inner, layout }
    }

    pub fn topology_state(&self) -> &StructuredTopologyState<B> {
        &self.inner
    }

    pub fn into_topology_state(self) -> StructuredTopologyState<B> {
        self.inner
    }

    pub fn layout(&self) -> GraphTopologyLayout {
        self.layout
    }

    pub fn node_state(&self) -> Tensor<B, 3> {
        unpack_state(self.inner.primary_state().clone(), self.layout.node_count)
    }

    pub fn node_dense_state(&self) -> Tensor<B, 3> {
        self.node_state()
    }

    pub fn cluster_state(&self) -> Tensor<B, 3> {
        unpack_state(
            self.inner.context_state().clone(),
            self.layout.cluster_count,
        )
    }

    pub fn cluster_dense_state(&self) -> Tensor<B, 3> {
        self.cluster_state()
    }

    pub fn node_rho(&self) -> Tensor<B, 4> {
        unpack_rho(self.inner.primary_rho().clone(), self.layout.node_count)
    }

    pub fn cluster_rho(&self) -> Tensor<B, 4> {
        unpack_rho(self.inner.context_rho().clone(), self.layout.cluster_count)
    }

    pub fn global_rho(&self) -> Tensor<B, 4> {
        self.inner.global_rho().clone()
    }

    pub fn temporal_position(&self) -> usize {
        self.inner.temporal_position
    }

    pub fn prediction_age(&self) -> usize {
        self.inner.prediction_age
    }
}

pub(crate) fn rho_to_target_major<B: Backend>(rho: Tensor<B, 4>) -> Tensor<B, 4> {
    rho.swap_dims(1, 3).swap_dims(2, 3)
}

pub(crate) fn rho_from_target_major<B: Backend>(rho: Tensor<B, 4>) -> Tensor<B, 4> {
    rho.swap_dims(2, 3).swap_dims(1, 3)
}

fn pack_state<B: Backend>(state: Tensor<B, 3>) -> Tensor<B, 4> {
    state.swap_dims(1, 2).unsqueeze_dim::<4>(3)
}

fn unpack_state<B: Backend>(state: Tensor<B, 4>, expected_len: usize) -> Tensor<B, 3> {
    let [batch, dim, len, width] = state.shape().dims::<4>();
    assert_eq!(len, expected_len, "packed graph state length mismatch");
    assert_eq!(width, 1, "packed graph state width must be 1");
    state
        .squeeze_dim::<3>(3)
        .reshape([batch, dim, len])
        .swap_dims(1, 2)
}

fn pack_rho<B: Backend>(rho: Tensor<B, 4>) -> Tensor<B, 5> {
    rho.unsqueeze_dim::<5>(4)
}

fn unpack_rho<B: Backend>(rho: Tensor<B, 5>, expected_len: usize) -> Tensor<B, 4> {
    let [batch, rank, value_dim, len, width] = rho.shape().dims::<5>();
    assert_eq!(len, expected_len, "packed graph rho length mismatch");
    assert_eq!(width, 1, "packed graph rho width must be 1");
    rho.squeeze_dim::<4>(4)
        .reshape([batch, rank, value_dim, len])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn::tensor::backend::Backend as BackendTrait;
    use burn_ndarray::NdArray;

    #[test]
    fn graph_topology_state_roundtrips_node_cluster_global_contract() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();

        let node_state =
            Tensor::<Backend, 3>::from_data(TensorData::new(vec![1.0; 12], [1, 3, 4]), &device);
        let cluster_state =
            Tensor::<Backend, 3>::from_data(TensorData::new(vec![2.0; 8], [1, 2, 4]), &device);
        let node_rho =
            Tensor::<Backend, 4>::from_data(TensorData::new(vec![3.0; 18], [1, 2, 3, 3]), &device);
        let cluster_rho =
            Tensor::<Backend, 4>::from_data(TensorData::new(vec![4.0; 12], [1, 2, 3, 2]), &device);
        let global_rho =
            Tensor::<Backend, 4>::from_data(TensorData::new(vec![5.0; 6], [1, 1, 2, 3]), &device);

        let state = GraphTopologyState::from_parts(
            node_state.clone(),
            cluster_state.clone(),
            node_rho.clone(),
            cluster_rho.clone(),
            global_rho.clone(),
            4,
            2,
        );

        assert_eq!(state.layout(), GraphTopologyLayout::new(3, 2, 1));
        assert_eq!(state.node_state().shape().dims::<3>(), [1, 3, 4]);
        assert_eq!(state.cluster_state().shape().dims::<3>(), [1, 2, 4]);
        assert_eq!(state.node_rho().shape().dims::<4>(), [1, 2, 3, 3]);
        assert_eq!(state.cluster_rho().shape().dims::<4>(), [1, 2, 3, 2]);
        assert_eq!(state.global_rho().shape().dims::<4>(), [1, 1, 2, 3]);
        assert_eq!(state.temporal_position(), 4);
        assert_eq!(state.prediction_age(), 2);

        let inner = state.into_topology_state();
        let rebuilt =
            GraphTopologyState::from_topology_state(inner, GraphTopologyLayout::new(3, 2, 1));
        assert_eq!(
            rebuilt.node_state().shape().dims::<3>(),
            node_state.shape().dims::<3>()
        );
        assert_eq!(
            rebuilt.cluster_state().shape().dims::<3>(),
            cluster_state.shape().dims::<3>()
        );
        assert_eq!(
            rebuilt.node_rho().shape().dims::<4>(),
            node_rho.shape().dims::<4>()
        );
        assert_eq!(
            rebuilt.cluster_rho().shape().dims::<4>(),
            cluster_rho.shape().dims::<4>()
        );
        assert_eq!(
            rebuilt.global_rho().shape().dims::<4>(),
            global_rho.shape().dims::<4>()
        );
    }
}
