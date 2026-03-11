use burn::prelude::*;

#[derive(Clone)]
pub struct BankedRhoState<B: Backend> {
    pub primary_rho: Tensor<B, 5>,
    pub context_rho: Tensor<B, 5>,
    pub global_rho: Tensor<B, 4>,
}

impl<B: Backend> BankedRhoState<B> {
    pub fn primary_rho(&self) -> &Tensor<B, 5> {
        &self.primary_rho
    }

    pub fn primary_rho_mut(&mut self) -> &mut Tensor<B, 5> {
        &mut self.primary_rho
    }

    pub fn context_rho(&self) -> &Tensor<B, 5> {
        &self.context_rho
    }

    pub fn context_rho_mut(&mut self) -> &mut Tensor<B, 5> {
        &mut self.context_rho
    }

    pub fn global_rho(&self) -> &Tensor<B, 4> {
        &self.global_rho
    }

    pub fn global_rho_mut(&mut self) -> &mut Tensor<B, 4> {
        &mut self.global_rho
    }

    pub fn patch_rho(&self) -> &Tensor<B, 5> {
        self.primary_rho()
    }

    pub fn patch_rho_mut(&mut self) -> &mut Tensor<B, 5> {
        self.primary_rho_mut()
    }

    pub fn coarse_rho(&self) -> &Tensor<B, 5> {
        self.context_rho()
    }

    pub fn coarse_rho_mut(&mut self) -> &mut Tensor<B, 5> {
        self.context_rho_mut()
    }

    pub fn hub_rho(&self) -> &Tensor<B, 4> {
        self.global_rho()
    }

    pub fn hub_rho_mut(&mut self) -> &mut Tensor<B, 4> {
        self.global_rho_mut()
    }

    pub fn detach(&self) -> Self {
        Self {
            primary_rho: self.primary_rho.clone().detach(),
            context_rho: self.context_rho.clone().detach(),
            global_rho: self.global_rho.clone().detach(),
        }
    }
}

#[derive(Clone)]
/// Topology-agnostic recurrent state for structured backbones.
///
/// The current vision pyramid path specializes this as patch/coarse/hub, while
/// future graph or grouped-constraint adapters can reuse the same contract with
/// different routing semantics.
pub struct StructuredTopologyState<B: Backend> {
    pub primary_state: Tensor<B, 4>,
    pub context_state: Tensor<B, 4>,
    pub rho: BankedRhoState<B>,
    pub temporal_position: usize,
    pub prediction_age: usize,
}

/// Compatibility alias for existing grid-shaped consumers.
pub type StructuredGridState<B> = StructuredTopologyState<B>;

impl<B: Backend> StructuredTopologyState<B> {
    pub fn primary_state(&self) -> &Tensor<B, 4> {
        &self.primary_state
    }

    pub fn primary_state_mut(&mut self) -> &mut Tensor<B, 4> {
        &mut self.primary_state
    }

    pub fn context_state(&self) -> &Tensor<B, 4> {
        &self.context_state
    }

    pub fn context_state_mut(&mut self) -> &mut Tensor<B, 4> {
        &mut self.context_state
    }

    pub fn primary_rho(&self) -> &Tensor<B, 5> {
        self.rho.primary_rho()
    }

    pub fn primary_rho_mut(&mut self) -> &mut Tensor<B, 5> {
        self.rho.primary_rho_mut()
    }

    pub fn context_rho(&self) -> &Tensor<B, 5> {
        self.rho.context_rho()
    }

    pub fn context_rho_mut(&mut self) -> &mut Tensor<B, 5> {
        self.rho.context_rho_mut()
    }

    pub fn global_rho(&self) -> &Tensor<B, 4> {
        self.rho.global_rho()
    }

    pub fn global_rho_mut(&mut self) -> &mut Tensor<B, 4> {
        self.rho.global_rho_mut()
    }

    pub fn patch_state(&self) -> &Tensor<B, 4> {
        self.primary_state()
    }

    pub fn patch_state_mut(&mut self) -> &mut Tensor<B, 4> {
        self.primary_state_mut()
    }

    pub fn coarse_state(&self) -> &Tensor<B, 4> {
        self.context_state()
    }

    pub fn coarse_state_mut(&mut self) -> &mut Tensor<B, 4> {
        self.context_state_mut()
    }

    pub fn patch_rho(&self) -> &Tensor<B, 5> {
        self.primary_rho()
    }

    pub fn patch_rho_mut(&mut self) -> &mut Tensor<B, 5> {
        self.primary_rho_mut()
    }

    pub fn coarse_rho(&self) -> &Tensor<B, 5> {
        self.context_rho()
    }

    pub fn coarse_rho_mut(&mut self) -> &mut Tensor<B, 5> {
        self.context_rho_mut()
    }

    pub fn hub_rho(&self) -> &Tensor<B, 4> {
        self.global_rho()
    }

    pub fn hub_rho_mut(&mut self) -> &mut Tensor<B, 4> {
        self.global_rho_mut()
    }

    pub fn detach(&self) -> Self {
        Self {
            primary_state: self.primary_state.clone().detach(),
            context_state: self.context_state.clone().detach(),
            rho: self.rho.detach(),
            temporal_position: self.temporal_position,
            prediction_age: self.prediction_age,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn::tensor::backend::Backend as BackendTrait;
    use burn_ndarray::NdArray;

    #[test]
    fn structured_topology_state_preserves_vision_compat_aliases() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();

        let state = StructuredTopologyState {
            primary_state: Tensor::<Backend, 4>::from_data(
                TensorData::new(vec![1.0; 8], [1, 2, 2, 2]),
                &device,
            ),
            context_state: Tensor::<Backend, 4>::from_data(
                TensorData::new(vec![2.0; 4], [1, 2, 1, 2]),
                &device,
            ),
            rho: BankedRhoState {
                primary_rho: Tensor::<Backend, 5>::from_data(
                    TensorData::new(vec![3.0; 8], [1, 2, 2, 1, 2]),
                    &device,
                ),
                context_rho: Tensor::<Backend, 5>::from_data(
                    TensorData::new(vec![4.0; 4], [1, 2, 1, 1, 2]),
                    &device,
                ),
                global_rho: Tensor::<Backend, 4>::from_data(
                    TensorData::new(vec![5.0; 4], [1, 1, 2, 2]),
                    &device,
                ),
            },
            temporal_position: 7,
            prediction_age: 3,
        };

        assert_eq!(
            state.patch_state().shape().dims::<4>(),
            state.primary_state().shape().dims::<4>()
        );
        assert_eq!(
            state.coarse_state().shape().dims::<4>(),
            state.context_state().shape().dims::<4>()
        );
        assert_eq!(
            state.patch_rho().shape().dims::<5>(),
            state.primary_rho().shape().dims::<5>()
        );
        assert_eq!(
            state.coarse_rho().shape().dims::<5>(),
            state.context_rho().shape().dims::<5>()
        );
        assert_eq!(
            state.hub_rho().shape().dims::<4>(),
            state.global_rho().shape().dims::<4>()
        );
        assert_eq!(state.temporal_position, 7);
        assert_eq!(state.prediction_age, 3);
    }
}
