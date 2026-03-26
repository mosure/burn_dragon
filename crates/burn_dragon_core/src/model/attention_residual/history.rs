use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

#[derive(Clone, Debug)]
pub(crate) struct ResidualHistory<B: Backend> {
    entries: Option<Vec<Tensor<B, 4>>>,
}

impl<B: Backend> ResidualHistory<B> {
    pub(crate) fn disabled() -> Self {
        Self { entries: None }
    }

    pub(crate) fn from_anchor(anchor: Tensor<B, 4>) -> Self {
        Self {
            entries: Some(vec![anchor]),
        }
    }

    pub(crate) fn from_anchor_if_enabled(enabled: bool, anchor: &Tensor<B, 4>) -> Self {
        if enabled {
            Self::from_anchor(anchor.clone())
        } else {
            Self::disabled()
        }
    }

    pub(crate) fn from_entries(entries: Vec<Tensor<B, 4>>) -> Self {
        if entries.is_empty() {
            Self::disabled()
        } else {
            Self {
                entries: Some(entries),
            }
        }
    }

    pub(crate) fn into_entries(self) -> Vec<Tensor<B, 4>> {
        self.entries.unwrap_or_default()
    }

    pub(crate) fn as_slice(&self) -> &[Tensor<B, 4>] {
        self.entries.as_deref().unwrap_or(&[])
    }

    pub(crate) fn capture_previous(&self, current: &Tensor<B, 4>) -> Option<Tensor<B, 4>> {
        self.entries.as_ref().map(|_| current.clone())
    }

    pub(crate) fn push_delta_from(&mut self, previous: Option<Tensor<B, 4>>, next: &Tensor<B, 4>) {
        if let (Some(entries), Some(previous)) = (self.entries.as_mut(), previous) {
            entries.push(next.clone().sub(previous));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ResidualHistory;
    use burn::tensor::{Tensor, TensorData};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn residual_history_starts_with_anchor() {
        let device = Default::default();
        let anchor = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 1, 4]),
            &device,
        );
        let history = ResidualHistory::from_anchor(anchor.clone());
        history.as_slice()[0]
            .clone()
            .into_data()
            .assert_eq(&anchor.into_data(), false);
    }

    #[test]
    fn residual_history_pushes_deltas_not_next_states() {
        let device = Default::default();
        let anchor = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 1, 4]),
            &device,
        );
        let previous = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 1, 4]),
            &device,
        );
        let next = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![1.5, 2.5, 4.0, 5.5], [1, 1, 1, 4]),
            &device,
        );
        let expected_delta = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![0.5, 0.5, 1.0, 1.5], [1, 1, 1, 4]),
            &device,
        );

        let mut history = ResidualHistory::from_anchor(anchor);
        history.push_delta_from(Some(previous), &next);

        history.as_slice()[1]
            .clone()
            .into_data()
            .assert_eq(&expected_delta.into_data(), false);
    }

    #[test]
    fn disabled_history_is_empty_and_noop() {
        let device = Default::default();
        let current = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 1, 4]),
            &device,
        );

        let mut history = ResidualHistory::disabled();
        let previous = history.capture_previous(&current);
        history.push_delta_from(previous, &current);

        assert!(history.as_slice().is_empty());
        assert!(history.into_entries().is_empty());
    }
}
