use super::{
    AttentionResidual, AttentionResidualConfig, BlockAttentionResidual,
    BlockAttentionResidualConfig, BlockAttentionResidualSummaryMode,
};
use burn::tensor::{Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;

type TestBackend = NdArray<f32>;
type AutodiffBackend = Autodiff<NdArray<f32>>;

fn assert_tensor_finite<B: burn::tensor::backend::Backend, const D: usize>(tensor: Tensor<B, D>) {
    let values = tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("tensor vec");
    assert!(values.iter().all(|value| value.is_finite()));
}

#[test]
fn attention_residual_single_history_element_is_identity() {
    let device = Default::default();
    let config = AttentionResidualConfig {
        enabled: true,
        num_heads: 2,
        ..AttentionResidualConfig::default()
    };
    let connector = AttentionResidual::<TestBackend>::new(&config, 4, &device);
    let current = Tensor::<TestBackend, 4>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 1, 4]),
        &device,
    );
    let branch_input = connector.branch_input(current.clone(), std::slice::from_ref(&current));
    branch_input
        .into_data()
        .assert_eq(&current.into_data(), false);
}

#[test]
fn attention_residual_history_window_limits_candidates() {
    let device = Default::default();
    let config = AttentionResidualConfig {
        enabled: true,
        num_heads: 2,
        history_window: Some(1),
        ..AttentionResidualConfig::default()
    };
    let connector = AttentionResidual::<TestBackend>::new(&config, 4, &device);
    let history = vec![
        Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![1.0, 0.0, 0.0, 1.0], [1, 1, 1, 4]),
            &device,
        ),
        Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![0.5, 0.5, 0.5, 0.5], [1, 1, 1, 4]),
            &device,
        ),
    ];
    let current = history.last().expect("current").clone();
    let branch_input = connector.branch_input(current.clone(), &history);
    branch_input
        .into_data()
        .assert_eq(&current.into_data(), false);
}

#[test]
fn attention_residual_zero_init_gate_is_exact_identity_with_multi_history() {
    let device = Default::default();
    let config = AttentionResidualConfig {
        enabled: true,
        num_heads: 2,
        history_window: None,
        ..AttentionResidualConfig::default()
    };
    let connector = AttentionResidual::<TestBackend>::new(&config, 4, &device);
    let anchor = Tensor::<TestBackend, 4>::from_data(
        TensorData::new(vec![1.0, 0.0, 0.0, 1.0], [1, 1, 1, 4]),
        &device,
    );
    let delta = Tensor::<TestBackend, 4>::from_data(
        TensorData::new(vec![0.25, 0.5, 0.75, 1.0], [1, 1, 1, 4]),
        &device,
    );
    let current = Tensor::<TestBackend, 4>::from_data(
        TensorData::new(vec![1.5, 2.0, 2.5, 3.0], [1, 1, 1, 4]),
        &device,
    );
    let branch_input = connector.branch_input(current.clone(), &[anchor, delta]);
    branch_input
        .into_data()
        .assert_eq(&current.into_data(), false);
}

#[test]
fn attention_residual_nonzero_gate_uses_history_mix() {
    let device = Default::default();
    let config = AttentionResidualConfig {
        enabled: true,
        num_heads: 2,
        history_window: None,
        ..AttentionResidualConfig::default()
    };
    let mut connector = AttentionResidual::<TestBackend>::new(&config, 4, &device);
    connector.debug_set_mix_gate_raw(4.0, &device);
    let anchor = Tensor::<TestBackend, 4>::from_data(
        TensorData::new(vec![1.0, 0.0, 0.0, 1.0], [1, 1, 1, 4]),
        &device,
    );
    let delta = Tensor::<TestBackend, 4>::from_data(
        TensorData::new(vec![0.25, 0.5, 0.75, 1.0], [1, 1, 1, 4]),
        &device,
    );
    let current = Tensor::<TestBackend, 4>::from_data(
        TensorData::new(vec![1.5, 2.0, 2.5, 3.0], [1, 1, 1, 4]),
        &device,
    );
    let branch_input = connector.branch_input(current.clone(), &[anchor, delta]);
    assert_ne!(
        branch_input
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("branch vec"),
        current
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("current vec")
    );
}

#[test]
fn block_attention_residual_single_history_element_is_identity() {
    let device = Default::default();
    let config = BlockAttentionResidualConfig {
        enabled: true,
        num_heads: 2,
        layers_per_block: 2,
        ..BlockAttentionResidualConfig::default()
    };
    let connector = BlockAttentionResidual::<TestBackend>::new(&config, 4, &device);
    let current = Tensor::<TestBackend, 4>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 1, 4]),
        &device,
    );
    let branch_input = connector.branch_input(current.clone(), std::slice::from_ref(&current));
    branch_input
        .into_data()
        .assert_eq(&current.into_data(), false);
}

#[test]
fn block_attention_residual_zero_init_gate_is_exact_identity_with_multi_history() {
    let device = Default::default();
    let config = BlockAttentionResidualConfig {
        enabled: true,
        num_heads: 2,
        layers_per_block: 2,
        ..BlockAttentionResidualConfig::default()
    };
    let connector = BlockAttentionResidual::<TestBackend>::new(&config, 4, &device);
    let history = vec![
        Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![1.0, 0.0, 0.0, 1.0], [1, 1, 1, 4]),
            &device,
        ),
        Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![0.5, 0.5, 0.5, 0.5], [1, 1, 1, 4]),
            &device,
        ),
    ];
    let current = Tensor::<TestBackend, 4>::from_data(
        TensorData::new(vec![1.5, 2.0, 2.5, 3.0], [1, 1, 1, 4]),
        &device,
    );
    let branch_input = connector.branch_input(current.clone(), &history);
    branch_input
        .into_data()
        .assert_eq(&current.into_data(), false);
}

#[test]
fn block_attention_residual_limits_old_history_by_block_window() {
    let device = Default::default();
    let config = BlockAttentionResidualConfig {
        enabled: true,
        num_heads: 2,
        layers_per_block: 2,
        block_history_window: Some(1),
        intra_block_history_window: Some(1),
        ..BlockAttentionResidualConfig::default()
    };
    let connector = BlockAttentionResidual::<TestBackend>::new(&config, 4, &device);
    let history = (0..5)
        .map(|value| {
            Tensor::<TestBackend, 4>::from_data(
                TensorData::new(vec![value as f32; 4], [1, 1, 1, 4]),
                &device,
            )
        })
        .collect::<Vec<_>>();
    let current = history.last().expect("current").clone();

    assert_eq!(connector.debug_candidate_count(current, &history), 2);
}

#[test]
fn block_attention_residual_layers_per_block_one_matches_full_attention_residual() {
    let device = Default::default();
    let full = AttentionResidual::<TestBackend>::new(
        &AttentionResidualConfig {
            enabled: true,
            num_heads: 2,
            history_window: None,
            dropout: 0.0,
            recency_bias: 1.5,
            ..AttentionResidualConfig::default()
        },
        4,
        &device,
    );
    let block = BlockAttentionResidual::<TestBackend>::new(
        &BlockAttentionResidualConfig {
            enabled: true,
            num_heads: 2,
            layers_per_block: 1,
            block_history_window: None,
            intra_block_history_window: Some(1),
            summary_mode: BlockAttentionResidualSummaryMode::LearnedProjection,
            dropout: 0.0,
            recency_bias: 1.5,
            ..BlockAttentionResidualConfig::default()
        },
        4,
        &device,
    );
    let history = (0..4)
        .map(|value| {
            Tensor::<TestBackend, 4>::from_data(
                TensorData::new(
                    vec![
                        value as f32,
                        value as f32 + 0.25,
                        value as f32 + 0.5,
                        value as f32 + 0.75,
                    ],
                    [1, 1, 1, 4],
                ),
                &device,
            )
        })
        .collect::<Vec<_>>();
    let current = history.last().expect("current").clone();

    let full_branch = full.branch_input(current.clone(), &history);
    let block_branch = block.branch_input(current, &history);
    block_branch
        .into_data()
        .assert_eq(&full_branch.into_data(), true);
}

#[test]
fn block_attention_residual_backward_gradients_are_finite() {
    let device = Default::default();
    let connector = BlockAttentionResidual::<AutodiffBackend>::new(
        &BlockAttentionResidualConfig {
            enabled: true,
            num_heads: 2,
            layers_per_block: 2,
            block_history_window: Some(2),
            intra_block_history_window: Some(1),
            dropout: 0.0,
            ..BlockAttentionResidualConfig::default()
        },
        4,
        &device,
    );
    let history_leaf = Tensor::<AutodiffBackend, 4>::from_data(
        TensorData::new(vec![0.1, 0.2, 0.3, 0.4], [1, 1, 1, 4]),
        &device,
    )
    .require_grad();
    let mut history = vec![history_leaf.clone()];
    history.extend((1..3).map(|value| {
        Tensor::<AutodiffBackend, 4>::from_data(
            TensorData::new(
                vec![
                    value as f32 + 0.1,
                    value as f32 + 0.2,
                    value as f32 + 0.3,
                    value as f32 + 0.4,
                ],
                [1, 1, 1, 4],
            ),
            &device,
        )
    }));
    let current = Tensor::<AutodiffBackend, 4>::from_data(
        TensorData::new(vec![1.0, 1.5, 2.0, 2.5], [1, 1, 1, 4]),
        &device,
    );
    let branch = connector.branch_input(current.clone(), &history);
    let grads = branch.mean().backward();
    let history_grad = history_leaf.grad(&grads).expect("history grad");
    assert_tensor_finite(history_grad);
}
