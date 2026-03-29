use super::*;
use crate::LatentFanoutScheduleConfig;
use crate::experimental::bitnet_reference::{PackedWeightEncoding, unpack_weight_artifact_to_f32};
use crate::model::low_bit::LowBitSavedActivationConfig;
use crate::model::low_bit_runtime::{unpack_rho_block_state, unpack_rho_int8_block_state_device};
use crate::model::sequence::mamba::MambaSequenceConfig;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Int, TensorData};
use burn_ndarray::NdArray;
use std::sync::{Mutex, OnceLock};

type RecurrenceBackend = NdArray<f32>;

fn kernel_linear_attention() -> SequenceKernelConfig {
    SequenceKernelConfig::reference(SequenceMemorySystem::LinearAttention)
}

fn kernel_linear_dense_score() -> SequenceKernelConfig {
    SequenceKernelConfig::dense_score_short_context()
}

fn kernel_rwkv8() -> SequenceKernelConfig {
    SequenceKernelConfig::reference(SequenceMemorySystem::Rwkv8StateSpace)
}

fn kernel_mamba1() -> SequenceKernelConfig {
    SequenceKernelConfig::reference(SequenceMemorySystem::Mamba1SelectiveScan)
}

fn kernel_mamba2() -> SequenceKernelConfig {
    SequenceKernelConfig::reference(SequenceMemorySystem::Mamba2StateSpaceDuality)
}

fn recurrence_test_model(config: BDHConfig) -> BDH<RecurrenceBackend> {
    static RECURRENCE_MODEL_INIT_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _guard = RECURRENCE_MODEL_INIT_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("recurrence model init lock");
    let device = <RecurrenceBackend as BackendTrait>::Device::default();
    <RecurrenceBackend as BackendTrait>::seed(&device, 2026);
    BDH::<RecurrenceBackend>::new(config, &device)
}

fn recurrence_test_model_with_shape(
    kernel: SequenceKernelConfig,
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    latent_total: usize,
    vocab_size: usize,
) -> BDH<RecurrenceBackend> {
    assert_eq!(
        latent_total % n_embd,
        0,
        "latent_total must be divisible by n_embd in recurrence tests"
    );
    let mut config = BDHConfig {
        n_layer,
        n_embd,
        n_head,
        mlp_internal_dim_multiplier: latent_total / n_embd,
        vocab_size,
        dropout: 0.0,
        sequence_kernel: kernel,
        fused_kernels: FusedKernelConfig {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    };
    if kernel.memory_system == SequenceMemorySystem::Mamba2StateSpaceDuality {
        config.mamba.headdim = n_embd.max(1);
    }
    recurrence_test_model(config)
}

fn low_bit_export_test_config() -> BDHConfig {
    BDHConfig {
        n_layer: 2,
        n_embd: 8,
        n_head: 2,
        mlp_internal_dim_multiplier: 4,
        vocab_size: 32,
        dropout: 0.0,
        fused_kernels: FusedKernelConfig {
            enabled: false,
            ..Default::default()
        },
        quant: crate::LowBitQuantizationConfig {
            enable: true,
            protocol: crate::BitNetLowBitProtocol::BitnetB158,
            weight_format: crate::LowBitWeightFormat::Ternary158,
            act_format: crate::LowBitActivationFormat::Int8,
            target_modules: vec![
                crate::LowBitTargetModule::Encoder,
                crate::LowBitTargetModule::DecoderY,
            ],
            decoder_x_mode: crate::LowBitWeightFormat::Int8,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn decoder_y_quality_recipe_test_config(sequence_kernel: SequenceKernelConfig) -> BDHConfig {
    let mut config = BDHConfig {
        n_layer: 2,
        n_embd: 16,
        n_head: 4,
        mlp_internal_dim_multiplier: 8,
        vocab_size: 32,
        dropout: 0.0,
        sequence_kernel,
        fused_kernels: FusedKernelConfig {
            enabled: false,
            ..Default::default()
        },
        residual_connector: ResidualConnectorKind::AttentionResidual,
        attention_residual: crate::AttentionResidualConfig {
            enabled: true,
            num_heads: 4,
            history_window: Some(4),
            dropout: 0.0,
            recency_bias: 2.0,
            ..Default::default()
        },
        quant: crate::LowBitQuantizationConfig {
            enable: true,
            protocol: crate::BitNetLowBitProtocol::BitnetB158,
            weight_format: crate::LowBitWeightFormat::Ternary158,
            act_format: crate::LowBitActivationFormat::Int8,
            target_modules: vec![crate::LowBitTargetModule::DecoderY],
            decoder_x_mode: crate::LowBitWeightFormat::Fp16,
            training_mode: crate::LowBitTrainingMode::QatSte,
            inference_mode: crate::LowBitInferenceMode::OfflinePack,
            strict_bitnet_reference: false,
            saved_activations: LowBitSavedActivationConfig {
                mode: crate::LowBitSavedActivationMode::QuantizedCacheRecomputeExp,
                format: crate::LowBitActivationFormat::Int8,
            },
            ..Default::default()
        },
        ..Default::default()
    };
    if sequence_kernel.memory_system == SequenceMemorySystem::Mamba1SelectiveScan {
        config.mamba = MambaSequenceConfig {
            d_state: 16,
            d_conv: 2,
            expand: 2,
            ..Default::default()
        };
    }
    config
}

fn allmat_quality_recipe_test_config(
    sequence_kernel: SequenceKernelConfig,
    decoder_x_mode: crate::LowBitWeightFormat,
) -> BDHConfig {
    let mut config = decoder_y_quality_recipe_test_config(sequence_kernel);
    config.quant.target_modules = vec![
        crate::LowBitTargetModule::Encoder,
        crate::LowBitTargetModule::DecoderX,
        crate::LowBitTargetModule::DecoderY,
    ];
    config.quant.decoder_x_mode = decoder_x_mode;
    config
}

fn recurrence_test_tokens_with_shape(
    device: &<RecurrenceBackend as BackendTrait>::Device,
    values: Vec<i64>,
    shape: [usize; 2],
) -> Tensor<RecurrenceBackend, 2, Int> {
    Tensor::<RecurrenceBackend, 2, Int>::from_data(TensorData::new(values, shape), device)
}

fn tensor_max_abs_diff<const D: usize>(
    lhs: Tensor<RecurrenceBackend, D>,
    rhs: Tensor<RecurrenceBackend, D>,
) -> f32 {
    let lhs_vec = lhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("lhs vec");
    let rhs_vec = rhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rhs vec");
    lhs_vec
        .iter()
        .zip(rhs_vec.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max)
}

fn tensor_mean_abs_diff<const D: usize>(
    lhs: Tensor<RecurrenceBackend, D>,
    rhs: Tensor<RecurrenceBackend, D>,
) -> f32 {
    let lhs_vec = lhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("lhs vec");
    let rhs_vec = rhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rhs vec");
    lhs_vec
        .iter()
        .zip(rhs_vec.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .sum::<f32>()
        / lhs_vec.len().max(1) as f32
}

fn tensor_values_f32<const D: usize>(tensor: Tensor<RecurrenceBackend, D>) -> Vec<f32> {
    tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("tensor vec")
}

fn mean_abs(values: &[f32]) -> f32 {
    values.iter().map(|value| value.abs()).sum::<f32>() / values.len().max(1) as f32
}

fn max_abs(values: &[f32]) -> f32 {
    values
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max)
}

#[derive(Clone, Debug)]
struct DecoderYLayerScaleStats {
    layer_index: usize,
    activation_mean_abs: f32,
    activation_max_abs: f32,
    activation_int8_scale: f32,
    activation_int8_saturation_fraction: f32,
    weight_mean_abs: f32,
    weight_max_abs: f32,
    weight_ternary_scale: f32,
    weight_active_fraction: f32,
}

#[derive(Clone, Debug)]
struct AllMatLayerSignalStats {
    layer_index: usize,
    x_neuron_mean_abs: f32,
    x_neuron_max_abs: f32,
    x_neuron_int8_scale: f32,
    x_neuron_int8_saturation_fraction: f32,
    attention_mean_abs: f32,
    attention_max_abs: f32,
    y_gate_mean_abs: f32,
    y_gate_max_abs: f32,
    y_gate_int8_scale: f32,
    y_gate_int8_saturation_fraction: f32,
    y_neuron_mean_abs: f32,
    y_neuron_max_abs: f32,
    y_neuron_int8_scale: f32,
    y_neuron_int8_saturation_fraction: f32,
}

fn collect_decoder_y_layer_scale_stats(
    model: &BDH<RecurrenceBackend>,
    tokens: Tensor<RecurrenceBackend, 2, Int>,
) -> Vec<DecoderYLayerScaleStats> {
    assert!(
        !model.y_neuron_recurrence.enabled,
        "decoder_y scale-stat helper expects y_neuron_recurrence to be disabled"
    );
    let mut state = model.init_state();
    let start_pos = state.position;
    let embedded = model.embed.forward(tokens);
    let [batch, time, embd] = embedded.shape().dims::<3>();
    let mut current = model.norm.forward(embedded.reshape([batch, 1, time, embd]));
    let fused = model.kernel.enabled;
    let mut residual_history = model.initialize_language_residual_history(&current);
    let mut stats = Vec::with_capacity(model.n_layer);

    for (layer_idx, layer_state) in state.layers.iter_mut().enumerate() {
        let connector = model.residual_connector_for_layer(layer_idx);
        let current_before = residual_history.capture_previous(&current);
        let bindings = model.split_language_residuals_for_layer(
            current,
            &connector,
            residual_history.as_slice(),
            None,
        );

        layer_state.clocked_slow_hidden = None;
        layer_state.summary_memory_hidden = None;
        layer_state.y_neuron_state = None;

        let [branch_batch, branch_views, branch_time, branch_dim] =
            bindings.branch_input.shape().dims::<4>();
        let flat_batch = branch_batch * branch_views;
        let branch_flat =
            bindings
                .branch_input
                .clone()
                .reshape([flat_batch, 1, branch_time, branch_dim]);
        let (encoder, encoder_v, decoder, latent) = model.layer_lowrank_weights(layer_idx);
        let latent_pattern = &model.kernel.block_sparse.latent;
        let sparse_mask = if fused && latent_pattern.is_sparse() {
            Some(latent_pattern.mask::<RecurrenceBackend>(latent, &branch_flat.device()))
        } else {
            None
        };
        let output = lowrank_residual_step_with_metrics_branch_thresholds(
            branch_flat,
            encoder,
            encoder_v.clone(),
            decoder,
            &model.dropout,
            fused && model.kernel.projection_executor.use_x(),
            fused && model.kernel.projection_executor.use_y(),
            model.x_relu_threshold,
            model.y_relu_threshold,
            true,
            model.low_bit_projection_plan(),
            model.low_bit_quant.0.saved_activations.clone(),
            model.packed_low_bit_projection_artifacts(),
            latent_pattern,
            model.kernel.lowrank_grad_input_executor,
            sparse_mask,
            |query, value| {
                model.recurrent_attention_with_plan(
                    query,
                    value,
                    layer_state,
                    start_pos,
                    RecurrentPositionMode::Sequential,
                    None,
                )
            },
            |values| activation::relu(values),
            |values| model.norm.forward(values),
        );

        let attn_values = tensor_values_f32(
            output
                .attention_readout
                .clone()
                .expect("attention_readout for decoder_y scale stats"),
        );
        let activation_mean_abs = mean_abs(&attn_values);
        let activation_max_abs = max_abs(&attn_values);
        let activation_int8_scale = (activation_mean_abs * 2.0 / 127.0).max(1.0e-8);
        let activation_int8_saturation_fraction = attn_values
            .iter()
            .filter(|value| {
                (*value / activation_int8_scale)
                    .round()
                    .clamp(-127.0, 127.0)
                    .abs()
                    >= 126.5
            })
            .count() as f32
            / attn_values.len().max(1) as f32;

        let weight_values = tensor_values_f32(encoder_v);
        let weight_mean_abs = mean_abs(&weight_values);
        let weight_max_abs = max_abs(&weight_values);
        let weight_ternary_scale = weight_mean_abs.max(1.0e-8);
        let weight_active_fraction = weight_values
            .iter()
            .filter(|value| value.abs() >= weight_ternary_scale)
            .count() as f32
            / weight_values.len().max(1) as f32;

        stats.push(DecoderYLayerScaleStats {
            layer_index: layer_idx,
            activation_mean_abs,
            activation_max_abs,
            activation_int8_scale,
            activation_int8_saturation_fraction,
            weight_mean_abs,
            weight_max_abs,
            weight_ternary_scale,
            weight_active_fraction,
        });

        let branch_out = output
            .next
            .reshape([branch_batch, branch_views, branch_time, branch_dim]);
        let next = model.merge_language_residuals_for_layer(branch_out, bindings, &connector, None);
        current = if model.residual_connector_needs_post_merge_norm(&connector) {
            model.norm.forward(next)
        } else {
            next
        };
        model.update_language_residual_history(&mut residual_history, current_before, &current);
    }

    assert_eq!(stats.len(), model.n_layer);
    stats
}

fn collect_allmat_layer_signal_stats(
    model: &BDH<RecurrenceBackend>,
    tokens: Tensor<RecurrenceBackend, 2, Int>,
) -> Vec<AllMatLayerSignalStats> {
    assert!(
        !model.y_neuron_recurrence.enabled,
        "all-matrix signal-stat helper expects y_neuron_recurrence to be disabled"
    );
    let mut state = model.init_state();
    let start_pos = state.position;
    let embedded = model.embed.forward(tokens);
    let [batch, time, embd] = embedded.shape().dims::<3>();
    let mut current = model.norm.forward(embedded.reshape([batch, 1, time, embd]));
    let fused = model.kernel.enabled;
    let mut residual_history = model.initialize_language_residual_history(&current);
    let mut stats = Vec::with_capacity(model.n_layer);

    for (layer_idx, layer_state) in state.layers.iter_mut().enumerate() {
        let connector = model.residual_connector_for_layer(layer_idx);
        let current_before = residual_history.capture_previous(&current);
        let bindings = model.split_language_residuals_for_layer(
            current,
            &connector,
            residual_history.as_slice(),
            None,
        );

        layer_state.clocked_slow_hidden = None;
        layer_state.summary_memory_hidden = None;
        layer_state.y_neuron_state = None;

        let [branch_batch, branch_views, branch_time, branch_dim] =
            bindings.branch_input.shape().dims::<4>();
        let flat_batch = branch_batch * branch_views;
        let branch_flat =
            bindings
                .branch_input
                .clone()
                .reshape([flat_batch, 1, branch_time, branch_dim]);
        let (encoder, encoder_v, decoder, latent) = model.layer_lowrank_weights(layer_idx);
        let latent_pattern = &model.kernel.block_sparse.latent;
        let sparse_mask = if fused && latent_pattern.is_sparse() {
            Some(latent_pattern.mask::<RecurrenceBackend>(latent, &branch_flat.device()))
        } else {
            None
        };
        let output = lowrank_residual_step_with_metrics_branch_thresholds(
            branch_flat,
            encoder,
            encoder_v,
            decoder,
            &model.dropout,
            fused && model.kernel.projection_executor.use_x(),
            fused && model.kernel.projection_executor.use_y(),
            model.x_relu_threshold,
            model.y_relu_threshold,
            true,
            model.low_bit_projection_plan(),
            model.low_bit_quant.0.saved_activations.clone(),
            model.packed_low_bit_projection_artifacts(),
            latent_pattern,
            model.kernel.lowrank_grad_input_executor,
            sparse_mask,
            |query, value| {
                model.recurrent_attention_with_plan(
                    query,
                    value,
                    layer_state,
                    start_pos,
                    RecurrentPositionMode::Sequential,
                    None,
                )
            },
            |values| activation::relu(values),
            |values| model.norm.forward(values),
        );

        let x_neuron = tensor_values_f32(output.x_neuron.clone());
        let attention = tensor_values_f32(
            output
                .attention_readout
                .clone()
                .expect("attention_readout for allmat signal stats"),
        );
        let y_gate = tensor_values_f32(output.y_gate.clone());
        let y_neuron = tensor_values_f32(output.y_neuron.clone());

        let x_neuron_mean_abs = mean_abs(&x_neuron);
        let x_neuron_max_abs = max_abs(&x_neuron);
        let x_neuron_int8_scale = (x_neuron_mean_abs * 2.0 / 127.0).max(1.0e-8);
        let x_neuron_int8_saturation_fraction = x_neuron
            .iter()
            .filter(|value| {
                (*value / x_neuron_int8_scale)
                    .round()
                    .clamp(-127.0, 127.0)
                    .abs()
                    >= 126.5
            })
            .count() as f32
            / x_neuron.len().max(1) as f32;

        let attention_mean_abs = mean_abs(&attention);
        let attention_max_abs = max_abs(&attention);

        let y_gate_mean_abs = mean_abs(&y_gate);
        let y_gate_max_abs = max_abs(&y_gate);
        let y_gate_int8_scale = (y_gate_mean_abs * 2.0 / 127.0).max(1.0e-8);
        let y_gate_int8_saturation_fraction = y_gate
            .iter()
            .filter(|value| {
                (*value / y_gate_int8_scale)
                    .round()
                    .clamp(-127.0, 127.0)
                    .abs()
                    >= 126.5
            })
            .count() as f32
            / y_gate.len().max(1) as f32;

        let y_neuron_mean_abs = mean_abs(&y_neuron);
        let y_neuron_max_abs = max_abs(&y_neuron);
        let y_neuron_int8_scale = (y_neuron_mean_abs * 2.0 / 127.0).max(1.0e-8);
        let y_neuron_int8_saturation_fraction = y_neuron
            .iter()
            .filter(|value| {
                (*value / y_neuron_int8_scale)
                    .round()
                    .clamp(-127.0, 127.0)
                    .abs()
                    >= 126.5
            })
            .count() as f32
            / y_neuron.len().max(1) as f32;

        stats.push(AllMatLayerSignalStats {
            layer_index: layer_idx,
            x_neuron_mean_abs,
            x_neuron_max_abs,
            x_neuron_int8_scale,
            x_neuron_int8_saturation_fraction,
            attention_mean_abs,
            attention_max_abs,
            y_gate_mean_abs,
            y_gate_max_abs,
            y_gate_int8_scale,
            y_gate_int8_saturation_fraction,
            y_neuron_mean_abs,
            y_neuron_max_abs,
            y_neuron_int8_scale,
            y_neuron_int8_saturation_fraction,
        });

        let branch_out = output
            .next
            .reshape([branch_batch, branch_views, branch_time, branch_dim]);
        let next = model.merge_language_residuals_for_layer(branch_out, bindings, &connector, None);
        current = if model.residual_connector_needs_post_merge_norm(&connector) {
            model.norm.forward(next)
        } else {
            next
        };
        model.update_language_residual_history(&mut residual_history, current_before, &current);
    }

    assert_eq!(stats.len(), model.n_layer);
    stats
}

fn option_tensor_max_abs_diff<const D: usize>(
    lhs: &Option<Tensor<RecurrenceBackend, D>>,
    rhs: &Option<Tensor<RecurrenceBackend, D>>,
) -> f32 {
    match (lhs, rhs) {
        (None, None) => 0.0,
        (Some(lhs), Some(rhs)) => tensor_max_abs_diff(lhs.clone(), rhs.clone()),
        _ => f32::INFINITY,
    }
}

fn model_state_max_abs_diff(
    lhs: &ModelState<RecurrenceBackend>,
    rhs: &ModelState<RecurrenceBackend>,
) -> f32 {
    if lhs.position != rhs.position || lhs.layers.len() != rhs.layers.len() {
        return f32::INFINITY;
    }

    let mut max_diff = 0.0f32;
    let device = <RecurrenceBackend as BackendTrait>::Device::default();
    for (lhs_layer, rhs_layer) in lhs.layers.iter().zip(rhs.layers.iter()) {
        max_diff = max_diff.max(option_tensor_max_abs_diff(&lhs_layer.rho, &rhs_layer.rho));
        let lhs_packed_rho = lhs_layer
            .packed_rho_int8_device
            .as_ref()
            .map(unpack_rho_int8_block_state_device::<RecurrenceBackend>)
            .or_else(|| {
                lhs_layer
                    .packed_rho
                    .as_ref()
                    .map(|packed| unpack_rho_block_state::<RecurrenceBackend>(packed, &device))
            });
        let rhs_packed_rho = rhs_layer
            .packed_rho_int8_device
            .as_ref()
            .map(unpack_rho_int8_block_state_device::<RecurrenceBackend>)
            .or_else(|| {
                rhs_layer
                    .packed_rho
                    .as_ref()
                    .map(|packed| unpack_rho_block_state::<RecurrenceBackend>(packed, &device))
            });
        max_diff = max_diff.max(option_tensor_max_abs_diff(&lhs_packed_rho, &rhs_packed_rho));
        max_diff = max_diff.max(option_tensor_max_abs_diff(
            &lhs_layer.rho_norm,
            &rhs_layer.rho_norm,
        ));
        max_diff = max_diff.max(option_tensor_max_abs_diff(
            &lhs_layer.sequence_aux,
            &rhs_layer.sequence_aux,
        ));
        max_diff = max_diff.max(option_tensor_max_abs_diff(
            &lhs_layer.y_neuron_state,
            &rhs_layer.y_neuron_state,
        ));
        max_diff = max_diff.max(option_tensor_max_abs_diff(
            &lhs_layer.clocked_slow_hidden,
            &rhs_layer.clocked_slow_hidden,
        ));
        max_diff = max_diff.max(option_tensor_max_abs_diff(
            &lhs_layer.summary_memory_hidden,
            &rhs_layer.summary_memory_hidden,
        ));
    }
    max_diff
}

fn assert_full_forward_matches_token_step_recurrence(kernel: SequenceKernelConfig) {
    let tokens = vec![1i64, 2, 3, 4, 5, 6];
    assert_full_forward_matches_token_step_recurrence_with_shape(
        kernel,
        2,
        8,
        2,
        32,
        32,
        tokens,
        [1, 6],
    );
}

fn assert_full_forward_matches_token_step_recurrence_with_shape(
    kernel: SequenceKernelConfig,
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    latent_total: usize,
    vocab_size: usize,
    token_values: Vec<i64>,
    token_shape: [usize; 2],
) {
    let device = <RecurrenceBackend as BackendTrait>::Device::default();
    let model =
        recurrence_test_model_with_shape(kernel, n_layer, n_embd, n_head, latent_total, vocab_size);
    let tokens = recurrence_test_tokens_with_shape(&device, token_values, token_shape);

    let logits_full = model.forward(tokens.clone());
    let mut recurrent_state = model.init_state();
    let mut logits_steps = Vec::new();
    for step in 0..tokens.shape().dims::<2>()[1] {
        let step_tokens = tokens.clone().slice_dim(1, step..step + 1);
        logits_steps.push(model.forward_with_state(step_tokens, &mut recurrent_state));
    }
    let logits_stepwise = Tensor::cat(logits_steps, 1);
    let max_diff = tensor_max_abs_diff(logits_full, logits_stepwise);

    assert!(
        max_diff <= 1.0e-4,
        "expected full forward and token-step recurrence to match for {kernel:?}, max diff {max_diff}"
    );
}

fn assert_chunked_recurrence_matches_uninterrupted_state(kernel: SequenceKernelConfig) {
    let tokens = vec![1i64, 2, 3, 4, 5, 6];
    assert_chunked_recurrence_matches_uninterrupted_state_with_shape(
        kernel,
        2,
        8,
        2,
        32,
        32,
        tokens,
        [1, 6],
        &[1, 2, 3],
    );
}

fn assert_chunked_recurrence_matches_uninterrupted_state_with_shape(
    kernel: SequenceKernelConfig,
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    latent_total: usize,
    vocab_size: usize,
    token_values: Vec<i64>,
    token_shape: [usize; 2],
    chunk_sizes: &[usize],
) {
    let device = <RecurrenceBackend as BackendTrait>::Device::default();
    let model =
        recurrence_test_model_with_shape(kernel, n_layer, n_embd, n_head, latent_total, vocab_size);
    let tokens = recurrence_test_tokens_with_shape(&device, token_values, token_shape);
    let logits_full = model.forward(tokens.clone());

    let mut uninterrupted_state = model.init_state();
    let logits_uninterrupted = model.forward_with_state(tokens.clone(), &mut uninterrupted_state);
    let uninterrupted_diff = tensor_max_abs_diff(logits_full.clone(), logits_uninterrupted);
    assert!(
        uninterrupted_diff <= 1.0e-4,
        "expected uninterrupted stateful forward to match full forward for {kernel:?}, max diff {uninterrupted_diff}"
    );

    for &chunk_tokens in chunk_sizes {
        let mut chunked_state = model.init_state();
        let mut logits_chunks = Vec::new();
        let seq_len = tokens.shape().dims::<2>()[1];
        for chunk_start in (0..seq_len).step_by(chunk_tokens) {
            let chunk_end = (chunk_start + chunk_tokens).min(seq_len);
            let chunk = tokens.clone().slice_dim(1, chunk_start..chunk_end);
            logits_chunks.push(model.forward_with_state(chunk, &mut chunked_state));
        }
        let logits_chunked = Tensor::cat(logits_chunks, 1);
        let logits_diff = tensor_max_abs_diff(logits_full.clone(), logits_chunked);
        let state_diff = model_state_max_abs_diff(&uninterrupted_state, &chunked_state);

        assert!(
            logits_diff <= 1.0e-4,
            "expected chunked recurrence to match full forward for {kernel:?} chunk_tokens={chunk_tokens}, max diff {logits_diff}"
        );
        assert!(
            state_diff <= 1.0e-4,
            "expected chunked recurrence state to match uninterrupted state for {kernel:?} chunk_tokens={chunk_tokens}, max diff {state_diff}"
        );
    }
}

fn assert_recurrent_attention_wrapper_preserves_position_semantics(
    kernel: SequenceKernelConfig,
    rotary_embedding: crate::RotaryEmbedding,
    position_mode: RecurrentPositionMode,
    position: usize,
) {
    let device = <RecurrenceBackend as BackendTrait>::Device::default();
    let model = recurrence_test_model(BDHConfig {
        n_layer: 1,
        n_embd: 4,
        n_head: 2,
        mlp_internal_dim_multiplier: 1,
        vocab_size: 16,
        dropout: 0.0,
        sequence_kernel: kernel,
        fused_kernels: FusedKernelConfig {
            enabled: true,
            wgpu_recurrent_kernel: true,
            wgpu_rollout_fused: true,
            rotary_embedding,
            ..Default::default()
        },
        ..Default::default()
    });
    let mut state = model.init_state();
    let query = Tensor::<RecurrenceBackend, 4>::from_data(
        TensorData::new(
            vec![
                1.0, 2.0, 2.0, 1.0, 0.5, 1.5, 3.0, 0.5, 1.25, 0.75, 0.25, 1.0,
            ],
            [1, 2, 3, 2],
        ),
        &device,
    );
    let value = Tensor::<RecurrenceBackend, 4>::from_data(
        TensorData::new(vec![1.0, 0.5, 0.25, 1.5, 2.0, 1.0], [1, 1, 3, 2]),
        &device,
    );

    let actual = model.recurrent_attention_with_plan(
        query.clone(),
        value.clone(),
        &mut state.layers[0],
        position,
        position_mode,
        None,
    );
    let rotated = match position_mode {
        RecurrentPositionMode::Sequential => model.attention.rotate_positions(query, position),
        RecurrentPositionMode::Fixed => model.attention.rotate_positions_fixed(query, position),
    };
    let decay = model.attention.alibi_decay();
    let (expected, expected_rho) = match kernel.executor {
        SequenceTrainingExecutor::Reference => {
            model.recurrent_attention_reference(rotated, value, None, decay)
        }
        SequenceTrainingExecutor::DenseScoreShortContext => {
            model.recurrent_attention_dense_score_reference(rotated, value, None, decay)
        }
    };

    let context_diff = tensor_max_abs_diff(actual, expected);
    let actual_rho = state.layers[0]
        .rho
        .clone()
        .expect("rho state written by recurrent wrapper");
    let rho_diff = tensor_max_abs_diff(actual_rho, expected_rho);

    assert!(
        context_diff <= 1.0e-4,
        "expected wrapper context to preserve {:?} positional semantics for {kernel:?}, max diff {context_diff}",
        position_mode
    );
    assert!(
        rho_diff <= 1.0e-4,
        "expected wrapper rho to preserve {:?} positional semantics for {kernel:?}, max diff {rho_diff}",
        position_mode
    );
}

#[test]
fn export_bitnet_static_artifacts_partial_safe_maps_repo_weights_to_roadmap_names() {
    let model = recurrence_test_model(low_bit_export_test_config());
    let artifacts = model.export_bitnet_static_artifacts();

    assert!(artifacts.decoder_x.is_none());
    assert_eq!(
        artifacts
            .decoder_y
            .as_ref()
            .expect("decoder_y artifact")
            .encoding,
        PackedWeightEncoding::Ternary2
    );
    assert_eq!(
        artifacts
            .encoder
            .as_ref()
            .expect("encoder artifact")
            .encoding,
        PackedWeightEncoding::Ternary2
    );
}

#[test]
fn export_bitnet_static_artifacts_round_trip_preserves_tensor_shape_and_finite_values() {
    let model = recurrence_test_model(BDHConfig {
        quant: crate::LowBitQuantizationConfig {
            enable: true,
            protocol: crate::BitNetLowBitProtocol::BitnetB158,
            weight_format: crate::LowBitWeightFormat::Ternary158,
            act_format: crate::LowBitActivationFormat::Int8,
            target_modules: vec![
                crate::LowBitTargetModule::Encoder,
                crate::LowBitTargetModule::DecoderX,
                crate::LowBitTargetModule::DecoderY,
            ],
            decoder_x_mode: crate::LowBitWeightFormat::Sign1,
            ..Default::default()
        },
        ..low_bit_export_test_config()
    });

    let artifacts = model.export_bitnet_static_artifacts();
    let decoder_x = artifacts.decoder_x.expect("decoder_x artifact");
    let decoder_y = artifacts.decoder_y.expect("decoder_y artifact");
    let encoder = artifacts.encoder.expect("encoder artifact");

    assert_eq!(decoder_x.logical_shape, vec![2, 8, 16]);
    assert_eq!(decoder_y.logical_shape, vec![2, 8, 16]);
    assert_eq!(encoder.logical_shape, vec![32, 8]);

    for values in [
        unpack_weight_artifact_to_f32(&decoder_x),
        unpack_weight_artifact_to_f32(&decoder_y),
        unpack_weight_artifact_to_f32(&encoder),
    ] {
        assert!(values.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn packed_bitnet_static_artifacts_match_fake_quant_full_model_logits() {
    let config = BDHConfig {
        quant: crate::LowBitQuantizationConfig {
            enable: true,
            protocol: crate::BitNetLowBitProtocol::BitnetB158,
            weight_format: crate::LowBitWeightFormat::Ternary158,
            act_format: crate::LowBitActivationFormat::Int8,
            target_modules: vec![
                crate::LowBitTargetModule::Encoder,
                crate::LowBitTargetModule::DecoderX,
                crate::LowBitTargetModule::DecoderY,
            ],
            decoder_x_mode: crate::LowBitWeightFormat::Sign1,
            ..Default::default()
        },
        ..low_bit_export_test_config()
    };
    let device = <RecurrenceBackend as BackendTrait>::Device::default();
    let model = recurrence_test_model(config.clone());
    let artifacts = model.export_bitnet_static_artifacts();
    let mut packed_model = recurrence_test_model(config);

    let expected_decoder_x =
        fake_quantize_weight_ste(model.encoder.val(), crate::LowBitWeightFormat::Sign1)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("decoder_x quantized vec");
    let expected_decoder_y =
        fake_quantize_weight_ste(model.encoder_v.val(), crate::LowBitWeightFormat::Ternary158)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("decoder_y quantized vec");
    let expected_encoder =
        fake_quantize_weight_ste(model.decoder.val(), crate::LowBitWeightFormat::Ternary158)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("encoder quantized vec");
    let artifact_decoder_x =
        unpack_weight_artifact_to_f32(artifacts.decoder_x.as_ref().expect("decoder_x artifact"));
    let artifact_decoder_y =
        unpack_weight_artifact_to_f32(artifacts.decoder_y.as_ref().expect("decoder_y artifact"));
    let artifact_encoder =
        unpack_weight_artifact_to_f32(artifacts.encoder.as_ref().expect("encoder artifact"));
    let decoder_x_weight_diff = expected_decoder_x
        .iter()
        .zip(artifact_decoder_x.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max);
    let decoder_y_weight_diff = expected_decoder_y
        .iter()
        .zip(artifact_decoder_y.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max);
    let encoder_weight_diff = expected_encoder
        .iter()
        .zip(artifact_encoder.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max);
    assert!(
        decoder_x_weight_diff <= 1.0e-6,
        "decoder_x export drifted from fake-quant path by {decoder_x_weight_diff}"
    );
    assert!(
        decoder_y_weight_diff <= 1.0e-6,
        "decoder_y export drifted from fake-quant path by {decoder_y_weight_diff}"
    );
    assert!(
        encoder_weight_diff <= 1.0e-6,
        "encoder export drifted from fake-quant path by {encoder_weight_diff}"
    );

    packed_model
        .apply_bitnet_static_artifacts(&artifacts, &device)
        .expect("apply bitnet artifacts");

    let tokens = recurrence_test_tokens_with_shape(&device, vec![1, 2, 3, 4, 5, 6], [1, 6]);
    let logits_fake_quant = model.forward(tokens.clone());
    let logits_packed = packed_model.forward(tokens);
    let max_diff = tensor_max_abs_diff(logits_fake_quant, logits_packed);
    assert!(
        max_diff <= 5.0e-1,
        "expected packed static artifact reload path to stay numerically close to fake-quant logits, max diff {max_diff}"
    );
}

#[test]
fn train_kernel_exp_forward_selects_native_runtime_and_emits_finite_logits() {
    let device = <RecurrenceBackend as BackendTrait>::Device::default();
    let qat_model = recurrence_test_model(BDHConfig {
        quant: crate::LowBitQuantizationConfig {
            training_mode: crate::LowBitTrainingMode::QatSte,
            inference_mode: crate::LowBitInferenceMode::RuntimeFakeQuant,
            ..low_bit_export_test_config().quant
        },
        ..low_bit_export_test_config()
    });
    let native_model = recurrence_test_model(BDHConfig {
        quant: crate::LowBitQuantizationConfig {
            training_mode: crate::LowBitTrainingMode::TrainKernelExp,
            inference_mode: crate::LowBitInferenceMode::RuntimeFakeQuant,
            ..low_bit_export_test_config().quant
        },
        ..low_bit_export_test_config()
    });
    let tokens = recurrence_test_tokens_with_shape(&device, vec![1, 2, 3, 4], [1, 4]);
    let qat_logits = qat_model.forward(tokens.clone());
    let native_logits = native_model.forward(tokens);
    let plan = resolve_low_bit_kernel_plan::<RecurrenceBackend>(
        &native_model.low_bit_quant.0,
        native_model.available_packed_low_bit_projection_artifacts(),
    );
    let max_diff = tensor_max_abs_diff(qat_logits, native_logits.clone());
    let logits = native_logits
        .into_data()
        .to_vec::<f32>()
        .expect("native logits");
    assert_eq!(
        plan.runtime,
        LowBitKernelRuntimeKind::PackedNativeTrainingForward
    );
    assert!(
        logits.iter().all(|value| value.is_finite()),
        "expected train-kernel-exp forward logits to remain finite"
    );
    assert!(
        max_diff <= 0.5,
        "expected train-kernel-exp forward to remain bounded vs qat reference, max diff {max_diff}"
    );
}

#[test]
fn train_kernel_exp_decoder_y_quality_recipe_remains_close_to_qat_reference() {
    let device = <RecurrenceBackend as BackendTrait>::Device::default();
    let base = decoder_y_quality_recipe_test_config(kernel_linear_dense_score());
    let qat_model = recurrence_test_model(base.clone());
    let native_model = recurrence_test_model(BDHConfig {
        quant: crate::LowBitQuantizationConfig {
            training_mode: crate::LowBitTrainingMode::TrainKernelExp,
            ..base.quant.clone()
        },
        ..base
    });
    let tokens = recurrence_test_tokens_with_shape(&device, vec![1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    let qat_logits = qat_model.forward(tokens.clone());
    let native_logits = native_model.forward(tokens);
    let max_diff = tensor_max_abs_diff(qat_logits.clone(), native_logits.clone());
    let mean_diff = tensor_mean_abs_diff(qat_logits, native_logits);
    eprintln!(
        "decoder_y_quality_recipe_parity max_abs_diff={max_diff:.6} mean_abs_diff={mean_diff:.6}"
    );
    assert!(
        mean_diff <= 0.10,
        "expected decoder_y native forward mean diff to stay bounded vs qat reference, mean diff {mean_diff}"
    );
    assert!(
        max_diff <= 0.35,
        "expected decoder_y native forward max diff to stay bounded vs qat reference, max diff {max_diff}"
    );
}

#[test]
fn train_kernel_exp_decoder_y_mamba_quality_recipe_reports_qat_parity() {
    let device = <RecurrenceBackend as BackendTrait>::Device::default();
    let base = decoder_y_quality_recipe_test_config(kernel_mamba1());
    let qat_model = recurrence_test_model(base.clone());
    let native_model = recurrence_test_model(BDHConfig {
        quant: crate::LowBitQuantizationConfig {
            training_mode: crate::LowBitTrainingMode::TrainKernelExp,
            ..base.quant.clone()
        },
        ..base
    });
    let tokens =
        recurrence_test_tokens_with_shape(&device, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 5]);
    let qat_logits = qat_model.forward(tokens.clone());
    let native_logits = native_model.forward(tokens);
    let max_diff = tensor_max_abs_diff(qat_logits.clone(), native_logits.clone());
    let mean_diff = tensor_mean_abs_diff(qat_logits, native_logits);
    eprintln!(
        "decoder_y_mamba_quality_recipe_parity max_abs_diff={max_diff:.6} mean_abs_diff={mean_diff:.6}"
    );
    assert!(max_diff.is_finite() && mean_diff.is_finite());
    assert!(
        mean_diff <= 0.10,
        "expected Mamba decoder_y native path to stay reasonably close to QAT, mean diff {mean_diff}"
    );
    assert!(
        max_diff <= 0.45,
        "expected Mamba decoder_y native path to avoid large outliers, max diff {max_diff}"
    );
}

#[test]
fn decoder_y_mamba_ablation_reports_fp32_qat_native_drift() {
    let device = <RecurrenceBackend as BackendTrait>::Device::default();
    let fp32_base = decoder_y_quality_recipe_test_config(kernel_mamba1());
    let fp32_model = recurrence_test_model(BDHConfig {
        quant: crate::LowBitQuantizationConfig {
            enable: false,
            target_modules: vec![],
            ..fp32_base.quant.clone()
        },
        ..fp32_base.clone()
    });
    let qat_model = recurrence_test_model(fp32_base.clone());
    let native_model = recurrence_test_model(BDHConfig {
        quant: crate::LowBitQuantizationConfig {
            training_mode: crate::LowBitTrainingMode::TrainKernelExp,
            ..fp32_base.quant.clone()
        },
        ..fp32_base
    });
    let tokens = recurrence_test_tokens_with_shape(
        &device,
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [2, 6],
    );
    let fp32_logits = fp32_model.forward(tokens.clone());
    let qat_logits = qat_model.forward(tokens.clone());
    let native_logits = native_model.forward(tokens);
    let fp32_qat_max = tensor_max_abs_diff(fp32_logits.clone(), qat_logits.clone());
    let fp32_qat_mean = tensor_mean_abs_diff(fp32_logits.clone(), qat_logits.clone());
    let fp32_native_max = tensor_max_abs_diff(fp32_logits.clone(), native_logits.clone());
    let fp32_native_mean = tensor_mean_abs_diff(fp32_logits.clone(), native_logits.clone());
    let qat_native_max = tensor_max_abs_diff(qat_logits.clone(), native_logits.clone());
    let qat_native_mean = tensor_mean_abs_diff(qat_logits, native_logits);
    eprintln!(
        "decoder_y_mamba_ablation fp32_vs_qat max_abs_diff={fp32_qat_max:.6} mean_abs_diff={fp32_qat_mean:.6} \
fp32_vs_native max_abs_diff={fp32_native_max:.6} mean_abs_diff={fp32_native_mean:.6} \
qat_vs_native max_abs_diff={qat_native_max:.6} mean_abs_diff={qat_native_mean:.6}"
    );
    assert!(
        fp32_qat_max.is_finite()
            && fp32_qat_mean.is_finite()
            && fp32_native_max.is_finite()
            && fp32_native_mean.is_finite()
            && qat_native_max.is_finite()
            && qat_native_mean.is_finite()
    );
    assert!(
        fp32_qat_mean <= 0.12,
        "expected fake-quant Mamba drift from FP32 to stay bounded, mean diff {fp32_qat_mean}"
    );
    assert!(
        fp32_native_mean <= 0.12,
        "expected native Mamba drift from FP32 to stay bounded, mean diff {fp32_native_mean}"
    );
    assert!(
        (fp32_native_mean - fp32_qat_mean).abs() <= 0.03,
        "expected native-vs-FP32 drift to track fake-quant-vs-FP32 drift closely; fp32_qat_mean={fp32_qat_mean}, fp32_native_mean={fp32_native_mean}"
    );
}

#[test]
fn decoder_y_scale_stats_compare_linear_and_mamba() {
    let device = <RecurrenceBackend as BackendTrait>::Device::default();
    let tokens = recurrence_test_tokens_with_shape(
        &device,
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [2, 6],
    );
    let linear_model = recurrence_test_model(decoder_y_quality_recipe_test_config(
        kernel_linear_dense_score(),
    ));
    let mamba_model = recurrence_test_model(decoder_y_quality_recipe_test_config(kernel_mamba1()));
    let linear_stats = collect_decoder_y_layer_scale_stats(&linear_model, tokens.clone());
    let mamba_stats = collect_decoder_y_layer_scale_stats(&mamba_model, tokens);

    assert_eq!(linear_stats.len(), mamba_stats.len());
    for (linear, mamba) in linear_stats.iter().zip(mamba_stats.iter()) {
        eprintln!(
            "decoder_y_scale_stats layer={} linear(act_mean_abs={:.6}, act_max_abs={:.6}, act_scale={:.6}, act_sat_frac={:.6}, weight_mean_abs={:.6}, weight_max_abs={:.6}, weight_scale={:.6}, weight_active_frac={:.6}) mamba(act_mean_abs={:.6}, act_max_abs={:.6}, act_scale={:.6}, act_sat_frac={:.6}, weight_mean_abs={:.6}, weight_max_abs={:.6}, weight_scale={:.6}, weight_active_frac={:.6})",
            linear.layer_index,
            linear.activation_mean_abs,
            linear.activation_max_abs,
            linear.activation_int8_scale,
            linear.activation_int8_saturation_fraction,
            linear.weight_mean_abs,
            linear.weight_max_abs,
            linear.weight_ternary_scale,
            linear.weight_active_fraction,
            mamba.activation_mean_abs,
            mamba.activation_max_abs,
            mamba.activation_int8_scale,
            mamba.activation_int8_saturation_fraction,
            mamba.weight_mean_abs,
            mamba.weight_max_abs,
            mamba.weight_ternary_scale,
            mamba.weight_active_fraction,
        );
        assert!(linear.activation_mean_abs.is_finite());
        assert!(mamba.activation_mean_abs.is_finite());
        assert!(linear.weight_mean_abs.is_finite());
        assert!(mamba.weight_mean_abs.is_finite());
    }
}

#[test]
fn allmat_decoder_x_sign1_qat_improves_fp32_drift_vs_ternary() {
    let device = <RecurrenceBackend as BackendTrait>::Device::default();
    let tokens = recurrence_test_tokens_with_shape(
        &device,
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [2, 6],
    );
    let mut fp32_config = decoder_y_quality_recipe_test_config(kernel_linear_dense_score());
    fp32_config.quant.enable = false;

    let mut ternary_config = allmat_quality_recipe_test_config(
        kernel_linear_dense_score(),
        crate::LowBitWeightFormat::Ternary158,
    );
    ternary_config.quant.training_mode = crate::LowBitTrainingMode::QatSte;

    let mut sign1_config = allmat_quality_recipe_test_config(
        kernel_linear_dense_score(),
        crate::LowBitWeightFormat::Sign1,
    );
    sign1_config.quant.training_mode = crate::LowBitTrainingMode::QatSte;

    let fp32_model = recurrence_test_model(fp32_config);
    let ternary_model = recurrence_test_model(ternary_config);
    let sign1_model = recurrence_test_model(sign1_config);

    let fp32_logits = fp32_model.forward(tokens.clone());
    let ternary_logits = ternary_model.forward(tokens.clone());
    let sign1_logits = sign1_model.forward(tokens.clone());
    let ternary_mean = tensor_mean_abs_diff(fp32_logits.clone(), ternary_logits);
    let sign1_mean = tensor_mean_abs_diff(fp32_logits, sign1_logits);

    let ternary_stats = collect_allmat_layer_signal_stats(&ternary_model, tokens.clone());
    let sign1_stats = collect_allmat_layer_signal_stats(&sign1_model, tokens);
    assert_eq!(ternary_stats.len(), sign1_stats.len());
    for (ternary, sign1) in ternary_stats.iter().zip(sign1_stats.iter()) {
        eprintln!(
            "allmat_decoder_x_mode_compare layer={} ternary(x_mean_abs={:.6}, x_max_abs={:.6}, x_scale={:.6}, x_sat={:.6}, attn_mean_abs={:.6}, attn_max_abs={:.6}, y_mean_abs={:.6}, y_max_abs={:.6}, y_scale={:.6}, y_sat={:.6}, y_neuron_mean_abs={:.6}, y_neuron_max_abs={:.6}, y_neuron_scale={:.6}, y_neuron_sat={:.6}) sign1(x_mean_abs={:.6}, x_max_abs={:.6}, x_scale={:.6}, x_sat={:.6}, attn_mean_abs={:.6}, attn_max_abs={:.6}, y_mean_abs={:.6}, y_max_abs={:.6}, y_scale={:.6}, y_sat={:.6}, y_neuron_mean_abs={:.6}, y_neuron_max_abs={:.6}, y_neuron_scale={:.6}, y_neuron_sat={:.6})",
            ternary.layer_index,
            ternary.x_neuron_mean_abs,
            ternary.x_neuron_max_abs,
            ternary.x_neuron_int8_scale,
            ternary.x_neuron_int8_saturation_fraction,
            ternary.attention_mean_abs,
            ternary.attention_max_abs,
            ternary.y_gate_mean_abs,
            ternary.y_gate_max_abs,
            ternary.y_gate_int8_scale,
            ternary.y_gate_int8_saturation_fraction,
            ternary.y_neuron_mean_abs,
            ternary.y_neuron_max_abs,
            ternary.y_neuron_int8_scale,
            ternary.y_neuron_int8_saturation_fraction,
            sign1.x_neuron_mean_abs,
            sign1.x_neuron_max_abs,
            sign1.x_neuron_int8_scale,
            sign1.x_neuron_int8_saturation_fraction,
            sign1.attention_mean_abs,
            sign1.attention_max_abs,
            sign1.y_gate_mean_abs,
            sign1.y_gate_max_abs,
            sign1.y_gate_int8_scale,
            sign1.y_gate_int8_saturation_fraction,
            sign1.y_neuron_mean_abs,
            sign1.y_neuron_max_abs,
            sign1.y_neuron_int8_scale,
            sign1.y_neuron_int8_saturation_fraction,
        );
        assert_eq!(ternary.layer_index, sign1.layer_index);
    }

    eprintln!(
        "allmat_decoder_x_mode_compare fp32_vs_ternary_mean_abs_diff={ternary_mean:.6} fp32_vs_sign1_mean_abs_diff={sign1_mean:.6}"
    );
    assert!(
        sign1_mean < ternary_mean,
        "expected decoder_x Sign1 to reduce all-matrix QAT drift vs ternary, sign1={sign1_mean} ternary={ternary_mean}"
    );
}

fn repeat_value_heads_for_recurrence(
    value: Tensor<RecurrenceBackend, 4>,
    heads: usize,
) -> Tensor<RecurrenceBackend, 4> {
    match value.shape().dims::<4>()[1] {
        1 => value.repeat_dim(1, heads),
        existing if existing == heads => value,
        existing => panic!("value heads {existing} must be 1 or {heads}"),
    }
}

fn exclusive_prefix_sum_time_5d(
    tensor: Tensor<RecurrenceBackend, 5>,
) -> Tensor<RecurrenceBackend, 5> {
    let [batch, heads, time, latent, embd] = tensor.shape().dims::<5>();
    let prefix = tensor.cumsum(2);
    let zero =
        Tensor::<RecurrenceBackend, 5>::zeros([batch, heads, 1, latent, embd], &prefix.device());
    if time == 1 {
        zero
    } else {
        Tensor::cat(vec![zero, prefix.slice_dim(2, 0..time - 1)], 2)
    }
}

fn exclusive_prefix_sum_time_4d(
    tensor: Tensor<RecurrenceBackend, 4>,
) -> Tensor<RecurrenceBackend, 4> {
    let [batch, heads, time, latent] = tensor.shape().dims::<4>();
    let prefix = tensor.cumsum(2);
    let zero = Tensor::<RecurrenceBackend, 4>::zeros([batch, heads, 1, latent], &prefix.device());
    if time == 1 {
        zero
    } else {
        Tensor::cat(vec![zero, prefix.slice_dim(2, 0..time - 1)], 2)
    }
}

fn recurrent_attention_tensorized_no_decay_reference(
    query: Tensor<RecurrenceBackend, 4>,
    value: Tensor<RecurrenceBackend, 4>,
) -> (Tensor<RecurrenceBackend, 4>, Tensor<RecurrenceBackend, 4>) {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let embd = value.shape().dims::<4>()[3];
    let value = repeat_value_heads_for_recurrence(value, heads);

    let delta = query.clone().unsqueeze_dim::<5>(4) * value.clone().unsqueeze_dim::<5>(3);
    let rho_before = exclusive_prefix_sum_time_5d(delta.clone());
    let context = (rho_before.clone() * query.clone().unsqueeze_dim::<5>(4))
        .sum_dim(3)
        .reshape([batch, heads, time, embd]);
    let rho = delta
        .cumsum(2)
        .slice_dim(2, time - 1..time)
        .reshape([batch, heads, latent, embd]);

    (context, rho)
}

fn recurrent_attention_dense_score_reference(
    query: Tensor<RecurrenceBackend, 4>,
    value: Tensor<RecurrenceBackend, 4>,
    decay: Option<Tensor<RecurrenceBackend, 1>>,
) -> (Tensor<RecurrenceBackend, 4>, Tensor<RecurrenceBackend, 4>) {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let embd = value.shape().dims::<4>()[3];
    let device = query.device();
    let value = repeat_value_heads_for_recurrence(value, heads);

    let pos_row = Tensor::<RecurrenceBackend, 1, Int>::arange(0..time as i64, &device)
        .float()
        .reshape([1, 1, time, 1]);
    let pos_col = Tensor::<RecurrenceBackend, 1, Int>::arange(0..time as i64, &device)
        .float()
        .reshape([1, 1, 1, time]);

    let mut scores = query.clone().matmul(query.clone().swap_dims(2, 3)).tril(-1);
    let rho = if let Some(decay) = decay {
        let diff = (pos_row.clone() - pos_col.clone())
            .tril(-1)
            .repeat_dim(1, heads);
        let decay_score = decay
            .clone()
            .reshape([1, heads, 1, 1])
            .repeat_dim(2, time)
            .repeat_dim(3, time);
        scores = scores * decay_score.powf(diff);

        let final_exponents = pos_row
            .clone()
            .mul_scalar(-1.0)
            .add_scalar(time as f32)
            .repeat_dim(1, heads);
        let decay_final = decay
            .reshape([1, heads, 1, 1])
            .repeat_dim(2, time)
            .powf(final_exponents);
        query.mul(decay_final).swap_dims(2, 3).matmul(value.clone())
    } else {
        query.swap_dims(2, 3).matmul(value.clone())
    };

    let context = scores.matmul(value).reshape([batch, heads, time, embd]);
    assert_eq!(rho.shape().dims::<4>(), [batch, heads, latent, embd]);

    (context, rho)
}

fn recurrent_rwkv8_tensorized_reference(
    query: Tensor<RecurrenceBackend, 4>,
    value: Tensor<RecurrenceBackend, 4>,
    decay: Tensor<RecurrenceBackend, 3>,
) -> (
    Tensor<RecurrenceBackend, 4>,
    Tensor<RecurrenceBackend, 4>,
    Tensor<RecurrenceBackend, 3>,
) {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let embd = value.shape().dims::<4>()[3];
    let device = query.device();
    let value = repeat_value_heads_for_recurrence(value, heads);
    let time_idx = Tensor::<RecurrenceBackend, 1, Int>::arange(0..time as i64, &device).float();

    let delta = query.clone().unsqueeze_dim::<5>(4) * value.clone().unsqueeze_dim::<5>(3);

    let decay5 = decay
        .clone()
        .reshape([1, heads, 1, latent, 1])
        .repeat_dim(2, time);
    let state_exp5 = time_idx
        .clone()
        .reshape([1, 1, time, 1, 1])
        .repeat_dim(1, heads)
        .repeat_dim(3, latent);
    let inv_exp5 = time_idx
        .clone()
        .add_scalar(1.0)
        .mul_scalar(-1.0)
        .reshape([1, 1, time, 1, 1])
        .repeat_dim(1, heads)
        .repeat_dim(3, latent);
    let rho_before = exclusive_prefix_sum_time_5d(delta.clone() * decay5.clone().powf(inv_exp5))
        * decay5.clone().powf(state_exp5);

    let decay4 = decay
        .clone()
        .reshape([1, heads, 1, latent])
        .repeat_dim(2, time);
    let state_exp4 = time_idx
        .clone()
        .reshape([1, 1, time, 1])
        .repeat_dim(1, heads)
        .repeat_dim(3, latent);
    let inv_exp4 = time_idx
        .clone()
        .add_scalar(1.0)
        .mul_scalar(-1.0)
        .reshape([1, 1, time, 1])
        .repeat_dim(1, heads)
        .repeat_dim(3, latent);
    let rho_norm_before =
        exclusive_prefix_sum_time_4d(query.clone() * decay4.clone().powf(inv_exp4))
            * decay4.clone().powf(state_exp4);

    let q_weights = query.clone().div(
        query
            .clone()
            .sum_dim(3)
            .add_scalar(1.0e-6)
            .reshape([batch, heads, time, 1]),
    );
    let context = rho_before
        .clone()
        .div(
            rho_norm_before
                .clone()
                .add_scalar(1.0e-6)
                .unsqueeze_dim::<5>(4),
        )
        .mul(q_weights.unsqueeze_dim::<5>(4))
        .sum_dim(3)
        .reshape([batch, heads, time, embd]);

    let last_rho_before = rho_before
        .slice_dim(2, time - 1..time)
        .reshape([batch, heads, latent, embd]);
    let last_rho_norm_before = rho_norm_before
        .slice_dim(2, time - 1..time)
        .reshape([batch, heads, latent]);
    let last_delta = delta
        .slice_dim(2, time - 1..time)
        .reshape([batch, heads, latent, embd]);
    let last_query = query
        .slice_dim(2, time - 1..time)
        .reshape([batch, heads, latent]);

    let rho = last_rho_before
        .mul(decay.clone().reshape([1, heads, latent, 1]))
        .add(last_delta);
    let rho_norm = last_rho_norm_before
        .mul(decay.reshape([1, heads, latent]))
        .add(last_query);

    (context, rho, rho_norm)
}

fn deterministic_y_neuron_recurrence_model_with_layers(
    recurrence: YNeuronRecurrenceConfig,
    n_layer: usize,
) -> BDH<NdArray<f32>> {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let mut model = BDH::<Backend>::new(
        BDHConfig {
            n_layer,
            n_embd: 2,
            n_head: 1,
            mlp_internal_dim_multiplier: 1,
            vocab_size: 8,
            dropout: 0.0,
            y_neuron_recurrence: recurrence,
            ..Default::default()
        },
        &device,
    );

    model.encoder = Param::from_tensor(Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![1.0, 0.0, 0.0, 1.0], [1, 2, 2]),
        &device,
    ));
    model.encoder_v = Param::from_tensor(Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![1.0, 0.0, 0.0, 1.0], [1, 2, 2]),
        &device,
    ));
    model.decoder = Param::from_tensor(Tensor::<Backend, 2>::from_data(
        TensorData::new(vec![0.0, 0.0, 1.0, 0.5], [2, 2]),
        &device,
    ));

    model
}

fn deterministic_y_neuron_recurrence_model(
    recurrence: YNeuronRecurrenceConfig,
) -> BDH<NdArray<f32>> {
    deterministic_y_neuron_recurrence_model_with_layers(recurrence, 1)
}

#[test]
fn recurrent_attention_reference_matches_outer_product_state_space_contract() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 1,
            n_embd: 2,
            n_head: 1,
            mlp_internal_dim_multiplier: 1,
            vocab_size: 8,
            dropout: 0.0,
            ..Default::default()
        },
        &device,
    );

    let query = Tensor::<Backend, 4>::from_data(
        TensorData::new(vec![2.0, 3.0, 5.0, 7.0], [1, 1, 2, 2]),
        &device,
    );
    let value = Tensor::<Backend, 4>::from_data(
        TensorData::new(vec![11.0, 13.0, 17.0, 19.0], [1, 1, 2, 2]),
        &device,
    );

    let (context, rho) = model.recurrent_attention_reference(query, value, None, None);

    let context_vec = context
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("context vec");
    let rho_vec = rho
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rho vec");

    assert_eq!(context_vec, vec![0.0, 0.0, 341.0, 403.0]);
    assert_eq!(rho_vec, vec![107.0, 121.0, 152.0, 172.0]);
}

#[test]
fn recurrent_rwkv8_state_space_reference_matches_decayed_normalized_contract() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 1,
            n_embd: 2,
            n_head: 1,
            mlp_internal_dim_multiplier: 1,
            vocab_size: 8,
            dropout: 0.0,
            sequence_kernel: kernel_rwkv8(),
            ..Default::default()
        },
        &device,
    );

    let query = Tensor::<Backend, 4>::from_data(
        TensorData::new(vec![2.0, 0.0, 1.0, 3.0], [1, 1, 2, 2]),
        &device,
    );
    let value = Tensor::<Backend, 4>::from_data(
        TensorData::new(vec![10.0, 20.0, 30.0, 40.0], [1, 1, 2, 2]),
        &device,
    );
    let decay = Tensor::<Backend, 1>::from_data(TensorData::new(vec![0.5, 0.5], [2]), &device)
        .reshape([1, 1, 2]);

    let (context, rho, rho_norm) =
        model.recurrent_rwkv8_state_space_reference(query, value, None, None, decay);

    let context_vec = context
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("context vec");
    let rho_vec = rho
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rho vec");
    let rho_norm_vec = rho_norm
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rho norm vec");

    for (actual, expected) in context_vec.iter().zip([0.0, 0.0, 2.5, 5.0]) {
        assert!((actual - expected).abs() < 1.0e-4);
    }
    for (actual, expected) in rho_vec.iter().zip([40.0, 60.0, 90.0, 120.0]) {
        assert!((actual - expected).abs() < 1.0e-4);
    }
    for (actual, expected) in rho_norm_vec.iter().zip([2.0, 3.0]) {
        assert!((actual - expected).abs() < 1.0e-4);
    }
}

#[test]
fn rwkv8_forward_with_state_populates_rho_norm() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 2,
            n_embd: 8,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.0,
            sequence_kernel: kernel_rwkv8(),
            ..Default::default()
        },
        &device,
    );
    let tokens =
        Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![1, 2, 3], [1, 3]), &device);
    let mut state = model.init_state();
    let _ = model.forward_with_state(tokens, &mut state);

    assert!(state.layers.iter().all(|layer| layer.rho.is_some()));
    assert!(state.layers.iter().all(|layer| layer.rho_norm.is_some()));
}

#[test]
fn linear_forward_with_rho_int8_chunk_compression_preserves_logits_and_compresses_state() {
    assert_linear_forward_with_rho_chunk_compression_preserves_logits_and_compresses_state(
        crate::RhoCompressionConfig::Int8BlockExp,
        1.0e-3,
    );
}

#[test]
fn linear_forward_with_rho_ternary_chunk_compression_preserves_logits_and_compresses_state() {
    assert_linear_forward_with_rho_chunk_compression_preserves_logits_and_compresses_state(
        crate::RhoCompressionConfig::TernaryBlockExp,
        1.5e-2,
    );
}

#[test]
fn linear_forward_with_rho_binary_chunk_compression_preserves_logits_and_compresses_state() {
    assert_linear_forward_with_rho_chunk_compression_preserves_logits_and_compresses_state(
        crate::RhoCompressionConfig::BinaryBlockExp,
        6.0e-2,
    );
}

fn assert_linear_forward_with_rho_chunk_compression_preserves_logits_and_compresses_state(
    compression: crate::RhoCompressionConfig,
    max_diff_tolerance: f32,
) {
    let device = <RecurrenceBackend as BackendTrait>::Device::default();
    let dense_model = recurrence_test_model(BDHConfig {
        n_layer: 2,
        n_embd: 8,
        n_head: 2,
        mlp_internal_dim_multiplier: 2,
        vocab_size: 32,
        dropout: 0.0,
        sequence_kernel: kernel_linear_attention(),
        ..Default::default()
    });
    let mut compressed_model = dense_model.clone();
    compressed_model.low_bit_rho = Ignored(crate::LowBitRhoConfig {
        compression,
        compression_interval: crate::RhoCompressionInterval::Chunk,
        ..Default::default()
    });
    let tokens = Tensor::<RecurrenceBackend, 2, Int>::from_data(
        TensorData::new(vec![1, 2, 3], [1, 3]),
        &device,
    );
    let mut dense_state = dense_model.init_state();
    let mut compressed_state = compressed_model.init_state();

    let dense_logits = dense_model.forward_with_state(tokens.clone(), &mut dense_state);
    let compressed_logits = compressed_model.forward_with_state(tokens, &mut compressed_state);
    let max_diff = tensor_max_abs_diff(dense_logits, compressed_logits);
    println!(
        "rho compression {:?} max_logit_diff={max_diff:.6}",
        compression
    );

    assert!(
        compressed_state.layers.iter().all(|layer| {
            (layer.packed_rho.is_some() || layer.packed_rho_int8_device.is_some())
                && layer.rho.is_none()
        }),
        "expected {compression:?} rho compression to store packed rho state"
    );
    assert!(
        max_diff <= max_diff_tolerance,
        "expected {compression:?} rho carry to stay close to dense carry, max diff {max_diff}"
    );
}

#[test]
fn mamba_forward_with_state_populates_sequence_aux() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 2,
            n_embd: 8,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.0,
            sequence_kernel: kernel_mamba1(),
            ..Default::default()
        },
        &device,
    );
    let tokens =
        Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![1, 2, 3], [1, 3]), &device);
    let mut state = model.init_state();
    let _ = model.forward_with_state(tokens, &mut state);

    assert!(state.layers.iter().all(|layer| layer.rho.is_some()));
    assert!(
        state
            .layers
            .iter()
            .all(|layer| layer.sequence_aux.is_some())
    );
    assert!(state.layers.iter().all(|layer| layer.rho_norm.is_none()));
}

#[test]
fn mamba2_forward_with_state_populates_sequence_aux() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 2,
            n_embd: 8,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.0,
            sequence_kernel: kernel_mamba2(),
            mamba: MambaSequenceConfig {
                headdim: 8,
                ..Default::default()
            },
            ..Default::default()
        },
        &device,
    );
    let tokens =
        Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![1, 2, 3], [1, 3]), &device);
    let mut state = model.init_state();
    let _ = model.forward_with_state(tokens, &mut state);

    assert!(state.layers.iter().all(|layer| layer.rho.is_some()));
    assert!(
        state
            .layers
            .iter()
            .all(|layer| layer.sequence_aux.is_some())
    );
    assert!(state.layers.iter().all(|layer| layer.rho_norm.is_none()));
}

#[test]
fn linear_full_forward_matches_token_step_recurrence() {
    assert_full_forward_matches_token_step_recurrence(kernel_linear_attention());
}

#[test]
fn vanilla_language_pipeline_state_does_not_track_residual_history() {
    let model = recurrence_test_model(BDHConfig {
        n_layer: 2,
        n_embd: 8,
        n_head: 2,
        mlp_internal_dim_multiplier: 4,
        vocab_size: 32,
        dropout: 0.0,
        sequence_kernel: kernel_linear_attention(),
        residual_connector: ResidualConnectorKind::Vanilla,
        fused_kernels: FusedKernelConfig {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    });
    let device = <RecurrenceBackend as BackendTrait>::Device::default();
    let tokens = recurrence_test_tokens_with_shape(&device, vec![1, 2, 3, 4], [1, 4]);
    let pipeline_state = model.begin_language_pipeline(tokens);

    assert!(pipeline_state.residual_history().is_empty());
}

#[test]
fn attention_residual_full_forward_matches_token_step_recurrence() {
    let model = recurrence_test_model(BDHConfig {
        n_layer: 3,
        n_embd: 8,
        n_head: 2,
        mlp_internal_dim_multiplier: 4,
        vocab_size: 32,
        dropout: 0.0,
        sequence_kernel: kernel_linear_attention(),
        residual_connector: ResidualConnectorKind::AttentionResidual,
        attention_residual: crate::AttentionResidualConfig {
            enabled: true,
            num_heads: 1,
            history_window: None,
            dropout: 0.0,
            recency_bias: 1.5,
            ..Default::default()
        },
        fused_kernels: FusedKernelConfig {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    });
    let device = <RecurrenceBackend as BackendTrait>::Device::default();
    let tokens = recurrence_test_tokens_with_shape(&device, vec![1, 2, 3, 4, 5, 6], [1, 6]);
    let logits_full = model.forward(tokens.clone());
    let mut recurrent_state = model.init_state();
    let mut logits_steps = Vec::new();
    for step in 0..tokens.shape().dims::<2>()[1] {
        let step_tokens = tokens.clone().slice_dim(1, step..step + 1);
        logits_steps.push(model.forward_with_state(step_tokens, &mut recurrent_state));
    }
    let logits_stepwise = Tensor::cat(logits_steps, 1);
    let max_diff = tensor_max_abs_diff(logits_full, logits_stepwise);
    assert!(
        max_diff <= 1.0e-4,
        "expected attention residual full forward and token-step recurrence to match, max diff {max_diff}"
    );
}

#[test]
fn block_attention_residual_full_forward_matches_token_step_recurrence() {
    let model = recurrence_test_model(BDHConfig {
        n_layer: 4,
        n_embd: 8,
        n_head: 2,
        mlp_internal_dim_multiplier: 4,
        vocab_size: 32,
        dropout: 0.0,
        sequence_kernel: kernel_linear_attention(),
        residual_connector: ResidualConnectorKind::BlockAttentionResidual,
        block_attention_residual: crate::BlockAttentionResidualConfig {
            enabled: true,
            num_heads: 1,
            layers_per_block: 2,
            block_history_window: Some(2),
            intra_block_history_window: Some(1),
            dropout: 0.0,
            recency_bias: 1.5,
            ..Default::default()
        },
        fused_kernels: FusedKernelConfig {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    });
    let device = <RecurrenceBackend as BackendTrait>::Device::default();
    let tokens = recurrence_test_tokens_with_shape(&device, vec![1, 2, 3, 4, 5, 6], [1, 6]);
    let logits_full = model.forward(tokens.clone());
    let mut recurrent_state = model.init_state();
    let mut logits_steps = Vec::new();
    for step in 0..tokens.shape().dims::<2>()[1] {
        let step_tokens = tokens.clone().slice_dim(1, step..step + 1);
        logits_steps.push(model.forward_with_state(step_tokens, &mut recurrent_state));
    }
    let logits_stepwise = Tensor::cat(logits_steps, 1);
    let max_diff = tensor_max_abs_diff(logits_full, logits_stepwise);
    assert!(
        max_diff <= 1.0e-4,
        "expected block attention residual full forward and token-step recurrence to match, max diff {max_diff}"
    );
}

#[test]
fn rwkv8_full_forward_matches_token_step_recurrence() {
    assert_full_forward_matches_token_step_recurrence(kernel_rwkv8());
}

#[test]
fn mamba_full_forward_matches_token_step_recurrence() {
    assert_full_forward_matches_token_step_recurrence(kernel_mamba1());
}

#[test]
fn mamba2_full_forward_matches_token_step_recurrence() {
    assert_full_forward_matches_token_step_recurrence(kernel_mamba2());
}

#[test]
fn linear_chunked_recurrence_matches_uninterrupted_state_and_logits() {
    assert_chunked_recurrence_matches_uninterrupted_state(kernel_linear_attention());
}

#[test]
fn linear_dense_score_full_forward_matches_token_step_recurrence() {
    assert_full_forward_matches_token_step_recurrence(kernel_linear_dense_score());
}

#[test]
fn linear_dense_score_chunked_recurrence_matches_uninterrupted_state_and_logits() {
    assert_chunked_recurrence_matches_uninterrupted_state(kernel_linear_dense_score());
}

#[test]
fn rwkv8_chunked_recurrence_matches_uninterrupted_state_and_logits() {
    assert_chunked_recurrence_matches_uninterrupted_state(kernel_rwkv8());
}

#[test]
fn mamba_chunked_recurrence_matches_uninterrupted_state_and_logits() {
    assert_chunked_recurrence_matches_uninterrupted_state(kernel_mamba1());
}

#[test]
fn mamba2_chunked_recurrence_matches_uninterrupted_state_and_logits() {
    assert_chunked_recurrence_matches_uninterrupted_state(kernel_mamba2());
}

#[test]
fn linear_multi_head_recurrence_matches_token_step_and_chunked_state() {
    let token_values = vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_full_forward_matches_token_step_recurrence_with_shape(
        kernel_linear_attention(),
        3,
        12,
        3,
        72,
        64,
        token_values.clone(),
        [2, 5],
    );
    assert_chunked_recurrence_matches_uninterrupted_state_with_shape(
        kernel_linear_attention(),
        3,
        12,
        3,
        72,
        64,
        token_values,
        [2, 5],
        &[1, 2, 4],
    );
}

#[test]
fn linear_dense_score_multi_head_recurrence_matches_token_step_and_chunked_state() {
    let token_values = vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_full_forward_matches_token_step_recurrence_with_shape(
        kernel_linear_dense_score(),
        3,
        12,
        3,
        72,
        64,
        token_values.clone(),
        [2, 5],
    );
    assert_chunked_recurrence_matches_uninterrupted_state_with_shape(
        kernel_linear_dense_score(),
        3,
        12,
        3,
        72,
        64,
        token_values,
        [2, 5],
        &[1, 2, 4],
    );
}

#[test]
fn rwkv8_multi_head_recurrence_matches_token_step_and_chunked_state() {
    let token_values = vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_full_forward_matches_token_step_recurrence_with_shape(
        kernel_rwkv8(),
        3,
        12,
        3,
        72,
        64,
        token_values.clone(),
        [2, 5],
    );
    assert_chunked_recurrence_matches_uninterrupted_state_with_shape(
        kernel_rwkv8(),
        3,
        12,
        3,
        72,
        64,
        token_values,
        [2, 5],
        &[1, 2, 4],
    );
}

#[test]
fn linear_tensorized_parallel_reference_matches_host_loop_reference() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 1,
            n_embd: 2,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 16,
            dropout: 0.0,
            ..Default::default()
        },
        &device,
    );

    let query = Tensor::<Backend, 4>::from_data(
        TensorData::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 1.5, 0.5, 0.5, 1.5, 2.5, 3.5, 3.0, 2.0, 1.0, 0.5,
                1.25, 0.75, 2.25, 1.75, 0.25, 1.0, 1.5, 2.0,
            ],
            [1, 2, 3, 4],
        ),
        &device,
    );
    let value = Tensor::<Backend, 4>::from_data(
        TensorData::new(vec![1.0, 0.5, 0.25, 1.5, 2.0, 1.0], [1, 1, 3, 2]),
        &device,
    );

    let (context_host, rho_host) =
        model.recurrent_attention_reference(query.clone(), value.clone(), None, None);
    let (context_tensorized, rho_tensorized) =
        recurrent_attention_tensorized_no_decay_reference(query, value);

    assert!(tensor_max_abs_diff(context_host, context_tensorized) <= 1.0e-4);
    assert!(tensor_max_abs_diff(rho_host, rho_tensorized) <= 1.0e-4);
}

#[test]
fn rwkv8_tensorized_parallel_reference_matches_host_loop_reference() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 1,
            n_embd: 2,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 16,
            dropout: 0.0,
            sequence_kernel: kernel_rwkv8(),
            ..Default::default()
        },
        &device,
    );

    let query = Tensor::<Backend, 4>::from_data(
        TensorData::new(
            vec![
                1.0, 2.0, 1.5, 0.5, 2.0, 1.0, 0.5, 1.5, 1.25, 0.75, 2.25, 1.75, 0.25, 1.0, 1.5,
                2.0, 0.75, 1.25, 1.0, 2.0, 2.5, 1.5, 0.75, 1.25,
            ],
            [1, 2, 3, 4],
        ),
        &device,
    );
    let value = Tensor::<Backend, 4>::from_data(
        TensorData::new(vec![1.0, 0.5, 0.25, 1.5, 2.0, 1.0], [1, 1, 3, 2]),
        &device,
    );
    let decay = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.95, 0.9, 0.85, 0.8, 0.9, 0.85, 0.8, 0.75], [1, 2, 4]),
        &device,
    );

    let (context_host, rho_host, rho_norm_host) = model.recurrent_rwkv8_state_space_reference(
        query.clone(),
        value.clone(),
        None,
        None,
        decay.clone(),
    );
    let (context_tensorized, rho_tensorized, rho_norm_tensorized) =
        recurrent_rwkv8_tensorized_reference(query, value, decay);

    assert!(tensor_max_abs_diff(context_host, context_tensorized) <= 1.0e-4);
    assert!(tensor_max_abs_diff(rho_host, rho_tensorized) <= 1.0e-4);
    assert!(tensor_max_abs_diff(rho_norm_host, rho_norm_tensorized) <= 1.0e-4);
}

#[test]
fn rwkv8_kernel_tensorized_forward_matches_host_loop_reference() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 1,
            n_embd: 2,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 16,
            dropout: 0.0,
            sequence_kernel: kernel_rwkv8(),
            ..Default::default()
        },
        &device,
    );

    let query = Tensor::<Backend, 4>::from_data(
        TensorData::new(
            vec![
                1.0, 2.0, 1.5, 0.5, 2.0, 1.0, 0.5, 1.5, 1.25, 0.75, 2.25, 1.75, 0.25, 1.0, 1.5,
                2.0, 0.75, 1.25, 1.0, 2.0, 2.5, 1.5, 0.75, 1.25,
            ],
            [1, 2, 3, 4],
        ),
        &device,
    );
    let value = Tensor::<Backend, 4>::from_data(
        TensorData::new(vec![1.0, 0.5, 0.25, 1.5, 2.0, 1.0], [1, 1, 3, 2]),
        &device,
    );
    let decay = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.95, 0.9, 0.85, 0.8, 0.9, 0.85, 0.8, 0.75], [1, 2, 4]),
        &device,
    );

    let (context_host, rho_host, rho_norm_host) = model.recurrent_rwkv8_state_space_reference(
        query.clone(),
        value.clone(),
        None,
        None,
        decay.clone(),
    );
    let tensorized = tensorized_rwkv8_forward(query, value, None, None, decay);

    assert!(tensor_max_abs_diff(context_host, tensorized.context) <= 1.0e-4);
    assert!(tensor_max_abs_diff(rho_host, tensorized.rho) <= 1.0e-4);
    assert!(tensor_max_abs_diff(rho_norm_host, tensorized.rho_norm) <= 1.0e-4);
}

#[test]
fn rwkv8_kernel_scan_fallback_matches_host_loop_reference() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 1,
            n_embd: 2,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 16,
            dropout: 0.0,
            sequence_kernel: kernel_rwkv8(),
            ..Default::default()
        },
        &device,
    );

    let query = Tensor::<Backend, 4>::from_data(
        TensorData::new(
            vec![
                1.0, 2.0, 1.5, 0.5, 2.0, 1.0, 0.5, 1.5, 1.25, 0.75, 2.25, 1.75, 0.25, 1.0, 1.5,
                2.0, 0.75, 1.25, 1.0, 2.0, 2.5, 1.5, 0.75, 1.25,
            ],
            [1, 2, 3, 4],
        ),
        &device,
    );
    let value = Tensor::<Backend, 4>::from_data(
        TensorData::new(vec![1.0, 0.5, 0.25, 1.5, 2.0, 1.0], [1, 1, 3, 2]),
        &device,
    );
    let decay = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.95, 0.9, 0.85, 0.8, 0.9, 0.85, 0.8, 0.75], [1, 2, 4]),
        &device,
    );

    let (context_host, rho_host, rho_norm_host) = model.recurrent_rwkv8_state_space_reference(
        query.clone(),
        value.clone(),
        None,
        None,
        decay.clone(),
    );

    unsafe {
        std::env::set_var(
            "BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_SCAN_THRESHOLD_BYTES",
            "1",
        )
    };
    let tensorized = tensorized_rwkv8_forward(query, value, None, None, decay);
    unsafe { std::env::remove_var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_SCAN_THRESHOLD_BYTES") };

    assert!(tensor_max_abs_diff(context_host, tensorized.context) <= 1.0e-4);
    assert!(tensor_max_abs_diff(rho_host, tensorized.rho) <= 1.0e-4);
    assert!(tensor_max_abs_diff(rho_norm_host, tensorized.rho_norm) <= 1.0e-4);
}

#[test]
fn rwkv8_kernel_matmul_fallback_matches_host_loop_reference() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 1,
            n_embd: 2,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 16,
            dropout: 0.0,
            sequence_kernel: kernel_rwkv8(),
            ..Default::default()
        },
        &device,
    );

    let query = Tensor::<Backend, 4>::from_data(
        TensorData::new(
            vec![
                1.0, 2.0, 1.5, 0.5, 2.0, 1.0, 0.5, 1.5, 1.25, 0.75, 2.25, 1.75, 0.25, 1.0, 1.5,
                2.0, 0.75, 1.25, 1.0, 2.0, 2.5, 1.5, 0.75, 1.25,
            ],
            [1, 2, 3, 4],
        ),
        &device,
    );
    let value = Tensor::<Backend, 4>::from_data(
        TensorData::new(vec![1.0, 0.5, 0.25, 1.5, 2.0, 1.0], [1, 1, 3, 2]),
        &device,
    );
    let decay = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.95, 0.9, 0.85, 0.8, 0.9, 0.85, 0.8, 0.75], [1, 2, 4]),
        &device,
    );

    let (context_host, rho_host, rho_norm_host) = model.recurrent_rwkv8_state_space_reference(
        query.clone(),
        value.clone(),
        None,
        None,
        decay.clone(),
    );

    unsafe {
        std::env::set_var(
            "BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_SCAN_THRESHOLD_BYTES",
            "1",
        );
        std::env::set_var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_MATMUL_MAX_CHUNK", "8");
    };
    let tensorized = tensorized_rwkv8_forward(query, value, None, None, decay);
    unsafe {
        std::env::remove_var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_SCAN_THRESHOLD_BYTES");
        std::env::remove_var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_MATMUL_MAX_CHUNK");
    };

    assert!(tensor_max_abs_diff(context_host, tensorized.context) <= 1.0e-4);
    assert!(tensor_max_abs_diff(rho_host, tensorized.rho) <= 1.0e-4);
    assert!(tensor_max_abs_diff(rho_norm_host, tensorized.rho_norm) <= 1.0e-4);
}

#[test]
fn mamba_kernel_tensorized_forward_matches_reference() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let config =
        MambaSequenceConfig::default().resolve(8, SequenceMemorySystem::Mamba1SelectiveScan);
    let params = MambaSequenceParameters::<Backend>::new(
        config,
        SequenceMemorySystem::Mamba1SelectiveScan,
        &device,
    );
    let hidden = Tensor::<Backend, 4>::from_data(
        TensorData::new(
            (0..(2 * 1 * 5 * 8))
                .map(|idx| ((idx % 19) as f32) / 19.0 - 0.3)
                .collect::<Vec<_>>(),
            [2, 1, 5, 8],
        ),
        &device,
    );

    let (context_host, state_host) = mamba_reference(hidden.clone(), &params, None);
    let params_mamba1 = params.mamba1().expect("mamba1 params");
    let tensorized = tensorized_mamba_forward(
        hidden,
        config.d_inner,
        config.d_state,
        config.d_conv,
        config.dt_rank,
        params_mamba1.in_proj_tensor(),
        params_mamba1.conv_weight_tensor(),
        params_mamba1.conv_bias_tensor(),
        params_mamba1.x_proj_tensor(),
        params_mamba1.dt_proj_weight_tensor(),
        params_mamba1.dt_proj_bias_tensor(),
        params_mamba1.a_log_tensor(),
        params_mamba1.d_skip_tensor(),
        params_mamba1.out_proj_tensor(),
        None,
    );

    assert!(tensor_max_abs_diff(context_host, tensorized.context) <= 1.0e-4);
    assert!(tensor_max_abs_diff(state_host.conv, tensorized.state.conv) <= 1.0e-4);
    assert!(tensor_max_abs_diff(state_host.ssm, tensorized.state.ssm) <= 1.0e-4);
}

#[test]
fn mamba2_kernel_tensorized_forward_matches_reference() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let config = MambaSequenceConfig {
        headdim: 8,
        ..Default::default()
    }
    .resolve(8, SequenceMemorySystem::Mamba2StateSpaceDuality);
    let params = MambaSequenceParameters::<Backend>::new(
        config,
        SequenceMemorySystem::Mamba2StateSpaceDuality,
        &device,
    );
    let hidden = Tensor::<Backend, 4>::from_data(
        TensorData::new(
            (0..(2 * 1 * 5 * 8))
                .map(|idx| ((idx % 19) as f32) / 19.0 - 0.3)
                .collect::<Vec<_>>(),
            [2, 1, 5, 8],
        ),
        &device,
    );

    let (context_host, state_host) = mamba_reference(hidden.clone(), &params, None);
    let params_mamba2 = params.mamba2().expect("mamba2 params");
    let tensorized = tensorized_mamba2_forward(
        hidden,
        config.d_inner,
        config.d_state,
        config.d_conv,
        config.headdim,
        config.ngroups,
        params_mamba2.in_proj_tensor(),
        params_mamba2.conv_weight_tensor(),
        params_mamba2.conv_bias_tensor(),
        params_mamba2.dt_bias_tensor(),
        params_mamba2.a_log_tensor(),
        params_mamba2.d_skip_tensor(),
        params_mamba2.norm_weight_tensor(),
        config.norm_eps,
        params_mamba2.out_proj_tensor(),
        None::<Mamba2TensorizedState<Backend>>,
    );

    assert!(tensor_max_abs_diff(context_host, tensorized.context) <= 1.0e-4);
    assert!(tensor_max_abs_diff(state_host.conv, tensorized.state.conv) <= 1.0e-4);
    assert!(tensor_max_abs_diff(state_host.ssm, tensorized.state.ssm) <= 1.0e-4);
}

#[test]
fn linear_dense_score_reference_matches_host_loop_reference_with_decay() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 1,
            n_embd: 2,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 16,
            dropout: 0.0,
            ..Default::default()
        },
        &device,
    );

    let query = Tensor::<Backend, 4>::from_data(
        TensorData::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 1.5, 0.5, 0.5, 1.5, 2.5, 3.5, 3.0, 2.0, 1.0, 0.5,
                1.25, 0.75, 2.25, 1.75, 0.25, 1.0, 1.5, 2.0,
            ],
            [1, 2, 3, 4],
        ),
        &device,
    );
    let value = Tensor::<Backend, 4>::from_data(
        TensorData::new(vec![1.0, 0.5, 0.25, 1.5, 2.0, 1.0], [1, 1, 3, 2]),
        &device,
    );
    let decay = Tensor::<Backend, 1>::from_data(TensorData::new(vec![0.9, 0.8], [2]), &device);

    let (context_host, rho_host) = model.recurrent_attention_reference(
        query.clone(),
        value.clone(),
        None,
        Some(decay.clone()),
    );
    let (context_dense, rho_dense) =
        recurrent_attention_dense_score_reference(query, value, Some(decay));

    assert!(tensor_max_abs_diff(context_host, context_dense) <= 1.0e-4);
    assert!(tensor_max_abs_diff(rho_host, rho_dense) <= 1.0e-4);
}

#[test]
fn linear_reference_wrapper_preserves_rope_sequential_positions() {
    assert_recurrent_attention_wrapper_preserves_position_semantics(
        kernel_linear_attention(),
        crate::RotaryEmbedding::Rope,
        RecurrentPositionMode::Sequential,
        3,
    );
}

#[test]
fn linear_reference_wrapper_preserves_alibi_fixed_positions() {
    assert_recurrent_attention_wrapper_preserves_position_semantics(
        kernel_linear_attention(),
        crate::RotaryEmbedding::Alibi,
        RecurrentPositionMode::Fixed,
        5,
    );
}

#[test]
fn linear_dense_score_wrapper_preserves_rope_sequential_positions() {
    assert_recurrent_attention_wrapper_preserves_position_semantics(
        kernel_linear_dense_score(),
        crate::RotaryEmbedding::Rope,
        RecurrentPositionMode::Sequential,
        2,
    );
}

#[test]
fn linear_dense_score_wrapper_preserves_alibi_fixed_positions() {
    assert_recurrent_attention_wrapper_preserves_position_semantics(
        kernel_linear_dense_score(),
        crate::RotaryEmbedding::Alibi,
        RecurrentPositionMode::Fixed,
        4,
    );
}

#[test]
fn bdh_mhc_two_view_wrapper_matches_manual_layer_contract() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 1,
            n_embd: 8,
            n_head: 1,
            mlp_internal_dim_multiplier: 1,
            vocab_size: 16,
            dropout: 0.0,
            mhc: super::super::mhc::ManifoldHyperConnectionsConfig {
                enabled: true,
                num_streams: 1,
                num_views: 2,
                mhc_iters: 4,
                mhc_tau: 0.1,
                add_branch_out_to_residual: true,
                dropout: 0.0,
                ..Default::default()
            },
            ..Default::default()
        },
        &device,
    );
    let tokens =
        Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![1, 2, 3], [1, 3]), &device);
    let mut state = model.init_state();
    let (hidden, _logits) = model.forward_with_hidden_and_state(tokens.clone(), &mut state);

    let embedded = model.embed.forward(tokens);
    let [batch, time, dim] = embedded.shape().dims::<3>();
    let current = model.norm.forward(embedded.reshape([batch, 1, time, dim]));
    let mhc = model.mhc_for_layer(0).expect("mhc");
    let coeffs = mhc.coefficients();
    let (branch_input, residuals_base, beta) =
        mhc_split_with_coefficients(Some(mhc), current, Some(&coeffs));
    let [branch_batch, branch_views, branch_time, branch_dim] = branch_input.shape().dims::<4>();
    let branch_flat =
        branch_input.reshape([branch_batch * branch_views, 1, branch_time, branch_dim]);

    let encoder = model
        .encoder
        .val()
        .reshape([1, 1, dim, model.mlp_internal_dim_multiplier * dim]);
    let encoder_v =
        model
            .encoder_v
            .val()
            .reshape([1, 1, dim, model.mlp_internal_dim_multiplier * dim]);
    let decoder = model.decoder.val();
    let mut layer_state = LayerState {
        rho: None,
        packed_rho: None,
        packed_rho_int8_device: None,
        rho_norm: None,
        sequence_aux: None,
        y_neuron_state: None,
        clocked_slow_hidden: None,
        summary_memory_hidden: None,
        #[cfg(any(feature = "viz", feature = "probe"))]
        viz: None,
    };
    let output = lowrank_residual_step(
        branch_flat,
        encoder,
        encoder_v,
        decoder,
        &model.dropout,
        false,
        false,
        0.0,
        true,
        LowBitProjectionPlan::default(),
        LowBitSavedActivationConfig::default(),
        PackedLowBitProjectionArtifacts::default(),
        &model.kernel.block_sparse.latent,
        model.kernel.lowrank_grad_input_executor,
        None,
        |query, value| {
            model.recurrent_attention_with_plan(
                query,
                value,
                &mut layer_state,
                0,
                RecurrentPositionMode::Sequential,
                None,
            )
        },
        activation::relu,
        |values| model.norm.forward(values),
    );
    let branch_out = output
        .next
        .reshape([branch_batch, branch_views, branch_time, branch_dim]);
    let manual = model
        .norm
        .forward(mhc_merge_with_coefficients(
            Some(mhc),
            branch_out,
            residuals_base,
            Some(&coeffs),
            beta,
        ))
        .reshape([batch, time, dim]);

    let hidden_vec = hidden
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("hidden vec");
    let manual_vec = manual
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("manual vec");

    assert_eq!(hidden_vec, manual_vec);
}

#[test]
fn bdh_mhc_dynamic_stream_wrapper_matches_manual_layer_contract() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 1,
            n_embd: 8,
            n_head: 1,
            mlp_internal_dim_multiplier: 1,
            vocab_size: 16,
            dropout: 0.0,
            mhc: super::super::mhc::ManifoldHyperConnectionsConfig {
                enabled: true,
                num_streams: 2,
                num_views: 1,
                coefficient_policy:
                    super::super::mhc::ManifoldHyperConnectionCoefficientPolicy::DynamicPositive,
                mhc_iters: 4,
                mhc_tau: 0.1,
                add_branch_out_to_residual: true,
                dropout: 0.0,
                ..Default::default()
            },
            ..Default::default()
        },
        &device,
    );
    let tokens =
        Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![1, 2, 3], [1, 3]), &device);
    let mut state = model.init_state();
    let (hidden, _logits) = model.forward_with_hidden_and_state(tokens.clone(), &mut state);

    let embedded = model.embed.forward(tokens);
    let [batch, time, dim] = embedded.shape().dims::<3>();
    let current = model.norm.forward(embedded.reshape([batch, 1, time, dim]));
    let mhc = model.mhc_for_layer(0).expect("mhc");
    let connector = model.residual_connector_for_layer(0);
    let current_residuals = model.prepare_language_residuals(current, &connector);
    let stream_output = mhc.stream_width_connection(current_residuals);
    let branch_input = stream_output.branch_input.clone();
    let encoder = model
        .encoder
        .val()
        .reshape([1, 1, dim, model.mlp_internal_dim_multiplier * dim]);
    let encoder_v =
        model
            .encoder_v
            .val()
            .reshape([1, 1, dim, model.mlp_internal_dim_multiplier * dim]);
    let decoder = model.decoder.val();
    let mut layer_state = LayerState {
        rho: None,
        packed_rho: None,
        packed_rho_int8_device: None,
        rho_norm: None,
        sequence_aux: None,
        y_neuron_state: None,
        clocked_slow_hidden: None,
        summary_memory_hidden: None,
        #[cfg(any(feature = "viz", feature = "probe"))]
        viz: None,
    };
    let output = lowrank_residual_step(
        branch_input.clone(),
        encoder,
        encoder_v,
        decoder,
        &model.dropout,
        false,
        false,
        0.0,
        true,
        LowBitProjectionPlan::default(),
        LowBitSavedActivationConfig::default(),
        PackedLowBitProjectionArtifacts::default(),
        &model.kernel.block_sparse.latent,
        model.kernel.lowrank_grad_input_executor,
        None,
        |query, value| {
            model.recurrent_attention_with_plan(
                query,
                value,
                &mut layer_state,
                0,
                RecurrentPositionMode::Sequential,
                None,
            )
        },
        activation::relu,
        |values| model.norm.forward(values),
    );
    let manual = model.collapse_language_streams(model.norm.forward(mhc.stream_depth_connection(
        output.next,
        stream_output.residuals_out,
        &stream_output.coefficients,
    )));

    let hidden_vec = hidden
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("hidden vec");
    let manual_vec = manual
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("manual vec");

    assert_eq!(hidden_vec, manual_vec);
}

#[test]
fn bdh_language_mhc_diagnostics_report_non_uniform_stream_behavior() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 1,
            n_embd: 8,
            n_head: 1,
            mlp_internal_dim_multiplier: 1,
            vocab_size: 16,
            dropout: 0.0,
            mhc: super::super::mhc::ManifoldHyperConnectionsConfig {
                enabled: true,
                num_streams: 3,
                num_views: 1,
                coefficient_policy:
                    super::super::mhc::ManifoldHyperConnectionCoefficientPolicy::DynamicPositive,
                mhc_iters: 4,
                mhc_tau: 0.1,
                add_branch_out_to_residual: true,
                dropout: 0.0,
                ..Default::default()
            },
            ..Default::default()
        },
        &device,
    );
    let tokens =
        Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![1, 2, 3], [1, 3]), &device);

    let diagnostics = model.collect_language_mhc_diagnostics(tokens);
    let layer0 = diagnostics.first().expect("layer diagnostics");

    assert_eq!(layer0.layer_index, 0);
    assert_eq!(layer0.num_streams, 3);
    assert!(layer0.stream_norm_mean.is_finite());
    assert!(layer0.alpha_entropy_normalized_mean < 1.0);
    assert!(
        layer0
            .pairwise_stream_cosine_mean
            .is_some_and(|value| value.is_finite()),
        "pairwise cosine should be finite for multi-stream diagnostics"
    );
    assert!(layer0.residual_distance_identity_l1_mean > 0.0);
}

#[test]
fn bdh_init_diagnostics_report_gated_lowrank_metrics() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 2,
            n_embd: 8,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.0,
            ..Default::default()
        },
        &device,
    );
    let tokens =
        Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![1, 2, 3, 4], [1, 4]), &device);

    let diagnostics = model.collect_language_bdh_init_diagnostics(tokens);

    assert_eq!(diagnostics.len(), 2);
    for (layer_idx, layer) in diagnostics.iter().enumerate() {
        assert_eq!(layer.layer_index, layer_idx);
        assert!(layer.lowrank_path_active);
        assert!(layer.finite);
        assert!(layer.p_x.is_some_and(|value| (0.0..=1.0).contains(&value)));
        assert!(layer.p_y.is_some_and(|value| (0.0..=1.0).contains(&value)));
        assert!(
            layer
                .current_rms
                .is_some_and(|value| value.is_finite() && value >= 0.0)
        );
        assert!(
            layer
                .recurrent_readout_ratio
                .is_some_and(|value| value.is_finite() && value >= 0.0)
        );
        assert!(
            layer
                .r_res
                .is_some_and(|value| value.is_finite() && value >= 0.0)
        );
    }
}

#[test]
fn bdh_explicit_firing_thresholds_reduce_sparse_activation_rates() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    Backend::seed(&device, 7);
    let control = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 1,
            n_embd: 8,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.0,
            ..Default::default()
        },
        &device,
    );
    Backend::seed(&device, 7);
    let sparse = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 1,
            n_embd: 8,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.0,
            initialization: super::super::init::BdhInitializationConfig {
                firing_targets: super::super::init::BdhFiringTargetConfig {
                    kind: super::super::init::BdhFiringTargetKind::ExplicitThresholds,
                    x_threshold: 10.0,
                    y_threshold: 10.0,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        },
        &device,
    );
    let tokens =
        Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![1, 2, 3, 4], [1, 4]), &device);

    let control_diag = control.collect_language_bdh_init_diagnostics(tokens.clone());
    let sparse_diag = sparse.collect_language_bdh_init_diagnostics(tokens);

    let control_layer = control_diag.first().expect("control layer");
    let sparse_layer = sparse_diag.first().expect("sparse layer");
    assert!(control_layer.p_x.expect("control p_x") > sparse_layer.p_x.expect("sparse p_x"));
    assert!(control_layer.p_y.expect("control p_y") > sparse_layer.p_y.expect("sparse p_y"));
}

#[test]
fn bdh_mhc_single_stream_single_view_skips_per_layer_mhc_allocation() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 2,
            n_embd: 8,
            n_head: 1,
            mlp_internal_dim_multiplier: 1,
            vocab_size: 16,
            dropout: 0.0,
            mhc: super::super::mhc::ManifoldHyperConnectionsConfig {
                enabled: true,
                num_streams: 1,
                num_views: 1,
                mhc_iters: 4,
                mhc_tau: 0.1,
                add_branch_out_to_residual: true,
                dropout: 0.0,
                ..Default::default()
            },
            ..Default::default()
        },
        &device,
    );
    let tokens =
        Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![1, 2, 3], [1, 3]), &device);
    let output = model.forward(tokens);
    let [batch, time, vocab] = output.shape().dims::<3>();

    assert!(model.mhc_shared.is_none());
    assert_eq!([batch, time, vocab], [1, 3, 16]);
}

#[test]
fn bdh_mhc_multi_stream_language_contract_collapses_back_to_hidden() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 2,
            n_embd: 8,
            n_head: 1,
            mlp_internal_dim_multiplier: 1,
            vocab_size: 16,
            dropout: 0.0,
            mhc: super::super::mhc::ManifoldHyperConnectionsConfig {
                enabled: true,
                num_streams: 2,
                num_views: 1,
                mhc_iters: 4,
                mhc_tau: 0.1,
                add_branch_out_to_residual: true,
                dropout: 0.0,
                ..Default::default()
            },
            ..Default::default()
        },
        &device,
    );
    let tokens =
        Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![1, 2, 3], [1, 3]), &device);
    let output = model.forward(tokens);
    let [batch, time, vocab] = output.shape().dims::<3>();

    assert!(model.mhc_shared.is_some());
    assert_eq!([batch, time, vocab], [1, 3, 16]);
}

#[test]
fn bdh_mhc_shared_wrapper_respects_last_layers_gate() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 4,
            n_embd: 8,
            n_head: 1,
            mlp_internal_dim_multiplier: 1,
            vocab_size: 16,
            dropout: 0.0,
            mhc: super::super::mhc::ManifoldHyperConnectionsConfig {
                enabled: true,
                num_streams: 2,
                num_views: 1,
                last_layers: Some(1),
                mhc_iters: 4,
                mhc_tau: 0.1,
                add_branch_out_to_residual: true,
                dropout: 0.0,
                ..Default::default()
            },
            ..Default::default()
        },
        &device,
    );
    assert!(model.mhc_shared.is_some());
    assert_eq!(model.mhc_first_layer, 3);
    assert!(model.mhc_for_layer(0).is_none());
    assert!(model.mhc_for_layer(1).is_none());
    assert!(model.mhc_for_layer(2).is_none());
    assert!(model.mhc_for_layer(3).is_some());
}

#[test]
fn bdh_mhc_reuses_same_wrapper_weights_across_active_layers() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 3,
            n_embd: 8,
            n_head: 1,
            mlp_internal_dim_multiplier: 1,
            vocab_size: 16,
            dropout: 0.0,
            mhc: super::super::mhc::ManifoldHyperConnectionsConfig {
                enabled: true,
                num_streams: 2,
                num_views: 1,
                coefficient_policy:
                    super::super::mhc::ManifoldHyperConnectionCoefficientPolicy::DynamicPositive,
                mhc_iters: 4,
                mhc_tau: 0.1,
                add_branch_out_to_residual: true,
                dropout: 0.0,
                ..Default::default()
            },
            ..Default::default()
        },
        &device,
    );

    let layer0 = model.mhc_for_layer(0).expect("layer 0 mhc");
    let layer1 = model.mhc_for_layer(1).expect("layer 1 mhc");
    let layer2 = model.mhc_for_layer(2).expect("layer 2 mhc");

    assert!(std::ptr::eq(layer0, layer1));
    assert!(std::ptr::eq(layer1, layer2));
}

#[test]
fn bdh_layer_lowrank_weights_follow_latent_fanout_schedule() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 4,
            n_embd: 8,
            n_head: 2,
            mlp_internal_dim_multiplier: 8,
            latent_fanout_schedule: Some(LatentFanoutScheduleConfig::LateLayer {
                base_latent_total: 16,
                last_layers: 2,
            }),
            vocab_size: 16,
            dropout: 0.0,
            ..Default::default()
        },
        &device,
    );

    let (encoder0, encoder_v0, decoder0, latent0) = model.layer_lowrank_weights(0);
    let (encoder3, encoder_v3, decoder3, latent3) = model.layer_lowrank_weights(3);

    assert_eq!(latent0, 8);
    assert_eq!(encoder0.shape().dims::<4>(), [1, 2, 8, 8]);
    assert_eq!(encoder_v0.shape().dims::<4>(), [1, 2, 8, 8]);
    assert_eq!(decoder0.shape().dims::<2>(), [16, 8]);

    assert_eq!(latent3, 32);
    assert_eq!(encoder3.shape().dims::<4>(), [1, 2, 8, 32]);
    assert_eq!(encoder_v3.shape().dims::<4>(), [1, 2, 8, 32]);
    assert_eq!(decoder3.shape().dims::<2>(), [64, 8]);
}

#[test]
fn y_neuron_recurrence_persists_across_calls_and_changes_next_token() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let prefix = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.0, 2.0, 0.0, 4.0], [1, 2, 2]),
        &device,
    );
    let suffix =
        Tensor::<Backend, 3>::from_data(TensorData::new(vec![0.0, 6.0], [1, 1, 2]), &device);

    let mut baseline = deterministic_y_neuron_recurrence_model(YNeuronRecurrenceConfig::default());
    let mut baseline_state = baseline.init_state();
    let _ = baseline.forward_with_hidden_and_state_embedded(prefix.clone(), &mut baseline_state);
    assert!(baseline_state.layers[0].y_neuron_state.is_none());
    let (baseline_hidden, _) =
        baseline.forward_with_hidden_and_state_embedded(suffix.clone(), &mut baseline_state);

    baseline.y_neuron_recurrence = YNeuronRecurrenceConfig {
        enabled: true,
        carry_in_scale: 0.5,
        last_layers: None,
        chunk_tokens: 1,
        state_decay: 1.0,
        state_update_scale: 1.0,
        state_rms_cap: None,
    };
    let mut recurrent_state = baseline.init_state();
    let _ = baseline.forward_with_hidden_and_state_embedded(prefix, &mut recurrent_state);
    let carried_state = recurrent_state.layers[0]
        .y_neuron_state
        .as_ref()
        .expect("recurrent y_neuron state");
    let carried_vec = carried_state
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("carried y_neuron state vec");
    assert!(
        carried_vec.iter().any(|value| value.abs() > 1.0e-5),
        "expected non-zero carried y_neuron state"
    );
    let (recurrent_hidden, _) =
        baseline.forward_with_hidden_and_state_embedded(suffix, &mut recurrent_state);

    let baseline_vec = baseline_hidden
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("baseline suffix hidden vec");
    let recurrent_vec = recurrent_hidden
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("recurrent suffix hidden vec");
    let diff: f32 = baseline_vec
        .iter()
        .zip(recurrent_vec.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .sum();
    assert!(
        diff > 1.0e-4,
        "expected y_neuron recurrence to change the next-token hidden state, got diff {diff}"
    );
}

#[test]
fn y_neuron_recurrence_state_rms_cap_bounds_carried_state() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let embedded = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.0, 2.0, 0.0, 4.0, 0.0, 6.0], [1, 3, 2]),
        &device,
    );
    let cap = 0.25f32;
    let model = deterministic_y_neuron_recurrence_model(YNeuronRecurrenceConfig {
        enabled: true,
        carry_in_scale: 0.5,
        last_layers: None,
        chunk_tokens: 1,
        state_decay: 1.0,
        state_update_scale: 4.0,
        state_rms_cap: Some(cap),
    });
    let mut state = model.init_state();
    let _ = model.forward_with_hidden_and_state_embedded(embedded, &mut state);
    let carried_state = state.layers[0]
        .y_neuron_state
        .as_ref()
        .expect("bounded carried y_neuron state")
        .clone();
    let rms = carried_state
        .clone()
        .powf_scalar(2.0)
        .mean_dim(2)
        .sqrt()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rms vec");
    assert!(
        rms.iter().all(|value| *value <= cap + 1.0e-4),
        "expected carried y_neuron state rms <= {cap}, got {rms:?}"
    );
}

#[test]
fn y_neuron_recurrence_chunked_mode_persists_across_calls_and_changes_next_token() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let prefix = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.0, 2.0, 0.0, 4.0], [1, 2, 2]),
        &device,
    );
    let suffix =
        Tensor::<Backend, 3>::from_data(TensorData::new(vec![0.0, 6.0], [1, 1, 2]), &device);

    let mut baseline = deterministic_y_neuron_recurrence_model(YNeuronRecurrenceConfig::default());
    let mut baseline_state = baseline.init_state();
    let _ = baseline.forward_with_hidden_and_state_embedded(prefix.clone(), &mut baseline_state);
    let (baseline_hidden, _) =
        baseline.forward_with_hidden_and_state_embedded(suffix.clone(), &mut baseline_state);

    baseline.y_neuron_recurrence = YNeuronRecurrenceConfig {
        enabled: true,
        carry_in_scale: 0.5,
        last_layers: None,
        chunk_tokens: 2,
        state_decay: 1.0,
        state_update_scale: 1.0,
        state_rms_cap: None,
    };
    let mut recurrent_state = baseline.init_state();
    let _ = baseline.forward_with_hidden_and_state_embedded(prefix, &mut recurrent_state);
    let carried_state = recurrent_state.layers[0]
        .y_neuron_state
        .as_ref()
        .expect("chunked recurrent y_neuron state");
    let carried_vec = carried_state
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("chunked carried state vec");
    assert!(
        carried_vec.iter().any(|value| value.abs() > 1.0e-5),
        "expected non-zero carried y_neuron state in chunked mode"
    );
    let (recurrent_hidden, _) =
        baseline.forward_with_hidden_and_state_embedded(suffix, &mut recurrent_state);

    let baseline_vec = baseline_hidden
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("baseline suffix hidden vec");
    let recurrent_vec = recurrent_hidden
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("chunked recurrent suffix hidden vec");
    let diff: f32 = baseline_vec
        .iter()
        .zip(recurrent_vec.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .sum();
    assert!(
        diff > 1.0e-4,
        "expected chunked y_neuron recurrence to change the next-token hidden state, got diff {diff}"
    );
}

#[test]
fn y_neuron_recurrence_last_layers_only_updates_top_layers() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let embedded = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.0, 2.0, 0.0, 4.0], [1, 2, 2]),
        &device,
    );
    let model = deterministic_y_neuron_recurrence_model_with_layers(
        YNeuronRecurrenceConfig {
            enabled: true,
            carry_in_scale: 0.5,
            last_layers: Some(1),
            chunk_tokens: 1,
            state_decay: 1.0,
            state_update_scale: 1.0,
            state_rms_cap: None,
        },
        2,
    );
    let mut state = model.init_state();
    let _ = model.forward_with_hidden_and_state_embedded(embedded, &mut state);
    assert!(
        state.layers[0].y_neuron_state.is_none(),
        "non-recurrent lower layers should not carry y_neuron state"
    );
    let top_state = state.layers[1]
        .y_neuron_state
        .as_ref()
        .expect("top recurrent layer should carry y_neuron state");
    let top_vec = top_state
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("top layer carried state vec");
    assert!(
        top_vec.iter().any(|value| value.abs() > 1.0e-5),
        "expected non-zero carried y_neuron state on the recurrent top layer"
    );
}

#[test]
fn summary_memory_reads_previous_chunk_instead_of_self_summary() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 1,
            n_embd: 4,
            n_head: 1,
            mlp_internal_dim_multiplier: 1,
            vocab_size: 16,
            dropout: 0.0,
            summary_memory: SummaryMemoryConfig {
                enabled: true,
                last_layers: Some(1),
                chunk_tokens: 2,
                residual_scale: 0.5,
                state_decay: 1.0,
                state_update_scale: 1.0,
                surprise_gate_threshold: 0.0,
                surprise_gate_sharpness: 8.0,
                write_trigger_text: None,
                write_trigger_token_ids: None,
            },
            ..Default::default()
        },
        &device,
    );

    let mut layer_state = LayerState {
        rho: None,
        packed_rho: None,
        packed_rho_int8_device: None,
        rho_norm: None,
        sequence_aux: None,
        y_neuron_state: None,
        clocked_slow_hidden: None,
        summary_memory_hidden: None,
        #[cfg(any(feature = "viz", feature = "probe"))]
        viz: None,
    };
    let first_chunk =
        Tensor::<Backend, 4>::from_data(TensorData::new(vec![2.0, 4.0], [1, 1, 2, 1]), &device);
    let first_out = model.forward_branch_summary_memory(first_chunk, &mut layer_state, 0, None);
    let first_vec = first_out
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("first chunk vec");
    assert_eq!(first_vec, vec![2.0, 4.0]);

    let second_chunk =
        Tensor::<Backend, 4>::from_data(TensorData::new(vec![10.0], [1, 1, 1, 1]), &device);
    let second_out = model.forward_branch_summary_memory(second_chunk, &mut layer_state, 2, None);
    let second_vec = second_out
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("second chunk vec");
    assert_eq!(second_vec, vec![11.5]);
}

#[test]
fn summary_memory_surprise_gate_preserves_prior_carry_when_chunk_is_unsurprising() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 1,
            n_embd: 4,
            n_head: 1,
            mlp_internal_dim_multiplier: 1,
            vocab_size: 16,
            dropout: 0.0,
            summary_memory: SummaryMemoryConfig {
                enabled: true,
                last_layers: Some(1),
                chunk_tokens: 1,
                residual_scale: 0.5,
                state_decay: 1.0,
                state_update_scale: 1.0,
                surprise_gate_threshold: 0.5,
                surprise_gate_sharpness: 8.0,
                write_trigger_text: None,
                write_trigger_token_ids: None,
            },
            ..Default::default()
        },
        &device,
    );

    let mut layer_state = LayerState {
        rho: None,
        packed_rho: None,
        packed_rho_int8_device: None,
        rho_norm: None,
        sequence_aux: None,
        y_neuron_state: None,
        clocked_slow_hidden: None,
        summary_memory_hidden: None,
        #[cfg(any(feature = "viz", feature = "probe"))]
        viz: None,
    };

    let first_chunk =
        Tensor::<Backend, 4>::from_data(TensorData::new(vec![2.0], [1, 1, 1, 1]), &device);
    let _ = model.forward_branch_summary_memory(first_chunk, &mut layer_state, 0, None);

    let nearly_same_chunk =
        Tensor::<Backend, 4>::from_data(TensorData::new(vec![2.01], [1, 1, 1, 1]), &device);
    let _ = model.forward_branch_summary_memory(nearly_same_chunk, &mut layer_state, 1, None);

    let probe_chunk =
        Tensor::<Backend, 4>::from_data(TensorData::new(vec![10.0], [1, 1, 1, 1]), &device);
    let probe_out = model.forward_branch_summary_memory(probe_chunk, &mut layer_state, 2, None);
    let probe_vec = probe_out
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("probe vec");
    assert!(
        (probe_vec[0] - 11.0).abs() < 1.0e-3,
        "expected prior carry to remain near 2.0, got {:?}",
        probe_vec
    );
}

#[test]
fn summary_memory_write_trigger_updates_only_on_event_chunks() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = BDH::<Backend>::new(
        BDHConfig {
            n_layer: 1,
            n_embd: 4,
            n_head: 1,
            mlp_internal_dim_multiplier: 1,
            vocab_size: 16,
            dropout: 0.0,
            summary_memory: SummaryMemoryConfig {
                enabled: true,
                last_layers: Some(1),
                chunk_tokens: 2,
                residual_scale: 0.5,
                state_decay: 1.0,
                state_update_scale: 1.0,
                surprise_gate_threshold: 0.0,
                surprise_gate_sharpness: 8.0,
                write_trigger_text: None,
                write_trigger_token_ids: Some(vec![7]),
            },
            ..Default::default()
        },
        &device,
    );

    let mut layer_state = LayerState {
        rho: None,
        packed_rho: None,
        packed_rho_int8_device: None,
        rho_norm: None,
        sequence_aux: None,
        y_neuron_state: None,
        clocked_slow_hidden: None,
        summary_memory_hidden: None,
        #[cfg(any(feature = "viz", feature = "probe"))]
        viz: None,
    };
    let first_chunk =
        Tensor::<Backend, 4>::from_data(TensorData::new(vec![2.0, 4.0], [1, 1, 2, 1]), &device);
    let no_event_mask =
        Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![0i64, 0], [1, 2]), &device);
    let _ =
        model.forward_branch_summary_memory(first_chunk, &mut layer_state, 0, Some(no_event_mask));

    let second_chunk =
        Tensor::<Backend, 4>::from_data(TensorData::new(vec![10.0], [1, 1, 1, 1]), &device);
    let no_update_out =
        model.forward_branch_summary_memory(second_chunk.clone(), &mut layer_state, 2, None);
    let no_update_vec = no_update_out
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("no update vec");
    assert!(
        (no_update_vec[0] - 10.0).abs() < 1.0e-4,
        "expected no summary carry before a trigger, got {:?}",
        no_update_vec
    );

    let event_chunk =
        Tensor::<Backend, 4>::from_data(TensorData::new(vec![6.0, 8.0], [1, 1, 2, 1]), &device);
    let event_mask =
        Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![0i64, 1], [1, 2]), &device);
    let _ = model.forward_branch_summary_memory(event_chunk, &mut layer_state, 2, Some(event_mask));

    let probe_out = model.forward_branch_summary_memory(second_chunk, &mut layer_state, 4, None);
    let probe_vec = probe_out
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("probe vec");
    assert!(
        probe_vec[0] > 10.0,
        "expected trigger-gated summary carry to affect the next chunk, got {:?}",
        probe_vec
    );
}
