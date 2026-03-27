use std::path::PathBuf;

use anyhow::Result;
use burn::module::Module;
use burn_dragon::core::{LowBitMemoryEstimateInput, estimate_low_bit_memory_buckets};
use burn_dragon::core::{RhoCompressionConfig, RhoPrecisionConfig};
use burn_dragon_language::train::prepare_dataset;
use burn_dragon_language::{
    BDH, SequenceKernelKind, TrainingConfig, build_model_config_with_tokenizer,
    load_training_config,
};
use burn_ndarray::NdArray;
use clap::Parser;
use serde::Serialize;

type StatsBackend = NdArray<f32>;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, required = true)]
    config: Vec<PathBuf>,
}

#[derive(Serialize)]
struct LanguageModelStatsReport {
    config: Vec<PathBuf>,
    block_size: usize,
    effective_kernel_block_size: usize,
    batch_size: usize,
    vocab_size: usize,
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    latent_total: usize,
    latent_per_head: usize,
    mlp_internal_dim_multiplier: usize,
    shared_layer_weights: bool,
    sequence_kernel: String,
    residual_connector: String,
    quant_enabled: bool,
    quant_target_modules: Vec<String>,
    rho_precision: String,
    rho_compression: String,
    params: usize,
    dense_param_bytes: u64,
    estimated_lowbit_execution_weight_bytes: u64,
    estimated_lowbit_activation_shell_bytes_train: u64,
    estimated_lowbit_saved_activation_bytes_train: u64,
    rho_state_elements_per_batch_view: u64,
    rho_state_bytes_per_batch_view: u64,
    rho_state_bytes_per_train_batch: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = load_training_config(&args.config)?;
    let report = build_report(&config, args.config)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn build_report(
    config: &TrainingConfig,
    config_paths: Vec<PathBuf>,
) -> Result<LanguageModelStatsReport> {
    let dataset = prepare_dataset(&config.dataset, &config.training)?;
    let tokenizer = dataset.tokenizer();
    let model_config = build_model_config_with_tokenizer(
        &config.model,
        config.training.block_size,
        tokenizer.as_ref(),
    )?;
    let device = <StatsBackend as burn::tensor::backend::Backend>::Device::default();
    let model = BDH::<StatsBackend>::new(model_config.clone(), &device);
    let effective_kernel_block_size = config
        .training
        .tbptt_chunk_size
        .filter(|chunk| *chunk > 0 && *chunk < config.training.block_size)
        .unwrap_or(config.training.block_size)
        .max(1);
    let memory_train = estimate_low_bit_memory_buckets(
        &model_config.quant,
        &model_config.rho,
        LowBitMemoryEstimateInput {
            batch_size: config.training.batch_size,
            time_steps: effective_kernel_block_size,
            n_layer: model_config.n_layer,
            n_head: model_config.n_head,
            n_embd: model_config.n_embd,
            latent_total: model_config.latent_total(),
        },
    );
    let params = model.num_params();
    let dense_param_bytes = params as u64 * 4;
    let rho_state_elements_per_batch_view = rho_state_elements_per_batch_view(&model_config);
    let (rho_num, rho_den) =
        rho_bytes_per_element(model_config.rho.precision, model_config.rho.compression);
    let rho_state_bytes_per_batch_view =
        estimate_packed_tensor_bytes(rho_state_elements_per_batch_view, rho_num, rho_den);
    let rho_state_bytes_per_train_batch =
        rho_state_bytes_per_batch_view.saturating_mul(config.training.batch_size as u64);

    Ok(LanguageModelStatsReport {
        config: config_paths,
        block_size: config.training.block_size,
        effective_kernel_block_size,
        batch_size: config.training.batch_size,
        vocab_size: tokenizer.len(),
        n_layer: model_config.n_layer,
        n_embd: model_config.n_embd,
        n_head: model_config.n_head,
        latent_total: model_config.latent_total(),
        latent_per_head: model_config.latent_per_head(),
        mlp_internal_dim_multiplier: model_config.mlp_internal_dim_multiplier,
        shared_layer_weights: true,
        sequence_kernel: format!("{:?}", model_config.sequence_kernel),
        residual_connector: format!("{:?}", model_config.resolved_residual_connector_kind()),
        quant_enabled: model_config.quant.enable,
        quant_target_modules: model_config
            .quant
            .target_modules
            .iter()
            .map(|module| format!("{module:?}"))
            .collect(),
        rho_precision: format!("{:?}", model_config.rho.precision),
        rho_compression: format!("{:?}", model_config.rho.compression),
        params,
        dense_param_bytes,
        estimated_lowbit_execution_weight_bytes: memory_train.execution_weight_bytes,
        estimated_lowbit_activation_shell_bytes_train: memory_train.activation_shell_bytes,
        estimated_lowbit_saved_activation_bytes_train: memory_train.saved_activation_bytes,
        rho_state_elements_per_batch_view,
        rho_state_bytes_per_batch_view,
        rho_state_bytes_per_train_batch,
    })
}

fn rho_state_elements_per_batch_view(model_config: &burn_dragon_language::BDHConfig) -> u64 {
    match model_config.sequence_kernel {
        SequenceKernelKind::MambaSelectiveSsmExperimental => {
            let mamba = model_config.mamba.resolve(model_config.n_embd);
            (model_config.n_layer * mamba.d_inner * mamba.d_state) as u64
        }
        _ => {
            (model_config.n_layer
                * model_config.n_head
                * model_config.latent_per_head()
                * model_config.n_embd) as u64
        }
    }
}

fn rho_bytes_per_element(
    precision: RhoPrecisionConfig,
    compression: RhoCompressionConfig,
) -> (u64, u64) {
    match compression {
        RhoCompressionConfig::Int8BlockExp => return (9, 8),
        RhoCompressionConfig::TernaryBlockExp => return (3, 8),
        RhoCompressionConfig::BinaryBlockExp => return (1, 4),
        _ => {}
    }

    match precision {
        RhoPrecisionConfig::Fp32 => (4, 1),
        RhoPrecisionConfig::Bf16 => (2, 1),
        RhoPrecisionConfig::Fp8Exp
        | RhoPrecisionConfig::Int8BlockExp
        | RhoPrecisionConfig::Blockfp8Exp
        | RhoPrecisionConfig::SparseTileExp => (1, 1),
    }
}

fn estimate_packed_tensor_bytes(elements: u64, bytes_num: u64, bytes_den: u64) -> u64 {
    elements
        .saturating_mul(bytes_num)
        .saturating_add(bytes_den.saturating_sub(1))
        .saturating_div(bytes_den.max(1))
}
