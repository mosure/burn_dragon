use burn_dragon_core::{
    BitNetLowBitProtocol, LowBitActivationFormat, LowBitMemoryEstimateInput,
    LowBitQuantizationConfig, LowBitRhoConfig, LowBitSavedActivationMode, LowBitTargetModule,
    LowBitTrainingMode, LowBitWeightFormat, build_low_bit_saved_activation_inventory,
    estimate_low_bit_memory_buckets,
};

fn main() {
    let input = LowBitMemoryEstimateInput {
        batch_size: 8,
        time_steps: 128,
        n_layer: 8,
        n_head: 4,
        n_embd: 256,
        latent_total: 32768,
    };

    let rho = LowBitRhoConfig::default();
    let base_quant = LowBitQuantizationConfig {
        enable: true,
        protocol: BitNetLowBitProtocol::BitnetB158,
        training_mode: LowBitTrainingMode::TrainKernelExp,
        weight_format: LowBitWeightFormat::Ternary158,
        act_format: LowBitActivationFormat::Int8,
        target_modules: vec![
            LowBitTargetModule::Encoder,
            LowBitTargetModule::DecoderY,
            LowBitTargetModule::DecoderX,
        ],
        decoder_x_mode: LowBitWeightFormat::Int8,
        ..Default::default()
    };

    let cases = [
        ("disabled", LowBitSavedActivationMode::Disabled),
        (
            "quantized_cache_exp",
            LowBitSavedActivationMode::QuantizedCacheExp,
        ),
        (
            "quantized_cache_recompute_exp",
            LowBitSavedActivationMode::QuantizedCacheRecomputeExp,
        ),
    ];

    println!(
        "mode\tmaster_weight_bytes\texecution_weight_bytes\tactivation_shell_bytes\tsaved_activation_bytes\trho_state_bytes\tworkspace_bytes\ttotal_bytes\tinventory_bytes"
    );
    for (name, mode) in cases {
        let mut quant = base_quant.clone();
        quant.saved_activations.mode = mode;
        let estimate = estimate_low_bit_memory_buckets(&quant, &rho, input);
        let inventory_bytes = build_low_bit_saved_activation_inventory(&quant, input)
            .map(|inventory| {
                inventory
                    .tensors
                    .into_iter()
                    .map(|entry| entry.estimated_bytes)
                    .sum::<u64>()
            })
            .unwrap_or(0);
        println!(
            "{name}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            estimate.master_weight_bytes,
            estimate.execution_weight_bytes,
            estimate.activation_shell_bytes,
            estimate.saved_activation_bytes,
            estimate.rho_state_bytes,
            estimate.workspace_bytes,
            estimate.estimated_total_bytes(),
            inventory_bytes,
        );
    }
}
