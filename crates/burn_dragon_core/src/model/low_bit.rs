use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BitNetLowBitProtocol {
    #[default]
    None,
    BitnetB1,
    BitnetB158,
    BitnetA48Exp,
    BitnetV2Exp,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LowBitTrainingMode {
    #[default]
    QatSte,
    TrainKernelExp,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LowBitInferenceMode {
    #[default]
    OfflinePack,
    RuntimeFakeQuant,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LowBitWeightFormat {
    #[default]
    Fp16,
    Int8,
    Sign1,
    Ternary158,
    Packed2,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum LowBitActivationFormat {
    #[default]
    Fp16,
    Int8,
    Int4Exp,
    Uint8PosExp,
}

impl LowBitActivationFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Int8 => "int8",
            Self::Int4Exp => "int4_exp",
            Self::Uint8PosExp => "uint8_pos_exp",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LowBitSavedActivationMode {
    #[default]
    Disabled,
    QuantizedCacheExp,
    QuantizedCacheRecomputeExp,
}

impl LowBitSavedActivationMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::QuantizedCacheExp => "quantized_cache_exp",
            Self::QuantizedCacheRecomputeExp => "quantized_cache_recompute_exp",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct LowBitSavedActivationConfig {
    pub mode: LowBitSavedActivationMode,
    pub format: LowBitActivationFormat,
}

impl Default for LowBitSavedActivationConfig {
    fn default() -> Self {
        Self {
            mode: LowBitSavedActivationMode::default(),
            format: LowBitActivationFormat::Int8,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LowBitActivationGrouping {
    #[default]
    PerToken,
    PerGroup,
    PerHeadGroup,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LowBitWeightGrouping {
    #[default]
    PerTensor,
    PerGroup,
    PerHeadGroup,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LowBitTargetModule {
    Encoder,
    DecoderX,
    DecoderY,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct LowBitQuantizationConfig {
    pub enable: bool,
    pub protocol: BitNetLowBitProtocol,
    pub training_mode: LowBitTrainingMode,
    pub inference_mode: LowBitInferenceMode,
    pub weight_format: LowBitWeightFormat,
    pub act_format: LowBitActivationFormat,
    pub saved_activations: LowBitSavedActivationConfig,
    pub target_modules: Vec<LowBitTargetModule>,
    pub decoder_x_mode: LowBitWeightFormat,
    pub act_grouping: LowBitActivationGrouping,
    pub weight_grouping: LowBitWeightGrouping,
    pub strict_bitnet_reference: bool,
}

impl Default for LowBitQuantizationConfig {
    fn default() -> Self {
        Self {
            enable: false,
            protocol: BitNetLowBitProtocol::default(),
            training_mode: LowBitTrainingMode::default(),
            inference_mode: LowBitInferenceMode::default(),
            weight_format: LowBitWeightFormat::default(),
            act_format: LowBitActivationFormat::default(),
            saved_activations: LowBitSavedActivationConfig::default(),
            target_modules: Vec::new(),
            decoder_x_mode: LowBitWeightFormat::default(),
            act_grouping: LowBitActivationGrouping::default(),
            weight_grouping: LowBitWeightGrouping::default(),
            strict_bitnet_reference: false,
        }
    }
}

impl LowBitQuantizationConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.enable {
            if !matches!(
                self.saved_activations.mode,
                LowBitSavedActivationMode::Disabled
            ) {
                return Err(anyhow!(
                    "model.quant.saved_activations.mode requires model.quant.enable = true"
                ));
            }
            return Ok(());
        }
        if matches!(self.protocol, BitNetLowBitProtocol::None) {
            return Err(anyhow!(
                "model.quant.protocol must not be \"none\" when model.quant.enable = true"
            ));
        }
        if self.strict_bitnet_reference && self.target_modules.is_empty() {
            return Err(anyhow!(
                "model.quant.target_modules must not be empty when strict_bitnet_reference is enabled"
            ));
        }
        if matches!(self.act_grouping, LowBitActivationGrouping::PerGroup)
            || matches!(self.act_grouping, LowBitActivationGrouping::PerHeadGroup)
        {
            return Err(anyhow!(
                "model.quant.act_grouping currently supports only \"per_token\""
            ));
        }
        if !matches!(
            self.saved_activations.format,
            LowBitActivationFormat::Fp16
                | LowBitActivationFormat::Int8
                | LowBitActivationFormat::Int4Exp
        ) {
            return Err(anyhow!(
                "model.quant.saved_activations.format currently supports only \"fp16\", \"int8\", or \"int4_exp\""
            ));
        }
        if !matches!(
            self.saved_activations.mode,
            LowBitSavedActivationMode::Disabled
        ) && !matches!(self.training_mode, LowBitTrainingMode::TrainKernelExp)
        {
            return Err(anyhow!(
                "model.quant.saved_activations.mode currently requires training_mode = \"train_kernel_exp\""
            ));
        }
        if matches!(self.weight_grouping, LowBitWeightGrouping::PerGroup)
            || matches!(self.weight_grouping, LowBitWeightGrouping::PerHeadGroup)
        {
            return Err(anyhow!(
                "model.quant.weight_grouping currently supports only \"per_tensor\""
            ));
        }
        if matches!(self.act_format, LowBitActivationFormat::Uint8PosExp)
            && self
                .target_modules
                .iter()
                .any(|module| !matches!(module, LowBitTargetModule::Encoder))
        {
            return Err(anyhow!(
                "model.quant.act_format = \"uint8_pos_exp\" is currently only supported when target_modules contains only \"encoder\""
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RhoPrecisionConfig {
    Fp32,
    #[default]
    Bf16,
    Fp8Exp,
    Int8BlockExp,
    Blockfp8Exp,
    SparseTileExp,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RhoCompressionConfig {
    #[default]
    None,
    Fp8BlockExp,
    Int8BlockExp,
    BlockfpExp,
    SparsePositiveExp,
    LowrankDeltaExp,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RhoCompressionInterval {
    Step,
    #[default]
    Chunk,
    EvalOnly,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct LowBitRhoConfig {
    pub precision: RhoPrecisionConfig,
    pub carry_across_tbptt: bool,
    pub detach_between_windows: bool,
    pub compression: RhoCompressionConfig,
    pub compression_interval: RhoCompressionInterval,
    pub stats_monitoring: bool,
}

impl Default for LowBitRhoConfig {
    fn default() -> Self {
        Self {
            precision: RhoPrecisionConfig::default(),
            carry_across_tbptt: true,
            detach_between_windows: true,
            compression: RhoCompressionConfig::default(),
            compression_interval: RhoCompressionInterval::default(),
            stats_monitoring: true,
        }
    }
}

impl LowBitRhoConfig {
    pub fn validate(&self) -> Result<()> {
        if matches!(self.precision, RhoPrecisionConfig::Fp8Exp)
            && matches!(self.compression, RhoCompressionConfig::Fp8BlockExp)
        {
            return Err(anyhow!(
                "model.rho.precision = \"fp8_exp\" and model.rho.compression = \"fp8_block_exp\" should not both be set at once"
            ));
        }
        if !matches!(
            self.precision,
            RhoPrecisionConfig::Fp32 | RhoPrecisionConfig::Bf16
        ) {
            return Err(anyhow!(
                "model.rho.precision currently supports only \"fp32\" or \"bf16\" in the canonical additive path; lower-precision rho remains a separate experimental follow-on track"
            ));
        }
        if !matches!(
            self.compression,
            RhoCompressionConfig::None | RhoCompressionConfig::Int8BlockExp
        ) {
            return Err(anyhow!(
                "model.rho.compression currently supports only \"none\" or \"int8_block_exp\""
            ));
        }
        if matches!(self.compression, RhoCompressionConfig::Int8BlockExp)
            && !matches!(self.compression_interval, RhoCompressionInterval::Chunk)
        {
            return Err(anyhow!(
                "model.rho.compression = \"int8_block_exp\" currently supports only compression_interval = \"chunk\""
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_bit_quantization_requires_protocol_when_enabled() {
        let config = LowBitQuantizationConfig {
            enable: true,
            protocol: BitNetLowBitProtocol::None,
            ..Default::default()
        };
        let err = config.validate().expect_err("expected validation failure");
        assert!(
            err.to_string().contains("model.quant.protocol"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn low_bit_quantization_accepts_enabled_bitnet_protocol() {
        let config = LowBitQuantizationConfig {
            enable: true,
            protocol: BitNetLowBitProtocol::BitnetB158,
            target_modules: vec![LowBitTargetModule::Encoder, LowBitTargetModule::DecoderY],
            ..Default::default()
        };
        config.validate().expect("expected valid config");
    }

    #[test]
    fn rho_config_rejects_double_fp8_setting() {
        let config = LowBitRhoConfig {
            precision: RhoPrecisionConfig::Fp8Exp,
            compression: RhoCompressionConfig::Fp8BlockExp,
            ..Default::default()
        };
        let err = config.validate().expect_err("expected validation failure");
        assert!(
            err.to_string().contains("should not both be set"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn low_bit_quantization_rejects_non_default_grouping_modes() {
        let config = LowBitQuantizationConfig {
            enable: true,
            protocol: BitNetLowBitProtocol::BitnetB158,
            act_grouping: LowBitActivationGrouping::PerGroup,
            ..Default::default()
        };
        let err = config.validate().expect_err("expected validation failure");
        assert!(
            err.to_string().contains("act_grouping"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn low_bit_quantization_rejects_saved_activation_mode_without_enable() {
        let config = LowBitQuantizationConfig {
            saved_activations: LowBitSavedActivationConfig {
                mode: LowBitSavedActivationMode::QuantizedCacheExp,
                ..Default::default()
            },
            ..Default::default()
        };
        let err = config.validate().expect_err("expected validation failure");
        assert!(
            err.to_string().contains("saved_activations.mode requires"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn low_bit_quantization_accepts_train_kernel_saved_activation_recompute() {
        let config = LowBitQuantizationConfig {
            enable: true,
            protocol: BitNetLowBitProtocol::BitnetB158,
            training_mode: LowBitTrainingMode::TrainKernelExp,
            saved_activations: LowBitSavedActivationConfig {
                mode: LowBitSavedActivationMode::QuantizedCacheRecomputeExp,
                format: LowBitActivationFormat::Int8,
            },
            target_modules: vec![LowBitTargetModule::Encoder],
            ..Default::default()
        };
        config.validate().expect("expected valid config");
    }

    #[test]
    fn rho_config_rejects_unimplemented_precision_modes() {
        let config = LowBitRhoConfig {
            precision: RhoPrecisionConfig::Fp8Exp,
            ..Default::default()
        };
        let err = config.validate().expect_err("expected validation failure");
        assert!(
            err.to_string().contains("currently supports only"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rho_config_accepts_int8_block_chunk_compression() {
        let config = LowBitRhoConfig {
            compression: RhoCompressionConfig::Int8BlockExp,
            compression_interval: RhoCompressionInterval::Chunk,
            ..Default::default()
        };
        config
            .validate()
            .expect("expected valid rho compression config");
    }

    #[test]
    fn rho_config_rejects_int8_block_non_chunk_interval() {
        let config = LowBitRhoConfig {
            compression: RhoCompressionConfig::Int8BlockExp,
            compression_interval: RhoCompressionInterval::Step,
            ..Default::default()
        };
        let err = config.validate().expect_err("expected validation failure");
        assert!(
            err.to_string().contains("compression_interval"),
            "unexpected error: {err}"
        );
    }
}
