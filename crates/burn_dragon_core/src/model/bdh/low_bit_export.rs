use super::language_head::LanguageHeadDeployScaffold;
use anyhow::{Result, anyhow};
use burn::module::{Module, Param};
use burn::nn::Embedding;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use std::convert::TryInto;

use crate::experimental::bitnet_reference::{
    BdhBitNetStaticArtifacts, PackedWeightArtifact, pack_weight_artifact_from_dequantized_values,
    pack_weight_artifact_from_format,
};
use crate::model::low_bit_runtime::{
    cache_decoder_tail_artifact, cache_lowrank_projection_artifact,
};
use crate::model::sequence::SequenceMemorySystem;
use crate::model::sequence::mamba::MambaSequenceParameters;
use crate::model::{
    AttentionResidual, BDH, BlockAttentionResidual, DragonNorm, LowBitWeightFormat,
    ManifoldHyperConnections, fake_quantize_weight_ste,
};

#[derive(Module, Debug)]
pub struct BdhBitNetDeployScaffold<B: Backend> {
    embed: Embedding<B>,
    norm: DragonNorm<B>,
    mhc_shared: Option<ManifoldHyperConnections<B>>,
    attention_residual_shared: Option<AttentionResidual<B>>,
    block_attention_residual_shared: Option<BlockAttentionResidual<B>>,
    rwkv_time_decay: Option<Param<Tensor<B, 2>>>,
    mamba: Option<MambaSequenceParameters<B>>,
    lm_head: Option<Param<Tensor<B, 2>>>,
    nca_factorized_lm_head: Option<Param<Tensor<B, 2>>>,
    nca_special_lm_head: Option<Param<Tensor<B, 2>>>,
}

impl<B: Backend> BDH<B> {
    pub fn export_bitnet_static_artifacts(&self) -> BdhBitNetStaticArtifacts {
        let plan = self.low_bit_projection_plan();
        BdhBitNetStaticArtifacts {
            decoder_x: export_weight_artifact(self.encoder.val(), plan.x_weight_format),
            decoder_y: export_weight_artifact(self.encoder_v.val(), plan.y_weight_format),
            encoder: export_weight_artifact(self.decoder.val(), plan.residual_weight_format),
        }
    }

    pub fn export_bitnet_deploy_scaffold(&self) -> BdhBitNetDeployScaffold<B> {
        let LanguageHeadDeployScaffold {
            lm_head,
            nca_factorized_lm_head,
            nca_special_lm_head,
        } = self.clone_language_head_deploy_scaffold();
        BdhBitNetDeployScaffold {
            embed: self.embed.clone(),
            norm: self.norm.clone(),
            mhc_shared: self.mhc_shared.clone(),
            attention_residual_shared: self.attention_residual_shared.clone(),
            block_attention_residual_shared: self.block_attention_residual_shared.clone(),
            rwkv_time_decay: (self.sequence_kernel.memory_system
                == SequenceMemorySystem::Rwkv8StateSpace)
                .then(|| self.rwkv_time_decay.clone()),
            mamba: self.mamba.clone(),
            lm_head,
            nca_factorized_lm_head,
            nca_special_lm_head,
        }
    }

    pub fn apply_bitnet_static_artifacts(
        &mut self,
        artifacts: &BdhBitNetStaticArtifacts,
        device: &B::Device,
    ) -> Result<()> {
        self.packed_decoder_x = validate_weight_artifact_shape(
            artifacts.decoder_x.as_ref(),
            self.encoder.val().shape().dims::<3>(),
            "decoder_x",
        )?;
        if let Some(artifact) = self.packed_decoder_x.as_ref() {
            let _ =
                cache_lowrank_projection_artifact::<B>(artifact, device, "cached packed decoder_x");
        }
        self.packed_decoder_y = validate_weight_artifact_shape(
            artifacts.decoder_y.as_ref(),
            self.encoder_v.val().shape().dims::<3>(),
            "decoder_y",
        )?;
        if let Some(artifact) = self.packed_decoder_y.as_ref() {
            let _ =
                cache_lowrank_projection_artifact::<B>(artifact, device, "cached packed decoder_y");
        }
        self.packed_encoder = validate_weight_artifact_shape(
            artifacts.encoder.as_ref(),
            self.decoder.val().shape().dims::<2>(),
            "encoder",
        )?;
        if let Some(artifact) = self.packed_encoder.as_ref() {
            let heads = self.n_head;
            let latent_per_head = self.decoder.val().shape().dims::<2>()[0] / heads;
            let _ = cache_decoder_tail_artifact::<B>(
                artifact,
                heads,
                latent_per_head,
                device,
                "cached packed encoder",
            );
        }
        Ok(())
    }

    pub fn clear_bitnet_static_artifacts(&mut self) {
        self.packed_decoder_x = None;
        self.packed_decoder_y = None;
        self.packed_encoder = None;
    }
}

fn export_weight_artifact<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    format: Option<LowBitWeightFormat>,
) -> Option<PackedWeightArtifact> {
    let format = format?;
    let shape = tensor.shape().dims::<D>().to_vec();
    let weights = tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("f32 weights");
    match format {
        LowBitWeightFormat::Int8 => pack_weight_artifact_from_format(&weights, &shape, format),
        LowBitWeightFormat::Fp16 => None,
        LowBitWeightFormat::Sign1
        | LowBitWeightFormat::Ternary158
        | LowBitWeightFormat::Packed2 => {
            let quantized = fake_quantize_weight_ste(tensor, format)
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("quantized weights");
            pack_weight_artifact_from_dequantized_values(&quantized, &shape, format)
        }
    }
}

fn validate_weight_artifact_shape<const D: usize>(
    artifact: Option<&PackedWeightArtifact>,
    expected_shape: [usize; D],
    name: &str,
) -> Result<Option<PackedWeightArtifact>> {
    let Some(artifact) = artifact else {
        return Ok(None);
    };
    let dims: [usize; D] = artifact
        .logical_shape
        .clone()
        .try_into()
        .map_err(|_| anyhow!("bitnet artifact `{name}` has wrong rank"))?;
    if dims != expected_shape {
        return Err(anyhow!(
            "bitnet artifact `{name}` shape mismatch: expected {:?}, got {:?}",
            expected_shape,
            dims
        ));
    }
    Ok(Some(artifact.clone()))
}
