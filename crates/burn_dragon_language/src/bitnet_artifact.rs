use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

pub const BITNET_ARTIFACT_BINARY_MAGIC: &[u8; 8] = b"BDBITN01";

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct LanguageBitNetArtifactBundle {
    pub schema_version: u32,
    pub source_checkpoint_epoch: usize,
    pub source_training_config_sha256: String,
    #[serde(default)]
    pub source_run_dir: Option<PathBuf>,
    #[serde(default)]
    pub kernel_abi_version: Option<u32>,
    pub quant: burn_dragon_core::LowBitQuantizationConfig,
    pub rho: burn_dragon_core::LowBitRhoConfig,
    #[serde(default)]
    pub deploy_base_burnpack: Option<Vec<u8>>,
    pub static_weights: burn_dragon_core::experimental::bitnet_reference::BdhBitNetStaticArtifacts,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
struct PackedWeightArtifactMetadata {
    encoding: burn_dragon_core::experimental::bitnet_reference::PackedWeightEncoding,
    logical_shape: Vec<usize>,
    scale: f32,
    len: usize,
    packed_len: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
struct BdhBitNetStaticArtifactsMetadata {
    decoder_x: Option<PackedWeightArtifactMetadata>,
    decoder_y: Option<PackedWeightArtifactMetadata>,
    encoder: Option<PackedWeightArtifactMetadata>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
struct DeployBaseBurnpackMetadata {
    len: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
struct LanguageBitNetArtifactBundleMetadata {
    schema_version: u32,
    source_checkpoint_epoch: usize,
    source_training_config_sha256: String,
    #[serde(default)]
    source_run_dir: Option<PathBuf>,
    #[serde(default)]
    kernel_abi_version: Option<u32>,
    quant: burn_dragon_core::LowBitQuantizationConfig,
    rho: burn_dragon_core::LowBitRhoConfig,
    #[serde(default)]
    deploy_base_burnpack: Option<DeployBaseBurnpackMetadata>,
    static_weights: BdhBitNetStaticArtifactsMetadata,
}

fn packed_weight_artifact_metadata(
    artifact: &burn_dragon_core::experimental::bitnet_reference::PackedWeightArtifact,
) -> PackedWeightArtifactMetadata {
    PackedWeightArtifactMetadata {
        encoding: artifact.encoding.clone(),
        logical_shape: artifact.logical_shape.clone(),
        scale: artifact.scale,
        len: artifact.len,
        packed_len: artifact.packed.len(),
    }
}

fn bitnet_artifact_bundle_metadata(
    bundle: &LanguageBitNetArtifactBundle,
) -> LanguageBitNetArtifactBundleMetadata {
    LanguageBitNetArtifactBundleMetadata {
        schema_version: bundle.schema_version,
        source_checkpoint_epoch: bundle.source_checkpoint_epoch,
        source_training_config_sha256: bundle.source_training_config_sha256.clone(),
        source_run_dir: bundle.source_run_dir.clone(),
        kernel_abi_version: bundle.kernel_abi_version,
        quant: bundle.quant.clone(),
        rho: bundle.rho.clone(),
        deploy_base_burnpack: bundle
            .deploy_base_burnpack
            .as_ref()
            .map(|bytes| DeployBaseBurnpackMetadata { len: bytes.len() }),
        static_weights: BdhBitNetStaticArtifactsMetadata {
            decoder_x: bundle
                .static_weights
                .decoder_x
                .as_ref()
                .map(packed_weight_artifact_metadata),
            decoder_y: bundle
                .static_weights
                .decoder_y
                .as_ref()
                .map(packed_weight_artifact_metadata),
            encoder: bundle
                .static_weights
                .encoder
                .as_ref()
                .map(packed_weight_artifact_metadata),
        },
    }
}

fn rebuild_packed_weight_artifact(
    metadata: Option<PackedWeightArtifactMetadata>,
    payload: &[u8],
    offset: &mut usize,
    label: &str,
    name: &str,
) -> Result<Option<burn_dragon_core::experimental::bitnet_reference::PackedWeightArtifact>> {
    let Some(metadata) = metadata else {
        return Ok(None);
    };
    let end = offset.saturating_add(metadata.packed_len);
    if end > payload.len() {
        return Err(anyhow!(
            "bitnet artifact {label} truncated while reading `{name}` payload"
        ));
    }
    let artifact = burn_dragon_core::experimental::bitnet_reference::PackedWeightArtifact {
        encoding: metadata.encoding,
        logical_shape: metadata.logical_shape,
        scale: metadata.scale,
        packed: payload[*offset..end].to_vec(),
        len: metadata.len,
    };
    *offset = end;
    Ok(Some(artifact))
}

fn rebuild_deploy_base_burnpack(
    metadata: Option<DeployBaseBurnpackMetadata>,
    payload: &[u8],
    offset: &mut usize,
    label: &str,
) -> Result<Option<Vec<u8>>> {
    let Some(metadata) = metadata else {
        return Ok(None);
    };
    let end = offset.saturating_add(metadata.len);
    if end > payload.len() {
        return Err(anyhow!(
            "bitnet artifact {label} truncated while reading deploy base burnpack payload"
        ));
    }
    let bytes = payload[*offset..end].to_vec();
    *offset = end;
    Ok(Some(bytes))
}

pub fn serialize_bitnet_artifact_binary(bundle: &LanguageBitNetArtifactBundle) -> Result<Vec<u8>> {
    let metadata = bitnet_artifact_bundle_metadata(bundle);
    let metadata_json =
        serde_json::to_vec(&metadata).context("serialize compact bitnet artifact metadata")?;
    let mut bytes = Vec::with_capacity(
        BITNET_ARTIFACT_BINARY_MAGIC.len()
            + core::mem::size_of::<u64>()
            + metadata_json.len()
            + bundle
                .deploy_base_burnpack
                .as_ref()
                .map_or(0, |payload| payload.len())
            + bundle
                .static_weights
                .decoder_x
                .as_ref()
                .map_or(0, |artifact| artifact.packed.len())
            + bundle
                .static_weights
                .decoder_y
                .as_ref()
                .map_or(0, |artifact| artifact.packed.len())
            + bundle
                .static_weights
                .encoder
                .as_ref()
                .map_or(0, |artifact| artifact.packed.len()),
    );
    bytes.extend_from_slice(BITNET_ARTIFACT_BINARY_MAGIC);
    bytes.extend_from_slice(&(metadata_json.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&metadata_json);
    if let Some(deploy_base_burnpack) = bundle.deploy_base_burnpack.as_ref() {
        bytes.extend_from_slice(deploy_base_burnpack);
    }
    if let Some(artifact) = bundle.static_weights.decoder_x.as_ref() {
        bytes.extend_from_slice(&artifact.packed);
    }
    if let Some(artifact) = bundle.static_weights.decoder_y.as_ref() {
        bytes.extend_from_slice(&artifact.packed);
    }
    if let Some(artifact) = bundle.static_weights.encoder.as_ref() {
        bytes.extend_from_slice(&artifact.packed);
    }
    Ok(bytes)
}

pub fn deserialize_bitnet_artifact_binary(
    bytes: &[u8],
    label: &str,
) -> Result<LanguageBitNetArtifactBundle> {
    if !bytes.starts_with(BITNET_ARTIFACT_BINARY_MAGIC) {
        return Err(anyhow!(
            "BitNet artifact {label} is not in the supported binary format; re-export it as `.bitnet_artifact.bin` or `.bitnet_artifact.bin.gz`"
        ));
    }
    if bytes.len() < BITNET_ARTIFACT_BINARY_MAGIC.len() + core::mem::size_of::<u64>() {
        return Err(anyhow!(
            "bitnet artifact {label} is too small to contain binary header"
        ));
    }
    let metadata_len_offset = BITNET_ARTIFACT_BINARY_MAGIC.len();
    let metadata_len_end = metadata_len_offset + core::mem::size_of::<u64>();
    let metadata_len = u64::from_le_bytes(
        bytes[metadata_len_offset..metadata_len_end]
            .try_into()
            .expect("fixed-size metadata header"),
    ) as usize;
    let metadata_start = metadata_len_end;
    let metadata_end = metadata_start.saturating_add(metadata_len);
    if metadata_end > bytes.len() {
        return Err(anyhow!(
            "bitnet artifact {label} has truncated binary metadata"
        ));
    }
    let metadata: LanguageBitNetArtifactBundleMetadata =
        serde_json::from_slice(&bytes[metadata_start..metadata_end]).with_context(|| {
            format!("failed to parse compact bitnet artifact metadata from {label}")
        })?;
    let payload = &bytes[metadata_end..];
    let mut offset = 0usize;
    let deploy_base_burnpack =
        rebuild_deploy_base_burnpack(metadata.deploy_base_burnpack, payload, &mut offset, label)?;
    let static_weights =
        burn_dragon_core::experimental::bitnet_reference::BdhBitNetStaticArtifacts {
            decoder_x: rebuild_packed_weight_artifact(
                metadata.static_weights.decoder_x,
                payload,
                &mut offset,
                label,
                "decoder_x",
            )?,
            decoder_y: rebuild_packed_weight_artifact(
                metadata.static_weights.decoder_y,
                payload,
                &mut offset,
                label,
                "decoder_y",
            )?,
            encoder: rebuild_packed_weight_artifact(
                metadata.static_weights.encoder,
                payload,
                &mut offset,
                label,
                "encoder",
            )?,
        };
    if offset != payload.len() {
        return Err(anyhow!(
            "bitnet artifact {label} has {} trailing payload bytes",
            payload.len().saturating_sub(offset)
        ));
    }
    Ok(LanguageBitNetArtifactBundle {
        schema_version: metadata.schema_version,
        source_checkpoint_epoch: metadata.source_checkpoint_epoch,
        source_training_config_sha256: metadata.source_training_config_sha256,
        source_run_dir: metadata.source_run_dir,
        kernel_abi_version: metadata.kernel_abi_version,
        quant: metadata.quant,
        rho: metadata.rho,
        deploy_base_burnpack,
        static_weights,
    })
}
