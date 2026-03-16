use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use burn::module::Module;
use burn::tensor::{Bytes, backend::Backend};
use burn_store::{ApplyResult, BurnpackStore, ModuleSnapshot};
use ciborium::Value;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const ONE_MIB: u64 = 1024 * 1024;
pub(crate) const BURNPACK_HEADER_SIZE: usize = 10;
pub(crate) const BURNPACK_MAGIC_NUMBER: u32 = 0x4255_524E;
const TENSOR_ALIGNMENT: u64 = 256;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurnpackPartsManifest {
    #[serde(default = "default_manifest_version")]
    pub version: u32,
    #[serde(default)]
    pub source_file: String,
    #[serde(default)]
    pub source_modified_unix_ms: u64,
    #[serde(default)]
    pub total_bytes: u64,
    #[serde(default)]
    pub max_part_bytes: u64,
    #[serde(default)]
    pub parts: Vec<BurnpackPartEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurnpackPartEntry {
    pub path: String,
    #[serde(default)]
    pub bytes: u64,
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub tensors: usize,
}

#[derive(Debug, Clone)]
pub struct BurnpackPartsReport {
    pub manifest_path: PathBuf,
    pub part_paths: Vec<PathBuf>,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct RawBurnpackMetadata {
    pub tensors: BTreeMap<String, RawTensorDescriptor>,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct RawTensorDescriptor {
    pub dtype: Value,
    pub shape: Vec<u64>,
    pub data_offsets: (u64, u64),
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param_id: Option<u64>,
}

#[derive(Debug, Clone)]
struct TensorRecord {
    name: String,
    descriptor: RawTensorDescriptor,
}

const fn default_manifest_version() -> u32 {
    1
}

#[inline]
const fn align_offset(offset: u64, alignment: u64) -> u64 {
    offset.div_ceil(alignment) * alignment
}

#[inline]
fn aligned_data_section_start(metadata_size: usize) -> usize {
    align_offset(
        (BURNPACK_HEADER_SIZE + metadata_size) as u64,
        TENSOR_ALIGNMENT,
    ) as usize
}

pub fn burnpack_parts_manifest_path(burnpack_path: &Path) -> PathBuf {
    let file_name = burnpack_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.bpk");
    burnpack_path.with_file_name(format!("{file_name}.parts.json"))
}

pub fn save_model_to_burnpack<M, B>(model: &M, output_base: &Path) -> Result<PathBuf, String>
where
    M: Module<B>,
    B: Backend,
{
    let output = normalize_extension(output_base, "bpk");
    let mut store = BurnpackStore::from_file(&output)
        .auto_extension(false)
        .overwrite(true);
    model
        .save_into(&mut store)
        .map_err(|err| format!("failed to save burnpack {}: {err:?}", output.display()))?;
    Ok(output)
}

pub fn save_model_to_burnpack_with_parts<M, B>(
    model: &M,
    output_base: &Path,
    max_part_size_mib: u64,
    overwrite: bool,
) -> Result<BurnpackPartsReport, String>
where
    M: Module<B>,
    B: Backend,
{
    let burnpack_path = save_model_to_burnpack(model, output_base)?;
    write_burnpack_parts(&burnpack_path, max_part_size_mib, overwrite)
}

pub fn ensure_burnpack_parts(
    burnpack_path: &Path,
    max_part_size_mib: u64,
    overwrite: bool,
) -> Result<Option<BurnpackPartsReport>, String> {
    write_burnpack_parts(burnpack_path, max_part_size_mib, overwrite).map(Some)
}

pub fn write_burnpack_parts(
    burnpack_path: &Path,
    max_part_size_mib: u64,
    overwrite: bool,
) -> Result<BurnpackPartsReport, String> {
    if !burnpack_path.exists() {
        return Err(format!(
            "burnpack does not exist for parting: {}",
            burnpack_path.display()
        ));
    }

    let max_part_bytes = max_part_size_mib
        .max(1)
        .checked_mul(ONE_MIB)
        .ok_or_else(|| "max part size overflow".to_string())?;

    let total_bytes = fs::metadata(burnpack_path)
        .map_err(|err| format!("failed to read {} metadata: {err}", burnpack_path.display()))?
        .len();
    let source_modified_unix_ms = file_modified_unix_ms(burnpack_path).unwrap_or(0);
    let manifest_path = burnpack_parts_manifest_path(burnpack_path);
    if manifest_path.exists()
        && !overwrite
        && manifest_has_all_parts(&manifest_path, Some(burnpack_path))
    {
        let manifest = read_parts_manifest(&manifest_path)?;
        let part_paths = manifest
            .parts
            .iter()
            .map(|entry| resolve_part_entry_path(&manifest_path, &entry.path))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(BurnpackPartsReport {
            manifest_path,
            part_paths,
            total_bytes: manifest.total_bytes,
        });
    }

    if overwrite {
        cleanup_existing_parts(&manifest_path)?;
    }
    ensure_parent_dir(&manifest_path)?;

    let mut source = fs::File::open(burnpack_path)
        .map_err(|err| format!("failed to open burnpack {}: {err}", burnpack_path.display()))?;
    let (version, metadata_size, metadata) = read_burnpack_metadata(&mut source, burnpack_path)?;
    let data_start = aligned_data_section_start(metadata_size as usize) as u64;

    let mut tensor_records = metadata
        .tensors
        .iter()
        .map(|(name, descriptor)| TensorRecord {
            name: name.clone(),
            descriptor: descriptor.clone(),
        })
        .collect::<Vec<_>>();
    if tensor_records.is_empty() {
        return Err(format!(
            "burnpack '{}' contains no tensor descriptors",
            burnpack_path.display()
        ));
    }
    tensor_records.sort_by_key(|record| record.descriptor.data_offsets.0);
    let groups = split_tensor_records(tensor_records, max_part_bytes, &metadata.metadata);

    let source_file_name = burnpack_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid burnpack name '{}'", burnpack_path.display()))?;
    let mut part_entries = Vec::with_capacity(groups.len());
    let mut part_paths = Vec::with_capacity(groups.len());
    for (index, group) in groups.iter().enumerate() {
        let part_name = format!("{source_file_name}.part-{index:05}.bpk");
        let part_path = burnpack_path.with_file_name(&part_name);
        if part_path.exists() && overwrite {
            fs::remove_file(&part_path).map_err(|err| {
                format!(
                    "failed to replace stale burnpack part {}: {err}",
                    part_path.display()
                )
            })?;
        }

        write_burnpack_part(
            &mut source,
            &part_path,
            version,
            data_start,
            &metadata.metadata,
            group,
        )?;
        let bytes = fs::metadata(&part_path)
            .map_err(|err| {
                format!(
                    "failed to stat burnpack part {}: {err}",
                    part_path.display()
                )
            })?
            .len();
        let sha256 = sha256_file(&part_path)?;
        part_entries.push(BurnpackPartEntry {
            path: part_name,
            bytes,
            sha256,
            tensors: group.len(),
        });
        part_paths.push(part_path);
    }

    let manifest = BurnpackPartsManifest {
        version: default_manifest_version(),
        source_file: source_file_name.to_string(),
        source_modified_unix_ms,
        total_bytes,
        max_part_bytes,
        parts: part_entries,
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|err| format!("failed to serialize parts manifest: {err}"))?;
    fs::write(&manifest_path, manifest_json).map_err(|err| {
        format!(
            "failed to write burnpack parts manifest {}: {err}",
            manifest_path.display()
        )
    })?;

    Ok(BurnpackPartsReport {
        manifest_path,
        part_paths,
        total_bytes,
    })
}

pub fn apply_burnpack_part_bytes<M, B>(
    model: &mut M,
    burnpack_bytes: Vec<u8>,
) -> Result<ApplyResult, String>
where
    M: Module<B>,
    B: Backend,
{
    let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(burnpack_bytes)))
        .allow_partial(true)
        .validate(true);
    model
        .load_from(&mut store)
        .map_err(|err| format!("{err:?}"))
}

pub fn apply_burnpack_parts_bytes_with_progress<M, B, F>(
    model: &mut M,
    parts: &[Vec<u8>],
    mut progress: F,
) -> Result<ApplyResult, String>
where
    M: Module<B>,
    B: Backend,
    F: FnMut(usize, usize),
{
    let mut applied = BTreeSet::new();
    let total = parts.len();
    for (index, part) in parts.iter().enumerate() {
        progress(index + 1, total);
        let result = apply_burnpack_part_bytes(model, part.clone())?;
        for key in result.applied {
            applied.insert(key);
        }
    }
    Ok(ApplyResult {
        applied: applied.into_iter().collect(),
        skipped: Vec::new(),
        missing: Vec::new(),
        unused: Vec::new(),
        errors: Vec::new(),
    })
}

pub fn load_model_from_burnpack_part_bytes<M, B, Init>(
    parts: &[Vec<u8>],
    init_model: Init,
) -> Result<(M, ApplyResult), String>
where
    M: Module<B>,
    B: Backend,
    Init: FnOnce() -> M,
{
    load_model_from_burnpack_part_bytes_with_progress(parts, init_model, |_index, _total| {})
}

pub fn load_model_from_burnpack_part_bytes_with_progress<M, B, Init, F>(
    parts: &[Vec<u8>],
    init_model: Init,
    mut progress: F,
) -> Result<(M, ApplyResult), String>
where
    M: Module<B>,
    B: Backend,
    Init: FnOnce() -> M,
    F: FnMut(usize, usize),
{
    let mut model = init_model();
    let result = apply_burnpack_parts_bytes_with_progress(&mut model, parts, &mut progress)?;
    Ok((model, result))
}

pub fn try_load_model_from_burnpack_parts<M, B, Init>(
    burnpack_candidates: &[PathBuf],
    label: &str,
    verify_checksums: bool,
    init_model: Init,
) -> Result<Option<(M, ApplyResult)>, String>
where
    M: Module<B>,
    B: Backend,
    Init: FnMut() -> M,
{
    try_load_model_from_burnpack_parts_with_progress(
        burnpack_candidates,
        label,
        verify_checksums,
        init_model,
        |_| {},
    )
}

pub fn try_load_model_from_burnpack_parts_with_progress<M, B, Init, F>(
    burnpack_candidates: &[PathBuf],
    label: &str,
    verify_checksums: bool,
    mut init_model: Init,
    mut progress: F,
) -> Result<Option<(M, ApplyResult)>, String>
where
    M: Module<B>,
    B: Backend,
    Init: FnMut() -> M,
    F: FnMut(String),
{
    let any_candidate_exists = burnpack_candidates
        .iter()
        .any(|candidate| candidate.exists());

    for candidate in burnpack_candidates {
        if any_candidate_exists && !candidate.exists() {
            continue;
        }
        let manifest_path = burnpack_parts_manifest_path(candidate);
        if !manifest_path.exists() {
            continue;
        }
        let manifest = read_parts_manifest(&manifest_path)?;
        if candidate.exists() && !manifest_matches_source_file(&manifest, candidate) {
            continue;
        }
        if manifest.parts.is_empty() {
            return Err(format!(
                "burnpack parts manifest {} contains no parts for {label}",
                manifest_path.display()
            ));
        }

        let mut model = init_model();
        let mut applied = BTreeSet::new();
        let total_parts = manifest.parts.len();
        for (index, part) in manifest.parts.iter().enumerate() {
            let part_path = resolve_part_entry_path(&manifest_path, &part.path)?;
            progress(format!(
                "loading {label} part {}/{}",
                index + 1,
                total_parts
            ));
            let bytes = fs::read(&part_path).map_err(|err| {
                format!(
                    "failed to read {} part {}: {err}",
                    label,
                    part_path.display()
                )
            })?;
            if part.bytes > 0 && bytes.len() as u64 != part.bytes {
                return Err(format!(
                    "{label} part {} expected {} bytes but found {}",
                    part_path.display(),
                    part.bytes,
                    bytes.len()
                ));
            }
            if verify_checksums && !part.sha256.trim().is_empty() {
                let actual_sha = sha256_bytes(bytes.as_slice());
                if !actual_sha.eq_ignore_ascii_case(part.sha256.trim()) {
                    return Err(format!(
                        "{label} part {} checksum mismatch: expected {}, got {}",
                        part_path.display(),
                        part.sha256.trim(),
                        actual_sha
                    ));
                }
            }
            let apply_result = apply_burnpack_part_bytes(&mut model, bytes).map_err(|err| {
                format!(
                    "failed to apply {label} part {}/{} ({}): {err}",
                    index + 1,
                    manifest.parts.len(),
                    part_path.display()
                )
            })?;
            for key in apply_result.applied {
                applied.insert(key);
            }
        }
        progress(format!(
            "loaded {label} parts ({total_parts}/{total_parts})"
        ));

        return Ok(Some((
            model,
            ApplyResult {
                applied: applied.into_iter().collect(),
                skipped: Vec::new(),
                missing: Vec::new(),
                unused: Vec::new(),
                errors: Vec::new(),
            },
        )));
    }
    Ok(None)
}

pub fn load_model_from_burnpack_parts<M, B, Init>(
    burnpack_candidates: &[PathBuf],
    label: &str,
    verify_checksums: bool,
    init_model: Init,
) -> Result<(M, ApplyResult), String>
where
    M: Module<B>,
    B: Backend,
    Init: FnMut() -> M,
{
    load_model_from_burnpack_parts_with_progress(
        burnpack_candidates,
        label,
        verify_checksums,
        init_model,
        |_| {},
    )
}

pub fn load_model_from_burnpack_parts_with_progress<M, B, Init, F>(
    burnpack_candidates: &[PathBuf],
    label: &str,
    verify_checksums: bool,
    init_model: Init,
    progress: F,
) -> Result<(M, ApplyResult), String>
where
    M: Module<B>,
    B: Backend,
    Init: FnMut() -> M,
    F: FnMut(String),
{
    try_load_model_from_burnpack_parts_with_progress(
        burnpack_candidates,
        label,
        verify_checksums,
        init_model,
        progress,
    )?
    .ok_or_else(|| format!("no burnpack parts manifest found for {label}"))
}

pub fn load_model_from_burnpack_file<M, B, Init>(
    burnpack_path: &Path,
    init_model: Init,
) -> Result<(M, ApplyResult), String>
where
    M: Module<B>,
    B: Backend,
    Init: FnOnce() -> M,
{
    let mut model = init_model();
    let mut store = BurnpackStore::from_file(burnpack_path)
        .auto_extension(false)
        .validate(true);
    let result = model.load_from(&mut store).map_err(|err| {
        format!(
            "failed to load monolithic burnpack {}: {err:?}",
            burnpack_path.display()
        )
    })?;
    Ok((model, result))
}

pub fn try_load_model_from_burnpack_candidates<M, B, Init>(
    burnpack_candidates: &[PathBuf],
    label: &str,
    verify_checksums: bool,
    init_model: Init,
) -> Result<Option<(M, ApplyResult)>, String>
where
    M: Module<B>,
    B: Backend,
    Init: FnMut() -> M,
{
    try_load_model_from_burnpack_candidates_with_progress(
        burnpack_candidates,
        label,
        verify_checksums,
        init_model,
        |_| {},
    )
}

pub fn try_load_model_from_burnpack_candidates_with_progress<M, B, Init, F>(
    burnpack_candidates: &[PathBuf],
    label: &str,
    verify_checksums: bool,
    mut init_model: Init,
    mut progress: F,
) -> Result<Option<(M, ApplyResult)>, String>
where
    M: Module<B>,
    B: Backend,
    Init: FnMut() -> M,
    F: FnMut(String),
{
    if let Some((model, result)) = try_load_model_from_burnpack_parts_with_progress(
        burnpack_candidates,
        label,
        verify_checksums,
        &mut init_model,
        &mut progress,
    )? {
        return Ok(Some((model, result)));
    }

    let fallback = first_existing_candidate(burnpack_candidates)?;
    if !fallback.exists() {
        return Ok(None);
    }
    progress(format!("loading {label} monolithic burnpack"));
    load_model_from_burnpack_file(fallback.as_path(), init_model).map(Some)
}

pub fn load_model_from_burnpack_candidates<M, B, Init>(
    burnpack_candidates: &[PathBuf],
    label: &str,
    verify_checksums: bool,
    init_model: Init,
) -> Result<(M, ApplyResult), String>
where
    M: Module<B>,
    B: Backend,
    Init: FnMut() -> M,
{
    load_model_from_burnpack_candidates_with_progress(
        burnpack_candidates,
        label,
        verify_checksums,
        init_model,
        |_| {},
    )
}

pub fn load_model_from_burnpack_candidates_with_progress<M, B, Init, F>(
    burnpack_candidates: &[PathBuf],
    label: &str,
    verify_checksums: bool,
    init_model: Init,
    progress: F,
) -> Result<(M, ApplyResult), String>
where
    M: Module<B>,
    B: Backend,
    Init: FnMut() -> M,
    F: FnMut(String),
{
    try_load_model_from_burnpack_candidates_with_progress(
        burnpack_candidates,
        label,
        verify_checksums,
        init_model,
        progress,
    )?
    .ok_or_else(|| format!("no burnpack weights found for {label}"))
}

pub fn read_parts_manifest(path: &Path) -> Result<BurnpackPartsManifest, String> {
    let bytes = fs::read(path).map_err(|err| {
        format!(
            "failed to read burnpack parts manifest {}: {err}",
            path.display()
        )
    })?;
    serde_json::from_slice(&bytes).map_err(|err| {
        format!(
            "failed to parse burnpack parts manifest {}: {err}",
            path.display()
        )
    })
}

pub fn resolve_part_entry_path(manifest_path: &Path, entry_path: &str) -> Result<PathBuf, String> {
    let entry_path = Path::new(entry_path);
    if entry_path.is_absolute() {
        return Ok(entry_path.to_path_buf());
    }
    manifest_path
        .parent()
        .map(|parent| parent.join(entry_path))
        .ok_or_else(|| format!("invalid manifest path '{}'", manifest_path.display()))
}

pub fn manifest_is_complete(manifest_path: &Path) -> Result<bool, String> {
    if !manifest_path.exists() {
        return Ok(false);
    }
    let manifest = match read_parts_manifest(manifest_path) {
        Ok(manifest) => manifest,
        Err(_) => return Ok(false),
    };
    if manifest.parts.is_empty() {
        return Ok(false);
    }
    for part in &manifest.parts {
        let path = resolve_part_entry_path(manifest_path, &part.path)?;
        if !part_matches_cache(&path, part)? {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(crate) fn read_burnpack_metadata(
    source: &mut fs::File,
    burnpack_path: &Path,
) -> Result<(u16, u32, RawBurnpackMetadata), String> {
    source
        .seek(SeekFrom::Start(0))
        .map_err(|err| format!("failed to seek {}: {err}", burnpack_path.display()))?;
    let mut header = [0u8; BURNPACK_HEADER_SIZE];
    source.read_exact(&mut header).map_err(|err| {
        format!(
            "failed to read burnpack header {}: {err}",
            burnpack_path.display()
        )
    })?;

    let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    if magic != BURNPACK_MAGIC_NUMBER {
        return Err(format!(
            "invalid burnpack magic in {}: expected {BURNPACK_MAGIC_NUMBER:#x}, found {magic:#x}",
            burnpack_path.display()
        ));
    }
    let version = u16::from_le_bytes([header[4], header[5]]);
    let metadata_size = u32::from_le_bytes([header[6], header[7], header[8], header[9]]);
    let mut metadata_bytes = vec![0u8; metadata_size as usize];
    source.read_exact(&mut metadata_bytes).map_err(|err| {
        format!(
            "failed to read burnpack metadata {}: {err}",
            burnpack_path.display()
        )
    })?;
    let metadata = ciborium::de::from_reader(metadata_bytes.as_slice()).map_err(|err| {
        format!(
            "failed to parse burnpack metadata {}: {err}",
            burnpack_path.display()
        )
    })?;
    Ok((version, metadata_size, metadata))
}

pub(crate) fn write_burnpack_file(
    output_path: &Path,
    version: u16,
    metadata: &RawBurnpackMetadata,
    payloads: Vec<Vec<u8>>,
) -> Result<(), String> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create output directory {}: {err}",
                parent.display()
            )
        })?;
    }

    let mut metadata_bytes = Vec::new();
    ciborium::ser::into_writer(metadata, &mut metadata_bytes).map_err(|err| {
        format!(
            "failed to serialize burnpack metadata for {}: {err}",
            output_path.display()
        )
    })?;
    let metadata_size = u32::try_from(metadata_bytes.len()).map_err(|_| {
        format!(
            "burnpack metadata too large for {}: {} bytes",
            output_path.display(),
            metadata_bytes.len()
        )
    })?;

    let mut out = fs::File::create(output_path).map_err(|err| {
        format!(
            "failed to create output burnpack {}: {err}",
            output_path.display()
        )
    })?;
    let data_section_start = aligned_data_section_start(metadata_bytes.len());
    let mut written = 0usize;
    let mut header = [0u8; BURNPACK_HEADER_SIZE];
    header[0..4].copy_from_slice(&BURNPACK_MAGIC_NUMBER.to_le_bytes());
    header[4..6].copy_from_slice(&version.to_le_bytes());
    header[6..10].copy_from_slice(&metadata_size.to_le_bytes());
    out.write_all(&header).map_err(|err| {
        format!(
            "failed to write burnpack header {}: {err}",
            output_path.display()
        )
    })?;
    written += BURNPACK_HEADER_SIZE;
    out.write_all(metadata_bytes.as_slice()).map_err(|err| {
        format!(
            "failed to write burnpack metadata {}: {err}",
            output_path.display()
        )
    })?;
    written += metadata_bytes.len();
    if data_section_start > written {
        let padding = vec![0u8; data_section_start - written];
        out.write_all(padding.as_slice()).map_err(|err| {
            format!(
                "failed to write burnpack data padding {}: {err}",
                output_path.display()
            )
        })?;
        written = data_section_start;
    }

    let mut descriptors = metadata.tensors.iter().collect::<Vec<_>>();
    descriptors.sort_by_key(|(_, descriptor)| descriptor.data_offsets.0);
    if descriptors.len() != payloads.len() {
        return Err(format!(
            "descriptor/payload mismatch for {}: {} descriptors, {} payloads",
            output_path.display(),
            descriptors.len(),
            payloads.len()
        ));
    }

    for ((name, descriptor), bytes) in descriptors.into_iter().zip(payloads) {
        let expected_len = descriptor
            .data_offsets
            .1
            .saturating_sub(descriptor.data_offsets.0);
        if expected_len != bytes.len() as u64 {
            return Err(format!(
                "tensor `{name}` payload length mismatch for {}: expected {expected_len}, got {}",
                output_path.display(),
                bytes.len()
            ));
        }

        let target_offset = data_section_start
            .checked_add(descriptor.data_offsets.0 as usize)
            .ok_or_else(|| {
                format!(
                    "tensor `{name}` target offset overflow while writing {}",
                    output_path.display()
                )
            })?;
        if target_offset > written {
            let padding = vec![0u8; target_offset - written];
            out.write_all(padding.as_slice()).map_err(|err| {
                format!(
                    "failed to write tensor alignment padding {}: {err}",
                    output_path.display()
                )
            })?;
            written = target_offset;
        }
        out.write_all(bytes.as_slice()).map_err(|err| {
            format!(
                "failed to write burnpack tensor bytes {}: {err}",
                output_path.display()
            )
        })?;
        written = written.saturating_add(bytes.len());
    }
    out.flush().map_err(|err| {
        format!(
            "failed to flush output burnpack {}: {err}",
            output_path.display()
        )
    })
}

fn split_tensor_records(
    records: Vec<TensorRecord>,
    max_part_bytes: u64,
    source_metadata: &BTreeMap<String, String>,
) -> Vec<Vec<TensorRecord>> {
    let mut groups = Vec::new();
    let mut current_group = Vec::new();

    for record in records {
        let mut candidate_group = current_group.clone();
        candidate_group.push(record.clone());

        let candidate_bytes =
            estimate_part_total_bytes(candidate_group.as_slice(), source_metadata)
                .unwrap_or(u64::MAX);
        let would_exceed = !current_group.is_empty() && candidate_bytes > max_part_bytes;
        if would_exceed {
            groups.push(current_group);
            current_group = vec![record];
        } else {
            current_group = candidate_group;
        }
    }
    if !current_group.is_empty() {
        groups.push(current_group);
    }
    groups
}

fn estimate_part_total_bytes(
    records: &[TensorRecord],
    source_metadata: &BTreeMap<String, String>,
) -> Result<u64, String> {
    let mut tensors = BTreeMap::new();
    let mut payload_bytes = 0u64;
    for record in records {
        let tensor_bytes = record
            .descriptor
            .data_offsets
            .1
            .saturating_sub(record.descriptor.data_offsets.0);
        let mut descriptor = record.descriptor.clone();
        let aligned_start = align_offset(payload_bytes, TENSOR_ALIGNMENT);
        descriptor.data_offsets = (aligned_start, aligned_start.saturating_add(tensor_bytes));
        payload_bytes = descriptor.data_offsets.1;
        tensors.insert(record.name.clone(), descriptor);
    }

    let metadata = RawBurnpackMetadata {
        tensors,
        metadata: source_metadata.clone(),
    };
    let mut metadata_bytes = Vec::new();
    ciborium::ser::into_writer(&metadata, &mut metadata_bytes)
        .map_err(|err| format!("failed to estimate burnpack part metadata size: {err}"))?;
    Ok(aligned_data_section_start(metadata_bytes.len()) as u64 + payload_bytes)
}

fn write_burnpack_part(
    source: &mut fs::File,
    destination: &Path,
    version: u16,
    data_start: u64,
    source_metadata: &BTreeMap<String, String>,
    records: &[TensorRecord],
) -> Result<(), String> {
    let mut tensors = BTreeMap::new();
    let mut next_offset = 0u64;
    for record in records {
        let tensor_bytes = record
            .descriptor
            .data_offsets
            .1
            .saturating_sub(record.descriptor.data_offsets.0);
        let mut descriptor = record.descriptor.clone();
        let aligned_start = align_offset(next_offset, TENSOR_ALIGNMENT);
        descriptor.data_offsets = (aligned_start, aligned_start.saturating_add(tensor_bytes));
        next_offset = descriptor.data_offsets.1;
        tensors.insert(record.name.clone(), descriptor);
    }

    let metadata = RawBurnpackMetadata {
        tensors,
        metadata: source_metadata.clone(),
    };
    let mut metadata_bytes = Vec::new();
    ciborium::ser::into_writer(&metadata, &mut metadata_bytes)
        .map_err(|err| format!("failed to serialize burnpack part metadata: {err}"))?;
    let metadata_size = u32::try_from(metadata_bytes.len())
        .map_err(|_| "burnpack part metadata size exceeds u32".to_string())?;

    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let mut out = fs::File::create(destination).map_err(|err| {
        format!(
            "failed to create burnpack part {}: {err}",
            destination.display()
        )
    })?;
    let data_section_start = aligned_data_section_start(metadata_bytes.len());
    let mut written = 0usize;

    let mut header = [0u8; BURNPACK_HEADER_SIZE];
    header[0..4].copy_from_slice(&BURNPACK_MAGIC_NUMBER.to_le_bytes());
    header[4..6].copy_from_slice(&version.to_le_bytes());
    header[6..10].copy_from_slice(&metadata_size.to_le_bytes());
    out.write_all(&header).map_err(|err| {
        format!(
            "failed to write burnpack header {}: {err}",
            destination.display()
        )
    })?;
    written += BURNPACK_HEADER_SIZE;
    out.write_all(metadata_bytes.as_slice()).map_err(|err| {
        format!(
            "failed to write burnpack metadata {}: {err}",
            destination.display()
        )
    })?;
    written += metadata_bytes.len();
    if data_section_start > written {
        let padding = vec![0u8; data_section_start - written];
        out.write_all(padding.as_slice()).map_err(|err| {
            format!(
                "failed to write burnpack part data padding {}: {err}",
                destination.display()
            )
        })?;
        written = data_section_start;
    }

    let mut buffer = Vec::new();
    for record in records {
        let descriptor = metadata.tensors.get(&record.name).ok_or_else(|| {
            format!(
                "internal error: tensor `{}` missing from part metadata {}",
                record.name,
                destination.display()
            )
        })?;
        let target_offset = data_section_start
            .checked_add(descriptor.data_offsets.0 as usize)
            .ok_or_else(|| format!("tensor offset overflow in {}", destination.display()))?;
        if target_offset > written {
            let padding = vec![0u8; target_offset - written];
            out.write_all(padding.as_slice()).map_err(|err| {
                format!(
                    "failed to write burnpack part tensor alignment padding {}: {err}",
                    destination.display()
                )
            })?;
            written = target_offset;
        }
        let start = record.descriptor.data_offsets.0;
        let end = record.descriptor.data_offsets.1;
        let seek_offset = data_start
            .checked_add(start)
            .ok_or_else(|| format!("tensor offset overflow in {}", destination.display()))?;
        let len = end.saturating_sub(start);
        let len_usize = usize::try_from(len)
            .map_err(|_| format!("tensor byte length overflow in {}", destination.display()))?;
        buffer.resize(len_usize, 0);
        source.seek(SeekFrom::Start(seek_offset)).map_err(|err| {
            format!(
                "failed to seek source burnpack {}: {err}",
                destination.display()
            )
        })?;
        source.read_exact(buffer.as_mut_slice()).map_err(|err| {
            format!(
                "failed to read source tensor bytes {}: {err}",
                destination.display()
            )
        })?;
        out.write_all(buffer.as_slice()).map_err(|err| {
            format!(
                "failed to write tensor bytes {}: {err}",
                destination.display()
            )
        })?;
        written = written.saturating_add(buffer.len());
    }
    out.flush().map_err(|err| {
        format!(
            "failed to flush burnpack part {}: {err}",
            destination.display()
        )
    })
}

pub(crate) fn part_matches_cache(path: &Path, part: &BurnpackPartEntry) -> Result<bool, String> {
    if !path.exists() {
        return Ok(false);
    }
    let bytes = fs::metadata(path)
        .map_err(|err| {
            format!(
                "failed to stat cached burnpack part {}: {err}",
                path.display()
            )
        })?
        .len();
    if part.bytes > 0 && bytes != part.bytes {
        return Ok(false);
    }
    if !part.sha256.trim().is_empty() {
        let actual_sha = sha256_file(path)?;
        if !actual_sha.eq_ignore_ascii_case(part.sha256.trim()) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn manifest_has_all_parts(path: &Path, source_burnpack_path: Option<&Path>) -> bool {
    let Ok(manifest) = read_parts_manifest(path) else {
        return false;
    };
    if manifest.parts.is_empty() {
        return false;
    }
    if let Some(source_burnpack_path) = source_burnpack_path
        && source_burnpack_path.exists()
        && !manifest_matches_source_file(&manifest, source_burnpack_path)
    {
        return false;
    }

    manifest.parts.iter().all(|entry| {
        resolve_part_entry_path(path, &entry.path)
            .ok()
            .and_then(|part_path| part_matches_cache(&part_path, entry).ok())
            .unwrap_or(false)
    })
}

fn manifest_matches_source_file(manifest: &BurnpackPartsManifest, source_path: &Path) -> bool {
    let source_file_name = source_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if !manifest.source_file.is_empty() && manifest.source_file != source_file_name {
        return false;
    }

    let Ok(actual_bytes) = fs::metadata(source_path).map(|meta| meta.len()) else {
        return false;
    };
    if manifest.total_bytes > 0 && manifest.total_bytes != actual_bytes {
        return false;
    }

    if manifest.source_modified_unix_ms == 0 {
        return true;
    }
    let Some(actual_modified_unix_ms) = file_modified_unix_ms(source_path) else {
        return false;
    };
    manifest.source_modified_unix_ms == actual_modified_unix_ms
}

fn first_existing_candidate(candidates: &[PathBuf]) -> Result<&PathBuf, String> {
    candidates
        .iter()
        .find(|candidate| candidate.exists())
        .or_else(|| candidates.first())
        .ok_or_else(|| "no burnpack candidate paths supplied".to_string())
}

fn cleanup_existing_parts(manifest_path: &Path) -> Result<(), String> {
    let manifest = match read_parts_manifest(manifest_path) {
        Ok(manifest) => manifest,
        Err(_) => return Ok(()),
    };
    for entry in &manifest.parts {
        let path = resolve_part_entry_path(manifest_path, &entry.path)?;
        if path.exists() {
            fs::remove_file(&path).map_err(|err| {
                format!(
                    "failed to remove stale burnpack part {}: {err}",
                    path.display()
                )
            })?;
        }
    }
    if manifest_path.exists() {
        fs::remove_file(manifest_path).map_err(|err| {
            format!(
                "failed to remove stale burnpack manifest {}: {err}",
                manifest_path.display()
            )
        })?;
    }
    Ok(())
}

fn ensure_parent_dir(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    Ok(())
}

fn file_modified_unix_ms(path: &Path) -> Option<u64> {
    let modified = fs::metadata(path).ok()?.modified().ok()?;
    let duration = modified.duration_since(UNIX_EPOCH).ok()?;
    Some(duration.as_millis() as u64)
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path)
        .map_err(|err| format!("failed to read {} for checksum: {err}", path.display()))?;
    Ok(sha256_bytes(bytes.as_slice()))
}

pub(crate) fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn normalize_extension(path: &Path, extension: &str) -> PathBuf {
    if path
        .extension()
        .map(|ext| ext.eq_ignore_ascii_case(extension))
        .unwrap_or(false)
    {
        path.to_path_buf()
    } else {
        path.with_extension(extension)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BurnpackPartEntry, BurnpackPartsManifest, apply_burnpack_parts_bytes_with_progress,
        burnpack_parts_manifest_path, load_model_from_burnpack_candidates,
        load_model_from_burnpack_part_bytes, manifest_is_complete, read_parts_manifest,
        save_model_to_burnpack, write_burnpack_parts,
    };
    use std::fs;

    use burn::module::Module;
    use burn::nn::{Linear, LinearConfig};
    use burn::tensor::{Tensor, backend::Backend};
    use burn_ndarray::NdArray;
    use tempfile::tempdir;

    type TestBackend = NdArray<f32>;

    #[derive(Module, Debug)]
    struct TinyModel<B: Backend> {
        linear: Linear<B>,
    }

    impl<B: Backend> TinyModel<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                linear: LinearConfig::new(4, 3).init(device),
            }
        }

        fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
            self.linear.forward(x)
        }
    }

    #[test]
    fn manifest_complete_requires_all_parts() {
        let dir = tempdir().expect("tempdir");
        let burnpack = dir.path().join("tiny.bpk");
        let manifest_path = burnpack_parts_manifest_path(&burnpack);
        let manifest = BurnpackPartsManifest {
            version: 1,
            source_file: "tiny.bpk".to_string(),
            source_modified_unix_ms: 0,
            total_bytes: 16,
            max_part_bytes: 8,
            parts: vec![BurnpackPartEntry {
                path: "tiny.bpk.part-00000.bpk".to_string(),
                bytes: 4,
                sha256: String::new(),
                tensors: 1,
            }],
        };
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
        fs::write(dir.path().join("tiny.bpk.part-00000.bpk"), [1u8, 2, 3, 4]).expect("write part");
        assert!(manifest_is_complete(&manifest_path).expect("complete"));
        fs::remove_file(dir.path().join("tiny.bpk.part-00000.bpk")).expect("remove part");
        assert!(!manifest_is_complete(&manifest_path).expect("incomplete"));
    }

    #[test]
    fn loads_model_from_parts_manifest() {
        let device = <TestBackend as Backend>::Device::default();
        TestBackend::seed(&device, 1337);
        let model = TinyModel::<TestBackend>::new(&device);
        let dir = tempdir().expect("tempdir");
        let burnpack = dir.path().join("tiny.bpk");
        let saved = save_model_to_burnpack(&model, &burnpack).expect("save burnpack");
        let report = write_burnpack_parts(&saved, 1, true).expect("write parts");
        let manifest = read_parts_manifest(&report.manifest_path).expect("read manifest");
        assert!(!manifest.parts.is_empty(), "expected at least one part");

        let (loaded, _result) = super::load_model_from_burnpack_parts(
            std::slice::from_ref(&saved),
            "tiny model",
            true,
            || TinyModel::<TestBackend>::new(&device),
        )
        .expect("load from parts");

        let input = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let reference = model
            .forward(input.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let restored = loaded.forward(input).into_data().to_vec::<f32>().unwrap();
        assert_eq!(reference, restored);
    }

    #[test]
    fn load_model_from_part_bytes_matches_sequential_apply() {
        let device = <TestBackend as Backend>::Device::default();
        TestBackend::seed(&device, 1337);
        let model = TinyModel::<TestBackend>::new(&device);
        let dir = tempdir().expect("tempdir");
        let burnpack = dir.path().join("tiny.bpk");
        let saved = save_model_to_burnpack(&model, &burnpack).expect("save burnpack");
        let report = write_burnpack_parts(&saved, 1, true).expect("write parts");
        let parts = report
            .part_paths
            .iter()
            .map(|path| fs::read(path).expect("read part"))
            .collect::<Vec<_>>();

        let (loaded, loaded_result) =
            load_model_from_burnpack_part_bytes(&parts, || TinyModel::<TestBackend>::new(&device))
                .expect("load from bytes");
        let mut manual = TinyModel::<TestBackend>::new(&device);
        let manual_result =
            apply_burnpack_parts_bytes_with_progress(&mut manual, &parts, |_i, _n| {})
                .expect("manual apply");
        assert_eq!(loaded_result.applied, manual_result.applied);

        let input = Tensor::<TestBackend, 2>::from_floats([[0.5, -1.0, 0.25, 2.0]], &device);
        let from_loaded = loaded
            .forward(input.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let from_manual = manual.forward(input).into_data().to_vec::<f32>().unwrap();
        assert_eq!(from_loaded, from_manual);
    }

    #[test]
    fn load_model_from_candidates_falls_back_to_monolithic_burnpack() {
        let device = <TestBackend as Backend>::Device::default();
        TestBackend::seed(&device, 1337);
        let model = TinyModel::<TestBackend>::new(&device);
        let dir = tempdir().expect("tempdir");
        let burnpack = dir.path().join("tiny.bpk");
        let saved = save_model_to_burnpack(&model, &burnpack).expect("save burnpack");

        let (loaded, _result) = load_model_from_burnpack_candidates(
            std::slice::from_ref(&saved),
            "tiny model",
            true,
            || TinyModel::<TestBackend>::new(&device),
        )
        .expect("load monolithic burnpack");

        let input = Tensor::<TestBackend, 2>::from_floats([[1.0, 0.0, -1.0, 2.0]], &device);
        let reference = model
            .forward(input.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let restored = loaded.forward(input).into_data().to_vec::<f32>().unwrap();
        assert_eq!(reference, restored);
    }
}
