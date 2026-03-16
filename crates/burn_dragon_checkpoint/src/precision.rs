use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use burn::tensor::{DType, TensorData};

use crate::parts::{BURNPACK_HEADER_SIZE, read_burnpack_metadata, write_burnpack_file};
use crate::policy::{BurnpackLoadPolicy, burnpack_path};

const TENSOR_ALIGNMENT: u64 = 256;

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

/// Float precision used when materializing burnpack deployment weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BurnpackFloatPrecision {
    F16,
    F32,
}

impl BurnpackFloatPrecision {
    const fn dtype(self) -> DType {
        match self {
            Self::F16 => DType::F16,
            Self::F32 => DType::F32,
        }
    }
}

pub fn convert_burnpack_precision(
    source_burnpack: &Path,
    output_base: &Path,
    precision: BurnpackFloatPrecision,
    policy: BurnpackLoadPolicy,
) -> Result<PathBuf, String> {
    let output = burnpack_path(
        output_base,
        matches!(precision, BurnpackFloatPrecision::F16),
        policy.f16_suffix,
    );
    convert_burnpack_float_precision(source_burnpack, output.as_path(), precision.dtype())?;
    Ok(output)
}

pub fn dtype_precision_label(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => "f16",
        DType::F32 => "f32",
        _ => "mixed",
    }
}

fn convert_burnpack_float_precision(
    source_burnpack: &Path,
    output_burnpack: &Path,
    target_dtype: DType,
) -> Result<(), String> {
    let mut source = fs::File::open(source_burnpack).map_err(|err| {
        format!(
            "failed to open source burnpack {}: {err}",
            source_burnpack.display()
        )
    })?;
    let (version, metadata_size, mut metadata) =
        read_burnpack_metadata(&mut source, source_burnpack)?;
    let data_start = aligned_data_section_start(metadata_size as usize) as u64;

    let mut descriptors = metadata.tensors.into_iter().collect::<Vec<_>>();
    descriptors.sort_by_key(|(_, descriptor)| descriptor.data_offsets.0);
    let mut converted_descriptors = std::collections::BTreeMap::new();
    let mut converted_payloads = Vec::with_capacity(descriptors.len());
    let mut next_offset = 0u64;

    for (name, mut descriptor) in descriptors {
        let source_bytes = read_tensor_payload(
            &mut source,
            data_start,
            descriptor.data_offsets,
            source_burnpack,
            &name,
        )?;
        let converted_dtype = convert_float_dtype(parse_dtype(&descriptor.dtype)?, target_dtype);
        let converted_bytes = if converted_dtype == parse_dtype(&descriptor.dtype)? {
            source_bytes
        } else {
            let shape = descriptor
                .shape
                .iter()
                .map(|&dim| {
                    usize::try_from(dim)
                        .map_err(|_| format!("tensor `{name}` shape dimension overflow: {dim}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let source_data =
                TensorData::from_bytes_vec(source_bytes, shape, parse_dtype(&descriptor.dtype)?);
            let converted = convert_tensor_data_float_precision(source_data, target_dtype);
            converted.bytes.to_vec()
        };

        let tensor_len = u64::try_from(converted_bytes.len()).map_err(|_| {
            format!(
                "tensor `{name}` byte length overflow: {}",
                converted_bytes.len()
            )
        })?;
        let aligned_start = align_offset(next_offset, TENSOR_ALIGNMENT);
        let end_offset = aligned_start.checked_add(tensor_len).ok_or_else(|| {
            format!("tensor `{name}` data offset overflow: {aligned_start} + {tensor_len}")
        })?;
        descriptor.dtype = serialize_dtype(converted_dtype)?;
        descriptor.data_offsets = (aligned_start, end_offset);
        next_offset = end_offset;
        converted_descriptors.insert(name, descriptor);
        converted_payloads.push(converted_bytes);
    }

    metadata.tensors = converted_descriptors;
    metadata.metadata.insert(
        "precision".to_string(),
        dtype_precision_label(target_dtype).to_string(),
    );
    write_burnpack_file(output_burnpack, version, &metadata, converted_payloads)
}

fn convert_tensor_data_float_precision(data: TensorData, target_dtype: DType) -> TensorData {
    match target_dtype {
        DType::F16 => match data.dtype {
            DType::F64 | DType::F32 | DType::BF16 | DType::Flex32 => data.convert_dtype(DType::F16),
            _ => data,
        },
        DType::F32 => match data.dtype {
            DType::F16 | DType::BF16 | DType::F64 | DType::Flex32 => data.convert_dtype(DType::F32),
            _ => data,
        },
        _ => data,
    }
}

fn convert_float_dtype(source: DType, target_dtype: DType) -> DType {
    match target_dtype {
        DType::F16 => {
            if source.is_float() {
                DType::F16
            } else {
                source
            }
        }
        DType::F32 => {
            if source.is_float() {
                DType::F32
            } else {
                source
            }
        }
        _ => source,
    }
}

fn read_tensor_payload(
    source: &mut fs::File,
    data_start: u64,
    data_offsets: (u64, u64),
    source_path: &Path,
    tensor_name: &str,
) -> Result<Vec<u8>, String> {
    let (start, end) = data_offsets;
    if end < start {
        return Err(format!(
            "tensor `{tensor_name}` has invalid data offsets ({start}, {end}) in {}",
            source_path.display()
        ));
    }
    let len = end - start;
    let len_usize = usize::try_from(len)
        .map_err(|_| format!("tensor `{tensor_name}` byte length overflow: {len}"))?;
    let seek_offset = data_start.checked_add(start).ok_or_else(|| {
        format!("tensor `{tensor_name}` data offset overflow: {data_start} + {start}")
    })?;
    source.seek(SeekFrom::Start(seek_offset)).map_err(|err| {
        format!(
            "failed to seek tensor `{tensor_name}` in {}: {err}",
            source_path.display()
        )
    })?;
    let mut bytes = vec![0u8; len_usize];
    source.read_exact(&mut bytes).map_err(|err| {
        format!(
            "failed to read tensor `{tensor_name}` bytes from {}: {err}",
            source_path.display()
        )
    })?;
    Ok(bytes)
}

fn parse_dtype(value: &ciborium::Value) -> Result<DType, String> {
    let mut bytes = Vec::new();
    ciborium::ser::into_writer(value, &mut bytes)
        .map_err(|err| format!("failed to encode burnpack dtype: {err}"))?;
    ciborium::de::from_reader(bytes.as_slice())
        .map_err(|err| format!("failed to parse burnpack dtype: {err}"))
}

fn serialize_dtype(dtype: DType) -> Result<ciborium::Value, String> {
    ciborium::value::Value::serialized(&dtype)
        .map_err(|err| format!("failed to serialize dtype {dtype:?}: {err}"))
}

#[cfg(test)]
mod tests {
    use super::{BurnpackFloatPrecision, convert_burnpack_precision};
    use crate::parts::{read_burnpack_metadata, save_model_to_burnpack};
    use crate::policy::BurnpackLoadPolicy;
    use burn::module::Module;
    use burn::nn::{Linear, LinearConfig};
    use burn::tensor::{DType, backend::Backend};
    use burn_ndarray::NdArray;
    use std::fs;
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
    }

    #[test]
    fn converts_burnpack_to_f16_metadata() {
        let device = <TestBackend as Backend>::Device::default();
        TestBackend::seed(&device, 1337);
        let model = TinyModel::<TestBackend>::new(&device);
        let dir = tempdir().expect("tempdir");
        let source =
            save_model_to_burnpack(&model, &dir.path().join("tiny")).expect("save f32 burnpack");
        let output = convert_burnpack_precision(
            source.as_path(),
            &dir.path().join("tiny"),
            BurnpackFloatPrecision::F16,
            BurnpackLoadPolicy::default()
                .with_precision(crate::policy::BurnpackPrecisionPreference::PreferF16),
        )
        .expect("convert f16");

        let mut file = fs::File::open(&output).expect("open converted burnpack");
        let (_version, _metadata_size, metadata) =
            read_burnpack_metadata(&mut file, &output).expect("read metadata");

        assert_eq!(
            metadata.metadata.get("precision").map(String::as_str),
            Some("f16")
        );

        let dtypes = metadata
            .tensors
            .values()
            .map(|descriptor| {
                let mut bytes = Vec::new();
                ciborium::ser::into_writer(&descriptor.dtype, &mut bytes).expect("encode dtype");
                ciborium::de::from_reader(bytes.as_slice()).expect("decode dtype")
            })
            .collect::<Vec<DType>>();
        assert!(!dtypes.is_empty());
        assert!(dtypes.iter().all(|dtype| *dtype == DType::F16));
    }
}
