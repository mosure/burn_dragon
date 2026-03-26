use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

const SCALE_EPSILON: f32 = 1.0e-8;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BitNetReferenceWeightMode {
    Binary,
    #[default]
    Ternary158,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BitNetReferenceActivationMode {
    #[default]
    Int8,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct BitLinearReferenceSpec {
    pub in_features: usize,
    pub out_features: usize,
    #[serde(default)]
    pub weight_mode: BitNetReferenceWeightMode,
    #[serde(default)]
    pub activation_mode: BitNetReferenceActivationMode,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedBuffer<T> {
    pub values: Vec<T>,
    pub scale: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedTernaryBuffer {
    pub packed: Vec<u8>,
    pub len: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PackedWeightEncoding {
    Binary1,
    Ternary2,
    Int8,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct PackedWeightArtifact {
    pub encoding: PackedWeightEncoding,
    pub logical_shape: Vec<usize>,
    pub scale: f32,
    pub packed: Vec<u8>,
    pub len: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default)]
pub struct BdhBitNetStaticArtifacts {
    pub decoder_x: Option<PackedWeightArtifact>,
    pub decoder_y: Option<PackedWeightArtifact>,
    pub encoder: Option<PackedWeightArtifact>,
}

pub fn ste_passthrough<B: Backend, const D: usize>(
    original: Tensor<B, D>,
    quantized: Tensor<B, D>,
) -> Tensor<B, D> {
    original.clone() + (quantized - original).detach()
}

pub fn quantize_binary_sign(weights: &[f32]) -> QuantizedBuffer<i8> {
    let scale = abs_mean(weights).max(SCALE_EPSILON);
    let values = weights
        .iter()
        .map(|value| if *value >= 0.0 { 1 } else { -1 })
        .collect::<Vec<_>>();
    QuantizedBuffer { values, scale }
}

pub fn quantize_ternary_absmean(weights: &[f32]) -> QuantizedBuffer<i8> {
    let scale = abs_mean(weights).max(SCALE_EPSILON);
    let values = weights
        .iter()
        .map(|value| {
            if value.abs() < scale {
                0
            } else if *value >= 0.0 {
                1
            } else {
                -1
            }
        })
        .collect::<Vec<_>>();
    QuantizedBuffer { values, scale }
}

pub fn dequantize_weight_codes(buffer: &QuantizedBuffer<i8>) -> Vec<f32> {
    buffer
        .values
        .iter()
        .map(|value| *value as f32 * buffer.scale)
        .collect::<Vec<_>>()
}

pub fn quantize_activation_symmetric_i8(values: &[f32]) -> QuantizedBuffer<i8> {
    let max_abs = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let scale = (max_abs / 127.0).max(SCALE_EPSILON);
    let quantized = values
        .iter()
        .map(|value| ((value / scale).round()).clamp(-127.0, 127.0) as i8)
        .collect::<Vec<_>>();
    QuantizedBuffer {
        values: quantized,
        scale,
    }
}

pub fn quantize_weight_symmetric_i8(values: &[f32]) -> QuantizedBuffer<i8> {
    let mean_abs = if values.is_empty() {
        0.0
    } else {
        values.iter().map(|value| value.abs()).sum::<f32>() / values.len() as f32
    };
    let dynamic_range = (mean_abs * 2.0).max(SCALE_EPSILON);
    let scale = (dynamic_range / 127.0).max(SCALE_EPSILON);
    let quantized = values
        .iter()
        .map(|value| ((value / scale).round()).clamp(-127.0, 127.0) as i8)
        .collect::<Vec<_>>();
    QuantizedBuffer {
        values: quantized,
        scale,
    }
}

pub fn dequantize_activation_i8(buffer: &QuantizedBuffer<i8>) -> Vec<f32> {
    buffer
        .values
        .iter()
        .map(|value| *value as f32 * buffer.scale)
        .collect::<Vec<_>>()
}

pub fn pack_ternary_2bit(values: &[i8]) -> PackedTernaryBuffer {
    let mut packed = Vec::with_capacity(values.len().div_ceil(4));
    let mut current = 0u8;

    for (index, value) in values.iter().enumerate() {
        let encoded = match *value {
            -1 => 0u8,
            0 => 1u8,
            1 => 2u8,
            other => panic!("ternary packing expects values in {{-1, 0, 1}}, got {other}"),
        };
        let shift = (index % 4) * 2;
        current |= encoded << shift;
        if index % 4 == 3 {
            packed.push(current);
            current = 0;
        }
    }

    if values.len() % 4 != 0 {
        packed.push(current);
    }

    PackedTernaryBuffer {
        packed,
        len: values.len(),
    }
}

pub fn pack_binary_1bit(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(values.len().div_ceil(8));
    let mut current = 0u8;

    for (index, value) in values.iter().enumerate() {
        let encoded = match *value {
            -1 => 0u8,
            1 => 1u8,
            other => panic!("binary packing expects values in {{-1, 1}}, got {other}"),
        };
        current |= encoded << (index % 8);
        if index % 8 == 7 {
            packed.push(current);
            current = 0;
        }
    }

    if values.len() % 8 != 0 {
        packed.push(current);
    }

    packed
}

pub fn unpack_binary_1bit(packed: &[u8], len: usize) -> Vec<i8> {
    let mut values = Vec::with_capacity(len);
    for byte in packed {
        for shift in 0..8 {
            if values.len() == len {
                break;
            }
            let bit = (byte >> shift) & 1;
            values.push(if bit == 0 { -1 } else { 1 });
        }
    }
    values
}

pub fn pack_weight_artifact_from_format(
    weights: &[f32],
    logical_shape: &[usize],
    format: crate::LowBitWeightFormat,
) -> Option<PackedWeightArtifact> {
    match format {
        crate::LowBitWeightFormat::Fp16 => None,
        crate::LowBitWeightFormat::Int8 => {
            let quantized = quantize_weight_symmetric_i8(weights);
            Some(PackedWeightArtifact {
                encoding: PackedWeightEncoding::Int8,
                logical_shape: logical_shape.to_vec(),
                scale: quantized.scale,
                packed: quantized.values.iter().map(|value| *value as u8).collect(),
                len: quantized.values.len(),
            })
        }
        crate::LowBitWeightFormat::Sign1 => {
            let quantized = quantize_binary_sign(weights);
            Some(PackedWeightArtifact {
                encoding: PackedWeightEncoding::Binary1,
                logical_shape: logical_shape.to_vec(),
                scale: quantized.scale,
                packed: pack_binary_1bit(&quantized.values),
                len: quantized.values.len(),
            })
        }
        crate::LowBitWeightFormat::Ternary158 | crate::LowBitWeightFormat::Packed2 => {
            let quantized = quantize_ternary_absmean(weights);
            let packed = pack_ternary_2bit(&quantized.values);
            Some(PackedWeightArtifact {
                encoding: PackedWeightEncoding::Ternary2,
                logical_shape: logical_shape.to_vec(),
                scale: quantized.scale,
                packed: packed.packed,
                len: packed.len,
            })
        }
    }
}

pub fn pack_weight_artifact_from_dequantized_values(
    values: &[f32],
    logical_shape: &[usize],
    format: crate::LowBitWeightFormat,
) -> Option<PackedWeightArtifact> {
    match format {
        crate::LowBitWeightFormat::Fp16 => None,
        crate::LowBitWeightFormat::Int8 => {
            pack_weight_artifact_from_format(values, logical_shape, format)
        }
        crate::LowBitWeightFormat::Sign1 => {
            let scale = values
                .iter()
                .map(|value| value.abs())
                .fold(0.0f32, f32::max)
                .max(SCALE_EPSILON);
            let quantized = values
                .iter()
                .map(|value| if *value >= 0.0 { 1 } else { -1 })
                .collect::<Vec<_>>();
            Some(PackedWeightArtifact {
                encoding: PackedWeightEncoding::Binary1,
                logical_shape: logical_shape.to_vec(),
                scale,
                packed: pack_binary_1bit(&quantized),
                len: quantized.len(),
            })
        }
        crate::LowBitWeightFormat::Ternary158 | crate::LowBitWeightFormat::Packed2 => {
            let scale = values
                .iter()
                .map(|value| value.abs())
                .fold(0.0f32, f32::max)
                .max(SCALE_EPSILON);
            let quantized = values
                .iter()
                .map(|value| {
                    if value.abs() <= scale * 0.5 {
                        0
                    } else if *value >= 0.0 {
                        1
                    } else {
                        -1
                    }
                })
                .collect::<Vec<_>>();
            let packed = pack_ternary_2bit(&quantized);
            Some(PackedWeightArtifact {
                encoding: PackedWeightEncoding::Ternary2,
                logical_shape: logical_shape.to_vec(),
                scale,
                packed: packed.packed,
                len: packed.len,
            })
        }
    }
}

pub fn unpack_weight_artifact_to_f32(artifact: &PackedWeightArtifact) -> Vec<f32> {
    match artifact.encoding {
        PackedWeightEncoding::Int8 => artifact
            .packed
            .iter()
            .take(artifact.len)
            .map(|value| (*value as i8) as f32 * artifact.scale)
            .collect(),
        PackedWeightEncoding::Binary1 => unpack_binary_1bit(&artifact.packed, artifact.len)
            .into_iter()
            .map(|value| value as f32 * artifact.scale)
            .collect(),
        PackedWeightEncoding::Ternary2 => unpack_ternary_2bit(&PackedTernaryBuffer {
            packed: artifact.packed.clone(),
            len: artifact.len,
        })
        .into_iter()
        .map(|value| value as f32 * artifact.scale)
        .collect(),
    }
}

pub fn unpack_weight_artifact_to_i8_codes(artifact: &PackedWeightArtifact) -> Vec<i8> {
    match artifact.encoding {
        PackedWeightEncoding::Int8 => artifact
            .packed
            .iter()
            .take(artifact.len)
            .map(|value| *value as i8)
            .collect(),
        PackedWeightEncoding::Binary1 => unpack_binary_1bit(&artifact.packed, artifact.len),
        PackedWeightEncoding::Ternary2 => unpack_ternary_2bit(&PackedTernaryBuffer {
            packed: artifact.packed.clone(),
            len: artifact.len,
        }),
    }
}

pub fn unpack_ternary_2bit(buffer: &PackedTernaryBuffer) -> Vec<i8> {
    let mut values = Vec::with_capacity(buffer.len);
    for byte in &buffer.packed {
        for shift in [0, 2, 4, 6] {
            if values.len() == buffer.len {
                break;
            }
            let code = (byte >> shift) & 0b11;
            let value = match code {
                0 => -1,
                1 => 0,
                2 => 1,
                other => panic!("invalid packed ternary code {other}"),
            };
            values.push(value);
        }
    }
    values
}

pub fn bitlinear_reference_forward(
    spec: &BitLinearReferenceSpec,
    input: &[f32],
    master_weights: &[f32],
) -> Vec<f32> {
    assert_eq!(
        input.len(),
        spec.in_features,
        "input length must match spec.in_features"
    );
    assert_eq!(
        master_weights.len(),
        spec.in_features * spec.out_features,
        "weight length must match out_features * in_features"
    );

    let dequant_input = match spec.activation_mode {
        BitNetReferenceActivationMode::Int8 => {
            dequantize_activation_i8(&quantize_activation_symmetric_i8(input))
        }
    };
    let dequant_weight = match spec.weight_mode {
        BitNetReferenceWeightMode::Binary => {
            dequantize_weight_codes(&quantize_binary_sign(master_weights))
        }
        BitNetReferenceWeightMode::Ternary158 => {
            dequantize_weight_codes(&quantize_ternary_absmean(master_weights))
        }
    };

    let mut output = vec![0.0f32; spec.out_features];
    for out_idx in 0..spec.out_features {
        let row = &dequant_weight[out_idx * spec.in_features..(out_idx + 1) * spec.in_features];
        output[out_idx] = row
            .iter()
            .zip(dequant_input.iter())
            .map(|(weight, value)| weight * value)
            .sum::<f32>();
    }
    output
}

fn abs_mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|value| value.abs()).sum::<f32>() / values.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;
    type AutodiffBackend = Autodiff<NdArray<f32>>;

    #[test]
    fn ternary_pack_round_trip_is_lossless() {
        let values = vec![-1, 0, 1, -1, 1, 0, 0];
        let packed = pack_ternary_2bit(&values);
        assert_eq!(unpack_ternary_2bit(&packed), values);
    }

    #[test]
    fn binary_pack_round_trip_is_lossless() {
        let values = vec![-1, 1, -1, -1, 1, 1, -1, 1, 1];
        let packed = pack_binary_1bit(&values);
        assert_eq!(unpack_binary_1bit(&packed, values.len()), values);
    }

    #[test]
    fn binary_quantization_uses_only_signed_codes() {
        let buffer = quantize_binary_sign(&[-1.5, 0.0, 0.2, 3.4]);
        assert!(buffer.values.iter().all(|value| matches!(value, -1 | 1)));
        assert!(buffer.scale.is_finite());
        assert!(buffer.scale > 0.0);
    }

    #[test]
    fn activation_quantization_returns_finite_scale() {
        let buffer = quantize_activation_symmetric_i8(&[-2.0, -0.25, 0.0, 0.5, 4.0]);
        assert_eq!(buffer.values.len(), 5);
        assert!(buffer.scale.is_finite());
        assert!(buffer.scale > 0.0);
    }

    #[test]
    fn bitlinear_reference_forward_respects_spec_shapes() {
        let spec = BitLinearReferenceSpec {
            in_features: 3,
            out_features: 2,
            weight_mode: BitNetReferenceWeightMode::Ternary158,
            activation_mode: BitNetReferenceActivationMode::Int8,
        };
        let output = bitlinear_reference_forward(
            &spec,
            &[1.0, -0.5, 2.0],
            &[
                0.4, -0.8, 1.2, //
                -0.1, 0.7, -1.4,
            ],
        );
        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn ste_passthrough_preserves_quantized_forward_value() {
        let device = Default::default();
        let original =
            Tensor::<TestBackend, 1>::from_data(TensorData::new(vec![1.0, -2.0], [2]), &device);
        let quantized =
            Tensor::<TestBackend, 1>::from_data(TensorData::new(vec![0.5, -1.5], [2]), &device);
        let output = ste_passthrough(original, quantized.clone());
        output.into_data().assert_eq(&quantized.into_data(), false);
    }

    #[test]
    fn ste_passthrough_routes_gradients_to_original_tensor() {
        let device = Default::default();
        let original =
            Tensor::<AutodiffBackend, 1>::from_data(TensorData::new(vec![1.0, -2.0], [2]), &device)
                .require_grad();
        let quantized =
            Tensor::<AutodiffBackend, 1>::from_data(TensorData::new(vec![0.5, -1.5], [2]), &device);
        let output = ste_passthrough(original.clone(), quantized);
        let grads = output.sum().backward();
        let grad = original.grad(&grads).expect("gradient");
        grad.into_data()
            .assert_eq(&TensorData::new(vec![1.0, 1.0], [2]), false);
    }

    #[test]
    fn packed_weight_artifact_preserves_binary_shape_metadata() {
        let artifact = pack_weight_artifact_from_format(
            &[0.1, -0.4, 0.9, -1.2],
            &[2, 2],
            crate::LowBitWeightFormat::Sign1,
        )
        .expect("artifact");
        assert_eq!(artifact.encoding, PackedWeightEncoding::Binary1);
        assert_eq!(artifact.logical_shape, vec![2, 2]);
        assert_eq!(artifact.len, 4);
    }

    #[test]
    fn ternary_weight_artifact_round_trip_matches_reference_dequantization() {
        let master = vec![0.05, -0.9, 0.0, 1.7, -0.2, 0.8];
        let artifact = pack_weight_artifact_from_format(
            &master,
            &[2, 3],
            crate::LowBitWeightFormat::Ternary158,
        )
        .expect("artifact");
        let dequantized = unpack_weight_artifact_to_f32(&artifact);
        let reference = dequantize_weight_codes(&quantize_ternary_absmean(&master));
        assert_eq!(dequantized, reference);
    }

    #[test]
    fn int8_weight_artifact_round_trip_matches_reference_dequantization() {
        let master = vec![0.3, -0.2, 0.0, 1.7, -1.2, 0.8];
        let artifact =
            pack_weight_artifact_from_format(&master, &[2, 3], crate::LowBitWeightFormat::Int8)
                .expect("artifact");
        let dequantized = unpack_weight_artifact_to_f32(&artifact);
        let reference = dequantize_activation_i8(&quantize_weight_symmetric_i8(&master));
        assert_eq!(dequantized, reference);
    }

    #[test]
    fn ternary_weight_artifact_round_trip_matches_reference_codes() {
        let master = vec![0.05, -0.9, 0.0, 1.7, -0.2, 0.8];
        let artifact = pack_weight_artifact_from_format(
            &master,
            &[2, 3],
            crate::LowBitWeightFormat::Ternary158,
        )
        .expect("artifact");
        let codes = unpack_weight_artifact_to_i8_codes(&artifact);
        let reference = quantize_ternary_absmean(&master).values;
        assert_eq!(codes, reference);
    }
}
