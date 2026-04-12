use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{Int, Tensor, TensorData};
use burn_autodiff::Autodiff;
#[cfg(feature = "cuda")]
use burn_autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn_cubecl::CubeBackend;
#[cfg(feature = "cuda")]
use burn_cuda::Cuda;
#[cfg(feature = "cuda")]
use burn_dragon_kernel::api::low_bit::try_fused_packed_lowrank_training_autodiff_cuda_device_projection_scale;
use burn_dragon_kernel::api::low_bit::{
    pack_decoder_input_codes_i8x4, pack_decoder_weight_codes_i8x4, pack_lowrank_input_codes_i8x4,
    pack_lowrank_weight_codes_i8x4, packed_decoder_tail_device_reference,
    packed_lowrank_projection_device_reference, try_cube_fused_packed_decoder_tail_wgpu,
    try_cube_fused_packed_lowrank_projection_wgpu, try_fused_packed_decoder_tail,
    try_fused_packed_decoder_tail_training_autodiff, try_fused_packed_lowrank_projection,
    try_fused_packed_lowrank_training_autodiff, try_wgpu_packed_dot_decoder_tail,
    try_wgpu_packed_dot_decoder_tail_device_scale,
    try_wgpu_packed_dot_decoder_tail_prepacked_input_device_scale,
    try_wgpu_packed_dot_lowrank_projection, try_wgpu_packed_dot_lowrank_projection_device_scale,
    try_wgpu_packed_dot_lowrank_projection_from_f32_device_scale,
    try_wgpu_packed_dot_lowrank_projection_prepacked_input_device_scale,
    try_wgpu_quantize_activation_codes_i32, try_wgpu_quantize_pack_activation_i8x4,
};
use burn_wgpu::{RuntimeOptions, WgpuRuntime, graphics};

type WgpuBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type AutodiffBackendImpl = Autodiff<WgpuBackend>;
#[cfg(feature = "cuda")]
type CudaBackend = Cuda<f32, i32>;
#[cfg(feature = "cuda")]
type CudaAutodiffBackendImpl = Autodiff<CudaBackend>;
#[cfg(feature = "cuda")]
type CudaBalancedAutodiffBackendImpl = Autodiff<CudaBackend, BalancedCheckpointing>;

fn init_runtime(device: &<WgpuBackend as BackendTrait>::Device) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

fn deterministic_values(len: usize, offset: f32) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            (((idx as f32) * 0.173) + offset).sin() * 0.5
                + (((idx as f32) * 0.117) + offset).cos() * 0.25
        })
        .collect()
}

fn quantize_signed_values(values: &[f32]) -> (Vec<i32>, f32) {
    let mean_abs = if values.is_empty() {
        0.0
    } else {
        values.iter().map(|value| value.abs()).sum::<f32>() / values.len() as f32
    };
    let scale = (mean_abs * 2.0 / 127.0).max(1.0e-8);
    let codes = values
        .iter()
        .map(|value| (value / scale).round().clamp(-127.0, 127.0) as i32)
        .collect();
    (codes, scale)
}

fn tensor_from_values<B: BackendTrait, const D: usize>(
    values: Vec<f32>,
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, D> {
    Tensor::<B, D>::from_data(TensorData::new(values, shape), device)
}

fn int_tensor_from_values<B: BackendTrait, const D: usize>(
    values: Vec<i32>,
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, D, Int> {
    Tensor::<B, D, Int>::from_data(TensorData::new(values, shape), device)
}

fn assert_close<const D: usize, B: BackendTrait>(
    actual: Tensor<B, D>,
    expected: Tensor<B, D>,
    atol: f32,
    rtol: f32,
) {
    let actual = actual
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("actual values");
    let expected = expected
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("expected values");
    assert_eq!(actual.len(), expected.len());
    for (idx, (actual, expected)) in actual.into_iter().zip(expected.into_iter()).enumerate() {
        let tol = atol + rtol * expected.abs();
        assert!(
            (actual - expected).abs() <= tol,
            "mismatch at {idx}: actual={actual}, expected={expected}, tol={tol}"
        );
    }
}

fn ste_quantized_from_int_codes<B: AutodiffBackend, const D: usize>(
    original: Tensor<B, D>,
    codes: &Tensor<B, D, Int>,
    scale: f32,
) -> Tensor<B, D> {
    let quantized = codes.clone().float().mul_scalar(scale);
    original.clone() + (quantized - original).detach()
}

fn ste_quantized_lowrank_weight<B: AutodiffBackend>(
    original: Tensor<B, 4>,
    codes: &Tensor<B, 3, Int>,
    scale: f32,
) -> Tensor<B, 4> {
    let [heads, embd, latent] = codes.shape().dims::<3>();
    let quantized = codes
        .clone()
        .float()
        .mul_scalar(scale)
        .reshape([1, heads, embd, latent]);
    original.clone() + (quantized - original).detach()
}

fn ste_quantized_decoder_weight<B: AutodiffBackend>(
    original: Tensor<B, 2>,
    codes: &Tensor<B, 2, Int>,
    scale: f32,
) -> Tensor<B, 2> {
    let quantized = codes.clone().float().mul_scalar(scale);
    original.clone() + (quantized - original).detach()
}

fn reference_decoder_tail<B: BackendTrait>(
    y_neuron: Tensor<B, 4>,
    decoder: Tensor<B, 2>,
) -> Tensor<B, 4> {
    let [batch, heads, time, latent] = y_neuron.shape().dims::<4>();
    let dim = decoder.shape().dims::<2>()[1];
    let decoder_by_head = decoder.reshape([heads, latent, dim]);
    let mixed_by_head = y_neuron
        .swap_dims(0, 1)
        .reshape([heads, batch * time, latent]);
    mixed_by_head
        .matmul(decoder_by_head)
        .sum_dim(0)
        .reshape([batch, 1, time, dim])
}

#[test]
fn packed_lowrank_forward_matches_reference_on_wgpu() {
    let device = <WgpuBackend as BackendTrait>::Device::default();
    init_runtime(&device);

    let input_shape = [2, 1, 5, 16];
    let weight_shape = [1, 4, 16, 8];
    let input_values = deterministic_values(input_shape.iter().product(), 0.3);
    let weight_values = deterministic_values(weight_shape.iter().product(), 0.9);

    let input_codes = {
        let (codes, scale) = quantize_signed_values(&input_values);
        (
            int_tensor_from_values::<WgpuBackend, 4>(codes, input_shape, &device),
            scale,
        )
    };
    let weight_codes = {
        let (codes, scale) = quantize_signed_values(&weight_values);
        (
            int_tensor_from_values::<WgpuBackend, 3>(codes, [4, 16, 8], &device),
            scale,
        )
    };

    let actual = try_fused_packed_lowrank_projection(
        &input_codes.0,
        &weight_codes.0,
        input_codes.1,
        weight_codes.1,
        8,
    )
    .expect("fused lowrank forward");
    let expected = packed_lowrank_projection_device_reference(
        input_codes.0.clone().float().mul_scalar(input_codes.1),
        weight_codes.0,
        weight_codes.1,
        8,
    );
    assert_close(actual, expected, 5.0e-4, 5.0e-4);
}

#[test]
fn packed_lowrank_packed_dot_wgsl_matches_reference_on_wgpu_when_available() {
    let device = <WgpuBackend as BackendTrait>::Device::default();
    init_runtime(&device);

    let input_shape = [2, 1, 5, 16];
    let weight_shape = [1, 4, 16, 8];
    let input_values = deterministic_values(input_shape.iter().product(), 0.31);
    let weight_values = deterministic_values(weight_shape.iter().product(), 0.93);

    let (input_codes_values, input_scale) = quantize_signed_values(&input_values);
    let input_codes =
        int_tensor_from_values::<WgpuBackend, 4>(input_codes_values, input_shape, &device);
    let (weight_codes_values, weight_scale) = quantize_signed_values(&weight_values);
    let weight_codes =
        int_tensor_from_values::<WgpuBackend, 3>(weight_codes_values, [4, 16, 8], &device);

    let Some(actual) = try_wgpu_packed_dot_lowrank_projection(
        &input_codes,
        &weight_codes,
        input_scale,
        weight_scale,
        8,
    ) else {
        return;
    };
    let expected = packed_lowrank_projection_device_reference(
        input_codes.clone().float().mul_scalar(input_scale),
        weight_codes,
        weight_scale,
        8,
    );
    assert_close(actual, expected, 5.0e-4, 5.0e-4);
}

#[test]
fn packed_lowrank_packed_dot_device_scale_matches_reference_on_wgpu_when_available() {
    let device = <WgpuBackend as BackendTrait>::Device::default();
    init_runtime(&device);

    let input_shape = [2, 1, 5, 16];
    let weight_shape = [1, 4, 16, 8];
    let input_values = deterministic_values(input_shape.iter().product(), 0.315);
    let weight_values = deterministic_values(weight_shape.iter().product(), 0.935);

    let (input_codes_values, input_scale) = quantize_signed_values(&input_values);
    let input_codes =
        int_tensor_from_values::<WgpuBackend, 4>(input_codes_values, input_shape, &device);
    let input_scale_tensor = Tensor::<WgpuBackend, 1>::from_data([input_scale], &device);
    let (weight_codes_values, weight_scale) = quantize_signed_values(&weight_values);
    let weight_codes =
        int_tensor_from_values::<WgpuBackend, 3>(weight_codes_values, [4, 16, 8], &device);

    let Some(actual) = try_wgpu_packed_dot_lowrank_projection_device_scale(
        &input_codes,
        &weight_codes,
        &input_scale_tensor,
        weight_scale,
        8,
    ) else {
        return;
    };
    let expected = packed_lowrank_projection_device_reference(
        input_codes.clone().float().mul_scalar(input_scale),
        weight_codes,
        weight_scale,
        8,
    );
    assert_close(actual, expected, 5.0e-4, 5.0e-4);
}

#[test]
fn quantize_pack_activation_i8x4_matches_host_pack_on_wgpu_when_available() {
    let device = <WgpuBackend as BackendTrait>::Device::default();
    init_runtime(&device);

    let input_shape = [2, 3, 5, 18];
    let input_values = deterministic_values(input_shape.iter().product(), 0.271);
    let input = tensor_from_values::<WgpuBackend, 4>(input_values.clone(), input_shape, &device);
    let (codes_values, scale) = quantize_signed_values(&input_values);
    let expected = pack_lowrank_input_codes_i8x4(
        &codes_values
            .iter()
            .map(|value| *value as i8)
            .collect::<Vec<_>>(),
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let scale_tensor = Tensor::<WgpuBackend, 1>::from_data([scale], &device);

    let Some(actual) = try_wgpu_quantize_pack_activation_i8x4(&input, &scale_tensor, 127, false)
    else {
        return;
    };
    let actual = actual
        .into_data()
        .convert::<i32>()
        .into_vec::<i32>()
        .expect("packed activation values");
    assert_eq!(actual, expected);
}

#[test]
fn quantize_activation_codes_i32_matches_host_quantization_on_wgpu_when_available() {
    let device = <WgpuBackend as BackendTrait>::Device::default();
    init_runtime(&device);

    let input_shape = [2, 3, 5, 18];
    let input_values = deterministic_values(input_shape.iter().product(), 0.287);
    let input = tensor_from_values::<WgpuBackend, 4>(input_values.clone(), input_shape, &device);
    let (expected, scale) = quantize_signed_values(&input_values);
    let scale_tensor = Tensor::<WgpuBackend, 1>::from_data([scale], &device);

    let Some(actual) = try_wgpu_quantize_activation_codes_i32(&input, &scale_tensor, 127, false)
    else {
        return;
    };
    let actual = actual
        .into_data()
        .convert::<i32>()
        .into_vec::<i32>()
        .expect("quantized activation codes");
    assert_eq!(actual, expected);
}

#[test]
fn packed_lowrank_prepacked_packed_dot_device_scale_matches_reference_on_wgpu_when_available() {
    let device = <WgpuBackend as BackendTrait>::Device::default();
    init_runtime(&device);

    let input_shape = [2, 1, 5, 18];
    let weight_shape = [1, 4, 18, 9];
    let input_values = deterministic_values(input_shape.iter().product(), 0.319);
    let weight_values = deterministic_values(weight_shape.iter().product(), 0.939);

    let (input_codes_values, input_scale) = quantize_signed_values(&input_values);
    let input_codes_i8 = input_codes_values
        .iter()
        .map(|value| *value as i8)
        .collect::<Vec<_>>();
    let input_packed_values = pack_lowrank_input_codes_i8x4(
        &input_codes_i8,
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let input_packed = int_tensor_from_values::<WgpuBackend, 4>(
        input_packed_values,
        [
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3].div_ceil(4),
        ],
        &device,
    );
    let input_scale_tensor = Tensor::<WgpuBackend, 1>::from_data([input_scale], &device);

    let (weight_codes_values, weight_scale) = quantize_signed_values(&weight_values);
    let weight_codes_i8 = weight_codes_values
        .iter()
        .map(|value| *value as i8)
        .collect::<Vec<_>>();
    let weight_packed_values =
        pack_lowrank_weight_codes_i8x4(&weight_codes_i8, 4, input_shape[3], 9);
    let weight_packed = int_tensor_from_values::<WgpuBackend, 3>(
        weight_packed_values,
        [4, input_shape[3].div_ceil(4), 9],
        &device,
    );
    let weight_codes = int_tensor_from_values::<WgpuBackend, 3>(
        weight_codes_values,
        [4, input_shape[3], 9],
        &device,
    );

    let Some(actual) = try_wgpu_packed_dot_lowrank_projection_prepacked_input_device_scale(
        &input_packed,
        &weight_packed,
        &input_scale_tensor,
        weight_scale,
        9,
    ) else {
        return;
    };
    let expected = packed_lowrank_projection_device_reference(
        int_tensor_from_values::<WgpuBackend, 4>(input_codes_values, input_shape, &device)
            .float()
            .mul_scalar(input_scale),
        weight_codes,
        weight_scale,
        9,
    );
    assert_close(actual, expected, 5.0e-4, 5.0e-4);
}

#[test]
fn packed_lowrank_from_f32_packed_dot_device_scale_matches_reference_on_wgpu_when_available() {
    let device = <WgpuBackend as BackendTrait>::Device::default();
    init_runtime(&device);

    let input_shape = [2, 1, 5, 18];
    let weight_shape = [1, 4, 18, 9];
    let input_values = deterministic_values(input_shape.iter().product(), 0.337);
    let weight_values = deterministic_values(weight_shape.iter().product(), 0.957);

    let input = tensor_from_values::<WgpuBackend, 4>(input_values.clone(), input_shape, &device);
    let (_, input_scale) = quantize_signed_values(&input_values);
    let input_scale_tensor = Tensor::<WgpuBackend, 1>::from_data([input_scale], &device);

    let (weight_codes_values, weight_scale) = quantize_signed_values(&weight_values);
    let weight_codes_i8 = weight_codes_values
        .iter()
        .map(|value| *value as i8)
        .collect::<Vec<_>>();
    let weight_packed_values =
        pack_lowrank_weight_codes_i8x4(&weight_codes_i8, 4, input_shape[3], 9);
    let weight_packed = int_tensor_from_values::<WgpuBackend, 3>(
        weight_packed_values,
        [4, input_shape[3].div_ceil(4), 9],
        &device,
    );
    let weight_codes = int_tensor_from_values::<WgpuBackend, 3>(
        weight_codes_values,
        [4, input_shape[3], 9],
        &device,
    );

    let Some(actual) = try_wgpu_packed_dot_lowrank_projection_from_f32_device_scale(
        &input,
        &weight_packed,
        &input_scale_tensor,
        weight_scale,
        9,
        127,
        false,
    ) else {
        return;
    };
    let expected = packed_lowrank_projection_device_reference(input, weight_codes, weight_scale, 9);
    assert_close(actual, expected, 5.0e-4, 5.0e-4);
}

#[test]
fn packed_decoder_tail_forward_matches_reference_on_wgpu() {
    let device = <WgpuBackend as BackendTrait>::Device::default();
    init_runtime(&device);

    let y_shape = [2, 4, 5, 8];
    let weight_shape = [32, 16];
    let y_values = deterministic_values(y_shape.iter().product(), 0.4);
    let weight_values = deterministic_values(weight_shape.iter().product(), 1.1);

    let (y_codes_values, y_scale) = quantize_signed_values(&y_values);
    let y_codes = int_tensor_from_values::<WgpuBackend, 4>(y_codes_values, y_shape, &device);
    let (weight_codes_values, weight_scale) = quantize_signed_values(&weight_values);
    let weight_codes =
        int_tensor_from_values::<WgpuBackend, 2>(weight_codes_values, weight_shape, &device);

    let actual = try_fused_packed_decoder_tail(&y_codes, &weight_codes, y_scale, weight_scale)
        .expect("fused decoder tail");
    let expected = packed_decoder_tail_device_reference(
        y_codes.clone().float().mul_scalar(y_scale),
        weight_codes,
        weight_scale,
    );
    assert_close(actual, expected, 5.0e-4, 5.0e-4);
}

#[test]
fn packed_decoder_tail_packed_dot_wgsl_matches_reference_on_wgpu_when_available() {
    let device = <WgpuBackend as BackendTrait>::Device::default();
    init_runtime(&device);

    let y_shape = [2, 4, 5, 8];
    let weight_shape = [32, 16];
    let y_values = deterministic_values(y_shape.iter().product(), 0.41);
    let weight_values = deterministic_values(weight_shape.iter().product(), 1.14);

    let (y_codes_values, y_scale) = quantize_signed_values(&y_values);
    let y_codes = int_tensor_from_values::<WgpuBackend, 4>(y_codes_values, y_shape, &device);
    let (weight_codes_values, weight_scale) = quantize_signed_values(&weight_values);
    let weight_codes =
        int_tensor_from_values::<WgpuBackend, 2>(weight_codes_values, weight_shape, &device);

    let Some(actual) =
        try_wgpu_packed_dot_decoder_tail(&y_codes, &weight_codes, y_scale, weight_scale)
    else {
        return;
    };
    let expected = packed_decoder_tail_device_reference(
        y_codes.clone().float().mul_scalar(y_scale),
        weight_codes,
        weight_scale,
    );
    assert_close(actual, expected, 5.0e-4, 5.0e-4);
}

#[test]
fn packed_decoder_tail_packed_dot_device_scale_matches_reference_on_wgpu_when_available() {
    let device = <WgpuBackend as BackendTrait>::Device::default();
    init_runtime(&device);

    let y_shape = [2, 4, 5, 8];
    let weight_shape = [32, 16];
    let y_values = deterministic_values(y_shape.iter().product(), 0.415);
    let weight_values = deterministic_values(weight_shape.iter().product(), 1.145);

    let (y_codes_values, y_scale) = quantize_signed_values(&y_values);
    let y_codes = int_tensor_from_values::<WgpuBackend, 4>(y_codes_values, y_shape, &device);
    let y_scale_tensor = Tensor::<WgpuBackend, 1>::from_data([y_scale], &device);
    let (weight_codes_values, weight_scale) = quantize_signed_values(&weight_values);
    let weight_codes =
        int_tensor_from_values::<WgpuBackend, 2>(weight_codes_values, weight_shape, &device);

    let Some(actual) = try_wgpu_packed_dot_decoder_tail_device_scale(
        &y_codes,
        &weight_codes,
        &y_scale_tensor,
        weight_scale,
    ) else {
        return;
    };
    let expected = packed_decoder_tail_device_reference(
        y_codes.clone().float().mul_scalar(y_scale),
        weight_codes,
        weight_scale,
    );
    assert_close(actual, expected, 5.0e-4, 5.0e-4);
}

#[test]
fn packed_decoder_tail_prepacked_packed_dot_device_scale_matches_reference_on_wgpu_when_available()
{
    let device = <WgpuBackend as BackendTrait>::Device::default();
    init_runtime(&device);

    let y_shape = [2, 4, 5, 10];
    let weight_shape = [40, 16];
    let y_values = deterministic_values(y_shape.iter().product(), 0.419);
    let weight_values = deterministic_values(weight_shape.iter().product(), 1.149);

    let (y_codes_values, y_scale) = quantize_signed_values(&y_values);
    let y_codes_i8 = y_codes_values
        .iter()
        .map(|value| *value as i8)
        .collect::<Vec<_>>();
    let y_packed_values =
        pack_decoder_input_codes_i8x4(&y_codes_i8, y_shape[0], y_shape[1], y_shape[2], y_shape[3]);
    let y_packed = int_tensor_from_values::<WgpuBackend, 4>(
        y_packed_values,
        [y_shape[0], y_shape[1], y_shape[2], y_shape[3].div_ceil(4)],
        &device,
    );
    let y_scale_tensor = Tensor::<WgpuBackend, 1>::from_data([y_scale], &device);

    let (weight_codes_values, weight_scale) = quantize_signed_values(&weight_values);
    let weight_codes_i8 = weight_codes_values
        .iter()
        .map(|value| *value as i8)
        .collect::<Vec<_>>();
    let weight_packed_values = pack_decoder_weight_codes_i8x4(&weight_codes_i8, 4, y_shape[3], 16);
    let weight_packed = int_tensor_from_values::<WgpuBackend, 2>(
        weight_packed_values,
        [4 * y_shape[3].div_ceil(4), 16],
        &device,
    );
    let weight_codes =
        int_tensor_from_values::<WgpuBackend, 2>(weight_codes_values, weight_shape, &device);

    let Some(actual) = try_wgpu_packed_dot_decoder_tail_prepacked_input_device_scale(
        &y_packed,
        &weight_packed,
        &y_scale_tensor,
        weight_scale,
    ) else {
        return;
    };
    let expected = packed_decoder_tail_device_reference(
        int_tensor_from_values::<WgpuBackend, 4>(y_codes_values, y_shape, &device)
            .float()
            .mul_scalar(y_scale),
        weight_codes,
        weight_scale,
    );
    assert_close(actual, expected, 5.0e-4, 5.0e-4);
}

#[test]
fn packed_dot_and_cube_forward_paths_agree_on_wgpu_when_both_available() {
    let device = <WgpuBackend as BackendTrait>::Device::default();
    init_runtime(&device);

    let input_shape = [2, 1, 5, 16];
    let weight_shape = [1, 4, 16, 8];
    let input_values = deterministic_values(input_shape.iter().product(), 0.29);
    let weight_values = deterministic_values(weight_shape.iter().product(), 0.88);

    let (input_codes_values, input_scale) = quantize_signed_values(&input_values);
    let input_codes =
        int_tensor_from_values::<WgpuBackend, 4>(input_codes_values, input_shape, &device);
    let (weight_codes_values, weight_scale) = quantize_signed_values(&weight_values);
    let weight_codes =
        int_tensor_from_values::<WgpuBackend, 3>(weight_codes_values, [4, 16, 8], &device);

    let Some(packed_dot) = try_wgpu_packed_dot_lowrank_projection(
        &input_codes,
        &weight_codes,
        input_scale,
        weight_scale,
        8,
    ) else {
        return;
    };
    let cube = try_cube_fused_packed_lowrank_projection_wgpu(
        &input_codes,
        &weight_codes,
        input_scale,
        weight_scale,
        8,
    )
    .expect("cube lowrank path");
    assert_close(packed_dot, cube, 5.0e-4, 5.0e-4);

    let y_shape = [2, 4, 5, 8];
    let decoder_shape = [32, 16];
    let y_values = deterministic_values(y_shape.iter().product(), 0.44);
    let decoder_values = deterministic_values(decoder_shape.iter().product(), 1.22);
    let (y_codes_values, y_scale) = quantize_signed_values(&y_values);
    let y_codes = int_tensor_from_values::<WgpuBackend, 4>(y_codes_values, y_shape, &device);
    let (decoder_codes_values, decoder_scale) = quantize_signed_values(&decoder_values);
    let decoder_codes =
        int_tensor_from_values::<WgpuBackend, 2>(decoder_codes_values, decoder_shape, &device);
    let Some(packed_dot_tail) =
        try_wgpu_packed_dot_decoder_tail(&y_codes, &decoder_codes, y_scale, decoder_scale)
    else {
        return;
    };
    let cube_tail =
        try_cube_fused_packed_decoder_tail_wgpu(&y_codes, &decoder_codes, y_scale, decoder_scale)
            .expect("cube decoder tail path");
    assert_close(packed_dot_tail, cube_tail, 5.0e-4, 5.0e-4);
}

#[test]
fn packed_lowrank_training_autodiff_matches_reference_gradients_on_wgpu() {
    let device = <AutodiffBackendImpl as BackendTrait>::Device::default();
    init_runtime(&device);

    let input_shape = [2, 1, 5, 16];
    let weight_shape = [1, 4, 16, 8];
    let input_values = deterministic_values(input_shape.iter().product(), 0.2);
    let weight_values = deterministic_values(weight_shape.iter().product(), 0.7);
    let output_weights_values = deterministic_values(2 * 4 * 5 * 8, 1.3);

    let input_fused =
        tensor_from_values::<AutodiffBackendImpl, 4>(input_values.clone(), input_shape, &device)
            .require_grad();
    let weight_fused =
        tensor_from_values::<AutodiffBackendImpl, 4>(weight_values.clone(), weight_shape, &device)
            .require_grad();
    let input_ref =
        tensor_from_values::<AutodiffBackendImpl, 4>(input_values.clone(), input_shape, &device)
            .require_grad();
    let weight_ref =
        tensor_from_values::<AutodiffBackendImpl, 4>(weight_values.clone(), weight_shape, &device)
            .require_grad();

    let (input_codes_values, input_scale) = quantize_signed_values(&input_values);
    let input_codes =
        int_tensor_from_values::<AutodiffBackendImpl, 4>(input_codes_values, input_shape, &device);
    let (weight_codes_values, weight_scale) = quantize_signed_values(&weight_values);
    let weight_codes =
        int_tensor_from_values::<AutodiffBackendImpl, 3>(weight_codes_values, [4, 16, 8], &device);

    let quantized_input_fused =
        ste_quantized_from_int_codes(input_fused.clone(), &input_codes, input_scale);
    let quantized_weight_fused =
        ste_quantized_lowrank_weight(weight_fused.clone(), &weight_codes, weight_scale);
    let quantized_input_ref =
        ste_quantized_from_int_codes(input_ref.clone(), &input_codes, input_scale);
    let quantized_weight_ref =
        ste_quantized_lowrank_weight(weight_ref.clone(), &weight_codes, weight_scale);

    let fused = try_fused_packed_lowrank_training_autodiff(
        &quantized_input_fused,
        &quantized_weight_fused,
        &input_codes,
        &weight_codes,
        input_scale,
        weight_scale,
        8,
        false,
        None,
    )
    .expect("fused autodiff lowrank");
    let reference = quantized_input_ref.matmul(quantized_weight_ref);
    let output_weights =
        tensor_from_values::<AutodiffBackendImpl, 4>(output_weights_values, [2, 4, 5, 8], &device);

    let fused_grads = (fused * output_weights.clone()).sum().backward();
    let reference_grads = (reference * output_weights).sum().backward();

    let fused_input_grad = input_fused.grad(&fused_grads).expect("fused input grad");
    let reference_input_grad = input_ref
        .grad(&reference_grads)
        .expect("reference input grad");
    let fused_weight_grad = weight_fused.grad(&fused_grads).expect("fused weight grad");
    let reference_weight_grad = weight_ref
        .grad(&reference_grads)
        .expect("reference weight grad");

    assert_close(fused_input_grad, reference_input_grad, 5.0e-4, 5.0e-4);
    assert_close(fused_weight_grad, reference_weight_grad, 5.0e-4, 5.0e-4);
}

#[cfg(feature = "cuda")]
#[test]
fn packed_lowrank_training_autodiff_matches_reference_gradients_on_cuda_bdh_shape() {
    let device = <CudaAutodiffBackendImpl as BackendTrait>::Device::default();

    let input_shape = [2, 4, 5, 16];
    let weight_shape = [1, 4, 16, 8];
    let input_values = deterministic_values(input_shape.iter().product(), 0.2);
    let weight_values = deterministic_values(weight_shape.iter().product(), 0.7);
    let output_weights_values = deterministic_values(2 * 4 * 5 * 8, 1.3);

    let input_fused = tensor_from_values::<CudaAutodiffBackendImpl, 4>(
        input_values.clone(),
        input_shape,
        &device,
    )
    .require_grad();
    let weight_fused = tensor_from_values::<CudaAutodiffBackendImpl, 4>(
        weight_values.clone(),
        weight_shape,
        &device,
    )
    .require_grad();
    let input_ref = tensor_from_values::<CudaAutodiffBackendImpl, 4>(
        input_values.clone(),
        input_shape,
        &device,
    )
    .require_grad();
    let weight_ref = tensor_from_values::<CudaAutodiffBackendImpl, 4>(
        weight_values.clone(),
        weight_shape,
        &device,
    )
    .require_grad();

    let (input_codes_values, input_scale) = quantize_signed_values(&input_values);
    let input_codes = int_tensor_from_values::<CudaAutodiffBackendImpl, 4>(
        input_codes_values,
        input_shape,
        &device,
    );
    let (weight_codes_values, weight_scale) = quantize_signed_values(&weight_values);
    let weight_codes = int_tensor_from_values::<CudaAutodiffBackendImpl, 3>(
        weight_codes_values,
        [4, 16, 8],
        &device,
    );

    let quantized_input_fused =
        ste_quantized_from_int_codes(input_fused.clone(), &input_codes, input_scale);
    let quantized_weight_fused =
        ste_quantized_lowrank_weight(weight_fused.clone(), &weight_codes, weight_scale);
    let quantized_input_ref =
        ste_quantized_from_int_codes(input_ref.clone(), &input_codes, input_scale);
    let quantized_weight_ref =
        ste_quantized_lowrank_weight(weight_ref.clone(), &weight_codes, weight_scale);

    let fused = try_fused_packed_lowrank_training_autodiff(
        &quantized_input_fused,
        &quantized_weight_fused,
        &input_codes,
        &weight_codes,
        input_scale,
        weight_scale,
        8,
        false,
        None,
    )
    .expect("fused autodiff lowrank cuda");
    let reference = quantized_input_ref.matmul(quantized_weight_ref);
    let output_weights = tensor_from_values::<CudaAutodiffBackendImpl, 4>(
        output_weights_values,
        [2, 4, 5, 8],
        &device,
    );

    let fused_grads = (fused * output_weights.clone()).sum().backward();
    let reference_grads = (reference * output_weights).sum().backward();

    let fused_input_grad = input_fused.grad(&fused_grads).expect("fused input grad");
    let reference_input_grad = input_ref
        .grad(&reference_grads)
        .expect("reference input grad");
    let fused_weight_grad = weight_fused.grad(&fused_grads).expect("fused weight grad");
    let reference_weight_grad = weight_ref
        .grad(&reference_grads)
        .expect("reference weight grad");

    assert_close(fused_input_grad, reference_input_grad, 5.0e-4, 5.0e-4);
    assert_close(fused_weight_grad, reference_weight_grad, 5.0e-4, 5.0e-4);
}

#[cfg(feature = "cuda")]
#[test]
fn packed_lowrank_training_autodiff_cuda_device_scale_matches_reference_gradients_on_cuda_bdh_shape()
 {
    let device = <CudaAutodiffBackendImpl as BackendTrait>::Device::default();

    let input_shape = [2, 4, 5, 16];
    let weight_shape = [1, 4, 16, 8];
    let input_values = deterministic_values(input_shape.iter().product(), 0.24);
    let weight_values = deterministic_values(weight_shape.iter().product(), 0.73);
    let output_weights_values = deterministic_values(2 * 4 * 5 * 8, 1.31);

    let input_fused = tensor_from_values::<CudaAutodiffBackendImpl, 4>(
        input_values.clone(),
        input_shape,
        &device,
    )
    .require_grad();
    let weight_fused = tensor_from_values::<CudaAutodiffBackendImpl, 4>(
        weight_values.clone(),
        weight_shape,
        &device,
    )
    .require_grad();
    let input_ref = tensor_from_values::<CudaAutodiffBackendImpl, 4>(
        input_values.clone(),
        input_shape,
        &device,
    )
    .require_grad();
    let weight_ref = tensor_from_values::<CudaAutodiffBackendImpl, 4>(
        weight_values.clone(),
        weight_shape,
        &device,
    )
    .require_grad();

    let (input_codes_values, input_scale) = quantize_signed_values(&input_values);
    let input_codes = int_tensor_from_values::<CudaAutodiffBackendImpl, 4>(
        input_codes_values,
        input_shape,
        &device,
    );
    let (weight_codes_values, weight_scale) = quantize_signed_values(&weight_values);
    let weight_codes = int_tensor_from_values::<CudaAutodiffBackendImpl, 3>(
        weight_codes_values,
        [4, 16, 8],
        &device,
    );
    let projection_scale =
        Tensor::<CudaAutodiffBackendImpl, 1>::from_data([input_scale * weight_scale], &device);

    let quantized_input_fused =
        ste_quantized_from_int_codes(input_fused.clone(), &input_codes, input_scale);
    let quantized_weight_fused =
        ste_quantized_lowrank_weight(weight_fused.clone(), &weight_codes, weight_scale);
    let quantized_input_ref =
        ste_quantized_from_int_codes(input_ref.clone(), &input_codes, input_scale);
    let quantized_weight_ref =
        ste_quantized_lowrank_weight(weight_ref.clone(), &weight_codes, weight_scale);

    let fused = try_fused_packed_lowrank_training_autodiff_cuda_device_projection_scale(
        &quantized_input_fused,
        &quantized_weight_fused,
        &input_codes,
        &weight_codes,
        input_scale,
        &projection_scale,
        8,
        false,
        None,
    )
    .expect("fused autodiff lowrank cuda device-scale");
    let reference = quantized_input_ref.matmul(quantized_weight_ref);
    let output_weights = tensor_from_values::<CudaAutodiffBackendImpl, 4>(
        output_weights_values,
        [2, 4, 5, 8],
        &device,
    );

    let fused_grads = (fused * output_weights.clone()).sum().backward();
    let reference_grads = (reference * output_weights).sum().backward();

    let fused_input_grad = input_fused.grad(&fused_grads).expect("fused input grad");
    let reference_input_grad = input_ref
        .grad(&reference_grads)
        .expect("reference input grad");
    let fused_weight_grad = weight_fused.grad(&fused_grads).expect("fused weight grad");
    let reference_weight_grad = weight_ref
        .grad(&reference_grads)
        .expect("reference weight grad");

    assert_close(fused_input_grad, reference_input_grad, 5.0e-4, 5.0e-4);
    assert_close(fused_weight_grad, reference_weight_grad, 5.0e-4, 5.0e-4);
}

#[cfg(feature = "cuda")]
#[test]
fn packed_lowrank_training_autodiff_cuda_device_scale_matches_reference_gradients_on_cuda_bdh_shape_balanced_checkpointing()
 {
    let device = <CudaBalancedAutodiffBackendImpl as BackendTrait>::Device::default();

    let input_shape = [2, 4, 5, 16];
    let weight_shape = [1, 4, 16, 8];
    let input_values = deterministic_values(input_shape.iter().product(), 0.26);
    let weight_values = deterministic_values(weight_shape.iter().product(), 0.75);
    let output_weights_values = deterministic_values(2 * 4 * 5 * 8, 1.33);

    let input_fused = tensor_from_values::<CudaBalancedAutodiffBackendImpl, 4>(
        input_values.clone(),
        input_shape,
        &device,
    )
    .require_grad();
    let weight_fused = tensor_from_values::<CudaBalancedAutodiffBackendImpl, 4>(
        weight_values.clone(),
        weight_shape,
        &device,
    )
    .require_grad();
    let input_ref = tensor_from_values::<CudaBalancedAutodiffBackendImpl, 4>(
        input_values.clone(),
        input_shape,
        &device,
    )
    .require_grad();
    let weight_ref = tensor_from_values::<CudaBalancedAutodiffBackendImpl, 4>(
        weight_values.clone(),
        weight_shape,
        &device,
    )
    .require_grad();

    let (input_codes_values, input_scale) = quantize_signed_values(&input_values);
    let input_codes = int_tensor_from_values::<CudaBalancedAutodiffBackendImpl, 4>(
        input_codes_values,
        input_shape,
        &device,
    );
    let (weight_codes_values, weight_scale) = quantize_signed_values(&weight_values);
    let weight_codes = int_tensor_from_values::<CudaBalancedAutodiffBackendImpl, 3>(
        weight_codes_values,
        [4, 16, 8],
        &device,
    );
    let projection_scale = Tensor::<CudaBalancedAutodiffBackendImpl, 1>::from_data(
        [input_scale * weight_scale],
        &device,
    );

    let quantized_input_fused =
        ste_quantized_from_int_codes(input_fused.clone(), &input_codes, input_scale);
    let quantized_weight_fused =
        ste_quantized_lowrank_weight(weight_fused.clone(), &weight_codes, weight_scale);
    let quantized_input_ref =
        ste_quantized_from_int_codes(input_ref.clone(), &input_codes, input_scale);
    let quantized_weight_ref =
        ste_quantized_lowrank_weight(weight_ref.clone(), &weight_codes, weight_scale);

    let fused = try_fused_packed_lowrank_training_autodiff_cuda_device_projection_scale(
        &quantized_input_fused,
        &quantized_weight_fused,
        &input_codes,
        &weight_codes,
        input_scale,
        &projection_scale,
        8,
        false,
        None,
    )
    .expect("fused autodiff lowrank cuda device-scale balanced");
    let reference = quantized_input_ref.matmul(quantized_weight_ref);
    let output_weights = tensor_from_values::<CudaBalancedAutodiffBackendImpl, 4>(
        output_weights_values,
        [2, 4, 5, 8],
        &device,
    );

    let fused_grads = (fused * output_weights.clone()).sum().backward();
    let reference_grads = (reference * output_weights).sum().backward();

    let fused_input_grad = input_fused.grad(&fused_grads).expect("fused input grad");
    let reference_input_grad = input_ref
        .grad(&reference_grads)
        .expect("reference input grad");
    let fused_weight_grad = weight_fused.grad(&fused_grads).expect("fused weight grad");
    let reference_weight_grad = weight_ref
        .grad(&reference_grads)
        .expect("reference weight grad");

    assert_close(fused_input_grad, reference_input_grad, 5.0e-4, 5.0e-4);
    assert_close(fused_weight_grad, reference_weight_grad, 5.0e-4, 5.0e-4);
}

#[cfg(feature = "cuda")]
#[test]
fn packed_lowrank_training_autodiff_cuda_device_scale_accepts_computed_projection_scale_tensor() {
    let device = <CudaAutodiffBackendImpl as BackendTrait>::Device::default();

    let input_shape = [2, 4, 5, 16];
    let weight_shape = [1, 4, 16, 8];
    let input_values = deterministic_values(input_shape.iter().product(), 0.261);
    let weight_values = deterministic_values(weight_shape.iter().product(), 0.751);
    let output_weights_values = deterministic_values(2 * 4 * 5 * 8, 1.37);

    let input_fused = tensor_from_values::<CudaAutodiffBackendImpl, 4>(
        input_values.clone(),
        input_shape,
        &device,
    )
    .require_grad();
    let weight_fused = tensor_from_values::<CudaAutodiffBackendImpl, 4>(
        weight_values.clone(),
        weight_shape,
        &device,
    )
    .require_grad();
    let input_ref = tensor_from_values::<CudaAutodiffBackendImpl, 4>(
        input_values.clone(),
        input_shape,
        &device,
    )
    .require_grad();
    let weight_ref = tensor_from_values::<CudaAutodiffBackendImpl, 4>(
        weight_values.clone(),
        weight_shape,
        &device,
    )
    .require_grad();

    let (input_codes_values, input_scale) = quantize_signed_values(&input_values);
    let input_codes = int_tensor_from_values::<CudaAutodiffBackendImpl, 4>(
        input_codes_values,
        input_shape,
        &device,
    );
    let (weight_codes_values, weight_scale) = quantize_signed_values(&weight_values);
    let weight_codes = int_tensor_from_values::<CudaAutodiffBackendImpl, 3>(
        weight_codes_values,
        [4, 16, 8],
        &device,
    );
    let projection_scale = Tensor::<CudaAutodiffBackendImpl, 1>::from_data([weight_scale], &device)
        .mul_scalar(input_scale);

    let quantized_input_fused =
        ste_quantized_from_int_codes(input_fused.clone(), &input_codes, input_scale);
    let quantized_weight_fused =
        ste_quantized_lowrank_weight(weight_fused.clone(), &weight_codes, weight_scale);
    let quantized_input_ref =
        ste_quantized_from_int_codes(input_ref.clone(), &input_codes, input_scale);
    let quantized_weight_ref =
        ste_quantized_lowrank_weight(weight_ref.clone(), &weight_codes, weight_scale);

    let fused = try_fused_packed_lowrank_training_autodiff_cuda_device_projection_scale(
        &quantized_input_fused,
        &quantized_weight_fused,
        &input_codes,
        &weight_codes,
        input_scale,
        &projection_scale,
        8,
        false,
        None,
    )
    .expect("fused autodiff lowrank cuda device-scale computed tensor");
    let reference = quantized_input_ref.matmul(quantized_weight_ref);
    let output_weights = tensor_from_values::<CudaAutodiffBackendImpl, 4>(
        output_weights_values,
        [2, 4, 5, 8],
        &device,
    );

    let fused_grads = (fused * output_weights.clone()).sum().backward();
    let reference_grads = (reference * output_weights).sum().backward();

    let fused_input_grad = input_fused.grad(&fused_grads).expect("fused input grad");
    let reference_input_grad = input_ref
        .grad(&reference_grads)
        .expect("reference input grad");
    let fused_weight_grad = weight_fused.grad(&fused_grads).expect("fused weight grad");
    let reference_weight_grad = weight_ref
        .grad(&reference_grads)
        .expect("reference weight grad");

    assert_close(fused_input_grad, reference_input_grad, 5.0e-4, 5.0e-4);
    assert_close(fused_weight_grad, reference_weight_grad, 5.0e-4, 5.0e-4);
}

#[test]
fn packed_decoder_tail_training_autodiff_matches_reference_gradients_on_wgpu() {
    let device = <AutodiffBackendImpl as BackendTrait>::Device::default();
    init_runtime(&device);

    let y_shape = [2, 4, 5, 8];
    let weight_shape = [32, 16];
    let y_values = deterministic_values(y_shape.iter().product(), 0.5);
    let weight_values = deterministic_values(weight_shape.iter().product(), 1.0);
    let output_weights_values = deterministic_values(2 * 1 * 5 * 16, 1.8);

    let y_fused = tensor_from_values::<AutodiffBackendImpl, 4>(y_values.clone(), y_shape, &device)
        .require_grad();
    let weight_fused =
        tensor_from_values::<AutodiffBackendImpl, 2>(weight_values.clone(), weight_shape, &device)
            .require_grad();
    let y_ref = tensor_from_values::<AutodiffBackendImpl, 4>(y_values.clone(), y_shape, &device)
        .require_grad();
    let weight_ref =
        tensor_from_values::<AutodiffBackendImpl, 2>(weight_values.clone(), weight_shape, &device)
            .require_grad();

    let (y_codes_values, y_scale) = quantize_signed_values(&y_values);
    let y_codes =
        int_tensor_from_values::<AutodiffBackendImpl, 4>(y_codes_values, y_shape, &device);
    let (weight_codes_values, weight_scale) = quantize_signed_values(&weight_values);
    let weight_codes = int_tensor_from_values::<AutodiffBackendImpl, 2>(
        weight_codes_values,
        weight_shape,
        &device,
    );

    let quantized_y_fused = ste_quantized_from_int_codes(y_fused.clone(), &y_codes, y_scale);
    let quantized_weight_fused =
        ste_quantized_decoder_weight(weight_fused.clone(), &weight_codes, weight_scale);
    let quantized_y_ref = ste_quantized_from_int_codes(y_ref.clone(), &y_codes, y_scale);
    let quantized_weight_ref =
        ste_quantized_decoder_weight(weight_ref.clone(), &weight_codes, weight_scale);

    let fused = try_fused_packed_decoder_tail_training_autodiff(
        &quantized_y_fused,
        &quantized_weight_fused,
        &y_codes,
        &weight_codes,
        y_scale,
        weight_scale,
        false,
    )
    .expect("fused autodiff decoder tail");
    let reference = reference_decoder_tail(quantized_y_ref, quantized_weight_ref);
    let output_weights =
        tensor_from_values::<AutodiffBackendImpl, 4>(output_weights_values, [2, 1, 5, 16], &device);

    let fused_grads = (fused * output_weights.clone()).sum().backward();
    let reference_grads = (reference * output_weights).sum().backward();

    let fused_y_grad = y_fused.grad(&fused_grads).expect("fused y grad");
    let reference_y_grad = y_ref.grad(&reference_grads).expect("reference y grad");
    let fused_weight_grad = weight_fused.grad(&fused_grads).expect("fused decoder grad");
    let reference_weight_grad = weight_ref
        .grad(&reference_grads)
        .expect("reference decoder grad");

    assert_close(fused_y_grad, reference_y_grad, 5.0e-4, 5.0e-4);
    assert_close(fused_weight_grad, reference_weight_grad, 5.0e-4, 5.0e-4);
}
