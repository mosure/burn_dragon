use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_ndarray::NdArray;
use burn_vjepa::Vjepa2Model;
use safetensors::SafeTensors;
use std::fs;
use std::path::Path;

type TestBackend = NdArray<f32>;

fn fixture_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("vjepa2_tiny_hf")
}

fn tensor_f32<B: Backend>(
    tensors: &SafeTensors<'_>,
    name: &str,
    device: &B::Device,
) -> Tensor<B, 5> {
    let view = tensors.tensor(name).expect("fixture tensor");
    let shape = view.shape();
    let data: Vec<f32> = view
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
        .collect();
    Tensor::from_data(
        TensorData::new(data, [shape[0], shape[1], shape[2], shape[3], shape[4]]),
        device,
    )
}

fn tensor_f32_3<B: Backend>(
    tensors: &SafeTensors<'_>,
    name: &str,
    device: &B::Device,
) -> Tensor<B, 3> {
    let view = tensors.tensor(name).expect("fixture tensor");
    let shape = view.shape();
    let data: Vec<f32> = view
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
        .collect();
    Tensor::from_data(
        TensorData::new(data, [shape[0], shape[1], shape[2]]),
        device,
    )
}

fn tensor_i64_2<B: Backend>(
    tensors: &SafeTensors<'_>,
    name: &str,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let view = tensors.tensor(name).expect("fixture tensor");
    let shape = view.shape();
    let data: Vec<i64> = view
        .data()
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("i64 chunk")))
        .collect();
    Tensor::from_data(TensorData::new(data, [shape[0], shape[1]]), device)
}

fn max_abs_diff<const D: usize>(
    actual: Tensor<TestBackend, D>,
    expected: Tensor<TestBackend, D>,
) -> f32 {
    let diff = actual
        .sub(expected)
        .abs()
        .into_data()
        .to_vec::<f32>()
        .expect("f32 vec");
    diff.into_iter().fold(0.0f32, f32::max)
}

#[test]
fn native_vjepa2_matches_tiny_hf_fixture() {
    let root = fixture_root();
    assert!(
        root.join("config.json").exists(),
        "missing config.json fixture"
    );
    assert!(
        root.join("model.safetensors").exists(),
        "missing model.safetensors fixture"
    );
    assert!(
        root.join("fixture_outputs.safetensors").exists(),
        "missing fixture_outputs.safetensors fixture"
    );

    let device = Default::default();
    let model =
        Vjepa2Model::<TestBackend>::from_hf_dir(&root, &device).expect("load native V-JEPA2");

    let bytes = fs::read(root.join("fixture_outputs.safetensors")).expect("read fixture outputs");
    let tensors = SafeTensors::deserialize(&bytes).expect("deserialize fixture outputs");

    let pixel_values_videos = tensor_f32::<TestBackend>(&tensors, "pixel_values_videos", &device);
    let context_mask = vec![tensor_i64_2::<TestBackend>(
        &tensors,
        "context_mask_0",
        &device,
    )];
    let target_mask = vec![tensor_i64_2::<TestBackend>(
        &tensors,
        "target_mask_0",
        &device,
    )];

    let actual = model.forward(
        pixel_values_videos,
        Some(context_mask.clone()),
        Some(target_mask.clone()),
        false,
    );

    let expected_last = tensor_f32_3::<TestBackend>(&tensors, "last_hidden_state", &device);
    let expected_masked = tensor_f32_3::<TestBackend>(&tensors, "masked_hidden_state", &device);
    let expected_pred_last =
        tensor_f32_3::<TestBackend>(&tensors, "predictor_last_hidden_state", &device);
    let expected_pred_target =
        tensor_f32_3::<TestBackend>(&tensors, "predictor_target_hidden_state", &device);

    let predictor = actual.predictor_output.expect("predictor output");

    let last_diff = max_abs_diff(actual.last_hidden_state, expected_last);
    let masked_diff = max_abs_diff(actual.masked_hidden_state, expected_masked);
    let pred_last_diff = max_abs_diff(predictor.last_hidden_state, expected_pred_last);
    let pred_target_diff = max_abs_diff(predictor.target_hidden_state, expected_pred_target);

    assert!(
        last_diff <= 1.0e-4,
        "last_hidden_state diff too large: {last_diff}"
    );
    assert!(
        masked_diff <= 1.0e-4,
        "masked_hidden_state diff too large: {masked_diff}"
    );
    assert!(
        pred_last_diff <= 1.0e-4,
        "predictor_last_hidden_state diff too large: {pred_last_diff}"
    );
    assert!(
        pred_target_diff <= 1.0e-4,
        "predictor_target_hidden_state diff too large: {pred_target_diff}"
    );
}
