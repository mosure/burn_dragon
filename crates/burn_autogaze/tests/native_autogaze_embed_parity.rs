use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_autogaze::NativeAutoGazeModel;
use burn_ndarray::NdArray;
use safetensors::SafeTensors;
use std::fs;
use std::path::Path;

type TestBackend = NdArray<f32>;

fn fixture_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("autogaze_official_embed")
}

fn tensor_f32_5<B: Backend>(
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
fn native_autogaze_embed_matches_official_fixture() {
    let fixture_root = fixture_root();
    assert!(
        fixture_root.join("fixture_outputs.safetensors").exists(),
        "missing AutoGaze fixture outputs; run scripts/vision/export_autogaze_official_embed_fixture.py"
    );
    let hf_root = Path::new(
        "/home/mosure/.cache/huggingface/hub/models--nvidia--AutoGaze/snapshots/5100fae739ec1bf3f875914fa1b703846a18943a",
    );

    let device = Default::default();
    let model = NativeAutoGazeModel::<TestBackend>::from_hf_dir(hf_root, &device)
        .expect("load native autogaze vision stack");

    let bytes = fs::read(fixture_root.join("fixture_outputs.safetensors")).expect("read fixture");
    let tensors = SafeTensors::deserialize(&bytes).expect("deserialize fixture");
    let video = tensor_f32_5::<TestBackend>(&tensors, "video", &device);
    let expected_embeddings = tensor_f32_3::<TestBackend>(&tensors, "embeddings", &device);

    let (actual_embeddings, _past) = model.gazing_model.embed_video(video, false, None);
    let [batch, time, tokens, dim] = actual_embeddings.shape().dims::<4>();
    let actual_embeddings = actual_embeddings.reshape([batch, time * tokens, dim]);
    let diff = max_abs_diff(actual_embeddings, expected_embeddings);
    assert!(diff <= 6.0e-4, "AutoGaze embed diff too large: {diff}");
}
