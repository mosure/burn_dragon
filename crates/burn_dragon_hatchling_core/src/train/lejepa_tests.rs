use crate::train::prelude::*;
use burn::tensor::Distribution;
use burn_ndarray::NdArray;

#[test]
fn lejepa_invariance_loss_is_finite() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();

    let proj = Tensor::<Backend, 3>::random([2, 4, 8], Distribution::Default, &device);
    let loss = lejepa_invariance_loss(proj);
    let value = loss
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss vec")[0];
    assert!(value.is_finite());
}

#[test]
fn lejepa_sigreg_loss_is_finite() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let config = VisionLejepaConfig::default();

    let proj = Tensor::<Backend, 3>::random([2, 4, 8], Distribution::Default, &device);
    let loss = lejepa_sigreg_loss(proj, &config.loss.lejepa);
    let value = loss
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss vec")[0];
    assert!(value.is_finite());
}

#[test]
fn patchify_roundtrip() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();

    let batch = 1;
    let channels = 3;
    let height = 4;
    let width = 4;
    let patch_size = 2;
    let total = batch * channels * height * width;
    let data: Vec<f32> = (0..total).map(|v| v as f32).collect();

    let images = Tensor::<Backend, 4>::from_data(
        TensorData::new(data.clone(), [batch, channels, height, width]),
        &device,
    );
    let patches = patchify(images.clone(), patch_size);
    let [patch_batch, tokens, patch_dim] = patches.shape().dims::<3>();
    assert_eq!(patch_batch, batch);
    assert_eq!(tokens, (height / patch_size) * (width / patch_size));
    assert_eq!(patch_dim, channels * patch_size * patch_size);

    let recon = unpatchify(patches, patch_size, height, width, channels);
    let out = recon
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("recon vec");
    assert_eq!(data, out);
}

