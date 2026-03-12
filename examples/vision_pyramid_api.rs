use burn::tensor::{Distribution, Tensor};
use burn_dragon::api::vision::model::{
    VisionBackboneKind, VisionDragon, VisionDragonConfig, VisionPatchEmbedMode,
};
use burn_ndarray::NdArray;

fn main() {
    type Backend = NdArray<f32>;

    let device = <Backend as burn::tensor::backend::Backend>::Device::default();
    let config = VisionDragonConfig {
        image_size: 32,
        patch_size: 8,
        patch_embed_mode: VisionPatchEmbedMode::default(),
        backbone: VisionBackboneKind::Pyramid,
        in_channels: 3,
        embed_dim: 32,
        steps: 2,
        n_head: 4,
        projection_dim: 16,
        projection_hidden_dim: 32,
        ..Default::default()
    };

    let model = VisionDragon::<Backend>::new(config, &device);
    let images = Tensor::<Backend, 4>::random([2, 3, 32, 32], Distribution::Default, &device);
    let output = model.forward_images(images);

    println!(
        "patch_tokens={:?}, cls_token={:?}",
        output.patch_tokens.shape().dims::<3>(),
        output.cls_token.shape().dims::<2>()
    );
}
