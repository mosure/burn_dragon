use burn::tensor::{Int, Tensor};
use burn_ndarray::NdArray;
use burn_dragon::api::multimodal;

type Backend = NdArray<f32>;

fn main() {
    let device = <Backend as burn::tensor::backend::Backend>::Device::default();
    let mut config = multimodal::config::VlJepaDragonConfig::default();
    config.vision.embed_dim = 16;
    config.vision.projection_dim = 16;
    config.vision.projection_hidden_dim = 16;
    config.vision.steps = 1;
    config.query_text.n_embd = 16;
    config.target_text.n_embd = 16;
    config.fusion.n_embd = 16;
    config.query_text.n_head = 2;
    config.target_text.n_head = 2;
    config.fusion.n_head = 2;
    config.fusion_dim = 16;
    config.target_dim = 16;

    let model = multimodal::model::VlJepaDragon::<Backend>::new(config, &device);
    let batch = multimodal::data::VisionLanguageTripletBatch {
        vision_x: Tensor::<Backend, 4>::zeros([1, 3, 32, 32], &device),
        query_q_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
        query_q_mask: None,
        target_y_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
        target_y_mask: None,
    };
    let output = model.forward_x_q_y(
        batch,
        model.init_state(),
        multimodal::data::MultimodalStepMode::Observe,
    );
    assert_eq!(output.fusion.predicted_target_embedding.shape().dims::<2>(), [1, 16]);
}
