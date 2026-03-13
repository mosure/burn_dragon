use burn::tensor::{Int, Tensor};
use burn_ndarray::NdArray;
use burn_dragon_multimodal::{
    MultimodalStepMode, VlJepaDragon, VlJepaDragonConfig, VideoLanguageTripletBatch,
    VisionLanguageTripletBatch,
};
use criterion::{Criterion, criterion_group, criterion_main};

type Backend = NdArray<f32>;

fn build_config() -> VlJepaDragonConfig {
    let mut config = VlJepaDragonConfig::default();
    config.vision.embed_dim = 32;
    config.vision.projection_dim = 32;
    config.vision.projection_hidden_dim = 32;
    config.vision.steps = 1;
    config.query_text.n_embd = 32;
    config.target_text.n_embd = 32;
    config.fusion.n_embd = 32;
    config.query_text.n_head = 4;
    config.target_text.n_head = 4;
    config.fusion.n_head = 4;
    config.fusion_dim = 32;
    config.target_dim = 32;
    config
}

fn benchmark_vl_jepa_forward(c: &mut Criterion) {
    let device = <Backend as burn::tensor::backend::Backend>::Device::default();
    let model = VlJepaDragon::<Backend>::new(build_config(), &device);
    let image_batch = VisionLanguageTripletBatch {
        vision_x: Tensor::<Backend, 4>::zeros([4, 3, 32, 32], &device),
        query_q_tokens: Tensor::<Backend, 2, Int>::zeros([4, 12], &device),
        query_q_mask: None,
        target_y_tokens: Tensor::<Backend, 2, Int>::zeros([4, 12], &device),
        target_y_mask: None,
    };
    let video_batch = VideoLanguageTripletBatch {
        video_x: Tensor::<Backend, 5>::zeros([2, 4, 3, 32, 32], &device),
        query_q_tokens: Tensor::<Backend, 2, Int>::zeros([2, 12], &device),
        query_q_mask: None,
        target_y_tokens: Tensor::<Backend, 2, Int>::zeros([2, 12], &device),
        target_y_mask: None,
    };

    c.bench_function("vl_jepa_image_forward", |b| {
        b.iter(|| {
            let _ = model.forward_x_q_y(
                image_batch.clone(),
                model.init_state(),
                MultimodalStepMode::Observe,
            );
        });
    });

    c.bench_function("vl_jepa_video_forward", |b| {
        b.iter(|| {
            let _ = model.forward_video_x_q_y(video_batch.clone(), model.init_state());
        });
    });
}

criterion_group!(benches, benchmark_vl_jepa_forward);
criterion_main!(benches);
