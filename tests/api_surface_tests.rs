use burn::tensor::{Int, Tensor, TensorData};
use burn_ndarray::NdArray;
use burn_dragon::api::{core, graph, language, multimodal, stream, vision};

type InferBackend = NdArray<f32>;

#[test]
fn root_api_core_language_surface_runs() {
    let device = <InferBackend as burn::tensor::backend::Backend>::Device::default();
    let config = core::config::BDHConfig {
        n_layer: 2,
        n_embd: 16,
        n_head: 2,
        vocab_size: 32,
        dropout: 0.0,
        ..Default::default()
    };
    let model = core::recurrent::BDH::<InferBackend>::new(config, &device);
    let tokens = Tensor::<InferBackend, 2, Int>::from_data(
        TensorData::new(vec![0_i64, 1, 2, 3, 4, 5], [2, 3]),
        &device,
    );
    let logits = model.forward(tokens);
    assert_eq!(logits.shape().dims::<3>(), [2, 3, 32]);

    let _settings = language::inference::GenerationSettings {
        max_new_tokens: Some(4),
        temperature: 1.0,
        top_k: Some(1),
        strategy: language::inference::ContextStrategy::Infinite,
    };
}

#[test]
fn root_api_vision_surface_runs() {
    let device = <InferBackend as burn::tensor::backend::Backend>::Device::default();
    let config = vision::model::VisionDragonConfig {
        image_size: 32,
        patch_size: 8,
        backbone: vision::model::VisionBackboneKind::Dense,
        embed_dim: 16,
        projection_dim: 8,
        projection_hidden_dim: 16,
        steps: 2,
        n_head: 2,
        dropout: 0.0,
        ..Default::default()
    };
    let model = vision::model::VisionDragon::<InferBackend>::new(config, &device);
    let images = Tensor::<InferBackend, 4>::zeros([1, 3, 32, 32], &device);
    let output = model.forward_images(images);
    assert_eq!(output.patch_tokens.shape().dims::<3>(), [1, 16, 8]);
}

#[test]
fn root_api_graph_surface_runs_with_executor() {
    let device = <InferBackend as burn::tensor::backend::Backend>::Device::default();
    let routing = graph::routing::GraphTopologyRouting::new(
        graph::routing::GraphCsrAdjacency::try_from_edges(3, 3, &[(0, 1), (1, 0), (1, 2), (2, 2)])
            .expect("valid node adjacency"),
    )
    .expect("valid routing")
    .with_cluster_assignments(2, &[0, 0, 1])
    .expect("valid cluster assignments")
    .with_node_global_assignments(1, &[0, 0, 0])
    .expect("valid node/global assignments")
    .with_cluster_global_assignments(1, &[0, 0])
    .expect("valid cluster/global assignments");

    let model = graph::execution::GraphDragon::<InferBackend>::new(
        graph::execution::GraphDragonConfig {
            embed_dim: 4,
            rank: 2,
            value_dim: 4,
            ..Default::default()
        },
        &device,
    );

    let node_obs = Tensor::<InferBackend, 3>::from_data(
        TensorData::new(
            vec![
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
            [1, 3, 4],
        ),
        &device,
    );
    let cluster_obs = Tensor::<InferBackend, 3>::from_data(
        TensorData::new(
            vec![
                1.0, 1.0, 0.0, 0.0, //
                0.0, 1.0, 1.0, 0.0,
            ],
            [1, 2, 4],
        ),
        &device,
    );
    let state = model
        .state_from_observations(&routing, node_obs, cluster_obs)
        .expect("state init");
    let executor = graph::execution::GraphCompiledExecutor::<InferBackend>::new(routing, &device);
    let output = model
        .step_with_executor(state, &executor, core::state::StructuredStepMode::Predict)
        .expect("graph step");

    assert_eq!(output.state.node_state().shape().dims::<3>(), [1, 3, 4]);
}

#[cfg(feature = "train")]
#[test]
fn root_api_train_surface_exposes_runtime_config() {
    let _cfg = burn_dragon::api::train::config::WgpuRuntimeConfig::default();
    let _bytes = burn_dragon::api::train::runtime::bytes_to_mb(1024 * 1024);
}

#[cfg(feature = "train")]
#[test]
fn root_api_multimodal_checkpoint_surface_exposes_exporter() {
    let _export_fn = burn_dragon::api::multimodal::checkpoint::export_multimodal_checkpoint_to_burnpack;
    let _default_dir = burn_dragon::api::multimodal::checkpoint::default_checkpoint_dir("runs/example");
}

#[cfg(feature = "train")]
#[test]
fn root_api_multimodal_runtime_surface_exposes_training_entrypoints() {
    let _cfg = burn_dragon::api::multimodal::runtime::MultimodalTrainingConfig::default();
    let _video_cfg = burn_dragon::api::multimodal::runtime::MultimodalVideoTrainingConfig::default();
    let _load = burn_dragon::api::multimodal::runtime::load_multimodal_training_runtime_config;
    let _load_video =
        burn_dragon::api::multimodal::runtime::load_multimodal_video_training_runtime_config;
    let _train = burn_dragon::api::multimodal::runtime::train_backend::<
        burn_autodiff::Autodiff<burn_ndarray::NdArray<f32>>,
        fn(&<burn_autodiff::Autodiff<burn_ndarray::NdArray<f32>> as burn::tensor::backend::Backend>::Device),
    >;
}

#[test]
fn root_api_stream_and_multimodal_surface_runs() {
    let device = <InferBackend as burn::tensor::backend::Backend>::Device::default();
    let window = stream::window::TbpttWindow::new(4, 2);
    assert_eq!(window.detach_prefix_steps(), 2);

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

    let model = multimodal::model::VlJepaDragon::<InferBackend>::new(config, &device);
    let batch = multimodal::data::VisionLanguageTripletBatch {
        vision_x: Tensor::<InferBackend, 4>::zeros([1, 3, 32, 32], &device),
        query_q_tokens: Tensor::<InferBackend, 2, Int>::zeros([1, 4], &device),
        query_q_mask: None,
        target_y_tokens: Tensor::<InferBackend, 2, Int>::zeros([1, 4], &device),
        target_y_mask: None,
    };
    let output = model.forward_x_q_y(
        batch,
        model.init_state(),
        multimodal::data::MultimodalStepMode::Observe,
    );
    assert_eq!(output.fusion.predicted_target_embedding.shape().dims::<2>(), [1, 16]);
}
