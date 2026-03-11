use super::*;

#[test]
fn distill_mode_parses() {
    let text = r#"
            [dataset]
            imagenet_root = "data/imagenet1k"
            train_dir = "train"
            val_dir = "val"

            [training]
            batch_size = 8
            max_iters = 10
            log_frequency = 2
            rollout_min_steps = 2
            rollout_max_steps = 3
            rollout_backprop_steps = 2

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.1

            [vision]
            image_size = 224
            patch_size = 14
            in_channels = 3
            embed_dim = 256
            steps = 4
            n_head = 4
            mlp_internal_dim_multiplier = 4
            dropout = 0.1
            projection_dim = 384
            projection_hidden_dim = 512
            use_cls_token = true
            pos_encoding = "learned2d"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [mode]
            type = "distill"
            rollout_supervision_frames = 3
            rollout_supervision_power = 1.5
            rollout_sampling_power = 1.25

            [mode.teacher]
            type = "features"
            train_cls_path = "train_cls.bin"
            train_patch_path = "train_patch.bin"
            val_cls_path = "val_cls.bin"
            val_patch_path = "val_patch.bin"
            feature_dim = 384
            patch_tokens = 256
        "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse distill config");
    assert_eq!(config.training.rollout_min_steps, Some(2));
    assert_eq!(config.training.rollout_max_steps, Some(3));
    assert_eq!(config.training.rollout_backprop_steps, Some(2));
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => match distill.teacher {
            VisionTeacherConfig::Features(teacher) => {
                assert_eq!(teacher.feature_dim, 384);
                assert_eq!(teacher.patch_tokens, Some(256));
                assert_eq!(distill.rollout_supervision_frames, 3);
                assert!((distill.rollout_supervision_power - 1.5).abs() < f32::EPSILON);
                assert!((distill.rollout_sampling_power - 1.25).abs() < f32::EPSILON);
            }
            other => panic!("unexpected teacher config: {other:?}"),
        },
        other => panic!("unexpected mode: {other:?}"),
    }
}

#[test]
fn distill_rejects_negative_rollout_sampling_power() {
    let text = r#"
            [dataset]
            imagenet_root = "data/imagenette2-160"
            train_dir = "train"
            val_dir = "val"

            [training]
            batch_size = 8
            max_iters = 10
            log_frequency = 2

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.1

            [vision]
            image_size = 224
            patch_size = 14
            in_channels = 3
            embed_dim = 256
            steps = 4
            n_head = 4
            mlp_internal_dim_multiplier = 4
            dropout = 0.1
            projection_dim = 384
            projection_hidden_dim = 512
            use_cls_token = true
            pos_encoding = "rope"
            attention_mode = "row_l1"

            [mode]
            type = "distill"
            rollout_sampling_power = -0.1

            [mode.teacher]
            type = "features"
            train_cls_path = "train_cls.bin"
            train_patch_path = "train_patch.bin"
            val_cls_path = "val_cls.bin"
            val_patch_path = "val_patch.bin"
            feature_dim = 384
            patch_tokens = 256

            [augment]
            image_size = 224
            resize_short = 256
            min_scale = 1.0
            max_scale = 1.0
            min_aspect_ratio = 1.0
            max_aspect_ratio = 1.0
            flip_prob = 0.0
            color_jitter_prob = 0.0
            brightness = 0.0
            contrast = 0.0
            saturation = 0.0
            hue = 0.0
            grayscale_prob = 0.0
            blur_prob = 0.0
            blur_sigma_min = 0.1
            blur_sigma_max = 2.0
            solarize_prob = 0.0
            solarize_threshold = 128
        "#;
    let config: VisionTrainingConfig = toml::from_str(text).expect("parse config");

    let err = config
        .validate()
        .expect_err("negative rollout sampling power should fail");
    assert!(
        err.to_string()
            .contains("mode.rollout_sampling_power must be >= 0")
    );
}

#[test]
fn distill_feature_teacher_requires_deterministic_train_augmentations() {
    let text = r#"
            [dataset]
            imagenet_root = "data/imagenette2-160"
            train_dir = "train"
            val_dir = "val"

            [training]
            batch_size = 8
            max_iters = 10
            log_frequency = 2

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.1

            [vision]
            image_size = 224
            patch_size = 14
            in_channels = 3
            embed_dim = 256
            steps = 4
            n_head = 4
            mlp_internal_dim_multiplier = 4
            dropout = 0.1
            projection_dim = 384
            projection_hidden_dim = 512
            use_cls_token = true
            pos_encoding = "rope"
            attention_mode = "row_l1"

            [mode]
            type = "distill"

            [mode.teacher]
            type = "features"
            train_cls_path = "train_cls.bin"
            train_patch_path = "train_patch.bin"
            val_cls_path = "val_cls.bin"
            val_patch_path = "val_patch.bin"
            feature_dim = 384
            patch_tokens = 256

            [augment]
            image_size = 224
            resize_short = 256
            min_scale = 0.08
            max_scale = 1.0
            min_aspect_ratio = 0.75
            max_aspect_ratio = 1.3333334
            flip_prob = 0.5
            color_jitter_prob = 0.0
            brightness = 0.0
            contrast = 0.0
            saturation = 0.0
            hue = 0.0
            grayscale_prob = 0.0
            blur_prob = 0.0
            blur_sigma_min = 0.1
            blur_sigma_max = 2.0
            solarize_prob = 0.0
            solarize_threshold = 128
        "#;
    let config: VisionTrainingConfig = toml::from_str(text).expect("parse config");

    let err = config
        .validate()
        .expect_err("expected stochastic feature distill validation error");
    assert!(
        err.to_string()
            .contains("requires deterministic train augmentations")
    );
}

#[test]
fn rope_positional_encoding_parses() {
    let text = r#"
            [dataset]
            source = "moving_mnist"

            [training]
            batch_size = 8
            max_iters = 4
            log_frequency = 1

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.01

            [vision]
            image_size = 32
            patch_size = 4
            in_channels = 3
            embed_dim = 64
            steps = 2
            n_head = 8
            mlp_internal_dim_multiplier = 2
            projection_dim = 32
            projection_hidden_dim = 64
            use_cls_token = true
            pos_encoding = "rope"

            [mode]
            type = "video_lejepa"
        "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse rope config");
    assert_eq!(config.vision.pos_encoding.to_string(), "Rope");
}

#[test]
fn lejepa_mode_parses() {
    let text = r#"
            [dataset]
            imagenet_root = "data/imagenet1k"
            train_dir = "train"
            val_dir = "val"

            [training]
            batch_size = 8
            max_iters = 10
            log_frequency = 2
            rollout_min_steps = 1
            rollout_max_steps = 4
            rollout_backprop_steps = 1

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.1

            [vision]
            image_size = 224
            patch_size = 14
            in_channels = 3
            embed_dim = 256
            steps = 4
            n_head = 4
            mlp_internal_dim_multiplier = 4
            dropout = 0.1
            projection_dim = 384
            projection_hidden_dim = 512
            use_cls_token = true
            pos_encoding = "learned2d"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [mode]
            type = "lejepa"
            views = 4
            global_views = 2
            local_views = 6
            local_image_size = 96
            local_min_scale = 0.05
            local_max_scale = 0.3
            min_view_overlap = 0.2
            view_overlap_attempts = 7
            artifact_output = "avi"
            artifact_fps = 6
            artifact_every = 5
            artifact_max_images = 3
            artifact_max_views = 2
            artifact_rollout_steps = 9
            artifact_rollout_frames = 5
            artifact_overwrite = true

            [mode.teacher_ema]
            enabled = true
            decay = 0.992

            [mode.loss.lejepa]
            enabled = true
            lambda = 0.05
            sigreg_knots = 19
            sigreg_t_max = 2.5
            sigreg_proj_dim = 128

            [mode.loss.recon]
            weight = 0.7
            mask_ratio = 0.6
            hidden_dim = 192
        "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse lejepa config");
    assert_eq!(config.training.rollout_min_steps, Some(1));
    assert_eq!(config.training.rollout_max_steps, Some(4));
    assert_eq!(config.training.rollout_backprop_steps, Some(1));
    match config.mode {
        VisionTrainingModeConfig::Lejepa(lejepa) => {
            assert!(lejepa.teacher_ema.enabled);
            assert!((lejepa.teacher_ema.decay - 0.992).abs() < f32::EPSILON);
            assert!(lejepa.loss.lejepa.enabled);
            assert!((lejepa.loss.lejepa.lambda - 0.05).abs() < f32::EPSILON);
            assert_eq!(lejepa.loss.lejepa.sigreg_knots, 19);
            assert!((lejepa.loss.lejepa.sigreg_t_max - 2.5).abs() < f32::EPSILON);
            assert_eq!(lejepa.loss.lejepa.sigreg_proj_dim, 128);
            assert!((lejepa.loss.recon.weight - 0.7).abs() < f32::EPSILON);
            assert!((lejepa.loss.recon.mask_ratio - 0.6).abs() < f32::EPSILON);
            assert_eq!(lejepa.loss.recon.hidden_dim, 192);
            assert_eq!(lejepa.views, 4);
            assert_eq!(lejepa.global_views, 2);
            assert_eq!(lejepa.local_views, 6);
            assert_eq!(lejepa.local_image_size, 96);
            assert!((lejepa.local_min_scale - 0.05).abs() < f32::EPSILON);
            assert!((lejepa.local_max_scale - 0.3).abs() < f32::EPSILON);
            assert!((lejepa.min_view_overlap - 0.2).abs() < f32::EPSILON);
            assert_eq!(lejepa.view_overlap_attempts, 7);
            assert_eq!(lejepa.artifact_output, VisionArtifactOutputMode::Avi);
            assert_eq!(lejepa.artifact_fps, 6);
            assert_eq!(lejepa.artifact_every, 5);
            assert_eq!(lejepa.artifact_max_images, 3);
            assert_eq!(lejepa.artifact_max_views, 2);
            assert_eq!(lejepa.artifact_rollout_steps, 9);
            assert_eq!(lejepa.artifact_rollout_frames, 5);
            assert!(lejepa.artifact_overwrite);
        }
        other => panic!("unexpected mode: {other:?}"),
    }
}

#[test]
fn video_lejepa_mode_parses() {
    let text = r#"
            [dataset]
            source = "moving_mnist"

            [dataset.moving_mnist]
            digit_size = 18
            min_velocity = 0.9
            max_velocity = 1.8
            train_seed = 11
            val_seed = 22

            [training]
            batch_size = 4
            max_iters = 10
            log_frequency = 2
            rollout_min_steps = 1
            rollout_max_steps = 3
            rollout_backprop_steps = 2

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.1

            [vision]
            image_size = 32
            patch_size = 4
            in_channels = 3
            embed_dim = 32
            steps = 3
            n_head = 4
            mlp_internal_dim_multiplier = 2
            dropout = 0.0
            projection_dim = 16
            projection_hidden_dim = 32
            use_cls_token = true
            pos_encoding = "learned2d"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [mode]
            type = "video_lejepa"
            context_frames = 4
            target_frames = 2
            train_target_frames_min = 2
            train_target_frames_max = 6
            train_target_warmup_steps = 100
            frame_stride = 1
            predictor_hidden_dim = 24
            artifact_output = "avi"
            artifact_fps = 8
            artifact_every = 3
            artifact_max_images = 2
            artifact_future_frames = 12
            artifact_upscale = 5
            artifact_overwrite = true

            [mode.teacher_ema]
            enabled = true
            decay = 0.994

            [mode.temporal]
            n_layer = 2
            n_head = 4
            mlp_internal_dim_multiplier = 2
            rollout_fast_steps_per_slow_step = 4
            predict_backprop_frames = 3
            mode_embeddings = true
            refine_passes = 2
            fused = true
            wgpu_recurrent_kernel = true
            wgpu_rollout_fused = true
            latent_block_size = 8
            time_block_size = 8

            [mode.loss]
            prediction_weight = 1.0
            observe_weight = 0.7
            cosine_weight = 0.2
            probe_weight = 0.5

            [mode.loss.sigreg]
            enabled = true
            lambda = 0.03
            sigreg_knots = 9
            sigreg_t_max = 2.0
            sigreg_proj_dim = 16
        "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse video lejepa config");
    match config.mode {
        VisionTrainingModeConfig::VideoLejepa(video) => {
            assert_eq!(config.dataset.source, VisionDatasetSource::MovingMnist);
            assert_eq!(config.dataset.moving_mnist.digit_size, 18);
            assert!(video.teacher_ema.enabled);
            assert!((video.teacher_ema.decay - 0.994).abs() < f32::EPSILON);
            assert_eq!(video.context_frames, 4);
            assert_eq!(video.target_frames, 2);
            assert_eq!(video.train_target_frames_min, 2);
            assert_eq!(video.train_target_frames_max, 6);
            assert_eq!(video.train_target_warmup_steps, 100);
            assert_eq!(video.temporal.rollout_fast_steps_per_slow_step, 4);
            assert_eq!(video.temporal.predict_backprop_frames, 3);
            assert!(video.temporal.mode_embeddings);
            assert_eq!(video.temporal.refine_passes, 2);
            assert!(video.temporal.fused);
            assert!(video.temporal.wgpu_rollout_fused);
            assert_eq!(video.temporal.latent_block_size, 8);
            assert!((video.loss.observe_weight - 0.7).abs() < f32::EPSILON);
            assert!((video.loss.cosine_weight - 0.2).abs() < f32::EPSILON);
            assert!((video.loss.debug_recon_weight - 1.0).abs() < f32::EPSILON);
            assert_eq!(video.loss.debug_recon_hidden_dim, 256);
            assert!((video.loss.sigreg.lambda - 0.03).abs() < f32::EPSILON);
            assert_eq!(video.artifact_output, VisionArtifactOutputMode::Avi);
            assert_eq!(video.artifact_future_frames, 12);
            assert_eq!(video.artifact_upscale, 5);
        }
        other => panic!("unexpected mode: {other:?}"),
    }
}

#[test]
fn video_lejepa_requires_moving_mnist_source() {
    let mut config = VisionTrainingConfig {
        dataset: VisionDatasetConfig::default(),
        training: VisionTrainingHyperparameters::default(),
        optimizer: burn_dragon_train::OptimizerConfig {
            learning_rate: 1e-3,
            weight_decay: 0.0,
            lr_schedule: None,
            grad_clip_norm: None,
            grad_clip_value: None,
        },
        wgpu: burn_dragon_train::WgpuRuntimeConfig::default(),
        vision: VisionModelConfig::default(),
        augment: VisionAugmentationConfig::default(),
        mode: VisionTrainingModeConfig::VideoLejepa(VisionVideoLejepaConfig::default()),
    };
    config.dataset.source = VisionDatasetSource::Imagenet;
    config.vision.image_size = 32;
    config.vision.patch_size = 4;
    config.vision.in_channels = 3;
    config.vision.embed_dim = 32;
    config.vision.steps = 2;
    config.vision.n_head = 4;
    config.vision.mlp_internal_dim_multiplier = 2;
    config.vision.projection_dim = 16;
    config.vision.projection_hidden_dim = 32;
    config.vision.use_cls_token = true;
    config.augment.image_size = 32;
    config.training.batch_size = 2;
    config.training.max_iters = 2;

    let err = config.validate().expect_err("expected validation error");
    assert!(err.to_string().contains("moving_mnist"));
}

#[test]
fn video_lejepa_artifact_future_must_cover_training_target() {
    let mut config = VisionTrainingConfig {
        dataset: VisionDatasetConfig::default(),
        training: VisionTrainingHyperparameters::default(),
        optimizer: burn_dragon_train::OptimizerConfig {
            learning_rate: 1e-3,
            weight_decay: 0.0,
            lr_schedule: None,
            grad_clip_norm: None,
            grad_clip_value: None,
        },
        wgpu: burn_dragon_train::WgpuRuntimeConfig::default(),
        vision: VisionModelConfig::default(),
        augment: VisionAugmentationConfig::default(),
        mode: VisionTrainingModeConfig::VideoLejepa(VisionVideoLejepaConfig::default()),
    };
    config.dataset.source = VisionDatasetSource::MovingMnist;
    config.vision.image_size = 32;
    config.vision.patch_size = 4;
    config.vision.in_channels = 3;
    config.vision.embed_dim = 32;
    config.vision.steps = 2;
    config.vision.n_head = 4;
    config.vision.mlp_internal_dim_multiplier = 2;
    config.vision.projection_dim = 16;
    config.vision.projection_hidden_dim = 32;
    config.vision.use_cls_token = true;
    config.augment.image_size = 32;
    config.training.batch_size = 2;
    config.training.max_iters = 2;
    if let VisionTrainingModeConfig::VideoLejepa(video) = &mut config.mode {
        video.target_frames = 4;
        video.artifact_future_frames = 3;
    }

    let err = config
        .validate()
        .expect_err("expected artifact future validation error");
    assert!(err.to_string().contains("artifact_future_frames"));
}

#[test]
fn video_lejepa_train_target_frame_range_must_be_ordered() {
    let mut config = VisionTrainingConfig {
        dataset: VisionDatasetConfig::default(),
        training: VisionTrainingHyperparameters::default(),
        optimizer: burn_dragon_train::OptimizerConfig {
            learning_rate: 1e-3,
            weight_decay: 0.0,
            lr_schedule: None,
            grad_clip_norm: None,
            grad_clip_value: None,
        },
        wgpu: burn_dragon_train::WgpuRuntimeConfig::default(),
        vision: VisionModelConfig::default(),
        augment: VisionAugmentationConfig::default(),
        mode: VisionTrainingModeConfig::VideoLejepa(VisionVideoLejepaConfig::default()),
    };
    config.dataset.source = VisionDatasetSource::MovingMnist;
    config.vision.image_size = 32;
    config.vision.patch_size = 4;
    config.vision.in_channels = 3;
    config.vision.embed_dim = 32;
    config.vision.steps = 2;
    config.vision.n_head = 4;
    config.vision.mlp_internal_dim_multiplier = 2;
    config.vision.projection_dim = 16;
    config.vision.projection_hidden_dim = 32;
    config.vision.use_cls_token = true;
    config.augment.image_size = 32;
    config.training.batch_size = 2;
    config.training.max_iters = 2;
    if let VisionTrainingModeConfig::VideoLejepa(video) = &mut config.mode {
        video.train_target_frames_min = 5;
        video.train_target_frames_max = 3;
    }

    let err = config
        .validate()
        .expect_err("expected train target frame range validation error");
    assert!(err.to_string().contains("train_target_frames_max"));
}

#[test]
fn video_lejepa_observe_weight_must_be_non_negative() {
    let mut config = VisionTrainingConfig {
        dataset: VisionDatasetConfig::default(),
        training: VisionTrainingHyperparameters::default(),
        optimizer: burn_dragon_train::OptimizerConfig {
            learning_rate: 1e-3,
            weight_decay: 0.0,
            lr_schedule: None,
            grad_clip_norm: None,
            grad_clip_value: None,
        },
        wgpu: burn_dragon_train::WgpuRuntimeConfig::default(),
        vision: VisionModelConfig::default(),
        augment: VisionAugmentationConfig::default(),
        mode: VisionTrainingModeConfig::VideoLejepa(VisionVideoLejepaConfig::default()),
    };
    config.dataset.source = VisionDatasetSource::MovingMnist;
    config.vision.image_size = 32;
    config.vision.patch_size = 4;
    config.vision.in_channels = 3;
    config.vision.embed_dim = 32;
    config.vision.steps = 2;
    config.vision.n_head = 4;
    config.vision.mlp_internal_dim_multiplier = 2;
    config.vision.projection_dim = 16;
    config.vision.projection_hidden_dim = 32;
    config.vision.use_cls_token = true;
    config.augment.image_size = 32;
    config.training.batch_size = 2;
    config.training.max_iters = 2;
    if let VisionTrainingModeConfig::VideoLejepa(video) = &mut config.mode {
        video.loss.observe_weight = -0.1;
    }

    let err = config
        .validate()
        .expect_err("expected observe weight validation error");
    assert!(err.to_string().contains("observe_weight"));
}

#[test]
fn lejepa_teacher_ema_decay_must_be_below_one() {
    let mut config = VisionTrainingConfig {
        dataset: VisionDatasetConfig::default(),
        training: VisionTrainingHyperparameters::default(),
        optimizer: burn_dragon_train::OptimizerConfig {
            learning_rate: 1e-3,
            weight_decay: 0.0,
            lr_schedule: None,
            grad_clip_norm: None,
            grad_clip_value: None,
        },
        wgpu: burn_dragon_train::WgpuRuntimeConfig::default(),
        vision: VisionModelConfig::default(),
        augment: VisionAugmentationConfig::default(),
        mode: VisionTrainingModeConfig::Lejepa(VisionLejepaConfig::default()),
    };
    config.dataset.source = VisionDatasetSource::Imagenet;
    config.vision.image_size = 32;
    config.vision.patch_size = 4;
    config.vision.in_channels = 3;
    config.vision.embed_dim = 32;
    config.vision.steps = 2;
    config.vision.n_head = 4;
    config.vision.mlp_internal_dim_multiplier = 2;
    config.vision.projection_dim = 16;
    config.vision.projection_hidden_dim = 32;
    config.vision.use_cls_token = true;
    config.augment.image_size = 32;
    config.training.batch_size = 2;
    config.training.max_iters = 2;
    if let VisionTrainingModeConfig::Lejepa(lejepa) = &mut config.mode {
        lejepa.teacher_ema.decay = 1.0;
    }

    let err = config
        .validate()
        .expect_err("expected teacher ema validation error");
    assert!(err.to_string().contains("mode.teacher_ema.decay"));
}

#[test]
fn mae_mode_parses() {
    let text = r#"
            [dataset]
            imagenet_root = "data/imagenet1k"
            train_dir = "train"
            val_dir = "val"

            [training]
            batch_size = 8
            max_iters = 10
            log_frequency = 2

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.1

            [vision]
            image_size = 224
            patch_size = 14
            in_channels = 3
            embed_dim = 256
            steps = 4
            n_head = 4
            mlp_internal_dim_multiplier = 4
            dropout = 0.1
            projection_dim = 384
            projection_hidden_dim = 512
            use_cls_token = true
            pos_encoding = "learned2d"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [mode]
            type = "mae"
            pyramid_levels = 2
            artifact_output = "images"
            artifact_fps = 5
            artifact_every = 3
            artifact_max_images = 2
            artifact_max_views = 1
            artifact_rollout_steps = 7
            artifact_rollout_frames = 4
            artifact_overwrite = true

            [mode.loss.recon]
            weight = 1.2
            mask_ratio = 0.8
            hidden_dim = 192
        "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse mae config");
    match config.mode {
        VisionTrainingModeConfig::Mae(mae) => {
            assert!((mae.loss.recon.mask_ratio - 0.8).abs() < f32::EPSILON);
            assert!((mae.loss.recon.weight - 1.2).abs() < f32::EPSILON);
            assert_eq!(mae.loss.recon.hidden_dim, 192);
            assert_eq!(mae.pyramid_levels, 2);
            assert_eq!(mae.artifact_output, VisionArtifactOutputMode::Images);
            assert_eq!(mae.artifact_fps, 5);
            assert_eq!(mae.artifact_every, 3);
            assert_eq!(mae.artifact_max_images, 2);
            assert_eq!(mae.artifact_max_views, 1);
            assert_eq!(mae.artifact_rollout_steps, 7);
            assert_eq!(mae.artifact_rollout_frames, 4);
            assert!(mae.artifact_overwrite);
        }
        other => panic!("unexpected mode: {other:?}"),
    }
}

#[test]
fn saccade_mode_parses() {
    let text = r#"
            [dataset]
            imagenet_root = "data/imagenet1k"
            train_dir = "train"
            val_dir = "val"

            [training]
            batch_size = 8
            max_iters = 10
            log_frequency = 2

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.1

            [vision]
            image_size = 224
            patch_size = 14
            in_channels = 3
            embed_dim = 256
            steps = 4
            n_head = 4
            mlp_internal_dim_multiplier = 4
            dropout = 0.1
            projection_dim = 384
            projection_hidden_dim = 512
            use_cls_token = true
            num_eyes = 2
            pos_encoding = "learned2d"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [mode]
            type = "saccade"
            mip_levels = 4
            pyramid_mode = "laplacian"
            fovea_sampling_mode = "subpatch"
            fovea_warp_mode = "patched"
            fovea_subpatch_size = 12
            inner_steps = 2
            artifact_output = "avi"
            artifact_fps = 7
            artifact_every = 4
            artifact_max_images = 3
            artifact_max_views = 2
            artifact_overwrite = false

            [mode.loss.lejepa]
            enabled = true
            lambda = 0.05
            sigreg_knots = 9
            sigreg_t_max = 2.0
            sigreg_proj_dim = 192

            [mode.loss.recon]
            weight = 0.9
            mask_ratio = 0.7
            hidden_dim = 320
        "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse saccade config");
    assert_eq!(config.vision.num_eyes, 2);
    match config.mode {
        VisionTrainingModeConfig::Saccade(saccade) => {
            assert_eq!(saccade.num_eyes, 0);
            assert_eq!(saccade.mip_levels, 4);
            assert_eq!(saccade.pyramid_mode, VisionPyramidMode::Laplacian);
            assert_eq!(
                saccade.fovea_sampling_mode,
                VisionFoveaSamplingMode::Subpatch
            );
            assert_eq!(saccade.fovea_warp_mode, VisionFoveaWarpMode::Patched);
            assert_eq!(saccade.fovea_subpatch_size, 12);
            assert_eq!(saccade.inner_steps, 2);
            assert!(saccade.loss.lejepa.enabled);
            assert!((saccade.loss.lejepa.lambda - 0.05).abs() < f32::EPSILON);
            assert_eq!(saccade.loss.lejepa.sigreg_knots, 9);
            assert!((saccade.loss.lejepa.sigreg_t_max - 2.0).abs() < f32::EPSILON);
            assert_eq!(saccade.loss.lejepa.sigreg_proj_dim, 192);
            assert!((saccade.loss.recon.weight - 0.9).abs() < f32::EPSILON);
            assert!((saccade.loss.recon.mask_ratio - 0.7).abs() < f32::EPSILON);
            assert_eq!(saccade.loss.recon.hidden_dim, 320);
            assert_eq!(saccade.artifact_output, VisionArtifactOutputMode::Avi);
            assert_eq!(saccade.artifact_fps, 7);
            assert_eq!(saccade.artifact_every, 4);
            assert_eq!(saccade.artifact_max_images, 3);
            assert_eq!(saccade.artifact_max_views, 2);
            assert!(!saccade.artifact_overwrite);
        }
        other => panic!("unexpected mode: {other:?}"),
    }
}

#[test]
fn pyramid_strict_rejects_local_grid_mismatch() {
    let text = r#"
            [dataset]
            imagenet_root = "data/imagenet1k"
            train_dir = "train"
            val_dir = "val"

            [training]
            batch_size = 8
            max_iters = 10
            log_frequency = 2

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.1

            [vision]
            image_size = 224
            patch_size = 14
            in_channels = 3
            embed_dim = 256
            steps = 4
            n_head = 4
            mlp_internal_dim_multiplier = 4
            dropout = 0.1
            projection_dim = 384
            projection_hidden_dim = 512
            use_cls_token = true
            pos_encoding = "learned2d"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [vision.trm_graph]
            enabled = true
            coarse_stride = 1
            rank = 4
            value_dim = 32
            local_radius = 1
            hub_count = 1
            local_diagonals = false
            local_self = false
            decay = 0.9
            hub_gates = true
            grid_mismatch_policy = "error"

            [mode]
            type = "lejepa"
            views = 4
            global_views = 2
            local_views = 2
            local_image_size = 96
            local_min_scale = 0.05
            local_max_scale = 0.3
            min_view_overlap = 0.2
            view_overlap_attempts = 7

            [mode.loss.lejepa]
            enabled = true
            lambda = 0.05
            sigreg_knots = 19
            sigreg_t_max = 2.5
            sigreg_proj_dim = 128

            [mode.loss.recon]
            weight = 0.7
            mask_ratio = 0.6
            hidden_dim = 192
        "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse config");
    let err = config
        .validate()
        .expect_err("strict pyramid backbone must reject mismatch");
    let message = format!("{err:#}");
    assert!(message.contains("pyramid backbone strict mode requires local view grid"));
}

#[test]
fn pyramid_fallback_allows_local_grid_mismatch() {
    let text = r#"
            [dataset]
            imagenet_root = "data/imagenet1k"
            train_dir = "train"
            val_dir = "val"

            [training]
            batch_size = 8
            max_iters = 10
            log_frequency = 2

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.1

            [vision]
            image_size = 224
            patch_size = 14
            in_channels = 3
            embed_dim = 256
            steps = 4
            n_head = 4
            mlp_internal_dim_multiplier = 4
            dropout = 0.1
            projection_dim = 384
            projection_hidden_dim = 512
            use_cls_token = true
            pos_encoding = "learned2d"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [vision.trm_graph]
            enabled = true
            coarse_stride = 1
            rank = 4
            value_dim = 32
            local_radius = 1
            hub_count = 1
            local_diagonals = false
            local_self = false
            decay = 0.9
            hub_gates = true
            grid_mismatch_policy = "fallback_default"

            [mode]
            type = "lejepa"
            views = 4
            global_views = 2
            local_views = 2
            local_image_size = 96
            local_min_scale = 0.05
            local_max_scale = 0.3
            min_view_overlap = 0.2
            view_overlap_attempts = 7

            [mode.loss.lejepa]
            enabled = true
            lambda = 0.05
            sigreg_knots = 19
            sigreg_t_max = 2.5
            sigreg_proj_dim = 128

            [mode.loss.recon]
            weight = 0.7
            mask_ratio = 0.6
            hidden_dim = 192
        "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse config");
    config
        .validate()
        .expect("explicit fallback should allow mismatch");
}

#[test]
fn pyramid_backbone_parses_without_legacy_enabled_flag() {
    let text = r#"
            [dataset]
            imagenet_root = "data/imagenet1k"
            train_dir = "train"
            val_dir = "val"

            [training]
            batch_size = 8
            max_iters = 10
            log_frequency = 2

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.1

            [vision]
            image_size = 224
            patch_size = 14
            backbone = "pyramid"
            in_channels = 3
            embed_dim = 256
            steps = 4
            n_head = 4
            mlp_internal_dim_multiplier = 4
            dropout = 0.1
            projection_dim = 384
            projection_hidden_dim = 512
            use_cls_token = true
            pos_encoding = "learned2d"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [vision.trm_graph]
            coarse_stride = 1
            rank = 4
            value_dim = 32
            local_radius = 1
            hub_count = 1
            local_diagonals = false
            local_self = false
            decay = 0.9
            hub_gates = true
            grid_mismatch_policy = "fallback_default"

            [mode]
            type = "lejepa"
            views = 2
            global_views = 2
            local_views = 0

            [mode.loss.lejepa]
            enabled = true
            lambda = 0.05
            sigreg_knots = 19
            sigreg_t_max = 2.5
            sigreg_proj_dim = 128

            [mode.loss.recon]
            weight = 0.7
            mask_ratio = 0.6
            hidden_dim = 192
        "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse pyramid config");
    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Pyramid
    );
    config.validate().expect("pyramid backbone should validate");
}

#[test]
fn cellular_backbone_parses_without_legacy_enabled_flag() {
    let text = r#"
            [dataset]
            imagenet_root = "data/imagenet1k"
            train_dir = "train"
            val_dir = "val"

            [training]
            batch_size = 8
            max_iters = 10
            log_frequency = 2

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.1

            [vision]
            image_size = 224
            patch_size = 14
            backbone = "cellular"
            in_channels = 3
            embed_dim = 256
            steps = 4
            n_head = 4
            mlp_internal_dim_multiplier = 4
            dropout = 0.1
            projection_dim = 384
            projection_hidden_dim = 512
            use_cls_token = true
            pos_encoding = "learned2d"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [vision.rho_stream]
            local_radius = 2
            local_diagonals = false
            local_self = true
            decay = 0.85
            wgpu_forward_kernel = true
            wgpu_rollout_fused = true

            [mode]
            type = "lejepa"
            views = 2
            global_views = 2
            local_views = 0

            [mode.loss.lejepa]
            enabled = true
            lambda = 0.05
            sigreg_knots = 19
            sigreg_t_max = 2.5
            sigreg_proj_dim = 128

            [mode.loss.recon]
            weight = 0.7
            mask_ratio = 0.6
            hidden_dim = 192
        "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse cellular config");
    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Cellular
    );
    config
        .validate()
        .expect("cellular backbone config should validate");
}

#[test]
fn explicit_dense_backbone_rejects_legacy_enabled_backbone_flags() {
    let text = r#"
            [dataset]
            imagenet_root = "data/imagenet1k"
            train_dir = "train"
            val_dir = "val"

            [training]
            batch_size = 8
            max_iters = 10
            log_frequency = 2

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.1

            [vision]
            image_size = 224
            patch_size = 14
            backbone = "dense"
            in_channels = 3
            embed_dim = 256
            steps = 4
            n_head = 4
            mlp_internal_dim_multiplier = 4
            dropout = 0.1
            projection_dim = 384
            projection_hidden_dim = 512
            use_cls_token = true
            pos_encoding = "learned2d"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [vision.trm_graph]
            enabled = true
            coarse_stride = 1
            rank = 4
            value_dim = 32
            local_radius = 1
            hub_count = 1
            local_diagonals = false
            local_self = false
            decay = 0.9
            hub_gates = true
            grid_mismatch_policy = "fallback_default"

            [mode]
            type = "lejepa"
            views = 2
            global_views = 2
            local_views = 0

            [mode.loss.lejepa]
            enabled = true
            lambda = 0.05
            sigreg_knots = 19
            sigreg_t_max = 2.5
            sigreg_proj_dim = 128

            [mode.loss.recon]
            weight = 0.7
            mask_ratio = 0.6
            hidden_dim = 192
        "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse dense-conflict config");
    let err = config
        .validate()
        .expect_err("explicit dense backbone should reject legacy enabled flags");
    let message = format!("{err:#}");
    assert!(message.contains("vision.backbone = \"dense\" conflicts"));
}

#[test]
fn rho_stream_config_parses_and_validates() {
    let text = r#"
            [dataset]
            imagenet_root = "data/imagenet1k"
            train_dir = "train"
            val_dir = "val"

            [training]
            batch_size = 8
            max_iters = 10
            log_frequency = 2

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.1

            [vision]
            image_size = 224
            patch_size = 14
            in_channels = 3
            embed_dim = 256
            steps = 4
            n_head = 4
            mlp_internal_dim_multiplier = 4
            dropout = 0.1
            projection_dim = 384
            projection_hidden_dim = 512
            use_cls_token = true
            pos_encoding = "learned2d"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [vision.rho_stream]
            enabled = true
            local_radius = 2
            local_diagonals = false
            local_self = true
            decay = 0.85
            wgpu_forward_kernel = true
            wgpu_rollout_fused = true

            [mode]
            type = "lejepa"
            views = 2
            global_views = 2
            local_views = 0
            local_image_size = 224
            local_min_scale = 0.2
            local_max_scale = 0.2
            min_view_overlap = 0.0
            view_overlap_attempts = 1

            [mode.loss.lejepa]
            enabled = true
            lambda = 0.05
            sigreg_knots = 19
            sigreg_t_max = 2.5
            sigreg_proj_dim = 128

            [mode.loss.recon]
            weight = 0.7
            mask_ratio = 0.6
            hidden_dim = 192
        "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse config");
    assert!(config.vision.rho_stream.enabled);
    assert_eq!(config.vision.rho_stream.local_radius, 2);
    assert!(!config.vision.rho_stream.local_diagonals);
    assert!(config.vision.rho_stream.local_self);
    assert!((config.vision.rho_stream.decay - 0.85).abs() < f32::EPSILON);
    assert!(config.vision.rho_stream.wgpu_forward_kernel);
    assert!(config.vision.rho_stream.wgpu_rollout_fused);
    config
        .validate()
        .expect("rho_stream config should validate");
}

#[test]
fn rho_stream_rejects_trm_graph_overlap() {
    let text = r#"
            [dataset]
            imagenet_root = "data/imagenet1k"
            train_dir = "train"
            val_dir = "val"

            [training]
            batch_size = 8
            max_iters = 10
            log_frequency = 2

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.1

            [vision]
            image_size = 224
            patch_size = 14
            in_channels = 3
            embed_dim = 256
            steps = 4
            n_head = 4
            mlp_internal_dim_multiplier = 4
            dropout = 0.1
            projection_dim = 384
            projection_hidden_dim = 512
            use_cls_token = true
            pos_encoding = "learned2d"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [vision.trm_graph]
            enabled = true
            coarse_stride = 1
            rank = 4
            value_dim = 32
            local_radius = 1
            hub_count = 1
            local_diagonals = false
            local_self = false
            decay = 0.9
            hub_gates = true
            grid_mismatch_policy = "fallback_default"

            [vision.rho_stream]
            enabled = true
            local_radius = 1
            local_diagonals = true
            local_self = true
            decay = 0.9

            [mode]
            type = "lejepa"
            views = 2
            global_views = 2
            local_views = 0
            local_image_size = 224
            local_min_scale = 0.2
            local_max_scale = 0.2
            min_view_overlap = 0.0
            view_overlap_attempts = 1

            [mode.loss.lejepa]
            enabled = true
            lambda = 0.05
            sigreg_knots = 19
            sigreg_t_max = 2.5
            sigreg_proj_dim = 128

            [mode.loss.recon]
            weight = 0.7
            mask_ratio = 0.6
            hidden_dim = 192
        "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse config");
    let err = config
        .validate()
        .expect_err("rho_stream and trm_graph must be exclusive");
    let message = format!("{err:#}");
    assert!(message.contains("vision.rho_stream and vision.trm_graph cannot both be enabled"));
}

#[test]
fn rho_stream_rejects_rollout_fused_without_forward_kernel() {
    let text = r#"
            [dataset]
            imagenet_root = "data/imagenet1k"
            train_dir = "train"
            val_dir = "val"

            [training]
            batch_size = 8
            max_iters = 10
            log_frequency = 2

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.1

            [vision]
            image_size = 224
            patch_size = 14
            in_channels = 3
            embed_dim = 256
            steps = 4
            n_head = 4
            mlp_internal_dim_multiplier = 4
            dropout = 0.1
            projection_dim = 384
            projection_hidden_dim = 512
            use_cls_token = true
            pos_encoding = "learned2d"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [vision.rho_stream]
            enabled = true
            local_radius = 1
            local_diagonals = true
            local_self = true
            decay = 0.9
            wgpu_forward_kernel = false
            wgpu_rollout_fused = true

            [mode]
            type = "lejepa"
            views = 2
            global_views = 2
            local_views = 0
            local_image_size = 224
            local_min_scale = 0.2
            local_max_scale = 0.2
            min_view_overlap = 0.0
            view_overlap_attempts = 1

            [mode.loss.lejepa]
            enabled = true
            lambda = 0.05
            sigreg_knots = 19
            sigreg_t_max = 2.5
            sigreg_proj_dim = 128

            [mode.loss.recon]
            weight = 0.7
            mask_ratio = 0.6
            hidden_dim = 192
        "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse config");
    let err = config
        .validate()
        .expect_err("rho_stream rollout fusion requires forward kernel");
    let message = format!("{err:#}");
    assert!(message.contains(
        "vision.rho_stream.wgpu_rollout_fused requires vision.rho_stream.wgpu_forward_kernel = true"
    ));
}
