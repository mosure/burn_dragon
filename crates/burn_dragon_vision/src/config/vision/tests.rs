use super::*;
use crate::model::VisionTrmPredictSubstepKind;
use burn_dragon_train::LearningRateScheduleConfig;
use std::path::PathBuf;

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
            weight_decay = 0.0
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
            rollout_supervision_stride = 2
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
                assert_eq!(distill.rollout_supervision_stride, 2);
                assert!((distill.rollout_supervision_power - 1.5).abs() < f32::EPSILON);
                assert!((distill.rollout_sampling_power - 1.25).abs() < f32::EPSILON);
            }
            other => panic!("unexpected teacher config: {other:?}"),
        },
        other => panic!("unexpected mode: {other:?}"),
    }
}

#[test]
fn distill_mode_parses_auxiliary_teacher_targets() {
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
            image_size = 280
            patch_size = 14
            in_channels = 3
            embed_dim = 320
            steps = 4
            n_head = 8
            mlp_internal_dim_multiplier = 4
            dropout = 0.0
            projection_dim = 768
            projection_hidden_dim = 1536
            use_cls_token = true
            pos_encoding = "learned2d"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [mode]
            type = "distill"

            [mode.teacher]
            type = "features"
            train_cls_path = "dinov2/train_cls.bin"
            train_patch_path = "dinov2/train_patch.bin"
            val_cls_path = "dinov2/val_cls.bin"
            val_patch_path = "dinov2/val_patch.bin"
            feature_dim = 768
            patch_tokens = 400

            [[mode.teacher_targets]]
            name = "siglip2_global"
            weight = 0.35
            target_kind = "global_only"

            [mode.teacher_targets.teacher]
            type = "features"
            train_cls_path = "siglip2/train_cls.bin"
            val_cls_path = "siglip2/val_cls.bin"
            feature_dim = 768

            [augment]
            image_size = 280
            resize_short = 320
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

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse distill config");
    config
        .validate()
        .expect("multi-teacher distill config should validate");

    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.teacher_targets.len(), 1);
            let target = &distill.teacher_targets[0];
            assert_eq!(target.name, "siglip2_global");
            assert_eq!(target.target_kind, VisionTeacherTargetKind::GlobalOnly);
            assert_eq!(
                target.decoder_mode,
                VisionTeacherDecoderMode::SharedProjection
            );
            assert_eq!(target.decoder_hidden_dim, None);
            match &target.teacher {
                VisionTeacherConfig::Features(teacher) => {
                    assert_eq!(teacher.feature_dim, 768);
                    assert!(teacher.train_patch_path.is_none());
                    assert!(teacher.val_patch_path.is_none());
                }
                other => panic!("unexpected teacher target config: {other:?}"),
            }
        }
        other => panic!("unexpected mode: {other:?}"),
    }
}

#[test]
fn rac_mode_parses_and_validates() {
    let text = r#"
            [dataset]
            source = "cifar10"
            cifar_root = "data/cifar"
            train_dir = "train"
            val_dir = "test"

            [training]
            batch_size = 8
            max_iters = 16
            log_frequency = 4

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.01

            [vision]
            image_size = 32
            patch_size = 4
            backbone = "cellular"
            in_channels = 3
            embed_dim = 48
            steps = 3
            n_head = 3
            mlp_internal_dim_multiplier = 3
            dropout = 0.0
            projection_dim = 32
            projection_hidden_dim = 64
            use_cls_token = true
            token_state_norm = true
            pos_encoding = "rope"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [vision.rho_stream]
            enabled = true
            local_radius = 1
            local_diagonals = true
            local_self = true

            [mode]
            type = "rac"
            sample_steps = 3
            random_time_grid = false
            state_channels = 3
            velocity_hidden_dim = 64
            artifact_output = "images"
            artifact_every = 1
            artifact_max_images = 2
            artifact_upscale = 2

            [mode.teacher]
            kind = "pooled_image"
            latent_downsample = 4

            [mode.memory]
            observe_steps = 2
            backprop_steps = 2
            flow_backprop_steps = 2
            reset_each_step = false
            detach_each_step = false
            eval_wipe_after_step = 2

            [augment]
            image_size = 32
            resize_short = 32
            min_scale = 1.0
            max_scale = 1.0
            min_aspect_ratio = 1.0
            max_aspect_ratio = 1.0
            normalize_mean = [0.0, 0.0, 0.0]
            normalize_std = [1.0, 1.0, 1.0]
        "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse rac config");
    config.validate().expect("rac config should validate");

    match config.mode {
        VisionTrainingModeConfig::Rac(rac) => {
            assert_eq!(rac.sample_steps, 3);
            assert!(!rac.random_time_grid);
            assert_eq!(rac.state_channels, 3);
            assert_eq!(rac.teacher.kind, VisionRacTeacherKind::PooledImage);
            assert_eq!(rac.memory.observe_steps, 2);
            assert_eq!(rac.memory.backprop_steps, 2);
            assert_eq!(rac.memory.flow_backprop_steps, Some(2));
            assert!(!rac.memory.detach_each_step);
            assert_eq!(rac.memory.eval_wipe_after_step, Some(2));
        }
        other => panic!("unexpected mode: {other:?}"),
    }
}

#[test]
fn rac_mode_imagenet_parses_and_validates() {
    let text = r#"
            [dataset]
            source = "imagenet"
            imagenet_root = "data/imagenet1k"
            train_dir = "train"
            val_dir = "val"
            max_records = 1024

            [training]
            batch_size = 8
            max_iters = 16
            log_frequency = 4

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.01

            [vision]
            image_size = 128
            patch_size = 8
            backbone = "dense"
            in_channels = 3
            embed_dim = 64
            steps = 4
            n_head = 4
            mlp_internal_dim_multiplier = 4
            dropout = 0.0
            projection_dim = 96
            projection_hidden_dim = 192
            use_cls_token = true
            token_state_norm = true
            pos_encoding = "rope"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [vision.rho_stream]
            enabled = false
            mode_embeddings = false

            [mode]
            type = "rac"
            sample_steps = 8
            random_time_grid = true
            state_channels = 3
            velocity_hidden_dim = 96
            artifact_output = "images"
            artifact_every = 1
            artifact_max_images = 2
            artifact_upscale = 1
            artifact_overwrite = false

            [mode.teacher]
            kind = "pooled_image"
            latent_downsample = 8
            state_mapping = "expand_nearest"

            [mode.memory]
            observe_steps = 2
            backprop_steps = 2

            [augment]
            image_size = 128
            resize_short = 128
            min_scale = 1.0
            max_scale = 1.0
            min_aspect_ratio = 1.0
            max_aspect_ratio = 1.0
            normalize_mean = [0.0, 0.0, 0.0]
            normalize_std = [1.0, 1.0, 1.0]
        "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse rac imagenet config");
    config
        .validate()
        .expect("rac imagenet config should validate");

    match config.mode {
        VisionTrainingModeConfig::Rac(rac) => {
            assert_eq!(rac.sample_steps, 8);
            assert!(rac.random_time_grid);
            assert_eq!(rac.teacher.latent_downsample, 8);
            assert!(!rac.artifact_overwrite);
        }
        other => panic!("unexpected mode: {other:?}"),
    }
}

#[test]
fn distill_mode_validates_dedicated_spatial_auxiliary_teacher_targets() {
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
            image_size = 280
            patch_size = 14
            in_channels = 3
            embed_dim = 320
            steps = 4
            n_head = 8
            mlp_internal_dim_multiplier = 4
            dropout = 0.0
            projection_dim = 768
            projection_hidden_dim = 1536
            use_cls_token = true
            pos_encoding = "learned2d"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [mode]
            type = "distill"

            [mode.teacher]
            type = "features"
            train_cls_path = "dinov2/train_cls.bin"
            train_patch_path = "dinov2/train_patch.bin"
            val_cls_path = "dinov2/val_cls.bin"
            val_patch_path = "dinov2/val_patch.bin"
            feature_dim = 768
            patch_tokens = 400

            [[mode.teacher_targets]]
            name = "siglip2_spatial"
            weight = 0.25
            target_kind = "patch_and_cls"
            decoder_mode = "dedicated_spatial_projection"
            decoder_hidden_dim = 1024

            [mode.teacher_targets.teacher]
            type = "features"
            train_cls_path = "siglip2/train_cls.bin"
            train_patch_path = "siglip2/train_patch.bin"
            val_cls_path = "siglip2/val_cls.bin"
            val_patch_path = "siglip2/val_patch.bin"
            feature_dim = 1152
            patch_tokens = 196

            [augment]
            image_size = 280
            resize_short = 320
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

    let config: VisionTrainingConfig =
        toml::from_str(text).expect("parse distill config with dedicated spatial teacher");
    config
        .validate()
        .expect("dedicated spatial auxiliary teacher target should validate");

    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.teacher_targets.len(), 1);
            let target = &distill.teacher_targets[0];
            assert_eq!(target.name, "siglip2_spatial");
            assert_eq!(target.target_kind, VisionTeacherTargetKind::PatchAndCls);
            assert_eq!(
                target.decoder_mode,
                VisionTeacherDecoderMode::DedicatedSpatialProjection
            );
            assert_eq!(target.decoder_hidden_dim, Some(1024));
            match &target.teacher {
                VisionTeacherConfig::Features(teacher) => {
                    assert_eq!(teacher.feature_dim, 1152);
                    assert_eq!(teacher.patch_tokens, Some(196));
                }
                other => panic!("unexpected teacher target config: {other:?}"),
            }
        }
        other => panic!("unexpected mode: {other:?}"),
    }
}

#[test]
fn rac_mode_imagenet_precomputed_latent_parses_and_validates() {
    let text = r#"
            [dataset]
            source = "imagenet"
            imagenet_root = "data/imagenet1k"
            train_dir = "train"
            val_dir = "val"
            max_records = 1024

            [training]
            batch_size = 8
            max_iters = 16
            log_frequency = 4

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.01

            [vision]
            image_size = 128
            patch_size = 8
            backbone = "dense"
            in_channels = 3
            embed_dim = 64
            steps = 4
            n_head = 4
            mlp_internal_dim_multiplier = 4
            dropout = 0.0
            projection_dim = 96
            projection_hidden_dim = 192
            use_cls_token = true
            token_state_norm = true
            pos_encoding = "rope"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [vision.rho_stream]
            enabled = false
            mode_embeddings = false

            [mode]
            type = "rac"
            sample_steps = 8
            random_time_grid = true
            state_channels = 4
            velocity_hidden_dim = 96
            artifact_output = "images"
            artifact_every = 1
            artifact_max_images = 2
            artifact_upscale = 1
            artifact_overwrite = false

            [mode.teacher]
            kind = "precomputed_latent"
            state_mapping = "centered_subpixel"

            [mode.teacher.precomputed_latent]
            train_path = "taesd/train_latent.bin"
            val_path = "taesd/val_latent.bin"
            channels = 4
            height = 16
            width = 16

            [mode.memory]
            observe_steps = 2
            backprop_steps = 2

            [augment]
            image_size = 128
            resize_short = 128
            min_scale = 1.0
            max_scale = 1.0
            min_aspect_ratio = 1.0
            max_aspect_ratio = 1.0
            flip_prob = 0.0
            color_jitter_prob = 0.0
            grayscale_prob = 0.0
            blur_prob = 0.0
            solarize_prob = 0.0
            normalize_mean = [0.0, 0.0, 0.0]
            normalize_std = [1.0, 1.0, 1.0]
        "#;

    let config: VisionTrainingConfig =
        toml::from_str(text).expect("parse rac precomputed latent config");
    config
        .validate()
        .expect("rac precomputed latent config should validate");

    match config.mode {
        VisionTrainingModeConfig::Rac(rac) => {
            assert_eq!(rac.teacher.kind, VisionRacTeacherKind::PrecomputedLatent);
            assert_eq!(
                rac.teacher.state_mapping,
                VisionRacStateMappingKind::CenteredSubpixel
            );
            let spec = rac
                .teacher
                .precomputed_latent
                .expect("precomputed latent spec");
            assert_eq!(spec.channels, 4);
            assert_eq!(spec.height, 16);
            assert_eq!(spec.width, 16);
        }
        other => panic!("unexpected mode: {other:?}"),
    }
}

#[test]
fn distill_mode_parses_student_checkpoint() {
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
            type = "distill"
            student_checkpoint = "runs/vision/example/checkpoint/model-1.bin"

            [mode.teacher]
            type = "features"
            train_cls_path = "train_cls.bin"
            train_patch_path = "train_patch.bin"
            val_cls_path = "val_cls.bin"
            val_patch_path = "val_patch.bin"
            feature_dim = 384
        "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse distill config");
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(
                distill.student_checkpoint,
                Some(PathBuf::from("runs/vision/example/checkpoint/model-1.bin"))
            );
        }
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
fn distill_rejects_zero_rollout_supervision_stride() {
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
            rollout_supervision_stride = 0

            [mode.teacher]
            type = "features"
            train_cls_path = "train_cls.bin"
            train_patch_path = "train_patch.bin"
            val_cls_path = "val_cls.bin"
            val_patch_path = "val_patch.bin"
            feature_dim = 384
            patch_tokens = 256
        "#;

    let err = toml::from_str::<VisionTrainingConfig>(text)
        .expect("parse config")
        .validate()
        .expect_err("zero rollout supervision stride should be rejected");
    assert!(
        err.to_string()
            .contains("mode.rollout_supervision_stride must be > 0 for distill mode")
    );
}

#[test]
fn distill_rejects_zero_rollout_supervision_explicit_step() {
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
            rollout_supervision_explicit_steps = [0, 4, 8]

            [mode.teacher]
            type = "features"
            train_cls_path = "train_cls.bin"
            train_patch_path = "train_patch.bin"
            val_cls_path = "val_cls.bin"
            val_patch_path = "val_patch.bin"
            feature_dim = 384
            patch_tokens = 256
        "#;

    let err = toml::from_str::<VisionTrainingConfig>(text)
        .expect("parse config")
        .validate()
        .expect_err("zero rollout supervision explicit step should be rejected");
    assert!(
        err.to_string()
            .contains("mode.rollout_supervision_explicit_steps entries must be > 0")
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
            name: burn_dragon_train::OptimizerKind::default(),
            learning_rate: 1e-3,
            weight_decay: 0.0,
            weight_decay_final: None,
            lr_schedule: None,
            schedule_mode: burn_dragon_train::OptimizerScheduleMode::default(),
            grad_clip_norm: None,
            grad_clip_value: None,
            muon: None,
        },
        wgpu: burn_dragon_train::WgpuRuntimeConfig::default(),
        run_layout: burn_dragon_train::RunLayoutConfig::default(),
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
fn video_vjepa21_allows_imagenet_source() {
    let mut config = VisionTrainingConfig {
        dataset: VisionDatasetConfig::default(),
        training: VisionTrainingHyperparameters::default(),
        optimizer: burn_dragon_train::OptimizerConfig {
            name: burn_dragon_train::OptimizerKind::default(),
            learning_rate: 1e-3,
            weight_decay: 0.0,
            weight_decay_final: None,
            lr_schedule: None,
            schedule_mode: burn_dragon_train::OptimizerScheduleMode::default(),
            grad_clip_norm: None,
            grad_clip_value: None,
            muon: None,
        },
        wgpu: burn_dragon_train::WgpuRuntimeConfig::default(),
        run_layout: burn_dragon_train::RunLayoutConfig::default(),
        vision: VisionModelConfig::default(),
        augment: VisionAugmentationConfig::default(),
        mode: VisionTrainingModeConfig::VideoLejepa(VisionVideoLejepaConfig::default()),
    };
    config.dataset.source = VisionDatasetSource::Imagenet;
    config.vision.image_size = 224;
    config.vision.patch_size = 16;
    config.vision.in_channels = 3;
    config.vision.embed_dim = 192;
    config.vision.steps = 8;
    config.vision.n_head = 6;
    config.vision.mlp_internal_dim_multiplier = 4;
    config.vision.projection_dim = 96;
    config.vision.projection_hidden_dim = 256;
    config.vision.use_cls_token = true;
    config.vision.pos_encoding = crate::SpatialPositionalEncodingKind::Rope;
    config.augment.image_size = 224;
    config.training.batch_size = 2;
    config.training.max_iters = 2;
    if let VisionTrainingModeConfig::VideoLejepa(video) = &mut config.mode {
        video.paradigm = VisionVideoParadigmKind::Vjepa21;
        video.vjepa21.clip_frames = 2;
        video.loss.probe_weight = 0.0;
        video.loss.debug_recon_weight = 0.0;
    }

    config
        .validate()
        .expect("V-JEPA 2.1 should accept ImageNet clips");
}

#[test]
fn video_lejepa_artifact_future_must_cover_training_target() {
    let mut config = VisionTrainingConfig {
        dataset: VisionDatasetConfig::default(),
        training: VisionTrainingHyperparameters::default(),
        optimizer: burn_dragon_train::OptimizerConfig {
            name: burn_dragon_train::OptimizerKind::default(),
            learning_rate: 1e-3,
            weight_decay: 0.0,
            weight_decay_final: None,
            lr_schedule: None,
            schedule_mode: burn_dragon_train::OptimizerScheduleMode::default(),
            grad_clip_norm: None,
            grad_clip_value: None,
            muon: None,
        },
        wgpu: burn_dragon_train::WgpuRuntimeConfig::default(),
        run_layout: burn_dragon_train::RunLayoutConfig::default(),
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
            name: burn_dragon_train::OptimizerKind::default(),
            learning_rate: 1e-3,
            weight_decay: 0.0,
            weight_decay_final: None,
            lr_schedule: None,
            schedule_mode: burn_dragon_train::OptimizerScheduleMode::default(),
            grad_clip_norm: None,
            grad_clip_value: None,
            muon: None,
        },
        wgpu: burn_dragon_train::WgpuRuntimeConfig::default(),
        run_layout: burn_dragon_train::RunLayoutConfig::default(),
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
            name: burn_dragon_train::OptimizerKind::default(),
            learning_rate: 1e-3,
            weight_decay: 0.0,
            weight_decay_final: None,
            lr_schedule: None,
            schedule_mode: burn_dragon_train::OptimizerScheduleMode::default(),
            grad_clip_norm: None,
            grad_clip_value: None,
            muon: None,
        },
        wgpu: burn_dragon_train::WgpuRuntimeConfig::default(),
        run_layout: burn_dragon_train::RunLayoutConfig::default(),
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
            name: burn_dragon_train::OptimizerKind::default(),
            learning_rate: 1e-3,
            weight_decay: 0.0,
            weight_decay_final: None,
            lr_schedule: None,
            schedule_mode: burn_dragon_train::OptimizerScheduleMode::default(),
            grad_clip_norm: None,
            grad_clip_value: None,
            muon: None,
        },
        wgpu: burn_dragon_train::WgpuRuntimeConfig::default(),
        run_layout: burn_dragon_train::RunLayoutConfig::default(),
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
fn saccade_mode_parses_conformal_warp() {
    let text = r#"
            [dataset]
            imagenet_root = "data/imagenet1k"
            train_dir = "train"
            val_dir = "val"

            [training]
            batch_size = 2
            max_iters = 1

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.0

            [vision]
            image_size = 64
            patch_size = 8
            in_channels = 3
            embed_dim = 64
            steps = 2
            n_head = 2
            mlp_internal_dim_multiplier = 4
            dropout = 0.0
            projection_dim = 64
            projection_hidden_dim = 128
            use_cls_token = true
            num_eyes = 1
            pos_encoding = "learned2d"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [mode]
            type = "saccade"
            fovea_warp_mode = "conformal"
        "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse conformal saccade");
    match config.mode {
        VisionTrainingModeConfig::Saccade(saccade) => {
            assert_eq!(saccade.fovea_warp_mode, VisionFoveaWarpMode::Conformal);
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
fn scene_slot_graph_bridge_preset_applies_to_training_config() {
    let mut config = VisionModelConfig::default();
    config.steps = 2;
    config.backbone = Some(VisionBackboneKind::Dense);
    config.rho_stream.enabled = true;

    config.apply_scene_slot_graph_bridge_preset();

    assert_eq!(config.backbone, Some(VisionBackboneKind::Pyramid));
    assert_eq!(config.steps, 4);
    assert_eq!(
        config.trm_graph,
        VisionTrmGraphConfig::scene_slot_graph_bridge_preset()
    );
    assert_eq!(
        config.resolved_backbone_kind().expect("resolved backbone"),
        VisionBackboneKind::Pyramid
    );
    assert!(!config.rho_stream.enabled);
}

#[test]
fn scene_slot_graph_preset_applies_to_runtime_config() {
    let mut config = VisionDragonConfig::default();
    config.steps = 1;
    config.backbone = VisionBackboneKind::Dense;
    config.rho_stream.enabled = true;

    config.apply_scene_slot_graph_preset();

    assert_eq!(config.backbone, VisionBackboneKind::Pyramid);
    assert_eq!(config.steps, 3);
    assert_eq!(
        config.trm_graph,
        VisionTrmGraphConfig::scene_slot_graph_preset()
    );
    assert!(!config.rho_stream.enabled);
}

#[test]
fn scene_slot_graph_bridge_baseline_224_sets_promoted_image_recipe() {
    let config = VisionModelConfig::scene_slot_graph_bridge_baseline_224();

    assert_eq!(config.image_size, 224);
    assert_eq!(config.patch_size, 16);
    assert_eq!(config.backbone, Some(VisionBackboneKind::Pyramid));
    assert_eq!(config.embed_dim, 160);
    assert_eq!(config.steps, 4);
    assert_eq!(config.n_head, 5);
    assert_eq!(config.pos_max_height, Some(14));
    assert_eq!(config.pos_max_width, Some(14));
    assert_eq!(
        config.trm_graph.predict_substep_kind,
        VisionTrmPredictSubstepKind::LocalBridge
    );
    assert_eq!(config.trm_graph.predict_coarse_substeps, 2);
    assert!(!config.rho_stream.enabled);
}

#[test]
fn scene_slot_graph_baseline_224_sets_control_image_recipe() {
    let config = VisionModelConfig::scene_slot_graph_baseline_224();

    assert_eq!(config.image_size, 224);
    assert_eq!(config.patch_size, 16);
    assert_eq!(config.backbone, Some(VisionBackboneKind::Pyramid));
    assert_eq!(config.embed_dim, 160);
    assert_eq!(config.steps, 3);
    assert_eq!(config.n_head, 5);
    assert_eq!(config.pos_max_height, Some(14));
    assert_eq!(config.pos_max_width, Some(14));
    assert_eq!(
        config.trm_graph.predict_substep_kind,
        VisionTrmPredictSubstepKind::CoarseOnly
    );
    assert_eq!(config.trm_graph.predict_coarse_substeps, 1);
    assert!(!config.rho_stream.enabled);
}

#[test]
fn scene_slot_graph_bridge_runtime_baseline_224_sets_promoted_image_recipe() {
    let config = VisionDragonConfig::scene_slot_graph_bridge_baseline_224();

    assert_eq!(config.image_size, 224);
    assert_eq!(config.patch_size, 16);
    assert_eq!(config.backbone, VisionBackboneKind::Pyramid);
    assert_eq!(config.embed_dim, 160);
    assert_eq!(config.steps, 4);
    assert_eq!(config.n_head, 5);
    assert_eq!(config.pos_max_height, 14);
    assert_eq!(config.pos_max_width, 14);
    assert_eq!(
        config.trm_graph.predict_substep_kind,
        VisionTrmPredictSubstepKind::LocalBridge
    );
    assert_eq!(config.trm_graph.predict_coarse_substeps, 2);
    assert!(!config.rho_stream.enabled);
}

#[test]
fn scene_slot_graph_runtime_baseline_224_sets_control_image_recipe() {
    let config = VisionDragonConfig::scene_slot_graph_baseline_224();

    assert_eq!(config.image_size, 224);
    assert_eq!(config.patch_size, 16);
    assert_eq!(config.backbone, VisionBackboneKind::Pyramid);
    assert_eq!(config.embed_dim, 160);
    assert_eq!(config.steps, 3);
    assert_eq!(config.n_head, 5);
    assert_eq!(config.pos_max_height, 14);
    assert_eq!(config.pos_max_width, 14);
    assert_eq!(
        config.trm_graph.predict_substep_kind,
        VisionTrmPredictSubstepKind::CoarseOnly
    );
    assert_eq!(config.trm_graph.predict_coarse_substeps, 1);
    assert!(!config.rho_stream.enabled);
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

#[test]
fn normalization_kind_parses_for_vision_model() {
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

            [vision.normalization]
            kind = "derf"

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
        "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse config");
    config
        .validate()
        .expect("vision normalization config should validate");
    assert_eq!(
        config.vision.normalization.kind,
        burn_dragon_core::DragonNormKind::Derf
    );
}

#[test]
fn convnext_patch_embed_mode_parses_for_vision_model() {
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
            patch_embed_mode = "conv_next"

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
        "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse config");
    config.validate().expect("vision config should validate");
    assert_eq!(
        config.vision.patch_embed_mode,
        VisionPatchEmbedMode::ConvNext
    );
}

#[test]
fn convnext_patch_embed_mode_alias_parses_for_vision_model() {
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
            patch_embed_mode = "convnext"

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
        "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse config");
    config.validate().expect("vision config should validate");
    assert_eq!(
        config.vision.patch_embed_mode,
        VisionPatchEmbedMode::ConvNext
    );
}

#[test]
fn scaleaware_trm_rank_overrides_load_from_overlay_stack() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/video_lejepa/moving_mnist_trm_norm_smoke_scaleaware_regionheavy_corefocus.toml",
    );
    let config =
        load_vision_training_config(&[config_path]).expect("load scale-aware region-heavy config");

    assert_eq!(config.vision.trm_graph.patch_rank, Some(32));
    assert_eq!(config.vision.trm_graph.coarse_rank, Some(64));
    assert_eq!(config.vision.trm_graph.global_rank, Some(8));
    assert_eq!(config.vision.trm_graph.rank, 8);
}

#[test]
fn imagenette_dinov2_pyramid_stageaware_smoke_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root
        .join("config/vision/distill/imagenette_dinov2_vits14_pyramid_stageaware_smoke.toml");
    let config = load_vision_training_config(&[config_path]).expect("load pyramid distill smoke");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Pyramid
    );
    assert_eq!(config.vision.trm_graph.patch_rank, Some(1));
    assert_eq!(config.vision.trm_graph.coarse_rank, Some(96));
    assert_eq!(config.vision.trm_graph.global_rank, Some(8));
    assert_eq!(config.vision.trm_graph.predict_coarse_substeps, 1);
    assert!(
        !config
            .vision
            .trm_graph
            .bank_schedule
            .predict
            .patch_local_read
    );
    assert!(
        !config
            .vision
            .trm_graph
            .bank_schedule
            .predict
            .patch_local_write
    );
    assert!(
        !config
            .vision
            .trm_graph
            .bank_schedule
            .predict
            .patch_from_coarse_read
    );
    assert!(
        !config
            .vision
            .trm_graph
            .bank_schedule
            .predict
            .patch_to_coarse_write
    );
}

#[test]
fn imagenette_dinov2_dense_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path =
        repo_root.join("config/vision/distill/imagenette_dinov2_vits14_dense_short.toml");
    let config = load_vision_training_config(&[config_path]).expect("load dense distill short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 96);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.training.max_iters, 128);
}

#[test]
fn imagenette_dinov2_dense_fixedtime90_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path =
        repo_root.join("config/vision/distill/imagenette_dinov2_vits14_dense_fixedtime90.toml");
    let config =
        load_vision_training_config(&[config_path]).expect("load dense distill fixedtime90");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 96);
    assert_eq!(config.dataset.max_records, Some(10752));
    assert_eq!(config.training.max_iters, 112);
}

#[test]
fn imagenette_dinov2_dense_h6_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path =
        repo_root.join("config/vision/distill/imagenette_dinov2_vits14_dense_h6_short.toml");
    let config = load_vision_training_config(&[config_path]).expect("load dense h6 short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 48);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.rollout_max_steps, Some(6));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 6);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h6_fixedtime120_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path =
        repo_root.join("config/vision/distill/imagenette_dinov2_vits14_dense_h6_fixedtime120.toml");
    let config = load_vision_training_config(&[config_path]).expect("load dense h6 fixedtime120");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 48);
    assert_eq!(config.dataset.max_records, Some(12288));
    assert_eq!(config.training.max_iters, 256);
    assert_eq!(config.training.rollout_max_steps, Some(6));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 6);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h8_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path =
        repo_root.join("config/vision/distill/imagenette_dinov2_vits14_dense_h8_short.toml");
    let config = load_vision_training_config(&[config_path]).expect("load dense h8 short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.rollout_max_steps, Some(8));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 8);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h8_fixedtime120_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path =
        repo_root.join("config/vision/distill/imagenette_dinov2_vits14_dense_h8_fixedtime120.toml");
    let config = load_vision_training_config(&[config_path]).expect("load dense h8 fixedtime120");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(12288));
    assert_eq!(config.training.max_iters, 256);
    assert_eq!(config.training.rollout_max_steps, Some(8));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 8);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h8_deepbias_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root
        .join("config/vision/distill/imagenette_dinov2_vits14_dense_h8_deepbias_short.toml");
    let config = load_vision_training_config(&[config_path]).expect("load dense h8 deepbias short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.rollout_max_steps, Some(8));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 8);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 1.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path =
        repo_root.join("config/vision/distill/imagenette_dinov2_vits14_dense_h10_short.toml");
    let config = load_vision_training_config(&[config_path]).expect("load dense h10 short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_wide_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path =
        repo_root.join("config/vision/distill/imagenette_dinov2_vits14_dense_h10_wide_short.toml");
    let config = load_vision_training_config(&[config_path]).expect("load dense h10 wide short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    assert_eq!(config.vision.embed_dim, 192);
    assert_eq!(config.vision.n_head, 6);
    assert_eq!(config.vision.projection_hidden_dim, 512);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_multiframe2_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root
        .join("config/vision/distill/imagenette_dinov2_vits14_dense_h10_multiframe2_short.toml");
    let config =
        load_vision_training_config(&[config_path]).expect("load dense h10 multiframe2 short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 2);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
            assert!((distill.rollout_improvement_weight - 0.0).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_proj512_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root
        .join("config/vision/distill/imagenette_dinov2_vits14_dense_h10_proj512_short.toml");
    let config = load_vision_training_config(&[config_path]).expect("load dense h10 proj512 short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    assert_eq!(config.vision.embed_dim, 128);
    assert_eq!(config.vision.n_head, 4);
    assert_eq!(config.vision.projection_hidden_dim, 512);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root
        .join("config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_short.toml");
    let config =
        load_vision_training_config(&[config_path]).expect("load dense h10 deepbias short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 1.0).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_wide_deepbias_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root
        .join("config/vision/distill/imagenette_dinov2_vits14_dense_h10_wide_deepbias_short.toml");
    let config =
        load_vision_training_config(&[config_path]).expect("load dense h10 wide deepbias short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    assert_eq!(config.vision.embed_dim, 192);
    assert_eq!(config.vision.n_head, 6);
    assert_eq!(config.vision.projection_hidden_dim, 512);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 1.0).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias1p5_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root
        .join("config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias1p5_short.toml");
    let config =
        load_vision_training_config(&[config_path]).expect("load dense h10 deepbias1p5 short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 1.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_rel2pct_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root
        .join("config/vision/distill/imagenette_dinov2_vits14_dense_h10_rel2pct_short.toml");
    let config = load_vision_training_config(&[config_path]).expect("load dense h10 rel2pct short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
            assert!((distill.loss.rel_weight - 0.02).abs() < f32::EPSILON);
            assert!((distill.loss.rel_tau - 0.07).abs() < f32::EPSILON);
            assert_eq!(distill.loss.rel_sample_tokens, Some(64));
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_rel1pct_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root
        .join("config/vision/distill/imagenette_dinov2_vits14_dense_h10_rel1pct_short.toml");
    let config = load_vision_training_config(&[config_path]).expect("load dense h10 rel1pct short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
            assert!((distill.loss.rel_weight - 0.01).abs() < f32::EPSILON);
            assert!((distill.loss.rel_tau - 0.07).abs() < f32::EPSILON);
            assert_eq!(distill.loss.rel_sample_tokens, Some(64));
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_bptt6_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path =
        repo_root.join("config/vision/distill/imagenette_dinov2_vits14_dense_h10_bptt6_short.toml");
    let config = load_vision_training_config(&[config_path]).expect("load dense h10 bptt6 short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(6));
    assert_eq!(config.vision.steps, 10);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_refinegain_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root
        .join("config/vision/distill/imagenette_dinov2_vits14_dense_h10_refinegain_short.toml");
    let config =
        load_vision_training_config(&[config_path]).expect("load dense h10 refinegain short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 3);
            assert!((distill.rollout_supervision_power - 1.25).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_improvement_weight - 0.25).abs() < f32::EPSILON);
            assert!((distill.rollout_improvement_margin - 0.0).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_multiframe_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root
        .join("config/vision/distill/imagenette_dinov2_vits14_dense_h10_multiframe_short.toml");
    let config =
        load_vision_training_config(&[config_path]).expect("load dense h10 multiframe short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 3);
            assert!((distill.rollout_supervision_power - 1.25).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 1.0).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_fixedtime120_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root
        .join("config/vision/distill/imagenette_dinov2_vits14_dense_h10_fixedtime120.toml");
    let config = load_vision_training_config(&[config_path]).expect("load dense h10 fixedtime120");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(12288));
    assert_eq!(config.training.max_iters, 256);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_promoted_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root
        .join("config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_promoted.toml");
    let config =
        load_vision_training_config(&[config_path]).expect("load dense h10 deepbias promoted");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(12288));
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 640);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 1.0).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_promoted_bptt4_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_promoted_bptt4.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h10 deepbias promoted bptt4");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(12288));
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 640);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(4));
    assert_eq!(config.vision.steps, 10);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 1.0).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_promoted_e3_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root
        .join("config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_promoted_e3.toml");
    let config =
        load_vision_training_config(&[config_path]).expect("load dense h10 deepbias promoted e3");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(12288));
    assert_eq!(config.training.epochs, Some(3));
    assert_eq!(config.training.max_iters, 960);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 1.0).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_promoted_e3_schedmatch_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_promoted_e3_schedmatch.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h10 deepbias promoted e3 schedmatch");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(12288));
    assert_eq!(config.training.epochs, Some(3));
    assert_eq!(config.training.max_iters, 960);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    match config.optimizer.lr_schedule {
        Some(LearningRateScheduleConfig::Cosine { num_iters, .. }) => {
            assert_eq!(num_iters, Some(960));
        }
        other => panic!("expected cosine lr schedule, got {other:?}"),
    }
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 1.0).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_promoted_schedmatch_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_promoted_schedmatch.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h10 deepbias promoted schedmatch");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(12288));
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 640);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    match config.optimizer.lr_schedule {
        Some(LearningRateScheduleConfig::Cosine { num_iters, .. }) => {
            assert_eq!(num_iters, Some(640));
        }
        other => panic!("expected cosine lr schedule, got {other:?}"),
    }
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 1.0).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias1p25_promoted_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias1p25_promoted.toml",
    );
    let config =
        load_vision_training_config(&[config_path]).expect("load dense h10 deepbias1p25 promoted");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(12288));
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 640);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 1.25).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_refinegain2pct_promoted_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_refinegain2pct_promoted.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h10 deepbias refinegain2pct promoted");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(12288));
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 640);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 2);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_improvement_weight - 0.02).abs() < f32::EPSILON);
            assert!((distill.rollout_improvement_margin - 0.0).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_rel0p5pct_promoted_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_rel0p5pct_promoted.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h10 deepbias rel0p5pct promoted");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(12288));
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 640);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.loss.rel_weight - 0.005).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_mid_deepbias_promoted_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_mid_deepbias_promoted.toml",
    );
    let config =
        load_vision_training_config(&[config_path]).expect("load dense h10 mid deepbias promoted");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(12288));
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 640);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    assert_eq!(config.vision.embed_dim, 160);
    assert_eq!(config.vision.n_head, 5);
    assert_eq!(config.vision.projection_hidden_dim, 448);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 1.0).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h12_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path =
        repo_root.join("config/vision/distill/imagenette_dinov2_vits14_dense_h12_short.toml");
    let config = load_vision_training_config(&[config_path]).expect("load dense h12 short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.rollout_max_steps, Some(12));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 12);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_fixedtime120_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_fixedtime120.toml",
    );
    let config =
        load_vision_training_config(&[config_path]).expect("load dense h10 deepbias fixedtime120");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(12288));
    assert_eq!(config.training.max_iters, 256);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 1.0).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_multiframe_fixedtime120_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_multiframe_fixedtime120.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h10 multiframe fixedtime120");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(12288));
    assert_eq!(config.training.max_iters, 256);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 3);
            assert!((distill.rollout_supervision_power - 1.25).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 1.0).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn moving_mnist_trm_stageaware_hybrid_compute_floor_dense_b160_vv_noaux_coarsesub1_h8_short_loads()
{
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/video_lejepa/moving_mnist_trm_stageaware_hybrid_compute_floor_dense_b160_vv_noaux_coarsesub1_h8_short.toml",
    );
    let config = load_vision_training_config(&[config_path]).expect("load video h8 transfer short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Pyramid
    );
    assert_eq!(config.training.batch_size, 160);
    assert_eq!(config.training.max_iters, 32);
    assert_eq!(config.vision.steps, 8);
    assert_eq!(config.vision.trm_graph.predict_coarse_substeps, 1);
    match config.mode {
        VisionTrainingModeConfig::VideoLejepa(video) => {
            assert_eq!(video.target_frames, 6);
        }
        other => panic!("expected video lejepa mode, got {other:?}"),
    }
}

#[test]
fn moving_mnist_trm_stageaware_hybrid_compute_floor_dense_b160_vv_jepaonly_coarsesub1_h8_short_loads()
 {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/video_lejepa/moving_mnist_trm_stageaware_hybrid_compute_floor_dense_b160_vv_jepaonly_coarsesub1_h8_short.toml",
    );
    let config =
        load_vision_training_config(&[config_path]).expect("load video h8 jepa-only short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Pyramid
    );
    assert_eq!(config.training.batch_size, 160);
    assert_eq!(config.training.max_iters, 32);
    assert_eq!(config.vision.steps, 8);
    assert_eq!(config.vision.trm_graph.predict_coarse_substeps, 1);
    match config.mode {
        VisionTrainingModeConfig::VideoLejepa(video) => {
            assert_eq!(video.target_frames, 6);
            assert!((video.loss.observe_weight - 0.0).abs() < f32::EPSILON);
        }
        other => panic!("expected video lejepa mode, got {other:?}"),
    }
}

#[test]
fn moving_mnist_trm_stageaware_hybrid_compute_floor_dense_b160_vv_noaux_coarsesub1_h8_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/video_lejepa/moving_mnist_trm_stageaware_hybrid_compute_floor_dense_b160_vv_noaux_coarsesub1_h8.toml",
    );
    let config = load_vision_training_config(&[config_path]).expect("load video h8 transfer");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Pyramid
    );
    assert_eq!(config.training.batch_size, 160);
    assert_eq!(config.training.max_iters, 96);
    assert_eq!(config.vision.steps, 8);
    assert_eq!(config.vision.trm_graph.predict_coarse_substeps, 1);
    match config.mode {
        VisionTrainingModeConfig::VideoLejepa(video) => {
            assert_eq!(video.target_frames, 6);
            assert!((video.loss.observe_weight - 1.0).abs() < f32::EPSILON);
        }
        other => panic!("expected video lejepa mode, got {other:?}"),
    }
}

#[test]
fn canonical_video_transfer_horizon6_promo2_experiment_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/video_lejepa/experiments/transfer/moving_mnist_trm_norm_smoke_horizon6_promo2.toml",
    );
    let config =
        load_vision_training_config(&[config_path]).expect("load canonical video transfer promo2");

    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.batch_size, 8);
    match config.mode {
        VisionTrainingModeConfig::VideoLejepa(video) => {
            assert_eq!(video.target_frames, 6);
        }
        other => panic!("expected video lejepa mode, got {other:?}"),
    }
}

#[test]
fn canonical_video_diagnostics_radius0_noself_experiment_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/video_lejepa/experiments/diagnostics/moving_mnist_trm_norm_smoke_scaleaware_regiondominant_predict_nocoarse_global2x_hub6_radius0_noself.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load canonical video diagnostics radius0 noself");

    assert_eq!(config.vision.trm_graph.local_radius, 0);
    assert!(!config.vision.trm_graph.local_self);
}

#[test]
fn video_experiments_compat_transfer_shim_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/video_lejepa/experiments/moving_mnist_trm_norm_smoke_horizon6_promo2.toml",
    );
    let config =
        load_vision_training_config(&[config_path]).expect("load compatibility transfer shim");

    match config.mode {
        VisionTrainingModeConfig::VideoLejepa(video) => {
            assert_eq!(video.target_frames, 6);
        }
        other => panic!("expected video lejepa mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h8_convnext_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root
        .join("config/vision/distill/imagenette_dinov2_vits14_dense_h8_convnext_short.toml");
    let config = load_vision_training_config(&[config_path]).expect("load dense h8 convnext short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(
        config.vision.patch_embed_mode,
        VisionPatchEmbedMode::ConvNext
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.rollout_max_steps, Some(8));
    assert_eq!(config.vision.steps, 8);
}

#[test]
fn imagenette_dinov2_cellular_h8_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path =
        repo_root.join("config/vision/distill/imagenette_dinov2_vits14_cellular_h8_short.toml");
    let config = load_vision_training_config(&[config_path]).expect("load cellular h8 short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Cellular
    );
    assert_eq!(config.training.batch_size, 64);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.rollout_max_steps, Some(8));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 8);
    assert!(config.vision.rho_stream.enabled);
    assert!(config.vision.rho_stream.wgpu_forward_kernel);
    assert!(config.vision.rho_stream.wgpu_rollout_fused);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_cellular_h8_short_b40_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path =
        repo_root.join("config/vision/distill/imagenette_dinov2_vits14_cellular_h8_short_b40.toml");
    let config = load_vision_training_config(&[config_path]).expect("load cellular h8 short b40");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Cellular
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.rollout_max_steps, Some(8));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 8);
}

#[test]
fn imagenette_dinov2_cellular_h8_short_b24_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path =
        repo_root.join("config/vision/distill/imagenette_dinov2_vits14_cellular_h8_short_b24.toml");
    let config = load_vision_training_config(&[config_path]).expect("load cellular h8 short b24");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Cellular
    );
    assert_eq!(config.training.batch_size, 24);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.rollout_max_steps, Some(8));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 8);
}

#[test]
fn imagenette_dinov2_pyramid_stageaware_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root
        .join("config/vision/distill/imagenette_dinov2_vits14_pyramid_stageaware_short.toml");
    let config = load_vision_training_config(&[config_path]).expect("load pyramid distill short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Pyramid
    );
    assert_eq!(config.training.batch_size, 96);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.vision.trm_graph.predict_coarse_substeps, 1);
}

#[test]
fn imagenette_dinov2_pyramid_stageaware_refine2_smoke_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_pyramid_stageaware_refine2_smoke.toml",
    );
    let config =
        load_vision_training_config(&[config_path]).expect("load pyramid distill refine2 smoke");

    assert_eq!(config.vision.trm_graph.predict_coarse_substeps, 2);
}

#[test]
fn imagenette_dinov2_pyramid_stageaware_refine2_coarse96_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_pyramid_stageaware_refine2_coarse96_short.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load pyramid distill refine2 coarse96 short");

    assert_eq!(config.training.batch_size, 96);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.vision.trm_graph.predict_coarse_substeps, 2);
    assert_eq!(config.vision.trm_graph.coarse_rank, Some(96));
}

#[test]
fn imagenette_dinov2_pyramid_stageaware_refine2_coarse96_densepolicy_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_pyramid_stageaware_refine2_coarse96_densepolicy_short.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load pyramid distill refine2 coarse96 densepolicy short");

    assert_eq!(config.training.batch_size, 96);
    assert_eq!(config.dataset.max_records, Some(2048));
    assert_eq!(config.vision.trm_graph.predict_coarse_substeps, 2);
    assert_eq!(config.vision.trm_graph.coarse_rank, Some(96));
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.0).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_pyramid_stageaware_refine2_coarse96_densepolicy_fixedtime90_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_pyramid_stageaware_refine2_coarse96_densepolicy_fixedtime90.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load pyramid distill refine2 coarse96 densepolicy fixedtime90");

    assert_eq!(config.training.batch_size, 96);
    assert_eq!(config.dataset.max_records, Some(5376));
    assert_eq!(config.training.max_iters, 56);
    assert_eq!(config.vision.trm_graph.predict_coarse_substeps, 2);
    assert_eq!(config.vision.trm_graph.coarse_rank, Some(96));
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.0).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_pyramid_stageaware_refine2_coarse96_densepolicy_h10_deepbias_promoted_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_pyramid_stageaware_refine2_coarse96_densepolicy_h10_deepbias_promoted.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load pyramid distill refine2 coarse96 densepolicy h10 deepbias promoted");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Pyramid
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(12288));
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 640);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    assert_eq!(config.vision.trm_graph.predict_coarse_substeps, 2);
    assert_eq!(config.vision.trm_graph.coarse_rank, Some(96));
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 1.0).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_pyramid_stageaware_refine2_coarse96_densepolicy_h8_deepbias0p5_280_short_loads()
 {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_pyramid_stageaware_refine2_coarse96_densepolicy_h8_deepbias0p5_280_short.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load pyramid distill refine2 coarse96 densepolicy h8 deepbias0p5 280 short");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Pyramid
    );
    assert_eq!(config.vision.image_size, 280);
    assert_eq!(config.vision.steps, 8);
    assert_eq!(config.training.batch_size, 24);
    assert_eq!(config.training.epochs, Some(1));
    assert_eq!(config.training.max_iters, 128);
    assert_eq!(config.training.rollout_max_steps, Some(8));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.trm_graph.predict_coarse_substeps, 2);
    assert_eq!(config.vision.trm_graph.coarse_rank, Some(96));
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
            match distill.teacher {
                VisionTeacherConfig::Features(teacher) => {
                    assert_eq!(teacher.patch_tokens, Some(400));
                }
                other => panic!("expected feature teacher, got {other:?}"),
            }
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_ff6_promoted_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_ff6_promoted.toml",
    );
    let config =
        load_vision_training_config(&[config_path]).expect("load dense h10 deepbias ff6 promoted");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(12288));
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 640);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 6);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 1.0).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_ff5_promoted_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_ff5_promoted.toml",
    );
    let config =
        load_vision_training_config(&[config_path]).expect("load dense h10 deepbias ff5 promoted");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.training.batch_size, 40);
    assert_eq!(config.dataset.max_records, Some(12288));
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 640);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    assert_eq!(config.training.rollout_backprop_steps, Some(3));
    assert_eq!(config.vision.steps, 10);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 5);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 1);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 1.0).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_ff5_rel0p25pct_promoted_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_ff5_rel0p25pct_promoted.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h10 deepbias ff5 rel0p25 promoted");

    assert_eq!(
        config
            .vision
            .resolved_backbone_kind()
            .expect("resolved backbone"),
        VisionBackboneKind::Dense
    );
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 5);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert!((distill.rollout_sampling_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.loss.rel_weight - 0.0025).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_224_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root
        .join("config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_224_short.toml");
    let config =
        load_vision_training_config(&[config_path]).expect("load dense h10 deepbias 224 short");

    assert_eq!(config.vision.image_size, 224);
    assert_eq!(config.augment.image_size, 224);
    assert_eq!(config.augment.resize_short, 256);
    assert_eq!(config.training.batch_size, 64);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => match distill.teacher {
            VisionTeacherConfig::Features(teacher) => {
                assert_eq!(teacher.patch_tokens, Some(256));
                assert!(
                    teacher
                        .train_patch_path
                        .expect("teacher patch path")
                        .to_string_lossy()
                        .ends_with(
                            "data/imagenette2-160/features/dinov2_vits14_224/train_patch.bin"
                        )
                );
            }
            other => panic!("expected feature teacher, got {other:?}"),
        },
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_ff5_224_short_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_ff5_224_short.toml",
    );
    let config =
        load_vision_training_config(&[config_path]).expect("load dense h10 deepbias ff5 224 short");

    assert_eq!(config.vision.image_size, 224);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 5);
    assert_eq!(config.training.batch_size, 64);
    match config.mode {
        VisionTrainingModeConfig::Distill(distill) => match distill.teacher {
            VisionTeacherConfig::Features(teacher) => {
                assert_eq!(teacher.patch_tokens, Some(256));
            }
            other => panic!("expected feature teacher, got {other:?}"),
        },
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_ff5_224_fixedtime120_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_ff5_224_fixedtime120.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h10 deepbias ff5 224 fixedtime120");

    assert_eq!(config.vision.image_size, 224);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 5);
    assert_eq!(config.training.max_iters, 256);
    assert_eq!(config.dataset.max_records, Some(12288));
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_ff6_224_fixedtime120_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_ff6_224_fixedtime120.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h10 deepbias ff6 224 fixedtime120");

    assert_eq!(config.vision.image_size, 224);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 6);
    assert_eq!(config.training.max_iters, 256);
    assert_eq!(config.dataset.max_records, Some(12288));
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_ff6_224_promoted_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_ff6_224_promoted.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h10 deepbias ff6 224 promoted");

    assert_eq!(config.vision.image_size, 224);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 6);
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 640);
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_ff7_224_fixedtime120_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_ff7_224_fixedtime120.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h10 deepbias ff7 224 fixedtime120");

    assert_eq!(config.vision.image_size, 224);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 7);
    assert_eq!(config.training.max_iters, 256);
    assert_eq!(config.dataset.max_records, Some(12288));
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_ff6_224_promoted_proj512_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_ff6_224_promoted_proj512.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h10 deepbias ff6 224 promoted proj512");

    assert_eq!(config.vision.image_size, 224);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 6);
    assert_eq!(config.vision.projection_hidden_dim, 512);
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 640);
}

#[test]
fn imagenette_dinov2_dense_h12_deepbias_ff6_224_promoted_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h12_deepbias_ff6_224_promoted.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h12 deepbias ff6 224 promoted");

    assert_eq!(config.vision.image_size, 224);
    assert_eq!(config.vision.steps, 12);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 6);
    assert_eq!(config.training.rollout_max_steps, Some(12));
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 640);
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_ff6_280_fixedtime120_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_ff6_280_fixedtime120.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h10 deepbias ff6 280 fixedtime120");

    assert_eq!(config.vision.image_size, 280);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 6);
    match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => match &distill.teacher {
            VisionTeacherConfig::Features(teacher) => {
                assert!(
                    teacher
                        .train_patch_path
                        .as_ref()
                        .expect("teacher patch path")
                        .ends_with(
                            "data/imagenette2-160/features/dinov2_vits14_280/train_patch.bin"
                        )
                );
                assert_eq!(teacher.patch_tokens, Some(400));
            }
            other => panic!("expected feature teacher, got {other:?}"),
        },
        other => panic!("expected distill mode, got {other:?}"),
    }
    assert_eq!(config.training.max_iters, 256);
    assert_eq!(config.augment.image_size, 280);
    assert_eq!(config.augment.resize_short, 320);
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_ff6_280_promoted_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_ff6_280_promoted.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h10 deepbias ff6 280 promoted");

    assert_eq!(config.vision.image_size, 280);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 6);
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 384);
    match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => match &distill.teacher {
            VisionTeacherConfig::Features(teacher) => {
                assert_eq!(teacher.patch_tokens, Some(400));
            }
            other => panic!("expected feature teacher, got {other:?}"),
        },
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_ff6_280_promoted_longer_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_ff6_280_promoted_longer.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h10 deepbias ff6 280 promoted longer");

    assert_eq!(config.vision.image_size, 280);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 6);
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 512);
    match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => match &distill.teacher {
            VisionTeacherConfig::Features(teacher) => {
                assert_eq!(teacher.patch_tokens, Some(400));
            }
            other => panic!("expected feature teacher, got {other:?}"),
        },
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h12_deepbias_ff6_280_promoted_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h12_deepbias_ff6_280_promoted.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h12 deepbias ff6 280 promoted");

    assert_eq!(config.vision.image_size, 280);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 6);
    assert_eq!(config.vision.steps, 12);
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 384);
    assert_eq!(config.training.rollout_max_steps, Some(12));
    match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => match &distill.teacher {
            VisionTeacherConfig::Features(teacher) => {
                assert_eq!(teacher.patch_tokens, Some(400));
            }
            other => panic!("expected feature teacher, got {other:?}"),
        },
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_ff5_280_promoted_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_ff5_280_promoted.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h10 deepbias ff5 280 promoted");

    assert_eq!(config.vision.image_size, 280);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 5);
    assert_eq!(config.vision.steps, 10);
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 384);
    match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => match &distill.teacher {
            VisionTeacherConfig::Features(teacher) => {
                assert_eq!(teacher.patch_tokens, Some(400));
            }
            other => panic!("expected feature teacher, got {other:?}"),
        },
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h8_deepbias_ff6_280_promoted_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h8_deepbias_ff6_280_promoted.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h8 deepbias ff6 280 promoted");

    assert_eq!(config.vision.image_size, 280);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 6);
    assert_eq!(config.vision.steps, 8);
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 384);
    assert_eq!(config.training.rollout_max_steps, Some(8));
    match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => match &distill.teacher {
            VisionTeacherConfig::Features(teacher) => {
                assert_eq!(teacher.patch_tokens, Some(400));
            }
            other => panic!("expected feature teacher, got {other:?}"),
        },
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h8_deepbias0p5_ff6_280_promoted_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h8_deepbias0p5_ff6_280_promoted.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h8 deepbias0p5 ff6 280 promoted");

    assert_eq!(config.vision.image_size, 280);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 6);
    assert_eq!(config.vision.steps, 8);
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 384);
    assert_eq!(config.training.rollout_max_steps, Some(8));
    match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
            match &distill.teacher {
                VisionTeacherConfig::Features(teacher) => {
                    assert_eq!(teacher.patch_tokens, Some(400));
                }
                other => panic!("expected feature teacher, got {other:?}"),
            }
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias0p5_ff6_280_promoted_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias0p5_ff6_280_promoted.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h10 deepbias0p5 ff6 280 promoted");

    assert_eq!(config.vision.image_size, 280);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 6);
    assert_eq!(config.vision.steps, 10);
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 384);
    assert_eq!(config.training.rollout_max_steps, Some(10));
    match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
            match &distill.teacher {
                VisionTeacherConfig::Features(teacher) => {
                    assert_eq!(teacher.patch_tokens, Some(400));
                }
                other => panic!("expected feature teacher, got {other:?}"),
            }
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h8_deepbias0p75_ff6_280_promoted_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h8_deepbias0p75_ff6_280_promoted.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h8 deepbias0p75 ff6 280 promoted");

    assert_eq!(config.vision.image_size, 280);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 6);
    assert_eq!(config.vision.steps, 8);
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 384);
    assert_eq!(config.training.rollout_max_steps, Some(8));
    match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert!((distill.rollout_sampling_power - 0.75).abs() < f32::EPSILON);
            match &distill.teacher {
                VisionTeacherConfig::Features(teacher) => {
                    assert_eq!(teacher.patch_tokens, Some(400));
                }
                other => panic!("expected feature teacher, got {other:?}"),
            }
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_ff6_224_promoted_stride2_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_ff6_224_promoted_stride2.toml",
    );
    let loaded = load_vision_training_config(&[config]).expect("load stride2 promoted 224 config");
    match loaded.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_stride, 2);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_ff6_224_promoted_stride4_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h10_deepbias_ff6_224_promoted_stride4.toml",
    );
    let loaded = load_vision_training_config(&[config]).expect("load stride4 promoted 224 config");
    match loaded.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_stride, 4);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h8_deepbias0p5_ff6_280_promoted_stride2_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h8_deepbias0p5_ff6_280_promoted_stride2.toml",
    );
    let loaded = load_vision_training_config(&[config]).expect("load stride2 promoted 280 config");
    match loaded.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_stride, 2);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h8_deepbias0p5_ff6_280_promoted_stride4_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h8_deepbias0p5_ff6_280_promoted_stride4.toml",
    );
    let loaded = load_vision_training_config(&[config]).expect("load stride4 promoted 280 config");
    match loaded.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_stride, 4);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_ff6_280_promoted_stride2_experiment_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/experiments/imagenette_dinov2_vits14_dense_h10_deepbias_ff6_280_promoted_stride2.toml",
    );
    let loaded =
        load_vision_training_config(&[config]).expect("load stride2 promoted 280 h10 experiment");
    match loaded.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_stride, 2);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h8_deepbias0p5_ff6_280_promoted_stride3_experiment_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/experiments/imagenette_dinov2_vits14_dense_h8_deepbias0p5_ff6_280_promoted_stride3.toml",
    );
    let loaded =
        load_vision_training_config(&[config]).expect("load stride3 promoted 280 h8 experiment");
    match loaded.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_stride, 3);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h8_deepbias0p5_ff6_280_promoted_multiframe4_stride3_experiment_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/experiments/imagenette_dinov2_vits14_dense_h8_deepbias0p5_ff6_280_promoted_multiframe4_stride3.toml",
    );
    let loaded = load_vision_training_config(&[config])
        .expect("load multiframe4 stride3 promoted 280 h8 experiment");
    match loaded.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 4);
            assert_eq!(distill.rollout_supervision_stride, 3);
            assert!((distill.rollout_supervision_power - 1.0).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h8_deepbias0p5_ff6_280_promoted_multiframe4_stride3_min4_experiment_loads()
 {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/experiments/imagenette_dinov2_vits14_dense_h8_deepbias0p5_ff6_280_promoted_multiframe4_stride3_min4.toml",
    );
    let loaded = load_vision_training_config(&[config])
        .expect("load multiframe4 stride3 min4 promoted 280 h8 experiment");
    assert_eq!(loaded.training.rollout_min_steps, Some(4));
    match loaded.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 4);
            assert_eq!(distill.rollout_supervision_stride, 3);
            assert!(distill.rollout_supervision_include_step1);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h8_deepbias0p5_ff6_280_promoted_multiframe4_stride3_min4_nos1_experiment_loads()
 {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/experiments/imagenette_dinov2_vits14_dense_h8_deepbias0p5_ff6_280_promoted_multiframe4_stride3_min4_nos1.toml",
    );
    let loaded = load_vision_training_config(&[config])
        .expect("load multiframe4 stride3 min4 no-s1 promoted 280 h8 experiment");
    assert_eq!(loaded.training.rollout_min_steps, Some(4));
    match loaded.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 4);
            assert_eq!(distill.rollout_supervision_stride, 3);
            assert!(!distill.rollout_supervision_include_step1);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h8_deepbias0p5_ff6_280_promoted_multiframe2_stride3_min4_nos1_experiment_loads()
 {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/experiments/imagenette_dinov2_vits14_dense_h8_deepbias0p5_ff6_280_promoted_multiframe2_stride3_min4_nos1.toml",
    );
    let loaded = load_vision_training_config(&[config])
        .expect("load multiframe2 stride3 min4 no-s1 promoted 280 h8 experiment");
    assert_eq!(loaded.training.rollout_min_steps, Some(4));
    match loaded.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 2);
            assert_eq!(distill.rollout_supervision_stride, 3);
            assert!(!distill.rollout_supervision_include_step1);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h8_deepbias0p5_ff6_280_promoted_explicit48_experiment_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/experiments/imagenette_dinov2_vits14_dense_h8_deepbias0p5_ff6_280_promoted_explicit48.toml",
    );
    let loaded =
        load_vision_training_config(&[config]).expect("load explicit48 promoted 280 h8 experiment");
    assert_eq!(loaded.training.rollout_min_steps, Some(4));
    match loaded.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_explicit_steps, vec![4, 8]);
            assert!((distill.rollout_supervision_power - 1.25).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn canonical_quality_family_explicit48_experiment_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/experiments/quality/imagenette_dinov2_vits14_dense_h8_deepbias0p5_ff6_280_promoted_explicit48.toml",
    );
    let loaded =
        load_vision_training_config(&[config]).expect("load canonical quality explicit48 config");
    assert_eq!(loaded.training.rollout_min_steps, Some(4));
    match loaded.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_explicit_steps, vec![4, 8]);
            assert!((distill.rollout_supervision_power - 1.25).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h10_deepbias_ff6_224_efficiency_bs128_fused_explicit48_experiment_loads()
{
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/experiments/imagenette_dinov2_vits14_dense_h10_deepbias_ff6_224_efficiency_prefetch8_device_preprocessed_teachercache_bs128_fused_explicit48.toml",
    );
    let loaded = load_vision_training_config(&[config])
        .expect("load explicit48 fused efficiency 224 h10 experiment");
    match loaded.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_explicit_steps, vec![4, 8]);
            assert!(!distill.rollout_supervision_include_step1);
            assert!((distill.rollout_supervision_power - 1.25).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn canonical_efficiency_family_sparse248_metriclight_experiment_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/experiments/efficiency/imagenette_dinov2_vits14_dense_h8_deepbias0p5_ff6_280_efficiency_prefetch8_device_preprocessed_teachercache_bs64_fused_scores_sparse248_metriclight.toml",
    );
    let loaded = load_vision_training_config(&[config])
        .expect("load canonical efficiency sparse248 metric-light config");
    match loaded.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_groups, 1);
            assert!(!distill.rollout_supervision_include_step1);
            assert!((distill.rollout_supervision_power - 1.25).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn canonical_archive_efficiency_accum2_experiment_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/experiments/archive/imagenette_dinov2_vits14_dense_h10_deepbias_ff6_224_efficiency_accum2.toml",
    );
    let loaded =
        load_vision_training_config(&[config]).expect("load canonical archive efficiency accum2");
    assert_eq!(loaded.training.gradient_accumulation_steps, 2);
    assert_eq!(loaded.training.max_iters, 128);
}

#[test]
fn canonical_archive_convnext_short_experiment_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/experiments/archive/imagenette_dinov2_vits14_dense_h8_convnext_short.toml",
    );
    let loaded =
        load_vision_training_config(&[config]).expect("load canonical archive convnext short");
    assert_eq!(loaded.training.batch_size, 40);
    assert_eq!(
        loaded.vision.patch_embed_mode,
        VisionPatchEmbedMode::ConvNext
    );
}

#[test]
fn imagenette_dinov2_dense_h8_deepbias0p5_ff6_280_efficiency_bs64_fused_scores_sparse248_metriclight_experiment_loads()
 {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/experiments/imagenette_dinov2_vits14_dense_h8_deepbias0p5_ff6_280_efficiency_prefetch8_device_preprocessed_teachercache_bs64_fused_scores_sparse248_metriclight.toml",
    );
    let loaded = load_vision_training_config(&[config])
        .expect("load sparse248 metric-light fused-scores efficiency 280 h8 experiment");
    match loaded.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_groups, 1);
            assert_eq!(
                distill.rollout_supervision_explicit_groups,
                vec![vec![4, 8], vec![4, 8], vec![4, 8], vec![2, 4, 8]]
            );
            assert!(!distill.rollout_supervision_include_step1);
            assert!((distill.rollout_supervision_power - 1.25).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn canonical_diagnostics_family_smoke_overlay_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/experiments/diagnostics/vision_smoke_dinov2_vits14_280_eval_overlay.toml",
    );
    let loaded =
        load_vision_training_config(&[config]).expect("load canonical diagnostics smoke overlay");
    assert_eq!(
        loaded.dataset.imagenet_root,
        PathBuf::from("data/vision_smoke/imagenet")
    );
    assert_eq!(loaded.training.batch_size, 8);
}

#[test]
fn imagenette_dinov2_dense_h8_deepbias0p5_ff6_280_promoted_multiframe2_stride3_min4_nos1_pow125_experiment_loads()
 {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/experiments/imagenette_dinov2_vits14_dense_h8_deepbias0p5_ff6_280_promoted_multiframe2_stride3_min4_nos1_pow125.toml",
    );
    let loaded = load_vision_training_config(&[config])
        .expect("load multiframe2 stride3 min4 no-s1 pow125 promoted 280 h8 experiment");
    assert_eq!(loaded.training.rollout_min_steps, Some(4));
    match loaded.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_frames, 2);
            assert_eq!(distill.rollout_supervision_stride, 3);
            assert!(!distill.rollout_supervision_include_step1);
            assert!((distill.rollout_supervision_power - 1.25).abs() < f32::EPSILON);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h8_deepbias0p5_ff6_280_promoted_stride2_longer_experiment_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/experiments/imagenette_dinov2_vits14_dense_h8_deepbias0p5_ff6_280_promoted_stride2_longer.toml",
    );
    let loaded = load_vision_training_config(&[config])
        .expect("load stride2 longer promoted 280 h8 experiment");
    assert_eq!(loaded.training.max_iters, 512);
    match loaded.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_stride, 2);
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn vision_smoke_dinov2_dense_h10_deepbias_ff6_224_eval_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/vision_smoke_dinov2_vits14_dense_h10_deepbias_ff6_224_eval.toml",
    );
    let loaded =
        load_vision_training_config(&[config]).expect("load smoke eval promoted 224 config");
    assert_eq!(
        loaded.dataset.imagenet_root,
        PathBuf::from("data/vision_smoke/imagenet")
    );
    let distill = match loaded.mode {
        VisionTrainingModeConfig::Distill(distill) => distill,
        _ => panic!("expected distill mode"),
    };
    let teacher = match distill.teacher {
        VisionTeacherConfig::Features(teacher) => teacher,
        _ => panic!("expected feature teacher"),
    };
    assert!(
        teacher
            .val_patch_path
            .as_ref()
            .expect("teacher patch path")
            .ends_with("data/vision_smoke/features/dinov2_vits14_224/val_patch.bin")
    );
}

#[test]
fn vision_smoke_dinov2_dense_h8_deepbias0p5_ff6_280_eval_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/vision_smoke_dinov2_vits14_dense_h8_deepbias0p5_ff6_280_eval.toml",
    );
    let loaded =
        load_vision_training_config(&[config]).expect("load smoke eval promoted 280 config");
    assert_eq!(
        loaded.dataset.imagenet_root,
        PathBuf::from("data/vision_smoke/imagenet")
    );
    let distill = match loaded.mode {
        VisionTrainingModeConfig::Distill(distill) => distill,
        _ => panic!("expected distill mode"),
    };
    let teacher = match distill.teacher {
        VisionTeacherConfig::Features(teacher) => teacher,
        _ => panic!("expected feature teacher"),
    };
    assert!(
        teacher
            .val_patch_path
            .as_ref()
            .expect("teacher patch path")
            .ends_with("data/vision_smoke/features/dinov2_vits14_280/val_patch.bin")
    );
}

#[test]
fn imagenette_dinov2_pyramid_stageaware_h10_deepbias_promoted_stride2_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_pyramid_stageaware_refine2_coarse96_densepolicy_h10_deepbias_promoted_stride2.toml",
    );
    let loaded = load_vision_training_config(&[config])
        .expect("load pyramid stage-aware stride2 promoted config");
    match loaded.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert_eq!(distill.rollout_supervision_stride, 2);
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h8_deepbias0p625_ff6_280_promoted_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h8_deepbias0p625_ff6_280_promoted.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h8 deepbias0p625 ff6 280 promoted");

    assert_eq!(config.vision.image_size, 280);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 6);
    assert_eq!(config.vision.steps, 8);
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 384);
    assert_eq!(config.training.rollout_max_steps, Some(8));
    match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert!((distill.rollout_sampling_power - 0.625).abs() < f32::EPSILON);
            match &distill.teacher {
                VisionTeacherConfig::Features(teacher) => {
                    assert_eq!(teacher.patch_tokens, Some(400));
                }
                other => panic!("expected feature teacher, got {other:?}"),
            }
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h8_deepbias0p6_ff6_280_promoted_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h8_deepbias0p6_ff6_280_promoted.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h8 deepbias0p6 ff6 280 promoted");

    assert_eq!(config.vision.image_size, 280);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 6);
    assert_eq!(config.vision.steps, 8);
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 384);
    assert_eq!(config.training.rollout_max_steps, Some(8));
    match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert!((distill.rollout_sampling_power - 0.6).abs() < f32::EPSILON);
            match &distill.teacher {
                VisionTeacherConfig::Features(teacher) => {
                    assert_eq!(teacher.patch_tokens, Some(400));
                }
                other => panic!("expected feature teacher, got {other:?}"),
            }
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h9_deepbias0p5_ff6_280_promoted_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h9_deepbias0p5_ff6_280_promoted.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h9 deepbias0p5 ff6 280 promoted");

    assert_eq!(config.vision.image_size, 280);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 6);
    assert_eq!(config.vision.steps, 9);
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 384);
    assert_eq!(config.training.rollout_max_steps, Some(9));
    match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
            match &distill.teacher {
                VisionTeacherConfig::Features(teacher) => {
                    assert_eq!(teacher.patch_tokens, Some(400));
                }
                other => panic!("expected feature teacher, got {other:?}"),
            }
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn imagenette_dinov2_dense_h8_deepbias0p5_ff6_280_promoted_min2_loads() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let config_path = repo_root.join(
        "config/vision/distill/imagenette_dinov2_vits14_dense_h8_deepbias0p5_ff6_280_promoted_min2.toml",
    );
    let config = load_vision_training_config(&[config_path])
        .expect("load dense h8 deepbias0p5 ff6 280 promoted min2");

    assert_eq!(config.vision.image_size, 280);
    assert_eq!(config.vision.mlp_internal_dim_multiplier, 6);
    assert_eq!(config.vision.steps, 8);
    assert_eq!(config.training.epochs, Some(2));
    assert_eq!(config.training.max_iters, 384);
    assert_eq!(config.training.rollout_min_steps, Some(2));
    assert_eq!(config.training.rollout_max_steps, Some(8));
    match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            assert!((distill.rollout_sampling_power - 0.5).abs() < f32::EPSILON);
            match &distill.teacher {
                VisionTeacherConfig::Features(teacher) => {
                    assert_eq!(teacher.patch_tokens, Some(400));
                }
                other => panic!("expected feature teacher, got {other:?}"),
            }
        }
        other => panic!("expected distill mode, got {other:?}"),
    }
}

#[test]
fn trm_predict_bank_schedule_parses_from_config() {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let overlay = r#"
        [vision]
        backbone = "pyramid"

        [vision.trm_graph]
        enabled = true

        [vision.trm_graph.bank_schedule.predict]
        patch_local_read = false
        patch_local_write = false
    "#;
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    let overlay_path = std::env::temp_dir().join(format!("vision-trm-bank-schedule-{unique}.toml"));
    fs::write(&overlay_path, overlay).expect("write overlay");
    let config = load_vision_training_config(&[
        repo_root.join("config/vision/base.toml"),
        overlay_path.clone(),
    ])
    .expect("load config with bank schedule overlay");
    let _ = fs::remove_file(overlay_path);

    assert!(
        !config
            .vision
            .trm_graph
            .bank_schedule
            .predict
            .patch_local_read
    );
    assert!(
        !config
            .vision
            .trm_graph
            .bank_schedule
            .predict
            .patch_local_write
    );
    assert!(
        config
            .vision
            .trm_graph
            .bank_schedule
            .observe
            .patch_local_read
    );
    assert!(
        config
            .vision
            .trm_graph
            .bank_schedule
            .observe
            .patch_local_write
    );
}

#[test]
fn trm_predict_bank_decay_scales_parse_from_config() {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let overlay = r#"
        [vision]
        backbone = "pyramid"

        [vision.trm_graph]
        enabled = true

        [vision.trm_graph.bank_schedule.predict]
        patch_decay_scale = 2.0
        coarse_decay_scale = 0.5
        global_decay_scale = 0.25
    "#;
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    let overlay_path =
        std::env::temp_dir().join(format!("vision-trm-bank-decay-scales-{unique}.toml"));
    fs::write(&overlay_path, overlay).expect("write overlay");
    let config = load_vision_training_config(&[
        repo_root.join("config/vision/base.toml"),
        overlay_path.clone(),
    ])
    .expect("load config with bank schedule overlay");
    let _ = fs::remove_file(overlay_path);

    assert_eq!(
        config
            .vision
            .trm_graph
            .bank_schedule
            .predict
            .patch_decay_scale,
        2.0
    );
    assert_eq!(
        config
            .vision
            .trm_graph
            .bank_schedule
            .predict
            .coarse_decay_scale,
        0.5
    );
    assert_eq!(
        config
            .vision
            .trm_graph
            .bank_schedule
            .predict
            .global_decay_scale,
        0.25
    );
}

#[test]
fn trm_predict_coarse_substeps_parse_from_config() {
    let text = r#"
        [dataset]
        moving_mnist_root = "data/moving_mnist"
        source = "moving_mnist"

        [training]
        batch_size = 4
        max_iters = 4
        log_frequency = 1

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [vision]
        image_size = 32
        patch_size = 4
        in_channels = 3
        embed_dim = 64
        steps = 4
        n_head = 8
        mlp_internal_dim_multiplier = 2
        dropout = 0.0
        projection_dim = 64
        projection_hidden_dim = 64
        pos_encoding = "rope"
        backbone = "pyramid"

        [vision.trm_graph]
        enabled = true
        predict_coarse_substeps = 3
        predict_substep_kind = "local_bridge"

        [mode]
        type = "video_lejepa"
        context_frames = 2
        target_frames = 2
        predict_backprop_frames = 2
    "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse config");
    config.validate().expect("vision config should validate");
    assert_eq!(config.vision.trm_graph.predict_coarse_substeps, 3);
    assert_eq!(
        config.vision.trm_graph.predict_substep_kind,
        VisionTrmPredictSubstepKind::LocalBridge
    );
}

#[test]
fn trm_coarse_local_topology_overrides_parse_from_config() {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let overlay = r#"
        [vision]
        backbone = "pyramid"

        [vision.trm_graph]
        enabled = true
        local_radius = 0
        local_self = true
        coarse_local_radius = 1
        coarse_local_diagonals = false
        coarse_local_self = true
    "#;
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    let overlay_path =
        std::env::temp_dir().join(format!("vision-trm-coarse-topology-{unique}.toml"));
    fs::write(&overlay_path, overlay).expect("write overlay");
    let config = load_vision_training_config(&[
        repo_root.join("config/vision/base.toml"),
        overlay_path.clone(),
    ])
    .expect("load config with coarse topology overlay");
    let _ = fs::remove_file(overlay_path);

    assert_eq!(config.vision.trm_graph.local_radius, 0);
    assert_eq!(config.vision.trm_graph.coarse_local_radius, Some(1));
    assert_eq!(config.vision.trm_graph.coarse_local_diagonals, Some(false));
    assert_eq!(config.vision.trm_graph.coarse_local_self, Some(true));
}

#[test]
fn trm_self_only_local_topology_validates_with_zero_radius() {
    let text = r#"
        [dataset]
        source = "moving_mnist"
        max_records = 16

        [training]
        batch_size = 4
        max_iters = 4
        log_frequency = 1

        [optimizer]
        learning_rate = 0.0003
        weight_decay = 0.0

        [vision]
        backbone = "pyramid"
        image_size = 32
        patch_size = 2
        in_channels = 3
        embed_dim = 64
        steps = 2
        n_head = 8
        projection_dim = 32
        projection_hidden_dim = 64

        [vision.trm_graph]
        enabled = true
        local_radius = 0
        local_self = true
        coarse_local_radius = 1
        coarse_local_self = true
        coarse_stride = 2
        rank = 16
        value_dim = 16
        hub_count = 2

        [mode]
        type = "video_lejepa"

        [mode.video_lejepa]
        context_frames = 2
        target_frames = 2
        predict_backprop_frames = 2
    "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse config");
    config
        .validate()
        .expect("self-only zero-radius local topology should validate");
}

#[test]
fn active_docs_and_program_reference_existing_vision_configs() {
    fn extract_config_paths(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let needle = "config/vision/";
        let bytes = text.as_bytes();
        let mut start = 0usize;
        while let Some(offset) = text[start..].find(needle) {
            let path_start = start + offset;
            let mut end = path_start;
            while end < bytes.len() {
                let ch = bytes[end] as char;
                let valid = ch.is_ascii_alphanumeric() || matches!(ch, '/' | '_' | '-' | '.');
                if !valid {
                    break;
                }
                end += 1;
                if text[path_start..end].ends_with(".toml") {
                    out.push(text[path_start..end].to_string());
                    break;
                }
            }
            start = end.max(path_start + needle.len());
        }
        out.sort();
        out.dedup();
        out
    }

    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let files = [
        repo_root.join("program.md"),
        repo_root.join("docs/vision/vision_dinov2_distill_roadmap.md"),
        repo_root.join("docs/vision/vision_gpu_training_efficiency_roadmap.md"),
        repo_root.join("docs/vision/vision_recurrent_block_training_roadmap.md"),
        repo_root.join("docs/vision/vision_surface_cleanup_roadmap.md"),
        repo_root.join("docs/vision/vision_surface_inventory.md"),
        repo_root.join("docs/video/video_shared_core_trm_roadmap.md"),
    ];

    let mut missing = Vec::new();
    for file in files {
        let text =
            std::fs::read_to_string(&file).unwrap_or_else(|err| panic!("read {:?}: {err}", file));
        for rel_path in extract_config_paths(&text) {
            let abs_path = repo_root.join(&rel_path);
            if !abs_path.exists() {
                missing.push(format!("{} -> {}", file.display(), rel_path));
            }
        }
    }

    assert!(
        missing.is_empty(),
        "missing config references in active docs/program:\n{}",
        missing.join("\n")
    );
}

#[test]
fn distill_rollout_supervision_groups_must_be_positive() {
    let text = r#"
        [dataset]
        source = "imagenet"
        train_dir = "train"
        val_dir = "val"
        max_records = 16

        [training]
        batch_size = 4
        max_iters = 4
        log_frequency = 1

        [optimizer]
        learning_rate = 0.0003
        weight_decay = 0.0

        [vision]
        image_size = 32
        patch_size = 4
        in_channels = 3
        embed_dim = 64
        steps = 4
        n_head = 8
        projection_dim = 32
        projection_hidden_dim = 64

        [mode]
        type = "distill"
        rollout_supervision_groups = 0

        [mode.teacher]
        type = "features"
        train_cls_path = "train_cls.bin"
        train_patch_path = "train_patch.bin"
        val_cls_path = "val_cls.bin"
        val_patch_path = "val_patch.bin"
        feature_dim = 32
        patch_tokens = 64
    "#;

    let config: VisionTrainingConfig = toml::from_str(text).expect("parse config");
    let err = config.validate().expect_err("groups=0 should fail");
    assert!(
        err.to_string()
            .contains("mode.rollout_supervision_groups must be > 0"),
        "unexpected error: {err}"
    );
}
