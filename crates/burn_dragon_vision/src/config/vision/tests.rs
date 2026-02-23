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
            }
            other => panic!("unexpected teacher config: {other:?}"),
        },
        other => panic!("unexpected mode: {other:?}"),
    }
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
fn trm_strict_rejects_local_grid_mismatch() {
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
        .expect_err("strict TRM must reject mismatch");
    let message = format!("{err:#}");
    assert!(message.contains("TRM graph strict mode requires local view grid"));
}

#[test]
fn trm_fallback_allows_local_grid_mismatch() {
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
