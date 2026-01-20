use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution, Tensor};
use burn_dragon::{FusedKernelConfig, ManifoldHyperConnectionsConfig};
use burn_dragon::vision::{
    PatchEmbed, PatchGrid, SpatialPositionalEncodingKind, VisionAttentionMode, VisionDragon,
    VisionDragonConfig, VisionLatentActivation, VisionPatchEmbedMode, pool_patch_tokens,
};
use burn_ndarray::NdArray;

#[test]
fn patch_embed_and_model_shapes() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();

    let config = VisionDragonConfig {
        image_size: 32,
        patch_size: 8,
        patch_embed_mode: VisionPatchEmbedMode::default(),
        in_channels: 3,
        embed_dim: 16,
        steps: 2,
        n_head: 4,
        mlp_internal_dim_multiplier: 2,
        dropout: 0.0,
        projection_dim: 8,
        projection_hidden_dim: 16,
        use_cls_token: true,
        cls_sync_alpha: 0.0,
        num_eyes: 1,
        cross_eye_steps: 0,
        token_state_norm: true,
        latent_activation: VisionLatentActivation::default(),
        pos_encoding: SpatialPositionalEncodingKind::Learned2d,
        pos_max_height: 4,
        pos_max_width: 4,
        attention_mode: VisionAttentionMode::RowL1,
        use_alibi: true,
        fused_kernels: FusedKernelConfig::default(),
        mhc: ManifoldHyperConnectionsConfig::default(),
    };

    let images = Tensor::<Backend, 4>::random([2, 3, 32, 32], Distribution::Default, &device);
    let patch_embed = PatchEmbed::new(&config, &device);
    let patch = patch_embed.forward(images.clone());
    assert_eq!(patch.tokens.shape().dims(), [2, 16, 16]);
    assert_eq!(patch.grid.height, 4);
    assert_eq!(patch.grid.width, 4);

    let model = VisionDragon::<Backend>::new(config, &device);
    let output = model.forward_images(images);
    assert_eq!(output.patch_tokens.shape().dims(), [2, 16, 8]);
    assert_eq!(output.cls_token.shape().dims(), [2, 8]);
}

#[test]
fn patch_embed_raw_matches_add_position() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();

    let config = VisionDragonConfig {
        image_size: 32,
        patch_size: 8,
        patch_embed_mode: VisionPatchEmbedMode::default(),
        in_channels: 3,
        embed_dim: 16,
        steps: 2,
        n_head: 4,
        mlp_internal_dim_multiplier: 2,
        dropout: 0.0,
        projection_dim: 8,
        projection_hidden_dim: 16,
        use_cls_token: true,
        cls_sync_alpha: 0.0,
        num_eyes: 1,
        cross_eye_steps: 0,
        token_state_norm: true,
        latent_activation: VisionLatentActivation::default(),
        pos_encoding: SpatialPositionalEncodingKind::Learned2d,
        pos_max_height: 4,
        pos_max_width: 4,
        attention_mode: VisionAttentionMode::RowL1,
        use_alibi: true,
        fused_kernels: FusedKernelConfig::default(),
        mhc: ManifoldHyperConnectionsConfig::default(),
    };

    let images = Tensor::<Backend, 4>::random([2, 3, 32, 32], Distribution::Default, &device);
    let model = VisionDragon::<Backend>::new(config, &device);
    let raw = model.patch_embed_raw(images.clone());
    let with_pos = model.add_patch_position(raw.tokens.clone(), raw.grid);
    let embedded = model.patch_embed(images).tokens;
    let diff = (embedded - with_pos).powf_scalar(2.0).mean();
    let value = diff
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("diff vec")[0];
    assert!(value < 1e-6);
}

#[test]
fn vision_forward_steps_shapes() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();

    let config = VisionDragonConfig {
        image_size: 32,
        patch_size: 8,
        patch_embed_mode: VisionPatchEmbedMode::default(),
        in_channels: 3,
        embed_dim: 16,
        steps: 3,
        n_head: 4,
        mlp_internal_dim_multiplier: 2,
        dropout: 0.0,
        projection_dim: 8,
        projection_hidden_dim: 16,
        use_cls_token: true,
        cls_sync_alpha: 0.0,
        num_eyes: 1,
        cross_eye_steps: 0,
        token_state_norm: true,
        latent_activation: VisionLatentActivation::default(),
        pos_encoding: SpatialPositionalEncodingKind::Learned2d,
        pos_max_height: 4,
        pos_max_width: 4,
        attention_mode: VisionAttentionMode::RowL1,
        use_alibi: true,
        fused_kernels: FusedKernelConfig::default(),
        mhc: ManifoldHyperConnectionsConfig::default(),
    };

    let images = Tensor::<Backend, 4>::random([2, 3, 32, 32], Distribution::Default, &device);
    let model = VisionDragon::<Backend>::new(config.clone(), &device);

    let out_min = model.forward_images_steps(images.clone(), 1);
    let out_full = model.forward_images_steps(images.clone(), 3);
    let out_clamped = model.forward_images_steps(images.clone(), 8);
    assert_eq!(out_min.patch_tokens.shape().dims(), [2, 16, 8]);
    assert_eq!(out_min.cls_token.shape().dims(), [2, 8]);
    assert_eq!(out_full.patch_tokens.shape().dims(), [2, 16, 8]);
    assert_eq!(out_full.cls_token.shape().dims(), [2, 8]);
    assert_eq!(out_clamped.patch_tokens.shape().dims(), [2, 16, 8]);
    assert_eq!(out_clamped.cls_token.shape().dims(), [2, 8]);

    let patch_embed = PatchEmbed::new(&config, &device);
    let patch = patch_embed.forward(images);
    let embed_out = model.forward_tokens_embed_steps(patch.tokens, 2);
    assert_eq!(embed_out.patch_tokens.shape().dims(), [2, 16, 16]);
    assert_eq!(embed_out.cls_token.shape().dims(), [2, 16]);
}

#[test]
fn pool_patch_tokens_downsamples() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();

    let tokens = Tensor::<Backend, 3>::random([1, 4, 8], Distribution::Default, &device);
    let grid = PatchGrid {
        height: 2,
        width: 2,
    };
    let (pooled, pooled_grid) = pool_patch_tokens(tokens, grid);
    assert_eq!(pooled.shape().dims(), [1, 1, 8]);
    assert_eq!(pooled_grid.height, 1);
    assert_eq!(pooled_grid.width, 1);
}

#[cfg(feature = "train")]
mod train_tests {
    use super::*;
    use burn::data::dataloader::DataLoader;
    use burn_autodiff::Autodiff;
    use burn_dragon::vision::{
        CifarDataset, CifarSplit, CifarType, DinoFeatureStore, ImageNetAugmentations,
        ImageNetDataLoader, ImageNetDataset, ImageNetDatasetConfig, ImageNetSplit, VisionNormalize,
    };
    use burn_dragon::vision::{VisionTrainingModeConfig, load_vision_training_config};
    use image::RgbImage;
    use std::fs;
    use std::path::Path;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn write_f32_file(path: &Path, values: &[f32]) {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(path, bytes).expect("write f32 file");
    }

    #[test]
    fn cifar_batch_shapes() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("cifar-10-batches-bin");
        fs::create_dir_all(&root).expect("create cifar dir");

        let record_len = 1 + 32 * 32 * 3;
        let mut record = vec![0u8; record_len];
        record[0] = 3;
        fs::write(root.join("test_batch.bin"), record).expect("write cifar record");

        let dataset =
            CifarDataset::new(dir.path(), CifarType::Cifar10, CifarSplit::Test).expect("dataset");
        assert_eq!(dataset.len(), 1);

        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let batch = dataset.sample_batch::<Backend>(2, &device);
        assert_eq!(batch.images.shape().dims(), [2, 3, 32, 32]);
        assert_eq!(batch.labels.shape().dims(), [2]);

        let labels = batch
            .labels
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("labels");
        assert!(labels.iter().all(|&value| value == 3));
    }

    #[test]
    fn dino_feature_store_reads_records() {
        let dir = tempdir().expect("tempdir");
        let cls_path = dir.path().join("cls.bin");
        let patch_path = dir.path().join("patch.bin");

        let feature_dim = 3;
        let patch_tokens = 2;
        let cls_values = vec![0.1, 0.2, 0.3, 1.1, 1.2, 1.3];
        let patch_values = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5];
        write_f32_file(&cls_path, &cls_values);
        write_f32_file(&patch_path, &patch_values);

        let store =
            DinoFeatureStore::new(&cls_path, &patch_path, feature_dim, patch_tokens, Some(2))
                .expect("feature store");

        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let (cls, patch) = store
            .load_batch::<Backend>(&[1], &device)
            .expect("load batch");

        assert_eq!(cls.shape().dims::<2>(), [1, 3]);
        assert_eq!(patch.shape().dims::<3>(), [1, 2, 3]);

        let cls_vec = cls
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("cls vec");
        assert_eq!(cls_vec, vec![1.1, 1.2, 1.3]);

        let patch_vec = patch
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("patch vec");
        assert_eq!(patch_vec, vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5]);
    }

    #[test]
    fn imagenet_dataset_batch_shape() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("imagenet");
        let class_dir = root.join("n000000");
        fs::create_dir_all(&class_dir).expect("create imagenet class dir");

        let image_path = class_dir.join("sample.png");
        let image = RgbImage::new(10, 12);
        image.save(&image_path).expect("write image");

        let augmentations = ImageNetAugmentations::new(
            ImageNetSplit::Val,
            8,
            8,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.1,
            2.0,
            0.0,
            128,
        );
        let normalize = VisionNormalize::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let dataset = ImageNetDataset::new(ImageNetDatasetConfig {
            root,
            split: ImageNetSplit::Val,
            max_records: None,
            augmentations,
            local_augmentations: None,
            normalize,
            teacher: None,
            views: 1,
            local_views: 0,
            min_view_overlap: 0.0,
            view_overlap_attempts: 1,
            cache_decoded: false,
            cache_capacity: 0,
            cache_preprocessed: false,
        })
        .expect("imagenet dataset");

        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let batch = dataset.sample_batch::<Backend>(2, &device);
        assert_eq!(batch.images.shape().dims(), [2, 3, 8, 8]);
        assert_eq!(batch.labels.shape().dims(), [2]);

        let labels = batch
            .labels
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("labels");
        assert!(labels.iter().all(|&value| value == 0));
    }

    #[test]
    fn imagenet_multicrop_batch_shape() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("imagenet");
        let class_dir = root.join("n000000");
        fs::create_dir_all(&class_dir).expect("create imagenet class dir");

        let image_path = class_dir.join("sample.png");
        let image = RgbImage::new(8, 8);
        image.save(&image_path).expect("write image");

        let augmentations = ImageNetAugmentations::new(
            ImageNetSplit::Val,
            8,
            8,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.1,
            2.0,
            0.0,
            128,
        );
        let local_augmentations = ImageNetAugmentations::new(
            ImageNetSplit::Val,
            4,
            4,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.1,
            2.0,
            0.0,
            128,
        );
        let normalize = VisionNormalize::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let dataset = ImageNetDataset::new(ImageNetDatasetConfig {
            root,
            split: ImageNetSplit::Val,
            max_records: None,
            augmentations,
            local_augmentations: Some(local_augmentations),
            normalize,
            teacher: None,
            views: 2,
            local_views: 3,
            min_view_overlap: 0.0,
            view_overlap_attempts: 1,
            cache_decoded: false,
            cache_capacity: 0,
            cache_preprocessed: false,
        })
        .expect("imagenet dataset");

        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let batch = dataset.sample_batch::<Backend>(2, &device);
        let global_views = batch.global_view_images.expect("global view images");
        let local_views = batch.local_view_images.expect("local view images");
        assert_eq!(global_views.shape().dims(), [2, 2, 3, 8, 8]);
        assert_eq!(local_views.shape().dims(), [2, 3, 3, 4, 4]);
    }

    #[test]
    fn imagenet_prefetch_loader_batch_shape() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("imagenet");
        let class_dir = root.join("n000000");
        fs::create_dir_all(&class_dir).expect("create imagenet class dir");

        let image_path = class_dir.join("sample.png");
        let image = RgbImage::new(8, 8);
        image.save(&image_path).expect("write image");

        let augmentations = ImageNetAugmentations::new(
            ImageNetSplit::Val,
            8,
            8,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.1,
            2.0,
            0.0,
            128,
        );
        let normalize = VisionNormalize::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let dataset = Arc::new(
            ImageNetDataset::new(ImageNetDatasetConfig {
                root,
                split: ImageNetSplit::Val,
                max_records: None,
                augmentations,
                local_augmentations: None,
                normalize,
                teacher: None,
                views: 1,
                local_views: 0,
                min_view_overlap: 0.0,
                view_overlap_attempts: 1,
                cache_decoded: false,
                cache_capacity: 0,
                cache_preprocessed: false,
            })
            .expect("imagenet dataset"),
        );

        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let loader = ImageNetDataLoader::<Backend>::new(dataset, 2, &device, 2, None, 2, 1, true);
        let mut iter = loader.iter();
        let batch = iter.next().expect("batch");
        assert_eq!(batch.images.shape().dims(), [2, 3, 8, 8]);
        let batch = iter.next().expect("batch");
        assert_eq!(batch.images.shape().dims(), [2, 3, 8, 8]);
    }

    #[test]
    fn lejepa_smoke_config_step() {
        let dir = tempdir().expect("tempdir");
        let imagenet_root = dir.path().join("imagenet");
        let train_dir = imagenet_root.join("train").join("n000000");
        let val_dir = imagenet_root.join("val").join("n000000");
        fs::create_dir_all(&train_dir).expect("create train dir");
        fs::create_dir_all(&val_dir).expect("create val dir");

        let train_image = train_dir.join("sample.png");
        let val_image = val_dir.join("sample.png");
        let image = RgbImage::new(8, 8);
        image.save(&train_image).expect("write train image");
        image.save(&val_image).expect("write val image");

        let root_str = imagenet_root.to_string_lossy().replace('\\', "/");
        let config_text = format!(
            r#"
                [dataset]
                imagenet_root = "{root_str}"
                train_dir = "train"
                val_dir = "val"

                [training]
                batch_size = 2
                max_iters = 1
                log_frequency = 1

                [optimizer]
                learning_rate = 0.001
                weight_decay = 0.0

                [vision]
                image_size = 8
                patch_size = 4
                in_channels = 3
                embed_dim = 16
                steps = 2
                n_head = 2
                mlp_internal_dim_multiplier = 2
                dropout = 0.0
                projection_dim = 8
                projection_hidden_dim = 16
                use_cls_token = true
                pos_encoding = "learned2d"
                attention_mode = "row_l1"
                fused_kernels = false
                relu_threshold = 0.0

                [mode]
                type = "lejepa"
                views = 2
                artifact_every = 0
                artifact_max_images = 2
                artifact_max_views = 2

                [mode.loss.lejepa]
                enabled = true
                lambda = 0.02
                sigreg_knots = 9
                sigreg_t_max = 2.0
                sigreg_proj_dim = 16

                [augment]
                image_size = 8
                resize_short = 8
                min_scale = 0.9
                max_scale = 1.0
                min_aspect_ratio = 0.9
                max_aspect_ratio = 1.1
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
                normalize_mean = [0.0, 0.0, 0.0]
                normalize_std = [1.0, 1.0, 1.0]
            "#
        );
        let config_path = dir.path().join("vision_lejepa.toml");
        fs::write(&config_path, config_text).expect("write config");

        let config =
            load_vision_training_config(&[config_path]).expect("load vision training config");
        let _lejepa = match &config.mode {
            VisionTrainingModeConfig::Lejepa(config) => config,
            other => panic!("expected lejepa mode, got {other:?}"),
        };

        let normalize =
            VisionNormalize::new(config.augment.normalize_mean, config.augment.normalize_std);
        let train_aug = ImageNetAugmentations::new(
            ImageNetSplit::Train,
            config.augment.image_size,
            config.augment.resize_short,
            config.augment.min_scale,
            config.augment.max_scale,
            config.augment.min_aspect_ratio,
            config.augment.max_aspect_ratio,
            config.augment.flip_prob,
            config.augment.color_jitter_prob,
            config.augment.brightness,
            config.augment.contrast,
            config.augment.saturation,
            config.augment.hue,
            config.augment.grayscale_prob,
            config.augment.blur_prob,
            config.augment.blur_sigma_min,
            config.augment.blur_sigma_max,
            config.augment.solarize_prob,
            config.augment.solarize_threshold,
        );
        let dataset = ImageNetDataset::new(ImageNetDatasetConfig {
            root: imagenet_root.join("train"),
            split: ImageNetSplit::Train,
            max_records: None,
            augmentations: train_aug,
            local_augmentations: None,
            normalize,
            teacher: None,
            views: 2,
            local_views: 0,
            min_view_overlap: 0.0,
            view_overlap_attempts: 1,
            cache_decoded: false,
            cache_capacity: 0,
            cache_preprocessed: false,
        })
        .expect("imagenet dataset");

        type Backend = Autodiff<NdArray<f32>>;
        let device = <Backend as BackendTrait>::Device::default();
        let batch = dataset.sample_batch::<Backend>(config.training.batch_size, &device);
        let view_images = batch
            .global_view_images
            .or(batch.view_images)
            .expect("view images");
        assert_eq!(
            view_images.shape().dims(),
            [config.training.batch_size, 2, 3, 8, 8]
        );
        let vision_config = config.vision.build();
        let model = VisionDragon::<Backend>::new(vision_config.clone(), &device);
        let [batch_size, view_count, channels, height, width] = view_images.shape().dims::<5>();
        let mut proj_views = Vec::with_capacity(view_count);
        for view_idx in 0..view_count {
            let view = view_images
                .clone()
                .slice_dim(1, view_idx..view_idx + 1)
                .reshape([batch_size, channels, height, width]);
            let output = model.forward_images(view);
            proj_views.push(output.cls_token.unsqueeze_dim::<3>(0));
        }
        let proj = Tensor::cat(proj_views, 0);
        let mean = proj.clone().mean_dim(0);
        let loss = (proj - mean).powf_scalar(2.0).mean();
        let value = loss
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("loss vec")[0];
        assert!(value.is_finite());
    }
}

