use super::*;
use burn::data::dataloader::Progress;
use burn::tensor::Tensor;
use burn_ndarray::NdArray;

fn tempdir_path() -> PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("burn_dragon_vision_artifact_{unique}"));
    fs::create_dir_all(&path).expect("create temp dir");
    path
}

fn test_metadata(iteration: usize) -> MetricMetadata {
    MetricMetadata {
        progress: Progress::new(1, 1),
        global_progress: Progress::new(0, 1),
        iteration: Some(iteration),
        lr: None,
    }
}

fn test_metadata_epoch(iteration: usize, epoch: usize) -> MetricMetadata {
    MetricMetadata {
        progress: Progress::new(1, 1),
        global_progress: Progress::new(epoch, 1),
        iteration: Some(iteration),
        lr: None,
    }
}

fn create_stub_ffmpeg(bin_dir: &std::path::Path) -> std::io::Result<PathBuf> {
    let path = bin_dir.join(if cfg!(windows) {
        "ffmpeg.bat"
    } else {
        "ffmpeg"
    });
    let script = if cfg!(windows) {
        "@echo off\r\nset OUT=%~1\r\nshift\r\n:loop\r\nif \"%~1\"==\"\" goto done\r\nset OUT=%~1\r\nshift\r\ngoto loop\r\n:done\r\ntype nul > \"%OUT%\"\r\n"
    } else {
        "#!/bin/sh\nout=\"\"\nfor arg in \"$@\"; do\n  out=\"$arg\"\ndone\n: > \"$out\"\n"
    };
    fs::write(&path, script)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&path)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&path, perms)?;
    }
    Ok(path)
}

#[test]
fn artifact_images_write_png() {
    type Backend = NdArray<f32>;

    let output_dir = tempdir_path();
    let views = Tensor::<Backend, 5>::zeros([1, 2, 3, 4, 4], &Default::default());
    let patch = Tensor::<Backend, 3>::ones([1, 2, 2], &Default::default());
    let mut input = VisionArtifactInput::empty();
    input.views = Some(views);
    input.patch_norms = Some(patch);

    let mut metric = VisionArtifactMetric::<Backend>::new(
        output_dir.clone(),
        1,
        VisionArtifactOutputMode::Images,
        4,
        8,
        [0.5; 3],
        [0.5; 3],
        false,
        None,
    );

    let entry = metric.update(&input, &test_metadata(0));
    assert_eq!(entry.serialized, "1");
    let sample = output_dir.join("lejepa_iter_000000_sample_00.png");
    assert!(sample.exists(), "expected {}", sample.display());
}

#[test]
fn artifact_images_respects_epoch_budget() {
    type Backend = NdArray<f32>;

    let output_dir = tempdir_path();
    let views = Tensor::<Backend, 5>::zeros([2, 2, 3, 4, 4], &Default::default());
    let patch = Tensor::<Backend, 3>::ones([2, 2, 2], &Default::default());
    let mut input = VisionArtifactInput::empty();
    input.views = Some(views);
    input.patch_norms = Some(patch);

    let mut metric = VisionArtifactMetric::<Backend>::new(
        output_dir.clone(),
        1,
        VisionArtifactOutputMode::Images,
        1,
        8,
        [0.5; 3],
        [0.5; 3],
        false,
        None,
    );

    let first = metric.update(&input, &test_metadata_epoch(0, 0));
    assert_eq!(first.serialized, "1");
    let exhausted = metric.update(&input, &test_metadata_epoch(1, 0));
    assert_eq!(exhausted.serialized, "0");
    assert_eq!(exhausted.formatted, "budget_exhausted");
    let reset = metric.update(&input, &test_metadata_epoch(2, 1));
    assert_eq!(reset.serialized, "1");
}

#[test]
fn artifact_avi_write_video() {
    type Backend = NdArray<f32>;

    let output_dir = tempdir_path();
    let ffmpeg_dir = tempdir_path();
    let ffmpeg = create_stub_ffmpeg(&ffmpeg_dir).expect("stub ffmpeg");
    let frames = Tensor::<Backend, 5>::zeros([1, 2, 3, 4, 4], &Default::default());
    let mut input = VisionArtifactInput::empty();
    input.frames = Some(frames);

    let mut metric = VisionArtifactMetric::<Backend>::new(
        output_dir.clone(),
        1,
        VisionArtifactOutputMode::Avi,
        4,
        8,
        [0.5; 3],
        [0.5; 3],
        false,
        Some(ffmpeg),
    );

    let entry = metric.update(&input, &test_metadata(3));
    assert_eq!(entry.serialized, "1");
    let sample = output_dir.join("iter_000003_sample_00.avi");
    assert!(sample.exists(), "expected {}", sample.display());
}

#[test]
fn artifact_avi_write_video_with_debug_reconstruction_and_upscale() {
    type Backend = NdArray<f32>;

    let output_dir = tempdir_path();
    let ffmpeg_dir = tempdir_path();
    let ffmpeg = create_stub_ffmpeg(&ffmpeg_dir).expect("stub ffmpeg");
    let frames = Tensor::<Backend, 5>::zeros([1, 2, 3, 4, 4], &Default::default());
    let debug = Tensor::<Backend, 5>::zeros([1, 2, 3, 4, 4], &Default::default());
    let patch = Tensor::<Backend, 4>::ones([1, 2, 2, 2], &Default::default());
    let pca = Tensor::<Backend, 5>::ones([1, 2, 3, 2, 2], &Default::default());
    let mut input = VisionArtifactInput::empty();
    input.frames = Some(frames);
    input.debug_recon_frames = Some(debug);
    input.patch_norms_steps = Some(patch);
    input.pca_rgb_steps = Some(pca);
    input.artifact_scale = 2;

    let mut metric = VisionArtifactMetric::<Backend>::new(
        output_dir.clone(),
        1,
        VisionArtifactOutputMode::Avi,
        4,
        8,
        [0.5; 3],
        [0.5; 3],
        false,
        Some(ffmpeg),
    );

    let entry = metric.update(&input, &test_metadata(4));
    assert_eq!(entry.serialized, "1");
    let sample = output_dir.join("iter_000004_sample_00.avi");
    assert!(sample.exists(), "expected {}", sample.display());
}

#[test]
fn artifact_avi_write_video_with_posterior_context_overlays() {
    type Backend = NdArray<f32>;

    let output_dir = tempdir_path();
    let ffmpeg_dir = tempdir_path();
    let ffmpeg = create_stub_ffmpeg(&ffmpeg_dir).expect("stub ffmpeg");
    let frames = Tensor::<Backend, 5>::zeros([1, 4, 3, 4, 4], &Default::default());
    let posterior_patch = Tensor::<Backend, 4>::ones([1, 4, 2, 2], &Default::default());
    let posterior_pca = Tensor::<Backend, 5>::ones([1, 4, 3, 2, 2], &Default::default());
    let patch = Tensor::<Backend, 4>::ones([1, 4, 2, 2], &Default::default());
    let pca = Tensor::<Backend, 5>::ones([1, 4, 3, 2, 2], &Default::default());
    let mut input = VisionArtifactInput::empty();
    input.frames = Some(frames);
    input.posterior_patch_norms_steps = Some(posterior_patch);
    input.posterior_pca_rgb_steps = Some(posterior_pca);
    input.patch_norms_steps = Some(patch);
    input.pca_rgb_steps = Some(pca);
    input.prediction_start = Some(2);

    let mut metric = VisionArtifactMetric::<Backend>::new(
        output_dir.clone(),
        1,
        VisionArtifactOutputMode::Avi,
        4,
        8,
        [0.5; 3],
        [0.5; 3],
        false,
        Some(ffmpeg),
    );

    let entry = metric.update(&input, &test_metadata(5));
    assert_eq!(entry.serialized, "1");
    let legend_path = output_dir.join("vision_artifacts_key.txt");
    let legend = fs::read_to_string(legend_path).expect("legend");
    assert!(legend.contains("observed context"));
    assert!(legend.contains("predictive future"));
}

#[test]
fn artifact_mp4_write_video() {
    type Backend = NdArray<f32>;

    let output_dir = tempdir_path();
    let ffmpeg_dir = tempdir_path();
    let ffmpeg = create_stub_ffmpeg(&ffmpeg_dir).expect("stub ffmpeg");
    let frames = Tensor::<Backend, 5>::zeros([1, 2, 3, 4, 4], &Default::default());
    let mut input = VisionArtifactInput::empty();
    input.frames = Some(frames);

    let mut metric = VisionArtifactMetric::<Backend>::new(
        output_dir.clone(),
        1,
        VisionArtifactOutputMode::Mp4,
        4,
        8,
        [0.5; 3],
        [0.5; 3],
        false,
        Some(ffmpeg),
    );

    let entry = metric.update(&input, &test_metadata(6));
    assert_eq!(entry.serialized, "1");
    let sample = output_dir.join("iter_000006_sample_00.mp4");
    assert!(sample.exists(), "expected {}", sample.display());
}
