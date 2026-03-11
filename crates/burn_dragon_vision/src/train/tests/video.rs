use super::*;

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn video_trm_artifact_validate_smoke_writes_mp4_and_legend() {
    type Backend = Autodiff<NdArray<f32>>;

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root");
    let config = load_vision_training_config(&[
        repo_root.join("config/vision/base.toml"),
        repo_root.join("config/vision/video_lejepa/moving_mnist_trm_artifact_validate.toml"),
    ])
    .expect("load TRM artifact validate config");

    let run_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("runs")
        .join("vision");
    let previous_latest = fs::read_to_string(run_root.join("latest")).ok();

    train_vision_backend::<Backend, _>(&config, "cpu-test", |_| {})
        .expect("video TRM artifact smoke training should complete");

    let latest_name = fs::read_to_string(run_root.join("latest"))
        .expect("latest run name")
        .trim()
        .to_string();
    let run_dir = run_root.join(&latest_name);
    let artifact_dir = run_dir.join("artifacts");
    let artifact_path = artifact_dir.join("sample_00.mp4");
    let key_path = artifact_dir.join("vision_artifacts_key.txt");

    assert!(run_dir.is_dir(), "run dir missing: {}", run_dir.display());
    assert!(
        artifact_path.is_file(),
        "artifact mp4 missing: {}",
        artifact_path.display()
    );
    assert!(
        key_path.is_file(),
        "artifact key missing: {}",
        key_path.display()
    );

    let key_contents = fs::read_to_string(&key_path).expect("read legend");
    assert!(key_contents.contains("posterior_context_state_pca_rgb"));
    assert!(key_contents.contains("posterior columns show the post-merge state"));
    assert!(key_contents.contains("predictive future"));

    if let Some(previous_latest) = previous_latest {
        fs::write(run_root.join("latest"), previous_latest).expect("restore latest");
    }
}
