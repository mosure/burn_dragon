use std::ffi::OsStr;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use names::Generator;
use serde::{Deserialize, Serialize};

use crate::config::RunLayoutConfig;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TrainingLaunchMode {
    #[default]
    Fresh,
    ResumeExactRun,
    ResumeLatestCheckpointIfPresent,
    InitFromCheckpoint,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedRunArtifacts {
    pub run_root: PathBuf,
    pub run_dir: PathBuf,
    pub run_name: String,
}

pub fn resolve_backend_partition_run_root(run_root: &Path, backend_name: &str) -> PathBuf {
    match backend_name {
        "cpu" => run_root.join("cpu"),
        "cuda" => run_root.join("cuda"),
        "wgpu" | "wgpu-nofusion" | "wgpu-fused-core" => run_root.join("wgpu"),
        _ => run_root.to_path_buf(),
    }
}

pub fn latest_run_marker_path(run_root: &Path) -> PathBuf {
    run_root.join("latest")
}

pub fn resolve_latest_run_name_in(run_root: &Path) -> Option<String> {
    let contents = fs::read_to_string(latest_run_marker_path(run_root)).ok()?;
    let name = contents.trim();
    (!name.is_empty()).then(|| name.to_string())
}

pub fn resolve_run_root_for_config_paths(
    domain: &str,
    run_layout: &RunLayoutConfig,
    config_paths: &[PathBuf],
) -> PathBuf {
    let mut run_root = run_layout
        .base_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("runs"))
        .join(domain);

    let category = run_layout
        .category
        .as_deref()
        .and_then(|path| normalize_run_category(domain, path))
        .or_else(|| {
            run_layout
                .mirror_config_path
                .then(|| derive_run_category_from_config_paths(domain, config_paths))
                .flatten()
        });

    if let Some(category) = category {
        run_root = run_root.join(category);
    }

    if let Some(bundle) = normalize_relative_path_option(run_layout.bundle.as_deref()) {
        run_root = run_root.join("bundles").join(bundle);
    }
    if let Some(stage) = normalize_relative_path_option(run_layout.stage.as_deref()) {
        run_root = run_root.join("stages").join(stage);
    }
    if let Some(variant) = normalize_relative_path_option(run_layout.variant.as_deref()) {
        run_root = run_root.join("variants").join(variant);
    }

    run_root
}

pub fn derive_run_category_from_config_paths(
    domain: &str,
    config_paths: &[PathBuf],
) -> Option<PathBuf> {
    config_paths
        .iter()
        .rev()
        .find_map(|path| derive_run_category_from_config_path(domain, path))
}

pub fn derive_run_category_from_config_path(domain: &str, config_path: &Path) -> Option<PathBuf> {
    let components: Vec<_> = config_path.components().collect();
    let mut config_root_index = None;
    let mut local_root_index = None;

    for index in 0..components.len().saturating_sub(1) {
        if components[index].as_os_str() == OsStr::new("config")
            && components[index + 1].as_os_str() == OsStr::new(domain)
        {
            config_root_index = Some(index);
        }
        if components[index].as_os_str() == OsStr::new("config")
            && components[index + 1].as_os_str() == OsStr::new("local")
        {
            local_root_index = Some(index);
        }
    }

    let stem = config_path.file_stem()?;
    if let Some(start) = config_root_index {
        let mut category = PathBuf::new();
        for component in &components[start + 2..components.len().saturating_sub(1)] {
            if let Component::Normal(segment) = component {
                category.push(segment);
            }
        }
        category.push(stem);
        return normalize_relative_path(&category);
    }

    let start = local_root_index?;
    let mut category = PathBuf::from("local");
    let local_components = &components[start + 2..components.len().saturating_sub(1)];
    let mut local_index = 0usize;
    if let Some(Component::Normal(segment)) = local_components.first()
        && *segment == OsStr::new(domain)
    {
        local_index = 1;
    }
    for component in &local_components[local_index..] {
        if let Component::Normal(segment) = component {
            category.push(segment);
        }
    }
    category.push(stem);
    normalize_relative_path(&category)
}

pub fn create_run_dir(run_root: &Path) -> Result<(PathBuf, String)> {
    let mut generator = Generator::default();

    for _ in 0..64 {
        let name = generator
            .next()
            .unwrap_or_else(|| "nameless-dragon".to_string());
        let candidate = run_root.join(&name);
        if !candidate.exists() {
            return Ok((candidate, name));
        }
    }

    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| anyhow!("failed to read system time: {err}"))?
        .as_secs();
    let name = format!("run-{suffix}");
    Ok((run_root.join(&name), name))
}

pub fn derive_run_name(run_dir: &Path) -> Result<String> {
    run_dir
        .file_name()
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow!("failed to derive run name from {}", run_dir.display()))
}

pub fn plan_run_artifacts(
    run_root: &Path,
    resume_run_dir: Option<&Path>,
) -> Result<PlannedRunArtifacts> {
    match resume_run_dir {
        Some(run_dir) => {
            fs::create_dir_all(run_dir).with_context(|| {
                format!(
                    "failed to create resume run directory {}",
                    run_dir.display()
                )
            })?;
            Ok(PlannedRunArtifacts {
                run_root: run_root.to_path_buf(),
                run_dir: run_dir.to_path_buf(),
                run_name: derive_run_name(run_dir)?,
            })
        }
        None => {
            let (run_dir, run_name) = create_run_dir(run_root)?;
            fs::create_dir_all(&run_dir).with_context(|| {
                format!(
                    "failed to create planned run directory {}",
                    run_dir.display()
                )
            })?;
            Ok(PlannedRunArtifacts {
                run_root: run_root.to_path_buf(),
                run_dir,
                run_name,
            })
        }
    }
}

pub fn activate_planned_run(planned_run: &PlannedRunArtifacts) -> Result<()> {
    write_latest_run(&planned_run.run_root, &planned_run.run_name)
}

pub fn write_latest_run(run_root: &Path, run_name: &str) -> Result<()> {
    fs::create_dir_all(run_root)
        .with_context(|| format!("failed to create run directory {}", run_root.display()))?;
    let path = latest_run_marker_path(run_root);
    fs::write(&path, run_name)
        .with_context(|| format!("failed to write latest run {}", path.display()))?;
    Ok(())
}

pub fn resolve_latest_run_dir_in(run_root: &Path) -> Option<PathBuf> {
    resolve_latest_run_name_in(run_root).map(|name| run_root.join(name))
}

pub fn require_latest_run_dir_in(run_root: &Path) -> Result<PathBuf> {
    let run_dir = resolve_latest_run_dir_in(run_root).ok_or_else(|| {
        anyhow!(
            "no latest run checkpoint family found under {}; start a run first or pass an explicit resume run directory",
            run_root.display()
        )
    })?;
    if !run_dir.is_dir() {
        return Err(anyhow!(
            "latest run directory does not exist or is not a directory: {}",
            run_dir.display()
        ));
    }
    Ok(run_dir)
}

pub fn run_dir_has_any_checkpoint(run_dir: &Path) -> bool {
    let checkpoint_dir = run_dir.join("checkpoint");
    let Ok(entries) = fs::read_dir(&checkpoint_dir) else {
        return false;
    };

    entries.filter_map(Result::ok).any(|entry| {
        entry
            .path()
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| {
                name.starts_with("model-") && (name.ends_with(".bin") || name.ends_with(".bin.gz"))
            })
    })
}

pub fn resolve_resume_run_dir(
    run_root: &Path,
    explicit_resume_run_dir: Option<&Path>,
    launch_mode: TrainingLaunchMode,
) -> Result<Option<PathBuf>> {
    match launch_mode {
        TrainingLaunchMode::Fresh | TrainingLaunchMode::InitFromCheckpoint => {
            if let Some(run_dir) = explicit_resume_run_dir {
                return Err(anyhow!(
                    "training.launch_mode = \"{}\" cannot be combined with training.resume_run_dir ({})",
                    match launch_mode {
                        TrainingLaunchMode::Fresh => "fresh",
                        TrainingLaunchMode::InitFromCheckpoint => "init_from_checkpoint",
                        _ => unreachable!(),
                    },
                    run_dir.display()
                ));
            }
            Ok(None)
        }
        TrainingLaunchMode::ResumeExactRun => {
            let run_dir = explicit_resume_run_dir.ok_or_else(|| {
                anyhow!(
                    "training.launch_mode = \"resume_exact_run\" requires training.resume_run_dir"
                )
            })?;
            if !run_dir.is_dir() {
                return Err(anyhow!(
                    "training.resume_run_dir does not exist or is not a directory: {}",
                    run_dir.display()
                ));
            }
            Ok(Some(run_dir.to_path_buf()))
        }
        TrainingLaunchMode::ResumeLatestCheckpointIfPresent => {
            if explicit_resume_run_dir.is_some() {
                return Err(anyhow!(
                    "training.launch_mode = \"resume_latest_checkpoint_if_present\" cannot be combined with training.resume_run_dir"
                ));
            }

            let Some(run_dir) = resolve_latest_run_dir_in(run_root) else {
                return Ok(None);
            };
            Ok(run_dir_has_any_checkpoint(&run_dir).then_some(run_dir))
        }
    }
}

fn normalize_relative_path(path: &Path) -> Option<PathBuf> {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        if let Component::Normal(segment) = component {
            normalized.push(segment);
        }
    }
    (!normalized.as_os_str().is_empty()).then_some(normalized)
}

fn normalize_relative_path_option(path: Option<&Path>) -> Option<PathBuf> {
    path.and_then(normalize_relative_path)
}

fn normalize_run_category(domain: &str, path: &Path) -> Option<PathBuf> {
    let normalized = normalize_relative_path(path)?;
    let mut components = normalized.components();
    match components.next() {
        Some(Component::Normal(segment)) if segment == OsStr::new(domain) => {
            let mut trimmed = PathBuf::new();
            for component in components {
                if let Component::Normal(segment) = component {
                    trimmed.push(segment);
                }
            }
            (!trimmed.as_os_str().is_empty()).then_some(trimmed)
        }
        _ => Some(normalized),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        TrainingLaunchMode, activate_planned_run, derive_run_category_from_config_path,
        derive_run_category_from_config_paths, latest_run_marker_path, plan_run_artifacts,
        require_latest_run_dir_in, resolve_backend_partition_run_root, resolve_latest_run_dir_in,
        resolve_latest_run_name_in, resolve_resume_run_dir, resolve_run_root_for_config_paths,
        run_dir_has_any_checkpoint,
    };
    use crate::config::RunLayoutConfig;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn resolves_run_root_by_mirroring_last_matching_config_stem() {
        let run_root = resolve_run_root_for_config_paths(
            "vision",
            &RunLayoutConfig::default(),
            &[
                PathBuf::from("config/vision/base.toml"),
                PathBuf::from("config/vision/video_lejepa/baselines/vjepa21_dense_promoted.toml"),
            ],
        );

        assert_eq!(
            run_root,
            PathBuf::from("runs/vision/video_lejepa/baselines/vjepa21_dense_promoted")
        );
    }

    #[test]
    fn ignores_non_repo_overlays_when_mirroring_config_tree() {
        let derived = derive_run_category_from_config_paths(
            "language",
            &[
                PathBuf::from("config/language/base.toml"),
                PathBuf::from("config/language/baselines/current_best_large.toml"),
                PathBuf::from("/tmp/runtime-overlay.toml"),
            ],
        );

        assert_eq!(derived, Some(PathBuf::from("baselines/current_best_large")));
    }

    #[test]
    fn mirrors_local_overlay_configs_under_local_namespace() {
        let derived = derive_run_category_from_config_path(
            "language",
            &PathBuf::from("config/local/shakespeare_kernel_ablation/mamba2.toml"),
        );

        assert_eq!(
            derived,
            Some(PathBuf::from("local/shakespeare_kernel_ablation/mamba2"))
        );
    }

    #[test]
    fn mirrors_domain_scoped_local_overlay_without_duplicate_domain_segment() {
        let derived = derive_run_category_from_config_path(
            "language",
            &PathBuf::from("config/local/language/sequence_kernels/mamba2.toml"),
        );

        assert_eq!(
            derived,
            Some(PathBuf::from("local/sequence_kernels/mamba2"))
        );
    }

    #[test]
    fn explicit_category_overrides_config_mirroring() {
        let run_root = resolve_run_root_for_config_paths(
            "vision",
            &RunLayoutConfig {
                base_dir: None,
                category: Some(PathBuf::from("vjepa/moving_mnist")),
                mirror_config_path: true,
                bundle: None,
                stage: None,
                variant: None,
            },
            &[PathBuf::from(
                "config/vision/video_lejepa/baselines/vjepa21_dense_promoted.toml",
            )],
        );

        assert_eq!(run_root, PathBuf::from("runs/vision/vjepa/moving_mnist"));
    }

    #[test]
    fn explicit_category_strips_matching_domain_prefix() {
        let run_root = resolve_run_root_for_config_paths(
            "vision",
            &RunLayoutConfig {
                base_dir: None,
                category: Some(PathBuf::from("vision/video_lejepa/vjepa21/imagenet1k")),
                mirror_config_path: false,
                bundle: None,
                stage: None,
                variant: None,
            },
            &[PathBuf::from(
                "config/vision/video_lejepa/baselines/vjepa21_imagenet1k_dense_long.toml",
            )],
        );

        assert_eq!(
            run_root,
            PathBuf::from("runs/vision/video_lejepa/vjepa21/imagenet1k")
        );
    }

    #[test]
    fn run_layout_appends_bundle_stage_and_variant_segments() {
        let run_root = resolve_run_root_for_config_paths(
            "language",
            &RunLayoutConfig {
                base_dir: None,
                category: Some(PathBuf::from("ablations/shakespeare")),
                mirror_config_path: false,
                bundle: Some(PathBuf::from("kernel_compare")),
                stage: Some(PathBuf::from("train")),
                variant: Some(PathBuf::from("mamba2")),
            },
            &[],
        );

        assert_eq!(
            run_root,
            PathBuf::from(
                "runs/language/ablations/shakespeare/bundles/kernel_compare/stages/train/variants/mamba2"
            )
        );
    }

    #[test]
    fn derive_category_uses_file_stem_even_for_domain_root_configs() {
        let derived = derive_run_category_from_config_path(
            "language",
            &PathBuf::from("config/language/base.toml"),
        );
        assert_eq!(derived, Some(PathBuf::from("base")));
    }

    #[test]
    fn plan_run_artifacts_creates_named_run_dir() {
        let dir = tempdir().expect("tempdir");
        let run_root = dir.path().join("runs").join("vision").join("video_lejepa");

        let planned = plan_run_artifacts(&run_root, None).expect("plan run artifacts");

        assert_eq!(planned.run_root, run_root);
        assert!(planned.run_dir.starts_with(&planned.run_root));
        assert!(planned.run_dir.is_dir());
        assert_eq!(
            planned.run_dir.file_name().and_then(|value| value.to_str()),
            Some(planned.run_name.as_str())
        );
    }

    #[test]
    fn activate_planned_run_writes_latest_marker() {
        let dir = tempdir().expect("tempdir");
        let run_root = dir.path().join("runs").join("vision");
        let planned = plan_run_artifacts(&run_root, None).expect("plan run");

        activate_planned_run(&planned).expect("activate planned run");

        assert_eq!(
            resolve_latest_run_name_in(&run_root),
            Some(planned.run_name.clone())
        );
    }

    #[test]
    fn resolve_backend_partition_run_root_groups_known_backends() {
        let base = PathBuf::from("runs/multimodal");

        assert_eq!(
            resolve_backend_partition_run_root(&base, "cpu"),
            PathBuf::from("runs/multimodal/cpu")
        );
        assert_eq!(
            resolve_backend_partition_run_root(&base, "cuda"),
            PathBuf::from("runs/multimodal/cuda")
        );
        assert_eq!(
            resolve_backend_partition_run_root(&base, "wgpu-fused-core"),
            PathBuf::from("runs/multimodal/wgpu")
        );
        assert_eq!(
            resolve_backend_partition_run_root(&base, "custom"),
            PathBuf::from("runs/multimodal")
        );
    }

    #[test]
    fn resolve_latest_run_dir_in_reads_latest_marker() {
        let dir = tempdir().expect("tempdir");
        let run_root = dir.path().join("runs").join("language").join("family");
        let run_dir = run_root.join("serious-dragon");
        fs::create_dir_all(&run_dir).expect("create run dir");
        fs::write(latest_run_marker_path(&run_root), "serious-dragon").expect("write latest");

        assert_eq!(resolve_latest_run_dir_in(&run_root), Some(run_dir));
    }

    #[test]
    fn resolve_latest_run_name_in_reads_latest_marker() {
        let dir = tempdir().expect("tempdir");
        let run_root = dir.path().join("runs").join("language").join("family");
        fs::create_dir_all(&run_root).expect("create run root");
        fs::write(latest_run_marker_path(&run_root), "serious-dragon").expect("write latest");

        assert_eq!(
            resolve_latest_run_name_in(&run_root),
            Some("serious-dragon".to_string())
        );
    }

    #[test]
    fn require_latest_run_dir_in_rejects_missing_latest_marker() {
        let dir = tempdir().expect("tempdir");
        let run_root = dir.path().join("runs").join("language").join("family");
        fs::create_dir_all(&run_root).expect("create run root");

        let err = require_latest_run_dir_in(&run_root).expect_err("missing latest should fail");
        assert!(
            err.to_string()
                .contains("no latest run checkpoint family found"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn resolve_resume_run_dir_prefers_explicit_directory() {
        let dir = tempdir().expect("tempdir");
        let run_root = dir.path().join("runs").join("language").join("family");
        let explicit = run_root.join("resume-me");
        fs::create_dir_all(&explicit).expect("create explicit run dir");

        let resolved = resolve_resume_run_dir(
            &run_root,
            Some(&explicit),
            TrainingLaunchMode::ResumeExactRun,
        )
        .expect("resolve explicit resume run dir");
        assert_eq!(resolved, Some(explicit));
    }

    #[test]
    fn resolve_resume_run_dir_uses_latest_when_requested() {
        let dir = tempdir().expect("tempdir");
        let run_root = dir.path().join("runs").join("language").join("family");
        let run_dir = run_root.join("serious-dragon");
        fs::create_dir_all(&run_dir).expect("create run dir");
        fs::write(latest_run_marker_path(&run_root), "serious-dragon").expect("write latest");
        fs::create_dir_all(run_dir.join("checkpoint")).expect("checkpoint dir");
        fs::write(run_dir.join("checkpoint/model-2.bin"), b"checkpoint").expect("checkpoint");

        let resolved = resolve_resume_run_dir(
            &run_root,
            None,
            TrainingLaunchMode::ResumeLatestCheckpointIfPresent,
        )
        .expect("resolve latest resume run dir");
        assert_eq!(resolved, Some(run_dir));
    }

    #[test]
    fn resolve_resume_run_dir_ignores_latest_run_without_checkpoint() {
        let dir = tempdir().expect("tempdir");
        let run_root = dir.path().join("runs").join("language").join("family");
        let run_dir = run_root.join("serious-dragon");
        fs::create_dir_all(&run_dir).expect("create run dir");
        fs::write(latest_run_marker_path(&run_root), "serious-dragon").expect("write latest");

        let resolved = resolve_resume_run_dir(
            &run_root,
            None,
            TrainingLaunchMode::ResumeLatestCheckpointIfPresent,
        )
        .expect("resolve latest resume run dir");
        assert_eq!(resolved, None);
    }

    #[test]
    fn run_dir_has_any_checkpoint_detects_model_checkpoint_files() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        fs::create_dir_all(run_dir.join("checkpoint")).expect("checkpoint dir");
        assert!(!run_dir_has_any_checkpoint(&run_dir));

        fs::write(run_dir.join("checkpoint/model-1.bin"), b"checkpoint").expect("checkpoint");
        assert!(run_dir_has_any_checkpoint(&run_dir));
    }
}
