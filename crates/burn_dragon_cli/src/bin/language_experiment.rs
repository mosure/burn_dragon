use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, anyhow};
use burn_dragon_language::{
    ExperimentStageArtifact, ExperimentStageKind, bundle_state_path, load_experiment_bundle_config,
    prepare_language_stage_config, prepare_universality_stage_config, resolve_bundle_root,
    resolve_stage_dir, resolve_training_stage_artifact, resolved_stage_config_path,
    write_resolved_config,
};
use burn_dragon_train::train::pipeline::{BundleExecutionOptions, execute_bundle};
use burn_dragon_universality::generate_nca_corpus;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(author, version, about = "Run staged language experiment bundles")]
struct Args {
    /// Path to the experiment bundle config TOML.
    #[arg(short = 'c', long = "config")]
    config: PathBuf,
    /// Stop once the named stage completes.
    #[arg(long)]
    stop_after_stage: Option<String>,
    /// Ignore the bundle-level skip-completed behavior and rerun completed stages.
    #[arg(long)]
    no_resume_from_last_completed_stage: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = load_experiment_bundle_config(&args.config)?;
    let bundle_root = resolve_bundle_root(&config);

    std::fs::create_dir_all(&bundle_root)
        .with_context(|| format!("failed to create {}", bundle_root.display()))?;

    let bundle_state = execute_bundle(
        &config.name,
        &bundle_root,
        &config.stages,
        &BundleExecutionOptions {
            resume_from_last_completed_stage: config.resume_from_last_completed_stage
                && !args.no_resume_from_last_completed_stage,
            stop_after_stage: args.stop_after_stage.clone(),
        },
        |stage| stage.name.as_str(),
        resolve_stage_dir,
        |stage, dependency_artifacts| validate_stage_dependencies(stage, dependency_artifacts),
        |_, stage, stage_dir, dependency_artifacts| {
            run_stage(&args.config, stage_dir, stage, dependency_artifacts)
        },
    )?;

    println!("bundle: {}", config.name);
    println!("root: {}", bundle_root.display());
    println!("state: {}", bundle_state_path(&bundle_root).display());
    if let Some(stage) = &bundle_state.latest_completed_stage {
        println!("latest_completed_stage: {stage}");
    }
    for stage in &bundle_state.stages {
        println!(
            "- {}: {:?}{}",
            stage.stage_name,
            stage.status,
            stage
                .last_error
                .as_ref()
                .map(|err| format!(" ({err})"))
                .unwrap_or_default()
        );
    }
    Ok(())
}

fn validate_stage_dependencies(
    stage: &burn_dragon_language::ExperimentStageConfig,
    dependency_artifacts: &BTreeMap<String, ExperimentStageArtifact>,
) -> Result<()> {
    for dependency in &stage.depends_on {
        let Some(artifact) = dependency_artifacts.get(dependency) else {
            return Err(anyhow!(
                "stage `{}` requires dependency `{dependency}` to be completed first",
                stage.name
            ));
        };
        if artifact.manifest_path.is_none()
            && artifact.latest_checkpoint_dir.is_none()
            && artifact.latest_run_dir.is_none()
        {
            return Err(anyhow!(
                "dependency `{dependency}` for stage `{}` has no usable artifact",
                stage.name
            ));
        }
    }
    Ok(())
}

fn run_stage(
    bundle_config_path: &Path,
    stage_dir: &Path,
    stage: &burn_dragon_language::ExperimentStageConfig,
    dependency_artifacts: &BTreeMap<String, ExperimentStageArtifact>,
) -> Result<ExperimentStageArtifact> {
    match &stage.kind {
        ExperimentStageKind::UniversalityGenerate { config } => {
            let resolved =
                prepare_universality_stage_config(bundle_config_path, stage_dir, config)?;
            let resolved_path = resolved_stage_config_path(stage_dir);
            write_resolved_config(&resolved_path, &resolved)?;
            let report = generate_nca_corpus(&resolved)?;
            Ok(ExperimentStageArtifact {
                corpus_output_dir: Some(resolved.output_dir.clone()),
                manifest_path: Some(report.manifest_path),
                sample_records_path: Some(report.sample_records_path),
                preview_dir: Some(report.preview_dir),
                resolved_config_path: Some(resolved_path),
                ..ExperimentStageArtifact::default()
            })
        }
        ExperimentStageKind::LanguageTrain {
            config, backend, ..
        } => {
            let resolved = prepare_language_stage_config(
                bundle_config_path,
                config,
                stage_dir,
                stage,
                dependency_artifacts,
            )?;
            let resolved_path = resolved_stage_config_path(stage_dir);
            write_resolved_config(&resolved_path, &resolved)?;
            run_language_train_child(stage_dir, &resolved_path, *backend)?;
            let mut artifact = resolve_training_stage_artifact(stage_dir)?;
            artifact.resolved_config_path = Some(resolved_path);
            Ok(artifact)
        }
    }
}

fn run_language_train_child(
    stage_dir: &Path,
    resolved_config_path: &Path,
    backend: burn_dragon_language::ExperimentBackend,
) -> Result<()> {
    let current_exe = std::env::current_exe().context("resolve current executable")?;
    let train_binary = current_exe
        .parent()
        .ok_or_else(|| anyhow!("failed to locate executable directory"))?
        .join("language_train");
    if !train_binary.is_file() {
        return Err(anyhow!(
            "language_train binary not found at {}; build it first",
            train_binary.display()
        ));
    }

    let run_root = stage_dir.join("runs");
    std::fs::create_dir_all(&run_root)
        .with_context(|| format!("failed to create {}", run_root.display()))?;
    let status = Command::new(&train_binary)
        .arg("language")
        .arg("--backend")
        .arg(backend.as_cli_arg())
        .arg("-c")
        .arg(resolved_config_path)
        .env("BURN_DRAGON_RUN_ROOT", &run_root)
        .status()
        .with_context(|| format!("failed to launch {}", train_binary.display()))?;
    if !status.success() {
        return Err(anyhow!(
            "language_train exited with status {} for {}",
            status,
            resolved_config_path.display()
        ));
    }
    Ok(())
}
