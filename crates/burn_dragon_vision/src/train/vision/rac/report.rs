use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
struct MetricSummary {
    name: String,
    path: String,
    count: usize,
    mean: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
    last: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct BestCheckpointRef {
    epoch: usize,
    path: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct DirectionBreakdown {
    recon_loss: Option<MetricSummary>,
    path_loss: Option<MetricSummary>,
    velocity_loss: Option<MetricSummary>,
    psnr: Option<MetricSummary>,
}

#[derive(Debug, Clone, Serialize)]
struct ReverseBreakdown {
    latent_loss: Option<MetricSummary>,
    to_init_loss: Option<MetricSummary>,
}

#[derive(Debug, Clone, Serialize)]
struct RoundtripBreakdown {
    recon_loss: Option<MetricSummary>,
    psnr: Option<MetricSummary>,
}

#[derive(Debug, Clone, Serialize)]
struct ProbeBreakdown {
    loss: Option<MetricSummary>,
    accuracy: Option<MetricSummary>,
}

#[derive(Debug, Clone, Serialize)]
struct RacBestCheckpointReport {
    schema: &'static str,
    run_name: String,
    run_dir: String,
    selection_rule: &'static str,
    best_epoch: Option<usize>,
    checkpoint: Option<BestCheckpointRef>,
    primary_metric: Option<MetricSummary>,
    forward: DirectionBreakdown,
    reverse: ReverseBreakdown,
    roundtrip: RoundtripBreakdown,
    probe: ProbeBreakdown,
    artifacts: Vec<String>,
    notes: Vec<String>,
}

fn rel_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn parse_metric_log(run_dir: &Path, path: &Path, name: &str) -> Result<Option<MetricSummary>> {
    if !path.exists() {
        return Ok(None);
    }
    let contents =
        fs::read_to_string(path).with_context(|| format!("read metric log {}", path.display()))?;
    let mut values = Vec::new();
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value_str = trimmed.split(',').next().unwrap_or(trimmed).trim();
        if let Ok(value) = value_str.parse::<f64>() {
            values.push(value);
        }
    }
    if values.is_empty() {
        return Ok(None);
    }
    let count = values.len();
    let sum: f64 = values.iter().sum();
    let mean = sum / count as f64;
    let min = values.iter().copied().reduce(f64::min);
    let max = values.iter().copied().reduce(f64::max);
    let last = values.last().copied();
    Ok(Some(MetricSummary {
        name: name.to_string(),
        path: rel_path(run_dir, path),
        count,
        mean: Some(mean),
        min,
        max,
        last,
    }))
}

fn valid_epoch_dirs(run_dir: &Path) -> Result<Vec<(usize, PathBuf)>> {
    let mut epochs = Vec::new();
    let valid_dir = run_dir.join("valid");
    if !valid_dir.exists() {
        return Ok(epochs);
    }
    for entry in fs::read_dir(&valid_dir)
        .with_context(|| format!("read valid dir {}", valid_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        let Some(raw_epoch) = name.strip_prefix("epoch-") else {
            continue;
        };
        let Ok(epoch) = raw_epoch.parse::<usize>() else {
            continue;
        };
        epochs.push((epoch, path));
    }
    epochs.sort_by_key(|(epoch, _)| *epoch);
    Ok(epochs)
}

fn collect_artifact_paths(run_dir: &Path) -> Result<Vec<String>> {
    let artifacts_dir = run_dir.join("artifacts");
    if !artifacts_dir.exists() {
        return Ok(Vec::new());
    }
    let mut artifacts = Vec::new();
    for entry in fs::read_dir(&artifacts_dir)
        .with_context(|| format!("read artifacts dir {}", artifacts_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if matches!(
            name,
            "rac_best_checkpoint_report.json" | "rac_best_checkpoint_report.md"
        ) {
            continue;
        }
        artifacts.push(rel_path(run_dir, &path));
    }
    artifacts.sort();
    Ok(artifacts)
}

fn checkpoint_ref(run_dir: &Path, epoch: usize) -> BestCheckpointRef {
    let checkpoint_path = run_dir
        .join("checkpoint")
        .join(format!("model-{epoch}.bin"));
    BestCheckpointRef {
        epoch,
        path: checkpoint_path
            .exists()
            .then(|| rel_path(run_dir, &checkpoint_path)),
    }
}

fn build_report(run_dir: &Path, run_name: &str) -> Result<RacBestCheckpointReport> {
    let epoch_dirs = valid_epoch_dirs(run_dir)?;
    let mut best_epoch = None;
    let mut primary_metric = None;
    let mut best_score = f64::INFINITY;

    for (epoch, epoch_dir) in &epoch_dirs {
        let loss_path = epoch_dir.join("Loss.log");
        let Some(summary) = parse_metric_log(run_dir, &loss_path, "Loss")? else {
            continue;
        };
        let score = summary.mean.or(summary.last).unwrap_or(f64::INFINITY);
        if score < best_score {
            best_score = score;
            best_epoch = Some(*epoch);
            primary_metric = Some(summary);
        }
    }

    let checkpoint = best_epoch.map(|epoch| checkpoint_ref(run_dir, epoch));
    let mut notes = Vec::new();
    notes.push(
        "best epoch is selected by lowest validation mean Loss across all batches in that epoch"
            .to_string(),
    );
    notes.push(
        "rac_inv_loss corresponds to forward decode reconstruction loss in the current RAC trainer"
            .to_string(),
    );
    notes.push(
        "artifact files may reflect the latest validation pass rather than the selected best epoch when artifact_overwrite=true"
            .to_string(),
    );

    let metric_for =
        |epoch: Option<usize>, file_name: &str, name: &str| -> Result<Option<MetricSummary>> {
            let Some(epoch) = epoch else {
                return Ok(None);
            };
            parse_metric_log(
                run_dir,
                &run_dir
                    .join("valid")
                    .join(format!("epoch-{epoch}"))
                    .join(file_name),
                name,
            )
        };

    Ok(RacBestCheckpointReport {
        schema: "vision_rac_best_checkpoint_report_v1",
        run_name: run_name.to_string(),
        run_dir: run_dir.display().to_string(),
        selection_rule: "lowest validation mean Loss",
        best_epoch,
        checkpoint,
        primary_metric,
        forward: DirectionBreakdown {
            recon_loss: metric_for(best_epoch, "rac_inv_loss.log", "rac_inv_loss")?,
            path_loss: metric_for(
                best_epoch,
                "rac_forward_path_loss.log",
                "rac_forward_path_loss",
            )?,
            velocity_loss: metric_for(
                best_epoch,
                "rac_forward_velocity_loss.log",
                "rac_forward_velocity_loss",
            )?,
            psnr: metric_for(
                best_epoch,
                "rac_recon_psnr_masked.log",
                "rac_recon_psnr_masked",
            )?,
        },
        reverse: ReverseBreakdown {
            latent_loss: metric_for(
                best_epoch,
                "rac_reverse_latent_loss.log",
                "rac_reverse_latent_loss",
            )?,
            to_init_loss: metric_for(
                best_epoch,
                "rac_reverse_to_init_loss.log",
                "rac_reverse_to_init_loss",
            )?,
        },
        roundtrip: RoundtripBreakdown {
            recon_loss: metric_for(best_epoch, "rac_recon_loss.log", "rac_recon_loss")?,
            psnr: metric_for(best_epoch, "rac_recon_psnr_full.log", "rac_recon_psnr_full")?,
        },
        probe: ProbeBreakdown {
            loss: metric_for(best_epoch, "rac_probe_loss.log", "rac_probe_loss")?,
            accuracy: metric_for(best_epoch, "rac_probe_acc.log", "rac_probe_acc")?,
        },
        artifacts: collect_artifact_paths(run_dir)?,
        notes,
    })
}

fn metric_line(label: &str, metric: &Option<MetricSummary>) -> String {
    match metric {
        Some(metric) => {
            let mean = metric.mean.unwrap_or_default();
            let last = metric.last.unwrap_or(mean);
            format!(
                "- {label}: mean {mean:.6}, last {last:.6} ({})",
                metric.path
            )
        }
        None => format!("- {label}: unavailable"),
    }
}

fn report_markdown(report: &RacBestCheckpointReport) -> String {
    let mut lines = Vec::new();
    lines.push("# RAC Best Checkpoint Report".to_string());
    lines.push(String::new());
    lines.push(format!("- Run: `{}`", report.run_name));
    lines.push(format!("- Run dir: `{}`", report.run_dir));
    lines.push(format!("- Selection rule: {}", report.selection_rule));
    match (&report.best_epoch, &report.primary_metric) {
        (Some(epoch), Some(metric)) => {
            let score = metric.mean.or(metric.last).unwrap_or_default();
            lines.push(format!("- Best epoch: `{epoch}`"));
            lines.push(format!(
                "- Primary metric: `{}` mean `{score:.6}`",
                metric.name
            ));
        }
        _ => lines.push("- Best epoch: unavailable".to_string()),
    }
    if let Some(checkpoint) = &report.checkpoint {
        lines.push(format!(
            "- Checkpoint: `{}`",
            checkpoint
                .path
                .as_deref()
                .unwrap_or("missing checkpoint file")
        ));
    }
    lines.push(String::new());
    lines.push("## Forward Decode".to_string());
    lines.push(metric_line("recon loss", &report.forward.recon_loss));
    lines.push(metric_line("path loss", &report.forward.path_loss));
    lines.push(metric_line("velocity loss", &report.forward.velocity_loss));
    lines.push(metric_line("PSNR", &report.forward.psnr));
    lines.push(String::new());
    lines.push("## Reverse Encode".to_string());
    lines.push(metric_line("latent loss", &report.reverse.latent_loss));
    lines.push(metric_line("to-init loss", &report.reverse.to_init_loss));
    lines.push(String::new());
    lines.push("## Roundtrip".to_string());
    lines.push(metric_line("roundtrip loss", &report.roundtrip.recon_loss));
    lines.push(metric_line("roundtrip PSNR", &report.roundtrip.psnr));
    lines.push(String::new());
    lines.push("## Probe".to_string());
    lines.push(metric_line("probe loss", &report.probe.loss));
    lines.push(metric_line("probe accuracy", &report.probe.accuracy));
    lines.push(String::new());
    lines.push("## Artifacts".to_string());
    if report.artifacts.is_empty() {
        lines.push("- none".to_string());
    } else {
        for artifact in &report.artifacts {
            lines.push(format!("- `{artifact}`"));
        }
    }
    lines.push(String::new());
    lines.push("## Notes".to_string());
    for note in &report.notes {
        lines.push(format!("- {note}"));
    }
    lines.push(String::new());
    lines.join("\n")
}

pub(crate) fn write_rac_best_checkpoint_report(run_dir: &Path, run_name: &str) -> Result<()> {
    let report = build_report(run_dir, run_name)?;
    let artifacts_dir = run_dir.join("artifacts");
    fs::create_dir_all(&artifacts_dir)
        .with_context(|| format!("create artifacts dir {}", artifacts_dir.display()))?;
    let markdown = report_markdown(&report);
    fs::write(
        artifacts_dir.join("rac_best_checkpoint_report.md"),
        markdown,
    )
    .with_context(|| format!("write markdown report in {}", artifacts_dir.display()))?;
    let json = serde_json::to_string_pretty(&report).context("serialize RAC report json")?;
    fs::write(artifacts_dir.join("rac_best_checkpoint_report.json"), json)
        .with_context(|| format!("write json report in {}", artifacts_dir.display()))?;
    Ok(())
}
