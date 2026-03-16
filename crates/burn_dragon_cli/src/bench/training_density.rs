use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use burn_dragon::vision::{VisionArtifactHeader, push_vision_artifact_markdown_prelude};
use serde::Serialize;
use serde_json::Value;

use crate::bench::artifact::write_text_artifact;

#[derive(Clone, Debug)]
pub struct TrainingDensityBenchConfig {
    pub cases: Vec<TrainingDensityBenchCaseSpec>,
    pub backend: String,
    pub sample_interval_ms: u64,
    pub warmup_gpu_samples: usize,
    pub warmup_iteration_deltas: usize,
    pub output_dir: PathBuf,
    pub markdown_path: Option<PathBuf>,
    pub json_path: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct TrainingDensityBenchCaseSpec {
    pub label: String,
    pub kind: TrainingDensityBenchCaseKind,
    pub configs: Vec<PathBuf>,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingDensityBenchCaseKind {
    Language,
    Vision,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainingDensityBenchGpuSample {
    pub sample_ts_s: f64,
    pub power_w: f64,
    pub util_pct: f64,
    pub memory_mb: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainingDensityBenchMetricSummary {
    pub min_value: f64,
    pub min_epoch: usize,
    pub max_value: f64,
    pub max_epoch: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TrainingDensityBenchStageProfileSummary {
    pub total_ms_per_step: f64,
    pub dataloader_cpu_ms_per_step: f64,
    pub dataloader_image_load_ms_per_step: f64,
    pub dataloader_image_transform_ms_per_step: f64,
    pub dataloader_teacher_load_ms_per_step: f64,
    pub dataloader_tensor_copy_ms_per_step: f64,
    pub forward_ms_per_step: f64,
    pub loss_backward_ms_per_step: f64,
    pub host_to_device_bytes_per_step: f64,
    pub host_sync_points_per_step: f64,
    pub train_steps_profiled: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainingDensityBenchCaseReport {
    pub label: String,
    pub kind: TrainingDensityBenchCaseKind,
    pub config: Vec<PathBuf>,
    pub run_dir: PathBuf,
    pub stdout_log: PathBuf,
    pub stderr_log: PathBuf,
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub effective_batch_size: usize,
    pub block_or_patch_tokens: usize,
    pub mean_step_ms: f64,
    pub std_step_ms: f64,
    pub step_cv: f64,
    pub measured_step_deltas: usize,
    pub effective_samples_per_sec: f64,
    pub work_items_per_sec: f64,
    pub gpu_power_w_mean: f64,
    pub gpu_power_w_p90: f64,
    pub gpu_util_pct_mean: f64,
    pub gpu_memory_gib_max: f64,
    pub gpu_samples_kept: usize,
    pub metrics: BTreeMap<String, TrainingDensityBenchMetricSummary>,
    pub locked_metric: Option<String>,
    pub locked_metric_min: Option<f64>,
    pub stage_profile: Option<TrainingDensityBenchStageProfileSummary>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainingDensityBenchGapSummary {
    pub language_label: String,
    pub vision_label: String,
    pub power_ratio_vs_language: f64,
    pub util_ratio_vs_language: f64,
    pub memory_ratio_vs_language: f64,
    pub sample_throughput_ratio_vs_language: f64,
    pub step_latency_ratio_vs_language: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainingDensityBenchReport {
    pub artifact: VisionArtifactHeader,
    pub benchmark: &'static str,
    pub backend: String,
    pub sample_interval_ms: u64,
    pub warmup_gpu_samples: usize,
    pub warmup_iteration_deltas: usize,
    pub cases: Vec<TrainingDensityBenchCaseReport>,
    pub gaps_vs_language: Vec<TrainingDensityBenchGapSummary>,
}

pub fn parse_training_density_case_spec(
    value: &str,
) -> Result<TrainingDensityBenchCaseSpec, String> {
    let (label, rest) = value.split_once('=').ok_or_else(|| {
        "case must look like label=language:path1,path2 or label=vision:path1,path2".to_string()
    })?;
    let (kind_raw, paths_raw) = rest
        .split_once(':')
        .ok_or_else(|| "case must include kind:path1,path2".to_string())?;
    let kind = match kind_raw {
        "language" => TrainingDensityBenchCaseKind::Language,
        "vision" => TrainingDensityBenchCaseKind::Vision,
        other => return Err(format!("unsupported case kind {other:?}")),
    };
    let configs = paths_raw
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if configs.is_empty() {
        return Err("case must include at least one config path".to_string());
    }
    Ok(TrainingDensityBenchCaseSpec {
        label: label.to_string(),
        kind,
        configs,
    })
}

impl TrainingDensityBenchReport {
    pub fn to_markdown(&self) -> String {
        let mut markdown = String::new();
        push_vision_artifact_markdown_prelude(
            &mut markdown,
            "Training Density Bench",
            &self.artifact,
        );
        writeln!(
            &mut markdown,
            "- backend: `{}`\n- sample_interval_ms: `{}`\n",
            self.backend, self.sample_interval_ms
        )
        .unwrap();
        writeln!(
            &mut markdown,
            "| case | kind | eff_batch | work_items | step_ms | step_cv | samples/s | work_items/s | power_w_mean | power_w_p90 | util_mean | mem_gib_max | locked_metric | locked_min |"
        )
        .unwrap();
        writeln!(
            &mut markdown,
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"
        )
        .unwrap();
        for case in &self.cases {
            writeln!(
                &mut markdown,
                "| {} | {:?} | {} | {} | {:.3} | {:.3} | {:.2} | {:.0} | {:.1} | {:.1} | {:.1} | {:.2} | {} | {} |",
                case.label,
                case.kind,
                case.effective_batch_size,
                case.block_or_patch_tokens,
                case.mean_step_ms,
                case.step_cv,
                case.effective_samples_per_sec,
                case.work_items_per_sec,
                case.gpu_power_w_mean,
                case.gpu_power_w_p90,
                case.gpu_util_pct_mean,
                case.gpu_memory_gib_max,
                case.locked_metric.as_deref().unwrap_or("-"),
                case.locked_metric_min
                    .map(|value| format!("{value:.4}"))
                    .unwrap_or_else(|| "-".to_string()),
            )
            .unwrap();
        }
        if !self.gaps_vs_language.is_empty() {
            writeln!(&mut markdown, "\n## Gaps Vs Language\n").unwrap();
            writeln!(
                &mut markdown,
                "| vision_case | power_ratio | util_ratio | memory_ratio | sample_throughput_ratio | step_latency_ratio |"
            )
            .unwrap();
            writeln!(&mut markdown, "|---|---:|---:|---:|---:|---:|").unwrap();
            for gap in &self.gaps_vs_language {
                writeln!(
                    &mut markdown,
                    "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
                    gap.vision_label,
                    gap.power_ratio_vs_language,
                    gap.util_ratio_vs_language,
                    gap.memory_ratio_vs_language,
                    gap.sample_throughput_ratio_vs_language,
                    gap.step_latency_ratio_vs_language,
                )
                .unwrap();
            }
        }
        writeln!(&mut markdown, "\n## Stage Profile\n").unwrap();
        for case in &self.cases {
            writeln!(&mut markdown, "### {}", case.label).unwrap();
            if let Some(stage) = &case.stage_profile {
                writeln!(
                    &mut markdown,
                    "- total_ms_per_step: `{:.3}`\n- dataloader_cpu_ms_per_step: `{:.3}`\n- dataloader_image_load_ms_per_step: `{:.3}`\n- dataloader_image_transform_ms_per_step: `{:.3}`\n- dataloader_teacher_load_ms_per_step: `{:.3}`\n- dataloader_tensor_copy_ms_per_step: `{:.3}`\n- forward_ms_per_step: `{:.3}`\n- loss_backward_ms_per_step: `{:.3}`\n- host_to_device_bytes_per_step: `{:.0}`\n- host_sync_points_per_step: `{:.3}`",
                    stage.total_ms_per_step,
                    stage.dataloader_cpu_ms_per_step,
                    stage.dataloader_image_load_ms_per_step,
                    stage.dataloader_image_transform_ms_per_step,
                    stage.dataloader_teacher_load_ms_per_step,
                    stage.dataloader_tensor_copy_ms_per_step,
                    stage.forward_ms_per_step,
                    stage.loss_backward_ms_per_step,
                    stage.host_to_device_bytes_per_step,
                    stage.host_sync_points_per_step,
                )
                .unwrap();
            } else {
                writeln!(&mut markdown, "- no stage profile found").unwrap();
            }
        }
        markdown
    }
}

pub fn run_training_density_bench(
    config: &TrainingDensityBenchConfig,
) -> Result<TrainingDensityBenchReport> {
    fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("create {}", config.output_dir.display()))?;
    let train_bin = train_binary_path()?;
    let mut cases = Vec::with_capacity(config.cases.len());
    for case in &config.cases {
        cases.push(run_case(case, config, &train_bin)?);
    }
    let gaps_vs_language = summarize_language_gaps(&cases);
    Ok(TrainingDensityBenchReport {
        artifact: VisionArtifactHeader::new("training_density_bench"),
        benchmark: "burn_dragon matched language-vs-vision training density benchmark",
        backend: config.backend.clone(),
        sample_interval_ms: config.sample_interval_ms,
        warmup_gpu_samples: config.warmup_gpu_samples,
        warmup_iteration_deltas: config.warmup_iteration_deltas,
        cases,
        gaps_vs_language,
    })
}

pub fn write_training_density_bench_artifacts(
    config: &TrainingDensityBenchConfig,
    report: &TrainingDensityBenchReport,
) -> Result<()> {
    let markdown = report.to_markdown();
    let json = serde_json::to_string_pretty(report).context("serialize density report")?;
    write_text_artifact(
        config
            .markdown_path
            .as_deref()
            .unwrap_or(&config.output_dir.join("training_density_bench.md")),
        &markdown,
        "markdown artifact",
    )?;
    write_text_artifact(
        config
            .json_path
            .as_deref()
            .unwrap_or(&config.output_dir.join("training_density_bench.json")),
        &json,
        "json artifact",
    )?;
    Ok(())
}

fn train_binary_path() -> Result<PathBuf> {
    let current = std::env::current_exe().context("resolve current executable path")?;
    let sibling = current.with_file_name(if cfg!(windows) { "train.exe" } else { "train" });
    if sibling.exists() {
        Ok(sibling)
    } else {
        bail!(
            "expected sibling train binary at {}; build it first with `cargo build -p burn_dragon_cli --features benchmark,train --bin train --bin training_density_bench`",
            sibling.display()
        )
    }
}

fn run_case(
    case: &TrainingDensityBenchCaseSpec,
    config: &TrainingDensityBenchConfig,
    train_bin: &Path,
) -> Result<TrainingDensityBenchCaseReport> {
    let run_root = match case.kind {
        TrainingDensityBenchCaseKind::Language => PathBuf::from("runs"),
        TrainingDensityBenchCaseKind::Vision => PathBuf::from("runs").join("vision"),
    };
    let latest_before = read_latest_run(&run_root);
    let stdout_log = config.output_dir.join(format!("{}.stdout.log", case.label));
    let stderr_log = config.output_dir.join(format!("{}.stderr.log", case.label));
    let stdout =
        File::create(&stdout_log).with_context(|| format!("create {}", stdout_log.display()))?;
    let stderr =
        File::create(&stderr_log).with_context(|| format!("create {}", stderr_log.display()))?;

    let mut command = Command::new(train_bin);
    command
        .env("RUST_LOG", "info")
        .env("BDH_STAGE_PROFILE", "1")
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    match case.kind {
        TrainingDensityBenchCaseKind::Language => {
            command.arg("language");
        }
        TrainingDensityBenchCaseKind::Vision => {
            command.arg("vision");
        }
    }
    command.arg("--backend").arg(&config.backend);
    for config_path in &case.configs {
        command.arg("--config").arg(config_path);
    }

    let mut child = command
        .spawn()
        .with_context(|| format!("spawn train binary for {}", case.label))?;
    let mut gpu_samples = Vec::new();
    loop {
        match child.try_wait()? {
            Some(status) => {
                if let Some(sample) = query_nvidia_smi() {
                    gpu_samples.push(sample);
                }
                if !status.success() {
                    let stderr_tail = tail_file(&stderr_log, 80)?;
                    bail!(
                        "train subprocess exited with status {status}; stderr tail:\n{stderr_tail}"
                    );
                }
                break;
            }
            None => {
                if let Some(sample) = query_nvidia_smi() {
                    gpu_samples.push(sample);
                }
                thread::sleep(Duration::from_millis(config.sample_interval_ms.max(50)));
            }
        }
    }

    let run_name = read_latest_run(&run_root).with_context(|| {
        format!(
            "resolve latest run after {} (previous latest {:?})",
            case.label, latest_before
        )
    })?;
    if latest_before.as_deref() == Some(run_name.as_str()) {
        bail!(
            "latest run in {} did not advance after case {}",
            run_root.display(),
            case.label
        );
    }
    let run_dir = run_root.join(&run_name);
    let experiment_log = run_dir.join("experiment.log");
    let log_text = fs::read_to_string(&experiment_log)
        .with_context(|| format!("read {}", experiment_log.display()))?;
    let stdout_log_text =
        fs::read_to_string(&stdout_log).with_context(|| format!("read {}", stdout_log.display()))?;
    let metrics = parse_metric_table(&stdout_log_text);
    let kept_intervals = trimmed_steady_state_intervals(&log_text, config.warmup_iteration_deltas);
    let kept_step_deltas = kept_intervals
        .iter()
        .map(|interval| interval.end_s - interval.start_s)
        .collect::<Vec<_>>();
    if kept_step_deltas.is_empty() {
        bail!(
            "no training iteration deltas found in {} for case {}",
            experiment_log.display(),
            case.label
        );
    }
    let mean_step_s = mean_f64(&kept_step_deltas);
    let std_step_s = stddev_f64(&kept_step_deltas, mean_step_s);
    let stage_profile = parse_stage_profile(&log_text, mean_step_s);
    let kept_gpu_samples = gpu_samples
        .iter()
        .skip(config.warmup_gpu_samples.min(gpu_samples.len()))
        .cloned()
        .collect::<Vec<_>>();
    let steady_state_gpu_samples = if kept_intervals.is_empty() {
        None
    } else {
        Some(
            kept_gpu_samples
                .iter()
                .filter(|sample| {
                    kept_intervals.iter().any(|interval| {
                        sample.sample_ts_s >= interval.start_s && sample.sample_ts_s <= interval.end_s
                    })
                })
                .cloned()
                .collect::<Vec<_>>(),
        )
        .filter(|samples| !samples.is_empty())
    };
    let used_gpu_samples = if kept_gpu_samples.is_empty() {
        gpu_samples
    } else if let Some(steady_state_samples) = steady_state_gpu_samples {
        steady_state_samples
    } else {
        kept_gpu_samples
    };
    let gpu_power_w_mean = mean_f64(
        &used_gpu_samples
            .iter()
            .map(|sample| sample.power_w)
            .collect::<Vec<_>>(),
    );
    let gpu_power_w_p90 = percentile_f64(
        &used_gpu_samples
            .iter()
            .map(|sample| sample.power_w)
            .collect::<Vec<_>>(),
        0.90,
    );
    let gpu_util_pct_mean = mean_f64(
        &used_gpu_samples
            .iter()
            .map(|sample| sample.util_pct)
            .collect::<Vec<_>>(),
    );
    let gpu_memory_gib_max = used_gpu_samples
        .iter()
        .map(|sample| sample.memory_mb)
        .fold(0.0f64, f64::max)
        / 1024.0;

    let resolved_training = load_resolved_training_config(case.kind, &run_dir)?;
    let batch_size = get_usize(&resolved_training, &["training", "batch_size"]).unwrap_or(0);
    let gradient_accumulation_steps =
        get_usize(&resolved_training, &["training", "gradient_accumulation_steps"])
            .unwrap_or(1)
            .max(1);
    let effective_batch_size = batch_size.saturating_mul(gradient_accumulation_steps);
    let block_or_patch_tokens = match case.kind {
        TrainingDensityBenchCaseKind::Language => {
            get_usize(&resolved_training, &["training", "block_size"]).unwrap_or(0)
        }
        TrainingDensityBenchCaseKind::Vision => {
            let image_size = get_usize(&resolved_training, &["vision", "image_size"]).unwrap_or(0);
            let patch_size = get_usize(&resolved_training, &["vision", "patch_size"])
                .unwrap_or(1)
                .max(1);
            let grid = image_size.div_ceil(patch_size).max(1);
            grid.saturating_mul(grid)
        }
    };
    let effective_samples_per_sec = effective_batch_size as f64 / mean_step_s.max(f64::EPSILON);
    let work_items_per_sec = effective_samples_per_sec * block_or_patch_tokens as f64;
    let locked_metric = match case.kind {
        TrainingDensityBenchCaseKind::Language => Some("Loss".to_string()),
        TrainingDensityBenchCaseKind::Vision => {
            let image_size = get_usize(&resolved_training, &["vision", "image_size"]).unwrap_or(0);
            Some(if image_size >= 280 {
                "distill_total_to_s2".to_string()
            } else {
                "distill_total_to_s8".to_string()
            })
        }
    };
    let locked_metric_min = locked_metric
        .as_ref()
        .and_then(|name| metrics.get(&format!("valid::{name}")))
        .map(|metric| metric.min_value);

    Ok(TrainingDensityBenchCaseReport {
        label: case.label.clone(),
        kind: case.kind,
        config: case.configs.clone(),
        run_dir,
        stdout_log,
        stderr_log,
        batch_size,
        gradient_accumulation_steps,
        effective_batch_size,
        block_or_patch_tokens,
        mean_step_ms: mean_step_s * 1000.0,
        std_step_ms: std_step_s * 1000.0,
        step_cv: if mean_step_s > 0.0 { std_step_s / mean_step_s } else { 0.0 },
        measured_step_deltas: kept_step_deltas.len(),
        effective_samples_per_sec,
        work_items_per_sec,
        gpu_power_w_mean,
        gpu_power_w_p90,
        gpu_util_pct_mean,
        gpu_memory_gib_max,
        gpu_samples_kept: used_gpu_samples.len(),
        metrics,
        locked_metric,
        locked_metric_min,
        stage_profile,
    })
}

fn read_latest_run(run_root: &Path) -> Option<String> {
    fs::read_to_string(run_root.join("latest"))
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn query_nvidia_smi() -> Option<TrainingDensityBenchGpuSample> {
    let sample_ts_s = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_secs_f64();
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=power.draw,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .trim()
        .to_string();
    let mut parts = line.split(',').map(str::trim);
    Some(TrainingDensityBenchGpuSample {
        sample_ts_s,
        power_w: parts.next()?.parse::<f64>().ok()?,
        util_pct: parts.next()?.parse::<f64>().ok()?,
        memory_mb: parts.next()?.parse::<f64>().ok()?,
    })
}

fn tail_file(path: &Path, lines: usize) -> Result<String> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let tail = text
        .lines()
        .rev()
        .take(lines.max(1))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n");
    Ok(tail)
}

fn load_resolved_training_config(kind: TrainingDensityBenchCaseKind, run_dir: &Path) -> Result<Value> {
    let path = match kind {
        TrainingDensityBenchCaseKind::Language => run_dir.join("training_config.json"),
        TrainingDensityBenchCaseKind::Vision => run_dir.join("vision_training_config.json"),
    };
    let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

fn get_usize(value: &Value, path: &[&str]) -> Option<usize> {
    let mut cursor = value;
    for key in path {
        cursor = cursor.get(*key)?;
    }
    cursor.as_u64().map(|value| value as usize)
}

fn parse_metric_table(text: &str) -> BTreeMap<String, TrainingDensityBenchMetricSummary> {
    let mut metrics = BTreeMap::new();
    for line in text.lines() {
        if !line.starts_with("| ") {
            continue;
        }
        let cells = line
            .split('|')
            .map(str::trim)
            .filter(|cell| !cell.is_empty())
            .collect::<Vec<_>>();
        if cells.len() != 6 {
            continue;
        }
        let split = match cells[0] {
            "Train" => "train",
            "Valid" => "valid",
            _ => continue,
        };
        let min_value = match cells[2].parse::<f64>() {
            Ok(value) => value,
            Err(_) => continue,
        };
        let min_epoch = match cells[3].parse::<usize>() {
            Ok(value) => value,
            Err(_) => continue,
        };
        let max_value = match cells[4].parse::<f64>() {
            Ok(value) => value,
            Err(_) => continue,
        };
        let max_epoch = match cells[5].parse::<usize>() {
            Ok(value) => value,
            Err(_) => continue,
        };
        metrics.insert(
            format!("{split}::{}", cells[1]),
            TrainingDensityBenchMetricSummary {
                min_value,
                min_epoch,
                max_value,
                max_epoch,
            },
        );
    }
    metrics
}

fn parse_iteration_timestamps_seconds(text: &str) -> Vec<f64> {
    let mut timestamps = Vec::new();
    for line in text.lines() {
        if !line.contains("Iteration ") {
            continue;
        }
        let Some(ts_raw) = line.split_whitespace().next() else {
            continue;
        };
        if let Some(ts) = parse_timestamp_seconds(ts_raw) {
            timestamps.push(ts);
        }
    }
    timestamps
}

#[derive(Clone, Copy, Debug)]
struct IterationInterval {
    start_s: f64,
    end_s: f64,
}

fn parse_iteration_intervals(text: &str) -> Vec<IterationInterval> {
    let timestamps = parse_iteration_timestamps_seconds(text);
    timestamps
        .windows(2)
        .filter_map(|pair| {
            let start_s = pair[0];
            let end_s = pair[1];
            let delta = end_s - start_s;
            (delta.is_finite() && delta > 0.0).then_some(IterationInterval { start_s, end_s })
        })
        .collect()
}

fn trimmed_steady_state_intervals(
    text: &str,
    warmup_iteration_deltas: usize,
) -> Vec<IterationInterval> {
    let warmed = parse_iteration_intervals(text)
        .into_iter()
        .skip(warmup_iteration_deltas)
        .collect::<Vec<_>>();
    if warmed.len() < 4 {
        return warmed;
    }
    let deltas = warmed
        .iter()
        .map(|interval| interval.end_s - interval.start_s)
        .collect::<Vec<_>>();
    let cutoff = percentile_f64(&deltas, 0.90);
    let trimmed = warmed
        .iter()
        .copied()
        .filter(|interval| (interval.end_s - interval.start_s) <= cutoff)
        .collect::<Vec<_>>();
    if trimmed.len() >= warmed.len() / 2 {
        trimmed
    } else {
        warmed
    }
}

fn parse_timestamp_seconds(value: &str) -> Option<f64> {
    let (date_raw, time_raw) = value.split_once('T')?;
    let time_raw = time_raw.strip_suffix('Z')?;
    let mut date_parts = date_raw.split('-');
    let year = date_parts.next()?.parse::<i32>().ok()?;
    let month = date_parts.next()?.parse::<u32>().ok()?;
    let day = date_parts.next()?.parse::<u32>().ok()?;
    let mut time_parts = time_raw.split(':');
    let hour = time_parts.next()?.parse::<u32>().ok()?;
    let minute = time_parts.next()?.parse::<u32>().ok()?;
    let second_part = time_parts.next()?;
    let (second_raw, frac_raw) = second_part.split_once('.').unwrap_or((second_part, "0"));
    let second = second_raw.parse::<u32>().ok()?;
    let frac_digits = frac_raw
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    let nanos = if frac_digits.is_empty() {
        0u32
    } else {
        let padded = format!("{:0<9}", frac_digits);
        padded[..9].parse::<u32>().ok()?
    };
    let days = civil_to_days(year, month, day);
    Some(
        (days as f64) * 86_400.0
            + (hour as f64) * 3600.0
            + (minute as f64) * 60.0
            + second as f64
            + (nanos as f64) / 1_000_000_000.0,
    )
}

fn civil_to_days(year: i32, month: u32, day: u32) -> i64 {
    let year = year - ((month <= 2) as i32);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let month = month as i32;
    let day = day as i32;
    let doy = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era * 146097 + doe - 719468) as i64
}

fn parse_stage_profile(
    text: &str,
    mean_step_s: f64,
) -> Option<TrainingDensityBenchStageProfileSummary> {
    let line = text
        .lines()
        .rev()
        .find(|line| line.contains("[stage-profile][training]"))?;
    let mut pairs = BTreeMap::new();
    for token in line.split_whitespace() {
        let Some((key, value)) = token.split_once('=') else {
            continue;
        };
        let cleaned = value.trim_end_matches(',');
        if let Ok(parsed) = cleaned.parse::<f64>() {
            pairs.insert(key.to_string(), parsed);
        }
    }
    let train_steps = pairs.get("train_steps").copied().unwrap_or(0.0).max(1.0);
    Some(TrainingDensityBenchStageProfileSummary {
        total_ms_per_step: mean_step_s * 1000.0,
        dataloader_cpu_ms_per_step: pairs.get("dataloader_cpu_ns").copied().unwrap_or(0.0)
            / train_steps
            / 1_000_000.0,
        dataloader_image_load_ms_per_step: pairs
            .get("dataloader_image_load_ns")
            .copied()
            .unwrap_or(0.0)
            / train_steps
            / 1_000_000.0,
        dataloader_image_transform_ms_per_step: pairs
            .get("dataloader_image_transform_ns")
            .copied()
            .unwrap_or(0.0)
            / train_steps
            / 1_000_000.0,
        dataloader_teacher_load_ms_per_step: pairs
            .get("dataloader_teacher_load_ns")
            .copied()
            .unwrap_or(0.0)
            / train_steps
            / 1_000_000.0,
        dataloader_tensor_copy_ms_per_step: pairs
            .get("dataloader_tensor_copy_ns")
            .copied()
            .unwrap_or(0.0)
            / train_steps
            / 1_000_000.0,
        forward_ms_per_step: pairs.get("forward_ns").copied().unwrap_or(0.0)
            / train_steps
            / 1_000_000.0,
        loss_backward_ms_per_step: pairs.get("loss_backward_ns").copied().unwrap_or(0.0)
            / train_steps
            / 1_000_000.0,
        host_to_device_bytes_per_step: pairs
            .get("dataloader_host_to_device_copy_bytes")
            .copied()
            .unwrap_or(0.0)
            / train_steps,
        host_sync_points_per_step: pairs.get("host_sync_points").copied().unwrap_or(0.0)
            / train_steps,
        train_steps_profiled: train_steps as usize,
    })
}

fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn stddev_f64(values: &[f64], mean: f64) -> f64 {
    if values.len() <= 1 {
        0.0
    } else {
        let variance = values
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .sum::<f64>()
            / values.len() as f64;
        variance.sqrt()
    }
}

fn percentile_f64(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((sorted.len() - 1) as f64 * percentile.clamp(0.0, 1.0)).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn summarize_language_gaps(
    cases: &[TrainingDensityBenchCaseReport],
) -> Vec<TrainingDensityBenchGapSummary> {
    let Some(language) = cases
        .iter()
        .find(|case| case.kind == TrainingDensityBenchCaseKind::Language)
    else {
        return Vec::new();
    };
    cases
        .iter()
        .filter(|case| case.kind == TrainingDensityBenchCaseKind::Vision)
        .map(|vision| TrainingDensityBenchGapSummary {
            language_label: language.label.clone(),
            vision_label: vision.label.clone(),
            power_ratio_vs_language: safe_ratio(vision.gpu_power_w_mean, language.gpu_power_w_mean),
            util_ratio_vs_language: safe_ratio(
                vision.gpu_util_pct_mean,
                language.gpu_util_pct_mean,
            ),
            memory_ratio_vs_language: safe_ratio(
                vision.gpu_memory_gib_max,
                language.gpu_memory_gib_max,
            ),
            sample_throughput_ratio_vs_language: safe_ratio(
                vision.effective_samples_per_sec,
                language.effective_samples_per_sec,
            ),
            step_latency_ratio_vs_language: safe_ratio(vision.mean_step_ms, language.mean_step_ms),
        })
        .collect()
}

fn safe_ratio(lhs: f64, rhs: f64) -> f64 {
    if rhs.abs() <= f64::EPSILON {
        0.0
    } else {
        lhs / rhs
    }
}
