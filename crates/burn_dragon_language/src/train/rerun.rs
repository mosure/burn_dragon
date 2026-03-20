use crate::train::prelude::*;
use burn_train::LearningComponentsTypes;
use burn_train::logger::{FileMetricLogger, MetricLogger};
use burn_train::metric::{
    MetricDefinition, MetricEntry, MetricId, NumericEntry,
    store::{EpochSummary, MetricsUpdate, Split},
};
use rerun::{MemoryLimit, RecordingStream, RecordingStreamBuilder, ServerOptions};
use std::collections::HashMap;
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const RERUN_APP_ID: &str = "burn_dragon_language_train";
const RERUN_SERVER_MEMORY_LIMIT_BYTES: u64 = 64 * 1024 * 1024;

static ACTIVE_SESSION: std::sync::OnceLock<Mutex<Option<TrainingRerunSession>>> =
    std::sync::OnceLock::new();

#[derive(Debug, Clone)]
pub struct TrainingRerunConfig {
    pub run_name: String,
    pub bind_ip: String,
    pub port: u16,
    pub telemetry_interval: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingRerunServerInfo {
    pub server_url: String,
    pub viewer_url: String,
}

struct TrainingRerunSession {
    recording: RecordingStream,
    shutdown: Arc<AtomicBool>,
    telemetry_thread: Option<JoinHandle<()>>,
    info: TrainingRerunServerInfo,
}

#[derive(Clone)]
struct RerunMetricLogger {
    recording: RecordingStream,
    metric_names: HashMap<MetricId, String>,
    split_samples: HashMap<String, i64>,
}

pub fn initialize_training_rerun(config: &TrainingRerunConfig) -> Result<TrainingRerunServerInfo> {
    shutdown_training_rerun();

    let server_url = rerun_server_url(&config.bind_ip, config.port);
    let viewer_url = rerun_viewer_url(&server_url);
    let recording = RecordingStreamBuilder::new(RERUN_APP_ID)
        .recording_name(config.run_name.clone())
        .serve_grpc_opts(
            &config.bind_ip,
            config.port,
            ServerOptions {
                memory_limit: MemoryLimit::from_bytes(RERUN_SERVER_MEMORY_LIMIT_BYTES),
                ..ServerOptions::default()
            },
        )
        .map_err(|err| anyhow!("failed to start rerun gRPC server: {err}"))?;

    let shutdown = Arc::new(AtomicBool::new(false));
    let telemetry_thread = spawn_gpu_telemetry_thread(
        recording.clone(),
        Arc::clone(&shutdown),
        config.telemetry_interval,
    );
    let info = TrainingRerunServerInfo {
        server_url,
        viewer_url,
    };

    active_session()
        .lock()
        .expect("rerun session lock")
        .replace(TrainingRerunSession {
            recording,
            shutdown,
            telemetry_thread,
            info: info.clone(),
        });

    Ok(info)
}

pub fn shutdown_training_rerun() {
    let Some(session) = active_session().lock().expect("rerun session lock").take() else {
        return;
    };

    session.shutdown.store(true, Ordering::Relaxed);
    if let Some(thread) = session.telemetry_thread {
        let _ = thread.join();
    }
    let _ = session.recording.flush_blocking();
}

pub(crate) fn attach_metric_loggers<LC>(
    builder: SupervisedTraining<LC>,
    run_dir: &Path,
) -> SupervisedTraining<LC>
where
    LC: LearningComponentsTypes,
{
    let Some(session) = active_recording() else {
        return builder;
    };

    builder
        .with_metric_logger(FileMetricLogger::new(run_dir))
        .with_metric_logger(RerunMetricLogger::new(session))
}

pub fn rerun_server_url(bind_ip: &str, port: u16) -> String {
    format!("rerun+http://{bind_ip}:{port}/proxy")
}

pub fn rerun_viewer_url(server_url: &str) -> String {
    format!(
        "https://rerun.io/viewer?url={}",
        urlencoding::encode(server_url)
    )
}

impl RerunMetricLogger {
    fn new(recording: RecordingStream) -> Self {
        Self {
            recording,
            metric_names: HashMap::new(),
            split_samples: HashMap::new(),
        }
    }

    fn next_sample(&mut self, split: &Split) -> i64 {
        let label = split_label(split).to_string();
        let counter = self.split_samples.entry(label).or_insert(0);
        *counter += 1;
        *counter
    }

    fn metric_name(&self, entry: &MetricEntry) -> Option<&str> {
        self.metric_names
            .get(&entry.metric_id)
            .map(std::string::String::as_str)
    }

    fn log_numeric_value(&self, path: &str, value: &NumericEntry) {
        let _ = self
            .recording
            .log(path, &rerun::Scalars::single(value.current()));
    }
}

impl MetricLogger for RerunMetricLogger {
    fn log(&mut self, update: MetricsUpdate, epoch: usize, split: &Split) {
        let sample = self.next_sample(split);
        let split_label = split_label(split);
        self.recording.set_time_sequence("epoch", epoch as i64);
        self.recording.set_time_sequence("sample", sample);

        for numeric in update.entries_numeric {
            let Some(metric_name) = self.metric_name(&numeric.entry) else {
                continue;
            };
            let metric_name = sanitize_metric_name(metric_name);
            let value_path = format!("metrics/{split_label}/{metric_name}/value");
            let running_path = format!("metrics/{split_label}/{metric_name}/running");
            self.log_numeric_value(&value_path, &numeric.numeric_entry);
            self.log_numeric_value(&running_path, &numeric.running_entry);
        }

        for entry in update.entries {
            let Some(metric_name) = self.metric_name(&entry) else {
                continue;
            };
            let text_path = format!(
                "metrics/{split_label}/{}/text",
                sanitize_metric_name(metric_name)
            );
            let _ = self.recording.log(
                text_path.as_str(),
                &rerun::TextLog::new(entry.serialized_entry.formatted.clone()),
            );
        }
    }

    fn read_numeric(
        &mut self,
        _name: &str,
        _epoch: usize,
        _split: &Split,
    ) -> std::result::Result<Vec<NumericEntry>, String> {
        Ok(Vec::new())
    }

    fn log_metric_definition(&mut self, definition: MetricDefinition) {
        self.metric_names
            .insert(definition.metric_id, definition.name.clone());
    }

    fn log_epoch_summary(&mut self, summary: EpochSummary) {
        let split_label = split_label(&summary.split);
        self.recording
            .set_time_sequence("epoch", summary.epoch_number as i64);
        let _ = self.recording.log(
            format!("metrics/{split_label}/epoch_summary"),
            &rerun::TextLog::new(format!(
                "completed {} epoch {}",
                split_label, summary.epoch_number
            )),
        );
    }
}

fn active_session() -> &'static Mutex<Option<TrainingRerunSession>> {
    ACTIVE_SESSION.get_or_init(|| Mutex::new(None))
}

fn active_recording() -> Option<RecordingStream> {
    active_session()
        .lock()
        .expect("rerun session lock")
        .as_ref()
        .map(|session| session.recording.clone())
}

fn split_label(split: &Split) -> &str {
    match split {
        Split::Train => "train",
        Split::Valid => "valid",
        Split::Test(_) => "test",
    }
}

fn sanitize_metric_name(metric_name: &str) -> String {
    metric_name
        .trim()
        .to_lowercase()
        .replace([' ', '/', ':'], "_")
}

fn spawn_gpu_telemetry_thread(
    recording: RecordingStream,
    shutdown: Arc<AtomicBool>,
    interval: Duration,
) -> Option<JoinHandle<()>> {
    if interval.is_zero() {
        return None;
    }

    Some(thread::spawn(move || {
        let mut sample = 0_i64;
        while !shutdown.load(Ordering::Relaxed) {
            if let Some(stats) = sample_nvidia_smi() {
                sample += 1;
                recording.set_time_sequence("gpu_sample", sample);
                let _ = recording.log(
                    "system/gpu/utilization_pct",
                    &rerun::Scalars::single(stats.utilization_pct),
                );
                let _ = recording.log(
                    "system/gpu/power_watts",
                    &rerun::Scalars::single(stats.power_watts),
                );
                let _ = recording.log(
                    "system/gpu/memory_used_mib",
                    &rerun::Scalars::single(stats.memory_used_mib),
                );
            }
            thread::sleep(interval);
        }
    }))
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct GpuStats {
    utilization_pct: f64,
    power_watts: f64,
    memory_used_mib: f64,
}

fn sample_nvidia_smi() -> Option<GpuStats> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,power.draw,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let line = String::from_utf8(output.stdout).ok()?;
    let mut fields = line.lines().next()?.split(',').map(str::trim);
    let utilization_pct = fields.next()?.parse().ok()?;
    let power_watts = fields.next()?.parse().ok()?;
    let memory_used_mib = fields.next()?.parse().ok()?;
    Some(GpuStats {
        utilization_pct,
        power_watts,
        memory_used_mib,
    })
}

#[cfg(test)]
mod tests {
    use super::{rerun_server_url, rerun_viewer_url, sanitize_metric_name};

    #[test]
    fn rerun_viewer_url_wraps_server_proxy_url() {
        let server_url = rerun_server_url("127.0.0.1", 9876);
        assert_eq!(server_url, "rerun+http://127.0.0.1:9876/proxy");
        assert_eq!(
            rerun_viewer_url(&server_url),
            "https://rerun.io/viewer?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9876%2Fproxy"
        );
    }

    #[test]
    fn sanitize_metric_name_normalizes_common_separators() {
        assert_eq!(sanitize_metric_name("Learning Rate"), "learning_rate");
        assert_eq!(sanitize_metric_name("Loss/Valid"), "loss_valid");
        assert_eq!(sanitize_metric_name("device:cuda"), "device_cuda");
    }
}
