use std::fs;
use std::path::Path;

use anyhow::{Result, anyhow};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

pub struct ExperimentLogGuard(Option<WorkerGuard>);

impl ExperimentLogGuard {
    pub fn is_file_backed(&self) -> bool {
        self.0.is_some()
    }
}

pub fn init_experiment_tracing(log_path: Option<&Path>) -> Result<ExperimentLogGuard> {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let stderr_layer = tracing_subscriber::fmt::layer().with_target(false);

    match log_path {
        Some(log_path) => {
            let parent = log_path.parent().ok_or_else(|| {
                anyhow!("failed to determine log parent for {}", log_path.display())
            })?;
            fs::create_dir_all(parent).map_err(|err| {
                anyhow!("failed to create log directory {}: {err}", parent.display())
            })?;
            let file_name = log_path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| anyhow!("invalid log file name {}", log_path.display()))?;
            let file_appender = tracing_appender::rolling::never(parent, file_name);
            let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
            tracing_subscriber::registry()
                .with(env_filter)
                .with(stderr_layer)
                .with(
                    tracing_subscriber::fmt::layer()
                        .with_ansi(false)
                        .with_target(false)
                        .with_writer(non_blocking),
                )
                .try_init()
                .map_err(|err| anyhow!("failed to initialize tracing subscriber: {err}"))?;
            Ok(ExperimentLogGuard(Some(guard)))
        }
        None => {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(stderr_layer)
                .try_init()
                .map_err(|err| anyhow!("failed to initialize tracing subscriber: {err}"))?;
            Ok(ExperimentLogGuard(None))
        }
    }
}
