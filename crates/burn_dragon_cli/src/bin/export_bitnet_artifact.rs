use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use burn_dragon::api::checkpoint::run::resolve_checkpoint_base;
use burn_dragon::api::language::checkpoint::{
    default_bitnet_artifact_path, export_language_checkpoint_to_bitnet_artifact,
};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Export language BDH checkpoints to standalone BitNet deploy bundles with packed static weights"
)]
struct Args {
    /// Path to the checkpoint directory or file.
    #[arg(long, value_name = "PATH")]
    checkpoint: PathBuf,
    /// Specific checkpoint epoch to use when the path is a directory.
    #[arg(long, value_name = "N")]
    epoch: Option<usize>,
    /// Configuration overlays applied in order before export. When omitted, the exporter will
    /// use the run-local snapshot/config flow.
    #[arg(short = 'c', long = "config", value_name = "PATH")]
    config: Vec<PathBuf>,
    /// Optional output path. Defaults to <run>/deploy/model-<epoch>.bitnet_artifact.bin.gz.
    #[arg(long, value_name = "PATH")]
    output: Option<PathBuf>,
    /// Backend label used only for locating run-local metadata when explicit configs are absent.
    #[arg(long, default_value = "wgpu")]
    backend_name: String,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    let (checkpoint_base, epoch) = resolve_checkpoint_base(&args.checkpoint, args.epoch)?;
    let output_path = args
        .output
        .clone()
        .unwrap_or_else(|| default_bitnet_artifact_path(&checkpoint_base, epoch));
    let report = export_language_checkpoint_to_bitnet_artifact(
        &args.checkpoint,
        Some(epoch),
        &args.config,
        &args.backend_name,
        &output_path,
    )?;

    eprintln!(
        "Exported language BitNet artifact epoch {} from {} -> {}",
        report.epoch,
        report.checkpoint_base.display(),
        report.artifact_path.display()
    );
    eprintln!(
        "Packed modules: decoder_x={} decoder_y={} encoder={} kernel_abi={:?}",
        report.bundle.static_weights.decoder_x.is_some(),
        report.bundle.static_weights.decoder_y.is_some(),
        report.bundle.static_weights.encoder.is_some(),
        report.bundle.kernel_abi_version
    );

    Ok(())
}
