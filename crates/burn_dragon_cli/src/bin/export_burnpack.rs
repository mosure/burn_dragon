use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{Parser, ValueEnum};

use burn_dragon::api::checkpoint::bundle::BurnpackBundleExportOptions;
use burn_dragon::api::checkpoint::burnpack::BurnpackFloatPrecision;
use burn_dragon::api::checkpoint::policy::{BurnpackLoadPolicy, BurnpackPrecisionPreference};
use burn_dragon::api::checkpoint::run::{CheckpointExportReport, resolve_checkpoint_base};
use burn_dragon::api::graph::checkpoint::export_graph_checkpoint_to_burnpack;
use burn_dragon::api::language::checkpoint::export_language_checkpoint_to_burnpack;
use burn_dragon::api::multimodal::checkpoint::export_multimodal_checkpoint_to_burnpack;
use burn_dragon::api::sudoku::checkpoint::export_sudoku_checkpoint_to_burnpack;
use burn_dragon::api::vision::checkpoint::export_vision_encoder_checkpoint_to_burnpack;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModelFamilyArg {
    Graph,
    Language,
    MultimodalVlJepa,
    Sudoku,
    VisionEncoder,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PrecisionArg {
    F16,
    F32,
}

impl From<PrecisionArg> for BurnpackFloatPrecision {
    fn from(value: PrecisionArg) -> Self {
        match value {
            PrecisionArg::F16 => Self::F16,
            PrecisionArg::F32 => Self::F32,
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Export Dragon checkpoints to deployable burnpack bundles"
)]
struct Args {
    /// Model family to export.
    #[arg(long, value_enum, default_value_t = ModelFamilyArg::Language)]
    family: ModelFamilyArg,
    /// Path to the checkpoint directory or file.
    #[arg(long, value_name = "PATH")]
    checkpoint: PathBuf,
    /// Specific checkpoint epoch to use when the path is a directory.
    #[arg(long, value_name = "N")]
    epoch: Option<usize>,
    /// Configuration overlays applied in order before export. When omitted, the exporter will
    /// use the run-local snapshot/config flow supported by the selected model family.
    #[arg(short = 'c', long = "config", value_name = "PATH")]
    config: Vec<PathBuf>,
    /// Optional output base path. Defaults to <run>/deploy/model-<epoch> when possible.
    #[arg(long, value_name = "PATH")]
    output: Option<PathBuf>,
    /// Float precision to materialize in the final burnpack.
    #[arg(long, value_enum, default_value_t = PrecisionArg::F16)]
    precision: PrecisionArg,
    /// Maximum part size in MiB. When omitted, export a monolithic burnpack only.
    #[arg(long, value_name = "MIB")]
    parts_mib: Option<u64>,
    /// Rewrite an existing multipart manifest and parts.
    #[arg(long)]
    overwrite_parts: bool,
    /// Keep the intermediate f32 burnpack when exporting an f16 deployment bundle.
    #[arg(long)]
    keep_intermediate_f32: bool,
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
    let output_base = args
        .output
        .clone()
        .unwrap_or_else(|| default_output_base(checkpoint_base.as_path(), epoch));

    let load_policy = BurnpackLoadPolicy::default().with_precision(match args.precision {
        PrecisionArg::F16 => BurnpackPrecisionPreference::PreferF16,
        PrecisionArg::F32 => BurnpackPrecisionPreference::PreferF32,
    });
    let bundle_options = BurnpackBundleExportOptions {
        precision: args.precision.into(),
        load_policy,
        max_part_size_mib: args.parts_mib,
        overwrite_parts: args.overwrite_parts,
        keep_intermediate_f32: args.keep_intermediate_f32,
    };

    match args.family {
        ModelFamilyArg::Graph => {
            let report = export_graph_checkpoint_to_burnpack(
                &args.checkpoint,
                Some(epoch),
                &args.config,
                &output_base,
                &bundle_options,
            )?;
            print_common_export_report("graph", &report);
        }
        ModelFamilyArg::Language => {
            let report = export_language_checkpoint_to_burnpack(
                &args.checkpoint,
                Some(epoch),
                &args.config,
                &args.backend_name,
                &output_base,
                &bundle_options,
            )?;
            eprintln!(
                "Exported language epoch {} from {} (vocab={}) -> {}",
                report.epoch,
                report.checkpoint_base.display(),
                report.vocab_size,
                report.bundle.burnpack_path.display()
            );
            print_bundle_artifacts(&report.bundle);
        }
        ModelFamilyArg::MultimodalVlJepa => {
            let report = export_multimodal_checkpoint_to_burnpack(
                &args.checkpoint,
                Some(epoch),
                &args.config,
                &output_base,
                &bundle_options,
            )?;
            print_common_export_report("multimodal-vl-jepa", &report);
        }
        ModelFamilyArg::Sudoku => {
            let report = export_sudoku_checkpoint_to_burnpack(
                &args.checkpoint,
                Some(epoch),
                &args.config,
                &output_base,
                &bundle_options,
            )?;
            print_common_export_report("sudoku", &report);
        }
        ModelFamilyArg::VisionEncoder => {
            let report = export_vision_encoder_checkpoint_to_burnpack(
                &args.checkpoint,
                Some(epoch),
                &args.config,
                &output_base,
                &bundle_options,
            )?;
            print_common_export_report("vision-encoder", &report);
        }
    }

    Ok(())
}

fn print_common_export_report(label: &str, report: &CheckpointExportReport) {
    eprintln!(
        "Exported {label} epoch {} from {} -> {}",
        report.epoch,
        report.checkpoint_base.display(),
        report.bundle.burnpack_path.display()
    );
    print_bundle_artifacts(&report.bundle);
}

fn print_bundle_artifacts(
    bundle: &burn_dragon::api::checkpoint::bundle::BurnpackBundleExportReport,
) {
    if let Some(parts) = &bundle.parts {
        eprintln!(
            "Multipart burnpack: {} parts via {}",
            parts.part_paths.len(),
            parts.manifest_path.display()
        );
    }
    if let Some(path) = &bundle.intermediate_f32_path {
        eprintln!("Kept intermediate f32 burnpack at {}", path.display());
    }
}

fn default_output_base(checkpoint_base: &Path, epoch: usize) -> PathBuf {
    let checkpoint_dir = checkpoint_base
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let run_dir = checkpoint_dir
        .file_name()
        .is_some_and(|name| name == "checkpoint")
        .then(|| checkpoint_dir.parent().map(Path::to_path_buf))
        .flatten()
        .unwrap_or(checkpoint_dir);
    run_dir.join("deploy").join(format!("model-{epoch}"))
}
