use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, HalfPrecisionSettings, Recorder};
use burn::tensor::backend::Backend as BackendTrait;
use burn_ndarray::NdArray;
use clap::Parser;

use burn_dragon::api::checkpoint::run::resolve_checkpoint_base;
use burn_dragon::core::BDH;

type QuantBackend = NdArray<f32>;

#[derive(Parser, Debug)]
#[command(author, version, about = "Quantize BDH checkpoints for web deployment")]
struct Args {
    /// Path to a checkpoint directory or model file.
    #[arg(long, value_name = "PATH")]
    checkpoint: PathBuf,
    /// Specific checkpoint epoch to use when the path is a directory.
    #[arg(long, value_name = "N")]
    epoch: Option<usize>,
    /// Output path for the quantized checkpoint (defaults alongside the input).
    #[arg(long, value_name = "PATH")]
    output: Option<PathBuf>,
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
    let output_base = resolve_output_base(args.output.as_ref(), &checkpoint_base)?;

    let device = <QuantBackend as BackendTrait>::Device::default();
    QuantBackend::seed(&device, 1337);

    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load::<<BDH<QuantBackend> as Module<QuantBackend>>::Record>(
            checkpoint_base.clone(),
            &device,
        )
        .with_context(|| {
            format!(
                "failed to load checkpoint {}",
                format_checkpoint(&checkpoint_base)
            )
        })?;

    BinFileRecorder::<HalfPrecisionSettings>::new()
        .record(record, output_base.clone())
        .with_context(|| {
            format!(
                "failed to write quantized checkpoint {}",
                format_checkpoint(&output_base)
            )
        })?;

    eprintln!(
        "Quantized epoch {epoch} -> {}",
        format_checkpoint(&output_base)
    );

    Ok(())
}

fn resolve_output_base(output: Option<&PathBuf>, checkpoint_base: &Path) -> Result<PathBuf> {
    if let Some(path) = output {
        let base = base_without_extension(path);
        return Ok(base);
    }

    let stem = checkpoint_base
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("unable to determine checkpoint file name"))?;
    let mut output_base = checkpoint_base.to_path_buf();
    output_base.set_file_name(format!("{stem}-f16"));
    Ok(output_base)
}

fn base_without_extension(path: &Path) -> PathBuf {
    let mut base = path.to_path_buf();
    if base.extension().is_some() {
        base.set_extension("");
    }
    base
}

fn format_checkpoint(base: &Path) -> String {
    let mut path = base.to_path_buf();
    path.set_extension("bin");
    path.display().to_string()
}
