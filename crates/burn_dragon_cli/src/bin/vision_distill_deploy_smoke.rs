#[cfg(not(feature = "benchmark"))]
fn main() {
    panic!("vision_distill_deploy_smoke requires --features benchmark");
}

#[cfg(feature = "benchmark")]
mod real {
    use std::path::PathBuf;

    use burn_dragon::vision::{
        VisionDistillDeploySmokePrecision, load_vision_training_config,
        run_vision_distill_deploy_smoke,
    };
    use burn_dragon_cli::bench::artifact::write_optional_report_artifacts;
    use clap::Parser;
    use clap::ValueEnum;
    use serde::Serialize;

    #[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
    enum PrecisionArg {
        F16,
        F32,
    }

    impl From<PrecisionArg> for VisionDistillDeploySmokePrecision {
        fn from(value: PrecisionArg) -> Self {
            match value {
                PrecisionArg::F16 => Self::F16,
                PrecisionArg::F32 => Self::F32,
            }
        }
    }

    #[derive(Parser, Debug)]
    struct Args {
        #[arg(long, required = true)]
        config: Vec<PathBuf>,
        #[arg(long, required = true)]
        checkpoint: PathBuf,
        #[arg(long, required = true)]
        step: usize,
        #[arg(long)]
        backprop_steps: Option<usize>,
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        #[arg(long, default_value_t = 5)]
        iterations: usize,
        #[arg(long, value_enum, default_value_t = PrecisionArg::F32)]
        precision: PrecisionArg,
        #[arg(long)]
        markdown_path: Option<PathBuf>,
        #[arg(long)]
        json_path: Option<PathBuf>,
    }

    pub fn main() {
        let args = Args::parse();
        let config = load_vision_training_config(&args.config).unwrap_or_else(|err| {
            panic!("failed to load config overlays {:?}: {err}", args.config)
        });
        let report = run_vision_distill_deploy_smoke(
            &config,
            &args.config,
            &args.checkpoint,
            args.step,
            args.backprop_steps,
            args.batch_size,
            args.warmup,
            args.iterations,
            args.precision.into(),
        )
        .unwrap_or_else(|err| panic!("vision deploy smoke failed: {err:#}"));
        let markdown = report.to_markdown();
        println!("{markdown}");
        write_optional_report_artifacts(
            args.markdown_path.as_deref(),
            args.json_path.as_deref(),
            &markdown,
            &report,
        )
        .unwrap_or_else(|err| panic!("failed to write deploy-smoke artifacts: {err:#}"));
    }
}

#[cfg(feature = "benchmark")]
fn main() {
    real::main();
}
