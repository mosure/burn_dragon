#[cfg(not(feature = "benchmark"))]
fn main() {
    panic!("vision_distill_feature_probe requires --features benchmark");
}

#[cfg(feature = "benchmark")]
mod real {
    use std::path::PathBuf;

    use burn_dragon_cli::bench::artifact::write_optional_report_artifacts;
    use burn_dragon::vision::{load_vision_training_config, run_vision_distill_feature_probe};
    use clap::Parser;

    #[derive(Parser, Debug)]
    struct Args {
        #[arg(long, required = true)]
        config: Vec<PathBuf>,
        #[arg(long, required = true)]
        checkpoint: PathBuf,
        #[arg(long = "step", required = true, num_args = 1..)]
        steps: Vec<usize>,
        #[arg(long, default_value_t = 32)]
        batch_size: usize,
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
        let report = run_vision_distill_feature_probe(
            &config,
            &args.config,
            &args.checkpoint,
            &args.steps,
            args.batch_size,
        )
        .unwrap_or_else(|err| panic!("feature probe failed: {err:#}"));
        let markdown = report.to_markdown();
        println!("{markdown}");
        write_optional_report_artifacts(
            args.markdown_path.as_deref(),
            args.json_path.as_deref(),
            &markdown,
            &report,
        )
        .unwrap_or_else(|err| panic!("failed to write feature-probe artifacts: {err:#}"));
    }
}

#[cfg(feature = "benchmark")]
fn main() {
    real::main();
}
