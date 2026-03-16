#[cfg(not(feature = "benchmark"))]
fn main() {
    panic!("vision_distill_resolution_bench requires --features benchmark");
}

#[cfg(feature = "benchmark")]
mod real {
    use std::path::PathBuf;

    use burn_dragon::vision::load_vision_training_config;
    use burn_dragon_cli::bench::artifact::write_optional_report_artifacts;
    use burn_dragon_cli::bench::vision_resolution::{
        VisionResolutionBenchConfig, run_vision_resolution_bench,
    };
    use clap::Parser;

    #[derive(Parser, Debug)]
    struct Args {
        #[arg(long, required = true)]
        config: Vec<PathBuf>,
        #[arg(long, value_delimiter = ',', num_args = 1.., default_values_t = [112usize, 168, 224, 280])]
        resolutions: Vec<usize>,
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        #[arg(long, default_value_t = 5)]
        iterations: usize,
        #[arg(long)]
        batch_size: Option<usize>,
        #[arg(long)]
        markdown_path: Option<PathBuf>,
        #[arg(long)]
        json_path: Option<PathBuf>,
    }

    pub fn main() {
        let args = Args::parse();
        let base_config = load_vision_training_config(&args.config).unwrap_or_else(|err| {
            panic!("failed to load config overlays {:?}: {err}", args.config)
        });
        let bench = VisionResolutionBenchConfig {
            config_paths: args.config.clone(),
            resolutions: args.resolutions,
            warmup: args.warmup,
            iterations: args.iterations,
            batch_size: args.batch_size,
        };
        let report = run_vision_resolution_bench(&base_config, &bench)
            .unwrap_or_else(|err| panic!("resolution bench failed: {err:#}"));
        let markdown = report.to_markdown();
        println!("{markdown}");
        write_optional_report_artifacts(
            args.markdown_path.as_deref(),
            args.json_path.as_deref(),
            &markdown,
            &report,
        )
        .unwrap_or_else(|err| panic!("failed to write resolution-bench artifacts: {err:#}"));
    }
}

#[cfg(feature = "benchmark")]
fn main() {
    real::main();
}
