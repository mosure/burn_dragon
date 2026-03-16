#[cfg(not(feature = "benchmark"))]
fn main() {
    panic!("vision_dense_step_bench requires --features benchmark");
}

#[cfg(feature = "benchmark")]
mod real {
    use std::path::PathBuf;

    use burn_dragon::vision::load_vision_training_config;
    use burn_dragon_cli::bench::artifact::write_optional_report_artifacts;
    use burn_dragon_cli::bench::vision_dense_step::{
        VisionDenseStepBenchConfig, run_vision_dense_step_bench,
    };
    use clap::Parser;

    #[derive(Parser, Debug)]
    #[command(name = "vision_dense_step_bench")]
    struct Args {
        #[arg(long, required = true)]
        config: Vec<PathBuf>,
        #[arg(long)]
        batch_size: Option<usize>,
        #[arg(long, default_value_t = 2)]
        warmup: usize,
        #[arg(long, default_value_t = 5)]
        iterations: usize,
        #[arg(long, value_delimiter = ',', default_values_t = vec![4usize, 8usize])]
        rollout_steps: Vec<usize>,
        #[arg(long, default_value_t = 0)]
        gpu_index: usize,
        #[arg(long, default_value_t = 100)]
        power_sample_ms: u64,
        #[arg(long, default_value_t = 2000)]
        power_phase_ms: u64,
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
        let bench = VisionDenseStepBenchConfig {
            batch_size: args.batch_size,
            warmup: args.warmup,
            iterations: args.iterations,
            rollout_steps: args.rollout_steps,
            gpu_index: args.gpu_index,
            power_sample_ms: args.power_sample_ms,
            power_phase_ms: args.power_phase_ms,
        };
        let report = run_vision_dense_step_bench(&config, &bench, &args.config)
            .unwrap_or_else(|err| panic!("dense step benchmark failed: {err:#}"));
        let markdown = report.to_markdown();
        println!("{markdown}");
        write_optional_report_artifacts(
            args.markdown_path.as_deref(),
            args.json_path.as_deref(),
            &markdown,
            &report,
        )
        .unwrap_or_else(|err| panic!("failed to write dense-step artifacts: {err:#}"));
    }
}

#[cfg(feature = "benchmark")]
fn main() {
    real::main();
}
