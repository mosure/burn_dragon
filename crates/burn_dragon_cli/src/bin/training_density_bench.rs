#[cfg(not(all(feature = "benchmark", feature = "train")))]
fn main() {
    panic!("training_density_bench requires --features benchmark,train");
}

#[cfg(all(feature = "benchmark", feature = "train"))]
mod real {
    use std::path::PathBuf;

    use burn_dragon_cli::bench::training_density::{
        TrainingDensityBenchConfig, parse_training_density_case_spec, run_training_density_bench,
        write_training_density_bench_artifacts,
    };
    use clap::Parser;

    #[derive(Parser, Debug)]
    #[command(name = "training_density_bench")]
    struct Args {
        #[arg(long = "case", value_parser = parse_training_density_case_spec, required = true)]
        cases: Vec<burn_dragon_cli::bench::training_density::TrainingDensityBenchCaseSpec>,
        #[arg(long, default_value = "wgpu")]
        backend: String,
        #[arg(long, default_value_t = 250)]
        sample_interval_ms: u64,
        #[arg(long, default_value_t = 4)]
        warmup_gpu_samples: usize,
        #[arg(long, default_value_t = 4)]
        warmup_iteration_deltas: usize,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long)]
        markdown_path: Option<PathBuf>,
        #[arg(long)]
        json_path: Option<PathBuf>,
    }

    pub fn main() {
        let args = Args::parse();
        let config = TrainingDensityBenchConfig {
            cases: args.cases,
            backend: args.backend,
            sample_interval_ms: args.sample_interval_ms,
            warmup_gpu_samples: args.warmup_gpu_samples,
            warmup_iteration_deltas: args.warmup_iteration_deltas,
            output_dir: args.output_dir,
            markdown_path: args.markdown_path,
            json_path: args.json_path,
        };
        let report = run_training_density_bench(&config).unwrap_or_else(|err| {
            panic!("training density bench failed: {err:#}");
        });
        let markdown = report.to_markdown();
        println!("{markdown}");
        write_training_density_bench_artifacts(&config, &report).unwrap_or_else(|err| {
            panic!("failed to write training density artifacts: {err:#}");
        });
    }
}

#[cfg(all(feature = "benchmark", feature = "train"))]
fn main() {
    real::main();
}
