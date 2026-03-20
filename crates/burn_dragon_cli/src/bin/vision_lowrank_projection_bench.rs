#[cfg(not(all(feature = "benchmark", feature = "train")))]
fn main() {
    panic!("vision_lowrank_projection_bench requires --features benchmark,train");
}

#[cfg(all(feature = "benchmark", feature = "train"))]
mod real {
    use std::path::PathBuf;

    use burn_dragon::vision::load_vision_training_config;
    use burn_dragon_cli::bench::artifact::write_optional_report_artifacts;
    use burn_dragon_cli::bench::vision_lowrank_projection::{
        VisionLowrankProjectionBenchBackendKind, VisionLowrankProjectionBenchConfig,
        run_vision_lowrank_projection_bench,
    };
    use clap::Parser;

    #[derive(Parser, Debug)]
    #[command(name = "vision_lowrank_projection_bench")]
    struct Args {
        #[arg(long, required = true)]
        config: Vec<PathBuf>,
        #[arg(long, default_value = "wgpu")]
        backend: String,
        #[arg(long)]
        batch_size: Option<usize>,
        #[arg(long, default_value_t = 2)]
        warmup: usize,
        #[arg(long, default_value_t = 5)]
        iterations: usize,
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
        let bench = VisionLowrankProjectionBenchConfig {
            config_paths: args.config.clone(),
            backend: match args.backend.as_str() {
                "wgpu" => VisionLowrankProjectionBenchBackendKind::Wgpu,
                "wgpu-no-fusion" | "wgpu-nofusion" => {
                    VisionLowrankProjectionBenchBackendKind::WgpuNoFusion
                }
                other => panic!("unsupported backend {other:?} (expected wgpu or wgpu-no-fusion)"),
            },
            batch_size: args.batch_size,
            warmup: args.warmup,
            iterations: args.iterations,
            gpu_index: args.gpu_index,
            power_sample_ms: args.power_sample_ms,
            power_phase_ms: args.power_phase_ms,
        };
        let report = run_vision_lowrank_projection_bench(&config, &bench)
            .unwrap_or_else(|err| panic!("vision lowrank projection bench failed: {err:#}"));
        let markdown = report.to_markdown();
        println!("{markdown}");
        write_optional_report_artifacts(
            args.markdown_path.as_deref(),
            args.json_path.as_deref(),
            &markdown,
            &report,
        )
        .unwrap_or_else(|err| panic!("failed to write projection-bench artifacts: {err:#}"));
    }
}

#[cfg(all(feature = "benchmark", feature = "train"))]
fn main() {
    real::main();
}
