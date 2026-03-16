#[cfg(not(feature = "benchmark"))]
fn main() {
    panic!("vision_distill_serving_bench requires --features benchmark");
}

#[cfg(feature = "benchmark")]
mod real {
    use std::path::PathBuf;

    use burn_dragon_cli::bench::artifact::write_optional_report_artifacts;
    use burn_dragon::vision::{
        load_vision_training_config, run_vision_distill_serving_benchmark,
    };
    use clap::Parser;

    #[derive(Parser, Debug)]
    struct Args {
        #[arg(long, required = true)]
        config: Vec<PathBuf>,
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long, required = true)]
        step: usize,
        #[arg(long)]
        backprop_steps: Option<usize>,
        #[arg(long, default_value_t = 1)]
        baseline_step: usize,
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        #[arg(long, default_value_t = 10)]
        iterations: usize,
        #[arg(long, default_value_t = 1)]
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
        let report = run_vision_distill_serving_benchmark(
            &config,
            &args.config,
            args.checkpoint.as_deref(),
            adapter_info(),
            args.step,
            args.backprop_steps,
            args.baseline_step,
            args.warmup,
            args.iterations,
            args.batch_size,
        )
        .unwrap_or_else(|err| panic!("vision serving benchmark failed: {err:#}"));
        let markdown = report.to_markdown();
        println!("{markdown}");
        write_optional_report_artifacts(
            args.markdown_path.as_deref(),
            args.json_path.as_deref(),
            &markdown,
            &report,
        )
        .unwrap_or_else(|err| panic!("failed to write serving artifacts: {err:#}"));
    }

    fn adapter_info() -> String {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("wgpu adapter");
        let info = adapter.get_info();
        format!("{} ({:?})", info.name, info.device_type)
    }
}

#[cfg(feature = "benchmark")]
fn main() {
    real::main();
}
