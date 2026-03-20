#[cfg(not(feature = "benchmark"))]
fn main() {
    panic!("vision_distill_decode_probe requires --features benchmark");
}

#[cfg(feature = "benchmark")]
mod real {
    use std::path::PathBuf;

    #[cfg(feature = "cuda")]
    use burn_dragon::vision::run_vision_distill_decode_probe_cuda_with_seed;
    use burn_dragon::vision::{
        load_vision_training_config, run_vision_distill_decode_probe_with_seed,
    };
    use burn_dragon_cli::bench::artifact::write_optional_report_artifacts;
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
        max_val_samples: Option<usize>,
        #[arg(long, default_value_t = 0)]
        subset_seed: u64,
        #[arg(long, default_value = "cpu")]
        backend: String,
        #[arg(long)]
        teacher_target: Option<String>,
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
        let report = match args.backend.as_str() {
            "cpu" | "ndarray" => run_vision_distill_decode_probe_with_seed(
                &config,
                &args.config,
                &args.checkpoint,
                &args.steps,
                args.batch_size,
                args.max_val_samples,
                args.subset_seed,
                args.teacher_target.as_deref(),
            ),
            "cuda" => {
                #[cfg(feature = "cuda")]
                {
                    run_vision_distill_decode_probe_cuda_with_seed(
                        &config,
                        &args.config,
                        &args.checkpoint,
                        &args.steps,
                        args.batch_size,
                        args.max_val_samples,
                        args.subset_seed,
                        args.teacher_target.as_deref(),
                    )
                }
                #[cfg(not(feature = "cuda"))]
                {
                    panic!("decode probe backend `cuda` requires --features cuda");
                }
            }
            other => panic!("unsupported decode probe backend `{other}` (expected cpu or cuda)"),
        }
        .unwrap_or_else(|err| panic!("decode probe failed: {err:#}"));
        let markdown = report.to_markdown();
        println!("{markdown}");
        write_optional_report_artifacts(
            args.markdown_path.as_deref(),
            args.json_path.as_deref(),
            &markdown,
            &report,
        )
        .unwrap_or_else(|err| panic!("failed to write decode-probe artifacts: {err:#}"));
    }
}

#[cfg(feature = "benchmark")]
fn main() {
    real::main();
}
