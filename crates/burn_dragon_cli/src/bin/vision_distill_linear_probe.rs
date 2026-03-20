#[cfg(not(feature = "benchmark"))]
fn main() {
    panic!("vision_distill_linear_probe requires --features benchmark");
}

#[cfg(feature = "benchmark")]
mod real {
    use std::path::PathBuf;

    use burn_dragon::vision::{
        load_vision_training_config, run_vision_distill_linear_probe_for_teacher_with_seed,
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
        max_train_samples: Option<usize>,
        #[arg(long)]
        max_val_samples: Option<usize>,
        #[arg(long, default_value_t = 0)]
        subset_seed: u64,
        #[arg(long, default_value_t = 1337)]
        train_seed: u64,
        #[arg(long)]
        teacher_target: Option<String>,
        #[arg(long, default_value_t = 200)]
        epochs: usize,
        #[arg(long, default_value_t = 0.2)]
        learning_rate: f32,
        #[arg(long, default_value_t = 1e-4)]
        weight_decay: f32,
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
        let report = run_vision_distill_linear_probe_for_teacher_with_seed(
            &config,
            &args.config,
            &args.checkpoint,
            &args.steps,
            args.batch_size,
            args.max_train_samples,
            args.max_val_samples,
            args.subset_seed,
            args.train_seed,
            args.epochs,
            args.learning_rate,
            args.weight_decay,
            args.teacher_target.as_deref(),
        )
        .unwrap_or_else(|err| panic!("linear probe failed: {err:#}"));
        let markdown = report.to_markdown();
        println!("{markdown}");
        write_optional_report_artifacts(
            args.markdown_path.as_deref(),
            args.json_path.as_deref(),
            &markdown,
            &report,
        )
        .unwrap_or_else(|err| panic!("failed to write linear-probe artifacts: {err:#}"));
    }
}

#[cfg(feature = "benchmark")]
fn main() {
    real::main();
}
