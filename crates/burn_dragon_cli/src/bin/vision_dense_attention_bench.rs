#[cfg(not(feature = "benchmark"))]
fn main() {
    panic!("vision_dense_attention_bench requires --features benchmark");
}

#[cfg(feature = "benchmark")]
mod real {
    use std::path::PathBuf;

    use burn_dragon::vision::{VisionAttentionMode, load_vision_training_config};
    use burn_dragon_cli::bench::artifact::write_optional_report_artifacts;
    use burn_dragon_cli::bench::vision_dense_attention::{
        VisionDenseAttentionBenchConfig, run_vision_dense_attention_bench,
    };
    use clap::Parser;

    #[derive(Parser, Debug)]
    struct Args {
        #[arg(long, required = true)]
        config: Vec<PathBuf>,
        #[arg(long)]
        batch_size: Option<usize>,
        #[arg(long, default_value_t = 2)]
        warmup: usize,
        #[arg(long, default_value_t = 5)]
        iterations: usize,
        #[arg(long, value_enum)]
        attention_mode: Option<AttentionModeArg>,
        #[arg(long)]
        use_alibi: Option<bool>,
        #[arg(long)]
        markdown_path: Option<PathBuf>,
        #[arg(long)]
        json_path: Option<PathBuf>,
    }

    #[derive(clap::ValueEnum, Clone, Copy, Debug)]
    enum AttentionModeArg {
        RowL1,
        Softmax,
    }

    pub fn main() {
        let args = Args::parse();
        let config = load_vision_training_config(&args.config).unwrap_or_else(|err| {
            panic!("failed to load config overlays {:?}: {err}", args.config)
        });
        let bench = VisionDenseAttentionBenchConfig {
            batch_size: args.batch_size,
            warmup: args.warmup,
            iterations: args.iterations,
            attention_mode: args.attention_mode.map(|mode| match mode {
                AttentionModeArg::RowL1 => VisionAttentionMode::RowL1,
                AttentionModeArg::Softmax => VisionAttentionMode::Softmax,
            }),
            use_alibi: args.use_alibi,
        };
        let report = run_vision_dense_attention_bench(&config, &bench)
            .unwrap_or_else(|err| panic!("dense attention benchmark failed: {err:#}"));
        let markdown = report.to_markdown();
        println!("{markdown}");
        write_optional_report_artifacts(
            args.markdown_path.as_deref(),
            args.json_path.as_deref(),
            &markdown,
            &report,
        )
        .unwrap_or_else(|err| panic!("failed to write dense-attention artifacts: {err:#}"));
    }
}

#[cfg(feature = "benchmark")]
fn main() {
    real::main();
}
