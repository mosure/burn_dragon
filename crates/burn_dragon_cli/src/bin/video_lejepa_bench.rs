#[cfg(not(feature = "benchmark"))]
fn main() {
    panic!("video_lejepa_bench requires --features benchmark");
}

#[cfg(feature = "benchmark")]
mod real {
    use std::path::PathBuf;

    use burn_dragon_cli::bench::artifact::write_optional_report_artifacts;
    use burn_dragon_cli::bench::video_lejepa::{VideoLejepaBenchConfig, run_video_lejepa_bench};
    use clap::Parser;

    #[derive(Parser, Debug)]
    struct Args {
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        #[arg(long, default_value_t = 5)]
        iterations: usize,
        #[arg(long)]
        markdown_path: Option<PathBuf>,
        #[arg(long)]
        json_path: Option<PathBuf>,
    }

    pub fn main() {
        let args = Args::parse();
        let config = VideoLejepaBenchConfig {
            warmup: args.warmup,
            iterations: args.iterations,
        };
        let report = run_video_lejepa_bench(&config)
            .unwrap_or_else(|err| panic!("video LEJEPA bench failed: {err:#}"));
        let markdown = report.to_markdown();
        println!("{markdown}");
        write_optional_report_artifacts(
            args.markdown_path.as_deref(),
            args.json_path.as_deref(),
            &markdown,
            &report,
        )
        .unwrap_or_else(|err| panic!("failed to write video bench artifacts: {err:#}"));
    }
}

#[cfg(feature = "benchmark")]
fn main() {
    real::main();
}
