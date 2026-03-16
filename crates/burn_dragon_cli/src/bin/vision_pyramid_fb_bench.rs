#[cfg(not(feature = "benchmark"))]
fn main() {
    panic!("vision_pyramid_fb_bench requires --features benchmark");
}

#[cfg(feature = "benchmark")]
mod real {
    use std::path::PathBuf;

    use burn_dragon_cli::bench::artifact::write_optional_report_artifacts;
    use burn_dragon_cli::bench::vision_pyramid_fb::{
        VisionPyramidForwardBackwardBenchConfig, run_vision_pyramid_forward_backward_bench,
    };
    use clap::Parser;

    #[derive(Parser, Debug)]
    struct Args {
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        #[arg(long, default_value_t = 5)]
        repetitions: usize,
        #[arg(long, default_value_t = 4)]
        steps: usize,
        #[arg(long)]
        markdown_path: Option<PathBuf>,
        #[arg(long)]
        json_path: Option<PathBuf>,
    }

    pub fn main() {
        let args = Args::parse();
        let report =
            run_vision_pyramid_forward_backward_bench(&VisionPyramidForwardBackwardBenchConfig {
                warmup: args.warmup,
                repetitions: args.repetitions,
                steps: args.steps,
            })
            .unwrap_or_else(|err| panic!("vision pyramid fb bench failed: {err:#}"));
        let markdown = report.to_markdown();
        println!("{markdown}");
        write_optional_report_artifacts(
            args.markdown_path.as_deref(),
            args.json_path.as_deref(),
            &markdown,
            &report,
        )
        .unwrap_or_else(|err| panic!("failed to write vision-pyramid-fb artifacts: {err:#}"));
    }
}

#[cfg(feature = "benchmark")]
fn main() {
    real::main();
}
