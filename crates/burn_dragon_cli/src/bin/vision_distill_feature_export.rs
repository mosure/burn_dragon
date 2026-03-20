#[cfg(not(feature = "benchmark"))]
fn main() {
    panic!("vision_distill_feature_export requires --features benchmark");
}

#[cfg(feature = "benchmark")]
mod real {
    use std::path::PathBuf;

    use burn_dragon::vision::{
        export_vision_distill_feature_embeddings, load_vision_training_config,
    };
    use clap::Parser;

    #[derive(Parser, Debug)]
    struct Args {
        #[arg(long, required = true)]
        config: Vec<PathBuf>,
        #[arg(long, required = true)]
        checkpoint: PathBuf,
        #[arg(long)]
        step: usize,
        #[arg(long, default_value_t = 32)]
        batch_size: usize,
        #[arg(long)]
        max_val_samples: Option<usize>,
        #[arg(long, required = true)]
        json_path: PathBuf,
    }

    pub fn main() {
        let args = Args::parse();
        let config = load_vision_training_config(&args.config).unwrap_or_else(|err| {
            panic!("failed to load config overlays {:?}: {err}", args.config)
        });
        let report = export_vision_distill_feature_embeddings(
            &config,
            &args.config,
            &args.checkpoint,
            args.step,
            args.batch_size,
            args.max_val_samples,
        )
        .unwrap_or_else(|err| panic!("feature export failed: {err:#}"));
        std::fs::write(
            &args.json_path,
            serde_json::to_vec_pretty(&report).expect("serialize feature export"),
        )
        .unwrap_or_else(|err| {
            panic!(
                "failed to write feature export {}: {err}",
                args.json_path.display()
            )
        });
        println!(
            "wrote {} embedding pairs to {}",
            report.records,
            args.json_path.display()
        );
    }
}

#[cfg(feature = "benchmark")]
fn main() {
    real::main();
}
