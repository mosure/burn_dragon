use std::path::PathBuf;

use anyhow::{Context, Result};
use burn_dragon_universality::{generate_nca_corpus, load_manifest, load_nca_config};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Generate NCA universality corpora for language pre-pretraining"
)]
struct Args {
    /// Path to the NCA corpus TOML config.
    #[arg(short = 'c', long = "config")]
    config: PathBuf,
    /// Number of preview samples to print after generation.
    #[arg(long, default_value_t = 3)]
    print_samples: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = load_nca_config(&args.config)?;
    let report = generate_nca_corpus(&config)?;
    let manifest = load_manifest(&report.manifest_path)?;

    println!("Generated NCA corpus:");
    println!("- manifest: {}", report.manifest_path.display());
    println!("- samples: {}", report.sample_records_path.display());
    println!("- preview_dir: {}", report.preview_dir.display());
    println!(
        "- train_samples: {} validation_samples: {}",
        report.train_samples, report.validation_samples
    );
    println!(
        "- train_tokens: {} validation_tokens: {} total_tokens: {}",
        report.train_token_count,
        report.val_token_count,
        report.train_token_count + report.val_token_count
    );
    println!(
        "- gzip_complexity mean={:.4} min={:.4} max={:.4}",
        manifest.stats.mean_gzip_complexity_ratio,
        manifest.stats.min_gzip_complexity_ratio,
        manifest.stats.max_gzip_complexity_ratio
    );
    println!(
        "- complexity_score mean={:.2} min={:.2} max={:.2}",
        manifest.stats.mean_complexity_score,
        manifest.stats.min_complexity_score,
        manifest.stats.max_complexity_score
    );

    let sample_lines = std::fs::read_to_string(&report.sample_records_path)
        .with_context(|| format!("read {}", report.sample_records_path.display()))?;
    for (index, line) in sample_lines.lines().take(args.print_samples).enumerate() {
        let sample: burn_dragon_universality::UniversalitySampleRecord =
            serde_json::from_str(line).with_context(|| "parse sample record")?;
        println!(
            "\n[sample {}] split={:?} family={} complexity_band={} gzip={:.4} complexity_score={:.2} tokens={} rule_seed={:?} matched={}",
            index,
            sample.split,
            sample.family,
            sample.complexity_band,
            sample.stats.gzip_complexity_ratio,
            sample.stats.complexity_score,
            sample.token_count,
            sample.rule_seed,
            sample.complexity_filter_matched
        );
        if let Some(preview_path) = &sample.preview_path {
            let preview = std::fs::read_to_string(report.preview_dir.join(preview_path))
                .with_context(|| format!("read preview {}", preview_path.display()))?;
            println!("{preview}");
        }
    }

    Ok(())
}
