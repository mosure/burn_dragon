#[cfg(not(feature = "benchmark"))]
fn main() {
    panic!("vision_distill_feature_probe_suite requires --features benchmark");
}

#[cfg(feature = "benchmark")]
mod real {
    use std::fmt::Write as _;
    use std::path::PathBuf;

    use burn_dragon::vision::{
        VISION_DISTILL_FEATURE_PROBE_HARNESS_VERSION, VisionDistillFeatureProbeReport,
        load_vision_training_config, run_vision_distill_feature_probe_with_seed,
    };
    use burn_dragon_cli::bench::artifact::write_optional_report_artifacts;
    use clap::Parser;
    use serde::Serialize;

    #[derive(Parser, Debug)]
    struct Args {
        #[arg(long, required = true)]
        config: Vec<PathBuf>,
        #[arg(long, required = true)]
        checkpoint: PathBuf,
        #[arg(long = "step", required = true, num_args = 1..)]
        steps: Vec<usize>,
        #[arg(long, required = true, num_args = 1..)]
        subset_seed: Vec<u64>,
        #[arg(long, default_value_t = 32)]
        batch_size: usize,
        #[arg(long)]
        max_train_samples: Option<usize>,
        #[arg(long)]
        max_val_samples: Option<usize>,
        #[arg(long)]
        markdown_path: Option<PathBuf>,
        #[arg(long)]
        json_path: Option<PathBuf>,
    }

    #[derive(Clone, Serialize)]
    struct ProbeSuiteScalarSummary {
        mean: f64,
        std: f64,
        min: f64,
        max: f64,
    }

    #[derive(Clone, Serialize)]
    struct ProbeSuiteStepSummary {
        step: usize,
        acc: ProbeSuiteScalarSummary,
        delta_vs_s1: ProbeSuiteScalarSummary,
        teacher_gap: ProbeSuiteScalarSummary,
    }

    #[derive(Clone, Serialize)]
    struct ProbeSuiteAggregate {
        teacher_acc: ProbeSuiteScalarSummary,
        best_acc: ProbeSuiteScalarSummary,
        best_minus_s1: ProbeSuiteScalarSummary,
        teacher_minus_best_gap: ProbeSuiteScalarSummary,
        steps: Vec<ProbeSuiteStepSummary>,
    }

    #[derive(Clone, Serialize)]
    struct VisionDistillFeatureProbeSuiteReport {
        benchmark: &'static str,
        harness_version: &'static str,
        config: Vec<PathBuf>,
        checkpoint: PathBuf,
        steps: Vec<usize>,
        subset_seeds: Vec<u64>,
        batch_size: usize,
        max_train_samples: Option<usize>,
        max_val_samples: Option<usize>,
        runs: Vec<VisionDistillFeatureProbeReport>,
        aggregate: ProbeSuiteAggregate,
    }

    impl VisionDistillFeatureProbeSuiteReport {
        fn to_markdown(&self) -> String {
            let mut out = String::new();
            writeln!(&mut out, "# Vision Distill Feature Probe Suite").unwrap();
            writeln!(&mut out).unwrap();
            writeln!(&mut out, "- harness version: {}", self.harness_version).unwrap();
            writeln!(&mut out, "- checkpoint: {}", self.checkpoint.display()).unwrap();
            writeln!(
                &mut out,
                "- config: {}",
                self.config
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(&mut out, "- subset seeds: {:?}", self.subset_seeds).unwrap();
            writeln!(&mut out, "- batch size: {}", self.batch_size).unwrap();
            if let Some(limit) = self.max_train_samples {
                writeln!(&mut out, "- max train samples: {}", limit).unwrap();
            }
            if let Some(limit) = self.max_val_samples {
                writeln!(&mut out, "- max val samples: {}", limit).unwrap();
            }
            writeln!(&mut out).unwrap();
            writeln!(&mut out, "## Aggregate").unwrap();
            writeln!(&mut out).unwrap();
            writeln!(&mut out, "| metric | mean | std | min | max |").unwrap();
            writeln!(&mut out, "|---|---:|---:|---:|---:|").unwrap();
            scalar_row(&mut out, "teacher acc", &self.aggregate.teacher_acc);
            scalar_row(&mut out, "best acc", &self.aggregate.best_acc);
            scalar_row(&mut out, "best minus s1", &self.aggregate.best_minus_s1);
            scalar_row(
                &mut out,
                "teacher minus best gap",
                &self.aggregate.teacher_minus_best_gap,
            );
            writeln!(&mut out).unwrap();
            writeln!(
                &mut out,
                "| step | acc mean | acc std | delta vs s1 mean | teacher gap mean |"
            )
            .unwrap();
            writeln!(&mut out, "|---:|---:|---:|---:|---:|").unwrap();
            for step in &self.aggregate.steps {
                writeln!(
                    &mut out,
                    "| s{} | {:.4} | {:.4} | {:.4} | {:.4} |",
                    step.step,
                    step.acc.mean,
                    step.acc.std,
                    step.delta_vs_s1.mean,
                    step.teacher_gap.mean,
                )
                .unwrap();
            }
            writeln!(&mut out).unwrap();
            writeln!(&mut out, "## Per-seed runs").unwrap();
            writeln!(&mut out).unwrap();
            writeln!(
                &mut out,
                "| subset seed | teacher acc | best step | best acc | best minus s1 | teacher minus best gap |"
            )
            .unwrap();
            writeln!(&mut out, "|---:|---:|---:|---:|---:|---:|").unwrap();
            for run in &self.runs {
                writeln!(
                    &mut out,
                    "| {} | {:.4} | s{} | {:.4} | {:.4} | {:.4} |",
                    run.subset_seed,
                    run.accuracy.teacher_acc,
                    run.accuracy.best_step,
                    run.accuracy.best_acc,
                    run.accuracy.best_minus_s1,
                    run.accuracy.teacher_minus_best_gap,
                )
                .unwrap();
            }
            out
        }
    }

    fn scalar_row(out: &mut String, label: &str, value: &ProbeSuiteScalarSummary) {
        writeln!(
            out,
            "| {} | {:.4} | {:.4} | {:.4} | {:.4} |",
            label, value.mean, value.std, value.min, value.max
        )
        .unwrap();
    }

    fn summarize(values: &[f64]) -> ProbeSuiteScalarSummary {
        let count = values.len().max(1) as f64;
        let mean = values.iter().sum::<f64>() / count;
        let variance = values
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .sum::<f64>()
            / count;
        ProbeSuiteScalarSummary {
            mean,
            std: variance.sqrt(),
            min: values.iter().copied().fold(f64::INFINITY, f64::min),
            max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }

    fn build_aggregate(runs: &[VisionDistillFeatureProbeReport]) -> ProbeSuiteAggregate {
        let teacher_acc = summarize(
            &runs
                .iter()
                .map(|run| run.accuracy.teacher_acc)
                .collect::<Vec<_>>(),
        );
        let best_acc = summarize(
            &runs
                .iter()
                .map(|run| run.accuracy.best_acc)
                .collect::<Vec<_>>(),
        );
        let best_minus_s1 = summarize(
            &runs
                .iter()
                .map(|run| run.accuracy.best_minus_s1)
                .collect::<Vec<_>>(),
        );
        let teacher_minus_best_gap = summarize(
            &runs
                .iter()
                .map(|run| run.accuracy.teacher_minus_best_gap)
                .collect::<Vec<_>>(),
        );

        let step_summaries = runs[0]
            .accuracy
            .step_accuracies
            .iter()
            .map(|step| {
                let acc = summarize(
                    &runs
                        .iter()
                        .map(|run| {
                            run.accuracy
                                .step_accuracies
                                .iter()
                                .find(|candidate| candidate.step == step.step)
                                .expect("step present in every run")
                                .acc
                        })
                        .collect::<Vec<_>>(),
                );
                let delta_vs_s1 = summarize(
                    &runs
                        .iter()
                        .map(|run| {
                            run.accuracy
                                .step_accuracies
                                .iter()
                                .find(|candidate| candidate.step == step.step)
                                .expect("step present in every run")
                                .delta_vs_s1
                        })
                        .collect::<Vec<_>>(),
                );
                let teacher_gap = summarize(
                    &runs
                        .iter()
                        .map(|run| {
                            run.accuracy
                                .step_accuracies
                                .iter()
                                .find(|candidate| candidate.step == step.step)
                                .expect("step present in every run")
                                .teacher_gap
                        })
                        .collect::<Vec<_>>(),
                );
                ProbeSuiteStepSummary {
                    step: step.step,
                    acc,
                    delta_vs_s1,
                    teacher_gap,
                }
            })
            .collect::<Vec<_>>();

        ProbeSuiteAggregate {
            teacher_acc,
            best_acc,
            best_minus_s1,
            teacher_minus_best_gap,
            steps: step_summaries,
        }
    }

    pub fn main() {
        let args = Args::parse();
        let config = load_vision_training_config(&args.config).unwrap_or_else(|err| {
            panic!("failed to load config overlays {:?}: {err}", args.config)
        });
        let mut runs = Vec::with_capacity(args.subset_seed.len());
        for subset_seed in &args.subset_seed {
            let report = run_vision_distill_feature_probe_with_seed(
                &config,
                &args.config,
                &args.checkpoint,
                &args.steps,
                args.batch_size,
                args.max_train_samples,
                args.max_val_samples,
                *subset_seed,
            )
            .unwrap_or_else(|err| panic!("feature probe failed for seed {subset_seed}: {err:#}"));
            runs.push(report);
        }
        let suite = VisionDistillFeatureProbeSuiteReport {
            benchmark: "burn_dragon vision distill feature probe suite",
            harness_version: VISION_DISTILL_FEATURE_PROBE_HARNESS_VERSION,
            config: args.config.clone(),
            checkpoint: args.checkpoint.clone(),
            steps: args.steps.clone(),
            subset_seeds: args.subset_seed.clone(),
            batch_size: args.batch_size,
            max_train_samples: args.max_train_samples,
            max_val_samples: args.max_val_samples,
            aggregate: build_aggregate(&runs),
            runs,
        };
        let markdown = suite.to_markdown();
        println!("{markdown}");
        write_optional_report_artifacts(
            args.markdown_path.as_deref(),
            args.json_path.as_deref(),
            &markdown,
            &suite,
        )
        .unwrap_or_else(|err| panic!("failed to write probe-suite artifacts: {err:#}"));
    }
}

#[cfg(feature = "benchmark")]
fn main() {
    real::main();
}
