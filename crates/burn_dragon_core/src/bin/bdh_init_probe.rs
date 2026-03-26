#![recursion_limit = "256"]

#[cfg(not(feature = "probe"))]
fn main() {
    panic!("bdh_init_probe requires --features probe");
}

#[cfg(feature = "probe")]
mod real {
    use std::collections::BTreeMap;
    use std::fmt::Write as _;
    use std::fs;
    use std::path::{Path, PathBuf};

    use anyhow::{Context, Result, anyhow, bail};
    use burn::nn::loss::CrossEntropyLossConfig;
    use burn::optim::{AdamWConfig, GradientsParams, LearningRate, Optimizer};
    use burn::tensor::backend::Backend as BackendTrait;
    use burn::tensor::{Int, Tensor, TensorData};
    use burn_autodiff::Autodiff;
    use burn_dragon_core::{
        BDH, BDHConfig, BdhFiringTargetConfig, BdhFiringTargetKind, BdhInitializationConfig,
        BdhInitializationKind, BdhNeuronGainConfig, BdhNeuronGainKind, BdhResidualScalingConfig,
        BdhResidualScalingKind, BdhTopologyPriorConfig, BdhTopologyPriorKind,
        LanguageBdhInitLayerDiagnostics, LayerVizState,
    };
    use burn_ndarray::NdArray;
    use clap::{Parser, ValueEnum};
    use serde::Serialize;

    type Backend = NdArray<f32>;
    type TrainBackend = Autodiff<NdArray<f32>>;

    const P_X_MIN: f64 = 0.05;
    const P_X_MAX: f64 = 0.35;
    const P_Y_MIN: f64 = 0.01;
    const P_Y_MAX: f64 = 0.10;
    const R_RES_MIN: f64 = 0.05;
    const R_RES_MAX: f64 = 0.50;

    #[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
    enum InitKindArg {
        NearCritical,
        SimpleNormal,
        HeGlorot,
        HeadwiseSemiOrthogonal,
    }

    impl From<InitKindArg> for BdhInitializationKind {
        fn from(value: InitKindArg) -> Self {
            match value {
                InitKindArg::NearCritical => Self::NearCritical,
                InitKindArg::SimpleNormal => Self::SimpleNormal,
                InitKindArg::HeGlorot => Self::HeGlorot,
                InitKindArg::HeadwiseSemiOrthogonal => Self::HeadwiseSemiOrthogonal,
            }
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
    enum ResidualScalingArg {
        FamilyDefault,
        Disabled,
        DepthScaled,
    }

    impl From<ResidualScalingArg> for BdhResidualScalingKind {
        fn from(value: ResidualScalingArg) -> Self {
            match value {
                ResidualScalingArg::FamilyDefault => Self::FamilyDefault,
                ResidualScalingArg::Disabled => Self::Disabled,
                ResidualScalingArg::DepthScaled => Self::DepthScaled,
            }
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
    enum NeuronGainArg {
        Iid,
        HeavyTailedLogNormal,
    }

    impl From<NeuronGainArg> for BdhNeuronGainKind {
        fn from(value: NeuronGainArg) -> Self {
            match value {
                NeuronGainArg::Iid => Self::Iid,
                NeuronGainArg::HeavyTailedLogNormal => Self::HeavyTailedLogNormal,
            }
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
    enum TopologyPriorArg {
        Iid,
        ModularBridges,
    }

    impl From<TopologyPriorArg> for BdhTopologyPriorKind {
        fn from(value: TopologyPriorArg) -> Self {
            match value {
                TopologyPriorArg::Iid => Self::Iid,
                TopologyPriorArg::ModularBridges => Self::ModularBridges,
            }
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
    enum FiringTargetsArg {
        Disabled,
        GaussianEstimate,
        ExplicitThresholds,
    }

    impl From<FiringTargetsArg> for BdhFiringTargetKind {
        fn from(value: FiringTargetsArg) -> Self {
            match value {
                FiringTargetsArg::Disabled => Self::Disabled,
                FiringTargetsArg::GaussianEstimate => Self::GaussianEstimate,
                FiringTargetsArg::ExplicitThresholds => Self::ExplicitThresholds,
            }
        }
    }

    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, ValueEnum)]
    #[serde(rename_all = "snake_case")]
    enum CalibrationArg {
        #[default]
        Disabled,
        FiringThresholds,
        LsuvBdh,
    }

    #[derive(Parser, Debug)]
    struct Args {
        #[arg(long, default_value_t = 8)]
        n_layer: usize,
        #[arg(long, default_value_t = 256)]
        n_embd: usize,
        #[arg(long, default_value_t = 4)]
        n_head: usize,
        #[arg(long, default_value_t = 32768)]
        latent_total: usize,
        #[arg(long, default_value_t = 32768)]
        vocab_size: usize,
        #[arg(long, default_value_t = 2)]
        batch_size: usize,
        #[arg(long, default_value_t = 64)]
        block_size: usize,
        #[arg(long, default_value_t = 4)]
        eval_batches: usize,
        #[arg(long, default_value_t = 1337)]
        seed: u64,
        #[arg(long, default_value_t = false)]
        carry_state: bool,
        #[arg(long, value_enum, default_value_t = InitKindArg::SimpleNormal)]
        initialization: InitKindArg,
        #[arg(long, value_enum, default_value_t = ResidualScalingArg::DepthScaled)]
        residual_scaling: ResidualScalingArg,
        #[arg(long, default_value_t = 1.0)]
        residual_gain: f64,
        #[arg(long, value_enum, default_value_t = NeuronGainArg::HeavyTailedLogNormal)]
        neuron_gains: NeuronGainArg,
        #[arg(long, default_value_t = 0.75)]
        gain_log_sigma: f64,
        #[arg(long, default_value_t = 4.0)]
        gain_max: f64,
        #[arg(long, value_enum, default_value_t = TopologyPriorArg::Iid)]
        topology_prior: TopologyPriorArg,
        #[arg(long, default_value_t = 4)]
        topology_community_count: usize,
        #[arg(long, default_value_t = 0.05)]
        topology_bridge_fraction: f64,
        #[arg(long, default_value_t = 1.5)]
        topology_intra_gain: f64,
        #[arg(long, default_value_t = 0.5)]
        topology_inter_gain: f64,
        #[arg(long, default_value_t = 1.0)]
        topology_bridge_gain: f64,
        #[arg(long, value_enum, default_value_t = FiringTargetsArg::Disabled)]
        firing_targets: FiringTargetsArg,
        #[arg(long, default_value_t = 0.15)]
        x_target: f64,
        #[arg(long, default_value_t = 0.05)]
        y_target: f64,
        #[arg(long, default_value_t = 0.0)]
        x_threshold: f64,
        #[arg(long, default_value_t = 0.0)]
        y_threshold: f64,
        #[arg(long, default_value_t = 0.02)]
        simple_normal_std: f64,
        #[arg(long, value_enum, default_value_t = CalibrationArg::Disabled)]
        calibration: CalibrationArg,
        #[arg(long, default_value_t = false, hide = true)]
        calibrate_firing_thresholds: bool,
        #[arg(long, default_value_t = 2)]
        calibration_batches: usize,
        #[arg(long, default_value_t = 2)]
        calibration_rounds: usize,
        #[arg(long, default_value_t = 0.15811388300841897)]
        target_r_res: f64,
        #[arg(long, default_value_t = 64.0)]
        max_residual_gain: f64,
        #[arg(long, default_value_t = 0)]
        backward_steps: usize,
        #[arg(long, default_value_t = 1.0e-4)]
        backward_learning_rate: f64,
        #[arg(long, default_value_t = 1.0e-4)]
        perturbation_epsilon: f64,
        #[arg(long, default_value_t = 0.0)]
        latent_activity_threshold: f64,
        #[arg(long, default_value_t = 1024)]
        graph_pair_samples: usize,
        #[arg(long)]
        markdown_path: Option<PathBuf>,
        #[arg(long)]
        json_path: Option<PathBuf>,
    }

    #[derive(Clone)]
    struct EvalBatch<B: BackendTrait> {
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    }

    #[derive(Clone, Debug, Default)]
    struct DiagnosticsAccumulator {
        count: usize,
        lowrank_active_count: usize,
        finite_count: usize,
        p_x_sum: f64,
        p_x_count: usize,
        p_y_sum: f64,
        p_y_count: usize,
        current_rms_sum: f64,
        current_rms_count: usize,
        recurrent_readout_rms_sum: f64,
        recurrent_readout_rms_count: usize,
        recurrent_readout_ratio_sum: f64,
        recurrent_readout_ratio_count: usize,
        residual_delta_rms_sum: f64,
        residual_delta_rms_count: usize,
        r_res_sum: f64,
        r_res_count: usize,
    }

    #[derive(Clone)]
    struct LatentActivityBatch {
        heads: usize,
        latent: usize,
        active_by_head: Vec<u8>,
    }

    #[derive(Clone, Serialize)]
    struct Phase0Criteria {
        p_x_band: [f64; 2],
        p_y_band: [f64; 2],
        r_res_band: [f64; 2],
        sparse_positive_pass: bool,
        backward_finite_pass: bool,
        phase0_pass: bool,
        passing_layers: Vec<usize>,
    }

    #[derive(Clone, Copy, Debug, Default, Serialize)]
    struct ProbeMetricSummary {
        finite: bool,
        p_x: Option<f64>,
        p_y: Option<f64>,
        r_res: Option<f64>,
        recurrent_readout_ratio: Option<f64>,
    }

    #[derive(Clone, Serialize)]
    struct CalibrationSummary {
        enabled: bool,
        kind: CalibrationArg,
        rounds: usize,
        calibration_batches: usize,
        target_r_res: Option<f64>,
        final_x_threshold: Option<f64>,
        final_y_threshold: Option<f64>,
        final_residual_gain: Option<f64>,
        post_calibration: Option<ProbeMetricSummary>,
    }

    #[derive(Clone, Serialize)]
    struct BackwardCheckSummary {
        enabled: bool,
        steps: usize,
        completed_steps: usize,
        tokens_per_step: usize,
        learning_rate: f64,
        finite: bool,
        losses: Vec<f64>,
        aulc: Option<f64>,
        min_loss: Option<f64>,
        max_loss: Option<f64>,
        final_loss: Option<f64>,
    }

    #[derive(Clone, Serialize)]
    struct PerturbationLayerSummary {
        layer_index: usize,
        samples: usize,
        input_delta_rms: Option<f64>,
        output_delta_rms: Option<f64>,
        gain: Option<f64>,
    }

    #[derive(Clone, Serialize)]
    struct PerturbationSummary {
        enabled: bool,
        epsilon: f64,
        layers: Vec<PerturbationLayerSummary>,
        mean_gain: Option<f64>,
    }

    #[derive(Clone, Serialize)]
    struct GraphLayerSummary {
        layer_index: usize,
        samples: usize,
        pair_samples: usize,
        active_rate_mean: Option<f64>,
        sampled_edge_density: Option<f64>,
        degree_tail_ratio: Option<f64>,
        community_concentration: Option<f64>,
        within_pair_coactivation: Option<f64>,
        across_pair_coactivation: Option<f64>,
        modularity_gap: Option<f64>,
    }

    #[derive(Clone, Serialize)]
    struct GraphSummary {
        enabled: bool,
        signal: &'static str,
        activation_threshold: f64,
        pair_samples: usize,
        community_count: usize,
        layers: Vec<GraphLayerSummary>,
    }

    #[derive(Clone, Serialize)]
    struct UpdateIntervalLayerSummary {
        layer_index: usize,
        samples: usize,
        interval_count: usize,
        active_rate_mean: Option<f64>,
        mean_interval: Option<f64>,
        max_interval: Option<usize>,
        tail_ratio: Option<f64>,
    }

    #[derive(Clone, Serialize)]
    struct UpdateIntervalSummary {
        enabled: bool,
        signal: &'static str,
        activation_threshold: f64,
        layers: Vec<UpdateIntervalLayerSummary>,
    }

    #[derive(Clone, Serialize)]
    struct Report {
        benchmark: &'static str,
        n_layer: usize,
        n_embd: usize,
        n_head: usize,
        latent_total: usize,
        vocab_size: usize,
        batch_size: usize,
        block_size: usize,
        eval_batches: usize,
        seed: u64,
        carry_state: bool,
        initialization: BdhInitializationKind,
        residual_scaling: BdhResidualScalingKind,
        neuron_gains: BdhNeuronGainKind,
        topology_prior: BdhTopologyPriorKind,
        firing_targets: BdhFiringTargetKind,
        avg_loss: f64,
        layers: Vec<LanguageBdhInitLayerDiagnostics>,
        phase0: Phase0Criteria,
        calibration: CalibrationSummary,
        backward: BackwardCheckSummary,
        perturbation: PerturbationSummary,
        graph: GraphSummary,
        update_intervals: UpdateIntervalSummary,
    }

    impl CalibrationSummary {
        fn disabled() -> Self {
            Self {
                enabled: false,
                kind: CalibrationArg::Disabled,
                rounds: 0,
                calibration_batches: 0,
                target_r_res: None,
                final_x_threshold: None,
                final_y_threshold: None,
                final_residual_gain: None,
                post_calibration: None,
            }
        }
    }

    impl BackwardCheckSummary {
        fn disabled() -> Self {
            Self {
                enabled: false,
                steps: 0,
                completed_steps: 0,
                tokens_per_step: 0,
                learning_rate: 0.0,
                finite: false,
                losses: Vec::new(),
                aulc: None,
                min_loss: None,
                max_loss: None,
                final_loss: None,
            }
        }
    }

    impl PerturbationSummary {
        fn disabled(epsilon: f64) -> Self {
            Self {
                enabled: false,
                epsilon,
                layers: Vec::new(),
                mean_gain: None,
            }
        }
    }

    impl GraphSummary {
        fn disabled(
            activation_threshold: f64,
            pair_samples: usize,
            community_count: usize,
        ) -> Self {
            Self {
                enabled: false,
                signal: "y_neuron_last",
                activation_threshold,
                pair_samples,
                community_count,
                layers: Vec::new(),
            }
        }
    }

    impl UpdateIntervalSummary {
        fn disabled(activation_threshold: f64) -> Self {
            Self {
                enabled: false,
                signal: "y_neuron_last",
                activation_threshold,
                layers: Vec::new(),
            }
        }
    }

    pub fn main() {
        if let Err(err) = run() {
            eprintln!("error: {err:#}");
            std::process::exit(1);
        }
    }

    fn run() -> Result<()> {
        let args = Args::parse();
        if args.n_embd == 0 || args.n_head == 0 || args.n_layer == 0 {
            bail!("n_layer, n_embd, and n_head must be positive");
        }
        if args.latent_total == 0 || args.latent_total % args.n_embd != 0 {
            bail!(
                "latent_total must be positive and divisible by n_embd (got {} and {})",
                args.latent_total,
                args.n_embd
            );
        }
        if args.latent_total % args.n_head != 0 {
            bail!(
                "latent_total must be divisible by n_head (got {} and {})",
                args.latent_total,
                args.n_head
            );
        }
        if args.vocab_size < 2 {
            bail!("vocab_size must be at least 2");
        }
        if !args.target_r_res.is_finite() || args.target_r_res <= 0.0 {
            bail!(
                "target_r_res must be finite and > 0 (got {})",
                args.target_r_res
            );
        }
        if !args.max_residual_gain.is_finite() || args.max_residual_gain <= 0.0 {
            bail!(
                "max_residual_gain must be finite and > 0 (got {})",
                args.max_residual_gain
            );
        }
        if !args.backward_learning_rate.is_finite() || args.backward_learning_rate <= 0.0 {
            bail!(
                "backward_learning_rate must be finite and > 0 (got {})",
                args.backward_learning_rate
            );
        }
        if !args.perturbation_epsilon.is_finite() || args.perturbation_epsilon < 0.0 {
            bail!(
                "perturbation_epsilon must be finite and >= 0 (got {})",
                args.perturbation_epsilon
            );
        }
        if !args.latent_activity_threshold.is_finite() || args.latent_activity_threshold < 0.0 {
            bail!(
                "latent_activity_threshold must be finite and >= 0 (got {})",
                args.latent_activity_threshold
            );
        }
        if args.graph_pair_samples == 0 {
            bail!("graph_pair_samples must be > 0");
        }

        let init = BdhInitializationConfig {
            kind: args.initialization.into(),
            residual_scaling: BdhResidualScalingConfig {
                kind: args.residual_scaling.into(),
                gain: args.residual_gain,
            },
            neuron_gains: BdhNeuronGainConfig {
                kind: args.neuron_gains.into(),
                log_sigma: args.gain_log_sigma,
                max_gain: args.gain_max,
            },
            topology_prior: BdhTopologyPriorConfig {
                kind: args.topology_prior.into(),
                community_count: args.topology_community_count,
                bridge_fraction: args.topology_bridge_fraction,
                intra_community_gain: args.topology_intra_gain,
                inter_community_gain: args.topology_inter_gain,
                bridge_gain: args.topology_bridge_gain,
            },
            firing_targets: BdhFiringTargetConfig {
                kind: args.firing_targets.into(),
                x_target: args.x_target,
                y_target: args.y_target,
                x_threshold: args.x_threshold,
                y_threshold: args.y_threshold,
            },
            simple_normal_std: args.simple_normal_std,
        };
        init.validate()
            .map_err(|err| anyhow!("invalid initialization config: {err}"))?;
        let device = <Backend as BackendTrait>::Device::default();
        let eval_batches = build_eval_batches::<Backend>(
            args.batch_size.max(1),
            args.block_size.max(1),
            args.eval_batches.max(1),
            args.vocab_size,
            &device,
        );
        let calibration_kind = resolved_calibration_kind(&args);
        let (effective_init, calibration) =
            calibrate_initialization(calibration_kind, &args, &init, &eval_batches, &device)?;
        let model = build_model(&args, &effective_init, &device);
        let mut loss_state = model.init_state();
        let mut diag_state = model.init_state();
        let mut loss_sum = 0.0f64;
        let mut diagnostics = BTreeMap::<usize, DiagnosticsAccumulator>::new();

        for batch in &eval_batches {
            let logits = if args.carry_state {
                model.forward_with_state(batch.inputs.clone(), &mut loss_state)
            } else {
                model.forward(batch.inputs.clone())
            };
            loss_sum += scalar(language_model_loss(logits, batch.targets.clone()));

            let batch_diagnostics = if args.carry_state {
                model.collect_language_bdh_init_diagnostics_with_state(
                    batch.inputs.clone(),
                    &mut diag_state,
                )
            } else {
                model.collect_language_bdh_init_diagnostics(batch.inputs.clone())
            };
            accumulate_diagnostics(&mut diagnostics, &batch_diagnostics);
        }

        let layers = finalize_diagnostics(diagnostics);
        let passing_layers = layers
            .iter()
            .filter(|diag| layer_passes_phase0(diag))
            .map(|diag| diag.layer_index)
            .collect::<Vec<_>>();
        let backward = run_backward_check(&args, &effective_init)?;
        let perturbation =
            compute_perturbation_summary(&args, &effective_init, &eval_batches, &device)?;
        let (graph, update_intervals) =
            compute_latent_state_summaries(&args, &effective_init, &eval_batches, &device)?;
        let sparse_positive_pass = !passing_layers.is_empty();
        let backward_finite_pass = backward.enabled && backward.finite;
        let report = Report {
            benchmark: "burn_dragon_core synthetic BDH init probe",
            n_layer: args.n_layer,
            n_embd: args.n_embd,
            n_head: args.n_head,
            latent_total: args.latent_total,
            vocab_size: args.vocab_size,
            batch_size: args.batch_size.max(1),
            block_size: args.block_size.max(1),
            eval_batches: eval_batches.len(),
            seed: args.seed,
            carry_state: args.carry_state,
            initialization: effective_init.kind,
            residual_scaling: effective_init.residual_scaling.kind,
            neuron_gains: effective_init.neuron_gains.kind,
            topology_prior: effective_init.topology_prior.kind,
            firing_targets: effective_init.firing_targets.kind,
            avg_loss: loss_sum / eval_batches.len().max(1) as f64,
            layers,
            phase0: Phase0Criteria {
                p_x_band: [P_X_MIN, P_X_MAX],
                p_y_band: [P_Y_MIN, P_Y_MAX],
                r_res_band: [R_RES_MIN, R_RES_MAX],
                sparse_positive_pass,
                backward_finite_pass,
                phase0_pass: sparse_positive_pass && backward_finite_pass,
                passing_layers,
            },
            calibration,
            backward,
            perturbation,
            graph,
            update_intervals,
        };

        let markdown = format_markdown(&report);
        let json = serde_json::to_string_pretty(&report).context("serialize BDH init report")?;
        println!("{markdown}");

        if let Some(path) = args.markdown_path.as_ref() {
            write_text_artifact(path, &markdown, "markdown artifact")?;
        }
        if let Some(path) = args.json_path.as_ref() {
            write_text_artifact(path, &json, "json artifact")?;
        }

        Ok(())
    }

    fn build_eval_batches<B: BackendTrait>(
        batch_size: usize,
        block_size: usize,
        num_batches: usize,
        vocab_size: usize,
        device: &B::Device,
    ) -> Vec<EvalBatch<B>> {
        let usable_vocab = vocab_size.saturating_sub(1).max(1);
        let mut cursor = 0usize;
        let mut batches = Vec::with_capacity(num_batches);

        for _ in 0..num_batches {
            let mut inputs = vec![0i64; batch_size * block_size];
            let mut targets = vec![0i64; batch_size * block_size];
            for batch_idx in 0..batch_size {
                let start = cursor + batch_idx * block_size;
                for t in 0..block_size {
                    let idx = batch_idx * block_size + t;
                    inputs[idx] = ((start + t) % usable_vocab + 1) as i64;
                    targets[idx] = ((start + t + 1) % usable_vocab + 1) as i64;
                }
            }
            cursor = cursor.saturating_add(batch_size * block_size);
            let inputs = Tensor::<B, 2, Int>::from_data(
                TensorData::new(inputs, [batch_size, block_size]),
                device,
            );
            let targets = Tensor::<B, 2, Int>::from_data(
                TensorData::new(targets, [batch_size, block_size]),
                device,
            );
            batches.push(EvalBatch { inputs, targets });
        }

        batches
    }

    fn resolved_calibration_kind(args: &Args) -> CalibrationArg {
        if args.calibrate_firing_thresholds && matches!(args.calibration, CalibrationArg::Disabled)
        {
            CalibrationArg::FiringThresholds
        } else {
            args.calibration
        }
    }

    fn build_model_with_backend<B: BackendTrait>(
        args: &Args,
        initialization: &BdhInitializationConfig,
        device: &B::Device,
    ) -> BDH<B> {
        B::seed(device, args.seed);
        BDH::<B>::new(
            BDHConfig {
                n_layer: args.n_layer,
                n_embd: args.n_embd,
                n_head: args.n_head,
                mlp_internal_dim_multiplier: args.latent_total / args.n_embd,
                vocab_size: args.vocab_size,
                dropout: 0.0,
                initialization: initialization.clone(),
                ..Default::default()
            },
            device,
        )
    }

    fn build_model(
        args: &Args,
        initialization: &BdhInitializationConfig,
        device: &<Backend as BackendTrait>::Device,
    ) -> BDH<Backend> {
        build_model_with_backend::<Backend>(args, initialization, device)
    }

    fn calibrate_initialization(
        calibration: CalibrationArg,
        args: &Args,
        init: &BdhInitializationConfig,
        eval_batches: &[EvalBatch<Backend>],
        device: &<Backend as BackendTrait>::Device,
    ) -> Result<(BdhInitializationConfig, CalibrationSummary)> {
        match calibration {
            CalibrationArg::Disabled => Ok((init.clone(), CalibrationSummary::disabled())),
            CalibrationArg::FiringThresholds => {
                calibrate_firing_thresholds_only(args, init, eval_batches, device)
            }
            CalibrationArg::LsuvBdh => calibrate_lsuv_bdh(args, init, eval_batches, device),
        }
    }

    fn calibrate_firing_thresholds_only(
        args: &Args,
        init: &BdhInitializationConfig,
        eval_batches: &[EvalBatch<Backend>],
        device: &<Backend as BackendTrait>::Device,
    ) -> Result<(BdhInitializationConfig, CalibrationSummary)> {
        let calibration_batches = args
            .calibration_batches
            .max(1)
            .min(eval_batches.len().max(1));
        let calibration_slice = &eval_batches[..calibration_batches];
        let mut x_threshold = init.firing_targets.x_threshold;
        let mut y_threshold = init.firing_targets.y_threshold;

        for _ in 0..args.calibration_rounds.max(1) {
            x_threshold = calibrate_branch_threshold(
                args,
                init,
                calibration_slice,
                device,
                BranchCalibrationTarget::X,
                x_threshold,
                y_threshold,
                init.residual_scaling.gain,
            )?;
            y_threshold = calibrate_branch_threshold(
                args,
                init,
                calibration_slice,
                device,
                BranchCalibrationTarget::Y,
                x_threshold,
                y_threshold,
                init.residual_scaling.gain,
            )?;
        }

        let mut calibrated = init.clone();
        calibrated.firing_targets.kind = BdhFiringTargetKind::ExplicitThresholds;
        calibrated.firing_targets.x_threshold = x_threshold;
        calibrated.firing_targets.y_threshold = y_threshold;
        calibrated.validate().map_err(anyhow::Error::msg)?;

        let final_residual_gain = calibrated.residual_scaling.gain;
        let post_calibration =
            mean_probe_metrics_for_config(args, &calibrated, calibration_slice, device)?;

        Ok((
            calibrated,
            CalibrationSummary {
                enabled: true,
                kind: CalibrationArg::FiringThresholds,
                rounds: args.calibration_rounds.max(1),
                calibration_batches,
                target_r_res: None,
                final_x_threshold: Some(x_threshold),
                final_y_threshold: Some(y_threshold),
                final_residual_gain: Some(final_residual_gain),
                post_calibration: Some(post_calibration),
            },
        ))
    }

    fn calibrate_lsuv_bdh(
        args: &Args,
        init: &BdhInitializationConfig,
        eval_batches: &[EvalBatch<Backend>],
        device: &<Backend as BackendTrait>::Device,
    ) -> Result<(BdhInitializationConfig, CalibrationSummary)> {
        let calibration_batches = args
            .calibration_batches
            .max(1)
            .min(eval_batches.len().max(1));
        let calibration_slice = &eval_batches[..calibration_batches];
        let mut x_threshold = init.firing_targets.x_threshold;
        let mut y_threshold = init.firing_targets.y_threshold;
        let mut residual_gain = init.residual_scaling.gain.max(1.0e-6);
        let mut best_init =
            candidate_initialization(init, x_threshold, y_threshold, residual_gain)?;
        let mut best_metrics =
            mean_probe_metrics_for_config(args, &best_init, calibration_slice, device)?;
        let mut best_error = phase0_band_error(&best_metrics);

        for _ in 0..(args.calibration_rounds.max(1) + 1) {
            x_threshold = calibrate_branch_threshold(
                args,
                init,
                calibration_slice,
                device,
                BranchCalibrationTarget::X,
                x_threshold,
                y_threshold,
                residual_gain,
            )?;
            y_threshold = calibrate_branch_threshold(
                args,
                init,
                calibration_slice,
                device,
                BranchCalibrationTarget::Y,
                x_threshold,
                y_threshold,
                residual_gain,
            )?;
            residual_gain = calibrate_residual_gain(
                args,
                init,
                calibration_slice,
                device,
                x_threshold,
                y_threshold,
                residual_gain,
            )?;
            let candidate =
                candidate_initialization(init, x_threshold, y_threshold, residual_gain)?;
            let candidate_metrics =
                mean_probe_metrics_for_config(args, &candidate, calibration_slice, device)?;
            let candidate_error = phase0_band_error(&candidate_metrics);
            if candidate_error < best_error {
                best_error = candidate_error;
                best_init = candidate;
                best_metrics = candidate_metrics;
            }
            if metrics_pass_phase0_bands(&best_metrics) {
                break;
            }
        }

        Ok((
            best_init.clone(),
            CalibrationSummary {
                enabled: true,
                kind: CalibrationArg::LsuvBdh,
                rounds: args.calibration_rounds.max(1),
                calibration_batches,
                target_r_res: Some(args.target_r_res),
                final_x_threshold: Some(best_init.firing_targets.x_threshold),
                final_y_threshold: Some(best_init.firing_targets.y_threshold),
                final_residual_gain: Some(best_init.residual_scaling.gain),
                post_calibration: Some(best_metrics),
            },
        ))
    }

    fn candidate_initialization(
        init: &BdhInitializationConfig,
        x_threshold: f64,
        y_threshold: f64,
        residual_gain: f64,
    ) -> Result<BdhInitializationConfig> {
        let mut calibration_init = init.clone();
        calibration_init.firing_targets.kind = BdhFiringTargetKind::ExplicitThresholds;
        calibration_init.firing_targets.x_threshold = x_threshold;
        calibration_init.firing_targets.y_threshold = y_threshold;
        calibration_init.residual_scaling.gain = residual_gain;
        calibration_init.validate().map_err(anyhow::Error::msg)?;
        Ok(calibration_init)
    }

    fn calibrate_residual_gain(
        args: &Args,
        init: &BdhInitializationConfig,
        eval_batches: &[EvalBatch<Backend>],
        device: &<Backend as BackendTrait>::Device,
        x_threshold: f64,
        y_threshold: f64,
        residual_gain: f64,
    ) -> Result<f64> {
        let min_gain = 1.0e-6;
        let target = args.target_r_res;
        let current_metrics = mean_probe_metrics_for_candidate(
            args,
            init,
            eval_batches,
            device,
            x_threshold,
            y_threshold,
            residual_gain,
        )?;
        let current_r_res = current_metrics
            .r_res
            .ok_or_else(|| anyhow!("missing r_res during LSUV-BDH calibration"))?;

        let (mut low, mut high) = if current_r_res < target {
            let low = residual_gain.max(min_gain);
            let mut high = low;
            let mut high_r_res = current_r_res;
            while high_r_res < target && high < args.max_residual_gain {
                high = (high * 2.0).min(args.max_residual_gain);
                high_r_res = mean_probe_metrics_for_candidate(
                    args,
                    init,
                    eval_batches,
                    device,
                    x_threshold,
                    y_threshold,
                    high,
                )?
                .r_res
                .ok_or_else(|| anyhow!("missing r_res during LSUV-BDH calibration"))?;
                if (high - low).abs() <= f64::EPSILON {
                    break;
                }
            }
            if high_r_res < target {
                return Ok(high);
            }
            (low, high)
        } else {
            let mut high = residual_gain.max(min_gain);
            let mut low = (high * 0.5).max(min_gain);
            let mut low_r_res = mean_probe_metrics_for_candidate(
                args,
                init,
                eval_batches,
                device,
                x_threshold,
                y_threshold,
                low,
            )?
            .r_res
            .ok_or_else(|| anyhow!("missing r_res during LSUV-BDH calibration"))?;
            while low_r_res > target && low > min_gain {
                high = low;
                low = (low * 0.5).max(min_gain);
                low_r_res = mean_probe_metrics_for_candidate(
                    args,
                    init,
                    eval_batches,
                    device,
                    x_threshold,
                    y_threshold,
                    low,
                )?
                .r_res
                .ok_or_else(|| anyhow!("missing r_res during LSUV-BDH calibration"))?;
                if (high - low).abs() <= f64::EPSILON {
                    break;
                }
            }
            if low_r_res > target {
                return Ok(low);
            }
            (low, high)
        };

        for _ in 0..12 {
            let mid = (low * high).sqrt().max(min_gain);
            let mid_r_res = mean_probe_metrics_for_candidate(
                args,
                init,
                eval_batches,
                device,
                x_threshold,
                y_threshold,
                mid,
            )?
            .r_res
            .ok_or_else(|| anyhow!("missing r_res during LSUV-BDH calibration"))?;
            if mid_r_res < target {
                low = mid;
            } else {
                high = mid;
            }
        }

        Ok(high.max(min_gain))
    }

    #[derive(Clone, Copy)]
    enum BranchCalibrationTarget {
        X,
        Y,
    }

    impl BranchCalibrationTarget {
        fn target(self, init: &BdhInitializationConfig) -> f64 {
            match self {
                Self::X => init.firing_targets.x_target,
                Self::Y => init.firing_targets.y_target,
            }
        }

        fn metric(self, metrics: &ProbeMetricSummary) -> Result<f64> {
            match self {
                Self::X => metrics
                    .p_x
                    .ok_or_else(|| anyhow!("missing p_x during firing-target calibration")),
                Self::Y => metrics
                    .p_y
                    .ok_or_else(|| anyhow!("missing p_y during firing-target calibration")),
            }
        }
    }

    fn calibrate_branch_threshold(
        args: &Args,
        init: &BdhInitializationConfig,
        eval_batches: &[EvalBatch<Backend>],
        device: &<Backend as BackendTrait>::Device,
        branch: BranchCalibrationTarget,
        x_threshold: f64,
        y_threshold: f64,
        residual_gain: f64,
    ) -> Result<f64> {
        let target = branch.target(init);
        let zero_metrics = mean_probe_metrics_for_candidate(
            args,
            init,
            eval_batches,
            device,
            if matches!(branch, BranchCalibrationTarget::X) {
                0.0
            } else {
                x_threshold
            },
            if matches!(branch, BranchCalibrationTarget::Y) {
                0.0
            } else {
                y_threshold
            },
            residual_gain,
        )?;
        let zero_metric = branch.metric(&zero_metrics)?;
        if zero_metric <= target {
            return Ok(0.0);
        }

        let mut low = 0.0;
        let mut high = match branch {
            BranchCalibrationTarget::X => x_threshold.max(1.0e-4),
            BranchCalibrationTarget::Y => y_threshold.max(1.0e-4),
        };
        let mut high_metrics = mean_probe_metrics_for_candidate(
            args,
            init,
            eval_batches,
            device,
            if matches!(branch, BranchCalibrationTarget::X) {
                high
            } else {
                x_threshold
            },
            if matches!(branch, BranchCalibrationTarget::Y) {
                high
            } else {
                y_threshold
            },
            residual_gain,
        )?;
        let mut high_metric = branch.metric(&high_metrics)?;
        while high_metric > target && high < 1.0e3 {
            high *= 2.0;
            high_metrics = mean_probe_metrics_for_candidate(
                args,
                init,
                eval_batches,
                device,
                if matches!(branch, BranchCalibrationTarget::X) {
                    high
                } else {
                    x_threshold
                },
                if matches!(branch, BranchCalibrationTarget::Y) {
                    high
                } else {
                    y_threshold
                },
                residual_gain,
            )?;
            high_metric = branch.metric(&high_metrics)?;
        }

        for _ in 0..12 {
            let mid = 0.5 * (low + high);
            let metrics = mean_probe_metrics_for_candidate(
                args,
                init,
                eval_batches,
                device,
                if matches!(branch, BranchCalibrationTarget::X) {
                    mid
                } else {
                    x_threshold
                },
                if matches!(branch, BranchCalibrationTarget::Y) {
                    mid
                } else {
                    y_threshold
                },
                residual_gain,
            )?;
            let metric = branch.metric(&metrics)?;
            if metric > target {
                low = mid;
            } else {
                high = mid;
            }
        }

        Ok(high)
    }

    fn mean_probe_metrics_for_candidate(
        args: &Args,
        init: &BdhInitializationConfig,
        eval_batches: &[EvalBatch<Backend>],
        device: &<Backend as BackendTrait>::Device,
        x_threshold: f64,
        y_threshold: f64,
        residual_gain: f64,
    ) -> Result<ProbeMetricSummary> {
        let calibration_init =
            candidate_initialization(init, x_threshold, y_threshold, residual_gain)?;
        mean_probe_metrics_for_config(args, &calibration_init, eval_batches, device)
    }

    fn mean_probe_metrics_for_config(
        args: &Args,
        init: &BdhInitializationConfig,
        eval_batches: &[EvalBatch<Backend>],
        device: &<Backend as BackendTrait>::Device,
    ) -> Result<ProbeMetricSummary> {
        let model = build_model(args, init, device);
        let mut state = model.init_state();
        let mut accumulators = BTreeMap::<usize, DiagnosticsAccumulator>::new();

        for batch in eval_batches {
            let diagnostics = if args.carry_state {
                model.collect_language_bdh_init_diagnostics_with_state(
                    batch.inputs.clone(),
                    &mut state,
                )
            } else {
                model.collect_language_bdh_init_diagnostics(batch.inputs.clone())
            };
            accumulate_diagnostics(&mut accumulators, &diagnostics);
        }

        Ok(summarize_probe_metrics(&finalize_diagnostics(accumulators)))
    }

    fn summarize_probe_metrics(layers: &[LanguageBdhInitLayerDiagnostics]) -> ProbeMetricSummary {
        ProbeMetricSummary {
            finite: !layers.is_empty() && layers.iter().all(|layer| layer.finite),
            p_x: mean_optional(
                layers
                    .iter()
                    .filter_map(|layer| layer.p_x)
                    .collect::<Vec<_>>()
                    .as_slice(),
            ),
            p_y: mean_optional(
                layers
                    .iter()
                    .filter_map(|layer| layer.p_y)
                    .collect::<Vec<_>>()
                    .as_slice(),
            ),
            r_res: mean_optional(
                layers
                    .iter()
                    .filter_map(|layer| layer.r_res)
                    .collect::<Vec<_>>()
                    .as_slice(),
            ),
            recurrent_readout_ratio: mean_optional(
                layers
                    .iter()
                    .filter_map(|layer| layer.recurrent_readout_ratio)
                    .collect::<Vec<_>>()
                    .as_slice(),
            ),
        }
    }

    fn mean_optional(values: &[f64]) -> Option<f64> {
        (!values.is_empty()).then_some(values.iter().copied().sum::<f64>() / values.len() as f64)
    }

    fn metric_band_error(value: Option<f64>, min: f64, max: f64) -> f64 {
        match value {
            Some(value) if value < min => min - value,
            Some(value) if value > max => value - max,
            Some(_) => 0.0,
            None => 1.0,
        }
    }

    fn phase0_band_error(metrics: &ProbeMetricSummary) -> f64 {
        metric_band_error(metrics.p_x, P_X_MIN, P_X_MAX)
            + metric_band_error(metrics.p_y, P_Y_MIN, P_Y_MAX)
            + metric_band_error(metrics.r_res, R_RES_MIN, R_RES_MAX)
            + if metrics.finite { 0.0 } else { 1.0 }
    }

    fn metrics_pass_phase0_bands(metrics: &ProbeMetricSummary) -> bool {
        metrics.finite
            && metrics
                .p_x
                .is_some_and(|value| (P_X_MIN..=P_X_MAX).contains(&value))
            && metrics
                .p_y
                .is_some_and(|value| (P_Y_MIN..=P_Y_MAX).contains(&value))
            && metrics
                .r_res
                .is_some_and(|value| (R_RES_MIN..=R_RES_MAX).contains(&value))
    }

    fn run_backward_check(
        args: &Args,
        init: &BdhInitializationConfig,
    ) -> Result<BackwardCheckSummary> {
        if args.backward_steps == 0 {
            return Ok(BackwardCheckSummary::disabled());
        }

        let device = <TrainBackend as BackendTrait>::Device::default();
        let train_batches = build_eval_batches::<TrainBackend>(
            args.batch_size.max(1),
            args.block_size.max(1),
            args.eval_batches.max(1),
            args.vocab_size,
            &device,
        );
        let mut model = build_model_with_backend::<TrainBackend>(args, init, &device);
        let mut state = model.init_state();
        let mut optimizer = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TrainBackend, BDH<TrainBackend>>();
        let lr: LearningRate = args.backward_learning_rate;

        let mut min_loss = f64::INFINITY;
        let mut max_loss = f64::NEG_INFINITY;
        let mut final_loss = None;
        let mut finite = true;
        let mut losses = Vec::with_capacity(args.backward_steps);

        for step in 0..args.backward_steps {
            let batch = &train_batches[step % train_batches.len().max(1)];
            let logits = if args.carry_state {
                model.forward_with_state(batch.inputs.clone(), &mut state)
            } else {
                model.forward(batch.inputs.clone())
            };
            let loss = language_model_loss(logits, batch.targets.clone());
            let loss_value = scalar(loss.clone());
            final_loss = Some(loss_value);
            if !loss_value.is_finite() {
                finite = false;
                break;
            }
            losses.push(loss_value);
            min_loss = min_loss.min(loss_value);
            max_loss = max_loss.max(loss_value);

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(lr, model, grads);
            if args.carry_state {
                state.detach_in_place();
            }
        }

        Ok(BackwardCheckSummary {
            enabled: true,
            steps: args.backward_steps,
            completed_steps: losses.len(),
            tokens_per_step: args.batch_size.max(1) * args.block_size.max(1),
            learning_rate: args.backward_learning_rate,
            finite,
            aulc: (!losses.is_empty()).then_some(losses.iter().copied().sum()),
            losses,
            min_loss: (min_loss.is_finite()).then_some(min_loss),
            max_loss: (max_loss.is_finite()).then_some(max_loss),
            final_loss,
        })
    }

    fn compute_perturbation_summary(
        args: &Args,
        init: &BdhInitializationConfig,
        eval_batches: &[EvalBatch<Backend>],
        device: &<Backend as BackendTrait>::Device,
    ) -> Result<PerturbationSummary> {
        if args.perturbation_epsilon == 0.0 {
            return Ok(PerturbationSummary::disabled(args.perturbation_epsilon));
        }

        #[derive(Clone, Default)]
        struct Accumulator {
            samples: usize,
            input_sum: f64,
            output_sum: f64,
            gain_sum: f64,
            gain_count: usize,
        }

        let model = build_model(args, init, device);
        let mut accumulators = vec![Accumulator::default(); args.n_layer];

        for batch in eval_batches {
            let embedded = model.embed_tokens(batch.inputs.clone());
            let perturbed = embedded.clone().add(make_deterministic_noise_like(
                embedded.shape().dims(),
                args.perturbation_epsilon,
                args.seed,
                &embedded.device(),
            ));

            let mut input_delta = tensor_diff_rms(
                model.forward_hidden_prefix_layers_from_embedded_for_profile(
                    embedded.clone(),
                    0,
                    None,
                ),
                model.forward_hidden_prefix_layers_from_embedded_for_profile(
                    perturbed.clone(),
                    0,
                    None,
                ),
            )?;

            for layer_limit in 1..=args.n_layer {
                let clean_hidden = model.forward_hidden_prefix_layers_from_embedded_for_profile(
                    embedded.clone(),
                    layer_limit,
                    None,
                );
                let perturbed_hidden = model
                    .forward_hidden_prefix_layers_from_embedded_for_profile(
                        perturbed.clone(),
                        layer_limit,
                        None,
                    );
                let output_delta = tensor_diff_rms(clean_hidden, perturbed_hidden)?;
                let accumulator = &mut accumulators[layer_limit - 1];
                accumulator.samples += 1;
                accumulator.input_sum += input_delta;
                accumulator.output_sum += output_delta;
                if input_delta > 0.0 {
                    accumulator.gain_sum += output_delta / input_delta.max(1.0e-12);
                    accumulator.gain_count += 1;
                }
                input_delta = output_delta;
            }
        }

        let layers = accumulators
            .into_iter()
            .enumerate()
            .map(|(layer_index, accumulator)| PerturbationLayerSummary {
                layer_index,
                samples: accumulator.samples,
                input_delta_rms: (accumulator.samples > 0)
                    .then_some(accumulator.input_sum / accumulator.samples as f64),
                output_delta_rms: (accumulator.samples > 0)
                    .then_some(accumulator.output_sum / accumulator.samples as f64),
                gain: (accumulator.gain_count > 0)
                    .then_some(accumulator.gain_sum / accumulator.gain_count as f64),
            })
            .collect::<Vec<_>>();
        let mean_gain = mean_optional(
            layers
                .iter()
                .filter_map(|layer| layer.gain)
                .collect::<Vec<_>>()
                .as_slice(),
        );

        Ok(PerturbationSummary {
            enabled: true,
            epsilon: args.perturbation_epsilon,
            layers,
            mean_gain,
        })
    }

    fn compute_latent_state_summaries(
        args: &Args,
        init: &BdhInitializationConfig,
        eval_batches: &[EvalBatch<Backend>],
        device: &<Backend as BackendTrait>::Device,
    ) -> Result<(GraphSummary, UpdateIntervalSummary)> {
        let model = build_model(args, init, device);
        let mut state = model.init_state();
        let mut layers = BTreeMap::<usize, Vec<LatentActivityBatch>>::new();

        for batch in eval_batches {
            if !args.carry_state {
                state = model.init_state();
            }
            let embedded = model.embed_tokens(batch.inputs.clone());
            let _ = model.forward_with_state_embedded(embedded, &mut state);
            for (layer_index, viz) in state.take_viz().into_iter().enumerate() {
                if let Some(viz) = viz {
                    layers
                        .entry(layer_index)
                        .or_default()
                        .push(latent_activity_batch_from_viz(
                            viz,
                            args.latent_activity_threshold,
                        )?);
                }
            }
        }

        let community_count = match init.topology_prior.kind {
            BdhTopologyPriorKind::ModularBridges => init.topology_prior.community_count.max(1),
            BdhTopologyPriorKind::Iid => init.topology_prior.community_count.max(1),
        };
        let graph = summarize_graph_layers(
            &layers,
            args.latent_activity_threshold,
            args.graph_pair_samples,
            community_count,
            args.seed,
        );
        let update_intervals =
            summarize_update_intervals(&layers, args.latent_activity_threshold, args.carry_state);

        Ok((graph, update_intervals))
    }

    fn latent_activity_batch_from_viz(
        viz: LayerVizState<Backend>,
        activation_threshold: f64,
    ) -> Result<LatentActivityBatch> {
        let [heads, latent] = viz.y_neuron_last.shape().dims::<2>();
        let values = tensor_values_f32(viz.y_neuron_last)?;
        let active_by_head = values
            .into_iter()
            .map(|value| u8::from((value as f64) > activation_threshold))
            .collect::<Vec<_>>();
        Ok(LatentActivityBatch {
            heads,
            latent,
            active_by_head,
        })
    }

    fn summarize_graph_layers(
        layers: &BTreeMap<usize, Vec<LatentActivityBatch>>,
        activation_threshold: f64,
        pair_samples: usize,
        community_count: usize,
        seed: u64,
    ) -> GraphSummary {
        if layers.is_empty() {
            return GraphSummary::disabled(activation_threshold, pair_samples, community_count);
        }

        let layer_summaries = layers
            .iter()
            .map(|(&layer_index, batches)| {
                let latent = batches.first().map(|batch| batch.latent).unwrap_or(0);
                let sample_count = batches.iter().map(|batch| batch.heads).sum::<usize>();
                if latent == 0 || sample_count == 0 {
                    return GraphLayerSummary {
                        layer_index,
                        samples: sample_count,
                        pair_samples,
                        active_rate_mean: None,
                        sampled_edge_density: None,
                        degree_tail_ratio: None,
                        community_concentration: None,
                        within_pair_coactivation: None,
                        across_pair_coactivation: None,
                        modularity_gap: None,
                    };
                }

                let mut active_counts = vec![0usize; latent];
                let mut concentration_sum = 0.0f64;
                let mut concentration_count = 0usize;
                for batch in batches {
                    for head in 0..batch.heads {
                        let row = &batch.active_by_head[head * latent..(head + 1) * latent];
                        let mut total_active = 0usize;
                        let mut community_active = vec![0usize; community_count.max(1)];
                        for (latent_idx, active) in row.iter().enumerate() {
                            if *active == 0 {
                                continue;
                            }
                            active_counts[latent_idx] += 1;
                            total_active += 1;
                            let community = latent_idx * community_count.max(1) / latent.max(1);
                            community_active[community.min(community_count.max(1) - 1)] += 1;
                        }
                        if total_active > 0 {
                            concentration_sum +=
                                1.0 - normalized_entropy(&community_active, total_active);
                            concentration_count += 1;
                        }
                    }
                }

                let active_rate_mean = Some(
                    active_counts.iter().copied().sum::<usize>() as f64
                        / (latent * sample_count) as f64,
                );
                let degree_tail_ratio = percentile_ratio(
                    &active_counts
                        .iter()
                        .map(|count| *count as f64 / sample_count as f64)
                        .collect::<Vec<_>>(),
                    0.95,
                );
                let community_concentration = (concentration_count > 0)
                    .then_some(concentration_sum / concentration_count as f64);
                let sampled = sample_pair_coactivations(
                    batches,
                    community_count.max(1),
                    pair_samples,
                    seed ^ layer_index as u64,
                );

                GraphLayerSummary {
                    layer_index,
                    samples: sample_count,
                    pair_samples,
                    active_rate_mean,
                    sampled_edge_density: sampled.edge_density,
                    degree_tail_ratio,
                    community_concentration,
                    within_pair_coactivation: sampled.within_mean,
                    across_pair_coactivation: sampled.across_mean,
                    modularity_gap: match (sampled.within_mean, sampled.across_mean) {
                        (Some(within), Some(across)) => Some(within - across),
                        _ => None,
                    },
                }
            })
            .collect::<Vec<_>>();

        GraphSummary {
            enabled: true,
            signal: "y_neuron_last",
            activation_threshold,
            pair_samples,
            community_count,
            layers: layer_summaries,
        }
    }

    fn summarize_update_intervals(
        layers: &BTreeMap<usize, Vec<LatentActivityBatch>>,
        activation_threshold: f64,
        carry_state: bool,
    ) -> UpdateIntervalSummary {
        if !carry_state || layers.is_empty() {
            return UpdateIntervalSummary::disabled(activation_threshold);
        }

        let layer_summaries = layers
            .iter()
            .map(|(&layer_index, batches)| {
                let latent = batches.first().map(|batch| batch.latent).unwrap_or(0);
                let sample_count = batches.len();
                if latent == 0 || sample_count == 0 {
                    return UpdateIntervalLayerSummary {
                        layer_index,
                        samples: sample_count,
                        interval_count: 0,
                        active_rate_mean: None,
                        mean_interval: None,
                        max_interval: None,
                        tail_ratio: None,
                    };
                }

                let mut active_counts = vec![0usize; latent];
                let mut last_active = vec![None; latent];
                let mut intervals = Vec::new();

                for (batch_index, batch) in batches.iter().enumerate() {
                    for latent_idx in 0..latent {
                        let active = (0..batch.heads)
                            .any(|head| batch.active_by_head[head * latent + latent_idx] != 0);
                        if active {
                            active_counts[latent_idx] += 1;
                            if let Some(last_idx) = last_active[latent_idx] {
                                intervals.push(batch_index - last_idx);
                            }
                            last_active[latent_idx] = Some(batch_index);
                        }
                    }
                }

                let interval_values = intervals
                    .iter()
                    .map(|interval| *interval as f64)
                    .collect::<Vec<_>>();
                UpdateIntervalLayerSummary {
                    layer_index,
                    samples: sample_count,
                    interval_count: intervals.len(),
                    active_rate_mean: Some(
                        active_counts.iter().copied().sum::<usize>() as f64
                            / (latent * sample_count) as f64,
                    ),
                    mean_interval: mean_optional(&interval_values),
                    max_interval: intervals.iter().copied().max(),
                    tail_ratio: percentile_ratio(&interval_values, 0.90),
                }
            })
            .collect::<Vec<_>>();

        UpdateIntervalSummary {
            enabled: true,
            signal: "y_neuron_last",
            activation_threshold,
            layers: layer_summaries,
        }
    }

    struct PairCoactivationSummary {
        edge_density: Option<f64>,
        within_mean: Option<f64>,
        across_mean: Option<f64>,
    }

    fn sample_pair_coactivations(
        batches: &[LatentActivityBatch],
        community_count: usize,
        pair_samples: usize,
        seed: u64,
    ) -> PairCoactivationSummary {
        let latent = match batches.first() {
            Some(batch) => batch.latent,
            None => {
                return PairCoactivationSummary {
                    edge_density: None,
                    within_mean: None,
                    across_mean: None,
                };
            }
        };
        if latent < 2 {
            return PairCoactivationSummary {
                edge_density: None,
                within_mean: None,
                across_mean: None,
            };
        }

        let sample_count = batches
            .iter()
            .map(|batch| batch.heads)
            .sum::<usize>()
            .max(1);
        let mut generator = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut edge_hits = 0usize;
        let mut within_sum = 0.0f64;
        let mut within_count = 0usize;
        let mut across_sum = 0.0f64;
        let mut across_count = 0usize;

        for sample_idx in 0..pair_samples {
            let want_within = sample_idx % 2 == 0;
            let (left, right) =
                sample_latent_pair(latent, community_count.max(1), want_within, &mut generator);
            let mut coactive = 0usize;
            for batch in batches {
                for head in 0..batch.heads {
                    let row = &batch.active_by_head[head * latent..(head + 1) * latent];
                    if row[left] != 0 && row[right] != 0 {
                        coactive += 1;
                    }
                }
            }
            let rate = coactive as f64 / sample_count as f64;
            edge_hits += usize::from(coactive > 0);
            let same_community =
                left * community_count.max(1) / latent == right * community_count.max(1) / latent;
            if same_community {
                within_sum += rate;
                within_count += 1;
            } else {
                across_sum += rate;
                across_count += 1;
            }
        }

        PairCoactivationSummary {
            edge_density: Some(edge_hits as f64 / pair_samples as f64),
            within_mean: (within_count > 0).then_some(within_sum / within_count as f64),
            across_mean: (across_count > 0).then_some(across_sum / across_count as f64),
        }
    }

    fn sample_latent_pair(
        latent: usize,
        community_count: usize,
        want_within: bool,
        generator: &mut u64,
    ) -> (usize, usize) {
        let communities = community_count.max(1);
        for _ in 0..32 {
            let left = next_index(generator, latent);
            let mut right = next_index(generator, latent.saturating_sub(1).max(1));
            if right >= left {
                right += 1;
            }
            let same_community = left * communities / latent == right * communities / latent;
            if same_community == want_within {
                return (left.min(right), left.max(right));
            }
        }
        let left = next_index(generator, latent);
        let mut right = next_index(generator, latent.saturating_sub(1).max(1));
        if right >= left {
            right += 1;
        }
        (left.min(right), left.max(right))
    }

    fn next_index(generator: &mut u64, upper: usize) -> usize {
        *generator = generator
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*generator as usize) % upper.max(1)
    }

    fn normalized_entropy(counts: &[usize], total: usize) -> f64 {
        if total == 0 || counts.len() <= 1 {
            return 0.0;
        }
        let total_f = total as f64;
        let entropy = counts
            .iter()
            .filter(|count| **count > 0)
            .map(|count| {
                let probability = *count as f64 / total_f;
                -probability * probability.ln()
            })
            .sum::<f64>();
        entropy / (counts.len() as f64).ln().max(1.0e-12)
    }

    fn percentile_ratio(values: &[f64], upper_quantile: f64) -> Option<f64> {
        let mut sorted = values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        if sorted.is_empty() {
            return None;
        }
        sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
        let upper = quantile_sorted(&sorted, upper_quantile)?;
        let median = quantile_sorted(&sorted, 0.5)?.max(1.0e-12);
        Some(upper / median)
    }

    fn quantile_sorted(sorted: &[f64], quantile: f64) -> Option<f64> {
        if sorted.is_empty() {
            return None;
        }
        let q = quantile.clamp(0.0, 1.0);
        let index = ((sorted.len() - 1) as f64 * q).round() as usize;
        sorted.get(index).copied()
    }

    fn make_deterministic_noise_like<B: BackendTrait>(
        shape: [usize; 3],
        epsilon: f64,
        seed: u64,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let count = shape.into_iter().product::<usize>();
        let mut generator = seed ^ 0x9E37_79B9_7F4A_7C15;
        let values = (0..count)
            .map(|_| {
                generator = generator
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = ((generator >> 40) as f32) / ((1u64 << 24) as f32);
                (2.0 * unit - 1.0) * epsilon as f32
            })
            .collect::<Vec<_>>();
        Tensor::<B, 3>::from_data(TensorData::new(values, shape), device)
    }

    fn tensor_diff_rms<const D: usize, B: BackendTrait>(
        left: Tensor<B, D>,
        right: Tensor<B, D>,
    ) -> Result<f64> {
        let left_values = tensor_values_f32(left)?;
        let right_values = tensor_values_f32(right)?;
        if left_values.len() != right_values.len() {
            bail!(
                "tensor_diff_rms length mismatch: {} vs {}",
                left_values.len(),
                right_values.len()
            );
        }
        let mean_square = left_values
            .iter()
            .zip(right_values.iter())
            .map(|(left, right)| {
                let delta = *left as f64 - *right as f64;
                delta * delta
            })
            .sum::<f64>()
            / left_values.len().max(1) as f64;
        Ok(mean_square.sqrt())
    }

    fn tensor_values_f32<const D: usize, B: BackendTrait>(
        tensor: Tensor<B, D>,
    ) -> Result<Vec<f32>> {
        tensor
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(|err| anyhow!("failed to extract tensor values: {err}"))
    }

    fn accumulate_diagnostics(
        accumulators: &mut BTreeMap<usize, DiagnosticsAccumulator>,
        diagnostics: &[LanguageBdhInitLayerDiagnostics],
    ) {
        for diag in diagnostics {
            let accumulator = accumulators.entry(diag.layer_index).or_default();
            accumulator.count += 1;
            accumulator.lowrank_active_count += usize::from(diag.lowrank_path_active);
            accumulator.finite_count += usize::from(diag.finite);
            if let Some(value) = diag.p_x {
                accumulator.p_x_sum += value;
                accumulator.p_x_count += 1;
            }
            if let Some(value) = diag.p_y {
                accumulator.p_y_sum += value;
                accumulator.p_y_count += 1;
            }
            if let Some(value) = diag.current_rms {
                accumulator.current_rms_sum += value;
                accumulator.current_rms_count += 1;
            }
            if let Some(value) = diag.recurrent_readout_rms {
                accumulator.recurrent_readout_rms_sum += value;
                accumulator.recurrent_readout_rms_count += 1;
            }
            if let Some(value) = diag.recurrent_readout_ratio {
                accumulator.recurrent_readout_ratio_sum += value;
                accumulator.recurrent_readout_ratio_count += 1;
            }
            if let Some(value) = diag.residual_delta_rms {
                accumulator.residual_delta_rms_sum += value;
                accumulator.residual_delta_rms_count += 1;
            }
            if let Some(value) = diag.r_res {
                accumulator.r_res_sum += value;
                accumulator.r_res_count += 1;
            }
        }
    }

    fn finalize_diagnostics(
        accumulators: BTreeMap<usize, DiagnosticsAccumulator>,
    ) -> Vec<LanguageBdhInitLayerDiagnostics> {
        accumulators
            .into_iter()
            .map(
                |(layer_index, accumulator)| LanguageBdhInitLayerDiagnostics {
                    layer_index,
                    lowrank_path_active: accumulator.lowrank_active_count * 2 >= accumulator.count,
                    finite: accumulator.finite_count == accumulator.count,
                    p_x: (accumulator.p_x_count > 0)
                        .then_some(accumulator.p_x_sum / accumulator.p_x_count as f64),
                    p_y: (accumulator.p_y_count > 0)
                        .then_some(accumulator.p_y_sum / accumulator.p_y_count as f64),
                    current_rms: (accumulator.current_rms_count > 0).then_some(
                        accumulator.current_rms_sum / accumulator.current_rms_count as f64,
                    ),
                    recurrent_readout_rms: (accumulator.recurrent_readout_rms_count > 0).then_some(
                        accumulator.recurrent_readout_rms_sum
                            / accumulator.recurrent_readout_rms_count as f64,
                    ),
                    recurrent_readout_ratio: (accumulator.recurrent_readout_ratio_count > 0)
                        .then_some(
                            accumulator.recurrent_readout_ratio_sum
                                / accumulator.recurrent_readout_ratio_count as f64,
                        ),
                    residual_delta_rms: (accumulator.residual_delta_rms_count > 0).then_some(
                        accumulator.residual_delta_rms_sum
                            / accumulator.residual_delta_rms_count as f64,
                    ),
                    r_res: (accumulator.r_res_count > 0)
                        .then_some(accumulator.r_res_sum / accumulator.r_res_count as f64),
                },
            )
            .collect()
    }

    fn layer_passes_phase0(diag: &LanguageBdhInitLayerDiagnostics) -> bool {
        diag.lowrank_path_active
            && diag.finite
            && diag
                .p_x
                .is_some_and(|value| (P_X_MIN..=P_X_MAX).contains(&value))
            && diag
                .p_y
                .is_some_and(|value| (P_Y_MIN..=P_Y_MAX).contains(&value))
            && diag
                .r_res
                .is_some_and(|value| (R_RES_MIN..=R_RES_MAX).contains(&value))
    }

    fn language_model_loss<B: BackendTrait>(
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1> {
        let [batch, time, vocab] = logits.shape().dims();
        let logits_flat = logits.reshape([batch * time, vocab]);
        let targets_flat = targets.reshape([batch * time]);
        let device = logits_flat.device();
        CrossEntropyLossConfig::new()
            .init::<B>(&device)
            .forward(logits_flat, targets_flat)
    }

    fn scalar<B: BackendTrait>(tensor: Tensor<B, 1>) -> f64 {
        tensor.to_data().to_vec::<f32>().expect("scalar value")[0] as f64
    }

    fn format_markdown(report: &Report) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "# BDH Init Probe");
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "- shape: layers={}, embd={}, heads={}, latent_total={}",
            report.n_layer, report.n_embd, report.n_head, report.latent_total
        );
        let _ = writeln!(out, "- vocab size: {}", report.vocab_size);
        let _ = writeln!(out, "- batch size: {}", report.batch_size);
        let _ = writeln!(out, "- block size: {}", report.block_size);
        let _ = writeln!(out, "- eval batches: {}", report.eval_batches);
        let _ = writeln!(out, "- seed: {}", report.seed);
        let _ = writeln!(out, "- carry state: {}", report.carry_state);
        let _ = writeln!(out, "- initialization: {:?}", report.initialization);
        let _ = writeln!(out, "- residual scaling: {:?}", report.residual_scaling);
        let _ = writeln!(out, "- neuron gains: {:?}", report.neuron_gains);
        let _ = writeln!(out, "- topology prior: {:?}", report.topology_prior);
        let _ = writeln!(out, "- firing targets: {:?}", report.firing_targets);
        if report.calibration.enabled {
            let _ = writeln!(
                out,
                "- calibration: kind={:?}, rounds={}, batches={}, x_threshold={:.6}, y_threshold={:.6}, residual_gain={:.6}",
                report.calibration.kind,
                report.calibration.rounds,
                report.calibration.calibration_batches,
                report.calibration.final_x_threshold.unwrap_or_default(),
                report.calibration.final_y_threshold.unwrap_or_default(),
                report.calibration.final_residual_gain.unwrap_or_default(),
            );
            if let Some(target_r_res) = report.calibration.target_r_res {
                let _ = writeln!(out, "- calibration target r_res: {:.6}", target_r_res);
            }
            if let Some(post) = report.calibration.post_calibration {
                let _ = writeln!(
                    out,
                    "- post-calibration metrics: finite={}, p_x={:.6}, p_y={:.6}, r_res={:.6}, readout_ratio={:.6}",
                    post.finite,
                    post.p_x.unwrap_or_default(),
                    post.p_y.unwrap_or_default(),
                    post.r_res.unwrap_or_default(),
                    post.recurrent_readout_ratio.unwrap_or_default(),
                );
            }
        }
        let _ = writeln!(out, "- avg loss: {:.6}", report.avg_loss);
        let _ = writeln!(
            out,
            "- phase0 sparse-positive pass: {}",
            report.phase0.sparse_positive_pass
        );
        let _ = writeln!(
            out,
            "- phase0 backward finite pass: {}",
            report.phase0.backward_finite_pass
        );
        let _ = writeln!(out, "- phase0 pass: {}", report.phase0.phase0_pass);
        let _ = writeln!(
            out,
            "- criteria: p_x in [{:.2}, {:.2}], p_y in [{:.2}, {:.2}], r_res in [{:.2}, {:.2}]",
            report.phase0.p_x_band[0],
            report.phase0.p_x_band[1],
            report.phase0.p_y_band[0],
            report.phase0.p_y_band[1],
            report.phase0.r_res_band[0],
            report.phase0.r_res_band[1],
        );
        if report.backward.enabled {
            let _ = writeln!(
                out,
                "- backward check: steps={}, completed_steps={}, tokens_per_step={}, lr={:.6}, finite={}, aulc={:.6}, min_loss={:.6}, max_loss={:.6}, final_loss={:.6}",
                report.backward.steps,
                report.backward.completed_steps,
                report.backward.tokens_per_step,
                report.backward.learning_rate,
                report.backward.finite,
                report.backward.aulc.unwrap_or_default(),
                report.backward.min_loss.unwrap_or_default(),
                report.backward.max_loss.unwrap_or_default(),
                report.backward.final_loss.unwrap_or_default(),
            );
        } else {
            let _ = writeln!(out, "- backward check: disabled");
        }
        if report.perturbation.enabled {
            let _ = writeln!(
                out,
                "- perturbation: epsilon={:.6}, mean_gain={:.6}",
                report.perturbation.epsilon,
                report.perturbation.mean_gain.unwrap_or_default(),
            );
        } else {
            let _ = writeln!(out, "- perturbation: disabled");
        }
        if report.graph.enabled {
            let _ = writeln!(
                out,
                "- latent graph: signal={}, threshold={:.6}, pair_samples={}, community_count={}",
                report.graph.signal,
                report.graph.activation_threshold,
                report.graph.pair_samples,
                report.graph.community_count,
            );
        } else {
            let _ = writeln!(out, "- latent graph: disabled");
        }
        if report.update_intervals.enabled {
            let _ = writeln!(
                out,
                "- update intervals: signal={}, threshold={:.6}",
                report.update_intervals.signal, report.update_intervals.activation_threshold,
            );
        } else {
            let _ = writeln!(out, "- update intervals: disabled");
        }
        if !report.phase0.passing_layers.is_empty() {
            let _ = writeln!(
                out,
                "- passing layers: {}",
                report
                    .phase0
                    .passing_layers
                    .iter()
                    .map(|layer| layer.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "| layer | active | finite | p_x | p_y | current rms | readout rms | readout ratio | residual rms | r_res |"
        );
        let _ = writeln!(out, "|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|");
        for layer in &report.layers {
            let p_x = layer
                .p_x
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "-".to_string());
            let p_y = layer
                .p_y
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "-".to_string());
            let current_rms = layer
                .current_rms
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "-".to_string());
            let readout_rms = layer
                .recurrent_readout_rms
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "-".to_string());
            let readout_ratio = layer
                .recurrent_readout_ratio
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "-".to_string());
            let residual_rms = layer
                .residual_delta_rms
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "-".to_string());
            let r_res = layer
                .r_res
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "-".to_string());
            let _ = writeln!(
                out,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                layer.layer_index,
                layer.lowrank_path_active,
                layer.finite,
                p_x,
                p_y,
                current_rms,
                readout_rms,
                readout_ratio,
                residual_rms,
                r_res,
            );
        }
        if report.perturbation.enabled {
            let _ = writeln!(out);
            let _ = writeln!(
                out,
                "| perturb layer | samples | input delta rms | output delta rms | gain |"
            );
            let _ = writeln!(out, "|---:|---:|---:|---:|---:|");
            for layer in &report.perturbation.layers {
                let _ = writeln!(
                    out,
                    "| {} | {} | {} | {} | {} |",
                    layer.layer_index,
                    layer.samples,
                    layer
                        .input_delta_rms
                        .map(|value| format!("{value:.6}"))
                        .unwrap_or_else(|| "-".to_string()),
                    layer
                        .output_delta_rms
                        .map(|value| format!("{value:.6}"))
                        .unwrap_or_else(|| "-".to_string()),
                    layer
                        .gain
                        .map(|value| format!("{value:.6}"))
                        .unwrap_or_else(|| "-".to_string()),
                );
            }
        }
        if report.graph.enabled {
            let _ = writeln!(out);
            let _ = writeln!(
                out,
                "| graph layer | samples | active rate | edge density | degree tail | concentration | within coact | across coact | gap |"
            );
            let _ = writeln!(out, "|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
            for layer in &report.graph.layers {
                let _ = writeln!(
                    out,
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                    layer.layer_index,
                    layer.samples,
                    layer
                        .active_rate_mean
                        .map(|value| format!("{value:.6}"))
                        .unwrap_or_else(|| "-".to_string()),
                    layer
                        .sampled_edge_density
                        .map(|value| format!("{value:.6}"))
                        .unwrap_or_else(|| "-".to_string()),
                    layer
                        .degree_tail_ratio
                        .map(|value| format!("{value:.6}"))
                        .unwrap_or_else(|| "-".to_string()),
                    layer
                        .community_concentration
                        .map(|value| format!("{value:.6}"))
                        .unwrap_or_else(|| "-".to_string()),
                    layer
                        .within_pair_coactivation
                        .map(|value| format!("{value:.6}"))
                        .unwrap_or_else(|| "-".to_string()),
                    layer
                        .across_pair_coactivation
                        .map(|value| format!("{value:.6}"))
                        .unwrap_or_else(|| "-".to_string()),
                    layer
                        .modularity_gap
                        .map(|value| format!("{value:.6}"))
                        .unwrap_or_else(|| "-".to_string()),
                );
            }
        }
        if report.update_intervals.enabled {
            let _ = writeln!(out);
            let _ = writeln!(
                out,
                "| interval layer | samples | intervals | active rate | mean interval | max interval | tail ratio |"
            );
            let _ = writeln!(out, "|---:|---:|---:|---:|---:|---:|---:|");
            for layer in &report.update_intervals.layers {
                let _ = writeln!(
                    out,
                    "| {} | {} | {} | {} | {} | {} | {} |",
                    layer.layer_index,
                    layer.samples,
                    layer.interval_count,
                    layer
                        .active_rate_mean
                        .map(|value| format!("{value:.6}"))
                        .unwrap_or_else(|| "-".to_string()),
                    layer
                        .mean_interval
                        .map(|value| format!("{value:.6}"))
                        .unwrap_or_else(|| "-".to_string()),
                    layer
                        .max_interval
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "-".to_string()),
                    layer
                        .tail_ratio
                        .map(|value| format!("{value:.6}"))
                        .unwrap_or_else(|| "-".to_string()),
                );
            }
        }
        out
    }

    fn write_text_artifact(path: &Path, contents: &str, label: &str) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create {} parent {}", label, parent.display())
            })?;
        }
        fs::write(path, contents)
            .with_context(|| format!("failed to write {} {}", label, path.display()))
    }
}

#[cfg(feature = "probe")]
fn main() {
    real::main();
}
