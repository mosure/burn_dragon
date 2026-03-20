use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_dragon_core::{BDH, BDHConfig, FusedKernelConfig, SequenceKernelKind};
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};

type TestBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

fn init_runtime(device: &<TestBackend as Backend>::Device) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

struct ParityCase {
    name: &'static str,
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    mlp_internal_dim_multiplier: usize,
    vocab_size: usize,
    token_sequences: &'static [&'static [i64]],
    logits_atol: f32,
    logits_rtol: f32,
    state_atol: f32,
    state_rtol: f32,
}

#[derive(Clone, Copy, Default)]
struct DiffSummary {
    max_abs: f32,
    mean_abs: f32,
}

impl DiffSummary {
    fn update(&mut self, sample: DiffSummary) {
        self.max_abs = self.max_abs.max(sample.max_abs);
        self.mean_abs = self.mean_abs.max(sample.mean_abs);
    }
}

fn build_config(
    case: &ParityCase,
    wgpu_recurrent_kernel: bool,
    rollout_fast_steps: usize,
    wgpu_rollout_fused: bool,
) -> BDHConfig {
    let mut config = BDHConfig {
        n_layer: case.n_layer,
        n_embd: case.n_embd,
        n_head: case.n_head,
        mlp_internal_dim_multiplier: case.mlp_internal_dim_multiplier,
        vocab_size: case.vocab_size,
        dropout: 0.0,
        fused_kernels: FusedKernelConfig {
            enabled: true,
            wgpu_recurrent_kernel,
            wgpu_rollout_fused,
            ..Default::default()
        },
        ..Default::default()
    };
    config
        .fused_kernels
        .set_block_sizes(8.min(case.n_embd), 8.min(case.n_embd));
    config.set_rollout_fast_steps_per_slow_step(rollout_fast_steps);
    config
}

fn build_dense_score_config(case: &ParityCase, wgpu_rollout_fused: bool) -> BDHConfig {
    let mut config = BDHConfig {
        n_layer: case.n_layer,
        n_embd: case.n_embd,
        n_head: case.n_head,
        mlp_internal_dim_multiplier: case.mlp_internal_dim_multiplier,
        vocab_size: case.vocab_size,
        dropout: 0.0,
        fused_kernels: FusedKernelConfig {
            enabled: true,
            wgpu_rollout_fused,
            ..Default::default()
        },
        sequence_kernel: SequenceKernelKind::BdhLinearDenseScoreExperimental,
        ..Default::default()
    };
    config
        .fused_kernels
        .set_block_sizes(8.min(case.n_embd), 8.min(case.n_embd));
    config
}

fn assert_close<const D: usize>(
    lhs: Tensor<TestBackend, D>,
    rhs: Tensor<TestBackend, D>,
    atol: f32,
    rtol: f32,
) -> DiffSummary {
    let lhs = lhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("lhs vec");
    let rhs = rhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rhs vec");

    assert_eq!(lhs.len(), rhs.len(), "length mismatch");

    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f32;
    for (a, b) in lhs.iter().zip(rhs.iter()) {
        let diff = (*a - *b).abs();
        max_abs = max_abs.max(diff);
        sum_abs += diff;
        let tol = atol + rtol * (*b).abs();
        assert!(
            diff <= tol,
            "difference {diff} exceeds tolerance {tol} (lhs={a}, rhs={b})"
        );
    }

    let mean_abs = if lhs.is_empty() {
        0.0
    } else {
        sum_abs / lhs.len() as f32
    };

    DiffSummary { max_abs, mean_abs }
}

fn make_tokens(
    device: &<TestBackend as Backend>::Device,
    token_ids: &[i64],
) -> Tensor<TestBackend, 2, Int> {
    Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(token_ids.to_vec(), [1, token_ids.len()]),
        device,
    )
}

fn run_case(
    case: &ParityCase,
    device: &<TestBackend as Backend>::Device,
    seed: u64,
    rollout_fast_steps: usize,
) {
    <TestBackend as Backend>::seed(device, seed);
    let baseline_model =
        BDH::<TestBackend>::new(build_config(case, false, rollout_fast_steps, false), device);

    <TestBackend as Backend>::seed(device, seed);
    let fused_model =
        BDH::<TestBackend>::new(build_config(case, true, rollout_fast_steps, true), device);

    let mut baseline_state = baseline_model.init_state();
    let mut fused_state = fused_model.init_state();
    let mut logits_summary = DiffSummary::default();

    for token_ids in case.token_sequences {
        let tokens = make_tokens(device, token_ids);
        let logits_baseline =
            baseline_model.forward_with_state(tokens.clone(), &mut baseline_state);
        let logits_fused = fused_model.forward_with_state(tokens, &mut fused_state);
        let summary = assert_close(
            logits_baseline,
            logits_fused,
            case.logits_atol,
            case.logits_rtol,
        );
        logits_summary.update(summary);
    }

    assert_eq!(
        baseline_state.position, fused_state.position,
        "state position mismatch for case {}",
        case.name
    );
    assert_eq!(
        baseline_state.layers.len(),
        fused_state.layers.len(),
        "state layer count mismatch for case {}",
        case.name
    );

    let mut state_summary = DiffSummary::default();
    for (baseline_layer, fused_layer) in baseline_state.layers.iter().zip(fused_state.layers.iter())
    {
        let baseline_rho = baseline_layer.rho.as_ref().expect("baseline rho").clone();
        let fused_rho = fused_layer.rho.as_ref().expect("fused rho").clone();
        let summary = assert_close(baseline_rho, fused_rho, case.state_atol, case.state_rtol);
        state_summary.update(summary);
    }

    println!(
        "rollout_forward_parity case={} fast_steps={} rollout_executor=wgpu_fused logits_max_abs={:.6} logits_mean_abs={:.6} state_max_abs={:.6} state_mean_abs={:.6} logits_atol={:.6} logits_rtol={:.6} state_atol={:.6} state_rtol={:.6} pass=true",
        case.name,
        rollout_fast_steps,
        logits_summary.max_abs,
        logits_summary.mean_abs,
        state_summary.max_abs,
        state_summary.mean_abs,
        case.logits_atol,
        case.logits_rtol,
        case.state_atol,
        case.state_rtol
    );
}

#[test]
fn recurrent_wgpu_kernel_matches_baseline_across_state_steps() {
    let device = <TestBackend as Backend>::Device::default();
    init_runtime(&device);
    const TOKENS_A: &[i64] = &[1, 2, 3, 4];
    const TOKENS_B: &[i64] = &[5, 6, 7];
    const TOKENS_C: &[i64] = &[8, 9];
    const TOKENS: &[&[i64]] = &[TOKENS_A, TOKENS_B, TOKENS_C];

    let case = ParityCase {
        name: "default_small",
        n_layer: 2,
        n_embd: 16,
        n_head: 2,
        mlp_internal_dim_multiplier: 2,
        vocab_size: 128,
        token_sequences: TOKENS,
        logits_atol: 4e-1,
        logits_rtol: 4e-1,
        state_atol: 5e-1,
        state_rtol: 5e-1,
    };

    for rollout_fast_steps in BDHConfig::SUPPORTED_ROLLOUT_FAST_STEPS {
        run_case(&case, &device, 2026, rollout_fast_steps);
    }
}

#[test]
fn recurrent_wgpu_kernel_matches_baseline_across_config_matrix() {
    let device = <TestBackend as Backend>::Device::default();
    init_runtime(&device);

    const CASE1_TOKENS_1: &[i64] = &[2, 4, 6];
    const CASE1_TOKENS_2: &[i64] = &[8, 10, 12, 14];
    const CASE1_TOKENS_3: &[i64] = &[16, 18];
    const CASE1_TOKENS: &[&[i64]] = &[CASE1_TOKENS_1, CASE1_TOKENS_2, CASE1_TOKENS_3];

    const CASE2_TOKENS_1: &[i64] = &[1, 3, 5, 7, 9];
    const CASE2_TOKENS_2: &[i64] = &[11, 13];
    const CASE2_TOKENS_3: &[i64] = &[15, 17, 19];
    const CASE2_TOKENS: &[&[i64]] = &[CASE2_TOKENS_1, CASE2_TOKENS_2, CASE2_TOKENS_3];

    const CASE3_TOKENS_1: &[i64] = &[21, 18, 15, 12];
    const CASE3_TOKENS_2: &[i64] = &[9, 6, 3];
    const CASE3_TOKENS_3: &[i64] = &[0, 1, 2, 3, 4];
    const CASE3_TOKENS: &[&[i64]] = &[CASE3_TOKENS_1, CASE3_TOKENS_2, CASE3_TOKENS_3];

    let cases = [
        ParityCase {
            name: "wider_heads",
            n_layer: 2,
            n_embd: 24,
            n_head: 4,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 192,
            token_sequences: CASE1_TOKENS,
            logits_atol: 8e-1,
            logits_rtol: 8e-1,
            state_atol: 8e-1,
            state_rtol: 8e-1,
        },
        ParityCase {
            name: "deeper_latent",
            n_layer: 3,
            n_embd: 32,
            n_head: 4,
            mlp_internal_dim_multiplier: 3,
            vocab_size: 256,
            token_sequences: CASE2_TOKENS,
            logits_atol: 8e-1,
            logits_rtol: 8e-1,
            state_atol: 8e-1,
            state_rtol: 8e-1,
        },
        ParityCase {
            name: "narrow_embed_high_multiplier",
            n_layer: 1,
            n_embd: 12,
            n_head: 3,
            mlp_internal_dim_multiplier: 4,
            vocab_size: 128,
            token_sequences: CASE3_TOKENS,
            logits_atol: 8e-1,
            logits_rtol: 8e-1,
            state_atol: 8e-1,
            state_rtol: 8e-1,
        },
    ];

    for (idx, case) in cases.iter().enumerate() {
        for rollout_fast_steps in BDHConfig::SUPPORTED_ROLLOUT_FAST_STEPS {
            run_case(case, &device, 3_000 + idx as u64, rollout_fast_steps);
        }
    }
}

#[test]
fn dense_score_wgpu_kernel_matches_baseline_across_state_steps() {
    let device = <TestBackend as Backend>::Device::default();
    init_runtime(&device);
    const TOKENS_A: &[i64] = &[1, 2, 3, 4, 5, 6];
    const TOKENS_B: &[i64] = &[7, 8, 9, 10];
    const TOKENS_C: &[i64] = &[11, 12, 13];
    const TOKENS: &[&[i64]] = &[TOKENS_A, TOKENS_B, TOKENS_C];

    let case = ParityCase {
        name: "dense_score_small",
        n_layer: 2,
        n_embd: 16,
        n_head: 2,
        mlp_internal_dim_multiplier: 2,
        vocab_size: 128,
        token_sequences: TOKENS,
        logits_atol: 4e-1,
        logits_rtol: 4e-1,
        state_atol: 4e-1,
        state_rtol: 4e-1,
    };

    <TestBackend as Backend>::seed(&device, 3030);
    let baseline_model = BDH::<TestBackend>::new(build_dense_score_config(&case, false), &device);
    <TestBackend as Backend>::seed(&device, 3030);
    let fused_model = BDH::<TestBackend>::new(build_dense_score_config(&case, true), &device);

    let mut baseline_state = baseline_model.init_state();
    let mut fused_state = fused_model.init_state();

    for token_ids in case.token_sequences {
        let tokens = make_tokens(&device, token_ids);
        let baseline_logits =
            baseline_model.forward_with_state(tokens.clone(), &mut baseline_state);
        let fused_logits = fused_model.forward_with_state(tokens, &mut fused_state);
        let _ = assert_close(
            baseline_logits,
            fused_logits,
            case.logits_atol,
            case.logits_rtol,
        );
    }

    for (baseline_layer, fused_layer) in baseline_state.layers.iter().zip(fused_state.layers.iter())
    {
        let baseline_rho = baseline_layer.rho.as_ref().expect("baseline rho").clone();
        let fused_rho = fused_layer.rho.as_ref().expect("fused rho").clone();
        let _ = assert_close(baseline_rho, fused_rho, case.state_atol, case.state_rtol);
    }
}
