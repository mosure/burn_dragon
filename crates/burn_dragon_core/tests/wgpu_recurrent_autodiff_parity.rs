use burn::optim::{AdamWConfig, GradientsParams, LearningRate, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_dragon_core::{BDH, BDHConfig, FusedKernelConfig};
use burn_dragon_wgpu::api::recurrent::supports_recurrent_backend;
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};

type InnerBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type TrainBackend = Autodiff<InnerBackend>;

#[derive(Clone, Copy)]
struct TrainParityCase {
    name: &'static str,
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    mlp_internal_dim_multiplier: usize,
    vocab_size: usize,
    loss_tol: f32,
    logits_atol: f32,
    logits_rtol: f32,
}

fn init_runtime(device: &<InnerBackend as Backend>::Device) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

fn build_case_config(
    case: &TrainParityCase,
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
    config.fused_kernels.set_block_sizes(8, 8);
    config.set_rollout_fast_steps_per_slow_step(rollout_fast_steps);
    config
}

fn sample_tokens(device: &<TrainBackend as Backend>::Device) -> Tensor<TrainBackend, 2, Int> {
    Tensor::<TrainBackend, 2, Int>::from_data(
        TensorData::new(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 5]),
        device,
    )
}

fn autoregressive_loss(logits: Tensor<TrainBackend, 3>) -> Tensor<TrainBackend, 1> {
    // Keep the loss value-dependent so output parity and backward parity are both exercised.
    logits.tanh().powf_scalar(2.0).mean()
}

fn train_one_step(
    mut model: BDH<TrainBackend>,
    inputs: Tensor<TrainBackend, 2, Int>,
) -> (BDH<TrainBackend>, f32) {
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<TrainBackend, BDH<TrainBackend>>();
    let lr: LearningRate = 1e-3;

    let logits = model.forward(inputs);
    let loss = autoregressive_loss(logits);
    let loss_value = loss
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss vec")[0];

    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    model = optimizer.step(lr, model, grads);

    (model, loss_value)
}

fn assert_close<const D: usize>(
    lhs: Tensor<InnerBackend, D>,
    rhs: Tensor<InnerBackend, D>,
    atol: f32,
    rtol: f32,
) -> (f32, f32) {
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

    let mut max_diff = 0.0_f32;
    let mut sum_diff = 0.0_f32;
    for (a, b) in lhs.iter().zip(rhs.iter()) {
        let diff = (*a - *b).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
        let tol = atol + rtol * b.abs();
        assert!(
            diff <= tol,
            "difference {diff} exceeds tolerance {tol} (lhs={a}, rhs={b})"
        );
    }

    let mean_diff = if lhs.is_empty() {
        0.0
    } else {
        sum_diff / lhs.len() as f32
    };
    (max_diff, mean_diff)
}

fn run_autodiff_case(
    case: &TrainParityCase,
    device: &<TrainBackend as Backend>::Device,
    seed: u64,
    rollout_fast_steps: usize,
) {
    <TrainBackend as Backend>::seed(device, seed);
    let baseline = BDH::<TrainBackend>::new(
        build_case_config(case, false, rollout_fast_steps, false),
        device,
    );
    <TrainBackend as Backend>::seed(device, seed);
    let fused = BDH::<TrainBackend>::new(
        build_case_config(case, true, rollout_fast_steps, true),
        device,
    );

    let inputs = sample_tokens(device);
    let (baseline_model, baseline_loss) = train_one_step(baseline, inputs.clone());
    let (fused_model, fused_loss) = train_one_step(fused, inputs.clone());

    let loss_diff = (baseline_loss - fused_loss).abs();
    assert!(
        loss_diff <= case.loss_tol,
        "case {} fast_steps={} training loss drift {loss_diff} exceeds tolerance {}",
        case.name,
        rollout_fast_steps,
        case.loss_tol
    );

    let baseline_logits = baseline_model.forward(inputs.clone()).inner();
    let fused_logits = fused_model.forward(inputs).inner();
    let (max_abs, mean_abs) = assert_close(
        baseline_logits,
        fused_logits,
        case.logits_atol,
        case.logits_rtol,
    );
    println!(
        "rollout_backward_parity case={} fast_steps={} rollout_executor=wgpu_fused loss_abs={loss_diff:.6} logits_max_abs={max_abs:.6} logits_mean_abs={mean_abs:.6} loss_tol={:.6} logits_atol={:.6} logits_rtol={:.6} pass=true",
        case.name, rollout_fast_steps, case.loss_tol, case.logits_atol, case.logits_rtol,
    );
}

#[test]
fn recurrent_wgpu_kernel_autodiff_tracks_forward_and_backward() {
    let device = <TrainBackend as Backend>::Device::default();
    init_runtime(&device);
    assert!(
        supports_recurrent_backend::<TrainBackend>(),
        "autodiff cube backend should be routable to the fused recurrent kernel"
    );

    let case = TrainParityCase {
        name: "default",
        n_layer: 2,
        n_embd: 32,
        n_head: 4,
        mlp_internal_dim_multiplier: 2,
        vocab_size: 128,
        loss_tol: 5e-2,
        logits_atol: 6e-1,
        logits_rtol: 6e-1,
    };
    for rollout_fast_steps in BDHConfig::SUPPORTED_ROLLOUT_FAST_STEPS {
        run_autodiff_case(&case, &device, 2026, rollout_fast_steps);
    }
}

#[test]
fn recurrent_wgpu_kernel_autodiff_tracks_forward_backward_across_config_matrix() {
    let device = <TrainBackend as Backend>::Device::default();
    init_runtime(&device);
    assert!(
        supports_recurrent_backend::<TrainBackend>(),
        "autodiff cube backend should be routable to the fused recurrent kernel"
    );

    let cases = [
        TrainParityCase {
            name: "tiny",
            n_layer: 2,
            n_embd: 32,
            n_head: 4,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 128,
            loss_tol: 5e-2,
            logits_atol: 6e-1,
            logits_rtol: 6e-1,
        },
        TrainParityCase {
            name: "wide",
            n_layer: 3,
            n_embd: 48,
            n_head: 6,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 192,
            loss_tol: 8e-2,
            logits_atol: 1.0,
            logits_rtol: 1.0,
        },
        TrainParityCase {
            name: "deep",
            n_layer: 4,
            n_embd: 64,
            n_head: 8,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 256,
            loss_tol: 1e-1,
            logits_atol: 1.2,
            logits_rtol: 1.2,
        },
    ];

    for (idx, case) in cases.iter().enumerate() {
        for rollout_fast_steps in BDHConfig::SUPPORTED_ROLLOUT_FAST_STEPS {
            run_autodiff_case(case, &device, 2_026 + idx as u64 * 17, rollout_fast_steps);
        }
    }
}
