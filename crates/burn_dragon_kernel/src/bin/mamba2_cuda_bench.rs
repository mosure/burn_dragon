#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("mamba2_cuda_bench requires --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
mod app {
    use std::process::{Command, Stdio};
    use std::thread;
    use std::time::Duration;
    use std::time::Instant;

    use burn::tensor::backend::Backend as BackendTrait;
    use burn::tensor::{Distribution, ElementConversion, Tensor, TensorData};
    use burn_autodiff::Autodiff;
    use burn_cubecl::cubecl::Runtime;
    use burn_cubecl::cubecl::cuda::CudaRuntime;
    use burn_cuda::Cuda;
    use burn_dragon_kernel::kernels::sequence::mamba2::forward::{
        CudaShellCoreMode, CudaSsdCoreMode, Mamba2TensorizedState,
        tensorized_mamba2_forward_custom_backward_with_cuda_modes,
        tensorized_mamba2_forward_direct_graph,
    };
    use serde::{Deserialize, Serialize};

    type Backend = Cuda<f32, i32>;
    type AutodiffBackend = Autodiff<Backend>;
    type Device = <Backend as BackendTrait>::Device;

    #[derive(Clone, Copy)]
    struct BenchCase {
        name: &'static str,
        batch: usize,
        time: usize,
        d_model: usize,
        d_state: usize,
        d_conv: usize,
        expand: usize,
        headdim: usize,
        ngroups: usize,
    }

    #[derive(Deserialize, Serialize)]
    struct BenchCaseReport {
        name: String,
        batch: usize,
        time: usize,
        d_model: usize,
        d_state: usize,
        d_conv: usize,
        expand: usize,
        headdim: usize,
        ngroups: usize,
    }

    impl From<BenchCase> for BenchCaseReport {
        fn from(value: BenchCase) -> Self {
            Self {
                name: value.name.to_string(),
                batch: value.batch,
                time: value.time,
                d_model: value.d_model,
                d_state: value.d_state,
                d_conv: value.d_conv,
                expand: value.expand,
                headdim: value.headdim,
                ngroups: value.ngroups,
            }
        }
    }

    #[derive(Clone, Copy, Deserialize, Serialize)]
    struct MemorySnapshot {
        reserved: u64,
        in_use: u64,
    }

    #[derive(Deserialize, Serialize)]
    struct BenchResult {
        case: BenchCaseReport,
        warmup: usize,
        repetitions: usize,
        graph_forward_ms: f64,
        wrapper_forward_ms: f64,
        ssd_fused_forward_ms: f64,
        shell_fused_forward_ms: f64,
        shell_fused_vs_ssd_fused_forward_speedup_x: f64,
        shell_fused_vs_wrapper_forward_speedup_x: f64,
        graph_backward_ms: f64,
        wrapper_backward_ms: f64,
        ssd_fused_backward_ms: f64,
        shell_fused_backward_ms: f64,
        shell_fused_vs_ssd_fused_backward_speedup_x: f64,
        shell_fused_vs_wrapper_backward_speedup_x: f64,
        output_max_abs: f64,
        conv_state_max_abs: f64,
        ssm_state_max_abs: f64,
        memory_before: MemorySnapshot,
        memory_after: MemorySnapshot,
    }

    #[derive(Serialize)]
    struct BenchReport {
        benchmark: &'static str,
        backend: &'static str,
        profile: &'static str,
        results: Vec<BenchResult>,
    }

    #[derive(Clone)]
    struct ParamsAutodiff {
        d_inner: usize,
        d_state: usize,
        d_conv: usize,
        headdim: usize,
        ngroups: usize,
        nheads: usize,
        norm_eps: f32,
        in_proj: Tensor<AutodiffBackend, 2>,
        conv_weight: Tensor<AutodiffBackend, 2>,
        conv_bias: Tensor<AutodiffBackend, 1>,
        dt_bias: Tensor<AutodiffBackend, 1>,
        a_log: Tensor<AutodiffBackend, 1>,
        d_skip: Tensor<AutodiffBackend, 1>,
        norm_weight: Tensor<AutodiffBackend, 1>,
        out_proj: Tensor<AutodiffBackend, 2>,
    }

    #[derive(Clone)]
    struct StateAutodiff {
        conv: Tensor<AutodiffBackend, 4>,
        ssm: Tensor<AutodiffBackend, 4>,
    }

    const COMPACT_CASES: &[BenchCase] = &[BenchCase {
        name: "cuda_b1_t16_dm64_ds8_dc4_e2_h32_g1",
        batch: 1,
        time: 16,
        d_model: 64,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        headdim: 32,
        ngroups: 1,
    }];

    const FULL_CASES: &[BenchCase] = &[
        BenchCase {
            name: "cuda_b1_t64_dm256_ds16_dc4_e2_h64_g1",
            batch: 1,
            time: 64,
            d_model: 256,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            headdim: 64,
            ngroups: 1,
        },
        BenchCase {
            name: "cuda_b1_t128_dm256_ds16_dc4_e2_h64_g1",
            batch: 1,
            time: 128,
            d_model: 256,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            headdim: 64,
            ngroups: 1,
        },
    ];

    pub fn main() {
        let profile = bench_profile();
        if let Some(case_name) = child_case_name() {
            let device = Device::default();
            <Backend as BackendTrait>::seed(&device, 20260328);
            let warmup = 1;
            let repetitions = if profile == "full" { 3 } else { 2 };
            let case = bench_cases(profile)
                .iter()
                .copied()
                .find(|candidate| candidate.name == case_name)
                .unwrap_or_else(|| panic!("unknown cuda bench case {case_name}"));
            println!(
                "{}",
                serde_json::to_string(&run_case(case, &device, warmup, repetitions))
                    .expect("serialize child mamba2 cuda bench result")
            );
            return;
        }

        let timeout = bench_case_timeout(profile);
        let results = bench_cases(profile)
            .iter()
            .copied()
            .map(|case| run_case_with_timeout(case, profile, timeout))
            .collect::<Vec<_>>();

        println!(
            "{}",
            serde_json::to_string_pretty(&BenchReport {
                benchmark: "burn_dragon_kernel mamba2 cuda analytic backward bench",
                backend: "cuda",
                profile,
                results,
            })
            .expect("serialize mamba2 cuda bench report")
        );
    }

    fn bench_profile() -> &'static str {
        match std::env::var("BURN_DRAGON_BENCH_PROFILE")
            .unwrap_or_else(|_| "compact".to_string())
            .as_str()
        {
            "full" => "full",
            _ => "compact",
        }
    }

    fn bench_cases(profile: &'static str) -> &'static [BenchCase] {
        if profile == "full" {
            FULL_CASES
        } else {
            COMPACT_CASES
        }
    }

    fn child_case_name() -> Option<String> {
        std::env::var("BURN_DRAGON_BENCH_CHILD_CASE").ok()
    }

    fn bench_case_timeout(profile: &'static str) -> Duration {
        let default_secs = if profile == "full" { 180 } else { 45 };
        let secs = std::env::var("BURN_DRAGON_BENCH_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(default_secs);
        Duration::from_secs(secs)
    }

    fn run_case_with_timeout(
        case: BenchCase,
        profile: &'static str,
        timeout: Duration,
    ) -> BenchResult {
        let mut child = Command::new(std::env::current_exe().expect("current exe"))
            .env("BURN_DRAGON_BENCH_PROFILE", profile)
            .env("BURN_DRAGON_BENCH_CHILD_CASE", case.name)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn mamba2 cuda bench child");
        let started = Instant::now();

        loop {
            if let Some(status) = child.try_wait().expect("poll mamba2 cuda bench child") {
                let output = child
                    .wait_with_output()
                    .expect("collect mamba2 cuda bench child output");
                if !status.success() {
                    panic!(
                        "mamba2 cuda bench child {} failed: {}",
                        case.name,
                        String::from_utf8_lossy(&output.stderr)
                    );
                }
                return serde_json::from_slice::<BenchResult>(&output.stdout).unwrap_or_else(
                    |error| {
                        panic!(
                            "parse mamba2 cuda bench child {} output failed: {error}; stdout={}",
                            case.name,
                            String::from_utf8_lossy(&output.stdout)
                        )
                    },
                );
            }
            if started.elapsed() > timeout {
                let _ = child.kill();
                let output = child
                    .wait_with_output()
                    .expect("collect timed-out mamba2 cuda bench child output");
                panic!(
                    "mamba2 cuda bench child {} exceeded {:?}; stderr={}",
                    case.name,
                    timeout,
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            thread::sleep(Duration::from_millis(100));
        }
    }

    fn run_case(
        case: BenchCase,
        device: &Device,
        warmup: usize,
        repetitions: usize,
    ) -> BenchResult {
        let hidden = Tensor::<AutodiffBackend, 4>::random(
            [case.batch, 1, case.time, case.d_model],
            Distribution::Uniform(-0.5, 0.5),
            device,
        )
        .require_grad();
        let params = build_params_autodiff(case, device);
        let memory_before = memory_snapshot(device);

        for _ in 0..warmup {
            let _ = mamba2_tensorized_autodiff_graph(hidden.clone(), &params);
            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            let _ = mamba2_tensorized_autodiff_custom_with_mode(
                hidden.clone(),
                &params,
                CudaSsdCoreMode::ForcedDisabled,
                CudaShellCoreMode::ForcedDisabled,
            );
            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            let _ = mamba2_tensorized_autodiff_custom_with_mode(
                hidden.clone(),
                &params,
                CudaSsdCoreMode::ForcedEnabled,
                CudaShellCoreMode::ForcedDisabled,
            );
            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            let _ = mamba2_tensorized_autodiff_custom_with_mode(
                hidden.clone(),
                &params,
                CudaSsdCoreMode::ForcedEnabled,
                CudaShellCoreMode::ForcedEnabled,
            );
            let _ = <AutodiffBackend as BackendTrait>::sync(device);

            let _ = mamba2_tensorized_autodiff_graph(hidden.clone(), &params)
                .0
                .sum()
                .backward();
            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            let _ = mamba2_tensorized_autodiff_custom_with_mode(
                hidden.clone(),
                &params,
                CudaSsdCoreMode::ForcedDisabled,
                CudaShellCoreMode::ForcedDisabled,
            )
            .0
            .sum()
            .backward();
            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            let _ = mamba2_tensorized_autodiff_custom_with_mode(
                hidden.clone(),
                &params,
                CudaSsdCoreMode::ForcedEnabled,
                CudaShellCoreMode::ForcedDisabled,
            )
            .0
            .sum()
            .backward();
            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            let _ = mamba2_tensorized_autodiff_custom_with_mode(
                hidden.clone(),
                &params,
                CudaSsdCoreMode::ForcedEnabled,
                CudaShellCoreMode::ForcedEnabled,
            )
            .0
            .sum()
            .backward();
            let _ = <AutodiffBackend as BackendTrait>::sync(device);
        }

        let mut graph_forward_ms = 0.0;
        let mut wrapper_forward_ms = 0.0;
        let mut ssd_fused_forward_ms = 0.0;
        let mut shell_fused_forward_ms = 0.0;
        for _ in 0..repetitions {
            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            let started = Instant::now();
            let _ = mamba2_tensorized_autodiff_graph(hidden.clone(), &params);
            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            graph_forward_ms += started.elapsed().as_secs_f64() * 1_000.0;

            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            let started = Instant::now();
            let _ = mamba2_tensorized_autodiff_custom_with_mode(
                hidden.clone(),
                &params,
                CudaSsdCoreMode::ForcedDisabled,
                CudaShellCoreMode::ForcedDisabled,
            );
            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            wrapper_forward_ms += started.elapsed().as_secs_f64() * 1_000.0;

            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            let started = Instant::now();
            let _ = mamba2_tensorized_autodiff_custom_with_mode(
                hidden.clone(),
                &params,
                CudaSsdCoreMode::ForcedEnabled,
                CudaShellCoreMode::ForcedDisabled,
            );
            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            ssd_fused_forward_ms += started.elapsed().as_secs_f64() * 1_000.0;

            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            let started = Instant::now();
            let _ = mamba2_tensorized_autodiff_custom_with_mode(
                hidden.clone(),
                &params,
                CudaSsdCoreMode::ForcedEnabled,
                CudaShellCoreMode::ForcedEnabled,
            );
            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            shell_fused_forward_ms += started.elapsed().as_secs_f64() * 1_000.0;
        }

        let mut graph_backward_ms = 0.0;
        let mut wrapper_backward_ms = 0.0;
        let mut ssd_fused_backward_ms = 0.0;
        let mut shell_fused_backward_ms = 0.0;
        for _ in 0..repetitions {
            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            let started = Instant::now();
            let graph_loss = mamba2_tensorized_autodiff_graph(hidden.clone(), &params)
                .0
                .sum();
            let _ = graph_loss.backward();
            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            graph_backward_ms += started.elapsed().as_secs_f64() * 1_000.0;

            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            let started = Instant::now();
            let wrapper_loss = mamba2_tensorized_autodiff_custom_with_mode(
                hidden.clone(),
                &params,
                CudaSsdCoreMode::ForcedDisabled,
                CudaShellCoreMode::ForcedDisabled,
            )
            .0
            .sum();
            let _ = wrapper_loss.backward();
            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            wrapper_backward_ms += started.elapsed().as_secs_f64() * 1_000.0;

            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            let started = Instant::now();
            let fused_loss = mamba2_tensorized_autodiff_custom_with_mode(
                hidden.clone(),
                &params,
                CudaSsdCoreMode::ForcedEnabled,
                CudaShellCoreMode::ForcedDisabled,
            )
            .0
            .sum();
            let _ = fused_loss.backward();
            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            ssd_fused_backward_ms += started.elapsed().as_secs_f64() * 1_000.0;

            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            let started = Instant::now();
            let fused_loss = mamba2_tensorized_autodiff_custom_with_mode(
                hidden.clone(),
                &params,
                CudaSsdCoreMode::ForcedEnabled,
                CudaShellCoreMode::ForcedEnabled,
            )
            .0
            .sum();
            let _ = fused_loss.backward();
            let _ = <AutodiffBackend as BackendTrait>::sync(device);
            shell_fused_backward_ms += started.elapsed().as_secs_f64() * 1_000.0;
        }

        let (graph_output, graph_state) = mamba2_tensorized_autodiff_graph(hidden.clone(), &params);
        let (wrapper_output, wrapper_state) = mamba2_tensorized_autodiff_custom_with_mode(
            hidden.clone(),
            &params,
            CudaSsdCoreMode::ForcedDisabled,
            CudaShellCoreMode::ForcedDisabled,
        );
        let (ssd_fused_output, ssd_fused_state) = mamba2_tensorized_autodiff_custom_with_mode(
            hidden.clone(),
            &params,
            CudaSsdCoreMode::ForcedEnabled,
            CudaShellCoreMode::ForcedDisabled,
        );
        let (shell_fused_output, shell_fused_state) = mamba2_tensorized_autodiff_custom_with_mode(
            hidden,
            &params,
            CudaSsdCoreMode::ForcedEnabled,
            CudaShellCoreMode::ForcedEnabled,
        );
        let _ = <AutodiffBackend as BackendTrait>::sync(device);
        let memory_after = memory_snapshot(device);

        BenchResult {
            case: case.into(),
            warmup,
            repetitions,
            graph_forward_ms: graph_forward_ms / repetitions.max(1) as f64,
            wrapper_forward_ms: wrapper_forward_ms / repetitions.max(1) as f64,
            ssd_fused_forward_ms: ssd_fused_forward_ms / repetitions.max(1) as f64,
            shell_fused_forward_ms: shell_fused_forward_ms / repetitions.max(1) as f64,
            shell_fused_vs_ssd_fused_forward_speedup_x: ssd_fused_forward_ms
                / shell_fused_forward_ms.max(f64::EPSILON),
            shell_fused_vs_wrapper_forward_speedup_x: wrapper_forward_ms
                / shell_fused_forward_ms.max(f64::EPSILON),
            graph_backward_ms: graph_backward_ms / repetitions.max(1) as f64,
            wrapper_backward_ms: wrapper_backward_ms / repetitions.max(1) as f64,
            ssd_fused_backward_ms: ssd_fused_backward_ms / repetitions.max(1) as f64,
            shell_fused_backward_ms: shell_fused_backward_ms / repetitions.max(1) as f64,
            shell_fused_vs_ssd_fused_backward_speedup_x: ssd_fused_backward_ms
                / shell_fused_backward_ms.max(f64::EPSILON),
            shell_fused_vs_wrapper_backward_speedup_x: wrapper_backward_ms
                / shell_fused_backward_ms.max(f64::EPSILON),
            output_max_abs: max_abs_4(wrapper_output.clone(), shell_fused_output.clone())
                .max(max_abs_4(
                    ssd_fused_output.clone(),
                    shell_fused_output.clone(),
                ))
                .max(max_abs_4(graph_output, shell_fused_output)),
            conv_state_max_abs: max_abs_4(
                wrapper_state.conv.clone(),
                shell_fused_state.conv.clone(),
            )
            .max(max_abs_4(
                ssd_fused_state.conv.clone(),
                shell_fused_state.conv.clone(),
            ))
            .max(max_abs_4(graph_state.conv, shell_fused_state.conv)),
            ssm_state_max_abs: max_abs_4(wrapper_state.ssm.clone(), shell_fused_state.ssm.clone())
                .max(max_abs_4(
                    ssd_fused_state.ssm.clone(),
                    shell_fused_state.ssm.clone(),
                ))
                .max(max_abs_4(graph_state.ssm, shell_fused_state.ssm)),
            memory_before,
            memory_after,
        }
    }

    fn build_params_autodiff(case: BenchCase, device: &Device) -> ParamsAutodiff {
        let d_inner = case.d_model * case.expand;
        let nheads = d_inner / case.headdim;
        let conv_dim = d_inner + 2 * case.ngroups * case.d_state;
        ParamsAutodiff {
            d_inner,
            d_state: case.d_state,
            d_conv: case.d_conv,
            headdim: case.headdim,
            ngroups: case.ngroups,
            nheads,
            norm_eps: 1.0e-5,
            in_proj: Tensor::<AutodiffBackend, 2>::random(
                [
                    case.d_model,
                    2 * d_inner + 2 * case.ngroups * case.d_state + nheads,
                ],
                Distribution::Uniform(-0.05, 0.05),
                device,
            )
            .require_grad(),
            conv_weight: Tensor::<AutodiffBackend, 2>::random(
                [conv_dim, case.d_conv],
                Distribution::Uniform(-0.05, 0.05),
                device,
            )
            .require_grad(),
            conv_bias: Tensor::<AutodiffBackend, 1>::zeros([conv_dim], device).require_grad(),
            dt_bias: Tensor::<AutodiffBackend, 1>::from_data(
                TensorData::new(vec![0.01; nheads], [nheads]),
                device,
            )
            .require_grad(),
            a_log: Tensor::<AutodiffBackend, 1>::from_data(
                TensorData::new(vec![1.0f32.ln(); nheads], [nheads]),
                device,
            )
            .require_grad(),
            d_skip: Tensor::<AutodiffBackend, 1>::ones([nheads], device).require_grad(),
            norm_weight: Tensor::<AutodiffBackend, 1>::ones([d_inner], device).require_grad(),
            out_proj: Tensor::<AutodiffBackend, 2>::random(
                [d_inner, case.d_model],
                Distribution::Uniform(-0.05, 0.05),
                device,
            )
            .require_grad(),
        }
    }

    fn mamba2_tensorized_autodiff_graph(
        hidden_states: Tensor<AutodiffBackend, 4>,
        params: &ParamsAutodiff,
    ) -> (Tensor<AutodiffBackend, 4>, StateAutodiff) {
        let batch = hidden_states.shape().dims::<4>()[0];
        let conv_dim = params.d_inner + 2 * params.ngroups * params.d_state;
        let device = hidden_states.device();
        let output = tensorized_mamba2_forward_direct_graph(
            hidden_states,
            params.d_inner,
            params.d_state,
            params.d_conv,
            params.headdim,
            params.ngroups,
            params.in_proj.clone(),
            params.conv_weight.clone(),
            Some(params.conv_bias.clone()),
            params.dt_bias.clone(),
            params.a_log.clone(),
            params.d_skip.clone(),
            params.norm_weight.clone(),
            params.norm_eps,
            params.out_proj.clone(),
            Some(Mamba2TensorizedState {
                conv: Tensor::<AutodiffBackend, 4>::zeros(
                    [batch, 1, conv_dim, params.d_conv],
                    &device,
                ),
                ssm: Tensor::<AutodiffBackend, 4>::zeros(
                    [batch, params.nheads, params.headdim, params.d_state],
                    &device,
                ),
            }),
        );
        (
            output.context,
            StateAutodiff {
                conv: output.state.conv,
                ssm: output.state.ssm,
            },
        )
    }

    fn mamba2_tensorized_autodiff_custom_with_mode(
        hidden_states: Tensor<AutodiffBackend, 4>,
        params: &ParamsAutodiff,
        cuda_ssd_core_mode: CudaSsdCoreMode,
        cuda_shell_core_mode: CudaShellCoreMode,
    ) -> (Tensor<AutodiffBackend, 4>, StateAutodiff) {
        let batch = hidden_states.shape().dims::<4>()[0];
        let conv_dim = params.d_inner + 2 * params.ngroups * params.d_state;
        let device = hidden_states.device();
        let output = tensorized_mamba2_forward_custom_backward_with_cuda_modes(
            hidden_states,
            params.d_inner,
            params.d_state,
            params.d_conv,
            params.headdim,
            params.ngroups,
            params.in_proj.clone(),
            params.conv_weight.clone(),
            Some(params.conv_bias.clone()),
            params.dt_bias.clone(),
            params.a_log.clone(),
            params.d_skip.clone(),
            params.norm_weight.clone(),
            params.norm_eps,
            params.out_proj.clone(),
            Some(Mamba2TensorizedState {
                conv: Tensor::<AutodiffBackend, 4>::zeros(
                    [batch, 1, conv_dim, params.d_conv],
                    &device,
                ),
                ssm: Tensor::<AutodiffBackend, 4>::zeros(
                    [batch, params.nheads, params.headdim, params.d_state],
                    &device,
                ),
            }),
            cuda_ssd_core_mode,
            cuda_shell_core_mode,
        )
        .expect("cuda custom backward path available");
        (
            output.context,
            StateAutodiff {
                conv: output.state.conv,
                ssm: output.state.ssm,
            },
        )
    }

    fn max_abs_4(lhs: Tensor<AutodiffBackend, 4>, rhs: Tensor<AutodiffBackend, 4>) -> f64 {
        lhs.sub(rhs).abs().max().into_scalar().elem::<f32>() as f64
    }

    fn memory_snapshot(device: &Device) -> MemorySnapshot {
        let usage = <CudaRuntime as Runtime>::client(device)
            .memory_usage()
            .expect("cuda memory usage");
        MemorySnapshot {
            reserved: usage.bytes_reserved,
            in_use: usage.bytes_in_use,
        }
    }
}

#[cfg(feature = "cuda")]
fn main() {
    app::main();
}
