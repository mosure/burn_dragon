use std::time::Instant;

use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Int, Tensor, TensorData};
use burn_cubecl::CubeBackend;
use burn_dragon_kernel::api::low_bit::{
    diagnose_wgpu_packed_dot_decoder_tail, diagnose_wgpu_packed_dot_lowrank_projection,
    diagnose_wgpu_quantize_pack_activation_i8x4, pack_lowrank_input_codes_i8x4,
    pack_rho_int8_block_device_reference, packed_decoder_tail_device_reference,
    packed_decoder_tail_grad_input_device_reference,
    packed_decoder_tail_grad_weight_device_reference, packed_lowrank_grad_input_device_reference,
    packed_lowrank_grad_weight_device_reference, packed_lowrank_projection_device_reference,
    try_cube_fused_packed_decoder_tail_wgpu, try_cube_fused_packed_lowrank_projection_wgpu,
    try_fused_packed_decoder_tail, try_fused_packed_decoder_tail_grad_input,
    try_fused_packed_decoder_tail_grad_weight, try_fused_packed_lowrank_grad_input,
    try_fused_packed_lowrank_grad_weight, try_fused_packed_lowrank_projection,
    try_wgpu_packed_dot_decoder_tail, try_wgpu_packed_dot_lowrank_projection,
    unpack_rho_int8_block_device_reference,
};
use burn_wgpu::{RuntimeOptions, WgpuRuntime, graphics};

type BenchBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

fn init_runtime(device: &<BenchBackend as BackendTrait>::Device) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

fn deterministic_values(len: usize, offset: f32) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            (((idx as f32) * 0.173) + offset).sin() * 0.5
                + (((idx as f32) * 0.117) + offset).cos() * 0.25
        })
        .collect()
}

fn quantize_signed_values(values: &[f32]) -> (Vec<i32>, f32) {
    let mean_abs = if values.is_empty() {
        0.0
    } else {
        values.iter().map(|value| value.abs()).sum::<f32>() / values.len() as f32
    };
    let scale = (mean_abs * 2.0 / 127.0).max(1.0e-8);
    let codes = values
        .iter()
        .map(|value| (value / scale).round().clamp(-127.0, 127.0) as i32)
        .collect();
    (codes, scale)
}

fn tensor_from_values<const D: usize>(
    values: Vec<f32>,
    shape: [usize; D],
    device: &<BenchBackend as BackendTrait>::Device,
) -> Tensor<BenchBackend, D> {
    Tensor::<BenchBackend, D>::from_data(TensorData::new(values, shape), device)
}

fn int_tensor_from_values<const D: usize>(
    values: Vec<i32>,
    shape: [usize; D],
    device: &<BenchBackend as BackendTrait>::Device,
) -> Tensor<BenchBackend, D, Int> {
    Tensor::<BenchBackend, D, Int>::from_data(TensorData::new(values, shape), device)
}

fn max_abs_diff<const D: usize>(lhs: Tensor<BenchBackend, D>, rhs: Tensor<BenchBackend, D>) -> f32 {
    let lhs = lhs
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("lhs values");
    let rhs = rhs
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rhs values");
    lhs.into_iter()
        .zip(rhs)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max)
}

fn bench_case(mut func: impl FnMut() -> Tensor<BenchBackend, 4>, iters: usize) -> f64 {
    for _ in 0..3 {
        let _ = func().sum().into_data();
    }
    let start = Instant::now();
    for _ in 0..iters {
        let _ = func().sum().into_data();
    }
    start.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn bench_case_2d(mut func: impl FnMut() -> Tensor<BenchBackend, 2>, iters: usize) -> f64 {
    for _ in 0..3 {
        let _ = func().sum().into_data();
    }
    let start = Instant::now();
    for _ in 0..iters {
        let _ = func().sum().into_data();
    }
    start.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn bench_case_int4(mut func: impl FnMut() -> Tensor<BenchBackend, 4, Int>, iters: usize) -> f64 {
    for _ in 0..3 {
        let _ = func().sum().into_data();
    }
    let start = Instant::now();
    for _ in 0..iters {
        let _ = func().sum().into_data();
    }
    start.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn main() {
    let iters = std::env::args()
        .skip(1)
        .find_map(|arg| {
            arg.strip_prefix("--iters=")
                .and_then(|value| value.parse::<usize>().ok())
        })
        .unwrap_or(20);

    let device = <BenchBackend as BackendTrait>::Device::default();
    init_runtime(&device);

    let input_shape = [8, 1, 128, 256];
    let lowrank_weight_shape = [4, 256, 128];
    let decoder_weight_shape = [512, 256];
    let y_shape = [8, 4, 128, 128];
    let grad_shape = [8, 4, 128, 128];
    let residual_grad_shape = [8, 1, 128, 256];
    let rho_shape = [8, 4, 128, 128];

    let input_values = deterministic_values(input_shape.iter().product(), 0.2);
    let lowrank_weight_values = deterministic_values(lowrank_weight_shape.iter().product(), 0.7);
    let y_values = deterministic_values(y_shape.iter().product(), 0.5);
    let decoder_weight_values = deterministic_values(decoder_weight_shape.iter().product(), 1.0);
    let grad_values = deterministic_values(grad_shape.iter().product(), 1.4);
    let residual_grad_values = deterministic_values(residual_grad_shape.iter().product(), 1.7);
    let rho_values = deterministic_values(rho_shape.iter().product(), 2.1);

    let input_codes = quantize_signed_values(&input_values);
    let lowrank_weight_codes = quantize_signed_values(&lowrank_weight_values);
    let y_codes = quantize_signed_values(&y_values);
    let decoder_weight_codes = quantize_signed_values(&decoder_weight_values);

    let input_codes_tensor = int_tensor_from_values(input_codes.0, input_shape, &device);
    let input_scale_tensor = Tensor::<BenchBackend, 1>::from_data([input_codes.1], &device);
    let lowrank_weight_codes_tensor =
        int_tensor_from_values(lowrank_weight_codes.0, lowrank_weight_shape, &device);
    let y_codes_tensor = int_tensor_from_values(y_codes.0, y_shape, &device);
    let decoder_weight_codes_tensor =
        int_tensor_from_values(decoder_weight_codes.0, decoder_weight_shape, &device);
    let grad_output = tensor_from_values(grad_values, grad_shape, &device);
    let residual_grad_output =
        tensor_from_values(residual_grad_values, residual_grad_shape, &device);
    let rho = tensor_from_values(rho_values, rho_shape, &device);

    let lowrank_forward_ref = packed_lowrank_projection_device_reference(
        input_codes_tensor.clone().float().mul_scalar(input_codes.1),
        lowrank_weight_codes_tensor.clone(),
        lowrank_weight_codes.1,
        lowrank_weight_shape[2],
    );
    let lowrank_forward_cube = try_cube_fused_packed_lowrank_projection_wgpu(
        &input_codes_tensor,
        &lowrank_weight_codes_tensor,
        input_codes.1,
        lowrank_weight_codes.1,
        lowrank_weight_shape[2],
    )
    .expect("cube lowrank forward");
    let lowrank_forward_packed_dot = diagnose_wgpu_packed_dot_lowrank_projection(
        &input_codes_tensor,
        &lowrank_weight_codes_tensor,
        input_codes.1,
        lowrank_weight_codes.1,
        lowrank_weight_shape[2],
    );
    let lowrank_forward_fused = try_fused_packed_lowrank_projection(
        &input_codes_tensor,
        &lowrank_weight_codes_tensor,
        input_codes.1,
        lowrank_weight_codes.1,
        lowrank_weight_shape[2],
    )
    .expect("fused lowrank forward");

    let decoder_forward_ref = packed_decoder_tail_device_reference(
        y_codes_tensor.clone().float().mul_scalar(y_codes.1),
        decoder_weight_codes_tensor.clone(),
        decoder_weight_codes.1,
    );
    let decoder_forward_cube = try_cube_fused_packed_decoder_tail_wgpu(
        &y_codes_tensor,
        &decoder_weight_codes_tensor,
        y_codes.1,
        decoder_weight_codes.1,
    )
    .expect("cube decoder tail");
    let decoder_forward_packed_dot = diagnose_wgpu_packed_dot_decoder_tail(
        &y_codes_tensor,
        &decoder_weight_codes_tensor,
        y_codes.1,
        decoder_weight_codes.1,
    );
    let decoder_forward_fused = try_fused_packed_decoder_tail(
        &y_codes_tensor,
        &decoder_weight_codes_tensor,
        y_codes.1,
        decoder_weight_codes.1,
    )
    .expect("fused decoder tail");

    let lowrank_grad_input_ref = packed_lowrank_grad_input_device_reference(
        grad_output.clone(),
        lowrank_weight_codes_tensor.clone(),
        lowrank_weight_codes.1,
        1,
    );
    let lowrank_grad_input_fused = try_fused_packed_lowrank_grad_input(
        &grad_output,
        &lowrank_weight_codes_tensor,
        lowrank_weight_codes.1,
        1,
    )
    .expect("fused lowrank grad input");

    let lowrank_grad_weight_ref = packed_lowrank_grad_weight_device_reference(
        input_codes_tensor.clone(),
        grad_output.clone(),
        input_codes.1,
    );
    let lowrank_grad_weight_fused =
        try_fused_packed_lowrank_grad_weight(&input_codes_tensor, &grad_output, input_codes.1)
            .expect("fused lowrank grad weight");

    let decoder_grad_input_ref = packed_decoder_tail_grad_input_device_reference(
        residual_grad_output.clone(),
        decoder_weight_codes_tensor.clone(),
        decoder_weight_codes.1,
        4,
        128,
    );
    let decoder_grad_input_fused = try_fused_packed_decoder_tail_grad_input(
        &residual_grad_output,
        &decoder_weight_codes_tensor,
        decoder_weight_codes.1,
        4,
        128,
    )
    .expect("fused decoder grad input");

    let decoder_grad_weight_ref = packed_decoder_tail_grad_weight_device_reference(
        y_codes_tensor.clone(),
        residual_grad_output.clone(),
        y_codes.1,
    );
    let decoder_grad_weight_fused = try_fused_packed_decoder_tail_grad_weight(
        &y_codes_tensor,
        &residual_grad_output,
        y_codes.1,
    )
    .expect("fused decoder grad weight");

    let lowrank_forward_ref_ms = bench_case(
        || {
            packed_lowrank_projection_device_reference(
                input_codes_tensor.clone().float().mul_scalar(input_codes.1),
                lowrank_weight_codes_tensor.clone(),
                lowrank_weight_codes.1,
                lowrank_weight_shape[2],
            )
        },
        iters,
    );
    let lowrank_forward_fused_ms = bench_case(
        || {
            try_fused_packed_lowrank_projection(
                &input_codes_tensor,
                &lowrank_weight_codes_tensor,
                input_codes.1,
                lowrank_weight_codes.1,
                lowrank_weight_shape[2],
            )
            .expect("fused lowrank forward")
        },
        iters,
    );
    let lowrank_forward_cube_ms = bench_case(
        || {
            try_cube_fused_packed_lowrank_projection_wgpu(
                &input_codes_tensor,
                &lowrank_weight_codes_tensor,
                input_codes.1,
                lowrank_weight_codes.1,
                lowrank_weight_shape[2],
            )
            .expect("cube lowrank forward")
        },
        iters,
    );
    let lowrank_forward_packed_dot_ms = lowrank_forward_packed_dot.as_ref().ok().map(|_| {
        bench_case(
            || {
                try_wgpu_packed_dot_lowrank_projection(
                    &input_codes_tensor,
                    &lowrank_weight_codes_tensor,
                    input_codes.1,
                    lowrank_weight_codes.1,
                    lowrank_weight_shape[2],
                )
                .expect("packed-dot lowrank forward")
            },
            iters,
        )
    });
    let decoder_forward_ref_ms = bench_case(
        || {
            packed_decoder_tail_device_reference(
                y_codes_tensor.clone().float().mul_scalar(y_codes.1),
                decoder_weight_codes_tensor.clone(),
                decoder_weight_codes.1,
            )
        },
        iters,
    );
    let decoder_forward_fused_ms = bench_case(
        || {
            try_fused_packed_decoder_tail(
                &y_codes_tensor,
                &decoder_weight_codes_tensor,
                y_codes.1,
                decoder_weight_codes.1,
            )
            .expect("fused decoder tail")
        },
        iters,
    );
    let decoder_forward_cube_ms = bench_case(
        || {
            try_cube_fused_packed_decoder_tail_wgpu(
                &y_codes_tensor,
                &decoder_weight_codes_tensor,
                y_codes.1,
                decoder_weight_codes.1,
            )
            .expect("cube decoder tail")
        },
        iters,
    );
    let decoder_forward_packed_dot_ms = decoder_forward_packed_dot.as_ref().ok().map(|_| {
        bench_case(
            || {
                try_wgpu_packed_dot_decoder_tail(
                    &y_codes_tensor,
                    &decoder_weight_codes_tensor,
                    y_codes.1,
                    decoder_weight_codes.1,
                )
                .expect("packed-dot decoder tail")
            },
            iters,
        )
    });
    let lowrank_grad_input_ref_ms = bench_case(
        || {
            packed_lowrank_grad_input_device_reference(
                grad_output.clone(),
                lowrank_weight_codes_tensor.clone(),
                lowrank_weight_codes.1,
                1,
            )
        },
        iters,
    );
    let lowrank_grad_input_fused_ms = bench_case(
        || {
            try_fused_packed_lowrank_grad_input(
                &grad_output,
                &lowrank_weight_codes_tensor,
                lowrank_weight_codes.1,
                1,
            )
            .expect("fused lowrank grad input")
        },
        iters,
    );
    let lowrank_grad_weight_ref_ms = bench_case(
        || {
            packed_lowrank_grad_weight_device_reference(
                input_codes_tensor.clone(),
                grad_output.clone(),
                input_codes.1,
            )
        },
        iters,
    );
    let lowrank_grad_weight_fused_ms = bench_case(
        || {
            try_fused_packed_lowrank_grad_weight(&input_codes_tensor, &grad_output, input_codes.1)
                .expect("fused lowrank grad weight")
        },
        iters,
    );
    let decoder_grad_input_ref_ms = bench_case(
        || {
            packed_decoder_tail_grad_input_device_reference(
                residual_grad_output.clone(),
                decoder_weight_codes_tensor.clone(),
                decoder_weight_codes.1,
                4,
                128,
            )
        },
        iters,
    );
    let decoder_grad_input_fused_ms = bench_case(
        || {
            try_fused_packed_decoder_tail_grad_input(
                &residual_grad_output,
                &decoder_weight_codes_tensor,
                decoder_weight_codes.1,
                4,
                128,
            )
            .expect("fused decoder grad input")
        },
        iters,
    );
    let decoder_grad_weight_ref_ms = bench_case_2d(
        || {
            packed_decoder_tail_grad_weight_device_reference(
                y_codes_tensor.clone(),
                residual_grad_output.clone(),
                y_codes.1,
            )
        },
        iters,
    );
    let decoder_grad_weight_fused_ms = bench_case_2d(
        || {
            try_fused_packed_decoder_tail_grad_weight(
                &y_codes_tensor,
                &residual_grad_output,
                y_codes.1,
            )
            .expect("fused decoder grad weight")
        },
        iters,
    );
    let rho_pack_ms = bench_case_2d(
        || {
            let packed = pack_rho_int8_block_device_reference(rho.clone(), 32);
            unpack_rho_int8_block_device_reference(packed.packed, packed.scales, rho_shape, 32)
                .reshape([rho_shape[0] * rho_shape[1], rho_shape[2] * rho_shape[3]])
        },
        iters,
    );
    let lowrank_quantize_pack_ref_ms = {
        let input_values = input_values.clone();
        bench_case_int4(
            || {
                let (codes, _) = quantize_signed_values(&input_values);
                let packed = pack_lowrank_input_codes_i8x4(
                    &codes.iter().map(|value| *value as i8).collect::<Vec<_>>(),
                    input_shape[0],
                    input_shape[1],
                    input_shape[2],
                    input_shape[3],
                );
                int_tensor_from_values(
                    packed,
                    [
                        input_shape[0],
                        input_shape[1],
                        input_shape[2],
                        input_shape[3].div_ceil(4),
                    ],
                    &device,
                )
            },
            iters,
        )
    };
    let lowrank_quantize_pack_wgpu_ms = bench_case_int4(
        || {
            diagnose_wgpu_quantize_pack_activation_i8x4(
                &input_codes_tensor.clone().float().mul_scalar(input_codes.1),
                &input_scale_tensor,
                127,
                false,
            )
            .expect("wgpu quantize-pack activation")
        },
        iters,
    );

    println!("low_bit_bench backend=wgpu iters={iters}");
    println!(
        "forward.lowrank ms_ref={:.3} ms_fused={:.3} speedup={:.2} max_abs_diff={:.6}",
        lowrank_forward_ref_ms,
        lowrank_forward_fused_ms,
        lowrank_forward_ref_ms / lowrank_forward_fused_ms,
        max_abs_diff(lowrank_forward_fused, lowrank_forward_ref),
    );
    println!(
        "forward.lowrank_cube ms_ref={:.3} ms_cube={:.3} speedup={:.2} max_abs_diff={:.6}",
        lowrank_forward_ref_ms,
        lowrank_forward_cube_ms,
        lowrank_forward_ref_ms / lowrank_forward_cube_ms,
        max_abs_diff(
            lowrank_forward_cube,
            packed_lowrank_projection_device_reference(
                input_codes_tensor.clone().float().mul_scalar(input_codes.1),
                lowrank_weight_codes_tensor.clone(),
                lowrank_weight_codes.1,
                lowrank_weight_shape[2],
            )
        ),
    );
    match (lowrank_forward_packed_dot_ms, lowrank_forward_packed_dot) {
        (Some(ms), Ok(actual)) => {
            println!(
                "forward.lowrank_packed_dot ms_ref={:.3} ms_packed_dot={:.3} speedup={:.2} max_abs_diff={:.6}",
                lowrank_forward_ref_ms,
                ms,
                lowrank_forward_ref_ms / ms,
                max_abs_diff(
                    actual,
                    packed_lowrank_projection_device_reference(
                        input_codes_tensor.clone().float().mul_scalar(input_codes.1),
                        lowrank_weight_codes_tensor.clone(),
                        lowrank_weight_codes.1,
                        lowrank_weight_shape[2],
                    )
                ),
            );
        }
        (_, Err(reason)) => {
            println!("forward.lowrank_packed_dot unavailable reason={reason}");
        }
        _ => {
            println!("forward.lowrank_packed_dot unavailable reason=bench gating skipped");
        }
    }
    println!(
        "forward.decoder_tail ms_ref={:.3} ms_fused={:.3} speedup={:.2} max_abs_diff={:.6}",
        decoder_forward_ref_ms,
        decoder_forward_fused_ms,
        decoder_forward_ref_ms / decoder_forward_fused_ms,
        max_abs_diff(decoder_forward_fused, decoder_forward_ref),
    );
    println!(
        "forward.decoder_tail_cube ms_ref={:.3} ms_cube={:.3} speedup={:.2} max_abs_diff={:.6}",
        decoder_forward_ref_ms,
        decoder_forward_cube_ms,
        decoder_forward_ref_ms / decoder_forward_cube_ms,
        max_abs_diff(
            decoder_forward_cube,
            packed_decoder_tail_device_reference(
                y_codes_tensor.clone().float().mul_scalar(y_codes.1),
                decoder_weight_codes_tensor.clone(),
                decoder_weight_codes.1,
            )
        ),
    );
    match (decoder_forward_packed_dot_ms, decoder_forward_packed_dot) {
        (Some(ms), Ok(actual)) => {
            println!(
                "forward.decoder_tail_packed_dot ms_ref={:.3} ms_packed_dot={:.3} speedup={:.2} max_abs_diff={:.6}",
                decoder_forward_ref_ms,
                ms,
                decoder_forward_ref_ms / ms,
                max_abs_diff(
                    actual,
                    packed_decoder_tail_device_reference(
                        y_codes_tensor.clone().float().mul_scalar(y_codes.1),
                        decoder_weight_codes_tensor.clone(),
                        decoder_weight_codes.1,
                    )
                ),
            );
        }
        (_, Err(reason)) => {
            println!("forward.decoder_tail_packed_dot unavailable reason={reason}");
        }
        _ => {
            println!("forward.decoder_tail_packed_dot unavailable reason=bench gating skipped");
        }
    }
    println!(
        "backward.lowrank_grad_input ms_ref={:.3} ms_fused={:.3} speedup={:.2} max_abs_diff={:.6}",
        lowrank_grad_input_ref_ms,
        lowrank_grad_input_fused_ms,
        lowrank_grad_input_ref_ms / lowrank_grad_input_fused_ms,
        max_abs_diff(lowrank_grad_input_fused, lowrank_grad_input_ref),
    );
    println!(
        "backward.lowrank_grad_weight ms_ref={:.3} ms_fused={:.3} speedup={:.2} max_abs_diff={:.6}",
        lowrank_grad_weight_ref_ms,
        lowrank_grad_weight_fused_ms,
        lowrank_grad_weight_ref_ms / lowrank_grad_weight_fused_ms,
        max_abs_diff(lowrank_grad_weight_fused, lowrank_grad_weight_ref),
    );
    println!(
        "backward.decoder_tail_grad_input ms_ref={:.3} ms_fused={:.3} speedup={:.2} max_abs_diff={:.6}",
        decoder_grad_input_ref_ms,
        decoder_grad_input_fused_ms,
        decoder_grad_input_ref_ms / decoder_grad_input_fused_ms,
        max_abs_diff(decoder_grad_input_fused, decoder_grad_input_ref),
    );
    println!(
        "backward.decoder_tail_grad_weight ms_ref={:.3} ms_fused={:.3} speedup={:.2} max_abs_diff={:.6}",
        decoder_grad_weight_ref_ms,
        decoder_grad_weight_fused_ms,
        decoder_grad_weight_ref_ms / decoder_grad_weight_fused_ms,
        max_abs_diff(decoder_grad_weight_fused, decoder_grad_weight_ref),
    );
    println!("rho.pack_roundtrip ms={rho_pack_ms:.3}");
    println!(
        "forward.lowrank_quantize_pack ms_ref={:.3} ms_wgpu={:.3} speedup={:.2}",
        lowrank_quantize_pack_ref_ms,
        lowrank_quantize_pack_wgpu_ms,
        lowrank_quantize_pack_ref_ms / lowrank_quantize_pack_wgpu_ms,
    );
}
