#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("low_bit_cuda_bench requires --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
mod app {
    use std::process::Command;
    use std::time::Instant;

    use burn::tensor::backend::Backend as BackendTrait;
    use burn::tensor::{Int, Tensor, TensorData};
    use burn_cubecl::CubeBackend;
    use burn_cubecl::cubecl::Runtime;
    use burn_cubecl::cubecl::cuda::CudaRuntime;
    use burn_dragon_kernel::api::low_bit::{
        pack_decoder_input_codes_i8x4, pack_decoder_weight_codes_i8x4,
        pack_lowrank_input_codes_i8x4, pack_lowrank_weight_codes_i8x4,
        pack_rho_int8_block_device_reference, packed_decoder_tail_device_reference,
        packed_decoder_tail_grad_input_device_reference,
        packed_decoder_tail_grad_input_from_float_decoder_cuda,
        packed_decoder_tail_grad_weight_device_reference,
        packed_lowrank_grad_input_device_reference,
        packed_lowrank_grad_input_from_float_weight_cuda,
        packed_lowrank_grad_input_from_transposed_float_weight_cuda,
        packed_lowrank_grad_weight_device_reference, packed_lowrank_projection_device_reference,
        try_fused_packed_decoder_tail, try_fused_packed_decoder_tail_grad_input,
        try_fused_packed_decoder_tail_grad_weight, try_fused_packed_lowrank_grad_input,
        try_fused_packed_lowrank_grad_weight, try_fused_packed_lowrank_projection,
        try_raw_cuda_packed_decoder_tail_grad_input, try_raw_cuda_packed_decoder_tail_grad_weight,
        try_raw_cuda_packed_decoder_tail_prepacked_input, try_raw_cuda_packed_lowrank_grad_input,
        try_raw_cuda_packed_lowrank_grad_weight,
        try_raw_cuda_packed_lowrank_projection_prepacked_input,
        unpack_rho_int8_block_device_reference,
    };
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

    type BenchBackend = CubeBackend<CudaRuntime, f32, i32, u8>;
    const CUDA_RAW_DOT_SRC: &str = r#"
extern "C" __global__ void packed_lowrank_dp4a(
    const int* input_packed,
    const int* weight_packed,
    float* output,
    int batch,
    int input_heads,
    int heads,
    int time,
    int pack_len,
    int latent,
    float activation_scale,
    float weight_scale
) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    int bh = blockIdx.z;
    if (l >= latent || t >= time || bh >= batch * heads) {
        return;
    }
    int h = bh % heads;
    int b = bh / heads;
    int input_head = input_heads == 1 ? 0 : h;
    int acc = 0;
    int input_base = ((b * input_heads + input_head) * time + t) * pack_len;
    int weight_base = (h * pack_len) * latent + l;
    #pragma unroll 4
    for (int p = 0; p < pack_len; ++p) {
        acc = __dp4a(input_packed[input_base + p], weight_packed[weight_base + p * latent], acc);
    }
    output[((b * heads + h) * time + t) * latent + l] = (float)acc * activation_scale * weight_scale;
}

extern "C" __global__ void packed_decoder_tail_dp4a(
    const int* y_packed,
    const int* weight_packed,
    float* output,
    int batch,
    int heads,
    int time,
    int pack_len,
    int dim,
    float activation_scale,
    float weight_scale
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    int b = blockIdx.z;
    if (d >= dim || t >= time || b >= batch) {
        return;
    }
    int acc = 0;
    for (int h = 0; h < heads; ++h) {
        int input_base = ((b * heads + h) * time + t) * pack_len;
        int weight_base = (h * pack_len) * dim + d;
        #pragma unroll 4
        for (int p = 0; p < pack_len; ++p) {
            acc = __dp4a(y_packed[input_base + p], weight_packed[weight_base + p * dim], acc);
        }
    }
    output[(b * time + t) * dim + d] = (float)acc * activation_scale * weight_scale;
}
"#;

    #[derive(Clone, Copy)]
    struct MemorySnapshot {
        reserved: u64,
        in_use: u64,
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

    fn max_abs_diff<const D: usize>(
        lhs: Tensor<BenchBackend, D>,
        rhs: Tensor<BenchBackend, D>,
    ) -> f32 {
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

    fn bench_env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(default)
    }

    fn median_ms(mut samples: Vec<f64>) -> f64 {
        samples.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
        samples[samples.len() / 2]
    }

    fn bench_case(mut func: impl FnMut() -> Tensor<BenchBackend, 4>, iters: usize) -> f64 {
        let warmup = bench_env_usize("LOW_BIT_BENCH_WARMUP", 5);
        let samples = bench_env_usize("LOW_BIT_BENCH_SAMPLES", 5);
        for _ in 0..warmup {
            let _ = func().sum().into_data();
        }
        median_ms(
            (0..samples)
                .map(|_| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let _ = func().sum().into_data();
                    }
                    start.elapsed().as_secs_f64() * 1000.0 / iters as f64
                })
                .collect(),
        )
    }

    fn bench_case_2d(mut func: impl FnMut() -> Tensor<BenchBackend, 2>, iters: usize) -> f64 {
        let warmup = bench_env_usize("LOW_BIT_BENCH_WARMUP", 5);
        let samples = bench_env_usize("LOW_BIT_BENCH_SAMPLES", 5);
        for _ in 0..warmup {
            let _ = func().sum().into_data();
        }
        median_ms(
            (0..samples)
                .map(|_| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let _ = func().sum().into_data();
                    }
                    start.elapsed().as_secs_f64() * 1000.0 / iters as f64
                })
                .collect(),
        )
    }

    fn bench_case_sync<const D: usize>(
        device: &<BenchBackend as BackendTrait>::Device,
        mut func: impl FnMut() -> Tensor<BenchBackend, D>,
        iters: usize,
    ) -> f64 {
        let warmup = bench_env_usize("LOW_BIT_BENCH_WARMUP", 5);
        let samples = bench_env_usize("LOW_BIT_BENCH_SAMPLES", 5);
        for _ in 0..warmup {
            let output = func();
            core::hint::black_box(output.shape());
            <BenchBackend as BackendTrait>::sync(device).expect("cuda bench sync warmup");
        }
        median_ms(
            (0..samples)
                .map(|_| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let output = func();
                        core::hint::black_box(output.shape());
                        <BenchBackend as BackendTrait>::sync(device).expect("cuda bench sync");
                    }
                    start.elapsed().as_secs_f64() * 1000.0 / iters as f64
                })
                .collect(),
        )
    }

    fn bench_raw_sync(mut func: impl FnMut(), iters: usize) -> f64 {
        let warmup = bench_env_usize("LOW_BIT_BENCH_WARMUP", 5);
        let samples = bench_env_usize("LOW_BIT_BENCH_SAMPLES", 5);
        for _ in 0..warmup {
            func();
        }
        median_ms(
            (0..samples)
                .map(|_| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        func();
                    }
                    start.elapsed().as_secs_f64() * 1000.0 / iters as f64
                })
                .collect(),
        )
    }

    fn detect_cuda_arch() -> String {
        if let Ok(value) = std::env::var("LOW_BIT_CUDA_NVRTC_ARCH") {
            if !value.trim().is_empty() {
                return value;
            }
        }
        if let Ok(output) = Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output()
        {
            if output.status.success() {
                if let Some(line) = String::from_utf8_lossy(&output.stdout).lines().next() {
                    let digits: String = line.chars().filter(|ch| ch.is_ascii_digit()).collect();
                    if digits.len() >= 2 {
                        return format!("compute_{digits}");
                    }
                }
            }
        }
        "compute_61".to_string()
    }

    fn pack_i8x4(v0: i32, v1: i32, v2: i32, v3: i32) -> i32 {
        let to_byte = |value: i32| (value.clamp(-127, 127) & 0xff) as u32;
        (to_byte(v0) | (to_byte(v1) << 8) | (to_byte(v2) << 16) | (to_byte(v3) << 24)) as i32
    }

    fn pack_lowrank_input_codes(
        codes: &[i32],
        batch: usize,
        input_heads: usize,
        time: usize,
        embd: usize,
    ) -> Vec<i32> {
        let pack_len = embd.div_ceil(4);
        let mut packed = vec![0i32; batch * input_heads * time * pack_len];
        for b in 0..batch {
            for h in 0..input_heads {
                for t in 0..time {
                    let base = ((b * input_heads + h) * time + t) * embd;
                    let out_base = ((b * input_heads + h) * time + t) * pack_len;
                    for p in 0..pack_len {
                        let e = p * 4;
                        let v0 = *codes.get(base + e).unwrap_or(&0);
                        let v1 = *codes.get(base + e + 1).unwrap_or(&0);
                        let v2 = *codes.get(base + e + 2).unwrap_or(&0);
                        let v3 = *codes.get(base + e + 3).unwrap_or(&0);
                        packed[out_base + p] = pack_i8x4(v0, v1, v2, v3);
                    }
                }
            }
        }
        packed
    }

    fn pack_lowrank_weight_codes(
        codes: &[i32],
        heads: usize,
        embd: usize,
        latent: usize,
    ) -> Vec<i32> {
        let pack_len = embd.div_ceil(4);
        let mut packed = vec![0i32; heads * pack_len * latent];
        for h in 0..heads {
            for p in 0..pack_len {
                for l in 0..latent {
                    let e = p * 4;
                    let v0 = if e < embd {
                        codes[(h * embd + e) * latent + l]
                    } else {
                        0
                    };
                    let v1 = if e + 1 < embd {
                        codes[(h * embd + e + 1) * latent + l]
                    } else {
                        0
                    };
                    let v2 = if e + 2 < embd {
                        codes[(h * embd + e + 2) * latent + l]
                    } else {
                        0
                    };
                    let v3 = if e + 3 < embd {
                        codes[(h * embd + e + 3) * latent + l]
                    } else {
                        0
                    };
                    packed[(h * pack_len + p) * latent + l] = pack_i8x4(v0, v1, v2, v3);
                }
            }
        }
        packed
    }

    fn pack_decoder_input_codes(
        codes: &[i32],
        batch: usize,
        heads: usize,
        time: usize,
        latent: usize,
    ) -> Vec<i32> {
        let pack_len = latent.div_ceil(4);
        let mut packed = vec![0i32; batch * heads * time * pack_len];
        for b in 0..batch {
            for h in 0..heads {
                for t in 0..time {
                    let base = ((b * heads + h) * time + t) * latent;
                    let out_base = ((b * heads + h) * time + t) * pack_len;
                    for p in 0..pack_len {
                        let l = p * 4;
                        let v0 = *codes.get(base + l).unwrap_or(&0);
                        let v1 = *codes.get(base + l + 1).unwrap_or(&0);
                        let v2 = *codes.get(base + l + 2).unwrap_or(&0);
                        let v3 = *codes.get(base + l + 3).unwrap_or(&0);
                        packed[out_base + p] = pack_i8x4(v0, v1, v2, v3);
                    }
                }
            }
        }
        packed
    }

    fn pack_decoder_weight_codes(
        codes: &[i32],
        heads: usize,
        latent_per_head: usize,
        dim: usize,
    ) -> Vec<i32> {
        let pack_len = latent_per_head.div_ceil(4);
        let mut packed = vec![0i32; heads * pack_len * dim];
        for h in 0..heads {
            for p in 0..pack_len {
                for d in 0..dim {
                    let l = p * 4;
                    let row0 = h * latent_per_head + l;
                    let row1 = h * latent_per_head + l + 1;
                    let row2 = h * latent_per_head + l + 2;
                    let row3 = h * latent_per_head + l + 3;
                    let v0 = if l < latent_per_head {
                        codes[row0 * dim + d]
                    } else {
                        0
                    };
                    let v1 = if l + 1 < latent_per_head {
                        codes[row1 * dim + d]
                    } else {
                        0
                    };
                    let v2 = if l + 2 < latent_per_head {
                        codes[row2 * dim + d]
                    } else {
                        0
                    };
                    let v3 = if l + 3 < latent_per_head {
                        codes[row3 * dim + d]
                    } else {
                        0
                    };
                    packed[(h * pack_len + p) * dim + d] = pack_i8x4(v0, v1, v2, v3);
                }
            }
        }
        packed
    }

    fn max_abs_diff_host(lhs: &[f32], rhs: &[f32]) -> f32 {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max)
    }

    fn memory_snapshot(device: &<BenchBackend as BackendTrait>::Device) -> MemorySnapshot {
        let usage = <CudaRuntime as Runtime>::client(device)
            .memory_usage()
            .expect("cuda memory usage");
        MemorySnapshot {
            reserved: usage.bytes_reserved,
            in_use: usage.bytes_in_use,
        }
    }

    pub fn main() {
        let iters = std::env::args()
            .skip(1)
            .find_map(|arg| {
                arg.strip_prefix("--iters=")
                    .and_then(|value| value.parse::<usize>().ok())
            })
            .unwrap_or(20);

        let device = <BenchBackend as BackendTrait>::Device::default();

        let input_shape = [8, 1, 128, 256];
        let lowrank_weight_shape = [4, 256, 128];
        let decoder_weight_shape = [512, 256];
        let y_shape = [8, 4, 128, 128];
        let grad_shape = [8, 4, 128, 128];
        let residual_grad_shape = [8, 1, 128, 256];
        let rho_shape = [8, 4, 128, 128];

        let input_values = deterministic_values(input_shape.iter().product(), 0.2);
        let lowrank_weight_values =
            deterministic_values(lowrank_weight_shape.iter().product(), 0.7);
        let y_values = deterministic_values(y_shape.iter().product(), 0.5);
        let decoder_weight_values =
            deterministic_values(decoder_weight_shape.iter().product(), 1.0);
        let grad_values = deterministic_values(grad_shape.iter().product(), 1.4);
        let residual_grad_values = deterministic_values(residual_grad_shape.iter().product(), 1.7);
        let rho_values = deterministic_values(rho_shape.iter().product(), 2.1);

        let input_codes = quantize_signed_values(&input_values);
        let lowrank_weight_codes = quantize_signed_values(&lowrank_weight_values);
        let y_codes = quantize_signed_values(&y_values);
        let decoder_weight_codes = quantize_signed_values(&decoder_weight_values);

        let input_codes_tensor =
            int_tensor_from_values(input_codes.0.clone(), input_shape, &device);
        let input_codes_packed_tensor = int_tensor_from_values::<4>(
            pack_lowrank_input_codes_i8x4(
                &input_codes
                    .0
                    .iter()
                    .map(|value| *value as i8)
                    .collect::<Vec<_>>(),
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            ),
            [
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3].div_ceil(4),
            ],
            &device,
        );
        let lowrank_weight_codes_tensor = int_tensor_from_values(
            lowrank_weight_codes.0.clone(),
            lowrank_weight_shape,
            &device,
        );
        let lowrank_weight_float_tensor = lowrank_weight_codes_tensor
            .clone()
            .float()
            .mul_scalar(lowrank_weight_codes.1)
            .reshape([
                1,
                lowrank_weight_shape[0],
                lowrank_weight_shape[1],
                lowrank_weight_shape[2],
            ]);
        let lowrank_weight_transposed_float_tensor = lowrank_weight_float_tensor
            .clone()
            .reshape([
                lowrank_weight_shape[0],
                lowrank_weight_shape[1],
                lowrank_weight_shape[2],
            ])
            .swap_dims(1, 2);
        let y_codes_tensor = int_tensor_from_values(y_codes.0.clone(), y_shape, &device);
        let y_codes_packed_tensor = int_tensor_from_values::<4>(
            pack_decoder_input_codes_i8x4(
                &y_codes
                    .0
                    .iter()
                    .map(|value| *value as i8)
                    .collect::<Vec<_>>(),
                y_shape[0],
                y_shape[1],
                y_shape[2],
                y_shape[3],
            ),
            [y_shape[0], y_shape[1], y_shape[2], y_shape[3].div_ceil(4)],
            &device,
        );
        let decoder_weight_codes_tensor = int_tensor_from_values(
            decoder_weight_codes.0.clone(),
            decoder_weight_shape,
            &device,
        );
        let decoder_weight_float_tensor = decoder_weight_codes_tensor
            .clone()
            .float()
            .mul_scalar(decoder_weight_codes.1);
        let lowrank_weight_packed_tensor = int_tensor_from_values::<3>(
            pack_lowrank_weight_codes_i8x4(
                &lowrank_weight_codes
                    .0
                    .iter()
                    .map(|value| *value as i8)
                    .collect::<Vec<_>>(),
                lowrank_weight_shape[0],
                lowrank_weight_shape[1],
                lowrank_weight_shape[2],
            ),
            [
                lowrank_weight_shape[0],
                lowrank_weight_shape[1].div_ceil(4),
                lowrank_weight_shape[2],
            ],
            &device,
        );
        let decoder_weight_packed_tensor = int_tensor_from_values::<2>(
            pack_decoder_weight_codes_i8x4(
                &decoder_weight_codes
                    .0
                    .iter()
                    .map(|value| *value as i8)
                    .collect::<Vec<_>>(),
                y_shape[1],
                y_shape[3],
                decoder_weight_shape[1],
            ),
            [y_shape[1] * y_shape[3].div_ceil(4), decoder_weight_shape[1]],
            &device,
        );
        let grad_output = tensor_from_values(grad_values, grad_shape, &device);
        let residual_grad_output =
            tensor_from_values(residual_grad_values, residual_grad_shape, &device);
        let rho = tensor_from_values(rho_values, rho_shape, &device);

        let memory_before = memory_snapshot(&device);

        let lowrank_forward_ref = packed_lowrank_projection_device_reference(
            input_codes_tensor.clone().float().mul_scalar(input_codes.1),
            lowrank_weight_codes_tensor.clone(),
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
        let lowrank_forward_raw_runtime = try_raw_cuda_packed_lowrank_projection_prepacked_input(
            &input_codes_packed_tensor,
            &lowrank_weight_packed_tensor,
            input_codes.1,
            lowrank_weight_codes.1,
            lowrank_weight_shape[2],
        )
        .expect("raw runtime lowrank forward");

        let decoder_forward_ref = packed_decoder_tail_device_reference(
            y_codes_tensor.clone().float().mul_scalar(y_codes.1),
            decoder_weight_codes_tensor.clone(),
            decoder_weight_codes.1,
        );
        let decoder_forward_fused = try_fused_packed_decoder_tail(
            &y_codes_tensor,
            &decoder_weight_codes_tensor,
            y_codes.1,
            decoder_weight_codes.1,
        )
        .expect("fused decoder tail");
        let decoder_forward_raw_runtime = try_raw_cuda_packed_decoder_tail_prepacked_input(
            &y_codes_packed_tensor,
            &decoder_weight_packed_tensor,
            y_codes.1,
            decoder_weight_codes.1,
        )
        .expect("raw runtime decoder tail");

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
        let lowrank_grad_input_raw = try_raw_cuda_packed_lowrank_grad_input(
            &grad_output,
            &lowrank_weight_codes_tensor,
            lowrank_weight_codes.1,
            1,
        )
        .expect("raw lowrank grad input");
        let lowrank_grad_input_cached_float = packed_lowrank_grad_input_from_float_weight_cuda(
            grad_output.clone(),
            lowrank_weight_float_tensor.clone(),
            1,
        );
        let lowrank_grad_input_cached_transposed =
            packed_lowrank_grad_input_from_transposed_float_weight_cuda(
                grad_output.clone(),
                lowrank_weight_transposed_float_tensor.clone(),
                1,
            );

        let lowrank_grad_weight_ref = packed_lowrank_grad_weight_device_reference(
            input_codes_tensor.clone(),
            grad_output.clone(),
            input_codes.1,
        );
        let lowrank_grad_weight_fused =
            try_fused_packed_lowrank_grad_weight(&input_codes_tensor, &grad_output, input_codes.1)
                .expect("fused lowrank grad weight");
        let lowrank_grad_weight_raw = try_raw_cuda_packed_lowrank_grad_weight(
            &input_codes_tensor,
            &grad_output,
            input_codes.1,
        )
        .expect("raw lowrank grad weight");

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
        let decoder_grad_input_raw = try_raw_cuda_packed_decoder_tail_grad_input(
            &residual_grad_output,
            &decoder_weight_codes_tensor,
            decoder_weight_codes.1,
            4,
            128,
        )
        .expect("raw decoder grad input");
        let decoder_grad_input_cached_float =
            packed_decoder_tail_grad_input_from_float_decoder_cuda(
                residual_grad_output.clone(),
                decoder_weight_float_tensor.clone(),
                4,
                128,
            );

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
        let decoder_grad_weight_raw = try_raw_cuda_packed_decoder_tail_grad_weight(
            &y_codes_tensor,
            &residual_grad_output,
            y_codes.1,
        )
        .expect("raw decoder grad weight");

        let raw_cuda_arch = Box::leak(detect_cuda_arch().into_boxed_str());
        let raw_cuda_ptx = compile_ptx_with_opts(
            CUDA_RAW_DOT_SRC,
            CompileOptions {
                arch: Some(raw_cuda_arch),
                fmad: Some(true),
                ..Default::default()
            },
        )
        .expect("compile raw cuda low-bit kernels");
        let raw_cuda_ctx = CudaContext::new(0).expect("cuda context");
        let raw_cuda_stream = raw_cuda_ctx.default_stream();
        let raw_cuda_module = raw_cuda_ctx
            .load_module(raw_cuda_ptx)
            .expect("load raw module");
        let raw_lowrank_fn = raw_cuda_module
            .load_function("packed_lowrank_dp4a")
            .expect("raw lowrank function");
        let raw_decoder_fn = raw_cuda_module
            .load_function("packed_decoder_tail_dp4a")
            .expect("raw decoder function");

        let input_packed_host = pack_lowrank_input_codes(
            &input_codes.0,
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let lowrank_weight_packed_host = pack_lowrank_weight_codes(
            &lowrank_weight_codes.0,
            lowrank_weight_shape[0],
            lowrank_weight_shape[1],
            lowrank_weight_shape[2],
        );
        let y_packed_host =
            pack_decoder_input_codes(&y_codes.0, y_shape[0], y_shape[1], y_shape[2], y_shape[3]);
        let decoder_weight_packed_host = pack_decoder_weight_codes(
            &decoder_weight_codes.0,
            y_shape[1],
            y_shape[3],
            decoder_weight_shape[1],
        );
        let input_packed_dev = raw_cuda_stream
            .memcpy_stod(&input_packed_host)
            .expect("copy packed input");
        let lowrank_weight_packed_dev = raw_cuda_stream
            .memcpy_stod(&lowrank_weight_packed_host)
            .expect("copy packed lowrank weight");
        let y_packed_dev = raw_cuda_stream
            .memcpy_stod(&y_packed_host)
            .expect("copy packed y");
        let decoder_weight_packed_dev = raw_cuda_stream
            .memcpy_stod(&decoder_weight_packed_host)
            .expect("copy packed decoder weight");
        let lowrank_pack_len = input_shape[3].div_ceil(4);
        let decoder_pack_len = y_shape[3].div_ceil(4);
        let lowrank_output_len =
            input_shape[0] * lowrank_weight_shape[0] * input_shape[2] * lowrank_weight_shape[2];
        let decoder_output_len = y_shape[0] * y_shape[2] * decoder_weight_shape[1];
        let lowrank_raw_launch_cfg = LaunchConfig {
            grid_dim: (
                lowrank_weight_shape[2].div_ceil(128) as u32,
                input_shape[2] as u32,
                (input_shape[0] * lowrank_weight_shape[0]) as u32,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let decoder_raw_launch_cfg = LaunchConfig {
            grid_dim: (
                decoder_weight_shape[1].div_ceil(128) as u32,
                y_shape[2] as u32,
                y_shape[0] as u32,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lowrank_output_dev = raw_cuda_stream
            .memcpy_stod(&vec![0.0f32; lowrank_output_len])
            .expect("alloc raw lowrank out");
        let mut decoder_output_dev = raw_cuda_stream
            .memcpy_stod(&vec![0.0f32; decoder_output_len])
            .expect("alloc raw decoder out");
        let launch_raw_lowrank_sync = |output_dev: &mut _| {
            let batch_i32 = input_shape[0] as i32;
            let heads_i32 = input_shape[1] as i32;
            let seq_i32 = lowrank_weight_shape[0] as i32;
            let tokens_i32 = input_shape[2] as i32;
            let pack_len_i32 = lowrank_pack_len as i32;
            let latent_out_i32 = lowrank_weight_shape[2] as i32;
            let mut builder = raw_cuda_stream.launch_builder(&raw_lowrank_fn);
            builder.arg(&input_packed_dev);
            builder.arg(&lowrank_weight_packed_dev);
            builder.arg(output_dev);
            builder.arg(&batch_i32);
            builder.arg(&heads_i32);
            builder.arg(&seq_i32);
            builder.arg(&tokens_i32);
            builder.arg(&pack_len_i32);
            builder.arg(&latent_out_i32);
            builder.arg(&input_codes.1);
            builder.arg(&lowrank_weight_codes.1);
            unsafe { builder.launch(lowrank_raw_launch_cfg) }.expect("launch raw lowrank");
            raw_cuda_stream.synchronize().expect("sync raw lowrank");
        };
        let launch_raw_decoder_sync = |output_dev: &mut _| {
            let batch_i32 = y_shape[0] as i32;
            let heads_i32 = y_shape[1] as i32;
            let tokens_i32 = y_shape[2] as i32;
            let pack_len_i32 = decoder_pack_len as i32;
            let residual_dim_i32 = decoder_weight_shape[1] as i32;
            let mut builder = raw_cuda_stream.launch_builder(&raw_decoder_fn);
            builder.arg(&y_packed_dev);
            builder.arg(&decoder_weight_packed_dev);
            builder.arg(output_dev);
            builder.arg(&batch_i32);
            builder.arg(&heads_i32);
            builder.arg(&tokens_i32);
            builder.arg(&pack_len_i32);
            builder.arg(&residual_dim_i32);
            builder.arg(&y_codes.1);
            builder.arg(&decoder_weight_codes.1);
            unsafe { builder.launch(decoder_raw_launch_cfg) }.expect("launch raw decoder");
            raw_cuda_stream.synchronize().expect("sync raw decoder");
        };
        launch_raw_lowrank_sync(&mut lowrank_output_dev);
        let lowrank_raw_host = raw_cuda_stream
            .memcpy_dtov(&lowrank_output_dev)
            .expect("read raw lowrank");
        launch_raw_decoder_sync(&mut decoder_output_dev);
        let decoder_raw_host = raw_cuda_stream
            .memcpy_dtov(&decoder_output_dev)
            .expect("read raw decoder");

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
        let lowrank_grad_input_raw_ms = bench_case(
            || {
                try_raw_cuda_packed_lowrank_grad_input(
                    &grad_output,
                    &lowrank_weight_codes_tensor,
                    lowrank_weight_codes.1,
                    1,
                )
                .expect("raw lowrank grad input")
            },
            iters,
        );
        let lowrank_grad_input_cached_float_ms = bench_case(
            || {
                packed_lowrank_grad_input_from_float_weight_cuda(
                    grad_output.clone(),
                    lowrank_weight_float_tensor.clone(),
                    1,
                )
            },
            iters,
        );
        let lowrank_grad_input_cached_transposed_ms = bench_case(
            || {
                packed_lowrank_grad_input_from_transposed_float_weight_cuda(
                    grad_output.clone(),
                    lowrank_weight_transposed_float_tensor.clone(),
                    1,
                )
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
                try_fused_packed_lowrank_grad_weight(
                    &input_codes_tensor,
                    &grad_output,
                    input_codes.1,
                )
                .expect("fused lowrank grad weight")
            },
            iters,
        );
        let lowrank_grad_weight_raw_ms = bench_case(
            || {
                try_raw_cuda_packed_lowrank_grad_weight(
                    &input_codes_tensor,
                    &grad_output,
                    input_codes.1,
                )
                .expect("raw lowrank grad weight")
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
        let decoder_grad_input_raw_ms = bench_case(
            || {
                try_raw_cuda_packed_decoder_tail_grad_input(
                    &residual_grad_output,
                    &decoder_weight_codes_tensor,
                    decoder_weight_codes.1,
                    4,
                    128,
                )
                .expect("raw decoder grad input")
            },
            iters,
        );
        let decoder_grad_input_cached_float_ms = bench_case(
            || {
                packed_decoder_tail_grad_input_from_float_decoder_cuda(
                    residual_grad_output.clone(),
                    decoder_weight_float_tensor.clone(),
                    4,
                    128,
                )
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
        let decoder_grad_weight_raw_ms = bench_case_2d(
            || {
                try_raw_cuda_packed_decoder_tail_grad_weight(
                    &y_codes_tensor,
                    &residual_grad_output,
                    y_codes.1,
                )
                .expect("raw decoder grad weight")
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
        let lowrank_forward_ref_sync_ms = bench_case_sync(
            &device,
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
        let lowrank_forward_fused_sync_ms = bench_case_sync(
            &device,
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
        let decoder_forward_ref_sync_ms = bench_case_sync(
            &device,
            || {
                packed_decoder_tail_device_reference(
                    y_codes_tensor.clone().float().mul_scalar(y_codes.1),
                    decoder_weight_codes_tensor.clone(),
                    decoder_weight_codes.1,
                )
            },
            iters,
        );
        let decoder_forward_fused_sync_ms = bench_case_sync(
            &device,
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
        let lowrank_forward_raw_runtime_sync_ms = bench_case_sync(
            &device,
            || {
                try_raw_cuda_packed_lowrank_projection_prepacked_input(
                    &input_codes_packed_tensor,
                    &lowrank_weight_packed_tensor,
                    input_codes.1,
                    lowrank_weight_codes.1,
                    lowrank_weight_shape[2],
                )
                .expect("raw runtime lowrank forward")
            },
            iters,
        );
        let decoder_forward_raw_runtime_sync_ms = bench_case_sync(
            &device,
            || {
                try_raw_cuda_packed_decoder_tail_prepacked_input(
                    &y_codes_packed_tensor,
                    &decoder_weight_packed_tensor,
                    y_codes.1,
                    decoder_weight_codes.1,
                )
                .expect("raw runtime decoder tail")
            },
            iters,
        );
        let lowrank_raw_ms =
            bench_raw_sync(|| launch_raw_lowrank_sync(&mut lowrank_output_dev), iters);
        let decoder_raw_ms =
            bench_raw_sync(|| launch_raw_decoder_sync(&mut decoder_output_dev), iters);

        let memory_after = memory_snapshot(&device);

        println!("low_bit_bench backend=cuda iters={iters}");
        println!(
            "memory reserved_before={} in_use_before={} reserved_after={} in_use_after={}",
            memory_before.reserved,
            memory_before.in_use,
            memory_after.reserved,
            memory_after.in_use,
        );
        println!(
            "forward.lowrank ms_ref={:.3} ms_fused={:.3} speedup={:.2} max_abs_diff={:.6}",
            lowrank_forward_ref_ms,
            lowrank_forward_fused_ms,
            lowrank_forward_ref_ms / lowrank_forward_fused_ms,
            max_abs_diff(lowrank_forward_fused, lowrank_forward_ref.clone()),
        );
        println!(
            "forward.lowrank_raw_cuda_dp4a ms_ref={:.3} ms_raw={:.3} speedup={:.2} max_abs_diff={:.6}",
            lowrank_forward_ref_ms,
            lowrank_raw_ms,
            lowrank_forward_ref_ms / lowrank_raw_ms,
            max_abs_diff_host(
                &lowrank_raw_host,
                &lowrank_forward_ref
                    .clone()
                    .into_data()
                    .convert::<f32>()
                    .into_vec::<f32>()
                    .expect("lowrank ref host"),
            ),
        );
        println!(
            "forward.lowrank_sync ms_ref={:.3} ms_fused={:.3} ms_raw={:.3} fused_speedup={:.2} raw_speedup={:.2}",
            lowrank_forward_ref_sync_ms,
            lowrank_forward_fused_sync_ms,
            lowrank_raw_ms,
            lowrank_forward_ref_sync_ms / lowrank_forward_fused_sync_ms,
            lowrank_forward_ref_sync_ms / lowrank_raw_ms,
        );
        println!(
            "forward.lowrank_raw_runtime_sync ms_ref={:.3} ms_raw_runtime={:.3} speedup={:.2} max_abs_diff={:.6}",
            lowrank_forward_ref_sync_ms,
            lowrank_forward_raw_runtime_sync_ms,
            lowrank_forward_ref_sync_ms / lowrank_forward_raw_runtime_sync_ms,
            max_abs_diff(lowrank_forward_raw_runtime, lowrank_forward_ref.clone()),
        );
        println!(
            "forward.decoder_tail ms_ref={:.3} ms_fused={:.3} speedup={:.2} max_abs_diff={:.6}",
            decoder_forward_ref_ms,
            decoder_forward_fused_ms,
            decoder_forward_ref_ms / decoder_forward_fused_ms,
            max_abs_diff(decoder_forward_fused, decoder_forward_ref.clone()),
        );
        println!(
            "forward.decoder_tail_raw_cuda_dp4a ms_ref={:.3} ms_raw={:.3} speedup={:.2} max_abs_diff={:.6}",
            decoder_forward_ref_ms,
            decoder_raw_ms,
            decoder_forward_ref_ms / decoder_raw_ms,
            max_abs_diff_host(
                &decoder_raw_host,
                &decoder_forward_ref
                    .clone()
                    .into_data()
                    .convert::<f32>()
                    .into_vec::<f32>()
                    .expect("decoder ref host"),
            ),
        );
        println!(
            "forward.decoder_tail_sync ms_ref={:.3} ms_fused={:.3} ms_raw={:.3} fused_speedup={:.2} raw_speedup={:.2}",
            decoder_forward_ref_sync_ms,
            decoder_forward_fused_sync_ms,
            decoder_raw_ms,
            decoder_forward_ref_sync_ms / decoder_forward_fused_sync_ms,
            decoder_forward_ref_sync_ms / decoder_raw_ms,
        );
        println!(
            "forward.decoder_tail_raw_runtime_sync ms_ref={:.3} ms_raw_runtime={:.3} speedup={:.2} max_abs_diff={:.6}",
            decoder_forward_ref_sync_ms,
            decoder_forward_raw_runtime_sync_ms,
            decoder_forward_ref_sync_ms / decoder_forward_raw_runtime_sync_ms,
            max_abs_diff(decoder_forward_raw_runtime, decoder_forward_ref.clone()),
        );
        println!(
            "backward.lowrank_grad_input ms_ref={:.3} ms_fused={:.3} ms_raw={:.3} ms_cached_float={:.3} ms_cached_transposed={:.3} fused_speedup={:.2} raw_speedup={:.2} cached_float_speedup={:.2} cached_transposed_speedup={:.2} fused_max_abs_diff={:.6} raw_max_abs_diff={:.6} cached_float_max_abs_diff={:.6} cached_transposed_max_abs_diff={:.6}",
            lowrank_grad_input_ref_ms,
            lowrank_grad_input_fused_ms,
            lowrank_grad_input_raw_ms,
            lowrank_grad_input_cached_float_ms,
            lowrank_grad_input_cached_transposed_ms,
            lowrank_grad_input_ref_ms / lowrank_grad_input_fused_ms,
            lowrank_grad_input_ref_ms / lowrank_grad_input_raw_ms,
            lowrank_grad_input_ref_ms / lowrank_grad_input_cached_float_ms,
            lowrank_grad_input_ref_ms / lowrank_grad_input_cached_transposed_ms,
            max_abs_diff(lowrank_grad_input_fused, lowrank_grad_input_ref.clone()),
            max_abs_diff(lowrank_grad_input_raw, lowrank_grad_input_ref.clone()),
            max_abs_diff(lowrank_grad_input_cached_float, lowrank_grad_input_ref),
            max_abs_diff(
                lowrank_grad_input_cached_transposed,
                packed_lowrank_grad_input_device_reference(
                    grad_output.clone(),
                    lowrank_weight_codes_tensor.clone(),
                    lowrank_weight_codes.1,
                    1,
                ),
            ),
        );
        println!(
            "backward.lowrank_grad_weight ms_ref={:.3} ms_fused={:.3} ms_raw={:.3} fused_speedup={:.2} raw_speedup={:.2} fused_max_abs_diff={:.6} raw_max_abs_diff={:.6}",
            lowrank_grad_weight_ref_ms,
            lowrank_grad_weight_fused_ms,
            lowrank_grad_weight_raw_ms,
            lowrank_grad_weight_ref_ms / lowrank_grad_weight_fused_ms,
            lowrank_grad_weight_ref_ms / lowrank_grad_weight_raw_ms,
            max_abs_diff(lowrank_grad_weight_fused, lowrank_grad_weight_ref.clone()),
            max_abs_diff(lowrank_grad_weight_raw, lowrank_grad_weight_ref),
        );
        println!(
            "backward.decoder_tail_grad_input ms_ref={:.3} ms_fused={:.3} ms_raw={:.3} ms_cached_float={:.3} fused_speedup={:.2} raw_speedup={:.2} cached_float_speedup={:.2} fused_max_abs_diff={:.6} raw_max_abs_diff={:.6} cached_float_max_abs_diff={:.6}",
            decoder_grad_input_ref_ms,
            decoder_grad_input_fused_ms,
            decoder_grad_input_raw_ms,
            decoder_grad_input_cached_float_ms,
            decoder_grad_input_ref_ms / decoder_grad_input_fused_ms,
            decoder_grad_input_ref_ms / decoder_grad_input_raw_ms,
            decoder_grad_input_ref_ms / decoder_grad_input_cached_float_ms,
            max_abs_diff(decoder_grad_input_fused, decoder_grad_input_ref.clone()),
            max_abs_diff(decoder_grad_input_raw, decoder_grad_input_ref.clone()),
            max_abs_diff(decoder_grad_input_cached_float, decoder_grad_input_ref),
        );
        println!(
            "backward.decoder_tail_grad_weight ms_ref={:.3} ms_fused={:.3} ms_raw={:.3} fused_speedup={:.2} raw_speedup={:.2} fused_max_abs_diff={:.6} raw_max_abs_diff={:.6}",
            decoder_grad_weight_ref_ms,
            decoder_grad_weight_fused_ms,
            decoder_grad_weight_raw_ms,
            decoder_grad_weight_ref_ms / decoder_grad_weight_fused_ms,
            decoder_grad_weight_ref_ms / decoder_grad_weight_raw_ms,
            max_abs_diff(decoder_grad_weight_fused, decoder_grad_weight_ref.clone()),
            max_abs_diff(decoder_grad_weight_raw, decoder_grad_weight_ref),
        );
        println!("rho.pack_roundtrip ms={rho_pack_ms:.3}");
    }
}

#[cfg(feature = "cuda")]
fn main() {
    app::main();
}
