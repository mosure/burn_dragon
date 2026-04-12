use burn::tensor::Tensor as BurnTensor;
use burn::tensor::{Shape, TensorData};
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::prelude::*;
use burn_cubecl::cubecl::wgpu::WgpuRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::tensor::CubeTensor;

type WgpuCubeBackend = burn_wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;

const PREPROCESS_PARAM_LEN: usize = 5;
const PREPROCESS_WGPU_WORKGROUP_X: u32 = 32;

pub(crate) struct Mamba3PreprocessWgpuForwardOutput {
    pub(crate) packed: CubeTensor<WgpuRuntime>,
}

pub(crate) struct Mamba3PreprocessWgpuBackwardOutput {
    pub(crate) grad_q: CubeTensor<WgpuRuntime>,
    pub(crate) grad_k: CubeTensor<WgpuRuntime>,
    pub(crate) grad_angle: CubeTensor<WgpuRuntime>,
    pub(crate) grad_gamma: CubeTensor<WgpuRuntime>,
    pub(crate) grad_scale: CubeTensor<WgpuRuntime>,
}

#[cube]
fn reduce_partials_wgpu(
    partials: &mut SharedMemory<f32>,
    lane: usize,
    #[comptime] workgroup_size: usize,
) {
    if comptime!(workgroup_size >= 32usize) {
        if lane < 16usize {
            partials[lane] = partials[lane] + partials[lane + 16usize];
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 16usize) {
        if lane < 8usize {
            partials[lane] = partials[lane] + partials[lane + 8usize];
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 8usize) {
        if lane < 4usize {
            partials[lane] = partials[lane] + partials[lane + 4usize];
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 4usize) {
        if lane < 2usize {
            partials[lane] = partials[lane] + partials[lane + 2usize];
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 2usize) {
        if lane < 1usize {
            partials[lane] = partials[lane] + partials[lane + 1usize];
        }
        sync_cube();
    }
}

fn params_tensor_wgpu(
    device: &<WgpuCubeBackend as burn::tensor::backend::Backend>::Device,
    values: [f32; PREPROCESS_PARAM_LEN],
) -> BurnTensor<WgpuCubeBackend, 1> {
    BurnTensor::<WgpuCubeBackend, 1>::from_data(
        TensorData::new(values.to_vec(), [PREPROCESS_PARAM_LEN]),
        device,
    )
}

#[cube(launch_unchecked)]
fn mamba3_preprocess_forward_wgpu_kernel(
    q: &Tensor<f32>,
    k: &Tensor<f32>,
    angles: &Tensor<f32>,
    gamma: &Tensor<f32>,
    scale: &Tensor<f32>,
    packed: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let time = u32::cast_from(params[1]) as usize;
    let nheads = u32::cast_from(params[2]) as usize;
    let width = u32::cast_from(params[3]) as usize;
    let num_rope_angles = u32::cast_from(params[4]) as usize;

    let row = CUBE_POS_Z as usize;
    let h = CUBE_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    let b = row / time.max(1);
    let t = row % time.max(1);
    if b >= batch || t >= time || h >= nheads {
        terminate!();
    }

    let rotary_dim = num_rope_angles * 2usize;
    let packed_width = packed.shape(3);
    let gamma_index = b * gamma.stride(0) + t * gamma.stride(1) + h * gamma.stride(2);
    let scale_index = b * scale.stride(0) + t * scale.stride(1) + h * scale.stride(2);
    let gamma_value = gamma[gamma_index];
    let scale_value = scale[scale_index];

    let mut partials =
        SharedMemory::<f32>::new_aligned(PREPROCESS_WGPU_WORKGROUP_X as usize, 1usize);
    let zero = f32::cast_from(0u32);
    let mut qk_partial = zero;

    let mut pair = lane;
    while pair < num_rope_angles {
        let base = pair * 2usize;
        let q0_index = b * q.stride(0) + t * q.stride(1) + h * q.stride(2) + base * q.stride(3);
        let q1_index = q0_index + q.stride(3);
        let k0_index = b * k.stride(0) + t * k.stride(1) + h * k.stride(2) + base * k.stride(3);
        let k1_index = k0_index + k.stride(3);
        let angle_index = b * angles.stride(0)
            + t * angles.stride(1)
            + h * angles.stride(2)
            + pair * angles.stride(3);

        let q0 = q[q0_index];
        let q1 = q[q1_index];
        let k0 = k[k0_index];
        let k1 = k[k1_index];
        let angle = angles[angle_index];
        let cos = f32::cos(angle);
        let sin = f32::sin(angle);

        let q_rot0 = q0 * cos - q1 * sin;
        let q_rot1 = q0 * sin + q1 * cos;
        let k_rot0 = k0 * cos - k1 * sin;
        let k_rot1 = k0 * sin + k1 * cos;

        let packed_base = b * packed.stride(0)
            + t * packed.stride(1)
            + h * packed.stride(2)
            + base * packed.stride(3);
        packed[packed_base] = q_rot0;
        packed[packed_base + packed.stride(3)] = q_rot1;
        packed[packed_base + width * packed.stride(3)] = k_rot0 * scale_value;
        packed[packed_base + (width + 1usize) * packed.stride(3)] = k_rot1 * scale_value;

        qk_partial += q0 * k0 + q1 * k1;
        pair += CUBE_DIM_X as usize;
    }

    let mut d = rotary_dim + lane;
    while d < width {
        let q_index = b * q.stride(0) + t * q.stride(1) + h * q.stride(2) + d * q.stride(3);
        let k_index = b * k.stride(0) + t * k.stride(1) + h * k.stride(2) + d * k.stride(3);
        let q_value = q[q_index];
        let k_value = k[k_index];
        let packed_index = b * packed.stride(0)
            + t * packed.stride(1)
            + h * packed.stride(2)
            + d * packed.stride(3);
        packed[packed_index] = q_value;
        packed[packed_index + width * packed.stride(3)] = k_value * scale_value;
        qk_partial += q_value * k_value;
        d += CUBE_DIM_X as usize;
    }

    partials[lane] = qk_partial;
    sync_cube();
    reduce_partials_wgpu(&mut partials, lane, PREPROCESS_WGPU_WORKGROUP_X as usize);
    if lane == 0usize {
        let qk_index = b * packed.stride(0)
            + t * packed.stride(1)
            + h * packed.stride(2)
            + (packed_width - 1usize) * packed.stride(3);
        packed[qk_index] = partials[0] * gamma_value;
    }
}

#[cube(launch_unchecked)]
fn mamba3_preprocess_backward_wgpu_kernel(
    q: &Tensor<f32>,
    k: &Tensor<f32>,
    angles: &Tensor<f32>,
    gamma: &Tensor<f32>,
    scale: &Tensor<f32>,
    grad_packed: &Tensor<f32>,
    grad_q: &mut Tensor<f32>,
    grad_k: &mut Tensor<f32>,
    grad_angle: &mut Tensor<f32>,
    grad_gamma: &mut Tensor<f32>,
    grad_scale: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let time = u32::cast_from(params[1]) as usize;
    let nheads = u32::cast_from(params[2]) as usize;
    let width = u32::cast_from(params[3]) as usize;
    let num_rope_angles = u32::cast_from(params[4]) as usize;

    let row = CUBE_POS_Z as usize;
    let h = CUBE_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    let b = row / time.max(1);
    let t = row % time.max(1);
    if b >= batch || t >= time || h >= nheads {
        terminate!();
    }

    let packed_width = grad_packed.shape(3);
    let gamma_index = b * gamma.stride(0) + t * gamma.stride(1) + h * gamma.stride(2);
    let scale_index = b * scale.stride(0) + t * scale.stride(1) + h * scale.stride(2);
    let gamma_value = gamma[gamma_index];
    let scale_value = scale[scale_index];
    let grad_qk_index = b * grad_packed.stride(0)
        + t * grad_packed.stride(1)
        + h * grad_packed.stride(2)
        + (packed_width - 1usize) * grad_packed.stride(3);
    let grad_qk_dot = grad_packed[grad_qk_index];
    let qk_scale = grad_qk_dot * gamma_value;

    let mut qk_partials =
        SharedMemory::<f32>::new_aligned(PREPROCESS_WGPU_WORKGROUP_X as usize, 1usize);
    let mut scale_partials =
        SharedMemory::<f32>::new_aligned(PREPROCESS_WGPU_WORKGROUP_X as usize, 1usize);
    let zero = f32::cast_from(0u32);
    let rotary_dim = num_rope_angles * 2usize;
    let mut qk_partial = zero;
    let mut scale_partial = zero;

    let mut pair = lane;
    while pair < num_rope_angles {
        let base = pair * 2usize;
        let q0_index = b * q.stride(0) + t * q.stride(1) + h * q.stride(2) + base * q.stride(3);
        let q1_index = q0_index + q.stride(3);
        let k0_index = b * k.stride(0) + t * k.stride(1) + h * k.stride(2) + base * k.stride(3);
        let k1_index = k0_index + k.stride(3);
        let angle_index = b * angles.stride(0)
            + t * angles.stride(1)
            + h * angles.stride(2)
            + pair * angles.stride(3);

        let q0 = q[q0_index];
        let q1 = q[q1_index];
        let k0 = k[k0_index];
        let k1 = k[k1_index];
        let angle = angles[angle_index];
        let cos = f32::cos(angle);
        let sin = f32::sin(angle);

        let k_rot0 = k0 * cos - k1 * sin;
        let k_rot1 = k0 * sin + k1 * cos;

        let packed_base = b * grad_packed.stride(0)
            + t * grad_packed.stride(1)
            + h * grad_packed.stride(2)
            + base * grad_packed.stride(3);
        let grad_q_rot0 = grad_packed[packed_base];
        let grad_q_rot1 = grad_packed[packed_base + grad_packed.stride(3)];
        let grad_k_scaled0 = grad_packed[packed_base + width * grad_packed.stride(3)];
        let grad_k_scaled1 = grad_packed[packed_base + (width + 1usize) * grad_packed.stride(3)];
        let grad_k_rot0 = grad_k_scaled0 * scale_value;
        let grad_k_rot1 = grad_k_scaled1 * scale_value;

        grad_q[q0_index] = qk_scale * k0 + grad_q_rot0 * cos + grad_q_rot1 * sin;
        grad_q[q1_index] = qk_scale * k1 - grad_q_rot0 * sin + grad_q_rot1 * cos;
        grad_k[k0_index] = qk_scale * q0 + grad_k_rot0 * cos + grad_k_rot1 * sin;
        grad_k[k1_index] = qk_scale * q1 - grad_k_rot0 * sin + grad_k_rot1 * cos;
        grad_angle[angle_index] = grad_q_rot0 * (-q0 * sin - q1 * cos)
            + grad_q_rot1 * (q0 * cos - q1 * sin)
            + grad_k_rot0 * (-k0 * sin - k1 * cos)
            + grad_k_rot1 * (k0 * cos - k1 * sin);

        qk_partial += q0 * k0 + q1 * k1;
        scale_partial += grad_k_scaled0 * k_rot0 + grad_k_scaled1 * k_rot1;
        pair += CUBE_DIM_X as usize;
    }

    let mut d = rotary_dim + lane;
    while d < width {
        let q_index = b * q.stride(0) + t * q.stride(1) + h * q.stride(2) + d * q.stride(3);
        let k_index = b * k.stride(0) + t * k.stride(1) + h * k.stride(2) + d * k.stride(3);
        let q_value = q[q_index];
        let k_value = k[k_index];
        let packed_index = b * grad_packed.stride(0)
            + t * grad_packed.stride(1)
            + h * grad_packed.stride(2)
            + d * grad_packed.stride(3);
        let grad_q_rot = grad_packed[packed_index];
        let grad_k_scaled = grad_packed[packed_index + width * grad_packed.stride(3)];
        grad_q[q_index] = qk_scale * k_value + grad_q_rot;
        grad_k[k_index] = qk_scale * q_value + grad_k_scaled * scale_value;
        qk_partial += q_value * k_value;
        scale_partial += grad_k_scaled * k_value;
        d += CUBE_DIM_X as usize;
    }

    qk_partials[lane] = qk_partial;
    scale_partials[lane] = scale_partial;
    sync_cube();
    reduce_partials_wgpu(&mut qk_partials, lane, PREPROCESS_WGPU_WORKGROUP_X as usize);
    reduce_partials_wgpu(
        &mut scale_partials,
        lane,
        PREPROCESS_WGPU_WORKGROUP_X as usize,
    );
    if lane == 0usize {
        let grad_scalar_index =
            b * grad_gamma.stride(0) + t * grad_gamma.stride(1) + h * grad_gamma.stride(2);
        grad_gamma[grad_scalar_index] = grad_qk_dot * qk_partials[0];
        grad_scale[grad_scalar_index] = scale_partials[0];
    }
}

pub(crate) fn fused_mamba3_preprocess_forward_wgpu(
    q: CubeTensor<WgpuRuntime>,
    k: CubeTensor<WgpuRuntime>,
    angles: CubeTensor<WgpuRuntime>,
    gamma: CubeTensor<WgpuRuntime>,
    scale: CubeTensor<WgpuRuntime>,
) -> Mamba3PreprocessWgpuForwardOutput {
    let q = into_contiguous(q);
    let k = into_contiguous(k);
    let angles = into_contiguous(angles);
    let gamma = into_contiguous(gamma);
    let scale = into_contiguous(scale);
    let [batch, time, nheads, width] = q.meta.shape.dims::<4>();
    let num_rope_angles = angles.meta.shape.dims::<4>()[3];
    let client = q.client.clone();
    let device = q.device.clone();
    let packed = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, nheads, width * 2 + 1]),
    );
    let params = params_tensor_wgpu(
        &device,
        [
            batch as f32,
            time as f32,
            nheads as f32,
            width as f32,
            num_rope_angles as f32,
        ],
    )
    .into_primitive()
    .tensor();
    let cube_dim = CubeDim::new_1d(PREPROCESS_WGPU_WORKGROUP_X);
    let cube_count = CubeCount::Static(1, nheads as u32, (batch * time) as u32);
    unsafe {
        let _ = mamba3_preprocess_forward_wgpu_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            q.into_tensor_arg(),
            k.into_tensor_arg(),
            angles.into_tensor_arg(),
            gamma.into_tensor_arg(),
            scale.into_tensor_arg(),
            packed.clone().into_tensor_arg(),
            params.into_tensor_arg(),
        );
    }
    Mamba3PreprocessWgpuForwardOutput { packed }
}

pub(crate) fn fused_mamba3_preprocess_backward_wgpu(
    q: CubeTensor<WgpuRuntime>,
    k: CubeTensor<WgpuRuntime>,
    angles: CubeTensor<WgpuRuntime>,
    gamma: CubeTensor<WgpuRuntime>,
    scale: CubeTensor<WgpuRuntime>,
    grad_packed: CubeTensor<WgpuRuntime>,
) -> Mamba3PreprocessWgpuBackwardOutput {
    let q = into_contiguous(q);
    let k = into_contiguous(k);
    let angles = into_contiguous(angles);
    let gamma = into_contiguous(gamma);
    let scale = into_contiguous(scale);
    let grad_packed = into_contiguous(grad_packed);
    let [batch, time, nheads, width] = q.meta.shape.dims::<4>();
    let num_rope_angles = angles.meta.shape.dims::<4>()[3];
    let client = q.client.clone();
    let device = q.device.clone();
    let grad_q = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, nheads, width]),
    );
    let grad_k = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, nheads, width]),
    );
    let grad_angle = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, nheads, num_rope_angles]),
    );
    let grad_gamma = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, nheads]),
    );
    let grad_scale = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, nheads]),
    );
    let params = params_tensor_wgpu(
        &device,
        [
            batch as f32,
            time as f32,
            nheads as f32,
            width as f32,
            num_rope_angles as f32,
        ],
    )
    .into_primitive()
    .tensor();
    let cube_dim = CubeDim::new_1d(PREPROCESS_WGPU_WORKGROUP_X);
    let cube_count = CubeCount::Static(1, nheads as u32, (batch * time) as u32);
    unsafe {
        let _ = mamba3_preprocess_backward_wgpu_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            q.into_tensor_arg(),
            k.into_tensor_arg(),
            angles.into_tensor_arg(),
            gamma.into_tensor_arg(),
            scale.into_tensor_arg(),
            grad_packed.into_tensor_arg(),
            grad_q.clone().into_tensor_arg(),
            grad_k.clone().into_tensor_arg(),
            grad_angle.clone().into_tensor_arg(),
            grad_gamma.clone().into_tensor_arg(),
            grad_scale.clone().into_tensor_arg(),
            params.into_tensor_arg(),
        );
    }
    Mamba3PreprocessWgpuBackwardOutput {
        grad_q,
        grad_k,
        grad_angle,
        grad_gamma,
        grad_scale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend as BackendTrait;
    use burn::tensor::{Tensor, TensorPrimitive};

    type WgpuBackend = WgpuCubeBackend;

    fn assert_close<const D: usize>(
        actual: Tensor<WgpuBackend, D>,
        expected: Tensor<WgpuBackend, D>,
        tol: f32,
    ) {
        let actual = actual.into_data().to_vec::<f32>().expect("actual");
        let expected = expected.into_data().to_vec::<f32>().expect("expected");
        assert_eq!(actual.len(), expected.len());
        for (idx, (lhs, rhs)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= tol,
                "mismatch at {idx}: actual={lhs} expected={rhs} diff={diff} tol={tol}"
            );
        }
    }

    #[test]
    fn mamba3_preprocess_runtime_matches_reference_on_wgpu() {
        let device = <WgpuBackend as BackendTrait>::Device::default();
        let batch = 1;
        let time = 3;
        let nheads = 2;
        let width = 6;
        let num_rope_angles = 2;

        let q = Tensor::<WgpuBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * time * nheads * width))
                    .map(|idx| ((idx % 37) as f32) / 37.0 - 0.35)
                    .collect::<Vec<_>>(),
                [batch, time, nheads, width],
            ),
            &device,
        );
        let k = Tensor::<WgpuBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * time * nheads * width))
                    .map(|idx| ((idx % 41) as f32) / 41.0 - 0.25)
                    .collect::<Vec<_>>(),
                [batch, time, nheads, width],
            ),
            &device,
        );
        let angles = Tensor::<WgpuBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * time * nheads * num_rope_angles))
                    .map(|idx| ((idx % 29) as f32) / 29.0 - 0.2)
                    .collect::<Vec<_>>(),
                [batch, time, nheads, num_rope_angles],
            ),
            &device,
        );
        let gamma = Tensor::<WgpuBackend, 3>::from_data(
            TensorData::new(
                (0..(batch * time * nheads))
                    .map(|idx| ((idx % 23) as f32) / 23.0 + 0.2)
                    .collect::<Vec<_>>(),
                [batch, time, nheads],
            ),
            &device,
        );
        let scale = Tensor::<WgpuBackend, 3>::from_data(
            TensorData::new(
                (0..(batch * time * nheads))
                    .map(|idx| ((idx % 19) as f32) / 19.0 + 0.5)
                    .collect::<Vec<_>>(),
                [batch, time, nheads],
            ),
            &device,
        );

        let runtime = fused_mamba3_preprocess_forward_wgpu(
            q.clone().into_primitive().tensor(),
            k.clone().into_primitive().tensor(),
            angles.clone().into_primitive().tensor(),
            gamma.clone().into_primitive().tensor(),
            scale.clone().into_primitive().tensor(),
        );
        let packed =
            Tensor::<WgpuBackend, 4>::from_primitive(TensorPrimitive::Float(runtime.packed));
        let q_rot = packed.clone().slice_dim(3, 0..width);
        let k_scaled = packed.clone().slice_dim(3, width..(width * 2));
        let qk_dot = packed
            .clone()
            .slice_dim(3, width * 2..width * 2 + 1)
            .reshape([batch, time, nheads]);

        let rotary_dim = num_rope_angles * 2;
        let cos = angles.clone().cos();
        let sin = angles.clone().sin();
        let q_rot_ref_head = q.clone().slice_dim(3, 0..rotary_dim).reshape([
            batch,
            time,
            nheads,
            num_rope_angles,
            2,
        ]);
        let k_rot_ref_head = k.clone().slice_dim(3, 0..rotary_dim).reshape([
            batch,
            time,
            nheads,
            num_rope_angles,
            2,
        ]);
        let q0 = q_rot_ref_head.clone().slice_dim(4, 0..1).reshape([
            batch,
            time,
            nheads,
            num_rope_angles,
        ]);
        let q1 = q_rot_ref_head
            .slice_dim(4, 1..2)
            .reshape([batch, time, nheads, num_rope_angles]);
        let k0 = k_rot_ref_head.clone().slice_dim(4, 0..1).reshape([
            batch,
            time,
            nheads,
            num_rope_angles,
        ]);
        let k1 = k_rot_ref_head
            .slice_dim(4, 1..2)
            .reshape([batch, time, nheads, num_rope_angles]);
        let q_rot_ref = Tensor::cat(
            vec![
                (q0.clone() * cos.clone() - q1.clone() * sin.clone()).unsqueeze_dim::<5>(4),
                (q0 * sin.clone() + q1 * cos.clone()).unsqueeze_dim::<5>(4),
            ],
            4,
        )
        .reshape([batch, time, nheads, rotary_dim]);
        let k_rot_ref = Tensor::cat(
            vec![
                (k0.clone() * cos.clone() - k1.clone() * sin.clone()).unsqueeze_dim::<5>(4),
                (k0 * sin + k1 * cos).unsqueeze_dim::<5>(4),
            ],
            4,
        )
        .reshape([batch, time, nheads, rotary_dim]);
        let q_rot_ref = Tensor::cat(
            vec![q_rot_ref, q.clone().slice_dim(3, rotary_dim..width)],
            3,
        );
        let k_scaled_ref = Tensor::cat(
            vec![k_rot_ref, k.clone().slice_dim(3, rotary_dim..width)],
            3,
        ) * scale.clone().unsqueeze_dim::<4>(3);
        let qk_ref = (q.clone() * k.clone())
            .sum_dim(3)
            .reshape([batch, time, nheads])
            * gamma.clone();

        assert_close(q_rot, q_rot_ref, 1.0e-4);
        assert_close(k_scaled, k_scaled_ref, 1.0e-4);
        assert_close(qk_dot, qk_ref, 1.0e-4);
    }
}
