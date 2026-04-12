#[cfg(feature = "cuda")]
use std::sync::Once;

#[cfg(feature = "cuda")]
use super::cuda::try_raw_cuda_pack_activation_codes_i8x4;
use super::wgpu::{
    packed_decoder_tail_packed_dot_wgsl_runtime, packed_lowrank_projection_packed_dot_wgsl_runtime,
};
use super::*;

#[cfg(feature = "cuda")]
fn low_bit_training_debug_enabled() -> bool {
    std::env::var_os("BDH_STAGE_PROFILE_LOWBIT_DEBUG").is_some()
}

#[cfg(feature = "cuda")]
fn emit_lowrank_training_debug_once(message: impl FnOnce() -> String) {
    if !low_bit_training_debug_enabled() {
        return;
    }
    static ONCE: Once = Once::new();
    ONCE.call_once(|| eprintln!("{message}", message = message()));
}

#[derive(Debug, Clone, Copy)]
struct PackedLowrankTrainingShape {
    input_heads: usize,
}

#[derive(Debug, Clone, Copy)]
struct PackedDecoderTailTrainingShape {
    heads: usize,
    latent: usize,
}

#[derive(Debug, Clone)]
enum PackedActivationCodesState<T> {
    Device(T),
    HostI8 {
        values: Vec<i8>,
        shape: [usize; 4],
    },
    #[cfg(feature = "cuda")]
    HostPackedI32 {
        packed: Vec<i32>,
        shape: [usize; 4],
    },
}

#[allow(dead_code)]
fn unpack_activation_codes_i8x4_host(packed: &[i32], shape: [usize; 4]) -> Vec<i32> {
    let [batch, heads, time, inner] = shape;
    let outer = batch * heads * time;
    let pack_len = inner.div_ceil(4);
    let mut values = vec![0i32; outer * inner];
    for outer_idx in 0..outer {
        let input_base = outer_idx * pack_len;
        let output_base = outer_idx * inner;
        for pack_offset in 0..pack_len {
            let packed_value = packed[input_base + pack_offset] as u32;
            for lane in 0..4 {
                let idx = pack_offset * 4 + lane;
                if idx >= inner {
                    break;
                }
                let byte = ((packed_value >> (lane * 8)) & 0xff) as u8;
                values[output_base + idx] = i32::from(byte as i8);
            }
        }
    }
    values
}

#[derive(Debug, Clone)]
struct PackedLowrankTrainingStateWgpu {
    input_codes: PackedActivationCodesState<CubeTensor<WgpuRuntime>>,
    weight_codes: CubeTensor<WgpuRuntime>,
    activation_scale: f32,
    weight_scale: f32,
    relu_threshold: Option<f32>,
    shape: PackedLowrankTrainingShape,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
enum PackedLowrankGradInputStateCuda {
    WeightCodes {
        weight_codes: CubeTensor<CudaRuntime>,
        weight_scale: f32,
    },
    WeightTransposedFloat {
        weight_transposed_float: CubeTensor<CudaRuntime>,
    },
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct PackedLowrankTrainingStateCuda {
    input_codes: PackedActivationCodesState<CubeTensor<CudaRuntime>>,
    weight_codes: CubeTensor<CudaRuntime>,
    grad_input_state: PackedLowrankGradInputStateCuda,
    activation_scale: f32,
    weight_scale: f32,
    projection_scale: Option<CubeTensor<CudaRuntime>>,
    relu_threshold: Option<f32>,
    shape: PackedLowrankTrainingShape,
}

#[derive(Debug, Clone)]
struct PackedDecoderTailTrainingStateWgpu {
    y_codes: PackedActivationCodesState<CubeTensor<WgpuRuntime>>,
    weight_codes: CubeTensor<WgpuRuntime>,
    activation_scale: f32,
    weight_scale: f32,
    shape: PackedDecoderTailTrainingShape,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
enum PackedDecoderTailGradInputStateCuda {
    WeightCodes {
        weight_codes: CubeTensor<CudaRuntime>,
        weight_scale: f32,
    },
    DecoderFloat {
        decoder_float: CubeTensor<CudaRuntime>,
    },
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct PackedDecoderTailTrainingStateCuda {
    y_codes: PackedActivationCodesState<CubeTensor<CudaRuntime>>,
    grad_input_state: PackedDecoderTailGradInputStateCuda,
    activation_scale: f32,
    shape: PackedDecoderTailTrainingShape,
}

#[derive(Debug, Clone)]
struct RetroPackedLowrankWgpu {
    input_codes: PackedActivationCodesState<CubeTensor<WgpuRuntime>>,
    weight_codes: CubeTensor<WgpuRuntime>,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
    relu_threshold: Option<f32>,
}

impl RetroForward for RetroPackedLowrankWgpu {
    fn forward(&self, states: &mut BackwardStates, out_node: NodeId) {
        let device = self.weight_codes.device.clone();
        let input_codes = restore_activation_codes_state_wgpu(&self.input_codes, &device);
        let input_codes_tensor = BurnTensor::<WgpuCubeBackend, 4, Int>::from_primitive(
            try_cast_int_backend::<WgpuCubeBackend, _>(input_codes.into_primitive())
                .expect("wgpu retro lowrank input codes"),
        );
        let weight_codes_tensor = BurnTensor::<WgpuCubeBackend, 3, Int>::from_primitive(
            try_cast_int_backend::<WgpuCubeBackend, _>(self.weight_codes.clone())
                .expect("wgpu retro lowrank weight codes"),
        );
        let output = try_fused_packed_lowrank_projection(
            &input_codes_tensor,
            &weight_codes_tensor,
            self.activation_scale,
            self.weight_scale,
            self.latent_out,
        )
        .unwrap_or_else(|| {
            packed_lowrank_projection_device_reference(
                input_codes_tensor.float().mul_scalar(self.activation_scale),
                weight_codes_tensor,
                self.weight_scale,
                self.latent_out,
            )
        });
        let output = if let Some(threshold) = self.relu_threshold {
            activation::relu(output.sub_scalar(threshold))
        } else {
            output
        };
        states.save(
            out_node,
            try_cast_float_primitive::<WgpuCubeBackend, CubeTensor<WgpuRuntime>>(
                output.into_primitive().tensor(),
            )
            .expect("wgpu retro lowrank output primitive"),
        );
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct RetroPackedLowrankCuda {
    input_codes: PackedActivationCodesState<CubeTensor<CudaRuntime>>,
    weight_codes: CubeTensor<CudaRuntime>,
    activation_scale: f32,
    weight_scale: f32,
    projection_scale: Option<CubeTensor<CudaRuntime>>,
    latent_out: usize,
    relu_threshold: Option<f32>,
}

#[cfg(feature = "cuda")]
impl RetroForward for RetroPackedLowrankCuda {
    fn forward(&self, states: &mut BackwardStates, out_node: NodeId) {
        let device = self.weight_codes.device.clone();
        let input_codes = restore_activation_codes_state_cuda(&self.input_codes, &device);
        let input_codes_tensor = BurnTensor::<CudaCubeBackend, 4, Int>::from_primitive(
            try_cast_int_backend::<CudaCubeBackend, _>(input_codes.into_primitive())
                .expect("cuda retro lowrank input codes"),
        );
        let weight_codes_tensor = BurnTensor::<CudaCubeBackend, 3, Int>::from_primitive(
            try_cast_int_backend::<CudaCubeBackend, _>(self.weight_codes.clone())
                .expect("cuda retro lowrank weight codes"),
        );
        let output = if let Some(projection_scale) = self.projection_scale.clone() {
            let projection_scale_tensor = BurnTensor::<CudaCubeBackend, 1>::from_primitive(
                TensorPrimitive::Float(projection_scale),
            );
            try_raw_cuda_packed_lowrank_projection_device_scale(
                &input_codes_tensor,
                &weight_codes_tensor,
                &projection_scale_tensor,
                1.0,
                self.latent_out,
            )
            .unwrap_or_else(|| {
                packed_lowrank_projection_device_reference(
                    input_codes_tensor
                        .float()
                        .mul(projection_scale_tensor.clone().reshape([1, 1, 1, 1])),
                    weight_codes_tensor,
                    1.0,
                    self.latent_out,
                )
            })
        } else {
            try_fused_packed_lowrank_projection(
                &input_codes_tensor,
                &weight_codes_tensor,
                self.activation_scale,
                self.weight_scale,
                self.latent_out,
            )
            .unwrap_or_else(|| {
                packed_lowrank_projection_device_reference(
                    input_codes_tensor.float().mul_scalar(self.activation_scale),
                    weight_codes_tensor,
                    self.weight_scale,
                    self.latent_out,
                )
            })
        };
        let output = if let Some(threshold) = self.relu_threshold {
            activation::relu(output.sub_scalar(threshold))
        } else {
            output
        };
        states.save(
            out_node,
            try_cast_float_primitive::<CudaCubeBackend, CubeTensor<CudaRuntime>>(
                output.into_primitive().tensor(),
            )
            .expect("cuda retro lowrank output primitive"),
        );
    }
}

#[derive(Debug, Clone)]
struct RetroPackedDecoderTailWgpu {
    y_codes: PackedActivationCodesState<CubeTensor<WgpuRuntime>>,
    weight_codes: CubeTensor<WgpuRuntime>,
    activation_scale: f32,
    weight_scale: f32,
}

impl RetroForward for RetroPackedDecoderTailWgpu {
    fn forward(&self, states: &mut BackwardStates, out_node: NodeId) {
        let device = self.weight_codes.device.clone();
        let y_codes = restore_activation_codes_state_wgpu(&self.y_codes, &device);
        let y_codes_tensor = BurnTensor::<WgpuCubeBackend, 4, Int>::from_primitive(
            try_cast_int_backend::<WgpuCubeBackend, _>(y_codes.into_primitive())
                .expect("wgpu retro decoder-tail activation codes"),
        );
        let weight_codes_tensor = BurnTensor::<WgpuCubeBackend, 2, Int>::from_primitive(
            try_cast_int_backend::<WgpuCubeBackend, _>(self.weight_codes.clone())
                .expect("wgpu retro decoder-tail weight codes"),
        );
        let output = try_fused_packed_decoder_tail(
            &y_codes_tensor,
            &weight_codes_tensor,
            self.activation_scale,
            self.weight_scale,
        )
        .unwrap_or_else(|| {
            packed_decoder_tail_device_reference(
                y_codes_tensor.float().mul_scalar(self.activation_scale),
                weight_codes_tensor,
                self.weight_scale,
            )
        });
        states.save(
            out_node,
            try_cast_float_primitive::<WgpuCubeBackend, CubeTensor<WgpuRuntime>>(
                output.into_primitive().tensor(),
            )
            .expect("wgpu retro decoder-tail output primitive"),
        );
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct RetroPackedDecoderTailCuda {
    y_codes: PackedActivationCodesState<CubeTensor<CudaRuntime>>,
    weight_codes: CubeTensor<CudaRuntime>,
    activation_scale: f32,
    weight_scale: f32,
}

#[cfg(feature = "cuda")]
impl RetroForward for RetroPackedDecoderTailCuda {
    fn forward(&self, states: &mut BackwardStates, out_node: NodeId) {
        let device = self.weight_codes.device.clone();
        let y_codes = restore_activation_codes_state_cuda(&self.y_codes, &device);
        let y_codes_tensor = BurnTensor::<CudaCubeBackend, 4, Int>::from_primitive(
            try_cast_int_backend::<CudaCubeBackend, _>(y_codes.into_primitive())
                .expect("cuda retro decoder-tail activation codes"),
        );
        let weight_codes_tensor = BurnTensor::<CudaCubeBackend, 2, Int>::from_primitive(
            try_cast_int_backend::<CudaCubeBackend, _>(self.weight_codes.clone())
                .expect("cuda retro decoder-tail weight codes"),
        );
        let output = try_fused_packed_decoder_tail(
            &y_codes_tensor,
            &weight_codes_tensor,
            self.activation_scale,
            self.weight_scale,
        )
        .unwrap_or_else(|| {
            packed_decoder_tail_device_reference(
                y_codes_tensor.float().mul_scalar(self.activation_scale),
                weight_codes_tensor,
                self.weight_scale,
            )
        });
        states.save(
            out_node,
            try_cast_float_primitive::<CudaCubeBackend, CubeTensor<CudaRuntime>>(
                output.into_primitive().tensor(),
            )
            .expect("cuda retro decoder-tail output primitive"),
        );
    }
}

#[derive(Debug)]
struct FusedPackedLowrankBackward<B>(PhantomData<B>);

impl Backward<WgpuCubeBackend, 2> for FusedPackedLowrankBackward<WgpuCubeBackend> {
    type State = PackedLowrankTrainingStateWgpu;

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let grad_output = grads.consume::<WgpuCubeBackend>(&ops.node);
        let state = ops.state;
        let parents = ops.parents;

        let grad_output =
            BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(grad_output));
        let weight_codes = BurnTensor::<WgpuCubeBackend, 3, Int>::from_primitive(
            try_cast_int_backend::<WgpuCubeBackend, _>(state.weight_codes.clone())
                .expect("wgpu lowrank backward weight codes"),
        );
        let input_codes =
            restore_activation_codes_state_wgpu(&state.input_codes, &state.weight_codes.device);
        let grad_projected = if let Some(threshold) = state.relu_threshold {
            let projected = packed_lowrank_projection_device_reference(
                input_codes
                    .clone()
                    .float()
                    .mul_scalar(state.activation_scale),
                weight_codes.clone(),
                state.weight_scale,
                weight_codes.shape().dims::<3>()[2],
            );
            let activation_mask = projected.sub_scalar(threshold).greater_elem(0.0).float();
            grad_output.clone() * activation_mask
        } else {
            grad_output.clone()
        };

        if let Some(parent) = &parents[0] {
            let grad_input = try_fused_packed_lowrank_grad_input(
                &grad_projected,
                &weight_codes,
                state.weight_scale,
                state.shape.input_heads,
            )
            .unwrap_or_else(|| {
                packed_lowrank_grad_input_device_reference(
                    grad_projected.clone(),
                    weight_codes.clone(),
                    state.weight_scale,
                    state.shape.input_heads,
                )
            });
            grads.register::<WgpuCubeBackend>(parent.id, grad_input.into_primitive().tensor());
        }

        if let Some(parent) = &parents[1] {
            let grad_weight = try_fused_packed_lowrank_grad_weight(
                &input_codes,
                &grad_projected,
                state.activation_scale,
            )
            .unwrap_or_else(|| {
                packed_lowrank_grad_weight_device_reference(
                    input_codes.clone(),
                    grad_projected.clone(),
                    state.activation_scale,
                )
            });
            grads.register::<WgpuCubeBackend>(parent.id, grad_weight.into_primitive().tensor());
        }
    }
}

#[cfg(feature = "cuda")]
impl Backward<CudaCubeBackend, 2> for FusedPackedLowrankBackward<CudaCubeBackend> {
    type State = PackedLowrankTrainingStateCuda;

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let grad_output = grads.consume::<CudaCubeBackend>(&ops.node);
        let state = ops.state;
        let parents = ops.parents;

        let grad_output =
            BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(grad_output));
        let grad_device = grad_output.device();
        let input_codes = restore_activation_codes_state_cuda(&state.input_codes, &grad_device);
        let weight_codes = BurnTensor::<CudaCubeBackend, 3, Int>::from_primitive(
            try_cast_int_backend::<CudaCubeBackend, _>(state.weight_codes.clone())
                .expect("cuda lowrank backward weight codes"),
        );
        let grad_projected = if let Some(threshold) = state.relu_threshold {
            let projected = if let Some(projection_scale) = state.projection_scale.clone() {
                let projection_scale_tensor = BurnTensor::<CudaCubeBackend, 1>::from_primitive(
                    TensorPrimitive::Float(projection_scale),
                );
                try_raw_cuda_packed_lowrank_projection_device_scale(
                    &input_codes,
                    &weight_codes,
                    &projection_scale_tensor,
                    1.0,
                    weight_codes.shape().dims::<3>()[2],
                )
                .unwrap_or_else(|| {
                    packed_lowrank_projection_device_reference(
                        input_codes
                            .clone()
                            .float()
                            .mul(projection_scale_tensor.clone().reshape([1, 1, 1, 1])),
                        weight_codes.clone(),
                        1.0,
                        weight_codes.shape().dims::<3>()[2],
                    )
                })
            } else {
                packed_lowrank_projection_device_reference(
                    input_codes
                        .clone()
                        .float()
                        .mul_scalar(state.activation_scale),
                    weight_codes.clone(),
                    state.weight_scale,
                    weight_codes.shape().dims::<3>()[2],
                )
            };
            let activation_mask = projected.sub_scalar(threshold).greater_elem(0.0).float();
            grad_output.clone() * activation_mask
        } else {
            grad_output.clone()
        };

        if let Some(parent) = &parents[0] {
            let grad_input = match &state.grad_input_state {
                PackedLowrankGradInputStateCuda::WeightCodes {
                    weight_codes,
                    weight_scale,
                } => {
                    let weight_codes = BurnTensor::<CudaCubeBackend, 3, Int>::from_primitive(
                        try_cast_int_backend::<CudaCubeBackend, _>(weight_codes.clone())
                            .expect("cuda lowrank backward weight codes"),
                    );
                    packed_lowrank_grad_input_device_reference(
                        grad_projected.clone(),
                        weight_codes,
                        *weight_scale,
                        state.shape.input_heads,
                    )
                }
                PackedLowrankGradInputStateCuda::WeightTransposedFloat {
                    weight_transposed_float,
                } => {
                    let weight_transposed_float = BurnTensor::<CudaCubeBackend, 3>::from_primitive(
                        TensorPrimitive::Float(weight_transposed_float.clone()),
                    );
                    packed_lowrank_grad_input_from_transposed_float_weight_cuda(
                        grad_projected.clone(),
                        weight_transposed_float,
                        state.shape.input_heads,
                    )
                }
            };
            grads.register::<CudaCubeBackend>(parent.id, grad_input.into_primitive().tensor());
        }

        if let Some(parent) = &parents[1] {
            // Prefer the exact reference weight-gradient path for training fidelity until the
            // raw/fused CUDA grad-weight kernels prove parity at the model level.
            let grad_weight = packed_lowrank_grad_weight_device_reference(
                input_codes.clone(),
                grad_projected.clone(),
                state.activation_scale,
            );
            grads.register::<CudaCubeBackend>(parent.id, grad_weight.into_primitive().tensor());
        }
    }
}

#[derive(Debug)]
struct FusedPackedDecoderTailBackward<B>(PhantomData<B>);

impl Backward<WgpuCubeBackend, 2> for FusedPackedDecoderTailBackward<WgpuCubeBackend> {
    type State = PackedDecoderTailTrainingStateWgpu;

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let grad_output = grads.consume::<WgpuCubeBackend>(&ops.node);
        let state = ops.state;
        let parents = ops.parents;

        let grad_output =
            BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(grad_output));
        let weight_codes = BurnTensor::<WgpuCubeBackend, 2, Int>::from_primitive(
            try_cast_int_backend::<WgpuCubeBackend, _>(state.weight_codes.clone())
                .expect("wgpu decoder-tail backward weight codes"),
        );

        if let Some(parent) = &parents[0] {
            let grad_input = try_fused_packed_decoder_tail_grad_input(
                &grad_output,
                &weight_codes,
                state.weight_scale,
                state.shape.heads,
                state.shape.latent,
            )
            .unwrap_or_else(|| {
                packed_decoder_tail_grad_input_device_reference(
                    grad_output.clone(),
                    weight_codes.clone(),
                    state.weight_scale,
                    state.shape.heads,
                    state.shape.latent,
                )
            });
            grads.register::<WgpuCubeBackend>(parent.id, grad_input.into_primitive().tensor());
        }

        if let Some(parent) = &parents[1] {
            let y_codes =
                restore_activation_codes_state_wgpu(&state.y_codes, &state.weight_codes.device);
            let grad_weight = try_fused_packed_decoder_tail_grad_weight(
                &y_codes,
                &grad_output,
                state.activation_scale,
            )
            .unwrap_or_else(|| {
                packed_decoder_tail_grad_weight_device_reference(
                    y_codes.clone(),
                    grad_output.clone(),
                    state.activation_scale,
                )
            });
            grads.register::<WgpuCubeBackend>(parent.id, grad_weight.into_primitive().tensor());
        }
    }
}

#[cfg(feature = "cuda")]
impl Backward<CudaCubeBackend, 2> for FusedPackedDecoderTailBackward<CudaCubeBackend> {
    type State = PackedDecoderTailTrainingStateCuda;

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let grad_output = grads.consume::<CudaCubeBackend>(&ops.node);
        let state = ops.state;
        let parents = ops.parents;

        let grad_output =
            BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(grad_output));
        let grad_device = grad_output.device();

        if let Some(parent) = &parents[0] {
            let grad_input = match &state.grad_input_state {
                PackedDecoderTailGradInputStateCuda::WeightCodes {
                    weight_codes,
                    weight_scale,
                } => {
                    let weight_codes = BurnTensor::<CudaCubeBackend, 2, Int>::from_primitive(
                        try_cast_int_backend::<CudaCubeBackend, _>(weight_codes.clone())
                            .expect("cuda decoder-tail backward weight codes"),
                    );
                    packed_decoder_tail_grad_input_device_reference(
                        grad_output.clone(),
                        weight_codes,
                        *weight_scale,
                        state.shape.heads,
                        state.shape.latent,
                    )
                }
                PackedDecoderTailGradInputStateCuda::DecoderFloat { decoder_float } => {
                    let decoder_float = BurnTensor::<CudaCubeBackend, 2>::from_primitive(
                        TensorPrimitive::Float(decoder_float.clone()),
                    );
                    packed_decoder_tail_grad_input_from_float_decoder_cuda(
                        grad_output.clone(),
                        decoder_float,
                        state.shape.heads,
                        state.shape.latent,
                    )
                }
            };
            grads.register::<CudaCubeBackend>(parent.id, grad_input.into_primitive().tensor());
        }

        if let Some(parent) = &parents[1] {
            let y_codes = restore_activation_codes_state_cuda(&state.y_codes, &grad_device);
            // Prefer the exact reference weight-gradient path for training fidelity until the
            // raw/fused CUDA grad-weight kernels prove parity at the model level.
            let grad_weight = packed_decoder_tail_grad_weight_device_reference(
                y_codes.clone(),
                grad_output.clone(),
                state.activation_scale,
            );
            grads.register::<CudaCubeBackend>(parent.id, grad_weight.into_primitive().tensor());
        }
    }
}

fn create_lowrank_params_wgpu(
    device: &<WgpuCubeBackend as BackendTrait>::Device,
    batch: usize,
    input_heads: usize,
    heads: usize,
    time: usize,
    embd: usize,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
) -> CubeTensor<WgpuRuntime> {
    let params = Tensor::<WgpuCubeBackend, 1>::from_data(
        [
            batch as f32,
            input_heads as f32,
            heads as f32,
            time as f32,
            embd as f32,
            latent_out as f32,
            activation_scale,
            weight_scale,
        ],
        device,
    );
    try_cast_float_primitive::<WgpuCubeBackend, _>(params.into_primitive().tensor())
        .expect("wgpu packed low-rank params tensor")
}

#[cfg(feature = "cuda")]
fn create_lowrank_params_cuda(
    device: &<CudaCubeBackend as BackendTrait>::Device,
    batch: usize,
    input_heads: usize,
    heads: usize,
    time: usize,
    embd: usize,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
) -> CubeTensor<CudaRuntime> {
    let params = Tensor::<CudaCubeBackend, 1>::from_data(
        [
            batch as f32,
            input_heads as f32,
            heads as f32,
            time as f32,
            embd as f32,
            latent_out as f32,
            activation_scale,
            weight_scale,
        ],
        device,
    );
    try_cast_float_primitive::<CudaCubeBackend, _>(params.into_primitive().tensor())
        .expect("cuda packed low-rank params tensor")
}

fn create_decoder_tail_params_wgpu(
    device: &<WgpuCubeBackend as BackendTrait>::Device,
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
    artifact_latent_per_head: usize,
    dim: usize,
    activation_scale: f32,
    weight_scale: f32,
) -> CubeTensor<WgpuRuntime> {
    let params = Tensor::<WgpuCubeBackend, 1>::from_data(
        [
            batch as f32,
            heads as f32,
            time as f32,
            latent as f32,
            artifact_latent_per_head as f32,
            dim as f32,
            activation_scale,
            weight_scale,
        ],
        device,
    );
    try_cast_float_primitive::<WgpuCubeBackend, _>(params.into_primitive().tensor())
        .expect("wgpu packed decoder tail params tensor")
}

#[cfg(feature = "cuda")]
fn create_decoder_tail_params_cuda(
    device: &<CudaCubeBackend as BackendTrait>::Device,
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
    artifact_latent_per_head: usize,
    dim: usize,
    activation_scale: f32,
    weight_scale: f32,
) -> CubeTensor<CudaRuntime> {
    let params = Tensor::<CudaCubeBackend, 1>::from_data(
        [
            batch as f32,
            heads as f32,
            time as f32,
            latent as f32,
            artifact_latent_per_head as f32,
            dim as f32,
            activation_scale,
            weight_scale,
        ],
        device,
    );
    try_cast_float_primitive::<CudaCubeBackend, _>(params.into_primitive().tensor())
        .expect("cuda packed decoder tail params tensor")
}

fn pack_activation_codes_state_wgpu(
    codes: CubeTensor<WgpuRuntime>,
    pack_to_host: bool,
) -> PackedActivationCodesState<CubeTensor<WgpuRuntime>> {
    if !pack_to_host {
        return PackedActivationCodesState::Device(codes);
    }
    let shape = codes.meta.shape.dims::<4>();
    let values = BurnTensor::<WgpuCubeBackend, 4, Int>::from_primitive(
        try_cast_int_backend::<WgpuCubeBackend, _>(codes)
            .expect("wgpu packed activation codes primitive"),
    )
    .into_data()
    .convert::<i32>()
    .into_vec::<i32>()
    .expect("wgpu packed activation codes values")
    .into_iter()
    .map(|value| value.clamp(-127, 127) as i8)
    .collect();
    PackedActivationCodesState::HostI8 { values, shape }
}

fn restore_activation_codes_state_wgpu(
    state: &PackedActivationCodesState<CubeTensor<WgpuRuntime>>,
    device: &<WgpuCubeBackend as BackendTrait>::Device,
) -> BurnTensor<WgpuCubeBackend, 4, Int> {
    match state {
        PackedActivationCodesState::Device(codes) => {
            BurnTensor::<WgpuCubeBackend, 4, Int>::from_primitive(
                try_cast_int_backend::<WgpuCubeBackend, _>(codes.clone())
                    .expect("wgpu activation codes primitive"),
            )
        }
        PackedActivationCodesState::HostI8 { values, shape } => {
            Tensor::<WgpuCubeBackend, 4, Int>::from_data(
                TensorData::new(
                    values.iter().map(|value| *value as i32).collect::<Vec<_>>(),
                    *shape,
                ),
                device,
            )
        }
        #[cfg(feature = "cuda")]
        PackedActivationCodesState::HostPackedI32 { packed, shape } => {
            Tensor::<WgpuCubeBackend, 4, Int>::from_data(
                TensorData::new(unpack_activation_codes_i8x4_host(packed, *shape), *shape),
                device,
            )
        }
    }
}

#[cfg(feature = "cuda")]
fn pack_activation_codes_state_cuda(
    codes: CubeTensor<CudaRuntime>,
    pack_to_host: bool,
) -> (
    PackedActivationCodesState<CubeTensor<CudaRuntime>>,
    Option<BurnTensor<CudaCubeBackend, 4, Int>>,
) {
    if !pack_to_host {
        return (PackedActivationCodesState::Device(codes), None);
    }
    let shape = codes.meta.shape.dims::<4>();
    let codes_tensor = BurnTensor::<CudaCubeBackend, 4, Int>::from_primitive(
        try_cast_int_backend::<CudaCubeBackend, _>(codes.clone())
            .expect("cuda packed activation codes primitive"),
    );
    if let Some(packed_tensor) = try_raw_cuda_pack_activation_codes_i8x4(&codes_tensor) {
        let packed = packed_tensor
            .clone()
            .into_data()
            .convert::<i32>()
            .into_vec::<i32>()
            .expect("cuda packed activation codes i8x4 values");
        return (
            PackedActivationCodesState::HostPackedI32 { packed, shape },
            Some(packed_tensor),
        );
    }
    let values = codes_tensor
        .into_data()
        .convert::<i32>()
        .into_vec::<i32>()
        .expect("cuda packed activation codes values")
        .into_iter()
        .map(|value| value.clamp(-127, 127) as i8)
        .collect();
    (PackedActivationCodesState::HostI8 { values, shape }, None)
}

#[cfg(feature = "cuda")]
fn restore_activation_codes_state_cuda(
    state: &PackedActivationCodesState<CubeTensor<CudaRuntime>>,
    device: &<CudaCubeBackend as BackendTrait>::Device,
) -> BurnTensor<CudaCubeBackend, 4, Int> {
    match state {
        PackedActivationCodesState::Device(codes) => {
            BurnTensor::<CudaCubeBackend, 4, Int>::from_primitive(
                try_cast_int_backend::<CudaCubeBackend, _>(codes.clone())
                    .expect("cuda activation codes primitive"),
            )
        }
        PackedActivationCodesState::HostI8 { values, shape } => {
            Tensor::<CudaCubeBackend, 4, Int>::from_data(
                TensorData::new(
                    values.iter().map(|value| *value as i32).collect::<Vec<_>>(),
                    *shape,
                ),
                device,
            )
        }
        PackedActivationCodesState::HostPackedI32 { packed, shape } => {
            Tensor::<CudaCubeBackend, 4, Int>::from_data(
                TensorData::new(unpack_activation_codes_i8x4_host(packed, *shape), *shape),
                device,
            )
        }
    }
}

#[cfg(feature = "cuda")]
fn packed_lowrank_input_tensor_from_activation_state_cuda(
    state: &PackedActivationCodesState<CubeTensor<CudaRuntime>>,
    device: &<CudaCubeBackend as BackendTrait>::Device,
) -> Option<BurnTensor<CudaCubeBackend, 4, Int>> {
    match state {
        PackedActivationCodesState::HostI8 { values, shape } => {
            let packed =
                pack_lowrank_input_codes_i8x4(values, shape[0], shape[1], shape[2], shape[3]);
            Some(Tensor::<CudaCubeBackend, 4, Int>::from_data(
                TensorData::new(
                    packed.into_iter().map(i64::from).collect::<Vec<_>>(),
                    [shape[0], shape[1], shape[2], shape[3].div_ceil(4)],
                ),
                device,
            ))
        }
        PackedActivationCodesState::HostPackedI32 { packed, shape } => {
            Some(Tensor::<CudaCubeBackend, 4, Int>::from_data(
                TensorData::new(
                    packed.iter().copied().map(i64::from).collect::<Vec<_>>(),
                    [shape[0], shape[1], shape[2], shape[3].div_ceil(4)],
                ),
                device,
            ))
        }
        PackedActivationCodesState::Device(_) => None,
    }
}

#[cfg(feature = "cuda")]
fn packed_decoder_input_tensor_from_activation_state_cuda(
    state: &PackedActivationCodesState<CubeTensor<CudaRuntime>>,
    device: &<CudaCubeBackend as BackendTrait>::Device,
) -> Option<BurnTensor<CudaCubeBackend, 4, Int>> {
    match state {
        PackedActivationCodesState::HostI8 { values, shape } => {
            let packed =
                pack_decoder_input_codes_i8x4(values, shape[0], shape[1], shape[2], shape[3]);
            Some(Tensor::<CudaCubeBackend, 4, Int>::from_data(
                TensorData::new(
                    packed.into_iter().map(i64::from).collect::<Vec<_>>(),
                    [shape[0], shape[1], shape[2], shape[3].div_ceil(4)],
                ),
                device,
            ))
        }
        PackedActivationCodesState::HostPackedI32 { packed, shape } => {
            Some(Tensor::<CudaCubeBackend, 4, Int>::from_data(
                TensorData::new(
                    packed.iter().copied().map(i64::from).collect::<Vec<_>>(),
                    [shape[0], shape[1], shape[2], shape[3].div_ceil(4)],
                ),
                device,
            ))
        }
        PackedActivationCodesState::Device(_) => None,
    }
}

fn fused_packed_lowrank_training_autodiff_wgpu<C: CheckpointStrategy>(
    input: WgpuCubeAutodiffTensor<C>,
    weight: WgpuCubeAutodiffTensor<C>,
    input_codes: CubeTensor<WgpuRuntime>,
    weight_codes: CubeTensor<WgpuRuntime>,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
    pack_activation_state_to_host: bool,
    relu_threshold: Option<f32>,
) -> WgpuCubeAutodiffTensor<C> {
    let input_inner = <WgpuCubeAutodiffBackend<C> as AutodiffBackend>::inner(input.clone());
    let [batch, input_heads, time, _] = input_inner.meta.shape.dims::<4>();
    let embd = input_inner.meta.shape.dims::<4>()[3];
    let heads = weight_codes.meta.shape.dims::<3>()[0];
    let artifact_latent = weight_codes.meta.shape.dims::<3>()[2];
    let output = packed_lowrank_projection_packed_dot_wgsl_runtime(
        input_codes.clone(),
        weight_codes.clone(),
        batch,
        input_heads,
        heads,
        time,
        embd,
        latent_out,
        artifact_latent,
        activation_scale,
        weight_scale,
    )
    .unwrap_or_else(|_| {
        let params = create_lowrank_params_wgpu(
            &input_codes.device,
            batch,
            input_heads,
            heads,
            time,
            embd,
            activation_scale,
            weight_scale,
            latent_out,
        );
        packed_lowrank_projection_cube_runtime::<WgpuRuntime>(
            input_codes.clone(),
            weight_codes.clone(),
            params,
            batch,
            heads,
            time,
            latent_out,
        )
    });
    let output = if let Some(threshold) = relu_threshold {
        let activated = activation::relu(
            BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(output))
                .sub_scalar(threshold),
        );
        try_cast_float_primitive::<WgpuCubeBackend, CubeTensor<WgpuRuntime>>(
            activated.into_primitive().tensor(),
        )
        .expect("wgpu packed lowrank relu output primitive")
    } else {
        output
    };
    let shape = PackedLowrankTrainingShape { input_heads };
    let input_codes_state =
        pack_activation_codes_state_wgpu(input_codes, pack_activation_state_to_host);
    match FusedPackedLowrankBackward::<WgpuCubeBackend>(PhantomData)
        .prepare::<C>([input.node.clone(), weight.node.clone()])
        .memory_bound()
        .retro_forward(RetroPackedLowrankWgpu {
            input_codes: input_codes_state.clone(),
            weight_codes: weight_codes.clone(),
            activation_scale,
            weight_scale,
            latent_out,
            relu_threshold,
        })
        .parents([&input, &weight])
        .stateful()
    {
        OpsKind::Tracked(mut prep) => {
            prep.checkpoint(&input);
            prep.checkpoint(&weight);
            prep.finish(
                PackedLowrankTrainingStateWgpu {
                    input_codes: input_codes_state,
                    weight_codes,
                    activation_scale,
                    weight_scale,
                    relu_threshold,
                    shape,
                },
                output,
            )
        }
        OpsKind::UnTracked(prep) => prep.finish(output),
    }
}

#[cfg(feature = "cuda")]
fn fused_packed_lowrank_training_autodiff_cuda<C: CheckpointStrategy>(
    input: CudaCubeAutodiffTensor<C>,
    weight: CudaCubeAutodiffTensor<C>,
    input_codes: CubeTensor<CudaRuntime>,
    weight_codes: CubeTensor<CudaRuntime>,
    activation_scale: f32,
    weight_scale: f32,
    projection_scale: Option<CubeTensor<CudaRuntime>>,
    latent_out: usize,
    pack_activation_state_to_host: bool,
    relu_threshold: Option<f32>,
) -> CudaCubeAutodiffTensor<C> {
    let input_inner = <CudaCubeAutodiffBackend<C> as AutodiffBackend>::inner(input.clone());
    let weight_inner = <CudaCubeAutodiffBackend<C> as AutodiffBackend>::inner(weight.clone());
    let [batch, input_heads, time, _] = input_inner.meta.shape.dims::<4>();
    let heads = weight_codes.meta.shape.dims::<3>()[0];
    let [outer, _, embd, latent] = weight_inner.meta.shape.dims::<4>();
    let input_device = input_codes.device.clone();
    let input_codes_tensor = BurnTensor::<CudaCubeBackend, 4, Int>::from_primitive(
        try_cast_int_backend::<CudaCubeBackend, _>(input_codes.clone())
            .expect("cuda lowrank training input codes tensor"),
    );
    let weight_codes_tensor = BurnTensor::<CudaCubeBackend, 3, Int>::from_primitive(
        try_cast_int_backend::<CudaCubeBackend, _>(weight_codes.clone())
            .expect("cuda lowrank training weight codes tensor"),
    );
    let grad_input_state = if projection_scale.is_none() && pack_activation_state_to_host {
        PackedLowrankGradInputStateCuda::WeightCodes {
            weight_codes: weight_codes.clone(),
            weight_scale,
        }
    } else {
        let weight_float =
            BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(weight_inner));
        let weight_transposed_float = try_cast_float_primitive::<CudaCubeBackend, _>(
            weight_float
                .slice([0..outer, 0..heads, 0..embd, 0..latent])
                .reshape([heads, embd, latent])
                .swap_dims(1, 2)
                .into_primitive()
                .tensor(),
        )
        .expect("cuda lowrank backward transposed float weight");
        PackedLowrankGradInputStateCuda::WeightTransposedFloat {
            weight_transposed_float,
        }
    };
    let (input_codes_state, packed_input_forward) =
        pack_activation_codes_state_cuda(input_codes, pack_activation_state_to_host);
    let output = if let Some(projection_scale_tensor) = projection_scale.clone() {
        let projection_scale = BurnTensor::<CudaCubeBackend, 1>::from_primitive(
            TensorPrimitive::Float(projection_scale_tensor.clone()),
        );
        packed_input_forward
            .or_else(|| {
                packed_lowrank_input_tensor_from_activation_state_cuda(
                    &input_codes_state,
                    &input_device,
                )
            })
            .and_then(|packed_input| {
                try_raw_cuda_packed_lowrank_projection_prepacked_input_device_scale(
                    &packed_input,
                    &weight_codes_tensor,
                    &projection_scale,
                    1.0,
                    latent_out,
                )
            })
            .or_else(|| {
                try_raw_cuda_packed_lowrank_projection_device_scale(
                    &input_codes_tensor,
                    &weight_codes_tensor,
                    &projection_scale,
                    1.0,
                    latent_out,
                )
            })
            .and_then(|tensor| {
                try_cast_float_primitive::<CudaCubeBackend, CubeTensor<CudaRuntime>>(
                    tensor.into_primitive().tensor(),
                )
            })
            .unwrap_or_else(|| {
                try_cast_float_primitive::<CudaCubeBackend, CubeTensor<CudaRuntime>>(
                    packed_lowrank_projection_device_reference(
                        input_codes_tensor
                            .clone()
                            .float()
                            .mul(projection_scale.clone().reshape([1, 1, 1, 1])),
                        weight_codes_tensor.clone(),
                        1.0,
                        latent_out,
                    )
                    .into_primitive()
                    .tensor(),
                )
                .expect("cuda lowrank device-scale reference output primitive")
            })
    } else {
        packed_input_forward
            .or_else(|| {
                packed_lowrank_input_tensor_from_activation_state_cuda(
                    &input_codes_state,
                    &input_device,
                )
            })
            .and_then(|packed_input| {
                try_raw_cuda_packed_lowrank_projection_prepacked_input(
                    &packed_input,
                    &weight_codes_tensor,
                    activation_scale,
                    weight_scale,
                    latent_out,
                )
            })
            .or_else(|| {
                try_raw_cuda_packed_lowrank_projection(
                    &input_codes_tensor,
                    &weight_codes_tensor,
                    activation_scale,
                    weight_scale,
                    latent_out,
                )
            })
            .and_then(|tensor| {
                try_cast_float_primitive::<CudaCubeBackend, CubeTensor<CudaRuntime>>(
                    tensor.into_primitive().tensor(),
                )
            })
            .unwrap_or_else(|| {
                let restored_input_codes =
                    restore_activation_codes_state_cuda(&input_codes_state, &input_device);
                let restored_input_codes = try_cast_int_primitive::<
                    CudaCubeBackend,
                    CubeTensor<CudaRuntime>,
                >(restored_input_codes.into_primitive())
                .expect("cuda lowrank restored activation codes primitive");
                let params = create_lowrank_params_cuda(
                    &input_device,
                    batch,
                    input_heads,
                    heads,
                    time,
                    input_inner.meta.shape.dims::<4>()[3],
                    activation_scale,
                    weight_scale,
                    latent_out,
                );
                packed_lowrank_projection_cube_runtime::<CudaRuntime>(
                    restored_input_codes,
                    weight_codes.clone(),
                    params,
                    batch,
                    heads,
                    time,
                    latent_out,
                )
            })
    };
    let output = if let Some(threshold) = relu_threshold {
        let activated = activation::relu(
            BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(output))
                .sub_scalar(threshold),
        );
        try_cast_float_primitive::<CudaCubeBackend, CubeTensor<CudaRuntime>>(
            activated.into_primitive().tensor(),
        )
        .expect("cuda packed lowrank relu output primitive")
    } else {
        output
    };
    let shape = PackedLowrankTrainingShape { input_heads };
    match FusedPackedLowrankBackward::<CudaCubeBackend>(PhantomData)
        .prepare::<C>([input.node.clone(), weight.node.clone()])
        .memory_bound()
        .retro_forward(RetroPackedLowrankCuda {
            input_codes: input_codes_state.clone(),
            weight_codes: weight_codes.clone(),
            activation_scale,
            weight_scale,
            projection_scale: projection_scale.clone(),
            latent_out,
            relu_threshold,
        })
        .parents([&input, &weight])
        .stateful()
    {
        OpsKind::Tracked(mut prep) => {
            prep.checkpoint(&input);
            prep.checkpoint(&weight);
            prep.finish(
                PackedLowrankTrainingStateCuda {
                    input_codes: input_codes_state,
                    weight_codes,
                    grad_input_state,
                    activation_scale,
                    weight_scale,
                    projection_scale,
                    relu_threshold,
                    shape,
                },
                output,
            )
        }
        OpsKind::UnTracked(prep) => prep.finish(output),
    }
}

fn fused_packed_decoder_tail_training_autodiff_wgpu<C: CheckpointStrategy>(
    y_neuron: WgpuCubeAutodiffTensor<C>,
    decoder: WgpuCubeAutodiffTensor<C>,
    y_codes: CubeTensor<WgpuRuntime>,
    weight_codes: CubeTensor<WgpuRuntime>,
    activation_scale: f32,
    weight_scale: f32,
    pack_activation_state_to_host: bool,
) -> WgpuCubeAutodiffTensor<C> {
    let y_inner = <WgpuCubeAutodiffBackend<C> as AutodiffBackend>::inner(y_neuron.clone());
    let [batch, heads, time, latent] = y_inner.meta.shape.dims::<4>();
    let decoder_tensor = BurnTensor::<WgpuCubeAutodiffBackend<C>, 2>::from_primitive(
        TensorPrimitive::Float(decoder.clone()),
    );
    let dim = decoder_tensor.shape().dims::<2>()[1];
    let artifact_latent_per_head = weight_codes.meta.shape.dims::<2>()[0] / heads;
    let output = packed_decoder_tail_packed_dot_wgsl_runtime(
        y_codes.clone(),
        weight_codes.clone(),
        batch,
        heads,
        time,
        latent,
        artifact_latent_per_head,
        dim,
        activation_scale,
        weight_scale,
    )
    .unwrap_or_else(|_| {
        let params = create_decoder_tail_params_wgpu(
            &y_codes.device,
            batch,
            heads,
            time,
            latent,
            artifact_latent_per_head,
            dim,
            activation_scale,
            weight_scale,
        );
        packed_decoder_tail_cube_runtime::<WgpuRuntime>(
            y_codes.clone(),
            weight_codes.clone(),
            params,
            batch,
            time,
            dim,
        )
    });
    let shape = PackedDecoderTailTrainingShape { heads, latent };
    let y_codes_state = pack_activation_codes_state_wgpu(y_codes, pack_activation_state_to_host);
    match FusedPackedDecoderTailBackward::<WgpuCubeBackend>(PhantomData)
        .prepare::<C>([y_neuron.node.clone(), decoder.node.clone()])
        .memory_bound()
        .retro_forward(RetroPackedDecoderTailWgpu {
            y_codes: y_codes_state.clone(),
            weight_codes: weight_codes.clone(),
            activation_scale,
            weight_scale,
        })
        .parents([&y_neuron, &decoder])
        .stateful()
    {
        OpsKind::Tracked(mut prep) => {
            prep.checkpoint(&y_neuron);
            prep.checkpoint(&decoder);
            prep.finish(
                PackedDecoderTailTrainingStateWgpu {
                    y_codes: y_codes_state,
                    weight_codes,
                    activation_scale,
                    weight_scale,
                    shape,
                },
                output,
            )
        }
        OpsKind::UnTracked(prep) => prep.finish(output),
    }
}

#[cfg(feature = "cuda")]
fn fused_packed_decoder_tail_training_autodiff_cuda<C: CheckpointStrategy>(
    y_neuron: CudaCubeAutodiffTensor<C>,
    decoder: CudaCubeAutodiffTensor<C>,
    y_codes: CubeTensor<CudaRuntime>,
    weight_codes: CubeTensor<CudaRuntime>,
    activation_scale: f32,
    weight_scale: f32,
    pack_activation_state_to_host: bool,
) -> CudaCubeAutodiffTensor<C> {
    let y_inner = <CudaCubeAutodiffBackend<C> as AutodiffBackend>::inner(y_neuron.clone());
    let [batch, heads, time, latent] = y_inner.meta.shape.dims::<4>();
    let y_device = y_codes.device.clone();
    let decoder_tensor = BurnTensor::<CudaCubeAutodiffBackend<C>, 2>::from_primitive(
        TensorPrimitive::Float(decoder.clone()),
    );
    let y_codes_tensor = BurnTensor::<CudaCubeBackend, 4, Int>::from_primitive(
        try_cast_int_backend::<CudaCubeBackend, _>(y_codes.clone())
            .expect("cuda decoder-tail training activation codes tensor"),
    );
    let weight_codes_tensor = BurnTensor::<CudaCubeBackend, 2, Int>::from_primitive(
        try_cast_int_backend::<CudaCubeBackend, _>(weight_codes.clone())
            .expect("cuda decoder-tail training weight codes tensor"),
    );
    let dim = decoder_tensor.shape().dims::<2>()[1];
    let grad_input_state = if pack_activation_state_to_host {
        PackedDecoderTailGradInputStateCuda::WeightCodes {
            weight_codes: weight_codes.clone(),
            weight_scale,
        }
    } else {
        PackedDecoderTailGradInputStateCuda::DecoderFloat {
            decoder_float: <CudaCubeAutodiffBackend<C> as AutodiffBackend>::inner(decoder.clone()),
        }
    };
    let artifact_latent_per_head = weight_codes.meta.shape.dims::<2>()[0] / heads;
    let (y_codes_state, packed_y_forward) =
        pack_activation_codes_state_cuda(y_codes, pack_activation_state_to_host);
    let output = packed_y_forward
        .or_else(|| {
            packed_decoder_input_tensor_from_activation_state_cuda(&y_codes_state, &y_device)
        })
        .and_then(|packed_input| {
            try_raw_cuda_packed_decoder_tail_prepacked_input(
                &packed_input,
                &weight_codes_tensor,
                activation_scale,
                weight_scale,
            )
        })
        .or_else(|| {
            try_raw_cuda_packed_decoder_tail(
                &y_codes_tensor,
                &weight_codes_tensor,
                activation_scale,
                weight_scale,
            )
        })
        .and_then(|tensor| {
            try_cast_float_primitive::<CudaCubeBackend, CubeTensor<CudaRuntime>>(
                tensor.into_primitive().tensor(),
            )
        })
        .unwrap_or_else(|| {
            let restored_y_codes = restore_activation_codes_state_cuda(&y_codes_state, &y_device);
            let restored_y_codes =
                try_cast_int_primitive::<CudaCubeBackend, CubeTensor<CudaRuntime>>(
                    restored_y_codes.into_primitive(),
                )
                .expect("cuda decoder restored activation codes primitive");
            let params = create_decoder_tail_params_cuda(
                &y_device,
                batch,
                heads,
                time,
                latent,
                artifact_latent_per_head,
                dim,
                activation_scale,
                weight_scale,
            );
            packed_decoder_tail_cube_runtime::<CudaRuntime>(
                restored_y_codes,
                weight_codes.clone(),
                params,
                batch,
                time,
                dim,
            )
        });
    let shape = PackedDecoderTailTrainingShape { heads, latent };
    match FusedPackedDecoderTailBackward::<CudaCubeBackend>(PhantomData)
        .prepare::<C>([y_neuron.node.clone(), decoder.node.clone()])
        .memory_bound()
        .retro_forward(RetroPackedDecoderTailCuda {
            y_codes: y_codes_state.clone(),
            weight_codes: weight_codes.clone(),
            activation_scale,
            weight_scale,
        })
        .parents([&y_neuron, &decoder])
        .stateful()
    {
        OpsKind::Tracked(mut prep) => {
            prep.checkpoint(&y_neuron);
            prep.checkpoint(&decoder);
            prep.finish(
                PackedDecoderTailTrainingStateCuda {
                    y_codes: y_codes_state,
                    grad_input_state,
                    activation_scale,
                    shape,
                },
                output,
            )
        }
        OpsKind::UnTracked(prep) => prep.finish(output),
    }
}

pub fn try_fused_packed_lowrank_training_autodiff<B: BackendTrait>(
    input: &BurnTensor<B, 4>,
    weight: &BurnTensor<B, 4>,
    input_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 3, Int>,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
    pack_activation_state_to_host: bool,
    relu_threshold: Option<f32>,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
    B::IntTensorPrimitive: 'static,
{
    let use_balanced_checkpointing = core::any::type_name::<B>().contains("BalancedCheckpointing");
    if let (Some(input_ad), Some(weight_ad), Some(input_codes_inner), Some(weight_codes_inner)) = (
        try_cast_float_primitive::<B, WgpuCubeAutodiffTensor>(
            input.clone().into_primitive().tensor(),
        ),
        try_cast_float_primitive::<B, WgpuCubeAutodiffTensor>(
            weight.clone().into_primitive().tensor(),
        ),
        try_cast_int_primitive::<B, CubeTensor<WgpuRuntime>>(input_codes.clone().into_primitive()),
        try_cast_int_primitive::<B, CubeTensor<WgpuRuntime>>(weight_codes.clone().into_primitive()),
    ) {
        let output = if use_balanced_checkpointing {
            fused_packed_lowrank_training_autodiff_wgpu::<BalancedCheckpointing>(
                input_ad,
                weight_ad,
                input_codes_inner,
                weight_codes_inner,
                activation_scale,
                weight_scale,
                latent_out,
                pack_activation_state_to_host,
                relu_threshold,
            )
        } else {
            fused_packed_lowrank_training_autodiff_wgpu::<NoCheckpointing>(
                input_ad,
                weight_ad,
                input_codes_inner,
                weight_codes_inner,
                activation_scale,
                weight_scale,
                latent_out,
                pack_activation_state_to_host,
                relu_threshold,
            )
        };
        return try_cast_float_backend::<B, _>(output)
            .map(|prim| BurnTensor::from_primitive(TensorPrimitive::Float(prim)));
    }
    #[cfg(feature = "cuda")]
    if let (Some(input_ad), Some(weight_ad), Some(input_codes_inner), Some(weight_codes_inner)) = (
        try_cast_float_primitive::<B, CudaCubeAutodiffTensor>(
            input.clone().into_primitive().tensor(),
        ),
        try_cast_float_primitive::<B, CudaCubeAutodiffTensor>(
            weight.clone().into_primitive().tensor(),
        ),
        try_cast_int_primitive::<B, CubeTensor<CudaRuntime>>(input_codes.clone().into_primitive()),
        try_cast_int_primitive::<B, CubeTensor<CudaRuntime>>(weight_codes.clone().into_primitive()),
    ) {
        let output = if use_balanced_checkpointing {
            fused_packed_lowrank_training_autodiff_cuda::<BalancedCheckpointing>(
                input_ad,
                weight_ad,
                input_codes_inner,
                weight_codes_inner,
                activation_scale,
                weight_scale,
                None,
                latent_out,
                pack_activation_state_to_host,
                relu_threshold,
            )
        } else {
            fused_packed_lowrank_training_autodiff_cuda::<NoCheckpointing>(
                input_ad,
                weight_ad,
                input_codes_inner,
                weight_codes_inner,
                activation_scale,
                weight_scale,
                None,
                latent_out,
                pack_activation_state_to_host,
                relu_threshold,
            )
        };
        return try_cast_float_backend::<B, _>(output)
            .map(|prim| BurnTensor::from_primitive(TensorPrimitive::Float(prim)));
    }
    #[cfg(feature = "cuda")]
    {
        let no_ckpt_input = try_cast_float_primitive::<B, CudaCubeAutodiffTensor<NoCheckpointing>>(
            input.clone().into_primitive().tensor(),
        )
        .is_some();
        let no_ckpt_weight =
            try_cast_float_primitive::<B, CudaCubeAutodiffTensor<NoCheckpointing>>(
                weight.clone().into_primitive().tensor(),
            )
            .is_some();
        let bal_input =
            try_cast_float_primitive::<B, CudaCubeAutodiffTensor<BalancedCheckpointing>>(
                input.clone().into_primitive().tensor(),
            )
            .is_some();
        let bal_weight =
            try_cast_float_primitive::<B, CudaCubeAutodiffTensor<BalancedCheckpointing>>(
                weight.clone().into_primitive().tensor(),
            )
            .is_some();
        let input_codes_cuda = try_cast_int_primitive::<B, CubeTensor<CudaRuntime>>(
            input_codes.clone().into_primitive(),
        )
        .is_some();
        let weight_codes_cuda = try_cast_int_primitive::<B, CubeTensor<CudaRuntime>>(
            weight_codes.clone().into_primitive(),
        )
        .is_some();
        emit_lowrank_training_debug_once(|| {
            format!(
                "low-bit fused training cast miss: backend={} float_prim={} int_prim={} no_ckpt_input={} no_ckpt_weight={} bal_input={} bal_weight={} input_codes_cuda={} weight_codes_cuda={}",
                core::any::type_name::<B>(),
                core::any::type_name::<B::FloatTensorPrimitive>(),
                core::any::type_name::<B::IntTensorPrimitive>(),
                no_ckpt_input,
                no_ckpt_weight,
                bal_input,
                bal_weight,
                input_codes_cuda,
                weight_codes_cuda,
            )
        });
    }
    None
}

pub fn try_fused_packed_lowrank_training_autodiff_cuda_device_projection_scale<B: BackendTrait>(
    _input: &BurnTensor<B, 4>,
    _weight: &BurnTensor<B, 4>,
    _input_codes: &BurnTensor<B, 4, Int>,
    _weight_codes: &BurnTensor<B, 3, Int>,
    _activation_scale: f32,
    _projection_scale: &BurnTensor<B, 1>,
    _latent_out: usize,
    _pack_activation_state_to_host: bool,
    _relu_threshold: Option<f32>,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
    B::IntTensorPrimitive: 'static,
{
    let _use_balanced_checkpointing = core::any::type_name::<B>().contains("BalancedCheckpointing");
    #[cfg(feature = "cuda")]
    fn try_cast_cuda_projection_scale<B: BackendTrait, C: CheckpointStrategy>(
        projection_scale: &BurnTensor<B, 1>,
    ) -> Option<CubeTensor<CudaRuntime>>
    where
        B::FloatTensorPrimitive: 'static,
    {
        try_cast_float_primitive::<B, CubeTensor<CudaRuntime>>(
            projection_scale.clone().into_primitive().tensor(),
        )
        .or_else(|| {
            try_cast_float_primitive::<B, CudaCubeAutodiffTensor<C>>(
                projection_scale.clone().into_primitive().tensor(),
            )
            .map(|tensor| <CudaCubeAutodiffBackend<C> as AutodiffBackend>::inner(tensor))
        })
    }
    #[cfg(feature = "cuda")]
    if _use_balanced_checkpointing {
        let output = if let (
            Some(input_ad),
            Some(weight_ad),
            Some(input_codes_inner),
            Some(weight_codes_inner),
            Some(projection_scale_inner),
        ) = (
            try_cast_float_primitive::<B, CudaCubeAutodiffTensor<BalancedCheckpointing>>(
                _input.clone().into_primitive().tensor(),
            ),
            try_cast_float_primitive::<B, CudaCubeAutodiffTensor<BalancedCheckpointing>>(
                _weight.clone().into_primitive().tensor(),
            ),
            try_cast_int_primitive::<B, CubeTensor<CudaRuntime>>(
                _input_codes.clone().into_primitive(),
            ),
            try_cast_int_primitive::<B, CubeTensor<CudaRuntime>>(
                _weight_codes.clone().into_primitive(),
            ),
            try_cast_cuda_projection_scale::<B, BalancedCheckpointing>(_projection_scale),
        ) {
            fused_packed_lowrank_training_autodiff_cuda::<BalancedCheckpointing>(
                input_ad,
                weight_ad,
                input_codes_inner,
                weight_codes_inner,
                _activation_scale,
                1.0,
                Some(projection_scale_inner),
                _latent_out,
                _pack_activation_state_to_host,
                _relu_threshold,
            )
        } else {
            return None;
        };
        return try_cast_float_backend::<B, _>(output)
            .map(|prim| BurnTensor::from_primitive(TensorPrimitive::Float(prim)));
    }
    #[cfg(feature = "cuda")]
    if let (
        Some(input_ad),
        Some(weight_ad),
        Some(input_codes_inner),
        Some(weight_codes_inner),
        Some(projection_scale_inner),
    ) = (
        try_cast_float_primitive::<B, CudaCubeAutodiffTensor<NoCheckpointing>>(
            _input.clone().into_primitive().tensor(),
        ),
        try_cast_float_primitive::<B, CudaCubeAutodiffTensor<NoCheckpointing>>(
            _weight.clone().into_primitive().tensor(),
        ),
        try_cast_int_primitive::<B, CubeTensor<CudaRuntime>>(_input_codes.clone().into_primitive()),
        try_cast_int_primitive::<B, CubeTensor<CudaRuntime>>(
            _weight_codes.clone().into_primitive(),
        ),
        try_cast_cuda_projection_scale::<B, NoCheckpointing>(_projection_scale),
    ) {
        let output = fused_packed_lowrank_training_autodiff_cuda::<NoCheckpointing>(
            input_ad,
            weight_ad,
            input_codes_inner,
            weight_codes_inner,
            _activation_scale,
            1.0,
            Some(projection_scale_inner),
            _latent_out,
            _pack_activation_state_to_host,
            _relu_threshold,
        );
        return try_cast_float_backend::<B, _>(output)
            .map(|prim| BurnTensor::from_primitive(TensorPrimitive::Float(prim)));
    }
    #[cfg(feature = "cuda")]
    {
        let no_ckpt_input = try_cast_float_primitive::<B, CudaCubeAutodiffTensor<NoCheckpointing>>(
            _input.clone().into_primitive().tensor(),
        )
        .is_some();
        let no_ckpt_weight =
            try_cast_float_primitive::<B, CudaCubeAutodiffTensor<NoCheckpointing>>(
                _weight.clone().into_primitive().tensor(),
            )
            .is_some();
        let bal_input =
            try_cast_float_primitive::<B, CudaCubeAutodiffTensor<BalancedCheckpointing>>(
                _input.clone().into_primitive().tensor(),
            )
            .is_some();
        let bal_weight =
            try_cast_float_primitive::<B, CudaCubeAutodiffTensor<BalancedCheckpointing>>(
                _weight.clone().into_primitive().tensor(),
            )
            .is_some();
        let input_codes_cuda = try_cast_int_primitive::<B, CubeTensor<CudaRuntime>>(
            _input_codes.clone().into_primitive(),
        )
        .is_some();
        let weight_codes_cuda = try_cast_int_primitive::<B, CubeTensor<CudaRuntime>>(
            _weight_codes.clone().into_primitive(),
        )
        .is_some();
        let proj_no_ckpt =
            try_cast_cuda_projection_scale::<B, NoCheckpointing>(_projection_scale).is_some();
        let proj_bal =
            try_cast_cuda_projection_scale::<B, BalancedCheckpointing>(_projection_scale).is_some();
        emit_lowrank_training_debug_once(|| {
            format!(
                "low-bit fused training device-scale cast miss: backend={} float_prim={} int_prim={} no_ckpt_input={} no_ckpt_weight={} bal_input={} bal_weight={} input_codes_cuda={} weight_codes_cuda={} proj_no_ckpt={} proj_bal={}",
                core::any::type_name::<B>(),
                core::any::type_name::<B::FloatTensorPrimitive>(),
                core::any::type_name::<B::IntTensorPrimitive>(),
                no_ckpt_input,
                no_ckpt_weight,
                bal_input,
                bal_weight,
                input_codes_cuda,
                weight_codes_cuda,
                proj_no_ckpt,
                proj_bal,
            )
        });
    }
    None
}

pub fn try_fused_packed_decoder_tail_training_autodiff<B: BackendTrait>(
    y_neuron: &BurnTensor<B, 4>,
    decoder: &BurnTensor<B, 2>,
    y_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 2, Int>,
    activation_scale: f32,
    weight_scale: f32,
    pack_activation_state_to_host: bool,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
    B::IntTensorPrimitive: 'static,
{
    let use_balanced_checkpointing = core::any::type_name::<B>().contains("BalancedCheckpointing");
    if let (Some(y_ad), Some(decoder_ad), Some(y_codes_inner), Some(weight_codes_inner)) = (
        try_cast_float_primitive::<B, WgpuCubeAutodiffTensor>(
            y_neuron.clone().into_primitive().tensor(),
        ),
        try_cast_float_primitive::<B, WgpuCubeAutodiffTensor>(
            decoder.clone().into_primitive().tensor(),
        ),
        try_cast_int_primitive::<B, CubeTensor<WgpuRuntime>>(y_codes.clone().into_primitive()),
        try_cast_int_primitive::<B, CubeTensor<WgpuRuntime>>(weight_codes.clone().into_primitive()),
    ) {
        let output = if use_balanced_checkpointing {
            fused_packed_decoder_tail_training_autodiff_wgpu::<BalancedCheckpointing>(
                y_ad,
                decoder_ad,
                y_codes_inner,
                weight_codes_inner,
                activation_scale,
                weight_scale,
                pack_activation_state_to_host,
            )
        } else {
            fused_packed_decoder_tail_training_autodiff_wgpu::<NoCheckpointing>(
                y_ad,
                decoder_ad,
                y_codes_inner,
                weight_codes_inner,
                activation_scale,
                weight_scale,
                pack_activation_state_to_host,
            )
        };
        return try_cast_float_backend::<B, _>(output)
            .map(|prim| BurnTensor::from_primitive(TensorPrimitive::Float(prim)));
    }
    #[cfg(feature = "cuda")]
    if let (Some(y_ad), Some(decoder_ad), Some(y_codes_inner), Some(weight_codes_inner)) = (
        try_cast_float_primitive::<B, CudaCubeAutodiffTensor>(
            y_neuron.clone().into_primitive().tensor(),
        ),
        try_cast_float_primitive::<B, CudaCubeAutodiffTensor>(
            decoder.clone().into_primitive().tensor(),
        ),
        try_cast_int_primitive::<B, CubeTensor<CudaRuntime>>(y_codes.clone().into_primitive()),
        try_cast_int_primitive::<B, CubeTensor<CudaRuntime>>(weight_codes.clone().into_primitive()),
    ) {
        let output = if use_balanced_checkpointing {
            fused_packed_decoder_tail_training_autodiff_cuda::<BalancedCheckpointing>(
                y_ad,
                decoder_ad,
                y_codes_inner,
                weight_codes_inner,
                activation_scale,
                weight_scale,
                pack_activation_state_to_host,
            )
        } else {
            fused_packed_decoder_tail_training_autodiff_cuda::<NoCheckpointing>(
                y_ad,
                decoder_ad,
                y_codes_inner,
                weight_codes_inner,
                activation_scale,
                weight_scale,
                pack_activation_state_to_host,
            )
        };
        return try_cast_float_backend::<B, _>(output)
            .map(|prim| BurnTensor::from_primitive(TensorPrimitive::Float(prim)));
    }
    None
}
