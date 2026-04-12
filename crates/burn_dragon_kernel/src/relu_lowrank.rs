use std::any::{Any, TypeId};
use std::marker::PhantomData;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use burn::tensor::Tensor as BurnTensor;
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{DType, Shape, TensorPrimitive};
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::{base::Checkpointer, strategy::NoCheckpointing};
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};
use burn_cubecl::cubecl;
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_cubecl::cubecl::{prelude::*, server::KernelArguments};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::kernel::into_contiguous_aligned;
use burn_cubecl::kernel::matmul::{MatmulStrategy, matmul};
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{BoolElement, CubeRuntime};
use burn_fusion::{Fusion, FusionTensor};
use burn_std::Metadata;
use burn_wgpu::{CubeBackend, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};

use crate::fusion_compat::register_fusion_float_tensor;
use crate::profiling::{
    KernelProfileSite, KernelProfileSnapshot, profile_enabled, profile_record, profile_reset,
    profile_snapshot,
};

mod forward_runtime;

use self::forward_runtime::{
    relu_lowrank_wgsl_runtime, try_direct_path_autodiff_cube_runtime, try_direct_path_runtime,
    try_fusion_path_autodiff_runtime, try_fusion_path_runtime,
};

const WORKGROUP_SIZE_X: u32 = 64;
const RELU_LOWRANK_SHADER: &str = include_str!("relu_lowrank.wgsl");
const RELU_LOWRANK_GRAD_INPUT_SHADER: &str = include_str!("relu_lowrank_grad_input.wgsl");
const RELU_LOWRANK_GRAD_INPUT_TILED_SHADER: &str =
    include_str!("relu_lowrank_grad_input_tiled.wgsl");

type WgpuCubeBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type WgpuCubeAutodiffBackend = Autodiff<WgpuCubeBackend>;
type WgpuCubeAutodiffTensor = <WgpuCubeAutodiffBackend as BackendTrait>::FloatTensorPrimitive;
type WgpuFusionBackend<BT> = Fusion<CubeBackend<WgpuRuntime, f32, i32, BT>>;
type WgpuFusionAutodiffBackend<BT> = Autodiff<WgpuFusionBackend<BT>>;
type WgpuFusionAutodiffTensor<BT> =
    <WgpuFusionAutodiffBackend<BT> as BackendTrait>::FloatTensorPrimitive;
#[cfg(feature = "cuda")]
type CudaCubeBackend = CubeBackend<CudaRuntime, f32, i32, u8>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffBackend = Autodiff<CudaCubeBackend>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffTensor = <CudaCubeAutodiffBackend as BackendTrait>::FloatTensorPrimitive;
#[cfg(feature = "cuda")]
type CudaFusionBackend<BT> = Fusion<CubeBackend<CudaRuntime, f32, i32, BT>>;
#[cfg(feature = "cuda")]
type CudaFusionAutodiffBackend<BT> = Autodiff<CudaFusionBackend<BT>>;
#[cfg(feature = "cuda")]
type CudaFusionAutodiffTensor<BT> =
    <CudaFusionAutodiffBackend<BT> as BackendTrait>::FloatTensorPrimitive;

pub type LowrankProjectionProfileSnapshot = KernelProfileSnapshot;

#[derive(Clone, Copy, Debug, Default)]
pub struct LowrankForwardRouteProfileSnapshot {
    pub attempts: u64,
    pub wgpu_fusion_autodiff: u64,
    pub wgpu_direct_autodiff: u64,
    pub wgpu_fusion_runtime: u64,
    pub wgpu_direct_runtime: u64,
    pub cuda_fusion_autodiff: u64,
    pub cuda_direct_autodiff: u64,
    pub cuda_fusion_runtime: u64,
    pub cuda_direct_runtime: u64,
}

impl LowrankForwardRouteProfileSnapshot {
    pub fn successes(&self) -> u64 {
        self.wgpu_fusion_autodiff
            .saturating_add(self.wgpu_direct_autodiff)
            .saturating_add(self.wgpu_fusion_runtime)
            .saturating_add(self.wgpu_direct_runtime)
            .saturating_add(self.cuda_fusion_autodiff)
            .saturating_add(self.cuda_direct_autodiff)
            .saturating_add(self.cuda_fusion_runtime)
            .saturating_add(self.cuda_direct_runtime)
    }

    pub fn fallbacks(&self) -> u64 {
        self.attempts.saturating_sub(self.successes())
    }
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
enum LowrankForwardRouteKind {
    WgpuFusionAutodiff,
    WgpuDirectAutodiff,
    WgpuFusionRuntime,
    WgpuDirectRuntime,
    CudaFusionAutodiff,
    CudaDirectAutodiff,
    CudaFusionRuntime,
    CudaDirectRuntime,
}

static RELU_LOWRANK_FORWARD_PROFILE: KernelProfileSite = KernelProfileSite::new();
static RELU_LOWRANK_GRAD_INPUT_PROFILE: KernelProfileSite = KernelProfileSite::new();
static RELU_LOWRANK_GRAD_WEIGHT_PROFILE: KernelProfileSite = KernelProfileSite::new();
static RELU_LOWRANK_FORWARD_ROUTE_PROFILE: OnceLock<Mutex<LowrankForwardRouteProfileSnapshot>> =
    OnceLock::new();

pub fn relu_lowrank_forward_profile_reset() {
    profile_reset(&RELU_LOWRANK_FORWARD_PROFILE);
}

pub fn relu_lowrank_forward_profile_snapshot() -> LowrankProjectionProfileSnapshot {
    profile_snapshot(&RELU_LOWRANK_FORWARD_PROFILE)
}

pub fn relu_lowrank_forward_route_profile_reset() {
    if let Ok(mut state) = RELU_LOWRANK_FORWARD_ROUTE_PROFILE
        .get_or_init(|| Mutex::new(LowrankForwardRouteProfileSnapshot::default()))
        .lock()
    {
        *state = LowrankForwardRouteProfileSnapshot::default();
    }
}

pub fn relu_lowrank_forward_route_profile_snapshot() -> LowrankForwardRouteProfileSnapshot {
    RELU_LOWRANK_FORWARD_ROUTE_PROFILE
        .get_or_init(|| Mutex::new(LowrankForwardRouteProfileSnapshot::default()))
        .lock()
        .map(|state| *state)
        .unwrap_or_default()
}

fn relu_lowrank_forward_route_profile_record_attempt() {
    if let Ok(mut state) = RELU_LOWRANK_FORWARD_ROUTE_PROFILE
        .get_or_init(|| Mutex::new(LowrankForwardRouteProfileSnapshot::default()))
        .lock()
    {
        state.attempts = state.attempts.saturating_add(1);
    }
}

fn relu_lowrank_forward_route_profile_record_route(route: LowrankForwardRouteKind) {
    if let Ok(mut state) = RELU_LOWRANK_FORWARD_ROUTE_PROFILE
        .get_or_init(|| Mutex::new(LowrankForwardRouteProfileSnapshot::default()))
        .lock()
    {
        match route {
            LowrankForwardRouteKind::WgpuFusionAutodiff => {
                state.wgpu_fusion_autodiff = state.wgpu_fusion_autodiff.saturating_add(1);
            }
            LowrankForwardRouteKind::WgpuDirectAutodiff => {
                state.wgpu_direct_autodiff = state.wgpu_direct_autodiff.saturating_add(1);
            }
            LowrankForwardRouteKind::WgpuFusionRuntime => {
                state.wgpu_fusion_runtime = state.wgpu_fusion_runtime.saturating_add(1);
            }
            LowrankForwardRouteKind::WgpuDirectRuntime => {
                state.wgpu_direct_runtime = state.wgpu_direct_runtime.saturating_add(1);
            }
            LowrankForwardRouteKind::CudaFusionAutodiff => {
                state.cuda_fusion_autodiff = state.cuda_fusion_autodiff.saturating_add(1);
            }
            LowrankForwardRouteKind::CudaDirectAutodiff => {
                state.cuda_direct_autodiff = state.cuda_direct_autodiff.saturating_add(1);
            }
            LowrankForwardRouteKind::CudaFusionRuntime => {
                state.cuda_fusion_runtime = state.cuda_fusion_runtime.saturating_add(1);
            }
            LowrankForwardRouteKind::CudaDirectRuntime => {
                state.cuda_direct_runtime = state.cuda_direct_runtime.saturating_add(1);
            }
        }
    }
}

pub fn relu_lowrank_grad_input_profile_reset() {
    profile_reset(&RELU_LOWRANK_GRAD_INPUT_PROFILE);
}

pub fn relu_lowrank_grad_input_profile_snapshot() -> LowrankProjectionProfileSnapshot {
    profile_snapshot(&RELU_LOWRANK_GRAD_INPUT_PROFILE)
}

pub fn relu_lowrank_grad_weight_profile_reset() {
    profile_reset(&RELU_LOWRANK_GRAD_WEIGHT_PROFILE);
}

pub fn relu_lowrank_grad_weight_profile_snapshot() -> LowrankProjectionProfileSnapshot {
    profile_snapshot(&RELU_LOWRANK_GRAD_WEIGHT_PROFILE)
}

#[derive(Clone, Copy, Debug, Default, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LowrankGradInputExecutor {
    #[default]
    Auto,
    Kernel,
    KernelTiled,
    AlignedMatmul,
    Direct,
}

#[derive(Clone, Copy, Debug)]
struct LowrankProjectionShape {
    batch: usize,
    input_heads: usize,
    heads: usize,
    time: usize,
    embd: usize,
    latent: usize,
    threshold: f32,
    has_mask: bool,
    grad_input_executor: LowrankGradInputExecutor,
}

impl LowrankProjectionShape {
    fn from_tensors<B: BackendTrait>(
        input: &BurnTensor<B, 4>,
        weight: &BurnTensor<B, 4>,
        threshold: f32,
        sparse_mask: Option<&BurnTensor<B, 4>>,
        grad_input_executor: LowrankGradInputExecutor,
    ) -> Option<Self> {
        let [batch, input_heads, time, embd] = input.shape().dims::<4>();
        let [weight_batch, heads, weight_embd, latent] = weight.shape().dims::<4>();
        if batch == 0
            || input_heads == 0
            || time == 0
            || embd == 0
            || weight_batch != 1
            || heads == 0
            || latent == 0
            || embd != weight_embd
            || !(input_heads == 1 || input_heads == heads)
        {
            return None;
        }
        let has_mask = match sparse_mask {
            Some(mask) => mask.shape().dims::<4>() == [1, 1, 1, latent],
            None => false,
        };
        if sparse_mask.is_some() && !has_mask {
            return None;
        }
        Some(Self {
            batch,
            input_heads,
            heads,
            time,
            embd,
            latent,
            threshold,
            has_mask,
            grad_input_executor,
        })
    }

    fn meta<B: BackendTrait>(&self, device: &B::Device) -> BurnTensor<B, 1> {
        BurnTensor::<B, 1>::from_floats(
            [
                self.batch as f32,
                self.input_heads as f32,
                self.heads as f32,
                self.time as f32,
                self.embd as f32,
                self.latent as f32,
                self.threshold,
                if self.has_mask { 1.0 } else { 0.0 },
            ],
            device,
        )
    }
}

fn single_stream_projection_flat<B: BackendTrait>(
    input: BurnTensor<B, 4>,
    weight: BurnTensor<B, 4>,
) -> Option<BurnTensor<B, 4>> {
    let [batch, streams, time, embd] = input.shape().dims::<4>();
    let [weight_batch, heads, weight_embd, latent] = weight.shape().dims::<4>();
    if streams != 1 || weight_batch != 1 || embd != weight_embd {
        return None;
    }

    let input_flat = input.reshape([batch * time, embd]);
    let weight_flat = weight
        .reshape([heads, embd, latent])
        .swap_dims(0, 1)
        .reshape([embd, heads * latent]);
    let projected = input_flat.matmul(weight_flat);
    Some(
        projected
            .reshape([batch, time, heads, latent])
            .swap_dims(1, 2),
    )
}

fn head_aligned_projection_flat<B: BackendTrait>(
    input: BurnTensor<B, 4>,
    weight: BurnTensor<B, 4>,
) -> Option<BurnTensor<B, 4>> {
    let [batch, heads, time, embd] = input.shape().dims::<4>();
    let [weight_batch, weight_heads, weight_embd, latent] = weight.shape().dims::<4>();
    if weight_batch != 1 || heads != weight_heads || embd != weight_embd {
        return None;
    }

    let input_by_head = input.swap_dims(0, 1).reshape([heads, batch * time, embd]);
    let weight_by_head = weight.reshape([heads, embd, latent]);
    let projected = input_by_head.matmul(weight_by_head);
    Some(
        projected
            .reshape([heads, batch, time, latent])
            .swap_dims(0, 1),
    )
}

fn lowrank_projection_reference<B: BackendTrait>(
    input: BurnTensor<B, 4>,
    weight: BurnTensor<B, 4>,
) -> BurnTensor<B, 4> {
    single_stream_projection_flat(input.clone(), weight.clone())
        .or_else(|| head_aligned_projection_flat(input.clone(), weight.clone()))
        .unwrap_or_else(|| input.matmul(weight))
}

#[cfg(test)]
fn lowrank_projection_reference_forward<B: BackendTrait>(
    input: BurnTensor<B, 4>,
    weight: BurnTensor<B, 4>,
    threshold: f32,
    sparse_mask: Option<BurnTensor<B, 4>>,
) -> BurnTensor<B, 4> {
    let projected = lowrank_projection_reference(input, weight);
    let activated = burn::tensor::activation::relu(projected.sub_scalar(threshold));
    match sparse_mask {
        Some(mask) => activated * mask,
        None => activated,
    }
}

fn relu_lowrank_backward_impl<B: BackendTrait>(
    ops: Ops<
        (
            B::FloatTensorPrimitive,
            B::FloatTensorPrimitive,
            Option<B::FloatTensorPrimitive>,
            LowrankProjectionShape,
        ),
        2,
    >,
    grads: &mut Gradients,
) {
    let grad_output = grads.consume::<B>(&ops.node);
    let (input_inner, weight_inner, mask_inner, shape) = ops.state;
    let parents = ops.parents;

    let input = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(input_inner));
    let weight = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(weight_inner));
    let grad_output = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(grad_output));
    let sparse_mask =
        mask_inner.map(|mask| BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(mask)));

    let projected = lowrank_projection_reference(input.clone(), weight.clone());
    let activation_mask = projected
        .sub_scalar(shape.threshold)
        .greater_elem(0.0)
        .float();
    let mut grad_projected = grad_output * activation_mask;
    if let Some(mask) = sparse_mask {
        grad_projected = grad_projected * mask;
    }

    if let Some(parent) = &parents[0] {
        let grad_input = if let Some(fused) =
            try_lowrank_grad_input_cuda_direct::<B>(&grad_projected, &weight, shape)
        {
            fused
        } else if shape.input_heads == 1 {
            let grad_flat = grad_projected
                .clone()
                .swap_dims(1, 2)
                .reshape([shape.batch * shape.time, shape.heads * shape.latent]);
            let weight_flat = weight
                .clone()
                .reshape([shape.heads, shape.embd, shape.latent])
                .swap_dims(0, 1)
                .reshape([shape.embd, shape.heads * shape.latent]);
            grad_flat
                .matmul(weight_flat.swap_dims(0, 1))
                .reshape([shape.batch, shape.time, shape.embd])
                .reshape([shape.batch, 1, shape.time, shape.embd])
        } else {
            try_head_aligned_grad_input_wgpu(&grad_projected, &weight, shape).unwrap_or_else(|| {
                let grad_by_head = grad_projected.clone().swap_dims(0, 1).reshape([
                    shape.heads,
                    shape.batch * shape.time,
                    shape.latent,
                ]);
                let weight_by_head =
                    weight
                        .clone()
                        .reshape([shape.heads, shape.embd, shape.latent]);
                grad_by_head
                    .matmul(weight_by_head.swap_dims(1, 2))
                    .reshape([shape.heads, shape.batch, shape.time, shape.embd])
                    .swap_dims(0, 1)
            })
        };
        grads.register::<B>(parent.id, grad_input.into_primitive().tensor());
    }

    if let Some(parent) = &parents[1] {
        let grad_weight_start = profile_enabled().then(Instant::now);
        let grad_weight = if shape.input_heads == 1 {
            let input_flat = input
                .clone()
                .reshape([shape.batch, shape.time, shape.embd])
                .reshape([shape.batch * shape.time, shape.embd]);
            let grad_flat = grad_projected
                .clone()
                .swap_dims(1, 2)
                .reshape([shape.batch * shape.time, shape.heads * shape.latent]);
            input_flat
                .swap_dims(0, 1)
                .matmul(grad_flat)
                .reshape([shape.embd, shape.heads, shape.latent])
                .swap_dims(0, 1)
                .reshape([1, shape.heads, shape.embd, shape.latent])
        } else {
            let input_by_head = input.clone().swap_dims(0, 1).reshape([
                shape.heads,
                shape.batch * shape.time,
                shape.embd,
            ]);
            let grad_by_head = grad_projected.clone().swap_dims(0, 1).reshape([
                shape.heads,
                shape.batch * shape.time,
                shape.latent,
            ]);
            input_by_head.swap_dims(1, 2).matmul(grad_by_head).reshape([
                1,
                shape.heads,
                shape.embd,
                shape.latent,
            ])
        };
        if let Some(start) = grad_weight_start {
            profile_record(&RELU_LOWRANK_GRAD_WEIGHT_PROFILE, |state| {
                state.calls = state.calls.saturating_add(1);
                state.total_ns = state.total_ns.saturating_add(start.elapsed().as_nanos());
            });
        }
        grads.register::<B>(parent.id, grad_weight.into_primitive().tensor());
    }
}

#[derive(Debug)]
struct FusedReluLowrankBackward<B>(PhantomData<B>);

impl Backward<WgpuCubeBackend, 2> for FusedReluLowrankBackward<WgpuCubeBackend> {
    type State = (
        CubeTensor<WgpuRuntime>,
        CubeTensor<WgpuRuntime>,
        Option<CubeTensor<WgpuRuntime>>,
        LowrankProjectionShape,
    );

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        relu_lowrank_backward_impl::<WgpuCubeBackend>(ops, grads);
    }
}

impl<BT> Backward<WgpuFusionBackend<BT>, 2> for FusedReluLowrankBackward<WgpuFusionBackend<BT>>
where
    BT: BoolElement + 'static,
{
    type State = (
        FusionTensor<FusionCubeRuntime<WgpuRuntime>>,
        FusionTensor<FusionCubeRuntime<WgpuRuntime>>,
        Option<FusionTensor<FusionCubeRuntime<WgpuRuntime>>>,
        LowrankProjectionShape,
    );

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        relu_lowrank_backward_impl::<WgpuFusionBackend<BT>>(ops, grads);
    }
}

#[cfg(feature = "cuda")]
impl Backward<CudaCubeBackend, 2> for FusedReluLowrankBackward<CudaCubeBackend> {
    type State = (
        CubeTensor<CudaRuntime>,
        CubeTensor<CudaRuntime>,
        Option<CubeTensor<CudaRuntime>>,
        LowrankProjectionShape,
    );

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        relu_lowrank_backward_impl::<CudaCubeBackend>(ops, grads);
    }
}

#[cfg(feature = "cuda")]
impl<BT> Backward<CudaFusionBackend<BT>, 2> for FusedReluLowrankBackward<CudaFusionBackend<BT>>
where
    BT: BoolElement + 'static,
{
    type State = (
        FusionTensor<FusionCubeRuntime<CudaRuntime>>,
        FusionTensor<FusionCubeRuntime<CudaRuntime>>,
        Option<FusionTensor<FusionCubeRuntime<CudaRuntime>>>,
        LowrankProjectionShape,
    );

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        relu_lowrank_backward_impl::<CudaFusionBackend<BT>>(ops, grads);
    }
}

fn fused_relu_lowrank_autodiff_wgpu(
    input: WgpuCubeAutodiffTensor,
    weight: WgpuCubeAutodiffTensor,
    mask: Option<WgpuCubeAutodiffTensor>,
    shape: LowrankProjectionShape,
    meta: CubeTensor<WgpuRuntime>,
) -> WgpuCubeAutodiffTensor {
    let input_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(input.clone());
    let weight_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(weight.clone());
    let mask_inner = mask
        .clone()
        .map(<WgpuCubeAutodiffBackend as AutodiffBackend>::inner);
    let output = relu_lowrank_wgsl_runtime::<WgpuRuntime>(
        input_inner.clone(),
        weight_inner.clone(),
        shape,
        meta,
        mask_inner.clone(),
    );

    match FusedReluLowrankBackward::<WgpuCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([input.node.clone(), weight.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => {
            prep.finish((input_inner, weight_inner, mask_inner, shape), output)
        }
        OpsKind::UnTracked(prep) => prep.finish(output),
    }
}

fn fused_relu_lowrank_autodiff_fusion_wgpu<BT: BoolElement + 'static>(
    input: WgpuFusionAutodiffTensor<BT>,
    weight: WgpuFusionAutodiffTensor<BT>,
    _mask: Option<WgpuFusionAutodiffTensor<BT>>,
    shape: LowrankProjectionShape,
    output: FusionTensor<FusionCubeRuntime<WgpuRuntime>>,
) -> WgpuFusionAutodiffTensor<BT> {
    let input_inner = <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(input.clone());
    let weight_inner = <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(weight.clone());
    let mask_inner = _mask
        .clone()
        .map(<WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner);

    match FusedReluLowrankBackward::<WgpuFusionBackend<BT>>(PhantomData)
        .prepare::<NoCheckpointing>([input.node.clone(), weight.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => {
            prep.finish((input_inner, weight_inner, mask_inner, shape), output)
        }
        OpsKind::UnTracked(prep) => prep.finish(output),
    }
}

#[cfg(feature = "cuda")]
fn fused_relu_lowrank_autodiff_cuda(
    input: CudaCubeAutodiffTensor,
    weight: CudaCubeAutodiffTensor,
    mask: Option<CudaCubeAutodiffTensor>,
    shape: LowrankProjectionShape,
    meta: CubeTensor<CudaRuntime>,
) -> CudaCubeAutodiffTensor {
    let input_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(input.clone());
    let weight_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(weight.clone());
    let mask_inner = mask
        .clone()
        .map(<CudaCubeAutodiffBackend as AutodiffBackend>::inner);
    let output = forward_runtime::relu_lowrank_runtime::<CudaRuntime>(
        input_inner.clone(),
        weight_inner.clone(),
        shape,
        meta,
        mask_inner.clone(),
    );

    match FusedReluLowrankBackward::<CudaCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([input.node.clone(), weight.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => {
            prep.finish((input_inner, weight_inner, mask_inner, shape), output)
        }
        OpsKind::UnTracked(prep) => prep.finish(output),
    }
}

#[cfg(feature = "cuda")]
fn fused_relu_lowrank_autodiff_fusion_cuda<BT: BoolElement + 'static>(
    input: CudaFusionAutodiffTensor<BT>,
    weight: CudaFusionAutodiffTensor<BT>,
    mask: Option<CudaFusionAutodiffTensor<BT>>,
    shape: LowrankProjectionShape,
    output: FusionTensor<FusionCubeRuntime<CudaRuntime>>,
) -> CudaFusionAutodiffTensor<BT> {
    let input_inner = <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(input.clone());
    let weight_inner = <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(weight.clone());
    let mask_inner = mask
        .clone()
        .map(<CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner);

    match FusedReluLowrankBackward::<CudaFusionBackend<BT>>(PhantomData)
        .prepare::<NoCheckpointing>([input.node.clone(), weight.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => {
            prep.finish((input_inner, weight_inner, mask_inner, shape), output)
        }
        OpsKind::UnTracked(prep) => prep.finish(output),
    }
}

/// Returns whether the fused WGPU low-rank projection kernel can run on the backend.
pub fn supports_relu_lowrank_projection_backend<B: BackendTrait>() -> bool
where
    B::FloatTensorPrimitive: 'static,
{
    matches_type::<B::FloatTensorPrimitive, CubeTensor<WgpuRuntime>>()
        || matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<WgpuRuntime>>>()
        || matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<WgpuRuntime>>>()
        || matches_type::<B::FloatTensorPrimitive, WgpuCubeAutodiffTensor>()
        || matches_type::<B::FloatTensorPrimitive, WgpuFusionAutodiffTensor<u32>>()
        || matches_type::<B::FloatTensorPrimitive, WgpuFusionAutodiffTensor<u8>>()
        || {
            #[cfg(feature = "cuda")]
            {
                matches_type::<B::FloatTensorPrimitive, CubeTensor<CudaRuntime>>()
                    || matches_type::<B::FloatTensorPrimitive, CudaCubeAutodiffTensor>()
                    || matches_type::<B::FloatTensorPrimitive, CudaFusionAutodiffTensor<u32>>()
                    || matches_type::<B::FloatTensorPrimitive, CudaFusionAutodiffTensor<u8>>()
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        }
}

/// Executes the fused WGPU low-rank projection kernel when the input/weight layout is supported.
///
/// Supported layouts:
/// - input `[batch, 1, time, embd]`, weight `[1, heads, embd, latent]`
/// - input `[batch, heads, time, embd]`, weight `[1, heads, embd, latent]`
pub fn try_fused_relu_lowrank_projection_wgpu<B: BackendTrait>(
    input: &BurnTensor<B, 4>,
    weight: &BurnTensor<B, 4>,
    threshold: f32,
    sparse_mask: Option<&BurnTensor<B, 4>>,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    try_fused_relu_lowrank_projection_wgpu_with_executor(
        input,
        weight,
        threshold,
        sparse_mask,
        LowrankGradInputExecutor::Auto,
    )
}

pub fn try_fused_relu_lowrank_projection_wgpu_with_executor<B: BackendTrait>(
    input: &BurnTensor<B, 4>,
    weight: &BurnTensor<B, 4>,
    threshold: f32,
    sparse_mask: Option<&BurnTensor<B, 4>>,
    grad_input_executor: LowrankGradInputExecutor,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !supports_relu_lowrank_projection_backend::<B>() {
        return None;
    }
    let shape = LowrankProjectionShape::from_tensors(
        input,
        weight,
        threshold,
        sparse_mask,
        grad_input_executor,
    )?;
    let meta = shape.meta(&input.device());
    relu_lowrank_forward_route_profile_record_attempt();

    if let Some(output) = try_fusion_path_autodiff_runtime::<B, u32, WgpuRuntime>(
        input,
        weight,
        threshold,
        sparse_mask,
        &meta,
        shape,
    ) {
        relu_lowrank_forward_route_profile_record_route(
            LowrankForwardRouteKind::WgpuFusionAutodiff,
        );
        return Some(output);
    }
    if let Some(output) = try_fusion_path_autodiff_runtime::<B, u8, WgpuRuntime>(
        input,
        weight,
        threshold,
        sparse_mask,
        &meta,
        shape,
    ) {
        relu_lowrank_forward_route_profile_record_route(
            LowrankForwardRouteKind::WgpuFusionAutodiff,
        );
        return Some(output);
    }
    if let Some(output) = try_direct_path_autodiff_cube_runtime::<B, WgpuRuntime>(
        input,
        weight,
        threshold,
        sparse_mask,
        &meta,
        shape,
    ) {
        relu_lowrank_forward_route_profile_record_route(
            LowrankForwardRouteKind::WgpuDirectAutodiff,
        );
        return Some(output);
    }
    if let Some(output) =
        try_fusion_path_runtime::<B, u32, WgpuRuntime>(input, weight, sparse_mask, &meta, shape)
    {
        relu_lowrank_forward_route_profile_record_route(LowrankForwardRouteKind::WgpuFusionRuntime);
        return Some(output);
    }
    if let Some(output) =
        try_fusion_path_runtime::<B, u8, WgpuRuntime>(input, weight, sparse_mask, &meta, shape)
    {
        relu_lowrank_forward_route_profile_record_route(LowrankForwardRouteKind::WgpuFusionRuntime);
        return Some(output);
    }
    if let Some(output) =
        try_direct_path_runtime::<B, WgpuRuntime>(input, weight, sparse_mask, &meta, shape)
    {
        relu_lowrank_forward_route_profile_record_route(LowrankForwardRouteKind::WgpuDirectRuntime);
        return Some(output);
    }

    #[cfg(feature = "cuda")]
    {
        if let Some(output) = try_fusion_path_autodiff_runtime::<B, u32, CudaRuntime>(
            input,
            weight,
            threshold,
            sparse_mask,
            &meta,
            shape,
        ) {
            relu_lowrank_forward_route_profile_record_route(
                LowrankForwardRouteKind::CudaFusionAutodiff,
            );
            return Some(output);
        }
        if let Some(output) = try_fusion_path_autodiff_runtime::<B, u8, CudaRuntime>(
            input,
            weight,
            threshold,
            sparse_mask,
            &meta,
            shape,
        ) {
            relu_lowrank_forward_route_profile_record_route(
                LowrankForwardRouteKind::CudaFusionAutodiff,
            );
            return Some(output);
        }
        if let Some(output) = try_direct_path_autodiff_cube_runtime::<B, CudaRuntime>(
            input,
            weight,
            threshold,
            sparse_mask,
            &meta,
            shape,
        ) {
            relu_lowrank_forward_route_profile_record_route(
                LowrankForwardRouteKind::CudaDirectAutodiff,
            );
            return Some(output);
        }
        if let Some(output) =
            try_fusion_path_runtime::<B, u32, CudaRuntime>(input, weight, sparse_mask, &meta, shape)
        {
            relu_lowrank_forward_route_profile_record_route(
                LowrankForwardRouteKind::CudaFusionRuntime,
            );
            return Some(output);
        }
        if let Some(output) =
            try_fusion_path_runtime::<B, u8, CudaRuntime>(input, weight, sparse_mask, &meta, shape)
        {
            relu_lowrank_forward_route_profile_record_route(
                LowrankForwardRouteKind::CudaFusionRuntime,
            );
            return Some(output);
        }
        if let Some(output) =
            try_direct_path_runtime::<B, CudaRuntime>(input, weight, sparse_mask, &meta, shape)
        {
            relu_lowrank_forward_route_profile_record_route(
                LowrankForwardRouteKind::CudaDirectRuntime,
            );
            return Some(output);
        }
    }

    None
}

fn try_head_aligned_grad_input_wgpu<B: BackendTrait>(
    grad_projected: &BurnTensor<B, 4>,
    weight: &BurnTensor<B, 4>,
    shape: LowrankProjectionShape,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    if shape.input_heads != shape.heads {
        return None;
    }

    match shape.grad_input_executor {
        LowrankGradInputExecutor::Auto => {
            if shape.time >= 400 {
                try_head_aligned_grad_input_kernel_wgpu::<B>(grad_projected, weight, shape).or_else(
                    || {
                        try_head_aligned_grad_input_aligned_matmul_wgpu(
                            grad_projected,
                            weight,
                            shape,
                        )
                    },
                )
            } else {
                try_head_aligned_grad_input_aligned_matmul_wgpu(grad_projected, weight, shape)
            }
        }
        LowrankGradInputExecutor::Kernel => {
            try_head_aligned_grad_input_kernel_wgpu::<B>(grad_projected, weight, shape)
        }
        LowrankGradInputExecutor::KernelTiled => {
            try_head_aligned_grad_input_tiled_wgpu::<B>(grad_projected, weight, shape)
        }
        LowrankGradInputExecutor::AlignedMatmul => {
            try_head_aligned_grad_input_aligned_matmul_wgpu(grad_projected, weight, shape)
        }
        LowrankGradInputExecutor::Direct => {
            try_head_aligned_grad_input_direct_wgpu::<B>(grad_projected, weight, shape)
        }
    }
}

fn try_head_aligned_grad_input_kernel_wgpu<B: BackendTrait>(
    grad_projected: &BurnTensor<B, 4>,
    weight: &BurnTensor<B, 4>,
    shape: LowrankProjectionShape,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    try_head_aligned_grad_input_fusion_wgpu::<B, u32>(grad_projected, weight, shape)
        .or_else(|| try_head_aligned_grad_input_fusion_wgpu::<B, u8>(grad_projected, weight, shape))
        .or_else(|| try_head_aligned_grad_input_direct_wgpu::<B>(grad_projected, weight, shape))
}

fn try_head_aligned_grad_input_tiled_wgpu<B: BackendTrait>(
    grad_projected: &BurnTensor<B, 4>,
    weight: &BurnTensor<B, 4>,
    shape: LowrankProjectionShape,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    try_head_aligned_grad_input_tiled_fusion_wgpu::<B, u32>(grad_projected, weight, shape)
        .or_else(|| {
            try_head_aligned_grad_input_tiled_fusion_wgpu::<B, u8>(grad_projected, weight, shape)
        })
        .or_else(|| {
            try_head_aligned_grad_input_tiled_direct_wgpu::<B>(grad_projected, weight, shape)
        })
}

fn try_head_aligned_grad_input_direct_wgpu<B: BackendTrait>(
    grad_projected: &BurnTensor<B, 4>,
    weight: &BurnTensor<B, 4>,
    shape: LowrankProjectionShape,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    let prim_grad = grad_projected.clone().into_primitive().tensor();
    let grad: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(prim_grad)?;
    if grad.dtype != DType::F32 {
        return None;
    }

    let prim_weight = weight.clone().into_primitive().tensor();
    let weight: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(prim_weight)?;
    if weight.dtype != DType::F32 {
        return None;
    }

    let meta: BurnTensor<B, 1> = shape.meta(&grad_projected.device());
    let prim_meta = meta.into_primitive().tensor();
    let meta: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(prim_meta)?;
    if meta.dtype != DType::F32 {
        return None;
    }

    let output = head_aligned_grad_input_wgsl_runtime::<WgpuRuntime>(grad, weight, shape, meta);
    let output_prim = try_cast_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

fn try_head_aligned_grad_input_tiled_direct_wgpu<B: BackendTrait>(
    grad_projected: &BurnTensor<B, 4>,
    weight: &BurnTensor<B, 4>,
    shape: LowrankProjectionShape,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    let prim_grad = grad_projected.clone().into_primitive().tensor();
    let grad: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(prim_grad)?;
    if grad.dtype != DType::F32 {
        return None;
    }

    let prim_weight = weight.clone().into_primitive().tensor();
    let weight: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(prim_weight)?;
    if weight.dtype != DType::F32 {
        return None;
    }

    let meta: BurnTensor<B, 1> = shape.meta(&grad_projected.device());
    let prim_meta = meta.into_primitive().tensor();
    let meta: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(prim_meta)?;
    if meta.dtype != DType::F32 {
        return None;
    }

    let output =
        head_aligned_grad_input_tiled_wgsl_runtime::<WgpuRuntime>(grad, weight, shape, meta);
    let output_prim = try_cast_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

fn try_head_aligned_grad_input_fusion_wgpu<B, BT>(
    grad_projected: &BurnTensor<B, 4>,
    weight: &BurnTensor<B, 4>,
    shape: LowrankProjectionShape,
) -> Option<BurnTensor<B, 4>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<WgpuRuntime>>>() {
        return None;
    }

    let prim_grad = grad_projected.clone().into_primitive().tensor();
    let fusion_grad: FusionTensor<FusionCubeRuntime<WgpuRuntime>> =
        try_cast_primitive::<B, _>(prim_grad)?;
    let fusion_client = fusion_grad.client.clone();
    let grad =
        fusion_client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion_grad);
    if grad.dtype != DType::F32 {
        return None;
    }

    let weight = resolve_fusion_tensor_wgpu::<B, BT, 4>(weight)?;
    let meta = resolve_fusion_tensor_wgpu::<B, BT, 1>(&shape.meta(&grad_projected.device()))?;
    let output = head_aligned_grad_input_wgsl_runtime::<WgpuRuntime>(grad, weight, shape, meta);
    let output_fusion = register_fusion_float_tensor(&fusion_client, output);
    let output_prim = try_cast_backend::<B, _>(output_fusion)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

fn try_head_aligned_grad_input_tiled_fusion_wgpu<B, BT>(
    grad_projected: &BurnTensor<B, 4>,
    weight: &BurnTensor<B, 4>,
    shape: LowrankProjectionShape,
) -> Option<BurnTensor<B, 4>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<WgpuRuntime>>>() {
        return None;
    }

    let prim_grad = grad_projected.clone().into_primitive().tensor();
    let fusion_grad: FusionTensor<FusionCubeRuntime<WgpuRuntime>> =
        try_cast_primitive::<B, _>(prim_grad)?;
    let fusion_client = fusion_grad.client.clone();
    let grad =
        fusion_client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion_grad);
    if grad.dtype != DType::F32 {
        return None;
    }

    let weight = resolve_fusion_tensor_wgpu::<B, BT, 4>(weight)?;
    let meta = resolve_fusion_tensor_wgpu::<B, BT, 1>(&shape.meta(&grad_projected.device()))?;
    let output =
        head_aligned_grad_input_tiled_wgsl_runtime::<WgpuRuntime>(grad, weight, shape, meta);
    let output_fusion = register_fusion_float_tensor(&fusion_client, output);
    let output_prim = try_cast_backend::<B, _>(output_fusion)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

fn try_head_aligned_grad_input_aligned_matmul_wgpu<B: BackendTrait>(
    grad_projected: &BurnTensor<B, 4>,
    weight: &BurnTensor<B, 4>,
    shape: LowrankProjectionShape,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    let prim_grad = grad_projected.clone().into_primitive().tensor();
    let grad: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(prim_grad)?;
    if grad.dtype != DType::F32 {
        return None;
    }

    let prim_weight = weight.clone().into_primitive().tensor();
    let weight: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(prim_weight)?;
    if weight.dtype != DType::F32 {
        return None;
    }

    let output = head_aligned_grad_input_aligned_matmul_cube(grad, weight, shape)?;
    let output_prim = try_cast_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

fn try_lowrank_grad_input_cuda_direct<B: BackendTrait>(
    grad_projected: &BurnTensor<B, 4>,
    weight: &BurnTensor<B, 4>,
    shape: LowrankProjectionShape,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    #[cfg(not(feature = "cuda"))]
    {
        let _ = grad_projected;
        let _ = weight;
        let _ = shape;
        None
    }

    #[cfg(feature = "cuda")]
    {
        let prim_grad = grad_projected.clone().into_primitive().tensor();
        let grad: CubeTensor<CudaRuntime> = try_cast_primitive::<B, _>(prim_grad)?;
        if grad.dtype != DType::F32 {
            return None;
        }

        let prim_weight = weight.clone().into_primitive().tensor();
        let weight: CubeTensor<CudaRuntime> = try_cast_primitive::<B, _>(prim_weight)?;
        if weight.dtype != DType::F32 {
            return None;
        }

        let meta: BurnTensor<B, 1> = shape.meta(&grad_projected.device());
        let prim_meta = meta.into_primitive().tensor();
        let meta: CubeTensor<CudaRuntime> = try_cast_primitive::<B, _>(prim_meta)?;
        if meta.dtype != DType::F32 {
            return None;
        }

        let output = lowrank_grad_input_cube_runtime::<CudaRuntime>(grad, weight, shape, meta);
        let output_prim = try_cast_backend::<B, _>(output)?;
        Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            output_prim,
        )))
    }
}

fn head_aligned_grad_input_wgsl_runtime<R: CubeRuntime>(
    grad: CubeTensor<R>,
    weight: CubeTensor<R>,
    shape: LowrankProjectionShape,
    meta: CubeTensor<R>,
) -> CubeTensor<R> {
    let client = grad.client.clone();
    let device = &grad.device;
    let output = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([shape.batch, shape.heads, shape.time, shape.embd]),
    );

    let workgroups_x = div_ceil_u32(shape.embd as u32, WORKGROUP_SIZE_X);
    let count = CubeCount::Static(
        workgroups_x,
        shape.time as u32,
        (shape.batch * shape.heads) as u32,
    );
    let kernel = SourceKernel::new(
        ReluLowrankGradInputKernel,
        CubeDim::new_3d(WORKGROUP_SIZE_X, 1, 1),
    );
    let bindings = KernelArguments::new().with_buffers(vec![
        grad.handle.clone().binding(),
        weight.handle.clone().binding(),
        output.handle.clone().binding(),
        meta.handle.clone().binding(),
    ]);
    client.launch(Box::new(kernel), count, bindings);
    output
}

fn head_aligned_grad_input_tiled_wgsl_runtime<R: CubeRuntime>(
    grad: CubeTensor<R>,
    weight: CubeTensor<R>,
    shape: LowrankProjectionShape,
    meta: CubeTensor<R>,
) -> CubeTensor<R> {
    let client = grad.client.clone();
    let device = &grad.device;
    let output = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([shape.batch, shape.heads, shape.time, shape.embd]),
    );

    let workgroups_x = div_ceil_u32(shape.embd as u32, WORKGROUP_SIZE_X);
    let count = CubeCount::Static(
        workgroups_x,
        shape.time as u32,
        (shape.batch * shape.heads) as u32,
    );
    let kernel = SourceKernel::new(
        ReluLowrankGradInputTiledKernel,
        CubeDim::new_3d(WORKGROUP_SIZE_X, 1, 1),
    );
    let bindings = KernelArguments::new().with_buffers(vec![
        grad.handle.clone().binding(),
        weight.handle.clone().binding(),
        output.handle.clone().binding(),
        meta.handle.clone().binding(),
    ]);
    client.launch(Box::new(kernel), count, bindings);
    output
}

#[cfg(feature = "cuda")]
fn lowrank_grad_input_cube_runtime<R: CubeRuntime>(
    grad: CubeTensor<R>,
    weight: CubeTensor<R>,
    shape: LowrankProjectionShape,
    meta: CubeTensor<R>,
) -> CubeTensor<R> {
    let prof_enabled = profile_enabled();
    let total_start = prof_enabled.then(Instant::now);
    let grad = into_contiguous(grad);
    let weight = into_contiguous(weight);
    let meta = into_contiguous(meta);

    let client = grad.client.clone();
    let device = grad.device.clone();
    let output = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([shape.batch, shape.input_heads, shape.time, shape.embd]),
    );

    let cube_dim = CubeDim::new_1d(WORKGROUP_SIZE_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32(shape.embd as u32, WORKGROUP_SIZE_X),
        shape.time as u32,
        (shape.batch * shape.input_heads) as u32,
    );

    let _ = lowrank_grad_input_cube_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        grad.clone().into_tensor_arg(),
        weight.clone().into_tensor_arg(),
        output.clone().into_tensor_arg(),
        meta.clone().into_tensor_arg(),
    );

    if let Some(start) = total_start {
        profile_record(&RELU_LOWRANK_GRAD_INPUT_PROFILE, |state| {
            state.calls = state.calls.saturating_add(1);
            state.launches = state.launches.saturating_add(1);
            state.total_ns = state.total_ns.saturating_add(start.elapsed().as_nanos());
        });
    }

    output
}

fn head_aligned_grad_input_aligned_matmul_cube(
    grad: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    shape: LowrankProjectionShape,
) -> Option<CubeTensor<WgpuRuntime>> {
    if shape.input_heads != shape.heads {
        return None;
    }

    let grad_by_head = cube_swap_dims(grad, 0, 1);
    let grad_by_head = into_contiguous_aligned(grad_by_head);
    let grad_by_head = cube_reshape_contiguous(
        grad_by_head,
        [shape.heads, shape.batch * shape.time, shape.latent],
    );

    let weight_by_head = cube_reshape_contiguous(weight, [shape.heads, shape.embd, shape.latent]);
    let weight_by_head = cube_swap_dims(weight_by_head, 1, 2);
    let weight_by_head = into_contiguous_aligned(weight_by_head);

    let output = matmul(
        grad_by_head,
        weight_by_head,
        None,
        MatmulStrategy::default(),
        DType::F32,
    )
    .ok()?;
    let output =
        cube_reshape_contiguous(output, [shape.heads, shape.batch, shape.time, shape.embd]);
    Some(cube_swap_dims(output, 0, 1))
}

fn cube_swap_dims<R: CubeRuntime>(
    mut tensor: CubeTensor<R>,
    dim1: usize,
    dim2: usize,
) -> CubeTensor<R> {
    let mut shape = tensor.meta.shape().clone();
    let mut strides = tensor.meta.strides.clone();
    shape.swap(dim1, dim2);
    strides.swap(dim1, dim2);
    *tensor.meta = Metadata::new(shape, strides);
    tensor
}

fn cube_reshape_contiguous<R: CubeRuntime, const D: usize>(
    tensor: CubeTensor<R>,
    dims: [usize; D],
) -> CubeTensor<R> {
    CubeTensor::new_contiguous(
        tensor.client.clone(),
        tensor.device.clone(),
        Shape::new(dims),
        tensor.handle.clone(),
        tensor.dtype,
    )
}

fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

fn resolve_fusion_tensor_wgpu<B, BT, const D: usize>(
    tensor: &BurnTensor<B, D>,
) -> Option<CubeTensor<WgpuRuntime>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
{
    let prim = tensor.clone().into_primitive().tensor();
    let fusion: FusionTensor<FusionCubeRuntime<WgpuRuntime>> = try_cast_primitive::<B, _>(prim)?;
    let client = fusion.client.clone();
    let cube = client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion);
    if cube.dtype != DType::F32 {
        return None;
    }
    Some(cube)
}

fn resolve_fusion_tensor_runtime<B, BT, R, const D: usize>(
    tensor: &BurnTensor<B, D>,
) -> Option<CubeTensor<R>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
    R: CubeRuntime + 'static,
{
    let prim = tensor.clone().into_primitive().tensor();
    let fusion: FusionTensor<FusionCubeRuntime<R>> = try_cast_primitive::<B, _>(prim)?;
    let client = fusion.client.clone();
    let cube = client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion);
    if cube.dtype != DType::F32 {
        return None;
    }
    Some(cube)
}

fn matches_autodiff_fusion_type<B, BT, R>() -> bool
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
    R: CubeRuntime + 'static,
{
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        matches_type::<B::FloatTensorPrimitive, WgpuFusionAutodiffTensor<BT>>()
    } else {
        #[cfg(feature = "cuda")]
        {
            if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
                return matches_type::<B::FloatTensorPrimitive, CudaFusionAutodiffTensor<BT>>();
            }
        }
        false
    }
}

fn extract_autodiff_inner<B, R>(value: B::FloatTensorPrimitive) -> Option<CubeTensor<R>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    R: CubeRuntime + 'static,
{
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        let ad: WgpuCubeAutodiffTensor = try_cast_primitive::<B, _>(value)?;
        let inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(ad);
        let boxed: Box<dyn Any> = Box::new(inner);
        return boxed.downcast::<CubeTensor<R>>().ok().map(|boxed| *boxed);
    }
    #[cfg(feature = "cuda")]
    {
        if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
            let ad: CudaCubeAutodiffTensor = try_cast_primitive::<B, _>(value)?;
            let inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(ad);
            let boxed: Box<dyn Any> = Box::new(inner);
            return boxed.downcast::<CubeTensor<R>>().ok().map(|boxed| *boxed);
        }
    }
    None
}

fn wrap_autodiff_inner<B, R>(value: CubeTensor<R>) -> Option<B::FloatTensorPrimitive>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    R: CubeRuntime + 'static,
{
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        let boxed: Box<dyn Any> = Box::new(value);
        let inner = boxed
            .downcast::<CubeTensor<WgpuRuntime>>()
            .ok()
            .map(|boxed| *boxed)?;
        let ad = <WgpuCubeAutodiffBackend as AutodiffBackend>::from_inner(inner);
        return try_cast_backend::<B, _>(ad);
    }
    #[cfg(feature = "cuda")]
    {
        if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
            let boxed: Box<dyn Any> = Box::new(value);
            let inner = boxed
                .downcast::<CubeTensor<CudaRuntime>>()
                .ok()
                .map(|boxed| *boxed)?;
            let ad = <CudaCubeAutodiffBackend as AutodiffBackend>::from_inner(inner);
            return try_cast_backend::<B, _>(ad);
        }
    }
    None
}

fn extract_fusion_autodiff_inner<B, BT, R>(
    value: B::FloatTensorPrimitive,
) -> Option<FusionTensor<FusionCubeRuntime<R>>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
    R: CubeRuntime + 'static,
{
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        let ad: WgpuFusionAutodiffTensor<BT> = try_cast_primitive::<B, _>(value)?;
        let inner = <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(ad);
        let boxed: Box<dyn Any> = Box::new(inner);
        return boxed
            .downcast::<FusionTensor<FusionCubeRuntime<R>>>()
            .ok()
            .map(|boxed| *boxed);
    }
    #[cfg(feature = "cuda")]
    {
        if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
            let ad: CudaFusionAutodiffTensor<BT> = try_cast_primitive::<B, _>(value)?;
            let inner = <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(ad);
            let boxed: Box<dyn Any> = Box::new(inner);
            return boxed
                .downcast::<FusionTensor<FusionCubeRuntime<R>>>()
                .ok()
                .map(|boxed| *boxed);
        }
    }
    None
}

fn wrap_fusion_autodiff_inner<B, BT, R>(
    value: FusionTensor<FusionCubeRuntime<R>>,
) -> Option<B::FloatTensorPrimitive>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
    R: CubeRuntime + 'static,
{
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        let boxed: Box<dyn Any> = Box::new(value);
        let inner = boxed
            .downcast::<FusionTensor<FusionCubeRuntime<WgpuRuntime>>>()
            .ok()
            .map(|boxed| *boxed)?;
        let ad = <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::from_inner(inner);
        return try_cast_backend::<B, _>(ad);
    }
    #[cfg(feature = "cuda")]
    {
        if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
            let boxed: Box<dyn Any> = Box::new(value);
            let inner = boxed
                .downcast::<FusionTensor<FusionCubeRuntime<CudaRuntime>>>()
                .ok()
                .map(|boxed| *boxed)?;
            let ad = <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::from_inner(inner);
            return try_cast_backend::<B, _>(ad);
        }
    }
    None
}

#[cube(launch)]
fn relu_lowrank_cube_kernel(
    input: &Tensor<f32>,
    weight: &Tensor<f32>,
    output: &mut Tensor<f32>,
    params: &Tensor<f32>,
    sparse_mask: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let input_heads = u32::cast_from(params[1]) as usize;
    let heads = u32::cast_from(params[2]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let embd = u32::cast_from(params[4]) as usize;
    let latent = u32::cast_from(params[5]) as usize;
    let threshold = params[6];
    let has_mask = params[7] > f32::cast_from(0u32);

    let l = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    let t = CUBE_POS_Y as usize;
    let bh = CUBE_POS_Z as usize;
    if l >= latent || t >= time || bh >= batch * heads {
        terminate!();
    }

    let h = bh % heads;
    let b = bh / heads;
    let mut input_head = h;
    if input_heads == 1usize {
        input_head = 0usize;
    }

    let mut sum = f32::cast_from(0u32);
    let mut e = 0usize;
    while e < embd {
        let input_index = ((b * input_heads + input_head) * time + t) * embd + e;
        let weight_index = (h * embd + e) * latent + l;
        sum += input[input_index] * weight[weight_index];
        e += 1usize;
    }

    sum -= threshold;
    if sum < f32::cast_from(0u32) {
        sum = f32::cast_from(0u32);
    }
    if has_mask {
        sum *= sparse_mask[l];
    }

    let output_index = ((b * heads + h) * time + t) * latent + l;
    output[output_index] = sum;
}

#[cube(launch)]
fn lowrank_grad_input_cube_kernel(
    grad: &Tensor<f32>,
    weight: &Tensor<f32>,
    output: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let input_heads = u32::cast_from(params[1]) as usize;
    let heads = u32::cast_from(params[2]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let embd = u32::cast_from(params[4]) as usize;
    let latent = u32::cast_from(params[5]) as usize;

    let e = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    let t = CUBE_POS_Y as usize;
    let bih = CUBE_POS_Z as usize;
    if e >= embd || t >= time || bih >= batch * input_heads {
        terminate!();
    }

    let input_head = bih % input_heads;
    let b = bih / input_heads;

    let mut acc = f32::cast_from(0u32);
    if input_heads == 1usize {
        let mut h = 0usize;
        while h < heads {
            let mut l = 0usize;
            while l < latent {
                let grad_index = ((b * heads + h) * time + t) * latent + l;
                let weight_index = (h * embd + e) * latent + l;
                acc += grad[grad_index] * weight[weight_index];
                l += 1usize;
            }
            h += 1usize;
        }
    } else {
        let h = input_head;
        let mut l = 0usize;
        while l < latent {
            let grad_index = ((b * heads + h) * time + t) * latent + l;
            let weight_index = (h * embd + e) * latent + l;
            acc += grad[grad_index] * weight[weight_index];
            l += 1usize;
        }
    }

    let output_index = ((b * input_heads + input_head) * time + t) * embd + e;
    output[output_index] = acc;
}

#[derive(Clone)]
struct ReluLowrankKernel;

#[derive(Clone)]
struct ReluLowrankGradInputKernel;

#[derive(Clone)]
struct ReluLowrankGradInputTiledKernel;

impl KernelSource for ReluLowrankKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(RELU_LOWRANK_SHADER)
    }

    fn id(&self) -> burn_cubecl::cubecl::prelude::KernelId {
        KernelId::new::<Self>()
    }
}

impl KernelSource for ReluLowrankGradInputKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(RELU_LOWRANK_GRAD_INPUT_SHADER)
    }

    fn id(&self) -> burn_cubecl::cubecl::prelude::KernelId {
        KernelId::new::<Self>()
    }
}

impl KernelSource for ReluLowrankGradInputTiledKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(RELU_LOWRANK_GRAD_INPUT_TILED_SHADER)
    }

    fn id(&self) -> burn_cubecl::cubecl::prelude::KernelId {
        KernelId::new::<Self>()
    }
}

fn matches_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

fn try_cast_primitive<B: BackendTrait, T: 'static>(value: B::FloatTensorPrimitive) -> Option<T>
where
    B::FloatTensorPrimitive: 'static,
{
    let boxed: Box<dyn Any> = Box::new(value);
    boxed.downcast::<T>().ok().map(|boxed| *boxed)
}

fn try_cast_backend<B: BackendTrait, T: 'static>(value: T) -> Option<B::FloatTensorPrimitive>
where
    B::FloatTensorPrimitive: 'static,
{
    let boxed: Box<dyn Any> = Box::new(value);
    boxed
        .downcast::<B::FloatTensorPrimitive>()
        .ok()
        .map(|boxed| *boxed)
}

#[cfg(test)]
mod tests;
