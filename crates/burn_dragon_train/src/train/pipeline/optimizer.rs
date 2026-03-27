use crate::train::prelude::*;
use crate::{MuonAdjustLrFn, OptimizerKind, OptimizerScheduleMode};
use burn::module::{AutodiffModule, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::optim::AdaptiveMomentumState;
use burn::optim::MultiGradientsParams;
use burn::optim::SimpleOptimizer;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::grad_clipping::GradientClipping;
use burn::optim::momentum::{Momentum, MomentumConfig, MomentumState};
use burn::optim::record::AdaptorRecord;
use hashbrown::HashMap;
use std::collections::HashSet;
use std::marker::PhantomData;

const DEFAULT_BITNET_SECOND_STAGE_START: f32 = 0.5;
const DEFAULT_BITNET_BETA_1: f32 = 0.9;
const DEFAULT_BITNET_BETA_2: f32 = 0.95;
const DEFAULT_BITNET_EPSILON: f32 = 1.0e-8;

const MUON_NS_A: f32 = 3.4445;
const MUON_NS_B: f32 = -4.775;
const MUON_NS_C: f32 = 2.0315;
const MUON_EPSILON: f32 = 1.0e-7;

pub enum ResolvedOptimizer<B, M>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    AdamW(OptimizerAdaptor<AdamW, M, B>),
    BitNetAdamW(OptimizerAdaptor<BitNetAdamW, M, B>),
    MuonHybrid(MuonHybridOptimizer<M, B>),
}

#[derive(Clone)]
pub struct BitNetAdamW {
    momentum: BitNetAdaptiveMomentum,
    weight_decay_initial: f32,
    weight_decay_final: f32,
    second_stage_start: f32,
    total_steps: usize,
}

#[derive(Record, Clone)]
pub struct BitNetAdamWState<B: BackendTrait, const D: usize> {
    pub momentum: AdaptiveMomentumState<B, D>,
}

impl<B: BackendTrait> SimpleOptimizer<B> for BitNetAdamW {
    type State<const D: usize> = BitNetAdamWState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let (raw_delta, momentum_state) = self
            .momentum
            .transform(grad, state.map(|state| state.momentum));
        let weight_decay = weight_decay_for_step(
            self.weight_decay_initial,
            self.weight_decay_final,
            self.second_stage_start,
            self.total_steps,
            momentum_state.time,
        );
        let decay_rate = lr * weight_decay as f64;
        let decayed_tensor = if decay_rate == 0.0 {
            tensor.clone()
        } else {
            tensor.clone().mul_scalar(1.0 - decay_rate)
        };
        let tensor_updated = decayed_tensor - raw_delta.mul_scalar(lr);
        let state = BitNetAdamWState {
            momentum: momentum_state,
        };
        (tensor_updated, Some(state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &B::Device) -> Self::State<D> {
        state.momentum = state.momentum.to_device(device);
        state
    }
}

#[derive(Clone)]
struct BitNetAdaptiveMomentum {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
}

impl BitNetAdaptiveMomentum {
    fn transform<B: BackendTrait, const D: usize>(
        &self,
        grad: Tensor<B, D>,
        state: Option<AdaptiveMomentumState<B, D>>,
    ) -> (Tensor<B, D>, AdaptiveMomentumState<B, D>) {
        let factor_1 = 1.0 - self.beta_1;
        let factor_2 = 1.0 - self.beta_2;

        let state = if let Some(mut state) = state {
            state.moment_1 = state
                .moment_1
                .mul_scalar(self.beta_1)
                .add(grad.clone().mul_scalar(factor_1));
            state.moment_2 = state
                .moment_2
                .mul_scalar(self.beta_2)
                .add(grad.square().mul_scalar(factor_2));
            state.time += 1;
            state
        } else {
            AdaptiveMomentumState {
                time: 1,
                moment_1: grad.clone().mul_scalar(factor_1),
                moment_2: grad.square().mul_scalar(factor_2),
                max_moment_2: None,
            }
        };

        let time = state.time as i32;
        let moment_1_corrected = state
            .moment_1
            .clone()
            .div_scalar(1f32 - self.beta_1.powi(time));
        let moment_2_corrected = state
            .moment_2
            .clone()
            .div_scalar(1f32 - self.beta_2.powi(time));
        let update_delta =
            moment_1_corrected.div(moment_2_corrected.sqrt().add_scalar(self.epsilon));

        (update_delta, state)
    }
}

#[derive(Clone)]
pub struct HeadwiseMuon<B: BackendTrait> {
    momentum: Momentum<B>,
    weight_decay_penalty: Option<f32>,
    epsilon: f32,
    ns_steps: usize,
    adjust_lr_fn: MuonAdjustLrFn,
    split_decoder_heads: bool,
}

#[derive(Record, Clone)]
pub struct HeadwiseMuonState<B: BackendTrait, const D: usize> {
    pub momentum: MomentumState<B, D>,
}

impl<B: BackendTrait> HeadwiseMuon<B> {
    fn adjust_lr(&self, lr: LearningRate, shape: [usize; 2]) -> LearningRate {
        let a = shape[0] as f64;
        let b = shape[1] as f64;
        match self.adjust_lr_fn {
            MuonAdjustLrFn::MatchRmsAdamw => lr * (0.2 * a.max(b).sqrt()),
        }
    }

    fn zeropower_via_newtonschulz<const D: usize>(&self, g: Tensor<B, D>) -> Tensor<B, D> {
        let shape = g.shape();
        let dim_m2 = shape[D - 2];
        let dim_m1 = shape[D - 1];
        let (mut x, needs_transpose) = if dim_m2 > dim_m1 {
            (g.swap_dims(D - 2, D - 1), true)
        } else {
            (g, false)
        };

        let norm = x
            .clone()
            .powf_scalar(2.0)
            .sum()
            .sqrt()
            .clamp_min(self.epsilon)
            .unsqueeze();
        x = x.div(norm);

        for _ in 0..self.ns_steps {
            let x_t = x.clone().swap_dims(D - 2, D - 1);
            let a_matrix = x.clone().matmul(x_t);
            let a_squared = a_matrix.clone().matmul(a_matrix.clone());
            let b_matrix = a_matrix
                .mul_scalar(MUON_NS_B)
                .add(a_squared.mul_scalar(MUON_NS_C));
            x = x
                .clone()
                .mul_scalar(MUON_NS_A)
                .add(b_matrix.matmul(x.clone()));
        }

        if needs_transpose {
            x.swap_dims(D - 2, D - 1)
        } else {
            x
        }
    }

    fn orthogonalize_rank3(&self, grad: Tensor<B, 3>) -> (Tensor<B, 3>, [usize; 2]) {
        let [heads, rows, cols] = grad.shape().dims::<3>();
        if self.split_decoder_heads {
            let mut updates = Vec::with_capacity(heads);
            for head_idx in 0..heads {
                let head = grad
                    .clone()
                    .slice([head_idx..(head_idx + 1), 0..rows, 0..cols])
                    .reshape([rows, cols]);
                let update = self
                    .zeropower_via_newtonschulz(head)
                    .reshape([1, rows, cols]);
                updates.push(update);
            }
            (Tensor::cat(updates, 0), [rows, cols])
        } else {
            let flattened = grad.clone().reshape([heads * rows, cols]);
            let update = self
                .zeropower_via_newtonschulz(flattened)
                .reshape([heads, rows, cols]);
            (update, [heads * rows, cols])
        }
    }
}

impl<B: BackendTrait> SimpleOptimizer<B> for HeadwiseMuon<B> {
    type State<const D: usize> = HeadwiseMuonState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        assert!(
            D == 2 || D == 3,
            "HeadwiseMuon expects 2D or 3D tensors, got {D}D"
        );
        let (grad, momentum_state) = self
            .momentum
            .transform(grad, state.map(|state| state.momentum));

        let (update, shape_for_lr) = if D == 2 {
            let shape = tensor.shape().dims::<2>();
            (self.zeropower_via_newtonschulz(grad), [shape[0], shape[1]])
        } else {
            let (update, shape) =
                self.orthogonalize_rank3(grad.reshape(tensor.shape().dims::<3>()));
            (update.reshape(tensor.shape()), shape)
        };

        let adjusted_lr = self.adjust_lr(lr, shape_for_lr);
        let tensor = if let Some(penalty) = self.weight_decay_penalty {
            tensor.mul_scalar(1.0 - lr * penalty as f64)
        } else {
            tensor
        };
        let new_state = HeadwiseMuonState {
            momentum: momentum_state,
        };
        (tensor - update.mul_scalar(adjusted_lr), Some(new_state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &B::Device) -> Self::State<D> {
        state.momentum = state.momentum.to_device(device);
        state
    }
}

#[derive(Record, Clone)]
pub struct MuonHybridOptimizerRecord<B: AutodiffBackend> {
    muon_records: HashMap<ParamId, AdaptorRecord<HeadwiseMuon<B::InnerBackend>, B>>,
    fallback_records: HashMap<ParamId, AdaptorRecord<BitNetAdamW, B>>,
}

#[derive(Clone)]
pub struct MuonHybridOptimizer<M, B>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    muon: HeadwiseMuon<B::InnerBackend>,
    fallback: BitNetAdamW,
    muon_records: HashMap<ParamId, AdaptorRecord<HeadwiseMuon<B::InnerBackend>, B>>,
    fallback_records: HashMap<ParamId, AdaptorRecord<BitNetAdamW, B>>,
    module: PhantomData<M>,
    grad_clipping: Option<GradientClipping>,
    target_modules: Option<HashSet<String>>,
}

impl<M, B> Optimizer<M, B> for MuonHybridOptimizer<M, B>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    type Record = MuonHybridOptimizerRecord<B>;

    fn step(&mut self, lr: LearningRate, module: M, grads: GradientsParams) -> M {
        let mut grads = HybridGradAdaptor::Single(grads);
        let targets = collect_muon_target_ids(&module, self.target_modules.as_ref());
        let mut mapper = MuonHybridMapper::<M, B>::new(
            &self.muon,
            &self.fallback,
            &mut self.muon_records,
            &mut self.fallback_records,
            &mut grads,
            lr,
            &targets,
            self.grad_clipping.as_ref(),
        );
        module.map(&mut mapper)
    }

    fn step_multi(&mut self, lr: LearningRate, module: M, grads: MultiGradientsParams) -> M {
        let mut grads = HybridGradAdaptor::Multi(grads);
        let targets = collect_muon_target_ids(&module, self.target_modules.as_ref());
        let mut mapper = MuonHybridMapper::<M, B>::new(
            &self.muon,
            &self.fallback,
            &mut self.muon_records,
            &mut self.fallback_records,
            &mut grads,
            lr,
            &targets,
            self.grad_clipping.as_ref(),
        );
        module.map(&mut mapper)
    }

    fn to_record(&self) -> Self::Record {
        MuonHybridOptimizerRecord {
            muon_records: self.muon_records.clone(),
            fallback_records: self.fallback_records.clone(),
        }
    }

    fn load_record(mut self, record: Self::Record) -> Self {
        self.muon_records = record.muon_records;
        self.fallback_records = record.fallback_records;
        self
    }
}

enum HybridGradAdaptor {
    Single(GradientsParams),
    Multi(MultiGradientsParams),
}

impl HybridGradAdaptor {
    fn remove<B: BackendTrait, const D: usize>(
        &mut self,
        id: ParamId,
    ) -> Option<(Tensor<B, D>, B::Device)> {
        match self {
            HybridGradAdaptor::Single(grads) => grads.remove(id).map(|tensor| {
                let device = tensor.device();
                (tensor, device)
            }),
            HybridGradAdaptor::Multi(grads) => grads.remove(id),
        }
    }
}

#[derive(Default)]
struct MuonTargetCollector {
    ids: HashSet<ParamId>,
}

impl MuonTargetCollector {
    fn should_route_to_muon<const D: usize>(
        path: &[String],
        target_modules: Option<&HashSet<String>>,
    ) -> bool {
        let Some(last) = path.last().map(String::as_str) else {
            return false;
        };
        if !(D == 2 || D == 3) {
            return false;
        }
        if let Some(target_modules) = target_modules {
            return target_modules.contains(last);
        }
        matches!(
            last,
            "encoder" | "encoder_v" | "decoder" | "decoder_x" | "decoder_y"
        )
    }
}

impl<B: BackendTrait> ModuleVisitor<B> for MuonTargetCollector {
    fn visit_float_with_path<const D: usize>(
        &mut self,
        path: &[String],
        id: ParamId,
        _tensor: &Tensor<B, D>,
    ) {
        if Self::should_route_to_muon::<D>(path, None) {
            self.ids.insert(id);
        }
    }
}

struct ConfigurableMuonTargetCollector<'a> {
    ids: HashSet<ParamId>,
    target_modules: Option<&'a HashSet<String>>,
}

impl<B: BackendTrait> ModuleVisitor<B> for ConfigurableMuonTargetCollector<'_> {
    fn visit_float_with_path<const D: usize>(
        &mut self,
        path: &[String],
        id: ParamId,
        _tensor: &Tensor<B, D>,
    ) {
        if MuonTargetCollector::should_route_to_muon::<D>(path, self.target_modules) {
            self.ids.insert(id);
        }
    }
}

fn collect_muon_target_ids<M, B>(
    module: &M,
    target_modules: Option<&HashSet<String>>,
) -> HashSet<ParamId>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    let mut collector = ConfigurableMuonTargetCollector {
        ids: HashSet::new(),
        target_modules,
    };
    module.visit(&mut collector);
    collector.ids
}

struct MuonHybridMapper<'a, M, B>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    muon: &'a HeadwiseMuon<B::InnerBackend>,
    fallback: &'a BitNetAdamW,
    muon_records: &'a mut HashMap<ParamId, AdaptorRecord<HeadwiseMuon<B::InnerBackend>, B>>,
    fallback_records: &'a mut HashMap<ParamId, AdaptorRecord<BitNetAdamW, B>>,
    grads: &'a mut HybridGradAdaptor,
    lr: LearningRate,
    muon_targets: &'a HashSet<ParamId>,
    grad_clipping: Option<&'a GradientClipping>,
    phantom: PhantomData<M>,
}

impl<M, B> MuonHybridMapper<'_, M, B>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    fn new<'a>(
        muon: &'a HeadwiseMuon<B::InnerBackend>,
        fallback: &'a BitNetAdamW,
        muon_records: &'a mut HashMap<ParamId, AdaptorRecord<HeadwiseMuon<B::InnerBackend>, B>>,
        fallback_records: &'a mut HashMap<ParamId, AdaptorRecord<BitNetAdamW, B>>,
        grads: &'a mut HybridGradAdaptor,
        lr: LearningRate,
        muon_targets: &'a HashSet<ParamId>,
        grad_clipping: Option<&'a GradientClipping>,
    ) -> MuonHybridMapper<'a, M, B> {
        MuonHybridMapper {
            muon,
            fallback,
            muon_records,
            fallback_records,
            grads,
            lr,
            muon_targets,
            grad_clipping,
            phantom: PhantomData,
        }
    }

    fn step_with_simple_optimizer<O, const D: usize>(
        optimizer: &O,
        records: &mut HashMap<ParamId, AdaptorRecord<O, B>>,
        grad_clipping: Option<&GradientClipping>,
        grads: &mut HybridGradAdaptor,
        lr: LearningRate,
        id: ParamId,
        tensor: Tensor<B, D>,
    ) -> Tensor<B, D>
    where
        O: SimpleOptimizer<B::InnerBackend>,
    {
        let Some((grad, device)) = grads.remove::<B::InnerBackend, D>(id) else {
            return tensor;
        };
        let is_require_grad = tensor.is_require_grad();
        let (key, record) = records.remove_entry(&id).unzip();
        let tensor = if tensor.device() != device {
            tensor.to_device(&device)
        } else {
            tensor
        };
        let grad = if let Some(grad_clipping) = grad_clipping {
            grad_clipping.clip_gradient(grad)
        } else {
            grad
        };
        let (tensor, state) = optimizer.step(
            lr,
            tensor.inner(),
            grad,
            record.map(|record| O::to_device(record.into_state(), &device)),
        );
        if let Some(state) = state {
            records.insert(key.unwrap_or(id), AdaptorRecord::from_state(state));
        }
        let mut tensor = Tensor::from_inner(tensor);
        if is_require_grad {
            tensor = tensor.require_grad();
        }
        tensor
    }
}

impl<M, B> ModuleMapper<B> for MuonHybridMapper<'_, M, B>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let tensor = if self.muon_targets.contains(&id) {
            Self::step_with_simple_optimizer(
                self.muon,
                self.muon_records,
                self.grad_clipping,
                self.grads,
                self.lr,
                id,
                tensor,
            )
        } else {
            Self::step_with_simple_optimizer(
                self.fallback,
                self.fallback_records,
                self.grad_clipping,
                self.grads,
                self.lr,
                id,
                tensor,
            )
        };
        Param::from_mapped_value(id, tensor, mapper)
    }
}

pub fn resolve_optimizer<B, M>(
    optimizer_cfg: &OptimizerConfig,
    total_steps: usize,
) -> Result<ResolvedOptimizer<B, M>>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    if matches!(optimizer_cfg.name, OptimizerKind::MuonHybridExp) {
        return Ok(ResolvedOptimizer::MuonHybrid(
            build_muon_hybrid_optimizer::<B, M>(optimizer_cfg, total_steps)?,
        ));
    }

    if let Some(profile) = resolve_bitnet_optimizer_profile(optimizer_cfg, total_steps) {
        let optimizer = build_bitnet_adamw_optimizer::<B, M>(optimizer_cfg, profile);
        return Ok(ResolvedOptimizer::BitNetAdamW(optimizer));
    }

    Ok(ResolvedOptimizer::AdamW(
        adamw_config_from_optimizer(optimizer_cfg).init::<B, M>(),
    ))
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct BitNetOptimizerProfile {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
    weight_decay_initial: f32,
    weight_decay_final: f32,
    second_stage_start: f32,
    total_steps: usize,
}

fn resolve_bitnet_optimizer_profile(
    optimizer_cfg: &OptimizerConfig,
    total_steps: usize,
) -> Option<BitNetOptimizerProfile> {
    let total_steps = total_steps.max(1);
    match optimizer_cfg.name {
        OptimizerKind::BitnetPublished => Some(BitNetOptimizerProfile {
            beta_1: DEFAULT_BITNET_BETA_1,
            beta_2: DEFAULT_BITNET_BETA_2,
            epsilon: DEFAULT_BITNET_EPSILON,
            weight_decay_initial: optimizer_cfg.weight_decay,
            weight_decay_final: optimizer_cfg.weight_decay_final.unwrap_or(0.0),
            second_stage_start: DEFAULT_BITNET_SECOND_STAGE_START,
            total_steps,
        }),
        OptimizerKind::Adamw => optimizer_cfg
            .weight_decay_final
            .filter(|_| {
                matches!(
                    optimizer_cfg.schedule_mode,
                    OptimizerScheduleMode::BitnetB158Reference | OptimizerScheduleMode::Hybrid
                )
            })
            .map(|weight_decay_final| BitNetOptimizerProfile {
                beta_1: DEFAULT_BITNET_BETA_1,
                beta_2: 0.999,
                epsilon: 1.0e-5,
                weight_decay_initial: optimizer_cfg.weight_decay,
                weight_decay_final,
                second_stage_start: DEFAULT_BITNET_SECOND_STAGE_START,
                total_steps,
            }),
        OptimizerKind::MuonHybridExp => None,
    }
}

fn fallback_profile_for_muon_hybrid(
    optimizer_cfg: &OptimizerConfig,
    total_steps: usize,
) -> BitNetOptimizerProfile {
    resolve_bitnet_optimizer_profile(
        &OptimizerConfig {
            name: OptimizerKind::Adamw,
            ..optimizer_cfg.clone()
        },
        total_steps,
    )
    .unwrap_or(BitNetOptimizerProfile {
        beta_1: DEFAULT_BITNET_BETA_1,
        beta_2: 0.999,
        epsilon: 1.0e-5,
        weight_decay_initial: optimizer_cfg.weight_decay,
        weight_decay_final: optimizer_cfg.weight_decay,
        second_stage_start: DEFAULT_BITNET_SECOND_STAGE_START,
        total_steps: total_steps.max(1),
    })
}

fn build_bitnet_adamw_optimizer<B, M>(
    optimizer_cfg: &OptimizerConfig,
    profile: BitNetOptimizerProfile,
) -> OptimizerAdaptor<BitNetAdamW, M, B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    let optimizer = BitNetAdamW {
        momentum: BitNetAdaptiveMomentum {
            beta_1: profile.beta_1,
            beta_2: profile.beta_2,
            epsilon: profile.epsilon,
        },
        weight_decay_initial: profile.weight_decay_initial,
        weight_decay_final: profile.weight_decay_final,
        second_stage_start: profile.second_stage_start,
        total_steps: profile.total_steps,
    };
    let mut optimizer = OptimizerAdaptor::from(optimizer);
    if let Some(clip) = optimizer_cfg.grad_clip_norm {
        optimizer = optimizer.with_grad_clipping(GradientClippingConfig::Norm(clip).init());
    } else if let Some(clip) = optimizer_cfg.grad_clip_value {
        optimizer = optimizer.with_grad_clipping(GradientClippingConfig::Value(clip).init());
    }
    optimizer
}

fn build_muon_hybrid_optimizer<B, M>(
    optimizer_cfg: &OptimizerConfig,
    total_steps: usize,
) -> Result<MuonHybridOptimizer<M, B>>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    let muon_cfg = optimizer_cfg.muon.as_ref().ok_or_else(|| {
        anyhow!("optimizer.muon must be set when optimizer.name = \"muon_hybrid_exp\"")
    })?;
    let fallback_profile = fallback_profile_for_muon_hybrid(optimizer_cfg, total_steps);
    Ok(MuonHybridOptimizer {
        muon: HeadwiseMuon {
            momentum: Momentum::new(&MomentumConfig {
                momentum: muon_cfg.momentum as f64,
                dampening: 0.0,
                nesterov: muon_cfg.nesterov,
            }),
            weight_decay_penalty: Some(optimizer_cfg.weight_decay),
            epsilon: MUON_EPSILON,
            ns_steps: muon_cfg.ns_steps,
            adjust_lr_fn: muon_cfg.adjust_lr_fn,
            split_decoder_heads: muon_cfg.split_decoder_heads,
        },
        fallback: BitNetAdamW {
            momentum: BitNetAdaptiveMomentum {
                beta_1: fallback_profile.beta_1,
                beta_2: fallback_profile.beta_2,
                epsilon: fallback_profile.epsilon,
            },
            weight_decay_initial: fallback_profile.weight_decay_initial,
            weight_decay_final: fallback_profile.weight_decay_final,
            second_stage_start: fallback_profile.second_stage_start,
            total_steps: fallback_profile.total_steps,
        },
        muon_records: HashMap::new(),
        fallback_records: HashMap::new(),
        module: PhantomData,
        grad_clipping: optimizer_cfg
            .grad_clip_norm
            .map(|clip| GradientClippingConfig::Norm(clip).init())
            .or_else(|| {
                optimizer_cfg
                    .grad_clip_value
                    .map(|clip| GradientClippingConfig::Value(clip).init())
            }),
        target_modules: muon_cfg
            .target_modules
            .as_ref()
            .map(|modules| modules.iter().cloned().collect()),
    })
}

fn weight_decay_for_step(
    initial: f32,
    final_value: f32,
    second_stage_start: f32,
    total_steps: usize,
    step: usize,
) -> f32 {
    let total_steps = total_steps.max(1);
    if total_steps <= 1 || initial == final_value {
        return initial;
    }

    let second_stage_start = second_stage_start.clamp(0.0, 1.0);
    let last_step_index = total_steps.saturating_sub(1);
    let step_index = step.saturating_sub(1).min(last_step_index);
    let stage_start_index = ((last_step_index as f32) * second_stage_start).round() as usize;
    if step_index <= stage_start_index {
        return initial;
    }

    let denom = last_step_index.saturating_sub(stage_start_index).max(1) as f32;
    let progress = (step_index.saturating_sub(stage_start_index)) as f32 / denom;
    initial + (final_value - initial) * progress
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MuonHybridConfig, OptimizerKind, OptimizerScheduleMode};
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type TestAutodiffBackend = burn_autodiff::Autodiff<NdArray<f32>>;
    type TestBackend = NdArray<f32>;
    type TestModule = burn::nn::Linear<TestAutodiffBackend>;

    fn optimizer_config() -> OptimizerConfig {
        OptimizerConfig {
            name: OptimizerKind::Adamw,
            learning_rate: 1.0e-3,
            weight_decay: 0.1,
            weight_decay_final: None,
            lr_schedule: None,
            schedule_mode: OptimizerScheduleMode::BdhReference,
            grad_clip_norm: None,
            grad_clip_value: None,
            muon: None,
        }
    }

    #[test]
    fn resolve_optimizer_uses_plain_adamw_by_default() {
        let optimizer =
            resolve_optimizer::<TestAutodiffBackend, TestModule>(&optimizer_config(), 128)
                .expect("optimizer");
        assert!(matches!(optimizer, ResolvedOptimizer::AdamW(_)));
    }

    #[test]
    fn resolve_optimizer_uses_bitnet_profile_for_bitnet_published() {
        let mut config = optimizer_config();
        config.name = OptimizerKind::BitnetPublished;

        let optimizer =
            resolve_optimizer::<TestAutodiffBackend, TestModule>(&config, 128).expect("optimizer");
        assert!(matches!(optimizer, ResolvedOptimizer::BitNetAdamW(_)));
    }

    #[test]
    fn resolve_optimizer_uses_muon_hybrid_when_configured() {
        let mut config = optimizer_config();
        config.name = OptimizerKind::MuonHybridExp;
        config.schedule_mode = OptimizerScheduleMode::Hybrid;
        config.weight_decay_final = Some(0.0);
        config.muon = Some(MuonHybridConfig {
            enabled: true,
            ..Default::default()
        });

        let optimizer =
            resolve_optimizer::<TestAutodiffBackend, TestModule>(&config, 128).expect("optimizer");
        assert!(matches!(optimizer, ResolvedOptimizer::MuonHybrid(_)));
    }

    #[test]
    fn headwise_muon_updates_rank3_tensor_without_panic() {
        let device = Default::default();
        let optimizer = HeadwiseMuon::<TestBackend> {
            momentum: Momentum::new(&MomentumConfig {
                momentum: 0.95,
                dampening: 0.0,
                nesterov: true,
            }),
            weight_decay_penalty: Some(0.1),
            epsilon: MUON_EPSILON,
            ns_steps: 2,
            adjust_lr_fn: MuonAdjustLrFn::MatchRmsAdamw,
            split_decoder_heads: true,
        };
        let tensor = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                (0..24).map(|i| (i as f32 + 1.0) * 0.01).collect::<Vec<_>>(),
                [2, 3, 4],
            ),
            &device,
        );
        let grad = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                (0..24)
                    .map(|i| ((i as f32) * 0.17).sin())
                    .collect::<Vec<_>>(),
                [2, 3, 4],
            ),
            &device,
        );

        let (updated, state) = optimizer.step(1.0e-3, tensor, grad, None);
        let values = updated.into_data().to_vec::<f32>().expect("updated vec");
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(state.is_some());
    }

    #[test]
    fn weight_decay_schedule_decays_to_final_value_in_stage_two() {
        assert!((weight_decay_for_step(0.1, 0.0, 0.5, 10, 1) - 0.1).abs() < 1.0e-6);
        assert!((weight_decay_for_step(0.1, 0.0, 0.5, 10, 5) - 0.1).abs() < 1.0e-6);
        assert!(weight_decay_for_step(0.1, 0.0, 0.5, 10, 10) <= 1.0e-6);
    }
}
