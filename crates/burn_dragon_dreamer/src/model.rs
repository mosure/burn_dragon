mod bdh_challenger_backend;
mod pooled_backend;
mod transformer_backend;

use crate::DreamerConfig;
use burn::module::Module;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_autogaze::FrameFixationTrace;
use burn_dragon_core::{BDH, BDHConfig, MicroTransformerBlock, ResidualConnectorKind};

#[derive(Clone, Debug)]
pub struct DreamerForward<B: Backend> {
    pub total: Tensor<B, 1>,
    pub current: Tensor<B, 1>,
    pub future: Tensor<B, 1>,
    pub prior: Tensor<B, 1>,
    pub gaze: Tensor<B, 1>,
    pub query: Tensor<B, 1>,
    pub recon: Tensor<B, 1>,
    pub tokenizer: Tensor<B, 1>,
    pub tokenizer_recon: Tensor<B, 1>,
    pub slot_align: Tensor<B, 1>,
    pub recon_current: Tensor<B, 1>,
    pub recon_future: Tensor<B, 1>,
    pub recon_edge: Tensor<B, 1>,
    pub recon_motion: Tensor<B, 1>,
}

#[derive(Clone, Debug)]
pub struct DreamerDebugOutput<B: Backend> {
    pub context_reference_frames: Tensor<B, 5>,
    pub context_reconstruction_frames: Tensor<B, 5>,
    pub future_reference_frames: Tensor<B, 5>,
    pub future_reconstruction_frames: Tensor<B, 5>,
    pub context_latents: Tensor<B, 3>,
    pub future_latents: Tensor<B, 3>,
    pub teacher_fixations: Tensor<B, 3>,
    pub predicted_fixations: Tensor<B, 3>,
}

#[derive(Module, Debug)]
pub struct DragonDreamer<B: Backend> {
    peripheral_in: Linear<B>,
    peripheral_hidden: Linear<B>,
    fovea_in: Linear<B>,
    fovea_hidden: Linear<B>,
    world_write_hidden: Linear<B>,
    world_write_out: Linear<B>,
    world_write_gate: Linear<B>,
    transformer_slot_pos_in: Linear<B>,
    transformer_slot_pos_out: Linear<B>,
    transformer_peripheral_token: Linear<B>,
    transformer_fixation_token: Linear<B>,
    transformer_tokenizer_in: Linear<B>,
    transformer_tokenizer_hidden: Linear<B>,
    transformer_prior_blocks: Vec<MicroTransformerBlock<B>>,
    transformer_posterior_blocks: Vec<MicroTransformerBlock<B>>,
    transformer_prior_gate: Linear<B>,
    transformer_posterior_gate: Linear<B>,
    transformer_observation_gate: Linear<B>,
    transformer_summary_in: Linear<B>,
    transformer_summary_out: Linear<B>,
    transformer_grid_in: Conv2d<B>,
    transformer_grid_up: ConvTranspose2d<B>,
    transformer_grid_hidden: Conv2d<B>,
    transformer_grid_out: Conv2d<B>,
    transformer_grid_occ_out: Conv2d<B>,
    transformer_fixation_slot_logits: Linear<B>,
    transformer_fixation_param_hidden: Linear<B>,
    transformer_fixation_param_out: Linear<B>,
    slot_state_from_summary_hidden: Linear<B>,
    slot_state_from_summary_out: Linear<B>,
    slot_state_mix_gate: Linear<B>,
    fixation_hidden: Linear<B>,
    fixation_out: Linear<B>,
    fixation_stop: Linear<B>,
    bdh_input_proj: Linear<B>,
    bdh: BDH<B>,
    observation_hidden: Linear<B>,
    observation_out: Linear<B>,
    prior_hidden: Linear<B>,
    prior_out: Linear<B>,
    posterior_gate_prior: Linear<B>,
    posterior_gate_obs: Linear<B>,
    current_head: Linear<B>,
    future_head: Linear<B>,
    query_head: Linear<B>,
    recon_hidden: Linear<B>,
    recon_hidden2: Linear<B>,
    recon_out: Linear<B>,
    recon_prev_in: Linear<B>,
    recon_fused_hidden: Linear<B>,
    recon_delta_out: Linear<B>,
    recon_blend_out: Linear<B>,
    recon_refine_in: Conv2d<B>,
    recon_refine_hidden: Conv2d<B>,
    recon_refine_out: Conv2d<B>,
    #[module(ignore)]
    use_transformer_baseline: bool,
    #[module(ignore)]
    use_bdh_challenger: bool,
    #[module(ignore)]
    k_fovea: usize,
    #[module(ignore)]
    crop_size: usize,
    #[module(ignore)]
    peripheral_dim: usize,
    #[module(ignore)]
    fovea_dim: usize,
    #[module(ignore)]
    latent_dim: usize,
    #[module(ignore)]
    slot_grid_size: usize,
    #[module(ignore)]
    slot_count: usize,
    #[module(ignore)]
    slot_patch_size: usize,
    #[module(ignore)]
    teacher_dim: usize,
    #[module(ignore)]
    crop_teacher_dim: usize,
    #[module(ignore)]
    channels: usize,
    #[module(ignore)]
    frame_size: usize,
    #[module(ignore)]
    use_bdh_posterior: bool,
    #[module(ignore)]
    current_loss_weight: f32,
    #[module(ignore)]
    future_loss_weight: f32,
    #[module(ignore)]
    prior_loss_weight: f32,
    #[module(ignore)]
    gaze_loss_weight: f32,
    #[module(ignore)]
    query_loss_weight: f32,
    #[module(ignore)]
    recon_loss_weight: f32,
    #[module(ignore)]
    tokenizer_loss_weight: f32,
    #[module(ignore)]
    tokenizer_slot_align_weight: f32,
    #[module(ignore)]
    recon_current_weight: f32,
    #[module(ignore)]
    recon_future_weight: f32,
    #[module(ignore)]
    recon_edge_weight: f32,
    #[module(ignore)]
    recon_motion_weight: f32,
}

impl<B: Backend> DragonDreamer<B> {
    pub fn restore_tokenizer_from(&mut self, source: &Self) {
        self.transformer_tokenizer_in = source.transformer_tokenizer_in.clone();
        self.transformer_tokenizer_hidden = source.transformer_tokenizer_hidden.clone();
        self.transformer_grid_in = source.transformer_grid_in.clone();
        self.transformer_grid_up = source.transformer_grid_up.clone();
        self.transformer_grid_hidden = source.transformer_grid_hidden.clone();
        self.transformer_grid_out = source.transformer_grid_out.clone();
        self.transformer_grid_occ_out = source.transformer_grid_occ_out.clone();
        self.recon_refine_in = source.recon_refine_in.clone();
        self.recon_refine_hidden = source.recon_refine_hidden.clone();
        self.recon_refine_out = source.recon_refine_out.clone();
    }

    pub fn new(config: DreamerConfig, device: &B::Device) -> Self {
        let frame_dim =
            config.channels.max(1) * config.frame_size.max(1) * config.frame_size.max(1);
        let crop_dim = config.channels.max(1) * config.crop_size.max(1) * config.crop_size.max(1);
        let fixation_dim = config.k_fovea.max(1) * 4;
        let fixation_summary_dim = 5;
        let slot_grid_size = config.slot_grid_size.max(1);
        let slot_count = slot_grid_size * slot_grid_size;
        let slot_patch_size = config.frame_size.max(1) / slot_grid_size;
        let decoder_hidden = (config.latent_dim.max(1) / 2).max(32);
        let decoder_hidden2 = (decoder_hidden / 2).max(16);
        assert!(
            slot_patch_size * slot_grid_size == config.frame_size.max(1),
            "slot_grid_size={} must divide frame_size={} for the transformer baseline",
            slot_grid_size,
            config.frame_size.max(1)
        );
        let patch_dim = config.channels.max(1) * slot_patch_size * slot_patch_size;
        let bdh_input_dim = config.latent_dim.max(1) * 2;
        let mut bdh_config = BDHConfig::default();
        bdh_config.n_layer = config.bdh_layers.max(1);
        bdh_config.n_embd = config.latent_dim.max(1);
        bdh_config.dropout = 0.0;
        bdh_config.n_head = config.bdh_heads.max(1);
        bdh_config.mlp_internal_dim_multiplier = config.bdh_mlp_internal_dim_multiplier.max(1);
        bdh_config.vocab_size = 16;
        bdh_config.rollout_fast_steps_per_slow_step = 1;
        bdh_config.residual_connector = ResidualConnectorKind::Vanilla;
        bdh_config.mhc.enabled = false;
        bdh_config.attention_residual.enabled = false;
        bdh_config.block_attention_residual.enabled = false;
        Self {
            peripheral_in: LinearConfig::new(frame_dim, config.peripheral_dim.max(1)).init(device),
            peripheral_hidden: LinearConfig::new(
                config.peripheral_dim.max(1),
                config.peripheral_dim.max(1),
            )
            .init(device),
            fovea_in: LinearConfig::new(crop_dim, config.fovea_dim.max(1)).init(device),
            fovea_hidden: LinearConfig::new(config.fovea_dim.max(1), config.fovea_dim.max(1))
                .init(device),
            world_write_hidden: LinearConfig::new(
                config.fovea_dim.max(1) + 4,
                config.latent_dim.max(1),
            )
            .init(device),
            world_write_out: LinearConfig::new(config.latent_dim.max(1), config.latent_dim.max(1))
                .init(device),
            world_write_gate: LinearConfig::new(config.latent_dim.max(1), 1).init(device),
            transformer_slot_pos_in: LinearConfig::new(2, config.latent_dim.max(1)).init(device),
            transformer_slot_pos_out: LinearConfig::new(
                config.latent_dim.max(1),
                config.latent_dim.max(1),
            )
            .init(device),
            transformer_peripheral_token: LinearConfig::new(
                config.peripheral_dim.max(1),
                config.latent_dim.max(1),
            )
            .init(device),
            transformer_fixation_token: LinearConfig::new(
                fixation_summary_dim,
                config.latent_dim.max(1),
            )
            .init(device),
            transformer_tokenizer_in: LinearConfig::new(patch_dim, config.latent_dim.max(1))
                .init(device),
            transformer_tokenizer_hidden: LinearConfig::new(
                config.latent_dim.max(1),
                config.latent_dim.max(1),
            )
            .init(device),
            transformer_prior_blocks: (0..config.transformer_layers.max(1))
                .map(|_| {
                    MicroTransformerBlock::new(
                        config.latent_dim.max(1),
                        config.transformer_heads.max(1),
                        4,
                        device,
                    )
                })
                .collect(),
            transformer_posterior_blocks: (0..config.transformer_layers.max(1))
                .map(|_| {
                    MicroTransformerBlock::new(
                        config.latent_dim.max(1),
                        config.transformer_heads.max(1),
                        4,
                        device,
                    )
                })
                .collect(),
            transformer_prior_gate: LinearConfig::new(
                config.latent_dim.max(1) * 2,
                config.latent_dim.max(1),
            )
            .init(device),
            transformer_posterior_gate: LinearConfig::new(
                config.latent_dim.max(1) * 2,
                config.latent_dim.max(1),
            )
            .init(device),
            transformer_observation_gate: LinearConfig::new(
                config.latent_dim.max(1) * 2,
                config.latent_dim.max(1),
            )
            .init(device),
            transformer_summary_in: LinearConfig::new(
                slot_count * config.latent_dim.max(1),
                config.latent_dim.max(1),
            )
            .init(device),
            transformer_summary_out: LinearConfig::new(
                config.latent_dim.max(1) * 2,
                config.latent_dim.max(1),
            )
            .init(device),
            transformer_grid_in: Conv2dConfig::new(
                [config.latent_dim.max(1), config.latent_dim.max(1)],
                [3, 3],
            )
            .with_padding(PaddingConfig2d::Same)
            .init(device),
            transformer_grid_up: ConvTranspose2dConfig::new(
                [config.latent_dim.max(1), decoder_hidden],
                [slot_patch_size, slot_patch_size],
            )
            .with_stride([slot_patch_size, slot_patch_size])
            .init(device),
            transformer_grid_hidden: Conv2dConfig::new([decoder_hidden, decoder_hidden2], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            transformer_grid_out: Conv2dConfig::new(
                [decoder_hidden2, config.channels.max(1)],
                [3, 3],
            )
            .with_padding(PaddingConfig2d::Same)
            .init(device),
            transformer_grid_occ_out: Conv2dConfig::new([decoder_hidden2, 1], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            transformer_fixation_slot_logits: LinearConfig::new(
                config.latent_dim.max(1),
                config.k_fovea.max(1),
            )
            .init(device),
            transformer_fixation_param_hidden: LinearConfig::new(
                config.latent_dim.max(1),
                config.latent_dim.max(1),
            )
            .init(device),
            transformer_fixation_param_out: LinearConfig::new(config.latent_dim.max(1), 2)
                .init(device),
            slot_state_from_summary_hidden: LinearConfig::new(
                config.latent_dim.max(1),
                config.latent_dim.max(1) * 2,
            )
            .init(device),
            slot_state_from_summary_out: LinearConfig::new(
                config.latent_dim.max(1) * 2,
                slot_count * config.latent_dim.max(1),
            )
            .init(device),
            slot_state_mix_gate: LinearConfig::new(
                config.latent_dim.max(1) * 3,
                config.latent_dim.max(1),
            )
            .init(device),
            fixation_hidden: LinearConfig::new(
                config.peripheral_dim.max(1) + config.latent_dim.max(1),
                config.latent_dim.max(1),
            )
            .init(device),
            fixation_out: LinearConfig::new(config.latent_dim.max(1), fixation_dim.max(1))
                .init(device),
            fixation_stop: LinearConfig::new(config.latent_dim.max(1), 1).init(device),
            bdh_input_proj: LinearConfig::new(bdh_input_dim, config.latent_dim.max(1)).init(device),
            bdh: BDH::new(bdh_config, device),
            observation_hidden: LinearConfig::new(
                config.peripheral_dim.max(1) + config.latent_dim.max(1) + fixation_summary_dim,
                config.latent_dim.max(1),
            )
            .init(device),
            observation_out: LinearConfig::new(config.latent_dim.max(1), config.latent_dim.max(1))
                .init(device),
            prior_hidden: LinearConfig::new(config.latent_dim.max(1), config.latent_dim.max(1))
                .init(device),
            prior_out: LinearConfig::new(config.latent_dim.max(1), config.latent_dim.max(1))
                .init(device),
            posterior_gate_prior: LinearConfig::new(
                config.latent_dim.max(1),
                config.latent_dim.max(1),
            )
            .init(device),
            posterior_gate_obs: LinearConfig::new(
                config.latent_dim.max(1),
                config.latent_dim.max(1),
            )
            .init(device),
            current_head: LinearConfig::new(config.latent_dim.max(1), config.teacher_dim.max(1))
                .init(device),
            future_head: LinearConfig::new(config.latent_dim.max(1), config.teacher_dim.max(1))
                .init(device),
            query_head: LinearConfig::new(
                config.latent_dim.max(1) + fixation_summary_dim,
                config.crop_teacher_dim.max(1),
            )
            .init(device),
            recon_hidden: LinearConfig::new(config.latent_dim.max(1), config.latent_dim.max(1) * 2)
                .init(device),
            recon_hidden2: LinearConfig::new(
                config.latent_dim.max(1) * 2,
                config.latent_dim.max(1) * 2,
            )
            .init(device),
            recon_out: LinearConfig::new(config.latent_dim.max(1) * 2, frame_dim).init(device),
            recon_prev_in: LinearConfig::new(frame_dim, config.latent_dim.max(1)).init(device),
            recon_fused_hidden: LinearConfig::new(
                config.latent_dim.max(1) * 2,
                config.latent_dim.max(1) * 2,
            )
            .init(device),
            recon_delta_out: LinearConfig::new(config.latent_dim.max(1) * 2, frame_dim)
                .init(device),
            recon_blend_out: LinearConfig::new(config.latent_dim.max(1) * 2, frame_dim)
                .init(device),
            recon_refine_in: Conv2dConfig::new([config.channels.max(1) * 2, 32], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            recon_refine_hidden: Conv2dConfig::new([32, 32], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            recon_refine_out: Conv2dConfig::new([32, config.channels.max(1)], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            use_transformer_baseline: config.latent_backend.is_transformer_baseline(),
            use_bdh_challenger: config.latent_backend.is_bdh_challenger(),
            k_fovea: config.k_fovea.max(1),
            crop_size: config.crop_size.max(1),
            peripheral_dim: config.peripheral_dim.max(1),
            fovea_dim: config.fovea_dim.max(1),
            latent_dim: config.latent_dim.max(1),
            slot_grid_size,
            slot_count,
            slot_patch_size,
            teacher_dim: config.teacher_dim.max(1),
            crop_teacher_dim: config.crop_teacher_dim.max(1),
            channels: config.channels.max(1),
            frame_size: config.frame_size.max(1),
            use_bdh_posterior: config.use_bdh_posterior || config.latent_backend.uses_bdh_core(),
            current_loss_weight: config.current_loss_weight,
            future_loss_weight: config.future_loss_weight,
            prior_loss_weight: config.prior_loss_weight,
            gaze_loss_weight: config.gaze_loss_weight,
            query_loss_weight: config.query_loss_weight,
            recon_loss_weight: config.recon_loss_weight,
            tokenizer_loss_weight: config.tokenizer_loss_weight,
            tokenizer_slot_align_weight: config.tokenizer_slot_align_weight,
            recon_current_weight: config.recon_current_weight,
            recon_future_weight: config.recon_future_weight,
            recon_edge_weight: config.recon_edge_weight,
            recon_motion_weight: config.recon_motion_weight,
        }
    }

    pub fn forward(
        &self,
        clip_frames: Tensor<B, 5>,
        traces: &[FrameFixationTrace],
        teacher_features: Tensor<B, 3>,
        crop_teacher_features: Tensor<B, 3>,
        context_len: usize,
        target_len: usize,
    ) -> DreamerForward<B> {
        self.forward_internal(
            clip_frames,
            traces,
            teacher_features,
            crop_teacher_features,
            context_len,
            target_len,
            false,
        )
        .0
    }

    pub fn forward_tokenizer_pretrain(
        &self,
        clip_frames: Tensor<B, 5>,
        traces: &[FrameFixationTrace],
        teacher_features: Tensor<B, 3>,
        crop_teacher_features: Tensor<B, 3>,
    ) -> DreamerForward<B> {
        if !(self.use_transformer_baseline || self.use_bdh_challenger) {
            return self.forward(
                clip_frames,
                traces,
                teacher_features,
                crop_teacher_features,
                1,
                1,
            );
        }
        self.forward_tokenizer_pretrain_internal(
            clip_frames,
            traces,
            teacher_features,
            crop_teacher_features,
        )
    }

    pub fn forward_with_debug(
        &self,
        clip_frames: Tensor<B, 5>,
        traces: &[FrameFixationTrace],
        teacher_features: Tensor<B, 3>,
        crop_teacher_features: Tensor<B, 3>,
        context_len: usize,
        target_len: usize,
    ) -> (DreamerForward<B>, DreamerDebugOutput<B>) {
        let (forward, debug) = self.forward_internal(
            clip_frames,
            traces,
            teacher_features,
            crop_teacher_features,
            context_len,
            target_len,
            true,
        );
        (
            forward,
            debug.expect("debug output should be present when capture is enabled"),
        )
    }

    fn forward_internal(
        &self,
        clip_frames: Tensor<B, 5>,
        traces: &[FrameFixationTrace],
        teacher_features: Tensor<B, 3>,
        crop_teacher_features: Tensor<B, 3>,
        context_len: usize,
        target_len: usize,
        capture_debug: bool,
    ) -> (DreamerForward<B>, Option<DreamerDebugOutput<B>>) {
        if self.use_transformer_baseline {
            self.forward_internal_transformer(
                clip_frames,
                traces,
                teacher_features,
                crop_teacher_features,
                context_len,
                target_len,
                capture_debug,
            )
        } else if self.use_bdh_challenger {
            self.forward_internal_bdh_challenger(
                clip_frames,
                traces,
                teacher_features,
                crop_teacher_features,
                context_len,
                target_len,
                capture_debug,
            )
        } else {
            self.forward_internal_pooled(
                clip_frames,
                traces,
                teacher_features,
                crop_teacher_features,
                context_len,
                target_len,
                capture_debug,
            )
        }
    }

    fn forward_tokenizer_pretrain_internal(
        &self,
        clip_frames: Tensor<B, 5>,
        traces: &[FrameFixationTrace],
        teacher_features: Tensor<B, 3>,
        crop_teacher_features: Tensor<B, 3>,
    ) -> DreamerForward<B> {
        let device = clip_frames.device();
        let [batch, clip_len, channels, height, width] = clip_frames.shape().dims::<5>();
        let zero = Tensor::<B, 1>::zeros([1], &device);
        let mut current_total = zero.clone();
        let future_total = zero.clone();
        let prior_total = zero.clone();
        let mut gaze_total = zero.clone();
        let mut query_total = zero.clone();
        let mut tokenizer_recon_total = zero.clone();
        let slot_align_total = zero.clone();
        let mut recon_current_total = zero.clone();
        let recon_future_total = zero.clone();
        let mut recon_edge_total = zero.clone();
        let mut recon_motion_total = zero.clone();
        let mut previous_target_frame: Option<Tensor<B, 4>> = None;

        for step in 0..clip_len {
            let frame = clip_frames
                .clone()
                .slice_dim(1, step..step + 1)
                .reshape([batch, channels, height, width]);
            let peripheral = self.encode_peripheral(frame.clone());
            let observed_slots = self.encode_frame_to_slot_tokens(frame.clone());
            let slot_summary = self.summarize_slots(observed_slots.clone());
            let predicted_fixation =
                self.predict_fixation_from_slots(observed_slots.clone(), Some(peripheral));
            let trace_points =
                fixation_tensor_from_traces::<B>(traces, step, self.k_fovea, &device);
            let fixation_points = trace_points
                .clone()
                .slice_dim(1, 0..self.k_fovea * 4)
                .reshape([batch, self.k_fovea, 4]);
            let fixation_stop = trace_points
                .clone()
                .slice_dim(1, self.k_fovea * 4..self.k_fovea * 4 + 1);
            let fixation_summary =
                summarize_fixation_set(fixation_points.clone(), fixation_stop.clone());
            gaze_total = gaze_total + mse_loss(predicted_fixation, trace_points.clone());

            let (tokenizer_recon, tokenizer_occupancy) =
                self.decode_frame_from_slots_components(observed_slots.clone(), None);
            tokenizer_recon_total = tokenizer_recon_total
                + tokenizer_reconstruction_loss_frame(
                    tokenizer_recon.clone(),
                    frame.clone().detach(),
                )
                + occupancy_loss_frame(tokenizer_occupancy, frame.clone().detach())
                    .mul_scalar(0.35);
            recon_current_total = recon_current_total
                + rollout_reconstruction_loss_frame(
                    tokenizer_recon.clone(),
                    frame.clone().detach(),
                );
            recon_edge_total = recon_edge_total
                + edge_mse_loss_frame(tokenizer_recon.clone(), frame.clone().detach());
            if let Some(previous_target) = previous_target_frame.clone() {
                recon_motion_total = recon_motion_total
                    + motion_mse_loss_frame(
                        tokenizer_recon.clone(),
                        previous_target.clone(),
                        frame.clone().detach(),
                        previous_target,
                    );
            }

            let current_pred = self.current_head.forward(slot_summary.clone());
            let current_target = teacher_features
                .clone()
                .slice_dim(1, step..step + 1)
                .reshape([batch, self.teacher_dim])
                .detach();
            current_total = current_total + mse_loss(current_pred, current_target);

            let query_pred = self
                .query_head
                .forward(Tensor::cat(vec![slot_summary, fixation_summary], 1));
            let query_target = crop_teacher_features
                .clone()
                .slice_dim(1, step..step + 1)
                .reshape([batch, self.crop_teacher_dim])
                .detach();
            query_total = query_total + mse_loss(query_pred, query_target);

            let detached_frame = frame.detach();
            previous_target_frame = Some(detached_frame);
        }

        let frame_denom = clip_len.max(1) as f32;
        let current = current_total.div_scalar(frame_denom);
        let future = future_total;
        let prior = prior_total;
        let gaze = gaze_total.div_scalar(frame_denom);
        let query = query_total.div_scalar(frame_denom);
        let tokenizer_recon = tokenizer_recon_total.div_scalar(frame_denom);
        let slot_align = slot_align_total;
        let tokenizer = tokenizer_recon.clone();
        let recon_current = recon_current_total.div_scalar(frame_denom);
        let recon_future = recon_future_total;
        let recon_edge = recon_edge_total.div_scalar(frame_denom);
        let recon_motion = recon_motion_total.div_scalar(clip_len.saturating_sub(1).max(1) as f32);
        let recon = recon_current.clone().mul_scalar(self.recon_current_weight)
            + recon_edge.clone().mul_scalar(self.recon_edge_weight)
            + recon_motion.clone().mul_scalar(self.recon_motion_weight);
        let total = current.clone().mul_scalar(self.current_loss_weight)
            + gaze.clone().mul_scalar(self.gaze_loss_weight)
            + query.clone().mul_scalar(self.query_loss_weight)
            + tokenizer.clone().mul_scalar(self.tokenizer_loss_weight)
            + recon.clone().mul_scalar(self.recon_loss_weight);

        DreamerForward {
            total,
            current,
            future,
            prior,
            gaze,
            query,
            recon,
            tokenizer,
            tokenizer_recon,
            slot_align,
            recon_current,
            recon_future,
            recon_edge,
            recon_motion,
        }
    }

    fn forward_internal_pooled(
        &self,
        clip_frames: Tensor<B, 5>,
        traces: &[FrameFixationTrace],
        teacher_features: Tensor<B, 3>,
        crop_teacher_features: Tensor<B, 3>,
        context_len: usize,
        target_len: usize,
        capture_debug: bool,
    ) -> (DreamerForward<B>, Option<DreamerDebugOutput<B>>) {
        let device = clip_frames.device();
        let [batch, clip_len, channels, height, width] = clip_frames.shape().dims::<5>();
        let context_len = context_len.clamp(1, clip_len.saturating_sub(1).max(1));
        let target_len = target_len.clamp(1, clip_len.saturating_sub(context_len).max(1));
        let zero = Tensor::<B, 1>::zeros([1], &device);
        let mut post = Tensor::<B, 2>::zeros([batch, self.latent_dim], &device);
        let mut current_total = zero.clone();
        let mut future_total = zero.clone();
        let mut prior_total = zero.clone();
        let mut gaze_total = zero.clone();
        let mut query_total = zero.clone();
        let tokenizer_total = zero.clone();
        let tokenizer_recon_total = zero.clone();
        let slot_align_total = zero.clone();
        let mut recon_current_total = zero.clone();
        let mut recon_future_total = zero.clone();
        let mut recon_edge_total = zero.clone();
        let mut recon_motion_total = zero.clone();
        let mut bdh_state = self.use_bdh_posterior.then(|| self.bdh.init_state());
        let mut context_reference_frames = Vec::new();
        let mut context_reconstruction_frames = Vec::new();
        let mut future_reference_frames = Vec::new();
        let mut future_reconstruction_frames = Vec::new();
        let mut context_latents = Vec::new();
        let mut future_latents = Vec::new();
        let mut teacher_fixations = Vec::new();
        let mut predicted_fixations = Vec::new();
        let mut previous_real_frame: Option<Tensor<B, 4>> = None;
        let mut previous_target_frame: Option<Tensor<B, 4>> = None;

        for step in 0..context_len {
            let frame = clip_frames
                .clone()
                .slice_dim(1, step..step + 1)
                .reshape([batch, channels, height, width]);
            let peripheral = self.encode_peripheral(frame.clone());
            let predicted_fixation = self.predict_fixation(peripheral.clone(), post.clone());
            let trace_points =
                fixation_tensor_from_traces::<B>(traces, step, self.k_fovea, &device);
            let fixation_points = trace_points
                .clone()
                .slice_dim(1, 0..self.k_fovea * 4)
                .reshape([batch, self.k_fovea, 4]);
            let fixation_stop = trace_points
                .clone()
                .slice_dim(1, self.k_fovea * 4..self.k_fovea * 4 + 1);
            let fixation_summary =
                summarize_fixation_set(fixation_points.clone(), fixation_stop.clone());
            gaze_total = gaze_total + mse_loss(predicted_fixation.clone(), trace_points.clone());

            let crops = extract_crops(frame.clone(), traces, step, self.crop_size, self.k_fovea);
            let fovea_tokens = self.encode_fovea_tokens(crops);
            let world_summary = self.merge_world_writes(fovea_tokens, fixation_points);
            let observation =
                self.merge_observation(peripheral, world_summary, fixation_summary.clone());
            let prior = self.prior_step(post.clone());
            post = self.posterior_step(prior.clone(), observation, bdh_state.as_mut());
            let current_recon = self.decode_frame(post.clone(), previous_real_frame.clone());
            let current_pred = self.current_head.forward(post.clone());
            let current_target = teacher_features
                .clone()
                .slice_dim(1, step..step + 1)
                .reshape([batch, self.teacher_dim])
                .detach();
            current_total = current_total + mse_loss(current_pred, current_target);
            prior_total = prior_total + mse_loss(prior, post.clone().detach());
            recon_current_total = recon_current_total
                + rollout_reconstruction_loss_frame(current_recon.clone(), frame.clone().detach());
            recon_edge_total = recon_edge_total
                + edge_mse_loss_frame(current_recon.clone(), frame.clone().detach());
            if let Some(previous_target) = previous_target_frame.clone() {
                recon_motion_total = recon_motion_total
                    + motion_mse_loss_frame(
                        current_recon.clone(),
                        previous_target.clone(),
                        frame.clone().detach(),
                        previous_target,
                    );
            }

            let query_pred = self
                .query_head
                .forward(Tensor::cat(vec![post.clone(), fixation_summary], 1));
            let query_target = crop_teacher_features
                .clone()
                .slice_dim(1, step..step + 1)
                .reshape([batch, self.crop_teacher_dim])
                .detach();
            query_total = query_total + mse_loss(query_pred, query_target);

            if capture_debug {
                context_reference_frames.push(frame.clone().detach().unsqueeze_dim::<5>(1));
                context_reconstruction_frames.push(current_recon.detach().unsqueeze_dim::<5>(1));
                context_latents.push(post.clone().detach().unsqueeze_dim::<3>(1));
                teacher_fixations.push(trace_points.clone().detach().unsqueeze_dim::<3>(1));
                predicted_fixations.push(predicted_fixation.detach().unsqueeze_dim::<3>(1));
            }
            let detached_frame = frame.detach();
            previous_real_frame = Some(detached_frame.clone());
            previous_target_frame = Some(detached_frame);
        }

        let mut rollout = post;
        let mut previous_rollout_frame = previous_real_frame;
        for step in 0..target_len {
            rollout = self.prior_step(rollout);
            let future_recon = self.decode_frame(rollout.clone(), previous_rollout_frame.clone());
            let future_pred = self.future_head.forward(rollout.clone());
            let target_idx = context_len + step;
            let future_target = teacher_features
                .clone()
                .slice_dim(1, target_idx..target_idx + 1)
                .reshape([batch, self.teacher_dim])
                .detach();
            future_total = future_total + mse_loss(future_pred, future_target);
            let future_frame = clip_frames
                .clone()
                .slice_dim(1, target_idx..target_idx + 1)
                .reshape([batch, channels, height, width]);
            recon_future_total = recon_future_total
                + rollout_reconstruction_loss_frame(
                    future_recon.clone(),
                    future_frame.clone().detach(),
                );
            recon_edge_total = recon_edge_total
                + edge_mse_loss_frame(future_recon.clone(), future_frame.clone().detach());
            if let (Some(previous_rollout), Some(previous_target)) = (
                previous_rollout_frame.clone(),
                previous_target_frame.clone(),
            ) {
                recon_motion_total = recon_motion_total
                    + motion_mse_loss_frame(
                        future_recon.clone(),
                        previous_rollout,
                        future_frame.clone().detach(),
                        previous_target,
                    );
            }
            let trace_points =
                fixation_tensor_from_traces::<B>(traces, target_idx, self.k_fovea, &device);
            let imagined_peripheral = self.encode_peripheral(future_recon.clone());
            let imagined_predicted_fixation =
                self.predict_fixation(imagined_peripheral, rollout.clone());
            let imagined_fixation_points = imagined_predicted_fixation
                .clone()
                .slice_dim(1, 0..self.k_fovea * 4)
                .reshape([batch, self.k_fovea, 4]);
            let imagined_fixation_stop = imagined_predicted_fixation
                .clone()
                .slice_dim(1, self.k_fovea * 4..self.k_fovea * 4 + 1);
            let imagined_fixation_summary =
                summarize_fixation_set(imagined_fixation_points, imagined_fixation_stop);
            gaze_total =
                gaze_total + mse_loss(imagined_predicted_fixation.clone(), trace_points.clone());
            let query_pred = self.query_head.forward(Tensor::cat(
                vec![rollout.clone(), imagined_fixation_summary],
                1,
            ));
            let query_target = crop_teacher_features
                .clone()
                .slice_dim(1, target_idx..target_idx + 1)
                .reshape([batch, self.crop_teacher_dim])
                .detach();
            query_total = query_total + mse_loss(query_pred, query_target);
            if capture_debug {
                future_reference_frames.push(future_frame.clone().detach().unsqueeze_dim::<5>(1));
                future_reconstruction_frames
                    .push(future_recon.clone().detach().unsqueeze_dim::<5>(1));
                future_latents.push(rollout.clone().detach().unsqueeze_dim::<3>(1));
                teacher_fixations.push(trace_points.detach().unsqueeze_dim::<3>(1));
                predicted_fixations
                    .push(imagined_predicted_fixation.detach().unsqueeze_dim::<3>(1));
            }
            previous_rollout_frame = Some(future_recon);
            previous_target_frame = Some(future_frame.clone().detach());
        }

        let context_denom = context_len as f32;
        let target_denom = target_len as f32;
        let fixation_denom = (context_len + target_len) as f32;
        let current = current_total.div_scalar(context_denom);
        let future = future_total.div_scalar(target_denom);
        let prior = prior_total.div_scalar(context_denom);
        let gaze = gaze_total.div_scalar(fixation_denom);
        let query = query_total.div_scalar(fixation_denom);
        let recon_current = recon_current_total.div_scalar(context_denom);
        let recon_future = recon_future_total.div_scalar(target_denom);
        let recon_edge = recon_edge_total.div_scalar((context_len + target_len) as f32);
        let recon_motion = recon_motion_total
            .div_scalar((context_len + target_len).saturating_sub(1).max(1) as f32);
        let recon = recon_current.clone().mul_scalar(self.recon_current_weight)
            + recon_future.clone().mul_scalar(self.recon_future_weight)
            + recon_edge.clone().mul_scalar(self.recon_edge_weight);
        let recon = recon + recon_motion.clone().mul_scalar(self.recon_motion_weight);
        let total = current.clone().mul_scalar(self.current_loss_weight)
            + future.clone().mul_scalar(self.future_loss_weight)
            + prior.clone().mul_scalar(self.prior_loss_weight)
            + gaze.clone().mul_scalar(self.gaze_loss_weight)
            + query.clone().mul_scalar(self.query_loss_weight)
            + recon.clone().mul_scalar(self.recon_loss_weight);

        let forward = DreamerForward {
            total,
            current,
            future,
            prior,
            gaze,
            query,
            recon,
            tokenizer: tokenizer_total,
            tokenizer_recon: tokenizer_recon_total,
            slot_align: slot_align_total,
            recon_current,
            recon_future,
            recon_edge,
            recon_motion,
        };
        let debug = if capture_debug {
            Some(DreamerDebugOutput {
                context_reference_frames: Tensor::cat(context_reference_frames, 1),
                context_reconstruction_frames: Tensor::cat(context_reconstruction_frames, 1),
                future_reference_frames: Tensor::cat(future_reference_frames, 1),
                future_reconstruction_frames: Tensor::cat(future_reconstruction_frames, 1),
                context_latents: Tensor::cat(context_latents, 1),
                future_latents: Tensor::cat(future_latents, 1),
                teacher_fixations: Tensor::cat(teacher_fixations, 1),
                predicted_fixations: Tensor::cat(predicted_fixations, 1),
            })
        } else {
            None
        };
        (forward, debug)
    }

    fn forward_internal_transformer(
        &self,
        clip_frames: Tensor<B, 5>,
        traces: &[FrameFixationTrace],
        teacher_features: Tensor<B, 3>,
        crop_teacher_features: Tensor<B, 3>,
        context_len: usize,
        target_len: usize,
        capture_debug: bool,
    ) -> (DreamerForward<B>, Option<DreamerDebugOutput<B>>) {
        let device = clip_frames.device();
        let [batch, clip_len, channels, height, width] = clip_frames.shape().dims::<5>();
        let context_len = context_len.clamp(1, clip_len.saturating_sub(1).max(1));
        let target_len = target_len.clamp(1, clip_len.saturating_sub(context_len).max(1));
        let zero = Tensor::<B, 1>::zeros([1], &device);
        let mut slots = self.zero_slot_state(batch, &device);
        let mut current_total = zero.clone();
        let mut future_total = zero.clone();
        let mut prior_total = zero.clone();
        let mut prior_terms = 0usize;
        let mut gaze_total = zero.clone();
        let mut query_total = zero.clone();
        let mut tokenizer_recon_total = zero.clone();
        let mut slot_align_total = zero.clone();
        let mut recon_current_total = zero.clone();
        let mut recon_future_total = zero.clone();
        let mut recon_edge_total = zero.clone();
        let mut recon_motion_total = zero.clone();
        let mut context_reference_frames = Vec::new();
        let mut context_reconstruction_frames = Vec::new();
        let mut future_reference_frames = Vec::new();
        let mut future_reconstruction_frames = Vec::new();
        let mut context_latents = Vec::new();
        let mut future_latents = Vec::new();
        let mut teacher_fixations = Vec::new();
        let mut predicted_fixations = Vec::new();
        let mut previous_real_frame: Option<Tensor<B, 4>> = None;
        let mut previous_target_frame: Option<Tensor<B, 4>> = None;

        for step in 0..context_len {
            let frame = clip_frames
                .clone()
                .slice_dim(1, step..step + 1)
                .reshape([batch, channels, height, width]);
            let peripheral = self.encode_peripheral(frame.clone());
            let predicted_fixation =
                self.predict_fixation_from_slots(slots.clone(), Some(peripheral.clone()));
            let trace_points =
                fixation_tensor_from_traces::<B>(traces, step, self.k_fovea, &device);
            let fixation_points = trace_points
                .clone()
                .slice_dim(1, 0..self.k_fovea * 4)
                .reshape([batch, self.k_fovea, 4]);
            let fixation_stop = trace_points
                .clone()
                .slice_dim(1, self.k_fovea * 4..self.k_fovea * 4 + 1);
            let fixation_summary =
                summarize_fixation_set(fixation_points.clone(), fixation_stop.clone());
            gaze_total = gaze_total + mse_loss(predicted_fixation.clone(), trace_points.clone());

            let crops = extract_crops(frame.clone(), traces, step, self.crop_size, self.k_fovea);
            let fovea_tokens = self.encode_fovea_tokens(crops);
            let observed_slots = self.encode_frame_to_slot_tokens(frame.clone());
            let prior_slots = self.prior_step_slots(slots.clone());
            let (tokenizer_recon, tokenizer_occupancy) =
                self.decode_frame_from_slots_components(observed_slots.clone(), None);
            slots = self.posterior_step_slots(
                prior_slots.clone(),
                observed_slots.clone(),
                peripheral,
                fovea_tokens,
                fixation_points,
                fixation_summary.clone(),
            );
            let post_summary = self.summarize_slots(slots.clone());
            let (current_recon, current_occupancy) =
                self.decode_frame_from_slots_components(slots.clone(), None);
            let current_pred = self.current_head.forward(post_summary.clone());
            let current_target = teacher_features
                .clone()
                .slice_dim(1, step..step + 1)
                .reshape([batch, self.teacher_dim])
                .detach();
            current_total = current_total + mse_loss(current_pred, current_target);
            prior_total = prior_total + mse_loss_tokens(prior_slots, slots.clone().detach());
            prior_terms += 1;
            slot_align_total =
                slot_align_total + mse_loss_tokens(slots.clone(), observed_slots.detach());
            tokenizer_recon_total = tokenizer_recon_total
                + tokenizer_reconstruction_loss_frame(tokenizer_recon, frame.clone().detach())
                + occupancy_loss_frame(tokenizer_occupancy, frame.clone().detach())
                    .mul_scalar(0.35);
            recon_current_total = recon_current_total
                + rollout_reconstruction_loss_frame(current_recon.clone(), frame.clone().detach())
                + occupancy_loss_frame(current_occupancy, frame.clone().detach()).mul_scalar(0.20);
            recon_edge_total = recon_edge_total
                + edge_mse_loss_frame(current_recon.clone(), frame.clone().detach());
            if let Some(previous_target) = previous_target_frame.clone() {
                recon_motion_total = recon_motion_total
                    + motion_mse_loss_frame(
                        current_recon.clone(),
                        previous_target.clone(),
                        frame.clone().detach(),
                        previous_target,
                    );
            }

            let query_pred = self
                .query_head
                .forward(Tensor::cat(vec![post_summary.clone(), fixation_summary], 1));
            let query_target = crop_teacher_features
                .clone()
                .slice_dim(1, step..step + 1)
                .reshape([batch, self.crop_teacher_dim])
                .detach();
            query_total = query_total + mse_loss(query_pred, query_target);

            if capture_debug {
                context_reference_frames.push(frame.clone().detach().unsqueeze_dim::<5>(1));
                context_reconstruction_frames.push(current_recon.detach().unsqueeze_dim::<5>(1));
                context_latents.push(post_summary.detach().unsqueeze_dim::<3>(1));
                teacher_fixations.push(trace_points.clone().detach().unsqueeze_dim::<3>(1));
                predicted_fixations.push(predicted_fixation.detach().unsqueeze_dim::<3>(1));
            }
            let detached_frame = frame.detach();
            previous_real_frame = Some(detached_frame.clone());
            previous_target_frame = Some(detached_frame);
        }

        let mut rollout_slots = slots;
        let mut previous_rollout_frame = previous_real_frame;
        for step in 0..target_len {
            rollout_slots = self.prior_step_slots(rollout_slots);
            let rollout_summary = self.summarize_slots(rollout_slots.clone());
            let (future_recon, future_occupancy) =
                self.decode_frame_from_slots_components(rollout_slots.clone(), None);
            let future_pred = self.future_head.forward(rollout_summary.clone());
            let target_idx = context_len + step;
            let future_target = teacher_features
                .clone()
                .slice_dim(1, target_idx..target_idx + 1)
                .reshape([batch, self.teacher_dim])
                .detach();
            future_total = future_total + mse_loss(future_pred, future_target);
            let future_frame = clip_frames
                .clone()
                .slice_dim(1, target_idx..target_idx + 1)
                .reshape([batch, channels, height, width]);
            let future_slot_target =
                self.encode_frame_to_slot_tokens(future_frame.clone().detach());
            let (future_tokenizer_recon, future_tokenizer_occupancy) =
                self.decode_frame_from_slots_components(future_slot_target.clone(), None);
            prior_total =
                prior_total + mse_loss_tokens(rollout_slots.clone(), future_slot_target.detach());
            prior_terms += 1;
            tokenizer_recon_total = tokenizer_recon_total
                + tokenizer_reconstruction_loss_frame(
                    future_tokenizer_recon,
                    future_frame.clone().detach(),
                )
                + occupancy_loss_frame(future_tokenizer_occupancy, future_frame.clone().detach())
                    .mul_scalar(0.35);
            recon_future_total = recon_future_total
                + rollout_reconstruction_loss_frame(
                    future_recon.clone(),
                    future_frame.clone().detach(),
                )
                + occupancy_loss_frame(future_occupancy, future_frame.clone().detach())
                    .mul_scalar(0.20);
            recon_edge_total = recon_edge_total
                + edge_mse_loss_frame(future_recon.clone(), future_frame.clone().detach());
            if let (Some(previous_rollout), Some(previous_target)) = (
                previous_rollout_frame.clone(),
                previous_target_frame.clone(),
            ) {
                recon_motion_total = recon_motion_total
                    + motion_mse_loss_frame(
                        future_recon.clone(),
                        previous_rollout,
                        future_frame.clone().detach(),
                        previous_target,
                    );
            }
            let trace_points =
                fixation_tensor_from_traces::<B>(traces, target_idx, self.k_fovea, &device);
            let imagined_predicted_fixation =
                self.predict_fixation_from_slots(rollout_slots.clone(), None);
            let imagined_fixation_points = imagined_predicted_fixation
                .clone()
                .slice_dim(1, 0..self.k_fovea * 4)
                .reshape([batch, self.k_fovea, 4]);
            let imagined_fixation_stop = imagined_predicted_fixation
                .clone()
                .slice_dim(1, self.k_fovea * 4..self.k_fovea * 4 + 1);
            let imagined_fixation_summary =
                summarize_fixation_set(imagined_fixation_points, imagined_fixation_stop);
            gaze_total =
                gaze_total + mse_loss(imagined_predicted_fixation.clone(), trace_points.clone());
            let query_pred = self.query_head.forward(Tensor::cat(
                vec![rollout_summary.clone(), imagined_fixation_summary],
                1,
            ));
            let query_target = crop_teacher_features
                .clone()
                .slice_dim(1, target_idx..target_idx + 1)
                .reshape([batch, self.crop_teacher_dim])
                .detach();
            query_total = query_total + mse_loss(query_pred, query_target);
            if capture_debug {
                future_reference_frames.push(future_frame.clone().detach().unsqueeze_dim::<5>(1));
                future_reconstruction_frames
                    .push(future_recon.clone().detach().unsqueeze_dim::<5>(1));
                future_latents.push(rollout_summary.clone().detach().unsqueeze_dim::<3>(1));
                teacher_fixations.push(trace_points.detach().unsqueeze_dim::<3>(1));
                predicted_fixations
                    .push(imagined_predicted_fixation.detach().unsqueeze_dim::<3>(1));
            }
            previous_rollout_frame = Some(future_recon);
            previous_target_frame = Some(future_frame.clone().detach());
        }

        let context_denom = context_len as f32;
        let target_denom = target_len as f32;
        let fixation_denom = (context_len + target_len) as f32;
        let current = current_total.div_scalar(context_denom);
        let future = future_total.div_scalar(target_denom);
        let prior = prior_total.div_scalar(prior_terms.max(1) as f32);
        let gaze = gaze_total.div_scalar(fixation_denom);
        let query = query_total.div_scalar(fixation_denom);
        let tokenizer_recon = tokenizer_recon_total.div_scalar((context_len + target_len) as f32);
        let slot_align = slot_align_total.div_scalar(context_denom);
        let tokenizer = tokenizer_recon.clone()
            + slot_align
                .clone()
                .mul_scalar(self.tokenizer_slot_align_weight);
        let recon_current = recon_current_total.div_scalar(context_denom);
        let recon_future = recon_future_total.div_scalar(target_denom);
        let recon_edge = recon_edge_total.div_scalar((context_len + target_len) as f32);
        let recon_motion = recon_motion_total
            .div_scalar((context_len + target_len).saturating_sub(1).max(1) as f32);
        let recon = recon_current.clone().mul_scalar(self.recon_current_weight)
            + recon_future.clone().mul_scalar(self.recon_future_weight)
            + recon_edge.clone().mul_scalar(self.recon_edge_weight);
        let recon = recon + recon_motion.clone().mul_scalar(self.recon_motion_weight);
        let total = current.clone().mul_scalar(self.current_loss_weight)
            + future.clone().mul_scalar(self.future_loss_weight)
            + prior.clone().mul_scalar(self.prior_loss_weight)
            + gaze.clone().mul_scalar(self.gaze_loss_weight)
            + query.clone().mul_scalar(self.query_loss_weight)
            + tokenizer.clone().mul_scalar(self.tokenizer_loss_weight)
            + recon.clone().mul_scalar(self.recon_loss_weight);

        let forward = DreamerForward {
            total,
            current,
            future,
            prior,
            gaze,
            query,
            recon,
            tokenizer,
            tokenizer_recon,
            slot_align,
            recon_current,
            recon_future,
            recon_edge,
            recon_motion,
        };
        let debug = if capture_debug {
            Some(DreamerDebugOutput {
                context_reference_frames: Tensor::cat(context_reference_frames, 1),
                context_reconstruction_frames: Tensor::cat(context_reconstruction_frames, 1),
                future_reference_frames: Tensor::cat(future_reference_frames, 1),
                future_reconstruction_frames: Tensor::cat(future_reconstruction_frames, 1),
                context_latents: Tensor::cat(context_latents, 1),
                future_latents: Tensor::cat(future_latents, 1),
                teacher_fixations: Tensor::cat(teacher_fixations, 1),
                predicted_fixations: Tensor::cat(predicted_fixations, 1),
            })
        } else {
            None
        };
        (forward, debug)
    }

    fn encode_peripheral(&self, frame: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch, channels, height, width] = frame.shape().dims::<4>();
        let flat = frame.reshape([batch, channels * height * width]);
        activation::gelu(
            self.peripheral_hidden
                .forward(activation::gelu(self.peripheral_in.forward(flat))),
        )
    }

    fn encode_fovea_tokens(&self, crops: Tensor<B, 5>) -> Tensor<B, 3> {
        let [batch, k, channels, crop_h, crop_w] = crops.shape().dims::<5>();
        let flat = crops.reshape([batch * k, channels * crop_h * crop_w]);
        activation::gelu(
            self.fovea_hidden
                .forward(activation::gelu(self.fovea_in.forward(flat))),
        )
        .reshape([batch, k, self.fovea_dim])
    }
}

fn mse_loss<B: Backend>(lhs: Tensor<B, 2>, rhs: Tensor<B, 2>) -> Tensor<B, 1> {
    (lhs - rhs).powf_scalar(2.0).mean().reshape([1])
}

fn mse_loss_tokens<B: Backend>(lhs: Tensor<B, 3>, rhs: Tensor<B, 3>) -> Tensor<B, 1> {
    (lhs - rhs).powf_scalar(2.0).mean().reshape([1])
}

fn mse_loss_frame<B: Backend>(lhs: Tensor<B, 4>, rhs: Tensor<B, 4>) -> Tensor<B, 1> {
    (lhs - rhs).powf_scalar(2.0).mean().reshape([1])
}

fn frame_unit_interval<B: Backend>(frame: Tensor<B, 4>) -> Tensor<B, 4> {
    frame.mul_scalar(0.5).add_scalar(0.5).clamp(0.0, 1.0)
}

fn foreground_target_frame<B: Backend>(frame: Tensor<B, 4>) -> Tensor<B, 4> {
    frame_unit_interval(frame).greater_elem(0.15).float()
}

fn rollout_reconstruction_loss_frame<B: Backend>(
    lhs: Tensor<B, 4>,
    rhs: Tensor<B, 4>,
) -> Tensor<B, 1> {
    let base = mse_loss_frame(lhs.clone(), rhs.clone());
    let target_occupancy = foreground_target_frame(rhs.clone());
    let foreground_weight = target_occupancy.clone().mul_scalar(6.0).add_scalar(1.0);
    let weighted = (lhs.clone() - rhs.clone())
        .powf_scalar(2.0)
        .mul(foreground_weight)
        .mean()
        .reshape([1]);
    let foreground_only = (lhs.clone() - rhs.clone())
        .powf_scalar(2.0)
        .mul(target_occupancy.clone())
        .sum()
        .div(target_occupancy.clone().sum().add_scalar(1.0))
        .reshape([1]);
    let lhs_unit = frame_unit_interval(lhs);
    let lhs_safe = lhs_unit.clone().clamp(1.0e-4, 1.0 - 1.0e-4);
    let one_minus_lhs = lhs_unit
        .clone()
        .mul_scalar(-1.0)
        .add_scalar(1.0)
        .clamp(1.0e-4, 1.0 - 1.0e-4);
    let one_minus_target = target_occupancy.clone().mul_scalar(-1.0).add_scalar(1.0);
    let bce = target_occupancy
        .clone()
        .mul(lhs_safe.log())
        .mul_scalar(3.0)
        .add(one_minus_target.clone().mul(one_minus_lhs.log()))
        .mul_scalar(-1.0)
        .mean()
        .reshape([1]);
    let overlap = lhs_unit
        .clone()
        .mul(target_occupancy.clone())
        .sum()
        .mul_scalar(2.0);
    let mass = lhs_unit.clone().sum() + target_occupancy.clone().sum();
    let dice_loss = overlap
        .add_scalar(1.0e-4)
        .div(mass.add_scalar(1.0e-4))
        .mul_scalar(-1.0)
        .add_scalar(1.0)
        .reshape([1]);
    let background_energy = lhs_unit.mul(one_minus_target).mean().reshape([1]);
    base.mul_scalar(0.10)
        + weighted.mul_scalar(0.28)
        + foreground_only.mul_scalar(0.24)
        + bce.mul_scalar(0.16)
        + dice_loss.mul_scalar(0.16)
        + background_energy.mul_scalar(0.06)
}

fn tokenizer_reconstruction_loss_frame<B: Backend>(
    lhs: Tensor<B, 4>,
    rhs: Tensor<B, 4>,
) -> Tensor<B, 1> {
    let base = rollout_reconstruction_loss_frame(lhs.clone(), rhs.clone());
    let center = center_of_mass_loss(
        frame_unit_interval(lhs.clone()),
        foreground_target_frame(rhs.clone()),
    );
    let mass = mass_difference_loss(frame_unit_interval(lhs), foreground_target_frame(rhs));
    base.mul_scalar(0.65) + center.mul_scalar(0.20) + mass.mul_scalar(0.15)
}

pub(crate) fn occupancy_loss_frame<B: Backend>(
    pred_occupancy: Tensor<B, 4>,
    rhs: Tensor<B, 4>,
) -> Tensor<B, 1> {
    let target = foreground_target_frame(rhs);
    let pred_safe = pred_occupancy.clone().clamp(1.0e-4, 1.0 - 1.0e-4);
    let one_minus_pred = pred_occupancy
        .clone()
        .mul_scalar(-1.0)
        .add_scalar(1.0)
        .clamp(1.0e-4, 1.0 - 1.0e-4);
    let one_minus_target = target.clone().mul_scalar(-1.0).add_scalar(1.0);
    let bce = target
        .clone()
        .mul(pred_safe.log())
        .mul_scalar(4.0)
        .add(one_minus_target.clone().mul(one_minus_pred.log()))
        .mul_scalar(-1.0)
        .mean()
        .reshape([1]);
    let overlap = pred_occupancy
        .clone()
        .mul(target.clone())
        .sum()
        .mul_scalar(2.0);
    let mass = pred_occupancy.clone().sum() + target.clone().sum();
    let dice = overlap
        .add_scalar(1.0e-4)
        .div(mass.add_scalar(1.0e-4))
        .mul_scalar(-1.0)
        .add_scalar(1.0)
        .reshape([1]);
    let mass = mass_difference_loss(pred_occupancy, target);
    bce.mul_scalar(0.5) + dice.mul_scalar(0.35) + mass.mul_scalar(0.15)
}

fn mass_difference_loss<B: Backend>(lhs: Tensor<B, 4>, rhs: Tensor<B, 4>) -> Tensor<B, 1> {
    let lhs_mass = lhs.mean().reshape([1]);
    let rhs_mass = rhs.mean().reshape([1]);
    (lhs_mass - rhs_mass).abs()
}

fn center_of_mass_loss<B: Backend>(lhs_unit: Tensor<B, 4>, rhs_unit: Tensor<B, 4>) -> Tensor<B, 1> {
    let [batch, _channels, height, width] = lhs_unit.shape().dims::<4>();
    if height == 0 || width == 0 {
        return Tensor::<B, 1>::zeros([1], &lhs_unit.device());
    }
    let lhs_mass = lhs_unit.mean_dim(1).reshape([batch, height * width]);
    let rhs_mass = rhs_unit.mean_dim(1).reshape([batch, height * width]);

    let mut x_coords = Vec::with_capacity(height * width);
    let mut y_coords = Vec::with_capacity(height * width);
    let width_denom = width.saturating_sub(1).max(1) as f32;
    let height_denom = height.saturating_sub(1).max(1) as f32;
    for y in 0..height {
        for x in 0..width {
            x_coords.push(x as f32 / width_denom);
            y_coords.push(y as f32 / height_denom);
        }
    }
    let device = lhs_mass.device();
    let x_coords =
        Tensor::<B, 2>::from_data(TensorData::new(x_coords, [1, height * width]), &device)
            .repeat_dim(0, batch);
    let y_coords =
        Tensor::<B, 2>::from_data(TensorData::new(y_coords, [1, height * width]), &device)
            .repeat_dim(0, batch);

    let lhs_denom = lhs_mass
        .clone()
        .sum_dim(1)
        .reshape([batch, 1])
        .add_scalar(1.0e-4);
    let rhs_denom = rhs_mass
        .clone()
        .sum_dim(1)
        .reshape([batch, 1])
        .add_scalar(1.0e-4);
    let lhs_center_x = lhs_mass
        .clone()
        .mul(x_coords.clone())
        .sum_dim(1)
        .reshape([batch, 1])
        .div(lhs_denom.clone());
    let lhs_center_y = lhs_mass
        .mul(y_coords.clone())
        .sum_dim(1)
        .reshape([batch, 1])
        .div(lhs_denom);
    let rhs_center_x = rhs_mass
        .clone()
        .mul(x_coords)
        .sum_dim(1)
        .reshape([batch, 1])
        .div(rhs_denom.clone());
    let rhs_center_y = rhs_mass
        .mul(y_coords)
        .sum_dim(1)
        .reshape([batch, 1])
        .div(rhs_denom);

    Tensor::cat(vec![lhs_center_x, lhs_center_y], 1)
        .sub(Tensor::cat(vec![rhs_center_x, rhs_center_y], 1))
        .powf_scalar(2.0)
        .mean()
        .reshape([1])
}

fn motion_mse_loss_frame<B: Backend>(
    lhs_current: Tensor<B, 4>,
    lhs_previous: Tensor<B, 4>,
    rhs_current: Tensor<B, 4>,
    rhs_previous: Tensor<B, 4>,
) -> Tensor<B, 1> {
    let lhs_delta = lhs_current - lhs_previous;
    let rhs_delta = rhs_current - rhs_previous;
    mse_loss_frame(lhs_delta, rhs_delta)
}

fn edge_mse_loss_frame<B: Backend>(lhs: Tensor<B, 4>, rhs: Tensor<B, 4>) -> Tensor<B, 1> {
    let [_batch, _channels, height, width] = lhs.shape().dims::<4>();
    if height < 2 || width < 2 {
        return mse_loss_frame(lhs, rhs);
    }
    let lhs_dx = lhs.clone().slice_dim(3, 1..width) - lhs.clone().slice_dim(3, 0..width - 1);
    let rhs_dx = rhs.clone().slice_dim(3, 1..width) - rhs.clone().slice_dim(3, 0..width - 1);
    let lhs_dy = lhs.clone().slice_dim(2, 1..height) - lhs.slice_dim(2, 0..height - 1);
    let rhs_dy = rhs.clone().slice_dim(2, 1..height) - rhs.slice_dim(2, 0..height - 1);
    let dx = (lhs_dx - rhs_dx).powf_scalar(2.0).mean();
    let dy = (lhs_dy - rhs_dy).powf_scalar(2.0).mean();
    (dx + dy).reshape([1])
}

fn summarize_fixation_set<B: Backend>(
    fixation_points: Tensor<B, 3>,
    fixation_stop: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let [batch, _, _] = fixation_points.shape().dims::<3>();
    let confidence = fixation_points
        .clone()
        .slice_dim(2, 3..4)
        .add_scalar(1.0e-4);
    let pooled = fixation_points
        .mul(confidence.clone())
        .sum_dim(1)
        .reshape([batch, 4]);
    let denom = confidence.sum_dim(1).reshape([batch, 1]).add_scalar(1.0e-6);
    Tensor::cat(vec![pooled / denom, fixation_stop], 1)
}

fn fixation_tensor_from_traces<B: Backend>(
    traces: &[FrameFixationTrace],
    step: usize,
    k: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut values = Vec::with_capacity(traces.len() * (k.max(1) * 4 + 1));
    for trace in traces {
        let frame = trace
            .frames
            .get(step)
            .or_else(|| trace.frames.last())
            .expect("trace has at least one frame");
        let mut sample = Vec::with_capacity(k.max(1) * 4 + 1);
        for point in frame.points.iter().take(k.max(1)) {
            sample.extend_from_slice(&[point.x, point.y, point.scale, point.confidence]);
        }
        while sample.len() < k.max(1) * 4 {
            sample.push(0.0);
        }
        sample.push(frame.stop_probability);
        values.extend(sample);
    }
    Tensor::<B, 2>::from_data(
        TensorData::new(values, [traces.len(), k.max(1) * 4 + 1]),
        device,
    )
}

fn extract_crops<B: Backend>(
    frame: Tensor<B, 4>,
    traces: &[FrameFixationTrace],
    step: usize,
    crop_size: usize,
    k: usize,
) -> Tensor<B, 5> {
    let [batch, channels, height, width] = frame.shape().dims::<4>();
    let mut patches = Vec::with_capacity(batch * k.max(1));
    let crop = crop_size.max(1);
    let half = crop / 2;
    for batch_idx in 0..batch {
        let sample = frame.clone().slice_dim(0, batch_idx..batch_idx + 1);
        let trace = traces
            .get(batch_idx)
            .expect("trace count matches batch size");
        let frame_trace = trace
            .frames
            .get(step)
            .or_else(|| trace.frames.last())
            .expect("trace has at least one frame");
        for point in frame_trace.points.iter().take(k.max(1)) {
            let center_x = (point.x * width as f32).round() as isize;
            let center_y = (point.y * height as f32).round() as isize;
            let x0 = center_x - half as isize;
            let y0 = center_y - half as isize;
            let x0 = x0.clamp(0, width.saturating_sub(crop) as isize) as usize;
            let y0 = y0.clamp(0, height.saturating_sub(crop) as isize) as usize;
            let patch = sample
                .clone()
                .slice_dim(2, y0..(y0 + crop).min(height))
                .slice_dim(3, x0..(x0 + crop).min(width));
            let [_, _, patch_h, patch_w] = patch.shape().dims::<4>();
            let patch = if patch_h == crop && patch_w == crop {
                patch
            } else {
                let mut padded = Tensor::<B, 4>::zeros([1, channels, crop, crop], &frame.device());
                padded = padded.slice_assign([0..1, 0..channels, 0..patch_h, 0..patch_w], patch);
                padded
            };
            patches.push(patch);
        }
    }
    Tensor::cat(patches, 0).reshape([batch, k.max(1), channels, crop, crop])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_autogaze::{FixationPoint, FixationSet};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn forward_emits_finite_losses() {
        let device = Default::default();
        let config = DreamerConfig::default();
        let model = DragonDreamer::<B>::new(config.clone(), &device);
        let clip = Tensor::<B, 5>::zeros(
            [2, 6, config.channels, config.frame_size, config.frame_size],
            &device,
        );
        let traces = vec![
            FrameFixationTrace::new(vec![
                FixationSet::new(
                    vec![FixationPoint::new(0.5, 0.5, 0.2, 1.0)],
                    0.0,
                    1,
                );
                6
            ]),
            FrameFixationTrace::new(vec![
                FixationSet::new(
                    vec![FixationPoint::new(0.4, 0.6, 0.2, 1.0)],
                    0.0,
                    1,
                );
                6
            ]),
        ];
        let teacher = Tensor::<B, 3>::zeros([2, 6, config.teacher_dim], &device);
        let crop_teacher = Tensor::<B, 3>::zeros([2, 6, config.crop_teacher_dim], &device);
        let out = model.forward(clip, &traces, teacher, crop_teacher, 4, 2);
        let total = out.total.into_data().to_vec::<f32>().expect("loss");
        assert!(total[0].is_finite());
    }

    #[test]
    fn world_write_merge_is_permutation_invariant() {
        let device = Default::default();
        let config = DreamerConfig {
            k_fovea: 2,
            ..Default::default()
        };
        let model = DragonDreamer::<B>::new(config, &device);
        let token_values: Vec<f32> = (0..(2 * model.fovea_dim))
            .map(|idx| (idx as f32 * 0.013).sin())
            .collect();
        let tokens = Tensor::<B, 3>::from_data(
            TensorData::new(token_values, [1, 2, model.fovea_dim]),
            &device,
        );
        let points = Tensor::<B, 3>::from_data(
            TensorData::new(vec![0.2, 0.3, 0.15, 0.9, 0.8, 0.7, 0.25, 0.6], [1, 2, 4]),
            &device,
        );
        let tokens_swapped = tokens.clone().slice_dim(1, 1..2);
        let tokens_swapped =
            Tensor::cat(vec![tokens_swapped, tokens.clone().slice_dim(1, 0..1)], 1);
        let points_swapped = points.clone().slice_dim(1, 1..2);
        let points_swapped =
            Tensor::cat(vec![points_swapped, points.clone().slice_dim(1, 0..1)], 1);
        let merged_a = model.merge_world_writes(tokens, points);
        let merged_b = model.merge_world_writes(tokens_swapped, points_swapped);
        let diff = (merged_a - merged_b)
            .abs()
            .mean()
            .into_data()
            .to_vec::<f32>()
            .expect("diff")[0];
        assert!(
            diff < 1.0e-5,
            "expected order-invariant merge, got diff {diff}"
        );
    }

    #[test]
    fn bdh_posterior_forward_emits_finite_losses() {
        let device = Default::default();
        let config = DreamerConfig {
            use_bdh_posterior: true,
            k_fovea: 2,
            ..Default::default()
        };
        let model = DragonDreamer::<B>::new(config.clone(), &device);
        let clip = Tensor::<B, 5>::zeros(
            [2, 6, config.channels, config.frame_size, config.frame_size],
            &device,
        );
        let traces = vec![
            FrameFixationTrace::new(vec![
                FixationSet::new(
                    vec![
                        FixationPoint::new(0.5, 0.5, 0.2, 1.0),
                        FixationPoint::new(0.4, 0.6, 0.25, 0.7),
                    ],
                    0.0,
                    2,
                );
                6
            ]),
            FrameFixationTrace::new(vec![
                FixationSet::new(
                    vec![
                        FixationPoint::new(0.4, 0.6, 0.2, 1.0),
                        FixationPoint::new(0.6, 0.4, 0.25, 0.7),
                    ],
                    0.0,
                    2,
                );
                6
            ]),
        ];
        let teacher = Tensor::<B, 3>::zeros([2, 6, config.teacher_dim], &device);
        let crop_teacher = Tensor::<B, 3>::zeros([2, 6, config.crop_teacher_dim], &device);
        let out = model.forward(clip, &traces, teacher, crop_teacher, 4, 2);
        let total = out.total.into_data().to_vec::<f32>().expect("loss");
        assert!(total[0].is_finite());
    }

    #[test]
    fn transformer_baseline_forward_emits_finite_losses() {
        let device = Default::default();
        let config = DreamerConfig {
            latent_backend: crate::DreamerLatentBackend::TransformerBaseline,
            k_fovea: 2,
            ..Default::default()
        };
        let model = DragonDreamer::<B>::new(config.clone(), &device);
        let clip = Tensor::<B, 5>::zeros(
            [2, 6, config.channels, config.frame_size, config.frame_size],
            &device,
        );
        let traces = vec![
            FrameFixationTrace::new(vec![
                FixationSet::new(
                    vec![
                        FixationPoint::new(0.5, 0.5, 0.2, 1.0),
                        FixationPoint::new(0.4, 0.6, 0.25, 0.7),
                    ],
                    0.0,
                    2,
                );
                6
            ]),
            FrameFixationTrace::new(vec![
                FixationSet::new(
                    vec![
                        FixationPoint::new(0.4, 0.6, 0.2, 1.0),
                        FixationPoint::new(0.6, 0.4, 0.25, 0.7),
                    ],
                    0.0,
                    2,
                );
                6
            ]),
        ];
        let teacher = Tensor::<B, 3>::zeros([2, 6, config.teacher_dim], &device);
        let crop_teacher = Tensor::<B, 3>::zeros([2, 6, config.crop_teacher_dim], &device);
        let out = model.forward(clip, &traces, teacher, crop_teacher, 4, 2);
        let total = out.total.into_data().to_vec::<f32>().expect("loss");
        assert!(total[0].is_finite());
    }

    #[test]
    fn bdh_challenger_forward_emits_finite_losses() {
        let device = Default::default();
        let config = DreamerConfig {
            latent_backend: crate::DreamerLatentBackend::BdhChallenger,
            use_bdh_posterior: true,
            k_fovea: 2,
            ..Default::default()
        };
        let model = DragonDreamer::<B>::new(config.clone(), &device);
        let clip = Tensor::<B, 5>::zeros(
            [2, 6, config.channels, config.frame_size, config.frame_size],
            &device,
        );
        let traces = vec![
            FrameFixationTrace::new(vec![
                FixationSet::new(
                    vec![
                        FixationPoint::new(0.5, 0.5, 0.2, 1.0),
                        FixationPoint::new(0.4, 0.6, 0.25, 0.7),
                    ],
                    0.0,
                    2,
                );
                6
            ]),
            FrameFixationTrace::new(vec![
                FixationSet::new(
                    vec![
                        FixationPoint::new(0.4, 0.6, 0.2, 1.0),
                        FixationPoint::new(0.6, 0.4, 0.25, 0.7),
                    ],
                    0.0,
                    2,
                );
                6
            ]),
        ];
        let teacher = Tensor::<B, 3>::zeros([2, 6, config.teacher_dim], &device);
        let crop_teacher = Tensor::<B, 3>::zeros([2, 6, config.crop_teacher_dim], &device);
        let out = model.forward(clip, &traces, teacher, crop_teacher, 4, 2);
        let total = out.total.into_data().to_vec::<f32>().expect("loss");
        assert!(total[0].is_finite());
    }

    #[test]
    fn forward_with_debug_returns_reconstruction_and_fixation_shapes() {
        let device = Default::default();
        let config = DreamerConfig {
            k_fovea: 2,
            ..Default::default()
        };
        let model = DragonDreamer::<B>::new(config.clone(), &device);
        let clip = Tensor::<B, 5>::zeros(
            [2, 6, config.channels, config.frame_size, config.frame_size],
            &device,
        );
        let traces = vec![
            FrameFixationTrace::new(vec![
                FixationSet::new(
                    vec![
                        FixationPoint::new(0.5, 0.5, 0.2, 1.0),
                        FixationPoint::new(0.4, 0.6, 0.25, 0.7),
                    ],
                    0.0,
                    2,
                );
                6
            ]),
            FrameFixationTrace::new(vec![
                FixationSet::new(
                    vec![
                        FixationPoint::new(0.4, 0.6, 0.2, 1.0),
                        FixationPoint::new(0.6, 0.4, 0.25, 0.7),
                    ],
                    0.0,
                    2,
                );
                6
            ]),
        ];
        let teacher = Tensor::<B, 3>::zeros([2, 6, config.teacher_dim], &device);
        let crop_teacher = Tensor::<B, 3>::zeros([2, 6, config.crop_teacher_dim], &device);
        let (forward, debug) = model.forward_with_debug(clip, &traces, teacher, crop_teacher, 4, 2);
        assert!(forward.recon.into_data().to_vec::<f32>().expect("recon")[0].is_finite());
        assert_eq!(
            debug.context_reconstruction_frames.shape().dims::<5>(),
            [2, 4, config.channels, config.frame_size, config.frame_size]
        );
        assert_eq!(
            debug.future_reconstruction_frames.shape().dims::<5>(),
            [2, 2, config.channels, config.frame_size, config.frame_size]
        );
        assert_eq!(
            debug.predicted_fixations.shape().dims::<3>(),
            [2, 6, config.k_fovea * 4 + 1]
        );
    }
}
