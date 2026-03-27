use super::DragonDreamer;
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_dragon_core::MicroTransformerBlock;

impl<B: Backend> DragonDreamer<B> {
    pub(super) fn predict_fixation_from_slots(
        &self,
        slots: Tensor<B, 3>,
        peripheral: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 2> {
        let [batch, slot_count, dim] = slots.shape().dims::<3>();
        let slot_tokens = slots.clone() + self.slot_position_embeddings(batch, &slots.device());
        let slot_logits = self
            .transformer_fixation_slot_logits
            .forward(slot_tokens.reshape([batch * slot_count, dim]))
            .reshape([batch, slot_count, self.k_fovea]);
        let attention = activation::softmax(slot_logits.swap_dims(1, 2), 2);
        let coordinates = self.slot_coordinates(batch, &slots.device());
        let xy = attention.clone().matmul(coordinates);
        let attended_slots = attention.matmul(slots.clone());
        let params_hidden = activation::gelu(
            self.transformer_fixation_param_hidden
                .forward(attended_slots.reshape([batch * self.k_fovea, self.latent_dim])),
        );
        let params = self
            .transformer_fixation_param_out
            .forward(params_hidden)
            .reshape([batch, self.k_fovea, 2]);
        let scale = activation::sigmoid(params.clone().slice_dim(2, 0..1))
            .mul_scalar(0.9)
            .add_scalar(0.05);
        let confidence = activation::sigmoid(params.slice_dim(2, 1..2));
        let points = Tensor::cat(vec![xy, scale, confidence], 2).reshape([batch, self.k_fovea * 4]);
        let summary = self.summarize_slots(slots);
        let batch = summary.shape().dims::<2>()[0];
        let peripheral = peripheral.unwrap_or_else(|| {
            Tensor::<B, 2>::zeros([batch, self.peripheral_dim], &summary.device())
        });
        let hidden = activation::gelu(
            self.fixation_hidden
                .forward(Tensor::cat(vec![peripheral, summary], 1)),
        );
        let stop = activation::sigmoid(self.fixation_stop.forward(hidden));
        Tensor::cat(vec![points, stop], 1)
    }

    pub(super) fn scatter_world_writes_to_slots(
        &self,
        fovea_tokens: Tensor<B, 3>,
        fixation_points: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let device = fovea_tokens.device();
        let projected_writes = self.world_write_tokens(fovea_tokens, fixation_points.clone());
        let [batch, k, _] = projected_writes.shape().dims::<3>();
        let slot_positions = self
            .slot_coordinates(1, &device)
            .reshape([1, self.slot_count, 2]);
        let fix_xy = fixation_points.clone().slice_dim(2, 0..2);
        let fix_scale = fixation_points.clone().slice_dim(2, 2..3);
        let confidence = fixation_points.slice_dim(2, 3..4).add_scalar(1.0e-4);
        let slot_xy = slot_positions
            .clone()
            .unsqueeze_dim::<4>(1)
            .repeat_dim(0, batch);
        let fix_xy = fix_xy.unsqueeze_dim::<4>(2);
        let diff = fix_xy - slot_xy;
        let dist_sq = diff
            .powf_scalar(2.0)
            .sum_dim(3)
            .reshape([batch, k, self.slot_count]);
        let sigma = fix_scale
            .mul_scalar(0.5)
            .add_scalar(1.0 / self.slot_grid_size.max(1) as f32)
            .reshape([batch, k, 1]);
        let sigma_sq = sigma.powf_scalar(2.0).add_scalar(1.0e-5);
        let weights = dist_sq
            .mul_scalar(-1.0)
            .div(sigma_sq.mul_scalar(2.0))
            .exp()
            .mul(confidence)
            .reshape([batch, k, self.slot_count, 1]);
        let writes = projected_writes
            .unsqueeze_dim::<4>(2)
            .repeat_dim(2, self.slot_count)
            .mul(weights.clone())
            .sum_dim(1)
            .reshape([batch, self.slot_count, self.latent_dim]);
        let denom = weights
            .sum_dim(1)
            .reshape([batch, self.slot_count, 1])
            .add_scalar(1.0e-6);
        writes / denom
    }

    pub(super) fn zero_slot_state(&self, batch: usize, device: &B::Device) -> Tensor<B, 3> {
        Tensor::<B, 3>::zeros([batch, self.slot_count, self.latent_dim], device)
    }

    pub(super) fn summarize_slots(&self, slots: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, slot_count, dim] = slots.shape().dims::<3>();
        let mean_summary = slots.clone().mean_dim(1).reshape([batch, dim]);
        let flat_summary = activation::gelu(
            self.transformer_summary_in
                .forward(slots.reshape([batch, slot_count * dim])),
        );
        activation::gelu(
            self.transformer_summary_out
                .forward(Tensor::cat(vec![mean_summary, flat_summary], 1)),
        )
    }

    pub(super) fn slot_position_embeddings(
        &self,
        batch: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let coord_tensor = self.slot_coordinates(batch, device);
        let hidden = activation::gelu(self.transformer_slot_pos_in.forward(coord_tensor));
        activation::gelu(self.transformer_slot_pos_out.forward(hidden))
    }

    pub(super) fn slot_coordinates(&self, batch: usize, device: &B::Device) -> Tensor<B, 3> {
        let mut coords = Vec::with_capacity(self.slot_count * 2);
        let denom = (self.slot_grid_size.saturating_sub(1)).max(1) as f32;
        for y in 0..self.slot_grid_size {
            for x in 0..self.slot_grid_size {
                coords.push(x as f32 / denom);
                coords.push(y as f32 / denom);
            }
        }
        Tensor::<B, 2>::from_data(TensorData::new(coords, [self.slot_count, 2]), device)
            .unsqueeze_dim::<3>(0)
            .repeat_dim(0, batch)
    }

    pub(super) fn encode_frame_to_slot_tokens(&self, frame: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, channels, height, width] = frame.shape().dims::<4>();
        debug_assert_eq!(height, self.frame_size);
        debug_assert_eq!(width, self.frame_size);
        let patches = frame
            .reshape([
                batch,
                channels,
                self.slot_grid_size,
                self.slot_patch_size,
                self.slot_grid_size,
                self.slot_patch_size,
            ])
            .swap_dims(3, 4)
            .swap_dims(2, 3)
            .swap_dims(1, 3)
            .reshape([
                batch,
                self.slot_count,
                channels * self.slot_patch_size * self.slot_patch_size,
            ]);
        let hidden = activation::gelu(self.transformer_tokenizer_in.forward(patches));
        activation::gelu(self.transformer_tokenizer_hidden.forward(hidden))
    }

    pub(super) fn apply_slot_blocks(
        &self,
        tokens: Tensor<B, 3>,
        blocks: &[MicroTransformerBlock<B>],
    ) -> Tensor<B, 3> {
        let mut hidden = tokens;
        for block in blocks {
            hidden = block.forward(hidden);
        }
        hidden
    }

    pub(super) fn prior_step_slots(&self, slots: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, slot_count, _] = slots.shape().dims::<3>();
        let positioned = slots.clone() + self.slot_position_embeddings(batch, &slots.device());
        let updated = self.apply_slot_blocks(positioned, &self.transformer_prior_blocks);
        let updated_slots = updated.slice_dim(1, 0..slot_count);
        let gate = activation::sigmoid(
            self.transformer_prior_gate.forward(
                Tensor::cat(vec![slots.clone(), updated_slots.clone()], 2)
                    .reshape([batch * slot_count, self.latent_dim * 2]),
            ),
        )
        .reshape([batch, slot_count, self.latent_dim]);
        slots.clone() + gate * (updated_slots - slots)
    }

    pub(super) fn posterior_step_slots(
        &self,
        prior_slots: Tensor<B, 3>,
        observed_slots: Tensor<B, 3>,
        peripheral: Tensor<B, 2>,
        fovea_tokens: Tensor<B, 3>,
        fixation_points: Tensor<B, 3>,
        fixation_summary: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let [batch, slot_count, _] = prior_slots.shape().dims::<3>();
        let sparse_writes = self.scatter_world_writes_to_slots(fovea_tokens, fixation_points);
        let observed_slots = observed_slots + sparse_writes;
        let slot_tokens =
            prior_slots.clone() + self.slot_position_embeddings(batch, &prior_slots.device());
        let observation_gate = activation::sigmoid(
            self.transformer_observation_gate.forward(
                Tensor::cat(vec![slot_tokens.clone(), observed_slots.clone()], 2)
                    .reshape([batch * slot_count, self.latent_dim * 2]),
            ),
        )
        .reshape([batch, slot_count, self.latent_dim]);
        let slot_tokens = slot_tokens.clone() + observation_gate * (observed_slots - slot_tokens);
        let peripheral_token =
            activation::gelu(self.transformer_peripheral_token.forward(peripheral))
                .unsqueeze_dim::<3>(1);
        let fixation_token =
            activation::gelu(self.transformer_fixation_token.forward(fixation_summary))
                .unsqueeze_dim::<3>(1);
        let fused = Tensor::cat(vec![slot_tokens, peripheral_token, fixation_token], 1);
        let updated = self.apply_slot_blocks(fused, &self.transformer_posterior_blocks);
        let updated_slots = updated.slice_dim(1, 0..slot_count);
        let gate = activation::sigmoid(
            self.transformer_posterior_gate.forward(
                Tensor::cat(vec![prior_slots.clone(), updated_slots.clone()], 2)
                    .reshape([batch * slot_count, self.latent_dim * 2]),
            ),
        )
        .reshape([batch, slot_count, self.latent_dim]);
        prior_slots.clone() + gate * (updated_slots - prior_slots)
    }

    pub(super) fn decode_frame_from_slots_components(
        &self,
        slots: Tensor<B, 3>,
        _previous_frame: Option<Tensor<B, 4>>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, _slot_count, _] = slots.shape().dims::<3>();
        let slot_grid = slots
            .reshape([
                batch,
                self.slot_grid_size,
                self.slot_grid_size,
                self.latent_dim,
            ])
            .swap_dims(1, 3)
            .swap_dims(2, 3);
        let base_hidden = activation::gelu(self.transformer_grid_in.forward(slot_grid));
        let base_hidden = activation::gelu(self.transformer_grid_up.forward(base_hidden));
        let base_hidden = activation::gelu(self.transformer_grid_hidden.forward(base_hidden));
        let base_intensity =
            activation::tanh(self.transformer_grid_out.forward(base_hidden.clone()));
        let base_occupancy =
            activation::sigmoid(self.transformer_grid_occ_out.forward(base_hidden));
        let background =
            Tensor::<B, 4>::ones(base_intensity.shape(), &base_intensity.device()).mul_scalar(-1.0);
        let base_frame = base_occupancy.clone().mul(base_intensity)
            + background.mul(base_occupancy.clone().mul_scalar(-1.0).add_scalar(1.0));
        let refine_input = Tensor::cat(
            vec![
                base_frame.clone(),
                base_occupancy.clone().mul_scalar(2.0).add_scalar(-1.0),
            ],
            1,
        );
        let refined = activation::gelu(self.recon_refine_in.forward(refine_input));
        let refined = activation::gelu(self.recon_refine_hidden.forward(refined));
        let refined = activation::tanh(self.recon_refine_out.forward(refined));
        (
            activation::tanh(base_frame + refined.mul_scalar(0.5)),
            base_occupancy,
        )
    }
}
