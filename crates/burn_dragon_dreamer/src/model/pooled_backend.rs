use super::DragonDreamer;
use burn::tensor::Tensor;
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn_dragon_core::ModelState;

impl<B: Backend> DragonDreamer<B> {
    pub(super) fn predict_fixation_from_state(
        &self,
        state: Tensor<B, 2>,
        peripheral: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 2> {
        let batch = state.shape().dims::<2>()[0];
        let peripheral = peripheral.unwrap_or_else(|| {
            Tensor::<B, 2>::zeros([batch, self.peripheral_dim], &state.device())
        });
        let hidden = activation::gelu(
            self.fixation_hidden
                .forward(Tensor::cat(vec![peripheral, state], 1)),
        );
        let points = activation::sigmoid(self.fixation_out.forward(hidden.clone()));
        let stop = activation::sigmoid(self.fixation_stop.forward(hidden));
        Tensor::cat(vec![points, stop], 1)
    }

    pub(super) fn predict_fixation(
        &self,
        peripheral: Tensor<B, 2>,
        post: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        self.predict_fixation_from_state(post, Some(peripheral))
    }

    pub(super) fn merge_observation(
        &self,
        peripheral: Tensor<B, 2>,
        world_summary: Tensor<B, 2>,
        fixation_summary: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let hidden = activation::gelu(self.observation_hidden.forward(Tensor::cat(
            vec![peripheral, world_summary, fixation_summary],
            1,
        )));
        activation::gelu(self.observation_out.forward(hidden))
    }

    pub(super) fn merge_world_writes(
        &self,
        fovea_tokens: Tensor<B, 3>,
        fixation_points: Tensor<B, 3>,
    ) -> Tensor<B, 2> {
        let [batch, k, _] = fovea_tokens.shape().dims::<3>();
        let write_inputs = Tensor::cat(vec![fovea_tokens, fixation_points.clone()], 2)
            .reshape([batch * k, self.fovea_dim + 4]);
        let hidden = activation::gelu(self.world_write_hidden.forward(write_inputs));
        let content = activation::gelu(self.world_write_out.forward(hidden.clone())).reshape([
            batch,
            k,
            self.latent_dim,
        ]);
        let gate =
            activation::sigmoid(self.world_write_gate.forward(hidden)).reshape([batch, k, 1]);
        let confidence = fixation_points.slice_dim(2, 3..4);
        let weights = confidence.mul(gate).add_scalar(1.0e-4);
        let merged = content
            .mul(weights.clone())
            .sum_dim(1)
            .reshape([batch, self.latent_dim]);
        let denom = weights.sum_dim(1).reshape([batch, 1]).add_scalar(1.0e-6);
        merged / denom
    }

    pub(super) fn world_write_tokens(
        &self,
        fovea_tokens: Tensor<B, 3>,
        fixation_points: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [batch, k, _] = fovea_tokens.shape().dims::<3>();
        let write_inputs = Tensor::cat(vec![fovea_tokens, fixation_points.clone()], 2)
            .reshape([batch * k, self.fovea_dim + 4]);
        let hidden = activation::gelu(self.world_write_hidden.forward(write_inputs));
        let content = activation::gelu(self.world_write_out.forward(hidden.clone())).reshape([
            batch,
            k,
            self.latent_dim,
        ]);
        let gate =
            activation::sigmoid(self.world_write_gate.forward(hidden)).reshape([batch, k, 1]);
        let confidence = fixation_points.slice_dim(2, 3..4);
        content.mul(confidence.mul(gate).add_scalar(1.0e-4))
    }

    pub(super) fn prior_step(&self, post: Tensor<B, 2>) -> Tensor<B, 2> {
        activation::gelu(
            self.prior_out
                .forward(activation::gelu(self.prior_hidden.forward(post))),
        )
    }

    pub(super) fn decode_frame(
        &self,
        latent: Tensor<B, 2>,
        previous_frame: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 4> {
        let batch = latent.shape().dims::<2>()[0];
        let frame_dim = self.channels * self.frame_size * self.frame_size;
        let previous_frame_for_refine = previous_frame.clone();
        let base_hidden = activation::gelu(self.recon_hidden.forward(latent.clone()));
        let base_hidden = activation::gelu(self.recon_hidden2.forward(base_hidden));
        let base = activation::tanh(self.recon_out.forward(base_hidden.clone()));
        let decoded = if let Some(previous_frame) = previous_frame {
            let previous_flat = previous_frame.reshape([batch, frame_dim]);
            let previous_embed =
                activation::gelu(self.recon_prev_in.forward(previous_flat.clone()));
            let fused = activation::gelu(
                self.recon_fused_hidden
                    .forward(Tensor::cat(vec![latent, previous_embed], 1)),
            );
            let delta = activation::tanh(self.recon_delta_out.forward(fused.clone()));
            let blend = activation::sigmoid(self.recon_blend_out.forward(fused));
            let carry = blend.clone().mul_scalar(-1.0).add_scalar(1.0);
            let motion = activation::tanh(previous_flat + delta.mul_scalar(0.5));
            activation::tanh(base * blend + motion * carry)
        } else {
            base
        };
        let base_frame = decoded.reshape([batch, self.channels, self.frame_size, self.frame_size]);
        let refine_skip = previous_frame_for_refine.unwrap_or_else(|| base_frame.clone());
        let refine_input = Tensor::cat(vec![base_frame.clone(), refine_skip], 1);
        let refined = activation::gelu(self.recon_refine_in.forward(refine_input));
        let refined = activation::gelu(self.recon_refine_hidden.forward(refined));
        let refined = activation::tanh(self.recon_refine_out.forward(refined));
        activation::tanh(base_frame + refined.mul_scalar(0.5))
    }

    pub(super) fn posterior_step(
        &self,
        prior: Tensor<B, 2>,
        obs: Tensor<B, 2>,
        bdh_state: Option<&mut ModelState<B>>,
    ) -> Tensor<B, 2> {
        if self.use_bdh_posterior {
            let fused = Tensor::cat(vec![prior, obs], 1);
            let embedded =
                activation::gelu(self.bdh_input_proj.forward(fused)).unsqueeze_dim::<3>(1);
            let Some(state) = bdh_state else {
                unreachable!("bdh posterior requested without state");
            };
            let (hidden, _logits) = self
                .bdh
                .forward_with_hidden_and_state_embedded(embedded, state);
            let batch = hidden.shape().dims::<3>()[0];
            return hidden.reshape([batch, self.latent_dim]);
        }
        let gate = activation::sigmoid(
            self.posterior_gate_prior.forward(prior.clone())
                + self.posterior_gate_obs.forward(obs.clone()),
        );
        prior.clone() + gate * (obs - prior)
    }
}
