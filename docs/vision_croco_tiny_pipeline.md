# Croco Tiny Training Pipeline and Model Architecture (Code-Referenced)

This document describes the current croco tiny config pipeline using only code references and the actual flow in the repo.
Config file: config/vision/croco/tiny.toml
Note: when running via the CLI, config/vision/base.toml is prepended in crates/burn_dragon_vision/src/train/cli.rs.

## 1) Config load, merge, and validation
- Config merge order: burn_dragon_vision::load_vision_training_config merges TOML overlays in order; later files override earlier files. See crates/burn_dragon_vision/src/config/vision.rs (load_vision_training_config).
- Core config types:
  - VisionTrainingConfig, VisionTrainingHyperparameters, VisionModelConfig, VisionSaccadeConfig, VisionSaccadeCrossViewConfig, VisionManifoldHyperConnectionsConfig are all in crates/burn_dragon_vision/src/config/vision.rs.
- Vision model config -> core model config:
  - VisionModelConfig::build creates VisionDragonConfig (crates/burn_dragon_vision/src/config/vision.rs, impl VisionModelConfig::build).
  - This is where vision.mhc is translated into core ManifoldHyperConnectionsConfig.
- Rollout validation is performed in validate_vision_rollout (crates/burn_dragon_vision/src/config/vision.rs).

## 2) Training entrypoint and dataset construction
- Training entrypoint for vision is train_vision_backend in crates/burn_dragon_vision/src/train/vision/train.rs.
- Dataset, augmentation, and dataloaders:
  - ImageNetAugmentations and ImageNetDataset are set up in train_vision_backend (crates/burn_dragon_vision/src/train/vision/train.rs).
  - Cross-view behavior is determined before dataset construction. If saccade.cross_view.enabled:
    - views = saccade.num_eyes
    - min_view_overlap = saccade.cross_view.min_overlap
    - view_overlap_attempts = saccade.cross_view.max_attempts
    (same function: train_vision_backend in crates/burn_dragon_vision/src/train/vision/train.rs).
  - ImageNetDataLoader is used for train/valid in train_vision_backend.

## 3) Training schedule and learner
- The training schedule is resolved in resolve_vision_train_schedule and resolve_vision_lr_scheduler (crates/burn_dragon_vision/src/train/vision/train.rs).
- The training loop is created in train_vision_with_scheduler (crates/burn_dragon_vision/src/train/pipeline/schedule.rs) and uses burn_train::LearnerBuilder.
- Metrics and device memory tracking are wired in train_vision_with_scheduler (same file) using:
  - ScalarMetric, LossMetric, LearningRateMetric, DeviceMetric
  - MemoryCleanupMetric and DeviceMemoryMetric when enabled by config.

## 4) Core vision model (VisionDragon)
Source: crates/burn_dragon_vision/src/model/vision.rs

### 4.1 Patch embedding
- PatchEmbed::new handles patch_embed_mode = conv | linear | identity in PatchEmbed (crates/burn_dragon_core/src/model/vision.rs).
- For patch_embed_mode = "conv", a stack of PatchEmbedStage + PatchConvNeXtBlock is used (same file) and then a 1x1 projection.
- For patch_embed_mode = "linear", patchify + Linear projection is used (patchify is also in the same file).
- Positional encoding is applied via SpatialPositionalEncoding::add_position in PatchEmbed::forward (crates/burn_dragon_core/src/model/vision.rs).

### 4.2 Residual stream and attention
- The core recurrent steps are implemented in encode_tokens_steps_inner_multi (crates/burn_dragon_core/src/model/vision.rs).
- Each step uses a low-rank latent projection defined by encoder / encoder_v / decoder parameters in VisionDragon.
- Attention is implemented in full_attention (crates/burn_dragon_core/src/model/vision.rs) with:
  - ALiBi bias when use_alibi is true.
  - VisionAttentionMode::Softmax or VisionAttentionMode::RowL1 normalization.
- Token normalization is optional (token_state_norm -> LayerNorm) applied in apply_token_norm (same file).

### 4.3 Multi-eye and cross-eye mixing
- If num_eyes > 1, eye_token is created in VisionDragon::new and added in encode_tokens_steps_inner_multi (crates/burn_dragon_vision/src/model/vision.rs).
- cls token sync is done in sync_cls_tokens_multi using cls_sync_alpha.
- After the main step loop, cross-eye mixing is optionally applied when cross_eye_steps > 0 via encode_tokens_steps_inner and then reshaping back (encode_tokens_steps_inner_multi).

## 5) Manifold-constrained hyper-connections (mHC)
Source: crates/burn_dragon_core/src/model/residual.rs

- Config: ManifoldHyperConnectionsConfig (fields: enabled, num_streams, num_views, mhc_iters, mhc_tau, add_branch_out_to_residual, dropout).
- Width connection:
  - width_connection computes sinkhorn-normalized residual mixing (h_res) and a per-view pre-mix (h_pre).
  - This yields (branch_input, residuals_out, beta).
- Depth connection:
  - depth_connection optionally mixes branch_output back into residuals using beta and Dropout.
- Integration into VisionDragon:
  - encode_tokens_steps_inner_multi uses mhc.width_connection before the step core and mhc.depth_connection after it (crates/burn_dragon_vision/src/model/vision.rs).

## 6) Saccade model used by croco tiny
Sources:
- Struct definition: crates/burn_dragon_vision/src/train/saccade/structs.rs (VisionSaccadeModel fields).
- Construction: crates/burn_dragon_vision/src/train/saccade/impls/core.rs (VisionSaccadeModel::new).

Key modules instantiated in VisionSaccadeModel::new:
- VisionReconstructionHead: crates/burn_dragon_vision/src/train/vision/models.rs (used for patch recon).
- trajectory_token: learned initial recurrent state for rollout.
- eye_token: per-eye identity bias (always allocated; random if num_eyes > 1).
- view_embed: Linear(4 -> embed_dim) only when cross_view.enabled and num_eyes > 1.
- input_proj: VisionSaccadeInputProjection (crates/burn_dragon_vision/src/train/vision/models.rs).
- fovea_proj, pyramid_in_proj, pyramid_out_proj, residual_proj: VisionSaccadeProjection (same file).
- saccade_head: VisionSaccadeHead for predicting [mean_x, mean_y, sigma].

## 7) Saccade forward + loss path
Sources:
- TrainStep impl: crates/burn_dragon_vision/src/train/saccade/sampler.rs
- Forward/loss path: crates/burn_dragon_vision/src/train/saccade/impls/core.rs

### 7.1 TrainStep wiring
- TrainStep::step for VisionSaccadeModel calls forward_losses_train (saccade/sampler.rs), which delegates to forward_losses_with_policy (saccade/impls/core.rs).
- Rollout steps are sampled using VisionRollout::sample_steps (crates/burn_dragon_vision/src/train/pipeline/schedule.rs).

### 7.2 Loss assembly and GDPO
- forward_losses_with_policy:
  - Pulls images and view_images from ImageNetBatch.
  - Expands batch if GDPO group_size > 1 (gdpo enabled).
  - Calls recon_loss (saccade/impls/core.rs).
  - Computes recon loss + optional LeJEPA losses + optional policy loss.
  - Policy loss uses GDPO utilities in crates/burn_dragon_vision/src/train/gdpo.rs via build_gdpo_policy_loss.

### 7.3 Cross-view recon path
- recon_loss (saccade/impls/core.rs) is the core cross-view logic:
  - Determines cross_view based on config.cross_view.enabled and view_images presence.
  - Splits view_images per eye; fills missing views with the primary image.
  - masked_eye determines which eye is masked for recon.
  - build_masked_mip_pyramid is used for masked view; build_mip_pyramid for unmasked views.
  - Target patches, loss masks, and patch grids are built per-eye and per-level.

## 8) Foveation and pyramid sampling
Sources:
- Foveation sampling: crates/burn_dragon_vision/src/train/saccade/impls/foveation.rs
- Pyramid handling: crates/burn_dragon_vision/src/train/saccade/impls/pyramid.rs
- Scatter/gather: crates/burn_dragon_vision/src/train/saccade/impls/scatter.rs

Key operations:
- build_mip_pyramid and build_masked_mip_pyramid create per-level tokens + images.
- foveated_patch_image and foveated_patch_image_with_radius use fovea sampling based on config.fovea_sampling_mode and config.fovea_warp_mode.
- scatter/gather functions update pyramid states and handle upsample/merge logic across levels.

## 9) Saccade policy and location embedding
Source: crates/burn_dragon_vision/src/train/saccade/policy.rs

- build_input_tokens combines input projection, state context, and fovea embedding.
- sample_policy_action applies Gaussian noise (action_noise_std) to mean/sigma and outputs log_prob + clamp rate.
- Location embedding modes (none/learned/sinusoidal/quantized/rope/pope) are implemented in fovea_embed, fixed_location_embedding, and rotary_location_embedding.

## 10) Reconstruction head
Source: crates/burn_dragon_vision/src/train/vision/models.rs

- VisionReconstructionHead is a LayerNorm + (optional) 2-layer MLP + Linear to patch-dim.
- Used by VisionSaccadeModel in saccade/impls/core.rs (recon_loss path).

## 11) Croco tiny specific config mapping
config/vision/croco/tiny.toml maps to the following code paths:
- [vision] -> VisionModelConfig::build -> VisionDragonConfig (crates/burn_dragon_vision/src/config/vision.rs).
- [vision.mhc] -> VisionManifoldHyperConnectionsConfig -> ManifoldHyperConnectionsConfig (same file + crates/burn_dragon_core/src/model/residual.rs).
- [mode] type = "saccade" -> VisionSaccadeConfig (crates/burn_dragon_vision/src/config/vision.rs) -> VisionSaccadeModel (crates/burn_dragon_vision/src/train/saccade/structs.rs).
- [mode.cross_view] -> VisionSaccadeCrossViewConfig (crates/burn_dragon_vision/src/config/vision.rs) and cross-view logic in recon_loss (crates/burn_dragon_vision/src/train/saccade/impls/core.rs).
- [mode.input_projection] -> VisionSaccadeInputProjection (crates/burn_dragon_vision/src/train/vision/models.rs).
- [mode.policy.*] -> policy.rs (location embedding, GDPO inputs, action sampling).
- [mode.loss.recon] -> recon_loss in saccade/impls/core.rs and VisionReconstructionHead in vision/models.rs.

## 12) Where to modify if you want design changes
- Patch embedding, attention, and recurrence: crates/burn_dragon_core/src/model/vision.rs.
- mHC details: crates/burn_dragon_core/src/model/residual.rs.
- Saccade rollout / cross-view logic: crates/burn_dragon_vision/src/train/saccade/impls/core.rs.
- Foveation details: crates/burn_dragon_vision/src/train/saccade/impls/foveation.rs and scatter.rs.
- Input projection: crates/burn_dragon_vision/src/train/vision/models.rs (VisionSaccadeInputProjection).
- Config schema: crates/burn_dragon_vision/src/config/vision.rs.
- Training schedule + metrics: crates/burn_dragon_vision/src/train/pipeline/schedule.rs.

