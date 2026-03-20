use crate::train::prelude::*;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};

mod distill;
mod input_projection;

pub(crate) use distill::{DistillTeacherModel, VisionDistillModel};
pub(crate) use input_projection::VisionSaccadeInputProjection;

type ReconArtifacts<B> = Option<(Vec<Tensor<B, 4>>, Tensor<B, 3>)>;
type ReconLossOutput<B> = (
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    ReconArtifacts<B>,
);
type ReconCrossViewOutput<B> = (
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    ReconArtifacts<B>,
);

#[derive(Clone, Copy, Debug)]
pub(crate) struct VisionReconstructionInit {
    pub(crate) patch_dim: usize,
    pub(crate) normalize_std: [f32; 3],
    pub(crate) patch_size: usize,
    pub(crate) in_channels: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct VisionLejepaInit {
    pub(crate) embed_dim: usize,
    pub(crate) num_classes: usize,
    pub(crate) rollout: VisionRollout,
    pub(crate) recon: VisionReconstructionInit,
    pub(crate) normalization: DragonNormConfig,
}

#[derive(Clone, Debug)]
pub(crate) struct VisionMaeInit {
    pub(crate) num_eyes: usize,
    pub(crate) embed_dim: usize,
    pub(crate) rollout: VisionRollout,
    pub(crate) recon: VisionReconstructionInit,
    pub(crate) normalization: DragonNormConfig,
}

struct ReconCrossViewRequest<B: BackendTrait> {
    views: Tensor<B, 5>,
    view_crops: Option<Tensor<B, 3>>,
    steps: usize,
    backprop_steps: usize,
    randomize_mask: bool,
    capture_artifacts: bool,
    loss_on_all_patches: bool,
}

fn build_denorm_std_patch<B: BackendTrait>(
    recon: VisionReconstructionInit,
    device: &B::Device,
) -> Option<Tensor<B, 3>> {
    if recon.patch_dim == 0 {
        return None;
    }
    let patch_dim = recon.patch_dim.max(1);
    let channels = recon.in_channels.max(1);
    let area = if patch_dim % channels == 0 {
        patch_dim / channels
    } else {
        recon
            .patch_size
            .max(1)
            .saturating_mul(recon.patch_size.max(1))
            .max(1)
    };
    let mut data = Vec::with_capacity(patch_dim);
    for c in 0..channels {
        let std = recon
            .normalize_std
            .get(c)
            .copied()
            .unwrap_or(recon.normalize_std[0]);
        for _ in 0..area {
            data.push(std);
        }
    }
    if data.len() < patch_dim {
        data.resize(patch_dim, recon.normalize_std[0]);
    }
    Some(Tensor::<B, 3>::from_data(
        TensorData::new(data, [1, 1, patch_dim]),
        device,
    ))
}

fn collect_refinement_artifact_maps<B: BackendTrait>(
    model: &VisionDragon<B>,
    source_view: Option<&Tensor<B, 4>>,
    image_count: usize,
    rollout_steps: usize,
    rollout_frames: usize,
) -> (Option<Tensor<B, 4>>, Option<Tensor<B, 5>>) {
    let Some(source_view) = source_view else {
        return (None, None);
    };
    if image_count == 0 || rollout_steps == 0 {
        return (None, None);
    }
    let [batch, _, _, _] = source_view.shape().dims::<4>();
    if batch == 0 {
        return (None, None);
    }
    let image_count = image_count.min(batch);
    if image_count == 0 {
        return (None, None);
    }
    let steps = rollout_steps.max(1);
    let indices = if rollout_frames == 0 {
        (0..steps).collect::<Vec<_>>()
    } else {
        select_trajectory_indices(steps, rollout_frames)
    };
    if indices.is_empty() {
        return (None, None);
    }

    let source = source_view.clone().slice_dim(0, 0..image_count);
    let patch = model.patch_embed(source);
    let tokens = patch.tokens;

    let mut patch_maps_steps: Vec<Tensor<B, 4>> = Vec::new();
    let mut pca_steps: Vec<Tensor<B, 5>> = Vec::new();

    for idx in indices {
        let step = idx + 1;
        let output = model.forward_tokens_embed_steps_rollout_unbounded(tokens.clone(), step, 1);
        let patch_tokens = output.patch_tokens;

        if let Some(patch_map) = patch_heatmap_or_norm(patch_tokens.clone(), image_count) {
            patch_maps_steps.push(patch_map.unsqueeze_dim::<4>(1));
        }
        if let Some(pca_rgb) = pca_patch_rgb(&patch_tokens, image_count) {
            pca_steps.push(pca_rgb.unsqueeze_dim::<5>(1));
        }
    }

    let patch_maps = if patch_maps_steps.is_empty() {
        None
    } else {
        Some(Tensor::cat(patch_maps_steps, 1))
    };
    let pca_maps = if pca_steps.is_empty() {
        None
    } else {
        Some(Tensor::cat(pca_steps, 1))
    };
    (patch_maps, pca_maps)
}

#[derive(Module, Debug)]
pub(crate) struct VisionProbe<B: BackendTrait> {
    pub(crate) norm: DragonNorm<B>,
    pub(crate) head: Linear<B>,
}

impl<B: BackendTrait> VisionProbe<B> {
    pub(crate) fn new(
        embed_dim: usize,
        num_classes: usize,
        norm_config: &DragonNormConfig,
        device: &B::Device,
    ) -> Self {
        let norm = DragonNorm::new(norm_config, embed_dim, device);
        let head = LinearConfig::new(embed_dim, num_classes.max(1)).init(device);
        Self { norm, head }
    }

    pub(crate) fn forward(&self, tokens: Tensor<B, 2>) -> Tensor<B, 2> {
        let tokens = self.norm.forward(tokens);
        self.head.forward(tokens)
    }
}

#[derive(Module, Debug)]
pub(crate) struct VisionReconstructionHead<B: BackendTrait> {
    pub(crate) norm: Option<DragonNorm<B>>,
    pub(crate) hidden: Option<Linear<B>>,
    pub(crate) hidden2: Option<Linear<B>>,
    pub(crate) out: Linear<B>,
}

impl<B: BackendTrait> VisionReconstructionHead<B> {
    pub(crate) fn new(
        embed_dim: usize,
        hidden_dim: usize,
        patch_dim: usize,
        use_norm: bool,
        norm_config: &DragonNormConfig,
        device: &B::Device,
    ) -> Self {
        let norm = if use_norm {
            Some(DragonNorm::new(norm_config, embed_dim, device))
        } else {
            None
        };
        let (hidden, hidden2, out_dim) = if hidden_dim > 0 {
            let hidden = Some(LinearConfig::new(embed_dim, hidden_dim).init(device));
            let hidden2 = Some(LinearConfig::new(hidden_dim, hidden_dim).init(device));
            (hidden, hidden2, hidden_dim.max(1))
        } else {
            (None, None, embed_dim.max(1))
        };
        let out = LinearConfig::new(out_dim, patch_dim.max(1)).init(device);
        Self {
            norm,
            hidden,
            hidden2,
            out,
        }
    }

    pub(crate) fn forward<const D: usize>(&self, tokens: Tensor<B, D>) -> Tensor<B, D> {
        let tokens = match &self.norm {
            Some(norm) => norm.forward(tokens),
            None => tokens,
        };
        let tokens = if let Some(hidden) = &self.hidden {
            activation::gelu(hidden.forward(tokens))
        } else {
            tokens
        };
        let tokens = if let Some(hidden2) = &self.hidden2 {
            activation::gelu(hidden2.forward(tokens))
        } else {
            tokens
        };
        self.out.forward(tokens)
    }
}

#[derive(Module, Debug)]
pub(crate) struct VisionSaccadeHead<B: BackendTrait> {
    pub(crate) norm: DragonNorm<B>,
    pub(crate) proj: Linear<B>,
}

impl<B: BackendTrait> VisionSaccadeHead<B> {
    pub(crate) fn new(
        embed_dim: usize,
        norm_config: &DragonNormConfig,
        device: &B::Device,
    ) -> Self {
        let norm = DragonNorm::new(norm_config, embed_dim, device);
        let proj = LinearConfig::new(embed_dim, 3).init(device);
        Self { norm, proj }
    }

    pub(crate) fn forward(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let tokens = self.norm.forward(tokens);
        self.proj.forward(tokens)
    }
}

#[derive(Module, Debug)]
pub(crate) struct VisionSaccadeProjection<B: BackendTrait> {
    pub(crate) norm: DragonNorm<B>,
    pub(crate) proj: Linear<B>,
}

impl<B: BackendTrait> VisionSaccadeProjection<B> {
    pub(crate) fn new(
        embed_dim: usize,
        out_dim: usize,
        norm_config: &DragonNormConfig,
        device: &B::Device,
    ) -> Self {
        let norm = DragonNorm::new(norm_config, embed_dim, device);
        let proj = LinearConfig::new(embed_dim, out_dim).init(device);
        Self { norm, proj }
    }

    pub(crate) fn forward(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let tokens = self.norm.forward(tokens);
        self.proj.forward(tokens)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct VisionLejepaModel<B: BackendTrait> {
    pub(crate) model: VisionDragon<B>,
    pub(crate) probe: VisionProbe<B>,
    pub(crate) probe_loss: burn::nn::loss::CrossEntropyLoss<B>,
    pub(crate) recon: Option<VisionReconstructionHead<B>>,
    pub(crate) mask_token: Option<Param<Tensor<B, 2>>>,
    pub(crate) config: VisionLejepaConfig,
    pub(crate) teacher: Option<VisionDragon<B>>,
    pub(crate) denorm_std_patch: Option<Tensor<B, 3>>,
    pub(crate) rollout: VisionRollout,
}

#[derive(burn::record::Record)]
pub(crate) struct VisionLejepaModelRecord<B: BackendTrait> {
    pub(crate) model: <VisionDragon<B> as Module<B>>::Record,
    pub(crate) probe: <VisionProbe<B> as Module<B>>::Record,
    pub(crate) probe_loss: <burn::nn::loss::CrossEntropyLoss<B> as Module<B>>::Record,
    pub(crate) recon: <Option<VisionReconstructionHead<B>> as Module<B>>::Record,
    pub(crate) mask_token: <Option<Param<Tensor<B, 2>>> as Module<B>>::Record,
    pub(crate) config: <VisionLejepaConfig as Module<B>>::Record,
}

pub(crate) struct VisionLejepaLosses<B: BackendTrait> {
    pub(crate) total: Tensor<B, 1>,
    pub(crate) inv: Tensor<B, 1>,
    pub(crate) sigreg: Tensor<B, 1>,
    pub(crate) recon: Tensor<B, 1>,
    pub(crate) recon_psnr_masked: Tensor<B, 1>,
    pub(crate) recon_psnr_full: Tensor<B, 1>,
    pub(crate) probe_loss: Tensor<B, 1>,
    pub(crate) probe_acc: Tensor<B, 1>,
    pub(crate) artifacts: Option<VisionArtifactInput<B>>,
}

pub(crate) struct ViewGroupOutput<B: BackendTrait> {
    pub(crate) proj: Tensor<B, 3>,
    pub(crate) embed: Tensor<B, 3>,
    pub(crate) patch_tokens: Tensor<B, 3>,
}

impl<B: BackendTrait> VisionLejepaModel<B> {
    pub(crate) fn new(
        model: VisionDragon<B>,
        config: VisionLejepaConfig,
        init: VisionLejepaInit,
        device: &B::Device,
    ) -> Self {
        let VisionLejepaInit {
            embed_dim,
            num_classes,
            rollout,
            recon: recon_init,
            normalization,
        } = init;
        let probe = VisionProbe::new(embed_dim, num_classes, &normalization, device);
        let probe_loss = CrossEntropyLossConfig::new().init(device);
        let recon_weight = config.loss.recon.weight;
        let recon = if recon_weight > 0.0 {
            if recon_init.patch_dim == 0 {
                None
            } else {
                Some(VisionReconstructionHead::new(
                    embed_dim,
                    config.loss.recon.hidden_dim,
                    recon_init.patch_dim,
                    config.loss.recon.recon_head_norm,
                    &normalization,
                    device,
                ))
            }
        } else {
            None
        };
        let mask_token = recon.as_ref().map(|_| {
            let token = Tensor::<B, 2>::random(
                [1, embed_dim.max(1)],
                TensorDistribution::Normal(0.0, 0.02),
                device,
            );
            Param::from_tensor(token)
        });
        let denorm_std_patch = if recon.is_some() {
            build_denorm_std_patch(recon_init, device)
        } else {
            None
        };
        let teacher = if config.loss.lejepa.enabled {
            init_momentum_teacher::<B, _>(&model, &config.teacher_ema)
        } else {
            None
        };
        Self {
            model,
            probe,
            probe_loss,
            recon,
            mask_token,
            config,
            teacher,
            denorm_std_patch,
            rollout,
        }
    }

    pub(crate) fn sync_teacher_from_student(mut self) -> Self {
        self.teacher = sync_optional_teacher_from_student::<B, _>(
            self.teacher.take(),
            &self.model,
            self.config.loss.lejepa.enabled && self.config.teacher_ema.enabled,
            self.config.teacher_ema.decay,
        );
        self
    }

    pub(crate) fn restore_teacher_from_student(mut self) -> Self {
        self.teacher = restore_optional_teacher_from_student::<B, _>(
            &self.model,
            self.config.loss.lejepa.enabled && self.config.teacher_ema.enabled,
        );
        self
    }

    pub(crate) fn forward_losses(
        &self,
        batch: ImageNetBatch<B>,
        steps: usize,
        backprop_steps: usize,
        randomize_mask: bool,
        capture_artifacts: bool,
        is_validation: bool,
    ) -> VisionLejepaLosses<B> {
        let ImageNetBatch {
            images,
            target_images,
            view_images,
            global_view_images,
            local_view_images,
            labels,
            ..
        } = batch;

        let device = labels.device();
        let collected = collect_views(
            images,
            target_images,
            view_images,
            global_view_images,
            local_view_images,
        );
        let mut proj_groups = Vec::new();
        let mut embed_groups = Vec::new();
        let mut heatmap_source = None;
        let mut pca_source = None;
        let mut artifact_views = None;
        let mut probe_primary = None;
        let mut probe_embed = None;
        let mut recon_loss_sum = Tensor::<B, 1>::zeros([1], &device);
        let mut recon_mask_sum = Tensor::<B, 1>::zeros([1], &device);
        let mut masked_mse_sum = Tensor::<B, 1>::zeros([1], &device);
        let mut full_mse_sum = Tensor::<B, 1>::zeros([1], &device);
        let mut full_mse_count = Tensor::<B, 1>::zeros([1], &device);
        let recon_enabled = self.recon.is_some();

        let capture_artifacts = capture_artifacts && self.config.artifact_every > 0;
        let loss_on_all_patches = self
            .config
            .loss
            .recon
            .loss_on_all_patches_for(is_validation);

        if !collected.global.is_empty() {
            let output = self.forward_view_group(&collected.global, steps, backprop_steps);
            probe_embed = Some(output.embed.clone());
            let [view_count, batch, dim] = output.embed.shape().dims::<3>();
            if view_count > 0 {
                probe_primary = Some(
                    output
                        .embed
                        .clone()
                        .slice_dim(0, 0..1)
                        .reshape([batch, dim]),
                );
                if heatmap_source.is_none() {
                    heatmap_source = Some(output.patch_tokens.clone().slice_dim(0, 0..batch));
                }
                if capture_artifacts && pca_source.is_none() {
                    pca_source = Some(output.patch_tokens.clone().slice_dim(0, 0..batch));
                }
            }

            if recon_enabled {
                let (loss_sum, mask_sum, full_sum, full_count, masked_sum, artifacts) = self
                    .recon_group_loss(
                        &collected.global,
                        steps,
                        backprop_steps,
                        capture_artifacts,
                        randomize_mask,
                        loss_on_all_patches,
                    );
                recon_loss_sum = recon_loss_sum + loss_sum;
                recon_mask_sum = recon_mask_sum + mask_sum;
                full_mse_sum = full_mse_sum + full_sum;
                full_mse_count = full_mse_count + full_count;
                masked_mse_sum = masked_mse_sum + masked_sum;
                if capture_artifacts && let Some((views, residual)) = artifacts {
                    artifact_views = Some(views);
                    heatmap_source = Some(residual);
                }
            }
            proj_groups.push(output.proj);
            embed_groups.push(output.embed);
        }
        if !collected.local.is_empty() {
            let output = self.forward_view_group(&collected.local, steps, backprop_steps);
            if capture_artifacts && heatmap_source.is_none() {
                let [_, batch, _] = output.embed.shape().dims::<3>();
                heatmap_source = Some(output.patch_tokens.clone().slice_dim(0, 0..batch));
            }
            if capture_artifacts && pca_source.is_none() {
                let [_, batch, _] = output.embed.shape().dims::<3>();
                pca_source = Some(output.patch_tokens.clone().slice_dim(0, 0..batch));
            }
            if recon_enabled {
                let (loss_sum, mask_sum, full_sum, full_count, masked_sum, _) = self
                    .recon_group_loss(
                        &collected.local,
                        steps,
                        backprop_steps,
                        false,
                        randomize_mask,
                        loss_on_all_patches,
                    );
                recon_loss_sum = recon_loss_sum + loss_sum;
                recon_mask_sum = recon_mask_sum + mask_sum;
                full_mse_sum = full_mse_sum + full_sum;
                full_mse_count = full_mse_count + full_count;
                masked_mse_sum = masked_mse_sum + masked_sum;
            }
            proj_groups.push(output.proj);
            embed_groups.push(output.embed);
        }
        if proj_groups.is_empty() {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return VisionLejepaLosses {
                total: zero.clone(),
                inv: zero.clone(),
                sigreg: zero.clone(),
                recon: zero.clone(),
                recon_psnr_masked: zero.clone(),
                recon_psnr_full: zero.clone(),
                probe_loss: zero.clone(),
                probe_acc: zero,
                artifacts: None,
            };
        }

        let proj = if proj_groups.len() == 1 {
            proj_groups.pop().expect("proj group")
        } else {
            Tensor::cat(proj_groups, 0)
        };
        let embed = if embed_groups.len() == 1 {
            embed_groups.pop().expect("embed group")
        } else {
            Tensor::cat(embed_groups, 0)
        };
        let teacher_proj = if self.config.loss.lejepa.enabled {
            let teacher_views = if !collected.global.is_empty() {
                collected.global.as_slice()
            } else {
                collected.all.as_slice()
            };
            self.teacher_proj_for_views(teacher_views, steps)
        } else {
            None
        };
        let zero = Tensor::<B, 1>::zeros([1], &device);
        let (inv, sigreg, mut total) = if self.config.loss.lejepa.enabled {
            let inv = if let Some(teacher_proj) = teacher_proj {
                lejepa_teacher_invariance_loss(proj.clone(), teacher_proj)
            } else {
                lejepa_invariance_loss(proj.clone())
            };
            let sigreg = lejepa_sigreg_loss(proj.clone(), &self.config.loss.lejepa);
            let lambda = self.config.loss.lejepa.lambda.clamp(0.0, 1.0);
            let total = inv.clone().mul_scalar(1.0 - lambda) + sigreg.clone().mul_scalar(lambda);
            (inv, sigreg, total)
        } else {
            (zero.clone(), zero.clone(), zero.clone())
        };
        let (recon, recon_psnr_masked, recon_psnr_full) = if recon_enabled {
            let denom = recon_mask_sum.clone().add_scalar(LEJEPA_EPS);
            let recon = recon_loss_sum / denom.clone();
            let weight = self.config.loss.recon.weight.max(0.0);
            total = total + recon.clone().mul_scalar(weight);
            let masked_mse = masked_mse_sum / denom;
            let masked_psnr = recon_psnr(masked_mse);
            let full_denom = full_mse_count.clone().add_scalar(LEJEPA_EPS);
            let full_mse = full_mse_sum / full_denom;
            let full_psnr = recon_psnr(full_mse);
            (recon, masked_psnr, full_psnr)
        } else {
            (zero.clone(), zero.clone(), zero.clone())
        };

        let probe_source = probe_embed.as_ref().unwrap_or(&embed);
        let [view_count, batch, embed_dim] = probe_source.shape().dims::<3>();
        let embed_flat = probe_source
            .clone()
            .reshape([view_count * batch, embed_dim])
            .detach();
        let labels_flat = labels.clone().repeat_dim(0, view_count);
        let probe_logits = self.probe.forward(embed_flat);
        let probe_loss = self
            .probe_loss
            .forward(probe_logits.clone(), labels_flat.clone());
        let probe_pred = probe_logits.clone().argmax(1).reshape([view_count * batch]);
        let probe_acc = probe_pred.equal(labels_flat).float().mean();

        let probe_primary = probe_primary.map(|embed| self.probe.forward(embed.detach()));
        let artifact_views = artifact_views.unwrap_or_else(|| collected.artifact_views());
        let legend = if artifact_views.is_empty() {
            None
        } else if recon_enabled && artifact_views.len() == 3 {
            Some(vec![
                "input".to_string(),
                "masked_input".to_string(),
                "reconstruction".to_string(),
            ])
        } else if artifact_views.len() == 1 {
            Some(vec!["input".to_string()])
        } else {
            Some(
                (0..artifact_views.len())
                    .map(|idx| format!("view_{idx}"))
                    .collect(),
            )
        };
        let (patch_norms_steps, pca_rgb_steps) = if capture_artifacts {
            let source_view = if !collected.global.is_empty() {
                collected.global.first()
            } else if !collected.local.is_empty() {
                collected.local.first()
            } else {
                artifact_views.first()
            };
            let rollout_steps = if self.config.artifact_rollout_steps > 0 {
                self.config.artifact_rollout_steps
            } else {
                steps.max(1)
            };
            collect_refinement_artifact_maps(
                &self.model,
                source_view,
                self.config.artifact_max_images,
                rollout_steps,
                self.config.artifact_rollout_frames,
            )
        } else {
            (None, None)
        };
        let artifacts = if capture_artifacts {
            build_lejepa_artifacts(
                &self.config,
                &artifact_views,
                LejepaArtifactBuildInput {
                    frames: None,
                    first_patch: heatmap_source,
                    pca_source,
                    patch_norms_steps,
                    pca_rgb_steps,
                    probe_logits: probe_primary,
                    labels: Some(labels),
                    legend,
                },
            )
        } else {
            None
        };

        VisionLejepaLosses {
            total,
            inv,
            sigreg,
            recon,
            recon_psnr_masked,
            recon_psnr_full,
            probe_loss,
            probe_acc,
            artifacts,
        }
    }

    pub(crate) fn recon_group_loss(
        &self,
        views: &[Tensor<B, 4>],
        steps: usize,
        backprop_steps: usize,
        capture_artifacts: bool,
        randomize_mask: bool,
        loss_on_all_patches: bool,
    ) -> ReconLossOutput<B> {
        let recon = match &self.recon {
            Some(recon) => recon,
            None => {
                let device = views.first().map(|view| view.device()).unwrap_or_default();
                let zero = Tensor::<B, 1>::zeros([1], &device);
                return (
                    zero.clone(),
                    zero.clone(),
                    zero.clone(),
                    zero.clone(),
                    zero,
                    None,
                );
            }
        };
        if views.is_empty() {
            let device = <B as BackendTrait>::Device::default();
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero,
                None,
            );
        }
        let device = views[0].device();
        let [batch, channels, height, width] = views[0].shape().dims::<4>();
        let stacked = stack_views(views);
        let patch = self.model.patch_embed_raw(stacked.clone());
        let [total, tokens, embed_dim] = patch.tokens.shape().dims::<3>();
        let grid_h = patch.grid.height;
        let grid_w = patch.grid.width;
        if grid_h == 0 || grid_w == 0 || grid_h * grid_w != tokens {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero,
                None,
            );
        }
        let patch_size = self.model.patch_size().max(1);

        let target_patches = patchify(stacked, patch_size);
        let mask_ratio = self.config.loss.recon.mask_ratio;
        let mask = sample_patch_mask(&device, total, tokens, mask_ratio, randomize_mask);
        let loss_mask = if loss_on_all_patches {
            Tensor::<B, 2>::ones([total, tokens], &device)
        } else {
            mask.clone()
        };
        let mask_expanded = mask.clone().unsqueeze_dim::<3>(2);
        let keep = mask_expanded.clone().mul_scalar(-1.0).add_scalar(1.0);
        let mut masked_tokens = patch.tokens.clone().mul(keep.clone());
        if let Some(mask_token) = &self.mask_token {
            let token = mask_token
                .val()
                .reshape([1, 1, embed_dim])
                .repeat_dim(0, total)
                .repeat_dim(1, tokens);
            masked_tokens = masked_tokens + token.mul(mask_expanded.clone());
        }
        let masked_tokens = self.model.add_patch_position(masked_tokens, patch.grid);
        let embed_out =
            self.model
                .forward_tokens_embed_steps_rollout(masked_tokens, steps, backprop_steps);

        let pred_patches = recon.forward(embed_out.patch_tokens);
        let [total, tokens, patch_dim] = pred_patches.shape().dims::<3>();
        debug_assert_eq!(
            target_patches.shape().dims::<3>(),
            pred_patches.shape().dims::<3>(),
            "recon patches shape mismatch"
        );
        if total == 0 || tokens == 0 || patch_dim == 0 {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero,
                None,
            );
        }

        let diff = pred_patches.clone() - target_patches.clone();
        let loss_sum = diff
            .clone()
            .powf_scalar(2.0)
            .mul(loss_mask.clone().unsqueeze_dim::<3>(2))
            .sum();
        let mask_sum = loss_mask.clone().sum().mul_scalar(patch_dim as f32);
        let diff_raw_masked = if let Some(std_patch) = &self.denorm_std_patch {
            diff.clone().mul(std_patch.clone())
        } else {
            diff.clone()
        };
        let masked_full_loss_sum = diff_raw_masked
            .powf_scalar(2.0)
            .mul(loss_mask.clone().unsqueeze_dim::<3>(2))
            .sum();
        let psnr_patches = if loss_on_all_patches {
            pred_patches.clone()
        } else {
            pred_patches.clone().mul(mask_expanded.clone())
                + target_patches.clone().mul(keep.clone())
        };
        let diff_raw = if let Some(std_patch) = &self.denorm_std_patch {
            (psnr_patches - target_patches.clone()).mul(std_patch.clone())
        } else {
            psnr_patches - target_patches.clone()
        };
        let full_loss_sum = diff_raw.powf_scalar(2.0).sum();
        let full_count = (total.saturating_mul(tokens).saturating_mul(patch_dim)) as f32;
        let full_count = Tensor::<B, 1>::ones([1], &device).mul_scalar(full_count);

        let artifacts = if capture_artifacts && batch > 0 {
            let pred_first = pred_patches.slice_dim(0, 0..batch);
            let target_first = target_patches.slice_dim(0, 0..batch);
            let mask_first = mask.slice_dim(0, 0..batch);
            let loss_mask_first = loss_mask.slice_dim(0, 0..batch);
            let mask_expanded = mask_first.clone().unsqueeze_dim::<3>(2);
            let loss_mask_expanded = loss_mask_first.clone().unsqueeze_dim::<3>(2);
            let keep = mask_expanded.clone().mul_scalar(-1.0).add_scalar(1.0);
            let masked_patches = target_first.clone().mul(keep.clone());
            let recon_patches = if loss_on_all_patches {
                pred_first.clone()
            } else {
                pred_first.clone().mul(mask_expanded.clone()) + target_first.clone().mul(keep)
            };
            let masked_view = unpatchify(masked_patches, patch_size, height, width, channels);
            let recon_view = unpatchify(recon_patches, patch_size, height, width, channels);
            let residual = (pred_first - target_first).mul(loss_mask_expanded);
            Some((vec![views[0].clone(), masked_view, recon_view], residual))
        } else {
            None
        };

        (
            loss_sum,
            mask_sum,
            full_loss_sum,
            full_count,
            masked_full_loss_sum,
            artifacts,
        )
    }

    pub(crate) fn forward_view_group(
        &self,
        views: &[Tensor<B, 4>],
        steps: usize,
        backprop_steps: usize,
    ) -> ViewGroupOutput<B> {
        Self::forward_view_group_with_model(&self.model, views, steps, backprop_steps)
    }

    pub(crate) fn teacher_proj_for_views(
        &self,
        views: &[Tensor<B, 4>],
        steps: usize,
    ) -> Option<Tensor<B, 3>> {
        let teacher = self.teacher.as_ref()?;
        if views.is_empty() {
            return None;
        }
        Some(Self::forward_view_group_with_model(teacher, views, steps, steps).proj)
    }

    fn forward_view_group_with_model(
        model: &VisionDragon<B>,
        views: &[Tensor<B, 4>],
        steps: usize,
        backprop_steps: usize,
    ) -> ViewGroupOutput<B> {
        let view_count = views.len();
        let [batch, _, _, _] = views[0].shape().dims::<4>();
        let stacked = stack_views(views);
        let patch = model.patch_embed(stacked);
        let embed_out =
            model.forward_tokens_embed_steps_rollout(patch.tokens, steps, backprop_steps);
        let cls_embed = embed_out.cls_token;
        let patch_tokens = embed_out.patch_tokens;
        let [total, embed_dim] = cls_embed.shape().dims::<2>();
        debug_assert_eq!(total, view_count * batch, "lejepa embed mismatch");
        let tokens = Tensor::cat(
            vec![
                cls_embed.clone().unsqueeze_dim::<3>(1),
                patch_tokens.clone(),
            ],
            1,
        );
        let proj_tokens = model.project_tokens(tokens);
        let proj_dim = proj_tokens.shape().dims::<3>()[2];
        let proj_cls = proj_tokens
            .slice_dim(1, 0..1)
            .reshape([view_count, batch, proj_dim]);
        let embed_cls = cls_embed.reshape([view_count, batch, embed_dim]);
        ViewGroupOutput {
            proj: proj_cls,
            embed: embed_cls,
            patch_tokens,
        }
    }
}

impl<B: BackendTrait> Module<B> for VisionLejepaModel<B> {
    type Record = VisionLejepaModelRecord<B>;

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        let devices = Module::collect_devices(&self.model, devices);
        let devices = Module::collect_devices(&self.probe, devices);
        let devices = Module::collect_devices(&self.probe_loss, devices);
        let devices = Module::collect_devices(&self.recon, devices);
        let devices = Module::collect_devices(&self.mask_token, devices);
        let devices = Module::<B>::collect_devices(&self.config, devices);
        let devices = Module::collect_devices(&self.teacher, devices);
        Module::collect_devices(&self.denorm_std_patch, devices)
    }

    fn fork(self, device: &B::Device) -> Self {
        Self {
            model: Module::fork(self.model, device),
            probe: Module::fork(self.probe, device),
            probe_loss: Module::fork(self.probe_loss, device),
            recon: Module::fork(self.recon, device),
            mask_token: Module::fork(self.mask_token, device),
            config: Module::<B>::fork(self.config, device),
            teacher: Module::fork(self.teacher, device),
            denorm_std_patch: Module::fork(self.denorm_std_patch, device),
            rollout: self.rollout,
        }
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            model: Module::to_device(self.model, device),
            probe: Module::to_device(self.probe, device),
            probe_loss: Module::to_device(self.probe_loss, device),
            recon: Module::to_device(self.recon, device),
            mask_token: Module::to_device(self.mask_token, device),
            config: Module::<B>::to_device(self.config, device),
            teacher: Module::to_device(self.teacher, device),
            denorm_std_patch: Module::to_device(self.denorm_std_patch, device),
            rollout: self.rollout,
        }
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, visitor: &mut Visitor) {
        Module::visit(&self.model, visitor);
        Module::visit(&self.probe, visitor);
        Module::visit(&self.probe_loss, visitor);
        Module::visit(&self.recon, visitor);
        Module::visit(&self.mask_token, visitor);
        Module::visit(&self.config, visitor);
    }

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, mapper: &mut Mapper) -> Self {
        Self {
            model: Module::map(self.model, mapper),
            probe: Module::map(self.probe, mapper),
            probe_loss: Module::map(self.probe_loss, mapper),
            recon: Module::map(self.recon, mapper),
            mask_token: Module::map(self.mask_token, mapper),
            config: Module::<B>::map(self.config, mapper),
            teacher: self.teacher,
            denorm_std_patch: self.denorm_std_patch,
            rollout: self.rollout,
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            model: Module::load_record(self.model, record.model),
            probe: Module::load_record(self.probe, record.probe),
            probe_loss: Module::load_record(self.probe_loss, record.probe_loss),
            recon: Module::load_record(self.recon, record.recon),
            mask_token: Module::load_record(self.mask_token, record.mask_token),
            config: {
                let _: () = record.config;
                Module::<B>::load_record(self.config, ())
            },
            teacher: None,
            denorm_std_patch: self.denorm_std_patch,
            rollout: self.rollout,
        }
        .restore_teacher_from_student()
    }

    fn into_record(self) -> Self::Record {
        VisionLejepaModelRecord {
            model: Module::into_record(self.model),
            probe: Module::into_record(self.probe),
            probe_loss: Module::into_record(self.probe_loss),
            recon: Module::into_record(self.recon),
            mask_token: Module::into_record(self.mask_token),
            config: Module::<B>::into_record(self.config),
        }
    }
}

impl<B: AutodiffBackend> AutodiffModule<B> for VisionLejepaModel<B> {
    type InnerModule = VisionLejepaModel<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        VisionLejepaModel {
            model: AutodiffModule::valid(&self.model),
            probe: AutodiffModule::valid(&self.probe),
            probe_loss: AutodiffModule::valid(&self.probe_loss),
            recon: AutodiffModule::valid(&self.recon),
            mask_token: AutodiffModule::valid(&self.mask_token),
            config: AutodiffModule::<B>::valid(&self.config),
            teacher: AutodiffModule::valid(&self.teacher),
            denorm_std_patch: AutodiffModule::valid(&self.denorm_std_patch),
            rollout: self.rollout,
        }
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        VisionLejepaModel {
            model: AutodiffModule::from_inner(module.model),
            probe: AutodiffModule::from_inner(module.probe),
            probe_loss: AutodiffModule::from_inner(module.probe_loss),
            recon: AutodiffModule::from_inner(module.recon),
            mask_token: AutodiffModule::from_inner(module.mask_token),
            config: AutodiffModule::<B>::from_inner(module.config),
            teacher: AutodiffModule::from_inner(module.teacher),
            denorm_std_patch: AutodiffModule::from_inner(module.denorm_std_patch),
            rollout: module.rollout,
        }
    }
}

impl<B: BackendTrait> core::fmt::Display for VisionLejepaModel<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&burn::module::ModuleDisplay::format(
            self,
            burn::module::DisplaySettings::default(),
        ))
    }
}

impl<B: BackendTrait> ModuleDisplayDefault for VisionLejepaModel<B> {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("model", &self.model)
            .add("probe", &self.probe)
            .add("recon", &self.recon)
            .add("config", &self.config)
            .optional()
    }

    fn num_params(&self) -> usize {
        Module::num_params(self)
    }
}

impl<B: BackendTrait> ModuleDisplay for VisionLejepaModel<B> {}

#[derive(Module, Debug)]
pub(crate) struct VisionMaeModel<B: BackendTrait> {
    pub(crate) model: VisionDragon<B>,
    pub(crate) recon: VisionReconstructionHead<B>,
    pub(crate) mask_token: Param<Tensor<B, 2>>,
    pub(crate) visible_token: Param<Tensor<B, 2>>,
    pub(crate) view_embed: Option<Linear<B>>,
    pub(crate) config: VisionMaeConfig,
    #[module(ignore)]
    pub(crate) denorm_std_patch: Option<Tensor<B, 3>>,
    #[module(ignore)]
    pub(crate) num_eyes: usize,
    #[module(ignore)]
    pub(crate) rollout: VisionRollout,
}

pub(crate) struct VisionMaeLosses<B: BackendTrait> {
    pub(crate) total: Tensor<B, 1>,
    pub(crate) recon: Tensor<B, 1>,
    pub(crate) recon_psnr_masked: Tensor<B, 1>,
    pub(crate) recon_psnr_full: Tensor<B, 1>,
    pub(crate) artifacts: Option<VisionArtifactInput<B>>,
}

impl<B: BackendTrait> VisionMaeModel<B> {
    pub(crate) fn new(
        model: VisionDragon<B>,
        config: VisionMaeConfig,
        init: VisionMaeInit,
        device: &B::Device,
    ) -> Self {
        let VisionMaeInit {
            num_eyes,
            embed_dim,
            rollout,
            recon: recon_init,
            normalization,
        } = init;
        let recon = VisionReconstructionHead::new(
            embed_dim,
            config.loss.recon.hidden_dim,
            recon_init.patch_dim,
            config.loss.recon.recon_head_norm,
            &normalization,
            device,
        );
        let token = Tensor::<B, 2>::random(
            [1, embed_dim.max(1)],
            TensorDistribution::Normal(0.0, 0.02),
            device,
        );
        let mask_token = Param::from_tensor(token);
        let visible_token = Param::from_tensor(Tensor::<B, 2>::random(
            [1, embed_dim.max(1)],
            TensorDistribution::Normal(0.0, 0.02),
            device,
        ));
        let view_embed = if num_eyes.max(1) > 1 {
            Some(LinearConfig::new(4, embed_dim.max(1)).init(device))
        } else {
            None
        };
        let denorm_std_patch = build_denorm_std_patch(recon_init, device);
        Self {
            model,
            recon,
            mask_token,
            visible_token,
            view_embed,
            config,
            denorm_std_patch,
            num_eyes: num_eyes.max(1),
            rollout,
        }
    }

    pub(crate) fn forward_losses(
        &self,
        batch: ImageNetBatch<B>,
        steps: usize,
        backprop_steps: usize,
        randomize_mask: bool,
        capture_artifacts: bool,
        is_validation: bool,
    ) -> VisionMaeLosses<B> {
        let ImageNetBatch {
            images,
            target_images,
            view_images,
            view_crops,
            labels,
            ..
        } = batch;
        let loss_on_all_patches = self
            .config
            .loss
            .recon
            .loss_on_all_patches_for(is_validation);
        let (
            loss_sum,
            mask_sum,
            full_loss_sum,
            full_count,
            masked_full_loss_sum,
            visible_loss_sum,
            visible_mask_sum,
            artifacts,
        ) = if self.config.cross_view.enabled {
            let num_eyes = self.num_eyes.max(1);
            let views = if let Some(view_images) = view_images {
                view_images
            } else if let Some(target_images) = target_images {
                let primary = images.clone().unsqueeze_dim::<5>(1);
                let target = target_images.unsqueeze_dim::<5>(1);
                Tensor::cat(vec![primary, target], 1)
            } else {
                let primary = images.clone().unsqueeze_dim::<5>(1);
                primary.repeat_dim(1, num_eyes.max(1))
            };
            let views = if num_eyes > 0 {
                views.slice_dim(1, 0..num_eyes)
            } else {
                views
            };
            let view_crops = view_crops.map(|crops| {
                if num_eyes > 0 {
                    crops.slice_dim(1, 0..num_eyes)
                } else {
                    crops
                }
            });
            let (
                loss_sum,
                mask_sum,
                masked_full_loss_sum,
                visible_loss_sum,
                visible_mask_sum,
                artifacts,
            ) = self.recon_loss_cross_view(ReconCrossViewRequest {
                views,
                view_crops,
                steps,
                backprop_steps,
                randomize_mask,
                capture_artifacts,
                loss_on_all_patches,
            });
            let zero = Tensor::<B, 1>::zeros([1], &loss_sum.device());
            (
                loss_sum,
                mask_sum,
                zero.clone(),
                zero.clone(),
                masked_full_loss_sum,
                visible_loss_sum,
                visible_mask_sum,
                artifacts,
            )
        } else {
            let (loss_sum, mask_sum, full_loss_sum, full_count, masked_full_loss_sum, artifacts) =
                self.recon_loss(
                    images,
                    steps,
                    backprop_steps,
                    randomize_mask,
                    capture_artifacts,
                    loss_on_all_patches,
                );
            let zero = Tensor::<B, 1>::zeros([1], &loss_sum.device());
            (
                loss_sum,
                mask_sum,
                full_loss_sum,
                full_count,
                masked_full_loss_sum,
                zero.clone(),
                zero,
                artifacts,
            )
        };
        let denom = mask_sum.clone().add_scalar(LEJEPA_EPS);
        let recon = loss_sum / denom.clone();
        let masked_mse = masked_full_loss_sum / denom;
        let recon_psnr_masked = recon_psnr(masked_mse);
        let recon_psnr_full = if self.config.cross_view.enabled {
            recon_psnr_masked.clone()
        } else {
            let full_denom = full_count.clone().add_scalar(LEJEPA_EPS);
            let full_mse = full_loss_sum / full_denom;
            recon_psnr(full_mse)
        };
        let visible = if self.config.cross_view.visible_weight > 0.0 {
            let denom = visible_mask_sum.clone().add_scalar(LEJEPA_EPS);
            visible_loss_sum / denom
        } else {
            Tensor::<B, 1>::zeros([1], &recon.device())
        };
        let total = recon
            .clone()
            .mul_scalar(self.config.loss.recon.weight.max(0.0))
            + visible.mul_scalar(self.config.cross_view.visible_weight.max(0.0));

        let (patch_norms_steps, pca_rgb_steps) = if capture_artifacts {
            let source_view = artifacts.as_ref().and_then(|(views, _)| views.first());
            let rollout_steps = if self.config.artifact_rollout_steps > 0 {
                self.config.artifact_rollout_steps
            } else {
                steps.max(1)
            };
            collect_refinement_artifact_maps(
                &self.model,
                source_view,
                self.config.artifact_max_images,
                rollout_steps,
                self.config.artifact_rollout_frames,
            )
        } else {
            (None, None)
        };

        let artifacts = if let Some((views, residual)) = artifacts {
            build_lejepa_artifacts(
                &VisionLejepaConfig {
                    artifact_every: self.config.artifact_every,
                    artifact_max_images: self.config.artifact_max_images,
                    artifact_max_views: self.config.artifact_max_views,
                    artifact_rollout_steps: self.config.artifact_rollout_steps,
                    artifact_rollout_frames: self.config.artifact_rollout_frames,
                    ..VisionLejepaConfig::default()
                },
                &views,
                LejepaArtifactBuildInput {
                    frames: None,
                    first_patch: Some(residual),
                    pca_source: None,
                    patch_norms_steps,
                    pca_rgb_steps,
                    probe_logits: None,
                    labels: Some(labels),
                    legend: Some(vec![
                        "input".to_string(),
                        "masked_input".to_string(),
                        "reconstruction".to_string(),
                    ]),
                },
            )
        } else {
            None
        };

        VisionMaeLosses {
            total,
            recon,
            recon_psnr_masked,
            recon_psnr_full,
            artifacts,
        }
    }

    pub(crate) fn recon_loss(
        &self,
        images: Tensor<B, 4>,
        steps: usize,
        backprop_steps: usize,
        randomize_mask: bool,
        capture_artifacts: bool,
        loss_on_all_patches: bool,
    ) -> ReconLossOutput<B> {
        let device = images.device();
        let patch_size = self.model.patch_size().max(1);
        let pyramid_levels = self.config.pyramid_levels.max(1);
        let mut pyramid = Vec::with_capacity(pyramid_levels);
        let mut current = images;
        pyramid.push(current.clone());
        for _ in 1..pyramid_levels {
            let Some(next) = downsample_image(current.clone()) else {
                break;
            };
            pyramid.push(next.clone());
            current = next;
        }
        pyramid.reverse();

        let level_count = pyramid.len();
        let mut loss_sum: Option<Tensor<B, 1>> = None;
        let mut mask_sum: Option<Tensor<B, 1>> = None;
        let mut full_loss_sum: Option<Tensor<B, 1>> = None;
        let mut full_count_sum: Option<Tensor<B, 1>> = None;
        let mut masked_full_loss_sum: Option<Tensor<B, 1>> = None;
        let mut artifacts: ReconArtifacts<B> = None;

        for (level_idx, level_images) in pyramid.into_iter().enumerate() {
            let [batch, channels, height, width] = level_images.shape().dims::<4>();
            let patch = self.model.patch_embed_raw(level_images.clone());
            let [_, tokens, embed_dim] = patch.tokens.shape().dims::<3>();
            let grid_h = patch.grid.height;
            let grid_w = patch.grid.width;
            if batch == 0 || grid_h == 0 || grid_w == 0 || grid_h * grid_w != tokens {
                continue;
            }

            let target_patches = patchify(level_images.clone(), patch_size);
            let mask_ratio = self.config.loss.recon.mask_ratio;
            let mask = sample_patch_mask(&device, batch, tokens, mask_ratio, randomize_mask);
            let loss_mask = if loss_on_all_patches {
                Tensor::<B, 2>::ones([batch, tokens], &device)
            } else {
                mask.clone()
            };

            let mask_expanded = mask.clone().unsqueeze_dim::<3>(2);
            let keep = mask_expanded.clone().mul_scalar(-1.0).add_scalar(1.0);
            let mut masked_tokens = patch.tokens.clone().mul(keep.clone());
            let visible_token = self
                .visible_token
                .val()
                .reshape([1, 1, embed_dim])
                .repeat_dim(0, batch)
                .repeat_dim(1, tokens);
            let mask_token = self
                .mask_token
                .val()
                .reshape([1, 1, embed_dim])
                .repeat_dim(0, batch)
                .repeat_dim(1, tokens);
            masked_tokens = masked_tokens
                + visible_token.mul(keep.clone())
                + mask_token.mul(mask_expanded.clone());
            let masked_tokens = self.model.add_patch_position(masked_tokens, patch.grid);
            let embed_out =
                self.model
                    .forward_tokens_embed_steps_rollout(masked_tokens, steps, backprop_steps);

            let pred_patches = self.recon.forward(embed_out.patch_tokens);
            let [total, tokens, patch_dim] = pred_patches.shape().dims::<3>();
            if total == 0 || tokens == 0 || patch_dim == 0 {
                continue;
            }

            let diff = pred_patches.clone() - target_patches.clone();
            let loss_sum_level = diff
                .clone()
                .powf_scalar(2.0)
                .mul(loss_mask.clone().unsqueeze_dim::<3>(2))
                .sum();
            let mask_sum_level = loss_mask.clone().sum().mul_scalar(patch_dim as f32);
            let diff_raw_masked = if let Some(std_patch) = &self.denorm_std_patch {
                diff.clone().mul(std_patch.clone())
            } else {
                diff.clone()
            };
            let masked_full_loss_sum_level = diff_raw_masked
                .powf_scalar(2.0)
                .mul(loss_mask.clone().unsqueeze_dim::<3>(2))
                .sum();
            let psnr_patches = if loss_on_all_patches {
                pred_patches.clone()
            } else {
                pred_patches.clone().mul(mask_expanded.clone())
                    + target_patches.clone().mul(keep.clone())
            };
            let diff_raw = if let Some(std_patch) = &self.denorm_std_patch {
                (psnr_patches - target_patches.clone()).mul(std_patch.clone())
            } else {
                psnr_patches - target_patches.clone()
            };
            let full_loss_sum_level = diff_raw.powf_scalar(2.0).sum();
            let full_count_level = (total.saturating_mul(tokens).saturating_mul(patch_dim)) as f32;
            let full_count_level = Tensor::<B, 1>::ones([1], &device).mul_scalar(full_count_level);
            loss_sum = Some(match loss_sum {
                Some(accum) => accum + loss_sum_level,
                None => loss_sum_level,
            });
            mask_sum = Some(match mask_sum {
                Some(accum) => accum + mask_sum_level,
                None => mask_sum_level,
            });
            full_loss_sum = Some(match full_loss_sum {
                Some(accum) => accum + full_loss_sum_level,
                None => full_loss_sum_level,
            });
            full_count_sum = Some(match full_count_sum {
                Some(accum) => accum + full_count_level,
                None => full_count_level,
            });
            masked_full_loss_sum = Some(match masked_full_loss_sum {
                Some(accum) => accum + masked_full_loss_sum_level,
                None => masked_full_loss_sum_level,
            });

            if capture_artifacts && level_idx + 1 == level_count && batch > 0 {
                let pred_first = pred_patches.slice_dim(0, 0..batch);
                let target_first = target_patches.slice_dim(0, 0..batch);
                let mask_first = mask.slice_dim(0, 0..batch);
                let loss_mask_first = loss_mask.slice_dim(0, 0..batch);
                let mask_expanded = mask_first.clone().unsqueeze_dim::<3>(2);
                let loss_mask_expanded = loss_mask_first.clone().unsqueeze_dim::<3>(2);
                let keep = mask_expanded.clone().mul_scalar(-1.0).add_scalar(1.0);
                let masked_patches = target_first.clone().mul(keep.clone());
                let recon_patches = if loss_on_all_patches {
                    pred_first.clone()
                } else {
                    pred_first.clone().mul(mask_expanded.clone()) + target_first.clone().mul(keep)
                };
                let masked_view = unpatchify(masked_patches, patch_size, height, width, channels);
                let recon_view = unpatchify(recon_patches, patch_size, height, width, channels);
                let residual = (pred_first - target_first).mul(loss_mask_expanded);
                artifacts = Some((
                    vec![level_images.clone(), masked_view, recon_view],
                    residual,
                ));
            }
        }

        let zero = Tensor::<B, 1>::zeros([1], &device);
        let loss_sum = loss_sum.unwrap_or(zero.clone());
        let mask_sum = mask_sum.unwrap_or(zero.clone());
        let full_loss_sum = full_loss_sum.unwrap_or(zero.clone());
        let full_count = full_count_sum.unwrap_or_else(|| zero.clone());
        let masked_full_loss_sum = masked_full_loss_sum.unwrap_or_else(|| zero.clone());
        (
            loss_sum,
            mask_sum,
            full_loss_sum,
            full_count,
            masked_full_loss_sum,
            artifacts,
        )
    }

    fn recon_loss_cross_view(&self, request: ReconCrossViewRequest<B>) -> ReconCrossViewOutput<B> {
        let ReconCrossViewRequest {
            views,
            view_crops,
            steps,
            backprop_steps,
            randomize_mask,
            capture_artifacts,
            loss_on_all_patches,
        } = request;
        let device = views.device();
        let patch_size = self.model.patch_size().max(1);
        let pyramid_levels = self.config.pyramid_levels.max(1);
        let mut pyramid = Vec::with_capacity(pyramid_levels);
        let mut current = views;
        pyramid.push(current.clone());
        for _ in 1..pyramid_levels {
            let [batch, eyes, channels, height, width] = current.shape().dims::<5>();
            let flat = current.reshape([batch * eyes, channels, height, width]);
            let Some(next) = downsample_image(flat) else {
                break;
            };
            let [_, _, next_h, next_w] = next.shape().dims::<4>();
            let next = next.reshape([batch, eyes, channels, next_h, next_w]);
            pyramid.push(next.clone());
            current = next;
        }
        pyramid.reverse();

        let level_count = pyramid.len();
        let mut loss_sum: Option<Tensor<B, 1>> = None;
        let mut mask_sum: Option<Tensor<B, 1>> = None;
        let mut visible_loss_sum: Option<Tensor<B, 1>> = None;
        let mut visible_mask_sum: Option<Tensor<B, 1>> = None;
        let mut masked_full_loss_sum: Option<Tensor<B, 1>> = None;
        let mut artifacts: ReconArtifacts<B> = None;
        let masked_eye = self.config.cross_view.masked_eye;
        let visible_weight = self.config.cross_view.visible_weight;

        for (level_idx, level_views) in pyramid.into_iter().enumerate() {
            let [batch, eyes, channels, height, width] = level_views.shape().dims::<5>();
            if batch == 0 || eyes == 0 {
                continue;
            }
            let flat = level_views
                .clone()
                .reshape([batch * eyes, channels, height, width]);
            let patch = self.model.patch_embed_raw(flat);
            let [_, tokens, embed_dim] = patch.tokens.shape().dims::<3>();
            let grid_h = patch.grid.height;
            let grid_w = patch.grid.width;
            if grid_h == 0 || grid_w == 0 || grid_h * grid_w != tokens {
                continue;
            }

            let target_images = level_views
                .clone()
                .slice_dim(1, masked_eye..masked_eye + 1)
                .reshape([batch, channels, height, width]);
            let target_patches = patchify(target_images.clone(), patch_size);

            let mask_ratio = self.config.loss.recon.mask_ratio;
            let mask = sample_patch_mask(&device, batch, tokens, mask_ratio, randomize_mask);
            let loss_mask = if loss_on_all_patches {
                Tensor::<B, 2>::ones([batch, tokens], &device)
            } else {
                mask.clone()
            };

            let mask_expanded = mask.clone().unsqueeze_dim::<3>(2);
            let keep = mask_expanded.clone().mul_scalar(-1.0).add_scalar(1.0);

            let view_tokens = patch.tokens.reshape([batch, eyes, tokens, embed_dim]);
            let target_tokens = view_tokens
                .clone()
                .slice_dim(1, masked_eye..masked_eye + 1)
                .reshape([batch, tokens, embed_dim]);
            let visible_token = self
                .visible_token
                .val()
                .reshape([1, 1, embed_dim])
                .repeat_dim(0, batch)
                .repeat_dim(1, tokens);
            let mask_token = self
                .mask_token
                .val()
                .reshape([1, 1, embed_dim])
                .repeat_dim(0, batch)
                .repeat_dim(1, tokens);
            let mut masked_target = target_tokens.clone().mul(keep.clone());
            masked_target = masked_target
                + visible_token.mul(keep.clone())
                + mask_token.mul(mask_expanded.clone());

            let mut eye_tokens = Vec::with_capacity(eyes);
            for eye_idx in 0..eyes {
                if eye_idx == masked_eye {
                    eye_tokens.push(masked_target.clone().reshape([batch, 1, tokens, embed_dim]));
                } else {
                    eye_tokens.push(view_tokens.clone().slice_dim(1, eye_idx..eye_idx + 1));
                }
            }
            let mut eye_tokens = Tensor::cat(eye_tokens, 1);
            eye_tokens = self.model.add_patch_position_multi(eye_tokens, patch.grid);
            if let (Some(view_crops), Some(view_embed)) = (&view_crops, self.view_embed.as_ref()) {
                let [batch, eyes, _] = view_crops.shape().dims::<3>();
                if batch > 0 && eyes > 0 {
                    let flat = view_crops.clone().reshape([batch * eyes, 4]);
                    let embed = view_embed
                        .forward(flat)
                        .reshape([batch, eyes, 1, embed_dim])
                        .repeat_dim(2, tokens);
                    eye_tokens = eye_tokens + embed;
                }
            }
            let embed_out = self.model.forward_tokens_embed_steps_rollout_multi(
                eye_tokens,
                steps,
                backprop_steps,
            );
            let patch_tokens = embed_out.patch_tokens.clone();
            let mut pred_tokens = patch_tokens
                .clone()
                .slice_dim(1, masked_eye..masked_eye + 1)
                .reshape([batch, tokens, embed_dim]);
            if eyes > 1 && self.config.cross_view.fuse_alpha > 0.0 {
                let alpha = self.config.cross_view.fuse_alpha.clamp(0.0, 1.0);
                let sum = patch_tokens
                    .clone()
                    .sum_dim(1)
                    .reshape([batch, tokens, embed_dim]);
                let other_count = (eyes - 1).max(1) as f32;
                let other_mean = (sum - pred_tokens.clone()).div_scalar(other_count);
                pred_tokens = pred_tokens.mul_scalar(1.0 - alpha) + other_mean.mul_scalar(alpha);
            }

            let pred_patches = self.recon.forward(pred_tokens);
            let [total, tokens, patch_dim] = pred_patches.shape().dims::<3>();
            if total == 0 || tokens == 0 || patch_dim == 0 {
                continue;
            }

            let diff = pred_patches.clone() - target_patches.clone();
            let loss_sum_level = diff
                .clone()
                .powf_scalar(2.0)
                .mul(loss_mask.clone().unsqueeze_dim::<3>(2))
                .sum();
            let mask_sum_level = loss_mask.clone().sum().mul_scalar(patch_dim as f32);
            let diff_raw_masked = if let Some(std_patch) = &self.denorm_std_patch {
                diff.clone().mul(std_patch.clone())
            } else {
                diff.clone()
            };
            let masked_full_loss_sum_level = diff_raw_masked
                .powf_scalar(2.0)
                .mul(loss_mask.clone().unsqueeze_dim::<3>(2))
                .sum();
            loss_sum = Some(match loss_sum {
                Some(accum) => accum + loss_sum_level,
                None => loss_sum_level,
            });
            mask_sum = Some(match mask_sum {
                Some(accum) => accum + mask_sum_level,
                None => mask_sum_level,
            });
            masked_full_loss_sum = Some(match masked_full_loss_sum {
                Some(accum) => accum + masked_full_loss_sum_level,
                None => masked_full_loss_sum_level,
            });

            if visible_weight > 0.0 && eyes > 1 {
                let mut visible_tokens = Vec::with_capacity(eyes.saturating_sub(1));
                let mut visible_targets = Vec::with_capacity(eyes.saturating_sub(1));
                for eye_idx in 0..eyes {
                    if eye_idx == masked_eye {
                        continue;
                    }
                    let tokens = patch_tokens
                        .clone()
                        .slice_dim(1, eye_idx..eye_idx + 1)
                        .reshape([batch, tokens, embed_dim]);
                    let view_images = level_views
                        .clone()
                        .slice_dim(1, eye_idx..eye_idx + 1)
                        .reshape([batch, channels, height, width]);
                    let target = patchify(view_images, patch_size);
                    visible_tokens.push(tokens);
                    visible_targets.push(target);
                }
                if !visible_tokens.is_empty() {
                    let pred_visible = self.recon.forward(Tensor::cat(visible_tokens, 0));
                    let target_visible = Tensor::cat(visible_targets, 0);
                    let diff = pred_visible.clone() - target_visible;
                    let loss_sum_level = diff.powf_scalar(2.0).sum();
                    let [vis_batch, vis_tokens, vis_dim] = pred_visible.shape().dims::<3>();
                    let mask_sum_level = Tensor::<B, 1>::ones([1], &device)
                        .mul_scalar((vis_batch * vis_tokens * vis_dim) as f32);
                    visible_loss_sum = Some(match visible_loss_sum {
                        Some(accum) => accum + loss_sum_level,
                        None => loss_sum_level,
                    });
                    visible_mask_sum = Some(match visible_mask_sum {
                        Some(accum) => accum + mask_sum_level,
                        None => mask_sum_level,
                    });
                }
            }

            if capture_artifacts && level_idx + 1 == level_count && batch > 0 {
                let pred_first = pred_patches.slice_dim(0, 0..batch);
                let target_first = target_patches.slice_dim(0, 0..batch);
                let mask_first = mask.slice_dim(0, 0..batch);
                let loss_mask_first = loss_mask.slice_dim(0, 0..batch);
                let mask_expanded = mask_first.clone().unsqueeze_dim::<3>(2);
                let loss_mask_expanded = loss_mask_first.clone().unsqueeze_dim::<3>(2);
                let keep = mask_expanded.clone().mul_scalar(-1.0).add_scalar(1.0);
                let masked_patches = target_first.clone().mul(keep.clone());
                let recon_patches = if loss_on_all_patches {
                    pred_first.clone()
                } else {
                    pred_first.clone().mul(mask_expanded.clone()) + target_first.clone().mul(keep)
                };
                let masked_view = unpatchify(masked_patches, patch_size, height, width, channels);
                let recon_view = unpatchify(recon_patches, patch_size, height, width, channels);
                let residual = (pred_first - target_first).mul(loss_mask_expanded);
                let input_view = target_images.clone();
                artifacts = Some((vec![input_view, masked_view, recon_view], residual));
            }
        }

        let zero = Tensor::<B, 1>::zeros([1], &device);
        let loss_sum = loss_sum.unwrap_or_else(|| zero.clone());
        let mask_sum = mask_sum.unwrap_or_else(|| zero.clone());
        let visible_loss_sum = visible_loss_sum.unwrap_or_else(|| zero.clone());
        let visible_mask_sum = visible_mask_sum.unwrap_or(zero.clone());
        let masked_full_loss_sum = masked_full_loss_sum.unwrap_or_else(|| zero.clone());
        (
            loss_sum,
            mask_sum,
            masked_full_loss_sum,
            visible_loss_sum,
            visible_mask_sum,
            artifacts,
        )
    }
}
