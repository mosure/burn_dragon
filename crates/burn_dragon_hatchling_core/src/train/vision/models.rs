use crate::train::prelude::*;

#[derive(Module, Debug)]
pub(crate) struct VisionDistillModel<B: BackendTrait> {
    pub(crate) model: VisionDragonHatchling<B>,
    pub(crate) loss: VisionDistillationLossConfig,
    pub(crate) teacher: Option<DinoVisionTransformer<B>>,
    #[module(ignore)]
    pub(crate) rollout: VisionRollout,
}

impl<B: BackendTrait> VisionDistillModel<B> {
    pub(crate) fn new(
        model: VisionDragonHatchling<B>,
        loss: VisionDistillationLossConfig,
        teacher: Option<DinoVisionTransformer<B>>,
        rollout: VisionRollout,
    ) -> Self {
        Self {
            model,
            loss,
            teacher,
            rollout,
        }
    }
}

#[derive(Module, Debug)]
pub(crate) struct VisionProbe<B: BackendTrait> {
    pub(crate) norm: LayerNorm<B>,
    pub(crate) head: Linear<B>,
}

impl<B: BackendTrait> VisionProbe<B> {
    pub(crate) fn new(embed_dim: usize, num_classes: usize, device: &B::Device) -> Self {
        let norm = LayerNormConfig::new(embed_dim).init(device);
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
    pub(crate) norm: LayerNorm<B>,
    pub(crate) hidden: Option<Linear<B>>,
    pub(crate) out: Linear<B>,
}

impl<B: BackendTrait> VisionReconstructionHead<B> {
    pub(crate) fn new(embed_dim: usize, hidden_dim: usize, patch_dim: usize, device: &B::Device) -> Self {
        let norm = LayerNormConfig::new(embed_dim).init(device);
        let hidden = if hidden_dim > 0 {
            Some(LinearConfig::new(embed_dim, hidden_dim).init(device))
        } else {
            None
        };
        let out_dim = if hidden.is_some() {
            hidden_dim.max(1)
        } else {
            embed_dim.max(1)
        };
        let out = LinearConfig::new(out_dim, patch_dim.max(1)).init(device);
        Self { norm, hidden, out }
    }

    pub(crate) fn forward<const D: usize>(&self, tokens: Tensor<B, D>) -> Tensor<B, D> {
        let tokens = self.norm.forward(tokens);
        let tokens = if let Some(hidden) = &self.hidden {
            activation::gelu(hidden.forward(tokens))
        } else {
            tokens
        };
        self.out.forward(tokens)
    }
}

#[derive(Module, Debug)]
pub(crate) struct VisionSaccadeHead<B: BackendTrait> {
    pub(crate) norm: LayerNorm<B>,
    pub(crate) proj: Linear<B>,
}

impl<B: BackendTrait> VisionSaccadeHead<B> {
    pub(crate) fn new(embed_dim: usize, device: &B::Device) -> Self {
        let norm = LayerNormConfig::new(embed_dim).init(device);
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
    pub(crate) norm: LayerNorm<B>,
    pub(crate) proj: Linear<B>,
}

impl<B: BackendTrait> VisionSaccadeProjection<B> {
    pub(crate) fn new(embed_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        let norm = LayerNormConfig::new(embed_dim).init(device);
        let proj = LinearConfig::new(embed_dim, out_dim).init(device);
        Self { norm, proj }
    }

    pub(crate) fn forward(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let tokens = self.norm.forward(tokens);
        self.proj.forward(tokens)
    }
}

#[derive(Module, Debug)]
pub(crate) struct VisionLejepaModel<B: BackendTrait> {
    pub(crate) model: VisionDragonHatchling<B>,
    pub(crate) probe: VisionProbe<B>,
    pub(crate) probe_loss: burn::nn::loss::CrossEntropyLoss<B>,
    pub(crate) recon: Option<VisionReconstructionHead<B>>,
    pub(crate) mask_token: Option<Param<Tensor<B, 2>>>,
    pub(crate) config: VisionLejepaConfig,
    #[module(ignore)]
    pub(crate) rollout: VisionRollout,
}

pub(crate) struct VisionLejepaLosses<B: BackendTrait> {
    pub(crate) total: Tensor<B, 1>,
    pub(crate) inv: Tensor<B, 1>,
    pub(crate) sigreg: Tensor<B, 1>,
    pub(crate) recon: Tensor<B, 1>,
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
        model: VisionDragonHatchling<B>,
        config: VisionLejepaConfig,
        embed_dim: usize,
        num_classes: usize,
        rollout: VisionRollout,
        recon_patch_dim: usize,
        device: &B::Device,
    ) -> Self {
        let probe = VisionProbe::new(embed_dim, num_classes, device);
        let probe_loss = CrossEntropyLossConfig::new().init(device);
        let recon_weight = config.loss.recon.weight;
        let recon = if recon_weight > 0.0 {
            if recon_patch_dim == 0 {
                None
            } else {
                Some(VisionReconstructionHead::new(
                    embed_dim,
                    config.loss.recon.hidden_dim,
                    recon_patch_dim,
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
        Self {
            model,
            probe,
            probe_loss,
            recon,
            mask_token,
            config,
            rollout,
        }
    }

    pub(crate) fn forward_losses(
        &self,
        batch: ImageNetBatch<B>,
        steps: usize,
        backprop_steps: usize,
        randomize_mask: bool,
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
        let mut artifact_views = None;
        let mut probe_primary = None;
        let mut probe_embed = None;
        let mut recon_loss_sum = Tensor::<B, 1>::zeros([1], &device);
        let mut recon_mask_sum = Tensor::<B, 1>::zeros([1], &device);
        let recon_enabled = self.recon.is_some();

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
                    heatmap_source =
                        Some(output.patch_tokens.clone().slice_dim(0, 0..batch));
                }
            }

            if recon_enabled {
                let (loss_sum, mask_sum, artifacts) =
                    self.recon_group_loss(
                        &collected.global,
                        steps,
                        backprop_steps,
                        true,
                        randomize_mask,
                    );
                recon_loss_sum = recon_loss_sum + loss_sum;
                recon_mask_sum = recon_mask_sum + mask_sum;
                if let Some((views, residual)) = artifacts {
                    artifact_views = Some(views);
                    heatmap_source = Some(residual);
                }
            }
            proj_groups.push(output.proj);
            embed_groups.push(output.embed);
        }
        if !collected.local.is_empty() {
            let output = self.forward_view_group(&collected.local, steps, backprop_steps);
            if heatmap_source.is_none() {
                let [_, batch, _] = output.embed.shape().dims::<3>();
                heatmap_source =
                    Some(output.patch_tokens.clone().slice_dim(0, 0..batch));
            }
            if recon_enabled {
                let (loss_sum, mask_sum, _) = self.recon_group_loss(
                    &collected.local,
                    steps,
                    backprop_steps,
                    false,
                    randomize_mask,
                );
                recon_loss_sum = recon_loss_sum + loss_sum;
                recon_mask_sum = recon_mask_sum + mask_sum;
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
        let zero = Tensor::<B, 1>::zeros([1], &device);
        let (inv, sigreg, mut total) = if self.config.loss.lejepa.enabled {
            let inv = lejepa_invariance_loss(proj.clone());
            let sigreg = lejepa_sigreg_loss(proj.clone(), &self.config.loss.lejepa);
            let lambda = self.config.loss.lejepa.lambda.clamp(0.0, 1.0);
            let total = inv.clone().mul_scalar(1.0 - lambda) + sigreg.clone().mul_scalar(lambda);
            (inv, sigreg, total)
        } else {
            (zero.clone(), zero.clone(), zero.clone())
        };
        let recon = if recon_enabled {
            let denom = recon_mask_sum.clone().add_scalar(LEJEPA_EPS);
            let recon = recon_loss_sum / denom;
            let weight = self.config.loss.recon.weight.max(0.0);
            total = total + recon.clone().mul_scalar(weight);
            recon
        } else {
            zero.clone()
        };

        let probe_source = probe_embed.as_ref().unwrap_or(&embed);
        let [view_count, batch, embed_dim] = probe_source.shape().dims::<3>();
        let embed_flat = probe_source
            .clone()
            .reshape([view_count * batch, embed_dim])
            .detach();
        let labels_flat = labels.clone().repeat_dim(0, view_count);
        let probe_logits = self.probe.forward(embed_flat);
        let probe_loss = self.probe_loss.forward(probe_logits.clone(), labels_flat.clone());
        let probe_pred = probe_logits
            .clone()
            .argmax(1)
            .reshape([view_count * batch]);
        let probe_acc = probe_pred
            .equal(labels_flat)
            .float()
            .mean();

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
        let artifacts = build_lejepa_artifacts(
            &self.config,
            &artifact_views,
            None,
            heatmap_source,
            probe_primary,
            Some(labels),
            legend,
        );

        VisionLejepaLosses {
            total,
            inv,
            sigreg,
            recon,
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
    ) -> (
        Tensor<B, 1>,
        Tensor<B, 1>,
        Option<(Vec<Tensor<B, 4>>, Tensor<B, 3>)>,
    ) {
        let recon = match &self.recon {
            Some(recon) => recon,
            None => {
                let device = views
                    .get(0)
                    .map(|view| view.device())
                    .unwrap_or_default();
                let zero = Tensor::<B, 1>::zeros([1], &device);
                return (zero.clone(), zero, None);
            }
        };
        if views.is_empty() {
            let device = <B as BackendTrait>::Device::default();
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero, None);
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
            return (zero.clone(), zero, None);
        }
        let patch_size = height / grid_h;
        if patch_size == 0 || height % grid_h != 0 || width % grid_w != 0 {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero, None);
        }

        let target_patches = patchify(stacked, patch_size);
        let mask = sample_patch_mask(
            &device,
            total,
            tokens,
            self.config.loss.recon.mask_ratio,
            randomize_mask,
        );
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
            return (zero.clone(), zero, None);
        }

        let diff = pred_patches.clone() - target_patches.clone();
        let loss_sum = diff
            .powf_scalar(2.0)
            .mul(mask_expanded.clone())
            .sum();
        let mask_sum = mask.clone().sum().mul_scalar(patch_dim as f32);

        let artifacts = if capture_artifacts && batch > 0 {
            let pred_first = pred_patches.slice_dim(0, 0..batch);
            let target_first = target_patches.slice_dim(0, 0..batch);
            let mask_first = mask.slice_dim(0, 0..batch);
            let mask_expanded = mask_first.clone().unsqueeze_dim::<3>(2);
            let keep = mask_expanded.clone().mul_scalar(-1.0).add_scalar(1.0);
            let masked_patches = target_first.clone().mul(keep.clone());
            let recon_patches =
                pred_first.clone().mul(mask_expanded.clone()) + target_first.clone().mul(keep);
            let masked_view = unpatchify(masked_patches, patch_size, height, width, channels);
            let recon_view = unpatchify(recon_patches, patch_size, height, width, channels);
            let residual = (pred_first - target_first).mul(mask_expanded);
            Some((vec![views[0].clone(), masked_view, recon_view], residual))
        } else {
            None
        };

        (loss_sum, mask_sum, artifacts)
    }

    pub(crate) fn forward_view_group(
        &self,
        views: &[Tensor<B, 4>],
        steps: usize,
        backprop_steps: usize,
    ) -> ViewGroupOutput<B> {
        let view_count = views.len();
        let [batch, _, _, _] = views[0].shape().dims::<4>();
        let stacked = stack_views(views);
        let patch = self.model.patch_embed(stacked);
        let embed_out = self
            .model
            .forward_tokens_embed_steps_rollout(patch.tokens, steps, backprop_steps);
        let cls_embed = embed_out.cls_token;
        let patch_tokens = embed_out.patch_tokens;
        let [total, embed_dim] = cls_embed.shape().dims::<2>();
        debug_assert_eq!(total, view_count * batch, "lejepa embed mismatch");
        let tokens = Tensor::cat(
            vec![cls_embed.clone().unsqueeze_dim::<3>(1), patch_tokens.clone()],
            1,
        );
        let proj_tokens = self.model.project_tokens(tokens);
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

#[derive(Module, Debug)]
pub(crate) struct VisionMaeModel<B: BackendTrait> {
    pub(crate) model: VisionDragonHatchling<B>,
    pub(crate) recon: VisionReconstructionHead<B>,
    pub(crate) mask_token: Param<Tensor<B, 2>>,
    pub(crate) config: VisionMaeConfig,
    #[module(ignore)]
    pub(crate) rollout: VisionRollout,
}

pub(crate) struct VisionMaeLosses<B: BackendTrait> {
    pub(crate) total: Tensor<B, 1>,
    pub(crate) recon: Tensor<B, 1>,
    pub(crate) artifacts: Option<VisionArtifactInput<B>>,
}

impl<B: BackendTrait> VisionMaeModel<B> {
    pub(crate) fn new(
        model: VisionDragonHatchling<B>,
        config: VisionMaeConfig,
        embed_dim: usize,
        rollout: VisionRollout,
        recon_patch_dim: usize,
        device: &B::Device,
    ) -> Self {
        let recon = VisionReconstructionHead::new(
            embed_dim,
            config.loss.recon.hidden_dim,
            recon_patch_dim,
            device,
        );
        let token = Tensor::<B, 2>::random(
            [1, embed_dim.max(1)],
            TensorDistribution::Normal(0.0, 0.02),
            device,
        );
        let mask_token = Param::from_tensor(token);
        Self {
            model,
            recon,
            mask_token,
            config,
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
    ) -> VisionMaeLosses<B> {
        let ImageNetBatch { images, labels, .. } = batch;
        let (loss_sum, mask_sum, artifacts) =
            self.recon_loss(images, steps, backprop_steps, randomize_mask, capture_artifacts);
        let denom = mask_sum.clone().add_scalar(LEJEPA_EPS);
        let recon = loss_sum / denom;
        let total = recon
            .clone()
            .mul_scalar(self.config.loss.recon.weight.max(0.0));

        let artifacts = artifacts.and_then(|(views, residual)| {
            build_lejepa_artifacts(
                &VisionLejepaConfig {
                    artifact_every: self.config.artifact_every,
                    artifact_max_images: self.config.artifact_max_images,
                    artifact_max_views: self.config.artifact_max_views,
                    ..VisionLejepaConfig::default()
                },
                &views,
                None,
                Some(residual),
                None,
                Some(labels),
                Some(vec![
                    "input".to_string(),
                    "masked_input".to_string(),
                    "reconstruction".to_string(),
                ]),
            )
        });

        VisionMaeLosses {
            total,
            recon,
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
    ) -> (
        Tensor<B, 1>,
        Tensor<B, 1>,
        Option<(Vec<Tensor<B, 4>>, Tensor<B, 3>)>,
    ) {
        let device = images.device();
        let [batch, channels, height, width] = images.shape().dims::<4>();
        let patch = self.model.patch_embed_raw(images.clone());
        let [_, tokens, embed_dim] = patch.tokens.shape().dims::<3>();
        let grid_h = patch.grid.height;
        let grid_w = patch.grid.width;
        if grid_h == 0 || grid_w == 0 || grid_h * grid_w != tokens {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero, None);
        }
        let patch_size = height / grid_h;
        if patch_size == 0 || height % grid_h != 0 || width % grid_w != 0 {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero, None);
        }

        let target_patches = patchify(images.clone(), patch_size);
        let mask = sample_patch_mask(
            &device,
            batch,
            tokens,
            self.config.loss.recon.mask_ratio,
            randomize_mask,
        );
        let mask_expanded = mask.clone().unsqueeze_dim::<3>(2);
        let keep = mask_expanded.clone().mul_scalar(-1.0).add_scalar(1.0);
        let mut masked_tokens = patch.tokens.clone().mul(keep.clone());
        let token = self
            .mask_token
            .val()
            .reshape([1, 1, embed_dim])
            .repeat_dim(0, batch)
            .repeat_dim(1, tokens);
        masked_tokens = masked_tokens + token.mul(mask_expanded.clone());
        let masked_tokens = self.model.add_patch_position(masked_tokens, patch.grid);
        let embed_out =
            self.model
                .forward_tokens_embed_steps_rollout(masked_tokens, steps, backprop_steps);

        let pred_patches = self.recon.forward(embed_out.patch_tokens);
        let [total, tokens, patch_dim] = pred_patches.shape().dims::<3>();
        if total == 0 || tokens == 0 || patch_dim == 0 {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero, None);
        }

        let diff = pred_patches.clone() - target_patches.clone();
        let loss_sum = diff
            .powf_scalar(2.0)
            .mul(mask_expanded.clone())
            .sum();
        let mask_sum = mask.clone().sum().mul_scalar(patch_dim as f32);

        let artifacts = if capture_artifacts && batch > 0 {
            let pred_first = pred_patches.slice_dim(0, 0..batch);
            let target_first = target_patches.slice_dim(0, 0..batch);
            let mask_first = mask.slice_dim(0, 0..batch);
            let mask_expanded = mask_first.clone().unsqueeze_dim::<3>(2);
            let keep = mask_expanded.clone().mul_scalar(-1.0).add_scalar(1.0);
            let masked_patches = target_first.clone().mul(keep.clone());
            let recon_patches =
                pred_first.clone().mul(mask_expanded.clone()) + target_first.clone().mul(keep);
            let masked_view = unpatchify(masked_patches, patch_size, height, width, channels);
            let recon_view = unpatchify(recon_patches, patch_size, height, width, channels);
            let residual = (pred_first - target_first).mul(mask_expanded);
            Some((vec![images.clone(), masked_view, recon_view], residual))
        } else {
            None
        };

        (loss_sum, mask_sum, artifacts)
    }
}


