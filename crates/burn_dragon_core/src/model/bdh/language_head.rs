use super::*;
use burn::module::{ModuleVisitor, ParamId};
use std::collections::HashSet;

pub(crate) enum LanguageHeadRuntimeRef<'a, B: Backend> {
    StandardTokenClassification {
        lm_head: &'a Param<Tensor<B, 2>>,
    },
    NcaFactorizedPatch {
        factorized_lm_head: &'a Param<Tensor<B, 2>>,
        special_lm_head: Option<&'a Param<Tensor<B, 2>>>,
        tables: &'a NcaFactorizedHeadTables,
    },
}

pub(crate) struct LanguageHeadDeployScaffold<B: Backend> {
    pub(crate) lm_head: Option<Param<Tensor<B, 2>>>,
    pub(crate) nca_factorized_lm_head: Option<Param<Tensor<B, 2>>>,
    pub(crate) nca_special_lm_head: Option<Param<Tensor<B, 2>>>,
}

#[derive(Default)]
struct ParamIdCollector {
    ids: HashSet<ParamId>,
}

impl<B: Backend> ModuleVisitor<B> for ParamIdCollector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.ids.insert(param.id);
    }
}

impl<B: Backend> BDH<B> {
    fn collect_param_ids_from_module<M: Module<B>>(module: &M) -> HashSet<ParamId> {
        let mut collector = ParamIdCollector::default();
        module.visit(&mut collector);
        collector.ids
    }

    fn collect_optional_param_ids_from_module<M: Module<B>>(
        module: Option<&M>,
    ) -> HashSet<ParamId> {
        module
            .map(Self::collect_param_ids_from_module)
            .unwrap_or_default()
    }

    pub fn language_module_lr_scale_param_ids(
        &self,
        target: LanguageModuleLrScaleTarget,
    ) -> Vec<ParamId> {
        let embedding = Self::collect_param_ids_from_module(&self.embed);
        let normalization = Self::collect_param_ids_from_module(&self.norm);
        let mut output_head = HashSet::new();
        if let Some(lm_head) = self.lm_head.as_ref() {
            output_head.insert(lm_head.id);
        }
        if let Some(factorized) = self.nca_factorized_lm_head.as_ref() {
            output_head.insert(factorized.id);
        }
        if let Some(special) = self.nca_special_lm_head.as_ref() {
            output_head.insert(special.id);
        }
        let shared_lowrank_encoder = HashSet::from([self.encoder.id, self.encoder_v.id]);
        let shared_lowrank_decoder = HashSet::from([self.decoder.id]);
        let shared_lowrank_decay = HashSet::from([self.rwkv_time_decay.id]);
        let attention = Self::collect_param_ids_from_module(&self.attention);
        let mamba = Self::collect_optional_param_ids_from_module(self.mamba.as_ref());
        let mut residual_modules =
            Self::collect_optional_param_ids_from_module(self.mhc_shared.as_ref());
        residual_modules.extend(Self::collect_optional_param_ids_from_module(
            self.attention_residual_shared.as_ref(),
        ));
        residual_modules.extend(Self::collect_optional_param_ids_from_module(
            self.block_attention_residual_shared.as_ref(),
        ));

        let ids = match target {
            LanguageModuleLrScaleTarget::Embedding => embedding,
            LanguageModuleLrScaleTarget::Normalization => normalization,
            LanguageModuleLrScaleTarget::OutputHead => output_head,
            LanguageModuleLrScaleTarget::SharedLowrankEncoder => shared_lowrank_encoder,
            LanguageModuleLrScaleTarget::SharedLowrankDecoder => shared_lowrank_decoder,
            LanguageModuleLrScaleTarget::SharedLowrankDecay => shared_lowrank_decay,
            LanguageModuleLrScaleTarget::Attention => attention,
            LanguageModuleLrScaleTarget::Mamba => mamba,
            LanguageModuleLrScaleTarget::ResidualModules => residual_modules,
            LanguageModuleLrScaleTarget::OtherBackbone => {
                let mut remaining = Self::collect_param_ids_from_module(self);
                for excluded in embedding
                    .into_iter()
                    .chain(normalization)
                    .chain(output_head)
                    .chain(shared_lowrank_encoder)
                    .chain(shared_lowrank_decoder)
                    .chain(shared_lowrank_decay)
                    .chain(attention)
                    .chain(mamba)
                    .chain(residual_modules)
                {
                    remaining.remove(&excluded);
                }
                remaining
            }
        };

        ids.into_iter().collect()
    }

    pub fn transfer_interface_surface_param_ids(
        &self,
        preserve_fresh_decoder: bool,
        preserve_fresh_norm: bool,
    ) -> Vec<ParamId> {
        let mut ids = Self::collect_param_ids_from_module(&self.embed);

        if preserve_fresh_norm {
            ids.extend(Self::collect_param_ids_from_module(&self.norm));
        }
        if preserve_fresh_decoder {
            ids.insert(self.decoder.id);
        }
        if let Some(lm_head) = self.lm_head.as_ref() {
            ids.insert(lm_head.id);
        }
        if let Some(factorized) = self.nca_factorized_lm_head.as_ref() {
            ids.insert(factorized.id);
        }
        if let Some(special) = self.nca_special_lm_head.as_ref() {
            ids.insert(special.id);
        }

        ids.into_iter().collect()
    }

    pub fn transferred_backbone_param_ids(
        &self,
        preserve_fresh_decoder: bool,
        preserve_fresh_norm: bool,
    ) -> Vec<ParamId> {
        let mut ids = Self::collect_param_ids_from_module(self);
        for excluded in
            self.transfer_interface_surface_param_ids(preserve_fresh_decoder, preserve_fresh_norm)
        {
            ids.remove(&excluded);
        }
        ids.into_iter().collect()
    }

    fn tensor_rms<const D: usize>(tensor: Tensor<B, D>) -> f32 {
        let values = tensor
            .powf_scalar(2.0)
            .mean()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("bdh rms scalar");
        values.first().copied().unwrap_or(0.0).sqrt()
    }

    fn blend_tensor<const D: usize>(
        source: Tensor<B, D>,
        fresh: Tensor<B, D>,
        alpha: f32,
    ) -> Tensor<B, D> {
        let alpha = alpha.clamp(0.0, 1.0);
        (fresh.mul_scalar(1.0 - alpha) + source.mul_scalar(alpha)).detach()
    }

    fn match_fresh_rms_tensor<const D: usize>(
        source: Tensor<B, D>,
        fresh: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let source_rms = Self::tensor_rms(source.clone());
        let fresh_rms = Self::tensor_rms(fresh);
        if source_rms <= 1.0e-8 || !source_rms.is_finite() || !fresh_rms.is_finite() {
            return source;
        }
        source.mul_scalar(fresh_rms / source_rms).detach()
    }

    fn reset_top_layers_2d(
        source: Tensor<B, 2>,
        fresh: Tensor<B, 2>,
        top_layers: usize,
    ) -> Tensor<B, 2> {
        let [layers, width] = source.shape().dims();
        let reset = top_layers.min(layers);
        let keep = layers.saturating_sub(reset);
        if keep == layers {
            return source;
        }
        if keep == 0 {
            return fresh;
        }
        Tensor::cat(
            vec![
                source.slice([0..keep, 0..width]),
                fresh.slice([keep..layers, 0..width]),
            ],
            0,
        )
        .detach()
    }

    fn reset_top_layers_3d(
        source: Tensor<B, 3>,
        fresh: Tensor<B, 3>,
        top_layers: usize,
    ) -> Tensor<B, 3> {
        let [layers, dim0, dim1] = source.shape().dims();
        let reset = top_layers.min(layers);
        let keep = layers.saturating_sub(reset);
        if keep == layers {
            return source;
        }
        if keep == 0 {
            return fresh;
        }
        Tensor::cat(
            vec![
                source.slice([0..keep, 0..dim0, 0..dim1]),
                fresh.slice([keep..layers, 0..dim0, 0..dim1]),
            ],
            0,
        )
        .detach()
    }

    pub fn load_record_preserving_tokenizer_surfaces(
        &self,
        record: <Self as Module<B>>::Record,
        preserve_input_embedding: bool,
        preserve_output_head: bool,
    ) -> Self {
        let target = self.clone();
        let mut loaded = target.clone().load_record(record);
        let language_head_scaffold =
            preserve_output_head.then(|| target.clone_language_head_deploy_scaffold());
        if preserve_input_embedding {
            loaded.embed = target.embed.clone();
        }
        if let Some(LanguageHeadDeployScaffold {
            lm_head,
            nca_factorized_lm_head,
            nca_special_lm_head,
        }) = language_head_scaffold
        {
            loaded.lm_head = lm_head;
            loaded.nca_factorized_lm_head = nca_factorized_lm_head;
            loaded.nca_special_lm_head = nca_special_lm_head;
        }
        loaded
    }

    pub fn with_tokenizer_surfaces_from(
        &self,
        donor: &Self,
        replace_input_embedding: bool,
        replace_output_head: bool,
    ) -> Self {
        let mut updated = self.clone();
        let language_head_scaffold =
            replace_output_head.then(|| donor.clone_language_head_deploy_scaffold());
        if replace_input_embedding {
            updated.embed = donor.embed.clone();
        }
        if let Some(LanguageHeadDeployScaffold {
            lm_head,
            nca_factorized_lm_head,
            nca_special_lm_head,
        }) = language_head_scaffold
        {
            updated.lm_head = lm_head;
            updated.nca_factorized_lm_head = nca_factorized_lm_head;
            updated.nca_special_lm_head = nca_special_lm_head;
        }
        updated
    }

    pub fn with_output_head_blended_from(&self, donor: &Self, donor_alpha: f32) -> Self {
        let mut updated = self.clone();
        let donor_alpha = donor_alpha.clamp(0.0, 1.0);

        if let (Some(current), Some(donor_head)) =
            (updated.lm_head.as_mut(), donor.lm_head.as_ref())
        {
            *current = Param::from_tensor(Self::blend_tensor(
                donor_head.val(),
                current.val(),
                donor_alpha,
            ));
        }
        if let (Some(current), Some(donor_head)) = (
            updated.nca_factorized_lm_head.as_mut(),
            donor.nca_factorized_lm_head.as_ref(),
        ) {
            *current = Param::from_tensor(Self::blend_tensor(
                donor_head.val(),
                current.val(),
                donor_alpha,
            ));
        }
        if let (Some(current), Some(donor_head)) = (
            updated.nca_special_lm_head.as_mut(),
            donor.nca_special_lm_head.as_ref(),
        ) {
            *current = Param::from_tensor(Self::blend_tensor(
                donor_head.val(),
                current.val(),
                donor_alpha,
            ));
        }

        updated
    }

    pub fn adapted_transferred_backbone(
        &self,
        fresh: &Self,
        backbone_blend_alpha: Option<f32>,
        decoder_blend_alpha: Option<f32>,
        norm_blend_alpha: Option<f32>,
        fresh_top_layers: Option<usize>,
        preserve_fresh_decoder: bool,
        preserve_fresh_norm: bool,
        match_fresh_rms: bool,
    ) -> Self {
        let mut adapted = self.clone();

        if match_fresh_rms {
            adapted.rwkv_time_decay = Param::from_tensor(Self::match_fresh_rms_tensor(
                adapted.rwkv_time_decay.val(),
                fresh.rwkv_time_decay.val(),
            ));
            adapted.encoder = Param::from_tensor(Self::match_fresh_rms_tensor(
                adapted.encoder.val(),
                fresh.encoder.val(),
            ));
            adapted.encoder_v = Param::from_tensor(Self::match_fresh_rms_tensor(
                adapted.encoder_v.val(),
                fresh.encoder_v.val(),
            ));
            adapted.decoder = Param::from_tensor(Self::match_fresh_rms_tensor(
                adapted.decoder.val(),
                fresh.decoder.val(),
            ));
            adapted.norm = adapted.norm.matched_fresh_rms(&fresh.norm);
            adapted.mamba = adapted
                .mamba
                .as_ref()
                .zip(fresh.mamba.as_ref())
                .map(|(source, fresh)| source.matched_fresh_rms(fresh));
        }

        if let Some(alpha) = backbone_blend_alpha {
            adapted.rwkv_time_decay = Param::from_tensor(Self::blend_tensor(
                adapted.rwkv_time_decay.val(),
                fresh.rwkv_time_decay.val(),
                alpha,
            ));
            adapted.encoder = Param::from_tensor(Self::blend_tensor(
                adapted.encoder.val(),
                fresh.encoder.val(),
                alpha,
            ));
            adapted.encoder_v = Param::from_tensor(Self::blend_tensor(
                adapted.encoder_v.val(),
                fresh.encoder_v.val(),
                alpha,
            ));
            adapted.decoder = Param::from_tensor(Self::blend_tensor(
                adapted.decoder.val(),
                fresh.decoder.val(),
                alpha,
            ));
            adapted.norm = adapted.norm.blended_with(&fresh.norm, alpha);
            adapted.mamba = adapted
                .mamba
                .as_ref()
                .zip(fresh.mamba.as_ref())
                .map(|(source, fresh)| source.blended_with(fresh, alpha));
        }

        if let Some(top_layers) = fresh_top_layers {
            adapted.rwkv_time_decay = Param::from_tensor(Self::reset_top_layers_2d(
                adapted.rwkv_time_decay.val(),
                fresh.rwkv_time_decay.val(),
                top_layers,
            ));
            adapted.encoder = Param::from_tensor(Self::reset_top_layers_3d(
                adapted.encoder.val(),
                fresh.encoder.val(),
                top_layers,
            ));
            adapted.encoder_v = Param::from_tensor(Self::reset_top_layers_3d(
                adapted.encoder_v.val(),
                fresh.encoder_v.val(),
                top_layers,
            ));
        }

        if let Some(alpha) = decoder_blend_alpha {
            adapted.decoder = Param::from_tensor(Self::blend_tensor(
                adapted.decoder.val(),
                fresh.decoder.val(),
                alpha,
            ));
        }
        if let Some(alpha) = norm_blend_alpha {
            adapted.norm = adapted.norm.blended_with(&fresh.norm, alpha);
        }

        if preserve_fresh_decoder {
            adapted.decoder = fresh.decoder.clone();
        }
        if preserve_fresh_norm {
            adapted.norm = fresh.norm.clone();
        }

        adapted
    }

    pub(crate) fn language_head_runtime(&self) -> LanguageHeadRuntimeRef<'_, B> {
        match &self.language_head {
            LanguageHeadRuntimeKind::StandardTokenClassification => {
                LanguageHeadRuntimeRef::StandardTokenClassification {
                    lm_head: self
                        .lm_head
                        .as_ref()
                        .expect("flat language-model head weights missing"),
                }
            }
            LanguageHeadRuntimeKind::NcaFactorizedPatch => {
                LanguageHeadRuntimeRef::NcaFactorizedPatch {
                    factorized_lm_head: self
                        .nca_factorized_lm_head
                        .as_ref()
                        .expect("factorized NCA head weights missing"),
                    special_lm_head: self.nca_special_lm_head.as_ref(),
                    tables: self
                        .nca_factorized_head_tables
                        .as_ref()
                        .expect("factorized NCA head tables missing"),
                }
            }
        }
    }

    pub(crate) fn clone_language_head_deploy_scaffold(&self) -> LanguageHeadDeployScaffold<B> {
        match self.language_head_runtime() {
            LanguageHeadRuntimeRef::StandardTokenClassification { lm_head } => {
                LanguageHeadDeployScaffold {
                    lm_head: Some(lm_head.clone()),
                    nca_factorized_lm_head: None,
                    nca_special_lm_head: None,
                }
            }
            LanguageHeadRuntimeRef::NcaFactorizedPatch {
                factorized_lm_head,
                special_lm_head,
                ..
            } => LanguageHeadDeployScaffold {
                lm_head: None,
                nca_factorized_lm_head: Some(factorized_lm_head.clone()),
                nca_special_lm_head: special_lm_head.cloned(),
            },
        }
    }

    pub fn language_token_losses_from_hidden(
        &self,
        hidden: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2> {
        match self.language_head_runtime() {
            LanguageHeadRuntimeRef::StandardTokenClassification { .. } => self
                .language_token_losses_from_logits(self.project_hidden_to_logits(hidden), targets),
            LanguageHeadRuntimeRef::NcaFactorizedPatch { tables, .. } => {
                self.nca_factorized_language_token_losses_from_hidden(hidden, targets, tables)
            }
        }
    }

    pub fn language_loss_from_hidden(
        &self,
        hidden: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1> {
        match self.language_head_runtime() {
            LanguageHeadRuntimeRef::StandardTokenClassification { .. } => {
                self.language_loss_from_logits(self.project_hidden_to_logits(hidden), targets)
            }
            LanguageHeadRuntimeRef::NcaFactorizedPatch { tables, .. } => {
                self.nca_factorized_language_loss_from_hidden(hidden, targets, tables)
            }
        }
    }

    pub fn language_loss_from_logits(
        &self,
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1> {
        self.language_token_losses_from_logits(logits, targets)
            .mean()
            .reshape([1])
    }

    pub fn language_token_losses_from_logits(
        &self,
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2> {
        let [batch, time, vocab] = logits.shape().dims();
        let logits_flat = logits.reshape([batch * time, vocab]);
        let targets_flat = targets.reshape([batch * time]);
        activation::log_softmax(logits_flat, 1)
            .gather(1, targets_flat.reshape([batch * time, 1]))
            .neg()
            .reshape([batch, time])
    }

    fn nca_factorized_language_token_losses_from_hidden(
        &self,
        hidden: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        tables: &NcaFactorizedHeadTables,
    ) -> Tensor<B, 2> {
        let [batch, time, dim] = hidden.shape().dims();
        let token_count = batch * time;
        let device = hidden.device();
        let hidden_flat = hidden.reshape([token_count, dim]);

        let LanguageHeadRuntimeRef::NcaFactorizedPatch {
            factorized_lm_head,
            special_lm_head,
            ..
        } = self.language_head_runtime()
        else {
            panic!("factorized NCA loss requires NCA factorized language head runtime");
        };

        let patch_logits = hidden_flat
            .clone()
            .matmul(factorized_lm_head.val())
            .reshape([token_count, tables.patch_cells, tables.state_count]);

        let targets_flat = targets.reshape([token_count]);
        let patch_mask = self.lookup_f32_table(
            &tables.patch_mask_table,
            targets_flat.clone(),
            &device,
            token_count,
        );
        let special_mask = self.lookup_f32_table(
            &tables.special_mask_table,
            targets_flat.clone(),
            &device,
            token_count,
        );

        let mut patch_nll = Tensor::<B, 1>::zeros([token_count], &device);
        for cell_idx in 0..tables.patch_cells {
            let cell_targets = self.lookup_i64_table(
                &tables.patch_digit_tables[cell_idx],
                targets_flat.clone(),
                &device,
                token_count,
            );
            let cell_logits = patch_logits
                .clone()
                .slice([
                    0..token_count,
                    cell_idx..cell_idx + 1,
                    0..tables.state_count,
                ])
                .reshape([token_count, tables.state_count]);
            let cell_nll = activation::log_softmax(cell_logits, 1)
                .gather(1, cell_targets.reshape([token_count, 1]))
                .neg()
                .reshape([token_count]);
            patch_nll = patch_nll + cell_nll;
        }

        let special_nll = if tables.special_count() > 0 {
            let special_targets = self.lookup_i64_table(
                &tables.special_index_table,
                targets_flat,
                &device,
                token_count,
            );
            let special_logits = hidden_flat
                .matmul(
                    special_lm_head
                        .expect("factorized NCA special-token head weights missing")
                        .val(),
                )
                .reshape([token_count, tables.special_count()]);
            activation::log_softmax(special_logits, 1)
                .gather(1, special_targets.reshape([token_count, 1]))
                .neg()
                .reshape([token_count])
        } else {
            Tensor::<B, 1>::zeros([token_count], &device)
        };

        (patch_nll.mul(patch_mask.clone()) + special_nll.mul(special_mask.clone()))
            .reshape([batch, time])
    }

    fn nca_factorized_language_loss_from_hidden(
        &self,
        hidden: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        tables: &NcaFactorizedHeadTables,
    ) -> Tensor<B, 1> {
        let [batch, time, _dim] = hidden.shape().dims();
        let token_count = batch * time;
        let device = hidden.device();
        let targets_flat = targets.clone().reshape([token_count]);
        let patch_mask = self.lookup_f32_table(
            &tables.patch_mask_table,
            targets_flat.clone(),
            &device,
            token_count,
        );
        let special_mask = self.lookup_f32_table(
            &tables.special_mask_table,
            targets_flat.clone(),
            &device,
            token_count,
        );

        let token_nll = self
            .nca_factorized_language_token_losses_from_hidden(hidden, targets, tables)
            .reshape([token_count]);
        let supported = patch_mask + special_mask;
        token_nll
            .sum()
            .div(supported.sum().clamp_min(1.0))
            .reshape([1])
    }

    fn lookup_i64_table(
        &self,
        values: &[i64],
        indices: Tensor<B, 1, Int>,
        device: &B::Device,
        token_count: usize,
    ) -> Tensor<B, 1, Int> {
        Tensor::<B, 2, Int>::from_data(TensorData::new(values.to_vec(), [1, values.len()]), device)
            .gather(1, indices.reshape([1, token_count]))
            .reshape([token_count])
    }

    fn lookup_f32_table(
        &self,
        values: &[f32],
        indices: Tensor<B, 1, Int>,
        device: &B::Device,
        token_count: usize,
    ) -> Tensor<B, 1> {
        Tensor::<B, 2>::from_data(TensorData::new(values.to_vec(), [1, values.len()]), device)
            .gather(1, indices.reshape([1, token_count]))
            .reshape([token_count])
    }
}
