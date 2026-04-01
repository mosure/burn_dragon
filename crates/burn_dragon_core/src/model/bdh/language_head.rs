use super::*;

impl<B: Backend> BDH<B> {
    pub fn language_token_losses_from_hidden(
        &self,
        hidden: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2> {
        match self.nca_factorized_head_tables.0.as_ref() {
            None => self.language_token_losses_from_logits(self.project_hidden_to_logits(hidden), targets),
            Some(tables) => {
                self.nca_factorized_language_token_losses_from_hidden(hidden, targets, tables)
            }
        }
    }

    pub fn language_loss_from_hidden(
        &self,
        hidden: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1> {
        match self.nca_factorized_head_tables.0.as_ref() {
            None => self.language_loss_from_logits(self.project_hidden_to_logits(hidden), targets),
            Some(tables) => self.nca_factorized_language_loss_from_hidden(hidden, targets, tables),
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

        let patch_logits = hidden_flat
            .clone()
            .matmul(
                self.nca_factorized_lm_head
                    .as_ref()
                    .expect("factorized NCA head weights missing")
                    .val(),
            )
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
                .slice([0..token_count, cell_idx..cell_idx + 1, 0..tables.state_count])
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
                    self.nca_special_lm_head
                        .as_ref()
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
        token_nll.sum().div(supported.sum().clamp_min(1.0)).reshape([1])
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
