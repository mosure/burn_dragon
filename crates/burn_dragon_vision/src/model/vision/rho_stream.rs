use super::*;

type RhoStreamStateShape = [usize; 5];

impl<B: Backend> VisionDragon<B> {
    pub(super) fn resolve_rho_stream_grid(&self, patch_tokens: usize) -> PatchGrid {
        let patch_tokens = patch_tokens.max(1);
        let grid_height = self.grid_height.max(1);
        let grid_width = self.grid_width.max(1);
        if grid_height * grid_width == patch_tokens {
            return PatchGrid {
                height: grid_height,
                width: grid_width,
            };
        }

        let side = (patch_tokens as f64).sqrt() as usize;
        if side * side == patch_tokens {
            return PatchGrid {
                height: side,
                width: side,
            };
        }

        PatchGrid {
            height: patch_tokens,
            width: 1,
        }
    }

    pub(super) fn rho_stream_rollout_executor_mode(&self) -> RhoStreamRolloutExecutorMode {
        if self.rho_stream.wgpu_rollout_fused && self.rho_stream_wgpu_forward_enabled() {
            return RhoStreamRolloutExecutorMode::WgpuFused;
        }
        RhoStreamRolloutExecutorMode::HostLoop
    }

    pub(super) fn rho_stream_wgpu_forward_enabled(&self) -> bool {
        self.rho_stream.wgpu_forward_kernel && supports_local_grid_rho_backend::<B>()
    }

    pub(super) fn resolve_rho_stream_state(
        &self,
        rho_state: Option<&Tensor<B, 5>>,
        expected: RhoStreamStateShape,
        device: &B::Device,
    ) -> Tensor<B, 5> {
        match rho_state {
            Some(existing) if existing.shape().dims::<5>() == expected => existing.clone(),
            _ => Tensor::<B, 5>::zeros(expected, device),
        }
    }

    pub(super) fn rho_stream_patch_io(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, bool) {
        let time = query.shape().dims::<4>()[2];
        if self.use_cls_token && time > 1 {
            (
                query.clone().slice_dim(2, 1..time),
                value.clone().slice_dim(2, 1..time),
                true,
            )
        } else {
            (query, value, false)
        }
    }

    pub(super) fn rho_stream_with_cls_context(
        &self,
        patch_context: Tensor<B, 4>,
        batch: usize,
        heads: usize,
        embd: usize,
        has_cls: bool,
    ) -> Tensor<B, 4> {
        if has_cls {
            let cls_context = patch_context
                .clone()
                .mean_dim(2)
                .reshape([batch, heads, 1, embd]);
            Tensor::cat(vec![cls_context, patch_context], 2)
        } else {
            patch_context
        }
    }

    pub(super) fn resolve_rho_stream_neighborhood(&self) -> LocalGridNeighborhood {
        if self.rho_stream.local_diagonals {
            LocalGridNeighborhood::moore(self.rho_stream.local_radius)
        } else {
            LocalGridNeighborhood::von_neumann(self.rho_stream.local_radius)
        }
        .with_self_edges(self.rho_stream.local_self)
    }

    #[allow(dead_code)]
    pub(super) fn rho_stream_attention_reference_with_state(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        state: Tensor<B, 5>,
    ) -> (Tensor<B, 4>, Tensor<B, 5>) {
        let device = query.device();
        self.rho_stream_attention_reference_with_state_decay_mode(
            query,
            value,
            state,
            self.scalar_cellular_decay(self.rho_stream.decay.clamp(0.0, 1.0), &device),
            StructuredStepMode::Predict,
        )
    }

    #[allow(dead_code)]
    pub(super) fn rho_stream_attention_reference_with_state_decay(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        state: Tensor<B, 5>,
        decay: Tensor<B, 1>,
    ) -> (Tensor<B, 4>, Tensor<B, 5>) {
        self.rho_stream_attention_reference_with_state_decay_mode(
            query,
            value,
            state,
            decay,
            StructuredStepMode::Predict,
        )
    }

    pub(super) fn rho_stream_attention_reference_with_state_decay_mode(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        state: Tensor<B, 5>,
        decay: Tensor<B, 1>,
        mode: StructuredStepMode,
    ) -> (Tensor<B, 4>, Tensor<B, 5>) {
        let [batch, heads, _time, _latent] = query.shape().dims::<4>();
        let embd = value.shape().dims::<4>()[3];
        let (patch_query, patch_value, has_cls) = self.rho_stream_patch_io(query, value);
        let (patch_query, patch_value) =
            self.apply_cellular_recurrent_mode(patch_query, patch_value, mode);
        let patch_tokens = patch_query.shape().dims::<4>()[2];
        let decay = decay.reshape([1, heads, 1, 1, 1]);
        let patch_context = self.rho_stream_local_read(
            state.clone(),
            patch_query.clone(),
            self.resolve_rho_stream_grid(patch_tokens),
        );
        let next_state = state.mul(decay) + self.rho_stream_outer_product(patch_query, patch_value);
        let context = self.rho_stream_with_cls_context(patch_context, batch, heads, embd, has_cls);

        (context, next_state)
    }

    #[allow(dead_code)]
    pub(super) fn rho_stream_attention(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        rho_state: &mut Option<Tensor<B, 5>>,
    ) -> Tensor<B, 4> {
        let device = query.device();
        self.rho_stream_attention_with_decay(
            query,
            value,
            rho_state,
            self.scalar_cellular_decay(self.rho_stream.decay.clamp(0.0, 1.0), &device),
            StructuredStepMode::Predict,
        )
    }

    pub(super) fn rho_stream_attention_with_decay(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        rho_state: &mut Option<Tensor<B, 5>>,
        decay: Tensor<B, 1>,
        mode: StructuredStepMode,
    ) -> Tensor<B, 4> {
        if self.rho_stream_wgpu_forward_enabled() {
            return self
                .rho_stream_attention_fused_with_decay(query, value, rho_state, decay, mode);
        }
        self.rho_stream_attention_reference_with_decay(query, value, rho_state, decay, mode)
    }

    #[allow(dead_code)]
    pub(super) fn rho_stream_attention_reference(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        rho_state: &mut Option<Tensor<B, 5>>,
    ) -> Tensor<B, 4> {
        let device = query.device();
        self.rho_stream_attention_reference_with_decay(
            query,
            value,
            rho_state,
            self.scalar_cellular_decay(self.rho_stream.decay.clamp(0.0, 1.0), &device),
            StructuredStepMode::Predict,
        )
    }

    pub(super) fn rho_stream_attention_reference_with_decay(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        rho_state: &mut Option<Tensor<B, 5>>,
        decay: Tensor<B, 1>,
        mode: StructuredStepMode,
    ) -> Tensor<B, 4> {
        let [batch, heads, time, latent] = query.shape().dims::<4>();
        let embd = value.shape().dims::<4>()[3];
        let device = value.device();
        if batch == 0 || heads == 0 || time == 0 || latent == 0 || embd == 0 {
            return Tensor::<B, 4>::zeros(
                [batch.max(1), heads.max(1), time.max(1), embd.max(1)],
                &device,
            );
        }

        let patch_tokens = if self.use_cls_token && time > 1 {
            time - 1
        } else {
            time
        };
        if patch_tokens == 0 {
            return Tensor::<B, 4>::zeros([batch, heads, time, embd], &device);
        }

        let state = self.resolve_rho_stream_state(
            rho_state.as_ref(),
            [batch, heads, patch_tokens, latent, embd],
            &device,
        );
        let (context, next_state) = self
            .rho_stream_attention_reference_with_state_decay_mode(query, value, state, decay, mode);
        *rho_state = Some(next_state);
        context
    }

    #[allow(dead_code)]
    pub(super) fn rho_stream_attention_fused(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        rho_state: &mut Option<Tensor<B, 5>>,
    ) -> Tensor<B, 4> {
        let device = query.device();
        self.rho_stream_attention_fused_with_decay(
            query,
            value,
            rho_state,
            self.scalar_cellular_decay(self.rho_stream.decay.clamp(0.0, 1.0), &device),
            StructuredStepMode::Predict,
        )
    }

    pub(super) fn rho_stream_attention_fused_with_decay(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        rho_state: &mut Option<Tensor<B, 5>>,
        decay: Tensor<B, 1>,
        mode: StructuredStepMode,
    ) -> Tensor<B, 4> {
        self.rho_stream_attention_fused_with_decay_plan(query, value, rho_state, decay, mode, None)
    }

    pub(super) fn rho_stream_attention_fused_with_decay_plan(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        rho_state: &mut Option<Tensor<B, 5>>,
        decay: Tensor<B, 1>,
        mode: StructuredStepMode,
        fused_plan: Option<&CompiledLocalGridRhoPlan<B>>,
    ) -> Tensor<B, 4> {
        let [batch, heads, time, latent] = query.shape().dims::<4>();
        let embd = value.shape().dims::<4>()[3];
        let device = value.device();
        if batch == 0 || heads == 0 || time == 0 || latent == 0 || embd == 0 {
            return Tensor::<B, 4>::zeros(
                [batch.max(1), heads.max(1), time.max(1), embd.max(1)],
                &device,
            );
        }

        let (patch_query, patch_value, has_cls) =
            self.rho_stream_patch_io(query.clone(), value.clone());
        let (patch_query, patch_value) =
            self.apply_cellular_recurrent_mode(patch_query, patch_value, mode);
        let patch_tokens = patch_query.shape().dims::<4>()[2];
        if patch_tokens == 0 {
            return Tensor::<B, 4>::zeros([batch, heads, time, embd], &device);
        }

        let state = self.resolve_rho_stream_state(
            rho_state.as_ref(),
            [batch, heads, patch_tokens, latent, embd],
            &device,
        );
        let grid = self.resolve_rho_stream_grid(patch_tokens);
        let neighborhood = self.resolve_rho_stream_neighborhood();
        let fused = if let Some(plan) = fused_plan {
            try_fused_local_grid_rho_attention_wgpu_head_decay_with_plan::<B>(
                &patch_query,
                &patch_value,
                Some(&state),
                &decay,
                plan,
            )
        } else {
            try_fused_local_grid_rho_attention_wgpu_head_decay::<B>(
                &patch_query,
                &patch_value,
                Some(&state),
                LocalGridShape2d::new(grid.height, grid.width),
                neighborhood,
                &decay,
            )
        };

        if let Some(output) = fused {
            let fused_context =
                self.rho_stream_with_cls_context(output.context, batch, heads, embd, has_cls);
            if B::ad_enabled(&query.device()) {
                let (reference_context, reference_rho) = self
                    .rho_stream_attention_reference_with_state_decay_mode(
                        query, value, state, decay, mode,
                    );
                let context =
                    reference_context.clone() + fused_context - reference_context.detach();
                let rho = reference_rho.clone() + output.rho - reference_rho.detach();
                *rho_state = Some(rho);
                return context;
            }
            *rho_state = Some(output.rho);
            return fused_context;
        }

        let (context, next_state) = self
            .rho_stream_attention_reference_with_state_decay_mode(query, value, state, decay, mode);
        *rho_state = Some(next_state);
        context
    }

    pub(super) fn rho_stream_outer_product(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
    ) -> Tensor<B, 5> {
        let query = query.unsqueeze_dim::<5>(4);
        let value = value.unsqueeze_dim::<5>(3);
        query.mul(value)
    }

    pub(super) fn rho_stream_local_read(
        &self,
        rho_state: Tensor<B, 5>,
        query: Tensor<B, 4>,
        grid: PatchGrid,
    ) -> Tensor<B, 4> {
        let [batch, heads, patch_tokens, latent, embd] = rho_state.shape().dims::<5>();
        if batch == 0 || heads == 0 || patch_tokens == 0 || latent == 0 || embd == 0 {
            return Tensor::<B, 4>::zeros(
                [batch.max(1), heads.max(1), patch_tokens.max(1), embd.max(1)],
                &rho_state.device(),
            );
        }

        let device = rho_state.device();
        let radius = self.rho_stream.local_radius as isize;
        let mut outputs = Vec::with_capacity(patch_tokens);

        for target in 0..patch_tokens {
            let ty = target / grid.width.max(1);
            let tx = target % grid.width.max(1);
            let query_t = query
                .clone()
                .slice_dim(2, target..target + 1)
                .unsqueeze_dim::<5>(4);
            let mut context_t = Tensor::<B, 4>::zeros([batch, heads, 1, embd], &device);

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    if dy == 0 && dx == 0 && !self.rho_stream.local_self {
                        continue;
                    }
                    if !self.rho_stream.local_diagonals && dy != 0 && dx != 0 {
                        continue;
                    }
                    let sy = ty as isize + dy;
                    let sx = tx as isize + dx;
                    if sy < 0 || sy >= grid.height as isize || sx < 0 || sx >= grid.width as isize {
                        continue;
                    }
                    let source = sy as usize * grid.width + sx as usize;
                    let source_state = rho_state.clone().slice_dim(2, source..source + 1);
                    let msg = source_state
                        .mul(query_t.clone())
                        .sum_dims_squeeze::<4, usize>(&[3]);
                    context_t = context_t + msg;
                }
            }

            outputs.push(context_t);
        }

        Tensor::cat(outputs, 2)
    }
}
