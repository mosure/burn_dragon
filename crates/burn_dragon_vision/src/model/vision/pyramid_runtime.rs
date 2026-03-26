use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PyramidRolloutExecutorMode {
    HostLoop,
    WgpuLocalPlans,
    WgpuFused,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PyramidRolloutGroupKey {
    backprop_steps: usize,
}

impl<B: Backend> VisionDragon<B> {
    fn advance_structured_temporal_metadata(
        temporal_position: usize,
        prediction_age: usize,
        mode: StructuredStepMode,
    ) -> (usize, usize) {
        let temporal_dt = mode.temporal_dt();
        if temporal_dt > 0 {
            (
                temporal_position.saturating_add(temporal_dt),
                prediction_age.saturating_add(temporal_dt),
            )
        } else if mode.resets_prediction_age() {
            (temporal_position, 0)
        } else {
            (temporal_position, prediction_age)
        }
    }

    fn pack_pyramid_states(
        &self,
        states: &[StructuredTopologyState<B>],
    ) -> StructuredTopologyState<B> {
        let first = states
            .first()
            .expect("packed pyramid state requires at least one input state");
        StructuredTopologyState {
            primary_state: Tensor::cat(
                states
                    .iter()
                    .map(|state| state.primary_state().clone())
                    .collect(),
                0,
            ),
            context_state: Tensor::cat(
                states
                    .iter()
                    .map(|state| state.context_state().clone())
                    .collect(),
                0,
            ),
            rho: BankedRhoState {
                primary_rho: Tensor::cat(
                    states
                        .iter()
                        .map(|state| state.patch_rho().clone())
                        .collect(),
                    0,
                ),
                context_rho: Tensor::cat(
                    states
                        .iter()
                        .map(|state| state.coarse_rho().clone())
                        .collect(),
                    0,
                ),
                global_rho: Tensor::cat(
                    states.iter().map(|state| state.hub_rho().clone()).collect(),
                    0,
                ),
            },
            temporal_position: first.temporal_position,
            prediction_age: first.prediction_age,
        }
    }

    fn split_pyramid_state(
        &self,
        state: StructuredTopologyState<B>,
        batch: usize,
        metadata: &[(usize, usize)],
    ) -> Vec<StructuredTopologyState<B>> {
        let chunk_count = metadata.len();
        let [total_batch, _, _, _] = state.primary_state().shape().dims::<4>();
        assert_eq!(
            total_batch,
            batch.saturating_mul(chunk_count),
            "packed pyramid state should split evenly into {chunk_count} chunks of batch {batch}"
        );
        let primary_state = state.primary_state;
        let context_state = state.context_state;
        let patch_rho = state.rho.primary_rho;
        let coarse_rho = state.rho.context_rho;
        let hub_rho = state.rho.global_rho;
        let mut outputs = Vec::with_capacity(chunk_count);
        for (index, (temporal_position, prediction_age)) in metadata.iter().copied().enumerate() {
            let start = index.saturating_mul(batch);
            let end = start + batch;
            outputs.push(StructuredTopologyState {
                primary_state: primary_state.clone().slice_dim(0, start..end),
                context_state: context_state.clone().slice_dim(0, start..end),
                rho: BankedRhoState {
                    primary_rho: patch_rho.clone().slice_dim(0, start..end),
                    context_rho: coarse_rho.clone().slice_dim(0, start..end),
                    global_rho: hub_rho.clone().slice_dim(0, start..end),
                },
                temporal_position,
                prediction_age,
            });
        }
        outputs
    }

    pub(super) fn forward_pyramid_rollout_schedule_from_cached(
        &self,
        schedule: &[(usize, usize)],
        cached: &BTreeMap<usize, StructuredTopologyState<B>>,
        mode: StructuredStepMode,
    ) -> Vec<(usize, StructuredTopologyState<B>)> {
        let batch = cached
            .get(&0)
            .expect("initial pyramid rollout state")
            .primary_state()
            .shape()
            .dims::<4>()[0];
        let mut grouped: BTreeMap<
            PyramidRolloutGroupKey,
            Vec<(usize, usize, StructuredTopologyState<B>)>,
        > = BTreeMap::new();
        let mut outputs: Vec<Option<(usize, StructuredTopologyState<B>)>> =
            vec![None; schedule.len()];
        for (index, &(step, backprop_steps)) in schedule.iter().enumerate() {
            let start = step.saturating_sub(backprop_steps);
            let start_state = cached.get(&start).expect("rollout start state").clone();
            grouped
                .entry(PyramidRolloutGroupKey { backprop_steps })
                .or_default()
                .push((index, step, start_state));
        }

        for (key, requests) in grouped {
            if requests.len() == 1 {
                let (index, step, start_state) = requests
                    .into_iter()
                    .next()
                    .expect("single pyramid rollout schedule request");
                let final_state = self.forward_pyramid_state_rollout_mode_unbounded(
                    start_state,
                    key.backprop_steps,
                    key.backprop_steps,
                    mode,
                );
                outputs[index] = Some((step, final_state));
                continue;
            }

            let packed_start = self.pack_pyramid_states(
                &requests
                    .iter()
                    .map(|(_, _, state)| state.clone())
                    .collect::<Vec<_>>(),
            );
            let packed_final = self.forward_pyramid_state_rollout_mode_unbounded(
                packed_start,
                key.backprop_steps,
                key.backprop_steps,
                mode,
            );
            let base_state = cached
                .get(&0)
                .expect("initial pyramid rollout state metadata");
            let split_metadata = requests
                .iter()
                .map(|_| {
                    Self::advance_structured_temporal_metadata(
                        base_state.temporal_position,
                        base_state.prediction_age,
                        mode,
                    )
                })
                .collect::<Vec<_>>();
            let split_states = self.split_pyramid_state(packed_final, batch, &split_metadata);
            for ((index, step, _), final_state) in
                requests.into_iter().zip(split_states.into_iter())
            {
                outputs[index] = Some((step, final_state));
            }
        }

        outputs
            .into_iter()
            .map(|output| output.expect("pyramid rollout schedule output"))
            .collect()
    }

    pub(super) fn cache_pyramid_rollout_prefix_states(
        &self,
        initial_state: StructuredTopologyState<B>,
        schedule: &[(usize, usize)],
        mode: StructuredStepMode,
    ) -> BTreeMap<usize, StructuredTopologyState<B>> {
        let base_temporal_position = initial_state.temporal_position;
        let base_prediction_age = initial_state.prediction_age;
        let mut starts = schedule
            .iter()
            .map(|(step, backprop_steps)| step.saturating_sub(*backprop_steps))
            .collect::<Vec<_>>();
        starts.push(0);
        starts.sort_unstable();
        starts.dedup();

        let mut cached = BTreeMap::new();
        cached.insert(0usize, initial_state);
        let mut previous_start = 0usize;
        for &start in starts.iter().skip(1) {
            let previous_state = cached
                .get(&previous_start)
                .expect("rollout prefix state")
                .clone();
            let delta = start.saturating_sub(previous_start);
            let state = self
                .forward_pyramid_state_rollout_mode_unbounded(previous_state, delta, delta, mode)
                .detach();
            let mut state = state;
            state.temporal_position = base_temporal_position;
            state.prediction_age = base_prediction_age;
            cached.insert(start, state);
            previous_start = start;
        }
        cached
    }

    pub(super) fn forward_pyramid_output_schedule_from_cached(
        &self,
        schedule: &[(usize, usize)],
        cached: &BTreeMap<usize, StructuredTopologyState<B>>,
        mode: StructuredStepMode,
    ) -> Vec<(usize, VisionDragonOutput<B>)> {
        let batch = cached
            .get(&0)
            .expect("initial pyramid rollout state")
            .primary_state()
            .shape()
            .dims::<4>()[0];
        let mut grouped: BTreeMap<
            PyramidRolloutGroupKey,
            Vec<(usize, usize, StructuredTopologyState<B>)>,
        > = BTreeMap::new();
        let mut outputs: Vec<Option<(usize, VisionDragonOutput<B>)>> = vec![None; schedule.len()];
        for (index, &(step, backprop_steps)) in schedule.iter().enumerate() {
            let start = step.saturating_sub(backprop_steps);
            let start_state = cached.get(&start).expect("rollout start state").clone();
            grouped
                .entry(PyramidRolloutGroupKey { backprop_steps })
                .or_default()
                .push((index, step, start_state));
        }

        for (key, requests) in grouped {
            if requests.len() == 1 {
                let (index, step, start_state) = requests
                    .into_iter()
                    .next()
                    .expect("single pyramid rollout output request");
                let final_state = self.forward_pyramid_state_rollout_mode_unbounded(
                    start_state,
                    key.backprop_steps,
                    key.backprop_steps,
                    mode,
                );
                let tokens = self.pyramid_readout_tokens(&final_state);
                let projected = self.projection.forward(tokens);
                outputs[index] = Some((step, self.split_output(projected)));
                continue;
            }

            let packed_start = self.pack_pyramid_states(
                &requests
                    .iter()
                    .map(|(_, _, state)| state.clone())
                    .collect::<Vec<_>>(),
            );
            let packed_final = self.forward_pyramid_state_rollout_mode_unbounded(
                packed_start,
                key.backprop_steps,
                key.backprop_steps,
                mode,
            );
            let packed_tokens = self.pyramid_readout_tokens(&packed_final);
            let packed_projected = self.projection.forward(packed_tokens);
            let mut batch_offset = 0usize;
            for (index, step, _) in requests {
                let next_offset = batch_offset + batch;
                let projected = packed_projected
                    .clone()
                    .slice_dim(0, batch_offset..next_offset);
                outputs[index] = Some((step, self.split_output(projected)));
                batch_offset = next_offset;
            }
        }

        outputs
            .into_iter()
            .map(|output| output.expect("pyramid rollout schedule output"))
            .collect()
    }

    /// Initializes the pyramid recurrent state from observed patch tokens.
    ///
    /// Persistence contract:
    /// - `primary_state` and `context_state` are the dense patch/coarse residual streams.
    /// - `rho.primary_rho` and `rho.context_rho` store the local and coarse associative banks.
    /// - `rho.global_rho` / `hub_rho` is the explicit global recurrent memory for the pyramid
    ///   backbone.
    /// - CLS is not part of this persistent state. When higher-level APIs request a CLS token, it
    ///   is derived as a readout-only summary of patch tokens and does not own its own `rho`.
    /// - `temporal_position` advances only during `Predict`; `prediction_age` resets on
    ///   `Observe` and advances during `Predict`.
    pub fn pyramid_state_from_patch_tokens(
        &self,
        patch_tokens: Tensor<B, 3>,
    ) -> StructuredTopologyState<B> {
        assert!(
            self.pyramid_backbone_enabled(),
            "structured pyramid state requires vision.backbone = \"pyramid\""
        );
        let [batch, patch_count, _dim] = patch_tokens.shape().dims::<3>();
        let grid_height = self.grid_height.max(1);
        let grid_width = self.grid_width.max(1);
        assert_eq!(
            patch_count,
            grid_height * grid_width,
            "structured pyramid state requires patch count {} to match grid {}x{}",
            patch_count,
            grid_height,
            grid_width
        );
        let h8 = self.pyramid_patch_tokens_to_spatial(patch_tokens);
        let coarse_state = self.pyramid_pool_patch_state(h8.clone());
        let patch_rank = self.trm_graph.patch_rank_resolved();
        let coarse_rank = self.trm_graph.coarse_rank_resolved();
        let global_rank = self.trm_graph.global_rank_resolved();
        let value_dim = self.trm_graph.value_dim.max(1);
        let [_, _, h32_height, h32_width] = coarse_state.shape().dims::<4>();
        let device = h8.device();
        StructuredTopologyState {
            primary_state: h8,
            context_state: coarse_state,
            rho: BankedRhoState {
                primary_rho: Tensor::<B, 5>::zeros(
                    [batch, patch_rank, value_dim, grid_height, grid_width],
                    &device,
                ),
                context_rho: Tensor::<B, 5>::zeros(
                    [
                        batch,
                        coarse_rank,
                        value_dim,
                        h32_height.max(1),
                        h32_width.max(1),
                    ],
                    &device,
                ),
                global_rho: Tensor::<B, 4>::zeros(
                    [
                        batch,
                        self.trm_graph.hub_count.max(1),
                        global_rank,
                        value_dim,
                    ],
                    &device,
                ),
            },
            temporal_position: 0,
            prediction_age: 0,
        }
    }

    /// Replaces the dense patch/coarse observation while preserving local/coarse/global `rho`.
    pub fn pyramid_state_with_patch_tokens(
        &self,
        mut state: StructuredTopologyState<B>,
        patch_tokens: Tensor<B, 3>,
    ) -> StructuredTopologyState<B> {
        let next_patch = self.pyramid_patch_tokens_to_spatial(patch_tokens);
        let next_coarse = self.pyramid_pool_patch_state(next_patch.clone());
        *state.primary_state_mut() = next_patch;
        *state.context_state_mut() = next_coarse;
        state
    }

    /// Replaces the dense patch observation in spatial form while preserving local/coarse/global
    /// `rho`.
    pub fn pyramid_state_with_patch_state(
        &self,
        mut state: StructuredTopologyState<B>,
        patch_state: Tensor<B, 4>,
    ) -> StructuredTopologyState<B> {
        let next_patch = self.apply_embed_norm_spatial(patch_state);
        let next_coarse = self.pyramid_pool_patch_state(next_patch.clone());
        *state.primary_state_mut() = next_patch;
        *state.context_state_mut() = next_coarse;
        state
    }

    /// Observation step for the pyramid backbone.
    ///
    /// This preserves local/coarse/global recurrent banks, replaces the current observed dense
    /// patch state, and runs the recurrent block in `Observe` mode without advancing time.
    pub fn observe_pyramid_state(
        &self,
        state: StructuredTopologyState<B>,
        patch_tokens: Tensor<B, 3>,
        steps: usize,
        backprop_steps: usize,
    ) -> StructuredTopologyState<B> {
        let state = self.pyramid_state_with_patch_tokens(state, patch_tokens);
        self.forward_pyramid_state_rollout_mode_unbounded(
            state,
            steps,
            backprop_steps,
            StructuredStepMode::Observe,
        )
    }

    /// Refinement step for the pyramid backbone.
    ///
    /// This reuses the current patch/coarse states and local/coarse/global `rho` without
    /// advancing temporal time.
    pub fn refine_pyramid_state(
        &self,
        state: StructuredTopologyState<B>,
        steps: usize,
        backprop_steps: usize,
    ) -> StructuredTopologyState<B> {
        self.forward_pyramid_state_rollout_mode_unbounded(
            state,
            steps,
            backprop_steps,
            StructuredStepMode::Refine,
        )
    }

    /// Predictive temporal rollout for the pyramid backbone.
    ///
    /// This reuses the current patch/coarse states and local/coarse/global banks, advances
    /// temporal counters, and applies predictive recurrent decay.
    pub fn predict_pyramid_state(
        &self,
        state: StructuredTopologyState<B>,
        steps: usize,
        backprop_steps: usize,
    ) -> StructuredTopologyState<B> {
        self.forward_pyramid_state_rollout_mode_unbounded(
            state,
            steps,
            backprop_steps,
            StructuredStepMode::Predict,
        )
    }

    pub fn pyramid_patch_tokens(&self, state: &StructuredTopologyState<B>) -> Tensor<B, 3> {
        self.pyramid_spatial_to_patch_tokens(state.primary_state().clone())
    }

    /// Returns the configured pyramid CLS readout summary.
    ///
    /// Depending on `vision.trm_graph.cls_readout`, this may be pooled from the patch bank or
    /// produced by a learned readout over `hub_rho` and coarse state.
    pub fn pyramid_summary(&self, state: &StructuredTopologyState<B>) -> Tensor<B, 2> {
        match self.trm_graph.cls_readout {
            VisionTrmClsReadoutKind::PatchMean => self.pyramid_patch_summary(state),
            VisionTrmClsReadoutKind::Hub => self.pyramid_hub_summary(state),
            VisionTrmClsReadoutKind::HubAndCoarse => self.pyramid_hub_and_coarse_summary(state),
        }
    }

    fn pyramid_patch_summary(&self, state: &StructuredTopologyState<B>) -> Tensor<B, 2> {
        let [batch, dim, _height, _width] = state.primary_state().shape().dims::<4>();
        state
            .primary_state()
            .clone()
            .mean_dim(2)
            .mean_dim(3)
            .reshape([batch, dim])
    }

    fn pyramid_coarse_summary(&self, state: &StructuredTopologyState<B>) -> Tensor<B, 2> {
        let [batch, dim, _height, _width] = state.context_state().shape().dims::<4>();
        state
            .context_state()
            .clone()
            .mean_dim(2)
            .mean_dim(3)
            .reshape([batch, dim])
    }

    fn pyramid_hub_summary(&self, state: &StructuredTopologyState<B>) -> Tensor<B, 2> {
        let [batch, hub_count, global_rank, value_dim] = state.hub_rho().shape().dims::<4>();
        let hub_flat = state
            .hub_rho()
            .clone()
            .reshape([batch, hub_count, global_rank * value_dim]);
        let projected = self
            .pyramid_hub_cls_proj
            .as_ref()
            .expect("pyramid hub cls projection should exist for pyramid backbone")
            .forward(hub_flat);
        projected.mean_dim(1).reshape([batch, self.embed_dim])
    }

    fn pyramid_hub_and_coarse_summary(&self, state: &StructuredTopologyState<B>) -> Tensor<B, 2> {
        let coarse = self.pyramid_coarse_summary(state);
        let hub = self.pyramid_hub_summary(state);
        let fused = Tensor::cat(vec![coarse, hub], 1);
        self.pyramid_cls_fuse_proj
            .as_ref()
            .expect("pyramid cls fuse projection should exist for pyramid backbone")
            .forward(fused)
    }

    pub(super) fn pyramid_readout_tokens(
        &self,
        state: &StructuredTopologyState<B>,
    ) -> Tensor<B, 3> {
        let patch_tokens = self.pyramid_patch_tokens(state);
        if self.use_cls_token {
            let cls = self.pyramid_summary(state);
            let [batch, dim] = cls.shape().dims::<2>();
            Tensor::cat(vec![cls.reshape([batch, 1, dim]), patch_tokens], 1)
        } else {
            patch_tokens
        }
    }

    fn pyramid_decay_by_rank(
        &self,
        rank: usize,
        temporal_dt: usize,
        device: &B::Device,
        decay_scale: f32,
    ) -> Tensor<B, 1> {
        if temporal_dt == 0 {
            return Tensor::<B, 1>::ones([rank.max(1)], device);
        }

        let base_decay = self.trm_graph.decay.clamp(0.0, 1.0);
        if base_decay <= 0.0 {
            return Tensor::<B, 1>::zeros([rank.max(1)], device);
        }
        if base_decay >= 1.0 {
            return Tensor::<B, 1>::ones([rank.max(1)], device);
        }

        let slopes = if self.use_alibi {
            burn_dragon_core::kernel::linear_attention::default_alibi_slopes(rank.max(1))
        } else {
            vec![1.0; rank.max(1)]
        };
        let dt = (temporal_dt as f32) * decay_scale.max(0.0);
        let values = slopes
            .into_iter()
            .map(|slope| base_decay.powf(slope * dt))
            .collect::<Vec<_>>();
        Tensor::<B, 1>::from_data(TensorData::new(values, [rank.max(1)]), device)
    }

    fn pyramid_rollout_executor_mode(&self) -> PyramidRolloutExecutorMode {
        if self.kernel.enabled && supports_structured_pyramid_rho_backend::<B>() {
            return PyramidRolloutExecutorMode::WgpuFused;
        }
        if self.kernel.enabled && supports_local_grid_rho_backend::<B>() {
            return PyramidRolloutExecutorMode::WgpuLocalPlans;
        }
        PyramidRolloutExecutorMode::HostLoop
    }

    fn pyramid_bank_mode_for_kernel(
        bank_mode: &VisionTrmGraphBankModeConfig,
    ) -> StructuredPyramidBankMode {
        StructuredPyramidBankMode {
            patch_local_read: bank_mode.patch_local_read,
            patch_local_write: bank_mode.patch_local_write,
            coarse_local_read: bank_mode.coarse_local_read,
            coarse_local_write: bank_mode.coarse_local_write,
            patch_from_coarse_read: bank_mode.patch_from_coarse_read,
            patch_from_hub_read: bank_mode.patch_from_hub_read,
            coarse_from_hub_read: bank_mode.coarse_from_hub_read,
            patch_to_coarse_write: bank_mode.patch_to_coarse_write,
            patch_to_global_write: bank_mode.patch_to_global_write,
            coarse_to_global_write: bank_mode.coarse_to_global_write,
        }
    }

    fn pyramid_shape(&self) -> StructuredPyramidShape {
        StructuredPyramidShape {
            patch: LocalGridShape2d::new(self.grid_height.max(1), self.grid_width.max(1)),
            coarse: LocalGridShape2d::new(
                (self.grid_height.max(1) / self.trm_graph.coarse_stride.max(1)).max(1),
                (self.grid_width.max(1) / self.trm_graph.coarse_stride.max(1)).max(1),
            ),
            coarse_stride: self.trm_graph.coarse_stride.max(1),
            hub_count: self.trm_graph.hub_count.max(1),
        }
    }

    fn pyramid_rho_step_with_plan(
        &self,
        shape: StructuredPyramidShape,
        input: StructuredPyramidRhoStepInput<B>,
        fused_plan: Option<&CompiledStructuredPyramidRhoPlan<B>>,
    ) -> StructuredPyramidRhoStepOutput<B> {
        if let Some(plan) = fused_plan
            && let Some(fused) =
                try_fused_structured_pyramid_rho_step_wgpu_with_plan(shape, input.clone(), plan)
        {
            return fused;
        }
        reference_structured_pyramid_rho_step(shape, input)
    }

    #[allow(clippy::too_many_arguments)]
    fn pyramid_reference_step_split(
        &self,
        patch_query: Tensor<B, 4>,
        patch_query_for_coarse: Tensor<B, 4>,
        patch_query_for_global: Tensor<B, 4>,
        patch_value: Tensor<B, 4>,
        coarse_query: Tensor<B, 4>,
        coarse_query_for_global: Tensor<B, 4>,
        coarse_value: Tensor<B, 4>,
        patch_rho: Tensor<B, 5>,
        coarse_rho: Tensor<B, 5>,
        global_rho: Tensor<B, 4>,
        patch_hub_weights: Option<Tensor<B, 4>>,
        coarse_hub_weights: Option<Tensor<B, 4>>,
        patch_decay: Tensor<B, 1>,
        coarse_decay: Tensor<B, 1>,
        global_decay: Tensor<B, 1>,
        bank_mode: &VisionTrmGraphBankModeConfig,
    ) -> StructuredPyramidRhoStepOutput<B> {
        let shape = self.pyramid_shape();
        let [patch_batch, patch_value_dim, patch_height, patch_width] =
            patch_value.shape().dims::<4>();
        let [coarse_batch, coarse_value_dim, coarse_height, coarse_width] =
            coarse_value.shape().dims::<4>();
        let patch_zero = || {
            Tensor::<B, 4>::zeros(
                [patch_batch, patch_value_dim, patch_height, patch_width],
                &patch_value.device(),
            )
        };
        let coarse_zero = || {
            Tensor::<B, 4>::zeros(
                [coarse_batch, coarse_value_dim, coarse_height, coarse_width],
                &coarse_value.device(),
            )
        };

        let patch_local_context = if bank_mode.patch_local_read {
            self.pyramid_local_read(patch_rho.clone(), patch_query.clone(), false)
        } else {
            patch_zero()
        };
        let coarse_local_context = if bank_mode.coarse_local_read {
            self.pyramid_local_read(coarse_rho.clone(), coarse_query.clone(), true)
        } else {
            coarse_zero()
        };
        let patch_from_coarse_context = if bank_mode.patch_from_coarse_read {
            self.pyramid_cross_scale_read(
                coarse_rho.clone(),
                patch_query_for_coarse.clone(),
                shape.coarse_stride.max(1),
            )
        } else {
            patch_zero()
        };
        let patch_from_hub_context = if bank_mode.patch_from_hub_read {
            self.pyramid_hub_read(
                global_rho.clone(),
                patch_query_for_global.clone(),
                patch_hub_weights.clone(),
            )
        } else {
            patch_zero()
        };
        let coarse_from_hub_context = if bank_mode.coarse_from_hub_read {
            self.pyramid_hub_read(
                global_rho.clone(),
                coarse_query_for_global.clone(),
                coarse_hub_weights.clone(),
            )
        } else {
            coarse_zero()
        };

        let patch_update = if bank_mode.patch_local_write {
            self.pyramid_outer_product(patch_query.clone(), patch_value.clone())
        } else {
            Tensor::<B, 5>::zeros(patch_rho.shape().dims::<5>(), &patch_rho.device())
        };
        let patch_to_coarse_update = bank_mode.patch_to_coarse_write.then(|| {
            self.pyramid_pool_outer(
                self.pyramid_outer_product(patch_query_for_coarse, patch_value.clone()),
                shape.coarse_stride.max(1),
            )
        });
        let patch_to_global_update = bank_mode
            .patch_to_global_write
            .then(|| self.pyramid_outer_product(patch_query_for_global, patch_value));
        let coarse_update = if bank_mode.coarse_local_write {
            self.pyramid_outer_product(coarse_query.clone(), coarse_value.clone())
        } else {
            Tensor::<B, 5>::zeros(coarse_rho.shape().dims::<5>(), &coarse_rho.device())
        };
        let coarse_to_global_update = bank_mode
            .coarse_to_global_write
            .then(|| self.pyramid_outer_product(coarse_query_for_global, coarse_value));

        let next_patch_rho = target_major_decay_add(
            Self::pyramid_rho_to_target_major(patch_rho),
            Self::pyramid_rho_to_target_major(patch_update),
            patch_decay,
        );
        let next_patch_rho = Self::pyramid_rho_from_target_major(
            next_patch_rho,
            shape.patch.height,
            shape.patch.width,
        );

        let coarse_rho_shape = coarse_rho.shape().dims::<5>();
        let coarse_rank = coarse_rho_shape[1];
        let coarse_rho_device = coarse_rho.device();
        let next_coarse_rho = target_major_decay_add(
            Self::pyramid_rho_to_target_major(coarse_rho),
            Self::pyramid_rho_to_target_major(coarse_update.clone()).add(
                patch_to_coarse_update
                    .map(Self::pyramid_rho_to_target_major)
                    .unwrap_or_else(|| {
                        Tensor::<B, 4>::zeros(
                            [
                                coarse_batch,
                                coarse_height * coarse_width,
                                coarse_rank,
                                coarse_value_dim,
                            ],
                            &coarse_rho_device,
                        )
                    }),
            ),
            coarse_decay,
        );
        let next_coarse_rho = Self::pyramid_rho_from_target_major(
            next_coarse_rho,
            shape.coarse.height,
            shape.coarse.width,
        );

        let next_hub_rho = self.pyramid_update_hub(
            global_rho,
            patch_to_global_update,
            coarse_to_global_update,
            patch_hub_weights,
            coarse_hub_weights,
            shape.hub_count.max(1),
            global_decay,
        );

        StructuredPyramidRhoStepOutput {
            patch_local_context,
            coarse_local_context,
            patch_from_coarse_context,
            patch_from_hub_context,
            coarse_from_hub_context,
            next_patch_rho,
            next_coarse_rho,
            next_hub_rho,
        }
    }

    pub fn forward_pyramid_state_rollout_unbounded(
        &self,
        state: StructuredTopologyState<B>,
        steps: usize,
        backprop_steps: usize,
    ) -> StructuredTopologyState<B> {
        self.forward_pyramid_state_rollout_mode_unbounded(
            state,
            steps,
            backprop_steps,
            StructuredStepMode::Predict,
        )
    }

    pub fn forward_pyramid_state_rollout_mode_unbounded(
        &self,
        state: StructuredTopologyState<B>,
        steps: usize,
        backprop_steps: usize,
        mode: StructuredStepMode,
    ) -> StructuredTopologyState<B> {
        self.forward_pyramid_state_rollout_mode_with_checkpoints_unbounded(
            state,
            steps,
            backprop_steps,
            &[],
            mode,
        )
        .0
    }

    fn forward_pyramid_state_rollout_mode_with_checkpoints_unbounded(
        &self,
        mut state: StructuredTopologyState<B>,
        steps: usize,
        backprop_steps: usize,
        checkpoint_steps: &[usize],
        mode: StructuredStepMode,
    ) -> (
        StructuredTopologyState<B>,
        Vec<(usize, StructuredTopologyState<B>)>,
    ) {
        assert!(
            self.pyramid_backbone_enabled(),
            "structured pyramid rollout requires vision.backbone = \"pyramid\""
        );
        let pyramid_patch_x_neuron_proj = self
            .pyramid_patch_x_neuron_proj
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_patch_x_neuron_proj");
        let pyramid_patch_to_coarse_query_proj = self
            .pyramid_patch_to_coarse_query_proj
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_patch_to_coarse_query_proj");
        let pyramid_patch_to_global_query_proj = self
            .pyramid_patch_to_global_query_proj
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_patch_to_global_query_proj");
        let pyramid_coarse_x_neuron_proj = self
            .pyramid_coarse_x_neuron_proj
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_coarse_x_neuron_proj");
        let pyramid_coarse_to_global_query_proj = self
            .pyramid_coarse_to_global_query_proj
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_coarse_to_global_query_proj");
        let pyramid_write_value_proj = self
            .pyramid_write_value_proj
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_write_value_proj");
        let pyramid_patch_y_gate_proj = self
            .pyramid_patch_y_gate_proj
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_patch_y_gate_proj");
        let pyramid_patch_delta_proj = self
            .pyramid_patch_delta_proj
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_patch_delta_proj");
        let pyramid_coarse_y_gate_proj = self
            .pyramid_coarse_y_gate_proj
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_coarse_y_gate_proj");
        let pyramid_coarse_delta_proj = self
            .pyramid_coarse_delta_proj
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_coarse_delta_proj");
        let pyramid_value_norm = self
            .pyramid_value_norm
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_value_norm");
        let paired_update_plan = pyramid_ops::CompiledStructuredDenseUpdatePairPlan::new(
            pyramid_patch_y_gate_proj,
            pyramid_patch_delta_proj,
            pyramid_coarse_y_gate_proj,
            pyramid_coarse_delta_proj,
        );
        let steps = steps.max(1);
        let backprop_steps = backprop_steps.max(1).min(steps);
        let detach_until = steps.saturating_sub(backprop_steps);
        let mut checkpoint_steps = checkpoint_steps
            .iter()
            .copied()
            .filter(|checkpoint| *checkpoint > 0 && *checkpoint <= steps)
            .collect::<Vec<_>>();
        checkpoint_steps.sort_unstable();
        checkpoint_steps.dedup();
        let hub_count = self.trm_graph.hub_count.max(1);
        let patch_rank = self.trm_graph.patch_rank_resolved();
        let coarse_rank = self.trm_graph.coarse_rank_resolved();
        let global_rank = self.trm_graph.global_rank_resolved();
        let temporal_dt = mode.temporal_dt();
        let final_checkpoint_metadata = Self::advance_structured_temporal_metadata(
            state.temporal_position,
            state.prediction_age,
            mode,
        );
        let mut captured_checkpoints = Vec::with_capacity(checkpoint_steps.len());
        let mut next_checkpoint_index = 0usize;
        let bank_mode = self.trm_graph.bank_mode(mode).clone();
        let predict_coarse_substeps = if matches!(mode, StructuredStepMode::Predict) {
            self.trm_graph.predict_coarse_substeps.max(1)
        } else {
            1
        };
        let predict_substep_kind = self.trm_graph.predict_substep_kind;
        let patch_decay = self.pyramid_decay_by_rank(
            patch_rank,
            temporal_dt,
            &state.primary_state().device(),
            bank_mode.patch_decay_scale,
        );
        let coarse_decay = self.pyramid_decay_by_rank(
            coarse_rank,
            temporal_dt,
            &state.primary_state().device(),
            bank_mode.coarse_decay_scale,
        );
        let global_decay = self.pyramid_decay_by_rank(
            global_rank,
            temporal_dt,
            &state.primary_state().device(),
            bank_mode.global_decay_scale,
        );
        let pyramid_shape = self.pyramid_shape();
        let executor_mode = self.pyramid_rollout_executor_mode();
        let uniform_fused_eligible = self.trm_graph.ranks_uniform()
            && self.trm_graph.uses_uniform_local_topology()
            && self.trm_graph.uses_default_bank_schedule()
            && predict_coarse_substeps == 1;
        let fused_plan = match executor_mode {
            PyramidRolloutExecutorMode::WgpuFused if uniform_fused_eligible => {
                Some(CompiledStructuredPyramidRhoPlan::new(
                    state.primary_state().shape().dims::<4>()[0],
                    patch_rank,
                    self.trm_graph.value_dim.max(1),
                    pyramid_shape,
                    self.resolve_rho_stream_neighborhood(),
                    &state.primary_state().device(),
                ))
            }
            _ => None,
        };
        let split_fused_plan = match executor_mode {
            // For custom schedules / heterogeneous ranks, the lighter local-plan executor still
            // benchmarks better overall than the current split-fused path.
            PyramidRolloutExecutorMode::WgpuFused if !uniform_fused_eligible => None,
            _ => None,
        };
        let patch_neighborhood = self.pyramid_patch_neighborhood();
        let coarse_neighborhood = self.pyramid_coarse_neighborhood();
        let stage_aware_plan = match executor_mode {
            PyramidRolloutExecutorMode::HostLoop => None,
            PyramidRolloutExecutorMode::WgpuLocalPlans | PyramidRolloutExecutorMode::WgpuFused => {
                Some(pyramid_ops::CompiledStageAwarePyramidLocalPlan::new(
                    pyramid_ops::CompiledStageAwarePyramidLocalPlanSpec {
                        batch: state.primary_state().shape().dims::<4>()[0],
                        patch_rank,
                        coarse_rank,
                        value_dim: self.trm_graph.value_dim.max(1),
                        patch_shape: pyramid_shape.patch,
                        coarse_shape: pyramid_shape.coarse,
                        patch_neighborhood,
                        coarse_neighborhood,
                        device: &state.primary_state().device(),
                    },
                ))
            }
        };
        let local_bridge_projection_pair_plan = match executor_mode {
            PyramidRolloutExecutorMode::HostLoop => None,
            PyramidRolloutExecutorMode::WgpuLocalPlans | PyramidRolloutExecutorMode::WgpuFused => {
                pyramid_ops::CompiledLocalBridgeProjectionPairPlan::new(
                    pyramid_patch_x_neuron_proj,
                    pyramid_coarse_x_neuron_proj,
                    pyramid_write_value_proj,
                )
            }
        };
        let patch_local_bridge_plan = match executor_mode {
            PyramidRolloutExecutorMode::HostLoop => None,
            PyramidRolloutExecutorMode::WgpuLocalPlans | PyramidRolloutExecutorMode::WgpuFused => {
                Some(CompiledLocalGridRhoPlan::new(
                    LocalGridRhoPlanSpec {
                        batch: state.primary_state().shape().dims::<4>()[0],
                        heads: patch_rank.max(1),
                        value_heads: 1,
                        patch_tokens: pyramid_shape.patch.token_count().max(1),
                        latent: 1,
                        embd: self.trm_graph.value_dim.max(1),
                        grid: pyramid_shape.patch,
                        neighborhood: patch_neighborhood,
                    },
                    &state.primary_state().device(),
                ))
            }
        };
        let coarse_local_bridge_plan = match executor_mode {
            PyramidRolloutExecutorMode::HostLoop => None,
            PyramidRolloutExecutorMode::WgpuLocalPlans | PyramidRolloutExecutorMode::WgpuFused => {
                Some(CompiledLocalGridRhoPlan::new(
                    LocalGridRhoPlanSpec {
                        batch: state.primary_state().shape().dims::<4>()[0],
                        heads: coarse_rank.max(1),
                        value_heads: 1,
                        patch_tokens: pyramid_shape.coarse.token_count().max(1),
                        latent: 1,
                        embd: self.trm_graph.value_dim.max(1),
                        grid: pyramid_shape.coarse,
                        neighborhood: coarse_neighborhood,
                    },
                    &state.primary_state().device(),
                ))
            }
        };
        let bridge_bank_mode = match predict_substep_kind {
            VisionTrmPredictSubstepKind::CoarseOnly => bank_mode.bridge_predict_substep(),
            VisionTrmPredictSubstepKind::LocalBridge => bank_mode.local_bridge_predict_substep(),
        };
        let build_patch_projection_spec = |active_bank_mode: &VisionTrmGraphBankModeConfig| {
            let patch_coarse_query_enabled =
                active_bank_mode.patch_from_coarse_read || active_bank_mode.patch_to_coarse_write;
            let patch_global_query_enabled =
                active_bank_mode.patch_from_hub_read || active_bank_mode.patch_to_global_write;
            let patch_value_enabled = active_bank_mode.patch_local_write
                || active_bank_mode.patch_to_coarse_write
                || active_bank_mode.patch_to_global_write;
            let patch_hub_weights_enabled = hub_count > 1
                && (active_bank_mode.patch_from_hub_read || active_bank_mode.patch_to_global_write);
            let patch_hub_gate_enabled =
                patch_hub_weights_enabled && self.pyramid_hub_gate.is_some();
            let mut patch_layers = vec![pyramid_patch_x_neuron_proj];
            if patch_coarse_query_enabled {
                patch_layers.push(pyramid_patch_to_coarse_query_proj);
            }
            if patch_global_query_enabled {
                patch_layers.push(pyramid_patch_to_global_query_proj);
            }
            if patch_value_enabled {
                patch_layers.push(pyramid_write_value_proj);
            }
            if let Some(hub_gate) = self.pyramid_hub_gate.as_ref()
                && patch_hub_gate_enabled
            {
                patch_layers.push(hub_gate);
            }
            (
                patch_coarse_query_enabled,
                patch_global_query_enabled,
                patch_value_enabled,
                patch_hub_weights_enabled,
                patch_hub_gate_enabled,
                pyramid_ops::CompiledSpatialProjectionPlan::new(&patch_layers),
            )
        };
        let build_coarse_projection_spec = |active_bank_mode: &VisionTrmGraphBankModeConfig| {
            let coarse_global_query_enabled =
                active_bank_mode.coarse_from_hub_read || active_bank_mode.coarse_to_global_write;
            let coarse_value_enabled =
                active_bank_mode.coarse_local_write || active_bank_mode.coarse_to_global_write;
            let coarse_hub_weights_enabled = hub_count > 1
                && (active_bank_mode.coarse_from_hub_read
                    || active_bank_mode.coarse_to_global_write);
            let coarse_hub_gate_enabled =
                coarse_hub_weights_enabled && self.pyramid_hub_gate.is_some();
            let mut coarse_layers = vec![pyramid_coarse_x_neuron_proj];
            if coarse_global_query_enabled {
                coarse_layers.push(pyramid_coarse_to_global_query_proj);
            }
            if coarse_value_enabled {
                coarse_layers.push(pyramid_write_value_proj);
            }
            if let Some(hub_gate) = self.pyramid_hub_gate.as_ref()
                && coarse_hub_gate_enabled
            {
                coarse_layers.push(hub_gate);
            }
            (
                coarse_global_query_enabled,
                coarse_value_enabled,
                coarse_hub_weights_enabled,
                coarse_hub_gate_enabled,
                pyramid_ops::CompiledSpatialProjectionPlan::new(&coarse_layers),
            )
        };
        let patch_predict_projection = build_patch_projection_spec(&bank_mode);
        let patch_bridge_projection = build_patch_projection_spec(&bridge_bank_mode);
        let coarse_predict_projection = build_coarse_projection_spec(&bank_mode);
        let coarse_bridge_projection = build_coarse_projection_spec(&bridge_bank_mode);

        for step_idx in 0..steps {
            let h8 = state.primary_state().clone();
            let h32 = state.context_state().clone();
            let mut current_patch_state = h8;
            let mut current_coarse_state = h32;
            let mut current_patch_rho = state.patch_rho().clone();
            let mut current_coarse_rho = state.coarse_rho().clone();
            let mut current_hub_rho = state.hub_rho().clone();
            let ones_patch_decay =
                Tensor::<B, 1>::ones([patch_rank.max(1)], &state.primary_state().device());
            let ones_coarse_decay =
                Tensor::<B, 1>::ones([coarse_rank.max(1)], &state.primary_state().device());
            let ones_global_decay =
                Tensor::<B, 1>::ones([global_rank.max(1)], &state.primary_state().device());
            let project_patch = |patch_state: Tensor<B, 4>,
                                 patch_coarse_query_enabled: bool,
                                 patch_global_query_enabled: bool,
                                 patch_value_enabled: bool,
                                 patch_hub_weights_enabled: bool,
                                 patch_hub_gate_enabled: bool,
                                 patch_plan: Option<
                &pyramid_ops::CompiledSpatialProjectionPlan<B>,
            >| {
                let [patch_batch, _, patch_height, patch_width] = patch_state.shape().dims::<4>();
                let mut patch_layers = vec![pyramid_patch_x_neuron_proj];
                if patch_coarse_query_enabled {
                    patch_layers.push(pyramid_patch_to_coarse_query_proj);
                }
                if patch_global_query_enabled {
                    patch_layers.push(pyramid_patch_to_global_query_proj);
                }
                if patch_value_enabled {
                    patch_layers.push(pyramid_write_value_proj);
                }
                if let Some(hub_gate) = self.pyramid_hub_gate.as_ref()
                    && patch_hub_gate_enabled
                {
                    patch_layers.push(hub_gate);
                }
                let mut patch_proj = if let Some(plan) = patch_plan {
                    self.project_spatial_many_with_plan(patch_state.clone(), plan)
                } else {
                    self.project_spatial_many(patch_state.clone(), &patch_layers)
                }
                .into_iter();
                let patch_x = activation::relu(
                    patch_proj
                        .next()
                        .expect("patch multi-projection should include x_neuron output"),
                );
                let patch_coarse_query = if patch_coarse_query_enabled {
                    activation::relu(
                        patch_proj
                            .next()
                            .expect("patch multi-projection should include coarse-query output"),
                    )
                } else {
                    patch_x.clone()
                };
                let patch_global_query = if patch_global_query_enabled {
                    activation::relu(
                        patch_proj
                            .next()
                            .expect("patch multi-projection should include global-query output"),
                    )
                } else {
                    patch_x.clone()
                };
                let patch_value = if patch_value_enabled {
                    patch_proj
                        .next()
                        .expect("patch multi-projection should include write-value output")
                } else {
                    Tensor::<B, 4>::zeros(
                        [
                            patch_batch,
                            self.trm_graph.value_dim.max(1),
                            patch_height,
                            patch_width,
                        ],
                        &patch_state.device(),
                    )
                };
                let patch_hub_weights = if patch_hub_gate_enabled {
                    Some(
                        self.normalize_hub_weights(
                            patch_proj
                                .next()
                                .expect("patch multi-projection should include hub-gate output"),
                        ),
                    )
                } else if patch_hub_weights_enabled {
                    Some(self.pyramid_hub_weights_single(patch_state, hub_count, None))
                } else {
                    None
                };
                (
                    patch_x,
                    patch_coarse_query,
                    patch_global_query,
                    patch_value,
                    patch_hub_weights,
                )
            };
            let project_coarse = |coarse_state: Tensor<B, 4>,
                                  coarse_global_query_enabled: bool,
                                  coarse_value_enabled: bool,
                                  coarse_hub_weights_enabled: bool,
                                  coarse_hub_gate_enabled: bool,
                                  coarse_plan: Option<
                &pyramid_ops::CompiledSpatialProjectionPlan<B>,
            >| {
                let [coarse_batch, _, coarse_height, coarse_width] =
                    coarse_state.shape().dims::<4>();
                let mut coarse_layers = vec![pyramid_coarse_x_neuron_proj];
                if coarse_global_query_enabled {
                    coarse_layers.push(pyramid_coarse_to_global_query_proj);
                }
                if coarse_value_enabled {
                    coarse_layers.push(pyramid_write_value_proj);
                }
                if let Some(hub_gate) = self.pyramid_hub_gate.as_ref()
                    && coarse_hub_gate_enabled
                {
                    coarse_layers.push(hub_gate);
                }
                let mut coarse_proj = if let Some(plan) = coarse_plan {
                    self.project_spatial_many_with_plan(coarse_state.clone(), plan)
                } else {
                    self.project_spatial_many(coarse_state.clone(), &coarse_layers)
                }
                .into_iter();
                let coarse_x = activation::relu(
                    coarse_proj
                        .next()
                        .expect("coarse multi-projection should include x_neuron output"),
                );
                let coarse_global_query = if coarse_global_query_enabled {
                    activation::relu(
                        coarse_proj
                            .next()
                            .expect("coarse multi-projection should include global-query output"),
                    )
                } else {
                    coarse_x.clone()
                };
                let coarse_value = if coarse_value_enabled {
                    coarse_proj
                        .next()
                        .expect("coarse multi-projection should include write-value output")
                } else {
                    Tensor::<B, 4>::zeros(
                        [
                            coarse_batch,
                            self.trm_graph.value_dim.max(1),
                            coarse_height,
                            coarse_width,
                        ],
                        &coarse_state.device(),
                    )
                };
                let coarse_hub_weights = if coarse_hub_gate_enabled {
                    Some(
                        self.normalize_hub_weights(
                            coarse_proj
                                .next()
                                .expect("coarse multi-projection should include hub-gate output"),
                        ),
                    )
                } else if coarse_hub_weights_enabled {
                    Some(self.pyramid_hub_weights_single(coarse_state, hub_count, None))
                } else {
                    None
                };
                (
                    coarse_x,
                    coarse_global_query,
                    coarse_value,
                    coarse_hub_weights,
                )
            };
            for _ in 1..predict_coarse_substeps {
                match predict_substep_kind {
                    VisionTrmPredictSubstepKind::CoarseOnly => {
                        let (patch_x, patch_coarse_query, patch_global_query, v8, hub_w8) =
                            project_patch(
                                current_patch_state.clone(),
                                patch_bridge_projection.0,
                                patch_bridge_projection.1,
                                patch_bridge_projection.2,
                                patch_bridge_projection.3,
                                patch_bridge_projection.4,
                                patch_bridge_projection.5.as_ref(),
                            );
                        let (coarse_x, coarse_global_query, v32, hub_w32) = project_coarse(
                            current_coarse_state.clone(),
                            coarse_bridge_projection.0,
                            coarse_bridge_projection.1,
                            coarse_bridge_projection.2,
                            coarse_bridge_projection.3,
                            coarse_bridge_projection.4.as_ref(),
                        );
                        if let Some(plan) = split_fused_plan.as_ref() {
                            let rho_step =
                                try_fused_structured_pyramid_coarse_only_no_patch_step_wgpu_with_plan(
                                    pyramid_shape,
                                    StructuredPyramidCoarseOnlyNoPatchStepInput {
                                        coarse_local_query: coarse_x.clone(),
                                        coarse_query_for_global: coarse_global_query.clone(),
                                        coarse_value: v32.clone(),
                                        coarse_rho: current_coarse_rho.clone(),
                                        hub_rho: current_hub_rho.clone(),
                                        coarse_hub_weights: hub_w32.clone(),
                                        coarse_decay: ones_coarse_decay.clone(),
                                        global_decay: ones_global_decay.clone(),
                                        bank_mode: Self::pyramid_bank_mode_for_kernel(
                                            &bridge_bank_mode,
                                        ),
                                    },
                                    plan,
                                )
                                .unwrap_or_else(|| {
                                    let rho_step = self.pyramid_reference_step_split(
                                        patch_x.clone(),
                                        patch_coarse_query.clone(),
                                        patch_global_query.clone(),
                                        v8.clone(),
                                        coarse_x.clone(),
                                        coarse_global_query,
                                        v32,
                                        current_patch_rho.clone(),
                                        current_coarse_rho,
                                        current_hub_rho,
                                        hub_w8.clone(),
                                        hub_w32.clone(),
                                        ones_patch_decay.clone(),
                                        ones_coarse_decay.clone(),
                                        ones_global_decay.clone(),
                                        &bridge_bank_mode,
                                    );
                                    StructuredPyramidCoarseOnlyStepOutput {
                                        coarse_local_context: rho_step.coarse_local_context,
                                        coarse_from_hub_context: rho_step.coarse_from_hub_context,
                                        next_coarse_rho: rho_step.next_coarse_rho,
                                        next_hub_rho: rho_step.next_hub_rho,
                                    }
                                });
                            current_coarse_state = self.pyramid_update_state(
                                current_coarse_state,
                                coarse_x.clone(),
                                rho_step.coarse_local_context + rho_step.coarse_from_hub_context,
                                pyramid_coarse_y_gate_proj,
                                pyramid_coarse_delta_proj,
                                pyramid_value_norm,
                            );
                            current_coarse_rho = rho_step.next_coarse_rho;
                            current_hub_rho = rho_step.next_hub_rho;
                        } else if let Some(plan) = stage_aware_plan.as_ref() {
                            let rho_step = self.pyramid_stage_aware_coarse_only_step_with_plan(
                                coarse_x.clone(),
                                coarse_global_query,
                                v32,
                                current_coarse_rho,
                                current_hub_rho,
                                hub_w32.clone(),
                                ones_coarse_decay.clone(),
                                ones_global_decay.clone(),
                                &bridge_bank_mode,
                                plan,
                            );
                            current_coarse_state = self.pyramid_update_state(
                                current_coarse_state,
                                coarse_x.clone(),
                                rho_step.coarse_local_context + rho_step.coarse_from_hub_context,
                                pyramid_coarse_y_gate_proj,
                                pyramid_coarse_delta_proj,
                                pyramid_value_norm,
                            );
                            current_coarse_rho = rho_step.next_coarse_rho;
                            current_hub_rho = rho_step.next_hub_rho;
                        } else {
                            let rho_step = self.pyramid_reference_step_split(
                                patch_x.clone(),
                                patch_coarse_query.clone(),
                                patch_global_query.clone(),
                                v8.clone(),
                                coarse_x.clone(),
                                coarse_global_query,
                                v32,
                                current_patch_rho.clone(),
                                current_coarse_rho,
                                current_hub_rho,
                                hub_w8.clone(),
                                hub_w32.clone(),
                                ones_patch_decay.clone(),
                                ones_coarse_decay.clone(),
                                ones_global_decay.clone(),
                                &bridge_bank_mode,
                            );
                            current_coarse_state = self.pyramid_update_state(
                                current_coarse_state,
                                coarse_x,
                                rho_step.coarse_local_context + rho_step.coarse_from_hub_context,
                                pyramid_coarse_y_gate_proj,
                                pyramid_coarse_delta_proj,
                                pyramid_value_norm,
                            );
                            current_coarse_rho = rho_step.next_coarse_rho;
                            current_hub_rho = rho_step.next_hub_rho;
                        }
                    }
                    VisionTrmPredictSubstepKind::LocalBridge => {
                        let local_bridge_backend =
                            !matches!(executor_mode, PyramidRolloutExecutorMode::HostLoop);
                        let bridge_pair_projection =
                            if local_bridge_backend {
                                Some(self.project_local_bridge_pair_with_plan(
                                    current_patch_state.clone(),
                                    current_coarse_state.clone(),
                                    local_bridge_projection_pair_plan.as_ref().expect(
                                        "local bridge backend requires projection pair plan",
                                    ),
                                ))
                            } else {
                                None
                            };
                        let (patch_x, patch_coarse_query, patch_global_query, v8, _) =
                            if local_bridge_backend {
                                let (patch_x, patch_value, _, _) = bridge_pair_projection
                                    .clone()
                                    .expect("local bridge backend requires pair projection output");
                                (
                                    activation::relu(patch_x),
                                    Tensor::<B, 4>::zeros(
                                        [
                                            current_patch_state.shape().dims::<4>()[0].max(1),
                                            patch_rank.max(1),
                                            current_patch_state.shape().dims::<4>()[2].max(1),
                                            current_patch_state.shape().dims::<4>()[3].max(1),
                                        ],
                                        &current_patch_state.device(),
                                    ),
                                    Tensor::<B, 4>::zeros(
                                        [
                                            current_patch_state.shape().dims::<4>()[0].max(1),
                                            patch_rank.max(1),
                                            current_patch_state.shape().dims::<4>()[2].max(1),
                                            current_patch_state.shape().dims::<4>()[3].max(1),
                                        ],
                                        &current_patch_state.device(),
                                    ),
                                    patch_value,
                                    None,
                                )
                            } else {
                                project_patch(
                                    current_patch_state.clone(),
                                    patch_bridge_projection.0,
                                    patch_bridge_projection.1,
                                    patch_bridge_projection.2,
                                    patch_bridge_projection.3,
                                    patch_bridge_projection.4,
                                    patch_bridge_projection.5.as_ref(),
                                )
                            };
                        let (coarse_x, coarse_global_query, v32, _) = if local_bridge_backend {
                            let (_, _, coarse_x, coarse_value) = bridge_pair_projection
                                .expect("local bridge backend requires pair projection output");
                            (
                                activation::relu(coarse_x),
                                Tensor::<B, 4>::zeros(
                                    [
                                        current_coarse_state.shape().dims::<4>()[0].max(1),
                                        coarse_rank.max(1),
                                        current_coarse_state.shape().dims::<4>()[2].max(1),
                                        current_coarse_state.shape().dims::<4>()[3].max(1),
                                    ],
                                    &current_coarse_state.device(),
                                ),
                                coarse_value,
                                None,
                            )
                        } else {
                            project_coarse(
                                current_coarse_state.clone(),
                                coarse_bridge_projection.0,
                                coarse_bridge_projection.1,
                                coarse_bridge_projection.2,
                                coarse_bridge_projection.3,
                                coarse_bridge_projection.4.as_ref(),
                            )
                        };
                        let rho_step = if local_bridge_backend {
                            let (patch_local_context, next_patch_rho) = self
                                .pyramid_local_step_with_plan(
                                    patch_x.clone(),
                                    v8.clone(),
                                    current_patch_rho.clone(),
                                    pyramid_shape.patch,
                                    ones_patch_decay.clone(),
                                    patch_local_bridge_plan.as_ref(),
                                    patch_neighborhood,
                                    false,
                                    bridge_bank_mode.patch_local_read,
                                    bridge_bank_mode.patch_local_write,
                                );
                            let (coarse_local_context, next_coarse_rho) = self
                                .pyramid_local_step_with_plan(
                                    coarse_x.clone(),
                                    v32.clone(),
                                    current_coarse_rho.clone(),
                                    pyramid_shape.coarse,
                                    ones_coarse_decay.clone(),
                                    coarse_local_bridge_plan.as_ref(),
                                    coarse_neighborhood,
                                    true,
                                    bridge_bank_mode.coarse_local_read,
                                    bridge_bank_mode.coarse_local_write,
                                );
                            StructuredPyramidRhoStepOutput {
                                patch_local_context,
                                coarse_local_context,
                                patch_from_coarse_context: Tensor::<B, 4>::zeros(
                                    [
                                        current_patch_state.shape().dims::<4>()[0].max(1),
                                        self.trm_graph.value_dim.max(1),
                                        current_patch_state.shape().dims::<4>()[2].max(1),
                                        current_patch_state.shape().dims::<4>()[3].max(1),
                                    ],
                                    &current_patch_state.device(),
                                ),
                                patch_from_hub_context: Tensor::<B, 4>::zeros(
                                    [
                                        current_patch_state.shape().dims::<4>()[0].max(1),
                                        self.trm_graph.value_dim.max(1),
                                        current_patch_state.shape().dims::<4>()[2].max(1),
                                        current_patch_state.shape().dims::<4>()[3].max(1),
                                    ],
                                    &current_patch_state.device(),
                                ),
                                coarse_from_hub_context: Tensor::<B, 4>::zeros(
                                    [
                                        current_coarse_state.shape().dims::<4>()[0].max(1),
                                        self.trm_graph.value_dim.max(1),
                                        current_coarse_state.shape().dims::<4>()[2].max(1),
                                        current_coarse_state.shape().dims::<4>()[3].max(1),
                                    ],
                                    &current_coarse_state.device(),
                                ),
                                next_patch_rho,
                                next_coarse_rho,
                                next_hub_rho: current_hub_rho,
                            }
                        } else if let Some(plan) = split_fused_plan.as_ref() {
                            try_fused_structured_pyramid_split_step_wgpu_with_plan(
                                pyramid_shape,
                                StructuredPyramidSplitRhoStepInput {
                                    patch_local_query: patch_x.clone(),
                                    patch_query_for_coarse: patch_coarse_query.clone(),
                                    patch_query_for_global: patch_global_query.clone(),
                                    patch_value: v8.clone(),
                                    coarse_local_query: coarse_x.clone(),
                                    coarse_query_for_global: coarse_global_query.clone(),
                                    coarse_value: v32.clone(),
                                    patch_rho: current_patch_rho.clone(),
                                    coarse_rho: current_coarse_rho.clone(),
                                    hub_rho: current_hub_rho.clone(),
                                    patch_hub_weights: None,
                                    coarse_hub_weights: None,
                                    patch_decay: ones_patch_decay.clone(),
                                    coarse_decay: ones_coarse_decay.clone(),
                                    global_decay: ones_global_decay.clone(),
                                    bank_mode: Self::pyramid_bank_mode_for_kernel(
                                        &bridge_bank_mode,
                                    ),
                                },
                                plan,
                            )
                            .unwrap_or_else(|| {
                                self.pyramid_reference_step_split(
                                    patch_x.clone(),
                                    patch_coarse_query.clone(),
                                    patch_global_query.clone(),
                                    v8.clone(),
                                    coarse_x.clone(),
                                    coarse_global_query.clone(),
                                    v32.clone(),
                                    current_patch_rho,
                                    current_coarse_rho,
                                    current_hub_rho,
                                    None,
                                    None,
                                    ones_patch_decay.clone(),
                                    ones_coarse_decay.clone(),
                                    ones_global_decay.clone(),
                                    &bridge_bank_mode,
                                )
                            })
                        } else if let Some(plan) = stage_aware_plan.as_ref() {
                            self.pyramid_stage_aware_step_split_with_plan(
                                patch_x.clone(),
                                patch_coarse_query.clone(),
                                patch_global_query.clone(),
                                v8.clone(),
                                coarse_x.clone(),
                                coarse_global_query.clone(),
                                v32.clone(),
                                current_patch_rho,
                                current_coarse_rho,
                                current_hub_rho,
                                None,
                                None,
                                ones_patch_decay.clone(),
                                ones_coarse_decay.clone(),
                                ones_global_decay.clone(),
                                &bridge_bank_mode,
                                plan,
                            )
                        } else {
                            self.pyramid_reference_step_split(
                                patch_x.clone(),
                                patch_coarse_query.clone(),
                                patch_global_query.clone(),
                                v8.clone(),
                                coarse_x.clone(),
                                coarse_global_query.clone(),
                                v32.clone(),
                                current_patch_rho,
                                current_coarse_rho,
                                current_hub_rho,
                                None,
                                None,
                                ones_patch_decay.clone(),
                                ones_coarse_decay.clone(),
                                ones_global_decay.clone(),
                                &bridge_bank_mode,
                            )
                        };

                        if let Some(plan) = paired_update_plan.as_ref() {
                            (current_patch_state, current_coarse_state) = self
                                .pyramid_update_states_separate_with_plan(
                                    current_patch_state,
                                    patch_x,
                                    rho_step.patch_local_context
                                        + rho_step.patch_from_coarse_context
                                        + rho_step.patch_from_hub_context,
                                    current_coarse_state,
                                    coarse_x,
                                    rho_step.coarse_local_context
                                        + rho_step.coarse_from_hub_context,
                                    pyramid_value_norm,
                                    plan,
                                );
                        } else {
                            current_patch_state = self.pyramid_update_state(
                                current_patch_state,
                                patch_x,
                                rho_step.patch_local_context
                                    + rho_step.patch_from_coarse_context
                                    + rho_step.patch_from_hub_context,
                                pyramid_patch_y_gate_proj,
                                pyramid_patch_delta_proj,
                                pyramid_value_norm,
                            );
                            current_coarse_state = self.pyramid_update_state(
                                current_coarse_state,
                                coarse_x,
                                rho_step.coarse_local_context + rho_step.coarse_from_hub_context,
                                pyramid_coarse_y_gate_proj,
                                pyramid_coarse_delta_proj,
                                pyramid_value_norm,
                            );
                        }
                        current_patch_rho = rho_step.next_patch_rho;
                        current_coarse_rho = rho_step.next_coarse_rho;
                        current_hub_rho = rho_step.next_hub_rho;
                    }
                }
            }
            let (patch_x, patch_coarse_query, patch_global_query, v8, hub_w8) = project_patch(
                current_patch_state.clone(),
                patch_predict_projection.0,
                patch_predict_projection.1,
                patch_predict_projection.2,
                patch_predict_projection.3,
                patch_predict_projection.4,
                patch_predict_projection.5.as_ref(),
            );
            let (coarse_x, coarse_global_query, v32, hub_w32) = project_coarse(
                current_coarse_state.clone(),
                coarse_predict_projection.0,
                coarse_predict_projection.1,
                coarse_predict_projection.2,
                coarse_predict_projection.3,
                coarse_predict_projection.4.as_ref(),
            );
            let rho_step = if let Some(plan) = split_fused_plan.as_ref() {
                try_fused_structured_pyramid_split_step_wgpu_with_plan(
                    pyramid_shape,
                    StructuredPyramidSplitRhoStepInput {
                        patch_local_query: patch_x.clone(),
                        patch_query_for_coarse: patch_coarse_query.clone(),
                        patch_query_for_global: patch_global_query.clone(),
                        patch_value: v8.clone(),
                        coarse_local_query: coarse_x.clone(),
                        coarse_query_for_global: coarse_global_query.clone(),
                        coarse_value: v32.clone(),
                        patch_rho: current_patch_rho.clone(),
                        coarse_rho: current_coarse_rho.clone(),
                        hub_rho: current_hub_rho.clone(),
                        patch_hub_weights: hub_w8.clone(),
                        coarse_hub_weights: hub_w32.clone(),
                        patch_decay: patch_decay.clone(),
                        coarse_decay: coarse_decay.clone(),
                        global_decay: global_decay.clone(),
                        bank_mode: Self::pyramid_bank_mode_for_kernel(&bank_mode),
                    },
                    plan,
                )
                .unwrap_or_else(|| {
                    self.pyramid_reference_step_split(
                        patch_x.clone(),
                        patch_coarse_query.clone(),
                        patch_global_query.clone(),
                        v8.clone(),
                        coarse_x.clone(),
                        coarse_global_query.clone(),
                        v32.clone(),
                        current_patch_rho,
                        current_coarse_rho,
                        current_hub_rho,
                        hub_w8.clone(),
                        hub_w32.clone(),
                        patch_decay.clone(),
                        coarse_decay.clone(),
                        global_decay.clone(),
                        &bank_mode,
                    )
                })
            } else if let Some(plan) = stage_aware_plan.as_ref() {
                self.pyramid_stage_aware_step_split_with_plan(
                    patch_x.clone(),
                    patch_coarse_query.clone(),
                    patch_global_query.clone(),
                    v8.clone(),
                    coarse_x.clone(),
                    coarse_global_query.clone(),
                    v32.clone(),
                    current_patch_rho,
                    current_coarse_rho,
                    current_hub_rho,
                    hub_w8.clone(),
                    hub_w32.clone(),
                    patch_decay.clone(),
                    coarse_decay.clone(),
                    global_decay.clone(),
                    &bank_mode,
                    plan,
                )
            } else if fused_plan.is_some() {
                self.pyramid_rho_step_with_plan(
                    pyramid_shape,
                    StructuredPyramidRhoStepInput {
                        patch_query: patch_x.clone(),
                        patch_value: v8.clone(),
                        coarse_query: coarse_x.clone(),
                        coarse_value: v32.clone(),
                        patch_rho: current_patch_rho,
                        coarse_rho: current_coarse_rho,
                        hub_rho: current_hub_rho,
                        patch_hub_weights: hub_w8.clone(),
                        coarse_hub_weights: hub_w32.clone(),
                        neighborhood: self.resolve_rho_stream_neighborhood(),
                        decay: patch_decay.clone(),
                    },
                    fused_plan.as_ref(),
                )
            } else {
                self.pyramid_reference_step_split(
                    patch_x.clone(),
                    patch_coarse_query.clone(),
                    patch_global_query.clone(),
                    v8.clone(),
                    coarse_x.clone(),
                    coarse_global_query.clone(),
                    v32.clone(),
                    current_patch_rho,
                    current_coarse_rho,
                    current_hub_rho,
                    hub_w8.clone(),
                    hub_w32.clone(),
                    patch_decay.clone(),
                    coarse_decay.clone(),
                    global_decay.clone(),
                    &bank_mode,
                )
            };

            let (next_patch_state, next_coarse_state) =
                if let Some(plan) = paired_update_plan.as_ref() {
                    self.pyramid_update_states_separate_with_plan(
                        current_patch_state,
                        patch_x.clone(),
                        rho_step.patch_local_context.clone()
                            + rho_step.patch_from_coarse_context.clone()
                            + rho_step.patch_from_hub_context.clone(),
                        current_coarse_state,
                        coarse_x.clone(),
                        rho_step.coarse_local_context.clone()
                            + rho_step.coarse_from_hub_context.clone(),
                        pyramid_value_norm,
                        plan,
                    )
                } else {
                    (
                        self.pyramid_update_state(
                            current_patch_state,
                            patch_x.clone(),
                            rho_step.patch_local_context.clone()
                                + rho_step.patch_from_coarse_context.clone()
                                + rho_step.patch_from_hub_context.clone(),
                            pyramid_patch_y_gate_proj,
                            pyramid_patch_delta_proj,
                            pyramid_value_norm,
                        ),
                        self.pyramid_update_state(
                            current_coarse_state,
                            coarse_x.clone(),
                            rho_step.coarse_local_context.clone()
                                + rho_step.coarse_from_hub_context.clone(),
                            pyramid_coarse_y_gate_proj,
                            pyramid_coarse_delta_proj,
                            pyramid_value_norm,
                        ),
                    )
                };

            *state.primary_state_mut() = if step_idx < detach_until {
                next_patch_state.detach()
            } else {
                next_patch_state
            };
            *state.context_state_mut() = if step_idx < detach_until {
                next_coarse_state.detach()
            } else {
                next_coarse_state
            };
            *state.patch_rho_mut() = if step_idx < detach_until {
                rho_step.next_patch_rho.detach()
            } else {
                rho_step.next_patch_rho
            };
            *state.coarse_rho_mut() = if step_idx < detach_until {
                rho_step.next_coarse_rho.detach()
            } else {
                rho_step.next_coarse_rho
            };
            *state.hub_rho_mut() = if step_idx < detach_until {
                rho_step.next_hub_rho.detach()
            } else {
                rho_step.next_hub_rho
            };

            let completed_step = step_idx + 1;
            while next_checkpoint_index < checkpoint_steps.len()
                && checkpoint_steps[next_checkpoint_index] == completed_step
            {
                let mut checkpoint = state.clone();
                checkpoint.temporal_position = final_checkpoint_metadata.0;
                checkpoint.prediction_age = final_checkpoint_metadata.1;
                captured_checkpoints.push((completed_step, checkpoint));
                next_checkpoint_index += 1;
            }
        }

        if temporal_dt > 0 {
            state.temporal_position = state.temporal_position.saturating_add(temporal_dt);
            state.prediction_age = state.prediction_age.saturating_add(temporal_dt);
        } else if mode.resets_prediction_age() {
            state.prediction_age = 0;
        }

        (state, captured_checkpoints)
    }

    fn pyramid_backbone_fallback(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        detach_until: usize,
        reason: &str,
    ) -> Tensor<B, 3> {
        match self.trm_graph.grid_mismatch_policy {
            VisionTrmGridMismatchPolicy::FallbackDefault => {
                self.encode_tokens_steps_inner_default(tokens, steps, detach_until, false)
            }
            VisionTrmGridMismatchPolicy::Error => {
                panic!(
                    "pyramid backbone path unavailable: {reason}. Set `vision.trm_graph.grid_mismatch_policy = \"fallback_default\"` to allow explicit fallback."
                )
            }
        }
    }

    pub(super) fn encode_tokens_steps_inner_pyramid(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        detach_until: usize,
        add_cls: bool,
    ) -> Tensor<B, 3> {
        let tokens = if add_cls && self.use_cls_token {
            self.prepend_cls(tokens)
        } else {
            tokens
        };

        let [batch, time, dim] = tokens.shape().dims::<3>();
        if batch == 0 || time == 0 || dim == 0 {
            return tokens;
        }

        let grid_height = self.grid_height.max(1);
        let grid_width = self.grid_width.max(1);
        let patch_count = grid_height * grid_width;
        let (patch_tokens, has_cls) = if self.use_cls_token && time == patch_count + 1 {
            let patch = tokens.clone().slice_dim(1, 1..time);
            (patch, true)
        } else {
            (tokens.clone(), false)
        };

        let patch_len = patch_tokens.shape().dims::<3>()[1];
        if patch_len != patch_count {
            return self.pyramid_backbone_fallback(
                tokens,
                steps,
                detach_until,
                &format!(
                    "token count mismatch (got {patch_len}, expected {patch_count} from grid {}x{})",
                    grid_height, grid_width
                ),
            );
        }

        if self.pyramid_patch_x_neuron_proj.is_none()
            || self.pyramid_patch_to_coarse_query_proj.is_none()
            || self.pyramid_patch_to_global_query_proj.is_none()
            || self.pyramid_coarse_x_neuron_proj.is_none()
            || self.pyramid_coarse_to_global_query_proj.is_none()
        {
            return self.pyramid_backbone_fallback(
                tokens,
                steps,
                detach_until,
                "missing pyramid backbone query projection layers",
            );
        }
        if self.pyramid_write_value_proj.is_none() {
            return self.pyramid_backbone_fallback(
                tokens,
                steps,
                detach_until,
                "missing pyramid backbone projection layer `pyramid_write_value_proj`",
            );
        }
        if self.pyramid_patch_y_gate_proj.is_none() || self.pyramid_coarse_y_gate_proj.is_none() {
            return self.pyramid_backbone_fallback(
                tokens,
                steps,
                detach_until,
                "missing pyramid backbone y-gate projection layers",
            );
        }
        if self.pyramid_patch_delta_proj.is_none() || self.pyramid_coarse_delta_proj.is_none() {
            return self.pyramid_backbone_fallback(
                tokens,
                steps,
                detach_until,
                "missing pyramid backbone delta projection layers",
            );
        }
        if self.pyramid_value_norm.is_none() {
            return self.pyramid_backbone_fallback(
                tokens,
                steps,
                detach_until,
                "missing pyramid backbone normalization layer `pyramid_value_norm`",
            );
        }

        let backprop_steps = steps.saturating_sub(detach_until).max(1);
        let state = self.pyramid_state_from_patch_tokens(patch_tokens);
        let state = self.forward_pyramid_state_rollout_mode_unbounded(
            state,
            steps,
            backprop_steps,
            StructuredStepMode::Predict,
        );
        let readout_tokens = self.pyramid_readout_tokens(&state);
        if has_cls {
            readout_tokens
        } else {
            readout_tokens
                .slice_dim(1, 1..patch_count + 1)
                .reshape([batch, patch_count, dim])
        }
    }
}
