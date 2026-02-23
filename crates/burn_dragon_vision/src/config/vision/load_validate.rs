use super::*;

pub fn load_vision_training_config(paths: &[PathBuf]) -> Result<VisionTrainingConfig> {
    if paths.is_empty() {
        return Err(anyhow!("at least one configuration path is required"));
    }

    let mut iter = paths.iter();
    let first_path = iter
        .next()
        .ok_or_else(|| anyhow!("configuration iterator unexpectedly empty"))?;
    let mut value = load_value(first_path)?;

    for path in iter {
        let overlay = load_value(path)?;
        merge_values(&mut value, overlay);
    }

    value
        .try_into::<VisionTrainingConfig>()
        .map_err(|err| anyhow!(err))
}

pub(super) fn validate_vision_rollout(
    training: &VisionTrainingHyperparameters,
    max_steps: usize,
) -> Result<()> {
    let max_steps = max_steps.max(1);
    let min_steps = training.rollout_min_steps.unwrap_or(max_steps);
    let max_steps_cfg = training.rollout_max_steps.unwrap_or(max_steps);
    let backprop_steps = training.rollout_backprop_steps.unwrap_or(max_steps_cfg);
    if min_steps == 0 || max_steps_cfg == 0 {
        return Err(anyhow!(
            "vision rollout steps must be > 0 (min={min_steps}, max={max_steps_cfg})"
        ));
    }
    if min_steps > max_steps_cfg {
        return Err(anyhow!(
            "vision rollout_min_steps ({min_steps}) must be <= rollout_max_steps ({max_steps_cfg})"
        ));
    }
    if max_steps_cfg > max_steps {
        return Err(anyhow!(
            "vision rollout_max_steps ({max_steps_cfg}) exceeds vision.steps ({max_steps})"
        ));
    }
    if backprop_steps > 0 && backprop_steps > max_steps_cfg {
        return Err(anyhow!(
            "vision rollout_backprop_steps ({backprop_steps}) must be <= rollout_max_steps ({max_steps_cfg})"
        ));
    }
    Ok(())
}

pub(super) fn validate_vision_mhc(vision: &VisionModelConfig) -> Result<()> {
    if !vision.mhc.enabled {
        return Ok(());
    }
    let num_eyes = vision.num_eyes.max(1);
    if vision.mhc.num_streams != 0 && vision.mhc.num_streams != num_eyes {
        return Err(anyhow!(
            "vision.mhc.num_streams ({}) must match vision.num_eyes ({})",
            vision.mhc.num_streams,
            num_eyes
        ));
    }
    if vision.mhc.num_views != 0 && vision.mhc.num_views != num_eyes {
        return Err(anyhow!(
            "vision.mhc.num_views ({}) must match vision.num_eyes ({})",
            vision.mhc.num_views,
            num_eyes
        ));
    }
    if vision.mhc.mhc_iters == 0 {
        return Err(anyhow!("vision.mhc.mhc_iters must be > 0"));
    }
    if vision.mhc.mhc_tau <= 0.0 {
        return Err(anyhow!(
            "vision.mhc.mhc_tau must be > 0 (got {})",
            vision.mhc.mhc_tau
        ));
    }
    if vision.mhc.dropout < 0.0 {
        return Err(anyhow!("vision.mhc.dropout must be >= 0"));
    }
    Ok(())
}

pub(super) fn validate_vision_trm_graph(vision: &VisionModelConfig) -> Result<()> {
    if !vision.trm_graph.enabled {
        return Ok(());
    }
    if vision.trm_graph.rank == 0 {
        return Err(anyhow!("vision.trm_graph.rank must be > 0"));
    }
    if vision.trm_graph.value_dim == 0 {
        return Err(anyhow!("vision.trm_graph.value_dim must be > 0"));
    }
    if vision.trm_graph.local_radius == 0 {
        return Err(anyhow!("vision.trm_graph.local_radius must be > 0"));
    }
    if vision.trm_graph.hub_count == 0 {
        return Err(anyhow!("vision.trm_graph.hub_count must be > 0"));
    }
    if vision.trm_graph.coarse_stride == 0 {
        return Err(anyhow!("vision.trm_graph.coarse_stride must be > 0"));
    }
    if !(0.0..=1.0).contains(&vision.trm_graph.decay) {
        return Err(anyhow!(
            "vision.trm_graph.decay must be in [0, 1] (got {})",
            vision.trm_graph.decay
        ));
    }
    let patch_size = vision.patch_size.max(1);
    let grid = vision.image_size.div_ceil(patch_size);
    let grid_h = vision.pos_max_height.unwrap_or(grid);
    let grid_w = vision.pos_max_width.unwrap_or(grid);
    if !grid_h.is_multiple_of(vision.trm_graph.coarse_stride) {
        return Err(anyhow!(
            "vision.trm_graph.coarse_stride ({}) must divide grid height ({})",
            vision.trm_graph.coarse_stride,
            grid_h
        ));
    }
    if !grid_w.is_multiple_of(vision.trm_graph.coarse_stride) {
        return Err(anyhow!(
            "vision.trm_graph.coarse_stride ({}) must divide grid width ({})",
            vision.trm_graph.coarse_stride,
            grid_w
        ));
    }
    Ok(())
}

pub(super) fn validate_vision_mode(
    mode: &VisionTrainingModeConfig,
    vision: &VisionModelConfig,
) -> Result<()> {
    match mode {
        VisionTrainingModeConfig::Distill(distill) => {
            validate_distill_loss(&distill.loss)?;
            match &distill.teacher {
                VisionTeacherConfig::Features(config) => {
                    if config.feature_dim == 0 {
                        return Err(anyhow!("mode.teacher.feature_dim must be > 0"));
                    }
                    if matches!(config.patch_tokens, Some(0)) {
                        return Err(anyhow!("mode.teacher.patch_tokens must be > 0 when set"));
                    }
                }
                VisionTeacherConfig::Model(config) => {
                    if matches!(config.image_size, Some(0)) {
                        return Err(anyhow!("mode.teacher.image_size must be > 0 when set"));
                    }
                    if matches!(config.patch_size, Some(0)) {
                        return Err(anyhow!("mode.teacher.patch_size must be > 0 when set"));
                    }
                    if matches!(config.feature_dim, Some(0)) {
                        return Err(anyhow!("mode.teacher.feature_dim must be > 0 when set"));
                    }
                    if matches!(config.patch_tokens, Some(0)) {
                        return Err(anyhow!("mode.teacher.patch_tokens must be > 0 when set"));
                    }
                }
            }
        }
        VisionTrainingModeConfig::Lejepa(lejepa) => {
            if lejepa.views == 0 {
                return Err(anyhow!("mode.views must be > 0"));
            }
            if lejepa.local_image_size == 0 {
                return Err(anyhow!("mode.local_image_size must be > 0"));
            }
            if !(0.0..=1.0).contains(&lejepa.local_min_scale) {
                return Err(anyhow!(
                    "mode.local_min_scale must be in [0, 1] (got {})",
                    lejepa.local_min_scale
                ));
            }
            if !(0.0..=1.0).contains(&lejepa.local_max_scale) {
                return Err(anyhow!(
                    "mode.local_max_scale must be in [0, 1] (got {})",
                    lejepa.local_max_scale
                ));
            }
            if lejepa.local_min_scale > lejepa.local_max_scale {
                return Err(anyhow!(
                    "mode.local_min_scale ({}) must be <= mode.local_max_scale ({})",
                    lejepa.local_min_scale,
                    lejepa.local_max_scale
                ));
            }
            if !(0.0..=1.0).contains(&lejepa.min_view_overlap) {
                return Err(anyhow!(
                    "mode.min_view_overlap must be in [0, 1] (got {})",
                    lejepa.min_view_overlap
                ));
            }
            if lejepa.view_overlap_attempts == 0 {
                return Err(anyhow!("mode.view_overlap_attempts must be > 0"));
            }
            if vision.trm_graph.enabled
                && vision.trm_graph.grid_mismatch_policy == VisionTrmGridMismatchPolicy::Error
                && lejepa.local_views > 0
            {
                let patch = vision.patch_size.max(1);
                let global_grid = vision.image_size.div_ceil(patch);
                let grid_h = vision.pos_max_height.unwrap_or(global_grid);
                let grid_w = vision.pos_max_width.unwrap_or(global_grid);
                let local_grid = lejepa.local_image_size.div_ceil(patch);
                if local_grid != grid_h || local_grid != grid_w {
                    return Err(anyhow!(
                        "TRM graph strict mode requires local view grid ({local_grid}x{local_grid}) to match vision grid ({grid_h}x{grid_w}); adjust mode.local_image_size / vision.patch_size / vision.pos_max_*, set mode.local_views=0, or set vision.trm_graph.grid_mismatch_policy = \"fallback_default\""
                    ));
                }
            }
            validate_lejepa_loss(&lejepa.loss.lejepa)?;
            validate_recon_loss("mode.loss.recon", &lejepa.loss.recon)?;
        }
        VisionTrainingModeConfig::Mae(mae) => {
            validate_recon_loss("mode.loss.recon", &mae.loss.recon)?;
            if mae.pyramid_levels == 0 {
                return Err(anyhow!("mode.pyramid_levels must be > 0"));
            }
            if mae.cross_view.enabled {
                let num_eyes = vision.num_eyes.max(1);
                if num_eyes < 2 {
                    return Err(anyhow!(
                        "vision.num_eyes must be >= 2 when cross_view is enabled"
                    ));
                }
                if !(0.0..=1.0).contains(&mae.cross_view.min_overlap) {
                    return Err(anyhow!(
                        "mode.cross_view.min_overlap must be in [0, 1] (got {})",
                        mae.cross_view.min_overlap
                    ));
                }
                if mae.cross_view.max_attempts == 0 {
                    return Err(anyhow!("mode.cross_view.max_attempts must be > 0"));
                }
                if !(0.0..=1.0).contains(&mae.cross_view.fuse_alpha) {
                    return Err(anyhow!(
                        "mode.cross_view.fuse_alpha must be in [0, 1] (got {})",
                        mae.cross_view.fuse_alpha
                    ));
                }
                if mae.cross_view.visible_weight < 0.0 {
                    return Err(anyhow!(
                        "mode.cross_view.visible_weight must be >= 0 (got {})",
                        mae.cross_view.visible_weight
                    ));
                }
                if mae.cross_view.masked_eye >= num_eyes {
                    return Err(anyhow!(
                        "mode.cross_view.masked_eye ({}) must be < vision.num_eyes ({})",
                        mae.cross_view.masked_eye,
                        num_eyes
                    ));
                }
                if !vision.mhc.enabled {
                    return Err(anyhow!(
                        "vision.mhc.enabled must be true when mode.cross_view.enabled is true"
                    ));
                }
                if vision.mhc.num_streams != 0 && vision.mhc.num_streams != num_eyes {
                    return Err(anyhow!(
                        "vision.mhc.num_streams ({}) must match vision.num_eyes ({})",
                        vision.mhc.num_streams,
                        num_eyes
                    ));
                }
                if vision.mhc.num_views != 0 && vision.mhc.num_views != num_eyes {
                    return Err(anyhow!(
                        "vision.mhc.num_views ({}) must match vision.num_eyes ({})",
                        vision.mhc.num_views,
                        num_eyes
                    ));
                }
            }
        }
        VisionTrainingModeConfig::Saccade(saccade) => {
            let num_eyes = if saccade.num_eyes == 0 {
                vision.num_eyes
            } else {
                saccade.num_eyes
            };
            if num_eyes == 0 {
                return Err(anyhow!("vision.num_eyes must be > 0"));
            }
            if saccade.num_eyes != 0 && saccade.num_eyes != vision.num_eyes {
                return Err(anyhow!(
                    "saccade.num_eyes ({}) must match vision.num_eyes ({})",
                    saccade.num_eyes,
                    vision.num_eyes
                ));
            }
            if saccade.traj_tokens == 0 {
                return Err(anyhow!("saccade.traj_tokens must be > 0"));
            }
            if !(0.0..=1.0).contains(&saccade.traj_update_alpha) {
                return Err(anyhow!(
                    "saccade.traj_update_alpha must be in [0, 1] (got {})",
                    saccade.traj_update_alpha
                ));
            }
            if saccade.mip_levels == 0 {
                return Err(anyhow!("saccade.mip_levels must be > 0"));
            }
            if saccade.inner_steps == 0 {
                return Err(anyhow!("saccade.inner_steps must be > 0"));
            }
            if saccade.fovea_subsamples == 0 {
                return Err(anyhow!("saccade.fovea_subsamples must be > 0"));
            }
            if saccade.fovea_radius_scale <= 0.0 {
                return Err(anyhow!(
                    "saccade.fovea_radius_scale must be > 0 (got {})",
                    saccade.fovea_radius_scale
                ));
            }
            if saccade.grid_sample_max_mb == 0 {
                return Err(anyhow!("saccade.grid_sample_max_mb must be > 0"));
            }
            if saccade.mip_concat_max_mb == 0 {
                return Err(anyhow!("saccade.mip_concat_max_mb must be > 0"));
            }
            if saccade.recon_max_elems == 0 {
                return Err(anyhow!("saccade.recon_max_elems must be > 0"));
            }
            validate_input_projection(&saccade.input_projection)?;
            if saccade.fovea_subpatch_size > 0 && saccade.fovea_subpatch_size > vision.patch_size {
                return Err(anyhow!(
                    "saccade.fovea_subpatch_size ({}) must be <= vision.patch_size ({})",
                    saccade.fovea_subpatch_size,
                    vision.patch_size
                ));
            }
            if matches!(saccade.pyramid_feature_dim, Some(0)) {
                return Err(anyhow!("saccade.pyramid_feature_dim must be > 0 when set"));
            }
            if saccade.cache.max_entries == 0 {
                return Err(anyhow!("saccade.cache.max_entries must be > 0"));
            }
            if saccade.policy.info_reward.stride == 0 {
                return Err(anyhow!("saccade.policy.info_reward.stride must be > 0"));
            }
            if saccade.policy.location_embedding.quantize_bins < 2 {
                return Err(anyhow!(
                    "saccade.policy.location_embedding.quantize_bins must be >= 2"
                ));
            }
            validate_recon_loss("saccade.loss.recon", &saccade.loss.recon)?;
            validate_lejepa_loss(&saccade.loss.lejepa)?;
            if saccade.cross_view.enabled {
                if num_eyes < 2 {
                    return Err(anyhow!(
                        "vision.num_eyes must be >= 2 when mode.cross_view.enabled is true"
                    ));
                }
                if !(0.0..=1.0).contains(&saccade.cross_view.min_overlap) {
                    return Err(anyhow!(
                        "mode.cross_view.min_overlap must be in [0, 1] (got {})",
                        saccade.cross_view.min_overlap
                    ));
                }
                if saccade.cross_view.max_attempts == 0 {
                    return Err(anyhow!(
                        "mode.cross_view.max_attempts must be > 0 when mode.cross_view.enabled is true"
                    ));
                }
                if saccade.cross_view.masked_eye >= num_eyes {
                    return Err(anyhow!(
                        "mode.cross_view.masked_eye ({}) must be < vision.num_eyes ({})",
                        saccade.cross_view.masked_eye,
                        num_eyes
                    ));
                }
            }

            if saccade.policy.gdpo.enabled {
                if saccade.policy.gdpo.group_size == 0 {
                    return Err(anyhow!("saccade.policy.gdpo.group_size must be > 0"));
                }
                if saccade.policy.action_noise_std <= 0.0 {
                    return Err(anyhow!(
                        "saccade.policy.action_noise_std must be > 0 when gdpo is enabled"
                    ));
                }
                if saccade.policy.gdpo.hard_weight < 0.0 {
                    return Err(anyhow!("saccade.policy.gdpo.hard_weight must be >= 0"));
                }
                if saccade.policy.gdpo.easy_weight < 0.0 {
                    return Err(anyhow!("saccade.policy.gdpo.easy_weight must be >= 0"));
                }
                if saccade.policy.gdpo.policy_weight < 0.0 {
                    return Err(anyhow!("saccade.policy.gdpo.policy_weight must be >= 0"));
                }
                if saccade.policy.gdpo.policy_clip_range < 0.0 {
                    return Err(anyhow!(
                        "saccade.policy.gdpo.policy_clip_range must be >= 0"
                    ));
                }
                if saccade.policy.gdpo.advantage_clip < 0.0 {
                    return Err(anyhow!(
                        "saccade.policy.gdpo.advantage_clip must be >= 0 (got {})",
                        saccade.policy.gdpo.advantage_clip
                    ));
                }
                if !(0.0..1.0).contains(&saccade.policy.gdpo.advantage_ema_decay) {
                    return Err(anyhow!(
                        "saccade.policy.gdpo.advantage_ema_decay must be in [0, 1) (got {})",
                        saccade.policy.gdpo.advantage_ema_decay
                    ));
                }
                if let GdpoHardGate::Percentile { quantile } = saccade.policy.gdpo.hard_gate
                    && !(0.0..=1.0).contains(&quantile)
                {
                    return Err(anyhow!(
                        "saccade.policy.gdpo.hard_gate.quantile must be in [0, 1] (got {})",
                        quantile
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_input_projection(config: &VisionSaccadeInputProjectionConfig) -> Result<()> {
    match config {
        VisionSaccadeInputProjectionConfig::Linear => Ok(()),
        VisionSaccadeInputProjectionConfig::Cnn(cfg) => {
            if matches!(cfg.channels, Some(0)) {
                return Err(anyhow!(
                    "saccade.input_projection.channels must be > 0 when set"
                ));
            }
            if cfg.expansion == 0 {
                return Err(anyhow!("saccade.input_projection.expansion must be > 0"));
            }
            if cfg.kernel != 0 && cfg.kernel % 2 == 0 {
                return Err(anyhow!(
                    "saccade.input_projection.kernel must be odd when set"
                ));
            }
            Ok(())
        }
        VisionSaccadeInputProjectionConfig::RadialMicroVit(cfg) => {
            if cfg.mlp_ratio == 0 {
                return Err(anyhow!("saccade.input_projection.mlp_ratio must be > 0"));
            }
            if cfg.radial_scale <= 0.0 {
                return Err(anyhow!("saccade.input_projection.radial_scale must be > 0"));
            }
            Ok(())
        }
    }
}

fn validate_recon_loss(label: &str, loss: &VisionReconLossConfig) -> Result<()> {
    if !(0.0..=1.0).contains(&loss.mask_ratio) {
        return Err(anyhow!(
            "{label}.mask_ratio must be in [0, 1] (got {})",
            loss.mask_ratio
        ));
    }
    if loss.weight < 0.0 {
        return Err(anyhow!("{label}.weight must be >= 0"));
    }
    Ok(())
}

fn validate_lejepa_loss(loss: &VisionLejepaLossConfig) -> Result<()> {
    if loss.enabled {
        if !(0.0..=1.0).contains(&loss.lambda) {
            return Err(anyhow!(
                "mode.loss.lejepa.lambda must be in [0, 1] (got {})",
                loss.lambda
            ));
        }
        if loss.sigreg_knots == 0 {
            return Err(anyhow!("mode.loss.lejepa.sigreg_knots must be > 0"));
        }
        if loss.sigreg_t_max <= 0.0 {
            return Err(anyhow!("mode.loss.lejepa.sigreg_t_max must be > 0"));
        }
        if loss.sigreg_proj_dim == 0 {
            return Err(anyhow!("mode.loss.lejepa.sigreg_proj_dim must be > 0"));
        }
    }
    Ok(())
}

fn validate_distill_loss(loss: &VisionDistillationLossConfig) -> Result<()> {
    if loss.patch_mse_weight < 0.0 {
        return Err(anyhow!("mode.loss.patch_mse_weight must be >= 0"));
    }
    if loss.cls_mse_weight < 0.0 {
        return Err(anyhow!("mode.loss.cls_mse_weight must be >= 0"));
    }
    if loss.cls_cosine_weight < 0.0 {
        return Err(anyhow!("mode.loss.cls_cosine_weight must be >= 0"));
    }
    if loss.rel_weight < 0.0 {
        return Err(anyhow!("mode.loss.rel_weight must be >= 0"));
    }
    if loss.rel_tau <= 0.0 {
        return Err(anyhow!("mode.loss.rel_tau must be > 0"));
    }
    if matches!(loss.rel_sample_tokens, Some(0)) {
        return Err(anyhow!("mode.loss.rel_sample_tokens must be > 0 when set"));
    }
    Ok(())
}

fn load_value(path: &Path) -> Result<Value> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read configuration file {}", path.display()))?;
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "yml" || ext == "yaml" {
        let value: serde_yaml::Value = serde_yaml::from_str(&content)
            .with_context(|| format!("failed to parse {} as YAML", path.display()))?;
        return yaml_to_toml(value)
            .with_context(|| format!("failed to convert {} from YAML", path.display()));
    }
    let table: toml::value::Table = toml::from_str(&content)
        .with_context(|| format!("failed to parse {} as TOML", path.display()))?;
    Ok(Value::Table(table))
}

fn yaml_to_toml(value: serde_yaml::Value) -> Result<Value> {
    match value {
        serde_yaml::Value::Null => Err(anyhow!("null values are not supported in config")),
        serde_yaml::Value::Bool(value) => Ok(Value::Boolean(value)),
        serde_yaml::Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                Ok(Value::Integer(value))
            } else if let Some(value) = value.as_u64() {
                Ok(Value::Integer(value as i64))
            } else if let Some(value) = value.as_f64() {
                Ok(Value::Float(value))
            } else {
                Err(anyhow!("unsupported YAML number"))
            }
        }
        serde_yaml::Value::String(value) => Ok(Value::String(value)),
        serde_yaml::Value::Sequence(values) => {
            let mut out = Vec::with_capacity(values.len());
            for value in values {
                out.push(yaml_to_toml(value)?);
            }
            Ok(Value::Array(out))
        }
        serde_yaml::Value::Mapping(values) => {
            let mut table = toml::value::Table::new();
            for (key, value) in values {
                let key = match key {
                    serde_yaml::Value::String(value) => value,
                    serde_yaml::Value::Number(value) => value.to_string(),
                    serde_yaml::Value::Bool(value) => value.to_string(),
                    serde_yaml::Value::Null => "null".to_string(),
                    other => format!("{other:?}"),
                };
                table.insert(key, yaml_to_toml(value)?);
            }
            Ok(Value::Table(table))
        }
        serde_yaml::Value::Tagged(tagged) => Err(anyhow!(
            "tagged YAML values are not supported: {:?}",
            tagged.tag
        )),
    }
}

fn merge_values(base: &mut Value, overlay: Value) {
    match (base, overlay) {
        (Value::Table(base_table), Value::Table(overlay_table)) => {
            if let Some(Value::String(overlay_type)) = overlay_table.get("type") {
                let type_changed = match base_table.get("type") {
                    Some(Value::String(base_type)) => base_type != overlay_type,
                    Some(_) => true,
                    None => !base_table.is_empty(),
                };
                if type_changed {
                    base_table.clear();
                }
            }
            for (key, overlay_value) in overlay_table {
                match base_table.get_mut(&key) {
                    Some(base_value) => merge_values(base_value, overlay_value),
                    None => {
                        base_table.insert(key, overlay_value);
                    }
                }
            }
        }
        (base_value, overlay_value) => {
            *base_value = overlay_value;
        }
    }
}
