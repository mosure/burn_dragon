use crate::train::prelude::*;
use burn_dragon_train::train::runtime::{
    DeviceMemoryUsage, cleanup_device_memory, device_memory_usage_safe,
};

#[derive(Debug, Clone, Serialize)]
pub struct StartupAutotuneProbe {
    pub batch_size: usize,
    pub reserved_mb: Option<f64>,
    pub in_use_mb: Option<f64>,
    pub fit_target: bool,
    pub status: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct StartupAutotuneReport {
    pub backend_name: String,
    pub target_device_memory_mb: usize,
    pub target_effective_batch_size: Option<usize>,
    pub min_batch_size: usize,
    pub max_batch_size: usize,
    pub probe_steps: usize,
    pub resolved_batch_size: usize,
    pub resolved_gradient_accumulation_steps: usize,
    pub resolved_effective_batch_size: usize,
    pub probes: Vec<StartupAutotuneProbe>,
}

pub fn resolve_startup_batch_size<B>(
    config: &TrainingConfig,
    dataset: &Arc<Dataset>,
    backend_name: &str,
    device: &B::Device,
) -> Result<Option<StartupAutotuneReport>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone + 'static,
{
    let autotune = &config.wgpu.training.startup_autotune;
    if !autotune.enabled || !backend_name.starts_with("wgpu") {
        return Ok(None);
    }

    let min_batch_size = autotune.min_batch_size.max(1);
    let max_batch_size = autotune
        .max_batch_size
        .unwrap_or(config.training.batch_size)
        .max(min_batch_size);
    let target_effective_batch_size = config
        .training
        .target_effective_batch_size
        .filter(|value| *value > 0);
    let training_kernel_block_size =
        crate::train::utils::effective_training_kernel_block_size(&config.training);

    let tokenizer = dataset.tokenizer();
    let mut model_config = build_model_config_with_tokenizer(
        &config.model,
        training_kernel_block_size,
        tokenizer.as_ref(),
    )?;
    apply_wgpu_fused_core_override(
        &mut model_config,
        backend_name,
        WgpuFusedCoreOverride {
            recurrent: config.wgpu.training.fused_core_recurrent,
            rollout: config.wgpu.training.fused_core_rollout,
        },
    );
    let summary_event_token_ids = model_config.summary_memory.write_trigger_token_ids.clone();

    let mut probes = Vec::new();
    let target_bytes = (autotune.target_device_memory_mb as u64).saturating_mul(1024 * 1024);
    let (mut low, mut high) = (min_batch_size, max_batch_size);
    let mut best_fit = None;

    for candidate in startup_candidate_sequence(min_batch_size, max_batch_size) {
        let probe = probe_batch_size::<B>(ProbeBatchRequest {
            dataset,
            model_config: &model_config,
            block_size: config.training.block_size,
            tbptt_chunk_size: config.training.tbptt_chunk_size,
            batch_size: candidate,
            probe_steps: autotune.probe_steps.max(1),
            target_bytes,
            summary_event_token_ids: summary_event_token_ids.as_deref(),
            device,
        });
        let fit_target = probe.fit_target;
        probes.push(probe);
        if fit_target {
            best_fit = Some(candidate);
            low = candidate;
            if candidate == max_batch_size {
                break;
            }
        } else {
            high = candidate;
            break;
        }
    }

    if autotune.binary_search && best_fit.is_some() && high > low + 1 {
        while high > low + 1 {
            let candidate = low + ((high - low) / 2);
            let probe = probe_batch_size::<B>(ProbeBatchRequest {
                dataset,
                model_config: &model_config,
                block_size: config.training.block_size,
                tbptt_chunk_size: config.training.tbptt_chunk_size,
                batch_size: candidate,
                probe_steps: autotune.probe_steps.max(1),
                target_bytes,
                summary_event_token_ids: summary_event_token_ids.as_deref(),
                device,
            });
            let fit_target = probe.fit_target;
            probes.push(probe);
            if fit_target {
                best_fit = Some(candidate);
                low = candidate;
            } else {
                high = candidate;
            }
        }
    }

    let Some(resolved_batch_size) = best_fit else {
        return Err(anyhow!(
            "startup autotune could not find a safe batch size between {} and {} under target {} MiB; probes={}",
            min_batch_size,
            max_batch_size,
            autotune.target_device_memory_mb,
            format_probe_summary(&probes)
        ));
    };

    info!(
        "startup autotune: resolved batch_size={} grad_accumulation_steps={} effective_batch_size={} (target={} MiB, probed {} candidates)",
        resolved_batch_size,
        resolve_gradient_accumulation_steps(
            resolved_batch_size,
            config.training.gradient_accumulation_steps,
            target_effective_batch_size,
        ),
        resolved_batch_size.saturating_mul(resolve_gradient_accumulation_steps(
            resolved_batch_size,
            config.training.gradient_accumulation_steps,
            target_effective_batch_size,
        )),
        autotune.target_device_memory_mb,
        probes.len(),
    );

    let resolved_gradient_accumulation_steps = resolve_gradient_accumulation_steps(
        resolved_batch_size,
        config.training.gradient_accumulation_steps,
        target_effective_batch_size,
    );
    let resolved_effective_batch_size =
        resolved_batch_size.saturating_mul(resolved_gradient_accumulation_steps);

    Ok(Some(StartupAutotuneReport {
        backend_name: backend_name.to_string(),
        target_device_memory_mb: autotune.target_device_memory_mb,
        target_effective_batch_size,
        min_batch_size,
        max_batch_size,
        probe_steps: autotune.probe_steps.max(1),
        resolved_batch_size,
        resolved_gradient_accumulation_steps,
        resolved_effective_batch_size,
        probes,
    }))
}

fn probe_batch_size<B>(request: ProbeBatchRequest<'_, B>) -> StartupAutotuneProbe
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone + 'static,
{
    let ProbeBatchRequest {
        dataset,
        model_config,
        block_size,
        tbptt_chunk_size,
        batch_size,
        probe_steps,
        target_bytes,
        summary_event_token_ids,
        device,
    } = request;
    cleanup_device_memory::<B>(device, false);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let model = LanguageTrainModel::new(BDH::<B>::new(model_config.clone(), device))
            .with_tbptt_chunk_size(tbptt_chunk_size);
        let mut peak_usage: Option<DeviceMemoryUsage> = None;

        for _ in 0..probe_steps {
            let batch = sample_batch_with_shape::<B, _>(
                &**dataset,
                DatasetSplit::Train,
                batch_size,
                block_size,
                summary_event_token_ids,
                0,
                device,
            );
            let output = burn_train::TrainStep::step(&model, batch);
            drop(output);
            let _ = B::sync(device);
            if let Some(usage) = device_memory_usage_safe::<B>(device) {
                peak_usage = Some(match peak_usage {
                    Some(current)
                        if current.reserved_bytes.max(current.in_use_bytes)
                            >= usage.reserved_bytes.max(usage.in_use_bytes) =>
                    {
                        current
                    }
                    _ => usage,
                });
            }
        }

        drop(model);
        cleanup_device_memory::<B>(device, false);
        peak_usage
    }));

    match result {
        Ok(peak_usage) => {
            let fit_target = peak_usage
                .map(|usage| usage.reserved_bytes.max(usage.in_use_bytes) <= target_bytes)
                .unwrap_or(false);
            StartupAutotuneProbe {
                batch_size,
                reserved_mb: peak_usage.map(DeviceMemoryUsage::reserved_mb),
                in_use_mb: peak_usage.map(DeviceMemoryUsage::in_use_mb),
                fit_target,
                status: if fit_target {
                    "fit".to_string()
                } else {
                    "over_target".to_string()
                },
            }
        }
        Err(_) => {
            cleanup_device_memory::<B>(device, false);
            StartupAutotuneProbe {
                batch_size,
                reserved_mb: None,
                in_use_mb: None,
                fit_target: false,
                status: "probe_failed".to_string(),
            }
        }
    }
}

struct ProbeBatchRequest<'a, B: AutodiffBackend> {
    dataset: &'a Arc<Dataset>,
    model_config: &'a BDHConfig,
    block_size: usize,
    tbptt_chunk_size: Option<usize>,
    batch_size: usize,
    probe_steps: usize,
    target_bytes: u64,
    summary_event_token_ids: Option<&'a [u32]>,
    device: &'a B::Device,
}

fn startup_candidate_sequence(min_batch_size: usize, max_batch_size: usize) -> Vec<usize> {
    let mut candidates = Vec::new();
    let mut current = min_batch_size.max(1);
    candidates.push(current);

    while current < max_batch_size {
        let next = current.saturating_mul(2).min(max_batch_size);
        if next == current {
            break;
        }
        candidates.push(next);
        current = next;
    }

    candidates
}

pub fn resolve_gradient_accumulation_steps(
    resolved_batch_size: usize,
    configured_gradient_accumulation_steps: usize,
    target_effective_batch_size: Option<usize>,
) -> usize {
    match target_effective_batch_size {
        Some(target_effective_batch_size) => target_effective_batch_size
            .max(resolved_batch_size.max(1))
            .div_ceil(resolved_batch_size.max(1))
            .max(1),
        None => configured_gradient_accumulation_steps.max(1),
    }
}

fn format_probe_summary(probes: &[StartupAutotuneProbe]) -> String {
    probes
        .iter()
        .map(|probe| match (probe.reserved_mb, probe.in_use_mb) {
            (Some(reserved), Some(in_use)) => format!(
                "bs{}:{}:{reserved:.1}/{in_use:.1}MiB",
                probe.batch_size, probe.status
            ),
            _ => format!("bs{}:{}", probe.batch_size, probe.status),
        })
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::{resolve_gradient_accumulation_steps, startup_candidate_sequence};

    #[test]
    fn startup_candidate_sequence_doubles_and_caps_at_max() {
        assert_eq!(startup_candidate_sequence(4, 4), vec![4]);
        assert_eq!(startup_candidate_sequence(4, 20), vec![4, 8, 16, 20]);
        assert_eq!(startup_candidate_sequence(3, 24), vec![3, 6, 12, 24]);
    }

    #[test]
    fn resolve_gradient_accumulation_steps_ceil_divides_to_target_effective_batch() {
        assert_eq!(resolve_gradient_accumulation_steps(64, 1, None), 1);
        assert_eq!(resolve_gradient_accumulation_steps(64, 3, None), 3);
        assert_eq!(resolve_gradient_accumulation_steps(64, 1, Some(64)), 1);
        assert_eq!(resolve_gradient_accumulation_steps(21, 1, Some(64)), 4);
        assert_eq!(resolve_gradient_accumulation_steps(16, 1, Some(128)), 8);
    }
}
