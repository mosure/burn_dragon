use crate::train::prelude::*;
use crate::train::startup_autotune::StartupAutotuneReport;

pub(crate) const LANGUAGE_ARCH_VERSION: &str = "dragon_bdh_v1";
pub(crate) const SHARD_LAYOUT_VERSION_UNSHARDED: u32 = 1;

#[derive(Clone)]
pub struct PreparedDatasets {
    pub train: Arc<Dataset>,
    pub valid: Arc<Dataset>,
}

pub fn build_vocab_only(config: &TrainingConfig) -> Result<()> {
    let datasets = prepare_datasets(&config.dataset, &config.training)?;
    let tokenizer = datasets.train.tokenizer();
    info!(
        "Tokenizer `{}` ready with {} tokens",
        config.dataset.tokenizer.kind_name(),
        tokenizer.len()
    );
    Ok(())
}

pub fn prepare_dataset(
    dataset_cfg: &DatasetConfig,
    training: &TrainingHyperparameters,
) -> Result<Arc<Dataset>> {
    Ok(prepare_datasets(dataset_cfg, training)?.train)
}

pub fn prepare_datasets(
    dataset_cfg: &DatasetConfig,
    training: &TrainingHyperparameters,
) -> Result<PreparedDatasets> {
    let tokenizer_path = dataset_cfg.tokenizer.storage_path(&dataset_cfg.cache_dir);
    let tokenizer_preexists = tokenizer_path
        .as_ref()
        .map(|path| path.is_file())
        .unwrap_or(false);

    let (dataset_enum, dataset_summary) = build_dataset(dataset_cfg, training)?;
    let dataset = Arc::new(dataset_enum);

    let tokenizer = dataset.tokenizer();
    match tokenizer_path {
        Some(path) if tokenizer_preexists => info!(
            "Loaded {} tokenizer with {} tokens from {}",
            dataset_cfg.tokenizer.kind_name(),
            tokenizer.len(),
            path.display()
        ),
        Some(path) => info!(
            "Built {} tokenizer with {} tokens at {}",
            dataset_cfg.tokenizer.kind_name(),
            tokenizer.len(),
            path.display()
        ),
        None => info!(
            "Initialized {} tokenizer with {} tokens (no persistence required)",
            dataset_cfg.tokenizer.kind_name(),
            tokenizer.len()
        ),
    };

    info!("{dataset_summary}");

    let valid = if let Some(validation_cfg) = &dataset_cfg.validation {
        let effective_cfg = build_validation_dataset_config(dataset_cfg, validation_cfg);
        let (dataset_enum, dataset_summary) = build_dataset(&effective_cfg, training)?;
        let dataset = Arc::new(dataset_enum);
        ensure_validation_tokenizer_compatible(
            tokenizer.as_ref(),
            dataset.tokenizer().as_ref(),
            &dataset_cfg.tokenizer.kind_name(),
        )?;
        info!("Prepared validation override dataset: {dataset_summary}");
        dataset
    } else {
        Arc::clone(&dataset)
    };

    Ok(PreparedDatasets {
        train: dataset,
        valid,
    })
}

fn build_validation_dataset_config(
    dataset_cfg: &DatasetConfig,
    validation_cfg: &ValidationDatasetConfig,
) -> DatasetConfig {
    DatasetConfig {
        cache_dir: validation_cfg
            .cache_dir
            .clone()
            .unwrap_or_else(|| dataset_cfg.cache_dir.join("validation")),
        train_split_ratio: validation_cfg
            .train_split_ratio
            .unwrap_or(dataset_cfg.train_split_ratio),
        validation: None,
        source: validation_cfg.source.clone(),
        tokenizer: dataset_cfg.tokenizer.clone(),
    }
}

fn ensure_validation_tokenizer_compatible(
    train_tokenizer: &dyn crate::tokenizer::Tokenizer,
    valid_tokenizer: &dyn crate::tokenizer::Tokenizer,
    tokenizer_label: &str,
) -> Result<()> {
    if train_tokenizer.len() != valid_tokenizer.len() {
        return Err(anyhow!(
            "validation dataset tokenizer is incompatible with the training tokenizer: vocab sizes differ (train={}, valid={}, tokenizer={tokenizer_label})",
            train_tokenizer.len(),
            valid_tokenizer.len(),
        ));
    }
    if train_tokenizer.bos_id() != valid_tokenizer.bos_id()
        || train_tokenizer.eos_id() != valid_tokenizer.eos_id()
        || train_tokenizer.pad_id() != valid_tokenizer.pad_id()
        || train_tokenizer.unk_id() != valid_tokenizer.unk_id()
    {
        return Err(anyhow!(
            "validation dataset tokenizer is incompatible with the training tokenizer: special token ids differ (tokenizer={tokenizer_label})"
        ));
    }
    Ok(())
}

pub fn log_theoretical_profile(config: &BDHConfig, batch: usize, block: usize, backend: &str) {
    let batch = batch as u64;
    let time = block as u64;
    let embed = config.n_embd as u64;
    let latent_per_head = config.latent_per_head() as u64;
    let latent_total = config.latent_total() as u64;
    let heads = config.n_head as u64;
    let bt = batch * time;

    let encoder_matmul = 2 * bt * embed * latent_total;
    let attn_scores = 2 * batch * heads * time * time * latent_per_head;
    let attn_value = 2 * batch * heads * time * time * embed;
    let decoder_matmul = 2 * bt * latent_total * embed;
    let total = encoder_matmul + attn_scores + attn_value + decoder_matmul;

    info!(
        "[train:{backend}] approx forward GFLOPs: total={total_gflops:.2}, encoder={enc:.2}, \
         attn_scores={scores:.2}, attn_value={value:.2}, decoder={dec:.2} (backward ~2x forward)",
        total_gflops = total as f64 / 1e9,
        enc = encoder_matmul as f64 / 1e9,
        scores = attn_scores as f64 / 1e9,
        value = attn_value as f64 / 1e9,
        dec = decoder_matmul as f64 / 1e9,
    );
}

#[derive(Serialize)]
pub struct RunConfigOutput {
    run_name: String,
    backend_name: String,
    arch_version: String,
    shard_layout_version: u32,
    block_size: usize,
    seed: u64,
    training_batch_size: usize,
    training_gradient_accumulation_steps: usize,
    training_effective_batch_size: usize,
    training_checkpoint_interval_iters: usize,
    training_execution_form: String,
    training_launch_mode_requested: burn_dragon_train::train::pipeline::TrainingLaunchMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    training_sequence_kernel_override: Option<SequenceKernelConfig>,
    optimizer_spec: OptimizerSpec,
    overrides: ModelOverrides,
    model_spec: ModelSpec,
    parallel_spec: ParallelSpec,
    kernel_spec: KernelSpec,
    state_layout: StateLayout,
    metrics_sink: MetricsSinkSpec,
    #[serde(skip_serializing_if = "Option::is_none")]
    startup_autotune: Option<StartupAutotuneReport>,
}

pub(crate) fn build_training_execution_form(config: &TrainingConfig) -> String {
    if config.parallel.pipeline.enabled {
        "pipeline".to_string()
    } else if config.training.tbptt_chunk_size.is_some() {
        "tbptt".to_string()
    } else {
        "default_stateful".to_string()
    }
}

pub(crate) fn effective_training_kernel_block_size(training: &TrainingHyperparameters) -> usize {
    training
        .tbptt_chunk_size
        .filter(|chunk| *chunk > 0 && *chunk < training.block_size)
        .unwrap_or(training.block_size)
        .max(1)
}

pub(crate) fn build_model_spec(model_config: &BDHConfig) -> ModelSpec {
    ModelSpec {
        arch: "dragon_bdh".to_string(),
        n_embd: model_config.n_embd,
        n_head: model_config.n_head,
        n_layer: model_config.n_layer,
        latent_total: model_config.latent_total(),
        latent_per_head: model_config.latent_per_head(),
        shared_layer_weights: true,
        sequence_kernel: model_config.sequence_kernel,
        bdh_initialization_kind: model_config.initialization.kind,
        bdh_residual_scaling_kind: model_config.initialization.residual_scaling.kind,
        bdh_neuron_gain_kind: model_config.initialization.neuron_gains.kind,
        bdh_topology_prior_kind: model_config.initialization.topology_prior.kind,
        bdh_firing_target_kind: model_config.initialization.firing_targets.kind,
        low_bit: model_config.quant.enable.then(|| LowBitModelSpec {
            enabled: model_config.quant.enable,
            protocol: model_config.quant.protocol,
            training_mode: model_config.quant.training_mode,
            inference_mode: model_config.quant.inference_mode,
            weight_format: model_config.quant.weight_format,
            activation_format: model_config.quant.act_format,
            decoder_x_mode: model_config.quant.decoder_x_mode,
            encoder_mode: model_config.quant.encoder_mode,
            activation_grouping: model_config.quant.act_grouping,
            weight_grouping: model_config.quant.weight_grouping,
            strict_bitnet_reference: model_config.quant.strict_bitnet_reference,
            target_modules: model_config.quant.target_modules.clone(),
            rho_precision: model_config.rho.precision,
            rho_compression: model_config.rho.compression,
        }),
    }
}

pub(crate) fn build_parallel_spec(config: &TrainingConfig) -> ParallelSpec {
    ParallelSpec {
        mode: config.parallel.mode,
        world_size: config.parallel.world_size,
        data_parallel_size: config.parallel.data.size,
        tensor_parallel_size: config.parallel.tensor.size,
        tensor_parallel_axis: config.parallel.tensor.axis,
        tensor_parallel_partition: config.parallel.tensor.partition,
        fsdp_enabled: config.parallel.fsdp.enabled,
        checkpoint_format: config.parallel.checkpoint.format,
        collective_num_nodes: config.parallel.data.collective_num_nodes,
        collective_global_address: config.parallel.data.collective_global_address.clone(),
        collective_node_address: config.parallel.data.collective_node_address.clone(),
        collective_data_service_port: config.parallel.data.collective_data_service_port,
        pipeline_enabled: config.parallel.pipeline.enabled,
        pipeline_stage_count: config.parallel.pipeline.stage_count,
        pipeline_virtual_stages_per_rank: config.parallel.pipeline.virtual_stages_per_rank,
        pipeline_schedule: config.parallel.pipeline.schedule,
        pipeline_microbatches: config.parallel.pipeline.microbatches,
        pipeline_partition: config.parallel.pipeline.partition,
        pipeline_activation_checkpointing: config.parallel.pipeline.activation_checkpointing,
        pipeline_shared_weight_sync: config.parallel.pipeline.shared_weight_sync,
        pipeline_communication: config.parallel.pipeline.communication,
        pipeline_cache_enabled: config.parallel.pipeline.cache.enabled,
        pipeline_cache_policy: config.parallel.pipeline.cache.policy,
        pipeline_cache_reuse_across_backward: config.parallel.pipeline.cache.reuse_across_backward,
        pipeline_cache_max_inflight_microbatches: config
            .parallel
            .pipeline
            .cache
            .max_inflight_microbatches,
        pipeline_cache_eviction: config.parallel.pipeline.cache.eviction,
        pipeline_cache_transport_dtype: config.parallel.pipeline.cache.transport_dtype,
    }
}

pub(crate) fn build_optimizer_spec(config: &TrainingConfig) -> OptimizerSpec {
    OptimizerSpec {
        name: config.optimizer.name,
        learning_rate: config.optimizer.learning_rate,
        weight_decay: config.optimizer.weight_decay,
        weight_decay_final: config.optimizer.weight_decay_final,
        schedule_mode: config.optimizer.schedule_mode,
    }
}

pub(crate) fn build_kernel_spec(
    config: &TrainingConfig,
    model_config: &BDHConfig,
    backend_name: &str,
) -> KernelSpec {
    let kernel_plan = burn_dragon_core::resolve_low_bit_kernel_plan_for_backend_name(
        backend_name,
        &model_config.quant,
        false,
    );
    let low_bit_memory = model_config.quant.enable.then(|| {
        let estimate = burn_dragon_core::estimate_low_bit_memory_buckets(
            &model_config.quant,
            &model_config.rho,
            burn_dragon_core::LowBitMemoryEstimateInput {
                batch_size: config.training.batch_size,
                time_steps: effective_training_kernel_block_size(&config.training),
                n_layer: model_config.n_layer,
                n_head: model_config.n_head,
                n_embd: model_config.n_embd,
                latent_total: model_config.latent_total(),
            },
        );
        LowBitMemorySpec {
            master_weight_bytes: estimate.master_weight_bytes,
            execution_weight_bytes: estimate.execution_weight_bytes,
            activation_shell_bytes: estimate.activation_shell_bytes,
            saved_activation_bytes: estimate.saved_activation_bytes,
            rho_state_bytes: estimate.rho_state_bytes,
            workspace_bytes: estimate.workspace_bytes,
            estimated_total_bytes: estimate.estimated_total_bytes(),
        }
    });
    let low_bit_inventory = model_config
        .quant
        .enable
        .then(|| {
            burn_dragon_core::build_low_bit_saved_activation_inventory(
                &model_config.quant,
                burn_dragon_core::LowBitMemoryEstimateInput {
                    batch_size: config.training.batch_size,
                    time_steps: effective_training_kernel_block_size(&config.training),
                    n_layer: model_config.n_layer,
                    n_head: model_config.n_head,
                    n_embd: model_config.n_embd,
                    latent_total: model_config.latent_total(),
                },
            )
        })
        .flatten()
        .map(
            |inventory| burn_dragon_train::LowBitSavedActivationInventorySpec {
                mode: inventory.mode,
                format: inventory.format.as_str().to_string(),
                requires_rho_window_anchor: inventory.requires_rho_window_anchor,
                tensors: inventory
                    .tensors
                    .into_iter()
                    .map(|entry| burn_dragon_train::LowBitSavedActivationTensorSpec {
                        name: entry.name,
                        shape: entry.shape,
                        element_count: entry.element_count,
                        estimated_bytes: entry.estimated_bytes,
                        recompute_policy: entry.recompute_policy.as_str().to_string(),
                    })
                    .collect(),
            },
        );

    KernelSpec {
        sequence_kernel: model_config.sequence_kernel,
        fused_kernels_enabled: model_config.fused_kernels.enabled,
        rollout_fast_steps_per_slow_step: model_config.rollout_fast_steps_per_slow_step,
        wgpu_fused_core_recurrent: config.wgpu.training.fused_core_recurrent,
        wgpu_fused_core_rollout: config.wgpu.training.fused_core_rollout,
        low_bit_kernel_abi_version: model_config.quant.enable.then_some(1),
        low_bit_runtime: model_config
            .quant
            .enable
            .then(|| kernel_plan.runtime.as_str().to_string()),
        low_bit_saved_activation_mode: model_config
            .quant
            .enable
            .then_some(model_config.quant.saved_activations.mode),
        low_bit_saved_activation_format: model_config.quant.enable.then(|| {
            model_config
                .quant
                .saved_activations
                .format
                .as_str()
                .to_string()
        }),
        low_bit_saved_activation_inventory: low_bit_inventory,
        low_bit_native_supported: model_config
            .quant
            .enable
            .then_some(kernel_plan.capabilities.any_native_supported()),
        low_bit_memory,
    }
}

pub(crate) fn build_state_layout(model_config: &BDHConfig) -> StateLayout {
    let stream_count = model_config.mhc.resolved_num_streams();
    let layers = (0..model_config.n_layer)
        .map(|layer_index| {
            let latent_total = model_config.latent_total_for_layer(layer_index);
            let latent_per_head = model_config.latent_per_head_for_layer(layer_index);
            let mut tensors = match model_config.sequence_kernel.memory_system {
                burn_dragon_core::SequenceMemorySystem::Mamba1SelectiveScan => {
                    let mamba = model_config.mamba.resolve(
                        model_config.n_embd,
                        burn_dragon_core::SequenceMemorySystem::Mamba1SelectiveScan,
                    );
                    vec![
                        StateTensorSpec {
                            name: "rho".to_string(),
                            axes: vec![
                                StateAxisSpec {
                                    name: "batch_views".to_string(),
                                    size: None,
                                },
                                StateAxisSpec {
                                    name: "streams".to_string(),
                                    size: Some(1),
                                },
                                StateAxisSpec {
                                    name: "mamba_inner".to_string(),
                                    size: Some(mamba.d_inner),
                                },
                                StateAxisSpec {
                                    name: "mamba_state".to_string(),
                                    size: Some(mamba.d_state),
                                },
                            ],
                        },
                        StateTensorSpec {
                            name: "sequence_aux".to_string(),
                            axes: vec![
                                StateAxisSpec {
                                    name: "batch_views".to_string(),
                                    size: None,
                                },
                                StateAxisSpec {
                                    name: "streams".to_string(),
                                    size: Some(1),
                                },
                                StateAxisSpec {
                                    name: "mamba_inner".to_string(),
                                    size: Some(mamba.d_inner),
                                },
                                StateAxisSpec {
                                    name: "mamba_conv".to_string(),
                                    size: Some(mamba.d_conv),
                                },
                            ],
                        },
                    ]
                }
                burn_dragon_core::SequenceMemorySystem::Mamba2StateSpaceDuality => {
                    let mamba = model_config.mamba.resolve(
                        model_config.n_embd,
                        burn_dragon_core::SequenceMemorySystem::Mamba2StateSpaceDuality,
                    );
                    vec![
                        StateTensorSpec {
                            name: "rho".to_string(),
                            axes: vec![
                                StateAxisSpec {
                                    name: "batch_views".to_string(),
                                    size: None,
                                },
                                StateAxisSpec {
                                    name: "mamba_heads".to_string(),
                                    size: Some(mamba.nheads),
                                },
                                StateAxisSpec {
                                    name: "mamba_head_dim".to_string(),
                                    size: Some(mamba.headdim),
                                },
                                StateAxisSpec {
                                    name: "mamba_state".to_string(),
                                    size: Some(mamba.d_state),
                                },
                            ],
                        },
                        StateTensorSpec {
                            name: "sequence_aux".to_string(),
                            axes: vec![
                                StateAxisSpec {
                                    name: "batch_views".to_string(),
                                    size: None,
                                },
                                StateAxisSpec {
                                    name: "streams".to_string(),
                                    size: Some(1),
                                },
                                StateAxisSpec {
                                    name: "mamba_conv_channels".to_string(),
                                    size: Some(mamba.mamba2_conv_dim()),
                                },
                                StateAxisSpec {
                                    name: "mamba_conv".to_string(),
                                    size: Some(mamba.d_conv),
                                },
                            ],
                        },
                    ]
                }
                burn_dragon_core::SequenceMemorySystem::Mamba3StateSpaceDuality => {
                    let mamba = model_config.mamba.resolve(
                        model_config.n_embd,
                        burn_dragon_core::SequenceMemorySystem::Mamba3StateSpaceDuality,
                    );
                    vec![
                        StateTensorSpec {
                            name: "rho".to_string(),
                            axes: vec![
                                StateAxisSpec {
                                    name: "batch_views".to_string(),
                                    size: None,
                                },
                                StateAxisSpec {
                                    name: "mamba_heads".to_string(),
                                    size: Some(mamba.nheads),
                                },
                                StateAxisSpec {
                                    name: "mamba_head_dim".to_string(),
                                    size: Some(mamba.headdim),
                                },
                                StateAxisSpec {
                                    name: "mamba_state".to_string(),
                                    size: Some(mamba.d_state),
                                },
                            ],
                        },
                        StateTensorSpec {
                            name: "mamba_angle_state".to_string(),
                            axes: vec![
                                StateAxisSpec {
                                    name: "batch_views".to_string(),
                                    size: None,
                                },
                                StateAxisSpec {
                                    name: "mamba_heads".to_string(),
                                    size: Some(mamba.nheads),
                                },
                                StateAxisSpec {
                                    name: "mamba_rope_angles".to_string(),
                                    size: Some(mamba.num_rope_angles),
                                },
                            ],
                        },
                        StateTensorSpec {
                            name: "mamba_k_state".to_string(),
                            axes: vec![
                                StateAxisSpec {
                                    name: "batch_views".to_string(),
                                    size: None,
                                },
                                StateAxisSpec {
                                    name: "mamba_heads".to_string(),
                                    size: Some(mamba.nheads),
                                },
                                StateAxisSpec {
                                    name: "mamba_state".to_string(),
                                    size: Some(mamba.d_state),
                                },
                            ],
                        },
                        StateTensorSpec {
                            name: "mamba_v_state".to_string(),
                            axes: vec![
                                StateAxisSpec {
                                    name: "batch_views".to_string(),
                                    size: None,
                                },
                                StateAxisSpec {
                                    name: "mamba_heads".to_string(),
                                    size: Some(mamba.nheads),
                                },
                                StateAxisSpec {
                                    name: "mamba_head_dim".to_string(),
                                    size: Some(mamba.headdim),
                                },
                            ],
                        },
                    ]
                }
                _ => {
                    vec![StateTensorSpec {
                        name: "rho".to_string(),
                        axes: vec![
                            StateAxisSpec {
                                name: "batch_views".to_string(),
                                size: None,
                            },
                            StateAxisSpec {
                                name: "heads".to_string(),
                                size: Some(model_config.n_head),
                            },
                            StateAxisSpec {
                                name: "latent_per_head".to_string(),
                                size: Some(latent_per_head),
                            },
                            StateAxisSpec {
                                name: "dense_dim".to_string(),
                                size: Some(model_config.n_embd),
                            },
                        ],
                    }]
                }
            };
            if model_config.sequence_kernel.memory_system
                == burn_dragon_core::SequenceMemorySystem::Rwkv8StateSpace
            {
                tensors.push(StateTensorSpec {
                    name: "rho_norm".to_string(),
                    axes: vec![
                        StateAxisSpec {
                            name: "batch_views".to_string(),
                            size: None,
                        },
                        StateAxisSpec {
                            name: "heads".to_string(),
                            size: Some(model_config.n_head),
                        },
                        StateAxisSpec {
                            name: "latent_per_head".to_string(),
                            size: Some(latent_per_head),
                        },
                    ],
                });
            }
            if model_config.y_neuron_recurrence.enabled {
                tensors.push(StateTensorSpec {
                    name: "y_neuron_state".to_string(),
                    axes: vec![
                        StateAxisSpec {
                            name: "batch_views".to_string(),
                            size: None,
                        },
                        StateAxisSpec {
                            name: "heads".to_string(),
                            size: Some(model_config.n_head),
                        },
                        StateAxisSpec {
                            name: "latent_per_head".to_string(),
                            size: Some(latent_per_head),
                        },
                    ],
                });
            }
            if model_config.clocked_slow_memory.enabled {
                tensors.push(StateTensorSpec {
                    name: "clocked_slow_hidden".to_string(),
                    axes: vec![
                        StateAxisSpec {
                            name: "batch".to_string(),
                            size: None,
                        },
                        StateAxisSpec {
                            name: "streams".to_string(),
                            size: Some(stream_count),
                        },
                        StateAxisSpec {
                            name: "time".to_string(),
                            size: Some(1),
                        },
                        StateAxisSpec {
                            name: "dense_dim".to_string(),
                            size: Some(model_config.n_embd),
                        },
                    ],
                });
            }
            if model_config.summary_memory.enabled {
                tensors.push(StateTensorSpec {
                    name: "summary_memory_hidden".to_string(),
                    axes: vec![
                        StateAxisSpec {
                            name: "batch".to_string(),
                            size: None,
                        },
                        StateAxisSpec {
                            name: "streams".to_string(),
                            size: Some(stream_count),
                        },
                        StateAxisSpec {
                            name: "time".to_string(),
                            size: Some(1),
                        },
                        StateAxisSpec {
                            name: "dense_dim".to_string(),
                            size: Some(model_config.n_embd),
                        },
                    ],
                });
            }
            LayerStateSpec {
                layer_index,
                latent_total,
                latent_per_head,
                tensors,
            }
        })
        .collect();

    StateLayout {
        state_family: "bdh_model_state".to_string(),
        position_tracked: true,
        layers,
    }
}

pub(crate) fn build_language_metrics_sink(metric_every: usize) -> MetricsSinkSpec {
    MetricsSinkSpec::new(
        "language_bdh_burn_train_v1",
        vec![
            MetricSinkEntry::new(
                "Loss",
                MetricSinkSplit::Train,
                MetricSinkValueKind::Numeric,
                metric_every,
            ),
            MetricSinkEntry::new(
                "Loss",
                MetricSinkSplit::Valid,
                MetricSinkValueKind::Numeric,
                metric_every,
            ),
            MetricSinkEntry::new(
                "Learning Rate",
                MetricSinkSplit::Train,
                MetricSinkValueKind::Numeric,
                metric_every,
            ),
            MetricSinkEntry::new(
                "device",
                MetricSinkSplit::Train,
                MetricSinkValueKind::Text,
                metric_every,
            ),
            MetricSinkEntry::new(
                "device",
                MetricSinkSplit::Valid,
                MetricSinkValueKind::Text,
                metric_every,
            ),
        ],
    )
}

pub fn write_run_config(
    config: &TrainingConfig,
    model_config: &BDHConfig,
    run_dir: &Path,
    run_name: &str,
    backend_name: &str,
    effective_training_sequence_kernel_override: Option<SequenceKernelConfig>,
    startup_autotune: Option<&StartupAutotuneReport>,
) -> Result<()> {
    fs::create_dir_all(run_dir)
        .with_context(|| format!("failed to create run directory {}", run_dir.display()))?;

    let block_size = config
        .model
        .block_size
        .unwrap_or(config.training.block_size)
        .max(1);
    let output = RunConfigOutput {
        run_name: run_name.to_string(),
        backend_name: backend_name.to_string(),
        arch_version: LANGUAGE_ARCH_VERSION.to_string(),
        shard_layout_version: SHARD_LAYOUT_VERSION_UNSHARDED,
        block_size,
        seed: config.training.seed,
        training_batch_size: config.training.batch_size,
        training_gradient_accumulation_steps: config.training.gradient_accumulation_steps,
        training_effective_batch_size: config
            .training
            .batch_size
            .saturating_mul(config.training.gradient_accumulation_steps),
        training_checkpoint_interval_iters: config.training.checkpoint_interval_iters,
        training_execution_form: build_training_execution_form(config),
        training_launch_mode_requested: config.training.launch_mode,
        training_sequence_kernel_override: effective_training_sequence_kernel_override,
        optimizer_spec: build_optimizer_spec(config),
        overrides: config.model.clone(),
        model_spec: build_model_spec(model_config),
        parallel_spec: build_parallel_spec(config),
        kernel_spec: build_kernel_spec(config, model_config, backend_name),
        state_layout: build_state_layout(model_config),
        metrics_sink: build_language_metrics_sink(config.training.log_frequency),
        startup_autotune: startup_autotune.cloned(),
    };
    let payload =
        serde_json::to_string_pretty(&output).context("failed to serialize web config")?;
    let path = run_dir.join("config.json");
    fs::write(&path, payload).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{effective_training_kernel_block_size, prepare_datasets, write_run_config};
    use crate::config::{
        ContextStrategyConfig, DatasetConfig, DatasetSourceConfig, GenerationConfig,
        ModelOverrides, TrainingConfig, TrainingHyperparameters, ValidationDatasetConfig,
    };
    use crate::dataset::TokenSequenceDataset;
    use crate::tokenizer::TokenizerConfig;
    use burn_dragon_core::{
        BDHConfig, BdhFiringTargetKind, BdhInitializationKind, BdhNeuronGainKind,
        BdhResidualScalingKind, BdhTopologyPriorKind, SequenceKernelConfig,
    };
    use burn_dragon_train::{
        OptimizerConfig, ParallelCheckpointFormat, ParallelConfig, ParallelismKind,
    };
    use serde_json::Value;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn effective_training_kernel_block_size_prefers_tbptt_subchunk() {
        let training = TrainingHyperparameters {
            block_size: 4096,
            tbptt_chunk_size: Some(256),
            tbptt_persist_across_steps: true,
            min_logical_block_size: Some(1024),
            batch_size: 16,
            seed: 1337,
            gradient_accumulation_steps: 1,
            target_effective_batch_size: Some(16),
            epochs: None,
            max_iters: 1024,
            checkpoint_interval_iters: 256,
            log_frequency: 32,
            launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode::Fresh,
            resume_run_dir: None,
            resume_checkpoint_epoch: None,
            init_checkpoint_path: None,
            init_checkpoint_epoch: None,
            context_strategy: ContextStrategyConfig::Infinite,
            sequence_kernel_override: None,
            gdpo: None,
        };

        assert_eq!(effective_training_kernel_block_size(&training), 256);
    }

    #[test]
    fn write_run_config_emits_phase0_model_parallel_metadata() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let config = TrainingConfig {
            dataset: DatasetConfig {
                cache_dir: dir.path().join("cache"),
                train_split_ratio: 0.9,
                validation: None,
                source: DatasetSourceConfig::Shakespeare { url: None },
                tokenizer: TokenizerConfig::default(),
            },
            training: TrainingHyperparameters {
                block_size: 64,
                tbptt_chunk_size: None,
                tbptt_persist_across_steps: false,
                min_logical_block_size: None,
                batch_size: 4,
                seed: 4242,
                gradient_accumulation_steps: 2,
                target_effective_batch_size: None,
                epochs: None,
                max_iters: 8,
                checkpoint_interval_iters: 2000,
                log_frequency: 1,
                launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode::Fresh,
                resume_run_dir: None,
                resume_checkpoint_epoch: None,
                init_checkpoint_path: None,
                init_checkpoint_epoch: None,
                context_strategy: ContextStrategyConfig::Infinite,
                sequence_kernel_override: None,
                gdpo: None,
            },
            optimizer: OptimizerConfig {
                name: burn_dragon_train::OptimizerKind::default(),
                learning_rate: 1.0e-3,
                weight_decay: 0.0,
                weight_decay_final: None,
                lr_schedule: None,
                schedule_mode: burn_dragon_train::OptimizerScheduleMode::default(),
                grad_clip_norm: None,
                grad_clip_value: None,
                muon: None,
            },
            parallel: ParallelConfig {
                mode: ParallelismKind::TensorParallelNeuron,
                world_size: 2,
                tensor: burn_dragon_train::ParallelTensorConfig {
                    size: 2,
                    ..Default::default()
                },
                checkpoint: burn_dragon_train::ParallelCheckpointConfig {
                    format: ParallelCheckpointFormat::ShardedV2,
                    async_write: false,
                },
                ..Default::default()
            },
            generation: GenerationConfig {
                prompt: "abc".to_string(),
                max_tokens: Some(4),
                max_chars: None,
                temperature: 1.0,
                top_k: None,
                context_strategy: ContextStrategyConfig::Infinite,
                prompt_tokenizer: Default::default(),
                decode_tokenizer: Default::default(),
                output_format: Default::default(),
            },
            wgpu: Default::default(),
            run_layout: burn_dragon_train::RunLayoutConfig::default(),
            model: ModelOverrides {
                n_layer: Some(8),
                n_embd: Some(256),
                n_head: Some(4),
                latent_total: Some(32768),
                ..ModelOverrides::default()
            },
        };
        let mut model_config = BDHConfig::default();
        model_config.n_layer = 8;
        model_config.n_embd = 256;
        model_config.n_head = 4;
        model_config.mlp_internal_dim_multiplier = 128;
        model_config.initialization.kind = BdhInitializationKind::HeadwiseSemiOrthogonal;
        model_config.initialization.residual_scaling.kind = BdhResidualScalingKind::DepthScaled;
        model_config.initialization.neuron_gains.kind = BdhNeuronGainKind::HeavyTailedLogNormal;
        model_config.initialization.topology_prior.kind = BdhTopologyPriorKind::ModularBridges;
        model_config.initialization.firing_targets.kind = BdhFiringTargetKind::GaussianEstimate;

        write_run_config(
            &config,
            &model_config,
            &run_dir,
            "test-run",
            "cuda",
            config.training.sequence_kernel_override,
            None,
        )
        .expect("write run config");

        let payload = std::fs::read_to_string(run_dir.join("config.json")).expect("read config");
        let json: Value = serde_json::from_str(&payload).expect("parse config json");
        assert_eq!(json["seed"], 4242);
        assert_eq!(json["arch_version"], "dragon_bdh_v1");
        assert_eq!(json["shard_layout_version"], 1);
        assert_eq!(json["model_spec"]["latent_total"], 32768);
        assert_eq!(
            json["model_spec"]["bdh_initialization_kind"],
            serde_json::Value::String("headwise_semi_orthogonal".to_string())
        );
        assert_eq!(
            json["model_spec"]["bdh_residual_scaling_kind"],
            serde_json::Value::String("depth_scaled".to_string())
        );
        assert_eq!(
            json["model_spec"]["bdh_neuron_gain_kind"],
            serde_json::Value::String("heavy_tailed_log_normal".to_string())
        );
        assert_eq!(
            json["model_spec"]["bdh_topology_prior_kind"],
            serde_json::Value::String("modular_bridges".to_string())
        );
        assert_eq!(
            json["model_spec"]["bdh_firing_target_kind"],
            serde_json::Value::String("gaussian_estimate".to_string())
        );
        assert_eq!(
            json["parallel_spec"]["mode"],
            serde_json::Value::String("tensor_parallel_neuron".to_string())
        );
        assert_eq!(
            json["parallel_spec"]["checkpoint_format"],
            serde_json::Value::String("sharded_v2".to_string())
        );
        assert_eq!(
            json["kernel_spec"]["sequence_kernel"],
            serde_json::Value::String("linear_attention".to_string())
        );
        assert_eq!(
            json["kernel_spec"]["low_bit_runtime"],
            serde_json::Value::Null
        );
        assert_eq!(json["optimizer_spec"]["name"], "adamw");
        assert_eq!(json["optimizer_spec"]["schedule_mode"], "bdh_reference");
        assert_eq!(json["training_execution_form"], "default_stateful");
        assert_eq!(json["training_launch_mode_requested"], "fresh");
        assert_eq!(json["state_layout"]["state_family"], "bdh_model_state");
        assert_eq!(
            json["state_layout"]["layers"][0]["tensors"][0]["name"],
            "rho"
        );
        assert_eq!(json["metrics_sink"]["family"], "language_bdh_burn_train_v1");
        assert_eq!(json["metrics_sink"]["entries"][2]["name"], "Learning Rate");
    }

    #[test]
    fn write_run_config_records_collective_parallel_metadata() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let config = TrainingConfig {
            dataset: DatasetConfig {
                cache_dir: dir.path().join("cache"),
                train_split_ratio: 0.9,
                validation: None,
                source: DatasetSourceConfig::Shakespeare { url: None },
                tokenizer: TokenizerConfig::default(),
            },
            training: TrainingHyperparameters {
                block_size: 64,
                tbptt_chunk_size: None,
                tbptt_persist_across_steps: false,
                min_logical_block_size: None,
                batch_size: 4,
                seed: 4242,
                gradient_accumulation_steps: 1,
                target_effective_batch_size: None,
                epochs: None,
                max_iters: 8,
                checkpoint_interval_iters: 2000,
                log_frequency: 1,
                launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode::Fresh,
                resume_run_dir: None,
                resume_checkpoint_epoch: None,
                init_checkpoint_path: None,
                init_checkpoint_epoch: None,
                context_strategy: ContextStrategyConfig::Infinite,
                sequence_kernel_override: None,
                gdpo: None,
            },
            optimizer: OptimizerConfig {
                name: burn_dragon_train::OptimizerKind::default(),
                learning_rate: 1.0e-3,
                weight_decay: 0.0,
                weight_decay_final: None,
                lr_schedule: None,
                schedule_mode: burn_dragon_train::OptimizerScheduleMode::default(),
                grad_clip_norm: None,
                grad_clip_value: None,
                muon: None,
            },
            parallel: ParallelConfig {
                mode: ParallelismKind::Ddp,
                world_size: 2,
                data: burn_dragon_train::ParallelDataConfig {
                    size: 2,
                    collective_num_nodes: Some(2),
                    collective_global_address: Some("127.0.0.1:32000".to_string()),
                    collective_node_address: Some("127.0.0.1:32001".to_string()),
                    collective_data_service_port: Some(32001),
                    ..Default::default()
                },
                ..Default::default()
            },
            generation: GenerationConfig {
                prompt: "abc".to_string(),
                max_tokens: Some(4),
                max_chars: None,
                temperature: 1.0,
                top_k: None,
                context_strategy: ContextStrategyConfig::Infinite,
                prompt_tokenizer: Default::default(),
                decode_tokenizer: Default::default(),
                output_format: Default::default(),
            },
            wgpu: Default::default(),
            run_layout: burn_dragon_train::RunLayoutConfig::default(),
            model: ModelOverrides {
                n_layer: Some(1),
                n_embd: Some(256),
                n_head: Some(4),
                latent_total: Some(32768),
                ..ModelOverrides::default()
            },
        };
        let mut model_config = BDHConfig::default();
        model_config.n_layer = 1;
        model_config.n_embd = 256;
        model_config.n_head = 4;
        model_config.mlp_internal_dim_multiplier = 128;

        write_run_config(
            &config,
            &model_config,
            &run_dir,
            "test-run",
            "cuda",
            config.training.sequence_kernel_override,
            None,
        )
        .expect("write run config");

        let payload = std::fs::read_to_string(run_dir.join("config.json")).expect("read config");
        let json: Value = serde_json::from_str(&payload).expect("parse config json");
        assert_eq!(json["parallel_spec"]["mode"], "ddp");
        assert_eq!(json["parallel_spec"]["collective_num_nodes"], 2);
        assert_eq!(
            json["parallel_spec"]["collective_global_address"],
            "127.0.0.1:32000"
        );
        assert_eq!(
            json["parallel_spec"]["collective_node_address"],
            "127.0.0.1:32001"
        );
        assert_eq!(json["parallel_spec"]["collective_data_service_port"], 32001);
    }

    #[test]
    fn write_run_config_records_pipeline_cache_metadata() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let config = TrainingConfig {
            dataset: DatasetConfig {
                cache_dir: dir.path().join("cache"),
                train_split_ratio: 0.9,
                validation: None,
                source: DatasetSourceConfig::Shakespeare { url: None },
                tokenizer: TokenizerConfig::default(),
            },
            training: TrainingHyperparameters {
                block_size: 64,
                tbptt_chunk_size: None,
                tbptt_persist_across_steps: false,
                min_logical_block_size: None,
                batch_size: 4,
                seed: 4242,
                gradient_accumulation_steps: 1,
                target_effective_batch_size: None,
                epochs: None,
                max_iters: 8,
                checkpoint_interval_iters: 2000,
                log_frequency: 1,
                launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode::Fresh,
                resume_run_dir: None,
                resume_checkpoint_epoch: None,
                init_checkpoint_path: None,
                init_checkpoint_epoch: None,
                context_strategy: ContextStrategyConfig::Infinite,
                sequence_kernel_override: None,
                gdpo: None,
            },
            optimizer: OptimizerConfig {
                name: burn_dragon_train::OptimizerKind::default(),
                learning_rate: 1.0e-3,
                weight_decay: 0.0,
                weight_decay_final: None,
                lr_schedule: None,
                schedule_mode: burn_dragon_train::OptimizerScheduleMode::default(),
                grad_clip_norm: None,
                grad_clip_value: None,
                muon: None,
            },
            parallel: ParallelConfig {
                mode: ParallelismKind::Ddp,
                world_size: 4,
                data: burn_dragon_train::ParallelDataConfig {
                    size: 4,
                    ..Default::default()
                },
                pipeline: burn_dragon_train::ParallelPipelineConfig {
                    enabled: true,
                    stage_count: 2,
                    virtual_stages_per_rank: 1,
                    schedule: burn_dragon_train::PipelineScheduleKind::Interleaved1f1b,
                    microbatches: 2,
                    communication: burn_dragon_train::PipelineCommunicationKind::BlockResidualCache,
                    cache: burn_dragon_train::ParallelPipelineCacheConfig {
                        enabled: true,
                        policy: burn_dragon_train::PipelineCachePolicy::ResidentBlockSummaries,
                        reuse_across_backward: true,
                        max_inflight_microbatches: 2,
                        transport_dtype: burn_dragon_train::PipelineTransportDtype::Bf16,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
            generation: GenerationConfig {
                prompt: "abc".to_string(),
                max_tokens: Some(4),
                max_chars: None,
                temperature: 1.0,
                top_k: None,
                context_strategy: ContextStrategyConfig::Infinite,
                prompt_tokenizer: Default::default(),
                decode_tokenizer: Default::default(),
                output_format: Default::default(),
            },
            wgpu: Default::default(),
            run_layout: burn_dragon_train::RunLayoutConfig::default(),
            model: ModelOverrides {
                n_layer: Some(2),
                n_embd: Some(256),
                n_head: Some(4),
                latent_total: Some(32768),
                ..ModelOverrides::default()
            },
        };
        let mut model_config = BDHConfig::default();
        model_config.n_layer = 2;
        model_config.n_embd = 256;
        model_config.n_head = 4;
        model_config.mlp_internal_dim_multiplier = 128;

        write_run_config(
            &config,
            &model_config,
            &run_dir,
            "test-run",
            "cuda",
            config.training.sequence_kernel_override,
            None,
        )
        .expect("write run config");

        let payload = std::fs::read_to_string(run_dir.join("config.json")).expect("read config");
        let json: Value = serde_json::from_str(&payload).expect("parse config json");
        assert_eq!(json["parallel_spec"]["pipeline_enabled"], true);
        assert_eq!(json["parallel_spec"]["pipeline_stage_count"], 2);
        assert_eq!(json["parallel_spec"]["pipeline_microbatches"], 2);
        assert_eq!(json["training_execution_form"], "pipeline");
        assert_eq!(
            json["parallel_spec"]["pipeline_communication"],
            "block_residual_cache"
        );
        assert_eq!(json["parallel_spec"]["pipeline_cache_enabled"], true);
        assert_eq!(
            json["parallel_spec"]["pipeline_cache_policy"],
            "resident_block_summaries"
        );
        assert_eq!(
            json["parallel_spec"]["pipeline_cache_transport_dtype"],
            "bf16"
        );
    }

    #[test]
    fn build_state_layout_records_rwkv8_rho_norm_tensor() {
        let mut model_config = BDHConfig::default();
        model_config.n_layer = 2;
        model_config.n_embd = 32;
        model_config.n_head = 2;
        model_config.mlp_internal_dim_multiplier = 4;
        model_config.sequence_kernel = SequenceKernelConfig::reference(
            burn_dragon_core::SequenceMemorySystem::Rwkv8StateSpace,
        );

        let layout = super::build_state_layout(&model_config);
        let tensor_names = layout.layers[0]
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>();

        assert!(tensor_names.contains(&"rho"));
        assert!(tensor_names.contains(&"rho_norm"));
    }

    #[test]
    fn build_state_layout_records_mamba_sequence_aux_tensor() {
        let mut model_config = BDHConfig::default();
        model_config.n_layer = 2;
        model_config.n_embd = 32;
        model_config.n_head = 2;
        model_config.sequence_kernel = SequenceKernelConfig::reference(
            burn_dragon_core::SequenceMemorySystem::Mamba1SelectiveScan,
        );
        model_config.mamba.expand = 3;
        model_config.mamba.d_state = 8;
        model_config.mamba.d_conv = 5;

        let layout = super::build_state_layout(&model_config);
        let layer0 = &layout.layers[0];
        let tensor_names = layer0
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>();

        assert!(tensor_names.contains(&"rho"));
        assert!(tensor_names.contains(&"sequence_aux"));
        let rho = layer0
            .tensors
            .iter()
            .find(|tensor| tensor.name == "rho")
            .expect("mamba rho tensor");
        assert_eq!(rho.axes[2].size, Some(96));
        assert_eq!(rho.axes[3].size, Some(8));
    }

    #[test]
    fn build_state_layout_records_mamba3_recurrent_state_tensors() {
        let mut model_config = BDHConfig::default();
        model_config.n_layer = 2;
        model_config.n_embd = 128;
        model_config.n_head = 2;
        model_config.sequence_kernel = SequenceKernelConfig::reference(
            burn_dragon_core::SequenceMemorySystem::Mamba3StateSpaceDuality,
        );
        model_config.mamba.headdim = 64;
        model_config.mamba.ngroups = 2;
        model_config.mamba.d_state = 16;
        model_config.mamba.rope_fraction = 0.5;

        let layout = super::build_state_layout(&model_config);
        let layer0 = &layout.layers[0];
        let tensor_names = layer0
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>();

        assert!(tensor_names.contains(&"rho"));
        assert!(tensor_names.contains(&"mamba_angle_state"));
        assert!(tensor_names.contains(&"mamba_k_state"));
        assert!(tensor_names.contains(&"mamba_v_state"));
    }

    #[test]
    fn write_run_config_records_training_sequence_kernel_override() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let config = TrainingConfig {
            dataset: DatasetConfig {
                cache_dir: dir.path().join("cache"),
                train_split_ratio: 0.9,
                validation: None,
                source: DatasetSourceConfig::Shakespeare { url: None },
                tokenizer: TokenizerConfig::default(),
            },
            training: TrainingHyperparameters {
                block_size: 64,
                tbptt_chunk_size: None,
                tbptt_persist_across_steps: false,
                min_logical_block_size: None,
                batch_size: 1,
                seed: 1337,
                gradient_accumulation_steps: 1,
                target_effective_batch_size: None,
                epochs: None,
                max_iters: 8,
                checkpoint_interval_iters: 2000,
                log_frequency: 1,
                launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode::Fresh,
                resume_run_dir: None,
                resume_checkpoint_epoch: None,
                init_checkpoint_path: None,
                init_checkpoint_epoch: None,
                context_strategy: ContextStrategyConfig::Infinite,
                sequence_kernel_override: Some(SequenceKernelConfig::dense_score_short_context()),
                gdpo: None,
            },
            optimizer: OptimizerConfig {
                name: burn_dragon_train::OptimizerKind::default(),
                learning_rate: 1.0e-3,
                weight_decay: 0.0,
                weight_decay_final: None,
                lr_schedule: None,
                schedule_mode: burn_dragon_train::OptimizerScheduleMode::default(),
                grad_clip_norm: None,
                grad_clip_value: None,
                muon: None,
            },
            parallel: ParallelConfig::default(),
            generation: GenerationConfig {
                prompt: "abc".to_string(),
                max_tokens: Some(4),
                max_chars: None,
                temperature: 1.0,
                top_k: None,
                context_strategy: ContextStrategyConfig::Infinite,
                prompt_tokenizer: Default::default(),
                decode_tokenizer: Default::default(),
                output_format: Default::default(),
            },
            wgpu: Default::default(),
            run_layout: burn_dragon_train::RunLayoutConfig::default(),
            model: ModelOverrides {
                sequence_kernel: Some(SequenceKernelConfig::reference(
                    burn_dragon_core::SequenceMemorySystem::LinearAttention,
                )),
                ..ModelOverrides::default()
            },
        };
        let mut model_config = BDHConfig::default();
        model_config.sequence_kernel = SequenceKernelConfig::dense_score_short_context();
        model_config.quant = burn_dragon_core::LowBitQuantizationConfig {
            enable: true,
            protocol: burn_dragon_core::BitNetLowBitProtocol::BitnetB158,
            training_mode: burn_dragon_core::LowBitTrainingMode::TrainKernelExp,
            inference_mode: burn_dragon_core::LowBitInferenceMode::OfflinePack,
            saved_activations: burn_dragon_core::LowBitSavedActivationConfig {
                mode: burn_dragon_core::LowBitSavedActivationMode::QuantizedCacheRecomputeExp,
                format: burn_dragon_core::LowBitActivationFormat::Int8,
            },
            ..Default::default()
        };

        write_run_config(
            &config,
            &model_config,
            &run_dir,
            "test-run",
            "cuda",
            config.training.sequence_kernel_override,
            None,
        )
        .expect("write run config");

        let payload = std::fs::read_to_string(run_dir.join("config.json")).expect("read config");
        let json: Value = serde_json::from_str(&payload).expect("parse config json");
        assert_eq!(
            json["training_sequence_kernel_override"],
            serde_json::json!({
                "memory_system": "linear_attention",
                "executor": "dense_score_short_context"
            })
        );
        assert_eq!(json["training_execution_form"], "default_stateful");
        assert_eq!(json["training_launch_mode_requested"], "fresh");
        assert_eq!(json["training_checkpoint_interval_iters"], 2000);
        assert_eq!(
            json["kernel_spec"]["sequence_kernel"],
            serde_json::json!({
                "memory_system": "linear_attention",
                "executor": "dense_score_short_context"
            })
        );
        assert_eq!(
            json["kernel_spec"]["low_bit_runtime"],
            serde_json::Value::String("packed_native_training_forward".to_string())
        );
        assert_eq!(
            json["kernel_spec"]["low_bit_saved_activation_mode"],
            serde_json::Value::String("quantized_cache_recompute_exp".to_string())
        );
        assert_eq!(
            json["kernel_spec"]["low_bit_saved_activation_format"],
            serde_json::Value::String("int8".to_string())
        );
        assert_eq!(
            json["kernel_spec"]["low_bit_saved_activation_inventory"]["requires_rho_window_anchor"],
            serde_json::Value::Bool(true)
        );
        assert_eq!(
            json["kernel_spec"]["low_bit_saved_activation_inventory"]["tensors"][0]["name"],
            serde_json::Value::String("x_projection_input".to_string())
        );
        assert_eq!(
            json["kernel_spec"]["low_bit_native_supported"],
            serde_json::Value::Bool(true)
        );
        assert!(
            json["kernel_spec"]["low_bit_memory"]["estimated_total_bytes"]
                .as_u64()
                .expect("low bit estimated bytes")
                > 0
        );
        assert!(
            json["kernel_spec"]["low_bit_memory"]["saved_activation_bytes"]
                .as_u64()
                .expect("saved activation bytes")
                > 0
        );
        assert_eq!(
            json["overrides"]["sequence_kernel"],
            serde_json::Value::String("linear_attention".to_string())
        );
    }

    #[test]
    fn write_run_config_can_record_effective_training_sequence_kernel_override() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let config = TrainingConfig {
            dataset: DatasetConfig {
                cache_dir: dir.path().join("cache"),
                train_split_ratio: 0.9,
                validation: None,
                source: DatasetSourceConfig::Shakespeare { url: None },
                tokenizer: TokenizerConfig::default(),
            },
            training: TrainingHyperparameters {
                block_size: 64,
                tbptt_chunk_size: None,
                tbptt_persist_across_steps: false,
                min_logical_block_size: None,
                batch_size: 1,
                seed: 1337,
                gradient_accumulation_steps: 1,
                target_effective_batch_size: None,
                epochs: None,
                max_iters: 8,
                checkpoint_interval_iters: 2000,
                log_frequency: 1,
                launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode::Fresh,
                resume_run_dir: None,
                resume_checkpoint_epoch: None,
                init_checkpoint_path: None,
                init_checkpoint_epoch: None,
                context_strategy: ContextStrategyConfig::Infinite,
                sequence_kernel_override: None,
                gdpo: None,
            },
            optimizer: OptimizerConfig {
                name: burn_dragon_train::OptimizerKind::default(),
                learning_rate: 1.0e-3,
                weight_decay: 0.0,
                weight_decay_final: None,
                lr_schedule: None,
                schedule_mode: burn_dragon_train::OptimizerScheduleMode::default(),
                grad_clip_norm: None,
                grad_clip_value: None,
                muon: None,
            },
            parallel: ParallelConfig::default(),
            generation: GenerationConfig {
                prompt: "abc".to_string(),
                max_tokens: Some(4),
                max_chars: None,
                temperature: 1.0,
                top_k: None,
                context_strategy: ContextStrategyConfig::Infinite,
                prompt_tokenizer: Default::default(),
                decode_tokenizer: Default::default(),
                output_format: Default::default(),
            },
            wgpu: Default::default(),
            run_layout: burn_dragon_train::RunLayoutConfig::default(),
            model: ModelOverrides {
                sequence_kernel: Some(SequenceKernelConfig::reference(
                    burn_dragon_core::SequenceMemorySystem::LinearAttention,
                )),
                ..ModelOverrides::default()
            },
        };
        let mut model_config = BDHConfig::default();
        model_config.sequence_kernel = SequenceKernelConfig::dense_score_short_context();

        write_run_config(
            &config,
            &model_config,
            &run_dir,
            "test-run",
            "cuda",
            Some(SequenceKernelConfig::dense_score_short_context()),
            None,
        )
        .expect("write run config");

        let payload = std::fs::read_to_string(run_dir.join("config.json")).expect("read config");
        let json: Value = serde_json::from_str(&payload).expect("parse config json");
        assert_eq!(
            json["training_sequence_kernel_override"],
            serde_json::json!({
                "memory_system": "linear_attention",
                "executor": "dense_score_short_context"
            })
        );
    }

    fn tiny_training_hparams() -> TrainingHyperparameters {
        TrainingHyperparameters {
            block_size: 8,
            tbptt_chunk_size: None,
            tbptt_persist_across_steps: false,
            min_logical_block_size: None,
            batch_size: 2,
            seed: 1337,
            gradient_accumulation_steps: 1,
            target_effective_batch_size: None,
            epochs: None,
            max_iters: 4,
            checkpoint_interval_iters: 2000,
            log_frequency: 1,
            launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode::Fresh,
            resume_run_dir: None,
            resume_checkpoint_epoch: None,
            init_checkpoint_path: None,
            init_checkpoint_epoch: None,
            context_strategy: ContextStrategyConfig::Infinite,
            sequence_kernel_override: None,
            gdpo: None,
        }
    }

    #[test]
    fn prepare_datasets_uses_validation_override_with_compatible_tokenizer() {
        let dir = tempdir().expect("tempdir");
        let train_cache = dir.path().join("train");
        let valid_cache = dir.path().join("valid");
        fs::create_dir_all(&train_cache).expect("create train cache");
        fs::create_dir_all(&valid_cache).expect("create valid cache");
        fs::write(
            train_cache.join("tinyshakespeare.txt"),
            "the training corpus stays on climbmix-like bytes\n",
        )
        .expect("write train corpus");
        fs::write(
            valid_cache.join("tinyshakespeare.txt"),
            "validation corpus uses a different byte stream\n",
        )
        .expect("write valid corpus");

        let config = DatasetConfig {
            cache_dir: train_cache,
            train_split_ratio: 0.9,
            validation: Some(ValidationDatasetConfig {
                cache_dir: Some(valid_cache),
                train_split_ratio: Some(0.75),
                source: DatasetSourceConfig::Shakespeare { url: None },
            }),
            source: DatasetSourceConfig::Shakespeare { url: None },
            tokenizer: TokenizerConfig {
                vocab_path: None,
                kind: crate::tokenizer::TokenizerKind::Byte(
                    crate::tokenizer::ByteTokenizerConfig::default(),
                ),
            },
        };

        let datasets = prepare_datasets(&config, &tiny_training_hparams())
            .expect("prepare datasets with validation override");

        assert_eq!(
            datasets.train.tokenizer().len(),
            datasets.valid.tokenizer().len()
        );
        assert_eq!(datasets.valid.train_split_ratio(), 0.75);
        let mut train_prefix = vec![0u32; 8];
        let mut valid_prefix = vec![0u32; 8];
        TokenSequenceDataset::copy_token_range(datasets.train.as_ref(), 0, &mut train_prefix);
        TokenSequenceDataset::copy_token_range(datasets.valid.as_ref(), 0, &mut valid_prefix);
        assert_ne!(train_prefix, valid_prefix);
    }

    #[test]
    fn prepare_datasets_rejects_incompatible_validation_tokenizer() {
        let dir = tempdir().expect("tempdir");
        let train_cache = dir.path().join("train");
        let valid_cache = dir.path().join("valid");
        fs::create_dir_all(&train_cache).expect("create train cache");
        fs::create_dir_all(&valid_cache).expect("create valid cache");
        fs::write(
            train_cache.join("tinyshakespeare.txt"),
            "abcdefabcdefabcdefabcdef",
        )
        .expect("write train corpus");
        fs::write(
            valid_cache.join("tinyshakespeare.txt"),
            "zzzzzzzzzzzzzzzzzzzzzzzz",
        )
        .expect("write valid corpus");

        let config = DatasetConfig {
            cache_dir: train_cache,
            train_split_ratio: 0.9,
            validation: Some(ValidationDatasetConfig {
                cache_dir: Some(valid_cache),
                train_split_ratio: Some(0.75),
                source: DatasetSourceConfig::Shakespeare { url: None },
            }),
            source: DatasetSourceConfig::Shakespeare { url: None },
            tokenizer: TokenizerConfig::default(),
        };

        let err = prepare_datasets(&config, &tiny_training_hparams())
            .err()
            .expect("incompatible validation tokenizer should fail");
        assert!(
            err.to_string()
                .contains("validation dataset tokenizer is incompatible"),
            "unexpected error: {err:#}"
        );
    }
}
