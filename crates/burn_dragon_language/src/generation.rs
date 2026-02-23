use anyhow::{Result, anyhow};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::cmp::Ordering;
use std::mem::size_of;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use burn_dragon_core::{BDH, ModelState};

use crate::GenerationConfig;
use crate::config::ContextStrategyConfig;
use crate::tokenizer::Tokenizer;

#[derive(Clone, Copy, Debug)]
pub enum ContextStrategy {
    Infinite,
    Sliding { window: usize },
}

#[derive(Clone, Copy, Debug)]
pub struct GenerationSettings {
    pub max_new_tokens: Option<usize>,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub strategy: ContextStrategy,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GenerationProfileSnapshot {
    pub prefill_forward_ns: u128,
    pub token_forward_ns: u128,
    pub sample_host_transfer_ns: u128,
    pub sample_cpu_ns: u128,
    pub token_tensor_copy_ns: u128,
    pub token_steps: u64,
    pub prefill_tokens: u64,
    pub host_sync_points: u64,
    pub host_to_device_copy_bytes: u128,
    pub device_to_host_copy_bytes: u128,
}

#[derive(Clone, Copy, Debug, Default)]
struct GenerationProfileState {
    prefill_forward_ns: u128,
    token_forward_ns: u128,
    sample_host_transfer_ns: u128,
    sample_cpu_ns: u128,
    token_tensor_copy_ns: u128,
    token_steps: u64,
    prefill_tokens: u64,
    host_sync_points: u64,
    host_to_device_copy_bytes: u128,
    device_to_host_copy_bytes: u128,
}

static GENERATION_PROFILE: OnceLock<Mutex<GenerationProfileState>> = OnceLock::new();

fn generation_profile_enabled() -> bool {
    std::env::var_os("BDH_STAGE_PROFILE").is_some()
}

fn generation_profile_state() -> &'static Mutex<GenerationProfileState> {
    GENERATION_PROFILE.get_or_init(|| Mutex::new(GenerationProfileState::default()))
}

fn generation_profile_record(mutator: impl FnOnce(&mut GenerationProfileState)) {
    if let Ok(mut state) = generation_profile_state().lock() {
        mutator(&mut state);
    }
}

pub fn generation_profile_reset() {
    if let Ok(mut state) = generation_profile_state().lock() {
        *state = GenerationProfileState::default();
    }
}

pub fn generation_profile_snapshot() -> GenerationProfileSnapshot {
    if let Ok(state) = generation_profile_state().lock() {
        return GenerationProfileSnapshot {
            prefill_forward_ns: state.prefill_forward_ns,
            token_forward_ns: state.token_forward_ns,
            sample_host_transfer_ns: state.sample_host_transfer_ns,
            sample_cpu_ns: state.sample_cpu_ns,
            token_tensor_copy_ns: state.token_tensor_copy_ns,
            token_steps: state.token_steps,
            prefill_tokens: state.prefill_tokens,
            host_sync_points: state.host_sync_points,
            host_to_device_copy_bytes: state.host_to_device_copy_bytes,
            device_to_host_copy_bytes: state.device_to_host_copy_bytes,
        };
    }
    GenerationProfileSnapshot::default()
}

fn sample_from_logits_values(mut logits_values: Vec<f32>, top_k: Option<usize>) -> Result<i64> {
    let vocab = logits_values.len();
    if vocab == 0 {
        return Err(anyhow!("logits are empty"));
    }

    if let Some(k) = top_k
        && k > 0
        && k < vocab
    {
        let mut sorted = logits_values.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let threshold = sorted[k - 1];
        for value in logits_values.iter_mut() {
            if *value < threshold {
                *value = f32::NEG_INFINITY;
            }
        }
    }

    let max_logit = logits_values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits_values
        .iter()
        .map(|value| (value - max_logit).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    if sum == 0.0 || sum.is_nan() {
        let uniform = 1.0 / vocab as f32;
        for p in probs.iter_mut() {
            *p = uniform;
        }
    } else {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }

    let dist = WeightedIndex::new(&probs).map_err(|err| anyhow!(err.to_string()))?;
    let mut rng = thread_rng();
    Ok(dist.sample(&mut rng) as i64)
}

fn sample_argmax_token<B: Backend>(logits_temp: Tensor<B, 1>) -> Result<i64> {
    let values = logits_temp
        .argmax(0)
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(|err| anyhow!("{err:?}"))?;
    values
        .first()
        .copied()
        .ok_or_else(|| anyhow!("argmax output is empty"))
}

pub fn prefill_state<B: Backend>(
    model: &BDH<B>,
    prompt_tokens: &[i64],
    device: &B::Device,
) -> Result<(ModelState<B>, Tensor<B, 1>)> {
    let prompt_len = prompt_tokens.len();
    if prompt_len == 0 {
        return Err(anyhow!("prompt must contain at least one token"));
    }

    let prof_enabled = generation_profile_enabled();
    if prof_enabled {
        let prompt_bytes = (prompt_len.saturating_mul(size_of::<i64>())) as u128;
        generation_profile_record(|profile| {
            profile.prefill_tokens = profile.prefill_tokens.saturating_add(prompt_len as u64);
            profile.host_to_device_copy_bytes = profile
                .host_to_device_copy_bytes
                .saturating_add(prompt_bytes);
        });
    }

    let prompt_tensor = Tensor::<B, 2, Int>::from_data(
        TensorData::new(prompt_tokens.to_vec(), [1, prompt_len]),
        device,
    );

    let mut state = model.init_state();
    let prefill_start = prof_enabled.then(Instant::now);
    let logits = model.forward_with_state(prompt_tensor, &mut state);
    if let Some(start) = prefill_start {
        let elapsed = start.elapsed().as_nanos();
        generation_profile_record(|profile| {
            profile.prefill_forward_ns = profile.prefill_forward_ns.saturating_add(elapsed);
        });
    }
    let [_, time, vocab] = logits.shape().dims::<3>();
    if time != prompt_len {
        return Err(anyhow!(
            "prefill produced mismatched length: expected {prompt_len}, got {time}"
        ));
    }

    let last_logits = logits.slice_dim(1, (time - 1)..time).reshape([vocab]);

    #[cfg(feature = "viz")]
    state.clear_viz();

    Ok((state, last_logits))
}

pub fn sample_next_token<B: Backend>(
    model: &BDH<B>,
    state: &mut ModelState<B>,
    last_logits: Tensor<B, 1>,
    temperature: f32,
    top_k: Option<usize>,
    device: &B::Device,
) -> Result<(i64, Tensor<B, 1>)> {
    let prof_enabled = generation_profile_enabled();
    let logits_temp = last_logits.clone().div_scalar(temperature);
    let next = if top_k == Some(1) {
        let host_start = prof_enabled.then(Instant::now);
        let token = sample_argmax_token(logits_temp)?;
        if let Some(start) = host_start {
            let elapsed = start.elapsed().as_nanos();
            generation_profile_record(|profile| {
                profile.sample_host_transfer_ns =
                    profile.sample_host_transfer_ns.saturating_add(elapsed);
                profile.host_sync_points = profile.host_sync_points.saturating_add(1);
                profile.device_to_host_copy_bytes = profile
                    .device_to_host_copy_bytes
                    .saturating_add(size_of::<i64>() as u128);
            });
        }
        token
    } else {
        let host_start = prof_enabled.then(Instant::now);
        let logits_values = logits_temp
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(|err| anyhow!("{err:?}"))?;
        if let Some(start) = host_start {
            let elapsed = start.elapsed().as_nanos();
            generation_profile_record(|profile| {
                profile.sample_host_transfer_ns =
                    profile.sample_host_transfer_ns.saturating_add(elapsed);
                profile.host_sync_points = profile.host_sync_points.saturating_add(1);
                profile.device_to_host_copy_bytes = profile
                    .device_to_host_copy_bytes
                    .saturating_add((logits_values.len().saturating_mul(size_of::<f32>())) as u128);
            });
        }
        let sample_start = prof_enabled.then(Instant::now);
        let token = sample_from_logits_values(logits_values, top_k)?;
        if let Some(start) = sample_start {
            let elapsed = start.elapsed().as_nanos();
            generation_profile_record(|profile| {
                profile.sample_cpu_ns = profile.sample_cpu_ns.saturating_add(elapsed);
            });
        }
        token
    };

    let tensor_copy_start = prof_enabled.then(Instant::now);
    let next_tensor = Tensor::<B, 2, Int>::from_data(TensorData::new(vec![next], [1, 1]), device);
    if let Some(start) = tensor_copy_start {
        let elapsed = start.elapsed().as_nanos();
        generation_profile_record(|profile| {
            profile.token_tensor_copy_ns = profile.token_tensor_copy_ns.saturating_add(elapsed);
            profile.host_to_device_copy_bytes = profile
                .host_to_device_copy_bytes
                .saturating_add(size_of::<i64>() as u128);
        });
    }

    let forward_start = prof_enabled.then(Instant::now);
    let logits = model.forward_with_state(next_tensor, state);
    if let Some(start) = forward_start {
        let elapsed = start.elapsed().as_nanos();
        generation_profile_record(|profile| {
            profile.token_forward_ns = profile.token_forward_ns.saturating_add(elapsed);
            profile.token_steps = profile.token_steps.saturating_add(1);
        });
    }
    let [_, time, vocab] = logits.shape().dims::<3>();
    let new_last_logits = logits.slice_dim(1, (time - 1)..time).reshape([vocab]);

    Ok((next, new_last_logits))
}

#[cfg(feature = "web")]
pub async fn sample_next_token_async<B: Backend>(
    model: &BDH<B>,
    state: &mut ModelState<B>,
    last_logits: Tensor<B, 1>,
    temperature: f32,
    top_k: Option<usize>,
    device: &B::Device,
) -> Result<(i64, Tensor<B, 1>)> {
    let prof_enabled = generation_profile_enabled();
    let logits_temp = last_logits.clone().div_scalar(temperature);
    let next = if top_k == Some(1) {
        let host_start = prof_enabled.then(Instant::now);
        let values = logits_temp
            .argmax(0)
            .to_data_async()
            .await
            .convert::<i64>()
            .into_vec::<i64>()
            .map_err(|err| anyhow!("{err:?}"))?;
        if let Some(start) = host_start {
            let elapsed = start.elapsed().as_nanos();
            generation_profile_record(|profile| {
                profile.sample_host_transfer_ns =
                    profile.sample_host_transfer_ns.saturating_add(elapsed);
                profile.host_sync_points = profile.host_sync_points.saturating_add(1);
                profile.device_to_host_copy_bytes = profile
                    .device_to_host_copy_bytes
                    .saturating_add(size_of::<i64>() as u128);
            });
        }
        values
            .first()
            .copied()
            .ok_or_else(|| anyhow!("argmax output is empty"))?
    } else {
        let host_start = prof_enabled.then(Instant::now);
        let logits_values = logits_temp
            .to_data_async()
            .await
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(|err| anyhow!("{err:?}"))?;
        if let Some(start) = host_start {
            let elapsed = start.elapsed().as_nanos();
            generation_profile_record(|profile| {
                profile.sample_host_transfer_ns =
                    profile.sample_host_transfer_ns.saturating_add(elapsed);
                profile.host_sync_points = profile.host_sync_points.saturating_add(1);
                profile.device_to_host_copy_bytes = profile
                    .device_to_host_copy_bytes
                    .saturating_add((logits_values.len().saturating_mul(size_of::<f32>())) as u128);
            });
        }
        let sample_start = prof_enabled.then(Instant::now);
        let token = sample_from_logits_values(logits_values, top_k)?;
        if let Some(start) = sample_start {
            let elapsed = start.elapsed().as_nanos();
            generation_profile_record(|profile| {
                profile.sample_cpu_ns = profile.sample_cpu_ns.saturating_add(elapsed);
            });
        }
        token
    };

    let tensor_copy_start = prof_enabled.then(Instant::now);
    let next_tensor = Tensor::<B, 2, Int>::from_data(TensorData::new(vec![next], [1, 1]), device);
    if let Some(start) = tensor_copy_start {
        let elapsed = start.elapsed().as_nanos();
        generation_profile_record(|profile| {
            profile.token_tensor_copy_ns = profile.token_tensor_copy_ns.saturating_add(elapsed);
            profile.host_to_device_copy_bytes = profile
                .host_to_device_copy_bytes
                .saturating_add(size_of::<i64>() as u128);
        });
    }

    let forward_start = prof_enabled.then(Instant::now);
    let logits = model.forward_with_state(next_tensor, state);
    if let Some(start) = forward_start {
        let elapsed = start.elapsed().as_nanos();
        generation_profile_record(|profile| {
            profile.token_forward_ns = profile.token_forward_ns.saturating_add(elapsed);
            profile.token_steps = profile.token_steps.saturating_add(1);
        });
    }
    let [_, time, vocab] = logits.shape().dims::<3>();
    let new_last_logits = logits.slice_dim(1, (time - 1)..time).reshape([vocab]);

    Ok((next, new_last_logits))
}

pub fn generate_tokens<B: Backend>(
    model: &BDH<B>,
    prompt_tokens: Vec<i64>,
    device: &B::Device,
    settings: GenerationSettings,
    mut on_token: Option<&mut dyn FnMut(i64)>,
) -> Result<Vec<i64>> {
    let GenerationSettings {
        max_new_tokens,
        temperature,
        top_k,
        strategy,
    } = settings;

    let mut full_tokens = prompt_tokens;
    let (mut state, mut last_logits) = prefill_state(model, &full_tokens, device)?;
    let mut generated = 0usize;

    if let ContextStrategy::Sliding { window } = strategy
        && window > 0
        && state.position > window
    {
        state.trim(window);
    }

    while max_new_tokens.is_none_or(|max| generated < max) {
        let (next, logits) =
            sample_next_token(model, &mut state, last_logits, temperature, top_k, device)?;
        full_tokens.push(next);
        last_logits = logits;
        generated = generated.saturating_add(1);

        if let Some(callback) = &mut on_token {
            callback(next);
        }

        if let ContextStrategy::Sliding { window } = strategy
            && window > 0
            && state.position > window
        {
            state.trim(window);
        }
    }

    Ok(full_tokens)
}

pub fn generate_text<B: Backend>(
    model: &BDH<B>,
    tokenizer: &dyn Tokenizer,
    device: &B::Device,
    block_size: usize,
    generation: &GenerationConfig,
) -> Result<String> {
    let strategy = resolve_context_strategy(&generation.context_strategy, block_size);
    let mut prompt_ids = tokenizer.encode(&generation.prompt, false, false);
    if let ContextStrategy::Sliding { window } = strategy
        && prompt_ids.len() > window
    {
        prompt_ids = prompt_ids[prompt_ids.len() - window..].to_vec();
    }

    let prompt_tokens: Vec<i64> = prompt_ids.iter().map(|&id| id as i64).collect();
    let max_new_tokens = normalize_max_tokens(generation.max_tokens);
    let settings = GenerationSettings {
        max_new_tokens,
        temperature: generation.temperature,
        top_k: generation.top_k,
        strategy,
    };
    let tokens_all = generate_tokens(model, prompt_tokens, device, settings, None)?;

    let decoded_ids: Vec<u32> = tokens_all
        .iter()
        .filter_map(|&tok| (tok >= 0).then_some(tok as u32))
        .collect();

    Ok(tokenizer.decode(&decoded_ids))
}

fn normalize_max_tokens(max_tokens: Option<i64>) -> Option<usize> {
    match max_tokens {
        Some(value) if value >= 0 => Some(value as usize),
        _ => None,
    }
}

pub fn resolve_context_strategy(
    config: &ContextStrategyConfig,
    default_window: usize,
) -> ContextStrategy {
    match config {
        ContextStrategyConfig::Infinite => ContextStrategy::Infinite,
        ContextStrategyConfig::Sliding { window } => {
            let win = if *window == 0 {
                default_window.max(1)
            } else {
                *window
            };
            ContextStrategy::Sliding { window: win }
        }
    }
}
