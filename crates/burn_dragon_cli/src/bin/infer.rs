#![recursion_limit = "256"]

use std::convert::TryFrom;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use clap::{Parser, ValueEnum};

use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::Backend;
use burn_dragon::checkpoint::{
    BurnpackLoadPolicy, BurnpackPrecisionPreference, burnpack_parts_manifest_path,
    candidate_burnpack_paths, try_load_model_from_burnpack_candidates,
};
use burn_dragon::core::BDH;
use burn_dragon::core::{
    logits_projection_profile_reset, logits_projection_profile_snapshot,
    low_bit_native_decoder_tail_profile_snapshot, low_bit_native_lowrank_profile_snapshot,
    low_bit_native_projection_profile_reset, lowrank_residual_profile_reset,
    lowrank_residual_profile_snapshot,
};
#[cfg(feature = "viz")]
use burn_dragon::language::build_model_config;
use burn_dragon::language::tokenizer::{
    SharedTokenizer, Tokenizer, TokenizerConfig, pretokenized::PretokenizedTokenizer,
};
use burn_dragon::language::{
    ContextStrategy, ContextStrategyConfig, GenerationConfig, GenerationOutputFormat,
    GenerationTokenizerSourceConfig, TrainingConfig, WgpuFusedCoreOverride,
    apply_bitnet_artifact_bundle_to_model, apply_wgpu_fused_core_override,
    build_model_config_with_tokenizer, candidate_bitnet_artifact_paths, default_checkpoint_dir,
    generate_tokens, generate_tokens_chunked, generation_profile_reset,
    generation_profile_snapshot, load_bitnet_artifact_bundle, load_training_config_for_checkpoint,
    prefill_state, resolve_context_strategy, sample_next_token,
};
use burn_dragon::train::WgpuGenerationExecutor;
use burn_dragon::train::wgpu::init_runtime;
use burn_dragon_kernel::api::projection::{
    relu_lowrank_forward_profile_reset, relu_lowrank_forward_profile_snapshot,
};
use burn_dragon_kernel::api::recurrent::{recurrent_profile_reset, recurrent_profile_snapshot};
use burn_wgpu::Wgpu;

#[cfg(feature = "cuda")]
use burn_cuda::Cuda;

#[cfg(feature = "viz")]
use burn_dragon::viz::{self, VizConfig, VizDimensions, VizEncoder};
#[cfg(feature = "viz")]
use std::sync::mpsc;
#[cfg(feature = "viz")]
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResolvedGenerationOutputFormat {
    DecodedText,
    TokenIds,
}

#[cfg(feature = "viz")]
struct VizRuntime<B: Backend> {
    encoder: VizEncoder<B>,
    sender: viz::VizSender<B>,
    stop: Arc<AtomicBool>,
}

fn default_or_explicit_config_paths(default_base: &str, explicit: &[PathBuf]) -> Vec<PathBuf> {
    if explicit.is_empty() {
        vec![PathBuf::from(default_base)]
    } else {
        explicit.to_vec()
    }
}

pub fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    let config_paths = default_or_explicit_config_paths("config/language/base.toml", &args.config);
    let config = load_training_config_for_checkpoint(
        &config_paths,
        args.checkpoint.as_ref(),
        backend_name(args.backend),
    )?;

    #[cfg(feature = "viz")]
    let use_viz = args.viz;
    #[cfg(all(not(feature = "viz"), feature = "cuda"))]
    let use_viz = false;
    #[cfg(all(not(feature = "viz"), not(feature = "cuda")))]
    let _use_viz = false;

    match args.backend {
        BackendArg::Wgpu => {
            let use_fused_core = config.wgpu.inference.fused_core_recurrent == Some(true);
            #[cfg(feature = "viz")]
            if use_viz {
                if use_fused_core {
                    return Err(anyhow!(
                        "wgpu.inference.fused_core_recurrent=true currently requires --viz off"
                    ));
                }
                let wgpu_config = config.wgpu.clone();
                return infer_backend_with_viz::<Wgpu<f32>, _>(
                    &config,
                    &args,
                    "wgpu",
                    move |device| init_runtime(device, &wgpu_config),
                );
            }
            let wgpu_config = config.wgpu.clone();
            let backend_name = if use_fused_core {
                "wgpu-fused-core"
            } else {
                "wgpu"
            };
            infer_backend::<Wgpu<f32>, _>(&config, &args, backend_name, move |device| {
                init_runtime(device, &wgpu_config)
            })
        }
        BackendArg::Cuda => {
            #[cfg(feature = "cuda")]
            {
                if use_viz {
                    return Err(anyhow!(
                        "viz overlay requires the wgpu backend; run with --backend wgpu"
                    ));
                }
                infer_backend::<Cuda<f32>, _>(&config, &args, "cuda", |_| {})
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(anyhow!(
                    "cuda backend selected but this build lacks `cuda` feature; rebuild with `--features cuda`"
                ))
            }
        }
    }
}

#[cfg(test)]
mod path_tests {
    use super::default_or_explicit_config_paths;
    use std::path::PathBuf;

    #[test]
    fn default_or_explicit_config_paths_uses_default_only_when_no_explicit_configs() {
        assert_eq!(
            default_or_explicit_config_paths("config/language/base.toml", &[]),
            vec![PathBuf::from("config/language/base.toml")]
        );
        assert_eq!(
            default_or_explicit_config_paths(
                "config/language/base.toml",
                &[PathBuf::from("config/language/custom.toml")]
            ),
            vec![PathBuf::from("config/language/custom.toml")]
        );
    }
}

fn initialize_tokenizer(
    tokenizer_config: &TokenizerConfig,
    cache_dir: &Path,
) -> Result<SharedTokenizer> {
    if let Some(path) = tokenizer_config.storage_path(cache_dir) {
        tokenizer_config
            .load(&path)
            .with_context(|| format!("failed to load tokenizer {}", path.display()))
    } else {
        tokenizer_config
            .fit(std::iter::empty::<&str>())
            .context("failed to initialize tokenizer")
    }
}

fn load_generation_tokenizer(
    config: &TrainingConfig,
    source: &GenerationTokenizerSourceConfig,
) -> Result<SharedTokenizer> {
    match source {
        GenerationTokenizerSourceConfig::Dataset => {
            initialize_tokenizer(&config.dataset.tokenizer, &config.dataset.cache_dir)
        }
        GenerationTokenizerSourceConfig::Config {
            cache_dir,
            tokenizer,
        } => initialize_tokenizer(
            tokenizer,
            cache_dir
                .as_deref()
                .unwrap_or(config.dataset.cache_dir.as_path()),
        ),
    }
}

fn resolve_generation_output_format(
    requested: GenerationOutputFormat,
    decode_tokenizer: &dyn Tokenizer,
) -> ResolvedGenerationOutputFormat {
    match requested {
        GenerationOutputFormat::Auto => {
            if decode_tokenizer.as_any().is::<PretokenizedTokenizer>() {
                ResolvedGenerationOutputFormat::TokenIds
            } else {
                ResolvedGenerationOutputFormat::DecodedText
            }
        }
        GenerationOutputFormat::DecodedText => ResolvedGenerationOutputFormat::DecodedText,
        GenerationOutputFormat::TokenIds => ResolvedGenerationOutputFormat::TokenIds,
    }
}

fn render_token_ids(ids: &[u32]) -> String {
    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
}

fn sanitize_display_text(text: &str) -> String {
    let mut sanitized = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_control() && !matches!(ch, '\n' | '\r' | '\t') {
            let code = ch as u32;
            if code <= 0xFF {
                sanitized.push_str(&format!("\\x{code:02X}"));
            } else {
                sanitized.push_str(&format!("\\u{{{code:X}}}"));
            }
        } else {
            sanitized.push(ch);
        }
    }
    sanitized
}

fn render_output(
    ids: &[u32],
    decode_tokenizer: &dyn Tokenizer,
    output_format: ResolvedGenerationOutputFormat,
    stop_at_eos: bool,
) -> String {
    match output_format {
        ResolvedGenerationOutputFormat::DecodedText => {
            sanitize_display_text(&decode_tokenizer.decode_with_options(ids, stop_at_eos))
        }
        ResolvedGenerationOutputFormat::TokenIds => render_token_ids(ids),
    }
}

fn should_decode_past_eos(
    generation: &GenerationConfig,
    output_format: ResolvedGenerationOutputFormat,
) -> bool {
    generation.max_chars.is_some()
        && matches!(output_format, ResolvedGenerationOutputFormat::DecodedText)
}

fn should_stop_on_eos(
    decode_tokenizer: &dyn Tokenizer,
    output_format: ResolvedGenerationOutputFormat,
    decode_past_eos: bool,
) -> Option<u32> {
    if decode_past_eos {
        return None;
    }
    match output_format {
        ResolvedGenerationOutputFormat::DecodedText | ResolvedGenerationOutputFormat::TokenIds => {
            decode_tokenizer.eos_id()
        }
    }
}

fn byte_len_for_char_limit(text: &str, max_chars: usize) -> usize {
    if max_chars == 0 {
        return 0;
    }
    let mut count = 0usize;
    for (idx, ch) in text.char_indices() {
        count += 1;
        if count == max_chars {
            return idx + ch.len_utf8();
        }
    }
    text.len()
}

fn write_token_id_chunk<W: Write>(
    writer: &mut W,
    ids: &[u32],
    wrote_any_output: &mut bool,
) -> Result<()> {
    for &id in ids {
        if *wrote_any_output {
            writer
                .write_all(b" ")
                .context("failed to write token separator")?;
        }
        writer
            .write_all(id.to_string().as_bytes())
            .context("failed to write token id")?;
        *wrote_any_output = true;
    }
    Ok(())
}

fn infer_backend<B, Init>(
    config: &TrainingConfig,
    args: &Args,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: Backend + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    let device = B::Device::default();
    #[cfg(feature = "viz")]
    {
        infer_backend_on_device::<B, Init>(config, args, backend_name, device, init_backend, None)
    }
    #[cfg(not(feature = "viz"))]
    {
        infer_backend_on_device::<B, Init>(config, args, backend_name, device, init_backend)
    }
}

fn infer_backend_on_device<B, Init>(
    config: &TrainingConfig,
    args: &Args,
    backend_name: &str,
    device: B::Device,
    init_backend: Init,
    #[cfg(feature = "viz")] mut viz_runtime: Option<VizRuntime<B>>,
) -> Result<()>
where
    B: Backend + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    B::seed(&device, 1337);
    init_backend(&device);

    let model_tokenizer =
        initialize_tokenizer(&config.dataset.tokenizer, &config.dataset.cache_dir)?;

    let checkpoint_dir = args
        .checkpoint
        .clone()
        .unwrap_or_else(|| default_checkpoint_dir(backend_name));
    let (checkpoint_base, epoch) = resolve_checkpoint_base(&checkpoint_dir, args.epoch)?;

    let mut model_config = build_model_config_with_tokenizer(
        &config.model,
        config.training.block_size,
        model_tokenizer.as_ref(),
    )?;
    apply_wgpu_fused_core_override(
        &mut model_config,
        backend_name,
        WgpuFusedCoreOverride {
            recurrent: config.wgpu.inference.fused_core_recurrent,
            rollout: config.wgpu.inference.fused_core_rollout,
        },
    );
    let bitnet_artifact_path = args.bitnet_artifact.clone().or_else(|| {
        candidate_bitnet_artifact_paths(&checkpoint_base, epoch)
            .into_iter()
            .find(|candidate| candidate.is_file())
    });
    let artifact_bundle = bitnet_artifact_path
        .as_ref()
        .map(|artifact_path| {
            load_bitnet_artifact_bundle(artifact_path).with_context(|| {
                format!("failed to load BitNet artifact {}", artifact_path.display())
            })
        })
        .transpose()?;
    let (model, checkpoint_display) = if let (Some(artifact_path), Some(artifact_bundle)) =
        (bitnet_artifact_path.as_ref(), artifact_bundle.as_ref())
        && artifact_bundle.deploy_base_burnpack.is_some()
    {
        let mut model = BDH::<B>::new(model_config.clone(), &device);
        apply_bitnet_artifact_bundle_to_model(&mut model, artifact_bundle, &device).with_context(
            || {
                format!(
                    "failed to apply standalone BitNet artifact {}",
                    artifact_path.display()
                )
            },
        )?;
        (
            model,
            format!("standalone BitNet artifact {}", artifact_path.display()),
        )
    } else {
        let burnpack_policy =
            BurnpackLoadPolicy::default().with_precision(BurnpackPrecisionPreference::PreferF16);
        let burnpack_candidates = candidate_burnpack_paths(&checkpoint_base, burnpack_policy);
        let (mut model, mut checkpoint_display) = if let Some((model, _result)) =
            try_load_model_from_burnpack_candidates(&burnpack_candidates, "BDH model", true, || {
                BDH::<B>::new(model_config.clone(), &device)
            })
            .map_err(|err| anyhow!(err))?
        {
            (model, format_burnpack_checkpoint(&burnpack_candidates))
        } else {
            let mut model = BDH::<B>::new(model_config, &device);
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            let record = recorder
                .load::<<BDH<B> as Module<B>>::Record>(checkpoint_base.clone(), &device)
                .with_context(|| {
                    format!(
                        "failed to load checkpoint {}",
                        format_checkpoint(&checkpoint_base)
                    )
                })?;
            model = model.load_record(record);
            (model, format_checkpoint(&checkpoint_base))
        };
        if let (Some(artifact_path), Some(artifact_bundle)) =
            (bitnet_artifact_path.as_ref(), artifact_bundle.as_ref())
        {
            apply_bitnet_artifact_bundle_to_model(&mut model, artifact_bundle, &device)
                .with_context(|| {
                    format!(
                        "failed to apply BitNet artifact static weights from {}",
                        artifact_path.display()
                    )
                })?;
            checkpoint_display = format!(
                "{checkpoint_display} + BitNet artifact {}",
                artifact_path.display()
            );
        }
        (model, checkpoint_display)
    };

    let mut generation = config.generation.clone();
    apply_generation_overrides(&mut generation, args, config.training.block_size);
    let prompt_tokenizer = load_generation_tokenizer(config, &generation.prompt_tokenizer)?;
    let decode_tokenizer = load_generation_tokenizer(config, &generation.decode_tokenizer)?;
    let output_format =
        resolve_generation_output_format(generation.output_format, decode_tokenizer.as_ref());
    let decode_past_eos = should_decode_past_eos(&generation, output_format);
    let stop_on_eos = should_stop_on_eos(decode_tokenizer.as_ref(), output_format, decode_past_eos);

    let status_msg =
        format!("Loaded epoch {epoch} from {checkpoint_display} using {backend_name} backend.",);
    let stage_profile = std::env::var_os("BDH_STAGE_PROFILE").is_some();
    if stage_profile {
        generation_profile_reset();
        recurrent_profile_reset();
        relu_lowrank_forward_profile_reset();
        logits_projection_profile_reset();
        low_bit_native_projection_profile_reset();
        lowrank_residual_profile_reset();
    }
    let infer_wall_start = stage_profile.then(Instant::now);

    let use_streaming = args.streaming;
    let chunked_plan = resolve_chunked_generation_plan(config, backend_name, &generation);

    if use_streaming {
        let strategy =
            resolve_context_strategy(&generation.context_strategy, config.training.block_size);
        eprintln!("{status_msg}");

        let mut prompt_ids = prompt_tokenizer.encode(&generation.prompt, false, false);
        if let ContextStrategy::Sliding { window } = strategy
            && prompt_ids.len() > window
        {
            prompt_ids = prompt_ids[prompt_ids.len() - window..].to_vec();
        }

        let prompt_tokens: Vec<i64> = prompt_ids.iter().map(|&id| id as i64).collect();
        let prompt_ids_u32: Vec<u32> = prompt_ids.to_vec();

        let mut writer = io::stdout();
        let prompt_text = render_output(
            &prompt_ids_u32,
            decode_tokenizer.as_ref(),
            output_format,
            true,
        );
        writer
            .write_all(prompt_text.as_bytes())
            .context("failed to write prompt to stdout")?;
        writer.flush().context("failed to flush stdout")?;

        let mut generated_ids: Vec<u32> = Vec::new();
        let mut last_print_len = 0usize;
        let mut wrote_any_output = !prompt_text.is_empty();
        let mut generated_display_chars = 0usize;
        let mut stream_err: Option<anyhow::Error> = None;
        let max_tokens = normalize_max_tokens(generation.max_tokens);
        let settings = burn_dragon::language::GenerationSettings {
            max_new_tokens: max_tokens,
            temperature: generation.temperature,
            top_k: generation.top_k,
            strategy,
        };

        let chunked_plan = chunked_plan.filter(|_| {
            if generation.max_chars.is_some() {
                return false;
            }
            #[cfg(feature = "viz")]
            {
                viz_runtime.is_none()
            }
            #[cfg(not(feature = "viz"))]
            {
                true
            }
        });

        if let Some((chunk_tokens, buffer_tokens)) = chunked_plan {
            let mut on_chunk = |chunk: &[i64]| {
                if stream_err.is_some() {
                    return;
                }
                let mut chunk_ids = Vec::with_capacity(chunk.len());
                for &token in chunk {
                    if let Ok(token_u32) = u32::try_from(token) {
                        chunk_ids.push(token_u32);
                        generated_ids.push(token_u32);
                    }
                }
                match output_format {
                    ResolvedGenerationOutputFormat::DecodedText => {
                        let decoded =
                            decode_tokenizer.decode_with_options(&generated_ids, !decode_past_eos);
                        if decoded.len() <= last_print_len {
                            return;
                        }
                        let new_text = &decoded[last_print_len..];
                        if new_text.is_empty() {
                            return;
                        }
                        let mut sanitized_new_text = sanitize_display_text(new_text);
                        if let Some(max_chars) = generation.max_chars {
                            let remaining = max_chars.saturating_sub(generated_display_chars);
                            if remaining == 0 {
                                return;
                            }
                            let keep_len = byte_len_for_char_limit(&sanitized_new_text, remaining);
                            sanitized_new_text.truncate(keep_len);
                            generated_display_chars += sanitized_new_text.chars().count();
                        }
                        if let Err(err) = writer.write_all(sanitized_new_text.as_bytes()) {
                            stream_err = Some(anyhow!("failed to write streamed chunk: {err}"));
                            return;
                        }
                        last_print_len = decoded.len();
                    }
                    ResolvedGenerationOutputFormat::TokenIds => {
                        if let Err(err) =
                            write_token_id_chunk(&mut writer, &chunk_ids, &mut wrote_any_output)
                        {
                            stream_err = Some(err.context("failed to write streamed token ids"));
                            return;
                        }
                    }
                }
                if let Err(err) = writer.flush() {
                    stream_err = Some(anyhow!("failed to flush stdout during streaming: {err}"));
                }
            };
            let _ = generate_tokens_chunked(
                &model,
                prompt_tokens,
                &device,
                settings,
                chunk_tokens,
                buffer_tokens,
                stop_on_eos.map(i64::from),
                Some(&mut on_chunk),
            )?;
        } else {
            let (mut state, mut last_logits) = prefill_state::<B>(&model, &prompt_tokens, &device)?;

            if let ContextStrategy::Sliding { window } = strategy
                && window > 0
                && state.position > window
            {
                state.trim(window);
            }

            let mut generated = 0usize;
            while max_tokens.is_none_or(|max| generated < max) {
                #[cfg(feature = "viz")]
                if let Some(viz) = viz_runtime.as_ref()
                    && viz.stop.load(Ordering::Relaxed)
                {
                    break;
                }

                let (next, logits) = sample_next_token(
                    &model,
                    &mut state,
                    last_logits,
                    generation.temperature,
                    generation.top_k,
                    &device,
                )?;
                last_logits = logits;
                generated = generated.saturating_add(1);

                if let Ok(token_u32) = u32::try_from(next) {
                    generated_ids.push(token_u32);
                    if Some(token_u32) == stop_on_eos {
                        break;
                    }
                    match output_format {
                        ResolvedGenerationOutputFormat::DecodedText => {
                            let decoded = decode_tokenizer
                                .decode_with_options(&generated_ids, !decode_past_eos);

                            if decoded.len() > last_print_len {
                                let new_text = &decoded[last_print_len..];
                                if !new_text.is_empty() {
                                    let mut sanitized_new_text = sanitize_display_text(new_text);
                                    let mut reached_char_limit = false;
                                    if let Some(max_chars) = generation.max_chars {
                                        let remaining =
                                            max_chars.saturating_sub(generated_display_chars);
                                        if remaining == 0 {
                                            break;
                                        }
                                        let keep_len =
                                            byte_len_for_char_limit(&sanitized_new_text, remaining);
                                        sanitized_new_text.truncate(keep_len);
                                        generated_display_chars +=
                                            sanitized_new_text.chars().count();
                                        reached_char_limit = generated_display_chars >= max_chars;
                                    }
                                    if let Err(err) =
                                        writer.write_all(sanitized_new_text.as_bytes())
                                    {
                                        stream_err =
                                            Some(anyhow!("failed to write streamed token: {err}"));
                                        break;
                                    }
                                    if let Err(err) = writer.flush() {
                                        stream_err = Some(anyhow!(
                                            "failed to flush stdout during streaming: {err}"
                                        ));
                                        break;
                                    }
                                    if reached_char_limit {
                                        break;
                                    }
                                }
                                last_print_len = decoded.len();
                            }
                        }
                        ResolvedGenerationOutputFormat::TokenIds => {
                            if let Err(err) = write_token_id_chunk(
                                &mut writer,
                                &[token_u32],
                                &mut wrote_any_output,
                            ) {
                                stream_err =
                                    Some(err.context("failed to write streamed token ids"));
                                break;
                            }
                            if let Err(err) = writer.flush() {
                                stream_err =
                                    Some(anyhow!("failed to flush stdout during streaming: {err}"));
                                break;
                            }
                        }
                    }
                }

                #[cfg(feature = "viz")]
                if let Some(viz) = viz_runtime.as_mut() {
                    let token_index = state.position.saturating_sub(1);
                    if viz.encoder.should_capture(token_index) {
                        let layers = state.take_viz();
                        let frame = viz.encoder.step(&layers, token_index);
                        viz.sender.try_send(frame);
                    }
                }

                if stream_err.is_some() {
                    break;
                }

                #[cfg(feature = "viz")]
                if let Some(viz) = viz_runtime.as_ref()
                    && viz.stop.load(Ordering::Relaxed)
                {
                    break;
                }

                if let ContextStrategy::Sliding { window } = strategy
                    && window > 0
                    && state.position > window
                {
                    state.trim(window);
                }
            }
        }

        if let Some(err) = stream_err {
            return Err(err);
        }

        writer
            .write_all(b"\n")
            .context("failed to write trailing newline")?;
        writer.flush().context("failed to flush stdout")?;
    } else {
        let output = if let Some((chunk_tokens, buffer_tokens)) = chunked_plan {
            generate_output_chunked::<B>(
                &model,
                prompt_tokenizer.as_ref(),
                decode_tokenizer.as_ref(),
                output_format,
                !decode_past_eos,
                stop_on_eos,
                &device,
                config.training.block_size,
                &generation,
                chunk_tokens,
                buffer_tokens,
            )?
        } else {
            generate_output::<B>(
                &model,
                prompt_tokenizer.as_ref(),
                decode_tokenizer.as_ref(),
                output_format,
                !decode_past_eos,
                stop_on_eos,
                &device,
                config.training.block_size,
                &generation,
            )?
        };

        eprintln!("{status_msg}");
        println!("{output}");
    }

    if let Some(start) = infer_wall_start {
        let elapsed_ns = start.elapsed().as_nanos();
        let generation = generation_profile_snapshot();
        let recurrent = recurrent_profile_snapshot();
        let lowrank_forward = relu_lowrank_forward_profile_snapshot();
        let logits_projection = logits_projection_profile_snapshot();
        let low_bit_lowrank = low_bit_native_lowrank_profile_snapshot();
        let low_bit_decoder_tail = low_bit_native_decoder_tail_profile_snapshot();
        let residual = lowrank_residual_profile_snapshot();
        eprintln!(
            "[stage-profile][inference] total_ns={elapsed_ns} prefill_forward_ns={} token_forward_ns={} sample_host_transfer_ns={} sample_cpu_ns={} token_tensor_copy_ns={} chunk_flush_ns={} token_steps={} prefill_tokens={} host_sync_points={} chunk_flushes={} chunk_flushed_tokens={} host_to_device_copy_bytes={} device_to_host_copy_bytes={} recurrent_calls={} recurrent_total_ns={} recurrent_setup_ns={} recurrent_copy_ns={} recurrent_dispatch_ns={}",
            generation.prefill_forward_ns,
            generation.token_forward_ns,
            generation.sample_host_transfer_ns,
            generation.sample_cpu_ns,
            generation.token_tensor_copy_ns,
            generation.chunk_flush_ns,
            generation.token_steps,
            generation.prefill_tokens,
            generation.host_sync_points,
            generation.chunk_flushes,
            generation.chunk_flushed_tokens,
            generation.host_to_device_copy_bytes,
            generation.device_to_host_copy_bytes,
            recurrent.calls,
            recurrent.total_ns,
            recurrent.setup_ns,
            recurrent.copy_ns,
            recurrent.dispatch_ns,
        );
        eprintln!(
            "[stage-profile][inference-lowrank-forward] calls={} launches={} total_ns={}",
            lowrank_forward.calls, lowrank_forward.launches, lowrank_forward.total_ns,
        );
        eprintln!(
            "[stage-profile][inference-lowbit-lowrank] calls={} total_ns={} quantize_ns={} prepacked_quantize_ns={} raw_cuda_ns={} fused_ns={} reference_ns={} dynamic_scale_calls={} cached_scale_hits={}",
            low_bit_lowrank.calls,
            low_bit_lowrank.total_ns,
            low_bit_lowrank.quantize_ns,
            low_bit_lowrank.prepacked_quantize_ns,
            low_bit_lowrank.raw_cuda_ns,
            low_bit_lowrank.fused_ns,
            low_bit_lowrank.reference_ns,
            low_bit_lowrank.dynamic_scale_calls,
            low_bit_lowrank.cached_scale_hits,
        );
        eprintln!(
            "[stage-profile][inference-lowbit-decoder-tail] calls={} total_ns={} quantize_ns={} prepacked_quantize_ns={} raw_cuda_ns={} fused_ns={} reference_ns={} dynamic_scale_calls={} cached_scale_hits={}",
            low_bit_decoder_tail.calls,
            low_bit_decoder_tail.total_ns,
            low_bit_decoder_tail.quantize_ns,
            low_bit_decoder_tail.prepacked_quantize_ns,
            low_bit_decoder_tail.raw_cuda_ns,
            low_bit_decoder_tail.fused_ns,
            low_bit_decoder_tail.reference_ns,
            low_bit_decoder_tail.dynamic_scale_calls,
            low_bit_decoder_tail.cached_scale_hits,
        );
        eprintln!(
            "[stage-profile][inference-logits-projection] calls={} total_ns={}",
            logits_projection.calls, logits_projection.total_ns,
        );
        eprintln!(
            "[stage-profile][inference-residual-step] calls={} total_ns={} x_projection_ns={} x_post_quant_ns={} attention_norm_ns={} attention_mixer_ns={} attention_post_norm_ns={} y_projection_ns={} y_post_quant_ns={} y_neuron_ns={} decoder_tail_ns={} mlp_norm_ns={} residual_combine_ns={}",
            residual.calls,
            residual.total_ns,
            residual.x_projection_ns,
            residual.x_post_quant_ns,
            residual.attention_norm_ns,
            residual.attention_mixer_ns,
            residual.attention_post_norm_ns,
            residual.y_projection_ns,
            residual.y_post_quant_ns,
            residual.y_neuron_ns,
            residual.decoder_tail_ns,
            residual.mlp_norm_ns,
            residual.residual_combine_ns,
        );
    }

    Ok(())
}

fn generate_output<B: Backend>(
    model: &BDH<B>,
    prompt_tokenizer: &dyn Tokenizer,
    decode_tokenizer: &dyn Tokenizer,
    output_format: ResolvedGenerationOutputFormat,
    stop_at_eos: bool,
    stop_on_eos: Option<u32>,
    device: &B::Device,
    block_size: usize,
    generation: &GenerationConfig,
) -> Result<String> {
    let strategy = resolve_context_strategy(&generation.context_strategy, block_size);
    let mut prompt_ids = prompt_tokenizer.encode(&generation.prompt, false, false);
    if let ContextStrategy::Sliding { window } = strategy
        && prompt_ids.len() > window
    {
        prompt_ids = prompt_ids[prompt_ids.len() - window..].to_vec();
    }

    let prompt_tokens: Vec<i64> = prompt_ids.iter().map(|&id| id as i64).collect();
    let settings = burn_dragon::language::GenerationSettings {
        max_new_tokens: normalize_max_tokens(generation.max_tokens),
        temperature: generation.temperature,
        top_k: generation.top_k,
        strategy,
    };
    let max_chars = generation.max_chars;
    if max_chars.is_some() && matches!(output_format, ResolvedGenerationOutputFormat::DecodedText) {
        let prompt_ids_u32 = prompt_ids.clone();
        let prompt_text = render_output(&prompt_ids_u32, decode_tokenizer, output_format, true);
        let (mut state, mut last_logits) = prefill_state::<B>(model, &prompt_tokens, device)?;
        if let ContextStrategy::Sliding { window } = strategy
            && window > 0
            && state.position > window
        {
            state.trim(window);
        }
        let mut generated_ids = Vec::new();
        let mut last_render_len = 0usize;
        let mut generated_suffix = String::new();
        let mut generated_display_chars = 0usize;
        let max_new_tokens = normalize_max_tokens(generation.max_tokens);
        let max_chars = max_chars.unwrap_or(usize::MAX);

        while max_new_tokens.is_none_or(|max| generated_ids.len() < max) {
            let (next, logits) = sample_next_token(
                model,
                &mut state,
                last_logits,
                generation.temperature,
                generation.top_k,
                device,
            )?;
            last_logits = logits;
            if let Ok(token_u32) = u32::try_from(next) {
                generated_ids.push(token_u32);
                if Some(token_u32) == stop_on_eos {
                    break;
                }
                let decoded = decode_tokenizer.decode_with_options(&generated_ids, stop_at_eos);
                if decoded.len() > last_render_len {
                    let new_text = &decoded[last_render_len..];
                    let mut sanitized_new_text = sanitize_display_text(new_text);
                    let remaining = max_chars.saturating_sub(generated_display_chars);
                    if remaining == 0 {
                        break;
                    }
                    let keep_len = byte_len_for_char_limit(&sanitized_new_text, remaining);
                    sanitized_new_text.truncate(keep_len);
                    generated_display_chars += sanitized_new_text.chars().count();
                    generated_suffix.push_str(&sanitized_new_text);
                    last_render_len = decoded.len();
                    if generated_display_chars >= max_chars {
                        break;
                    }
                }
            }
            if let ContextStrategy::Sliding { window } = strategy
                && window > 0
                && state.position > window
            {
                state.trim(window);
            }
        }

        return Ok(format!("{prompt_text}{generated_suffix}"));
    }

    let tokens_all = if stop_on_eos.is_some() {
        let (mut state, mut last_logits) = prefill_state::<B>(model, &prompt_tokens, device)?;
        if let ContextStrategy::Sliding { window } = strategy
            && window > 0
            && state.position > window
        {
            state.trim(window);
        }
        let mut generated_ids = Vec::new();
        let max_new_tokens = normalize_max_tokens(generation.max_tokens);
        while max_new_tokens.is_none_or(|max| generated_ids.len() < max) {
            let (next, logits) = sample_next_token(
                model,
                &mut state,
                last_logits,
                generation.temperature,
                generation.top_k,
                device,
            )?;
            last_logits = logits;
            if let Ok(token_u32) = u32::try_from(next) {
                generated_ids.push(token_u32);
                if Some(token_u32) == stop_on_eos {
                    break;
                }
            }
            if let ContextStrategy::Sliding { window } = strategy
                && window > 0
                && state.position > window
            {
                state.trim(window);
            }
        }
        let mut tokens_all = prompt_tokens.clone();
        tokens_all.extend(generated_ids.into_iter().map(i64::from));
        tokens_all
    } else {
        generate_tokens(model, prompt_tokens, device, settings, None)?
    };
    let decoded_ids: Vec<u32> = tokens_all
        .iter()
        .filter_map(|&tok| (tok >= 0).then_some(tok as u32))
        .collect();
    Ok(render_output(
        &decoded_ids,
        decode_tokenizer,
        output_format,
        stop_at_eos,
    ))
}

fn generate_output_chunked<B: Backend>(
    model: &BDH<B>,
    prompt_tokenizer: &dyn Tokenizer,
    decode_tokenizer: &dyn Tokenizer,
    output_format: ResolvedGenerationOutputFormat,
    stop_at_eos: bool,
    stop_on_eos: Option<u32>,
    device: &B::Device,
    block_size: usize,
    generation: &GenerationConfig,
    chunk_tokens: usize,
    device_buffer_tokens: usize,
) -> Result<String> {
    let strategy = resolve_context_strategy(&generation.context_strategy, block_size);
    let mut prompt_ids = prompt_tokenizer.encode(&generation.prompt, false, false);
    if let ContextStrategy::Sliding { window } = strategy
        && prompt_ids.len() > window
    {
        prompt_ids = prompt_ids[prompt_ids.len() - window..].to_vec();
    }

    let prompt_tokens: Vec<i64> = prompt_ids.iter().map(|&id| id as i64).collect();
    let settings = burn_dragon::language::GenerationSettings {
        max_new_tokens: normalize_max_tokens(generation.max_tokens),
        temperature: generation.temperature,
        top_k: generation.top_k,
        strategy,
    };
    let tokens_all = generate_tokens_chunked(
        model,
        prompt_tokens,
        device,
        settings,
        chunk_tokens,
        device_buffer_tokens,
        stop_on_eos.map(i64::from),
        None,
    )?;
    let decoded_ids: Vec<u32> = tokens_all
        .iter()
        .filter_map(|&tok| (tok >= 0).then_some(tok as u32))
        .collect();
    Ok(render_output(
        &decoded_ids,
        decode_tokenizer,
        output_format,
        stop_at_eos,
    ))
}

#[cfg(feature = "viz")]
fn infer_backend_with_viz<B, Init>(
    config: &TrainingConfig,
    args: &Args,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: Backend<Device = burn_wgpu::WgpuDevice> + 'static,
    B::Device: Default + Clone + Send + Sync + 'static,
    Init: Fn(&B::Device) + Send + 'static,
    (): bevy_burn::gpu_burn_to_bevy::BurnBevyPrepare<B>,
{
    let model_config = build_model_config(&config.model, config.training.block_size);
    let dims = VizDimensions {
        layers: model_config.n_layer,
        heads: model_config.n_head,
        latent_per_head: model_config.latent_per_head(),
    };
    let viz_config = VizConfig::default();

    let (exit_tx, exit_rx) = mpsc::channel();
    let overlay = viz::start_overlay_native::<B>(viz_config.clone(), dims, Some(exit_rx));
    let stop_flag = overlay.handle().stop_flag();
    let (viz_handle, mut app) = overlay.split();
    let device = viz_handle.device().clone();
    let sender = viz_handle.sender();

    #[cfg(not(target_arch = "wasm32"))]
    {
        let ctrlc_stop = stop_flag.clone();
        let ctrlc_exit = exit_tx.clone();
        let _ = ctrlc::set_handler(move || {
            ctrlc_stop.store(true, Ordering::Relaxed);
            let _ = ctrlc_exit.send(());
        });
    }

    let backend_name = backend_name.to_string();
    let config = config.clone();
    let args = args.clone();
    let stop_for_thread = stop_flag.clone();
    let infer_thread = std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let viz_runtime = VizRuntime {
                encoder: VizEncoder::new(
                    viz_config,
                    dims.layers,
                    dims.heads,
                    dims.latent_per_head,
                    &device,
                ),
                sender,
                stop: stop_for_thread,
            };
            infer_backend_on_device::<B, Init>(
                &config,
                &args,
                backend_name.as_str(),
                device,
                init_backend,
                Some(viz_runtime),
            )
        }));

        let _ = exit_tx.send(());
        match result {
            Ok(outcome) => outcome,
            Err(_) => Err(anyhow!("inference thread panicked")),
        }
    });

    app.run();
    infer_thread
        .join()
        .map_err(|_| anyhow!("inference thread crashed"))??;
    Ok(())
}

fn apply_generation_overrides(generation: &mut GenerationConfig, args: &Args, block_size: usize) {
    if let Some(prompt) = &args.prompt {
        generation.prompt = prompt.clone();
    }
    if let Some(max_tokens) = args.max_tokens {
        generation.max_tokens = if max_tokens < 0 {
            None
        } else {
            Some(max_tokens)
        };
    }
    if let Some(max_chars) = args.max_chars {
        generation.max_chars = Some(max_chars);
    }
    if let Some(temperature) = args.temperature {
        generation.temperature = temperature;
    }
    if let Some(top_k) = args.top_k {
        generation.top_k = Some(top_k);
    }
    if let Some(mode) = args.context_mode {
        generation.context_strategy = match mode {
            ContextModeArg::Infinite => ContextStrategyConfig::Infinite,
            ContextModeArg::Sliding => ContextStrategyConfig::Sliding {
                window: args.context_window.unwrap_or(block_size).max(1),
            },
        };
    }
}

fn resolve_checkpoint_base(path: &Path, epoch: Option<usize>) -> Result<(PathBuf, usize)> {
    if path.is_dir() {
        let target_epoch = epoch.unwrap_or(find_latest_epoch(path)?);
        let base = path.join(format!("model-{target_epoch}"));
        ensure_checkpoint_exists(&base)?;
        return Ok((base, target_epoch));
    }

    let mut base = strip_checkpoint_extension(path);

    let detected_epoch = parse_epoch_from_stem(&base);
    let target_epoch = match (epoch, detected_epoch) {
        (Some(explicit), Some(detected)) if explicit != detected => {
            let parent = base.parent().map(Path::to_path_buf).unwrap_or_default();
            base = parent.join(format!("model-{explicit}"));
            explicit
        }
        (Some(explicit), _) => {
            if detected_epoch.is_none() {
                let parent = base
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from("runs").join("checkpoint"));
                base = parent.join(format!("model-{explicit}"));
            }
            explicit
        }
        (None, Some(detected)) => detected,
        (None, None) => {
            return Err(anyhow!(
                "unable to infer checkpoint epoch from {}; provide --epoch",
                path.display()
            ));
        }
    };

    ensure_checkpoint_exists(&base)?;
    Ok((base, target_epoch))
}

fn ensure_checkpoint_exists(base: &Path) -> Result<()> {
    let mut candidate = base.to_path_buf();
    candidate.set_extension("bin");
    if candidate.is_file() {
        return Ok(());
    }

    let burnpack_candidates = candidate_burnpack_paths(
        base,
        BurnpackLoadPolicy::default().with_precision(BurnpackPrecisionPreference::PreferF16),
    );
    for candidate in burnpack_candidates {
        if candidate.is_file() || burnpack_parts_manifest_path(candidate.as_path()).is_file() {
            return Ok(());
        }
    }

    Err(anyhow!(
        "checkpoint weights not found for {} (.bin, .bpk, or .bpk.parts.json)",
        base.display()
    ))
}

fn find_latest_epoch(dir: &Path) -> Result<usize> {
    let mut max_epoch = None;
    for entry in fs::read_dir(dir)
        .with_context(|| format!("failed to read checkpoint directory {}", dir.display()))?
    {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let mut base = entry.path();
        base.set_extension("");
        if let Some(epoch) = parse_epoch_from_stem(&base) {
            let updated = max_epoch
                .map(|current: usize| current.max(epoch))
                .unwrap_or(epoch);
            max_epoch = Some(updated);
        }
    }

    max_epoch.ok_or_else(|| anyhow!("no model checkpoints found in {}", dir.display()))
}

fn parse_epoch_from_stem(path: &Path) -> Option<usize> {
    let stem = path.file_name()?.to_string_lossy();
    let stem = stem.strip_suffix(".bin").unwrap_or(&stem);
    let stem = stem.strip_suffix(".bpk").unwrap_or(stem);
    let epoch_part = stem.strip_prefix("model-")?;
    epoch_part.parse().ok()
}

fn format_checkpoint(base: &Path) -> String {
    let mut path = base.to_path_buf();
    path.set_extension("bin");
    path.display().to_string()
}

fn format_burnpack_checkpoint(candidates: &[PathBuf]) -> String {
    for candidate in candidates {
        let manifest = burnpack_parts_manifest_path(candidate.as_path());
        if manifest.is_file() {
            return manifest.display().to_string();
        }
        if candidate.is_file() {
            return candidate.display().to_string();
        }
    }
    candidates
        .first()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "<missing burnpack>".to_string())
}

fn strip_checkpoint_extension(path: &Path) -> PathBuf {
    let display = path.to_string_lossy();
    if let Some(stripped) = display.strip_suffix(".parts.json") {
        let mut base = PathBuf::from(stripped);
        if base.extension().is_some() {
            base.set_extension("");
        }
        return base;
    }

    let mut base = path.to_path_buf();
    if base.extension().is_some() {
        base.set_extension("");
    }
    base
}

#[derive(Parser, Debug, Clone)]
#[command(
    author,
    version,
    about = "Run inference with a trained Baby Dragon Hatchling model"
)]
struct Args {
    /// Additional configuration files applied in order (later files override earlier ones).
    #[arg(short = 'c', long = "config", value_name = "PATH")]
    config: Vec<PathBuf>,
    /// Backend to use for inference.
    #[arg(long, value_enum, default_value_t = BackendArg::Cuda)]
    backend: BackendArg,
    /// Path to the checkpoint directory or file.
    #[arg(long, value_name = "PATH")]
    checkpoint: Option<PathBuf>,
    /// Specific checkpoint epoch to load.
    #[arg(long, value_name = "N")]
    epoch: Option<usize>,
    /// Optional BitNet packed static-weight artifact override.
    #[arg(long, value_name = "PATH")]
    bitnet_artifact: Option<PathBuf>,
    /// Override the prompt used for generation.
    #[arg(long)]
    prompt: Option<String>,
    /// Override the number of tokens to generate.
    #[arg(long, value_name = "N")]
    max_tokens: Option<i64>,
    /// Stop after emitting this many generated display characters.
    #[arg(long, value_name = "N")]
    max_chars: Option<usize>,
    /// Override the sampling temperature.
    #[arg(long, value_name = "T")]
    temperature: Option<f32>,
    /// Override the top-k sampling parameter.
    #[arg(long, value_name = "K")]
    top_k: Option<usize>,
    /// Override the context strategy.
    #[arg(long, value_enum)]
    context_mode: Option<ContextModeArg>,
    /// Sliding window size when using `--context-mode=sliding`.
    #[arg(long, value_name = "N")]
    context_window: Option<usize>,
    /// Stream tokens to stdout as they are generated.
    #[arg(long)]
    streaming: bool,
    /// whether or not to spawn visualization app, wgpu only
    #[arg(long)]
    #[cfg(feature = "viz")]
    viz: bool,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendArg {
    Wgpu,
    Cuda,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ContextModeArg {
    Infinite,
    Sliding,
}

fn backend_name(backend: BackendArg) -> &'static str {
    match backend {
        BackendArg::Wgpu => "wgpu",
        BackendArg::Cuda => "cuda",
    }
}

fn normalize_max_tokens(max_tokens: Option<i64>) -> Option<usize> {
    match max_tokens {
        Some(value) if value >= 0 => Some(value as usize),
        _ => None,
    }
}

fn parse_chunked_override(raw: Option<&str>) -> Option<bool> {
    let normalized = raw?.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn resolve_chunked_generation_plan_for_values(
    is_wgpu_backend: bool,
    generation_executor: WgpuGenerationExecutor,
    top_k: Option<usize>,
    chunk_tokens: usize,
    device_buffer_tokens: usize,
    env_override: Option<&str>,
) -> Option<(usize, usize)> {
    if top_k != Some(1) {
        return None;
    }

    match parse_chunked_override(env_override) {
        Some(false) => return None,
        Some(true) => {}
        None => {}
    }

    let chunk_tokens = chunk_tokens.max(1);
    let device_buffer_tokens = device_buffer_tokens.max(chunk_tokens);
    if is_wgpu_backend || matches!(generation_executor, WgpuGenerationExecutor::RolloutChunked) {
        return Some((chunk_tokens, device_buffer_tokens));
    }
    Some((chunk_tokens, device_buffer_tokens))
}

fn resolve_chunked_generation_plan(
    config: &TrainingConfig,
    backend_name: &str,
    generation: &GenerationConfig,
) -> Option<(usize, usize)> {
    if generation.max_chars.is_some() {
        return None;
    }
    resolve_chunked_generation_plan_for_values(
        burn_dragon::language::is_wgpu_backend_name(backend_name),
        config.wgpu.inference.generation_executor.clone(),
        generation.top_k,
        config.wgpu.inference.generation_chunk_tokens,
        config.wgpu.inference.generation_device_buffer_tokens,
        std::env::var("BDH_INFER_CHUNKED").ok().as_deref(),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        Args, ResolvedGenerationOutputFormat, WgpuFusedCoreOverride,
        apply_wgpu_fused_core_override, parse_chunked_override, render_output, render_token_ids,
        resolve_chunked_generation_plan_for_values, resolve_generation_output_format,
        sanitize_display_text,
    };
    use burn_dragon::core::BDHConfig;
    use burn_dragon::train::WgpuGenerationExecutor;
    use burn_dragon_language::{
        GenerationOutputFormat,
        tokenizer::{Tokenizer, byte::ByteTokenizer, pretokenized::PretokenizedTokenizer},
    };
    use clap::Parser;
    use std::path::PathBuf;

    #[test]
    fn wgpu_backend_override_enables_fused_recurrent_path() {
        let mut model_config = BDHConfig::default();
        model_config.fused_kernels.enabled = false;
        model_config.fused_kernels.set_wgpu_recurrent_kernel(false);
        model_config.fused_kernels.set_wgpu_rollout_fused(false);

        apply_wgpu_fused_core_override(
            &mut model_config,
            "wgpu",
            WgpuFusedCoreOverride {
                recurrent: Some(true),
                rollout: None,
            },
        );

        assert!(
            model_config.fused_kernels.enabled,
            "wgpu backend override should enable fused kernels for recurrent path selection"
        );
        assert!(
            model_config.fused_kernels.wgpu_recurrent_kernel,
            "wgpu recurrent kernel should be enabled by override"
        );
        assert!(model_config.fused_kernels.wgpu_rollout_fused);
    }

    #[test]
    fn wgpu_backend_override_can_disable_recurrent_kernel_without_disabling_other_fusion() {
        let mut model_config = BDHConfig::default();
        model_config.fused_kernels.enabled = true;
        model_config.fused_kernels.set_wgpu_recurrent_kernel(true);
        model_config.fused_kernels.set_wgpu_rollout_fused(true);

        apply_wgpu_fused_core_override(
            &mut model_config,
            "wgpu",
            WgpuFusedCoreOverride {
                recurrent: Some(false),
                rollout: None,
            },
        );

        assert!(
            model_config.fused_kernels.enabled,
            "disabling recurrent override should preserve other fused kernel settings"
        );
        assert!(
            !model_config.fused_kernels.wgpu_recurrent_kernel,
            "wgpu recurrent kernel should be disabled by override"
        );
        assert!(!model_config.fused_kernels.wgpu_rollout_fused);
    }

    #[test]
    fn non_wgpu_backends_ignore_override() {
        let mut model_config = BDHConfig::default();
        model_config.fused_kernels.enabled = false;
        model_config.fused_kernels.set_wgpu_recurrent_kernel(false);
        model_config.fused_kernels.set_wgpu_rollout_fused(false);

        apply_wgpu_fused_core_override(
            &mut model_config,
            "cuda",
            WgpuFusedCoreOverride {
                recurrent: Some(true),
                rollout: Some(true),
            },
        );

        assert!(!model_config.fused_kernels.enabled);
        assert!(!model_config.fused_kernels.wgpu_recurrent_kernel);
        assert!(!model_config.fused_kernels.wgpu_rollout_fused);
    }

    #[test]
    fn parse_chunked_override_accepts_common_boolean_spellings() {
        assert_eq!(parse_chunked_override(Some("1")), Some(true));
        assert_eq!(parse_chunked_override(Some("true")), Some(true));
        assert_eq!(parse_chunked_override(Some("on")), Some(true));
        assert_eq!(parse_chunked_override(Some("0")), Some(false));
        assert_eq!(parse_chunked_override(Some("false")), Some(false));
        assert_eq!(parse_chunked_override(Some("off")), Some(false));
        assert_eq!(parse_chunked_override(Some("maybe")), None);
        assert_eq!(parse_chunked_override(None), None);
    }

    #[test]
    fn cuda_chunked_inference_defaults_on_for_argmax_sampling() {
        let plan = resolve_chunked_generation_plan_for_values(
            false,
            WgpuGenerationExecutor::Baseline,
            Some(1),
            8,
            64,
            None,
        );
        assert_eq!(plan, Some((8, 64)));
    }

    #[test]
    fn wgpu_chunked_inference_defaults_on_for_argmax_sampling() {
        let plan = resolve_chunked_generation_plan_for_values(
            true,
            WgpuGenerationExecutor::Baseline,
            Some(1),
            8,
            64,
            None,
        );
        assert_eq!(plan, Some((8, 64)));
    }

    #[test]
    fn override_can_force_or_disable_chunked_inference() {
        let forced = resolve_chunked_generation_plan_for_values(
            true,
            WgpuGenerationExecutor::Baseline,
            Some(1),
            8,
            64,
            Some("1"),
        );
        assert_eq!(forced, Some((8, 64)));

        let disabled = resolve_chunked_generation_plan_for_values(
            false,
            WgpuGenerationExecutor::RolloutChunked,
            Some(1),
            8,
            64,
            Some("0"),
        );
        assert_eq!(disabled, None);
    }

    #[test]
    fn bitnet_artifact_arg_parses() {
        let args = Args::parse_from([
            "infer",
            "--bitnet-artifact",
            "runs/example/deploy/model-3.bitnet_artifact.bin.gz",
        ]);
        assert_eq!(
            args.bitnet_artifact,
            Some(PathBuf::from(
                "runs/example/deploy/model-3.bitnet_artifact.bin.gz"
            ))
        );
    }

    #[test]
    fn auto_output_format_uses_token_ids_for_pretokenized_decode() {
        let tokenizer = PretokenizedTokenizer::new(32, None, None, None, None);
        assert_eq!(
            resolve_generation_output_format(GenerationOutputFormat::Auto, &tokenizer),
            ResolvedGenerationOutputFormat::TokenIds
        );
    }

    #[test]
    fn auto_output_format_uses_decoded_text_for_textual_decode_tokenizers() {
        let tokenizer = ByteTokenizer::new(true);
        assert_eq!(
            resolve_generation_output_format(GenerationOutputFormat::Auto, &tokenizer),
            ResolvedGenerationOutputFormat::DecodedText
        );
    }

    #[test]
    fn render_token_ids_formats_with_single_spaces() {
        assert_eq!(render_token_ids(&[464, 329, 262]), "464 329 262");
        assert_eq!(render_token_ids(&[]), "");
    }

    #[test]
    fn render_output_switches_between_decoded_text_and_token_ids() {
        let byte = ByteTokenizer::new(true);
        let pretokenized = PretokenizedTokenizer::new(1024, None, None, None, None);
        let hello_ids = byte.encode("hello", false, false);

        assert_eq!(
            render_output(
                &hello_ids,
                &byte,
                ResolvedGenerationOutputFormat::DecodedText,
                true,
            ),
            "hello"
        );
        assert_eq!(
            render_output(
                &[464, 329, 262],
                &pretokenized,
                ResolvedGenerationOutputFormat::TokenIds,
                true,
            ),
            "464 329 262"
        );
    }

    #[test]
    fn sanitize_display_text_escapes_non_printable_controls() {
        assert_eq!(sanitize_display_text("a\x08b\n"), "a\\x08b\n");
    }

    #[test]
    fn render_output_can_decode_past_eos_when_requested() {
        let byte = ByteTokenizer::new(true);
        let eos = byte.eos_id().expect("eos");
        let ids = vec![b'A' as u32, eos, b'B' as u32];
        assert_eq!(
            render_output(
                &ids,
                &byte,
                ResolvedGenerationOutputFormat::DecodedText,
                true
            ),
            "A"
        );
        assert_eq!(
            render_output(
                &ids,
                &byte,
                ResolvedGenerationOutputFormat::DecodedText,
                false
            ),
            "AB"
        );
    }
}
