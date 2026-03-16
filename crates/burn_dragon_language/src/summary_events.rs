use anyhow::{Result, anyhow};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use crate::BDHConfig;
use crate::tokenizer::Tokenizer;

pub fn resolve_summary_memory_write_triggers(
    model_config: &mut BDHConfig,
    tokenizer: &dyn Tokenizer,
) -> Result<()> {
    let Some(write_trigger_text) = model_config.summary_memory.write_trigger_text.as_ref() else {
        return Ok(());
    };
    let write_trigger_text = write_trigger_text.trim_end_matches('\0');
    if write_trigger_text.is_empty() {
        return Err(anyhow!(
            "model.summary_memory.write_trigger_text must not be empty when set"
        ));
    }
    let token_ids = tokenizer.encode(write_trigger_text, false, false);
    if token_ids.is_empty() {
        return Err(anyhow!(
            "model.summary_memory.write_trigger_text resolved to an empty token sequence"
        ));
    }
    model_config.summary_memory.write_trigger_token_ids = Some(token_ids);
    Ok(())
}

pub fn summary_event_mask_from_tokens(tokens: &[i64], trigger_token_ids: &[u32]) -> Vec<i64> {
    let mut mask = vec![0i64; tokens.len()];
    if tokens.is_empty() || trigger_token_ids.is_empty() || tokens.len() < trigger_token_ids.len() {
        return mask;
    }

    let trigger = trigger_token_ids
        .iter()
        .copied()
        .map(i64::from)
        .collect::<Vec<_>>();
    let trigger_len = trigger.len();
    for end in trigger_len - 1..tokens.len() {
        if tokens[end + 1 - trigger_len..=end] == trigger[..] {
            mask[end] = 1;
        }
    }
    mask
}

pub fn summary_event_mask_from_flat_batch(
    inputs: &[i64],
    batch_size: usize,
    block_size: usize,
    trigger_token_ids: &[u32],
) -> Vec<i64> {
    let mut mask = vec![0i64; inputs.len()];
    if batch_size == 0 || block_size == 0 || trigger_token_ids.is_empty() {
        return mask;
    }

    for batch_idx in 0..batch_size {
        let start = batch_idx * block_size;
        let end = start + block_size;
        let batch_mask = summary_event_mask_from_tokens(&inputs[start..end], trigger_token_ids);
        mask[start..end].copy_from_slice(&batch_mask);
    }

    mask
}

pub fn summary_event_mask_tensor<B: Backend>(
    inputs: &[i64],
    batch_size: usize,
    block_size: usize,
    trigger_token_ids: Option<&[u32]>,
    device: &B::Device,
) -> Option<Tensor<B, 2, Int>> {
    let trigger_token_ids = trigger_token_ids?;
    let mask =
        summary_event_mask_from_flat_batch(inputs, batch_size, block_size, trigger_token_ids);
    Some(Tensor::<B, 2, Int>::from_data(
        TensorData::new(mask, [batch_size, block_size]),
        device,
    ))
}
