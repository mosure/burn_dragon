use std::path::PathBuf;

use anyhow::Result;

use super::TrainingConfig;
use crate::config::merge::load_merged_toml;

pub fn load_training_config(paths: &[PathBuf]) -> Result<TrainingConfig> {
    load_merged_toml(paths)
}
