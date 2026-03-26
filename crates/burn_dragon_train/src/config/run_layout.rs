use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RunLayoutConfig {
    #[serde(default)]
    pub base_dir: Option<PathBuf>,
    #[serde(default)]
    pub category: Option<PathBuf>,
    #[serde(default = "default_true")]
    pub mirror_config_path: bool,
}

impl Default for RunLayoutConfig {
    fn default() -> Self {
        Self {
            base_dir: None,
            category: None,
            mirror_config_path: default_true(),
        }
    }
}

const fn default_true() -> bool {
    true
}
