use std::fmt;

use burn::module::{Content, ModuleDisplay, ModuleDisplayDefault};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionArtifactOutputMode {
    #[default]
    Images,
    Avi,
    Mp4,
}

impl fmt::Display for VisionArtifactOutputMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Images => write!(f, "images"),
            Self::Avi => write!(f, "avi"),
            Self::Mp4 => write!(f, "mp4"),
        }
    }
}

impl ModuleDisplayDefault for VisionArtifactOutputMode {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for VisionArtifactOutputMode {}
