use std::fmt::Write as _;

use serde::Serialize;

pub const VISION_ARTIFACT_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct VisionArtifactHeader {
    pub artifact_kind: &'static str,
    pub schema_version: u32,
}

impl VisionArtifactHeader {
    pub const fn new(artifact_kind: &'static str) -> Self {
        Self {
            artifact_kind,
            schema_version: VISION_ARTIFACT_SCHEMA_VERSION,
        }
    }
}

pub fn push_vision_artifact_markdown_prelude(
    out: &mut String,
    title: &str,
    artifact: &VisionArtifactHeader,
) {
    let _ = writeln!(out, "# {title}");
    let _ = writeln!(out);
    let _ = writeln!(out, "- artifact kind: `{}`", artifact.artifact_kind);
    let _ = writeln!(out, "- schema version: `{}`", artifact.schema_version);
}
