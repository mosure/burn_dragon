use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

pub fn write_text_artifact(path: &Path, text: &str, label: &str) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("create {label} parent {}", parent.display()))?;
    }
    fs::write(path, text).with_context(|| format!("write {label} {}", path.display()))?;
    Ok(())
}

pub fn write_optional_report_artifacts<T: Serialize>(
    markdown_path: Option<&Path>,
    json_path: Option<&Path>,
    markdown: &str,
    report: &T,
) -> Result<()> {
    if let Some(path) = markdown_path {
        write_text_artifact(path, markdown, "markdown artifact")?;
    }
    if let Some(path) = json_path {
        let json = serde_json::to_string_pretty(report).context("serialize json artifact")?;
        write_text_artifact(path, &json, "json artifact")?;
    }
    Ok(())
}
