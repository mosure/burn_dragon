use std::fmt::Display;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::bundle::BurnpackBundleExportReport;

#[derive(Debug, Clone)]
pub struct CheckpointExportReport {
    pub checkpoint_base: PathBuf,
    pub epoch: usize,
    pub run_dir: Option<PathBuf>,
    pub bundle: BurnpackBundleExportReport,
}

pub fn write_json_snapshot<T>(run_dir: &Path, file_name: &str, value: &T) -> Result<()>
where
    T: Serialize,
{
    fs::create_dir_all(run_dir)
        .with_context(|| format!("failed to create run directory {}", run_dir.display()))?;
    let payload =
        serde_json::to_string_pretty(value).context("failed to serialize run snapshot")?;
    let path = run_snapshot_path(run_dir, file_name);
    fs::write(&path, payload).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

pub fn load_json_snapshot<T>(run_dir: &Path, file_name: &str) -> Result<T>
where
    T: DeserializeOwned,
{
    let path = run_snapshot_path(run_dir, file_name);
    let contents =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&contents).with_context(|| format!("failed to parse {}", path.display()))
}

pub fn run_snapshot_path(run_dir: &Path, file_name: &str) -> PathBuf {
    run_dir.join(file_name)
}

pub fn resolve_checkpoint_run_dir(checkpoint: &Path) -> Option<PathBuf> {
    if checkpoint.is_dir() {
        return if checkpoint
            .file_name()
            .is_some_and(|name| name == "checkpoint")
        {
            checkpoint.parent().map(Path::to_path_buf)
        } else {
            Some(checkpoint.to_path_buf())
        };
    }

    checkpoint.parent().and_then(|parent| {
        if parent.file_name().is_some_and(|name| name == "checkpoint") {
            parent.parent().map(Path::to_path_buf)
        } else {
            Some(parent.to_path_buf())
        }
    })
}

pub fn resolve_checkpoint_base(path: &Path, epoch: Option<usize>) -> Result<(PathBuf, usize)> {
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
                    .unwrap_or_else(|| PathBuf::from("runs"));
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

pub fn checkpoint_bin_path(base: &Path) -> PathBuf {
    let mut path = base.to_path_buf();
    path.set_extension("bin");
    path
}

pub fn format_checkpoint_load_error(base: &Path, err: impl Display) -> String {
    let checkpoint_path = checkpoint_bin_path(base);
    let message = err.to_string();
    if message.contains("Metadata has a different Burn version") {
        format!(
            "failed to load checkpoint {}: {}. This checkpoint was recorded with a different Burn version; export/import helpers currently require a checkpoint recorded with the current Burn runtime/layout used by this repository.",
            checkpoint_path.display(),
            message
        )
    } else {
        format!(
            "failed to load checkpoint {}: {}",
            checkpoint_path.display(),
            message
        )
    }
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

fn ensure_checkpoint_exists(base: &Path) -> Result<()> {
    let candidate = checkpoint_bin_path(base);
    if candidate.is_file() {
        return Ok(());
    }

    Err(anyhow!("checkpoint file {} not found", candidate.display()))
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
    let epoch_part = stem.strip_prefix("model-")?;
    epoch_part.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::{
        checkpoint_bin_path, load_json_snapshot, resolve_checkpoint_base,
        resolve_checkpoint_run_dir, run_snapshot_path, write_json_snapshot,
    };
    use serde::{Deserialize, Serialize};
    use std::fs;
    use tempfile::tempdir;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct Snapshot {
        value: usize,
    }

    #[test]
    fn json_snapshot_roundtrip_works() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let snapshot = Snapshot { value: 7 };

        write_json_snapshot(&run_dir, "snapshot.json", &snapshot).expect("write snapshot");
        let loaded: Snapshot =
            load_json_snapshot(&run_dir, "snapshot.json").expect("load snapshot");

        assert_eq!(loaded, snapshot);
        assert!(run_snapshot_path(&run_dir, "snapshot.json").is_file());
    }

    #[test]
    fn resolve_checkpoint_helpers_handle_directory_layouts() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let checkpoint_dir = run_dir.join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");
        fs::write(checkpoint_dir.join("model-3.bin"), b"checkpoint").expect("write checkpoint");

        let (base, epoch) = resolve_checkpoint_base(&checkpoint_dir, None).expect("resolve base");
        assert_eq!(epoch, 3);
        assert_eq!(
            checkpoint_bin_path(&base),
            checkpoint_dir.join("model-3.bin")
        );
        assert_eq!(
            resolve_checkpoint_run_dir(&checkpoint_dir).expect("run dir"),
            run_dir
        );
        assert_eq!(
            resolve_checkpoint_run_dir(&checkpoint_dir.join("model-3.bin")).expect("run dir"),
            run_dir
        );
    }

    #[test]
    fn resolve_checkpoint_base_handles_parts_manifest_paths() {
        let dir = tempdir().expect("tempdir");
        let checkpoint_dir = dir.path().join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");
        fs::write(checkpoint_dir.join("model-2.bin"), b"checkpoint").expect("write checkpoint");

        let manifest = checkpoint_dir.join("model-2.bin.parts.json");
        let (base, epoch) = resolve_checkpoint_base(&manifest, None).expect("resolve manifest");
        assert_eq!(epoch, 2);
        assert_eq!(
            checkpoint_bin_path(&base),
            checkpoint_dir.join("model-2.bin")
        );
    }
}
