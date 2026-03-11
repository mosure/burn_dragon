use std::fs;
use std::path::{Path, PathBuf};

use tempfile::tempdir;

use super::*;

fn write_config(dir: &Path, name: &str, contents: &str) -> PathBuf {
    let path = dir.join(name);
    let trimmed_lines: Vec<&str> = contents.lines().map(|line| line.trim_start()).collect();
    let mut formatted = trimmed_lines.join("\n");
    if formatted.starts_with('\n') {
        formatted = formatted.trim_start_matches('\n').to_string();
    }
    fs::write(&path, formatted).expect("write config");
    path
}

#[test]
fn load_merges_in_order() {
    let dir = tempdir().expect("tempdir");

    let base_contents = [
        "[dataset]",
        "cache_dir = \"data\"",
        "train_split_ratio = 0.8",
        "type = \"hugging_face\"",
        "repo_id = \"Ritvik19/Sudoku-Dataset\"",
        "train_files = [\"train_0.parquet\"]",
        "puzzle_field = \"puzzle\"",
        "solution_field = \"solution\"",
        "",
        "[training]",
        "batch_size = 8",
        "max_iters = 1000",
        "log_frequency = 50",
        "",
        "[training.rollout]",
        "steps = 4",
        "",
        "[training.policy]",
        "noise = 0.5",
        "",
        "[optimizer]",
        "learning_rate = 0.001",
        "weight_decay = 0.05",
    ]
    .join("\n");
    let base = write_config(dir.path(), "base.toml", &base_contents);

    let override_contents = [
        "[training]",
        "max_iters = 2000",
        "",
        "[optimizer]",
        "learning_rate = 0.0005",
    ]
    .join("\n");
    let override_cfg = write_config(dir.path(), "override.toml", &override_contents);

    let config = load_training_config(&[base, override_cfg]).expect("load config");

    assert_eq!(config.training.batch_size, 8);
    assert_eq!(config.training.max_iters, 2000);
    assert_eq!(config.training.rollout.steps, 4);
    assert!((config.optimizer.learning_rate - 0.0005).abs() < f64::EPSILON);
}
