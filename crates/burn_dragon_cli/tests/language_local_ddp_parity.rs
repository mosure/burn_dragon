#![cfg(all(feature = "language-train", feature = "language-ddp"))]

use std::fs;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use burn::tensor::{Int, Tensor, TensorData};
use burn_dragon_language::checkpoint::load_language_core_from_checkpoint;
use burn_dragon_train::train::pipeline::resolve_latest_run_dir_in;
use burn_ndarray::NdArray;
use tempfile::tempdir;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

fn toml_escape_path(path: &Path) -> String {
    path.display()
        .to_string()
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
}

fn write_tiny_shakespeare_cache(cache_dir: &Path) {
    fs::create_dir_all(cache_dir).expect("cache dir");
    fs::write(
        cache_dir.join("tinyshakespeare.txt"),
        b"Once more unto the breach, dear friends, once more.\n".repeat(4),
    )
    .expect("write tiny shakespeare");
}

fn write_smoke_config(
    path: &Path,
    cache_dir: &Path,
    max_iters: usize,
    epochs: Option<usize>,
    resume_run_dir: Option<&Path>,
    resume_checkpoint_epoch: Option<usize>,
) {
    let mut contents = format!(
        r#"[dataset]
cache_dir = "{}"
train_split_ratio = 0.9
type = "shakespeare"

[dataset.tokenizer]
type = "char"
include_unknown = true
vocab_path = "vocab.json"

[training]
block_size = 8
batch_size = 4
seed = 1337
gradient_accumulation_steps = 1
max_iters = {max_iters}
log_frequency = 1
"#,
        toml_escape_path(cache_dir)
    );
    if let Some(epochs) = epochs {
        contents.push_str(&format!("epochs = {epochs}\n"));
    }
    if let Some(resume_run_dir) = resume_run_dir {
        contents.push_str(&format!(
            "resume_run_dir = \"{}\"\n",
            toml_escape_path(resume_run_dir)
        ));
    }
    if let Some(epoch) = resume_checkpoint_epoch {
        contents.push_str(&format!("resume_checkpoint_epoch = {epoch}\n"));
    }
    contents.push_str(
        r#"

[optimizer]
learning_rate = 0.001
weight_decay = 0.0

[optimizer.lr_schedule]
type = "constant"

[generation]
prompt = "To be or "
temperature = 1.0
top_k = 1

[model]
n_layer = 1
n_embd = 8
n_head = 1
mlp_internal_dim_multiplier = 1
dropout = 0.0
"#,
    );
    fs::write(path, contents).expect("write smoke config");
}

fn run_train(cwd: &Path, run_root: &Path, args: &[String]) -> Output {
    let output = Command::new(env!("CARGO_BIN_EXE_language_train"))
        .current_dir(cwd)
        .env("BURN_DRAGON_RUN_ROOT", run_root)
        .args(args)
        .output()
        .expect("spawn language_train binary");

    assert!(
        output.status.success(),
        "train command failed: args={args:?}\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    output
}

fn latest_run_dir(run_root: &Path) -> PathBuf {
    resolve_latest_run_dir_in(run_root).expect("latest run dir")
}

fn parse_single_valid_loss(stdout: &str) -> f64 {
    let line = stdout
        .lines()
        .find(|line| line.contains("| Valid | Loss"))
        .expect("valid loss row in single-rank output");
    let columns = line
        .split('|')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    columns[2].parse::<f64>().expect("single valid loss")
}

fn parse_rank_zero_valid_loss(run_dir: &Path) -> f64 {
    let stdout = fs::read_to_string(run_dir.join("rank-0.stdout.log")).expect("rank0 stdout");
    let line = stdout
        .lines()
        .find(|line| line.contains("valid epoch=") && line.contains("loss="))
        .expect("rank0 valid loss line");
    let start = line.find("loss=").expect("loss marker") + "loss=".len();
    let value = &line[start..];
    let value = value.split_whitespace().next().expect("loss token");
    value.parse::<f64>().expect("rank0 valid loss")
}

fn reserve_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind free port");
    let port = listener.local_addr().expect("local addr").port();
    drop(listener);
    port
}

fn load_checkpoint_logits(run_dir: &Path) -> Vec<f32> {
    type InferenceBackend = NdArray<f32>;

    let device = <InferenceBackend as burn::tensor::backend::Backend>::Device::default();
    let model = load_language_core_from_checkpoint::<InferenceBackend>(
        &run_dir.join("checkpoint"),
        None,
        &[],
        "cpu",
        &device,
    )
    .expect("load checkpoint");
    let logits = model.forward(Tensor::<InferenceBackend, 2, Int>::from_data(
        TensorData::new(vec![0i64, 1, 2, 3, 4, 5, 6, 7], [1, 8]),
        &device,
    ));
    logits
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("logits vec")
}

#[test]
fn language_local_ddp_valid_loss_matches_single_rank_smoke() {
    let temp = tempdir().expect("tempdir");
    let cwd = workspace_root();
    let cache_dir = temp.path().join("cache");
    write_tiny_shakespeare_cache(&cache_dir);
    let config_path = temp.path().join("language_local_ddp_smoke.toml");
    write_smoke_config(&config_path, &cache_dir, 1, None, None, None);

    let single_run_root = temp.path().join("single-runs");
    let single_args = vec![
        "language".to_string(),
        "--backend".to_string(),
        "ndarray".to_string(),
        "--config".to_string(),
        config_path.display().to_string(),
    ];
    let single_output = run_train(&cwd, &single_run_root, &single_args);
    let single_stdout = String::from_utf8_lossy(&single_output.stdout);
    let single_valid_loss = parse_single_valid_loss(&single_stdout);
    let single_run_dir = latest_run_dir(&single_run_root);

    let ddp_run_root = temp.path().join("ddp-runs");
    let port = reserve_port();
    let ddp_args = vec![
        "language-local-ddp".to_string(),
        "--backend".to_string(),
        "ndarray".to_string(),
        "--config".to_string(),
        config_path.display().to_string(),
        "--world-size".to_string(),
        "2".to_string(),
        "--orchestrator-port".to_string(),
        port.to_string(),
    ];
    run_train(&cwd, &ddp_run_root, &ddp_args);
    let ddp_run_dir = latest_run_dir(&ddp_run_root);
    let ddp_valid_loss = parse_rank_zero_valid_loss(&ddp_run_dir);
    let single_logits = load_checkpoint_logits(&single_run_dir);
    let ddp_logits = load_checkpoint_logits(&ddp_run_dir);
    assert_eq!(single_logits.len(), ddp_logits.len(), "logit shapes differ");
    let mean_abs_diff = single_logits
        .iter()
        .zip(ddp_logits.iter())
        .map(|(single, ddp)| (single - ddp).abs() as f64)
        .sum::<f64>()
        / single_logits.len().max(1) as f64;

    assert!(
        single_run_dir
            .join("checkpoint")
            .join("model-1.bin")
            .is_file(),
        "expected single-rank checkpoint in {}",
        single_run_dir.display()
    );
    assert!(
        ddp_run_dir.join("checkpoint").join("model-1.bin").is_file(),
        "expected rank-0 model checkpoint in {}",
        ddp_run_dir.display()
    );
    assert!(
        (single_valid_loss - ddp_valid_loss).abs() <= 0.05,
        "single-rank valid loss {single_valid_loss:.4} and local-ddp valid loss {ddp_valid_loss:.4} drifted too far"
    );
    assert!(
        mean_abs_diff <= 0.05,
        "single-rank and local-ddp checkpoint logits drifted too far: mean_abs_diff={mean_abs_diff:.6}"
    );
}

#[test]
fn language_local_ddp_can_resume_existing_run_dir() {
    let temp = tempdir().expect("tempdir");
    let cwd = workspace_root();
    let cache_dir = temp.path().join("cache");
    write_tiny_shakespeare_cache(&cache_dir);

    let initial_config = temp.path().join("language_local_ddp_initial.toml");
    write_smoke_config(&initial_config, &cache_dir, 1, Some(1), None, None);

    let ddp_run_root = temp.path().join("ddp-runs");
    let initial_args = vec![
        "language-local-ddp".to_string(),
        "--backend".to_string(),
        "ndarray".to_string(),
        "--config".to_string(),
        initial_config.display().to_string(),
        "--world-size".to_string(),
        "2".to_string(),
        "--orchestrator-port".to_string(),
        reserve_port().to_string(),
    ];
    run_train(&cwd, &ddp_run_root, &initial_args);
    let run_dir = latest_run_dir(&ddp_run_root);
    assert!(run_dir.join("checkpoint").join("model-1.bin").is_file());

    let resumed_config = temp.path().join("language_local_ddp_resume.toml");
    write_smoke_config(
        &resumed_config,
        &cache_dir,
        1,
        Some(2),
        Some(&run_dir),
        Some(1),
    );
    let resumed_args = vec![
        "language-local-ddp".to_string(),
        "--backend".to_string(),
        "ndarray".to_string(),
        "--config".to_string(),
        resumed_config.display().to_string(),
        "--world-size".to_string(),
        "2".to_string(),
        "--orchestrator-port".to_string(),
        reserve_port().to_string(),
    ];
    run_train(&cwd, &ddp_run_root, &resumed_args);

    let resumed_latest = latest_run_dir(&ddp_run_root);
    assert_eq!(
        resumed_latest, run_dir,
        "resumed local DDP should reuse the existing run directory"
    );
    assert!(run_dir.join("checkpoint").join("model-2.bin").is_file());

    let rank_zero_stdout =
        fs::read_to_string(run_dir.join("rank-0.stdout.log")).expect("rank0 stdout");
    assert!(
        rank_zero_stdout.contains("Executing process-group DDP epoch 2"),
        "expected resumed rank-0 log to include epoch 2, got:\n{rank_zero_stdout}"
    );

    let logits = load_checkpoint_logits(&run_dir);
    assert!(!logits.is_empty());
    assert!(logits.iter().all(|value| value.is_finite()));
}
