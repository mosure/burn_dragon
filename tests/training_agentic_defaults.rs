use std::fs;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn training_manifests_do_not_enable_burn_tui_by_default() {
    let repo_root = repo_root();
    for manifest in [
        "crates/burn_dragon_train/Cargo.toml",
        "crates/burn_dragon_language/Cargo.toml",
        "crates/burn_dragon_vision/Cargo.toml",
        "crates/burn_dragon_sudoku/Cargo.toml",
    ] {
        let content = fs::read_to_string(repo_root.join(manifest)).expect("read manifest");
        assert!(
            !content.contains("features = [\"tui\"]"),
            "{manifest} should not enable burn-train tui by default"
        );
    }
}

#[test]
fn training_schedules_disable_application_logger() {
    let repo_root = repo_root();
    for schedule in [
        "crates/burn_dragon_language/src/train/schedule.rs",
        "crates/burn_dragon_vision/src/train/pipeline/schedule.rs",
        "crates/burn_dragon_sudoku/src/train/schedule.rs",
    ] {
        let content = fs::read_to_string(repo_root.join(schedule)).expect("read schedule");
        assert!(
            content.contains(".with_application_logger(None)"),
            "{schedule} should explicitly disable the Burn application logger"
        );
    }
}
