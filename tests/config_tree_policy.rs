use std::path::PathBuf;
use std::process::Command;

#[test]
fn active_config_tree_matches_curated_policy() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python3")
        .arg("tools/config_tree.py")
        .current_dir(&repo_root)
        .output()
        .expect("run config tree audit");

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("config tree audit failed\nstdout:\n{stdout}\nstderr:\n{stderr}");
    }
}
