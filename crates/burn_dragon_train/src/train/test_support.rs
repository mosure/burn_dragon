use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub(crate) fn create_stub_ffmpeg(bin_dir: &Path) -> io::Result<PathBuf> {
    fs::create_dir_all(bin_dir)?;
    #[cfg(windows)]
    let script_path = bin_dir.join("ffmpeg.cmd");
    #[cfg(not(windows))]
    let script_path = bin_dir.join("ffmpeg");
    #[cfg(windows)]
    let script = r#"@echo off
set OUT=
for %%A in (%*) do set OUT=%%A
type nul > "%OUT%"
exit /b 0
"#;
    #[cfg(not(windows))]
    let script = r#"#!/bin/sh
out=""
for arg in "$@"; do out="$arg"; done
: > "$out"
"#;
    fs::write(&script_path, script)?;
    #[cfg(not(windows))]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&script_path)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&script_path, perms)?;
    }
    Ok(script_path)
}
