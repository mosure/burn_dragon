use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::Duration;

use crate::parts::{
    burnpack_parts_manifest_path, manifest_is_complete, part_matches_cache, read_parts_manifest,
    resolve_part_entry_path,
};
use crate::policy::{BurnpackLoadPolicy, candidate_burnpack_paths};

const DOWNLOAD_ATTEMPTS: u32 = 4;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);
const READ_TIMEOUT: Duration = Duration::from_secs(60);
const WRITE_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Clone)]
pub struct BurnpackBootstrapConfig {
    pub cache_root: Option<PathBuf>,
    pub prefer_parts: bool,
    pub verify_checksums: bool,
    pub load_policy: BurnpackLoadPolicy,
}

impl Default for BurnpackBootstrapConfig {
    fn default() -> Self {
        Self {
            cache_root: None,
            prefer_parts: true,
            verify_checksums: true,
            load_policy: BurnpackLoadPolicy::default(),
        }
    }
}

pub fn default_cache_root() -> Result<PathBuf, String> {
    if let Some(explicit) = std::env::var_os("BURN_DRAGON_CACHE_DIR") {
        return Ok(PathBuf::from(explicit));
    }

    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| "failed to resolve user home directory for burn_dragon cache".to_string())?;
    Ok(home.join(".cache").join("burn_dragon").join("models"))
}

pub fn candidate_burnpack_urls(base_url: &str, policy: BurnpackLoadPolicy) -> Vec<String> {
    let f32 = burnpack_url(base_url, false, policy.f16_suffix);
    let f16 = burnpack_url(base_url, true, policy.f16_suffix);
    if f16 == f32 {
        vec![f32]
    } else if policy.precision.prefer_f16() {
        vec![f16, f32]
    } else {
        vec![f32, f16]
    }
}

pub fn resolve_or_bootstrap_burnpack(
    local_base: &Path,
    remote_base_url: &str,
    config: &BurnpackBootstrapConfig,
) -> Result<PathBuf, String> {
    resolve_or_bootstrap_burnpack_with_progress(local_base, remote_base_url, config, |_| {})
}

pub fn resolve_or_bootstrap_burnpack_with_progress<F>(
    local_base: &Path,
    remote_base_url: &str,
    config: &BurnpackBootstrapConfig,
    mut progress: F,
) -> Result<PathBuf, String>
where
    F: FnMut(String),
{
    let cache_root = match config.cache_root.as_ref() {
        Some(root) => root.clone(),
        None => default_cache_root()?,
    };
    fs::create_dir_all(&cache_root).map_err(|err| {
        format!(
            "failed to create burn_dragon cache {}: {err}",
            cache_root.display()
        )
    })?;

    let local_base = resolve_local_base(local_base, &cache_root);
    let local_candidates = candidate_burnpack_paths(local_base.as_path(), config.load_policy);
    let remote_candidates = candidate_burnpack_urls(remote_base_url, config.load_policy);
    let mut last_error = None;

    for (local_candidate, remote_candidate) in local_candidates.iter().zip(remote_candidates.iter())
    {
        let manifest_path = burnpack_parts_manifest_path(local_candidate);
        if manifest_is_complete(&manifest_path).unwrap_or(false) {
            progress(format!(
                "using cached multipart burnpack {}",
                manifest_path.display()
            ));
            return Ok(local_candidate.clone());
        }
        if local_candidate.is_file() {
            progress(format!(
                "using cached monolithic burnpack {}",
                local_candidate.display()
            ));
            return Ok(local_candidate.clone());
        }

        if config.prefer_parts {
            let manifest_url = format!("{remote_candidate}.parts.json");
            progress(format!("downloading burnpack manifest {manifest_url}"));
            match download_optional_bytes(manifest_url.as_str()) {
                Err(_err) => {}
                Ok(None) => {}
                Ok(Some(manifest_bytes)) => {
                    if let Some(parent) = manifest_path.parent() {
                        fs::create_dir_all(parent).map_err(|err| {
                            format!(
                                "failed to create burnpack manifest directory {}: {err}",
                                parent.display()
                            )
                        })?;
                    }
                    write_bytes_atomically(manifest_path.as_path(), manifest_bytes.as_slice())?;
                    let manifest = read_parts_manifest(&manifest_path)?;
                    if !manifest.parts.is_empty() {
                        let mut manifest_ok = true;
                        for (index, part) in manifest.parts.iter().enumerate() {
                            let local_part_path =
                                resolve_part_entry_path(&manifest_path, &part.path)?;
                            if part_matches_cache(local_part_path.as_path(), part)? {
                                progress(format!(
                                    "cached burnpack part {}/{}",
                                    index + 1,
                                    manifest.parts.len()
                                ));
                                continue;
                            }
                            let part_url = resolve_manifest_entry_url(
                                manifest_url.as_str(),
                                part.path.as_str(),
                            );
                            progress(format!(
                                "downloading burnpack part {}/{}",
                                index + 1,
                                manifest.parts.len()
                            ));
                            if ensure_file_cached(local_part_path.as_path(), part_url.as_str())
                                .is_err()
                            {
                                manifest_ok = false;
                                break;
                            }
                            if config.verify_checksums
                                && !part_matches_cache(local_part_path.as_path(), part)?
                            {
                                manifest_ok = false;
                                break;
                            }
                        }

                        if manifest_ok && manifest_is_complete(&manifest_path).unwrap_or(false) {
                            progress(format!(
                                "downloaded burnpack parts ({}/{})",
                                manifest.parts.len(),
                                manifest.parts.len()
                            ));
                            return Ok(local_candidate.clone());
                        }
                    }
                }
            }
        }

        progress(format!(
            "parts unavailable; downloading monolithic burnpack {remote_candidate}"
        ));
        match ensure_file_cached(local_candidate.as_path(), remote_candidate.as_str()) {
            Ok(()) => return Ok(local_candidate.clone()),
            Err(err) => {
                last_error = Some(err);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| {
        format!(
            "failed to resolve burnpack weights for {}",
            local_base.display()
        )
    }))
}

fn resolve_local_base(local_base: &Path, cache_root: &Path) -> PathBuf {
    if local_base.is_absolute() {
        local_base.to_path_buf()
    } else {
        cache_root.join(local_base)
    }
}

fn burnpack_url(base_url: &str, use_f16: bool, f16_suffix: &str) -> String {
    let base = if base_url.ends_with(".bpk") {
        base_url.to_string()
    } else {
        format!("{base_url}.bpk")
    };
    if use_f16 {
        with_url_stem_suffix(base.as_str(), f16_suffix)
    } else {
        base
    }
}

fn with_url_stem_suffix(url: &str, suffix: &str) -> String {
    let Some(last_slash) = url.rfind('/') else {
        return with_file_like_suffix(url, suffix);
    };
    let (prefix, tail) = url.split_at(last_slash + 1);
    format!("{prefix}{}", with_file_like_suffix(tail, suffix))
}

fn with_file_like_suffix(path: &str, suffix: &str) -> String {
    match path.rsplit_once('.') {
        Some((stem, ext)) if !stem.ends_with(suffix) => format!("{stem}{suffix}.{ext}"),
        Some((_stem, _ext)) => path.to_string(),
        None if !path.ends_with(suffix) => format!("{path}{suffix}"),
        None => path.to_string(),
    }
}

fn resolve_manifest_entry_url(manifest_url: &str, entry_path: &str) -> String {
    if entry_path.starts_with("http://") || entry_path.starts_with("https://") {
        return entry_path.to_string();
    }
    let base = manifest_url
        .rsplit_once('/')
        .map(|(left, _)| left)
        .unwrap_or("");
    if base.is_empty() {
        entry_path.trim_start_matches('/').to_string()
    } else {
        format!(
            "{}/{}",
            base.trim_end_matches('/'),
            entry_path.trim_start_matches('/')
        )
    }
}

fn ensure_file_cached(path: &Path, url: &str) -> Result<(), String> {
    if path.is_file() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create cache directory {}: {err}",
                parent.display()
            )
        })?;
    }
    let bytes = download_required_bytes(url)?;
    write_bytes_atomically(path, bytes.as_slice())
}

fn download_optional_bytes(url: &str) -> Result<Option<Vec<u8>>, String> {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(CONNECT_TIMEOUT)
        .timeout_read(READ_TIMEOUT)
        .timeout_write(WRITE_TIMEOUT)
        .build();
    match agent.get(url).call() {
        Ok(response) => {
            let mut reader = response.into_reader();
            let mut bytes = Vec::new();
            reader
                .read_to_end(&mut bytes)
                .map_err(|err| format!("failed reading {url}: {err}"))?;
            Ok(Some(bytes))
        }
        Err(ureq::Error::Status(code, _)) if code == 403 || code == 404 => Ok(None),
        Err(err) => Err(format!("failed to download {url}: {err}")),
    }
}

fn download_required_bytes(url: &str) -> Result<Vec<u8>, String> {
    let mut attempt = 0;
    let mut backoff = Duration::from_millis(250);
    loop {
        attempt += 1;
        match download_optional_bytes(url) {
            Ok(Some(bytes)) => return Ok(bytes),
            Ok(None) => return Err(format!("remote burnpack resource not found: {url}")),
            Err(err) => {
                if attempt >= DOWNLOAD_ATTEMPTS {
                    return Err(format!(
                        "failed to download {url} after {attempt} attempts: {err}"
                    ));
                }
            }
        }
        sleep(backoff);
        backoff = backoff.saturating_mul(2);
    }
}

fn write_bytes_atomically(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let temp_path = path.with_extension(format!(
        "{}.partial",
        path.extension().and_then(|ext| ext.to_str()).unwrap_or("")
    ));
    {
        let mut file = fs::File::create(&temp_path).map_err(|err| {
            format!(
                "failed to create temporary file {}: {err}",
                temp_path.display()
            )
        })?;
        file.write_all(bytes).map_err(|err| {
            format!(
                "failed to write temporary file {}: {err}",
                temp_path.display()
            )
        })?;
        file.flush().map_err(|err| {
            format!(
                "failed to flush temporary file {}: {err}",
                temp_path.display()
            )
        })?;
    }
    fs::rename(&temp_path, path).map_err(|err| {
        format!(
            "failed to move temporary file {} into place {}: {err}",
            temp_path.display(),
            path.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::{
        BurnpackBootstrapConfig, candidate_burnpack_urls, default_cache_root,
        resolve_or_bootstrap_burnpack,
    };
    use crate::parts::{BurnpackPartEntry, BurnpackPartsManifest, burnpack_parts_manifest_path};
    use crate::policy::{BurnpackLoadPolicy, BurnpackPrecisionPreference};
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn candidate_urls_follow_precision_preference() {
        let urls = candidate_burnpack_urls(
            "https://example.com/model",
            BurnpackLoadPolicy::default().with_precision(BurnpackPrecisionPreference::PreferF16),
        );
        assert_eq!(urls[0], "https://example.com/model_f16.bpk");
        assert_eq!(urls[1], "https://example.com/model.bpk");
    }

    #[test]
    fn default_cache_root_uses_explicit_env_override() {
        let root = tempdir().expect("tempdir");
        unsafe {
            std::env::set_var("BURN_DRAGON_CACHE_DIR", root.path());
        }
        let resolved = default_cache_root().expect("cache root");
        assert_eq!(resolved, root.path());
        unsafe {
            std::env::remove_var("BURN_DRAGON_CACHE_DIR");
        }
    }

    #[test]
    fn resolve_bootstrap_prefers_complete_cached_parts() {
        let cache = tempdir().expect("tempdir");
        let local_base = cache.path().join("language").join("model");
        let candidate = local_base.with_extension("bpk");
        let manifest_path = burnpack_parts_manifest_path(&candidate);
        fs::create_dir_all(manifest_path.parent().expect("manifest parent")).expect("mkdir");
        let part_path = candidate.with_file_name("model.bpk.part-00000.bpk");
        let part_bytes = vec![1u8, 2, 3, 4];
        fs::write(&part_path, &part_bytes).expect("write part");
        let manifest = BurnpackPartsManifest {
            version: 1,
            source_file: "model.bpk".to_string(),
            source_modified_unix_ms: 0,
            total_bytes: part_bytes.len() as u64,
            max_part_bytes: part_bytes.len() as u64,
            parts: vec![BurnpackPartEntry {
                path: "model.bpk.part-00000.bpk".to_string(),
                bytes: part_bytes.len() as u64,
                sha256: crate::parts::sha256_bytes(part_bytes.as_slice()),
                tensors: 1,
            }],
        };
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("manifest json"),
        )
        .expect("write manifest");

        let resolved = resolve_or_bootstrap_burnpack(
            local_base.as_path(),
            "https://example.invalid/model",
            &BurnpackBootstrapConfig {
                cache_root: Some(cache.path().to_path_buf()),
                ..BurnpackBootstrapConfig::default()
            },
        )
        .expect("resolve from cache");
        assert_eq!(resolved, candidate);
    }
}
