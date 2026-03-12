use std::path::{Path, PathBuf};

const DEFAULT_F16_SUFFIX: &str = "_f16";

/// Preferred precision order when selecting burnpack weight files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BurnpackPrecisionPreference {
    /// Prefer `_f16.bpk` files first, then fallback to `.bpk`.
    PreferF16,
    /// Prefer `.bpk` files first, then fallback to `_f16.bpk`.
    PreferF32,
}

impl BurnpackPrecisionPreference {
    pub const fn prefer_f16(self) -> bool {
        matches!(self, Self::PreferF16)
    }
}

/// Burnpack path selection policy used by loaders and deployment helpers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BurnpackLoadPolicy {
    pub precision: BurnpackPrecisionPreference,
    pub f16_suffix: &'static str,
}

impl Default for BurnpackLoadPolicy {
    fn default() -> Self {
        Self {
            precision: BurnpackPrecisionPreference::PreferF32,
            f16_suffix: DEFAULT_F16_SUFFIX,
        }
    }
}

impl BurnpackLoadPolicy {
    pub const fn with_precision(self, precision: BurnpackPrecisionPreference) -> Self {
        Self { precision, ..self }
    }

    pub const fn with_f16_suffix(self, f16_suffix: &'static str) -> Self {
        Self { f16_suffix, ..self }
    }
}

pub fn candidate_burnpack_paths(path: &Path, policy: BurnpackLoadPolicy) -> Vec<PathBuf> {
    let default = burnpack_path(path, false, policy.f16_suffix);
    let f16 = burnpack_path(path, true, policy.f16_suffix);
    if f16 == default {
        vec![default]
    } else if policy.precision.prefer_f16() {
        vec![f16, default]
    } else {
        vec![default, f16]
    }
}

pub fn burnpack_path(path: &Path, use_f16: bool, f16_suffix: &str) -> PathBuf {
    let path = if path
        .extension()
        .map(|ext| ext.eq_ignore_ascii_case("bpk"))
        .unwrap_or(false)
    {
        path.to_path_buf()
    } else {
        path.with_extension("bpk")
    };

    if use_f16 {
        with_file_stem_suffix(&path, f16_suffix)
    } else {
        path
    }
}

fn with_file_stem_suffix(path: &Path, suffix: &str) -> PathBuf {
    let Some(stem) = path.file_stem() else {
        return path.to_path_buf();
    };
    let stem = stem.to_string_lossy();
    if stem.ends_with(suffix) {
        return path.to_path_buf();
    }

    let ext = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
    let mut file_name = format!("{stem}{suffix}");
    if !ext.is_empty() {
        file_name.push('.');
        file_name.push_str(ext);
    }
    path.with_file_name(file_name)
}

#[cfg(test)]
mod tests {
    use super::{
        BurnpackLoadPolicy, BurnpackPrecisionPreference, burnpack_path, candidate_burnpack_paths,
    };

    #[test]
    fn default_policy_prefers_f32() {
        let default = BurnpackLoadPolicy::default();
        assert_eq!(default.precision, BurnpackPrecisionPreference::PreferF32);
    }

    #[test]
    fn path_candidates_follow_precision_preference() {
        let path = std::path::Path::new("model");

        let f32_first = candidate_burnpack_paths(path, BurnpackLoadPolicy::default());
        assert_eq!(f32_first[0], burnpack_path(path, false, "_f16"));
        assert_eq!(f32_first[1], burnpack_path(path, true, "_f16"));

        let f16_first = candidate_burnpack_paths(
            path,
            BurnpackLoadPolicy::default().with_precision(BurnpackPrecisionPreference::PreferF16),
        );
        assert_eq!(f16_first[0], burnpack_path(path, true, "_f16"));
        assert_eq!(f16_first[1], burnpack_path(path, false, "_f16"));
    }
}
