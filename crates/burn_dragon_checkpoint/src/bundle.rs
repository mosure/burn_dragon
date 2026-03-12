use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use burn::module::Module;
use burn::tensor::backend::Backend;

use crate::parts::{BurnpackPartsReport, ensure_burnpack_parts, save_model_to_burnpack};
use crate::policy::BurnpackLoadPolicy;
use crate::precision::{BurnpackFloatPrecision, convert_burnpack_precision};

#[derive(Debug, Clone)]
pub struct BurnpackBundleExportOptions {
    pub precision: BurnpackFloatPrecision,
    pub load_policy: BurnpackLoadPolicy,
    pub max_part_size_mib: Option<u64>,
    pub overwrite_parts: bool,
    pub keep_intermediate_f32: bool,
}

impl Default for BurnpackBundleExportOptions {
    fn default() -> Self {
        Self {
            precision: BurnpackFloatPrecision::F16,
            load_policy: BurnpackLoadPolicy::default(),
            max_part_size_mib: None,
            overwrite_parts: false,
            keep_intermediate_f32: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BurnpackBundleExportReport {
    pub burnpack_path: PathBuf,
    pub precision: BurnpackFloatPrecision,
    pub parts: Option<BurnpackPartsReport>,
    pub intermediate_f32_path: Option<PathBuf>,
}

pub fn export_model_to_burnpack_bundle<M, B>(
    model: &M,
    output_base: &Path,
    options: &BurnpackBundleExportOptions,
) -> Result<BurnpackBundleExportReport, String>
where
    M: Module<B>,
    B: Backend,
{
    ensure_parent_dir(output_base)?;
    let (burnpack_path, intermediate_f32_path) = match options.precision {
        BurnpackFloatPrecision::F32 => {
            let path = save_model_to_burnpack(model, output_base)?;
            (path, None)
        }
        BurnpackFloatPrecision::F16 => {
            let temp_base = temporary_f32_export_base(output_base)?;
            let temp_source = save_model_to_burnpack(model, &temp_base)?;
            let converted = convert_burnpack_precision(
                temp_source.as_path(),
                output_base,
                BurnpackFloatPrecision::F16,
                options.load_policy,
            )?;
            let keep_path = options.keep_intermediate_f32.then_some(temp_source.clone());
            if !options.keep_intermediate_f32 && temp_source.exists() {
                fs::remove_file(&temp_source).map_err(|err| {
                    format!(
                        "failed to remove intermediate burnpack {}: {err}",
                        temp_source.display()
                    )
                })?;
            }
            (converted, keep_path)
        }
    };

    let parts = if let Some(max_part_size_mib) = options.max_part_size_mib {
        ensure_burnpack_parts(&burnpack_path, max_part_size_mib, options.overwrite_parts)?
    } else {
        None
    };

    Ok(BurnpackBundleExportReport {
        burnpack_path,
        precision: options.precision,
        parts,
        intermediate_f32_path,
    })
}

fn ensure_parent_dir(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create burnpack output directory {}: {err}",
                parent.display()
            )
        })?;
    }
    Ok(())
}

fn temporary_f32_export_base(output_base: &Path) -> Result<PathBuf, String> {
    let parent = output_base
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = output_base
        .file_stem()
        .and_then(|value| value.to_str())
        .or_else(|| output_base.file_name().and_then(|value| value.to_str()))
        .unwrap_or("model");
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| format!("system clock error while creating temporary export path: {err}"))?
        .as_millis();
    Ok(parent.join(format!(".{stem}.export-f32-{nonce}")))
}

#[cfg(test)]
mod tests {
    use super::{BurnpackBundleExportOptions, export_model_to_burnpack_bundle};
    use crate::parts::{burnpack_parts_manifest_path, manifest_is_complete};
    use crate::precision::BurnpackFloatPrecision;
    use burn::module::Module;
    use burn::nn::{Linear, LinearConfig};
    use burn::tensor::backend::Backend;
    use burn_ndarray::NdArray;
    use tempfile::tempdir;

    type TestBackend = NdArray<f32>;

    #[derive(Module, Debug)]
    struct TinyModel<B: Backend> {
        linear: Linear<B>,
    }

    impl<B: Backend> TinyModel<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                linear: LinearConfig::new(4, 3).init(device),
            }
        }
    }

    #[test]
    fn exports_f16_bundle_with_parts_and_removes_intermediate() {
        let device = <TestBackend as Backend>::Device::default();
        TestBackend::seed(&device, 1337);
        let model = TinyModel::<TestBackend>::new(&device);
        let dir = tempdir().expect("tempdir");
        let output_base = dir.path().join("tiny");

        let report = export_model_to_burnpack_bundle(
            &model,
            &output_base,
            &BurnpackBundleExportOptions {
                precision: BurnpackFloatPrecision::F16,
                max_part_size_mib: Some(1),
                overwrite_parts: true,
                keep_intermediate_f32: false,
                ..BurnpackBundleExportOptions::default()
            },
        )
        .expect("export bundle");

        assert!(report.burnpack_path.is_file(), "burnpack should be written");
        assert!(
            report
                .burnpack_path
                .file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|name| name.ends_with("_f16.bpk"))
        );
        assert!(report.intermediate_f32_path.is_none());
        let manifest_path = burnpack_parts_manifest_path(&report.burnpack_path);
        assert!(
            manifest_is_complete(&manifest_path).expect("manifest status"),
            "multipart manifest should be complete"
        );
    }
}
