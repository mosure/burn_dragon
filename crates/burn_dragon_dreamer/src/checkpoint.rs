use anyhow::{Context, Result};
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::Backend;
use std::path::{Path, PathBuf};

pub(crate) fn save_module_checkpoint<B, M>(model: &M, checkpoint_base: &Path) -> Result<()>
where
    B: Backend,
    M: Module<B> + Clone,
{
    BinFileRecorder::<FullPrecisionSettings>::new()
        .record(model.clone().into_record(), checkpoint_base.to_path_buf())
        .with_context(|| format!("write checkpoint {}", checkpoint_base.display()))
}

pub(crate) fn load_module_checkpoint<B, M>(
    model: M,
    checkpoint_base: &Path,
    device: &B::Device,
) -> Result<M>
where
    B: Backend,
    M: Module<B>,
{
    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load::<M::Record>(checkpoint_base.to_path_buf(), device)
        .with_context(|| format!("load checkpoint {}", checkpoint_base.display()))?;
    Ok(model.load_record(record))
}

pub(crate) fn checkpoint_base(run_dir: &Path, stem: &str) -> PathBuf {
    run_dir.join(stem)
}
