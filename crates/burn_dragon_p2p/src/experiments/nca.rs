use anyhow::Result;
use burn::tensor::backend::AutodiffBackend;

use crate::config::{DragonExperimentKind, DragonNativeAuthBundle, DragonNativePeerConfig};
use crate::experiments::common::{PreparedNativePeer, prepare_language_peer_for_backend};

pub fn prepare_nca_peer_for_backend<B>(
    native: &DragonNativePeerConfig,
    backend_label: &str,
    device: B::Device,
    auth_bundle: Option<&DragonNativeAuthBundle>,
) -> Result<PreparedNativePeer<B>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    prepare_language_peer_for_backend::<B>(
        native,
        DragonExperimentKind::NcaPrepretraining,
        backend_label,
        device,
        auth_bundle,
    )
}
