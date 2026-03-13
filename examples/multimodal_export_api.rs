use burn_dragon::api::{checkpoint, multimodal};

fn main() {
    let options = checkpoint::bundle::BurnpackBundleExportOptions::default();
    let export_fn = multimodal::checkpoint::export_multimodal_checkpoint_to_burnpack;

    let _ = options;
    let _ = export_fn;
}
