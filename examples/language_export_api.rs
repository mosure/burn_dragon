use burn_dragon::api::{checkpoint, language};

fn main() {
    let options = checkpoint::bundle::BurnpackBundleExportOptions::default();
    let export_fn = language::checkpoint::export_language_checkpoint_to_burnpack;

    let _ = options;
    let _ = export_fn;
}
