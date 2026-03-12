use burn_dragon::api::{checkpoint, vision};

fn main() {
    let options = checkpoint::bundle::BurnpackBundleExportOptions::default();
    let export_fn = vision::checkpoint::export_vision_encoder_checkpoint_to_burnpack;

    let _ = options;
    let _ = export_fn;
}
