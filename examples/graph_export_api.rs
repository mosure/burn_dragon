use burn_dragon::api::{checkpoint, graph};

fn main() {
    let options = checkpoint::bundle::BurnpackBundleExportOptions::default();
    let export_fn = graph::checkpoint::export_graph_checkpoint_to_burnpack;

    let _ = options;
    let _ = export_fn;
}
