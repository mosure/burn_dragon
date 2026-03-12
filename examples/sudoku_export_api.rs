use burn_dragon::api::{checkpoint, sudoku};

fn main() {
    let options = checkpoint::bundle::BurnpackBundleExportOptions::default();
    let export_fn = sudoku::checkpoint::export_sudoku_checkpoint_to_burnpack;

    let _ = options;
    let _ = export_fn;
}
