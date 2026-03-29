use burn_dragon::api::{checkpoint, graph, language, multimodal, sudoku, vision};

fn main() {
    let options = checkpoint::bundle::BurnpackBundleExportOptions::default();

    let graph_export = graph::checkpoint::export_graph_checkpoint_to_burnpack;
    let language_export = language::checkpoint::export_language_checkpoint_to_burnpack;
    let multimodal_export = multimodal::checkpoint::export_multimodal_checkpoint_to_burnpack;
    let sudoku_export = sudoku::checkpoint::export_sudoku_checkpoint_to_burnpack;
    let vision_export = vision::checkpoint::export_vision_encoder_checkpoint_to_burnpack;

    let _ = (
        options,
        graph_export,
        language_export,
        multimodal_export,
        sudoku_export,
        vision_export,
    );
}
