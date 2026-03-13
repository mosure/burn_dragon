use burn_dragon::api::stream;

fn main() {
    let window = stream::window::TbpttWindow::new(8, 3);
    let boundary = stream::boundary::StreamBoundary::ResetEpisode;
    let _segment = stream::segment::StreamSegment {
        payload: vec![1_u32, 2, 3],
        stream: stream::segment::StreamStepMetadata {
            sample_id: stream::ids::StreamSampleId {
                source_id: 1,
                episode_id: 2,
                segment_id: 3,
            },
            boundary,
            step_index: 0,
            absolute_time: 0,
        },
    };

    assert_eq!(window.detach_prefix_steps(), 5);
}
