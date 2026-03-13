use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};
use burn_dragon_stream::StreamSegment;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MultimodalStepMode {
    #[default]
    Observe,
    Refine,
    Predict,
}

#[derive(Clone)]
pub struct VisionLanguageTripletBatch<B: Backend> {
    pub vision_x: Tensor<B, 4>,
    pub query_q_tokens: Tensor<B, 2, Int>,
    pub query_q_mask: Option<Tensor<B, 2, Bool>>,
    pub target_y_tokens: Tensor<B, 2, Int>,
    pub target_y_mask: Option<Tensor<B, 2, Bool>>,
}

#[derive(Clone)]
pub struct VideoLanguageTripletBatch<B: Backend> {
    pub video_x: Tensor<B, 5>,
    pub query_q_tokens: Tensor<B, 2, Int>,
    pub query_q_mask: Option<Tensor<B, 2, Bool>>,
    pub target_y_tokens: Tensor<B, 2, Int>,
    pub target_y_mask: Option<Tensor<B, 2, Bool>>,
}

pub type VisionLanguageTripletSegment<B> = StreamSegment<VisionLanguageTripletBatch<B>>;
pub type VideoLanguageTripletSegment<B> = StreamSegment<VideoLanguageTripletBatch<B>>;
