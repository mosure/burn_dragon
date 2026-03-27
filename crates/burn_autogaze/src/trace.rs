use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FixationPoint {
    pub x: f32,
    pub y: f32,
    pub scale: f32,
    pub confidence: f32,
}

impl FixationPoint {
    pub fn new(x: f32, y: f32, scale: f32, confidence: f32) -> Self {
        Self {
            x: x.clamp(0.0, 1.0),
            y: y.clamp(0.0, 1.0),
            scale: scale.clamp(0.01, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FixationSet {
    pub points: Vec<FixationPoint>,
    pub stop_probability: f32,
}

impl FixationSet {
    pub fn new(mut points: Vec<FixationPoint>, stop_probability: f32, k: usize) -> Self {
        points.truncate(k.max(1));
        while points.len() < k.max(1) {
            points.push(FixationPoint::new(0.5, 0.5, 0.25, 0.0));
        }
        Self {
            points,
            stop_probability: stop_probability.clamp(0.0, 1.0),
        }
    }

    pub fn top_point(&self) -> FixationPoint {
        self.points
            .first()
            .copied()
            .unwrap_or_else(|| FixationPoint::new(0.5, 0.5, 0.25, 0.0))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FrameFixationTrace {
    pub frames: Vec<FixationSet>,
}

impl FrameFixationTrace {
    pub fn new(frames: Vec<FixationSet>) -> Self {
        Self { frames }
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}
