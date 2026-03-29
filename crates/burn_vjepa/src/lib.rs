mod config;
mod model;
mod safetensors_io;
mod teacher;

pub use config::Vjepa2Config;
pub use model::{Vjepa2Model, Vjepa2ModelOutput, Vjepa2PredictorOutput};
pub use safetensors_io::PrecomputedClipFeatureStore;
#[cfg(feature = "train")]
pub use teacher::{CheckpointVisionDragonTeacher, VisionDragonTeacher};
pub use teacher::{ClipFeatureTeacher, NativeVjepa2Teacher};
