pub mod vision;

pub use vision::*;
pub use burn_dragon_train::VisionTeacherVariant;

#[cfg(all(test, feature = "train"))]
mod tests;
