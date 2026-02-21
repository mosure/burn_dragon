pub mod vision;

pub use burn_dragon_train::VisionTeacherVariant;
pub use vision::*;

#[cfg(all(test, feature = "train"))]
mod tests;
