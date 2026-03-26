//! Intermediate video training surface.
//!
//! This module supports the current `video_lejepa` path, but it is intentionally not the final
//! public video architecture. The long-term target remains the shared-core TRM design in
//! `docs/video/video_shared_core_trm_spec.md`.

pub(crate) mod dataset;
pub(crate) mod dynamics;
pub(crate) mod lejepa;
pub mod profile;
pub(crate) mod vjepa21;

pub(crate) use lejepa::{VisionVideoLejepaLosses, VisionVideoLejepaModel};
pub(crate) use vjepa21::VisionVideoVjepa21Model;
