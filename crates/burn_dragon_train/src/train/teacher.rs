use crate::train::prelude::*;

pub fn teacher_variant_dim(variant: VisionTeacherVariant) -> usize {
    match variant {
        VisionTeacherVariant::Vits => 384,
        VisionTeacherVariant::Vitb => 768,
        VisionTeacherVariant::Vitl => 1024,
        VisionTeacherVariant::Vitg => 1536,
    }
}

#[cfg(feature = "burn_dino")]
pub fn build_dino_config(
    variant: VisionTeacherVariant,
    image_size: usize,
    patch_size: usize,
) -> DinoVisionTransformerConfig {
    match variant {
        VisionTeacherVariant::Vits => {
            DinoVisionTransformerConfig::vits(Some(image_size), Some(patch_size))
        }
        VisionTeacherVariant::Vitb => {
            DinoVisionTransformerConfig::vitb(Some(image_size), Some(patch_size))
        }
        VisionTeacherVariant::Vitl => {
            DinoVisionTransformerConfig::vitl(Some(image_size), Some(patch_size))
        }
        VisionTeacherVariant::Vitg => {
            DinoVisionTransformerConfig::vitg(Some(image_size), Some(patch_size))
        }
    }
}
