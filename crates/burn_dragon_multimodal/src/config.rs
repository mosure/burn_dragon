use burn_dragon_core::BDHConfig;
use burn_dragon_stream::{FusionCarryPolicy, StateCarryPolicy, TargetAlignmentPolicy, TbpttWindow};
use burn_dragon_vision::VisionDragonConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FusionSlotConfig {
    pub slot_count: usize,
    pub use_modality_type_embeddings: bool,
}

impl Default for FusionSlotConfig {
    fn default() -> Self {
        Self {
            slot_count: 2,
            use_modality_type_embeddings: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TargetTeacherConfig {
    pub enabled: bool,
    pub decay: f32,
}

impl Default for TargetTeacherConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            decay: 0.99,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TargetTextEncoderKind {
    #[default]
    Dragon,
    FixedFourierMean,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MultimodalTbpttConfig {
    pub unroll_steps: usize,
    pub backprop_steps: usize,
    pub state_carry_policy: StateCarryPolicy,
    pub fusion_carry_policy: FusionCarryPolicy,
    pub target_alignment_policy: TargetAlignmentPolicy,
}

impl Default for MultimodalTbpttConfig {
    fn default() -> Self {
        Self {
            unroll_steps: 4,
            backprop_steps: 2,
            state_carry_policy: StateCarryPolicy::UntilBoundary,
            fusion_carry_policy: FusionCarryPolicy::UntilBoundary,
            target_alignment_policy: TargetAlignmentPolicy::SameStep,
        }
    }
}

impl MultimodalTbpttConfig {
    pub fn window(&self) -> TbpttWindow {
        TbpttWindow::new(self.unroll_steps, self.backprop_steps)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VlJepaDragonConfig {
    pub vision: VisionDragonConfig,
    pub query_text: BDHConfig,
    pub target_text: BDHConfig,
    pub fusion: BDHConfig,
    pub target_text_encoder: TargetTextEncoderKind,
    pub fusion_dim: usize,
    pub target_dim: usize,
    pub fusion_slots: FusionSlotConfig,
    pub pairwise_loss_weight: f32,
    pub target_bank_loss_weight: f32,
    pub vision_rollout_steps: usize,
    pub vision_backprop_steps: usize,
    pub fusion_refine_steps: usize,
    pub eval_fusion_refine_steps: Option<usize>,
    pub refine_loss_power: f32,
    pub video_interleave_refine_steps: usize,
    pub temperature: f32,
    pub target_teacher: TargetTeacherConfig,
    pub tbptt: MultimodalTbpttConfig,
}

impl Default for VlJepaDragonConfig {
    fn default() -> Self {
        Self {
            vision: VisionDragonConfig::default(),
            query_text: BDHConfig {
                n_layer: 2,
                n_embd: 64,
                n_head: 4,
                mlp_internal_dim_multiplier: 1,
                vocab_size: 256,
                ..Default::default()
            },
            target_text: BDHConfig {
                n_layer: 2,
                n_embd: 64,
                n_head: 4,
                mlp_internal_dim_multiplier: 1,
                vocab_size: 256,
                ..Default::default()
            },
            fusion: BDHConfig {
                n_layer: 2,
                n_embd: 64,
                n_head: 4,
                mlp_internal_dim_multiplier: 1,
                vocab_size: 1,
                ..Default::default()
            },
            target_text_encoder: TargetTextEncoderKind::Dragon,
            fusion_dim: 64,
            target_dim: 64,
            fusion_slots: FusionSlotConfig::default(),
            pairwise_loss_weight: 1.0,
            target_bank_loss_weight: 0.0,
            vision_rollout_steps: 2,
            vision_backprop_steps: 1,
            fusion_refine_steps: 2,
            eval_fusion_refine_steps: None,
            refine_loss_power: 1.0,
            video_interleave_refine_steps: 0,
            temperature: 0.07,
            target_teacher: TargetTeacherConfig::default(),
            tbptt: MultimodalTbpttConfig::default(),
        }
    }
}
