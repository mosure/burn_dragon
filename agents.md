# Future Agent Strategy

This file captures the current repository-level lessons from the latest Dragon vision cycle.

It is a strategy memo, not a permanent truth table. Future agents may overturn it, but they should
do so with stronger evidence than the current validated read.

## Current standing vision read

- Dense vision Dragon is a control and systems-test line, not the current modeling frontier.
- The graph-backed scene-slot family is the current main image line.
- The graph bridge is the current promoted quality model.
- Plain `scene_slots8` is the simpler runtime/control baseline.

## What recent evidence means

- The bridge is no longer justified by Imagenette alone.
- The bridge now survives broader checks on CIFAR-10, CIFAR-100, STL10, and Oxford Pets.
- The bridge also preserves a meaningful recurrent test-time scaling story.
- The bridge now has a plausible serving/runtime path on the current static Triton graph executor.

## How to make decisions

- Do not confuse a checked-in config with a completed run.
- Do not confuse a roadmap with validated evidence.
- Report blocked large-scale work explicitly.
- Prefer fixed-train-time quality and learning efficiency over tiny throughput deltas at small and
  medium scale.
- If a harder surface is underfit on a short recipe, run a modest longer-horizon follow-up before
  calling the branch weak.
- Require at least one broader dataset and one second resolution before promoting a new vision
  branch.

## How to handle large-scale vision claims

- If `ImageNet-1k` data or teacher assets are missing, say the large-scale line is blocked.
- If the teacher path is feature-file backed, do not describe the recipe as strong-augmentation
  multiteacher training unless the code explicitly supports alignment-safe online teachers.
- Distinguish between:
  - shared-head supervision in one student projection space
  - teacher-specific decoder heads over a shared Dragon recurrent state

## Current multiteacher default

The finalized multimode vision preference is:

1. primary DINOv2 `patch_and_cls`
2. auxiliary SigLIP2 `patch_and_cls` with `decoder_mode = "dedicated_spatial_projection"` when
   full spatial features are available
3. auxiliary SigLIP2 `global_only` only as the lower-storage fallback

Do not treat the global-only auxiliary path as the main architecture if the spatial path is
available. Treat it as the operational fallback.

## Near-term priorities

1. Make the large-scale ImageNet line real rather than merely launch-ready.
2. Prefer the finalized multimode spatial path over the global-only fallback whenever storage and
   export budget allow.
3. Scale the bridge line in width and resolution before reopening dense-centric tuning.
4. Reopen irregular graph runtime work only after a stronger large-scale model line exists.

## Vision default baseline pair

When testing new vision ideas, the default pair should usually be:

1. plain scene-slot graph
2. graph bridge

Use dense vision only when the comparison genuinely needs a dense control.
