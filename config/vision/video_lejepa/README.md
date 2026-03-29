This directory now uses config inheritance via top-level `extends`.

Base configs live under `base/`:
- `base/moving_mnist_dataset.toml`: shared Moving-MNIST dataset and augmentation defaults
- `base/moving_mnist_cellular.toml`: shared token-local/cellular video LEJEPA defaults
- `base/moving_mnist_pyramid.toml`: shared pyramid/structured recurrent video LEJEPA defaults

Curated tracked aliases live under `baselines/`:
- `baselines/vjepa21_dense_promoted.toml`: canonical dense V-JEPA 2.1 Moving-MNIST baseline;
  currently mirrors `moving_mnist_vjepa21_dense_wgpu_diag_probe025_recon1_256_wide.toml`
- `baselines/vjepa21_imagenet1k_dense_smoke.toml`: tiny ImageNet-1k V-JEPA 2.1 dense smoke
  config using image multi-view clips
- `baselines/vjepa21_imagenet1k_dense_long.toml`: longer ImageNet-1k V-JEPA 2.1 dense launch
  config using image multi-view clips

The vision loader resolves `extends` relative to the current config file and merges bases before
the local overrides.

Checked-in video LEJEPA configs are limited to `base/` plus promoted `baselines/`. Transfer,
diagnostic, and temporary experiment overlays belong under `config/local/vision/video_lejepa/`.
