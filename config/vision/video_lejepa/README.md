This directory now uses config inheritance via top-level `extends`.

Base configs live under `base/`:
- `base/moving_mnist_dataset.toml`: shared Moving-MNIST dataset and augmentation defaults
- `base/moving_mnist_cellular.toml`: shared token-local/cellular video LEJEPA defaults
- `base/moving_mnist_pyramid.toml`: shared pyramid/structured recurrent video LEJEPA defaults

Experiment files in this directory are intended to be runnable directly with a single config path.
The vision loader resolves `extends` relative to the current config file and merges bases before
the local overrides.

Naming stays stable for existing experiment references, but duplication is now concentrated in the
base configs instead of copied across each run file.
