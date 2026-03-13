# burn_dragon research program

This file defines the autonomous research loop for this repository.

It is inspired by `autoresearch/program.md`, but it is adapted to the actual structure, crates, configs, metrics, and design priorities of `burn_dragon`.

## Mission

Continuously improve two tracks:

1. language Dragon on Shakespeare
2. vision Dragon encoder on the current V-JEPA2-like path

Do this by running short, careful, reproducible experiments; keeping only improvements; and preserving the design integrity of the repository.

This program is not for multimodal work right now. Do not spend time on VL-JEPA or multimodal composition unless a language or vision change explicitly requires touching shared infrastructure.

## Scope

Primary tracks:

- `language`: Shakespeare next-token training
- `vision`: V-JEPA2-like Vision Dragon encoder iteration via the current `video_lejepa` / `Pyramid` path

Secondary work only when justified by the active experiment:

- `burn_dragon_core`
- `burn_dragon_wgpu`
- `burn_dragon_train`
- `config/language/*`
- `config/vision/*`
- relevant docs, tests, and benches

Do not introduce new crates for this program unless there is a clearly reusable core/framework need.

## Design principles

### 1. Bitter lesson first

Prefer:

- shared recurrent computation
- scalable state topology
- efficient kernels
- simple, general mechanisms

Avoid:

- ad hoc auxiliary towers
- clever one-off heads
- brittle task-specific hacks that do not generalize

### 2. Current API is the source of truth

This repository is not a published stable library yet. Do not preserve obsolete paths just for compatibility.

Prefer:

- current curated `api` surfaces
- current configs
- current preferred backbones

### 3. Canonical Dragon semantics matter

Keep the model conceptually aligned with:

- `x_neuron`
- `y_gate`
- `y_neuron`
- `rho`

Do not blur these with vague or transformer-generic terminology when changing core recurrent paths.

### 4. Simplicity wins ties

If two variants are similar in quality, prefer:

- simpler implementation
- cleaner crate boundaries
- lower VRAM
- faster iteration

### 5. Keep the preferred paths preferred

Current preferred paths:

- language: BDH with fused recurrent WGPU path when it is actually beneficial
- vision: `Pyramid` backbone

Do not create another parallel “research-only” vision backbone when the correct move is to improve `Pyramid`.

## Read this first

Before starting a new line of experiments, review enough of these to understand the current state:

- [README.md](./README.md)
- [docs/dragon_framework_support_matrix.md](./docs/dragon_framework_support_matrix.md)
- [docs/dragon_hatchling_alignment_spec.md](./docs/dragon_hatchling_alignment_spec.md)
- [docs/vision_scale_aware_rho_roadmap.md](./docs/vision_scale_aware_rho_roadmap.md)

For language:

- `crates/burn_dragon_core/src/model/bdh.rs`
- `crates/burn_dragon_core/src/model/norm.rs`
- `crates/burn_dragon_core/src/model/config.rs`
- `crates/burn_dragon_language/src/config/train/*`
- `crates/burn_dragon_language/src/train/*`
- `config/language/*.toml`

For vision:

- `crates/burn_dragon_vision/src/model/vision.rs`
- `crates/burn_dragon_vision/src/model/vision/pyramid_ops.rs`
- `crates/burn_dragon_vision/src/model/vision/config.rs`
- `crates/burn_dragon_vision/src/train/vision/video/*`
- `config/vision/video_lejepa/*`

Read only enough to support the current experiment. Do not mass-load irrelevant files.

## Build once, then reuse binaries

Do not pay Cargo startup cost every run.

Recommended builds:

```bash
cargo build -p burn_dragon_cli --no-default-features --features train --bin train
cargo build -p burn_dragon_cli --features benchmark --bins
```

Use:

- `target/debug/train` for training
- benchmark bins under `target/debug/`

## Results files

Keep two untracked TSVs at repo root:

- `results_shakespeare.tsv`
- `results_vjepa2.tsv`

Do not commit them.

### `results_shakespeare.tsv`

Header:

```text
commit	valid_loss	memory_gb	wall_seconds	status	description
```

### `results_vjepa2.tsv`

Header:

```text
commit	valid_loss	inv_loss	iter_s	memory_gb	wall_seconds	status	description
```

Status values:

- `keep`
- `discard`
- `crash`

Use `0` values for crashes where needed.

## Baselines

Always establish the baseline first for each fresh line of inquiry.

### Language baseline

Use the current Shakespeare smoke stack:

```bash
target/debug/train language \
  -c config/language/tiny.toml \
  -c config/language/shakespeare_smoke.toml \
  -c config/language/shakespeare_fused.toml \
  --backend wgpu
```

If the active line of work is normalization, neuron-space, or recurrence, this is the baseline.

Current practical guidance:

- `RMSNorm` is the strongest norm swap so far
- `DyT` and `Derf` are not current defaults
- `y_neuron` carry is experimental only

### Vision baseline

Use the current cheap V-JEPA2-like `Pyramid` smoke:

```bash
target/debug/train vision \
  -c config/vision/base.toml \
  -c config/vision/video_lejepa/moving_mnist_trm_norm_smoke.toml \
  --backend wgpu
```

This is the default cheap structured-vision baseline for encoder iteration.

If the active line of work is specifically scale-aware `rho`, stage-local sharing, connectivity, or stems, this is the baseline.

### Promotion baseline

Only after a cheap vision variant wins clearly, promote to a stronger config such as one of:

- `config/vision/video_lejepa/moving_mnist_trm_baseline.toml`
- `config/vision/video_lejepa/moving_mnist_trm_perf_pilot.toml`
- other explicitly stronger configs already present in `config/vision/video_lejepa/`

Do not jump to larger runs first.

## How to read results

Do not rely on stdout summaries. Read the run directory.

### Language

The latest language run directory is typically under `runs/<name>/`.

Important files:

- `valid/epoch-*/Loss.log`
- `train/epoch-*/Loss.log`
- `config.json`
- `experiment.log`

Example metric extraction:

```bash
latest=$(find runs -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
tail -n 1 "$latest/valid/epoch-1/Loss.log"
```

Lower valid loss is better.

### Vision

The latest vision run directory is typically under `runs/vision/<name>/`.

Important files:

- `valid/epoch-*/Loss.log`
- `valid/epoch-*/video_lejepa_inv_loss.log`
- `valid/epoch-*/video_lejepa_long_rollout_inv_to_h*.log`
- `train/epoch-*/Iteration_Speed.log`
- `artifacts/`

Example metric extraction:

```bash
latest=$(find runs/vision -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
tail -n 1 "$latest/valid/epoch-1/Loss.log"
tail -n 1 "$latest/valid/epoch-1/video_lejepa_inv_loss.log"
tail -n 1 "$latest/train/epoch-1/Iteration_Speed.log"
```

For vision, do not optimize one metric blindly.

Keep a change only if it is good on the whole picture:

1. overall valid loss
2. invariant latent loss
3. long-rollout invariant metrics when available
4. throughput
5. VRAM

If metrics conflict:

- prefer variants that improve or preserve overall valid loss
- then prefer lower invariant loss
- then prefer better long-rollout behavior
- then prefer faster/simpler variants

## What you may change

You may modify any relevant files in:

- `crates/burn_dragon_core`
- `crates/burn_dragon_wgpu`
- `crates/burn_dragon_train`
- `crates/burn_dragon_language`
- `crates/burn_dragon_vision`
- `config/language`
- `config/vision`
- relevant tests/benches/docs

You may:

- change architecture
- change optimizer and LR schedules
- change neuron-space size
- change norms
- change recurrent topology
- change stems and positive maps
- change train-step organization
- add or refine benchmarks/tests

You must keep crate boundaries clean.

## What you should not do casually

- do not spend time on multimodal
- do not add new dependencies
- do not add a new vision backbone family when `Pyramid` should be improved instead
- do not add ugly special-case heads unless the gain is very clear
- do not change deployment/export code unless your experiment requires it
- do not quietly ignore VRAM pathologies, low GPU utilization, or host-transfer regressions

## Experiment loop

Loop continuously until interrupted by the user or blocked by a real issue.

1. Check git state and current local changes.
2. Pick one experiment idea or one tightly related bundle of changes.
3. Make the change cleanly.
4. Run targeted unit tests first.
5. Run the cheap training smoke for the relevant track.
6. Read metrics from the run directory.
7. Record the result in the appropriate TSV.
8. Keep the change only if it is a clear win by the rules below.
9. If it is not a win, revert only your own changes and try the next idea.

Do not stop to ask whether to continue after every run. Continue the loop.

## Keep / discard rules

### Keep a change if

- it clearly improves the primary metric
- or it preserves the primary metric and meaningfully improves:
  - speed
  - VRAM
  - simplicity
  - long-horizon/refinement behavior

### Discard a change if

- it worsens the primary metric without a compelling tradeoff
- it regresses speed materially with no quality gain
- it increases VRAM substantially with no quality gain
- it adds ugly complexity for a marginal gain

### Crash handling

If a run crashes:

- fix obvious implementation mistakes and rerun
- if the idea is structurally bad, log it as `crash` and move on

If a run exceeds the intended smoke budget badly, kill it and mark it as failure.

## Track-specific research priorities

## Language: Shakespeare

Focus on:

- neuron-space / `rho` capacity
- practical norms (`RMSNorm` first)
- fused recurrent efficiency
- avoiding pathological host transfers or low GPU utilization
- simple recurrence changes that survive the smoke loop

Good language lines of inquiry:

1. larger neuron-space with bounded VRAM
2. `RMSNorm` defaults and retuning
3. fused recurrent path quality/perf tuning
4. better startup autotune and effective-batch policy
5. carefully constrained `y_neuron` carry only if it earns its keep

Bad language lines of inquiry:

- chasing `DyT` / `Derf` endlessly when they are clearly behind
- exact `y_neuron` carry paths that destroy throughput

## Vision: V-JEPA2-like encoder

Focus on:

- `Pyramid` only
- scale-aware `rho` banks
- stage-local sharing
- bank connectivity and activation schedules
- local stem + positive map
- test-time latent refinement that genuinely improves metrics

Follow [docs/vision_scale_aware_rho_roadmap.md](./docs/vision_scale_aware_rho_roadmap.md).

Good vision lines of inquiry:

1. heterogeneous patch/coarse/global bank sizes
2. stage-local sharing instead of uniform sharing
3. explicit bank enable/read/write schedules by mode
4. local stem / positive map ablations
5. directional local banks only after the above are stable

Bad vision lines of inquiry:

- making heads the main capacity lever
- per-neuron bespoke `rho`
- adding large explicit auxiliary readout towers
- jumping to large datasets before the cheap ladder is won

## Cheap-first promotion policy

Never promote a variant to larger runs unless it wins on the cheap ladder.

### Language promotion

Promote only if:

- valid loss improves
- or valid loss is flat while throughput/VRAM improve meaningfully

### Vision promotion

Promote only if:

- cheap smoke valid loss improves or stays flat
- invariant loss improves
- no major throughput or memory regression
- extra refine depth helps, or at least does not reveal a broken refinement story

## Quality bar for merged/kept changes

Before keeping a nontrivial change, run:

- relevant unit tests
- relevant targeted integration tests
- `cargo clippy` on touched crates with `-D warnings`

For kernel or recurrent changes, also run:

- the relevant benchmark bin
- the relevant bounded-memory regression if it exists

## Notes on working in this repository

- The worktree may already be dirty. Do not revert unrelated changes.
- Use focused commits.
- Prefer `rg` for search.
- Use the current config overlay pattern instead of inventing new ad hoc flags.
- Reuse the current benchmark and run-directory flow instead of building a parallel system.

## Default operating posture

Be autonomous.

The program is:

- baseline
- change
- test
- run
- measure
- keep or discard
- repeat

Keep iterating on Shakespeare and the V-JEPA2-like vision encoder until interrupted, with the cheap smoke ladders as the gate for further promotion.
