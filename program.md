# burn_dragon autoresearch program

This file defines the autonomous research loop for this repository.

The goal is open-ended autoresearch: generate hypotheses, test them cheaply, keep what works, and avoid unnecessary attachment to prior architectural preferences.

## Mission

Continuously improve the repository by finding changes that measurably help at least one of:

- model quality
- training stability
- inference or training speed
- memory efficiency
- reproducibility
- usability of the existing training and evaluation paths

Work from evidence, not attachment to a specific implementation, paper analogy, roadmap, or earlier experiment narrative.

## Architectural north star

The standing architectural bias for this program is Dragon itself.

The aim is to develop Dragon as a post-transformer architecture that can take on the kinds of problems current transformers handle across the repository's supported domains.

Keep the Dragon core concepts legible when working on core model paths:

- `x_neuron`
- `y_gate`
- `y_neuron`
- `rho`

This is a directional bias, not a freeze on experimentation. Question current implementations freely, but prefer work that strengthens Dragon as a general-purpose alternative rather than drifting into arbitrary one-off designs.

## Source of truth

Use the current repository state as ground truth:

- `README.md`
- runnable code in `crates/*`
- active configs in `config/*`
- tests, benches, and examples that still compile and run
- metrics produced by current run directories under `runs/`

Do not use architectural roadmaps, specs, or progress trackers as the basis for experiment choice or evaluation. They may be useful historical artifacts, but they are not authoritative for autoresearch.

Read only enough of the code and configs to support the current experiment. Do not mass-load unrelated files.

## Scope

Any repository-supported family is in scope if it has runnable code and a measurable outcome, including:

- language
- vision
- multimodal
- graph
- sudoku
- shared infrastructure in core, stream, train, checkpoint, and WGPU paths

Prefer directions with:

- a cheap reproducible baseline
- a clear validation metric
- a short iteration loop

Language and vision may often be the cheapest places to iterate, but they are not the only valid targets.

## Research posture

Treat this as search under uncertainty.

- Do not assume the current default config is optimal.
- Do not assume an existing backbone, norm, state layout, or training recipe is privileged.
- Do not keep pushing a line of work just because it matches earlier plans.
- Do diversify when a direction stalls or repeatedly loses.
- Do prefer experiments that teach something even when they fail.
- Do favor changes that make Dragon more capable as a reusable post-transformer system.

When choosing between ideas, prioritize expected information gain per unit time.

## Evidence discipline

Do not confuse any of the following:

- a checked-in config
- a roadmap phase
- a launch-ready path
- a completed run
- a validated frontier result

Report each state explicitly.

If a large-scale run is blocked by missing data, missing teacher exports, invalid augmentation
contracts, or runtime failures, say so directly. Do not write as if the model already exists.

When a branch is only supported by one small surface, say that too. Promote claims only after the
branch survives the right validation surface.

## Build once, then reuse binaries

Avoid paying Cargo startup cost every run when working through a loop.

Typical builds:

```bash
cargo build -p burn_dragon_cli --no-default-features --features train --bin train
cargo build -p burn_dragon_cli --features benchmark --bins
```

Use:

- `target/debug/train` for training runs
- benchmark bins under `target/debug/` when relevant

If only one crate or bench matters for the current task, build the narrowest useful target.

## Results file

Keep one untracked TSV at repo root:

- `results_autoresearch.tsv`

Do not commit it.

Header:

```text
commit	track	config	primary_metric	primary_value	secondary_metrics	wall_seconds	memory_gb	status	description	run_dir
```

Guidance:

- `track`: short family or subsystem label such as `language`, `vision`, `multimodal`, `graph`, `kernel`, or `train`
- `config`: main config or benchmark identifier
- `primary_metric`: the main metric used for the decision
- `wall_seconds`: active training or benchmark wall time used for the decision; exclude one-time startup and compilation cost when comparing training variants
- `secondary_metrics`: compact `key=value` pairs for supporting evidence
- `secondary_metrics` should usually include throughput information such as `iter_s`, `tokens_s`, `samples_s`, or equivalent when training speed may affect fairness
- `status`: `keep`, `discard`, `crash`, or `inconclusive`

If you inherit older per-track TSVs, do not rely on them as the operating contract for this program.

## Experiment delimiter

The default delimiter for training experiments is a fixed wall-clock training budget.

- Compare baseline and variant on the same task, same data split, and same active training wall time.
- Exclude one-time startup, compile, and binary-build cost from that budget unless startup cost is itself the subject of the experiment.
- Do not treat equal steps, equal epochs, or equal tokens as the main fairness rule when throughput differs materially.
- If a change improves quality only by consuming much more training wall time, treat that as a throughput tradeoff rather than a clean architecture or pipeline win.

For vision experiments, a fixed-time comparison is only meaningful after the run clears a minimum useful compute floor. If the workload is obviously underpowered, the result is not a trustworthy architecture-selection signal even if the wall-clock delimiter was respected.

## Vision compute floor

Cheap vision smokes are allowed, but only for:

- crash detection
- gross regression detection
- verifying that a structural change points in the right direction

Do not use underpowered vision runs to declare architectural winners.

Treat a vision experiment as underpowered when most of the following are true:

- GPU utilization is consistently low enough that the device is mostly waiting on host orchestration
- VRAM footprint is tiny relative to the available device memory
- the run is dominated by tiny fragmented structured steps instead of dense compute
- candidate branches differ only by very small metric deltas while the model is still clearly below a realistic learning regime
- the experiment family is so small that it cannot plausibly clear the current learning-noise floor

Underpowered runs may still be useful for directional filtering. They are not sufficient for claiming a best architecture.

When a vision family is underpowered:

1. keep only gross directional lessons
2. stop fine-grained scalar sweeps
3. promote to a larger-capacity or more GPU-saturating baseline before continuing model selection

Examples of acceptable promotions:

- larger recurrent/state capacity
- larger active training subset or longer fixed-time budget
- larger batch if the path is stable
- a more fused or less host-fragmented executor
- a more realistic multi-stage memory path instead of a toy single-stage branch

## Current vision frontier policy

The current standing read for vision is:

- the old dense vision Dragon line is a control and systems-test line, not the main modeling bet
- the graph-backed scene-slot family is the current main image line
- the graph bridge is the current promoted quality model
- the plain scene-slot graph is the simpler runtime/control baseline

Treat this as current evidence, not a permanent freeze. Future agents may overturn it, but only
with stronger evidence than the current broader-validation read.

When comparing new vision variants, the default baseline pair should usually be:

1. plain scene-slot graph
2. graph bridge

Do not spend long cycles retuning dense vision unless the goal is:

- regression detection
- evaluator hardening
- systems comparison
- testing whether a new idea helps the dense control too

### Vision performance protocol

When a vision branch clears the old toy floor but still shows bursty, low-power behavior, treat it as an execution problem first, not an architecture-selection problem.

In that regime:

1. Use GPU power draw as the primary operational signal.
2. Use VRAM footprint and train-step throughput as secondary operational signals.
3. Treat averaged GPU-util percentages as supporting context only.
4. Pause fine-grained architecture sweeps until the main host-orchestration bottlenecks are benchmarked.
5. Prefer executor and kernel work over more schedule or capacity micro-ablations.

Preferred order of attack:

1. benchmark and identify the dominant host-side structured bucket
2. fuse the remaining coarse/patch spatial structured path
3. fuse the hub/global path
4. reduce host-driven recurrent substep orchestration
5. only then resume architecture selection inside the denser training regime

### Vision promotion rules

Do not promote a new vision model line on a single easy surface.

Prefer this minimum promotion bar:

1. one matched fixed-time win on the current main surface
2. confirmation on at least one broader non-Imagenette surface
3. confirmation on at least one second resolution

If a harder surface is obviously underfit under the short recipe, run a modest longer-horizon
follow-up before declaring the branch weak. Underfit short runs are screening signals, not final
judgments.

At small and medium scales, prioritize:

1. quality reached in fixed train time
2. quality reached after a modest longer-horizon follow-up
3. stability across seeds
4. only then small throughput deltas

Do not let tiny throughput differences dominate model selection while the architecture question is
still moving materially on quality.

### Vision multiteacher discipline

Feature-file teachers and strong online augmentations are not interchangeable.

If any teacher path is feature-file backed:

- treat deterministic augmentation as the valid default unless the code explicitly supports
  alignment-safe online teachers
- do not describe the run as a strong-augmentation multiteacher recipe

For multiteacher work, distinguish clearly between:

- shared-head supervision in one projection space
- teacher-specific decoder heads over a shared recurrent state

Those are different capability levels and should not be reported as the same thing.

The current preferred multimode vision recipe is:

1. primary DINOv2 `patch_and_cls`
2. auxiliary SigLIP2 `patch_and_cls` with `decoder_mode = "dedicated_spatial_projection"` when
   full spatial features are available
3. auxiliary SigLIP2 `global_only` only as the lower-storage fallback

Do not treat the global-only auxiliary path as the main architecture if the spatial path is
available. Treat it as the storage-constrained fallback.

## Baseline policy

Always establish a baseline before claiming an improvement.

For each new line of inquiry:

1. Pick the cheapest existing config or benchmark that still exercises the behavior you plan to change.
2. Run the baseline and record it.
3. Change one idea, or one tightly related bundle of ideas.
4. Compare against the baseline on the same task and the same fixed active training-time budget.

Do not jump straight to large runs unless the small run cannot observe the behavior of interest.

If multiple cheap baselines exist, prefer the one with:

- clearer metrics
- lower cost
- lower variance
- better coverage of the code you are modifying

## How to choose experiments

Pick experiments from measurable uncertainty in the current codebase, not from ideological preference.

Good starting categories:

- architecture changes
- optimizer or schedule changes
- state layout or recurrence changes
- normalization changes
- dataflow or train-step changes
- kernel and memory-path improvements
- config simplifications
- benchmark or test additions that expose hidden regressions

Useful heuristics:

- exploit a recent win when follow-on variants are cheap
- explore neglected but plausible directions when the current line stalls
- prefer changes with a clear pass/fail signal
- prefer one sharp hypothesis over a large mixed patch

## How to evaluate results

Do not rely on stdout summaries alone. Inspect the run directory and task artifacts.

Common places to inspect:

- `config.json`
- `experiment.log`
- validation loss logs
- train throughput or iteration speed logs
- task-specific artifacts under `artifacts/`

Generic discovery example:

```bash
latest=$(find runs -mindepth 1 -maxdepth 3 -type d -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
find "$latest" -maxdepth 3 -type f | rg 'Loss|loss|Speed|Throughput|memory|experiment\.log|config\.json|artifacts'
```

Use a primary metric that matches the task. For training experiments, the usual question is: what task quality did this variant reach within the fixed active wall-clock budget?

Typical priorities are:

1. task quality reached within the fixed training-time budget
2. stability and reproducibility within that budget
3. throughput, especially when it changes how much learning fits inside the budget
4. memory usage
5. implementation complexity

Do not optimize one metric blindly if the broader result is clearly worse.

A slower variant is not a clean win just because it eventually reaches a better metric after running much longer. Under this program, quality gains must survive the fixed-time comparison.

For cheap vision JEPA work, prefer a pair of primary metrics:

1. the core JEPA metric you actually care about, such as invariant or long-horizon predictive error
2. a practical training metric such as composite validation loss or throughput

If a branch only wins by tiny changes in the JEPA metric while the run is clearly underpowered, mark the result as directional or inconclusive instead of promoting it as the new default.

## What you may change

You may modify any relevant files in:

- `crates/*`
- `config/*`
- relevant tests, benches, examples, and docs

You may change:

- model structure
- training logic
- kernels
- optimizer settings
- schedules
- configs
- tests and benchmarks
- evaluation code

Keep crate boundaries clean and avoid unrelated refactors.

## What not to do casually

- do not add new dependencies or crates without a clear reusable need
- do not preserve obsolete paths just for compatibility
- do not introduce complexity without a measurable upside
- do not ignore regressions in VRAM, throughput, or stability
- do not edit unrelated deployment or export code unless the experiment touches it
- do not keep retrying a repeatedly losing idea without changing the hypothesis

## Experiment loop

Loop continuously until interrupted by the user or blocked by a real issue.

1. Check git state and understand any existing local changes.
2. Choose one experiment with a clear success criterion.
3. Make the smallest clean change that tests the idea.
4. Run targeted tests first.
5. Run the cheapest relevant training, eval, or benchmark baseline if needed.
6. Run the changed version on the same target and active training-time budget.
7. Read metrics from the run directory or benchmark output.
8. Record the outcome in `results_autoresearch.tsv`.
9. Keep the change only if the evidence supports it.
10. If the result is negative or inconclusive, adjust the hypothesis and continue.

Do not stop to ask whether to continue after every run. Continue the loop.

## Keep / discard rules

### Keep a change if

- it clearly improves the primary metric within the same fixed training-time budget
- or it preserves the primary metric while materially improving speed, memory, stability, or simplicity
- or it exposes a useful new benchmark or regression test with low maintenance cost

### Discard a change if

- it worsens the main task without a compelling tradeoff
- it adds substantial complexity for marginal gain
- it regresses speed or memory without quality benefit
- it improves quality only by taking materially longer wall-clock training time
- it makes behavior harder to reproduce or reason about

### Mark inconclusive if

- the metric noise is too high to call
- the run budget was too small to observe the intended effect
- baseline and variant were not measured at matched active training-time budgets
- the implementation changed too many variables at once
- the vision run is obviously underpowered and therefore below a trustworthy architecture-selection signal floor

In inconclusive cases, simplify the experiment or rerun with a more stable setup before drawing conclusions.

## Crash handling

If a run crashes:

- fix obvious implementation mistakes and rerun if the idea is still sound
- record `crash` if the variant is unstable or structurally bad
- move on when the failure teaches enough

If a run greatly exceeds the intended smoke budget, terminate it, record the outcome, and shrink the test.

## Cheap-first promotion policy

Earn expensive runs.

- Start with smoke, tiny, or otherwise bounded configs when possible.
- Promote only after a cheap run shows a repeatable fixed-time win or a strong reason to believe the cheap proxy is misleading.
- For infra or kernel work, prefer targeted benches before large training runs.

Large runs are for confirmation, not for discovering whether an idea obviously fails.

For vision specifically:

- use cheap single-stage smokes to reject obviously bad topology or schedule ideas
- do not stay in a toy single-stage family once the branch differences are down in the likely noise floor
- if GPU utilization remains poor and the family is still learning below the task noise floor, promote before continuing ablations
- once promoted, compare branches only inside the promoted regime; do not mix conclusions from underpowered and promoted runs

## Quality bar for kept changes

Before keeping a nontrivial change, run:

- relevant unit tests
- relevant targeted integration tests
- relevant benchmarks when making performance claims
- `cargo clippy` on touched crates with `-D warnings` when practical

If behavior, config usage, or benchmarks changed materially, update the corresponding docs or examples after the result is accepted.

## Notes on working in this repository

- The worktree may already be dirty. Do not revert unrelated changes.
- Use focused commits when committing.
- Prefer `rg` for search.
- Prefer existing config overlay patterns over ad hoc flags.
- Reuse the current run-directory and benchmark flow instead of inventing a parallel system unless the system itself is the subject of research.

## Default operating posture

Be autonomous, empirical, and open-ended.

The loop is:

- establish baseline
- form hypothesis
- make a focused change
- test
- run
- measure
- record
- keep or discard
- repeat

Optimize for learning rate and verified improvement, not for preserving prior architectural commitments.
