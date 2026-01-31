# TODO

- [ ] Add gating or residual scaling (e.g. `traj_next = traj + proj(out)`).
- [ ] Add a viz panel for decoded activations (`mixed_flat.matmul(decoder)`) or per-token energy/entropy over time to surface projection health (saturation/dead channels).
- [ ] dataset traits and implementations for streams in burn_dragon_stream


## Fused Scan Kernels (BDH Recurrent)

- [ ] Design a generic fused-scan interface for BDH (shared by Sudoku + Vision saccade).
- [ ] Define a minimal step I/O trait (state + step input -> state + step output) and optional policy/value head hooks.
- [ ] Implement a fused forward kernel for BDH recurrence over T steps.
- [ ] Add optional cache-update hook inside the scan (per-step or per-token write).
- [ ] Add optional policy/value head inside scan for RL rollouts.
- [ ] Implement custom backward or checkpointed backward for the fused scan.
- [ ] Provide thin adaptors in burn_dragon_sudoku and burn_dragon_vision to use the fused scan.
- [ ] Benchmark vs. current recurrent TBPTT (utilization, throughput, memory) and document wins.

