# TODO

- [ ] Add gating or residual scaling (e.g. `traj_next = traj + proj(out)`).
- [ ] Add a viz panel for decoded activations (`mixed_flat.matmul(decoder)`) or per-token energy/entropy over time to surface projection health (saturation/dead channels).
