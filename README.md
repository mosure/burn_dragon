# burn_dragon 🔥🐉🐣

[![test](https://github.com/mosure/burn_dragon/workflows/test/badge.svg)](https://github.com/Mosure/burn_dragon/actions?query=workflow%3Atest)
[![GitHub License](https://img.shields.io/github/license/mosure/burn_dragon)](https://raw.githubusercontent.com/mosure/burn_dragon/main/LICENSE)
[![crates.io](https://img.shields.io/crates/v/burn_dragon.svg)](https://crates.io/crates/burn_dragon)


burn inference and training of the [dragon model](https://arxiv.org/abs/2509.26507)


![Alt text](./docs/vocab.png)
![Alt text](./docs/bdh.png)


## features

- [x] cached inference
- [x] training benchmarks and reporting
- [x] [pope](https://arxiv.org/abs/2509.10534v1) positional embeddings
- [x] wasm deployments
- [x] sparsity metrics and visualization
- [x] vision dragon [cortical column/stack](https://arxiv.org/abs/2412.18354)
- [x] recurrent, foveated saccade training and inference
- [x] [GDPO](https://arxiv.org/abs/2601.05242)
- [ ] adaptive tool discovery
- [ ] conditional (deep) gating
- [ ] document-coherent dataloading and scale mixup
- [ ] episodic memory
- [ ] fused kernels
- [ ] hierarchical, memory-aware recurrent state
- [ ] mixture-of-expert routing
- [ ] multi-modal architecture
- [ ] multi-stream truncated backpropagation through time
- [ ] neuromorphic backend
- [ ] streaming, sparse synaptic backpropagation
- [ ] temporal neuron dampening


Dataset configuration (built-in presets and Hugging Face examples) is documented inline in `config/language/base.toml`.

## training

- `cargo run -p burn_dragon_cli --release` (defaults to the cuda backend)


## inference

- `cargo run -p burn_dragon_cli --bin infer -- --max-tokens 2048 --streaming`


## benchmarks

- `cargo bench -p burn_dragon --features train,benchmark` (executes both wgpu and cuda benchmarks)
- open `target/criterion/report/index.html`


## compatible burn versions

| `burn_dragon` | `burn` |
| :--                     | :--    |
| `0.2`                   | `0.19` |
| `0.1`                   | `0.18` |


## license
licensed under either of

 - Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
 - MIT license (http://opensource.org/licenses/MIT)

at your option.


## contribution

unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.


![Alt text](./docs/community.png)
