# burn_dragon_p2p

`burn_dragon_p2p` integrates `burn_p2p` with burn_dragon language experiments.

Current supported experiment families:

- NCA pre-pre-training
- ClimbMix pre-training

The crate is intentionally split into three layers:

- [config](src/config.rs): stable experiment, auth, and backend configuration
- [native](src/native.rs): native peer preparation for CPU, WGPU, and CUDA
- [wasm](src/wasm/mod.rs): browser auth, Dioxus UI, and WebGPU browser training

It is still a library crate first, but both operator surfaces now exist:

- browser: productized through the Dioxus component and browser runtime
- native: productized through the `burn_dragon_p2p_native` operator binary

Deployment assets live in [deploy](deploy):

- [deploy/README.md](deploy/README.md): GitHub Actions, Terraform, and required repo/environment secrets
- [deploy/profiles](deploy/profiles): initial Dragon training-profile sources and published network profile payloads
- [deploy/terraform/aws](deploy/terraform/aws): checked-in AWS bootstrap/edge Terraform root

## Target Matrix

- native CPU:
  - feature set: `native`
  - intended for validation, reducers, and low-scale local trainer smoke
- native WGPU:
  - feature set: `native,wgpu`
  - intended for native GPU trainer peers
- native CUDA:
  - feature set: `native,cuda`
  - intended for native GPU trainer peers on CUDA hosts
- browser WebGPU:
  - feature set: `wasm-ui,wasm-peer,wgpu`
  - intended for real browser trainer and verifier participation
- browser CPU:
  - feature set: `wasm-ui,wasm-peer`
  - smoke and development only

Browser CPU is not treated as a real deployment mode. The actual browser trainer path is WebGPU.

## Features

- `native`
  - enables native learner integration and shard-backed experiment prep
- `wasm-ui`
  - enables the Dioxus browser UI and browser auth/session flows
- `wasm-peer`
  - enables browser-local Dragon training and token-source loaders
- `wgpu`
  - enables native WGPU and browser WebGPU backends
- `cuda`
  - enables native CUDA peers

There is intentionally no Cargo feature called `internet-scale`. GitHub-authenticated network participation is part of the normal runtime policy of this crate.

## Auth Model

For network participation:

- native peers require a GitHub-authenticated auth bundle
- browser peers require a GitHub-authenticated browser session when `require_github_auth` is set
- browser training submission requires WebGPU
- dynamic admin edits are authenticated with a GitHub-backed session id, not a shared bootstrap token

The relevant seams are in:

- [auth.rs](src/auth.rs)
- [native.rs](src/native.rs)
- [wasm/mod.rs](src/wasm/mod.rs)

## Automatic Trainer Downgrade

Peers do not assume they can train just because the binary was built with `wgpu` or `cuda`.

Both native and browser paths now run a local preflight assessment before advertising a trainer role:

- estimate model + optimizer + activation footprint from the actual Dragon revision config
- compare that estimate against the configured trainer memory budget
- downgrade automatically when the fit looks unsafe

Current default budgets are conservative:

- native CPU: `8 GiB`
- native WGPU: `4 GiB`
- native CUDA: `6 GiB`
- browser WebGPU: `2 GiB`

Fallback policy:

- native peers: `trainer -> validator`
- browser peers: `browser_trainer_wgpu -> browser_verifier`

This is still a heuristic fit model, not a portable exact VRAM probe. The important product behavior is that undersized peers should downgrade before training starts instead of crashing on first optimizer allocation.

Native and browser peers also persist downgrade state for a specific workload fingerprint:

- experiment kind
- backend
- model config
- batch size
- block size

If a trainer run fails with a probable fit error like OOM / failed allocation / device loss, the next startup comes back as validator or verifier automatically instead of retrying trainer blindly. The downgrade record stops binding automatically if the configured trainer budget increases above the recorded failed footprint, and native peers can also clear it explicitly.

The browser app now renders the local capability decision directly:

- recommended role
- estimated training footprint
- trainer memory budget
- estimated tokens/sec
- checkpoint / shard / window budgets

## Browser Data Sources

Browser-local training supports:

- inline token windows
- HTTP JSON token-window shards
- HTTP shard manifests with per-shard integrity verification
- generated NCA corpora

That covers:

- synthetic NCA pre-pre-training
- shard-backed ClimbMix pre-training

For ClimbMix, the intended browser path is the shard-manifest form. The browser fetches
`fetch-manifest.json`, selects a bounded per-peer shard subset from the full shard pool,
downloads only those shard files on demand, verifies shard byte length and content hash, and then
decodes the token-window records locally. The checked-in profile uses deterministic peer selection
with a bounded shard window instead of walking the entire manifest from the front. When the host
runtime provides an exact browser training lease, the browser uses those assigned microshards
directly instead of the deterministic fallback.

## Join Mainnet

The mainnet join flow requires an edge URL from the deployment operator. This README uses `MAINNET_EDGE_URL` as a placeholder for that value.

The deployed network can publish Dragon experiment profiles directly in the directory. When those profiles are present, peers do not need a matching static experiment config on disk.
The deployed initial ClimbMix revision should point at a full external shard pool base URL. The
AWS deploy workflow publishes `${base_url}/fetch-manifest.json` into the initial browser profile,
so browser peers still fetch only the shards they train on without relying on repo-tracked shard blobs.
When the browser runtime has already persisted an exact training lease for the current assignment,
the Dragon browser app now picks that lease up automatically before local training starts.

### Browser Peer

The browser path is the intended product surface for browser operators.

Build the WebGPU browser client:

```bash
cargo build -p burn_dragon_p2p \
  --target wasm32-unknown-unknown \
  --no-default-features \
  --features wasm-ui,wasm-peer,wgpu
```

Render [DragonBrowserApp](src/wasm/mod.rs) from your Dioxus host and point it at the edge:

```rust
use burn_dragon_p2p::config::{DragonBrowserAppConfig, DragonPeerNetworkConfig};
use burn_dragon_p2p::wasm::{DragonBrowserApp, DragonBrowserAppProps};

let config = DragonBrowserAppConfig {
    network: DragonPeerNetworkConfig::default()
        .with_edge_base_url(Some(std::env::var("MAINNET_EDGE_URL").unwrap()))
        .with_seed_node_urls(None),
    selected_experiment_id: None,
    selected_revision_id: None,
    requested_scopes: Default::default(),
    require_github_auth: true,
    training: None,
};
```

At runtime:

1. open the browser app
2. connect to `MAINNET_EDGE_URL`
3. complete the GitHub login flow
4. resolve the selected experiment from the network directory
5. join as a WebGPU trainer or verifier

The browser app also accepts network overrides from query params:

- `?edge=https://edge.example`
- `?seed=/dnsaddr/seed-1.example/tcp/4001/p2p/...`
- repeated or comma-separated `seed` values

The browser runtime still bootstraps through the edge today, but the seed URL list is carried through the config surface so hardcoded defaults can be added later without changing the shape.

If the selected directory entry includes Dragon profile metadata, browser training can run without a static embedded `training` config in the host app.

### Native Peer

The native join surface is now a real operator binary:

- `burn_dragon_p2p_native resolve-config`
- `burn_dragon_p2p_native assess-capability`
- `burn_dragon_p2p_native begin-github-login`
- `burn_dragon_p2p_native complete-github-login`
- `burn_dragon_p2p_native run-peer`

Build the target you want:

```bash
# CPU
cargo build -p burn_dragon_p2p --features native --bin burn_dragon_p2p_native

# WGPU
cargo build -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native

# CUDA
cargo build -p burn_dragon_p2p --features native,cuda --bin burn_dragon_p2p_native
```

Start from the example config in [deploy/native-peer.toml.example](deploy/native-peer.toml.example).

Resolve the config against a specific network before launching:

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  resolve-config \
  --config path/to/peer.toml \
  --edge-url "$MAINNET_EDGE_URL" \
  --seed-node-url "/dnsaddr/seed-1.example/tcp/4001/p2p/..." \
  --seed-node-url "/dnsaddr/seed-2.example/tcp/4001/p2p/..."
```

That resolves the effective edge URL and seed node set without hardcoding defaults into the crate. The same override surface is used by `run-peer`.

If the selected directory entry includes Dragon profile metadata, native peers can leave `training_config_paths` empty and let the network-provided profile materialize the training config locally under the peer storage root.

Inspect the preflight capability decision before launching:

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  assess-capability \
  --config path/to/peer.toml \
  --experiment-kind nca \
  --backend wgpu \
  --native-wgpu-memory-budget-mib 6144 \
  --output-format json
```

Useful override flags for both `resolve-config` and `assess-capability`:

- `--native-cpu-memory-budget-mib`
- `--native-wgpu-memory-budget-mib`
- `--native-cuda-memory-budget-mib`
- `--browser-wgpu-memory-budget-mib`
- `--no-native-validator-fallback`
- `--no-browser-verifier-fallback`

Provision GitHub auth:

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  begin-github-login \
  --config path/to/peer.toml \
  --experiment-kind nca \
  --backend wgpu \
  --edge-url "$MAINNET_EDGE_URL" \
  --pending-out /var/lib/burn_dragon_p2p/pending-login.json
```

That prints the authorize URL when the edge is using the interactive flow and writes the pending login state to disk.

Complete the login after you have the provider code:

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  complete-github-login \
  --config path/to/peer.toml \
  --pending /var/lib/burn_dragon_p2p/pending-login.json \
  --provider-code "$GITHUB_PROVIDER_CODE" \
  --auth-bundle-out /var/lib/burn_dragon_p2p/auth-bundle.json
```

Run the long-lived peer:

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  run-peer \
  --config path/to/peer.toml \
  --experiment-kind nca \
  --backend wgpu \
  --auth-bundle /var/lib/burn_dragon_p2p/auth-bundle.json \
  --status-interval-secs 30
```

`run-peer` installs a Ctrl-C handler, requests upstream shutdown, and waits for the runtime to exit cleanly instead of dropping detached background work.

There is also a deploy example systemd unit in [deploy/burn-dragon-p2p-native.service](deploy/burn-dragon-p2p-native.service).

If a native trainer failed at runtime and you want to inspect or override the persisted downgrade state, the helper binary also supports:

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  mark-runtime-failure \
  --config path/to/peer.toml \
  --experiment-kind nca \
  --backend wgpu \
  --reason "out of memory allocating optimizer state"
```

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  clear-downgrade \
  --config path/to/peer.toml \
  --experiment-kind nca \
  --backend wgpu
```

For downstream native launchers, the library still exposes the managed runtime seam that the operator binary itself uses:

- [spawn_prepared_native_peer](src/native_runtime.rs)
- [ManagedRunningNativePeer](src/native_runtime.rs)

## Dynamic Experiment Admin

The deployed bootstrap can publish updated Dragon experiment profiles without forcing peers to ship a new static config.

The secure admin path is:

1. deploy the network with explicit GitHub admin logins
2. authenticate through the normal GitHub login flow
3. use the returned session-backed auth bundle for admin actions
4. roll updated directory entries through `RolloutAuthPolicy`

Generate a network-publishable Dragon profile from a local training config:

```bash
cargo run -p burn_dragon_p2p --features native --bin burn_dragon_p2p_native -- \
  build-profile \
  --training-config crates/burn_dragon_p2p/deploy/profiles/nca-r1.training.toml \
  --experiment-kind nca \
  --output /tmp/nca-r2.profile.json
```

Inspect the current network directory:

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  admin-export-directory \
  --edge-url "$MAINNET_EDGE_URL"
```

Roll a replacement directory entry from a local Dragon config:

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  admin-rollout-profile \
  --config path/to/native-peer.toml \
  --experiment-kind nca \
  --backend wgpu \
  --auth-bundle /var/lib/burn_dragon_p2p/auth-bundle.json
```

The rollout is session-authenticated. There is intentionally no deploy-time shared admin token in the production path.

## Build And Validation Harness

Install the local task runner:

```bash
cargo install --path xtask --force
```

Build coverage for the peer targets:

```bash
xtask build-native
xtask build-native-wgpu
xtask build-native-cuda
xtask build-browser-cpu
xtask build-browser
xtask build-matrix
```

Validation ladder:

- `xtask smoke`
  - native WGPU smoke for:
    - NCA shard export + leased training windows
    - ClimbMix existing-shard multi-peer windows
    - browser/native manifest conformance on the same experiment net
  - real browser wasm smoke in headless Chrome/WebGPU via `wasm-bindgen-test-runner`
  - native CUDA build surface check
- `xtask mixed-fleet`
  - browser/native same-net mixed-fleet soak for:
    - NCA native windows plus browser trainer/verifier receipt cycles
    - ClimbMix multi-peer native windows plus browser trainer/verifier receipt cycles
  - ignored medium mixed-fleet rung for both experiments
- `xtask edge-drill`
  - local HTTP edge drill for both experiments
  - real native GitHub-style login + enrollment
  - real browser GitHub-style login + enrollment
  - session-gated directory access
  - browser training and validation receipt submission/ack against the same edge
- `xtask all`
  - build matrix
  - smoke
  - medium native scale rung
  - mixed-fleet smoke + scale rung
  - large native scale rung
  - edge-backed deployment rung

The wasm/browser smoke specifically covers:

- generated NCA training
- HTTP JSON shard training
- real Chrome + chromedriver execution with WebGPU flags
