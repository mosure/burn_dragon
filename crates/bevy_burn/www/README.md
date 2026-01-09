# bevy_burn for web

## wasm support

1. Make sure the `wasm32-unknown-unknown` rustup target is installed:

   ```bash
   rustup target add wasm32-unknown-unknown
   ```

2. If you normally pass host-specific `RUSTFLAGS` (for example `-C target-cpu=native`), clear them for the wasm build so the compiler does not try to enable unsupported CPU features:

   ```bash
   RUSTFLAGS="" cargo build --example gpu_interop --target wasm32-unknown-unknown --release
   ```

3. Install a `wasm-bindgen-cli` version that matches the crate we depend on (currently `0.2.105`):

   ```bash
   cargo install -f wasm-bindgen-cli --version 0.2.105
   ```

4. Generate the JS/TypeScript bindings into `www/out/`:

   ```bash
   wasm-bindgen --out-dir ./www/out/ --target web ./target/wasm32-unknown-unknown/release/examples/gpu_interop.wasm
   ```
