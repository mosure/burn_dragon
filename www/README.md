# burn_dragon web

- `cargo build --target wasm32-unknown-unknown --release --no-default-features --features web`
- `wasm-bindgen --out-dir ./www/out/ --target web ./target/wasm32-unknown-unknown/release/burn_dragon.wasm`
