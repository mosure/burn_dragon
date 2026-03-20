//! Shared CLI library surface for Dragon benches and wrappers.

#[cfg(all(feature = "benchmark", feature = "train"))]
/// Benchmark/probe helpers shared by the CLI binaries.
pub mod bench;
