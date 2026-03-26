//! Shared CLI library surface for Dragon benches and wrappers.

#[cfg(all(feature = "benchmark", feature = "train"))]
/// Benchmark/probe helpers shared by the CLI binaries.
pub mod bench;

#[cfg(feature = "language-train")]
/// Shared TTCL runner used by both the legacy language-named bin and the reasoning-owned bin.
pub mod reasoning_ttcl;
