#[cfg(not(feature = "benchmark"))]
fn main() {
    panic!("video_lejepa_stageaware_bench requires --features benchmark");
}

#[cfg(feature = "benchmark")]
fn main() {
    burn_dragon_cli::bench::video_lejepa_stageaware::real::main();
}
