#[derive(Clone, Copy, Debug)]
pub struct Rwkv8BenchCase {
    pub batch: usize,
    pub heads: usize,
    pub time: usize,
    pub latent: usize,
    pub embd: usize,
}

pub const LARGE_RUNG_CASE: Rwkv8BenchCase = Rwkv8BenchCase {
    batch: 1,
    heads: 8,
    time: 256,
    latent: 8192,
    embd: 1024,
};
