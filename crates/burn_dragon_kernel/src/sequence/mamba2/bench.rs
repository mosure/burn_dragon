#[derive(Clone, Copy, Debug)]
pub struct Mamba2BenchCase {
    pub batch: usize,
    pub time: usize,
    pub d_model: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub expand: usize,
    pub headdim: usize,
    pub ngroups: usize,
}

pub const LARGE_RUNG_CASE: Mamba2BenchCase = Mamba2BenchCase {
    batch: 1,
    time: 256,
    d_model: 1024,
    d_state: 64,
    d_conv: 4,
    expand: 2,
    headdim: 128,
    ngroups: 1,
};
