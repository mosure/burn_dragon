#[derive(Clone, Copy, Debug)]
pub struct MambaBenchCase {
    pub batch: usize,
    pub time: usize,
    pub d_model: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub expand: usize,
}

pub const LARGE_RUNG_CASE: MambaBenchCase = MambaBenchCase {
    batch: 1,
    time: 256,
    d_model: 1024,
    d_state: 16,
    d_conv: 4,
    expand: 2,
};
