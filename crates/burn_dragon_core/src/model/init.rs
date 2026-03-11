use burn::module::Initializer;

const CONTROLLED_INIT_STD_CAP: f64 = 0.02;

pub fn near_critical_embedding_std(width: usize) -> f64 {
    (1.0 / (width.max(1) as f64).sqrt()).min(CONTROLLED_INIT_STD_CAP)
}

pub fn near_critical_projection_std(fan_in: usize, fan_out: usize) -> f64 {
    (1.0 / ((fan_in.max(1) + fan_out.max(1)) as f64).sqrt()).min(CONTROLLED_INIT_STD_CAP)
}

pub fn near_critical_residual_output_std(
    fan_in: usize,
    fan_out: usize,
    residual_depth: usize,
) -> f64 {
    let base = 1.0 / ((fan_in.max(1) + fan_out.max(1)) as f64).sqrt();
    (base / (residual_depth.max(1) as f64).sqrt()).min(CONTROLLED_INIT_STD_CAP)
}

pub fn near_critical_embedding_initializer(width: usize) -> Initializer {
    Initializer::Normal {
        mean: 0.0,
        std: near_critical_embedding_std(width),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_std_caps_small_models_and_scales_large_ones() {
        assert!((near_critical_embedding_std(256) - 0.02).abs() < 1e-12);
        assert!((near_critical_embedding_std(4096) - 0.015625).abs() < 1e-12);
    }

    #[test]
    fn projection_std_caps_small_models_and_scales_large_ones() {
        assert!((near_critical_projection_std(64, 64) - 0.02).abs() < 1e-12);
        assert!((near_critical_projection_std(2048, 2048) - 0.015625).abs() < 1e-12);
    }

    #[test]
    fn residual_output_std_shrinks_with_depth() {
        let shallow = near_critical_residual_output_std(2048, 2048, 1);
        let deep = near_critical_residual_output_std(2048, 2048, 16);
        assert!((deep * 4.0 - shallow).abs() < 1e-12);
    }
}
