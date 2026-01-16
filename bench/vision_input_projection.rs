#![recursion_limit = "256"]

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

#[cfg(all(feature = "train", feature = "benchmark"))]
mod projection_bench {
    use super::*;
    use burn::tensor::backend::Backend as BackendTrait;
    use burn_dragon_hatchling::{
        VisionSaccadeInputProjectionConfig, VisionSaccadeInputProjectionCnnConfig,
        VisionSaccadeInputProjectionMicroVitConfig,
        vision::train::bench::VisionInputProjectionBench,
    };
    use burn_ndarray::NdArray;
    use std::hint::black_box;

    const PATCH_SIZES: [usize; 4] = [16, 32, 64, 128];

    fn token_count(image_size: usize, patch_size: usize) -> usize {
        let patch_size = patch_size.max(1);
        let grid = (image_size / patch_size).max(1);
        grid * grid
    }

    fn config_linear() -> VisionSaccadeInputProjectionConfig {
        VisionSaccadeInputProjectionConfig::Linear
    }

    fn config_cnn() -> VisionSaccadeInputProjectionConfig {
        VisionSaccadeInputProjectionConfig::Cnn(
            VisionSaccadeInputProjectionCnnConfig::default(),
        )
    }

    fn config_micro_vit() -> VisionSaccadeInputProjectionConfig {
        VisionSaccadeInputProjectionConfig::RadialMicroVit(
            VisionSaccadeInputProjectionMicroVitConfig::default(),
        )
    }

    pub fn vision_input_projection_bench(c: &mut Criterion) {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let embed_dim = 256usize;
        let image_size = 128usize;
        let batch = 8usize;

        let configs: [(&str, fn() -> VisionSaccadeInputProjectionConfig); 3] = [
            ("linear", config_linear),
            ("cnn", config_cnn),
            ("radial_micro_vit", config_micro_vit),
        ];

        let mut group = c.benchmark_group("vision_input_projection");
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(2));
        group.sample_size(10);

        for &patch_size in &PATCH_SIZES {
            let tokens = token_count(image_size, patch_size);
            for (name, make_config) in configs {
                let config = make_config();
                let bench = VisionInputProjectionBench::<Backend>::new(
                    embed_dim,
                    patch_size,
                    tokens,
                    batch,
                    config,
                    &device,
                );
                let param_count = bench.param_count();
                println!(
                    "vision_input_projection/{name}/p{patch_size}: params={param_count}"
                );
                group.bench_function(BenchmarkId::new(name, patch_size), |b| {
                    b.iter(|| black_box(bench.forward()));
                });
            }
        }

        group.finish();
    }
}

#[cfg(all(feature = "train", feature = "benchmark"))]
criterion_group!(benches, projection_bench::vision_input_projection_bench);
#[cfg(all(feature = "train", feature = "benchmark"))]
criterion_main!(benches);

#[cfg(not(all(feature = "train", feature = "benchmark")))]
fn main() {
    eprintln!("vision_input_projection benchmarks require --features train,benchmark");
}
