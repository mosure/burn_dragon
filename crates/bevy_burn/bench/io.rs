use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use bevy::{asset::RenderAssetUsages, prelude::*, render::render_resource::*};
use burn::tensor::Tensor;
use burn_wgpu::Wgpu;

use bevy_burn::{BevyBurnBridgePlugin, BevyBurnHandle, BindingDirection, TransferKind};

type BurnBackend = Wgpu<f32, i32>;

const SIZE: u32 = 512;

fn default_app() -> App {
    let mut app = App::new();

    app.add_plugins(MinimalPlugins);
    app.insert_resource(Assets::<Image>::default());
    app.add_plugins(BevyBurnBridgePlugin::<BurnBackend> {
        cpu_only: true,
        ..default()
    });

    app
}

fn make_image() -> Image {
    let mut img = Image::new_fill(
        Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    img.texture_descriptor.usage |= TextureUsages::COPY_DST | TextureUsages::COPY_SRC;
    img
}

fn bench_burn_to_bevy_cpu(crit: &mut Criterion) {
    let mut app = default_app();

    app.add_systems(
        Startup,
        |mut cmds: Commands, mut images: ResMut<Assets<Image>>| {
            let handle = images.add(make_image());
            let tensor = Tensor::<BurnBackend, 3>::zeros(
                [SIZE as usize, SIZE as usize, 4],
                &Default::default(),
            );
            cmds.spawn(BevyBurnHandle {
                bevy_image: handle,
                tensor,
                upload: true,
                direction: BindingDirection::BurnToBevy,
                xfer: TransferKind::Cpu,
            });
        },
    );

    app.add_systems(
        PostUpdate,
        |mut query: Query<&mut BevyBurnHandle<BurnBackend>>| {
            for mut handle in query.iter_mut() {
                handle.upload = true;
            }
        },
    );

    app.update();

    let mut group = crit.benchmark_group("burn_to_bevy");
    group.throughput(Throughput::Elements(1));

    group.bench_function(BenchmarkId::new("tensor_to_image", SIZE), |b| {
        b.iter(|| {
            app.update();
        });
    });
    group.finish();
}

fn bench_bevy_to_burn_cpu(crit: &mut Criterion) {
    let mut app = default_app();

    app.add_systems(
        Startup,
        |mut cmds: Commands, mut images: ResMut<Assets<Image>>| {
            let handle = images.add(make_image());
            let tensor = Tensor::<BurnBackend, 3>::zeros(
                [SIZE as usize, SIZE as usize, 4],
                &Default::default(),
            );
            cmds.spawn(BevyBurnHandle {
                bevy_image: handle,
                tensor,
                upload: true,
                direction: BindingDirection::BevyToBurn,
                xfer: TransferKind::Cpu,
            });
        },
    );

    app.add_systems(
        PostUpdate,
        |mut query: Query<&mut BevyBurnHandle<BurnBackend>>| {
            for mut handle in query.iter_mut() {
                handle.upload = true;
            }
        },
    );

    app.update();

    let mut group = crit.benchmark_group("bevy_to_burn");
    group.throughput(Throughput::Elements(1));

    group.bench_function(BenchmarkId::new("image_to_tensor", SIZE), |b| {
        b.iter(|| {
            app.update();
        });
    });
    group.finish();
}

// TODO: fix main thread issues for gpu benchmarking/tests
// fn bench_burn_to_bevy_gpu(crit: &mut Criterion) {
//     let mut app = default_app();

//     app.add_systems(
//         Startup,
//         |
//                     mut cmds: Commands,
//                     mut images: ResMut<Assets<Image>>,
//                 | {
//                     let handle = images.add(make_image());
//                     let tensor = Tensor::<BurnBackend, 3>::zeros([
//                             SIZE as usize,
//                             SIZE as usize,
//                             4,
//                         ],
//                         &Default::default(),
//                     );
//                     cmds.spawn(BevyBurnHandle {
//                         bevy_image: handle,
//                         tensor,
//                         upload: true,
//                         direction: BindingDirection::BurnToBevy,
//                         xfer: TransferKind::Gpu,
//                     });
//                 },
//     );

//     app.add_systems(
//         PostUpdate,
//         |mut query: Query<&mut BevyBurnHandle<BurnBackend>>| {
//             for mut handle in query.iter_mut() {
//                 handle.upload = true;
//             }
//         },
//     );

//     app.update();

//     let mut group = crit.benchmark_group("burn_to_bevy");
//     group.throughput(Throughput::Elements(1));

//     group.bench_function(BenchmarkId::new("tensor_to_image", SIZE), |b| {
//         b.iter(|| {
//             app.update();
//         });
//     });
//     group.finish();
// }

// fn bench_bevy_to_burn_gpu(crit: &mut Criterion) {
//     let mut app = default_app();

//     app.add_systems(
//         Startup,
//         |
//                     mut cmds: Commands,
//                     mut images: ResMut<Assets<Image>>,
//                 | {
//                     let handle = images.add(make_image());
//                     let tensor = Tensor::<BurnBackend, 3>::zeros([
//                             SIZE as usize,
//                             SIZE as usize,
//                             4,
//                         ],
//                         &Default::default(),
//                     );
//                     cmds.spawn(BevyBurnHandle {
//                         bevy_image: handle,
//                         tensor,
//                         upload: true,
//                         direction: BindingDirection::BevyToBurn,
//                         xfer: TransferKind::Gpu,
//                     });
//                 },
//     );

//     app.add_systems(
//         PostUpdate,
//         |mut query: Query<&mut BevyBurnHandle<BurnBackend>>| {
//             for mut handle in query.iter_mut() {
//                 handle.upload = true;
//             }
//         },
//     );

//     app.update();

//     let mut group = crit.benchmark_group("bevy_to_burn");
//     group.throughput(Throughput::Elements(1));

//     group.bench_function(BenchmarkId::new("image_to_tensor", SIZE), |b| {
//         b.iter(|| {
//             app.update();
//         });
//     });
//     group.finish();
// }

criterion_group! {
    name = io_benches;
    config = Criterion::default().sample_size(10);
    targets = bench_burn_to_bevy_cpu,
              bench_bevy_to_burn_cpu,
            //   bench_burn_to_bevy_gpu,
            //   bench_bevy_to_burn_gpu,
}
criterion_main!(io_benches);
