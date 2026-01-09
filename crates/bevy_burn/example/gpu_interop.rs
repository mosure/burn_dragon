#![recursion_limit = "256"]

use bevy::prelude::*;
use bevy::{asset::RenderAssetUsages, render::render_resource::*};
use bevy_burn::*;
use burn_core::tensor::{Int, Tensor};
use burn_wgpu::Wgpu as BurnWgpu;

type BB = BurnWgpu<f32, i32>;

const SIZE: u32 = 1024;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(BevyBurnBridgePlugin::<BB>::default())
        .add_systems(Startup, setup)
        .add_systems(Update, animate_plasma)
        .run();
}

#[derive(Component)]
struct Plasma {
    x: Tensor<BB, 2>,
    y: Tensor<BB, 2>,
    a: Tensor<BB, 2>,
}

fn setup(mut cmds: Commands, mut images: ResMut<Assets<Image>>, burn: Res<BurnDevice>) {
    let Some(dev) = burn.device() else {
        return;
    };
    let size = Extent3d {
        width: SIZE,
        height: SIZE,
        depth_or_array_layers: 1,
    };
    let mut img = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0; 16],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    img.texture_descriptor.usage |= TextureUsages::COPY_SRC
        | TextureUsages::COPY_DST
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::STORAGE_BINDING;
    let handle = images.add(img);

    let h = size.height as usize;
    let w = size.width as usize;
    let xs = Tensor::<BB, 1, Int>::arange(0..w as i64, dev)
        .float()
        .div_scalar((w - 1) as f32)
        .mul_scalar(2.0)
        .add_scalar(-1.0)
        .reshape([1, w]); // [1, w]
    let ys = Tensor::<BB, 1, Int>::arange(0..h as i64, dev)
        .float()
        .div_scalar((h - 1) as f32)
        .mul_scalar(2.0)
        .add_scalar(-1.0)
        .reshape([h, 1]); // [h, 1]
    let x = Tensor::<BB, 2>::zeros([h, 1], dev) + xs; // [h, w]
    let y = ys + Tensor::<BB, 2>::zeros([1, w], dev); // [h, w]
    let a1 = Tensor::<BB, 2>::ones([h, w], dev);
    let boot_r = x
        .clone()
        .mul_scalar(8.0)
        .sin()
        .mul_scalar(0.5)
        .add_scalar(0.5);
    let boot_g = x
        .clone()
        .add(y.clone())
        .mul_scalar(6.0)
        .sin()
        .mul_scalar(0.5)
        .add_scalar(0.5);
    let boot_b = y
        .clone()
        .mul_scalar(8.0)
        .sin()
        .mul_scalar(0.5)
        .add_scalar(0.5);
    let rgba = Tensor::<BB, 2>::stack(vec![boot_r, boot_g, boot_b, a1.clone()], 2);

    cmds.spawn((
        ImageNode {
            image: handle.clone(),
            ..default()
        },
        BevyBurnHandle::<BB> {
            bevy_image: handle,
            tensor: rgba,
            upload: true,
            direction: BindingDirection::BurnToBevy,
            xfer: TransferKind::Gpu,
        },
        Plasma { x, y, a: a1 },
    ));

    cmds.spawn(Camera2d);
}

fn animate_plasma(time: Res<Time>, mut q: Query<(&Plasma, &mut BevyBurnHandle<BB>)>) {
    let t = time.elapsed_secs();
    for (p, mut h) in &mut q {
        let r =
            p.x.clone()
                .mul_scalar(8.0)
                .add_scalar(t * 1.00)
                .sin()
                .mul_scalar(0.5)
                .add_scalar(0.5);
        let g =
            p.x.clone()
                .add(p.y.clone())
                .mul_scalar(6.0)
                .add_scalar(t * 1.37)
                .sin()
                .mul_scalar(0.5)
                .add_scalar(0.5);
        let b =
            p.y.clone()
                .mul_scalar(9.0)
                .add_scalar(-t * 0.73)
                .sin()
                .mul_scalar(0.5)
                .add_scalar(0.5);
        let rgba = Tensor::<BB, 2>::stack(vec![r, g, b, p.a.clone()], 2);
        h.tensor = rgba;
        h.upload = true;
    }
}
