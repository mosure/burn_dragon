# bevy_burn 🕊️🔥

[![GitHub License](https://img.shields.io/github/license/mosure/bevy_burn)](https://raw.githubusercontent.com/mosure/bevy_burn/main/LICENSE)
[![crates.io](https://img.shields.io/crates/v/bevy_burn.svg)](https://crates.io/crates/bevy_burn)

bevy burn data bridge plugin


## features
- [ ] bevy texture -> burn tensor
- [x] burn tensor -> bevy texture


## usage

### burn -> bevy gpu example

`cargo run --bin gpu_interop`

```rust
use bevy::prelude::*;
use bevy::{
    asset::RenderAssetUsages,
    render::render_resource::*,
};
use burn_core::tensor::Int;
use burn_wgpu::Wgpu as BurnWgpu;
use bevy_burn::*;

type BB = BurnWgpu<f32, i32>;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(BevyBurnBridgePlugin::<BB>::default())
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut cmds: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    let size = Extent3d {
        width: 256,
        height: 256,
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
    let dev = Default::default();
    let xs = burn_core::tensor::Tensor::<BB, 1, Int>::arange(0..(w * h * 4) as i64, &dev)
        .float()
        .div_scalar((w * h * 4) as f32);
    let rgba = xs.reshape([h, w, 4]);

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
    ));

    cmds.spawn(Camera2d);
}
```


## compatible versions

| `bevy_burn` | `bevy`  | `burn` |
| :--         | :--     | :--    |
| `0.4`       | `0.17`  | `0.19` |
| `0.3`       | `0.17-dev*` | `0.18` |

> *`wgpu` version must match


## license
licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.


## analytics
![alt](https://repobeats.axiom.co/api/embed/0a4b4e14072c91c5a971db920bd9a3df0e430a65.svg "analytics")
