#![recursion_limit = "256"]

use std::marker::PhantomData;

use bevy::{
    asset::{Handle, RenderAssetUsages},
    prelude::*,
    render::{
        render_asset::RenderAssets,
        render_resource::*,
        renderer::{
            RenderAdapter, RenderAdapterInfo, RenderDevice, RenderInstance, RenderQueue,
            WgpuWrapper,
        },
        sync_world::{RenderEntity, SyncToRenderWorld},
        texture::GpuImage,
        Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    },
};
#[cfg(target_family = "wasm")]
use bevy::tasks::{IoTaskPool, Task};
use burn::{
    prelude::Backend,
    tensor::{Int, Tensor, TensorData},
};
use burn_wgpu::{
    init_device as init_burn_device, RuntimeOptions as BurnRuntimeOptions,
    WgpuDevice as BurnWgpuDevice, WgpuSetup as BurnWgpuSetup,
};
#[cfg(target_family = "wasm")]
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
};

pub mod gpu_burn_to_bevy;
use gpu_burn_to_bevy::{BurnBevyPrepare, GpuBurnToBevyPlugin};

#[derive(Resource, Clone, Debug, Default)]
pub struct BurnDevice {
    inner: Option<BurnWgpuDevice>,
}

impl BurnDevice {
    pub fn pending() -> Self {
        Self { inner: None }
    }

    pub fn ready(device: BurnWgpuDevice) -> Self {
        Self {
            inner: Some(device),
        }
    }

    pub fn device(&self) -> Option<&BurnWgpuDevice> {
        self.inner.as_ref()
    }

    pub fn device_mut(&mut self) -> Option<&mut BurnWgpuDevice> {
        self.inner.as_mut()
    }

    pub fn is_ready(&self) -> bool {
        self.inner.is_some()
    }
}

impl From<BurnWgpuDevice> for BurnDevice {
    fn from(value: BurnWgpuDevice) -> Self {
        Self::ready(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingDirection {
    BurnToBevy,
    BevyToBurn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferKind {
    Cpu,
    Gpu,
}

#[derive(Component, Clone)]
pub struct BevyBurnHandle<B: Backend> {
    pub bevy_image: Handle<Image>,
    pub tensor: Tensor<B, 3>,
    pub upload: bool,
    pub direction: BindingDirection,
    pub xfer: TransferKind,
}

impl<B: Backend> Default for BevyBurnHandle<B> {
    fn default() -> Self {
        Self {
            bevy_image: Handle::default(),
            tensor: Tensor::<B, 3>::zeros([0, 0, 0], &Default::default()),
            upload: true,
            direction: BindingDirection::BurnToBevy,
            xfer: TransferKind::Cpu,
        }
    }
}

#[derive(Default)]
pub struct BevyBurnBridgePlugin<B: Backend> {
    pub cpu_only: bool,
    pub _marker: PhantomData<B>,
}

impl<B: Backend> Plugin for BevyBurnBridgePlugin<B>
where
    B: Backend + 'static,
    (): BurnBevyPrepare<B>,
{
    fn build(&self, app: &mut App) {
        // cpu path in main world
        app.add_systems(Update, (bevy_to_burn_update::<B>, burn_to_bevy_update::<B>));

        if self.cpu_only {
            return;
        }

        app.add_systems(First, ensure_sync_to_render_world::<B>);
    }

    fn finish(&self, app: &mut App) {
        if self.cpu_only {
            return;
        }

        let render_app = app
            .get_sub_app_mut(RenderApp)
            .expect("Failed to setup Burn plugin: RenderApp not found");

        let burn_device = {
            let bevy_adapter = render_app.world().resource::<RenderAdapter>();
            let wgpu_adapter = unwrap_wgpu_wrapper(&bevy_adapter.0);

            let bevy_device = render_app.world().resource::<RenderDevice>();
            let wgpu_device = bevy_device.wgpu_device().clone();

            let bevy_instance = render_app.world().resource::<RenderInstance>();
            let wgpu_instance = unwrap_wgpu_wrapper(&bevy_instance.0);

            let bevy_queue = render_app.world().resource::<RenderQueue>();
            let wgpu_queue = unwrap_wgpu_wrapper(&bevy_queue.0);

            let render_adapter_info = render_app.world().resource::<RenderAdapterInfo>();
            let wgpu_backend = render_adapter_info.backend;

            let wgpu_setup = BurnWgpuSetup {
                adapter: wgpu_adapter,
                device: wgpu_device,
                instance: wgpu_instance,
                queue: wgpu_queue,
                backend: wgpu_backend,
            };

            let runtime_options = BurnRuntimeOptions::default();
            let burn_device = init_burn_device(wgpu_setup, runtime_options);

            render_app
                .add_systems(ExtractSchedule, extract_gpu_handles::<B>)
                .add_systems(Render, gpu_bevy_to_burn::<B>.in_set(RenderSystems::Queue));

            burn_device
        };

        let burn_device = BurnDevice::from(burn_device);
        render_app.insert_resource(burn_device.clone());
        app.insert_resource(burn_device);

        app.add_plugins(GpuBurnToBevyPlugin::<B>::default());
    }
}

/// make sure entities we care about are synced into the render world
fn ensure_sync_to_render_world<B: Backend>(
    mut commands: Commands,
    q: Query<(Entity, Option<&SyncToRenderWorld>), With<BevyBurnHandle<B>>>,
) {
    for (e, synced) in &q {
        if synced.is_none() {
            commands.entity(e).insert(SyncToRenderWorld);
        }
    }
}

fn unwrap_wgpu_wrapper<T: Clone>(wrapper: &WgpuWrapper<T>) -> T {
    <WgpuWrapper<T> as Clone>::clone(wrapper).into_inner()
}

// ---------- cpu path ----------

fn bevy_to_burn_update<B: Backend>(
    images: Res<Assets<Image>>,
    mut q: Query<&mut BevyBurnHandle<B>>,
) {
    for mut handle in &mut q {
        if handle.direction != BindingDirection::BevyToBurn || handle.xfer != TransferKind::Cpu {
            continue;
        }

        if handle.upload {
            let Some(img) = images.get(&handle.bevy_image) else {
                continue;
            };
            let size = img.size();
            let (width, height) = (size.x as usize, size.y as usize);
            let Some(raw) = &img.data else { continue };
            if raw.len() != width * height * 4 {
                continue;
            }

            let device = handle.tensor.device();

            let raw_tensor = Tensor::<B, 1, Int>::from_data(&raw[..], &device);
            let float_tensor = raw_tensor.float();
            let normalised = float_tensor.div_scalar(255.0);
            let new_tensor = normalised.reshape([height, width, 4]);

            handle.tensor = new_tensor;
            handle.upload = false;
        }
    }
}

#[cfg(not(target_family = "wasm"))]
fn burn_to_bevy_update<B: Backend>(
    mut images: ResMut<Assets<Image>>,
    mut q: Query<&mut BevyBurnHandle<B>>,
) {
    for mut handle in &mut q {
        if handle.direction != BindingDirection::BurnToBevy || handle.xfer != TransferKind::Cpu {
            continue;
        }

        if handle.upload {
            let data = handle.tensor.to_data();
            if let Some(bytes) = tensor_data_to_rgba_bytes(&data) {
                write_tensor_bytes_to_image(&mut handle, &mut images, bytes);
                handle.upload = false;
            }
        }
    }
}

fn tensor_data_to_rgba_bytes(data: &TensorData) -> Option<Vec<u8>> {
    let floats = data.to_vec::<f32>().ok()?;
    let mut bytes = Vec::with_capacity(floats.len());
    for value in floats {
        let v = value.clamp(0.0, 1.0) * 255.0;
        bytes.push(v.round() as u8);
    }
    Some(bytes)
}

fn write_tensor_bytes_to_image<B: Backend>(
    handle: &mut BevyBurnHandle<B>,
    images: &mut ResMut<Assets<Image>>,
    bytes: Vec<u8>,
) {
    let height = handle.tensor.shape().dims[0] as u32;
    let width = handle.tensor.shape().dims[1] as u32;

    if let Some(img) = images.get_mut(&handle.bevy_image) {
        if img.height() != height || img.width() != width {
            info!(
                "resizing image from {}x{} to {}x{}",
                img.width(),
                img.height(),
                width,
                height
            );

            img.resize(Extent3d {
                height,
                width,
                depth_or_array_layers: 1,
            });
        }

        match img.data {
            Some(ref mut d) => {
                if d.len() == bytes.len() {
                    d.copy_from_slice(&bytes);
                } else {
                    *d = bytes;
                }
            }
            None => img.data = Some(bytes),
        }
    } else {
        let img = Image::new_fill(
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &bytes,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        );
        handle.bevy_image = images.add(img);
    }
}

#[cfg(target_family = "wasm")]
#[derive(Component)]
struct PendingTensorDownload {
    task: Task<TensorData>,
}

#[cfg(target_family = "wasm")]
fn burn_to_bevy_update<B: Backend>(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut q: Query<(Entity, &mut BevyBurnHandle<B>, Option<&mut PendingTensorDownload>)>,
) {
    for (entity, mut handle, pending) in &mut q {
        let is_cpu_download = handle.direction == BindingDirection::BurnToBevy
            && handle.xfer == TransferKind::Cpu;
        if !is_cpu_download {
            if pending.is_some() {
                commands
                    .entity(entity)
                    .remove::<PendingTensorDownload>();
            }
            continue;
        }

        if !handle.upload {
            if pending.is_some() {
                commands
                    .entity(entity)
                    .remove::<PendingTensorDownload>();
            }
            continue;
        }

        if let Some(mut pending_task) = pending {
            if let Some(data) = poll_task(&mut pending_task.task) {
                commands
                    .entity(entity)
                    .remove::<PendingTensorDownload>();
                if let Some(bytes) = tensor_data_to_rgba_bytes(&data) {
                    write_tensor_bytes_to_image(&mut handle, &mut images, bytes);
                    handle.upload = false;
                }
            }
            continue;
        }

        let tensor = handle.tensor.clone();
        let task = IoTaskPool::get().spawn_local(async move { tensor.into_data_async().await });
        commands
            .entity(entity)
            .insert(PendingTensorDownload { task });
    }
}

#[cfg(target_family = "wasm")]
fn poll_task<T>(task: &mut Task<T>) -> Option<T> {
    let waker = noop_waker();
    let mut ctx = Context::from_waker(&waker);
    match Pin::new(task).poll(&mut ctx) {
        Poll::Ready(res) => Some(res),
        Poll::Pending => None,
    }
}

#[cfg(target_family = "wasm")]
fn noop_waker() -> Waker {
    use std::ptr;

    unsafe fn clone(_: *const ()) -> RawWaker {
        RawWaker::new(ptr::null(), &VTABLE)
    }
    unsafe fn wake(_: *const ()) {}
    unsafe fn wake_by_ref(_: *const ()) {}
    unsafe fn drop(_: *const ()) {}

    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop);
    unsafe { Waker::from_raw(RawWaker::new(ptr::null(), &VTABLE)) }
}

// ---------- gpu path (render world) ----------

#[derive(Component, Clone, Debug)]
struct ExtractedGpuHandle<B: Backend> {
    image: Handle<Image>,
    tensor: Tensor<B, 3>,
    direction: BindingDirection,
    upload: bool,
}

fn extract_gpu_handles<B: Backend>(
    mut commands: Commands,
    q: Extract<Query<(RenderEntity, &BevyBurnHandle<B>)>>,
) {
    let mut seen = 0usize;

    for (render_entity, h) in &q {
        seen += 1;
        if h.xfer != TransferKind::Gpu {
            continue;
        }

        commands
            .entity(render_entity)
            .insert(ExtractedGpuHandle::<B> {
                image: h.bevy_image.clone(),
                tensor: h.tensor.clone(),
                direction: h.direction,
                upload: h.upload,
            });

        seen += 1;
    }

    debug!(
        target: "bevy_burn::extract",
        "extract_gpu_handles: seen={}",
        seen
    );
}

#[inline]
fn padded_bytes_per_row(width: u32, bytes_per_pixel: u32) -> u32 {
    // wgpu COPY_BYTES_PER_ROW_ALIGNMENT is 256
    const ALIGN: u32 = 256;
    let row = width * bytes_per_pixel;
    row.div_ceil(ALIGN) * ALIGN
}

/// bevy image -> burn (gpu-side). schedules copy + readback; blocks to map.
fn gpu_bevy_to_burn<B: Backend>(
    // burn_device: Res<BurnDevice>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    images: Res<RenderAssets<GpuImage>>,
    mut q: Query<&mut ExtractedGpuHandle<B>>,
) {
    for mut h in &mut q {
        if h.direction != BindingDirection::BevyToBurn || !h.upload {
            continue;
        }
        let Some(gpu_image) = images.get(&h.image) else {
            continue;
        };

        let bpp = 4u32; // RGBA8
        let extent = Extent3d {
            width: gpu_image.size.width,
            height: gpu_image.size.height,
            depth_or_array_layers: 1,
        };
        let row_bytes = extent.width * bpp;
        let padded_row = padded_bytes_per_row(extent.width, bpp);
        let total = (padded_row as u64) * (extent.height as u64);

        // staging buffer
        let staging = render_device.create_buffer(&BufferDescriptor {
            label: Some("bevy_burn.gpu_t2b.staging"),
            size: total,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // copy texture -> buffer (wgpu v25)
        let mut enc = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("bevy_burn.copy_tex_to_buf"),
        });
        enc.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &gpu_image.texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &staging,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(extent.height),
                },
            },
            extent,
        );
        render_queue.submit([enc.finish()]);

        // map & normalize to tensor
        staging.slice(..).map_async(MapMode::Read, |_| {});
        let _ = render_device.wgpu_device().poll(PollType::Wait);

        let view = staging.slice(..).get_mapped_range();
        // strip row padding
        let mut compact = Vec::with_capacity((row_bytes as usize) * (extent.height as usize));
        for y in 0..extent.height as usize {
            let src_off = y * padded_row as usize;
            compact.extend_from_slice(&view[src_off..src_off + row_bytes as usize]);
        }
        drop(view);

        let device = h.tensor.device();
        let raw_tensor = Tensor::<B, 1, Int>::from_data(&compact[..], &device);
        let float_tensor = raw_tensor.float();
        let normalised = float_tensor.div_scalar(255.0);
        h.tensor = normalised.reshape([extent.height as usize, extent.width as usize, 4]);

        staging.unmap();
        h.upload = false; // render-world flag
    }
}

// ---------- tests ----------

#[cfg(test)]
mod cpu_tests {
    use super::*;
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
    use burn_wgpu::Wgpu;

    type BurnBackend = Wgpu<f32, i32>;

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

    #[test]
    fn bevy_to_burn_cpu_1x1() {
        let mut app = default_app();

        let pixel = [255, 128, 0, 255];
        let img = Image::new_fill(
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &pixel,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        );
        let handle = {
            let mut images = app.world_mut().resource_mut::<Assets<Image>>();
            images.add(img)
        };

        let tensor = Tensor::<BurnBackend, 3>::zeros([1, 1, 4], &Default::default());
        let entity = app
            .world_mut()
            .spawn(BevyBurnHandle {
                bevy_image: handle.clone(),
                tensor,
                upload: true,
                direction: BindingDirection::BevyToBurn,
                xfer: TransferKind::Cpu,
            })
            .id();

        app.update();
        let comp = app
            .world()
            .get::<BevyBurnHandle<BurnBackend>>(entity)
            .unwrap();
        let data = comp.tensor.to_data();
        let floats: Vec<f32> = data.to_vec::<f32>().unwrap();

        let max_err = pixel
            .iter()
            .enumerate()
            .map(|(i, &x)| (x as f32 / 255.0 - floats[i]).abs())
            .fold(0.0, f32::max);

        assert!(max_err < 0.0001, "max error: {}", max_err);
    }

    #[test]
    fn burn_to_bevy_cpu_1x1() {
        let mut app = default_app();

        let img = Image::new_fill(
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &[0, 0, 0, 0],
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        );
        let handle = {
            let mut images = app.world_mut().resource_mut::<Assets<Image>>();
            images.add(img)
        };

        let tensor =
            Tensor::<BurnBackend, 3>::from_data([[[0.0f32, 0.5, 1.0, 1.0]]], &Default::default());
        app.world_mut().spawn(BevyBurnHandle {
            bevy_image: handle.clone(),
            tensor,
            upload: true,
            direction: BindingDirection::BurnToBevy,
            xfer: TransferKind::Cpu,
        });

        app.update();
        let images = app.world().resource::<Assets<Image>>();
        let updated = images.get(&handle).unwrap();
        assert_eq!(updated.data.as_deref().unwrap(), &[0, 128, 255, 255]);
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;
    use bevy::{
        asset::Handle,
        ecs::system::SystemState,
        render::{
            render_asset::RenderAssets,
            render_resource::{
                BindGroupLayout, BindGroupLayoutEntry, BindingType, BufferBindingType,
                BufferDescriptor, BufferUsages, ComputePipeline, Extent3d, Origin3d,
                ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess,
                TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
                TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
                TextureViewDescriptor, TextureViewDimension,
            },
            renderer::{RenderDevice, RenderQueue, WgpuWrapper},
            texture::GpuImage,
        },
    };
    use burn_wgpu::{
        init_device as init_burn_device, RuntimeOptions as BurnRuntimeOptions,
        WgpuSetup as BurnWgpuSetup,
    };
    use burn_wgpu::Wgpu as BurnBackend;
    use futures::executor::block_on;
    use std::sync::Arc;
    use wgpu::{
        util::TextureDataOrder, CommandEncoderDescriptor, ComputePassDescriptor, DeviceDescriptor,
        Features, MapMode, PipelineCompilationOptions, PollType, PowerPreference,
        RequestAdapterOptions,
    };
    struct TestGpuContext {
        render_device: RenderDevice,
        render_queue: RenderQueue,
        burn_device: BurnDevice,
        device: wgpu::Device,
    }

    impl TestGpuContext {
        fn new() -> Self {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
            let adapter = block_on(instance.request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            }))
            .expect("No suitable GPU adapter found for tests");
            let adapter_info = adapter.get_info();
            let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
                label: Some("bevy_burn_test_device"),
                required_features: Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
            }))
            .expect("Failed to create test wgpu device");

            let render_device = RenderDevice::new(WgpuWrapper::new(device.clone()));
            let render_queue = RenderQueue(Arc::new(WgpuWrapper::new(queue.clone())));

            let burn_setup = BurnWgpuSetup {
                adapter: adapter.clone(),
                device: device.clone(),
                instance: instance.clone(),
                queue: queue.clone(),
                backend: adapter_info.backend,
            };
            let burn_device =
                BurnDevice::from(init_burn_device(burn_setup, BurnRuntimeOptions::default()));

            Self {
                render_device,
                render_queue,
                burn_device,
                device,
            }
        }

        fn poll_wait(&self) {
            let _ = self.device.poll(PollType::Wait);
        }
    }

    fn create_rgba_pipeline(device: &RenderDevice) -> (BindGroupLayout, ComputePipeline) {
        let layout = device.create_bind_group_layout(
            "buffer_to_rgba32f layout",
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        );

        let shader = device
            .wgpu_device()
            .create_shader_module(ShaderModuleDescriptor {
                label: Some("buffer_to_rgba32f shader"),
                source: ShaderSource::Wgsl(include_str!("buffer_to_rgba32f.wgsl").into()),
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("buffer_to_rgba32f layout"),
            bind_group_layouts: &[layout.value()],
            push_constant_ranges: &[],
        });

        let pipeline = ComputePipeline::from(device.wgpu_device().create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("buffer_to_rgba32f pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            },
        ));

        (layout, pipeline)
    }

    #[test]
    #[ignore = "gpu-dependent; run with `cargo test -- --ignored`"]
    fn burn_to_bevy_gpu_smoke() {
        let ctx = TestGpuContext::new();
        let extent = Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };
        let texture = ctx.render_device.create_texture(&TextureDescriptor {
            label: Some("burn_to_bevy_texture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let (layout, pipeline) = create_rgba_pipeline(&ctx.render_device);

        let tensor =
            Tensor::<BurnBackend, 3>::from_data(
                [[[0.0f32, 0.5, 1.0, 1.0]]],
                ctx.burn_device.device().expect("burn device ready"),
            );

        let copy = <() as BurnBevyPrepare<BurnBackend>>::prepare_bind_group(
            &tensor,
            ctx.burn_device.device().expect("burn device ready"),
            &ctx.render_device,
            &ctx.render_queue,
            &layout,
            &*texture,
            extent,
        )
        .expect("bind group");

        let mut encoder = ctx
            .render_device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("burn_to_bevy_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("burn_to_bevy_pass"),
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &copy.bg, &[]);
            pass.dispatch_workgroups(copy.workgroups[0], copy.workgroups[1], copy.workgroups[2]);
        }
        ctx.render_queue.submit([encoder.finish()]);

        let row_bytes = extent.width * 16;
        let padded_row = padded_bytes_per_row(extent.width, 16);
        let total_bytes = padded_row as u64 * extent.height as u64;

        let readback = ctx.render_device.create_buffer(&BufferDescriptor {
            label: Some("burn_to_bevy_readback"),
            size: total_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .render_device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("burn_to_bevy_copy_encoder"),
            });
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(extent.height),
                },
            },
            extent,
        );
        ctx.render_queue.submit([encoder.finish()]);

        let slice = readback.slice(..row_bytes as u64);
        slice.map_async(MapMode::Read, |_| {});
        ctx.poll_wait();
        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data[..row_bytes as usize]);
        let expected = [0.0f32, 0.5, 1.0, 1.0];
        let max_err = floats
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0, f32::max);
        assert!(max_err < 0.001, "gpu write mismatch: {floats:?}");
        drop(data);
        readback.unmap();
    }

    #[test]
    #[ignore = "gpu-dependent; run with `cargo test -- --ignored`"]
    fn bevy_to_burn_gpu_1x1() {
        let ctx = TestGpuContext::new();
        let extent = Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };
        let pixel = [255u8, 128, 0, 255];
        let texture = ctx.render_device.create_texture_with_data(
            &ctx.render_queue,
            &TextureDescriptor {
                label: Some("bevy_to_burn_texture"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8UnormSrgb,
                usage: TextureUsages::COPY_SRC
                    | TextureUsages::COPY_DST
                    | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            TextureDataOrder::LayerMajor,
            &pixel,
        );
        let texture_view = texture.create_view(&TextureViewDescriptor::default());
        let sampler = ctx
            .render_device
            .create_sampler(&wgpu::SamplerDescriptor::default());

        let gpu_image = GpuImage {
            texture: texture.clone(),
            texture_view,
            texture_format: TextureFormat::Rgba8UnormSrgb,
            sampler,
            size: extent,
            mip_level_count: 1,
        };

        let mut world = World::new();
        world.insert_resource(ctx.render_device.clone());
        world.insert_resource(ctx.render_queue.clone());
        world.insert_resource(RenderAssets::<GpuImage>::default());

        let handle = Handle::<Image>::default();
        {
            let mut images = world.resource_mut::<RenderAssets<GpuImage>>();
            images.insert(handle.id(), gpu_image);
        }

        let tensor = Tensor::<BurnBackend, 3>::zeros(
            [1, 1, 4],
            ctx.burn_device.device().expect("burn device ready"),
        );
        world.spawn(ExtractedGpuHandle::<BurnBackend> {
            image: handle.clone(),
            tensor,
            direction: BindingDirection::BevyToBurn,
            upload: true,
        });
        let mut system_state: SystemState<(
            Res<RenderDevice>,
            Res<RenderQueue>,
            Res<RenderAssets<GpuImage>>,
            Query<&mut ExtractedGpuHandle<BurnBackend>>,
        )> = SystemState::new(&mut world);
        {
            let (render_device, render_queue, images, query) = system_state.get_mut(&mut world);
            gpu_bevy_to_burn::<BurnBackend>(render_device, render_queue, images, query);
        }
        system_state.apply(&mut world);
        ctx.poll_wait();

        let handle_comp = world
            .query::<&ExtractedGpuHandle<BurnBackend>>()
            .single(&world)
            .expect("gpu handle in world");
        assert!(!handle_comp.upload, "texture readback flag not cleared");
        let data = handle_comp.tensor.to_data();
        let floats: Vec<f32> = data.to_vec::<f32>().unwrap();
        let expected = [
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
            pixel[3] as f32 / 255.0,
        ];
        let max_err = floats
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0, f32::max);
        assert!(max_err < 0.02, "bevy->burn mismatch: {floats:?}");
    }
}
