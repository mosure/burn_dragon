use bevy::asset::RenderAssetUsages;
use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::image::{Image, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource, BindingType,
    BufferBindingType, CachedComputePipelineId, ComputePipelineDescriptor, FilterMode,
    PipelineCache, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages, ShaderType,
    StorageTextureAccess, TextureDimension, TextureFormat, TextureSampleType,
    TextureUsages, TextureViewDimension, UniformBuffer,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy_shader::Shader;
#[cfg(test)]
use half::f16;
use wgpu::Extent3d;

use crate::{
    radius_norm_from_sample, radius_px_from_norm, resolve_foveation_sample,
    sigma_norm_from_settings, sigma_px_from_norm, FoveaWarpMode, FoveationBackendMode,
    FoveationNoiseSample, FoveationRuntime, FoveationSettings, PyramidMode, SourceImage,
};
use burn_dragon_hatchling_vision::foveation;
use burn_dragon_hatchling_vision::{FOVEATION_SHADER, PYRAMID_SHADER};
#[cfg(test)]
use crate::{ImageLevel, PyramidCache};

const WORKGROUP_SIZE: u32 = 8;
const SHADER_SOURCE: &str = FOVEATION_SHADER;
const PYRAMID_SHADER_SOURCE: &str = PYRAMID_SHADER;

#[derive(Clone, Copy, Default, ShaderType)]
pub(crate) struct FoveationUniform {
    image_size: Vec2,
    inv_image_size: Vec2,
    center: Vec2,
    sigma: Vec2,
    sample_scale: f32,
    lod_sigma: f32,
    patch_size: f32,
    pyramid_levels: u32,
    mode: u32,
    warp_mode: u32,
    _pad0: u32,
    _pad1: u32,
}

#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct FoveationGpuParams {
    pub uniform: FoveationUniform,
    pub dispatch: UVec2,
    pub enabled: bool,
}

impl Default for FoveationGpuParams {
    fn default() -> Self {
        Self {
            uniform: FoveationUniform {
                image_size: Vec2::ONE,
                inv_image_size: Vec2::ONE,
                center: Vec2::ZERO,
                sigma: Vec2::ONE,
                sample_scale: 1.0,
                lod_sigma: 1.0,
                patch_size: 1.0,
                pyramid_levels: 1,
                mode: 0,
                warp_mode: 0,
                _pad0: 0,
                _pad1: 0,
            },
            dispatch: UVec2::ONE,
            enabled: false,
        }
    }
}

#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct FoveationGpuImages {
    pub gaussian: Handle<Image>,
    pub residual: Handle<Image>,
    pub output: Handle<Image>,
}

#[derive(Resource, Clone)]
pub(crate) struct FoveationInputImage {
    pub handle: Handle<Image>,
}

#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct FoveationShaderHandles {
    pub foveation: Handle<Shader>,
    pub pyramid: Handle<Shader>,
}

#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct FoveationPyramidConfig {
    pub input: Handle<Image>,
    pub gaussian: Handle<Image>,
    pub residual: Handle<Image>,
    pub size: UVec2,
    pub levels: u32,
    pub version: u64,
    pub enabled: bool,
}

impl Default for FoveationPyramidConfig {
    fn default() -> Self {
        Self {
            input: Handle::default(),
            gaussian: Handle::default(),
            residual: Handle::default(),
            size: UVec2::ONE,
            levels: 1,
            version: 0,
            enabled: false,
        }
    }
}

#[derive(Clone, Copy, Default, ShaderType)]
struct PyramidUniform {
    src_dst: UVec4,
}

#[derive(Resource)]
struct FoveationGpuPipeline {
    bind_group_layout: BindGroupLayout,
    pipeline_id: CachedComputePipelineId,
    sampler: Sampler,
    _shader: Handle<Shader>,
}

#[derive(Resource, Default)]
struct FoveationGpuBindGroup {
    bind_group: Option<BindGroup>,
    uniform_buffer: UniformBuffer<FoveationUniform>,
}

#[derive(Resource)]
struct FoveationPyramidPipeline {
    bind_group_layout: BindGroupLayout,
    downsample_pipeline: CachedComputePipelineId,
    residual_pipeline: CachedComputePipelineId,
    sampler: Sampler,
    _shader: Handle<Shader>,
}

#[derive(Resource, Default)]
struct FoveationPyramidState {
    uniform_buffer: UniformBuffer<PyramidUniform>,
    last_version: u64,
}

pub(crate) struct FoveationGpuPlugin;

impl Plugin for FoveationGpuPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<FoveationShaderHandles>() {
            let mut shaders = app.world_mut().resource_mut::<Assets<Shader>>();
            let foveation = shaders.add(Shader::from_wgsl(SHADER_SOURCE, "foveation.wgsl"));
            let pyramid = shaders.add(Shader::from_wgsl(PYRAMID_SHADER_SOURCE, "pyramid.wgsl"));
            app.insert_resource(FoveationShaderHandles { foveation, pyramid });
        }
        app.init_resource::<FoveationGpuParams>()
            .init_resource::<FoveationPyramidConfig>()
            .add_plugins(ExtractResourcePlugin::<FoveationGpuParams>::default())
            .add_plugins(ExtractResourcePlugin::<FoveationGpuImages>::default())
            .add_plugins(ExtractResourcePlugin::<FoveationPyramidConfig>::default())
            .add_plugins(ExtractResourcePlugin::<FoveationShaderHandles>::default());

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<FoveationGpuBindGroup>()
                .init_resource::<FoveationPyramidState>()
                .add_systems(
                    Render,
                    init_foveation_pipeline.in_set(RenderSystems::PrepareResources),
                )
                .add_systems(
                    Render,
                    init_pyramid_pipeline.in_set(RenderSystems::PrepareResources),
                )
                .add_systems(
                    Render,
                    prepare_foveation_uniform.in_set(RenderSystems::PrepareResources),
                )
                .add_systems(
                    Render,
                    prepare_foveation_bind_group.in_set(RenderSystems::PrepareBindGroups),
                )
                .add_systems(
                    Render,
                    dispatch_pyramid_compute
                        .in_set(RenderSystems::Render)
                        .before(dispatch_foveation_compute),
                )
                .add_systems(Render, dispatch_foveation_compute.in_set(RenderSystems::Render));
        }
    }
}

fn init_foveation_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    shaders: Option<Res<FoveationShaderHandles>>,
    existing: Option<Res<FoveationGpuPipeline>>,
) {
    if existing.is_some() {
        return;
    }
    let Some(shaders) = shaders else {
        return;
    };
    let shader = shaders.foveation.clone();
    let layout_entries = [
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Sampler(SamplerBindingType::Filtering),
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 3,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::Rgba8Unorm,
                view_dimension: TextureViewDimension::D2,
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 4,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(FoveationUniform::min_size()),
            },
            count: None,
        },
    ];
    let bind_group_layout = render_device
        .create_bind_group_layout("foveation_bind_group_layout", &layout_entries);

    let sampler = render_device.create_sampler(&SamplerDescriptor {
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        ..Default::default()
    });
    let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("foveation_compute_pipeline".into()),
        layout: vec![bind_group_layout.clone()],
        shader: shader.clone(),
        shader_defs: vec![],
        entry_point: Some("main".into()),
        push_constant_ranges: vec![],
        zero_initialize_workgroup_memory: true,
    });

    commands.insert_resource(FoveationGpuPipeline {
        bind_group_layout,
        pipeline_id,
        sampler,
        _shader: shader,
    });
}

fn init_pyramid_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    shaders: Option<Res<FoveationShaderHandles>>,
    existing: Option<Res<FoveationPyramidPipeline>>,
) {
    if existing.is_some() {
        return;
    }
    let Some(shaders) = shaders else {
        return;
    };
    let shader = shaders.pyramid.clone();
    let layout_entries = [
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Sampler(SamplerBindingType::Filtering),
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 3,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::Rgba16Float,
                view_dimension: TextureViewDimension::D2,
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 4,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(PyramidUniform::min_size()),
            },
            count: None,
        },
    ];
    let bind_group_layout = render_device
        .create_bind_group_layout("foveation_pyramid_bind_group_layout", &layout_entries);

    let sampler = render_device.create_sampler(&SamplerDescriptor {
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        ..Default::default()
    });
    let downsample_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("foveation_pyramid_downsample_pipeline".into()),
        layout: vec![bind_group_layout.clone()],
        shader: shader.clone(),
        shader_defs: vec![],
        entry_point: Some("downsample".into()),
        push_constant_ranges: vec![],
        zero_initialize_workgroup_memory: true,
    });
    let residual_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("foveation_pyramid_residual_pipeline".into()),
        layout: vec![bind_group_layout.clone()],
        shader: shader.clone(),
        shader_defs: vec![],
        entry_point: Some("residual".into()),
        push_constant_ranges: vec![],
        zero_initialize_workgroup_memory: true,
    });

    commands.insert_resource(FoveationPyramidPipeline {
        bind_group_layout,
        downsample_pipeline,
        residual_pipeline,
        sampler,
        _shader: shader,
    });
}

pub(crate) fn init_gpu_images(
    commands: &mut Commands,
    images: &mut Assets<Image>,
    output: Handle<Image>,
) {
    let gaussian = images.add(blank_mip_image(1, 1, 1));
    let residual = images.add(blank_mip_image(1, 1, 1));
    commands.insert_resource(FoveationGpuImages {
        gaussian,
        residual,
        output,
    });
}

pub(crate) fn update_gpu_params(
    settings: Res<FoveationSettings>,
    source: Res<SourceImage>,
    runtime: Res<FoveationRuntime>,
    noise: Res<FoveationNoiseSample>,
    mut params: ResMut<FoveationGpuParams>,
) {
    let enabled = matches!(settings.backend, FoveationBackendMode::Wgsl);
    params.enabled = enabled;
    if !enabled {
        return;
    }
    let patch = runtime.patch_size.max(1) as f32;
    let patch_u32 = runtime.patch_size.max(1) as u32;
    let image_size = Vec2::new(source.width as f32, source.height as f32).max(Vec2::ONE);
    let inv_image_size = Vec2::new(1.0 / image_size.x, 1.0 / image_size.y);
    let sample = resolve_foveation_sample(&settings, &noise);
    let center = Vec2::new(
        sample.mean_x.clamp(0.0, 1.0) * image_size.x,
        sample.mean_y.clamp(0.0, 1.0) * image_size.y,
    );
    let radius_norm = radius_norm_from_sample(&sample);
    let sigma_norm = sigma_norm_from_settings(&settings, &sample);
    let sigma_px = sigma_px_from_norm(sigma_norm, &source);
    let radius_px = radius_px_from_norm(radius_norm, &source);
    let sigma = Vec2::splat(sigma_px);
    let lod_sigma = foveation::lod_sigma_from_sigma(sigma_norm);
    let sample_scale = (radius_px * 2.0) / patch.max(1.0);
    let pyramid_levels = settings.pyramid_depth.max(2) as u32;
    let mode = match settings.mode {
        PyramidMode::Gaussian => 0,
        PyramidMode::Laplacian => 1,
    };
    let warp_mode = match settings.warp_mode {
        FoveaWarpMode::Warped => 0,
        FoveaWarpMode::Patched => 1,
    };

    params.uniform = FoveationUniform {
        image_size,
        inv_image_size,
        center,
        sigma,
        sample_scale,
        lod_sigma,
        patch_size: patch,
        pyramid_levels,
        mode,
        warp_mode,
        _pad0: 0,
        _pad1: 0,
    };
    let groups_x = patch_u32.div_ceil(WORKGROUP_SIZE);
    let groups_y = patch_u32.div_ceil(WORKGROUP_SIZE);
    params.dispatch = UVec2::new(groups_x.max(1), groups_y.max(1));
}

pub(crate) fn update_gpu_pyramid_textures(
    source: Res<SourceImage>,
    settings: Res<FoveationSettings>,
    input: Res<FoveationInputImage>,
    mut images: ResMut<Assets<Image>>,
    gpu_images: Res<FoveationGpuImages>,
    mut config: ResMut<FoveationPyramidConfig>,
) {
    let size = UVec2::new(source.width as u32, source.height as u32).max(UVec2::ONE);
    let levels = settings.pyramid_depth.max(2) as u32;
    let enabled = matches!(settings.backend, FoveationBackendMode::Wgsl);
    let mut changed = false;

    if config.enabled != enabled {
        changed = true;
    }
    if config.size != size || config.levels != levels {
        let gaussian = blank_mip_image(size.x, size.y, levels);
        let residual = blank_mip_image(size.x, size.y, levels);
        if let Some(image) = images.get_mut(&gpu_images.gaussian) {
            *image = gaussian;
        }
        if let Some(image) = images.get_mut(&gpu_images.residual) {
            *image = residual;
        }
        changed = true;
    }

    if config.input != input.handle
        || config.gaussian != gpu_images.gaussian
        || config.residual != gpu_images.residual
    {
        changed = true;
    }

    if changed {
        config.version = config.version.wrapping_add(1);
    }
    config.input = input.handle.clone();
    config.gaussian = gpu_images.gaussian.clone();
    config.residual = gpu_images.residual.clone();
    config.size = size;
    config.levels = levels;
    config.enabled = enabled;
}

fn prepare_foveation_uniform(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    params: Res<FoveationGpuParams>,
    mut bind_group: ResMut<FoveationGpuBindGroup>,
) {
    if !params.enabled {
        return;
    }
    let buffer = bind_group.uniform_buffer.get_mut();
    *buffer = params.uniform;
    bind_group
        .uniform_buffer
        .write_buffer(&render_device, &render_queue);
}

fn prepare_foveation_bind_group(
    render_device: Res<RenderDevice>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    images: Res<FoveationGpuImages>,
    pipeline: Res<FoveationGpuPipeline>,
    params: Res<FoveationGpuParams>,
    mut bind_group: ResMut<FoveationGpuBindGroup>,
) {
    if !params.enabled {
        bind_group.bind_group = None;
        return;
    }
    let Some(gaussian) = gpu_images.get(&images.gaussian) else {
        bind_group.bind_group = None;
        return;
    };
    let Some(residual) = gpu_images.get(&images.residual) else {
        bind_group.bind_group = None;
        return;
    };
    let Some(output) = gpu_images.get(&images.output) else {
        bind_group.bind_group = None;
        return;
    };

    let uniform_binding = bind_group
        .uniform_buffer
        .binding()
        .expect("uniform buffer binding");

    let entries = [
        BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureView(&gaussian.texture_view),
        },
        BindGroupEntry {
            binding: 1,
            resource: BindingResource::Sampler(&pipeline.sampler),
        },
        BindGroupEntry {
            binding: 2,
            resource: BindingResource::TextureView(&residual.texture_view),
        },
        BindGroupEntry {
            binding: 3,
            resource: BindingResource::TextureView(&output.texture_view),
        },
        BindGroupEntry {
            binding: 4,
            resource: uniform_binding,
        },
    ];
    let group = render_device.create_bind_group(
        "foveation_bind_group",
        &pipeline.bind_group_layout,
        &entries,
    );

    bind_group.bind_group = Some(group);
}

fn dispatch_pyramid_compute(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<FoveationPyramidPipeline>,
    mut state: ResMut<FoveationPyramidState>,
    config: Res<FoveationPyramidConfig>,
    gpu_images: Res<RenderAssets<GpuImage>>,
) {
    if !config.enabled {
        return;
    }
    if config.version == state.last_version {
        return;
    }

    let Some(input) = gpu_images.get(&config.input) else {
        return;
    };
    let Some(gaussian) = gpu_images.get(&config.gaussian) else {
        return;
    };
    let Some(residual) = gpu_images.get(&config.residual) else {
        return;
    };
    let Some(downsample_pipeline) =
        pipeline_cache.get_compute_pipeline(pipeline.downsample_pipeline)
    else {
        return;
    };
    let Some(residual_pipeline) =
        pipeline_cache.get_compute_pipeline(pipeline.residual_pipeline)
    else {
        return;
    };

    let levels = config.levels.max(1);
    let mut sizes = Vec::with_capacity(levels as usize);
    let mut width = config.size.x.max(1);
    let mut height = config.size.y.max(1);
    for _ in 0..levels {
        sizes.push(UVec2::new(width.max(1), height.max(1)));
        width = (width / 2).max(1);
        height = (height / 2).max(1);
    }

    let dummy_view = input
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let dispatch_pass = |encoder: &mut wgpu::CommandEncoder,
                         pipeline_ref: &wgpu::ComputePipeline,
                         bind_group: &BindGroup,
                         dispatch: UVec2| {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("foveation_pyramid_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline_ref);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(dispatch.x, dispatch.y, 1);
    };

    let mut make_bind_group = |src_size: UVec2,
                               dst_size: UVec2,
                               src_view: &wgpu::TextureView,
                               coarse_view: &wgpu::TextureView,
                               dst_view: &wgpu::TextureView| {
        let buffer = state.uniform_buffer.get_mut();
        buffer.src_dst = UVec4::new(src_size.x, src_size.y, dst_size.x, dst_size.y);
        state
            .uniform_buffer
            .write_buffer(&render_device, &render_queue);
        let uniform_binding = state
            .uniform_buffer
            .binding()
            .expect("pyramid uniform binding");
        render_device.create_bind_group(
            "foveation_pyramid_bind_group",
            &pipeline.bind_group_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(src_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(coarse_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&pipeline.sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(dst_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: uniform_binding,
                },
            ],
        )
    };

    if let Some(size) = sizes.first().copied() {
        let mut encoder = render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("foveation_pyramid_copy_base"),
        });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &input.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &gaussian.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
        );
        render_queue.submit([encoder.finish()]);
    }

    let mut scratch_textures = Vec::new();
    for level in 0..levels.saturating_sub(1) {
        let src_size = sizes[level as usize];
        let dst_size = sizes[(level + 1) as usize];
        let scratch = render_device.create_texture(&wgpu::TextureDescriptor {
            label: Some("foveation_pyramid_scratch"),
            size: wgpu::Extent3d {
                width: src_size.x,
                height: src_size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        scratch_textures.push(scratch);
        let scratch_ref = scratch_textures
            .last()
            .expect("scratch texture available");

        let mut copy_encoder =
            render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("foveation_pyramid_copy_level"),
            });
        copy_encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &gaussian.texture,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: scratch_ref,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: src_size.x,
                height: src_size.y,
                depth_or_array_layers: 1,
            },
        );
        render_queue.submit([copy_encoder.finish()]);

        let src_view = scratch_ref.create_view(&wgpu::TextureViewDescriptor::default());
        let dst_view = gaussian.texture.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: level + 1,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let bind_group = make_bind_group(src_size, dst_size, &src_view, &dummy_view, &dst_view);
        let dispatch = workgroup_dispatch(dst_size);
        let mut compute_encoder =
            render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("foveation_pyramid_downsample"),
            });
        dispatch_pass(&mut compute_encoder, downsample_pipeline, &bind_group, dispatch);
        render_queue.submit([compute_encoder.finish()]);
    }

    for level in 0..levels.saturating_sub(1) {
        let size = sizes[level as usize];
        let fine_view = gaussian.texture.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: level,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let coarse_view = gaussian.texture.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: level + 1,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let dst_view = residual.texture.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: level,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let bind_group = make_bind_group(size, size, &fine_view, &coarse_view, &dst_view);
        let dispatch = workgroup_dispatch(size);
        let mut compute_encoder =
            render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("foveation_pyramid_residual"),
            });
        dispatch_pass(&mut compute_encoder, residual_pipeline, &bind_group, dispatch);
        render_queue.submit([compute_encoder.finish()]);
    }
    state.last_version = config.version;
}

fn dispatch_foveation_compute(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<FoveationGpuPipeline>,
    params: Res<FoveationGpuParams>,
    bind_group: Res<FoveationGpuBindGroup>,
) {
    if !params.enabled {
        return;
    }
    let Some(bind_group) = bind_group.bind_group.as_ref() else {
        return;
    };
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline_id) else {
        return;
    };

    let mut encoder = render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("foveation_compute_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("foveation_compute_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(params.dispatch.x, params.dispatch.y, 1);
    }
    render_queue.submit([encoder.finish()]);
}

fn workgroup_dispatch(size: UVec2) -> UVec2 {
    let groups_x = size.x.div_ceil(WORKGROUP_SIZE);
    let groups_y = size.y.div_ceil(WORKGROUP_SIZE);
    UVec2::new(groups_x.max(1), groups_y.max(1))
}

fn blank_mip_image(width: u32, height: u32, levels: u32) -> Image {
    let size = Extent3d {
        width: width.max(1),
        height: height.max(1),
        depth_or_array_layers: 1,
    };
    let mut image = Image::new_uninit(
        size,
        TextureDimension::D2,
        TextureFormat::Rgba16Float,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage |= TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC
        | TextureUsages::STORAGE_BINDING;
    image.texture_descriptor.mip_level_count = levels.max(1);
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor::linear());
    image
}

#[cfg(test)]
fn build_residual_levels(cache: &PyramidCache) -> Vec<ImageLevel> {
    let mut levels = cache.laplacian.clone();
    if let Some(coarse) = &cache.coarse {
        let mut data = vec![0.0; coarse.width * coarse.height * 3];
        for value in &mut data {
            *value = 0.0;
        }
        levels.push(ImageLevel {
            width: coarse.width,
            height: coarse.height,
            data,
        });
    }
    levels
}

#[cfg(test)]
fn push_f16(data: &mut Vec<u8>, value: f32) {
    let bits = f16::from_f32(value).to_bits();
    data.extend_from_slice(&bits.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        FoveaWarpMode, FoveationBackendMode, FoveationSample, ImageLevel, PyramidCache,
        build_gaussian_pyramid, build_laplacian_pyramid, make_minimal_vision_config,
        map_pyramid_mode, map_warp_mode, radius_norm_from_sample, render_patch_f32,
        sigma_norm_from_settings,
    };
    use burn::tensor::backend::Backend;
    use burn::tensor::{Tensor, TensorData};
    use burn_dragon_hatchling_vision::train::SaccadeFoveationSampler;
    use burn_dragon_hatchling_core::{VisionFoveaSamplingMode, VisionSaccadeConfig};
    use burn_wgpu::{self, RuntimeOptions, Wgpu};
    use burn_wgpu::graphics;
    use image::RgbImage;
    use std::collections::HashSet;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::mpsc;
    use std::sync::Once;
    use wgpu::util::DeviceExt;

    type BurnBackend = Wgpu<f32>;

    fn init_burn_device() -> burn_wgpu::WgpuDevice {
        static INIT: Once = Once::new();
        let device = burn_wgpu::WgpuDevice::default();
        INIT.call_once(|| {
            burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(&device, RuntimeOptions::default());
        });
        device
    }

    fn make_checkerboard(width: usize, height: usize) -> SourceImage {
        let mut data = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                let value = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
                data.push(value);
                data.push(value);
                data.push(value);
            }
        }
        SourceImage { width, height, data }
    }

    fn make_gradient(width: usize, height: usize) -> SourceImage {
        let mut data = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                let fx = x as f32 / (width as f32 - 1.0);
                let fy = y as f32 / (height as f32 - 1.0);
                data.push(fx);
                data.push(fy);
                data.push(0.5);
            }
        }
        SourceImage { width, height, data }
    }

    fn make_radial(width: usize, height: usize) -> SourceImage {
        let mut data = Vec::with_capacity(width * height * 3);
        let cx = (width.saturating_sub(1)) as f32 * 0.5;
        let cy = (height.saturating_sub(1)) as f32 * 0.5;
        let max_r = (cx * cx + cy * cy).sqrt().max(1.0);
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let r = ((dx * dx + dy * dy).sqrt() / max_r).clamp(0.0, 1.0);
                let angle = dy.atan2(dx);
                let wave = (angle * 6.0).sin() * 0.5 + 0.5;
                data.push(r);
                data.push(1.0 - r);
                data.push(wave);
            }
        }
        SourceImage { width, height, data }
    }

    fn settings_for_mode(mode: PyramidMode) -> FoveationSettings {
        let mut settings = FoveationSettings::default();
        settings.patch_size = 16;
        settings.pyramid_depth = 4;
        settings.radius_norm = 0.25;
        settings.focus = 0.5;
        settings.mean_x = 0.5;
        settings.mean_y = 0.5;
        settings.mode = mode;
        settings.backend = FoveationBackendMode::Wgsl;
        settings
    }

    fn tensor_from_source<B: Backend>(source: &SourceImage, device: &B::Device) -> Tensor<B, 4> {
        let width = source.width.max(1);
        let height = source.height.max(1);
        let mut data = vec![0.0f32; 3 * width * height];
        for y in 0..height {
            for x in 0..width {
                let src = (y * width + x) * 3;
                let dst = y * width + x;
                data[dst] = source.data[src];
                data[dst + height * width] = source.data[src + 1];
                data[dst + 2 * height * width] = source.data[src + 2];
            }
        }
        Tensor::<B, 4>::from_data(TensorData::new(data, [1, 3, height, width]), device)
    }

    fn render_patch_burn(source: &SourceImage, settings: &FoveationSettings) -> Vec<f32> {
        let device = init_burn_device();
        let patch = settings
            .patch_size
            .max(1)
            .min(source.width.min(source.height));
        let mut vision = make_minimal_vision_config(source.width, source.height, patch);
        vision.patch_size = patch;
        let mut saccade = VisionSaccadeConfig::default();
        saccade.mip_levels = settings.pyramid_depth.max(1);
        saccade.pyramid_mode = map_pyramid_mode(settings.mode);
        saccade.fovea_warp_mode = map_warp_mode(settings.warp_mode);
        if settings.backend == FoveationBackendMode::Cubecl {
            saccade.fovea_sampling_mode = VisionFoveaSamplingMode::Cubecl;
        }
        let mut sampler = SaccadeFoveationSampler::<BurnBackend>::new(vision, saccade, &device);
        let input = tensor_from_source::<BurnBackend>(source, &device);
        sampler.update_image(input);

        let sample = FoveationSample {
            mean_x: settings.mean_x,
            mean_y: settings.mean_y,
            radius_norm: settings.radius_norm,
        };
        let radius_norm = radius_norm_from_sample(&sample);
        let sigma_norm = sigma_norm_from_settings(settings, &sample);
        let mean = Tensor::<BurnBackend, 2>::from_data(
            TensorData::new(vec![sample.mean_x, sample.mean_y], [1, 2]),
            &device,
        );
        let sigma = Tensor::<BurnBackend, 2>::from_data(
            TensorData::new(vec![sigma_norm], [1, 1]),
            &device,
        );
        let radius = Tensor::<BurnBackend, 2>::from_data(
            TensorData::new(vec![radius_norm], [1, 1]),
            &device,
        );
        let patch_tensor = sampler.sample_patch_with_radius(mean, sigma, radius);
        let [batch, channels, height, width] = patch_tensor.shape().dims::<4>();
        let data = patch_tensor.into_data();
        let values = data.as_slice::<f32>().expect("f32 tensor data");
        let mut out = vec![0.0f32; height * width * 3];
        if batch == 0 || channels < 3 {
            return out;
        }
        let plane = height * width;
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let out_idx = idx * 3;
                out[out_idx] = values[idx];
                out[out_idx + 1] = values[idx + plane];
                out[out_idx + 2] = values[idx + plane * 2];
            }
        }
        out
    }

    fn max_abs_diff_f32(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    fn quantize_f16(values: &[f32]) -> Vec<f32> {
        values
            .iter()
            .map(|value| f16::from_f32(value.clamp(0.0, 1.0)).to_f32())
            .collect()
    }

    fn mse_f32(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for (x, y) in a.iter().zip(b.iter()) {
            let diff = x - y;
            sum += diff * diff;
            count += 1;
        }
        if count == 0 {
            return 0.0;
        }
        sum / count as f32
    }

    fn assert_patch_close(label: &str, a: &[f32], b: &[f32], max_abs: f32, mse: f32) {
        let max_diff = max_abs_diff_f32(a, b);
        let mse_diff = mse_f32(a, b);
        assert!(
            max_diff <= max_abs && mse_diff <= mse,
            "{label} max_abs {max_diff} mse {mse_diff}"
        );
    }

    fn fovea_test_root() -> PathBuf {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let root = manifest_dir
            .parent()
            .and_then(|parent| parent.parent())
            .map(PathBuf::from)
            .unwrap_or(manifest_dir);
        root.join("runs").join("fovea_test")
    }

    fn save_patch_image(path: &Path, patch: &[f32], patch_size: usize) {
        let expected = patch_size * patch_size * 3;
        assert_eq!(
            patch.len(),
            expected,
            "expected {expected} rgb values, got {}",
            patch.len()
        );
        let mut bytes = Vec::with_capacity(expected);
        for value in patch {
            let scaled = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
            bytes.push(scaled);
        }
        let image = RgbImage::from_raw(patch_size as u32, patch_size as u32, bytes)
            .expect("rgb patch buffer");
        image.save(path).expect("write fovea test image");
    }

    fn save_source_image(path: &Path, source: &SourceImage) {
        let expected = source.width * source.height * 3;
        assert_eq!(
            source.data.len(),
            expected,
            "expected {expected} rgb values, got {}",
            source.data.len()
        );
        let mut bytes = Vec::with_capacity(expected);
        for value in source.data.iter().copied() {
            let scaled = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
            bytes.push(scaled);
        }
        let image = RgbImage::from_raw(source.width as u32, source.height as u32, bytes)
            .expect("rgb source buffer");
        image.save(path).expect("write fovea test source image");
    }

    fn save_source_identity(
        root: &Path,
        source_name: &str,
        source: &SourceImage,
        seen: &mut HashSet<String>,
    ) {
        let key = format!("{source_name}/{}x{}", source.width, source.height);
        if !seen.insert(key) {
            return;
        }
        let dir = root
            .join(source_name)
            .join(format!("{}x{}", source.width, source.height));
        fs::create_dir_all(&dir).expect("create fovea_test source dir");
        let info = format!(
            "source={source_name}\nwidth={}\nheight={}\n",
            source.width, source.height
        );
        fs::write(dir.join("source.txt"), info).expect("write fovea_test source identity");
        save_source_image(&dir.join("source.png"), source);
    }

    fn save_patch_output(
        root: &Path,
        source_name: &str,
        source: &SourceImage,
        mode: PyramidMode,
        warp_mode: FoveaWarpMode,
        case_idx: usize,
        mean_x: f32,
        mean_y: f32,
        radius: f32,
        focus: f32,
        patch_size: usize,
        depth: usize,
        backend: &str,
        patch: &[f32],
    ) {
        let case_dir = format!(
            "case_{case_idx}_mx_{mean_x:.2}_my_{mean_y:.2}_r_{radius:.2}_f_{focus:.2}_patch_{patch_size}_depth_{depth}"
        );
        let dir = root
            .join(source_name)
            .join(format!("{}x{}", source.width, source.height))
            .join(format!("mode_{mode:?}"))
            .join(format!("warp_{warp_mode:?}"))
            .join(case_dir);
        fs::create_dir_all(&dir).expect("create fovea_test output dir");
        let path = dir.join(format!("{backend}.png"));
        save_patch_image(&path, patch, patch_size);
    }

    fn shader_source_f16() -> String {
        SHADER_SOURCE.replace("rgba8unorm", "rgba16float")
    }

    fn render_patch_gpu_f32(
        source: &SourceImage,
        cache: &PyramidCache,
        settings: &FoveationSettings,
    ) -> Option<Vec<f32>> {
        let instance = wgpu::Instance::default();
        let supports_format = |adapter: &wgpu::Adapter| {
            let features = adapter.get_texture_format_features(TextureFormat::Rgba16Float);
            features
                .flags
                .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
                && features
                    .flags
                    .contains(wgpu::TextureFormatFeatureFlags::STORAGE_WRITE_ONLY)
        };

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        if !supports_format(&adapter) {
            return None;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .ok()?;

        let gaussian_texture = create_mip_texture(&device, &queue, &cache.gaussian);
        let residual_levels = build_residual_levels(cache);
        let residual_texture = create_mip_texture(&device, &queue, &residual_levels);

        let output_size = settings.patch_size as u32;
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("foveation_output_f16"),
            size: wgpu::Extent3d {
                width: output_size.max(1),
                height: output_size.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniform = uniform_for_settings(settings, source);
        let mut encased = bevy::render::render_resource::encase::UniformBuffer::new(Vec::new());
        encased.write(&uniform).ok()?;
        let uniform_bytes = encased.into_inner();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("foveation_uniform"),
            contents: &uniform_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let shader_source = shader_source_f16();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("foveation_shader_f16"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("foveation_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("foveation_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &gaussian_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &residual_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("foveation_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("foveation_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("foveation_compute_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("foveation_compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups_x = (output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            let groups_y = (output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }

        let bytes_per_row = output_size * 8;
        let aligned_bytes_per_row = align_bytes_per_row(bytes_per_row);
        let buffer_size = (aligned_bytes_per_row * output_size) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("foveation_output_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bytes_per_row),
                    rows_per_image: Some(output_size),
                },
            },
            wgpu::Extent3d {
                width: output_size,
                height: output_size,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        let slice = output_buffer.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).ok();
        });
        device.poll(wgpu::PollType::Wait).ok();
        let map_result = rx.recv().ok()?;
        map_result.ok()?;
        let data = slice.get_mapped_range().to_vec();
        output_buffer.unmap();
        let trimmed = trim_padded_rows_bytes(
            &data,
            output_size,
            output_size,
            8,
            aligned_bytes_per_row,
        );
        Some(decode_f16_rgb(&trimmed, output_size as usize, output_size as usize))
    }

    fn make_cache(source: &SourceImage, depth: usize) -> PyramidCache {
        let base = ImageLevel {
            width: source.width,
            height: source.height,
            data: source.data.clone(),
        };
        let gaussian = build_gaussian_pyramid(&base, depth);
        let (laplacian, coarse) = build_laplacian_pyramid(&gaussian);
        let gaussian = quantize_levels(gaussian);
        let laplacian = quantize_levels(laplacian);
        let coarse = quantize_level(coarse);
        PyramidCache {
            gaussian,
            laplacian,
            coarse: Some(coarse),
        }
    }

    fn quantize_levels(levels: Vec<ImageLevel>) -> Vec<ImageLevel> {
        levels
            .into_iter()
            .map(quantize_level)
            .collect()
    }

    fn quantize_level(mut level: ImageLevel) -> ImageLevel {
        for value in &mut level.data {
            *value = f16::from_f32(*value).to_f32();
        }
        level
    }

    fn build_gaussian_pyramid_quantized(base: &ImageLevel, depth: usize) -> Vec<ImageLevel> {
        let mut out = Vec::with_capacity(depth.max(1));
        let mut current = quantize_level(base.clone());
        out.push(current.clone());
        for _ in 1..depth {
            current = downsample_level(&current);
            current = quantize_level(current);
            out.push(current.clone());
        }
        out
    }

    fn downsample_level(level: &ImageLevel) -> ImageLevel {
        let new_w = (level.width / 2).max(1);
        let new_h = (level.height / 2).max(1);
        let mut data = vec![0.0; new_w * new_h * 3];
        let weights = [1.0_f32, 4.0, 6.0, 4.0, 1.0];
        for y in 0..new_h {
            for x in 0..new_w {
                let mut accum = [0.0; 3];
                for ky in 0..5 {
                    let wy = weights[ky];
                    let sy = (y * 2).saturating_add(ky).saturating_sub(2);
                    let sy = sy.min(level.height - 1);
                    for kx in 0..5 {
                        let wx = weights[kx];
                        let sx = (x * 2).saturating_add(kx).saturating_sub(2);
                        let sx = sx.min(level.width - 1);
                        let weight = wx * wy;
                        let idx = (sy * level.width + sx) * 3;
                        accum[0] += level.data[idx] * weight;
                        accum[1] += level.data[idx + 1] * weight;
                        accum[2] += level.data[idx + 2] * weight;
                    }
                }
                let idx = (y * new_w + x) * 3;
                data[idx] = accum[0] / 256.0;
                data[idx + 1] = accum[1] / 256.0;
                data[idx + 2] = accum[2] / 256.0;
            }
        }
        ImageLevel {
            width: new_w,
            height: new_h,
            data,
        }
    }

    fn build_laplacian_pyramid_gpu(gaussian: &[ImageLevel]) -> Vec<ImageLevel> {
        let mut residuals = Vec::with_capacity(gaussian.len().saturating_sub(1));
        for idx in 0..gaussian.len().saturating_sub(1) {
            let current = &gaussian[idx];
            let next = &gaussian[idx + 1];
            let up = resample_gpu(next, current.width, current.height);
            let mut data = vec![0.0; current.width * current.height * 3];
            for i in 0..data.len() {
                data[i] = current.data[i] - up.data[i];
            }
            residuals.push(ImageLevel {
                width: current.width,
                height: current.height,
                data,
            });
        }
        residuals
    }

    fn resample_gpu(level: &ImageLevel, width: usize, height: usize) -> ImageLevel {
        let mut data = vec![0.0; width * height * 3];
        for y in 0..height {
            let fy = (y as f32 + 0.5) / height as f32;
            for x in 0..width {
                let fx = (x as f32 + 0.5) / width as f32;
                let sample = sample_bilinear_gpu(level, fx, fy);
                let idx = (y * width + x) * 3;
                data[idx] = sample[0];
                data[idx + 1] = sample[1];
                data[idx + 2] = sample[2];
            }
        }
        ImageLevel { width, height, data }
    }

    fn sample_bilinear_gpu(level: &ImageLevel, fx: f32, fy: f32) -> [f32; 3] {
        let grid_x = if level.width > 1 {
            (fx * level.width as f32 - 0.5) * (2.0 / (level.width - 1) as f32) - 1.0
        } else {
            0.0
        }
        .clamp(-1.0, 1.0);
        let grid_y = if level.height > 1 {
            (fy * level.height as f32 - 0.5) * (2.0 / (level.height - 1) as f32) - 1.0
        } else {
            0.0
        }
        .clamp(-1.0, 1.0);
        let x_half = (level.width - 1) as f32 * 0.5;
        let y_half = (level.height - 1) as f32 * 0.5;
        let x = grid_x * x_half + x_half;
        let y = grid_y * y_half + y_half;
        let x0 = x.floor();
        let y0 = y.floor();
        let x1 = (x + 1.0).floor();
        let y1 = (y + 1.0).floor();
        let x0i = x0.clamp(0.0, (level.width - 1) as f32) as usize;
        let y0i = y0.clamp(0.0, (level.height - 1) as f32) as usize;
        let x1i = x1.clamp(0.0, (level.width - 1) as f32) as usize;
        let y1i = y1.clamp(0.0, (level.height - 1) as f32) as usize;

        let idx00 = (y0i * level.width + x0i) * 3;
        let idx10 = (y0i * level.width + x1i) * 3;
        let idx01 = (y1i * level.width + x0i) * 3;
        let idx11 = (y1i * level.width + x1i) * 3;

        let c00 = [
            level.data[idx00],
            level.data[idx00 + 1],
            level.data[idx00 + 2],
        ];
        let c10 = [
            level.data[idx10],
            level.data[idx10 + 1],
            level.data[idx10 + 2],
        ];
        let c01 = [
            level.data[idx01],
            level.data[idx01 + 1],
            level.data[idx01 + 2],
        ];
        let c11 = [
            level.data[idx11],
            level.data[idx11 + 1],
            level.data[idx11 + 2],
        ];

        let weight_00 = (x1 - x) * (y1 - y);
        let weight_10 = (x - x0) * (y1 - y);
        let weight_01 = (x1 - x) * (y - y0);
        let weight_11 = (x - x0) * (y - y0);

        [
            c00[0] * weight_00 + c10[0] * weight_10 + c01[0] * weight_01 + c11[0] * weight_11,
            c00[1] * weight_00 + c10[1] * weight_10 + c01[1] * weight_01 + c11[1] * weight_11,
            c00[2] * weight_00 + c10[2] * weight_10 + c01[2] * weight_01 + c11[2] * weight_11,
        ]
    }

    #[test]
    fn lod_prefers_center_detail() {
        let source = make_checkerboard(64, 64);
        let settings = settings_for_mode(PyramidMode::Gaussian);
        let cache = make_cache(&source, settings.pyramid_depth);
        let patch = render_patch_f32(&source, &cache, &settings, settings.patch_size);

        let center = settings.patch_size / 2;
        let center_idx = (center * settings.patch_size + center) * 3;
        let corner_idx = 0;
        let center_value = patch[center_idx];
        let corner_value = patch[corner_idx];

        let src_center = source.data
            [(source.height / 2 * source.width + source.width / 2) * 3];
        let src_corner = source.data[0];

        let center_error = (center_value - src_center).abs();
        let corner_error = (corner_value - src_corner).abs();
        assert!(
            center_error <= corner_error,
            "expected center to preserve more detail (center error {center_error} vs corner {corner_error})"
        );
    }

    #[test]
    fn foveation_backends_match_across_settings() {
        let sources = [
            ("gradient", make_gradient(64, 64)),
            ("gradient_wide", make_gradient(80, 48)),
            ("checkerboard", make_checkerboard(64, 64)),
            ("radial", make_radial(64, 64)),
        ];
        let cases = [
            (0.5, 0.5, 0.25, 0.5, 16, 4),
            (0.2, 0.8, 0.1, 0.2, 12, 3),
            (0.8, 0.2, 0.4, 0.7, 24, 4),
            (0.05, 0.95, 0.15, 0.35, 8, 2),
            (0.5, 0.5, 0.35, 1.0, 64, 4),
            (0.25, 0.75, 0.2, 0.8, 64, 5),
            (0.8, 0.2, 0.6, 0.4, 64, 6),
            (0.1, 0.9, 0.12, 1.0, 64, 3),
            (0.9, 0.1, 0.45, 0.6, 64, 5),
        ];
        let modes = [PyramidMode::Gaussian, PyramidMode::Laplacian];
        let warp_modes = [FoveaWarpMode::Warped, FoveaWarpMode::Patched];
        let max_abs_threshold = 1e-3;
        let max_abs_threshold_wgsl = 6e-2;
        let mse_threshold = 1e-6;
        let mse_threshold_wgsl = 3e-5;
        let output_root = fovea_test_root();
        let mut saved_sources = HashSet::new();

        for (source_idx, (source_name, source)) in sources.iter().enumerate() {
            let save_outputs = true;
            if save_outputs {
                save_source_identity(&output_root, source_name, source, &mut saved_sources);
            }
            for mode in modes.iter().copied() {
                for warp_mode in warp_modes.iter().copied() {
                    for (case_idx, (mean_x, mean_y, radius, focus, patch_size, depth)) in
                        cases.iter().copied().enumerate()
                    {
                        let max_patch = source.width.min(source.height);
                        if patch_size > max_patch {
                            continue;
                        }
                        let mut settings = settings_for_mode(mode);
                        settings.mean_x = mean_x;
                        settings.mean_y = mean_y;
                        settings.radius_norm = radius;
                        settings.focus = focus;
                        settings.patch_size = patch_size;
                        settings.pyramid_depth = depth;
                        settings.warp_mode = warp_mode;

                        let cache = make_cache(source, settings.pyramid_depth);
                        let cpu = render_patch_f32(source, &cache, &settings, settings.patch_size);
                        settings.backend = FoveationBackendMode::Burn;
                        let burn = render_patch_burn(source, &settings);
                        settings.backend = FoveationBackendMode::Cubecl;
                        let cubecl = render_patch_burn(source, &settings);
                        let label = format!(
                            "source {source_idx} case {case_idx} mode {mode:?} warp {warp_mode:?}"
                        );
                        if save_outputs {
                            save_patch_output(
                                &output_root,
                                source_name,
                                source,
                                mode,
                                warp_mode,
                                case_idx,
                                mean_x,
                                mean_y,
                                radius,
                                focus,
                                patch_size,
                                depth,
                                "cpu",
                                &cpu,
                            );
                            save_patch_output(
                                &output_root,
                                source_name,
                                source,
                                mode,
                                warp_mode,
                                case_idx,
                                mean_x,
                                mean_y,
                                radius,
                                focus,
                                patch_size,
                                depth,
                                "burn",
                                &burn,
                            );
                            save_patch_output(
                                &output_root,
                                source_name,
                                source,
                                mode,
                                warp_mode,
                                case_idx,
                                mean_x,
                                mean_y,
                                radius,
                                focus,
                                patch_size,
                                depth,
                                "cubecl",
                                &cubecl,
                            );
                        }
                        assert_patch_close(
                            &format!("{label} burn vs cpu"),
                            &burn,
                            &cpu,
                            max_abs_threshold,
                            mse_threshold,
                        );
                        assert_patch_close(
                            &format!("{label} cubecl vs cpu"),
                            &cubecl,
                            &cpu,
                            max_abs_threshold,
                            mse_threshold,
                        );

                        settings.backend = FoveationBackendMode::Wgsl;
                        if let Some(gpu) = render_patch_gpu_f32(source, &cache, &settings) {
                            let cpu_f16 = quantize_f16(&cpu);
                            let burn_f16 = quantize_f16(&burn);
                            let cubecl_f16 = quantize_f16(&cubecl);
                            if save_outputs {
                                save_patch_output(
                                    &output_root,
                                    source_name,
                                    source,
                                    mode,
                                    warp_mode,
                                    case_idx,
                                    mean_x,
                                    mean_y,
                                    radius,
                                    focus,
                                    patch_size,
                                    depth,
                                    "wgsl",
                                    &gpu,
                                );
                            }
                            assert_patch_close(
                                &format!("{label} wgsl vs cpu"),
                                &gpu,
                                &cpu_f16,
                                max_abs_threshold_wgsl,
                                mse_threshold_wgsl,
                            );
                            assert_patch_close(
                                &format!("{label} burn vs wgsl"),
                                &burn_f16,
                                &gpu,
                                max_abs_threshold_wgsl,
                                mse_threshold_wgsl,
                            );
                            assert_patch_close(
                                &format!("{label} cubecl vs wgsl"),
                                &cubecl_f16,
                                &gpu,
                                max_abs_threshold_wgsl,
                                mse_threshold_wgsl,
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn gpu_pyramid_matches_cpu() {
        let source = make_gradient(32, 32);
        let depth = 4;
        let base = ImageLevel {
            width: source.width,
            height: source.height,
            data: source.data.clone(),
        };
        let gaussian = build_gaussian_pyramid_quantized(&base, depth);
        let laplacian = build_laplacian_pyramid_gpu(&gaussian);
        let laplacian = quantize_levels(laplacian);

        let Some((gpu_gaussian, gpu_residual)) = render_pyramid_gpu(&source, depth) else {
            return;
        };

        assert_levels_close("gaussian", &gaussian, &gpu_gaussian, 0.015);
        assert_levels_close(
            "residual",
            &laplacian,
            &gpu_residual[..laplacian.len()],
            0.02,
        );
    }

    fn render_patch_gpu(
        source: &SourceImage,
        cache: &PyramidCache,
        settings: &FoveationSettings,
    ) -> Option<Vec<u8>> {
        let instance = wgpu::Instance::default();
        let supports_format = |adapter: &wgpu::Adapter| {
            let features = adapter.get_texture_format_features(TextureFormat::Rgba16Float);
            features
                .flags
                .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
                && features
                    .flags
                    .contains(wgpu::TextureFormatFeatureFlags::STORAGE_WRITE_ONLY)
        };

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        if !supports_format(&adapter) {
            return None;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .ok()?;

        let gaussian_texture = create_mip_texture(&device, &queue, &cache.gaussian);
        let residual_levels = build_residual_levels(cache);
        let residual_texture = create_mip_texture(&device, &queue, &residual_levels);

        let output_size = settings.patch_size as u32;
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("foveation_output"),
            size: wgpu::Extent3d {
                width: output_size.max(1),
                height: output_size.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniform = uniform_for_settings(settings, source);
        let mut encased = bevy::render::render_resource::encase::UniformBuffer::new(Vec::new());
        encased.write(&uniform).ok()?;
        let uniform_bytes = encased.into_inner();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("foveation_uniform"),
            contents: &uniform_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("foveation_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("foveation_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("foveation_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &gaussian_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &residual_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("foveation_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("foveation_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("foveation_compute_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("foveation_compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups_x = (output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            let groups_y = (output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }

        let bytes_per_row = output_size * 4;
        let aligned_bytes_per_row = align_bytes_per_row(bytes_per_row);
        let buffer_size = (aligned_bytes_per_row * output_size) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("foveation_output_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bytes_per_row),
                    rows_per_image: Some(output_size),
                },
            },
            wgpu::Extent3d {
                width: output_size,
                height: output_size,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        let slice = output_buffer.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).ok();
        });
        device.poll(wgpu::PollType::Wait).ok();
        let map_result = rx.recv().ok()?;
        map_result.ok()?;
        let data = slice.get_mapped_range().to_vec();
        output_buffer.unmap();
        Some(trim_padded_rows(&data, output_size, aligned_bytes_per_row))
    }

    fn render_pyramid_gpu(
        source: &SourceImage,
        depth: usize,
    ) -> Option<(Vec<ImageLevel>, Vec<ImageLevel>)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        let features = adapter.get_texture_format_features(TextureFormat::Rgba16Float);
        if !features
            .flags
            .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
        {
            return None;
        }
        if !features
            .flags
            .contains(wgpu::TextureFormatFeatureFlags::STORAGE_WRITE_ONLY)
        {
            return None;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .ok()?;

        let input_texture = create_input_texture(&device, &queue, source);
        let gaussian_texture =
            create_storage_mip_texture(&device, source.width, source.height, depth as u32);
        let residual_texture =
            create_storage_mip_texture(&device, source.width, source.height, depth as u32);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let mut uniform_bytes = encode_pyramid_uniform(PyramidUniform {
            src_dst: UVec4::new(
                source.width as u32,
                source.height as u32,
                source.width as u32,
                source.height as u32,
            ),
        })?;
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pyramid_uniform"),
            contents: &uniform_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pyramid_shader"),
            source: wgpu::ShaderSource::Wgsl(PYRAMID_SHADER_SOURCE.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pyramid_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pyramid_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let downsample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pyramid_downsample_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("downsample"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let residual_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pyramid_residual_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("residual"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let mut sizes = Vec::with_capacity(depth.max(1));
        let mut width = source.width as u32;
        let mut height = source.height as u32;
        let levels = depth.max(1) as u32;
        for _ in 0..levels {
            sizes.push(UVec2::new(width.max(1), height.max(1)));
            width = (width / 2).max(1);
            height = (height / 2).max(1);
        }

        let dummy_view = input_texture.create_view(&wgpu::TextureViewDescriptor::default());
        if let Some(size) = sizes.first().copied() {
            let mut copy_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pyramid_copy_base"),
            });
            copy_encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &input_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &gaussian_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: 1,
                },
            );
            queue.submit([copy_encoder.finish()]);
        }

        let mut scratch_textures = Vec::new();
        for level in 0..levels.saturating_sub(1) {
            let src_size = sizes[level as usize];
            let dst_size = sizes[(level + 1) as usize];
            uniform_bytes = encode_pyramid_uniform(PyramidUniform {
                src_dst: UVec4::new(src_size.x, src_size.y, dst_size.x, dst_size.y),
            })?;
            queue.write_buffer(&uniform_buffer, 0, &uniform_bytes);
            let scratch = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("pyramid_scratch"),
                size: wgpu::Extent3d {
                    width: src_size.x,
                    height: src_size.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            scratch_textures.push(scratch);
            let scratch_ref = scratch_textures
                .last()
                .expect("scratch texture available");
            let mut copy_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pyramid_copy_level"),
            });
            copy_encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &gaussian_texture,
                    mip_level: level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: scratch_ref,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: src_size.x,
                    height: src_size.y,
                    depth_or_array_layers: 1,
                },
            );
            queue.submit([copy_encoder.finish()]);
            let src_view = scratch_ref.create_view(&wgpu::TextureViewDescriptor::default());
            let dst_view = gaussian_texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level + 1,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let bind_group = make_pyramid_bind_group(
                &device,
                &bind_group_layout,
                &src_view,
                &dummy_view,
                &sampler,
                &dst_view,
                &uniform_buffer,
            );
            let mut compute_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pyramid_downsample"),
                });
            dispatch_pyramid_pass(
                &mut compute_encoder,
                &downsample_pipeline,
                &bind_group,
                workgroup_dispatch(dst_size),
            );
            queue.submit([compute_encoder.finish()]);
        }

        for level in 0..levels.saturating_sub(1) {
            let size = sizes[level as usize];
            uniform_bytes = encode_pyramid_uniform(PyramidUniform {
                src_dst: UVec4::new(size.x, size.y, size.x, size.y),
            })?;
            queue.write_buffer(&uniform_buffer, 0, &uniform_bytes);
            let fine_view = gaussian_texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let coarse_view = gaussian_texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level + 1,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let dst_view = residual_texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let bind_group = make_pyramid_bind_group(
                &device,
                &bind_group_layout,
                &fine_view,
                &coarse_view,
                &sampler,
                &dst_view,
                &uniform_buffer,
            );
            let mut compute_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pyramid_residual"),
                });
            dispatch_pyramid_pass(
                &mut compute_encoder,
                &residual_pipeline,
                &bind_group,
                workgroup_dispatch(size),
            );
            queue.submit([compute_encoder.finish()]);
        }

        let gaussian_levels =
            read_pyramid_levels(&device, &queue, &gaussian_texture, &sizes)?;
        let residual_levels =
            read_pyramid_levels(&device, &queue, &residual_texture, &sizes)?;

        Some((gaussian_levels, residual_levels))
    }

    fn uniform_for_settings(settings: &FoveationSettings, source: &SourceImage) -> FoveationUniform {
        let image_size = Vec2::new(source.width as f32, source.height as f32).max(Vec2::ONE);
        let inv_image_size = Vec2::new(1.0 / image_size.x, 1.0 / image_size.y);
        let center = Vec2::new(
            settings.mean_x.clamp(0.0, 1.0) * image_size.x,
            settings.mean_y.clamp(0.0, 1.0) * image_size.y,
        );
        let sample = crate::FoveationSample {
            mean_x: settings.mean_x,
            mean_y: settings.mean_y,
            radius_norm: settings.radius_norm,
        };
        let radius_norm = crate::radius_norm_from_sample(&sample);
        let sigma_norm = crate::sigma_norm_from_settings(settings, &sample);
        let sigma_px = crate::sigma_px_from_norm(sigma_norm, source);
        let radius_px = crate::radius_px_from_norm(radius_norm, source);
        let sigma = Vec2::splat(sigma_px);
        let lod_sigma = foveation::lod_sigma_from_sigma(sigma_norm);
        let patch = settings.patch_size.max(1) as f32;
        let sample_scale = (radius_px * 2.0) / patch.max(1.0);
        let pyramid_levels = settings.pyramid_depth.max(2) as u32;
        let mode = match settings.mode {
            PyramidMode::Gaussian => 0,
            PyramidMode::Laplacian => 1,
        };
        let warp_mode = match settings.warp_mode {
            FoveaWarpMode::Warped => 0,
            FoveaWarpMode::Patched => 1,
        };

        FoveationUniform {
            image_size,
            inv_image_size,
            center,
            sigma,
            sample_scale,
            lod_sigma,
            patch_size: patch,
            pyramid_levels,
            mode,
            warp_mode,
            _pad0: 0,
            _pad1: 0,
        }
    }

    fn create_mip_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        levels: &[ImageLevel],
    ) -> wgpu::Texture {
        let base = levels.first().expect("mip levels");
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("foveation_mip_texture"),
            size: wgpu::Extent3d {
                width: base.width as u32,
                height: base.height as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        for (level_idx, level) in levels.iter().enumerate() {
            let bytes = level_to_f16_bytes(level);
            let bytes_per_row = 8 * level.width as u32;
            let aligned_bytes_per_row = align_bytes_per_row(bytes_per_row);
            let padded = if aligned_bytes_per_row == bytes_per_row {
                bytes
            } else {
                pad_rows(&bytes, level.width as u32, level.height as u32, 8, aligned_bytes_per_row)
            };
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level_idx as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &padded,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bytes_per_row),
                    rows_per_image: Some(level.height as u32),
                },
                wgpu::Extent3d {
                    width: level.width as u32,
                    height: level.height as u32,
                    depth_or_array_layers: 1,
                },
            );
        }
        texture
    }

    fn align_bytes_per_row(bytes_per_row: u32) -> u32 {
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        ((bytes_per_row + align - 1) / align) * align
    }

    fn assert_levels_close(label: &str, expected: &[ImageLevel], actual: &[ImageLevel], tol: f32) {
        assert_eq!(
            expected.len(),
            actual.len(),
            "{label} levels count mismatch"
        );
        for (idx, (exp, act)) in expected.iter().zip(actual.iter()).enumerate() {
            assert_eq!(exp.width, act.width, "{label} level {idx} width mismatch");
            assert_eq!(exp.height, act.height, "{label} level {idx} height mismatch");
            let max_diff = max_abs_diff_float(&exp.data, &act.data);
            assert!(
                max_diff <= tol,
                "{label} level {idx} max diff {max_diff} exceeded {tol}"
            );
        }
    }

    fn max_abs_diff_float(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    fn pad_rows(
        data: &[u8],
        width: u32,
        height: u32,
        bytes_per_pixel: u32,
        padded_bytes_per_row: u32,
    ) -> Vec<u8> {
        let row_bytes = width * bytes_per_pixel;
        let mut padded = vec![0u8; (padded_bytes_per_row * height) as usize];
        for row in 0..height {
            let src_start = (row * row_bytes) as usize;
            let dst_start = (row * padded_bytes_per_row) as usize;
            padded[dst_start..dst_start + row_bytes as usize]
                .copy_from_slice(&data[src_start..src_start + row_bytes as usize]);
        }
        padded
    }

    fn trim_padded_rows(data: &[u8], size: u32, padded_bytes_per_row: u32) -> Vec<u8> {
        let row_bytes = size * 4;
        let mut out = Vec::with_capacity((row_bytes * size) as usize);
        for row in 0..size {
            let start = (row * padded_bytes_per_row) as usize;
            out.extend_from_slice(&data[start..start + row_bytes as usize]);
        }
        out
    }

    fn trim_padded_rows_bytes(
        data: &[u8],
        width: u32,
        height: u32,
        bytes_per_pixel: u32,
        padded_bytes_per_row: u32,
    ) -> Vec<u8> {
        let row_bytes = width * bytes_per_pixel;
        let mut out = Vec::with_capacity((row_bytes * height) as usize);
        for row in 0..height {
            let start = (row * padded_bytes_per_row) as usize;
            out.extend_from_slice(&data[start..start + row_bytes as usize]);
        }
        out
    }

    fn create_input_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &SourceImage,
    ) -> wgpu::Texture {
        let level = ImageLevel {
            width: source.width,
            height: source.height,
            data: source.data.clone(),
        };
        let bytes = level_to_f16_bytes(&level);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pyramid_input"),
            size: wgpu::Extent3d {
                width: source.width as u32,
                height: source.height as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let bytes_per_row = 8 * source.width as u32;
        let aligned_bytes_per_row = align_bytes_per_row(bytes_per_row);
        let padded = if aligned_bytes_per_row == bytes_per_row {
            bytes
        } else {
            pad_rows(
                &bytes,
                source.width as u32,
                source.height as u32,
                8,
                aligned_bytes_per_row,
            )
        };
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned_bytes_per_row),
                rows_per_image: Some(source.height as u32),
            },
            wgpu::Extent3d {
                width: source.width as u32,
                height: source.height as u32,
                depth_or_array_layers: 1,
            },
        );
        texture
    }

    fn create_storage_mip_texture(
        device: &wgpu::Device,
        width: usize,
        height: usize,
        levels: u32,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pyramid_storage"),
            size: wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels.max(1),
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    fn encode_pyramid_uniform(uniform: PyramidUniform) -> Option<Vec<u8>> {
        let mut encased = bevy::render::render_resource::encase::UniformBuffer::new(Vec::new());
        encased.write(&uniform).ok()?;
        Some(encased.into_inner())
    }

    fn make_pyramid_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        fine_view: &wgpu::TextureView,
        coarse_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        dst_view: &wgpu::TextureView,
        uniform_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pyramid_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(fine_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(coarse_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(dst_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn dispatch_pyramid_pass(
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        dispatch: UVec2,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pyramid_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(dispatch.x, dispatch.y, 1);
    }

    fn read_pyramid_levels(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        sizes: &[UVec2],
    ) -> Option<Vec<ImageLevel>> {
        let mut levels = Vec::with_capacity(sizes.len());
        for (level, size) in sizes.iter().enumerate() {
            let bytes = read_texture_level(device, queue, texture, *size, level as u32)?;
            levels.push(ImageLevel {
                width: size.x as usize,
                height: size.y as usize,
                data: decode_f16_rgb(&bytes, size.x as usize, size.y as usize),
            });
        }
        Some(levels)
    }

    fn read_texture_level(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        size: UVec2,
        level: u32,
    ) -> Option<Vec<u8>> {
        let bytes_per_row = size.x * 8;
        let aligned_bytes_per_row = align_bytes_per_row(bytes_per_row);
        let buffer_size = (aligned_bytes_per_row * size.y) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pyramid_readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pyramid_readback_encoder"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bytes_per_row),
                    rows_per_image: Some(size.y),
                },
            },
            wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        let slice = output_buffer.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).ok();
        });
        device.poll(wgpu::PollType::Wait).ok();
        let map_result = rx.recv().ok()?;
        map_result.ok()?;
        let data = slice.get_mapped_range().to_vec();
        output_buffer.unmap();
        Some(trim_padded_rows_bytes(
            &data,
            size.x,
            size.y,
            8,
            aligned_bytes_per_row,
        ))
    }

    fn decode_f16_rgb(data: &[u8], width: usize, height: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(width * height * 3);
        let mut idx = 0usize;
        for _ in 0..(width * height) {
            let r = f16::from_bits(u16::from_le_bytes([data[idx], data[idx + 1]])).to_f32();
            let g = f16::from_bits(u16::from_le_bytes([data[idx + 2], data[idx + 3]])).to_f32();
            let b = f16::from_bits(u16::from_le_bytes([data[idx + 4], data[idx + 5]])).to_f32();
            idx += 8;
            out.push(r);
            out.push(g);
            out.push(b);
        }
        out
    }

    fn level_to_f16_bytes(level: &ImageLevel) -> Vec<u8> {
        let mut data = Vec::with_capacity(level.width * level.height * 8);
        let mut idx = 0usize;
        for _ in 0..(level.width * level.height) {
            let r = level.data[idx];
            let g = level.data[idx + 1];
            let b = level.data[idx + 2];
            idx += 3;
            push_f16(&mut data, r);
            push_f16(&mut data, g);
            push_f16(&mut data, b);
            push_f16(&mut data, 1.0);
        }
        data
    }
}
