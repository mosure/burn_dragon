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
    StorageTextureAccess, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    TextureViewDimension, UniformBuffer,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy_shader::Shader;
#[cfg(test)]
use half::f16;
use wgpu::Extent3d;

use crate::{
    FoveaWarpMode, FoveationBackendMode, FoveationNoiseSample, FoveationRuntime, FoveationSettings,
    PyramidMode, SourceImage, radius_norm_from_sample, radius_px_from_norm,
    resolve_foveation_sample, sigma_norm_from_settings, sigma_px_from_norm,
};
#[cfg(test)]
use crate::{ImageLevel, PyramidCache};
use burn_dragon_vision::foveation;
use burn_dragon_vision::{FOVEATION_SHADER, PYRAMID_SHADER};

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
                .add_systems(
                    Render,
                    dispatch_foveation_compute.in_set(RenderSystems::Render),
                );
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
    let bind_group_layout =
        render_device.create_bind_group_layout("foveation_bind_group_layout", &layout_entries);

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
    let Some(residual_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.residual_pipeline)
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
        let scratch_ref = scratch_textures.last().expect("scratch texture available");

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
        dispatch_pass(
            &mut compute_encoder,
            downsample_pipeline,
            &bind_group,
            dispatch,
        );
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
        dispatch_pass(
            &mut compute_encoder,
            residual_pipeline,
            &bind_group,
            dispatch,
        );
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
        data.fill(0.0);
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
mod tests;
