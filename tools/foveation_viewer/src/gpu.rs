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
    resolve_foveation_sample, FoveationNoiseSample, FoveationRuntime, FoveationSettings,
    PyramidMode, SourceImage,
};
#[cfg(test)]
use crate::{ImageLevel, PyramidCache};

const WORKGROUP_SIZE: u32 = 8;
const SHADER_SOURCE: &str = include_str!("foveation.wgsl");
const PYRAMID_SHADER_SOURCE: &str = include_str!("pyramid.wgsl");

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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct FoveationGpuParams {
    pub uniform: FoveationUniform,
    pub dispatch: UVec2,
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
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
            dispatch: UVec2::ONE,
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
    let patch = runtime.patch_size.max(1) as f32;
    let image_size = Vec2::new(source.width as f32, source.height as f32).max(Vec2::ONE);
    let inv_image_size = Vec2::new(1.0 / image_size.x, 1.0 / image_size.y);
    let sample = resolve_foveation_sample(&settings, &noise);
    let center = Vec2::new(
        sample.mean_x.clamp(0.0, 1.0) * image_size.x,
        sample.mean_y.clamp(0.0, 1.0) * image_size.y,
    );
    let radius = fovea_radius_pixels_norm(sample.radius_norm, &source);
    let sigma = Vec2::splat(radius * focus_scale(settings.focus));
    let lod_sigma = lod_sigma_from_focus(settings.focus);
    let sample_scale = (radius * 2.0) / patch.max(1.0);
    let pyramid_levels = settings.pyramid_depth.max(2) as u32;
    let mode = match settings.mode {
        PyramidMode::Gaussian => 0,
        PyramidMode::Laplacian => 1,
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
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let groups_x = (runtime.patch_size.max(1) as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    let groups_y = (runtime.patch_size.max(1) as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
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
    let mut changed = false;

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
}

fn prepare_foveation_uniform(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    params: Res<FoveationGpuParams>,
    mut bind_group: ResMut<FoveationGpuBindGroup>,
) {
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
    mut bind_group: ResMut<FoveationGpuBindGroup>,
) {
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
    let groups_x = (size.x + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    let groups_y = (size.y + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    UVec2::new(groups_x.max(1), groups_y.max(1))
}

fn focus_scale(value: f32) -> f32 {
    let t = value.clamp(0.0, 1.0);
    let log2 = -3.0 + t * 3.0;
    2.0_f32.powf(log2)
}

fn lod_sigma_from_focus(focus: f32) -> f32 {
    let t = focus.clamp(0.0, 1.0);
    let log2 = -2.0 + t * 3.0;
    2.0_f32.powf(log2)
}

fn fovea_radius_pixels_norm(radius_norm: f32, source: &SourceImage) -> f32 {
    let min_dim = source.width.min(source.height).max(1) as f32;
    (radius_norm.clamp(0.0, 1.0) * 0.5 * min_dim).max(1.0)
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
        build_gaussian_pyramid, build_laplacian_pyramid, lerp, render_patch, ImageLevel,
        PyramidCache,
    };
    use std::sync::mpsc;
    use wgpu::util::DeviceExt;

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

    fn settings_for_mode(mode: PyramidMode) -> FoveationSettings {
        let mut settings = FoveationSettings::default();
        settings.patch_size = 16;
        settings.pyramid_depth = 4;
        settings.radius_norm = 0.25;
        settings.focus = 0.5;
        settings.mean_x = 0.5;
        settings.mean_y = 0.5;
        settings.mode = mode;
        settings
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
        for y in 0..new_h {
            for x in 0..new_w {
                let mut accum = [0.0; 3];
                for dy in 0..2 {
                    for dx in 0..2 {
                        let sx = (x * 2 + dx).min(level.width - 1);
                        let sy = (y * 2 + dy).min(level.height - 1);
                        let idx = (sy * level.width + sx) * 3;
                        accum[0] += level.data[idx];
                        accum[1] += level.data[idx + 1];
                        accum[2] += level.data[idx + 2];
                    }
                }
                let idx = (y * new_w + x) * 3;
                data[idx] = accum[0] * 0.25;
                data[idx + 1] = accum[1] * 0.25;
                data[idx + 2] = accum[2] * 0.25;
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
        let x = fx.clamp(0.0, 1.0) * level.width as f32 - 0.5;
        let y = fy.clamp(0.0, 1.0) * level.height as f32 - 0.5;
        let x0 = x.floor();
        let y0 = y.floor();
        let x1 = x0 + 1.0;
        let y1 = y0 + 1.0;
        let tx = x - x0;
        let ty = y - y0;
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

        let a = [
            lerp(c00[0], c10[0], tx),
            lerp(c00[1], c10[1], tx),
            lerp(c00[2], c10[2], tx),
        ];
        let b = [
            lerp(c01[0], c11[0], tx),
            lerp(c01[1], c11[1], tx),
            lerp(c01[2], c11[2], tx),
        ];
        [
            lerp(a[0], b[0], ty),
            lerp(a[1], b[1], ty),
            lerp(a[2], b[2], ty),
        ]
    }

    #[test]
    fn lod_prefers_center_detail() {
        let source = make_checkerboard(64, 64);
        let settings = settings_for_mode(PyramidMode::Gaussian);
        let cache = make_cache(&source, settings.pyramid_depth);
        let patch = render_patch(&source, &cache, &settings, settings.patch_size);

        let center = settings.patch_size / 2;
        let center_idx = (center * settings.patch_size + center) * 4;
        let corner_idx = 0;
        let center_value = patch[center_idx] as f32 / 255.0;
        let corner_value = patch[corner_idx] as f32 / 255.0;

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
    fn gpu_matches_cpu_gaussian() {
        let source = make_gradient(64, 64);
        let settings = settings_for_mode(PyramidMode::Gaussian);
        let cache = make_cache(&source, settings.pyramid_depth);
        let cpu = render_patch(&source, &cache, &settings, settings.patch_size);
        let Some(gpu) = render_patch_gpu(&source, &cache, &settings) else {
            return;
        };
        let max_diff = max_abs_diff(&cpu, &gpu);
        assert!(
            max_diff <= 3,
            "gpu gaussian mismatch max diff {max_diff}"
        );
    }

    #[test]
    fn gpu_matches_cpu_laplacian() {
        let source = make_gradient(64, 64);
        let settings = settings_for_mode(PyramidMode::Laplacian);
        let cache = make_cache(&source, settings.pyramid_depth);
        let cpu = render_patch(&source, &cache, &settings, settings.patch_size);
        let Some(gpu) = render_patch_gpu(&source, &cache, &settings) else {
            return;
        };
        let max_diff = max_abs_diff(&cpu, &gpu);
        assert!(
            max_diff <= 4,
            "gpu laplacian mismatch max diff {max_diff}"
        );
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

    fn max_abs_diff(a: &[u8], b: &[u8]) -> u8 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap_or(0)
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
        let radius = fovea_radius_pixels(settings, source);
        let sigma = Vec2::splat(radius * focus_scale(settings.focus));
        let lod_sigma = lod_sigma_from_focus(settings.focus);
        let patch = settings.patch_size.max(1) as f32;
        let sample_scale = (radius * 2.0) / patch.max(1.0);
        let pyramid_levels = settings.pyramid_depth.max(2) as u32;
        let mode = match settings.mode {
            PyramidMode::Gaussian => 0,
            PyramidMode::Laplacian => 1,
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
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
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
