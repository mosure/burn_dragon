mod gpu;

use std::env;
use std::path::PathBuf;

use bevy::asset::RenderAssetUsages;
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::image::{ImageSampler, ImageSamplerDescriptor};
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::render::settings::{RenderCreation, WgpuFeatures, WgpuSettings};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::RenderPlugin;
use bevy::text::{TextColor, TextFont};
use bevy::ui::{
    BackgroundColor, ComputedNode, Display, FlexDirection, Node, Overflow, PositionType,
    UiGlobalTransform, UiRect, Val,
};
use bevy::window::PrimaryWindow;
use bevy_burn::{BevyBurnBridgePlugin, BevyBurnHandle, BindingDirection, BurnDevice, TransferKind};
use bevy_egui::EguiPlugin;
use bevy_inspector_egui::inspector_options::std_options::NumberDisplay;
use bevy_inspector_egui::prelude::*;
use bevy_inspector_egui::quick::ResourceInspectorPlugin;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_wgpu::Wgpu;
use burn_dragon_hatchling_vision::foveation;
use burn_dragon_hatchling_core::train::SaccadeFoveationSampler;
use burn_dragon_hatchling_core::{
    SpatialPositionalEncodingKind, VisionAttentionMode, VisionDragonHatchlingConfig,
    VisionFoveaSamplingMode, VisionFoveaWarpMode, VisionPyramidMode, VisionSaccadeConfig,
};
use half::f16;
use image::ImageReader;
use noise::{NoiseFn, OpenSimplex};

use gpu::{
    update_gpu_params, update_gpu_pyramid_textures, FoveationGpuImages, FoveationGpuPlugin,
    FoveationInputImage,
};

const UI_PADDING: f32 = 12.0;
const IMAGE_GAP: f32 = 12.0;
const PANEL_BG: Color = Color::srgb(0.05, 0.05, 0.06);
const OVERLAY_RING_SIZE: usize = 256;
const OVERLAY_RING_THICKNESS: f32 = 3.0;
const OVERLAY_RING_OUTER: [u8; 4] = [255, 80, 40, 220];
const OVERLAY_RING_INNER: [u8; 4] = [60, 200, 255, 220];
const FOVEA_PARAM_EPS: f32 = 1e-3;
const FOVEA_AA_THRESHOLD: f32 = 1.25;
// Avoid oversized GPU buffers when feeding full-resolution images to the burn backend.
const BURN_MAX_BUFFER_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const BURN_MAX_IMAGE_SIDE: usize = 1024;
type BurnBackend = Wgpu<f32>;

fn main() {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!(
            "usage: cargo run --manifest-path crates/foveation_viewer/Cargo.toml -- <image_path>"
        );
        std::process::exit(1);
    };
    let path = PathBuf::from(path);
    let source = load_image(&path).unwrap_or_else(|err| {
        eprintln!("failed to load image {path:?}: {err}");
        std::process::exit(1);
    });

    App::new()
        .insert_resource(source)
        .insert_resource(FoveationSettings::default())
        .insert_resource(FoveationNoiseSample::default())
        .insert_resource(FoveationRuntime::default())
        .insert_resource(CpuFoveationCache::default())
        .insert_non_send_resource(BurnFoveationState::default())
        .insert_resource(FoveationBackendState::default())
        .insert_resource(BurnHandleState::default())
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Foveation Viewer".to_string(),
                ..Default::default()
            }),
            ..Default::default()
        })
        .set(RenderPlugin {
            render_creation: RenderCreation::Automatic(WgpuSettings {
                features: WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                ..Default::default()
            }),
            ..Default::default()
        }))
        .add_plugins(FoveationGpuPlugin)
        .add_plugins(BevyBurnBridgePlugin::<BurnBackend>::default())
        .add_plugins(EguiPlugin::default())
        .add_plugins(ResourceInspectorPlugin::<FoveationSettings>::default())
        .register_type::<FoveationSettings>()
        .register_type::<PyramidMode>()
        .register_type::<FoveaWarpMode>()
        .register_type::<FoveationBackendMode>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                update_noise_mode,
                sync_patch_backend,
                update_patch_texture,
                attach_burn_handle,
                update_cpu_patch,
                update_burn_patch,
                update_gpu_pyramid_textures,
                update_gpu_params,
                update_foveation_overlay,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (update_pan_zoom_bounds, pan_zoom_input, apply_pan_zoom).chain(),
        )
        .run();
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Reflect, InspectorOptions)]
#[reflect(InspectorOptions)]
pub(crate) enum PyramidMode {
    Gaussian,
    Laplacian,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Reflect, InspectorOptions)]
#[reflect(InspectorOptions)]
pub(crate) enum FoveaWarpMode {
    Warped,
    Patched,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Reflect, InspectorOptions)]
#[reflect(InspectorOptions)]
pub(crate) enum FoveationBackendMode {
    Cpu,
    Wgsl,
    Burn,
    Cubecl,
}

#[derive(Resource, Reflect, InspectorOptions)]
#[reflect(Resource, InspectorOptions)]
pub(crate) struct FoveationSettings {
    #[inspector(min = 0.0, max = 1.0, speed = 0.001, display = NumberDisplay::Drag)]
    mean_x: f32,
    #[inspector(min = 0.0, max = 1.0, speed = 0.001, display = NumberDisplay::Drag)]
    mean_y: f32,
    #[inspector(min = 0.0, max = 1.0, speed = 0.005, display = NumberDisplay::Drag)]
    radius_norm: f32,
    #[inspector(min = 0.0, max = 1.0, speed = 0.01, display = NumberDisplay::Drag)]
    focus: f32,
    #[inspector(min = 1, max = 64)]
    patch_size: usize,
    #[inspector(min = 2, max = 8)]
    pyramid_depth: usize,
    mode: PyramidMode,
    warp_mode: FoveaWarpMode,
    backend: FoveationBackendMode,
    noise: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FoveationSample {
    mean_x: f32,
    mean_y: f32,
    radius_norm: f32,
}

#[derive(Resource, Debug)]
pub(crate) struct FoveationNoiseSample {
    mean_x: f32,
    mean_y: f32,
    radius_norm: f32,
}

impl Default for FoveationNoiseSample {
    fn default() -> Self {
        Self {
            mean_x: 0.5,
            mean_y: 0.5,
            radius_norm: 0.4,
        }
    }
}

pub(crate) fn resolve_foveation_sample(
    settings: &FoveationSettings,
    noise: &FoveationNoiseSample,
) -> FoveationSample {
    let (mean_x, mean_y, radius_norm) = if settings.noise {
        (noise.mean_x, noise.mean_y, noise.radius_norm)
    } else {
        (settings.mean_x, settings.mean_y, settings.radius_norm)
    };
    FoveationSample {
        mean_x: mean_x.clamp(FOVEA_PARAM_EPS, 1.0 - FOVEA_PARAM_EPS),
        mean_y: mean_y.clamp(FOVEA_PARAM_EPS, 1.0 - FOVEA_PARAM_EPS),
        radius_norm: radius_norm.clamp(FOVEA_PARAM_EPS, 1.0),
    }
}

impl Default for FoveationSettings {
    fn default() -> Self {
        Self {
            mean_x: 0.5,
            mean_y: 0.5,
            radius_norm: 0.4,
            focus: 1.0,
            patch_size: 16,
            pyramid_depth: 4,
            mode: PyramidMode::Gaussian,
            warp_mode: FoveaWarpMode::Warped,
            backend: FoveationBackendMode::Wgsl,
            noise: false,
        }
    }
}

#[derive(Resource)]
pub(crate) struct FoveationRuntime {
    patch_size: usize,
    noise_seed: OpenSimplex,
}

impl Default for FoveationRuntime {
    fn default() -> Self {
        Self {
            patch_size: 0,
            noise_seed: OpenSimplex::new(0),
        }
    }
}

#[derive(Resource, Clone)]
struct FoveationOutputHandles {
    cpu: Handle<Image>,
    burn: Handle<Image>,
}

#[derive(Resource)]
struct FoveationBackendState {
    backend: FoveationBackendMode,
}

impl Default for FoveationBackendState {
    fn default() -> Self {
        Self {
            backend: FoveationBackendMode::Wgsl,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CpuFoveationKey {
    width: usize,
    height: usize,
    depth: usize,
    mode: PyramidMode,
}

#[derive(Resource, Default)]
struct CpuFoveationCache {
    key: Option<CpuFoveationKey>,
    cache: Option<foveation::CpuPyramidCache>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BurnFoveationKey {
    width: usize,
    height: usize,
    patch_size: usize,
    depth: usize,
    mode: PyramidMode,
}

#[derive(Default)]
struct BurnFoveationState {
    device: Option<<BurnBackend as Backend>::Device>,
    sampler: Option<SaccadeFoveationSampler<BurnBackend>>,
    input: Option<Tensor<BurnBackend, 4>>,
    key: Option<BurnFoveationKey>,
}

#[derive(Resource, Default)]
struct BurnHandleState {
    attached: bool,
}

#[derive(Resource)]
pub(crate) struct SourceImage {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

#[derive(Clone)]
pub(crate) struct ImageLevel {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

#[cfg(test)]
#[derive(Resource, Default)]
pub(crate) struct PyramidCache {
    gaussian: Vec<ImageLevel>,
    laplacian: Vec<ImageLevel>,
    coarse: Option<ImageLevel>,
}

#[derive(Resource, Debug)]
struct PanZoomState {
    scale: f32,
    min_scale: f32,
    max_scale: f32,
    offset: Vec2,
    viewport_size: Vec2,
    inverse_scale_factor: f32,
    initialized: bool,
    dragging: bool,
    last_cursor: Option<Vec2>,
    touch_active_viewport: Option<Entity>,
    touch_last_center: Option<Vec2>,
    touch_last_distance: Option<f32>,
}

impl Default for PanZoomState {
    fn default() -> Self {
        Self {
            scale: 1.0,
            min_scale: 1.0,
            max_scale: 1.0,
            offset: Vec2::ZERO,
            viewport_size: Vec2::ZERO,
            inverse_scale_factor: 1.0,
            initialized: false,
            dragging: false,
            last_cursor: None,
            touch_active_viewport: None,
            touch_last_center: None,
            touch_last_distance: None,
        }
    }
}

#[derive(Resource, Clone, Copy, Debug)]
struct PanZoomTextures {
    input: Vec2,
    patch: Vec2,
}

impl PanZoomTextures {
    fn size(&self, kind: PanelKind) -> Vec2 {
        match kind {
            PanelKind::Input => self.input,
            PanelKind::Patch => self.patch,
        }
    }
}

#[derive(Resource, Debug)]
struct PanZoomStates {
    input: PanZoomState,
    patch: PanZoomState,
}

impl Default for PanZoomStates {
    fn default() -> Self {
        Self {
            input: PanZoomState::default(),
            patch: PanZoomState::default(),
        }
    }
}

#[derive(Resource, Default, Debug)]
struct ActiveFoveationView {
    patch: Option<Entity>,
}

impl PanZoomStates {
    fn state_mut(&mut self, kind: PanelKind) -> &mut PanZoomState {
        match kind {
            PanelKind::Input => &mut self.input,
            PanelKind::Patch => &mut self.patch,
        }
    }

    fn state(&self, kind: PanelKind) -> &PanZoomState {
        match kind {
            PanelKind::Input => &self.input,
            PanelKind::Patch => &self.patch,
        }
    }
}

#[derive(Component)]
struct PanZoomViewport;

#[derive(Component)]
struct PanZoomImage;

#[derive(Component, Clone, Copy)]
enum FoveationOverlayKind {
    Outer,
    Inner,
}

#[derive(Component)]
struct FoveationOverlay {
    kind: FoveationOverlayKind,
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
enum PanelKind {
    Input,
    Patch,
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    source: Res<SourceImage>,
    settings: Res<FoveationSettings>,
) {
    commands.spawn((
        Camera2d,
        Transform::default(),
        GlobalTransform::default(),
    ));

    let left_handle = create_bevy_image(&source, &mut images);
    let input_handle = create_gpu_input_image(&source, &mut images);
    let patch_size = settings
        .patch_size
        .max(1)
        .min(source.width.min(source.height));
    let right_handle = create_blank_image(patch_size, &mut images);
    let burn_handle = create_burn_image(patch_size, &mut images);
    let overlay_outer = create_overlay_ring(
        OVERLAY_RING_SIZE,
        OVERLAY_RING_THICKNESS,
        OVERLAY_RING_OUTER,
        &mut images,
    );
    let overlay_inner = create_overlay_ring(
        OVERLAY_RING_SIZE,
        OVERLAY_RING_THICKNESS,
        OVERLAY_RING_INNER,
        &mut images,
    );
    let image_size = Vec2::new(source.width as f32, source.height as f32);
    let patch_size = patch_size as f32;

    commands.insert_resource(FoveationInputImage {
        handle: input_handle,
    });
    commands.insert_resource(FoveationOutputHandles {
        cpu: right_handle.clone(),
        burn: burn_handle.clone(),
    });
    commands.insert_resource(PanZoomStates::default());
    commands.insert_resource(PanZoomTextures {
        input: image_size,
        patch: Vec2::new(patch_size, patch_size),
    });
    gpu::init_gpu_images(&mut commands, &mut images, right_handle.clone());

    let mut input_entity = None;
    let mut patch_entity = None;

    commands
        .spawn((
            Node {
                display: Display::Flex,
                flex_direction: FlexDirection::Row,
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                padding: UiRect::all(Val::Px(UI_PADDING)),
                column_gap: Val::Px(IMAGE_GAP),
                ..Default::default()
            },
            BackgroundColor(PANEL_BG),
        ))
        .with_children(|row| {
            input_entity = Some(spawn_panel(
                row,
                PanelKind::Input,
                "input",
                left_handle,
                image_size,
            ));
            patch_entity = Some(spawn_panel(
                row,
                PanelKind::Patch,
                "foveated patch",
                right_handle,
                Vec2::new(patch_size, patch_size),
            ));
        });

    commands.insert_resource(ActiveFoveationView { patch: patch_entity });

    if let Some(entity) = input_entity {
        commands.entity(entity).with_children(|parent| {
            parent.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Percent(0.0),
                    top: Val::Percent(0.0),
                    width: Val::Percent(0.0),
                    height: Val::Percent(0.0),
                    ..Default::default()
                },
                ImageNode::new(overlay_outer).with_mode(NodeImageMode::Stretch),
                FoveationOverlay {
                    kind: FoveationOverlayKind::Outer,
                },
            ));
            parent.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Percent(0.0),
                    top: Val::Percent(0.0),
                    width: Val::Percent(0.0),
                    height: Val::Percent(0.0),
                    ..Default::default()
                },
                ImageNode::new(overlay_inner).with_mode(NodeImageMode::Stretch),
                FoveationOverlay {
                    kind: FoveationOverlayKind::Inner,
                },
            ));
        });
    }
}

fn sync_patch_backend(
    settings: Res<FoveationSettings>,
    output: Res<FoveationOutputHandles>,
    mut state: ResMut<FoveationBackendState>,
    mut panels: Query<(&PanelKind, &mut ImageNode), With<PanZoomImage>>,
    mut burn_handles: Query<&mut BevyBurnHandle<BurnBackend>>,
) {
    if state.backend == settings.backend {
        return;
    }
    state.backend = settings.backend;
    let patch_handle = match settings.backend {
        FoveationBackendMode::Burn | FoveationBackendMode::Cubecl => output.burn.clone(),
        _ => output.cpu.clone(),
    };
    for (kind, mut image) in &mut panels {
        if *kind == PanelKind::Patch {
            image.image = patch_handle.clone();
        }
    }
    if !matches!(
        settings.backend,
        FoveationBackendMode::Burn | FoveationBackendMode::Cubecl
    ) {
        for mut handle in &mut burn_handles {
            handle.upload = false;
        }
    }
}

fn spawn_panel(
    parent: &mut ChildSpawnerCommands,
    kind: PanelKind,
    label: &str,
    handle: Handle<Image>,
    image_size: Vec2,
) -> Entity {
    let mut image_entity = None;

    parent
        .spawn(Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            width: Val::Percent(50.0),
            height: Val::Percent(100.0),
            padding: UiRect::all(Val::Px(8.0)),
            row_gap: Val::Px(6.0),
            ..Default::default()
        })
        .with_children(|panel| {
            panel.spawn((
                Text::new(label),
                TextFont {
                    font_size: 14.0,
                    ..Default::default()
                },
                TextColor(Color::WHITE),
            ));
            panel
                .spawn((
                    Node {
                        width: Val::Percent(100.0),
                        flex_grow: 1.0,
                        position_type: PositionType::Relative,
                        overflow: Overflow::clip(),
                        ..Default::default()
                    },
                    PanZoomViewport,
                    kind,
                ))
                .with_children(|viewport| {
                    image_entity = Some(
                        viewport
                            .spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: Val::Px(0.0),
                            top: Val::Px(0.0),
                            width: Val::Px(image_size.x),
                            height: Val::Px(image_size.y),
                            ..Default::default()
                        },
                        ImageNode::new(handle).with_mode(NodeImageMode::Stretch),
                        PanZoomImage,
                        kind,
                    ))
                            .id(),
                    );
                });
        });

    image_entity.expect("panel image entity")
}

fn update_noise_mode(
    time: Res<Time>,
    settings: Res<FoveationSettings>,
    runtime: Res<FoveationRuntime>,
    mut noise: ResMut<FoveationNoiseSample>,
) {
    if !settings.noise {
        return;
    }
    let t = time.elapsed_secs_f64();

    let noise_seed = &runtime.noise_seed;
    noise.mean_x = remap_noise_uniform(noise_seed.get([t * 0.15, 0.0]), 0.1, 0.9);
    noise.mean_y = remap_noise_uniform(noise_seed.get([t * 0.15, 10.0]), 0.1, 0.9);
    noise.radius_norm = remap_noise_uniform(noise_seed.get([t * 0.12, 40.0]), 0.05, 0.95);
}

fn attach_burn_handle(
    mut commands: Commands,
    settings: Res<FoveationSettings>,
    source: Res<SourceImage>,
    output: Res<FoveationOutputHandles>,
    active: Res<ActiveFoveationView>,
    burn_device: Option<Res<BurnDevice>>,
    mut state: ResMut<BurnHandleState>,
) {
    if state.attached {
        return;
    }
    let Some(burn_device) = burn_device else {
        return;
    };
    let Some(device) = burn_device.device().cloned() else {
        return;
    };
    let Some(entity) = active.patch else {
        return;
    };
    let patch = settings
        .patch_size
        .max(1)
        .min(source.width.min(source.height));
    let tensor = Tensor::<BurnBackend, 3>::zeros([patch, patch, 4], &device);
    commands.entity(entity).insert(BevyBurnHandle::<BurnBackend> {
        bevy_image: output.burn.clone(),
        tensor,
        upload: false,
        direction: BindingDirection::BurnToBevy,
        xfer: TransferKind::Gpu,
    });
    state.attached = true;
}

fn update_patch_texture(
    settings: Res<FoveationSettings>,
    source: Res<SourceImage>,
    mut runtime: ResMut<FoveationRuntime>,
    mut states: ResMut<PanZoomStates>,
    mut textures: ResMut<PanZoomTextures>,
    mut images: ResMut<Assets<Image>>,
    mut gpu_images: ResMut<FoveationGpuImages>,
    mut output_handles: ResMut<FoveationOutputHandles>,
    mut panels: Query<(&PanelKind, &mut ImageNode), With<PanZoomImage>>,
    mut burn_handles: Query<&mut BevyBurnHandle<BurnBackend>>,
) {
    let patch = settings
        .patch_size
        .max(1)
        .min(source.width.min(source.height));
    let patch_changed = runtime.patch_size != patch;
    runtime.patch_size = patch;
    textures.patch = Vec2::new(patch as f32, patch as f32);

    if !patch_changed {
        return;
    }
    output_handles.cpu = create_blank_image(patch, &mut images);
    output_handles.burn = create_burn_image(patch, &mut images);
    gpu_images.output = output_handles.cpu.clone();
    let patch_handle = match settings.backend {
        FoveationBackendMode::Burn | FoveationBackendMode::Cubecl => {
            output_handles.burn.clone()
        }
        _ => output_handles.cpu.clone(),
    };
    for (kind, mut image) in &mut panels {
        if *kind == PanelKind::Patch {
            image.image = patch_handle.clone();
        }
    }
    for mut handle in &mut burn_handles {
        let device = handle.tensor.device();
        handle.tensor = Tensor::<BurnBackend, 3>::zeros([patch, patch, 4], &device);
        handle.upload = false;
        handle.bevy_image = output_handles.burn.clone();
    }
    let state = states.state_mut(PanelKind::Patch);
    state.initialized = false;
    state.dragging = false;
    state.last_cursor = None;
}

fn update_cpu_patch(
    settings: Res<FoveationSettings>,
    source: Res<SourceImage>,
    runtime: Res<FoveationRuntime>,
    noise: Res<FoveationNoiseSample>,
    output: Res<FoveationOutputHandles>,
    mut images: ResMut<Assets<Image>>,
    mut cache: ResMut<CpuFoveationCache>,
) {
    if settings.backend != FoveationBackendMode::Cpu {
        return;
    }
    let patch = runtime
        .patch_size
        .max(1)
        .min(source.width.min(source.height));
    let key = CpuFoveationKey {
        width: source.width,
        height: source.height,
        depth: settings.pyramid_depth.max(1),
        mode: settings.mode,
    };
    if cache.key != Some(key) {
        let image = foveation::CpuImageLevel {
            width: source.width,
            height: source.height,
            data: source.data.clone(),
        };
        let mode = map_pyramid_mode_cpu(settings.mode);
        cache.cache = Some(foveation::build_pyramid_cache(image, key.depth, mode));
        cache.key = Some(key);
    }
    let Some(cache) = cache.cache.as_ref() else {
        return;
    };
    let sample = resolve_foveation_sample(&settings, &noise);
    let radius_norm = radius_norm_from_sample(&sample);
    let sigma_norm = sigma_norm_from_settings(&settings, &sample);
    let patch_data = foveation::render_foveated_patch_with_radius(
        cache,
        [sample.mean_x, sample.mean_y],
        sigma_norm,
        radius_norm,
        patch,
        map_warp_mode_cpu(settings.warp_mode),
    );
    let mut rgba = vec![0u8; patch * patch * 4];
    for idx in 0..(patch * patch) {
        let base = idx * 3;
        let out = idx * 4;
        rgba[out] = (patch_data[base].clamp(0.0, 1.0) * 255.0).round() as u8;
        rgba[out + 1] = (patch_data[base + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
        rgba[out + 2] = (patch_data[base + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
        rgba[out + 3] = 255;
    }
    if let Some(image) = images.get_mut(&output.cpu) {
        image.data = Some(rgba);
    }
}

fn update_burn_patch(
    settings: Res<FoveationSettings>,
    source: Res<SourceImage>,
    runtime: Res<FoveationRuntime>,
    noise: Res<FoveationNoiseSample>,
    burn_device: Option<Res<BurnDevice>>,
    mut state: NonSendMut<BurnFoveationState>,
    mut handles: Query<&mut BevyBurnHandle<BurnBackend>>,
) {
    if !matches!(
        settings.backend,
        FoveationBackendMode::Burn | FoveationBackendMode::Cubecl
    ) {
        return;
    }
    let Some(burn_device) = burn_device else {
        return;
    };
    let Some(device) = burn_device.device().cloned() else {
        return;
    };
    let (burn_width, burn_height) = burn_target_dims(&source, &settings);
    let patch = runtime
        .patch_size
        .max(1)
        .min(burn_width.min(burn_height));
    let key = BurnFoveationKey {
        width: burn_width,
        height: burn_height,
        patch_size: patch,
        depth: settings.pyramid_depth.max(1),
        mode: settings.mode,
    };
    let needs_rebuild = state.key != Some(key)
        || state.device.is_none()
        || state.sampler.is_none()
        || state.input.is_none();
    if needs_rebuild {
        let mut vision = make_minimal_vision_config(burn_width, burn_height, patch);
        vision.patch_size = patch;
        let mut saccade = VisionSaccadeConfig::default();
        saccade.mip_levels = key.depth;
        saccade.pyramid_mode = map_pyramid_mode(settings.mode);
        saccade.fovea_warp_mode = map_warp_mode(settings.warp_mode);
        if settings.backend == FoveationBackendMode::Cubecl {
            saccade.fovea_sampling_mode = VisionFoveaSamplingMode::Cubecl;
        }
        let mut sampler = SaccadeFoveationSampler::<BurnBackend>::new(vision, saccade, &device);
        let input =
            tensor_from_source_resized::<BurnBackend>(&source, burn_width, burn_height, &device);
        sampler.update_image(input.clone());
        state.device = Some(device.clone());
        state.sampler = Some(sampler);
        state.input = Some(input);
        state.key = Some(key);
    }

    let Some(sampler) = state.sampler.as_ref() else {
        return;
    };
    let sample = resolve_foveation_sample(&settings, &noise);
    let radius_norm = radius_norm_from_sample(&sample);
    let sigma_norm = sigma_norm_from_settings(&settings, &sample);
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
    let patch = sampler.sample_patch_with_radius(mean, sigma, radius);
    let rgba = patch_to_rgba::<BurnBackend>(patch);
    for mut handle in &mut handles {
        handle.tensor = rgba.clone();
        handle.upload = true;
    }
}

fn update_foveation_overlay(
    settings: Res<FoveationSettings>,
    source: Res<SourceImage>,
    noise: Res<FoveationNoiseSample>,
    mut overlays: Query<(&FoveationOverlay, &mut Node)>,
) {
    let width = source.width.max(1) as f32;
    let height = source.height.max(1) as f32;
    let sample = resolve_foveation_sample(&settings, &noise);
    let center_x = sample.mean_x.clamp(0.0, 1.0) * width;
    let center_y = sample.mean_y.clamp(0.0, 1.0) * height;
    let radius_norm = radius_norm_from_sample(&sample);
    let sigma_norm = sigma_norm_from_settings(&settings, &sample);
    let sigma_px = sigma_px_from_norm(sigma_norm, &source);
    let outer = radius_px_from_norm(radius_norm, &source);

    for (overlay, mut node) in &mut overlays {
        let ring = match overlay.kind {
            FoveationOverlayKind::Outer => outer,
            FoveationOverlayKind::Inner => sigma_px,
        };
        let left = (center_x - ring) / width * 100.0;
        let top = (center_y - ring) / height * 100.0;
        let size_x = (ring * 2.0 / width) * 100.0;
        let size_y = (ring * 2.0 / height) * 100.0;
        node.left = Val::Percent(left);
        node.top = Val::Percent(top);
        node.width = Val::Percent(size_x.max(0.0));
        node.height = Val::Percent(size_y.max(0.0));
    }
}

fn update_pan_zoom_bounds(
    mut states: ResMut<PanZoomStates>,
    textures: Res<PanZoomTextures>,
    viewports: Query<(&PanelKind, &ComputedNode), With<PanZoomViewport>>,
) {
    for (kind, node) in &viewports {
        if node.is_empty() {
            continue;
        }
        let viewport = node.size();
        let inverse_scale_factor = node.inverse_scale_factor();
        if viewport.x <= 0.0 || viewport.y <= 0.0 {
            continue;
        }

        let texture = textures.size(*kind);
        let scale_x = viewport.x / texture.x.max(1.0);
        let scale_y = viewport.y / texture.y.max(1.0);
        let min_scale = scale_x.min(scale_y).max(0.0001);
        let max_scale = (min_scale * 512.0).max(min_scale);

        let state = states.state_mut(*kind);
        let viewport_changed = state.viewport_size != viewport
            || (state.inverse_scale_factor - inverse_scale_factor).abs() > f32::EPSILON;

        state.viewport_size = viewport;
        state.inverse_scale_factor = inverse_scale_factor;
        state.min_scale = min_scale;
        state.max_scale = max_scale;

        if !state.initialized {
            state.scale = min_scale;
            state.offset = clamp_offset(Vec2::ZERO, viewport, texture, state.scale);
            state.initialized = true;
            continue;
        }

        if state.scale < min_scale || viewport_changed {
            state.scale = state.scale.max(min_scale).min(state.max_scale);
            state.offset = clamp_offset(state.offset, viewport, texture, state.scale);
        }
    }
}

fn pan_zoom_input(
    mut states: ResMut<PanZoomStates>,
    textures: Res<PanZoomTextures>,
    windows: Query<&Window, With<PrimaryWindow>>,
    viewports: Query<(Entity, &PanelKind, &ComputedNode, &UiGlobalTransform), With<PanZoomViewport>>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut scroll_events: MessageReader<MouseWheel>,
    touches: Res<Touches>,
) {
    let window: &Window = match windows.single() {
        Ok(window) => window,
        Err(_) => return,
    };
    let window_scale_factor = window.scale_factor() as f32;

    let mut touch_points: Vec<(u64, Vec2)> = touches
        .iter()
        .map(|touch| (touch.id(), touch.position()))
        .collect();

    if !touch_points.is_empty() {
        touch_points.sort_by_key(|(id, _)| *id);
        let mut active = None;
        for (entity, kind, node, transform) in &viewports {
            for (_, position) in &touch_points {
                let physical = *position * window_scale_factor;
                if node.contains_point(*transform, physical) {
                    active = Some((entity, *kind));
                    break;
                }
            }
            if active.is_some() {
                break;
            }
        }
        let Some((_active_entity, active_kind)) = active else {
            reset_touch_state(&mut states.input);
            reset_touch_state(&mut states.patch);
            return;
        };

        let state = states.state_mut(active_kind);
        let mut active_entity = state
            .touch_active_viewport
            .filter(|entity| viewports.get(*entity).is_ok());
        if active_entity.is_none() {
            state.touch_active_viewport = None;
        }
        if active_entity.is_none() {
            for (_, position) in &touch_points {
                let physical = *position * window_scale_factor;
                if let Some(active) = active_viewport(physical, &viewports) {
                    if active.kind == active_kind {
                        state.touch_active_viewport = Some(active.entity);
                        active_entity = Some(active.entity);
                    }
                    break;
                }
            }
        }

        let Some(active_entity) = active_entity else {
            state.touch_active_viewport = None;
            state.touch_last_center = None;
            state.touch_last_distance = None;
            state.dragging = false;
            state.last_cursor = None;
            return;
        };

        let (_, _, node, transform) = match viewports.get(active_entity) {
            Ok(parts) => parts,
            Err(_) => {
                state.touch_active_viewport = None;
                state.touch_last_center = None;
                state.touch_last_distance = None;
                state.dragging = false;
                state.last_cursor = None;
                return;
            }
        };

        let viewport_size = node.size();
        if viewport_size.x <= 0.0 || viewport_size.y <= 0.0 {
            state.touch_last_center = None;
            state.touch_last_distance = None;
            state.dragging = false;
            state.last_cursor = None;
            return;
        }

        state.viewport_size = viewport_size;
        state.inverse_scale_factor = node.inverse_scale_factor();

        let scale_factor = if node.inverse_scale_factor() > 0.0 {
            1.0 / node.inverse_scale_factor()
        } else {
            window_scale_factor.max(0.0001)
        };

        let touch_points: Vec<(u64, Vec2)> = touch_points
            .into_iter()
            .map(|(id, position)| (id, position * scale_factor))
            .collect();

        if touch_points.len() == 1 {
            let position = touch_points[0].1;
            if state.touch_last_distance.is_some() {
                state.touch_last_center = Some(position);
                state.touch_last_distance = None;
                state.dragging = false;
                state.last_cursor = None;
                return;
            }
            if let Some(last) = state.touch_last_center {
                let delta = position - last;
                state.offset += delta;
                let texture = textures.size(active_kind);
                state.offset = clamp_offset(state.offset, viewport_size, texture, state.scale);
            }
            state.touch_last_center = Some(position);
            state.touch_last_distance = None;
            state.dragging = false;
            state.last_cursor = None;
            return;
        }

        let pos_a = touch_points[0].1;
        let pos_b = touch_points[1].1;
        let center = (pos_a + pos_b) * 0.5;
        let distance = pos_a.distance(pos_b).max(0.0001);
        if state.touch_last_distance.is_none() {
            state.touch_last_center = Some(center);
            state.touch_last_distance = Some(distance);
            state.dragging = false;
            state.last_cursor = None;
            return;
        }
        if let (Some(last_center), Some(last_distance)) =
            (state.touch_last_center, state.touch_last_distance)
        {
            let delta = center - last_center;
            state.offset += delta;
            let zoom_factor = distance / last_distance;
            let next_scale = (state.scale * zoom_factor).clamp(state.min_scale, state.max_scale);
            if (next_scale - state.scale).abs() > f32::EPSILON {
                if let Some(pivot) = cursor_local_from_viewport(center, node, transform) {
                    let image_pos = (pivot - state.offset) / state.scale;
                    state.scale = next_scale;
                    state.offset = pivot - image_pos * state.scale;
                } else {
                    state.scale = next_scale;
                }
            }
            let texture = textures.size(active_kind);
            state.offset = clamp_offset(state.offset, viewport_size, texture, state.scale);
        }
        state.touch_last_center = Some(center);
        state.touch_last_distance = Some(distance);
        state.dragging = false;
        state.last_cursor = None;
        return;
    }

    reset_touch_only(&mut states.input);
    reset_touch_only(&mut states.patch);

    let Some(cursor) = window.physical_cursor_position() else {
        reset_drag_state(&mut states.input);
        reset_drag_state(&mut states.patch);
        return;
    };

    let active = active_viewport(cursor, &viewports);
    let (active_kind, cursor_local, viewport_size, inverse_scale) = if let Some(active) = active {
        (Some(active.kind), Some(active.cursor_local), active.size, active.inverse_scale_factor)
    } else {
        (None, None, Vec2::ZERO, 1.0)
    };
    for kind in [PanelKind::Input, PanelKind::Patch] {
        if Some(kind) != active_kind {
            let state = states.state_mut(kind);
            state.dragging = false;
            state.last_cursor = None;
        }
    }
    let Some(kind) = active_kind else {
        return;
    };
    let state = states.state_mut(kind);
    state.viewport_size = viewport_size;
    state.inverse_scale_factor = inverse_scale;

    let mut scroll = 0.0f32;
    for event in scroll_events.read() {
        let delta = match event.unit {
            MouseScrollUnit::Line => event.y,
            MouseScrollUnit::Pixel => event.y / 100.0,
        };
        scroll += delta;
    }

    if scroll.abs() > f32::EPSILON && cursor_local.is_some() {
        let zoom_factor = 1.1_f32.powf(scroll);
        let next_scale = (state.scale * zoom_factor).clamp(state.min_scale, state.max_scale);
        if (next_scale - state.scale).abs() > f32::EPSILON {
            let pivot = cursor_local.expect("cursor in viewport");
            let image_pos = (pivot - state.offset) / state.scale;
            state.scale = next_scale;
            state.offset = pivot - image_pos * state.scale;
            let texture = textures.size(kind);
            state.offset = clamp_offset(state.offset, viewport_size, texture, state.scale);
        }
    }

    let left_pressed = buttons.pressed(MouseButton::Left);
    if !left_pressed {
        state.dragging = false;
    } else if !state.dragging && cursor_local.is_some() {
        state.dragging = true;
    }

    if state.dragging {
        if let Some(last) = state.last_cursor {
            let delta = cursor - last;
            state.offset += delta;
            let texture = textures.size(kind);
            state.offset = clamp_offset(state.offset, viewport_size, texture, state.scale);
        }
    }

    state.last_cursor = Some(cursor);
}

fn apply_pan_zoom(
    states: Res<PanZoomStates>,
    textures: Res<PanZoomTextures>,
    mut images: Query<(&PanelKind, &mut Node), With<PanZoomImage>>,
) {
    for (kind, mut node) in &mut images {
        let state = states.state(*kind);
        if !state.initialized {
            continue;
        }
        let scale_factor = state.inverse_scale_factor;
        if scale_factor <= 0.0 {
            continue;
        }
        let texture = textures.size(*kind);
        let scaled = texture * state.scale;
        let scaled_logical = scaled * scale_factor;
        let offset_logical = state.offset * scale_factor;
        node.width = Val::Px(scaled_logical.x);
        node.height = Val::Px(scaled_logical.y);
        node.left = Val::Px(offset_logical.x);
        node.top = Val::Px(offset_logical.y);
    }
}

#[derive(Clone, Copy, Debug)]
struct ActiveViewport {
    entity: Entity,
    kind: PanelKind,
    cursor_local: Vec2,
    size: Vec2,
    inverse_scale_factor: f32,
}

fn cursor_local_from_viewport(
    cursor: Vec2,
    node: &ComputedNode,
    transform: &UiGlobalTransform,
) -> Option<Vec2> {
    let Some(local) = transform
        .try_inverse()
        .map(|affine| affine.transform_point2(cursor))
    else {
        return None;
    };
    let size = node.size();
    Some(local + size * 0.5)
}

fn active_viewport(
    cursor: Vec2,
    viewports: &Query<(Entity, &PanelKind, &ComputedNode, &UiGlobalTransform), With<PanZoomViewport>>,
) -> Option<ActiveViewport> {
    for (entity, kind, node, transform) in viewports.iter() {
        if node.contains_point(*transform, cursor) {
            let Some(local_top_left) = cursor_local_from_viewport(cursor, node, transform) else {
                continue;
            };
            return Some(ActiveViewport {
                entity,
                kind: *kind,
                cursor_local: local_top_left,
                size: node.size(),
                inverse_scale_factor: node.inverse_scale_factor(),
            });
        }
    }
    None
}

fn clamp_offset(offset: Vec2, viewport: Vec2, texture: Vec2, scale: f32) -> Vec2 {
    let scaled = texture * scale;
    let mut out = offset;
    if scaled.x <= viewport.x {
        out.x = (viewport.x - scaled.x) * 0.5;
    } else {
        let min_x = viewport.x - scaled.x;
        out.x = out.x.clamp(min_x, 0.0);
    }
    if scaled.y <= viewport.y {
        out.y = (viewport.y - scaled.y) * 0.5;
    } else {
        let min_y = viewport.y - scaled.y;
        out.y = out.y.clamp(min_y, 0.0);
    }
    out
}

fn reset_touch_state(state: &mut PanZoomState) {
    state.touch_active_viewport = None;
    state.touch_last_center = None;
    state.touch_last_distance = None;
    state.dragging = false;
    state.last_cursor = None;
}

fn reset_touch_only(state: &mut PanZoomState) {
    state.touch_active_viewport = None;
    state.touch_last_center = None;
    state.touch_last_distance = None;
}

fn reset_drag_state(state: &mut PanZoomState) {
    state.dragging = false;
    state.last_cursor = None;
}

#[cfg(test)]
pub(crate) fn render_patch(
    source: &SourceImage,
    cache: &PyramidCache,
    settings: &FoveationSettings,
    patch_size: usize,
) -> Vec<u8> {
    const SUBSAMPLES: usize = 4;
    let patch = patch_size.max(1);
    let width = patch;
    let height = patch;
    let sample = FoveationSample {
        mean_x: settings.mean_x,
        mean_y: settings.mean_y,
        radius_norm: settings.radius_norm,
    };
    let mean_x = sample.mean_x.clamp(0.0, 1.0);
    let mean_y = sample.mean_y.clamp(0.0, 1.0);
    let radius_norm = radius_norm_from_sample(&sample);
    let sigma_norm = sigma_norm_from_settings(settings, &sample);
    let sigma = sigma_px_from_norm(sigma_norm, source);
    let radius = radius_px_from_norm(radius_norm, source);
    let lod_sigma = foveation::lod_sigma_from_sigma(sigma_norm);
    let mut out = vec![0u8; width * height * 4];

    let half = patch as f32 * 0.5;
    let pixel_du = 1.0 / half;
    if matches!(settings.warp_mode, FoveaWarpMode::Patched) {
        let (level, level_w, level_h) = match settings.mode {
            PyramidMode::Gaussian => {
                let max_level = cache.gaussian.len().saturating_sub(1);
                let level = patched_level_from_radius(radius_norm, max_level);
                let level_w = cache
                    .gaussian
                    .get(level)
                    .map(|level| level.width)
                    .unwrap_or(source.width)
                    .max(1);
                let level_h = cache
                    .gaussian
                    .get(level)
                    .map(|level| level.height)
                    .unwrap_or(source.height)
                    .max(1);
                (level, level_w, level_h)
            }
            PyramidMode::Laplacian => {
                let max_level = cache.laplacian.len();
                let level = patched_level_from_radius(radius_norm, max_level);
                if level >= cache.laplacian.len() {
                    let (coarse_w, coarse_h) = cache
                        .coarse
                        .as_ref()
                        .or_else(|| cache.gaussian.last())
                        .map(|level| (level.width.max(1), level.height.max(1)))
                        .unwrap_or((source.width.max(1), source.height.max(1)));
                    (level, coarse_w, coarse_h)
                } else {
                    let level_w = cache.laplacian[level].width.max(1);
                    let level_h = cache.laplacian[level].height.max(1);
                    (level, level_w, level_h)
                }
            }
        };
        let center_x = mean_x * level_w as f32;
        let center_y = mean_y * level_h as f32;
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 + 0.5 - half;
                let dy = y as f32 + 0.5 - half;
                let fx = (center_x + dx) / level_w as f32;
                let fy = (center_y + dy) / level_h as f32;
                let sample = match settings.mode {
                    PyramidMode::Gaussian => {
                        let level_img = cache
                            .gaussian
                            .get(level)
                            .unwrap_or_else(|| cache.gaussian.first().expect("gaussian level"));
                        sample_bilinear(level_img, fx, fy)
                    }
                    PyramidMode::Laplacian => {
                        let coarse = cache
                            .coarse
                            .as_ref()
                            .unwrap_or_else(|| cache.gaussian.last().expect("coarse level"));
                        sample_laplacian_at(&cache.laplacian, coarse, level, fx, fy)
                    }
                };
                let idx = (y * width + x) * 4;
                out[idx] = (sample[0].clamp(0.0, 1.0) * 255.0).round() as u8;
                out[idx + 1] = (sample[1].clamp(0.0, 1.0) * 255.0).round() as u8;
                out[idx + 2] = (sample[2].clamp(0.0, 1.0) * 255.0).round() as u8;
                out[idx + 3] = 255;
            }
        }
        return out;
    }
    let center_x = mean_x * source.width as f32;
    let center_y = mean_y * source.height as f32;
    for y in 0..height {
        for x in 0..width {
            let base_dx = x as f32 + 0.5 - half;
            let base_dy = y as f32 + 0.5 - half;
            let ux_base = base_dx / half;
            let uy_base = base_dy / half;
            let warp_x_base = foveated_warp(ux_base, sigma, radius);
            let warp_y_base = foveated_warp(uy_base, sigma, radius);
            let local_scale_base = warp_x_base
                .deriv
                .abs()
                .max(warp_y_base.deriv.abs())
                * pixel_du;
            let mut color = [0.0; 3];
            let mut count = 0.0;
            if local_scale_base <= FOVEA_AA_THRESHOLD {
                let offset_x = warp_x_base.offset;
                let offset_y = warp_y_base.offset;
                let img_x = center_x + offset_x;
                let img_y = center_y + offset_y;
                let fx = img_x / source.width as f32;
                let fy = img_y / source.height as f32;
                let sample = match settings.mode {
                    PyramidMode::Gaussian => sample_gaussian_foveated(
                        &cache.gaussian,
                        offset_x,
                        offset_y,
                        sigma,
                        sigma,
                        local_scale_base,
                        lod_sigma,
                        fx,
                        fy,
                        settings.warp_mode,
                    ),
                    PyramidMode::Laplacian => sample_laplacian_foveated(
                        &cache.laplacian,
                        cache.coarse.as_ref(),
                        offset_x,
                        offset_y,
                        sigma,
                        sigma,
                        local_scale_base,
                        lod_sigma,
                        fx,
                        fy,
                        settings.warp_mode,
                    ),
                };
                color = sample;
                count = 1.0;
            } else {
                for sy in 0..SUBSAMPLES {
                    for sx in 0..SUBSAMPLES {
                        let jitter_x = (sx as f32 + 0.5) / SUBSAMPLES as f32 - 0.5;
                        let jitter_y = (sy as f32 + 0.5) / SUBSAMPLES as f32 - 0.5;
                        let ux = (base_dx + jitter_x) / half;
                        let uy = (base_dy + jitter_y) / half;
                        let warp_x = foveated_warp(ux, sigma, radius);
                        let warp_y = foveated_warp(uy, sigma, radius);
                        let offset_x = warp_x.offset;
                        let offset_y = warp_y.offset;
                        let local_scale = warp_x
                            .deriv
                            .abs()
                            .max(warp_y.deriv.abs())
                            * pixel_du;
                        let img_x = center_x + offset_x;
                        let img_y = center_y + offset_y;
                        let fx = img_x / source.width as f32;
                        let fy = img_y / source.height as f32;
                        let sample = match settings.mode {
                            PyramidMode::Gaussian => sample_gaussian_foveated(
                                &cache.gaussian,
                                offset_x,
                                offset_y,
                                sigma,
                                sigma,
                                local_scale,
                                lod_sigma,
                                fx,
                                fy,
                                settings.warp_mode,
                            ),
                            PyramidMode::Laplacian => sample_laplacian_foveated(
                                &cache.laplacian,
                                cache.coarse.as_ref(),
                                offset_x,
                                offset_y,
                                sigma,
                                sigma,
                                local_scale,
                                lod_sigma,
                                fx,
                                fy,
                                settings.warp_mode,
                            ),
                        };
                        color[0] += sample[0];
                        color[1] += sample[1];
                        color[2] += sample[2];
                        count += 1.0;
                    }
                }
            }
            if count > 0.0 {
                color[0] /= count;
                color[1] /= count;
                color[2] /= count;
            }
            let idx = (y * width + x) * 4;
            out[idx] = (color[0].clamp(0.0, 1.0) * 255.0).round() as u8;
            out[idx + 1] = (color[1].clamp(0.0, 1.0) * 255.0).round() as u8;
            out[idx + 2] = (color[2].clamp(0.0, 1.0) * 255.0).round() as u8;
            out[idx + 3] = 255;
        }
    }
    out
}

#[cfg(test)]
pub(crate) fn render_patch_f32(
    source: &SourceImage,
    cache: &PyramidCache,
    settings: &FoveationSettings,
    patch_size: usize,
) -> Vec<f32> {
    const SUBSAMPLES: usize = 4;
    let patch = patch_size.max(1);
    let width = patch;
    let height = patch;
    let sample = FoveationSample {
        mean_x: settings.mean_x,
        mean_y: settings.mean_y,
        radius_norm: settings.radius_norm,
    };
    let mean_x = sample.mean_x.clamp(0.0, 1.0);
    let mean_y = sample.mean_y.clamp(0.0, 1.0);
    let radius_norm = radius_norm_from_sample(&sample);
    let sigma_norm = sigma_norm_from_settings(settings, &sample);
    let sigma = sigma_px_from_norm(sigma_norm, source);
    let radius = radius_px_from_norm(radius_norm, source);
    let lod_sigma = foveation::lod_sigma_from_sigma(sigma_norm);
    let mut out = vec![0.0f32; width * height * 3];

    let half = patch as f32 * 0.5;
    let pixel_du = 1.0 / half;
    if matches!(settings.warp_mode, FoveaWarpMode::Patched) {
        let (level, level_w, level_h) = match settings.mode {
            PyramidMode::Gaussian => {
                let max_level = cache.gaussian.len().saturating_sub(1);
                let level = patched_level_from_radius(radius_norm, max_level);
                let level_w = cache
                    .gaussian
                    .get(level)
                    .map(|level| level.width)
                    .unwrap_or(source.width)
                    .max(1);
                let level_h = cache
                    .gaussian
                    .get(level)
                    .map(|level| level.height)
                    .unwrap_or(source.height)
                    .max(1);
                (level, level_w, level_h)
            }
            PyramidMode::Laplacian => {
                let max_level = cache.laplacian.len();
                let level = patched_level_from_radius(radius_norm, max_level);
                if level >= cache.laplacian.len() {
                    let (coarse_w, coarse_h) = cache
                        .coarse
                        .as_ref()
                        .or_else(|| cache.gaussian.last())
                        .map(|level| (level.width.max(1), level.height.max(1)))
                        .unwrap_or((source.width.max(1), source.height.max(1)));
                    (level, coarse_w, coarse_h)
                } else {
                    let level_w = cache.laplacian[level].width.max(1);
                    let level_h = cache.laplacian[level].height.max(1);
                    (level, level_w, level_h)
                }
            }
        };
        let center_x = mean_x * level_w as f32;
        let center_y = mean_y * level_h as f32;
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 + 0.5 - half;
                let dy = y as f32 + 0.5 - half;
                let fx = (center_x + dx) / level_w as f32;
                let fy = (center_y + dy) / level_h as f32;
                let sample = match settings.mode {
                    PyramidMode::Gaussian => {
                        let level_img = cache
                            .gaussian
                            .get(level)
                            .unwrap_or_else(|| cache.gaussian.first().expect("gaussian level"));
                        sample_bilinear(level_img, fx, fy)
                    }
                    PyramidMode::Laplacian => {
                        let coarse = cache
                            .coarse
                            .as_ref()
                            .unwrap_or_else(|| cache.gaussian.last().expect("coarse level"));
                        sample_laplacian_at(&cache.laplacian, coarse, level, fx, fy)
                    }
                };
                let idx = (y * width + x) * 3;
                out[idx] = sample[0];
                out[idx + 1] = sample[1];
                out[idx + 2] = sample[2];
            }
        }
        return out;
    }
    let center_x = mean_x * source.width as f32;
    let center_y = mean_y * source.height as f32;
    for y in 0..height {
        for x in 0..width {
            let base_dx = x as f32 + 0.5 - half;
            let base_dy = y as f32 + 0.5 - half;
            let ux_base = base_dx / half;
            let uy_base = base_dy / half;
            let warp_x_base = foveated_warp(ux_base, sigma, radius);
            let warp_y_base = foveated_warp(uy_base, sigma, radius);
            let local_scale_base = warp_x_base
                .deriv
                .abs()
                .max(warp_y_base.deriv.abs())
                * pixel_du;
            let mut color = [0.0; 3];
            let mut count = 0.0;
            if local_scale_base <= FOVEA_AA_THRESHOLD {
                let offset_x = warp_x_base.offset;
                let offset_y = warp_y_base.offset;
                let img_x = center_x + offset_x;
                let img_y = center_y + offset_y;
                let fx = img_x / source.width as f32;
                let fy = img_y / source.height as f32;
                let sample = match settings.mode {
                    PyramidMode::Gaussian => sample_gaussian_foveated(
                        &cache.gaussian,
                        offset_x,
                        offset_y,
                        sigma,
                        sigma,
                        local_scale_base,
                        lod_sigma,
                        fx,
                        fy,
                        settings.warp_mode,
                    ),
                    PyramidMode::Laplacian => sample_laplacian_foveated(
                        &cache.laplacian,
                        cache.coarse.as_ref(),
                        offset_x,
                        offset_y,
                        sigma,
                        sigma,
                        local_scale_base,
                        lod_sigma,
                        fx,
                        fy,
                        settings.warp_mode,
                    ),
                };
                color = sample;
                count = 1.0;
            } else {
                for sy in 0..SUBSAMPLES {
                    for sx in 0..SUBSAMPLES {
                        let jitter_x = (sx as f32 + 0.5) / SUBSAMPLES as f32 - 0.5;
                        let jitter_y = (sy as f32 + 0.5) / SUBSAMPLES as f32 - 0.5;
                        let ux = (base_dx + jitter_x) / half;
                        let uy = (base_dy + jitter_y) / half;
                        let warp_x = foveated_warp(ux, sigma, radius);
                        let warp_y = foveated_warp(uy, sigma, radius);
                        let offset_x = warp_x.offset;
                        let offset_y = warp_y.offset;
                        let local_scale = warp_x
                            .deriv
                            .abs()
                            .max(warp_y.deriv.abs())
                            * pixel_du;
                        let img_x = center_x + offset_x;
                        let img_y = center_y + offset_y;
                        let fx = img_x / source.width as f32;
                        let fy = img_y / source.height as f32;
                        let sample = match settings.mode {
                            PyramidMode::Gaussian => sample_gaussian_foveated(
                                &cache.gaussian,
                                offset_x,
                                offset_y,
                                sigma,
                                sigma,
                                local_scale,
                                lod_sigma,
                                fx,
                                fy,
                                settings.warp_mode,
                            ),
                            PyramidMode::Laplacian => sample_laplacian_foveated(
                                &cache.laplacian,
                                cache.coarse.as_ref(),
                                offset_x,
                                offset_y,
                                sigma,
                                sigma,
                                local_scale,
                                lod_sigma,
                                fx,
                                fy,
                                settings.warp_mode,
                            ),
                        };
                        color[0] += sample[0];
                        color[1] += sample[1];
                        color[2] += sample[2];
                        count += 1.0;
                    }
                }
            }
            if count > 0.0 {
                color[0] /= count;
                color[1] /= count;
                color[2] /= count;
            }
            let idx = (y * width + x) * 3;
            out[idx] = color[0];
            out[idx + 1] = color[1];
            out[idx + 2] = color[2];
        }
    }
    out
}

#[cfg(test)]
const SQRT2: f32 = std::f32::consts::SQRT_2;

#[cfg(test)]
const PI: f32 = std::f32::consts::PI;

#[cfg(test)]
const ERF_A: f32 = 0.147;

#[cfg(test)]
const SQRT_PI_OVER_2: f32 = 0.88622692545;

#[cfg(test)]
fn erf_approx(x: f32) -> f32 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

#[cfg(test)]
fn erfinv_approx(x: f32) -> f32 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let xx = x.clamp(-0.999, 0.999);
    let ln = (1.0 - xx * xx).ln();
    let term = 2.0 / (PI * ERF_A) + ln * 0.5;
    let inside = (term * term - ln / ERF_A).max(0.0);
    let result = (inside.sqrt() - term).max(0.0);
    sign * result.sqrt()
}

#[cfg(test)]
fn foveated_warp(u: f32, sigma: f32, radius: f32) -> FoveaWarp {
    let sigma = sigma.max(1e-3);
    let radius = radius.max(1e-3);
    let k = radius / sigma;
    let u_max = erf_approx(k / SQRT2).min(0.999);
    let u_scaled = u.clamp(-1.0, 1.0) * u_max;
    let erf_inv = erfinv_approx(u_scaled);
    let offset = sigma * SQRT2 * erf_inv;
    let deriv = sigma * SQRT2 * u_max * SQRT_PI_OVER_2 * (erf_inv * erf_inv).exp();
    FoveaWarp { offset, deriv }
}

#[cfg(test)]
struct FoveaWarp {
    offset: f32,
    deriv: f32,
}

#[cfg(test)]
const LOD_WINDOW: i32 = 3;

#[cfg(test)]
pub(crate) fn sample_gaussian_foveated(
    levels: &[ImageLevel],
    dx: f32,
    dy: f32,
    sigma_x: f32,
    sigma_y: f32,
    local_scale: f32,
    lod_sigma: f32,
    fx: f32,
    fy: f32,
    warp_mode: FoveaWarpMode,
) -> [f32; 3] {
    if levels.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let max_level = (levels.len().saturating_sub(1)) as f32;
    let lod_center = compute_lod(dx, dy, sigma_x, sigma_y, max_level, local_scale);
    if matches!(warp_mode, FoveaWarpMode::Patched) {
        let level = (lod_center + 0.5).floor().clamp(0.0, max_level) as usize;
        return sample_bilinear(&levels[level], fx, fy);
    }
    let mut color = [0.0; 3];
    let mut weight_sum = 0.0;
    let base = lod_center.floor() as i32;
    let start = (base - LOD_WINDOW).max(0);
    let end = (base + LOD_WINDOW).min(levels.len() as i32 - 1);
    for level_idx in start..=end {
        let level = &levels[level_idx as usize];
        let diff = (level_idx as f32 - lod_center) / lod_sigma.max(1e-3);
        let weight = (-0.5 * diff * diff).exp();
        let sample = sample_bilinear(level, fx, fy);
        color[0] += sample[0] * weight;
        color[1] += sample[1] * weight;
        color[2] += sample[2] * weight;
        weight_sum += weight;
    }
    if weight_sum > 0.0 {
        color[0] /= weight_sum;
        color[1] /= weight_sum;
        color[2] /= weight_sum;
    }
    color
}

#[cfg(test)]
pub(crate) fn sample_laplacian_foveated(
    residuals: &[ImageLevel],
    coarse: Option<&ImageLevel>,
    dx: f32,
    dy: f32,
    sigma_x: f32,
    sigma_y: f32,
    local_scale: f32,
    lod_sigma: f32,
    fx: f32,
    fy: f32,
    warp_mode: FoveaWarpMode,
) -> [f32; 3] {
    let Some(coarse) = coarse else {
        return [0.0, 0.0, 0.0];
    };
    let max_level = residuals.len() as f32;
    let lod_center = compute_lod(dx, dy, sigma_x, sigma_y, max_level, local_scale);
    if matches!(warp_mode, FoveaWarpMode::Patched) {
        let level = (lod_center + 0.5).floor().clamp(0.0, max_level) as usize;
        return sample_laplacian_at(residuals, coarse, level, fx, fy);
    }
    let mut color = [0.0; 3];
    let mut weight_sum = 0.0;
    let base = lod_center.floor() as i32;
    let start = (base - LOD_WINDOW).max(0);
    let end = (base + LOD_WINDOW).min(residuals.len() as i32);
    for level_idx in start..=end {
        let diff = (level_idx as f32 - lod_center) / lod_sigma.max(1e-3);
        let weight = (-0.5 * diff * diff).exp();
        let sample = sample_laplacian_at(residuals, coarse, level_idx as usize, fx, fy);
        color[0] += sample[0] * weight;
        color[1] += sample[1] * weight;
        color[2] += sample[2] * weight;
        weight_sum += weight;
    }
    if weight_sum > 0.0 {
        color[0] /= weight_sum;
        color[1] /= weight_sum;
        color[2] /= weight_sum;
    }
    color
}

#[cfg(test)]
fn patched_level_from_radius(radius_norm: f32, max_level: usize) -> usize {
    if max_level == 0 {
        return 0;
    }
    let max_level_f = max_level as f32;
    let level = (radius_norm.clamp(0.0, 1.0) * max_level_f + 0.5)
        .floor()
        .clamp(0.0, max_level_f);
    level as usize
}

#[cfg(test)]
pub(crate) fn compute_lod(
    dx: f32,
    dy: f32,
    sigma_x: f32,
    sigma_y: f32,
    max_level: f32,
    local_scale: f32,
) -> f32 {
    if max_level <= 0.0 {
        return 0.0;
    }
    let sx = sigma_x.max(1e-3);
    let sy = sigma_y.max(1e-3);
    let dist = ((dx * dx) / (sx * sx) + (dy * dy) / (sy * sy)).sqrt();
    let lod_dist = if dist <= 1.0 {
        0.0
    } else {
        dist.ln() / std::f32::consts::LN_2
    };
    let lod_scale = if local_scale <= FOVEA_AA_THRESHOLD {
        0.0
    } else {
        (local_scale / FOVEA_AA_THRESHOLD).ln() / std::f32::consts::LN_2
    };
    lod_dist.max(lod_scale).clamp(0.0, max_level)
}

#[cfg(test)]
fn sample_laplacian_at(
    residuals: &[ImageLevel],
    coarse: &ImageLevel,
    start_idx: usize,
    fx: f32,
    fy: f32,
) -> [f32; 3] {
    let mut color = sample_bilinear(coarse, fx, fy);
    for (idx, residual) in residuals.iter().enumerate() {
        if idx < start_idx {
            continue;
        }
        let sample = sample_bilinear(residual, fx, fy);
        color[0] += sample[0];
        color[1] += sample[1];
        color[2] += sample[2];
    }
    color
}

#[cfg(test)]
pub(crate) fn build_gaussian_pyramid(base: &ImageLevel, depth: usize) -> Vec<ImageLevel> {
    let mut out = Vec::with_capacity(depth.max(1));
    out.push(base.clone());
    for _ in 1..depth {
        let next = downsample(out.last().expect("pyramid level"));
        out.push(next);
    }
    out
}

#[cfg(test)]
pub(crate) fn build_laplacian_pyramid(gaussian: &[ImageLevel]) -> (Vec<ImageLevel>, ImageLevel) {
    if gaussian.is_empty() {
        return (
            Vec::new(),
            ImageLevel {
                width: 1,
                height: 1,
                data: vec![0.0; 3],
            },
        );
    }
    let mut residuals = Vec::with_capacity(gaussian.len().saturating_sub(1));
    for idx in 0..gaussian.len().saturating_sub(1) {
        let current = &gaussian[idx];
        let next = &gaussian[idx + 1];
        let up = resample(next, current.width, current.height);
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
    let coarse = gaussian.last().cloned().expect("coarse");
    (residuals, coarse)
}

#[cfg(test)]
fn downsample(level: &ImageLevel) -> ImageLevel {
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
                    let sample = get_pixel(level, sx, sy);
                    accum[0] += sample[0] * weight;
                    accum[1] += sample[1] * weight;
                    accum[2] += sample[2] * weight;
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

#[cfg(test)]
fn resample(level: &ImageLevel, width: usize, height: usize) -> ImageLevel {
    let mut data = vec![0.0; width * height * 3];
    for y in 0..height {
        let fy = (y as f32 + 0.5) / height as f32;
        for x in 0..width {
            let fx = (x as f32 + 0.5) / width as f32;
            let sample = sample_bilinear(level, fx, fy);
            let idx = (y * width + x) * 3;
            data[idx] = sample[0];
            data[idx + 1] = sample[1];
            data[idx + 2] = sample[2];
        }
    }
    ImageLevel { width, height, data }
}

#[cfg(test)]
fn sample_bilinear(level: &ImageLevel, fx: f32, fy: f32) -> [f32; 3] {
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

    let c00 = get_pixel(level, x0i, y0i);
    let c10 = get_pixel(level, x1i, y0i);
    let c01 = get_pixel(level, x0i, y1i);
    let c11 = get_pixel(level, x1i, y1i);

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

#[cfg(test)]
fn get_pixel(level: &ImageLevel, x: usize, y: usize) -> [f32; 3] {
    let idx = (y * level.width + x) * 3;
    [
        level.data[idx],
        level.data[idx + 1],
        level.data[idx + 2],
    ]
}

fn load_image(path: &PathBuf) -> anyhow::Result<SourceImage> {
    let image = ImageReader::open(path)?.decode()?;
    let image = image.to_rgb8();
    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity((width * height * 3) as usize);
    for pixel in image.pixels() {
        data.push(pixel[0] as f32 / 255.0);
        data.push(pixel[1] as f32 / 255.0);
        data.push(pixel[2] as f32 / 255.0);
    }
    Ok(SourceImage {
        width: width as usize,
        height: height as usize,
        data,
    })
}

fn create_bevy_image(source: &SourceImage, images: &mut Assets<Image>) -> Handle<Image> {
    let level = ImageLevel {
        width: source.width,
        height: source.height,
        data: source.data.clone(),
    };
    create_bevy_image_level(&level, images)
}

fn create_gpu_input_image(source: &SourceImage, images: &mut Assets<Image>) -> Handle<Image> {
    let size = Extent3d {
        width: source.width as u32,
        height: source.height as u32,
        depth_or_array_layers: 1,
    };
    let mut data = Vec::with_capacity(source.width * source.height * 8);
    for idx in 0..(source.width * source.height) {
        let base = idx * 3;
        push_f16(&mut data, source.data[base]);
        push_f16(&mut data, source.data[base + 1]);
        push_f16(&mut data, source.data[base + 2]);
        push_f16(&mut data, 1.0);
    }
    let mut image = Image::new(
        size,
        TextureDimension::D2,
        data,
        TextureFormat::Rgba16Float,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC
        | TextureUsages::TEXTURE_BINDING;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor::linear());
    images.add(image)
}

fn create_bevy_image_level(level: &ImageLevel, images: &mut Assets<Image>) -> Handle<Image> {
    let size = Extent3d {
        width: level.width as u32,
        height: level.height as u32,
        depth_or_array_layers: 1,
    };
    let mut data = Vec::with_capacity(level.width * level.height * 4);
    for idx in 0..(level.width * level.height) {
        let base = idx * 3;
        data.push((level.data[base].clamp(0.0, 1.0) * 255.0).round() as u8);
        data.push((level.data[base + 1].clamp(0.0, 1.0) * 255.0).round() as u8);
        data.push((level.data[base + 2].clamp(0.0, 1.0) * 255.0).round() as u8);
        data.push(255);
    }
    let mut image = Image::new_fill(
        size,
        TextureDimension::D2,
        &data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    images.add(image)
}

fn create_overlay_ring(
    size: usize,
    thickness: f32,
    color: [u8; 4],
    images: &mut Assets<Image>,
) -> Handle<Image> {
    let size = size.max(2);
    let extent = Extent3d {
        width: size as u32,
        height: size as u32,
        depth_or_array_layers: 1,
    };
    let mut data = vec![0u8; size * size * 4];
    let radius = size as f32 * 0.5;
    let inner = (radius - thickness.max(1.0)).max(0.0);
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 + 0.5 - radius;
            let dy = y as f32 + 0.5 - radius;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist >= inner && dist <= radius {
                let idx = (y * size + x) * 4;
                data[idx] = color[0];
                data[idx + 1] = color[1];
                data[idx + 2] = color[2];
                data[idx + 3] = color[3];
            }
        }
    }
    let mut image = Image::new_fill(
        extent,
        TextureDimension::D2,
        &data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage |= TextureUsages::TEXTURE_BINDING;
    images.add(image)
}

fn create_blank_image(size: usize, images: &mut Assets<Image>) -> Handle<Image> {
    let size = size.max(1);
    let extent = Extent3d {
        width: size as u32,
        height: size as u32,
        depth_or_array_layers: 1,
    };
    let data = vec![0u8; size * size * 4];
    let mut image = Image::new_fill(
        extent,
        TextureDimension::D2,
        &data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_DST
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::STORAGE_BINDING;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor::linear());
    images.add(image)
}

fn create_burn_image(size: usize, images: &mut Assets<Image>) -> Handle<Image> {
    let size = size.max(1);
    let extent = Extent3d {
        width: size as u32,
        height: size as u32,
        depth_or_array_layers: 1,
    };
    let mut image = Image::new_fill(
        extent,
        TextureDimension::D2,
        &[0u8; 16],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_DST
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::STORAGE_BINDING;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor::linear());
    images.add(image)
}

fn push_f16(data: &mut Vec<u8>, value: f32) {
    let bits = f16::from_f32(value).to_bits();
    data.extend_from_slice(&bits.to_le_bytes());
}

fn remap_noise_uniform(value: f64, min: f32, max: f32) -> f32 {
    let mut t = value as f32;
    t = t.clamp(-1.0, 1.0);
    t = t.signum() * t.abs().powf(0.6);
    let u = (0.5 + 0.5 * t).clamp(0.0, 1.0);
    min + (max - min) * u
}

pub(crate) fn radius_norm_from_sample(sample: &FoveationSample) -> f32 {
    foveation::sigma_from_unit(sample.radius_norm)
}

pub(crate) fn sigma_norm_from_settings(
    settings: &FoveationSettings,
    sample: &FoveationSample,
) -> f32 {
    let radius = radius_norm_from_sample(sample);
    let focus = settings.focus.clamp(0.0, 1.0);
    let sigma = radius * focus;
    sigma.clamp(FOVEA_PARAM_EPS, radius)
}

fn sigma_px_from_norm(sigma_norm: f32, source: &SourceImage) -> f32 {
    let min_dim = source.width.min(source.height).max(1) as f32;
    (sigma_norm * min_dim).max(FOVEA_PARAM_EPS)
}

pub(crate) fn radius_px_from_norm(radius_norm: f32, source: &SourceImage) -> f32 {
    let min_dim = source.width.min(source.height).max(1) as f32;
    (radius_norm * min_dim).max(FOVEA_PARAM_EPS)
}

pub(crate) fn map_pyramid_mode(mode: PyramidMode) -> VisionPyramidMode {
    match mode {
        PyramidMode::Gaussian => VisionPyramidMode::Stacked,
        PyramidMode::Laplacian => VisionPyramidMode::Laplacian,
    }
}

pub(crate) fn map_warp_mode(mode: FoveaWarpMode) -> VisionFoveaWarpMode {
    match mode {
        FoveaWarpMode::Warped => VisionFoveaWarpMode::Warped,
        FoveaWarpMode::Patched => VisionFoveaWarpMode::Patched,
    }
}

fn map_pyramid_mode_cpu(mode: PyramidMode) -> foveation::PyramidMode {
    match mode {
        PyramidMode::Gaussian => foveation::PyramidMode::Stacked,
        PyramidMode::Laplacian => foveation::PyramidMode::Laplacian,
    }
}

fn map_warp_mode_cpu(mode: FoveaWarpMode) -> foveation::FoveaWarpMode {
    match mode {
        FoveaWarpMode::Warped => foveation::FoveaWarpMode::Warped,
        FoveaWarpMode::Patched => foveation::FoveaWarpMode::Patched,
    }
}

pub(crate) fn make_minimal_vision_config(
    width: usize,
    height: usize,
    patch_size: usize,
) -> VisionDragonHatchlingConfig {
    let image_size = width.max(height).max(patch_size.max(1));
    let grid = (image_size / patch_size.max(1)).max(1);
    let mut config = VisionDragonHatchlingConfig::default();
    config.image_size = image_size;
    config.patch_size = patch_size.max(1);
    config.in_channels = 3;
    config.embed_dim = 32;
    config.steps = 1;
    config.n_head = 1;
    config.mlp_internal_dim_multiplier = 1;
    config.dropout = 0.0;
    config.projection_dim = 32;
    config.projection_hidden_dim = 32;
    config.use_cls_token = false;
    config.pos_encoding = SpatialPositionalEncodingKind::None;
    config.pos_max_height = grid;
    config.pos_max_width = grid;
    config.attention_mode = VisionAttentionMode::RowL1;
    config
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

fn burn_target_dims(source: &SourceImage, settings: &FoveationSettings) -> (usize, usize) {
    let width = source.width.max(1);
    let height = source.height.max(1);
    let (width, height) = scale_to_max_side(width, height, BURN_MAX_IMAGE_SIDE);
    let depth = settings.pyramid_depth.max(1);
    let mode = map_pyramid_mode(settings.mode);
    let (scaled_w, scaled_h) =
        scale_to_burn_limits(width, height, depth, mode, BURN_MAX_BUFFER_BYTES);
    (scaled_w, scaled_h)
}

fn scale_to_max_side(width: usize, height: usize, max_side: usize) -> (usize, usize) {
    let max_side = max_side.max(1);
    let max_dim = width.max(height);
    if max_dim <= max_side {
        return (width.max(1), height.max(1));
    }
    let scale = max_side as f64 / max_dim as f64;
    let scaled_w = ((width as f64) * scale).round().max(1.0) as usize;
    let scaled_h = ((height as f64) * scale).round().max(1.0) as usize;
    (scaled_w, scaled_h)
}

fn tensor_bytes_f32(batch: usize, channels: usize, height: usize, width: usize) -> u64 {
    let elems = (batch as u64)
        .saturating_mul(channels as u64)
        .saturating_mul(height as u64)
        .saturating_mul(width as u64);
    elems.saturating_mul(4)
}

// burn-tensor's default grid_sample expands buffers to N*C*H_out*W_out*W_in.
fn grid_sample_max_buffer_bytes(
    batch: usize,
    channels: usize,
    height_in: usize,
    width_in: usize,
    height_out: usize,
    width_out: usize,
) -> u64 {
    let n = batch as u64;
    let c = channels as u64;
    let h = height_in.max(height_out).max(1) as u64;
    let w_out = width_out.max(1) as u64;
    let w_in = width_in.max(1) as u64;
    n.saturating_mul(c)
        .saturating_mul(h)
        .saturating_mul(w_out)
        .saturating_mul(w_in)
        .saturating_mul(4)
}

fn pyramid_level_dims(width: usize, height: usize, depth: usize) -> Vec<(usize, usize)> {
    let mut levels = Vec::with_capacity(depth.max(1));
    let mut w = width.max(1);
    let mut h = height.max(1);
    for _ in 0..depth.max(1) {
        levels.push((w, h));
        if w < 2 || h < 2 {
            break;
        }
        let even_w = w - (w % 2);
        let even_h = h - (h % 2);
        if even_w == 0 || even_h == 0 {
            break;
        }
        w = (even_w / 2).max(1);
        h = (even_h / 2).max(1);
    }
    levels
}

#[derive(Clone, Copy, Debug, Default)]
struct BurnPyramidSize {
    max_tensor_bytes: u64,
    total_bytes: u64,
}

fn estimate_pyramid_bytes(
    batch: usize,
    channels: usize,
    width: usize,
    height: usize,
    depth: usize,
    mode: VisionPyramidMode,
) -> BurnPyramidSize {
    let levels = pyramid_level_dims(width, height, depth);
    if levels.is_empty() {
        return BurnPyramidSize::default();
    }
    let mut total = 0u64;
    let mut max_tensor = 0u64;
    for (idx, (w, h)) in levels.iter().copied().enumerate() {
        let level_bytes = tensor_bytes_f32(batch, channels, h, w);
        total = total.saturating_add(level_bytes);
        max_tensor = max_tensor.max(level_bytes);
        let grid_bytes = tensor_bytes_f32(batch, 2, h, w);
        max_tensor = max_tensor.max(grid_bytes);
        if matches!(mode, VisionPyramidMode::Laplacian) && idx + 1 < levels.len() {
            total = total.saturating_add(level_bytes);
        }
    }
    if matches!(mode, VisionPyramidMode::Laplacian) {
        let mut max_grid_sample = 0u64;
        for window in levels.windows(2) {
            let (w_out, h_out) = window[0];
            let (w_in, h_in) = window[1];
            let bytes =
                grid_sample_max_buffer_bytes(batch, channels, h_in, w_in, h_out, w_out);
            max_grid_sample = max_grid_sample.max(bytes);
        }
        max_tensor = max_tensor.max(max_grid_sample);
    }
    BurnPyramidSize {
        max_tensor_bytes: max_tensor,
        total_bytes: total,
    }
}

fn scale_to_burn_limits(
    width: usize,
    height: usize,
    depth: usize,
    mode: VisionPyramidMode,
    max_bytes: u64,
) -> (usize, usize) {
    let mut w = width.max(1);
    let mut h = height.max(1);
    loop {
        let estimate = estimate_pyramid_bytes(1, 3, w, h, depth, mode);
        let max_needed = estimate.max_tensor_bytes.max(estimate.total_bytes);
        if max_needed <= max_bytes {
            return (w, h);
        }
        let scale = (max_bytes as f64 / max_needed as f64).sqrt().min(1.0);
        let next_w = ((w as f64) * scale).floor().max(1.0) as usize;
        let next_h = ((h as f64) * scale).floor().max(1.0) as usize;
        if next_w == w && next_h == h {
            w = w.saturating_sub(1).max(1);
            h = h.saturating_sub(1).max(1);
        } else {
            w = next_w;
            h = next_h;
        }
    }
}

fn tensor_from_source_resized<B: Backend>(
    source: &SourceImage,
    target_width: usize,
    target_height: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let width = target_width.max(1);
    let height = target_height.max(1);
    if width == source.width.max(1) && height == source.height.max(1) {
        return tensor_from_source(source, device);
    }
    let mut data = vec![0.0f32; 3 * width * height];
    let src_w = source.width.max(1) as f32;
    let src_h = source.height.max(1) as f32;
    let max_x = (source.width.max(1) - 1) as f32;
    let max_y = (source.height.max(1) - 1) as f32;
    for y in 0..height {
        let fy = (y as f32 + 0.5) / height as f32;
        let sy = fy * src_h - 0.5;
        let y0 = sy.floor();
        let y1 = y0 + 1.0;
        let wy = sy - y0;
        let y0i = y0.clamp(0.0, max_y) as usize;
        let y1i = y1.clamp(0.0, max_y) as usize;
        for x in 0..width {
            let fx = (x as f32 + 0.5) / width as f32;
            let sx = fx * src_w - 0.5;
            let x0 = sx.floor();
            let x1 = x0 + 1.0;
            let wx = sx - x0;
            let x0i = x0.clamp(0.0, max_x) as usize;
            let x1i = x1.clamp(0.0, max_x) as usize;
            let c00 = source_pixel(source, x0i, y0i);
            let c10 = source_pixel(source, x1i, y0i);
            let c01 = source_pixel(source, x0i, y1i);
            let c11 = source_pixel(source, x1i, y1i);
            let c0 = [
                lerp_f32(c00[0], c10[0], wx),
                lerp_f32(c00[1], c10[1], wx),
                lerp_f32(c00[2], c10[2], wx),
            ];
            let c1 = [
                lerp_f32(c01[0], c11[0], wx),
                lerp_f32(c01[1], c11[1], wx),
                lerp_f32(c01[2], c11[2], wx),
            ];
            let c = [
                lerp_f32(c0[0], c1[0], wy),
                lerp_f32(c0[1], c1[1], wy),
                lerp_f32(c0[2], c1[2], wy),
            ];
            let dst = y * width + x;
            data[dst] = c[0];
            data[dst + height * width] = c[1];
            data[dst + 2 * height * width] = c[2];
        }
    }
    Tensor::<B, 4>::from_data(TensorData::new(data, [1, 3, height, width]), device)
}

fn source_pixel(source: &SourceImage, x: usize, y: usize) -> [f32; 3] {
    let idx = (y * source.width + x) * 3;
    [
        source.data[idx],
        source.data[idx + 1],
        source.data[idx + 2],
    ]
}

fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn patch_to_rgba<B: Backend>(patch: Tensor<B, 4>) -> Tensor<B, 3> {
    let device = patch.device();
    let [batch, channels, height, width] = patch.shape().dims::<4>();
    let patch = if batch > 1 {
        patch.slice_dim(0, 0..1)
    } else {
        patch
    };
    let patch = patch.swap_dims(1, 2).swap_dims(2, 3);
    let patch = if channels == 0 || height == 0 || width == 0 {
        Tensor::<B, 3>::zeros([height.max(1), width.max(1), 4], &device)
    } else {
        let patch = patch.squeeze_dim::<3>(0).clamp_min(0.0).clamp_max(1.0);
        let alpha = Tensor::<B, 3>::ones([height, width, 1], &device);
        Tensor::cat(vec![patch, alpha], 2)
    };
    patch
}
