#![recursion_limit = "256"]

use bevy::{
    asset::RenderAssetUsages,
    color::palettes::css::GOLD,
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    image::ImagePlugin,
    prelude::*,
    render::render_resource::*,
    tasks::{block_on, Task},
};
#[cfg(target_arch = "wasm32")]
use bevy::tasks::futures_lite::future;
#[cfg(target_arch = "wasm32")]
use bevy::tasks::AsyncComputeTaskPool;
use bevy_burn::{BevyBurnBridgePlugin, BevyBurnHandle, BindingDirection, BurnDevice, TransferKind};
use burn_core::tensor::{backend::Backend, ElementConversion, Int, Tensor};
use burn_wgpu::{Wgpu, WgpuDevice as BurnWgpuDevice};

type BurnBackend = Wgpu<f32, i32>;

// TODO: convert to resource and add world inspector
const CFL: f32 = 0.3;
const MAX_DT: f32 = 0.01;
const MAX_VEL: f32 = 1000.0;
const PRESSURE_ITERS: usize = 8;
const SIZE: u32 = 1024;
const SOURCE_MAG: f32 = 10.5;
const SOURCE_POS: (usize, usize) = (SIZE as usize / 2, SIZE as usize / 2);
const SOURCE_SIGMA: f32 = 3.0;
const VISCOSITY: f32 = 1e-3;
const DIFF_ITERS: usize = 2;
const OMEGA: f32 = 1.6;
const PRESSURE_RES_TOL: f32 = 1e-3;
const VIZ_TAU_RISE: f32 = 0.15;
const VIZ_TAU_FALL: f32 = 0.60;
const MAX_TIME_ACCUM: f32 = 0.25;
const FIXED_DT: f32 = 1.0 / 240.0;
const MAX_SUBSTEPS_PER_FRAME: usize = 8;
const CFL_RECOMP_INTERVAL: usize = 4;

fn velocity_to_rgba<B: Backend>(v: &Tensor<B, 4>, scale: f32, device: &B::Device) -> Tensor<B, 3> {
    let vx = v
        .clone()
        .slice_dim(0, 0..1)
        .slice_dim(1, 0..1)
        .squeeze_dims::<2>(&[0, 1]);

    let vy = v
        .clone()
        .slice_dim(0, 0..1)
        .slice_dim(1, 1..2)
        .squeeze_dims::<2>(&[0, 1]);

    let mag = (vx.powf_scalar(2.) + vy.powf_scalar(2.)).sqrt();
    let x = mag.div_scalar(scale.max(1e-8)).clamp(0.0, 1.0);

    // Simple Jet colormap approximation
    // r = clamp(1.5 - |4x - 3|, 0, 1)
    // g = clamp(1.5 - |4x - 2|, 0, 1)
    // b = clamp(1.5 - |4x - 1|, 0, 1)
    let four_x = x.clone().mul_scalar(4.0);
    let r = (four_x.clone().sub_scalar(3.0))
        .abs()
        .mul_scalar(-1.0)
        .add_scalar(1.5)
        .clamp(0.0, 1.0);
    let g = (four_x.clone().sub_scalar(2.0))
        .abs()
        .mul_scalar(-1.0)
        .add_scalar(1.5)
        .clamp(0.0, 1.0);
    let b = (four_x.sub_scalar(1.0))
        .abs()
        .mul_scalar(-1.0)
        .add_scalar(1.5)
        .clamp(0.0, 1.0);

    let r = r.unsqueeze_dim::<3>(2);
    let g = g.unsqueeze_dim::<3>(2);
    let b = b.unsqueeze_dim::<3>(2);
    let a = Tensor::<B, 3>::ones(r.dims(), device);

    Tensor::cat(vec![r, g, b, a], 2)
}

fn velocity_point_source<B: Backend>(grid_like: &Tensor<B, 4>, device: &B::Device) -> Tensor<B, 4> {
    let [b, _, h, w] = grid_like.dims();

    let xs = Tensor::<B, 1, Int>::arange(0..w as i64, device)
        .float()
        .reshape([1, 1, 1, w]);
    let ys = Tensor::<B, 1, Int>::arange(0..h as i64, device)
        .float()
        .reshape([1, 1, h, 1]);

    let dx2 = (xs.sub_scalar(SOURCE_POS.0 as f32)).powf_scalar(2.);
    let dy2 = (ys.sub_scalar(SOURCE_POS.1 as f32)).powf_scalar(2.);

    let gaussian = (dx2 + dy2)
        .div_scalar(-2.0 * SOURCE_SIGMA * SOURCE_SIGMA)
        .exp()
        .mul_scalar(SOURCE_MAG);

    let gaussian = gaussian.repeat(&[b, 1, 1, 1]);

    let zero = Tensor::<B, 4>::zeros([b, 1, h, w], device);
    Tensor::cat(vec![zero, gaussian], 1)
}

/// * `v`   - velocity field `[b, 2, h, w]`
/// * `nu`  - kinematic viscosity
/// * `dt`  - time step
async fn navier_stokes<B: Backend>(
    v: Tensor<B, 4>,
    nu: f32,
    dt: f32,
    xs: &Tensor<B, 4>,
    ys: &Tensor<B, 4>,
    mask_red: &Tensor<B, 4>,
    mask_black: &Tensor<B, 4>,
    device: &B::Device,
) -> Tensor<B, 4> {
    if dt <= 0.0 {
        return v;
    }

    let v = v.clamp(-MAX_VEL, MAX_VEL);

    let v1 = advect::<B>(v, dt, xs, ys);
    let v2 = diffuse::<B>(v1, nu, dt);

    let u = v2.clone().slice_dim(1, 0..1);
    let wv = v2.slice_dim(1, 1..2);

    let div = divergence(&u, &wv);
    let mut p = Tensor::<B, 4>::zeros(div.dims(), device);
    let rhs = div.clone() / dt;

    let scaled = ((PRESSURE_ITERS as f32) * (dt / MAX_DT).clamp(0.5, 1.0)).ceil() as usize;
    let iters = scaled.max(2).min(PRESSURE_ITERS);
    let tol = PRESSURE_RES_TOL;
    for _ in 0..iters {
        let sum_nb = neigh_sum(&p) * mask_red.clone();
        let p_new = (sum_nb - rhs.clone()) * 0.25;
        p = p.clone() + (p_new - p) * mask_red.clone() * OMEGA;

        let sum_nb = neigh_sum(&p) * mask_black.clone();
        let p_new = (sum_nb - rhs.clone()) * 0.25;
        p = p.clone() + (p_new - p) * mask_black.clone() * OMEGA;

        // residual r = Laplace(p) - rhs = (sum_nb - 4p) - rhs
        let lap = neigh_sum(&p) - p.clone() * 4.0;
        let r = lap - rhs.clone();
        let r_max = r
            .abs()
            .max()
            .into_scalar_async()
            .await
            .elem::<f32>();
        if r_max <= tol {
            break;
        }
    }

    let (dp_dx, dp_dy) = grad_p(&p);
    let u_corr = u - dp_dx * dt;
    let w_corr = wv - dp_dy * dt;
    Tensor::cat(vec![u_corr, w_corr], 1).clamp(-MAX_VEL, MAX_VEL)
}

// first‑order finite differences forming an adjoint pair
fn dx_fwd<B: Backend>(f: &Tensor<B, 4>) -> Tensor<B, 4> {
    // f(i+1) - f(i)
    f.clone().roll(&[-1], &[3]) - f.clone()
}
fn dy_fwd<B: Backend>(f: &Tensor<B, 4>) -> Tensor<B, 4> {
    // f(j+1) - f(j)
    f.clone().roll(&[-1], &[2]) - f.clone()
}
fn dx_bwd<B: Backend>(f: &Tensor<B, 4>) -> Tensor<B, 4> {
    // f(i) - f(i-1)
    f.clone() - f.clone().roll(&[1], &[3])
}
fn dy_bwd<B: Backend>(f: &Tensor<B, 4>) -> Tensor<B, 4> {
    // f(j) - f(j-1)
    f.clone() - f.clone().roll(&[1], &[2])
}

fn divergence<B: Backend>(u: &Tensor<B, 4>, v: &Tensor<B, 4>) -> Tensor<B, 4> {
    dx_bwd(u) + dy_bwd(v)
}

fn grad_p<B: Backend>(p: &Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
    (dx_fwd(p), dy_fwd(p))
}

fn neigh_sum<B: Backend>(v: &Tensor<B, 4>) -> Tensor<B, 4> {
    v.clone().roll(&[1], &[2])
        + v.clone().roll(&[-1], &[2])
        + v.clone().roll(&[1], &[3])
        + v.clone().roll(&[-1], &[3])
}

fn colour_mask<B: Backend>(h: usize, w: usize, red: bool, device: &B::Device) -> Tensor<B, 4> {
    let xs = Tensor::<B, 1, Int>::arange(0..w as i64, device)
        .float()
        .reshape([1, 1, 1, w]);
    let ys = Tensor::<B, 1, Int>::arange(0..h as i64, device)
        .float()
        .reshape([1, 1, h, 1]);

    let parity = (xs + ys).remainder_scalar(2.0f32);

    parity.equal_elem(if red { 0.0f32 } else { 1.0f32 }).float()
}

fn advect<B: Backend>(
    v: Tensor<B, 4>,
    dt: f32,
    xs: &Tensor<B, 4>,
    ys: &Tensor<B, 4>,
) -> Tensor<B, 4> {
    // Bilinear semi-lagrangian step with donor tracking
    let [_, _, h, w] = v.dims();

    let u = v.clone().slice_dim(1, 0..1);
    let wv = v.clone().slice_dim(1, 1..2);

    let x_back = xs.clone() - u * dt;
    let y_back = ys.clone() - wv * dt;

    // periodic wrap into [0, size)
    let x_back = x_back.clone()
        - (x_back.clone().div_scalar(w as f32))
            .floor()
            .mul_scalar(w as f32);
    let y_back = y_back.clone()
        - (y_back.clone().div_scalar(h as f32))
            .floor()
            .mul_scalar(h as f32);

    let x0 = x_back.clone().floor();
    let y0 = y_back.clone().floor();
    let x1 = (x0.clone() + 1.0)
        - ((x0.clone() + 1.0).div_scalar(w as f32))
            .floor()
            .mul_scalar(w as f32);
    let y1 = (y0.clone() + 1.0)
        - ((y0.clone() + 1.0).div_scalar(h as f32))
            .floor()
            .mul_scalar(h as f32);

    let wx = x_back - x0.clone();
    let wy = y_back - y0.clone();

    let gather = |xx: Tensor<B, 4>, yy: Tensor<B, 4>| {
        let mut xi = xx.int();
        let mut yi = yy.int();
        let v = v.clone();
        let c = v.dims()[1];
        xi = xi.repeat(&[1, c, 1, 1]);
        yi = yi.repeat(&[1, c, 1, 1]);
        v.gather(3, xi).gather(2, yi)
    };

    let v00 = gather(x0.clone(), y0.clone());
    let v10 = gather(x1.clone(), y0);
    let v01 = gather(x0, y1.clone());
    let v11 = gather(x1, y1);

    let wxr = wx.repeat(&[1, v.dims()[1], 1, 1]);
    let wyr = wy.repeat(&[1, v.dims()[1], 1, 1]);
    let v0 = v00.clone() * (1.0 - wxr.clone()) + v10.clone() * wxr.clone();
    let v1x = v01.clone() * (1.0 - wxr.clone()) + v11.clone() * wxr;
    let v1: Tensor<B, 4> = v0 * (1.0 - wyr.clone()) + v1x * wyr;

    // Forward step to estimate error (BFECC)
    let v_back = advect_linear::<B>(v1.clone(), -dt, xs, ys);
    let mut v2 = v1 + (v - v_back) * 0.5;

    // Monotonic limiter: clamp to donor min/max to avoid creating new extrema
    let smin = tmin(
        tmin(v00.clone(), v10.clone()),
        tmin(v01.clone(), v11.clone()),
    );
    let smax = tmax(tmax(v00, v10), tmax(v01, v11));
    v2 = tmax(tmin(v2, smax.clone()), smin);

    // Final safety clamp
    v2.clamp(-MAX_VEL, MAX_VEL)
}

fn tmin<B: Backend>(a: Tensor<B, 4>, b: Tensor<B, 4>) -> Tensor<B, 4> {
    let diff = a.clone() - b.clone();
    (a + b - diff.abs()) * 0.5
}

fn tmax<B: Backend>(a: Tensor<B, 4>, b: Tensor<B, 4>) -> Tensor<B, 4> {
    let diff = a.clone() - b.clone();
    (a + b + diff.abs()) * 0.5
}

fn advect_linear<B: Backend>(
    v: Tensor<B, 4>,
    dt: f32,
    xs: &Tensor<B, 4>,
    ys: &Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [_, _, h, w] = v.dims();

    let u = v.clone().slice_dim(1, 0..1);
    let wv = v.clone().slice_dim(1, 1..2);

    let x_back = xs.clone() - u * dt;
    let y_back = ys.clone() - wv * dt;

    // periodic wrap into [0, size) using floor-based modulo.
    // using `remainder_scalar` biases samples toward the left/top edges.
    let x_back = x_back.clone()
        - (x_back.clone().div_scalar(w as f32))
            .floor()
            .mul_scalar(w as f32);
    let y_back = y_back.clone()
        - (y_back.clone().div_scalar(h as f32))
            .floor()
            .mul_scalar(h as f32);

    let x0 = x_back.clone().floor();
    let y0 = y_back.clone().floor();
    let x1 = (x0.clone() + 1.0)
        - ((x0.clone() + 1.0).div_scalar(w as f32))
            .floor()
            .mul_scalar(w as f32);
    let y1 = (y0.clone() + 1.0)
        - ((y0.clone() + 1.0).div_scalar(h as f32))
            .floor()
            .mul_scalar(h as f32);

    let wx = x_back - x0.clone();
    let wy = y_back - y0.clone();

    let gather = |xx: Tensor<B, 4>, yy: Tensor<B, 4>| {
        let mut xi = xx.int();
        let mut yi = yy.int();
        let v = v.clone();
        let c = v.dims()[1];
        xi = xi.repeat(&[1, c, 1, 1]);
        yi = yi.repeat(&[1, c, 1, 1]);
        v.gather(3, xi).gather(2, yi)
    };

    let v00 = gather(x0.clone(), y0.clone());
    let v10 = gather(x1.clone(), y0);
    let v01 = gather(x0, y1.clone());
    let v11 = gather(x1, y1);

    let wx = wx.repeat(&[1, v.dims()[1], 1, 1]);
    let wy = wy.repeat(&[1, v.dims()[1], 1, 1]);
    let v0 = v00 * (1.0 - wx.clone()) + v10 * wx.clone();
    let v1 = v01 * (1.0 - wx.clone()) + v11 * wx;
    v0 * (1.0 - wy.clone()) + v1 * wy
}

fn diffuse<B: Backend>(v: Tensor<B, 4>, nu: f32, dt: f32) -> Tensor<B, 4> {
    if nu == 0.0 {
        return v;
    }
    let a = nu * dt;
    let mut x = v.clone();
    for _ in 0..DIFF_ITERS {
        let s = neigh_sum(&x);
        x = (v.clone() + s * a) / (1.0 + 4.0 * a);
    }
    x
}

#[derive(Default, Resource)]
struct NavierStokesState<B: Backend> {
    data: Option<NavierStokesData<B>>,
}

struct NavierStokesData<B: Backend> {
    velocity: Tensor<B, 4>,
    xs: Tensor<B, 4>,
    ys: Tensor<B, 4>,
    mask_red: Tensor<B, 4>,
    mask_black: Tensor<B, 4>,
    viz_scale: f32,
    time_accum: f32,
}

impl<B: Backend> NavierStokesData<B> {
    fn new(device: &B::Device) -> Self {
        let velocity = Tensor::<B, 4>::zeros([1, 2, SIZE as usize, SIZE as usize], device);

        let xs = Tensor::<B, 1, Int>::arange(0..SIZE as i64, device)
            .float()
            .reshape([1, 1, 1, SIZE as usize]);
        let ys = Tensor::<B, 1, Int>::arange(0..SIZE as i64, device)
            .float()
            .reshape([1, 1, SIZE as usize, 1]);

        let mask_red =
            colour_mask::<B>(SIZE as usize, SIZE as usize, true, device).repeat(&[1, 1, 1, 1]);
        let mask_black =
            colour_mask::<B>(SIZE as usize, SIZE as usize, false, device).repeat(&[1, 1, 1, 1]);

        NavierStokesData {
            velocity,
            xs,
            ys,
            mask_red,
            mask_black,
            viz_scale: 1.0,
            time_accum: 0.0,
        }
    }
}

impl<B: Backend> NavierStokesState<B> {
    fn ensure_initialized(&mut self, device: &B::Device) {
        if self.data.is_none() {
            self.data = Some(NavierStokesData::new(device));
        }
    }
}

struct NavierStokesJobInput<B: Backend> {
    velocity: Tensor<B, 4>,
    xs: Tensor<B, 4>,
    ys: Tensor<B, 4>,
    mask_red: Tensor<B, 4>,
    mask_black: Tensor<B, 4>,
    viz_scale: f32,
    time_accum: f32,
    frame_dt: f32,
    device: B::Device,
}

struct NavierStokesJobResult<B: Backend> {
    velocity: Tensor<B, 4>,
    viz_scale: f32,
    time_accum: f32,
    rgba: Tensor<B, 3>,
}

async fn run_navier_stokes_job<B: Backend>(
    input: NavierStokesJobInput<B>,
) -> NavierStokesJobResult<B> {
    let device = &input.device;
    let mut velocity = input.velocity;
    let mut time_accum = (input.time_accum + input.frame_dt).min(MAX_TIME_ACCUM);
    let mut substeps: usize = 0;

    let uu = velocity.clone().slice_dim(1, 0..1);
    let vv = velocity.clone().slice_dim(1, 1..2);
    let mut vmax = (uu.clone().powf_scalar(2.) + vv.clone().powf_scalar(2.))
        .sqrt()
        .max()
        .into_scalar_async()
        .await
        .elem::<f32>()
        .max(1e-3);

    let mut safe_dt = (CFL / vmax).min(MAX_DT);
    let mut dt = FIXED_DT.min(safe_dt).max(1e-6);

    while time_accum >= dt && substeps < MAX_SUBSTEPS_PER_FRAME {
        velocity = navier_stokes(
            velocity.clone(),
            VISCOSITY,
            dt,
            &input.xs,
            &input.ys,
            &input.mask_red,
            &input.mask_black,
            device,
        )
        .await;

        time_accum -= dt;
        substeps += 1;

        if substeps % CFL_RECOMP_INTERVAL == 0 {
            let u2 = velocity.clone().slice_dim(1, 0..1);
            let v2 = velocity.clone().slice_dim(1, 1..2);
            vmax = (u2.clone().powf_scalar(2.) + v2.clone().powf_scalar(2.))
                .sqrt()
                .max()
                .into_scalar_async()
                .await
                .elem::<f32>()
                .max(1e-3);
            safe_dt = (CFL / vmax).min(MAX_DT);
            dt = FIXED_DT.min(safe_dt).max(1e-6);
        }
    }

    let u = velocity.clone().slice_dim(1, 0..1);
    let v = velocity.clone().slice_dim(1, 1..2);
    let mag_max = (u.clone().powf_scalar(2.) + v.clone().powf_scalar(2.))
        .sqrt()
        .max()
        .into_scalar_async()
        .await
        .elem::<f32>();

    let eps = 1e-3f32;
    let current = input.viz_scale.max(eps);
    let target = mag_max.max(eps);
    let alpha_up = 1.0 - (-input.frame_dt / VIZ_TAU_RISE).exp();
    let alpha_down = 1.0 - (-input.frame_dt / VIZ_TAU_FALL).exp();
    let viz_scale = if target > current {
        current * (1.0 - alpha_up) + target * alpha_up
    } else {
        current * (1.0 - alpha_down) + target * alpha_down
    };

    let rgba = velocity_to_rgba(&velocity, viz_scale, device);

    NavierStokesJobResult {
        velocity,
        viz_scale,
        time_accum,
        rgba,
    }
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut ns: ResMut<NavierStokesState<BurnBackend>>,
    burn_device: Res<BurnDevice>,
) {
    let Some(dev) = burn_device.device() else {
        return;
    };

    ns.ensure_initialized(dev);
    let Some(ns) = ns.data.as_mut() else {
        return;
    };

    ns.velocity = ns.velocity.clone() + velocity_point_source::<BurnBackend>(&ns.velocity, dev);
    let rgba = velocity_to_rgba(&ns.velocity, ns.viz_scale, dev);

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
    let bevy_image = images.add(img);

    commands.spawn(BevyBurnHandle {
        bevy_image: bevy_image.clone(),
        tensor: rgba,
        upload: true,
        direction: BindingDirection::BurnToBevy,
        xfer: TransferKind::Gpu,
    });

    commands.spawn(Camera2d::default());

    commands.spawn((
        Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            ..default()
        },
        ImageNode::new(bevy_image),
    ));
}

fn fps_display_setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands
        .spawn((
            Text("fps: ".to_string()),
            TextFont {
                font: asset_server.load("fonts/Caveat-Bold.ttf"),
                font_size: 60.0,
                ..Default::default()
            },
            TextColor(Color::WHITE),
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(5.0),
                left: Val::Px(15.0),
                ..default()
            },
            ZIndex(2),
        ))
        .with_child((
            FpsText,
            TextColor(Color::Srgba(GOLD)),
            TextFont {
                font: asset_server.load("fonts/Caveat-Bold.ttf"),
                font_size: 60.0,
                ..Default::default()
            },
            TextSpan::default(),
        ));
}

#[derive(Component)]
struct FpsText;

fn fps_update_system(
    diagnostics: Res<DiagnosticsStore>,
    mut query: Query<&mut TextSpan, With<FpsText>>,
) {
    for mut text in &mut query {
        if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
            if let Some(value) = fps.smoothed() {
                **text = format!("{value:.2}");
            }
        }
    }
}

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(ImagePlugin::default_nearest()),
            FrameTimeDiagnosticsPlugin::default(),
            BevyBurnBridgePlugin::<BurnBackend>::default(),
        ))
        .init_resource::<NavierStokesState<BurnBackend>>()
        .add_systems(Startup, (fps_display_setup, setup))
        .add_systems(Update, (fps_update_system, update_tensor))
        .run();
}
fn update_tensor(
    time: Res<Time>,
    mut handles: Query<&mut BevyBurnHandle<BurnBackend>>,
    mut ns: ResMut<NavierStokesState<BurnBackend>>,
    burn_device: Res<BurnDevice>,
    mut pending: Local<Option<Task<NavierStokesJobResult<BurnBackend>>>>,
) {
    let Some(device) = burn_device.device() else {
        return;
    };
    let mut handle = match handles.iter_mut().next() {
        Some(handle) => handle,
        None => return,
    };

    ns.ensure_initialized(device);
    let Some(ns) = ns.data.as_mut() else {
        return;
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        pending.take();
        let input = build_job_input(ns, device, time.delta_secs());
        let result = block_on(run_navier_stokes_job(input));
        apply_job_result(result, ns, &mut handle);
        return;
    }

    #[cfg(target_arch = "wasm32")]
    {
        if let Some(task) = pending.as_mut() {
            if let Some(result) = block_on(future::poll_once(task)) {
                apply_job_result(result, ns, &mut handle);
                *pending = None;
            }
        }

        if pending.is_some() {
            return;
        }

        let input = build_job_input(ns, device, time.delta_secs());
        let pool = AsyncComputeTaskPool::get();
        *pending = Some(pool.spawn(run_navier_stokes_job(input)));
    }
}

fn build_job_input(
    ns: &NavierStokesData<BurnBackend>,
    device: &BurnWgpuDevice,
    frame_dt: f32,
) -> NavierStokesJobInput<BurnBackend> {
    NavierStokesJobInput {
        velocity: ns.velocity.clone(),
        xs: ns.xs.clone(),
        ys: ns.ys.clone(),
        mask_red: ns.mask_red.clone(),
        mask_black: ns.mask_black.clone(),
        viz_scale: ns.viz_scale,
        time_accum: ns.time_accum,
        frame_dt,
        device: device.clone(),
    }
}

fn apply_job_result(
    result: NavierStokesJobResult<BurnBackend>,
    ns: &mut NavierStokesData<BurnBackend>,
    handle: &mut BevyBurnHandle<BurnBackend>,
) {
    ns.velocity = result.velocity;
    ns.viz_scale = result.viz_scale;
    ns.time_accum = result.time_accum;
    handle.tensor = result.rgba;
    handle.upload = true;
}
