use burn_dragon_vision::{
    FoveaWarpMode, PyramidMode, build_pyramid_cache, load_cpu_artifact_source,
    render_foveated_patch_with_radius, rgb_f32_to_rgba_image,
};
use image::imageops::FilterType;
use image::{DynamicImage, ImageBuffer, Rgba, RgbaImage};
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

const DEFAULT_SOURCE: &str = "docs/assets/community.png";
const DEFAULT_OUTPUT_DIR: &str = "artifacts/fovea_methods";
const MAX_SOURCE_SIDE: u32 = 1600;
const PATCH_SIZE: usize = 512;
const PANEL_SIZE: u32 = 512;
const GUTTER: u32 = 24;
const DEPTH: usize = 5;

#[derive(Clone, Copy)]
struct CaseSpec {
    name: &'static str,
    mean: [f32; 2],
    sigma: f32,
    radius: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let source_path = args
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_SOURCE));
    let output_dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_OUTPUT_DIR));

    fs::create_dir_all(&output_dir)?;

    let source = load_cpu_artifact_source(&source_path, MAX_SOURCE_SIDE)?;
    let cache = build_pyramid_cache(source.cpu_level.clone(), DEPTH, PyramidMode::Laplacian);
    let cases = [
        CaseSpec {
            name: "center_medium",
            mean: [0.50, 0.50],
            sigma: 0.08,
            radius: 0.28,
        },
        CaseSpec {
            name: "upper_right_tight",
            mean: [0.72, 0.33],
            sigma: 0.05,
            radius: 0.16,
        },
        CaseSpec {
            name: "lower_left_broad",
            mean: [0.23, 0.77],
            sigma: 0.11,
            radius: 0.36,
        },
    ];
    let warp_modes = [
        ("warped", FoveaWarpMode::Warped),
        ("conformal", FoveaWarpMode::Conformal),
        ("patched", FoveaWarpMode::Patched),
    ];

    let mut manifest = String::new();
    writeln!(&mut manifest, "source={}", source_path.display())?;
    writeln!(
        &mut manifest,
        "resized_source={}x{}",
        source.width(),
        source.height()
    )?;
    writeln!(&mut manifest, "pyramid_mode=laplacian")?;
    writeln!(&mut manifest, "pyramid_depth={DEPTH}")?;
    writeln!(&mut manifest, "patch_size={PATCH_SIZE}")?;
    writeln!(
        &mut manifest,
        "comparison_sheet_order=source_overlay,warped,conformal,patched"
    )?;

    for case in cases {
        let case_dir = output_dir.join(case.name);
        fs::create_dir_all(&case_dir)?;
        writeln!(
            &mut manifest,
            "\n[{}]\nmean_x={:.3}\nmean_y={:.3}\nsigma={:.3}\nradius={:.3}",
            case.name, case.mean[0], case.mean[1], case.sigma, case.radius
        )?;

        let overlay = draw_source_overlay(&source.rgba, case);
        let overlay_path = case_dir.join("source_overlay.png");
        overlay.save(&overlay_path)?;

        let mut rendered = Vec::new();
        for (label, warp_mode) in warp_modes {
            let patch = render_foveated_patch_with_radius(
                &cache,
                case.mean,
                case.sigma,
                case.radius,
                PATCH_SIZE,
                warp_mode,
            );
            let image = rgb_f32_to_rgba_image(&patch, PATCH_SIZE as u32, PATCH_SIZE as u32);
            let path = case_dir.join(format!("{label}.png"));
            image.save(&path)?;
            writeln!(&mut manifest, "{label}={}", path.display())?;
            rendered.push((label, image));
        }

        let comparison = build_comparison_sheet(&overlay, &rendered);
        let comparison_path = case_dir.join("comparison.png");
        comparison.save(&comparison_path)?;
        writeln!(&mut manifest, "comparison={}", comparison_path.display())?;
    }

    fs::write(output_dir.join("manifest.txt"), manifest)?;
    println!("{}", output_dir.display());
    Ok(())
}

fn draw_source_overlay(source: &RgbaImage, case: CaseSpec) -> RgbaImage {
    let mut overlay = source.clone();
    let (width, height) = overlay.dimensions();
    let center_x = case.mean[0] * width as f32;
    let center_y = case.mean[1] * height as f32;
    let radius_px = case.radius * width.min(height) as f32;
    let sigma_px = case.sigma * width.min(height) as f32;

    draw_cross(
        &mut overlay,
        center_x.round() as i32,
        center_y.round() as i32,
        14,
        [255, 64, 64, 255],
    );
    draw_ring(
        &mut overlay,
        center_x,
        center_y,
        radius_px,
        2.5,
        [255, 64, 64, 255],
    );
    draw_ring(
        &mut overlay,
        center_x,
        center_y,
        sigma_px,
        2.0,
        [64, 200, 255, 255],
    );
    overlay
}

fn build_comparison_sheet(overlay: &RgbaImage, rendered: &[(&str, RgbaImage)]) -> RgbaImage {
    let columns = 1 + rendered.len() as u32;
    let width = columns * PANEL_SIZE + (columns + 1) * GUTTER;
    let height = PANEL_SIZE + 2 * GUTTER;
    let mut sheet = ImageBuffer::from_pixel(width, height, Rgba([245, 245, 245, 255]));

    let overlay_panel = fit_to_panel(overlay, PANEL_SIZE);
    blit(
        &mut sheet,
        &overlay_panel,
        GUTTER + (PANEL_SIZE - overlay_panel.width()) / 2,
        GUTTER + (PANEL_SIZE - overlay_panel.height()) / 2,
    );

    for (idx, (_label, image)) in rendered.iter().enumerate() {
        let x0 = GUTTER + (idx as u32 + 1) * (PANEL_SIZE + GUTTER);
        let panel = fit_to_panel(image, PANEL_SIZE);
        blit(
            &mut sheet,
            &panel,
            x0 + (PANEL_SIZE - panel.width()) / 2,
            GUTTER + (PANEL_SIZE - panel.height()) / 2,
        );
    }

    draw_panel_frame(&mut sheet, 0);
    for idx in 0..rendered.len() {
        draw_panel_frame(&mut sheet, idx as u32 + 1);
    }
    sheet
}

fn fit_to_panel(image: &RgbaImage, panel_size: u32) -> RgbaImage {
    let (width, height) = image.dimensions();
    if width == panel_size && height == panel_size {
        return image.clone();
    }
    DynamicImage::ImageRgba8(image.clone())
        .resize(panel_size, panel_size, FilterType::Lanczos3)
        .to_rgba8()
}

fn draw_panel_frame(image: &mut RgbaImage, panel_idx: u32) {
    let x0 = GUTTER + panel_idx * (PANEL_SIZE + GUTTER);
    let y0 = GUTTER;
    let x1 = x0 + PANEL_SIZE - 1;
    let y1 = y0 + PANEL_SIZE - 1;
    let color = Rgba([30, 30, 30, 255]);
    for x in x0..=x1 {
        image.put_pixel(x, y0, color);
        image.put_pixel(x, y1, color);
    }
    for y in y0..=y1 {
        image.put_pixel(x0, y, color);
        image.put_pixel(x1, y, color);
    }
}

fn blit(dst: &mut RgbaImage, src: &RgbaImage, x0: u32, y0: u32) {
    for y in 0..src.height() {
        for x in 0..src.width() {
            dst.put_pixel(x0 + x, y0 + y, *src.get_pixel(x, y));
        }
    }
}

fn draw_cross(image: &mut RgbaImage, cx: i32, cy: i32, size: i32, color: [u8; 4]) {
    for delta in -size..=size {
        put_pixel_safe(image, cx + delta, cy, color);
        put_pixel_safe(image, cx, cy + delta, color);
    }
}

fn draw_ring(image: &mut RgbaImage, cx: f32, cy: f32, radius: f32, thickness: f32, color: [u8; 4]) {
    let min_x = (cx - radius - thickness - 1.0).floor() as i32;
    let max_x = (cx + radius + thickness + 1.0).ceil() as i32;
    let min_y = (cy - radius - thickness - 1.0).floor() as i32;
    let max_y = (cy + radius + thickness + 1.0).ceil() as i32;
    let inner = (radius - thickness).max(0.0);
    let outer = radius + thickness;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist >= inner && dist <= outer {
                put_pixel_safe(image, x, y, color);
            }
        }
    }
}

fn put_pixel_safe(image: &mut RgbaImage, x: i32, y: i32, color: [u8; 4]) {
    if x < 0 || y < 0 {
        return;
    }
    let x = x as u32;
    let y = y as u32;
    if x >= image.width() || y >= image.height() {
        return;
    }
    image.put_pixel(x, y, Rgba(color));
}
