#![cfg_attr(not(feature = "cli"), allow(dead_code))]

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, fmt};

use anyhow::{Context, Result, anyhow};
use image::RgbImage;

use crate::config::VisionArtifactOutputMode;

pub const ARTIFACT_DEFAULT_FPS: u32 = 4;

#[derive(Clone, Debug)]
pub struct ArtifactFrame {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

impl ArtifactFrame {
    pub fn upscale_nearest(&self, scale: usize) -> Self {
        let scale = scale.max(1);
        if scale == 1 || self.width == 0 || self.height == 0 {
            return self.clone();
        }
        let width = self.width * scale;
        let height = self.height * scale;
        let mut rgb = vec![0u8; width * height * 3];
        for y in 0..height {
            let src_y = y / scale;
            for x in 0..width {
                let src_x = x / scale;
                let src = (src_y * self.width + src_x) * 3;
                let dst = (y * width + x) * 3;
                rgb[dst..dst + 3].copy_from_slice(&self.rgb[src..src + 3]);
            }
        }
        Self { width, height, rgb }
    }
}

#[derive(Clone, Debug)]
pub struct ArtifactWriteOutcome {
    pub saved: usize,
    pub mode: VisionArtifactOutputMode,
    pub path: PathBuf,
}

#[allow(clippy::too_many_arguments)]
pub fn collect_frames(
    data: &[f32],
    batch: usize,
    frames: usize,
    channels: usize,
    height: usize,
    width: usize,
    batch_idx: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> Vec<ArtifactFrame> {
    if batch_idx >= batch || frames == 0 || channels < 3 || height == 0 || width == 0 {
        return Vec::new();
    }
    let channel_stride = height * width;
    let frame_stride = channels * channel_stride;
    let mut out = Vec::with_capacity(frames);
    for frame_idx in 0..frames {
        let frame_base = (batch_idx * frames + frame_idx) * frame_stride;
        let mut rgb = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = frame_base + y * width + x;
                let r = denormalize(data[idx], 0, mean, std);
                let g = denormalize(data[idx + channel_stride], 1, mean, std);
                let b = denormalize(data[idx + 2 * channel_stride], 2, mean, std);
                let out_idx = (y * width + x) * 3;
                rgb[out_idx] = r;
                rgb[out_idx + 1] = g;
                rgb[out_idx + 2] = b;
            }
        }
        out.push(ArtifactFrame { width, height, rgb });
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub fn write_video(
    output_dir: &Path,
    output_mode: VisionArtifactOutputMode,
    overwrite: bool,
    epoch: usize,
    iteration: usize,
    sample_idx: usize,
    frames: &[ArtifactFrame],
    fps: u32,
    ffmpeg_path: Option<&Path>,
) -> Result<ArtifactWriteOutcome> {
    if frames.is_empty() {
        return Err(anyhow!("no frames to write"));
    }
    fs::create_dir_all(output_dir).context("create artifact output dir")?;
    let fps = if fps == 0 { ARTIFACT_DEFAULT_FPS } else { fps };
    match output_mode {
        VisionArtifactOutputMode::Avi => {
            let filename = video_filename(output_mode, overwrite, epoch, iteration, sample_idx);
            let path = output_dir.join(filename);
            write_avi(&path, frames, fps, ffmpeg_path)?;
            Ok(ArtifactWriteOutcome {
                saved: 1,
                mode: VisionArtifactOutputMode::Avi,
                path,
            })
        }
        VisionArtifactOutputMode::Mp4 => {
            let filename = video_filename(output_mode, overwrite, epoch, iteration, sample_idx);
            let path = output_dir.join(filename);
            match write_mp4(&path, frames, fps, ffmpeg_path) {
                Ok(()) => Ok(ArtifactWriteOutcome {
                    saved: 1,
                    mode: VisionArtifactOutputMode::Mp4,
                    path,
                }),
                Err(_) => {
                    let fallback_name = video_filename(
                        VisionArtifactOutputMode::Avi,
                        overwrite,
                        epoch,
                        iteration,
                        sample_idx,
                    );
                    let fallback_path = output_dir.join(fallback_name);
                    write_avi(&fallback_path, frames, fps, ffmpeg_path)?;
                    Ok(ArtifactWriteOutcome {
                        saved: 1,
                        mode: VisionArtifactOutputMode::Avi,
                        path: fallback_path,
                    })
                }
            }
        }
        VisionArtifactOutputMode::Images => Err(anyhow!("video output requested in images mode")),
    }
}

fn denormalize(value: f32, channel: usize, mean: [f32; 3], std: [f32; 3]) -> u8 {
    let mut value = value * std[channel] + mean[channel];
    value = value.clamp(0.0, 1.0);
    (value * 255.0).round() as u8
}

fn video_filename(
    mode: VisionArtifactOutputMode,
    overwrite: bool,
    epoch: usize,
    iteration: usize,
    sample_idx: usize,
) -> String {
    let extension = match mode {
        VisionArtifactOutputMode::Avi => "avi",
        VisionArtifactOutputMode::Mp4 => "mp4",
        VisionArtifactOutputMode::Images => "png",
    };
    if overwrite {
        format!("sample_{:02}.{extension}", sample_idx)
    } else {
        format!("epoch_{epoch:03}_iter_{iteration:06}_sample_{sample_idx:02}.{extension}")
    }
}

fn write_mp4(
    path: &Path,
    frames: &[ArtifactFrame],
    fps: u32,
    ffmpeg_path: Option<&Path>,
) -> Result<()> {
    let output_path = prepare_output_path(path)?;
    let ffmpeg = resolve_ffmpeg(ffmpeg_path).ok_or_else(|| anyhow!("ffmpeg not found"))?;
    let temp_dir = create_temp_dir("artifact_frames")?;
    let temp_path = temp_dir.as_path();
    for (idx, frame) in frames.iter().enumerate() {
        let filename = format!("frame_{idx:05}.png");
        let frame_path = temp_path.join(filename);
        let image = RgbImage::from_vec(frame.width as u32, frame.height as u32, frame.rgb.clone())
            .ok_or_else(|| anyhow!("invalid frame buffer"))?;
        image.save(&frame_path).context("save mp4 frame")?;
    }
    let mut cmd = Command::new(ffmpeg);
    cmd.current_dir(temp_path)
        .arg("-y")
        .arg("-hide_banner")
        .arg("-nostats")
        .arg("-nostdin")
        .arg("-loglevel")
        .arg("error")
        .arg("-framerate")
        .arg(fps.to_string())
        .arg("-i")
        .arg("frame_%05d.png")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg(&output_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let status = cmd.status().context("run ffmpeg")?;
    let _ = fs::remove_dir_all(temp_path);
    if status.success() {
        Ok(())
    } else {
        Err(anyhow!("ffmpeg failed"))
    }
}

fn resolve_ffmpeg(ffmpeg_path: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = ffmpeg_path
        && path.exists()
    {
        return Some(path.to_path_buf());
    }
    let status = Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    if status.map(|status| status.success()).unwrap_or(false) {
        return Some(PathBuf::from("ffmpeg"));
    }
    None
}

fn create_temp_dir(prefix: &str) -> Result<PathBuf> {
    let mut base = env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    base.push(format!("{prefix}_{nanos}"));
    fs::create_dir_all(&base).context("create temp dir")?;
    Ok(base)
}

fn write_avi(
    path: &Path,
    frames: &[ArtifactFrame],
    fps: u32,
    ffmpeg_path: Option<&Path>,
) -> Result<()> {
    if resolve_ffmpeg(ffmpeg_path).is_some()
        && let Ok(()) = write_avi_ffmpeg(path, frames, fps, ffmpeg_path)
    {
        return Ok(());
    }
    write_avi_raw(path, frames, fps)
}

fn write_avi_ffmpeg(
    path: &Path,
    frames: &[ArtifactFrame],
    fps: u32,
    ffmpeg_path: Option<&Path>,
) -> Result<()> {
    let output_path = prepare_output_path(path)?;
    let ffmpeg = resolve_ffmpeg(ffmpeg_path).ok_or_else(|| anyhow!("ffmpeg not found"))?;
    let temp_dir = create_temp_dir("artifact_frames")?;
    let temp_path = temp_dir.as_path();
    for (idx, frame) in frames.iter().enumerate() {
        let filename = format!("frame_{idx:05}.png");
        let frame_path = temp_path.join(filename);
        let image = RgbImage::from_vec(frame.width as u32, frame.height as u32, frame.rgb.clone())
            .ok_or_else(|| anyhow!("invalid frame buffer"))?;
        image.save(&frame_path).context("save avi frame")?;
    }
    let mut cmd = Command::new(ffmpeg);
    cmd.current_dir(temp_path)
        .arg("-y")
        .arg("-hide_banner")
        .arg("-nostats")
        .arg("-nostdin")
        .arg("-loglevel")
        .arg("error")
        .arg("-framerate")
        .arg(fps.to_string())
        .arg("-i")
        .arg("frame_%05d.png")
        .arg("-c:v")
        .arg("mjpeg")
        .arg("-pix_fmt")
        .arg("yuvj420p")
        .arg(&output_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let status = cmd.status().context("run ffmpeg")?;
    let _ = fs::remove_dir_all(temp_path);
    if status.success() {
        Ok(())
    } else {
        Err(anyhow!("ffmpeg failed"))
    }
}

fn write_avi_raw(path: &Path, frames: &[ArtifactFrame], fps: u32) -> Result<()> {
    if frames.is_empty() {
        return Err(anyhow!("no frames to write"));
    }
    let width = frames[0].width;
    let height = frames[0].height;
    if width == 0 || height == 0 {
        return Err(anyhow!("invalid frame size"));
    }
    for frame in frames.iter().skip(1) {
        if frame.width != width || frame.height != height {
            return Err(anyhow!("frame size mismatch"));
        }
    }

    let row_stride = (width * 3).div_ceil(4) * 4;
    let frame_size = row_stride * height;
    let mut buf = Vec::new();

    write_fourcc(&mut buf, "RIFF");
    let riff_size_pos = buf.len();
    write_u32(&mut buf, 0);
    write_fourcc(&mut buf, "AVI ");

    write_fourcc(&mut buf, "LIST");
    let hdrl_size_pos = buf.len();
    write_u32(&mut buf, 0);
    write_fourcc(&mut buf, "hdrl");

    write_fourcc(&mut buf, "avih");
    write_u32(&mut buf, 56);
    write_u32(&mut buf, 1_000_000 / fps.max(1));
    write_u32(&mut buf, (frame_size * fps as usize) as u32);
    write_u32(&mut buf, 0);
    write_u32(&mut buf, 0x10);
    let avih_frames_pos = buf.len();
    write_u16(&mut buf, 0);
    write_u16(&mut buf, 0);
    write_u32(&mut buf, 1);
    write_u32(&mut buf, frame_size as u32);
    write_u32(&mut buf, width as u32);
    write_u32(&mut buf, height as u32);
    for _ in 0..4 {
        write_u32(&mut buf, 0);
    }

    write_fourcc(&mut buf, "LIST");
    let strl_size_pos = buf.len();
    write_u32(&mut buf, 0);
    write_fourcc(&mut buf, "strl");

    write_fourcc(&mut buf, "strh");
    write_u32(&mut buf, 56);
    write_fourcc(&mut buf, "vids");
    write_fourcc(&mut buf, "DIB ");
    write_u32(&mut buf, 0);
    write_u32(&mut buf, 0);
    write_u32(&mut buf, 0);
    write_u32(&mut buf, 1);
    write_u32(&mut buf, fps);
    write_u32(&mut buf, 0);
    let strh_frames_pos = buf.len();
    write_u32(&mut buf, 0);
    write_u32(&mut buf, frame_size as u32);
    write_u32(&mut buf, 0xFFFF_FFFF);
    write_u32(&mut buf, 0);
    write_i16(&mut buf, 0);
    write_i16(&mut buf, 0);
    write_i16(&mut buf, clamp_i16(width));
    write_i16(&mut buf, clamp_i16(height));

    write_fourcc(&mut buf, "strf");
    write_u32(&mut buf, 40);
    write_u32(&mut buf, 40);
    write_i32(&mut buf, width as i32);
    write_i32(&mut buf, height as i32);
    write_u16(&mut buf, 1);
    write_u16(&mut buf, 24);
    write_u32(&mut buf, 0);
    write_u32(&mut buf, frame_size as u32);
    write_i32(&mut buf, 0);
    write_i32(&mut buf, 0);
    write_u32(&mut buf, 0);
    write_u32(&mut buf, 0);

    let hdrl_end = buf.len();
    patch_u32(
        &mut buf,
        strl_size_pos,
        (hdrl_end - (strl_size_pos + 4)) as u32,
    );
    patch_u32(
        &mut buf,
        hdrl_size_pos,
        (hdrl_end - (hdrl_size_pos + 4)) as u32,
    );

    write_fourcc(&mut buf, "LIST");
    let movi_size_pos = buf.len();
    write_u32(&mut buf, 0);
    write_fourcc(&mut buf, "movi");
    let movi_start = buf.len();

    let mut idx_entries = Vec::with_capacity(frames.len());
    for frame in frames {
        let chunk_pos = buf.len();
        write_fourcc(&mut buf, "00db");
        let encoded = encode_bgr(frame, row_stride);
        let chunk_size = encoded.len();
        write_u32(&mut buf, chunk_size as u32);
        buf.extend_from_slice(&encoded);
        if chunk_size % 2 != 0 {
            buf.push(0);
        }
        idx_entries.push(AviIndexEntry {
            offset: (chunk_pos - movi_start) as u32,
            size: chunk_size as u32,
        });
    }

    let movi_end = buf.len();
    patch_u32(
        &mut buf,
        movi_size_pos,
        (movi_end - (movi_size_pos + 4)) as u32,
    );

    write_fourcc(&mut buf, "idx1");
    write_u32(&mut buf, (idx_entries.len() * 16) as u32);
    for entry in &idx_entries {
        write_fourcc(&mut buf, "00db");
        write_u32(&mut buf, 0x10);
        write_u32(&mut buf, entry.offset);
        write_u32(&mut buf, entry.size);
    }

    let riff_size = buf.len().saturating_sub(8);
    patch_u32(&mut buf, avih_frames_pos, frames.len() as u32);
    patch_u32(&mut buf, strh_frames_pos, frames.len() as u32);
    patch_u32(&mut buf, riff_size_pos, riff_size as u32);

    let mut file = fs::File::create(path).context("create avi file")?;
    file.write_all(&buf).context("write avi file")?;
    file.flush().context("flush avi file")?;
    Ok(())
}

fn prepare_output_path(path: &Path) -> Result<PathBuf> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).context("create video output dir")?;
    }
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        let cwd = env::current_dir().context("read current dir")?;
        Ok(cwd.join(path))
    }
}

#[derive(Clone, Copy, Debug)]
struct AviIndexEntry {
    offset: u32,
    size: u32,
}

fn encode_bgr(frame: &ArtifactFrame, row_stride: usize) -> Vec<u8> {
    let mut out = vec![0u8; row_stride * frame.height];
    for y in 0..frame.height {
        let dst_row = frame.height - 1 - y;
        for x in 0..frame.width {
            let src = (y * frame.width + x) * 3;
            let dst = dst_row * row_stride + x * 3;
            out[dst] = frame.rgb[src + 2];
            out[dst + 1] = frame.rgb[src + 1];
            out[dst + 2] = frame.rgb[src];
        }
    }
    out
}

fn write_fourcc(buf: &mut Vec<u8>, code: &str) {
    buf.extend_from_slice(code.as_bytes());
}

fn write_u16(buf: &mut Vec<u8>, value: u16) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn write_i16(buf: &mut Vec<u8>, value: i16) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn write_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn write_i32(buf: &mut Vec<u8>, value: i32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn clamp_i16(value: usize) -> i16 {
    value.min(i16::MAX as usize) as i16
}

fn patch_u32(buf: &mut [u8], pos: usize, value: u32) {
    let bytes = value.to_le_bytes();
    buf[pos..pos + 4].copy_from_slice(&bytes);
}

impl fmt::Display for ArtifactWriteOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "saved={} mode={:?} path={}",
            self.saved,
            self.mode,
            self.path.display()
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::train::artifacts::*;
    use crate::train::test_support::create_stub_ffmpeg;
    use avirus::AVI;

    fn sample_frames() -> Vec<ArtifactFrame> {
        vec![
            ArtifactFrame {
                width: 4,
                height: 4,
                rgb: vec![255u8; 4 * 4 * 3],
            },
            ArtifactFrame {
                width: 4,
                height: 4,
                rgb: vec![0u8; 4 * 4 * 3],
            },
        ]
    }

    fn find_fourcc(bytes: &[u8], tag: &[u8; 4]) -> Option<usize> {
        bytes.windows(4).position(|window| window == tag)
    }

    fn read_u32_le(bytes: &[u8], pos: usize) -> u32 {
        let mut data = [0u8; 4];
        data.copy_from_slice(&bytes[pos..pos + 4]);
        u32::from_le_bytes(data)
    }

    #[test]
    fn avi_writer_emits_riff_header() {
        let temp_dir = create_temp_dir("avi_test").expect("temp dir");
        let path = temp_dir.join("sample.avi");
        write_avi_raw(&path, &sample_frames(), 8).expect("avi write");
        let bytes = fs::read(&path).expect("read avi");
        assert!(bytes.starts_with(b"RIFF"));
        assert!(bytes[8..12].starts_with(b"AVI "));
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn avi_writer_idx_offsets_point_to_chunks() {
        let temp_dir = create_temp_dir("avi_idx").expect("temp dir");
        let path = temp_dir.join("sample.avi");
        write_avi_raw(&path, &sample_frames(), 8).expect("avi write");
        let bytes = fs::read(&path).expect("read avi");
        let movi_pos = find_fourcc(&bytes, b"movi").expect("movi");
        let movi_data_start = movi_pos + 4;
        let idx_pos = find_fourcc(&bytes, b"idx1").expect("idx1");
        let entry_start = idx_pos + 8;
        let offset = read_u32_le(&bytes, entry_start + 8) as usize;
        let size = read_u32_le(&bytes, entry_start + 12) as usize;
        let chunk_pos = movi_data_start + offset;
        assert!(chunk_pos + 8 <= bytes.len());
        assert_eq!(&bytes[chunk_pos..chunk_pos + 4], b"00db");
        let chunk_size = read_u32_le(&bytes, chunk_pos + 4) as usize;
        assert_eq!(chunk_size, size);
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn avi_writer_strh_chunk_is_well_formed() {
        let temp_dir = create_temp_dir("avi_strh").expect("temp dir");
        let path = temp_dir.join("sample.avi");
        write_avi_raw(&path, &sample_frames(), 8).expect("avi write");
        let bytes = fs::read(&path).expect("read avi");
        let strh_pos = find_fourcc(&bytes, b"strh").expect("strh");
        let size = read_u32_le(&bytes, strh_pos + 4) as usize;
        assert_eq!(size, 56);
        let next_pos = strh_pos + 8 + size;
        assert!(next_pos + 4 <= bytes.len());
        assert_eq!(&bytes[next_pos..next_pos + 4], b"strf");
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn avi_writer_roundtrip_parses_with_avirus() {
        let temp_dir = create_temp_dir("avi_roundtrip").expect("temp dir");
        let path = temp_dir.join("sample.avi");
        write_avi_raw(&path, &sample_frames(), 8).expect("avi write");
        let avi = AVI::new(&path).expect("parse avi");
        assert!(!avi.frames.meta.is_empty());
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn mp4_writer_uses_stub_ffmpeg() {
        let temp_dir = create_temp_dir("mp4_test").expect("temp dir");
        let bin_dir = temp_dir.join("bin");
        let script_path = create_stub_ffmpeg(&bin_dir).expect("ffmpeg stub");

        let output = temp_dir.join("sample.mp4");
        write_mp4(&output, &sample_frames(), 8, Some(&script_path)).expect("mp4 write");
        assert!(output.is_file());
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn collect_frames_respects_layout_and_channels() {
        let data = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let frames = collect_frames(&data, 1, 1, 3, 1, 2, 0, [0.0; 3], [1.0; 3]);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].rgb, vec![255, 0, 0, 0, 255, 0]);
    }
}
