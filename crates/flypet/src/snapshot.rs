//! Save a rendered frame as a PNG, so what the windows show can be checked
//! without a screen.

use std::path::Path;

use anyhow::{Context, Result, anyhow};

use crate::gpu::Gpu;

/// Copy `texture` (which must have `COPY_SRC` usage and an 8-bit RGBA or
/// BGRA format) to `path` as an opaque PNG. Blocks until the copy is done.
pub fn save_png(gpu: &Gpu, texture: &wgpu::Texture, path: &Path) -> Result<()> {
    let (width, height) = (texture.width(), texture.height());
    let bgra = match texture.format() {
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => false,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => true,
        other => return Err(anyhow!("cannot snapshot texture format {other:?}")),
    };
    let unpadded_row = width * 4;
    let padded_row = unpadded_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("snapshot"),
        size: u64::from(padded_row) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(height),
            },
        },
        texture.size(),
    );
    gpu.queue.submit([encoder.finish()]);

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .context("waiting for snapshot copy")?;
    receiver
        .recv()
        .context("snapshot map callback never ran")?
        .context("mapping snapshot buffer")?;

    let padded = slice.get_mapped_range().context("reading snapshot")?;
    let mut rgba = Vec::with_capacity((unpadded_row * height) as usize);
    for row in padded.chunks(padded_row as usize) {
        for px in row[..unpadded_row as usize].chunks(4) {
            let (r, b) = if bgra { (px[2], px[0]) } else { (px[0], px[2]) };
            rgba.extend_from_slice(&[r, px[1], b, 255]);
        }
    }
    drop(padded);
    staging.unmap();

    let file =
        std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .context("png header")?
        .write_image_data(&rgba)
        .context("png data")?;
    Ok(())
}
