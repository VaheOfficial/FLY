//! Presenting to the transparent pet window. For now the whole window is
//! one translucent colour whose opacity follows a firing rate: enough to see
//! the brain's output on the desktop before any fly is drawn.

use anyhow::Result;

use crate::gpu::{Gpu, acquire};

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
}

impl Renderer {
    pub fn new(gpu: &Gpu, surface: wgpu::Surface<'static>, size: (u32, u32)) -> Result<Self> {
        let config = gpu.surface_config(&surface, size, true)?;
        Ok(Self { surface, config })
    }

    pub fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&gpu.device, &self.config);
        }
    }

    /// Fill the window with `rgb` at `opacity` (premultiplied), in 0..=1.
    pub fn present(&mut self, gpu: &Gpu, rgb: [f64; 3], opacity: f64) -> Result<()> {
        let Some(frame) = acquire(gpu, &self.surface, &self.config)? else {
            return Ok(());
        };
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: rgb[0] * opacity,
                        g: rgb[1] * opacity,
                        b: rgb[2] * opacity,
                        a: opacity,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            ..Default::default()
        });
        gpu.queue.submit([encoder.finish()]);
        gpu.queue.present(frame);
        Ok(())
    }
}
