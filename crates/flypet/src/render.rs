//! Presenting to the transparent window. For now the whole window is one
//! translucent colour whose opacity follows a firing rate: enough to see the
//! brain's output on the desktop before any fly is drawn.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use flybrain::connectome::Connectome;
use flybrain::gpu::buffers::ConnectomeBuffers;
use flybrain::gpu::{self, GpuContext};
use winit::window::Window;

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Renderer {
    /// Open the GPU once for both presenting and simulating. Returns the
    /// renderer and the simulation context sharing its device.
    pub fn new(window: Arc<Window>, net: &Connectome) -> Result<(Self, GpuContext)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(window.clone())
            .context("creating window surface")?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .map_err(|e| anyhow!("no adapter can present to this window: {e}"))?;
        let limits = gpu::required_limits(&adapter, ConnectomeBuffers::largest_buffer_bytes(net))
            .map_err(|e| anyhow!("{e}"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("flypet"),
            required_limits: limits,
            ..Default::default()
        }))
        .context("requesting device")?;

        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or_else(|| anyhow!("surface is not supported by the adapter"))?;
        let capabilities = surface.get_capabilities(&adapter);
        config.alpha_mode = [
            wgpu::CompositeAlphaMode::PreMultiplied,
            wgpu::CompositeAlphaMode::PostMultiplied,
            wgpu::CompositeAlphaMode::Inherit,
        ]
        .into_iter()
        .find(|mode| capabilities.alpha_modes.contains(mode))
        .unwrap_or(config.alpha_mode);
        surface.configure(&device, &config);

        let sim_context = GpuContext::from_parts(&adapter, device.clone(), queue.clone());
        Ok((
            Self {
                surface,
                config,
                device,
                queue,
            },
            sim_context,
        ))
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Fill the window with `rgb` at `opacity` (premultiplied), in 0..=1.
    pub fn present(&mut self, rgb: [f64; 3], opacity: f64) -> Result<()> {
        use wgpu::CurrentSurfaceTexture as Acquired;
        let frame = match self.surface.get_current_texture() {
            Acquired::Success(frame) => frame,
            Acquired::Suboptimal(frame) => {
                self.surface.configure(&self.device, &self.config);
                frame
            }
            Acquired::Timeout | Acquired::Occluded => return Ok(()),
            Acquired::Outdated | Acquired::Lost => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            Acquired::Validation => return Err(anyhow!("surface validation error")),
        };
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());
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
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
        Ok(())
    }
}
