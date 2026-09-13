//! The one GPU device the shell opens, shared by every window and the
//! simulation.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use flybrain::connectome::Connectome;
use flybrain::gpu::buffers::ConnectomeBuffers;
use flybrain::gpu::{self, GpuContext};
use winit::window::Window;

pub struct Gpu {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Gpu {
    /// Open the adapter that can present to `window`, with limits large
    /// enough to simulate `net`.
    pub fn open(window: &Arc<Window>, net: &Connectome) -> Result<(Self, wgpu::Surface<'static>)> {
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
        Ok((
            Self {
                instance,
                adapter,
                device,
                queue,
            },
            surface,
        ))
    }

    /// A surface for another window on the same device.
    pub fn surface(&self, window: &Arc<Window>) -> Result<wgpu::Surface<'static>> {
        self.instance
            .create_surface(window.clone())
            .context("creating window surface")
    }

    /// Default configuration for a surface at `size`, preferring an alpha
    /// mode that lets the window be transparent when `transparent` is set.
    pub fn surface_config(
        &self,
        surface: &wgpu::Surface<'_>,
        size: (u32, u32),
        transparent: bool,
    ) -> Result<wgpu::SurfaceConfiguration> {
        let mut config = surface
            .get_default_config(&self.adapter, size.0.max(1), size.1.max(1))
            .ok_or_else(|| anyhow!("surface is not supported by the adapter"))?;
        let capabilities = surface.get_capabilities(&self.adapter);
        if transparent {
            config.alpha_mode = [
                wgpu::CompositeAlphaMode::PreMultiplied,
                wgpu::CompositeAlphaMode::PostMultiplied,
                wgpu::CompositeAlphaMode::Inherit,
            ]
            .into_iter()
            .find(|mode| capabilities.alpha_modes.contains(mode))
            .unwrap_or(config.alpha_mode);
        }
        if capabilities.usages.contains(wgpu::TextureUsages::COPY_SRC) {
            config.usage |= wgpu::TextureUsages::COPY_SRC;
        }
        surface.configure(&self.device, &config);
        Ok(config)
    }

    /// The simulation's view of this device.
    pub fn sim_context(&self) -> GpuContext {
        GpuContext::from_parts(&self.adapter, self.device.clone(), self.queue.clone())
    }
}

/// Acquire the next frame of a surface, reconfiguring when the surface says
/// so. `None` means skip this frame.
pub fn acquire(
    gpu: &Gpu,
    surface: &wgpu::Surface<'_>,
    config: &wgpu::SurfaceConfiguration,
) -> Result<Option<wgpu::SurfaceTexture>> {
    use wgpu::CurrentSurfaceTexture as Acquired;
    Ok(match surface.get_current_texture() {
        Acquired::Success(frame) => Some(frame),
        Acquired::Suboptimal(frame) => {
            surface.configure(&gpu.device, config);
            Some(frame)
        }
        Acquired::Timeout | Acquired::Occluded => None,
        Acquired::Outdated | Acquired::Lost => {
            surface.configure(&gpu.device, config);
            None
        }
        Acquired::Validation => return Err(anyhow!("surface validation error")),
    })
}
