//! The brain viewer: a second window showing every brain neuron, seen from
//! the front, coloured by cell type and lit as it spikes. Branching
//! morphologies are drawn when a morphology file is available; cell bodies
//! are always drawn.

mod lines;
mod palette;
mod points;
mod taper;

use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use flybrain::connectome::Connectome;
use flybrain::morphology::Morphology;
use wgpu::util::DeviceExt;

use crate::gpu::{Gpu, acquire};
use crate::snapshot::save_png;
use lines::Lines;
use points::Points;

/// How long a spike stays visible: the glow falls to 1/e after this.
const GLOW_DECAY: Duration = Duration::from_millis(180);
/// In this specimen the head is flexed: the brain faces anterior (its frontal
/// plane is dataset x-y) while the nerve cord runs posterior along z. No soma
/// lies in the neck connective, so everything below this z is cord.
const NECK_Z: i32 = 46_000;

/// Light a silent line vertex emits before region weight and depth: with
/// millions of lines adding up, small is right.
const RESTING_LIGHT: f32 = 0.011;
/// Depth range of the brain in packed morphology units (16 nm): anterior
/// surface to posterior surface. Lines fade across it.
const DEPTH_NEAR: f32 = 6_000.0;
const DEPTH_FAR: f32 = 20_000.0;

/// Uniform shared by both pipelines; layout matches the shaders.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct View {
    scale: [f32; 2],
    offset: [f32; 2],
    half_size: [f32; 2],
    voxels_per_unit: f32,
    resting_light: f32,
    depth_near: f32,
    depth_far: f32,
    _pad: [f32; 2],
}

/// Light accumulates: source colour is added to what is already there.
pub(super) const ADDITIVE: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent::REPLACE,
};

pub struct Viewer {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    view_uniform: wgpu::Buffer,
    activity_buffer: wgpu::Buffer,
    /// Per neuron, 0 (silent) to 1 (just spiked).
    activity: Vec<f32>,
    points: Points,
    lines: Option<Lines>,
    /// Soma bounds in the viewing plane: min x, min y, max x, max y.
    bounds: [f32; 4],
    pub show_somas: bool,
}

impl Viewer {
    pub fn new(
        gpu: &Gpu,
        surface: wgpu::Surface<'static>,
        size: (u32, u32),
        net: &Connectome,
        morphology: Option<&Morphology>,
    ) -> Result<Self> {
        let config = gpu.surface_config(&surface, size, false)?;

        let colours: Vec<[f32; 4]> = net
            .neurons
            .iter()
            .map(|n| palette::for_neuron(net, n))
            .collect();
        let activity = vec![0.0f32; net.neuron_count()];
        let colours_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("neuron colours"),
                contents: bytemuck::cast_slice(&colours),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let activity_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("neuron activity"),
                contents: bytemuck::cast_slice(&activity),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let view_uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view"),
            size: std::mem::size_of::<View>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let points = Points::new(
            gpu,
            config.format,
            net,
            NECK_Z,
            &view_uniform,
            &colours_buffer,
            &activity_buffer,
        )?;
        let lines = morphology
            .map(|m| {
                Lines::new(
                    gpu,
                    config.format,
                    m,
                    NECK_Z,
                    &view_uniform,
                    &colours_buffer,
                    &activity_buffer,
                )
            })
            .transpose()?;

        Ok(Self {
            surface,
            config,
            view_uniform,
            activity_buffer,
            activity,
            bounds: points.bounds,
            points,
            lines,
            show_somas: true,
        })
    }

    pub fn describe(&self) -> String {
        match &self.lines {
            Some(lines) => format!(
                "{} somas, {} morphology vertices in {} strips",
                self.points.count, lines.vertex_count, lines.strip_count
            ),
            None => format!("{} somas, no morphology file", self.points.count),
        }
    }

    pub fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&gpu.device, &self.config);
        }
    }

    /// Decay the glow and add this frame's spikes.
    pub fn update(&mut self, counts: &[u32], elapsed: Duration) {
        let keep = (-elapsed.as_secs_f32() / GLOW_DECAY.as_secs_f32()).exp();
        for (glow, &spikes) in self.activity.iter_mut().zip(counts) {
            *glow = (*glow * keep + spikes as f32 * 0.6).min(1.0);
        }
        if counts.is_empty() {
            for glow in &mut self.activity {
                *glow *= keep;
            }
        }
    }

    /// Draw the brain; if `snapshot` is given, also save the frame there.
    pub fn draw(&mut self, gpu: &Gpu, snapshot: Option<&Path>) -> Result<()> {
        let Some(frame) = acquire(gpu, &self.surface, &self.config)? else {
            return Ok(());
        };
        gpu.queue.write_buffer(
            &self.activity_buffer,
            0,
            bytemuck::cast_slice(&self.activity),
        );
        gpu.queue
            .write_buffer(&self.view_uniform, 0, bytemuck::bytes_of(&self.view()));

        let view = frame.texture.create_view(&Default::default());
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("brain"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0015,
                            g: 0.002,
                            b: 0.004,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });
            if let Some(lines) = &self.lines {
                lines.draw(&mut pass);
            }
            if self.show_somas {
                self.points.draw(&mut pass);
            }
        }
        gpu.queue.submit([encoder.finish()]);
        if let Some(path) = snapshot {
            save_png(gpu, &frame.texture, path)?;
        }
        gpu.queue.present(frame);
        Ok(())
    }

    /// Fit the soma bounds into the window with a margin, preserving aspect.
    fn view(&self) -> View {
        let [min_x, min_y, max_x, max_y] = self.bounds;
        let (w, h) = (self.config.width as f32, self.config.height as f32);
        let span = (max_x - min_x).max(1.0);
        let rise = (max_y - min_y).max(1.0);
        // Clip units per voxel that fit both axes at 92% of the window.
        let per_voxel = (1.84 / span).min(1.84 * (w / h) / rise);
        let scale_x = per_voxel;
        let scale_y = per_voxel * (w / h);
        View {
            scale: [scale_x, -scale_y],
            offset: [
                -(min_x + max_x) * 0.5 * scale_x,
                (min_y + max_y) * 0.5 * scale_y,
            ],
            half_size: [points::POINT_PX / w, points::POINT_PX / h],
            voxels_per_unit: lines::VOXELS_PER_UNIT,
            resting_light: RESTING_LIGHT,
            depth_near: DEPTH_NEAR,
            depth_far: DEPTH_FAR,
            _pad: [0.0; 2],
        }
    }
}

fn binding(index: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: index,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn entry(index: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding: index,
        resource: buffer.as_entire_binding(),
    }
}
