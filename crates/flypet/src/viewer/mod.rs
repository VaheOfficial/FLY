//! The brain viewer: a second window showing every brain neuron with a known
//! soma, seen from the front, coloured by cell type and lit as it spikes.

mod palette;

use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use flybrain::connectome::Connectome;
use wgpu::util::DeviceExt;

use crate::gpu::{Gpu, acquire};
use crate::snapshot::save_png;

/// How long a spike stays visible: the glow falls to 1/e after this.
const GLOW_DECAY: Duration = Duration::from_millis(180);
/// Point size in pixels.
const POINT_PX: f32 = 1.5;
/// In this specimen the head is flexed: the brain faces anterior (its frontal
/// plane is dataset x-y) while the nerve cord runs posterior along z. No soma
/// lies in the neck connective, so everything below this z is cord.
const NECK_Z: i32 = 46_000;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct View {
    scale: [f32; 2],
    offset: [f32; 2],
    half_size: [f32; 2],
    _pad: [f32; 2],
}

pub struct Viewer {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    view_uniform: wgpu::Buffer,
    activity_buffer: wgpu::Buffer,
    /// Neuron index of each drawn point, so activity can be gathered.
    neuron_of_point: Vec<u32>,
    activity: Vec<f32>,
    /// Soma bounds in the viewing plane: min x, min y, max x, max y.
    bounds: [f32; 4],
}

impl Viewer {
    pub fn new(
        gpu: &Gpu,
        surface: wgpu::Surface<'static>,
        size: (u32, u32),
        net: &Connectome,
    ) -> Result<Self> {
        let config = gpu.surface_config(&surface, size, false)?;

        let (neuron_of_point, positions) = frontal_brain_positions(net);
        let bounds = bounds_of(&positions);
        let colours: Vec<[f32; 4]> = neuron_of_point
            .iter()
            .map(|&i| palette::for_type(net.neurons[i as usize].type_name))
            .collect();
        let activity = vec![0.0f32; positions.len()];

        let storage = |label: &str, contents: &[u8], writable: bool| {
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage: if writable {
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
                    } else {
                        wgpu::BufferUsages::STORAGE
                    },
                })
        };
        let positions_buffer = storage("soma positions", bytemuck::cast_slice(&positions), false);
        let colours_buffer = storage("colours", bytemuck::cast_slice(&colours), false);
        let activity_buffer = storage("activity", bytemuck::cast_slice(&activity), true);
        let view_uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view"),
            size: std::mem::size_of::<View>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("points"),
                source: wgpu::ShaderSource::Wgsl(include_str!("points.wgsl").into()),
            });
        let read_only = wgpu::BufferBindingType::Storage { read_only: true };
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("points"),
                entries: &[
                    binding(0, wgpu::BufferBindingType::Uniform),
                    binding(1, read_only),
                    binding(2, read_only),
                    binding(3, read_only),
                ],
            });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("points"),
            layout: &layout,
            entries: &[
                entry(0, &view_uniform),
                entry(1, &positions_buffer),
                entry(2, &colours_buffer),
                entry(3, &activity_buffer),
            ],
        });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("points"),
                bind_group_layouts: &[Some(&layout)],
                ..Default::default()
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("points"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vertex"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fragment"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        Ok(Self {
            surface,
            config,
            pipeline,
            bind_group,
            view_uniform,
            activity_buffer,
            neuron_of_point,
            activity,
            bounds,
        })
    }

    pub fn point_count(&self) -> usize {
        self.neuron_of_point.len()
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
        for (glow, &neuron) in self.activity.iter_mut().zip(&self.neuron_of_point) {
            let spikes = counts.get(neuron as usize).copied().unwrap_or(0);
            *glow = (*glow * keep + spikes as f32 * 0.6).min(1.0);
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
                            r: 0.002,
                            g: 0.0025,
                            b: 0.006,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..6, 0..self.point_count() as u32);
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
            half_size: [POINT_PX / w, POINT_PX / h],
            _pad: [0.0; 2],
        }
    }
}

fn entry(index: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding: index,
        resource: buffer.as_entire_binding(),
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

/// Brain neurons with a soma, seen from the front: dataset x across, y
/// down, so dorsal is at the top.
fn frontal_brain_positions(net: &Connectome) -> (Vec<u32>, Vec<[f32; 2]>) {
    net.neurons
        .iter()
        .enumerate()
        .filter_map(|(i, n)| match n.soma {
            Some([x, y, z]) if z < NECK_Z => Some((i as u32, [x as f32, y as f32])),
            _ => None,
        })
        .unzip()
}

fn bounds_of(points: &[[f32; 2]]) -> [f32; 4] {
    points.iter().fold(
        [f32::MAX, f32::MAX, f32::MIN, f32::MIN],
        |[x0, y0, x1, y1], &[x, y]| [x0.min(x), y0.min(y), x1.max(x), y1.max(y)],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_cover_all_points() {
        assert_eq!(
            bounds_of(&[[1.0, 5.0], [-2.0, 3.0], [4.0, -1.0]]),
            [-2.0, -1.0, 4.0, 5.0]
        );
    }
}
