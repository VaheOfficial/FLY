//! Cell bodies as small squares in the frontal plane.

use anyhow::Result;
use flybrain::connectome::Connectome;
use wgpu::util::DeviceExt;

use super::{ADDITIVE, binding, entry};
use crate::gpu::Gpu;

/// Point size in pixels.
pub const POINT_PX: f32 = 1.5;

pub struct Points {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    pub count: usize,
    /// Soma bounds in the viewing plane: min x, min y, max x, max y.
    pub bounds: [f32; 4],
}

impl Points {
    /// One point per brain neuron with a soma (dataset z above the neck),
    /// seen from the front: x across, y down, so dorsal is at the top.
    pub fn new(
        gpu: &Gpu,
        format: wgpu::TextureFormat,
        net: &Connectome,
        neck_z: i32,
        view_uniform: &wgpu::Buffer,
        colours_buffer: &wgpu::Buffer,
        activity_buffer: &wgpu::Buffer,
    ) -> Result<Self> {
        let (neurons, positions): (Vec<u32>, Vec<[f32; 4]>) = net
            .neurons
            .iter()
            .enumerate()
            .filter_map(|(i, n)| match n.soma {
                Some([x, y, z]) if z < neck_z => {
                    Some((i as u32, [x as f32, y as f32, z as f32, 0.0]))
                }
                _ => None,
            })
            .unzip();
        let bounds = positions.iter().fold(
            [f32::MAX, f32::MAX, f32::MIN, f32::MIN],
            |[x0, y0, x1, y1], &[x, y, _, _]| [x0.min(x), y0.min(y), x1.max(x), y1.max(y)],
        );

        let storage = |label: &str, contents: &[u8]| {
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let positions_buffer = storage("soma positions", bytemuck::cast_slice(&positions));
        let neuron_buffer = storage("point neuron", bytemuck::cast_slice(&neurons));

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
                    binding(4, read_only),
                ],
            });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("points"),
            layout: &layout,
            entries: &[
                entry(0, view_uniform),
                entry(1, &positions_buffer),
                entry(2, &neuron_buffer),
                entry(3, colours_buffer),
                entry(4, activity_buffer),
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
                        format,
                        blend: Some(ADDITIVE),
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
            pipeline,
            bind_group,
            count: positions.len(),
            bounds,
        })
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..6, 0..self.count as u32);
    }
}
