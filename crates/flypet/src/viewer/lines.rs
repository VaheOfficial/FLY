//! Neuron morphologies as line strips: every branch of every brain neuron,
//! coloured by cell type and lit by activity.

use anyhow::Result;
use flybrain::morphology::{Morphology, POSITION_UNIT_NM};
use wgpu::util::DeviceExt;

use super::{ADDITIVE, binding, entry, taper};
use crate::gpu::Gpu;

/// Index buffers restart a strip on this value.
const RESTART: u32 = u32::MAX;
/// Voxel size of the dataset; soma positions are in voxels, morphology in nm.
const VOXEL_NM: f32 = 8.0;

/// Dataset voxels per packed morphology position unit.
pub const VOXELS_PER_UNIT: f32 = POSITION_UNIT_NM / VOXEL_NM;

/// GPU-resident morphology of the brain, ready to draw.
pub struct Lines {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    pub vertex_count: usize,
    pub strip_count: usize,
}

impl Lines {
    /// Upload every strip vertex above the neck (brain only). The view
    /// uniform, per-neuron colours and per-neuron activity are the viewer's.
    pub fn new(
        gpu: &Gpu,
        format: wgpu::TextureFormat,
        morphology: &Morphology,
        neck_z_voxels: i32,
        view_uniform: &wgpu::Buffer,
        colours_buffer: &wgpu::Buffer,
        activity_buffer: &wgpu::Buffer,
    ) -> Result<Self> {
        let neck_z = (neck_z_voxels as f32 / VOXELS_PER_UNIT) as u16;
        let fade = taper::tip_fade(morphology);
        let positions: Vec<[u32; 2]> = morphology
            .positions
            .iter()
            .zip(&fade)
            .map(|(&[x, y, z], &fade)| {
                [
                    u32::from(x) | (u32::from(y) << 16),
                    u32::from(z) | (u32::from(fade) << 16),
                ]
            })
            .collect();

        // One index list of all brain strips, split wherever a strip leaves
        // the brain, with a restart marker between strips.
        let mut indices: Vec<u32> =
            Vec::with_capacity(morphology.vertex_count() + morphology.strips.len());
        let mut strip_count = 0usize;
        for strip in &morphology.strips {
            let mut run = 0u32;
            for v in strip.first..strip.first + strip.len {
                if morphology.positions[v as usize][2] < neck_z {
                    indices.push(v);
                    run += 1;
                } else if run > 0 {
                    indices.push(RESTART);
                    strip_count += 1;
                    run = 0;
                }
            }
            if run > 0 {
                indices.push(RESTART);
                strip_count += 1;
            }
        }

        let storage = |label: &str, contents: &[u8], usage: wgpu::BufferUsages| {
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage,
                })
        };
        let positions_buffer = storage(
            "morphology positions",
            bytemuck::cast_slice(&positions),
            wgpu::BufferUsages::STORAGE,
        );
        let neuron_buffer = storage(
            "vertex neuron",
            bytemuck::cast_slice(&morphology.vertex_neuron),
            wgpu::BufferUsages::STORAGE,
        );
        let index_buffer = storage(
            "morphology indices",
            bytemuck::cast_slice(&indices),
            wgpu::BufferUsages::INDEX,
        );

        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("lines"),
                source: wgpu::ShaderSource::Wgsl(include_str!("lines.wgsl").into()),
            });
        let read_only = wgpu::BufferBindingType::Storage { read_only: true };
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lines"),
                entries: &[
                    binding(0, wgpu::BufferBindingType::Uniform),
                    binding(1, read_only),
                    binding(2, read_only),
                    binding(3, read_only),
                    binding(4, read_only),
                ],
            });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lines"),
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
                label: Some("lines"),
                bind_group_layouts: &[Some(&layout)],
                ..Default::default()
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("lines"),
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
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineStrip,
                    strip_index_format: Some(wgpu::IndexFormat::Uint32),
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        Ok(Self {
            pipeline,
            bind_group,
            index_count: indices.len() as u32,
            index_buffer,
            vertex_count: positions.len(),
            strip_count,
        })
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}
