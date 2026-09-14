//! The GPU simulator: the CPU model's step, batched into command buffers.
//!
//! Steps are queued with [`GpuSim::queue_step`] and executed together by
//! [`GpuSim::submit`], so a frame's worth of steps costs one submission.
//! Spike counts accumulate on the GPU until [`GpuSim::take_spike_counts`].

use bytemuck::{Pod, Zeroable};
use wgpu::{BindGroup, Buffer, BufferUsages, ComputePipeline};

use super::GpuContext;
use super::buffers::{ConnectomeBuffers, download, init_buffer, zeroed_buffer};
use crate::backend::Simulator;
use crate::connectome::Connectome;
use crate::sim::{LifParams, SignPolicy, StepCoefficients};

/// Fixed-point units per mV in the pending-drive accumulator: 6e-5 mV
/// resolution with a range of about ±130 V, far beyond any real drive.
const FIXED_SCALE: f32 = 16384.0;

/// Most external injections one batch can carry.
const INJECTION_CAPACITY: usize = 1 << 16;

/// Uniform block read by every kernel; layout matches `lif.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    v_rest: f32,
    v_thresh: f32,
    v_reset: f32,
    decay_m: f32,
    decay_syn: f32,
    coupling: f32,
    fixed_scale: f32,
    neuron_count: u32,
    refractory_steps: u32,
    step: u32,
    slot: u32,
    inject_start: u32,
    inject_count: u32,
    ring_stride: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Injection {
    neuron: u32,
    drive_fixed: i32,
}

pub struct GpuSim {
    ctx: GpuContext,
    params: LifParams,
    neuron_count: u32,
    delay_steps: u32,
    ring_stride: u32,
    step: u32,
    template: Params,

    inject_pipeline: ComputePipeline,
    propagate_pipeline: ComputePipeline,
    integrate_pipeline: ComputePipeline,
    bind_group: BindGroup,

    params_uniform: Buffer,
    /// One `Params` per queued step, copied into the uniform before each step.
    params_batch: Buffer,
    /// Per neuron `(v, g)`.
    state: Buffer,
    spike_ring: Buffer,
    spike_counts: Buffer,
    /// Per ring slot: `[spike count, 1, 1]` indirect dispatch arguments.
    indirect: Buffer,
    injections: Buffer,

    queued_params: Vec<Params>,
    queued_injections: Vec<Injection>,
    /// Injections added since the last `queue_step`, applied on the next step.
    pending_injections: usize,
}

impl GpuSim {
    pub fn new(ctx: GpuContext, net: &Connectome, params: LifParams, signs: SignPolicy) -> Self {
        let n = net.neuron_count() as u32;
        let coefficients = StepCoefficients::new(&params);
        let delay_steps = coefficients.delay_steps as u32;
        let ring_stride = n + 1;

        let table = ConnectomeBuffers::upload(&ctx, net);
        let drive_scale: Vec<f32> = signs
            .per_neuron(net)
            .iter()
            .map(|sign| sign * params.w_syn * FIXED_SCALE)
            .collect();
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        let drive_scale = init_buffer(&ctx, "drive_scale", &drive_scale, BufferUsages::STORAGE);
        let state = init_buffer(
            &ctx,
            "state",
            &vec![[params.v_rest, 0.0f32]; n as usize],
            storage,
        );
        let refractory_until = zeroed_buffer::<u32>(&ctx, "refractory_until", n as usize, storage);
        let pending_drive = zeroed_buffer::<i32>(&ctx, "pending_drive", n as usize, storage);
        let spike_ring = zeroed_buffer::<u32>(
            &ctx,
            "spike_ring",
            (delay_steps * ring_stride) as usize,
            storage,
        );
        let spike_counts = zeroed_buffer::<u32>(&ctx, "spike_counts", n as usize, storage);
        let indirect_init: Vec<u32> = (0..delay_steps).flat_map(|_| [0, 1, 1]).collect();
        let indirect = init_buffer(
            &ctx,
            "indirect",
            &indirect_init,
            BufferUsages::INDIRECT | BufferUsages::COPY_DST,
        );
        let injections =
            zeroed_buffer::<Injection>(&ctx, "injections", INJECTION_CAPACITY, storage);
        let params_uniform = zeroed_buffer::<Params>(
            &ctx,
            "params",
            1,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );
        let params_batch = zeroed_buffer::<Params>(
            &ctx,
            "params_batch",
            1,
            BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );

        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("lif"),
                source: wgpu::ShaderSource::Wgsl(include_str!("lif.wgsl").into()),
            });
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lif"),
                entries: &layout_entries(),
            });
        let buffers: [&Buffer; 11] = [
            &params_uniform,
            &table.row_ptr,
            &table.post,
            &table.weight_pairs,
            &drive_scale,
            &state,
            &refractory_until,
            &pending_drive,
            &spike_ring,
            &spike_counts,
            &injections,
        ];
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lif"),
            layout: &layout,
            entries: &entries,
        });
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("lif"),
                bind_group_layouts: &[Some(&layout)],
                ..Default::default()
            });
        let pipeline = |entry: &str| {
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };

        let template = Params {
            v_rest: params.v_rest,
            v_thresh: params.v_thresh,
            v_reset: params.v_reset,
            decay_m: coefficients.decay_m,
            decay_syn: coefficients.decay_syn,
            coupling: coefficients.coupling,
            fixed_scale: FIXED_SCALE,
            neuron_count: n,
            refractory_steps: coefficients.refractory_steps,
            step: 0,
            slot: 0,
            inject_start: 0,
            inject_count: 0,
            ring_stride,
            _pad: [0; 2],
        };

        Self {
            inject_pipeline: pipeline("inject"),
            propagate_pipeline: pipeline("propagate"),
            integrate_pipeline: pipeline("integrate"),
            bind_group,
            params_uniform,
            params_batch,
            state,
            spike_ring,
            spike_counts,
            indirect,
            injections,
            ctx,
            params,
            neuron_count: n,
            delay_steps,
            ring_stride,
            step: 0,
            template,
            queued_params: Vec::new(),
            queued_injections: Vec::new(),
            pending_injections: 0,
        }
    }

    pub fn params(&self) -> &LifParams {
        &self.params
    }

    pub fn adapter_name(&self) -> &str {
        &self.ctx.adapter_info.name
    }

    /// Steps executed or queued so far.
    pub fn step_count(&self) -> u64 {
        u64::from(self.step)
    }

    /// Add external drive to one neuron, in mV, landing on the next queued step.
    pub fn inject(&mut self, neuron: u32, delta_g: f32) {
        self.queued_injections.push(Injection {
            neuron,
            drive_fixed: (delta_g * FIXED_SCALE).round() as i32,
        });
        self.pending_injections += 1;
    }

    /// Queue one step. Nothing runs until [`Self::submit`].
    pub fn queue_step(&mut self) {
        let inject_count = self.pending_injections as u32;
        self.queued_params.push(Params {
            step: self.step,
            slot: self.step % self.delay_steps,
            inject_start: (self.queued_injections.len() - self.pending_injections) as u32,
            inject_count,
            ..self.template
        });
        self.pending_injections = 0;
        self.step += 1;
    }

    /// Execute every queued step in one command buffer. Returns without
    /// waiting; the queue orders later buffer writes after this work.
    pub fn submit(&mut self) {
        if self.queued_params.is_empty() {
            return;
        }
        assert!(
            self.queued_injections.len() <= INJECTION_CAPACITY,
            "too many injections in one batch: {} > {INJECTION_CAPACITY}",
            self.queued_injections.len()
        );
        self.ensure_params_batch_capacity();
        self.ctx.queue.write_buffer(
            &self.params_batch,
            0,
            bytemuck::cast_slice(&self.queued_params),
        );
        if !self.queued_injections.is_empty() {
            self.ctx.queue.write_buffer(
                &self.injections,
                0,
                bytemuck::cast_slice(&self.queued_injections),
            );
        }

        let params_size = std::mem::size_of::<Params>() as u64;
        let mut encoder = self.ctx.device.create_command_encoder(&Default::default());
        for (k, p) in self.queued_params.iter().enumerate() {
            let slot_bytes = u64::from(p.slot * self.ring_stride) * 4;
            encoder.copy_buffer_to_buffer(
                &self.params_batch,
                k as u64 * params_size,
                &self.params_uniform,
                0,
                params_size,
            );
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_bind_group(0, &self.bind_group, &[]);
                if p.inject_count > 0 {
                    pass.set_pipeline(&self.inject_pipeline);
                    pass.dispatch_workgroups(p.inject_count.div_ceil(64), 1, 1);
                }
                pass.set_pipeline(&self.propagate_pipeline);
                pass.dispatch_workgroups_indirect(&self.indirect, u64::from(p.slot) * 12);
            }
            // The delivered slot is refilled with this step's spikes.
            encoder.clear_buffer(&self.spike_ring, slot_bytes, Some(4));
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_pipeline(&self.integrate_pipeline);
                pass.dispatch_workgroups(self.neuron_count.div_ceil(256), 1, 1);
            }
            encoder.copy_buffer_to_buffer(
                &self.spike_ring,
                slot_bytes,
                &self.indirect,
                u64::from(p.slot) * 12,
                4,
            );
        }
        self.ctx.queue.submit([encoder.finish()]);
        self.queued_params.clear();
        self.queued_injections.clear();
    }

    /// Membrane potential and synaptic drive of every neuron, in mV.
    pub fn download_state(&self) -> Vec<[f32; 2]> {
        download(&self.ctx, &self.state)
    }

    /// Spike counts per neuron since the last call, then reset to zero.
    pub fn take_spike_counts(&mut self) -> Vec<u32> {
        let counts = download::<u32>(&self.ctx, &self.spike_counts);
        let mut encoder = self.ctx.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.spike_counts, 0, None);
        self.ctx.queue.submit([encoder.finish()]);
        counts
    }

    fn ensure_params_batch_capacity(&mut self) {
        let needed = (self.queued_params.len() * std::mem::size_of::<Params>()) as u64;
        if self.params_batch.size() < needed {
            self.params_batch = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("params_batch"),
                size: needed.next_power_of_two(),
                usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
    }
}

impl Simulator for GpuSim {
    fn params(&self) -> &LifParams {
        &self.params
    }

    fn neuron_count(&self) -> usize {
        self.neuron_count as usize
    }

    fn backend_name(&self) -> String {
        format!("gpu: {}", self.ctx.adapter_info.name)
    }

    fn inject(&mut self, neuron: u32, delta_g: f32) {
        GpuSim::inject(self, neuron, delta_g);
    }

    fn advance(&mut self) {
        self.queue_step();
    }

    fn flush(&mut self) {
        self.submit();
    }

    fn take_spike_counts(&mut self) -> Vec<u32> {
        self.submit();
        GpuSim::take_spike_counts(self)
    }
}

fn layout_entries() -> [wgpu::BindGroupLayoutEntry; 11] {
    let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let read = wgpu::BufferBindingType::Storage { read_only: true };
    let read_write = wgpu::BufferBindingType::Storage { read_only: false };
    [
        entry(0, wgpu::BufferBindingType::Uniform),
        entry(1, read),
        entry(2, read),
        entry(3, read),
        entry(4, read),
        entry(5, read_write),
        entry(6, read_write),
        entry(7, read_write),
        entry(8, read_write),
        entry(9, read_write),
        entry(10, read),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{TestGpu, context_or_skip};
    use crate::sim::test_net::{ACH, GABA, net};
    use crate::sim::{CpuSim, PoissonDrive};

    /// Run both backends one step at a time with identical injections and
    /// require identical spike sets on every step.
    fn assert_lockstep(
        net: &Connectome,
        steps: usize,
        mut inject: impl FnMut(u64) -> Vec<(u32, f32)>,
    ) {
        let Some(TestGpu { ctx, _serial }) = context_or_skip() else {
            return;
        };
        let params = LifParams::default();
        let mut cpu = CpuSim::new(net, params.clone(), SignPolicy::default());
        let mut gpu = GpuSim::new(ctx, net, params, SignPolicy::default());
        let mut cpu_total = 0;
        for step in 0..steps as u64 {
            for (neuron, mv) in inject(step) {
                cpu.inject(neuron, mv);
                gpu.inject(neuron, mv);
            }
            let cpu_spikes: Vec<u32> = cpu.step().to_vec();
            gpu.queue_step();
            gpu.submit();
            let gpu_spikes: Vec<u32> = gpu
                .take_spike_counts()
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c > 0)
                .map(|(i, _)| i as u32)
                .collect();
            assert_eq!(gpu_spikes, cpu_spikes, "spike sets differ at step {step}");
            cpu_total += cpu_spikes.len();
        }
        assert!(
            cpu_total > 0,
            "test network never spiked; it proves nothing"
        );
        for (i, [v, g]) in gpu.download_state().into_iter().enumerate() {
            let i = i as u32;
            assert!(
                (v - cpu.potential(i)).abs() < 1e-3,
                "neuron {i}: v {v} vs {}",
                cpu.potential(i)
            );
            assert!(
                (g - cpu.drive(i)).abs() < 1e-3,
                "neuron {i}: g {g} vs {}",
                cpu.drive(i)
            );
        }
    }

    #[test]
    fn excitatory_chain_matches_cpu_every_step() {
        let c = net(&[ACH, ACH, ACH], &[(0, 1, 200), (1, 2, 200)]);
        assert_lockstep(
            &c,
            200,
            |step| if step == 0 { vec![(0, 100.0)] } else { vec![] },
        );
    }

    #[test]
    fn balanced_inhibition_matches_cpu_every_step() {
        let c = net(&[ACH, ACH, GABA], &[(0, 1, 200), (2, 1, 200)]);
        assert_lockstep(&c, 200, |step| {
            if step == 0 {
                vec![(0, 100.0), (2, 100.0)]
            } else {
                vec![]
            }
        });
    }

    #[test]
    fn poisson_driven_chain_matches_cpu_every_step() {
        let c = net(
            &[ACH, ACH, GABA, ACH],
            &[(0, 1, 180), (1, 3, 180), (2, 3, 400), (0, 2, 60)],
        );
        let mut drive = PoissonDrive::shiu(vec![0, 2], 300.0, 11);
        let dt = LifParams::default().dt;
        assert_lockstep(&c, 1000, |_| {
            let weight = drive.weight;
            drive.sample(dt).map(|n| (n, weight)).collect()
        });
    }

    #[test]
    fn batched_steps_match_single_steps() {
        let Some(TestGpu { ctx, _serial }) = context_or_skip() else {
            return;
        };
        let c = net(&[ACH, ACH], &[(0, 1, 200)]);
        let params = LifParams::default();
        let mut cpu = CpuSim::new(&c, params.clone(), SignPolicy::default());
        let mut gpu = GpuSim::new(ctx, &c, params, SignPolicy::default());
        cpu.inject(0, 100.0);
        gpu.inject(0, 100.0);
        let steps = 300;
        let mut cpu_counts = vec![0u32; 2];
        for _ in 0..steps {
            for &n in cpu.step() {
                cpu_counts[n as usize] += 1;
            }
            gpu.queue_step();
        }
        gpu.submit();
        assert_eq!(gpu.take_spike_counts(), cpu_counts);
        assert!(cpu_counts[1] > 0);
    }
}
