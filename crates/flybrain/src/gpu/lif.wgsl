// Leaky integrate-and-fire step over the connectome. Mirrors
// `crate::sim::CpuSim` exactly; see that module for the equations.
//
// Synaptic increments are accumulated in fixed point (`pending_drive`) so
// that atomic adds from many presynaptic neurons commute exactly.

struct Params {
    v_rest: f32,
    v_thresh: f32,
    v_reset: f32,
    decay_m: f32,
    decay_syn: f32,
    coupling: f32,
    // Fixed-point units per mV for `pending_drive`.
    fixed_scale: f32,
    neuron_count: u32,
    refractory_steps: u32,
    step: u32,
    // Ring slot whose spikes are delivered this step and then refilled.
    slot: u32,
    inject_start: u32,
    inject_count: u32,
    // One slot of the spike ring: a count followed by `neuron_count` ids.
    ring_stride: u32,
    _pad0: u32,
    _pad1: u32,
}

struct Injection {
    neuron: u32,
    drive_fixed: i32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(2) var<storage, read> post: array<u32>;
@group(0) @binding(3) var<storage, read> weight_pairs: array<u32>;
// Per presynaptic neuron: sign * w_syn * fixed_scale.
@group(0) @binding(4) var<storage, read> drive_scale: array<f32>;
// Per neuron: (membrane potential v, synaptic drive g), both mV.
@group(0) @binding(5) var<storage, read_write> state: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> refractory_until: array<u32>;
@group(0) @binding(7) var<storage, read_write> pending_drive: array<atomic<i32>>;
@group(0) @binding(8) var<storage, read_write> spike_ring: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> spike_counts: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read> injections: array<Injection>;

fn synapse_count(edge: u32) -> u32 {
    return (weight_pairs[edge >> 1u] >> ((edge & 1u) * 16u)) & 0xFFFFu;
}

// External drive for this step, queued by the host.
@compute @workgroup_size(64)
fn inject(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.inject_count {
        return;
    }
    let entry = injections[params.inject_start + gid.x];
    atomicAdd(&pending_drive[entry.neuron], entry.drive_fixed);
}

// One workgroup per spike due this step; threads stride across its row.
@compute @workgroup_size(64)
fn propagate(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let base = params.slot * params.ring_stride;
    let pre = atomicLoad(&spike_ring[base + 1u + group.x]);
    let scale = drive_scale[pre];
    let begin = row_ptr[pre];
    let end = row_ptr[pre + 1u];
    for (var edge = begin + local.x; edge < end; edge += 64u) {
        let increment = i32(round(scale * f32(synapse_count(edge))));
        atomicAdd(&pending_drive[post[edge]], increment);
    }
}

// Fold in pending drive, integrate one step, detect spikes.
@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.neuron_count {
        return;
    }
    let drive_fixed = atomicExchange(&pending_drive[i], 0);
    var s = state[i];
    s.y += f32(drive_fixed) / params.fixed_scale;

    // Refractory: drive accumulates but nothing integrates or decays.
    if refractory_until[i] > params.step {
        state[i] = s;
        return;
    }

    let u = s.x - params.v_rest;
    s.x = params.v_rest + u * params.decay_m + s.y * params.coupling;
    s.y *= params.decay_syn;
    if s.x > params.v_thresh {
        s.x = params.v_reset;
        refractory_until[i] = params.step + 1u + params.refractory_steps;
        let base = params.slot * params.ring_stride;
        let index = atomicAdd(&spike_ring[base], 1u);
        atomicStore(&spike_ring[base + 1u + index], i);
        atomicAdd(&spike_counts[i], 1u);
    }
    state[i] = s;
}
