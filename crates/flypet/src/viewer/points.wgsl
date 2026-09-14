// Cell bodies as small squares, drawn additively on top of the morphology.
// One instance per neuron, six vertices per square.

struct View {
    // Maps soma coordinates to clip space: clip = soma * scale + offset.
    scale: vec2<f32>,
    offset: vec2<f32>,
    // Half size of a point in clip units, per axis (accounts for aspect).
    half_size: vec2<f32>,
    voxels_per_unit: f32,
    resting_light: f32,
    depth_near: f32,
    depth_far: f32,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
// Soma position in dataset voxels: x, y in the frontal plane, z depth.
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
// Neuron index of each point.
@group(0) @binding(2) var<storage, read> point_neuron: array<u32>;
// Region colour per neuron (linear RGB) and its resting weight in .a.
@group(0) @binding(3) var<storage, read> colours: array<vec4<f32>>;
// Recent activity per neuron, 0 (silent) to 1 (just spiked).
@group(0) @binding(4) var<storage, read> activity: array<f32>;

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) light: vec3<f32>,
}

const CORNERS = array<vec2<f32>, 6>(
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
    vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
);

// A soma is a small dot next to its neuron's whole tree: keep it modest.
const SOMA_LIGHT: f32 = 0.9;
const FAR_LIGHT: f32 = 0.25;

@vertex
fn vertex(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VertexOut {
    let p = positions[i];
    let depth_units = p.z / view.voxels_per_unit;
    let depth = clamp((depth_units - view.depth_near) / (view.depth_far - view.depth_near), 0.0, 1.0);
    let depth_cue = mix(1.0, FAR_LIGHT, depth);

    let neuron = point_neuron[i];
    let region = colours[neuron];
    let glow = activity[neuron];
    let resting = region.rgb * (SOMA_LIGHT * view.resting_light * region.a * depth_cue);
    let firing = mix(region.rgb, vec3(1.0), 0.5) * 0.35;

    var out: VertexOut;
    out.clip = vec4(p.xy * view.scale + view.offset + CORNERS[v] * view.half_size, 0.0, 1.0);
    out.light = mix(resting, firing, glow);
    return out;
}

@fragment
fn fragment(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4(in.light, 1.0);
}
