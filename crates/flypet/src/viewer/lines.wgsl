// Neuron morphology as line strips, drawn additively: dense tracts add up
// to bright bundles, and a firing neuron burns through the rest.

struct View {
    scale: vec2<f32>,
    offset: vec2<f32>,
    half_size: vec2<f32>,
    // Dataset voxels per packed position unit.
    voxels_per_unit: f32,
    // Light emitted per silent line vertex, before the region weight.
    resting_light: f32,
    // Depth (packed units, anterior first) over which lines fade.
    depth_near: f32,
    depth_far: f32,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
// Vertex position: x in the low 16 bits of .x, y in the high; z in the low
// 16 bits of .y with the tip fade (0..255) above it.
@group(0) @binding(1) var<storage, read> positions: array<vec2<u32>>;
// Neuron index of each vertex.
@group(0) @binding(2) var<storage, read> vertex_neuron: array<u32>;
// Region colour per neuron (linear RGB) and its resting weight in .a.
@group(0) @binding(3) var<storage, read> colours: array<vec4<f32>>;
// Recent activity per neuron, 0 (silent) to 1 (just spiked).
@group(0) @binding(4) var<storage, read> activity: array<f32>;

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) light: vec3<f32>,
}

// Posterior branches keep this fraction of their light.
const FAR_LIGHT: f32 = 0.25;

@vertex
fn vertex(@builtin(vertex_index) v: u32) -> VertexOut {
    let packed = positions[v];
    let xy = vec2(f32(packed.x & 0xFFFFu), f32(packed.x >> 16u)) * view.voxels_per_unit;
    let z = f32(packed.y & 0xFFFFu);
    let taper = f32(packed.y >> 16u) / 255.0;
    let depth = clamp((z - view.depth_near) / (view.depth_far - view.depth_near), 0.0, 1.0);
    let depth_cue = mix(1.0, FAR_LIGHT, depth) * taper;

    let neuron = vertex_neuron[v];
    let region = colours[neuron];
    let glow = activity[neuron];
    let resting = region.rgb * (view.resting_light * region.a * depth_cue);
    // A spike is white-hot and ignores depth so the pathway stays readable.
    let firing = mix(region.rgb, vec3(1.0), 0.3) * 0.11;

    var out: VertexOut;
    out.clip = vec4(xy * view.scale + view.offset, 0.0, 1.0);
    out.light = mix(resting, firing, glow);
    return out;
}

@fragment
fn fragment(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4(in.light, 1.0);
}
