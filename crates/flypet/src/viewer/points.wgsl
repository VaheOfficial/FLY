// Every brain neuron with a known soma as a small square in its cell-type
// colour, brightening as it spikes. One instance per neuron, six vertices
// per square.

struct View {
    // Maps soma coordinates to clip space: clip = soma * scale + offset.
    scale: vec2<f32>,
    offset: vec2<f32>,
    // Half size of a point in clip units, per axis (accounts for aspect).
    half_size: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
// Soma position in the frontal plane, dataset voxel units.
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
// Resting colour per neuron, linear RGB.
@group(0) @binding(2) var<storage, read> colours: array<vec4<f32>>;
// Recent activity per neuron, 0 (silent) to 1 (just spiked).
@group(0) @binding(3) var<storage, read> activity: array<f32>;

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) colour: vec4<f32>,
}

const CORNERS = array<vec2<f32>, 6>(
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
    vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
);

// How dim a silent neuron is drawn, as a fraction of its full colour.
const RESTING: f32 = 0.3;

@vertex
fn vertex(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VertexOut {
    let centre = positions[i] * view.scale + view.offset;
    let glow = activity[i];
    let base = colours[i].rgb;
    // Silent: a dim version of the type colour. Firing: the full colour
    // pushed towards white so it reads as light, not just as a hue.
    let lit = mix(base * RESTING, mix(base, vec3(1.0), 0.6) * 1.4, glow);
    var out: VertexOut;
    out.clip = vec4(centre + CORNERS[v] * view.half_size, 0.0, 1.0);
    out.colour = vec4(lit, 0.75 + 0.25 * glow);
    return out;
}

@fragment
fn fragment(in: VertexOut) -> @location(0) vec4<f32> {
    return in.colour;
}
