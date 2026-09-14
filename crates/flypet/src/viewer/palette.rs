//! Colours that make the brain legible: one hue per functional region, so
//! the mushroom body, central complex, antennal lobe, and the sensory,
//! descending and visual systems each read as a block, while the two big
//! undifferentiated masses (unclassified central-brain neurons and the
//! optic-lobe columns) stay muted. A small per-type hue shift keeps
//! neighbouring neurons of a region distinguishable.
//!
//! The alpha channel is not opacity but a weight: how much light a silent
//! neuron of that region contributes, so the optic lobes can be held back.

use flybrain::connectome::{Connectome, NO_STRING, Neuron};

/// Linear RGB plus contribution weight for a neuron at rest.
pub fn for_neuron(net: &Connectome, neuron: &Neuron) -> [f32; 4] {
    let class = net.string(neuron.class).unwrap_or("");
    let superclass = net.string(neuron.superclass).unwrap_or("");
    let (srgb, weight) = region_colour(class, superclass);
    let [r, g, b] = shift_hue(srgb, type_shift(neuron.type_name));
    [
        srgb_to_linear(r),
        srgb_to_linear(g),
        srgb_to_linear(b),
        weight,
    ]
}

/// Display-space colour and weight for a region, by class first, then
/// superclass.
fn region_colour(class: &str, superclass: &str) -> ([f32; 3], f32) {
    match class {
        "Kenyon_Cell" => ([1.0, 0.80, 0.20], 1.0),
        "MBON" => ([1.0, 0.55, 0.15], 1.2),
        "DAN" => ([1.0, 0.35, 0.30], 1.2),
        "CX" => ([0.90, 0.35, 0.95], 1.0),
        "olfactory" => ([0.35, 0.90, 0.45], 1.0),
        "ALPN" | "ALON" => ([0.55, 1.0, 0.55], 1.1),
        "ALLN" | "ALIN" => ([0.25, 0.70, 0.35], 1.0),
        "gustatory" => ([0.75, 0.95, 0.25], 1.1),
        "hygrosensory" | "thermosensory" | "chemosensory" => ([0.40, 0.95, 0.85], 1.1),
        c if c.starts_with("mechanosensory") => ([0.90, 0.90, 0.35], 1.0),
        "visual" | "ol_bilateral" => ([0.30, 0.75, 0.95], 0.9),
        "SEZPN" => ([0.80, 0.60, 1.0], 1.0),
        _ => match superclass {
            "ol_intrinsic" => ([0.30, 0.36, 0.48], 0.35),
            "ol_sensory" => ([0.28, 0.30, 0.40], 0.3),
            "visual_projection" => ([0.25, 0.80, 0.95], 0.8),
            "visual_centrifugal" => ([0.25, 0.70, 0.65], 0.9),
            "descending_neuron" | "sensory_descending" => ([0.90, 0.93, 1.0], 0.8),
            "ascending_neuron" | "sensory_ascending" => ([0.70, 0.78, 1.0], 1.0),
            "cb_sensory" => ([0.65, 0.90, 0.55], 1.0),
            "cb_motor" | "cb_endocrine" | "cb_efferent" => ([1.0, 0.60, 0.65], 1.2),
            "cb_intrinsic" => ([0.48, 0.44, 0.62], 0.55),
            _ => ([0.50, 0.50, 0.55], 0.6),
        },
    }
}

/// Hue shift in turns, within ±0.035, derived from the type index.
fn type_shift(type_index: u32) -> f32 {
    if type_index == NO_STRING {
        return 0.0;
    }
    let mixed = type_index.wrapping_mul(0x9E37_79B9) >> 8;
    (mixed as f32 / (1u32 << 24) as f32 - 0.5) * 0.07
}

fn shift_hue([r, g, b]: [f32; 3], turns: f32) -> [f32; 3] {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    if delta < 1e-5 {
        return [r, g, b];
    }
    let hue = if max == r {
        ((g - b) / delta).rem_euclid(6.0)
    } else if max == g {
        (b - r) / delta + 2.0
    } else {
        (r - g) / delta + 4.0
    } / 6.0;
    let h = (hue + turns).rem_euclid(1.0) * 6.0;
    let c = delta;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r1, g1, b1) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r1 + min, g1 + min, b1 + min]
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hue_shift_preserves_brightness_and_is_small() {
        let base = [1.0, 0.80, 0.20];
        let shifted = shift_hue(base, 0.03);
        let max = |c: [f32; 3]| c[0].max(c[1]).max(c[2]);
        let min = |c: [f32; 3]| c[0].min(c[1]).min(c[2]);
        assert!((max(shifted) - max(base)).abs() < 1e-5);
        assert!((min(shifted) - min(base)).abs() < 1e-5);
        assert!(shifted != base);
        assert_eq!(
            shift_hue([0.5, 0.5, 0.5], 0.3),
            [0.5, 0.5, 0.5],
            "grey has no hue"
        );
    }

    #[test]
    fn regions_are_distinct_and_optic_lobes_are_held_back() {
        let (kc, _) = region_colour("Kenyon_Cell", "cb_intrinsic");
        let (cx, _) = region_colour("CX", "cb_intrinsic");
        assert_ne!(kc, cx);
        let (_, ol_weight) = region_colour("", "ol_intrinsic");
        let (_, dn_weight) = region_colour("", "descending_neuron");
        assert!(ol_weight < 0.5 && dn_weight > ol_weight);
    }

    #[test]
    fn type_shift_is_bounded_and_zero_for_untyped() {
        assert_eq!(type_shift(NO_STRING), 0.0);
        for t in [0, 1, 7, 4000, 11_000] {
            assert!(type_shift(t).abs() <= 0.035);
        }
    }
}
