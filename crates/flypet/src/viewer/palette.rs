//! One colour per cell type, the way connectome renderings are usually
//! coloured: a well-spread hue per type so neighbouring types differ.

use flybrain::connectome::NO_STRING;

/// Resting colour for a cell type, as linear premultiplied-ready RGBA.
/// Untyped neurons get a neutral grey.
pub fn for_type(type_index: u32) -> [f32; 4] {
    if type_index == NO_STRING {
        return [0.35, 0.35, 0.4, 1.0];
    }
    // Golden-ratio hue stepping spreads consecutive type indices around the
    // wheel; the hash on top decorrelates neighbouring indices further.
    let mixed = type_index.wrapping_mul(0x9E37_79B9);
    let hue = (mixed >> 8) as f32 / (1u32 << 24) as f32;
    let saturation = 0.65 + 0.3 * ((mixed & 0xFF) as f32 / 255.0);
    let [r, g, b] = hsv_to_linear_rgb(hue, saturation, 1.0);
    [r, g, b, 1.0]
}

fn hsv_to_linear_rgb(hue: f32, saturation: f32, value: f32) -> [f32; 3] {
    let h = (hue.fract() + 1.0).fract() * 6.0;
    let c = value * saturation;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let m = value - c;
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    // Treat the HSV result as display (sRGB) intent and linearise it so the
    // sRGB surface shows the intended hue.
    [
        srgb_to_linear(r + m),
        srgb_to_linear(g + m),
        srgb_to_linear(b + m),
    ]
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
    fn colours_are_in_range_and_types_differ() {
        let a = for_type(0);
        let b = for_type(1);
        assert_ne!(a, b);
        for t in [0, 1, 2, 500, 11_000] {
            for c in for_type(t) {
                assert!((0.0..=1.0).contains(&c));
            }
        }
        assert_eq!(for_type(NO_STRING), [0.35, 0.35, 0.4, 1.0]);
    }

    #[test]
    fn hsv_primaries() {
        let [r, g, b] = hsv_to_linear_rgb(0.0, 1.0, 1.0);
        assert!(r > 0.99 && g < 0.01 && b < 0.01);
        let [r, g, b] = hsv_to_linear_rgb(1.0 / 3.0, 1.0, 1.0);
        assert!(g > 0.99 && r < 0.01 && b < 0.01);
    }
}
