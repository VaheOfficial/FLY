//! Fibres fade out towards their tips.
//!
//! Where the reconstruction's mask cut through the optic lobes, the cut ends
//! of long axons pile up a micron or two inside the 4 µm chunk faces and,
//! drawn additively from the front, those piles read as walls and tiles.
//! Easing every fibre out over its last stretch turns each wall into a
//! gradient with no edge to catch the eye, and natural tips taper the way
//! they do in any projection of a brain.

use std::collections::{HashMap, HashSet};

use flybrain::morphology::{Morphology, POSITION_UNIT_NM};

/// Length over which a fibre eases out to nothing before a tip. Shorter,
/// or a linear fade, still leaves a visible line where the cut ends start.
const TIP_FADE_NM: f32 = 24_000.0;
/// Side of the chunks the reconstruction was computed in, in packed units
/// (4096 nm).
const CHUNK: u16 = 256;

/// Light factor per vertex, 0 to 255.
pub fn tip_fade(morphology: &Morphology) -> Vec<u8> {
    let tips = tips_of(morphology);
    let positions = &morphology.positions;
    let fade_units = TIP_FADE_NM / POSITION_UNIT_NM;
    let mut fade = vec![u8::MAX; morphology.vertex_count()];
    for strip in &morphology.strips {
        let first = strip.first as usize;
        let last = first + strip.len as usize - 1;
        if tips.contains(&positions[first]) {
            fade_from_tip(positions, first..=last, fade_units, &mut fade);
        }
        if tips.contains(&positions[last]) {
            fade_from_tip(positions, (first..=last).rev(), fade_units, &mut fade);
        }
    }
    fade
}

/// Walk inward from a tip, easing vertices in by their arc length from it.
fn fade_from_tip(
    positions: &[[u16; 3]],
    inward: impl Iterator<Item = usize>,
    fade_units: f32,
    fade: &mut [u8],
) {
    let mut distance = 0.0f32;
    let mut previous: Option<[u16; 3]> = None;
    for v in inward {
        let p = positions[v];
        if let Some(q) = previous {
            distance += length(p, q);
        }
        if distance >= fade_units {
            break;
        }
        let t = distance / fade_units;
        let eased = t * t * (3.0 - 2.0 * t);
        fade[v] = fade[v].min((eased * 255.0).round() as u8);
        previous = Some(p);
    }
}

/// Positions where exactly one strip ends: the fibres' tips. An end shared
/// with another strip is a branch point, and an end lying on a chunk plane
/// was cut there by the packer, not by the reconstruction.
fn tips_of(morphology: &Morphology) -> HashSet<[u16; 3]> {
    let mut ends: HashMap<[u16; 3], u8> = HashMap::new();
    for strip in &morphology.strips {
        let first = strip.first as usize;
        for v in [first, first + strip.len as usize - 1] {
            let count = ends.entry(morphology.positions[v]).or_default();
            *count = count.saturating_add(1);
        }
    }
    ends.into_iter()
        .filter(|&(p, count)| count == 1 && !p.iter().any(|&v| v % CHUNK == 0))
        .map(|(p, _)| p)
        .collect()
}

fn length(a: [u16; 3], b: [u16; 3]) -> f32 {
    let d = |i: usize| f32::from(a[i]) - f32::from(b[i]);
    (d(0) * d(0) + d(1) * d(1) + d(2) * d(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use flybrain::morphology::Strip;

    /// Vertices 500 units (8 µm) apart along x from `x0`, one strip.
    fn strip_along_x(x0: u16, count: u32) -> Morphology {
        let positions = (0..count as u16)
            .map(|i| [x0 + i * 500, 300, 300])
            .collect();
        Morphology {
            neuron_count: 1,
            vertex_neuron: vec![0; count as usize],
            positions,
            strips: vec![Strip {
                first: 0,
                len: count,
            }],
        }
    }

    #[test]
    fn both_tips_of_a_lone_strip_ease_in_over_the_fade_length() {
        // Fade length is 1500 units: vertices a third and two thirds of the
        // way in sit on the smoothstep, the one at 1500 is out of reach.
        let fade = tip_fade(&strip_along_x(100, 7));
        assert_eq!(fade, [0, 66, 189, 255, 189, 66, 0]);
    }

    #[test]
    fn a_branch_point_is_not_a_tip() {
        let mut m = strip_along_x(100, 5);
        // A second strip starting where the first ends, long enough that
        // its own tip is out of reach of the junction.
        let junction = m.positions[4];
        let x = junction[0];
        m.positions
            .extend([junction, [x, 900, 300], [x, 1500, 300], [x, 2500, 300]]);
        m.vertex_neuron.extend([0, 0, 0, 0]);
        m.strips.push(Strip { first: 5, len: 4 });
        let fade = tip_fade(&m);
        assert_eq!(fade[4], 255, "junction keeps full light");
        assert_eq!(fade[5], 255);
        assert_eq!((fade[0], fade[8]), (0, 0), "the free ends still fade");
    }

    #[test]
    fn an_end_on_a_chunk_plane_is_not_a_tip() {
        let fade = tip_fade(&strip_along_x(CHUNK * 2, 5));
        assert_eq!(fade[0], 255, "cut by the packer, not a tip");
        assert_eq!(fade[4], 0);
    }
}
