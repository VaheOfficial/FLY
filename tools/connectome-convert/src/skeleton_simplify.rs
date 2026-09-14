//! Turn a neuroglancer precomputed skeleton into a few polylines with
//! vertices spaced at least `spacing_nm` apart.
//!
//! The skeleton is a tree (occasionally several). It is cut into paths at
//! branch points and leaves; terminal paths shorter than `prune_nm` are
//! dropped, since automated skeletons sprout many tiny twigs that add
//! nothing to the picture; and each surviving path is decimated by walking
//! it and keeping a vertex only once the arc length since the last kept
//! vertex exceeds the spacing. Endpoints are always kept, so branch points
//! stay connected and the overall shape survives.
//!
//! The published skeletons were computed in axis-aligned chunks, and where
//! a branch hugs a chunk face the skeleton runs flat along that plane. Such
//! segments line up across thousands of neurons and draw as a visible grid,
//! so [`split_at_chunk_planes`] removes any segment lying in one.

use std::collections::HashSet;

use anyhow::{Result, bail};

/// Vertices in nanometres and undirected edges as vertex index pairs.
pub struct RawSkeleton {
    pub vertices: Vec<[f32; 3]>,
    pub edges: Vec<[u32; 2]>,
}

/// Parse the precomputed binary: `n_vertices u32, n_edges u32, vertices
/// f32x3 each, edges u32x2 each`.
pub fn parse_precomputed(bytes: &[u8]) -> Result<RawSkeleton> {
    if bytes.len() < 8 {
        bail!("skeleton shorter than its header");
    }
    let n_vertices = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let n_edges = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let needed = 8 + n_vertices * 12 + n_edges * 8;
    if bytes.len() < needed {
        bail!(
            "skeleton declares {n_vertices} vertices and {n_edges} edges but has {} bytes",
            bytes.len()
        );
    }
    let f32_at = |i: usize| f32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
    let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
    let vertices = (0..n_vertices)
        .map(|v| {
            let at = 8 + v * 12;
            [f32_at(at), f32_at(at + 4), f32_at(at + 8)]
        })
        .collect();
    let edge_base = 8 + n_vertices * 12;
    let edges = (0..n_edges)
        .map(|e| {
            let at = edge_base + e * 8;
            [u32_at(at), u32_at(at + 4)]
        })
        .collect::<Vec<_>>();
    if edges.iter().flatten().any(|&v| v as usize >= n_vertices) {
        bail!("skeleton edge refers to a vertex out of range");
    }
    Ok(RawSkeleton { vertices, edges })
}

/// Decimated polylines, each a list of nanometre positions.
pub fn simplify(skeleton: &RawSkeleton, spacing_nm: f32, prune_nm: f32) -> Vec<Vec<[f32; 3]>> {
    let n = skeleton.vertices.len();
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &[a, b] in &skeleton.edges {
        adjacency[a as usize].push(b);
        adjacency[b as usize].push(a);
    }
    // Pruning a twig can turn its branch point into a pass-through vertex
    // and expose a new, longer twig behind it, so prune until nothing
    // short is left. Each round removes the twig's vertices from the graph.
    loop {
        let twigs: Vec<Vec<u32>> = paths_of(&adjacency)
            .into_iter()
            .filter(|path| {
                let leaf_start = adjacency[path[0] as usize].len() == 1;
                let leaf_end = adjacency[*path.last().unwrap() as usize].len() == 1;
                (leaf_start || leaf_end) && arc_length(path, &skeleton.vertices) < prune_nm
            })
            .collect();
        if twigs.is_empty() {
            break;
        }
        for twig in twigs {
            // Keep the end that is a branch point; drop the rest.
            let keep_first = adjacency[twig[0] as usize].len() > 1;
            let doomed = if keep_first {
                &twig[1..]
            } else {
                &twig[..twig.len() - 1]
            };
            for &v in doomed {
                for neighbour in std::mem::take(&mut adjacency[v as usize]) {
                    adjacency[neighbour as usize].retain(|&u| u != v);
                }
            }
        }
    }
    paths_of(&adjacency)
        .iter()
        .map(|path| decimate(path, &skeleton.vertices, spacing_nm))
        .collect()
}

/// Cut the graph into paths at branch points and leaves; each undirected
/// edge appears in exactly one path.
fn paths_of(adjacency: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let n = adjacency.len();
    let mut used_edges: HashSet<(u32, u32)> = HashSet::new();
    let mut paths = Vec::new();
    let mut trace_from = |start: u32| {
        for &neighbour in &adjacency[start as usize] {
            if !used_edges.insert(edge_key(start, neighbour)) {
                continue;
            }
            let mut path = vec![start, neighbour];
            let (mut previous, mut current) = (start, neighbour);
            // Follow the chain through degree-2 vertices until a branch,
            // a leaf, or an edge already walked (a cycle closing).
            while adjacency[current as usize].len() == 2 {
                let next = adjacency[current as usize]
                    .iter()
                    .copied()
                    .find(|&v| v != previous)
                    .unwrap_or(previous);
                if !used_edges.insert(edge_key(current, next)) {
                    break;
                }
                path.push(next);
                previous = current;
                current = next;
            }
            paths.push(path);
        }
    };
    for v in 0..n as u32 {
        if adjacency[v as usize].len() != 2 {
            trace_from(v);
        }
    }
    // Pure cycles have no branch or leaf; start anywhere on them.
    for v in 0..n as u32 {
        if adjacency[v as usize].len() == 2 {
            trace_from(v);
        }
    }
    paths
}

/// Cut every segment whose two ends share a coordinate that is a multiple of
/// `chunk_nm` (within `tolerance_nm`): a skeletonisation artifact, not
/// anatomy. Pieces left with a single vertex are dropped.
pub fn split_at_chunk_planes(
    paths: Vec<Vec<[f32; 3]>>,
    chunk_nm: f32,
    tolerance_nm: f32,
) -> Vec<Vec<[f32; 3]>> {
    if chunk_nm <= 0.0 {
        return paths;
    }
    // Plane index 0 is the volume origin, outside any tissue; a coordinate of
    // exactly zero is data, not a chunk face.
    let on_plane = |v: f32| {
        let index = (v / chunk_nm).round();
        index >= 1.0 && (v - index * chunk_nm).abs() <= tolerance_nm
    };
    let flat = |a: [f32; 3], b: [f32; 3]| {
        (0..3).any(|axis| (a[axis] - b[axis]).abs() <= tolerance_nm && on_plane(a[axis]))
    };
    let mut out = Vec::with_capacity(paths.len());
    for path in paths {
        let mut piece: Vec<[f32; 3]> = Vec::new();
        for pair in path.windows(2) {
            if piece.is_empty() {
                piece.push(pair[0]);
            }
            if flat(pair[0], pair[1]) {
                if piece.len() >= 2 {
                    out.push(std::mem::take(&mut piece));
                } else {
                    piece.clear();
                }
            } else {
                piece.push(pair[1]);
            }
        }
        if piece.len() >= 2 {
            out.push(piece);
        }
    }
    out
}

fn edge_key(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

fn decimate(path: &[u32], vertices: &[[f32; 3]], spacing_nm: f32) -> Vec<[f32; 3]> {
    let mut kept = vec![vertices[path[0] as usize]];
    let mut since_kept = 0.0f32;
    for pair in path.windows(2) {
        let (a, b) = (vertices[pair[0] as usize], vertices[pair[1] as usize]);
        since_kept += distance(a, b);
        if since_kept >= spacing_nm {
            kept.push(b);
            since_kept = 0.0;
        }
    }
    let last = vertices[*path.last().unwrap() as usize];
    // A closed loop returns to its start; always keep at least a segment.
    if kept.last() != Some(&last) || kept.len() < 2 {
        kept.push(last);
    }
    kept
}

fn arc_length(path: &[u32], vertices: &[[f32; 3]]) -> f32 {
    path.windows(2)
        .map(|pair| distance(vertices[pair[0] as usize], vertices[pair[1] as usize]))
        .sum()
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(vertices: &[[f32; 3]], edges: &[[u32; 2]]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&(vertices.len() as u32).to_le_bytes());
        b.extend_from_slice(&(edges.len() as u32).to_le_bytes());
        for v in vertices {
            for c in v {
                b.extend_from_slice(&c.to_le_bytes());
            }
        }
        for e in edges {
            for i in e {
                b.extend_from_slice(&i.to_le_bytes());
            }
        }
        b
    }

    /// A Y: a chain 0-1-2-3 with a branch 2-4-5.
    fn y_shape() -> RawSkeleton {
        let vertices = [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [200.0, 0.0, 0.0],
            [300.0, 0.0, 0.0],
            [200.0, 100.0, 0.0],
            [200.0, 200.0, 0.0],
        ];
        parse_precomputed(&encode(
            &vertices,
            &[[0, 1], [1, 2], [2, 3], [2, 4], [4, 5]],
        ))
        .unwrap()
    }

    #[test]
    fn parses_and_rejects_bad_edges() {
        let s = y_shape();
        assert_eq!(s.vertices.len(), 6);
        assert_eq!(s.edges.len(), 5);
        assert!(parse_precomputed(&encode(&[[0.0; 3]], &[[0, 7]])).is_err());
        assert!(parse_precomputed(&[1, 2, 3]).is_err());
    }

    #[test]
    fn y_becomes_three_paths_meeting_at_the_branch() {
        let paths = simplify(&y_shape(), 1.0, 0.0);
        assert_eq!(paths.len(), 3, "{paths:?}");
        let branch = [200.0, 0.0, 0.0];
        assert!(
            paths
                .iter()
                .all(|p| p.first() == Some(&branch) || p.last() == Some(&branch))
        );
        let total: usize = paths.iter().map(|p| p.len()).sum();
        assert_eq!(
            total,
            3 + 2 + 3,
            "no interior vertex dropped at 1 nm spacing"
        );
    }

    #[test]
    fn coarse_spacing_keeps_only_endpoints() {
        let paths = simplify(&y_shape(), 10_000.0, 0.0);
        assert_eq!(paths.len(), 3);
        assert!(paths.iter().all(|p| p.len() == 2), "{paths:?}");
    }

    #[test]
    fn short_twigs_are_pruned_and_the_rest_merges_into_one_path() {
        // The Y's arms: 0-1-2 is 200 long, 2-3 is 100, 2-4-5 is 200.
        // Pruning 2-3 leaves vertex 2 as a pass-through, so the two long
        // arms become a single path 0-1-2-4-5.
        let paths = simplify(&y_shape(), 1.0, 150.0);
        assert_eq!(paths.len(), 1, "{paths:?}");
        assert_eq!(paths[0].len(), 5);
    }

    #[test]
    fn pruning_repeats_until_nothing_short_remains() {
        // A trunk 0-1 (1000 long) with a 50-long twig at 1 and, behind it,
        // a 140-long twig 1-3-4. Pruning under 150 must remove both even
        // though the second only becomes a leaf twig after the first goes.
        let vertices = [
            [0.0, 0.0, 0.0],
            [1000.0, 0.0, 0.0],
            [1000.0, 50.0, 0.0],
            [1120.0, 0.0, 0.0],
            [1120.0, 20.0, 0.0],
        ];
        let s = parse_precomputed(&encode(&vertices, &[[0, 1], [1, 2], [1, 3], [3, 4]])).unwrap();
        let paths = simplify(&s, 1.0, 150.0);
        assert_eq!(paths.len(), 1, "{paths:?}");
        assert_eq!(paths[0], vec![[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]]);
    }

    #[test]
    fn segments_flat_in_a_chunk_plane_are_cut_out() {
        // Second and third points share x = 4096 exactly: that segment goes,
        // leaving the two halves of the path.
        let path = vec![
            [1000.0, 0.0, 0.0],
            [4096.0, 500.0, 0.0],
            [4096.0, 900.0, 0.0],
            [6000.0, 900.0, 0.0],
        ];
        let pieces = split_at_chunk_planes(vec![path.clone()], 4096.0, 1.0);
        assert_eq!(pieces, vec![path[..2].to_vec(), path[2..].to_vec()]);
        assert_eq!(
            split_at_chunk_planes(vec![path.clone()], 0.0, 1.0),
            vec![path.clone()]
        );
        // A segment merely crossing the plane is kept.
        let crossing = vec![[4000.0, 0.0, 0.0], [4200.0, 0.0, 0.0]];
        assert_eq!(
            split_at_chunk_planes(vec![crossing.clone()], 4096.0, 1.0),
            vec![crossing]
        );
    }

    #[test]
    fn isolated_vertices_and_cycles_are_handled() {
        let square = parse_precomputed(&encode(
            &[
                [0.0; 3],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
                [99.0; 3],
            ],
            &[[0, 1], [1, 2], [2, 3], [3, 0]],
        ))
        .unwrap();
        let paths = simplify(&square, 1.0, 0.0);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].len(), 5, "cycle returns to its start");
    }
}
