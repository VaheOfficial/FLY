//! `build-skeletons`: simplify every fetched skeleton and write the packed
//! morphology file, with vertices tagged by dense neuron index.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use flybrain::connectome::io;
use flybrain::morphology::{self, Morphology, Strip};

use crate::skeleton_pack::read_all;
use crate::skeleton_simplify::{parse_precomputed, simplify, split_at_chunk_planes};

/// The published skeletons were computed in chunks this size; segments lying
/// flat in a chunk face are artifacts and are removed.
const SKELETON_CHUNK_NM: f32 = 4096.0;
const CHUNK_PLANE_TOLERANCE_NM: f32 = 1.0;

pub fn run(
    pack: &Path,
    connectome: &Path,
    spacing_nm: f32,
    prune_nm: f32,
    out: &Path,
) -> Result<()> {
    let started = Instant::now();
    let net = io::read(BufReader::new(
        File::open(connectome).with_context(|| format!("opening {}", connectome.display()))?,
    ))
    .with_context(|| format!("reading {}", connectome.display()))?;
    let index_of_body: HashMap<i64, u32> = net
        .neurons
        .iter()
        .enumerate()
        .map(|(i, n)| (n.body_id, i as u32))
        .collect();

    let mut m = Morphology {
        neuron_count: net.neuron_count() as u32,
        ..Default::default()
    };
    let mut neurons_with_skeleton = 0usize;
    let mut unknown_bodies = 0usize;
    let mut missing = 0usize;
    let mut raw_vertices = 0u64;
    read_all(pack, |body_id, bytes| {
        if bytes.is_empty() {
            missing += 1;
            return Ok(());
        }
        let Some(&neuron) = index_of_body.get(&(body_id as i64)) else {
            unknown_bodies += 1;
            return Ok(());
        };
        let skeleton =
            parse_precomputed(bytes).with_context(|| format!("skeleton of body {body_id}"))?;
        raw_vertices += skeleton.vertices.len() as u64;
        let paths = split_at_chunk_planes(
            simplify(&skeleton, spacing_nm, prune_nm),
            SKELETON_CHUNK_NM,
            CHUNK_PLANE_TOLERANCE_NM,
        );
        for path in paths {
            m.strips.push(Strip {
                first: m.positions.len() as u32,
                len: path.len() as u32,
            });
            for p in path {
                m.positions.push(Morphology::quantise(p));
                m.vertex_neuron.push(neuron);
            }
        }
        neurons_with_skeleton += 1;
        if neurons_with_skeleton.is_multiple_of(20_000) {
            println!(
                "{neurons_with_skeleton} neurons, {} vertices so far ({:.0?})",
                m.positions.len(),
                started.elapsed()
            );
        }
        Ok(())
    })?;
    m.validate()
        .map_err(|why| anyhow::anyhow!("built morphology is invalid: {why}"))?;

    let file = File::create(out).with_context(|| format!("creating {}", out.display()))?;
    morphology::write(&m, BufWriter::new(file))
        .with_context(|| format!("writing {}", out.display()))?;
    let bytes = std::fs::metadata(out)?.len();
    println!(
        "{neurons_with_skeleton} neurons with skeletons ({missing} without, {unknown_bodies} not in connectome)"
    );
    println!(
        "{} raw vertices -> {} kept in {} strips at {spacing_nm} nm spacing, twigs under {prune_nm} nm pruned",
        raw_vertices,
        m.vertex_count(),
        m.strips.len()
    );
    println!(
        "wrote {} ({:.1} MB, {:.0?})",
        out.display(),
        bytes as f64 / 1e6,
        started.elapsed()
    );
    Ok(())
}
