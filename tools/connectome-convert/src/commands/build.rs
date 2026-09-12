//! `build`: join the tables and write the packed connectome file.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use flybrain::connectome;

use crate::assemble;
use crate::edges;
use crate::malecns::{ANNOTATIONS, TRANSMITTERS, WEIGHTS};
use crate::neurons::NeuronTable;
use crate::transmitters;

pub fn run(data: &Path, out: &Path) -> Result<()> {
    let t0 = Instant::now();
    let neurons = NeuronTable::load(&data.join(ANNOTATIONS))?;
    println!("neurons: {}", neurons.len());
    println!(
        "  with soma position: {}",
        neurons.neurons.iter().filter(|n| n.soma.is_some()).count()
    );

    let (nt, nt_stats) = transmitters::load(&data.join(TRANSMITTERS), &neurons)?;
    println!(
        "transmitters: {} rows, {} matched neurons, {} neurons without a row",
        nt_stats.rows_seen, nt_stats.matched, nt_stats.neurons_without_row
    );
    println!(
        "  consensus unclear: {}   ground truth known: {}",
        nt_stats.consensus_unclear, nt_stats.with_ground_truth
    );

    let (edges, edge_stats) = edges::load(&data.join(WEIGHTS), &neurons)?;
    println!(
        "edges: {} kept of {} rows, {} synapses",
        edge_stats.kept, edge_stats.rows_seen, edge_stats.kept_weight
    );

    let connectome = assemble::connectome(&neurons, &nt, edges)?;
    println!(
        "assembled: {} neurons, {} edges, {} strings  ({:.1?})",
        connectome.neuron_count(),
        connectome.edge_count(),
        connectome.strings.len(),
        t0.elapsed()
    );

    let file = File::create(out).with_context(|| format!("creating {}", out.display()))?;
    connectome::io::write(&connectome, BufWriter::new(file))
        .with_context(|| format!("writing {}", out.display()))?;
    let bytes = std::fs::metadata(out)?.len();
    println!(
        "wrote {} ({:.1} MB, {:.1?} total)",
        out.display(),
        bytes as f64 / 1e6,
        t0.elapsed()
    );
    Ok(())
}
