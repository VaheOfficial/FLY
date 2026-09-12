//! `summarize`: neuron-set and edge-filter totals for comparison with the paper.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use anyhow::Result;

use crate::edges;
use crate::malecns::{ANNOTATIONS, WEIGHTS};
use crate::neurons::{self, NeuronTable};

pub fn run(data: &Path) -> Result<()> {
    let t0 = Instant::now();
    let neurons = NeuronTable::load(&data.join(ANNOTATIONS))?;
    println!("neurons: {}  ({:.1?})", neurons.len(), t0.elapsed());
    let count =
        |pred: fn(&neurons::Neuron) -> bool| neurons.neurons.iter().filter(|n| pred(n)).count();
    println!("  with type: {}", count(|n| n.type_name.is_some()));
    println!("  with instance: {}", count(|n| n.instance.is_some()));
    println!("  with class: {}", count(|n| n.class.is_some()));
    println!("  with soma side: {}", count(|n| n.soma_side.is_some()));
    println!("  with soma position: {}", count(|n| n.soma.is_some()));
    println!(
        "  status Traced: {}",
        count(|n| n.status.as_deref() == Some("Traced"))
    );
    let mut by_superclass: HashMap<&str, usize> = HashMap::new();
    for n in &neurons.neurons {
        *by_superclass.entry(n.superclass.as_str()).or_default() += 1;
    }
    let mut by_superclass: Vec<_> = by_superclass.into_iter().collect();
    by_superclass.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    println!("  by superclass:");
    for (superclass, n) in by_superclass {
        println!("    {n:>8}  {superclass}");
    }

    let t1 = Instant::now();
    let (edges, stats) = edges::load(&data.join(WEIGHTS), &neurons)?;
    println!(
        "weights table: {} rows  ({:.1?})",
        stats.rows_seen,
        t1.elapsed()
    );
    println!(
        "  kept neuron->neuron edges: {}  total synapses: {}",
        stats.kept, stats.kept_weight
    );
    println!(
        "  dropped, post not a neuron: {}",
        stats.dropped_post_not_neuron
    );
    println!(
        "  dropped, pre not a neuron:  {}",
        stats.dropped_pre_not_neuron
    );
    println!("  dropped, neither a neuron:  {}", stats.dropped_neither);
    println!("  dropped synapses total:     {}", stats.dropped_weight);

    let mut has_out = vec![false; neurons.len()];
    let mut has_in = vec![false; neurons.len()];
    let mut max_w = 0u32;
    for e in &edges {
        has_out[e.pre as usize] = true;
        has_in[e.post as usize] = true;
        max_w = max_w.max(e.weight);
    }
    let connected = has_out
        .iter()
        .zip(&has_in)
        .filter(|(o, i)| **o || **i)
        .count();
    println!(
        "neurons with at least one edge: {connected}  (isolated: {})",
        neurons.len() - connected
    );
    println!("max edge weight: {max_w}");
    Ok(())
}
