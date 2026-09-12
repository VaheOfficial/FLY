//! `check`: reload a packed file, validate it, and spot-check a cell type's outputs.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use flybrain::connectome::{self, Transmitter};

pub fn run(path: &Path, cell_type: &str, top: usize) -> Result<()> {
    let t0 = Instant::now();
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let c = connectome::io::read(BufReader::new(file))
        .with_context(|| format!("reading {}", path.display()))?;
    println!(
        "{}: {} neurons, {} edges, {} strings, loaded and validated in {:.1?}",
        path.display(),
        c.neuron_count(),
        c.edge_count(),
        c.strings.len(),
        t0.elapsed()
    );

    let synapses: u64 = c.weight.iter().map(|&w| u64::from(w)).sum();
    let with_outputs = (0..c.neuron_count() as u32)
        .filter(|&i| c.targets(i).next().is_some())
        .count();
    let mut has_input = vec![false; c.neuron_count()];
    for &p in &c.post {
        has_input[p as usize] = true;
    }
    let connected = (0..c.neuron_count())
        .filter(|&i| has_input[i] || c.targets(i as u32).next().is_some())
        .count();
    println!("  synapses: {synapses}");
    println!(
        "  neurons with outputs: {with_outputs}   with any connection: {connected}   isolated: {}",
        c.neuron_count() - connected
    );
    println!(
        "  with soma position: {}",
        c.neurons.iter().filter(|n| n.soma.is_some()).count()
    );

    println!("  consensus transmitter:");
    for t in Transmitter::ALL {
        let n = c.neurons.iter().filter(|nr| nr.nt_consensus == t).count();
        println!("    {n:>8}  {}", t.name());
    }

    let members: Vec<u32> = (0..c.neuron_count() as u32)
        .filter(|&i| c.string(c.neurons[i as usize].type_name) == Some(cell_type))
        .collect();
    println!("{cell_type}: {} neuron(s)", members.len());
    for &i in &members {
        let nr = &c.neurons[i as usize];
        let mut outs: Vec<(u32, u16)> = c.targets(i).collect();
        outs.sort_by_key(|&(_, w)| std::cmp::Reverse(w));
        println!(
            "  body {} {:?} {} soma {:?}: {} outputs, {} synapses",
            nr.body_id,
            nr.side,
            nr.nt_consensus.name(),
            nr.soma,
            outs.len(),
            outs.iter().map(|&(_, w)| u64::from(w)).sum::<u64>()
        );
        for (post, w) in outs.iter().take(top) {
            let p = &c.neurons[*post as usize];
            println!(
                "    {w:>5}  -> {:<20} body {} {:?} {}",
                c.string(p.type_name).unwrap_or("?"),
                p.body_id,
                p.side,
                c.string(p.superclass).unwrap_or("?")
            );
        }
    }
    Ok(())
}
