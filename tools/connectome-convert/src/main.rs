//! Converts raw connectome exports (neurons, synapses, cell types, transmitter
//! predictions) into the packed CSR binary that `flybrain` loads at startup.

mod build;
mod edges;
mod feather;
mod neurons;
mod transmitters;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use arrow::array::Array;
use arrow::util::pretty::print_batches;
use clap::{Parser, Subcommand};
use flybrain::connectome::{self, Transmitter};

use crate::neurons::NeuronTable;

const ANNOTATIONS: &str = "body-annotations-male-cns-v1.0-minconf-0.5.feather";
const TRANSMITTERS: &str = "body-neurotransmitters-male-cns-v1.0.feather";
const WEIGHTS: &str = "connectome-weights-male-cns-v1.0-minconf-0.5.feather";

#[derive(Parser)]
#[command(about, version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print the schema, row count, and first rows of a Feather (Arrow IPC) table.
    Inspect {
        /// Path to a `.feather` file.
        path: PathBuf,
        /// How many leading rows to print.
        #[arg(long, default_value_t = 5)]
        rows: usize,
    },
    /// Print value counts for a string column, most frequent first.
    Counts {
        /// Path to a `.feather` file.
        path: PathBuf,
        /// Column name (Utf8 or dictionary-encoded Utf8).
        column: String,
        /// Only print the top N values.
        #[arg(long, default_value_t = 40)]
        top: usize,
        /// Only count rows where this column is non-null.
        #[arg(long)]
        nonnull: Option<String>,
    },
    /// Load the neuron and weights tables and report what survives filtering
    /// to neuron-to-neuron edges, for comparison against published totals.
    Summarize {
        /// Directory holding the downloaded MaleCNS flat-connectome tables.
        #[arg(long, default_value = "data/malecns-v1.0")]
        data: PathBuf,
    },
    /// Build the packed connectome file that flybrain loads.
    Build {
        /// Directory holding the downloaded MaleCNS flat-connectome tables.
        #[arg(long, default_value = "data/malecns-v1.0")]
        data: PathBuf,
        /// Where to write the packed file.
        #[arg(long, default_value = "data/malecns-v1.0.flycnx")]
        out: PathBuf,
    },
    /// Load a packed connectome file, verify it, and print a summary plus the
    /// strongest outputs of a named cell type as a sanity check.
    Check {
        /// Packed connectome file.
        #[arg(default_value = "data/malecns-v1.0.flycnx")]
        path: PathBuf,
        /// Cell type whose outputs to list.
        #[arg(long, default_value = "DNp01")]
        cell_type: String,
        /// How many outputs to list per neuron.
        #[arg(long, default_value_t = 8)]
        top: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect { path, rows } => inspect(&path, rows),
        Command::Counts {
            path,
            column,
            top,
            nonnull,
        } => counts(&path, &column, top, nonnull.as_deref()),
        Command::Summarize { data } => summarize(&data),
        Command::Build { data, out } => build(&data, &out),
        Command::Check {
            path,
            cell_type,
            top,
        } => check(&path, &cell_type, top),
    }
}

fn inspect(path: &Path, rows: usize) -> Result<()> {
    let reader = feather::open(path)?;

    println!("{}", path.display());
    println!("schema:");
    for field in reader.schema().fields() {
        println!(
            "  {:<28} {:?}{}",
            field.name(),
            field.data_type(),
            if field.is_nullable() {
                ""
            } else {
                " (non-null)"
            }
        );
    }

    let mut total_rows = 0usize;
    let mut batches = 0usize;
    let mut head = None;
    for batch in reader {
        let batch = batch.context("reading record batch")?;
        if head.is_none() {
            head = Some(batch.slice(0, rows.min(batch.num_rows())));
        }
        total_rows += batch.num_rows();
        batches += 1;
    }
    println!(
        "rows: {total_rows}  (in {batches} batch{})",
        if batches == 1 { "" } else { "es" }
    );

    if let Some(head) = head {
        println!("first {} rows:", head.num_rows());
        print_batches(&[head])?;
    }
    Ok(())
}

fn counts(path: &Path, column: &str, top: usize, nonnull: Option<&str>) -> Result<()> {
    let reader = feather::open(path)?;
    let schema = reader.schema();
    let idx = feather::column_index(&schema, column)?;
    let filter_idx = nonnull
        .map(|c| feather::column_index(&schema, c))
        .transpose()?;

    let mut tally: HashMap<String, usize> = HashMap::new();
    let mut nulls = 0usize;
    let mut total = 0usize;
    for batch in reader {
        let batch = batch.context("reading record batch")?;
        let col = feather::str_column(&batch, idx)?;
        for i in 0..batch.num_rows() {
            if let Some(f) = filter_idx
                && batch.column(f).is_null(i)
            {
                continue;
            }
            total += 1;
            match col.get(i) {
                Some(s) => *tally.entry(s.to_owned()).or_default() += 1,
                None => nulls += 1,
            }
        }
    }

    let mut rows: Vec<(String, usize)> = tally.into_iter().collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    println!(
        "{column}: {} distinct values, {nulls} nulls, {total} rows",
        rows.len()
    );
    for (value, n) in rows.iter().take(top) {
        println!("{n:>10}  {value}");
    }
    if rows.len() > top {
        println!("       ...  ({} more)", rows.len() - top);
    }
    Ok(())
}

fn summarize(data: &Path) -> Result<()> {
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

fn build(data: &Path, out: &Path) -> Result<()> {
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

    let connectome = build::assemble(&neurons, &nt, edges)?;
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

fn check(path: &Path, cell_type: &str, top: usize) -> Result<()> {
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
