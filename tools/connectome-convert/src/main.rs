//! Converts raw connectome exports (neurons, synapses, cell types, transmitter
//! predictions) into the packed CSR binary that `flybrain` loads at startup.

mod edges;
mod feather;
mod neurons;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use arrow::array::Array;
use arrow::util::pretty::print_batches;
use clap::{Parser, Subcommand};

use crate::neurons::NeuronTable;

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
    let neurons =
        NeuronTable::load(&data.join("body-annotations-male-cns-v1.0-minconf-0.5.feather"))?;
    println!("neurons: {}  ({:.1?})", neurons.len(), t0.elapsed());
    let count =
        |pred: fn(&neurons::Neuron) -> bool| neurons.neurons.iter().filter(|n| pred(n)).count();
    println!("  with type: {}", count(|n| n.type_name.is_some()));
    println!("  with instance: {}", count(|n| n.instance.is_some()));
    println!("  with class: {}", count(|n| n.class.is_some()));
    println!("  with soma side: {}", count(|n| n.soma_side.is_some()));
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
    let (edges, stats) = edges::load(
        &data.join("connectome-weights-male-cns-v1.0-minconf-0.5.feather"),
        &neurons,
    )?;
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
