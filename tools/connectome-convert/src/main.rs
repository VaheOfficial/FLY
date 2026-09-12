//! Converts raw connectome exports (neurons, synapses, cell types, transmitter
//! predictions) into the packed CSR binary that `flybrain` loads at startup.

mod assemble;
mod commands;
mod edges;
mod feather;
mod malecns;
mod neurons;
mod transmitters;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

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
        /// Only count rows where a string column equals a value, as COLUMN=VALUE.
        #[arg(long, value_name = "COLUMN=VALUE")]
        r#where: Option<String>,
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
        Command::Inspect { path, rows } => commands::inspect::run(&path, rows),
        Command::Counts {
            path,
            column,
            top,
            nonnull,
            r#where,
        } => {
            let filter = r#where
                .as_deref()
                .map(|w| {
                    w.split_once('=')
                        .with_context(|| format!("--where expects COLUMN=VALUE, got {w:?}"))
                })
                .transpose()?;
            commands::counts::run(&path, &column, top, nonnull.as_deref(), filter)
        }
        Command::Summarize { data } => commands::summarize::run(&data),
        Command::Build { data, out } => commands::build::run(&data, &out),
        Command::Check {
            path,
            cell_type,
            top,
        } => commands::check::run(&path, &cell_type, top),
    }
}
