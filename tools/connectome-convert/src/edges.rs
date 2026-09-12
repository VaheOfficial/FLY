//! Neuron-to-neuron connections, filtered from the body-level weights table.

use std::path::Path;

use anyhow::{Context, Result};

use crate::feather;
use crate::neurons::NeuronTable;

/// A directed connection between two neurons, by dense neuron index.
#[derive(Debug, Clone, Copy)]
pub struct Edge {
    pub pre: u32,
    pub post: u32,
    /// Synapse count.
    pub weight: u32,
}

/// What happened to the rows of the weights table during filtering.
#[derive(Debug, Default)]
pub struct EdgeStats {
    pub rows_seen: u64,
    pub kept: u64,
    pub kept_weight: u64,
    /// Rows where only the presynaptic body was a neuron.
    pub dropped_post_not_neuron: u64,
    /// Rows where only the postsynaptic body was a neuron.
    pub dropped_pre_not_neuron: u64,
    /// Rows where neither body was a neuron.
    pub dropped_neither: u64,
    pub dropped_weight: u64,
}

/// Stream the weights table and keep only edges whose both ends are neurons.
pub fn load(path: &Path, neurons: &NeuronTable) -> Result<(Vec<Edge>, EdgeStats)> {
    let reader = feather::open(path)?;
    let schema = reader.schema();
    let c_pre = feather::column_index(&schema, "body_pre")?;
    let c_post = feather::column_index(&schema, "body_post")?;
    let c_weight = feather::column_index(&schema, "weight")?;

    let mut edges = Vec::new();
    let mut stats = EdgeStats::default();
    for batch in reader {
        let batch = batch.context("reading weights batch")?;
        let pre = feather::i64_column(&batch, c_pre)?;
        let post = feather::i64_column(&batch, c_post)?;
        let weight = feather::i64_column(&batch, c_weight)?;
        for i in 0..batch.num_rows() {
            stats.rows_seen += 1;
            let w = weight.value(i);
            let w = u32::try_from(w)
                .with_context(|| format!("weight {w} out of range at row {}", stats.rows_seen))?;
            match (
                neurons.index_of(pre.value(i)),
                neurons.index_of(post.value(i)),
            ) {
                (Some(pre), Some(post)) => {
                    stats.kept += 1;
                    stats.kept_weight += u64::from(w);
                    edges.push(Edge {
                        pre,
                        post,
                        weight: w,
                    });
                }
                (Some(_), None) => {
                    stats.dropped_post_not_neuron += 1;
                    stats.dropped_weight += u64::from(w);
                }
                (None, Some(_)) => {
                    stats.dropped_pre_not_neuron += 1;
                    stats.dropped_weight += u64::from(w);
                }
                (None, None) => {
                    stats.dropped_neither += 1;
                    stats.dropped_weight += u64::from(w);
                }
            }
        }
    }
    Ok((edges, stats))
}
