//! Per-neuron neurotransmitter predictions, joined onto the neuron set.

use std::path::Path;

use anyhow::{Context, Result, bail};
use arrow::array::{Array, AsArray};
use arrow::datatypes::Float64Type;
use flybrain::connectome::Transmitter;

use crate::feather;
use crate::neurons::NeuronTable;

/// Transmitter fields for one neuron.
#[derive(Debug, Clone, Copy, Default)]
pub struct NtRecord {
    pub consensus: Transmitter,
    pub predicted: Transmitter,
    pub confidence: f32,
    pub ground_truth: Transmitter,
}

/// Summary of the join.
#[derive(Debug, Default)]
pub struct NtStats {
    pub rows_seen: u64,
    /// Rows whose body is a neuron.
    pub matched: u64,
    /// Neurons that never appeared in the table (left at `Unclear`).
    pub neurons_without_row: u64,
    pub with_ground_truth: u64,
    pub consensus_unclear: u64,
}

/// Load the transmitter table and return one record per neuron, in neuron order.
pub fn load(path: &Path, neurons: &NeuronTable) -> Result<(Vec<NtRecord>, NtStats)> {
    let reader = feather::open(path)?;
    let schema = reader.schema();
    let c_body = feather::column_index(&schema, "body")?;
    let c_consensus = feather::column_index(&schema, "consensus_nt")?;
    let c_predicted = feather::column_index(&schema, "predicted_nt")?;
    let c_confidence = feather::column_index(&schema, "predicted_nt_confidence")?;
    let c_truth = feather::column_index(&schema, "ground_truth")?;

    let mut records = vec![None::<NtRecord>; neurons.len()];
    let mut stats = NtStats::default();
    for batch in reader {
        let batch = batch.context("reading transmitter batch")?;
        let body = feather::i64_column(&batch, c_body)?;
        let consensus = feather::str_column(&batch, c_consensus)?;
        let predicted = feather::str_column(&batch, c_predicted)?;
        let truth = feather::str_column(&batch, c_truth)?;
        let confidence = batch.column(c_confidence);
        let confidence = confidence
            .as_primitive_opt::<Float64Type>()
            .with_context(|| {
                format!(
                    "predicted_nt_confidence has type {:?}, expected Float64",
                    confidence.data_type()
                )
            })?;

        for i in 0..batch.num_rows() {
            stats.rows_seen += 1;
            let Some(idx) = neurons.index_of(body.value(i)) else {
                continue;
            };
            stats.matched += 1;
            let parse = |name: Option<&str>, field: &str| -> Result<Transmitter> {
                match name {
                    None => Ok(Transmitter::Unclear),
                    Some(s) => Transmitter::parse(s).with_context(|| {
                        format!(
                            "unknown transmitter {s:?} in column {field} for body {}",
                            body.value(i)
                        )
                    }),
                }
            };
            let record = NtRecord {
                consensus: parse(consensus.get(i), "consensus_nt")?,
                predicted: parse(predicted.get(i), "predicted_nt")?,
                confidence: if confidence.is_null(i) {
                    0.0
                } else {
                    confidence.value(i) as f32
                },
                ground_truth: parse(truth.get(i), "ground_truth")?,
            };
            if records[idx as usize].replace(record).is_some() {
                bail!(
                    "body {} appears twice in the transmitter table",
                    body.value(i)
                );
            }
        }
    }

    let records: Vec<NtRecord> = records
        .into_iter()
        .map(|r| {
            r.unwrap_or_else(|| {
                stats.neurons_without_row += 1;
                NtRecord::default()
            })
        })
        .collect();
    stats.with_ground_truth = records
        .iter()
        .filter(|r| r.ground_truth != Transmitter::Unclear)
        .count() as u64;
    stats.consensus_unclear = records
        .iter()
        .filter(|r| r.consensus == Transmitter::Unclear)
        .count() as u64;
    Ok((records, stats))
}
