//! The neuron table: every body the dataset calls a neuron, in a stable order.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use arrow::array::Array;

use crate::feather;

/// One neuron from the body annotations table.
#[derive(Debug, Clone)]
pub struct Neuron {
    pub body_id: i64,
    /// Published cell-type name, e.g. `DNp01`. Missing for a few untyped neurons.
    pub type_name: Option<String>,
    /// Type plus side, e.g. `DNp01(GF)_R`.
    pub instance: Option<String>,
    /// Coarse category, e.g. `descending_neuron`, `vnc_motor`. Present on every neuron.
    pub superclass: String,
    /// Finer functional class where annotated, e.g. `Kenyon_Cell`, `DAN`.
    pub class: Option<String>,
    pub soma_side: Option<String>,
    pub status: Option<String>,
}

/// All neurons, sorted by body id, plus a body id to dense index map.
pub struct NeuronTable {
    pub neurons: Vec<Neuron>,
    index: HashMap<i64, u32>,
}

impl NeuronTable {
    /// Load every body with a non-null `superclass`.
    ///
    /// That predicate reproduces the published neuron count exactly: the
    /// annotations table also lists glia, orphan fragments and out-of-scope
    /// bodies, none of which carry a superclass.
    pub fn load(path: &Path) -> Result<Self> {
        let reader = feather::open(path)?;
        let schema = reader.schema();
        let c_body = feather::column_index(&schema, "bodyId")?;
        let c_type = feather::column_index(&schema, "type")?;
        let c_instance = feather::column_index(&schema, "instance")?;
        let c_superclass = feather::column_index(&schema, "superclass")?;
        let c_class = feather::column_index(&schema, "class")?;
        let c_side = feather::column_index(&schema, "somaSide")?;
        let c_status = feather::column_index(&schema, "status")?;

        let mut neurons = Vec::new();
        for batch in reader {
            let batch = batch.context("reading annotations batch")?;
            let body = feather::i64_column(&batch, c_body)?;
            let ty = feather::str_column(&batch, c_type)?;
            let instance = feather::str_column(&batch, c_instance)?;
            let superclass = feather::str_column(&batch, c_superclass)?;
            let class = feather::str_column(&batch, c_class)?;
            let side = feather::str_column(&batch, c_side)?;
            let status = feather::str_column(&batch, c_status)?;
            for i in 0..batch.num_rows() {
                let Some(superclass) = superclass.get(i) else {
                    continue;
                };
                if body.is_null(i) {
                    bail!("neuron row {i} has a null bodyId");
                }
                neurons.push(Neuron {
                    body_id: body.value(i),
                    type_name: ty.get(i).map(str::to_owned),
                    instance: instance.get(i).map(str::to_owned),
                    superclass: superclass.to_owned(),
                    class: class.get(i).map(str::to_owned),
                    soma_side: side.get(i).map(str::to_owned),
                    status: status.get(i).map(str::to_owned),
                });
            }
        }

        neurons.sort_by_key(|n| n.body_id);
        let mut index = HashMap::with_capacity(neurons.len());
        for (i, n) in neurons.iter().enumerate() {
            if index.insert(n.body_id, i as u32).is_some() {
                bail!("duplicate bodyId {} in annotations", n.body_id);
            }
        }
        Ok(Self { neurons, index })
    }

    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    /// Dense index of a body id, if it is a neuron.
    pub fn index_of(&self, body_id: i64) -> Option<u32> {
        self.index.get(&body_id).copied()
    }
}
