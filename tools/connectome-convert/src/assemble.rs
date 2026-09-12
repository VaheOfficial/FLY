//! Assemble the packed [`Connectome`] from the loaded tables.

use std::collections::HashMap;

use anyhow::{Result, bail};
use flybrain::connectome::{Connectome, NO_STRING, Neuron, Side};

use crate::edges::Edge;
use crate::neurons::NeuronTable;
use crate::transmitters::NtRecord;

/// Interns strings, handing out dense indices.
#[derive(Default)]
struct Interner {
    strings: Vec<String>,
    index: HashMap<String, u32>,
}

impl Interner {
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.index.get(s) {
            return i;
        }
        let i = self.strings.len() as u32;
        self.strings.push(s.to_owned());
        self.index.insert(s.to_owned(), i);
        i
    }

    fn intern_opt(&mut self, s: Option<&str>) -> u32 {
        s.map_or(NO_STRING, |s| self.intern(s))
    }
}

/// Build the connectome. `edges` may be in any order; they are sorted here.
/// Fails on duplicate (pre, post) pairs or weights that do not fit in `u16`.
pub fn connectome(
    neurons: &NeuronTable,
    nt: &[NtRecord],
    mut edges: Vec<Edge>,
) -> Result<Connectome> {
    assert_eq!(neurons.len(), nt.len(), "one transmitter record per neuron");

    let mut interner = Interner::default();
    let packed: Vec<Neuron> = neurons
        .neurons
        .iter()
        .zip(nt)
        .map(|(n, t)| Neuron {
            body_id: n.body_id,
            type_name: interner.intern_opt(n.type_name.as_deref()),
            superclass: interner.intern(&n.superclass),
            class: interner.intern_opt(n.class.as_deref()),
            side: n
                .soma_side
                .as_deref()
                .and_then(Side::parse)
                .unwrap_or_default(),
            nt_consensus: t.consensus,
            nt_predicted: t.predicted,
            nt_confidence: t.confidence.clamp(0.0, 1.0),
            nt_ground_truth: t.ground_truth,
            soma: n.soma,
        })
        .collect();

    edges.sort_unstable_by_key(|e| (e.pre, e.post));
    if let Some(dup) = edges
        .windows(2)
        .find(|w| w[0].pre == w[1].pre && w[0].post == w[1].post)
    {
        bail!(
            "duplicate edge {} -> {} in weights table",
            neurons.neurons[dup[0].pre as usize].body_id,
            neurons.neurons[dup[0].post as usize].body_id
        );
    }

    let n = neurons.len();
    let mut row_ptr = vec![0u32; n + 1];
    for e in &edges {
        row_ptr[e.pre as usize + 1] += 1;
    }
    for i in 0..n {
        row_ptr[i + 1] += row_ptr[i];
    }

    let mut post = Vec::with_capacity(edges.len());
    let mut weight = Vec::with_capacity(edges.len());
    for e in &edges {
        post.push(e.post);
        weight.push(u16::try_from(e.weight).map_err(|_| {
            anyhow::anyhow!(
                "edge weight {} exceeds u16 (body {})",
                e.weight,
                neurons.neurons[e.pre as usize].body_id
            )
        })?);
    }

    let c = Connectome {
        strings: interner.strings,
        neurons: packed,
        row_ptr,
        post,
        weight,
    };
    c.validate()
        .map_err(|why| anyhow::anyhow!("assembled connectome is invalid: {why}"))?;
    Ok(c)
}
