//! Hand-built micro-networks for simulator tests.

use crate::connectome::{Connectome, NO_STRING, Neuron, Side, Transmitter};

/// A tiny network: neuron `i` with transmitter `nts[i]`, edges as
/// `(pre, post, synapse_count)`.
pub(super) fn net(nts: &[Transmitter], edges: &[(u32, u32, u16)]) -> Connectome {
    let n = nts.len();
    let neurons = nts
        .iter()
        .enumerate()
        .map(|(i, &nt)| Neuron {
            body_id: i as i64,
            type_name: NO_STRING,
            superclass: 0,
            class: NO_STRING,
            side: Side::Unknown,
            nt_consensus: nt,
            nt_predicted: nt,
            nt_confidence: 1.0,
            nt_ground_truth: Transmitter::Unclear,
            soma: None,
        })
        .collect();
    let mut sorted = edges.to_vec();
    sorted.sort();
    let mut row_ptr = vec![0u32; n + 1];
    for &(pre, _, _) in &sorted {
        row_ptr[pre as usize + 1] += 1;
    }
    for i in 0..n {
        row_ptr[i + 1] += row_ptr[i];
    }
    let c = Connectome {
        strings: vec!["test".into()],
        neurons,
        row_ptr,
        post: sorted.iter().map(|e| e.1).collect(),
        weight: sorted.iter().map(|e| e.2).collect(),
    };
    c.validate().unwrap();
    c
}

pub(super) const ACH: Transmitter = Transmitter::Acetylcholine;
pub(super) const GABA: Transmitter = Transmitter::Gaba;
