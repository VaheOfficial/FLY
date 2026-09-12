//! Named groups of neurons, resolved once from published cell types.

use crate::connectome::Connectome;

/// Handle to a registered population.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PopulationId(pub(super) usize);

/// A named set of neuron indices, e.g. the labellar sugar GRNs.
#[derive(Debug, Clone)]
pub struct Population {
    pub name: String,
    pub neurons: Vec<u32>,
}

impl Population {
    /// All neurons whose cell type is one of `types`. Empty if none match.
    pub fn from_types<S: AsRef<str>>(net: &Connectome, name: &str, types: &[S]) -> Self {
        Self {
            name: name.to_owned(),
            neurons: net.neurons_of_types(types),
        }
    }

    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty()
    }
}
