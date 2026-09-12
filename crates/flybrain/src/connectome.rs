//! The packed connectome: every neuron and every neuron-to-neuron connection,
//! in the compact form the simulation loads at startup.
//!
//! Neurons are addressed by dense index in `0..neuron_count`. Connections are
//! stored as an outgoing compressed sparse row (CSR) matrix: for presynaptic
//! neuron `i`, its targets are `post[row_ptr[i]..row_ptr[i + 1]]` with matching
//! synapse counts in `weight[..]`. Within a row, targets are sorted by index.
//!
//! Synaptic sign is not stored per connection. It is a property of the
//! presynaptic neuron's transmitter, which the simulation resolves.
//!
//! The on-disk format is little-endian, sectioned, and versioned; see [`io`].

pub mod io;

/// Predicted neurotransmitter of a neuron.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Transmitter {
    /// No prediction, or predictions disagree.
    #[default]
    Unclear = 0,
    Acetylcholine = 1,
    Glutamate = 2,
    Gaba = 3,
    Histamine = 4,
    Dopamine = 5,
    Octopamine = 6,
    Serotonin = 7,
}

impl Transmitter {
    pub const ALL: [Transmitter; 8] = [
        Transmitter::Unclear,
        Transmitter::Acetylcholine,
        Transmitter::Glutamate,
        Transmitter::Gaba,
        Transmitter::Histamine,
        Transmitter::Dopamine,
        Transmitter::Octopamine,
        Transmitter::Serotonin,
    ];

    pub fn from_u8(v: u8) -> Option<Self> {
        Self::ALL.get(usize::from(v)).copied()
    }

    /// Parse the lowercase names used in the MaleCNS neurotransmitter table.
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "unclear" => Self::Unclear,
            "acetylcholine" => Self::Acetylcholine,
            "glutamate" => Self::Glutamate,
            "gaba" => Self::Gaba,
            "histamine" => Self::Histamine,
            "dopamine" => Self::Dopamine,
            "octopamine" => Self::Octopamine,
            "serotonin" => Self::Serotonin,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Unclear => "unclear",
            Self::Acetylcholine => "acetylcholine",
            Self::Glutamate => "glutamate",
            Self::Gaba => "gaba",
            Self::Histamine => "histamine",
            Self::Dopamine => "dopamine",
            Self::Octopamine => "octopamine",
            Self::Serotonin => "serotonin",
        }
    }
}

/// Which side of the body a neuron's soma sits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Side {
    #[default]
    Unknown = 0,
    Left = 1,
    Right = 2,
    Midline = 3,
}

impl Side {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::Unknown,
            1 => Self::Left,
            2 => Self::Right,
            3 => Self::Midline,
            _ => return None,
        })
    }

    /// Parse the `somaSide` codes used in the MaleCNS annotations table.
    pub fn parse(code: &str) -> Option<Self> {
        Some(match code {
            "L" => Self::Left,
            "R" => Self::Right,
            "M" => Self::Midline,
            _ => return None,
        })
    }
}

/// Sentinel for "no string" in the string-index columns.
pub const NO_STRING: u32 = u32::MAX;

/// Everything known about one neuron, apart from its connections.
///
/// String-valued fields are indices into [`Connectome::strings`].
#[derive(Debug, Clone, PartialEq)]
pub struct Neuron {
    /// The dataset's body id. Stable across dataset versions for a given neuron.
    pub body_id: i64,
    /// Published cell type name, or [`NO_STRING`].
    pub type_name: u32,
    /// Coarse category such as `descending_neuron` or `vnc_motor`. Always present.
    pub superclass: u32,
    /// Finer functional class such as `Kenyon_Cell`, or [`NO_STRING`].
    pub class: u32,
    pub side: Side,
    /// Transmitter agreed on by per-neuron and per-type predictions.
    pub nt_consensus: Transmitter,
    /// Per-neuron prediction alone, with its confidence in `0.0..=1.0`.
    pub nt_predicted: Transmitter,
    pub nt_confidence: f32,
    /// Experimentally known transmitter where available.
    pub nt_ground_truth: Transmitter,
    /// Soma position in the dataset's EM voxel grid (8 nm isotropic), if known.
    pub soma: Option<[i32; 3]>,
}

/// A whole nervous system: neurons plus outgoing CSR connections.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Connectome {
    /// Interned strings referenced by [`Neuron`] fields.
    pub strings: Vec<String>,
    pub neurons: Vec<Neuron>,
    /// `neurons.len() + 1` entries; row `i` spans `row_ptr[i]..row_ptr[i + 1]`.
    pub row_ptr: Vec<u32>,
    /// Postsynaptic neuron index of each connection.
    pub post: Vec<u32>,
    /// Synapse count of each connection.
    pub weight: Vec<u16>,
}

impl Connectome {
    pub fn neuron_count(&self) -> usize {
        self.neurons.len()
    }

    pub fn edge_count(&self) -> usize {
        self.post.len()
    }

    /// Outgoing connections of neuron `pre` as `(post, weight)` pairs.
    pub fn targets(&self, pre: u32) -> impl Iterator<Item = (u32, u16)> + '_ {
        let (a, b) = (
            self.row_ptr[pre as usize] as usize,
            self.row_ptr[pre as usize + 1] as usize,
        );
        self.post[a..b]
            .iter()
            .copied()
            .zip(self.weight[a..b].iter().copied())
    }

    /// Resolve a string index, treating [`NO_STRING`] as `None`.
    pub fn string(&self, idx: u32) -> Option<&str> {
        (idx != NO_STRING).then(|| self.strings[idx as usize].as_str())
    }

    /// Check the structural invariants a well-formed connectome must satisfy.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.neurons.len();
        if self.row_ptr.len() != n + 1 {
            return Err(format!(
                "row_ptr has {} entries, expected {}",
                self.row_ptr.len(),
                n + 1
            ));
        }
        if self.row_ptr[0] != 0 {
            return Err("row_ptr[0] must be 0".into());
        }
        if self.row_ptr[n] as usize != self.post.len() {
            return Err(format!(
                "row_ptr[n] = {} but there are {} edges",
                self.row_ptr[n],
                self.post.len()
            ));
        }
        if self.post.len() != self.weight.len() {
            return Err(format!(
                "{} post indices but {} weights",
                self.post.len(),
                self.weight.len()
            ));
        }
        for i in 0..n {
            let (a, b) = (self.row_ptr[i] as usize, self.row_ptr[i + 1] as usize);
            if a > b {
                return Err(format!("row_ptr decreases at neuron {i}"));
            }
            let row = &self.post[a..b];
            if row.iter().any(|&p| p as usize >= n) {
                return Err(format!("neuron {i} targets a neuron index out of range"));
            }
            if row.windows(2).any(|w| w[0] >= w[1]) {
                return Err(format!("targets of neuron {i} are not strictly increasing"));
            }
            if self.weight[a..b].contains(&0) {
                return Err(format!("neuron {i} has a zero-weight connection"));
            }
        }
        for (i, nr) in self.neurons.iter().enumerate() {
            for (field, idx) in [
                ("type", nr.type_name),
                ("superclass", nr.superclass),
                ("class", nr.class),
            ] {
                if idx != NO_STRING && idx as usize >= self.strings.len() {
                    return Err(format!(
                        "neuron {i} has {field} string index {idx} out of range"
                    ));
                }
            }
            if nr.superclass == NO_STRING {
                return Err(format!("neuron {i} has no superclass"));
            }
            if !(0.0..=1.0).contains(&nr.nt_confidence) {
                return Err(format!(
                    "neuron {i} has transmitter confidence {} outside 0..=1",
                    nr.nt_confidence
                ));
            }
        }
        Ok(())
    }
}
