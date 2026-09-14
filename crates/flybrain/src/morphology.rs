//! Simplified neuron morphologies for drawing: every neuron's branching
//! tree as polylines, quantised, in one packed file.
//!
//! Layout, all little-endian:
//!
//! ```text
//! magic          8 bytes   b"FLYSKEL\0"
//! version        u32       1
//! neuron_count   u32       (of the connectome this was built against)
//! vertex_count   u32
//! strip_count    u32
//! positions      vertex_count x 3 x u16   (units of POSITION_UNIT_NM)
//! vertex_neuron  vertex_count x u32       (dense neuron index)
//! strips         strip_count x (first u32, len u32)
//! ```
//!
//! A strip's vertices are consecutive in the arrays, so a strip draws as a
//! line strip over `first..first + len`.

use std::fmt;
use std::io::{self, Read, Write};

const MAGIC: &[u8; 8] = b"FLYSKEL\0";
const VERSION: u32 = 1;

/// Nanometres per quantised position unit: 16 nm keeps the dataset's whole
/// extent inside `u16` with two-voxel precision.
pub const POSITION_UNIT_NM: f32 = 16.0;

/// One polyline of a neuron's skeleton.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Strip {
    pub first: u32,
    pub len: u32,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Morphology {
    pub neuron_count: u32,
    /// Quantised positions; multiply by [`POSITION_UNIT_NM`] for nanometres.
    pub positions: Vec<[u16; 3]>,
    pub vertex_neuron: Vec<u32>,
    pub strips: Vec<Strip>,
}

impl Morphology {
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Quantise a nanometre position, saturating at the `u16` range.
    pub fn quantise(nm: [f32; 3]) -> [u16; 3] {
        nm.map(|v| {
            (v / POSITION_UNIT_NM)
                .round()
                .clamp(0.0, f32::from(u16::MAX)) as u16
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.vertex_neuron.len() != self.positions.len() {
            return Err(format!(
                "{} positions but {} vertex neuron ids",
                self.positions.len(),
                self.vertex_neuron.len()
            ));
        }
        if let Some(bad) = self.vertex_neuron.iter().find(|&&n| n >= self.neuron_count) {
            return Err(format!(
                "vertex refers to neuron {bad} of {}",
                self.neuron_count
            ));
        }
        let mut expected_first = 0u32;
        for (i, s) in self.strips.iter().enumerate() {
            if s.first != expected_first {
                return Err(format!(
                    "strip {i} starts at {} but the previous ended at {expected_first}",
                    s.first
                ));
            }
            if s.len < 2 {
                return Err(format!(
                    "strip {i} has {} vertices; a line needs two",
                    s.len
                ));
            }
            expected_first = s.first.checked_add(s.len).ok_or("strip overflow")?;
        }
        if expected_first as usize != self.positions.len() {
            return Err(format!(
                "strips cover {expected_first} vertices but there are {}",
                self.positions.len()
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum ReadError {
    Io(io::Error),
    BadMagic,
    UnsupportedVersion(u32),
    Truncated,
    Corrupt(String),
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "i/o error: {e}"),
            Self::BadMagic => write!(f, "not a flybrain morphology file"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported morphology format version {v}"),
            Self::Truncated => write!(f, "morphology file is truncated"),
            Self::Corrupt(why) => write!(f, "morphology file is corrupt: {why}"),
        }
    }
}

impl std::error::Error for ReadError {}

impl From<io::Error> for ReadError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

pub fn write<W: Write>(m: &Morphology, mut w: W) -> io::Result<()> {
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&m.neuron_count.to_le_bytes())?;
    w.write_all(&(m.positions.len() as u32).to_le_bytes())?;
    w.write_all(&(m.strips.len() as u32).to_le_bytes())?;
    let mut buf = Vec::with_capacity(m.positions.len() * 6);
    for p in &m.positions {
        for v in p {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    w.write_all(&buf)?;
    buf.clear();
    for n in &m.vertex_neuron {
        buf.extend_from_slice(&n.to_le_bytes());
    }
    w.write_all(&buf)?;
    buf.clear();
    for s in &m.strips {
        buf.extend_from_slice(&s.first.to_le_bytes());
        buf.extend_from_slice(&s.len.to_le_bytes());
    }
    w.write_all(&buf)?;
    w.flush()
}

pub fn read<R: Read>(mut r: R) -> Result<Morphology, ReadError> {
    let mut bytes = Vec::new();
    r.read_to_end(&mut bytes)?;
    let mut pos = 0usize;
    let mut take = |n: usize| -> Result<&[u8], ReadError> {
        let s = bytes.get(pos..pos + n).ok_or(ReadError::Truncated)?;
        pos += n;
        Ok(s)
    };
    if take(8)? != MAGIC {
        return Err(ReadError::BadMagic);
    }
    let u32_at = |s: &[u8]| u32::from_le_bytes(s.try_into().unwrap());
    let version = u32_at(take(4)?);
    if version != VERSION {
        return Err(ReadError::UnsupportedVersion(version));
    }
    let neuron_count = u32_at(take(4)?);
    let vertex_count = u32_at(take(4)?) as usize;
    let strip_count = u32_at(take(4)?) as usize;
    let positions = take(vertex_count.checked_mul(6).ok_or(ReadError::Truncated)?)?
        .chunks_exact(6)
        .map(|c| {
            [
                u16::from_le_bytes([c[0], c[1]]),
                u16::from_le_bytes([c[2], c[3]]),
                u16::from_le_bytes([c[4], c[5]]),
            ]
        })
        .collect();
    let vertex_neuron = take(vertex_count * 4)?
        .chunks_exact(4)
        .map(u32_at)
        .collect();
    let strips = take(strip_count * 8)?
        .chunks_exact(8)
        .map(|c| Strip {
            first: u32_at(&c[..4]),
            len: u32_at(&c[4..]),
        })
        .collect();
    if pos != bytes.len() {
        return Err(ReadError::Corrupt(format!(
            "{} trailing bytes",
            bytes.len() - pos
        )));
    }
    let m = Morphology {
        neuron_count,
        positions,
        vertex_neuron,
        strips,
    };
    m.validate().map_err(ReadError::Corrupt)?;
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Morphology {
        Morphology {
            neuron_count: 3,
            positions: vec![
                [0, 0, 0],
                [10, 0, 0],
                [10, 5, 0],
                [100, 100, 100],
                [110, 100, 90],
            ],
            vertex_neuron: vec![0, 0, 0, 2, 2],
            strips: vec![Strip { first: 0, len: 3 }, Strip { first: 3, len: 2 }],
        }
    }

    #[test]
    fn roundtrip() {
        let m = sample();
        m.validate().unwrap();
        let mut buf = Vec::new();
        write(&m, &mut buf).unwrap();
        assert_eq!(read(buf.as_slice()).unwrap(), m);
    }

    #[test]
    fn quantisation_and_validation() {
        assert_eq!(
            Morphology::quantise([16.0, 24.0, 2_000_000.0]),
            [1, 2, u16::MAX]
        );
        let mut m = sample();
        m.strips[1].first = 4;
        assert!(m.validate().unwrap_err().contains("starts at 4"));
        let mut m = sample();
        m.vertex_neuron[0] = 3;
        assert!(m.validate().unwrap_err().contains("neuron 3"));
        let mut m = sample();
        m.strips = vec![Strip { first: 0, len: 5 }];
        m.validate().unwrap();
    }
}
