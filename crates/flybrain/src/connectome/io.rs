//! On-disk format for [`Connectome`].
//!
//! Layout, all little-endian:
//!
//! ```text
//! magic            8 bytes   b"FLYCNX\0\0"
//! version          u32       1
//! neuron_count     u32
//! edge_count       u64
//! string_count     u32
//! strings          string_count x (u16 length, utf-8 bytes)
//! body_id          neuron_count x i64
//! type_name        neuron_count x u32
//! superclass       neuron_count x u32
//! class            neuron_count x u32
//! side             neuron_count x u8
//! nt_consensus     neuron_count x u8
//! nt_predicted     neuron_count x u8
//! nt_ground_truth  neuron_count x u8
//! nt_confidence    neuron_count x f32
//! soma_present     neuron_count x u8
//! soma             neuron_count x 3 x i32   (zeros where absent)
//! row_ptr          (neuron_count + 1) x u32
//! post             edge_count x u32
//! weight           edge_count x u16
//! ```
//!
//! Struct-of-arrays so each column can be uploaded to the GPU without
//! repacking. No compression: the file is read once and the largest column
//! (post indices) is already dense.

use std::fmt;
use std::io::{self, Read, Write};

use super::{Connectome, Neuron, Side, Transmitter};

const MAGIC: &[u8; 8] = b"FLYCNX\0\0";
const VERSION: u32 = 1;

/// Why a connectome file could not be read.
#[derive(Debug)]
pub enum ReadError {
    Io(io::Error),
    BadMagic,
    UnsupportedVersion(u32),
    /// The file ended before the declared contents.
    Truncated,
    /// A field held a value that has no meaning, e.g. an unknown transmitter code.
    Corrupt(String),
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "i/o error: {e}"),
            Self::BadMagic => write!(f, "not a flybrain connectome file"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported connectome format version {v}"),
            Self::Truncated => write!(f, "connectome file is truncated"),
            Self::Corrupt(why) => write!(f, "connectome file is corrupt: {why}"),
        }
    }
}

impl std::error::Error for ReadError {}

impl From<io::Error> for ReadError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// Serialize a connectome. The caller is expected to have run
/// [`Connectome::validate`] first; this writes whatever it is given.
pub fn write<W: Write>(c: &Connectome, mut w: W) -> io::Result<()> {
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(c.neurons.len() as u32).to_le_bytes())?;
    w.write_all(&(c.post.len() as u64).to_le_bytes())?;

    w.write_all(&(c.strings.len() as u32).to_le_bytes())?;
    for s in &c.strings {
        let len = u16::try_from(s.len())
            .map_err(|_| io::Error::other(format!("string too long: {s:?}")))?;
        w.write_all(&len.to_le_bytes())?;
        w.write_all(s.as_bytes())?;
    }

    write_column(&mut w, c.neurons.iter().map(|n| n.body_id.to_le_bytes()))?;
    write_column(&mut w, c.neurons.iter().map(|n| n.type_name.to_le_bytes()))?;
    write_column(&mut w, c.neurons.iter().map(|n| n.superclass.to_le_bytes()))?;
    write_column(&mut w, c.neurons.iter().map(|n| n.class.to_le_bytes()))?;
    write_column(&mut w, c.neurons.iter().map(|n| [n.side as u8]))?;
    write_column(&mut w, c.neurons.iter().map(|n| [n.nt_consensus as u8]))?;
    write_column(&mut w, c.neurons.iter().map(|n| [n.nt_predicted as u8]))?;
    write_column(&mut w, c.neurons.iter().map(|n| [n.nt_ground_truth as u8]))?;
    write_column(
        &mut w,
        c.neurons.iter().map(|n| n.nt_confidence.to_le_bytes()),
    )?;
    write_column(&mut w, c.neurons.iter().map(|n| [n.soma.is_some() as u8]))?;
    for n in &c.neurons {
        for v in n.soma.unwrap_or([0; 3]) {
            w.write_all(&v.to_le_bytes())?;
        }
    }

    write_column(&mut w, c.row_ptr.iter().map(|v| v.to_le_bytes()))?;
    write_column(&mut w, c.post.iter().map(|v| v.to_le_bytes()))?;
    write_column(&mut w, c.weight.iter().map(|v| v.to_le_bytes()))?;
    w.flush()
}

fn write_column<W: Write, const N: usize>(
    w: &mut W,
    items: impl Iterator<Item = [u8; N]>,
) -> io::Result<()> {
    // Batch small fixed-size items into large writes.
    let mut buf = Vec::with_capacity(1 << 20);
    for item in items {
        buf.extend_from_slice(&item);
        if buf.len() >= (1 << 20) {
            w.write_all(&buf)?;
            buf.clear();
        }
    }
    w.write_all(&buf)
}

/// Deserialize a connectome and check its invariants.
pub fn read<R: Read>(mut r: R) -> Result<Connectome, ReadError> {
    let mut bytes = Vec::new();
    r.read_to_end(&mut bytes)?;
    let mut cur = Cursor {
        bytes: &bytes,
        pos: 0,
    };

    if cur.take(8)? != MAGIC {
        return Err(ReadError::BadMagic);
    }
    let version = cur.u32()?;
    if version != VERSION {
        return Err(ReadError::UnsupportedVersion(version));
    }
    let n = cur.u32()? as usize;
    let e = usize::try_from(cur.u64()?)
        .map_err(|_| ReadError::Corrupt("edge count too large".into()))?;

    let string_count = cur.u32()? as usize;
    let mut strings = Vec::with_capacity(string_count);
    for _ in 0..string_count {
        let len = cur.u16()? as usize;
        let s = std::str::from_utf8(cur.take(len)?)
            .map_err(|_| ReadError::Corrupt("string is not utf-8".into()))?;
        strings.push(s.to_owned());
    }

    let body_id = cur.column(n, i64::from_le_bytes)?;
    let type_name = cur.column(n, u32::from_le_bytes)?;
    let superclass = cur.column(n, u32::from_le_bytes)?;
    let class = cur.column(n, u32::from_le_bytes)?;
    let side = cur.column(n, |[b]| b)?;
    let nt_consensus = cur.column(n, |[b]| b)?;
    let nt_predicted = cur.column(n, |[b]| b)?;
    let nt_ground_truth = cur.column(n, |[b]| b)?;
    let nt_confidence = cur.column(n, f32::from_le_bytes)?;
    let soma_present = cur.column(n, |[b]| b)?;
    let soma_xyz = cur.column(n * 3, i32::from_le_bytes)?;

    let mut neurons = Vec::with_capacity(n);
    for i in 0..n {
        let bad = |what: &str, v: u8| {
            ReadError::Corrupt(format!("neuron {i} has unknown {what} code {v}"))
        };
        neurons.push(Neuron {
            body_id: body_id[i],
            type_name: type_name[i],
            superclass: superclass[i],
            class: class[i],
            side: Side::from_u8(side[i]).ok_or_else(|| bad("side", side[i]))?,
            nt_consensus: Transmitter::from_u8(nt_consensus[i])
                .ok_or_else(|| bad("transmitter", nt_consensus[i]))?,
            nt_predicted: Transmitter::from_u8(nt_predicted[i])
                .ok_or_else(|| bad("transmitter", nt_predicted[i]))?,
            nt_ground_truth: Transmitter::from_u8(nt_ground_truth[i])
                .ok_or_else(|| bad("transmitter", nt_ground_truth[i]))?,
            nt_confidence: nt_confidence[i],
            soma: match soma_present[i] {
                0 => None,
                1 => Some([soma_xyz[3 * i], soma_xyz[3 * i + 1], soma_xyz[3 * i + 2]]),
                v => return Err(bad("soma flag", v)),
            },
        });
    }

    let row_ptr = cur.column(n + 1, u32::from_le_bytes)?;
    let post = cur.column(e, u32::from_le_bytes)?;
    let weight = cur.column(e, u16::from_le_bytes)?;
    if cur.pos != bytes.len() {
        return Err(ReadError::Corrupt(format!(
            "{} trailing bytes",
            bytes.len() - cur.pos
        )));
    }

    let c = Connectome {
        strings,
        neurons,
        row_ptr,
        post,
        weight,
    };
    c.validate().map_err(ReadError::Corrupt)?;
    Ok(c)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], ReadError> {
        let end = self.pos.checked_add(len).ok_or(ReadError::Truncated)?;
        let s = self.bytes.get(self.pos..end).ok_or(ReadError::Truncated)?;
        self.pos = end;
        Ok(s)
    }

    fn u16(&mut self) -> Result<u16, ReadError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, ReadError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, ReadError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn column<T, const N: usize>(
        &mut self,
        count: usize,
        decode: fn([u8; N]) -> T,
    ) -> Result<Vec<T>, ReadError> {
        let raw = self.take(count.checked_mul(N).ok_or(ReadError::Truncated)?)?;
        Ok(raw
            .chunks_exact(N)
            .map(|c| decode(c.try_into().unwrap()))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectome::NO_STRING;

    fn sample() -> Connectome {
        let neuron = |body_id, type_name, side, nt, soma| Neuron {
            body_id,
            type_name,
            superclass: 2,
            class: NO_STRING,
            side,
            nt_consensus: nt,
            nt_predicted: nt,
            nt_confidence: 0.75,
            nt_ground_truth: Transmitter::Unclear,
            soma,
        };
        Connectome {
            strings: vec!["DNp01".into(), "TTMn".into(), "descending_neuron".into()],
            neurons: vec![
                neuron(
                    10001,
                    0,
                    Side::Right,
                    Transmitter::Acetylcholine,
                    Some([1000, -2, 300]),
                ),
                neuron(10010, 0, Side::Left, Transmitter::Acetylcholine, None),
                neuron(20000, 1, Side::Unknown, Transmitter::Gaba, Some([0, 0, 0])),
            ],
            // 0 -> 2 (w 40), 1 -> 0 (w 3), 1 -> 2 (w 41), 2 has no outputs.
            row_ptr: vec![0, 1, 3, 3],
            post: vec![2, 0, 2],
            weight: vec![40, 3, 41],
        }
    }

    #[test]
    fn roundtrip_preserves_everything() {
        let c = sample();
        c.validate().unwrap();
        let mut buf = Vec::new();
        write(&c, &mut buf).unwrap();
        let back = read(buf.as_slice()).unwrap();
        assert_eq!(back, c);
        assert_eq!(back.targets(1).collect::<Vec<_>>(), vec![(0, 3), (2, 41)]);
        assert_eq!(back.string(back.neurons[2].type_name), Some("TTMn"));
        assert_eq!(back.string(back.neurons[2].class), None);
    }

    #[test]
    fn rejects_bad_magic_and_truncation() {
        let mut buf = Vec::new();
        write(&sample(), &mut buf).unwrap();

        let mut bad = buf.clone();
        bad[0] = b'X';
        assert!(matches!(read(bad.as_slice()), Err(ReadError::BadMagic)));

        assert!(matches!(
            read(&buf[..buf.len() - 1]),
            Err(ReadError::Truncated)
        ));

        let mut trailing = buf.clone();
        trailing.push(0);
        assert!(matches!(
            read(trailing.as_slice()),
            Err(ReadError::Corrupt(_))
        ));
    }

    #[test]
    fn neurons_of_types_matches_by_published_name() {
        let c = sample();
        assert_eq!(c.neurons_of_types(&["DNp01"]), vec![0, 1]);
        assert_eq!(c.neurons_of_types(&["TTMn", "no such type"]), vec![2]);
        assert_eq!(c.neurons_of_types::<&str>(&[]), Vec::<u32>::new());
        // The superclass string must not be mistaken for a type name.
        assert_eq!(
            c.neurons_of_types(&["descending_neuron"]),
            Vec::<u32>::new()
        );
    }

    #[test]
    fn validate_catches_unsorted_targets() {
        let mut c = sample();
        c.post = vec![2, 2, 0];
        assert!(c.validate().unwrap_err().contains("strictly increasing"));
    }
}
