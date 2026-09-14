//! A single append-only file holding raw downloaded skeletons, so 166,700
//! small downloads do not become 166,700 files.
//!
//! Record layout, repeated: `body_id: u64`, `len: u32`, then `len` bytes of
//! the neuroglancer precomputed skeleton. A zero length records that the
//! server had no skeleton for that body, so a resumed fetch skips it.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::{Context, Result};

const HEADER_BYTES: usize = 8 + 4;

pub struct PackWriter {
    file: BufWriter<File>,
}

impl PackWriter {
    /// Open for appending, creating the file if needed.
    pub fn append(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("opening {} for append", path.display()))?;
        Ok(Self {
            file: BufWriter::with_capacity(1 << 20, file),
        })
    }

    pub fn write(&mut self, body_id: u64, bytes: &[u8]) -> Result<()> {
        self.file.write_all(&body_id.to_le_bytes())?;
        self.file.write_all(&(bytes.len() as u32).to_le_bytes())?;
        self.file.write_all(bytes)?;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.file.flush().context("flushing skeleton pack")
    }
}

/// Body ids already present in a pack, tolerating a truncated final record
/// from an interrupted run (the file is cut back to the last whole record).
pub fn fetched_ids(path: &Path) -> Result<HashSet<u64>> {
    let mut ids = HashSet::new();
    let Ok(file) = File::open(path) else {
        return Ok(ids);
    };
    let mut reader = BufReader::new(file);
    let mut good_end = 0u64;
    loop {
        let mut header = [0u8; HEADER_BYTES];
        match reader.read_exact(&mut header) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let body_id = u64::from_le_bytes(header[..8].try_into().unwrap());
        let len = u32::from_le_bytes(header[8..].try_into().unwrap());
        if reader.seek(SeekFrom::Current(i64::from(len))).is_err() {
            break;
        }
        let end = reader.stream_position()?;
        if end > file_len(path)? {
            break;
        }
        good_end = end;
        ids.insert(body_id);
    }
    drop(reader);
    if good_end < file_len(path)? {
        OpenOptions::new()
            .write(true)
            .open(path)?
            .set_len(good_end)
            .context("truncating partial record")?;
    }
    Ok(ids)
}

/// Iterate `(body_id, bytes)` over every whole record; a torn final record
/// from a fetch still in progress is ignored. Empty bytes mean no skeleton.
pub fn read_all(path: &Path, mut each: impl FnMut(u64, &[u8]) -> Result<()>) -> Result<()> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut bytes = Vec::new();
    loop {
        let mut header = [0u8; HEADER_BYTES];
        match reader.read_exact(&mut header) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e.into()),
        }
        let body_id = u64::from_le_bytes(header[..8].try_into().unwrap());
        let len = u32::from_le_bytes(header[8..].try_into().unwrap()) as usize;
        bytes.resize(len, 0);
        match reader.read_exact(&mut bytes) {
            Ok(()) => each(body_id, &bytes)?,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e.into()),
        }
    }
}

fn file_len(path: &Path) -> Result<u64> {
    Ok(std::fs::metadata(path)?.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_resume_after_truncation() {
        let dir = std::env::temp_dir().join(format!("flyskel-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.pack");

        let mut w = PackWriter::append(&path).unwrap();
        w.write(7, b"hello").unwrap();
        w.write(8, b"").unwrap();
        w.write(9, b"xyz").unwrap();
        w.flush().unwrap();
        drop(w);

        // Chop the last record in half, as an interrupted download would.
        let len = std::fs::metadata(&path).unwrap().len();
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(len - 2)
            .unwrap();

        let ids = fetched_ids(&path).unwrap();
        assert_eq!(ids, HashSet::from([7, 8]));
        // The torn record is gone; a resumed run appends cleanly after it.
        let mut w = PackWriter::append(&path).unwrap();
        w.write(9, b"xyz").unwrap();
        w.flush().unwrap();
        drop(w);
        assert_eq!(fetched_ids(&path).unwrap(), HashSet::from([7, 8, 9]));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
