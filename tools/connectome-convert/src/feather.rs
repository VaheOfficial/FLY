//! Small helpers over Arrow IPC ("Feather v2") files.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail};
use arrow::array::{Array, ArrayRef, AsArray, Int64Array, StringArray};
use arrow::datatypes::{DataType, Int64Type, Schema};
use arrow::ipc::reader::FileReader;
use arrow::record_batch::RecordBatch;

/// Open a Feather file for streaming record batches.
pub fn open(path: &Path) -> Result<FileReader<File>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    FileReader::try_new(file, None)
        .with_context(|| format!("reading Arrow IPC header of {}", path.display()))
}

/// Index of a named column, with a readable error if it is missing.
pub fn column_index(schema: &Schema, name: &str) -> Result<usize> {
    schema
        .index_of(name)
        .with_context(|| format!("no column named {name:?}"))
}

/// A string column that may be plain Utf8 or dictionary-encoded Utf8.
pub enum StrColumn<'a> {
    Plain(&'a StringArray),
    Dict {
        keys: Vec<usize>,
        values: &'a StringArray,
        nulls: &'a ArrayRef,
    },
}

impl<'a> StrColumn<'a> {
    pub fn new(col: &'a ArrayRef, name: &str) -> Result<Self> {
        match col.data_type() {
            DataType::Utf8 => Ok(Self::Plain(col.as_string::<i32>())),
            DataType::Dictionary(_, value_type) if **value_type == DataType::Utf8 => {
                let dict = col.as_any_dictionary();
                Ok(Self::Dict {
                    keys: dict.normalized_keys(),
                    values: dict.values().as_string::<i32>(),
                    nulls: col,
                })
            }
            other => bail!("column {name:?} has type {other:?}, expected a string column"),
        }
    }

    pub fn get(&self, i: usize) -> Option<&'a str> {
        match self {
            Self::Plain(a) => (!a.is_null(i)).then(|| a.value(i)),
            Self::Dict {
                keys,
                values,
                nulls,
            } => (!nulls.is_null(i)).then(|| values.value(keys[i])),
        }
    }
}

/// Borrow a string column out of a batch.
pub fn str_column<'a>(batch: &'a RecordBatch, idx: usize) -> Result<StrColumn<'a>> {
    StrColumn::new(batch.column(idx), batch.schema_ref().field(idx).name())
}

/// Borrow an Int64 column out of a batch.
pub fn i64_column(batch: &RecordBatch, idx: usize) -> Result<&Int64Array> {
    let col = batch.column(idx);
    match col.data_type() {
        DataType::Int64 => Ok(col.as_primitive::<Int64Type>()),
        other => bail!(
            "column {:?} has type {other:?}, expected Int64",
            batch.schema_ref().field(idx).name()
        ),
    }
}
