//! `counts`: value tally of one string column, with optional row filters.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use arrow::array::Array;

use crate::feather;

pub fn run(
    path: &Path,
    column: &str,
    top: usize,
    nonnull: Option<&str>,
    r#where: Option<(&str, &str)>,
) -> Result<()> {
    let reader = feather::open(path)?;
    let schema = reader.schema();
    let idx = feather::column_index(&schema, column)?;
    let filter_idx = nonnull
        .map(|c| feather::column_index(&schema, c))
        .transpose()?;
    let where_idx = r#where
        .map(|(c, v)| feather::column_index(&schema, c).map(|i| (i, v)))
        .transpose()?;

    let mut tally: HashMap<String, usize> = HashMap::new();
    let mut nulls = 0usize;
    let mut total = 0usize;
    for batch in reader {
        let batch = batch.context("reading record batch")?;
        let col = feather::str_column(&batch, idx)?;
        let where_col = where_idx
            .map(|(i, v)| feather::str_column(&batch, i).map(|c| (c, v)))
            .transpose()?;
        for i in 0..batch.num_rows() {
            if let Some(f) = filter_idx
                && batch.column(f).is_null(i)
            {
                continue;
            }
            if let Some((ref c, v)) = where_col
                && c.get(i) != Some(v)
            {
                continue;
            }
            total += 1;
            match col.get(i) {
                Some(s) => *tally.entry(s.to_owned()).or_default() += 1,
                None => nulls += 1,
            }
        }
    }

    let mut rows: Vec<(String, usize)> = tally.into_iter().collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    println!(
        "{column}: {} distinct values, {nulls} nulls, {total} rows",
        rows.len()
    );
    for (value, n) in rows.iter().take(top) {
        println!("{n:>10}  {value}");
    }
    if rows.len() > top {
        println!("       ...  ({} more)", rows.len() - top);
    }
    Ok(())
}
