//! `inspect`: schema, row count and head of a Feather table.

use std::path::Path;

use anyhow::{Context, Result};
use arrow::util::pretty::print_batches;

use crate::feather;

pub fn run(path: &Path, rows: usize) -> Result<()> {
    let reader = feather::open(path)?;

    println!("{}", path.display());
    println!("schema:");
    for field in reader.schema().fields() {
        println!(
            "  {:<28} {:?}{}",
            field.name(),
            field.data_type(),
            if field.is_nullable() {
                ""
            } else {
                " (non-null)"
            }
        );
    }

    let mut total_rows = 0usize;
    let mut batches = 0usize;
    let mut head = None;
    for batch in reader {
        let batch = batch.context("reading record batch")?;
        if head.is_none() {
            head = Some(batch.slice(0, rows.min(batch.num_rows())));
        }
        total_rows += batch.num_rows();
        batches += 1;
    }
    println!(
        "rows: {total_rows}  (in {batches} batch{})",
        if batches == 1 { "" } else { "es" }
    );

    if let Some(head) = head {
        println!("first {} rows:", head.num_rows());
        print_batches(&[head])?;
    }
    Ok(())
}
