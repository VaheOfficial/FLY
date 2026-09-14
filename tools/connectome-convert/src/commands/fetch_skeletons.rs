//! `fetch-skeletons`: download every neuron's low-resolution skeleton from
//! the MaleCNS bucket into one pack file. Resumable and parallel.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::malecns::{ANNOTATIONS, SKELETON_URL_PREFIX};
use crate::neurons::NeuronTable;
use crate::skeleton_pack::{PackWriter, fetched_ids};

/// Downloads in flight at once. The bucket serves small objects quickly;
/// this many keeps a home connection busy without tripping rate limits.
const WORKERS: usize = 32;
const ATTEMPTS: usize = 4;

pub fn run(data: &Path, out: &Path) -> Result<()> {
    let neurons = NeuronTable::load(&data.join(ANNOTATIONS))?;
    let done = fetched_ids(out)?;
    let todo: Vec<u64> = neurons
        .neurons
        .iter()
        .map(|n| n.body_id as u64)
        .filter(|id| !done.contains(id))
        .collect();
    println!(
        "{} neurons, {} already fetched, {} to go -> {}",
        neurons.len(),
        done.len(),
        todo.len(),
        out.display()
    );
    if todo.is_empty() {
        return Ok(());
    }

    let todo = Arc::new(todo);
    let next = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = mpsc::sync_channel::<(u64, Vec<u8>)>(WORKERS * 4);
    let workers: Vec<_> = (0..WORKERS)
        .map(|_| {
            let todo = Arc::clone(&todo);
            let next = Arc::clone(&next);
            let sender = sender.clone();
            std::thread::spawn(move || {
                let agent: ureq::Agent = ureq::Agent::config_builder()
                    .timeout_global(Some(Duration::from_secs(60)))
                    .build()
                    .into();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&body_id) = todo.get(i) else { break };
                    let bytes = fetch_with_retries(&agent, body_id);
                    if sender.send((body_id, bytes)).is_err() {
                        break;
                    }
                }
            })
        })
        .collect();
    drop(sender);

    let mut writer = PackWriter::append(out)?;
    let started = Instant::now();
    let mut written = 0usize;
    let mut missing = 0usize;
    let mut bytes_total = 0u64;
    for (body_id, bytes) in receiver {
        if bytes.is_empty() {
            missing += 1;
        }
        bytes_total += bytes.len() as u64;
        writer.write(body_id, &bytes)?;
        written += 1;
        if written.is_multiple_of(2000) {
            writer.flush()?;
            let rate = written as f32 / started.elapsed().as_secs_f32();
            println!(
                "{written}/{} fetched, {missing} missing, {:.1} MB, {rate:.0}/s, ~{:.0} s left",
                todo.len(),
                bytes_total as f64 / 1e6,
                (todo.len() - written) as f32 / rate.max(1.0)
            );
        }
    }
    writer.flush()?;
    for worker in workers {
        worker.join().expect("download worker panicked");
    }
    println!(
        "done: {written} fetched ({missing} with no skeleton), {:.1} MB in {:.0?}",
        bytes_total as f64 / 1e6,
        started.elapsed()
    );
    Ok(())
}

/// The skeleton bytes, or empty if the bucket has none for this body.
fn fetch_with_retries(agent: &ureq::Agent, body_id: u64) -> Vec<u8> {
    let url = format!("{SKELETON_URL_PREFIX}/{body_id}");
    for attempt in 1..=ATTEMPTS {
        match agent.get(&url).call() {
            Ok(mut response) => match response.body_mut().read_to_vec() {
                Ok(bytes) => return bytes,
                Err(e) => eprintln!("body {body_id}: read failed ({e}), attempt {attempt}"),
            },
            Err(ureq::Error::StatusCode(404)) => return Vec::new(),
            Err(e) => eprintln!("body {body_id}: {e}, attempt {attempt}"),
        }
        std::thread::sleep(Duration::from_millis(500 * attempt as u64));
    }
    eprintln!("body {body_id}: giving up after {ATTEMPTS} attempts; recorded as missing");
    Vec::new()
}
