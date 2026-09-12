//! Drive the brain in real time from a wall clock and watch a motor
//! population's rate, the way the desktop shell will.
//!
//! ```text
//! cargo run --release -p flybrain --example realtime -- --seconds 5
//! ```

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use flybrain::backend::{self, Preference};
use flybrain::connectome::io;
use flybrain::runtime::{Population, Runtime};
use flybrain::sim::{LifParams, SignPolicy};

#[derive(Parser)]
#[command(about)]
struct Args {
    #[arg(long, default_value = "data/malecns-v1.0.flycnx")]
    connectome: PathBuf,
    /// Real seconds to run.
    #[arg(long, default_value_t = 5.0)]
    seconds: f32,
    /// Target frame period, ms.
    #[arg(long, default_value_t = 16.0)]
    frame_ms: f32,
    /// Poisson rate onto the sugar GRNs, Hz.
    #[arg(long, default_value_t = 100.0)]
    rate: f32,
    #[arg(long, default_value_t = 1.0)]
    dt: f32,
    #[arg(long, default_value_t = 0.1)]
    w_syn: f32,
    #[arg(long)]
    cpu: bool,
}

fn main() {
    let args = Args::parse();
    let net = io::read(BufReader::new(
        File::open(&args.connectome).expect("open connectome"),
    ))
    .expect("read connectome");
    let params = LifParams {
        dt: args.dt,
        w_syn: args.w_syn,
        ..LifParams::default()
    };
    let preference = if args.cpu {
        Preference::CpuOnly
    } else {
        Preference::GpuThenCpu
    };
    let (sim, rejected) = backend::open(&net, params, SignPolicy::default(), preference);
    if let Some(why) = rejected {
        println!("gpu unavailable ({why}), using cpu");
    }
    let mut brain = Runtime::new(sim);
    println!("backend: {}", brain.backend_name());

    let sugar = brain.add_population(Population::from_types(
        &net,
        "sugar GRNs",
        &["LB3b", "LB3c"],
    ));
    let mn9 = brain.add_population(Population::from_types(&net, "MN9", &["MN9"]));
    println!(
        "{}: {} neurons, {}: {} neurons",
        brain.population(sugar).name,
        brain.population(sugar).len(),
        brain.population(mn9).name,
        brain.population(mn9).len()
    );
    brain.add_stimulus(sugar, args.rate);

    let frame = Duration::from_secs_f32(args.frame_ms / 1000.0);
    let start = Instant::now();
    let mut last = start;
    let mut next_print = start + Duration::from_secs(1);
    let mut dropped = 0u32;
    let mut frames = 0u32;
    let mut busy = Duration::ZERO;
    while start.elapsed().as_secs_f32() < args.seconds {
        let now = Instant::now();
        let work = Instant::now();
        let report = brain.tick(now - last);
        busy += work.elapsed();
        last = now;
        dropped += report.dropped_steps;
        frames += 1;
        if now >= next_print {
            next_print += Duration::from_secs(1);
            println!(
                "t={:4.1}s  sim={:7.0} ms  sugar {:6.1} Hz  MN9 {:6.1} Hz  busy {:4.1}%  dropped {dropped}",
                start.elapsed().as_secs_f32(),
                brain.sim_time_ms(),
                brain.rate_hz(sugar),
                brain.rate_hz(mn9),
                100.0 * busy.as_secs_f32() / start.elapsed().as_secs_f32(),
            );
        }
        std::thread::sleep(frame.saturating_sub(now.elapsed()));
    }
    println!(
        "{frames} frames, sim/real = {:.3}, host busy {:.1}%",
        brain.sim_time_ms() / 1000.0 / start.elapsed().as_secs_f32(),
        100.0 * busy.as_secs_f32() / start.elapsed().as_secs_f32()
    );
}
