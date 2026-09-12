//! Circuit probe: drive neurons of chosen cell types with Poisson input and
//! report who fires. This is how model behaviour gets checked against known
//! biology, e.g. sugar-sensing GRNs (`LB3b`, `LB3c`) should drive the
//! proboscis motor neuron `MN9`.
//!
//! ```text
//! cargo run --release -p flybrain --example stimulate -- \
//!     --stim LB3b,LB3c --report MN9 --rate 100 --ms 1000
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use flybrain::connectome::{Connectome, io};
use flybrain::sim::{CpuSim, LifParams, PoissonDrive, SignPolicy};

#[derive(Parser)]
#[command(about)]
struct Args {
    /// Packed connectome file.
    #[arg(long, default_value = "data/malecns-v1.0.flycnx")]
    connectome: PathBuf,
    /// Cell types to stimulate, comma separated.
    #[arg(long, value_delimiter = ',')]
    stim: Vec<String>,
    /// Cell types whose firing to report, comma separated.
    #[arg(long, value_delimiter = ',')]
    report: Vec<String>,
    /// Poisson event rate onto each stimulated neuron, Hz.
    #[arg(long, default_value_t = 100.0)]
    rate: f32,
    /// Simulated duration, ms.
    #[arg(long, default_value_t = 1000.0)]
    ms: f32,
    /// Integration step, ms.
    #[arg(long, default_value_t = 0.1)]
    dt: f32,
    /// Unitary synaptic weight, mV. Shiu et al. used 0.275.
    #[arg(long, default_value_t = 0.275)]
    w_syn: f32,
    /// RNG seed for the Poisson drive.
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// How many of the most active non-stimulated neurons to list.
    #[arg(long, default_value_t = 25)]
    top: usize,
}

fn main() {
    let args = Args::parse();
    let t0 = Instant::now();
    let file = File::open(&args.connectome).expect("open connectome");
    let net = io::read(BufReader::new(file)).expect("read connectome");
    println!(
        "loaded {} neurons, {} edges in {:.1?}",
        net.neuron_count(),
        net.edge_count(),
        t0.elapsed()
    );

    let stim = neurons_of_types(&net, &args.stim);
    if stim.is_empty() {
        eprintln!("no neurons match --stim {:?}", args.stim);
        std::process::exit(1);
    }
    println!(
        "stimulating {} neurons of types {:?}",
        stim.len(),
        args.stim
    );

    let params = LifParams {
        dt: args.dt,
        w_syn: args.w_syn,
        ..LifParams::default()
    };
    let mut sim = CpuSim::new(&net, params.clone(), SignPolicy::default());
    let mut drive = PoissonDrive::shiu(stim.clone(), args.rate, args.seed);

    let mut counts = vec![0u32; net.neuron_count()];
    let t1 = Instant::now();
    sim.run(args.ms, |s| {
        drive.apply(s);
        for &n in s.spiked() {
            counts[n as usize] += 1;
        }
    });
    let seconds = args.ms / 1000.0;
    println!(
        "ran {} ms in {:.1?}: {} spikes total",
        args.ms,
        t1.elapsed(),
        sim.total_spikes()
    );

    let active = counts.iter().filter(|&&c| c > 0).count();
    println!(
        "neurons that fired: {active} ({:.2}% of {})",
        100.0 * active as f32 / net.neuron_count() as f32,
        net.neuron_count()
    );

    let is_stim: Vec<bool> = {
        let mut v = vec![false; net.neuron_count()];
        for &n in &stim {
            v[n as usize] = true;
        }
        v
    };
    let mean_stim_hz = stim
        .iter()
        .map(|&n| counts[n as usize] as f32 / seconds)
        .sum::<f32>()
        / stim.len() as f32;
    println!("stimulated neurons fired at {mean_stim_hz:.1} Hz on average");

    for ty in &args.report {
        let members = neurons_of_types(&net, std::slice::from_ref(ty));
        println!("{ty}: {} neuron(s)", members.len());
        for &n in &members {
            let nr = &net.neurons[n as usize];
            println!(
                "  body {:<8} {:?} {:>6} spikes  {:>7.1} Hz",
                nr.body_id,
                nr.side,
                counts[n as usize],
                counts[n as usize] as f32 / seconds
            );
        }
    }

    let mut ranked: Vec<u32> = (0..net.neuron_count() as u32)
        .filter(|&n| !is_stim[n as usize] && counts[n as usize] > 0)
        .collect();
    ranked.sort_by_key(|&n| std::cmp::Reverse(counts[n as usize]));
    println!("most active non-stimulated neurons:");
    for &n in ranked.iter().take(args.top) {
        let nr = &net.neurons[n as usize];
        println!(
            "  {:>7.1} Hz  {:<16} body {:<8} {:?} {} {}",
            counts[n as usize] as f32 / seconds,
            net.string(nr.type_name).unwrap_or("?"),
            nr.body_id,
            nr.side,
            net.string(nr.superclass).unwrap_or("?"),
            nr.nt_consensus.name()
        );
    }

    let mut by_superclass: HashMap<&str, (usize, u64)> = HashMap::new();
    for n in 0..net.neuron_count() {
        if counts[n] > 0 && !is_stim[n] {
            let sc = net.string(net.neurons[n].superclass).unwrap_or("?");
            let e = by_superclass.entry(sc).or_default();
            e.0 += 1;
            e.1 += u64::from(counts[n]);
        }
    }
    let mut by_superclass: Vec<_> = by_superclass.into_iter().collect();
    by_superclass.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
    println!("active non-stimulated neurons by superclass:");
    for (sc, (n, spikes)) in by_superclass {
        println!("  {n:>7} neurons  {spikes:>9} spikes  {sc}");
    }
}

fn neurons_of_types(net: &Connectome, types: &[String]) -> Vec<u32> {
    (0..net.neuron_count() as u32)
        .filter(|&i| {
            net.string(net.neurons[i as usize].type_name)
                .is_some_and(|t| types.iter().any(|want| want == t))
        })
        .collect()
}
