//! The nervous system as the shell sees it: load, open a backend on the
//! shared GPU device, register the populations the desktop talks to.
//!
//! Sensory and motor populations are named after fly anatomy here; the
//! mapping from desktop events onto them belongs to the input layer.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use flybrain::backend::{self, Preference};
use flybrain::connectome::{Connectome, io};
use flybrain::gpu::GpuContext;
use flybrain::runtime::{Population, PopulationId, Runtime, TickReport};
use flybrain::sim::{LifParams, SignPolicy};

/// Unitary synaptic weight calibrated for MaleCNS (see docs/validation.md).
const W_SYN_MV: f32 = 0.1;
/// Real-time step. The GPU runs the full nervous system 20x faster than
/// this needs.
const DT_MS: f32 = 1.0;

pub fn load_connectome(path: &Path) -> Result<Connectome> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    io::read(BufReader::new(file)).with_context(|| format!("reading {}", path.display()))
}

/// The running brain plus handles to the populations the shell reads and drives.
pub struct Brain<'a> {
    runtime: Runtime<'a>,
    pub sugar_grns: PopulationId,
    pub proboscis_motor: PopulationId,
    sugar_channel: usize,
}

impl<'a> Brain<'a> {
    /// Open on `gpu` if given (shared with the renderer), else fall back per
    /// `backend::open`.
    pub fn new(net: &'a Connectome, gpu: Option<GpuContext>) -> Self {
        let params = LifParams {
            dt: DT_MS,
            w_syn: W_SYN_MV,
            ..LifParams::default()
        };
        let sim = match gpu {
            Some(ctx) => backend::open_on(ctx, net, params, SignPolicy::default()),
            None => backend::open(net, params, SignPolicy::default(), Preference::CpuOnly).0,
        };
        let mut runtime = Runtime::new(sim);
        let sugar_grns = runtime.add_population(Population::from_types(
            net,
            "labellar sugar GRNs",
            &["LB3b", "LB3c"],
        ));
        let proboscis_motor = runtime.add_population(Population::from_types(net, "MN9", &["MN9"]));
        let sugar_channel = runtime.add_stimulus(sugar_grns, 0.0);
        Self {
            runtime,
            sugar_grns,
            proboscis_motor,
            sugar_channel,
        }
    }

    pub fn backend_name(&self) -> String {
        self.runtime.backend_name()
    }

    /// Taste of sugar on the labellum, as a Poisson rate onto the sugar GRNs.
    pub fn set_sugar_hz(&mut self, rate_hz: f32) {
        self.runtime.set_rate_hz(self.sugar_channel, rate_hz);
    }

    pub fn rate_hz(&self, population: PopulationId) -> f32 {
        self.runtime.rate_hz(population)
    }

    pub fn tick(&mut self, elapsed: Duration) -> TickReport {
        self.runtime.tick(elapsed)
    }
}
