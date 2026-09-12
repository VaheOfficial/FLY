//! The nervous system as the shell sees it: load, open a backend on the
//! shared GPU device, register the populations the desktop talks to.
//!
//! Sensory and motor populations are named after fly anatomy here; the
//! mapping from desktop events onto them lives in `senses`.

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

use crate::senses::Vibration;

/// Unitary synaptic weight calibrated for MaleCNS (see docs/validation.md).
const W_SYN_MV: f32 = 0.1;
/// Real-time step. The GPU runs the full nervous system 20x faster than
/// this needs.
const DT_MS: f32 = 1.0;

/// Leg chordotonal organ sensory neurons: every MaleCNS type with subclass
/// "chordotonal organ", all entering through leg nerves. They report
/// vibration of the surface the fly stands on.
const LEG_CHORDOTONAL_TYPES: &[&str] = &[
    "SNpp17",
    "SNpp18",
    "SNpp22",
    "SNpp39",
    "SNpp40",
    "SNpp41",
    "SNpp42",
    "SNpp43",
    "SNpp44",
    "SNpp46",
    "SNpp47",
    "SNpp48",
    "SNpp49",
    "SNpp50",
    "SNpp51",
    "SNpp56",
    "SNpp57",
    "SNpp58",
    "SNpp59",
    "SNpp60",
    "SApp23",
    "SApp23,SNpp56",
];

/// Johnston's organ groups A and B: the sound-sensitive auditory neurons
/// that carry courtship song. Groups C to E sense gravity and wind.
const JOHNSTONS_ORGAN_SOUND_TYPES: &[&str] = &[
    "JO-A1",
    "JO-A2",
    "JO-A3",
    "JO-A4",
    "JO-A-unclear",
    "JO-B1_a",
    "JO-B1_b",
    "JO-B1_c",
    "JO-B2",
    "JO-B3",
    "JO-B4_a",
    "JO-B4_b",
    "JO-B-unclear",
];

pub fn load_connectome(path: &Path) -> Result<Connectome> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    io::read(BufReader::new(file)).with_context(|| format!("reading {}", path.display()))
}

/// The running brain plus handles to the populations the shell reads and drives.
pub struct Brain<'a> {
    runtime: Runtime<'a>,
    pub leg_chordotonal: PopulationId,
    pub johnstons_organ: PopulationId,
    /// pC1 / P1: the courtship arousal population.
    pub courtship_arousal: PopulationId,
    substrate_channel: usize,
    song_channel: usize,
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
        let leg_chordotonal = runtime.add_population(Population::from_types(
            net,
            "leg chordotonal organs",
            LEG_CHORDOTONAL_TYPES,
        ));
        let johnstons_organ = runtime.add_population(Population::from_types(
            net,
            "Johnston's organ sound neurons",
            JOHNSTONS_ORGAN_SOUND_TYPES,
        ));
        let courtship_arousal =
            runtime.add_population(Population::from_types(net, "pC1", &pc1_types(net)));
        let substrate_channel = runtime.add_stimulus(leg_chordotonal, 0.0);
        let song_channel = runtime.add_stimulus(johnstons_organ, 0.0);
        Self {
            runtime,
            leg_chordotonal,
            johnstons_organ,
            courtship_arousal,
            substrate_channel,
            song_channel,
        }
    }

    pub fn backend_name(&self) -> String {
        self.runtime.backend_name()
    }

    pub fn population_size(&self, id: PopulationId) -> usize {
        self.runtime.population(id).len()
    }

    /// Deliver this frame's sensory drive.
    pub fn feel(&mut self, vibration: Vibration) {
        self.runtime
            .set_rate_hz(self.substrate_channel, vibration.substrate_hz);
        self.runtime
            .set_rate_hz(self.song_channel, vibration.song_hz);
    }

    pub fn rate_hz(&self, population: PopulationId) -> f32 {
        self.runtime.rate_hz(population)
    }

    pub fn tick(&mut self, elapsed: Duration) -> TickReport {
        self.runtime.tick(elapsed)
    }
}

/// Every published pC1 subtype (`pC1_1a`, `pC1_16b`, ...). P1, the male
/// courtship command population, is the fruitless-expressing part of pC1.
fn pc1_types(net: &Connectome) -> Vec<String> {
    net.strings
        .iter()
        .filter(|s| s.starts_with("pC1_"))
        .cloned()
        .collect()
}
