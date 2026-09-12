//! Continuous sensory drive onto a population, adjustable while running.

use super::population::PopulationId;
use crate::backend::Simulator;
use crate::sim::PoissonDrive;

/// Poisson events onto every neuron of one population. The rate can change
/// at any time; zero switches the channel off without removing it.
pub struct Stimulus {
    pub population: PopulationId,
    drive: PoissonDrive,
}

impl Stimulus {
    pub fn new(population: PopulationId, neurons: Vec<u32>, rate_hz: f32, seed: u64) -> Self {
        Self {
            population,
            drive: PoissonDrive::shiu(neurons, rate_hz, seed),
        }
    }

    pub fn rate_hz(&self) -> f32 {
        self.drive.rate_hz
    }

    pub fn set_rate_hz(&mut self, rate_hz: f32) {
        self.drive.rate_hz = rate_hz.max(0.0);
    }

    /// Deliver this step's events.
    pub fn apply(&mut self, sim: &mut dyn Simulator) {
        if self.drive.rate_hz > 0.0 {
            self.drive.apply(sim);
        }
    }
}
