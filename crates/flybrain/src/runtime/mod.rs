//! Real-time driver for a simulator: advances the brain by wall-clock time,
//! feeds named sensory populations, and reports named motor populations.
//!
//! The host calls [`Runtime::tick`] once per frame with the elapsed real
//! time. The runtime steps the simulator as many times as that time covers
//! at the model step size, flushes, and reads spike counts back once. A
//! stalled frame is dropped rather than replayed, so the brain never races
//! to catch up.

pub mod population;
pub mod readout;
pub mod stimulus;

use std::time::Duration;

pub use population::{Population, PopulationId};

use crate::backend::Simulator;
use readout::Readout;
use stimulus::Stimulus;

/// Steps a single tick may run before the rest is dropped: 250 ms of
/// simulated time at a 1 ms step.
pub const DEFAULT_MAX_STEPS_PER_TICK: u32 = 250;

/// Firing rates are averaged over this much simulated time.
pub const DEFAULT_READOUT_WINDOW_MS: f32 = 200.0;

/// What one tick did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TickReport {
    pub steps: u32,
    /// Steps owed by elapsed time but skipped because of the per-tick cap.
    pub dropped_steps: u32,
}

pub struct Runtime<'a> {
    sim: Box<dyn Simulator + 'a>,
    populations: Vec<Population>,
    stimuli: Vec<Stimulus>,
    readout: Readout,
    /// Real time not yet turned into steps, in seconds.
    backlog_s: f64,
    max_steps_per_tick: u32,
    steps_run: u64,
    seed: u64,
}

impl<'a> Runtime<'a> {
    pub fn new(sim: Box<dyn Simulator + 'a>) -> Self {
        Self {
            sim,
            populations: Vec::new(),
            stimuli: Vec::new(),
            readout: Readout::new(0, DEFAULT_READOUT_WINDOW_MS),
            backlog_s: 0.0,
            max_steps_per_tick: DEFAULT_MAX_STEPS_PER_TICK,
            steps_run: 0,
            seed: 0x5EED,
        }
    }

    pub fn with_max_steps_per_tick(mut self, steps: u32) -> Self {
        self.max_steps_per_tick = steps.max(1);
        self
    }

    pub fn with_readout_window_ms(mut self, window_ms: f32) -> Self {
        self.readout = Readout::new(self.populations.len(), window_ms);
        self
    }

    pub fn backend_name(&self) -> String {
        self.sim.backend_name()
    }

    /// Simulated time elapsed, in ms.
    pub fn sim_time_ms(&self) -> f32 {
        self.steps_run as f32 * self.sim.params().dt
    }

    /// Register a population for stimulation and readout.
    pub fn add_population(&mut self, population: Population) -> PopulationId {
        self.populations.push(population);
        self.readout = Readout::new(self.populations.len(), self.readout.window_ms());
        PopulationId(self.populations.len() - 1)
    }

    pub fn population(&self, id: PopulationId) -> &Population {
        &self.populations[id.0]
    }

    /// Open a Poisson channel onto a population. Returns its index in the
    /// stimulus list; use it with [`Self::set_rate_hz`].
    pub fn add_stimulus(&mut self, population: PopulationId, rate_hz: f32) -> usize {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let neurons = self.populations[population.0].neurons.clone();
        self.stimuli
            .push(Stimulus::new(population, neurons, rate_hz, self.seed));
        self.stimuli.len() - 1
    }

    pub fn set_rate_hz(&mut self, stimulus: usize, rate_hz: f32) {
        self.stimuli[stimulus].set_rate_hz(rate_hz);
    }

    pub fn stimulus_rate_hz(&self, stimulus: usize) -> f32 {
        self.stimuli[stimulus].rate_hz()
    }

    /// Mean firing rate of a population over the readout window, Hz per neuron.
    pub fn rate_hz(&self, population: PopulationId) -> f32 {
        self.readout
            .rate_hz(population.0, self.populations[population.0].len())
    }

    /// Advance by `elapsed` of real time.
    pub fn tick(&mut self, elapsed: Duration) -> TickReport {
        let dt_s = f64::from(self.sim.params().dt) / 1000.0;
        self.backlog_s += elapsed.as_secs_f64();
        let owed = (self.backlog_s / dt_s).floor() as u32;
        let steps = owed.min(self.max_steps_per_tick);
        // Dropped time is forgotten, not carried: a stall must not become a
        // burst of catch-up steps on the next frame.
        self.backlog_s -= f64::from(owed) * dt_s;

        for _ in 0..steps {
            for stimulus in &mut self.stimuli {
                stimulus.apply(self.sim.as_mut());
            }
            self.sim.advance();
        }
        if steps > 0 {
            self.sim.flush();
            self.steps_run += u64::from(steps);
            let counts = self.sim.take_spike_counts();
            self.readout
                .record(self.sim_time_ms(), &counts, &self.populations);
        }
        TickReport {
            steps,
            dropped_steps: owed - steps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Preference, open};
    use crate::sim::test_net::{ACH, net};
    use crate::sim::{LifParams, SignPolicy};

    fn runtime_over_chain<'a>(c: &'a crate::connectome::Connectome) -> Runtime<'a> {
        let params = LifParams {
            dt: 1.0,
            ..LifParams::default()
        };
        let (sim, _) = open(c, params, SignPolicy::default(), Preference::CpuOnly);
        Runtime::new(sim)
    }

    #[test]
    fn elapsed_time_becomes_steps_with_remainder_carried() {
        let c = net(&[ACH], &[]);
        let mut rt = runtime_over_chain(&c);
        assert_eq!(rt.tick(Duration::from_micros(2500)).steps, 2);
        // 0.5 ms carried; another 0.6 ms owes one step.
        assert_eq!(rt.tick(Duration::from_micros(600)).steps, 1);
        assert_eq!(rt.tick(Duration::from_micros(50)).steps, 0);
        assert!((rt.sim_time_ms() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn a_stall_is_capped_and_the_excess_dropped() {
        let c = net(&[ACH], &[]);
        let mut rt = runtime_over_chain(&c).with_max_steps_per_tick(10);
        let report = rt.tick(Duration::from_millis(100));
        assert_eq!(
            report,
            TickReport {
                steps: 10,
                dropped_steps: 90
            }
        );
        assert_eq!(
            rt.tick(Duration::from_millis(1)).steps,
            1,
            "no catch-up burst"
        );
    }

    #[test]
    fn stimulated_population_drives_its_target_population() {
        let c = net(&[ACH, ACH], &[(0, 1, 200)]);
        let mut rt = runtime_over_chain(&c).with_readout_window_ms(500.0);
        let source = rt.add_population(Population {
            name: "source".into(),
            neurons: vec![0],
        });
        let target = rt.add_population(Population {
            name: "target".into(),
            neurons: vec![1],
        });
        let channel = rt.add_stimulus(source, 100.0);
        for _ in 0..50 {
            rt.tick(Duration::from_millis(10));
        }
        assert!(rt.rate_hz(source) > 50.0, "source {}", rt.rate_hz(source));
        assert!(rt.rate_hz(target) > 10.0, "target {}", rt.rate_hz(target));

        rt.set_rate_hz(channel, 0.0);
        for _ in 0..100 {
            rt.tick(Duration::from_millis(10));
        }
        assert_eq!(rt.rate_hz(source), 0.0, "channel off, window drained");
        assert_eq!(rt.rate_hz(target), 0.0);
    }
}
