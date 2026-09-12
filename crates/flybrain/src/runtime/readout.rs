//! Firing-rate estimates per population over a sliding window.

use std::collections::VecDeque;

use super::population::Population;

/// Spike totals per population, kept for the last `window_ms` of simulated
/// time. Rates are mean spikes per neuron per second.
pub struct Readout {
    window_ms: f32,
    /// Per population: `(sim_time_ms at end of tick, spikes in that tick)`.
    history: Vec<VecDeque<(f32, u32)>>,
}

impl Readout {
    pub fn new(population_count: usize, window_ms: f32) -> Self {
        assert!(window_ms > 0.0, "readout window must be positive");
        Self {
            window_ms,
            history: vec![VecDeque::new(); population_count],
        }
    }

    pub fn window_ms(&self) -> f32 {
        self.window_ms
    }

    /// Fold one tick's per-neuron spike counts into every population.
    pub fn record(&mut self, sim_time_ms: f32, counts: &[u32], populations: &[Population]) {
        for (history, population) in self.history.iter_mut().zip(populations) {
            let spikes = population.neurons.iter().map(|&n| counts[n as usize]).sum();
            history.push_back((sim_time_ms, spikes));
            let horizon = sim_time_ms - self.window_ms;
            while history.front().is_some_and(|&(t, _)| t <= horizon) {
                history.pop_front();
            }
        }
    }

    /// Mean firing rate of a population over the window, in Hz per neuron.
    pub fn rate_hz(&self, population: usize, neuron_count: usize) -> f32 {
        if neuron_count == 0 {
            return 0.0;
        }
        let spikes: u32 = self.history[population].iter().map(|&(_, s)| s).sum();
        spikes as f32 / (self.window_ms / 1000.0) / neuron_count as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn population(neurons: &[u32]) -> Population {
        Population {
            name: "p".into(),
            neurons: neurons.to_vec(),
        }
    }

    #[test]
    fn rate_is_spikes_per_neuron_per_second_over_the_window() {
        let pops = [population(&[0, 1])];
        let mut readout = Readout::new(1, 100.0);
        // Ten ticks of 10 ms, each neuron spiking once per tick: 100 Hz.
        for tick in 1..=10 {
            readout.record(tick as f32 * 10.0, &[1, 1, 7], &pops);
        }
        assert!((readout.rate_hz(0, 2) - 100.0).abs() < 1e-3);
    }

    #[test]
    fn old_ticks_fall_out_of_the_window() {
        let pops = [population(&[0])];
        let mut readout = Readout::new(1, 50.0);
        readout.record(10.0, &[5], &pops);
        for tick in 2..=10 {
            readout.record(tick as f32 * 10.0, &[0], &pops);
        }
        assert_eq!(
            readout.rate_hz(0, 1),
            0.0,
            "the burst at 10 ms is outside a 50 ms window ending at 100 ms"
        );
    }

    #[test]
    fn empty_population_has_zero_rate() {
        let readout = Readout::new(1, 100.0);
        assert_eq!(readout.rate_hz(0, 0), 0.0);
    }
}
