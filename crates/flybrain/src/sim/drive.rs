//! External input: Poisson spike trains onto chosen neurons.

use super::cpu::CpuSim;

/// Independent Poisson spike trains onto a set of neurons, the stimulation
/// protocol of Shiu et al. Each Poisson event adds `weight` mV of drive.
pub struct PoissonDrive {
    pub neurons: Vec<u32>,
    pub rate_hz: f32,
    pub weight: f32,
    rng: SplitMix64,
}

impl PoissonDrive {
    /// Drive per Poisson event in Shiu et al.: 250 times their unitary weight
    /// of 0.275 mV. Far above threshold, so every event is a guaranteed spike.
    /// Fixed in absolute terms so that sweeping the network's `w_syn` does not
    /// also change how hard the stimulus hits.
    pub const SHIU_EVENT_MV: f32 = 250.0 * 0.275;

    /// Shiu et al. stimulation: Poisson events at `rate_hz` (they used
    /// 10 to 200 Hz), each delivering [`Self::SHIU_EVENT_MV`].
    pub fn shiu(neurons: Vec<u32>, rate_hz: f32, seed: u64) -> Self {
        Self::new(neurons, rate_hz, Self::SHIU_EVENT_MV, seed)
    }

    pub fn new(neurons: Vec<u32>, rate_hz: f32, weight: f32, seed: u64) -> Self {
        Self {
            neurons,
            rate_hz,
            weight,
            rng: SplitMix64::new(seed),
        }
    }

    /// Draw this step's events: the neurons that receive a Poisson spike.
    pub fn sample(&mut self, dt: f32) -> impl Iterator<Item = u32> + '_ {
        let p_event = self.rate_hz * dt * 1e-3;
        let rng = &mut self.rng;
        self.neurons
            .iter()
            .copied()
            .filter(move |_| rng.next_f32() < p_event)
    }

    /// Apply one step of drive.
    pub fn apply(&mut self, sim: &mut CpuSim<'_>) {
        let weight = self.weight;
        for n in self.sample(sim.params().dt) {
            sim.inject(n, weight);
        }
    }
}

/// Small deterministic PRNG so the crate stays dependency-free.
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::test_net::{ACH, net};
    use crate::sim::{LifParams, SignPolicy};

    #[test]
    fn poisson_drive_events_arrive_at_the_requested_rate() {
        let p = LifParams::default();
        let mut drive = PoissonDrive::shiu(vec![0, 1, 2, 3], 150.0, 7);
        let steps = (10_000.0 / p.dt) as usize; // 10 s
        let events: usize = (0..steps).map(|_| drive.sample(p.dt).count()).sum();
        let rate_hz = events as f32 / 4.0 / 10.0;
        // 150 Hz per neuron; 4 neurons over 10 s gives 6000 expected events
        // with a standard deviation of about 77, so 5% is a generous bound.
        assert!((142.0..=158.0).contains(&rate_hz), "measured {rate_hz} Hz");
    }

    #[test]
    fn shiu_drive_makes_stimulated_neuron_fire_hard() {
        // Each event injects 250 unitary weights, far above threshold, and g
        // is frozen through the refractory period, so a driven neuron bursts.
        // Its rate must exceed the event rate and stay under the refractory
        // ceiling of 1000 / (refractory + dt) Hz.
        let c = net(&[ACH], &[]);
        let p = LifParams::default();
        let mut sim = CpuSim::new(&c, p.clone(), SignPolicy::default());
        let mut drive = PoissonDrive::shiu(vec![0], 150.0, 7);
        sim.run(2000.0, |s| drive.apply(s));
        let rate_hz = sim.total_spikes() as f32 / 2.0;
        let ceiling = 1000.0 / (p.refractory + p.dt);
        assert!(
            rate_hz > 150.0 && rate_hz < ceiling,
            "measured {rate_hz} Hz, ceiling {ceiling}"
        );
    }
}
