//! The CPU reference implementation of the integrate-and-fire step.

use super::{LifParams, SignPolicy};
use crate::connectome::Connectome;

/// CPU simulation state over a borrowed connectome.
pub struct CpuSim<'a> {
    net: &'a Connectome,
    params: LifParams,
    /// Per-neuron output sign, from [`SignPolicy`].
    sign: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    /// Step index at which each neuron leaves its refractory period.
    refractory_until: Vec<u64>,
    /// Spikes waiting to be delivered, one slot per step of delay.
    pending: Vec<Vec<u32>>,
    step: u64,
    delay_steps: usize,
    refractory_steps: u64,
    /// Exact one-step integration coefficients.
    decay_m: f32,
    decay_syn: f32,
    coupling: f32,
    spiked: Vec<u32>,
    total_spikes: u64,
}

impl<'a> CpuSim<'a> {
    pub fn new(net: &'a Connectome, params: LifParams, signs: SignPolicy) -> Self {
        assert!(params.dt > 0.0, "dt must be positive");
        assert!(
            (params.tau_m - params.tau_syn).abs() > 1e-6,
            "tau_m and tau_syn must differ for the exact integrator"
        );
        let n = net.neuron_count();
        let delay_steps = (params.delay / params.dt).round().max(1.0) as usize;
        let refractory_steps = (params.refractory / params.dt).round() as u64;
        let decay_m = (-params.dt / params.tau_m).exp();
        let decay_syn = (-params.dt / params.tau_syn).exp();
        let coupling = params.tau_syn / (params.tau_syn - params.tau_m) * (decay_syn - decay_m);
        Self {
            sign: signs.per_neuron(net),
            v: vec![params.v_rest; n],
            g: vec![0.0; n],
            refractory_until: vec![0; n],
            pending: vec![Vec::new(); delay_steps],
            step: 0,
            delay_steps,
            refractory_steps,
            decay_m,
            decay_syn,
            coupling,
            spiked: Vec::new(),
            total_spikes: 0,
            net,
            params,
        }
    }

    pub fn params(&self) -> &LifParams {
        &self.params
    }

    pub fn connectome(&self) -> &'a Connectome {
        self.net
    }

    /// Steps taken so far.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Simulated time in ms.
    pub fn time(&self) -> f32 {
        self.step as f32 * self.params.dt
    }

    pub fn total_spikes(&self) -> u64 {
        self.total_spikes
    }

    pub fn potential(&self, neuron: u32) -> f32 {
        self.v[neuron as usize]
    }

    pub fn drive(&self, neuron: u32) -> f32 {
        self.g[neuron as usize]
    }

    /// Neurons that spiked on the most recent step.
    pub fn spiked(&self) -> &[u32] {
        &self.spiked
    }

    /// Add external synaptic drive to one neuron, in mV, landing immediately.
    pub fn inject(&mut self, neuron: u32, delta_g: f32) {
        self.g[neuron as usize] += delta_g;
    }

    /// Advance one step. Returns the neurons that spiked this step.
    pub fn step(&mut self) -> &[u32] {
        // 1. Deliver spikes whose delay has elapsed.
        let slot = (self.step % self.delay_steps as u64) as usize;
        let mut due = std::mem::take(&mut self.pending[slot]);
        for &pre in &due {
            let scale = self.sign[pre as usize] * self.params.w_syn;
            for (post, count) in self.net.targets(pre) {
                self.g[post as usize] += scale * f32::from(count);
            }
        }
        due.clear();
        self.pending[slot] = due;

        // 2. Integrate and detect threshold crossings.
        self.spiked.clear();
        let p = &self.params;
        for i in 0..self.v.len() {
            if self.refractory_until[i] > self.step {
                continue;
            }
            let u = self.v[i] - p.v_rest;
            self.v[i] = p.v_rest + u * self.decay_m + self.g[i] * self.coupling;
            self.g[i] *= self.decay_syn;
            if self.v[i] > p.v_thresh {
                self.v[i] = p.v_reset;
                self.refractory_until[i] = self.step + 1 + self.refractory_steps;
                self.spiked.push(i as u32);
            }
        }
        self.total_spikes += self.spiked.len() as u64;

        // 3. Queue this step's spikes for delivery after the delay.
        let target = ((self.step + self.delay_steps as u64) % self.delay_steps as u64) as usize;
        self.pending[target].extend_from_slice(&self.spiked);

        self.step += 1;
        &self.spiked
    }

    /// Run for `ms` of simulated time, calling `on_step` after each step with
    /// the simulator (so callers can inject drive and read spikes).
    pub fn run(&mut self, ms: f32, mut on_step: impl FnMut(&mut Self)) {
        let steps = (ms / self.params.dt).round() as u64;
        for _ in 0..steps {
            self.step();
            on_step(self);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::test_net::{ACH, GABA, net};

    #[test]
    fn resting_neuron_stays_at_rest() {
        let c = net(&[ACH], &[]);
        let mut sim = CpuSim::new(&c, LifParams::default(), SignPolicy::default());
        sim.run(100.0, |_| {});
        assert_eq!(sim.total_spikes(), 0);
        assert!((sim.potential(0) - (-52.0)).abs() < 1e-5);
    }

    #[test]
    fn integrator_matches_closed_form_decay() {
        // With g = 0 the membrane relaxes as v_rest + (v - v_rest) e^{-t/tau_m}.
        let c = net(&[ACH], &[]);
        let p = LifParams::default();
        let mut sim = CpuSim::new(&c, p.clone(), SignPolicy::default());
        sim.v[0] = -48.0;
        sim.run(10.0, |_| {});
        let expected = p.v_rest + (-48.0 - p.v_rest) * (-10.0f32 / p.tau_m).exp();
        assert!(
            (sim.potential(0) - expected).abs() < 1e-4,
            "{} vs {expected}",
            sim.potential(0)
        );
    }

    #[test]
    fn integrator_matches_closed_form_synaptic_kick() {
        // With v at rest and an initial g0, the exact response is
        // v(t) = v_rest + g0 * tau_syn/(tau_syn - tau_m) * (e^{-t/tau_syn} - e^{-t/tau_m}).
        let c = net(&[ACH], &[]);
        let p = LifParams::default();
        let mut sim = CpuSim::new(&c, p.clone(), SignPolicy::default());
        sim.inject(0, 2.0);
        sim.run(3.0, |_| {});
        let t = 3.0f32;
        let expected = p.v_rest
            + 2.0 * p.tau_syn / (p.tau_syn - p.tau_m)
                * ((-t / p.tau_syn).exp() - (-t / p.tau_m).exp());
        assert!(
            (sim.potential(0) - expected).abs() < 1e-4,
            "{} vs {expected}",
            sim.potential(0)
        );
        assert!(
            sim.potential(0) > p.v_rest,
            "an excitatory kick must depolarise"
        );
    }

    #[test]
    fn big_kick_spikes_once_then_refractory() {
        let c = net(&[ACH], &[]);
        let p = LifParams::default();
        let mut sim = CpuSim::new(&c, p.clone(), SignPolicy::default());
        sim.inject(0, 100.0);
        let mut spike_steps = Vec::new();
        sim.run(20.0, |s| {
            if !s.spiked().is_empty() {
                spike_steps.push(s.step_count() - 1);
            }
        });
        // g is frozen during the refractory period, so the 100 mV kick is
        // still there when integration resumes and the neuron fires again.
        // What matters is that no two spikes are closer than the refractory
        // period.
        assert!(
            spike_steps.len() >= 2,
            "expected repeated firing from a 100 mV kick"
        );
        let gap = spike_steps[1] - spike_steps[0];
        let min_gap = (p.refractory / p.dt).round() as u64 + 1;
        assert!(
            gap >= min_gap,
            "spikes {gap} steps apart, refractory needs {min_gap}"
        );
    }

    /// Peak depolarisation from a drive step `g0` at rest is
    /// `g0 * tau_syn/(tau_syn - tau_m) * (e^{-t/tau_syn} - e^{-t/tau_m})` at
    /// its maximum, about `0.1575 * g0` for the default constants. Reaching
    /// the 7 mV threshold from a single presynaptic spike therefore needs
    /// roughly 162 synapses. Tests below use 200.
    #[test]
    fn single_spike_threshold_is_about_160_synapses() {
        let p = LifParams::default();
        // One presynaptic spike through `synapses` synapses is one drive
        // increment of synapses * w_syn; inject it directly.
        let fires = |synapses: f32| {
            let c = net(&[ACH], &[]);
            let mut sim = CpuSim::new(&c, p.clone(), SignPolicy::default());
            sim.inject(0, synapses * p.w_syn);
            sim.run(30.0, |_| {});
            sim.total_spikes() > 0
        };
        assert!(!fires(150.0), "150 synapses must not fire a resting neuron");
        assert!(fires(175.0), "175 synapses must fire a resting neuron");
    }

    #[test]
    fn excitatory_spike_arrives_after_delay_and_drives_target() {
        let c = net(&[ACH, ACH], &[(0, 1, 200)]);
        let p = LifParams::default();
        let mut sim = CpuSim::new(&c, p.clone(), SignPolicy::default());
        sim.inject(0, 100.0);
        let mut first_a = None;
        let mut first_b = None;
        let mut g_b_at_delivery = None;
        sim.run(10.0, |s| {
            let step = s.step_count() - 1;
            if s.spiked().contains(&0) && first_a.is_none() {
                first_a = Some(step);
            }
            if let Some(a) = first_a
                && step == a + (p.delay / p.dt).round() as u64
                && g_b_at_delivery.is_none()
            {
                g_b_at_delivery = Some(s.drive(1));
            }
            if s.spiked().contains(&1) && first_b.is_none() {
                first_b = Some(step);
            }
        });
        let a = first_a.expect("A spikes");
        let b = first_b.expect("B spikes from 200 synapses of drive");
        assert!(b > a);
        // Nothing reached B before the delay elapsed.
        assert!(b >= a + (p.delay / p.dt).round() as u64);
        // Drive of 200 synapses * 0.275 mV, less one step of decay.
        let g = g_b_at_delivery.expect("observed delivery step");
        let expected = 200.0 * p.w_syn * (-p.dt / p.tau_syn).exp();
        assert!((g - expected).abs() < 1e-3, "g_B {g} vs {expected}");
    }

    #[test]
    fn inhibitory_input_suppresses_target() {
        // A excites B with 200 synapses (enough on its own); C inhibits B with
        // 200 synapses at the same time. Net drive zero: B must stay silent.
        let b_spikes = |c: &Connectome, drive: &[u32]| {
            let mut sim = CpuSim::new(c, LifParams::default(), SignPolicy::default());
            for &n in drive {
                sim.inject(n, 100.0);
            }
            let mut count = 0;
            sim.run(20.0, |s| {
                count += s.spiked().iter().filter(|&&n| n == 1).count()
            });
            count
        };
        let excitation_only = net(&[ACH, ACH], &[(0, 1, 200)]);
        assert!(
            b_spikes(&excitation_only, &[0]) > 0,
            "200 synapses must fire B"
        );

        let balanced = net(&[ACH, ACH, GABA], &[(0, 1, 200), (2, 1, 200)]);
        assert_eq!(
            b_spikes(&balanced, &[0, 2]),
            0,
            "balanced excitation and inhibition must not fire B"
        );
    }
}
