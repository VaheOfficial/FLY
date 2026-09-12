//! Leaky integrate-and-fire dynamics over a [`Connectome`], on the CPU.
//!
//! This is the reference implementation: small, exact, and easy to test. The
//! GPU version must reproduce it. Model and default constants follow Shiu et
//! al. 2024 (*A Drosophila computational brain model reveals sensorimotor
//! processing*, Nature), the only whole-brain fly model validated against
//! behaviour so far.
//!
//! Per neuron, with membrane potential `v` and synaptic drive `g` (both in mV):
//!
//! ```text
//! dv/dt = (v_rest - v + g) / tau_m        (frozen while refractory)
//! dg/dt = -g / tau_syn                    (frozen while refractory)
//! ```
//!
//! A presynaptic spike from neuron `i` arriving at `j` after `delay` adds
//! `sign(i) * w_syn * synapse_count(i, j)` to `g[j]`. When `v` crosses
//! `v_thresh` the neuron spikes, `v` is reset, and integration pauses for the
//! refractory period. Increments to `g` still land during that pause.
//!
//! Integration is the exact solution of the linear system over one step, so
//! it is stable at any step size; only spike timing is quantised to `dt`.

mod cpu;
mod drive;
#[cfg(test)]
pub(crate) mod test_net;

pub use cpu::CpuSim;
pub use drive::{PoissonDrive, SplitMix64};

use crate::connectome::{Connectome, Transmitter};

/// Model constants. Times in ms, potentials in mV.
#[derive(Debug, Clone, PartialEq)]
pub struct LifParams {
    pub v_rest: f32,
    pub v_thresh: f32,
    pub v_reset: f32,
    pub tau_m: f32,
    pub tau_syn: f32,
    pub refractory: f32,
    /// Drive added to `g` per synapse per presynaptic spike.
    pub w_syn: f32,
    /// Conduction plus synaptic delay.
    pub delay: f32,
    /// Integration step.
    pub dt: f32,
}

impl Default for LifParams {
    /// Shiu et al. 2024 values.
    fn default() -> Self {
        Self {
            v_rest: -52.0,
            v_thresh: -45.0,
            v_reset: -52.0,
            tau_m: 20.0,
            tau_syn: 5.0,
            refractory: 2.2,
            w_syn: 0.275,
            delay: 1.8,
            dt: 0.1,
        }
    }
}

/// Per-step constants derived from [`LifParams`], shared by the CPU and GPU
/// backends so both integrate identically.
///
/// The membrane and drive form a linear system whose one-step solution is
/// `v' = v_rest + (v - v_rest) * decay_m + g * coupling` and
/// `g' = g * decay_syn`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepCoefficients {
    pub decay_m: f32,
    pub decay_syn: f32,
    pub coupling: f32,
    pub delay_steps: usize,
    pub refractory_steps: u32,
}

impl StepCoefficients {
    pub fn new(p: &LifParams) -> Self {
        assert!(p.dt > 0.0, "dt must be positive");
        assert!(
            (p.tau_m - p.tau_syn).abs() > 1e-6,
            "tau_m and tau_syn must differ for the exact integrator"
        );
        let decay_m = (-p.dt / p.tau_m).exp();
        let decay_syn = (-p.dt / p.tau_syn).exp();
        Self {
            decay_m,
            decay_syn,
            coupling: p.tau_syn / (p.tau_syn - p.tau_m) * (decay_syn - decay_m),
            delay_steps: (p.delay / p.dt).round().max(1.0) as usize,
            refractory_steps: (p.refractory / p.dt).round() as u32,
        }
    }
}

/// Maps a neuron's transmitter to the sign of its output synapses.
///
/// The default follows the fly literature: acetylcholine excites; GABA and
/// glutamate inhibit (glutamate acts on GluCl chloride channels in the fly
/// central nervous system); histamine, the photoreceptor transmitter, inhibits
/// via HisCl channels; the monoamines are treated as excitatory, as in Shiu et
/// al. Neurons with no usable prediction are treated as excitatory, the
/// majority class.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignPolicy {
    pub glutamate_inhibits: bool,
    pub histamine_inhibits: bool,
    pub unclear_sign: f32,
}

impl Default for SignPolicy {
    fn default() -> Self {
        Self {
            glutamate_inhibits: true,
            histamine_inhibits: true,
            unclear_sign: 1.0,
        }
    }
}

impl SignPolicy {
    pub fn sign(&self, nt: Transmitter) -> f32 {
        match nt {
            Transmitter::Acetylcholine
            | Transmitter::Dopamine
            | Transmitter::Octopamine
            | Transmitter::Serotonin => 1.0,
            Transmitter::Gaba => -1.0,
            Transmitter::Glutamate => {
                if self.glutamate_inhibits {
                    -1.0
                } else {
                    1.0
                }
            }
            Transmitter::Histamine => {
                if self.histamine_inhibits {
                    -1.0
                } else {
                    1.0
                }
            }
            Transmitter::Unclear => self.unclear_sign,
        }
    }

    /// Sign of every neuron's output, preferring the consensus prediction and
    /// falling back to the per-neuron prediction when the consensus is unclear.
    pub fn per_neuron(&self, net: &Connectome) -> Vec<f32> {
        net.neurons
            .iter()
            .map(|n| {
                let nt = if n.nt_consensus != Transmitter::Unclear {
                    n.nt_consensus
                } else {
                    n.nt_predicted
                };
                self.sign(nt)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_policy_defaults() {
        let s = SignPolicy::default();
        assert_eq!(s.sign(Transmitter::Acetylcholine), 1.0);
        assert_eq!(s.sign(Transmitter::Gaba), -1.0);
        assert_eq!(s.sign(Transmitter::Glutamate), -1.0);
        assert_eq!(s.sign(Transmitter::Histamine), -1.0);
        assert_eq!(s.sign(Transmitter::Dopamine), 1.0);
        assert_eq!(s.sign(Transmitter::Unclear), 1.0);
    }
}
