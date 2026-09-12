//! One interface over both simulation backends, and the startup choice
//! between them: GPU when an adapter fits, CPU otherwise.

use crate::connectome::Connectome;
use crate::gpu::buffers::ConnectomeBuffers;
use crate::gpu::{GpuContext, GpuSim, GpuUnavailable};
use crate::sim::{CpuSim, LifParams, SignPolicy};

/// What the desktop shell needs from a running nervous system.
///
/// Calls are batch friendly: `inject` and `advance` may only queue work, and
/// nothing is guaranteed to have executed until `take_spike_counts`, which is
/// the one synchronising call. The GPU backend uses this to keep the host
/// thread idle between frames.
pub trait Simulator {
    fn params(&self) -> &LifParams;
    fn neuron_count(&self) -> usize;
    /// Human-readable backend description, e.g. the GPU adapter name.
    fn backend_name(&self) -> String;
    /// External synaptic drive in mV, landing on the next step.
    fn inject(&mut self, neuron: u32, delta_g: f32);
    /// Advance one step of `params().dt` milliseconds.
    fn advance(&mut self);
    /// Start executing whatever `advance` queued. Never blocks.
    fn flush(&mut self);
    /// Spikes per neuron since the previous call, then reset. Blocks until
    /// all queued steps have run.
    fn take_spike_counts(&mut self) -> Vec<u32>;
}

/// Which backend to try first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preference {
    /// GPU if one is usable, otherwise CPU.
    GpuThenCpu,
    CpuOnly,
}

/// Open a GPU simulator on a context the host already created, e.g. one
/// shared with a window's renderer.
pub fn open_on<'a>(
    ctx: GpuContext,
    net: &'a Connectome,
    params: LifParams,
    signs: SignPolicy,
) -> Box<dyn Simulator + 'a> {
    Box::new(GpuSim::new(ctx, net, params, signs))
}

/// Open a simulator over `net`, honouring `preference`. Returns the
/// simulator and, if the GPU was tried and rejected, why.
pub fn open<'a>(
    net: &'a Connectome,
    params: LifParams,
    signs: SignPolicy,
    preference: Preference,
) -> (Box<dyn Simulator + 'a>, Option<GpuUnavailable>) {
    if preference == Preference::GpuThenCpu {
        match GpuContext::new(ConnectomeBuffers::largest_buffer_bytes(net)) {
            Ok(ctx) => return (Box::new(GpuSim::new(ctx, net, params, signs)), None),
            Err(why) => return (Box::new(CpuSim::new(net, params, signs)), Some(why)),
        }
    }
    (Box::new(CpuSim::new(net, params, signs)), None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::test_net::{ACH, net};

    #[test]
    fn cpu_only_never_touches_the_gpu() {
        let c = net(&[ACH, ACH], &[(0, 1, 200)]);
        let (mut sim, rejected) = open(
            &c,
            LifParams::default(),
            SignPolicy::default(),
            Preference::CpuOnly,
        );
        assert!(rejected.is_none());
        assert_eq!(sim.backend_name(), "cpu");
        sim.inject(0, 100.0);
        for _ in 0..200 {
            sim.advance();
        }
        sim.flush();
        let counts = sim.take_spike_counts();
        assert!(counts[0] > 0 && counts[1] > 0);
        assert_eq!(
            sim.take_spike_counts(),
            vec![0, 0],
            "counts reset after taking"
        );
    }

    #[test]
    fn gpu_then_cpu_always_yields_a_working_simulator() {
        let c = net(&[ACH, ACH], &[(0, 1, 200)]);
        let (mut sim, _) = open(
            &c,
            LifParams::default(),
            SignPolicy::default(),
            Preference::GpuThenCpu,
        );
        sim.inject(0, 100.0);
        for _ in 0..200 {
            sim.advance();
        }
        sim.flush();
        assert!(
            sim.take_spike_counts()[1] > 0,
            "backend {}",
            sim.backend_name()
        );
    }
}
