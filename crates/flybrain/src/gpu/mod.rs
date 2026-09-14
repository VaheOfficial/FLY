//! GPU backend on `wgpu` compute.
//!
//! The GPU is the primary simulation backend; [`crate::sim::CpuSim`] is the
//! reference it must match and the fallback when no adapter is available.
//! Nothing here touches windowing: the context is headless.

pub mod buffers;
pub mod sim;

pub use sim::GpuSim;

use wgpu::{Adapter, Device, Queue};

/// How many storage buffers the simulation kernels bind at once. WebGPU's
/// minimum is 8; desktop GPUs offer far more, and a device that cannot give
/// us this many falls back to the CPU.
pub const STORAGE_BUFFERS_NEEDED: u32 = 12;

/// A headless device and queue, plus what we learned about the adapter.
pub struct GpuContext {
    pub device: Device,
    pub queue: Queue,
    pub adapter_info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
}

/// Why no GPU context could be created. Each case is a reason to fall back to
/// the CPU, not an error the user needs to fix.
#[derive(Debug)]
pub enum GpuUnavailable {
    NoAdapter,
    /// The adapter exists but cannot bind enough storage buffers or a large
    /// enough buffer for the connection table.
    LimitsTooLow(String),
    DeviceRequestFailed(wgpu::RequestDeviceError),
}

impl std::fmt::Display for GpuUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no compatible GPU adapter"),
            Self::LimitsTooLow(why) => write!(f, "GPU limits too low: {why}"),
            Self::DeviceRequestFailed(e) => write!(f, "GPU device request failed: {e}"),
        }
    }
}

impl std::error::Error for GpuUnavailable {}

impl GpuContext {
    /// Pick the highest-performance adapter and open a device sized for the
    /// simulation. `min_buffer_bytes` is the largest single buffer we will
    /// need, so a too-small adapter is rejected up front.
    pub fn new(min_buffer_bytes: u64) -> Result<Self, GpuUnavailable> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|_| GpuUnavailable::NoAdapter)?;

        let limits = required_limits(&adapter, min_buffer_bytes)?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("flybrain"),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(GpuUnavailable::DeviceRequestFailed)?;

        Ok(Self {
            device,
            queue,
            adapter_info: adapter.get_info(),
            limits,
        })
    }

    /// Wrap a device the host already opened, e.g. one that also presents to
    /// a window, so the simulation and the renderer share one GPU queue.
    /// The device must have been requested with at least [`required_limits`].
    pub fn from_parts(adapter: &Adapter, device: Device, queue: Queue) -> Self {
        Self {
            device,
            queue,
            adapter_info: adapter.get_info(),
            limits: adapter.limits(),
        }
    }

    /// Submit and block until the GPU has finished everything queued so far.
    pub fn wait_idle(&self) {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
    }
}

/// The limits we ask the device for: WebGPU defaults, raised where the
/// simulation needs more, and checked against what the adapter offers.
pub fn required_limits(
    adapter: &Adapter,
    min_buffer_bytes: u64,
) -> Result<wgpu::Limits, GpuUnavailable> {
    let offered = adapter.limits();
    if offered.max_storage_buffers_per_shader_stage < STORAGE_BUFFERS_NEEDED {
        return Err(GpuUnavailable::LimitsTooLow(format!(
            "{} storage buffers per stage, need {STORAGE_BUFFERS_NEEDED}",
            offered.max_storage_buffers_per_shader_stage
        )));
    }
    let offered_buffer = offered
        .max_buffer_size
        .min(offered.max_storage_buffer_binding_size);
    if offered_buffer < min_buffer_bytes {
        return Err(GpuUnavailable::LimitsTooLow(format!(
            "largest buffer {offered_buffer} bytes, need {min_buffer_bytes}"
        )));
    }
    let defaults = wgpu::Limits::default();
    Ok(wgpu::Limits {
        max_storage_buffers_per_shader_stage: STORAGE_BUFFERS_NEEDED,
        max_storage_buffer_binding_size: min_buffer_bytes
            .max(defaults.max_storage_buffer_binding_size),
        max_buffer_size: min_buffer_bytes.max(defaults.max_buffer_size),
        ..defaults
    })
}

/// GPU tests run one at a time: several test threads opening devices on the
/// same adapter at once has deadlocked in the driver.
#[cfg(test)]
static GPU_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Hold this for the whole of any test that touches the GPU.
#[cfg(test)]
pub(crate) fn gpu_test_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_TESTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A test's GPU context together with the serialising lock.
#[cfg(test)]
pub(crate) struct TestGpu {
    pub ctx: GpuContext,
    pub _serial: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
pub(crate) fn context_or_skip() -> Option<TestGpu> {
    let serial = gpu_test_lock();
    match GpuContext::new(1 << 20) {
        Ok(ctx) => Some(TestGpu {
            ctx,
            _serial: serial,
        }),
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_opens_and_reports_adapter() {
        let Some(TestGpu { ctx, _serial }) = context_or_skip() else {
            return;
        };
        assert!(!ctx.adapter_info.name.is_empty());
        assert!(ctx.limits.max_storage_buffers_per_shader_stage >= STORAGE_BUFFERS_NEEDED);
        ctx.wait_idle();
    }
}
