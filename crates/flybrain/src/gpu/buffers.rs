//! Uploading the connectome to the GPU and reading buffers back.

use bytemuck::Pod;
use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferUsages};

use super::GpuContext;
use crate::connectome::Connectome;

/// The static connection table, resident on the GPU for the life of a sim.
pub struct ConnectomeBuffers {
    /// `neuron_count + 1` row offsets into `post` and `weight`.
    pub row_ptr: Buffer,
    /// Postsynaptic neuron index per connection.
    pub post: Buffer,
    /// Synapse counts, two `u16` packed per `u32` (low half first).
    pub weight_pairs: Buffer,
    pub neuron_count: u32,
    pub edge_count: u32,
}

impl ConnectomeBuffers {
    /// Bytes of the largest buffer this connectome needs, for sizing limits.
    pub fn largest_buffer_bytes(net: &Connectome) -> u64 {
        (net.edge_count().max(net.neuron_count() + 1) as u64) * 4
    }

    pub fn upload(ctx: &GpuContext, net: &Connectome) -> Self {
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        Self {
            row_ptr: init_buffer(ctx, "row_ptr", &net.row_ptr, storage),
            post: init_buffer(ctx, "post", &net.post, storage),
            weight_pairs: init_buffer(ctx, "weight_pairs", &pack_u16_pairs(&net.weight), storage),
            neuron_count: net.neuron_count() as u32,
            edge_count: net.edge_count() as u32,
        }
    }
}

/// Two `u16` per `u32`, low half first; odd tails are zero padded.
pub fn pack_u16_pairs(values: &[u16]) -> Vec<u32> {
    values
        .chunks(2)
        .map(|pair| u32::from(pair[0]) | (u32::from(*pair.get(1).unwrap_or(&0)) << 16))
        .collect()
}

/// Create a buffer holding `data`.
pub fn init_buffer<T: Pod>(
    ctx: &GpuContext,
    label: &str,
    data: &[T],
    usage: BufferUsages,
) -> Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage,
        })
}

/// Create a zero-filled buffer of `len` elements of `T`.
pub fn zeroed_buffer<T: Pod>(
    ctx: &GpuContext,
    label: &str,
    len: usize,
    usage: BufferUsages,
) -> Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (len * std::mem::size_of::<T>()) as u64,
        usage,
        mapped_at_creation: false,
    })
}

/// Copy a buffer to the host. Blocks until the copy completes. The source
/// must have been created with `COPY_SRC`.
pub fn download<T: Pod>(ctx: &GpuContext, source: &Buffer) -> Vec<T> {
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("download staging"),
        size: source.size(),
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    ctx.queue.submit([encoder.finish()]);

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).expect("map callback receiver dropped");
    });
    ctx.wait_idle();
    receiver
        .recv()
        .expect("map callback never ran")
        .expect("buffer map failed");
    let data = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped range")).to_vec();
    staging.unmap();
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::context_or_skip;
    use crate::sim::test_net::{ACH, GABA, net};

    #[test]
    fn packs_u16_pairs_low_half_first() {
        assert_eq!(pack_u16_pairs(&[1, 2, 3]), vec![1 | (2 << 16), 3]);
        assert_eq!(pack_u16_pairs(&[]), Vec::<u32>::new());
    }

    #[test]
    fn upload_then_download_roundtrips() {
        let Some(ctx) = context_or_skip() else { return };
        let c = net(&[ACH, GABA, ACH], &[(0, 1, 5), (0, 2, 7), (2, 1, 9)]);
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let post = init_buffer(&ctx, "post", &c.post, usage);
        let weights = init_buffer(&ctx, "w", &pack_u16_pairs(&c.weight), usage);
        assert_eq!(download::<u32>(&ctx, &post), c.post);
        assert_eq!(download::<u32>(&ctx, &weights), pack_u16_pairs(&c.weight));

        let buffers = ConnectomeBuffers::upload(&ctx, &c);
        assert_eq!(buffers.neuron_count, 3);
        assert_eq!(buffers.edge_count, 3);
        assert_eq!(buffers.row_ptr.size(), 4 * 4);
    }
}
