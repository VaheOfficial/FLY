# FLY

A desktop pet fruit fly driven by a simulation of a real *Drosophila* connectome.

Rust, single binary, GPU compute and rendering through `wgpu`.

## Layout

| Path | What |
|---|---|
| `crates/flybrain` | The nervous-system simulation. Portable; no windowing. |
| `crates/flypet` | The desktop shell: transparent window, input capture, sprite rendering. |
| `tools/connectome-convert` | Build-time converter from published connectome tables to the packed binary the sim loads. |
| `experiments/` | Throwaway explorations, excluded from the workspace. |
| `data/` | Fetched datasets and generated binaries. Git-ignored. |
| `docs/` | Engineering notes and decision records. |

## Build

```
cargo build
```
