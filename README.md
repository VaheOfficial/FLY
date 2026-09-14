# FLY

A desktop pet fruit fly driven by a simulation of a real *Drosophila* connectome.

Rust, single binary, GPU compute and rendering through `wgpu`.

## Layout

| Path | What |
|---|---|
| `crates/flybrain` | The nervous-system simulation. Portable; no windowing. GPU backend with CPU fallback. |
| `crates/flypet` | The desktop shell: transparent window, input capture, rendering. |
| `tools/connectome-convert` | Build-time converter from published connectome tables to the packed binary the sim loads. |
| `tools/fetch-malecns.ps1` | Downloads and verifies the MaleCNS tables the converter reads. |
| `experiments/` | Throwaway explorations, excluded from the workspace. |
| `data/` | Fetched datasets and generated binaries. Git-ignored. |
| `docs/` | Engineering documentation. |

## Build and run

```
cargo build
.\tools\fetch-malecns.ps1
cargo run --release -p connectome-convert -- build
cargo run --release -p flybrain --example stimulate -- --stim LB3b,LB3c --report MN9 --w-syn 0.1
```

The last command drives the labellar sugar-sensing neurons and reports the
proboscis motor neuron firing. See `docs/validation.md` for the circuits the
model is checked against and how fast it runs.

The pet itself, with its brain viewer window:

```
cargo run --release -p flypet
```

Typing anywhere on the desktop shakes the surface under the fly. Add
`--sugar-hz 100` to feed it sugar, or `--snapshot brain.png` to save the
viewer to a file after three seconds and exit. To see full neuron
morphologies in the viewer rather than cell bodies, fetch and build the
skeletons as described in `data/README.md`.
