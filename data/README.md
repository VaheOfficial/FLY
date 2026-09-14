# data

Raw connectome downloads and the packed binaries produced by
`tools/connectome-convert`. Everything in this directory except this file is
git-ignored; datasets are fetched, not committed.

## MaleCNS v1.0

Fetch the three tables the converter needs (about 1.1 GB, checksum-verified):

```powershell
.\tools\fetch-malecns.ps1
```

They land in `data/malecns-v1.0/`. Re-running the script skips files that are
already present and verified.

| File | Contents |
|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | One row per body: cell type, superclass, class, soma side, status |
| `body-neurotransmitters-male-cns-v1.0.feather` | Per-body predicted and consensus neurotransmitter |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | One row per connected body pair: presynaptic body, postsynaptic body, synapse count |

Source: Janelia FlyEM, `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`,
licensed CC-BY 4.0.

## Skeletons for the brain viewer

The viewer draws neuron morphologies from the low-resolution precomputed
skeletons in the same bucket. Two steps, both from the repo root:

```
cargo run --release -p connectome-convert -- fetch-skeletons
cargo run --release -p connectome-convert -- build-skeletons
```

The first downloads all 166,700 skeletons (about 5.4 GB, ten minutes on a
fast connection) into `data/malecns-v1.0/skeletons-precomputed.pack`; it is
resumable, so rerun it after an interruption. The second prunes terminal
twigs shorter than 20 µm, decimates branches to 8 µm spacing, and writes
`data/malecns-v1.0.flyskel` (about 127 MB, 11 million vertices). Both
settings are flags on `build-skeletons`. Without the morphology file flypet
still runs and draws cell bodies only.
