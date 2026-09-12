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
