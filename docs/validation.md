# Circuit validation

How to check that the simulation reproduces known fly biology. Every
experiment uses the `stimulate` example: it drives named cell types with
Poisson input and reports who fires.

```
cargo run --release -p flybrain --example stimulate -- --stim <types> --report <types> --w-syn 0.1
```

Defaults: 100 Hz Poisson events onto each stimulated neuron, 1000 ms, 0.1 ms
step. `--w-syn 0.1` is the unitary synaptic weight calibrated for MaleCNS (see
below). Cell type names are the MaleCNS `type` column.

## Unitary weight

The model has one free parameter, the drive per synapse per spike. Shiu et al.
2024 calibrated it to 0.275 mV on FlyWire so that 100 Hz sugar stimulation
gave about 80% of the maximal MN9 rate. MaleCNS has roughly 2.5x the synapses
per connection, and the same criterion lands at 0.1 mV here:

| w_syn | sugar to MN9 | fraction of brain firing |
|---|---|---|
| 0.08 | silent | 0.1% |
| 0.10 | ~80% of max | 3.8% |
| 0.15 | saturated | 13% (runaway) |
| 0.275 | saturated | 20% (runaway) |

## Step size and speed

The integrator is exact per step, so the step only quantises spike timing,
the 1.8 ms delay and the 2.2 ms refractory period. Both circuits below still
behave correctly at a 1 ms step, with fewer neurons recruited than at 0.1 ms.
Single-threaded CPU cost for 1000 ms of the whole nervous system:

| step | wall time | relative to real time |
|---|---|---|
| 0.1 ms | ~1.3 s | 0.8x |
| 0.5 ms | ~0.3 s | 3.3x |
| 1.0 ms | ~0.15 s | 6.7x |

The GPU backend (the default; `--cpu` forces the reference) reproduces the
CPU spike-for-spike on small networks and statistically on the full brain
(same MN9 rates, active counts within 0.1%). On an RTX 4090, after about
0.45 s of setup (upload plus shader compilation), 1000 ms costs ~0.45 s at a
0.1 ms step and ~0.05 s at a 1 ms step, with the host thread only queueing
work. Per-step cost is fixed dispatch overhead, so batch size barely matters.

## Feeding: sugar drives proboscis extension

Labellar sugar GRNs are types `LB3b` and `LB3c` (Gr64f). Bitter GRNs are
`LB1a` to `LB1d` (Gr33a). Water GRNs are `LB3a` (ppk28). The proboscis
extension motor neuron is `MN9`.

```
--stim LB3b,LB3c --report MN9      # MN9 fires, ~340 Hz left
--stim LB1a,LB1b,LB1c,LB1d --report MN9   # MN9 silent
```

Open questions: the right MN9 responds far weaker than the left (about 40 Hz
versus 340 Hz) although the GRNs are balanced 17 per side; water GRNs do not
drive MN9 here although Shiu et al. reported they do on FlyWire.

## Escape: looming drives the Giant Fiber and the jump motor neuron

Looming detectors `LPLC2` and `LC4` project to the Giant Fiber `DNp01`, which
drives the tergotrochanteral (jump) motor neuron `TTMn`. `LC4` is also known
to target the descending neurons `DNp02`, `DNp04`, and `DNp11`. The
small-object tracking neurons `LC10a` project elsewhere (anterior optic
tubercle) and are the control.

```
--stim LC4 --report DNp01,TTMn,DNp02,DNp04,DNp11    # all fire; TTMn ~300 Hz right
--stim LPLC2 --report DNp01,TTMn,DNp04              # DNp01 and DNp04 saturate, TTMn fires
--stim LC10a --report DNp01,TTMn                    # both silent; AOTU neurons fire instead
```

Limitation: the Giant Fiber's outputs to `TTMn` and to the peripherally
synapsing interneuron `PSI` are largely electrical in the real fly. Gap
junctions are not in the connectome, so `PSI` stays silent and `TTMn` is
driven only by the chemical component.
