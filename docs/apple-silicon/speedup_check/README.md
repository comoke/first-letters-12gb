# Parked speed-up check (not part of the release claims)

What was tested: the BN fold plus GPU-batched stitch, an experimental fast path whose code is not included in this repository (so `mac/run_ahn.py` here has no `--fast` option, and `mac_logs/chain2.sh` can't be rerun from it). It ran as a full-window
A-hn s0 run with `run_ahn.py --fast both`, on 2026-11-16 at 22:32, with the B0 native window, batch 1 and fp16 as the reference.

**Outcome:** one of seven parts finished before the Mac slept. The run was then stopped with SIGTERM; finished parts are kept,
so it can resume.

**Speed:** part 1 (2,640 tiles) took 483 s, or 5.5 tiles/s (`s0fast_full.log`). The same part of the plain runs, with the Mac
idle, took 552–556 s for s1 and s2 (`../mac_logs/s1_full.log`, `../mac_logs/s2_full.log`). That is about 1.15× faster. It comes
from one part and is not an interleaved A/B; the queued A/B never ran.

**Exactness:** compared on rows 6–195 of the window, which only part 1's tiles cover (`s0fast_part1_compare.json`):

| | r | max abs diff | same 8-bit level |
|---|---|---|---|
| vs the 4070's CUDA s0 | 0.999997 | 3.6e-3 | 94.4 % |
| vs the plain Mac s0 | 0.999997 | 3.8e-3 | 94.5 % |

The plain Mac run gives r 0.99999996 and max 6.3e-4. The fast variant passes the r ≥ 0.999 gate, but it moves the map about
6× more than the device change does. The likely cause is that folding BN changes where the fp16 rounding falls.

**Decision:** kept out of the release text. A 1.15× gain does not justify a larger departure from the reference maps.

**Peak memory of the fast run:** 2.15 GB MPS driver (vs 3.47 GB plain) and 1.46 GB RSS.
