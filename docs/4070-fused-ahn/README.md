# The 4070's optional fused A-hn forward: evidence

The write-up is in [`../details.md`](../details.md#an-optional-fused-a-hn-forward-on-the-4070). Everything here ran on one RTX 4070 12 GB in the winner's image `fl-claim-a7:final` (PyTorch 2.14.1+cu130), on 2026-11-22.

| File | What it holds |
|---|---|
| `gate_cufuse.json` | The gate, from `pc/opt4070/ahn_gate.py`: each member's r, max and mean abs diff and finite/empty changes against the unmodified path's map on the B0 native and no-text windows, then the fused controls (no-text false-ink share, decoy overlaps, letter AP/AUC) for both. `"pass": true`. |
| `ahn_b0_cufuse.log` | The fused runs: one job per member on the native window, then the four members in one process on the no-text window. Each job's start and `O4070_DONE` lines carry its clock times; hn4_s0 took 10:22:05–10:31:03 (538 s). |
| `ahn_b0_stock.log` | The unmodified forward through the same runner, hn4_s0 on the native window: 11:12:00–11:27:09 (909 s). |
| `b0_finish_cufuse.log` | The per-member timings, the gate's output (`AHN_GATE_PASS`), and the unmodified same-runner map against the reference map: `identical True`. |
| `ahnbench_3.json` | The forward alone on 256 B0 tiles of hn4_s0 at batch 4 and 8: 19.32 tiles/s unmodified, 33.75 fused (batch 4), with r against the unmodified forward and a torch.profiler table of the fused forward. |
| `ahn_b0_cufuse_export_smoke.log` | The exported job scripts run once more, as documented, after the path change (below). |
| `smoke_cmp.py`, `smoke_finish.log` | That run's map against the reference map, compared as the gate compares hn4_s0 native, and the result (`"pass": true`). |

The timing runs were `MEMBERS=<member> WINDOWS=native GATE=0 bash pc/opt4070/job_ahn_b0.sh cufuse 4` (and `stock` for the baseline); the no-text run was `WINDOWS=notext GATE=0`, and the gate ran afterwards on the CPU. The job scripts `job_ahn_b0.sh`, `job_ahnbench.sh` and `dk_job.sh` are the ones those runs used, with the data and scratch paths now read from `pc/env.sh` instead of fixed to my machine. After that change, `job_ahn_b0.sh` and `dk_job.sh` ran once more from the exported tree, with the documented command `MEMBERS=hn4_s0 WINDOWS=native GATE=0 bash pc/opt4070/job_ahn_b0.sh cufuse 4`: `ahn_b0_cufuse_export_smoke.log` shows it finished, and `smoke_finish.log` holds the comparison by `smoke_cmp.py`, which compares the new map with the reference map the way the gate does. It gives r 0.9999983288735558, the gate's hn4_s0 native value to every digit, with max abs diff 0.0043, the same 7,749,169 finite pixels and no finite/empty changes. Its forward ran at the same 33 tiles/s in every part as the timed run; the whole job took 585 s, against 538 s, and it isn't used for any timing. `job_ahnbench.sh` was not run again. The runs were scheduled one at a time behind other GPU work on the same card, by wrappers that aren't included here.
