# Evidence for the Apple Silicon appendix (`apple-silicon.md`)

Every number in the Apple Silicon appendix comes from a file in this folder.
- **What's here:** logs and comparison outputs only. The winner's maps and weights are CC BY-NC and are not included.
- **Path edits:** home-directory paths were shortened to `~/` and nothing else was edited.
- **Window:** the B0 native window, render rows 1516–2915 and columns 2038–7638. That is 1,399 × 5,600 px of the claim render, about 5.85 cm².
- **Hardware:** MacBook Pro M5 Pro (24 GB, macOS 27.0.1, build 26A434) and the PC's RTX 4070.

## Exactness (Mac vs the 4070's CUDA maps)
| number | file |
|---|---|
| TRACE r 0.999999, max 2.4e-3, 99.4 % same u8, 0 NaN mismatches | `compare/trace.json` (key `cuda`) |
| A-hn s0 / s1 / s2 / decoy r, max diff, u8 agreement, NaN mismatches | `compare/s0.json`, `s1.json`, `s2.json`, `decoy.json` |
| Fused A, product and decoy product: r, max diff, top-1 % Jaccard (window and claim box; vs the 4070 and vs shipped) | `compare/fused.json` (written by `mac/fuse_compare.py`); fusion inputs and outputs by sha256 in `compare/fuse_mac.json` |
| The 4070's fused maps vs shipped, on the claim box | `compare/fused_4070_vs_shipped_box.txt` |

## Timing
| number | file |
|---|---|
| Mac TRACE 443.8 s compute, 444.9 s wall | `mac_logs/trace_native_timing.json` (`tr: 443.8`); `mac_logs/trace_native.log` (last line, `wall_s`) |
| 4070 TRACE 260.2 s compute | `pc_4070_logs/timing.json` (`tr: 260.2`); `pc_4070_logs/trace.log` (`tr 260.2s`) |
| 4070 TRACE 307 s stage wall (container included) | `pc_4070_logs/native_chain.log` (`TRACE exit=0 oom=false (307s)`) |
| Mac A-hn walls: s0 4,395.8 s, s1 3,550.9 s, s2 3,630.8 s, decoy 5,365.5 s; per-part rates | `mac_logs/{s0,s1,s2,decoy}_full.log` (last line `wall_s`; `part k/7` lines); start and end times in `mac_logs/members_chain.log` |
| 4070 A-hn walls: 907 / 905 / 906 / 905 s | `pc_4070_logs/native_chain.log` (`pass 1 exit=0 ... (907s)` etc.); `pc_4070_logs/b0_score.json` (`stage_wall_seconds`) |
| 4070 5.36 cm²/GPU-h, area 5.8536 cm² | `pc_4070_logs/b0_score.json` |
| Mac 1.21 cm²/GPU-h | 5.8536 cm² / ((444.9 + 4,395.8 + 3,550.9 + 3,630.8 + 5,365.5) s / 3600) |
| Mac fusion time (seconds, CPU) | `mac_logs/fuse_and_compare.log` |

## Memory
| number | file |
|---|---|
| A-hn 3.47 GB MPS driver + 0.93 GB RSS | `mac_logs/{s0,s1,s2,decoy}_full.log` (last line: `mps_driver_peak_gb`, `max_rss_gb`) |
| TRACE 2.31 GB MPS driver + 2.03 GB RSS | `mac_logs/trace_native.log` (last line: `mps_driver_gb`, `max_rss_gb`) |

## Run recipes
`mac_logs/run_s0_full.sh` (s0) and `mac_logs/chain2.sh` (s1, s2 and decoy, then the speed-up check). `*_full.meta.json` holds each member's `infer.py` sidecar: ROI, frame, grid origin, checkpoint sha256, tile count and Mac settings.
