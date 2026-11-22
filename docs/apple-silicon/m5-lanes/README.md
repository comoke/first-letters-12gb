# Evidence for the faster A-hn path (`apple-silicon.md`, "A faster A-hn path")

Every number in that section comes from a file in this folder, or from the files of the main Apple Silicon appendix (one folder up), as the table below says.
- **What's here:** logs, run settings and comparison outputs only. The member maps, the winner's weights and the Core ML packages derived from those weights are CC BY-NC, so they are not included.
- **Path edits:** home-directory paths were shortened to `~/`, and the PC's data directory to `$VESUVIUS_DATA/`. Nothing else was edited.
- **Hardware:** MacBook Pro M5 Pro (24 GB, macOS 27.0.1) and the PC's RTX 4070. The runs are from 2026-11-22.

| number | file |
|---|---|
| Rewrite = stock network: float64 max logit diff 3.5e-14 (stack and sum forms); through `infer.py` on a 64 × 64 px B0 window, CPU, r 0.99999999999996 | `fold_check_cpu.json` (`f64_small_stack_max_abs_logit_diff`, `f64_small_sum_max_abs_logit_diff`, `folded_vs_stock_cpu_r`), written by `mac/ahn_fold_check.py` on the PC's CPU |
| Backends on 64 B0 tiles, batch 1: 4.65 / 9.95 / 7.37 / 11.3 tiles/s; where each Core ML op runs | `bench.json` (`stock-mps`, `fold-mps-stack-fp16-b1`, `coreml-CPU_AND_NE`, `coreml-CPU_AND_GPU`; `ops_by_device`), written by `mac/ahn_bench.py` |
| Two-lane walls on the B0 native window: s0 1,094.3 s, decoy 1,083.6 s, s1 1,082.4 s, s2 1,081.8 s; peak 1.97 GB MPS driver, 1.43–1.44 GB RSS | `chain.log`; the commands are in `chain.sh` |
| Two-lane rate 15.4–15.6 tiles/s | 16,896 tiles (`*_lanes.meta.json`, `tiles`) / the walls in `chain.log` |
| Recipe: ROI, grid origin, `clip_to_roi`, checkpoint sha256, backend and lane pattern | `{s0,s1,s2,decoy}_lanes.meta.json` (the `infer.py` sidecar, with the `mac` block) |
| Members against the 4070's maps: r, max abs diff, NaN mismatches, same 8-bit level | `compare_native.json`, computed on the PC from the Mac's four maps and the 4070's native maps (render rows 1516–2915 and columns 2038–7638, end-exclusive, as the run's ROI) |
| Fused A, product and decoy product against the 4070's fused maps: r, max abs diff, top-1 % Jaccard (window and claim box; also against shipped) | `fused_compare.json` (keys `ensA`, `prod`, `prodd`; `cuda.crop` is the window) |
| No-text control | `notext_check.log`: members and fused maps against the 4070's no-text maps, and the control numbers next to the 4070's |
| Stock-path walls, TRACE 444.9 s, the 4070's walls and 5.36 cm²/GPU-h | the main appendix's files, listed in [`../README.md`](../README.md) |
| Chain 4.40 cm²/GPU-h | 5.8536 cm² / ((444.9 + 1,094.3 + 1,083.6 + 1,082.4 + 1,081.8) s / 3600) |
