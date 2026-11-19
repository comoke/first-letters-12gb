# Fine-tune demo: run logs, raw scores and configs

Text files only, metrics only (Scroll 1 data terms: no map, label or image of Scroll 1 is published). The README's demo section and `docs/finetune-results-2026-11-17.json` are built from these files.

| file | what |
|---|---|
| `pair1_ft_chain.log` | `pc/finetune/ft_chain.sh` log for pair 1 (train w03, test w09), every start from 06:29 to `FT_DONE` at 11:36 on 2026-11-17 |
| `pair2_ft_chain.log` | the same for pair 2 (train w06, test w00), 23:56 on 2026-11-17 to `FT_DONE` at 04:11 on 2026-11-18 |
| `pair1_order_A.json`, `pair2_order_A.json` | `pc/finetune/score.py` output for the layer-order check on the training segment (zero-shot s0, both orders) |
| `pair1_score_B.json`, `pair2_score_B.json` | `score.py` output on the test segment: per-arm AUC, top-1% precision, F0.5 at 0.5, pixel counts, and the paired tile bootstrap for every pair of arms |
| `configs/ft_*.json`, `configs/ft2_*.json` | the eight training configs as the chain wrote them (pair 1 `ft_`, pair 2 `ft2_`): the shipped `hn4_s0.json` with name, init, anchor list, steps, seed and the two 12 GB keys changed. The `note` field is copied unchanged from the shipped config. |
| `blocktest.json` | row-block evaluation against one pass on a 640 × 594 test crop (see the README, "Deviations") |

**How the logs were cleaned:** the pair-1 log's planning header (six lines written before the first run) was removed, and "run card above unchanged" was shortened to "settings unchanged". Nothing else was changed. Paths under `/home/liam/...` are inside the winner's Docker image.

**Runs.** Pair 1's first two starts (06:29:11, 07:39:54) stopped in the CPU data chain before any GPU step. The third (07:47:58) used the GPU from 07:59:00 and ran out of memory after 196 s in the order check (`orderA_rev exit=1`), which led to the row-block change. The reported run, `20261117-080700-32067`, then ran in one pass. Pair 2 ran in one pass, run `20261117-192233-47550` (its data chain started at 19:22:33).
