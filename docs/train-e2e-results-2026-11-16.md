# Results: the winner's `train` mode end to end on an RTX 4070 (12 GB), 2026-11-15/16

Plan and planned metrics: `docs/train-e2e-plan-2026-11-15.md`. The comparison JSONs, the chain log and the teacher logs cited below are bundled in `docs/train-e2e-compare/` (`e2e_chain.log` below means the copy there). Other data paths (`maps/`, `self/`) are in the local data directory (`$VESUVIUS_DATA/train_e2e/`), not published. The chain log is `docs/train-e2e-compare/e2e_chain.log`; the final run is `20261115-234340-83204`, which ended with `E2E_DONE` at 2026-11-16 05:08:10.

## Headline

- **Each 12 GB member is closer to its shipped counterpart than the winner's seeds are to each other.** Each A-hn member was retrained with its own seed (0, 1, 2), as shipped. The maps of our retrained members match the maps of the winner's checkpoints at r 0.9925–0.9934 over the member ROI (27.5 M px; table "Stages 7–8" below, source `docs/train-e2e-compare/maps.json`). The yardstick for training variation is the winner's own seed-to-seed spread: their three members' maps agree with each other at r 0.932–0.942 (`seed_spread_r` in `docs/retrain-s0-summary-2026-11-15.json`).
- **The fused claim map matches at r 0.9992298** (0.9995508 in the claim box), with 96.7% of its top-1% box pixels shared. Most of that agreement comes from the shared, exactly reproduced TRACE map (see "Reading" below).
- **Hardware:** one RTX 4070 12 GB, trainer budget 10.5 GB, in the trainer's own block-checkpointing mode. The trainer `train_cloud_hn.py` defaults to `grad_ckpt 'block'` and `vram_gb 9.0` (docstring line 10, lines 64 and 113); the shipped hn4 configs opt out (`false`, `30.0`), and the winner's `docs/REPRODUCE.md` (line 41) says retraining needs one GPU with 24 GB or more (19 GB reported peak, line 28). About 8.2 GPU-hours for stages 2, 6 and 7, plus 775 s for the TRACE map (follow-up run).

## Scope: what ran here and what was reused

The chain ran the winner's stage scripts inside their image `fl-claim-a7:final`, unmodified, one by one (the image's `train` entrypoint was not invoked; stage names and numbers are ours), driven by wrappers that set the recipe (ROI, grid origin, `clip_to_roi`) and the 12 GB config, with the hf-data inputs mounted at the original paths. Stage 7 used their `infer.py` with the shipped member recipe (`pc/exact_s0.py`), not `render_leadA.py`. Training ran with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Ran on this PC:**
- stage 1 pool masks;
- stage 2 teacher maps (Reader V3, TRACE v3B, Reader v2, base v8-in);
- stage 3 machine labels;
- stage 4 exclusion audit;
- stage 5 packs for the 5 regions;
- stage 6 training of A-hn s0, s1, s2 and the decoy;
- stage 7 member claim maps with the shipped member recipe;
- stage 8 fusion.

**Reused, not regenerated:**
- **Inputs:** the shipped renders, meshes and exclusion zone.
- **Model weights:** the shipped teacher and base weights (Reader V3, TRACE v3B, Reader v2, and base v8-in, which training starts from). `train` mode trains only the A-hn members and the decoy.
- **Anchor packs:** the shipped `a_w058.npz` and `a_w060.npz`, sha256-checked against MANIFEST (`e2e_chain.log` 22:06:04–22:06:12). A later CPU rebuild from the raw public PHerc1447 stacks with the winner's `make_packs.py` gives byte-identical packs (sha256 = MANIFEST; `docs/train-e2e-anchors-2026-11-16.md`).
- **TRACE claim map in the fusion:** our earlier exact reproduction (`maps/fwd_tr.npy`; row `tr` below). TRACE isn't retrained in `train` mode. A follow-up run regenerated it inside this data tree (`maps_v2/fwd_tr.npy`, shipped TRACE weights): bit-identical to the map used here, and re-fusing with it reproduces every row below (`docs/train-e2e-compare/rerun_v2.json`, `maps_v2.json`; section "Follow-up runs").

**Not run in this chain:**
- stage 9, the claim image (run afterwards on CPU: section "Stage 9" at the end);
- the depth-shuffle control maps (`shuf_render.zarr` isn't in hf-data). A follow-up run built it with the winner's own `shuffle_control.py` and ran it: section "Follow-up runs".

**Deviations from the shipped run:**
- **Trainer config:** `vram_gb 30 → 10.5` and `grad_ckpt false → block` (`configs/`). This was verified exact for BatchNorm in the s0 retrain, `docs/retrain-s0-2026-11-15.md`.
- **s1 resumed twice:** s1 trained in three chunks (steps 0–370, 370–540, 540–600) because the host guard halted twice when it couldn't read Windows commit (`e2e_chain.log` 23:24:46 and 23:41:29). Each chunk resumed from the trainer's checkpoint. The other three members each trained in a single pass.

"End to end" therefore means stages 1–8 regenerated from the shipped inputs and weights, using the shipped anchor packs (rebuilt byte-identical separately) and a TRACE map regenerated bit-identically in the follow-up run.

## Stages 1–5 (reported 2026-11-15)

- **Sources:** `docs/train-e2e-compare/{masks,teach,labels,exclusion,packs}.json`.
- **Results:**
  - Masks are identical.
  - Teacher maps r ≥ 0.99999989.
  - Labels change class at 189–413 px per variant.
  - The exclusion audit gives 23,332 tiles (shipped 23,331).
  - Pack CT arrays are bit-identical; there is one extra tile in w053.

## Stage 6: training loss curves (`docs/train-e2e-compare/trainlog.json`)

Each row compares 60 logged steps (every 10th step) with the shipped `run_records/<m>/train_log.jsonl`.

| member | Pearson r of loss | max abs diff | final loss at step 600, ours / shipped |
|---|---|---|---|
| hn4_s0 | 0.98756 | 0.0104 | 0.49826 / 0.49800 |
| hn4_s1 | 0.97181 | 0.0352 | 0.54096 / 0.53815 |
| hn4_s2 | 0.98897 | 0.0084 | 0.57946 / 0.57842 |
| hn4_decoy | 0.99993 | 0.00076 | 0.55611 / 0.55609 |

- s1, the member that was resumed twice, has the lowest curve correlation.
- Checkpoint sha256 values are in `e2e_chain.log`. As expected, they differ from the shipped checkpoints.

## Stages 7–8: claim maps vs shipped (`docs/train-e2e-compare/maps.json`; top-k: `docs/train-e2e-compare/fused_topk.json`)

- **Map extent:** the 27.5 M finite px of the member ROI on the 4900×7780 render raster. The claim box is render px y 1639–2745, x 2431–6829 (4,864,188 px).
- **Reference bars:**
  - the shipped seed spread, r 0.932–0.942 (`seed_spread_r` in `docs/retrain-s0-summary-2026-11-15.json`);
  - our s0 retrain from the *shipped* packs, r 0.99958 (`docs/retrain-s0-2026-11-15.md`).

| map | role | r whole | mean abs diff | max abs diff | r in box | max abs diff in box |
|---|---|---|---|---|---|---|
| tr | TRACE v3B (reused exact reproduction) | 1.0000000 | 7.9e-10 | 7.3e-4 | 1.0000000 | 0 |
| s0 | A-hn member | 0.9925977 | 0.0106 | 0.259 | 0.9943591 | 0.259 |
| s1 | A-hn member | 0.9933982 | 0.0099 | 0.264 | 0.9960165 | 0.170 |
| s2 | A-hn member | 0.9925289 | 0.0114 | 0.259 | 0.9950954 | 0.196 |
| decoy | decoy member | 0.9994044 | 0.0027 | 0.046 | 0.9994856 | 0.035 |
| ensA | mean of the 3 A-hn members | 0.9974271 | 0.0065 | 0.115 | 0.9983319 | 0.115 |
| **prod** | **TRACE × mean(A-hn): the fused claim map** | **0.9992298** | 0.0016 | 0.050 | **0.9995508** | 0.041 |
| prodd | decoy product (TRACE × decoy) | 0.9998515 | 0.0006 | 0.021 | 0.9999005 | 0.012 |

**Which row is the fused output:** `fwd_prod`. The winner's docs say so: `docs/REPRODUCE.md` step 4 ("multiplies TRACE v3B by the mean of the three A-hn maps. Compare with `results/fused/fwd_prod.npy`") and `docs/MODELS_AND_DATA.md` ("TRACE v3B × A-hn (`fwd_prod.npy`)").

**Top-k agreement inside the claim box** (top-1% Jaccard, as planned; top-5% added):

| map | top-1% Jaccard | top-5% Jaccard |
|---|---|---|
| prod | 0.9672 | 0.9856 |
| ensA | 0.7032 | 0.8827 |
| prodd | 0.9744 | 0.9886 |

**Reference pairs** for the members (claim box; `docs/train-e2e-compare/topk_ref.json`, `pc/train_e2e/topk_ref.py`): the shipped seeds against each other r 0.957–0.963, top-1% Jaccard 0.28–0.32; our s0 retrain from the shipped packs against shipped s0 r 0.9997, Jaccard 0.908; our e2e s0 / s1 / s2 against shipped r 0.9944 / 0.9960 / 0.9951, Jaccard 0.54 / 0.60 / 0.67; decoy Jaccard 0.87.

Reading, at the scope above:
- The maps of the retrained A-hn members match those of the shipped checkpoints at r 0.9925–0.9934 over the member ROI. That is far closer than the shipped members are to each other (0.932–0.942), and below the 0.99958 that the shipped-pack retrain reached. What is proven about the difference upstream of training (`docs/train-e2e-w053-extra-tile-2026-11-16.json`, `docs/train-e2e-w053-decisive-pixel-2026-11-16.json`; scripts `pc/train_e2e/w053_*`):
  - **The extra w053 tile** is the 64 px tile at pack origin (1056, 1952), with 0 ink pixels. The packer keeps a tile when at least 20% of it is labelled (`train_self.py:54`, threshold 819.2 px). Ours has 820 labelled px, shipped 819.
  - **The one differing pixel** (w053 bbox 1015, 1902) is background in ours and ignored in theirs. The labeller marks background when the fused teacher z is ≤ 0.3 (`labels.py:46`): ours 0.299445, shipped 0.300127.
  - **The only teacher that differs there is v8-in**: 0.6455078 vs 0.6459961, exactly one float16 step (2^-11). v3, TRACE and Reader v2 are bit-identical at that pixel. Swapping in only the shipped v8 map reproduces the shipped z.
  - **Cause:** v8 teacher inference runs under fp16 autocast with no cuDNN determinism (`aclick/bench/infer.py:122`) and is saved as float16, so 5.6% of valid v8 pixels in w053 (204,162 of 3,660,032) round one step differently on our card. One of them sat 0.0006 from the 0.3 threshold.
  - **Not proven:** how much of the member-map gap this one tile causes. It shifts the seeded sampler for every later w053 tile, but the other label changes (189–413 px per variant) and GPU non-determinism in training also contribute; separating them needs a retrain without the tile.
- The fused claim map `prod` matches at r 0.99923 (0.99955 in the box), and 96.7% of its top-1% pixels in the box coincide.
- **Most of that fused-map agreement comes from the exactly reproduced TRACE map**, which is shared by both sides of the comparison (row `tr`: r 1.0000000). The retrained A-hn members alone agree much less on the top pixels: the member mean `ensA` has a top-1% Jaccard of 0.703, against 0.967 for `prod`.

## Peak VRAM and wall time per stage

| stage | wall time | peak VRAM | source |
|---|---|---|---|
| 1 masks (CPU) | 729 s | none | `e2e_chain.log` 18:01:19 |
| 2a readers | 2046 s | 3.73 GiB (torch peak) | `docs/train-e2e-compare/teach_readers.log` |
| 2b v8, stride 32 | 2310 s | 5.98 GiB (torch peak) | `docs/train-e2e-compare/teach_v8.log` |
| 3 labels (CPU) | 158 s | none | `e2e_chain.log` |
| 4 exclusion (CPU) | 101 s | none | `e2e_chain.log` |
| 5 packs (CPU) | 274 s | none | `e2e_chain.log` |
| 5b anchors (shipped, sha check) | 12 s | none | `e2e_chain.log` |
| 6 train s0 | 2828 s | 7.94 GiB | `e2e_chain.log` 22:53:30 (`peak_gb`) |
| 6 train s1 | 3217 s (1867 + 943 + 407) | 8.56 GiB | `e2e_chain.log` 23:24:46, 23:41:29, 23:50:40 |
| 6 train s2 | 2962 s | 7.94 GiB | `e2e_chain.log` 00:40:29 |
| 6 train decoy | 2831 s | 7.94 GiB | `e2e_chain.log` 01:27:50 |
| 7 map s0 / s1 / s2 / decoy | 3331 / 3286 / 3268 / 3265 s | not logged in this chain; follow-up rerun of s0: 9,931 MiB whole card | `e2e_chain.log` 02:23:38–05:07:17; `docs/train-e2e-compare/vram_map_s0.csv` |
| 8 fuse (CPU) | 53 s | none | `e2e_chain.log` 05:08:10 |

`teach_readers.log` opens with a traceback (`ModuleNotFoundError: No module named 'vesuvius'`). That is the first attempt, which stopped at 20:40:53 because the villa package wasn't importable (`e2e_chain.log`, `E2E_FAIL`). The stage was rerun from 20:41:39 and finished in 2,046 s; the rest of the log is that run.

- **Total:** GPU about 8.4 h (30,119 s: stages 2, 6 and 7, 29,344 s, plus the TRACE map, 775 s, from the follow-up run) and CPU about 0.4 h.
- **Map-stage VRAM:** not recorded by this chain. The follow-up run logged the whole card (`nvidia-smi` memory.used every 500 ms, desktop included): 986 MiB idle, 2,285 MiB peak for TRACE (775 s), 9,931 MiB peak for the s0 member map (3,246 s), whose output is bit-identical to the chain's (`docs/train-e2e-compare/vram_trace.csv`, `vram_map_s0.csv`, `night_chain.log`). An earlier inference run peaked at 10.2 GB (`docs/fidelity-2026-11-13.md`).
- **Trainer peaks:** the trainer's `peak_gb` is torch's allocated peak in GiB (`max_memory_allocated / 2**30`), not the card total. Whole-card VRAM was not logged in this chain. A follow-up training run with the same config and packs (s0, `SUFFIX=vram`, `pc/train_e2e/train_vram.sh`) logged it every 500 ms: peak 9,706 MiB against 1,008 MiB idle (`docs/train-e2e-compare/vram_train_hn4_s0_vram.csv`), 2,829 s, loss r 0.988 vs the shipped run record (`trainlog_vram.json`). Windows free commit fell from 16,021 MB to 1,710 MB during it (about 14.3 GB used; container RAM up to 5.1 GiB; `host_mem_train_vram.csv`). Its weights differ from the e2e s0 run (same data and seed) in all 284 tensors (`train_rerun_weights.json`): training on this card is not bit-reproducible. The earlier s0 retrain peaked at 10,655 MiB for the whole card, sampled every 15 s with more desktop load (`docs/logs/retrain-s0/gpu_train.csv`).

## Follow-up runs (2026-11-16/17)

Driver `pc/train_e2e/overnight_chain.sh`, log `docs/train-e2e-compare/night_chain.log` (the first two attempts ended on script errors in the comparison step and the VRAM-peak line, fixed and resumed; no model output was affected).

- **Determinism:** TRACE regenerated, and the s0 member map rerun from the same e2e checkpoint, are both bit-identical to the maps of this chain (r 1.0, max 0; `rerun_v2.json`). `maps_v2.json` repeats the "Stages 7–8" comparison with the regenerated TRACE: identical numbers.
- **Depth shuffle on our maps:** the winner's `shuffle_control.py` prep (render layers 8–31 permuted, seed 1, claim box + 256 px), then TRACE and our four e2e members on it, fused (`shuffle.json`, claim box, 4,864,188 px). Shuffled vs unshuffled: r 0.043 (TRACE), 0.088 / 0.095 / 0.058 (s0 / s1 / s2), 0.006 (decoy), 0.078 (ensA), 0.051 (prod). Top-1% Jaccard 0.0004 (ensA), 0.0003 (prod); no shuffled pixel reaches the unshuffled top 1%. TRACE ~ ensA r falls from 0.767 to 0.123. The winner's own check with their checkpoints: fused r 0.0495 (`submission/method/checks/depth_shuffle_variants_v7_trace_x_Ahn_product.json`, `c6_grey`).
- **The shipped shuffle maps reproduced:** the shipped checkpoints with the shipped shuffle recipe (ROI = claim box, `clip_to_roi` off) on the same shuffled render, against `results/member_maps/shuf_*` and `results/fused/shuf_*` (`shuf_vs_shipped.json`): TRACE bit-identical; s0 / s1 / s2 r 0.9999947 / 0.9999957 / 0.9999989, max abs 4.3e-4 / 4.1e-4 / 5.9e-4; ensA r 0.9999966; prod r 0.9999979, max 1.2e-4. Every pixel is within 1e-3. This also confirms our shuffled render equals theirs.

## Stage 9: claim image (added 2026-11-16)

The winner's `final` mode, CPU only, redrawn from our maps: the lead claim image differs from the package by more than 8 / 255 on 0.042% of its pixels (mean 0.40 / 255); the 148 pixels above 64 / 255 are one caption digit (display level z 7.96 vs 7.95). 11 of 20 variants are byte-identical. On the shipped maps the same step reproduces the package byte for byte. Details: `docs/train-e2e-stage9-2026-11-16.md`.
