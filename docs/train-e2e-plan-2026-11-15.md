# Plan: the winner's `train` mode end to end on the RTX 4070 (12 GB), 2026-11-15

Goal: run every stage of the winner's training pipeline, from pool masks to the claim map, on this PC, and compare each stage with the shipped artifact.

The winner's `docs/REPRODUCE.md` marks the `train` mode "not run end to end" (needs ≥24 GB; training peaked at 19 GB). Source: the winner's `docs/REPRODUCE.md`, table row `train` and step 6.

## Stages

Code: the image's `/opt/fl/train/W/.claude/worktrees/leadA-push3/x_push3/m306/`. The stage list comes from the image's `run.sh` mode `train` (lines 208–260) and from `make_packs.py`.

| # | stage (script) | inputs | outputs | shipped artifact to compare (hf-data) |
|---|---|---|---|---|
| 1 | pool masks (`self/code/masks.py`) | 5 region renders (`data/0_inputs/renders/{A6,w052,w053,w059,w109}.zarr`), meshes (`data/0_inputs/meshes/`), exclusion zone (`/opt/fl/train/exclusion`, `data/0_exclusion_zone/`) | `<r>_masks.npz/.json` | `data/1_pool_masks/` |
| 2a | teachers: Reader V3, TRACE v3B, Reader v2 (`teachers.py readers`) | renders, masks, reader weights in the image | `<r>_{v3,trace,rv2}.npy`, `<r>_teach.json` | `data/2_teacher_maps/` |
| 2b | teacher: base v8-in, stride 32 (`teachers.py v8`) | renders, masks | `<r>_v8.npy` | `data/2_teacher_maps/` |
| 3 | machine labels (`labels.py`) | teacher maps, masks | `<r>_labels.npz`, `stats.json` | `data/3_machine_labels/` |
| 4 | exclusion audit (`exclusion_audit_v7.py recompute`) | labels, zone | audit json | `data/0_exclusion_zone/exclusion_audit.json` |
| 5 | packs (`make_packs.py 306`, plus anchors) | labels, renders, zone; anchors: raw PHerc1447 w058/w060 | `306_<r>.npz`, `a_w058/a_w060.npz`, MANIFEST | `data/3b_cloud_packs/` (sha256 in MANIFEST.json) |
| 6 | train A-hn s0, s1, s2 (+ decoy) (`train_cloud_hn.py`) | packs, base v8-in | `<m>_final/model.safetensors`, `train_log.jsonl` | hf-model `checkpoints/`, `run_records/hn4_*/train_log.jsonl` |
| 7 | member claim maps (shipped member recipe: `exact_s0.py` pattern; roi 115 4166 694 7638, grid 115 694, clip_to_roi, TTA 1, nsig 3) | claim render (`results/claim_surface_render_41layers.zarr`) | `fwd_hn4_*.npy` | `results/member_maps/fwd_hn4_*.npy` |
| 8 | fusion (`fuse_v7.py`): TRACE × mean(A-hn), plus the decoy product | our TRACE claim map (already reproduced exactly: `docs/fidelity-2026-11-13.md`) and the member maps | `fwd_ensA`, `fwd_prod`, `fwd_prodd` | `results/fused/` |
| 9 | claim image (`run.sh final` with `MAPS=` our maps) | fused maps, render CT, mesh | claim PNG | the image's `data/final` PNGs |

**Not covered:**
- **The depth-shuffle control maps.** These need `shuf_render.zarr` (INPUTS.tsv row `x_req/20261020/A10/v4c_work/shuf_render.zarr`), which isn't in hf-data (`grep shuf_render SHA256SUMS`: no hit). It was later rebuilt with the winner's own shuffle_control.py prep (overnight chain).
- **PHerc0343 / PHerc0139 anchor packs.** The hn4 configs use only the anchors w058 and w060 (`hn4_s0.json` `anchor`), so these aren't needed.

## Estimates for this PC

16 GB RAM, RTX 4070 12 GB, WSL with a 12 GB VM. Commit ceiling 40.9 GB (RAM plus the 24 GB pagefile max), enforced by `pc/hostguard.sh`. Region sizes come from each render's `.zarray`; pool areas from `data/1_pool_masks/*_masks.json`.

| stage | device | estimate | peak RAM / commit | risk |
|---|---|---|---|---|
| 1 masks | CPU | 15–40 min | < 2 GB (row bands) | needs the meshes at `renders.json` paths |
| 2a readers | GPU | 1.5–3 h for 3 models × ~99 Mpx (TRACE measured at ~35 s/Mpx with TTA 8; map3 is mirror4) | `region_vol` loads the whole bbox: A6 41 × 6440 × 6440 ≈ 1.7 GB uint8, plus float working copies, plus VRAM ≤ 9 GB charged to commit | **A6 is the tight one.** Container cap 10 GB; host guard halts cleanly if commit runs out |
| 2b v8 stride 32 | GPU | ~0.5 h (pool ≈ 26.7 cm² ≈ 36 Mpx; base v8-in at stride 21 measured ~116 s/Mpx) | as 2a | resumable in bands (exit 3) |
| 3 labels | CPU | < 15 min | ~2–3 GB for A6 | none |
| 4 exclusion | CPU | ~1 min (shipped: 44 s) | small | none |
| 5 packs | CPU | ~20 min; anchors need about a 1.2 GB download of the raw PHerc1447 stacks (`YoussefMoNader/ink-8um-pherc1447-surfaces`) | ~2 GB per region | if the raw anchors can't be fetched, reuse the shipped anchor packs (sha256-checked) and say so |
| 6 train 3 members (+ decoy) | GPU | 3 × 51 min = 2.6 h (+0.85 h for the decoy). Our s0 retrain: 50.4 min, `docs/retrain-s0-2026-11-15.md` | peak 8.56 GB VRAM; ~13.5 GB commit; needs pagefile growth (fine now) | the shipped `vram_gb 30 / grad_ckpt false` → **10.5 / block** (verified exact for BN) |
| 7 member maps | GPU | 3 × 53 min = 2.65 h (+0.9 h decoy); exact s0 took 52.7 min, `docs/fidelity-2026-11-13.md` | ~10 GB card peak; fine | none |
| 8–9 fusion + image | CPU | < 10 min | small | none |

**Total GPU about 7.5–9.5 h (with the decoy), CPU about 1.5 h.**

The only parts that assume more than 12 GB are the trainer's `vram_gb: 30` / `grad_ckpt: false`. They're replaced as above (config diff only; verified for s0). No stage is expected not to fit. A6's teacher pass is the one memory risk, and it's guarded.

## Comparison metrics (fixed before any results)

1. **Masks:** array equality per key (`dist`, `qual`, `cov24p`), plus max abs diff on `dist`/`qual`, plus the `_masks.json` numbers.
   - Claim word: "identical" only if equal; otherwise give the diff.
2. **Teacher maps:** per region × model, Pearson r, mean and max abs diff on finite pixels, plus the finite-mask agreement fraction.
   - Claim word: "reproduces to fp16" if max abs diff ≤ 1e-3; otherwise give r and the diffs.
3. **Labels:** per region × variant (main / sc / decoy):
   - fraction of pixels with the same class (ink / bg / ignored);
   - ink-pixel Jaccard;
   - count of pixels that changed class;
   - `fz` r;
   - `stats.json` numbers side by side.
4. **Exclusion audit:** tile count, all-allowed flag, nearest distance, versus `exclusion_audit.json`.
5. **Packs:**
   - per array key: equality, else the count of differing elements;
   - `org` tile lists: set difference;
   - sha256 vs `MANIFEST.json`.
6. **Training,** per member:
   - Pearson r of our `train_log.jsonl` loss series with the shipped `run_records/<m>/train_log.jsonl` (same steps);
   - final loss;
   - the checkpoint sha256 (expected to differ; GPU kernels).
7. **Member claim maps:** r, mean and max abs diff vs the shipped `fwd_hn4_*.npy` (exact recipe).
   - Reference bars: the shipped seed spread (r 0.932–0.942, `docs/retrain-s0-summary-2026-11-15.json`) and our s0 retrain from the shipped packs (r 0.99958).
8. **Fusion:**
   - `fwd_prod` r, mean and max abs diff vs `results/fused/fwd_prod.npy`, over the context and inside the claim box;
   - top-1% Jaccard in the box;
   - the same for `ensA` and `prodd`.
9. **Claim image:** pixel difference stats vs the shipped image (mean, max, % of pixels changed).

Every number in the results doc cites its source file. README claims are edited only after a separate claim check.

## Execution

One detached chain per stage group, each with the host guard, run-ID markers and private script copies:
1. Mount hf-data inputs read-only at the paths the scripts expect (`/home/liam/projects/vesuvius-work/<INPUTS.tsv path>`), with **writable D: directories mounted over the output folders** (`m306/self/{pool,teach,labels,packs,runs}`). Nothing lands in the Docker VHD on C:.
2. Then: stage 1 → 2a → 2b → 3 → 4 → 5 → 6 → 7 → 8 → 9, writing a compare JSON after each stage.
