# Retraining the PHerc. 0306B First Letters winner's members on a 12 GB GPU

**What this gives the community:**
- **An independent end-to-end retrain of a First Letters winner, checked stage by stage, on one 12 GB card.** The PHerc. 0306B winner's own guide marks their `train` mode "not run end to end" and says retraining "needs one GPU with 24 GB or more". We ran stages 1–8 of their training with their own image, code and data on an RTX 4070 and compared each stage with their shipped files. As far as we know, this had not been done for a First Letters winner before.
- **Fine-tuning that model on new labels, about 25 minutes a training run on the same card,** with a held-out gain on two pre-registered train/test pairs from Scroll 1.
- **The same inference on a Mac** (Apple Silicon, through MPS), matching the CUDA maps to float16 precision.

| | result | section |
|---|---|---|
| retrain on 12 GB | stages 1–8 of the winner's training for the A-hn members (PHerc. 1447 anchor packs from their release, also rebuilt byte-identical) on an RTX 4070 with 16 GB of RAM: 8.4 GPU-hours, whole-card peak 9,706 MiB | "What stages 1–8 means", "Windows/WSL memory" |
| how close | retrained members r 0.9925–0.9934 against the shipped maps, closer than the winner's own seeds agree (0.932–0.942); fused claim map r 0.9992; the redrawn claim image differs by more than 8/255 on 0.042% of pixels | "Verification" |
| exact inference | every map the claim uses, rerun with the shipped recipe: r ≥ 0.9999997 against the shipped maps (the depth-shuffle control maps r ≥ 0.999994) | "Inference on 12 GB" |
| fine-tune demo | with the new labels, +0.046 / +0.050 AUC over zero-shot on pair 1 and +0.136 / +0.141 on pair 2, and ahead of the same training without them in every comparison; 24–27 min per training run | "Demo: fine-tuning on new labels from Scroll 1" |
| Apple Silicon | all five maps and the winner's fusion on an M5 Pro: TRACE r 0.999999, members r ≥ 0.99999993, fused product r 0.99999977 against the 4070 | [`apple-silicon.md`](apple-silicon.md) |

**Upstream:** a pull request to the winner's repository adds the 12 GB configs and a REPRODUCE note: [SirProdigle/pherc0306b-first-letters#8](https://github.com/SirProdigle/pherc0306b-first-letters/pull/8).

## The retrain: scope, cost and what makes it fit

With this repository you can retrain the trained members (A-hn) of the model that won PHerc. 0306B First Letters (SirProdigle / Liam Warner and Nicola Bodill, [`pherc0306b-first-letters`](https://github.com/SirProdigle/pherc0306b-first-letters)) on one 12 GB consumer card, then redraw its claim image. TRACE v3B, the other half of the claim map, is a third-party model that their `train` mode doesn't train; it runs with its shipped weights. We did it on an RTX 4070 with 16 GB of RAM (Windows 11, WSL2, Docker Desktop), using the winner's own Docker image, code, weights and data. Their guide says retraining "needs one GPU with 24 GB or more" ([`docs/REPRODUCE.md`](https://github.com/SirProdigle/pherc0306b-first-letters/blob/main/docs/REPRODUCE.md), line 41; their training peaked at 19 GB, line 28) and marks its `train` mode **"not run end to end"** (lines 25–26).

- **Time:** about 8.4 GPU-hours plus 0.4 CPU-hours for the whole training chain: teacher maps 1.2 h, four training runs of 47–54 min, TRACE map 13 min, four member maps of about 55 min each. The claim image then takes 7 min on the CPU.
- **Memory** (whole card, desktop included): peak 9,706 MiB while training, against 1,008 MiB idle just before that run (`docs/train-e2e-compare/vram_train_hn4_s0_vram.csv`, `train_vram.log`); 9,931 MiB for a member map and 2,285 MiB for TRACE, against 986 MiB idle before those runs (`vram_map_s0.csv`, `vram_trace.csv`; `night_chain.log` gives the idle value, and its 22:23 TRACE line understates the peak because of a string-compare bug corrected in its 23:20 line).
- **Disk:** about 23 GB of downloads (model 6.3 GB, dataset 17 GB), 12 GB for the unpacked image tar and about 9 GB of outputs for the training chain and claim image on the data drive, plus 24 GB for the loaded Docker image (listing: `docs/logs/sizes.txt`).

**What makes it fit, and whose idea it is.** The winner's trainer already supports 12 GB: `train_cloud_hn.py` defaults to per-block activation checkpointing (`grad_ckpt 'block'`) and a 9 GB budget (`vram_gb 9.0`; its docstring, line 10, and lines 64 and 113). Their shipped A-hn configs opt out (`grad_ckpt false`, `vram_gb 30.0`), and their cloud launcher already falls back to block mode after an out-of-memory error (`cloud_box_hn.sh` line 11 sets `GCK_FORCE=block`, read at `train_cloud_hn.py` line 114). On the 4070, a no-checkpoint training step at micro-batch 4 runs out of memory in our BatchNorm-check harness (`docs/logs/retrain-s0/bn_check.log`: "plain arm OOM at micro 4"). The recipe here therefore sets their own block mode directly, with a 10.5 GB budget, plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. What we add:
1. the measurement that their block mode fits a 12 GB card and reproduces the whole training chain, which they mark "not run end to end";
2. the check that BatchNorm statistics under checkpointing equal the no-checkpoint run exactly;
3. the Windows/WSL memory findings, and a host guard, without which a 16 GB PC can crash the WSL VM during training (ours did once, when the pagefile couldn't grow);
4. scripts that run every stage in their image and compare it with their shipped artifact.

**Reused inputs.** The main training chain used the shipped PHerc. 1447 anchor packs and our earlier exact TRACE map. Follow-up runs rebuilt the anchor packs from the raw public data (byte-identical) and regenerated TRACE inside the chain (bit-identical).

## How to run it

**Tested on:** one RTX 4070 12 GB, 16 GB RAM, Windows 11 with WSL2 Ubuntu and Docker Desktop (GPU support), using the native Linux `docker` CLI inside WSL, with a pagefile that can grow. Native Linux is untested; there the host guard is a no-op.

**Prerequisites:** an NVIDIA GPU with 12 GB and Docker with GPU support (`nvidia-smi` working), `git`, `uv` (the Hugging Face downloads and the image load), `tmux` for detached runs, and `python3` on the host (the fine-tune chain reads its scores with it). The Mac recipe also needs `zstd`, and Docker on some machine to copy the code out of the image.

**Settings** (`pc/env.sh`): `VESUVIUS_DATA` is the data directory (default `/mnt/d/Workspace/vesuvius-data`). `DOCKER` is the docker CLI (default `/usr/bin/docker` when present, else `docker` on PATH), and `NVIDIA_SMI` is found on PATH or at the WSL path. The scripts find each other from their own location, so the repository can live anywhere.

0. **Choose the data directory** first, on any drive with about 45 GB free for the retrain (see Disk above): `export VESUVIUS_DATA=/path/to/vesuvius-data`. Every script reads it; the default is our machine's path.
1. **Download** (about 23 GB): `bash pc/download_winner.sh` fetches the winner's repository, their model repo `Prodigle/pherc0306b-ink-ahn` and their dataset `Prodigle/pherc0306b-first-letters`. The dataset includes the Docker image archive `docker/fl-claim-a7_final.tar.zst`, which the script sha256-checks. Then load the image: `bash pc/zstd_load.sh "$VESUVIUS_DATA/winner/hf-data/docker/fl-claim-a7_final.tar.zst"`.
2. **Train end to end:** `STAGES=masks,teach_readers,teach_v8,labels,exclusion,packs,anchors_shipped,train,members,fuse bash pc/train_e2e/e2e_chain.sh`. Run it detached (for example in `tmux`); it resumes from finished outputs.
   - Each stage compares its output with the shipped file (`pc/train_e2e/e2e_compare.py`, `pc/exact_all_compare.py`).
   - The `train` stage writes the 12 GB configs and logs whole-card VRAM every 500 ms. `SUFFIX=<name>` trains a second copy without touching the first.
3. **Claim image:** `bash pc/train_e2e/stage9.sh` (CPU).
4. **Inference with the exact shipped recipe:** `bash pc/exact_all_chain.sh` (all members, fusion, comparison; it runs `pc/exact_chain.sh` first for TRACE and s0 when they are missing). The general driver is `bash pc/ink_run.sh <render.zarr> <outdir> [y0 y1 x0 x1]`.
5. **Optional:** rebuild the anchor packs from the raw data (`bash pc/train_e2e/anchors_rebuild.sh`), and run the depth-shuffle and rerun checks (`bash pc/train_e2e/overnight_chain.sh`).
6. **Fine-tune demo:** see "Running the demo" in the demo section.

**Tested from the published file set.** We exported this repository's public files with our export script (not included; no git history; export r6, 188 files, on 2026-11-18) into a new folder and pointed `VESUVIUS_DATA` at a new data directory on another drive, holding only the inputs those stages read. We ran from `/tmp` with no other settings (`docs/logs/fresh-clone/`: script, log, outputs). Results:
- the image archive's sha256 checks OK;
- the masks stage reproduces the shipped pool masks exactly in all 5 regions (25 of 25 checks equal; `masks.json`);
- the claim-image step prints MATCH on the shipped maps and gives the same 0.042% difference on ours.

`download_winner.sh` and `zstd_load.sh` were checked (syntax, the archive's sha256) but not executed end to end in this test, and the Docker image was already loaded; the 23 GB download was not repeated. The GPU stages (`teach_readers` to `fuse`), `exact_all_chain.sh`, `overnight_chain.sh`, the fine-tune demo and the Mac recipe were not run from the exported folder either; they ran from our working copy, with the logs under `docs/`.

**Changed after that test** (in this release): the documentation (`README.md`, `CLAIMS.md`, `apple-silicon.md`, files under `docs/`, and the new `docs/finetune-runs/`), the test's own logs, and these scripts:
- `pc/env.sh`: with no docker installed, the docker variable now falls back to `docker`, so a step that needs it fails with "command not found" instead of the script exiting silently. With docker present, which the tested stages need, nothing changes.
- `pc/finetune/ft_chain.sh`, `data_chain.sh`, `labels_chain.sh`: the resume fix and the portability fixes described in the demo section.
- `mac/fuse_compare.py`: skips the comparison when the 4070's reference maps are absent.
- comments only: `pc/b0/b0_make.py`, `pc/b0/b0_chain.sh`, `pc/retrain/retrain_chain.sh`, `docs/logs/fresh-clone/fresh_clone_test.sh`.

**Windows/WSL memory.** Under WSL2, GPU memory is charged to Windows commit memory. In a logged training run (`docs/train-e2e-compare/host_mem_train_vram.csv`, every 30 s), Windows free commit fell from 16,021 MB to 1,710 MB: the run took about 14.3 GB of commit, the card's 8,698 MiB of training memory above idle plus about 5 GiB of training data and workers in the container's RAM, against a commit limit of 32.7 GB (16 GB of RAM plus a pagefile already grown to about 16 GB). So on a 16 GB PC, training fits only if the pagefile can grow. Our first attempt crashed the WSL VM when a nearly full C: drive stopped the pagefile growing. The chains therefore run a host guard (`pc/hostguard.sh`): a C: free-space floor, a commit check that knows when the pagefile can still grow, a periodic WSL page-cache drop and container memory caps. During the end-to-end run, the guard twice failed to read Windows commit and halted A-hn s1, which resumed from its own checkpoint both times (steps 0–370, 370–540, 540–600). Those were false alarms, not memory shortages. When the guard trips, a chain stops its containers cleanly (the trainer saves its state first) and ends with a FAIL line naming the reason; rerunning the same command resumes from the finished outputs and the trainer's checkpoint.

## What "stages 1–8" means

The stage names and numbers are ours. We did not invoke the image's `train` entrypoint. We drove its stage scripts one by one with the shipped inputs mounted where the scripts expect them, in the order of the image's `run.sh` `train` mode.
- Stages 1–5 run their scripts unchanged (`masks.py`, `teachers.py`, `labels.py`, `exclusion_audit_v7.py`, `make_packs.py`).
- Stage 6 runs `train_cloud_hn.py` with the configs above.
- Stage 7 (member claim maps) uses their inference script `infer.py` with the shipped member recipe (`pc/exact_s0.py`), not `render_leadA.py`.
- Stage 8 runs `fuse_v7.py`, and the claim image runs their `run.sh final`.
- Not run: `pipeline` mode's mesh and render stages. We start from the shipped render.

| stage | ours | result (RES section) |
|---|---|---|
| mesh, render | not run: the shipped meshes and renders | "Scope" |
| 1 pool masks | **fresh** | identical to shipped ("Stages 1–5") |
| 2 teacher maps | **fresh**, shipped teacher weights | r ≥ 0.99999989 |
| 3 machine labels | **fresh** | 189–413 px change class per variant |
| 4 exclusion audit | **fresh** | 23,332 tiles vs 23,331 shipped |
| 5 PHerc0306B packs | **fresh** | CT arrays bit-identical; one extra tile in w053, traced to one float16 step in the v8-in teacher |
| 5b PHerc1447 anchor packs | shipped packs (sha256-checked); separately rebuilt from the raw public stacks with their `make_packs.py` | rebuild **byte-identical** ([`docs/train-e2e-anchors-2026-11-16.md`](docs/train-e2e-anchors-2026-11-16.md)) |
| 6 training: A-hn s0, s1, s2, decoy | **fresh**, block mode, 10.5 GB budget | loss curves r 0.972–0.9999 vs shipped logs ("Stage 6") |
| TRACE claim map | **fresh**, shipped TRACE weights (`train` mode doesn't train TRACE) | bit-identical to our earlier exact run; r 1.0000000 vs shipped |
| 7 member claim maps | **fresh** | r 0.9925–0.9934 (members), 0.9994 (decoy) |
| 8 fusion | **fresh** | r 0.9992 whole, 0.9996 in the claim box |
| claim image | **fresh** (CPU) | 0.042% of pixels differ by more than 8 / 255 |

Evidence:
- **Results (RES):** [`docs/train-e2e-results-2026-11-16.md`](docs/train-e2e-results-2026-11-16.md).
- **Plan:** [`docs/train-e2e-plan-2026-11-15.md`](docs/train-e2e-plan-2026-11-15.md).
- **Comparison JSONs and logs:** [`docs/train-e2e-compare/`](docs/train-e2e-compare/), [`docs/logs/`](docs/logs/).
- **Claim-by-claim sources:** [`CLAIMS.md`](CLAIMS.md).

## Verification

### How close the retrained members are

Each A-hn member was retrained with its own seed (0, 1, 2), as shipped. The training data was regenerated from the shipped inputs, so it is slightly different, and GPU training is not bit-exact. The yardsticks are two bars:
- **Training variation in the winner's own run:** how much their three seeds' maps differ from each other.
- **The same-recipe bar:** our first retrain of s0 from the shipped packs.

| pair (A-hn maps) | r, member ROI (27.5 M px) | r, claim box | top-1% Jaccard, claim box |
|---|---|---|---|
| the winner's seeds vs each other (s0~s1, s0~s2, s1~s2) | 0.932–0.942 | 0.957–0.963 | 0.28–0.32 |
| our s0 retrain from the shipped packs vs shipped s0 | 0.99958 | 0.9997 | 0.908 |
| **our end-to-end s0 / s1 / s2 vs shipped** | **0.9926 / 0.9934 / 0.9925** | **0.9944 / 0.9960 / 0.9951** | **0.54 / 0.60 / 0.67** |
| our end-to-end decoy vs shipped | 0.9994 | 0.9995 | 0.87 |
| our A-hn mean (ensA) vs shipped | 0.9974 | 0.9983 | 0.70 |
| **fused claim map** (TRACE × A-hn mean) | **0.9992** | **0.9996** | **0.967** |

Sources: `docs/train-e2e-compare/maps.json`, `fused_topk.json`, `topk_ref.json` (`pc/train_e2e/topk_ref.py`); seed spread over the member ROI from [`docs/retrain-s0-2026-11-15.md`](docs/retrain-s0-2026-11-15.md).

Each end-to-end member is much closer to its shipped counterpart than the winner's seeds are to each other, but not as close as a retrain from the shipped packs.
- **Why they fall short of the same-recipe bar.** The training data differs slightly: the regenerated labels change class at 189–413 px per variant, and one extra w053 tile passes the packer's 20%-labelled threshold (820 vs 819.2 px). GPU non-determinism in training adds to that. The share of each isn't measured.
- **The extra tile.** We traced it to a single pixel where the v8-in teacher map differs by exactly one float16 step (fp16 inference without cuDNN determinism). That pixel moves the fused teacher z from 0.300127 to 0.299445, across the labeller's 0.3 threshold (RES, "Stages 7–8").
- **Where the fused match comes from.** Most of the fused-map agreement comes from TRACE, which both sides share and which reproduces exactly.

The 12 GB config keeps everything else as shipped:
- micro 4 × accum 4 (effective batch 16) and each member's own seed;
- lr 2e-5 → 2e-6 cosine, 600 steps;
- bf16 autocast and BatchNorm in train mode.

With the trainer's own restore, BatchNorm buffers under checkpointing equal the no-checkpoint run exactly (diff 0.0). This was checked at micro-batch 2, because the no-checkpoint run doesn't fit 12 GB at micro 4 (`pc/retrain/bn_check.py`, `docs/logs/retrain-s0/bn_check.log`).

### Determinism and the depth-shuffle control

Driver [`pc/train_e2e/overnight_chain.sh`](pc/train_e2e/overnight_chain.sh); log `docs/train-e2e-compare/night_chain.log`.
- **Determinism.** Regenerating TRACE gives a map bit-identical to our earlier run, and re-running the s0 member map from the same checkpoint gives a bit-identical map (`rerun_v2.json`). Re-fusing with the regenerated TRACE reproduces every number above (`maps_v2.json`).
- **Depth shuffle.** The winner's own `shuffle_control.py` permutes render layers 8–31 (seed 1) around the claim box, and on that render our retrained models lose the claim (`shuffle.json`). In the claim box (4.86 M px):
  - the fused map correlates with the unshuffled one at only r 0.051;
  - its top-1% pixels overlap at Jaccard 0.0003;
  - TRACE ~ A-hn agreement falls from 0.767 to 0.123.

  The winner's own check with their checkpoints shows the same collapse: fused r 0.0495 for the lead view (`submission/method/checks/depth_shuffle_variants_v7_trace_x_Ahn_product.json`, `c6_grey`).

### The claim image from our maps

We redrew it with the winner's `final` mode (CPU) from our end-to-end maps and pixel-diffed it against their shipped claim image ([`docs/train-e2e-stage9-2026-11-16.md`](docs/train-e2e-stage9-2026-11-16.md), [`pc/train_e2e/stage9.sh`](pc/train_e2e/stage9.sh)). As a check of the tool, `final` on the shipped maps reproduces the package's PNGs byte for byte. Its log (`docs/logs/stage9/shipped.log`) shows the lead view `c6_grey` is `variants/TRACE_x_v8306H3_grey_page.png`.

| lead claim image (TRACE × A-hn, 4417 × 7385 px) | value |
|---|---|
| pixels differing by more than 8 / 255 | 0.042% (13,580) |
| pixels differing by more than 64 / 255 | 148 (0.00045%), all in one caption digit: display level "z 7.96" vs ours "z 7.95" |
| mean abs difference | 0.40 / 255 |
| byte-identical | no (retrained weights cannot be) |

- 11 of the 20 image variants are byte-identical. 4 more, which use our TRACE map but no retrained member, differ by at most 1 / 255.
- The variant most exposed to the retrain, the A-hn mean alone, differs by more than 8 / 255 on 3.3% of pixels.
- Side-by-side strip of the claim box: [`docs/img/stage9-claimbox-strip-shipped-vs-ours.png`](docs/img/stage9-claimbox-strip-shipped-vs-ours.png). The caption digit: [`docs/img/stage9-header-shipped-vs-ours.png`](docs/img/stage9-header-shipped-vs-ours.png).

### Inference on 12 GB: every map the claim uses, to float16 precision

These runs use the winner's image, the shipped claim-surface render (41 × 4900 × 7780), the shipped member recipe (ROI, grid origin, `clip_to_roi`) and the shipped checkpoints ([`docs/fidelity-exact-all-2026-11-15.md`](docs/fidelity-exact-all-2026-11-15.md) and its JSON). The maps covered are every map the claim and its controls are built from: TRACE, the A-hn members, the decoy, their fusions, and the shipped depth-shuffle maps. The winner also ships five reading-aid maps that the claim doesn't use (`fwd_reader_v2`, `fwd_reader_v3`, `fwd_v8in`, `fwd_v8in_3sigma`, `fwd_v8Q`); we didn't reproduce those, and our claim-image run (stage 9) takes them as shipped.

| map | Pearson r vs shipped | max abs diff |
|---|---|---|
| TRACE v3B | 1.0000000 (99.99988% of pixels bit-equal) | 7.3e-4 |
| A-hn s0 / s1 / s2 | 0.9999999 each | 6.9e-4 – 8.0e-4 |
| decoy | 0.9999997 | 1.02e-3 |
| A-hn mean, fused product, decoy product | 0.9999998 – 0.9999999 | 4.9e-4 |
| depth-shuffle TRACE | 1.0 (bit-identical) | 0 |
| depth-shuffle A-hn s0 / s1 / s2 | 0.9999947 / 0.9999957 / 0.9999989 | 4.3e-4 / 4.1e-4 / 5.9e-4 |
| depth-shuffle A-hn mean, fused product | 0.9999966 / 0.9999979 | 4.9e-4 / 1.2e-4 |

- **The differences** are at the float16 storage step of the shipped maps. Only 2 decoy pixels exceed 1e-3.
- **The depth-shuffle maps** were made with their shuffle recipe (claim-box ROI, `clip_to_roi` off) on a shuffled render built by their own script (`pc/train_e2e/shuf_vs_shipped.py`, `docs/train-e2e-compare/shuf_vs_shipped.json`). The winner ships no shuffled decoy map.
- **Times and memory:** whole-render times on the 4070 are 12–14 min for TRACE and 53–63 min per A-hn member. The whole card peaked at 10,180 MiB, desktop included ([`docs/fidelity-2026-11-13.md`](docs/fidelity-2026-11-13.md); logs in `docs/logs/fidelity-2026-11-13/`).

## Demo: fine-tuning on new labels from Scroll 1

**Read these limits first.**
- **One scroll, two train/test pairs:** each pair trains on one segment and tests on another. The second pair was added by a dated amendment after the first pair's result and before any of its GPU runs, with the same arms, metrics, success rule and kill rule.
- **Pseudo-labels:** the test segments' labels are the organisers' hand strokes refined by iterative pseudo-labelling. No manually labelled Scroll 1 segment exists in the release.
- **A different scanner:** the ESRF 2.4 µm scan, resampled to 9.6 µm, against the model's 8.64 µm.
- **Two seeds per arm.**
- **One labelling process:** the training and test segments' labels come from the same model-in-the-loop labelling, so part of the gain may be agreement with that labeller rather than with the ink. With no hand-labelled Scroll 1 segment, this can't be separated.
- **Retention not measured:** we did not check how well the fine-tuned models still read PHerc. 0306B.
- **Stroke width follows the labels:** part of a stroke's width in any ink map comes from how thick its training labels were drawn, and our scores compare pixels with labels, so they inherit that thickness. For scale, Yohann reports that Reader v3 paints strokes about 1.3× wider than the ink in infrared photos on PHerc. 343P and 500P2 (author-reported, Vesuvius Challenge Discord, #general, November 2026). We did not measure stroke width.

It is a demonstration that the 12 GB recipe adapts the winner's model to new labels, not a result about Scroll 1.

**Question.** Can the winner's A-hn model, retrained with this repository's 12 GB recipe, learn from a new scroll's labels and improve on papyrus it never saw? The protocol ([`docs/finetune-protocol-2026-11-17.md`](docs/finetune-protocol-2026-11-17.md)) was committed before the first GPU run, and its amendment adding pair 2 before the first GPU run of pair 2. This folder has no git history, so the timeline below gives the commit times from our private history, which the judges can have on request, against the first GPU step in each run log.

**Timeline** (2026-11-17 unless noted, US Eastern; run logs, raw scores and configs in [`docs/finetune-runs/`](docs/finetune-runs/README.md)):

| time | event |
|---|---|
| 06:04:57 | protocol committed (commit 40e2eac in our private history) |
| 06:29:11, 07:39:54 | two chain starts stop in the CPU data chain, before any GPU step |
| 07:59:00 | first GPU step: the order check runs out of memory after 196 s (see "Deviations") |
| 08:07:00 | pair 1's reported run (`20261117-080700-32067`) starts; one pass to `FT_DONE` at 11:36:22 |
| 19:22:30 | Amendment 1 committed (cc5b5c7); pair 2's CPU data chain starts at 19:22:33 |
| 19:50:15 | a one-sentence wording change to the amendment (a2deba9) |
| 23:56:19 | first pair-2 GPU step; run `20261117-192233-47550`, one pass to `FT_DONE` at 04:11:29 on 2026-11-18 |

**Data.**
- **Volume:** Scroll 1 (PHercParis4), scan `20260411134726`, level 2.
- **Labels:** the organisers' `ink_9um` aligned labels.
- **Pair 1:** train A = segment w03 (`20231106155351`), test B = w09 (`20230929220926`).
- **Pair 2:** train A2 = w06 (`20231210121321`), test B2 = w00 (`20231016151002`). Of the six unused labelled segments (w00, w01, w02, w05, w06, w07), only w00 with w06 and w00 with w07 are clear of both pair-1 segments and of each other in the geometry check; w07's level-2 volume is 34 GB against w06's 12.4 GB, so we took w06. (This choice rests on our geometry gate over all eight segments, which isn't published; the public geometry file covers the four segments used.) The larger segment trains.
- **No shared papyrus detected:** for every ordered pairing of these four segments, none of 60,000 sampled labelled points of one segment lies within 50 voxels of the 2.4 µm scan (0.12 mm) of the other's surface. For 11 of the 12 pairings the median distance is above the 500-voxel (1.2 mm) search cap; for the twelfth (w03's points against w06's surface) it is 438 voxels (1.05 mm). Each pair's training and test segments lie 3.9 mm (pair 1) and 3.6 mm (pair 2) apart in median radius from the scroll axis ([`docs/finetune-geometry-2026-11-17.json`](docs/finetune-geometry-2026-11-17.json)). This is a sampled check, not a proof over every point of the papyrus.
- **Packing:** the winner's own packing code, unchanged (`pc/finetune/`).

**Layer order, decided on the training segment only.** The shipped s0, zero-shot, scores AUC 0.861 on A with the stored order and 0.461 reversed, and 0.869 and 0.507 on A2. The stored order was used for every arm of both pairs, and the kill line (< 0.55 in both orders) was not reached.

**Arms** (the same for both pairs).
- **(a)** The shipped s0, zero-shot.
- **(b)** s0 plus the pair's training segment (A or A2) as a third anchor pack, 300 steps, seeds 1 and 2.
- **(c)** The same 300 steps without it, seeds 1 and 2. This separates "more training" from "new labels".

**Pair 1, on B** (inside its supervision mask, 5.24 M px, 28% ink):

| arm | AUC | top-1% precision | F0.5 at 0.5 |
|---|---|---|---|
| (a) shipped s0, zero-shot | 0.899 | 0.971 | 0.716 |
| (b1) + A, seed 1 | **0.945** | 0.942 | 0.784 |
| (b2) + A, seed 2 | **0.949** | 0.962 | 0.822 |
| (c1) no A, seed 1 | 0.895 | 0.961 | 0.587 |
| (c2) no A, seed 2 | 0.896 | 0.982 | 0.713 |

**Pair 2, on B2** (inside its supervision mask, 5.80 M px, 23% ink):

| arm | AUC | top-1% precision | F0.5 at 0.5 |
|---|---|---|---|
| (a) shipped s0, zero-shot | 0.828 | 0.920 | 0.597 |
| (b1) + A2, seed 1 | **0.964** | 0.995 | 0.845 |
| (b2) + A2, seed 2 | **0.969** | 0.999 | 0.763 |
| (c1) no A2, seed 1 | 0.850 | 0.940 | 0.479 |
| (c2) no A2, seed 2 | 0.830 | 0.936 | 0.579 |

**Verdict: the protocol's success rule is met on both pairs.**
- **Pair 1:** both (b) seeds beat (a) by +0.046 and +0.050 AUC and both (c) seeds by +0.050 to +0.054. Each of these six differences has a paired 95% bootstrap interval over 36 tiles of 512 px that excludes 0; the lowest lower bound is +0.028. (The bootstrap weights the tiles equally, see "Deviations".)
- **Pair 2:** both (b) seeds beat (a) by +0.136 and +0.141 AUC and both (c) seeds by +0.113 to +0.139. Each interval, over 48 tiles, excludes 0; the lowest lower bound is +0.079.
- **Training alone:** on pair 1 the (c) seeds stay at (a)'s level (−0.004 and −0.003; both intervals include 0). On pair 2 one (c) seed gains +0.023 over (a) (interval +0.013 to +0.028) and the other +0.002 (interval includes 0). The two pair-2 control seeds also differ from each other by 0.020 (interval +0.011 to +0.025). So with two seeds per arm, shifts of about 0.02 AUC can come from the seed alone, from more training, or both; the (b) gains over the controls, +0.11 or more on pair 2 and +0.05 or more on pair 1, are well above that.

**What differs between the pairs.** The gain over zero-shot is about three times larger on pair 2, where zero-shot starts lower (0.828 against 0.899). Its size depends on the segment, and two pairs can't say how much it varies. Top-1% precision also moves differently. On pair 1 it is slightly lower for (b) (0.942 and 0.962) than for (a) (0.971) and (c) (up to 0.982): there the new labels improve the ranking across the mask as a whole but make the very top-ranked pixels slightly less precise. On pair 2 it is higher for (b) (0.995 and 0.999) than for (a) (0.920) and (c) (0.940 and 0.936). F0.5 at the fixed 0.5 threshold is higher for (b) than for (a) in both pairs (0.784 and 0.822 against 0.716; 0.845 and 0.763 against 0.597), but it swings with the seed: (b) differs by 0.08 between seeds on pair 2, and (c1) falls below (a) on both pairs (0.587 and 0.479). Read it as sensitive to calibration at one threshold, not as a stable measure at two seeds.

**Cost:** 3.4 GPU-hours for pair 1 as the chain counts it and 4.1 for pair 2, on the RTX 4070. Each training run takes 24–27 min (whole-card peak 9,747 MiB), each map of a test segment 14–17 min (whole-card peak 10,424 MiB), and the two-order check 33 min on A and 61 min on the larger A2. All numbers are in [`docs/finetune-results-2026-11-17.json`](docs/finetune-results-2026-11-17.json) (pair 2 under `pair_2`), built from the raw files in [`docs/finetune-runs/`](docs/finetune-runs/README.md). Pair 1's count (12,411 s) includes the 196 s failed attempt; with the row-block test's two runs (87 s, 08:05–08:06, outside the chain's counter) it is 12,498 s, 3.5 GPU-hours.

**Deviations from the protocol, all small, none touching the arms, data split, metrics or rule:**
- **Row blocks.** The first GPU attempt ran out of memory after 196 s in the order check, because `eval_map.py` uploaded the whole crop at once. It now evaluates in row blocks. On a 640 × 594 test crop of w03, blocks of 150 rows and one pass give the same finite pixels, 99.78% bit-equal, with a largest difference of 2.4e-4, one float16 step ([`docs/finetune-runs/blocktest.json`](docs/finetune-runs/blocktest.json)).
- **Planes 7–102,** not 7–103 as the protocol's input recipe says; 7–103 can't pool into 24 groups of 4 (dated erratum in the protocol).
- **Tile weighting.** The bootstrap subsamples each 512 px tile to at most 4,000 pixels, so tiles count equally (tiles need more than 10% valid pixels), while the point estimates use every valid pixel. A point estimate can therefore sit near one end of its interval (pair 2, b1 − a: +0.136, interval +0.098 to +0.148). Every interval is far from 0 either way.
- **The resume bug, found and fixed after both pairs ran.** The protocol says a pause changes no result. As run, `ft_chain.sh` would have flipped the stored layers back on a rerun after the order decision, so that was not true of the code. Each pair ran in one pass after its order check (runs `20261117-080700-32067` and `20261117-192233-47550`, logs in `docs/finetune-runs/`), so no published number is affected. The chain now flips each segment at most once (its `FLIPPED` marker), stops with a message if a flip was interrupted mid-way, and deletes the training segment's old-order pack before any swap.

**Images and data.** This repository publishes **metrics only**. The Scroll 1 data terms forbid publishing ink-label images, or text revealed from them, without the Vesuvius Challenge's written consent. So no Scroll 1 map, label or image is included, and no data is redistributed. The data is CC BY-NC 4.0 (Vesuvius Challenge open data) and is credited in [`CREDITS.md`](CREDITS.md). Data used in the preparation of this work were obtained from the EduceLab-Scrolls dataset (Parsons et al., "EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using X-ray CT", arXiv:2304.02084, 2023).

### Running the demo

After steps 0 and 1 of "How to run it" (data directory, download, image load). Each command runs for minutes to hours, so run it detached, for example in `tmux`. The data chain needs 40 GB free on the data drive (its floor) and reads about 1.3–2.7 GB per segment from the public buckets, 2.4–4.6 GB on disk each.

```bash
F=$VESUVIUS_DATA/finetune; mkdir -p $F
# pair 1: fetch, check and pack w03 and w09 (CPU)
TAGS="w03 w09" bash pc/finetune/data_chain.sh >> $F/data_chain.log 2>&1
# order check on w03, four 300-step training runs, five maps of w09, scores (GPU, about 3.4 h)
bash pc/finetune/ft_chain.sh >> $F/ft_chain.log 2>&1
# pair 2 (Amendment 1): w06 is packed after its order check
TAGS="w06 w00" PACK="" bash pc/finetune/data_chain.sh >> $F/data_chain_p2.log 2>&1
A_TAG=w06 B_TAG=w00 PFX=ft2 RES_NAME=results_p2 DLOG=$F/data_chain_p2.log bash pc/finetune/ft_chain.sh >> $F/ft_chain_p2.log 2>&1
```

Each chain ends its log with a marker: `P4DATA_DONE` or `P4DATA_FAIL`, and `FT_DONE`, `FT_KILL` or `FT_FAIL`. Scores land in `$F/results/score_B.json` (pair 2: `$F/results_p2/`). Rerunning a command resumes from its finished steps. `ft_chain.sh` stops with a message if the data chain hasn't finished.

**Fine-tune on your own labelled segment.** Put each segment in `$VESUVIUS_DATA/finetune/paris4/<tag>/`, in the layout `pc/finetune/fetch_paris4.py` writes:
- `labels/inklabels.png` and `labels/mask.png`: uint8, 0 or 255, the ink labels and the supervision mask;
- `layers/00.tif` … `layers/23.tif`: uint8, 24 layers around the surface, the same shape as the labels, in the scan's order. The chain's order check on your training segment decides which way to feed them. The demo used about 9.6 µm layers against the model's 8.64 µm.

The data chain skips the download for a folder that already has `layers/23.tif` and `labels/mask.png`, and only checks it:

```bash
TAGS="mytrain mytest" PACK="" bash pc/finetune/data_chain.sh >> $F/data_chain_my.log 2>&1
A_TAG=mytrain B_TAG=mytest PFX=my RES_NAME=results_my DLOG=$F/data_chain_my.log bash pc/finetune/ft_chain.sh >> $F/ft_chain_my.log 2>&1
```

The training segment becomes a third anchor pack (`p4mytrain`) beside the winner's two, and the test segment is only mapped and scored, inside its mask. Use each segment folder in one pair only: the chain may flip a folder's stored layers, and a folder already flipped by an earlier pair would leave a new test segment in the wrong order. The fine-tuned checkpoints are in `$F/self/runs/my_b1/final/` and `my_b2/final/`.

## Apple Silicon

See [`apple-silicon.md`](apple-silicon.md). All five maps (TRACE v3B and A-hn s0, s1, s2, decoy) and the winner's own fusion (`fuse_v7.py`) run on an Apple M5 Pro (24 GB, macOS 27.0.1) through MPS, on the B0 native window of the claim render (B0 is our name for a fixed test window around the claim box, about 5.85 cm²; window from [`docs/b0-baseline-2026-11-15.md`](docs/b0-baseline-2026-11-15.md), code `pc/b0/`). Only a device shim (`mac/mps_shim.py`) and a small harness (`mac/run_ahn.py`, `mac/run_trace.py`) are added, with no model, weight or method change. Every map matches the 4070's CUDA map to float16 precision: TRACE r 0.999999, members r ≥ 0.99999993, fused product r 0.99999977 (top-1% Jaccard 0.999). Measured on both machines: TRACE takes 443.8 s against 260.2 s on the 4070 (1.71×, model compute), and the whole product + decoy chain runs at 1.21 cm²/GPU-h against the 4070's 5.36. Every number has a source file in [`docs/apple-silicon/`](docs/apple-silicon/README.md).

## Licence and credits

- **Our code and documentation:** MIT ([`LICENSE`](LICENSE)).
- **Maps, numbers and images derived from the winner's or the Vesuvius Challenge data:** CC BY-NC 4.0 ([`LICENSE-DATA`](LICENSE-DATA)), with credit to the winner and the Vesuvius Challenge.
- **No weights are shipped.** Retrained checkpoints derive from CC BY-NC 4.0 data and the CC BY-NC-SA 4.0 TRACE v3B weights. The scripts download the winner's own image, weights and data.
- **The winner's code** is MIT and runs inside their image unmodified. Our wrappers set the recipe (inputs, ROI, grid origin, `clip_to_roi`) and the 12 GB config. Full credits: [`CREDITS.md`](CREDITS.md).
