# Details: the retrain, its checks and the fine-tuning demo

This is the full write-up behind the [README](../README.md). File paths in backticks are from the repository root. [`CLAIMS.md`](../CLAIMS.md) lists every claim with its evidence.

Contents: [How to run it](#how-to-run-it) · [The retrain](#the-retrain-scope-cost-and-what-makes-it-fit) · [What "stages 1–8" means](#what-stages-18-means) · [Verification](#verification) · [Demo: fine-tuning on new labels](#demo-fine-tuning-on-new-labels-from-scroll-1)

## How to run it

I tested this on one RTX 4070 12 GB with 16 GB of RAM, on Windows 11 with WSL2 Ubuntu and Docker Desktop with GPU support. I used the native Linux `docker` CLI inside WSL, with a pagefile that can grow. Native Linux is untested, and there the host guard does nothing.

You need an NVIDIA GPU with 12 GB and Docker with GPU support, with `nvidia-smi` working. You also need `git`, `uv` for the Hugging Face downloads and the image load, `tmux` for detached runs, and `python3` on the host, which the fine-tune chain uses to read its scores. The Mac recipe also needs `zstd`, and Docker on some machine to copy the code out of the image.

The settings live in `pc/env.sh`. `VESUVIUS_DATA` is the data directory, and its default is my machine's path, `/mnt/d/Workspace/vesuvius-data`. `DOCKER` is the docker CLI, which defaults to `/usr/bin/docker` when present and to `docker` on PATH otherwise. `NVIDIA_SMI` is found on PATH or at the WSL path. The scripts find each other from their own location, so the repository can live anywhere.

The steps are these.

0. Choose the data directory first, on any drive with about 45 GB free for the retrain, and export it: `export VESUVIUS_DATA=/path/to/vesuvius-data`. Every script reads it.
1. Download about 23 GB with `bash pc/download_winner.sh`. It fetches the winner's repository, their model repo `Prodigle/pherc0306b-ink-ahn` and their dataset `Prodigle/pherc0306b-first-letters`. The dataset includes the Docker image archive `docker/fl-claim-a7_final.tar.zst`, which the script checks against its sha256. Then load the image with `bash pc/zstd_load.sh "$VESUVIUS_DATA/winner/hf-data/docker/fl-claim-a7_final.tar.zst"`.
2. Train end to end with `STAGES=masks,teach_readers,teach_v8,labels,exclusion,packs,anchors_shipped,train,members,fuse bash pc/train_e2e/e2e_chain.sh`. Run it detached, for example in `tmux`. It resumes from finished outputs. Each stage compares its output with the shipped file through `pc/train_e2e/e2e_compare.py` and `pc/exact_all_compare.py`. The `train` stage writes the 12 GB configs and logs whole-card VRAM every 500 ms. Setting `SUFFIX=<name>` trains a second copy without touching the first.
3. Redraw the claim image on the CPU with `bash pc/train_e2e/stage9.sh`.
4. For inference with the exact shipped recipe, run `bash pc/exact_all_chain.sh`. It covers all members, the fusion and the comparison, and runs `pc/exact_chain.sh` first for TRACE and s0 when they are missing. The general driver is `bash pc/ink_run.sh <render.zarr> <outdir> [y0 y1 x0 x1]`.
5. Optionally, rebuild the anchor packs from the raw data with `bash pc/train_e2e/anchors_rebuild.sh`, and run the depth-shuffle and rerun checks with `bash pc/train_e2e/overnight_chain.sh`.
6. For the fine-tune demo, see [Running the demo](#running-the-demo).

### A test from the published file set

I exported this repository's public files with my export script into a new folder with no git history. That was export r6, 188 files, on 2026-11-18, and the export script itself isn't included. I pointed `VESUVIUS_DATA` at a new data directory on another drive that held only the inputs those stages read, and ran from `/tmp` with no other settings. The image archive's sha256 checked OK. The masks stage reproduced the shipped pool masks exactly in all 5 regions, with 25 of 25 checks equal. The claim-image step printed MATCH on the shipped maps and gave the same 0.042% difference on mine.

That test did not run everything. I checked `download_winner.sh` and `zstd_load.sh` for syntax and the archive's sha256 but didn't execute them end to end, the Docker image was already loaded, and the 23 GB download wasn't repeated. The GPU stages from `teach_readers` to `fuse`, `exact_all_chain.sh`, `overnight_chain.sh`, the fine-tune demo and the Mac recipe weren't run from the exported folder either. They ran from my working copy, and their logs are under `docs/`.

Sources: `docs/logs/fresh-clone/` holds the test's script, log and outputs, and `masks.json` the masks result.

Some files changed after that test, in this release. They are the documentation (`README.md`, `CLAIMS.md`, `apple-silicon.md`, the files under `docs/` and the new `docs/finetune-runs/`), the test's own logs, and these scripts:

- In `pc/env.sh`, with no docker installed, the docker variable now falls back to `docker`. A step that needs it then fails with "command not found" instead of the script exiting silently. With docker present, which the tested stages need, nothing changes.
- `pc/finetune/ft_chain.sh`, `data_chain.sh` and `labels_chain.sh` have the resume fix and the portability fixes described in the demo section.
- `mac/fuse_compare.py` skips the comparison when the 4070's reference maps are absent.
- Only comments changed in `pc/b0/b0_make.py`, `pc/b0/b0_chain.sh`, `pc/retrain/retrain_chain.sh` and `docs/logs/fresh-clone/fresh_clone_test.sh`.

### Windows/WSL memory

Under WSL2, GPU memory is charged to Windows commit memory. In a logged training run, Windows free commit fell from 16,021 MB to 1,710 MB. The run took about 14.3 GB of commit: the card's 8,698 MiB of training memory above idle, plus about 5 GiB of training data and workers in the container's RAM. The commit limit was 32.7 GB, which is 16 GB of RAM plus a pagefile already grown to about 16 GB. So on a 16 GB PC, training fits only if the pagefile can grow. My first attempt crashed the WSL VM when a nearly full C: drive stopped the pagefile growing.

The chains therefore run a host guard, `pc/hostguard.sh`. It keeps a free-space floor on C:, runs a commit check that knows when the pagefile can still grow, drops the WSL page cache periodically and caps container memory. During the end-to-end run, the guard twice failed to read Windows commit and halted A-hn s1. Both were false alarms, not memory shortages, and s1 resumed from its own checkpoint both times, over steps 0–370, 370–540 and 540–600. When the guard trips, a chain stops its containers cleanly after the trainer saves its state, and ends with a FAIL line naming the reason. Rerunning the same command resumes from the finished outputs and the trainer's checkpoint.

Source: `docs/train-e2e-compare/host_mem_train_vram.csv`, sampled every 30 s.

## The retrain: scope, cost and what makes it fit

With this repository you can retrain the trained members, A-hn, of the model that won PHerc. 0306B First Letters on one 12 GB consumer card, then redraw its claim image. TRACE v3B, the other half of the claim map, is a third-party model that the winner's `train` mode doesn't train, so it runs with its shipped weights. I did it on an RTX 4070 with 16 GB of RAM, on Windows 11 with WSL2 and Docker Desktop, using the winner's own Docker image, code, weights and data. Their guide says retraining "needs one GPU with 24 GB or more", notes that their training peaked at 19 GB, and marks its `train` mode "not run end to end".

Source: the winner's [`docs/REPRODUCE.md`](https://github.com/SirProdigle/pherc0306b-first-letters/blob/main/docs/REPRODUCE.md), lines 41, 28 and 25–26.

The whole training chain took about 8.4 GPU-hours plus 0.4 CPU-hours. The teacher maps took 1.2 h, the four training runs 47–54 min each, the TRACE map 13 min and the four member maps about 55 min each. The claim image then took 7 min on the CPU.

For memory, I measured the whole card with the desktop included. Training peaked at 9,706 MiB, against 1,008 MiB idle just before that run. A member map peaked at 9,931 MiB and TRACE at 2,285 MiB, against 986 MiB idle before those runs. The chain log's 22:23 TRACE line understates the peak because of a string-compare bug, which its 23:20 line corrects.

Sources: `docs/train-e2e-compare/vram_train_hn4_s0_vram.csv`, `train_vram.log`, `vram_map_s0.csv`, `vram_trace.csv`, and `night_chain.log` for the idle value.

On disk, the downloads come to about 23 GB, with the model at 6.3 GB and the dataset at 17 GB. The unpacked image tar takes 12 GB, the training chain and claim image write about 9 GB of outputs on the data drive, and the loaded Docker image takes another 24 GB. The listing is in `docs/logs/sizes.txt`.

### What makes it fit, and whose idea it is

The winner's trainer already supports 12 GB. `train_cloud_hn.py` defaults to per-block activation checkpointing, `grad_ckpt 'block'`, and a 9 GB budget, `vram_gb 9.0`. Their shipped A-hn configs opt out with `grad_ckpt false` and `vram_gb 30.0`. Their cloud launcher already falls back to block mode after an out-of-memory error, because `cloud_box_hn.sh` sets `GCK_FORCE=block`, which `train_cloud_hn.py` reads. On the 4070, a training step without checkpointing at micro-batch 4 runs out of memory in my BatchNorm-check harness, whose log says "plain arm OOM at micro 4". So my recipe sets their own block mode directly, with a 10.5 GB budget, plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Sources: `train_cloud_hn.py` docstring line 10 and lines 64, 113 and 114; `cloud_box_hn.sh` line 11; `docs/logs/retrain-s0/bn_check.log`.

What I add is four things:

1. the measurement that their block mode fits a 12 GB card and reproduces the whole training chain, which they mark "not run end to end";
2. the check that BatchNorm statistics under checkpointing equal the run without it exactly;
3. the Windows/WSL memory findings, and a host guard without which a 16 GB PC can crash the WSL VM during training, as mine did once when the pagefile couldn't grow;
4. scripts that run every stage in their image and compare it with their shipped artifact.

### Reused inputs

The main training chain used the shipped PHerc. 1447 anchor packs and my earlier exact TRACE map. Follow-up runs rebuilt the anchor packs from the raw public data, byte-identical, and regenerated TRACE inside the chain, bit-identical.

## What "stages 1–8" means

The stage names and numbers are mine. I did not invoke the image's `train` entrypoint. I drove its stage scripts one by one, with the shipped inputs mounted where the scripts expect them, in the order of the image's `run.sh` `train` mode. Stages 1–5 run their scripts unchanged: `masks.py`, `teachers.py`, `labels.py`, `exclusion_audit_v7.py` and `make_packs.py`. Stage 6 runs `train_cloud_hn.py` with the configs above. Stage 7, the member claim maps, uses their inference script `infer.py` with the shipped member recipe in `pc/exact_s0.py`, not `render_leadA.py`. Stage 8 runs `fuse_v7.py`, and the claim image runs their `run.sh final`. I did not run the mesh and render stages of `pipeline` mode, and started from the shipped render.

| stage | mine | result (RES section) |
|---|---|---|
| mesh, render | not run: the shipped meshes and renders | "Scope" |
| 1 pool masks | **fresh** | identical to shipped ("Stages 1–5") |
| 2 teacher maps | **fresh**, shipped teacher weights | r ≥ 0.99999989 |
| 3 machine labels | **fresh** | 189–413 px change class per variant |
| 4 exclusion audit | **fresh** | 23,332 tiles vs 23,331 shipped |
| 5 PHerc0306B packs | **fresh** | CT arrays bit-identical; one extra tile in w053, traced to one float16 step in the v8-in teacher |
| 5b PHerc1447 anchor packs | shipped packs (sha256-checked); separately rebuilt from the raw public stacks with their `make_packs.py` | rebuild **byte-identical** ([`train-e2e-anchors-2026-11-16.md`](train-e2e-anchors-2026-11-16.md)) |
| 6 training: A-hn s0, s1, s2, decoy | **fresh**, block mode, 10.5 GB budget | loss curves r 0.972–0.9999 vs shipped logs ("Stage 6") |
| TRACE claim map | **fresh**, shipped TRACE weights (`train` mode doesn't train TRACE) | bit-identical to my earlier exact run; r 1.0000000 vs shipped |
| 7 member claim maps | **fresh** | r 0.9925–0.9934 (members), 0.9994 (decoy) |
| 8 fusion | **fresh** | r 0.9992 whole, 0.9996 in the claim box |
| claim image | **fresh** (CPU) | 0.042% of pixels differ by more than 8 / 255 |

The results (RES) are in [`train-e2e-results-2026-11-16.md`](train-e2e-results-2026-11-16.md) and the plan in [`train-e2e-plan-2026-11-15.md`](train-e2e-plan-2026-11-15.md). The comparison JSONs and logs are in [`train-e2e-compare/`](train-e2e-compare/) and [`logs/`](logs/).

## Verification

### How close the retrained members are

Each A-hn member was retrained with its own seed, 0, 1 or 2, as shipped. I regenerated the training data from the shipped inputs, so it is slightly different, and GPU training is not bit-exact. I use two yardsticks. One is the training variation in the winner's own run, which is how much their three seeds' maps differ from each other. The other is the same-recipe bar, my first retrain of s0 from the shipped packs.

| pair (A-hn maps) | r, member ROI (27.5 M px) | r, claim box | top-1% Jaccard, claim box |
|---|---|---|---|
| the winner's seeds vs each other (s0~s1, s0~s2, s1~s2) | 0.932–0.942 | 0.957–0.963 | 0.28–0.32 |
| my s0 retrain from the shipped packs vs shipped s0 | 0.99958 | 0.9997 | 0.908 |
| **my end-to-end s0 / s1 / s2 vs shipped** | **0.9926 / 0.9934 / 0.9925** | **0.9944 / 0.9960 / 0.9951** | **0.54 / 0.60 / 0.67** |
| my end-to-end decoy vs shipped | 0.9994 | 0.9995 | 0.87 |
| my A-hn mean (ensA) vs shipped | 0.9974 | 0.9983 | 0.70 |
| **fused claim map** (TRACE × A-hn mean) | **0.9992** | **0.9996** | **0.967** |

Sources: `docs/train-e2e-compare/maps.json`, `fused_topk.json` and `topk_ref.json`, from `pc/train_e2e/topk_ref.py`. The seed spread over the member ROI is from [`retrain-s0-2026-11-15.md`](retrain-s0-2026-11-15.md).

Each end-to-end member is much closer to its shipped counterpart than the winner's seeds are to each other, but not as close as a retrain from the shipped packs. They fall short of the same-recipe bar because the training data differs slightly. The regenerated labels change class at 189–413 px per variant, and one extra w053 tile passes the packer's 20%-labelled threshold, at 820 against 819.2 px. GPU non-determinism in training adds to that, and I haven't measured the share of each.

I traced the extra tile to a single pixel where the v8-in teacher map differs by exactly one float16 step, from fp16 inference without cuDNN determinism. That pixel moves the fused teacher z from 0.300127 to 0.299445, across the labeller's 0.3 threshold. The RES section "Stages 7–8" has the detail. Most of the fused-map agreement comes from TRACE, which both sides share and which reproduces exactly.

The 12 GB config keeps everything else as shipped: micro 4 × accum 4 for an effective batch of 16, each member's own seed, lr 2e-5 → 2e-6 cosine over 600 steps, bf16 autocast and BatchNorm in train mode. With the trainer's own restore, BatchNorm buffers under checkpointing equal the run without it exactly, with a difference of 0.0. I checked this at micro-batch 2, because the run without checkpointing doesn't fit 12 GB at micro 4. The check is `pc/retrain/bn_check.py`, and its log is `docs/logs/retrain-s0/bn_check.log`.

### Determinism and the depth-shuffle control

Regenerating TRACE gives a map bit-identical to my earlier run, and re-running the s0 member map from the same checkpoint gives a bit-identical map. Re-fusing with the regenerated TRACE reproduces every number above.

The winner's own `shuffle_control.py` permutes render layers 8–31 around the claim box, with seed 1, and on that render my retrained models lose the claim. In the claim box, 4.86 M px, the fused map correlates with the unshuffled one at only r 0.051, and its top-1% pixels overlap at Jaccard 0.0003. TRACE ~ A-hn agreement falls from 0.767 to 0.123. The winner's own check with their checkpoints shows the same collapse, a fused r of 0.0495 for the lead view.

Sources: the driver [`pc/train_e2e/overnight_chain.sh`](../pc/train_e2e/overnight_chain.sh) and its log `docs/train-e2e-compare/night_chain.log`; `rerun_v2.json`, `maps_v2.json` and `shuffle.json` in `docs/train-e2e-compare/`; the winner's `submission/method/checks/depth_shuffle_variants_v7_trace_x_Ahn_product.json`, view `c6_grey`.

### The claim image from my maps

I redrew the claim image with the winner's `final` mode, on the CPU, from my end-to-end maps, and pixel-diffed it against their shipped claim image. As a check of the tool, `final` on the shipped maps reproduces the package's PNGs byte for byte. Its log shows the lead view `c6_grey` is `variants/TRACE_x_v8306H3_grey_page.png`.

| lead claim image (TRACE × A-hn, 4417 × 7385 px) | value |
|---|---|
| pixels differing by more than 8 / 255 | 0.042% (13,580) |
| pixels differing by more than 64 / 255 | 148 (0.00045%), all in one caption digit: display level "z 7.96" vs mine "z 7.95" |
| mean abs difference | 0.40 / 255 |
| byte-identical | no (retrained weights cannot be) |

11 of the 20 image variants are byte-identical. 4 more, which use my TRACE map but no retrained member, differ by at most 1 / 255. The variant most exposed to the retrain, the A-hn mean alone, differs by more than 8 / 255 on 3.3% of pixels. There is a side-by-side strip of the claim box in [`img/stage9-claimbox-strip-shipped-vs-ours.png`](img/stage9-claimbox-strip-shipped-vs-ours.png), and one of the caption digit in [`img/stage9-header-shipped-vs-ours.png`](img/stage9-header-shipped-vs-ours.png).

Sources: [`train-e2e-stage9-2026-11-16.md`](train-e2e-stage9-2026-11-16.md), [`pc/train_e2e/stage9.sh`](../pc/train_e2e/stage9.sh) and `docs/logs/stage9/shipped.log`.

### Inference on 12 GB: every map the claim uses, to float16 precision

These runs use the winner's image, the shipped claim-surface render of 41 × 4900 × 7780, the shipped member recipe and the shipped checkpoints. The recipe sets the ROI, the grid origin and `clip_to_roi`. The maps covered are every map the claim and its controls are built from: TRACE, the A-hn members, the decoy, their fusions and the shipped depth-shuffle maps. The winner also ships five reading-aid maps that the claim doesn't use, `fwd_reader_v2`, `fwd_reader_v3`, `fwd_v8in`, `fwd_v8in_3sigma` and `fwd_v8Q`. I didn't reproduce those, and my claim-image run in stage 9 takes them as shipped.

| map | Pearson r vs shipped | max abs diff |
|---|---|---|
| TRACE v3B | 1.0000000 (99.99988% of pixels bit-equal) | 7.3e-4 |
| A-hn s0 / s1 / s2 | 0.9999999 each | 6.9e-4 – 8.0e-4 |
| decoy | 0.9999997 | 1.02e-3 |
| A-hn mean, fused product, decoy product | 0.9999998 – 0.9999999 | 4.9e-4 |
| depth-shuffle TRACE | 1.0 (bit-identical) | 0 |
| depth-shuffle A-hn s0 / s1 / s2 | 0.9999947 / 0.9999957 / 0.9999989 | 4.3e-4 / 4.1e-4 / 5.9e-4 |
| depth-shuffle A-hn mean, fused product | 0.9999966 / 0.9999979 | 4.9e-4 / 1.2e-4 |

The differences are at the float16 storage step of the shipped maps, and only 2 decoy pixels exceed 1e-3. I made the depth-shuffle maps with the winner's shuffle recipe, with the claim-box ROI and `clip_to_roi` off, on a shuffled render built by their own script. The winner ships no shuffled decoy map. On the 4070, whole-render times are 12–14 min for TRACE and 53–63 min per A-hn member, and the whole card peaked at 10,180 MiB with the desktop included.

Sources: [`fidelity-exact-all-2026-11-15.md`](fidelity-exact-all-2026-11-15.md) and its JSON; `pc/train_e2e/shuf_vs_shipped.py` and `docs/train-e2e-compare/shuf_vs_shipped.json`; [`fidelity-2026-11-13.md`](fidelity-2026-11-13.md) and the logs in `docs/logs/fidelity-2026-11-13/`.

## Demo: fine-tuning on new labels from Scroll 1

### Limits

Read these first. The demo uses one scroll and two train/test pairs, and each pair trains on one segment and tests on another. I added the second pair by a dated amendment after the first pair's result and before any of its GPU runs, with the same arms, metrics, success rule and kill rule.

The test segments' labels are the organisers' hand strokes refined by iterative pseudo-labelling, and no manually labelled Scroll 1 segment exists in the release. The training and test segments' labels come from the same model-in-the-loop labelling, so part of the gain may be agreement with that labeller rather than with the ink. With no hand-labelled Scroll 1 segment, I can't separate the two.

The scan is a different one from the model's: the ESRF 2.4 µm scan, resampled to 9.6 µm, against the model's 8.64 µm. Each arm has two seeds. I did not check how well the fine-tuned models still read PHerc. 0306B.

Part of a stroke's width in any ink map comes from how thick its training labels were drawn, and my scores compare pixels with labels, so they inherit that thickness. For scale, Yohann reports that Reader v3 paints strokes about 1.3× wider than the ink in infrared photos on PHerc. 343P and 500P2 (author-reported, Vesuvius Challenge Discord, #general, November 2026). I did not measure stroke width.

So this is a demonstration that the 12 GB recipe adapts the winner's model to new labels, not a result about Scroll 1.

### Question and timeline

Can the winner's A-hn model, retrained with this repository's 12 GB recipe, learn from a new scroll's labels and improve on papyrus it never saw? I committed the protocol, [`finetune-protocol-2026-11-17.md`](finetune-protocol-2026-11-17.md), before the first GPU run, and its amendment adding pair 2 before the first GPU run of pair 2. The published repository has no history from that time, so the timeline below gives the commit times from my private history, which the judges can have on request, against the first GPU step in each run log.

All times are on 2026-11-17 unless noted, US Eastern. The run logs, raw scores and configs are in [`finetune-runs/`](finetune-runs/README.md).

| time | event |
|---|---|
| 06:04:57 | protocol committed (commit 40e2eac in my private history) |
| 06:29:11, 07:39:54 | two chain starts stop in the CPU data chain, before any GPU step |
| 07:59:00 | first GPU step: the order check runs out of memory after 196 s (see "Deviations") |
| 08:07:00 | pair 1's reported run (`20261117-080700-32067`) starts; one pass to `FT_DONE` at 11:36:22 |
| 19:22:30 | Amendment 1 committed (cc5b5c7); pair 2's CPU data chain starts at 19:22:33 |
| 19:50:15 | a one-sentence wording change to the amendment (a2deba9) |
| 23:56:19 | first pair-2 GPU step; run `20261117-192233-47550`, one pass to `FT_DONE` at 04:11:29 on 2026-11-18 |

### Data

The volume is Scroll 1, PHercParis4, scan `20260411134726` at level 2, and the labels are the organisers' `ink_9um` aligned labels. Pair 1 trains on A, segment w03 (`20231106155351`), and tests on B, w09 (`20230929220926`). Pair 2 trains on A2, w06 (`20231210121321`), and tests on B2, w00 (`20231016151002`). The larger segment of each pair trains.

Of the six unused labelled segments, w00, w01, w02, w05, w06 and w07, only w00 with w06 and w00 with w07 are clear of both pair-1 segments and of each other in the geometry check. w07's level-2 volume is 34 GB against w06's 12.4 GB, so I took w06. This choice rests on my geometry gate over all eight segments, which isn't published. The public geometry file covers the four segments used.

I found no shared papyrus. For every ordered pairing of these four segments, none of 60,000 sampled labelled points of one segment lies within 50 voxels of the 2.4 µm scan, 0.12 mm, of the other's surface. For 11 of the 12 pairings the median distance is above the 500-voxel search cap of 1.2 mm. For the twelfth, w03's points against w06's surface, it is 438 voxels, 1.05 mm. Each pair's training and test segments lie 3.9 mm (pair 1) and 3.6 mm (pair 2) apart in median radius from the scroll axis. This is a sampled check, not a proof over every point of the papyrus. The packing is the winner's own code, unchanged, in `pc/finetune/`.

Source: [`finetune-geometry-2026-11-17.json`](finetune-geometry-2026-11-17.json).

### Layer order, decided on the training segment only

The shipped s0, zero-shot, scores AUC 0.861 on A with the stored order and 0.461 reversed, and 0.869 and 0.507 on A2. I used the stored order for every arm of both pairs, and the kill line, below 0.55 in both orders, was not reached.

### Arms

The arms are the same for both pairs. Arm (a) is the shipped s0, zero-shot. Arm (b) is s0 plus the pair's training segment, A or A2, as a third anchor pack, trained for 300 steps with seeds 1 and 2. Arm (c) is the same 300 steps without it, with seeds 1 and 2, which separates "more training" from "new labels".

### Results

Pair 1, on B, inside its supervision mask of 5.24 M px with 28% ink:

| arm | AUC | top-1% precision | F0.5 at 0.5 |
|---|---|---|---|
| (a) shipped s0, zero-shot | 0.899 | 0.971 | 0.716 |
| (b1) + A, seed 1 | **0.945** | 0.942 | 0.784 |
| (b2) + A, seed 2 | **0.949** | 0.962 | 0.822 |
| (c1) no A, seed 1 | 0.895 | 0.961 | 0.587 |
| (c2) no A, seed 2 | 0.896 | 0.982 | 0.713 |

Pair 2, on B2, inside its supervision mask of 5.80 M px with 23% ink:

| arm | AUC | top-1% precision | F0.5 at 0.5 |
|---|---|---|---|
| (a) shipped s0, zero-shot | 0.828 | 0.920 | 0.597 |
| (b1) + A2, seed 1 | **0.964** | 0.995 | 0.845 |
| (b2) + A2, seed 2 | **0.969** | 0.999 | 0.763 |
| (c1) no A2, seed 1 | 0.850 | 0.940 | 0.479 |
| (c2) no A2, seed 2 | 0.830 | 0.936 | 0.579 |

The protocol's success rule is met on both pairs. On pair 1, both (b) seeds beat (a) by +0.046 and +0.050 AUC and both (c) seeds by +0.050 to +0.054. Each of these six differences has a paired 95% bootstrap interval over 36 tiles of 512 px that excludes 0, and the lowest lower bound is +0.028. The bootstrap weights the tiles equally, as "Deviations" explains. On pair 2, both (b) seeds beat (a) by +0.136 and +0.141 AUC and both (c) seeds by +0.113 to +0.139. Each interval, over 48 tiles, excludes 0, and the lowest lower bound is +0.079.

Training alone moves things less. On pair 1 the (c) seeds stay at (a)'s level, at −0.004 and −0.003, and both intervals include 0. On pair 2 one (c) seed gains +0.023 over (a), with an interval of +0.013 to +0.028, and the other gains +0.002, with an interval that includes 0. The two pair-2 control seeds also differ from each other by 0.020, with an interval of +0.011 to +0.025. So with two seeds per arm, shifts of about 0.02 AUC can come from the seed alone, from more training, or both. The (b) gains over the controls, +0.11 or more on pair 2 and +0.05 or more on pair 1, are well above that.

### What differs between the pairs

The gain over zero-shot is about three times larger on pair 2, where zero-shot starts lower, at 0.828 against 0.899. Its size depends on the segment, and two pairs can't say how much it varies.

Top-1% precision also moves differently. On pair 1 it is slightly lower for (b), at 0.942 and 0.962, than for (a) at 0.971 and (c) at up to 0.982. There the new labels improve the ranking across the mask as a whole but make the very top-ranked pixels slightly less precise. On pair 2 it is higher for (b), at 0.995 and 0.999, than for (a) at 0.920 and (c) at 0.940 and 0.936.

F0.5 at the fixed 0.5 threshold is higher for (b) than for (a) in both pairs: 0.784 and 0.822 against 0.716, and 0.845 and 0.763 against 0.597. But it swings with the seed. (b) differs by 0.08 between seeds on pair 2, and (c1) falls below (a) on both pairs, at 0.587 and 0.479. Read it as sensitive to calibration at one threshold, not as a stable measure at two seeds.

### Cost

Pair 1 took 3.4 GPU-hours as the chain counts it and pair 2 took 4.1, on the RTX 4070. Each training run takes 24–27 min, with a whole-card peak of 9,747 MiB. Each map of a test segment takes 14–17 min, with a whole-card peak of 10,424 MiB. The two-order check takes 33 min on A and 61 min on the larger A2. Pair 1's count of 12,411 s includes the 196 s failed attempt. With the row-block test's two runs, 87 s between 08:05 and 08:06 and outside the chain's counter, it is 12,498 s, or 3.5 GPU-hours.

Sources: [`finetune-results-2026-11-17.json`](finetune-results-2026-11-17.json), with pair 2 under `pair_2`, built from the raw files in [`finetune-runs/`](finetune-runs/README.md).

### Deviations

Every deviation from the protocol is small, and none touches the arms, data split, metrics or rule.

- Row blocks. The first GPU attempt ran out of memory after 196 s in the order check, because `eval_map.py` uploaded the whole crop at once. It now evaluates in row blocks. On a 640 × 594 test crop of w03, blocks of 150 rows and one pass give the same finite pixels and are 99.78% bit-equal, with a largest difference of 2.4e-4, one float16 step. The test is [`finetune-runs/blocktest.json`](finetune-runs/blocktest.json).
- Planes 7–102, not 7–103 as the protocol's input recipe says. 7–103 can't pool into 24 groups of 4, and the protocol has a dated erratum.
- Tile weighting. The bootstrap subsamples each 512 px tile to at most 4,000 pixels, so tiles count equally, and tiles need more than 10% valid pixels. The point estimates use every valid pixel. A point estimate can therefore sit near one end of its interval, as on pair 2, where b1 − a is +0.136 with an interval of +0.098 to +0.148. Every interval is far from 0 either way.
- The resume bug, found and fixed after both pairs ran. The protocol says a pause changes no result. As run, `ft_chain.sh` would have flipped the stored layers back on a rerun after the order decision, so that was not true of the code. Each pair ran in one pass after its order check, in runs `20261117-080700-32067` and `20261117-192233-47550` with logs in `docs/finetune-runs/`, so no published number is affected. The chain now flips each segment at most once, recorded by its `FLIPPED` marker. It stops with a message if a flip was interrupted mid-way, and deletes the training segment's old-order pack before any swap.

### Images and data

This repository publishes metrics only for Scroll 1. The Scroll 1 data terms forbid publishing ink-label images, or text revealed from them, without the Vesuvius Challenge's written consent. So no Scroll 1 map, label or image is included, and no data is redistributed. The data is CC BY-NC 4.0, Vesuvius Challenge open data, and is credited in [`CREDITS.md`](../CREDITS.md). Data used in the preparation of this work were obtained from the EduceLab-Scrolls dataset (Parsons et al., "EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using X-ray CT", arXiv:2304.02084, 2023).

### Running the demo

Start after steps 0 and 1 of [How to run it](#how-to-run-it), the data directory, the download and the image load. Each command runs for minutes to hours, so run it detached, for example in `tmux`. The data chain needs 40 GB free on the data drive, which is its floor. It reads about 1.3–2.7 GB per segment from the public buckets and takes 2.4–4.6 GB on disk for each.

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

Each chain ends its log with a marker. The data chain writes `P4DATA_DONE` or `P4DATA_FAIL`, and the fine-tune chain writes `FT_DONE`, `FT_KILL` or `FT_FAIL`. Scores land in `$F/results/score_B.json`, and pair 2's in `$F/results_p2/`. Rerunning a command resumes from its finished steps, and `ft_chain.sh` stops with a message if the data chain hasn't finished.

### Fine-tune on your own labelled segment

Put each segment in `$VESUVIUS_DATA/finetune/paris4/<tag>/`, in the layout that `pc/finetune/fetch_paris4.py` writes. `labels/inklabels.png` and `labels/mask.png` are the ink labels and the supervision mask, as uint8 with values 0 or 255. `layers/00.tif` to `layers/23.tif` are 24 uint8 layers around the surface, the same shape as the labels, in the scan's order. The chain's order check on your training segment decides which way to feed them. The demo used about 9.6 µm layers against the model's 8.64 µm.

The data chain skips the download for a folder that already has `layers/23.tif` and `labels/mask.png`, and only checks it:

```bash
TAGS="mytrain mytest" PACK="" bash pc/finetune/data_chain.sh >> $F/data_chain_my.log 2>&1
A_TAG=mytrain B_TAG=mytest PFX=my RES_NAME=results_my DLOG=$F/data_chain_my.log bash pc/finetune/ft_chain.sh >> $F/ft_chain_my.log 2>&1
```

The training segment becomes a third anchor pack, `p4mytrain`, beside the winner's two, and the test segment is only mapped and scored, inside its mask. Use each segment folder in one pair only. The chain may flip a folder's stored layers, and a folder already flipped by an earlier pair would leave a new test segment in the wrong order. The fine-tuned checkpoints are in `$F/self/runs/my_b1/final/` and `my_b2/final/`.
