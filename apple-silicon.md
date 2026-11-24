# Appendix: the same pipeline on Apple Silicon (M5 Pro)

This is a supporting note to the main entry, which is about the RTX 4070.

The winner's inference code also runs on an Apple M5 Pro through PyTorch's MPS backend. That code is `ink_v2.py` for TRACE and `infer.py` for A-hn. I ran all five maps and the fusion on the B0 native window, which is 1,399 × 5,600 px of the claim render around the claim box, about 5.85 cm² of the claim. The maps are TRACE v3B and the four A-hn members, s0, s1, s2 and the decoy. Every map matches the 4070's CUDA map to float16 precision. On that window the Mac is 1.7× slower on TRACE and 4–6× slower per A-hn member. A separate path, which rewrites A-hn for Core ML and is gated rather than float16-exact, has [its own section](#a-faster-a-hn-path-the-same-network-as-2-d-convolutions-on-core-ml). Every number below has a source file in [`docs/apple-silicon/`](docs/apple-silicon/README.md).

## What I change

I add a device shim, `mac/mps_shim.py`, and a small harness, `mac/run_ahn.py` and `mac/run_trace.py`. On this path there is no model, weight or method change.

The shim redirects the reference code's hardcoded `torch.autocast('cuda', dtype=torch.float16)` to MPS at the same fp16 precision, and sets the device. The harness calls the winner's `infer.render` and `ink_v2.py` with the same recipe as the 4070 run: weights, crop, ROI, grid origin and `clip_to_roi`.

`run_ahn.py` wraps `infer.band_sums` only to time it. The wrapper synchronises the GPU before and after each band, records peak memory, and returns the original function's result untouched. Its benchmark-only `--max-tiles` mode truncates the tile list and writes no map, so it can't change a map. The fusion is the winner's own `fuse_v7.py`, unchanged, called from `mac/fuse_compare.py`.

## Exactness on the B0 native window, Mac against the 4070's CUDA maps

| map | pixels | Pearson r | max abs diff | same 8-bit level | top-1 % Jaccard | NaN mismatches |
|---|---|---|---|---|---|---|
| TRACE v3B | 7,841,400 | **0.999999** | 2.4e-3 | 99.4 % | – | 0 |
| A-hn s0 | 7,749,169 | **0.99999996** | 6.3e-4 | 98.2 % | – | 0 |
| A-hn s1 | 7,749,169 | **0.99999996** | 5.6e-4 | 98.4 % | – | 0 |
| A-hn s2 | 7,749,169 | **0.99999995** | 5.9e-4 | 98.4 % | – | 0 |
| A-hn decoy | 7,749,169 | **0.99999993** | 3.7e-4 | 99.0 % | – | 0 |
| fused A (mean of s0–s2) | 7,749,169 | **0.99999988** | 4.9e-4 | – | 0.994 | 0 |
| **fused product (TRACE × A)** | 7,749,169 | **0.99999977** | 2.0e-3 | – | **0.999** | 0 |
| fused decoy control (TRACE × decoy) | 7,749,169 | **0.99999962** | 1.6e-3 | – | 0.999 | 0 |

The max differences sit at float16 storage precision, the same size as the 4070's own differences from the shipped maps, which are in [`docs/details.md`](docs/details.md#inference-on-12-gb-every-map-the-claim-uses-to-float16-precision). Against the winner's shipped product map, on the 4.86 M-pixel claim box inside the window, the Mac's product map gives r 0.99999976 and top-1 % Jaccard 0.994. The 4070's gives r 0.99999994. Outside the box, both the Mac's and the 4070's window runs differ from the shipped full-render map at the window edges in the same way. That is an effect of the crop, not of the device.

## Speed and memory

I measured these on a MacBook Pro M5 Pro with a 16-core GPU and 24 GB, on macOS 27.0.1 (26A434).

| B0 native window, 5.85 cm² | M5 Pro | RTX 4070 | ratio |
|---|---|---|---|
| TRACE v3B, compute | 443.8 s | 260.2 s | 1.71× |
| A-hn s0, wall | 4,396 s | 907 s | 4.8× |
| A-hn s1, wall | 3,551 s | 905 s | 3.9× |
| A-hn s2, wall | 3,631 s | 906 s | 4.0× |
| A-hn decoy, wall | 5,366 s | 905 s | 5.9× |
| **Product + decoy chain** (TRACE + 4 members) | 17,388 s = 4.83 h → **1.21 cm²/GPU-h** | 3,930 s = 1.09 h → **5.36 cm²/GPU-h** | 4.4× |

TRACE compute is the time `ink_v2.py` reports for the model pass, in both logs. The chain figure uses stage walls on both sides. The Mac's TRACE wall was 444.9 s, and the 4070's was 307 s including container start. The 4070 chain log records it as `TRACE exit=0 oom=false (307s)`, and it is one of the inputs to the 5.36 cm²/GPU-h. Fusion takes seconds on the CPU and isn't counted on either side.

The A-hn rate depends on what else the Mac is doing. It was 4.7–4.8 tiles/s while idle, for s1 and s2, and 3.2 tiles/s in normal desktop use, for the decoy and the first half of s0. A fully idle chain would give about 1.4 cm²/GPU-h.

At batch 1, A-hn uses 3.47 GB in the MPS driver plus 0.93 GB of process RSS for every member, and TRACE uses 2.31 GB plus 2.03 GB. Every run used batch 1. The 4070 peaks at 10.2 GB for the whole card.

## A faster A-hn path: the same network as 2-D convolutions on Core ML

Everything above runs the winner's network unchanged, and A-hn is where the Mac loses its time: 3.9–5.9× the 4070 per member. The idea for the fix comes from two places. Villa PR #2634 by campbelltron1 (open) runs villa's 3-D networks through 2-D kernels on Apple-silicon GPUs, 4–5× faster on an M4 Pro. HerculeX by Adam Betka folds its 3-D depth stem into 2-D convolutions the same way (`stock_model.folded`). I applied that idea to the winner's A-hn. What is mine here: the rewrite of A-hn's whole ResNet3D-50 encoder with its BatchNorms folded in, its conversion to Core ML, running the M5 Pro's Neural Engine and GPU as two lanes, and the gate against the 4070 with the no-text control. The winner's weights, tiling, stitching and fusion stay as they are. On this window a member now takes about 18 minutes on the M5 Pro, against 59 to 89 minutes on the path above, with under 2 GB of GPU memory. Every number here has a source file in [`docs/apple-silicon/m5-lanes/`](docs/apple-silicon/m5-lanes/README.md).

**What changes.**
- `mac/ahn_fold.py` builds the same network from the loaded checkpoint, with every operation in 2-D:
  - every BatchNorm is folded into the convolution before it, in float64;
  - each 3-D convolution becomes 2-D convolutions over its depth taps, with depth on the batch axis;
  - the input's trilinear upsample is split into a fixed depth matrix and a bilinear 2-D upsample.
- In float64 on the CPU, for member s0 at a reduced size (the input upsampled to 48 × 128 × 128 instead of 96 × 256 × 256; the same layers and weights), the rewrite matches the stock network to 3.5e-14 in the logits. At full size in float32 the largest logit difference is 3.8e-6. Through the winner's own `infer.py` tiling and stitch on a 64 × 64 px B0 window, on the CPU, it matches the stock network at r 0.99999999999996.
- `mac/ahn_coreml.py` converts the rewrite to a float16 Core ML program.
- `mac/ahn_lanes.py` runs that program on the Neural Engine and the GPU at once.
  - Each tile goes to one of them by a fixed pattern over the run's tile order, so an uninterrupted rerun sends every tile to the same unit. A run resumed from its saved parts restarts the pattern.
  - The calls go through a small C bridge, `mac/ahn_engine.m`, outside Python's GIL.
- `run_ahn.py --backend lanes` hands it to the winner's `infer.render` in place of the model. The recipe is the same as above.

This path does change the arithmetic: the convolutions run in Core ML's float16 instead of under PyTorch's float16 autocast. So it has to pass a gate: r ≥ 0.999 against the 4070's map on every member and every fused map, with 0 NaN mismatches, and the no-text control has to behave as it does on the 4070.

An earlier, smaller change on the stock path, folding BatchNorm and batching the stitch on the GPU, gave about 1.15× on the one of seven parts that finished, and was kept out of the release as too little gain for its departure from the reference maps ([`speedup_check/`](docs/apple-silicon/speedup_check/README.md)). This rewrite departs further, so it is reported as its own gated path, with its agreement shown in full below, and the path above stays the reference.

**Backends compared** (A-hn s0, 64 tiles from the B0 claim box, batch 1; `bench.json`):

| backend | tiles/s |
|---|---|
| the winner's network on MPS, float16 autocast (the path above) | 4.65 |
| the 2-D rewrite on MPS, float16 | 9.95 |
| the 2-D rewrite on Core ML, Neural Engine | 7.37 |
| the 2-D rewrite on Core ML, GPU | 11.3 |
| **Core ML on the Neural Engine and the GPU at once, in the full B0 runs below** | **15.4–15.6** |
| RTX 4070, CUDA float16 (16,896 tiles in 905–907 s) | 18.6–18.7 |

**The B0 native window**, with the same window, ROI, grid origin and `clip_to_roi` as above, 16,896 tiles per member. The 4070 column is the winner's unmodified code (CUDA, float16 autocast); the Mac columns are the path above and the rewritten network:

| member | 4070 | Mac, the path above | Mac, two lanes | r against the 4070 | max abs diff | same 8-bit level | NaN mismatches |
|---|---|---|---|---|---|---|---|
| s0 | 907 s | 4,396 s | **1,094 s** | 0.9999969 | 6.4e-3 | 91.9 % | 0 |
| s1 | 905 s | 3,551 s | **1,082 s** | 0.9999959 | 7.3e-3 | 91.6 % | 0 |
| s2 | 906 s | 3,631 s | **1,082 s** | 0.9999967 | 7.2e-3 | 91.2 % | 0 |
| decoy | 905 s | 5,366 s | **1,084 s** | 0.9999943 | 4.0e-3 | 90.4 % | 0 |

- On the rewritten path 8–10 % of pixels land on a different 8-bit level from the 4070's map. On the path above it is 1–2 % (98.2–99.0 % the same, first table).
- The four members take 4,342 s on the M5 Pro against 3,623 s on the 4070's unmodified path, 1.20× as long; with the path above it was 4.7×.
- With TRACE as above (444.9 s), the product and decoy chain runs at **4.40 cm²/GPU-h**, against 1.21 for the path above and 5.36 on the 4070.
- Each member is 3.3–5.0× faster than its run on the path above. Those runs' speed depended on what else the Mac was doing, as noted above. For s1 and s2, whose runs on the path above were on an idle Mac, the gain is 3.3–3.4×.
- Peak memory: 1.97 GB in the MPS driver and 1.44 GB of process RSS.

Fused with the winner's own `fuse_v7.py` and compared with the 4070's fused maps on the window (`fused_compare.json`):

| fused map | r | max abs diff | top-1 % Jaccard |
|---|---|---|---|
| A (mean of s0–s2) | 0.9999985 | 3.9e-3 | 0.988 |
| product (TRACE × A) | **0.9999993** | 2.0e-3 | 0.999 |
| decoy control (TRACE × decoy) | 0.9999985 | 1.7e-3 | 0.998 |

These are further from the 4070 than the path above (product r 0.99999977), as Core ML's float16 arithmetic implies, and well inside the gate.

**The no-text control** is the B0 no-text render, 700 × 4,700 px, with no ink expected. Its recipe is the 4070's: the whole render, grid origin 0 0, no ROI clip. Result, from `notext_check.log`:
- the four members: r 0.9999877–0.9999929 against the 4070;
- the fused maps: r 0.9999919–0.9999976, with 0 NaN mismatches;
- no product pixel reaches the native product's top-1 % threshold, on the Mac or on the 4070;
- product p99 0.1477 on both, and decoy top-1 % overlap 0.442 against the 4070's 0.444.

**Run it.** After the steps under "How to run it" below, install `coremltools==9.0` (it is in `mac/requirements.txt`) and have the Xcode command-line tools for `clang`. Then:

```bash
for m in s0 s1 s2 decoy; do
  python mac/ahn_coreml.py --ckpt $MD/ckpt/hn4_${m}_final/model.safetensors --out $MD/coreml/hn4_${m}_stack --mode stack
  caffeinate -i python mac/run_ahn.py --ckpt $MD/ckpt/hn4_${m}_final/model.safetensors --zarr $R \
    --out $O/${m}_lanes.npy --roi 1516 2915 2038 7638 --grid-origin 115 694 --batch 32 \
    --backend lanes --mlpackage $MD/coreml/hn4_${m}_stack.mlpackage --lanes GGNGN
done
```

- The Core ML packages hold weights derived from the winner's CC BY-NC checkpoints, so build them locally and don't publish them.
- `mac/ahn_bench.py` reproduces the backend table.
- I measured all of this on one M5 Pro. The two-lane split pays off because this GPU runs the network faster than its Neural Engine; on other Apple chips the balance may differ, and I haven't tested them.

## HerculeX on the M5 Pro

[HerculeX](https://github.com/adambetka2/HerculeX) by Adam Betka (stokarz, MIT) unwraps whole scrolls and reads their ink on Apple silicon, with Domenico Russo's Reader v3 as its ink reader. On the same M5 Pro I sped up its reader and changed its pieces stage and its sampling, in four code commits on HerculeX 9f59e96, in [our fork](https://github.com/comoke/HerculeX/tree/m5) (branch `m5`): two for the reader (5b1576b, 1e72245), one for the pieces stage (922a5a1) and one for sampling (856cb05). The reader now runs a share of tiles' network bodies on the GPU beside the Neural Engine, and its depth stem and, from 1e72245, its normalisation statistics on Metal 4 tensor ops. With stock's own 4 samplers, the initial reader commit 5b1576b read a 404-piece unit in 59.60 s against stock's 70.04 s on the same Mac (1.18×), one run each; the current reader (1e72245) wasn't timed at 4 samplers, and runs with 8 samplers, a setting stock already has, aren't compared, because stock wasn't timed with 8. The current reader (1e72245) doesn't give bit-identical ink maps: in one check on the same captured inputs they match stock's at pooled r 0.99997–0.99998, worst single map r 0.99897, max abs diff 0.010–0.013; 5b1576b's maps have no saved comparison. In the pieces stage the draws' memory budget is now a setting (`HERCULEX_DRAW_BYTES`, default unchanged); raised, it gave one unit's 30,188 piece files byte-identical to stock's, and no speedup is claimed because that unit's stock time wasn't saved. Sampling now announces the brick planes a piece will read before it is sampled (`HERCULEX_READ_AHEAD`, on by default in the fork); 48 of 48 sampled captures were byte-identical to stock's, and no gain is claimed for it. `HERCULEX_BODY_LANES=E`, `HERCULEX_STEM=kernel` and `HERCULEX_READ_AHEAD=0` restore stock's choices. Sources: [`docs/apple-silicon/herculex-m5/`](docs/apple-silicon/herculex-m5/README.md).

## Related work

Two public runners already run the winner's A-hn models. [inkrs](https://github.com/praveenperera/inkrs) by Praveen Perera (praveenperera), released on November 13, reimplements ink models natively, among them v8-in and the winner's three A-hn fine-tunes. On an RTX 5090 it runs the whole 0306B recipe, TRACE v3B with its test-time augmentation, the three A-hn seeds and the winner's `fuse_v7.py`, about 1.7× faster than the winner's own scripts, with each map gated against theirs. On Apple silicon its Metal backend runs villa's readers and Hecate, with an optional Neural Engine offload for the readers. Villa PR #2525 (open) adds inkrs as an opt-in engine for villa's own flat ink inference, about 11× faster than PyTorch on MPS on an M4 Pro. [ink-sdk](https://github.com/LimeGS/ink-sdk) by LimeGS and pscamillo has run the A-hn seeds, TRACE v3B and the 0306B product rule since November 10, with A-hn going through v8-in's own `predict.py`. Since November 22, a change by campbelltron1 ([ink-sdk#26](https://github.com/LimeGS/ink-sdk/pull/26)) runs that lane on a Mac GPU through MPS, with v8-in's 3-D convolutions as 2-D kernels: 2.0× faster than PyTorch's own 3-D convolutions on an A-hn seed, on an M4 Pro. Neither trains. This appendix runs something different from both: the winner's own `infer.py` and `ink_v2.py`, unchanged and matched to the 4070 to float16 precision, and a Core ML rewrite of A-hn on the Neural Engine and GPU together.

Villa PR #2634 by campbelltron1 (open) runs villa's 3-D networks through 2-D kernels on Apple-silicon GPUs, and HerculeX by Adam Betka folds its 3-D depth stem the same way; the faster A-hn path above applies that idea to the winner's network. Villa PR #2595 by vincentl065 (open) does the same for the 3-D stem of villa's ink inference on AMD ROCm.

Villa PR #2598 by ChaseHendrick, merged on November 22, makes spiral-fitting exports on MPS match the CPU by building the winding angles in float64. It is adjacent Apple Silicon exactness work in the Vesuvius stack, on surface geometry rather than ink inference. #2646 by kamenlevi (open) gives the same spiral fitting Metal ports of its fused CUDA kernels, so a fit on MPS uses CUDA's interpolant and runs several times faster per step. Two more open villa PRs are separate macOS and CPU fixes. #2608 by nerln stops villa's own flat ink inference waiting 5 s per DataLoader worker on macOS, and #2607 by vcTaz chooses FP32 or BF16 for CPU inference by what the CPU supports. #2658 by campbelltron1 makes the same fix as #2608 for `vesuvius.predict` and villa's trainers.

None of #2525, #2598, #2607, #2608, #2646 and #2658 changes code this path runs. The A-hn `infer.py` uses no DataLoader. On TRACE's path, `ink_v2.py` calls `zoomlib.py`, which calls `fastink.py`. `fastink.py` imports villa's `ink_detection/inference/infer.py`, the file #2608 and #2525 edit, but calls only its model-loading and tiling helpers, not the DataLoader loop that #2608 changes, and it uses the copy pinned inside the winner's image. Both models run float16 autocast on MPS, not on the CPU.

## How to run it

You need an Apple Silicon Mac. I tested only on an M5 Pro with 24 GB and macOS 27.0.1. Peak use stays under 5 GB, so 16 GB should work, but that is untested. You also need Python 3.13 with the packages in `mac/requirements.txt`, which include PyTorch 2.15 with MPS, plus `zstd` to decompress the archive, and `uv`.

The code comes out of the winner's Docker image `fl-claim-a7:final`, whose archive is `docker/fl-claim-a7_final.tar.zst`, 6.6 GB, in the dataset repo `Prodigle/pherc0306b-first-letters`. Docker isn't needed: `mac/fetch_winner.sh unpack` takes the code and the v8-in and TRACE weights out of the archive with `zstd` and Python, and needs about 14 GB free while it runs (the 12.0 GB unpacked archive and 2 GB of output). On the M5 Pro, [`unpack_test_m5.log`](docs/apple-silicon/unpack_test_m5.log) compared the unpacked files with an earlier copy, and the setup below was then run once from a clean folder with the exported repository, ending with a 16-tile A-hn run from the unpacked code ([`recipe_test_m5.log`](docs/apple-silicon/recipe_test_m5.log)). It needs a `python3` on the PATH; it was tested with macOS's `/usr/bin/python3`. (`fetch_winner.sh code` does the same through Docker, if you have it.) The weights and data come from the Hugging Face repos `Prodigle/pherc0306b-ink-ahn` and `Prodigle/pherc0306b-first-letters`.

From a checkout of this repo:

```bash
export MD=~/Workspace/vesuvius-data-mac
# the winner's image archive, 6.6 GB, from their Hugging Face dataset; then its sha256
curl -L -o fl-claim-a7_final.tar.zst \
  https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters/resolve/main/docker/fl-claim-a7_final.tar.zst
echo "1832832330ab67b134795eaedd43d6d5c3dcf004596e1fd287d2e71b3af73536  fl-claim-a7_final.tar.zst" | shasum -a 256 -c
bash mac/fetch_winner.sh unpack fl-claim-a7_final.tar.zst   # the code and v8-in/TRACE weights, out of the image archive, no Docker
uv venv --python 3.13 $MD/.venv && uv pip install --python $MD/.venv/bin/python -r mac/requirements.txt
PY=$MD/.venv/bin/python bash mac/fetch_winner.sh hf    # A-hn members + claim-surface render (level 0)
bash mac/fetch_winner.sh link                          # links + sha256 check of the v8-in base weights
source mac/paths.sh
R=$MD/render/claim_surface_render_41layers.zarr; O=$MD/runs/native
# TRACE v3B on the B0 native window
python mac/run_trace.py $R $O --models tr --crop 1516,2916,2038,7639
# the four A-hn members on the same window; each resumes from <out>.parts/ if interrupted
for m in s0 s1 s2 decoy; do
  caffeinate -i python mac/run_ahn.py --ckpt $MD/ckpt/hn4_${m}_final/model.safetensors --zarr $R \
    --out $O/${m}.npy --roi 1516 2915 2038 7638 --grid-origin 115 694 --batch 1
done
# fusion with the winner's fuse_v7.py (+ comparison, when the 4070 reference maps are present)
python mac/fuse_compare.py $O/fwd_tr.npy $O/s0.npy $O/s1.npy $O/s2.npy $O/decoy.npy $O/fused
```

`caffeinate -i` stops idle sleep, but closing the lid still sleeps the Mac, and a run interrupted that way resumes from its finished parts. `mac/compare.py` and `mac/fuse_compare.py` produce the exactness tables. They need the 4070's maps from the main recipe as the reference, and this bundle doesn't include them.

## Licensing

The terms are the same as the main entry's. My code in `mac/` is under the same terms as `pc/`. The winner's code is MIT and used unchanged. The winner's weights, data and maps are CC BY-NC 4.0, and this bundle does not redistribute them. `docs/apple-silicon/` holds only logs and comparison numbers, no maps, and `fetch_winner.sh` downloads the rest from the winner's own image and Hugging Face repos.
