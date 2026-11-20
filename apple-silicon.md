# Appendix: the same pipeline on Apple Silicon (MPS)

This is a supporting note to the main entry, which is about the RTX 4070.

The winner's inference code also runs on an Apple M5 Pro through PyTorch's MPS backend. That code is `ink_v2.py` for TRACE and `infer.py` for A-hn. I ran all five maps and the fusion on the B0 native window, which is 1,399 × 5,600 px of the claim render around the claim box, about 5.85 cm² of the claim. The maps are TRACE v3B and the four A-hn members, s0, s1, s2 and the decoy. Every map matches the 4070's CUDA map to float16 precision. On that window the Mac is 1.7× slower on TRACE and 4–6× slower per A-hn member. Every number below has a source file in [`docs/apple-silicon/`](docs/apple-silicon/README.md).

## What I change

I add a device shim, `mac/mps_shim.py`, and a small harness, `mac/run_ahn.py` and `mac/run_trace.py`. There is no model, weight or method change.

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

## Related work

Villa PR #2598 by ChaseHendrick, which is open, makes spiral-fitting exports on MPS match the CPU by building the winding angles in float64. It is adjacent Apple Silicon exactness work in the Vesuvius stack, on surface geometry rather than ink inference. Two more open villa PRs are separate macOS and CPU fixes. #2608 by nerln stops villa's own flat ink inference waiting 5 s per DataLoader worker on macOS, and #2607 by vcTaz chooses FP32 or BF16 for CPU inference by what the CPU supports.

Neither changes code this path runs. The A-hn `infer.py` uses no DataLoader. On TRACE's path, `ink_v2.py` calls `zoomlib.py`, which calls `fastink.py`. `fastink.py` imports villa's `ink_detection/inference/infer.py`, the file #2608 edits, but calls only its model-loading and tiling helpers, not the DataLoader loop that #2608 changes, and it uses the copy pinned inside the winner's image. Both models run float16 autocast on MPS, not on the CPU.

## How to run it

You need an Apple Silicon Mac. I tested only on an M5 Pro with 24 GB and macOS 27.0.1. Peak use stays under 5 GB, so 16 GB should work, but that is untested. You also need Python 3.13 with the packages in `mac/requirements.txt`, which include PyTorch 2.15 with MPS, plus `zstd` to decompress the archive, and `uv`.

The code comes out of the winner's Docker image `fl-claim-a7:final`, so you need Docker on some machine, with no GPU needed. The image archive is `docker/fl-claim-a7_final.tar.zst`, 6.6 GB, in the dataset repo `Prodigle/pherc0306b-first-letters`. The `code` step takes its path, or `-` if the image is already loaded. The weights and data come from the Hugging Face repos `Prodigle/pherc0306b-ink-ahn` and `Prodigle/pherc0306b-first-letters`.

From a checkout of this repo:

```bash
export MD=~/Workspace/vesuvius-data-mac
bash mac/fetch_winner.sh code fl-claim-a7_final.tar.zst   # copies the code and v8-in/TRACE weights out of the image
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

If the Mac has no Docker, run the `code` step on any machine that has it and copy `$MD/code` and `$MD/fl` across. `caffeinate -i` stops idle sleep, but closing the lid still sleeps the Mac, and a run interrupted that way resumes from its finished parts. `mac/compare.py` and `mac/fuse_compare.py` produce the exactness tables. They need the 4070's maps from the main recipe as the reference, and this bundle doesn't include them.

## Licensing

The terms are the same as the main entry's. My code in `mac/` is under the same terms as `pc/`. The winner's code is MIT and used unchanged. The winner's weights, data and maps are CC BY-NC 4.0, and this bundle does not redistribute them. `docs/apple-silicon/` holds only logs and comparison numbers, no maps, and `fetch_winner.sh` downloads the rest from the winner's own image and Hugging Face repos.
