# PHerc1447 anchor packs rebuilt from the raw public stacks, 2026-11-16

The end-to-end training used the winner's shipped anchor packs `a_w058.npz` and `a_w060.npz` (sha256-checked, `pc/train_e2e/e2e_chain.sh` stage `anchors_shipped`). This rebuild shows they are exactly what the winner's own code makes from the public raw data.

**Code (the winner's, unchanged):**
- `submission/code/training/x_push3/m306/self/code/make_packs.py`: `anchors()` (lines 47–49) calls `save_anchor(w, L.src_1447(w), '1447')`; `save_anchor` (lines 40–44).
- `submission/code/training/x_push3/m306/train/v8306lib.py`: `src_1447` (lines 59–106) with its defaults (tile 64, depth 24, clip 200, stride 48, soft labels).
- Our driver: `pc/train_e2e/anchors_rebuild.sh` with `anchors_rebuild.py`. That file adds two import stubs that leave the anchor code alone: the model class `v8306lib` imports at load time, and `train_self`, needed only for its output root. Comparison: `pc/train_e2e/anchors_compare.py`.
- Run inside `fl-claim-a7:final` with `--network none`, CPU only: 298 s.

**Raw inputs:** Hugging Face dataset `YoussefMoNader/ink-8um-pherc1447-surfaces`, revision `7e4d918712a8d8642a6fdeee45b80217c1e98317` (the revision named in the winner's `docs/MODELS_AND_DATA.md` line 49). For each of w058 and w060: `labels/inklabels.png`, `labels/mask.png`, `layers/00..23.tif`. That is 52 files, 1,086,990,423 bytes, each sha256-checked against the dataset's `SHA256SUMS`.

**Result:**

| pack | sha256 (rebuilt = shipped = MANIFEST) | bytes | tiles |
|---|---|---|---|
| `a_w058.npz` | `80084f2a001bc8a3222f1c2f16d14a6fff0387e081d637f6928c66ffb2c939cd` | 183,443,822 | 1512 |
| `a_w060.npz` | `9e092b87ddaed4f54d7ba2a45ca094cc084d15d9b1e7d5390064d2b0783cb666` | 283,303,502 | 1206 |

Every array (`img`, `lab`, `keep`, `org`, `group`, `info`) has 0 differing elements, and every zip member's bytes match (`docs/train-e2e-anchors-compare-a_w058-2026-11-16.json`, `docs/train-e2e-anchors-compare-a_w060-2026-11-16.json`).
